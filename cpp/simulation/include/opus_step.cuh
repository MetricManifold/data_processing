#pragma once
// ---------------------------------------------------------------------------
// opus_step — single-pass fused step kernel.
//
// Replaces the legacy scatter_S + evolve pipeline with one kernel launch
// that, per 32x32 sub-tile:
//   1. loads a 34x34 phi halo into shared memory,
//   2. accumulates fresh V/Ix/Iy (and on full-reduce steps perim/Cx/Cy/Cxx/Cyy)
//      of phi_in using the matching current-step S,
//   3. evolves phi_in -> phi_out with Palmieri RHS using LAGGED V/Ix/Iy
//      coefficients (one-step lag; ~0.1% per dt; spec-permitted),
//   4. atomicAdds phi_out^2 into the next-step S buffer.
//
// Buffers are parity-paired: at step s with parity p the kernel reads
// phi[p], S[p], V/Ix/Iy[p] (lagged), writes phi[p^1], S[p^1], V/Ix/Iy[p^1]
// (fresh). Caller flips parity after the launch.
//
// Gated by -DCELL_SIM_LEGACY_STEP=1 to fall back to the legacy path;
// default off (opus is the production step).
// ---------------------------------------------------------------------------

#include "types.cuh"
#include <cstddef>
#include <vector>

// 32 wide x 8 high threads (256), each thread owns RY=4 stacked rows ->
// 32x32 output sub-tile per block, 34x34 shared halo.
namespace opus {
static constexpr int BX = 32;
static constexpr int BY = 8;
static constexpr int RY = 4;
static constexpr int OW = BX;          // output width  = 32
static constexpr int OH = BY * RY;     // output height = 32
}  // namespace opus

// Flat work-list entry: one 32x32 sub-tile to evaluate. `tile` is the cell
// index, (sx, sy) is the tile-local origin of the sub-tile. List is built
// host-side from the per-cell rect and uploaded; rebuilt after every
// rebind (and after migration).
struct WorkItem { int tile; int sx; int sy; };

// Worst-case work items per cell: ceil((TILE_T-2)/OW) * ceil((TILE_T-2)/OH).
// At TILE_T=320 with OW=OH=32: ceil(318/32)=10 -> 100 items per cell.
static constexpr int OPUS_MAX_WORKITEMS_PER_CELL =
    ((TILE_T - 2 + opus::OW - 1) / opus::OW) *
    ((TILE_T - 2 + opus::OH - 1) / opus::OH);

// ---------------------------------------------------------------------------
// One-step launch. parity = current half (read from [parity], write to [^1]).
// need_full -> also accumulate perim, Cx, Cy, Cxx, Cyy of phi_in.
// Caller must flip parity after this returns.
// ---------------------------------------------------------------------------
void launch_opus_step(CellArrays& c, const SimParams& p,
                      int parity, bool need_full,
                      cudaStream_t stream = 0);

// Refresh per-cell vx_out/vy_out from the freshly-reduced V/Ix/Iy held in
// V_pool/Ix_pool/Iy_pool[parity]. Called on every step so observers
// (trajectory, vtk, viewer) see same-step velocities. Cheap — one thread
// per cell. (Inside the fused step we compute the LAGGED velocity for
// advection but don't write it back; this kernel writes the FRESH one
// using the just-reduced moments.)
void launch_opus_finalize_velocity(CellArrays& c, const SimParams& p,
                                   int parity, cudaStream_t stream = 0);

// Mirror the parity-`from` halves of S/V/Ix/Iy into the parity-`from^1`
// halves via cudaMemcpyDeviceToDevice. Called at fresh init (from=0,
// after launch_initial_velocity has filled the parity-0 halves) so the
// first step has a consistent lagged-moment set regardless of which
// parity is current.
void launch_opus_seed_parity_mirror(CellArrays& c, const SimParams& p,
                                    int from_parity,
                                    cudaStream_t stream = 0);

// Device-side worklist build. Resets the in-band atomic counter, runs a
// 1-thread-per-cell kernel that emits one WorkItem per (cell, sub-tile-
// in-rect), then reads the count back via a 4-byte pinned-memory D2H +
// stream sync (so the caller knows the launch grid for subsequent step
// kernels). Sets c.workCount.
//
// Reused for BOTH regular and rebind steps: new_rect is clamped to a
// subset of old_rect by compute_rebind_meta, so the source-frame worklist
// over old_rect fully covers every destination pixel of the new rect.
void launch_opus_build_worklist(CellArrays& c, cudaStream_t stream = 0);

// Host-built worklist alternative. Syncs the stream, reads c.rect D2H,
// emits one WorkItem per (cell, sub-tile-in-rect), H2D copies the array,
// sets c.workCount = actual count, and mirrors the count into
// c.d_work_count (so the in-kernel early-exit on k_opus_step is a no-op
// when the launch grid equals workCount). Used in the cell_sim integration
// because rect counts vary widely per step and launching the worst-case
// grid leaves most blocks idle (~30% perf regression at N=1152).
int build_opus_work_list_host(CellArrays& c);

// ---------------------------------------------------------------------------
// Fused rebind path. Replaces the separate launch_rebind kernel and the
// scatter_S+reduce+mirror reseed. Sequence per rebind cycle:
//
//   1. launch_opus_compute_rebind_meta(c, p, parity, stream)
//      Per cell, reads V/Cx/Cy/Cxx/Cyy from the parity-current half (which
//      were freshly produced by the previous DO_EXT step), computes the
//      integer shift (sx, sy) so COM lands at (T/2, T/2), and the new rect
//      from second-moment width. Clamps new_rect to a SUBSET of old_rect
//      so the source-frame worklist remains valid. Writes c.shift_xy and
//      c.new_rect.
//
//   2. launch_opus_step_rebind(c, p, parity, stream)
//      k_opus_step<DO_EXT=false, DO_REBIND=true>. Work-item (sx, sy) is
//      destination-frame; source = dest + shift. Reads phi[parity] at
//      source, writes phi[parity^1] at destination; periphery destinations
//      get 0. S[parity^1] scatter is at the SOURCE-frame global address
//      (which equals destination-frame global address with the post-rebind
//      origin), so S ends up exactly correct post-rebind.
//
//   3. flip parity (caller).
//
//   4. launch_opus_apply_rebind_meta(c, stream)
//      Per cell, applies origin += shift_xy and rect = new_rect.
//
//   5. launch_opus_step_cleanup(c, p, parity, stream)
//      Required: an additional DO_REBIND step with shift=0 and
//      src_rect == dst_rect == new_rect. Clears stale order-1 phi from the
//      OTHER ping-pong buffer's old\new periphery (those pixels were
//      evolved into the other buffer two parities ago and would otherwise
//      be read as stale halo on the step-after-next). Caller flips parity
//      after this returns.
//
//   6. launch_opus_build_worklist(c, stream)
//      Rebuild the (now-tightened) worklist for the upcoming regular steps.
// ---------------------------------------------------------------------------

void launch_opus_compute_rebind_meta(CellArrays& c, const SimParams& p,
                                     int parity, cudaStream_t stream = 0);

void launch_opus_step_rebind(CellArrays& c, const SimParams& p,
                             int parity, cudaStream_t stream = 0);

void launch_opus_apply_rebind_meta(CellArrays& c, cudaStream_t stream = 0);

void launch_opus_step_cleanup(CellArrays& c, const SimParams& p,
                              int parity, cudaStream_t stream = 0);
