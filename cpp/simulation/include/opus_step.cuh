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

// Build the work list from cells.rect on the host. D2H the rect, walk it,
// upload. Returns the new work count (also written to cells.workCount).
// Call after every rebind and at fresh init.
int build_opus_work_list_host(CellArrays& c);

// Mirror the parity-`from` halves of S/V/Ix/Iy into the parity-`from^1`
// halves via cudaMemcpyDeviceToDevice. Called at fresh init (from=0,
// after launch_initial_velocity has filled the parity-0 halves) and
// after every rebind (from=current parity, after re-scatter+reduce has
// filled the parity-current halves) so the next opus step has a
// consistent lagged-moment set regardless of which parity is current.
void launch_opus_seed_parity_mirror(CellArrays& c, const SimParams& p,
                                    int from_parity,
                                    cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Fused rebind path. Replaces the separate launch_rebind kernel + the
// scatter_S+reduce+mirror reseed. Sequence per rebind:
//
//   1. launch_opus_compute_rebind_meta(c, p, parity, stream)
//      Per cell, reads V/Cx/Cy/Cxx/Cyy from the parity-current half (which
//      were freshly produced by the previous DO_EXT step), computes the
//      integer shift (sx, sy) so COM lands at (T/2, T/2), and the new rect
//      from second-moment width. Writes c.shift_xy and c.new_rect.
//
//   2. build_opus_work_list_for_rebind(c)  (host helper; syncs the stream)
//      D2H c.rect, c.shift_xy, c.new_rect. Builds the source-frame union
//      bounding box per cell (= bbox of old_rect U new_rect_shifted_to_src)
//      and enumerates 32x32 sub-tiles covering it. Uploads to c.d_work.
//      Returns new workCount.
//
//   3. launch_opus_step_rebind(c, p, parity, stream)
//      Same as launch_opus_step but with DO_REBIND=true. Reads phi[parity],
//      writes phi[parity^1] at SHIFTED destination addresses (lx-sx, ly-sy).
//      Scatters phi_new^2 into S[parity^1] at the SAME global address as
//      the unshifted scatter would (because rebind preserves global pixel
//      positions), so S[parity^1] is exactly correct post-rebind.
//      Writes 0 at destinations outside new_rect (periphery cleanup).
//      Accumulates fresh moments of phi_in into V/Ix/Iy_pool[parity^1].
//
//   4. flip parity (caller).
//
//   5. launch_opus_apply_rebind_meta(c, stream)
//      Per cell, applies origin += shift_xy and rect = new_rect.
//
//   6. build_opus_work_list_host(c)  (rebuild regular worklist from new rect)
// ---------------------------------------------------------------------------

void launch_opus_compute_rebind_meta(CellArrays& c, const SimParams& p,
                                     int parity, cudaStream_t stream = 0);

int  build_opus_work_list_for_rebind(CellArrays& c);

void launch_opus_step_rebind(CellArrays& c, const SimParams& p,
                             int parity, cudaStream_t stream = 0);

void launch_opus_apply_rebind_meta(CellArrays& c, cudaStream_t stream = 0);
