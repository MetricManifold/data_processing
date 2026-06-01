#pragma once
#include "types.cuh"
#include <cstddef>

// ---------------------------------------------------------------------------
// sim_v3 launch wrappers.
// ---------------------------------------------------------------------------
// All kernels operate on a fixed power-of-two tile (TILE_T) and a unified
// phi pool of N*TILE_AREA floats. There is no neighbour list, no spatial
// hash, no halo, and no per-cell variable W/H. Cell-cell interaction is
// mediated by a global sum field S(x,y) = sum_n phi_n^2(x,y), built each
// step by atomic scatter and read back during evolve.
// ---------------------------------------------------------------------------

// Multi-block evolve chunking. Each block handles CHUNK_PIXELS pixels of
// one cell's active rect; with TILE_AREA pixels per cell there are
// REDUCE_CHUNKS_PER_CELL blocks per cell. Hoisted here so alloc_gpu can
// size the deterministic-reduce partial buffer to match the launch.
static constexpr int CHUNK_PIXELS           = 4096;
static constexpr int REDUCE_CHUNKS_PER_CELL =
    (TILE_AREA + CHUNK_PIXELS - 1) / CHUNK_PIXELS;
// 8 moments accumulated by k_reduce_mb_full: V, Ix, Iy, perim, Cx, Cy, Cxx, Cyy.
static constexpr int REDUCE_NMOMENTS        = 8;

// Polarity update (RTP or ABP, per p.abp). Cheap: one thread per cell.
void launch_polar(CellArrays& c, const SimParams& p, double cur_time,
                  cudaStream_t stream = 0);

// Apply a list of scripted tumble events: for each i in [0, count),
// theta[d_cid[i]] = d_theta[i]; px = cos(theta); py = sin(theta).
// Used in deterministic-replay mode (--scripted-events).
void launch_apply_scripted(CellArrays& c,
                           const int* d_cid,
                           const float* d_theta,
                           int count,
                           cudaStream_t stream = 0);

// Zero S then scatter phi^2 into it (one CTA per cell).
void launch_scatter_S(CellArrays& c, const SimParams& p, cudaStream_t stream = 0);

// Pack origin + per-cell observables into a contiguous device buffer for
// single-memcpy trajectory I/O. `out` must point to >= N TrajPackedCell.
void launch_pack_traj(CellArrays& c, TrajPackedCell* out, int N, cudaStream_t stream = 0);

// Fused two-pass evolve. Pass 1 reduces V/Cx/Cy/Ix/Iy, broadcasts vx/vy.
// Pass 2 reads S again, computes laplacian/double-well/repulsion/advection,
// and writes phi_out plus reduces perimeter. Also writes velocities,
// volumes, Cx, Cy, perimeters into the per-cell observable arrays.
//
// `need_full_reduce` controls whether Cx/Cy/Cxx/Cyy and perimeter are also
// computed. Set true on rebind steps (rebind reads Cx/Cy/Cxx/Cyy) and on
// trajectory/VTK/checkpoint output steps (host reads V, Cx, Cy, perimeter).
// On non-output, non-rebind steps it can be false; the mb path then skips
// 5 atomicAdds per chunk and 4 block-reductions.
void launch_evolve(CellArrays& c, const SimParams& p, bool need_full_reduce,
                   cudaStream_t stream = 0);

// COM-recentre: shift each cell's tile so its COM lands at (T/2, T/2).
// Adjusts origin[n] and copies the (possibly shifted) tile into phi_out.
// After this kernel, the *caller* must std::swap(phi_in, phi_out) so the
// rebound tile becomes the current state.
void launch_rebind(CellArrays& c, float bbox_k, float gamma_ref,
                   cudaStream_t stream = 0);

// One-shot host helpers used only at init / resume.
void launch_rng_init(CellArrays& c, unsigned long seed,
                     const int* d_global_ids = nullptr);

// Draw initial per-cell next_tumble_time = cur_time + Exponential(1/tau).
// Called once at fresh init after launch_rng_init has seeded the per-cell
// curand streams. NOT called on resume (next_tumble_time restored from
// checkpoint sidecar instead).
void launch_init_tumble_schedule(CellArrays& c, const SimParams& p,
                                 double cur_time);

// Initialise phi tiles as tanh(2(r - R_eff)/(sqrt(2)*lambda)) profiles.
// h_cx/h_cy are global-coord cell COMs (passed via temporary device arrays
// allocated inside the launcher). Used only at fresh init.
void launch_init_phi(CellArrays& c, const SimParams& p,
                     const float* d_cx, const float* d_cy);

// Compute initial velocities from current phi + per-cell v_A + polarity,
// without advancing phi. Used after both fresh init and resume so that the
// first trajectory write has a meaningful velocity.
void launch_initial_velocity(CellArrays& c, const SimParams& p);

// Two-step variant used by multi-GPU init: scatter, then a halo exchange,
// then the velocity reduce. Single-GPU should keep using launch_initial_velocity.
void launch_initial_scatter(CellArrays& c, const SimParams& p);
void launch_initial_velocity_reduce(CellArrays& c, const SimParams& p);

// Halo support: fused two-pair in-place add. dst0[i] += src0[i] and
// dst1[i] += src1[i] in one kernel launch (both pairs share length).
// Used by the multi-GPU halo exchange to fold neighbour-rank contributions
// into the local S band after ncclSend/ncclRecv has placed the neighbour's
// data in a staging buffer. Saves ~10-15 us per step vs. two separate
// launches.
void launch_halo_add_pair(float* dst0, const float* src0,
                          float* dst1, const float* src1,
                          std::size_t n_floats, cudaStream_t stream);

// ===========================================================================
// Cell migration (multi-GPU only). When a cell's rebound COM crosses
// outside its owning rank's slab, the cell must be moved to the
// neighbour rank that now owns its COM. This happens at rebind cadence
// (every REBIND_EVERY=8 steps).
// ===========================================================================

// Default maximum cells migrating in any one direction per rebind round.
// Used as a floor; the runtime value is sized to max(MAX_MIGRANTS_DEFAULT,
// capacity / 4) at alloc time so it scales with per-rank cell count.
// Memory: 4 * runtime_max * pack_size_per_cell bytes per rank.
// At pack_size ~ 410 KB and runtime_max=128: ~205 MB.
// At capacity=6400 (N=12800 G=4), runtime_max=1600: ~2.6 GB per rank.
// Failure to fit at runtime is a fatal error (host-side check after classify).
static constexpr int MAX_MIGRANTS_DEFAULT = 128;

// Per-cell pack size in bytes. Includes the full TILE_AREA phi tile + a
// small header (origin/rect/global_id/scalars/rng_state). Defined in
// kernels.cu so the curandState size dependency stays out of this header.
extern const std::size_t CELL_PACK_BYTES;

// Classify all local cells into stay/up/down based on rebound COM y vs
// slab boundaries. After this kernel:
//   *d_n_stay = number of cells staying on this rank
//   *d_n_up   = number going to prev_rank (= (rank-1+G)%G)
//   *d_n_down = number going to next_rank (= (rank+1)%G)
//   stay_idx[0..n_stay)  = local indices of stayers
//   up_idx[0..n_up)      = local indices of cells leaving up
//   down_idx[0..n_down)  = local indices of cells leaving down
//
// Counters MUST be zeroed by the caller before launch (cudaMemsetAsync).
//
// For G=2 prev_rank == next_rank: a cell that needs to leave is always
// classified as "up" (down list stays empty). The orchestrator's symmetric
// send/recv pattern still works because n_down=0 turns the down-direction
// NCCL calls into 0-byte no-ops.
void launch_classify_migrants(
    const int* origin, int N,
    int slab_y_lo, int slab_y_hi, int Ny,
    int rank, int gpus,
    int* d_n_stay, int* d_n_up, int* d_n_down,
    int* stay_idx, int* up_idx, int* down_idx,
    cudaStream_t stream);

// Pack `count` migrants into a contiguous byte buffer (count * CELL_PACK_BYTES).
// Source per-cell data is gathered from CellArrays at the indices in
// migrant_idx. The phi tile is copied from phi_in (current state half).
void launch_pack_migrants(
    const CellArrays& c,
    const int* migrant_idx, int count,
    const int* d_global_id_src,
    void* pack_buf,
    cudaStream_t stream);

// Unpack `count` arrivals into per-cell arrays starting at slot dst_offset.
// Phi tiles are written into phi_out (the scratch half — caller is
// responsible for swapping after compact + unpack are both done). The
// caller must also ensure dst_offset + count <= capacity.
void launch_unpack_migrants(
    CellArrays& c,
    const void* pack_buf, int count,
    int dst_offset,
    int* h_global_id_dst,   // device-resident int array used by host post-step
    cudaStream_t stream);

// Compact the stays into the front of phi_out + scalar scratch arrays.
// Reads from phi_in / current cell arrays, writes to phi_out / scratch.
// stay_idx contains the local indices of cells to keep.
void launch_compact_stays(
    const CellArrays& c,
    const int* stay_idx, int n_stay,
    float* phi_dst,                  // = phi_out (scratch half)
    int*   origin_dst, int* rect_dst,
    float* gamma_dst, float* v_A_dst, float* tgt_R_dst,
    float* polar_theta_dst, float* polar_x_dst, float* polar_y_dst,
    void*  rng_dst,                  // curandState array
    cudaStream_t stream);

