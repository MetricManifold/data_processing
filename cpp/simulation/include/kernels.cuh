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

// Polarity update (RTP or ABP, per p.abp). Cheap: one thread per cell.
void launch_polar(CellArrays& c, const SimParams& p, cudaStream_t stream = 0);

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
void launch_rng_init(CellArrays& c, unsigned long seed);

// Initialise phi tiles as tanh(2(r - R_eff)/(sqrt(2)*lambda)) profiles.
// h_cx/h_cy are global-coord cell COMs (passed via temporary device arrays
// allocated inside the launcher). Used only at fresh init.
void launch_init_phi(CellArrays& c, const SimParams& p,
                     const float* d_cx, const float* d_cy);

// Compute initial velocities from current phi + per-cell v_A + polarity,
// without advancing phi. Used after both fresh init and resume so that the
// first trajectory write has a meaningful velocity.
void launch_initial_velocity(CellArrays& c, const SimParams& p);

// Halo support: in-place add. dst[i] += src[i] for i in [0, n_floats).
// Used by the multi-GPU halo exchange to fold neighbour-rank contributions
// into the local S band after ncclSend/ncclRecv has placed the neighbour's
// data in a staging buffer.
void launch_halo_add(float* dst, const float* src, std::size_t n_floats,
                     cudaStream_t stream);

// ===========================================================================
// Cell migration (multi-GPU only). When a cell's rebound COM crosses
// outside its owning rank's slab, the cell must be moved to the
// neighbour rank that now owns its COM. This happens at rebind cadence
// (every REBIND_EVERY=8 steps).
// ===========================================================================

// Maximum cells migrating in any one direction per rebind round. Sized
// generously vs typical drift (drift per rebind ≈ 0.04 px ≪ slab height,
// so almost no cells cross at any one rebind), but bounded so the pack
// buffers stay reasonable in VRAM. Failure to fit is a fatal error
// (host-side check after classify).
//   memory: 4 * MAX_MIGRANTS_PER_DIR * pack_size_per_cell bytes per rank.
//   At pack_size ~ 410 KB and the value below: 4 * 128 * 410 KB = ~205 MB.
static constexpr int MAX_MIGRANTS_PER_DIR = 128;

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

