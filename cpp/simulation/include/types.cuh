#pragma once
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Physics coefficient helpers (single source of truth)
// ---------------------------------------------------------------------------
// Free `__host__ __device__` templates so both SimParams (double, host)
// and GPU kernels (float, device) call into one definition. The literal
// constants `30` and `60` come from the variational derivative of the
// Cahn-Hilliard + repulsion + advection free energy at Palmieri's
// scaling; see study/adhesion/manuscript.tex Eq. (S15). Both kernels
// and SimParams used to recompute these inline, drifting silently if
// the physics changed.
//
// Note: the Rust CPU reference (rust/cpu_ref/src/sim.rs) uses a
// factor-of-2 different form because it integrates the raw -delta F /
// delta phi while the GPU integrates -1/2 delta F / delta phi. That
// split is deliberate (independent oracle) — don't fold them.
// ---------------------------------------------------------------------------
template <typename T>
__host__ __device__ inline T bulk_coeff(T lambda) {
    return T(30) / (lambda * lambda);
}
template <typename T>
__host__ __device__ inline T interaction_coeff(T kappa, T lambda) {
    return T(30) * kappa / (lambda * lambda);
}
template <typename T>
__host__ __device__ inline T motility_coeff(T kappa, T xi, T lambda) {
    return T(60) * kappa / (xi * lambda * lambda);
}

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
// Scalar physics/numerical knobs in double precision so time accumulation,
// step-count math, and derived coefficients never lose precision. Per-pixel
// phi and per-cell observables remain f32 — they live on the GPU.
// ---------------------------------------------------------------------------
struct SimParams {
    int Nx = 0, Ny = 0;
    double dx = 1.0, dy = 1.0;
    double dt = 0.01;
    double t_end = 100.0;
    double lambda = 7.0;
    double gamma = 1.0;
    double kappa = 10.0;
    double target_radius = 20.0;
    double mu = 1.0;
    double v_A = 0.0;
    double xi = 1500.0;
    double tau = 10000.0;
    // Adaptive-rect half-width multiplier on cell sigma (per-axis):
    //   hwx = ceil(subdomain_padding * sigma_x + R/2), clamped & aligned.
    // Default 2.0; tighter values trade physics fidelity for speed. The
    // field was a dead leftover in v3..v7 and is repurposed here, so
    // resumes from older checkpoints reset it to the default at load.
    double subdomain_padding = 2.0;
    // [DEPRECATED RUNTIME, KEPT FOR BINARY LAYOUT]
    //
    // Legacy v3-v6 checkpoints stored a per-cell tile halo dimension
    // alongside an interior W/H pair. The fixed-tile rewrite eliminated
    // both — every cell now uses TILE_T x TILE_T with the active region
    // tracked by CellArrays::rect. No runtime code reads `halo` anymore;
    // decode_simparams threads the saved value as `halo_legacy` to
    // decode_cell_records, which uses it ONLY when re-tiling legacy
    // checkpoint rows.
    //
    // The field stays in SimParams because removing it would shift
    // sizeof(SimParams) and break v8 binary layout (which fwrites the
    // struct verbatim). Do NOT add new uses; do NOT delete without a
    // checkpoint format bump.
    int halo = 0;
    int save_interval = 0;
    int print_interval = 100;
    int trajectory_samples = 100;
    unsigned int seed = 0;
    unsigned int polarity_seed = 0;
    bool abp = false;

    __host__ __device__ double bulk_coeff()        const { return ::bulk_coeff(lambda); }
    __host__ __device__ double interaction_coeff() const { return ::interaction_coeff(kappa, lambda); }
    __host__ __device__ double target_area()       const { return M_PI * target_radius * target_radius; }
    __host__ __device__ double volume_coeff()      const { return mu / target_area(); }
    __host__ __device__ double motility_coeff()    const { return ::motility_coeff(kappa, xi, lambda); }
    __host__ __device__ double dA()                const { return dx * dy; }
};

// ---------------------------------------------------------------------------
// Tile size and adaptive rect.
// ---------------------------------------------------------------------------
// The phi buffer is a fixed TILE_T x TILE_T tile per cell, but kernels
// iterate only the *active rect* (rx0, ry0, rw, rh) inside it. The rect is
// recomputed every REBIND_EVERY steps from the cell's second moments
// (Cxx/Cyy), with the COM landing on (T/2, T/2). This lets the buffer
// accommodate elongated cells (up to ~T-2 px) without paying full T^2
// work for round cells.
//
// Sizing: at R=49, lambda=7 the tanh interface decays to <1e-3 by r =
// R + 3*lambda = 70 px from COM (~10*lambda for FP-noise floor). For
// elongated cells the LONG-AXIS extent is what counts: a cell at aspect
// ratio a/b=5 reaches ~97 px from COM, needing hw >= 97 + 3*lambda =
// 118 px for safe interface decay (>= 132 px for full FP-floor budget).
// T=320 leaves T/2-1 = 159 px clearance — comfortably handles the
// stretchy-cell-through-gap regime. T=192 (ceiling 95) was tight for
// extreme deformation: a cell at aspect 4 already grazes the ceiling.
// The runtime guards against domains smaller than TILE_T to keep the
// bbox-comparison helpers alias-free.
//
// Per-cell memory cost (double-buffered, f32): 2*T^2*4 bytes/cell.
// T=192 -> 295 KB/cell ; T=320 -> 819 KB/cell. At N=4608 that is
// 3.8 GB — fine on H100/A100, irrelevant for typical N=288/1152 runs.
//
// All kernels iterate p = 0..rw*rh and decode (lx, ly) by /, %; T does
// NOT need to be a power of two.
// ---------------------------------------------------------------------------
#ifndef CELL_SIM_TILE_T
#define CELL_SIM_TILE_T 320
#endif
static constexpr int TILE_T        = CELL_SIM_TILE_T;
static_assert(TILE_T % 2 == 0, "TILE_T must be even");
static constexpr int TILE_AREA     = TILE_T * TILE_T;
static constexpr int REBIND_EVERY  = 8;

// ---------------------------------------------------------------------------
// Multi-GPU slab decomposition (1D along y).
//
// HALO_H is the maximum number of pixels a cell's nonzero phi can extend
// past its COM along either axis. It is hard-fixed by the rect clamp in
// k_rebind: hmax = TILE_T/2 - 1, after which the cell's phi is identically
// zero. So if a cell's rebind-time COM is at row cy_r, all its nonzero
// pixels lie in rows [cy_r - HALO_H, cy_r + HALO_H]. Owning rank g for
// rows [slab_lo[g], slab_hi[g]) therefore needs S allocated for rows
// [slab_lo[g] - HALO_H, slab_hi[g] + HALO_H) — the "extended slab".
// No alignment slack needed; the contract is tight by construction.
//
// Cell migration (transfer of ownership when COM crosses slab boundary) is
// done at rebind, before scatter, so the contract is re-established every
// REBIND_EVERY steps using the up-to-date COM. Between rebinds, COM drift
// is bounded by velocity * REBIND_EVERY * dt and is at most a few pixels.
// ---------------------------------------------------------------------------
static constexpr int HALO_H        = TILE_T / 2 - 1;

// Adaptive rect parameters (host constants used in init / k_rebind).
//   bbox_k       half-width = ceil(k * sigma + margin) per axis
//   bbox_margin  px added on each side; sized to R/2 in k_rebind so rect
//                over-covers the tanh tail (decays to <1e-3 at R + 3*lambda).
//   bbox_align   round half-width up to multiple (so rw = 2*hw is warp-aligned)
//   bbox_min     minimum half-width (avoids degenerate small rects)
static constexpr int TILE_BBOX_ALIGN = 16;
static constexpr int TILE_BBOX_MIN   = 32;

// ---------------------------------------------------------------------------
// SlabInfo — per-rank y-axis partition.
//
// Holds enough state for a kernel to translate a global pixel y in
// [0, Ny) to a local row index in [0, ext_height). The slab covers global
// rows [y_lo, y_hi) and stores HALO_H rows on each side, giving an
// extended height ext_height = (y_hi - y_lo) + 2 * HALO_H. The buffer
// for S is allocated with that height; row 0 of the buffer corresponds
// to global row (y_lo - HALO_H) mod Ny.
//
// For the single-GPU build path (gpus == 1) we set y_lo = 0, y_hi = Ny,
// halo_h = 0, ext_height = Ny — i.e. the slab is the whole grid and the
// translation is the identity. This keeps the kernel code uniform
// regardless of gpus.
// ---------------------------------------------------------------------------
struct SlabInfo {
    int y_lo      = 0;   // first global row this rank owns
    int y_hi      = 0;   // one past last global row this rank owns
    int halo_h    = 0;   // HALO_H for multi-rank, 0 for single-rank
    int ext_height = 0;  // (y_hi - y_lo) + 2 * halo_h, == Ny for single-rank
    int Ny        = 0;   // global Ny (needed for periodic wrap)
};

// __device__ y-translation helper. Maps a global row index gy (in
// [0, Ny)) to the local row index in our slab buffer.
//
// For multi-GPU, the slab "wraps around" the periodic boundary if
// (y_lo - halo_h) is negative or (y_hi + halo_h) exceeds Ny. The math is
// uniform: subtract the buffer's leftmost global row (y_lo - halo_h) and
// add Ny once if we underflow, mod Ny.
//
// For G=1 this is exactly the identity (since y_lo=0, halo_h=0, ext=Ny).
//
// IMPORTANT: returns a value in [0, Ny). The caller is responsible for
// ensuring it is also in [0, ext_height) — i.e. that this rank's slab
// actually covers this global row. For multi-GPU runs this is the
// "ownership contract": a cell owned by rank g writes into S only at
// rows within g's window. Use slab_in_window() below to check.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__
int slab_local_y(int gy, int y_lo, int halo_h, int ext_height, int Ny) {
    int dy = gy - y_lo + halo_h;
    if (dy < 0)            dy += Ny;
    else if (dy >= Ny)     dy -= Ny;
    return dy;
}

// True iff a slab_local_y() result is a valid index into the slab buffer
// (i.e. the global row is inside this rank's [y_lo - halo_h, y_hi + halo_h)
// window). Always true for G=1 (ext_height == Ny). Used for debug
// bounds-checking under -DCELL_SIM_SLAB_BOUNDS_CHECK.
__host__ __device__ __forceinline__
bool slab_in_window(int sy_local, int ext_height) {
    return (unsigned)sy_local < (unsigned)ext_height;
}

// ---------------------------------------------------------------------------
// All GPU arrays — SoA layout:
//   * no per-cell variable W/H, no halo, no shifts, no neighbour list,
//     no spatial hash, no resize tracking, no second-moment scratch.
//   * single contiguous phi pool of 2*N*TILE_AREA floats (double-buffered).
//   * per-cell scalars: gamma_cell + v_A_cell only (other physics constants
//     are global, broadcast to the kernel as launch arguments).
//   * observables (Cx, Cy, V, perimeter, velocities) are computed inside
//     the fused evolve kernel and exposed on the host for trajectory I/O.
// ---------------------------------------------------------------------------

// Packed-per-cell layout used by the async trajectory writer. Produced by
// k_pack_traj into a contiguous device buffer; one cudaMemcpyAsync moves
// the whole batch to pinned host. Field order = the order the formatter
// reads in write_trajectory_snapshot; keep them aligned.
struct TrajPackedCell {
    int   ox, oy;        // origin (cells.origin[2i], [2i+1])
    float V, Cx, Cy;
    float per;
    float vx, vy;
    float px, py;
    float vA;
};

struct CellArrays {
    int num_cells = 0;
    // Allocated capacity for all per-cell arrays. For G=1 capacity ==
    // num_cells. For G>1 capacity > num_cells so cell migration between
    // ranks (at rebind boundaries) can grow num_cells without realloc.
    // Set in alloc_gpu(); kernels iterate n < num_cells but allocations
    // are sized by capacity.
    int capacity  = 0;

    // Phi double buffer. `phi_in` points at the half currently holding state;
    // `phi_out` points at the scratch half. After each step we std::swap them
    // on the host (no kernel needed). Both point inside `phi_pool`.
    float* phi_pool = nullptr;     // [2 * capacity * TILE_AREA]
    float* phi_in   = nullptr;     // alias into phi_pool (current state half)
    float* phi_out  = nullptr;     // alias into phi_pool (scratch half)

    // Global sum field S(x,y) = sum_n phi_n(x,y)^2. Atomic-scatter target.
    //
    // S is double-buffered:
    // S_pool[0] and S_pool[1] are one contiguous cudaMalloc of 2*S_bytes
    // (so a single L2 access-policy window pins both halves). `S` is an
    // alias into S_pool[parity], refreshed by Simulation::sync_pool_to_parity.
    float* S          = nullptr;     // [Nx * Ny]  (alias of S_pool[parity])
    float* S_pool[2]  = {nullptr, nullptr};

    // Slab partition descriptor for S (single-GPU defaults: covers the
    // whole grid). All kernels that touch S use these to translate a
    // global y coordinate to a local row index in the S buffer; for the
    // single-GPU defaults the translation is the identity.
    int S_y_lo       = 0;          // first global row this rank's S covers
    int S_halo_h     = 0;          // halo rows on each side (G=1: 0)
    int S_ext_height = 0;          // (y_hi - y_lo) + 2*halo_h; equals Ny for G=1

    // Per-cell tile origin in global coords (gx0, gy0 interleaved).
    int* origin = nullptr;         // [2 * N]

    // Per-cell active rect (rx0, ry0, rw, rh) inside the TILE_T x TILE_T
    // buffer. Set by k_rebind from second-moment statistics; iterated by
    // every kernel that touches phi. Pixels outside the rect are
    // guaranteed zero (rebind zeros them), so skipping them is exact.
    int* rect = nullptr;           // [4 * N]

    // Per-cell observables produced by the multi-block reduce / RHS path.
    //
    // volumes/Ix/Iy are
    // double-buffered (V_pool[2], Ix_pool[2], Iy_pool[2]) so the kernel
    // can read lagged moments from one half while atomicAdding fresh
    // moments into the other. `volumes`, `Ix`, `Iy` are aliases into the
    // parity-current half, refreshed by Simulation::sync_pool_to_parity.
    // Per-cell scalar accumulators: stored as f64 so that the cross-CTA
    // atomicAdd reductions are commutative within FP precision (f32
    // atomicAdd order varies run-to-run, costing a few ULP). Read by
    // velocity finalisation, rebind COM, visualizer, trajectory writer
    // (all of which downcast to float at the consumer).
    double* volumes      = nullptr; // [N] : sum phi^2 (tile-local)
    double* Ix           = nullptr; // [N] : tile-local sum(c * grad_x * S_other)
    double* Iy           = nullptr; // [N] : tile-local sum(c * grad_y * S_other)
    double* V_pool [2]   = {nullptr, nullptr};
    double* Ix_pool[2]   = {nullptr, nullptr};
    double* Iy_pool[2]   = {nullptr, nullptr};
    double* Cx           = nullptr; // [N] : tile-local sum(phi^2 * lx)
    double* Cy           = nullptr; // [N] : tile-local sum(phi^2 * ly)
    double* Cxx          = nullptr; // [N] : tile-local sum(phi^2 * lx^2)
    double* Cyy          = nullptr; // [N] : tile-local sum(phi^2 * ly^2)
    double* perimeters   = nullptr; // [N] : sum |grad phi| (tile-local)
    float* velocities_x = nullptr; // [N] : interaction integral + v_A * polar_x
    float* velocities_y = nullptr; // [N] : interaction integral + v_A * polar_y

    // Per-CTA work list of 32x32 sub-tiles to
    // evaluate (one CTA per item). Built host-side from cells.rect after
    // every rebind. d_work is sized for the worst-case at capacity (every
    // cell with a full TILE_T-2 rect); workCount is the active size used
    // as the launch grid. d_work_cap counts WorkItems (not bytes).
    void* d_work       = nullptr;  // really WorkItem*; void* to keep this header light
    int*  d_work_count = nullptr;  // device-side count of valid worklist entries
    int   d_work_cap   = 0;
    int   workCount    = 0;

    // Fused-rebind scratch: per-cell (sx, sy) shift and (rx0, ry0, rw, rh)
    // new-rect computed by launch_compute_rebind_meta before the rebind
    // step. Consumed by launch_step_rebind, then applied to origin/rect
    // by launch_apply_rebind_meta. Sized to capacity in alloc_gpu.
    int*  shift_xy     = nullptr;  // [2 * cap]
    int*  new_rect     = nullptr;  // [4 * cap]

    // Polarisation. theta is the persistent angle; (px, py) = (cos, sin).
    float* polar_theta  = nullptr; // [N]
    float* polar_x      = nullptr; // [N]
    float* polar_y      = nullptr; // [N]

    // Next scheduled tumble time per cell (continuous time, double).
    // k_polar fires a tumble iff cur_time >= next_tumble_time[n]. On a fire,
    // it re-draws (theta, next_tumble_time) from the per-cell curand stream.
    // Avoids the per-step Bernoulli draw of the legacy k_polar; for typical
    // dt/tau ~ 1e-6, the per-step check is essentially free.
    double* next_tumble_time = nullptr; // [N]

    // Per-cell physics scalars (gamma & v_A may vary per cell).
    float* gamma_cell   = nullptr; // [N]
    float* v_A_cell     = nullptr; // [N]
    float* tgt_radius   = nullptr; // [N]  (kept per-cell so future R disorder is trivial)

    // RNG state for k_polar (curandState, one per cell).
    void* rng_states    = nullptr;
};

// ---------------------------------------------------------------------------
// Host-side cell descriptor used during initialisation only.
// ---------------------------------------------------------------------------
struct CellHost {
    double cx, cy;          // cell COM in global coords
    double radius;          // target radius
    double gamma;           // surface tension
    double v_A;             // active speed
    int    ox, oy;          // tile origin (global coord of phi_pool[n,0,0])
};
