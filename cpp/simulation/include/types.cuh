#pragma once
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// sim_v3 simulation parameters
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
    int halo = 0;                  // unused in v3 (kept for checkpoint round-trip)
    int save_interval = 0;
    int print_interval = 100;
    int trajectory_samples = 100;
    unsigned int seed = 0;
    unsigned int polarity_seed = 0;
    bool abp = false;

    __host__ __device__ double bulk_coeff()        const { return 30.0 / (lambda * lambda); }
    __host__ __device__ double interaction_coeff() const { return 30.0 * kappa / (lambda * lambda); }
    __host__ __device__ double target_area()       const { return M_PI * target_radius * target_radius; }
    __host__ __device__ double volume_coeff()      const { return mu / target_area(); }
    __host__ __device__ double motility_coeff()    const { return 60.0 * kappa / (xi * lambda * lambda); }
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
static constexpr int TILE_T        = 192;
static constexpr int TILE_AREA     = TILE_T * TILE_T;
static constexpr int REBIND_EVERY  = 10;

// Adaptive rect parameters (host constants used in init / k_rebind).
//   bbox_k       half-width = ceil(k * sigma + margin) per axis
//   bbox_margin  px added on each side; sized to R/2 in k_rebind so rect
//                over-covers the tanh tail (decays to <1e-3 at R + 3*lambda).
//   bbox_align   round half-width up to multiple (so rw = 2*hw is warp-aligned)
//   bbox_min     minimum half-width (avoids degenerate small rects)
static constexpr int TILE_BBOX_ALIGN = 16;
static constexpr int TILE_BBOX_MIN   = 32;

// ---------------------------------------------------------------------------
// All GPU arrays — SoA layout. Vastly simplified from sim_v2:
//   * no per-cell variable W/H, no halo, no shifts, no neighbour list,
//     no spatial hash, no resize tracking, no second-moment scratch.
//   * single contiguous phi pool of 2*N*TILE_AREA floats (double-buffered).
//   * per-cell scalars: gamma_cell + v_A_cell only (other physics constants
//     are global, broadcast to the kernel as launch arguments).
//   * observables (Cx, Cy, V, perimeter, velocities) are computed inside
//     the fused evolve kernel and exposed on the host for trajectory I/O.
// ---------------------------------------------------------------------------
struct CellArrays {
    int num_cells = 0;

    // Phi double buffer. `phi_in` points at the half currently holding state;
    // `phi_out` points at the scratch half. After each step we std::swap them
    // on the host (no kernel needed). Both point inside `phi_pool`.
    float* phi_pool = nullptr;     // [2 * N * TILE_AREA]
    float* phi_in   = nullptr;     // alias into phi_pool
    float* phi_out  = nullptr;     // alias into phi_pool

    // Global sum field S(x,y) = sum_n phi_n(x,y)^2. Atomic-scatter target.
    float* S = nullptr;            // [Nx * Ny]

    // Per-cell tile origin in global coords (gx0, gy0 interleaved).
    int* origin = nullptr;         // [2 * N]

    // Per-cell active rect (rx0, ry0, rw, rh) inside the TILE_T x TILE_T
    // buffer. Set by k_rebind from second-moment statistics; iterated by
    // every kernel that touches phi. Pixels outside the rect are
    // guaranteed zero (rebind zeros them), so skipping them is exact.
    int* rect = nullptr;           // [4 * N]

    // Per-cell observables produced by k_evolve_l1.
    float* volumes      = nullptr; // [N] : sum phi (tile-local; multiply by dA for area)
    float* Cx           = nullptr; // [N] : tile-local sum(phi^2 * lx)
    float* Cy           = nullptr; // [N] : tile-local sum(phi^2 * ly)
    float* Cxx          = nullptr; // [N] : tile-local sum(phi^2 * lx^2)
    float* Cyy          = nullptr; // [N] : tile-local sum(phi^2 * ly^2)
    float* perimeters   = nullptr; // [N] : sum |grad phi| (tile-local)
    float* velocities_x = nullptr; // [N] : interaction integral + v_A * polar_x
    float* velocities_y = nullptr; // [N] : interaction integral + v_A * polar_y

    // Polarisation. theta is the persistent angle; (px, py) = (cos, sin).
    float* polar_theta  = nullptr; // [N]
    float* polar_x      = nullptr; // [N]
    float* polar_y      = nullptr; // [N]

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
