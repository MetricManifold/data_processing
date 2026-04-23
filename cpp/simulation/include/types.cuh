#pragma once
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
// Scalar physics/numerical knobs use double precision so that time accumulation
// (cur_time += dt), step-count math (t_end / dt), and derived coefficients
// never lose precision. Per-pixel φ and per-cell state remain single precision
// — they live on the GPU and are multiplied into f32 math regardless.
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
    double subdomain_padding = 0.6;
    int halo = 4;
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
// Constants
// ---------------------------------------------------------------------------
static constexpr int K_MAX = 48;          // max neighbors per cell
static constexpr int HASH_MAX_PER_BIN = 128;  // bin capacity; must exceed max cells per bin.
                                              // With small domains (e.g. 2x2 bin grid, 72 cells)
                                              // one bin easily holds 30+ cells. Silent overflow
                                              // drops cells from the hash → asymmetric neighbor
                                              // lists → cells fuse. 128 is safe for all tested sizes.

// ---------------------------------------------------------------------------
// Neighbor entry stored in the per-cell neighbor list.
// We store identity only. Positions and dimensions are fetched from the
// current OX/OY/W/H arrays inside k_fused so coordinates are always in
// the "now" frame — never stale across non-rebuild steps.
// ---------------------------------------------------------------------------
struct NeighborEntry {
    int cell_id;
};

// ---------------------------------------------------------------------------
// All GPU arrays — SoA layout
// ---------------------------------------------------------------------------
struct CellArrays {
    int num_cells = 0;
    size_t slot_size = 0;
    int max_side = 0;

    // Phi double buffer (contiguous pool)
    float*  phi_pool     = nullptr;
    float** phi_ptrs     = nullptr;   // [N] current read
    float** phi_out_ptrs = nullptr;   // [N] current write

    // Geometry
    int* offsets_x   = nullptr;
    int* offsets_y   = nullptr;
    int* widths      = nullptr;
    int* heights     = nullptr;
    int* old_widths  = nullptr;
    int* old_heights = nullptr;
    int* shift_x     = nullptr;
    int* shift_y     = nullptr;

    // Per-cell dynamics
    float* velocities_x = nullptr;
    float* velocities_y = nullptr;
    float* volumes       = nullptr;
    float* volume_devs   = nullptr;
    float* centroids_x   = nullptr;
    float* centroids_y   = nullptr;
    float* ref_x         = nullptr;
    float* ref_y         = nullptr;
    float* perimeters    = nullptr;
    float* moment_x      = nullptr;
    float* moment_y      = nullptr;

    // Polarization
    float* polar_x = nullptr;
    float* polar_y = nullptr;
    float* polar_theta = nullptr;  // persistent angle; (polar_x, polar_y) are derived

    // Per-cell constants (set once at init)
    float* two_gamma      = nullptr;
    float* two_gamma_bulk = nullptr;
    float* vol_coeff      = nullptr;
    float* tgt_area       = nullptr;
    float* tgt_radius     = nullptr;
    float* v_A_cell       = nullptr;

    // Neighbor list
    NeighborEntry* nbr_list  = nullptr;   // [N * K_MAX]
    int*           nbr_count = nullptr;   // [N]

    // Spatial hash
    int* hash_ids    = nullptr;
    int* hash_counts = nullptr;
    int  hash_bin_sz = 0;
    int  hash_nx = 0, hash_ny = 0;

    // Resize tracking
    int* d_max_wh = nullptr;   // [2]

    // RNG
    void* rng_states = nullptr;
};

// ---------------------------------------------------------------------------
// Host-side cell for initialization
// ---------------------------------------------------------------------------
struct CellHost {
    double cx, cy, radius, gamma, v_A;
    int ox, oy, w, h;
};
