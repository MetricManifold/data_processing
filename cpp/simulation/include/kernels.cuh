#pragma once

#include "cell.cuh"
#include "domain.cuh"
#include "types.cuh"
#ifdef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"
#endif
// Also need diagnostics.cuh for StressFieldBuffers when stress fields enabled
#if defined(STRESS_FIELDS_ENABLED) && !defined(DIAGNOSTICS_ENABLED)
#include "diagnostics.cuh"
#endif

//=============================================================================
// CUDA error checking macro — aborts with VRAM diagnostics on failure
//=============================================================================
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      size_t free_mem = 0, total_mem = 0;                                      \
      cudaMemGetInfo(&free_mem, &total_mem);                                   \
      fprintf(stderr,                                                          \
              "CUDA ERROR at %s:%d — %s\n"                                     \
              "  Call: %s\n"                                                    \
              "  VRAM: %.1f MB free / %.1f MB total\n",                        \
              __FILE__, __LINE__, cudaGetErrorString(err), #call,              \
              free_mem / (1024.0 * 1024.0),                                    \
              total_mem / (1024.0 * 1024.0));                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace cellsim {

//=============================================================================
// Constants for V4 neighbor-list optimization
//=============================================================================

#define MAX_NEIGHBORS 128 // Max neighbors per cell

//=============================================================================
// Kernel Launch Configuration
//=============================================================================

struct KernelConfig {
  dim3 block;
  dim3 grid;

  static KernelConfig for_cell(const Cell &cell) {
    dim3 block(16, 16);
    dim3 grid((cell.width() + block.x - 1) / block.x,
              (cell.height() + block.y - 1) / block.y);
    return {block, grid};
  }
};

//=============================================================================
// Host-side kernel launchers
//=============================================================================

// Forward declaration
class Integrator;

//=============================================================================
// MAIN SOLVER
//=============================================================================

// Main solver step function with neighbor-list optimization for interaction
// sync_centroids: if true, read centroids back to host for bbox updates
// rebuild_neighbors: if true, rebuild the neighbor list this step
// cached_max_size/max_w/max_h: pre-computed max dimensions (updated on bbox change)
// Note: MAX_NEIGHBORS defined at top of file
void step_fused(Domain &domain, float dt,
                float **d_all_phi_ptrs, float **d_all_phi_out_ptrs,
                int *d_all_widths,
                int *d_all_heights, int *d_all_offsets_x,
                int *d_all_offsets_y, int *d_all_field_sizes,
                float *d_volumes, float *d_integrals_x, float *d_integrals_y,
                float *d_centroid_sums, float *d_volume_deviations,
                float *d_velocities_x, float *d_velocities_y, float *d_ref_x,
                float *d_ref_y, float *d_polarization_x,
                float *d_polarization_y, float *d_centroids_x,
                float *d_centroids_y, int *d_neighbor_counts,
                int *d_neighbor_lists,
                float *d_v_A,
                float *d_gamma,
                float *d_two_gamma,
                float *d_two_gamma_bulk,
                float *d_target_area,
                float *d_volume_coeff,
                float *d_perimeters,
                float *d_second_moment_x,
                float *d_second_moment_y,
                int *d_old_widths,
                int *d_old_heights,
                int pool_max_side,
                int *d_max_wh,
                int *d_shift_x,
                int *d_shift_y,
                int *d_block_arrival,
                float *d_sum_field,
                float *d_sum_field_linear,
                float *d_next_sum_field,
                int &cached_max_size, int &cached_max_w, int &cached_max_h,
                bool sync_centroids = true,
                bool rebuild_neighbors = true,
                bool centroid_sums_precomputed = false,
                int step_counter = 0);

//=============================================================================
// DIAGNOSTICS (optional, enabled via DIAGNOSTICS_ENABLED)
//=============================================================================

#ifdef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"

// Run diagnostic computation after a step
// Recomputes gradients from phi on-the-fly (does not need work buffer)
void run_diagnostics(
    Domain &domain,
    float **d_all_phi_ptrs,
    int *d_all_widths,
    int *d_all_heights,
    int *d_all_offsets_x,
    int *d_all_offsets_y,
    int *d_neighbor_counts,
    int *d_neighbor_lists,
    DiagnosticBuffers &diag);
#endif

//=============================================================================
// STRESS FIELDS (optional, enabled via STRESS_FIELDS_ENABLED)
//=============================================================================

#ifdef STRESS_FIELDS_ENABLED
#include "diagnostics.cuh"

// Compute stress tensor fields: σ_xx(x,y), σ_yy(x,y), σ_xy(x,y), P(x,y)
// Call before exporting VTK if stress fields are desired
void compute_stress_fields(
    Domain &domain,
    float **d_all_phi_ptrs,
    int *d_all_widths,
    int *d_all_heights,
    int *d_all_offsets_x,
    int *d_all_offsets_y,
    StressFieldBuffers &stress);
#endif

//=============================================================================
// Narrow-Band Inline Skip (2D)
//
// Instead of maintaining compact active-pixel lists, each batched kernel
// tests phi*(1-phi) < threshold and early-exits for pixels deep inside
// the cell interior (phi≈1) or exterior (phi≈0). Only the interface band
// where the phase field transitions needs full computation.
//
// For R=49, λ=7: interface band ≈ 20 pixels wide → ~22% of subdomain is
// active. This skips ~78% of per-pixel computation with zero data-structure
// overhead.
//=============================================================================

// Threshold for narrow-band skip: phi*(1-phi) < this value → skip.
// Pixels deep inside cell interiors (phi≈1) or exteriors (phi≈0) have
// negligible dynamics and can be safely skipped.
//
// Threshold selection: f'(phi) = (60/λ²)*phi*(1-phi)*(1-2*phi).
//   threshold=0.10 → skip phi>0.887: |f'|=0.095, TOO AGGRESSIVE (prevents
} // namespace cellsim
