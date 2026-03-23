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
                float *d_target_area,
                float *d_volume_coeff,
                float *d_perimeters,
                float *d_sum_field,
                float *d_sum_field_linear,
                float *d_next_sum_field,
                int cached_max_size, int cached_max_w, int cached_max_h,
                bool sync_centroids = true,
                bool rebuild_neighbors = true,
                bool centroid_sums_precomputed = false);

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
// GPU Bounding Box Scan kernel (per-cell, finds extent + edge proximity)
// Output: d_results[7] = {max_dist_x, max_dist_y,
//                          min_lx, max_lx, min_ly, max_ly, found_any}
//=============================================================================

__global__ void kernel_init_bbox_scan_results(int *results, int num_cells);

__global__ void kernel_bbox_scan_2d(
    const float *__restrict__ phi,
    int width, int height,
    int offset_x, int offset_y,
    const float *__restrict__ d_centroids_x,
    const float *__restrict__ d_centroids_y,
    int cell_idx,
    int Nx, int Ny,
    int halo, float threshold,
    int *__restrict__ results);

// GPU remap kernel (copies phi from old bbox to new bbox, all on device)
__global__ void kernel_bbox_remap_2d(
    const float *__restrict__ old_phi,
    float *__restrict__ new_phi,
    int old_w, int old_h,
    int old_ox, int old_oy,
    int new_w, int new_h,
    int new_ox, int new_oy,
    int Nx, int Ny);

// Async bbox scan + change detection. Launches scan kernels and copies the
// change flag to pinned host memory via cudaMemcpyAsync (no pipeline drain).
// The caller reads h_any_change_pinned on a later step.
void gpu_launch_bbox_scan_async_2d(
    float **d_all_phi_ptrs,
    int *d_all_widths, int *d_all_heights,
    int *d_all_offsets_x, int *d_all_offsets_y,
    float *d_centroids_x, float *d_centroids_y,
    const SimParams &params,
    int num_cells, int max_field_size,
    int *d_bbox_scan_results,
    int *d_any_change_flag,
    int *h_any_change_pinned);

// Host function: GPU-accelerated bbox update for all 2D cells (every step)
// Returns true if any bbox changed
// Pool params: if pool is active, remap within pool slots instead of cudaMalloc/Free
// Device arrays: phi pointers, widths, heights, offsets for batched scan kernel
// Also patches device arrays directly on GPU (eliminates update_interaction_arrays H→D)
bool gpu_update_all_bboxes_2d(Domain &domain, int *d_bbox_scan_results,
                              float *d_centroids_x, float *d_centroids_y,
                              float *d_phi_pool = nullptr,
                              size_t pool_slot_size = 0,
                              int pool_num_cells = 0,
                              bool *pool_needs_grow = nullptr,
                              float **d_all_phi_ptrs = nullptr,
                              float **d_all_phi_out_ptrs = nullptr,
                              int *d_all_widths = nullptr,
                              int *d_all_heights = nullptr,
                              int *d_all_offsets_x = nullptr,
                              int *d_all_offsets_y = nullptr,
                              int *d_all_field_sizes = nullptr,
                              int max_field_size = 0);

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
