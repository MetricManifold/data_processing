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
// Local Term Kernels (per-cell, no interaction)
//=============================================================================

// Compute Laplacian using 5-point stencil (halo provides boundary data)
// wrap_x/wrap_y indicate if subdomain wraps around domain boundary
__global__ void kernel_laplacian(const float *__restrict__ phi,
                                 float *__restrict__ laplacian, int width,
                                 int height, float dx, float dy, int halo,
                                 bool wrap_x = false, bool wrap_y = false);

// Compute bulk potential derivative: f'(φ) = (60/λ²) * φ(1-φ)(1-2φ)
__global__ void kernel_bulk_potential(const float *__restrict__ phi,
                                      float *__restrict__ bulk_term, int width,
                                      int height,
                                      float bulk_coeff // 30/λ²
);

// Compute φ² for volume integral (stores result, needs reduction)
__global__ void kernel_phi_squared(const float *__restrict__ phi,
                                   float *__restrict__ phi_sq, int width,
                                   int height,
                                   int halo // Skip halo cells in integration
);

// Parallel reduction for volume integral
__global__ void kernel_reduce_sum(const float *__restrict__ input,
                                  float *__restrict__ output, int n);

// Compute volume constraint contribution: 2*volume_coeff*(πR² - volume)*φ
__global__ void kernel_volume_constraint(const float *__restrict__ phi,
                                         float *__restrict__ constraint_term,
                                         int width, int height,
                                         float volume_deviation, // (πR² - ∫φ²)
                                         float volume_coeff      // μ/(πR²)
);

// Compute gradient components for advection (halo provides boundary data)
// wrap_x/wrap_y indicate if subdomain wraps around domain boundary
__global__ void kernel_gradient(const float *__restrict__ phi,
                                float *__restrict__ grad_x,
                                float *__restrict__ grad_y, int width,
                                int height, float dx, float dy, int halo,
                                bool wrap_x = false, bool wrap_y = false);

// Compute advection term: v · ∇φ
__global__ void kernel_advection(const float *__restrict__ grad_x,
                                 const float *__restrict__ grad_y,
                                 float *__restrict__ advection_term, int width,
                                 int height, float vx, float vy);

//=============================================================================
// Motility Kernels
//=============================================================================

// Compute φ_n * ∇φ_n * Σ_m φ_m² for motility integral
// This produces a vector field that gets integrated to find velocity
__global__ void kernel_motility_integrand(
    const float *__restrict__ phi, const float *__restrict__ grad_x,
    const float *__restrict__ grad_y,
    const float *__restrict__ interaction_sum, // Σ_m φ_m²
    float *__restrict__ integrand_x, float *__restrict__ integrand_y, int width,
    int height,
    int halo // Skip halo in integration
);

//=============================================================================
// Interaction Kernels (cell-cell repulsion)
//=============================================================================

// Compute sum of φ_j² from all other cells at each point of cell i's subdomain
// This requires reading from potentially overlapping subdomains
__global__ void
kernel_interaction_sum(const float *__restrict__ phi_i,     // Current cell's φ
                       float *__restrict__ interaction_sum, // Output: Σ_j φ_j²
                       int width_i, int height_i, int offset_x_i,
                       int offset_y_i,         // Cell i's bbox offset
                       float **other_phi_ptrs, // Pointers to other cells' φ
                       int *other_widths, int *other_heights,
                       int *other_offsets_x, int *other_offsets_y,
                       int num_other_cells, int Nx, int Ny // Global domain size
);

// Compute repulsion term: (60κ/λ²) * φ_i * Σ_j φ_j²
__global__ void
kernel_repulsion(const float *__restrict__ phi,
                 const float *__restrict__ interaction_sum,
                 float *__restrict__ repulsion_term, int width, int height,
                 float interaction_coeff // 30κ/λ² (doubled in EOM)
);

//=============================================================================
// Combined RHS Kernel
//=============================================================================

// Combine all terms into dφ/dt
// dφ/dt = -v·∇φ - 0.5 * (-2γ∇²φ + f'(φ) + volume_constraint + repulsion)
__global__ void kernel_combine_rhs(float *__restrict__ dphi_dt,
                                   const float *__restrict__ laplacian,
                                   const float *__restrict__ bulk_term,
                                   const float *__restrict__ constraint_term,
                                   const float *__restrict__ repulsion_term,
                                   const float *__restrict__ advection_term,
                                   int width, int height, float gamma);

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
// Note: MAX_NEIGHBORS defined at top of file
void step_fused(Domain &domain, float dt, float *d_work_buffer,
                float **d_all_phi_ptrs, int *d_all_widths,
                int *d_all_heights, int *d_all_offsets_x,
                int *d_all_offsets_y, int *d_all_field_sizes,
                float *d_volumes, float *d_integrals_x, float *d_integrals_y,
                float *d_centroid_sums, float *d_volume_deviations,
                float *d_velocities_x, float *d_velocities_y, float *d_ref_x,
                float *d_ref_y, float *d_polarization_x,
                float *d_polarization_y, float *d_centroids_x,
                float *d_centroids_y, int *d_neighbor_counts,
                int *d_neighbor_lists, bool sync_centroids = true,
                bool rebuild_neighbors = true);

//=============================================================================
// DIAGNOSTICS (optional, enabled via DIAGNOSTICS_ENABLED)
//=============================================================================

#ifdef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"

// Run diagnostic computation after a step
// Must be called after step_fused while work_buffer still contains gradients
void run_diagnostics(
    Domain &domain,
    float *d_work_buffer,
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

} // namespace cellsim
