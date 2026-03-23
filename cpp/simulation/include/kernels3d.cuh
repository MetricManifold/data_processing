#pragma once

#include "cell3d.cuh"
#include "domain3d.cuh"
#include "types3d.cuh"
#include <curand_kernel.h>

namespace cellsim {

//=============================================================================
// 3D Kernel Launch Configuration
//=============================================================================

struct KernelConfig3D {
  dim3 block;
  dim3 grid;

  static KernelConfig3D for_cell(const Cell3D &cell) {
    dim3 block(8, 8, 8); // 512 threads per block
    dim3 grid((cell.width() + block.x - 1) / block.x,
              (cell.height() + block.y - 1) / block.y,
              (cell.depth() + block.z - 1) / block.z);
    return {block, grid};
  }

  static KernelConfig3D for_dims(int w, int h, int d) {
    dim3 block(8, 8, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y,
              (d + block.z - 1) / block.z);
    return {block, grid};
  }
};

//=============================================================================
// 3D Kernel Declarations
//=============================================================================

// Laplacian - 7-point stencil
__global__ void kernel_laplacian_3d(const float *__restrict__ phi,
                                    float *__restrict__ laplacian, int width,
                                    int height, int depth, float dx, float dy,
                                    float dz);

// Bulk potential derivative
__global__ void kernel_bulk_potential_3d(const float *__restrict__ phi,
                                         float *__restrict__ bulk_term,
                                         int width, int height, int depth,
                                         float bulk_coeff);

// Phi squared for volume integral
__global__ void kernel_phi_squared_3d(const float *__restrict__ phi,
                                      float *__restrict__ phi_sq, int width,
                                      int height, int depth, int halo);

// Volume constraint term
__global__ void kernel_volume_constraint_3d(const float *__restrict__ phi,
                                            float *__restrict__ constraint_term,
                                            int width, int height, int depth,
                                            float volume_deviation,
                                            float volume_coeff);

// Gradient computation
__global__ void kernel_gradient_3d(const float *__restrict__ phi,
                                   float *__restrict__ grad_x,
                                   float *__restrict__ grad_y,
                                   float *__restrict__ grad_z, int width,
                                   int height, int depth, float dx, float dy,
                                   float dz);

// Advection term
__global__ void kernel_advection_3d(const float *__restrict__ grad_x,
                                    const float *__restrict__ grad_y,
                                    const float *__restrict__ grad_z,
                                    float *__restrict__ advection_term,
                                    int width, int height, int depth, float vx,
                                    float vy, float vz);

// Motility integrand
__global__ void kernel_motility_integrand_3d(
    const float *__restrict__ phi, const float *__restrict__ grad_x,
    const float *__restrict__ grad_y, const float *__restrict__ grad_z,
    const float *__restrict__ interaction_sum, float *__restrict__ integrand_x,
    float *__restrict__ integrand_y, float *__restrict__ integrand_z, int width,
    int height, int depth, int halo);

// Interaction sum (cell-cell)
__global__ void kernel_interaction_sum_3d(
    const float *__restrict__ phi_i, float *__restrict__ interaction_sum,
    int width_i, int height_i, int depth_i, int offset_x_i, int offset_y_i,
    int offset_z_i, float **other_phi_ptrs, int *other_widths,
    int *other_heights, int *other_depths, int *other_offsets_x,
    int *other_offsets_y, int *other_offsets_z, int num_other_cells, int Nx,
    int Ny, int Nz);

// Repulsion term
__global__ void kernel_repulsion_3d(const float *__restrict__ phi,
                                    const float *__restrict__ interaction_sum,
                                    float *__restrict__ repulsion_term,
                                    int width, int height, int depth,
                                    float interaction_coeff);

// Combined RHS
__global__ void kernel_combine_rhs_3d(float *__restrict__ dphi_dt,
                                      const float *__restrict__ laplacian,
                                      const float *__restrict__ bulk_term,
                                      const float *__restrict__ constraint_term,
                                      const float *__restrict__ repulsion_term,
                                      const float *__restrict__ advection_term,
                                      int width, int height, int depth,
                                      float gamma);

// Forward Euler step
__global__ void kernel_euler_step_3d(float *__restrict__ phi,
                                     const float *__restrict__ dphi_dt,
                                     int size, float dt);

// Parallel reduction
__global__ void kernel_reduce_sum_3d(const float *__restrict__ input,
                                     float *__restrict__ output, int n);

// Fused local terms
__global__ void kernel_fused_local_3d(
    const float *__restrict__ phi, float *__restrict__ laplacian,
    float *__restrict__ bulk_term, float *__restrict__ grad_x,
    float *__restrict__ grad_y, float *__restrict__ grad_z, int width,
    int height, int depth, float dx, float dy, float dz, float bulk_coeff);

//=============================================================================
// Host-side functions
//=============================================================================

// Legacy functions - disabled when USE_HALF_PRECISION_3D is enabled
#ifndef USE_HALF_PRECISION_3D
// Compute volume integral using reduction
float compute_volume_integral_3d(const float *d_phi, float *d_work,
                                 int field_size, int halo, int width,
                                 int height, int depth);

// Compute all local terms for a single 3D cell
void compute_local_terms_3d(Cell3D &cell, const SimParams3D &params,
                            float *d_work_buffer, int buffer_stride);

// Compute interaction terms for all 3D cells
void compute_interaction_terms_3d(Domain3D &domain, float *d_work_buffer);

// Perform one Forward Euler step for all 3D cells (legacy)
void step_euler_3d(Domain3D &domain, float dt, float *d_work_buffer);
#endif // USE_HALF_PRECISION_3D

// MAX_NEIGHBORS_3D for neighbor list
constexpr int MAX_NEIGHBORS_3D = 32;

//=============================================================================
// Spatial Hash Grid Constants for O(N) Neighbor List Building
//=============================================================================
constexpr int SPATIAL_HASH_THRESHOLD = 64;  // Use spatial hash for N > 64 cells
constexpr int MAX_CELLS_PER_GRID = 16;      // Max cells per spatial grid cell

//=============================================================================
// GPU-side RNG kernels for polarization updates (curand)
//=============================================================================

// Initialize curand RNG states (call once at start)
__global__ void kernel_init_rng_states_3d(curandState *states,
                                          unsigned long long seed,
                                          int num_cells);

// Update polarizations on GPU using curand (Run-and-Tumble or ABP)
__global__ void kernel_update_polarizations_3d(
    curandState *states, float *polarizations_x, float *polarizations_y,
    float *polarizations_z, float dt, float tau, bool use_abp, int num_cells);

//=============================================================================
// Spatial Hash Grid Kernels for O(N) Neighbor List Building
//=============================================================================

// Build spatial hash grid: assign cells to grid bins
__global__ void kernel_build_spatial_grid_3d(
    const float *centroids_x, const float *centroids_y, const float *centroids_z,
    int *grid_counts, int *grid_cells,
    int Nx, int Ny, int Nz,
    int grid_nx, int grid_ny, int grid_nz,
    float cell_size, int num_cells);

// Build neighbor list using spatial hash grid - O(N) complexity
__global__ void kernel_build_neighbor_list_spatial_3d(
    const float *centroids_x, const float *centroids_y, const float *centroids_z,
    const int *grid_counts, const int *grid_cells,
    int *neighbor_counts, int *neighbor_lists,
    int Nx, int Ny, int Nz,
    int grid_nx, int grid_ny, int grid_nz,
    float cell_size, float search_radius, int num_cells);

// Kernel profiling - prints timing breakdown of step_fused_3d phases
// Enable with -DENABLE_KERNEL_PROFILING=ON in cmake
void print_3d_kernel_profile();

//=============================================================================
// Batched helper kernels (used by fused path in integrator)
//=============================================================================

__global__ void kernel_compute_ref_points_3d(
    float *__restrict__ ref_x, float *__restrict__ ref_y,
    float *__restrict__ ref_z,
    const int *__restrict__ offsets_x, const int *__restrict__ offsets_y,
    const int *__restrict__ offsets_z,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths,
    int Nx, int Ny, int Nz, int num_cells);

__global__ void kernel_reduce_volumes_batched_3d(
    float **__restrict__ phi_ptrs, float *__restrict__ volumes,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, const int *__restrict__ field_sizes,
    int halo, int num_cells);

__global__ void kernel_reduce_centroid_sums_batched_3d(
    float **__restrict__ phi_ptrs, float *__restrict__ centroid_sums,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y, const int *__restrict__ offsets_z,
    const int *__restrict__ field_sizes, const float *__restrict__ ref_x,
    const float *__restrict__ ref_y, const float *__restrict__ ref_z,
    int halo, int Nx, int Ny, int Nz, int num_cells);

__global__ void kernel_compute_centroids_and_deviations_3d(
    float *__restrict__ centroids_x, float *__restrict__ centroids_y,
    float *__restrict__ centroids_z, float *__restrict__ volume_deviations,
    const float *__restrict__ centroid_sums, const float *__restrict__ volumes,
    const float *__restrict__ ref_x, const float *__restrict__ ref_y,
    const float *__restrict__ ref_z, float target_volume, float dV,
    int Nx, int Ny, int Nz, int num_cells);

__global__ void kernel_compute_velocities_3d(
    float *__restrict__ velocities_x, float *__restrict__ velocities_y,
    float *__restrict__ velocities_z,
    const float *__restrict__ integrals_x,
    const float *__restrict__ integrals_y,
    const float *__restrict__ integrals_z,
    const float *__restrict__ polarizations_x,
    const float *__restrict__ polarizations_y,
    const float *__restrict__ polarizations_z,
    float motility_coeff, float dV, float v_A, int num_cells);

__global__ void __launch_bounds__(256, 4) kernel_velocity_integral_3d(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ depths,
    const int *__restrict__ field_sizes,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const int *__restrict__ offsets_z,
    const float *__restrict__ sum_field,
    float *__restrict__ d_integrals_x,
    float *__restrict__ d_integrals_y,
    float *__restrict__ d_integrals_z,
    float dx_grid, float dy_grid, float dz_grid,
    int halo, int Nx, int Ny, int Nz,
    int num_cells, int max_field_size);

// Fused ref_points + centroids (velocities computed separately after scatter)
__global__ void kernel_ref_centroid_vel_fused_3d(
    float *__restrict__ ref_x, float *__restrict__ ref_y,
    float *__restrict__ ref_z,
    const int *__restrict__ offsets_x, const int *__restrict__ offsets_y,
    const int *__restrict__ offsets_z,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths,
    float *__restrict__ centroids_x, float *__restrict__ centroids_y,
    float *__restrict__ centroids_z, float *__restrict__ volume_deviations,
    const float *__restrict__ centroid_sums, const float *__restrict__ volumes,
    float target_volume, float dV,
    int Nx, int Ny, int Nz, int num_cells);

//=============================================================================
// SCATTER: Accumulate φ²(x,y,z) from all cells onto global N³ sum field.
// S(x,y,z) = Σ_all φ_k²(x,y,z)
// Replaces O(k) neighbor-list interaction with O(1) sum field lookup.
//=============================================================================
__global__ void kernel_scatter_phi_sq_3d(
    float **__restrict__ phi_ptrs,
    float *__restrict__ sum_field,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ depths,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const int *__restrict__ offsets_z,
    const int *__restrict__ field_sizes,
    int Nx, int Ny, int Nz, int num_cells);

//=============================================================================
// FUSED KERNEL: Single-pass computes laplacian + bulk + constraint +
// interaction (via sum field) + advection + Euler step.
// Eliminates all work buffers. Everything in registers.
// Also accumulates volume + centroid sums via block-level shared-mem reduction.
//=============================================================================
// GPU-side phi pointer swap (shared with 2D, declared in kernels_shared.cu)
__global__ void kernel_swap_phi_ptrs(float **phi_ptrs, float **phi_out_ptrs, int num_cells);

//=============================================================================
// Fused step kernel for 3D
//=============================================================================
__global__ void kernel_fused_step_3d(
    float **__restrict__ phi_ptrs,
    float **__restrict__ phi_out_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ depths,
    const int *__restrict__ field_sizes,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const int *__restrict__ offsets_z,
    const float *__restrict__ sum_field,
    const float *__restrict__ volume_deviations,
    const float *__restrict__ velocities_x,
    const float *__restrict__ velocities_y,
    const float *__restrict__ velocities_z,
    float *__restrict__ d_centroid_sums,
    float *__restrict__ d_volumes,
    const float *__restrict__ ref_x,
    const float *__restrict__ ref_y,
    const float *__restrict__ ref_z,
    float volume_coeff, float interaction_coeff, float bulk_coeff,
    float gamma, float dx_grid, float dy_grid, float dz_grid, float dt,
    int halo, int Nx, int Ny, int Nz,
    int num_cells, int max_field_size);

// Optimized fused step for 3D - batched kernels, GPU-side reductions, neighbor list
// Uses FieldType3D for phi storage (FP16 when USE_HALF_PRECISION_3D is enabled)
// Now supports spatial hash grid for O(N) neighbor finding with large cell counts
void step_fused_3d(Domain3D &domain, float dt, float *d_work_buffer,
                   FieldType3D **d_all_phi_ptrs, int *d_all_widths,
                   int *d_all_heights, int *d_all_depths, int *d_all_offsets_x,
                   int *d_all_offsets_y, int *d_all_offsets_z,
                   int *d_all_field_sizes, float *d_volumes,
                   float *d_integrals_x, float *d_integrals_y,
                   float *d_integrals_z, float *d_centroid_sums,
                   float *d_volume_deviations, float *d_velocities_x,
                   float *d_velocities_y, float *d_velocities_z, float *d_ref_x,
                   float *d_ref_y, float *d_ref_z, float *d_polarization_x,
                   float *d_polarization_y, float *d_polarization_z,
                   float *d_centroids_x, float *d_centroids_y,
                   float *d_centroids_z, int *d_neighbor_counts,
                   int *d_neighbor_lists, bool sync_centroids,
                   bool rebuild_neighbors,
                   int *d_grid_counts = nullptr, int *d_grid_cells = nullptr,
                   cudaTextureObject_t *d_phi_textures = nullptr);

//=============================================================================
// GPU BOUNDING BOX PIPELINE FOR 3D
// Port of gpu_update_all_bboxes_2d — batched scan, GPU early exit,
// batched remap, GPU-side array patching.
//=============================================================================

// Returns true if any cell's bbox changed
bool gpu_update_all_bboxes_3d(
    Domain3D &domain,
    int *d_bbox_scan_results,
    float *d_centroids_x, float *d_centroids_y, float *d_centroids_z,
    float **d_phi_ptrs, int *d_widths, int *d_heights, int *d_depths,
    int *d_offsets_x, int *d_offsets_y, int *d_offsets_z,
    int *d_field_sizes, int max_field_size);

} // namespace cellsim
