#pragma once

#include "cell3d.cuh"
#include "domain3d.cuh"
#include "types3d.cuh"

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

// MAX_NEIGHBORS_3D for neighbor list
constexpr int MAX_NEIGHBORS_3D = 32;

// Optimized fused step for 3D - batched kernels, GPU-side reductions, neighbor list
void step_fused_3d(Domain3D &domain, float dt, float *d_work_buffer,
                   float **d_all_phi_ptrs, int *d_all_widths,
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
                   bool rebuild_neighbors);

} // namespace cellsim
