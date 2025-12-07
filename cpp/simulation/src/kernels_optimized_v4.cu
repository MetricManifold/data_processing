#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace cellsim {

//=============================================================================
// V4 OPTIMIZATION: NEIGHBOR-LIST BASED INTERACTION (ALL ON GPU)
//
// Instead of O(N²) loop over all cells per pixel, use a neighbor list
// to iterate only over potentially interacting cells.
//=============================================================================

// MAX_NEIGHBORS_V4 is defined in kernels.cuh

//=============================================================================
// GPU kernel to compute reference points from bbox data (eliminates CPU memcpy)
// ref = bbox center wrapped to [0, N)
//=============================================================================

__global__ void kernel_compute_ref_points(float *__restrict__ ref_x,
                                          float *__restrict__ ref_y,
                                          const int *__restrict__ offsets_x,
                                          const int *__restrict__ offsets_y,
                                          const int *__restrict__ widths,
                                          const int *__restrict__ heights,
                                          int Nx, int Ny, int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells)
    return;

  // Compute bbox center from offset and dimensions
  float rx = (float)offsets_x[i] + (float)widths[i] * 0.5f;
  float ry = (float)offsets_y[i] + (float)heights[i] * 0.5f;

  // Wrap to [0, N) using proper modulo for negative values
  rx = fmodf(fmodf(rx, (float)Nx) + (float)Nx, (float)Nx);
  ry = fmodf(fmodf(ry, (float)Ny) + (float)Ny, (float)Ny);

  ref_x[i] = rx;
  ref_y[i] = ry;
}

//=============================================================================
// GPU kernel to build neighbor list based on centroid distance
//
// Two cells can only interact if their subdomains overlap. Since subdomains
// extend ~R+padding from the centroid, cells whose centroids are more than
// ~2*(R+padding) apart cannot have overlapping subdomains.
//
// We use 4*R as a conservative search radius - this guarantees we catch all
// potential neighbors while still providing O(k) speedup for large systems.
// For 72 cells in 800x800 with R=49, average density means ~8-12 neighbors.
// For 288 cells, this will be even more important.
//=============================================================================

__global__ void kernel_build_neighbor_list(
    const float *__restrict__ centroids_x,
    const float *__restrict__ centroids_y, const int *__restrict__ widths,
    const int *__restrict__ heights, const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y, int *__restrict__ neighbor_counts,
    int *__restrict__ neighbor_lists, // [MAX_NEIGHBORS_V4 * num_cells]
    int Nx, int Ny, int num_cells,
    float search_radius) // Should be ~4*R to be safe
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells)
    return;

  float cx_i = centroids_x[i];
  float cy_i = centroids_y[i];

  // Search radius squared for comparison
  float search_r2 = search_radius * search_radius;

  int count = 0;
  int *my_neighbors = neighbor_lists + i * MAX_NEIGHBORS_V4;

  for (int j = 0; j < num_cells; ++j) {
    if (j == i)
      continue;

    float cx_j = centroids_x[j];
    float cy_j = centroids_y[j];

    // Compute distance with periodic wrapping
    float dx = cx_j - cx_i;
    float dy = cy_j - cy_i;

    // Periodic boundary: if distance > half domain, wrap
    if (dx > Nx * 0.5f)
      dx -= Nx;
    else if (dx < -Nx * 0.5f)
      dx += Nx;
    if (dy > Ny * 0.5f)
      dy -= Ny;
    else if (dy < -Ny * 0.5f)
      dy += Ny;

    float dist2 = dx * dx + dy * dy;

    // Include as neighbor if within search radius
    if (dist2 <= search_r2) {
      if (count < MAX_NEIGHBORS_V4) {
        my_neighbors[count] = j;
        count++;
      }
    }
  }

  neighbor_counts[i] = count;
}

//=============================================================================
// Neighbor-list version of interaction kernel
//=============================================================================

__global__ void kernel_interaction_neighborlist_v4(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ offsets_x, const int *__restrict__ offsets_y,
    const int *__restrict__ neighbor_counts, // Number of neighbors per cell
    const int *__restrict__ neighbor_lists,  // Flattened neighbor indices
                                             // [MAX_NEIGHBORS_V4 * num_cells]
    float interaction_coeff, int halo, int Nx, int Ny, int num_cells,
    int max_field_size) {
  int cell_idx = blockIdx.z;
  if (cell_idx >= num_cells)
    return;

  int width = widths[cell_idx];
  int height = heights[cell_idx];

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;

  if (lx >= width || ly >= height)
    return;

  int idx = ly * width + lx;
  int base = cell_idx * 9 * max_field_size;

  const float *phi_i = phi_ptrs[cell_idx];
  const float *d_grad_x = work_buffer + base + 3 * max_field_size;
  const float *d_grad_y = work_buffer + base + 4 * max_field_size;
  float *d_repulsion = work_buffer + base + 6 * max_field_size;
  float *d_integrand_x = work_buffer + base + 7 * max_field_size;
  float *d_integrand_y = work_buffer + base + 8 * max_field_size;

  int offset_x_i = offsets_x[cell_idx];
  int offset_y_i = offsets_y[cell_idx];

  int gx = ((offset_x_i + lx) % Nx + Nx) % Nx;
  int gy = ((offset_y_i + ly) % Ny + Ny) % Ny;

  float phi_i_val = phi_i[idx];
  float grad_x = d_grad_x[idx];
  float grad_y = d_grad_y[idx];

  // Sum of φ_j² over neighbor cells (NOT (Σφ_j)²)
  float sum_phi_j_sq = 0.0f;

  // Only iterate over neighbors (O(k) instead of O(N))
  int num_neighbors = neighbor_counts[cell_idx];
  const int *my_neighbors = neighbor_lists + cell_idx * MAX_NEIGHBORS_V4;

  for (int n = 0; n < num_neighbors; ++n) {
    int j = my_neighbors[n];

    int ow = widths[j];
    int oh = heights[j];
    int ox = offsets_x[j];
    int oy = offsets_y[j];

    int ljx = ((gx - ox) % Nx + Nx) % Nx;
    int ljy = ((gy - oy) % Ny + Ny) % Ny;

    if (ljx < ow && ljy < oh) {
      float phi_j = phi_ptrs[j][ljy * ow + ljx];
      sum_phi_j_sq += phi_j * phi_j; // Σ(φ_j²)
    }
  }

  // Repulsion: δF_int/δφ_n = 2 * (30κ/λ²) * φ_n * Σ_{m≠n} φ_m²
  d_repulsion[idx] = 2.0f * interaction_coeff * phi_i_val * sum_phi_j_sq;

  // Velocity integrand: φ_n * (∇φ_n) * Σ_{m≠n} φ_m²
  if (lx < halo || lx >= width - halo || ly < halo || ly >= height - halo) {
    d_integrand_x[idx] = 0.0f;
    d_integrand_y[idx] = 0.0f;
  } else {
    d_integrand_x[idx] = phi_i_val * grad_x * sum_phi_j_sq;
    d_integrand_y[idx] = phi_i_val * grad_y * sum_phi_j_sq;
  }
}

//=============================================================================
// V4 step function: Same as V2 but with neighbor-list for interaction
//=============================================================================

// Forward declarations - these are defined in kernels_optimized_v2.cu
__global__ void kernel_fused_local_batched_v2(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights, float dx,
    float dy, float bulk_coeff, int halo_width, int num_cells,
    int max_field_size);

__global__ void kernel_reduce_volumes_batched_v2(
    const float *__restrict__ work_buffer, float *__restrict__ volumes,
    const int *__restrict__ field_sizes, int num_cells, int max_field_size);

__global__ void kernel_reduce_centroid_sums_batched(
    float **__restrict__ phi_ptrs, float *__restrict__ centroid_sums,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ offsets_x, const int *__restrict__ offsets_y,
    const float *__restrict__ ref_x, const float *__restrict__ ref_y,
    int halo_width, int Nx, int Ny, int num_cells);

__global__ void kernel_compute_centroids_and_deviations(
    float *__restrict__ centroids_x, float *__restrict__ centroids_y,
    float *__restrict__ volume_deviations,
    const float *__restrict__ centroid_sums, const float *__restrict__ volumes,
    const float *__restrict__ ref_x, const float *__restrict__ ref_y,
    float target_area, float dA, int Nx, int Ny, int num_cells);

__global__ void kernel_volume_constraint_batched_v2(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const float *__restrict__ volume_deviations, float volume_coeff,
    int num_cells, int max_field_size);

__global__ void kernel_reduce_integrals_batched_v2(
    const float *__restrict__ work_buffer, float *__restrict__ integrals_x,
    float *__restrict__ integrals_y, const int *__restrict__ field_sizes,
    int num_cells, int max_field_size);

__global__ void kernel_compute_velocities(
    float *__restrict__ velocities_x, float *__restrict__ velocities_y,
    const float *__restrict__ integrals_x,
    const float *__restrict__ integrals_y,
    const float *__restrict__ polarizations_x,
    const float *__restrict__ polarizations_y, float motility_coeff, float dA,
    float v_A, int num_cells);

__global__ void kernel_fused_rhs_step_batched(
    float **phi_ptrs, const float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ work_offsets,
    const float *__restrict__ velocities_x,
    const float *__restrict__ velocities_y, float gamma, float dt,
    int num_cells, int max_field_size);

void step_fused_v4(Domain &domain, float dt, float *d_work_buffer,
                   float **d_all_phi_ptrs, int *d_all_widths,
                   int *d_all_heights, int *d_all_offsets_x,
                   int *d_all_offsets_y, int *d_all_field_sizes,
                   float *d_volumes, float *d_integrals_x, float *d_integrals_y,
                   float *d_centroid_sums, float *d_volume_deviations,
                   float *d_velocities_x, float *d_velocities_y, float *d_ref_x,
                   float *d_ref_y, float *d_polarization_x,
                   float *d_polarization_y, float *d_centroids_x,
                   float *d_centroids_y, int *d_neighbor_counts,
                   int *d_neighbor_lists, bool sync_centroids) {
  const SimParams &params = domain.params;
  int num_cells = domain.num_cells();
  if (num_cells == 0)
    return;

  // Find max dimensions + collect widths/heights for neighbor building
  int max_size = 0, max_w = 0, max_h = 0;
  std::vector<int> h_widths(num_cells), h_heights(num_cells);
  for (int i = 0; i < num_cells; ++i) {
    max_size = std::max(max_size, domain.cells[i]->field_size);
    max_w = std::max(max_w, domain.cells[i]->width());
    max_h = std::max(max_h, domain.cells[i]->height());
    h_widths[i] = domain.cells[i]->width();
    h_heights[i] = domain.cells[i]->height();
  }

  float dA = params.dx * params.dy;
  float target_area = params.target_area();

  // Zero accumulators (async)
  cudaMemsetAsync(d_volumes, 0, num_cells * sizeof(float));
  cudaMemsetAsync(d_integrals_x, 0, num_cells * sizeof(float));
  cudaMemsetAsync(d_integrals_y, 0, num_cells * sizeof(float));
  cudaMemsetAsync(d_centroid_sums, 0, num_cells * 3 * sizeof(float));

  // Compute reference points on GPU from bbox data (eliminates CPU memcpy)
  {
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    kernel_compute_ref_points<<<blocks, threads>>>(
        d_ref_x, d_ref_y, d_all_offsets_x, d_all_offsets_y, d_all_widths,
        d_all_heights, params.Nx, params.Ny, num_cells);
  }

  // Upload polarizations only (still computed on CPU due to random number
  // generation)
  std::vector<float> h_pol_x(num_cells), h_pol_y(num_cells);
  for (int i = 0; i < num_cells; ++i) {
    h_pol_x[i] = domain.cells[i]->polarization.x;
    h_pol_y[i] = domain.cells[i]->polarization.y;
  }
  cudaMemcpyAsync(d_polarization_x, h_pol_x.data(), num_cells * sizeof(float),
                  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_polarization_y, h_pol_y.data(), num_cells * sizeof(float),
                  cudaMemcpyHostToDevice);

  // Common grid config
  dim3 block(16, 16, 1);
  dim3 grid((max_w + 15) / 16, (max_h + 15) / 16, num_cells);

  // =========================================================================
  // PHASE 1: Batched local terms
  // =========================================================================
  kernel_fused_local_batched_v2<<<grid, block>>>(
      d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights, params.dx,
      params.dy, params.bulk_coeff(), params.halo_width, num_cells, max_size);

  // =========================================================================
  // PHASE 2: Batched reductions
  // =========================================================================
  {
    int threads = 256;
    int blocks_per_cell = std::min((max_size + threads - 1) / threads, 32);
    dim3 reduce_grid(blocks_per_cell, num_cells);

    kernel_reduce_volumes_batched_v2<<<reduce_grid, threads,
                                       threads * sizeof(float)>>>(
        d_work_buffer, d_volumes, d_all_field_sizes, num_cells, max_size);

    kernel_reduce_centroid_sums_batched<<<reduce_grid, threads,
                                          3 * threads * sizeof(float)>>>(
        d_all_phi_ptrs, d_centroid_sums, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y, d_ref_x, d_ref_y, params.halo_width,
        params.Nx, params.Ny, num_cells);
  }

  // SYNC: Wait for reductions
  cudaDeviceSynchronize();

  // =========================================================================
  // PHASE 3: GPU-side centroid + volume deviation computation
  // =========================================================================
  int threads_1d = 256;
  int blocks_1d = (num_cells + threads_1d - 1) / threads_1d;

  kernel_compute_centroids_and_deviations<<<blocks_1d, threads_1d>>>(
      d_centroids_x, d_centroids_y, d_volume_deviations, d_centroid_sums,
      d_volumes, d_ref_x, d_ref_y, target_area, dA, params.Nx, params.Ny,
      num_cells);

  // =========================================================================
  // PHASE 4: Batched volume constraint
  // =========================================================================
  kernel_volume_constraint_batched_v2<<<grid, block>>>(
      d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights,
      d_volume_deviations, params.volume_coeff(), num_cells, max_size);

  // =========================================================================
  // PHASE 5: Build neighbor list + Interaction with neighbor list
  // =========================================================================
  if (num_cells > 1) {
    // Build neighbor list using centroid-based distance check
    // Search radius = 4*R is conservative: subdomains extend ~R+padding from
    // centroid, so cells need centroids within ~2*(R+padding) to have
    // overlapping subdomains. 4*R provides safety margin and works for all
    // reasonable padding values.
    float search_radius = 4.0f * params.target_radius;

    int neighbor_threads = std::min(num_cells, 256);
    int neighbor_blocks = (num_cells + neighbor_threads - 1) / neighbor_threads;
    kernel_build_neighbor_list<<<neighbor_blocks, neighbor_threads>>>(
        d_centroids_x, d_centroids_y, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y, d_neighbor_counts, d_neighbor_lists,
        params.Nx, params.Ny, num_cells, search_radius);

    // Interaction with neighbor list (O(k) instead of O(N) per pixel)
    kernel_interaction_neighborlist_v4<<<grid, block>>>(
        d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y, d_neighbor_counts, d_neighbor_lists,
        params.interaction_coeff(), params.halo_width, params.Nx, params.Ny,
        num_cells, max_size);

    // Batched integral reduction
    int threads = 256;
    int blocks_per_cell = std::min((max_size + threads - 1) / threads, 32);
    dim3 reduce_grid(blocks_per_cell, num_cells);
    kernel_reduce_integrals_batched_v2<<<reduce_grid, threads,
                                         2 * threads * sizeof(float)>>>(
        d_work_buffer, d_integrals_x, d_integrals_y, d_all_field_sizes,
        num_cells, max_size);

    // GPU-side velocity computation
    kernel_compute_velocities<<<blocks_1d, threads_1d>>>(
        d_velocities_x, d_velocities_y, d_integrals_x, d_integrals_y,
        d_polarization_x, d_polarization_y, params.motility_coeff(), dA,
        params.v_A, num_cells);
  } else {
    cudaMemsetAsync(d_work_buffer + 8 * max_size, 0, max_size * sizeof(float));
    kernel_compute_velocities<<<1, 1>>>(
        d_velocities_x, d_velocities_y, d_integrals_x, d_integrals_y,
        d_polarization_x, d_polarization_y, 0.0f, dA, params.v_A, 1);
  }

  // =========================================================================
  // PHASE 6: Batched RHS + Euler step
  // =========================================================================
  kernel_fused_rhs_step_batched<<<grid, block>>>(
      d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights,
      d_all_offsets_x, d_velocities_x, d_velocities_y, params.gamma, dt,
      num_cells, max_size);

  // =========================================================================
  // FINAL SYNC
  // =========================================================================
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return;
  }

  if (sync_centroids) {
    std::vector<float> h_centroids_x(num_cells), h_centroids_y(num_cells);
    std::vector<float> h_volumes(num_cells);
    std::vector<float> h_vx(num_cells), h_vy(num_cells);

    cudaMemcpy(h_centroids_x.data(), d_centroids_x, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids_y.data(), d_centroids_y, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_volumes.data(), d_volumes, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vx.data(), d_velocities_x, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vy.data(), d_velocities_y, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_cells; ++i) {
      domain.cells[i]->centroid.x = h_centroids_x[i];
      domain.cells[i]->centroid.y = h_centroids_y[i];
      domain.cells[i]->volume = h_volumes[i] * dA;
      domain.cells[i]->volume_deviation = target_area - domain.cells[i]->volume;
      domain.cells[i]->velocity.x = h_vx[i];
      domain.cells[i]->velocity.y = h_vy[i];
    }
  }
}

} // namespace cellsim
