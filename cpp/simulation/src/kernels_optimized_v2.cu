#include "kernels.cuh"
#include <cstdio>

namespace cellsim {

//=============================================================================
// GPU kernel to compute reference points from bbox data (eliminates CPU memcpy)
// ref = bbox center wrapped to [0, N)
//=============================================================================

__global__ void kernel_compute_ref_points_v2(float *__restrict__ ref_x,
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
// OPTIMIZED V2 KERNEL: Volume constraint with device-side volume deviation
// Reads volume_deviation from device array instead of per-cell host value
//=============================================================================

__global__ void kernel_volume_constraint_v2(
    const float *__restrict__ phi, float *__restrict__ constraint_term,
    int width, int height,
    const float *__restrict__ volume_deviations, // Device array
    int cell_idx, float volume_coeff) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;

  if (lx >= width || ly >= height)
    return;

  int idx = ly * width + lx;
  float phi_val = phi[idx];

  // Read volume deviation from device array
  float volume_deviation = volume_deviations[cell_idx];

  // δV/δφ = -2 * k_V * (A_target - A) * φ = 2 * k_V * volume_deviation * φ
  constraint_term[idx] = 2.0f * volume_coeff * volume_deviation * phi_val;
}

//=============================================================================
// BATCHED: Compute volume constraint for all cells from device-side volumes
//=============================================================================

__global__ void kernel_volume_constraint_batched(
    float **phi_ptrs,
    float *__restrict__ constraint_batched, // Output: contiguous
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ work_offsets,
    const float *__restrict__ volumes, // Device array of current volumes
    float target_area, float volume_coeff, int num_cells) {
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
  int output_offset = work_offsets[cell_idx];

  float phi_val = phi_ptrs[cell_idx][idx];

  // Compute volume deviation on GPU
  float volume_deviation = target_area - volumes[cell_idx];

  // δV/δφ = 2 * k_V * volume_deviation * φ
  constraint_batched[output_offset + idx] =
      2.0f * volume_coeff * volume_deviation * phi_val;
}

//=============================================================================
// FUSED V2: Combine RHS + Euler Step with batched work arrays
// Reads from work buffer offsets instead of separate arrays
//=============================================================================

__global__ void kernel_fused_rhs_step_batched(
    float **phi_ptrs,
    const float *__restrict__ work_buffer, // All work arrays contiguous
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ work_offsets,
    const float *__restrict__ velocities_x, // Device array
    const float *__restrict__ velocities_y, // Device array
    float gamma, float dt, int num_cells,
    int max_field_size) // Stride between work arrays
{
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

  // Work buffer layout per cell (9 buffers):
  // [0] laplacian, [1] bulk, [2] constraint, [3] grad_x, [4] grad_y,
  // [5] phi_sq, [6] repulsion, [7] integrand_x, [8] integrand_y
  int base = cell_idx * 9 * max_field_size;
  const float *d_laplacian = work_buffer + base;
  const float *d_bulk = work_buffer + base + max_field_size;
  const float *d_constraint = work_buffer + base + 2 * max_field_size;
  const float *d_grad_x = work_buffer + base + 3 * max_field_size;
  const float *d_grad_y = work_buffer + base + 4 * max_field_size;
  const float *d_repulsion = work_buffer + base + 6 * max_field_size;

  float vx = velocities_x[cell_idx];
  float vy = velocities_y[cell_idx];

  // Advection: v · ∇φ
  float advection = vx * d_grad_x[idx] + vy * d_grad_y[idx];

  // Variational derivative: δF/δφ = -2γ∇²φ + f'(φ) + constraint + repulsion
  float var_deriv = -2.0f * gamma * d_laplacian[idx] + d_bulk[idx] +
                    d_constraint[idx] + d_repulsion[idx];

  // dφ/dt = -v·∇φ - 0.5 * δF/δφ
  float dphi_dt = -advection - 0.5f * var_deriv;

  // Euler step with clamping
  float *phi = phi_ptrs[cell_idx];
  float new_phi = phi[idx] + dt * dphi_dt;
  phi[idx] = fmaxf(0.0f, fminf(1.0f, new_phi));
}

//=============================================================================
// GPU-SIDE VELOCITY COMPUTATION: Eliminates host readback for velocities
// Kernel computes velocity from reduced integrals + polarization
//=============================================================================

__global__ void kernel_compute_velocities(
    float *__restrict__ velocities_x, float *__restrict__ velocities_y,
    const float *__restrict__ integrals_x,
    const float *__restrict__ integrals_y,
    const float *__restrict__ polarizations_x,
    const float *__restrict__ polarizations_y, float motility_coeff, float dA,
    float v_A, int num_cells) {
  int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_idx >= num_cells)
    return;

  // velocity = motility_coeff * integral * dA + v_A * polarization
  velocities_x[cell_idx] = motility_coeff * integrals_x[cell_idx] * dA +
                           v_A * polarizations_x[cell_idx];
  velocities_y[cell_idx] = motility_coeff * integrals_y[cell_idx] * dA +
                           v_A * polarizations_y[cell_idx];
}

//=============================================================================
// GPU-SIDE CENTROID + VOLUME DEVIATION: Compute from reduction results
//=============================================================================

__global__ void kernel_compute_centroids_and_deviations(
    float *__restrict__ centroids_x, float *__restrict__ centroids_y,
    float *__restrict__ volume_deviations,
    const float *__restrict__ centroid_sums, // [dx_phi2, dy_phi2, phi2] * N
    const float *__restrict__ volumes, const float *__restrict__ ref_x,
    const float *__restrict__ ref_y, float target_area, float dA, int Nx,
    int Ny, int num_cells) {
  int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_idx >= num_cells)
    return;

  // Compute volume deviation
  float volume = volumes[cell_idx] * dA;
  volume_deviations[cell_idx] = target_area - volume;

  // Compute centroid
  float sum_dx = centroid_sums[cell_idx * 3 + 0];
  float sum_dy = centroid_sums[cell_idx * 3 + 1];
  float sum_phi2 = centroid_sums[cell_idx * 3 + 2];

  if (sum_phi2 > 1e-8f) {
    float cx = ref_x[cell_idx] + sum_dx / sum_phi2;
    float cy = ref_y[cell_idx] + sum_dy / sum_phi2;

    // Wrap to domain
    cx = fmodf(fmodf(cx, (float)Nx) + (float)Nx, (float)Nx);
    cy = fmodf(fmodf(cy, (float)Ny) + (float)Ny, (float)Ny);

    centroids_x[cell_idx] = cx;
    centroids_y[cell_idx] = cy;
  }
}

//=============================================================================
// OPTIMIZED step_fused_v2: BATCHED KERNELS + Minimal GPU-CPU syncs
//
// Key optimizations:
// 1. ALL per-cell kernels batched into single launches (Z-dimension)
// 2. Volume deviation computed on GPU (kernel_compute_centroids_and_deviations)
// 3. Velocities computed on GPU (kernel_compute_velocities)
// 4. Centroids only synced to host when sync_centroids=true (for bbox updates)
// 5. Only ONE cudaDeviceSynchronize per timestep (at end)
// 6. ~8 kernel launches per step instead of ~504
//=============================================================================

// Batched local terms kernel (from v3) - WITHOUT centroid atomics
__global__ void kernel_fused_local_batched_v2(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights, float dx,
    float dy, float bulk_coeff, int halo_width, int num_cells,
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
  const float *phi = phi_ptrs[cell_idx];

  int base = cell_idx * 9 * max_field_size;
  float *d_laplacian = work_buffer + base;
  float *d_bulk = work_buffer + base + max_field_size;
  float *d_grad_x = work_buffer + base + 3 * max_field_size;
  float *d_grad_y = work_buffer + base + 4 * max_field_size;
  float *d_phi_sq = work_buffer + base + 5 * max_field_size;

  float inv_dx2 = 1.0f / (dx * dx);
  float inv_dy2 = 1.0f / (dy * dy);
  float inv_2dx = 0.5f / dx;
  float inv_2dy = 0.5f / dy;

  int xm = (lx > 0) ? lx - 1 : 0;
  int xp = (lx < width - 1) ? lx + 1 : width - 1;
  int ym = (ly > 0) ? ly - 1 : 0;
  int yp = (ly < height - 1) ? ly + 1 : height - 1;

  float phi_c = phi[idx];
  float phi_xm = phi[ly * width + xm];
  float phi_xp = phi[ly * width + xp];
  float phi_ym = phi[ym * width + lx];
  float phi_yp = phi[yp * width + lx];

  // Laplacian
  d_laplacian[idx] = (phi_xp - 2.0f * phi_c + phi_xm) * inv_dx2 +
                     (phi_yp - 2.0f * phi_c + phi_ym) * inv_dy2;

  // Bulk
  d_bulk[idx] =
      2.0f * bulk_coeff * phi_c * (1.0f - phi_c) * (1.0f - 2.0f * phi_c);

  // Gradients
  d_grad_x[idx] = (phi_xp - phi_xm) * inv_2dx;
  d_grad_y[idx] = (phi_yp - phi_ym) * inv_2dy;

  // φ² for volume (only in non-halo region)
  float phi_sq = 0.0f;
  if (lx >= halo_width && lx < width - halo_width && ly >= halo_width &&
      ly < height - halo_width) {
    phi_sq = phi_c * phi_c;
  }
  d_phi_sq[idx] = phi_sq;
}

// Batched volume reduction
__global__ void kernel_reduce_volumes_batched_v2(
    const float *__restrict__ work_buffer, float *__restrict__ volumes,
    const int *__restrict__ field_sizes, int num_cells, int max_field_size) {
  extern __shared__ float sdata[];

  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  int field_size = field_sizes[cell_idx];
  const float *d_phi_sq =
      work_buffer + cell_idx * 9 * max_field_size + 5 * max_field_size;

  int tid = threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  int global_idx = blockIdx.x * blockDim.x + tid;

  float sum = 0.0f;
  for (int i = global_idx; i < field_size; i += grid_stride) {
    sum += d_phi_sq[i];
  }

  sdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(&volumes[cell_idx], sdata[0]);
}

// Batched centroid sums reduction (block-level reduction, not pixel atomics)
__global__ void kernel_reduce_centroid_sums_batched(
    float **__restrict__ phi_ptrs,
    float *__restrict__ centroid_sums, // [sum_dx, sum_dy, sum_phi2] * N
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ offsets_x, const int *__restrict__ offsets_y,
    const float *__restrict__ ref_x, const float *__restrict__ ref_y,
    int halo_width, int Nx, int Ny, int num_cells) {
  extern __shared__ float sdata[];
  float *sdata_dx = sdata;
  float *sdata_dy = sdata + blockDim.x;
  float *sdata_phi2 = sdata + 2 * blockDim.x;

  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  int width = widths[cell_idx];
  int height = heights[cell_idx];
  int inner_w = width - 2 * halo_width;
  int inner_h = height - 2 * halo_width;
  int inner_size = inner_w * inner_h;

  const float *phi = phi_ptrs[cell_idx];
  float ref_xi = ref_x[cell_idx];
  float ref_yi = ref_y[cell_idx];
  int offset_x = offsets_x[cell_idx];
  int offset_y = offsets_y[cell_idx];

  int tid = threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  int global_idx = blockIdx.x * blockDim.x + tid;

  float sum_dx = 0.0f, sum_dy = 0.0f, sum_phi2 = 0.0f;

  for (int i = global_idx; i < inner_size; i += grid_stride) {
    int inner_lx = i % inner_w;
    int inner_ly = i / inner_w;
    int lx = inner_lx + halo_width;
    int ly = inner_ly + halo_width;
    int idx = ly * width + lx;

    float phi_val = phi[idx];
    float phi_sq = phi_val * phi_val;

    float gx = (float)(offset_x + lx);
    float gy = (float)(offset_y + ly);

    float dx_from_ref = gx - ref_xi;
    float dy_from_ref = gy - ref_yi;

    if (dx_from_ref > Nx * 0.5f)
      dx_from_ref -= Nx;
    if (dx_from_ref < -Nx * 0.5f)
      dx_from_ref += Nx;
    if (dy_from_ref > Ny * 0.5f)
      dy_from_ref -= Ny;
    if (dy_from_ref < -Ny * 0.5f)
      dy_from_ref += Ny;

    sum_dx += dx_from_ref * phi_sq;
    sum_dy += dy_from_ref * phi_sq;
    sum_phi2 += phi_sq;
  }

  sdata_dx[tid] = sum_dx;
  sdata_dy[tid] = sum_dy;
  sdata_phi2[tid] = sum_phi2;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata_dx[tid] += sdata_dx[tid + s];
      sdata_dy[tid] += sdata_dy[tid + s];
      sdata_phi2[tid] += sdata_phi2[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&centroid_sums[cell_idx * 3 + 0], sdata_dx[0]);
    atomicAdd(&centroid_sums[cell_idx * 3 + 1], sdata_dy[0]);
    atomicAdd(&centroid_sums[cell_idx * 3 + 2], sdata_phi2[0]);
  }
}

// Batched volume constraint
__global__ void kernel_volume_constraint_batched_v2(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const float *__restrict__ volume_deviations, float volume_coeff,
    int num_cells, int max_field_size) {
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
  float *d_constraint = work_buffer + base + 2 * max_field_size;

  float phi_val = phi_ptrs[cell_idx][idx];
  float volume_deviation = volume_deviations[cell_idx];

  // d/dφ[ (μ/πR²)(πR² - ∫φ²)² ] = -2 * (μ/πR²) * (πR² - ∫φ²) * 2φ
  // = -4 * volume_coeff * volume_deviation * φ
  d_constraint[idx] = -4.0f * volume_coeff * volume_deviation * phi_val;
}

// Batched interaction kernel
__global__ void kernel_interaction_batched_v2(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ offsets_x, const int *__restrict__ offsets_y,
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

  // Sum of φ_j² over all other cells (NOT (Σφ_j)²)
  float sum_phi_j_sq = 0.0f;

  for (int j = 0; j < num_cells; ++j) {
    if (j == cell_idx)
      continue;

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

// Batched integral reduction
__global__ void kernel_reduce_integrals_batched_v2(
    const float *__restrict__ work_buffer, float *__restrict__ integrals_x,
    float *__restrict__ integrals_y, const int *__restrict__ field_sizes,
    int num_cells, int max_field_size) {
  extern __shared__ float sdata[];
  float *sdata_x = sdata;
  float *sdata_y = sdata + blockDim.x;

  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  int field_size = field_sizes[cell_idx];
  int base = cell_idx * 9 * max_field_size;
  const float *d_integrand_x = work_buffer + base + 7 * max_field_size;
  const float *d_integrand_y = work_buffer + base + 8 * max_field_size;

  int tid = threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  int global_idx = blockIdx.x * blockDim.x + tid;

  float sum_x = 0.0f, sum_y = 0.0f;
  for (int i = global_idx; i < field_size; i += grid_stride) {
    sum_x += d_integrand_x[i];
    sum_y += d_integrand_y[i];
  }

  sdata_x[tid] = sum_x;
  sdata_y[tid] = sum_y;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata_x[tid] += sdata_x[tid + s];
      sdata_y[tid] += sdata_y[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&integrals_x[cell_idx], sdata_x[0]);
    atomicAdd(&integrals_y[cell_idx], sdata_y[0]);
  }
}

void step_fused_v2(Domain &domain, float dt, float *d_work_buffer,
                   float **d_all_phi_ptrs, int *d_all_widths,
                   int *d_all_heights, int *d_all_offsets_x,
                   int *d_all_offsets_y, int *d_all_field_sizes,
                   float *d_volumes, float *d_integrals_x, float *d_integrals_y,
                   float *d_centroid_sums, float *d_volume_deviations,
                   float *d_velocities_x, float *d_velocities_y, float *d_ref_x,
                   float *d_ref_y, float *d_polarization_x,
                   float *d_polarization_y, float *d_centroids_x,
                   float *d_centroids_y, bool sync_centroids) {
  const SimParams &params = domain.params;
  int num_cells = domain.num_cells();
  if (num_cells == 0)
    return;

  // Find max dimensions
  int max_size = 0, max_w = 0, max_h = 0;
  for (const auto &cell : domain.cells) {
    max_size = std::max(max_size, cell->field_size);
    max_w = std::max(max_w, cell->width());
    max_h = std::max(max_h, cell->height());
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
    kernel_compute_ref_points_v2<<<blocks, threads>>>(
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

  // Common grid config for batched kernels
  dim3 block(16, 16, 1);
  dim3 grid((max_w + 15) / 16, (max_h + 15) / 16, num_cells);

  // =========================================================================
  // PHASE 1: Batched local terms (ALL cells in ONE launch)
  // =========================================================================
  kernel_fused_local_batched_v2<<<grid, block>>>(
      d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights, params.dx,
      params.dy, params.bulk_coeff(), params.halo_width, num_cells, max_size);

  // =========================================================================
  // PHASE 2: Batched reductions (volume + centroid sums)
  // =========================================================================
  {
    int threads = 256;
    int blocks_per_cell = std::min((max_size + threads - 1) / threads, 32);
    dim3 reduce_grid(blocks_per_cell, num_cells);

    // Volume reduction
    kernel_reduce_volumes_batched_v2<<<reduce_grid, threads,
                                       threads * sizeof(float)>>>(
        d_work_buffer, d_volumes, d_all_field_sizes, num_cells, max_size);

    // Centroid sums reduction (block-level, not pixel atomics)
    kernel_reduce_centroid_sums_batched<<<reduce_grid, threads,
                                          3 * threads * sizeof(float)>>>(
        d_all_phi_ptrs, d_centroid_sums, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y, d_ref_x, d_ref_y, params.halo_width,
        params.Nx, params.Ny, num_cells);
  }

  // *** SYNC: Wait for reductions to complete before computing deviations ***
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
  // PHASE 5: Batched interaction (if multiple cells)
  // =========================================================================
  if (num_cells > 1) {
    kernel_interaction_batched_v2<<<grid, block>>>(
        d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y, params.interaction_coeff(),
        params.halo_width, params.Nx, params.Ny, num_cells, max_size);

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
    // Single cell: just self-propulsion
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
  // SINGLE SYNC: Only at end (or when centroids needed)
  // =========================================================================
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return;
  }

  if (sync_centroids) {
    // Read centroids and volumes back to host for Cell structs
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
