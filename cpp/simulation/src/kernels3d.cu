#include "cell3d.cuh"
#include "domain3d.cuh"
#include "kernels3d.cuh"
#include "physics.cuh"
#include "types3d.cuh"
#include <cstdio>
#include <vector>
#include <curand_kernel.h>

namespace cellsim {

//=============================================================================
// Laplacian Kernel - 7-point stencil with Neumann BC
//=============================================================================

__global__ void kernel_laplacian_3d(const float *__restrict__ phi,
                                    float *__restrict__ laplacian, int width,
                                    int height, int depth, float dx, float dy,
                                    float dz) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int wh = width * height;
  int idx = lz * wh + ly * width + lx;

  float inv_dx2 = 1.0f / (dx * dx);
  float inv_dy2 = 1.0f / (dy * dy);
  float inv_dz2 = 1.0f / (dz * dz);

  laplacian[idx] = laplacian_7pt(phi, idx, width, height, depth, lx, ly, lz,
                                 inv_dx2, inv_dy2, inv_dz2);
}

//=============================================================================
// Bulk Potential Kernel
//=============================================================================

__global__ void kernel_bulk_potential_3d(const float *__restrict__ phi,
                                         float *__restrict__ bulk_term,
                                         int width, int height, int depth,
                                         float bulk_coeff) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;
  bulk_term[idx] = compute_bulk_term(phi[idx], bulk_coeff);
}

//=============================================================================
// Phi Squared Kernel (for volume integral)
//=============================================================================

__global__ void kernel_phi_squared_3d(const float *__restrict__ phi,
                                      float *__restrict__ phi_sq, int width,
                                      int height, int depth, int halo) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;

  // Zero out halo regions
  if (lx < halo || lx >= width - halo || ly < halo || ly >= height - halo ||
      lz < halo || lz >= depth - halo) {
    phi_sq[idx] = 0.0f;
  } else {
    float p = phi[idx];
    phi_sq[idx] = p * p;
  }
}

//=============================================================================
// Volume Constraint Kernel
//=============================================================================

__global__ void kernel_volume_constraint_3d(const float *__restrict__ phi,
                                            float *__restrict__ constraint_term,
                                            int width, int height, int depth,
                                            float volume_deviation,
                                            float volume_coeff) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;
  constraint_term[idx] =
      compute_volume_constraint_term(phi[idx], volume_deviation, volume_coeff);
}

//=============================================================================
// Gradient Kernel
//=============================================================================

__global__ void kernel_gradient_3d(const float *__restrict__ phi,
                                   float *__restrict__ grad_x,
                                   float *__restrict__ grad_y,
                                   float *__restrict__ grad_z, int width,
                                   int height, int depth, float dx, float dy,
                                   float dz) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;

  float inv_2dx = 1.0f / (2.0f * dx);
  float inv_2dy = 1.0f / (2.0f * dy);
  float inv_2dz = 1.0f / (2.0f * dz);

  float gx, gy, gz;
  gradient_3d(phi, idx, width, height, depth, lx, ly, lz, inv_2dx, inv_2dy,
              inv_2dz, gx, gy, gz);

  grad_x[idx] = gx;
  grad_y[idx] = gy;
  grad_z[idx] = gz;
}

//=============================================================================
// Advection Kernel
//=============================================================================

__global__ void kernel_advection_3d(const float *__restrict__ grad_x,
                                    const float *__restrict__ grad_y,
                                    const float *__restrict__ grad_z,
                                    float *__restrict__ advection_term,
                                    int width, int height, int depth, float vx,
                                    float vy, float vz) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;
  advection_term[idx] = compute_advection_term_3d(grad_x[idx], grad_y[idx],
                                                  grad_z[idx], vx, vy, vz);
}

//=============================================================================
// Motility Integrand Kernel
//=============================================================================

__global__ void kernel_motility_integrand_3d(
    const float *__restrict__ phi, const float *__restrict__ grad_x,
    const float *__restrict__ grad_y, const float *__restrict__ grad_z,
    const float *__restrict__ interaction_sum, float *__restrict__ integrand_x,
    float *__restrict__ integrand_y, float *__restrict__ integrand_z, int width,
    int height, int depth, int halo) {

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;

  // Zero out halo regions
  if (lx < halo || lx >= width - halo || ly < halo || ly >= height - halo ||
      lz < halo || lz >= depth - halo) {
    integrand_x[idx] = 0.0f;
    integrand_y[idx] = 0.0f;
    integrand_z[idx] = 0.0f;
  } else {
    float ix, iy, iz;
    motility_integrand_3d(phi[idx], grad_x[idx], grad_y[idx], grad_z[idx],
                          interaction_sum[idx], ix, iy, iz);
    integrand_x[idx] = ix;
    integrand_y[idx] = iy;
    integrand_z[idx] = iz;
  }
}

//=============================================================================
// Interaction Sum Kernel (cell-cell repulsion)
//=============================================================================

__global__ void kernel_interaction_sum_3d(
    const float *__restrict__ phi_i, float *__restrict__ interaction_sum,
    int width_i, int height_i, int depth_i, int offset_x_i, int offset_y_i,
    int offset_z_i, float **other_phi_ptrs, int *other_widths,
    int *other_heights, int *other_depths, int *other_offsets_x,
    int *other_offsets_y, int *other_offsets_z, int num_other_cells, int Nx,
    int Ny, int Nz) {

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width_i || ly >= height_i || lz >= depth_i)
    return;

  // Global coordinates of this point
  int gx = (offset_x_i + lx) % Nx;
  if (gx < 0)
    gx += Nx;
  int gy = (offset_y_i + ly) % Ny;
  if (gy < 0)
    gy += Ny;
  int gz = (offset_z_i + lz) % Nz;
  if (gz < 0)
    gz += Nz;

  float sum = 0.0f;

  // Sum φ_j² from all other cells
  for (int j = 0; j < num_other_cells; ++j) {
    int ox = other_offsets_x[j];
    int oy = other_offsets_y[j];
    int oz = other_offsets_z[j];
    int ow = other_widths[j];
    int oh = other_heights[j];
    int od = other_depths[j];

    // Check if point is in cell j's subdomain
    int jlx = gx - ox;
    int jly = gy - oy;
    int jlz = gz - oz;

    // Handle periodic wrapping
    if (jlx < 0)
      jlx += Nx;
    if (jlx >= Nx)
      jlx -= Nx;
    if (jly < 0)
      jly += Ny;
    if (jly >= Ny)
      jly -= Ny;
    if (jlz < 0)
      jlz += Nz;
    if (jlz >= Nz)
      jlz -= Nz;

    if (jlx >= 0 && jlx < ow && jly >= 0 && jly < oh && jlz >= 0 && jlz < od) {
      float phi_j = other_phi_ptrs[j][jlz * (ow * oh) + jly * ow + jlx];
      sum += phi_j * phi_j;
    }
  }

  int idx = lz * (width_i * height_i) + ly * width_i + lx;
  interaction_sum[idx] = sum;
}

//=============================================================================
// Repulsion Kernel
//=============================================================================

__global__ void kernel_repulsion_3d(const float *__restrict__ phi,
                                    const float *__restrict__ interaction_sum,
                                    float *__restrict__ repulsion_term,
                                    int width, int height, int depth,
                                    float interaction_coeff) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;
  repulsion_term[idx] =
      compute_repulsion_term(phi[idx], interaction_sum[idx], interaction_coeff);
}

//=============================================================================
// Combined RHS Kernel
//=============================================================================

__global__ void kernel_combine_rhs_3d(float *__restrict__ dphi_dt,
                                      const float *__restrict__ laplacian,
                                      const float *__restrict__ bulk_term,
                                      const float *__restrict__ constraint_term,
                                      const float *__restrict__ repulsion_term,
                                      const float *__restrict__ advection_term,
                                      int width, int height, int depth,
                                      float gamma) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;

  dphi_dt[idx] =
      combine_rhs_terms(laplacian[idx], bulk_term[idx], constraint_term[idx],
                        repulsion_term[idx], advection_term[idx], gamma);
}

//=============================================================================
// Forward Euler Step
//=============================================================================

__global__ void kernel_euler_step_3d(float *__restrict__ phi,
                                     const float *__restrict__ dphi_dt,
                                     int size, float dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    phi[idx] += dt * dphi_dt[idx];
  }
}

//=============================================================================
// Parallel Reduction for Sum (same as 2D, dimension-independent)
//=============================================================================

__global__ void kernel_reduce_sum_3d(const float *__restrict__ input,
                                     float *__restrict__ output, int n) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float sum = 0.0f;
  if (i < n)
    sum = input[i];
  if (i + blockDim.x < n)
    sum += input[i + blockDim.x];
  sdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

//=============================================================================
// Fused local terms kernel - computes laplacian, bulk, gradient in one pass
//=============================================================================

__global__ void kernel_fused_local_3d(
    const float *__restrict__ phi, float *__restrict__ laplacian,
    float *__restrict__ bulk_term, float *__restrict__ grad_x,
    float *__restrict__ grad_y, float *__restrict__ grad_z, int width,
    int height, int depth, float dx, float dy, float dz, float bulk_coeff) {

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;

  float inv_dx2 = 1.0f / (dx * dx);
  float inv_dy2 = 1.0f / (dy * dy);
  float inv_dz2 = 1.0f / (dz * dz);
  float inv_2dx = 1.0f / (2.0f * dx);
  float inv_2dy = 1.0f / (2.0f * dy);
  float inv_2dz = 1.0f / (2.0f * dz);

  // Compute laplacian
  laplacian[idx] = laplacian_7pt(phi, idx, width, height, depth, lx, ly, lz,
                                 inv_dx2, inv_dy2, inv_dz2);

  // Compute bulk term
  bulk_term[idx] = compute_bulk_term(phi[idx], bulk_coeff);

  // Compute gradient
  float gx, gy, gz;
  gradient_3d(phi, idx, width, height, depth, lx, ly, lz, inv_2dx, inv_2dy,
              inv_2dz, gx, gy, gz);
  grad_x[idx] = gx;
  grad_y[idx] = gy;
  grad_z[idx] = gz;
}

//=============================================================================
// Fused local terms kernel with periodic BC support
// Use this for cells whose subdomain wraps around domain boundaries
//=============================================================================

__global__ void kernel_fused_local_3d_periodic(
    const float *__restrict__ phi, float *__restrict__ laplacian,
    float *__restrict__ bulk_term, float *__restrict__ grad_x,
    float *__restrict__ grad_y, float *__restrict__ grad_z, int width,
    int height, int depth, float dx, float dy, float dz, float bulk_coeff,
    bool wrap_x, bool wrap_y, bool wrap_z) {

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;

  float inv_dx2 = 1.0f / (dx * dx);
  float inv_dy2 = 1.0f / (dy * dy);
  float inv_dz2 = 1.0f / (dz * dz);
  float inv_2dx = 1.0f / (2.0f * dx);
  float inv_2dy = 1.0f / (2.0f * dy);
  float inv_2dz = 1.0f / (2.0f * dz);

  // Compute laplacian with periodic BC
  laplacian[idx] =
      laplacian_7pt_periodic(phi, idx, width, height, depth, lx, ly, lz,
                             inv_dx2, inv_dy2, inv_dz2, wrap_x, wrap_y, wrap_z);

  // Compute bulk term (no stencil)
  bulk_term[idx] = compute_bulk_term(phi[idx], bulk_coeff);

  // Compute gradient with periodic BC
  float gx, gy, gz;
  gradient_3d_periodic(phi, idx, width, height, depth, lx, ly, lz, inv_2dx,
                       inv_2dy, inv_2dz, gx, gy, gz, wrap_x, wrap_y, wrap_z);
  grad_x[idx] = gx;
  grad_y[idx] = gy;
  grad_z[idx] = gz;
}

//=============================================================================
// Host function: Compute volume integral using reduction
//=============================================================================

float compute_volume_integral_3d(const float *d_phi, float *d_work,
                                 int field_size, int halo, int width,
                                 int height, int depth) {
  // First kernel: compute φ²
  KernelConfig3D cfg = KernelConfig3D::for_dims(width, height, depth);
  kernel_phi_squared_3d<<<cfg.grid, cfg.block>>>(d_phi, d_work, width, height,
                                                 depth, halo);

  // Reduction
  int n = field_size;
  float *d_in = d_work;
  float *d_out = d_work + field_size;

  int threads = 256;
  while (n > 1) {
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    kernel_reduce_sum_3d<<<blocks, threads, threads * sizeof(float)>>>(
        d_in, d_out, n);
    n = blocks;
    float *temp = d_in;
    d_in = d_out;
    d_out = temp;
  }

  float result;
  cudaMemcpy(&result, d_in, sizeof(float), cudaMemcpyDeviceToHost);
  return result;
}

//=============================================================================
// Super-fused kernel: computes laplacian + bulk + constraint + advection
// in a single pass. Eliminates grad_x, grad_y, grad_z intermediate buffers.
// Reduces buffer count from 10 to 7 per cell (30% memory savings).
//=============================================================================

__global__ void kernel_fused_all_local_3d(
    const float *__restrict__ phi,
    float *__restrict__ laplacian,  // Output: laplacian term
    float *__restrict__ bulk,       // Output: bulk term
    float *__restrict__ constraint, // Output: constraint term
    float *__restrict__ advection,  // Output: advection term
    int width, int height, int depth, float dx, float dy, float dz,
    float bulk_coeff, float volume_coeff, float volume_deviation, float vx,
    float vy, float vz) {

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;

  float inv_dx2 = 1.0f / (dx * dx);
  float inv_dy2 = 1.0f / (dy * dy);
  float inv_dz2 = 1.0f / (dz * dz);
  float inv_2dx = 1.0f / (2.0f * dx);
  float inv_2dy = 1.0f / (2.0f * dy);
  float inv_2dz = 1.0f / (2.0f * dz);

  float p = phi[idx];

  // Laplacian (7-point stencil)
  laplacian[idx] = laplacian_7pt(phi, idx, width, height, depth, lx, ly, lz,
                                 inv_dx2, inv_dy2, inv_dz2);

  // Bulk term
  bulk[idx] = compute_bulk_term(p, bulk_coeff);

  // Constraint term
  constraint[idx] =
      compute_volume_constraint_term(p, volume_deviation, volume_coeff);

  // Gradient + advection (fused - no intermediate gradient storage)
  float gx, gy, gz;
  gradient_3d(phi, idx, width, height, depth, lx, ly, lz, inv_2dx, inv_2dy,
              inv_2dz, gx, gy, gz);
  advection[idx] = -(vx * gx + vy * gy + vz * gz);
}

// Periodic version
__global__ void kernel_fused_all_local_3d_periodic(
    const float *__restrict__ phi, float *__restrict__ laplacian,
    float *__restrict__ bulk, float *__restrict__ constraint,
    float *__restrict__ advection, int width, int height, int depth, float dx,
    float dy, float dz, float bulk_coeff, float volume_coeff,
    float volume_deviation, float vx, float vy, float vz, bool wrap_x,
    bool wrap_y, bool wrap_z) {

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;

  if (lx >= width || ly >= height || lz >= depth)
    return;

  int idx = lz * (width * height) + ly * width + lx;

  float inv_dx2 = 1.0f / (dx * dx);
  float inv_dy2 = 1.0f / (dy * dy);
  float inv_dz2 = 1.0f / (dz * dz);
  float inv_2dx = 1.0f / (2.0f * dx);
  float inv_2dy = 1.0f / (2.0f * dy);
  float inv_2dz = 1.0f / (2.0f * dz);

  float p = phi[idx];

  // Laplacian with periodic BC
  laplacian[idx] =
      laplacian_7pt_periodic(phi, idx, width, height, depth, lx, ly, lz,
                             inv_dx2, inv_dy2, inv_dz2, wrap_x, wrap_y, wrap_z);

  // Bulk term
  bulk[idx] = compute_bulk_term(p, bulk_coeff);

  // Constraint term
  constraint[idx] =
      compute_volume_constraint_term(p, volume_deviation, volume_coeff);

  // Gradient + advection (fused)
  float gx, gy, gz;
  gradient_3d_periodic(phi, idx, width, height, depth, lx, ly, lz, inv_2dx,
                       inv_2dy, inv_2dz, gx, gy, gz, wrap_x, wrap_y, wrap_z);
  advection[idx] = -(vx * gx + vy * gy + vz * gz);
}

//=============================================================================
// Host function: Compute all local terms for a single 3D cell
// Uses super-fused kernel: 5 buffers (optimized from 7)
// Buffer layout: [laplacian][bulk][constraint][advection][repulsion]
// NOTE: This is the LEGACY per-cell function. The fused batched path is preferred.
//=============================================================================

void compute_local_terms_3d(Cell3D &cell, const SimParams3D &params,
                            float *d_work_buffer, int buffer_stride) {
  int w = cell.width();
  int h = cell.height();
  int d = cell.depth();
  int size = cell.field_size;

  KernelConfig3D cfg = KernelConfig3D::for_cell(cell);

  // Partition work buffer (5 buffers)
  float *d_laplacian = d_work_buffer;
  float *d_bulk = d_work_buffer + buffer_stride;
  float *d_constraint = d_work_buffer + 2 * buffer_stride;
  float *d_advection = d_work_buffer + 3 * buffer_stride;
  // Buffer 4 is for repulsion (used later)
  
  // For volume computation, use shared memory reduction (no dedicated buffer needed)
  // Allocate temporary buffer for reduction
  float *d_reduction_temp;
  cudaMalloc(&d_reduction_temp, size * sizeof(float));

  // Compute volume integral FIRST (needed for constraint term)
  float volume = compute_volume_integral_3d(cell.d_phi, d_reduction_temp, size,
                                            params.halo_width, w, h, d);
  cell.volume = volume * params.dx * params.dy * params.dz;
  cell.volume_deviation = params.target_volume() - cell.volume;
  
  cudaFree(d_reduction_temp);

  // Check if cell wraps around domain boundaries
  bool wrap_x = cell.wraps_x(params.Nx);
  bool wrap_y = cell.wraps_y(params.Ny);
  bool wrap_z = cell.wraps_z(params.Nz);

  // Super-fused kernel: laplacian + bulk + constraint + advection in ONE pass
  // Eliminates grad_x, grad_y, grad_z intermediate buffers
  if (wrap_x || wrap_y || wrap_z) {
    kernel_fused_all_local_3d_periodic<<<cfg.grid, cfg.block>>>(
        cell.d_phi, d_laplacian, d_bulk, d_constraint, d_advection, w, h, d,
        params.dx, params.dy, params.dz, params.bulk_coeff(),
        params.volume_coeff(), cell.volume_deviation, cell.velocity.x,
        cell.velocity.y, cell.velocity.z, wrap_x, wrap_y, wrap_z);
  } else {
    kernel_fused_all_local_3d<<<cfg.grid, cfg.block>>>(
        cell.d_phi, d_laplacian, d_bulk, d_constraint, d_advection, w, h, d,
        params.dx, params.dy, params.dz, params.bulk_coeff(),
        params.volume_coeff(), cell.volume_deviation, cell.velocity.x,
        cell.velocity.y, cell.velocity.z);
  }

  cudaDeviceSynchronize();
}

//=============================================================================
// Host function: Compute interaction terms for all 3D cells
// Buffer layout (5 per cell): [laplacian][bulk][constraint][advection][repulsion]
// NOTE: This is the LEGACY function. The fused batched path is preferred.
//=============================================================================

void compute_interaction_terms_3d(Domain3D &domain, float *d_work_buffer) {
  int num_cells = domain.num_cells();
  if (num_cells < 2) {
    // Zero out repulsion
    return;
  }

  domain.sync_device_arrays();

  // Find max field size
  int max_size = 0;
  for (const auto &cell : domain.cells) {
    max_size = max(max_size, cell->field_size);
  }

  // Allocate device arrays for other cells' info
  std::vector<float *> phi_ptrs(num_cells);
  std::vector<int> widths(num_cells), heights(num_cells), depths(num_cells);
  std::vector<int> offsets_x(num_cells), offsets_y(num_cells),
      offsets_z(num_cells);

  for (int i = 0; i < num_cells; ++i) {
    phi_ptrs[i] = domain.cells[i]->d_phi;
    widths[i] = domain.cells[i]->width();
    heights[i] = domain.cells[i]->height();
    depths[i] = domain.cells[i]->depth();
    offsets_x[i] = domain.cells[i]->bbox_with_halo.x0;
    offsets_y[i] = domain.cells[i]->bbox_with_halo.y0;
    offsets_z[i] = domain.cells[i]->bbox_with_halo.z0;
  }

  // Upload to device
  float **d_phi_ptrs;
  int *d_widths, *d_heights, *d_depths;
  int *d_offsets_x, *d_offsets_y, *d_offsets_z;

  cudaMalloc(&d_phi_ptrs, num_cells * sizeof(float *));
  cudaMalloc(&d_widths, num_cells * sizeof(int));
  cudaMalloc(&d_heights, num_cells * sizeof(int));
  cudaMalloc(&d_depths, num_cells * sizeof(int));
  cudaMalloc(&d_offsets_x, num_cells * sizeof(int));
  cudaMalloc(&d_offsets_y, num_cells * sizeof(int));
  cudaMalloc(&d_offsets_z, num_cells * sizeof(int));

  cudaMemcpy(d_phi_ptrs, phi_ptrs.data(), num_cells * sizeof(float *),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_widths, widths.data(), num_cells * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_heights, heights.data(), num_cells * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_depths, depths.data(), num_cells * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets_x, offsets_x.data(), num_cells * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets_y, offsets_y.data(), num_cells * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets_z, offsets_z.data(), num_cells * sizeof(int),
             cudaMemcpyHostToDevice);

  // Compute interaction for each cell
  // Buffer layout: [0:laplacian][1:bulk][2:constraint][3:advection][4:repulsion]
  // Need a temp buffer for interaction_sum since we removed it from main layout
  float *d_interaction_temp;
  cudaMalloc(&d_interaction_temp, max_size * sizeof(float));
  
  for (int i = 0; i < num_cells; ++i) {
    Cell3D &cell = *domain.cells[i];
    int w = cell.width(), h = cell.height(), d = cell.depth();

    float *d_repulsion = d_work_buffer + i * 5 * max_size + 4 * max_size;

    KernelConfig3D cfg = KernelConfig3D::for_cell(cell);

    kernel_interaction_sum_3d<<<cfg.grid, cfg.block>>>(
        cell.d_phi, d_interaction_temp, w, h, d, cell.bbox_with_halo.x0,
        cell.bbox_with_halo.y0, cell.bbox_with_halo.z0, d_phi_ptrs, d_widths,
        d_heights, d_depths, d_offsets_x, d_offsets_y, d_offsets_z, num_cells,
        domain.params.Nx, domain.params.Ny, domain.params.Nz);

    kernel_repulsion_3d<<<cfg.grid, cfg.block>>>(
        cell.d_phi, d_interaction_temp, d_repulsion, w, h, d,
        domain.params.interaction_coeff());
  }
  
  cudaFree(d_interaction_temp);

  // Free temporary device arrays
  cudaFree(d_phi_ptrs);
  cudaFree(d_widths);
  cudaFree(d_heights);
  cudaFree(d_depths);
  cudaFree(d_offsets_x);
  cudaFree(d_offsets_y);
  cudaFree(d_offsets_z);

  cudaDeviceSynchronize();
}

//=============================================================================
// Host function: Forward Euler step for all 3D cells
// DEPRECATED: This legacy function is not used. Use step_fused_3d instead.
// Left for reference only - would need updating to work with new buffer layout.
//=============================================================================

#if 0  // DEPRECATED - not used, step_fused_3d is the active path
void step_euler_3d(Domain3D &domain, float dt, float *d_work_buffer) {
  int num_cells = domain.num_cells();

  // Find max field size
  int max_size = 0;
  for (const auto &cell : domain.cells) {
    max_size = max(max_size, cell->field_size);
  }

  // Compute local terms for all cells IN PARALLEL (5 buffers per cell)
  for (int i = 0; i < num_cells; ++i) {
    compute_local_terms_3d(*domain.cells[i], domain.params,
                           d_work_buffer + i * 5 * max_size, max_size);
  }

  // Compute interaction terms
  compute_interaction_terms_3d(domain, d_work_buffer);

  // Combine RHS and do Euler step
  // Buffer layout: [0:laplacian][1:bulk][2:constraint][3:advection][4:repulsion]
  for (int i = 0; i < num_cells; ++i) {
    Cell3D &cell = *domain.cells[i];
    int w = cell.width(), h = cell.height(), d = cell.depth();

    float *d_cell_work = d_work_buffer + i * 5 * max_size;
    float *d_laplacian = d_cell_work;
    float *d_bulk = d_cell_work + max_size;
    float *d_constraint = d_cell_work + 2 * max_size;
    float *d_advection = d_cell_work + 3 * max_size;
    float *d_repulsion = d_cell_work + 4 * max_size;

    KernelConfig3D cfg = KernelConfig3D::for_cell(cell);
    
    // NOTE: This would need a temp buffer for dphi_dt since it was removed from Cell3D
    // kernel_combine_rhs_3d<<<cfg.grid, cfg.block>>>(...);
    // kernel_euler_step_3d<<<blocks, threads>>>(...);
  }

  cudaDeviceSynchronize();
}
#endif

//=============================================================================
// OPTIMIZED BATCHED 3D KERNELS
// These kernels process all cells in a single launch for better GPU utilization
//=============================================================================

//-----------------------------------------------------------------------------
// Compute reference points on GPU from bbox data (eliminates CPU memcpy)
//-----------------------------------------------------------------------------
__global__ void kernel_compute_ref_points_3d(
    float *__restrict__ ref_x, float *__restrict__ ref_y,
    float *__restrict__ ref_z, const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y, const int *__restrict__ offsets_z,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, int Nx, int Ny, int Nz, int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells)
    return;

  // Compute bbox center from offset and dimensions
  float rx = (float)offsets_x[i] + (float)widths[i] * 0.5f;
  float ry = (float)offsets_y[i] + (float)heights[i] * 0.5f;
  float rz = (float)offsets_z[i] + (float)depths[i] * 0.5f;

  // Wrap to [0, N)
  rx = fmodf(fmodf(rx, (float)Nx) + (float)Nx, (float)Nx);
  ry = fmodf(fmodf(ry, (float)Ny) + (float)Ny, (float)Ny);
  rz = fmodf(fmodf(rz, (float)Nz) + (float)Nz, (float)Nz);

  ref_x[i] = rx;
  ref_y[i] = ry;
  ref_z[i] = rz;
}

//-----------------------------------------------------------------------------
// Batched local terms: laplacian + bulk for all cells
// Uses flattened index to parallelize all 3 dimensions
// Buffer layout: [lap][bulk][constraint][advection][repulsion] (5 buffers)
//-----------------------------------------------------------------------------
__global__ void kernel_fused_local_batched_3d(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, const int *__restrict__ field_sizes,
    float dx, float dy, float dz, float bulk_coeff, int num_cells,
    int max_field_size) {
  // blockIdx.y indexes the cell
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int d = depths[cell_idx];
  int field_size = field_sizes[cell_idx];
  int wh = w * h;

  // Flattened thread index - each thread processes one 3D point
  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= field_size)
    return;

  // Convert flat index to 3D coordinates
  int lz = flat_idx / wh;
  int rem = flat_idx % wh;
  int ly = rem / w;
  int lx = rem % w;

  size_t base = (size_t)cell_idx * 5 * max_field_size;

  const float *phi = phi_ptrs[cell_idx];
  float *d_laplacian = work_buffer + base;
  float *d_bulk = work_buffer + base + max_field_size;

  float inv_dx2 = 1.0f / (dx * dx);
  float inv_dy2 = 1.0f / (dy * dy);
  float inv_dz2 = 1.0f / (dz * dz);

  // Compute laplacian (7-point stencil)
  d_laplacian[flat_idx] = laplacian_7pt(phi, flat_idx, w, h, d, lx, ly, lz,
                                        inv_dx2, inv_dy2, inv_dz2);

  // Compute bulk term
  d_bulk[flat_idx] = compute_bulk_term(phi[flat_idx], bulk_coeff);
}

//-----------------------------------------------------------------------------
// Batched volume reduction: reduce φ² over all cells
//-----------------------------------------------------------------------------
__global__ void kernel_reduce_volumes_batched_3d(
    float **__restrict__ phi_ptrs, float *__restrict__ volumes,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, const int *__restrict__ field_sizes,
    int halo, int num_cells) {
  extern __shared__ float sdata[];

  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  int tid = threadIdx.x;
  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int d = depths[cell_idx];
  int field_size = field_sizes[cell_idx];
  int wh = w * h;

  const float *phi = phi_ptrs[cell_idx];

  // Grid-stride loop
  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < field_size;
       i += blockDim.x * gridDim.x) {
    int lz = i / wh;
    int rem = i % wh;
    int ly = rem / w;
    int lx = rem % w;

    // Skip halo
    if (lx >= halo && lx < w - halo && ly >= halo && ly < h - halo &&
        lz >= halo && lz < d - halo) {
      float p = phi[i];
      sum += p * p;
    }
  }

  sdata[tid] = sum;
  __syncthreads();

  // Reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&volumes[cell_idx], sdata[0]);
  }
}

//-----------------------------------------------------------------------------
// Batched centroid sum reduction: compute weighted displacement from ref point
//-----------------------------------------------------------------------------
__global__ void kernel_reduce_centroid_sums_batched_3d(
    float **__restrict__ phi_ptrs, float *__restrict__ centroid_sums,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y, const int *__restrict__ offsets_z,
    const int *__restrict__ field_sizes, const float *__restrict__ ref_x,
    const float *__restrict__ ref_y, const float *__restrict__ ref_z, int halo,
    int Nx, int Ny, int Nz, int num_cells) {
  extern __shared__ float sdata[];

  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  float *sdx = sdata;
  float *sdy = sdata + blockDim.x;
  float *sdz = sdata + 2 * blockDim.x;
  float *sw = sdata + 3 * blockDim.x;

  int tid = threadIdx.x;
  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int d = depths[cell_idx];
  int field_size = field_sizes[cell_idx];
  int ox = offsets_x[cell_idx];
  int oy = offsets_y[cell_idx];
  int oz = offsets_z[cell_idx];
  float rx = ref_x[cell_idx];
  float ry = ref_y[cell_idx];
  float rz = ref_z[cell_idx];
  int wh = w * h;

  const float *phi = phi_ptrs[cell_idx];

  float sum_dx = 0.0f, sum_dy = 0.0f, sum_dz = 0.0f, sum_w = 0.0f;

  for (int i = blockIdx.x * blockDim.x + tid; i < field_size;
       i += blockDim.x * gridDim.x) {
    int lz = i / wh;
    int rem = i % wh;
    int ly = rem / w;
    int lx = rem % w;

    if (lx >= halo && lx < w - halo && ly >= halo && ly < h - halo &&
        lz >= halo && lz < d - halo) {
      float p = phi[i];
      float weight = p * p;

      // Global coords
      float gx = (float)((ox + lx) % Nx);
      float gy = (float)((oy + ly) % Ny);
      float gz = (float)((oz + lz) % Nz);

      // Displacement from reference (with periodic wrapping)
      float dx_disp = gx - rx;
      float dy_disp = gy - ry;
      float dz_disp = gz - rz;

      if (dx_disp > Nx * 0.5f)
        dx_disp -= Nx;
      else if (dx_disp < -Nx * 0.5f)
        dx_disp += Nx;
      if (dy_disp > Ny * 0.5f)
        dy_disp -= Ny;
      else if (dy_disp < -Ny * 0.5f)
        dy_disp += Ny;
      if (dz_disp > Nz * 0.5f)
        dz_disp -= Nz;
      else if (dz_disp < -Nz * 0.5f)
        dz_disp += Nz;

      sum_dx += weight * dx_disp;
      sum_dy += weight * dy_disp;
      sum_dz += weight * dz_disp;
      sum_w += weight;
    }
  }

  sdx[tid] = sum_dx;
  sdy[tid] = sum_dy;
  sdz[tid] = sum_dz;
  sw[tid] = sum_w;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdx[tid] += sdx[tid + s];
      sdy[tid] += sdy[tid + s];
      sdz[tid] += sdz[tid + s];
      sw[tid] += sw[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&centroid_sums[cell_idx * 4 + 0], sdx[0]);
    atomicAdd(&centroid_sums[cell_idx * 4 + 1], sdy[0]);
    atomicAdd(&centroid_sums[cell_idx * 4 + 2], sdz[0]);
    atomicAdd(&centroid_sums[cell_idx * 4 + 3], sw[0]);
  }
}

//-----------------------------------------------------------------------------
// Compute centroids and volume deviations from reduction results
//-----------------------------------------------------------------------------
__global__ void kernel_compute_centroids_and_deviations_3d(
    float *__restrict__ centroids_x, float *__restrict__ centroids_y,
    float *__restrict__ centroids_z, float *__restrict__ volume_deviations,
    const float *__restrict__ centroid_sums, const float *__restrict__ volumes,
    const float *__restrict__ ref_x, const float *__restrict__ ref_y,
    const float *__restrict__ ref_z, float target_volume, float dV, int Nx,
    int Ny, int Nz, int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells)
    return;

  float sum_dx = centroid_sums[i * 4 + 0];
  float sum_dy = centroid_sums[i * 4 + 1];
  float sum_dz = centroid_sums[i * 4 + 2];
  float sum_w = centroid_sums[i * 4 + 3];

  float cx, cy, cz;
  if (sum_w > 0.0f) {
    cx = ref_x[i] + sum_dx / sum_w;
    cy = ref_y[i] + sum_dy / sum_w;
    cz = ref_z[i] + sum_dz / sum_w;

    // Wrap to [0, N)
    cx = fmodf(fmodf(cx, (float)Nx) + (float)Nx, (float)Nx);
    cy = fmodf(fmodf(cy, (float)Ny) + (float)Ny, (float)Ny);
    cz = fmodf(fmodf(cz, (float)Nz) + (float)Nz, (float)Nz);
  } else {
    cx = ref_x[i];
    cy = ref_y[i];
    cz = ref_z[i];
  }

  centroids_x[i] = cx;
  centroids_y[i] = cy;
  centroids_z[i] = cz;

  float volume = volumes[i] * dV;
  volume_deviations[i] = target_volume - volume;
}

//-----------------------------------------------------------------------------
// Batched volume constraint kernel (flattened for parallelism)
// Buffer layout: [lap][bulk][constraint][advection][repulsion] (5 buffers)
//-----------------------------------------------------------------------------
__global__ void kernel_volume_constraint_batched_3d(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, const int *__restrict__ field_sizes,
    const float *__restrict__ volume_deviations, float volume_coeff,
    int num_cells, int max_field_size) {
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int d = depths[cell_idx];
  int field_size = field_sizes[cell_idx];
  int wh = w * h;

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= field_size)
    return;

  size_t base = (size_t)cell_idx * 5 * max_field_size;
  float vol_dev = volume_deviations[cell_idx];

  const float *phi = phi_ptrs[cell_idx];
  float *d_constraint = work_buffer + base + 2 * max_field_size;

  d_constraint[flat_idx] =
      compute_volume_constraint_term(phi[flat_idx], vol_dev, volume_coeff);
}

//-----------------------------------------------------------------------------
// Batched advection kernel (flattened for parallelism)
// Buffer layout: [lap][bulk][constraint][advection][repulsion] (5 buffers)
//-----------------------------------------------------------------------------
__global__ void kernel_advection_batched_3d(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, const int *__restrict__ field_sizes,
    const float *__restrict__ velocities_x,
    const float *__restrict__ velocities_y,
    const float *__restrict__ velocities_z, float dx, float dy, float dz,
    int num_cells, int max_field_size) {
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int d = depths[cell_idx];
  int field_size = field_sizes[cell_idx];
  int wh = w * h;

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= field_size)
    return;

  // Convert flat index to 3D coordinates
  int lz = flat_idx / wh;
  int rem = flat_idx % wh;
  int ly = rem / w;
  int lx = rem % w;

  size_t base = (size_t)cell_idx * 5 * max_field_size;

  float vx = velocities_x[cell_idx];
  float vy = velocities_y[cell_idx];
  float vz = velocities_z[cell_idx];

  float inv_2dx = 1.0f / (2.0f * dx);
  float inv_2dy = 1.0f / (2.0f * dy);
  float inv_2dz = 1.0f / (2.0f * dz);

  const float *phi = phi_ptrs[cell_idx];
  float *d_advection = work_buffer + base + 3 * max_field_size;

  // Compute gradient
  float gx, gy, gz;
  gradient_3d(phi, flat_idx, w, h, d, lx, ly, lz, inv_2dx, inv_2dy, inv_2dz, gx,
              gy, gz);

  d_advection[flat_idx] = -(vx * gx + vy * gy + vz * gz);
}

//-----------------------------------------------------------------------------
// GPU kernel to build neighbor list based on centroid distance (3D)
//
// Two cells can only interact if their subdomains overlap. Since subdomains
// extend ~R+padding from the centroid, cells whose centroids are more than
// ~2*(R+padding) apart cannot have overlapping subdomains.
//
// We use 4*R as a conservative search radius - this guarantees we catch all
// potential neighbors while still providing O(k) speedup for large systems.
//-----------------------------------------------------------------------------
__global__ void kernel_build_neighbor_list_3d(
    const float *__restrict__ centroids_x,
    const float *__restrict__ centroids_y,
    const float *__restrict__ centroids_z,
    int *__restrict__ neighbor_counts,
    int *__restrict__ neighbor_lists, // [MAX_NEIGHBORS_3D * num_cells]
    int Nx, int Ny, int Nz, int num_cells,
    float search_radius) // Should be ~4*R to be safe
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells)
    return;

  float cx_i = centroids_x[i];
  float cy_i = centroids_y[i];
  float cz_i = centroids_z[i];

  // Search radius squared for comparison
  float search_r2 = search_radius * search_radius;

  int count = 0;
  int *my_neighbors = neighbor_lists + i * MAX_NEIGHBORS_3D;

  for (int j = 0; j < num_cells; ++j) {
    if (j == i)
      continue;

    float cx_j = centroids_x[j];
    float cy_j = centroids_y[j];
    float cz_j = centroids_z[j];

    // Compute distance with periodic wrapping
    float dx = cx_j - cx_i;
    float dy = cy_j - cy_i;
    float dz = cz_j - cz_i;

    // Periodic boundary: if distance > half domain, wrap
    if (dx > Nx * 0.5f)
      dx -= Nx;
    else if (dx < -Nx * 0.5f)
      dx += Nx;
    if (dy > Ny * 0.5f)
      dy -= Ny;
    else if (dy < -Ny * 0.5f)
      dy += Ny;
    if (dz > Nz * 0.5f)
      dz -= Nz;
    else if (dz < -Nz * 0.5f)
      dz += Nz;

    float dist2 = dx * dx + dy * dy + dz * dz;

    // Include as neighbor if within search radius
    if (dist2 <= search_r2) {
      if (count < MAX_NEIGHBORS_3D) {
        my_neighbors[count] = j;
        count++;
      }
    }
  }

  neighbor_counts[i] = count;
}

//-----------------------------------------------------------------------------
// Neighbor-list version of interaction kernel (3D)
// O(k) instead of O(N) per voxel, where k = number of neighbors
// Buffer layout: [lap][bulk][constraint][advection][repulsion] (5 buffers)
//-----------------------------------------------------------------------------
// Structure to cache neighbor metadata in shared memory
struct NeighborInfo {
  float *phi_ptr;
  int width, height, depth, wh;
  int delta_ox, delta_oy, delta_oz;  // Precomputed offset differences
};

__global__ void kernel_interaction_neighborlist_3d(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, const int *__restrict__ field_sizes,
    const int *__restrict__ offsets_x, const int *__restrict__ offsets_y,
    const int *__restrict__ offsets_z,
    const int *__restrict__ neighbor_counts,
    const int *__restrict__ neighbor_lists,
    float interaction_coeff, int Nx, int Ny, int Nz,
    int num_cells, int max_field_size) {
  
  // Shared memory for neighbor metadata (loaded once by thread 0)
  __shared__ NeighborInfo s_neighbors[MAX_NEIGHBORS_3D];
  __shared__ int s_num_neighbors;
  __shared__ int s_ox_i, s_oy_i, s_oz_i;  // Cell i offsets
  __shared__ int s_w, s_h, s_d, s_wh, s_field_size;
  __shared__ float *s_phi_i;
  __shared__ float *s_repulsion;
  
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  // Thread 0 loads cell i metadata and all neighbor metadata into shared memory
  if (threadIdx.x == 0) {
    s_w = widths[cell_idx];
    s_h = heights[cell_idx];
    s_d = depths[cell_idx];
    s_wh = s_w * s_h;
    s_field_size = field_sizes[cell_idx];
    s_ox_i = offsets_x[cell_idx];
    s_oy_i = offsets_y[cell_idx];
    s_oz_i = offsets_z[cell_idx];
    s_phi_i = phi_ptrs[cell_idx];
    
    size_t base = (size_t)cell_idx * 5 * max_field_size;
    s_repulsion = work_buffer + base + 4 * max_field_size;
    
    s_num_neighbors = neighbor_counts[cell_idx];
    const int *my_neighbors = neighbor_lists + cell_idx * MAX_NEIGHBORS_3D;
    
    // Preload all neighbor metadata
    for (int n = 0; n < s_num_neighbors; ++n) {
      int j = my_neighbors[n];
      s_neighbors[n].phi_ptr = phi_ptrs[j];
      s_neighbors[n].width = widths[j];
      s_neighbors[n].height = heights[j];
      s_neighbors[n].depth = depths[j];
      s_neighbors[n].wh = s_neighbors[n].width * s_neighbors[n].height;
      // Precompute offset differences (cell i offset - cell j offset)
      s_neighbors[n].delta_ox = s_ox_i - offsets_x[j];
      s_neighbors[n].delta_oy = s_oy_i - offsets_y[j];
      s_neighbors[n].delta_oz = s_oz_i - offsets_z[j];
    }
  }
  __syncthreads();

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= s_field_size)
    return;

  // Convert flat index to 3D local coordinates
  int lz = flat_idx / s_wh;
  int rem = flat_idx % s_wh;
  int ly = rem / s_w;
  int lx = rem % s_w;

  // Sum φ_j² over NEIGHBOR cells only (O(k) instead of O(N))
  float sum_phi_j_sq = 0.0f;

  #pragma unroll 4
  for (int n = 0; n < s_num_neighbors; ++n) {
    const NeighborInfo &nb = s_neighbors[n];
    
    // Local coords in cell j = (local_in_i + delta_offset) with periodic wrap
    // delta_ox = ox_i - ox_j, so: ljx = lx + delta_ox (mod Nx), clamped to [0, ow)
    int ljx = lx + nb.delta_ox;
    int ljy = ly + nb.delta_oy;
    int ljz = lz + nb.delta_oz;
    
    // Periodic wrap (optimized: use branch instead of double modulo for common case)
    if (ljx < 0) ljx += Nx; else if (ljx >= Nx) ljx -= Nx;
    if (ljy < 0) ljy += Ny; else if (ljy >= Ny) ljy -= Ny;
    if (ljz < 0) ljz += Nz; else if (ljz >= Nz) ljz -= Nz;

    if (ljx < nb.width && ljy < nb.height && ljz < nb.depth) {
      float phi_j = nb.phi_ptr[ljz * nb.wh + ljy * nb.width + ljx];
      sum_phi_j_sq += phi_j * phi_j;
    }
  }

  // Repulsion: 2 * κ_int * φ_i * Σ φ_j²
  s_repulsion[flat_idx] =
      2.0f * interaction_coeff * s_phi_i[flat_idx] * sum_phi_j_sq;
}

//-----------------------------------------------------------------------------
// Batched interaction kernel (O(N²) version - all pairs, flattened)
// DEPRECATED: Use kernel_interaction_neighborlist_3d instead
// Buffer layout: [lap][bulk][constraint][advection][repulsion] (5 buffers)
//-----------------------------------------------------------------------------
__global__ void kernel_interaction_batched_3d(
    float **__restrict__ phi_ptrs, float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, const int *__restrict__ field_sizes,
    const int *__restrict__ offsets_x, const int *__restrict__ offsets_y,
    const int *__restrict__ offsets_z, float interaction_coeff, int Nx, int Ny,
    int Nz, int num_cells, int max_field_size) {
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int d = depths[cell_idx];
  int field_size = field_sizes[cell_idx];
  int ox_i = offsets_x[cell_idx];
  int oy_i = offsets_y[cell_idx];
  int oz_i = offsets_z[cell_idx];
  int wh = w * h;

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= field_size)
    return;

  // Convert flat index to 3D coordinates
  int lz = flat_idx / wh;
  int rem = flat_idx % wh;
  int ly = rem / w;
  int lx = rem % w;

  size_t base = (size_t)cell_idx * 5 * max_field_size;

  const float *phi_i = phi_ptrs[cell_idx];
  float *d_repulsion = work_buffer + base + 4 * max_field_size;

  // Global coords
  int gx = ((ox_i + lx) % Nx + Nx) % Nx;
  int gy = ((oy_i + ly) % Ny + Ny) % Ny;
  int gz = ((oz_i + lz) % Nz + Nz) % Nz;

  // Sum φ_j² over all other cells
  float sum_phi_j_sq = 0.0f;
  for (int j = 0; j < num_cells; ++j) {
    if (j == cell_idx)
      continue;

    int ow = widths[j];
    int oh = heights[j];
    int od = depths[j];
    int ox = offsets_x[j];
    int oy = offsets_y[j];
    int oz = offsets_z[j];

    // Local coords in cell j
    int ljx = ((gx - ox) % Nx + Nx) % Nx;
    int ljy = ((gy - oy) % Ny + Ny) % Ny;
    int ljz = ((gz - oz) % Nz + Nz) % Nz;

    if (ljx < ow && ljy < oh && ljz < od) {
      float phi_j = phi_ptrs[j][ljz * (ow * oh) + ljy * ow + ljx];
      sum_phi_j_sq += phi_j * phi_j;
    }
  }

  // Repulsion: 2 * κ_int * φ_i * Σ φ_j²
  d_repulsion[flat_idx] =
      2.0f * interaction_coeff * phi_i[flat_idx] * sum_phi_j_sq;
}

//-----------------------------------------------------------------------------
// Batched RHS + Euler step kernel (flattened for parallelism)
// Buffer layout: [lap][bulk][constraint][advection][repulsion] (5 buffers)
//-----------------------------------------------------------------------------
__global__ void kernel_fused_rhs_step_batched_3d(
    float **__restrict__ phi_ptrs, const float *__restrict__ work_buffer,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ depths, const int *__restrict__ field_sizes,
    float gamma, float dt, int num_cells, int max_field_size) {
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells)
    return;

  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int d = depths[cell_idx];
  int field_size = field_sizes[cell_idx];
  int wh = w * h;

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= field_size)
    return;

  size_t base = (size_t)cell_idx * 5 * max_field_size;

  // Buffer layout: [lap][bulk][constraint][advection][repulsion]
  const float *d_laplacian = work_buffer + base;
  const float *d_bulk = work_buffer + base + max_field_size;
  const float *d_constraint = work_buffer + base + 2 * max_field_size;
  const float *d_advection = work_buffer + base + 3 * max_field_size;
  const float *d_repulsion = work_buffer + base + 4 * max_field_size;

  float *phi = phi_ptrs[cell_idx];

  // Combine RHS
  float dphi_dt = combine_rhs_terms(
      d_laplacian[flat_idx], d_bulk[flat_idx], d_constraint[flat_idx],
      d_repulsion[flat_idx], d_advection[flat_idx], gamma);

  // Euler step with clamping
  float new_phi = phi[flat_idx] + dt * dphi_dt;
  phi[flat_idx] = fmaxf(0.0f, fminf(1.0f, new_phi));
}

//-----------------------------------------------------------------------------
// Compute velocities from polarization (constant v_A model for 3D)
//-----------------------------------------------------------------------------
__global__ void kernel_compute_velocities_3d(
    float *__restrict__ velocities_x, float *__restrict__ velocities_y,
    float *__restrict__ velocities_z,
    const float *__restrict__ integrals_x,
    const float *__restrict__ integrals_y,
    const float *__restrict__ integrals_z,
    const float *__restrict__ polarizations_x,
    const float *__restrict__ polarizations_y,
    const float *__restrict__ polarizations_z,
    float motility_coeff, float dV, float v_A, int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells)
    return;

  // v = v_I + v_A = motility_coeff * integral * dV + v_A * polarization
  velocities_x[i] = motility_coeff * integrals_x[i] * dV + v_A * polarizations_x[i];
  velocities_y[i] = motility_coeff * integrals_y[i] * dV + v_A * polarizations_y[i];
  velocities_z[i] = motility_coeff * integrals_z[i] * dV + v_A * polarizations_z[i];
}

//=============================================================================
// Fused ref_points + centroids_and_deviations + velocities kernel
// Eliminates 2 kernel launches per step (~200 μs pipeline bubble savings).
// Safe because each thread i only reads/writes its own index.
//=============================================================================
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
    float *__restrict__ velocities_x, float *__restrict__ velocities_y,
    float *__restrict__ velocities_z,
    const float *__restrict__ polarizations_x,
    const float *__restrict__ polarizations_y,
    const float *__restrict__ polarizations_z,
    float v_A,
    int Nx, int Ny, int Nz, int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells) return;

  // Phase 1: Compute reference points from bbox center
  float rx = (float)offsets_x[i] + (float)widths[i] * 0.5f;
  float ry = (float)offsets_y[i] + (float)heights[i] * 0.5f;
  float rz = (float)offsets_z[i] + (float)depths[i] * 0.5f;
  rx = fmodf(fmodf(rx, (float)Nx) + (float)Nx, (float)Nx);
  ry = fmodf(fmodf(ry, (float)Ny) + (float)Ny, (float)Ny);
  rz = fmodf(fmodf(rz, (float)Nz) + (float)Nz, (float)Nz);
  ref_x[i] = rx;
  ref_y[i] = ry;
  ref_z[i] = rz;

  // Phase 2: Centroids + volume deviations (uses rx/ry/rz from registers)
  float sum_dx = centroid_sums[i * 4 + 0];
  float sum_dy = centroid_sums[i * 4 + 1];
  float sum_dz = centroid_sums[i * 4 + 2];
  float sum_w  = centroid_sums[i * 4 + 3];

  float cx, cy, cz;
  if (sum_w > 0.0f) {
    cx = rx + sum_dx / sum_w;
    cy = ry + sum_dy / sum_w;
    cz = rz + sum_dz / sum_w;
    cx = fmodf(fmodf(cx, (float)Nx) + (float)Nx, (float)Nx);
    cy = fmodf(fmodf(cy, (float)Ny) + (float)Ny, (float)Ny);
    cz = fmodf(fmodf(cz, (float)Nz) + (float)Nz, (float)Nz);
  } else {
    cx = rx; cy = ry; cz = rz;
  }
  centroids_x[i] = cx;
  centroids_y[i] = cy;
  centroids_z[i] = cz;
  volume_deviations[i] = target_volume - volumes[i] * dV;

  // Note: velocities are now computed AFTER scatter + velocity_integral_3d,
  // not here. This kernel only computes ref_points + centroids + volume_devs.
}

#ifdef ENABLE_KERNEL_PROFILING
static double g_phase1_time = 0, g_phase2_time = 0, g_phase34_time = 0;
static double g_phase5_time = 0, g_phase6_time = 0, g_phase7a_time = 0;
static double g_phase7b_time = 0, g_phase8_time = 0, g_sync_time = 0;
static int g_profile_count = 0;
#endif

//=============================================================================
// SCATTER: Accumulate φ²(x,y,z) from all cells onto global N³ sum field.
// S(x,y,z) = Σ_all φ_k²(x,y,z)
//
// The fused kernel reads S(gx,gy,gz) - φ_i²(gx,gy,gz) to get Σ_{j≠i} φ_j²,
// converting O(k) scattered neighbor reads into 1 coalesced read per voxel.
//=============================================================================
__global__ void __launch_bounds__(256, 4) kernel_scatter_phi_sq_3d(
    float **__restrict__ phi_ptrs,
    float *__restrict__ sum_field,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ depths,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const int *__restrict__ offsets_z,
    const int *__restrict__ field_sizes,
    int Nx, int Ny, int Nz, int num_cells)
{
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells) return;

  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int d = depths[cell_idx];
  int wh = w * h;
  int field_size = field_sizes[cell_idx];

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= field_size) return;

  const float *phi = phi_ptrs[cell_idx];
  float phi_val = phi[flat_idx];
  float phi_sq = phi_val * phi_val;

  if (phi_sq < 1e-8f) return;  // Skip negligible contributions

  int ox = offsets_x[cell_idx];
  int oy = offsets_y[cell_idx];
  int oz = offsets_z[cell_idx];

  // Fast 3D index recovery: float reciprocal replaces 4 expensive integer divisions
  float rcp_wh = __frcp_rn((float)wh);
  int lz = __float2int_rd((float)flat_idx * rcp_wh);
  int t1 = lz * wh;
  lz += (t1 + wh <= flat_idx) - (t1 > flat_idx);
  int rem = flat_idx - lz * wh;
  float rcp_w = __frcp_rn((float)w);
  int ly = __float2int_rd((float)rem * rcp_w);
  int t2 = ly * w;
  ly += (t2 + w <= rem) - (t2 > rem);
  int lx = rem - ly * w;

  // Branchless periodic wrap: coordinates are in [-margin, N+margin) so single correction suffices
  int gx = ox + lx; gx += (gx < 0) * Nx - (gx >= Nx) * Nx;
  int gy = oy + ly; gy += (gy < 0) * Ny - (gy >= Ny) * Ny;
  int gz = oz + lz; gz += (gz < 0) * Nz - (gz >= Nz) * Nz;

  atomicAdd(&sum_field[(size_t)gz * Nx * Ny + gy * Nx + gx], phi_sq);
}

//=============================================================================
// VELOCITY INTEGRAL 3D: Computes v_I from current phi and sum field.
//
// v_{n,I} = (60κ/λ²ξ) ∫ φ_n (∇φ_n) Σ_{m≠n} φ_m² dV
//
// Run AFTER scatter (sum field available) and BEFORE the fused step kernel,
// so the fused kernel uses the CURRENT step's velocity for advection.
//
// 3-channel block reduction: integral_x, integral_y, integral_z per cell.
//=============================================================================

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
    int num_cells, int max_field_size)
{
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells) return;

  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int dpth = depths[cell_idx];
  int wh = w * h;
  int field_size = field_sizes[cell_idx];

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  float my_int_x = 0.0f, my_int_y = 0.0f, my_int_z = 0.0f;

  if (flat_idx < field_size) {
    float phi_val = phi_ptrs[cell_idx][flat_idx];
    if (phi_val > 1e-4f) {
      // Fast 3D index recovery
      float rcp_wh = __frcp_rn((float)wh);
      int lz = __float2int_rd((float)flat_idx * rcp_wh);
      int t1 = lz * wh;
      lz += (t1 + wh <= flat_idx) - (t1 > flat_idx);
      int rem = flat_idx - lz * wh;
      float rcp_w = __frcp_rn((float)w);
      int ly = __float2int_rd((float)rem * rcp_w);
      int t2 = ly * w;
      ly += (t2 + w <= rem) - (t2 > rem);
      int lx = rem - ly * w;

      bool in_inner = (lx >= halo && lx < w - halo &&
                       ly >= halo && ly < h - halo &&
                       lz >= halo && lz < dpth - halo);

      if (in_inner) {
        const float *phi = phi_ptrs[cell_idx];
        float inv_2dx = 0.5f / dx_grid;
        float inv_2dy = 0.5f / dy_grid;
        float inv_2dz = 0.5f / dz_grid;

        // Gradient (central differences, inner guaranteed safe)
        float grad_x = (phi[lz*wh + ly*w + lx+1] - phi[lz*wh + ly*w + lx-1]) * inv_2dx;
        float grad_y = (phi[lz*wh + (ly+1)*w + lx] - phi[lz*wh + (ly-1)*w + lx]) * inv_2dy;
        float grad_z = (phi[(lz+1)*wh + ly*w + lx] - phi[(lz-1)*wh + ly*w + lx]) * inv_2dz;

        // Σ_{j≠i} φ_j² from sum field
        int gx = offsets_x[cell_idx] + lx; gx += (gx < 0) * Nx - (gx >= Nx) * Nx;
        int gy = offsets_y[cell_idx] + ly; gy += (gy < 0) * Ny - (gy >= Ny) * Ny;
        int gz = offsets_z[cell_idx] + lz; gz += (gz < 0) * Nz - (gz >= Nz) * Nz;
        float S_xyz = sum_field[(size_t)gz * Nx * Ny + gy * Nx + gx];
        float sum_phi_j_sq = fmaxf(0.0f, S_xyz - phi_val * phi_val);

        my_int_x = phi_val * grad_x * sum_phi_j_sq;
        my_int_y = phi_val * grad_y * sum_phi_j_sq;
        my_int_z = phi_val * grad_z * sum_phi_j_sq;
      }
    }
  }

  // 3-channel hybrid warp-shuffle + shared memory reduction
  __shared__ float s_warp[3][8];  // 3 channels × 8 warps = 96 bytes

  // Intra-warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    my_int_x += __shfl_down_sync(0xffffffff, my_int_x, offset);
    my_int_y += __shfl_down_sync(0xffffffff, my_int_y, offset);
    my_int_z += __shfl_down_sync(0xffffffff, my_int_z, offset);
  }

  int warp_id = tid / 32;
  int lane = tid % 32;
  if (lane == 0) {
    s_warp[0][warp_id] = my_int_x;
    s_warp[1][warp_id] = my_int_y;
    s_warp[2][warp_id] = my_int_z;
  }
  __syncthreads();

  // Final reduction: warp 0 reduces 8 partial sums
  if (tid < 8) {
    float vx = s_warp[0][tid];
    float vy = s_warp[1][tid];
    float vz = s_warp[2][tid];
    for (int offset = 4; offset > 0; offset >>= 1) {
      vx += __shfl_down_sync(0xff, vx, offset);
      vy += __shfl_down_sync(0xff, vy, offset);
      vz += __shfl_down_sync(0xff, vz, offset);
    }
    if (tid == 0) {
      atomicAdd(&d_integrals_x[cell_idx], vx);
      atomicAdd(&d_integrals_y[cell_idx], vy);
      atomicAdd(&d_integrals_z[cell_idx], vz);
    }
  }
}

//=============================================================================
// FUSED KERNEL for 3D: Single pass computes laplacian, bulk, constraint,
// interaction (via sum field), advection, Euler step.
// Also accumulates centroid sums + volume via block reduction + atomicAdd.
//
// Eliminates ALL work buffers. ~75% of old 8-phase pipeline merged into 1 kernel.
// Writes updated phi IN-PLACE (no double buffer needed for forward Euler).
//=============================================================================
__global__ void __launch_bounds__(256, 4) kernel_fused_step_3d(
    float **__restrict__ phi_ptrs,
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
    int num_cells, int max_field_size)
{
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells) return;

  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int dpth = depths[cell_idx];
  int field_size = field_sizes[cell_idx];
  int wh = w * h;

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  // Per-thread accumulation values for block reduction (4 channels)
  float my_cent_dx = 0.0f, my_cent_dy = 0.0f, my_cent_dz = 0.0f, my_cent_w = 0.0f;

  // Early phi load: skip near-zero voxels before expensive coordinate calculation
  bool active = (flat_idx < field_size);
  float phi_val = 0.0f;
  if (active) {
    phi_val = phi_ptrs[cell_idx][flat_idx];
    active = (phi_val > 1e-6f);
  }

  if (active) {
    float *phi = phi_ptrs[cell_idx];

    // Fast 3D index recovery: float reciprocal replaces 4 expensive integer divisions
    float rcp_wh = __frcp_rn((float)wh);
    int lz = __float2int_rd((float)flat_idx * rcp_wh);
    int t1 = lz * wh;
    lz += (t1 + wh <= flat_idx) - (t1 > flat_idx);
    int rem = flat_idx - lz * wh;
    float rcp_w = __frcp_rn((float)w);
    int ly = __float2int_rd((float)rem * rcp_w);
    int t2 = ly * w;
    ly += (t2 + w <= rem) - (t2 > rem);
    int lx = rem - ly * w;

    int offset_x_i = offsets_x[cell_idx];
    int offset_y_i = offsets_y[cell_idx];
    int offset_z_i = offsets_z[cell_idx];

    bool in_inner = (lx >= halo && lx < w - halo &&
                     ly >= halo && ly < h - halo &&
                     lz >= halo && lz < dpth - halo);

    float new_phi = phi_val;  // Default for skipped voxels

    // Only compute dynamics for non-trivial phi
    if (phi_val * (1.0f - phi_val) >= 1e-8f) {
      // --- FUSED stencil: load 7 neighbors ONCE, compute both laplacian and gradient ---
      float inv_dx2 = 1.0f / (dx_grid * dx_grid);
      float inv_dy2 = 1.0f / (dy_grid * dy_grid);
      float inv_dz2 = 1.0f / (dz_grid * dz_grid);
      float inv_2dx = 0.5f / dx_grid;
      float inv_2dy = 0.5f / dy_grid;
      float inv_2dz = 0.5f / dz_grid;

      // Neumann BC neighbor indices
      int lx_m = (lx > 0) ? lx - 1 : 0;
      int lx_p = (lx < w - 1) ? lx + 1 : w - 1;
      int ly_m = (ly > 0) ? ly - 1 : 0;
      int ly_p = (ly < h - 1) ? ly + 1 : h - 1;
      int lz_m = (lz > 0) ? lz - 1 : 0;
      int lz_p = (lz < dpth - 1) ? lz + 1 : dpth - 1;

      // 6 face neighbors (shared between laplacian + gradient)
      float phi_xm = phi[lz * wh + ly * w + lx_m];
      float phi_xp = phi[lz * wh + ly * w + lx_p];
      float phi_ym = phi[lz * wh + ly_m * w + lx];
      float phi_yp = phi[lz * wh + ly_p * w + lx];
      float phi_zm = phi[lz_m * wh + ly * w + lx];
      float phi_zp = phi[lz_p * wh + ly * w + lx];

      // 12 edge neighbors (for 19-point isotropic stencil)
      // xy-plane edges
      float phi_xmym = phi[lz * wh + ly_m * w + lx_m];
      float phi_xpym = phi[lz * wh + ly_m * w + lx_p];
      float phi_xmyp = phi[lz * wh + ly_p * w + lx_m];
      float phi_xpyp = phi[lz * wh + ly_p * w + lx_p];
      // xz-plane edges
      float phi_xmzm = phi[lz_m * wh + ly * w + lx_m];
      float phi_xpzm = phi[lz_m * wh + ly * w + lx_p];
      float phi_xmzp = phi[lz_p * wh + ly * w + lx_m];
      float phi_xpzp = phi[lz_p * wh + ly * w + lx_p];
      // yz-plane edges
      float phi_ymzm = phi[lz_m * wh + ly_m * w + lx];
      float phi_ypzm = phi[lz_m * wh + ly_p * w + lx];
      float phi_ymzp = phi[lz_p * wh + ly_m * w + lx];
      float phi_ypzp = phi[lz_p * wh + ly_p * w + lx];

      // 19-point isotropic Laplacian (eliminates O(h²) grid anisotropy)
      // ∇²f = [4*(6 face) + 1*(12 edge) - 36*C] / (6h²)
      // Same structure as the 2D McLellan 9-point stencil extended to 3D.
      float inv_h2 = inv_dx2;  // dx = dy = dz = h
      float face_sum = phi_xp + phi_xm + phi_yp + phi_ym + phi_zp + phi_zm;
      float edge_sum = phi_xmym + phi_xpym + phi_xmyp + phi_xpyp
                     + phi_xmzm + phi_xpzm + phi_xmzp + phi_xpzp
                     + phi_ymzm + phi_ypzm + phi_ymzp + phi_ypzp;
      float laplacian = (4.0f * face_sum + edge_sum - 36.0f * phi_val) * inv_h2 / 6.0f;

      // Gradient from face neighbors (central differences)
      float grad_x = (phi_xp - phi_xm) * inv_2dx;
      float grad_y = (phi_yp - phi_ym) * inv_2dy;
      float grad_z = (phi_zp - phi_zm) * inv_2dz;

      // Bulk term (γ multiplies entire bracket per Palmieri convention)
      float bulk = gamma * compute_bulk_term(phi_val, bulk_coeff);

      // Volume constraint (lagged from previous step)
      float vol_dev = volume_deviations[cell_idx];
      float constraint = compute_volume_constraint_term(phi_val, vol_dev, volume_coeff);

      // Interaction via sum field: S(x,y,z) - φ_i²(x,y,z) = Σ_{j≠i} φ_j²
      // Branchless periodic wrap (avoids 6 integer divisions per voxel)
      int gx = offset_x_i + lx; gx += (gx < 0) * Nx - (gx >= Nx) * Nx;
      int gy = offset_y_i + ly; gy += (gy < 0) * Ny - (gy >= Ny) * Ny;
      int gz = offset_z_i + lz; gz += (gz < 0) * Nz - (gz >= Nz) * Nz;

      float sum_phi_j_sq = 0.0f;
      if (sum_field) {
        float S_xyz = sum_field[(size_t)gz * Nx * Ny + gy * Nx + gx];
        sum_phi_j_sq = fmaxf(0.0f, S_xyz - phi_val * phi_val);
      }

      float repulsion = 2.0f * interaction_coeff * phi_val * sum_phi_j_sq;

      // Advection
      float vx = velocities_x[cell_idx];
      float vy = velocities_y[cell_idx];
      float vz = velocities_z[cell_idx];
      float advection = vx * grad_x + vy * grad_y + vz * grad_z;

      // Full RHS + Euler step
      float var_deriv = -2.0f * gamma * laplacian + bulk + constraint + repulsion;
      float dphi_dt_val = -0.5f * var_deriv - advection;

      new_phi = phi_val + dt * dphi_dt_val;
    }

    // Write updated phi in-place
    phi_ptrs[cell_idx][flat_idx] = new_phi;

    // Centroid sums of NEW phi (for next step's volume/centroid)
    if (in_inner) {
      float new_phi_sq = new_phi * new_phi;
      float rx = ref_x[cell_idx];
      float ry = ref_y[cell_idx];
      float rz = ref_z[cell_idx];

      float gx_f = (float)(offset_x_i + lx);
      float gy_f = (float)(offset_y_i + ly);
      float gz_f = (float)(offset_z_i + lz);

      float dx_disp = gx_f - rx;
      float dy_disp = gy_f - ry;
      float dz_disp = gz_f - rz;

      if (dx_disp > Nx * 0.5f) dx_disp -= Nx;
      if (dx_disp < -Nx * 0.5f) dx_disp += Nx;
      if (dy_disp > Ny * 0.5f) dy_disp -= Ny;
      if (dy_disp < -Ny * 0.5f) dy_disp += Ny;
      if (dz_disp > Nz * 0.5f) dz_disp -= Nz;
      if (dz_disp < -Nz * 0.5f) dz_disp += Nz;

      my_cent_dx = dx_disp * new_phi_sq;
      my_cent_dy = dy_disp * new_phi_sq;
      my_cent_dz = dz_disp * new_phi_sq;
      my_cent_w = new_phi_sq;
    }
  }

  // === Hybrid warp-shuffle + shared-memory reduction (4 channels) ===
  // Stage 1: Intra-warp reduction via shuffle (no shared mem or sync needed)
  for (int offset = 16; offset > 0; offset >>= 1) {
    my_cent_dx += __shfl_down_sync(0xffffffff, my_cent_dx, offset);
    my_cent_dy += __shfl_down_sync(0xffffffff, my_cent_dy, offset);
    my_cent_dz += __shfl_down_sync(0xffffffff, my_cent_dz, offset);
    my_cent_w  += __shfl_down_sync(0xffffffff, my_cent_w, offset);
  }

  // Stage 2: Combine warp results via small shared buffer (8 warps × 4 floats = 128B)
  __shared__ float s_warp[4][8];  // [channel][warp_id]
  int warp_id = tid >> 5;
  int lane    = tid & 31;
  if (lane == 0) {
    s_warp[0][warp_id] = my_cent_dx;
    s_warp[1][warp_id] = my_cent_dy;
    s_warp[2][warp_id] = my_cent_dz;
    s_warp[3][warp_id] = my_cent_w;
  }
  __syncthreads();

  // First warp reduces the 8 partial sums
  if (warp_id == 0 && lane < 8) {
    float sum_dx = s_warp[0][lane];
    float sum_dy = s_warp[1][lane];
    float sum_dz = s_warp[2][lane];
    float sum_w  = s_warp[3][lane];
    // Reduce 8 values within first warp
    for (int offset = 4; offset > 0; offset >>= 1) {
      sum_dx += __shfl_down_sync(0xff, sum_dx, offset);
      sum_dy += __shfl_down_sync(0xff, sum_dy, offset);
      sum_dz += __shfl_down_sync(0xff, sum_dz, offset);
      sum_w  += __shfl_down_sync(0xff, sum_w, offset);
    }
    // Lane 0 writes single atomicAdd per block
    if (lane == 0 && sum_w != 0.0f) {
      atomicAdd(&d_centroid_sums[cell_idx * 4 + 0], sum_dx);
      atomicAdd(&d_centroid_sums[cell_idx * 4 + 1], sum_dy);
      atomicAdd(&d_centroid_sums[cell_idx * 4 + 2], sum_dz);
      atomicAdd(&d_centroid_sums[cell_idx * 4 + 3], sum_w);
      atomicAdd(&d_volumes[cell_idx], sum_w);
    }
  }
}

//=============================================================================
// Optimized Fused Step Function for 3D
// Similar to step_fused_v4 in 2D: batched kernels, GPU-side reductions
// Now with neighbor-list based interaction (O(k) instead of O(N²))
//=============================================================================

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
                   int *d_grid_counts, int *d_grid_cells,
                   cudaTextureObject_t *d_phi_textures) {
  const SimParams3D &params = domain.params;
  int num_cells = domain.num_cells();
  if (num_cells == 0)
    return;

  // Find max dimensions
  int max_size = 0, max_w = 0, max_h = 0;
  for (int i = 0; i < num_cells; ++i) {
    max_size = std::max(max_size, domain.cells[i]->field_size);
    max_w = std::max(max_w, domain.cells[i]->width());
    max_h = std::max(max_h, domain.cells[i]->height());
  }

  float dV = params.dx * params.dy * params.dz;
  float target_volume = params.target_volume();

  // Zero accumulators
  cudaMemsetAsync(d_volumes, 0, num_cells * sizeof(float));
  cudaMemsetAsync(d_centroid_sums, 0, num_cells * 4 * sizeof(float));

  // Compute reference points on GPU
  {
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    kernel_compute_ref_points_3d<<<blocks, threads>>>(
        d_ref_x, d_ref_y, d_ref_z, d_all_offsets_x, d_all_offsets_y,
        d_all_offsets_z, d_all_widths, d_all_heights, d_all_depths, params.Nx,
        params.Ny, params.Nz, num_cells);
  }

  // Upload polarizations (computed on CPU due to RNG)
  std::vector<float> h_pol_x(num_cells), h_pol_y(num_cells), h_pol_z(num_cells);
  for (int i = 0; i < num_cells; ++i) {
    h_pol_x[i] = domain.cells[i]->polarization.x;
    h_pol_y[i] = domain.cells[i]->polarization.y;
    h_pol_z[i] = domain.cells[i]->polarization.z;
  }
  cudaMemcpyAsync(d_polarization_x, h_pol_x.data(), num_cells * sizeof(float),
                  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_polarization_y, h_pol_y.data(), num_cells * sizeof(float),
                  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_polarization_z, h_pol_z.data(), num_cells * sizeof(float),
                  cudaMemcpyHostToDevice);

  // Flattened grid config: 1D threads, cells in y dimension
  int threads_flat = 256;
  dim3 block(threads_flat, 1, 1);
  dim3 grid((max_size + threads_flat - 1) / threads_flat, num_cells, 1);

#ifdef ENABLE_KERNEL_PROFILING
  cudaEvent_t ev_start, ev_p1, ev_p2, ev_p34, ev_p56, ev_p7, ev_p8, ev_end;
  cudaEventCreate(&ev_start); cudaEventCreate(&ev_p1); cudaEventCreate(&ev_p2);
  cudaEventCreate(&ev_p34); cudaEventCreate(&ev_p56); cudaEventCreate(&ev_p7);
  cudaEventCreate(&ev_p8); cudaEventCreate(&ev_end);
  cudaEventRecord(ev_start);
#endif

  // =========================================================================
  // PHASE 1: Batched local terms (laplacian + bulk)
  // =========================================================================
  kernel_fused_local_batched_3d<<<grid, block>>>(
      d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights, d_all_depths,
      d_all_field_sizes, params.dx, params.dy, params.dz, params.bulk_coeff(),
      num_cells, max_size);

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_p1);
#endif

  // =========================================================================
  // PHASE 2: Batched reductions (volume + centroids)
  // =========================================================================
  {
    int threads = 256;
    int blocks_per_cell = std::min((max_size + threads - 1) / threads, 32);
    dim3 reduce_grid(blocks_per_cell, num_cells);

    kernel_reduce_volumes_batched_3d<<<reduce_grid, threads,
                                       threads * sizeof(float)>>>(
        d_all_phi_ptrs, d_volumes, d_all_widths, d_all_heights, d_all_depths,
        d_all_field_sizes, params.halo_width, num_cells);

    kernel_reduce_centroid_sums_batched_3d<<<reduce_grid, threads,
                                             4 * threads * sizeof(float)>>>(
        d_all_phi_ptrs, d_centroid_sums, d_all_widths, d_all_heights,
        d_all_depths, d_all_offsets_x, d_all_offsets_y, d_all_offsets_z,
        d_all_field_sizes, d_ref_x, d_ref_y, d_ref_z, params.halo_width,
        params.Nx, params.Ny, params.Nz, num_cells);
  }

  // SYNC: Wait for reductions
  cudaDeviceSynchronize();

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_p2);
#endif

  // =========================================================================
  // PHASE 3: GPU-side centroid + volume deviation computation
  // =========================================================================
  int threads_1d = 256;
  int blocks_1d = (num_cells + threads_1d - 1) / threads_1d;

  kernel_compute_centroids_and_deviations_3d<<<blocks_1d, threads_1d>>>(
      d_centroids_x, d_centroids_y, d_centroids_z, d_volume_deviations,
      d_centroid_sums, d_volumes, d_ref_x, d_ref_y, d_ref_z, target_volume, dV,
      params.Nx, params.Ny, params.Nz, num_cells);

  // =========================================================================
  // PHASE 4: Compute velocities (v_A only, no interaction in legacy path)
  // =========================================================================
  cudaMemsetAsync(d_integrals_x, 0, num_cells * sizeof(float));
  cudaMemsetAsync(d_integrals_y, 0, num_cells * sizeof(float));
  cudaMemsetAsync(d_integrals_z, 0, num_cells * sizeof(float));
  kernel_compute_velocities_3d<<<blocks_1d, threads_1d>>>(
      d_velocities_x, d_velocities_y, d_velocities_z,
      d_integrals_x, d_integrals_y, d_integrals_z,
      d_polarization_x, d_polarization_y, d_polarization_z,
      params.motility_coeff(), dV, params.v_A, num_cells);

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_p34);
#endif

  // =========================================================================
  // PHASE 5: Batched volume constraint
  // =========================================================================
  kernel_volume_constraint_batched_3d<<<grid, block>>>(
      d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights, d_all_depths,
      d_all_field_sizes, d_volume_deviations, params.volume_coeff(), num_cells,
      max_size);

  // =========================================================================
  // PHASE 6: Batched advection
  // =========================================================================
  kernel_advection_batched_3d<<<grid, block>>>(
      d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights, d_all_depths,
      d_all_field_sizes, d_velocities_x, d_velocities_y, d_velocities_z,
      params.dx, params.dy, params.dz, num_cells, max_size);

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_p56);
#endif

  // =========================================================================
  // PHASE 7: Build neighbor list (if needed) + Interaction with neighbor list
  // =========================================================================
  if (num_cells > 1) {
    // Only rebuild neighbor list if requested (adaptive caching)
    if (rebuild_neighbors) {
      // Build neighbor list using centroid-based distance check
      // Subdomains extend R+halo from centroid. Two cells interact if their
      // subdomains overlap, requiring centroids within 2*(R+halo).
      // With halo ~ 0.15*R, this is ~2.3*R. Use 3*R for safety margin.
      float search_radius = 3.0f * params.target_radius;

      int neighbor_threads = std::min(num_cells, 256);
      int neighbor_blocks = (num_cells + neighbor_threads - 1) / neighbor_threads;
      kernel_build_neighbor_list_3d<<<neighbor_blocks, neighbor_threads>>>(
          d_centroids_x, d_centroids_y, d_centroids_z,
          d_neighbor_counts, d_neighbor_lists,
          params.Nx, params.Ny, params.Nz, num_cells, search_radius);
    }

    // Interaction with neighbor list (O(k) instead of O(N) per voxel)
    kernel_interaction_neighborlist_3d<<<grid, block>>>(
        d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights,
        d_all_depths, d_all_field_sizes, d_all_offsets_x, d_all_offsets_y,
        d_all_offsets_z, d_neighbor_counts, d_neighbor_lists,
        params.interaction_coeff(), params.Nx, params.Ny,
        params.Nz, num_cells, max_size);
  }

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_p7);
#endif

  // =========================================================================
  // PHASE 8: Batched RHS + Euler step
  // =========================================================================
  kernel_fused_rhs_step_batched_3d<<<grid, block>>>(
      d_all_phi_ptrs, d_work_buffer, d_all_widths, d_all_heights, d_all_depths,
      d_all_field_sizes, params.gamma, dt, num_cells, max_size);

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_p8);
#endif

  // =========================================================================
  // FINAL SYNC
  // =========================================================================
  cudaError_t err = cudaDeviceSynchronize();

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_end);
  cudaEventSynchronize(ev_end);
  float t_p1, t_p2, t_p34, t_p56, t_p7, t_p8, t_total;
  cudaEventElapsedTime(&t_p1, ev_start, ev_p1);
  cudaEventElapsedTime(&t_p2, ev_p1, ev_p2);
  cudaEventElapsedTime(&t_p34, ev_p2, ev_p34);
  cudaEventElapsedTime(&t_p56, ev_p34, ev_p56);
  cudaEventElapsedTime(&t_p7, ev_p56, ev_p7);
  cudaEventElapsedTime(&t_p8, ev_p7, ev_p8);
  cudaEventElapsedTime(&t_total, ev_start, ev_end);
  g_phase1_time += t_p1; g_phase2_time += t_p2; g_phase34_time += t_p34;
  g_phase5_time += t_p56; g_phase7b_time += t_p7; g_phase8_time += t_p8;
  g_sync_time += (t_total - t_p1 - t_p2 - t_p34 - t_p56 - t_p7 - t_p8);
  g_profile_count++;
  if (g_profile_count % 100 == 0) {
    float n = (float)g_profile_count;
    float total_all = g_phase1_time+g_phase2_time+g_phase34_time+g_phase5_time+g_phase7b_time+g_phase8_time+g_sync_time;
    printf("\n=== 3D Profile (avg %d steps) ==="
           "\n  P1 local:     %7.3f ms (%5.1f%%)"  
           "\n  P2 reduce:    %7.3f ms (%5.1f%%)"  
           "\n  P3+4 cen/vel: %7.3f ms (%5.1f%%)"  
           "\n  P5+6 con/adv: %7.3f ms (%5.1f%%)"  
           "\n  P7 interact:  %7.3f ms (%5.1f%%)"  
           "\n  P8 rhs+step:  %7.3f ms (%5.1f%%)"  
           "\n  Sync/other:   %7.3f ms (%5.1f%%)"  
           "\n  TOTAL:        %7.3f ms\n",
           g_profile_count,
           g_phase1_time/n, 100*g_phase1_time/total_all,
           g_phase2_time/n, 100*g_phase2_time/total_all,
           g_phase34_time/n, 100*g_phase34_time/total_all,
           g_phase5_time/n, 100*g_phase5_time/total_all,
           g_phase7b_time/n, 100*g_phase7b_time/total_all,
           g_phase8_time/n, 100*g_phase8_time/total_all,
           g_sync_time/n, 100*g_sync_time/total_all,
           total_all/n);
    g_phase1_time=g_phase2_time=g_phase34_time=g_phase5_time=0;
    g_phase7b_time=g_phase8_time=g_sync_time=0; g_profile_count=0;
  }
  cudaEventDestroy(ev_start); cudaEventDestroy(ev_p1); cudaEventDestroy(ev_p2);
  cudaEventDestroy(ev_p34); cudaEventDestroy(ev_p56); cudaEventDestroy(ev_p7);
  cudaEventDestroy(ev_p8); cudaEventDestroy(ev_end);
#endif
  if (err != cudaSuccess) {
    printf("CUDA error in step_fused_3d: %s\n", cudaGetErrorString(err));
    return;
  }

  // Sync centroids back to host when needed
  if (sync_centroids) {
    std::vector<float> h_centroids_x(num_cells), h_centroids_y(num_cells),
        h_centroids_z(num_cells);
    std::vector<float> h_volumes(num_cells);
    std::vector<float> h_vx(num_cells), h_vy(num_cells), h_vz(num_cells);

    cudaMemcpy(h_centroids_x.data(), d_centroids_x, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids_y.data(), d_centroids_y, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids_z.data(), d_centroids_z, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_volumes.data(), d_volumes, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vx.data(), d_velocities_x, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vy.data(), d_velocities_y, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vz.data(), d_velocities_z, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_cells; ++i) {
      domain.cells[i]->centroid.x = h_centroids_x[i];
      domain.cells[i]->centroid.y = h_centroids_y[i];
      domain.cells[i]->centroid.z = h_centroids_z[i];
      domain.cells[i]->volume = h_volumes[i] * dV;
      domain.cells[i]->volume_deviation =
          target_volume - domain.cells[i]->volume;
      domain.cells[i]->velocity.x = h_vx[i];
      domain.cells[i]->velocity.y = h_vy[i];
      domain.cells[i]->velocity.z = h_vz[i];
    }
  }
}

//=============================================================================
// GPU-side RNG for Polarization Updates
//=============================================================================

__global__ void kernel_init_rng_states_3d(curandState *states,
                                          unsigned long long seed,
                                          int num_cells) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_cells) {
    curand_init(seed, idx, 0, &states[idx]);
  }
}

__global__ void kernel_update_polarizations_3d(
    curandState *states, float *polarizations_x, float *polarizations_y,
    float *polarizations_z, float dt, float tau, bool use_abp, int num_cells) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_cells) return;

  curandState local_state = states[idx];
  
  float px = polarizations_x[idx];
  float py = polarizations_y[idx];
  float pz = polarizations_z[idx];

  if (use_abp) {
    // Active Brownian Particle: rotational diffusion
    float D_r = 1.0f / tau;
    float sigma = sqrtf(2.0f * D_r * dt);
    
    // Generate random rotation angles
    float theta = curand_normal(&local_state) * sigma;
    float phi_angle = curand_uniform(&local_state) * 2.0f * 3.14159265f;
    
    // Rodrigues' rotation formula for small angles
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    
    // Random rotation axis perpendicular to current polarization
    float ax = -py * sinf(phi_angle) + pz * cosf(phi_angle);
    float ay = px * sinf(phi_angle);
    float az = -px * cosf(phi_angle);
    float anorm = sqrtf(ax*ax + ay*ay + az*az);
    if (anorm > 1e-6f) {
      ax /= anorm; ay /= anorm; az /= anorm;
    }
    
    // Rotate polarization
    float dot = ax*px + ay*py + az*pz;
    float cx = ay*pz - az*py;
    float cy = az*px - ax*pz;
    float cz = ax*py - ay*px;
    
    px = px*cos_t + cx*sin_t + ax*dot*(1.0f - cos_t);
    py = py*cos_t + cy*sin_t + ay*dot*(1.0f - cos_t);
    pz = pz*cos_t + cz*sin_t + az*dot*(1.0f - cos_t);
  } else {
    // Run-and-Tumble: Poisson tumbles with random new direction
    float p_tumble = 1.0f - expf(-dt / tau);
    
    if (curand_uniform(&local_state) < p_tumble) {
      // Generate uniform random direction on sphere
      float u = curand_uniform(&local_state) * 2.0f - 1.0f;
      float theta = curand_uniform(&local_state) * 2.0f * 3.14159265f;
      float r = sqrtf(1.0f - u*u);
      
      px = r * cosf(theta);
      py = r * sinf(theta);
      pz = u;
    }
  }
  
  // Normalize
  float norm = sqrtf(px*px + py*py + pz*pz);
  if (norm > 1e-6f) {
    px /= norm; py /= norm; pz /= norm;
  }
  
  polarizations_x[idx] = px;
  polarizations_y[idx] = py;
  polarizations_z[idx] = pz;
  states[idx] = local_state;
}

//=============================================================================
// Kernel Profiling
//=============================================================================

// g_phase*_time and g_profile_count are declared before step_fused_3d above

void print_3d_kernel_profile() {
#ifdef ENABLE_KERNEL_PROFILING
  if (g_profile_count == 0) {
    printf("\n3D Kernel profiling not enabled or no steps recorded.\n");
    return;
  }
  
  double total = g_phase1_time + g_phase2_time + g_phase34_time + g_phase5_time +
                 g_phase6_time + g_phase7a_time + g_phase7b_time + g_phase8_time +
                 g_sync_time;
  
  printf("\n=== 3D Kernel Profile (avg over %d steps) ===\n", g_profile_count);
  printf("Phase 1 (local terms):    %8.3f ms (%5.1f%%)\n", 
         g_phase1_time / g_profile_count, 100.0 * g_phase1_time / total);
  printf("Phase 2 (reductions):     %8.3f ms (%5.1f%%)\n",
         g_phase2_time / g_profile_count, 100.0 * g_phase2_time / total);
  printf("Phase 3+4 (centroid/vel): %8.3f ms (%5.1f%%)\n",
         g_phase34_time / g_profile_count, 100.0 * g_phase34_time / total);
  printf("Phase 5 (vol constraint): %8.3f ms (%5.1f%%)\n",
         g_phase5_time / g_profile_count, 100.0 * g_phase5_time / total);
  printf("Phase 6 (advection):      %8.3f ms (%5.1f%%)\n",
         g_phase6_time / g_profile_count, 100.0 * g_phase6_time / total);
  printf("Phase 7a (neighbor list): %8.3f ms (%5.1f%%)\n",
         g_phase7a_time / g_profile_count, 100.0 * g_phase7a_time / total);
  printf("Phase 7b (interaction):   %8.3f ms (%5.1f%%)\n",
         g_phase7b_time / g_profile_count, 100.0 * g_phase7b_time / total);
  printf("Phase 8 (RHS + step):     %8.3f ms (%5.1f%%)\n",
         g_phase8_time / g_profile_count, 100.0 * g_phase8_time / total);
  printf("SYNC overhead:            %8.3f ms (%5.1f%%)\n",
         g_sync_time / g_profile_count, 100.0 * g_sync_time / total);
  printf("Total per step:           %8.3f ms\n", total / g_profile_count);
  printf("==============================================\n\n");
#else
  printf("\nKernel profiling disabled. Rebuild with -DENABLE_KERNEL_PROFILING=ON\n");
#endif
}

//=============================================================================
// GPU BOUNDING BOX PIPELINE FOR 3D
//
// Port of the 2D gpu_update_all_bboxes_2d system:
// 1. Batched scan: one kernel finds extent for ALL cells
// 2. GPU early exit: if no cell needs resize, skip D→H entirely
// 3. CPU decision: which cells to resize, compute new bboxes
// 4. Batched remap: copy phi data from old→new layout in one kernel
// 5. GPU array patching: update device arrays without H→D memcpy
//=============================================================================

// Initialize bbox scan results buffer (10 ints per cell for 3D)
__global__ void kernel_init_bbox_scan_results_3d(int *results, int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells) return;
  int base = i * 10;
  results[base + 0] = 0;     // max_dist_x
  results[base + 1] = 0;     // max_dist_y
  results[base + 2] = 0;     // max_dist_z
  results[base + 3] = 99999; // min_lx
  results[base + 4] = -1;    // max_lx
  results[base + 5] = 99999; // min_ly
  results[base + 6] = -1;    // max_ly
  results[base + 7] = 99999; // min_lz
  results[base + 8] = -1;    // max_lz
  results[base + 9] = 0;     // found_any
}

// Batched bbox scan: all cells in one launch (blockIdx.y = cell)
__global__ void kernel_bbox_scan_batched_3d(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ depths,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const int *__restrict__ offsets_z,
    const int *__restrict__ field_sizes,
    const float *__restrict__ d_centroids_x,
    const float *__restrict__ d_centroids_y,
    const float *__restrict__ d_centroids_z,
    int Nx, int Ny, int Nz,
    int halo, float threshold,
    int *__restrict__ results,
    int num_cells)
{
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells) return;

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int w = widths[cell_idx];
  int h = heights[cell_idx];
  int d = depths[cell_idx];
  int fs = field_sizes[cell_idx];
  if (flat_idx >= fs) return;

  int wh = w * h;
  int lz = flat_idx / wh;
  int ly = (flat_idx % wh) / w;
  int lx = flat_idx % w;

  if (lx < halo || lx >= w - halo ||
      ly < halo || ly >= h - halo ||
      lz < halo || lz >= d - halo)
    return;

  float val = phi_ptrs[cell_idx][flat_idx];
  if (val <= threshold) return;

  int *res = results + cell_idx * 10;
  atomicMax(&res[9], 1);
  atomicMin(&res[3], lx);
  atomicMax(&res[4], lx);
  atomicMin(&res[5], ly);
  atomicMax(&res[6], ly);
  atomicMin(&res[7], lz);
  atomicMax(&res[8], lz);

  float centroid_x = d_centroids_x[cell_idx];
  float centroid_y = d_centroids_y[cell_idx];
  float centroid_z = d_centroids_z[cell_idx];

  int ox = offsets_x[cell_idx];
  int oy = offsets_y[cell_idx];
  int oz = offsets_z[cell_idx];

  int gx = ((ox + lx) % Nx + Nx) % Nx;
  int gy = ((oy + ly) % Ny + Ny) % Ny;
  int gz = ((oz + lz) % Nz + Nz) % Nz;

  float dx = (float)gx - centroid_x;
  float dy = (float)gy - centroid_y;
  float dz = (float)gz - centroid_z;
  if (dx > Nx * 0.5f) dx -= Nx;
  if (dx < -Nx * 0.5f) dx += Nx;
  if (dy > Ny * 0.5f) dy -= Ny;
  if (dy < -Ny * 0.5f) dy += Ny;
  if (dz > Nz * 0.5f) dz -= Nz;
  if (dz < -Nz * 0.5f) dz += Nz;

  atomicMax(&res[0], (int)ceilf(fabsf(dx)));
  atomicMax(&res[1], (int)ceilf(fabsf(dy)));
  atomicMax(&res[2], (int)ceilf(fabsf(dz)));
}

// GPU-side early exit check: if no cell needs resize, return 0
__global__ void kernel_bbox_check_any_change_3d(
    const int *__restrict__ results,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ depths,
    int halo, float lambda, int min_subdomain_size,
    int *__restrict__ any_change_flag,
    int num_cells)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells) return;

  int base = i * 10;
  if (!results[base + 9]) return;  // no phi found for this cell

  int min_lx = results[base + 3], max_lx = results[base + 4];
  int min_ly = results[base + 5], max_ly = results[base + 6];
  int min_lz = results[base + 7], max_lz = results[base + 8];
  int old_w = widths[i], old_h = heights[i], old_d = depths[i];

  bool touching = (min_lx <= halo+1) || (max_lx >= old_w - halo - 2) ||
                  (min_ly <= halo+1) || (max_ly >= old_h - halo - 2) ||
                  (min_lz <= halo+1) || (max_lz >= old_d - halo - 2);

  int margin = (int)(2.0f * lambda) + halo;
  int half_w = max(results[base+0] + margin, min_subdomain_size/2);
  int half_h = max(results[base+1] + margin, min_subdomain_size/2);
  int half_d = max(results[base+2] + margin, min_subdomain_size/2);
  int new_total = (2*half_w+2*halo) * (2*half_h+2*halo) * (2*half_d+2*halo);
  int old_total = old_w * old_h * old_d;
  bool worth_shrinking = (new_total < old_total * 4 / 5);

  if (touching || worth_shrinking)
    atomicMax(any_change_flag, 1);
}

// Batched 3D remap kernel: copies phi from old bbox to new bbox
struct BboxRemapParams3D {
  float *src;
  float *dst;
  int old_w, old_h, old_d, old_ox, old_oy, old_oz;
  int new_w, new_h, new_d, new_ox, new_oy, new_oz;
  int new_size;
};

__global__ void kernel_bbox_remap_3d_batched(
    const BboxRemapParams3D *__restrict__ params_arr,
    int Nx, int Ny, int Nz, int num_remaps)
{
  int remap_idx = blockIdx.y;
  if (remap_idx >= num_remaps) return;

  const BboxRemapParams3D &p = params_arr[remap_idx];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= p.new_size) return;

  int nwh = p.new_w * p.new_h;
  int new_lz = idx / nwh;
  int new_ly = (idx % nwh) / p.new_w;
  int new_lx = idx % p.new_w;

  int gx = ((p.new_ox + new_lx) % Nx + Nx) % Nx;
  int gy = ((p.new_oy + new_ly) % Ny + Ny) % Ny;
  int gz = ((p.new_oz + new_lz) % Nz + Nz) % Nz;

  int old_lx = ((gx - p.old_ox) % Nx + Nx) % Nx;
  int old_ly = ((gy - p.old_oy) % Ny + Ny) % Ny;
  int old_lz = ((gz - p.old_oz) % Nz + Nz) % Nz;

  float val = 0.0f;
  if (old_lx < p.old_w && old_ly < p.old_h && old_lz < p.old_d) {
    val = p.src[old_lz * (p.old_w * p.old_h) + old_ly * p.old_w + old_lx];
  }
  p.dst[idx] = val;
}

// GPU-side device array patching
struct DeviceArrayPatch3D {
  int cell_idx;
  float *new_phi;
  int new_w, new_h, new_d;
  int new_ox, new_oy, new_oz;
  int new_field_size;
};

__global__ void kernel_patch_device_arrays_3d(
    float **__restrict__ phi_ptrs,
    int *__restrict__ widths,
    int *__restrict__ heights,
    int *__restrict__ depths,
    int *__restrict__ offsets_x,
    int *__restrict__ offsets_y,
    int *__restrict__ offsets_z,
    int *__restrict__ field_sizes,
    const DeviceArrayPatch3D *__restrict__ patches,
    int num_patches)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_patches) return;
  const DeviceArrayPatch3D &p = patches[i];
  phi_ptrs[p.cell_idx] = p.new_phi;
  widths[p.cell_idx] = p.new_w;
  heights[p.cell_idx] = p.new_h;
  depths[p.cell_idx] = p.new_d;
  offsets_x[p.cell_idx] = p.new_ox;
  offsets_y[p.cell_idx] = p.new_oy;
  offsets_z[p.cell_idx] = p.new_oz;
  field_sizes[p.cell_idx] = p.new_field_size;
}

//=============================================================================
// Host function: GPU-accelerated bbox update for all 3D cells
//=============================================================================
bool gpu_update_all_bboxes_3d(
    Domain3D &domain,
    int *d_bbox_scan_results,
    float *d_centroids_x, float *d_centroids_y, float *d_centroids_z,
    float **d_phi_ptrs, int *d_widths, int *d_heights, int *d_depths,
    int *d_offsets_x, int *d_offsets_y, int *d_offsets_z,
    int *d_field_sizes, int max_field_size)
{
  const SimParams3D &params = domain.params;
  int num_cells = domain.num_cells();
  if (num_cells == 0) return false;

  int halo = params.halo_width;
  int Nx = params.Nx, Ny = params.Ny, Nz = params.Nz;
  float threshold = 0.01f;
  int adaptive_margin = static_cast<int>(2.0f * params.lambda) + halo;

  // 1. Initialize scan results on GPU
  {
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    kernel_init_bbox_scan_results_3d<<<blocks, threads>>>(d_bbox_scan_results, num_cells);
  }

  // 2. Batched scan: one kernel for ALL cells
  {
    dim3 block(256);
    dim3 grid((max_field_size + 255) / 256, num_cells);
    kernel_bbox_scan_batched_3d<<<grid, block>>>(
        d_phi_ptrs, d_widths, d_heights, d_depths,
        d_offsets_x, d_offsets_y, d_offsets_z, d_field_sizes,
        d_centroids_x, d_centroids_y, d_centroids_z,
        Nx, Ny, Nz, halo, threshold,
        d_bbox_scan_results, num_cells);
  }

  // 3. GPU-side early exit
  {
    static int *d_any_change = nullptr;
    if (!d_any_change) cudaMalloc(&d_any_change, sizeof(int));
    cudaMemsetAsync(d_any_change, 0, sizeof(int));

    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    kernel_bbox_check_any_change_3d<<<blocks, threads>>>(
        d_bbox_scan_results, d_widths, d_heights, d_depths,
        halo, params.lambda, params.min_subdomain_size,
        d_any_change, num_cells);

    int h_any = 0;
    cudaMemcpy(&h_any, d_any_change, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_any == 0) return false;  // No cells need resize — skip everything
  }

  // 4. Need resize — read scan results + centroids to host
  std::vector<int> h_results(num_cells * 10);
  cudaMemcpy(h_results.data(), d_bbox_scan_results,
             num_cells * 10 * sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<float> h_cx(num_cells), h_cy(num_cells), h_cz(num_cells);
  cudaMemcpy(h_cx.data(), d_centroids_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_cy.data(), d_centroids_y, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_cz.data(), d_centroids_z, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

  // 5. CPU decision: which cells need resize, collect remap work
  struct RemapWork {
    int cell_idx;
    BoundingBox3D new_bbox, new_bbox_halo;
    int new_size;
  };
  std::vector<RemapWork> remaps;

  for (int i = 0; i < num_cells; ++i) {
    auto &cell = domain.cells[i];
    int base = i * 10;
    int found_any = h_results[base + 9];
    if (!found_any) continue;

    cell->centroid.x = h_cx[i];
    cell->centroid.y = h_cy[i];
    cell->centroid.z = h_cz[i];

    int max_dist_x = h_results[base+0], max_dist_y = h_results[base+1], max_dist_z = h_results[base+2];
    int min_lx = h_results[base+3], max_lx = h_results[base+4];
    int min_ly = h_results[base+5], max_ly = h_results[base+6];
    int min_lz = h_results[base+7], max_lz = h_results[base+8];

    int old_w = cell->width(), old_h = cell->height(), old_d = cell->depth();

    int half_w = std::max(max_dist_x + adaptive_margin, params.min_subdomain_size/2);
    int half_h = std::max(max_dist_y + adaptive_margin, params.min_subdomain_size/2);
    int half_d = std::max(max_dist_z + adaptive_margin, params.min_subdomain_size/2);

    int new_cx = (int)h_cx[i], new_cy = (int)h_cy[i], new_cz = (int)h_cz[i];

    BoundingBox3D new_bbox = {new_cx-half_w, new_cy-half_h, new_cz-half_d,
                              new_cx+half_w, new_cy+half_h, new_cz+half_d};
    BoundingBox3D new_bbox_halo = {new_bbox.x0-halo, new_bbox.y0-halo, new_bbox.z0-halo,
                                   new_bbox.x1+halo, new_bbox.y1+halo, new_bbox.z1+halo};

    bool touching = (min_lx<=halo+1)||(max_lx>=old_w-halo-2)||
                    (min_ly<=halo+1)||(max_ly>=old_h-halo-2)||
                    (min_lz<=halo+1)||(max_lz>=old_d-halo-2);
    bool worth_shrinking = (new_bbox_halo.size() < cell->bbox_with_halo.size() * 4/5);

    if (!touching && !worth_shrinking) continue;

    if (touching && !worth_shrinking) {
      int overshoot = (int)(0.25f * adaptive_margin);
      half_w += overshoot; half_h += overshoot; half_d += overshoot;
      new_bbox = {new_cx-half_w, new_cy-half_h, new_cz-half_d,
                  new_cx+half_w, new_cy+half_h, new_cz+half_d};
      new_bbox_halo = {new_bbox.x0-halo, new_bbox.y0-halo, new_bbox.z0-halo,
                       new_bbox.x1+halo, new_bbox.y1+halo, new_bbox.z1+halo};
    }

    int abs_new_size = abs(new_bbox_halo.width()) * abs(new_bbox_halo.height()) * abs(new_bbox_halo.depth());
    remaps.push_back({i, new_bbox, new_bbox_halo, abs_new_size});
  }

  if (remaps.empty()) return false;

  // 6. Batched remap: use pool's alternate slot (d_dphi_dt) as destination
  //    For pool-managed cells: remap d_phi → d_dphi_dt, then swap pointers.
  //    For non-pool cells: fallback to cudaMalloc/Free.
  int num_remaps = (int)remaps.size();
  std::vector<BboxRemapParams3D> h_remap_params(num_remaps);
  std::vector<float *> old_phi_ptrs_to_free; // Only for non-pool cells
  int max_new_size = 0;

  for (int r = 0; r < num_remaps; ++r) {
    auto &work = remaps[r];
    auto &cell = domain.cells[work.cell_idx];

    h_remap_params[r].src = (float *)cell->d_phi;  // Read from current phi
    h_remap_params[r].old_w = cell->width();
    h_remap_params[r].old_h = cell->height();
    h_remap_params[r].old_d = cell->depth();
    h_remap_params[r].old_ox = cell->bbox_with_halo.x0;
    h_remap_params[r].old_oy = cell->bbox_with_halo.y0;
    h_remap_params[r].old_oz = cell->bbox_with_halo.z0;
    h_remap_params[r].new_w = work.new_bbox_halo.width();
    h_remap_params[r].new_h = work.new_bbox_halo.height();
    h_remap_params[r].new_d = work.new_bbox_halo.depth();
    h_remap_params[r].new_ox = work.new_bbox_halo.x0;
    h_remap_params[r].new_oy = work.new_bbox_halo.y0;
    h_remap_params[r].new_oz = work.new_bbox_halo.z0;
    h_remap_params[r].new_size = work.new_size;
    max_new_size = std::max(max_new_size, work.new_size);

    if (cell->pool_managed) {
      // Pool path: remap into d_dphi_dt slot, then swap pointers
      h_remap_params[r].dst = (float *)cell->d_dphi_dt;
      // Swap phi ↔ dphi_dt (remapped data is now in d_phi after swap)
      FieldType3D *old_phi = cell->d_phi;
      cell->d_phi = cell->d_dphi_dt;
      cell->d_dphi_dt = old_phi;
    } else {
      // Non-pool fallback: allocate new buffer
      float *d_phi_new = nullptr;
      cudaMalloc(&d_phi_new, work.new_size * sizeof(float));
      h_remap_params[r].dst = d_phi_new;
      old_phi_ptrs_to_free.push_back((float *)cell->d_phi);
      cell->d_phi = (FieldType3D *)d_phi_new;
    }

    cell->field_size = work.new_size;
    cell->bbox = work.new_bbox;
    cell->bbox_with_halo = work.new_bbox_halo;
  }

  // Upload remap params and launch batched remap
  static BboxRemapParams3D *d_remap_params = nullptr;
  static size_t remap_cap = 0;
  if ((size_t)num_remaps > remap_cap) {
    if (d_remap_params) cudaFree(d_remap_params);
    remap_cap = std::max((size_t)num_remaps, std::max(remap_cap*2, (size_t)16));
    cudaMalloc(&d_remap_params, remap_cap * sizeof(BboxRemapParams3D));
  }
  cudaMemcpy(d_remap_params, h_remap_params.data(),
             num_remaps * sizeof(BboxRemapParams3D), cudaMemcpyHostToDevice);

  dim3 remap_grid((max_new_size + 255) / 256, num_remaps);
  kernel_bbox_remap_3d_batched<<<remap_grid, 256>>>(
      d_remap_params, Nx, Ny, Nz, num_remaps);

  // 7. GPU-side device array patching
  std::vector<DeviceArrayPatch3D> h_patches(num_remaps);
  for (int r = 0; r < num_remaps; ++r) {
    auto &work = remaps[r];
    auto &cell = domain.cells[work.cell_idx];
    h_patches[r].cell_idx = work.cell_idx;
    h_patches[r].new_phi = (float *)cell->d_phi;
    h_patches[r].new_w = work.new_bbox_halo.width();
    h_patches[r].new_h = work.new_bbox_halo.height();
    h_patches[r].new_d = work.new_bbox_halo.depth();
    h_patches[r].new_ox = work.new_bbox_halo.x0;
    h_patches[r].new_oy = work.new_bbox_halo.y0;
    h_patches[r].new_oz = work.new_bbox_halo.z0;
    h_patches[r].new_field_size = work.new_size;
  }

  static DeviceArrayPatch3D *d_patches = nullptr;
  static size_t patch_cap = 0;
  if ((size_t)num_remaps > patch_cap) {
    if (d_patches) cudaFree(d_patches);
    patch_cap = std::max((size_t)num_remaps, std::max(patch_cap*2, (size_t)16));
    cudaMalloc(&d_patches, patch_cap * sizeof(DeviceArrayPatch3D));
  }
  cudaMemcpy(d_patches, h_patches.data(),
             num_remaps * sizeof(DeviceArrayPatch3D), cudaMemcpyHostToDevice);

  kernel_patch_device_arrays_3d<<<(num_remaps+255)/256, 256>>>(
      d_phi_ptrs, d_widths, d_heights, d_depths,
      d_offsets_x, d_offsets_y, d_offsets_z, d_field_sizes,
      d_patches, num_remaps);

  // Deferred free for non-pool allocations only
  if (!old_phi_ptrs_to_free.empty()) {
    cudaDeviceSynchronize();
    for (float *ptr : old_phi_ptrs_to_free) {
      cudaFree(ptr);
    }
  }

  return true;
}

} // namespace cellsim
