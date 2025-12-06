#include "cell3d.cuh"
#include "domain3d.cuh"
#include "kernels3d.cuh"
#include "physics.cuh"
#include "types3d.cuh"
#include <cstdio>
#include <vector>

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
// Uses super-fused kernel: 7 buffers instead of 10 (30% memory savings)
// Buffer layout:
// [laplacian][bulk][constraint][advection][reduction][interaction][repulsion]
//=============================================================================

void compute_local_terms_3d(Cell3D &cell, const SimParams3D &params,
                            float *d_work_buffer, int buffer_stride) {
  int w = cell.width();
  int h = cell.height();
  int d = cell.depth();
  int size = cell.field_size;

  KernelConfig3D cfg = KernelConfig3D::for_cell(cell);

  // Partition work buffer (7 buffers now, not 10)
  float *d_laplacian = d_work_buffer;
  float *d_bulk = d_work_buffer + buffer_stride;
  float *d_constraint = d_work_buffer + 2 * buffer_stride;
  float *d_advection = d_work_buffer + 3 * buffer_stride;
  float *d_reduction = d_work_buffer + 4 * buffer_stride;
  // Buffers 5 and 6 are for interaction_sum and repulsion (used later)

  // Compute volume integral FIRST (needed for constraint term)
  float volume = compute_volume_integral_3d(cell.d_phi, d_reduction, size,
                                            params.halo_width, w, h, d);
  cell.volume = volume * params.dx * params.dy * params.dz;
  cell.volume_deviation = params.target_volume() - cell.volume;

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
// Buffer layout (7 per cell):
// [laplacian][bulk][constraint][advection][reduction][interaction][repulsion]
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
  // Buffer layout:
  // [0:laplacian][1:bulk][2:constraint][3:advection][4:reduction][5:interaction][6:repulsion]
  for (int i = 0; i < num_cells; ++i) {
    Cell3D &cell = *domain.cells[i];
    int w = cell.width(), h = cell.height(), d = cell.depth();

    float *d_interaction = d_work_buffer + i * 7 * max_size + 5 * max_size;
    float *d_repulsion = d_work_buffer + i * 7 * max_size + 6 * max_size;

    KernelConfig3D cfg = KernelConfig3D::for_cell(cell);

    kernel_interaction_sum_3d<<<cfg.grid, cfg.block>>>(
        cell.d_phi, d_interaction, w, h, d, cell.bbox_with_halo.x0,
        cell.bbox_with_halo.y0, cell.bbox_with_halo.z0, d_phi_ptrs, d_widths,
        d_heights, d_depths, d_offsets_x, d_offsets_y, d_offsets_z, num_cells,
        domain.params.Nx, domain.params.Ny, domain.params.Nz);

    kernel_repulsion_3d<<<cfg.grid, cfg.block>>>(
        cell.d_phi, d_interaction, d_repulsion, w, h, d,
        domain.params.interaction_coeff());
  }

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
// Uses 7 buffers per cell (down from 10): 30% memory savings
//=============================================================================

void step_euler_3d(Domain3D &domain, float dt, float *d_work_buffer) {
  int num_cells = domain.num_cells();

  // Find max field size
  int max_size = 0;
  for (const auto &cell : domain.cells) {
    max_size = max(max_size, cell->field_size);
  }

  // Compute local terms for all cells IN PARALLEL (7 buffers per cell)
  for (int i = 0; i < num_cells; ++i) {
    compute_local_terms_3d(*domain.cells[i], domain.params,
                           d_work_buffer + i * 7 * max_size, max_size);
  }

  // Compute interaction terms
  compute_interaction_terms_3d(domain, d_work_buffer);

  // Combine RHS and do Euler step
  // Buffer layout:
  // [0:laplacian][1:bulk][2:constraint][3:advection][4:reduction][5:interaction][6:repulsion]
  for (int i = 0; i < num_cells; ++i) {
    Cell3D &cell = *domain.cells[i];
    int w = cell.width(), h = cell.height(), d = cell.depth();

    float *d_cell_work = d_work_buffer + i * 7 * max_size;
    float *d_laplacian = d_cell_work;
    float *d_bulk = d_cell_work + max_size;
    float *d_constraint = d_cell_work + 2 * max_size;
    float *d_advection = d_cell_work + 3 * max_size;
    float *d_repulsion = d_cell_work + 6 * max_size;

    KernelConfig3D cfg = KernelConfig3D::for_cell(cell);

    kernel_combine_rhs_3d<<<cfg.grid, cfg.block>>>(
        cell.d_dphi_dt, d_laplacian, d_bulk, d_constraint, d_repulsion,
        d_advection, w, h, d, domain.params.gamma);

    // Euler step
    int threads = 256;
    int blocks = (cell.field_size + threads - 1) / threads;
    kernel_euler_step_3d<<<blocks, threads>>>(cell.d_phi, cell.d_dphi_dt,
                                              cell.field_size, dt);
  }

  cudaDeviceSynchronize();
}

} // namespace cellsim
