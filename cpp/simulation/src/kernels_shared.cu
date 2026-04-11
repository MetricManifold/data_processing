#include "kernels.cuh"
#ifdef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"
#endif
#include <cstdio>

// Periodic coordinate wrap: maps any integer to [0, N).
// Handles arbitrarily large positive or negative values (full modulo).
__device__ __forceinline__ int wrap_coord(int x, int N) {
  return ((x % N) + N) % N;
}

// Work buffer layout: 2 sub-buffers per cell
// [0] integrand_x  (written by fused kernel, read by integral reduction)
// [1] integrand_y  (written by fused kernel, read by integral reduction)
// partial_rhs was eliminated by fusing the Euler step into the main kernel.
// Laplacian/bulk/gradients are recomputed from phi on-the-fly.
#define NUM_WORK_BUFFERS 2

namespace cellsim {

//=============================================================================
// GPU kernel to compute reference points from bbox data (eliminates CPU memcpy)
// ref = bbox center wrapped to [0, N)
// NOTE: MOVED TO kernels_solver.cu to keep in same translation unit
//=============================================================================

/*
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

  // DEBUG: Write known values to confirm kernel runs
  ref_x[i] = 999.0f;
  ref_y[i] = 888.0f;
  
  // DISABLED: actual computation
  ...
}
*/



//=============================================================================
// SLIM EULER STEP: Apply advection + Euler integration using partial_rhs
//
// Used with the fused constraint+interaction+partialrhs kernel above.
// partial_rhs already contains -0.5 * var_deriv (stored in slot [0]).
// This kernel just adds the advection term and applies the time step.
//
// dφ/dt = partial_rhs - v·∇φ
// φ_new = φ + dt * dφ/dt  (clamped to [0,1])
//
// Reads: partial_rhs (slot [0]), grad_x (slot [3]), grad_y (slot [4]) → 3 reads
// vs old kernel_fused_rhs_step_batched: 6 reads (laplacian, bulk, constraint,
// grad_x, grad_y, repulsion)
//=============================================================================

__global__ void kernel_euler_step_with_advection(
    float **phi_ptrs,
    const float *__restrict__ work_buffer,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const float *__restrict__ velocities_x,
    const float *__restrict__ velocities_y,
    float dx, float dy,
    float dt, int num_cells, int max_field_size)
{
  int cell_idx = blockIdx.z;
  if (cell_idx >= num_cells) return;

  int width = widths[cell_idx];
  int height = heights[cell_idx];

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;

  if (lx >= width || ly >= height) return;

  int idx = ly * width + lx;

  const float *phi = phi_ptrs[cell_idx];
  float phi_cur = phi[idx];

  int base = cell_idx * NUM_WORK_BUFFERS * max_field_size;
  float partial_rhs = work_buffer[base + idx];  // slot [0], pre-computed by fused kernel

  // Recompute gradients from phi (cheaper than reading from work buffer—
  // phi array is ~97KB per cell, fits in L2; work buffer is 228MB, doesn't)
  float inv_2dx = 0.5f / dx;
  float inv_2dy = 0.5f / dy;
  int xm = (lx > 0) ? lx - 1 : 0;
  int xp = (lx < width - 1) ? lx + 1 : width - 1;
  int ym = (ly > 0) ? ly - 1 : 0;
  int yp = (ly < height - 1) ? ly + 1 : height - 1;
  float grad_x = (phi[ly * width + xp] - phi[ly * width + xm]) * inv_2dx;
  float grad_y = (phi[yp * width + lx] - phi[ym * width + lx]) * inv_2dy;

  float vx = velocities_x[cell_idx];
  float vy = velocities_y[cell_idx];

  float advection = vx * grad_x + vy * grad_y;
  float dphi_dt = partial_rhs - advection;

  float *phi_out = phi_ptrs[cell_idx];
  float new_phi = phi_cur + dt * dphi_dt;
  phi_out[idx] = new_phi;
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
    const float *__restrict__ d_v_A, int num_cells) {
  int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_idx >= num_cells)
    return;

  float cell_v_A = d_v_A[cell_idx];
  // velocity = motility_coeff * integral * dA + v_A * polarization
  velocities_x[cell_idx] = motility_coeff * integrals_x[cell_idx] * dA +
                           cell_v_A * polarizations_x[cell_idx];
  velocities_y[cell_idx] = motility_coeff * integrals_y[cell_idx] * dA +
                           cell_v_A * polarizations_y[cell_idx];
}

//=============================================================================
// GPU-SIDE CENTROID + VOLUME DEVIATION: Compute from reduction results
//=============================================================================

__global__ void kernel_compute_centroids_and_deviations(
    float *__restrict__ centroids_x, float *__restrict__ centroids_y,
    float *__restrict__ volume_deviations,
    float *__restrict__ volumes,  // also write raw volume for sync_centroids D→H
    const float *__restrict__ centroid_sums, // [dx_phi2, dy_phi2, phi2] * N
    const float *__restrict__ ref_x,
    const float *__restrict__ ref_y,
    const float *__restrict__ d_target_area, float dA, int Nx,
    int Ny, int num_cells) {
  int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_idx >= num_cells)
    return;

  float sum_dx = centroid_sums[cell_idx * 3 + 0];
  float sum_dy = centroid_sums[cell_idx * 3 + 1];
  float sum_phi2 = centroid_sums[cell_idx * 3 + 2];

  // Compute volume deviation (volume = sum_phi2 * dA)
  float volume = sum_phi2 * dA;
  volumes[cell_idx] = volume;
  volume_deviations[cell_idx] = d_target_area[cell_idx] - volume;

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
// SHARED BATCHED KERNELS
//
// These kernels are used by the main solver in kernels_solver.cu.
// Key features:
// 1. ALL per-cell kernels batched into single launches (Z-dimension)
// 2. Block-level reductions with atomicAdd for volume/centroid sums
// 3. GPU-side computation of volume deviations, centroids, and velocities
//=============================================================================

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

//=============================================================================
// SCATTER: Accumulate φ²(x,y) from all cells onto global N×N sum field.
//
// S(x,y) = Σ_all φ_k²(x,y)
//
// The fused kernel then reads S(gx,gy) - φ_i²(gx,gy) instead of looping
// over ~10 neighbors with scattered random reads. This converts O(k) random
// L2/DRAM reads per pixel into 1 coalesced L2 read.
//
// At 288 cells the sum field is 1562²×4 = 9.4 MB (fits in 48 MB L2).
// At 10k cells it's still 9.4 MB — cost is independent of cell count.
//=============================================================================

__global__ void kernel_scatter_phi_sq(
    float **__restrict__ phi_ptrs,
    float *__restrict__ sum_field,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    int Nx, int Ny, int num_cells)
{
  int cell_idx = blockIdx.z;
  if (cell_idx >= num_cells) return;

  int width = widths[cell_idx];
  int height = heights[cell_idx];
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;

  if (lx >= width || ly >= height) return;

  const float *phi = phi_ptrs[cell_idx];
  float phi_val = phi[ly * width + lx];
  float phi_sq = phi_val * phi_val;

  int ox = offsets_x[cell_idx];
  int oy = offsets_y[cell_idx];
  int gx = wrap_coord(ox + lx, Nx);
  int gy = wrap_coord(oy + ly, Ny);

  atomicAdd(&sum_field[gy * Nx + gx], phi_sq);
}

//=============================================================================
// SCATTER: Accumulate φ from all cells for gradient adhesion coupling.
//
// S_lin(x,y) = Σ_all φ_k(x,y)
//
// Only allocated and launched when adhesion_J > 0.
// The fused kernel computes ∇²(S_lin) and subtracts ∇²φ_i to get
// Σ_{j≠i} ∇²φ_j, which is the variational derivative of the gradient
// coupling energy F_adh = J Σ_{i<j} ∫ ∇φ_i·∇φ_j dA.
//
// The gradient coupling acts as surface-tension reduction at shared
// interfaces. It attracts cells from afar and repels deep interpenetration,
// creating a natural equilibrium at first interface contact (d ≈ 2R).
//=============================================================================

__global__ void kernel_scatter_phi_linear(
    float **__restrict__ phi_ptrs,
    float *__restrict__ sum_field_linear,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    int Nx, int Ny, int num_cells)
{
  int cell_idx = blockIdx.z;
  if (cell_idx >= num_cells) return;

  int width = widths[cell_idx];
  int height = heights[cell_idx];
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;

  if (lx >= width || ly >= height) return;

  const float *phi = phi_ptrs[cell_idx];
  float phi_val = phi[ly * width + lx];

  int ox = offsets_x[cell_idx];
  int oy = offsets_y[cell_idx];
  int gx = wrap_coord(ox + lx, Nx);
  int gy = wrap_coord(oy + ly, Ny);

  atomicAdd(&sum_field_linear[gy * Nx + gx], phi_val);
}

//=============================================================================
// VELOCITY INTEGRAL: Computes v_I from current phi and sum field.
//
// v_{n,I} = (60κ/λ²ξ) ∫ φ_n (∇φ_n) Σ_{m≠n} φ_m² dA
//
// Run AFTER scatter (sum field available) and BEFORE the fused step kernel,
// so the fused kernel uses the CURRENT step's velocity for advection
// (eliminates the 1-step velocity lag).
//
// 2-channel block reduction: integral_x, integral_y per cell.
//=============================================================================

__global__ void kernel_velocity_integral_2d(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const float *__restrict__ sum_field,
    float *__restrict__ scatter_sum_field,
    float *__restrict__ d_integrals_x,
    float *__restrict__ d_integrals_y,
    int *__restrict__ d_block_arrival,
    float *__restrict__ d_velocities_x,
    float *__restrict__ d_velocities_y,
    const float *__restrict__ d_polarization_x,
    const float *__restrict__ d_polarization_y,
    const float *__restrict__ d_v_A,
    float motility_coeff, float dA,
    float dx_grid, float dy_grid,
    int halo, int Nx, int Ny,
    int num_cells)
{
  int cell_idx = blockIdx.z;
  if (cell_idx >= num_cells) return;

  int width = widths[cell_idx];
  int height = heights[cell_idx];

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int block_size = blockDim.x * blockDim.y;

  float my_int_x = 0.0f, my_int_y = 0.0f;

  bool active = (lx < width && ly < height);
  if (active) {
    const float *phi = phi_ptrs[cell_idx];
    float phi_val = phi[ly * width + lx];

    // Scatter φ² to next-step sum field (fused with velocity integral to avoid
    // redundant phi load). Only runs when scatter_sum_field is non-null (N>288).
    if (scatter_sum_field) {
      float phi_sq = phi_val * phi_val;
      int ox = offsets_x[cell_idx];
      int oy = offsets_y[cell_idx];
      int gx = wrap_coord(ox + lx, Nx);
      int gy = wrap_coord(oy + ly, Ny);
      atomicAdd(&scatter_sum_field[gy * Nx + gx], phi_sq);
    }

    bool in_inner = (lx >= halo && lx < width - halo &&
                     ly >= halo && ly < height - halo);
    if (in_inner) {
      {
        float inv_2dx = 0.5f / dx_grid;
        float inv_2dy = 0.5f / dy_grid;

        int xm = lx - 1; // safe: in_inner guarantees lx >= halo >= 1
        int xp = lx + 1;
        int ym = ly - 1;
        int yp = ly + 1;

        float grad_x = (phi[ly * width + xp] - phi[ly * width + xm]) * inv_2dx;
        float grad_y = (phi[yp * width + lx] - phi[ym * width + lx]) * inv_2dy;

        // Σ_{j≠i} φ_j² = S(x,y) - φ_i²
        int offset_x_i = offsets_x[cell_idx];
        int offset_y_i = offsets_y[cell_idx];
        int gx = wrap_coord(offset_x_i + lx, Nx);
        int gy = wrap_coord(offset_y_i + ly, Ny);

        float S_xy = sum_field[gy * Nx + gx];
        float sum_phi_j_sq = fmaxf(0.0f, S_xy - phi_val * phi_val);

        my_int_x = phi_val * grad_x * sum_phi_j_sq;
        my_int_y = phi_val * grad_y * sum_phi_j_sq;
      }
    }
  }

  // 2-channel block reduction
  extern __shared__ float smem[];
  float *s0 = smem;                  // int_x
  float *s1 = smem + block_size;     // int_y
  s0[tid] = my_int_x;
  s1[tid] = my_int_y;
  __syncthreads();

  for (int s = block_size / 2; s > 32; s >>= 1) {
    if (tid < s) {
      s0[tid] += s0[tid + s];
      s1[tid] += s1[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    float v0 = s0[tid] + s0[tid + 32];
    float v1 = s1[tid] + s1[tid + 32];
    for (int offset = 16; offset > 0; offset >>= 1) {
      v0 += __shfl_down_sync(0xffffffff, v0, offset);
      v1 += __shfl_down_sync(0xffffffff, v1, offset);
    }
    if (tid == 0) {
      atomicAdd(&d_integrals_x[cell_idx], v0);
      atomicAdd(&d_integrals_y[cell_idx], v1);

      // Last-arriving block computes final velocity, eliminating a separate kernel launch.
      // __threadfence() ensures our integral atomicAdds are globally visible before
      // we increment the arrival counter, so the last block sees all contributions.
      __threadfence();
      int total_blocks = gridDim.x * gridDim.y;
      int arrived = atomicAdd(&d_block_arrival[cell_idx], 1);
      if (arrived == total_blocks - 1) {
        float vA = d_v_A[cell_idx];
        d_velocities_x[cell_idx] = motility_coeff * d_integrals_x[cell_idx] * dA
                                 + vA * d_polarization_x[cell_idx];
        d_velocities_y[cell_idx] = motility_coeff * d_integrals_y[cell_idx] * dA
                                 + vA * d_polarization_y[cell_idx];
      }
    }
  }
}

//=============================================================================
// FUSED: Constraint + Interaction + Euler step + centroid/perimeter reduction
//
// Computes the variational derivative, advection (using CURRENT velocity),
// Euler step, and accumulates centroid sums + perimeter for the next step.
//
// Velocity integrals are computed in a separate pre-pass kernel
// (kernel_velocity_integral_2d) so that the current step's velocity
// is available for advection — no 1-step lag.
//
// 4-channel block reduction: cent_dx, cent_dy, cent_phi2, grad_mag.
//=============================================================================
//=============================================================================

__global__ void kernel_fused_step(
    float **__restrict__ phi_ptrs,
    float **__restrict__ phi_out_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const float *__restrict__ sum_field,
    const float *__restrict__ sum_field_linear,
    float *__restrict__ next_sum_field,
    const float *__restrict__ volume_deviations,
    const float *__restrict__ velocities_x,
    const float *__restrict__ velocities_y,
    float *__restrict__ d_centroid_sums,
    float *__restrict__ d_perimeters,
    float *__restrict__ d_second_moment_x,
    float *__restrict__ d_second_moment_y,
    const int *__restrict__ d_shift_x,
    const int *__restrict__ d_shift_y,
    const int *__restrict__ old_widths,
    const int *__restrict__ old_heights,
    const float *__restrict__ ref_x,
    const float *__restrict__ ref_y,
    const float *__restrict__ d_volume_coeff,
    float two_interaction_coeff,
    float adhesion_J,
    const float *__restrict__ d_two_gamma_bulk,
    const float *__restrict__ d_two_gamma,
    float inv_h2, float inv_2dx, float inv_2dy,
    float dt,
    int halo, int Nx, int Ny,
    int num_cells,
    bool compute_moments,
    bool has_remap)
{
  int cell_idx = blockIdx.z;
  if (cell_idx >= num_cells) return;

  int width = widths[cell_idx];
  int height = heights[cell_idx];

  // When no remap is happening (9 of 10 steps), skip loading old dims + shifts.
  // old_w == width, old_h == height, sx == sy == 0 on those steps.
  int old_w, old_h, sx, sy;
  if (has_remap) {
    old_w = old_widths[cell_idx];
    old_h = old_heights[cell_idx];
    sx = d_shift_x[cell_idx];
    sy = d_shift_y[cell_idx];
  } else {
    old_w = width;
    old_h = height;
    sx = 0;
    sy = 0;
  }

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  // Per-thread accumulation values for block reduction
  float my_cent_dx = 0.0f, my_cent_dy = 0.0f, my_cent_phi2 = 0.0f;
  float my_grad_mag = 0.0f;
  float my_dx2 = 0.0f, my_dy2 = 0.0f;  // second moments for bbox extent

  bool active = (lx < width && ly < height);

  if (active) {
    int offset_x_i = offsets_x[cell_idx];
    int offset_y_i = offsets_y[cell_idx];

    // Inline remap shifts: use pre-loaded sx/sy (skips global read when has_remap=false)
    int rx = lx + sx;  // source coord in old buffer
    int ry = ly + sy;

    const float *phi = phi_ptrs[cell_idx];
    // If source pixel is out of old buffer bounds, phi is 0 (clean edge from remap/resize)
    float phi_val = (rx >= 0 && rx < old_w && ry >= 0 && ry < old_h)
                    ? phi[ry * old_w + rx] : 0.0f;

    bool in_inner = (lx >= halo && lx < width - halo &&
                     ly >= halo && ly < height - halo);

    float new_phi = phi_val;

    // Global coordinates for sum field READ: use OLD offsets + shifted source coords
    // (sum field was scattered by previous step using old offsets)
    int gx = wrap_coord(offset_x_i + rx, Nx);
    int gy = wrap_coord(offset_y_i + ry, Ny);

    // NEW global coordinates for scatter (offset + lx + sx = new_offset + lx)
    int ngx = wrap_coord(offset_x_i + lx + sx, Nx);
    int ngy = wrap_coord(offset_y_i + ly + sy, Ny);
    // Guard: skip PDE if source pixel is out of buffer bounds (remap strip)
    bool source_valid = (rx >= 0 && rx < old_w && ry >= 0 && ry < old_h);
    if (source_valid) {
      // --- Stencil + PDE (inv_h2, inv_2dx, inv_2dy are precomputed kernel params) ---
      // Stencil reads from OLD buffer at shifted coordinates
      int xm = (rx > 0) ? rx - 1 : 0;
      int xp = (rx < old_w - 1) ? rx + 1 : old_w - 1;
      int ym = (ry > 0) ? ry - 1 : 0;
      int yp = (ry < old_h - 1) ? ry + 1 : old_h - 1;

      float phi_xm = phi[ry * old_w + xm];
      float phi_xp = phi[ry * old_w + xp];
      float phi_ym = phi[ym * old_w + rx];
      float phi_yp = phi[yp * old_w + rx];

      // Diagonal neighbors (for McLellan isotropic 9-point stencil)
      float phi_mm = phi[ym * old_w + xm];
      float phi_pm = phi[ym * old_w + xp];
      float phi_mp = phi[yp * old_w + xm];
      float phi_pp = phi[yp * old_w + xp];

      // McLellan isotropic 9-point Laplacian (eliminates O(h²) grid anisotropy)
      // ∇²f = [4(N+S+E+W) + (NE+NW+SE+SW) - 20*C] / (6h²)
      float laplacian = (4.0f * (phi_xp + phi_xm + phi_yp + phi_ym)
                       + (phi_pp + phi_pm + phi_mp + phi_mm)
                       - 20.0f * phi_val) * inv_h2 / 6.0f;
      // Bulk term: 2γ(60/λ²) * φ(1-φ)(1-2φ)  — d_two_gamma_bulk[cell_idx] = 2*gamma*bulk_coeff
      float bulk = d_two_gamma_bulk[cell_idx] * phi_val * (1.0f - phi_val) * (1.0f - 2.0f * phi_val);
      float grad_x = (phi_xp - phi_xm) * inv_2dx;
      float grad_y = (phi_yp - phi_ym) * inv_2dy;

      // --- Volume constraint (per-cell) ---
      float volume_deviation = volume_deviations[cell_idx];
      float constraint = -4.0f * d_volume_coeff[cell_idx] * volume_deviation * phi_val;

      // --- Interaction via sum field (gx, gy already computed above) ---
      float S_xy = sum_field[gy * Nx + gx];
      float sum_phi_j_sq = fmaxf(0.0f, S_xy - phi_val * phi_val);

      float repulsion = two_interaction_coeff * phi_val * sum_phi_j_sq;

      // --- Adhesion ---
      float adhesion = 0.0f;
      if (adhesion_J > 0.0f && sum_field_linear) {
        int gxm = (gx > 0) ? gx - 1 : Nx - 1;
        int gxp = (gx < Nx - 1) ? gx + 1 : 0;
        int gym = (gy > 0) ? gy - 1 : Ny - 1;
        int gyp = (gy < Ny - 1) ? gy + 1 : 0;
        // Cardinal neighbors
        float sl_c  = sum_field_linear[gy  * Nx + gx];
        float sl_xm = sum_field_linear[gy  * Nx + gxm];
        float sl_xp = sum_field_linear[gy  * Nx + gxp];
        float sl_ym = sum_field_linear[gym * Nx + gx];
        float sl_yp = sum_field_linear[gyp * Nx + gx];
        // Diagonal neighbors
        float sl_mm = sum_field_linear[gym * Nx + gxm];
        float sl_pm = sum_field_linear[gym * Nx + gxp];
        float sl_mp = sum_field_linear[gyp * Nx + gxm];
        float sl_pp = sum_field_linear[gyp * Nx + gxp];
        // McLellan isotropic 9-point Laplacian (matches phi stencil)
        // ∇²f = [4(N+S+E+W) + (NE+NW+SE+SW) - 20*C] / (6h²)
        float lap_S_lin = (4.0f * (sl_xp + sl_xm + sl_yp + sl_ym)
                         + (sl_pp + sl_pm + sl_mp + sl_mm)
                         - 20.0f * sl_c) * inv_h2 / 6.0f;
        float lap_neighbors = lap_S_lin - laplacian;
        adhesion = -adhesion_J * lap_neighbors;
      }

      // --- Perimeter ---
      if (in_inner) {
        my_grad_mag = sqrtf(grad_x * grad_x + grad_y * grad_y);
      }

      // --- Full RHS including advection (uses CURRENT velocity) ---
      // Palmieri Eq. 1: ∂φ/∂t + v·∇φ = -(1/2) δF/δφ
      float var_deriv = -d_two_gamma[cell_idx] * laplacian + bulk + constraint + repulsion + adhesion;
      float vx = velocities_x[cell_idx];
      float vy = velocities_y[cell_idx];
      float advection = vx * grad_x + vy * grad_y;
      float dphi_dt_val = -0.5f * var_deriv - advection;

      new_phi = phi_val + dt * dphi_dt_val;
    }

    // Enforce zero phi in the halo region (Dirichlet BC).
    // Prevents phi from "sticking" to the subdomain boundary
    // where the clamped stencil would otherwise act as Neumann BC.
    if (!in_inner) {
      new_phi = 0.0f;
    }

    // --- Write Euler output to double buffer at NEW local coords ---
    phi_out_ptrs[cell_idx][ly * width + lx] = new_phi;

    // --- Scatter new_phi² to NEXT step's sum field (uses NEW global coords) ---
    if (next_sum_field) {
      float new_phi_sq_scatter = new_phi * new_phi;
      atomicAdd(&next_sum_field[ngy * Nx + ngx], new_phi_sq_scatter);
    }

    // --- Centroid sums of NEW phi (for next step's volume/centroid) ---
    // Displacement relative to ref using actual global position (offset + lx + sx)
    if (in_inner) {
      float new_phi_sq = new_phi * new_phi;
      float dx_from_ref = (float)(offset_x_i + lx + sx) - ref_x[cell_idx];
      float dy_from_ref = (float)(offset_y_i + ly + sy) - ref_y[cell_idx];
      if (dx_from_ref > Nx * 0.5f) dx_from_ref -= Nx;
      if (dx_from_ref < -Nx * 0.5f) dx_from_ref += Nx;
      if (dy_from_ref > Ny * 0.5f) dy_from_ref -= Ny;
      if (dy_from_ref < -Ny * 0.5f) dy_from_ref += Ny;
      my_cent_dx = dx_from_ref * new_phi_sq;
      my_cent_dy = dy_from_ref * new_phi_sq;
      my_cent_phi2 = new_phi_sq;
      if (compute_moments) {
        my_dx2 = dx_from_ref * dx_from_ref * new_phi_sq;
        my_dy2 = dy_from_ref * dy_from_ref * new_phi_sq;
      }
    }
  }

  // === Block-level reduction: 4 or 6 channels ===
  extern __shared__ float smem[];
  int block_size = blockDim.x * blockDim.y;
  float *s0 = smem;                    // cent_dx
  float *s1 = smem + block_size;       // cent_dy
  float *s2 = smem + 2 * block_size;   // cent_phi2
  float *s3 = smem + 3 * block_size;   // grad_mag (perimeter)

  s0[tid] = my_cent_dx;
  s1[tid] = my_cent_dy;
  s2[tid] = my_cent_phi2;
  s3[tid] = my_grad_mag;

  if (compute_moments) {
    float *s4 = smem + 4 * block_size;
    float *s5 = smem + 5 * block_size;
    s4[tid] = my_dx2;
    s5[tid] = my_dy2;
    __syncthreads();

    for (int s = block_size / 2; s > 32; s >>= 1) {
      if (tid < s) {
        s0[tid] += s0[tid + s];
        s1[tid] += s1[tid + s];
        s2[tid] += s2[tid + s];
        s3[tid] += s3[tid + s];
        s4[tid] += s4[tid + s];
        s5[tid] += s5[tid + s];
      }
      __syncthreads();
    }

    if (tid < 32) {
      float v0 = s0[tid] + s0[tid + 32];
      float v1 = s1[tid] + s1[tid + 32];
      float v2 = s2[tid] + s2[tid + 32];
      float v3 = s3[tid] + s3[tid + 32];
      float v4 = s4[tid] + s4[tid + 32];
      float v5 = s5[tid] + s5[tid + 32];
      for (int offset = 16; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffff, v0, offset);
        v1 += __shfl_down_sync(0xffffffff, v1, offset);
        v2 += __shfl_down_sync(0xffffffff, v2, offset);
        v3 += __shfl_down_sync(0xffffffff, v3, offset);
        v4 += __shfl_down_sync(0xffffffff, v4, offset);
        v5 += __shfl_down_sync(0xffffffff, v5, offset);
      }
      if (tid == 0) {
        if (v2 > 0.0f || v3 > 0.0f) {
          atomicAdd(&d_centroid_sums[cell_idx * 3 + 0], v0);
          atomicAdd(&d_centroid_sums[cell_idx * 3 + 1], v1);
          atomicAdd(&d_centroid_sums[cell_idx * 3 + 2], v2);
          atomicAdd(&d_perimeters[cell_idx], v3);
          atomicAdd(&d_second_moment_x[cell_idx], v4);
          atomicAdd(&d_second_moment_y[cell_idx], v5);
        }
      }
    }
  } else {
    __syncthreads();

    for (int s = block_size / 2; s > 32; s >>= 1) {
      if (tid < s) {
        s0[tid] += s0[tid + s];
        s1[tid] += s1[tid + s];
        s2[tid] += s2[tid + s];
        s3[tid] += s3[tid + s];
      }
      __syncthreads();
    }

    if (tid < 32) {
      float v0 = s0[tid] + s0[tid + 32];
      float v1 = s1[tid] + s1[tid + 32];
      float v2 = s2[tid] + s2[tid + 32];
      float v3 = s3[tid] + s3[tid + 32];
      for (int offset = 16; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffff, v0, offset);
        v1 += __shfl_down_sync(0xffffffff, v1, offset);
        v2 += __shfl_down_sync(0xffffffff, v2, offset);
        v3 += __shfl_down_sync(0xffffffff, v3, offset);
      }
      if (tid == 0) {
        if (v2 > 0.0f || v3 > 0.0f) {
          atomicAdd(&d_centroid_sums[cell_idx * 3 + 0], v0);
          atomicAdd(&d_centroid_sums[cell_idx * 3 + 1], v1);
          atomicAdd(&d_centroid_sums[cell_idx * 3 + 2], v2);
          atomicAdd(&d_perimeters[cell_idx], v3);
        }
      }
    }
  }
}

// GPU-side pointer swap + offset apply + resize commit
__global__ void kernel_swap_phi_ptrs(float **phi_ptrs, float **phi_out_ptrs,
                                      int *offsets_x, int *offsets_y,
                                      const int *shift_x, const int *shift_y,
                                      int num_cells, int Nx, int Ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells) return;
  float *tmp = phi_ptrs[i];
  phi_ptrs[i] = phi_out_ptrs[i];
  phi_out_ptrs[i] = tmp;
  // Apply deferred offset shifts and normalize to [0, N)
  offsets_x[i] = wrap_coord(offsets_x[i] + shift_x[i], Nx);
  offsets_y[i] = wrap_coord(offsets_y[i] + shift_y[i], Ny);
}

// Backward-compatible swap-only overload (no offset update, used by 3D integrator)
__global__ void kernel_swap_phi_ptrs(float **phi_ptrs, float **phi_out_ptrs,
                                      int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells) return;
  float *tmp = phi_ptrs[i];
  phi_ptrs[i] = phi_out_ptrs[i];
  phi_out_ptrs[i] = tmp;
}

//=============================================================================
// Dynamic subdomain resize: remap phi to new (larger) layout
// Runs between steps. Uses phi_out as scratch, then swaps back.
// d_grow: [N*4] = {grow_left, grow_right, grow_top, grow_bottom} per cell
// One thread block per row of the output cell.
//=============================================================================
__global__ void kernel_remap_grow(
    float **__restrict__ phi_ptrs,      // source (current phi)
    float **__restrict__ phi_out_ptrs,  // destination (scratch)
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ grow,  // [N*4]: gl, gr, gt, gb per cell
    int num_cells) {
  int cell_idx = blockIdx.z;
  if (cell_idx >= num_cells) return;

  int gl = grow[cell_idx * 4 + 0];
  int gr = grow[cell_idx * 4 + 1];
  int gt = grow[cell_idx * 4 + 2];
  int gb = grow[cell_idx * 4 + 3];
  if (gl == 0 && gr == 0 && gt == 0 && gb == 0) return;

  int old_w = widths[cell_idx];
  int old_h = heights[cell_idx];
  int new_w = old_w + gl + gr;
  int new_h = old_h + gt + gb;

  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  if (lx >= new_w || ly >= new_h) return;

  const float *src = phi_ptrs[cell_idx];
  float *dst = phi_out_ptrs[cell_idx];

  // Map new local coords to old local coords
  int ox = lx - gl;
  int oy = ly - gb;
  float val = (ox >= 0 && ox < old_w && oy >= 0 && oy < old_h)
              ? src[oy * old_w + ox] : 0.0f;
  dst[ly * new_w + lx] = val;
}

// Apply resize: swap pointers + update dims/offsets for resized cells only
__global__ void kernel_apply_resize(
    float **__restrict__ phi_ptrs,
    float **__restrict__ phi_out_ptrs,
    int *__restrict__ widths, int *__restrict__ heights,
    int *__restrict__ offsets_x, int *__restrict__ offsets_y,
    const int *__restrict__ grow, int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells) return;
  int gl = grow[i * 4 + 0], gr = grow[i * 4 + 1];
  int gt = grow[i * 4 + 2], gb = grow[i * 4 + 3];
  if (gl == 0 && gr == 0 && gt == 0 && gb == 0) return;
  // Swap pointers so remapped data becomes current
  float *tmp = phi_ptrs[i];
  phi_ptrs[i] = phi_out_ptrs[i];
  phi_out_ptrs[i] = tmp;
  // Update dims and offsets
  widths[i] += gl + gr;
  heights[i] += gt + gb;
  offsets_x[i] -= gl;
  offsets_y[i] -= gb;
}

// Batched integral reduction
__global__ void kernel_reduce_integrals_batched(
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
  int base = cell_idx * NUM_WORK_BUFFERS * max_field_size;
  const float *d_integrand_x = work_buffer + base;
  const float *d_integrand_y = work_buffer + base + max_field_size;

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

//=============================================================================
// DIAGNOSTIC KERNELS (only compiled when DIAGNOSTICS_ENABLED)
//=============================================================================

#ifdef DIAGNOSTICS_ENABLED

// Accumulate energy and stress — recomputes gradients from phi on-the-fly
// (work buffer no longer stores gradients since work buffer reduction 9→3)
__global__ void kernel_diagnostics_energy_stress(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    float *__restrict__ d_E_gradient,
    float *__restrict__ d_E_bulk,
    float *__restrict__ d_sigma_xx,
    float *__restrict__ d_sigma_yy,
    float *__restrict__ d_sigma_xy,
    float *__restrict__ d_sigma_isotropic,
    float gamma,
    float bulk_coeff,  // 30/λ² for energy
    float dx, float dy,
    float dA,
    int halo_width,
    int num_cells
) {
    int cell_idx = blockIdx.z;
    if (cell_idx >= num_cells) return;
    
    int width = widths[cell_idx];
    int height = heights[cell_idx];
    
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (lx >= width || ly >= height) return;
    
    // Skip halo region
    if (lx < halo_width || lx >= width - halo_width ||
        ly < halo_width || ly >= height - halo_width) return;
    
    int idx = ly * width + lx;
    const float *phi = phi_ptrs[cell_idx];
    float phi_val = phi[idx];

    // Recompute gradients from phi stencil (same approach as euler step kernel)
    float inv_2dx = 0.5f / dx;
    float inv_2dy = 0.5f / dy;
    int xm = lx - 1;  // safe: halo check above guarantees lx >= halo_width >= 1
    int xp = lx + 1;
    int ym = ly - 1;
    int yp = ly + 1;
    float gx = (phi[ly * width + xp] - phi[ly * width + xm]) * inv_2dx;
    float gy = (phi[yp * width + lx] - phi[ym * width + lx]) * inv_2dy;
    float grad_sq = gx * gx + gy * gy;
    
    // E_gradient = γ|∇φ|²
    atomicAdd(&d_E_gradient[cell_idx], gamma * grad_sq * dA);
    
    // E_bulk = (30/λ²)φ²(1-φ)²
    float phi_clamped = fmaxf(0.0f, fminf(1.0f, phi_val));
    float omp = 1.0f - phi_clamped;
    float bulk_energy = bulk_coeff * phi_clamped * phi_clamped * omp * omp;
    atomicAdd(&d_E_bulk[cell_idx], bulk_energy * dA);
    
    // Stress tensor: σ_ij = -γ (∂φ/∂x_i)(∂φ/∂x_j) + δ_ij * isotropic
    atomicAdd(d_sigma_xx, -gamma * gx * gx * dA);
    atomicAdd(d_sigma_yy, -gamma * gy * gy * dA);
    atomicAdd(d_sigma_xy, -gamma * gx * gy * dA);
    atomicAdd(d_sigma_isotropic, (0.5f * gamma * grad_sq + bulk_energy) * dA);
}

// Accumulate interaction energy from repulsion term
__global__ void kernel_diagnostics_interaction(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const int *__restrict__ neighbor_counts,
    const int *__restrict__ neighbor_lists,
    float *__restrict__ d_E_interaction,
    float interaction_coeff,  // 30κ/λ²
    float dA,
    int halo,
    int Nx, int Ny,
    int num_cells,
    int max_field_size
) {
    int cell_idx = blockIdx.z;
    if (cell_idx >= num_cells) return;
    
    int width = widths[cell_idx];
    int height = heights[cell_idx];
    
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (lx >= width || ly >= height) return;
    
    // Skip halo region
    if (lx < halo || lx >= width - halo ||
        ly < halo || ly >= height - halo) return;
    
    int idx = ly * width + lx;
    
    int offset_x_i = offsets_x[cell_idx];
    int offset_y_i = offsets_y[cell_idx];
    
    int gx = ((offset_x_i + lx) % Nx + Nx) % Nx;
    int gy = ((offset_y_i + ly) % Ny + Ny) % Ny;
    
    float phi_i_val = phi_ptrs[cell_idx][idx];
    float phi_i_sq = phi_i_val * phi_i_val;
    
    // Sum of φ_j² over neighbors
    float sum_phi_j_sq = 0.0f;
    int num_neighbors = neighbor_counts[cell_idx];
    const int *my_neighbors = neighbor_lists + cell_idx * MAX_NEIGHBORS;
    
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
            sum_phi_j_sq += phi_j * phi_j;
        }
    }
    
    // E_interaction = (30κ/λ²) φ_n² Σ_{m≠n} φ_m²
    float interaction_energy = interaction_coeff * phi_i_sq * sum_phi_j_sq;
    atomicAdd(&d_E_interaction[cell_idx], interaction_energy * dA);
}

// Count contacts per cell (one thread per cell)
__global__ void kernel_diagnostics_count_contacts(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const int *__restrict__ neighbor_counts,
    const int *__restrict__ neighbor_lists,
    int *__restrict__ d_contacts,
    float contact_threshold,
    int Nx, int Ny,
    int num_cells
) {
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_cells) return;
    
    int contact_count = 0;
    int num_neighbors = neighbor_counts[cell_idx];
    const int *my_neighbors = neighbor_lists + cell_idx * MAX_NEIGHBORS;
    
    int w_i = widths[cell_idx];
    int h_i = heights[cell_idx];
    int ox_i = offsets_x[cell_idx];
    int oy_i = offsets_y[cell_idx];
    const float *phi_i = phi_ptrs[cell_idx];
    
    float threshold_sq = contact_threshold * contact_threshold;
    
    for (int n = 0; n < num_neighbors; ++n) {
        int j = my_neighbors[n];
        
        int w_j = widths[j];
        int h_j = heights[j];
        int ox_j = offsets_x[j];
        int oy_j = offsets_y[j];
        const float *phi_j = phi_ptrs[j];
        
        bool in_contact = false;
        
        // Sample every 4th point for efficiency
        for (int ly = 4; ly < h_i - 4 && !in_contact; ly += 4) {
            for (int lx = 4; lx < w_i - 4 && !in_contact; lx += 4) {
                int idx_i = ly * w_i + lx;
                float p_i = phi_i[idx_i];
                
                if (p_i < 0.1f) continue;
                
                int gx = ((ox_i + lx) % Nx + Nx) % Nx;
                int gy = ((oy_i + ly) % Ny + Ny) % Ny;
                
                int lx_j = ((gx - ox_j) % Nx + Nx) % Nx;
                int ly_j = ((gy - oy_j) % Ny + Ny) % Ny;
                
                if (lx_j < w_j && ly_j < h_j) {
                    float p_j = phi_j[ly_j * w_j + lx_j];
                    if (p_i * p_j > threshold_sq) {
                        in_contact = true;
                    }
                }
            }
        }
        
        if (in_contact) contact_count++;
    }
    
    d_contacts[cell_idx] = contact_count;
}

#endif // DIAGNOSTICS_ENABLED

//=============================================================================
// STRESS FIELD KERNELS (only compiled when STRESS_FIELDS_ENABLED)
//=============================================================================

#ifdef STRESS_FIELDS_ENABLED

// Compute stress tensor field from cell phase fields
// σ_αβ(x,y) = Σ_i γ (∂φ_i/∂x_α)(∂φ_i/∂x_β)
// P(x,y) = -½[σ_xx(x,y) + σ_yy(x,y)]
__global__ void kernel_compute_stress_fields(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    float *__restrict__ sigma_xx_field,
    float *__restrict__ sigma_yy_field,
    float *__restrict__ sigma_xy_field,
    float *__restrict__ pressure_field,
    float gamma,
    float dx, float dy,
    int Nx, int Ny,
    int halo_width,
    int num_cells
) {
    // Each thread handles one global grid point
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (gx >= Nx || gy >= Ny) return;
    
    int global_idx = gy * Nx + gx;
    
    // Accumulate stress contributions from all cells at this point
    float sxx = 0.0f;
    float syy = 0.0f;
    float sxy = 0.0f;
    
    float inv_2dx = 0.5f / dx;
    float inv_2dy = 0.5f / dy;
    
    for (int c = 0; c < num_cells; ++c) {
        int w = widths[c];
        int h = heights[c];
        int ox = offsets_x[c];
        int oy = offsets_y[c];
        
        // Convert global coords to local coords for this cell
        // Note: Must check that the cell actually covers this global point
        int lx = ((gx - ox) % Nx + Nx) % Nx;
        int ly = ((gy - oy) % Ny + Ny) % Ny;
        
        // The cell covers global points from ox to ox+w-1 (wrapped)
        // After the modulo, lx is valid only if lx < w (cell covers this x)
        // Similarly for ly < h
        // Also need margin for gradient stencil (1 pixel each side)
        if (lx < w && ly < h &&
            lx >= halo_width && lx < w - halo_width &&
            ly >= halo_width && ly < h - halo_width) {
            
            const float *phi = phi_ptrs[c];
            int idx = ly * w + lx;
            
            // Compute gradients using central differences
            float dphi_dx = (phi[idx + 1] - phi[idx - 1]) * inv_2dx;
            float dphi_dy = (phi[idx + w] - phi[idx - w]) * inv_2dy;
            
            // Accumulate stress: σ_αβ = -γ (∂φ/∂x_α)(∂φ/∂x_β)
            // Note: sign convention - negative for compressive
            sxx += -gamma * dphi_dx * dphi_dx;
            syy += -gamma * dphi_dy * dphi_dy;
            sxy += -gamma * dphi_dx * dphi_dy;
        }
    }
    
    // Write to global stress fields
    sigma_xx_field[global_idx] = sxx;
    sigma_yy_field[global_idx] = syy;
    sigma_xy_field[global_idx] = sxy;
    
    // Pressure = -½ tr(σ) = -½(σ_xx + σ_yy)
    pressure_field[global_idx] = -0.5f * (sxx + syy);
}

// Add isotropic contribution from bulk energy to stress field
// This kernel adds: f(φ) = (30/λ²)φ²(1-φ)² to diagonal components
__global__ void kernel_add_isotropic_stress(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    float *__restrict__ sigma_xx_field,
    float *__restrict__ sigma_yy_field,
    float *__restrict__ pressure_field,
    float bulk_coeff,  // 30/λ²
    float gamma,
    float dx, float dy,
    int Nx, int Ny,
    int halo_width,
    int num_cells
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (gx >= Nx || gy >= Ny) return;
    
    int global_idx = gy * Nx + gx;
    
    float isotropic = 0.0f;
    float inv_2dx = 0.5f / dx;
    float inv_2dy = 0.5f / dy;
    
    for (int c = 0; c < num_cells; ++c) {
        int w = widths[c];
        int h = heights[c];
        int ox = offsets_x[c];
        int oy = offsets_y[c];
        
        int lx = ((gx - ox) % Nx + Nx) % Nx;
        int ly = ((gy - oy) % Ny + Ny) % Ny;
        
        // Check cell coverage AND halo margin for gradient stencil
        if (lx < w && ly < h &&
            lx >= halo_width && lx < w - halo_width &&
            ly >= halo_width && ly < h - halo_width) {
            
            const float *phi = phi_ptrs[c];
            int idx = ly * w + lx;
            float phi_val = phi[idx];
            
            // Bulk energy density: f(φ) = (30/λ²)φ²(1-φ)²
            float phi_clamped = fmaxf(0.0f, fminf(1.0f, phi_val));
            float omp = 1.0f - phi_clamped;
            float bulk = bulk_coeff * phi_clamped * phi_clamped * omp * omp;
            
            // Gradient squared for isotropic stress contribution
            float dphi_dx = (phi[idx + 1] - phi[idx - 1]) * inv_2dx;
            float dphi_dy = (phi[idx + w] - phi[idx - w]) * inv_2dy;
            float grad_sq = dphi_dx * dphi_dx + dphi_dy * dphi_dy;
            
            // Isotropic part: ½γ|∇φ|² + f(φ)
            isotropic += 0.5f * gamma * grad_sq + bulk;
        }
    }
    
    // Add isotropic contribution to diagonal components
    sigma_xx_field[global_idx] += isotropic;
    sigma_yy_field[global_idx] += isotropic;
    
    // Update pressure with isotropic contribution
    pressure_field[global_idx] -= isotropic;  // P = -½(σ_xx + σ_yy)
}

#endif // STRESS_FIELDS_ENABLED

} // namespace cellsim
