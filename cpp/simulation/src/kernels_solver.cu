#include "kernels.cuh"
#ifdef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"
#endif
#ifdef STRESS_FIELDS_ENABLED
#ifndef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"
#endif
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace cellsim {

//=============================================================================
// NEIGHBOR-LIST BASED INTERACTION (ALL ON GPU)
//
// Instead of O(N²) loop over all cells per pixel, use a neighbor list
// to iterate only over potentially interacting cells.
//=============================================================================

// MAX_NEIGHBORS defined in kernels.cuh

// Fused per-cell kernel: ref_points + centroids + volume deviations (pre-step)
// Replaces kernel_compute_ref_points + kernel_compute_centroids_and_deviations
__global__ void kernel_pre_step(
    float *__restrict__ ref_x, float *__restrict__ ref_y,
    int *__restrict__ offsets_x, int *__restrict__ offsets_y,
    int *__restrict__ widths, int *__restrict__ heights,
    int *__restrict__ old_widths, int *__restrict__ old_heights,
    float *__restrict__ centroids_x, float *__restrict__ centroids_y,
    float *__restrict__ volume_deviations, float *__restrict__ volumes,
    float *__restrict__ centroid_sums,
    float *__restrict__ d_second_moment_x,
    float *__restrict__ d_second_moment_y,
    float *__restrict__ d_integrals_x,
    float *__restrict__ d_integrals_y,
    float *__restrict__ d_perimeters,
    int *__restrict__ d_block_arrival,
    const float *__restrict__ d_target_area, float dA,
    const float *__restrict__ d_target_radius,
    int *__restrict__ d_shift_x, int *__restrict__ d_shift_y,
    int *__restrict__ d_max_wh,  // [0]=max_w, [1]=max_h (atomicMax target)
    bool compute_shifts, bool zero_moments, int max_side,
    int Nx, int Ny, int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells) return;

  int old_off_x = offsets_x[i];
  int old_off_y = offsets_y[i];
  int w = widths[i];
  int h = heights[i];

  // Save current dims as READ dims for fused kernel + velocity_integral
  old_widths[i] = w;
  old_heights[i] = h;

  // 1. Compute ref points (bbox center, wrapped)
  float rx = (float)old_off_x + (float)w * 0.5f;
  float ry = (float)old_off_y + (float)h * 0.5f;
  rx = fmodf(fmodf(rx, (float)Nx) + (float)Nx, (float)Nx);
  ry = fmodf(fmodf(ry, (float)Ny) + (float)Ny, (float)Ny);

  // 2. Compute centroids + volume deviations from centroid sums
  float sum_dx = centroid_sums[i * 3 + 0];
  float sum_dy = centroid_sums[i * 3 + 1];
  float sum_phi2 = centroid_sums[i * 3 + 2];

  float volume = sum_phi2 * dA;
  volumes[i] = volume;
  volume_deviations[i] = d_target_area[i] - volume;

  // Zero centroid_sums for fused kernel to accumulate NEXT step's sums.
  centroid_sums[i * 3 + 0] = 0.0f;
  centroid_sums[i * 3 + 1] = 0.0f;
  centroid_sums[i * 3 + 2] = 0.0f;

  // Zero reduction accumulators (replaces cudaMemsetAsync of 4N floats)
  d_integrals_x[i] = 0.0f;
  d_integrals_y[i] = 0.0f;
  d_perimeters[i] = 0.0f;
  d_block_arrival[i] = 0;

  // Read second moments. Zero them before accumulation (zero_moments=true
  // on the step BEFORE fused kernel accumulates, so moments persist for
  // the visualizer and resize reads on the following step).
  float moment_x = d_second_moment_x[i];
  float moment_y = d_second_moment_y[i];
  if (zero_moments) {
    d_second_moment_x[i] = 0.0f;
    d_second_moment_y[i] = 0.0f;
  }

  // 3. Compute centroid shift + dynamic resize
  // sx/sy encode the mapping: new local (lx,ly) reads from old local (lx+sx, ly+sy)
  int sx = 0, sy = 0;
  int new_w = w, new_h = h;

  if (sum_phi2 > 1e-8f) {
    float cx = rx + sum_dx / sum_phi2;
    float cy = ry + sum_dy / sum_phi2;
    cx = fmodf(fmodf(cx, (float)Nx) + (float)Nx, (float)Nx);
    cy = fmodf(fmodf(cy, (float)Ny) + (float)Ny, (float)Ny);
    centroids_x[i] = cx;
    centroids_y[i] = cy;

    if (compute_shifts) {
      // --- Centroid shift (recenter subdomain) ---
      float sub_cx = (float)old_off_x + (float)w * 0.5f;
      float sub_cy = (float)old_off_y + (float)h * 0.5f;
      float dx = cx - sub_cx;
      float dy = cy - sub_cy;
      if (dx >  Nx * 0.5f) dx -= Nx;
      if (dx < -Nx * 0.5f) dx += Nx;
      if (dy >  Ny * 0.5f) dy -= Ny;
      if (dy < -Ny * 0.5f) dy += Ny;
      int candidate_sx = (int)roundf(dx);
      int candidate_sy = (int)roundf(dy);
      if (abs(candidate_sx) > 2) sx = candidate_sx;
      if (abs(candidate_sy) > 2) sy = candidate_sy;

      // --- Dynamic resize using second moments ---
      if (max_side > 0 && sum_phi2 > 1.0f) {
        float var_x = moment_x / sum_phi2;
        float var_y = moment_y / sum_phi2;
        // Guard: only resize when moments are meaningful (σ > 2 pixels).
        // On step 0 moments are zero → var ~ 0, skip to avoid catastrophic shrink.
        if (var_x > 4.0f && var_y > 4.0f) {
          float sigma_x = sqrtf(var_x);
          float sigma_y = sqrtf(var_y);
          // Target half-size: 2σ + R/2 + halo (additive margin from cell radius)
          float R = d_target_radius[i];
          int additive_margin = (int)ceilf(R * 0.5f) + 4;  // R/2 + halo
          int target_half_x = (int)ceilf(2.0f * sigma_x) + additive_margin;
          int target_half_y = (int)ceilf(2.0f * sigma_y) + additive_margin;
          int target_w = (2 * target_half_x) & ~1;  // even
          int target_h = (2 * target_half_y) & ~1;
          // Clamp to pool slot
          target_w = min(max(target_w, 32), max_side);
          target_h = min(max(target_h, 32), max_side);

          // Grow or shrink symmetrically around centroid.
          // delta > 0 → grow: sx shifts read LEFT to expose new zero-filled border.
          // delta < 0 → shrink: sx shifts read RIGHT, dropping border pixels (safe
          //             because 3σ + 12 margin ensures phi ≈ 0 at the cut edge).
          if (target_w != w) {
            new_w = target_w;
            sx -= (target_w - w) / 2;
          }
          if (target_h != h) {
            new_h = target_h;
            sy -= (target_h - h) / 2;
          }
        }
      }
    }
  }

  // Write new dims (for fused kernel WRITE path + grid launch)
  widths[i] = new_w;
  heights[i] = new_h;

  d_shift_x[i] = sx;
  d_shift_y[i] = sy;

  // Track max of NEW dims for grid launch
  atomicMax(&d_max_wh[0], new_w);
  atomicMax(&d_max_wh[1], new_h);

  // Ref point is based on OLD offset (matches fused kernel's global coord computation)
  ref_x[i] = rx;
  ref_y[i] = ry;
}

// Fused per-cell kernel: compute_velocities + swap_phi_ptrs (post-step)
// Replaces kernel_compute_velocities + kernel_swap_phi_ptrs
__global__ void kernel_post_step(
    float *__restrict__ velocities_x, float *__restrict__ velocities_y,
    const float *__restrict__ integrals_x, const float *__restrict__ integrals_y,
    const float *__restrict__ polarizations_x, const float *__restrict__ polarizations_y,
    float motility_coeff, float dA, const float *__restrict__ d_v_A,
    float **__restrict__ phi_ptrs, float **__restrict__ phi_out_ptrs,
    int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells) return;

  // 1. Compute velocity = motility_coeff * integral * dA + v_A * polarization
  float cell_v_A = d_v_A[i];
  velocities_x[i] = motility_coeff * integrals_x[i] * dA + cell_v_A * polarizations_x[i];
  velocities_y[i] = motility_coeff * integrals_y[i] * dA + cell_v_A * polarizations_y[i];

  // 2. Swap phi double-buffer pointers
  float *tmp = phi_ptrs[i];
  phi_ptrs[i] = phi_out_ptrs[i];
  phi_out_ptrs[i] = tmp;
}

// GPU kernel to compute reference points from bbox data (eliminates CPU memcpy)
// ref = bbox center wrapped to [0, N)
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

  float rx = (float)offsets_x[i] + (float)widths[i] * 0.5f;
  float ry = (float)offsets_y[i] + (float)heights[i] * 0.5f;
  // Wrap to [0, N)
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
    int *__restrict__ neighbor_lists, // [MAX_NEIGHBORS * num_cells]
    int Nx, int Ny, int num_cells,
    float search_radius) // kept for API compat, bbox overlap used instead
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells)
    return;

  // Compute bbox center for cell i
  float bcx_i = (float)offsets_x[i] + (float)widths[i] * 0.5f;
  float bcy_i = (float)offsets_y[i] + (float)heights[i] * 0.5f;
  float hw_i = (float)widths[i] * 0.5f;
  float hh_i = (float)heights[i] * 0.5f;

  int count = 0;
  int *my_neighbors = neighbor_lists + i * MAX_NEIGHBORS;

  for (int j = 0; j < num_cells; ++j) {
    if (j == i)
      continue;

    // Bbox center for cell j
    float bcx_j = (float)offsets_x[j] + (float)widths[j] * 0.5f;
    float bcy_j = (float)offsets_y[j] + (float)heights[j] * 0.5f;
    float hw_j = (float)widths[j] * 0.5f;
    float hh_j = (float)heights[j] * 0.5f;

    // Periodic distance between bbox centers
    float dx = bcx_j - bcx_i;
    float dy = bcy_j - bcy_i;
    if (dx > Nx * 0.5f)
      dx -= Nx;
    else if (dx < -Nx * 0.5f)
      dx += Nx;
    if (dy > Ny * 0.5f)
      dy -= Ny;
    else if (dy < -Ny * 0.5f)
      dy += Ny;

    // Overlap check: bboxes overlap if distance < sum of half-widths
    if (fabsf(dx) < hw_i + hw_j && fabsf(dy) < hh_i + hh_j) {
      if (count < MAX_NEIGHBORS) {
        my_neighbors[count] = j;
        count++;
      }
    }
  }

  neighbor_counts[i] = count;
}

//=============================================================================
// Main solver step function
//=============================================================================

// Forward declarations - these are defined in kernels_shared.cu
__global__ void kernel_reduce_centroid_sums_batched(
    float **__restrict__ phi_ptrs, float *__restrict__ centroid_sums,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ offsets_x, const int *__restrict__ offsets_y,
    const float *__restrict__ ref_x, const float *__restrict__ ref_y,
    int halo_width, int Nx, int Ny, int num_cells);

__global__ void kernel_compute_centroids_and_deviations(
    float *__restrict__ centroids_x, float *__restrict__ centroids_y,
    float *__restrict__ volume_deviations,
    float *__restrict__ volumes,
    const float *__restrict__ centroid_sums,
    const float *__restrict__ ref_x, const float *__restrict__ ref_y,
    const float *__restrict__ d_target_area, float dA, int Nx, int Ny, int num_cells);

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
    bool has_remap,
    float *__restrict__ d_integrals_x,
    float *__restrict__ d_integrals_y,
    int *__restrict__ d_block_arrival,
    float *__restrict__ out_velocities_x,
    float *__restrict__ out_velocities_y,
    const float *__restrict__ d_polarization_x,
    const float *__restrict__ d_polarization_y,
    const float *__restrict__ d_v_A,
    float motility_coeff, float dA);

__global__ void kernel_scatter_phi_sq(
    float **__restrict__ phi_ptrs,
    float *__restrict__ sum_field,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    int Nx, int Ny, int num_cells);

__global__ void kernel_scatter_phi_linear(
    float **__restrict__ phi_ptrs,
    float *__restrict__ sum_field_linear,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    int Nx, int Ny, int num_cells);

__global__ void kernel_swap_phi_ptrs(float **phi_ptrs, float **phi_out_ptrs,
                                      int *offsets_x, int *offsets_y,
                                      const int *shift_x, const int *shift_y,
                                      int num_cells);

__global__ void kernel_compute_velocities(
    float *__restrict__ velocities_x, float *__restrict__ velocities_y,
    const float *__restrict__ integrals_x,
    const float *__restrict__ integrals_y,
    const float *__restrict__ polarizations_x,
    const float *__restrict__ polarizations_y, float motility_coeff, float dA,
    const float *__restrict__ d_v_A, int num_cells);

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
    int num_cells);

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
                   float *d_target_radius,
                   int pool_max_side,
                   int *d_max_wh,
                   int *d_shift_x,
                   int *d_shift_y,
                   int *d_block_arrival,
                   float *d_sum_field,
                   float *d_sum_field_linear,
                   float *d_next_sum_field,
                   int &cached_max_size, int &cached_max_w, int &cached_max_h,
                   bool sync_centroids,
                   bool rebuild_neighbors,
                   bool centroid_sums_precomputed,
                   int step_counter) {
  const SimParams &params = domain.params;
  int num_cells = domain.num_cells();
  if (num_cells == 0)
    return;

#ifdef ENABLE_KERNEL_PROFILING
  // Per-phase timing with CUDA events
  static int profile_step = 0;
  static float accum_memset = 0, accum_centroid = 0, accum_neighbor = 0;
  static float accum_fused = 0, accum_velocity = 0;
  static float accum_total = 0;
  static int accum_count = 0;
  cudaEvent_t ev_start, ev_memset, ev_centroid, ev_neighbor, ev_fused;
  cudaEvent_t ev_velocity, ev_end;
  cudaEventCreate(&ev_start); cudaEventCreate(&ev_memset);
  cudaEventCreate(&ev_centroid); cudaEventCreate(&ev_neighbor);
  cudaEventCreate(&ev_fused); cudaEventCreate(&ev_velocity);
  cudaEventCreate(&ev_end);
  cudaEventRecord(ev_start);
#endif

  // Use cached max dimensions (updated on bbox change, not every step)
  int max_size = cached_max_size;
  int max_w = cached_max_w;
  int max_h = cached_max_h;

  float dA = params.dx * params.dy;

  // Common grid config (needed for scatter, velocity integral, and fused kernel)
  // 32×8 block: better coalescing for row-major phi (warp spans 1 row, not 2)
  dim3 block(32, 8, 1);
  dim3 grid((max_w + 31) / 32, (max_h + 7) / 8, num_cells);
  bool compute_moments = (step_counter % 10 == 9);
  size_t smem_fused = (compute_moments ? 8 : 6) * block.x * block.y * sizeof(float);
  size_t smem_vint  = 2 * block.x * block.y * sizeof(float);  // 2 channels: int_x, int_y

  // Reduction accumulators (integrals, perimeters, block_arrival) are zeroed
  // inside kernel_pre_step to eliminate a cudaMemsetAsync API call.

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_memset);
#endif

  int threads_1d = 256;
  int blocks_1d = (num_cells + threads_1d - 1) / threads_1d;

  // =========================================================================
  // CENTROID + VOLUME: Either use precomputed sums or reduce from phi
  // =========================================================================
  if (!centroid_sums_precomputed) {
    // First step or after bbox change: need ref points before reduce
    kernel_compute_ref_points<<<blocks_1d, threads_1d>>>(
        d_ref_x, d_ref_y, d_all_offsets_x, d_all_offsets_y,
        d_all_widths, d_all_heights, params.Nx, params.Ny, num_cells);

    cudaMemsetAsync(d_centroid_sums, 0, num_cells * 3 * sizeof(float));
    int threads = 256;
    int blocks_per_cell = std::min((max_size + threads - 1) / threads, 32);
    dim3 reduce_grid(blocks_per_cell, num_cells);
    kernel_reduce_centroid_sums_batched<<<reduce_grid, threads,
                                          3 * threads * sizeof(float)>>>(
        d_all_phi_ptrs, d_centroid_sums, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y, d_ref_x, d_ref_y, params.halo_width,
        params.Nx, params.Ny, num_cells);
  }

  // Pre-step: compute ref points + centroids + volume deviations + remap shifts
  // Also zeros centroid_sums for the fused kernel to accumulate NEXT step's sums.
  // Zero d_max_wh before pre_step so atomicMax accumulates fresh values.
  if (step_counter % 10 == 0) {
    cudaMemsetAsync(d_max_wh, 0, 2 * sizeof(int));
  }
  kernel_pre_step<<<blocks_1d, threads_1d>>>(
      d_ref_x, d_ref_y, d_all_offsets_x, d_all_offsets_y,
      d_all_widths, d_all_heights,
      d_old_widths, d_old_heights,
      d_centroids_x, d_centroids_y, d_volume_deviations, d_volumes,
      d_centroid_sums,
      d_second_moment_x, d_second_moment_y,
      d_integrals_x, d_integrals_y,
      d_perimeters, d_block_arrival,
      d_target_area, dA,
      d_target_radius,
      d_shift_x, d_shift_y,
      d_max_wh,
      (step_counter % 10 == 0), (step_counter % 10 == 9), pool_max_side,
      params.Nx, params.Ny, num_cells);

  // After resize check, read GPU-computed max dims for the FUSED kernel (write path).
  // Velocity_integral and scatter use the OLD grid (max_w/max_h from before pre_step)
  // since they read phi at the old stride.
  dim3 grid_read = grid;  // Save old grid for read-path kernels
  if (pool_max_side > 0 && step_counter % 10 == 0) {
    int h_max_wh[2];
    cudaMemcpy(h_max_wh, d_max_wh, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int new_max_w = h_max_wh[0];
    int new_max_h = h_max_wh[1];
    if (new_max_w > 0 && new_max_h > 0) {
      // Update grid for fused kernel (write path uses new dims)
      grid = dim3((new_max_w + 31) / 32, (new_max_h + 7) / 8, num_cells);
      cached_max_w = new_max_w;
      cached_max_h = new_max_h;
    }
  }

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_centroid);
#endif

  // =========================================================================
  // FUSED: constraint + interaction + Euler step + integral/centroid reduction
  //
  // Single kernel computes everything and accumulates integrals + centroid
  // sums via block-level shared memory reduction + atomicAdd.
  // No work buffer needed. No separate reduction kernel.
  // =========================================================================

  // Build neighbor list if needed (only when sum field is not available)
  if (num_cells > 1 && rebuild_neighbors && !d_sum_field) {
    float search_radius = 4.0f * params.target_radius;
    int neighbor_threads = std::min(num_cells, 256);
    int neighbor_blocks = (num_cells + neighbor_threads - 1) / neighbor_threads;
    kernel_build_neighbor_list<<<neighbor_blocks, neighbor_threads>>>(
        d_centroids_x, d_centroids_y, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y, d_neighbor_counts, d_neighbor_lists,
        params.Nx, params.Ny, num_cells, search_radius);
  }

  // Build sum field: standalone scatter when inline scatter is disabled (large N)
  // or on first step (centroid_sums not yet precomputed by fused kernel).
  if (d_sum_field && (!d_next_sum_field || !centroid_sums_precomputed)) {
    kernel_scatter_phi_sq<<<grid_read, block>>>(  
        d_all_phi_ptrs, d_sum_field, d_old_widths, d_old_heights,
        d_all_offsets_x, d_all_offsets_y, params.Nx, params.Ny, num_cells);
  }

  // Adhesion linear sum field: only scatter when J > 0
  if (d_sum_field_linear && params.adhesion_J > 0.0f) {
    kernel_scatter_phi_linear<<<grid_read, block>>>(
        d_all_phi_ptrs, d_sum_field_linear, d_old_widths, d_old_heights,
        d_all_offsets_x, d_all_offsets_y, params.Nx, params.Ny, num_cells);
  }

  // Velocity integral is folded into the fused kernel (1-step lag).
  // Advection uses previous-step velocity; fused kernel accumulates
  // new integrals and last-arriving block computes velocity for step N+1.

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_neighbor);
#endif

  // Launch fused kernel (uses PREVIOUS velocity for advection — 1-step lag)
  // Velocity integral is accumulated inside and converted to velocity for step N+1.
  float inv_h2 = 1.0f / (params.dx * params.dx);
  float inv_2dx = 0.5f / params.dx;
  float inv_2dy = 0.5f / params.dy;
  kernel_fused_step<<<grid, block, smem_fused>>>(
      d_all_phi_ptrs, d_all_phi_out_ptrs, d_all_widths, d_all_heights,
      d_all_offsets_x, d_all_offsets_y,
      d_sum_field, d_sum_field_linear, d_next_sum_field,
      d_volume_deviations, d_velocities_x, d_velocities_y,
      d_centroid_sums, d_perimeters,
      d_second_moment_x, d_second_moment_y,
      d_shift_x, d_shift_y,
      d_old_widths, d_old_heights,
      d_ref_x, d_ref_y,
      d_volume_coeff, 2.0f * params.interaction_coeff(),
      params.adhesion_J,
      d_two_gamma_bulk, d_two_gamma,
      inv_h2, inv_2dx, inv_2dy, dt,
      params.halo_width, params.Nx, params.Ny,
      num_cells, compute_moments, (step_counter % 10 == 0),
      d_integrals_x, d_integrals_y,
      d_block_arrival,
      d_velocities_x, d_velocities_y,
      d_polarization_x, d_polarization_y,
      d_v_A, params.motility_coeff(), dA);

  // Phi pointer swap is now done in kernel_pre_step of the NEXT step
  // (saves 1 kernel launch per step)

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_fused);
  cudaEventRecord(ev_velocity);
#endif

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_end);
  cudaEventSynchronize(ev_end);
  float t_memset, t_centroid, t_neighbor, t_fused, t_velocity, t_total;
  cudaEventElapsedTime(&t_memset, ev_start, ev_memset);
  cudaEventElapsedTime(&t_centroid, ev_memset, ev_centroid);
  cudaEventElapsedTime(&t_neighbor, ev_centroid, ev_neighbor);
  cudaEventElapsedTime(&t_fused, ev_neighbor, ev_fused);
  cudaEventElapsedTime(&t_velocity, ev_fused, ev_velocity);
  cudaEventElapsedTime(&t_total, ev_start, ev_end);
  accum_memset += t_memset; accum_centroid += t_centroid;
  accum_neighbor += t_neighbor; accum_fused += t_fused;
  accum_velocity += t_velocity;
  accum_total += t_total; accum_count++;
  profile_step++;
  if (profile_step % 1000 == 0) {
    float n = (float)accum_count;
    printf("\n=== 2D Kernel Profiling (avg over %d steps) ==="
           "\n  memset:     %.3f ms (%.1f%%)"
           "\n  centroid:   %.3f ms (%.1f%%)"
           "\n  neighbor:   %.3f ms (%.1f%%)"
           "\n  FUSED:      %.3f ms (%.1f%%)"
           "\n  velocity:   %.3f ms (%.1f%%)"
           "\n  TOTAL:      %.3f ms\n",
           accum_count,
           accum_memset/n, 100*accum_memset/accum_total,
           accum_centroid/n, 100*accum_centroid/accum_total,
           accum_neighbor/n, 100*accum_neighbor/accum_total,
           accum_fused/n, 100*accum_fused/accum_total,
           accum_velocity/n, 100*accum_velocity/accum_total,
           accum_total/n);
    accum_memset = accum_centroid = accum_neighbor = accum_fused = 0;
    accum_velocity = accum_total = 0; accum_count = 0;
  }
  cudaEventDestroy(ev_start); cudaEventDestroy(ev_memset);
  cudaEventDestroy(ev_centroid); cudaEventDestroy(ev_neighbor);
  cudaEventDestroy(ev_fused); cudaEventDestroy(ev_velocity);
  cudaEventDestroy(ev_end);
#endif

  // =========================================================================
  // FINAL SYNC — only when we need to copy data back to host
  // When sync_centroids=false, the next GPU operation (bbox scan, next step)
  // provides implicit ordering via default stream.
  // =========================================================================
  if (sync_centroids) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return;
    }

    std::vector<float> h_centroids_x(num_cells), h_centroids_y(num_cells);
    std::vector<float> h_volumes(num_cells);
    std::vector<float> h_vx(num_cells), h_vy(num_cells);
    std::vector<float> h_perimeters(num_cells);
    std::vector<float> h_target_area(num_cells);

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
    cudaMemcpy(h_perimeters.data(), d_perimeters, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_target_area.data(), d_target_area, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_cells; ++i) {
      domain.cells[i]->centroid.x = h_centroids_x[i];
      domain.cells[i]->centroid.y = h_centroids_y[i];
      domain.cells[i]->volume = h_volumes[i];  // already includes dA factor
      domain.cells[i]->volume_deviation = h_target_area[i] - domain.cells[i]->volume;
      domain.cells[i]->velocity.x = h_vx[i];
      domain.cells[i]->velocity.y = h_vy[i];
      // Perimeter: ∫|∇φ| dA (multiply by grid spacing since kernel sums |∇φ|)
      domain.cells[i]->perimeter = h_perimeters[i] * dA;
    }
  }
}

//=============================================================================
// DIAGNOSTIC COMPUTATION (separate function for modularity)
//=============================================================================

#ifdef DIAGNOSTICS_ENABLED

// Forward declarations for diagnostic kernels (defined in kernels_shared.cu)
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
    float bulk_coeff,
    float dx, float dy,
    float dA,
    int halo_width,
    int num_cells);

__global__ void kernel_diagnostics_interaction(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const int *__restrict__ neighbor_counts,
    const int *__restrict__ neighbor_lists,
    float *__restrict__ d_E_interaction,
    float interaction_coeff,
    float dA,
    int halo,
    int Nx, int Ny,
    int num_cells,
    int max_field_size);

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
    int num_cells);

void run_diagnostics(
    Domain &domain,
    float **d_all_phi_ptrs,
    int *d_all_widths,
    int *d_all_heights,
    int *d_all_offsets_x,
    int *d_all_offsets_y,
    int *d_neighbor_counts,
    int *d_neighbor_lists,
    DiagnosticBuffers &diag
) {
    const SimParams &params = domain.params;
    int num_cells = domain.num_cells();
    if (num_cells == 0 || !diag.allocated) return;
    
    // Find max dimensions
    int max_size = 0, max_w = 0, max_h = 0;
    for (int i = 0; i < num_cells; ++i) {
        max_size = std::max(max_size, domain.cells[i]->field_size);
        max_w = std::max(max_w, domain.cells[i]->width());
        max_h = std::max(max_h, domain.cells[i]->height());
    }
    
    float dA = params.dx * params.dy;
    
    // Grid config
    dim3 block(16, 16, 1);
    dim3 grid((max_w + 15) / 16, (max_h + 15) / 16, num_cells);
    
    // Energy and stress from gradients (recomputed from phi on-the-fly)
    kernel_diagnostics_energy_stress<<<grid, block>>>(
        d_all_phi_ptrs,
        d_all_widths, d_all_heights,
        diag.d_E_gradient, diag.d_E_bulk,
        diag.d_sigma_xx, diag.d_sigma_yy,
        diag.d_sigma_xy, diag.d_sigma_isotropic,
        params.gamma, params.bulk_coeff(),
        params.dx, params.dy, dA,
        params.halo_width, num_cells);
    
    // Interaction energy
    if (num_cells > 1) {
        kernel_diagnostics_interaction<<<grid, block>>>(
            d_all_phi_ptrs,
            d_all_widths, d_all_heights,
            d_all_offsets_x, d_all_offsets_y,
            d_neighbor_counts, d_neighbor_lists,
            diag.d_E_interaction,
            params.interaction_coeff(), dA,
            params.halo_width,
            params.Nx, params.Ny,
            num_cells, max_size);
        
        // Contact counting (one thread per cell)
        int threads = 256;
        int blocks = (num_cells + threads - 1) / threads;
        kernel_diagnostics_count_contacts<<<blocks, threads>>>(
            d_all_phi_ptrs,
            d_all_widths, d_all_heights,
            d_all_offsets_x, d_all_offsets_y,
            d_neighbor_counts, d_neighbor_lists,
            diag.d_contacts,
            0.01f,  // contact threshold (φ_i * φ_j > 0.1 * 0.1)
            params.Nx, params.Ny,
            num_cells);
    }
    
    cudaDeviceSynchronize();
}

#endif // DIAGNOSTICS_ENABLED

//=============================================================================
// STRESS FIELD COMPUTATION
//=============================================================================

#ifdef STRESS_FIELDS_ENABLED

// Kernel declarations (defined in kernels_shared.cu)
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
);

__global__ void kernel_add_isotropic_stress(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    float *__restrict__ sigma_xx_field,
    float *__restrict__ sigma_yy_field,
    float *__restrict__ pressure_field,
    float bulk_coeff,
    float gamma,
    float dx, float dy,
    int Nx, int Ny,
    int halo_width,
    int num_cells
);

void compute_stress_fields(
    Domain &domain,
    float **d_all_phi_ptrs,
    int *d_all_widths,
    int *d_all_heights,
    int *d_all_offsets_x,
    int *d_all_offsets_y,
    StressFieldBuffers &stress)
{
    if (!stress.allocated) {
        printf("[STRESS_FIELDS] Error: buffers not allocated\n");
        return;
    }
    
    const SimParams &params = domain.params;
    int num_cells = domain.num_cells();
    int Nx = params.Nx;
    int Ny = params.Ny;
    
    // Reset stress fields to zero
    stress_fields_reset(stress);
    
    // Launch configuration for domain-sized grid
    dim3 block(16, 16);
    dim3 grid((Nx + block.x - 1) / block.x,
              (Ny + block.y - 1) / block.y);
    
    // Compute deviatoric stress: σ_αβ = -γ (∂φ/∂x_α)(∂φ/∂x_β)
    kernel_compute_stress_fields<<<grid, block>>>(
        d_all_phi_ptrs,
        d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y,
        stress.d_sigma_xx_field,
        stress.d_sigma_yy_field,
        stress.d_sigma_xy_field,
        stress.d_pressure_field,
        params.gamma,
        params.dx, params.dy,
        Nx, Ny,
        params.halo_width,
        num_cells);
    
    // Add isotropic contribution: ½γ|∇φ|² + f(φ) to diagonal
    kernel_add_isotropic_stress<<<grid, block>>>(
        d_all_phi_ptrs,
        d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y,
        stress.d_sigma_xx_field,
        stress.d_sigma_yy_field,
        stress.d_pressure_field,
        params.bulk_coeff(),
        params.gamma,
        params.dx, params.dy,
        Nx, Ny,
        params.halo_width,
        num_cells);
    
    cudaDeviceSynchronize();
}

#endif // STRESS_FIELDS_ENABLED

//=============================================================================
// GPU Bounding Box Scan Kernel
//
// Scans a cell's phi field on GPU to find:
//   - Maximum periodic distance from centroid in x and y
//   - Minimum local-coordinate bounds (for edge detection)
//   - Maximum local-coordinate bounds (for edge detection)
//   - Whether any voxel exceeds threshold
//
// Uses atomicMin/atomicMax on int — hardware-supported, no shared memory needed.
// One kernel per cell, launched with enough threads to cover field_size.
//
// Output layout: d_results[7] = {max_dist_x, max_dist_y,
//                                 min_lx, max_lx, min_ly, max_ly, found_any}
// Initialize before launch: max_dist = 0, min_l = BIG, max_l = -1, found = 0
//=============================================================================

__global__ void kernel_bbox_scan_2d(
    const float *__restrict__ phi,
    int width, int height,
    int offset_x, int offset_y,   // bbox_with_halo x0, y0
    const float *__restrict__ d_centroids_x,
    const float *__restrict__ d_centroids_y,
    int cell_idx,
    int Nx, int Ny,
    int halo, float threshold,
    int *__restrict__ results)     // [7]: see layout above
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height) return;

  int ly = idx / width;
  int lx = idx % width;

  // Skip halo region
  if (lx < halo || lx >= width - halo || ly < halo || ly >= height - halo)
    return;

  float val = phi[idx];
  if (val <= threshold) return;

  // Mark found
  atomicMax(&results[6], 1);

  // Track local bounds (for edge detection)
  atomicMin(&results[2], lx);
  atomicMax(&results[3], lx);
  atomicMin(&results[4], ly);
  atomicMax(&results[5], ly);

  // Read centroid from device array (no D→H copy needed)
  float centroid_x = d_centroids_x[cell_idx];
  float centroid_y = d_centroids_y[cell_idx];

  // Compute global coordinate
  int gx = ((offset_x + lx) % Nx + Nx) % Nx;
  int gy = ((offset_y + ly) % Ny + Ny) % Ny;

  // Periodic distance from centroid
  float dx = static_cast<float>(gx) - centroid_x;
  float dy = static_cast<float>(gy) - centroid_y;
  if (dx > Nx * 0.5f) dx -= Nx;
  if (dx < -Nx * 0.5f) dx += Nx;
  if (dy > Ny * 0.5f) dy -= Ny;
  if (dy < -Ny * 0.5f) dy += Ny;

  // Store ceil of absolute distance as int for atomicMax
  int dist_x = static_cast<int>(ceilf(fabsf(dx)));
  int dist_y = static_cast<int>(ceilf(fabsf(dy)));
  atomicMax(&results[0], dist_x);
  atomicMax(&results[1], dist_y);
}

//=============================================================================
// GPU Bounding Box Remap Kernel
//
// Copies phi data from old bbox layout to new bbox layout entirely on GPU.
// Each thread handles one voxel in the NEW field: computes its global coord,
// looks up the corresponding position in the old field, copies the value.
// Voxels not covered by the old field remain 0 (from memset).
//=============================================================================

__global__ void kernel_bbox_remap_2d(
    const float *__restrict__ old_phi,
    float *__restrict__ new_phi,
    int old_w, int old_h,
    int old_ox, int old_oy,       // old bbox_with_halo x0, y0
    int new_w, int new_h,
    int new_ox, int new_oy,       // new bbox_with_halo x0, y0
    int Nx, int Ny)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= new_w * new_h) return;

  int new_ly = idx / new_w;
  int new_lx = idx % new_w;

  // Convert new local -> global
  int gx = ((new_ox + new_lx) % Nx + Nx) % Nx;
  int gy = ((new_oy + new_ly) % Ny + Ny) % Ny;

  // Convert global -> old local
  int old_lx = ((gx - old_ox) % Nx + Nx) % Nx;
  int old_ly = ((gy - old_oy) % Ny + Ny) % Ny;
  // Clamp (handles wraparound)
  if (old_lx >= old_w) old_lx -= Nx;
  if (old_ly >= old_h) old_ly -= Ny;

  if (old_lx >= 0 && old_lx < old_w && old_ly >= 0 && old_ly < old_h) {
    new_phi[idx] = old_phi[old_ly * old_w + old_lx];
  }
  // else: new_phi[idx] stays 0 from memset
}

//=============================================================================
// Host function: GPU-accelerated bounding box update for all 2D cells
//
// Replaces the old CPU-based per-cell update_bounding_box() with:
// 1. GPU scan kernels (one per cell) — find extent + edge proximity
// 2. Tiny readback (~28 bytes per cell)
// 3. CPU decision: grow / shrink / nothing
// 4. GPU remap kernel for cells that need resize
//
// Called every few steps. Returns true if any bbox changed.
//=============================================================================

// Tiny kernel to initialize bbox scan results buffer on GPU (avoids H->D copy)
__global__ void kernel_init_bbox_scan_results(int *results, int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells) return;
  int base = i * 9;
  results[base + 0] = 0;     // max_dist_x
  results[base + 1] = 0;     // max_dist_y
  results[base + 2] = 99999; // min_lx
  results[base + 3] = -1;    // max_lx
  results[base + 4] = 99999; // min_ly
  results[base + 5] = -1;    // max_ly
  results[base + 6] = 0;     // found_any
  results[base + 7] = 0;     // centroid_x (as int bits)
  results[base + 8] = 0;     // centroid_y (as int bits)
}

//=============================================================================
// BATCHED Bbox Scan: Single launch for all cells (replaces serial per-cell loop)
// Uses blockIdx.y = cell_idx, same atomics as the single-cell version.
// Also embeds centroid values in the results buffer to eliminate separate D→H copies.
//=============================================================================

__global__ void kernel_bbox_scan_2d_batched(
    float **__restrict__ phi_ptrs,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    const float *__restrict__ d_centroids_x,
    const float *__restrict__ d_centroids_y,
    int Nx, int Ny, int halo, float threshold,
    int *__restrict__ results,   // [num_cells * 9]
    int num_cells)
{
  int cell_idx = blockIdx.y;
  if (cell_idx >= num_cells) return;

  int width = widths[cell_idx];
  int height = heights[cell_idx];
  int field_size = width * height;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // First thread of first block for each cell: embed centroids in results
  if (idx == 0) {
    int *res = results + cell_idx * 9;
    res[7] = __float_as_int(d_centroids_x[cell_idx]);
    res[8] = __float_as_int(d_centroids_y[cell_idx]);
  }

  if (idx >= field_size) return;

  int ly = idx / width;
  int lx = idx % width;

  if (lx < halo || lx >= width - halo || ly < halo || ly >= height - halo)
    return;

  float val = phi_ptrs[cell_idx][idx];
  if (val <= threshold) return;

  int *res = results + cell_idx * 9;
  atomicMax(&res[6], 1);
  atomicMin(&res[2], lx);
  atomicMax(&res[3], lx);
  atomicMin(&res[4], ly);
  atomicMax(&res[5], ly);

  float centroid_x = d_centroids_x[cell_idx];
  float centroid_y = d_centroids_y[cell_idx];

  int offset_x = offsets_x[cell_idx];
  int offset_y = offsets_y[cell_idx];
  int gx = ((offset_x + lx) % Nx + Nx) % Nx;
  int gy = ((offset_y + ly) % Ny + Ny) % Ny;

  float dx = static_cast<float>(gx) - centroid_x;
  float dy = static_cast<float>(gy) - centroid_y;
  if (dx > Nx * 0.5f) dx -= Nx;
  if (dx < -Nx * 0.5f) dx += Nx;
  if (dy > Ny * 0.5f) dy -= Ny;
  if (dy < -Ny * 0.5f) dy += Ny;

  int dist_x = static_cast<int>(ceilf(fabsf(dx)));
  int dist_y = static_cast<int>(ceilf(fabsf(dy)));
  atomicMax(&res[0], dist_x);
  atomicMax(&res[1], dist_y);
}

//=============================================================================
// GPU-side device array patching: update phi pointers, widths, heights, offsets,
// field_sizes directly on GPU after bbox remap — eliminates 7× H→D memcpy.
//=============================================================================

//=============================================================================
// GPU-side bbox change detection: evaluates scan results on GPU to produce
// a single "any cell needs resize" flag. When flag=0 (most steps at steady
// state), the host skips the D→H copy of full scan results + CPU decision
// loop, eliminating the pipeline drain.
//=============================================================================

__global__ void kernel_bbox_check_any_change(
    const int *__restrict__ results,     // [num_cells * 9]
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    int halo, int Nx, int Ny,
    float lambda, int min_subdomain_size,
    int *__restrict__ any_change_flag,
    int num_cells)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_cells) return;

  int base = i * 9;
  int found_any = results[base + 6];
  if (!found_any) return;

  int min_lx = results[base + 2];
  int max_lx = results[base + 3];
  int min_ly = results[base + 4];
  int max_ly = results[base + 5];
  int old_w = widths[i];
  int old_h = heights[i];

  // Emergency: phi touching bbox boundary
  bool touching_edge = (min_lx <= halo + 1) || (max_lx >= old_w - halo - 2) ||
                       (min_ly <= halo + 1) || (max_ly >= old_h - halo - 2);

  // Quick shrink check: optimal size < 80% of current
  int adaptive_margin = (int)(3.0f * lambda) + halo;
  int max_dist_x = results[base + 0];
  int max_dist_y = results[base + 1];
  int half_w = max(max_dist_x + adaptive_margin, min_subdomain_size / 2);
  int half_h = max(max_dist_y + adaptive_margin, min_subdomain_size / 2);
  int new_total = (2 * half_w + 2 * halo) * (2 * half_h + 2 * halo);
  int old_total = old_w * old_h;
  bool worth_shrinking = (new_total < old_total * 4 / 5);

  // Center drift check
  float cx, cy;
  memcpy(&cx, &results[base + 7], sizeof(float));
  memcpy(&cy, &results[base + 8], sizeof(float));
  int new_cx = (int)cx;
  int new_cy = (int)cy;
  int old_cx_bb = offsets_x[i] + old_w / 2;
  int old_cy_bb = offsets_y[i] + old_h / 2;
  int shift_x = new_cx - old_cx_bb;
  int shift_y = new_cy - old_cy_bb;
  if (shift_x > Nx / 2) shift_x -= Nx;
  if (shift_x < -Nx / 2) shift_x += Nx;
  if (shift_y > Ny / 2) shift_y -= Ny;
  if (shift_y < -Ny / 2) shift_y += Ny;
  bool center_drifted = (abs(shift_x) >= 5 || abs(shift_y) >= 5);

  if (touching_edge || worth_shrinking || center_drifted) {
    atomicMax(any_change_flag, 1);
  }
}

//=============================================================================
// GPU-side device array patching structs and kernel
//=============================================================================

struct DeviceArrayPatch {
  int cell_idx;
  float *new_phi;
  float *new_phi_out;
  int new_w, new_h;
  int new_ox, new_oy;
  int new_field_size;
};

__global__ void kernel_patch_device_arrays(
    float **__restrict__ phi_ptrs,
    float **__restrict__ phi_out_ptrs,
    int *__restrict__ widths,
    int *__restrict__ heights,
    int *__restrict__ offsets_x,
    int *__restrict__ offsets_y,
    int *__restrict__ field_sizes,
    const DeviceArrayPatch *__restrict__ patches,
    int num_patches)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_patches) return;
  const DeviceArrayPatch &p = patches[i];
  phi_ptrs[p.cell_idx] = p.new_phi;
  phi_out_ptrs[p.cell_idx] = p.new_phi_out;
  widths[p.cell_idx] = p.new_w;
  heights[p.cell_idx] = p.new_h;
  offsets_x[p.cell_idx] = p.new_ox;
  offsets_y[p.cell_idx] = p.new_oy;
  field_sizes[p.cell_idx] = p.new_field_size;
}

//=============================================================================
// Batched Bbox Remap: Remap multiple cells in a single kernel launch.
// Each remap is indexed by blockIdx.y. The kernel writes 0 for pixels not
// covered by the old field, eliminating the need for separate cudaMemset.
//=============================================================================

struct BboxRemapParams {
  float *src;   // old phi pointer
  float *dst;   // destination pointer
  int old_w, old_h, old_ox, old_oy;
  int new_w, new_h, new_ox, new_oy;
  int new_size;
};

__global__ void kernel_bbox_remap_2d_batched(
    const BboxRemapParams *__restrict__ params,
    int Nx, int Ny, int num_remaps)
{
  int remap_idx = blockIdx.y;
  if (remap_idx >= num_remaps) return;

  const BboxRemapParams &p = params[remap_idx];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= p.new_size) return;

  int new_ly = idx / p.new_w;
  int new_lx = idx % p.new_w;

  // Convert new local -> global
  int gx = ((p.new_ox + new_lx) % Nx + Nx) % Nx;
  int gy = ((p.new_oy + new_ly) % Ny + Ny) % Ny;

  // Convert global -> old local
  int old_lx = ((gx - p.old_ox) % Nx + Nx) % Nx;
  int old_ly = ((gy - p.old_oy) % Ny + Ny) % Ny;
  if (old_lx >= p.old_w) old_lx -= Nx;
  if (old_ly >= p.old_h) old_ly -= Ny;

  float val = 0.0f;
  if (old_lx >= 0 && old_lx < p.old_w && old_ly >= 0 && old_ly < p.old_h) {
    val = p.src[old_ly * p.old_w + old_lx];
  }
  p.dst[idx] = val;
}

//=============================================================================
// Async bbox scan + change detection (no pipeline drain)
//
// Launches scan + check kernels, then copies the change flag to pinned host
// memory via cudaMemcpyAsync. The caller reads the flag on a later step.
//=============================================================================

void gpu_launch_bbox_scan_async_2d(
    float **d_all_phi_ptrs,
    int *d_all_widths, int *d_all_heights,
    int *d_all_offsets_x, int *d_all_offsets_y,
    float *d_centroids_x, float *d_centroids_y,
    const SimParams &params,
    int num_cells, int max_field_size,
    int *d_bbox_scan_results,
    int *d_any_change_flag,
    int *h_any_change_pinned)
{
  if (num_cells == 0) return;

  int halo = params.halo_width;
  int Nx = params.Nx, Ny = params.Ny;
  float threshold = 0.01f;

  // Init scan results + zero change flag in one launch
  {
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    kernel_init_bbox_scan_results<<<blocks, threads>>>(d_bbox_scan_results, num_cells);
    // Zero the change flag on the first thread (piggyback)
    cudaMemsetAsync(d_any_change_flag, 0, sizeof(int));
  }

  // Batched scan: single kernel launch for ALL cells
  {
    int threads = 256;
    int blocks_per_cell = (max_field_size + threads - 1) / threads;
    dim3 grid(blocks_per_cell, num_cells);
    kernel_bbox_scan_2d_batched<<<grid, threads>>>(
        d_all_phi_ptrs, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y,
        d_centroids_x, d_centroids_y,
        Nx, Ny, halo, threshold,
        d_bbox_scan_results, num_cells);
  }

  // Change detection + async flag copy
  {
    int eval_threads = 256;
    int eval_blocks = (num_cells + eval_threads - 1) / eval_threads;
    kernel_bbox_check_any_change<<<eval_blocks, eval_threads>>>(
        d_bbox_scan_results, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y,
        halo, Nx, Ny, params.lambda, params.min_subdomain_size,
        d_any_change_flag, num_cells);
  }
  cudaMemcpyAsync(h_any_change_pinned, d_any_change_flag, sizeof(int),
                  cudaMemcpyDeviceToHost);
}

bool gpu_update_all_bboxes_2d(Domain &domain, int *d_bbox_scan_results,
                              float *d_centroids_x, float *d_centroids_y,
                              float *d_phi_pool,
                              size_t pool_slot_size,
                              int pool_num_cells,
                              bool *pool_needs_grow,
                              float **d_all_phi_ptrs,
                              float **d_all_phi_out_ptrs,
                              int *d_all_widths,
                              int *d_all_heights,
                              int *d_all_offsets_x,
                              int *d_all_offsets_y,
                              int *d_all_field_sizes,
                              int max_field_size) {
  const SimParams &params = domain.params;
  int num_cells = domain.num_cells();
  if (num_cells == 0) return false;

  int halo = params.halo_width;
  int Nx = params.Nx, Ny = params.Ny;
  float threshold = 0.01f;
  // Use 3*lambda margin to ensure phi is negligible at the bbox boundary.
  // With adhesion (J > 0), the effective interface width is wider:
  // lambda_eff = lambda * sqrt(gamma / (gamma - J/2)).
  float lambda_margin = params.lambda;
  if (params.adhesion_J > 0.0f && params.gamma > params.adhesion_J / 2.0f) {
    lambda_margin = params.lambda * sqrtf(params.gamma / (params.gamma - params.adhesion_J / 2.0f));
  }
  int adaptive_margin = static_cast<int>(3.0f * lambda_margin) + halo;

  // Initialize scan results on GPU — 9 ints per cell (7 scan + 2 embedded centroids)
  {
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    kernel_init_bbox_scan_results<<<blocks, threads>>>(d_bbox_scan_results, num_cells);
  }

  // *** BATCHED scan: single kernel launch for ALL cells ***
  // Replaces the serial per-cell loop that launched N separate kernels.
  {
    int threads = 256;
    int blocks_per_cell = (max_field_size + threads - 1) / threads;
    dim3 grid(blocks_per_cell, num_cells);
    kernel_bbox_scan_2d_batched<<<grid, threads>>>(
        d_all_phi_ptrs, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y,
        d_centroids_x, d_centroids_y,
        Nx, Ny, halo, threshold,
        d_bbox_scan_results, num_cells);
  }

  // =========================================================================
  // GPU-side early-exit: evaluate scan results on GPU to check if ANY cell
  // needs a bbox resize. If not (most steps at steady state), skip the
  // expensive D→H copy + CPU decision loop entirely.
  // =========================================================================
  {
    static int *d_any_change_flag = nullptr;
    if (!d_any_change_flag) CUDA_CHECK(cudaMalloc(&d_any_change_flag, sizeof(int)));
    cudaMemsetAsync(d_any_change_flag, 0, sizeof(int));

    int eval_threads = 256;
    int eval_blocks = (num_cells + eval_threads - 1) / eval_threads;
    kernel_bbox_check_any_change<<<eval_blocks, eval_threads>>>(
        d_bbox_scan_results, d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y,
        halo, Nx, Ny, params.lambda, params.min_subdomain_size,
        d_any_change_flag, num_cells);

    int h_any_change = 0;
    cudaMemcpy(&h_any_change, d_any_change_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_any_change == 0) {
      return false; // No cells need bbox update — skip everything
    }
  }

  // At least one cell needs resize — do the full D→H + CPU decision
  std::vector<int> h_results(num_cells * 9);
  cudaMemcpy(h_results.data(), d_bbox_scan_results,
             num_cells * 9 * sizeof(int), cudaMemcpyDeviceToHost);

  // =========================================================================
  // Decision pass: evaluate all cells, collect pool remaps for batching
  // =========================================================================
  struct RemapWork {
    int cell_idx;
    BoundingBox new_bbox;
    BoundingBox new_bbox_halo;
    int new_size;
  };
  std::vector<RemapWork> pool_remaps;
  std::vector<RemapWork> malloc_remaps;
  bool any_changed = false;

  for (int i = 0; i < num_cells; ++i) {
    auto &cell = domain.cells[i];
    int base = i * 9;
    int max_dist_x = h_results[base + 0];
    int max_dist_y = h_results[base + 1];
    int min_lx = h_results[base + 2];
    int max_lx = h_results[base + 3];
    int min_ly = h_results[base + 4];
    int max_ly = h_results[base + 5];
    int found_any = h_results[base + 6];
    float cx, cy;
    memcpy(&cx, &h_results[base + 7], sizeof(float));
    memcpy(&cy, &h_results[base + 8], sizeof(float));

    if (!found_any) continue;

    // Update centroid on host cell
    cell->centroid.x = cx;
    cell->centroid.y = cy;

    int old_w = cell->width();
    int old_h = cell->height();

    // Compute optimal half-sizes
    int half_w = max_dist_x + adaptive_margin;
    int half_h = max_dist_y + adaptive_margin;
    half_w = std::max(half_w, params.min_subdomain_size / 2);
    half_h = std::max(half_h, params.min_subdomain_size / 2);

    // Cap to theoretical max (prevents runaway growth at high confluence)
    int max_half_cap = static_cast<int>(params.target_radius + 3.0f * params.lambda)
                       + adaptive_margin + 4;
    half_w = std::min(half_w, max_half_cap);
    half_h = std::min(half_h, max_half_cap);

    int new_cx = static_cast<int>(cx);
    int new_cy = static_cast<int>(cy);

    BoundingBox new_bbox = {new_cx - half_w, new_cy - half_h,
                            new_cx + half_w, new_cy + half_h};
    BoundingBox new_bbox_halo = new_bbox.expanded(halo);

    // --- Decision logic with asymmetric hysteresis ---

    // 1. Emergency grow: field approaching bbox boundary
    bool touching_edge = (min_lx <= halo + 1) || (max_lx >= old_w - halo - 2) ||
                         (min_ly <= halo + 1) || (max_ly >= old_h - halo - 2);

    // 2. Opportunistic shrink: optimal bbox significantly smaller
    int new_total = new_bbox_halo.size();
    int old_total = cell->bbox_with_halo.size();
    bool worth_shrinking = (new_total < old_total * 4 / 5); // >20% smaller

    // 3. Center drift
    int old_cx_bb = (cell->bbox_with_halo.x0 + cell->bbox_with_halo.x1) / 2;
    int old_cy_bb = (cell->bbox_with_halo.y0 + cell->bbox_with_halo.y1) / 2;
    int shift_x = new_cx - old_cx_bb;
    int shift_y = new_cy - old_cy_bb;
    if (shift_x > Nx / 2) shift_x -= Nx;
    if (shift_x < -Nx / 2) shift_x += Nx;
    if (shift_y > Ny / 2) shift_y -= Ny;
    if (shift_y < -Ny / 2) shift_y += Ny;
    bool center_drifted = (abs(shift_x) >= 5 || abs(shift_y) >= 5);

    // Only resize if there's a reason
    if (!touching_edge && !worth_shrinking && !center_drifted) continue;

    // When growing due to emergency, add extra 25% overshoot to avoid
    // repeated growth
    if (touching_edge && !worth_shrinking) {
      int overshoot = static_cast<int>(0.25f * adaptive_margin);
      half_w += overshoot;
      half_h += overshoot;
      new_bbox = {new_cx - half_w, new_cy - half_h,
                  new_cx + half_w, new_cy + half_h};
      new_bbox_halo = new_bbox.expanded(halo);
    }

    int new_size = new_bbox_halo.size();

    bool use_pool = (d_phi_pool != nullptr && cell->pool_managed);
    if (use_pool && static_cast<size_t>(new_size) > pool_slot_size) {
      // Fixed-page pool: cap the bbox to fit within the pool slot.
      // This should be extremely rare if compute_max_page_size() is correct.
      // Rather than triggering a pool regrow (which requires full reallocation),
      // we shrink the bbox to the maximum size that fits in the pool slot.
      int max_side = static_cast<int>(sqrtf(static_cast<float>(pool_slot_size)));
      // Ensure even so centering works
      if (max_side % 2 != 0) max_side--;
      int capped_half = max_side / 2;
      new_bbox = {new_cx - capped_half + halo, new_cy - capped_half + halo,
                  new_cx + capped_half - halo, new_cy + capped_half - halo};
      new_bbox_halo = new_bbox.expanded(halo);
      new_size = new_bbox_halo.size();
      // Warn once
      fprintf(stderr, "WARNING: Cell %d bbox capped to %dx%d (%d) to fit pool slot %zu. "
              "Consider increasing safety margin in compute_max_page_size().\n",
              cell->id, new_bbox_halo.width(), new_bbox_halo.height(),
              new_size, pool_slot_size);
    }

    RemapWork work = {i, new_bbox, new_bbox_halo, new_size};
    if (use_pool) {
      pool_remaps.push_back(work);
    } else {
      malloc_remaps.push_back(work);
    }
  }

  // =========================================================================
  // Batched pool remaps: single kernel launch, no per-cell memset
  // =========================================================================
  if (!pool_remaps.empty()) {
    int num_remaps = static_cast<int>(pool_remaps.size());
    std::vector<BboxRemapParams> h_params(num_remaps);
    int max_new_size = 0;

    for (int r = 0; r < num_remaps; ++r) {
      const auto &work = pool_remaps[r];
      auto &cell = domain.cells[work.cell_idx];
      h_params[r].src = cell->d_phi;
      h_params[r].dst = cell->d_dphi_dt;
      h_params[r].old_w = cell->width();
      h_params[r].old_h = cell->height();
      h_params[r].old_ox = cell->bbox_with_halo.x0;
      h_params[r].old_oy = cell->bbox_with_halo.y0;
      h_params[r].new_w = work.new_bbox_halo.width();
      h_params[r].new_h = work.new_bbox_halo.height();
      h_params[r].new_ox = work.new_bbox_halo.x0;
      h_params[r].new_oy = work.new_bbox_halo.y0;
      h_params[r].new_size = work.new_size;
      max_new_size = std::max(max_new_size, work.new_size);
    }

    // Upload remap params to device (persistent grow-only buffer)
    static BboxRemapParams *d_remap_params = nullptr;
    static size_t remap_params_cap = 0;
    if (static_cast<size_t>(num_remaps) > remap_params_cap) {
      if (d_remap_params) cudaFree(d_remap_params);
      remap_params_cap = std::max(static_cast<size_t>(num_remaps),
                                  std::max(remap_params_cap * 2, size_t(16)));
      CUDA_CHECK(cudaMalloc(&d_remap_params, remap_params_cap * sizeof(BboxRemapParams)));
    }
    cudaMemcpy(d_remap_params, h_params.data(),
               num_remaps * sizeof(BboxRemapParams), cudaMemcpyHostToDevice);

    // Single batched remap launch — writes 0 for non-overlapping pixels,
    // eliminating the need for separate cudaMemset before remap.
    int threads = 256;
    int blocks_per_remap = (max_new_size + threads - 1) / threads;
    dim3 remap_grid(blocks_per_remap, num_remaps);
    kernel_bbox_remap_2d_batched<<<remap_grid, threads>>>(
        d_remap_params, Nx, Ny, num_remaps);

    // Host-side pointer swaps and metadata updates for all pool remaps
    for (const auto &work : pool_remaps) {
      auto &cell = domain.cells[work.cell_idx];
      float *old_phi = cell->d_phi;
      cell->d_phi = cell->d_dphi_dt;
      cell->d_dphi_dt = old_phi;
      // No post-swap memset needed: fused kernel overwrites all active pixels
      // in phi_out each step, and pixels beyond field_size are never read.
      cell->field_size = work.new_size;
      cell->bbox = work.new_bbox;
      cell->bbox_with_halo = work.new_bbox_halo;
    }
    any_changed = true;
  }

  // =========================================================================
  // GPU-side device array patching for pool remaps
  // (eliminates update_interaction_arrays → 7× H→D memcpy)
  // =========================================================================
  if (!pool_remaps.empty()) {
    int num_patches = static_cast<int>(pool_remaps.size());
    std::vector<DeviceArrayPatch> h_patches(num_patches);
    for (int r = 0; r < num_patches; ++r) {
      const auto &work = pool_remaps[r];
      auto &cell = domain.cells[work.cell_idx];
      h_patches[r].cell_idx = work.cell_idx;
      h_patches[r].new_phi = cell->d_phi;       // already swapped above
      h_patches[r].new_phi_out = cell->d_dphi_dt;
      h_patches[r].new_w = work.new_bbox_halo.width();
      h_patches[r].new_h = work.new_bbox_halo.height();
      h_patches[r].new_ox = work.new_bbox_halo.x0;
      h_patches[r].new_oy = work.new_bbox_halo.y0;
      h_patches[r].new_field_size = work.new_size;
    }
    // Persistent grow-only patch buffer (shared with malloc remaps below)
    static DeviceArrayPatch *d_patches = nullptr;
    static size_t patch_cap = 0;
    if (static_cast<size_t>(num_patches) > patch_cap) {
      if (d_patches) cudaFree(d_patches);
      patch_cap = std::max(static_cast<size_t>(num_patches),
                           std::max(patch_cap * 2, size_t(16)));
      CUDA_CHECK(cudaMalloc(&d_patches, patch_cap * sizeof(DeviceArrayPatch)));
    }
    cudaMemcpy(d_patches, h_patches.data(),
               num_patches * sizeof(DeviceArrayPatch), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (num_patches + threads - 1) / threads;
    kernel_patch_device_arrays<<<blocks, threads>>>(
        d_all_phi_ptrs, d_all_phi_out_ptrs,
        d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y, d_all_field_sizes,
        d_patches, num_patches);
  }

  // =========================================================================
  // Non-pool remaps: individual cudaMalloc (rare — only on pool overflow)
  // =========================================================================
  for (const auto &work : malloc_remaps) {
    auto &cell = domain.cells[work.cell_idx];
    bool was_pool = cell->pool_managed;
    int new_size = work.new_size;
    int new_w = work.new_bbox_halo.width();
    int new_h = work.new_bbox_halo.height();

    float *d_phi_new = nullptr;
    float *d_dphi_dt_new = nullptr;
    CUDA_CHECK(cudaMalloc(&d_phi_new, new_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dphi_dt_new, new_size * sizeof(float)));
    cudaMemset(d_phi_new, 0, new_size * sizeof(float));
    cudaMemset(d_dphi_dt_new, 0, new_size * sizeof(float));

    int threads = 256;
    int blocks = (new_size + threads - 1) / threads;
    kernel_bbox_remap_2d<<<blocks, threads>>>(
        cell->d_phi, d_phi_new,
        cell->width(), cell->height(),
        cell->bbox_with_halo.x0, cell->bbox_with_halo.y0,
        new_w, new_h,
        work.new_bbox_halo.x0, work.new_bbox_halo.y0,
        Nx, Ny);

    if (!was_pool) {
      cudaFree(cell->d_phi);
      cudaFree(cell->d_dphi_dt);
    }
    cell->d_phi = d_phi_new;
    cell->d_dphi_dt = d_dphi_dt_new;
    cell->pool_managed = false;

    cell->field_size = new_size;
    cell->bbox = work.new_bbox;
    cell->bbox_with_halo = work.new_bbox_halo;
    any_changed = true;
  }

  // GPU-side device array patching for malloc remaps
  if (!malloc_remaps.empty()) {
    int num_patches = static_cast<int>(malloc_remaps.size());
    std::vector<DeviceArrayPatch> h_patches(num_patches);
    for (int r = 0; r < num_patches; ++r) {
      const auto &work = malloc_remaps[r];
      auto &cell = domain.cells[work.cell_idx];
      h_patches[r].cell_idx = work.cell_idx;
      h_patches[r].new_phi = cell->d_phi;
      h_patches[r].new_phi_out = cell->d_dphi_dt;
      h_patches[r].new_w = work.new_bbox_halo.width();
      h_patches[r].new_h = work.new_bbox_halo.height();
      h_patches[r].new_ox = work.new_bbox_halo.x0;
      h_patches[r].new_oy = work.new_bbox_halo.y0;
      h_patches[r].new_field_size = work.new_size;
    }
    // Persistent grow-only patch buffer for malloc remaps
    static DeviceArrayPatch *d_patches_mr = nullptr;
    static size_t patch_mr_cap = 0;
    if (static_cast<size_t>(num_patches) > patch_mr_cap) {
      if (d_patches_mr) cudaFree(d_patches_mr);
      patch_mr_cap = std::max(static_cast<size_t>(num_patches),
                              std::max(patch_mr_cap * 2, size_t(16)));
      CUDA_CHECK(cudaMalloc(&d_patches_mr, patch_mr_cap * sizeof(DeviceArrayPatch)));
    }
    cudaMemcpy(d_patches_mr, h_patches.data(),
               num_patches * sizeof(DeviceArrayPatch), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (num_patches + threads - 1) / threads;
    kernel_patch_device_arrays<<<blocks, threads>>>(
        d_all_phi_ptrs, d_all_phi_out_ptrs,
        d_all_widths, d_all_heights,
        d_all_offsets_x, d_all_offsets_y, d_all_field_sizes,
        d_patches_mr, num_patches);
  }

  return any_changed;
}

} // namespace cellsim
