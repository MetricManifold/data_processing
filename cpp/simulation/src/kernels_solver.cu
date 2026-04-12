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
    float subdomain_padding,
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

  // 2. Volume/centroid computation is done in kernel_fused_step's last-arriving block.
  //    Read centroids from d_centroids_x/y (already updated by fused kernel).
  float cx = centroids_x[i];
  float cy = centroids_y[i];

  // Read second moments. Zero them before accumulation (zero_moments=true
  // on the step BEFORE fused kernel accumulates, so moments persist for
  // the visualizer and resize reads on the following step).
  float moment_x = d_second_moment_x[i];
  float moment_y = d_second_moment_y[i];
  if (zero_moments) {
    d_second_moment_x[i] = 0.0f;
    d_second_moment_y[i] = 0.0f;
  }

  // Zero perimeter accumulator (read by host sync, then cleared for next step's fused kernel)
  d_perimeters[i] = 0.0f;

  // 3. Compute centroid shift + dynamic resize
  // sx/sy encode the mapping: new local (lx,ly) reads from old local (lx+sx, ly+sy)
  int sx = 0, sy = 0;
  int new_w = w, new_h = h;

  // Volume already computed by fused kernel's last block
  float volume = volumes[i];

  if (volume > 1e-8f) {
    if (compute_shifts) {
      // --- Centroid shift (recenter subdomain) ---
      float sub_cx = fmodf((float)old_off_x + (float)w * 0.5f, (float)Nx);
      float sub_cy = fmodf((float)old_off_y + (float)h * 0.5f, (float)Ny);
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
      float sum_phi2 = volume / dA;
      if (max_side > 0 && sum_phi2 > 1.0f) {
        // Correct variance using parallel axis theorem:
        // Var(x) = E[(x-ref)²] - (centroid-ref)²
        // moment_x = Σ (x-ref)² φ², centroid = ref + Σ(x-ref)φ² / Σφ²
        float dx_cent = cx - rx;  // centroid - ref (rx is ref_x for this cell)
        float dy_cent = cy - ry;
        if (dx_cent > Nx * 0.5f) dx_cent -= Nx;
        if (dx_cent < -Nx * 0.5f) dx_cent += Nx;
        if (dy_cent > Ny * 0.5f) dy_cent -= Ny;
        if (dy_cent < -Ny * 0.5f) dy_cent += Ny;
        float var_x = moment_x / sum_phi2 - dx_cent * dx_cent;
        float var_y = moment_y / sum_phi2 - dy_cent * dy_cent;
        // Guard: only resize when moments are meaningful (σ > 2 pixels).
        // On step 0 moments are zero → var ~ 0, skip to avoid catastrophic shrink.
        if (var_x > 4.0f && var_y > 4.0f) {
          float sigma_x = sqrtf(var_x);
          float sigma_y = sqrtf(var_y);
          // Target half-size: 2σ + padding*R
          float R = d_target_radius[i];
          int additive_margin = (int)ceilf(subdomain_padding * R);
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
    int *__restrict__ d_block_arrival_fused,
    float *__restrict__ d_volumes,
    float *__restrict__ d_volume_deviations_out,
    float *__restrict__ d_centroids_x,
    float *__restrict__ d_centroids_y,
    const float *__restrict__ d_target_area,
    float *__restrict__ d_integrals_x_zero,
    float *__restrict__ d_integrals_y_zero,
    int *__restrict__ d_block_arrival_zero,
    float dA);

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
                                      int num_cells, int Nx, int Ny);

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
                   int *d_block_arrival_fused,
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
  size_t smem_fused = (compute_moments ? 6 : 4) * block.x * block.y * sizeof(float);
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
      params.subdomain_padding,
      params.Nx, params.Ny, num_cells);

  // After resize check, read GPU-computed max dims for the FUSED kernel (write path).
  // Velocity_integral and scatter use the OLD grid (max_w/max_h from before pre_step)
  // since they read phi at the old stride.
  dim3 grid_read = grid;  // Save old grid for read-path kernels
  if (pool_max_side > 0 && step_counter % 10 == 0) {
    // Read new max dims from GPU (synchronizing D→H copy, every 10 steps).
    // This stalls the pipeline but is needed for correct grid sizing after resize.
    int h_max_wh[2];
    cudaMemcpy(h_max_wh, d_max_wh, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int new_max_w = h_max_wh[0];
    int new_max_h = h_max_wh[1];
    if (new_max_w > 0 && new_max_h > 0) {
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

  // ===================================================================
  // VELOCITY PRE-PASS: Compute v = v_I + v_A * p from CURRENT phi + sum field.
  // The last-arriving block per cell converts integrals → velocity inline,
  // eliminating the need for a separate kernel_compute_velocities launch.
  // ===================================================================
  if (d_sum_field) {
    kernel_velocity_integral_2d<<<grid_read, block, smem_vint>>>(
        d_all_phi_ptrs, d_old_widths, d_old_heights,
        d_all_offsets_x, d_all_offsets_y,
        d_sum_field,
        d_integrals_x, d_integrals_y,
        d_block_arrival,
        d_velocities_x, d_velocities_y,
        d_polarization_x, d_polarization_y,
        d_v_A, params.motility_coeff(), dA,
        params.dx, params.dy,
        params.halo_width, params.Nx, params.Ny, num_cells);
  }

#ifdef ENABLE_KERNEL_PROFILING
  cudaEventRecord(ev_neighbor);
#endif

  // Launch fused kernel (uses CURRENT velocity for advection)
  // Precompute stencil constants (avoids per-thread FP division)
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
      d_block_arrival_fused,
      d_volumes, d_volume_deviations, d_centroids_x, d_centroids_y,
      d_target_area,
      d_integrals_x, d_integrals_y, d_block_arrival,
      dA);

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

} // namespace cellsim
