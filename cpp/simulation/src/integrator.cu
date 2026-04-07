#include "integrator.cuh"
#include "kernels.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace cellsim {

// Forward declaration: GPU pointer swap + offset apply (defined in kernels_shared.cu)
__global__ void kernel_swap_phi_ptrs(float **phi_ptrs, float **phi_out_ptrs,
                                      int *offsets_x, int *offsets_y,
                                      const int *shift_x, const int *shift_y,
                                      int num_cells);
// Swap-only overload (no offset/dim update)
__global__ void kernel_swap_phi_ptrs(float **phi_ptrs, float **phi_out_ptrs,
                                      int num_cells);

// Forward declaration: remap kernel for dynamic subdomain resize
__global__ void kernel_remap_grow(
    float **__restrict__ phi_ptrs, float **__restrict__ phi_out_ptrs,
    const int *__restrict__ widths, const int *__restrict__ heights,
    const int *__restrict__ grow, int num_cells);
// Forward declaration: apply resize (swap + update dims) after remap
__global__ void kernel_apply_resize(
    float **__restrict__ phi_ptrs, float **__restrict__ phi_out_ptrs,
    int *__restrict__ widths, int *__restrict__ heights,
    int *__restrict__ offsets_x, int *__restrict__ offsets_y,
    const int *__restrict__ grow, int num_cells);

//=============================================================================
// GPU-side RNG Kernels for Polarization Updates
//=============================================================================

// Initialize curand states (called once per cell)
__global__ void kernel_init_rng_states(curandState *states, unsigned long seed,
                                       int num_cells) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_cells)
    return;
  // Each cell gets a unique sequence based on its index
  curand_init(seed, idx, 0, &states[idx]);
}

// Update polarization directions on GPU
// Supports both Run-and-Tumble (discrete reorientations) and ABP (continuous
// diffusion)
__global__ void kernel_update_polarization(
    curandState *__restrict__ rng_states, float *__restrict__ polarization_x,
    float *__restrict__ polarization_y, float *__restrict__ theta,
    float dt, float tau, int motility_model, // 0 = RunAndTumble, 1 = ABP
    int num_cells) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_cells)
    return;

  // Load RNG state to local memory for efficiency
  curandState local_state = rng_states[idx];
  float cell_theta = theta[idx];

  if (motility_model == 0) {
    // Run-and-Tumble: Poisson reorientation events
    // Probability of tumble in dt: P = 1 - exp(-dt/τ)
    float p_tumble = 1.0f - expf(-dt / tau);
    float rand_val = curand_uniform(&local_state);
    if (rand_val < p_tumble) {
      // Pick completely new random direction
      cell_theta = curand_uniform(&local_state) * 2.0f * M_PI;
    }
  } else {
    // Active Brownian Particle: continuous rotational diffusion
    // dθ/dt = η(t) where η is white noise with <η(t)η(t')> = (2/τ)δ(t-t')
    float noise_strength = sqrtf(2.0f * dt / tau);
    // curand_normal gives standard normal N(0,1)
    float dtheta = noise_strength * curand_normal(&local_state);
    cell_theta += dtheta;

    // Keep angle in [0, 2π)
    cell_theta = fmodf(cell_theta, 2.0f * M_PI);
    if (cell_theta < 0)
      cell_theta += 2.0f * M_PI;
  }

  // Update polarization vector
  polarization_x[idx] = cosf(cell_theta);
  polarization_y[idx] = sinf(cell_theta);
  theta[idx] = cell_theta;

  // Save RNG state back
  rng_states[idx] = local_state;
}

//=============================================================================
// Adaptive Neighbor List Caching Kernels
//=============================================================================

// Compute max displacement (squared) since last neighbor rebuild
// Uses parallel reduction to find max across all cells
__global__ void kernel_compute_max_displacement(
    const float *__restrict__ centroids_x,
    const float *__restrict__ centroids_y,
    const float *__restrict__ prev_centroids_x,
    const float *__restrict__ prev_centroids_y,
    float *__restrict__ max_disp_out, int Nx, int Ny, int num_cells) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread computes displacement for its cell
  float my_disp_sq = 0.0f;
  if (idx < num_cells) {
    float dx = centroids_x[idx] - prev_centroids_x[idx];
    float dy = centroids_y[idx] - prev_centroids_y[idx];

    // Periodic wrap for displacement
    if (dx > Nx * 0.5f)
      dx -= Nx;
    else if (dx < -Nx * 0.5f)
      dx += Nx;
    if (dy > Ny * 0.5f)
      dy -= Ny;
    else if (dy < -Ny * 0.5f)
      dy += Ny;

    my_disp_sq = dx * dx + dy * dy;
  }

  sdata[tid] = my_disp_sq;
  __syncthreads();

  // Reduction to find max in block
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  // Block leader writes result
  if (tid == 0) {
    atomicMax((int *)max_disp_out,
              __float_as_int(sdata[0])); // Atomic max for floats via int cast
  }
}

// Copy current centroids to prev_centroids (for next displacement check)
__global__ void kernel_copy_centroids(const float *__restrict__ src_x,
                                      const float *__restrict__ src_y,
                                      float *__restrict__ dst_x,
                                      float *__restrict__ dst_y, int num_cells) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_cells)
    return;
  dst_x[idx] = src_x[idx];
  dst_y[idx] = src_y[idx];
}

//=============================================================================
// Euler Step Kernel
//=============================================================================

__global__ void kernel_euler_step(float *__restrict__ phi,
                                  const float *__restrict__ dphi_dt, int size,
                                  float dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  phi[idx] += dt * dphi_dt[idx];

  // Clamp to [0, 1] to prevent numerical instability
  phi[idx] = fmaxf(0.0f, fminf(1.0f, phi[idx]));
}

//=============================================================================
// Integrator Implementation
//=============================================================================

Integrator::Integrator(Method m)
    : method(m), centroid_sums_ready(false), num_streams(0),
      d_all_phi_ptrs(nullptr), d_all_phi_out_ptrs(nullptr),
      d_all_widths(nullptr), d_all_heights(nullptr),
      d_all_offsets_x(nullptr), d_all_offsets_y(nullptr),
      d_all_field_sizes(nullptr), interaction_array_capacity(0),
      d_volumes(nullptr), d_integrals_x(nullptr), d_integrals_y(nullptr),
      d_centroid_sums(nullptr), d_reduction_block(nullptr),
      d_block_arrival(nullptr),
      reduction_block_floats(0), reduction_array_capacity(0),
      d_second_moment_x(nullptr), d_second_moment_y(nullptr),
      d_shift_x(nullptr), d_shift_y(nullptr),
      d_volume_deviations(nullptr), d_velocities_x(nullptr),
      d_velocities_y(nullptr), d_ref_x(nullptr), d_ref_y(nullptr),
      d_polarization_x(nullptr), d_polarization_y(nullptr), d_theta(nullptr),
      d_v_A(nullptr),
      d_gamma(nullptr),
      d_two_gamma(nullptr), d_two_gamma_bulk(nullptr),
      d_target_radius(nullptr), d_target_area(nullptr), d_volume_coeff(nullptr),
      d_centroids_x(nullptr), d_centroids_y(nullptr),
      d_perimeters(nullptr),
      d_neighbor_counts(nullptr), d_neighbor_lists(nullptr),
      d_rng_states(nullptr), rng_initialized(false),
      d_prev_centroids_x(nullptr), d_prev_centroids_y(nullptr),
      d_max_displacement(nullptr), neighbor_list_valid(false),
      neighbor_rebuild_threshold(5.0f), // Default: rebuild when any cell moves >5 grid units
      neighbor_rebuild_count(0), neighbor_skip_count(0),
      cached_max_size(0), cached_max_w(0), cached_max_h(0),
      cached_dims_valid(false),
      d_phi_pool(nullptr), pool_slot_size(0), pool_num_cells(0),
      pool_max_side(0), pool_active(false), d_grow(nullptr),
      d_old_widths(nullptr), d_old_heights(nullptr), d_max_wh(nullptr),
      d_sum_field(nullptr), d_sum_field_b(nullptr), sum_field_size(0),
      d_sum_field_linear(nullptr), d_sum_field_linear_b(nullptr),
      sum_field_clear_stream(nullptr),
      sum_field_read_done_event(nullptr), sum_field_clear_done_event(nullptr),
      step_counter(0), host_ptrs_stale(false) {
  create_streams();
}

Integrator::~Integrator() {
  free_sum_field();
  free_phi_pool();
  free_interaction_arrays();
  free_reduction_arrays();
  destroy_streams();
}

void Integrator::create_streams(int n) {
  destroy_streams();
  num_streams = std::min(n, MAX_STREAMS);
  streams.resize(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    cudaStreamCreate(&streams[i]);
  }
}

void Integrator::destroy_streams() {
  for (auto &s : streams) {
    cudaStreamDestroy(s);
  }
  streams.clear();
  num_streams = 0;
}

void Integrator::allocate_interaction_arrays(int num_cells) {
  if (num_cells <= static_cast<int>(interaction_array_capacity)) {
    return; // Already have enough capacity
  }

  free_interaction_arrays();

  // Allocate with some headroom to avoid frequent reallocation
  size_t new_capacity =
      std::max(static_cast<size_t>(num_cells), interaction_array_capacity * 2);
  new_capacity = std::max(new_capacity, static_cast<size_t>(16));

  cudaMalloc(&d_all_phi_ptrs, new_capacity * sizeof(float *));
  cudaMalloc(&d_all_phi_out_ptrs, new_capacity * sizeof(float *));
  cudaMalloc(&d_all_widths, new_capacity * sizeof(int));
  cudaMalloc(&d_all_heights, new_capacity * sizeof(int));
  cudaMalloc(&d_all_offsets_x, new_capacity * sizeof(int));
  cudaMalloc(&d_all_offsets_y, new_capacity * sizeof(int));
  cudaMalloc(&d_all_field_sizes, new_capacity * sizeof(int));

  interaction_array_capacity = new_capacity;
}

void Integrator::free_interaction_arrays() {
  if (d_all_phi_ptrs) {
    cudaFree(d_all_phi_ptrs);
    d_all_phi_ptrs = nullptr;
  }
  if (d_all_phi_out_ptrs) {
    cudaFree(d_all_phi_out_ptrs);
    d_all_phi_out_ptrs = nullptr;
  }
  if (d_all_widths) {
    cudaFree(d_all_widths);
    d_all_widths = nullptr;
  }
  if (d_all_heights) {
    cudaFree(d_all_heights);
    d_all_heights = nullptr;
  }
  if (d_all_offsets_x) {
    cudaFree(d_all_offsets_x);
    d_all_offsets_x = nullptr;
  }
  if (d_all_offsets_y) {
    cudaFree(d_all_offsets_y);
    d_all_offsets_y = nullptr;
  }
  if (d_all_field_sizes) {
    cudaFree(d_all_field_sizes);
    d_all_field_sizes = nullptr;
  }
  interaction_array_capacity = 0;
}

void Integrator::update_interaction_arrays(const Domain &domain) {
  int n = domain.num_cells();
  allocate_interaction_arrays(n);

  // Collect host data
  std::vector<float *> h_phi_ptrs(n);
  std::vector<int> h_widths(n), h_heights(n), h_offsets_x(n), h_offsets_y(n);
  std::vector<int> h_field_sizes(n);

  for (int i = 0; i < n; ++i) {
    h_phi_ptrs[i] = domain.cells[i]->d_phi;
    h_widths[i] = domain.cells[i]->width();
    h_heights[i] = domain.cells[i]->height();
    h_offsets_x[i] = domain.cells[i]->bbox_with_halo.x0;
    h_offsets_y[i] = domain.cells[i]->bbox_with_halo.y0;
    h_field_sizes[i] = domain.cells[i]->field_size;
  }

  // Upload to device (single batch copy is fast)
  cudaMemcpy(d_all_phi_ptrs, h_phi_ptrs.data(), n * sizeof(float *),
             cudaMemcpyHostToDevice);

  // Also set up phi output pointers (double buffer: output -> d_dphi_dt)
  std::vector<float *> h_phi_out_ptrs(n);
  for (int i = 0; i < n; ++i) {
    h_phi_out_ptrs[i] = domain.cells[i]->d_dphi_dt;
  }
  cudaMemcpy(d_all_phi_out_ptrs, h_phi_out_ptrs.data(), n * sizeof(float *),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_widths, h_widths.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_heights, h_heights.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_offsets_x, h_offsets_x.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_offsets_y, h_offsets_y.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_field_sizes, h_field_sizes.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);

  // Initialize old_widths/heights to match current (for fused kernel's first step)
  if (d_old_widths && d_old_heights) {
    cudaMemcpy(d_old_widths, h_widths.data(), n * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_heights, h_heights.data(), n * sizeof(int),
               cudaMemcpyHostToDevice);
  }
}

size_t Integrator::compute_max_page_size(const SimParams &params) {
  // Compute the theoretical maximum bbox size any cell can ever have.
  // This is determined by the physics parameters and bbox update logic:
  //   half_size = max_dist + adaptive_margin + overshoot
  //   max_dist  = R + 3*lambda  (phi > threshold extent)
  //   adaptive_margin = int(3*lambda) + halo   *** MUST match kernels_solver.cu ***
  //   overshoot = int(0.25 * adaptive_margin)  (emergency grow case)
  //   bbox_with_halo adds +halo to each side
  // We add an extra safety margin of 10 pixels.
  //
  // When adhesion is active (J > 0), the effective interface stiffness at
  // shared contacts is reduced from gamma to gamma - J/2, widening the
  // interface: lambda_eff = lambda * sqrt(gamma / (gamma - J/2)).
  // Use lambda_eff in place of lambda to guarantee the pool slot is large
  // enough for adhesion-widened interfaces.
  float lambda_eff = params.lambda;
  if (params.adhesion_J > 0.0f && params.gamma > params.adhesion_J / 2.0f) {
    lambda_eff = params.lambda * sqrtf(params.gamma / (params.gamma - params.adhesion_J / 2.0f));
  }
  int halo = params.halo_width;
  int adaptive_margin = static_cast<int>(3.0f * lambda_eff) + halo;
  int overshoot = static_cast<int>(0.25f * adaptive_margin);
  int max_dist = static_cast<int>(params.target_radius + 3.0f * lambda_eff) + 1;
  int max_half = max_dist + adaptive_margin + overshoot + 10; // safety
  int max_side = 2 * max_half + 2 * halo; // +halo each side from bbox.expanded(halo)
  return static_cast<size_t>(max_side) * max_side;
}

void Integrator::allocate_phi_pool(Domain &domain) {
  int num_cells = domain.num_cells();
  if (num_cells == 0) return;

  // Fixed page size: computed from physics parameters, never changes.
  // Every cell gets the same slot size = maximum possible bbox area.
  // This eliminates pool regrow entirely — no reallocation ever needed.
  size_t max_page = compute_max_page_size(domain.params);

  // Ensure page is at least as large as current max field_size
  for (int i = 0; i < num_cells; ++i) {
    max_page = std::max(max_page, static_cast<size_t>(domain.cells[i]->field_size));
  }

  if (pool_active && num_cells == pool_num_cells && max_page <= pool_slot_size) {
    return; // Pool is already allocated with sufficient page size
  }

  // Need to (re)allocate — sync first to ensure no in-flight kernels are
  // reading/writing the old pool (e.g. batched remap or patch kernels).
  bool was_active = pool_active;
  if (was_active) {
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float *old_pool = d_phi_pool;

  pool_slot_size = max_page;
  pool_num_cells = num_cells;

  size_t total_floats = 2 * static_cast<size_t>(num_cells) * pool_slot_size;
  size_t alloc_bytes = total_floats * sizeof(float);

  // Report VRAM before allocation
  {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Phi pool %s: %d cells x %zu slot (fixed page) = %.1f MB  "
           "(VRAM: %.1f MB free / %.1f MB total)\n",
           was_active ? "REALLOC" : "INIT", num_cells, pool_slot_size,
           alloc_bytes / (1024.0 * 1024.0),
           free_mem / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0));
  }

  CUDA_CHECK(cudaMalloc(&d_phi_pool, alloc_bytes));
  CUDA_CHECK(cudaMemset(d_phi_pool, 0, alloc_bytes));

  // Compute max side length any cell can grow to within the slot
  pool_max_side = static_cast<int>(sqrtf(static_cast<float>(pool_slot_size)));

  // Allocate grow buffer for dynamic resize
  if (d_grow) cudaFree(d_grow);
  CUDA_CHECK(cudaMalloc(&d_grow, num_cells * 4 * sizeof(int)));
  if (d_old_widths) cudaFree(d_old_widths);
  if (d_old_heights) cudaFree(d_old_heights);
  CUDA_CHECK(cudaMalloc(&d_old_widths, num_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_old_heights, num_cells * sizeof(int)));
  if (d_max_wh) cudaFree(d_max_wh);
  CUDA_CHECK(cudaMalloc(&d_max_wh, 2 * sizeof(int)));

  // Migrate existing phi data into pool
  for (int i = 0; i < num_cells; ++i) {
    auto &cell = domain.cells[i];
    float *pool_phi = d_phi_pool + static_cast<size_t>(i) * pool_slot_size;
    float *pool_out = d_phi_pool + static_cast<size_t>(num_cells + i) * pool_slot_size;

    // Copy current phi data into pool slot
    if (cell->d_phi && cell->field_size > 0) {
      CUDA_CHECK(cudaMemcpy(pool_phi, cell->d_phi, cell->field_size * sizeof(float),
                 cudaMemcpyDeviceToDevice));
    }

    // Free old individual allocation (only if not already pool-managed)
    if (!cell->pool_managed) {
      if (cell->d_phi) cudaFree(cell->d_phi);
      if (cell->d_dphi_dt) cudaFree(cell->d_dphi_dt);
    }

    // Point cell into pool
    cell->d_phi = pool_phi;
    cell->d_dphi_dt = pool_out;
    cell->pool_managed = true;
  }

  // Free old pool if we were reallocating
  // Sync first: D2D copies above are async in the default stream
  if (was_active && old_pool) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(old_pool));
  }

  pool_active = true;

  // ---- Phi integrity check: verify at least one cell has non-zero phi ----
  // Catches silent D2D copy failures that leave phi at zero from cudaMemset.
  {
    bool any_nonzero = false;
    std::vector<float> probe(1);
    for (int i = 0; i < num_cells && !any_nonzero; ++i) {
      auto &cell = domain.cells[i];
      if (cell->field_size > 0) {
        // Sample the center of the inner region (avoid halo which is zeroed)
        int w = cell->width();
        int h = cell->height();
        int hw = domain.params.halo_width;
        size_t center = static_cast<size_t>((h / 2)) * w + (w / 2);
        if (center < static_cast<size_t>(cell->field_size)) {
          CUDA_CHECK(cudaMemcpy(probe.data(), cell->d_phi + center,
                     sizeof(float), cudaMemcpyDeviceToHost));
          if (probe[0] != 0.0f) any_nonzero = true;
        }
      }
    }
    if (!any_nonzero && num_cells > 0) {
      fprintf(stderr,
              "PHI INTEGRITY FAILURE: All cells have zero phi after pool "
              "migration! Possible silent CUDA error during D2D copy.\n");
      // Check for sticky CUDA errors
      cudaError_t last_err = cudaGetLastError();
      if (last_err != cudaSuccess) {
        fprintf(stderr, "  Sticky CUDA error: %s\n",
                cudaGetErrorString(last_err));
      }
    }
  }

  // Pool (re)allocation changed all cell d_phi/d_dphi_dt pointers —
  // mark device arrays dirty so update_interaction_arrays runs before
  // any kernel reads d_all_phi_ptrs.
  domain.device_arrays_dirty = true;
  cached_dims_valid = false;
  centroid_sums_ready = false;
  host_ptrs_stale = false;  // Host pointers were just set by this function
}

void Integrator::free_phi_pool() {
  if (d_phi_pool) {
    cudaFree(d_phi_pool);
    d_phi_pool = nullptr;
  }
  if (d_grow) {
    cudaFree(d_grow);
    d_grow = nullptr;
  }
  if (d_old_widths) { cudaFree(d_old_widths); d_old_widths = nullptr; }
  if (d_old_heights) { cudaFree(d_old_heights); d_old_heights = nullptr; }
  if (d_max_wh) { cudaFree(d_max_wh); d_max_wh = nullptr; }
  pool_slot_size = 0;
  pool_num_cells = 0;
  pool_max_side = 0;
  pool_active = false;
}

void Integrator::allocate_sum_field(const Domain &domain) {
  size_t needed = static_cast<size_t>(domain.params.Nx) * domain.params.Ny;
  if (needed == sum_field_size && d_sum_field) return;  // Already correct size
  free_sum_field();
  sum_field_size = needed;
  // Ping-pong: allocate two sum fields so we can async-clear the just-read
  // buffer while the next step’s scatter writes to the other one.
  CUDA_CHECK(cudaMalloc(&d_sum_field, sum_field_size * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_sum_field, 0, sum_field_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sum_field_b, sum_field_size * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_sum_field_b, 0, sum_field_size * sizeof(float)));
  // Async stream + events for background sum field clearing
  cudaStreamCreate(&sum_field_clear_stream);
  cudaEventCreateWithFlags(&sum_field_read_done_event, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&sum_field_clear_done_event, cudaEventDisableTiming);

  // Adhesion linear sum field: only allocate when J > 0 (zero overhead otherwise)
  if (domain.params.adhesion_J > 0.0f) {
    CUDA_CHECK(cudaMalloc(&d_sum_field_linear, sum_field_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_field_linear, 0, sum_field_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_field_linear_b, sum_field_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_field_linear_b, 0, sum_field_size * sizeof(float)));
    printf("Sum field (linear, adhesion J=%.4f): %.1f MB (x2 ping-pong)\n",
           domain.params.adhesion_J,
           sum_field_size * sizeof(float) / (1024.0 * 1024.0));
  }

  printf("Sum field (quadratic): %d x %d = %.1f MB (x2 ping-pong)\n",
         domain.params.Nx, domain.params.Ny,
         sum_field_size * sizeof(float) / (1024.0 * 1024.0));
}

void Integrator::free_sum_field() {
  if (d_sum_field) { cudaFree(d_sum_field); d_sum_field = nullptr; }
  if (d_sum_field_b) { cudaFree(d_sum_field_b); d_sum_field_b = nullptr; }
  if (d_sum_field_linear) { cudaFree(d_sum_field_linear); d_sum_field_linear = nullptr; }
  if (d_sum_field_linear_b) { cudaFree(d_sum_field_linear_b); d_sum_field_linear_b = nullptr; }
  if (sum_field_clear_stream) { cudaStreamDestroy(sum_field_clear_stream); sum_field_clear_stream = nullptr; }
  if (sum_field_read_done_event) { cudaEventDestroy(sum_field_read_done_event); sum_field_read_done_event = nullptr; }
  if (sum_field_clear_done_event) { cudaEventDestroy(sum_field_clear_done_event); sum_field_clear_done_event = nullptr; }
  sum_field_size = 0;
}

void Integrator::grow_phi_pool(Domain &domain) {
  // With fixed-page allocation, pool growth should NEVER be needed.
  // If we get here, it means a cell's bbox exceeded the theoretical maximum,
  // which indicates a bug in compute_max_page_size() or bbox update logic.
  fprintf(stderr,
          "FATAL: Pool grow requested but pool uses fixed pages (slot=%zu). "
          "This should never happen — max page size computation is wrong.\n",
          pool_slot_size);
  // Find the offending cell
  for (int i = 0; i < domain.num_cells(); ++i) {
    auto &cell = domain.cells[i];
    if (static_cast<size_t>(cell->field_size) > pool_slot_size) {
      fprintf(stderr, "  Cell %d: field_size=%d, bbox=%dx%d, pool_slot=%zu\n",
              cell->id, cell->field_size, cell->width(), cell->height(),
              pool_slot_size);
    }
  }
  // Abort — this is a critical logic error, not a recoverable situation
  abort();
}

void Integrator::allocate_reduction_arrays(int num_cells) {
  if (static_cast<size_t>(num_cells) <= reduction_array_capacity) {
    return;
  }

  free_reduction_arrays();

  size_t new_capacity =
      std::max(static_cast<size_t>(num_cells), reduction_array_capacity * 2);
  new_capacity = std::max(new_capacity, static_cast<size_t>(16));

  CUDA_CHECK(cudaMalloc(&d_volumes, new_capacity * sizeof(float)));

  // Allocate reduction arrays as one contiguous block for single-memset zeroing.
  // Layout: [integrals_x (N) | integrals_y (N) | perimeters (N) | block_arrival (N) | moment_x (N) | moment_y (N) | centroid_sums (3N)]
  reduction_block_floats = 9 * new_capacity;
  CUDA_CHECK(cudaMalloc(&d_reduction_block, reduction_block_floats * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_reduction_block, 0, reduction_block_floats * sizeof(float)));
  d_integrals_x      = d_reduction_block;
  d_integrals_y      = d_reduction_block + new_capacity;
  d_perimeters       = d_reduction_block + 2 * new_capacity;
  d_block_arrival    = reinterpret_cast<int*>(d_reduction_block + 3 * new_capacity);
  d_second_moment_x  = d_reduction_block + 4 * new_capacity;
  d_second_moment_y  = d_reduction_block + 5 * new_capacity;
  d_centroid_sums    = d_reduction_block + 6 * new_capacity;

  // Additional arrays for GPU-side computation
  CUDA_CHECK(cudaMalloc(&d_volume_deviations, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_velocities_x, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_velocities_y, new_capacity * sizeof(float)));

  // Zero-init: first step uses v=0 for advection before velocity integral runs
  cudaMemset(d_velocities_x, 0, new_capacity * sizeof(float));
  cudaMemset(d_velocities_y, 0, new_capacity * sizeof(float));
  // Shift arrays for inline remap (zero = no shift on first step)
  CUDA_CHECK(cudaMalloc(&d_shift_x, new_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_shift_y, new_capacity * sizeof(int)));
  cudaMemset(d_shift_x, 0, new_capacity * sizeof(int));
  cudaMemset(d_shift_y, 0, new_capacity * sizeof(int));
  CUDA_CHECK(cudaMalloc(&d_ref_x, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ref_y, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_polarization_x, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_polarization_y, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_theta, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v_A, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_two_gamma, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_two_gamma_bulk, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target_radius, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target_area, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_volume_coeff, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_centroids_x, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_centroids_y, new_capacity * sizeof(float)));
  // d_perimeters is already allocated as part of d_reduction_block above

  // Neighbor list arrays for V4 optimization
  CUDA_CHECK(cudaMalloc(&d_neighbor_counts, new_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_neighbor_lists, MAX_NEIGHBORS * new_capacity * sizeof(int)));

  // GPU-side RNG states for polarization updates
  CUDA_CHECK(cudaMalloc(&d_rng_states, new_capacity * sizeof(curandState)));
  rng_initialized = false; // Need to reinitialize after reallocation

  // Adaptive neighbor list caching arrays
  CUDA_CHECK(cudaMalloc(&d_prev_centroids_x, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_prev_centroids_y, new_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_max_displacement, sizeof(float))); // Single value for reduction
  neighbor_list_valid = false; // Force rebuild on first use

  reduction_array_capacity = new_capacity;
}

void Integrator::free_reduction_arrays() {
  if (d_volumes) {
    cudaFree(d_volumes);
    d_volumes = nullptr;
  }
  // d_integrals_x/y, d_perimeters, d_block_arrival, d_second_moment_x/y, d_centroid_sums are aliases into d_reduction_block
  if (d_reduction_block) {
    cudaFree(d_reduction_block);
    d_reduction_block = nullptr;
    d_integrals_x = nullptr;
    d_integrals_y = nullptr;
    d_perimeters = nullptr;
    d_block_arrival = nullptr;
    d_second_moment_x = nullptr;
    d_second_moment_y = nullptr;
    d_centroid_sums = nullptr;
  }
  // Free persistent kernel arrays
  if (d_volume_deviations) {
    cudaFree(d_volume_deviations);
    d_volume_deviations = nullptr;
  }
  if (d_velocities_x) {
    cudaFree(d_velocities_x);
    d_velocities_x = nullptr;
  }
  if (d_velocities_y) {
    cudaFree(d_velocities_y);
    d_velocities_y = nullptr;
  }
  if (d_shift_x) { cudaFree(d_shift_x); d_shift_x = nullptr; }
  if (d_shift_y) { cudaFree(d_shift_y); d_shift_y = nullptr; }
  if (d_ref_x) {
    cudaFree(d_ref_x);
    d_ref_x = nullptr;
  }
  if (d_ref_y) {
    cudaFree(d_ref_y);
    d_ref_y = nullptr;
  }
  if (d_polarization_x) {
    cudaFree(d_polarization_x);
    d_polarization_x = nullptr;
  }
  if (d_polarization_y) {
    cudaFree(d_polarization_y);
    d_polarization_y = nullptr;
  }
  if (d_centroids_x) {
    cudaFree(d_centroids_x);
    d_centroids_x = nullptr;
  }
  if (d_centroids_y) {
    cudaFree(d_centroids_y);
    d_centroids_y = nullptr;
  }
  if (d_neighbor_counts) {
    cudaFree(d_neighbor_counts);
    d_neighbor_counts = nullptr;
  }
  if (d_neighbor_lists) {
    cudaFree(d_neighbor_lists);
    d_neighbor_lists = nullptr;
  }
  if (d_theta) {
    cudaFree(d_theta);
    d_theta = nullptr;
  }
  if (d_v_A) {
    cudaFree(d_v_A);
    d_v_A = nullptr;
  }
  if (d_gamma) {
    cudaFree(d_gamma);
    d_gamma = nullptr;
  }
  if (d_two_gamma) { cudaFree(d_two_gamma); d_two_gamma = nullptr; }
  if (d_two_gamma_bulk) { cudaFree(d_two_gamma_bulk); d_two_gamma_bulk = nullptr; }
  if (d_target_radius) {
    cudaFree(d_target_radius);
    d_target_radius = nullptr;
  }
  if (d_target_area) {
    cudaFree(d_target_area);
    d_target_area = nullptr;
  }
  if (d_volume_coeff) {
    cudaFree(d_volume_coeff);
    d_volume_coeff = nullptr;
  }
  if (d_rng_states) {
    cudaFree(d_rng_states);
    d_rng_states = nullptr;
  }
  rng_initialized = false;
  
  // Free adaptive neighbor list caching arrays
  if (d_prev_centroids_x) {
    cudaFree(d_prev_centroids_x);
    d_prev_centroids_x = nullptr;
  }
  if (d_prev_centroids_y) {
    cudaFree(d_prev_centroids_y);
    d_prev_centroids_y = nullptr;
  }
  if (d_max_displacement) {
    cudaFree(d_max_displacement);
    d_max_displacement = nullptr;
  }
  neighbor_list_valid = false;
  
  reduction_array_capacity = 0;
}

void Integrator::step(Domain &domain, float dt, bool sync_polarization_to_host, bool sync_centroids_to_host) {
  if (domain.num_cells() == 0)
    return;

#ifdef ENABLE_STEP_PROFILING
  static auto t_start = std::chrono::high_resolution_clock::now();
  static double acc_alloc = 0, acc_dims = 0, acc_polar = 0, acc_sumfield = 0;
  static double acc_step_fused = 0, acc_clear = 0, acc_bbox = 0, acc_swap = 0, acc_total = 0;
  static int prof_count = 0;
  auto tp0 = std::chrono::high_resolution_clock::now();
#define PROF_LAP(var) { auto _now = std::chrono::high_resolution_clock::now(); \
  var += std::chrono::duration<double, std::micro>(_now - tp0).count(); tp0 = _now; }
#endif

  // Ensure buffers are allocated
  allocate_reduction_arrays(domain.num_cells());

  // Allocate contiguous phi pool (migrates cells on first call, no-op after)
  allocate_phi_pool(domain);

  // Allocate global sum field for O(1) interaction reads (first call only)
  allocate_sum_field(domain);

  // Always update interaction arrays on first call or if domain changed
  // (interaction_array_capacity == 0 means first call)
  if (domain.device_arrays_dirty || interaction_array_capacity == 0) {
    update_interaction_arrays(domain);
    domain.device_arrays_dirty = false; // Clear after updating
    // Update cached max dimensions whenever arrays are re-uploaded
    cached_dims_valid = false;
  }

  // Update cached max dimensions if needed (avoids host loop every step)
  if (!cached_dims_valid) {
    int nc = domain.num_cells();
    cached_max_size = 0;
    cached_max_w = 0;
    cached_max_h = 0;
    for (int i = 0; i < nc; ++i) {
      cached_max_size = std::max(cached_max_size, domain.cells[i]->field_size);
      cached_max_w = std::max(cached_max_w, domain.cells[i]->width());
      cached_max_h = std::max(cached_max_h, domain.cells[i]->height());
    }
    cached_dims_valid = true;
  }

  const SimParams &params = domain.params;
  int num_cells = domain.num_cells();

#ifdef ENABLE_STEP_PROFILING
  PROF_LAP(acc_alloc);
#endif

  // Initialize GPU RNG states if needed (first call or after reallocation)
  if (!rng_initialized) {
    // Upload initial theta values from cells
    std::vector<float> h_theta(num_cells);
    std::vector<float> h_pol_x(num_cells), h_pol_y(num_cells);
    for (int i = 0; i < num_cells; ++i) {
      h_theta[i] = domain.cells[i]->theta;
      h_pol_x[i] = domain.cells[i]->polarization.x;
      h_pol_y[i] = domain.cells[i]->polarization.y;
    }
    cudaMemcpy(d_theta, h_theta.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_polarization_x, h_pol_x.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_polarization_y, h_pol_y.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Initialize per-cell v_A values
    // Priority: checkpoint values > freshly generated
    {
      std::vector<float> h_v_A(num_cells);
      if (!checkpoint_v_A.empty() && 
          static_cast<int>(checkpoint_v_A.size()) == num_cells) {
        // Use values loaded from checkpoint (preserves quenched disorder)
        h_v_A = checkpoint_v_A;
        printf("Per-cell v_A: restored %d values from checkpoint\n", num_cells);
        // Print summary
        float v_min = *std::min_element(h_v_A.begin(), h_v_A.end());
        float v_max = *std::max_element(h_v_A.begin(), h_v_A.end());
        float v_sum = 0;
        for (float v : h_v_A) v_sum += v;
        printf("  Restored range: [%.4f, %.4f], mean=%.4f\n",
               v_min, v_max, v_sum / num_cells);
        checkpoint_v_A.clear(); // Free memory, used only once
      } else if (params.v_A_sigma > 0.0f && params.v_A > 0.0f) {
        // Log-normal: mean = params.v_A, std = params.v_A_sigma
        // For log-normal with desired mean m and std s:
        //   mu_ln = ln(m) - 0.5 * sigma_ln^2
        //   sigma_ln = sqrt(ln(1 + (s/m)^2))
        float cv = params.v_A_sigma / params.v_A; // coefficient of variation
        float sigma_ln = sqrtf(logf(1.0f + cv * cv));
        float mu_ln = logf(params.v_A) - 0.5f * sigma_ln * sigma_ln;
        printf("Per-cell v_A: log-normal(mean=%.4f, sigma=%.4f)\n",
               params.v_A, params.v_A_sigma);
        printf("  Log-normal params: mu_ln=%.4f, sigma_ln=%.4f\n",
               mu_ln, sigma_ln);
        for (int i = 0; i < num_cells; ++i) {
          // Box-Muller for normal sample
          float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
          float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
          float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
          h_v_A[i] = expf(mu_ln + sigma_ln * z);
        }
        // Print summary
        float v_min = *std::min_element(h_v_A.begin(), h_v_A.end());
        float v_max = *std::max_element(h_v_A.begin(), h_v_A.end());
        float v_sum = 0;
        for (float v : h_v_A) v_sum += v;
        printf("  Actual range: [%.4f, %.4f], mean=%.4f\n",
               v_min, v_max, v_sum / num_cells);
      } else {
        // Uniform: all cells get params.v_A
        for (int i = 0; i < num_cells; ++i) {
          h_v_A[i] = params.v_A;
        }
      }
      cudaMemcpy(d_v_A, h_v_A.data(), num_cells * sizeof(float),
                 cudaMemcpyHostToDevice);
    }

    // Initialize per-cell gamma values
    // Priority: gamma_overrides APPLIED ON TOP OF checkpoint values > legacy soft_cell_id > uniform gamma
    {
      std::vector<float> h_gamma(num_cells);
      bool loaded_from_checkpoint = false;
      if (!checkpoint_gamma.empty() &&
          static_cast<int>(checkpoint_gamma.size()) == num_cells) {
        h_gamma = checkpoint_gamma;
        loaded_from_checkpoint = true;
        printf("Per-cell gamma: restored %d values from checkpoint\n", num_cells);
        float g_min = *std::min_element(h_gamma.begin(), h_gamma.end());
        float g_max = *std::max_element(h_gamma.begin(), h_gamma.end());
        printf("  Restored range: [%.4f, %.4f]\n", g_min, g_max);
        checkpoint_gamma.clear();
      } else if (!gamma_overrides_set) {
        // No checkpoint and no overrides: use uniform or legacy
        if (params.soft_cell_id >= 0 && params.soft_cell_id < num_cells) {
          for (int i = 0; i < num_cells; ++i) {
            h_gamma[i] = params.gamma;
          }
          h_gamma[params.soft_cell_id] = params.gamma_soft;
          printf("Per-cell gamma: cell %d is soft (gamma=%.4f), rest normal (gamma=%.4f)\n",
                 params.soft_cell_id, params.gamma_soft, params.gamma);
        } else {
          for (int i = 0; i < num_cells; ++i) {
            h_gamma[i] = params.gamma;
          }
        }
      } else {
        // gamma_overrides_set but no checkpoint: start from uniform base
        if (!loaded_from_checkpoint) {
          for (int i = 0; i < num_cells; ++i) {
            h_gamma[i] = params.gamma;
          }
        }
      }

      // Apply gamma overrides ON TOP of whatever base was loaded
      // (checkpoint values, uniform, or legacy)
      if (gamma_overrides_set) {
        std::vector<bool> assigned(num_cells, false);

        // Pass 1: fraction-based overrides (random selection)
        for (const auto &ov : gamma_overrides) {
          if (ov.type != SimParams::GammaOverride::Type::Fraction) continue;
          int count = static_cast<int>(ov.fraction * num_cells + 0.5f);
          std::vector<int> eligible;
          eligible.reserve(num_cells);
          for (int i = 0; i < num_cells; ++i) {
            if (!assigned[i]) eligible.push_back(i);
          }
          int n = std::min(count, static_cast<int>(eligible.size()));
          for (int i = 0; i < n; ++i) {
            int j = i + rand() % (static_cast<int>(eligible.size()) - i);
            std::swap(eligible[i], eligible[j]);
          }
          for (int i = 0; i < n; ++i) {
            h_gamma[eligible[i]] = ov.value;
            assigned[eligible[i]] = true;
          }
          printf("Per-cell gamma: %.1f%% (%d cells) set to %.4f\n",
                 ov.fraction * 100.0f, n, ov.value);
        }

        // Pass 2: cell-specific overrides (highest priority)
        for (const auto &ov : gamma_overrides) {
          if (ov.type != SimParams::GammaOverride::Type::Cells) continue;
          for (int id : ov.cell_ids) {
            if (id >= 0 && id < num_cells) {
              h_gamma[id] = ov.value;
              assigned[id] = true;
            } else {
              printf("Warning: --gamma cell ID %d out of range [0, %d)\n",
                     id, num_cells);
            }
          }
          printf("Per-cell gamma: %d specific cell(s) set to %.4f\n",
                 static_cast<int>(ov.cell_ids.size()), ov.value);
        }

        float g_min = *std::min_element(h_gamma.begin(), h_gamma.end());
        float g_max = *std::max_element(h_gamma.begin(), h_gamma.end());
        printf("Per-cell gamma: final range [%.4f, %.4f]\n", g_min, g_max);
      }

      cudaMemcpy(d_gamma, h_gamma.data(), num_cells * sizeof(float),
                 cudaMemcpyHostToDevice);

      // Precompute derived arrays for fused kernel
      float bc = params.bulk_coeff();
      std::vector<float> h_two_gamma(num_cells), h_two_gamma_bulk(num_cells);
      for (int i = 0; i < num_cells; ++i) {
        h_two_gamma[i] = 2.0f * h_gamma[i];
        h_two_gamma_bulk[i] = 2.0f * h_gamma[i] * bc;
      }
      cudaMemcpy(d_two_gamma, h_two_gamma.data(), num_cells * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(d_two_gamma_bulk, h_two_gamma_bulk.data(), num_cells * sizeof(float),
                 cudaMemcpyHostToDevice);
    }

    // Initialize per-cell target radius and derived arrays (target_area, volume_coeff)
    // Priority: checkpoint values > radius_overrides > uniform params.target_radius
    {
      std::vector<float> h_radius(num_cells);
      if (!checkpoint_target_radius.empty() &&
          static_cast<int>(checkpoint_target_radius.size()) == num_cells) {
        h_radius = checkpoint_target_radius;
        printf("Per-cell radius: restored %d values from checkpoint\n", num_cells);
        float r_min = *std::min_element(h_radius.begin(), h_radius.end());
        float r_max = *std::max_element(h_radius.begin(), h_radius.end());
        printf("  Restored range: [%.2f, %.2f]\n", r_min, r_max);
        checkpoint_target_radius.clear();
      } else if (radius_overrides_set) {
        // Start with base radius for all cells
        for (int i = 0; i < num_cells; ++i) {
          h_radius[i] = params.target_radius;
        }
        std::vector<bool> assigned(num_cells, false);

        // Pass 1: CV type — Gaussian distribution (overrides all cells)
        for (const auto &ov : radius_overrides) {
          if (ov.type != SimParams::RadiusOverride::Type::CV) continue;
          float mean = ov.value;
          float sigma = mean * ov.cv;
          printf("Per-cell radius: Gaussian(mean=%.2f, CV=%.2f, sigma=%.2f)\n",
                 mean, ov.cv, sigma);
          for (int i = 0; i < num_cells; ++i) {
            float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
            float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
            float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            float r = mean + sigma * z;
            // Clamp to [0.5*mean, 1.5*mean] to avoid pathological sizes
            r = fmaxf(0.5f * mean, fminf(1.5f * mean, r));
            h_radius[i] = r;
            assigned[i] = true;
          }
        }

        // Pass 2: Fraction-based overrides (random selection)
        for (const auto &ov : radius_overrides) {
          if (ov.type != SimParams::RadiusOverride::Type::Fraction) continue;
          int count = static_cast<int>(ov.fraction * num_cells + 0.5f);
          std::vector<int> eligible;
          for (int i = 0; i < num_cells; ++i) {
            if (!assigned[i]) eligible.push_back(i);
          }
          int n = std::min(count, static_cast<int>(eligible.size()));
          for (int i = 0; i < n; ++i) {
            int j = i + rand() % (static_cast<int>(eligible.size()) - i);
            std::swap(eligible[i], eligible[j]);
          }
          for (int i = 0; i < n; ++i) {
            h_radius[eligible[i]] = ov.value;
            assigned[eligible[i]] = true;
          }
          printf("Per-cell radius: %.1f%% (%d cells) set to %.2f\n",
                 ov.fraction * 100.0f, n, ov.value);
        }

        // Pass 3: Cell-specific overrides (highest priority)
        for (const auto &ov : radius_overrides) {
          if (ov.type != SimParams::RadiusOverride::Type::Cells) continue;
          for (int id : ov.cell_ids) {
            if (id >= 0 && id < num_cells) {
              h_radius[id] = ov.value;
            }
          }
          printf("Per-cell radius: %d specific cell(s) set to %.2f\n",
                 static_cast<int>(ov.cell_ids.size()), ov.value);
        }

        float r_min = *std::min_element(h_radius.begin(), h_radius.end());
        float r_max = *std::max_element(h_radius.begin(), h_radius.end());
        printf("Per-cell radius: range [%.2f, %.2f]\n", r_min, r_max);
      } else {
        // Uniform: all cells get params.target_radius
        for (int i = 0; i < num_cells; ++i) {
          h_radius[i] = params.target_radius;
        }
      }

      // Compute derived arrays
      std::vector<float> h_target_area(num_cells);
      std::vector<float> h_volume_coeff(num_cells);
      for (int i = 0; i < num_cells; ++i) {
        h_target_area[i] = static_cast<float>(M_PI) * h_radius[i] * h_radius[i];
        h_volume_coeff[i] = params.mu / h_target_area[i];
      }

      // Upload all three arrays
      cudaMemcpy(d_target_radius, h_radius.data(), num_cells * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(d_target_area, h_target_area.data(), num_cells * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(d_volume_coeff, h_volume_coeff.data(), num_cells * sizeof(float),
                 cudaMemcpyHostToDevice);
    }

    // Initialize RNG states with time-based seed
    unsigned long seed = static_cast<unsigned long>(time(nullptr));
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    kernel_init_rng_states<<<blocks, threads>>>(d_rng_states, seed, num_cells);
    cudaDeviceSynchronize();
    rng_initialized = true;
  }

  // Update polarization direction on GPU (eliminates host->device transfer)
  {
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    int motility_model =
        (params.motility_model == SimParams::MotilityModel::RunAndTumble) ? 0
                                                                          : 1;
    kernel_update_polarization<<<blocks, threads>>>(
        d_rng_states, d_polarization_x, d_polarization_y, d_theta, dt,
        params.tau, motility_model, num_cells);
    
    // Sync polarization back to host cells ONLY if requested
    // (only needed when saving trajectories - avoid unnecessary GPU->CPU transfer)
    if (sync_polarization_to_host) {
      std::vector<float> h_pol_x(num_cells), h_pol_y(num_cells), h_theta(num_cells);
      cudaMemcpy(h_pol_x.data(), d_polarization_x, num_cells * sizeof(float),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pol_y.data(), d_polarization_y, num_cells * sizeof(float),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(h_theta.data(), d_theta, num_cells * sizeof(float),
                 cudaMemcpyDeviceToHost);
      for (int i = 0; i < num_cells; ++i) {
        domain.cells[i]->polarization.x = h_pol_x[i];
        domain.cells[i]->polarization.y = h_pol_y[i];
        domain.cells[i]->theta = h_theta[i];
      }
    }
  }

  // Increment step counter
  step_counter++;

#ifdef ENABLE_STEP_PROFILING
  PROF_LAP(acc_polar);
#endif

  // =========================================================================
  // GPU Bounding Box Updates (every step)
  // =========================================================================
  // With GPU-accelerated scan + remap, bbox checking is cheap enough to run
  // every step. The scan kernel reads centroids directly from d_centroids_x/y
  // (already on GPU), so we do NOT need to force sync_centroids=true every step.
  //
  // The step_fused sync_centroids (which copies centroids/volumes/velocities to
  // host) only needs to happen periodically for host-side bookkeeping (printing,
  // checkpointing, etc). We keep the interval for that, but bbox updates happen
  // independently on GPU every step.
  // =========================================================================

  // Sync centroids/volumes/velocities to host only when caller needs them
  // (print, save, trajectory steps). Avoids unnecessary D→H + cudaDeviceSynchronize.
  bool sync_centroids = sync_centroids_to_host;

  // =========================================================================
  // Adaptive Neighbor List Rebuild Decision
  // With sum field active, neighbor lists are not used in the main solver —
  // skip the displacement check and rebuild entirely.
  // =========================================================================
  bool rebuild_neighbors = false;
  
  if (!d_sum_field) {
    // Fallback path without sum field: use neighbor lists
    if (!neighbor_list_valid || num_cells <= 1) {
      rebuild_neighbors = true;
    } else if (num_cells > 1 && sync_centroids) {
      cudaMemsetAsync(d_max_displacement, 0, sizeof(float));
      
      int threads = 256;
      int blocks = (num_cells + threads - 1) / threads;
      kernel_compute_max_displacement<<<blocks, threads, threads * sizeof(float)>>>(
          d_centroids_x, d_centroids_y, d_prev_centroids_x, d_prev_centroids_y,
          d_max_displacement, params.Nx, params.Ny, num_cells);
      
      float h_max_disp_sq;
      cudaMemcpy(&h_max_disp_sq, d_max_displacement, sizeof(float),
                 cudaMemcpyDeviceToHost);
      float max_disp = sqrtf(h_max_disp_sq);
      
      float adaptive_threshold = neighbor_rebuild_threshold;
      float v_A_hint = params.v_A;
      if (v_A_hint < 0.001f) {
        adaptive_threshold = params.target_radius * 0.5f;
      } else {
        adaptive_threshold = std::max(1.0f, params.target_radius * 0.25f);
      }
      
      rebuild_neighbors = (max_disp > adaptive_threshold);
    }
  }

  // =========================================================================
  // Ping-pong sum field: select which buffer to use this step.
  // The buffer was pre-cleared by the async memset from 2 steps ago
  // (or by initial allocation on first use).
  // =========================================================================
  float *current_sum_field = nullptr;
  if (d_sum_field) {
    current_sum_field = (step_counter % 2 == 0) ? d_sum_field : d_sum_field_b;
  }

  // Ping-pong sum field: for small cell counts, the fused kernel scatters NEW phi²
  // to next_sum_field inline (saves a kernel launch). For large cell counts,
  // atomicAdd contention makes inline scatter slower — use standalone kernel instead.
  float *next_sum_field_for_scatter = nullptr;
  bool use_inline_scatter = (num_cells <= 2048);
  if (d_sum_field && use_inline_scatter) {
    next_sum_field_for_scatter = (step_counter % 2 == 0) ? d_sum_field_b : d_sum_field;
  }

  // Ping-pong linear sum field (adhesion): same strategy, nullptr when J=0
  float *current_sum_field_linear = nullptr;
  if (d_sum_field_linear && params.adhesion_J > 0.0f) {
    current_sum_field_linear = (step_counter % 2 == 0) ? d_sum_field_linear : d_sum_field_linear_b;
  }

#ifdef ENABLE_STEP_PROFILING
  PROF_LAP(acc_sumfield);
#endif

  step_fused(domain, dt, d_all_phi_ptrs, d_all_phi_out_ptrs,
             d_all_widths,
             d_all_heights, d_all_offsets_x, d_all_offsets_y,
             d_all_field_sizes, d_volumes, d_integrals_x, d_integrals_y,
             d_centroid_sums, d_volume_deviations, d_velocities_x,
             d_velocities_y, d_ref_x, d_ref_y, d_polarization_x,
             d_polarization_y, d_centroids_x, d_centroids_y,
             d_neighbor_counts, d_neighbor_lists,
             d_v_A,
             d_gamma,
             d_two_gamma,
             d_two_gamma_bulk,
             d_target_area,
             d_volume_coeff,
             d_perimeters,
             d_second_moment_x,
             d_second_moment_y,
             d_old_widths,
             d_old_heights,
             d_target_radius,
             pool_max_side,
             d_max_wh,
             d_shift_x,
             d_shift_y,
             d_block_arrival,
             current_sum_field,
             current_sum_field_linear,
             next_sum_field_for_scatter,
             cached_max_size, cached_max_w, cached_max_h,
             sync_centroids,
             rebuild_neighbors,
             centroid_sums_ready,
             step_counter);

  // Fused kernel populated centroid_sums for next step
  centroid_sums_ready = true;

#ifdef ENABLE_STEP_PROFILING
  PROF_LAP(acc_step_fused);
#endif

  // Clear sum field for reuse in step+2. The memset is small enough (<1 MB)
  // that it completes before the next step's scatter kernels, so no event
  // synchronization is needed — just queue it on the default stream.
  if (current_sum_field) {
    cudaMemsetAsync(current_sum_field, 0, sum_field_size * sizeof(float));
    if (current_sum_field_linear) {
      cudaMemsetAsync(current_sum_field_linear, 0, sum_field_size * sizeof(float));
    }
  }

#ifdef ENABLE_STEP_PROFILING
  PROF_LAP(acc_clear);
#endif

  // Double-buffer swap: fused kernel wrote new phi to phi_out_ptrs.
  // GPU-side pointer swap (avoids synchronous H→D memcpy each step)
  {
    int nc = domain.num_cells();
    int swap_threads = 256;
    int swap_blocks = (nc + swap_threads - 1) / swap_threads;
    kernel_swap_phi_ptrs<<<swap_blocks, swap_threads>>>(
        d_all_phi_ptrs, d_all_phi_out_ptrs,
        d_all_offsets_x, d_all_offsets_y,
        d_shift_x, d_shift_y, nc);
    host_ptrs_stale = true;
  }

  // Sync GPU offsets + widths/heights to host bbox structs (after swap kernel applied shifts)
  if (sync_centroids_to_host) {
    cudaDeviceSynchronize();  // ensure swap kernel finished
    int nc = domain.num_cells();
    std::vector<int> h_off_x(nc), h_off_y(nc), h_w(nc), h_h(nc);
    cudaMemcpy(h_off_x.data(), d_all_offsets_x, nc * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_off_y.data(), d_all_offsets_y, nc * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w.data(), d_all_widths, nc * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_h.data(), d_all_heights, nc * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < nc; ++i) {
      domain.cells[i]->bbox_with_halo.x0 = h_off_x[i];
      domain.cells[i]->bbox_with_halo.y0 = h_off_y[i];
      domain.cells[i]->bbox_with_halo.x1 = h_off_x[i] + h_w[i];
      domain.cells[i]->bbox_with_halo.y1 = h_off_y[i] + h_h[i];
      domain.cells[i]->field_size = h_w[i] * h_h[i];
    }
  }

  // After rebuild, save current centroids as reference for next displacement check
  if (!d_sum_field && rebuild_neighbors && num_cells > 1) {
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    kernel_copy_centroids<<<blocks, threads>>>(
        d_centroids_x, d_centroids_y, d_prev_centroids_x, d_prev_centroids_y,
        num_cells);
    neighbor_list_valid = true;
    neighbor_rebuild_count++;
  } else if (!d_sum_field && num_cells > 1) {
    neighbor_skip_count++;
  }

  // Lazy host pointer flush: ensure host Cell structs have correct d_phi/d_dphi_dt
  // before returning (callers may access them for checkpoints, VTK saves, etc.).
  // Lazy host pointer flush: only swap when caller actually needs host Cell
  // pointers (checkpoints, VTK saves, trajectory). Skip on pure compute steps.
  // The GPU-side kernel_swap_phi_ptrs already handles the double buffer swap.
  if (host_ptrs_stale && (sync_centroids_to_host || sync_polarization_to_host)) {
    int nc = domain.num_cells();
    for (int i = 0; i < nc; ++i) {
      std::swap(domain.cells[i]->d_phi, domain.cells[i]->d_dphi_dt);
    }
    host_ptrs_stale = false;
  }

#ifdef ENABLE_STEP_PROFILING
  PROF_LAP(acc_bbox);
  auto tp_end = std::chrono::high_resolution_clock::now();
  acc_total += std::chrono::duration<double, std::micro>(tp_end - tp0).count() + acc_alloc + acc_polar + acc_sumfield + acc_step_fused + acc_clear + acc_bbox;
  prof_count++;
  if (prof_count % 10000 == 0) {
    double n = (double)prof_count;
    printf("\n=== Step Profiling (avg over %d steps, us) ===\n", prof_count);
    printf("  alloc+dims:   %.1f (%.1f%%)\n", acc_alloc/n, 100*acc_alloc/(acc_alloc+acc_polar+acc_sumfield+acc_step_fused+acc_clear+acc_bbox));
    printf("  polarization: %.1f (%.1f%%)\n", acc_polar/n, 100*acc_polar/(acc_alloc+acc_polar+acc_sumfield+acc_step_fused+acc_clear+acc_bbox));
    printf("  sumfield:     %.1f (%.1f%%)\n", acc_sumfield/n, 100*acc_sumfield/(acc_alloc+acc_polar+acc_sumfield+acc_step_fused+acc_clear+acc_bbox));
    printf("  step_fused:   %.1f (%.1f%%)\n", acc_step_fused/n, 100*acc_step_fused/(acc_alloc+acc_polar+acc_sumfield+acc_step_fused+acc_clear+acc_bbox));
    printf("  async_clear:  %.1f (%.1f%%)\n", acc_clear/n, 100*acc_clear/(acc_alloc+acc_polar+acc_sumfield+acc_step_fused+acc_clear+acc_bbox));
    printf("  bbox+swap:    %.1f (%.1f%%)\n", acc_bbox/n, 100*acc_bbox/(acc_alloc+acc_polar+acc_sumfield+acc_step_fused+acc_clear+acc_bbox));
    printf("  TOTAL:        %.1f us/step\n\n", (acc_alloc+acc_polar+acc_sumfield+acc_step_fused+acc_clear+acc_bbox)/n);
    fflush(stdout);
  }
#endif
}

#ifdef DIAGNOSTICS_ENABLED
void Integrator::compute_diagnostics(Domain &domain, DiagnosticBuffers &diag) {
  int num_cells = domain.num_cells();
  if (num_cells == 0 || !diag.allocated) return;
  
  // Make sure arrays are up to date
  if (domain.device_arrays_dirty) {
    update_interaction_arrays(domain);
    domain.sync_device_arrays();
  }
  
  // Call the diagnostic kernels with our pre-allocated arrays
  run_diagnostics(
      domain,
      d_all_phi_ptrs,
      d_all_widths,
      d_all_heights,
      d_all_offsets_x,
      d_all_offsets_y,
      d_neighbor_counts,
      d_neighbor_lists,
      diag);
}
#endif

#ifdef STRESS_FIELDS_ENABLED
void Integrator::compute_stress_fields(Domain &domain, StressFieldBuffers &stress) {
  int num_cells = domain.num_cells();
  if (num_cells == 0 || !stress.allocated) return;
  
  // Make sure arrays are up to date (check both dirty flag AND if first call)
  if (domain.device_arrays_dirty || interaction_array_capacity == 0) {
    update_interaction_arrays(domain);
    domain.sync_device_arrays();
  }
  
  // Call the stress field computation function
  cellsim::compute_stress_fields(
      domain,
      d_all_phi_ptrs,
      d_all_widths,
      d_all_heights,
      d_all_offsets_x,
      d_all_offsets_y,
      stress);
}
#endif

} // namespace cellsim
