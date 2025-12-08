#include "integrator.cuh"
#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace cellsim {

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
    : method(m), d_work_buffer(nullptr), work_buffer_size(0), num_streams(0),
      d_all_phi_ptrs(nullptr), d_all_widths(nullptr), d_all_heights(nullptr),
      d_all_offsets_x(nullptr), d_all_offsets_y(nullptr),
      d_all_field_sizes(nullptr), interaction_array_capacity(0),
      d_volumes(nullptr), d_integrals_x(nullptr), d_integrals_y(nullptr),
      d_centroid_sums(nullptr), reduction_array_capacity(0),
      d_volume_deviations(nullptr), d_velocities_x(nullptr),
      d_velocities_y(nullptr), d_ref_x(nullptr), d_ref_y(nullptr),
      d_polarization_x(nullptr), d_polarization_y(nullptr), d_theta(nullptr),
      d_centroids_x(nullptr), d_centroids_y(nullptr),
      d_neighbor_counts(nullptr), d_neighbor_lists(nullptr),
      d_rng_states(nullptr), rng_initialized(false),
      d_prev_centroids_x(nullptr), d_prev_centroids_y(nullptr),
      d_max_displacement(nullptr), neighbor_list_valid(false),
      neighbor_rebuild_threshold(5.0f), // Default: rebuild when any cell moves >5 grid units
      neighbor_rebuild_count(0), neighbor_skip_count(0),
      bbox_update_interval(10), step_counter(0) {
  create_streams();
}

Integrator::~Integrator() {
  free_work_buffer();
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
}

void Integrator::allocate_work_buffer(const Domain &domain) {
  // Calculate required buffer size
  int max_field_size = 0;
  for (const auto &cell : domain.cells) {
    max_field_size = std::max(max_field_size, cell->field_size);
  }

  // Work buffer layout per cell (9 buffers):
  // [0] laplacian, [1] bulk, [2] constraint, [3] grad_x, [4] grad_y,
  // [5] phi_sq, [6] repulsion, [7] integrand_x, [8] integrand_y
  size_t required = domain.num_cells() * 9 * max_field_size * sizeof(float);

  if (required > work_buffer_size) {
    free_work_buffer();
    cudaMalloc(&d_work_buffer, required);
    work_buffer_size = required;
  }
}

void Integrator::free_work_buffer() {
  if (d_work_buffer) {
    cudaFree(d_work_buffer);
    d_work_buffer = nullptr;
  }
  work_buffer_size = 0;
}

void Integrator::allocate_reduction_arrays(int num_cells) {
  if (static_cast<size_t>(num_cells) <= reduction_array_capacity) {
    return;
  }

  free_reduction_arrays();

  size_t new_capacity =
      std::max(static_cast<size_t>(num_cells), reduction_array_capacity * 2);
  new_capacity = std::max(new_capacity, static_cast<size_t>(16));

  cudaMalloc(&d_volumes, new_capacity * sizeof(float));
  cudaMalloc(&d_integrals_x, new_capacity * sizeof(float));
  cudaMalloc(&d_integrals_y, new_capacity * sizeof(float));
  cudaMalloc(&d_centroid_sums,
             new_capacity * 3 * sizeof(float)); // 3 values per cell

  // Additional arrays for GPU-side computation
  cudaMalloc(&d_volume_deviations, new_capacity * sizeof(float));
  cudaMalloc(&d_velocities_x, new_capacity * sizeof(float));
  cudaMalloc(&d_velocities_y, new_capacity * sizeof(float));
  cudaMalloc(&d_ref_x, new_capacity * sizeof(float));
  cudaMalloc(&d_ref_y, new_capacity * sizeof(float));
  cudaMalloc(&d_polarization_x, new_capacity * sizeof(float));
  cudaMalloc(&d_polarization_y, new_capacity * sizeof(float));
  cudaMalloc(&d_theta, new_capacity * sizeof(float));
  cudaMalloc(&d_centroids_x, new_capacity * sizeof(float));
  cudaMalloc(&d_centroids_y, new_capacity * sizeof(float));

  // Neighbor list arrays for V4 optimization
  cudaMalloc(&d_neighbor_counts, new_capacity * sizeof(int));
  cudaMalloc(&d_neighbor_lists, MAX_NEIGHBORS * new_capacity * sizeof(int));

  // GPU-side RNG states for polarization updates
  cudaMalloc(&d_rng_states, new_capacity * sizeof(curandState));
  rng_initialized = false; // Need to reinitialize after reallocation

  // Adaptive neighbor list caching arrays
  cudaMalloc(&d_prev_centroids_x, new_capacity * sizeof(float));
  cudaMalloc(&d_prev_centroids_y, new_capacity * sizeof(float));
  cudaMalloc(&d_max_displacement, sizeof(float)); // Single value for reduction
  neighbor_list_valid = false; // Force rebuild on first use

  reduction_array_capacity = new_capacity;
}

void Integrator::free_reduction_arrays() {
  if (d_volumes) {
    cudaFree(d_volumes);
    d_volumes = nullptr;
  }
  if (d_integrals_x) {
    cudaFree(d_integrals_x);
    d_integrals_x = nullptr;
  }
  if (d_integrals_y) {
    cudaFree(d_integrals_y);
    d_integrals_y = nullptr;
  }
  if (d_centroid_sums) {
    cudaFree(d_centroid_sums);
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

void Integrator::step(Domain &domain, float dt) {
  if (domain.num_cells() == 0)
    return;

  // Ensure buffers are allocated
  allocate_work_buffer(domain);
  allocate_reduction_arrays(domain.num_cells());

  // Always update interaction arrays on first call or if domain changed
  // (interaction_array_capacity == 0 means first call)
  if (domain.device_arrays_dirty || interaction_array_capacity == 0) {
    update_interaction_arrays(domain);
    domain.device_arrays_dirty = false; // Clear after updating
  }

  const SimParams &params = domain.params;
  int num_cells = domain.num_cells();

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
  }

  // Increment step counter and determine if we need to sync centroids
  step_counter++;
  bool sync_centroids =
      (step_counter == 1) || (step_counter % bbox_update_interval == 0);

  // =========================================================================
  // Adaptive Neighbor List Rebuild Decision
  // =========================================================================
  // Rebuild neighbor list only when cells have moved significantly since last
  // rebuild. This is safe because the search radius (4*R) has margin - as long
  // as cells haven't moved more than ~R, the cached list is still valid.
  //
  // We only check displacement when sync_centroids is true to avoid extra
  // GPU-CPU syncs. Between checks, we use the cached neighbor list.
  //
  // For relaxation (v_A=0): cells move slowly, rarely need rebuild after equilibration
  // For motile cells: rebuild more often based on displacement
  // =========================================================================
  bool rebuild_neighbors = false;
  
  if (!neighbor_list_valid || num_cells <= 1) {
    // First step or list invalidated - must rebuild
    rebuild_neighbors = true;
  } else if (num_cells > 1 && sync_centroids) {
    // Only check displacement when we're already syncing centroids
    // This avoids extra GPU-CPU synchronization overhead
    
    // Reset max displacement to 0
    float zero = 0.0f;
    cudaMemcpy(d_max_displacement, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute max displacement (need centroids from previous step)
    // Note: d_centroids_x/y are updated by step_fused, so we compare against
    // d_prev_centroids from last rebuild
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    kernel_compute_max_displacement<<<blocks, threads, threads * sizeof(float)>>>(
        d_centroids_x, d_centroids_y, d_prev_centroids_x, d_prev_centroids_y,
        d_max_displacement, params.Nx, params.Ny, num_cells);
    
    // Read back max displacement (this is a single float, fast)
    // We're already doing a sync for centroids, so this doesn't add latency
    float h_max_disp_sq;
    cudaMemcpy(&h_max_disp_sq, d_max_displacement, sizeof(float),
               cudaMemcpyDeviceToHost);
    float max_disp = sqrtf(h_max_disp_sq);
    
    // Adaptive threshold based on motility
    // Base threshold: ~R/4 (with R=49, this is ~12 grid units)
    // This ensures we rebuild before cells could miss a new neighbor
    float adaptive_threshold = neighbor_rebuild_threshold;
    
    // For low/no motility, cells move very slowly - use larger threshold
    if (params.v_A < 0.001f) {
      // Relaxation mode: cells only move due to interactions
      // Can be very conservative, rebuild less often
      adaptive_threshold = params.target_radius * 0.5f; // R/2
    } else {
      // Motile cells: scale threshold inversely with velocity
      // Higher motility = more frequent rebuilds
      adaptive_threshold = std::max(1.0f, params.target_radius * 0.25f);
    }
    
    rebuild_neighbors = (max_disp > adaptive_threshold);
  }
  // Note: if sync_centroids is false, we skip the displacement check and
  // reuse the cached neighbor list. This is safe as long as bbox_update_interval
  // is small enough that cells can't move more than the threshold in that time.

  // V4 path: neighbor-list optimization (O(k) instead of O(N²))
  step_fused(domain, dt, d_work_buffer, d_all_phi_ptrs, d_all_widths,
             d_all_heights, d_all_offsets_x, d_all_offsets_y,
             d_all_field_sizes, d_volumes, d_integrals_x, d_integrals_y,
             d_centroid_sums, d_volume_deviations, d_velocities_x,
             d_velocities_y, d_ref_x, d_ref_y, d_polarization_x,
             d_polarization_y, d_centroids_x, d_centroids_y,
             d_neighbor_counts, d_neighbor_lists, sync_centroids,
             rebuild_neighbors);

  // After rebuild, save current centroids as reference for next displacement check
  if (rebuild_neighbors && num_cells > 1) {
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    kernel_copy_centroids<<<blocks, threads>>>(
        d_centroids_x, d_centroids_y, d_prev_centroids_x, d_prev_centroids_y,
        num_cells);
    neighbor_list_valid = true;
    neighbor_rebuild_count++;
  } else if (num_cells > 1) {
    neighbor_skip_count++;
  }

  // Update bboxes when we sync centroids to host
  if (sync_centroids) {
    bool any_bbox_changed = false;
    for (auto &cell : domain.cells) {
      if (cell->update_bounding_box(domain.params)) {
        any_bbox_changed = true;
      }
    }
    if (any_bbox_changed) {
      domain.device_arrays_dirty = true;
      domain.update_overlap_pairs();
      // Bbox changed means neighbor list may be stale
      neighbor_list_valid = false;
    }
  }
}

} // namespace cellsim
