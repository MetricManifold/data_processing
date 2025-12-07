#include "integrator.cuh"
#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace cellsim {

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
      d_polarization_x(nullptr), d_polarization_y(nullptr),
      d_centroids_x(nullptr), d_centroids_y(nullptr),
      d_neighbor_counts(nullptr), d_neighbor_lists(nullptr), use_fused_v2(true),
      use_fused_v4(false), bbox_update_interval(10), step_counter(0) {
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
  cudaMalloc(&d_centroids_x, new_capacity * sizeof(float));
  cudaMalloc(&d_centroids_y, new_capacity * sizeof(float));

  // Neighbor list arrays for V4 optimization
  cudaMalloc(&d_neighbor_counts, new_capacity * sizeof(int));
  cudaMalloc(&d_neighbor_lists, MAX_NEIGHBORS_V4 * new_capacity * sizeof(int));

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

  // Update polarization direction BEFORE fused step
  const SimParams &params = domain.params;
  int num_cells = domain.num_cells();

  for (auto &cell : domain.cells) {
    if (params.motility_model == SimParams::MotilityModel::RunAndTumble) {
      // Run-and-Tumble: Poisson reorientation events
      // Probability of tumble in dt: P = 1 - exp(-dt/τ) ≈ dt/τ for small dt
      float p_tumble = 1.0f - expf(-dt / params.tau);
      if ((float)rand() / RAND_MAX < p_tumble) {
        // Pick completely new random direction
        cell->theta = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
      }
    } else {
      // Active Brownian Particle: continuous rotational diffusion
      // dθ/dt = η(t) where η is white noise with <η(t)η(t')> = (2/τ)δ(t-t')
      float noise_strength = sqrtf(2.0f * dt / params.tau);
      float dtheta = noise_strength * ((float)rand() / RAND_MAX - 0.5f) * 2.0f *
                     sqrtf(3.0f);
      cell->theta += dtheta;

      // Keep angle in [0, 2π)
      while (cell->theta < 0)
        cell->theta += 2.0f * M_PI;
      while (cell->theta >= 2.0f * M_PI)
        cell->theta -= 2.0f * M_PI;
    }

    cell->polarization.x = cosf(cell->theta);
    cell->polarization.y = sinf(cell->theta);
  }

  // Increment step counter and determine if we need to sync centroids
  step_counter++;
  bool sync_centroids =
      (step_counter == 1) || (step_counter % bbox_update_interval == 0);

  if (use_fused_v4) {
    // V4 path: neighbor-list optimization (O(N) instead of O(N²))
    step_fused_v4(domain, dt, d_work_buffer, d_all_phi_ptrs, d_all_widths,
                  d_all_heights, d_all_offsets_x, d_all_offsets_y,
                  d_all_field_sizes, d_volumes, d_integrals_x, d_integrals_y,
                  d_centroid_sums, d_volume_deviations, d_velocities_x,
                  d_velocities_y, d_ref_x, d_ref_y, d_polarization_x,
                  d_polarization_y, d_centroids_x, d_centroids_y,
                  d_neighbor_counts, d_neighbor_lists, sync_centroids);

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
      }
    }
    return;
  }

  if (use_fused_v2) {
    // Optimized fused v2 path - batched kernels, minimal GPU-CPU syncs
    step_fused_v2(domain, dt, d_work_buffer, d_all_phi_ptrs, d_all_widths,
                  d_all_heights, d_all_offsets_x, d_all_offsets_y,
                  d_all_field_sizes, // For batched reductions
                  d_volumes, d_integrals_x, d_integrals_y, d_centroid_sums,
                  d_volume_deviations, d_velocities_x, d_velocities_y, d_ref_x,
                  d_ref_y, d_polarization_x, d_polarization_y, d_centroids_x,
                  d_centroids_y, sync_centroids);

    // Only update bboxes when we sync centroids to host
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
      }
    }

    return;
  }

  // V2 is the default optimized path - no fallback needed
  // If we reach here, v2 is disabled but v4 is also disabled - error
  printf("ERROR: No kernel path enabled. Set use_fused_v2=true or "
         "use_fused_v4=true\n");
}

} // namespace cellsim
