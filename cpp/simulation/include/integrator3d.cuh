#pragma once

#include "domain3d.cuh"
#include "gpu_memory_tracker.cuh"
#include "kernels3d.cuh"
#include "types3d.cuh"
#include <vector>

namespace cellsim {

//=============================================================================
// 3D Time Integration
//=============================================================================

class Integrator3D {
public:
  enum class Method {
    ForwardEuler,
  };

  Method method;
  float *d_work_buffer;
  size_t work_buffer_size;

  // CUDA streams for parallel cell processing
  static constexpr int MAX_STREAMS = 8;
  std::vector<cudaStream_t> streams;
  int num_streams;

  // Pre-allocated device arrays for interaction computation
  float **d_all_phi_ptrs;
  int *d_all_widths;
  int *d_all_heights;
  int *d_all_depths;
  int *d_all_offsets_x;
  int *d_all_offsets_y;
  int *d_all_offsets_z;
  int *d_all_field_sizes;
  size_t interaction_array_capacity;

  // Device arrays for reduction outputs
  float *d_volumes;
  float *d_integrals_x;
  float *d_integrals_y;
  float *d_integrals_z;
  float *d_centroid_sums;
  size_t reduction_array_capacity;

  // Additional GPU-side computation arrays (for fused step)
  float *d_volume_deviations;
  float *d_velocities_x;
  float *d_velocities_y;
  float *d_velocities_z;
  float *d_ref_x;
  float *d_ref_y;
  float *d_ref_z;
  float *d_polarization_x;
  float *d_polarization_y;
  float *d_polarization_z;
  float *d_centroids_x;
  float *d_centroids_y;
  float *d_centroids_z;

  // Bounding box update control
  int bbox_update_interval;
  int step_counter;

  // Neighbor list for interaction kernel (O(k) instead of O(N²))
  static constexpr int MAX_NEIGHBORS_3D = 32; // Max neighbors per cell
  int *d_neighbor_counts;  // Number of neighbors per cell [num_cells]
  int *d_neighbor_lists;   // Flattened neighbor indices [MAX_NEIGHBORS_3D * num_cells]
  bool neighbor_list_valid; // True if neighbor list is up-to-date

public:
  Integrator3D(Method m = Method::ForwardEuler);
  ~Integrator3D();

  // Allocate work buffer based on domain size
  void allocate_work_buffer(const Domain3D &domain);
  void free_work_buffer();

  // Allocate/resize interaction arrays
  void allocate_interaction_arrays(int num_cells);
  void free_interaction_arrays();
  void update_interaction_arrays(const Domain3D &domain);

  // Allocate/resize reduction arrays
  void allocate_reduction_arrays(int num_cells);
  void free_reduction_arrays();

  // Initialize/destroy CUDA streams
  void create_streams(int n = MAX_STREAMS);
  void destroy_streams();

  // Perform one time step
  void step(Domain3D &domain, float dt);

  // Update cell velocities (motility model)
  void update_velocities(Domain3D &domain);
};

//=============================================================================
// Integrator3D Implementation
//=============================================================================

inline Integrator3D::Integrator3D(Method m)
    : method(m), d_work_buffer(nullptr), work_buffer_size(0), num_streams(0),
      d_all_phi_ptrs(nullptr), d_all_widths(nullptr), d_all_heights(nullptr),
      d_all_depths(nullptr), d_all_offsets_x(nullptr), d_all_offsets_y(nullptr),
      d_all_offsets_z(nullptr), d_all_field_sizes(nullptr),
      interaction_array_capacity(0), d_volumes(nullptr), d_integrals_x(nullptr),
      d_integrals_y(nullptr), d_integrals_z(nullptr), d_centroid_sums(nullptr),
      reduction_array_capacity(0), d_volume_deviations(nullptr),
      d_velocities_x(nullptr), d_velocities_y(nullptr), d_velocities_z(nullptr),
      d_ref_x(nullptr), d_ref_y(nullptr), d_ref_z(nullptr),
      d_polarization_x(nullptr), d_polarization_y(nullptr),
      d_polarization_z(nullptr), d_centroids_x(nullptr), d_centroids_y(nullptr),
      d_centroids_z(nullptr), bbox_update_interval(10), step_counter(0),
      d_neighbor_counts(nullptr), d_neighbor_lists(nullptr),
      neighbor_list_valid(false) {}

inline Integrator3D::~Integrator3D() {
  free_work_buffer();
  free_interaction_arrays();
  free_reduction_arrays();
  destroy_streams();
}

inline void Integrator3D::allocate_work_buffer(const Domain3D &domain) {
  // Find max field size across all cells
  int max_size = 0;
  for (const auto &cell : domain.cells) {
    max_size = max(max_size, cell->field_size);
  }

  // Sanity check for max_size
  if (max_size <= 0 || max_size > 500000000) {
    printf("ERROR: Invalid max_size=%d in allocate_work_buffer\n", max_size);
    return;
  }

  // Parallel allocation: N cells × 5 buffers each (down from 7!)
  // Buffer layout: [laplacian][bulk][constraint][advection][repulsion]
  // - Removed interaction_sum (unused in fused path)
  // - Reduction is done in shared memory, no buffer needed
  // Memory savings: 7 → 5 buffers = 29% reduction
  size_t needed = (size_t)domain.num_cells() * 5 * max_size * sizeof(float);

  if (needed > work_buffer_size) {
    free_work_buffer();
    cudaError_t err = cudaMalloc(&d_work_buffer, needed);
    if (err != cudaSuccess) {
      printf("ERROR: cudaMalloc failed for work buffer (%.1f MB): %s\n",
             needed / (1024.0 * 1024.0), cudaGetErrorString(err));
      d_work_buffer = nullptr;
      work_buffer_size = 0;
      return;
    }
    work_buffer_size = needed;
    printf("3D Work buffer: %.1f MB for %d cells (5 buffers/cell)\n",
           needed / (1024.0 * 1024.0), domain.num_cells());
  }
}

inline void Integrator3D::free_work_buffer() {
  if (d_work_buffer) {
    CUDA_FREE(d_work_buffer, work_buffer_size);
    d_work_buffer = nullptr;
    work_buffer_size = 0;
  }
}

inline void Integrator3D::allocate_interaction_arrays(int num_cells) {
  if (num_cells <= (int)interaction_array_capacity)
    return;

  free_interaction_arrays();

  cudaMalloc(&d_all_phi_ptrs, num_cells * sizeof(float *));
  cudaMalloc(&d_all_widths, num_cells * sizeof(int));
  cudaMalloc(&d_all_heights, num_cells * sizeof(int));
  cudaMalloc(&d_all_depths, num_cells * sizeof(int));
  cudaMalloc(&d_all_offsets_x, num_cells * sizeof(int));
  cudaMalloc(&d_all_offsets_y, num_cells * sizeof(int));
  cudaMalloc(&d_all_offsets_z, num_cells * sizeof(int));
  cudaMalloc(&d_all_field_sizes, num_cells * sizeof(int));

  // Neighbor list arrays
  cudaMalloc(&d_neighbor_counts, num_cells * sizeof(int));
  cudaMalloc(&d_neighbor_lists, MAX_NEIGHBORS_3D * num_cells * sizeof(int));
  neighbor_list_valid = false; // Force rebuild on first use

  interaction_array_capacity = num_cells;
}

inline void Integrator3D::free_interaction_arrays() {
  if (d_all_phi_ptrs)
    cudaFree(d_all_phi_ptrs);
  if (d_all_widths)
    cudaFree(d_all_widths);
  if (d_all_heights)
    cudaFree(d_all_heights);
  if (d_all_depths)
    cudaFree(d_all_depths);
  if (d_all_offsets_x)
    cudaFree(d_all_offsets_x);
  if (d_all_offsets_y)
    cudaFree(d_all_offsets_y);
  if (d_all_offsets_z)
    cudaFree(d_all_offsets_z);
  if (d_all_field_sizes)
    cudaFree(d_all_field_sizes);
  if (d_neighbor_counts)
    cudaFree(d_neighbor_counts);
  if (d_neighbor_lists)
    cudaFree(d_neighbor_lists);

  d_all_phi_ptrs = nullptr;
  d_all_widths = nullptr;
  d_all_heights = nullptr;
  d_all_depths = nullptr;
  d_all_offsets_x = nullptr;
  d_all_offsets_y = nullptr;
  d_all_offsets_z = nullptr;
  d_all_field_sizes = nullptr;
  d_neighbor_counts = nullptr;
  d_neighbor_lists = nullptr;
  neighbor_list_valid = false;
  interaction_array_capacity = 0;
}

inline void Integrator3D::update_interaction_arrays(const Domain3D &domain) {
  int n = domain.num_cells();
  allocate_interaction_arrays(n);

  std::vector<float *> phi_ptrs(n);
  std::vector<int> widths(n), heights(n), depths(n);
  std::vector<int> offsets_x(n), offsets_y(n), offsets_z(n);
  std::vector<int> field_sizes(n);

  for (int i = 0; i < n; ++i) {
    phi_ptrs[i] = domain.cells[i]->d_phi;
    widths[i] = domain.cells[i]->width();
    heights[i] = domain.cells[i]->height();
    depths[i] = domain.cells[i]->depth();
    offsets_x[i] = domain.cells[i]->bbox_with_halo.x0;
    offsets_y[i] = domain.cells[i]->bbox_with_halo.y0;
    offsets_z[i] = domain.cells[i]->bbox_with_halo.z0;
    field_sizes[i] = domain.cells[i]->field_size;
  }

  cudaMemcpy(d_all_phi_ptrs, phi_ptrs.data(), n * sizeof(float *),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_widths, widths.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_heights, heights.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_depths, depths.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_offsets_x, offsets_x.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_offsets_y, offsets_y.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_offsets_z, offsets_z.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_all_field_sizes, field_sizes.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
}

inline void Integrator3D::allocate_reduction_arrays(int num_cells) {
  if (num_cells <= (int)reduction_array_capacity)
    return;

  free_reduction_arrays();

  cudaMalloc(&d_volumes, num_cells * sizeof(float));
  cudaMalloc(&d_integrals_x, num_cells * sizeof(float));
  cudaMalloc(&d_integrals_y, num_cells * sizeof(float));
  cudaMalloc(&d_integrals_z, num_cells * sizeof(float));
  cudaMalloc(&d_centroid_sums,
             num_cells * 4 * sizeof(float)); // dx, dy, dz, weight

  // Additional GPU-side computation arrays
  cudaMalloc(&d_volume_deviations, num_cells * sizeof(float));
  cudaMalloc(&d_velocities_x, num_cells * sizeof(float));
  cudaMalloc(&d_velocities_y, num_cells * sizeof(float));
  cudaMalloc(&d_velocities_z, num_cells * sizeof(float));
  cudaMalloc(&d_ref_x, num_cells * sizeof(float));
  cudaMalloc(&d_ref_y, num_cells * sizeof(float));
  cudaMalloc(&d_ref_z, num_cells * sizeof(float));
  cudaMalloc(&d_polarization_x, num_cells * sizeof(float));
  cudaMalloc(&d_polarization_y, num_cells * sizeof(float));
  cudaMalloc(&d_polarization_z, num_cells * sizeof(float));
  cudaMalloc(&d_centroids_x, num_cells * sizeof(float));
  cudaMalloc(&d_centroids_y, num_cells * sizeof(float));
  cudaMalloc(&d_centroids_z, num_cells * sizeof(float));

  reduction_array_capacity = num_cells;
}

inline void Integrator3D::free_reduction_arrays() {
  if (d_volumes)
    cudaFree(d_volumes);
  if (d_integrals_x)
    cudaFree(d_integrals_x);
  if (d_integrals_y)
    cudaFree(d_integrals_y);
  if (d_integrals_z)
    cudaFree(d_integrals_z);
  if (d_centroid_sums)
    cudaFree(d_centroid_sums);
  if (d_volume_deviations)
    cudaFree(d_volume_deviations);
  if (d_velocities_x)
    cudaFree(d_velocities_x);
  if (d_velocities_y)
    cudaFree(d_velocities_y);
  if (d_velocities_z)
    cudaFree(d_velocities_z);
  if (d_ref_x)
    cudaFree(d_ref_x);
  if (d_ref_y)
    cudaFree(d_ref_y);
  if (d_ref_z)
    cudaFree(d_ref_z);
  if (d_polarization_x)
    cudaFree(d_polarization_x);
  if (d_polarization_y)
    cudaFree(d_polarization_y);
  if (d_polarization_z)
    cudaFree(d_polarization_z);
  if (d_centroids_x)
    cudaFree(d_centroids_x);
  if (d_centroids_y)
    cudaFree(d_centroids_y);
  if (d_centroids_z)
    cudaFree(d_centroids_z);

  d_volumes = nullptr;
  d_integrals_x = nullptr;
  d_integrals_y = nullptr;
  d_integrals_z = nullptr;
  d_centroid_sums = nullptr;
  d_volume_deviations = nullptr;
  d_velocities_x = nullptr;
  d_velocities_y = nullptr;
  d_velocities_z = nullptr;
  d_ref_x = nullptr;
  d_ref_y = nullptr;
  d_ref_z = nullptr;
  d_polarization_x = nullptr;
  d_polarization_y = nullptr;
  d_polarization_z = nullptr;
  d_centroids_x = nullptr;
  d_centroids_y = nullptr;
  d_centroids_z = nullptr;
  reduction_array_capacity = 0;
}

inline void Integrator3D::create_streams(int n) {
  destroy_streams();
  num_streams = min(n, MAX_STREAMS);
  streams.resize(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    cudaStreamCreate(&streams[i]);
  }
}

inline void Integrator3D::destroy_streams() {
  for (auto &s : streams) {
    cudaStreamDestroy(s);
  }
  streams.clear();
  num_streams = 0;
}

inline void Integrator3D::step(Domain3D &domain, float dt) {
  if (domain.num_cells() == 0)
    return;

  allocate_work_buffer(domain);
  allocate_reduction_arrays(domain.num_cells());

  // Only update interaction arrays when bboxes change
  // (first call sets capacity=0 triggering initial update)
  if (interaction_array_capacity == 0) {
    update_interaction_arrays(domain);
  }

  // Update polarization direction BEFORE fused step (on CPU due to RNG)
  const SimParams3D &params = domain.params;
  for (auto &cell : domain.cells) {
    if (params.motility_model == SimParams::MotilityModel::RunAndTumble) {
      float p_tumble = 1.0f - expf(-dt / params.tau);
      if ((float)rand() / RAND_MAX < p_tumble) {
        // Pick new random direction on unit sphere
        float theta = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float phi_angle = acosf(2.0f * ((float)rand() / RAND_MAX) - 1.0f);
        cell->polarization.x = sinf(phi_angle) * cosf(theta);
        cell->polarization.y = sinf(phi_angle) * sinf(theta);
        cell->polarization.z = cosf(phi_angle);
      }
    } else {
      // ABP: rotational diffusion on sphere
      float noise_strength = sqrtf(2.0f * dt / params.tau);
      float dtheta = noise_strength * ((float)rand() / RAND_MAX - 0.5f) * 2.0f *
                     sqrtf(3.0f);
      float dphi = noise_strength * ((float)rand() / RAND_MAX - 0.5f) * 2.0f *
                   sqrtf(3.0f);

      // Convert to cartesian, apply rotation, normalize
      Vec3 p = cell->polarization;
      float theta = atan2f(p.y, p.x) + dtheta;
      float phi_angle = acosf(fminf(1.0f, fmaxf(-1.0f, p.z))) + dphi;
      phi_angle = fmaxf(0.01f, fminf((float)M_PI - 0.01f, phi_angle));

      cell->polarization.x = sinf(phi_angle) * cosf(theta);
      cell->polarization.y = sinf(phi_angle) * sinf(theta);
      cell->polarization.z = cosf(phi_angle);
    }
  }

  // Increment step counter and determine if we need to sync centroids
  step_counter++;
  bool sync_centroids =
      (step_counter == 1) || (step_counter % bbox_update_interval == 0);

  // Determine if neighbor list rebuild is needed
  // Rebuild on first step, when bboxes sync, or when explicitly invalidated
  int num_cells = domain.num_cells();
  bool rebuild_neighbors = !neighbor_list_valid || num_cells <= 1 || sync_centroids;

  // Use optimized fused step function with neighbor list
  step_fused_3d(domain, dt, d_work_buffer, d_all_phi_ptrs, d_all_widths,
                d_all_heights, d_all_depths, d_all_offsets_x, d_all_offsets_y,
                d_all_offsets_z, d_all_field_sizes, d_volumes, d_integrals_x,
                d_integrals_y, d_integrals_z, d_centroid_sums,
                d_volume_deviations, d_velocities_x, d_velocities_y,
                d_velocities_z, d_ref_x, d_ref_y, d_ref_z, d_polarization_x,
                d_polarization_y, d_polarization_z, d_centroids_x,
                d_centroids_y, d_centroids_z, d_neighbor_counts,
                d_neighbor_lists, sync_centroids, rebuild_neighbors);

  // Mark neighbor list as valid after rebuild
  if (rebuild_neighbors && num_cells > 1) {
    neighbor_list_valid = true;
  }

  // Update bboxes periodically
  if (sync_centroids) {
    bool any_changed = false;
    for (auto &cell : domain.cells) {
      if (cell->update_bounding_box(params)) {
        any_changed = true;
      }
    }
    if (any_changed) {
      update_interaction_arrays(domain);
      neighbor_list_valid = false; // Force rebuild after bbox changes
    }
  }
}

inline void Integrator3D::update_velocities(Domain3D &domain) {
  // Velocities are now computed on GPU inside step_fused_3d
  // This function is kept for compatibility but does nothing
}

} // namespace cellsim
