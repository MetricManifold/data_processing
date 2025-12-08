#pragma once

#include "domain.cuh"
#include "types.cuh"
#include <curand_kernel.h>
#include <vector>

namespace cellsim {

//=============================================================================
// Time Integration
//=============================================================================

// Forward Euler step: φ += dt * dφ/dt
__global__ void kernel_euler_step(float *__restrict__ phi,
                                  const float *__restrict__ dphi_dt, int size,
                                  float dt);

// Integrator class with optimized memory management
class Integrator {
public:
  enum class Method {
    ForwardEuler,
    // Future: RK4, SemiImplicit
  };

  Method method;
  float *d_work_buffer;
  size_t work_buffer_size;

  // CUDA streams for parallel cell processing
  static constexpr int MAX_STREAMS = 8;
  std::vector<cudaStream_t> streams;
  int num_streams;

  // Pre-allocated device arrays for interaction computation
  // (avoids cudaMalloc/cudaFree every step)
  float **d_all_phi_ptrs; // Pointers to all cell phi arrays
  int *d_all_widths;      // All cell widths
  int *d_all_heights;     // All cell heights
  int *d_all_offsets_x;   // All cell x offsets
  int *d_all_offsets_y;   // All cell y offsets
  int *d_all_field_sizes; // All cell field sizes (for batched reductions)
  size_t interaction_array_capacity; // Current capacity (num cells)

  // Device arrays for fused reduction outputs
  float *d_volumes;       // Volume integral per cell
  float *d_integrals_x;   // Motility integral X per cell
  float *d_integrals_y;   // Motility integral Y per cell
  float *d_centroid_sums; // Centroid sums: [dx*phi², dy*phi², phi²] per cell
  size_t reduction_array_capacity; // Capacity for reduction arrays

  // Additional arrays for GPU-side computation
  float *d_volume_deviations; // Volume deviations per cell
  float *d_velocities_x;      // Velocities X per cell
  float *d_velocities_y;      // Velocities Y per cell
  float *d_ref_x;             // Reference points X for centroid
  float *d_ref_y;             // Reference points Y for centroid
  float *d_polarization_x;    // Polarization directions X
  float *d_polarization_y;    // Polarization directions Y
  float *d_theta;             // Polarization angles (for GPU-side RNG)
  float *d_centroids_x;       // Computed centroids X (GPU-side)
  float *d_centroids_y;       // Computed centroids Y (GPU-side)

  // Neighbor-list arrays for O(k) interaction instead of O(N²)
  int *d_neighbor_counts; // Number of neighbors per cell
  int *d_neighbor_lists;  // Flattened neighbor indices [MAX_NEIGHBORS *
                          // num_cells]

  // GPU-side RNG for polarization updates (eliminates host->device transfer)
  curandState *d_rng_states;  // One RNG state per cell
  bool rng_initialized;       // Track if RNG states have been initialized

  // Adaptive neighbor list caching (rebuilds only when cells move significantly)
  float *d_prev_centroids_x;       // Centroids at last neighbor rebuild
  float *d_prev_centroids_y;       // Centroids at last neighbor rebuild
  float *d_max_displacement;       // Reduction buffer for max displacement
  bool neighbor_list_valid;        // True if neighbor list is up-to-date
  float neighbor_rebuild_threshold; // Rebuild when max displacement exceeds this
  int neighbor_rebuild_count;      // Stats: how many rebuilds occurred
  int neighbor_skip_count;         // Stats: how many rebuilds skipped

  // Bounding box update control (reduces GPU-CPU syncs)
  int bbox_update_interval; // Update bboxes every N steps (default: 10)
  int step_counter;         // Internal step counter

public:
  Integrator(Method m = Method::ForwardEuler);
  ~Integrator();

  // Allocate work buffer based on domain size
  void allocate_work_buffer(const Domain &domain);
  void free_work_buffer();

  // Allocate/resize interaction arrays
  void allocate_interaction_arrays(int num_cells);
  void free_interaction_arrays();
  void update_interaction_arrays(const Domain &domain);

  // Allocate/resize reduction arrays for fused kernels
  void allocate_reduction_arrays(int num_cells);
  void free_reduction_arrays();

  // Initialize/destroy CUDA streams
  void create_streams(int n = MAX_STREAMS);
  void destroy_streams();

  // Perform one time step
  void step(Domain &domain, float dt);
};

} // namespace cellsim
