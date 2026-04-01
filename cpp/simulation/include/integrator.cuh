#pragma once

#include "domain.cuh"
#include "types.cuh"
#ifdef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"
#endif
#ifdef STRESS_FIELDS_ENABLED
#ifndef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"
#endif
#endif
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
  bool centroid_sums_ready; // True after fused kernel populates centroid_sums

  // CUDA streams for parallel cell processing
  static constexpr int MAX_STREAMS = 8;
  std::vector<cudaStream_t> streams;
  int num_streams;

  // Pre-allocated device arrays for interaction computation
  // (avoids cudaMalloc/cudaFree every step)
  float **d_all_phi_ptrs; // Pointers to all cell phi arrays
  float **d_all_phi_out_ptrs; // Pointers to all cell phi output arrays (double buffer)
  int *d_all_widths;      // All cell widths
  int *d_all_heights;     // All cell heights
  int *d_all_offsets_x;   // All cell x offsets
  int *d_all_offsets_y;   // All cell y offsets
  int *d_all_field_sizes; // All cell field sizes (for batched reductions)
  size_t interaction_array_capacity; // Current capacity (num cells)

  // Device arrays for fused reduction outputs
  float *d_volumes;       // Volume integral per cell
  float *d_integrals_x;   // Motility integral X per cell (alias into d_reduction_block)
  float *d_integrals_y;   // Motility integral Y per cell (alias into d_reduction_block)
  float *d_centroid_sums; // Centroid sums: [dx*phi², dy*phi², phi²] per cell (alias)
  float *d_reduction_block; // Contiguous block: [integrals_x | integrals_y | perimeters | block_arrival | centroid_sums]
  size_t reduction_block_floats; // Total floats in the block (7 * capacity)
  int *d_block_arrival;   // Block arrival counter per cell (alias into d_reduction_block)
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
  float *d_v_A;               // Per-cell active motility speed
  std::vector<float> checkpoint_v_A; // v_A values loaded from checkpoint (used once, then cleared)
  float *d_gamma;             // Per-cell gradient coefficient (stiffness)
  float *d_two_gamma;         // Per-cell 2*gamma (precomputed for fused kernel)
  float *d_two_gamma_bulk;    // Per-cell 2*gamma*bulk_coeff (precomputed)
  std::vector<float> checkpoint_gamma; // Per-cell gamma loaded from checkpoint
  std::vector<SimParams::GammaOverride> gamma_overrides; // CLI --gamma V:selector overrides
  bool gamma_overrides_set = false;
  float *d_target_radius;     // Per-cell target radius
  float *d_target_area;       // Per-cell target area = π*R²
  float *d_volume_coeff;      // Per-cell volume constraint coefficient = μ/A₀
  std::vector<float> checkpoint_target_radius; // Per-cell radius from checkpoint
  std::vector<SimParams::RadiusOverride> radius_overrides; // CLI --radius V:selector overrides
  bool radius_overrides_set = false;
  float *d_centroids_x;       // Computed centroids X (GPU-side)
  float *d_centroids_y;       // Computed centroids Y (GPU-side)
  float *d_perimeters;        // Per-cell ∫|∇φ| dA (for normalized perimeter L_n)

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

  // GPU-accelerated bounding box update (every step, replaces interval-based CPU scan)
  int *d_bbox_scan_results;  // [num_cells * 9] scan output buffer
  size_t bbox_scan_capacity; // Allocated capacity in cells
  int step_counter;          // Internal step counter
  bool host_ptrs_stale;      // True when GPU pointers swapped but host not yet synced

  // Deferred bbox check: async D→H of change flag avoids pipeline drain.
  // Scan + check kernels launch async, result read on NEXT bbox-check step.
  int *d_bbox_any_change_flag;  // Device flag for deferred change detection
  int *h_bbox_any_change;       // Pinned host flag (async D→H, no pipeline drain)
  cudaEvent_t bbox_check_event; // Signals async bbox check complete
  bool bbox_async_pending;      // True when deferred bbox check is in flight

  // Cached max dimensions (updated on bbox change, avoids host loop every step)
  int cached_max_size;       // Max field_size across all cells
  int cached_max_w;          // Max width across all cells
  int cached_max_h;          // Max height across all cells
  bool cached_dims_valid;    // True if cached values are up-to-date

  // Contiguous phi memory pool (single large allocation for all cells)
  // Reduces TLB pressure from ~20k mappings to 1 at 10k cells.
  float *d_phi_pool;         // Single allocation: [phi slots... | phi_out slots...]
  size_t pool_slot_size;     // Floats per slot (>= max field_size across all cells)
  int pool_num_cells;        // Number of cells pool was allocated for
  bool pool_active;          // True once cells have been migrated to pool

  // Global sum field: S(x,y) = Σ_all φ_i²(x,y) on N×N grid.
  // Eliminates scattered neighbor reads in fused kernel:
  //   repulsion at pixel (x,y) = Σ_j φ_j²(x,y) = S(x,y) - φ_i²(x,y)
  // Single coalesced read replaces ~10 random neighbor reads per pixel.
  // Ping-pong: two buffers, async memset of the just-read buffer overlaps
  // with next step's compute, eliminating memset from critical path.
  float *d_sum_field;          // N*N global sum field A
  float *d_sum_field_b;        // N*N global sum field B (ping-pong)
  size_t sum_field_size;       // N*N (number of floats)
  cudaStream_t sum_field_clear_stream;     // Async stream for sum field memset
  cudaEvent_t sum_field_read_done_event;   // Signals fused kernel done reading
  cudaEvent_t sum_field_clear_done_event;  // Signals async memset complete

  // Laplacian sum field for adhesion: S_lap(x,y) = Σ_all ∇²φ_i(x,y)
  // Gradient coupling: δF/δφ_i = -ε Σ ∇²φ_j. Only allocated when adhesion_J > 0.
  // Same ping-pong strategy as sum_field above.
  float *d_sum_field_linear;     // N*N linear sum field A
  float *d_sum_field_linear_b;   // N*N linear sum field B (ping-pong)

public:
  Integrator(Method m = Method::ForwardEuler);
  ~Integrator();

  // Allocate/resize interaction arrays
  void allocate_interaction_arrays(int num_cells);
  void free_interaction_arrays();
  void update_interaction_arrays(const Domain &domain);

  // Contiguous phi pool management (fixed-page allocation)
  static size_t compute_max_page_size(const SimParams &params);
  void allocate_phi_pool(Domain &domain);
  void free_phi_pool();
  void grow_phi_pool(Domain &domain); // FATAL: should never be called with fixed pages

  // Global sum field management
  void allocate_sum_field(const Domain &domain);
  void free_sum_field();

  // Allocate/resize reduction arrays for fused kernels
  void allocate_reduction_arrays(int num_cells);
  void free_reduction_arrays();

  // Initialize/destroy CUDA streams
  void create_streams(int n = MAX_STREAMS);
  void destroy_streams();

  // Perform one time step
  // sync_polarization_to_host: if true, copy GPU polarization back to host cells
  //                           (only needed if saving trajectories)
  // sync_centroids_to_host: if true, copy centroids/volumes/velocities to host
  //                         (only needed on print/save/trajectory steps)
  void step(Domain &domain, float dt, bool sync_polarization_to_host = false,
            bool sync_centroids_to_host = false);

#ifdef DIAGNOSTICS_ENABLED
  // Run GPU-side diagnostic measurements
  // Must call step() first to ensure arrays are populated
  void compute_diagnostics(Domain &domain, DiagnosticBuffers &diag);
#endif

#ifdef STRESS_FIELDS_ENABLED
  // Compute stress tensor fields on GPU
  // Must call step() first to ensure arrays are populated
  void compute_stress_fields(Domain &domain, StressFieldBuffers &stress);
#endif
};

} // namespace cellsim
