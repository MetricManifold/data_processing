#pragma once

#include "domain3d.cuh"
#include "gpu_memory_tracker.cuh"
#include "kernels3d.cuh"
#include "types3d.cuh"
#include <vector>
#include <ctime>
#include <curand_kernel.h>

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
  FieldType3D **d_all_phi_ptrs;
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

  // GPU bounding box scan buffer (10 ints per cell for 3D)
  int *d_bbox_scan_results;
  size_t bbox_scan_capacity;

  // Global sum field for O(1) interaction (replaces neighbor-list interaction)
  // S(x,y,z) = Σ_all φ_k²(x,y,z) — scatter, then each cell reads S - φ_i²
  // Ping-pong: two buffers for async clear on separate stream
  float *d_sum_field;
  float *d_sum_field_b;           // Second buffer for ping-pong
  size_t sum_field_capacity;      // Allocated bytes per buffer
  cudaStream_t sum_field_clear_stream;   // Async clear stream
  cudaEvent_t sum_field_read_done_event; // Signals fused kernel done reading
  cudaEvent_t sum_field_clear_done_event;// Signals async clear complete
  bool use_fused_kernel;          // True when sum field is available

  // Cached max dimensions (avoids per-step host loop over cells)
  int cached_max_size;
  bool cached_dims_valid;

  // Fused kernel centroid sums precomputed flag
  bool centroid_sums_ready;

  // Contiguous phi pool: single cudaMalloc for ALL cells (phi + dphi_dt)
  // Eliminates per-cell malloc/free on bbox resize.
  // Layout: [phi_0][phi_1]...[phi_N-1][out_0][out_1]...[out_N-1]
  // Each slot = pool_slot_size floats (fixed page, never regrows).
  float *d_phi_pool;
  size_t pool_slot_size;  // Floats per slot (max possible field_size)
  int pool_num_cells;     // Capacity in cells
  bool pool_active;       // True after first allocation

  // Spatial hash grid for O(N) neighbor list building (large cell counts)
  static constexpr int MAX_GRID_CELLS = 512;   // Max spatial grid cells (8x8x8)
  static constexpr int MAX_CELLS_PER_GRID = 16; // Max cells per grid cell
  int *d_grid_counts;    // Cells per grid bin [MAX_GRID_CELLS]
  int *d_grid_cells;     // Cell indices [MAX_GRID_CELLS * MAX_CELLS_PER_GRID]

  // GPU-side RNG for polarization updates (curand)
  curandState *d_rng_states;  // RNG state per cell [num_cells]
  size_t rng_states_capacity; // Allocated capacity
  bool rng_initialized;       // True if RNG states are seeded

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
  // sync_to_host: if true, sync centroids/volumes/velocities back to host cells
  //               (needed for printing, checkpointing, trajectory output)
  //               In steady state, pass false for maximum GPU throughput.
  void step(Domain3D &domain, float dt, bool sync_to_host = false);

  // Allocate contiguous phi pool and migrate cells into it
  void allocate_phi_pool(Domain3D &domain);

  // Compute max possible page size for phi pool (from physics params)
  static size_t compute_max_page_size_3d(const SimParams3D &params);

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
      d_centroids_z(nullptr), bbox_update_interval(4), step_counter(0),
      d_neighbor_counts(nullptr), d_neighbor_lists(nullptr),
      neighbor_list_valid(false),
      d_bbox_scan_results(nullptr), bbox_scan_capacity(0),
      d_sum_field(nullptr), d_sum_field_b(nullptr), sum_field_capacity(0),
      sum_field_clear_stream(nullptr), sum_field_read_done_event(nullptr),
      sum_field_clear_done_event(nullptr), use_fused_kernel(false),
      cached_max_size(0), cached_dims_valid(false), centroid_sums_ready(false),
      d_phi_pool(nullptr), pool_slot_size(0), pool_num_cells(0), pool_active(false),
      d_grid_counts(nullptr), d_grid_cells(nullptr),
      d_rng_states(nullptr), rng_states_capacity(0), rng_initialized(false) {}

inline Integrator3D::~Integrator3D() {
  free_work_buffer();
  free_interaction_arrays();
  free_reduction_arrays();
  destroy_streams();
  // Free spatial grid buffers
  if (d_grid_counts) {
    cudaFree(d_grid_counts);
    d_grid_counts = nullptr;
  }
  if (d_grid_cells) {
    cudaFree(d_grid_cells);
    d_grid_cells = nullptr;
  }
  // Free sum field (both ping-pong buffers + stream/events)
  if (d_sum_field) { cudaFree(d_sum_field); d_sum_field = nullptr; }
  if (d_sum_field_b) { cudaFree(d_sum_field_b); d_sum_field_b = nullptr; }
  if (d_bbox_scan_results) { cudaFree(d_bbox_scan_results); d_bbox_scan_results = nullptr; }
  if (sum_field_clear_stream) { cudaStreamDestroy(sum_field_clear_stream); sum_field_clear_stream = nullptr; }
  if (sum_field_read_done_event) { cudaEventDestroy(sum_field_read_done_event); sum_field_read_done_event = nullptr; }
  if (sum_field_clear_done_event) { cudaEventDestroy(sum_field_clear_done_event); sum_field_clear_done_event = nullptr; }
  // Free phi pool
  if (d_phi_pool) { cudaFree(d_phi_pool); d_phi_pool = nullptr; pool_active = false; }
  // Free RNG states
  if (d_rng_states) {
    cudaFree(d_rng_states);
    d_rng_states = nullptr;
  }
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

  cudaMalloc(&d_all_phi_ptrs, num_cells * sizeof(FieldType3D *));
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

  // Spatial hash grid for O(N) neighbor finding (only needed for large N)
  // Allocate once at max capacity - memory is small (~36 KB)
  if (d_grid_counts == nullptr) {
    cudaMalloc(&d_grid_counts, MAX_GRID_CELLS * sizeof(int));
    cudaMalloc(&d_grid_cells, MAX_GRID_CELLS * MAX_CELLS_PER_GRID * sizeof(int));
  }

  // Allocate RNG states if needed
  if (num_cells > (int)rng_states_capacity) {
    if (d_rng_states) cudaFree(d_rng_states);
    cudaMalloc(&d_rng_states, num_cells * sizeof(curandState));
    rng_states_capacity = num_cells;
    rng_initialized = false; // Force re-initialization
  }

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

  std::vector<FieldType3D *> phi_ptrs(n);
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

  cudaMemcpy(d_all_phi_ptrs, phi_ptrs.data(), n * sizeof(FieldType3D *),
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

//=============================================================================
// Contiguous Phi Pool
//=============================================================================

inline size_t Integrator3D::compute_max_page_size_3d(const SimParams3D &params) {
  // Theoretical maximum bbox size for any cell.
  // half = max_dist + adaptive_margin + overshoot + safety
  int halo = params.halo_width;
  int adaptive_margin = static_cast<int>(2.0f * params.lambda) + halo;
  int overshoot = static_cast<int>(0.25f * adaptive_margin);
  int max_dist = static_cast<int>(params.target_radius + 3.0f * params.lambda) + 1;
  int max_half = max_dist + adaptive_margin + overshoot + 10;
  int max_side = 2 * max_half + 2 * halo;
  return static_cast<size_t>(max_side) * max_side * max_side;
}

inline void Integrator3D::allocate_phi_pool(Domain3D &domain) {
  int num_cells = domain.num_cells();
  if (num_cells == 0) return;

  size_t max_page = compute_max_page_size_3d(domain.params);
  for (int i = 0; i < num_cells; ++i)
    max_page = std::max(max_page, static_cast<size_t>(domain.cells[i]->field_size));

  if (pool_active && num_cells == pool_num_cells && max_page <= pool_slot_size)
    return;  // Already allocated

  if (pool_active) cudaDeviceSynchronize();  // Sync before realloc

  float *old_pool = d_phi_pool;
  pool_slot_size = max_page;
  pool_num_cells = num_cells;

  size_t total_floats = 2 * static_cast<size_t>(num_cells) * pool_slot_size;
  size_t alloc_bytes = total_floats * sizeof(float);

  {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("3D Phi pool %s: %d cells x %zu slot = %.1f MB  "
           "(VRAM: %.1f MB free / %.1f MB total)\n",
           pool_active ? "REALLOC" : "INIT", num_cells, pool_slot_size,
           alloc_bytes / (1024.0 * 1024.0),
           free_mem / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0));
  }

  cudaMalloc(&d_phi_pool, alloc_bytes);
  cudaMemset(d_phi_pool, 0, alloc_bytes);

  // Migrate cells into pool
  for (int i = 0; i < num_cells; ++i) {
    auto &cell = domain.cells[i];
    float *pool_phi = d_phi_pool + static_cast<size_t>(i) * pool_slot_size;
    float *pool_out = d_phi_pool + static_cast<size_t>(num_cells + i) * pool_slot_size;

    if (cell->d_phi && cell->field_size > 0) {
      cudaMemcpy(pool_phi, cell->d_phi, cell->field_size * sizeof(float),
                 cudaMemcpyDeviceToDevice);
    }

    if (!cell->pool_managed) {
      if (cell->d_phi) cudaFree(cell->d_phi);
      if (cell->d_dphi_dt) cudaFree(cell->d_dphi_dt);
    }

    cell->d_phi = reinterpret_cast<FieldType3D *>(pool_phi);
    cell->d_dphi_dt = reinterpret_cast<FieldType3D *>(pool_out);
    cell->pool_managed = true;
  }

  if (pool_active && old_pool) {
    cudaDeviceSynchronize();
    cudaFree(old_pool);
  }

  pool_active = true;
}

inline void Integrator3D::step(Domain3D &domain, float dt, bool sync_to_host) {
  if (domain.num_cells() == 0)
    return;

  allocate_reduction_arrays(domain.num_cells());

  // Allocate contiguous phi pool (migrates cells on first call, no-op after)
  allocate_phi_pool(domain);

  // Only update interaction arrays when bboxes change
  // (first call sets capacity=0 triggering initial update)
  if (interaction_array_capacity == 0) {
    update_interaction_arrays(domain);
  }

  const SimParams3D &params = domain.params;
  int num_cells = domain.num_cells();

  // Allocate sum field for O(1) interaction — ping-pong with async clear
  {
    size_t needed = (size_t)params.Nx * params.Ny * params.Nz * sizeof(float);
    if (needed > sum_field_capacity) {
      if (d_sum_field) cudaFree(d_sum_field);
      if (d_sum_field_b) cudaFree(d_sum_field_b);
      cudaMalloc(&d_sum_field, needed);
      cudaMalloc(&d_sum_field_b, needed);
      cudaMemset(d_sum_field, 0, needed);
      cudaMemset(d_sum_field_b, 0, needed);
      sum_field_capacity = needed;
      if (!sum_field_clear_stream) {
        cudaStreamCreate(&sum_field_clear_stream);
        cudaEventCreateWithFlags(&sum_field_read_done_event, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&sum_field_clear_done_event, cudaEventDisableTiming);
        cudaEventRecord(sum_field_clear_done_event, sum_field_clear_stream);
      }
      use_fused_kernel = true;
      printf("3D Sum field: %.1f MB x2 ping-pong (%dx%dx%d) — fused kernel enabled\n",
             needed / (1024.0 * 1024.0), params.Nx, params.Ny, params.Nz);
    }
  }

  // Allocate work buffer only if NOT using fused kernel
  if (!use_fused_kernel) {
    allocate_work_buffer(domain);
  }

  // Cache max dimensions (avoids host loop every step)
  if (!cached_dims_valid) {
    cached_max_size = 0;
    for (int i = 0; i < num_cells; ++i)
      cached_max_size = std::max(cached_max_size, domain.cells[i]->field_size);
    cached_dims_valid = true;
  }

  // Initialize RNG states on first use (GPU-side curand)
  if (!rng_initialized && d_rng_states != nullptr) {
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
    kernel_init_rng_states_3d<<<blocks, threads>>>(d_rng_states, seed, num_cells);
    
    // Upload initial polarizations from host cells
    std::vector<float> h_pol_x(num_cells), h_pol_y(num_cells), h_pol_z(num_cells);
    for (int i = 0; i < num_cells; ++i) {
      h_pol_x[i] = domain.cells[i]->polarization.x;
      h_pol_y[i] = domain.cells[i]->polarization.y;
      h_pol_z[i] = domain.cells[i]->polarization.z;
    }
    cudaMemcpy(d_polarization_x, h_pol_x.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polarization_y, h_pol_y.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polarization_z, h_pol_z.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    
    rng_initialized = true;
  }

  // Update polarization directions on GPU (no CPU→GPU transfer each step!)
  if (d_rng_states != nullptr && rng_initialized) {
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;
    bool use_abp = (params.motility_model == SimParams::MotilityModel::ABP);
    kernel_update_polarizations_3d<<<blocks, threads>>>(
        d_rng_states, d_polarization_x, d_polarization_y, d_polarization_z,
        dt, params.tau, use_abp, num_cells);
  }

  // Increment step counter
  step_counter++;
  // Bbox check interval: when to check if cells need resizing
  bool do_bbox_check = (step_counter == 1) || (step_counter % bbox_update_interval == 0);

  if (use_fused_kernel) {
    // =====================================================================
    // FUSED KERNEL PATH — matches 2D pattern: ~zero host blocking
    //
    // Per-step flow (steady state):
    //   1. ref_points kernel (tiny)
    //   2. centroids+deviations+velocities kernel (tiny, reads precomputed sums)
    //   3. zero centroid/volume accumulators (async)
    //   4. ping-pong sum_field select + wait for async clear
    //   5. scatter kernel (writes to sum_field)
    //   6. fused kernel (reads sum_field, writes phi in-place, accumulates sums)
    //   7. async clear of used sum_field on side stream
    //
    // NO cudaDeviceSynchronize in steady state.
    // D→H copies only when caller requests (save/print steps).
    // =====================================================================

    float dV = params.dx * params.dy * params.dz;
    float target_volume = params.target_volume();
    int max_size = cached_max_size;

    int threads_flat = 256;
    dim3 block(threads_flat, 1, 1);
    dim3 grid((max_size + threads_flat - 1) / threads_flat, num_cells, 1);
    int threads_1d = 256;
    int blocks_1d = (num_cells + threads_1d - 1) / threads_1d;

#ifdef ENABLE_KERNEL_PROFILING
    static cudaEvent_t ev[10];
    static bool ev_created = false;
    static double t_ref=0, t_centroid=0, t_zero=0, t_scatter=0, t_fused=0, t_clear=0, t_total=0;
    static int prof_n = 0;
    if (!ev_created) {
      for (int i=0;i<10;i++) cudaEventCreate(&ev[i]);
      ev_created = true;
    }
    cudaEventRecord(ev[0]); // start
#endif

    // 1-3. Reference points + centroids + velocities
    if (!centroid_sums_ready) {
      // First step or after bbox change: separate kernels with explicit reduction
      kernel_compute_ref_points_3d<<<blocks_1d, threads_1d>>>(
          d_ref_x, d_ref_y, d_ref_z, d_all_offsets_x, d_all_offsets_y,
          d_all_offsets_z, d_all_widths, d_all_heights, d_all_depths,
          params.Nx, params.Ny, params.Nz, num_cells);

      cudaMemsetAsync(d_volumes, 0, num_cells * sizeof(float));
      cudaMemsetAsync(d_centroid_sums, 0, num_cells * 4 * sizeof(float));

      int blocks_per_cell = std::min((max_size + 255) / 256, 32);
      dim3 reduce_grid(blocks_per_cell, num_cells);
      kernel_reduce_volumes_batched_3d<<<reduce_grid, 256,
                                         256 * sizeof(float)>>>(
          d_all_phi_ptrs, d_volumes, d_all_widths, d_all_heights, d_all_depths,
          d_all_field_sizes, params.halo_width, num_cells);
      kernel_reduce_centroid_sums_batched_3d<<<reduce_grid, 256,
                                               4 * 256 * sizeof(float)>>>(
          d_all_phi_ptrs, d_centroid_sums, d_all_widths, d_all_heights,
          d_all_depths, d_all_offsets_x, d_all_offsets_y, d_all_offsets_z,
          d_all_field_sizes, d_ref_x, d_ref_y, d_ref_z, params.halo_width,
          params.Nx, params.Ny, params.Nz, num_cells);

      kernel_compute_centroids_and_deviations_3d<<<blocks_1d, threads_1d>>>(
          d_centroids_x, d_centroids_y, d_centroids_z, d_volume_deviations,
          d_centroid_sums, d_volumes, d_ref_x, d_ref_y, d_ref_z,
          target_volume, dV, params.Nx, params.Ny, params.Nz, num_cells);
      // Velocity will be computed after scatter + velocity_integral (below)
      // For now, just zero the integrals so compute_velocities gives v_A only
      cudaMemsetAsync(d_integrals_x, 0, num_cells * sizeof(float));
      cudaMemsetAsync(d_integrals_y, 0, num_cells * sizeof(float));
      cudaMemsetAsync(d_integrals_z, 0, num_cells * sizeof(float));
      kernel_compute_velocities_3d<<<blocks_1d, threads_1d>>>(
          d_velocities_x, d_velocities_y, d_velocities_z,
          d_integrals_x, d_integrals_y, d_integrals_z,
          d_polarization_x, d_polarization_y, d_polarization_z,
          params.motility_coeff(), dV, params.v_A, num_cells);
    } else {
      // Steady state: single fused kernel (eliminates 2 kernel launch bubbles)
      kernel_ref_centroid_vel_fused_3d<<<blocks_1d, threads_1d>>>(
          d_ref_x, d_ref_y, d_ref_z,
          d_all_offsets_x, d_all_offsets_y, d_all_offsets_z,
          d_all_widths, d_all_heights, d_all_depths,
          d_centroids_x, d_centroids_y, d_centroids_z,
          d_volume_deviations, d_centroid_sums, d_volumes,
          target_volume, dV,
          d_velocities_x, d_velocities_y, d_velocities_z,
          d_polarization_x, d_polarization_y, d_polarization_z,
          params.v_A,
          params.Nx, params.Ny, params.Nz, num_cells);
    }

#ifdef ENABLE_KERNEL_PROFILING
    cudaEventRecord(ev[1]); // after ref+centroid+vel
#endif

    // 4. Zero accumulators for fused kernel (async, no blocking)
    cudaMemsetAsync(d_centroid_sums, 0, num_cells * 4 * sizeof(float));
    cudaMemsetAsync(d_volumes, 0, num_cells * sizeof(float));

    // 5. Ping-pong sum field: select buffer, wait for async clear
    float *current_sum_field = (step_counter % 2 == 0) ? d_sum_field : d_sum_field_b;
    if (sum_field_clear_done_event) {
      cudaStreamWaitEvent(0, sum_field_clear_done_event, 0);
    }

#ifdef ENABLE_KERNEL_PROFILING
    cudaEventRecord(ev[2]); // after zero+wait
#endif

    // 6. Scatter phi² to sum_field
    if (num_cells > 1) {
      kernel_scatter_phi_sq_3d<<<grid, block>>>(
          (float **)d_all_phi_ptrs, current_sum_field,
          d_all_widths, d_all_heights, d_all_depths,
          d_all_offsets_x, d_all_offsets_y, d_all_offsets_z,
          d_all_field_sizes,
          params.Nx, params.Ny, params.Nz, num_cells);
    }

#ifdef ENABLE_KERNEL_PROFILING
    cudaEventRecord(ev[3]); // after scatter
#endif

    // 6b. VELOCITY INTEGRAL: Compute v_I from current phi + sum field
    //     so the fused kernel uses CURRENT velocity for advection (no lag).
    if (num_cells > 1 && current_sum_field) {
      cudaMemsetAsync(d_integrals_x, 0, num_cells * sizeof(float));
      cudaMemsetAsync(d_integrals_y, 0, num_cells * sizeof(float));
      cudaMemsetAsync(d_integrals_z, 0, num_cells * sizeof(float));

      kernel_velocity_integral_3d<<<grid, block>>>(
          (float **)d_all_phi_ptrs,
          d_all_widths, d_all_heights, d_all_depths, d_all_field_sizes,
          d_all_offsets_x, d_all_offsets_y, d_all_offsets_z,
          current_sum_field,
          d_integrals_x, d_integrals_y, d_integrals_z,
          params.dx, params.dy, params.dz,
          params.halo_width, params.Nx, params.Ny, params.Nz,
          num_cells, max_size);

      float dV = params.dx * params.dy * params.dz;
      kernel_compute_velocities_3d<<<blocks_1d, threads_1d>>>(
          d_velocities_x, d_velocities_y, d_velocities_z,
          d_integrals_x, d_integrals_y, d_integrals_z,
          d_polarization_x, d_polarization_y, d_polarization_z,
          params.motility_coeff(), dV, params.v_A, num_cells);
    }

    // 7. FUSED KERNEL: Euler step + centroid/volume accumulation
    kernel_fused_step_3d<<<grid, block>>>(
        (float **)d_all_phi_ptrs,
        d_all_widths, d_all_heights, d_all_depths, d_all_field_sizes,
        d_all_offsets_x, d_all_offsets_y, d_all_offsets_z,
        (num_cells > 1) ? current_sum_field : nullptr,
        d_volume_deviations,
        d_velocities_x, d_velocities_y, d_velocities_z,
        d_centroid_sums, d_volumes,
        d_ref_x, d_ref_y, d_ref_z,
        params.volume_coeff(), params.interaction_coeff(), params.bulk_coeff(),
        params.gamma, params.dx, params.dy, params.dz, dt,
        params.halo_width, params.Nx, params.Ny, params.Nz,
        num_cells, max_size);

#ifdef ENABLE_KERNEL_PROFILING
    cudaEventRecord(ev[4]); // after fused kernel
#endif

    // 8. Async clear of used sum field on side stream (hides latency)
    if (num_cells > 1) {
      cudaEventRecord(sum_field_read_done_event, 0);
      cudaStreamWaitEvent(sum_field_clear_stream, sum_field_read_done_event, 0);
      cudaMemsetAsync(current_sum_field, 0, sum_field_capacity, sum_field_clear_stream);
      cudaEventRecord(sum_field_clear_done_event, sum_field_clear_stream);
    }

    // Fused kernel populated centroid_sums for next step
    centroid_sums_ready = true;

#ifdef ENABLE_KERNEL_PROFILING
    cudaEventRecord(ev[5]); // after async clear enqueue
    cudaEventSynchronize(ev[5]);
    float dt_ref, dt_zero, dt_scatter, dt_fused, dt_clear, dt_all;
    cudaEventElapsedTime(&dt_ref, ev[0], ev[1]);
    cudaEventElapsedTime(&dt_zero, ev[1], ev[2]);
    cudaEventElapsedTime(&dt_scatter, ev[2], ev[3]);
    cudaEventElapsedTime(&dt_fused, ev[3], ev[4]);
    cudaEventElapsedTime(&dt_clear, ev[4], ev[5]);
    cudaEventElapsedTime(&dt_all, ev[0], ev[5]);
    t_ref += dt_ref; t_zero += dt_zero; t_scatter += dt_scatter;
    t_fused += dt_fused; t_clear += dt_clear; t_total += dt_all;
    prof_n++;
    if (prof_n % 50 == 0) {
      float n = (float)prof_n;
      printf("\n=== 3D Fused Profile (avg %d steps, %d cells, max_fs=%d) ==="
             "\n  ref+cent+vel: %7.3f ms (%5.1f%%)"
             "\n  zero+wait:    %7.3f ms (%5.1f%%)"
             "\n  scatter:      %7.3f ms (%5.1f%%)"
             "\n  FUSED:        %7.3f ms (%5.1f%%)"
             "\n  async_clear:  %7.3f ms (%5.1f%%)"
             "\n  TOTAL:        %7.3f ms"
             "\n  grid: (%d, %d), block: %d\n",
             prof_n, num_cells, max_size,
             t_ref/n, 100*t_ref/t_total,
             t_zero/n, 100*t_zero/t_total,
             t_scatter/n, 100*t_scatter/t_total,
             t_fused/n, 100*t_fused/t_total,
             t_clear/n, 100*t_clear/t_total,
             t_total/n,
             grid.x, grid.y, threads_flat);
      t_ref=t_zero=t_scatter=t_fused=t_clear=t_total=0; prof_n=0;
    }
#endif

    // =====================================================================
    // CONDITIONAL SYNC: Only when caller needs host data (print/save steps)
    // In steady state, nothing below this point runs → zero blocking.
    // =====================================================================
    if (sync_to_host) {
      cudaDeviceSynchronize();
      std::vector<float> h_cx(num_cells), h_cy(num_cells), h_cz(num_cells);
      std::vector<float> h_vol(num_cells);
      std::vector<float> h_vx(num_cells), h_vy(num_cells), h_vz(num_cells);
      cudaMemcpy(h_cx.data(), d_centroids_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_cy.data(), d_centroids_y, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_cz.data(), d_centroids_z, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_vol.data(), d_volumes, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_vx.data(), d_velocities_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_vy.data(), d_velocities_y, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_vz.data(), d_velocities_z, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
      for (int i = 0; i < num_cells; ++i) {
        domain.cells[i]->centroid.x = h_cx[i];
        domain.cells[i]->centroid.y = h_cy[i];
        domain.cells[i]->centroid.z = h_cz[i];
        domain.cells[i]->volume = h_vol[i] * dV;
        domain.cells[i]->volume_deviation = target_volume - domain.cells[i]->volume;
        domain.cells[i]->velocity.x = h_vx[i];
        domain.cells[i]->velocity.y = h_vy[i];
        domain.cells[i]->velocity.z = h_vz[i];
      }
    }

    // =====================================================================
    // GPU-ACCELERATED BBOX UPDATE
    // Uses GPU scan + GPU early exit — NO cudaDeviceSynchronize in steady state.
    // Only syncs to host + CPU decision when cells actually need resize.
    // =====================================================================
    if (do_bbox_check) {
      // Allocate scan results buffer if needed
      if (bbox_scan_capacity < (size_t)num_cells) {
        if (d_bbox_scan_results) cudaFree(d_bbox_scan_results);
        bbox_scan_capacity = num_cells;
        cudaMalloc(&d_bbox_scan_results, num_cells * 10 * sizeof(int));
      }

      bool any_changed = gpu_update_all_bboxes_3d(
          domain, d_bbox_scan_results,
          d_centroids_x, d_centroids_y, d_centroids_z,
          (float **)d_all_phi_ptrs,
          d_all_widths, d_all_heights, d_all_depths,
          d_all_offsets_x, d_all_offsets_y, d_all_offsets_z,
          d_all_field_sizes, max_size);

      if (any_changed) {
        // GPU arrays already patched by gpu_update_all_bboxes_3d.
        // Only need to re-upload phi pointers (new allocations).
        update_interaction_arrays(domain);
        centroid_sums_ready = false;
        cached_dims_valid = false;
      }
    }

  } else {
    // =====================================================================
    // LEGACY 8-PHASE PATH (fallback when sum field not available)
    // =====================================================================
    bool rebuild_neighbors = !neighbor_list_valid || num_cells <= 1 || do_bbox_check;

    step_fused_3d(domain, dt, d_work_buffer, d_all_phi_ptrs, d_all_widths,
                  d_all_heights, d_all_depths, d_all_offsets_x, d_all_offsets_y,
                  d_all_offsets_z, d_all_field_sizes, d_volumes, d_integrals_x,
                  d_integrals_y, d_integrals_z, d_centroid_sums,
                  d_volume_deviations, d_velocities_x, d_velocities_y,
                  d_velocities_z, d_ref_x, d_ref_y, d_ref_z, d_polarization_x,
                  d_polarization_y, d_polarization_z, d_centroids_x,
                  d_centroids_y, d_centroids_z, d_neighbor_counts,
                  d_neighbor_lists, sync_to_host, rebuild_neighbors,
                  d_grid_counts, d_grid_cells, nullptr /* textures removed */);

    if (rebuild_neighbors && num_cells > 1) {
      neighbor_list_valid = true;
    }

    // Update bboxes periodically
    if (do_bbox_check) {
      bool any_changed = false;
      for (auto &cell : domain.cells) {
        if (cell->update_bounding_box(params)) {
          any_changed = true;
        }
      }
      if (any_changed) {
        update_interaction_arrays(domain);
        neighbor_list_valid = false;
      }
    }
  }
}

inline void Integrator3D::update_velocities(Domain3D &domain) {
  // Velocities are now computed on GPU inside step_fused_3d
  // This function is kept for compatibility but does nothing
}

} // namespace cellsim
