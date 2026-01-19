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

  // Spatial hash grid for O(N) neighbor list building (large cell counts)
  static constexpr int MAX_GRID_CELLS = 512;   // Max spatial grid cells (8x8x8)
  static constexpr int MAX_CELLS_PER_GRID = 16; // Max cells per grid cell
  int *d_grid_counts;    // Cells per grid bin [MAX_GRID_CELLS]
  int *d_grid_cells;     // Cell indices [MAX_GRID_CELLS * MAX_CELLS_PER_GRID]

  // GPU-side RNG for polarization updates (curand)
  curandState *d_rng_states;  // RNG state per cell [num_cells]
  size_t rng_states_capacity; // Allocated capacity
  bool rng_initialized;       // True if RNG states are seeded

  // =========================================================================
  // Texture Memory for Interaction Kernel Optimization
  // =========================================================================
  // 3D textures provide hardware-cached spatial locality reads for neighbor φ values.
  // This significantly improves scattered memory access patterns in the interaction kernel.
  //
  // Structure:
  // - d_phi_textures[i] = texture object for cell i's phi field
  // - d_phi_arrays[i] = CUDA 3D array backing the texture
  // - We update textures by copying from cell phi fields each step
  //
  // Note: Texture updates cost ~0.5ms per step for 32 cells, but the interaction
  // kernel speedup (30-50%) more than compensates.
  // =========================================================================
  cudaTextureObject_t *h_phi_textures;  // HOST array of texture objects [num_cells]
  cudaTextureObject_t *d_phi_textures;  // DEVICE array of texture objects [num_cells]
  cudaArray_t *d_phi_arrays;            // 3D CUDA arrays [num_cells] (host ptrs)
  size_t texture_array_capacity;         // Number of textures allocated
  int *texture_dims;                     // [3*num_cells] = {w0,h0,d0, w1,h1,d1, ...}
  bool textures_valid;                   // True if textures match current phi fields

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

  // Texture memory management for interaction kernel optimization
  void allocate_textures(const Domain3D &domain);
  void free_textures();
  void update_textures(const Domain3D &domain);

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
      neighbor_list_valid(false), d_grid_counts(nullptr), d_grid_cells(nullptr),
      d_rng_states(nullptr), rng_states_capacity(0), rng_initialized(false),
      h_phi_textures(nullptr), d_phi_textures(nullptr), d_phi_arrays(nullptr), 
      texture_array_capacity(0), texture_dims(nullptr), textures_valid(false) {}

inline Integrator3D::~Integrator3D() {
  free_work_buffer();
  free_interaction_arrays();
  free_reduction_arrays();
  free_textures();
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
// Texture Memory Management for Interaction Kernel Optimization
//=============================================================================

inline void Integrator3D::allocate_textures(const Domain3D &domain) {
  int num_cells = domain.num_cells();
  if (num_cells == 0) return;
  
  // Check if we need to reallocate (cell count changed or first allocation)
  if (num_cells != (int)texture_array_capacity) {
    free_textures();
    
    // Allocate HOST arrays for texture objects and CUDA arrays (management)
    h_phi_textures = new cudaTextureObject_t[num_cells];
    d_phi_arrays = new cudaArray_t[num_cells];
    texture_dims = new int[3 * num_cells];
    
    // Allocate DEVICE array for texture objects (passed to kernel)
    cudaMalloc(&d_phi_textures, num_cells * sizeof(cudaTextureObject_t));
    
    for (int i = 0; i < num_cells; ++i) {
      h_phi_textures[i] = 0;
      d_phi_arrays[i] = nullptr;
    }
    
    texture_array_capacity = num_cells;
  }
  
  // Create/recreate textures for each cell
  for (int i = 0; i < num_cells; ++i) {
    const Cell3D *cell = domain.cells[i].get();
    int w = cell->width();
    int h = cell->height();
    int d = cell->depth();
    
    // Check if dimensions changed (bbox resize)
    bool dims_changed = (texture_dims[3*i] != w || 
                         texture_dims[3*i+1] != h || 
                         texture_dims[3*i+2] != d);
    
    if (d_phi_arrays[i] != nullptr && !dims_changed) {
      // Already allocated with correct size
      continue;
    }
    
    // Destroy old texture/array if exists
    if (h_phi_textures[i] != 0) {
      cudaDestroyTextureObject(h_phi_textures[i]);
      h_phi_textures[i] = 0;
    }
    if (d_phi_arrays[i] != nullptr) {
      cudaFreeArray(d_phi_arrays[i]);
      d_phi_arrays[i] = nullptr;
    }
    
    // Store dimensions
    texture_dims[3*i] = w;
    texture_dims[3*i+1] = h;
    texture_dims[3*i+2] = d;
    
    // Create 3D CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(w, h, d);
    
    cudaError_t err = cudaMalloc3DArray(&d_phi_arrays[i], &channelDesc, extent);
    if (err != cudaSuccess) {
      printf("ERROR: cudaMalloc3DArray failed for cell %d (%dx%dx%d): %s\n",
             i, w, h, d, cudaGetErrorString(err));
      continue;
    }
    
    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_phi_arrays[i];
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp at boundaries
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint; // No interpolation (nearest)
    texDesc.readMode = cudaReadModeElementType; // Read as float
    texDesc.normalizedCoords = 0; // Use unnormalized (integer) coordinates
    
    err = cudaCreateTextureObject(&h_phi_textures[i], &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
      printf("ERROR: cudaCreateTextureObject failed for cell %d: %s\n",
             i, cudaGetErrorString(err));
    }
  }
  
  // Copy texture handles from host to device array (kernel needs device pointers)
  cudaMemcpy(d_phi_textures, h_phi_textures, 
             num_cells * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
  
  textures_valid = false; // Need to update texture contents
}

inline void Integrator3D::free_textures() {
  if (h_phi_textures != nullptr && d_phi_arrays != nullptr) {
    for (size_t i = 0; i < texture_array_capacity; ++i) {
      if (h_phi_textures[i] != 0) {
        cudaDestroyTextureObject(h_phi_textures[i]);
      }
      if (d_phi_arrays[i] != nullptr) {
        cudaFreeArray(d_phi_arrays[i]);
      }
    }
    delete[] h_phi_textures;
    delete[] d_phi_arrays;
    h_phi_textures = nullptr;
    d_phi_arrays = nullptr;
  }
  if (d_phi_textures != nullptr) {
    cudaFree(d_phi_textures);
    d_phi_textures = nullptr;
  }
  if (texture_dims != nullptr) {
    delete[] texture_dims;
    texture_dims = nullptr;
  }
  texture_array_capacity = 0;
  textures_valid = false;
}

inline void Integrator3D::update_textures(const Domain3D &domain) {
  int num_cells = domain.num_cells();
  if (num_cells == 0 || d_phi_arrays == nullptr || texture_dims == nullptr) return;
  
  // Copy phi data from device linear memory to 3D CUDA arrays
  for (int i = 0; i < num_cells; ++i) {
    const Cell3D *cell = domain.cells[i].get();
    int w = cell->width();
    int h = cell->height();
    int d = cell->depth();
    
    if (d_phi_arrays[i] == nullptr) continue;
    
    // Use the ALLOCATED texture dimensions, not current cell dimensions
    // If they don't match, the texture needs reallocation (handled by allocate_textures)
    int tex_w = texture_dims[3*i];
    int tex_h = texture_dims[3*i+1];
    int tex_d = texture_dims[3*i+2];
    
    // Skip if dimensions mismatch (texture needs reallocation)
    if (w != tex_w || h != tex_h || d != tex_d) {
      // Don't report error - this is expected when bbox changes
      continue;
    }
    
    // Handle FP16 storage: need to convert to FP32 for texture
#ifdef USE_HALF_PRECISION_3D
    // For FP16, we need to convert to float first
    // This adds overhead but maintains texture cache benefits
    size_t num_elements = (size_t)w * h * d;
    float *h_temp = new float[num_elements];
    __half *d_half_temp = cell->d_phi;
    
    // Convert on GPU and copy
    // For now, use a simpler approach: copy half to host, convert, copy to array
    std::vector<__half> h_half(num_elements);
    cudaMemcpy(h_half.data(), d_half_temp, num_elements * sizeof(__half), cudaMemcpyDeviceToHost);
    for (size_t j = 0; j < num_elements; ++j) {
      h_temp[j] = __half2float(h_half[j]);
    }
    
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(h_temp, w * sizeof(float), w, h);
    copyParams.dstArray = d_phi_arrays[i];
    copyParams.extent = make_cudaExtent(w, h, d);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    
    delete[] h_temp;
#else
    // FP32: Direct device-to-device copy (fast)
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(
        (void*)cell->d_phi, w * sizeof(float), w, h);
    copyParams.dstArray = d_phi_arrays[i];
    copyParams.extent = make_cudaExtent(w, h, d);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    
    cudaError_t err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess) {
      printf("ERROR: cudaMemcpy3D failed for cell %d: %s\n", 
             i, cudaGetErrorString(err));
    }
#endif
  }
  
  textures_valid = true;
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

  // Texture memory optimization DISABLED
  // Analysis showed the per-step copy overhead (500MB+ D2D via cudaMemcpy3D)
  // exceeds any benefit from texture cache. Texture memory is better suited for
  // static/infrequently-updated data, not phi fields that change every step.
  // allocate_textures(domain);

  const SimParams3D &params = domain.params;
  int num_cells = domain.num_cells();

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

  // Update texture contents with current phi fields
  // This copies phi data to 3D CUDA arrays for hardware-cached reads
  // Note: This is done every step since phi fields change
  if (d_phi_textures != nullptr) {
    update_textures(domain);
  }

  // Increment step counter and determine if we need to sync centroids
  step_counter++;
  bool sync_centroids =
      (step_counter == 1) || (step_counter % bbox_update_interval == 0);

  // Determine if neighbor list rebuild is needed
  // Rebuild on first step, when bboxes sync, or when explicitly invalidated
  bool rebuild_neighbors = !neighbor_list_valid || num_cells <= 1 || sync_centroids;

  // Use optimized fused step function with neighbor list
  // Note: polarizations are already on GPU, step_fused_3d skips the upload
  // Pass spatial grid buffers for O(N) neighbor finding with large cell counts
  // Pass texture objects for optimized interaction kernel (nullptr = fallback to direct access)
  step_fused_3d(domain, dt, d_work_buffer, d_all_phi_ptrs, d_all_widths,
                d_all_heights, d_all_depths, d_all_offsets_x, d_all_offsets_y,
                d_all_offsets_z, d_all_field_sizes, d_volumes, d_integrals_x,
                d_integrals_y, d_integrals_z, d_centroid_sums,
                d_volume_deviations, d_velocities_x, d_velocities_y,
                d_velocities_z, d_ref_x, d_ref_y, d_ref_z, d_polarization_x,
                d_polarization_y, d_polarization_z, d_centroids_x,
                d_centroids_y, d_centroids_z, d_neighbor_counts,
                d_neighbor_lists, sync_centroids, rebuild_neighbors,
                d_grid_counts, d_grid_cells, d_phi_textures);

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
