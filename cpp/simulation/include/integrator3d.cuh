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
  size_t interaction_array_capacity;

  // Device arrays for reduction outputs
  float *d_volumes;
  float *d_integrals_x;
  float *d_integrals_y;
  float *d_integrals_z;
  float *d_centroid_sums;
  size_t reduction_array_capacity;

  // Bounding box update control
  int bbox_update_interval;
  int step_counter;

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
      d_all_offsets_z(nullptr), interaction_array_capacity(0),
      d_volumes(nullptr), d_integrals_x(nullptr), d_integrals_y(nullptr),
      d_integrals_z(nullptr), d_centroid_sums(nullptr),
      reduction_array_capacity(0), bbox_update_interval(10), step_counter(0) {}

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

  // Parallel allocation: N cells × 7 buffers each (down from 10!)
  // Buffers per cell: laplacian, bulk, constraint, advection, reduction,
  //                   interaction_sum, repulsion
  // Fused kernel eliminates grad_x, grad_y, grad_z → 30% memory savings
  size_t needed = domain.num_cells() * 7 * max_size * sizeof(float);

  if (needed > work_buffer_size) {
    free_work_buffer();
    CUDA_MALLOC(&d_work_buffer, needed);
    work_buffer_size = needed;
    printf("3D Work buffer: %.1f MB for %d cells (7 buffers/cell)\n",
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

  d_all_phi_ptrs = nullptr;
  d_all_widths = nullptr;
  d_all_heights = nullptr;
  d_all_depths = nullptr;
  d_all_offsets_x = nullptr;
  d_all_offsets_y = nullptr;
  d_all_offsets_z = nullptr;
  interaction_array_capacity = 0;
}

inline void Integrator3D::update_interaction_arrays(const Domain3D &domain) {
  int n = domain.num_cells();
  allocate_interaction_arrays(n);

  std::vector<float *> phi_ptrs(n);
  std::vector<int> widths(n), heights(n), depths(n);
  std::vector<int> offsets_x(n), offsets_y(n), offsets_z(n);

  for (int i = 0; i < n; ++i) {
    phi_ptrs[i] = domain.cells[i]->d_phi;
    widths[i] = domain.cells[i]->width();
    heights[i] = domain.cells[i]->height();
    depths[i] = domain.cells[i]->depth();
    offsets_x[i] = domain.cells[i]->bbox_with_halo.x0;
    offsets_y[i] = domain.cells[i]->bbox_with_halo.y0;
    offsets_z[i] = domain.cells[i]->bbox_with_halo.z0;
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

  d_volumes = nullptr;
  d_integrals_x = nullptr;
  d_integrals_y = nullptr;
  d_integrals_z = nullptr;
  d_centroid_sums = nullptr;
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
  allocate_work_buffer(domain);
  update_interaction_arrays(domain);

  // Use the host function from kernels3d.cu
  step_euler_3d(domain, dt, d_work_buffer);

  // Periodically update bounding boxes
  step_counter++;
  if (step_counter >= bbox_update_interval) {
    domain.update_all_bounding_boxes();
    step_counter = 0;
  }
}

inline void Integrator3D::update_velocities(Domain3D &domain) {
  // For now, use constant velocity in polarization direction
  // TODO: Implement motility integral for 3D
  for (auto &cell : domain.cells) {
    cell->velocity = cell->polarization * domain.params.v_A;
  }
}

} // namespace cellsim
