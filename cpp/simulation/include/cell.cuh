#pragma once

#include "types.cuh"
#include <memory>
#include <vector>

namespace cellsim {

//=============================================================================
// Cell - Single cell with phase field on subdomain
//=============================================================================

class Cell {
public:
  int id;          // Unique cell identifier
  CellState state; // Cell state (active, dividing, etc.)

  // Subdomain definition
  BoundingBox bbox;           // Bounding box in global coordinates
  BoundingBox bbox_with_halo; // Including ghost cells

  // Phase field data (on subdomain, stored on device)
  float *d_phi;     // Device pointer to phase field φ
  float *d_dphi_dt; // Device pointer to time derivative
  int field_size;   // Total elements in subdomain

  // Cell properties (computed from φ)
  float volume;  // ∫φ² dx (area integral)
  Vec2 centroid; // Center of mass
  Vec2 velocity; // Cell velocity for motility

  // Self-propulsion (polarization)
  Vec2 polarization; // Unit vector p_n giving self-propulsion direction
  float theta;       // Polarization angle (for Brownian rotation)

  // Volume constraint
  float volume_deviation; // (πR² - ∫φ²) for constraint term

public:
  Cell(int id_, const BoundingBox &initial_bbox, int halo_width);
  ~Cell();

  // No copy (device memory)
  Cell(const Cell &) = delete;
  Cell &operator=(const Cell &) = delete;

  // Move allowed
  Cell(Cell &&other) noexcept;
  Cell &operator=(Cell &&other) noexcept;

  // Memory management
  void allocate_device_memory();
  void free_device_memory();

  // Initialize phase field (circular cell)
  void initialize_circular(float cx, float cy, float radius,
                           const SimParams &params);

  // Update bounding box based on current field
  // Returns true if bbox changed and reallocation needed
  bool update_bounding_box(const SimParams &params, float threshold = 0.01f);

  // Compute shape factor (circularity = 4πA/P² where A=area, P=perimeter)
  // Returns 1.0 for perfect circle, <1 for non-circular shapes
  float compute_shape_factor(const SimParams &params) const;

  // Get local index from local coordinates
  __host__ __device__ int local_index(int lx, int ly) const {
    return ly * bbox_with_halo.width() + lx;
  }

  // Get local index from global coordinates
  __host__ __device__ int global_to_index(int gx, int gy, int Nx,
                                          int Ny) const {
    int lx, ly;
    bbox_with_halo.global_to_local(gx, gy, lx, ly, Nx, Ny);
    return local_index(lx, ly);
  }

  // Subdomain dimensions (including halo)
  int width() const { return bbox_with_halo.width(); }
  int height() const { return bbox_with_halo.height(); }

  // Check if subdomain wraps around domain boundaries
  // If x0 < 0 or x1 > Nx, the subdomain wraps in x
  bool wraps_x(int Nx) const {
    return bbox_with_halo.x0 < 0 || bbox_with_halo.x1 > Nx;
  }
  bool wraps_y(int Ny) const {
    return bbox_with_halo.y0 < 0 || bbox_with_halo.y1 > Ny;
  }
};

//=============================================================================
// Cell Implementation
//=============================================================================

inline Cell::Cell(int id_, const BoundingBox &initial_bbox, int halo_width)
    : id(id_), state(CellState::Active), bbox(initial_bbox),
      bbox_with_halo(initial_bbox.expanded(halo_width)), d_phi(nullptr),
      d_dphi_dt(nullptr), field_size(0), volume(0), centroid{0, 0},
      velocity{0, 0}, polarization{1, 0}, theta(0), volume_deviation(0) {
  // Initialize with random polarization direction
  theta = static_cast<float>(rand()) / RAND_MAX * 2.0f * M_PI;
  polarization.x = cosf(theta);
  polarization.y = sinf(theta);
  allocate_device_memory();
}

inline Cell::~Cell() { free_device_memory(); }

inline Cell::Cell(Cell &&other) noexcept
    : id(other.id), state(other.state), bbox(other.bbox),
      bbox_with_halo(other.bbox_with_halo), d_phi(other.d_phi),
      d_dphi_dt(other.d_dphi_dt), field_size(other.field_size),
      volume(other.volume), centroid(other.centroid), velocity(other.velocity),
      polarization(other.polarization), theta(other.theta),
      volume_deviation(other.volume_deviation) {
  other.d_phi = nullptr;
  other.d_dphi_dt = nullptr;
}

inline Cell &Cell::operator=(Cell &&other) noexcept {
  if (this != &other) {
    free_device_memory();

    id = other.id;
    state = other.state;
    bbox = other.bbox;
    bbox_with_halo = other.bbox_with_halo;
    d_phi = other.d_phi;
    d_dphi_dt = other.d_dphi_dt;
    field_size = other.field_size;
    volume = other.volume;
    centroid = other.centroid;
    velocity = other.velocity;
    polarization = other.polarization;
    theta = other.theta;
    volume_deviation = other.volume_deviation;

    other.d_phi = nullptr;
    other.d_dphi_dt = nullptr;
  }
  return *this;
}

inline void Cell::allocate_device_memory() {
  field_size = bbox_with_halo.size();
  if (field_size > 0) {
    cudaMalloc(&d_phi, field_size * sizeof(float));
    cudaMalloc(&d_dphi_dt, field_size * sizeof(float));
    cudaMemset(d_phi, 0, field_size * sizeof(float));
    cudaMemset(d_dphi_dt, 0, field_size * sizeof(float));
  }
}

inline void Cell::free_device_memory() {
  if (d_phi) {
    cudaFree(d_phi);
    d_phi = nullptr;
  }
  if (d_dphi_dt) {
    cudaFree(d_dphi_dt);
    d_dphi_dt = nullptr;
  }
  field_size = 0;
}

inline void Cell::initialize_circular(float cx, float cy, float radius,
                                      const SimParams &params) {
  // Allocate temporary host buffer
  std::vector<float> h_phi(field_size);

  float lambda = params.lambda;

  // For a tanh profile φ = 0.5(1 - tanh((r-R)/w)) with w = sqrt(2)*λ,
  // the volume integral ∫φ² dA ≈ π*R² + π*w²/3 (due to interface contribution)
  // To get the correct target volume, we need to use a slightly smaller radius
  // Solve: π*R_init² + π*w²/3 = π*R_target²
  // => R_init = sqrt(R_target² - w²/3)
  float interface_width = sqrtf(2.0f) * lambda;
  float w2_over_3 = (interface_width * interface_width) / 3.0f;
  float effective_radius = radius;
  if (radius * radius > w2_over_3) {
    effective_radius = sqrtf(radius * radius - w2_over_3);
  }

  for (int ly = 0; ly < height(); ++ly) {
    for (int lx = 0; lx < width(); ++lx) {
      // Get global coordinates
      int gx, gy;
      bbox_with_halo.local_to_global(lx, ly, gx, gy, params.Nx, params.Ny);

      // Distance from cell center (with periodic BC)
      float dx = gx - cx;
      float dy = gy - cy;

      // Periodic distance
      if (dx > params.Nx * 0.5f)
        dx -= params.Nx;
      if (dx < -params.Nx * 0.5f)
        dx += params.Nx;
      if (dy > params.Ny * 0.5f)
        dy -= params.Ny;
      if (dy < -params.Ny * 0.5f)
        dy += params.Ny;

      float r = sqrtf(dx * dx + dy * dy);

      // Smooth interface: φ = 0.5 * (1 - tanh((r - R) / (sqrt(2)*λ)))
      // The sqrt(2) factor comes from the equilibrium profile of the
      // double-well potential This gives φ ≈ 1 inside, φ ≈ 0 outside, smooth
      // transition over width ~λ
      // Use effective_radius (not target radius) to get correct initial volume
      float phi =
          0.5f * (1.0f - tanhf((r - effective_radius) / interface_width));

      h_phi[local_index(lx, ly)] = phi;
    }
  }

  // Copy to device
  cudaMemcpy(d_phi, h_phi.data(), field_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Set initial centroid and volume
  centroid = {cx, cy};
  volume = M_PI * radius * radius; // Approximate, will be recomputed
}

inline float Cell::compute_shape_factor(const SimParams &params) const {
  // Compute circularity = 4πA/P²
  // Use the cell's computed volume (∫φ² dx) for area
  // Use ∫|∇φ| dx for perimeter (gives geometric perimeter for tanh profile)
  // A perfect circle has circularity ≈ 1

  std::vector<float> h_phi(field_size);
  cudaMemcpy(h_phi.data(), d_phi, field_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  float dx = params.dx;
  float dy = params.dy;
  int w = width();
  int h = height();

  float perimeter = 0.0f;

  for (int ly = 1; ly < h - 1; ++ly) {
    for (int lx = 1; lx < w - 1; ++lx) {
      int idx = ly * w + lx;

      // Perimeter: |∇φ| using central differences
      float phi_xp = h_phi[idx + 1];
      float phi_xm = h_phi[idx - 1];
      float phi_yp = h_phi[idx + w];
      float phi_ym = h_phi[idx - w];

      float grad_x = (phi_xp - phi_xm) / (2.0f * dx);
      float grad_y = (phi_yp - phi_ym) / (2.0f * dy);
      float grad_mag = sqrtf(grad_x * grad_x + grad_y * grad_y);

      perimeter += grad_mag;
    }
  }

  perimeter *= dx * dy; // ∫|∇φ| dA gives geometric perimeter for tanh profile

  if (perimeter < 1e-8f)
    return 1.0f;

  // Use the cell's tracked volume as area
  float area = volume; // Already computed as ∫φ² * dA

  // circularity = 4πA / P²
  float circularity = 4.0f * M_PI * area / (perimeter * perimeter);

  return circularity;
}

inline bool Cell::update_bounding_box(const SimParams &params,
                                      float threshold) {
  // Copy current field to host
  std::vector<float> h_phi(field_size);
  cudaMemcpy(h_phi.data(), d_phi, field_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  int halo = params.halo_width;
  int old_w = width();
  int old_h = height();
  int Nx = params.Nx;
  int Ny = params.Ny;

  // Find maximum extent from centroid using PERIODIC distance
  // This correctly handles cells that span domain boundaries
  // Also track local bounds to detect when cell touches window edge
  float max_dist_x = 0.0f;
  float max_dist_y = 0.0f;
  int min_lx = old_w, max_lx = -1;
  int min_ly = old_h, max_ly = -1;
  bool found_any = false;

  for (int ly = halo; ly < old_h - halo; ++ly) {
    for (int lx = halo; lx < old_w - halo; ++lx) {
      if (h_phi[local_index(lx, ly)] > threshold) {
        found_any = true;

        // Track local bounds for edge detection
        min_lx = std::min(min_lx, lx);
        max_lx = std::max(max_lx, lx);
        min_ly = std::min(min_ly, ly);
        max_ly = std::max(max_ly, ly);

        // Get global coordinates
        int gx, gy;
        bbox_with_halo.local_to_global(lx, ly, gx, gy, Nx, Ny);

        // Compute periodic distance from centroid
        float dx = static_cast<float>(gx) - centroid.x;
        float dy = static_cast<float>(gy) - centroid.y;

        // Wrap to nearest image
        if (dx > Nx * 0.5f)
          dx -= Nx;
        if (dx < -Nx * 0.5f)
          dx += Nx;
        if (dy > Ny * 0.5f)
          dy -= Ny;
        if (dy < -Ny * 0.5f)
          dy += Ny;

        // Track maximum extent in each direction
        max_dist_x = std::max(max_dist_x, fabsf(dx));
        max_dist_y = std::max(max_dist_y, fabsf(dy));
      }
    }
  }

  if (!found_any) {
    // No field above threshold - cell has disappeared
    return false;
  }

  // Add padding for cell growth and movement
  int padding = static_cast<int>(params.target_radius *
                                 (params.subdomain_padding - 1.0f)) +
                halo;

  // Use the already-computed centroid (which handles periodic boundaries
  // correctly) instead of computing new center from local bounds
  int new_cx = static_cast<int>(centroid.x);
  int new_cy = static_cast<int>(centroid.y);

  // Compute half-size from maximum periodic distance to any point with φ >
  // threshold
  int half_w = static_cast<int>(max_dist_x) + padding;
  int half_h = static_cast<int>(max_dist_y) + padding;

  // Minimum size
  half_w = std::max(half_w, static_cast<int>(params.target_radius *
                                             params.subdomain_padding));
  half_h = std::max(half_h, static_cast<int>(params.target_radius *
                                             params.subdomain_padding));

  // New bounding box centered on centroid
  BoundingBox new_bbox = {new_cx - half_w, new_cy - half_h, new_cx + half_w,
                          new_cy + half_h};

  // Add halo
  BoundingBox new_bbox_with_halo = {new_bbox.x0 - halo, new_bbox.y0 - halo,
                                    new_bbox.x1 + halo, new_bbox.y1 + halo};

  // Check if we need to reallocate
  int old_cx = (bbox_with_halo.x0 + bbox_with_halo.x1) / 2;
  int old_cy = (bbox_with_halo.y0 + bbox_with_halo.y1) / 2;

  int shift_x = new_cx - old_cx;
  int shift_y = new_cy - old_cy;

  // Handle periodic wrapping of shift
  if (shift_x > Nx / 2)
    shift_x -= Nx;
  if (shift_x < -Nx / 2)
    shift_x += Nx;
  if (shift_y > Ny / 2)
    shift_y -= Ny;
  if (shift_y < -Ny / 2)
    shift_y += Ny;

  // Check if cell is touching window boundary (rare but critical case)
  // This happens when a cell is pushed/flattened toward the edge
  bool touching_edge = (min_lx <= halo + 1) || (max_lx >= old_w - halo - 2) ||
                       (min_ly <= halo + 1) || (max_ly >= old_h - halo - 2);

  // Reposition if:
  // 1. Center moved significantly (conservative threshold for performance), OR
  // 2. Cell is touching window boundary (rare, but must handle to prevent
  // cutoff)
  if (abs(shift_x) < 5 && abs(shift_y) < 5 && !touching_edge) {
    return false;
  }

  // Allocate new field
  int new_size = new_bbox_with_halo.size();
  float *d_phi_new = nullptr;
  float *d_dphi_dt_new = nullptr;
  cudaMalloc(&d_phi_new, new_size * sizeof(float));
  cudaMalloc(&d_dphi_dt_new, new_size * sizeof(float));
  cudaMemset(d_phi_new, 0, new_size * sizeof(float));
  cudaMemset(d_dphi_dt_new, 0, new_size * sizeof(float));

  // Copy old field to new position
  int new_w = new_bbox_with_halo.width();
  int new_h = new_bbox_with_halo.height();

  std::vector<float> h_phi_new(new_size, 0.0f);

  for (int old_ly = 0; old_ly < old_h; ++old_ly) {
    for (int old_lx = 0; old_lx < old_w; ++old_lx) {
      // Get global coordinate from old bbox
      int gx, gy;
      bbox_with_halo.local_to_global(old_lx, old_ly, gx, gy, params.Nx,
                                     params.Ny);

      // Convert to local coordinate in new bbox
      int new_lx, new_ly;
      new_bbox_with_halo.global_to_local(gx, gy, new_lx, new_ly, params.Nx,
                                         params.Ny);

      if (new_lx >= 0 && new_lx < new_w && new_ly >= 0 && new_ly < new_h) {
        h_phi_new[new_ly * new_w + new_lx] = h_phi[old_ly * old_w + old_lx];
      }
    }
  }

  // Upload new field
  cudaMemcpy(d_phi_new, h_phi_new.data(), new_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Free old memory and update pointers
  cudaFree(d_phi);
  cudaFree(d_dphi_dt);

  d_phi = d_phi_new;
  d_dphi_dt = d_dphi_dt_new;
  field_size = new_size;
  bbox = new_bbox;
  bbox_with_halo = new_bbox_with_halo;

  return true;
}

} // namespace cellsim
