#pragma once

#include "types.hpp"
#include <memory>
#include <vector>

namespace cellsim {

//=============================================================================
// Cell - Single cell with phase field on subdomain (CPU version)
//=============================================================================

class Cell {
public:
  int id;          // Unique cell identifier
  CellState state; // Cell state (active, dividing, etc.)

  // Subdomain definition
  BoundingBox bbox;           // Bounding box in global coordinates
  BoundingBox bbox_with_halo; // Including ghost cells

  // Phase field data (on subdomain, stored in CPU memory)
  std::vector<float> phi;     // Phase field φ
  std::vector<float> dphi_dt; // Time derivative buffer
  int field_size;             // Total elements in subdomain

  // Cell properties (computed from φ)
  float volume;  // ∫φ² dx (area integral)
  Vec2 centroid; // Center of mass
  Vec2 velocity; // Cell velocity for motility

  // Self-propulsion (polarization)
  Vec2 polarization; // Unit vector giving self-propulsion direction
  float theta;       // Polarization angle

  // Volume constraint
  float volume_deviation; // (πR² - ∫φ²) for constraint term

public:
  Cell(int id_, const BoundingBox &initial_bbox, int halo_width);
  ~Cell() = default;

  // Copy and move
  Cell(const Cell &other) = default;
  Cell &operator=(const Cell &other) = default;
  Cell(Cell &&other) noexcept = default;
  Cell &operator=(Cell &&other) noexcept = default;

  // Memory management
  void allocate_memory();

  // Initialize phase field (circular cell)
  void initialize_circular(float cx, float cy, float radius,
                           const SimParams &params);

  // Update bounding box based on current field
  bool update_bounding_box(const SimParams &params, float threshold = 0.01f);

  // Get local index from local coordinates
  int local_index(int lx, int ly) const {
    return ly * bbox_with_halo.width() + lx;
  }

  // Subdomain dimensions (including halo)
  int width() const { return bbox_with_halo.width(); }
  int height() const { return bbox_with_halo.height(); }

  // Check if subdomain wraps around domain boundaries
  bool wraps_x(int Nx) const {
    return bbox_with_halo.x0 < 0 || bbox_with_halo.x1 > Nx;
  }
  bool wraps_y(int Ny) const {
    return bbox_with_halo.y0 < 0 || bbox_with_halo.y1 > Ny;
  }
  
  // Accessors for MPI communication
  const BoundingBox& get_bbox() const { return bbox; }
  const BoundingBox& get_bbox_with_halo() const { return bbox_with_halo; }
  int get_field_size() const { return field_size; }
  float cx() const { return centroid.x; }
  float cy() const { return centroid.y; }
  float vx() const { return velocity.x; }
  float vy() const { return velocity.y; }
  float px() const { return polarization.x; }
  float py() const { return polarization.y; }
  
  // Copy phi to host buffer (for CPU version, just copy)
  void copy_phi_to_host(float* dest) const {
    std::copy(phi.begin(), phi.end(), dest);
  }
  
  // Update phi from host buffer (for receiving ghost cells)
  void update_phi_from_host(const float* src) {
    std::copy(src, src + field_size, phi.begin());
  }
};

} // namespace cellsim
