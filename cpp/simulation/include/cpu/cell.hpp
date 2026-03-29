#pragma once

#include "physics.hpp"
#include "types.hpp"
#include <memory>
#include <vector>

namespace cellsim {

/**
 * Cell class for CPU implementation.
 *
 * Each cell maintains its own phase field on a subdomain. The subdomain is
 * sized to contain the cell with padding for boundary conditions.
 */
class Cell {
public:
  Cell() = default;

  /**
   * Initialize cell with position and radius.
   * Creates the subdomain and initializes the phase field with tanh profile.
   * padding_factor should be >= 2.0 to capture the full tanh interface (which
   * extends ~3λ beyond R) plus room for growth. Default 2.0 matches CUDA.
   */
  void initialize(int id, float cx, float cy, float radius, float lambda,
                  float dx, float dy, int global_Lx, int global_Ly,
                  float padding_factor = 2.0f);

  /**
   * Update subdomain bounds and resize if necessary.
   * Called when cell position changes significantly.
   */
  void update_subdomain(float new_cx, float new_cy, float margin_factor = 2.0f);

  /**
   * Compute volume integral: V = Σ φ² * dx * dy
   */
  float compute_volume(float dx, float dy) const;

  /**
   * Compute center of mass: (Σ x*φ² / Σ φ², Σ y*φ² / Σ φ²)
   */
  void compute_center_of_mass(float dx, float dy, float &com_x,
                              float &com_y) const;

  /**
   * Copy phase field values to a global array at correct positions.
   */
  void copy_to_global(float *global_phi, int global_Lx, int global_Ly) const;

  /**
   * Sample phase field values from other cells for repulsion calculation.
   */
  void sample_interaction_field(const std::vector<Cell> &cells,
                                int exclude_id);

  // Accessors
  int get_id() const { return id_; }
  float get_cx() const { return cx_; }
  float get_cy() const { return cy_; }
  float get_vx() const { return vx_; }
  float get_vy() const { return vy_; }
  void set_velocity(float vx, float vy) {
    vx_ = vx;
    vy_ = vy;
  }

  float get_theta() const { return theta_; }
  void set_theta(float theta) { theta_ = theta; }

  const BoundingBox &get_bounds() const { return bounds_; }
  int get_local_Lx() const { return local_Lx_; }
  int get_local_Ly() const { return local_Ly_; }

  float *get_phi() { return phi_.data(); }
  const float *get_phi() const { return phi_.data(); }
  float *get_phi_new() { return phi_new_.data(); }
  const float *get_phi_new() const { return phi_new_.data(); }
  float *get_interaction_sum() { return interaction_sum_.data(); }
  const float *get_interaction_sum() const { return interaction_sum_.data(); }

  // Swap phi and phi_new buffers
  void swap_buffers() { std::swap(phi_, phi_new_); }

  // Target volume accessor
  float get_target_volume() const { return target_volume_; }
  void set_target_volume(float v) { target_volume_ = v; }

private:
  int id_ = -1;
  float cx_ = 0.0f; // Current center x
  float cy_ = 0.0f; // Current center y
  float vx_ = 0.0f; // Velocity x
  float vy_ = 0.0f; // Velocity y
  float theta_ = 0.0f; // Orientation angle for motility

  BoundingBox bounds_;
  int local_Lx_ = 0;
  int local_Ly_ = 0;
  int global_Lx_ = 0;
  int global_Ly_ = 0;

  float target_volume_ = 0.0f;

  std::vector<float> phi_;             // Current phase field
  std::vector<float> phi_new_;         // Next phase field
  std::vector<float> interaction_sum_; // Σ_j φ_j² from other cells
};

} // namespace cellsim
