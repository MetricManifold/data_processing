#include "cell.hpp"
#include <algorithm>
#include <cmath>

namespace cellsim {

void Cell::initialize(int id, float cx, float cy, float radius, float lambda,
                      float dx, float dy, int global_Lx, int global_Ly,
                      float padding_factor) {
  id_ = id;
  cx_ = cx;
  cy_ = cy;
  global_Lx_ = global_Lx;
  global_Ly_ = global_Ly;

  // Compute subdomain bounds with padding
  // Unlike before, we allow coordinates outside [0, N) for periodic wrapping
  float margin = radius * padding_factor;
  int half_w = static_cast<int>(std::ceil(margin / dx));
  int half_h = static_cast<int>(std::ceil(margin / dy));
  
  // Center the bounding box on the cell center
  // Coordinates can be negative or exceed domain size - that's okay for periodic BC
  int cx_grid = static_cast<int>(std::round(cx / dx));
  int cy_grid = static_cast<int>(std::round(cy / dy));
  
  bounds_.x_min = cx_grid - half_w;
  bounds_.x_max = cx_grid + half_w;
  bounds_.y_min = cy_grid - half_h;
  bounds_.y_max = cy_grid + half_h;

  local_Lx_ = bounds_.x_max - bounds_.x_min + 1;
  local_Ly_ = bounds_.y_max - bounds_.y_min + 1;

  // Allocate buffers
  int size = local_Lx_ * local_Ly_;
  phi_.resize(size, 0.0f);
  phi_new_.resize(size, 0.0f);
  interaction_sum_.resize(size, 0.0f);

  // Initialize phase field with tanh profile
  float R_eff = effective_radius_2d(radius, lambda);

  for (int ly = 0; ly < local_Ly_; ++ly) {
    for (int lx = 0; lx < local_Lx_; ++lx) {
      // Get global coordinates with periodic wrapping
      int gx, gy;
      bounds_.local_to_global(lx, ly, gx, gy, global_Lx, global_Ly);
      
      float x = gx * dx;
      float y = gy * dy;

      // Compute distance with periodic boundary conditions
      float dist_x = x - cx;
      float dist_y = y - cy;
      
      // Wrap to nearest image
      if (dist_x > global_Lx * dx * 0.5f) dist_x -= global_Lx * dx;
      if (dist_x < -global_Lx * dx * 0.5f) dist_x += global_Lx * dx;
      if (dist_y > global_Ly * dy * 0.5f) dist_y -= global_Ly * dy;
      if (dist_y < -global_Ly * dy * 0.5f) dist_y += global_Ly * dy;
      
      float dist = std::sqrt(dist_x * dist_x + dist_y * dist_y);
      int idx = ly * local_Lx_ + lx;
      phi_[idx] = tanh_profile(dist, R_eff, lambda);
    }
  }

  // Target volume is pi*R^2 (the desired final area), NOT the initial volume
  // This matches the CUDA backend which uses params.target_area() = pi*R^2
  target_volume_ = 3.14159265f * radius * radius;
}

void Cell::update_subdomain(float new_cx, float new_cy, float margin_factor) {
  // Update cell center
  float old_cx = cx_;
  float old_cy = cy_;
  cx_ = new_cx;
  cy_ = new_cy;
  
  // Check if the cell has moved significantly (more than 5 grid points)
  // If so, we need to reposition the subdomain
  float Lx = global_Lx_;
  float Ly = global_Ly_;
  
  // Compute periodic displacement
  float disp_x = new_cx - old_cx;
  float disp_y = new_cy - old_cy;
  if (disp_x > Lx * 0.5f) disp_x -= Lx;
  if (disp_x < -Lx * 0.5f) disp_x += Lx;
  if (disp_y > Ly * 0.5f) disp_y -= Ly;
  if (disp_y < -Ly * 0.5f) disp_y += Ly;
  
  // Check distance from subdomain center
  float bbox_cx = (bounds_.x_min + bounds_.x_max) * 0.5f;
  float bbox_cy = (bounds_.y_min + bounds_.y_max) * 0.5f;
  
  float shift_x = new_cx - bbox_cx;
  float shift_y = new_cy - bbox_cy;
  
  // Handle periodic wrapping of shift
  if (shift_x > Lx * 0.5f) shift_x -= Lx;
  if (shift_x < -Lx * 0.5f) shift_x += Lx;
  if (shift_y > Ly * 0.5f) shift_y -= Ly;
  if (shift_y < -Ly * 0.5f) shift_y += Ly;
  
  // Only reposition if shift is significant
  if (std::abs(shift_x) < 5 && std::abs(shift_y) < 5) {
    return;
  }
  
  // Create new bounding box centered on new position
  int half_w = local_Lx_ / 2;
  int half_h = local_Ly_ / 2;
  
  int new_cx_grid = static_cast<int>(std::round(new_cx));
  int new_cy_grid = static_cast<int>(std::round(new_cy));
  
  BoundingBox new_bounds;
  new_bounds.x_min = new_cx_grid - half_w;
  new_bounds.x_max = new_cx_grid + half_w;
  new_bounds.y_min = new_cy_grid - half_h;
  new_bounds.y_max = new_cy_grid + half_h;
  
  int new_Lx = new_bounds.x_max - new_bounds.x_min + 1;
  int new_Ly = new_bounds.y_max - new_bounds.y_min + 1;
  
  // Allocate new buffers
  std::vector<float> new_phi(new_Lx * new_Ly, 0.0f);
  
  // Copy data from old subdomain to new subdomain
  for (int old_ly = 0; old_ly < local_Ly_; ++old_ly) {
    for (int old_lx = 0; old_lx < local_Lx_; ++old_lx) {
      // Get global coordinate from old bbox
      int gx, gy;
      bounds_.local_to_global(old_lx, old_ly, gx, gy, global_Lx_, global_Ly_);
      
      // Convert to local coordinate in new bbox
      int new_lx, new_ly;
      new_bounds.global_to_local(gx, gy, new_lx, new_ly, global_Lx_, global_Ly_);
      
      if (new_lx >= 0 && new_lx < new_Lx && new_ly >= 0 && new_ly < new_Ly) {
        new_phi[new_ly * new_Lx + new_lx] = phi_[old_ly * local_Lx_ + old_lx];
      }
    }
  }
  
  // Update bounds and buffers
  bounds_ = new_bounds;
  local_Lx_ = new_Lx;
  local_Ly_ = new_Ly;
  phi_ = std::move(new_phi);
  phi_new_.resize(local_Lx_ * local_Ly_, 0.0f);
  interaction_sum_.resize(local_Lx_ * local_Ly_, 0.0f);
}

float Cell::compute_volume(float dx, float dy) const {
  float volume = 0.0f;
  float cell_area = dx * dy;

#pragma omp parallel for reduction(+ : volume)
  for (int i = 0; i < static_cast<int>(phi_.size()); ++i) {
    volume += phi_[i] * phi_[i] * cell_area;
  }

  return volume;
}

void Cell::compute_center_of_mass(float dx, float dy, float &com_x,
                                  float &com_y) const {
  // Use reference point method for periodic boundaries
  // This computes the center of mass relative to a reference point,
  // then converts back to absolute coordinates
  float ref_x = cx_;
  float ref_y = cy_;
  
  float sum_phi2 = 0.0f;
  float sum_dx = 0.0f;  // Sum of displacements from reference
  float sum_dy = 0.0f;

#pragma omp parallel for reduction(+ : sum_phi2, sum_dx, sum_dy)
  for (int ly = 0; ly < local_Ly_; ++ly) {
    for (int lx = 0; lx < local_Lx_; ++lx) {
      // Get global coordinates with periodic wrapping
      int gx, gy;
      bounds_.local_to_global(lx, ly, gx, gy, global_Lx_, global_Ly_);
      
      float x = gx * dx;
      float y = gy * dy;
      
      // Compute displacement from reference with periodic BC
      float disp_x = x - ref_x;
      float disp_y = y - ref_y;
      
      // Wrap to nearest image
      float Lx = global_Lx_ * dx;
      float Ly = global_Ly_ * dy;
      if (disp_x > Lx * 0.5f) disp_x -= Lx;
      if (disp_x < -Lx * 0.5f) disp_x += Lx;
      if (disp_y > Ly * 0.5f) disp_y -= Ly;
      if (disp_y < -Ly * 0.5f) disp_y += Ly;

      int idx = ly * local_Lx_ + lx;
      float phi2 = phi_[idx] * phi_[idx];

      sum_phi2 += phi2;
      sum_dx += disp_x * phi2;
      sum_dy += disp_y * phi2;
    }
  }

  if (sum_phi2 > 1e-10f) {
    // Compute center of mass as reference + weighted displacement
    com_x = ref_x + sum_dx / sum_phi2;
    com_y = ref_y + sum_dy / sum_phi2;
    
    // Wrap back to [0, L)
    float Lx = global_Lx_ * dx;
    float Ly = global_Ly_ * dy;
    while (com_x < 0) com_x += Lx;
    while (com_x >= Lx) com_x -= Lx;
    while (com_y < 0) com_y += Ly;
    while (com_y >= Ly) com_y -= Ly;
  } else {
    com_x = cx_;
    com_y = cy_;
  }
}

void Cell::copy_to_global(float *global_phi, int global_Lx,
                          int global_Ly) const {
  for (int ly = 0; ly < local_Ly_; ++ly) {
    for (int lx = 0; lx < local_Lx_; ++lx) {
      // Get global coordinates with periodic wrapping
      int gx, gy;
      bounds_.local_to_global(lx, ly, gx, gy, global_Lx, global_Ly);

      // gx and gy are now guaranteed to be in [0, N) range
      int local_idx = ly * local_Lx_ + lx;
      int global_idx = gy * global_Lx + gx;
      global_phi[global_idx] += phi_[local_idx];
    }
  }
}

void Cell::sample_interaction_field(const std::vector<Cell> &cells,
                                    int exclude_id) {
  // Clear interaction sum
  std::fill(interaction_sum_.begin(), interaction_sum_.end(), 0.0f);

  for (const auto &other : cells) {
    if (other.get_id() == exclude_id)
      continue;

    const auto &other_bounds = other.get_bounds();

    // Check for overlap using periodic distance
    if (!bounds_.overlaps(other_bounds, global_Lx_, global_Ly_)) {
      continue; // No overlap
    }

    const float *other_phi = other.get_phi();
    int other_Lx = other.get_local_Lx();
    int other_Ly = other.get_local_Ly();

    // For each point in my subdomain, check if it's in the other's subdomain
    for (int my_ly = 0; my_ly < local_Ly_; ++my_ly) {
      for (int my_lx = 0; my_lx < local_Lx_; ++my_lx) {
        // Get my global coordinate
        int gx, gy;
        bounds_.local_to_global(my_lx, my_ly, gx, gy, global_Lx_, global_Ly_);
        
        // Convert to other's local coordinate
        int other_lx, other_ly;
        other_bounds.global_to_local(gx, gy, other_lx, other_ly, global_Lx_, global_Ly_);
        
        // Check if this point is within other's subdomain
        if (other_lx >= 0 && other_lx < other_Lx && 
            other_ly >= 0 && other_ly < other_Ly) {
          int my_idx = my_ly * local_Lx_ + my_lx;
          int other_idx = other_ly * other_Lx + other_lx;
          
          float other_val = other_phi[other_idx];
          interaction_sum_[my_idx] += other_val * other_val;
        }
      }
    }
  }
}

} // namespace cellsim
