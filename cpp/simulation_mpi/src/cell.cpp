#include "cell.hpp"
#include "physics.hpp"
#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace cellsim {

Cell::Cell(int id_, const BoundingBox &initial_bbox, int halo_width)
    : id(id_), state(CellState::Active), bbox(initial_bbox),
      bbox_with_halo(initial_bbox.expanded(halo_width)),
      field_size(0), volume(0), centroid{0, 0}, velocity{0, 0},
      polarization{1, 0}, theta(0), volume_deviation(0) {
  // Initialize with random polarization direction
  theta = static_cast<float>(rand()) / RAND_MAX * 2.0f * static_cast<float>(M_PI);
  polarization.x = cosf(theta);
  polarization.y = sinf(theta);
  allocate_memory();
}

void Cell::allocate_memory() {
  field_size = bbox_with_halo.size();
  if (field_size > 0) {
    phi.resize(field_size, 0.0f);
    dphi_dt.resize(field_size, 0.0f);
  }
}

void Cell::initialize_circular(float cx, float cy, float radius,
                               const SimParams &params) {
  float lambda = params.lambda;

  // Effective radius correction for tanh profile
  float interface_width = sqrtf(2.0f) * lambda;
  float w2_over_3 = (interface_width * interface_width) / 3.0f;
  float eff_radius = radius;
  if (radius * radius > w2_over_3) {
    eff_radius = sqrtf(radius * radius - w2_over_3);
  }

  for (int ly = 0; ly < height(); ++ly) {
    for (int lx = 0; lx < width(); ++lx) {
      // Get global coordinates
      int gx, gy;
      bbox_with_halo.local_to_global(lx, ly, gx, gy, params.Nx, params.Ny);

      // Distance from cell center (with periodic BC)
      float dx = static_cast<float>(gx) - cx;
      float dy = static_cast<float>(gy) - cy;

      // Periodic distance
      if (dx > params.Nx * 0.5f) dx -= params.Nx;
      if (dx < -params.Nx * 0.5f) dx += params.Nx;
      if (dy > params.Ny * 0.5f) dy -= params.Ny;
      if (dy < -params.Ny * 0.5f) dy += params.Ny;

      float r = sqrtf(dx * dx + dy * dy);

      // Tanh profile
      float phi_val = 0.5f * (1.0f - tanhf((r - eff_radius) / interface_width));
      phi[local_index(lx, ly)] = phi_val;
    }
  }

  // Set initial centroid and volume
  centroid = {cx, cy};
  volume = static_cast<float>(M_PI) * radius * radius; // Approximate, will be recomputed
}

bool Cell::update_bounding_box(const SimParams &params, float threshold) {
  int halo = params.halo_width;
  int old_w = width();
  int old_h = height();
  int Nx = params.Nx;
  int Ny = params.Ny;

  // Find maximum extent from centroid using periodic distance
  float max_dist_x = 0.0f;
  float max_dist_y = 0.0f;
  int min_lx = old_w, max_lx = -1;
  int min_ly = old_h, max_ly = -1;
  bool found_any = false;

  for (int ly = halo; ly < old_h - halo; ++ly) {
    for (int lx = halo; lx < old_w - halo; ++lx) {
      if (phi[local_index(lx, ly)] > threshold) {
        found_any = true;

        // Track local bounds
        min_lx = std::min(min_lx, lx);
        max_lx = std::max(max_lx, lx);
        min_ly = std::min(min_ly, ly);
        max_ly = std::max(max_ly, ly);

        // Get global coordinates
        int gx, gy;
        bbox_with_halo.local_to_global(lx, ly, gx, gy, Nx, Ny);

        // Periodic distance from centroid
        float dx = static_cast<float>(gx) - centroid.x;
        float dy = static_cast<float>(gy) - centroid.y;

        if (dx > Nx * 0.5f) dx -= Nx;
        if (dx < -Nx * 0.5f) dx += Nx;
        if (dy > Ny * 0.5f) dy -= Ny;
        if (dy < -Ny * 0.5f) dy += Ny;

        max_dist_x = std::max(max_dist_x, fabsf(dx));
        max_dist_y = std::max(max_dist_y, fabsf(dy));
      }
    }
  }

  if (!found_any) {
    return false;
  }

  // Padding
  int padding = static_cast<int>(params.target_radius * 
                                 (params.subdomain_padding - 1.0f)) + halo;

  int new_cx = static_cast<int>(centroid.x);
  int new_cy = static_cast<int>(centroid.y);

  int half_w = static_cast<int>(max_dist_x) + padding;
  int half_h = static_cast<int>(max_dist_y) + padding;

  // Minimum size
  half_w = std::max(half_w, static_cast<int>(params.target_radius * 
                                             params.subdomain_padding));
  half_h = std::max(half_h, static_cast<int>(params.target_radius * 
                                             params.subdomain_padding));

  // New bounding box
  BoundingBox new_bbox = {new_cx - half_w, new_cy - half_h, 
                          new_cx + half_w, new_cy + half_h};
  BoundingBox new_bbox_with_halo = {new_bbox.x0 - halo, new_bbox.y0 - halo,
                                    new_bbox.x1 + halo, new_bbox.y1 + halo};

  // Check shift
  int old_cx = (bbox_with_halo.x0 + bbox_with_halo.x1) / 2;
  int old_cy = (bbox_with_halo.y0 + bbox_with_halo.y1) / 2;

  int shift_x = new_cx - old_cx;
  int shift_y = new_cy - old_cy;

  if (shift_x > Nx / 2) shift_x -= Nx;
  if (shift_x < -Nx / 2) shift_x += Nx;
  if (shift_y > Ny / 2) shift_y -= Ny;
  if (shift_y < -Ny / 2) shift_y += Ny;

  // Check edge touching
  bool touching_edge = (min_lx <= halo + 1) || (max_lx >= old_w - halo - 2) ||
                       (min_ly <= halo + 1) || (max_ly >= old_h - halo - 2);

  if (abs(shift_x) < 5 && abs(shift_y) < 5 && !touching_edge) {
    return false;
  }

  // Copy old field to new position
  int new_w = new_bbox_with_halo.width();
  int new_h = new_bbox_with_halo.height();
  int new_size = new_w * new_h;

  std::vector<float> phi_new(new_size, 0.0f);

  for (int old_ly = 0; old_ly < old_h; ++old_ly) {
    for (int old_lx = 0; old_lx < old_w; ++old_lx) {
      int gx, gy;
      bbox_with_halo.local_to_global(old_lx, old_ly, gx, gy, Nx, Ny);

      int new_lx, new_ly;
      new_bbox_with_halo.global_to_local(gx, gy, new_lx, new_ly, Nx, Ny);

      if (new_lx >= 0 && new_lx < new_w && new_ly >= 0 && new_ly < new_h) {
        phi_new[new_ly * new_w + new_lx] = phi[old_ly * old_w + old_lx];
      }
    }
  }

  // Update
  phi = std::move(phi_new);
  dphi_dt.assign(new_size, 0.0f);
  field_size = new_size;
  bbox = new_bbox;
  bbox_with_halo = new_bbox_with_halo;

  return true;
}

} // namespace cellsim
