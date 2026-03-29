#include "domain.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cellsim {

void Domain::initialize(const SimParams &params) { params_ = params; }

void Domain::place_cells_random(int num_cells, float target_confluence,
                                unsigned int seed) {
  rng_.seed(seed);

  cells_.clear();
  cells_.reserve(num_cells);

  float R = params_.target_radius;
  float domain_Nx = params_.Nx * params_.dx;
  float domain_Ny = params_.Ny * params_.dy;

  // Calculate minimum separation based on confluence and cell count
  // This matches the CUDA backend logic
  float domain_area = domain_Nx * domain_Ny;
  float area_per_cell = domain_area / num_cells;
  float ideal_spacing = std::sqrt(area_per_cell);
  
  // min_sep = gap between cell surfaces = center_spacing - 2*radius
  // Can be negative (overlapping) for high confluence
  float min_sep = ideal_spacing - 2.0f * R;
  
  // Clamp to at least 1.0 to avoid exact overlap which can cause numerical issues
  min_sep = std::max(1.0f, min_sep);

  std::vector<std::pair<float, float>> centers;
  centers.reserve(num_cells);

  std::uniform_real_distribution<float> dist_x(R + 1.0f, domain_Nx - R - 1.0f);
  std::uniform_real_distribution<float> dist_y(R + 1.0f, domain_Ny - R - 1.0f);
  std::uniform_real_distribution<float> dist_01(0.0f, 1.0f);

  int max_attempts = 100000;
  int attempts = 0;

  while (static_cast<int>(centers.size()) < num_cells && attempts < max_attempts) {
    float cx = dist_x(rng_);
    float cy = dist_y(rng_);

    // Check separation from existing cells
    bool valid = true;
    for (const auto &c : centers) {
      float dx_sep = cx - c.first;
      float dy_sep = cy - c.second;
      float dist = std::sqrt(dx_sep * dx_sep + dy_sep * dy_sep);
      if (dist < min_sep) {
        valid = false;
        break;
      }
    }

    if (valid) {
      centers.emplace_back(cx, cy);
    }

    ++attempts;
  }

  if (static_cast<int>(centers.size()) < num_cells) {
    throw std::runtime_error("Could not place all cells without overlap");
  }

  // Initialize cells at the computed positions
  for (int i = 0; i < num_cells; ++i) {
    Cell cell;
    cell.initialize(i, centers[i].first, centers[i].second, R, params_.lambda,
                    params_.dx, params_.dy, params_.Nx, params_.Ny);

    // Random initial orientation
    float theta = dist_01(rng_) * 2.0f * 3.14159265f;
    cell.set_theta(theta);
    cell.set_velocity(params_.v_A * std::cos(theta),
                      params_.v_A * std::sin(theta));

    cells_.push_back(std::move(cell));
  }
}

void Domain::place_cells_grid(int num_cells) {
  cells_.clear();
  cells_.reserve(num_cells);

  float R = params_.target_radius;
  float domain_Nx = params_.Nx * params_.dx;
  float domain_Ny = params_.Ny * params_.dy;

  // Compute grid dimensions
  int nx = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(num_cells))));
  int ny = (num_cells + nx - 1) / nx;

  float spacing_x = domain_Nx / (nx + 1);
  float spacing_y = domain_Ny / (ny + 1);

  int cell_id = 0;
  for (int j = 0; j < ny && cell_id < num_cells; ++j) {
    for (int i = 0; i < nx && cell_id < num_cells; ++i) {
      float cx = (i + 1) * spacing_x;
      float cy = (j + 1) * spacing_y;

      Cell cell;
      cell.initialize(cell_id, cx, cy, R, params_.lambda, params_.dx,
                      params_.dy, params_.Nx, params_.Ny);
      cell.set_velocity(0.0f, 0.0f);
      cell.set_theta(0.0f);

      cells_.push_back(std::move(cell));
      ++cell_id;
    }
  }
}

void Domain::load_from_checkpoint(const std::vector<float> &centers_x,
                                  const std::vector<float> &centers_y,
                                  const std::vector<float> &velocities_x,
                                  const std::vector<float> &velocities_y) {
  int num_cells = static_cast<int>(centers_x.size());
  cells_.clear();
  cells_.reserve(num_cells);

  for (int i = 0; i < num_cells; ++i) {
    Cell cell;
    cell.initialize(i, centers_x[i], centers_y[i], params_.target_radius,
                    params_.lambda, params_.dx, params_.dy, params_.Nx,
                    params_.Ny);
    cell.set_velocity(velocities_x[i], velocities_y[i]);

    // Compute theta from velocity
    float vx = velocities_x[i];
    float vy = velocities_y[i];
    if (std::abs(vx) > 1e-10f || std::abs(vy) > 1e-10f) {
      cell.set_theta(std::atan2(vy, vx));
    }

    cells_.push_back(std::move(cell));
  }
}

void Domain::compute_interactions() {
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(cells_.size()); ++i) {
    cells_[i].sample_interaction_field(cells_, cells_[i].get_id());
  }
}

void Domain::assemble_global_fields(std::vector<float> &global_phi_sum,
                                    std::vector<float> &global_phi2_sum) const {
  int size = params_.Nx * params_.Ny;
  global_phi_sum.assign(size, 0.0f);
  global_phi2_sum.assign(size, 0.0f);

  for (const auto &cell : cells_) {
    const auto &bounds = cell.get_bounds();
    const float *phi = cell.get_phi();
    int local_Lx = cell.get_local_Lx();
    int local_Ly = cell.get_local_Ly();

    for (int ly = 0; ly < local_Ly; ++ly) {
      for (int lx = 0; lx < local_Lx; ++lx) {
        // Use periodic coordinate conversion
        int gx, gy;
        bounds.local_to_global(lx, ly, gx, gy, params_.Nx, params_.Ny);

        int local_idx = ly * local_Lx + lx;
        int global_idx = gy * params_.Nx + gx;

        float val = phi[local_idx];
        global_phi_sum[global_idx] += val;
        global_phi2_sum[global_idx] += val * val;
      }
    }
  }
}

} // namespace cellsim
