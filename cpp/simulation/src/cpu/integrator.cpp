#include "integrator.hpp"
#include "io.hpp"
#include <chrono>
#include <cmath>
#include <iostream>

namespace cellsim {

void Integrator::initialize(Domain &domain, unsigned int seed) {
  current_step_ = 0;
  rng_.seed(seed);
  total_time_ = 0.0;
}

void Integrator::compute_rhs(Cell &cell, const SimParams &params) {
  const float *phi = cell.get_phi();
  float *phi_new = cell.get_phi_new();
  const float *interaction = cell.get_interaction_sum();

  int local_Lx = cell.get_local_Lx();
  int local_Ly = cell.get_local_Ly();
  float vx = cell.get_vx();
  float vy = cell.get_vy();

  // Precompute coefficients
  float inv_dx2 = 1.0f / (params.dx * params.dx);
  float inv_dy2 = 1.0f / (params.dy * params.dy);
  float inv_2dx = 1.0f / (2.0f * params.dx);
  float inv_2dy = 1.0f / (2.0f * params.dy);

  float bulk_coeff = 30.0f / (params.lambda * params.lambda);
  float interaction_coeff = 30.0f * params.kappa / (params.lambda * params.lambda);

  // Compute volume deviation (absolute, matching CUDA)
  float volume = cell.compute_volume(params.dx, params.dy);
  float target_volume = cell.get_target_volume();
  float volume_deviation = target_volume - volume;  // absolute difference
  float volume_coeff = params.mu / target_volume;

#pragma omp parallel for collapse(2)
  for (int ly = 0; ly < local_Ly; ++ly) {
    for (int lx = 0; lx < local_Lx; ++lx) {
      int idx = ly * local_Lx + lx;
      float phi_val = phi[idx];

      // Skip near-zero regions
      if (phi_val < 1e-6f && interaction[idx] < 1e-6f) {
        phi_new[idx] = phi_val;
        continue;
      }

      // Laplacian
      float laplacian =
          laplacian_5pt(phi, idx, local_Lx, local_Ly, lx, ly, inv_dx2, inv_dy2);

      // Gradient for advection
      float grad_x, grad_y;
      gradient_2d(phi, idx, local_Lx, local_Ly, lx, ly, inv_2dx, inv_2dy,
                  grad_x, grad_y);

      // Physics terms
      float bulk_term = compute_bulk_term(phi_val, bulk_coeff);
      float constraint_term =
          compute_volume_constraint_term(phi_val, volume_deviation, volume_coeff);
      float repulsion_term =
          compute_repulsion_term(phi_val, interaction[idx], interaction_coeff);
      float advection_term = compute_advection_term(grad_x, grad_y, vx, vy);

      // Combined RHS
      float rhs = combine_rhs_terms(laplacian, bulk_term, constraint_term,
                                    repulsion_term, advection_term, params.gamma);

      // Forward Euler step
      phi_new[idx] = phi_val + params.dt * rhs;

      // Clamp to [0, 1]
      phi_new[idx] = std::max(0.0f, std::min(1.0f, phi_new[idx]));
    }
  }
}

void Integrator::update_motility(Domain &domain) {
  const SimParams &params = domain.get_params();

  if (params.v_A < 1e-10f) {
    return; // No motility
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (auto &cell : domain.get_cells()) {
    // Check for tumble event
    if (check_tumble_event(params.dt, params.tau, dist(rng_))) {
      // New random orientation
      float theta = dist(rng_) * 2.0f * 3.14159265f;
      cell.set_theta(theta);
      cell.set_velocity(params.v_A * std::cos(theta),
                        params.v_A * std::sin(theta));
    }
  }
}

void Integrator::update_positions(Domain &domain) {
  const SimParams &params = domain.get_params();

  for (auto &cell : domain.get_cells()) {
    float com_x, com_y;
    cell.compute_center_of_mass(params.dx, params.dy, com_x, com_y);
    cell.update_subdomain(com_x, com_y);
  }
}

void Integrator::step(Domain &domain) {
  // 1. Compute cell-cell interactions
  domain.compute_interactions();

  // 2. Compute RHS and advance each cell
  for (auto &cell : domain.get_cells()) {
    compute_rhs(cell, domain.get_params());
  }

  // 3. Swap buffers
  for (auto &cell : domain.get_cells()) {
    cell.swap_buffers();
  }

  // 4. Update motility
  update_motility(domain);

  // 5. Update cell positions
  update_positions(domain);

  ++current_step_;
}

void Integrator::run(Domain &domain, int num_steps, int output_interval,
                     const std::string &output_dir,
                     std::function<void(int, float)> callback) {
  auto start_time = std::chrono::high_resolution_clock::now();

  for (int s = 0; s < num_steps; ++s) {
    step(domain);

    if (output_interval > 0 && current_step_ % output_interval == 0) {
      save_vtk(domain, current_step_, output_dir);
      save_checkpoint(domain, current_step_, output_dir);

      auto now = std::chrono::high_resolution_clock::now();
      double elapsed =
          std::chrono::duration<double>(now - start_time).count();
      float steps_per_sec = (s + 1) / elapsed;

      std::cout << "Step " << current_step_ << " / " << (current_step_ + num_steps - s - 1)
                << " (" << steps_per_sec << " steps/s)" << std::endl;

      if (callback) {
        callback(current_step_, steps_per_sec);
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  total_time_ = std::chrono::duration<double>(end_time - start_time).count();
}

} // namespace cellsim
