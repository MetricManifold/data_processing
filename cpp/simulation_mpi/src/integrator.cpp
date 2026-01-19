#include "integrator.hpp"
#include "physics.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>

namespace cellsim {

Integrator::Integrator()
    : rng(static_cast<unsigned>(time(nullptr))),
      uniform_dist(0.0f, 1.0f),
      normal_dist(0.0f, 1.0f),
      bbox_update_interval(10),
      step_counter(0) {}

void Integrator::set_seed(unsigned int seed) {
  rng.seed(seed);
}

void Integrator::step(Domain &domain, float dt) {
  if (domain.num_cells() == 0) return;

  const SimParams &params = domain.params;
  int num_cells = domain.num_cells();
  int halo = params.halo_width;

  // Temporary buffers for each cell
  std::vector<std::vector<float>> all_laplacian(num_cells);
  std::vector<std::vector<float>> all_bulk(num_cells);
  std::vector<std::vector<float>> all_grad_x(num_cells);
  std::vector<std::vector<float>> all_grad_y(num_cells);
  std::vector<std::vector<float>> all_interaction_sum(num_cells);

  // Step 1: Compute local terms for all cells
  #ifdef USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for (int i = 0; i < num_cells; ++i) {
    auto &cell = domain.cells[i];
    all_laplacian[i].resize(cell->field_size);
    all_bulk[i].resize(cell->field_size);
    all_grad_x[i].resize(cell->field_size);
    all_grad_y[i].resize(cell->field_size);
    all_interaction_sum[i].resize(cell->field_size, 0.0f);

    // Compute laplacian, bulk, and gradients
    compute_local_terms(*cell, params, all_laplacian[i], all_bulk[i],
                        all_grad_x[i], all_grad_y[i]);
  }

  // Step 2: Compute volume and centroid for all cells
  #ifdef USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for (int i = 0; i < num_cells; ++i) {
    compute_volume_and_centroid(*domain.cells[i], params);
  }

  // Step 3: Compute volume deviations
  float target_area = params.target_area();
  for (int i = 0; i < num_cells; ++i) {
    domain.cells[i]->volume_deviation = target_area - domain.cells[i]->volume;
  }

  // Step 4: Compute interaction sums (cell-cell interactions)
  if (num_cells > 1) {
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < num_cells; ++i) {
      compute_interaction_sum(domain, i, all_interaction_sum[i]);
    }
  }

  // Step 5: Compute velocities
  #ifdef USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for (int i = 0; i < num_cells; ++i) {
    compute_velocity_integrals(*domain.cells[i], params,
                               all_grad_x[i], all_grad_y[i],
                               all_interaction_sum[i]);
  }

  // Step 6: Update polarization directions (sequential due to RNG)
  for (int i = 0; i < num_cells; ++i) {
    update_polarization(*domain.cells[i], params, dt);
  }

  // Step 7: Compute RHS and apply Euler step
  #ifdef USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for (int i = 0; i < num_cells; ++i) {
    compute_rhs_and_step(*domain.cells[i], params, dt,
                         all_laplacian[i], all_bulk[i],
                         all_grad_x[i], all_grad_y[i],
                         all_interaction_sum[i]);
  }

  // Step 8: Update bounding boxes periodically
  step_counter++;
  if (step_counter % bbox_update_interval == 0) {
    domain.update_all_bounding_boxes();
  }
}

void Integrator::compute_volume_and_centroid(Cell &cell, const SimParams &params) {
  int width = cell.width();
  int height = cell.height();
  int halo = params.halo_width;
  float dA = params.dx * params.dy;

  float volume = 0.0f;
  float sum_phi2_x = 0.0f;
  float sum_phi2_y = 0.0f;

  // Use reference point (bbox center) for proper centroid calculation
  float ref_x = static_cast<float>(cell.bbox_with_halo.x0 + width / 2);
  float ref_y = static_cast<float>(cell.bbox_with_halo.y0 + height / 2);
  // Wrap reference to [0, N)
  ref_x = fmodf(fmodf(ref_x, (float)params.Nx) + params.Nx, (float)params.Nx);
  ref_y = fmodf(fmodf(ref_y, (float)params.Ny) + params.Ny, (float)params.Ny);

  for (int ly = halo; ly < height - halo; ++ly) {
    for (int lx = halo; lx < width - halo; ++lx) {
      int idx = ly * width + lx;
      float phi = cell.phi[idx];
      float phi_sq = phi * phi;

      volume += phi_sq;

      // Get global coordinate
      int gx, gy;
      cell.bbox_with_halo.local_to_global(lx, ly, gx, gy, params.Nx, params.Ny);

      // Compute displacement from reference (with periodic wrapping)
      float dx = static_cast<float>(gx) - ref_x;
      float dy = static_cast<float>(gy) - ref_y;
      
      if (dx > params.Nx * 0.5f) dx -= params.Nx;
      if (dx < -params.Nx * 0.5f) dx += params.Nx;
      if (dy > params.Ny * 0.5f) dy -= params.Ny;
      if (dy < -params.Ny * 0.5f) dy += params.Ny;

      sum_phi2_x += phi_sq * dx;
      sum_phi2_y += phi_sq * dy;
    }
  }

  // Scale by grid spacing
  cell.volume = volume * dA;

  // Compute centroid = ref + (weighted sum / total weight)
  if (volume > 1e-8f) {
    float cx = ref_x + sum_phi2_x / volume;
    float cy = ref_y + sum_phi2_y / volume;
    
    // Wrap to [0, N)
    cx = fmodf(fmodf(cx, (float)params.Nx) + params.Nx, (float)params.Nx);
    cy = fmodf(fmodf(cy, (float)params.Ny) + params.Ny, (float)params.Ny);
    
    cell.centroid.x = cx;
    cell.centroid.y = cy;
  }
}

void Integrator::compute_local_terms(Cell &cell, const SimParams &params,
                                     std::vector<float> &laplacian,
                                     std::vector<float> &bulk,
                                     std::vector<float> &grad_x,
                                     std::vector<float> &grad_y) {
  int width = cell.width();
  int height = cell.height();
  float inv_dx2 = 1.0f / (params.dx * params.dx);
  float inv_dy2 = 1.0f / (params.dy * params.dy);
  float inv_2dx = 1.0f / (2.0f * params.dx);
  float inv_2dy = 1.0f / (2.0f * params.dy);
  float bulk_coeff = params.bulk_coeff();

  for (int ly = 0; ly < height; ++ly) {
    for (int lx = 0; lx < width; ++lx) {
      int idx = ly * width + lx;
      float phi = cell.phi[idx];

      // Laplacian
      laplacian[idx] = laplacian_5pt(cell.phi.data(), idx, width, height,
                                      lx, ly, inv_dx2, inv_dy2);

      // Bulk term
      bulk[idx] = compute_bulk_term(phi, bulk_coeff);

      // Gradient
      gradient_2d(cell.phi.data(), idx, width, height, lx, ly,
                  inv_2dx, inv_2dy, grad_x[idx], grad_y[idx]);
    }
  }
}

void Integrator::compute_interaction_sum(const Domain &domain, int cell_idx,
                                         std::vector<float> &interaction_sum) {
  const Cell &cell_i = *domain.cells[cell_idx];
  const SimParams &params = domain.params;
  int width = cell_i.width();
  int height = cell_i.height();
  int Nx = params.Nx;
  int Ny = params.Ny;

  // Zero the interaction sum
  std::fill(interaction_sum.begin(), interaction_sum.end(), 0.0f);

  // Loop over all other cells
  for (int j = 0; j < domain.num_cells(); ++j) {
    if (j == cell_idx) continue;

    const Cell &cell_j = *domain.cells[j];

    // Check if bounding boxes overlap
    if (!cell_i.bbox_with_halo.overlaps(cell_j.bbox_with_halo, Nx, Ny)) {
      continue;
    }

    // For each point in cell_i's subdomain
    for (int ly = 0; ly < height; ++ly) {
      for (int lx = 0; lx < width; ++lx) {
        // Get global coordinate
        int gx, gy;
        cell_i.bbox_with_halo.local_to_global(lx, ly, gx, gy, Nx, Ny);

        // Check if this point is in cell_j's bounding box
        if (!cell_j.bbox_with_halo.contains(gx, gy, Nx, Ny)) {
          continue;
        }

        // Get local coordinates in cell_j
        int ljx, ljy;
        cell_j.bbox_with_halo.global_to_local(gx, gy, ljx, ljy, Nx, Ny);

        if (ljx >= 0 && ljx < cell_j.width() && 
            ljy >= 0 && ljy < cell_j.height()) {
          int jdx = ljy * cell_j.width() + ljx;
          float phi_j = cell_j.phi[jdx];
          
          int idx = ly * width + lx;
          interaction_sum[idx] += phi_j * phi_j;  // Σ φ_j²
        }
      }
    }
  }
}

void Integrator::compute_velocity_integrals(Cell &cell, const SimParams &params,
                                            const std::vector<float> &grad_x,
                                            const std::vector<float> &grad_y,
                                            const std::vector<float> &interaction_sum) {
  int width = cell.width();
  int height = cell.height();
  int halo = params.halo_width;
  float dA = params.dx * params.dy;
  float motility_coeff = params.motility_coeff();

  // Compute velocity integrals: ∫ φ (∇φ) Σ_m φ_m² dA
  float integral_x = 0.0f;
  float integral_y = 0.0f;

  for (int ly = halo; ly < height - halo; ++ly) {
    for (int lx = halo; lx < width - halo; ++lx) {
      int idx = ly * width + lx;
      float phi = cell.phi[idx];
      float isum = interaction_sum[idx];

      integral_x += phi * grad_x[idx] * isum;
      integral_y += phi * grad_y[idx] * isum;
    }
  }

  // Interaction velocity: v_n,I = (60κ/ξλ²) ∫ φ_n (∇φ_n) Σ_m φ_m² dV
  float vx_interaction = motility_coeff * integral_x * dA;
  float vy_interaction = motility_coeff * integral_y * dA;

  // Active velocity: v_n,A = v_A × p_n
  float vx_active = params.v_A * cell.polarization.x;
  float vy_active = params.v_A * cell.polarization.y;

  // Total velocity
  cell.velocity.x = vx_interaction + vx_active;
  cell.velocity.y = vy_interaction + vy_active;
}

void Integrator::compute_rhs_and_step(Cell &cell, const SimParams &params, float dt,
                                      const std::vector<float> &laplacian,
                                      const std::vector<float> &bulk,
                                      const std::vector<float> &grad_x,
                                      const std::vector<float> &grad_y,
                                      const std::vector<float> &interaction_sum) {
  int width = cell.width();
  int height = cell.height();
  float gamma = params.gamma;
  float volume_coeff = params.volume_coeff();
  float interaction_coeff = params.interaction_coeff();
  float volume_deviation = cell.volume_deviation;
  float vx = cell.velocity.x;
  float vy = cell.velocity.y;

  for (int ly = 0; ly < height; ++ly) {
    for (int lx = 0; lx < width; ++lx) {
      int idx = ly * width + lx;
      float phi = cell.phi[idx];

      // Volume constraint term
      float constraint_term = compute_volume_constraint_term(phi, volume_deviation, 
                                                              volume_coeff);

      // Repulsion term
      float repulsion_term = compute_repulsion_term(phi, interaction_sum[idx], 
                                                     interaction_coeff);

      // Advection term
      float advection_term = compute_advection_term(grad_x[idx], grad_y[idx], vx, vy);

      // Combine all terms
      float dphi_dt = combine_rhs_terms(laplacian[idx], bulk[idx], 
                                         constraint_term, repulsion_term,
                                         advection_term, gamma);

      // Euler step
      float new_phi = phi + dt * dphi_dt;

      // Clamp to [0, 1]
      cell.phi[idx] = std::max(0.0f, std::min(1.0f, new_phi));
    }
  }
}

void Integrator::update_polarization(Cell &cell, const SimParams &params, float dt) {
  if (params.motility_model == SimParams::MotilityModel::RunAndTumble) {
    // Run-and-Tumble: Poisson reorientation
    float p_tumble = 1.0f - expf(-dt / params.tau);
    float rand_val = uniform_dist(rng);
    if (rand_val < p_tumble) {
      // New random direction
      cell.theta = uniform_dist(rng) * 2.0f * static_cast<float>(M_PI);
    }
  } else {
    // ABP: Continuous rotational diffusion
    float noise_strength = sqrtf(2.0f * dt / params.tau);
    float dtheta = noise_strength * normal_dist(rng);
    cell.theta += dtheta;

    // Keep in [0, 2π)
    cell.theta = fmodf(cell.theta, 2.0f * static_cast<float>(M_PI));
    if (cell.theta < 0) cell.theta += 2.0f * static_cast<float>(M_PI);
  }

  cell.polarization.x = cosf(cell.theta);
  cell.polarization.y = sinf(cell.theta);
}

} // namespace cellsim
