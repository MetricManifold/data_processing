#pragma once

#include "domain.hpp"
#include <random>

namespace cellsim {

//=============================================================================
// Integrator - Time stepping and physics computation (CPU/MPI version)
//=============================================================================

class Integrator {
public:
  // Random number generator for polarization updates
  std::mt19937 rng;
  std::uniform_real_distribution<float> uniform_dist;
  std::normal_distribution<float> normal_dist;

  // Bounding box update interval
  int bbox_update_interval = 10;
  int step_counter = 0;

public:
  Integrator();
  ~Integrator() = default;

  // Set random seed (for reproducibility)
  void set_seed(unsigned int seed);

  // Main stepping function
  void step(Domain &domain, float dt);

private:
  // Compute volume and centroid for a cell
  void compute_volume_and_centroid(Cell &cell, const SimParams &params);

  // Compute all local terms (laplacian, bulk, gradient) for a cell
  void compute_local_terms(Cell &cell, const SimParams &params,
                           std::vector<float> &laplacian,
                           std::vector<float> &bulk,
                           std::vector<float> &grad_x,
                           std::vector<float> &grad_y);

  // Compute interaction sum at each point (Σ_j φ_j² for j≠i)
  void compute_interaction_sum(const Domain &domain, int cell_idx,
                               std::vector<float> &interaction_sum);

  // Compute velocity integrals and update cell velocity
  void compute_velocity_integrals(Cell &cell, const SimParams &params,
                                  const std::vector<float> &grad_x,
                                  const std::vector<float> &grad_y,
                                  const std::vector<float> &interaction_sum);

  // Compute final RHS and apply Euler step
  void compute_rhs_and_step(Cell &cell, const SimParams &params, float dt,
                            const std::vector<float> &laplacian,
                            const std::vector<float> &bulk,
                            const std::vector<float> &grad_x,
                            const std::vector<float> &grad_y,
                            const std::vector<float> &interaction_sum);

  // Update polarization direction
  void update_polarization(Cell &cell, const SimParams &params, float dt);
};

} // namespace cellsim
