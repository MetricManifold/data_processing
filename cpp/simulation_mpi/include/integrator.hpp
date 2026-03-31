#pragma once

#include "domain.hpp"
#include <random>

// Forward declaration for MPI spatial
namespace cellsim { class MPISpatial; }

namespace cellsim {

//=============================================================================
// Integrator - Time stepping and physics computation
//
// Supports two modes:
// 1. OpenMP-only: All cells processed locally with thread parallelism
// 2. MPI+OpenMP: Spatial domain decomposition with hybrid parallelism
//
// Per-timestep workflow:
// 1. Compute local terms (laplacian, bulk, gradients) for cells (parallel)
// 2. Compute volume and centroid for cells (parallel)
// 3. Compute interaction sums for cells (parallel)
// 4. Compute velocities for cells (parallel)
// 5. Update polarization for cells (sequential - RNG)
// 6. Apply time step for cells (parallel)
// 7. Periodically update bounding boxes
// 8. (MPI only) Synchronize cell data across ranks
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

  // Main stepping function (OpenMP parallel, no MPI)
  void step(Domain &domain, float dt);
  
#ifdef USE_MPI
  // MPI stepping function (spatial decomposition with OpenMP)
  // Each rank updates cells whose centers are in its tile,
  // then synchronizes phi fields across all ranks
  void step_mpi(Domain &domain, float dt, MPISpatial &mpi);
#endif

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
