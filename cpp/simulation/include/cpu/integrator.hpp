#pragma once

#include "domain.hpp"
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace cellsim {

/**
 * Time integration for the phase field model.
 *
 * Handles the main simulation loop, including:
 * - Computing RHS of the equation of motion
 * - Time stepping with forward Euler
 * - Volume measurement and constraint enforcement
 * - Motility updates (Run-and-Tumble)
 */
class Integrator {
public:
  Integrator() = default;

  /**
   * Initialize integrator with simulation parameters.
   */
  void initialize(Domain &domain, unsigned int seed);

  /**
   * Perform one time step for all cells.
   */
  void step(Domain &domain);

  /**
   * Run simulation for given number of steps.
   */
  void run(Domain &domain, int num_steps, int output_interval,
           const std::string &output_dir,
           std::function<void(int, float)> callback = nullptr);

  /**
   * Set current step (for continuing from checkpoint).
   */
  void set_current_step(int step) { current_step_ = step; }
  int get_current_step() const { return current_step_; }

  /**
   * Set random seed for motility.
   */
  void set_seed(unsigned int seed) { rng_.seed(seed); }

  /**
   * Get timing statistics.
   */
  double get_total_time() const { return total_time_; }

private:
  /**
   * Compute RHS for a single cell.
   */
  void compute_rhs(Cell &cell, const SimParams &params);

  /**
   * Update cell velocities based on motility model.
   */
  void update_motility(Domain &domain);

  /**
   * Update cell positions based on CoM.
   */
  void update_positions(Domain &domain);

  int current_step_ = 0;
  std::mt19937 rng_;
  double total_time_ = 0.0;
};

} // namespace cellsim
