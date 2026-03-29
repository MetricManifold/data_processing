#pragma once

#include "cell.hpp"
#include "types.hpp"
#include <functional>
#include <random>
#include <vector>

namespace cellsim {

/**
 * Domain class managing all cells and global operations.
 *
 * Handles initialization, cell placement, and global field assembly.
 */
class Domain {
public:
  Domain() = default;

  /**
   * Initialize the domain with given dimensions and parameters.
   */
  void initialize(const SimParams &params);

  /**
   * Place cells randomly with non-overlapping positions.
   */
  void place_cells_random(int num_cells, float target_confluence,
                          unsigned int seed);

  /**
   * Place cells on a grid pattern.
   */
  void place_cells_grid(int num_cells);

  /**
   * Load cell positions from checkpoint data.
   */
  void load_from_checkpoint(const std::vector<float> &centers_x,
                            const std::vector<float> &centers_y,
                            const std::vector<float> &velocities_x,
                            const std::vector<float> &velocities_y);

  /**
   * Compute interaction fields for all cells.
   */
  void compute_interactions();

  /**
   * Assemble global phase fields for output.
   * Returns the sum of all φ fields and the sum of φ² fields.
   */
  void assemble_global_fields(std::vector<float> &global_phi_sum,
                              std::vector<float> &global_phi2_sum) const;

  /**
   * Get total number of cells.
   */
  int get_num_cells() const { return static_cast<int>(cells_.size()); }

  /**
   * Get reference to cells.
   */
  std::vector<Cell> &get_cells() { return cells_; }
  const std::vector<Cell> &get_cells() const { return cells_; }

  /**
   * Get simulation parameters.
   */
  const SimParams &get_params() const { return params_; }

  /**
   * Get domain dimensions.
   */
  int get_Nx() const { return params_.Nx; }
  int get_Ny() const { return params_.Ny; }

private:
  SimParams params_;
  std::vector<Cell> cells_;
  std::mt19937 rng_;
};

} // namespace cellsim
