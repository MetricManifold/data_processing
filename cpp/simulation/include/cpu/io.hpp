#pragma once

#include "domain.hpp"
#include <string>
#include <vector>

namespace cellsim {

// Checkpoint file format version
constexpr int CHECKPOINT_VERSION = 4;

/**
 * Save simulation state to VTK file.
 * Saves global phi_sum, phi2_sum, and cell-labeled fields.
 */
void save_vtk(const Domain &domain, int step, const std::string &output_dir);

/**
 * Save checkpoint for resuming simulation.
 * Format v4 includes:
 * - Header: version, num_cells, current_step, Lx, Ly, params
 * - Per-cell: center, velocity, target_volume, bounds, phi data
 */
void save_checkpoint(const Domain &domain, int step,
                     const std::string &output_dir);

/**
 * Load checkpoint and restore simulation state.
 * Returns true on success, false on failure.
 */
bool load_checkpoint(Domain &domain, int &step, const std::string &filepath);

/**
 * Load initial cell positions from JSON file.
 * Returns true on success, false on failure.
 */
bool load_initial_conditions_json(Domain &domain, const std::string &filepath);

/**
 * Load specific field from VTK file.
 */
bool load_vtk_field(const std::string &filepath, const std::string &field_name,
                    std::vector<float> &data, int &Lx, int &Ly);

/**
 * Save observables time series (MSD, volume, etc.)
 */
void save_observables(const std::string &output_dir,
                      const std::vector<float> &times,
                      const std::vector<float> &msd,
                      const std::vector<float> &mean_volume,
                      const std::vector<float> &volume_std);

/**
 * Checkpoint header structure - matches CUDA version exactly
 */
struct CheckpointHeader {
  int version;
  int num_cells;
  int current_step;
  int Nx;
  int Ny;
  float dx;
  float dy;
  float dt;
  float lambda;
  float gamma;
  float kappa;
  float mu;
  float target_radius;
  float v_A;       // motility speed
  float tau;       // tumble time
};

} // namespace cellsim
