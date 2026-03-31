#pragma once

#include "domain.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace cellsim {

//=============================================================================
// Checkpoint Header - matches CUDA version exactly
//=============================================================================

struct CheckpointHeader {
  uint32_t magic;      // 0x43454C4C = "CELL"
  uint32_t version;    // Currently 4
  int current_step;    // Order must match CUDA version!
  float current_time;
  int num_cells;

  // Runtime options (v3+)
  int save_interval;
  int checkpoint_interval;
  int trajectory_samples;
  bool save_vtk;
  bool save_tracking;
  bool compute_diagnostics;
  bool save_individual_fields;

  // Size of SimParams for version compatibility
  uint32_t sim_params_size;
};

//=============================================================================
// I/O Functions
//=============================================================================

// Save checkpoint (compatible with CUDA version)
// If h_v_A is provided, per-cell v_A values are appended.
void save_checkpoint(const Domain &domain, const std::string &filename,
                     const CheckpointHeader &header,
                     const float *h_v_A = nullptr, int num_v_A = 0);

// Load checkpoint (compatible with CUDA version)
// If out_v_A is provided and checkpoint contains v_A data, values are loaded.
bool load_checkpoint(Domain &domain, const std::string &filename,
                     CheckpointHeader &out_header,
                     std::vector<float> *out_v_A = nullptr);

// Export VTK file
void export_vtk(const Domain &domain, const std::string &base_filename, int frame);

// Export trajectory data for MSD computation
void export_trajectory(const Domain &domain, const std::string &filename,
                       float current_time);

} // namespace cellsim
