#pragma once

#include "domain.cuh"
#include "types.cuh"
#include <fstream>
#include <string>

namespace cellsim {

//=============================================================================
// Checkpoint header for version control and simulation state
//=============================================================================
struct CheckpointHeader {
  uint32_t magic = 0x43454C4C; // "CELL" magic number
  uint32_t version = 4;        // Version 4 includes SimParams size
  int current_step = 0;
  float current_time = 0.0f;
  int num_cells = 0;

  // Runtime options (added in v3)
  int save_interval = 100; // Steps between VTK saves
  int checkpoint_interval =
      -1; // Steps between checkpoints (-1 = save_interval*10)
  int trajectory_samples = 100; // Number of trajectory samples
  bool save_vtk = true;
  bool save_tracking = true;
  bool compute_diagnostics = false;
  bool save_individual_fields =
      false; // Save individual cell fields for energy analysis
  // int32_t _padding = 0; // Padding removed - struct layout changed

  // SimParams size tracking (added in v4)
  uint32_t sim_params_size = sizeof(SimParams);
};

//=============================================================================
// File I/O
//=============================================================================

// Save current state to checkpoint (includes simulation time/step and runtime
// options)
void save_checkpoint(const Domain &domain, const std::string &filename,
                     const CheckpointHeader &header);

// Load state from checkpoint, returns header with step, time, and runtime
// options Returns true if loaded successfully, false otherwise
bool load_checkpoint(Domain &domain, const std::string &filename,
                     CheckpointHeader &out_header);

// Export single frame for visualization (simple text format)
void export_frame_txt(const Domain &domain, const std::string &filename,
                      int frame);

// Export cell tracking data
void export_tracking_data(const Domain &domain, const std::string &filename,
                          float time);

//=============================================================================
// VTK Export for ParaView visualization
//=============================================================================

// Export all cells as VTK structured grid (combined field using max)
void export_vtk(const Domain &domain, const std::string &filename, int frame);

// Export individual cell fields as separate VTK files
// Creates files: base_NNNNNN_cell_MM.vtk for each cell
// Also creates base_NNNNNN_sum.vtk with the actual sum (not max) for energy
// analysis
void export_vtk_individual(const Domain &domain, const std::string &filename,
                           int frame);

// Export energy metrics computed during simulation
// This is more accurate than post-processing since we have access to individual
// fields
void export_energy_metrics(const Domain &domain, const std::string &filename,
                           int frame, float time);

} // namespace cellsim
