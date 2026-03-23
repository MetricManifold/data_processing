#pragma once

#include "domain.cuh"
#include "types.cuh"
#include <fstream>
#include <string>
#include <thread>
#include <atomic>

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
// options). If h_v_A is provided, per-cell v_A values are appended for
// quenched disorder preservation across restarts.
void save_checkpoint(const Domain &domain, const std::string &filename,
                     const CheckpointHeader &header,
                     const float *h_v_A = nullptr, int num_v_A = 0,
                     const float *h_gamma = nullptr, int num_gamma = 0,
                     const float *h_target_radius = nullptr, int num_target_radius = 0);

// Load state from checkpoint, returns header with step, time, and runtime
// options. If out_v_A is provided and the checkpoint contains per-cell v_A
// data, the values are loaded into it.
// Returns true if loaded successfully, false otherwise
bool load_checkpoint(Domain &domain, const std::string &filename,
                     CheckpointHeader &out_header,
                     std::vector<float> *out_v_A = nullptr,
                     std::vector<float> *out_gamma = nullptr,
                     std::vector<float> *out_target_radius = nullptr);

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

//=============================================================================
// Stress Field Export (enabled via STRESS_FIELDS_ENABLED)
//=============================================================================

#ifdef STRESS_FIELDS_ENABLED
// Forward declare StressFieldBuffers to avoid circular include
struct StressFieldBuffers;

// Export VTK with stress tensor fields: σ_xx, σ_yy, σ_xy, P
// Call after compute_stress_fields() to include stress data in VTK output
void export_vtk_with_stress(const Domain &domain, 
                           const StressFieldBuffers &stress,
                           const std::string &base_filename, int frame);
#endif

//=============================================================================
// Async Binary VTK Writer
//
// Replaces the blocking ASCII export_vtk() path with:
//   1. GPU scatter kernel: assembles full field on-device (~2ms)
//   2. Single D→H copy to pinned host buffer (~6ms for 6400²)
//   3. Background thread: byte-swap + binary file write (non-blocking)
//
// Total GPU blocking: ~8ms per save (vs ~5s for old ASCII path)
// File sizes: identical (~164 MB for 6400²) but written as raw binary
//=============================================================================

class AsyncVTKWriter {
public:
  AsyncVTKWriter();
  ~AsyncVTKWriter();

  // Allocate GPU + pinned host buffers for domain of size Nx × Ny
  void initialize(int Nx, int Ny);

  // Queue an async binary VTK write:
  //   Blocks GPU ~8ms (scatter + D→H), then writes file on background thread.
  //   Call wait() before checkpoint saves or program exit.
  void submit(float **d_phi_ptrs,
              const int *d_widths, const int *d_heights,
              const int *d_offsets_x, const int *d_offsets_y,
              int num_cells, int max_w, int max_h,
              int Nx, int Ny, int halo,
              float dx, float dy,
              const std::string &base_filename, int frame);

  // Block until any pending background write completes
  void wait();

  bool is_ready() const { return is_initialized; }

private:
  float *d_full_field;      // GPU scatter target [Nx × Ny]
  float *h_full_field;      // Pinned host buffer [Nx × Ny]
  size_t field_count;       // Nx * Ny elements
  bool is_initialized;

  std::thread writer_thread;

  // Static helper: write binary VTK file (called on background thread)
  static void write_binary_vtk(const std::string &filename,
                                float *data, size_t count,
                                int Nx, int Ny,
                                float dx, float dy, int frame);
};

} // namespace cellsim
