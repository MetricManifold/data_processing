#pragma once

#include "domain3d.cuh"
#include "integrator3d.cuh"
#include "types3d.cuh"
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>

// Defined in main.cu — set by SIGTERM handler for clean SLURM shutdown
extern volatile std::sig_atomic_t g_shutdown_requested;

namespace cellsim {

//=============================================================================
// Forward declarations for I/O functions
//=============================================================================

void save_checkpoint_3d(const char *filename, const Domain3D &domain, int step,
                        float time);
bool load_checkpoint_3d(const char *filename, Domain3D &domain, int &step,
                        float &time);
// Scan checkpoint to get memory requirements without allocating GPU memory
size_t scan_checkpoint_3d_memory(const char *filename, int &out_num_cells);
void save_vtk_3d(const char *filename, const Domain3D &domain);
void save_cell_vtk_3d(const char *filename, const Cell3D &cell,
                      const SimParams3D &params);

//=============================================================================
// Non-Blocking Async Checkpoint Writer for 3D
// D→H copy to pinned buffer → background thread writes binary file
// GPU continues computing while file write happens.
//=============================================================================
class AsyncCheckpointWriter3D {
public:
  AsyncCheckpointWriter3D();
  ~AsyncCheckpointWriter3D();

  // Queue an async checkpoint write. Blocks GPU for D→H copy (~few ms),
  // then writes file on background thread. Call wait() before exit.
  void submit(const Domain3D &domain, int step, float time,
              const std::string &filename);

  // Block until pending write completes
  void wait();

private:
  void ensure_buffer(size_t needed);
  char *h_buffer;           // Pinned host buffer
  size_t buffer_size;
  bool initialized;
  std::thread writer_thread;
};

//=============================================================================
// Simulation3D - Main simulation controller for 3D
//=============================================================================

class Simulation3D {
public:
  Domain3D domain;
  Integrator3D integrator;
  AsyncCheckpointWriter3D async_writer;  // Non-blocking checkpoint saves

  // Simulation state
  int current_step;
  double current_time;  // double to avoid float32 precision loss at large t

  // Output settings
  std::string output_dir;
  int save_interval;
  int print_interval;         // Steps between progress output (-1 = use save_interval)
  int checkpoint_interval;    // Steps between checkpoints (-1 = save_interval*10)
  int trajectory_samples;     // Number of trajectory data points to save
  int trajectory_interval;    // Steps between trajectory saves (-1 = auto, 0 = from samples)
  bool save_individual_fields_flag;
  bool resumed_from_checkpoint;

  // Timing
  std::chrono::steady_clock::time_point start_time;

public:
  Simulation3D(const SimParams3D &params);

  // Initialize cells
  void initialize_random(int num_cells, float radius, float min_spacing);
  void initialize_grid(int num_cells, float radius, float confluence);
  void initialize_grid_fcc(int num_cells, float radius, float confluence);

  // Load from checkpoint
  bool load_checkpoint(const char *filename);

  // Run simulation
  void run(float t_end);

  // Single step
  void step(bool sync_to_host = false);

  // Save current state
  void save_checkpoint();
  void save_vtk();
  void save_individual_cell_fields();
  void save_trajectory();

  // Print status
  void print_status();
};

//=============================================================================
// Simulation3D Implementation
//=============================================================================

inline Simulation3D::Simulation3D(const SimParams3D &params)
    : domain(params), current_step(0), current_time(0.0), save_interval(100),
      print_interval(-1), checkpoint_interval(-1), trajectory_samples(100),
      trajectory_interval(-1), save_individual_fields_flag(false),
      resumed_from_checkpoint(false) {}

inline void Simulation3D::initialize_random(int num_cells, float radius,
                                            float min_spacing) {
  domain.initialize_random_cells(num_cells, radius, min_spacing);
  printf("Initialized %d 3D cells\n", domain.num_cells());
  for (int i = 0; i < domain.num_cells(); ++i) {
    auto &cell = domain.cells[i];
    printf("  Cell %d: center=(%.1f, %.1f, %.1f), "
           "subdomain=[%d,%d,%d]->[%d,%d,%d] (%dx%dx%d)\n",
           cell->id, cell->centroid.x, cell->centroid.y, cell->centroid.z,
           cell->bbox.x0, cell->bbox.y0, cell->bbox.z0, cell->bbox.x1,
           cell->bbox.y1, cell->bbox.z1, cell->width(), cell->height(),
           cell->depth());
  }
}

inline void Simulation3D::initialize_grid_fcc(int num_cells, float radius,
                                              float confluence) {
  domain.initialize_grid_fcc(num_cells, radius, confluence);
  printf("Initialized %d 3D cells on FCC lattice\n", domain.num_cells());
  for (int i = 0; i < domain.num_cells(); ++i) {
    auto &cell = domain.cells[i];
    printf("  Cell %d: center=(%.1f, %.1f, %.1f), "
           "subdomain=[%d,%d,%d]->[%d,%d,%d] (%dx%dx%d)\n",
           cell->id, cell->centroid.x, cell->centroid.y, cell->centroid.z,
           cell->bbox_with_halo.x0, cell->bbox_with_halo.y0,
           cell->bbox_with_halo.z0, cell->bbox_with_halo.x1,
           cell->bbox_with_halo.y1, cell->bbox_with_halo.z1, cell->width(),
           cell->height(), cell->depth());
  }
}

inline void Simulation3D::initialize_grid(int num_cells, float radius,
                                          float confluence) {
  domain.initialize_grid(num_cells, radius, confluence);
  printf("Initialized %d 3D cells on grid\n", domain.num_cells());
  for (int i = 0; i < domain.num_cells(); ++i) {
    auto &cell = domain.cells[i];
    printf("  Cell %d: center=(%.1f, %.1f, %.1f), "
           "subdomain=[%d,%d,%d]->[%d,%d,%d] (%dx%dx%d)\n",
           cell->id, cell->centroid.x, cell->centroid.y, cell->centroid.z,
           cell->bbox.x0, cell->bbox.y0, cell->bbox.z0, cell->bbox.x1,
           cell->bbox.y1, cell->bbox.z1, cell->width(), cell->height(),
           cell->depth());
  }
}

inline bool Simulation3D::load_checkpoint(const char *filename) {
  float loaded_time = 0.0f;
  bool ok = load_checkpoint_3d(filename, domain, current_step, loaded_time);
  current_time = static_cast<double>(loaded_time);
  return ok;
}

inline void Simulation3D::run(float t_end) {
  start_time = std::chrono::steady_clock::now();

  printf("Starting 3D simulation: t_end=%.2f, dt=%.4f\n", t_end,
         domain.params.dt);

  // Compute trajectory save interval (same logic as 2D)
  // trajectory_interval: -1 = auto, 0 = compute from samples, >0 = use directly
  int total_steps =
      static_cast<int>((t_end - current_time) / domain.params.dt);
  int traj_interval;
  if (trajectory_interval > 0) {
    traj_interval = trajectory_interval;
  } else if (trajectory_interval == 0 || (trajectory_interval == -1 && trajectory_samples > 0)) {
    traj_interval = (trajectory_samples > 0)
                        ? std::max(1, total_steps / trajectory_samples)
                        : 0;
  } else {
    traj_interval = save_interval;
  }

  // Setup trajectory file
  if (traj_interval > 0) {
    std::string trajectory_file = output_dir + "/trajectory.txt";
    std::ifstream check_file(trajectory_file);
    if (check_file.good() && resumed_from_checkpoint) {
      check_file.close();
      // Truncate to the last complete line — removes any partial writes or
      // null bytes left by a previous SIGTERM at a chain-job boundary.
      {
        std::fstream repair(trajectory_file,
                            std::ios::in | std::ios::out | std::ios::binary);
        if (repair.good()) {
          repair.seekg(0, std::ios::end);
          long long fsize = static_cast<long long>(repair.tellg());
          if (fsize > 0) {
            long long scan_limit = std::min(fsize, (long long)8192);
            std::vector<char> tail(scan_limit);
            repair.seekg(fsize - scan_limit);
            repair.read(tail.data(), scan_limit);
            long long good_end = 0;
            for (long long i = scan_limit - 1; i >= 0; --i) {
              if (tail[i] == '\n' && i > 0 && tail[i - 1] != '\0') {
                good_end = (fsize - scan_limit) + i + 1;
                break;
              }
            }
            if (good_end > 0 && good_end < fsize) {
              std::filesystem::resize_file(trajectory_file, good_end);
              printf("Trajectory repair: truncated from %lld to %lld bytes "
                     "(removed %lld bytes of partial/corrupt data)\n",
                     fsize, good_end, fsize - good_end);
            }
          }
        }
      }
      printf("Trajectory output: appending (every %d steps)\n", traj_interval);
    } else {
      std::ofstream hdr(trajectory_file, std::ios::trunc);
      hdr << "# 3D Trajectory data\n";
      hdr << "# Format: time cell_id x y z vx vy vz px py pz theta phi_pol v_A_i\n";
      hdr << "# dim=3"
          << " v_A=" << domain.params.v_A
          << " N=" << domain.num_cells()
          << " Lx=" << domain.params.Nx
          << " Ly=" << domain.params.Ny
          << " Lz=" << domain.params.Nz
          << "\n";
      hdr.close();
      printf("Trajectory output: every %d steps\n", traj_interval);
    }
  }

  // Print interval (same as 2D: use print_interval if set, else save_interval, else 10000)
  int effective_print_interval = (print_interval > 0) ? print_interval
                                  : (save_interval > 0) ? save_interval : 10000;

  // Do first step to trigger all lazy allocations
  step(true);  // Sync on first step for memory profiling

  // Memory profiling: report GPU memory usage after allocations
  {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;
    printf("\n=== GPU Memory Profile (after initialization) ===\n");
    printf("  Total GPU memory:     %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Used (all processes): %.2f GB (%.1f%%)\n", 
           used_mem / (1024.0 * 1024.0 * 1024.0),
           100.0 * used_mem / total_mem);
    printf("  Free:                 %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
    
    // Detailed breakdown
    size_t cell_phi_mem = domain.total_gpu_memory_bytes();
    size_t work_buffer_mem = integrator.work_buffer_size;
    size_t reduction_mem = integrator.reduction_array_capacity * sizeof(float) * 16;
    size_t interaction_mem = integrator.interaction_array_capacity * sizeof(float) * 8;
    size_t neighbor_mem = domain.num_cells() * (1 + Integrator3D::MAX_NEIGHBORS_3D) * sizeof(int);
    size_t total_sim_mem = cell_phi_mem + work_buffer_mem + reduction_mem + interaction_mem + neighbor_mem;
    
    printf("\n  Simulation memory breakdown:\n");
    printf("    Cell phi fields:    %7.2f MB (%d cells)\n", 
           cell_phi_mem / (1024.0 * 1024.0), domain.num_cells());
    printf("    Work buffers (5x):  %7.2f MB\n", 
           work_buffer_mem / (1024.0 * 1024.0));
    printf("    Reduction arrays:   %7.2f MB\n", 
           reduction_mem / (1024.0 * 1024.0));
    printf("    Interaction arrays: %7.2f MB\n", 
           interaction_mem / (1024.0 * 1024.0));
    printf("    Neighbor lists:     %7.2f MB\n", 
           neighbor_mem / (1024.0 * 1024.0));
    printf("    ---------------------------------\n");
    printf("    Estimated total:    %7.2f MB (%.2f GB)\n",
           total_sim_mem / (1024.0 * 1024.0),
           total_sim_mem / (1024.0 * 1024.0 * 1024.0));
    
    // Scaling estimate
    float mem_per_cell = total_sim_mem / (float)domain.num_cells();
    int max_cells_fit = (int)(free_mem / mem_per_cell);
    printf("\n  Scaling estimates (current domain size):\n");
    printf("    Memory per cell:    %7.2f MB\n", mem_per_cell / (1024.0 * 1024.0));
    printf("    Max cells (free):   %d cells\n", max_cells_fit);
    printf("    Max cells (total):  %d cells\n", (int)(total_mem / mem_per_cell));
    printf("================================================\n\n");
  }

  // Checkpoint interval (same as 2D: default = save_interval*10, or 50000 if save_interval=0)
  int ckpt_interval = (checkpoint_interval > 0) ? checkpoint_interval
                       : (save_interval > 0) ? save_interval * 10 : 50000;

  // Save initial trajectory point (same as 2D)
  if (traj_interval > 0 && !resumed_from_checkpoint) {
    save_trajectory();
  }

  while (current_time < t_end) {
    // Check for SIGTERM (SLURM walltime limit) — clean shutdown
    if (g_shutdown_requested) {
      printf("\nSIGTERM received — shutting down cleanly at step %d, t=%.4f\n",
             current_step, current_time);
      fflush(stdout);
      break;
    }
    // Determine if we need host data this step
    int next_step = current_step + 1;
    bool need_host_sync = (effective_print_interval > 0 && next_step % effective_print_interval == 0) ||
                          (save_interval > 0 && next_step % save_interval == 0) ||
                          (traj_interval > 0 && next_step % traj_interval == 0) ||
                          (ckpt_interval > 0 && next_step % ckpt_interval == 0);

    step(need_host_sync);

    if (effective_print_interval > 0 && current_step % effective_print_interval == 0) {
      print_status();
    }

    if (traj_interval > 0 && current_step % traj_interval == 0) {
      save_trajectory();
    }

    // Periodic checkpoint (independent of VTK, for chain reliability)
    if (ckpt_interval > 0 && current_step % ckpt_interval == 0) {
      save_checkpoint();
    }

    if (save_interval > 0 && current_step % save_interval == 0) {
      save_vtk();
      if (save_individual_fields_flag) {
        save_individual_cell_fields();
      }
    }
  }

  // Always save final trajectory + checkpoint
  if (traj_interval > 0 && current_step % traj_interval != 0) {
    save_trajectory();
  }
  // Always save checkpoint for chain resumability
  save_checkpoint();
  if (save_interval > 0 && current_step % save_interval != 0) {
    save_vtk();
    if (save_individual_fields_flag) {
      save_individual_cell_fields();
    }
  }

  // Wait for any pending async checkpoint write before stopping timer
  async_writer.wait();

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  if (g_shutdown_requested) {
    printf("\nClean shutdown complete: checkpoint + trajectory saved at step %d, t=%.2f\n",
           current_step, current_time);
  } else {
    printf("\n3D Simulation complete: %d steps, t=%.2f\n", current_step,
           current_time);
  }
  printf("Total wall time: %.3f seconds\n", elapsed.count() / 1000.0);
  fflush(stdout);
  
  // Print kernel profiling results if enabled
  print_3d_kernel_profile();
}

inline void Simulation3D::step(bool sync_to_host) {
  float dt = domain.params.dt;

  // Polarization and velocity are updated on GPU inside integrator
  // (kernel_update_polarizations_3d + kernel_compute_velocities_3d)
  integrator.step(domain, dt, sync_to_host);

  current_step++;
  current_time += static_cast<double>(dt);  // double avoids precision loss at large t
}

inline void Simulation3D::save_checkpoint() {
  if (output_dir.empty())
    return;

  // Always overwrite checkpoint.bin (same as 2D, for chain resumability)
  std::string filename = output_dir + "/checkpoint.bin";
  async_writer.submit(domain, current_step, static_cast<float>(current_time),
                      filename);
  printf("Saved 3D checkpoint: step=%d, t=%.4f, cells=%d\n", current_step,
         current_time, domain.num_cells());
}

inline void Simulation3D::save_vtk() {
  if (output_dir.empty())
    return;

  char filename[256];
  snprintf(filename, sizeof(filename), "%s/cells_3d_%06d.vtk",
           output_dir.c_str(), current_step);
  save_vtk_3d(filename, domain);
}

inline void Simulation3D::save_individual_cell_fields() {
  if (output_dir.empty())
    return;

  for (int i = 0; i < domain.num_cells(); ++i) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/cell3d_%d_%06d.vtk",
             output_dir.c_str(), domain.cells[i]->id, current_step);
    save_cell_vtk_3d(filename, *domain.cells[i], domain.params);
  }
}

inline void Simulation3D::save_trajectory() {
  if (output_dir.empty()) return;
  int num_cells = domain.num_cells();
  std::string trajectory_file = output_dir + "/trajectory.txt";

  std::ofstream file(trajectory_file, std::ios::app);
  file << std::fixed << std::setprecision(6);
  for (int i = 0; i < num_cells; ++i) {
    const auto &cell = domain.cells[i];
    file << current_time << " " << cell->id
         << " " << cell->centroid.x << " " << cell->centroid.y << " " << cell->centroid.z
         << " " << cell->velocity.x << " " << cell->velocity.y << " " << cell->velocity.z
         << " " << cell->polarization.x << " " << cell->polarization.y << " " << cell->polarization.z
         << " " << cell->theta << " " << cell->phi_pol
         << " " << domain.params.v_A
         << "\n";
  }
  file.flush();  // Explicit flush before close — survives SIGTERM between flush and destructor
  file.close();
}

inline void Simulation3D::print_status() {
  printf("Step %6d | t=%.4f\n", current_step, current_time);
}

} // namespace cellsim
