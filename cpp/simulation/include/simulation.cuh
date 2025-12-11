#pragma once

#include "cell.cuh"
#include "domain.cuh"
#include "integrator.cuh"
#include "io.cuh"
#include "kernels.cuh"
#include "types.cuh"
#ifdef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"
#endif

// Stress fields also uses diagnostics.cuh for StressFieldBuffers
#ifdef STRESS_FIELDS_ENABLED
#ifndef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"
#endif
#endif

#include <algorithm>
#include <filesystem>
#include <iomanip>

namespace cellsim {

//=============================================================================
// Simulation - Top-level simulation controller
//=============================================================================

class Simulation {
public:
  Domain domain;
  Integrator integrator;

  float current_time;
  int current_step;

  // Output settings
  std::string output_dir;
  int save_interval;
  int checkpoint_interval; // Steps between checkpoints (-1 = save_interval*10)
  int trajectory_samples; // Number of trajectory samples to save (default: 100)
  int trajectory_interval; // Steps between trajectory saves (-1 = use
                           // save_interval, 0 = compute from samples)
  int observable_interval; // Steps between GPU-side diagnostic measurements (0 = disabled)
  bool save_vtk;
  bool save_tracking;
  bool compute_diagnostics;     // Compute volume/shape (disable for speed)
  bool resumed_from_checkpoint; // True if initialized from checkpoint
  bool
      save_individual_fields; // Save individual cell fields for energy analysis

#ifdef DIAGNOSTICS_ENABLED
  DiagnosticBuffers diag_buffers;
  bool diag_initialized;
#endif

#ifdef STRESS_FIELDS_ENABLED
  StressFieldBuffers stress_buffers;
  bool stress_initialized;
  bool save_stress_fields;  // Whether to include stress in VTK output
#endif

public:
  Simulation(const SimParams &params);
  ~Simulation() = default;

  // Initialize with random cells
  void initialize_random(int num_cells, float radius, float min_spacing);

  // Initialize with cells at edges/corners for boundary testing
  void initialize_edge_test(float radius);

  // Corner push test: one cell in corner, rest clustered to push it
  void initialize_corner_push_test(int num_cells, float radius);

  // Initialize from checkpoint (returns false if file not found or invalid)
  bool initialize_from_checkpoint(const std::string &filename);

  // Save checkpoint with current state
  void save_current_checkpoint(const std::string &filename);

  // Run simulation
  void run();

  // Single step (for interactive use)
  void step(bool sync_polarization = false);

  // Save current state
  void save_output();

  // Save trajectory data for MSD computation
  void save_trajectory();

  // Diagnostics
  void print_diagnostics() const;
};

//=============================================================================
// Simulation Implementation
//=============================================================================

inline Simulation::Simulation(const SimParams &params)
    : domain(params), integrator(Integrator::Method::ForwardEuler),
      current_time(0.0f), current_step(0), output_dir("./output"),
      save_interval(100), checkpoint_interval(-1), trajectory_samples(100),
      trajectory_interval(0), observable_interval(0), save_vtk(true), save_tracking(true),
      compute_diagnostics(false), resumed_from_checkpoint(false),
      save_individual_fields(false)
#ifdef DIAGNOSTICS_ENABLED
      , diag_buffers{}, diag_initialized(false)
#endif
#ifdef STRESS_FIELDS_ENABLED
      , stress_buffers{}, stress_initialized(false), save_stress_fields(false)
#endif
{}

inline void Simulation::initialize_random(int num_cells, float radius,
                                          float min_spacing) {
  domain.initialize_random_cells(num_cells, radius, min_spacing);
  current_time = 0.0f;
  current_step = 0;

  printf("Initialized %d cells\n", domain.num_cells());

  // Print subdomain info for all cells
  for (int i = 0; i < domain.num_cells(); ++i) {
    const auto &cell = domain.cells[i];
    printf(
        "  Cell %d: center=(%.1f, %.1f), subdomain=[%d,%d]->[%d,%d] (%dx%d)\n",
        i, cell->centroid.x, cell->centroid.y, cell->bbox_with_halo.x0,
        cell->bbox_with_halo.y0, cell->bbox_with_halo.x1,
        cell->bbox_with_halo.y1, cell->width(), cell->height());
  }

  if (compute_diagnostics) {
    print_diagnostics();
  }
}

inline void Simulation::initialize_edge_test(float radius) {
  // Place 3 cells at challenging positions:
  // Cell 0: Near bottom-left corner (will have negative x0 and y0)
  // Cell 1: Near right edge (will have x1 > Nx)
  // Cell 2: Near top edge (will have y1 > Ny)

  int Nx = domain.params.Nx;
  int Ny = domain.params.Ny;

  // Place cells so their subdomains will wrap
  float offset = radius * 0.5f; // Place center close to edge

  // Just one cell at corner for debugging
  domain.add_cell(offset, offset, radius); // Bottom-left corner
  // domain.add_cell(Nx - offset, Ny / 2.0f, radius);            // Right edge
  // domain.add_cell(Nx / 2.0f, Ny - offset, radius);            // Top edge

  domain.update_overlap_pairs();
  domain.sync_device_arrays();

  current_time = 0.0f;
  current_step = 0;

  printf("Initialized %d cells (edge test)\n", domain.num_cells());

  // Print subdomain info for all cells
  for (int i = 0; i < domain.num_cells(); ++i) {
    const auto &cell = domain.cells[i];
    printf(
        "  Cell %d: center=(%.1f, %.1f), subdomain=[%d,%d]->[%d,%d] (%dx%d)\n",
        i, cell->centroid.x, cell->centroid.y, cell->bbox_with_halo.x0,
        cell->bbox_with_halo.y0, cell->bbox_with_halo.x1,
        cell->bbox_with_halo.y1, cell->width(), cell->height());
  }

  if (compute_diagnostics) {
    print_diagnostics();
  }
}

inline void Simulation::initialize_corner_push_test(int num_cells,
                                                    float radius) {
  // Stress test for window tracking:
  // - Place one cell in the corner (near periodic boundary)
  // - Arrange remaining cells in an arc around it, all pushing TOWARD the
  // corner
  // - When system relaxes, the corner cell gets pushed hard into the boundary
  // - This tests whether window tracking follows a rapidly displaced cell

  int Nx = domain.params.Nx;
  int Ny = domain.params.Ny;

  // Cell 0: Place in corner, very close to boundary
  float corner_offset = radius * 0.6f;
  domain.add_cell(corner_offset, corner_offset, radius);

  // Remaining cells: arrange in an arc around the corner cell
  // The arc spans from ~45° to ~135° (pointing toward the corner)
  // Each cell overlaps with the corner cell, pushing it into the corner
  int num_pushers = num_cells - 1;
  float arc_radius = radius * 1.4f; // Distance from corner cell center
  float arc_start = M_PI * 0.25f;   // 45 degrees
  float arc_end = M_PI * 1.25f;     // 225 degrees (wraps around the corner)

  for (int i = 0; i < num_pushers; ++i) {
    // Distribute cells along the arc
    float angle = arc_start + (arc_end - arc_start) * i /
                                  (num_pushers > 1 ? num_pushers - 1 : 1);
    float cx = corner_offset + arc_radius * cosf(angle);
    float cy = corner_offset + arc_radius * sinf(angle);
    domain.add_cell(cx, cy, radius);
  }

  domain.update_overlap_pairs();
  domain.sync_device_arrays();

  current_time = 0.0f;
  current_step = 0;

  printf("Initialized %d cells (corner push test)\n", domain.num_cells());
  printf("  Corner cell at (%.1f, %.1f) - will be pushed into boundary\n",
         corner_offset, corner_offset);
  printf(
      "  %d cells in arc (r=%.1f) from %.0f° to %.0f° pushing toward corner\n",
      num_pushers, arc_radius, arc_start * 180.0f / M_PI,
      arc_end * 180.0f / M_PI);

  // Print subdomain info for all cells
  for (int i = 0; i < domain.num_cells(); ++i) {
    const auto &cell = domain.cells[i];
    printf(
        "  Cell %d: center=(%.1f, %.1f), subdomain=[%d,%d]->[%d,%d] (%dx%d)\n",
        i, cell->centroid.x, cell->centroid.y, cell->bbox_with_halo.x0,
        cell->bbox_with_halo.y0, cell->bbox_with_halo.x1,
        cell->bbox_with_halo.y1, cell->width(), cell->height());
  }

  if (compute_diagnostics) {
    print_diagnostics();
  }
}

inline bool
Simulation::initialize_from_checkpoint(const std::string &filename) {
  CheckpointHeader header;
  if (!load_checkpoint(domain, filename, header)) {
    return false;
  }
  current_step = header.current_step;
  current_time = header.current_time;

  // Restore runtime options from checkpoint (v3+)
  save_interval = header.save_interval;
  checkpoint_interval = header.checkpoint_interval;
  trajectory_samples = header.trajectory_samples;
  save_vtk = header.save_vtk;
  save_tracking = header.save_tracking;
  compute_diagnostics = header.compute_diagnostics;
  save_individual_fields = header.save_individual_fields;
  resumed_from_checkpoint = true;

  return true;
}

inline void Simulation::step(bool sync_polarization) {
  integrator.step(domain, domain.params.dt, sync_polarization);
  current_time += domain.params.dt;
  current_step++;
}

inline void Simulation::run() {
  printf("Starting simulation: t_end=%.2f, dt=%.4f\n", domain.params.t_end,
         domain.params.dt);

  // Create fields subdirectory if saving individual fields
  if (save_individual_fields) {
    std::string fields_dir = output_dir + "/fields";
    std::filesystem::create_directories(fields_dir);
    printf("Individual cell fields will be saved to: %s\n", fields_dir.c_str());
  }

#ifdef DIAGNOSTICS_ENABLED
  // Initialize GPU-side diagnostic buffers if observable_interval > 0
  if (observable_interval > 0 && !diag_initialized) {
    int num_cells = domain.num_cells();
    cudaError_t err = diagnostics_allocate(diag_buffers, num_cells);
    if (err != cudaSuccess) {
      printf("ERROR: Failed to allocate diagnostic buffers: %s\n", 
             cudaGetErrorString(err));
      observable_interval = 0; // Disable diagnostics
    } else {
      diag_initialized = true;
      
      // Create/clear observables file with header
      std::string obs_file = output_dir + "/observables.csv";
      FILE* f = fopen(obs_file.c_str(), "w");
      if (f) {
        diagnostics_write_header(f);
        fclose(f);
      }
      printf("GPU diagnostics enabled: every %d steps -> %s\n", 
             observable_interval, obs_file.c_str());
    }
  }
#endif

#ifdef STRESS_FIELDS_ENABLED
  // Initialize stress field buffers if stress output is enabled
  if (save_stress_fields && !stress_initialized) {
    // DEBUG: Check GPU state before allocation
    cudaDeviceSynchronize();
    cudaError_t pre_err = cudaGetLastError();
    printf("[DEBUG] Pre-allocation GPU state: %s\n", 
           pre_err == cudaSuccess ? "OK" : cudaGetErrorString(pre_err));
    
    cudaError_t err = stress_fields_allocate(stress_buffers, 
                                             domain.params.Nx, 
                                             domain.params.Ny);
    if (err != cudaSuccess) {
      printf("ERROR: Failed to allocate stress field buffers: %s\n",
             cudaGetErrorString(err));
      save_stress_fields = false;
    } else {
      // DEBUG: Check GPU state after allocation
      cudaDeviceSynchronize();
      cudaError_t post_err = cudaGetLastError();
      printf("[DEBUG] Post-allocation GPU state: %s\n", 
             post_err == cudaSuccess ? "OK" : cudaGetErrorString(post_err));
      
      stress_initialized = true;
      printf("Stress field output enabled in VTK files\n");
    }
  }
#endif

  // Compute effective checkpoint interval
  int ckpt_interval = (checkpoint_interval > 0)
                          ? checkpoint_interval
                          : (save_interval > 0 ? save_interval * 10 : 1000);

  // Compute trajectory save interval
  // trajectory_interval: -1 = use save_interval, 0 = compute from samples, >0 =
  // use directly
  int total_steps =
      static_cast<int>((domain.params.t_end - current_time) / domain.params.dt);
  int traj_interval;
  if (trajectory_interval == -1) {
    // Use save_interval
    traj_interval = save_interval;
  } else if (trajectory_interval > 0) {
    // Use the explicitly set interval
    traj_interval = trajectory_interval;
  } else {
    // Compute from trajectory_samples
    traj_interval = (trajectory_samples > 0)
                        ? std::max(1, total_steps / trajectory_samples)
                        : 0;
  }

  // Setup trajectory file
  if (traj_interval > 0) {
    std::string trajectory_file = output_dir + "/trajectory.txt";

    // Check if trajectory file exists (for appending)
    std::ifstream check_file(trajectory_file);
    bool file_exists = check_file.good();
    check_file.close();

    if (resumed_from_checkpoint && file_exists) {
      // Append to existing trajectory file
      printf("Trajectory output: appending (every %d steps)\n", traj_interval);
    } else {
      // New simulation or new output directory: create fresh trajectory file
      // with header
      std::ofstream traj_out(trajectory_file, std::ios::trunc);
      traj_out << "# Trajectory data for MSD computation\n";
      traj_out << "# Format: time cell_id x y vx vy px py theta\n";
      traj_out << "# v_A=" << domain.params.v_A << " N=" << domain.num_cells()
               << " Lx=" << domain.params.Nx << " Ly=" << domain.params.Ny
               << "\n";
      traj_out.close();
      printf("Trajectory output: every %d steps\n", traj_interval);
    }
  }

  if (save_vtk) {
    save_output(); // Save initial state
  } else {
    // Save initial checkpoint even if no VTK output
    save_current_checkpoint(output_dir + "/checkpoint.bin");
  }

  // Save initial trajectory point
  if (traj_interval > 0) {
    save_trajectory();
  }

  while (current_time < domain.params.t_end) {
    // Determine if we need to sync polarization for trajectory output
    // We check (current_step + 1) because we're about to increment it
    bool need_polarization_sync = (traj_interval > 0) && 
                                   ((current_step + 1) % traj_interval == 0);
    step(need_polarization_sync);

    // Periodic checkpointing (independent of VTK saves)
    if (ckpt_interval > 0 && current_step % ckpt_interval == 0) {
      save_current_checkpoint(output_dir + "/checkpoint.bin");
    }

    // Trajectory data (for MSD computation)
    if (traj_interval > 0 && current_step % traj_interval == 0) {
      save_trajectory();
    }

#ifdef DIAGNOSTICS_ENABLED
    // GPU-side diagnostic measurements
    if (observable_interval > 0 && current_step % observable_interval == 0) {
      // Reset buffers
      diagnostics_reset(diag_buffers);
      
      // Run GPU diagnostics through integrator (has all the device arrays)
      integrator.compute_diagnostics(domain, diag_buffers);
      
      // Collect results
      DiagnosticSample sample;
      diagnostics_collect(diag_buffers, sample, current_time, current_step);
      
      // Write to file
      std::string obs_file = output_dir + "/observables.csv";
      FILE* f = fopen(obs_file.c_str(), "a");
      if (f) {
        diagnostics_write(f, sample);
        fclose(f);
      }
    }
#endif

    if (save_interval > 0 && current_step % save_interval == 0) {
      if (save_vtk) {
        save_output();
      }
      if (compute_diagnostics) {
        print_diagnostics();
      } else {
        // Minimal progress output
        printf("Step %6d | t=%.4f\n", current_step, current_time);
      }
    }
  }

  // Save final trajectory point
  if (traj_interval > 0) {
    save_trajectory();
  }

  if (save_vtk) {
    save_output(); // Save final state
  }
  printf("Simulation complete: %d steps, t=%.2f\n", current_step, current_time);
  
#ifdef DIAGNOSTICS_ENABLED
  // Free diagnostic buffers
  if (diag_initialized) {
    diagnostics_free(diag_buffers);
    diag_initialized = false;
    printf("GPU diagnostic buffers freed\n");
  }
#endif

#ifdef STRESS_FIELDS_ENABLED
  // Free stress field buffers
  if (stress_initialized) {
    stress_fields_free(stress_buffers);
    stress_initialized = false;
    printf("Stress field buffers freed\n");
  }
#endif
  
  // Print neighbor list caching stats
  if (domain.num_cells() > 1) {
    int rebuilds = integrator.neighbor_rebuild_count;
    int skips = integrator.neighbor_skip_count;
    int total = rebuilds + skips;
    if (total > 0) {
      printf("Neighbor list: %d rebuilds, %d cached (%.1f%% cache hit rate)\n",
             rebuilds, skips, 100.0f * skips / total);
    }
  }
}

inline void Simulation::save_output() {
  std::string base = output_dir + "/frame";

  if (save_vtk) {
#ifdef STRESS_FIELDS_ENABLED
    if (save_stress_fields && stress_initialized) {
      // Compute stress fields before export
      integrator.compute_stress_fields(domain, stress_buffers);
      export_vtk_with_stress(domain, stress_buffers, base, current_step);
    } else {
      export_vtk(domain, base, current_step);
    }
#else
    export_vtk(domain, base, current_step);
#endif
  }

  // Save individual cell fields for energy analysis (if enabled)
  if (save_individual_fields) {
    std::string fields_base = output_dir + "/fields/frame";
    export_vtk_individual(domain, fields_base, current_step);

    // Also save energy metrics computed from individual fields
    std::string energy_file = output_dir + "/energy_metrics.txt";
    export_energy_metrics(domain, energy_file, current_step, current_time);
  }

  if (save_tracking) {
    std::string tracking_file = output_dir + "/tracking.txt";
    export_tracking_data(domain, tracking_file, current_time);
  }
  // Note: Checkpointing is now handled in run() loop for more control
}

inline void Simulation::save_trajectory() {
  // Save trajectory data for MSD/diffusion computation
  // Format: time cell_id x y vx vy px py theta
  std::string trajectory_file = output_dir + "/trajectory.txt";
  std::ofstream file(trajectory_file, std::ios::app);
  file << std::fixed << std::setprecision(6);

  for (const auto &cell : domain.cells) {
    file << current_time << " " << cell->id << " " << cell->centroid.x << " "
         << cell->centroid.y << " " << cell->velocity.x << " "
         << cell->velocity.y << " " << cell->polarization.x << " "
         << cell->polarization.y << " " << cell->theta << "\n";
  }
  file.close();
}

inline void Simulation::save_current_checkpoint(const std::string &filename) {
  CheckpointHeader header;
  header.current_step = current_step;
  header.current_time = current_time;
  header.save_interval = save_interval;
  header.checkpoint_interval = checkpoint_interval;
  header.trajectory_samples = trajectory_samples;
  header.save_vtk = save_vtk;
  header.save_tracking = save_tracking;
  header.compute_diagnostics = compute_diagnostics;
  header.save_individual_fields = save_individual_fields;
  save_checkpoint(domain, filename, header);
}

inline void Simulation::print_diagnostics() const {
  printf("Step %6d | t=%.4f | Cells=%d", current_step, current_time,
         domain.num_cells());

  if (domain.num_cells() > 0) {
    float total_volume = 0.0f;
    float total_shape = 0.0f;
    for (const auto &cell : domain.cells) {
      total_volume += cell->volume;
      total_shape += cell->compute_shape_factor(domain.params);
    }
    float avg_volume = total_volume / domain.num_cells();
    float avg_shape = total_shape / domain.num_cells();
    float target = domain.params.target_area();

    // For single cell, print more info
    if (domain.num_cells() == 1) {
      const auto &cell = domain.cells[0];
      printf(" | Vol=%.1f | R_eff=%.1f | Shape=%.3f", avg_volume,
             sqrtf(avg_volume / M_PI), avg_shape);
    } else if (domain.num_cells() <= 4) {
      // Print each cell's volume for debugging
      printf(" | Vols=[");
      for (int i = 0; i < domain.num_cells(); ++i) {
        printf("%.0f", domain.cells[i]->volume);
        if (i < domain.num_cells() - 1)
          printf(",");
      }
      printf("] (%.1f) | Shape=%.3f", target, avg_shape);
    } else {
      printf(" | Vol=%.1f (%.1f) | Shape=%.3f", avg_volume, target, avg_shape);
    }
  }

  printf("\n");
}

} // namespace cellsim
