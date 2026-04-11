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

#ifdef ENABLE_VISUALIZER
#include "visualizer.cuh"
#endif

// Stress fields also uses diagnostics.cuh for StressFieldBuffers
#ifdef STRESS_FIELDS_ENABLED
#ifndef DIAGNOSTICS_ENABLED
#include "diagnostics.cuh"
#endif
#endif

#include <algorithm>
#include <csignal>
#include <filesystem>
#include <iomanip>

// Defined in main.cu — set by SIGTERM handler for clean SLURM shutdown
extern volatile std::sig_atomic_t g_shutdown_requested;

namespace cellsim {

//=============================================================================
// Simulation - Top-level simulation controller
//=============================================================================

class Simulation {
public:
  Domain domain;
  Integrator integrator;

  double current_time;   // double to avoid float32 precision loss at large t
  int current_step;

  // Output settings
  std::string output_dir;
  int save_interval;
  int print_interval;     // Steps between progress output (-1 = use save_interval)
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

  std::vector<float> loaded_v_A; // Per-cell v_A loaded from checkpoint (empty if not present)
  std::vector<float> loaded_gamma; // Per-cell gamma loaded from checkpoint
  std::vector<float> loaded_target_radius; // Per-cell radius loaded from checkpoint
  std::vector<SimParams::GammaOverride> gamma_overrides; // CLI --gamma V:selector overrides
  bool gamma_overrides_set = false;
  std::vector<SimParams::RadiusOverride> radius_overrides; // CLI --radius V:selector overrides
  bool radius_overrides_set = false;

  AsyncVTKWriter vtk_writer;  // Async binary VTK output (GPU scatter + background file write)

#ifdef ENABLE_VISUALIZER
  Visualizer *visualizer = nullptr;   // Non-owning pointer; set externally
  int visualize_interval = 100;       // Steps between display updates
#endif

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
  void step(bool sync_polarization = false, bool sync_centroids = false);

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
      current_time(0.0), current_step(0), output_dir("./output"),
      save_interval(0), print_interval(-1), checkpoint_interval(-1), trajectory_samples(100),
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
  current_time = 0.0;
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

  current_time = 0.0;
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

  current_time = 0.0;
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
  std::vector<float> checkpoint_v_A;
  std::vector<float> checkpoint_gamma;
  std::vector<float> checkpoint_target_radius;
  if (!load_checkpoint(domain, filename, header, &checkpoint_v_A, &checkpoint_gamma, &checkpoint_target_radius)) {
    return false;
  }
  current_step = header.current_step;
  current_time = static_cast<double>(header.current_time);

  // Restore runtime options from checkpoint (v3+)
  save_interval = header.save_interval;
  checkpoint_interval = header.checkpoint_interval;
  trajectory_samples = header.trajectory_samples;
  save_vtk = header.save_vtk;
  save_tracking = header.save_tracking;
  compute_diagnostics = header.compute_diagnostics;
  save_individual_fields = header.save_individual_fields;
  resumed_from_checkpoint = true;

  // Store loaded v_A, gamma, and radius for later upload to integrator (before first step)
  loaded_v_A = std::move(checkpoint_v_A);
  loaded_gamma = std::move(checkpoint_gamma);
  loaded_target_radius = std::move(checkpoint_target_radius);

  return true;
}

inline void Simulation::step(bool sync_polarization, bool sync_centroids) {
  integrator.step(domain, domain.params.dt, sync_polarization, sync_centroids);
  current_time += static_cast<double>(domain.params.dt);
  current_step++;
}

inline void Simulation::run() {
  printf("Starting simulation: t_end=%.2f, dt=%.4f\n", domain.params.t_end,
         domain.params.dt);

  // Pass checkpoint-loaded v_A values to integrator (before first step)
  if (!loaded_v_A.empty()) {
    integrator.checkpoint_v_A = std::move(loaded_v_A);
  }
  // Pass checkpoint-loaded gamma values to integrator
  if (!loaded_gamma.empty()) {
    integrator.checkpoint_gamma = std::move(loaded_gamma);
  }
  // Pass checkpoint-loaded radius values to integrator
  if (!loaded_target_radius.empty()) {
    integrator.checkpoint_target_radius = std::move(loaded_target_radius);
  }
  // Pass CLI gamma overrides to integrator
  if (gamma_overrides_set) {
    integrator.gamma_overrides = std::move(gamma_overrides);
    integrator.gamma_overrides_set = true;
  }
  // Pass CLI radius overrides to integrator
  if (radius_overrides_set) {
    integrator.radius_overrides = std::move(radius_overrides);
    integrator.radius_overrides_set = true;
  }

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
    cudaError_t err = stress_fields_allocate(stress_buffers, 
                                             domain.params.Nx, 
                                             domain.params.Ny);
    if (err != cudaSuccess) {
      printf("ERROR: Failed to allocate stress field buffers: %s\n",
             cudaGetErrorString(err));
      save_stress_fields = false;
    } else {
      stress_initialized = true;
      printf("Stress field output enabled in VTK files\n");
    }
  }
#endif

  // Compute effective checkpoint interval
  // -1 = auto (save_interval*10 or 1000), 0 = disabled, >0 = explicit
  int ckpt_interval = 0;
  if (checkpoint_interval > 0) {
    ckpt_interval = checkpoint_interval;
  } else if (checkpoint_interval < 0) {
    ckpt_interval = (save_interval > 0) ? save_interval * 10 : 1000;
  }
  // checkpoint_interval == 0 → ckpt_interval stays 0 (disabled)

  // Compute trajectory save interval
  // trajectory_interval: -1 = auto, 0 = compute from samples, >0 = use directly
  int total_steps =
      static_cast<int>((domain.params.t_end - current_time) / domain.params.dt);
  int traj_interval;
  if (trajectory_interval > 0) {
    // Explicit interval set by user
    traj_interval = trajectory_interval;
  } else if (trajectory_interval == 0 || (trajectory_interval == -1 && trajectory_samples > 0)) {
    // Compute from samples (either explicitly requested or auto with samples set)
    traj_interval = (trajectory_samples > 0)
                        ? std::max(1, total_steps / trajectory_samples)
                        : 0;
  } else {
    // -1 with no samples: fall back to save_interval
    traj_interval = save_interval;
  }

  // Setup trajectory file
  if (traj_interval > 0) {
    std::string trajectory_file = output_dir + "/trajectory.txt";

    // Check if trajectory file exists (for appending)
    std::ifstream check_file(trajectory_file);
    bool file_exists = check_file.good();
    check_file.close();

    if (resumed_from_checkpoint && file_exists) {
      // Truncate to the last complete line — removes any partial writes or
      // null bytes left by a previous SIGTERM at a chain-job boundary.
      {
        std::fstream repair(trajectory_file,
                            std::ios::in | std::ios::out | std::ios::binary);
        if (repair.good()) {
          repair.seekg(0, std::ios::end);
          long long fsize = static_cast<long long>(repair.tellg());
          if (fsize > 0) {
            // Scan backwards to find the last newline preceded by valid data
            long long scan_limit = std::min(fsize, (long long)8192);
            std::vector<char> tail(scan_limit);
            repair.seekg(fsize - scan_limit);
            repair.read(tail.data(), scan_limit);
            long long good_end = 0;
            for (long long i = scan_limit - 1; i >= 0; --i) {
              if (tail[i] == '\n' && i > 0 && tail[i - 1] != '\0') {
                // Found a newline preceded by a non-null byte — this is
                // the end of the last complete, uncorrupted line.
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
      // Append to existing trajectory file
      printf("Trajectory output: appending (every %d steps)\n", traj_interval);
    } else {
      // New simulation or new output directory: create fresh trajectory file
      // with header
      std::ofstream traj_out(trajectory_file, std::ios::trunc);
      traj_out << "# Trajectory data for MSD computation\n";
      traj_out << "# Format: time cell_id x y vx vy px py theta v_A_i L_n volume\n";
      traj_out << "# v_A=" << domain.params.v_A
               << " v_A_sigma=" << domain.params.v_A_sigma
               << " N=" << domain.num_cells()
               << " Lx=" << domain.params.Nx << " Ly=" << domain.params.Ny
               << " dt=" << domain.params.dt
               << " tau=" << domain.params.tau
               << " subdomain_padding=" << domain.params.subdomain_padding
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

  // Save initial trajectory point (skip on resume — L_n is 0 before first step)
  if (traj_interval > 0 && !resumed_from_checkpoint) {
    save_trajectory();
  }

  // Compute effective print interval (-1 means use save_interval)
  int effective_print_interval = (print_interval > 0) ? print_interval : save_interval;

  while (current_time < domain.params.t_end) {
    // Check for SIGTERM (SLURM walltime limit) — clean shutdown.
    // Force centroids sync so the final checkpoint has correct host state.
    if (g_shutdown_requested) {
      step(false, true);
      printf("\nSIGTERM received — shutting down cleanly at step %d, t=%.4f\n",
             current_step, current_time);
      fflush(stdout);
      break;
    }
    // Determine if we need to sync polarization for trajectory output
    // We check (current_step + 1) because we're about to increment it
    bool need_polarization_sync = (traj_interval > 0) && 
                                   ((current_step + 1) % traj_interval == 0);
    // Only sync centroids/volumes/velocities to host when actually needed
    // (saves cudaDeviceSynchronize + 5× D→H copies on non-output steps)
    int next_step = current_step + 1;
    bool need_centroids = false;
    if (traj_interval > 0 && next_step % traj_interval == 0)
      need_centroids = true;
    if (effective_print_interval > 0 && next_step % effective_print_interval == 0)
      need_centroids = true;
    if (save_interval > 0 && next_step % save_interval == 0)
      need_centroids = true;
    if (ckpt_interval > 0 && next_step % ckpt_interval == 0)
      need_centroids = true;
#ifdef DIAGNOSTICS_ENABLED
    if (observable_interval > 0 && next_step % observable_interval == 0)
      need_centroids = true;
#endif
    // Force sync on last step so final checkpoint has correct host state
    if (current_time + domain.params.dt >= domain.params.t_end)
      need_centroids = true;
    step(need_polarization_sync, need_centroids);

    // Periodic checkpointing (independent of VTK saves)
    if (ckpt_interval > 0 && current_step % ckpt_interval == 0) {
      vtk_writer.wait();  // Ensure async VTK write done before checkpoint D→H
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
    }
    
    if (effective_print_interval > 0 && current_step % effective_print_interval == 0) {
      if (compute_diagnostics) {
        print_diagnostics();
      } else {
        // Minimal progress output
        printf("Step %6d | t=%.4f\n", current_step, current_time);
      }
    }

#ifdef ENABLE_VISUALIZER
    if (visualizer && visualizer->is_initialized() &&
        current_step % visualize_interval == 0) {
      visualizer->update(integrator.get_sum_field(), domain.params.Nx,
                         domain.params.Ny,
                         integrator.d_centroids_x, integrator.d_centroids_y,
                         integrator.d_polarization_x, integrator.d_polarization_y,
                         integrator.d_all_offsets_x, integrator.d_all_offsets_y,
                         integrator.d_all_widths, integrator.d_all_heights,
                         integrator.d_second_moment_x, integrator.d_second_moment_y,
                         integrator.d_volumes,
                         domain.params.dx * domain.params.dy,
                         domain.num_cells(), (float)current_time,
                         visualizer->show_arrows, visualizer->show_bboxes);
      if (visualizer->should_close()) break;
    }
#endif
  }

  // Save final trajectory point
  if (traj_interval > 0) {
    save_trajectory();
  }

  // Always save a final checkpoint for chain resumability
  vtk_writer.wait();
  save_current_checkpoint(output_dir + "/checkpoint.bin");

  if (save_vtk) {
    save_output(); // Save final state
  }
  vtk_writer.wait();  // Ensure all async writes complete before exit
  if (g_shutdown_requested) {
    printf("Clean shutdown complete: checkpoint + trajectory saved at step %d, t=%.2f\n",
           current_step, current_time);
  } else {
    printf("Simulation complete: %d steps, t=%.2f\n", current_step, current_time);
  }
  fflush(stdout);
  
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
      // Stress fields still use the old synchronous path
      integrator.compute_stress_fields(domain, stress_buffers);
      export_vtk_with_stress(domain, stress_buffers, base, current_step);
    } else {
#endif
      // Use async binary VTK writer if Integrator arrays are ready
      if (integrator.interaction_array_capacity > 0) {
        // Lazy-initialize the writer on first use
        if (!vtk_writer.is_ready()) {
          vtk_writer.initialize(domain.params.Nx, domain.params.Ny);
        }
        vtk_writer.submit(
            integrator.d_all_phi_ptrs,
            integrator.d_all_widths, integrator.d_all_heights,
            integrator.d_all_offsets_x, integrator.d_all_offsets_y,
            domain.num_cells(),
            integrator.cached_max_w, integrator.cached_max_h,
            domain.params.Nx, domain.params.Ny, domain.params.halo_width,
            domain.params.dx, domain.params.dy,
            base, current_step);
      } else {
        // Initial save (before first step): use Domain's arrays
        if (!vtk_writer.is_ready()) {
          vtk_writer.initialize(domain.params.Nx, domain.params.Ny);
        }
        domain.sync_device_arrays();  // Ensure device arrays are up-to-date
        int max_w = 0, max_h = 0;
        for (const auto &cell : domain.cells) {
          max_w = std::max(max_w, cell->width());
          max_h = std::max(max_h, cell->height());
        }
        vtk_writer.submit(
            domain.d_cell_phi_ptrs,
            domain.d_cell_widths, domain.d_cell_heights,
            domain.d_cell_offsets_x, domain.d_cell_offsets_y,
            domain.num_cells(), max_w, max_h,
            domain.params.Nx, domain.params.Ny, domain.params.halo_width,
            domain.params.dx, domain.params.dy,
            base, current_step);
      }
#ifdef STRESS_FIELDS_ENABLED
    }
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
  // Format: time cell_id x y vx vy px py theta v_A_i L_n
  int num_cells = domain.num_cells();
  float R = domain.params.target_radius;

  // Copy per-cell v_A from GPU once per trajectory save
  std::vector<float> h_v_A(num_cells, domain.params.v_A);
  if (integrator.d_v_A != nullptr && num_cells > 0) {
    cudaMemcpy(h_v_A.data(), integrator.d_v_A,
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);
  }

  std::string trajectory_file = output_dir + "/trajectory.txt";
  std::ofstream file(trajectory_file, std::ios::app);
  file << std::fixed << std::setprecision(6);

  for (int i = 0; i < num_cells; ++i) {
    const auto &cell = domain.cells[i];
    // Normalized perimeter: L_n = (1/(2πR)) ∫|∇φ| dA
    // Factor of 2πR normalizes so L_n = 1 for perfectly circular cells
    // (∫|∇φ| dA ≈ 2πR for a tanh-profile circle with our interface width)
    float L_n = cell->perimeter / (2.0f * M_PI * R);
    file << current_time << " " << cell->id << " " << cell->centroid.x << " "
         << cell->centroid.y << " " << cell->velocity.x << " "
         << cell->velocity.y << " " << cell->polarization.x << " "
         << cell->polarization.y << " " << cell->theta << " "
         << h_v_A[i] << " " << L_n << " " << cell->volume << "\n";
  }
  file.flush();  // Explicit flush before close — survives SIGTERM between flush and destructor
  file.close();
}

inline void Simulation::save_current_checkpoint(const std::string &filename) {
  // Force sync of bbox/field_size from GPU to ensure checkpoint has valid phi fields
  integrator.sync_bbox_to_host(domain);

  CheckpointHeader header;
  header.current_step = current_step;
  header.current_time = static_cast<float>(current_time);  // checkpoint stores float
  header.save_interval = save_interval;
  header.checkpoint_interval = checkpoint_interval;
  header.trajectory_samples = trajectory_samples;
  header.save_vtk = save_vtk;
  header.save_tracking = save_tracking;
  header.compute_diagnostics = compute_diagnostics;
  header.save_individual_fields = save_individual_fields;

  // Copy per-cell v_A from GPU for checkpoint persistence
  int num_cells = domain.num_cells();
  std::vector<float> h_v_A;
  if (integrator.d_v_A != nullptr && num_cells > 0) {
    h_v_A.resize(num_cells);
    cudaMemcpy(h_v_A.data(), integrator.d_v_A,
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);
  }

  // Copy per-cell gamma from GPU for checkpoint persistence
  std::vector<float> h_gamma;
  if (integrator.d_gamma != nullptr && num_cells > 0) {
    h_gamma.resize(num_cells);
    cudaMemcpy(h_gamma.data(), integrator.d_gamma,
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);
  }

  // Copy per-cell target radius from GPU for checkpoint persistence
  std::vector<float> h_target_radius;
  if (integrator.d_target_radius != nullptr && num_cells > 0) {
    h_target_radius.resize(num_cells);
    cudaMemcpy(h_target_radius.data(), integrator.d_target_radius,
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);
  }

  save_checkpoint(domain, filename, header,
                  h_v_A.empty() ? nullptr : h_v_A.data(),
                  static_cast<int>(h_v_A.size()),
                  h_gamma.empty() ? nullptr : h_gamma.data(),
                  static_cast<int>(h_gamma.size()),
                  h_target_radius.empty() ? nullptr : h_target_radius.data(),
                  static_cast<int>(h_target_radius.size()));
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
