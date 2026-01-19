#include "cell.hpp"
#include "domain.hpp"
#include "integrator.hpp"
#include "io.hpp"
#include "types.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>

using namespace cellsim;

void print_usage(const char *program) {
  printf("Usage: %s [options]\n", program);
  printf("MPI/CPU implementation of cell simulation\n\n");
  printf("Options:\n");
  printf("  -n <num>      Number of cells (default: 8)\n");
  printf("  -r <radius>   Cell radius (default: 49)\n");
  printf("  -s <space>    Minimum spacing between cells (default: auto)\n");
  printf("  -N <size>     Domain size NxN (default: 256)\n");
  printf("  -t <time>     End time (default: 100)\n");
  printf("  -dt <step>    Time step (default: 0.01)\n");
  printf("  -o <dir>      Output directory (default: ./output)\n");
  printf("  -c <file>     Load from checkpoint (resume simulation)\n");
  printf("  --seed <n>    Random seed for reproducible results\n");
  printf("  --v-A <f>     Active motility velocity (default: 0)\n");
  printf("  --tau <f>     Reorientation time (default: 10000)\n");
  printf("  --abp         Use ABP model instead of Run-and-Tumble\n");
  printf("  --save-interval <n>   Steps between VTK saves (0=none, default: 100)\n");
  printf("  --print-interval <n>  Steps between progress output\n");
  printf("  --checkpoint-interval <n>  Steps between checkpoints\n");
  printf("  --trajectory-interval <n>  Steps between trajectory saves\n");
  printf("  --save-final-checkpoint  Save checkpoint at end\n");
  printf("  -h            Show this help\n");
}

int main(int argc, char *argv[]) {
  // Default parameters
  SimParams params;
  params.Nx = 256;
  params.Ny = 256;
  params.dt = 0.01f;
  params.t_end = 100.0f;
  params.target_radius = 49.0f;

  int num_cells = 8;
  float radius = 49.0f;
  float min_spacing = -1.0f;
  std::string output_dir = "./output";
  std::string checkpoint_file = "";
  int random_seed = -1;
  float v_A_override = -1.0f;
  float tau_override = -1.0f;
  bool use_abp = false;
  int save_interval = 100;
  int print_interval = -1;
  int checkpoint_interval = -1;
  int trajectory_interval = -1;
  bool save_final_checkpoint = false;

  // Parse command line
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-n" && i + 1 < argc) {
      num_cells = atoi(argv[++i]);
    } else if (arg == "-r" && i + 1 < argc) {
      radius = atof(argv[++i]);
      params.target_radius = radius;
    } else if (arg == "-N" && i + 1 < argc) {
      int size = atoi(argv[++i]);
      params.Nx = size;
      params.Ny = size;
    } else if ((arg == "-t" || arg == "-T") && i + 1 < argc) {
      params.t_end = atof(argv[++i]);
    } else if ((arg == "-dt" || arg == "--dt") && i + 1 < argc) {
      params.dt = atof(argv[++i]);
    } else if (arg == "-o" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if ((arg == "-s" || arg == "--min-spacing") && i + 1 < argc) {
      min_spacing = atof(argv[++i]);
    } else if (arg == "-c" && i + 1 < argc) {
      checkpoint_file = argv[++i];
    } else if (arg == "--seed" && i + 1 < argc) {
      random_seed = atoi(argv[++i]);
    } else if (arg == "--v-A" && i + 1 < argc) {
      v_A_override = atof(argv[++i]);
    } else if (arg == "--tau" && i + 1 < argc) {
      tau_override = atof(argv[++i]);
    } else if (arg == "--abp") {
      use_abp = true;
    } else if (arg == "--save-interval" && i + 1 < argc) {
      save_interval = atoi(argv[++i]);
    } else if (arg == "--print-interval" && i + 1 < argc) {
      print_interval = atoi(argv[++i]);
    } else if (arg == "--checkpoint-interval" && i + 1 < argc) {
      checkpoint_interval = atoi(argv[++i]);
    } else if (arg == "--trajectory-interval" && i + 1 < argc) {
      trajectory_interval = atoi(argv[++i]);
    } else if (arg == "--save-final-checkpoint") {
      save_final_checkpoint = true;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else {
      printf("Unknown option: %s\n", arg.c_str());
      print_usage(argv[0]);
      return 1;
    }
  }

  // Create output directory
  std::filesystem::create_directories(output_dir);

  // Seed random number generator
  if (random_seed >= 0) {
    srand(static_cast<unsigned>(random_seed));
    printf("Using random seed: %d\n", random_seed);
  } else {
    random_seed = static_cast<int>(time(nullptr));
    srand(static_cast<unsigned>(random_seed));
  }

  // Apply v_A and tau overrides
  if (v_A_override >= 0.0f) {
    params.v_A = v_A_override;
  }
  if (tau_override > 0.0f) {
    params.tau = tau_override;
  }
  if (use_abp) {
    params.motility_model = SimParams::MotilityModel::ABP;
  }

  // Print parameters
  printf("MPI Cell Simulation (CPU)\n");
  printf("=========================\n");
  printf("Simulation Parameters:\n");
  printf("  Domain: %d x %d\n", params.Nx, params.Ny);
  printf("  Time step: dt=%.4f\n", params.dt);
  printf("  End time: t_end=%.1f\n", params.t_end);
  printf("  Target radius: R=%.1f (area=%.1f)\n", params.target_radius, 
         params.target_area());
  printf("  Active velocity: v_A=%.6f\n", params.v_A);
  printf("  Reorientation time: tau=%.1f\n", params.tau);
  printf("  Motility model: %s\n",
         params.motility_model == SimParams::MotilityModel::ABP
             ? "ABP" : "Run-and-Tumble");
  printf("  Cells: %d\n", num_cells);
  printf("\n");

  // Create domain and integrator
  Domain domain(params);
  Integrator integrator;
  integrator.set_seed(static_cast<unsigned>(random_seed));

  float current_time = 0.0f;
  int current_step = 0;

  // Initialize
  bool resumed = false;
  if (!checkpoint_file.empty()) {
    CheckpointHeader header;
    float saved_t_end = params.t_end;
    
    if (load_checkpoint(domain, checkpoint_file, header)) {
      resumed = true;
      current_step = header.current_step;
      current_time = header.current_time;
      
      // Restore t_end from command line (checkpoint shouldn't override target)
      domain.params.t_end = saved_t_end;
      
      // Apply command-line overrides
      if (v_A_override >= 0.0f) {
        domain.params.v_A = v_A_override;
      }
      if (tau_override > 0.0f) {
        domain.params.tau = tau_override;
      }
      if (use_abp) {
        domain.params.motility_model = SimParams::MotilityModel::ABP;
      }
      
      printf("Resumed from checkpoint: step=%d, t=%.4f\n", current_step, current_time);
      
      if (current_time >= saved_t_end) {
        printf("Simulation already complete\n");
        return 0;
      }
    } else {
      printf("Warning: Could not load checkpoint, starting fresh\n");
    }
  }

  if (!resumed) {
    // Auto-calculate min_spacing
    if (min_spacing < 0) {
      float domain_area = static_cast<float>(params.Nx * params.Ny);
      float area_per_cell = domain_area / num_cells;
      float ideal_spacing = sqrtf(area_per_cell);
      min_spacing = fmaxf(2.0f * radius, ideal_spacing * 0.8f);
      printf("Auto min_spacing: %.1f\n", min_spacing);
    }
    
    domain.initialize_random_cells(num_cells, radius, min_spacing);
    
    printf("Initialized %d cells\n", domain.num_cells());
    for (int i = 0; i < domain.num_cells(); ++i) {
      const auto &cell = domain.cells[i];
      printf("  Cell %d: center=(%.1f, %.1f), size=%dx%d\n",
             i, cell->centroid.x, cell->centroid.y, 
             cell->width(), cell->height());
    }
  }

  // Compute effective intervals
  int ckpt_interval = (checkpoint_interval > 0) ? checkpoint_interval :
                      (save_interval > 0 ? save_interval * 10 : 1000);
  int traj_interval = (trajectory_interval > 0) ? trajectory_interval : save_interval;
  int prnt_interval = (print_interval > 0) ? print_interval : save_interval;

  // Setup trajectory file
  if (traj_interval > 0) {
    std::string trajectory_file = output_dir + "/trajectory.txt";
    if (!resumed) {
      std::ofstream traj_out(trajectory_file, std::ios::trunc);
      traj_out << "# Trajectory data for MSD computation\n";
      traj_out << "# Format: time cell_id x y vx vy px py theta\n";
      traj_out << "# v_A=" << domain.params.v_A << " N=" << domain.num_cells()
               << " Lx=" << domain.params.Nx << " Ly=" << domain.params.Ny << "\n";
      traj_out.close();
    }
    printf("Trajectory output: every %d steps\n", traj_interval);
  }

  // Save initial state
  if (save_interval > 0 && !resumed) {
    export_vtk(domain, output_dir + "/frame", current_step);
  }
  if (traj_interval > 0 && !resumed) {
    export_trajectory(domain, output_dir + "/trajectory.txt", current_time);
  }

  // Run simulation
  printf("\nStarting simulation...\n");
  auto start_time = std::chrono::high_resolution_clock::now();

  while (current_time < domain.params.t_end) {
    integrator.step(domain, domain.params.dt);
    current_time += domain.params.dt;
    current_step++;

    // Checkpoint
    if (ckpt_interval > 0 && current_step % ckpt_interval == 0) {
      CheckpointHeader header;
      header.current_step = current_step;
      header.current_time = current_time;
      header.save_interval = save_interval;
      header.checkpoint_interval = checkpoint_interval;
      header.trajectory_samples = 0;
      header.save_vtk = (save_interval > 0);
      header.save_tracking = true;
      header.compute_diagnostics = false;
      header.save_individual_fields = false;
      save_checkpoint(domain, output_dir + "/checkpoint.bin", header);
    }

    // Trajectory
    if (traj_interval > 0 && current_step % traj_interval == 0) {
      export_trajectory(domain, output_dir + "/trajectory.txt", current_time);
    }

    // VTK output
    if (save_interval > 0 && current_step % save_interval == 0) {
      export_vtk(domain, output_dir + "/frame", current_step);
    }

    // Progress output
    if (prnt_interval > 0 && current_step % prnt_interval == 0) {
      // Compute average volume
      float total_volume = 0.0f;
      for (const auto &cell : domain.cells) {
        total_volume += cell->volume;
      }
      float avg_volume = total_volume / domain.num_cells();
      
      printf("Step %6d | t=%.4f | avg_vol=%.1f (target=%.1f)\n",
             current_step, current_time, avg_volume, domain.params.target_area());
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(end_time - start_time).count();

  // Final outputs
  if (traj_interval > 0) {
    export_trajectory(domain, output_dir + "/trajectory.txt", current_time);
  }
  if (save_interval > 0) {
    export_vtk(domain, output_dir + "/frame", current_step);
  }
  if (save_final_checkpoint) {
    CheckpointHeader header;
    header.current_step = current_step;
    header.current_time = current_time;
    header.save_interval = save_interval;
    header.checkpoint_interval = checkpoint_interval;
    header.trajectory_samples = 0;
    header.save_vtk = (save_interval > 0);
    header.save_tracking = true;
    header.compute_diagnostics = false;
    header.save_individual_fields = false;
    save_checkpoint(domain, output_dir + "/checkpoint.bin", header);
  }

  printf("\nSimulation finished!\n");
  printf("Total wall time: %.3f seconds\n", elapsed);
  printf("Final state: step=%d, t=%.4f\n", current_step, current_time);
  printf("Output saved to: %s\n", output_dir.c_str());

  return 0;
}
