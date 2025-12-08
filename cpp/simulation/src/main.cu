#include "simulation.cuh"
#include "simulation3d.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

using namespace cellsim;

void print_usage(const char *program) {
  printf("Usage: %s [options]\n", program);
  printf("Options:\n");
  printf("  --3d          Run 3D simulation (default: 2D)\n");
  printf("  -n <num>      Number of cells (default: 8)\n");
  printf("  -r <radius>   Cell radius (default: 20)\n");
  printf("  -s <space>    Minimum spacing between cells (default: auto)\n");
  printf("  -N <size>     Domain size NxN (2D) or NxNxN (3D) (default: 256)\n");
  printf("  -Nz <size>    Z dimension for 3D (default: same as N)\n");
  printf("  -t <time>     End time (default: 100)\n");
  printf("  -dt <step>    Time step (default: 0.01)\n");
  printf("  -o <dir>      Output directory (default: ./output)\n");
  printf("  -c <file>     Load from checkpoint (resume simulation)\n");
  printf(
      "  --edge-test   Place 3 cells at edges/corners for boundary testing\n");
  printf("  --corner-push-test  Stress test: corner cell + clustered cells "
         "pushing it\n");
  printf("  --no-self-propulsion  Disable active self-propulsion (v_A = 0)\n");
  printf("  --use-diagnostics     Enable volume/shape computation (disabled by "
         "default for speed)\n");
  printf("  --save-interval <n>   Steps between VTK saves (0 = no saves, "
         "default: 100)\n");
  printf("  --subdomain-padding <f>  Cell window size as multiple of R "
         "(default: 2.0, use 3.0 for ~6R window)\n");
  printf(
      "  --save-final-checkpoint  Save checkpoint at end (for job chaining)\n");
  printf("  --checkpoint-interval <n>  Steps between checkpoints (default: "
         "save_interval*10)\n");
  printf("  --seed <n>    Random seed for reproducible initial conditions\n");
  printf("  --trajectory-samples <n>  Number of trajectory samples to save "
         "(default: 100)\n");
  printf("  --trajectory-interval <n>  Steps between trajectory saves (-1 = "
         "use save_interval)\n");
  printf("  --v-A <f>     Active motility velocity (default: from params)\n");
  printf("  --abp         Use Active Brownian Particle model instead of "
         "Run-and-Tumble\n");
  printf("  --save-individual-fields  Save individual cell fields for energy "
         "analysis\n");
  printf("  --grid        Use grid-based cell initialization (for high "
         "confluence)\n");
  printf("  --confluence <f>  Target confluence 0-1 (implies --grid, default: "
         "0.85)\n");
  printf("  --legacy      Use legacy kernels (slow, for reference only)\n");
  printf("  -h            Show this help\n");
}

int main(int argc, char *argv[]) {
  // Default parameters
  SimParams params;
  params.Nx = 256;
  params.Ny = 256;
  params.dt = 0.01f;
  params.t_end = 100.0f;
  params.target_radius = 20.0f;

  // 3D-specific defaults
  bool run_3d = false;
  int Nz = -1; // -1 means use same as Nx/Ny
  bool domain_size_set = false; // Track if user explicitly set -N

  int num_cells = 8;
  float radius = 20.0f;
  float min_spacing =
      -1.0f; // -1 means auto-calculate based on radius and cell count
  std::string output_dir = "./output";
  std::string checkpoint_file = "";
  bool edge_test = false;
  bool corner_push_test = false;
  bool no_self_propulsion = false;
  bool use_diagnostics = false;
  bool save_final_checkpoint = false;
  bool save_individual_fields =
      false; // Save individual cell fields for energy analysis
  int save_interval = 100;
  int checkpoint_interval = -1; // -1 means use save_interval * 10
  int random_seed = -1;         // -1 means use time-based seed
  int trajectory_samples = 100; // Number of trajectory data points to save
  int trajectory_interval =
      -1; // -1 = use save_interval, 0 = compute from samples, >0 = explicit
  float v_A_override = -1.0f; // -1 means use default from params
  bool use_abp = false;       // Use ABP model instead of Run-and-Tumble
  bool safe_mode = false;   // Limit memory allocation to 1GB
  bool use_grid_init = false; // Use grid-based initialization instead of random
  float confluence = 0.85f;   // Target confluence for grid initialization

  // Parse command line
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--3d") {
      run_3d = true;
    } else if (arg == "-Nz" && i + 1 < argc) {
      Nz = atoi(argv[++i]);
    } else if (arg == "-n" && i + 1 < argc) {
      num_cells = atoi(argv[++i]);
    } else if (arg == "-r" && i + 1 < argc) {
      radius = atof(argv[++i]);
      params.target_radius = radius;
    } else if (arg == "-N" && i + 1 < argc) {
      int size = atoi(argv[++i]);
      params.Nx = size;
      params.Ny = size;
      domain_size_set = true;
    } else if ((arg == "-t" || arg == "-T") && i + 1 < argc) {
      params.t_end = atof(argv[++i]);
    } else if ((arg == "-dt" || arg == "--dt") && i + 1 < argc) {
      params.dt = atof(argv[++i]);
    } else if ((arg == "--lambda" || arg == "-l") && i + 1 < argc) {
      params.lambda = atof(argv[++i]);
    } else if (arg == "-o" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if ((arg == "-s" || arg == "--min-spacing") && i + 1 < argc) {
      min_spacing = atof(argv[++i]);
    } else if (arg == "-c" && i + 1 < argc) {
      checkpoint_file = argv[++i];
    } else if (arg == "--edge-test") {
      edge_test = true;
      num_cells = 3;
    } else if (arg == "--corner-push-test") {
      corner_push_test = true;
    } else if (arg == "--no-self-propulsion") {
      no_self_propulsion = true;
    } else if (arg == "--use-diagnostics") {
      use_diagnostics = true;
    } else if (arg == "--save-interval" && i + 1 < argc) {
      save_interval = atoi(argv[++i]);
    } else if (arg == "--subdomain-padding" && i + 1 < argc) {
      params.subdomain_padding = atof(argv[++i]);
    } else if (arg == "--save-final-checkpoint") {
      save_final_checkpoint = true;
    } else if (arg == "--checkpoint-interval" && i + 1 < argc) {
      checkpoint_interval = atoi(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      random_seed = atoi(argv[++i]);
    } else if (arg == "--trajectory-samples" && i + 1 < argc) {
      trajectory_samples = atoi(argv[++i]);
    } else if (arg == "--trajectory-interval" && i + 1 < argc) {
      trajectory_interval = atoi(argv[++i]);
    } else if (arg == "--v-A" && i + 1 < argc) {
      v_A_override = atof(argv[++i]);
    } else if (arg == "--abp") {
      use_abp = true;
    } else if (arg == "--save-individual-fields") {
      save_individual_fields = true;
    } else if (arg == "--safe-mode") {
      // Limit memory allocation to prevent runaway GPU memory usage
      safe_mode = true;
    } else if (arg == "--grid") {
      // Use grid-based initialization for high confluence
      use_grid_init = true;
    } else if (arg == "--confluence" && i + 1 < argc) {
      confluence = atof(argv[++i]);
      use_grid_init = true; // --confluence implies --grid
    } else if (arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
  }

// Create output directory
#ifdef _WIN32
  _mkdir(output_dir.c_str());
#else
  mkdir(output_dir.c_str(), 0755);
#endif

  // Seed random number generator
  if (random_seed >= 0) {
    srand(static_cast<unsigned>(random_seed));
    printf("Using random seed: %d\n", random_seed);
  } else {
    srand(static_cast<unsigned>(time(nullptr)));
  }

  // Apply no-self-propulsion flag or v_A override
  if (no_self_propulsion) {
    params.v_A = 0.0f;
  } else if (v_A_override >= 0.0f) {
    params.v_A = v_A_override;
  }

  // Apply motility model selection
  if (use_abp) {
    params.motility_model = SimParams::MotilityModel::ABP;
  }

  // Print CUDA device info with proper initialization
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  if (deviceCount == 0) {
    fprintf(stderr, "No CUDA devices found!\n");
    return 1;
  }
  
  int device = 0;
  err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: cudaSetDevice failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  printf("Using GPU: %s\n", prop.name);
  printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("  Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
  printf("\n");

  //=========================================================================
  // 3D Simulation Branch
  //=========================================================================
  if (run_3d) {
    // 3D defaults: R=49, 85% confluence
    if (radius == 20.0f) {
      radius = 49.0f; // Default radius for 3D
    }
    
    // Auto-compute domain size for target confluence if not explicitly set
    if (!domain_size_set) {
      // confluence = (n_cells * (4/3)*pi*R^3) / N^3
      // N^3 = n_cells * (4/3)*pi*R^3 / confluence
      // N = cbrt(n_cells * (4/3)*pi*R^3 / confluence)
      float cell_volume = (4.0f / 3.0f) * M_PI * radius * radius * radius;
      float total_cell_volume = num_cells * cell_volume;
      float domain_volume = total_cell_volume / confluence;
      int N = static_cast<int>(ceilf(cbrtf(domain_volume)));
      params.Nx = N;
      params.Ny = N;
      printf("Auto-computed domain size N=%d for %d cells, R=%.0f, confluence=%.0f%%\n",
             N, num_cells, radius, confluence * 100.0f);
    }
    
    // Create 3D parameters
    SimParams3D params3d;
    params3d.Nx = params.Nx;
    params3d.Ny = params.Ny;
    params3d.Nz = (Nz > 0) ? Nz : params.Nx; // Use Nz or default to Nx
    params3d.dx = params.dx;
    params3d.dy = params.dy;
    params3d.dz = params.dx; // Same as dx
    params3d.dt = params.dt;
    params3d.t_end = params.t_end;
    params3d.lambda = params.lambda;
    params3d.gamma = params.gamma;
    params3d.kappa = params.kappa;
    params3d.target_radius = radius;
    params3d.v_A = no_self_propulsion
                       ? 0.0f
                       : (v_A_override >= 0.0f ? v_A_override : params.v_A);
    params3d.xi = params.xi;
    params3d.tau = params.tau;
    params3d.subdomain_padding = params.subdomain_padding;
    params3d.motility_model = use_abp ? SimParams::MotilityModel::ABP
                                      : SimParams::MotilityModel::RunAndTumble;

    printf("3D Simulation Parameters:\n");
    printf("  Domain: %d x %d x %d\n", params3d.Nx, params3d.Ny, params3d.Nz);
    printf("  Grid spacing: dx=%.3f, dy=%.3f, dz=%.3f\n", params3d.dx,
           params3d.dy, params3d.dz);
    printf("  Time step: dt=%.4f\n", params3d.dt);
    printf("  End time: t_end=%.1f\n", params3d.t_end);
    printf("  Interface width: lambda=%.3f\n", params3d.lambda);
    printf("  Gradient coeff: gamma=%.3f\n", params3d.gamma);
    printf("  Bulk coeff (30/λ²): %.3f\n", params3d.bulk_coeff());
    printf("  Interaction coeff (30κ/λ²): %.3f\n",
           params3d.interaction_coeff());
    printf("  Target radius: R=%.1f (volume=%.1f)\n", params3d.target_radius,
           params3d.target_volume());
    printf("  Volume constraint: mu=%.3f (coeff=%.6f)\n", params3d.mu,
           params3d.volume_coeff());
    printf("  Active velocity: v_A=%.4f\n", params3d.v_A);
    printf("  Motility model: %s\n",
           params3d.motility_model == SimParams::MotilityModel::ABP
               ? "ABP (Active Brownian Particle)"
               : "Run-and-Tumble");
    printf("  Cells: %d\n", num_cells);
    printf("\n");

    // Estimate memory usage for 3D
    size_t voxels_per_cell =
        static_cast<size_t>(2 * radius * params3d.subdomain_padding);
    voxels_per_cell = voxels_per_cell * voxels_per_cell * voxels_per_cell;
    size_t total_bytes = voxels_per_cell * num_cells * sizeof(float) *
                         4; // phi, rhs, work buffers
    printf("Estimated GPU memory: %.1f MB per cell, %.1f MB total\n",
           (voxels_per_cell * sizeof(float) * 4) / (1024.0 * 1024.0),
           total_bytes / (1024.0 * 1024.0));

    if (safe_mode && total_bytes > 1ULL * 1024 * 1024 * 1024) {
      printf("SAFE MODE: Estimated memory %.2f GB exceeds 1GB limit\n",
             total_bytes / (1024.0 * 1024.0 * 1024.0));
      printf("Consider reducing domain size (-N), cell count (-n), or radius "
             "(-r)\n");
      return 1;
    }
    printf("\n");

    // Create 3D simulation
    Simulation3D sim3d(params3d);
    sim3d.output_dir = output_dir;
    sim3d.save_interval = save_interval;
    sim3d.trajectory_interval =
        (trajectory_interval > 0) ? trajectory_interval : save_interval;
    sim3d.save_individual_fields_flag = save_individual_fields;

    // Initialize or load checkpoint
    bool resumed = false;
    if (!checkpoint_file.empty()) {
      // SAFE MODE: Scan checkpoint BEFORE loading to check memory requirements
      if (safe_mode) {
        int checkpoint_num_cells = 0;
        size_t required_bytes = scan_checkpoint_3d_memory(
            checkpoint_file.c_str(), checkpoint_num_cells);
        if (required_bytes == 0) {
          printf("SAFE MODE: Failed to scan checkpoint file\n");
          return 1;
        }
        printf("SAFE MODE: Checkpoint requires %.2f MB for %d cells\n",
               required_bytes / (1024.0 * 1024.0), checkpoint_num_cells);
        if (required_bytes > 1ULL * 1024 * 1024 * 1024) {
          printf("SAFE MODE: Required memory %.2f GB exceeds 1GB limit\n",
                 required_bytes / (1024.0 * 1024.0 * 1024.0));
          printf("The checkpoint has too many/large cells for safe mode.\n");
          printf("Aborting before GPU memory allocation.\n");
          return 1;
        }
      }

      if (sim3d.load_checkpoint(checkpoint_file.c_str())) {
        resumed = true;
        printf("Resumed 3D from checkpoint: step=%d, t=%.4f\n",
               sim3d.current_step, sim3d.current_time);
      } else {
        printf("Warning: Could not load 3D checkpoint, starting fresh\n");
      }
    }

    if (!resumed) {
      if (use_grid_init) {
        // Grid-based initialization for target confluence
        sim3d.initialize_grid(num_cells, radius, confluence);
      } else {
        // Random placement mode
        if (min_spacing < 0) {
          // Auto-calculate min_spacing based on domain and cell count
          float domain_volume =
              static_cast<float>(params3d.Nx * params3d.Ny * params3d.Nz);
          float cell_volume = (4.0f / 3.0f) * M_PI * radius * radius * radius;
          float total_cell_volume = num_cells * cell_volume;
          float actual_confluence = total_cell_volume / domain_volume;

          // Calculate ideal center-to-center spacing from volume per cell
          float volume_per_cell = domain_volume / num_cells;
          float ideal_spacing = cbrtf(volume_per_cell);

          // min_spacing = gap between cell surfaces = center_spacing - 2*radius
          min_spacing = ideal_spacing - 2.0f * radius;

          printf("Random init: confluence=%.1f%%, ideal_spacing=%.1f, "
                 "diameter=%.0f\n",
                 actual_confluence * 100.0f, ideal_spacing, 2.0f * radius);
          printf("  min_spacing=%.1f (%s)\n", min_spacing,
                 min_spacing < 0 ? "cells overlap" : "cells separated");
        }
        sim3d.initialize_random(num_cells, radius, min_spacing);
      }

      // SAFE MODE: Check memory after initialization
      if (safe_mode) {
        size_t actual_bytes = sim3d.domain.total_gpu_memory_bytes();
        printf("Actual GPU memory usage: %.2f MB (%d cells)\n",
               actual_bytes / (1024.0 * 1024.0), sim3d.domain.num_cells());
        if (actual_bytes > 1ULL * 1024 * 1024 * 1024) {
          printf("SAFE MODE: Actual memory %.2f GB exceeds 1GB limit\n",
                 actual_bytes / (1024.0 * 1024.0 * 1024.0));
          printf("Consider reducing domain size (-N), cell count (-n), or "
                 "radius (-r)\n");
          return 1;
        }
      }
    }

    // Run 3D simulation
    auto start_time = std::chrono::high_resolution_clock::now();
    sim3d.run(params3d.t_end);
    auto end_time = std::chrono::high_resolution_clock::now();

    double elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    if (save_final_checkpoint) {
      sim3d.save_checkpoint();
    }

    printf("\n3D Simulation finished successfully!\n");
    printf("Total wall time: %.3f seconds\n", elapsed_seconds);
    printf("Final state: step=%d, t=%.4f\n", sim3d.current_step,
           sim3d.current_time);
    printf("Output saved to: %s\n", output_dir.c_str());
    return 0;
  }

  //=========================================================================
  // 2D Simulation (original code)
  //=========================================================================

  // Print simulation parameters
  printf("Simulation Parameters:\n");
  printf("  Domain: %d x %d\n", params.Nx, params.Ny);
  printf("  Grid spacing: dx=%.3f, dy=%.3f\n", params.dx, params.dy);
  printf("  Time step: dt=%.4f\n", params.dt);
  printf("  End time: t_end=%.1f\n", params.t_end);
  printf("  Interface width: lambda=%.3f\n", params.lambda);
  printf("  Gradient coeff: gamma=%.3f\n", params.gamma);
  printf("  Bulk coeff (30/λ²): %.3f\n", params.bulk_coeff());
  printf("  Interaction coeff (30κ/λ²): %.3f\n", params.interaction_coeff());
  printf("  Target radius: R=%.1f (area=%.1f)\n", params.target_radius,
         params.target_area());
  printf("  Volume constraint: mu=%.3f (coeff=%.6f)\n", params.mu,
         params.volume_coeff());
  printf("  Active velocity: v_A=%.4f\n", params.v_A);
  printf("  Motility model: %s\n",
         params.motility_model == SimParams::MotilityModel::ABP
             ? "ABP (Active Brownian Particle)"
             : "Run-and-Tumble");
  printf("  Cells: %d\n", num_cells);
  printf("\n");

  // Create simulation
  Simulation sim(params);
  sim.output_dir = output_dir;
  sim.save_interval = save_interval;
  sim.checkpoint_interval = checkpoint_interval;
  sim.trajectory_samples = trajectory_samples;
  // Use save_interval as default for trajectory (same as 3D)
  sim.trajectory_interval =
      (trajectory_interval > 0) ? trajectory_interval
      : (trajectory_interval == -1)
          ? save_interval
          : trajectory_interval; // 0 = compute from samples
  sim.compute_diagnostics = use_diagnostics;
  sim.save_vtk = (save_interval > 0);
  sim.save_individual_fields = save_individual_fields;

  // Track whether runtime options were explicitly set on command line
  bool save_interval_set = false;
  bool checkpoint_interval_set = false;
  bool trajectory_samples_set = false;

  // Re-parse to detect explicit settings (bit of a hack, but simple)
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--save-interval")
      save_interval_set = true;
    else if (arg == "--checkpoint-interval")
      checkpoint_interval_set = true;
    else if (arg == "--trajectory-samples")
      trajectory_samples_set = true;
  }

  // Initialize
  bool resumed = false;
  if (!checkpoint_file.empty()) {
    // Save command-line overrides before loading checkpoint (which overwrites
    // params)
    float cmd_t_end = params.t_end;
    float cmd_v_A = params.v_A; // Save v_A in case user overrode it
    int cmd_save_interval = save_interval;
    int cmd_checkpoint_interval = checkpoint_interval;
    int cmd_trajectory_samples = trajectory_samples;

    printf("DEBUG: About to load checkpoint from: %s\n",
           checkpoint_file.c_str());
    fflush(stdout);

    if (sim.initialize_from_checkpoint(checkpoint_file)) {
      printf("DEBUG: Checkpoint loaded successfully\n");
      fflush(stdout);
      resumed = true;

      // Safe mode check: limit domain size to prevent runaway GPU memory
      if (safe_mode) {
        size_t domain_pixels =
            (size_t)sim.domain.params.Nx * (size_t)sim.domain.params.Ny;
        size_t estimated_bytes = domain_pixels * sizeof(float) *
                                 sim.domain.num_cells() * 2; // rough estimate
        const size_t MAX_BYTES = 1ULL * 1024 * 1024 * 1024;  // 1 GB
        if (estimated_bytes > MAX_BYTES) {
          printf("SAFE MODE: Estimated memory %.2f GB exceeds 1GB limit\n",
                 estimated_bytes / (1024.0 * 1024.0 * 1024.0));
          printf("  Domain: %d x %d, Cells: %d\n", sim.domain.params.Nx,
                 sim.domain.params.Ny, sim.domain.num_cells());
          printf("  Use --safe-mode to disable this check if you're sure\n");
          return 1;
        }
        printf("SAFE MODE: Estimated memory %.2f MB (OK)\n",
               estimated_bytes / (1024.0 * 1024.0));
      }

      // Restore command-line t_end (checkpoint should not override target end
      // time)
      sim.domain.params.t_end = cmd_t_end;

      // Restore v_A if user explicitly overrode it
      if (no_self_propulsion) {
        sim.domain.params.v_A = 0.0f;
      } else if (v_A_override >= 0.0f) {
        sim.domain.params.v_A = v_A_override;
      }

      // Apply command-line overrides for runtime options if specified
      if (save_interval_set) {
        sim.save_interval = cmd_save_interval;
        sim.save_vtk = (cmd_save_interval > 0);
      }
      if (checkpoint_interval_set) {
        sim.checkpoint_interval = cmd_checkpoint_interval;
      }
      if (trajectory_samples_set) {
        sim.trajectory_samples = cmd_trajectory_samples;
      }
      if (use_diagnostics) {
        sim.compute_diagnostics = true;
      }

      printf("Resumed from checkpoint: step=%d, t=%.4f, target t_end=%.4f\n",
             sim.current_step, sim.current_time, cmd_t_end);
      printf("  Domain: %d x %d, R=%.1f, cells=%d\n", sim.domain.params.Nx,
             sim.domain.params.Ny, sim.domain.params.target_radius,
             sim.domain.num_cells());
      printf("  Using: v_A=%.4f, save_interval=%d, checkpoint_interval=%d, "
             "trajectory_samples=%d\n",
             sim.domain.params.v_A, sim.save_interval, sim.checkpoint_interval,
             sim.trajectory_samples);

      // Check if we've already reached the target time
      if (sim.current_time >= cmd_t_end) {
        printf("Simulation already complete (t=%.4f >= t_end=%.4f)\n",
               sim.current_time, cmd_t_end);
        return 0;
      }
    } else {
      printf("Warning: Could not load checkpoint, starting fresh\n");
    }
  }

  if (!resumed) {
    if (edge_test) {
      sim.initialize_edge_test(radius);
    } else if (corner_push_test) {
      sim.initialize_corner_push_test(num_cells, radius);
    } else {
      // Auto-calculate min_spacing if not specified
      if (min_spacing < 0) {
        // Calculate ideal spacing based on available area per cell
        float domain_area = static_cast<float>(params.Nx * params.Ny);
        float area_per_cell = domain_area / num_cells;
        // Ideal spacing is sqrt(area_per_cell) but at least 2*radius
        float ideal_spacing = sqrtf(area_per_cell);
        min_spacing = fmaxf(2.0f * radius, ideal_spacing * 0.8f);
        printf("Auto min_spacing: %.1f (ideal: %.1f, 2R: %.1f)\n", min_spacing,
               ideal_spacing, 2.0f * radius);
      }
      sim.initialize_random(num_cells, radius, min_spacing);
    }
  }

  // Run simulation with timing
  auto start_time = std::chrono::high_resolution_clock::now();
  sim.run();
  auto end_time = std::chrono::high_resolution_clock::now();

  double elapsed_seconds =
      std::chrono::duration<double>(end_time - start_time).count();

  // Save final checkpoint for job chaining
  if (save_final_checkpoint) {
    std::string final_checkpoint = output_dir + "/checkpoint.bin";
    sim.save_current_checkpoint(final_checkpoint);
  }

  printf("\nSimulation finished successfully!\n");
  printf("Total wall time: %.3f seconds\n", elapsed_seconds);
  printf("Final state: step=%d, t=%.4f\n", sim.current_step, sim.current_time);
  printf("Output saved to: %s\n", output_dir.c_str());

  return 0;
}
