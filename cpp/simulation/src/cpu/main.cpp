/**
 * Cell Simulation - CPU Backend
 *
 * Phase-field model for multi-cellular systems with:
 * - Volume constraint (mu)
 * - Cell-cell repulsion (kappa)
 * - Active motility (Run-and-Tumble)
 *
 * Supports both serial and MPI parallel execution.
 */

#include "domain.hpp"
#include "integrator.hpp"
#include "io.hpp"
#include "types.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

void print_usage(const char *prog) {
  std::cout << "Usage: " << prog << " [options]\n"
            << "\nSimulation parameters:\n"
            << "  -n <int>             Number of cells (default: 8)\n"
            << "  -N <int>             Domain size NxN (default: 512)\n"
            << "  -r <float>           Target cell radius (default: 49.0)\n"
            << "  --confluence <float> Target confluence [0-1] (auto-computes N from -n)\n"
            << "  -t <float>           Total simulation time (default: 10.0)\n"
            << "  --dt <float>         Time step (default: 0.01)\n"
            << "  --save-interval <int> Save every N steps (default: 100)\n"
            << "  -o <str>             Output directory (default: ./output)\n"
            << "  -c <str>             Load from checkpoint file\n"
            << "  -i <str>             Load initial conditions from JSON file\n"
            << "  --lambda <float>     Interface width (default: 7.0)\n"
            << "  --gamma <float>      Surface tension (default: 1.0)\n"
            << "  --kappa <float>      Repulsion strength (default: 10.0)\n"
            << "  --mu <float>         Volume constraint (default: 1.0)\n"
            << "  --motility <float>   Motility speed (default: 0.0)\n"
            << "  --tumble-time <float> Mean tumble time (default: 100.0)\n"
            << "  --seed <int>         Random seed (default: 42)\n"
            << "  --help, -h           Print this help message\n";
}

int main(int argc, char **argv) {
  int mpi_rank = 0;
  int mpi_size = 1;

#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  // Default parameters
  cellsim::SimParams params;
  params.Nx = 512;
  params.Ny = 512;
  params.dx = 1.0f;
  params.dy = 1.0f;
  params.dt = 0.01f;
  params.lambda = 7.0f;
  params.gamma = 1.0f;
  params.kappa = 10.0f;
  params.mu = 1.0f;
  params.target_radius = 49.0f;
  params.v_A = 0.0f;
  params.tau = 100.0f;

  int num_cells = 8;
  float t_end = 10.0f;  // Total simulation time
  int save_interval = 100;
  std::string output_dir = "./output";
  std::string checkpoint_file = "";
  std::string init_file = "";  // JSON initial conditions file
  unsigned int seed = 42;
  float target_confluence = -1.0f; // Negative means not set
  bool domain_size_set = false;    // Track if -N was explicitly set
  bool num_cells_set = false;      // Track if -n was explicitly set

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      if (mpi_rank == 0) {
        print_usage(argv[0]);
      }
#ifdef USE_MPI
      MPI_Finalize();
#endif
      return 0;
    } else if (arg == "-n" && i + 1 < argc) {
      num_cells = std::atoi(argv[++i]);
      num_cells_set = true;
    } else if (arg == "-N" && i + 1 < argc) {
      int size = std::atoi(argv[++i]);
      params.Nx = size;
      params.Ny = size;
      domain_size_set = true;
    } else if (arg == "-r" && i + 1 < argc) {
      params.target_radius = std::atof(argv[++i]);
    } else if ((arg == "-t" || arg == "-T") && i + 1 < argc) {
      t_end = std::atof(argv[++i]);
    } else if (arg == "--dt" && i + 1 < argc) {
      params.dt = std::atof(argv[++i]);
    } else if (arg == "--lambda" && i + 1 < argc) {
      params.lambda = std::atof(argv[++i]);
    } else if (arg == "--gamma" && i + 1 < argc) {
      params.gamma = std::atof(argv[++i]);
    } else if (arg == "--kappa" && i + 1 < argc) {
      params.kappa = std::atof(argv[++i]);
    } else if (arg == "--mu" && i + 1 < argc) {
      params.mu = std::atof(argv[++i]);
    } else if (arg == "--confluence" && i + 1 < argc) {
      target_confluence = std::atof(argv[++i]);
    } else if (arg == "--save-interval" && i + 1 < argc) {
      save_interval = std::atoi(argv[++i]);
    } else if (arg == "-o" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "-c" && i + 1 < argc) {
      checkpoint_file = argv[++i];
    } else if (arg == "-i" && i + 1 < argc) {
      init_file = argv[++i];
    } else if (arg == "--motility" && i + 1 < argc) {
      params.v_A = std::atof(argv[++i]);
    } else if (arg == "--tumble-time" && i + 1 < argc) {
      params.tau = std::atof(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = static_cast<unsigned int>(std::atoi(argv[++i]));
    }
  }

  // Create output directory
  if (mpi_rank == 0) {
    std::filesystem::create_directories(output_dir);
  }

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Handle confluence parameter
  // If both -n and --confluence are set, compute domain size from them
  // If only --confluence is set, compute num_cells from domain size
  if (target_confluence > 0.0f) {
    float R = params.target_radius;
    float cell_area = 3.14159265f * R * R;

    if (num_cells_set && !domain_size_set) {
      // Compute domain size for target confluence
      // confluence = (num_cells * pi * R^2) / N^2
      // N^2 = num_cells * pi * R^2 / confluence
      // N = sqrt(num_cells * pi * R^2 / confluence)
      float total_cell_area = num_cells * cell_area;
      float domain_area = total_cell_area / target_confluence;
      int N = static_cast<int>(std::ceil(std::sqrt(domain_area)));
      params.Nx = N;
      params.Ny = N;
      if (mpi_rank == 0) {
        std::cout << "Auto-computed domain size N=" << N << " for " << num_cells
                  << " cells, R=" << R << ", confluence=" 
                  << (target_confluence * 100.0f) << "%" << std::endl;
      }
    } else if (!num_cells_set) {
      // Compute num_cells from confluence and domain size
      float domain_area = params.Nx * params.dx * params.Ny * params.dy;
      num_cells = static_cast<int>(target_confluence * domain_area / cell_area);
      num_cells = std::max(1, num_cells);
      if (mpi_rank == 0) {
        std::cout << "Computed " << num_cells << " cells for confluence="
                  << (target_confluence * 100.0f) << "% in " << params.Nx 
                  << "x" << params.Ny << " domain" << std::endl;
      }
    }
  }

  // Compute number of steps from t_end and dt
  int num_steps = static_cast<int>(std::ceil(t_end / params.dt));

  // Initialize domain
  cellsim::Domain domain;
  domain.initialize(params);

  int start_step = 0;

  if (!checkpoint_file.empty()) {
    // Load from checkpoint
    if (!cellsim::load_checkpoint(domain, start_step, checkpoint_file)) {
      if (mpi_rank == 0) {
        std::cerr << "Error: Failed to load checkpoint" << std::endl;
      }
#ifdef USE_MPI
      MPI_Finalize();
#endif
      return 1;
    }
  } else if (!init_file.empty()) {
    // Load from JSON initial conditions file
    if (!cellsim::load_initial_conditions_json(domain, init_file)) {
      if (mpi_rank == 0) {
        std::cerr << "Error: Failed to load initial conditions from " << init_file << std::endl;
      }
#ifdef USE_MPI
      MPI_Finalize();
#endif
      return 1;
    }
  } else {
    // Place cells randomly
    try {
      domain.place_cells_random(num_cells, target_confluence, seed);
    } catch (const std::exception& e) {
      if (mpi_rank == 0) {
        std::cerr << "Error: " << e.what() << std::endl;
      }
#ifdef USE_MPI
      MPI_Finalize();
#endif
      return 1;
    }
  }

  // Print configuration
  if (mpi_rank == 0) {
    const auto& dom_params = domain.get_params();
    std::cout << "=== Cell Simulation - CPU Backend ===" << std::endl;
#ifdef USE_MPI
    std::cout << "MPI Enabled: " << mpi_size << " processes" << std::endl;
#else
    std::cout << "Serial execution (no MPI)" << std::endl;
#endif
    std::cout << "Domain: " << dom_params.Nx << " x " << dom_params.Ny << std::endl;
    std::cout << "Grid spacing: dx=" << dom_params.dx << ", dy=" << dom_params.dy
              << std::endl;
    std::cout << "Time step: dt=" << dom_params.dt << std::endl;
    std::cout << "Parameters: λ=" << dom_params.lambda << ", γ=" << dom_params.gamma
              << ", κ=" << dom_params.kappa << ", μ=" << dom_params.mu << std::endl;
    std::cout << "Cell radius: R=" << dom_params.target_radius << std::endl;
    std::cout << "Number of cells: " << domain.get_num_cells() << std::endl;
    std::cout << "Motility: v=" << dom_params.v_A
              << ", τ=" << dom_params.tau << std::endl;
    std::cout << "Steps: " << num_steps << " (output every " << save_interval
              << ")" << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    std::cout << "======================================" << std::endl;
  }

  // Initialize integrator
  cellsim::Integrator integrator;
  integrator.initialize(domain, seed);
  integrator.set_current_step(start_step);

  // Save initial state
  if (mpi_rank == 0 && start_step == 0) {
    cellsim::save_vtk(domain, 0, output_dir);
    cellsim::save_checkpoint(domain, 0, output_dir);
  }

  // Run simulation
  integrator.run(domain, num_steps, save_interval, output_dir);

  // Print final statistics
  if (mpi_rank == 0) {
    double total_time = integrator.get_total_time();
    std::cout << "\n=== Simulation Complete ===" << std::endl;
    std::cout << "Total time: " << total_time << " s" << std::endl;
    std::cout << "Average speed: " << num_steps / total_time << " steps/s"
              << std::endl;
    std::cout << "===========================" << std::endl;
  }

#ifdef USE_MPI
  MPI_Finalize();
#endif

  return 0;
}
