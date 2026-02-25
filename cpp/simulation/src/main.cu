#include "simulation.cuh"
#include "simulation3d.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

using namespace cellsim;

//=============================================================================
// JSON Initial Conditions Loading (simplified for NVCC compatibility)
//=============================================================================

// Load initial conditions from JSON file
// Returns: number of cells loaded, or -1 on error
static int load_initial_conditions_json(const char *filename, 
                                        SimParams &params,
                                        std::vector<float> &cx_out,
                                        std::vector<float> &cy_out) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    printf("Error: Could not open JSON file: %s\n", filename);
    return -1;
  }
  
  // Read entire file
  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  
  char *json = (char*)malloc(file_size + 1);
  fread(json, 1, file_size, f);
  json[file_size] = '\0';
  fclose(f);
  
  // Simple parsing using strstr
  const char *p;
  
  // Extract Nx
  p = strstr(json, "\"Nx\"");
  if (p) {
    p = strchr(p, ':');
    if (p) params.Nx = atoi(p + 1);
  }
  
  // Extract Ny  
  p = strstr(json, "\"Ny\"");
  if (p) {
    p = strchr(p, ':');
    if (p) params.Ny = atoi(p + 1);
  }
  
  // Extract target_radius
  p = strstr(json, "\"target_radius\"");
  if (p) {
    p = strchr(p, ':');
    if (p) params.target_radius = (float)atof(p + 1);
  }
  
  // Extract lambda
  p = strstr(json, "\"lambda\"");
  if (p) {
    p = strchr(p, ':');
    if (p) params.lambda = (float)atof(p + 1);
  }
  
  // Extract kappa
  p = strstr(json, "\"kappa\"");
  if (p) {
    p = strchr(p, ':');
    if (p) params.kappa = (float)atof(p + 1);
  }
  
  // Extract v_A
  p = strstr(json, "\"v_A\"");
  if (p) {
    p = strchr(p, ':');
    if (p) params.v_A = (float)atof(p + 1);
  }
  
  // Extract tau
  p = strstr(json, "\"tau\"");
  if (p) {
    p = strchr(p, ':');
    if (p) params.tau = (float)atof(p + 1);
  }
  
  // Find cells array and extract cx/cy pairs
  p = strstr(json, "\"cells\"");
  if (!p) {
    printf("Error: No 'cells' array found in JSON\n");
    free(json);
    return -1;
  }
  
  // Find all cx/cy pairs
  const char *search = p;
  while ((p = strstr(search, "\"cx\"")) != NULL) {
    const char *colon = strchr(p, ':');
    if (!colon) break;
    float cx = (float)atof(colon + 1);
    
    // Find cy after cx
    const char *cy_pos = strstr(p, "\"cy\"");
    if (!cy_pos) break;
    colon = strchr(cy_pos, ':');
    if (!colon) break;
    float cy = (float)atof(colon + 1);
    
    cx_out.push_back(cx);
    cy_out.push_back(cy);
    
    search = cy_pos + 1;
  }
  
  free(json);
  
  int num_cells = (int)cx_out.size();
  printf("Loaded %d cells from JSON: %s\n", num_cells, filename);
  printf("  Domain: %d x %d\n", params.Nx, params.Ny);
  printf("  Target radius: %.1f\n", params.target_radius);
  
  return num_cells;
}

// Parse a --gamma argument: either bare "V" or "V:selector"
// selector is "N%" for fraction or "cellN" / "cellN,M,..." for specific cells.
// Returns true on success, false on parse error.
static bool parse_gamma_arg(const char *arg, cellsim::SimParams &params,
                           std::vector<cellsim::SimParams::GammaOverride> &overrides,
                           bool &overrides_set) {
  std::string s(arg);
  auto colon = s.find(':');
  if (colon == std::string::npos) {
    // Bare value: set base gamma
    params.gamma = static_cast<float>(atof(arg));
    return true;
  }
  // Value:Selector
  float value = static_cast<float>(atof(s.substr(0, colon).c_str()));
  std::string selector = s.substr(colon + 1);
  if (selector.empty()) {
    fprintf(stderr, "Error: --gamma %s has empty selector after ':'\n", arg);
    return false;
  }
  cellsim::SimParams::GammaOverride ov;
  ov.value = value;
  ov.fraction = 0.0f;
  if (selector.back() == '%') {
    // Fraction selector: "20%"
    ov.type = cellsim::SimParams::GammaOverride::Type::Fraction;
    ov.fraction = static_cast<float>(atof(selector.c_str())) / 100.0f;
    if (ov.fraction <= 0.0f || ov.fraction > 1.0f) {
      fprintf(stderr, "Error: --gamma %s fraction must be 0-100%%\n", arg);
      return false;
    }
  } else if (selector.substr(0, 4) == "cell") {
    // Cell selector: "cell0" or "cell0,5,12"
    ov.type = cellsim::SimParams::GammaOverride::Type::Cells;
    std::string ids_str = selector.substr(4);
    // Split on commas
    size_t pos = 0;
    while (pos < ids_str.size()) {
      size_t comma = ids_str.find(',', pos);
      if (comma == std::string::npos) comma = ids_str.size();
      int id = atoi(ids_str.substr(pos, comma - pos).c_str());
      ov.cell_ids.push_back(id);
      pos = comma + 1;
    }
    if (ov.cell_ids.empty()) {
      fprintf(stderr, "Error: --gamma %s has no cell IDs after 'cell'\n", arg);
      return false;
    }
  } else {
    fprintf(stderr, "Error: --gamma %s unknown selector '%s' (use N%% or cellN)\n",
            arg, selector.c_str());
    return false;
  }
  overrides.push_back(ov);
  overrides_set = true;
  return true;
}

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
  printf("  -i <file>     Load initial conditions from JSON file\n");
  printf(
      "  --edge-test   Place 3 cells at edges/corners for boundary testing\n");
  printf("  --corner-push-test  Stress test: corner cell + clustered cells "
         "pushing it\n");
  printf("  --no-self-propulsion  Disable active self-propulsion (v_A = 0)\n");
  printf("  --use-diagnostics     Enable volume/shape computation (disabled by "
         "default for speed)\n");
  printf("  --save-interval <n>   Steps between VTK saves (0 = no saves, "
         "default: 100)\n");
  printf("  --print-interval <n>  Steps between progress output (-1 = use "
         "save_interval, default: -1)\n");
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
  printf("  --observable-interval <n>  Steps between GPU diagnostic measurements "
         "(energy, stress, contacts; requires -DENABLE_DIAGNOSTICS=ON)\n");
  printf("  --stress-fields  Include stress tensor fields (σ_xx, σ_yy, σ_xy, P) "
         "in VTK output (requires -DENABLE_STRESS_FIELDS=ON)\n");
  printf("  --v-A <f>     Active motility velocity (default: from params)\n");
  printf("  --tau <f>     Reorientation time (default: 10000)\n");
  printf("  --gamma <f[:selector]>  Stiffness / gradient coefficient (default: 1.0).\n"
         "                          Repeat with selector for heterogeneous populations:\n"
         "                            --gamma 1.0           Base value for all cells\n"
         "                            --gamma 0.35:20%%      20%% of cells get gamma=0.35\n"
         "                            --gamma 0.35:cell0     Cell 0 gets gamma=0.35\n"
         "                            --gamma 0.35:cell0,5   Cells 0 and 5 get gamma=0.35\n"
         "                          Processing order: base -> fractions (random) -> cells\n");
  printf("  --soft-cell <id>  [Deprecated: use --gamma V:cellN] Make cell <id> soft\n");
  printf("  --gamma-soft <f>  [Deprecated: use --gamma V:cellN] Stiffness for soft cell (default: 0.35)\n");
  printf("  --kappa <f>   Interaction strength (default: 10.0)\n");
  printf("  --mu <f>      Volume constraint strength (default: 1.0)\n");
  printf("  --xi <f>      Friction coefficient (default: 1500)\n");
  printf("  --abp         Use Active Brownian Particle model instead of "
         "Run-and-Tumble\n");
  printf("  --adhesion <J>  Adhesion strength J (default: 0 = off). "
         "Adds -J*Σφ_j attraction.\n");
  printf("  --save-individual-fields  Save individual cell fields for energy "
         "analysis\n");
  printf("  --grid        Use FCC lattice initialization (for high "
         "confluence)\n");
  printf("  --sc-grid     Use simple cubic grid instead of FCC lattice\n");
  printf("  --confluence <f>  Target confluence 0-1 (default: 0.85). Works with "
         "random or grid init\n");
  printf("  -h            Show this help\n");
}

int main(int argc, char *argv[]) {
  // Default parameters
  SimParams params;
  params.Nx = 256;
  params.Ny = 256;
  params.dt = 0.02f;
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
  std::string init_file = "";  // JSON initial conditions file
  bool edge_test = false;
  bool corner_push_test = false;
  bool no_self_propulsion = false;
  bool use_diagnostics = false;
  bool save_final_checkpoint = false;
  bool save_individual_fields =
      false; // Save individual cell fields for energy analysis
  int save_interval = 100;
  int print_interval = -1; // -1 means use save_interval
  int checkpoint_interval = -1; // -1 means use save_interval * 10
  int random_seed = -1;         // -1 means use time-based seed
  int trajectory_samples = 100; // Number of trajectory data points to save
  int trajectory_interval =
      -1; // -1 = use save_interval, 0 = compute from samples, >0 = explicit
  int observable_interval = 0; // 0 = disabled, >0 = GPU diagnostic measurements
  bool stress_fields = false;  // Include stress tensor fields in VTK output
  float v_A_override = -1.0f; // -1 means use default from params
  float tau_override = -1.0f;  // -1 means use default from params
  bool use_abp = false;       // Use ABP model instead of Run-and-Tumble
  bool safe_mode = false;   // Limit memory allocation to 1GB
  bool use_grid_init = false; // Use grid-based initialization instead of random
  bool use_fcc = true;        // Use FCC lattice (default) vs simple cubic
  float confluence = 0.85f;   // Target confluence for grid initialization
  bool subdomain_padding_set = false; // Track if user explicitly set padding
  std::vector<cellsim::SimParams::GammaOverride> gamma_overrides;
  bool gamma_overrides_set = false;

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
    } else if ((arg == "-c" || arg == "--load") && i + 1 < argc) {
      checkpoint_file = argv[++i];
    } else if (arg == "-i" && i + 1 < argc) {
      init_file = argv[++i];
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
    } else if (arg == "--print-interval" && i + 1 < argc) {
      print_interval = atoi(argv[++i]);
    } else if (arg == "--subdomain-padding" && i + 1 < argc) {
      params.subdomain_padding = atof(argv[++i]);
      subdomain_padding_set = true;
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
    } else if (arg == "--observable-interval" && i + 1 < argc) {
      observable_interval = atoi(argv[++i]);
    } else if (arg == "--stress-fields") {
      stress_fields = true;
    } else if (arg == "--v-A" && i + 1 < argc) {
      v_A_override = atof(argv[++i]);
    } else if (arg == "--v-A-sigma" && i + 1 < argc) {
      params.v_A_sigma = atof(argv[++i]);
    } else if (arg == "--tau" && i + 1 < argc) {
      tau_override = atof(argv[++i]);
    } else if (arg == "--abp") {
      use_abp = true;
    } else if (arg == "--gamma" && i + 1 < argc) {
      if (!parse_gamma_arg(argv[++i], params, gamma_overrides, gamma_overrides_set)) {
        return 1;
      }
    } else if (arg == "--soft-cell" && i + 1 < argc) {
      // Deprecated: convert to gamma_override internally
      params.soft_cell_id = atoi(argv[++i]);
    } else if (arg == "--gamma-soft" && i + 1 < argc) {
      // Deprecated: will be combined with --soft-cell after arg parsing
      params.gamma_soft = atof(argv[++i]);
    } else if (arg == "--kappa" && i + 1 < argc) {
      params.kappa = atof(argv[++i]);
    } else if (arg == "--mu" && i + 1 < argc) {
      params.mu = atof(argv[++i]);
    } else if (arg == "--xi" && i + 1 < argc) {
      params.xi = atof(argv[++i]);
    } else if (arg == "--adhesion" && i + 1 < argc) {
      params.adhesion_J = atof(argv[++i]);
    } else if (arg == "--save-individual-fields") {
      save_individual_fields = true;
    } else if (arg == "--safe-mode") {
      // Limit memory allocation to prevent runaway GPU memory usage
      safe_mode = true;
    } else if (arg == "--grid") {
      // Use FCC grid-based initialization for high confluence
      use_grid_init = true;
    } else if (arg == "--sc-grid") {
      // Use simple cubic grid (legacy)
      use_grid_init = true;
      use_fcc = false;
    } else if (arg == "--confluence" && i + 1 < argc) {
      confluence = atof(argv[++i]);
      // --confluence no longer implies --grid; it just sets the target confluence
      // for domain size calculation. Use --grid or --sc-grid to select lattice init.
    } else if (arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "ERROR: Unrecognized argument '%s'\n", argv[i]);
      fprintf(stderr, "Use -h for usage information.\n");
      return 1;
    }
  }

  // gamma_overrides live outside SimParams (SimParams is raw-serialized in
  // checkpoints and can't contain std::vector).

  // Convert deprecated --soft-cell/--gamma-soft to gamma_overrides
  if (params.soft_cell_id >= 0 && !gamma_overrides_set) {
    cellsim::SimParams::GammaOverride ov;
    ov.value = params.gamma_soft;
    ov.type = cellsim::SimParams::GammaOverride::Type::Cells;
    ov.fraction = 0.0f;
    ov.cell_ids.push_back(params.soft_cell_id);
    gamma_overrides.push_back(ov);
    gamma_overrides_set = true;
    printf("Note: --soft-cell/--gamma-soft is deprecated. "
           "Use --gamma %.4f:cell%d instead.\n",
           params.gamma_soft, params.soft_cell_id);
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
  
  // Apply tau override
  if (tau_override > 0.0f) {
    params.tau = tau_override;
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
    params3d.motility_model = use_abp ? SimParams::MotilityModel::ABP
                                      : SimParams::MotilityModel::RunAndTumble;
    // Note: params3d.subdomain_padding uses its own default (1.4 for tight bbox)
    // Only override if user explicitly set it on command line
    if (subdomain_padding_set) {
      params3d.subdomain_padding = params.subdomain_padding;
    }

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
    printf("  Reorientation time: tau=%.1f\n", params3d.tau);
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
        if (use_fcc) {
          // FCC lattice — most spherical Voronoi cells
          sim3d.initialize_grid_fcc(num_cells, radius, confluence);
        } else {
          // Simple cubic grid (legacy)
          sim3d.initialize_grid(num_cells, radius, confluence);
        }
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

  // Load initial conditions from JSON if specified
  std::vector<float> init_cx, init_cy;
  if (!init_file.empty()) {
    int loaded_cells = load_initial_conditions_json(init_file.c_str(), params, init_cx, init_cy);
    if (loaded_cells < 0) {
      printf("Error: Failed to load initial conditions from %s\n", init_file.c_str());
      return 1;
    }
    num_cells = loaded_cells;
    radius = params.target_radius;
  }

  // Create simulation
  Simulation sim(params);
  sim.output_dir = output_dir;
  sim.save_interval = save_interval;
  // Pass gamma overrides (live on Simulation, not SimParams)
  if (gamma_overrides_set) {
    sim.gamma_overrides = gamma_overrides;
    sim.gamma_overrides_set = true;
  }
  sim.print_interval = print_interval;
  sim.checkpoint_interval = checkpoint_interval;
  sim.trajectory_samples = trajectory_samples;
  // Use save_interval as default for trajectory (same as 3D)
  sim.trajectory_interval =
      (trajectory_interval > 0) ? trajectory_interval
      : (trajectory_interval == -1)
          ? save_interval
          : trajectory_interval; // 0 = compute from samples
  sim.observable_interval = observable_interval;
#ifdef STRESS_FIELDS_ENABLED
  sim.save_stress_fields = stress_fields;
#else
  if (stress_fields) {
    printf("Warning: --stress-fields requires -DENABLE_STRESS_FIELDS=ON at build time\n");
  }
#endif
  sim.compute_diagnostics = use_diagnostics;
  sim.save_vtk = (save_interval > 0);
  sim.save_individual_fields = save_individual_fields;

  // Track whether runtime options were explicitly set on command line
  bool save_interval_set = false;
  bool checkpoint_interval_set = false;
  bool trajectory_samples_set = false;
  bool trajectory_interval_set = false;

  // Re-parse to detect explicit settings (bit of a hack, but simple)
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--save-interval")
      save_interval_set = true;
    else if (arg == "--checkpoint-interval")
      checkpoint_interval_set = true;
    else if (arg == "--trajectory-samples")
      trajectory_samples_set = true;
    else if (arg == "--trajectory-interval")
      trajectory_interval_set = true;
  }

  // Initialize
  bool resumed = false;
  if (!checkpoint_file.empty()) {
    // Save command-line overrides before loading checkpoint (which overwrites
    // params)
    float cmd_t_end = params.t_end;
    float cmd_v_A = params.v_A; // Save v_A in case user overrode it
    float cmd_adhesion_J = params.adhesion_J; // Save adhesion in case user overrode it
    int cmd_save_interval = save_interval;
    int cmd_checkpoint_interval = checkpoint_interval;
    int cmd_trajectory_samples = trajectory_samples;

    printf("Loading checkpoint: %s\n", checkpoint_file.c_str());
    fflush(stdout);

    if (sim.initialize_from_checkpoint(checkpoint_file)) {
      printf("Checkpoint loaded successfully.\n");
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

      // Restore adhesion_J from command line (checkpoint stores J=0 from
      // equilibration, but quench experiments need the CLI-specified value)
      if (cmd_adhesion_J > 0.0f) {
        sim.domain.params.adhesion_J = cmd_adhesion_J;
      }

      // Restore v_A if user explicitly overrode it
      if (no_self_propulsion) {
        sim.domain.params.v_A = 0.0f;
      } else if (v_A_override >= 0.0f) {
        sim.domain.params.v_A = v_A_override;
      }

      // If user specified --v-A or --v-A-sigma on command line, regenerate
      // per-cell v_A instead of using checkpoint values (needed when starting
      // production from an equilibration checkpoint with v_A=0)
      if (v_A_override >= 0.0f || params.v_A_sigma > 0.0f) {
        sim.loaded_v_A.clear();
        sim.domain.params.v_A_sigma = params.v_A_sigma;
        printf("  Per-cell v_A will be regenerated (--v-A or --v-A-sigma specified)\n");
      }

      // If user specified gamma overrides (or deprecated --soft-cell), regenerate per-cell gamma
      if (gamma_overrides_set) {
        sim.loaded_gamma.clear();
        sim.gamma_overrides = gamma_overrides;
        sim.gamma_overrides_set = true;
        printf("  Per-cell gamma will be regenerated (--gamma overrides specified)\n");
      }

      // Apply command-line overrides for runtime options if specified
      if (save_interval_set) {
        sim.save_interval = cmd_save_interval;
        sim.save_vtk = (cmd_save_interval > 0);
      }
      if (checkpoint_interval_set) {
        sim.checkpoint_interval = cmd_checkpoint_interval;
      }
      if (trajectory_interval_set) {
        sim.trajectory_interval = trajectory_interval;
      }
      if (trajectory_samples_set) {
        sim.trajectory_samples = cmd_trajectory_samples;
        // Reset trajectory_interval to "compute from samples" mode unless the
        // user also explicitly passed --trajectory-interval. Without this, the
        // pre-checkpoint trajectory_interval (= save_interval) would take
        // precedence over trajectory_samples in run().
        if (!trajectory_interval_set) {
          sim.trajectory_interval = 0;
        }
      }
      if (use_diagnostics) {
        sim.compute_diagnostics = true;
      }

      printf("Resumed from checkpoint: step=%d, t=%.4f, target t_end=%.4f\n",
             sim.current_step, sim.current_time, cmd_t_end);
      fflush(stdout);

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

  // Print effective simulation parameters AFTER checkpoint load + CLI overrides
  // (so the log always shows what the simulation is actually using)
  {
    const SimParams &p = sim.domain.params;
    printf("\nEffective Simulation Parameters:\n");
    printf("  Domain: %d x %d\n", p.Nx, p.Ny);
    printf("  Grid spacing: dx=%.3f, dy=%.3f\n", p.dx, p.dy);
    printf("  Time step: dt=%.4f\n", p.dt);
    printf("  End time: t_end=%.1f\n", p.t_end);
    printf("  Interface width: lambda=%.3f\n", p.lambda);
    printf("  Gradient coeff: gamma=%.3f\n", p.gamma);
    printf("  Bulk coeff (30/lambda^2): %.3f\n", p.bulk_coeff());
    printf("  Interaction coeff (30*kappa/lambda^2): %.3f\n", p.interaction_coeff());
    printf("  Target radius: R=%.1f (area=%.1f)\n", p.target_radius, p.target_area());
    printf("  Volume constraint: mu=%.3f (coeff=%.6f)\n", p.mu, p.volume_coeff());
    printf("  Active velocity: v_A=%.4f\n", p.v_A);
    if (p.v_A_sigma > 0.0f) {
      printf("  v_A disorder: sigma=%.4f (log-normal)\n", p.v_A_sigma);
    }
    printf("  Reorientation time: tau=%.1f\n", p.tau);
    printf("  Motility model: %s\n",
           p.motility_model == SimParams::MotilityModel::ABP
               ? "ABP (Active Brownian Particle)"
               : "Run-and-Tumble");
    if (p.adhesion_J > 0.0f) {
      printf("  Adhesion: J=%.4f (J/kappa=%.4f)\n", p.adhesion_J,
             p.adhesion_J / p.kappa);
    }
    printf("  Friction: xi=%.1f\n", p.xi);
    printf("  Interaction: kappa=%.1f\n", p.kappa);
    printf("  Cells: %d\n", sim.domain.num_cells());
    printf("  I/O: save_interval=%d, checkpoint_interval=%d, "
           "trajectory_interval=%d, trajectory_samples=%d\n",
           sim.save_interval, sim.checkpoint_interval,
           sim.trajectory_interval, sim.trajectory_samples);
    printf("\n");
    fflush(stdout);
  }

  if (!resumed) {
    if (!init_file.empty()) {
      // Initialize from JSON positions
      printf("Initializing %d cells from JSON positions...\n", (int)init_cx.size());
      for (size_t i = 0; i < init_cx.size(); ++i) {
        sim.domain.add_cell(init_cx[i], init_cy[i], radius);
      }
      sim.domain.update_overlap_pairs();
      sim.domain.sync_device_arrays();
    } else if (edge_test) {
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
