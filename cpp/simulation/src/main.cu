#include "simulation.cuh"
#ifdef ENABLE_VISUALIZER
#include "visualizer.cuh"
#endif
#include "simulation3d.cuh"
#include <algorithm>
#include <chrono>
#include <csignal>
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
// Graceful shutdown on SIGTERM (SLURM walltime limit)
//=============================================================================

// Global flag checked by simulation run loops.  When set, the current step
// completes, a final checkpoint + trajectory flush is written, and the
// process exits cleanly.  This prevents null-byte corruption in trajectory
// files at SLURM chain-job boundaries.
volatile std::sig_atomic_t g_shutdown_requested = 0;

static void sigterm_handler(int /*sig*/) {
  g_shutdown_requested = 1;
}

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
  } else if (selector.substr(0, 8) == "nearest(") {
    // Nearest selector: "nearest(x,y)" — cell closest to (x,y)
    ov.type = cellsim::SimParams::GammaOverride::Type::Nearest;
    auto paren_end = selector.find(')');
    if (paren_end == std::string::npos) {
      fprintf(stderr, "Error: --gamma %s missing ')' in nearest(x,y)\n", arg);
      return false;
    }
    std::string coords = selector.substr(8, paren_end - 8);
    auto comma = coords.find(',');
    if (comma == std::string::npos) {
      fprintf(stderr, "Error: --gamma %s nearest(x,y) requires two coordinates\n", arg);
      return false;
    }
    ov.pos_x = static_cast<float>(atof(coords.substr(0, comma).c_str()));
    ov.pos_y = static_cast<float>(atof(coords.substr(comma + 1).c_str()));
    ov.fraction = 0.0f;
  } else if (selector.substr(0, 8) == "cluster(") {
    // Cluster selector: "cluster(N%,x,y)" — nearest N% of cells to (x,y)
    ov.type = cellsim::SimParams::GammaOverride::Type::Cluster;
    auto paren_end = selector.find(')');
    if (paren_end == std::string::npos) {
      fprintf(stderr, "Error: --gamma %s missing ')' in cluster(N%%,x,y)\n", arg);
      return false;
    }
    std::string inner = selector.substr(8, paren_end - 8);
    // Parse "N%,x,y"
    auto c1 = inner.find(',');
    auto c2 = inner.find(',', c1 + 1);
    if (c1 == std::string::npos || c2 == std::string::npos) {
      fprintf(stderr, "Error: --gamma %s cluster(N%%,x,y) requires fraction and two coords\n", arg);
      return false;
    }
    std::string frac_str = inner.substr(0, c1);
    if (frac_str.back() == '%') {
      ov.fraction = static_cast<float>(atof(frac_str.c_str())) / 100.0f;
    } else {
      ov.fraction = static_cast<float>(atof(frac_str.c_str()));
    }
    ov.pos_x = static_cast<float>(atof(inner.substr(c1 + 1, c2 - c1 - 1).c_str()));
    ov.pos_y = static_cast<float>(atof(inner.substr(c2 + 1).c_str()));
    if (ov.fraction <= 0.0f || ov.fraction > 1.0f) {
      fprintf(stderr, "Error: --gamma %s cluster fraction must be 0-100%%\n", arg);
      return false;
    }
  } else {
    fprintf(stderr, "Error: --gamma %s unknown selector '%s' (use N%%, cellN, nearest(x,y), or cluster(N%%,x,y))\n",
            arg, selector.c_str());
    return false;
  }
  overrides.push_back(ov);
  overrides_set = true;
  return true;
}

static bool parse_radius_arg(const char *arg, cellsim::SimParams &params,
                            std::vector<cellsim::SimParams::RadiusOverride> &overrides,
                            bool &overrides_set) {
  std::string s(arg);
  auto colon = s.find(':');
  if (colon == std::string::npos) {
    // Bare value: set base target_radius (used for unassigned cells)
    params.target_radius = static_cast<float>(atof(arg));
    return true;
  }
  // Value:Selector
  float value = static_cast<float>(atof(s.substr(0, colon).c_str()));
  std::string selector = s.substr(colon + 1);
  if (selector.empty()) {
    fprintf(stderr, "Error: --radius %s has empty selector after ':'\n", arg);
    return false;
  }
  cellsim::SimParams::RadiusOverride ov;
  ov.value = value;
  ov.fraction = 0.0f;
  ov.cv = 0.0f;
  if (selector.substr(0, 2) == "cv") {
    // CV selector: "cv0.10" → Gaussian with coefficient of variation
    ov.type = cellsim::SimParams::RadiusOverride::Type::CV;
    ov.cv = static_cast<float>(atof(selector.substr(2).c_str()));
    if (ov.cv <= 0.0f || ov.cv > 1.0f) {
      fprintf(stderr, "Error: --radius %s CV must be 0-1 (got %.3f)\n", arg, ov.cv);
      return false;
    }
  } else if (selector.back() == '%') {
    // Fraction selector: "20%"
    ov.type = cellsim::SimParams::RadiusOverride::Type::Fraction;
    ov.fraction = static_cast<float>(atof(selector.c_str())) / 100.0f;
    if (ov.fraction <= 0.0f || ov.fraction > 1.0f) {
      fprintf(stderr, "Error: --radius %s fraction must be 0-100%%\n", arg);
      return false;
    }
  } else if (selector.substr(0, 4) == "cell") {
    // Cell selector: "cell0" or "cell0,5,12"
    ov.type = cellsim::SimParams::RadiusOverride::Type::Cells;
    std::string ids_str = selector.substr(4);
    size_t pos = 0;
    while (pos < ids_str.size()) {
      size_t comma = ids_str.find(',', pos);
      if (comma == std::string::npos) comma = ids_str.size();
      int id = atoi(ids_str.substr(pos, comma - pos).c_str());
      ov.cell_ids.push_back(id);
      pos = comma + 1;
    }
    if (ov.cell_ids.empty()) {
      fprintf(stderr, "Error: --radius %s has no cell IDs after 'cell'\n", arg);
      return false;
    }
  } else {
    fprintf(stderr, "Error: --radius %s unknown selector '%s' (use N%%, cellN, or cvN)\n",
            arg, selector.c_str());
    return false;
  }
  overrides.push_back(ov);
  overrides_set = true;
  return true;
}

void print_usage(const char *program) {
  // Construct defaults from the same sources main() uses, so help can never drift.
  cellsim::SimParams p;    // struct defaults (gamma, kappa, mu, xi, tau, v_A, lambda, ...)
  // main() overrides these before arg parsing:
  p.Nx = 256; p.Ny = 256; p.dt = 0.01f; p.t_end = 100.0f; p.target_radius = 20.0f;
  int    def_n_cells             = 8;
  int    def_save_interval       = 100;
  int    def_trajectory_samples  = 100;
  float  def_confluence          = 0.85f;

  printf("Usage: %s [options]\n", program);
  printf("\nPhase field cell simulation (2D/3D). GPU-accelerated with CUDA.\n");
  printf("Checkpoint-compatible across resume, chain jobs, and CUDA/MPI backends.\n\n");

  printf("  -n <num>        Number of cells (default: %d)\n", def_n_cells);
  printf("  -r <radius>     Cell target radius R (default: %.0f). Production: 49\n", p.target_radius);
  printf("  -N <size>       Domain size LxL (2D) or LxLxL (3D) (default: %d)\n", p.Nx);
  printf("                  Production formula: L = 1562 * sqrt(N_cells / 288) for R=49 at 89%% confluence\n");
  printf("  -Nz <size>      Z dimension for 3D (default: same as -N)\n");
  printf("  -s <space>      Minimum spacing between cells (default: auto)\n");
  printf("  --3d            Run 3D simulation (default: 2D)\n");
  printf("  --confluence <f>  Target packing fraction 0-1 (default: %.2f). Standard production: 0.89\n", def_confluence);
  printf("  -t <time>       End time (default: %.0f). Equilibration: 80000, production: 880000\n", p.t_end);
  printf("  -dt <step>      Time step (default: %.2f). Smaller = more accurate but slower\n", p.dt);
  printf("  --v-A <f>       Active motility velocity (default: %.1f). Typical: 0.004-0.05\n", p.v_A);
  printf("  --v-A-sigma <f> Std dev for per-cell v_A disorder (default: %.1f). Griffiths studies\n", p.v_A_sigma);
  printf("  --tau <f>       Reorientation/persistence time (default: %.0f)\n", p.tau);
  printf("  --gamma <f[:selector]>  Gradient coefficient gamma (default: %.1f).\n", p.gamma);
  printf("                  Controls cell stiffness. Palmieri: 1.0, Bresler: 3.75\n");
  printf("                  Repeat for heterogeneous populations:\n");
  printf("                    --gamma 1.0           Base value for all cells\n");
  printf("                    --gamma 0.35:20%%      20%% of cells get gamma=0.35\n");
  printf("                    --gamma 0.35:cell0,5   Cells 0 and 5 get gamma=0.35\n");
  printf("                    --gamma 0.35:nearest(x,y)       Cell nearest to (x,y)\n");
  printf("                    --gamma 0.35:cluster(20%%,x,y)   20%% nearest to (x,y)\n");
  printf("                  Order: base -> spatial -> fractions -> cells (override)\n");
  printf("  --radius <f[:selector]>  Per-cell target radius (default: same as -r).\n");
  printf("                  Enables polydisperse populations (different-sized cells).\n");
  printf("                  Each cell gets its own target_area = pi*R^2 and volume_coeff = mu/A0.\n");
  printf("                  Repeat for heterogeneous populations:\n");
  printf("                    --radius 20           Base radius for all cells\n");
  printf("                    --radius 15:25%%       25%% of cells get R=15\n");
  printf("                    --radius 20:cv0.10    Gaussian dist with mean=20, CV=10%%\n");
  printf("                    --radius 15:cell0     Cell 0 gets R=15\n");
  printf("                  Order: cv (all) -> fractions (random) -> cells (override)\n");
  printf("  --kappa <f>     Cell-cell repulsion kappa (default: %.1f)\n", p.kappa);
  printf("  --mu <f>        Volume constraint strength mu (default: %.1f). Bresler: 0.5\n", p.mu);
  printf("  --xi <f>        Friction coefficient xi (default: %.0f). Bresler: 1000\n", p.xi);
  printf("  --adhesion <J>  Adhesion strength J (default: %.1f, 0 = off). Adds -J*sum(phi_j)\n", p.adhesion_J);
  printf("  --abp           Use Active Brownian Particle model instead of Run-and-Tumble\n");
  printf("  -o <dir>        Output directory (default: ./output)\n");
  printf("  -c <file>       Resume from checkpoint file (inherits geometry + physics)\n");
  printf("  -i <file>       Load initial conditions from JSON file\n");
  printf("  --save-interval <n>   Steps between VTK frame saves (default: %d, 0 = disabled)\n", def_save_interval);
  printf("                        Production: use 0 (VTK disabled). Trajectory data is sufficient.\n");
  printf("  --print-interval <n>  Steps between progress output (default: -1 = use save_interval)\n");
  printf("  --save-final-checkpoint  Save checkpoint.bin at simulation end (for job chaining)\n");
  printf("  --checkpoint-interval <n>  Steps between checkpoint saves (default: -1 = auto, 0 = disabled)\n");
  printf("  --trajectory-samples <n>  Total trajectory snapshots over the run (default: %d). Production: 2000\n", def_trajectory_samples);
  printf("  --trajectory-interval <n>  Steps between trajectory saves (default: -1 = auto from samples)\n");
  printf("  --seed <n>      Random seed (default: time-based). For reproducible runs\n");
  printf("  --polarity-seed <n>  Separate seed for velocity/polarity RNG (default: derived from --seed)\n");
  printf("  --use-diagnostics       Enable volume/shape computation (slower)\n");
  printf("  --observable-interval <n>  Steps between GPU diagnostic measurements\n");
  printf("                            (energy, stress, contacts; requires -DENABLE_DIAGNOSTICS=ON)\n");
  printf("  --stress-fields         Include stress tensor fields in VTK output\n");
  printf("                          (requires -DENABLE_STRESS_FIELDS=ON)\n");
  printf("  --visualize [interval]  Open real-time display window (requires -DENABLE_VISUALIZER=ON)\n");
  printf("                          interval = steps between updates (default: 100)\n");
  printf("  --save-individual-fields  Save per-cell phi fields for energy analysis\n");
  printf("  --subdomain-padding <f>  Bbox buffer beyond cell extent, in units of R (default: %.2f)\n", p.subdomain_padding);
  printf("  --safe-mode     Limit memory allocation to prevent runaway VRAM usage\n");
  printf("  --batch <file>  Run multiple independent systems from a JSON config file.\n");
  printf("                  Each system resumes from its own checkpoint with independent\n");
  printf("                  v_A, gamma, and RNG seed. All systems share one GPU process.\n");
  printf("                  Periodic per-system checkpoints enable resume after interruption.\n");
  printf("                  JSON format:\n");
  printf("                    {\n");
  printf("                      \"t_end\": 1000,\n");
  printf("                      \"trajectory_samples\": 2000,\n");
  printf("                      \"checkpoint_interval\": 500000,\n");
  printf("                      \"print_interval\": 50000,\n");
  printf("                      \"systems\": [\n");
  printf("                        {\n");
  printf("                          \"checkpoint\": \"path/to/checkpoint.bin\",\n");
  printf("                          \"output\": \"path/to/output_dir\",\n");
  printf("                          \"v_A\": 0.004,\n");
  printf("                          \"gamma\": \"0.35:cell0\",\n");
  printf("                          \"seed_offset\": 42\n");
  printf("                        }\n");
  printf("                      ]\n");
  printf("                    }\n");
  printf("                  To resume a batch: point checkpoints at previous output dirs.\n");
  printf("\nParameter set reference:\n");
  printf("  Palmieri:  --gamma 1.0 --kappa 10 --mu 1.0 --xi 1500 (binary defaults)\n");
  printf("  Bresler:   --gamma 3.75 --kappa 10 --mu 0.5 --xi 1000\n");
  printf("  Production (288c, R=49, phi=0.89):  -n 288 -r 49 --confluence 0.89\n");
  printf("  Equilibration:  --v-A 0 -t 80000 --save-interval 0 --trajectory-samples 0\n");
  printf("  Motility run:   --v-A 0.008 -t 880000 --trajectory-samples 2000 --save-interval 0\n");
  printf("\n  -h            Show this help\n");
}

//=============================================================================
// BATCH MODE: Run multiple independent systems in a single GPU process
//
// All systems are concatenated into one set of device arrays.
// Each kernel uses d_system_id to look up per-system Nx/Ny/sum_field.
// Requirements: all systems must share the same domain dimensions (Nx, Ny).
// This is satisfied when all systems use the same (N_cells, R, rho).
//=============================================================================

struct BatchSystemConfig {
  std::string checkpoint;   // Path to checkpoint file
  std::string output;       // Output directory
  float v_A = -1.0f;       // Override v_A (-1 = use checkpoint value)
  std::string gamma_spec;   // Gamma override spec (e.g., "0.35:cell0")
  int seed_offset = 0;     // RNG seed offset for independent replicas
};

struct BatchConfig {
  std::vector<BatchSystemConfig> systems;
  float t_end = -1.0f;          // Override t_end (-1 = use checkpoint value)
  int trajectory_samples = 2000;
  int checkpoint_interval = -1;  // -1 = auto
  int print_interval = 1000000;
};

// Simple JSON batch config parser
static bool parse_batch_config(const char *filename, BatchConfig &config) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    printf("Error: Could not open batch config: %s\n", filename);
    return false;
  }
  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::string json(file_size, '\0');
  fread(&json[0], 1, file_size, f);
  fclose(f);

  // Parse t_end
  {
    const char *p = strstr(json.c_str(), "\"t_end\"");
    if (p) { p = strchr(p, ':'); if (p) config.t_end = (float)atof(p + 1); }
  }
  // Parse trajectory_samples
  {
    const char *p = strstr(json.c_str(), "\"trajectory_samples\"");
    if (p) { p = strchr(p, ':'); if (p) config.trajectory_samples = atoi(p + 1); }
  }
  // Parse checkpoint_interval
  {
    const char *p = strstr(json.c_str(), "\"checkpoint_interval\"");
    if (p) { p = strchr(p, ':'); if (p) config.checkpoint_interval = atoi(p + 1); }
  }
  // Parse print_interval
  {
    const char *p = strstr(json.c_str(), "\"print_interval\"");
    if (p) { p = strchr(p, ':'); if (p) config.print_interval = atoi(p + 1); }
  }

  // Parse systems array
  const char *systems_start = strstr(json.c_str(), "\"systems\"");
  if (!systems_start) {
    printf("Error: batch config missing 'systems' array\n");
    return false;
  }
  systems_start = strchr(systems_start, '[');
  if (!systems_start) return false;

  // Find each system object
  const char *pos = systems_start;
  while (true) {
    const char *obj_start = strchr(pos, '{');
    if (!obj_start) break;
    const char *obj_end = strchr(obj_start, '}');
    if (!obj_end) break;

    std::string obj(obj_start, obj_end - obj_start + 1);
    BatchSystemConfig sys;

    // Parse checkpoint
    const char *ck = strstr(obj.c_str(), "\"checkpoint\"");
    if (ck) {
      ck = strchr(ck + 12, '"'); if (ck) { ck++;
      const char *ce = strchr(ck, '"');
      if (ce) sys.checkpoint = std::string(ck, ce - ck);
    }}
    // Parse output
    const char *ou = strstr(obj.c_str(), "\"output\"");
    if (ou) {
      ou = strchr(ou + 8, '"'); if (ou) { ou++;
      const char *oe = strchr(ou, '"');
      if (oe) sys.output = std::string(ou, oe - ou);
    }}
    // Parse v_A
    const char *va = strstr(obj.c_str(), "\"v_A\"");
    if (va) { va = strchr(va, ':'); if (va) sys.v_A = (float)atof(va + 1); }
    // Parse gamma
    const char *ga = strstr(obj.c_str(), "\"gamma\"");
    if (ga) {
      ga = strchr(ga + 7, '"'); if (ga) { ga++;
      const char *ge = strchr(ga, '"');
      if (ge) sys.gamma_spec = std::string(ga, ge - ga);
    }}
    // Parse seed_offset
    const char *so = strstr(obj.c_str(), "\"seed_offset\"");
    if (so) { so = strchr(so, ':'); if (so) sys.seed_offset = atoi(so + 1); }

    if (!sys.checkpoint.empty()) {
      config.systems.push_back(sys);
    }
    pos = obj_end + 1;
  }

  printf("Batch config: %zu systems, t_end=%.0f, trajectory_samples=%d\n",
         config.systems.size(), config.t_end, config.trajectory_samples);
  for (size_t i = 0; i < config.systems.size(); ++i) {
    printf("  [%zu] checkpoint=%s output=%s", i, config.systems[i].checkpoint.c_str(),
           config.systems[i].output.c_str());
    if (config.systems[i].v_A >= 0) printf(" v_A=%.4f", config.systems[i].v_A);
    if (!config.systems[i].gamma_spec.empty()) printf(" gamma=%s", config.systems[i].gamma_spec.c_str());
    printf("\n");
  }

  return !config.systems.empty();
}

// Run batch mode: multiple independent systems in a single GPU process.
// Each system is a separate Simulation instance with its own Domain+Integrator.
// Kernels are launched sequentially per system per step — no kernel changes needed.
// GPU memory is shared via the CUDA context; each Integrator allocates its own arrays.
static int run_batch(const BatchConfig &config, int base_seed) {
  int num_systems = (int)config.systems.size();
  printf("=== BATCH MODE: %d independent systems ===\n\n", num_systems);

  // Phase 1: Create N independent Simulation instances from checkpoints
  std::vector<std::unique_ptr<Simulation>> sims;
  SimParams base_params;

  for (int s = 0; s < num_systems; ++s) {
    printf("Loading system %d: %s\n", s, config.systems[s].checkpoint.c_str());

    // Load checkpoint to get params
    Domain temp_domain(SimParams{});
    CheckpointHeader hdr;
    std::vector<float> ck_v_A, ck_gamma, ck_radius;
    if (!load_checkpoint(temp_domain, config.systems[s].checkpoint,
                         hdr, &ck_v_A, &ck_gamma, &ck_radius)) {
      printf("ERROR: Failed to load checkpoint for system %d\n", s);
      return 1;
    }

    // Apply config overrides
    SimParams params = temp_domain.params;
    if (config.t_end > 0) params.t_end = config.t_end;
    if (s == 0) base_params = params;

    printf("  System %d: %d cells, domain %dx%d, t=%.1f\n",
           s, temp_domain.num_cells(), params.Nx, params.Ny, hdr.current_time);

    // Create simulation
    auto sim = std::make_unique<Simulation>(params);
    sim->output_dir = config.systems[s].output;
    sim->save_interval = 0;
    sim->save_vtk = false;
    sim->checkpoint_interval = (config.checkpoint_interval > 0) ? config.checkpoint_interval : 8000000;
    sim->trajectory_samples = config.trajectory_samples;
    sim->print_interval = config.print_interval;
    sim->current_time = (double)hdr.current_time;
    sim->current_step = hdr.current_step;
    sim->resumed_from_checkpoint = true;

    // Move cells from temp domain into sim domain
    sim->domain.cells = std::move(temp_domain.cells);
    sim->domain.device_arrays_dirty = true;
    sim->domain.sync_device_arrays();

    // Per-system v_A override
    if (config.systems[s].v_A >= 0) {
      int nc = sim->domain.num_cells();
      std::vector<float> v_A(nc, config.systems[s].v_A);
      sim->integrator.checkpoint_v_A = std::move(v_A);
    } else if (!ck_v_A.empty()) {
      sim->integrator.checkpoint_v_A = std::move(ck_v_A);
    }

    // Per-system gamma override
    if (!config.systems[s].gamma_spec.empty()) {
      const std::string &spec = config.systems[s].gamma_spec;
      size_t colon = spec.find(':');
      if (colon != std::string::npos) {
        float gamma_val = std::stof(spec.substr(0, colon));
        std::string sel = spec.substr(colon + 1);
        if (sel.substr(0, 4) == "cell") {
          std::string ids = sel.substr(4);
          size_t p = 0;
          while (p < ids.size()) {
            size_t c = ids.find(',', p);
            if (c == std::string::npos) c = ids.size();
            int local = std::stoi(ids.substr(p, c - p));
            SimParams::GammaOverride ov;
            ov.value = gamma_val;
            ov.type = SimParams::GammaOverride::Type::Cells;
            ov.cell_ids.push_back(local);
            sim->integrator.gamma_overrides.push_back(ov);
            p = c + 1;
          }
          sim->integrator.gamma_overrides_set = true;
        }
      }
    }
    if (!ck_gamma.empty()) {
      sim->integrator.checkpoint_gamma = std::move(ck_gamma);
    }
    if (!ck_radius.empty()) {
      sim->integrator.checkpoint_target_radius = std::move(ck_radius);
    }

    // RNG seed offset for independent replicas
    srand(base_seed + config.systems[s].seed_offset);

    std::filesystem::create_directories(config.systems[s].output);
    sims.push_back(std::move(sim));
  }

  printf("\nStarting batch: %d systems, t=%.0f to t=%.0f\n",
         num_systems, sims[0]->current_time, base_params.t_end);
  fflush(stdout);

  // Phase 2: Run in lockstep — each step advances all systems
  auto t0 = std::chrono::high_resolution_clock::now();
  float dt = base_params.dt;
  float t_end = base_params.t_end;

  // Compute trajectory interval for each system
  std::vector<int> traj_intervals(num_systems);
  for (int s = 0; s < num_systems; ++s) {
    int total_steps = (int)((t_end - sims[s]->current_time) / dt + 0.5f);
    traj_intervals[s] = std::max(1, total_steps / config.trajectory_samples);
  }

  // Checkpoint interval (from config or default)
  int ckpt_interval = (config.checkpoint_interval > 0) ? config.checkpoint_interval : 8000000;

  int step_count = 0;
  int batch_size = 10;  // Queue this many steps before checking print/save
  while (sims[0]->current_time < t_end && !g_shutdown_requested) {
    // Determine how many pure-compute steps we can batch before next event
    int steps_to_next_print = config.print_interval - (step_count % config.print_interval);
    int steps_to_next_ckpt = ckpt_interval - (step_count % ckpt_interval);
    int steps_to_next_traj = traj_intervals[0] - (step_count % traj_intervals[0]);
    int steps_to_end = (int)((t_end - sims[0]->current_time) / dt + 0.5f);
    int batch = batch_size;
    if (steps_to_next_print > 0) batch = std::min(batch, steps_to_next_print);
    if (steps_to_next_ckpt > 0) batch = std::min(batch, steps_to_next_ckpt);
    if (traj_intervals[0] > 0 && steps_to_next_traj > 0) batch = std::min(batch, steps_to_next_traj);
    if (steps_to_end > 0) batch = std::min(batch, steps_to_end);
    batch = std::max(batch, 1);

    // Queue batch of pure-compute steps (no sync, no save)
    for (int b = 0; b < batch; ++b) {
      step_count++;
      bool is_last = (b == batch - 1);
      bool is_print_step = (step_count % config.print_interval == 0);
      bool is_ckpt_step = (step_count % ckpt_interval == 0);
      bool save_traj = (traj_intervals[0] > 0 && step_count % traj_intervals[0] == 0);
      bool need_sync = is_last && (save_traj || is_print_step || is_ckpt_step);

      for (int s = 0; s < num_systems; ++s) {
        sims[s]->integrator.step(sims[s]->domain, dt, save_traj && is_last, need_sync);
        sims[s]->current_time += dt;
        sims[s]->current_step++;

        if (save_traj && is_last) {
          sims[s]->save_trajectory();
        }
      }
    }

    // Handle events at batch boundary
    bool is_print_step = (step_count % config.print_interval == 0);
    bool is_ckpt_step = (step_count % ckpt_interval == 0);

    if (is_print_step) {
      printf("Step %7d | t=%.4f\n", sims[0]->current_step, sims[0]->current_time);
      fflush(stdout);
    }

    if (is_ckpt_step) {
      printf("Checkpoint at step %d (t=%.2f)...\n", sims[0]->current_step, sims[0]->current_time);
      for (int s = 0; s < num_systems; ++s) {
        sims[s]->save_current_checkpoint(config.systems[s].output + "/checkpoint.bin");
      }
      fflush(stdout);
    }
  }

  // Save on SIGTERM (clean shutdown for SLURM chain jobs)
  if (g_shutdown_requested) {
    printf("\nSIGTERM received — saving checkpoints for resume...\n");
  }

  double elapsed = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - t0).count();

  // Phase 3: Save per-system checkpoints
  printf("\nSaving per-system checkpoints...\n");
  for (int s = 0; s < num_systems; ++s) {
    sims[s]->save_current_checkpoint(config.systems[s].output + "/checkpoint.bin");
    printf("  System %d: %s/checkpoint.bin\n", s, config.systems[s].output.c_str());
  }

  printf("\nBatch finished: %d systems, %.1f seconds\n", num_systems, elapsed);
  printf("Total wall time: %.3f seconds\n", elapsed);
  return 0;
}

int main(int argc, char *argv[]) {
  // Register SIGTERM handler for clean SLURM chain-job transitions
  std::signal(SIGTERM, sigterm_handler);

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
  bool confluence_set = false;  // Track if user explicitly set --confluence

  int num_cells = 8;
  float min_spacing =
      -1.0f; // -1 means auto-calculate based on radius and cell count
  std::string output_dir = "./output";
  std::string checkpoint_file = "";
  std::string init_file = "";  // JSON initial conditions file
  std::string batch_config_file = "";  // --batch mode
  bool edge_test = false;
  bool corner_push_test = false;
  bool use_diagnostics = false;
  bool save_final_checkpoint = false;
  bool save_individual_fields =
      false; // Save individual cell fields for energy analysis
  int save_interval = 100;
  bool save_interval_set = false;
  int print_interval = -1; // -1 means use save_interval
  int checkpoint_interval = -1; // -1 means use save_interval * 10
  bool checkpoint_interval_set = false;
  int random_seed = -1;         // -1 means use time-based seed
  int polarity_seed = -1;       // -1 means derive from --seed
  int trajectory_samples = 100; // Number of trajectory data points to save
  bool trajectory_samples_set = false;
  int trajectory_interval =
      -1; // -1 = use save_interval, 0 = compute from samples, >0 = explicit
  bool trajectory_interval_set = false;
  int observable_interval = 0; // 0 = disabled, >0 = GPU diagnostic measurements
  bool stress_fields = false;  // Include stress tensor fields in VTK output
  bool enable_visualizer = false;
  int visualize_interval = 100;
  float v_A_override = -1.0f; // -1 means use default from params
  float tau_override = -1.0f;  // -1 means use default from params
  bool use_abp = false;       // Use ABP model instead of Run-and-Tumble
  bool safe_mode = false;   // Limit memory allocation to 1GB
  bool use_fcc = true;        // Use FCC lattice (default) vs simple cubic
  float confluence = -1.0f;   // -1 = not set (must provide -N or --confluence)
  bool subdomain_padding_set = false; // Track if user explicitly set padding
  bool adhesion_J_set = false; // Track if user explicitly set --adhesion
  bool gamma_base_set = false; // Track if bare --gamma was already set
  std::vector<cellsim::SimParams::GammaOverride> gamma_overrides;
  bool gamma_overrides_set = false;
  std::vector<cellsim::SimParams::RadiusOverride> radius_overrides;
  bool radius_overrides_set = false;

  // Parse command line
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--3d") {
      run_3d = true;
    } else if (arg == "-Nz" && i + 1 < argc) {
      Nz = atoi(argv[++i]);
    } else if (arg == "-n" && i + 1 < argc) {
      num_cells = atoi(argv[++i]);
    } else if (arg == "--radius" && i + 1 < argc) {
      if (!parse_radius_arg(argv[++i], params, radius_overrides, radius_overrides_set)) {
        return 1;
      }
    } else if (arg == "-r" && i + 1 < argc) {
      // -r is shorthand for --radius (bare value only)
      params.target_radius = static_cast<float>(atof(argv[++i]));
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
    } else if (arg == "--batch" && i + 1 < argc) {
      batch_config_file = argv[++i];
    } else if (arg == "-i" && i + 1 < argc) {
      init_file = argv[++i];
    } else if (arg == "--edge-test") {
      edge_test = true;
      num_cells = 3;
    } else if (arg == "--corner-push-test") {
      corner_push_test = true;
    } else if (arg == "--use-diagnostics") {
      use_diagnostics = true;
    } else if (arg == "--save-interval" && i + 1 < argc) {
      save_interval = atoi(argv[++i]);
      save_interval_set = true;
    } else if (arg == "--print-interval" && i + 1 < argc) {
      print_interval = atoi(argv[++i]);
    } else if (arg == "--subdomain-padding" && i + 1 < argc) {
      params.subdomain_padding = atof(argv[++i]);
      subdomain_padding_set = true;
    } else if (arg == "--save-final-checkpoint") {
      save_final_checkpoint = true;
    } else if (arg == "--checkpoint-interval" && i + 1 < argc) {
      checkpoint_interval = atoi(argv[++i]);
      checkpoint_interval_set = true;
    } else if (arg == "--seed" && i + 1 < argc) {
      random_seed = atoi(argv[++i]);
    } else if (arg == "--polarity-seed" && i + 1 < argc) {
      polarity_seed = atoi(argv[++i]);
    } else if (arg == "--trajectory-samples" && i + 1 < argc) {
      trajectory_samples = atoi(argv[++i]);
      trajectory_samples_set = true;
    } else if (arg == "--trajectory-interval" && i + 1 < argc) {
      trajectory_interval = atoi(argv[++i]);
      trajectory_interval_set = true;
    } else if (arg == "--observable-interval" && i + 1 < argc) {
      observable_interval = atoi(argv[++i]);
    } else if (arg == "--stress-fields") {
      stress_fields = true;
    } else if (arg == "--visualize") {
      enable_visualizer = true;
      // Optional: next arg is interval if it's a number
      if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
        visualize_interval = atoi(argv[++i]);
      }
    } else if (arg == "--v-A" && i + 1 < argc) {
      v_A_override = atof(argv[++i]);
    } else if (arg == "--v-A-sigma" && i + 1 < argc) {
      params.v_A_sigma = atof(argv[++i]);
    } else if (arg == "--tau" && i + 1 < argc) {
      tau_override = atof(argv[++i]);
    } else if (arg == "--abp") {
      use_abp = true;
    } else if (arg == "--gamma" && i + 1 < argc) {
      // Check for duplicate bare --gamma (conflict)
      const char *gval = argv[++i];
      std::string gs(gval);
      bool is_bare = (gs.find(':') == std::string::npos);
      if (is_bare && gamma_base_set) {
        fprintf(stderr, "ERROR: --gamma base value specified multiple times. Use a single bare --gamma for the base.\n");
        return 1;
      }
      if (!parse_gamma_arg(gval, params, gamma_overrides, gamma_overrides_set)) {
        return 1;
      }
      if (is_bare) gamma_base_set = true;
    } else if (arg == "--kappa" && i + 1 < argc) {
      params.kappa = atof(argv[++i]);
    } else if (arg == "--mu" && i + 1 < argc) {
      params.mu = atof(argv[++i]);
    } else if (arg == "--xi" && i + 1 < argc) {
      params.xi = atof(argv[++i]);
    } else if (arg == "--adhesion" && i + 1 < argc) {
      params.adhesion_J = atof(argv[++i]);
      adhesion_J_set = true;
    } else if (arg == "--save-individual-fields") {
      save_individual_fields = true;
    } else if (arg == "--safe-mode") {
      safe_mode = true;
    } else if (arg == "--confluence" && i + 1 < argc) {
      confluence = atof(argv[++i]);
      confluence_set = true;
    } else if (arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "ERROR: Unrecognized argument '%s'\n", argv[i]);
      fprintf(stderr, "Use -h for usage information.\n");
      return 1;
    }
  }

  // =========================================================================
  // Post-parse validation: detect conflicting flags
  // =========================================================================

  // #1: -N and --confluence are mutually exclusive
  if (domain_size_set && confluence_set) {
    fprintf(stderr, "ERROR: Both -N (domain size) and --confluence specified. Use one or the other.\n");
    return 1;
  }

  // #5: --trajectory-samples and --trajectory-interval are mutually exclusive
  if (trajectory_samples_set && trajectory_interval_set) {
    fprintf(stderr, "ERROR: Both --trajectory-samples and --trajectory-interval specified. Use one or the other.\n");
    return 1;
  }

  // #10: For fresh starts (no checkpoint), require -N or --confluence
  if (checkpoint_file.empty() && init_file.empty() && !domain_size_set && !confluence_set) {
    // No domain size info provided — use default confluence
    confluence = 0.85f;
    confluence_set = true;
    printf("Note: No -N or --confluence specified. Using default confluence=0.85.\n");
  }

  // Use target_radius for local radius variable (used in domain sizing)
  float radius = params.target_radius;

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

  // Apply v_A override
  if (v_A_override >= 0.0f) {
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
  // Batch Mode Branch
  //=========================================================================
  if (!batch_config_file.empty()) {
    BatchConfig batch_config;
    if (!parse_batch_config(batch_config_file.c_str(), batch_config)) {
      return 1;
    }
    return run_batch(batch_config, random_seed >= 0 ? random_seed : (int)time(nullptr));
  }

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
    params3d.mu = params.mu;
    params3d.target_radius = radius;
    params3d.v_A = (v_A_override >= 0.0f) ? v_A_override : params.v_A;
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
    sim3d.print_interval = print_interval;
    sim3d.checkpoint_interval = checkpoint_interval;
    sim3d.trajectory_samples = trajectory_samples;
    sim3d.trajectory_interval = trajectory_interval;
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
        sim3d.resumed_from_checkpoint = true;
        printf("Resumed 3D from checkpoint: step=%d, t=%.4f\n",
               sim3d.current_step, sim3d.current_time);

        // Apply CLI overrides (same logic as 2D)
        sim3d.domain.params.t_end = params.t_end;

        if (v_A_override >= 0.0f) {
          sim3d.domain.params.v_A = v_A_override;
        }

        // Override physics from CLI
        sim3d.domain.params.gamma = params.gamma;
        sim3d.domain.params.kappa = params.kappa;
        sim3d.domain.params.mu = params.mu;
        sim3d.domain.params.xi = params.xi;
        sim3d.domain.params.tau = params.tau;

        // Restore save/trajectory settings from CLI
        sim3d.save_interval = save_interval;
        sim3d.print_interval = print_interval;
        sim3d.checkpoint_interval = checkpoint_interval;
        sim3d.trajectory_samples = trajectory_samples;
        sim3d.trajectory_interval = trajectory_interval;

        fflush(stdout);
      } else {
        printf("Warning: Could not load 3D checkpoint, starting fresh\n");
      }
    }

    if (!resumed) {
      {
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

  // #10: Auto-compute 2D domain size from confluence if not explicitly set
  if (!domain_size_set && confluence_set && checkpoint_file.empty()) {
    float cell_area = M_PI * radius * radius;
    float total_area = num_cells * cell_area;
    float domain_area = total_area / confluence;
    int N = static_cast<int>(ceilf(sqrtf(domain_area)));
    params.Nx = N;
    params.Ny = N;
    printf("Auto-computed 2D domain size N=%d for %d cells, R=%.0f, confluence=%.0f%%\n",
           N, num_cells, radius, confluence * 100.0f);
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
  // Pass radius overrides (live on Simulation, not SimParams)
  if (radius_overrides_set) {
    sim.radius_overrides = radius_overrides;
    sim.radius_overrides_set = true;
  }
  sim.print_interval = print_interval;
  sim.checkpoint_interval = checkpoint_interval;
  sim.trajectory_samples = trajectory_samples;
  // Set polarity seed on integrator (before first step initializes RNG)
  sim.integrator.polarity_seed = polarity_seed;
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

#ifdef ENABLE_VISUALIZER
  static cellsim::Visualizer vis_instance;
  if (enable_visualizer) {
    sim.visualizer = &vis_instance;
    sim.visualize_interval = visualize_interval;
  }
#else
  if (enable_visualizer) {
    printf("Warning: --visualize requires -DENABLE_VISUALIZER=ON at build time\n");
  }
#endif

  // Initialize
  bool resumed = false;
  if (!checkpoint_file.empty()) {
    // Save command-line overrides before loading checkpoint (which overwrites
    // params)
    float cmd_t_end = params.t_end;
    float cmd_dt = params.dt;
    float cmd_v_A = params.v_A;
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

      // #8: Always restore physics parameters from CLI (checkpoint stores
      // equilibration values; production runs may use different params).
      // Print info when overriding.
      auto &cp = sim.domain.params;
      if (cp.dt != cmd_dt) {
        printf("  Override dt: %.4f -> %.4f\n", cp.dt, cmd_dt);
        cp.dt = cmd_dt;
      }
      if (cp.gamma != params.gamma) {
        printf("  Override gamma: %.4f -> %.4f\n", cp.gamma, params.gamma);
        cp.gamma = params.gamma;
      }
      if (cp.kappa != params.kappa) {
        printf("  Override kappa: %.4f -> %.4f\n", cp.kappa, params.kappa);
        cp.kappa = params.kappa;
      }
      if (cp.mu != params.mu) {
        printf("  Override mu: %.4f -> %.4f\n", cp.mu, params.mu);
        cp.mu = params.mu;
      }
      if (cp.xi != params.xi) {
        printf("  Override xi: %.1f -> %.1f\n", cp.xi, params.xi);
        cp.xi = params.xi;
      }

      // #7: Restore adhesion_J unconditionally when explicitly set (allows --adhesion 0)
      if (adhesion_J_set) {
        if (cp.adhesion_J != params.adhesion_J) {
          printf("  Override adhesion_J: %.4f -> %.4f\n", cp.adhesion_J, params.adhesion_J);
        }
        cp.adhesion_J = params.adhesion_J;
      }

      // Restore subdomain_padding if explicitly set
      if (subdomain_padding_set) {
        if (cp.subdomain_padding != params.subdomain_padding) {
          printf("  Override subdomain_padding: %.2f -> %.2f\n", cp.subdomain_padding, params.subdomain_padding);
        }
        cp.subdomain_padding = params.subdomain_padding;
      }

      // Restore v_A if user explicitly overrode it
      if (v_A_override >= 0.0f) {
        if (cp.v_A != v_A_override) {
          printf("  Override v_A: %.6f -> %.6f\n", cp.v_A, v_A_override);
        }
        cp.v_A = v_A_override;
      }

      // Restore tau if user explicitly overrode it
      if (tau_override > 0.0f) {
        if (cp.tau != tau_override) {
          printf("  Override tau: %.1f -> %.1f\n", cp.tau, tau_override);
        }
        cp.tau = tau_override;
      }

      // If user specified --v-A or --v-A-sigma on command line, regenerate
      // per-cell v_A instead of using checkpoint values (needed when starting
      // production from an equilibration checkpoint with v_A=0)
      if (v_A_override >= 0.0f || params.v_A_sigma > 0.0f) {
        sim.loaded_v_A.clear();
        sim.domain.params.v_A_sigma = params.v_A_sigma;
        printf("  Per-cell v_A will be regenerated (--v-A or --v-A-sigma specified)\n");
      }

      // If user specified gamma overrides, regenerate per-cell gamma
      if (gamma_overrides_set) {
        sim.loaded_gamma.clear();
        sim.gamma_overrides = gamma_overrides;
        sim.gamma_overrides_set = true;
        printf("  Per-cell gamma will be regenerated (--gamma overrides specified)\n");
      }

      // If user specified radius overrides, regenerate per-cell radius
      if (radius_overrides_set) {
        sim.loaded_target_radius.clear();
        sim.radius_overrides = radius_overrides;
        sim.radius_overrides_set = true;
        printf("  Per-cell radius will be regenerated (--radius overrides specified)\n");
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
#ifdef ENABLE_VISUALIZER
  if (sim.visualizer) {
    char title[256];
    snprintf(title, sizeof(title), "Cell Sim: %d cells, %dx%d",
             sim.domain.num_cells(), sim.domain.params.Nx, sim.domain.params.Ny);
    sim.visualizer->init(sim.domain.params.Nx, sim.domain.params.Ny, title);
  }
#endif
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
