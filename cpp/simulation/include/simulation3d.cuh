#pragma once

#include "domain3d.cuh"
#include "integrator3d.cuh"
#include "types3d.cuh"
#include <chrono>
#include <cstdio>
#include <string>

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
// Simulation3D - Main simulation controller for 3D
//=============================================================================

class Simulation3D {
public:
  Domain3D domain;
  Integrator3D integrator;

  // Simulation state
  int current_step;
  float current_time;

  // Output settings
  std::string output_dir;
  int save_interval;
  int trajectory_interval;
  bool save_individual_fields_flag;

  // Timing
  std::chrono::steady_clock::time_point start_time;

public:
  Simulation3D(const SimParams3D &params);

  // Initialize cells
  void initialize_random(int num_cells, float radius, float min_spacing);
  void initialize_grid(int num_cells, float radius, float confluence);

  // Load from checkpoint
  bool load_checkpoint(const char *filename);

  // Run simulation
  void run(float t_end);

  // Single step
  void step();

  // Save current state
  void save_checkpoint();
  void save_vtk();
  void save_individual_cell_fields();

  // Print status
  void print_status();
};

//=============================================================================
// Simulation3D Implementation
//=============================================================================

inline Simulation3D::Simulation3D(const SimParams3D &params)
    : domain(params), current_step(0), current_time(0), save_interval(100),
      trajectory_interval(100), save_individual_fields_flag(false) {}

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
  return load_checkpoint_3d(filename, domain, current_step, current_time);
}

inline void Simulation3D::run(float t_end) {
  start_time = std::chrono::steady_clock::now();

  printf("Starting 3D simulation: t_end=%.2f, dt=%.4f\n", t_end,
         domain.params.dt);

  while (current_time < t_end) {
    step();

    if (current_step % 50 == 0) {
      print_status();
    }

    if (save_interval > 0 && current_step % save_interval == 0) {
      save_checkpoint();
      if (save_individual_fields_flag) {
        save_individual_cell_fields();
      }
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  printf("\n3D Simulation complete: %d steps, t=%.2f\n", current_step,
         current_time);
  printf("Total wall time: %.3f seconds\n", elapsed.count() / 1000.0);
}

// Host version of tumble check (same logic as device function)
inline bool check_tumble_event_host(float dt, float tau, float rand_val) {
  float P_tumble = dt / tau;
  return rand_val < P_tumble;
}

inline void Simulation3D::step() {
  // Update polarization (rotational diffusion or tumble)
  float dt = domain.params.dt;
  float tau = domain.params.tau;

  for (auto &cell : domain.cells) {
    if (domain.params.motility_model ==
        SimParams::MotilityModel::RunAndTumble) {
      // Check for tumble event
      float rand_val = static_cast<float>(rand()) / RAND_MAX;
      if (check_tumble_event_host(dt, tau, rand_val)) {
        // Random new direction on unit sphere
        cell->theta = static_cast<float>(rand()) / RAND_MAX * 2.0f * M_PI;
        cell->phi_pol =
            acosf(2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
        cell->polarization.x = sinf(cell->phi_pol) * cosf(cell->theta);
        cell->polarization.y = sinf(cell->phi_pol) * sinf(cell->theta);
        cell->polarization.z = cosf(cell->phi_pol);
      }
    } else {
      // ABP: rotational diffusion on sphere
      // Use small random perturbations
      float D_r = 1.0f / (2.0f * tau);
      float sigma = sqrtf(2.0f * D_r * dt);

      float dtheta =
          sigma * (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
      float dphi =
          sigma * (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;

      cell->theta += dtheta;
      cell->phi_pol += dphi;

      // Clamp phi_pol to [0, pi]
      if (cell->phi_pol < 0)
        cell->phi_pol = -cell->phi_pol;
      if (cell->phi_pol > M_PI)
        cell->phi_pol = 2.0f * M_PI - cell->phi_pol;

      cell->polarization.x = sinf(cell->phi_pol) * cosf(cell->theta);
      cell->polarization.y = sinf(cell->phi_pol) * sinf(cell->theta);
      cell->polarization.z = cosf(cell->phi_pol);
    }

    // Set velocity from polarization
    cell->velocity = cell->polarization * domain.params.v_A;
  }

  // Integrate
  integrator.step(domain, dt);

  current_step++;
  current_time += dt;
}

inline void Simulation3D::save_checkpoint() {
  if (output_dir.empty())
    return;

  char filename[256];
  snprintf(filename, sizeof(filename), "%s/checkpoint_3d_%06d.bin",
           output_dir.c_str(), current_step);
  save_checkpoint_3d(filename, domain, current_step, current_time);
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

inline void Simulation3D::print_status() {
  printf("Step %6d | t=%.4f\n", current_step, current_time);
}

} // namespace cellsim
