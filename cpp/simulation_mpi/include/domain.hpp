#pragma once

#include "cell.hpp"
#include "types.hpp"
#include <memory>
#include <vector>

namespace cellsim {

//=============================================================================
// Domain - Manages global domain and cell collection (CPU/OpenMP version)
//=============================================================================

class Domain {
public:
  SimParams params;

  // Cell collection
  std::vector<std::unique_ptr<Cell>> cells;
  int next_cell_id;

public:
  Domain(const SimParams &p);
  ~Domain() = default;

  // Cell management
  Cell *add_cell(float cx, float cy, float radius);
  void remove_cell(int cell_id);
  Cell *get_cell(int cell_id);
  int num_cells() const { return static_cast<int>(cells.size()); }
  
  // Access cell by index (const and non-const)
  const Cell& cell(int idx) const { return *cells[idx]; }
  Cell& cell(int idx) { return *cells[idx]; }

  // Update all cell bounding boxes
  void update_all_bounding_boxes();
  
  // Update ghost cell data (received from MPI)
  void update_ghost_cell(int cell_id, const float* phi_data,
                         float volume, float cx, float cy,
                         float vx, float vy, float theta, float px, float py);

  // Initialize random cell configuration
  void initialize_random_cells(int num_cells, float radius, float min_spacing);
};

} // namespace cellsim
