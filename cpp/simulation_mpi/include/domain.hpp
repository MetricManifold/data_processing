#pragma once

#include "cell.hpp"
#include "types.hpp"
#include <memory>
#include <vector>

namespace cellsim {

//=============================================================================
// Domain - Manages global domain and cell collection (CPU/MPI version)
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

  // Update all cell bounding boxes
  void update_all_bounding_boxes();

  // Initialize random cell configuration
  void initialize_random_cells(int num_cells, float radius, float min_spacing);
};

} // namespace cellsim
