#include "domain.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace cellsim {

Domain::Domain(const SimParams &p) : params(p), next_cell_id(0) {}

Cell *Domain::add_cell(float cx, float cy, float radius) {
  // Compute bounding box for this cell
  int margin = static_cast<int>(radius * params.subdomain_padding) + 
               params.halo_width;
  BoundingBox bbox = {
      static_cast<int>(cx) - margin, static_cast<int>(cy) - margin,
      static_cast<int>(cx) + margin, static_cast<int>(cy) + margin};

  // Ensure minimum size
  if (bbox.width() < params.min_subdomain_size) {
    int expand = (params.min_subdomain_size - bbox.width()) / 2 + 1;
    bbox.x0 -= expand;
    bbox.x1 += expand;
  }
  if (bbox.height() < params.min_subdomain_size) {
    int expand = (params.min_subdomain_size - bbox.height()) / 2 + 1;
    bbox.y0 -= expand;
    bbox.y1 += expand;
  }

  // Create cell
  auto cell = std::make_unique<Cell>(next_cell_id++, bbox, params.halo_width);
  cell->initialize_circular(cx, cy, radius, params);

  Cell *ptr = cell.get();
  cells.push_back(std::move(cell));

  return ptr;
}

void Domain::remove_cell(int cell_id) {
  auto it = std::find_if(
      cells.begin(), cells.end(),
      [cell_id](const std::unique_ptr<Cell> &c) { return c->id == cell_id; });

  if (it != cells.end()) {
    cells.erase(it);
  }
}

Cell *Domain::get_cell(int cell_id) {
  auto it = std::find_if(
      cells.begin(), cells.end(),
      [cell_id](const std::unique_ptr<Cell> &c) { return c->id == cell_id; });

  return (it != cells.end()) ? it->get() : nullptr;
}

void Domain::update_all_bounding_boxes() {
  for (auto &cell : cells) {
    cell->update_bounding_box(params);
  }
}

void Domain::initialize_random_cells(int num_cells_to_add, float radius,
                                     float min_spacing) {
  // For single cell, place at domain center
  if (num_cells_to_add == 1) {
    float cx = params.Nx / 2.0f;
    float cy = params.Ny / 2.0f;
    add_cell(cx, cy, radius);
    return;
  }

  // Random placement with rejection sampling
  std::vector<Vec2> centers;
  int max_attempts = 10000;
  float current_spacing = min_spacing;

  while (static_cast<int>(centers.size()) < num_cells_to_add) {
    bool placed = false;

    for (int attempt = 0; attempt < max_attempts && !placed; ++attempt) {
      float cx = static_cast<float>(rand()) / RAND_MAX * params.Nx;
      float cy = static_cast<float>(rand()) / RAND_MAX * params.Ny;

      bool valid = true;
      for (const auto &c : centers) {
        float dx = fabsf(cx - c.x);
        float dy = fabsf(cy - c.y);

        // Periodic distance
        if (dx > params.Nx * 0.5f) dx = params.Nx - dx;
        if (dy > params.Ny * 0.5f) dy = params.Ny - dy;

        float dist = sqrtf(dx * dx + dy * dy);
        if (dist < current_spacing) {
          valid = false;
          break;
        }
      }

      if (valid) {
        centers.push_back({cx, cy});
        add_cell(cx, cy, radius);
        placed = true;
      }
    }

    if (!placed) {
      current_spacing *= 0.95f;
      if (current_spacing < radius) {
        printf("Warning: Could only place %d of %d cells\n",
               static_cast<int>(centers.size()), num_cells_to_add);
        break;
      }
    }
  }
}

} // namespace cellsim
