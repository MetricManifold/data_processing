#pragma once

#include "cell.cuh"
#include "types.cuh"
#include <algorithm>
#include <memory>
#include <vector>

namespace cellsim {

//=============================================================================
// OverlapPair - Tracks which cells have overlapping bounding boxes
//=============================================================================

struct OverlapPair {
  int cell_i;
  int cell_j;
  BoundingBox overlap_region; // Intersection of the two bounding boxes
};

//=============================================================================
// Domain - Manages global domain and cell collection
//=============================================================================

class Domain {
public:
  SimParams params;

  // Cell collection
  std::vector<std::unique_ptr<Cell>> cells;
  int next_cell_id;

  // Interaction tracking
  std::vector<OverlapPair> overlap_pairs;

  // Device arrays for batch operations
  float **d_cell_phi_ptrs;     // Array of pointers to each cell's φ
  float **d_cell_dphi_dt_ptrs; // Array of pointers to each cell's dφ/dt
  int *d_cell_widths;          // Width of each cell's subdomain
  int *d_cell_heights;         // Height of each cell's subdomain
  int *d_cell_offsets_x;       // x0 of each cell's bbox
  int *d_cell_offsets_y;       // y0 of each cell's bbox

  bool device_arrays_dirty; // Need to re-upload cell pointers

public:
  Domain(const SimParams &p);
  ~Domain();

  // Cell management
  Cell *add_cell(float cx, float cy, float radius);
  void remove_cell(int cell_id);
  Cell *get_cell(int cell_id);
  int num_cells() const { return static_cast<int>(cells.size()); }

  // Find overlapping bounding boxes
  void update_overlap_pairs();

  // Synchronize device arrays after cell changes
  void sync_device_arrays();

  // Update all cell bounding boxes and reallocate if needed
  void update_all_bounding_boxes();

  // Initialize random cell configuration
  void initialize_random_cells(int num_cells, float radius, float min_spacing);

  // Initialize from checkpoint file
  void load_checkpoint(const char *filename);
  void save_checkpoint(const char *filename) const;

private:
  void allocate_device_arrays();
  void free_device_arrays();
};

//=============================================================================
// Domain Implementation
//=============================================================================

inline Domain::Domain(const SimParams &p)
    : params(p), next_cell_id(0), d_cell_phi_ptrs(nullptr),
      d_cell_dphi_dt_ptrs(nullptr), d_cell_widths(nullptr),
      d_cell_heights(nullptr), d_cell_offsets_x(nullptr),
      d_cell_offsets_y(nullptr), device_arrays_dirty(true) {}

inline Domain::~Domain() {
  free_device_arrays();
  cells.clear();
}

inline Cell *Domain::add_cell(float cx, float cy, float radius) {
  // Compute bounding box for this cell
  int margin =
      static_cast<int>(radius * params.subdomain_padding) + params.halo_width;
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

  device_arrays_dirty = true;

  return ptr;
}

inline void Domain::remove_cell(int cell_id) {
  auto it = std::find_if(
      cells.begin(), cells.end(),
      [cell_id](const std::unique_ptr<Cell> &c) { return c->id == cell_id; });

  if (it != cells.end()) {
    cells.erase(it);
    device_arrays_dirty = true;
  }
}

inline Cell *Domain::get_cell(int cell_id) {
  auto it = std::find_if(
      cells.begin(), cells.end(),
      [cell_id](const std::unique_ptr<Cell> &c) { return c->id == cell_id; });

  return (it != cells.end()) ? it->get() : nullptr;
}

inline void Domain::update_overlap_pairs() {
  overlap_pairs.clear();

  int n = num_cells();
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (cells[i]->bbox_with_halo.overlaps(cells[j]->bbox_with_halo, params.Nx,
                                            params.Ny)) {
        OverlapPair pair;
        pair.cell_i = cells[i]->id;
        pair.cell_j = cells[j]->id;
        // TODO: Compute actual overlap region for optimized interaction
        overlap_pairs.push_back(pair);
      }
    }
  }
}

inline void Domain::allocate_device_arrays() {
  int n = num_cells();
  if (n == 0)
    return;

  cudaMalloc(&d_cell_phi_ptrs, n * sizeof(float *));
  cudaMalloc(&d_cell_dphi_dt_ptrs, n * sizeof(float *));
  cudaMalloc(&d_cell_widths, n * sizeof(int));
  cudaMalloc(&d_cell_heights, n * sizeof(int));
  cudaMalloc(&d_cell_offsets_x, n * sizeof(int));
  cudaMalloc(&d_cell_offsets_y, n * sizeof(int));
}

inline void Domain::free_device_arrays() {
  if (d_cell_phi_ptrs) {
    cudaFree(d_cell_phi_ptrs);
    d_cell_phi_ptrs = nullptr;
  }
  if (d_cell_dphi_dt_ptrs) {
    cudaFree(d_cell_dphi_dt_ptrs);
    d_cell_dphi_dt_ptrs = nullptr;
  }
  if (d_cell_widths) {
    cudaFree(d_cell_widths);
    d_cell_widths = nullptr;
  }
  if (d_cell_heights) {
    cudaFree(d_cell_heights);
    d_cell_heights = nullptr;
  }
  if (d_cell_offsets_x) {
    cudaFree(d_cell_offsets_x);
    d_cell_offsets_x = nullptr;
  }
  if (d_cell_offsets_y) {
    cudaFree(d_cell_offsets_y);
    d_cell_offsets_y = nullptr;
  }
}

inline void Domain::sync_device_arrays() {
  if (!device_arrays_dirty)
    return;

  free_device_arrays();
  allocate_device_arrays();

  int n = num_cells();
  if (n == 0)
    return;

  // Collect host data
  std::vector<float *> h_phi_ptrs(n);
  std::vector<float *> h_dphi_dt_ptrs(n);
  std::vector<int> h_widths(n);
  std::vector<int> h_heights(n);
  std::vector<int> h_offsets_x(n);
  std::vector<int> h_offsets_y(n);

  for (int i = 0; i < n; ++i) {
    h_phi_ptrs[i] = cells[i]->d_phi;
    h_dphi_dt_ptrs[i] = cells[i]->d_dphi_dt;
    h_widths[i] = cells[i]->width();
    h_heights[i] = cells[i]->height();
    h_offsets_x[i] = cells[i]->bbox_with_halo.x0;
    h_offsets_y[i] = cells[i]->bbox_with_halo.y0;
  }

  // Upload to device
  cudaMemcpy(d_cell_phi_ptrs, h_phi_ptrs.data(), n * sizeof(float *),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_dphi_dt_ptrs, h_dphi_dt_ptrs.data(), n * sizeof(float *),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_widths, h_widths.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_heights, h_heights.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_offsets_x, h_offsets_x.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_offsets_y, h_offsets_y.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);

  device_arrays_dirty = false;
}

inline void Domain::initialize_random_cells(int num_cells_to_add, float radius,
                                            float min_spacing) {
  // For single cell, place at domain center to avoid boundary issues
  if (num_cells_to_add == 1) {
    float cx = params.Nx / 2.0f;
    float cy = params.Ny / 2.0f;
    add_cell(cx, cy, radius);
    update_overlap_pairs();
    sync_device_arrays();
    return;
  }

  // Pure random placement with rejection sampling
  // This ensures truly random configurations for ensemble statistics
  std::vector<Vec2> centers;
  int max_attempts = 10000;
  float current_spacing = min_spacing;

  while (static_cast<int>(centers.size()) < num_cells_to_add) {
    bool placed = false;

    for (int attempt = 0; attempt < max_attempts && !placed; ++attempt) {
      // Place anywhere in domain (periodic boundaries handle wrapping)
      float cx = static_cast<float>(rand()) / RAND_MAX * params.Nx;
      float cy = static_cast<float>(rand()) / RAND_MAX * params.Ny;

      // Check distance to all existing centers
      bool valid = true;
      for (const auto &c : centers) {
        float dx = fabsf(cx - c.x);
        float dy = fabsf(cy - c.y);

        // Periodic distance
        if (dx > params.Nx * 0.5f)
          dx = params.Nx - dx;
        if (dy > params.Ny * 0.5f)
          dy = params.Ny - dy;

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
      // Reduce spacing and try again
      current_spacing *= 0.95f;
      if (current_spacing < radius) {
        printf("Warning: Could only place %d of %d cells (reduced spacing to "
               "%.1f)\n",
               static_cast<int>(centers.size()), num_cells_to_add,
               current_spacing);
        break;
      }
    }
  }

  update_overlap_pairs();
  sync_device_arrays();
}

} // namespace cellsim
