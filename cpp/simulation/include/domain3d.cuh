#pragma once

#include "cell3d.cuh"
#include "types3d.cuh"
#include <algorithm>
#include <memory>
#include <vector>

namespace cellsim {

//=============================================================================
// OverlapPair3D - Tracks which cells have overlapping bounding boxes
//=============================================================================

struct OverlapPair3D {
  int cell_i;
  int cell_j;
  BoundingBox3D overlap_region; // Intersection of the two bounding boxes
};

//=============================================================================
// Domain3D - Manages global 3D domain and cell collection
//=============================================================================

class Domain3D {
public:
  SimParams3D params;

  // Cell collection
  std::vector<std::unique_ptr<Cell3D>> cells;
  int next_cell_id;

  // Interaction tracking
  std::vector<OverlapPair3D> overlap_pairs;

  // Device arrays for batch operations
  float **d_cell_phi_ptrs;     // Array of pointers to each cell's φ
  int *d_cell_widths;          // Width of each cell's subdomain
  int *d_cell_heights;         // Height of each cell's subdomain
  int *d_cell_depths;          // Depth of each cell's subdomain
  int *d_cell_offsets_x;       // x0 of each cell's bbox
  int *d_cell_offsets_y;       // y0 of each cell's bbox
  int *d_cell_offsets_z;       // z0 of each cell's bbox

  bool device_arrays_dirty; // Need to re-upload cell pointers

public:
  Domain3D(const SimParams3D &p);
  ~Domain3D();

  // Non-copyable, but movable
  Domain3D(const Domain3D &) = delete;
  Domain3D &operator=(const Domain3D &) = delete;
  Domain3D(Domain3D &&other) noexcept;
  Domain3D &operator=(Domain3D &&other) noexcept;

  // Cell management
  Cell3D *add_cell(float cx, float cy, float cz, float radius);
  Cell3D *add_cell(std::unique_ptr<Cell3D> cell); // For checkpoint loading
  void remove_cell(int cell_id);
  Cell3D *get_cell(int cell_id);
  int num_cells() const { return static_cast<int>(cells.size()); }

  // Calculate total GPU memory usage for all cells
  size_t total_gpu_memory_bytes() const {
    size_t total = 0;
    for (const auto &cell : cells) {
      // Each cell has: d_phi only (work buffers managed by Integrator3D)
      size_t cell_bytes = cell->bbox_with_halo.size() * sizeof(float);
      total += cell_bytes;
    }
    return total;
  }

  // Find overlapping bounding boxes
  void update_overlap_pairs();

  // Synchronize device arrays after cell changes
  void sync_device_arrays();

  // Update all cell bounding boxes and reallocate if needed
  void update_all_bounding_boxes();

  // Compute properties for all cells
  void compute_all_properties();

  // Initialize random cell configuration
  void initialize_random_cells(int num_cells, float radius, float min_spacing);

  // Initialize cells on a regular grid for target confluence
  // confluence: target packing fraction (0-1), e.g., 0.85 for 85%
  void initialize_grid(int num_cells, float radius, float confluence);

private:
  void allocate_device_arrays();
  void free_device_arrays();
};

//=============================================================================
// Domain3D Implementation
//=============================================================================

inline Domain3D::Domain3D(const SimParams3D &p)
    : params(p), next_cell_id(0), d_cell_phi_ptrs(nullptr),
      d_cell_widths(nullptr),
      d_cell_heights(nullptr), d_cell_depths(nullptr),
      d_cell_offsets_x(nullptr), d_cell_offsets_y(nullptr),
      d_cell_offsets_z(nullptr), device_arrays_dirty(true) {}

inline Domain3D::Domain3D(Domain3D &&other) noexcept
    : params(other.params), cells(std::move(other.cells)),
      next_cell_id(other.next_cell_id), d_cell_phi_ptrs(other.d_cell_phi_ptrs),
      d_cell_widths(other.d_cell_widths), d_cell_heights(other.d_cell_heights),
      d_cell_depths(other.d_cell_depths),
      d_cell_offsets_x(other.d_cell_offsets_x),
      d_cell_offsets_y(other.d_cell_offsets_y),
      d_cell_offsets_z(other.d_cell_offsets_z),
      device_arrays_dirty(other.device_arrays_dirty) {
  // Clear other's pointers so destructor doesn't free
  other.d_cell_phi_ptrs = nullptr;
  other.d_cell_widths = nullptr;
  other.d_cell_heights = nullptr;
  other.d_cell_depths = nullptr;
  other.d_cell_offsets_x = nullptr;
  other.d_cell_offsets_y = nullptr;
  other.d_cell_offsets_z = nullptr;
}

inline Domain3D &Domain3D::operator=(Domain3D &&other) noexcept {
  if (this != &other) {
    free_device_arrays();
    cells.clear();

    params = other.params;
    cells = std::move(other.cells);
    next_cell_id = other.next_cell_id;
    d_cell_phi_ptrs = other.d_cell_phi_ptrs;
    d_cell_widths = other.d_cell_widths;
    d_cell_heights = other.d_cell_heights;
    d_cell_depths = other.d_cell_depths;
    d_cell_offsets_x = other.d_cell_offsets_x;
    d_cell_offsets_y = other.d_cell_offsets_y;
    d_cell_offsets_z = other.d_cell_offsets_z;
    device_arrays_dirty = other.device_arrays_dirty;

    other.d_cell_phi_ptrs = nullptr;
    other.d_cell_widths = nullptr;
    other.d_cell_heights = nullptr;
    other.d_cell_depths = nullptr;
    other.d_cell_offsets_x = nullptr;
    other.d_cell_offsets_y = nullptr;
    other.d_cell_offsets_z = nullptr;
  }
  return *this;
}

inline Domain3D::~Domain3D() {
  free_device_arrays();
  cells.clear();
}

inline Cell3D *Domain3D::add_cell(float cx, float cy, float cz, float radius) {
  // Compute bounding box for this cell
  int margin =
      static_cast<int>(radius * params.subdomain_padding) + params.halo_width;
  BoundingBox3D bbox = {
      static_cast<int>(cx) - margin, static_cast<int>(cy) - margin,
      static_cast<int>(cz) - margin, static_cast<int>(cx) + margin,
      static_cast<int>(cy) + margin, static_cast<int>(cz) + margin};

  // Ensure minimum size
  auto ensure_min = [&](int &lo, int &hi) {
    int size = hi - lo;
    if (size < params.min_subdomain_size) {
      int expand = (params.min_subdomain_size - size) / 2 + 1;
      lo -= expand;
      hi += expand;
    }
  };
  ensure_min(bbox.x0, bbox.x1);
  ensure_min(bbox.y0, bbox.y1);
  ensure_min(bbox.z0, bbox.z1);

  // Create cell
  auto cell = std::make_unique<Cell3D>(next_cell_id++, bbox, params.halo_width);
  cell->initialize_spherical(cx, cy, cz, radius, params);

  Cell3D *ptr = cell.get();
  cells.push_back(std::move(cell));

  device_arrays_dirty = true;

  return ptr;
}

inline Cell3D *Domain3D::add_cell(std::unique_ptr<Cell3D> cell) {
  // For checkpoint loading - cell is already fully initialized
  Cell3D *ptr = cell.get();
  cells.push_back(std::move(cell));
  device_arrays_dirty = true;
  return ptr;
}

inline void Domain3D::remove_cell(int cell_id) {
  auto it = std::find_if(
      cells.begin(), cells.end(),
      [cell_id](const std::unique_ptr<Cell3D> &c) { return c->id == cell_id; });

  if (it != cells.end()) {
    cells.erase(it);
    device_arrays_dirty = true;
  }
}

inline Cell3D *Domain3D::get_cell(int cell_id) {
  auto it = std::find_if(
      cells.begin(), cells.end(),
      [cell_id](const std::unique_ptr<Cell3D> &c) { return c->id == cell_id; });

  return (it != cells.end()) ? it->get() : nullptr;
}

inline void Domain3D::update_overlap_pairs() {
  overlap_pairs.clear();

  int n = num_cells();
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (cells[i]->bbox_with_halo.overlaps(cells[j]->bbox_with_halo, params.Nx,
                                            params.Ny, params.Nz)) {
        OverlapPair3D pair;
        pair.cell_i = cells[i]->id;
        pair.cell_j = cells[j]->id;
        overlap_pairs.push_back(pair);
      }
    }
  }
}

inline void Domain3D::allocate_device_arrays() {
  int n = num_cells();
  if (n == 0)
    return;

  cudaMalloc(&d_cell_phi_ptrs, n * sizeof(float *));
  cudaMalloc(&d_cell_widths, n * sizeof(int));
  cudaMalloc(&d_cell_heights, n * sizeof(int));
  cudaMalloc(&d_cell_depths, n * sizeof(int));
  cudaMalloc(&d_cell_offsets_x, n * sizeof(int));
  cudaMalloc(&d_cell_offsets_y, n * sizeof(int));
  cudaMalloc(&d_cell_offsets_z, n * sizeof(int));
}

inline void Domain3D::free_device_arrays() {
  if (d_cell_phi_ptrs) {
    cudaFree(d_cell_phi_ptrs);
    d_cell_phi_ptrs = nullptr;
  }
  if (d_cell_widths) {
    cudaFree(d_cell_widths);
    d_cell_widths = nullptr;
  }
  if (d_cell_heights) {
    cudaFree(d_cell_heights);
    d_cell_heights = nullptr;
  }
  if (d_cell_depths) {
    cudaFree(d_cell_depths);
    d_cell_depths = nullptr;
  }
  if (d_cell_offsets_x) {
    cudaFree(d_cell_offsets_x);
    d_cell_offsets_x = nullptr;
  }
  if (d_cell_offsets_y) {
    cudaFree(d_cell_offsets_y);
    d_cell_offsets_y = nullptr;
  }
  if (d_cell_offsets_z) {
    cudaFree(d_cell_offsets_z);
    d_cell_offsets_z = nullptr;
  }
}

inline void Domain3D::sync_device_arrays() {
  if (!device_arrays_dirty)
    return;

  free_device_arrays();
  allocate_device_arrays();

  int n = num_cells();
  if (n == 0)
    return;

  std::vector<float *> phi_ptrs(n);
  std::vector<int> widths(n), heights(n), depths(n);
  std::vector<int> offsets_x(n), offsets_y(n), offsets_z(n);

  for (int i = 0; i < n; ++i) {
    phi_ptrs[i] = cells[i]->d_phi;
    widths[i] = cells[i]->width();
    heights[i] = cells[i]->height();
    depths[i] = cells[i]->depth();
    offsets_x[i] = cells[i]->bbox_with_halo.x0;
    offsets_y[i] = cells[i]->bbox_with_halo.y0;
    offsets_z[i] = cells[i]->bbox_with_halo.z0;
  }

  cudaMemcpy(d_cell_phi_ptrs, phi_ptrs.data(), n * sizeof(float *),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_widths, widths.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_heights, heights.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_depths, depths.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_offsets_x, offsets_x.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_offsets_y, offsets_y.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell_offsets_z, offsets_z.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);

  device_arrays_dirty = false;
}

inline void Domain3D::update_all_bounding_boxes() {
  bool any_changed = false;
  for (auto &cell : cells) {
    if (cell->update_bounding_box(params)) {
      any_changed = true;
    }
  }
  if (any_changed) {
    device_arrays_dirty = true;
    sync_device_arrays();
    update_overlap_pairs();
  }
}

inline void Domain3D::compute_all_properties() {
  for (auto &cell : cells) {
    cell->compute_properties(params);
  }
}

inline void Domain3D::initialize_random_cells(int num_cells_to_add,
                                              float radius, float min_spacing) {
  // For single cell, place at domain center to avoid boundary issues
  if (num_cells_to_add == 1) {
    float cx = params.Nx / 2.0f;
    float cy = params.Ny / 2.0f;
    float cz = params.Nz / 2.0f;
    add_cell(cx, cy, cz, radius);
    update_overlap_pairs();
    sync_device_arrays();
    printf("Initialized 1 3D cell at center\n");
    printf("  Cell 0: center=(%.1f, %.1f, %.1f)\n", cx, cy, cz);
    return;
  }

  // Place cells randomly with minimum spacing
  // For high confluence (negative min_spacing), allow overlap
  int max_attempts = 10000;
  int placed = 0;

  // Minimum distance between cell centers
  // If min_spacing >= 0: cells don't touch (gap = min_spacing)
  // If min_spacing < 0: cells can overlap (overlap = -min_spacing)
  float effective_min_dist = 2 * radius + min_spacing;
  // For high confluence, allow more overlap - floor at 0.75 * diameter
  // This allows cells to overlap by 25% of their diameter
  float min_allowed = 1.5f * radius; // Centers at least 0.75*diameter apart
  if (effective_min_dist < min_allowed) {
    effective_min_dist = min_allowed;
  }

  printf("Random initialization:\n");
  printf("  Placing %d cells with target min distance=%.2f\n", num_cells_to_add,
         effective_min_dist);

  // Adaptive placement: try with strict distance, relax if needed
  float current_min_dist = effective_min_dist;
  float absolute_min_dist =
      radius * 1.2f; // Never allow centers closer than 1.2*R

  while (placed < num_cells_to_add) {
    bool success = false;
    int attempts_at_this_level = 0;
    int max_attempts_per_level = 5000;

    while (!success && attempts_at_this_level < max_attempts_per_level) {
      attempts_at_this_level++;

      float cx = static_cast<float>(rand()) / RAND_MAX * params.Nx;
      float cy = static_cast<float>(rand()) / RAND_MAX * params.Ny;
      float cz = static_cast<float>(rand()) / RAND_MAX * params.Nz;

      // Check distance to all existing cells
      bool too_close = false;
      for (const auto &cell : cells) {
        float dx = cx - cell->centroid.x;
        float dy = cy - cell->centroid.y;
        float dz = cz - cell->centroid.z;

        // Periodic distance
        if (dx > params.Nx * 0.5f)
          dx -= params.Nx;
        if (dx < -params.Nx * 0.5f)
          dx += params.Nx;
        if (dy > params.Ny * 0.5f)
          dy -= params.Ny;
        if (dy < -params.Ny * 0.5f)
          dy += params.Ny;
        if (dz > params.Nz * 0.5f)
          dz -= params.Nz;
        if (dz < -params.Nz * 0.5f)
          dz += params.Nz;

        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (dist < current_min_dist) {
          too_close = true;
          break;
        }
      }

      if (!too_close) {
        add_cell(cx, cy, cz, radius);
        placed++;
        success = true;
      }
    }

    if (!success) {
      // Relax the distance constraint and try again
      if (current_min_dist > absolute_min_dist) {
        float old_dist = current_min_dist;
        current_min_dist *= 0.9f; // Reduce by 10%
        if (current_min_dist < absolute_min_dist) {
          current_min_dist = absolute_min_dist;
        }
        printf("  Relaxing min_dist: %.1f -> %.1f (placed %d/%d)\n", old_dist,
               current_min_dist, placed, num_cells_to_add);
      } else {
        printf("Warning: Could not place cell %d even at min distance %.1f\n",
               placed, current_min_dist);
        break;
      }
    }
  }

  printf("  Placed %d cells (final min_dist=%.1f)\n", placed, current_min_dist);

  sync_device_arrays();
  update_overlap_pairs();
}

inline void Domain3D::initialize_grid(int num_cells, float radius,
                                      float confluence) {
  // Calculate domain volume and required cell volume for target confluence
  float domain_volume = static_cast<float>(params.Nx) * params.Ny * params.Nz;
  float cell_volume = (4.0f / 3.0f) * M_PI * radius * radius * radius;
  float total_cell_volume = num_cells * cell_volume;

  // Calculate actual confluence we'll achieve
  float actual_confluence = total_cell_volume / domain_volume;
  printf("Grid initialization:\n");
  printf("  Domain: %d x %d x %d (volume=%.0f)\n", params.Nx, params.Ny,
         params.Nz, domain_volume);
  printf("  Cells: %d, radius=%.1f (cell volume=%.1f)\n", num_cells, radius,
         cell_volume);
  printf("  Target confluence: %.1f%%, Actual: %.1f%%\n", confluence * 100.0f,
         actual_confluence * 100.0f);

  // Determine grid dimensions (try to make it as cubic as possible)
  // Find nx, ny, nz such that nx*ny*nz >= num_cells and grid is roughly cubic
  int nx = 1, ny = 1, nz = 1;

  // Start with cube root approximation
  float cube_root = cbrtf(static_cast<float>(num_cells));
  nx = static_cast<int>(ceilf(cube_root));
  ny = nx;
  nz = nx;

  // Reduce until we have just enough cells
  while (nx * ny * nz > num_cells && nx > 1) {
    // Try reducing each dimension
    if ((nx - 1) * ny * nz >= num_cells) {
      nx--;
    } else if (nx * (ny - 1) * nz >= num_cells) {
      ny--;
    } else if (nx * ny * (nz - 1) >= num_cells) {
      nz--;
    } else {
      break;
    }
  }

  // Ensure we have enough spots
  while (nx * ny * nz < num_cells) {
    // Increase the smallest dimension
    if (nx <= ny && nx <= nz)
      nx++;
    else if (ny <= nz)
      ny++;
    else
      nz++;
  }

  printf("  Grid layout: %d x %d x %d = %d spots\n", nx, ny, nz, nx * ny * nz);

  // Calculate spacing between cell centers
  float spacing_x = static_cast<float>(params.Nx) / nx;
  float spacing_y = static_cast<float>(params.Ny) / ny;
  float spacing_z = static_cast<float>(params.Nz) / nz;

  printf("  Cell spacing: (%.2f, %.2f, %.2f)\n", spacing_x, spacing_y,
         spacing_z);
  printf("  Cell diameter: %.1f, Min gap: %.2f\n", 2.0f * radius,
         fminf(fminf(spacing_x, spacing_y), spacing_z) - 2.0f * radius);

  // Place cells on the grid
  int placed = 0;
  for (int iz = 0; iz < nz && placed < num_cells; ++iz) {
    for (int iy = 0; iy < ny && placed < num_cells; ++iy) {
      for (int ix = 0; ix < nx && placed < num_cells; ++ix) {
        // Center of this grid cell, offset by half spacing
        float cx = (ix + 0.5f) * spacing_x;
        float cy = (iy + 0.5f) * spacing_y;
        float cz = (iz + 0.5f) * spacing_z;

        add_cell(cx, cy, cz, radius);
        placed++;
      }
    }
  }

  printf("  Placed %d cells\n", placed);

  sync_device_arrays();
  update_overlap_pairs();
}

} // namespace cellsim
