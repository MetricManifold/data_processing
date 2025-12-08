#include "simulation3d.cuh"
#include <cstdio>
#include <cstring>

namespace cellsim {

//=============================================================================
// Checkpoint file format for 3D
// Header:
//   - Magic number (4 bytes): "CS3D"
//   - Version (4 bytes): int
//   - Step (4 bytes): int
//   - Time (4 bytes): float
//   - Num cells (4 bytes): int
//   - SimParams3D (sizeof(SimParams3D))
// Per cell:
//   - Cell3D metadata (id, bbox, centroid, volume, polarization, velocity)
//   - Phase field data (width * height * depth * sizeof(float))
//=============================================================================

static const char MAGIC_3D[] = "CS3D";
static const int VERSION_3D = 1;

void save_checkpoint_3d(const char *filename, const Domain3D &domain, int step,
                        float time) {
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Failed to open file for writing: %s\n", filename);
    return;
  }

  // Write header
  fwrite(MAGIC_3D, 1, 4, fp);
  fwrite(&VERSION_3D, sizeof(int), 1, fp);
  fwrite(&step, sizeof(int), 1, fp);
  fwrite(&time, sizeof(float), 1, fp);
  int num_cells = domain.num_cells();
  fwrite(&num_cells, sizeof(int), 1, fp);
  fwrite(&domain.params, sizeof(SimParams3D), 1, fp);

  // Write each cell
  for (int i = 0; i < num_cells; ++i) {
    const Cell3D &cell = *domain.cells[i];

    // Write metadata
    fwrite(&cell.id, sizeof(int), 1, fp);
    fwrite(&cell.bbox, sizeof(BoundingBox3D), 1, fp);
    // Also write bbox_with_halo (so reader knows actual field dimensions)
    fwrite(&cell.bbox_with_halo, sizeof(BoundingBox3D), 1, fp);
    fwrite(&cell.centroid, sizeof(Vec3), 1, fp);
    fwrite(&cell.volume, sizeof(float), 1, fp);
    fwrite(&cell.theta, sizeof(float), 1, fp);
    fwrite(&cell.phi_pol, sizeof(float), 1, fp);
    fwrite(&cell.polarization, sizeof(Vec3), 1, fp);
    fwrite(&cell.velocity, sizeof(Vec3), 1, fp);

    // Download and write field data
    int size = cell.width() * cell.height() * cell.depth();
    std::vector<float> host_phi(size);
    cudaMemcpy(host_phi.data(), cell.d_phi, size * sizeof(float),
               cudaMemcpyDeviceToHost);
    fwrite(host_phi.data(), sizeof(float), size, fp);
  }

  fclose(fp);
}

// Scan checkpoint file to get memory requirements without allocating
// Returns total GPU memory needed in bytes, or 0 on error
size_t scan_checkpoint_3d_memory(const char *filename, int &out_num_cells) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    return 0;
  }

  // Read and verify header
  char magic[4];
  fread(magic, 1, 4, fp);
  if (memcmp(magic, MAGIC_3D, 4) != 0) {
    fclose(fp);
    return 0;
  }

  int version;
  fread(&version, sizeof(int), 1, fp);
  if (version != VERSION_3D) {
    fclose(fp);
    return 0;
  }

  int step;
  float time;
  fread(&step, sizeof(int), 1, fp);
  fread(&time, sizeof(float), 1, fp);

  int num_cells;
  fread(&num_cells, sizeof(int), 1, fp);
  out_num_cells = num_cells;

  // Skip SimParams3D
  fseek(fp, sizeof(SimParams3D), SEEK_CUR);

  size_t total_memory = 0;

  // Scan each cell's bounding box to calculate memory
  for (int i = 0; i < num_cells; ++i) {
    // Skip id
    fseek(fp, sizeof(int), SEEK_CUR);

    // Skip bbox
    fseek(fp, sizeof(BoundingBox3D), SEEK_CUR);

    // Read bbox_with_halo to get dimensions
    BoundingBox3D bbox_with_halo;
    fread(&bbox_with_halo, sizeof(BoundingBox3D), 1, fp);

    int w = bbox_with_halo.width();
    int h = bbox_with_halo.height();
    int d = bbox_with_halo.depth();
    int size = w * h * d;

    // Each cell has 1 buffer: d_phi (d_dphi_dt removed for memory optimization)
    total_memory += size * sizeof(float);

    // Skip remaining metadata (centroid, volume, theta, phi_pol, polarization,
    // velocity)
    fseek(fp, sizeof(Vec3) + sizeof(float) * 3 + sizeof(Vec3) * 2, SEEK_CUR);

    // Skip field data
    fseek(fp, size * sizeof(float), SEEK_CUR);
  }

  fclose(fp);
  return total_memory;
}

bool load_checkpoint_3d(const char *filename, Domain3D &domain, int &step,
                        float &time) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Failed to open file for reading: %s\n", filename);
    return false;
  }

  // Read and verify header
  char magic[4];
  fread(magic, 1, 4, fp);
  if (memcmp(magic, MAGIC_3D, 4) != 0) {
    fprintf(stderr, "Invalid checkpoint file magic\n");
    fclose(fp);
    return false;
  }

  int version;
  fread(&version, sizeof(int), 1, fp);
  if (version != VERSION_3D) {
    fprintf(stderr, "Unsupported checkpoint version: %d\n", version);
    fclose(fp);
    return false;
  }

  fread(&step, sizeof(int), 1, fp);
  fread(&time, sizeof(float), 1, fp);

  int num_cells;
  fread(&num_cells, sizeof(int), 1, fp);

  SimParams3D params;
  fread(&params, sizeof(SimParams3D), 1, fp);
  domain = Domain3D(params);

  // Read cells
  for (int i = 0; i < num_cells; ++i) {
    int id;
    BoundingBox3D bbox;
    BoundingBox3D bbox_with_halo;
    Vec3 centroid;
    float volume, theta, phi_pol;
    Vec3 polarization, velocity;

    fread(&id, sizeof(int), 1, fp);
    fread(&bbox, sizeof(BoundingBox3D), 1, fp);
    fread(&bbox_with_halo, sizeof(BoundingBox3D), 1, fp);
    fread(&centroid, sizeof(Vec3), 1, fp);
    fread(&volume, sizeof(float), 1, fp);
    fread(&theta, sizeof(float), 1, fp);
    fread(&phi_pol, sizeof(float), 1, fp);
    fread(&polarization, sizeof(Vec3), 1, fp);
    fread(&velocity, sizeof(Vec3), 1, fp);

    // Create cell with saved bounding boxes
    // Note: We use bbox_with_halo from the file to ensure correct sizing
    auto cell = std::make_unique<Cell3D>(id, bbox, bbox_with_halo);
    cell->centroid = centroid;
    cell->volume = volume;
    cell->theta = theta;
    cell->phi_pol = phi_pol;
    cell->polarization = polarization;
    cell->velocity = velocity;

    // Read field data - use size from the stored bbox_with_halo
    int size = bbox_with_halo.size();
    std::vector<float> host_phi(size);
    fread(host_phi.data(), sizeof(float), size, fp);
    cudaMemcpy(cell->d_phi, host_phi.data(), size * sizeof(float),
               cudaMemcpyHostToDevice);

    domain.add_cell(std::move(cell));
  }

  fclose(fp);
  printf("Loaded 3D checkpoint: step=%d, t=%.4f, cells=%d\n", step, time,
         num_cells);
  return true;
}

//=============================================================================
// VTK export for 3D
//=============================================================================

void save_vtk_3d(const char *filename, const Domain3D &domain) {
  // Allocate full domain array
  const auto &p = domain.params;
  int total = p.Nx * p.Ny * p.Nz;
  std::vector<float> phi_combined(total, 0.0f);
  std::vector<int> cell_ids(total, -1);

  // Combine all cells
  for (int c = 0; c < domain.num_cells(); ++c) {
    const Cell3D &cell = *domain.cells[c];
    std::vector<float> host_phi(cell.width() * cell.height() * cell.depth());
    cudaMemcpy(host_phi.data(), cell.d_phi, host_phi.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int lz = 0; lz < cell.depth(); ++lz) {
      for (int ly = 0; ly < cell.height(); ++ly) {
        for (int lx = 0; lx < cell.width(); ++lx) {
          int gx = (cell.bbox.x0 + lx) % p.Nx;
          int gy = (cell.bbox.y0 + ly) % p.Ny;
          int gz = (cell.bbox.z0 + lz) % p.Nz;
          if (gx < 0)
            gx += p.Nx;
          if (gy < 0)
            gy += p.Ny;
          if (gz < 0)
            gz += p.Nz;

          int global_idx = gx + gy * p.Nx + gz * p.Nx * p.Ny;
          int local_idx =
              lx + ly * cell.width() + lz * cell.width() * cell.height();

          float val = host_phi[local_idx];
          if (val > phi_combined[global_idx]) {
            phi_combined[global_idx] = val;
            if (val > 0.5f) {
              cell_ids[global_idx] = cell.id;
            }
          }
        }
      }
    }
  }

  // Write VTK structured points file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Failed to open VTK file: %s\n", filename);
    return;
  }

  fprintf(fp, "# vtk DataFile Version 3.0\n");
  fprintf(fp, "3D Cell Simulation\n");
  fprintf(fp, "ASCII\n");
  fprintf(fp, "DATASET STRUCTURED_POINTS\n");
  fprintf(fp, "DIMENSIONS %d %d %d\n", p.Nx, p.Ny, p.Nz);
  fprintf(fp, "ORIGIN 0 0 0\n");
  fprintf(fp, "SPACING %f %f %f\n", p.dx, p.dy, p.dz);
  fprintf(fp, "POINT_DATA %d\n", total);

  // Write phi field
  fprintf(fp, "SCALARS phi float 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (int i = 0; i < total; ++i) {
    fprintf(fp, "%.6f\n", phi_combined[i]);
  }

  // Write cell IDs
  fprintf(fp, "SCALARS cell_id int 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (int i = 0; i < total; ++i) {
    fprintf(fp, "%d\n", cell_ids[i]);
  }

  fclose(fp);
}

void save_cell_vtk_3d(const char *filename, const Cell3D &cell,
                      const SimParams3D &params) {
  int w = cell.width();
  int h = cell.height();
  int d = cell.depth();
  int total = w * h * d;

  std::vector<float> host_phi(total);
  cudaMemcpy(host_phi.data(), cell.d_phi, total * sizeof(float),
             cudaMemcpyDeviceToHost);

  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Failed to open VTK file: %s\n", filename);
    return;
  }

  fprintf(fp, "# vtk DataFile Version 3.0\n");
  fprintf(fp, "3D Cell %d\n", cell.id);
  fprintf(fp, "ASCII\n");
  fprintf(fp, "DATASET STRUCTURED_POINTS\n");
  fprintf(fp, "DIMENSIONS %d %d %d\n", w, h, d);
  fprintf(fp, "ORIGIN %f %f %f\n", cell.bbox.x0 * params.dx,
          cell.bbox.y0 * params.dy, cell.bbox.z0 * params.dz);
  fprintf(fp, "SPACING %f %f %f\n", params.dx, params.dy, params.dz);
  fprintf(fp, "POINT_DATA %d\n", total);

  fprintf(fp, "SCALARS phi float 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (int i = 0; i < total; ++i) {
    fprintf(fp, "%.6f\n", host_phi[i]);
  }

  fclose(fp);
}

//=============================================================================
// Trajectory output for 3D
//=============================================================================

void write_trajectory_header_3d(FILE *fp) {
  fprintf(fp, "# 3D Cell Trajectory Data\n");
  fprintf(fp, "# step time cell_id cx cy cz px py pz vx vy vz volume\n");
}

void write_trajectory_step_3d(FILE *fp, const Domain3D &domain, int step,
                              float time) {
  for (int i = 0; i < domain.num_cells(); ++i) {
    const Cell3D &cell = *domain.cells[i];
    fprintf(
        fp, "%d %.6f %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
        step, time, cell.id, cell.centroid.x, cell.centroid.y, cell.centroid.z,
        cell.polarization.x, cell.polarization.y, cell.polarization.z,
        cell.velocity.x, cell.velocity.y, cell.velocity.z, cell.volume);
  }
}

} // namespace cellsim
