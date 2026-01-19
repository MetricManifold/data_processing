#include "io.hpp"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace cellsim {

void save_checkpoint(const Domain &domain, const std::string &filename,
                     const CheckpointHeader &header) {
  // Write to temporary file first, then rename atomically
  std::string temp_filename = filename + ".tmp";

  std::ofstream file(temp_filename, std::ios::binary);
  if (!file.is_open()) {
    printf("ERROR: Could not open checkpoint file for writing: %s\n",
           temp_filename.c_str());
    return;
  }

  // Create header copy with correct num_cells
  CheckpointHeader hdr = header;
  hdr.magic = 0x43454C4C;
  hdr.version = 4;
  hdr.num_cells = domain.num_cells();
  hdr.sim_params_size = sizeof(SimParams);
  file.write(reinterpret_cast<const char *>(&hdr), sizeof(CheckpointHeader));

  // Write params
  file.write(reinterpret_cast<const char *>(&domain.params), sizeof(SimParams));

  // Write each cell
  for (const auto &cell : domain.cells) {
    file.write(reinterpret_cast<const char *>(&cell->id), sizeof(int));
    file.write(reinterpret_cast<const char *>(&cell->bbox), sizeof(BoundingBox));
    file.write(reinterpret_cast<const char *>(&cell->centroid), sizeof(Vec2));
    file.write(reinterpret_cast<const char *>(&cell->velocity), sizeof(Vec2));
    file.write(reinterpret_cast<const char *>(&cell->volume), sizeof(float));

    // Write field
    file.write(reinterpret_cast<const char *>(cell->phi.data()),
               cell->field_size * sizeof(float));
  }

  file.close();

  // Atomic rename
#ifdef _WIN32
  std::remove(filename.c_str());
#endif
  if (std::rename(temp_filename.c_str(), filename.c_str()) != 0) {
    printf("ERROR: Could not rename checkpoint file\n");
    return;
  }

  printf("Saved checkpoint: step=%d, t=%.4f, cells=%d\n", 
         hdr.current_step, hdr.current_time, hdr.num_cells);
}

bool load_checkpoint(Domain &domain, const std::string &filename,
                     CheckpointHeader &out_header) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    printf("Warning: Could not open checkpoint file: %s\n", filename.c_str());
    return false;
  }

  // Read minimum header to get version
  struct MinHeader {
    uint32_t magic;
    uint32_t version;
  };
  MinHeader min_hdr;
  file.read(reinterpret_cast<char *>(&min_hdr), sizeof(MinHeader));

  if (min_hdr.magic != 0x43454C4C) {
    printf("Error: Invalid checkpoint file (bad magic number)\n");
    return false;
  }

  if (min_hdr.version < 2 || min_hdr.version > 4) {
    printf("Error: Unsupported checkpoint version %d (expected 2, 3, or 4)\n",
           min_hdr.version);
    return false;
  }

  // Seek back and read full header
  file.seekg(0);
  
  CheckpointHeader header;
  if (min_hdr.version <= 3) {
    // v2/v3 header doesn't have sim_params_size field
    size_t old_header_size = sizeof(CheckpointHeader) - sizeof(uint32_t);
    file.read(reinterpret_cast<char *>(&header), old_header_size);
    header.sim_params_size = 0;
    
    if (min_hdr.version == 2) {
      printf("Note: Loading v2 checkpoint - using default runtime options\n");
    }
    printf("Note: Loading v%d checkpoint\n", min_hdr.version);
  } else {
    // v4: Check for old format with padding
    file.seekg(36);
    uint32_t val_at_36, val_at_40;
    file.read(reinterpret_cast<char *>(&val_at_36), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&val_at_40), sizeof(uint32_t));
    file.seekg(0);

    bool is_old_format = (val_at_36 == 0 && val_at_40 == sizeof(SimParams));

    if (is_old_format) {
      printf("Note: Loading old v4 checkpoint format\n");
      file.read(reinterpret_cast<char *>(&header), 36);
      file.seekg(40);
      file.read(reinterpret_cast<char *>(&header.sim_params_size), sizeof(uint32_t));
    } else {
      file.read(reinterpret_cast<char *>(&header), sizeof(CheckpointHeader));
    }
  }

  out_header = header;
  int num_cells = header.num_cells;

  // Handle SimParams size mismatch
  size_t old_sim_params_size = sizeof(SimParams) - sizeof(SimParams::MotilityModel);

  if (header.version <= 3 || header.sim_params_size == 0) {
    file.read(reinterpret_cast<char *>(&domain.params), old_sim_params_size);
    domain.params.motility_model = SimParams::MotilityModel::RunAndTumble;
  } else if (header.sim_params_size != sizeof(SimParams)) {
    printf("Warning: SimParams size mismatch (file: %u, current: %zu)\n",
           header.sim_params_size, sizeof(SimParams));
    size_t read_size = std::min((size_t)header.sim_params_size, sizeof(SimParams));
    file.read(reinterpret_cast<char *>(&domain.params), read_size);
    if (header.sim_params_size > sizeof(SimParams)) {
      file.seekg(header.sim_params_size - sizeof(SimParams), std::ios::cur);
    }
  } else {
    file.read(reinterpret_cast<char *>(&domain.params), sizeof(SimParams));
  }

  // Validate domain size
  const size_t MAX_DOMAIN_PIXELS = 65536ULL * 65536ULL;
  size_t domain_pixels = (size_t)domain.params.Nx * (size_t)domain.params.Ny;
  if (domain_pixels > MAX_DOMAIN_PIXELS || domain.params.Nx <= 0 || domain.params.Ny <= 0) {
    printf("Error: Invalid domain size Nx=%d, Ny=%d\n", 
           domain.params.Nx, domain.params.Ny);
    return false;
  }

  // Clear existing cells
  domain.cells.clear();

  // Read each cell
  for (int i = 0; i < num_cells; ++i) {
    int id;
    BoundingBox bbox;
    Vec2 centroid, velocity;
    float volume;

    file.read(reinterpret_cast<char *>(&id), sizeof(int));
    file.read(reinterpret_cast<char *>(&bbox), sizeof(BoundingBox));
    file.read(reinterpret_cast<char *>(&centroid), sizeof(Vec2));
    file.read(reinterpret_cast<char *>(&velocity), sizeof(Vec2));
    file.read(reinterpret_cast<char *>(&volume), sizeof(float));

    auto cell = std::make_unique<Cell>(id, bbox, domain.params.halo_width);
    cell->centroid = centroid;
    cell->velocity = velocity;
    cell->volume = volume;

    // Read field
    file.read(reinterpret_cast<char *>(cell->phi.data()),
              cell->field_size * sizeof(float));

    domain.cells.push_back(std::move(cell));
    domain.next_cell_id = std::max(domain.next_cell_id, id + 1);
  }

  file.close();

  printf("Loaded checkpoint: step=%d, t=%.4f, cells=%d\n",
         out_header.current_step, out_header.current_time, num_cells);
  return true;
}

void export_vtk(const Domain &domain, const std::string &base_filename, int frame) {
  std::stringstream ss;
  ss << base_filename << "_" << std::setfill('0') << std::setw(6) << frame << ".vtk";
  std::string filename = ss.str();

  int Nx = domain.params.Nx;
  int Ny = domain.params.Ny;

  // Reconstruct full field
  std::vector<float> full_field(Nx * Ny, 0.0f);

  for (const auto &cell : domain.cells) {
    int halo = domain.params.halo_width;

    for (int ly = halo; ly < cell->height() - halo; ++ly) {
      for (int lx = halo; lx < cell->width() - halo; ++lx) {
        int gx, gy;
        cell->bbox_with_halo.local_to_global(lx, ly, gx, gy, Nx, Ny);

        int local_idx = ly * cell->width() + lx;
        int global_idx = gy * Nx + gx;

        full_field[global_idx] = std::max(full_field[global_idx], 
                                          cell->phi[local_idx]);
      }
    }
  }

  // Write VTK file
  std::ofstream file(filename);

  file << "# vtk DataFile Version 3.0\n";
  file << "Phase field simulation frame " << frame << "\n";
  file << "ASCII\n";
  file << "DATASET STRUCTURED_POINTS\n";
  file << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
  file << "ORIGIN 0 0 0\n";
  file << "SPACING " << domain.params.dx << " " << domain.params.dy << " 1\n";
  file << "POINT_DATA " << (Nx * Ny) << "\n";
  file << "SCALARS phi float 1\n";
  file << "LOOKUP_TABLE default\n";

  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      file << full_field[y * Nx + x] << "\n";
    }
  }

  file.close();
}

void export_trajectory(const Domain &domain, const std::string &filename,
                       float current_time) {
  std::ofstream file(filename, std::ios::app);
  file << std::fixed << std::setprecision(6);

  for (const auto &cell : domain.cells) {
    file << current_time << " " << cell->id << " "
         << cell->centroid.x << " " << cell->centroid.y << " "
         << cell->velocity.x << " " << cell->velocity.y << " "
         << cell->polarization.x << " " << cell->polarization.y << " "
         << cell->theta << "\n";
  }

  file.close();
}

} // namespace cellsim
