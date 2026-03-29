#include "io.hpp"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace cellsim {

void save_vtk(const Domain &domain, int step, const std::string &output_dir) {
  std::ostringstream filename;
  filename << output_dir << "/phi_" << std::setfill('0') << std::setw(6) << step
           << ".vtk";

  std::ofstream file(filename.str(), std::ios::binary);
  if (!file) {
    std::cerr << "Error: Could not open " << filename.str() << " for writing"
              << std::endl;
    return;
  }

  const SimParams &params = domain.get_params();
  int Nx = params.Nx;
  int Ny = params.Ny;

  // Assemble global fields
  std::vector<float> phi_sum, phi2_sum;
  domain.assemble_global_fields(phi_sum, phi2_sum);

  // Write VTK header
  file << "# vtk DataFile Version 3.0\n";
  file << "Phase field step " << step << "\n";
  file << "BINARY\n";
  file << "DATASET STRUCTURED_POINTS\n";
  file << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
  file << "ORIGIN 0 0 0\n";
  file << "SPACING " << params.dx << " " << params.dy << " 1\n";
  file << "POINT_DATA " << (Nx * Ny) << "\n";

  // Helper to write binary float data (big-endian)
  auto write_float_array = [&](const std::vector<float> &data,
                               const std::string &name) {
    file << "SCALARS " << name << " float 1\n";
    file << "LOOKUP_TABLE default\n";

    std::vector<float> swapped(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      // Swap endianness for VTK binary format
      uint32_t val;
      std::memcpy(&val, &data[i], sizeof(float));
      val = ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
            ((val << 8) & 0xFF0000) | ((val << 24) & 0xFF000000);
      std::memcpy(&swapped[i], &val, sizeof(float));
    }
    file.write(reinterpret_cast<const char *>(swapped.data()),
               swapped.size() * sizeof(float));
  };

  write_float_array(phi_sum, "phi_sum");
  write_float_array(phi2_sum, "phi2_sum");

  // Write cell labels
  std::vector<float> labels(Nx * Ny, -1.0f);
  for (const auto &cell : domain.get_cells()) {
    const auto &bounds = cell.get_bounds();
    const float *phi = cell.get_phi();
    int local_Lx = cell.get_local_Lx();
    int local_Ly = cell.get_local_Ly();

    for (int ly = 0; ly < local_Ly; ++ly) {
      for (int lx = 0; lx < local_Lx; ++lx) {
        // Use periodic coordinate conversion
        int gx, gy;
        bounds.local_to_global(lx, ly, gx, gy, Nx, Ny);

        int local_idx = ly * local_Lx + lx;
        int global_idx = gy * Nx + gx;

        if (phi[local_idx] > 0.5f) {
          labels[global_idx] = static_cast<float>(cell.get_id());
        }
      }
    }
  }
  write_float_array(labels, "cell_label");

  file.close();
  std::cout << "Saved " << filename.str() << std::endl;
}

void save_checkpoint(const Domain &domain, int step,
                     const std::string &output_dir) {
  std::ostringstream filename;
  filename << output_dir << "/checkpoint_" << std::setfill('0') << std::setw(6)
           << step << ".bin";

  std::ofstream file(filename.str(), std::ios::binary);
  if (!file) {
    std::cerr << "Error: Could not open " << filename.str() << " for writing"
              << std::endl;
    return;
  }

  const SimParams &params = domain.get_params();
  const auto &cells = domain.get_cells();

  // Write header - matches CUDA version exactly
  CheckpointHeader header;
  header.version = CHECKPOINT_VERSION;
  header.num_cells = static_cast<int>(cells.size());
  header.current_step = step;
  header.Nx = params.Nx;
  header.Ny = params.Ny;
  header.dx = params.dx;
  header.dy = params.dy;
  header.dt = params.dt;
  header.lambda = params.lambda;
  header.gamma = params.gamma;
  header.kappa = params.kappa;
  header.mu = params.mu;
  header.target_radius = params.target_radius;
  header.v_A = params.v_A;
  header.tau = params.tau;

  file.write(reinterpret_cast<const char *>(&header), sizeof(header));

  // Write per-cell data
  for (const auto &cell : cells) {
    float cx = cell.get_cx();
    float cy = cell.get_cy();
    float vx = cell.get_vx();
    float vy = cell.get_vy();
    float target_vol = cell.get_target_volume();

    file.write(reinterpret_cast<const char *>(&cx), sizeof(float));
    file.write(reinterpret_cast<const char *>(&cy), sizeof(float));
    file.write(reinterpret_cast<const char *>(&vx), sizeof(float));
    file.write(reinterpret_cast<const char *>(&vy), sizeof(float));
    file.write(reinterpret_cast<const char *>(&target_vol), sizeof(float));

    // Write bounding box
    const auto &bounds = cell.get_bounds();
    file.write(reinterpret_cast<const char *>(&bounds), sizeof(BoundingBox));

    // Write local dimensions
    int local_Lx = cell.get_local_Lx();
    int local_Ly = cell.get_local_Ly();
    file.write(reinterpret_cast<const char *>(&local_Lx), sizeof(int));
    file.write(reinterpret_cast<const char *>(&local_Ly), sizeof(int));

    // Write phi data
    int size = local_Lx * local_Ly;
    file.write(reinterpret_cast<const char *>(cell.get_phi()),
               size * sizeof(float));
  }

  file.close();
  std::cout << "Saved checkpoint " << filename.str() << std::endl;
}

bool load_checkpoint(Domain &domain, int &step, const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file) {
    std::cerr << "Error: Could not open " << filepath << " for reading"
              << std::endl;
    return false;
  }

  // Read header
  CheckpointHeader header;
  file.read(reinterpret_cast<char *>(&header), sizeof(header));

  if (header.version != CHECKPOINT_VERSION) {
    std::cerr << "Error: Checkpoint version mismatch (got " << header.version
              << ", expected " << CHECKPOINT_VERSION << ")" << std::endl;
    return false;
  }

  // Initialize domain with parameters from checkpoint
  SimParams params;
  params.Nx = header.Nx;
  params.Ny = header.Ny;
  params.dx = header.dx;
  params.dy = header.dy;
  params.dt = header.dt;
  params.lambda = header.lambda;
  params.gamma = header.gamma;
  params.kappa = header.kappa;
  params.mu = header.mu;
  params.target_radius = header.target_radius;
  params.v_A = header.v_A;
  params.tau = header.tau;

  domain.initialize(params);

  step = header.current_step;

  // Read per-cell data
  std::vector<float> centers_x(header.num_cells);
  std::vector<float> centers_y(header.num_cells);
  std::vector<float> velocities_x(header.num_cells);
  std::vector<float> velocities_y(header.num_cells);

  // We need to load the full cell data including phi
  auto &cells = domain.get_cells();
  cells.clear();
  cells.reserve(header.num_cells);

  for (int i = 0; i < header.num_cells; ++i) {
    float cx, cy, vx, vy, target_vol;
    file.read(reinterpret_cast<char *>(&cx), sizeof(float));
    file.read(reinterpret_cast<char *>(&cy), sizeof(float));
    file.read(reinterpret_cast<char *>(&vx), sizeof(float));
    file.read(reinterpret_cast<char *>(&vy), sizeof(float));
    file.read(reinterpret_cast<char *>(&target_vol), sizeof(float));

    BoundingBox bounds;
    file.read(reinterpret_cast<char *>(&bounds), sizeof(BoundingBox));

    int local_Lx, local_Ly;
    file.read(reinterpret_cast<char *>(&local_Lx), sizeof(int));
    file.read(reinterpret_cast<char *>(&local_Ly), sizeof(int));

    // Create cell and initialize
    Cell cell;
    cell.initialize(i, cx, cy, params.target_radius, params.lambda, params.dx,
                    params.dy, params.Nx, params.Ny);
    cell.set_velocity(vx, vy);
    cell.set_target_volume(target_vol);

    // Read phi data directly
    int size = local_Lx * local_Ly;
    file.read(reinterpret_cast<char *>(cell.get_phi()), size * sizeof(float));

    cells.push_back(std::move(cell));
  }

  file.close();
  std::cout << "Loaded checkpoint from " << filepath << " (step " << step << ")"
            << std::endl;

  return true;
}

bool load_vtk_field(const std::string &filepath, const std::string &field_name,
                    std::vector<float> &data, int &Lx, int &Ly) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file) {
    return false;
  }

  std::string line;
  bool found_dimensions = false;
  bool found_field = false;
  int num_points = 0;

  // Parse header
  while (std::getline(file, line)) {
    if (line.find("DIMENSIONS") != std::string::npos) {
      int Lz;
      sscanf(line.c_str(), "DIMENSIONS %d %d %d", &Lx, &Ly, &Lz);
      num_points = Lx * Ly;
      found_dimensions = true;
    }
    if (line.find("SCALARS " + field_name) != std::string::npos) {
      // Skip LOOKUP_TABLE line
      std::getline(file, line);
      found_field = true;
      break;
    }
  }

  if (!found_dimensions || !found_field) {
    return false;
  }

  // Read binary data
  data.resize(num_points);
  file.read(reinterpret_cast<char *>(data.data()), num_points * sizeof(float));

  // Swap endianness
  for (auto &val : data) {
    uint32_t tmp;
    std::memcpy(&tmp, &val, sizeof(float));
    tmp = ((tmp >> 24) & 0xFF) | ((tmp >> 8) & 0xFF00) |
          ((tmp << 8) & 0xFF0000) | ((tmp << 24) & 0xFF000000);
    std::memcpy(&val, &tmp, sizeof(float));
  }

  return true;
}

void save_observables(const std::string &output_dir,
                      const std::vector<float> &times,
                      const std::vector<float> &msd,
                      const std::vector<float> &mean_volume,
                      const std::vector<float> &volume_std) {
  std::ofstream file(output_dir + "/observables.csv");
  if (!file) {
    std::cerr << "Error: Could not open observables file for writing"
              << std::endl;
    return;
  }

  file << "time,msd,mean_volume,volume_std\n";
  for (size_t i = 0; i < times.size(); ++i) {
    file << times[i] << "," << msd[i] << "," << mean_volume[i] << ","
         << volume_std[i] << "\n";
  }

  file.close();
}

bool load_initial_conditions_json(Domain &domain, const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file) {
    std::cerr << "Error: Could not open " << filepath << " for reading"
              << std::endl;
    return false;
  }

  // Simple JSON parser - just extract what we need
  std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  file.close();

  // Parse domain size
  auto find_int = [&content](const std::string &key) -> int {
    size_t pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return -1;
    pos = content.find(":", pos);
    if (pos == std::string::npos) return -1;
    return std::atoi(content.c_str() + pos + 1);
  };

  auto find_float = [&content](const std::string &key) -> float {
    size_t pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return -1.0f;
    pos = content.find(":", pos);
    if (pos == std::string::npos) return -1.0f;
    return std::atof(content.c_str() + pos + 1);
  };

  // Get domain parameters
  int Nx = find_int("Nx");
  int Ny = find_int("Ny");
  if (Nx <= 0 || Ny <= 0) {
    std::cerr << "Error: Invalid domain size in JSON" << std::endl;
    return false;
  }

  // Get simulation parameters
  SimParams params = domain.get_params();  // Keep existing params as defaults
  params.Nx = Nx;
  params.Ny = Ny;
  
  float lambda = find_float("lambda");
  if (lambda > 0) params.lambda = lambda;
  
  float gamma = find_float("gamma");
  if (gamma > 0) params.gamma = gamma;
  
  float kappa = find_float("kappa");
  if (kappa > 0) params.kappa = kappa;
  
  float mu = find_float("mu");
  if (mu > 0) params.mu = mu;
  
  float target_radius = find_float("target_radius");
  if (target_radius > 0) params.target_radius = target_radius;
  
  float v_A = find_float("v_A");
  if (v_A >= 0) params.v_A = v_A;
  
  float tau = find_float("tau");
  if (tau > 0) params.tau = tau;

  domain.initialize(params);

  // Parse cell positions
  std::vector<float> cx_list, cy_list, vx_list, vy_list;
  
  // Find "cells" array
  size_t cells_pos = content.find("\"cells\"");
  if (cells_pos == std::string::npos) {
    std::cerr << "Error: No cells array found in JSON" << std::endl;
    return false;
  }

  // Find each cell object and extract cx, cy
  size_t pos = cells_pos;
  while ((pos = content.find("{", pos)) != std::string::npos) {
    size_t end = content.find("}", pos);
    if (end == std::string::npos) break;
    
    std::string cell_str = content.substr(pos, end - pos + 1);
    
    // Look for cx in this cell object
    size_t cx_pos = cell_str.find("\"cx\"");
    if (cx_pos == std::string::npos) {
      pos = end + 1;
      continue;  // Not a cell object
    }
    
    // Parse cx
    size_t colon = cell_str.find(":", cx_pos);
    if (colon == std::string::npos) break;
    float cx = std::atof(cell_str.c_str() + colon + 1);
    
    // Parse cy
    size_t cy_pos = cell_str.find("\"cy\"");
    if (cy_pos == std::string::npos) break;
    colon = cell_str.find(":", cy_pos);
    if (colon == std::string::npos) break;
    float cy = std::atof(cell_str.c_str() + colon + 1);
    
    // Parse vx (optional)
    float vx = 0.0f;
    size_t vx_pos = cell_str.find("\"vx\"");
    if (vx_pos != std::string::npos) {
      colon = cell_str.find(":", vx_pos);
      if (colon != std::string::npos) vx = std::atof(cell_str.c_str() + colon + 1);
    }
    
    // Parse vy (optional)
    float vy = 0.0f;
    size_t vy_pos = cell_str.find("\"vy\"");
    if (vy_pos != std::string::npos) {
      colon = cell_str.find(":", vy_pos);
      if (colon != std::string::npos) vy = std::atof(cell_str.c_str() + colon + 1);
    }
    
    cx_list.push_back(cx);
    cy_list.push_back(cy);
    vx_list.push_back(vx);
    vy_list.push_back(vy);
    
    pos = end + 1;
  }

  if (cx_list.empty()) {
    std::cerr << "Error: No cell positions found in JSON" << std::endl;
    return false;
  }

  std::cout << "Loading " << cx_list.size() << " cells from JSON" << std::endl;

  // Load the cells
  domain.load_from_checkpoint(cx_list, cy_list, vx_list, vy_list);

  return true;
}

} // namespace cellsim
