#include "io.cuh"
#include "kernels.cuh"  // for CUDA_CHECK
#ifdef STRESS_FIELDS_ENABLED
#include "diagnostics.cuh"
#endif
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <thread>

namespace cellsim {

//=============================================================================
// GPU kernel: scatter cell phi fields into a full-domain field using max
//
// Uses atomicMax on the int reinterpretation of positive floats.
// For non-negative IEEE 754 floats, integer ordering = float ordering,
// so atomicMax(int*, __float_as_int(val)) gives the correct float max.
//=============================================================================

__global__ void kernel_scatter_phi_max(
    float **__restrict__ phi_ptrs,
    float *__restrict__ full_field,
    const int *__restrict__ widths,
    const int *__restrict__ heights,
    const int *__restrict__ offsets_x,
    const int *__restrict__ offsets_y,
    int Nx, int Ny, int num_cells,
    int halo)
{
  int cell_idx = blockIdx.z;
  if (cell_idx >= num_cells) return;

  int width  = widths[cell_idx];
  int height = heights[cell_idx];

  // Thread covers inner region only (skip halo on all sides)
  int inner_x = blockIdx.x * blockDim.x + threadIdx.x;
  int inner_y = blockIdx.y * blockDim.y + threadIdx.y;
  int lx = inner_x + halo;
  int ly = inner_y + halo;

  if (lx >= width - halo || ly >= height - halo) return;

  const float *phi = phi_ptrs[cell_idx];
  float phi_val = phi[ly * width + lx];

  if (phi_val < 1e-6f) return;  // Skip negligible values

  // Periodic global coordinate (same logic as kernel_scatter_phi_sq)
  int ox = offsets_x[cell_idx];
  int oy = offsets_y[cell_idx];
  int gx_raw = ox + lx;
  int gx = gx_raw - ((gx_raw >= Nx) - (gx_raw < 0)) * Nx;
  int gy_raw = oy + ly;
  int gy = gy_raw - ((gy_raw >= Ny) - (gy_raw < 0)) * Ny;

  // Atomic float max via int reinterpretation (valid for non-negative floats)
  atomicMax((int*)&full_field[gy * Nx + gx], __float_as_int(phi_val));
}

//=============================================================================
// Export frame to simple text format (compatible with your Python loader)
//=============================================================================

void export_frame_txt(const Domain &domain, const std::string &filename,
                      int frame) {
  // Reconstruct full domain field from cell subdomains
  int Nx = domain.params.Nx;
  int Ny = domain.params.Ny;

  std::vector<float> full_field(Nx * Ny, 0.0f);

  // For each cell, copy its field to the global domain
  for (const auto &cell : domain.cells) {
    std::vector<float> h_phi(cell->field_size);
    cudaMemcpy(h_phi.data(), cell->d_phi, cell->field_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    int halo = domain.params.halo_width;

    // Copy inner region (excluding halo) to global field
    for (int ly = halo; ly < cell->height() - halo; ++ly) {
      for (int lx = halo; lx < cell->width() - halo; ++lx) {
        int gx, gy;
        cell->bbox_with_halo.local_to_global(lx, ly, gx, gy, Nx, Ny);

        int local_idx = ly * cell->width() + lx;
        int global_idx = gy * Nx + gx;

        // Take maximum if cells overlap (shouldn't happen if initialized
        // correctly)
        full_field[global_idx] =
            fmaxf(full_field[global_idx], h_phi[local_idx]);
      }
    }
  }

  // Write to file in your format
  std::ofstream file(filename);
  file << std::fixed << std::setprecision(6);

  // First row: x coordinates (column headers)
  file << 0.0f; // Corner element
  for (int x = 0; x < Nx; ++x) {
    file << " " << (x * domain.params.dx);
  }
  file << "\n";

  // Data rows: y coordinate followed by field values
  for (int y = 0; y < Ny; ++y) {
    file << (y * domain.params.dy);
    for (int x = 0; x < Nx; ++x) {
      file << " " << full_field[y * Nx + x];
    }
    file << "\n";
  }

  file.close();
}

//=============================================================================
// Export cell tracking data
//=============================================================================

void export_tracking_data(const Domain &domain, const std::string &filename,
                          float time) {
  std::ofstream file(filename, std::ios::app); // Append mode
  file << std::fixed << std::setprecision(6);

  for (const auto &cell : domain.cells) {
    // Format: time, x, y, cell_id
    file << time << " " << cell->centroid.x << " " << cell->centroid.y << " "
         << cell->id << "\n";
  }

  file.close();
}

//=============================================================================
// VTK Export
//=============================================================================

void export_vtk(const Domain &domain, const std::string &base_filename,
                int frame) {
  std::stringstream ss;
  ss << base_filename << "_" << std::setfill('0') << std::setw(6) << frame
     << ".vtk";
  std::string filename = ss.str();

  int Nx = domain.params.Nx;
  int Ny = domain.params.Ny;

  // Reconstruct full field + cell identity map (argmax of phi_i)
  std::vector<float> full_field(Nx * Ny, 0.0f);
  std::vector<int> cell_id_field(Nx * Ny, -1);

  for (const auto &cell : domain.cells) {
    std::vector<float> h_phi(cell->field_size);
    cudaMemcpy(h_phi.data(), cell->d_phi, cell->field_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    int halo = domain.params.halo_width;

    for (int ly = halo; ly < cell->height() - halo; ++ly) {
      for (int lx = halo; lx < cell->width() - halo; ++lx) {
        int gx, gy;
        cell->bbox_with_halo.local_to_global(lx, ly, gx, gy, Nx, Ny);

        int local_idx = ly * cell->width() + lx;
        int global_idx = gy * Nx + gx;

        if (h_phi[local_idx] > full_field[global_idx]) {
          full_field[global_idx] = h_phi[local_idx];
          cell_id_field[global_idx] = cell->id;
        }
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

  // Cell identity field (which cell owns each pixel)
  file << "SCALARS cell_id int 1\n";
  file << "LOOKUP_TABLE default\n";
  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      file << cell_id_field[y * Nx + x] << "\n";
    }
  }

  file.close();
}

//=============================================================================
// Export individual cell fields for energy analysis
//=============================================================================

void export_vtk_individual(const Domain &domain,
                           const std::string &base_filename, int frame) {
  int Nx = domain.params.Nx;
  int Ny = domain.params.Ny;

  // Create sum field (for accurate energy calculation)
  std::vector<float> sum_field(Nx * Ny, 0.0f);

  // First, collect all cell fields and compute sum
  std::vector<std::vector<float>> cell_fields;
  cell_fields.reserve(domain.cells.size());

  for (const auto &cell : domain.cells) {
    // Get cell field from GPU
    std::vector<float> h_phi(cell->field_size);
    cudaMemcpy(h_phi.data(), cell->d_phi, cell->field_size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cell_fields.push_back(std::move(h_phi));

    int halo = domain.params.halo_width;

    // Add to sum field
    for (int ly = halo; ly < cell->height() - halo; ++ly) {
      for (int lx = halo; lx < cell->width() - halo; ++lx) {
        int gx, gy;
        cell->bbox_with_halo.local_to_global(lx, ly, gx, gy, Nx, Ny);

        int local_idx = ly * cell->width() + lx;
        int global_idx = gy * Nx + gx;

        sum_field[global_idx] += cell_fields.back()[local_idx];
      }
    }
  }

  // Write sum field VTK (for visualization and energy calculation)
  {
    std::stringstream ss;
    ss << base_filename << "_" << std::setfill('0') << std::setw(6) << frame
       << "_sum.vtk";
    std::string filename = ss.str();

    std::ofstream file(filename);
    file << "# vtk DataFile Version 3.0\n";
    file << "Phase field sum (all cells) frame " << frame << "\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << domain.params.dx << " " << domain.params.dy << " 1\n";
    file << "POINT_DATA " << (Nx * Ny) << "\n";
    file << "SCALARS phi_sum float 1\n";
    file << "LOOKUP_TABLE default\n";

    for (int y = 0; y < Ny; ++y) {
      for (int x = 0; x < Nx; ++x) {
        file << sum_field[y * Nx + x] << "\n";
      }
    }
    file.close();
  }

  // Write individual cell VTK files
  for (size_t c = 0; c < domain.cells.size(); ++c) {
    const auto &cell = domain.cells[c];
    const auto &h_phi = cell_fields[c];

    std::stringstream ss;
    ss << base_filename << "_" << std::setfill('0') << std::setw(6) << frame
       << "_cell_" << std::setfill('0') << std::setw(3) << cell->id << ".vtk";
    std::string filename = ss.str();

    // Create full-domain field for this cell
    std::vector<float> cell_full(Nx * Ny, 0.0f);

    int halo = domain.params.halo_width;
    for (int ly = halo; ly < cell->height() - halo; ++ly) {
      for (int lx = halo; lx < cell->width() - halo; ++lx) {
        int gx, gy;
        cell->bbox_with_halo.local_to_global(lx, ly, gx, gy, Nx, Ny);

        int local_idx = ly * cell->width() + lx;
        int global_idx = gy * Nx + gx;

        cell_full[global_idx] = h_phi[local_idx];
      }
    }

    std::ofstream file(filename);
    file << "# vtk DataFile Version 3.0\n";
    file << "Phase field cell " << cell->id << " frame " << frame << "\n";
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
        file << cell_full[y * Nx + x] << "\n";
      }
    }
    file.close();
  }
}

//=============================================================================
// Export energy metrics (computed during simulation for accuracy)
//=============================================================================

void export_energy_metrics(const Domain &domain, const std::string &filename,
                           int frame, float time) {
  int Nx = domain.params.Nx;
  int Ny = domain.params.Ny;
  float dx = domain.params.dx;
  float dy = domain.params.dy;
  float dA = dx * dy;

  // Parameters for energy calculation
  float lambda = domain.params.lambda;
  float gamma = domain.params.gamma;
  float kappa = domain.params.kappa;
  float bulk_coeff = 30.0f / (lambda * lambda);
  float interaction_coeff = 30.0f * kappa / (lambda * lambda);

  // Collect all cell fields
  std::vector<std::vector<float>> cell_fields;
  cell_fields.reserve(domain.cells.size());

  for (const auto &cell : domain.cells) {
    std::vector<float> h_phi(cell->field_size);
    cudaMemcpy(h_phi.data(), cell->d_phi, cell->field_size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cell_fields.push_back(std::move(h_phi));
  }

  // Compute energies
  float total_gradient = 0.0f;
  float total_bulk = 0.0f;
  float total_interaction = 0.0f;
  float total_adhesion = 0.0f;

  // For each cell, compute gradient and bulk energy
  for (size_t c = 0; c < domain.cells.size(); ++c) {
    const auto &cell = domain.cells[c];
    const auto &h_phi = cell_fields[c];

    int halo = domain.params.halo_width;
    int w = cell->width();

    // Gradient energy for this cell (using central differences)
    for (int ly = halo; ly < cell->height() - halo; ++ly) {
      for (int lx = halo; lx < cell->width() - halo; ++lx) {
        int idx = ly * w + lx;
        float phi = h_phi[idx];

        // Central differences for gradient
        float dphi_dx = (h_phi[idx + 1] - h_phi[idx - 1]) / (2.0f * dx);
        float dphi_dy = (h_phi[idx + w] - h_phi[idx - w]) / (2.0f * dy);
        float grad_sq = dphi_dx * dphi_dx + dphi_dy * dphi_dy;

        total_gradient += gamma * grad_sq * dA;

        // Bulk energy
        float phi_clamped = fminf(fmaxf(phi, 0.0f), 1.0f);
        float bulk = bulk_coeff * phi_clamped * phi_clamped *
                     (1.0f - phi_clamped) * (1.0f - phi_clamped);
        total_bulk += bulk * dA;
      }
    }
  }

  // Compute interaction energy: sum over all pairs of cells
  // E_int = (30κ/λ²) ∫ Σ_{n≠m} φ_n² φ_m² dx
  for (size_t c1 = 0; c1 < domain.cells.size(); ++c1) {
    for (size_t c2 = c1 + 1; c2 < domain.cells.size(); ++c2) {
      const auto &cell1 = domain.cells[c1];
      const auto &cell2 = domain.cells[c2];

      // Check if bounding boxes overlap
      if (!cell1->bbox_with_halo.overlaps(cell2->bbox_with_halo, Nx, Ny)) {
        continue;
      }

      const auto &phi1 = cell_fields[c1];
      const auto &phi2 = cell_fields[c2];

      int halo = domain.params.halo_width;

      // Iterate over overlapping region
      for (int ly1 = halo; ly1 < cell1->height() - halo; ++ly1) {
        for (int lx1 = halo; lx1 < cell1->width() - halo; ++lx1) {
          int gx, gy;
          cell1->bbox_with_halo.local_to_global(lx1, ly1, gx, gy, Nx, Ny);

          // Check if this point is also in cell2's bounding box
          if (!cell2->bbox_with_halo.contains(gx, gy, Nx, Ny)) {
            continue;
          }

          int lx2, ly2;
          cell2->bbox_with_halo.global_to_local(gx, gy, lx2, ly2, Nx, Ny);

          if (lx2 < halo || lx2 >= cell2->width() - halo || ly2 < halo ||
              ly2 >= cell2->height() - halo) {
            continue;
          }

          int idx1 = ly1 * cell1->width() + lx1;
          int idx2 = ly2 * cell2->width() + lx2;

          float p1 = phi1[idx1];
          float p2 = phi2[idx2];

          // Interaction: φ_n² φ_m²
          float interaction = interaction_coeff * p1 * p1 * p2 * p2;
          total_interaction += interaction * dA;

          // Adhesion: gradient coupling ∇φ_i·∇φ_j (surface tension reduction)
          // F_adh = J Σ_{i<j} ∫ ∇φ_i·∇φ_j dA  (negative at shared interfaces)
          if (domain.params.adhesion_J > 0.0f) {
            int w1 = cell1->width();
            int w2 = cell2->width();
            float inv2dx = 1.0f / (2.0f * dx);
            float inv2dy = 1.0f / (2.0f * dy);
            float dphi1_dx = (phi1[idx1 + 1] - phi1[idx1 - 1]) * inv2dx;
            float dphi1_dy = (phi1[idx1 + w1] - phi1[idx1 - w1]) * inv2dy;
            float dphi2_dx = (phi2[idx2 + 1] - phi2[idx2 - 1]) * inv2dx;
            float dphi2_dy = (phi2[idx2 + w2] - phi2[idx2 - w2]) * inv2dy;
            float grad_dot = dphi1_dx * dphi2_dx + dphi1_dy * dphi2_dy;
            total_adhesion += domain.params.adhesion_J * grad_dot * dA;
          }
        }
      }
    }
  }

  float total_energy = total_gradient + total_bulk + total_interaction + total_adhesion;

  // Append to file
  std::ofstream file(filename, std::ios::app);
  if (file.tellp() == 0) {
    // Write header if file is empty
    file << "# Frame Time TotalEnergy GradientEnergy BulkEnergy "
            "InteractionEnergy AdhesionEnergy\n";
  }

  file << std::fixed << std::setprecision(6);
  file << frame << " " << time << " " << total_energy << " " << total_gradient
       << " " << total_bulk << " " << total_interaction
       << " " << total_adhesion << "\n";
  file.close();
}

//=============================================================================
// Checkpoint Save/Load
//=============================================================================

void save_checkpoint(const Domain &domain, const std::string &filename,
                     const CheckpointHeader &header,
                     const float *h_v_A, int num_v_A,
                     const float *h_gamma, int num_gamma,
                     const float *h_target_radius, int num_target_radius) {
  // Write to temporary file first, then rename atomically
  // This prevents corruption if process is killed mid-write
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
    file.write(reinterpret_cast<const char *>(&cell->bbox),
               sizeof(BoundingBox));
    file.write(reinterpret_cast<const char *>(&cell->centroid), sizeof(Vec2));
    file.write(reinterpret_cast<const char *>(&cell->velocity), sizeof(Vec2));
    file.write(reinterpret_cast<const char *>(&cell->volume), sizeof(float));

    // Copy field from device and write
    std::vector<float> h_phi(cell->field_size);
    CUDA_CHECK(cudaMemcpy(h_phi.data(), cell->d_phi, cell->field_size * sizeof(float),
               cudaMemcpyDeviceToHost));
    file.write(reinterpret_cast<const char *>(h_phi.data()),
               cell->field_size * sizeof(float));
  }

  // Write per-cell v_A values for quenched disorder preservation
  // Magic marker: 0x56415F41 = "VA_A" followed by count and data
  if (h_v_A != nullptr && num_v_A > 0) {
    uint32_t v_A_magic = 0x56415F41; // "VA_A"
    file.write(reinterpret_cast<const char *>(&v_A_magic), sizeof(uint32_t));
    int32_t v_A_count = num_v_A;
    file.write(reinterpret_cast<const char *>(&v_A_count), sizeof(int32_t));
    file.write(reinterpret_cast<const char *>(h_v_A),
               num_v_A * sizeof(float));
  }

  // Write per-cell gamma values for elasticity mismatch preservation
  // Magic marker: 0x47414D41 = "GAMA" followed by count and data
  if (h_gamma != nullptr && num_gamma > 0) {
    uint32_t gamma_magic = 0x47414D41; // "GAMA"
    file.write(reinterpret_cast<const char *>(&gamma_magic), sizeof(uint32_t));
    int32_t gamma_count = num_gamma;
    file.write(reinterpret_cast<const char *>(&gamma_count), sizeof(int32_t));
    file.write(reinterpret_cast<const char *>(h_gamma),
               num_gamma * sizeof(float));
  }

  // Write per-cell target radius for polydispersity preservation
  // Magic marker: 0x52414449 = "RADI" followed by count and data
  if (h_target_radius != nullptr && num_target_radius > 0) {
    uint32_t radius_magic = 0x52414449; // "RADI"
    file.write(reinterpret_cast<const char *>(&radius_magic), sizeof(uint32_t));
    int32_t radius_count = num_target_radius;
    file.write(reinterpret_cast<const char *>(&radius_count), sizeof(int32_t));
    file.write(reinterpret_cast<const char *>(h_target_radius),
               num_target_radius * sizeof(float));
  }

  file.close();

  // Atomic rename: if this fails, the old checkpoint is still valid
  // On POSIX, rename() is atomic. On Windows, we need to remove first.
#ifdef _WIN32
  std::remove(filename.c_str()); // Remove old file (if exists)
#endif
  if (std::rename(temp_filename.c_str(), filename.c_str()) != 0) {
    printf("ERROR: Could not rename checkpoint file\n");
    return;
  }

  printf("Saved checkpoint: step=%d, t=%.4f, cells=%d\n", hdr.current_step,
         hdr.current_time, hdr.num_cells);
}

bool load_checkpoint(Domain &domain, const std::string &filename,
                     CheckpointHeader &out_header,
                     std::vector<float> *out_v_A,
                     std::vector<float> *out_gamma,
                     std::vector<float> *out_target_radius) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    printf("Warning: Could not open checkpoint file: %s\n", filename.c_str());
    return false;
  }

  // Read header and check magic/version
  // For v2/v3, the header is smaller (no sim_params_size field)
  // Read minimum header first to get version
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

  if (min_hdr.version < 2 || min_hdr.version > 5) {
    printf("Error: Unsupported checkpoint version %d (expected 2-5)\n",
           min_hdr.version);
    return false;
  }

  // Seek back to start and read appropriate header size
  file.seekg(0);

  CheckpointHeader header;
  if (min_hdr.version <= 3) {
    // v2/v3 header doesn't have sim_params_size field
    // Read just the old header size (without sim_params_size)
    size_t old_header_size = sizeof(CheckpointHeader) - sizeof(uint32_t);
    file.read(reinterpret_cast<char *>(&header), old_header_size);
    header.sim_params_size = 0; // Will be set below

    if (min_hdr.version == 2) {
      printf("Note: Loading v2 checkpoint - using default runtime options\n");
    }
    printf("Note: Loading v3 checkpoint - using default motility model "
           "(Run-and-Tumble)\n");
  } else {
    // v4: Check if this is an old-format checkpoint with _padding field
    // Old format had: 3 bools + 1 implicit padding + 4-byte _padding +
    // sim_params_size New format has: 4 bools + sim_params_size (at offset 36)

    // Read bytes 36-43 to detect format
    file.seekg(36);
    uint32_t val_at_36, val_at_40;
    file.read(reinterpret_cast<char *>(&val_at_36), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&val_at_40), sizeof(uint32_t));
    file.seekg(0);

    // Old format: val_at_36 = 0 (old _padding), val_at_40 = sim_params_size
    // New format: val_at_36 = sim_params_size (>= 64), val_at_40 = start of SimParams
    // NOTE: Don't compare val_at_40 against sizeof(SimParams) — that breaks
    // when SimParams grows (e.g., adding v_A_sigma changed 76 → 80).
    // Instead, just check if val_at_36 == 0 (always true for old _padding,
    // never true for sim_params_size which is >= 64).
    bool is_old_format = (val_at_36 == 0 && val_at_40 >= 64 && val_at_40 <= 512);

    if (is_old_format) {
      // Old v4 format with _padding field - header was 44 bytes
      printf("Note: Loading old v4 checkpoint format (with padding field)\n");

      // Read up to offset 36 (before _padding)
      file.read(reinterpret_cast<char *>(&header), 36);

      // Skip _padding (4 bytes)
      file.seekg(40);

      // Read sim_params_size
      file.read(reinterpret_cast<char *>(&header.sim_params_size),
                sizeof(uint32_t));

      // Now at offset 44, which is start of SimParams
    } else {
      // New v4 format - read full header
      file.read(reinterpret_cast<char *>(&header), sizeof(CheckpointHeader));
    }
  }

  // Copy header to output
  out_header = header;
  int num_cells = header.num_cells;

  // Handle SimParams size mismatch for old checkpoints
  // v3 SimParams had 18 fields (Nx through subdomain_padding) = 72 bytes.
  // Hardcoded because the struct has grown since then.
  constexpr size_t v3_sim_params_size = 72;

  if (header.version <= 3 || header.sim_params_size == 0) {
    // Old checkpoint without motility_model or v_A_sigma fields
    // Zero-init first, then read only what exists
    memset(&domain.params, 0, sizeof(SimParams));
    file.read(reinterpret_cast<char *>(&domain.params), v3_sim_params_size);
    domain.params.motility_model =
        SimParams::MotilityModel::RunAndTumble; // Default
    domain.params.v_A_sigma = 0.0f;             // Default: no disorder
    domain.params.soft_cell_id = -1;            // Default: no soft cell
    domain.params.gamma_soft = 0.35f;           // Default
  } else {
    // v4+: use sim_params_size for backward/forward compatibility.
    // Zero-init the entire struct first so any fields beyond what the
    // checkpoint stores get sensible defaults (0 for floats/ints, which
    // means: no adhesion, no soft-cell, no disorder, etc.)
    memset(&domain.params, 0, sizeof(SimParams));
    // Then overwrite with however many bytes the checkpoint has
    size_t read_size =
        std::min((size_t)header.sim_params_size, sizeof(SimParams));
    file.read(reinterpret_cast<char *>(&domain.params), read_size);
    // Skip extra bytes if checkpoint is from a newer binary
    if (header.sim_params_size > sizeof(SimParams)) {
      file.seekg(header.sim_params_size - sizeof(SimParams), std::ios::cur);
    }
    // Fix up sentinel values that shouldn't be 0
    // When loading from a smaller checkpoint, fields that didn't exist in the
    // old struct get either zero (from memset) or garbage (from bytes that
    // belonged to a different field in the old layout). For soft_cell_id,
    // the safe default is -1 (no soft cell), not 0 (which means "cell 0 is soft").
    if (read_size < sizeof(SimParams)) {
      domain.params.soft_cell_id = -1;  // Always reset when struct size differs
      domain.params.gamma_soft = 0.35f; // Default
    }
    if (read_size != sizeof(SimParams)) {
      printf("Note: Checkpoint SimParams size %u differs from binary %zu — "
             "new fields initialized to defaults.\n",
             header.sim_params_size, sizeof(SimParams));
    }
  }

  printf("Checkpoint SimParams: Nx=%d, Ny=%d, dx=%.3f, dy=%.3f, dt=%.4f, "
         "R=%.1f, v_A=%.6f, v_A_sigma=%.6f\n",
         domain.params.Nx, domain.params.Ny, domain.params.dx, domain.params.dy,
         domain.params.dt, domain.params.target_radius, domain.params.v_A,
         domain.params.v_A_sigma);
  fflush(stdout);

  // Safety check: validate domain size is reasonable (max ~4GB for single
  // field)
  const size_t MAX_DOMAIN_PIXELS = 65536ULL * 65536ULL; // ~4 billion pixels
  size_t domain_pixels = (size_t)domain.params.Nx * (size_t)domain.params.Ny;
  if (domain_pixels > MAX_DOMAIN_PIXELS || domain.params.Nx <= 0 ||
      domain.params.Ny <= 0) {
    printf("Error: Invalid domain size Nx=%d, Ny=%d (corrupted checkpoint?)\n",
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

    // bbox from checkpoint is the inner bbox (without halo).
    auto cell = std::make_unique<Cell>(id, bbox, domain.params.halo_width);
    cell->centroid = centroid;
    cell->velocity = velocity;
    cell->volume = volume;

    // Read and upload field
    std::vector<float> h_phi(cell->field_size);
    file.read(reinterpret_cast<char *>(h_phi.data()),
              cell->field_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(cell->d_phi, h_phi.data(), cell->field_size * sizeof(float),
               cudaMemcpyHostToDevice));

    domain.cells.push_back(std::move(cell));
    domain.next_cell_id = std::max(domain.next_cell_id, id + 1);
  }

  domain.device_arrays_dirty = true;
  domain.sync_device_arrays();
  domain.update_overlap_pairs();

  // Try to read per-cell v_A values (appended after cell data)
  if (out_v_A != nullptr) {
    uint32_t v_A_magic = 0;
    file.read(reinterpret_cast<char *>(&v_A_magic), sizeof(uint32_t));
    if (file.good() && v_A_magic == 0x56415F41) { // "VA_A"
      int32_t v_A_count = 0;
      file.read(reinterpret_cast<char *>(&v_A_count), sizeof(int32_t));
      if (file.good() && v_A_count > 0 && v_A_count <= num_cells) {
        out_v_A->resize(v_A_count);
        file.read(reinterpret_cast<char *>(out_v_A->data()),
                  v_A_count * sizeof(float));
        printf("  Loaded per-cell v_A: %d values from checkpoint\n", v_A_count);
      }
    }
    // If magic doesn't match or read fails, out_v_A stays empty
    // (old checkpoint without v_A data — will be re-generated)
  }

  // Try to read per-cell gamma values (appended after v_A data)
  if (out_gamma != nullptr && file.good()) {
    uint32_t gamma_magic = 0;
    file.read(reinterpret_cast<char *>(&gamma_magic), sizeof(uint32_t));
    if (file.good() && gamma_magic == 0x47414D41) { // "GAMA"
      int32_t gamma_count = 0;
      file.read(reinterpret_cast<char *>(&gamma_count), sizeof(int32_t));
      if (file.good() && gamma_count > 0 && gamma_count <= num_cells) {
        out_gamma->resize(gamma_count);
        file.read(reinterpret_cast<char *>(out_gamma->data()),
                  gamma_count * sizeof(float));
        printf("  Loaded per-cell gamma: %d values from checkpoint\n", gamma_count);
      }
    }
  }

  // Try to read per-cell target radius values (appended after gamma data)
  if (out_target_radius != nullptr && file.good()) {
    uint32_t radius_magic = 0;
    file.read(reinterpret_cast<char *>(&radius_magic), sizeof(uint32_t));
    if (file.good() && radius_magic == 0x52414449) { // "RADI"
      int32_t radius_count = 0;
      file.read(reinterpret_cast<char *>(&radius_count), sizeof(int32_t));
      if (file.good() && radius_count > 0 && radius_count <= num_cells) {
        out_target_radius->resize(radius_count);
        file.read(reinterpret_cast<char *>(out_target_radius->data()),
                  radius_count * sizeof(float));
        printf("  Loaded per-cell radius: %d values from checkpoint\n", radius_count);
      }
    }
  }

  file.close();

  printf("Loaded checkpoint: step=%d, t=%.4f, cells=%d\n",
         out_header.current_step, out_header.current_time, num_cells);
  if (header.version >= 3) {
    printf("  Runtime options: save_interval=%d, checkpoint_interval=%d, "
           "trajectory_samples=%d\n",
           out_header.save_interval, out_header.checkpoint_interval,
           out_header.trajectory_samples);
  }
  return true;
}

//=============================================================================
// Stress Field VTK Export
//=============================================================================

#ifdef STRESS_FIELDS_ENABLED

void export_vtk_with_stress(const Domain &domain, 
                           const StressFieldBuffers &stress,
                           const std::string &base_filename, int frame) {
  std::stringstream ss;
  ss << base_filename << "_" << std::setfill('0') << std::setw(6) << frame
     << ".vtk";
  std::string filename = ss.str();

  int Nx = domain.params.Nx;
  int Ny = domain.params.Ny;
  size_t field_size = (size_t)Nx * Ny;

  // Reconstruct phi field + cell identity map (same as export_vtk)
  std::vector<float> full_field(field_size, 0.0f);
  std::vector<int> cell_id_field(field_size, -1);

  for (const auto &cell : domain.cells) {
    std::vector<float> h_phi(cell->field_size);
    cudaMemcpy(h_phi.data(), cell->d_phi, cell->field_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    int halo = domain.params.halo_width;

    for (int ly = halo; ly < cell->height() - halo; ++ly) {
      for (int lx = halo; lx < cell->width() - halo; ++lx) {
        int gx, gy;
        cell->bbox_with_halo.local_to_global(lx, ly, gx, gy, Nx, Ny);

        int local_idx = ly * cell->width() + lx;
        int global_idx = gy * Nx + gx;

        if (h_phi[local_idx] > full_field[global_idx]) {
          full_field[global_idx] = h_phi[local_idx];
          cell_id_field[global_idx] = cell->id;
        }
      }
    }
  }

  // Download stress fields from GPU
  std::vector<float> h_sigma_xx(field_size);
  std::vector<float> h_sigma_yy(field_size);
  std::vector<float> h_sigma_xy(field_size);
  std::vector<float> h_pressure(field_size);

  cudaMemcpy(h_sigma_xx.data(), stress.d_sigma_xx_field, 
             field_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sigma_yy.data(), stress.d_sigma_yy_field,
             field_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sigma_xy.data(), stress.d_sigma_xy_field,
             field_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pressure.data(), stress.d_pressure_field,
             field_size * sizeof(float), cudaMemcpyDeviceToHost);

  // Write VTK file with multiple scalar fields
  std::ofstream file(filename);

  file << "# vtk DataFile Version 3.0\n";
  file << "Phase field simulation with stress fields, frame " << frame << "\n";
  file << "ASCII\n";
  file << "DATASET STRUCTURED_POINTS\n";
  file << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
  file << "ORIGIN 0 0 0\n";
  file << "SPACING " << domain.params.dx << " " << domain.params.dy << " 1\n";
  file << "POINT_DATA " << field_size << "\n";

  // Phase field
  file << "SCALARS phi float 1\n";
  file << "LOOKUP_TABLE default\n";
  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      file << full_field[y * Nx + x] << "\n";
    }
  }

  // Stress fields
  file << "SCALARS sigma_xx float 1\n";
  file << "LOOKUP_TABLE default\n";
  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      file << h_sigma_xx[y * Nx + x] << "\n";
    }
  }

  file << "SCALARS sigma_yy float 1\n";
  file << "LOOKUP_TABLE default\n";
  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      file << h_sigma_yy[y * Nx + x] << "\n";
    }
  }

  file << "SCALARS sigma_xy float 1\n";
  file << "LOOKUP_TABLE default\n";
  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      file << h_sigma_xy[y * Nx + x] << "\n";
    }
  }

  file << "SCALARS pressure float 1\n";
  file << "LOOKUP_TABLE default\n";
  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      file << h_pressure[y * Nx + x] << "\n";
    }
  }

  // Cell identity field (which cell owns each pixel)
  file << "SCALARS cell_id int 1\n";
  file << "LOOKUP_TABLE default\n";
  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      file << cell_id_field[y * Nx + x] << "\n";
    }
  }

  file.close();
}

#endif // STRESS_FIELDS_ENABLED

//=============================================================================
// AsyncVTKWriter Implementation
//=============================================================================

AsyncVTKWriter::AsyncVTKWriter()
    : d_full_field(nullptr), h_full_field(nullptr),
      field_count(0), is_initialized(false) {}

AsyncVTKWriter::~AsyncVTKWriter() {
  wait();
  if (d_full_field) { cudaFree(d_full_field); d_full_field = nullptr; }
  if (h_full_field) { cudaFreeHost(h_full_field); h_full_field = nullptr; }
  is_initialized = false;
}

void AsyncVTKWriter::initialize(int Nx, int Ny) {
  if (is_initialized) return;

  field_count = (size_t)Nx * Ny;
  size_t bytes = field_count * sizeof(float);

  cudaMalloc(&d_full_field, bytes);
  cudaMemset(d_full_field, 0, bytes);

  cudaMallocHost(&h_full_field, bytes);  // Pinned for fast DMA

  is_initialized = true;
  printf("Async VTK writer: %.1f MB GPU + %.1f MB pinned host\n",
         bytes / (1024.0 * 1024.0), bytes / (1024.0 * 1024.0));
}

void AsyncVTKWriter::submit(
    float **d_phi_ptrs,
    const int *d_widths, const int *d_heights,
    const int *d_offsets_x, const int *d_offsets_y,
    int num_cells, int max_w, int max_h,
    int Nx, int Ny, int halo,
    float dx, float dy,
    const std::string &base_filename, int frame)
{
  // Wait for any previous background write to finish
  wait();

  // 1. Clear the GPU full field buffer
  cudaMemset(d_full_field, 0, field_count * sizeof(float));

  // 2. Launch scatter kernel
  int inner_w = max_w - 2 * halo;
  int inner_h = max_h - 2 * halo;
  if (inner_w <= 0 || inner_h <= 0) return;

  dim3 block(16, 16);
  dim3 grid((inner_w + block.x - 1) / block.x,
            (inner_h + block.y - 1) / block.y,
            num_cells);

  kernel_scatter_phi_max<<<grid, block>>>(
      d_phi_ptrs, d_full_field,
      d_widths, d_heights, d_offsets_x, d_offsets_y,
      Nx, Ny, num_cells, halo);

  // 3. Synchronous D→H copy to pinned buffer (fast DMA, ~6ms for 6400²)
  cudaMemcpy(h_full_field, d_full_field,
             field_count * sizeof(float), cudaMemcpyDeviceToHost);

  // 4. Build filename
  std::stringstream ss;
  ss << base_filename << "_" << std::setfill('0') << std::setw(6) << frame
     << ".vtk";
  std::string filename = ss.str();

  // 5. Launch background thread for byte-swap + file write
  //    Capture a copy of the pointer and metadata; the pinned buffer
  //    is safe to read from the thread because wait() is called before
  //    the next submit() or destructor.
  float *buf = h_full_field;
  size_t count = field_count;
  writer_thread = std::thread(write_binary_vtk,
                              filename, buf, count,
                              Nx, Ny, dx, dy, frame);
}

void AsyncVTKWriter::wait() {
  if (writer_thread.joinable()) {
    writer_thread.join();
  }
}

// Static: runs on background thread, does NOT touch GPU
void AsyncVTKWriter::write_binary_vtk(
    const std::string &filename,
    float *data, size_t count,
    int Nx, int Ny,
    float dx, float dy, int frame)
{
  // Byte-swap to big-endian (VTK legacy binary requires it)
  {
    uint32_t *p = reinterpret_cast<uint32_t*>(data);
    for (size_t i = 0; i < count; ++i) {
      uint32_t v = p[i];
      p[i] = (v >> 24) | ((v >> 8) & 0xFF00) |
             ((v << 8) & 0xFF0000) | (v << 24);
    }
  }

  // Write VTK file: ASCII header + binary data
  FILE *fp = fopen(filename.c_str(), "wb");
  if (!fp) {
    fprintf(stderr, "ERROR: Could not open VTK file: %s\n", filename.c_str());
    return;
  }

  // ASCII header (written with fprintf, no big-endian issues)
  fprintf(fp, "# vtk DataFile Version 3.0\n");
  fprintf(fp, "Phase field simulation frame %d\n", frame);
  fprintf(fp, "BINARY\n");
  fprintf(fp, "DATASET STRUCTURED_POINTS\n");
  fprintf(fp, "DIMENSIONS %d %d 1\n", Nx, Ny);
  fprintf(fp, "ORIGIN 0 0 0\n");
  fprintf(fp, "SPACING %g %g 1\n", dx, dy);
  fprintf(fp, "POINT_DATA %d\n", Nx * Ny);
  fprintf(fp, "SCALARS phi float 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");

  // Binary data (already big-endian)
  fwrite(data, sizeof(float), count, fp);

  fclose(fp);
}

} // namespace cellsim
