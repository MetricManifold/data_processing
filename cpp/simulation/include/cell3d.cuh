#pragma once

#include "gpu_memory_tracker.cuh"
#include "physics.cuh"
#include "types3d.cuh"
#include <memory>
#include <vector>

namespace cellsim {

//=============================================================================
// Cell3D - Single 3D cell with phase field on subdomain
//=============================================================================

class Cell3D {
public:
  int id;          // Unique cell identifier
  CellState state; // Cell state (active, dividing, etc.)

  // Subdomain definition
  BoundingBox3D bbox;           // Bounding box in global coordinates
  BoundingBox3D bbox_with_halo; // Including ghost cells

  // Phase field data (on subdomain, stored on device)
  float *d_phi;     // Device pointer to phase field φ
  float *d_dphi_dt; // Device pointer to time derivative
  int field_size;   // Total elements in subdomain

  // Cell properties (computed from φ)
  float volume;  // ∫φ² dV (volume integral)
  Vec3 centroid; // Center of mass
  Vec3 velocity; // Cell velocity for motility

  // Self-propulsion (polarization)
  Vec3 polarization; // Unit vector p_n giving self-propulsion direction
  float theta;       // Azimuthal angle (for rotational diffusion)
  float phi_pol;     // Polar angle (for 3D rotational diffusion)

  // Volume constraint
  float volume_deviation; // ((4/3)πR³ - ∫φ²) for constraint term

public:
  Cell3D(int id_, const BoundingBox3D &initial_bbox, int halo_width);
  // Constructor for checkpoint loading - use exact bbox_with_halo from file
  Cell3D(int id_, const BoundingBox3D &bbox_,
         const BoundingBox3D &bbox_with_halo_);
  ~Cell3D();

  // No copy (device memory)
  Cell3D(const Cell3D &) = delete;
  Cell3D &operator=(const Cell3D &) = delete;

  // Move allowed
  Cell3D(Cell3D &&other) noexcept;
  Cell3D &operator=(Cell3D &&other) noexcept;

  // Memory management
  void allocate_device_memory();
  void free_device_memory();

  // Initialize phase field (spherical cell)
  void initialize_spherical(float cx, float cy, float cz, float radius,
                            const SimParams3D &params);

  // Update bounding box based on current field
  // Returns true if bbox changed and reallocation needed
  bool update_bounding_box(const SimParams3D &params, float threshold = 0.01f);

  // Compute derived quantities (volume, centroid)
  void compute_properties(const SimParams3D &params);

  // Get local index from local coordinates
  __host__ __device__ int local_index(int lx, int ly, int lz) const {
    int w = bbox_with_halo.width();
    int h = bbox_with_halo.height();
    return lz * (w * h) + ly * w + lx;
  }

  // Get local index from global coordinates
  __host__ __device__ int global_to_index(int gx, int gy, int gz, int Nx,
                                          int Ny, int Nz) const {
    int lx, ly, lz;
    bbox_with_halo.global_to_local(gx, gy, gz, lx, ly, lz, Nx, Ny, Nz);
    return local_index(lx, ly, lz);
  }

  // Subdomain dimensions (including halo)
  int width() const { return bbox_with_halo.width(); }
  int height() const { return bbox_with_halo.height(); }
  int depth() const { return bbox_with_halo.depth(); }

  // Check if subdomain wraps around domain boundaries
  bool wraps_x(int Nx) const {
    return bbox_with_halo.x0 < 0 || bbox_with_halo.x1 > Nx;
  }
  bool wraps_y(int Ny) const {
    return bbox_with_halo.y0 < 0 || bbox_with_halo.y1 > Ny;
  }
  bool wraps_z(int Nz) const {
    return bbox_with_halo.z0 < 0 || bbox_with_halo.z1 > Nz;
  }
};

//=============================================================================
// Cell3D Implementation
//=============================================================================

inline Cell3D::Cell3D(int id_, const BoundingBox3D &initial_bbox,
                      int halo_width)
    : id(id_), state(CellState::Active), bbox(initial_bbox),
      bbox_with_halo(initial_bbox.expanded(halo_width)), d_phi(nullptr),
      d_dphi_dt(nullptr), field_size(0), volume(0), centroid{0, 0, 0},
      velocity{0, 0, 0}, polarization{1, 0, 0}, theta(0), phi_pol(M_PI / 2),
      volume_deviation(0) {
  // Initialize with random polarization direction on unit sphere
  theta = static_cast<float>(rand()) / RAND_MAX * 2.0f * M_PI;
  phi_pol = acosf(2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
  polarization.x = sinf(phi_pol) * cosf(theta);
  polarization.y = sinf(phi_pol) * sinf(theta);
  polarization.z = cosf(phi_pol);
  allocate_device_memory();
}

// Constructor for checkpoint loading - use exact bounding boxes from file
inline Cell3D::Cell3D(int id_, const BoundingBox3D &bbox_,
                      const BoundingBox3D &bbox_with_halo_)
    : id(id_), state(CellState::Active), bbox(bbox_),
      bbox_with_halo(bbox_with_halo_), d_phi(nullptr), d_dphi_dt(nullptr),
      field_size(0), volume(0), centroid{0, 0, 0}, velocity{0, 0, 0},
      polarization{1, 0, 0}, theta(0), phi_pol(M_PI / 2), volume_deviation(0) {
  allocate_device_memory();
}

inline Cell3D::~Cell3D() { free_device_memory(); }

inline Cell3D::Cell3D(Cell3D &&other) noexcept
    : id(other.id), state(other.state), bbox(other.bbox),
      bbox_with_halo(other.bbox_with_halo), d_phi(other.d_phi),
      d_dphi_dt(other.d_dphi_dt), field_size(other.field_size),
      volume(other.volume), centroid(other.centroid), velocity(other.velocity),
      polarization(other.polarization), theta(other.theta),
      phi_pol(other.phi_pol), volume_deviation(other.volume_deviation) {
  other.d_phi = nullptr;
  other.d_dphi_dt = nullptr;
}

inline Cell3D &Cell3D::operator=(Cell3D &&other) noexcept {
  if (this != &other) {
    free_device_memory();

    id = other.id;
    state = other.state;
    bbox = other.bbox;
    bbox_with_halo = other.bbox_with_halo;
    d_phi = other.d_phi;
    d_dphi_dt = other.d_dphi_dt;
    field_size = other.field_size;
    volume = other.volume;
    centroid = other.centroid;
    velocity = other.velocity;
    polarization = other.polarization;
    theta = other.theta;
    phi_pol = other.phi_pol;
    volume_deviation = other.volume_deviation;

    other.d_phi = nullptr;
    other.d_dphi_dt = nullptr;
  }
  return *this;
}

inline void Cell3D::allocate_device_memory() {
  field_size = bbox_with_halo.size();
  if (field_size > 0) {
    CUDA_MALLOC(&d_phi, field_size * sizeof(float));
    CUDA_MALLOC(&d_dphi_dt, field_size * sizeof(float));
    cudaMemset(d_phi, 0, field_size * sizeof(float));
    cudaMemset(d_dphi_dt, 0, field_size * sizeof(float));
  }
}

inline void Cell3D::free_device_memory() {
  if (d_phi) {
    CUDA_FREE(d_phi, field_size * sizeof(float));
    d_phi = nullptr;
  }
  if (d_dphi_dt) {
    CUDA_FREE(d_dphi_dt, field_size * sizeof(float));
    d_dphi_dt = nullptr;
  }
  field_size = 0;
}

inline void Cell3D::initialize_spherical(float cx, float cy, float cz,
                                         float radius,
                                         const SimParams3D &params) {
  // Allocate temporary host buffer
  std::vector<float> h_phi(field_size);

  float lambda = params.lambda;
  float interface_width = sqrtf(2.0f) * lambda;

  // Use effective radius to get correct initial volume
  float effective_radius = effective_radius_3d(radius, lambda);

  int Nx = params.Nx;
  int Ny = params.Ny;
  int Nz = params.Nz;

  for (int lz = 0; lz < depth(); ++lz) {
    for (int ly = 0; ly < height(); ++ly) {
      for (int lx = 0; lx < width(); ++lx) {
        // Get global coordinates
        int gx, gy, gz;
        bbox_with_halo.local_to_global(lx, ly, lz, gx, gy, gz, Nx, Ny, Nz);

        // Distance from cell center (with periodic BC)
        float dx = gx - cx;
        float dy = gy - cy;
        float dz = gz - cz;

        // Periodic distance
        if (dx > Nx * 0.5f)
          dx -= Nx;
        if (dx < -Nx * 0.5f)
          dx += Nx;
        if (dy > Ny * 0.5f)
          dy -= Ny;
        if (dy < -Ny * 0.5f)
          dy += Ny;
        if (dz > Nz * 0.5f)
          dz -= Nz;
        if (dz < -Nz * 0.5f)
          dz += Nz;

        float r = sqrtf(dx * dx + dy * dy + dz * dz);

        // Use shared tanh profile function
        float phi = tanh_profile(r, effective_radius, lambda);

        h_phi[local_index(lx, ly, lz)] = phi;
      }
    }
  }

  // Copy to device
  cudaMemcpy(d_phi, h_phi.data(), field_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Set initial centroid and volume
  centroid = {cx, cy, cz};
  volume = (4.0f / 3.0f) * M_PI * radius * radius * radius; // Approximate
}

inline void Cell3D::compute_properties(const SimParams3D &params) {
  // Copy field to host for property computation
  std::vector<float> h_phi(field_size);
  cudaMemcpy(h_phi.data(), d_phi, field_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  int halo = params.halo_width;
  float dV = params.dx * params.dy * params.dz;
  int Nx = params.Nx;
  int Ny = params.Ny;
  int Nz = params.Nz;

  // Compute volume (∫φ² dV) and centroid
  float vol = 0.0f;
  float dcx = 0.0f, dcy = 0.0f, dcz = 0.0f;
  float weight_sum = 0.0f;

  // Use bbox center as reference point for periodic averaging
  float ref_x = (bbox_with_halo.x0 + bbox_with_halo.x1) * 0.5f;
  float ref_y = (bbox_with_halo.y0 + bbox_with_halo.y1) * 0.5f;
  float ref_z = (bbox_with_halo.z0 + bbox_with_halo.z1) * 0.5f;
  ref_x = fmodf(fmodf(ref_x, (float)Nx) + (float)Nx, (float)Nx);
  ref_y = fmodf(fmodf(ref_y, (float)Ny) + (float)Ny, (float)Ny);
  ref_z = fmodf(fmodf(ref_z, (float)Nz) + (float)Nz, (float)Nz);

  for (int lz = halo; lz < depth() - halo; ++lz) {
    for (int ly = halo; ly < height() - halo; ++ly) {
      for (int lx = halo; lx < width() - halo; ++lx) {
        int gx, gy, gz;
        bbox_with_halo.local_to_global(lx, ly, lz, gx, gy, gz, Nx, Ny, Nz);

        float phi = h_phi[local_index(lx, ly, lz)];
        float phi_sq = phi * phi;

        vol += phi_sq;

        // Compute periodic distance from reference point
        float dx = gx - ref_x;
        float dy = gy - ref_y;
        float dz = gz - ref_z;

        if (dx > Nx * 0.5f)
          dx -= Nx;
        if (dx < -Nx * 0.5f)
          dx += Nx;
        if (dy > Ny * 0.5f)
          dy -= Ny;
        if (dy < -Ny * 0.5f)
          dy += Ny;
        if (dz > Nz * 0.5f)
          dz -= Nz;
        if (dz < -Nz * 0.5f)
          dz += Nz;

        dcx += dx * phi_sq;
        dcy += dy * phi_sq;
        dcz += dz * phi_sq;
        weight_sum += phi_sq;
      }
    }
  }

  volume = vol * dV;

  if (weight_sum > 1e-8f) {
    centroid.x = ref_x + dcx / weight_sum;
    centroid.y = ref_y + dcy / weight_sum;
    centroid.z = ref_z + dcz / weight_sum;

    // Wrap centroid to [0, N)
    centroid.x = fmodf(fmodf(centroid.x, (float)Nx) + (float)Nx, (float)Nx);
    centroid.y = fmodf(fmodf(centroid.y, (float)Ny) + (float)Ny, (float)Ny);
    centroid.z = fmodf(fmodf(centroid.z, (float)Nz) + (float)Nz, (float)Nz);
  }

  volume_deviation = params.target_volume() - volume;
}

inline bool Cell3D::update_bounding_box(const SimParams3D &params,
                                        float threshold) {
  // Copy current field to host
  std::vector<float> h_phi(field_size);
  cudaMemcpy(h_phi.data(), d_phi, field_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  int halo = params.halo_width;
  int old_w = width();
  int old_h = height();
  int old_d = depth();
  int Nx = params.Nx, Ny = params.Ny, Nz = params.Nz;

  // Find maximum extent from centroid using PERIODIC distance
  // This correctly handles cells that span domain boundaries
  // Also track local bounds to detect when cell touches window edge
  float max_dist_x = 0.0f;
  float max_dist_y = 0.0f;
  float max_dist_z = 0.0f;
  int min_lx = old_w, max_lx = -1;
  int min_ly = old_h, max_ly = -1;
  int min_lz = old_d, max_lz = -1;
  bool found_any = false;

  for (int lz = halo; lz < old_d - halo; ++lz) {
    for (int ly = halo; ly < old_h - halo; ++ly) {
      for (int lx = halo; lx < old_w - halo; ++lx) {
        if (h_phi[local_index(lx, ly, lz)] > threshold) {
          found_any = true;

          // Track local bounds for edge detection
          min_lx = min(min_lx, lx);
          max_lx = max(max_lx, lx);
          min_ly = min(min_ly, ly);
          max_ly = max(max_ly, ly);
          min_lz = min(min_lz, lz);
          max_lz = max(max_lz, lz);

          // Get global coordinates
          int gx, gy, gz;
          bbox_with_halo.local_to_global(lx, ly, lz, gx, gy, gz, Nx, Ny, Nz);

          // Compute periodic distance from centroid
          float dx = static_cast<float>(gx) - centroid.x;
          float dy = static_cast<float>(gy) - centroid.y;
          float dz = static_cast<float>(gz) - centroid.z;

          // Wrap to nearest image
          if (dx > Nx * 0.5f)
            dx -= Nx;
          if (dx < -Nx * 0.5f)
            dx += Nx;
          if (dy > Ny * 0.5f)
            dy -= Ny;
          if (dy < -Ny * 0.5f)
            dy += Ny;
          if (dz > Nz * 0.5f)
            dz -= Nz;
          if (dz < -Nz * 0.5f)
            dz += Nz;

          // Track maximum extent in each direction
          max_dist_x = max(max_dist_x, fabsf(dx));
          max_dist_y = max(max_dist_y, fabsf(dy));
          max_dist_z = max(max_dist_z, fabsf(dz));
        }
      }
    }
  }

  if (!found_any) {
    // No field above threshold - cell has disappeared
    return false;
  }

  // Add padding for cell growth and movement
  int padding = static_cast<int>(params.target_radius *
                                 (params.subdomain_padding - 1.0f)) +
                halo;

  // Use the already-computed centroid (which handles periodic boundaries
  // correctly)
  int new_cx = static_cast<int>(centroid.x);
  int new_cy = static_cast<int>(centroid.y);
  int new_cz = static_cast<int>(centroid.z);

  // Compute half-size from maximum periodic distance to any point with φ >
  // threshold
  int half_w = static_cast<int>(max_dist_x) + padding;
  int half_h = static_cast<int>(max_dist_y) + padding;
  int half_d = static_cast<int>(max_dist_z) + padding;

  // Minimum size
  int min_half =
      static_cast<int>(params.target_radius * params.subdomain_padding);
  half_w = max(half_w, min_half);
  half_h = max(half_h, min_half);
  half_d = max(half_d, min_half);

  // New bounding box centered on centroid
  BoundingBox3D new_bbox = {new_cx - half_w, new_cy - half_h, new_cz - half_d,
                            new_cx + half_w, new_cy + half_h, new_cz + half_d};

  // Add halo
  BoundingBox3D new_bbox_with_halo = {new_bbox.x0 - halo, new_bbox.y0 - halo,
                                      new_bbox.z0 - halo, new_bbox.x1 + halo,
                                      new_bbox.y1 + halo, new_bbox.z1 + halo};

  // Check if we need to reallocate
  int old_cx = (bbox_with_halo.x0 + bbox_with_halo.x1) / 2;
  int old_cy = (bbox_with_halo.y0 + bbox_with_halo.y1) / 2;
  int old_cz = (bbox_with_halo.z0 + bbox_with_halo.z1) / 2;

  int shift_x = new_cx - old_cx;
  int shift_y = new_cy - old_cy;
  int shift_z = new_cz - old_cz;

  // Handle periodic wrapping of shift
  if (shift_x > Nx / 2)
    shift_x -= Nx;
  if (shift_x < -Nx / 2)
    shift_x += Nx;
  if (shift_y > Ny / 2)
    shift_y -= Ny;
  if (shift_y < -Ny / 2)
    shift_y += Ny;
  if (shift_z > Nz / 2)
    shift_z -= Nz;
  if (shift_z < -Nz / 2)
    shift_z += Nz;

  // Check if cell is touching window boundary (rare but critical case)
  bool touching_edge = (min_lx <= halo + 1) || (max_lx >= old_w - halo - 2) ||
                       (min_ly <= halo + 1) || (max_ly >= old_h - halo - 2) ||
                       (min_lz <= halo + 1) || (max_lz >= old_d - halo - 2);

  // Reposition if:
  // 1. Center moved significantly (conservative threshold for performance), OR
  // 2. Cell is touching window boundary (rare, but must handle to prevent
  // cutoff)
  if (abs(shift_x) < 5 && abs(shift_y) < 5 && abs(shift_z) < 5 &&
      !touching_edge) {
    return false;
  }

  // Allocate new field
  int new_size = new_bbox_with_halo.size();
  int new_w = new_bbox_with_halo.width();
  int new_h = new_bbox_with_halo.height();
  int new_d = new_bbox_with_halo.depth();

  std::vector<float> h_phi_new(new_size, 0.0f);

  // Copy old field to new position
  // KEY: Iterate over OLD coordinates, map to NEW using proper periodic
  // conversion
  for (int old_lz = 0; old_lz < old_d; ++old_lz) {
    for (int old_ly = 0; old_ly < old_h; ++old_ly) {
      for (int old_lx = 0; old_lx < old_w; ++old_lx) {
        // Get global coordinate from old bbox
        int gx, gy, gz;
        bbox_with_halo.local_to_global(old_lx, old_ly, old_lz, gx, gy, gz, Nx,
                                       Ny, Nz);

        // Convert to local coordinate in new bbox using proper periodic
        // conversion
        int new_lx, new_ly, new_lz;
        new_bbox_with_halo.global_to_local(gx, gy, gz, new_lx, new_ly, new_lz,
                                           Nx, Ny, Nz);

        // Bounds check on destination
        if (new_lx >= 0 && new_lx < new_w && new_ly >= 0 && new_ly < new_h &&
            new_lz >= 0 && new_lz < new_d) {
          int old_idx = old_lz * (old_w * old_h) + old_ly * old_w + old_lx;
          int new_idx = new_lz * (new_w * new_h) + new_ly * new_w + new_lx;
          h_phi_new[new_idx] = h_phi[old_idx];
        }
      }
    }
  }

  // Free old memory
  free_device_memory();

  // Update bbox
  bbox = new_bbox;
  bbox_with_halo = new_bbox_with_halo;
  field_size = new_size;

  // Allocate and upload new field
  allocate_device_memory();
  cudaMemcpy(d_phi, h_phi_new.data(), new_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(d_dphi_dt, 0, new_size * sizeof(float));

  return true;
}

} // namespace cellsim
