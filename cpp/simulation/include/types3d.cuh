#pragma once

#include "types.cuh"

namespace cellsim {

//=============================================================================
// 3D Vector helper
//=============================================================================

struct Vec3 {
  float x, y, z;

  __host__ __device__ Vec3() : x(0), y(0), z(0) {}
  __host__ __device__ Vec3(float x_, float y_, float z_)
      : x(x_), y(y_), z(z_) {}

  // Convert from Vec2 (z = 0)
  __host__ __device__ explicit Vec3(const Vec2 &v) : x(v.x), y(v.y), z(0) {}

  __host__ __device__ Vec3 operator+(const Vec3 &v) const {
    return {x + v.x, y + v.y, z + v.z};
  }
  __host__ __device__ Vec3 operator-(const Vec3 &v) const {
    return {x - v.x, y - v.y, z - v.z};
  }
  __host__ __device__ Vec3 operator*(float s) const {
    return {x * s, y * s, z * s};
  }
  __host__ __device__ Vec3 &operator+=(const Vec3 &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  __host__ __device__ Vec3 &operator-=(const Vec3 &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  __host__ __device__ Vec3 &operator*=(float s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }
  __host__ __device__ float dot(const Vec3 &v) const {
    return x * v.x + y * v.y + z * v.z;
  }
  __host__ __device__ Vec3 cross(const Vec3 &v) const {
    return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
  }
  __host__ __device__ float norm() const {
    return sqrtf(x * x + y * y + z * z);
  }
  __host__ __device__ float norm_sq() const { return x * x + y * y + z * z; }
  __host__ __device__ Vec3 normalized() const {
    float n = norm();
    return n > 1e-8f ? Vec3{x / n, y / n, z / n} : Vec3{0, 0, 0};
  }

  // Project to 2D (drop z)
  __host__ __device__ Vec2 xy() const { return {x, y}; }
};

// Scalar multiplication from left
__host__ __device__ inline Vec3 operator*(float s, const Vec3 &v) {
  return v * s;
}

//=============================================================================
// 3D Bounding Box - Subdomain definition with periodic wrapping
//=============================================================================

struct BoundingBox3D {
  int x0, y0, z0; // Lower corner (in global coords)
  int x1, y1, z1; // Upper corner (exclusive)

  __host__ __device__ int width() const { return x1 - x0; }
  __host__ __device__ int height() const { return y1 - y0; }
  __host__ __device__ int depth() const { return z1 - z0; }
  __host__ __device__ int size() const { return width() * height() * depth(); }

  // 3D index from local coordinates
  __host__ __device__ int index(int lx, int ly, int lz) const {
    return lz * (width() * height()) + ly * width() + lx;
  }

  // Check if a global coordinate is inside this box (with periodic wrapping)
  __host__ __device__ bool contains(int gx, int gy, int gz, int Nx, int Ny,
                                    int Nz) const {
    // Wrap to [0, N)
    int wx = ((gx % Nx) + Nx) % Nx;
    int wy = ((gy % Ny) + Ny) % Ny;
    int wz = ((gz % Nz) + Nz) % Nz;

    // Handle periodic box that wraps around domain edge
    auto in_range = [](int coord, int lo, int hi, int N) -> bool {
      if (lo < 0) {
        return (coord >= (lo + N)) || (coord < hi);
      } else if (hi > N) {
        return (coord >= lo) || (coord < (hi - N));
      } else {
        return (coord >= lo) && (coord < hi);
      }
    };

    return in_range(wx, x0, x1, Nx) && in_range(wy, y0, y1, Ny) &&
           in_range(wz, z0, z1, Nz);
  }

  // Convert global coords to local subdomain coords
  __host__ __device__ void global_to_local(int gx, int gy, int gz, int &lx,
                                           int &ly, int &lz, int Nx, int Ny,
                                           int Nz) const {
    lx = ((gx - x0) % Nx + Nx) % Nx;
    ly = ((gy - y0) % Ny + Ny) % Ny;
    lz = ((gz - z0) % Nz + Nz) % Nz;
    // Clamp to subdomain size (handles wraparound)
    if (lx >= width())
      lx -= Nx;
    if (ly >= height())
      ly -= Ny;
    if (lz >= depth())
      lz -= Nz;
  }

  // Convert local subdomain coords to global coords
  __host__ __device__ void local_to_global(int lx, int ly, int lz, int &gx,
                                           int &gy, int &gz, int Nx, int Ny,
                                           int Nz) const {
    gx = ((x0 + lx) % Nx + Nx) % Nx;
    gy = ((y0 + ly) % Ny + Ny) % Ny;
    gz = ((z0 + lz) % Nz + Nz) % Nz;
  }

  // Check if two bounding boxes overlap (considering periodic BC)
  __host__ __device__ bool overlaps(const BoundingBox3D &other, int Nx, int Ny,
                                    int Nz) const {
    // Check if distance between centers is less than sum of half-widths
    float cx1 = x0 + width() * 0.5f;
    float cy1 = y0 + height() * 0.5f;
    float cz1 = z0 + depth() * 0.5f;
    float cx2 = other.x0 + other.width() * 0.5f;
    float cy2 = other.y0 + other.height() * 0.5f;
    float cz2 = other.z0 + other.depth() * 0.5f;

    // Periodic distance
    float dx = fabsf(cx2 - cx1);
    float dy = fabsf(cy2 - cy1);
    float dz = fabsf(cz2 - cz1);
    if (dx > Nx * 0.5f)
      dx = Nx - dx;
    if (dy > Ny * 0.5f)
      dy = Ny - dy;
    if (dz > Nz * 0.5f)
      dz = Nz - dz;

    float hw = (width() + other.width()) * 0.5f;
    float hh = (height() + other.height()) * 0.5f;
    float hd = (depth() + other.depth()) * 0.5f;

    return (dx < hw) && (dy < hh) && (dz < hd);
  }

  // Expand box by a margin (for halo cells)
  __host__ BoundingBox3D expanded(int margin) const {
    return {x0 - margin, y0 - margin, z0 - margin,
            x1 + margin, y1 + margin, z1 + margin};
  }
};

//=============================================================================
// 3D Simulation Parameters
//=============================================================================

struct SimParams3D {
  // Domain size
  int Nx = 128;    // Global domain width
  int Ny = 128;    // Global domain height
  int Nz = 128;    // Global domain depth
  float dx = 1.0f; // Grid spacing x
  float dy = 1.0f; // Grid spacing y
  float dz = 1.0f; // Grid spacing z

  // Time stepping
  float dt = 0.01f;        // Time step
  float t_end = 100.0f;    // End time
  int save_interval = 100; // Steps between saves

  // Interface parameters (from paper Table 1)
  float lambda = 7.0f; // Interface width λ = 7
  float gamma = 1.0f;  // Gradient coefficient γ = 1
  __host__ __device__ float bulk_coeff() const {
    return 30.0f / (lambda * lambda);
  }

  // Interaction
  float kappa = 10.0f; // Interaction strength κ = 10
  __host__ __device__ float interaction_coeff() const {
    return 30.0f * kappa / (lambda * lambda);
  }

  // Volume constraint - 3D uses sphere volume
  float target_radius = 20.0f; // Smaller default for 3D (memory)
  __host__ __device__ float target_volume() const {
    return (4.0f / 3.0f) * M_PI * target_radius * target_radius * target_radius;
  }
  // Volume constraint strength μ (energy scale)
  // The relaxation rate for δV/V₀ is 4μ, independent of dimension.
  // Use μ = 1 for same physics as 2D.
  float mu = 1.0f; // Volume constraint strength
  __host__ __device__ float volume_coeff() const {
    return mu / target_volume();
  }

  // Motility
  float v_A = 0.0f;   // Active motility speed (default 0 = no motility)
  float xi = 1.5e3f;  // Friction coefficient
  float tau = 1.0e4f; // Reorientation time

  __host__ __device__ float motility_coeff() const {
    return 60.0f * kappa / (xi * lambda * lambda);
  }

  // Subdomain management
  int halo_width = 4;
  int min_subdomain_size = 16;
  float subdomain_padding = 2.0f;

  // Motility model
  SimParams::MotilityModel motility_model =
      SimParams::MotilityModel::RunAndTumble;

  // Grid cell volume for integration
  __host__ __device__ float cell_volume() const { return dx * dy * dz; }
};

//=============================================================================
// Dimension-agnostic physics coefficients
// These can be used in templated kernels
//=============================================================================

struct PhysicsCoeffs {
  float bulk_coeff;        // 30/λ²
  float interaction_coeff; // 30κ/λ²
  float volume_coeff;      // μ/target_volume
  float motility_coeff;    // 60κ/(ξλ²)
  float gamma;             // Gradient coefficient
  float target_volume;     // πR² (2D) or (4/3)πR³ (3D)
  float cell_volume;       // dx*dy (2D) or dx*dy*dz (3D)

  // Construct from 2D params
  __host__ static PhysicsCoeffs from_2d(const SimParams &p) {
    PhysicsCoeffs c;
    c.bulk_coeff = p.bulk_coeff();
    c.interaction_coeff = p.interaction_coeff();
    c.volume_coeff = p.volume_coeff();
    c.motility_coeff = p.motility_coeff();
    c.gamma = p.gamma;
    c.target_volume = p.target_area();
    c.cell_volume = p.dx * p.dy;
    return c;
  }

  // Construct from 3D params
  __host__ static PhysicsCoeffs from_3d(const SimParams3D &p) {
    PhysicsCoeffs c;
    c.bulk_coeff = p.bulk_coeff();
    c.interaction_coeff = p.interaction_coeff();
    c.volume_coeff = p.volume_coeff();
    c.motility_coeff = p.motility_coeff();
    c.gamma = p.gamma;
    c.target_volume = p.target_volume();
    c.cell_volume = p.cell_volume();
    return c;
  }
};

//=============================================================================
// Dimension trait for template specialization
//=============================================================================

template <int Dim> struct DimTraits;

template <> struct DimTraits<2> {
  using Vec = Vec2;
  using BBox = BoundingBox;
  using Params = SimParams;
  static constexpr int dimensions = 2;
  static constexpr int stencil_size = 5; // 5-point stencil
};

template <> struct DimTraits<3> {
  using Vec = Vec3;
  using BBox = BoundingBox3D;
  using Params = SimParams3D;
  static constexpr int dimensions = 3;
  static constexpr int stencil_size = 7; // 7-point stencil
};

} // namespace cellsim
