#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cellsim {

//=============================================================================
// Simulation Parameters - matches CUDA version exactly
//=============================================================================

struct SimParams {
  // Domain size
  int Nx = 800;    // Global domain width
  int Ny = 800;    // Global domain height
  float dx = 1.0f; // Grid spacing x
  float dy = 1.0f; // Grid spacing y

  // Time stepping
  float dt = 0.02f;        // Time step
  float t_end = 100.0f;    // End time
  int save_interval = 100; // Steps between saves

  // Interface parameters (from paper Table 1)
  float lambda = 7.0f; // Interface width λ = 7
  float gamma = 1.0f;  // Gradient coefficient γ = 1
  float bulk_coeff() const { return 30.0f / (lambda * lambda); } // 30/λ²

  // Interaction
  float kappa = 10.0f; // Interaction strength κ = 10
  float interaction_coeff() const { return 30.0f * kappa / (lambda * lambda); }

  // Volume constraint
  float target_radius = 49.0f; // Target cell radius R = 49
  float target_area() const { return M_PI * target_radius * target_radius; }
  float mu = 1.0f; // Volume constraint strength μ = 1 (from paper Table 1)
  float volume_coeff() const { return mu / target_area(); }

  // Motility
  float v_A = 0.0f;   // Active motility speed (default 0 = no motility)
  float xi = 1.5e3f;  // Friction coefficient ξ = 1.5 × 10^3
  float tau = 1.0e4f; // Reorientation time τ = 10000 (run-and-tumble persistence)

  float motility_coeff() const {
    return 60.0f * kappa / (xi * lambda * lambda);
  } // 60κ/(ξλ²)

  // Subdomain management
  int halo_width = 4;          // Ghost cell width for periodic BC
  int min_subdomain_size = 16; // Minimum subdomain dimension
  float subdomain_padding = 2.0f; // Expand bbox by this factor

  // Motility model
  enum class MotilityModel { RunAndTumble, ABP };
  MotilityModel motility_model = MotilityModel::RunAndTumble;
};

//=============================================================================
// Bounding Box - Subdomain definition with periodic wrapping
//=============================================================================

struct BoundingBox {
  int x_min, y_min; // Lower-left corner (can be negative for periodic wrapping)
  int x_max, y_max; // Upper-right corner (can exceed domain for periodic wrapping)

  // Aliases for compatibility with CUDA version
  int x0() const { return x_min; }
  int y0() const { return y_min; }
  int x1() const { return x_max + 1; }  // Convert to exclusive
  int y1() const { return y_max + 1; }

  int width() const { return x_max - x_min + 1; }
  int height() const { return y_max - y_min + 1; }
  int size() const { return width() * height(); }

  // Expand box by a margin (for halo cells)
  BoundingBox expanded(int margin) const {
    return {x_min - margin, y_min - margin, x_max + margin, y_max + margin};
  }

  // Convert local subdomain coords to global coords (with periodic wrapping)
  void local_to_global(int lx, int ly, int &gx, int &gy, int Nx, int Ny) const {
    gx = ((x_min + lx) % Nx + Nx) % Nx;
    gy = ((y_min + ly) % Ny + Ny) % Ny;
  }

  // Convert global coords to local subdomain coords (with periodic wrapping)
  void global_to_local(int gx, int gy, int &lx, int &ly, int Nx, int Ny) const {
    lx = ((gx - x_min) % Nx + Nx) % Nx;
    ly = ((gy - y_min) % Ny + Ny) % Ny;
    // Clamp to subdomain size (handles wraparound)
    if (lx >= width()) lx -= Nx;
    if (ly >= height()) ly -= Ny;
  }

  // Check if two bounding boxes overlap (considering periodic BC)
  bool overlaps(const BoundingBox &other, int Nx, int Ny) const {
    // Check if distance between centers is less than sum of half-widths
    float cx1 = x_min + width() * 0.5f;
    float cy1 = y_min + height() * 0.5f;
    float cx2 = other.x_min + other.width() * 0.5f;
    float cy2 = other.y_min + other.height() * 0.5f;

    // Periodic distance
    float dx = std::fabs(cx2 - cx1);
    float dy = std::fabs(cy2 - cy1);
    if (dx > Nx * 0.5f) dx = Nx - dx;
    if (dy > Ny * 0.5f) dy = Ny - dy;

    float hw = (width() + other.width()) * 0.5f;
    float hh = (height() + other.height()) * 0.5f;

    return (dx < hw) && (dy < hh);
  }
};

//=============================================================================
// 2D Vector helper
//=============================================================================

struct Vec2 {
  float x, y;

  Vec2() : x(0), y(0) {}
  Vec2(float x_, float y_) : x(x_), y(y_) {}

  Vec2 operator+(const Vec2 &v) const { return {x + v.x, y + v.y}; }
  Vec2 operator-(const Vec2 &v) const { return {x - v.x, y - v.y}; }
  Vec2 operator*(float s) const { return {x * s, y * s}; }
  float dot(const Vec2 &v) const { return x * v.x + y * v.y; }
  float norm() const { return sqrtf(x * x + y * y); }
  Vec2 normalized() const {
    float n = norm();
    return n > 1e-8f ? Vec2{x / n, y / n} : Vec2{0, 0};
  }
};

//=============================================================================
// Cell state flags
//=============================================================================

enum class CellState : uint8_t {
  Active = 0,
  Dividing = 1,
  Dying = 2,
  Frozen = 3
};

} // namespace cellsim
