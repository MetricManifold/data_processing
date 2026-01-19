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
  float dt = 0.01f;        // Time step
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
  int x0, y0; // Lower-left corner (in global coords)
  int x1, y1; // Upper-right corner (exclusive)

  int width() const { return x1 - x0; }
  int height() const { return y1 - y0; }
  int size() const { return width() * height(); }

  // Check if a global coordinate is inside this box (with periodic wrapping)
  bool contains(int gx, int gy, int Nx, int Ny) const {
    // Wrap to [0, N)
    int wx = ((gx % Nx) + Nx) % Nx;
    int wy = ((gy % Ny) + Ny) % Ny;

    // Handle periodic box that wraps around domain edge
    bool in_x, in_y;
    if (x0 < 0) {
      in_x = (wx >= (x0 + Nx)) || (wx < x1);
    } else if (x1 > Nx) {
      in_x = (wx >= x0) || (wx < (x1 - Nx));
    } else {
      in_x = (wx >= x0) && (wx < x1);
    }

    if (y0 < 0) {
      in_y = (wy >= (y0 + Ny)) || (wy < y1);
    } else if (y1 > Ny) {
      in_y = (wy >= y0) || (wy < (y1 - Ny));
    } else {
      in_y = (wy >= y0) && (wy < y1);
    }

    return in_x && in_y;
  }

  // Convert global coords to local subdomain coords
  void global_to_local(int gx, int gy, int &lx, int &ly, int Nx, int Ny) const {
    lx = ((gx - x0) % Nx + Nx) % Nx;
    ly = ((gy - y0) % Ny + Ny) % Ny;
    // Clamp to subdomain size
    if (lx >= width()) lx -= Nx;
    if (ly >= height()) ly -= Ny;
  }

  // Convert local subdomain coords to global coords
  void local_to_global(int lx, int ly, int &gx, int &gy, int Nx, int Ny) const {
    gx = ((x0 + lx) % Nx + Nx) % Nx;
    gy = ((y0 + ly) % Ny + Ny) % Ny;
  }

  // Check if two bounding boxes overlap (considering periodic BC)
  bool overlaps(const BoundingBox &other, int Nx, int Ny) const {
    float cx1 = x0 + width() * 0.5f;
    float cy1 = y0 + height() * 0.5f;
    float cx2 = other.x0 + other.width() * 0.5f;
    float cy2 = other.y0 + other.height() * 0.5f;

    // Periodic distance
    float dx = fabsf(cx2 - cx1);
    float dy = fabsf(cy2 - cy1);
    if (dx > Nx * 0.5f) dx = Nx - dx;
    if (dy > Ny * 0.5f) dy = Ny - dy;

    float hw = (width() + other.width()) * 0.5f;
    float hh = (height() + other.height()) * 0.5f;

    return (dx < hw) && (dy < hh);
  }

  // Expand box by a margin (for halo cells)
  BoundingBox expanded(int margin) const {
    return {x0 - margin, y0 - margin, x1 + margin, y1 + margin};
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
