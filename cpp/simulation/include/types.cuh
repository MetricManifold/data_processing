#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cellsim {

// Adhesion h(φ) exponent: h(φ) = φ²(1-φ)^ADHESION_N, peaks at φ* = 2/(N+2).
// "φ² matching" normalization: h_norm(φ*) = φ*² (adhesion density matches
// repulsion density at the h peak). For n=4: norm = 1/(2/3)^4 = 81/16.
// Compile-time tunable — update both N and H_PEAK_INV together.
#define ADHESION_N 4
// h_peak_inv = 1/(N/(N+2))^N = ((N+2)/N)^N.  For n=4: (3/2)^4 = 81/16 = 5.0625
#define ADHESION_H_PEAK_INV 5.0625f

//=============================================================================
// Simulation Parameters
//=============================================================================

// CHECKPOINT COMPATIBILITY RULE:
// SimParams is serialized via raw memcpy in checkpoints. To maintain backward
// compatibility with old checkpoints:
//   1. ONLY append new fields at the END of this struct
//   2. NEVER reorder, remove, or change the type of existing fields
//   3. New fields MUST have sensible zero-initialized defaults (0.0f, 0, -1)
//      because old checkpoints will have zeros in those positions
//   4. sizeof(SimParams) is recorded in the checkpoint header as sim_params_size
//   5. On load, io.cu reads min(stored_size, current_size) and zero-inits the rest
// If you need a non-zero default for a new field, add a fixup in io.cu's load path.
struct SimParams {
  // Domain size
  int Nx = 800;    // Global domain width
  int Ny = 800;    // Global domain height
  float dx = 1.0f; // Grid spacing x
  float dy = 1.0f; // Grid spacing y

  // Time stepping
  float dt = 0.01f;        // Time step (Palmieri SI S5: dt = 0.01)
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
  float v_A = 0.0f;       // Active motility speed (default 0 = no motility)
  float xi = 1.5e3f;     // Friction coefficient ξ = 1.5 × 10^3
  float tau = 1.0e4f;    // Reorientation time τ = 10000 (run-and-tumble persistence)

  float motility_coeff() const {
    return 60.0f * kappa / (xi * lambda * lambda);
  } // 60κ/(ξλ²)

  // Subdomain management
  int halo_width = 4;          // Ghost cell width for periodic BC
  int min_subdomain_size = 16; // Minimum subdomain dimension
  float subdomain_padding =
      2.5f; // Expand bbox by this factor (5R total, matching Palmieri SI Sec S5)

  // Motility model: Run-and-Tumble (discrete Poisson reorientations) or
  // Active Brownian Particle (continuous rotational diffusion)
  // NOTE: This field is at the END of the struct for backward compatibility
  // with v3 checkpoints that didn't have this field.
  enum class MotilityModel { RunAndTumble, ABP };
  MotilityModel motility_model = MotilityModel::RunAndTumble;

  // Per-cell motility disorder (added after MotilityModel for checkpoint compat)
  // Not stored in checkpoint — set via CLI only. Old checkpoints read v_A_sigma=0.
  float v_A_sigma = 0.0f; // Std dev for per-cell v_A (log-normal disorder, 0 = uniform)

  // Per-cell stiffness overrides (population or per-cell)
  // Parsed from repeated --gamma flags: --gamma 1.0 --gamma 0.35:20% --gamma 0.5:cell0
  struct GammaOverride {
    float value;
    enum class Type { Fraction, Cells } type;
    float fraction;             // used when type == Fraction (0-1)
    std::vector<int> cell_ids;  // used when type == Cells
  };
  // NOTE: gamma_overrides live on Simulation/Integrator, NOT here,
  // because SimParams is raw-serialized in checkpoints (no std::vector allowed).

  // Per-cell radius overrides (population or per-cell)
  // Parsed from repeated --radius flags: --radius 49 --radius 40:20% --radius 49:cv0.10
  struct RadiusOverride {
    float value;
    enum class Type { Fraction, Cells, CV } type;
    float fraction;             // used when type == Fraction (0-1)
    std::vector<int> cell_ids;  // used when type == Cells
    float cv;                   // used when type == CV (coefficient of variation)
  };
  // NOTE: radius_overrides live on Simulation/Integrator, NOT here.

  // Legacy: --soft-cell / --gamma-soft (backward compat, maps to gamma_overrides)
  int soft_cell_id = -1;
  float gamma_soft = 0.35f;

  // Cell-cell adhesion: gradient coupling (surface tension reduction)
  // F_adh = J Σ_{i<j} ∫ ∇φ_i·∇φ_j dA  (negative at shared interfaces → favorable)
  // δF/δφ_i = -J Σ_{j≠i} ∇²φ_j
  // Implemented via sum field: scatter Σφ_k, compute ∇²(Σφ_k) - ∇²φ_i in fused kernel.
  // Attractive from afar + repulsive at deep overlap → equilibrium at d ≈ 2R.
  // No nucleation, no bulk force, no squishing.  Standard in Nonomura/Löber models.
  // J=0 (default): adhesion disabled.
  float adhesion_J = 0.0f;
};

//=============================================================================
// Bounding Box - Subdomain definition with periodic wrapping
//=============================================================================

struct BoundingBox {
  int x0, y0; // Lower-left corner (in global coords)
  int x1, y1; // Upper-right corner (exclusive)

  __host__ __device__ int width() const { return x1 - x0; }
  __host__ __device__ int height() const { return y1 - y0; }
  __host__ __device__ int size() const { return width() * height(); }

  // Check if a global coordinate is inside this box (with periodic wrapping)
  __host__ __device__ bool contains(int gx, int gy, int Nx, int Ny) const {
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
  __host__ __device__ void global_to_local(int gx, int gy, int &lx, int &ly,
                                           int Nx, int Ny) const {
    lx = ((gx - x0) % Nx + Nx) % Nx;
    ly = ((gy - y0) % Ny + Ny) % Ny;
    // Clamp to subdomain size (handles wraparound)
    if (lx >= width())
      lx -= Nx;
    if (ly >= height())
      ly -= Ny;
  }

  // Convert local subdomain coords to global coords
  __host__ __device__ void local_to_global(int lx, int ly, int &gx, int &gy,
                                           int Nx, int Ny) const {
    gx = ((x0 + lx) % Nx + Nx) % Nx;
    gy = ((y0 + ly) % Ny + Ny) % Ny;
  }

  // Check if two bounding boxes overlap (considering periodic BC)
  __host__ __device__ bool overlaps(const BoundingBox &other, int Nx,
                                    int Ny) const {
    // This is complex with periodic BC - check if any corner of one box is in
    // the other Simplified: check if distance between centers is less than sum
    // of half-widths
    float cx1 = x0 + width() * 0.5f;
    float cy1 = y0 + height() * 0.5f;
    float cx2 = other.x0 + other.width() * 0.5f;
    float cy2 = other.y0 + other.height() * 0.5f;

    // Periodic distance
    float dx = fabsf(cx2 - cx1);
    float dy = fabsf(cy2 - cy1);
    if (dx > Nx * 0.5f)
      dx = Nx - dx;
    if (dy > Ny * 0.5f)
      dy = Ny - dy;

    float hw = (width() + other.width()) * 0.5f;
    float hh = (height() + other.height()) * 0.5f;

    return (dx < hw) && (dy < hh);
  }

  // Expand box by a margin (for halo cells)
  __host__ BoundingBox expanded(int margin) const {
    return {x0 - margin, y0 - margin, x1 + margin, y1 + margin};
  }
};

//=============================================================================
// 2D Vector helper
//=============================================================================

struct Vec2 {
  float x, y;

  __host__ __device__ Vec2() : x(0), y(0) {}
  __host__ __device__ Vec2(float x_, float y_) : x(x_), y(y_) {}

  __host__ __device__ Vec2 operator+(const Vec2 &v) const {
    return {x + v.x, y + v.y};
  }
  __host__ __device__ Vec2 operator-(const Vec2 &v) const {
    return {x - v.x, y - v.y};
  }
  __host__ __device__ Vec2 operator*(float s) const { return {x * s, y * s}; }
  __host__ __device__ float dot(const Vec2 &v) const {
    return x * v.x + y * v.y;
  }
  __host__ __device__ float norm() const { return sqrtf(x * x + y * y); }
  __host__ __device__ Vec2 normalized() const {
    float n = norm();
    return n > 1e-8f ? Vec2{x / n, y / n} : Vec2{0, 0};
  }
};

//=============================================================================
// Cell state flags
//=============================================================================

enum class CellState : uint8_t {
  Active = 0,   // Normal active cell
  Dividing = 1, // Cell is dividing
  Dying = 2,    // Cell is dying/being removed
  Frozen = 3    // Cell is frozen (for debugging)
};

} // namespace cellsim
