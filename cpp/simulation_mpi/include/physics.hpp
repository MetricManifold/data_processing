#pragma once

#include "types.hpp"
#include <cmath>

namespace cellsim {

//=============================================================================
// Physics Functions - CPU versions matching CUDA exactly
//=============================================================================

/**
 * Bulk potential derivative: f'(φ) = (60/λ²) * φ(1-φ)(1-2φ)
 */
inline float compute_bulk_term(float phi, float bulk_coeff) {
  float one_minus_phi = 1.0f - phi;
  float one_minus_2phi = 1.0f - 2.0f * phi;
  return 2.0f * bulk_coeff * phi * one_minus_phi * one_minus_2phi;
}

/**
 * Volume constraint term contribution.
 * δE/δφ = -4 * (μ/V_target) * (V_target - V) * φ
 */
inline float compute_volume_constraint_term(float phi, float volume_deviation,
                                            float volume_coeff) {
  return -4.0f * volume_coeff * volume_deviation * phi;
}

/**
 * Repulsion term from cell-cell interactions.
 * δE/δφ_i = (60κ/λ²) * φ_i * Σ_j φ_j²
 */
inline float compute_repulsion_term(float phi, float interaction_sum,
                                    float interaction_coeff) {
  return 2.0f * interaction_coeff * phi * interaction_sum;
}

/**
 * Advection term: v · ∇φ
 */
inline float compute_advection_term(float grad_x, float grad_y,
                                    float vx, float vy) {
  return vx * grad_x + vy * grad_y;
}

/**
 * Combine all terms into the full equation of motion.
 * dφ/dt = -v·∇φ - 0.5 * (-2γ∇²φ + f'(φ) + volume_constraint + repulsion)
 */
inline float combine_rhs_terms(float laplacian, float bulk_term,
                               float constraint_term, float repulsion_term,
                               float advection_term, float gamma) {
  float functional_derivative =
      -2.0f * gamma * laplacian + bulk_term + constraint_term + repulsion_term;
  return -advection_term - 0.5f * functional_derivative;
}

//=============================================================================
// Stencil Operations - 2D with Neumann BC
//=============================================================================

/**
 * 5-point Laplacian stencil with Neumann BC at subdomain boundaries.
 */
inline float laplacian_5pt(const float *phi, int idx, int width, int height,
                           int lx, int ly, float inv_dx2, float inv_dy2) {
  // Neumann BC at subdomain boundaries
  int lx_m = (lx > 0) ? lx - 1 : 0;
  int lx_p = (lx < width - 1) ? lx + 1 : width - 1;
  int ly_m = (ly > 0) ? ly - 1 : 0;
  int ly_p = (ly < height - 1) ? ly + 1 : height - 1;

  float phi_c = phi[idx];
  float phi_xm = phi[ly * width + lx_m];
  float phi_xp = phi[ly * width + lx_p];
  float phi_ym = phi[ly_m * width + lx];
  float phi_yp = phi[ly_p * width + lx];

  float d2x = (phi_xp - 2.0f * phi_c + phi_xm) * inv_dx2;
  float d2y = (phi_yp - 2.0f * phi_c + phi_ym) * inv_dy2;

  return d2x + d2y;
}

/**
 * 2D gradient using central differences with Neumann BC.
 */
inline void gradient_2d(const float *phi, int idx, int width, int height,
                        int lx, int ly, float inv_2dx, float inv_2dy,
                        float &grad_x, float &grad_y) {
  // Neumann BC at subdomain boundaries
  int lx_m = (lx > 0) ? lx - 1 : 0;
  int lx_p = (lx < width - 1) ? lx + 1 : width - 1;
  int ly_m = (ly > 0) ? ly - 1 : 0;
  int ly_p = (ly < height - 1) ? ly + 1 : height - 1;

  grad_x = (phi[ly * width + lx_p] - phi[ly * width + lx_m]) * inv_2dx;
  grad_y = (phi[ly_p * width + lx] - phi[ly_m * width + lx]) * inv_2dy;
}

//=============================================================================
// Tanh interface profile - initialization helper
//=============================================================================

/**
 * Compute tanh interface profile for initializing cells.
 */
inline float tanh_profile(float r, float R, float lambda) {
  float w = sqrtf(2.0f) * lambda;
  return 0.5f * (1.0f - tanhf((r - R) / w));
}

/**
 * Compute effective radius for initialization.
 */
inline float effective_radius_2d(float target_radius, float lambda) {
  float w = sqrtf(2.0f) * lambda;
  float w2_over_3 = (w * w) / 3.0f;
  if (target_radius * target_radius > w2_over_3) {
    return sqrtf(target_radius * target_radius - w2_over_3);
  }
  return target_radius;
}

//=============================================================================
// Polarization dynamics
//=============================================================================

/**
 * Check for Run-and-Tumble reorientation event.
 */
inline bool check_tumble_event(float dt, float tau, float rand_val) {
  float p_tumble = 1.0f - expf(-dt / tau);
  return rand_val < p_tumble;
}

} // namespace cellsim
