#pragma once

#include "types.cuh"
#include "types3d.cuh"
#include <cuda_runtime.h>

namespace cellsim {

//=============================================================================
// Physics Device Functions - Reusable across 2D and 3D
// These compute the local contributions to the equations of motion
//=============================================================================

/**
 * Bulk potential derivative: f'(φ) = (60/λ²) * φ(1-φ)(1-2φ)
 *
 * The potential is f(φ) = (30/λ²) * φ²(1-φ)²
 * This enforces the phase field stays between 0 and 1.
 *
 * @param phi Phase field value at this point
 * @param bulk_coeff Pre-computed 30/λ²
 * @return f'(φ) contribution to dφ/dt
 */
__device__ __forceinline__ float compute_bulk_term(float phi,
                                                   float bulk_coeff) {
  float one_minus_phi = 1.0f - phi;
  float one_minus_2phi = 1.0f - 2.0f * phi;
  // f'(φ) = 2 * (30/λ²) * φ(1-φ)(1-2φ)
  return 2.0f * bulk_coeff * phi * one_minus_phi * one_minus_2phi;
}

/**
 * Volume constraint term contribution to each grid point.
 *
 * Energy: E_vol = (μ/V_target) * (V_target - ∫φ² dV)²
 * Functional derivative: δE/δφ = -4 * (μ/V_target) * (V_target - V) * φ
 *
 * @param phi Phase field value at this point
 * @param volume_deviation (V_target - current_volume)
 * @param volume_coeff Pre-computed μ/V_target
 * @return Volume constraint contribution to dφ/dt
 */
__device__ __forceinline__ float
compute_volume_constraint_term(float phi, float volume_deviation,
                               float volume_coeff) {
  // Sign is negative because deviation = target - current
  return -4.0f * volume_coeff * volume_deviation * phi;
}

/**
 * Repulsion term from cell-cell interactions.
 *
 * Energy: E_int = (30κ/λ²) ∫ φ_i² * Σ_j φ_j² dV
 * Functional derivative: δE/δφ_i = (60κ/λ²) * φ_i * Σ_j φ_j²
 *
 * @param phi Phase field value of this cell at this point
 * @param interaction_sum Σ_j φ_j² from all other cells at this point
 * @param interaction_coeff Pre-computed 30κ/λ² (doubled in derivative)
 * @return Repulsion contribution to dφ/dt
 */
__device__ __forceinline__ float
compute_repulsion_term(float phi, float interaction_sum,
                       float interaction_coeff) {
  // Factor of 2 comes from derivative of φ_i²
  return 2.0f * interaction_coeff * phi * interaction_sum;
}

/**
 * Advection term: v · ∇φ
 *
 * This transports the phase field with velocity v.
 *
 * @param grad Gradient of φ (2D or 3D vector)
 * @param velocity Cell velocity vector
 * @return Advection contribution to dφ/dt
 */
__device__ __forceinline__ float
compute_advection_term_2d(float grad_x, float grad_y, float vx, float vy) {
  return vx * grad_x + vy * grad_y;
}

__device__ __forceinline__ float
compute_advection_term_3d(float grad_x, float grad_y, float grad_z, float vx,
                          float vy, float vz) {
  return vx * grad_x + vy * grad_y + vz * grad_z;
}

/**
 * Combine all terms into the full equation of motion.
 *
 * dφ/dt = -v·∇φ - 0.5 * (-2γ∇²φ + f'(φ) + volume_constraint + repulsion)
 *
 * The 0.5 factor comes from the relaxational dynamics.
 * The -2γ is the coefficient of the Laplacian (stabilizes interface).
 *
 * @param laplacian ∇²φ at this point
 * @param bulk_term f'(φ)
 * @param constraint_term Volume constraint contribution
 * @param repulsion_term Cell-cell repulsion contribution
 * @param advection_term v·∇φ
 * @param gamma Gradient energy coefficient
 * @return Full dφ/dt
 */
__device__ __forceinline__ float
combine_rhs_terms(float laplacian, float bulk_term, float constraint_term,
                  float repulsion_term, float advection_term, float gamma) {
  // Relaxational dynamics: dφ/dt = -δF/δφ
  // F = ∫[γ(∇φ)² + f(φ) + volume_term + interaction_term] dV
  // -δF/δφ = 2γ∇²φ - f'(φ) - volume_term - interaction_term
  // With advection: dφ/dt = -v·∇φ + 0.5 * (2γ∇²φ - f'(φ) - ...)
  float functional_derivative =
      -2.0f * gamma * laplacian + bulk_term + constraint_term + repulsion_term;
  return -advection_term - 0.5f * functional_derivative;
}

//=============================================================================
// Stencil Operations - 2D
//=============================================================================

/**
 * 5-point Laplacian stencil for 2D.
 * Uses Neumann BC (zero gradient) at subdomain boundaries.
 */
__device__ __forceinline__ float laplacian_5pt(const float *phi, int idx,
                                               int width, int height, int lx,
                                               int ly, float inv_dx2,
                                               float inv_dy2) {

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
__device__ __forceinline__ void
gradient_2d(const float *phi, int idx, int width, int height, int lx, int ly,
            float inv_2dx, float inv_2dy, float &grad_x, float &grad_y) {

  // Neumann BC at subdomain boundaries
  int lx_m = (lx > 0) ? lx - 1 : 0;
  int lx_p = (lx < width - 1) ? lx + 1 : width - 1;
  int ly_m = (ly > 0) ? ly - 1 : 0;
  int ly_p = (ly < height - 1) ? ly + 1 : height - 1;

  grad_x = (phi[ly * width + lx_p] - phi[ly * width + lx_m]) * inv_2dx;
  grad_y = (phi[ly_p * width + lx] - phi[ly_m * width + lx]) * inv_2dy;
}

//=============================================================================
// Stencil Operations - 3D
//=============================================================================

/**
 * 7-point Laplacian stencil for 3D.
 * Uses Neumann BC (zero gradient) at subdomain boundaries.
 * Note: For cells crossing domain boundaries, the halo should be filled
 * with wrapped data before calling this function.
 */
__device__ __forceinline__ float laplacian_7pt(const float *phi, int idx,
                                               int width, int height, int depth,
                                               int lx, int ly, int lz,
                                               float inv_dx2, float inv_dy2,
                                               float inv_dz2) {

  int wh = width * height; // stride for z

  // Neumann BC at subdomain boundaries (assumes halo is properly filled for
  // cells that wrap)
  int lx_m = (lx > 0) ? lx - 1 : 0;
  int lx_p = (lx < width - 1) ? lx + 1 : width - 1;
  int ly_m = (ly > 0) ? ly - 1 : 0;
  int ly_p = (ly < height - 1) ? ly + 1 : height - 1;
  int lz_m = (lz > 0) ? lz - 1 : 0;
  int lz_p = (lz < depth - 1) ? lz + 1 : depth - 1;

  float phi_c = phi[idx];
  float phi_xm = phi[lz * wh + ly * width + lx_m];
  float phi_xp = phi[lz * wh + ly * width + lx_p];
  float phi_ym = phi[lz * wh + ly_m * width + lx];
  float phi_yp = phi[lz * wh + ly_p * width + lx];
  float phi_zm = phi[lz_m * wh + ly * width + lx];
  float phi_zp = phi[lz_p * wh + ly * width + lx];

  float d2x = (phi_xp - 2.0f * phi_c + phi_xm) * inv_dx2;
  float d2y = (phi_yp - 2.0f * phi_c + phi_ym) * inv_dy2;
  float d2z = (phi_zp - 2.0f * phi_c + phi_zm) * inv_dz2;

  return d2x + d2y + d2z;
}

/**
 * 3D gradient using central differences with Neumann BC.
 * Note: For cells crossing domain boundaries, the halo should be filled
 * with wrapped data before calling this function.
 */
__device__ __forceinline__ void
gradient_3d(const float *phi, int idx, int width, int height, int depth, int lx,
            int ly, int lz, float inv_2dx, float inv_2dy, float inv_2dz,
            float &grad_x, float &grad_y, float &grad_z) {

  int wh = width * height;

  // Neumann BC at subdomain boundaries
  int lx_m = (lx > 0) ? lx - 1 : 0;
  int lx_p = (lx < width - 1) ? lx + 1 : width - 1;
  int ly_m = (ly > 0) ? ly - 1 : 0;
  int ly_p = (ly < height - 1) ? ly + 1 : height - 1;
  int lz_m = (lz > 0) ? lz - 1 : 0;
  int lz_p = (lz < depth - 1) ? lz + 1 : depth - 1;

  grad_x =
      (phi[lz * wh + ly * width + lx_p] - phi[lz * wh + ly * width + lx_m]) *
      inv_2dx;
  grad_y =
      (phi[lz * wh + ly_p * width + lx] - phi[lz * wh + ly_m * width + lx]) *
      inv_2dy;
  grad_z =
      (phi[lz_p * wh + ly * width + lx] - phi[lz_m * wh + ly * width + lx]) *
      inv_2dz;
}

//=============================================================================
// Periodic-aware stencil operations for 3D
// These use periodic BC within the subdomain when the subdomain wraps around
// the domain boundary.
//=============================================================================

/**
 * 7-point Laplacian stencil for 3D with periodic BC within subdomain.
 * Use this when the cell's subdomain wraps around the domain boundary.
 *
 * @param wrap_x True if subdomain wraps in x (i.e., x0 < 0 or x1 > Nx)
 * @param wrap_y True if subdomain wraps in y
 * @param wrap_z True if subdomain wraps in z
 */
__device__ __forceinline__ float
laplacian_7pt_periodic(const float *phi, int idx, int width, int height,
                       int depth, int lx, int ly, int lz, float inv_dx2,
                       float inv_dy2, float inv_dz2, bool wrap_x, bool wrap_y,
                       bool wrap_z) {

  int wh = width * height;

  // Handle each dimension based on whether it wraps
  int lx_m, lx_p, ly_m, ly_p, lz_m, lz_p;

  if (wrap_x) {
    // Periodic wrapping within subdomain
    lx_m = (lx > 0) ? lx - 1 : width - 1;
    lx_p = (lx < width - 1) ? lx + 1 : 0;
  } else {
    // Neumann BC (clamp)
    lx_m = (lx > 0) ? lx - 1 : 0;
    lx_p = (lx < width - 1) ? lx + 1 : width - 1;
  }

  if (wrap_y) {
    ly_m = (ly > 0) ? ly - 1 : height - 1;
    ly_p = (ly < height - 1) ? ly + 1 : 0;
  } else {
    ly_m = (ly > 0) ? ly - 1 : 0;
    ly_p = (ly < height - 1) ? ly + 1 : height - 1;
  }

  if (wrap_z) {
    lz_m = (lz > 0) ? lz - 1 : depth - 1;
    lz_p = (lz < depth - 1) ? lz + 1 : 0;
  } else {
    lz_m = (lz > 0) ? lz - 1 : 0;
    lz_p = (lz < depth - 1) ? lz + 1 : depth - 1;
  }

  float phi_c = phi[idx];
  float phi_xm = phi[lz * wh + ly * width + lx_m];
  float phi_xp = phi[lz * wh + ly * width + lx_p];
  float phi_ym = phi[lz * wh + ly_m * width + lx];
  float phi_yp = phi[lz * wh + ly_p * width + lx];
  float phi_zm = phi[lz_m * wh + ly * width + lx];
  float phi_zp = phi[lz_p * wh + ly * width + lx];

  float d2x = (phi_xp - 2.0f * phi_c + phi_xm) * inv_dx2;
  float d2y = (phi_yp - 2.0f * phi_c + phi_ym) * inv_dy2;
  float d2z = (phi_zp - 2.0f * phi_c + phi_zm) * inv_dz2;

  return d2x + d2y + d2z;
}

/**
 * 3D gradient with periodic BC within subdomain for wrapping cells.
 */
__device__ __forceinline__ void
gradient_3d_periodic(const float *phi, int idx, int width, int height,
                     int depth, int lx, int ly, int lz, float inv_2dx,
                     float inv_2dy, float inv_2dz, float &grad_x, float &grad_y,
                     float &grad_z, bool wrap_x, bool wrap_y, bool wrap_z) {

  int wh = width * height;

  int lx_m, lx_p, ly_m, ly_p, lz_m, lz_p;

  if (wrap_x) {
    lx_m = (lx > 0) ? lx - 1 : width - 1;
    lx_p = (lx < width - 1) ? lx + 1 : 0;
  } else {
    lx_m = (lx > 0) ? lx - 1 : 0;
    lx_p = (lx < width - 1) ? lx + 1 : width - 1;
  }

  if (wrap_y) {
    ly_m = (ly > 0) ? ly - 1 : height - 1;
    ly_p = (ly < height - 1) ? ly + 1 : 0;
  } else {
    ly_m = (ly > 0) ? ly - 1 : 0;
    ly_p = (ly < height - 1) ? ly + 1 : height - 1;
  }

  if (wrap_z) {
    lz_m = (lz > 0) ? lz - 1 : depth - 1;
    lz_p = (lz < depth - 1) ? lz + 1 : 0;
  } else {
    lz_m = (lz > 0) ? lz - 1 : 0;
    lz_p = (lz < depth - 1) ? lz + 1 : depth - 1;
  }

  grad_x =
      (phi[lz * wh + ly * width + lx_p] - phi[lz * wh + ly * width + lx_m]) *
      inv_2dx;
  grad_y =
      (phi[lz * wh + ly_p * width + lx] - phi[lz * wh + ly_m * width + lx]) *
      inv_2dy;
  grad_z =
      (phi[lz_p * wh + ly * width + lx] - phi[lz_m * wh + ly * width + lx]) *
      inv_2dz;
}

//=============================================================================
// Motility integrands - used for computing cell velocities
//=============================================================================

/**
 * Compute motility integrand: φ * ∇φ * Σ_m φ_m²
 * This is integrated over the cell domain to get the velocity.
 *
 * v_n = (60κ/ξλ²) ∫ φ_n (∇φ_n) Σ_m φ_m² dV
 */
__device__ __forceinline__ void motility_integrand_2d(float phi, float grad_x,
                                                      float grad_y,
                                                      float interaction_sum,
                                                      float &integrand_x,
                                                      float &integrand_y) {
  float factor = phi * interaction_sum;
  integrand_x = factor * grad_x;
  integrand_y = factor * grad_y;
}

__device__ __forceinline__ void
motility_integrand_3d(float phi, float grad_x, float grad_y, float grad_z,
                      float interaction_sum, float &integrand_x,
                      float &integrand_y, float &integrand_z) {
  float factor = phi * interaction_sum;
  integrand_x = factor * grad_x;
  integrand_y = factor * grad_y;
  integrand_z = factor * grad_z;
}

//=============================================================================
// Polarization dynamics - same in 2D and 3D
//=============================================================================

/**
 * Update polarization direction for Active Brownian Particle model.
 * Uses Wiener process for rotational diffusion.
 *
 * @param p Current polarization (unit vector)
 * @param D_r Rotational diffusion coefficient = 1/(2τ)
 * @param dt Time step
 * @param rand_val Random value from normal distribution N(0,1)
 * @return New polarization (unit vector)
 */
__device__ __forceinline__ Vec2 update_polarization_abp_2d(Vec2 p, float D_r,
                                                           float dt,
                                                           float rand_val) {
  // Current angle
  float theta = atan2f(p.y, p.x);
  // Add rotational noise
  float dtheta = sqrtf(2.0f * D_r * dt) * rand_val;
  theta += dtheta;
  return {cosf(theta), sinf(theta)};
}

/**
 * Check for Run-and-Tumble reorientation event.
 *
 * @param dt Time step
 * @param tau Mean run time
 * @param rand_val Uniform random value in [0,1]
 * @return true if tumble event occurs
 */
__device__ __forceinline__ bool check_tumble_event(float dt, float tau,
                                                   float rand_val) {
  // Poisson process: probability of tumble in dt is 1 - exp(-dt/τ) ≈ dt/τ
  float p_tumble = 1.0f - expf(-dt / tau);
  return rand_val < p_tumble;
}

//=============================================================================
// Tanh interface profile - initialization helper
//=============================================================================

/**
 * Compute tanh interface profile for initializing cells.
 * φ = 0.5 * (1 - tanh((r - R) / w))
 * where w = sqrt(2) * λ is the interface width.
 *
 * @param r Distance from cell center
 * @param R Cell radius (effective, adjusted for interface width)
 * @param lambda Interface width parameter
 * @return Phase field value
 */
__device__ __host__ __forceinline__ float tanh_profile(float r, float R,
                                                       float lambda) {
  float w = sqrtf(2.0f) * lambda;
  return 0.5f * (1.0f - tanhf((r - R) / w));
}

/**
 * Compute effective radius for initialization.
 *
 * For a tanh profile, the volume integral ∫φ² dA includes interface
 * contribution. To get target volume V_target = πR², we use a smaller
 * effective radius R_eff = sqrt(R² - w²/3) for the tanh profile.
 *
 * @param target_radius Target cell radius
 * @param lambda Interface width parameter
 * @return Effective radius for tanh profile initialization
 */
__host__ __forceinline__ float effective_radius_2d(float target_radius,
                                                   float lambda) {
  float w = sqrtf(2.0f) * lambda;
  float w2_over_3 = (w * w) / 3.0f;
  if (target_radius * target_radius > w2_over_3) {
    return sqrtf(target_radius * target_radius - w2_over_3);
  }
  return target_radius;
}

/**
 * Compute effective radius for 3D initialization.
 *
 * For a spherical cell with tanh profile φ = 0.5(1 - tanh((r-R_eff)/w)),
 * the volume integral ∫φ² dV depends on the interface width.
 *
 * Empirical formula (fitted to numerical integration):
 *   R_eff = R_target + 0.693*λ - 1.018*λ²/R_target
 *
 * This gives initial volume within 0.3% of target for R >= 20, λ = 7.
 */
__host__ __forceinline__ float effective_radius_3d(float target_radius,
                                                   float lambda) {
  // Empirical correction for 3D spherical cells with tanh interface
  // R_eff = R + c1*λ - c2*λ²/R where c1 ≈ 0.693, c2 ≈ 1.018
  float c1 = 0.693f;
  float c2 = 1.018f;
  return target_radius + c1 * lambda - c2 * lambda * lambda / target_radius;
}

} // namespace cellsim
