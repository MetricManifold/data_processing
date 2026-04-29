#pragma once
#include "types.cuh"

// ---------------------------------------------------------------------------
// sim_v3 launch wrappers.
// ---------------------------------------------------------------------------
// All kernels operate on a fixed power-of-two tile (TILE_T) and a unified
// phi pool of N*TILE_AREA floats. There is no neighbour list, no spatial
// hash, no halo, and no per-cell variable W/H. Cell-cell interaction is
// mediated by a global sum field S(x,y) = sum_n phi_n^2(x,y), built each
// step by atomic scatter and read back during evolve.
// ---------------------------------------------------------------------------

// Polarity update (RTP or ABP, per p.abp). Cheap: one thread per cell.
void launch_polar(CellArrays& c, const SimParams& p);

// Apply a list of scripted tumble events: for each i in [0, count),
// theta[d_cid[i]] = d_theta[i]; px = cos(theta); py = sin(theta).
// Used in deterministic-replay mode (--scripted-events).
void launch_apply_scripted(CellArrays& c,
                           const int* d_cid,
                           const float* d_theta,
                           int count);

// Zero S then scatter phi^2 into it (one CTA per cell).
void launch_scatter_S(CellArrays& c, const SimParams& p);

// Fused two-pass evolve. Pass 1 reduces V/Cx/Cy/Ix/Iy, broadcasts vx/vy.
// Pass 2 reads S again, computes laplacian/double-well/repulsion/advection,
// and writes phi_out plus reduces perimeter. Also writes velocities,
// volumes, Cx, Cy, perimeters into the per-cell observable arrays.
void launch_evolve(CellArrays& c, const SimParams& p);

// COM-recentre: shift each cell's tile so its COM lands at (T/2, T/2).
// Adjusts origin[n] and copies the (possibly shifted) tile into phi_out.
// After this kernel, the *caller* must std::swap(phi_in, phi_out) so the
// rebound tile becomes the current state.
void launch_rebind(CellArrays& c, float lambda);

// One-shot host helpers used only at init / resume.
void launch_rng_init(CellArrays& c, unsigned long seed);

// Initialise phi tiles as tanh(2(r - R_eff)/(sqrt(2)*lambda)) profiles.
// h_cx/h_cy are global-coord cell COMs (passed via temporary device arrays
// allocated inside the launcher). Used only at fresh init.
void launch_init_phi(CellArrays& c, const SimParams& p,
                     const float* d_cx, const float* d_cy);

// Compute initial velocities from current phi + per-cell v_A + polarity,
// without advancing phi. Used after both fresh init and resume so that the
// first trajectory write has a meaningful velocity.
void launch_initial_velocity(CellArrays& c, const SimParams& p);
