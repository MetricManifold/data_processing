// sim_v3 kernels — fixed-tile unified-pool architecture.
//
// Layout:
//   1. Device helpers
//   2. k_scatter_S       — atomicAdd phi^2 into global S
//   3. k_evolve_l1       — fused two-pass evolve (reduce → broadcast → write)
//   4. k_rebind          — COM-recentre tile (origin shift + tile copy)
//   5. k_polar           — per-cell polarity update (RTP + ABP)
//   6. k_init_phi        — tanh init profile
//   7. k_initial_velocity — one-shot velocity reduce (init / resume)
//   8. k_rng_init        — curand state seeding
//
// The reference design is in C:\Users\stevensilber\Downloads\testsim\src\sim.cu.
// This file ports those kernels to the production codebase and adds:
//   - per-cell gamma / v_A reads from CellArrays
//   - perimeter accumulation in the evolve reduction
//   - ABP polarity branch in k_polar (reference is RTP-only)
//
// ---------------------------------------------------------------------------
// CELL_SIM_BBOX_TELEMETRY \u2014 OPTIONAL build-time switch.
//
// SHOULD NOT BE ENABLED IN GENERAL. Off by default. Enable only for one-off
// audits of bbox saturation: each rebind atomicMaxes the pre-clamp half-width
// into a device counter. Cost is one global atomicMax per cell per rebind\n// (negligible vs the rebind itself) plus a 4-byte D2H copy at print_interval.
// Host code prints a summary line in print_status() and a lifetime max at\n// end of run. Logs are quiet unless a clamp event occurs.\n//\n// To enable: add -DCELL_SIM_BBOX_TELEMETRY to the compiler command line\n// (e.g. via `cmake -DCMAKE_CUDA_FLAGS=-DCELL_SIM_BBOX_TELEMETRY ...`).\n// ---------------------------------------------------------------------------\n#ifdef CELL_SIM_BBOX_TELEMETRY\n__device__ int g_bbox_max_raw_hw = 0;\n__device__ int g_bbox_clamp_events = 0;\n#endif\n\n// All kernels assume a fixed power-of-two tile (TILE_T) and a unified phi
// pool of N*TILE_AREA floats.  No neighbour list, no halo, no spatial hash.

#include "kernels.cuh"
#include <curand_kernel.h>
#include <cstdio>
#include <vector>

#ifndef PI
#define PI 3.14159265358979323846f
#endif

// ---------------------------------------------------------------------------
// 1. Device helpers
// ---------------------------------------------------------------------------

// Robust periodic wrap for arbitrary int x.
__device__ __forceinline__ int wrap_i(int x, int L) {
    if (x >= 0 && x <  L) return x;
    if (x <  0 && x > -L) return x + L;
    int m = x % L;
    return (m < 0) ? m + L : m;
}

// 9-point isotropic Laplacian (h=1).
__device__ __forceinline__ float lap9(
    float c, float xm, float xp, float ym, float yp,
    float xmym, float xpym, float xmyp, float xpyp)
{
    return (1.0f / 6.0f) * (
        4.0f * (xm + xp + ym + yp)
      + (xmym + xpym + xmyp + xpyp)
      - 20.0f * c
    );
}

// ---------------------------------------------------------------------------
// 2. k_scatter_S — atomicAdd phi^2 into global S
// ---------------------------------------------------------------------------
// Iterates only the active rect (rx0, ry0, rw, rh) inside each cell's
// TILE_T x TILE_T buffer. Pixels outside the rect are zero by
// construction (k_rebind zeroes them), so skipping is exact.
// ---------------------------------------------------------------------------
__global__ void k_scatter_S(
    const float* __restrict__ phi,
    const int*   __restrict__ origin,
    const int*   __restrict__ rect,
    float* __restrict__ S,
    int N, int L)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const float* tile = phi + (size_t)n * TILE_AREA;
    const int gx0 = origin[2*n + 0];
    const int gy0 = origin[2*n + 1];
    const int rx0 = rect[4*n + 0];
    const int ry0 = rect[4*n + 1];
    const int rw  = rect[4*n + 2];
    const int rh  = rect[4*n + 3];
    const int total = rw * rh;
    const int BS  = blockDim.x;
    const int tid = threadIdx.x;

    int lx = rx0 + (tid % rw);
    int ly = ry0 + (tid / rw);
    for (int p = tid; p < total; p += BS) {
        float v = tile[ly * TILE_T + lx];
        if (v >= 1e-6f) {
            int gx = wrap_i(gx0 + lx, L);
            int gy = wrap_i(gy0 + ly, L);
            atomicAdd(&S[gy * L + gx], v * v);
        }
        lx += BS;
        while (lx >= rx0 + rw) { lx -= rw; ly += 1; }
    }
}

void launch_scatter_S(CellArrays& c, const SimParams& p) {
    const int N = c.num_cells;
    if (N == 0) return;
    const size_t Sbytes = (size_t)p.Nx * p.Ny * sizeof(float);
    cudaMemsetAsync(c.S, 0, Sbytes);
    k_scatter_S<<<N, 256>>>(c.phi_in, c.origin, c.rect, c.S, N, p.Nx);
}

// ---------------------------------------------------------------------------
// 3. k_evolve_l1 — fused two-pass evolve.
// ---------------------------------------------------------------------------
// One CTA per cell, BS=256 threads. phi is read from L1 (no shared-mem
// caching — the 64KB shared variant tested in the reference forced 1 block
// per SM and was slower). The compiler keeps a hot working set in registers
// and the L1/texture cache picks up the spatial reuse from neighbouring
// thread iterations.
//
// Pass 1 accumulates V, Ix, Iy (interaction integrals) and Cx, Cy
// (tile-local centroid moments) per cell. A 5-channel block reduction
// produces those scalars in shared memory; thread 0 broadcasts (Vn, vx, vy)
// for use in Pass 2 and writes the per-cell observables.
//
// Pass 2 re-reads phi (L1-hot) and S (L2-hot), computes the full PDE RHS
// (Laplacian + double-well + volume constraint + repulsion), advects with
// the freshly-broadcast velocity, and writes phi_out. Perimeter
// (sum |grad phi|) is accumulated in the same pass and reduced separately.
//
// Per-cell scalars read inline:
//   gamma_cell[n], v_A_cell[n]   — vary per cell (gamma_spec / v_A_sigma)
//   tgt_radius[n]                — used for piR2 / volume constraint
// All other physics constants are passed as kernel arguments and broadcast
// from constant-bandwidth registers.
// ---------------------------------------------------------------------------
__global__ void k_evolve_l1(
    const float* __restrict__ phi,
    const int*   __restrict__ origin,
    const int*   __restrict__ rect,
    const float* __restrict__ S,
    const float* __restrict__ gamma_cell,
    const float* __restrict__ v_A_cell,
    const float* __restrict__ tgt_radius,
    const float* __restrict__ dirx,
    const float* __restrict__ diry,
    float* __restrict__ V_out,
    float* __restrict__ Cx_out,
    float* __restrict__ Cy_out,
    float* __restrict__ Cxx_out,
    float* __restrict__ Cyy_out,
    float* __restrict__ peri_out,
    float* __restrict__ vx_out,
    float* __restrict__ vy_out,
    float* __restrict__ phi_out,
    int N, int L,
    float lambda_, float kappa, float mu,
    float xi, float dt)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int BS  = blockDim.x;          // expect 256
    const int tid = threadIdx.x;

    const float* tile = phi     + (size_t)n * TILE_AREA;
    float*       outp = phi_out + (size_t)n * TILE_AREA;
    const int gx0 = origin[2*n + 0];
    const int gy0 = origin[2*n + 1];
    const int rx0 = rect[4*n + 0];
    const int ry0 = rect[4*n + 1];
    const int rw  = rect[4*n + 2];
    const int rh  = rect[4*n + 3];
    const int rect_total = rw * rh;
    const float gam = gamma_cell[n];
    const float vA  = v_A_cell[n];
    const float R   = tgt_radius[n];

    // ----- Pass 1: V, Ix, Iy, Cx, Cy, Cxx, Cyy -----
    int lx0 = rx0 + (tid % rw);
    int ly  = ry0 + (tid / rw);
    int lx  = lx0;
    float sV = 0.0f, sIx = 0.0f, sIy = 0.0f;
    float sCx = 0.0f, sCy = 0.0f, sCxx = 0.0f, sCyy = 0.0f;
    for (int p = tid; p < rect_total; p += BS) {
        int idx = ly * TILE_T + lx;
        float c   = __ldg(tile + idx);
        float xp_ = (lx + 1 < TILE_T) ? __ldg(tile + idx + 1)      : 0.0f;
        float xm_ = (lx     > 0)      ? __ldg(tile + idx - 1)      : 0.0f;
        float yp_ = (ly + 1 < TILE_T) ? __ldg(tile + idx + TILE_T) : 0.0f;
        float ym_ = (ly     > 0)      ? __ldg(tile + idx - TILE_T) : 0.0f;
        float gx = 0.5f * (xp_ - xm_);
        float gy = 0.5f * (yp_ - ym_);
        int gxg = wrap_i(gx0 + lx, L);
        int gyg = wrap_i(gy0 + ly, L);
        float Sv   = S[gyg * L + gxg];
        float Soth = Sv - c * c;
        if (Soth < 0.0f) Soth = 0.0f;
        float c2 = c * c;
        float fx = (float)lx, fy = (float)ly;
        sV   += c2;
        sIx  += c * gx * Soth;
        sIy  += c * gy * Soth;
        sCx  += c2 * fx;
        sCy  += c2 * fy;
        sCxx += c2 * fx * fx;
        sCyy += c2 * fy * fy;
        lx += BS;
        while (lx >= rx0 + rw) { lx -= rw; ly += 1; }
    }

    // 7-channel block reduction (7 * 256 floats = 7 KB shared).
    __shared__ float sred[256 * 7];
    __shared__ float sbroad[7];
    sred[tid          ] = sV;
    sred[tid +   256  ] = sIx;
    sred[tid + 2*256  ] = sIy;
    sred[tid + 3*256  ] = sCx;
    sred[tid + 4*256  ] = sCy;
    sred[tid + 5*256  ] = sCxx;
    sred[tid + 6*256  ] = sCyy;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sred[tid          ] += sred[tid + s          ];
            sred[tid +   256  ] += sred[tid + s +   256  ];
            sred[tid + 2*256  ] += sred[tid + s + 2*256  ];
            sred[tid + 3*256  ] += sred[tid + s + 3*256  ];
            sred[tid + 4*256  ] += sred[tid + s + 4*256  ];
            sred[tid + 5*256  ] += sred[tid + s + 5*256  ];
            sred[tid + 6*256  ] += sred[tid + s + 6*256  ];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sbroad[0] = sred[0];
        sbroad[1] = sred[256];
        sbroad[2] = sred[2*256];
        sbroad[3] = sred[3*256];
        sbroad[4] = sred[4*256];
        sbroad[5] = sred[5*256];
        sbroad[6] = sred[6*256];
    }
    __syncthreads();
    const float Vn  = sbroad[0];
    const float Ixn = sbroad[1];
    const float Iyn = sbroad[2];

    const float invXi  = 1.0f / xi;
    const float coeffV = 60.0f * kappa * invXi / (lambda_ * lambda_);
    const float vx     = coeffV * Ixn + vA * dirx[n];
    const float vy     = coeffV * Iyn + vA * diry[n];

    if (tid == 0) {
        V_out[n]   = Vn;
        Cx_out[n]  = sbroad[3];
        Cy_out[n]  = sbroad[4];
        Cxx_out[n] = sbroad[5];
        Cyy_out[n] = sbroad[6];
        vx_out[n]  = vx;
        vy_out[n]  = vy;
    }

    const float piR2 = PI * R * R;
    const float volC = (2.0f * mu / piR2) * (piR2 - Vn);
    const float dwC  = 30.0f * gam   / (lambda_ * lambda_);
    const float repC = 30.0f * kappa / (lambda_ * lambda_);

    // ----- Pass 2: PDE update + perimeter reduction -----
    // Only writes pixels inside the rect. Pixels outside the rect of
    // phi_out remain zero: alloc_gpu zeroes the pool initially, and
    // k_rebind zeroes outside-new-rect on every rebind step. Between
    // rebinds the rect doesn't grow, so the outside pixels of phi_out
    // (which become phi_in two steps from now) stay clean.

    lx = lx0; ly = ry0 + (tid / rw);
    float sPeri = 0.0f;
    for (int p = tid; p < rect_total; p += BS) {
        int idx = ly * TILE_T + lx;
        float c    = __ldg(tile + idx);
        float xp_  = (lx + 1 < TILE_T)                     ? __ldg(tile + idx + 1)          : 0.0f;
        float xm_  = (lx     > 0)                          ? __ldg(tile + idx - 1)          : 0.0f;
        float yp_  = (ly + 1 < TILE_T)                     ? __ldg(tile + idx + TILE_T)     : 0.0f;
        float ym_  = (ly     > 0)                          ? __ldg(tile + idx - TILE_T)     : 0.0f;
        float xpyp = (lx + 1 < TILE_T && ly + 1 < TILE_T)  ? __ldg(tile + idx + TILE_T + 1) : 0.0f;
        float xpym = (lx + 1 < TILE_T && ly     > 0)       ? __ldg(tile + idx - TILE_T + 1) : 0.0f;
        float xmyp = (lx     > 0      && ly + 1 < TILE_T)  ? __ldg(tile + idx + TILE_T - 1) : 0.0f;
        float xmym = (lx     > 0      && ly     > 0)       ? __ldg(tile + idx - TILE_T - 1) : 0.0f;

        float lap = lap9(c, xm_, xp_, ym_, yp_, xmym, xpym, xmyp, xpyp);
        float gx  = 0.5f * (xp_ - xm_);
        float gy  = 0.5f * (yp_ - ym_);

        int gxg = wrap_i(gx0 + lx, L);
        int gyg = wrap_i(gy0 + ly, L);
        float Sv   = S[gyg * L + gxg];
        float Soth = Sv - c * c;
        if (Soth < 0.0f) Soth = 0.0f;

        float dw  = c * (1.0f - c) * (1.0f - 2.0f * c);
        float rhs = gam * lap - dwC * dw + volC * c - repC * c * Soth;
        float adv = vx * gx + vy * gy;
        outp[idx] = c + dt * (rhs - adv);

        sPeri += sqrtf(gx * gx + gy * gy);
        lx += BS;
        while (lx >= rx0 + rw) { lx -= rw; ly += 1; }
    }

    // 1-channel perimeter reduction. Reuse sred[0..256].
    __syncthreads();
    sred[tid] = sPeri;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sred[tid] += sred[tid + s];
        __syncthreads();
    }
    if (tid == 0) peri_out[n] = sred[0];
}

void launch_evolve(CellArrays& c, const SimParams& p) {
    const int N = c.num_cells;
    if (N == 0) return;
    k_evolve_l1<<<N, 256>>>(
        c.phi_in, c.origin, c.rect, c.S,
        c.gamma_cell, c.v_A_cell, c.tgt_radius,
        c.polar_x, c.polar_y,
        c.volumes, c.Cx, c.Cy, c.Cxx, c.Cyy, c.perimeters,
        c.velocities_x, c.velocities_y,
        c.phi_out,
        N, p.Nx,
        (float)p.lambda, (float)p.kappa, (float)p.mu,
        (float)p.xi,     (float)p.dt);
}

// ---------------------------------------------------------------------------
// 4. k_rebind — COM-recentre tile + adapt rect from second moments.
// ---------------------------------------------------------------------------
// Reads V, Cx, Cy, Cxx, Cyy (tile-local) computed by the most recent
// evolve pass, then:
//   1. Compute COM = (Cx/V, Cy/V) and integer shift to land at (T/2, T/2).
//   2. Compute sigma_x = sqrt(Cxx/V - mx^2), likewise sigma_y, and pick
//      a new rect half-width hw = ceil(k*sigma + margin), aligned to
//      bbox_align, clamped to [bbox_min, T/2 - 1].
//   3. Copy phi_in -> phi_out with the shift; pixels outside the new rect
//      are zeroed; out-of-source destinations are zeroed.
//   4. Update origin += shift, write new rect.
//
// The caller std::swap(phi_in, phi_out) so the rebound tile becomes the
// current state.
// ---------------------------------------------------------------------------
__global__ void k_rebind(
    float* __restrict__ phi_in,
    float* __restrict__ phi_out,
    int* __restrict__ origin,
    int* __restrict__ rect,
    const float* __restrict__ V,
    const float* __restrict__ Cx,
    const float* __restrict__ Cy,
    const float* __restrict__ Cxx,
    const float* __restrict__ Cyy,
    const float* __restrict__ tgt_radius,
    const float* __restrict__ gamma_cell,
    int N,
    float bbox_k, float gamma_ref, int bbox_align, int bbox_min)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int BS  = blockDim.x;
    const int tid = threadIdx.x;
    const int Th  = TILE_T >> 1;

    float* tin  = phi_in  + (size_t)n * TILE_AREA;
    float* tout = phi_out + (size_t)n * TILE_AREA;

    __shared__ int sshift[2];
    __shared__ int srect[4];
    if (tid == 0) {
        float Vn = V[n];
        float invV = (Vn > 1e-6f) ? 1.0f / Vn : 0.0f;
        float mx = Cx[n] * invV;
        float my = Cy[n] * invV;
        int sx = __float2int_rn(mx) - Th;
        int sy = __float2int_rn(my) - Th;
        sshift[0] = sx;
        sshift[1] = sy;
        origin[2*n + 0] += sx;
        origin[2*n + 1] += sy;

        float varx = Cxx[n] * invV - mx * mx;
        float vary = Cyy[n] * invV - my * my;
        if (varx < 0.0f) varx = 0.0f;
        if (vary < 0.0f) vary = 0.0f;
        float sigx = sqrtf(varx);
        float sigy = sqrtf(vary);
        // Per-axis half-widths: hw = ceil(K_eff*sigma + R/2). Margin scales
        // with cell radius so the rect tracks shape change between rebinds.
        // K (subdomain_padding) controls how aggressively the rect tracks
        // the second moments; the only hard floor is bbox_min and the only
        // ceiling is Th-1 (preserves stencil halo).
        //
        // Per-cell K scaling: softer cells (lower gamma) deform more between
        // rebinds and squeeze through gaps with high transient aspect ratio,
        // so they need extra padding. K_eff = K * sqrt(max(1, gamma_ref/gamma)).
        // At gamma == gamma_ref this is a no-op; at gamma = gamma_ref/4
        // (e.g. 0.25 vs 1.0) it doubles K. Stiffer-than-reference cells get
        // the unscaled K -- they don't need the extra room.
        float R = tgt_radius[n];
        float gn = gamma_cell[n];
        float k_scale = 1.0f;
        if (gn > 0.0f && gn < gamma_ref) {
            k_scale = sqrtf(gamma_ref / gn);
        }
        float k_eff = bbox_k * k_scale;
        float margin = 0.5f * R;
        int hwx = (int)ceilf(k_eff * sigx + margin);
        int hwy = (int)ceilf(k_eff * sigy + margin);
        hwx = ((hwx + bbox_align - 1) / bbox_align) * bbox_align;
        hwy = ((hwy + bbox_align - 1) / bbox_align) * bbox_align;
        const int hmax = Th - 1;     // keep 1px stencil halo
#ifdef CELL_SIM_BBOX_TELEMETRY
        // Record peak unclamped hw and count clamp events. NOT enabled by
        // default \u2014 see comment in kernels.cu file header.
        int hwmax_pre = hwx > hwy ? hwx : hwy;
        atomicMax(&g_bbox_max_raw_hw, hwmax_pre);
        if (hwx > hmax || hwy > hmax) atomicAdd(&g_bbox_clamp_events, 1);
#endif
        if (hwx > hmax) hwx = hmax;
        if (hwy > hmax) hwy = hmax;
        if (hwx < bbox_min) hwx = bbox_min;
        if (hwy < bbox_min) hwy = bbox_min;
        srect[0] = Th - hwx;
        srect[1] = Th - hwy;
        srect[2] = 2 * hwx;
        srect[3] = 2 * hwy;
        rect[4*n + 0] = srect[0];
        rect[4*n + 1] = srect[1];
        rect[4*n + 2] = srect[2];
        rect[4*n + 3] = srect[3];
    }
    __syncthreads();
    const int sx = sshift[0];
    const int sy = sshift[1];
    const int rx0 = srect[0], ry0 = srect[1];
    const int rx1 = rx0 + srect[2], ry1 = ry0 + srect[3];

    // Phase 1: write shifted source data into phi_out (destination).
    // Pixels in new rect get shifted source; outside = 0.
    for (int p = tid; p < TILE_AREA; p += BS) {
        int ox = p % TILE_T;
        int oy = p / TILE_T;
        float v = 0.0f;
        if (ox >= rx0 && ox < rx1 && oy >= ry0 && oy < ry1) {
            int sxi = ox + sx;
            int syi = oy + sy;
            if (sxi >= 0 && sxi < TILE_T && syi >= 0 && syi < TILE_T) {
                v = tin[syi * TILE_T + sxi];
            }
        }
        tout[p] = v;
    }
    __syncthreads();

    // Phase 2: scrub source-buffer halo. After the caller swaps, this
    // buffer becomes phi_out for the next evolve. Evolve only writes
    // INSIDE the new rect, so any pixels outside-new-rect carry stale
    // cell data from before this rebind (when the rect shrinks). On the
    // following swap those stale pixels become phi_in and the boundary
    // stencil reads them. Zeroing now is the cheapest fix; no race
    // because Phase 1 above already finished reading from tin.
    for (int p = tid; p < TILE_AREA; p += BS) {
        int ox = p % TILE_T;
        int oy = p / TILE_T;
        bool inside = (ox >= rx0 && ox < rx1 && oy >= ry0 && oy < ry1);
        if (!inside) tin[p] = 0.0f;
    }
}

void launch_rebind(CellArrays& c, float bbox_k, float gamma_ref) {
    const int N = c.num_cells;
    if (N == 0) return;
    k_rebind<<<N, 256>>>(c.phi_in, c.phi_out, c.origin, c.rect,
                         c.volumes, c.Cx, c.Cy, c.Cxx, c.Cyy,
                         c.tgt_radius, c.gamma_cell, N,
                         bbox_k, gamma_ref, TILE_BBOX_ALIGN, TILE_BBOX_MIN);
}

// ---------------------------------------------------------------------------
// 5. k_polar — per-cell polarity update.
// ---------------------------------------------------------------------------
// theta is the persistent polar angle; (px, py) = (cos, sin) are the
// derived unit vector. ABP: theta diffuses each step. RTP: with prob
// (1 - exp(-dt/tau)) re-randomise theta uniformly.
//
// One thread per cell. Skipped entirely when v_A == 0 (no motility) or
// when tau <= 0 (sentinel: hold polarity fixed forever).
// ---------------------------------------------------------------------------
__global__ void k_polar(
    curandState* __restrict__ st,
    float* __restrict__ theta_arr,
    float* __restrict__ px, float* __restrict__ py,
    float dt, float tau, bool abp, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curandState s = st[i];
    float theta = theta_arr[i];
    bool changed = false;
    if (abp) {
        theta += sqrtf(2.0f * dt / tau) * curand_normal(&s);
        changed = true;
    } else {
        if (curand_uniform(&s) < 1.0f - expf(-dt / tau)) {
            theta = curand_uniform(&s) * 2.0f * PI;
            changed = true;
        }
    }
    if (changed) {
        theta_arr[i] = theta;
        px[i] = cosf(theta);
        py[i] = sinf(theta);
    }
    st[i] = s;
}

void launch_polar(CellArrays& c, const SimParams& p) {
    const int N = c.num_cells;
    if (N == 0 || p.v_A == 0.0 || p.tau <= 0.0) return;
    k_polar<<<(N + 255) / 256, 256>>>(
        (curandState*)c.rng_states,
        c.polar_theta, c.polar_x, c.polar_y,
        (float)p.dt, (float)p.tau, p.abp, N);
}

// ---------------------------------------------------------------------------
// 5b. k_apply_scripted — write pre-determined (cid, theta) pairs.
// ---------------------------------------------------------------------------
// Used by --scripted-events deterministic replay. One thread per event.
// ---------------------------------------------------------------------------
__global__ void k_apply_scripted(
    float* __restrict__ theta_arr,
    float* __restrict__ px, float* __restrict__ py,
    const int* __restrict__ cids,
    const float* __restrict__ thetas,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int   c = cids[i];
    float t = thetas[i];
    theta_arr[c] = t;
    px[c] = cosf(t);
    py[c] = sinf(t);
}

void launch_apply_scripted(CellArrays& c,
                           const int* d_cid,
                           const float* d_theta,
                           int count)
{
    if (count <= 0) return;
    k_apply_scripted<<<(count + 31) / 32, 32>>>(
        c.polar_theta, c.polar_x, c.polar_y,
        d_cid, d_theta, count);
}

// ---------------------------------------------------------------------------
// 6. k_init_phi — tanh init profile, fresh start only.
// ---------------------------------------------------------------------------
// phi(r) = 0.5 * (1 - tanh(2*(r - R_eff) / iw))
// with R_eff = R + 0.7088*lambda - 0.5887*lambda^2/R and iw = sqrt(2)*lambda.
// (These corrections compensate the f32 tanh interface so that the
// converged volume matches pi*R^2 to ~1%.)
// ---------------------------------------------------------------------------
__global__ void k_init_phi(
    float* phi, const int* origin,
    const float* cx, const float* cy,
    int N, int L, float R, float lambda_)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    float* tile = phi + (size_t)n * TILE_AREA;
    const int gx0 = origin[2*n + 0];
    const int gy0 = origin[2*n + 1];
    const float ccx = cx[n], ccy = cy[n];

    const float R_eff = R + 0.7088f * lambda_ - 0.5887f * lambda_ * lambda_ / R;
    const float iw    = 1.4142135f * lambda_;

    for (int p = threadIdx.x; p < TILE_AREA; p += blockDim.x) {
        int lx = p % TILE_T;
        int ly = p / TILE_T;
        float xg = (float)(gx0 + lx);
        float yg = (float)(gy0 + ly);
        float dx = xg - ccx;
        float dy = yg - ccy;
        if (dx >  0.5f * L) dx -= L;
        if (dx < -0.5f * L) dx += L;
        if (dy >  0.5f * L) dy -= L;
        if (dy < -0.5f * L) dy += L;
        float r = sqrtf(dx*dx + dy*dy);
        tile[p] = 0.5f * (1.0f - tanhf(2.0f * (r - R_eff) / iw));
    }
}

void launch_init_phi(CellArrays& c, const SimParams& p,
                     const float* d_cx, const float* d_cy) {
    const int N = c.num_cells;
    if (N == 0) return;
    k_init_phi<<<N, 256>>>(c.phi_in, c.origin, d_cx, d_cy,
                            N, p.Nx,
                            (float)p.target_radius, (float)p.lambda);
}

// ---------------------------------------------------------------------------
// 7. k_initial_velocity — one-shot velocity reduction.
// ---------------------------------------------------------------------------
// Computes the same V/Cx/Cy/Ix/Iy reduction as k_evolve_l1's Pass 1, plus
// perimeter, but does NOT advance phi. Used after init / resume so that the
// first trajectory write contains a meaningful velocity (interaction +
// active component), and so that downstream code doesn't see zero volumes.
//
// Implemented by reusing k_evolve_l1 with a tagged dt=0 fast path is
// possible, but it's cleaner to have a dedicated kernel that skips Pass 2.
// ---------------------------------------------------------------------------
__global__ void k_initial_velocity(
    const float* __restrict__ phi,
    const int*   __restrict__ origin,
    const int*   __restrict__ rect,
    const float* __restrict__ S,
    const float* __restrict__ v_A_cell,
    const float* __restrict__ dirx,
    const float* __restrict__ diry,
    float* __restrict__ V_out,
    float* __restrict__ Cx_out,
    float* __restrict__ Cy_out,
    float* __restrict__ Cxx_out,
    float* __restrict__ Cyy_out,
    float* __restrict__ peri_out,
    float* __restrict__ vx_out,
    float* __restrict__ vy_out,
    int N, int L,
    float lambda_, float kappa, float xi)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int BS  = blockDim.x;
    const int tid = threadIdx.x;

    const float* tile = phi + (size_t)n * TILE_AREA;
    const int gx0 = origin[2*n + 0];
    const int gy0 = origin[2*n + 1];
    const int rx0 = rect[4*n + 0];
    const int ry0 = rect[4*n + 1];
    const int rw  = rect[4*n + 2];
    const int rh  = rect[4*n + 3];
    const int rect_total = rw * rh;
    const float vA = v_A_cell[n];

    int lx = rx0 + (tid % rw);
    int ly = ry0 + (tid / rw);
    float sV = 0.0f, sIx = 0.0f, sIy = 0.0f;
    float sCx = 0.0f, sCy = 0.0f, sCxx = 0.0f, sCyy = 0.0f, sP = 0.0f;
    for (int p = tid; p < rect_total; p += BS) {
        int idx = ly * TILE_T + lx;
        float c   = __ldg(tile + idx);
        float xp_ = (lx + 1 < TILE_T) ? __ldg(tile + idx + 1)      : 0.0f;
        float xm_ = (lx     > 0)      ? __ldg(tile + idx - 1)      : 0.0f;
        float yp_ = (ly + 1 < TILE_T) ? __ldg(tile + idx + TILE_T) : 0.0f;
        float ym_ = (ly     > 0)      ? __ldg(tile + idx - TILE_T) : 0.0f;
        float gx = 0.5f * (xp_ - xm_);
        float gy = 0.5f * (yp_ - ym_);
        int gxg = wrap_i(gx0 + lx, L);
        int gyg = wrap_i(gy0 + ly, L);
        float Sv   = S[gyg * L + gxg];
        float Soth = Sv - c * c;
        if (Soth < 0.0f) Soth = 0.0f;
        float c2 = c * c;
        float fx = (float)lx, fy = (float)ly;
        sV   += c2;
        sIx  += c * gx * Soth;
        sIy  += c * gy * Soth;
        sCx  += c2 * fx;
        sCy  += c2 * fy;
        sCxx += c2 * fx * fx;
        sCyy += c2 * fy * fy;
        sP   += sqrtf(gx * gx + gy * gy);
        lx += BS;
        while (lx >= rx0 + rw) { lx -= rw; ly += 1; }
    }

    __shared__ float sred[256 * 8];
    sred[tid          ] = sV;
    sred[tid +   256  ] = sIx;
    sred[tid + 2*256  ] = sIy;
    sred[tid + 3*256  ] = sCx;
    sred[tid + 4*256  ] = sCy;
    sred[tid + 5*256  ] = sCxx;
    sred[tid + 6*256  ] = sCyy;
    sred[tid + 7*256  ] = sP;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sred[tid          ] += sred[tid + s          ];
            sred[tid +   256  ] += sred[tid + s +   256  ];
            sred[tid + 2*256  ] += sred[tid + s + 2*256  ];
            sred[tid + 3*256  ] += sred[tid + s + 3*256  ];
            sred[tid + 4*256  ] += sred[tid + s + 4*256  ];
            sred[tid + 5*256  ] += sred[tid + s + 5*256  ];
            sred[tid + 6*256  ] += sred[tid + s + 6*256  ];
            sred[tid + 7*256  ] += sred[tid + s + 7*256  ];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float Vn  = sred[0];
        float Ixn = sred[256];
        float Iyn = sred[2*256];
        float coeffV = 60.0f * kappa / (xi * lambda_ * lambda_);
        V_out[n]    = Vn;
        Cx_out[n]   = sred[3*256];
        Cy_out[n]   = sred[4*256];
        Cxx_out[n]  = sred[5*256];
        Cyy_out[n]  = sred[6*256];
        peri_out[n] = sred[7*256];
        vx_out[n]   = coeffV * Ixn + vA * dirx[n];
        vy_out[n]   = coeffV * Iyn + vA * diry[n];
    }
}

void launch_initial_velocity(CellArrays& c, const SimParams& p) {
    const int N = c.num_cells;
    if (N == 0) return;
    const size_t Sbytes = (size_t)p.Nx * p.Ny * sizeof(float);
    cudaMemsetAsync(c.S, 0, Sbytes);
    k_scatter_S<<<N, 256>>>(c.phi_in, c.origin, c.rect, c.S, N, p.Nx);
    k_initial_velocity<<<N, 256>>>(
        c.phi_in, c.origin, c.rect, c.S,
        c.v_A_cell, c.polar_x, c.polar_y,
        c.volumes, c.Cx, c.Cy, c.Cxx, c.Cyy, c.perimeters,
        c.velocities_x, c.velocities_y,
        N, p.Nx,
        (float)p.lambda, (float)p.kappa, (float)p.xi);
}

// ---------------------------------------------------------------------------
// 8. k_rng_init
// ---------------------------------------------------------------------------
__global__ void k_rng_init(curandState* st, unsigned long seed, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curand_init(seed, i, 0, &st[i]);
}

void launch_rng_init(CellArrays& c, unsigned long seed) {
    const int N = c.num_cells;
    if (N == 0) return;
    k_rng_init<<<(N + 255) / 256, 256>>>(
        (curandState*)c.rng_states, seed, N);
}
