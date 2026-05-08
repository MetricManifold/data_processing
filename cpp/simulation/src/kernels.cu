// CUDA kernels — fixed-tile unified-pool architecture.
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

// Warp-level sum reduce (single value). 32-thread warp.
__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

// Block-level sum reduce. Uses warp-shuffle within each warp + a single
// shared-memory pass across warp leaders. Block size must be a multiple of
// 32 and <= 1024 (so warp count fits in a single warp). After this returns,
// thread 0 holds the block-wide sum; other threads' return value is
// undefined. Caller should broadcast via shared memory if needed.
//
//   smem must point at >= 32 floats (one per warp leader).
__device__ __forceinline__ float block_sum(float v, float* smem) {
    int lane    = threadIdx.x & 31;
    int warpId  = threadIdx.x >> 5;
    int nWarps  = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    if (lane == 0) smem[warpId] = v;
    __syncthreads();
    if (warpId == 0) {
        float s = (threadIdx.x < nWarps) ? smem[threadIdx.x] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) smem[0] = s;
    }
    __syncthreads();
    return smem[0];
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
    int N, int L, int CHUNK_PIXELS,
    int y_lo, int halo_h)
{
    const int n  = blockIdx.y;
    const int cb = blockIdx.x;       // chunk index within this cell
    if (n >= N) return;
    const float* tile = phi + (size_t)n * TILE_AREA;
    const int gx0 = origin[2*n + 0];
    const int gy0 = origin[2*n + 1];
    const int rx0 = rect[4*n + 0];
    const int ry0 = rect[4*n + 1];
    const int rw  = rect[4*n + 2];
    const int rh  = rect[4*n + 3];
    const int total = rw * rh;
    const int chunk_start = cb * CHUNK_PIXELS;
    if (chunk_start >= total) return;
    const int chunk_end = min(total, chunk_start + CHUNK_PIXELS);
    const int BS  = blockDim.x;
    const int tid = threadIdx.x;

    const int step_x = BS % rw;
    const int step_y = BS / rw;
    const int rx_end = rx0 + rw;
    int p0 = chunk_start + tid;
    int lx = rx0 + (p0 % rw);
    int ly = ry0 + (p0 / rw);
    for (int p = p0; p < chunk_end; p += BS) {
        float v = __ldg(tile + ly * TILE_T + lx);
        // Every pixel inside the rect contributes to S (no background-skip):
        // skipping changes the global S field by ~rect_area * 1e-12 which is
        // below f32 epsilon but not bit-exact.
        int gx = wrap_i(gx0 + lx, L);
        int gy = wrap_i(gy0 + ly, L);
        int sy = slab_local_y(gy, y_lo, halo_h, L, L);
        atomicAdd(&S[sy * L + gx], v * v);
        lx += step_x; ly += step_y;
        if (lx >= rx_end) { lx -= rw; ly += 1; }
    }
}

void launch_scatter_S(CellArrays& c, const SimParams& p, cudaStream_t stream) {
    const int N = c.num_cells;
    if (N == 0) return;
    // S is sized by the slab's extended height (== Ny for G=1).
    const size_t Sbytes = (size_t)c.S_ext_height * p.Nx * sizeof(float);
    cudaMemsetAsync(c.S, 0, Sbytes, stream);
    // Always multi-block: chunking by ~4096 pixels keeps the SMs saturated
    // even at large N (1152 cells * 9 chunks/cell = 10368 blocks vs the
    // 1152-block "fused" alternative which left ~85% of warps idle on
    // a 76-SM device).
    constexpr int CHUNK_PIXELS = 4096;
    constexpr int BS = 256;
    constexpr int chunks_per_cell = (TILE_AREA + CHUNK_PIXELS - 1) / CHUNK_PIXELS;
    k_scatter_S<<<dim3(chunks_per_cell, N), BS, 0, stream>>>(
        c.phi_in, c.origin, c.rect, c.S, N, p.Nx, CHUNK_PIXELS,
        c.S_y_lo, c.S_halo_h);
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
    float xi, float dt,
    int y_lo, int halo_h)
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
        int sy  = slab_local_y(gyg, y_lo, halo_h, L, L);
        float Sv   = S[sy * L + gxg];
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

    // 7-channel block reduction via warp-shuffle.
    // Shared mem: one warp-leader slot per warp per channel (=> 7*32 floats max),
    // plus one float per channel for broadcast = 7 floats. Total 7*32 + 7 = 231
    // floats (~924 B). Independent of BS (so BS can change without reworking
    // the shmem footprint).
    __shared__ float ssmem[7 * 32];
    __shared__ float sbroad[7];
    {
        float v0 = block_sum(sV,   ssmem + 0*32);
        float v1 = block_sum(sIx,  ssmem + 1*32);
        float v2 = block_sum(sIy,  ssmem + 2*32);
        float v3 = block_sum(sCx,  ssmem + 3*32);
        float v4 = block_sum(sCy,  ssmem + 4*32);
        float v5 = block_sum(sCxx, ssmem + 5*32);
        float v6 = block_sum(sCyy, ssmem + 6*32);
        if (tid == 0) {
            sbroad[0] = v0; sbroad[1] = v1; sbroad[2] = v2;
            sbroad[3] = v3; sbroad[4] = v4; sbroad[5] = v5;
            sbroad[6] = v6;
        }
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
        int sy  = slab_local_y(gyg, y_lo, halo_h, L, L);
        float Sv   = S[sy * L + gxg];
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

    // 1-channel perimeter reduction via warp-shuffle. Reuse ssmem (first 32).
    __syncthreads();
    float pn = block_sum(sPeri, ssmem);
    if (tid == 0) peri_out[n] = pn;
}

// ---------------------------------------------------------------------------
// 3b. Multi-block scatter/reduce/RHS pipeline.
// ---------------------------------------------------------------------------
// Grid shape (chunks_per_cell, N). Each block processes CHUNK_PIXELS pixels
// of the active rect for one cell. Splitting the cell across many blocks
// pumps occupancy when N is small. Per-cell accumulators are filled by
// atomicAdd from one thread per block; the host pre-zeros them.
// All pixels inside the active rect are evaluated unconditionally; we do
// NOT background-skip on phi < threshold. The rect itself is the locality
// optimisation; further skipping would change reductions / RHS by amounts
// below f32 epsilon but not bit-exact, which is unacceptable here.
// ---------------------------------------------------------------------------

__global__ void k_zero_per_cell(
    float* a, float* b, float* c, float* d,
    float* e, float* f, float* g, float* h,
    int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    a[n] = 0.0f; b[n] = 0.0f; c[n] = 0.0f; d[n] = 0.0f;
    e[n] = 0.0f; f[n] = 0.0f; g[n] = 0.0f; h[n] = 0.0f;
}

__global__ void k_zero_per_cell3(
    float* a, float* b, float* c, int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    a[n] = 0.0f; b[n] = 0.0f; c[n] = 0.0f;
}

// Fast reduce: only V, Ix, Iy. Used on non-rebind, non-output steps.
// Saves 5 atomicAdds per chunk and 5 block-reductions vs the full variant.
__global__ void k_reduce_mb_fast(
    const float* __restrict__ phi,
    const int*   __restrict__ origin,
    const int*   __restrict__ rect,
    const float* __restrict__ S,
    float* __restrict__ V_out,
    float* __restrict__ Ix_out,
    float* __restrict__ Iy_out,
    int N, int L, int CHUNK_PIXELS,
    int y_lo, int halo_h)
{
    const int n  = blockIdx.y;
    const int cb = blockIdx.x;
    if (n >= N) return;
    const int rx0 = rect[4*n + 0];
    const int ry0 = rect[4*n + 1];
    const int rw  = rect[4*n + 2];
    const int rh  = rect[4*n + 3];
    const int rect_total = rw * rh;
    const int p_start = cb * CHUNK_PIXELS;
    if (p_start >= rect_total) return;
    const int p_end = min(p_start + CHUNK_PIXELS, rect_total);
    const int gx0 = origin[2*n + 0];
    const int gy0 = origin[2*n + 1];
    const float* tile = phi + (size_t)n * TILE_AREA;
    const int tid = threadIdx.x;
    const int BS  = blockDim.x;

    float sV = 0.0f, sIx = 0.0f, sIy = 0.0f;

    const int step_x = BS % rw;
    const int step_y = BS / rw;
    const int rx_end = rx0 + rw;
    int p0 = p_start + tid;
    int lx = rx0 + (p0 % rw);
    int ly = ry0 + (p0 / rw);
    for (int p = p0; p < p_end; p += BS) {
        int idx = ly * TILE_T + lx;
        // rect invariant: rx0 >= 1 && rx0+rw <= TILE_T-1, same for y.
        // All 5-pt stencil neighbors are in-tile, no boundary guards.
        float c   = __ldg(tile + idx);
        float xp_ = __ldg(tile + idx + 1);
        float xm_ = __ldg(tile + idx - 1);
        float yp_ = __ldg(tile + idx + TILE_T);
        float ym_ = __ldg(tile + idx - TILE_T);
        float gx = 0.5f * (xp_ - xm_);
        float gy = 0.5f * (yp_ - ym_);
        int gxg = wrap_i(gx0 + lx, L);
        int gyg = wrap_i(gy0 + ly, L);
        int sy  = slab_local_y(gyg, y_lo, halo_h, L, L);
        float Sv   = __ldg(S + sy * L + gxg);
        float Soth = Sv - c * c;
        if (Soth < 0.0f) Soth = 0.0f;
        sV  += c * c;
        sIx += c * gx * Soth;
        sIy += c * gy * Soth;
        lx += step_x; ly += step_y;
        if (lx >= rx_end) { lx -= rw; ly += 1; }
    }

    __shared__ float ws[32];
    sV  = block_sum(sV,  ws);
    sIx = block_sum(sIx, ws);
    sIy = block_sum(sIy, ws);
    if (tid == 0) {
        atomicAdd(&V_out[n],  sV);
        atomicAdd(&Ix_out[n], sIx);
        atomicAdd(&Iy_out[n], sIy);
    }
}

// Full reduce: V, Ix, Iy, perimeter, Cx, Cy, Cxx, Cyy.
// Used on rebind steps (rebind needs Cx/Cy/Cxx/Cyy) and on output steps
// (host reads V, Cx, Cy, perimeter for trajectory/VTK/checkpoint).
__global__ void k_reduce_mb_full(
    const float* __restrict__ phi,
    const int*   __restrict__ origin,
    const int*   __restrict__ rect,
    const float* __restrict__ S,
    float* __restrict__ V_out,
    float* __restrict__ Ix_out,
    float* __restrict__ Iy_out,
    float* __restrict__ peri_out,
    float* __restrict__ Cx_out,
    float* __restrict__ Cy_out,
    float* __restrict__ Cxx_out,
    float* __restrict__ Cyy_out,
    int N, int L, int CHUNK_PIXELS,
    int y_lo, int halo_h)
{
    const int n  = blockIdx.y;
    const int cb = blockIdx.x;
    if (n >= N) return;
    const int rx0 = rect[4*n + 0];
    const int ry0 = rect[4*n + 1];
    const int rw  = rect[4*n + 2];
    const int rh  = rect[4*n + 3];
    const int rect_total = rw * rh;
    const int p_start = cb * CHUNK_PIXELS;
    if (p_start >= rect_total) return;
    const int p_end = min(p_start + CHUNK_PIXELS, rect_total);
    const int gx0 = origin[2*n + 0];
    const int gy0 = origin[2*n + 1];
    const float* tile = phi + (size_t)n * TILE_AREA;
    const int tid = threadIdx.x;
    const int BS  = blockDim.x;

    float sV = 0.0f, sIx = 0.0f, sIy = 0.0f, sPeri = 0.0f;
    float sCx = 0.0f, sCy = 0.0f, sCxx = 0.0f, sCyy = 0.0f;

    const int step_x = BS % rw;
    const int step_y = BS / rw;
    const int rx_end = rx0 + rw;
    int p0 = p_start + tid;
    int lx = rx0 + (p0 % rw);
    int ly = ry0 + (p0 / rw);
    for (int p = p0; p < p_end; p += BS) {
        int idx = ly * TILE_T + lx;
        // rect invariant: stencil neighbors in-tile, no boundary guards.
        float c   = __ldg(tile + idx);
        float xp_ = __ldg(tile + idx + 1);
        float xm_ = __ldg(tile + idx - 1);
        float yp_ = __ldg(tile + idx + TILE_T);
        float ym_ = __ldg(tile + idx - TILE_T);
        float gx = 0.5f * (xp_ - xm_);
        float gy = 0.5f * (yp_ - ym_);
        int gxg = wrap_i(gx0 + lx, L);
        int gyg = wrap_i(gy0 + ly, L);
        int sy  = slab_local_y(gyg, y_lo, halo_h, L, L);
        float Sv   = __ldg(S + sy * L + gxg);
        float Soth = Sv - c * c;
        if (Soth < 0.0f) Soth = 0.0f;
        float c2 = c * c;
        float fx = (float)lx, fy = (float)ly;
        sV    += c2;
        sIx   += c * gx * Soth;
        sIy   += c * gy * Soth;
        sPeri += sqrtf(gx * gx + gy * gy);
        sCx   += c2 * fx;
        sCy   += c2 * fy;
        sCxx  += c2 * fx * fx;
        sCyy  += c2 * fy * fy;
        lx += step_x; ly += step_y;
        if (lx >= rx_end) { lx -= rw; ly += 1; }
    }

    __shared__ float ws[32];
    sV    = block_sum(sV,    ws);
    sIx   = block_sum(sIx,   ws);
    sIy   = block_sum(sIy,   ws);
    sPeri = block_sum(sPeri, ws);
    sCx   = block_sum(sCx,   ws);
    sCy   = block_sum(sCy,   ws);
    sCxx  = block_sum(sCxx,  ws);
    sCyy  = block_sum(sCyy,  ws);
    if (tid == 0) {
        atomicAdd(&V_out[n],    sV);
        atomicAdd(&Ix_out[n],   sIx);
        atomicAdd(&Iy_out[n],   sIy);
        atomicAdd(&peri_out[n], sPeri);
        atomicAdd(&Cx_out[n],   sCx);
        atomicAdd(&Cy_out[n],   sCy);
        atomicAdd(&Cxx_out[n],  sCxx);
        atomicAdd(&Cyy_out[n],  sCyy);
    }
}

__global__ void k_rhs_mb(
    const float* __restrict__ phi,
    const int*   __restrict__ origin,
    const int*   __restrict__ rect,
    const float* __restrict__ S,
    const float* __restrict__ gamma_cell,
    const float* __restrict__ v_A_cell,
    const float* __restrict__ dirx,
    const float* __restrict__ diry,
    const float* __restrict__ V_in,
    const float* __restrict__ Ix_in,
    const float* __restrict__ Iy_in,
    const float* __restrict__ tgt_radius,
    float* __restrict__ vx_out,
    float* __restrict__ vy_out,
    float* __restrict__ phi_out,
    int N, int L, int CHUNK_PIXELS,
    float lambda_, float kappa, float mu,
    float xi, float dt,
    int y_lo, int halo_h)
{
    const int n  = blockIdx.y;
    const int cb = blockIdx.x;
    if (n >= N) return;
    const int rx0 = rect[4*n + 0];
    const int ry0 = rect[4*n + 1];
    const int rw  = rect[4*n + 2];
    const int rh  = rect[4*n + 3];
    const int rect_total = rw * rh;
    const int p_start = cb * CHUNK_PIXELS;
    if (p_start >= rect_total) return;
    const int p_end = min(p_start + CHUNK_PIXELS, rect_total);
    const int gx0 = origin[2*n + 0];
    const int gy0 = origin[2*n + 1];
    const float* tile = phi     + (size_t)n * TILE_AREA;
    float*       outp = phi_out + (size_t)n * TILE_AREA;
    const int tid = threadIdx.x;
    const int BS  = blockDim.x;

    // Per-cell coefficients computed once per block, broadcast via shared.
    __shared__ float vx_s, vy_s, volC_s, dwC_s, repC_s, gam_s;
    if (tid == 0) {
        const float gam    = gamma_cell[n];
        const float vA     = v_A_cell[n];
        const float R      = tgt_radius[n];
        const float invXi  = 1.0f / xi;
        const float coeffV = 60.0f * kappa * invXi / (lambda_ * lambda_);
        const float piR2   = PI * R * R;
        const float Vn     = V_in[n];
        const float Ixn    = Ix_in[n];
        const float Iyn    = Iy_in[n];
        const float vx     = coeffV * Ixn + vA * dirx[n];
        const float vy     = coeffV * Iyn + vA * diry[n];
        vx_s   = vx;
        vy_s   = vy;
        volC_s = (2.0f * mu / piR2) * (piR2 - Vn);
        dwC_s  = 30.0f * gam   / (lambda_ * lambda_);
        repC_s = 30.0f * kappa / (lambda_ * lambda_);
        gam_s  = gam;
        // Only chunk 0 writes the per-cell velocity (avoids redundant writes).
        if (cb == 0) {
            vx_out[n] = vx;
            vy_out[n] = vy;
        }
    }
    __syncthreads();
    const float vx = vx_s, vy = vy_s;
    const float volC = volC_s, dwC = dwC_s, repC = repC_s;
    const float gam  = gam_s;

    const int step_x = BS % rw;
    const int step_y = BS / rw;
    const int rx_end = rx0 + rw;
    int p0 = p_start + tid;
    int lx = rx0 + (p0 % rw);
    int ly = ry0 + (p0 / rw);
    for (int p = p0; p < p_end; p += BS) {
        int idx = ly * TILE_T + lx;
        // rect invariant: stencil neighbors in-tile, no boundary guards.
        float c    = __ldg(tile + idx);
        float xp_  = __ldg(tile + idx + 1);
        float xm_  = __ldg(tile + idx - 1);
        float yp_  = __ldg(tile + idx + TILE_T);
        float ym_  = __ldg(tile + idx - TILE_T);
        float xpyp = __ldg(tile + idx + TILE_T + 1);
        float xpym = __ldg(tile + idx - TILE_T + 1);
        float xmyp = __ldg(tile + idx + TILE_T - 1);
        float xmym = __ldg(tile + idx - TILE_T - 1);
        float lap = lap9(c, xm_, xp_, ym_, yp_, xmym, xpym, xmyp, xpyp);
        float gx  = 0.5f * (xp_ - xm_);
        float gy  = 0.5f * (yp_ - ym_);
        int gxg = wrap_i(gx0 + lx, L);
        int gyg = wrap_i(gy0 + ly, L);
        int sy  = slab_local_y(gyg, y_lo, halo_h, L, L);
        float Sv   = __ldg(S + sy * L + gxg);
        float Soth = Sv - c * c;
        if (Soth < 0.0f) Soth = 0.0f;
        float dw  = c * (1.0f - c) * (1.0f - 2.0f * c);
        float rhs = gam * lap - dwC * dw + volC * c - repC * c * Soth;
        float adv = vx * gx + vy * gy;
        outp[idx] = c + dt * (rhs - adv);
        lx += step_x; ly += step_y;
        if (lx >= rx_end) { lx -= rw; ly += 1; }
    }
}

void launch_evolve(CellArrays& c, const SimParams& p, bool need_full_reduce,
                   cudaStream_t stream) {
    const int N = c.num_cells;
    if (N == 0) return;
    // Multi-block scatter+reduce+RHS pipeline (replaces fused k_evolve_l1).
    // Splitting reduce/RHS across many blocks pumps SM occupancy at all N,
    // including the large-N regime where a single block per cell would
    // bottleneck on the heavy fused kernel's register footprint.
    constexpr int CHUNK_PIXELS = 4096;
    constexpr int BS = 256;
    constexpr int chunks_per_cell = (TILE_AREA + CHUNK_PIXELS - 1) / CHUNK_PIXELS;
    dim3 grid_mb(chunks_per_cell, N);

    {
        int bsz = 128, gsz = (N + bsz - 1) / bsz;
        if (need_full_reduce) {
            k_zero_per_cell<<<gsz, bsz, 0, stream>>>(
                c.volumes, c.Ix, c.Iy, c.perimeters,
                c.Cx, c.Cy, c.Cxx, c.Cyy, N);
        } else {
            k_zero_per_cell3<<<gsz, bsz, 0, stream>>>(
                c.volumes, c.Ix, c.Iy, N);
        }
    }
    if (need_full_reduce) {
        k_reduce_mb_full<<<grid_mb, BS, 0, stream>>>(
            c.phi_in, c.origin, c.rect, c.S,
            c.volumes, c.Ix, c.Iy, c.perimeters,
            c.Cx, c.Cy, c.Cxx, c.Cyy,
            N, p.Nx, CHUNK_PIXELS,
            c.S_y_lo, c.S_halo_h);
    } else {
        k_reduce_mb_fast<<<grid_mb, BS, 0, stream>>>(
            c.phi_in, c.origin, c.rect, c.S,
            c.volumes, c.Ix, c.Iy,
            N, p.Nx, CHUNK_PIXELS,
            c.S_y_lo, c.S_halo_h);
    }
    k_rhs_mb<<<grid_mb, BS, 0, stream>>>(
        c.phi_in, c.origin, c.rect, c.S,
        c.gamma_cell, c.v_A_cell,
        c.polar_x, c.polar_y,
        c.volumes, c.Ix, c.Iy, c.tgt_radius,
        c.velocities_x, c.velocities_y,
        c.phi_out,
        N, p.Nx, CHUNK_PIXELS,
        (float)p.lambda, (float)p.kappa, (float)p.mu,
        (float)p.xi,     (float)p.dt,
        c.S_y_lo, c.S_halo_h);
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
    __shared__ int sold[4];   // rect_k (the rect before this rebind). Both
                              // buffers are zero outside this rect, so all
                              // work in this kernel lives in old ∪ new.
    __shared__ int sunion[4]; // [ux0, uy0, uw, uh]
    if (tid == 0) {
        // Snapshot the previous rect before we overwrite it. The kernel
        // invariant (maintained by every rebind) is: both phi_in and
        // phi_out are zero outside `sold` at kernel entry.
        sold[0] = rect[4*n + 0];
        sold[1] = rect[4*n + 1];
        sold[2] = rect[4*n + 2];
        sold[3] = rect[4*n + 3];

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
        // Per-axis half-widths: hw = ceil(2*sigma + margin). The 2*sigma
        // term tracks the cell's actual per-axis extent (for a uniform
        // ellipse, semi-axis ~= 2*sigma), so a horizontally-stretched
        // cell gets a wide hw_x and a short hw_y, accurately reflecting
        // the cell wall.
        //
        // The additive margin is the safety buffer past the cell wall:
        //   margin = K * (R/4) * sqrt(gamma_ref / gamma)
        // K (subdomain_padding, default 2) controls the buffer size.
        // K=2 reproduces the historical R/2 padding for stiff cells.
        // Soft cells (gamma < gamma_ref) move/deform faster between
        // rebinds, so they get a sqrt(gamma_ref/gamma) larger cushion.
        // The cushion is bounded by the hmax = Th-1 clamp below.
        float R = tgt_radius[n];
        float gn = gamma_cell[n];
        float soft_scale = 1.0f;
        if (gn > 0.0f && gn < gamma_ref) {
            soft_scale = sqrtf(gamma_ref / gn);
        }
        float margin = bbox_k * 0.25f * R * soft_scale;
        int hwx = (int)ceilf(2.0f * sigx + margin);
        int hwy = (int)ceilf(2.0f * sigy + margin);
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

        // Union of old and new rects. We only need to touch pixels in
        // this region: outside the union both buffers are already zero
        // (invariant from the previous rebind + evolve writing only
        // inside the rect).
        int ox0 = sold[0], oy0 = sold[1];
        int ox1 = ox0 + sold[2], oy1 = oy0 + sold[3];
        int nx0 = srect[0], ny0 = srect[1];
        int nx1 = nx0 + srect[2], ny1 = ny0 + srect[3];
        int ux0 = ox0 < nx0 ? ox0 : nx0;
        int uy0 = oy0 < ny0 ? oy0 : ny0;
        int ux1 = ox1 > nx1 ? ox1 : nx1;
        int uy1 = oy1 > ny1 ? oy1 : ny1;
        sunion[0] = ux0;
        sunion[1] = uy0;
        sunion[2] = ux1 - ux0;
        sunion[3] = uy1 - uy0;
    }
    __syncthreads();
    const int sx = sshift[0];
    const int sy = sshift[1];
    const int rx0 = srect[0], ry0 = srect[1];
    const int rx1 = rx0 + srect[2], ry1 = ry0 + srect[3];
    const int ox0_old = sold[0], oy0_old = sold[1];
    const int ox1_old = ox0_old + sold[2], oy1_old = oy0_old + sold[3];
    const int ux0 = sunion[0], uy0 = sunion[1];
    const int uw  = sunion[2], uh  = sunion[3];
    const int u_total = uw * uh;

    // Phase 1: write shifted source into phi_out, but only inside the
    // union bbox. Outside the union, phi_out is already 0 (invariant).
    //   - pixel in new rect:        write shifted phi_in lookup
    //   - pixel in old rect only:   write 0 (clears stale evolve data)
    //   - pixel in neither (rare):  already 0; skip
    for (int p = tid; p < u_total; p += BS) {
        int ox = ux0 + (p % uw);
        int oy = uy0 + (p / uw);
        bool in_new = (ox >= rx0 && ox < rx1 && oy >= ry0 && oy < ry1);
        bool in_old = (ox >= ox0_old && ox < ox1_old && oy >= oy0_old && oy < oy1_old);
        float v = 0.0f;
        if (in_new) {
            int sxi = ox + sx;
            int syi = oy + sy;
            if (sxi >= 0 && sxi < TILE_T && syi >= 0 && syi < TILE_T) {
                v = tin[syi * TILE_T + sxi];
            }
        }
        if (in_new || in_old) {
            tout[oy * TILE_T + ox] = v;
        }
    }
    __syncthreads();

    // Phase 2: scrub stale data in tin (the source buffer). After the
    // caller swaps, this becomes phi_out for the next evolve, which only
    // writes INSIDE the new rect; we must zero everything in old\new now.
    // No race: Phase 1 finished reading tin (sync above).
    for (int p = tid; p < u_total; p += BS) {
        int ox = ux0 + (p % uw);
        int oy = uy0 + (p / uw);
        bool in_new = (ox >= rx0 && ox < rx1 && oy >= ry0 && oy < ry1);
        bool in_old = (ox >= ox0_old && ox < ox1_old && oy >= oy0_old && oy < oy1_old);
        if (in_old && !in_new) {
            tin[oy * TILE_T + ox] = 0.0f;
        }
    }
}

void launch_rebind(CellArrays& c, float bbox_k, float gamma_ref,
                   cudaStream_t stream) {
    const int N = c.num_cells;
    if (N == 0) return;
    k_rebind<<<N, 256, 0, stream>>>(c.phi_in, c.phi_out, c.origin, c.rect,
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

void launch_polar(CellArrays& c, const SimParams& p, cudaStream_t stream) {
    const int N = c.num_cells;
    if (N == 0 || p.v_A == 0.0 || p.tau <= 0.0) return;
    k_polar<<<(N + 255) / 256, 256, 0, stream>>>(
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
                           int count,
                           cudaStream_t stream)
{
    if (count <= 0) return;
    k_apply_scripted<<<(count + 31) / 32, 32, 0, stream>>>(
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
    float lambda_, float kappa, float xi,
    int y_lo, int halo_h)
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
        int sy  = slab_local_y(gyg, y_lo, halo_h, L, L);
        float Sv   = S[sy * L + gxg];
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

    __shared__ float ssmem[8 * 32];
    float v0 = block_sum(sV,   ssmem + 0*32);
    float v1 = block_sum(sIx,  ssmem + 1*32);
    float v2 = block_sum(sIy,  ssmem + 2*32);
    float v3 = block_sum(sCx,  ssmem + 3*32);
    float v4 = block_sum(sCy,  ssmem + 4*32);
    float v5 = block_sum(sCxx, ssmem + 5*32);
    float v6 = block_sum(sCyy, ssmem + 6*32);
    float v7 = block_sum(sP,   ssmem + 7*32);
    if (tid == 0) {
        float Vn  = v0;
        float Ixn = v1;
        float Iyn = v2;
        float coeffV = 60.0f * kappa / (xi * lambda_ * lambda_);
        V_out[n]    = Vn;
        Cx_out[n]   = v3;
        Cy_out[n]   = v4;
        Cxx_out[n]  = v5;
        Cyy_out[n]  = v6;
        peri_out[n] = v7;
        vx_out[n]   = coeffV * Ixn + vA * dirx[n];
        vy_out[n]   = coeffV * Iyn + vA * diry[n];
    }
}

void launch_initial_velocity(CellArrays& c, const SimParams& p) {
    const int N = c.num_cells;
    if (N == 0) return;
    // S sized by slab (== Ny for G=1).
    const size_t Sbytes = (size_t)c.S_ext_height * p.Nx * sizeof(float);
    cudaMemsetAsync(c.S, 0, Sbytes);
    {
        constexpr int CHUNK_PIXELS = 2048;
        constexpr int BS = 128;
        constexpr int MAX_CHUNKS = (TILE_AREA + CHUNK_PIXELS - 1) / CHUNK_PIXELS;
        dim3 grid(MAX_CHUNKS, N);
        k_scatter_S<<<grid, BS>>>(c.phi_in, c.origin, c.rect, c.S,
                                  N, p.Nx, CHUNK_PIXELS,
                                  c.S_y_lo, c.S_halo_h);
    }
    k_initial_velocity<<<N, 256>>>(
        c.phi_in, c.origin, c.rect, c.S,
        c.v_A_cell, c.polar_x, c.polar_y,
        c.volumes, c.Cx, c.Cy, c.Cxx, c.Cyy, c.perimeters,
        c.velocities_x, c.velocities_y,
        N, p.Nx,
        (float)p.lambda, (float)p.kappa, (float)p.xi,
        c.S_y_lo, c.S_halo_h);
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
