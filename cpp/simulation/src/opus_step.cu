// opus_step.cu — single-pass fused step kernel implementation.
//
// See opus_step.cuh for design overview. Ported from the standalone
// reference in opus_port/cellstep.cu, integrated against cell_sim's
// CellArrays + SimParams + slab_local_y conventions.

#include "opus_step.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

using namespace opus;

namespace {

constexpr int NTH   = BX * BY;       // 256
constexpr int NWARP = NTH / 32;      // 8
constexpr int HW    = OW + 2;        // 34
constexpr int HH    = OH + 2;        // 34

#define OPUS_CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){              \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,               \
            cudaGetErrorString(e_)); std::exit(1);} }while(0)

__device__ __forceinline__ int wrap_g(int v, int L) {
    if (v >= L) return v - L;
    if (v < 0)  return v + L;
    return v;
}

__device__ __forceinline__ float warp_sum_op(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Block reduction over NTH=256 threads (NWARP=8). Full sum lands in thread 0.
// `sm` must point at >= NWARP floats. Trailing __syncthreads makes back-to-back
// calls safe with the same scratch.
__device__ __forceinline__ float block_sum_op(float v, float* sm) {
    v = warp_sum_op(v);
    int lane = threadIdx.x & 31;
    int wid  = (threadIdx.y * BX + threadIdx.x) >> 5;
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    float r = 0.f;
    if (wid == 0) {
        r = (lane < NWARP) ? sm[lane] : 0.f;
        r = warp_sum_op(r);
    }
    __syncthreads();
    return r;
}

template<bool DO_EXT, bool DO_REBIND>
__global__ void k_opus_step(
    const float* __restrict__ phiIn,
    float*       __restrict__ phiOut,
    const float* __restrict__ Sin,
    float*       __restrict__ Sout,
    const double* __restrict__ Vlag,
    const double* __restrict__ Ixlag,
    const double* __restrict__ Iylag,
    double*       __restrict__ Vout,
    double*       __restrict__ Ixout,
    double*       __restrict__ Iyout,
    double*       __restrict__ perim_out,
    double*       __restrict__ Cx_out,
    double*       __restrict__ Cy_out,
    double*       __restrict__ Cxx_out,
    double*       __restrict__ Cyy_out,
    const int*       __restrict__ origin,      // SOURCE-frame origin
    const int*       __restrict__ rect,        // SOURCE rect (worklist domain)
    const int*       __restrict__ dst_rect,    // DEST rect (DO_REBIND only; nullptr otherwise)
    const int*       __restrict__ shift_xy,    // (sx, sy) per cell (DO_REBIND only; nullptr otherwise)
    const WorkItem*  __restrict__ work,
    const int*       __restrict__ work_count,  // device-side actual count; early-exit if blockIdx.x >= *count
    const float* __restrict__ gamma_cell,
    const float* __restrict__ vA_cell,
    const float* __restrict__ dirx_c,
    const float* __restrict__ diry_c,
    const float* __restrict__ tgt_R_c,
    int Nx, int Ny,
    int S_y_lo, int S_halo_h, int S_ext_height,
    float lambda_, float kappa, float mu, float xi, float dt)
{
    __shared__ float sm[HH][HW];
    __shared__ float red[NWARP * (DO_EXT ? 8 : 3)];

    // Early-exit on grid blocks beyond the actual worklist count. Grid is
    // launched at d_work_cap (worst-case) so we don't need to sync the
    // count back to host; the in-kernel check skips dead blocks for ~free.
    if (blockIdx.x >= *work_count) return;

    const WorkItem wi = work[blockIdx.x];
    const int n        = wi.tile;
    // Work-item (sx, sy) is interpreted as the DESTINATION-frame sub-tile origin.
    // On non-rebind steps shift = (0, 0) so destination == source.
    const int dst_sx   = wi.sx;
    const int dst_sy   = wi.sy;

    const int gx0 = origin[2*n + 0];
    const int gy0 = origin[2*n + 1];
    const int rx0 = rect[4*n + 0];
    const int ry0 = rect[4*n + 1];
    const int rw  = rect[4*n + 2];
    const int rh  = rect[4*n + 3];
    const int rxe = rx0 + rw;
    const int rye = ry0 + rh;

    // Shift: source = destination + shift (note: matches opus's convention,
    // where shiftXY = -recenter_shift, so new_origin = old_origin + (-shift)).
    int shf_x = 0, shf_y = 0;
    if constexpr (DO_REBIND) {
        shf_x = shift_xy[2*n + 0];
        shf_y = shift_xy[2*n + 1];
    }
    // Destination rect (only meaningful on rebind steps). Pre-clamped subset
    // of source rect by compute_rebind_meta, so the worklist over source
    // rect fully covers the destination footprint.
    int drx0 = rx0, dry0 = ry0, drxe = rxe, drye = rye;
    if constexpr (DO_REBIND) {
        drx0 = dst_rect[4*n + 0];
        dry0 = dst_rect[4*n + 1];
        drxe = drx0 + dst_rect[4*n + 2];
        drye = dry0 + dst_rect[4*n + 3];
    }

    const float* ph = phiIn  + (size_t)n * TILE_AREA;
    float*       po = phiOut + (size_t)n * TILE_AREA;
    const int tid   = threadIdx.y * BX + threadIdx.x;

    // 34x34 halo load from SOURCE frame (dst + shift - 1).
    // Non-rebind: shift=0 so this is just (sx-1..sx+OW, sy-1..sy+OH) inside [0, TILE_T).
    // Rebind: same bounds shifted; we guard against OOB defensively.
    const int hx0 = dst_sx + shf_x - 1;
    const int hy0 = dst_sy + shf_y - 1;
    #pragma unroll
    for (int idx = tid; idx < HW * HH; idx += NTH) {
        int lxi = idx % HW, lyi = idx / HW;
        int xi  = hx0 + lxi;
        int yi  = hy0 + lyi;
        float v = 0.0f;
        if constexpr (DO_REBIND) {
            if ((unsigned)xi < (unsigned)TILE_T && (unsigned)yi < (unsigned)TILE_T) {
                v = ph[(size_t)yi * TILE_T + xi];
            }
        } else {
            v = ph[(size_t)yi * TILE_T + xi];
        }
        sm[lyi][lxi] = v;
    }
    __syncthreads();

    // Per-tile coefficients from LAGGED V/Ix/Iy. Lagged moments are physical
    // (frame-invariant) so they're correct under rebind without any fixup.
    const float gam    = gamma_cell[n];
    const float vA     = vA_cell[n];
    const float R      = tgt_R_c[n];
    const float piR2   = (float)M_PI * R * R;
    const float V_lag  = (float)Vlag[n];
    const float Ix_lag = (float)Ixlag[n];
    const float Iy_lag = (float)Iylag[n];
    const float dwC    = gam * bulk_coeff<float>(lambda_);
    const float repC   = interaction_coeff<float>(kappa, lambda_);
    const float volC   = (2.0f * mu / piR2) * (piR2 - V_lag);
    const float coeffV = motility_coeff<float>(kappa, xi, lambda_);
    const float vx     = coeffV * Ix_lag + vA * dirx_c[n];
    const float vy     = coeffV * Iy_lag + vA * diry_c[n];

    // Per-pixel reductions in f32 (bounded values, ~1024 adds per CTA —
    // well within f32 mantissa). The cross-CTA combine via atomicAdd to the
    // per-cell outputs is in f64 to commute that nondeterministic step.
    float v=0.f, ix=0.f, iy=0.f;
    float pp=0.f, cx=0.f, cy=0.f, cxx=0.f, cyy=0.f;

    #pragma unroll
    for (int r = 0; r < RY; ++r) {
        const int oy = threadIdx.y + r * BY;
        // Destination tile-local pixel.
        const int dlx = dst_sx + threadIdx.x;
        const int dly = dst_sy + oy;
        // Source tile-local pixel (== destination on non-rebind).
        const int slx = dlx + shf_x;
        const int sly = dly + shf_y;

        // Worklist domain is the SOURCE rect (where phi is non-zero). Skip
        // pixels whose source is outside the source rect.
        if (slx >= rxe || sly >= rye) continue;

        // On rebind steps, check destination is inside new (subset) rect.
        bool dst_in_new = true;
        if constexpr (DO_REBIND) {
            dst_in_new = (dlx >= drx0 && dlx < drxe &&
                          dly >= dry0 && dly < drye);
        }

        if constexpr (DO_REBIND) {
            if (!dst_in_new) {
                // Destination outside new rect: zero it (clears stale phi
                // from a prior parity write in this slot).
                if ((unsigned)dlx < (unsigned)TILE_T &&
                    (unsigned)dly < (unsigned)TILE_T) {
                    po[(size_t)dly * TILE_T + dlx] = 0.0f;
                }
                continue;
            }
        }

        const int tx = threadIdx.x + 1, ty = oy + 1;
        const float c  = sm[ty  ][tx  ];
        const float e  = sm[ty  ][tx+1];
        const float w  = sm[ty  ][tx-1];
        const float nN = sm[ty+1][tx  ];
        const float sS = sm[ty-1][tx  ];
        const float ne = sm[ty+1][tx+1];
        const float nw = sm[ty+1][tx-1];
        const float se = sm[ty-1][tx+1];
        const float sw = sm[ty-1][tx-1];

        // 9-point isotropic Laplacian (h=1), central-difference gradients.
        const float lap = (4.0f*(e+w+nN+sS) + (ne+nw+se+sw) - 20.0f*c) * (1.0f/6.0f);
        const float gx = 0.5f * (e  - w);
        const float gy = 0.5f * (nN - sS);

        // Scatter into S at SOURCE-frame global address. Origin is updated by
        // apply_rebind_meta AFTER this kernel, so (gx0 + slx) here equals
        // (new_origin + dlx) on the next step — the same physical pixel.
        const int gxg = wrap_g(gx0 + slx, Nx);
        const int gyg = wrap_g(gy0 + sly, Ny);
        const int syL = slab_local_y(gyg, S_y_lo, S_halo_h, S_ext_height, Ny);
        const size_t gIdx = (size_t)syL * Nx + gxg;

        const float Sg = Sin[gIdx];
        const float term = fmaxf(0.0f, Sg - c*c);

        // (a) FRESH reductions of phi_in. Frame-invariant sums; use source
        // values regardless of DO_REBIND.
        const float c2 = c * c;
        v  += c2;
        ix += c * gx * term;
        iy += c * gy * term;
        if constexpr (DO_EXT) {
            // Position moments: use DESTINATION coords so the NEXT rebind's
            // COM is expressed in the post-rebind frame. (On non-rebind
            // steps dlx == slx so the choice doesn't matter.)
            const float flx = (float)dlx, fly = (float)dly;
            pp  += sqrtf(gx*gx + gy*gy);
            cx  += c2 * flx;
            cy  += c2 * fly;
            cxx += c2 * flx * flx;
            cyy += c2 * fly * fly;
        }

        // (b) Palmieri RHS + scatter next-step S.
        const float dw  = c * (1.0f - c) * (1.0f - 2.0f * c);
        const float rhs = gam * lap - dwC * dw + volC * c - repC * c * term;
        const float adv = vx * gx + vy * gy;
        const float pn  = c + dt * (rhs - adv);
        po[(size_t)dly * TILE_T + dlx] = pn;
        atomicAdd(&Sout[gIdx], pn * pn);
    }

    float bv = block_sum_op(v,  red);
    float bx = block_sum_op(ix, red +   NWARP);
    float by = block_sum_op(iy, red + 2*NWARP);
    if (tid == 0) {
        // Widen to f64 at the cross-CTA combine. f64 atomicAdd of values
        // that fit comfortably in f64 mantissa is order-insensitive at the
        // ULP level, removing the bulk of run-to-run nondeterminism.
        atomicAdd(&Vout [n], (double)bv);
        atomicAdd(&Ixout[n], (double)bx);
        atomicAdd(&Iyout[n], (double)by);
    }
    if constexpr (DO_EXT) {
        float bp  = block_sum_op(pp,  red + 3*NWARP);
        float bcx = block_sum_op(cx,  red + 4*NWARP);
        float bcy = block_sum_op(cy,  red + 5*NWARP);
        float bxx = block_sum_op(cxx, red + 6*NWARP);
        float byy = block_sum_op(cyy, red + 7*NWARP);
        if (tid == 0) {
            atomicAdd(&perim_out[n], (double)bp);
            atomicAdd(&Cx_out [n],   (double)bcx);
            atomicAdd(&Cy_out [n],   (double)bcy);
            atomicAdd(&Cxx_out[n],   (double)bxx);
            atomicAdd(&Cyy_out[n],   (double)byy);
        }
    }
}

__global__ void k_opus_finalize_velocity(
    int N,
    const double* __restrict__ Ix, const double* __restrict__ Iy,
    const float* __restrict__ vA, const float* __restrict__ dirx,
    const float* __restrict__ diry,
    float* __restrict__ vx_out, float* __restrict__ vy_out,
    float lambda_, float kappa, float xi)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float c = motility_coeff<float>(kappa, xi, lambda_);
    vx_out[n] = c * (float)Ix[n] + vA[n] * dirx[n];
    vy_out[n] = c * (float)Iy[n] + vA[n] * diry[n];
}

// ---------------------------------------------------------------------------
// k_opus_compute_rebind_meta — one thread per cell. Computes the integer
// shift (sx, sy) so the cell's COM lands at tile-center, and the new rect
// from second-moment width. Mirrors the math in kernels.cu's k_rebind.
//
// Reads V, Cx, Cy, Cxx, Cyy from the parity-current pools (populated by
// the previous DO_EXT step). Writes shift_xy[2n..2n+2] and new_rect[4n..4n+4].
//
// Note: V/Cx/Cy/Cxx/Cyy at this point characterize the phi we are ABOUT to
// evolve (phi_in to the upcoming step), not the post-evolution phi. This
// matches legacy semantics (one-step lag between moments and the phi being
// shifted), since rebind uses the previous evolve's moments to shift the
// just-evolved phi.
// ---------------------------------------------------------------------------
__global__ void k_opus_compute_rebind_meta(
    int N,
    const double* __restrict__ V,
    const double* __restrict__ Cx,
    const double* __restrict__ Cy,
    const double* __restrict__ Cxx,
    const double* __restrict__ Cyy,
    const int*   __restrict__ rect,
    const float* __restrict__ gamma_cell,
    const float* __restrict__ tgt_radius,
    int*  __restrict__ shift_xy,
    int*  __restrict__ new_rect,
    float bbox_k, float gamma_ref,
    int   bbox_align, int bbox_min)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const int Th = TILE_T >> 1;

    double Vn   = V[n];
    double invV = (Vn > 1e-6) ? 1.0 / Vn : 0.0;
    double mxd  = Cx[n] * invV;
    double myd  = Cy[n] * invV;
    int   sx   = __double2int_rn(mxd) - Th;
    int   sy   = __double2int_rn(myd) - Th;
    shift_xy[2*n + 0] = sx;
    shift_xy[2*n + 1] = sy;

    double varxd = Cxx[n] * invV - mxd * mxd;
    double varyd = Cyy[n] * invV - myd * myd;
    if (varxd < 0.0) varxd = 0.0;
    if (varyd < 0.0) varyd = 0.0;
    float sigx = (float)sqrt(varxd);
    float sigy = (float)sqrt(varyd);

    float gn = gamma_cell[n];
    float soft_scale = 1.0f;
    if (gn > 0.0f && gn < gamma_ref) {
        soft_scale = sqrtf(gamma_ref / gn);
    }
    float R       = tgt_radius[n];
    float margin  = bbox_k * 0.25f * R * soft_scale;
    int   hwx     = (int)ceilf(2.0f * sigx + margin);
    int   hwy     = (int)ceilf(2.0f * sigy + margin);
    hwx = ((hwx + bbox_align - 1) / bbox_align) * bbox_align;
    hwy = ((hwy + bbox_align - 1) / bbox_align) * bbox_align;
    const int hmax = Th - 1;
    if (hwx > hmax) hwx = hmax;
    if (hwy > hmax) hwy = hmax;
    if (hwx < bbox_min) hwx = bbox_min;
    if (hwy < bbox_min) hwy = bbox_min;

    // New rect (destination frame), centered at tile center.
    int nrx0 = Th - hwx;
    int nry0 = Th - hwy;
    int nrxe = Th + hwx;
    int nrye = Th + hwy;

    // CLAMP new_rect to subset of old_rect so that the source-frame worklist
    // (over old_rect) fully covers every destination pixel of the new rect.
    // Without this we'd need a union worklist; with it the same worklist is
    // reusable for regular and rebind steps.
    const int orx0 = rect[4*n + 0];
    const int ory0 = rect[4*n + 1];
    const int orxe = orx0 + rect[4*n + 2];
    const int orye = ory0 + rect[4*n + 3];
    if (nrx0 < orx0) nrx0 = orx0;
    if (nry0 < ory0) nry0 = ory0;
    if (nrxe > orxe) nrxe = orxe;
    if (nrye > orye) nrye = orye;
    // Also guarantee 1px halo margin (stencil reads).
    if (nrx0 < 1)            nrx0 = 1;
    if (nry0 < 1)            nry0 = 1;
    if (nrxe > TILE_T - 1)   nrxe = TILE_T - 1;
    if (nrye > TILE_T - 1)   nrye = TILE_T - 1;

    new_rect[4*n + 0] = nrx0;
    new_rect[4*n + 1] = nry0;
    new_rect[4*n + 2] = nrxe - nrx0;
    new_rect[4*n + 3] = nrye - nry0;
}

// ---------------------------------------------------------------------------
// k_opus_apply_rebind_meta — one thread per cell. Applies the per-cell
// (sx, sy) shift to origin and copies new_rect into rect. Called after the
// fused-rebind step kernel completes so subsequent kernels see the new
// geometry.
// ---------------------------------------------------------------------------
__global__ void k_opus_apply_rebind_meta(
    int N,
    int* __restrict__ origin,
    int* __restrict__ rect,
    const int* __restrict__ shift_xy,
    const int* __restrict__ new_rect)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    origin[2*n + 0] += shift_xy[2*n + 0];
    origin[2*n + 1] += shift_xy[2*n + 1];
    rect[4*n + 0]    = new_rect[4*n + 0];
    rect[4*n + 1]    = new_rect[4*n + 1];
    rect[4*n + 2]    = new_rect[4*n + 2];
    rect[4*n + 3]    = new_rect[4*n + 3];
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// k_opus_build_worklist — device-side worklist construction.
//
// One thread per cell. Each thread enumerates its cell's 32x32 sub-tiles
// inside the cell's rect and appends them as WorkItem entries via a single
// global atomic counter. Avoids the host round-trip + blocking stream sync
// of the previous host-side build, keeping the pipeline full across rebinds.
//
// Sub-tile emission order is non-deterministic across runs (atomic order),
// but within a single launch the work is identical — every WorkItem ends
// up in the array, just at a non-deterministic slot. The opus kernel is
// indifferent to work order, so this has no observable effect.
// ---------------------------------------------------------------------------
__global__ void k_opus_build_worklist(
    int N,
    const int* __restrict__ rect,
    WorkItem*  __restrict__ out_work,
    int*       __restrict__ out_count)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int rx0 = rect[4*n + 0];
    const int ry0 = rect[4*n + 1];
    const int rw  = rect[4*n + 2];
    const int rh  = rect[4*n + 3];
    for (int sy = ry0; sy < ry0 + rh; sy += OH) {
        for (int sx = rx0; sx < rx0 + rw; sx += OW) {
            int idx = atomicAdd(out_count, 1);
            WorkItem w = { n, sx, sy };
            out_work[idx] = w;
        }
    }
}

}  // namespace

void launch_opus_step(CellArrays& c, const SimParams& p,
                      int parity, bool need_full, cudaStream_t stream)
{
    const int N = c.num_cells;
    if (N == 0 || c.workCount == 0) return;
    const int q = parity ^ 1;
    const size_t S_elems = (size_t)c.S_ext_height * p.Nx;

    // Zero next-step scatter/atomic targets.
    OPUS_CK(cudaMemsetAsync(c.S_pool [q], 0, S_elems * sizeof(float), stream));
    OPUS_CK(cudaMemsetAsync(c.V_pool [q], 0, N * sizeof(double), stream));
    OPUS_CK(cudaMemsetAsync(c.Ix_pool[q], 0, N * sizeof(double), stream));
    OPUS_CK(cudaMemsetAsync(c.Iy_pool[q], 0, N * sizeof(double), stream));
    if (need_full) {
        OPUS_CK(cudaMemsetAsync(c.perimeters, 0, N * sizeof(double), stream));
        OPUS_CK(cudaMemsetAsync(c.Cx,         0, N * sizeof(double), stream));
        OPUS_CK(cudaMemsetAsync(c.Cy,         0, N * sizeof(double), stream));
        OPUS_CK(cudaMemsetAsync(c.Cxx,        0, N * sizeof(double), stream));
        OPUS_CK(cudaMemsetAsync(c.Cyy,        0, N * sizeof(double), stream));
    }

    dim3 blk(BX, BY);
    if (need_full) {
        k_opus_step<true, false><<<c.workCount, blk, 0, stream>>>(
            c.phi_in, c.phi_out,
            c.S_pool[parity], c.S_pool[q],
            c.V_pool [parity], c.Ix_pool[parity], c.Iy_pool[parity],
            c.V_pool [q],      c.Ix_pool[q],      c.Iy_pool[q],
            c.perimeters, c.Cx, c.Cy, c.Cxx, c.Cyy,
            c.origin, c.rect, /*dst_rect=*/nullptr, /*shift_xy=*/nullptr,
            (const WorkItem*)c.d_work, c.d_work_count,
            c.gamma_cell, c.v_A_cell, c.polar_x, c.polar_y, c.tgt_radius,
            (int)p.Nx, (int)p.Ny,
            c.S_y_lo, c.S_halo_h, c.S_ext_height,
            (float)p.lambda, (float)p.kappa, (float)p.mu,
            (float)p.xi, (float)p.dt);
    } else {
        k_opus_step<false, false><<<c.workCount, blk, 0, stream>>>(
            c.phi_in, c.phi_out,
            c.S_pool[parity], c.S_pool[q],
            c.V_pool [parity], c.Ix_pool[parity], c.Iy_pool[parity],
            c.V_pool [q],      c.Ix_pool[q],      c.Iy_pool[q],
            nullptr, nullptr, nullptr, nullptr, nullptr,
            c.origin, c.rect, /*dst_rect=*/nullptr, /*shift_xy=*/nullptr,
            (const WorkItem*)c.d_work, c.d_work_count,
            c.gamma_cell, c.v_A_cell, c.polar_x, c.polar_y, c.tgt_radius,
            (int)p.Nx, (int)p.Ny,
            c.S_y_lo, c.S_halo_h, c.S_ext_height,
            (float)p.lambda, (float)p.kappa, (float)p.mu,
            (float)p.xi, (float)p.dt);
    }
}

void launch_opus_finalize_velocity(CellArrays& c, const SimParams& p,
                                   int parity, cudaStream_t stream)
{
    const int N = c.num_cells;
    if (N == 0) return;
    constexpr int BS = 128;
    int blocks = (N + BS - 1) / BS;
    k_opus_finalize_velocity<<<blocks, BS, 0, stream>>>(
        N, c.Ix_pool[parity], c.Iy_pool[parity],
        c.v_A_cell, c.polar_x, c.polar_y,
        c.velocities_x, c.velocities_y,
        (float)p.lambda, (float)p.kappa, (float)p.xi);
}

int build_opus_work_list_host(CellArrays& c)
{
    const int N = c.num_cells;
    if (N == 0) { c.workCount = 0; return 0; }
    // The cudaMemcpy below runs on the default stream; cell_sim kernels run
    // on step_stream. Make sure any pending writes to c.rect are visible.
    OPUS_CK(cudaDeviceSynchronize());
    std::vector<int> h_rect(4 * N);
    OPUS_CK(cudaMemcpy(h_rect.data(), c.rect, 4 * N * sizeof(int),
                       cudaMemcpyDeviceToHost));
    std::vector<WorkItem> h_work;
    h_work.reserve(N * OPUS_MAX_WORKITEMS_PER_CELL);
    for (int n = 0; n < N; ++n) {
        const int rx0 = h_rect[4*n + 0];
        const int ry0 = h_rect[4*n + 1];
        const int rw  = h_rect[4*n + 2];
        const int rh  = h_rect[4*n + 3];
        for (int sy = ry0; sy < ry0 + rh; sy += OH)
            for (int sx = rx0; sx < rx0 + rw; sx += OW)
                h_work.push_back({n, sx, sy});
    }
    const int wc = (int)h_work.size();
    if (wc > c.d_work_cap) {
        fprintf(stderr,
            "[opus] work list (%d) exceeds capacity (%d). "
            "Increase d_work_cap in alloc_gpu.\n", wc, c.d_work_cap);
        std::exit(1);
    }
    OPUS_CK(cudaMemcpy((WorkItem*)c.d_work, h_work.data(), wc * sizeof(WorkItem),
                       cudaMemcpyHostToDevice));
    // Mirror count to device so the in-kernel early-exit check on
    // k_opus_step (blockIdx.x >= *work_count) is a no-op when grid == wc.
    if (c.d_work_count) {
        OPUS_CK(cudaMemcpy(c.d_work_count, &wc, sizeof(int),
                           cudaMemcpyHostToDevice));
    }
    c.workCount = wc;
    return wc;
}

void launch_opus_seed_parity_mirror(CellArrays& c, const SimParams& p,
                                    int from_parity, cudaStream_t stream)
{
    const int N = c.num_cells;
    if (N == 0) return;
    const int f = from_parity & 1;
    const int t = f ^ 1;
    const size_t S_elems = (size_t)c.S_ext_height * p.Nx;
    OPUS_CK(cudaMemcpyAsync(c.S_pool [t], c.S_pool [f], S_elems*sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
    OPUS_CK(cudaMemcpyAsync(c.V_pool [t], c.V_pool [f], N*sizeof(double),
                            cudaMemcpyDeviceToDevice, stream));
    OPUS_CK(cudaMemcpyAsync(c.Ix_pool[t], c.Ix_pool[f], N*sizeof(double),
                            cudaMemcpyDeviceToDevice, stream));
    OPUS_CK(cudaMemcpyAsync(c.Iy_pool[t], c.Iy_pool[f], N*sizeof(double),
                            cudaMemcpyDeviceToDevice, stream));
}

namespace {
__global__ void k_opus_init_lagged_moments(
    int N,
    const float* __restrict__ tgt_radius,
    double* __restrict__ V_out,
    double* __restrict__ Ix_out,
    double* __restrict__ Iy_out)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float R = tgt_radius[n];
    V_out[n]  = (double)(M_PI * R * R);
    Ix_out[n] = 0.0;
    Iy_out[n] = 0.0;
}
}  // namespace

void launch_opus_init_lagged_moments(CellArrays& c, const SimParams& /*p*/,
                                     int parity, cudaStream_t stream)
{
    const int N = c.num_cells;
    if (N == 0) return;
    const int q = parity & 1;
    constexpr int BS = 128;
    int blocks = (N + BS - 1) / BS;
    k_opus_init_lagged_moments<<<blocks, BS, 0, stream>>>(
        N, c.tgt_radius,
        c.V_pool[q], c.Ix_pool[q], c.Iy_pool[q]);
}

// ---------------------------------------------------------------------------
// Fused-rebind launchers
// ---------------------------------------------------------------------------

void launch_opus_compute_rebind_meta(CellArrays& c, const SimParams& p,
                                     int parity, cudaStream_t stream)
{
    const int N = c.num_cells;
    if (N == 0) return;
    constexpr int BS = 128;
    int blocks = (N + BS - 1) / BS;
    k_opus_compute_rebind_meta<<<blocks, BS, 0, stream>>>(
        N,
        c.V_pool [parity], c.Cx, c.Cy, c.Cxx, c.Cyy,
        c.rect,
        c.gamma_cell, c.tgt_radius,
        c.shift_xy, c.new_rect,
        (float)p.subdomain_padding, (float)p.gamma,
        TILE_BBOX_ALIGN, TILE_BBOX_MIN);
}

void launch_opus_apply_rebind_meta(CellArrays& c, cudaStream_t stream)
{
    const int N = c.num_cells;
    if (N == 0) return;
    constexpr int BS = 128;
    int blocks = (N + BS - 1) / BS;
    k_opus_apply_rebind_meta<<<blocks, BS, 0, stream>>>(
        N, c.origin, c.rect, c.shift_xy, c.new_rect);
}

// Device-side worklist build. Reuses the existing worklist for BOTH regular
// and rebind steps: new_rect is clamped to a subset of old_rect by
// compute_rebind_meta, so the source-frame worklist over old_rect fully
// covers every destination pixel that needs writing on the rebind step.
//
// Called once at fresh init AND inside each rebind cycle (after the rebind
// step + apply_meta have updated cells.rect). Resets the device counter to
// zero, then atomic-emits one WorkItem per (cell, sub-tile-in-rect).
//
// We DO NOT sync to read the count back. Subsequent k_opus_step launches
// use c.d_work_cap (worst-case count) as the grid size; dead blocks
// early-exit on the in-kernel rect check. This avoids a per-rebind stream
// drain that would otherwise serialize ~3 ms of pending step work.
//
// Slots in c.d_work beyond the current count retain stale (cell, sx, sy)
// triples from previous worklist builds. Those triples are themselves
// valid old worklist entries — the kernel safely processes or skips them
// via the rect intersection check. The atomicAdd into S only fires for
// in-rect pixels, so no double-counting.
void launch_opus_build_worklist(CellArrays& c, cudaStream_t stream)
{
    const int N = c.num_cells;
    if (N == 0) { c.workCount = 0; return; }
    OPUS_CK(cudaMemsetAsync(c.d_work_count, 0, sizeof(int), stream));
    constexpr int BS = 128;
    int blocks = (N + BS - 1) / BS;
    k_opus_build_worklist<<<blocks, BS, 0, stream>>>(
        N, c.rect, (WorkItem*)c.d_work, c.d_work_count);
    // No host-readback / sync. Subsequent step launches use d_work_cap as
    // the grid size; the kernel early-exits per CTA on blockIdx.x >= count.
    c.workCount = c.d_work_cap;
}

void launch_opus_step_rebind(CellArrays& c, const SimParams& p,
                             int parity, cudaStream_t stream)
{
    const int N = c.num_cells;
    if (N == 0 || c.workCount == 0) return;
    const int q = parity ^ 1;
    const size_t S_elems = (size_t)c.S_ext_height * p.Nx;

    // Zero next-step scatter/atomic targets.
    OPUS_CK(cudaMemsetAsync(c.S_pool [q], 0, S_elems * sizeof(float), stream));
    OPUS_CK(cudaMemsetAsync(c.V_pool [q], 0, N * sizeof(double), stream));
    OPUS_CK(cudaMemsetAsync(c.Ix_pool[q], 0, N * sizeof(double), stream));
    OPUS_CK(cudaMemsetAsync(c.Iy_pool[q], 0, N * sizeof(double), stream));

    dim3 blk(BX, BY);
    k_opus_step<false, true><<<c.workCount, blk, 0, stream>>>(
        c.phi_in, c.phi_out,
        c.S_pool[parity], c.S_pool[q],
        c.V_pool [parity], c.Ix_pool[parity], c.Iy_pool[parity],
        c.V_pool [q],      c.Ix_pool[q],      c.Iy_pool[q],
        nullptr, nullptr, nullptr, nullptr, nullptr,
        c.origin, c.rect, c.new_rect, c.shift_xy,
        (const WorkItem*)c.d_work, c.d_work_count,
        c.gamma_cell, c.v_A_cell, c.polar_x, c.polar_y, c.tgt_radius,
        (int)p.Nx, (int)p.Ny,
        c.S_y_lo, c.S_halo_h, c.S_ext_height,
        (float)p.lambda, (float)p.kappa, (float)p.mu,
        (float)p.xi, (float)p.dt);
}

// Cleanup step: an extra DO_REBIND step with shift = (0, 0) and
// src_rect == dst_rect == new_rect. Runs after the rebind step (and after
// parity flip + apply_meta) to zero out the OTHER ping-pong phi buffer's
// periphery — pixels that were inside the OLD rect (and got evolved into
// the OTHER buffer ~2 parities ago) but are now outside the new rect. Without
// this, the step after next reads stale phi at the new-rect boundary's halo.
// S is not affected (rebuilt by scatter every step from in-rect phi only).
void launch_opus_step_cleanup(CellArrays& c, const SimParams& p,
                              int parity, cudaStream_t stream)
{
    const int N = c.num_cells;
    if (N == 0 || c.workCount == 0) return;
    const int q = parity ^ 1;
    const size_t S_elems = (size_t)c.S_ext_height * p.Nx;

    OPUS_CK(cudaMemsetAsync(c.S_pool [q], 0, S_elems * sizeof(float), stream));
    OPUS_CK(cudaMemsetAsync(c.V_pool [q], 0, N * sizeof(double), stream));
    OPUS_CK(cudaMemsetAsync(c.Ix_pool[q], 0, N * sizeof(double), stream));
    OPUS_CK(cudaMemsetAsync(c.Iy_pool[q], 0, N * sizeof(double), stream));

    // Allocate-once zero-shift scratch (used as the shift_xy argument).
    static int* d_zero_shift = nullptr;
    static int  d_zero_shift_cap = 0;
    if (d_zero_shift_cap < 2 * N) {
        if (d_zero_shift) cudaFree(d_zero_shift);
        OPUS_CK(cudaMalloc(&d_zero_shift, 2 * N * sizeof(int)));
        OPUS_CK(cudaMemset(d_zero_shift, 0, 2 * N * sizeof(int)));
        d_zero_shift_cap = 2 * N;
    }

    dim3 blk(BX, BY);
    k_opus_step<false, true><<<c.workCount, blk, 0, stream>>>(
        c.phi_in, c.phi_out,
        c.S_pool[parity], c.S_pool[q],
        c.V_pool [parity], c.Ix_pool[parity], c.Iy_pool[parity],
        c.V_pool [q],      c.Ix_pool[q],      c.Iy_pool[q],
        nullptr, nullptr, nullptr, nullptr, nullptr,
        c.origin, c.rect, c.rect, d_zero_shift,   // src_rect == dst_rect == new (current) rect
        (const WorkItem*)c.d_work, c.d_work_count,
        c.gamma_cell, c.v_A_cell, c.polar_x, c.polar_y, c.tgt_radius,
        (int)p.Nx, (int)p.Ny,
        c.S_y_lo, c.S_halo_h, c.S_ext_height,
        (float)p.lambda, (float)p.kappa, (float)p.mu,
        (float)p.xi, (float)p.dt);
}
