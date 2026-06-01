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

template<bool DO_EXT>
__global__ void k_opus_step(
    const float* __restrict__ phiIn,
    float*       __restrict__ phiOut,
    const float* __restrict__ Sin,
    float*       __restrict__ Sout,
    const float* __restrict__ Vlag,
    const float* __restrict__ Ixlag,
    const float* __restrict__ Iylag,
    float*       __restrict__ Vout,
    float*       __restrict__ Ixout,
    float*       __restrict__ Iyout,
    float*       __restrict__ perim_out,
    float*       __restrict__ Cx_out,
    float*       __restrict__ Cy_out,
    float*       __restrict__ Cxx_out,
    float*       __restrict__ Cyy_out,
    const int*       __restrict__ origin,
    const int*       __restrict__ rect,
    const WorkItem*  __restrict__ work,
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

    const WorkItem wi = work[blockIdx.x];
    const int n   = wi.tile;
    const int sx  = wi.sx;
    const int sy  = wi.sy;

    const int gx0 = origin[2*n + 0];
    const int gy0 = origin[2*n + 1];
    const int rx0 = rect[4*n + 0];
    const int ry0 = rect[4*n + 1];
    const int rw  = rect[4*n + 2];
    const int rh  = rect[4*n + 3];

    const float* ph = phiIn  + (size_t)n * TILE_AREA;
    float*       po = phiOut + (size_t)n * TILE_AREA;
    const int tid   = threadIdx.y * BX + threadIdx.x;

    // 34x34 halo load. cell_sim's rect invariant guarantees
    // (sx-1..sx+OW) and (sy-1..sy+OH) sit fully inside [0, TILE_T-1].
    #pragma unroll
    for (int idx = tid; idx < HW * HH; idx += NTH) {
        int lxi = idx % HW, lyi = idx / HW;
        sm[lyi][lxi] = ph[(size_t)(sy - 1 + lyi) * TILE_T + (sx - 1 + lxi)];
    }
    __syncthreads();

    // Per-tile coefficients from LAGGED V/Ix/Iy.
    const float gam    = gamma_cell[n];
    const float vA     = vA_cell[n];
    const float R      = tgt_R_c[n];
    const float piR2   = (float)M_PI * R * R;
    const float V_lag  = Vlag[n];
    const float Ix_lag = Ixlag[n];
    const float Iy_lag = Iylag[n];
    const float dwC    = gam * bulk_coeff<float>(lambda_);
    const float repC   = interaction_coeff<float>(kappa, lambda_);
    const float volC   = (2.0f * mu / piR2) * (piR2 - V_lag);
    const float coeffV = motility_coeff<float>(kappa, xi, lambda_);
    const float vx     = coeffV * Ix_lag + vA * dirx_c[n];
    const float vy     = coeffV * Iy_lag + vA * diry_c[n];

    const int rxe = rx0 + rw, rye = ry0 + rh;
    float v=0.f, ix=0.f, iy=0.f;
    float pp=0.f, cx=0.f, cy=0.f, cxx=0.f, cyy=0.f;

    #pragma unroll
    for (int r = 0; r < RY; ++r) {
        const int oy = threadIdx.y + r * BY;
        const int lx = sx + threadIdx.x;
        const int ly = sy + oy;
        if (lx >= rxe || ly >= rye) continue;

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

        // Slab-aware S index. Single-GPU: identity in [0, Ny).
        const int gxg = wrap_g(gx0 + lx, Nx);
        const int gyg = wrap_g(gy0 + ly, Ny);
        const int syL = slab_local_y(gyg, S_y_lo, S_halo_h, S_ext_height, Ny);
        const size_t gIdx = (size_t)syL * Nx + gxg;

        const float Sg = Sin[gIdx];
        const float term = fmaxf(0.0f, Sg - c*c);

        // (a) FRESH reductions of phi_in.
        const float c2 = c * c;
        v  += c2;
        ix += c * gx * term;
        iy += c * gy * term;
        if constexpr (DO_EXT) {
            const float flx = (float)lx, fly = (float)ly;
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
        po[(size_t)ly * TILE_T + lx] = pn;
        atomicAdd(&Sout[gIdx], pn * pn);
    }

    float bv = block_sum_op(v,  red);
    float bx = block_sum_op(ix, red +   NWARP);
    float by = block_sum_op(iy, red + 2*NWARP);
    if (tid == 0) {
        atomicAdd(&Vout [n], bv);
        atomicAdd(&Ixout[n], bx);
        atomicAdd(&Iyout[n], by);
    }
    if constexpr (DO_EXT) {
        float bp  = block_sum_op(pp,  red + 3*NWARP);
        float bcx = block_sum_op(cx,  red + 4*NWARP);
        float bcy = block_sum_op(cy,  red + 5*NWARP);
        float bxx = block_sum_op(cxx, red + 6*NWARP);
        float byy = block_sum_op(cyy, red + 7*NWARP);
        if (tid == 0) {
            atomicAdd(&perim_out[n], bp);
            atomicAdd(&Cx_out [n],   bcx);
            atomicAdd(&Cy_out [n],   bcy);
            atomicAdd(&Cxx_out[n],   bxx);
            atomicAdd(&Cyy_out[n],   byy);
        }
    }
}

__global__ void k_opus_finalize_velocity(
    int N,
    const float* __restrict__ Ix, const float* __restrict__ Iy,
    const float* __restrict__ vA, const float* __restrict__ dirx,
    const float* __restrict__ diry,
    float* __restrict__ vx_out, float* __restrict__ vy_out,
    float lambda_, float kappa, float xi)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float c = motility_coeff<float>(kappa, xi, lambda_);
    vx_out[n] = c * Ix[n] + vA[n] * dirx[n];
    vy_out[n] = c * Iy[n] + vA[n] * diry[n];
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
    const float* __restrict__ V,
    const float* __restrict__ Cx,
    const float* __restrict__ Cy,
    const float* __restrict__ Cxx,
    const float* __restrict__ Cyy,
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

    float Vn   = V[n];
    float invV = (Vn > 1e-6f) ? 1.0f / Vn : 0.0f;
    float mx   = Cx[n] * invV;
    float my   = Cy[n] * invV;
    int   sx   = __float2int_rn(mx) - Th;
    int   sy   = __float2int_rn(my) - Th;
    shift_xy[2*n + 0] = sx;
    shift_xy[2*n + 1] = sy;

    float varx = Cxx[n] * invV - mx * mx;
    float vary = Cyy[n] * invV - my * my;
    if (varx < 0.0f) varx = 0.0f;
    if (vary < 0.0f) vary = 0.0f;
    float sigx = sqrtf(varx);
    float sigy = sqrtf(vary);

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
    new_rect[4*n + 0] = Th - hwx;
    new_rect[4*n + 1] = Th - hwy;
    new_rect[4*n + 2] = 2 * hwx;
    new_rect[4*n + 3] = 2 * hwy;
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
// k_opus_step_rebind — fused rebind+evolve step kernel.
//
// Sibling of k_opus_step but with two extra behaviors:
//
//   * Each pixel processed has a SOURCE-frame address (sx, sy) (the
//     work-list sub-tile origin) and the kernel reads phi/halo at that
//     source-frame location. It computes RHS as usual.
//
//   * The WRITE goes to a DESTINATION-frame address (sx - shift_x,
//     sy - shift_y). If the destination is inside new_rect, phi_new is
//     written and phi_new^2 is scattered to S at the (source-frame) global
//     address. If the destination is OUTSIDE new_rect (periphery being
//     trimmed), 0 is written and no scatter happens.
//
// Why the scatter address can stay source-frame: origin is unchanged by
// this kernel (host applies origin += shift afterward), so the global
// address of pixel (sx, sy) computed using the CURRENT origin equals the
// global address of pixel (sx - shift_x, sy - shift_y) computed using the
// POST-rebind origin (= current origin + shift). Both expressions name
// the same physical pixel.
//
// Worklist coverage: the host builds a union worklist covering the source-
// frame bounding box of (old_rect U new_rect_shifted_back). This ensures
// every periphery destination pixel that needs zeroing is reached, and
// every new-rect destination pixel that needs evolving is reached.
//
// Fresh moments (V/Ix/Iy) are accumulated over source-frame pixels — they
// characterize phi_in and are invariant under the tile-local shift.
// ---------------------------------------------------------------------------
__global__ void k_opus_step_rebind(
    const float* __restrict__ phiIn,
    float*       __restrict__ phiOut,
    const float* __restrict__ Sin,
    float*       __restrict__ Sout,
    const float* __restrict__ Vlag,
    const float* __restrict__ Ixlag,
    const float* __restrict__ Iylag,
    float*       __restrict__ Vout,
    float*       __restrict__ Ixout,
    float*       __restrict__ Iyout,
    const int*       __restrict__ origin,
    const int*       __restrict__ rect,
    const int*       __restrict__ shift_xy,
    const int*       __restrict__ new_rect_arr,
    const WorkItem*  __restrict__ work,
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
    __shared__ float red[NWARP * 3];

    const WorkItem wi = work[blockIdx.x];
    const int n   = wi.tile;
    const int sx  = wi.sx;  // source-frame sub-tile origin
    const int sy  = wi.sy;

    const int gx0  = origin[2*n + 0];
    const int gy0  = origin[2*n + 1];
    const int rx0  = rect[4*n + 0];
    const int ry0  = rect[4*n + 1];
    const int rw   = rect[4*n + 2];
    const int rh   = rect[4*n + 3];
    const int sh_x = shift_xy[2*n + 0];
    const int sh_y = shift_xy[2*n + 1];
    const int nrx0 = new_rect_arr[4*n + 0];
    const int nry0 = new_rect_arr[4*n + 1];
    const int nrw  = new_rect_arr[4*n + 2];
    const int nrh  = new_rect_arr[4*n + 3];

    const float* ph = phiIn  + (size_t)n * TILE_AREA;
    float*       po = phiOut + (size_t)n * TILE_AREA;
    const int tid   = threadIdx.y * BX + threadIdx.x;

    // Halo load at source-frame (sx, sy). The halo extent (sx-1..sx+OW,
    // sy-1..sy+OH) must lie inside [0, TILE_T-1]. The work-list builder
    // is responsible for not emitting sub-tiles whose halo would bust the
    // tile edge; the rect invariant + shift bound guarantees this for any
    // sub-tile that intersects the old rect. For sub-tiles that lie
    // entirely outside the old rect (purely periphery in the dest frame),
    // we still load the halo but every pixel will be 0 by invariant.
    #pragma unroll
    for (int idx = tid; idx < HW * HH; idx += NTH) {
        int lxi = idx % HW, lyi = idx / HW;
        int yi  = sy - 1 + lyi;
        int xi  = sx - 1 + lxi;
        // Defensive guard against worklist edge cases; never trips for valid lists.
        float v = 0.0f;
        if ((unsigned)xi < (unsigned)TILE_T && (unsigned)yi < (unsigned)TILE_T) {
            v = ph[(size_t)yi * TILE_T + xi];
        }
        sm[lyi][lxi] = v;
    }
    __syncthreads();

    // Lagged per-tile coefficients (identical to non-rebind step).
    const float gam    = gamma_cell[n];
    const float vA     = vA_cell[n];
    const float R      = tgt_R_c[n];
    const float piR2   = (float)M_PI * R * R;
    const float V_lag  = Vlag[n];
    const float Ix_lag = Ixlag[n];
    const float Iy_lag = Iylag[n];
    const float dwC    = gam * bulk_coeff<float>(lambda_);
    const float repC   = interaction_coeff<float>(kappa, lambda_);
    const float volC   = (2.0f * mu / piR2) * (piR2 - V_lag);
    const float coeffV = motility_coeff<float>(kappa, xi, lambda_);
    const float vx     = coeffV * Ix_lag + vA * dirx_c[n];
    const float vy     = coeffV * Iy_lag + vA * diry_c[n];

    const int rxe  = rx0  + rw;
    const int rye  = ry0  + rh;
    const int nrxe = nrx0 + nrw;
    const int nrye = nry0 + nrh;

    float v_acc=0.f, ix_acc=0.f, iy_acc=0.f;

    #pragma unroll
    for (int r = 0; r < RY; ++r) {
        const int oy = threadIdx.y + r * BY;
        const int lx = sx + threadIdx.x;        // source-frame
        const int ly = sy + oy;

        // Destination-frame coordinates.
        const int out_lx = lx - sh_x;
        const int out_ly = ly - sh_y;

        // Skip if destination is outside the tile entirely (shouldn't happen
        // under invariants, defensive guard).
        if ((unsigned)out_lx >= (unsigned)TILE_T) continue;
        if ((unsigned)out_ly >= (unsigned)TILE_T) continue;

        const bool dst_in_new = (out_lx >= nrx0 && out_lx < nrxe &&
                                 out_ly >= nry0 && out_ly < nrye);
        const bool src_in_old = (lx >= rx0 && lx < rxe &&
                                 ly >= ry0 && ly < rye);

        if (!dst_in_new) {
            // Periphery: zero the destination pixel. No scatter, no moments.
            // (We still need to write 0 because phi_out's previous content
            // may be stale from two parities ago.)
            po[(size_t)out_ly * TILE_T + out_lx] = 0.0f;
            continue;
        }

        if (!src_in_old) {
            // Destination inside new rect but source outside old rect (the
            // "rect grew" case). Source phi was 0; phi_new from RHS at
            // c=0 with all-0 halo is 0.
            po[(size_t)out_ly * TILE_T + out_lx] = 0.0f;
            continue;
        }

        // Standard path: source inside old rect, destination inside new rect.
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

        const float lap = (4.0f*(e+w+nN+sS) + (ne+nw+se+sw) - 20.0f*c) * (1.0f/6.0f);
        const float gx = 0.5f * (e  - w);
        const float gy = 0.5f * (nN - sS);

        const int gxg = wrap_g(gx0 + lx, Nx);
        const int gyg = wrap_g(gy0 + ly, Ny);
        const int syL = slab_local_y(gyg, S_y_lo, S_halo_h, S_ext_height, Ny);
        const size_t gIdx = (size_t)syL * Nx + gxg;

        const float Sg = Sin[gIdx];
        const float term = fmaxf(0.0f, Sg - c*c);

        // Fresh moments — source-frame values (physical, invariant under shift).
        const float c2 = c * c;
        v_acc  += c2;
        ix_acc += c * gx * term;
        iy_acc += c * gy * term;

        const float dw  = c * (1.0f - c) * (1.0f - 2.0f * c);
        const float rhs = gam * lap - dwC * dw + volC * c - repC * c * term;
        const float adv = vx * gx + vy * gy;
        const float pn  = c + dt * (rhs - adv);

        // Write at DESTINATION-frame address; scatter S at source-frame
        // global address (== destination-frame global address with the
        // post-rebind origin, since origin += shift after this kernel).
        po[(size_t)out_ly * TILE_T + out_lx] = pn;
        atomicAdd(&Sout[gIdx], pn * pn);
    }

    float bv = block_sum_op(v_acc,  red);
    float bx = block_sum_op(ix_acc, red +   NWARP);
    float by = block_sum_op(iy_acc, red + 2*NWARP);
    if (tid == 0) {
        atomicAdd(&Vout [n], bv);
        atomicAdd(&Ixout[n], bx);
        atomicAdd(&Iyout[n], by);
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
    OPUS_CK(cudaMemsetAsync(c.V_pool [q], 0, N * sizeof(float), stream));
    OPUS_CK(cudaMemsetAsync(c.Ix_pool[q], 0, N * sizeof(float), stream));
    OPUS_CK(cudaMemsetAsync(c.Iy_pool[q], 0, N * sizeof(float), stream));
    if (need_full) {
        OPUS_CK(cudaMemsetAsync(c.perimeters, 0, N * sizeof(float), stream));
        OPUS_CK(cudaMemsetAsync(c.Cx,         0, N * sizeof(float), stream));
        OPUS_CK(cudaMemsetAsync(c.Cy,         0, N * sizeof(float), stream));
        OPUS_CK(cudaMemsetAsync(c.Cxx,        0, N * sizeof(float), stream));
        OPUS_CK(cudaMemsetAsync(c.Cyy,        0, N * sizeof(float), stream));
    }

    dim3 blk(BX, BY);
    if (need_full) {
        k_opus_step<true><<<c.workCount, blk, 0, stream>>>(
            c.phi_in, c.phi_out,
            c.S_pool[parity], c.S_pool[q],
            c.V_pool [parity], c.Ix_pool[parity], c.Iy_pool[parity],
            c.V_pool [q],      c.Ix_pool[q],      c.Iy_pool[q],
            c.perimeters, c.Cx, c.Cy, c.Cxx, c.Cyy,
            c.origin, c.rect, (const WorkItem*)c.d_work,
            c.gamma_cell, c.v_A_cell, c.polar_x, c.polar_y, c.tgt_radius,
            (int)p.Nx, (int)p.Ny,
            c.S_y_lo, c.S_halo_h, c.S_ext_height,
            (float)p.lambda, (float)p.kappa, (float)p.mu,
            (float)p.xi, (float)p.dt);
    } else {
        k_opus_step<false><<<c.workCount, blk, 0, stream>>>(
            c.phi_in, c.phi_out,
            c.S_pool[parity], c.S_pool[q],
            c.V_pool [parity], c.Ix_pool[parity], c.Iy_pool[parity],
            c.V_pool [q],      c.Ix_pool[q],      c.Iy_pool[q],
            nullptr, nullptr, nullptr, nullptr, nullptr,
            c.origin, c.rect, (const WorkItem*)c.d_work,
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
    OPUS_CK(cudaMemcpyAsync(c.V_pool [t], c.V_pool [f], N*sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
    OPUS_CK(cudaMemcpyAsync(c.Ix_pool[t], c.Ix_pool[f], N*sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
    OPUS_CK(cudaMemcpyAsync(c.Iy_pool[t], c.Iy_pool[f], N*sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
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

// Build a per-cell source-frame union worklist: bounding box of
// (old_rect U new_rect_shifted_back_to_source_frame) where
// new_rect_shifted_back = (nrx0 + sx, nry0 + sy, nrw, nrh).
//
// This is called between launch_opus_compute_rebind_meta and
// launch_opus_step_rebind on rebind cadence. It syncs the stream
// because it must read the just-computed shift_xy and new_rect.
int build_opus_work_list_for_rebind(CellArrays& c)
{
    const int N = c.num_cells;
    if (N == 0) { c.workCount = 0; return 0; }

    // Wait for compute_rebind_meta to finish on step_stream before reading.
    OPUS_CK(cudaDeviceSynchronize());

    std::vector<int> h_rect(4 * N);
    std::vector<int> h_new (4 * N);
    std::vector<int> h_shf (2 * N);
    OPUS_CK(cudaMemcpy(h_rect.data(), c.rect,     4 * N * sizeof(int),
                       cudaMemcpyDeviceToHost));
    OPUS_CK(cudaMemcpy(h_new.data(),  c.new_rect, 4 * N * sizeof(int),
                       cudaMemcpyDeviceToHost));
    OPUS_CK(cudaMemcpy(h_shf.data(),  c.shift_xy, 2 * N * sizeof(int),
                       cudaMemcpyDeviceToHost));

    std::vector<WorkItem> h_work;
    // Worst case is ~2x the regular worklist per cell on rebind.
    h_work.reserve(N * 2 * OPUS_MAX_WORKITEMS_PER_CELL);

    for (int n = 0; n < N; ++n) {
        const int orx0 = h_rect[4*n + 0];
        const int ory0 = h_rect[4*n + 1];
        const int orw  = h_rect[4*n + 2];
        const int orh  = h_rect[4*n + 3];
        const int nrx0 = h_new [4*n + 0];
        const int nry0 = h_new [4*n + 1];
        const int nrw  = h_new [4*n + 2];
        const int nrh  = h_new [4*n + 3];
        const int sx   = h_shf [2*n + 0];
        const int sy   = h_shf [2*n + 1];

        // new_rect shifted back into source frame.
        const int srx0 = nrx0 + sx;
        const int sry0 = nry0 + sy;

        // Source-frame bounding box of (old_rect U shifted_new_rect).
        int x_lo = orx0  < srx0  ? orx0  : srx0;
        int y_lo = ory0  < sry0  ? ory0  : sry0;
        int x_hi = orx0 + orw  > srx0 + nrw  ? orx0 + orw  : srx0 + nrw;
        int y_hi = ory0 + orh  > sry0 + nrh  ? ory0 + orh  : sry0 + nrh;

        // Clamp to the safe halo-load region [1, TILE_T-1) on both axes —
        // the halo at (sx-1, sy-1) and (sx+OW, sy+OH) must stay inside the
        // tile. Anything that gets clipped here was outside both rects on
        // that side anyway, so nothing to do at those pixels.
        if (x_lo < 1)            x_lo = 1;
        if (y_lo < 1)            y_lo = 1;
        if (x_hi > TILE_T - 1)   x_hi = TILE_T - 1;
        if (y_hi > TILE_T - 1)   y_hi = TILE_T - 1;
        if (x_hi <= x_lo || y_hi <= y_lo) continue;

        for (int wsy = y_lo; wsy < y_hi; wsy += OH)
            for (int wsx = x_lo; wsx < x_hi; wsx += OW)
                h_work.push_back({n, wsx, wsy});
    }
    const int wc = (int)h_work.size();
    if (wc > c.d_work_cap) {
        fprintf(stderr,
            "[opus] rebind worklist (%d) exceeds capacity (%d). "
            "Increase d_work_cap in alloc_gpu.\n", wc, c.d_work_cap);
        std::exit(1);
    }
    OPUS_CK(cudaMemcpy((WorkItem*)c.d_work, h_work.data(),
                       wc * sizeof(WorkItem), cudaMemcpyHostToDevice));
    c.workCount = wc;
    return wc;
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
    OPUS_CK(cudaMemsetAsync(c.V_pool [q], 0, N * sizeof(float), stream));
    OPUS_CK(cudaMemsetAsync(c.Ix_pool[q], 0, N * sizeof(float), stream));
    OPUS_CK(cudaMemsetAsync(c.Iy_pool[q], 0, N * sizeof(float), stream));

    dim3 blk(BX, BY);
    k_opus_step_rebind<<<c.workCount, blk, 0, stream>>>(
        c.phi_in, c.phi_out,
        c.S_pool[parity], c.S_pool[q],
        c.V_pool [parity], c.Ix_pool[parity], c.Iy_pool[parity],
        c.V_pool [q],      c.Ix_pool[q],      c.Iy_pool[q],
        c.origin, c.rect, c.shift_xy, c.new_rect,
        (const WorkItem*)c.d_work,
        c.gamma_cell, c.v_A_cell, c.polar_x, c.polar_y, c.tgt_radius,
        (int)p.Nx, (int)p.Ny,
        c.S_y_lo, c.S_halo_h, c.S_ext_height,
        (float)p.lambda, (float)p.kappa, (float)p.mu,
        (float)p.xi, (float)p.dt);
}
