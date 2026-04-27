// kernels.cu — All CUDA kernels, written from scratch per PLAN.md
//
// Layout:
//   1. Device helpers
//   2. Spatial hash (insert + query)
//   3. Pre-step (ref pts, shifts, bbox resize)
//   4. Fused 1-CTA-per-cell PDE kernel
//   5. Swap kernel
//   6. Polarization update
//   7. RNG init
//   8. Initial reductions (centroid + volume)
//   9. Initial velocity integral

#include "kernels.cuh"
#include <curand_kernel.h>
#include <cstdio>

// ===== 1. Device helpers =====================================================

__device__ __forceinline__ int wrap(int x, int N) {
    return ((x % N) + N) % N;
}

__device__ __forceinline__ float pdelta(float d, float L) {
    if (d >  L * 0.5f) d -= L;
    if (d < -L * 0.5f) d += L;
    return d;
}

// ===== 2. Spatial hash =======================================================

__global__ void k_hash_insert(
    const int* __restrict__ ox, const int* __restrict__ oy,
    const int* __restrict__ w,  const int* __restrict__ h,
    int* __restrict__ ids, int* __restrict__ counts,
    int bsz, int gnx, int gny, int Nx, int Ny, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cx = wrap(ox[i] + w[i] / 2, Nx);
    int cy = wrap(oy[i] + h[i] / 2, Ny);
    int bx = min(cx / bsz, gnx - 1);
    int by = min(cy / bsz, gny - 1);
    int bin = by * gnx + bx;
    int slot = atomicAdd(&counts[bin], 1);
    if (slot < HASH_MAX_PER_BIN)
        ids[bin * HASH_MAX_PER_BIN + slot] = i;
    else
        printf("[HASH OVERFLOW] cell %d bin %d slot %d >= %d\n",
               i, bin, slot, HASH_MAX_PER_BIN);
}

__global__ void k_hash_query(
    const int* __restrict__ ox, const int* __restrict__ oy,
    const int* __restrict__ w,  const int* __restrict__ h,
    const int* __restrict__ ids, const int* __restrict__ counts,
    NeighborEntry* __restrict__ nlist, int* __restrict__ ncnt,
    int bsz, int gnx, int gny, int Nx, int Ny, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int oxi = ox[i], oyi = oy[i], wi = w[i], hi = h[i];
    int cx = wrap(oxi + wi / 2, Nx);
    int cy = wrap(oyi + hi / 2, Ny);
    int bx = min(cx / bsz, gnx - 1);
    int by = min(cy / bsz, gny - 1);
    int cnt = 0;

    // Neighbor rule: bbox-overlap <=> neighbor. Outside a cell's bbox there
    // is no phi (no memory allocated), so a non-overlapping pair cannot
    // contribute to the interaction sum. No margin needed.

    // 3x3 bin search with dedup for small grids
    int visited[9]; int nv = 0;
    for (int dby = -1; dby <= 1; dby++) {
        for (int dbx = -1; dbx <= 1; dbx++) {
            int nbx = (bx + dbx + gnx) % gnx;
            int nby = (by + dby + gny) % gny;
            int bin = nby * gnx + nbx;
            bool dup = false;
            for (int v = 0; v < nv; v++) if (visited[v] == bin) { dup = true; break; }
            if (dup) continue;
            visited[nv++] = bin;

            int bc = min(counts[bin], HASH_MAX_PER_BIN);
            for (int s = 0; s < bc && cnt < K_MAX; s++) {
                int j = ids[bin * HASH_MAX_PER_BIN + s];
                if (j == i) continue;
                int dx = ox[j] - oxi;
                int dy = oy[j] - oyi;
                if (dx >  Nx / 2) dx -= Nx;
                if (dx < -Nx / 2) dx += Nx;
                if (dy >  Ny / 2) dy -= Ny;
                if (dy < -Ny / 2) dy += Ny;
                int wj = w[j], hj = h[j];
                // AABB overlap: inner regions intersect iff the outer boxes
                // do (halo is a subset of either box's extent).
                if ((dx + wj > 0) && (dx < wi) &&
                    (dy + hj > 0) && (dy < hi)) {
                    nlist[i * K_MAX + cnt].cell_id = j;
                    cnt++;
                }
            }
        }
    }
    ncnt[i] = cnt;
}

void launch_hash_build(CellArrays& c, int Nx, int Ny) {
    int n = c.num_cells;
    if (n <= 1) return;

    // Size bins to fit the largest cell. +1 slack so a cell exactly at a bin
    // boundary can't straddle more than the 3x3 neighborhood we search.
    int req = c.max_side + 1;
    if (c.hash_bin_sz == 0 || req > c.hash_bin_sz) {
        if (c.hash_ids) cudaFree(c.hash_ids);
        if (c.hash_counts) cudaFree(c.hash_counts);
        c.hash_bin_sz = req;
        c.hash_nx = (Nx + req - 1) / req;
        c.hash_ny = (Ny + req - 1) / req;
        int nb = c.hash_nx * c.hash_ny;
        cudaMalloc(&c.hash_ids, nb * HASH_MAX_PER_BIN * sizeof(int));
        cudaMalloc(&c.hash_counts, nb * sizeof(int));
    }
    int nb = c.hash_nx * c.hash_ny;
    cudaMemset(c.hash_counts, 0, nb * sizeof(int));

    int blk = (n + 255) / 256;
    k_hash_insert<<<blk, 256>>>(
        c.offsets_x, c.offsets_y, c.widths, c.heights,
        c.hash_ids, c.hash_counts,
        c.hash_bin_sz, c.hash_nx, c.hash_ny, Nx, Ny, n);
    k_hash_query<<<blk, 256>>>(
        c.offsets_x, c.offsets_y, c.widths, c.heights,
        c.hash_ids, c.hash_counts,
        c.nbr_list, c.nbr_count,
        c.hash_bin_sz, c.hash_nx, c.hash_ny, Nx, Ny, n);
}

// ===== 3. Pre-step ===========================================================

__global__ void k_pre_step(
    float* __restrict__ rx, float* __restrict__ ry,
    int* __restrict__ ox, int* __restrict__ oy,
    int* __restrict__ w, int* __restrict__ h,
    int* __restrict__ ow, int* __restrict__ oh,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ vol,
    float* __restrict__ mx, float* __restrict__ my,
    const float* __restrict__ tgt_r,
    int* __restrict__ sx, int* __restrict__ sy,
    int* __restrict__ d_maxwh,
    bool do_resize, bool zero_mom, int max_side,
    float sub_pad, float dA, int Nx, int Ny, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int wi = w[i], hi = h[i];
    ow[i] = wi; oh[i] = hi;

    // Ref point = bbox center
    float refx = fmodf(fmodf((float)ox[i] + wi * 0.5f, (float)Nx) + Nx, (float)Nx);
    float refy = fmodf(fmodf((float)oy[i] + hi * 0.5f, (float)Ny) + Ny, (float)Ny);

    int shx = 0, shy = 0;
    int nw = wi, nh = hi;

    float v = vol[i];
    if (v > 1e-8f && do_resize) {
        float cxi = cx[i], cyi = cy[i];

        // Centroid shift – recenter if drift > 2 pixels
        float scx = fmodf((float)ox[i] + wi * 0.5f, (float)Nx);
        float scy = fmodf((float)oy[i] + hi * 0.5f, (float)Ny);
        float dx = cxi - scx, dy = cyi - scy;
        if (dx >  Nx * 0.5f) dx -= Nx;
        if (dx < -Nx * 0.5f) dx += Nx;
        if (dy >  Ny * 0.5f) dy -= Ny;
        if (dy < -Ny * 0.5f) dy += Ny;
        int csx = (int)roundf(dx), csy = (int)roundf(dy);
        if (abs(csx) > 2) shx = csx;
        if (abs(csy) > 2) shy = csy;

        // Variance-based resize (parallel axis theorem)
        float sp2 = v / dA;
        float Mx = mx[i], My = my[i];
        if (max_side > 0 && sp2 > 1.0f) {
            float dc_x = cxi - refx, dc_y = cyi - refy;
            if (dc_x >  Nx * 0.5f) dc_x -= Nx;
            if (dc_x < -Nx * 0.5f) dc_x += Nx;
            if (dc_y >  Ny * 0.5f) dc_y -= Ny;
            if (dc_y < -Ny * 0.5f) dc_y += Ny;
            float vx = Mx / sp2 - dc_x * dc_x;
            float vy = My / sp2 - dc_y * dc_y;
            if (vx > 4.0f && vy > 4.0f) {
                float R = tgt_r[i];
                int pad = (int)ceilf(sub_pad * R);
                int tw = (2 * ((int)ceilf(2.0f * sqrtf(vx)) + pad)) & ~1;
                int th = (2 * ((int)ceilf(2.0f * sqrtf(vy)) + pad)) & ~1;
                tw = min(max(tw, 32), max_side);
                th = min(max(th, 32), max_side);
                if (tw != wi) { nw = tw; shx -= (tw - wi) / 2; }
                if (th != hi) { nh = th; shy -= (th - hi) / 2; }
            }
        }
    }

    if (zero_mom) { mx[i] = 0.0f; my[i] = 0.0f; }

    w[i] = nw; h[i] = nh;
    sx[i] = shx; sy[i] = shy;
    rx[i] = refx; ry[i] = refy;
    if (do_resize) {
        atomicMax(&d_maxwh[0], nw);
        atomicMax(&d_maxwh[1], nh);
    }
}

void launch_pre_step(CellArrays& c, const SimParams& p, int step,
                     int& cache_w, int& cache_h) {
    int n = c.num_cells;
    if (n == 0) return;
    bool resize = (step % 10 == 0);
    bool zmom   = (step % 10 == 9);
    if (resize) cudaMemset(c.d_max_wh, 0, 2 * sizeof(int));

    k_pre_step<<<(n + 255) / 256, 256>>>(
        c.ref_x, c.ref_y, c.offsets_x, c.offsets_y,
        c.widths, c.heights, c.old_widths, c.old_heights,
        c.centroids_x, c.centroids_y, c.volumes,
        c.moment_x, c.moment_y, c.tgt_radius,
        c.shift_x, c.shift_y, c.d_max_wh,
        resize, zmom, c.max_side,
        p.subdomain_padding, p.dA(), p.Nx, p.Ny, n);

    if (resize) {
        int hm[2];
        cudaMemcpy(hm, c.d_max_wh, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        if (hm[0] > 0) cache_w = hm[0];
        if (hm[1] > 0) cache_h = hm[1];
    }
}

// ===== 4. Fused 1-CTA-per-cell kernel ========================================

template <bool REMAP, bool MOMENTS>
__global__ void __launch_bounds__(256, 4)
k_fused(
    float** __restrict__ phi_in,
    float** __restrict__ phi_out,
    const int* __restrict__ W,  const int* __restrict__ H,
    const int* __restrict__ OX, const int* __restrict__ OY,
    const int* __restrict__ OW, const int* __restrict__ OH,
    const int* __restrict__ SX, const int* __restrict__ SY,
    const NeighborEntry* __restrict__ nlist,
    const int* __restrict__ ncnt,
    // vx_in/vy_in/vdev are in/out (read by every thread, written by thread 0
    // after __syncthreads). No __restrict__ because the output aliases the input.
    float* vx_in, float* vy_in,
    float* vdev,
    const float* __restrict__ RX, const float* __restrict__ RY,
    const float* __restrict__ d_tg,  const float* __restrict__ d_tgb,
    const float* __restrict__ d_vc,  const float* __restrict__ d_ta,
    const float* __restrict__ d_vA,
    const float* __restrict__ d_px,  const float* __restrict__ d_py,
    float* __restrict__ d_vol,  float* __restrict__ d_vdev,
    float* __restrict__ d_cx,   float* __restrict__ d_cy,
    float* __restrict__ d_peri,
    float* __restrict__ d_mx,   float* __restrict__ d_my,
    float two_keff, float inv_h2, float inv_2dx, float inv_2dy,
    float dt, float dA, float mc,
    int halo, int Nx, int Ny, int Ncells)
{
    int ci = blockIdx.x;
    if (ci >= Ncells) return;
    const int tid = threadIdx.x;
    const int BS = 256;

    // Shared: neighbor metadata persists through both phases; reduction buffers after it
    extern __shared__ char smem[];
    float** s_phi = (float**)smem;                         // K_MAX ptrs
    int*    s_meta = (int*)(smem + K_MAX * sizeof(float*)); // K_MAX*4 ints
    // Reduction scratch starts after neighbor metadata
    float*  s_reduce = (float*)(smem + K_MAX * sizeof(float*) + K_MAX * 4 * sizeof(int));

    // Per-cell constants
    int width = W[ci], height = H[ci];
    const float* phi  = phi_in[ci];
    float*       pout = phi_out[ci];
    int oxi = OX[ci], oyi = OY[ci];
    float vd = vdev[ci];
    float rx = RX[ci], ry = RY[ci];
    float tg = d_tg[ci], tgb = d_tgb[ci], vc = d_vc[ci];

    int old_w, old_h, sx, sy;
    if constexpr (REMAP) {
        old_w = OW[ci]; old_h = OH[ci]; sx = SX[ci]; sy = SY[ci];
    } else {
        old_w = width; old_h = height; sx = 0; sy = 0;
    }

    // Load neighbor metadata into shared memory.
    // We fetch OX/OY/OW/OH of each neighbor from the CURRENT global arrays
    // so the periodic delta is always the "now" delta — never stale from a
    // rebuild 9 steps ago. Note: fused runs before swap, so OX/OY still hold
    // the pre-swap (old-buffer) origins, which is the correct frame for the
    // phi_in reads below. OW/OH hold each cell's own old-buffer dimensions.
    int k = ncnt[ci];
    int oxi_pre = OX[ci], oyi_pre = OY[ci];
    for (int ni = tid; ni < k; ni += BS) {
        int nid = nlist[ci * K_MAX + ni].cell_id;
        s_phi[ni] = phi_in[nid];
        int dx = OX[nid] - oxi_pre;
        int dy = OY[nid] - oyi_pre;
        if (dx >  Nx / 2) dx -= Nx;
        if (dx < -Nx / 2) dx += Nx;
        if (dy >  Ny / 2) dy -= Ny;
        if (dy < -Ny / 2) dy += Ny;
        s_meta[ni * 4 + 0] = dx;
        s_meta[ni * 4 + 1] = dy;
        s_meta[ni * 4 + 2] = OW[nid];
        s_meta[ni * 4 + 3] = OH[nid];
    }
    __syncthreads();

    int total = width * height;
    int stride_dy = BS / width;
    int stride_dx = BS - stride_dy * width;

    // ====== PHASE 1: Heavy pass — stencil, neighbors, velocity integral ======
    // Computes var_deriv, gx, gy per pixel and stores them in phi_out as scratch.
    // Also accumulates velocity integral for reduction.
    {
        float a_vix = 0, a_viy = 0;
        int lx = tid % width, ly = tid / width;
        for (int flat = tid; flat < total; flat += BS) {
            int srx = lx + sx, sry = ly + sy;
            bool src_ok;
            if constexpr (REMAP) {
                src_ok = (srx >= 0 && srx < old_w && sry >= 0 && sry < old_h);
            } else { src_ok = true; }
            float pv = src_ok ? __ldg(&phi[sry * old_w + srx]) : 0.0f;
            bool inner = (lx >= halo && lx < width - halo &&
                          ly >= halo && ly < height - halo);

            float var_deriv = 0.0f, gx = 0.0f, gy = 0.0f;
            if (src_ok && inner) {
                int xm = max(srx-1, 0), xp = min(srx+1, old_w-1);
                int ym = max(sry-1, 0), yp = min(sry+1, old_h-1);
                float pE  = __ldg(&phi[sry * old_w + xp]);
                float pW  = __ldg(&phi[sry * old_w + xm]);
                float pN  = __ldg(&phi[yp  * old_w + srx]);
                float pS  = __ldg(&phi[ym  * old_w + srx]);
                float pNE = __ldg(&phi[yp  * old_w + xp]);
                float pNW = __ldg(&phi[yp  * old_w + xm]);
                float pSE = __ldg(&phi[ym  * old_w + xp]);
                float pSW = __ldg(&phi[ym  * old_w + xm]);
                float lap = (4.0f*(pE+pW+pN+pS) + (pNE+pNW+pSE+pSW) - 20.0f*pv) * inv_h2 / 6.0f;
                gx = (pE - pW) * inv_2dx;
                gy = (pN - pS) * inv_2dy;
                float S = 0.0f;
                for (int ni = 0; ni < k; ni++) {
                    int nlx = srx - s_meta[ni*4], nly = sry - s_meta[ni*4+1];
                    int nw = s_meta[ni*4+2], nh = s_meta[ni*4+3];
                    if (nlx >= halo && nlx < nw-halo && nly >= halo && nly < nh-halo) {
                        float pm = __ldg(&s_phi[ni][nly * nw + nlx]);
                        S += pm * pm;
                    }
                }
                float bulk       = tgb * pv * (1.0f - pv) * (1.0f - 2.0f * pv);
                float constraint = -4.0f * vc * vd * pv;
                float repulsion  = two_keff * pv * S;
                var_deriv = -tg * lap + bulk + constraint + repulsion;
                a_vix += pv * gx * S;
                a_viy += pv * gy * S;
            }
            // Store scratch: var_deriv only. gx/gy will be recomputed cheaply in Phase 2.
            int pidx = ly * width + lx;
            pout[pidx] = var_deriv;

            lx += stride_dx; int wrap = lx >= width; lx -= wrap * width; ly += stride_dy + wrap;
        }
        // Reduce velocity integral (2 channels)
        __syncthreads();
        float* sr = s_reduce;
        sr[tid] = a_vix; sr[tid + BS] = a_viy;
        __syncthreads();
        for (int s = BS/2; s > 32; s >>= 1) {
            if (tid < s) { sr[tid] += sr[tid+s]; sr[tid+BS] += sr[tid+BS+s]; }
            __syncthreads();
        }
        if (tid < 32) {
            float v0 = sr[tid]+sr[tid+32], v1 = sr[tid+BS]+sr[tid+BS+32];
            for (int off = 16; off > 0; off >>= 1) {
                v0 += __shfl_down_sync(0xffffffff, v0, off);
                v1 += __shfl_down_sync(0xffffffff, v1, off);
            }
            if (tid == 0) {
                vx_in[ci] = mc * v0 * dA + d_vA[ci] * d_px[ci];
                vy_in[ci] = mc * v1 * dA + d_vA[ci] * d_py[ci];
            }
        }
    }
    __syncthreads();
    float vx = vx_in[ci], vy = vy_in[ci];

    // ====== PHASE 2: Lightweight pass — advection + phi write ======
    // Reads scratch (var_deriv, gx, gy) from phi_out, applies advection with
    // the freshly-computed velocity, writes final phi_new. No stencil, no
    // neighbor reads — just memory + arithmetic.
    {
        float a_cdx = 0, a_cdy = 0, a_p2 = 0, a_grd = 0;
        float a_mx = 0, a_my = 0;
        int lx = tid % width, ly = tid / width;
        for (int flat = tid; flat < total; flat += BS) {
            int srx = lx + sx, sry = ly + sy;
            bool src_ok;
            if constexpr (REMAP) {
                src_ok = (srx >= 0 && srx < old_w && sry >= 0 && sry < old_h);
            } else { src_ok = true; }
            float pv = src_ok ? __ldg(&phi[sry * old_w + srx]) : 0.0f;
            bool inner = (lx >= halo && lx < width - halo &&
                          ly >= halo && ly < height - halo);
            float np = pv;
            int pidx = ly * width + lx;
            if (src_ok && inner) {
                float var_deriv = pout[pidx];
                // Recompute gradient cheaply from phi_in (4 reads, L1 cache hot from Phase 1)
                float gx = (__ldg(&phi[sry * old_w + min(srx+1, old_w-1)])
                          - __ldg(&phi[sry * old_w + max(srx-1, 0)])) * inv_2dx;
                float gy = (__ldg(&phi[min(sry+1, old_h-1) * old_w + srx])
                          - __ldg(&phi[max(sry-1, 0) * old_w + srx])) * inv_2dy;
                float advection  = vx * gx + vy * gy;
                np = pv + dt * (-0.5f * var_deriv - advection);
                a_grd += sqrtf(gx * gx + gy * gy);
                float np2 = np * np;
                float gxf = (float)(oxi + lx + sx);
                float gyf = (float)(oyi + ly + sy);
                float drx = pdelta(gxf - rx, (float)Nx);
                float dry = pdelta(gyf - ry, (float)Ny);
                a_cdx += drx * np2; a_cdy += dry * np2; a_p2 += np2;
                if constexpr (MOMENTS) {
                    a_mx += drx * drx * np2;
                    a_my += dry * dry * np2;
                }
            }
            if (!inner) np = 0.0f;
            pout[pidx] = np;
            lx += stride_dx; int wrap = lx >= width; lx -= wrap * width; ly += stride_dy + wrap;
        }

        // ---- Block reduction: 4 channels (centroid + volume + perimeter) ----
        __syncthreads();
        float* sr = s_reduce;
        float *r0 = sr, *r1 = sr + BS, *r2 = sr + 2*BS, *r3 = sr + 3*BS;
        r0[tid] = a_cdx; r1[tid] = a_cdy; r2[tid] = a_p2; r3[tid] = a_grd;
        __syncthreads();
        for (int s = BS/2; s > 32; s >>= 1) {
            if (tid < s) { r0[tid] += r0[tid+s]; r1[tid] += r1[tid+s]; r2[tid] += r2[tid+s]; r3[tid] += r3[tid+s]; }
            __syncthreads();
        }
        if (tid < 32) {
            float v0 = r0[tid]+r0[tid+32], v1 = r1[tid]+r1[tid+32];
            float v2 = r2[tid]+r2[tid+32], v3 = r3[tid]+r3[tid+32];
            for (int off = 16; off > 0; off >>= 1) {
                v0 += __shfl_down_sync(0xffffffff, v0, off);
                v1 += __shfl_down_sync(0xffffffff, v1, off);
                v2 += __shfl_down_sync(0xffffffff, v2, off);
                v3 += __shfl_down_sync(0xffffffff, v3, off);
            }
            if (tid == 0) {
                float vol = v2 * dA;
                d_vol[ci] = vol;
                d_vdev[ci] = d_ta[ci] - vol;
                if (v2 > 1e-8f) {
                    float ccx = rx + v0 / v2;
                    float ccy = ry + v1 / v2;
                    ccx = fmodf(fmodf(ccx, (float)Nx) + Nx, (float)Nx);
                    ccy = fmodf(fmodf(ccy, (float)Ny) + Ny, (float)Ny);
                    d_cx[ci] = ccx;
                    d_cy[ci] = ccy;
                }
                d_peri[ci] = v3 * dA;
            }
        }
        // Second moments (rare path)
        if constexpr (MOMENTS) {
            __syncthreads();
            float *m0 = s_reduce, *m1 = m0 + BS;
            m0[tid] = a_mx; m1[tid] = a_my;
            __syncthreads();
            for (int s = BS/2; s > 32; s >>= 1) {
                if (tid < s) { m0[tid] += m0[tid+s]; m1[tid] += m1[tid+s]; }
                __syncthreads();
            }
            if (tid < 32) {
                float mx = m0[tid]+m0[tid+32], my = m1[tid]+m1[tid+32];
                for (int off = 16; off > 0; off >>= 1) {
                    mx += __shfl_down_sync(0xffffffff, mx, off);
                    my += __shfl_down_sync(0xffffffff, my, off);
                }
                if (tid == 0) { d_mx[ci] = mx; d_my[ci] = my; }
            }
        }
    }
}

void launch_fused(CellArrays& c, const SimParams& p,
                  int max_w, int max_h, int step) {
    int n = c.num_cells;
    if (n == 0) return;

    bool remap  = (step % 10 == 0);
    bool moment = (step % 10 == 9);
    size_t smem = K_MAX * sizeof(float*) + K_MAX * 4 * sizeof(int) + 4 * 256 * sizeof(float);
    float ih2   = 1.0f / (p.dx * p.dx);
    float i2dx  = 0.5f / p.dx;
    float i2dy  = 0.5f / p.dy;

    #define LAUNCH(R, M) k_fused<R, M><<<n, 256, smem>>>( \
        c.phi_ptrs, c.phi_out_ptrs, \
        c.widths, c.heights, c.offsets_x, c.offsets_y, \
        c.old_widths, c.old_heights, c.shift_x, c.shift_y, \
        c.nbr_list, c.nbr_count, \
        c.velocities_x, c.velocities_y, c.volume_devs, \
        c.ref_x, c.ref_y, \
        c.two_gamma, c.two_gamma_bulk, c.vol_coeff, c.tgt_area, \
        c.v_A_cell, c.polar_x, c.polar_y, \
        c.volumes, c.volume_devs, c.centroids_x, c.centroids_y, \
        c.perimeters, c.moment_x, c.moment_y, \
        2.0f * p.interaction_coeff(), ih2, i2dx, i2dy, \
        p.dt, p.dA(), p.motility_coeff(), \
        p.halo, p.Nx, p.Ny, n)

    if      ( remap &&  moment) LAUNCH(true,  true);
    else if ( remap && !moment) LAUNCH(true,  false);
    else if (!remap &&  moment) LAUNCH(false, true);
    else                        LAUNCH(false, false);
    #undef LAUNCH
}

// ===== 5. Swap ===============================================================

__global__ void k_swap(
    float** __restrict__ pin, float** __restrict__ pout,
    int* __restrict__ ox, int* __restrict__ oy,
    const int* __restrict__ sx, const int* __restrict__ sy,
    int Nx, int Ny, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float* tmp = pin[i]; pin[i] = pout[i]; pout[i] = tmp;
    ox[i] = wrap(ox[i] + sx[i], Nx);
    oy[i] = wrap(oy[i] + sy[i], Ny);
}

void launch_swap(CellArrays& c, int Nx, int Ny) {
    k_swap<<<(c.num_cells + 255) / 256, 256>>>(
        c.phi_ptrs, c.phi_out_ptrs,
        c.offsets_x, c.offsets_y, c.shift_x, c.shift_y,
        Nx, Ny, c.num_cells);
}

// ===== 6. Polarization =======================================================
// We store theta directly to avoid losing precision through an atan2 round-trip
// each step. (polar_x, polar_y) are kept in sync for the fused kernel.

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
            theta = curand_uniform(&s) * 6.2831853f;
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
    int n = c.num_cells;
    if (n == 0 || p.v_A == 0.0f) return;
    k_polar<<<(n + 255) / 256, 256>>>(
        (curandState*)c.rng_states,
        c.polar_theta, c.polar_x, c.polar_y,
        p.dt, p.tau, p.abp, n);
}

// ===== 7. RNG init ===========================================================

__global__ void k_rng_init(curandState* st, unsigned long seed, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curand_init(seed, i, 0, &st[i]);
}

void launch_rng_init(CellArrays& c, unsigned long seed) {
    k_rng_init<<<(c.num_cells + 255) / 256, 256>>>(
        (curandState*)c.rng_states, seed, c.num_cells);
}

// ===== 8. Initial centroid + volume reduction ================================

__global__ void k_init_reduce(
    float** __restrict__ phi_in,
    const int* __restrict__ W, const int* __restrict__ H,
    const int* __restrict__ OX, const int* __restrict__ OY,
    const float* __restrict__ RX, const float* __restrict__ RY,
    float* __restrict__ d_cx, float* __restrict__ d_cy,
    float* __restrict__ d_vol,
    int halo, int Nx, int Ny, int Ncells)
{
    int ci = blockIdx.z;
    if (ci >= Ncells) return;
    int w = W[ci], h = H[ci];
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int BS = blockDim.x * blockDim.y;

    float my_dx = 0, my_dy = 0, my_p2 = 0;
    if (lx < w && ly < h &&
        lx >= halo && lx < w - halo &&
        ly >= halo && ly < h - halo) {
        float p = phi_in[ci][ly * w + lx];
        float p2 = p * p;
        float gx = (float)(OX[ci] + lx);
        float gy = (float)(OY[ci] + ly);
        my_dx = pdelta(gx - RX[ci], (float)Nx) * p2;
        my_dy = pdelta(gy - RY[ci], (float)Ny) * p2;
        my_p2 = p2;
    }

    extern __shared__ float sm[];
    float *s0 = sm, *s1 = sm + BS, *s2 = sm + 2 * BS;
    s0[tid] = my_dx; s1[tid] = my_dy; s2[tid] = my_p2;
    __syncthreads();
    for (int s = BS / 2; s > 32; s >>= 1) {
        if (tid < s) { s0[tid] += s0[tid+s]; s1[tid] += s1[tid+s]; s2[tid] += s2[tid+s]; }
        __syncthreads();
    }
    if (tid < 32) {
        float v0 = s0[tid]+s0[tid+32], v1 = s1[tid]+s1[tid+32], v2 = s2[tid]+s2[tid+32];
        for (int off = 16; off > 0; off >>= 1) {
            v0 += __shfl_down_sync(0xffffffff, v0, off);
            v1 += __shfl_down_sync(0xffffffff, v1, off);
            v2 += __shfl_down_sync(0xffffffff, v2, off);
        }
        if (tid == 0) {
            atomicAdd(&d_cx[ci], v0);
            atomicAdd(&d_cy[ci], v1);
            atomicAdd(&d_vol[ci], v2);
        }
    }
}

__global__ void k_init_finalize(
    float* __restrict__ cx, float* __restrict__ cy,
    float* __restrict__ vol, float* __restrict__ vdev,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ ta, float dA, int Nx, int Ny, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sp2 = vol[i];
    if (sp2 > 1e-8f) {
        float ccx = rx[i] + cx[i] / sp2;
        float ccy = ry[i] + cy[i] / sp2;
        ccx = fmodf(fmodf(ccx, (float)Nx) + Nx, (float)Nx);
        ccy = fmodf(fmodf(ccy, (float)Ny) + Ny, (float)Ny);
        cx[i] = ccx; cy[i] = ccy;
    }
    float v = sp2 * dA;
    vol[i] = v;
    vdev[i] = ta[i] - v;
}

void launch_initial_reduce(CellArrays& c, const SimParams& p,
                           int max_w, int max_h) {
    int n = c.num_cells;
    if (n == 0) return;
    cudaMemset(c.centroids_x, 0, n * sizeof(float));
    cudaMemset(c.centroids_y, 0, n * sizeof(float));
    cudaMemset(c.volumes, 0, n * sizeof(float));

    dim3 blk(32, 8);
    dim3 grd((max_w + 31) / 32, (max_h + 7) / 8, n);
    k_init_reduce<<<grd, blk, 3 * 256 * sizeof(float)>>>(
        c.phi_ptrs, c.widths, c.heights,
        c.offsets_x, c.offsets_y, c.ref_x, c.ref_y,
        c.centroids_x, c.centroids_y, c.volumes,
        p.halo, p.Nx, p.Ny, n);

    k_init_finalize<<<(n + 255) / 256, 256>>>(
        c.centroids_x, c.centroids_y,
        c.volumes, c.volume_devs,
        c.ref_x, c.ref_y, c.tgt_area,
        p.dA(), p.Nx, p.Ny, n);
}

// ===== 9. Initial velocity integral ==========================================

__global__ void k_vel_integral(
    float** __restrict__ phi_in,
    const int* __restrict__ W, const int* __restrict__ H,
    const int* __restrict__ OX, const int* __restrict__ OY,
    const NeighborEntry* __restrict__ nlist, const int* __restrict__ ncnt,
    float* __restrict__ ix, float* __restrict__ iy,
    float i2dx, float i2dy, int halo, int Nx, int Ny, int Ncells)
{
    int ci = blockIdx.z;
    if (ci >= Ncells) return;
    int w = W[ci], h = H[ci];
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int BS = blockDim.x * blockDim.y;

    float mx = 0, my = 0;
    if (lx < w && ly < h &&
        lx >= halo && lx < w - halo &&
        ly >= halo && ly < h - halo) {
        const float* phi = phi_in[ci];
        float pv = phi[ly * w + lx];
        float gx = (phi[ly * w + lx + 1] - phi[ly * w + lx - 1]) * i2dx;
        float gy = (phi[(ly + 1) * w + lx] - phi[(ly - 1) * w + lx]) * i2dy;

        float S = 0;
        int kn = ncnt[ci];
        for (int ni = 0; ni < kn; ni++) {
            int nid = nlist[ci * K_MAX + ni].cell_id;
            int dx = OX[nid] - OX[ci];
            int dy = OY[nid] - OY[ci];
            if (dx >  Nx / 2) dx -= Nx;
            if (dx < -Nx / 2) dx += Nx;
            if (dy >  Ny / 2) dy -= Ny;
            if (dy < -Ny / 2) dy += Ny;
            int nw = W[nid], nh = H[nid];
            int nlx = lx - dx, nly = ly - dy;
            if (nlx >= halo && nlx < nw - halo &&
                nly >= halo && nly < nh - halo) {
                float pm = phi_in[nid][nly * nw + nlx];
                S += pm * pm;
            }
        }
        mx = pv * gx * S;
        my = pv * gy * S;
    }

    extern __shared__ float sm[];
    float *s0 = sm, *s1 = sm + BS;
    s0[tid] = mx; s1[tid] = my;
    __syncthreads();
    for (int s = BS / 2; s > 32; s >>= 1) {
        if (tid < s) { s0[tid] += s0[tid+s]; s1[tid] += s1[tid+s]; }
        __syncthreads();
    }
    if (tid < 32) {
        float v0 = s0[tid]+s0[tid+32], v1 = s1[tid]+s1[tid+32];
        for (int off = 16; off > 0; off >>= 1) {
            v0 += __shfl_down_sync(0xffffffff, v0, off);
            v1 += __shfl_down_sync(0xffffffff, v1, off);
        }
        if (tid == 0) { atomicAdd(&ix[ci], v0); atomicAdd(&iy[ci], v1); }
    }
}

__global__ void k_vel_finalize(
    float* __restrict__ vx, float* __restrict__ vy,
    const float* __restrict__ ix, const float* __restrict__ iy,
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ vA, float mc, float dA, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    vx[i] = mc * ix[i] * dA + vA[i] * px[i];
    vy[i] = mc * iy[i] * dA + vA[i] * py[i];
}

void launch_initial_velocity(CellArrays& c, const SimParams& p,
                             int max_w, int max_h) {
    int n = c.num_cells;
    if (n == 0) return;

    // Reuse moment_x/y as scratch accumulators — they're zeroed at init and
    // only consumed from step 10 onward, so safe to clobber here then zero.
    float* d_ix = c.moment_x;
    float* d_iy = c.moment_y;
    cudaMemset(d_ix, 0, n * sizeof(float));
    cudaMemset(d_iy, 0, n * sizeof(float));

    dim3 blk(32, 8);
    dim3 grd((max_w + 31) / 32, (max_h + 7) / 8, n);
    k_vel_integral<<<grd, blk, 2 * 256 * sizeof(float)>>>(
        c.phi_ptrs, c.widths, c.heights,
        c.offsets_x, c.offsets_y,
        c.nbr_list, c.nbr_count,
        d_ix, d_iy,
        0.5f / p.dx, 0.5f / p.dy,
        p.halo, p.Nx, p.Ny, n);

    k_vel_finalize<<<(n + 255) / 256, 256>>>(
        c.velocities_x, c.velocities_y,
        d_ix, d_iy,
        c.polar_x, c.polar_y,
        c.v_A_cell, p.motility_coeff(), p.dA(), n);

    // Restore moment buffers to zero so the first remap step reads valid data.
    cudaMemset(d_ix, 0, n * sizeof(float));
    cudaMemset(d_iy, 0, n * sizeof(float));
}
