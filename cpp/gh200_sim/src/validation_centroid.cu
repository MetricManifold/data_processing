// ===========================================================================
// Validation-only phi- and phi^2-weighted periodic centroids.
// ===========================================================================

#include "../include/validation_centroid.cuh"

#include <cmath>

namespace pf {
namespace {

constexpr int kValidationCentroidWarps = kValidationCentroidThreads / 32;
constexpr int kValidationCentroidSlots = 6;

__device__ __forceinline__ double wrap_periodic(double value, int L) {
    const double period = (double)L;
    return value - floor(value / period) * period;
}

__global__ __launch_bounds__(kValidationCentroidThreads)
void k_validation_centroids(const float* __restrict__ phi,
                            const CellState* __restrict__ cell,
                            const uint8_t* __restrict__ clsv,
                            ValidationCentroidCell* __restrict__ out,
                            int N, int L)
{
    __shared__ double reduced[kValidationCentroidWarps]
                             [kValidationCentroidSlots];
    const int n = (int)blockIdx.x;
    if (n >= N) return;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int cls = (int)clsv[n];
    if (cls < 0 || cls >= kNumClasses) {
        if (tid == 0) {
            ValidationCentroidCell invalid{};
            invalid.global_id = cell[n].global_id;
            invalid.cls = cls;
            out[n] = invalid;
        }
        return;
    }

    const ShapeClass sc = class_of(cls);
    const int wx = sc.wx, wy = sc.wy;
    const int tx0 = sc.tx0, ty0 = sc.ty0;
    const float* tile = phi + (size_t)n * kTileArea;

    double s1 = 0.0, x1 = 0.0, y1 = 0.0;
    double s2 = 0.0, x2 = 0.0, y2 = 0.0;
    for (int p = tid; p < wx * wy; p += kValidationCentroidThreads) {
        const int a = p % wx;
        const int b = p / wx;
        const double value = (double)tile[(size_t)(ty0 + b) * kTilePitch
                                          + tx0 + a];
        const double value2 = value * value;
        s1 += value;
        x1 += value * (double)a;
        y1 += value * (double)b;
        s2 += value2;
        x2 += value2 * (double)a;
        y2 += value2 * (double)b;
    }

#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        s1 += __shfl_down_sync(0xFFFFFFFFu, s1, delta);
        x1 += __shfl_down_sync(0xFFFFFFFFu, x1, delta);
        y1 += __shfl_down_sync(0xFFFFFFFFu, y1, delta);
        s2 += __shfl_down_sync(0xFFFFFFFFu, s2, delta);
        x2 += __shfl_down_sync(0xFFFFFFFFu, x2, delta);
        y2 += __shfl_down_sync(0xFFFFFFFFu, y2, delta);
    }
    if (lane == 0) {
        reduced[warp][0] = s1;
        reduced[warp][1] = x1;
        reduced[warp][2] = y1;
        reduced[warp][3] = s2;
        reduced[warp][4] = x2;
        reduced[warp][5] = y2;
    }
    __syncthreads();

    if (tid == 0) {
        double total1 = 0.0, total_x1 = 0.0, total_y1 = 0.0;
        double total2 = 0.0, total_x2 = 0.0, total_y2 = 0.0;
        // Fixed warp-index order, matching the deterministic reduction policy
        // used by the solver moments without sharing their state.
        for (int w = 0; w < kValidationCentroidWarps; ++w) {
            total1   += reduced[w][0];
            total_x1 += reduced[w][1];
            total_y1 += reduced[w][2];
            total2   += reduced[w][3];
            total_x2 += reduced[w][4];
            total_y2 += reduced[w][5];
        }

        const CellState cs = cell[n];
        ValidationCentroidCell result{};
        result.global_id = cs.global_id;
        result.cls = cls;
        result.sum_phi = total1;
        result.sum_phi2 = total2;
        if (isfinite(total1) && isfinite(total_x1) && isfinite(total_y1)
            && total1 > 0.0) {
            result.valid_mask |= kCentroidPhiValid;
            result.cx_phi = wrap_periodic((double)cs.gx0 + total_x1 / total1, L);
            result.cy_phi = wrap_periodic((double)cs.gy0 + total_y1 / total1, L);
        }
        if (isfinite(total2) && isfinite(total_x2) && isfinite(total_y2)
            && total2 > 0.0) {
            result.valid_mask |= kCentroidPhi2Valid;
            result.cx_phi2 = wrap_periodic((double)cs.gx0 + total_x2 / total2, L);
            result.cy_phi2 = wrap_periodic((double)cs.gy0 + total_y2 / total2, L);
        }
        out[n] = result;
    }
}

}  // namespace

void launch_validation_centroids(const float* phi,
                                 const CellState* cell,
                                 const uint8_t* cls,
                                 ValidationCentroidCell* out,
                                 int N, int L, cudaStream_t stream)
{
    k_validation_centroids<<<N, kValidationCentroidThreads, 0, stream>>>(
        phi, cell, cls, out, N, L);
}

}  // namespace pf
