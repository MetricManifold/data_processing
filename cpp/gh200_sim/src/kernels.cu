// ===========================================================================
// FUSE-1R solver kernels.
//
// k_step is the ONLY kernel in the steady-state loop. One CTA per cell, the
// cell's whole rect resident in shared memory:
//
//   P0  scalar setup + run-and-tumble + synthesised stencil ring
//   P1  3-stage cp.async strip pipeline; exact Ix/Iy over this step's phi & S
//   P1b fp64 fixed-order reduction -> v_n (no lag, no split, no second kernel)
//   P2  RHS sweep with a 3-row rolling window; phi^{n+1} staged back into S_s
//   P3  shifted store, fused S scatter for step n+1, next step's moments/bbox
//   P3b fp64 fixed-order reduction of V/Cx/Cy/perim + integer bbox reduce
//
// Every coefficient comes from params.cuh. There is no 30, 60 or 120 below.
// ===========================================================================

#include "../include/kernels.cuh"

#include <cuda_pipeline.h>

namespace pf {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ int wrapi(int v, int L) {
    v %= L;
    return v < 0 ? v + L : v;
}

// Exact non-negative self-subtraction in Q5.27.
//
// Step n reads exactly the phi buffer that step n-1 scattered from, with the
// same q_of() and the same rect->global map, so q_S - q_of(phi_n) is an exact
// uint32 subtraction and is provably non-negative. There is therefore NO
// "if (Soth < 0) Soth = 0" clamp here. If the subtraction ever does go
// negative an invariant has been broken upstream; that is counted in
// FLAG_S_NEGATIVE and reported, never silently floored.
__device__ __forceinline__ float s_other(uint32_t qS, float phi_self,
                                         uint32_t* flags) {
    const uint32_t qc = q_of(phi_self);
    if (qS >= qc) return (float)(qS - qc) * kQInvF;
    PF_FATAL_ADD(flags, FLAG_S_NEGATIVE);
    return 0.0f;
}

// class_containing() -- the smallest-window class that CONTAINS an (ex, ey)
// support -- now lives in params.cuh next to the class table, because the
// checkpoint loader has to make exactly the same decision when it repacks a
// foreign tile. Same body, same semantics, one definition.

// __restrict__ is a promise that, for as long as the pointer is in scope, the
// object it designates is not read or written through any other pointer. That
// is true of the fused path's output tile: P3 is the only phase that touches
// it. It is NOT true in the large path, where P2 has already written the same
// tile through a pointer of its own and P3 reads those bytes back. Rather than
// drop the qualifier for everyone -- which would change codegen for the
// classes this change is required to leave alone -- let it follow the path.
template <bool STAGED> struct TileOutPtr;
template <> struct TileOutPtr<true>  { typedef float* __restrict__ type; };
template <> struct TileOutPtr<false> { typedef float* type; };

__device__ __forceinline__ uint32_t part1by1(uint32_t x) {
    x &= 0x0000FFFFu;
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x;
}
__device__ __forceinline__ uint32_t morton2d(uint32_t x, uint32_t y) {
    return part1by1(x) | (part1by1(y) << 1);
}

// ---------------------------------------------------------------------------
// process_cell<CLS> -- one cell, one CTA, one step.
//
// The SOURCE geometry (the class of the field being read) is a template
// parameter, so the thread -> pixel map and the reduction tree in P1/P1b/P2
// depend only on compile-time constants: bit-reproducibility is independent of
// SM count, of the work-cursor interleaving, and of cell ordering.
//
// The DESTINATION geometry is a runtime quadruple. It equals the source on
// every ordinary step; it differs only on the rare step where a cell changes
// shape class, and P3's 2-D (warp-row, lane-column) walk needs no integer
// division, so making it runtime costs nothing and avoids nine template
// instantiations of the store loop.
//
// ---- TWO BODIES, selected at COMPILE TIME by kStagesS<CLS> -----------------
//
// STAGED (classes 0..3, kStagesS true) -- unchanged, byte for byte:
//   P1  cp.async stages phi AND S into smem; S_s[y*WX+i] feeds the Ix/Iy
//       integrand.
//   P2  reads S from S_s and OVERWRITES S_s with phi^{n+1}.
//   P3  reads phi^{n+1} out of S_s, does the shifted store, the S scatter and
//       the moments.
//
// LARGE (kClassLarge, kStagesS false) -- S_s does not exist:
//   P1  cp.async stages phi ONLY; S is read straight from global at the same
//       (rotation slot, rect->global map, word) the staged path would have
//       copied, so the Q5.27 exactness argument is untouched -- see the note
//       at the read site.
//   P2  reads S from global and writes phi^{n+1} STRAIGHT TO GLOBAL through
//       the shifted-store map P3 applies in the staged path, then a frame pass
//       zeroes the destination pixels that have no source pixel. Between them
//       the two write every destination pixel exactly once, so the global
//       result is bit-identical to what the staged path would have produced.
//   P3  RE-READS phi^{n+1} from global (the __syncthreads() at the end of P2 is
//       the block-scope fence that makes P2's stores visible) and does the
//       scatter, moments, bbox and perimeter from it. That extra HBM read is
//       what the large path costs; only a minority of cells are ever in it.
//
// The split is `if constexpr`, NOT a runtime branch: classes 0..3 do not emit a
// single instruction of the large path, so their instruction schedule is what
// it was. They DO share k_step's register allocation with the large body,
// which is the one thing to re-measure after this change (regs and spill are
// printed at startup and by -Xptxas -v).
// ---------------------------------------------------------------------------
template <int CLS>
__device__ void process_cell(int n, const StepArgs& A, char* smem,
                             unsigned long long step)
{
    constexpr int WX  = kClasses[CLS].wx;
    constexpr int WY  = kClasses[CLS].wy;
    constexpr int TX0 = kClasses[CLS].tx0;
    constexpr int TY0 = kClasses[CLS].ty0;
    constexpr int PX  = phi_pitch(WX);
    constexpr int NS  = WY / kStripRows;
    constexpr int RB  = (WY + kWarpsPerBlock - 1) / kWarpsPerBlock;
    constexpr int kRing = 2 * (WX + 2) + 2 * WY;

    // The CTA's shared-memory request is kSmemBytes = max over classes, so this
    // instantiation must not need more than the class table says it does.
    static_assert(class_smem_of(CLS) <= kSmemRaw,
                  "this shape class needs more shared memory than the launch "
                  "requests");
    static_assert(RB * kWarpsPerBlock >= WY,
                  "P2's row bands must cover the whole rect");
    static_assert(WY % kStripRows == 0, "P1's strips must tile the rect exactly");

    double*   red_s = reinterpret_cast<double*>(smem);
    int*      bci   = reinterpret_cast<int*>(smem + kRedBytes);
    float*    bcf   = reinterpret_cast<float*>(smem + kRedBytes);
    float*    phi_s = reinterpret_cast<float*>(smem + kScalarBytes);
    // nullptr, not an out-of-range offset, in the large instantiation: every
    // use of S_s below sits inside an `if constexpr (kStagesS<CLS>)`, and a
    // missed guard is then an immediate null dereference rather than a silent
    // overwrite of whatever happens to follow phi_s.
    [[maybe_unused]] uint32_t* const S_s =
        kStagesS<CLS> ? reinterpret_cast<uint32_t*>(smem + kScalarBytes
                                                    + phi_bytes(WX, WY))
                      : nullptr;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // =====================================================================
    // P0 -- scalar setup (thread 0) and the synthesised 1-px zero ring.
    //
    // By invariant I1 the tile pixels immediately outside the window are
    // exactly 0.0f, so the ring is written rather than loaded: phi is read
    // exactly WX*WY per cell per step, with no halo at all.
    // =====================================================================
    if (tid == 0) {
        // REFERENCE, not a by-value copy. CellState is 192 B, of which this
        // block reads 17 scalars and never touches reserved[18] (72 B). The
        // by-value copy materialised the whole aggregate, which ptxas could not
        // hold in the 85-register budget a 768-thread CTA allows (65536/768),
        // so it placed the record on the stack: 368 B frame with spill traffic,
        // against a README that gates on zero spill. Binding a reference lets
        // the compiler issue loads only for the members actually read.
        //
        // Safe here because this whole block is READ-ONLY on A.cell[n]: the
        // fused path publishes its CellState update in P3, after the barrier.
        // (process_cell_rhs writes in P0 and therefore snapshots explicitly.)
        const CellState& cs = A.cell[n];

        // --- run and tumble: P(t_r) = (1/tau) exp(-t_r/tau) ---------------
        // A.p_tumble is -expm1(-dt/tau) computed in double on the host.
        // Recomputing p_hat from the angle every step keeps |p_hat| == 1 by
        // construction: no renormalisation drift, no RNG state to checkpoint.
        const Philox4 r = philox4x32_10(
            (uint32_t)(step & 0xFFFFFFFFull), (uint32_t)(step >> 32),
            (uint32_t)cs.global_id, 0xA5A5A5A5u,
            (uint32_t)(A.polarity_seed & 0xFFFFFFFFull),
            (uint32_t)(A.polarity_seed >> 32));
        float theta = cs.theta;
        int tumbled = 0;
        if (philox_uniform53(r.v[0], r.v[1]) < A.p_tumble) {
            theta = (float)(2.0 * kPi * philox_uniform53(r.v[2], r.v[3]));
            tumbled = 1;
        }
        float ph_sin, ph_cos;
        sincosf(theta, &ph_sin, &ph_cos);

        // --- per-cell coefficients ---------------------------------------
        const float gam  = cs.gamma;
        const float dwC  = A.bulk_scale * gam;
        const float volC = (float)(A.vol_scale * (A.A0 - cs.V));

        // --- destination shape class, growth-only hysteresis --------------
        // The destination is the smallest class that CONTAINS the support on
        // both axes. Three cases, and none of them clips:
        //   out of margin here -> move now, to any class that holds both
        //     extents (this is also the only wide<->tall path);
        //   fits here, and a strictly smaller window holds it with the wider
        //     demote margin -> move after kDemoteDwell consecutive checks;
        //   no class holds it -> FLAG_CLASS_EXHAUSTED and stay put. With
        //     kClassLarge in the table that now means an extent above
        //     selected large edge - kPromoteSlack (184 or 200 px), not the
        //     far tighter
        //     "no class is larger than the round one in both axes" it used to
        //     mean. Truncating phi to fake a fit is still not an option.
        const int ex = cs.bb_hi_x - cs.bb_lo_x + 1;
        const int ey = cs.bb_hi_y - cs.bb_lo_y + 1;
        int dcls = CLS;
        unsigned pctr = cs.promote_ctr;
        if (ex > 0) {                    // ex <= 0 <=> the support bbox is empty
            if (ex + kPromoteSlack > WX || ey + kPromoteSlack > WY) {
                const int grow = class_containing(ex, ey, kPromoteSlack);
                if (grow >= 0) dcls = grow;
                // ADD, not OR. Both are sticky (the counter only ever grows
                // from zero) and both make flag_is_fatal() stop the run, but OR
                // saturates at 1 and throws away the only quantitative evidence
                // there is. Job 666491 reported class_exhausted=1 and nothing
                // else: not how many cells, not for how many steps, not how far
                // past the limit. The atomic is on a path that fires only once
                // the run is already invalid, so its cost is irrelevant.
                else PF_FATAL_ADD(A.flags, FLAG_CLASS_EXHAUSTED);
                pctr = 0u;
            } else {
                const int small = class_containing(ex, ey, kDemoteSlack);
                if (small >= 0 && small != CLS
                    && class_of(small).wx * class_of(small).wy < WX * WY) {
                    if (++pctr >= (unsigned)kDemoteDwell) { dcls = small; pctr = 0u; }
                } else {
                    pctr = 0u;
                }
            }
        }
        const ShapeClass dc = class_of(dcls);
        const int dwx = dc.wx,  dwy = dc.wy;
        const int dtx0 = dc.tx0, dty0 = dc.ty0;

        // --- recentring shift ---------------------------------------------
        // The tile window never moves; the shift is applied in P3 by reading
        // shared memory at a shifted index, so a rebind costs 0 extra HBM
        // traffic and invariant I1 needs no frame bookkeeping. The same
        // formula also re-anchors the rect on a class change (where the shift
        // is large and therefore uncapped).
        int sx = 0, sy = 0;
        if (cs.V > 0.0) {
            const double cxr = cs.Cx / cs.V;
            const double cyr = cs.Cy / cs.V;
            sx = __double2int_rn(cxr - 0.5 * (double)(dwx - 1));
            sy = __double2int_rn(cyr - 0.5 * (double)(dwy - 1));
            if (dcls == CLS) {
                sx = max(-kMaxShiftPerStep, min(kMaxShiftPerStep, sx));
                sy = max(-kMaxShiftPerStep, min(kMaxShiftPerStep, sy));
            } else {
                sx = max(-WX, min(WX, sx));
                sy = max(-WY, min(WY, sy));
            }
        } else {
            PF_FATAL_OR(A.flags, FLAG_V_NONPOS);
        }

        // (step + 1), not step. Launch k reads phi^k and writes phi^{k+1}, and P3
        // accumulates the perimeter from the buffer P2 has already overwritten, so
        // perim is of phi^{k+1}. Gating on `step` fired at k = 0, 100, ... and
        // deposited perim(phi^1), perim(phi^101), ... while the trajectory samples
        // at k = 99, 199, ... -- disjoint frame sets, so the perim column ran a
        // fixed 99 steps behind every other column in its own row and biased the
        // shape index q = P/sqrt(A) by pairing numerator and denominator from
        // different times. Diagnostic only: perim never enters the RHS or v_n.
        const int fm = (A.full_moment_every > 0 &&
                        ((step + 1ull) % (unsigned long long)A.full_moment_every) == 0ull);

        bci[0]  = sx;             bci[1]  = sy;
        bci[2]  = cs.gx0;         bci[3]  = cs.gy0;
        bci[4]  = wrapi(cs.gx0 + sx, A.L);
        bci[5]  = wrapi(cs.gy0 + sy, A.L);
        bci[6]  = dcls;
        bci[7]  = (dcls != (int)cs.cls_written[A.parity_out]);
        bcf[8]  = gam;            bcf[9]  = dwC;
        bcf[10] = volC;           bcf[11] = cs.v_A;
        bcf[12] = ph_cos;         bcf[13] = ph_sin;
        bcf[16] = theta;
        bci[17] = fm;
        bci[18] = (int)pctr;
        bci[19] = tumbled;
        bci[20] = dwx;            bci[21] = dwy;
        bci[22] = dtx0;           bci[23] = dty0;
    }

    for (int i = tid; i < kRing; i += kBlockThreads) {
        if (i < WX + 2) {
            phi_s[(kPhiPadLeft - 1) + i] = 0.0f;                        // row -1
        } else if (i < 2 * (WX + 2)) {
            const int k = i - (WX + 2);
            phi_s[(WY + 1) * PX + (kPhiPadLeft - 1) + k] = 0.0f;        // row WY
        } else if (i < 2 * (WX + 2) + WY) {
            const int k = i - 2 * (WX + 2);
            phi_s[(k + 1) * PX + (kPhiPadLeft - 1)] = 0.0f;             // col -1
        } else {
            const int k = i - 2 * (WX + 2) - WY;
            phi_s[(k + 1) * PX + (kPhiPadLeft + WX)] = 0.0f;            // col WX
        }
    }
    __syncthreads();

    const int gx0 = bci[2];
    const int gy0 = bci[3];
    // Periodic x-wrap resolved ONCE per rect, into two contiguous segments;
    // the y-wrap is one add/select on the row index. No per-pixel modulo.
    const int split = min(WX, A.L - gx0);
    const float* __restrict__ tile_in = A.phi_in + (size_t)n * kTileArea;

    // =====================================================================
    // P1 -- strip-pipelined load + exact interaction-velocity integral.
    // =====================================================================
    auto issue_strip = [&](int s) {
        constexpr int kChunksPerRow = WX / 4;                 // 16 B chunks
        constexpr int kPhiUnits     = kStripRows * kChunksPerRow;
        for (int u = tid; u < kPhiUnits; u += kBlockThreads) {
            const int cch = u % kChunksPerRow;
            const int y   = s * kStripRows + u / kChunksPerRow;
            __pipeline_memcpy_async(
                &phi_s[(y + 1) * PX + kPhiPadLeft + cch * 4],
                &tile_in[(size_t)(TY0 + y) * kTilePitch + TX0 + cch * 4],
                16);
        }
        if constexpr (kStagesS<CLS>) {
            // S cannot use 16 B copies: its row start is an arbitrary 4 B
            // aligned global column. 4 B units still coalesce, and the average
            // sector overhead of an unaligned 576 B row is ~2.4%, not worth a
            // 32 px shift quantum (which would force W = 176, +22% traffic).
            constexpr int kSUnits = kStripRows * WX;
            for (int u = tid; u < kSUnits; u += kBlockThreads) {
                const int i = u % WX;
                const int y = s * kStripRows + u / WX;
                int gy = gy0 + y;
                if (gy >= A.L) gy -= A.L;
                const int gx = (i < split) ? (gx0 + i) : (i - split);
                __pipeline_memcpy_async(&S_s[y * WX + i],
                                        &A.S_rd[(size_t)gy * A.P + gx], 4);
            }
        }
        // The large class stages no S: the pipeline carries phi alone. The
        // commit/wait structure is untouched -- the same NS groups are still
        // committed and waited on in the same order, they simply contain fewer
        // copies each.
    };

    int committed = 0;
    for (int s = 0; s < kPipeStages && s < NS; ++s) {
        issue_strip(s);
        __pipeline_commit();
        ++committed;
    }

    double aIx = 0.0, aIy = 0.0;
    for (int s = 0; s < NS; ++s) {
        // Strip s's stencil needs row 16(s+1), i.e. strip s+1's first row.
        // With the prologue depth held at 3, exactly one group may remain
        // pending until the last two strips -- both literals, because
        // cp.async.wait_group takes an immediate.
        if (s <= NS - 3) __pipeline_wait_prior(1);
        else             __pipeline_wait_prior(0);
        __syncthreads();

        // NOTE: a 2-way split of these fp64 accumulators was tried and measured
        // 1.1-1.6% SLOWER at every N from 132 to 4224. The `wait` stall (1.90 of
        // 10.88 cycles/instruction) is evidently not dominated by this chain, and
        // the extra branch plus four more fp64 registers cost more than the ILP
        // gained. Keep the single accumulator.
        const int ybase = s * kStripRows;
        for (int idx = tid; idx < kStripRows * WX; idx += kBlockThreads) {
            const int i = idx % WX;
            const int y = ybase + idx / WX;
            const float* p = &phi_s[(y + 1) * PX + kPhiPadLeft + i];
            const float c  = p[0];
            const float pE = p[1],  pW = p[-1];
            const float pN = p[PX], pS = p[-PX];
            // S is POINTWISE -- it never enters a stencil -- so the large class
            // reads the word directly from global instead of from S_s. The
            // Q5.27 exactness argument is preserved because this is the SAME
            // word: same rotation slot (A.S_rd, chosen by the launch, not by
            // the class), same rect origin (gx0/gy0, this step's, published by
            // P0 into bci[2]/bci[3]), and the same rect->global map -- the two
            // expressions below are character-for-character the ones the staged
            // cp.async uses above. So q_S - q_of(phi_n) is still the exact,
            // provably non-negative self-subtraction that s_other() asserts.
            uint32_t qS;
            if constexpr (kStagesS<CLS>) {
                qS = S_s[y * WX + i];
            } else {
                int ggy = gy0 + y;
                if (ggy >= A.L) ggy -= A.L;
                const int ggx = (i < split) ? (gx0 + i) : (i - split);
                qS = A.S_rd[(size_t)ggy * A.P + ggx];
            }
            const float So = s_other(qS, c, A.flags);
            const float gx = 0.5f * (pE - pW);
            const float gy = 0.5f * (pN - pS);
            aIx += (double)(c * gx * So);
            aIy += (double)(c * gy * So);
        }

        if (committed < NS) { issue_strip(committed); __pipeline_commit(); ++committed; }
    }

    // ---- P1b: fp64 warp butterfly, then a fixed-order serial sum over the
    // 24 warp slots in ascending warp index. Order depends only on
    // (WX, WY, blockDim), all compile-time constants.
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        aIx += __shfl_down_sync(0xFFFFFFFFu, aIx, d);
        aIy += __shfl_down_sync(0xFFFFFFFFu, aIy, d);
    }
    if (lane == 0) {
        red_s[warp * kRedSlots + 0] = aIx;
        red_s[warp * kRedSlots + 1] = aIy;
    }
    __syncthreads();
    if (tid == 0) {
        double sIx = 0.0, sIy = 0.0;
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sIx += red_s[w * kRedSlots + 0];
            sIy += red_s[w * kRedSlots + 1];
        }
        // v_n = v_A p_hat + motility * integral(phi grad(phi) S_other dA).
        // The interaction term is repulsive: for cell n left of cell m the
        // overlap sits on n's right, where grad_x(phi_n) < 0, so Ix < 0 and
        // n accelerates away from m.
        const double vxd = (double)bcf[11] * (double)bcf[12]
                         + (double)A.mot_coeff * sIx;
        const double vyd = (double)bcf[11] * (double)bcf[13]
                         + (double)A.mot_coeff * sIy;
        bcf[14] = (float)vxd;
        bcf[15] = (float)vyd;
        CellState* cs = &A.cell[n];
        cs->vx = (float)vxd;  cs->vy = (float)vyd;
        cs->Ix = sIx;         cs->Iy = sIy;
    }
    __syncthreads();

    // ---- LARGE path only: the class-change tile zeroing must precede the P2
    // stores, because in this path P2 IS the store. The staged path still does
    // it at the top of P3, untouched, where its stores also follow it.
    // bci[7] is a broadcast word, identical in every thread, so the
    // __syncthreads() inside the branch is block-uniform.
    if constexpr (!kStagesS<CLS>) {
        if (bci[7]) {
            float* tz = A.phi_out + (size_t)n * kTileArea;
            for (int i = tid; i < kTileArea; i += kBlockThreads) tz[i] = 0.0f;
            __syncthreads();
        }
    }

    // =====================================================================
    // P2 -- RHS sweep. Thread (lane, warp) owns columns lane + 32k and a
    // contiguous band of RB rows, walking DOWN its column with a 3x3 rolling
    // window in registers: 3 shared reads per row instead of 9.
    //
    // STAGED: phi^{n+1} is staged back into S_s, which is safe because S is
    // used pointwise and no other thread ever reads that slot.
    // LARGE:  S comes from global and phi^{n+1} goes to global, through the
    // shifted-store map the staged path applies in P3.
    // =====================================================================
    {
        const float gam  = bcf[8];
        const float dwC  = bcf[9];
        const float volC = bcf[10];
        const float vxf  = bcf[14];
        const float vyf  = bcf[15];
        const float repC = A.rep_coeff;
        const float dtf  = A.dt;
        const int   y0   = warp * RB;

        // Destination map, for the LARGE path's store only, read from the same
        // broadcast words P3 reads below so the two cannot disagree about where
        // a pixel goes. Every one of these folds to a compile-time 0 (or a
        // nullptr) in the staged instantiations, where the only references to
        // them live inside discarded `if constexpr` branches -- so no shared
        // load is issued and no register is held. [[maybe_unused]] is what says
        // that on purpose rather than leaving a warning behind.
        [[maybe_unused]] const int dsx   = kStagesS<CLS> ? 0 : bci[0];
        [[maybe_unused]] const int dsy   = kStagesS<CLS> ? 0 : bci[1];
        [[maybe_unused]] const int ddwx  = kStagesS<CLS> ? 0 : bci[20];
        [[maybe_unused]] const int ddwy  = kStagesS<CLS> ? 0 : bci[21];
        [[maybe_unused]] const int ddtx0 = kStagesS<CLS> ? 0 : bci[22];
        [[maybe_unused]] const int ddty0 = kStagesS<CLS> ? 0 : bci[23];
        [[maybe_unused]] float* const tout =
            kStagesS<CLS> ? nullptr : (A.phi_out + (size_t)n * kTileArea);

        if (y0 < WY) {
            const int y1 = min(y0 + RB, WY);
            for (int xb = 0; xb < WX; xb += 32) {
                const int x = lane + xb;
                if (x >= WX) break;
                const float* p = &phi_s[y0 * PX + kPhiPadLeft + x];  // row y0-1
                float sW = p[-1], sC = p[0], sE = p[1];
                p += PX;                                             // row y0
                float cW = p[-1], cC = p[0], cE = p[1];
                for (int y = y0; y < y1; ++y) {
                    p += PX;                                         // row y+1
                    const float nW = p[-1], nC = p[0], nE = p[1];
                    const float lap = ((float)kLapEdgeW * (nC + sC + cE + cW)
                                     + (float)kLapDiagW * (nE + nW + sE + sW)
                                     + (float)kLapCentreW * cC)
                                    * (float)(1.0 / kLapDenom);
                    const float gx = 0.5f * (cE - cW);
                    const float gy = 0.5f * (nC - sC);
                    // Same word, same rotation slot, same rect->global map as
                    // the staged load: see the identical note in P1.
                    uint32_t qS;
                    if constexpr (kStagesS<CLS>) {
                        qS = S_s[y * WX + x];
                    } else {
                        int ggy = gy0 + y;
                        if (ggy >= A.L) ggy -= A.L;
                        // Loop-invariant in y; hoisted by LICM rather than by
                        // hand so it does not exist at all in the staged path.
                        const int ggx = (x < split) ? (gx0 + x) : (x - split);
                        qS = A.S_rd[(size_t)ggy * A.P + ggx];
                    }
                    const float So = s_other(qS, cC, A.flags);
                    const float rhs = gam * lap
                                    - dwC * (cC * (1.0f - cC) * (1.0f - 2.0f * cC))
                                    + volC * cC
                                    - repC * cC * So
                                    - (vxf * gx + vyf * gy);
                    const float pnew = cC + dtf * rhs;
                    if constexpr (kStagesS<CLS>) {
                        S_s[y * WX + x] = __float_as_uint(pnew);
                    } else {
                        // destination (a,b) <- source (a+sx, b+sy), i.e. source
                        // (x,y) -> (x-sx, y-sy) when that is inside the
                        // destination window. Plain store, NOT __stcs: P3
                        // re-reads these bytes one barrier later, so an
                        // evict-first hint is exactly the wrong one here.
                        // a and its bound test are loop-invariant in y; left to
                        // LICM so they cost the staged path nothing.
                        const int a = x - dsx;
                        const int b = y - dsy;
                        if ((unsigned)a < (unsigned)ddwx &&
                            (unsigned)b < (unsigned)ddwy)
                            tout[(size_t)(ddty0 + b) * kTilePitch + ddtx0 + a]
                                = pnew;
                    }
                    sW = cW; sC = cC; sE = cE;
                    cW = nW; cC = nC; cE = nE;
                }
            }
        }

        // ---- LARGE path: frame pass. Destination pixels with no source pixel
        // get 0.0f. The predicate is P3's own (jin && iin), so the two paths
        // agree exactly on which pixels are zero. It is disjoint from the
        // stores above -- the two sets partition the destination window, since
        // the store fires precisely when (x-sx, y-sy) is in range, i.e. when
        // (a+sx, b+sy) is in the source rect -- so no barrier is needed between
        // them, and after the __syncthreads() below EVERY destination pixel has
        // been written exactly once. That is what lets P3 read the whole window
        // back from global with no predicate at all.
        if constexpr (!kStagesS<CLS>) {
            for (int b = warp; b < ddwy; b += kWarpsPerBlock) {
                const int j = b + dsy;
                const bool jin = ((unsigned)j < (unsigned)WY);
                for (int aa = lane; aa < ddwx; aa += 32) {
                    const int i = aa + dsx;
                    const bool iin = ((unsigned)i < (unsigned)WX);
                    if (!(jin && iin))
                        tout[(size_t)(ddty0 + b) * kTilePitch + ddtx0 + aa]
                            = 0.0f;
                }
            }
        }
    }
    // Block-scope memory fence as well as a barrier: in the large path this is
    // what makes P2's global stores visible to P3's loads below. All the
    // threads involved are in one CTA and therefore on one SM, sharing one L1.
    __syncthreads();

    // =====================================================================
    // P3 -- shifted store + fused S scatter for step n+1 + next step's
    // moments and support bbox. Four things happen for the price of the store
    // that was going to happen anyway.
    // =====================================================================
    const int sx = bci[0], sy = bci[1];
    const int gx0n = bci[4], gy0n = bci[5];
    const int dcls = bci[6];
    const int fm   = bci[17];
    const int dwx = bci[20], dwy = bci[21], dtx0 = bci[22], dty0 = bci[23];

    // __restrict__ in the staged instantiations only: there P3 is the sole
    // accessor of this tile, so the promise is true. In the large instantiation
    // P2 has already written it, and the qualifier would be a lie -- see
    // TileOutPtr at the top of this file.
    typename TileOutPtr<kStagesS<CLS>>::type tile_out =
        A.phi_out + (size_t)n * kTileArea;
    if constexpr (kStagesS<CLS>) {
        if (bci[7]) {
            // The only place a tile is ever zeroed: the destination window of
            // the new class does not contain the old one, so the residue
            // outside it would break invariant I1. Fires only on a genuine
            // class change. The large path did this before P2 instead.
            for (int i = tid; i < kTileArea; i += kBlockThreads) tile_out[i] = 0.0f;
            __syncthreads();
        }
    }

    double aV = 0.0, aCx = 0.0, aCy = 0.0, aPer = 0.0;
    int blox = dwx, bhix = -1, bloy = dwy, bhiy = -1, pmaxb = 0;

    for (int b = warp; b < dwy; b += kWarpsPerBlock) {
        // The source pixel of destination (a,b), and whether it exists. Used by
        // the staged path only: the large path's source rect is not resident,
        // and the same information is already baked into what P2 and the frame
        // pass wrote to global.
        [[maybe_unused]] const int j = b + sy;
        [[maybe_unused]] const bool jin = ((unsigned)j < (unsigned)WY);
        int gy = gy0n + b;
        if (gy >= A.L) gy -= A.L;
        for (int a = lane; a < dwx; a += 32) {
            [[maybe_unused]] const int i = a + sx;
            [[maybe_unused]] const bool iin = ((unsigned)i < (unsigned)WX);
            // STAGED: read phi^{n+1} out of the staged rect and store it.
            // LARGE:  P2 stored it, and the frame pass zeroed the rest, so the
            //         value is already in global and is re-read here. A float
            //         stored and loaded back is bit-for-bit the same value, so
            //         pn is exactly what the staged path would have held --
            //         including the 0.0f of the frame, which the staged path
            //         gets from the (jin && iin) predicate instead.
            float pn;
            if constexpr (kStagesS<CLS>) {
                pn = (jin && iin) ? __uint_as_float(S_s[j * WX + i]) : 0.0f;
                __stcs(&tile_out[(size_t)(dty0 + b) * kTilePitch + dtx0 + a], pn);
            } else {
                pn = tile_out[(size_t)(dty0 + b) * kTilePitch + dtx0 + a];
            }

            if (!isfinite(pn)) PF_FATAL_OR(A.flags, FLAG_NONFINITE);
            if (pn * pn > kQClampPhiSq) PF_FATAL_OR(A.flags, FLAG_Q_CLAMP);
            const int pb = (int)__float_as_uint(fabsf(pn));
            pmaxb = max(pmaxb, pb);

            const uint32_t q = q_of(pn);
            if (q) {                       // adding 0 is a no-op: bit-exact skip
                int gx = gx0n + a;
                if (gx >= A.L) gx -= A.L;
                const uint32_t old = atomicAdd(&A.S_sc[(size_t)gy * A.P + gx], q);
                if (old > 0xFFFFFFFFu - q) PF_FATAL_OR(A.flags, FLAG_S_OVERFLOW);
            }

            // V(phi^{n+1}) is accumulated over the frame actually stored, so
            // it is exactly consistent with the field the next step reads.
            const double d = (double)pn * (double)pn;
            aV  += d;
            aCx += d * (double)a;
            aCy += d * (double)b;

            if (pn > kSupportEps) {
                blox = min(blox, a); bhix = max(bhix, a);
                bloy = min(bloy, b); bhiy = max(bhiy, b);
            }

            if (fm) {
                float e = 0.0f, w = 0.0f, nn = 0.0f, ss = 0.0f;
                if constexpr (kStagesS<CLS>) {
                    if (jin) {
                        if ((unsigned)(i + 1) < (unsigned)WX) e = __uint_as_float(S_s[j * WX + i + 1]);
                        if ((unsigned)(i - 1) < (unsigned)WX) w = __uint_as_float(S_s[j * WX + i - 1]);
                    }
                    if (iin) {
                        if ((unsigned)(j + 1) < (unsigned)WY) nn = __uint_as_float(S_s[(j + 1) * WX + i]);
                        if ((unsigned)(j - 1) < (unsigned)WY) ss = __uint_as_float(S_s[(j - 1) * WX + i]);
                    }
                } else {
                    // |grad phi| of the STORED frame, clamped at the
                    // destination window with the I1 zero ring -- consistent
                    // with how the Laplacian and the Ix/Iy gradients are taken
                    // everywhere else. Safe to read: every destination pixel
                    // was written before the __syncthreads() that ended P2.
                    //
                    // This is the ONE place the large path is not bitwise what
                    // a (hypothetical) staged version of the same class would
                    // be, and it is a genuine information loss, flagged rather
                    // than papered over -- the same one the split path carries,
                    // for the same reason. The staged path reads these four
                    // neighbours out of the SOURCE rect, so on the border it can
                    // pick up a source pixel that lies OUTSIDE the destination
                    // window and is therefore never stored anywhere: at
                    // a == dwx-1 when dwx+sx < WX (any sx < 0 on an ordinary
                    // step), symmetrically at a == 0 when sx > 0, and the same
                    // on b/sy. The large path has no such pixel to read and
                    // uses 0, exactly as the I1 ring does.
                    //
                    // Affected: `perim` ONLY -- pn, V, Cx, Cy, the bbox,
                    // phi_max and the scatter all use pn alone and are
                    // bit-identical. perim is diagnostic: it never enters the
                    // RHS or v_n. It bites only on full-moment steps, only on
                    // the 1-px border of the window, where phi < kSupportEps
                    // unless FLAG_SUPPORT_CLIP is already firing.
                    const size_t row = (size_t)(dty0 + b) * kTilePitch + dtx0;
                    if (a + 1 < dwx) e = tile_out[row + a + 1];
                    if (a - 1 >= 0)  w = tile_out[row + a - 1];
                    if (b + 1 < dwy)
                        nn = tile_out[(size_t)(dty0 + b + 1) * kTilePitch + dtx0 + a];
                    if (b - 1 >= 0)
                        ss = tile_out[(size_t)(dty0 + b - 1) * kTilePitch + dtx0 + a];
                }
                const float pgx = 0.5f * (e - w), pgy = 0.5f * (nn - ss);
                aPer += (double)sqrtf(pgx * pgx + pgy * pgy);
            }
        }
    }

    // ---- P3b: same fixed-order fp64 tree; bbox and |phi|max by integer
    // min/max (associative, commutative and exact, so order is irrelevant).
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        aV   += __shfl_down_sync(0xFFFFFFFFu, aV,   d);
        aCx  += __shfl_down_sync(0xFFFFFFFFu, aCx,  d);
        aCy  += __shfl_down_sync(0xFFFFFFFFu, aCy,  d);
        aPer += __shfl_down_sync(0xFFFFFFFFu, aPer, d);
        blox  = min(blox, __shfl_down_sync(0xFFFFFFFFu, blox, d));
        bhix  = max(bhix, __shfl_down_sync(0xFFFFFFFFu, bhix, d));
        bloy  = min(bloy, __shfl_down_sync(0xFFFFFFFFu, bloy, d));
        bhiy  = max(bhiy, __shfl_down_sync(0xFFFFFFFFu, bhiy, d));
        pmaxb = max(pmaxb, __shfl_down_sync(0xFFFFFFFFu, pmaxb, d));
    }
    if (lane == 0) {
        double* rw = red_s + warp * kRedSlots;
        rw[0] = aV; rw[1] = aCx; rw[2] = aCy; rw[3] = aPer;
        int* iw = reinterpret_cast<int*>(rw + 4);
        iw[0] = blox; iw[1] = bhix; iw[2] = bloy; iw[3] = bhiy; iw[4] = pmaxb;
    }
    __syncthreads();
    if (tid == 0) {
        double sV = 0.0, sCx = 0.0, sCy = 0.0, sPer = 0.0;
        int Blox = dwx, Bhix = -1, Bloy = dwy, Bhiy = -1, Pmax = 0;
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            const double* rw = red_s + w * kRedSlots;
            sV += rw[0]; sCx += rw[1]; sCy += rw[2]; sPer += rw[3];
            const int* iw = reinterpret_cast<const int*>(rw + 4);
            Blox = min(Blox, iw[0]); Bhix = max(Bhix, iw[1]);
            Bloy = min(Bloy, iw[2]); Bhiy = max(Bhiy, iw[3]);
            Pmax = max(Pmax, iw[4]);
        }
        CellState* cs = &A.cell[n];
        cs->gx0 = gx0n;  cs->gy0 = gy0n;
        cs->cls = (uint8_t)dcls;
        cs->cls_written[A.parity_out] = (uint8_t)dcls;
        cs->theta = bcf[16];
        cs->V = sV;  cs->Cx = sCx;  cs->Cy = sCy;
        if (fm) cs->perim = sPer;
        cs->bb_lo_x = Blox;  cs->bb_hi_x = Bhix;
        cs->bb_lo_y = Bloy;  cs->bb_hi_y = Bhiy;
        cs->promote_ctr = (uint32_t)bci[18];
        cs->shift_ctr  += (sx | sy) ? 1u : 0u;
        cs->tumble_ctr += (uint32_t)bci[19];
        cs->phi_max = __uint_as_float((uint32_t)Pmax);
        A.cell_cls[n] = (uint8_t)dcls;
        if (Bhix >= 0 && (Blox == 0 || Bhix == dwx - 1 ||
                          Bloy == 0 || Bhiy == dwy - 1))
            PF_ADVISORY_ADD(A.flags, FLAG_SUPPORT_CLIP);
    }
    __syncthreads();   // smem is reused by the next cell in this CTA
}

// ---------------------------------------------------------------------------
// k_step -- the only kernel in the steady-state loop.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kBlockThreads, 1)
void k_step(PF_GRID_CONSTANT const StepArgs A)
{
    // Declared as uint4 so the 16 B alignment the cp.async destinations require
    // comes from the type system rather than from an attribute whose placement
    // after `extern` is compiler-dependent. With no static __shared__ in this
    // kernel the dynamic region also starts at the shared-memory base.
    extern __shared__ uint4 smem_raw[];
    char* smem = reinterpret_cast<char*>(smem_raw);
    int* ctrl = reinterpret_cast<int*>(smem + kRedBytes);

    const int tid = (int)threadIdx.x;

    // ---- Phase C: clear-ahead of S[(step+2)%3]. Issued FIRST so the stores
    // retire under the first cell's load latency. Race-free: this kernel
    // neither reads nor scatters into that buffer, and the kernel boundary
    // orders it against the launch that will. st.global.cs so the clear cannot
    // evict the L2-pinned read buffer.
    if (A.clear_ahead_words > 0ull) {
        uint4* dst = reinterpret_cast<uint4*>(A.S_cl);
        const size_t nvec   = (size_t)(A.clear_ahead_words >> 2);
        const size_t stride = (size_t)gridDim.x * (size_t)kBlockThreads;
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
        for (size_t i = (size_t)blockIdx.x * kBlockThreads + tid; i < nvec; i += stride)
            __stcs(&dst[i], z);
    }

    // The work cursor is monotone within a step and is reset for the NEXT
    // launch here, not by a separate kernel: the two slots alternate with the
    // phi parity and the kernel boundary is the only ordering needed.
    // The device step counter is read from the slot the previous launch wrote
    // and published into the other one, so the value every CTA sees is stable
    // for the whole kernel.
    if (blockIdx.x == 0 && tid == 0) {
        *A.cursor_clear = 0ull;
        *A.step_wr = *A.step_rd + 1ull;
    }
    const unsigned long long step = *A.step_rd;

    for (;;) {
        if (tid == 0) {
            const unsigned long long k = atomicAdd(A.cursor_use, 1ull);
            if (k < (unsigned long long)A.N) {
                const int nn = (int)A.perm[(size_t)k];
                ctrl[kBcastCtrlN]   = nn;
                ctrl[kBcastCtrlCls] = (int)A.cell_cls[nn];
            } else {
                ctrl[kBcastCtrlN]   = -1;
                ctrl[kBcastCtrlCls] = -1;
            }
        }
        __syncthreads();
        const int cls = ctrl[kBcastCtrlCls];
        if (cls < 0) break;
        const int n = ctrl[kBcastCtrlN];
        // EVERY class gets its own case. The previous form ended in
        // `default: process_cell<kClassTall>`, which silently ran a 160x160
        // class-3 cell as a 144x176 class-2 one -- latent only because class 3
        // was never selected. `default:` is now unreachable by construction
        // (cell_cls is only ever written from a validated dcls) and is a loud,
        // counted refusal rather than a wrong geometry.
        static_assert(kNumClasses == 5,
                      "k_step's dispatch switch must enumerate every class");
        switch (cls) {
            case kClassRound: process_cell<kClassRound>(n, A, smem, step); break;
            case kClassWide:  process_cell<kClassWide >(n, A, smem, step); break;
            case kClassTall:  process_cell<kClassTall >(n, A, smem, step); break;
            case kClassBig:   process_cell<kClassBig  >(n, A, smem, step); break;
            case kClassLarge: process_cell<kClassLarge>(n, A, smem, step); break;
            default:
                if (tid == 0) atomicAdd(&A.flags[FLAG_CLASS_UNSUPPORTED], 1u);
                // process_cell ends with a __syncthreads() that protects ctrl
                // from the next iteration's tid-0 write; the skip path has to
                // supply it too. cls is a broadcast word, so this is uniform.
                __syncthreads();
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// Initialisation / bootstrap
// ---------------------------------------------------------------------------
__global__ void k_zero_u32(uint32_t* p, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = 0u;
}

__global__ void k_init_tiles(float* phi_a, float* phi_b, CellState* cell,
                             const uint8_t* clsv, int N, int L,
                             const float* seed_cx, const float* seed_cy,
                             float radius_eff, float kappa_iface)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int cls = (int)clsv[n];
    const ShapeClass sc = class_of(cls);
    const int wx = sc.wx,  wy = sc.wy;
    const int tx0 = sc.tx0, ty0 = sc.ty0;

    float* ta = phi_a + (size_t)n * kTileArea;
    float* tb = phi_b + (size_t)n * kTileArea;
    for (int i = (int)threadIdx.x; i < kTileArea; i += (int)blockDim.x) {
        ta[i] = 0.0f;
        tb[i] = 0.0f;
    }
    __syncthreads();

    const float cx = seed_cx[n], cy = seed_cy[n];
    const int gx0 = wrapi((int)lrintf(cx - 0.5f * (float)(wx - 1)), L);
    const int gy0 = wrapi((int)lrintf(cy - 0.5f * (float)(wy - 1)), L);

    for (int p = (int)threadIdx.x; p < wx * wy; p += (int)blockDim.x) {
        const int a = p % wx, b = p / wx;
        float dx = (float)(gx0 + a) - cx;
        float dy = (float)(gy0 + b) - cy;
        if (dx >  0.5f * L) dx -= (float)L;
        if (dx < -0.5f * L) dx += (float)L;
        if (dy >  0.5f * L) dy -= (float)L;
        if (dy < -0.5f * L) dy += (float)L;
        const float r = sqrtf(dx * dx + dy * dy);
        ta[(size_t)(ty0 + b) * kTilePitch + tx0 + a] =
            0.5f * (1.0f - tanhf(kappa_iface * (r - radius_eff)));
    }

    if (threadIdx.x == 0) {
        CellState* cs = &cell[n];
        cs->gx0 = gx0;  cs->gy0 = gy0;
        cs->cls = (uint8_t)cls;
        cs->cls_written[0] = (uint8_t)cls;
        cs->cls_written[1] = (uint8_t)cls;
    }
}

// Populates V, Cx, Cy and the support bbox so that step 0's volume term and
// recentring decision are valid. Uses the same 2-D walk and the same
// fixed-order fp64 tree as P3/P3b.
__global__ __launch_bounds__(kBlockThreads, 1)
void k_init_moments(CellState* cell, const uint8_t* clsv,
                    const float* phi, int N)
{
    __shared__ double red_s[kWarpsPerBlock * kRedSlots];
    const int n = blockIdx.x;
    if (n >= N) return;
    const int cls = (int)clsv[n];
    const ShapeClass sc = class_of(cls);
    const int wx = sc.wx,  wy = sc.wy;
    const int tx0 = sc.tx0, ty0 = sc.ty0;
    const float* t = phi + (size_t)n * kTileArea;

    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;

    double aV = 0.0, aCx = 0.0, aCy = 0.0;
    int blox = wx, bhix = -1, bloy = wy, bhiy = -1, pmaxb = 0;
    for (int b = warp; b < wy; b += kWarpsPerBlock) {
        for (int a = lane; a < wx; a += 32) {
            const float pn = t[(size_t)(ty0 + b) * kTilePitch + tx0 + a];
            const double d = (double)pn * (double)pn;
            aV += d; aCx += d * (double)a; aCy += d * (double)b;
            pmaxb = max(pmaxb, (int)__float_as_uint(fabsf(pn)));
            if (pn > kSupportEps) {
                blox = min(blox, a); bhix = max(bhix, a);
                bloy = min(bloy, b); bhiy = max(bhiy, b);
            }
        }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        aV  += __shfl_down_sync(0xFFFFFFFFu, aV,  d);
        aCx += __shfl_down_sync(0xFFFFFFFFu, aCx, d);
        aCy += __shfl_down_sync(0xFFFFFFFFu, aCy, d);
        blox = min(blox, __shfl_down_sync(0xFFFFFFFFu, blox, d));
        bhix = max(bhix, __shfl_down_sync(0xFFFFFFFFu, bhix, d));
        bloy = min(bloy, __shfl_down_sync(0xFFFFFFFFu, bloy, d));
        bhiy = max(bhiy, __shfl_down_sync(0xFFFFFFFFu, bhiy, d));
        pmaxb = max(pmaxb, __shfl_down_sync(0xFFFFFFFFu, pmaxb, d));
    }
    if (lane == 0) {
        double* rw = red_s + warp * kRedSlots;
        rw[0] = aV; rw[1] = aCx; rw[2] = aCy;
        int* iw = reinterpret_cast<int*>(rw + 4);
        iw[0] = blox; iw[1] = bhix; iw[2] = bloy; iw[3] = bhiy; iw[4] = pmaxb;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double sV = 0.0, sCx = 0.0, sCy = 0.0;
        int Blox = wx, Bhix = -1, Bloy = wy, Bhiy = -1, Pmax = 0;
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            const double* rw = red_s + w * kRedSlots;
            sV += rw[0]; sCx += rw[1]; sCy += rw[2];
            const int* iw = reinterpret_cast<const int*>(rw + 4);
            Blox = min(Blox, iw[0]); Bhix = max(Bhix, iw[1]);
            Bloy = min(Bloy, iw[2]); Bhiy = max(Bhiy, iw[3]);
            Pmax = max(Pmax, iw[4]);
        }
        CellState* cs = &cell[n];
        cs->V = sV; cs->Cx = sCx; cs->Cy = sCy; cs->perim = 0.0;
        cs->bb_lo_x = Blox; cs->bb_hi_x = Bhix;
        cs->bb_lo_y = Bloy; cs->bb_hi_y = Bhiy;
        cs->phi_max = __uint_as_float((uint32_t)Pmax);
        cs->promote_ctr = 0u;
        cs->Ix = 0.0; cs->Iy = 0.0;
    }
}

__global__ void k_scatter_all(const float* phi, const CellState* cell,
                              const uint8_t* clsv, uint32_t* S,
                              int N, int L, int P, uint32_t* flags)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int cls = (int)clsv[n];
    const ShapeClass sc = class_of(cls);
    const int wx = sc.wx,  wy = sc.wy;
    const int tx0 = sc.tx0, ty0 = sc.ty0;
    const CellState cs = cell[n];
    const float* t = phi + (size_t)n * kTileArea;
    for (int p = (int)threadIdx.x; p < wx * wy; p += (int)blockDim.x) {
        const int a = p % wx, b = p / wx;
        const float pn = t[(size_t)(ty0 + b) * kTilePitch + tx0 + a];
        const uint32_t q = q_of(pn);
        if (!q) continue;
        const int gx = wrapi(cs.gx0 + a, L);
        const int gy = wrapi(cs.gy0 + b, L);
        const uint32_t old = atomicAdd(&S[(size_t)gy * P + gx], q);
        if (old > 0xFFFFFFFFu - q) atomicOr(&flags[FLAG_S_OVERFLOW], 1u);
        if (pn * pn > kQClampPhiSq) atomicOr(&flags[FLAG_Q_CLAMP], 1u);
    }
}

// ---------------------------------------------------------------------------
// Debug / observability
// ---------------------------------------------------------------------------
__global__ void k_verify_cells(const float* phi, const CellState* cell,
                               const uint8_t* clsv, int N,
                               double* out_V, float* out_outside_max)
{
    __shared__ double vred[kWarpsPerBlock];
    __shared__ float  ored[kWarpsPerBlock];
    const int n = blockIdx.x;
    if (n >= N) return;
    const int cls = (int)clsv[n];
    const ShapeClass sc = class_of(cls);
    const int wx = sc.wx,  wy = sc.wy;
    const int tx0 = sc.tx0, ty0 = sc.ty0;
    const float* t = phi + (size_t)n * kTileArea;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int nwarp = (int)blockDim.x >> 5;

    double v = 0.0;
    float  omax = 0.0f;
    for (int i = (int)threadIdx.x; i < kTileArea; i += (int)blockDim.x) {
        const int x = i % kTilePitch, y = i / kTilePitch;
        const float p = t[i];
        const bool inside = (x >= tx0 && x < tx0 + wx && y >= ty0 && y < ty0 + wy);
        if (inside) v += (double)p * (double)p;
        else        omax = fmaxf(omax, fabsf(p));
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        v    += __shfl_down_sync(0xFFFFFFFFu, v, d);
        omax  = fmaxf(omax, __shfl_down_sync(0xFFFFFFFFu, omax, d));
    }
    if (lane == 0) { vred[warp] = v; ored[warp] = omax; }
    __syncthreads();
    if (threadIdx.x == 0) {
        double sv = 0.0; float so = 0.0f;
        for (int w = 0; w < nwarp; ++w) { sv += vred[w]; so = fmaxf(so, ored[w]); }
        out_V[n] = sv;
        out_outside_max[n] = so;
    }
}

__global__ void k_verify_S(const uint32_t* S, size_t n, uint32_t* out_max) {
    uint32_t m = 0u;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        m = max(m, S[i]);
    atomicMax(out_max, m);
}

__global__ void k_pack_traj(const CellState* cell, const uint8_t* clsv,
                            TrajPackedCell* out, int N, int L)
{
    const int n = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (n >= N) return;
    const CellState cs = cell[n];
    const int cls = (int)clsv[n];
    const double V = cs.V > 0.0 ? cs.V : 1.0;
    const double cx = (double)cs.gx0 + cs.Cx / V;
    const double cy = (double)cs.gy0 + cs.Cy / V;
    TrajPackedCell t;
    t.global_id = cs.global_id;
    t.cls   = cls;
    t.cx    = (float)(cx - floor(cx / (double)L) * (double)L);
    t.cy    = (float)(cy - floor(cy / (double)L) * (double)L);
    t.vx    = cs.vx;   t.vy = cs.vy;
    t.theta = cs.theta;
    t.volume = (float)cs.V;
    t.perim  = (float)cs.perim;
    t.gamma  = cs.gamma;
    t.v_A    = cs.v_A;
    t.phi_max = cs.phi_max;
    out[n] = t;                    // host-pinned: written over NVLink-C2C
}

__global__ void k_morton_sort(const CellState* cell, uint32_t* perm,
                              int N, int M, int L)
{
    extern __shared__ unsigned long long ks[];
    for (int i = (int)threadIdx.x; i < M; i += (int)blockDim.x) {
        unsigned long long k = 0xFFFFFFFFFFFFFFFFull;
        if (i < N) {
            const CellState cs = cell[i];
            const double V = cs.V > 0.0 ? cs.V : 1.0;
            const int cx = wrapi(cs.gx0 + (int)(cs.Cx / V), L);
            const int cy = wrapi(cs.gy0 + (int)(cs.Cy / V), L);
            k = ((unsigned long long)morton2d((uint32_t)cx, (uint32_t)cy) << 32)
              | (unsigned long long)(uint32_t)i;
        }
        ks[i] = k;
    }
    __syncthreads();
    for (int k = 2; k <= M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = (int)threadIdx.x; i < M; i += (int)blockDim.x) {
                const int q = i ^ j;
                if (q > i) {
                    const bool up = ((i & k) == 0);
                    if ((ks[i] > ks[q]) == up) {
                        const unsigned long long tmp = ks[i];
                        ks[i] = ks[q];
                        ks[q] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = (int)threadIdx.x; i < N; i += (int)blockDim.x)
        perm[i] = (uint32_t)(ks[i] & 0xFFFFFFFFull);
}

// ---------------------------------------------------------------------------
// Host-side launch helpers
// ---------------------------------------------------------------------------
void configure_k_step_smem() {
    cudaFuncSetAttribute(reinterpret_cast<const void*>(k_step),
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    cudaFuncSetAttribute(reinterpret_cast<const void*>(k_step),
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

void configure_morton_smem(int smem_bytes) {
    cudaFuncSetAttribute(reinterpret_cast<const void*>(k_morton_sort),
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
}

int k_step_grid(int device) {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    return sms > 0 ? sms : 1;
}

void launch_step(const StepArgs& A, int grid, cudaStream_t stream,
                 const void* l2_base, size_t l2_bytes, float l2_hit_ratio)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid, 1, 1);
    cfg.blockDim = dim3((unsigned)kBlockThreads, 1, 1);
    cfg.dynamicSmemBytes = (size_t)kSmemBytes;
    cfg.stream = stream;

    cudaLaunchAttribute attr[1];
    int nattr = 0;
    if (l2_base != nullptr && l2_bytes > 0) {
        attr[0].id = cudaLaunchAttributeAccessPolicyWindow;
        attr[0].val.accessPolicyWindow.base_ptr = const_cast<void*>(l2_base);
        attr[0].val.accessPolicyWindow.num_bytes = l2_bytes;
        attr[0].val.accessPolicyWindow.hitRatio = l2_hit_ratio;
        attr[0].val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr[0].val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        nattr = 1;
    }
    cfg.attrs = nattr ? attr : nullptr;
    cfg.numAttrs = (unsigned)nattr;

    cudaLaunchKernelEx(&cfg, k_step, A);
}

// ---------------------------------------------------------------------------
// Runtime occupancy report.
//
// ctas_per_sm is the driver's answer, which accounts for registers, shared
// memory and the MaxDynamicSharedMemorySize opt-in together -- so the
// configure_* helpers must have run first. reg_limited_ctas is the register
// ceiling computed by hand from the same numbers, printed alongside so it is
// immediately visible WHICH resource is binding when the two disagree.
// ---------------------------------------------------------------------------
bool query_kernel_stats(const void* fn, int block_threads, int dynamic_smem,
                        int device, KernelStats* out)
{
    if (out == nullptr) return false;
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, fn) != cudaSuccess) return false;

    int blocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, fn, block_threads, (size_t)dynamic_smem) != cudaSuccess)
        return false;

    int max_threads = 0;
    if (cudaDeviceGetAttribute(&max_threads,
                               cudaDevAttrMaxThreadsPerMultiProcessor,
                               device) != cudaSuccess || max_threads <= 0)
        max_threads = kMaxThreadsPerSmSm90;

    out->regs         = fa.numRegs;
    out->local_bytes  = (size_t)fa.localSizeBytes;
    out->static_smem  = (int)fa.sharedSizeBytes;
    out->dynamic_smem = dynamic_smem;
    out->ctas_per_sm  = blocks;
    out->reg_limited_ctas =
        (fa.numRegs > 0 && block_threads > 0)
            ? kRegsPerSmSm90 / (fa.numRegs * block_threads) : 0;
    out->warps_per_sm = blocks * block_threads / 32;
    out->occupancy    = (double)(blocks * block_threads) / (double)max_threads;
    return true;
}

}  // namespace pf
