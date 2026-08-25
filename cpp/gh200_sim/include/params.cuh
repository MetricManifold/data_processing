#pragma once
// ===========================================================================
// FUSE-1R -- GH200 phase-field cell engine.
//
// SINGLE SOURCE OF TRUTH for every physics coefficient and every piece of
// tile/rect geometry in this tree. Nothing else in the repository is allowed
// to spell out 30, 60, or 120: the previous codebase carried a factor-of-2
// error in the repulsion coefficient for eight months precisely because the
// constant was duplicated across three call sites.
// ===========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace pf {

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Physics: Palmieri et al. 2015, Sci Rep 5:11745, Eq. (S15). Mobility M = 1/2.
//
// Every helper returns the coefficient AS IT APPEARS IN dphi/dt -- the M = 1/2
// factor is already folded in. Do not rescale at the call site.
//
//   dphi_n/dt = gamma_n * lap(phi_n)
//             - bulk_coeff(lambda) * gamma_n * phi(1-phi)(1-2phi)
//             + volume_coeff(mu, A0) * (A0 - V_n) * phi
//             - interaction_coeff(kappa, lambda) * phi_n * sum_{m!=n} phi_m^2
//             - (vx * dphi/dx + vy * dphi/dy)
//
//   v_n = v_A * p_hat_n
//       + motility_coeff(kappa, xi, lambda)
//         * integral( phi_n * grad(phi_n) * sum_{m!=n} phi_m^2 dA )
//
// kNumerInteraction is 60, not 30: Eq. (10)'s interaction free energy sums
// over ORDERED pairs, so dF_int/dphi_n = 120*kappa/lambda^2, and x(-1/2)
// yields 60. The convention-free invariant that catches a mistake here is
//
//     interaction_coeff / motility_coeff == xi
//
// enforced at compile time below and again at startup in validate().
// ---------------------------------------------------------------------------
constexpr int kNumerBulk        = 30;
constexpr int kNumerInteraction = 60;
constexpr int kNumerVolume      = 2;

static_assert(kNumerInteraction == 2 * kNumerBulk,
              "the interaction numerator must be exactly twice the bulk "
              "numerator (60 = 2*30): Eq. (10) sums over ordered pairs");

template <typename T>
__host__ __device__ constexpr T bulk_coeff(T lambda) {
    return T(kNumerBulk) / (lambda * lambda);
}

template <typename T>
__host__ __device__ constexpr T interaction_coeff(T kappa, T lambda) {
    return T(kNumerInteraction) * kappa / (lambda * lambda);
}

// Defined as interaction/xi so the invariant holds structurally, not by luck.
template <typename T>
__host__ __device__ constexpr T motility_coeff(T kappa, T xi, T lambda) {
    return interaction_coeff(kappa, lambda) / xi;
}

template <typename T>
__host__ __device__ constexpr T volume_coeff(T mu, T area0) {
    return T(kNumerVolume) * mu / area0;
}

template <typename T>
__host__ __device__ constexpr T target_area(T radius) {
    return T(kPi) * radius * radius;
}

// Stationary interface of phi'' = (30/lambda^2) phi(1-phi)(1-2phi) is
//   phi(d) = (1 - tanh(k d))/2,  k = sqrt(30)/(2 lambda) = sqrt(7.5)/lambda.
// The phi^2 tail therefore decays as exp(-4 k d) = exp(-1.565 d); the phi tail
// as exp(-2 k d) = exp(-0.7825 d). (The commonly quoted exp(-0.404 d) belongs
// to a different profile normalisation and is wrong by 1.94x here.)
template <typename T>
__host__ __device__ inline T interface_k(T lambda) {
    return T(2.7386127875258306) / lambda;   // sqrt(7.5)
}

// Radius at which to seed the tanh profile so that integral(phi^2 dA) starts
// at pi R^2 rather than pi R^2 - pi R / k. Expanding
//   V = 2 pi/k^2 integral (f(u) - H(-u)) (kR + u) du + pi R^2,
//   f(u) = ((1 - tanh u)/2)^2,   integral (f - H(-u)) du = -1/2  (exact),
// gives V ~ pi R^2 - pi R / k, i.e. R_eff = R + 1/(2k) to leading order.
template <typename T>
__host__ __device__ inline T init_radius(T radius, T lambda) {
    return radius + T(0.5) / interface_k(lambda);
}

// --- compile-time enforcement of interaction/motility == xi -----------------
// Exact double equality is not available after two roundings, so the check is
// a constexpr relative-tolerance test at 1e-12 (six orders of magnitude
// tighter than the factor-of-2 class of bug it exists to catch).
namespace detail {
constexpr bool ratio_is_xi(double kappa, double xi, double lambda) {
    const double a = interaction_coeff(kappa, lambda);
    const double b = motility_coeff(kappa, xi, lambda);
    const double r = a / b;
    return r > xi * (1.0 - 1e-12) && r < xi * (1.0 + 1e-12);
}
static_assert(ratio_is_xi(10.0, 1500.0, 7.0),
              "interaction_coeff/motility_coeff must equal xi (production set)");
static_assert(ratio_is_xi(1.0, 1.0, 1.0), "invariant broken at unit scale");
static_assert(ratio_is_xi(3.7, 977.0, 5.25), "invariant broken off-lattice");
static_assert(ratio_is_xi(0.125, 65536.0, 0.5), "invariant broken at extremes");
}  // namespace detail

// ---------------------------------------------------------------------------
// Numerics. h = dx = dy = 1 throughout; validate() refuses to run otherwise.
//
// 9-point isotropic McLellan Laplacian:
//   lap = (4*(N+S+E+W) + (NE+NW+SE+SW) - 20*c) / (6 h^2)
// Weights sum to 4*4 + 4*1 - 20 = 0, and lap(x^2) = (4*(0+0+1+1) +
// (1+1+1+1) - 0)/6 = 2 exactly. Spectral radius 40/6 = 6.6667.
// ---------------------------------------------------------------------------
constexpr int    kLapEdgeW   = 4;
constexpr int    kLapDiagW   = 1;
constexpr int    kLapCentreW = -20;
constexpr double kLapDenom   = 6.0;

static_assert(4 * kLapEdgeW + 4 * kLapDiagW + kLapCentreW == 0,
              "9-point Laplacian weights must sum to zero");
// f = x^2 about the origin: c=0, E=W=1, N=S=0, NE=NW=SE=SW=1.
static_assert(kLapEdgeW * (0 + 0 + 1 + 1) + kLapDiagW * (1 + 1 + 1 + 1)
                  + kLapCentreW * 0 == 12,
              "9-point Laplacian must give lap(x^2) == 12/6 == 2 exactly at h = 1");
static_assert(kLapDenom == 6.0, "lap(x^2) == 2 requires the denominator 6");

constexpr double kLapSpectralRadius = (4.0 * kLapEdgeW + 4.0 * kLapDiagW
                                       - kLapCentreW) / kLapDenom;   // 40/6

// ---------------------------------------------------------------------------
// Q5.27 unsigned fixed point for the shared field S = sum_m phi_m^2.
//   q = round(phi^2 * 2^27), value = q * 2^-27, range [0, 32), quantum 7.45e-9.
// Each contribution is clamped at phi^2 <= 4 so eight saturated contributors
// are still needed to wrap; overflow is detected exactly (not assumed away)
// from the return value of the atomicAdd and latched in a sticky flag.
// ---------------------------------------------------------------------------
constexpr float  kQScaleF     = 134217728.0f;                 // 2^27
constexpr float  kQInvF       = 7.45058059692382812e-09f;     // 2^-27, exact
constexpr double kQInvD       = 7.45058059692382812e-09;
constexpr float  kQClampPhiSq = 4.0f;

__host__ __device__ __forceinline__ uint32_t q_of(float phi) {
    float s = phi * phi;                 // one multiply: no FMA contraction
    s = fminf(s, kQClampPhiSq);
#if defined(__CUDA_ARCH__)
    return __float2uint_rn(s * kQScaleF);
#else
    return (uint32_t)std::lrint(s * kQScaleF);  // round-to-nearest-even, as _rn
#endif
}

// Support threshold for the bounding box that drives the shape-class decision.
constexpr float kSupportEps = 1e-5f;

// ---------------------------------------------------------------------------
// Tile pool and rect windows.
//
// The phi pool is [N][T][T] floats, T = 256, so every tile row is 1024 B and
// therefore 128 B aligned. The rect window inside a tile is FIXED per shape
// class, forever; recentring is applied by reading shared memory at a shifted
// index, never by moving the window. That makes invariant I1 unconditional:
//
//   I1: outside the current class's window a tile is exactly 0.0f.
//
// I1 is what lets the CTA synthesise the 1-pixel stencil ring in shared memory
// instead of loading a halo, so phi is read exactly WX*WY per cell per step.
//
// Sizing (h = 1, R = 49, lambda = 7): a jammed cell's hexagonal Voronoi
// circumradius is sqrt(A0 / (3 sqrt3 / 2)) = 53.88 px; phi^2 drops below half
// a Q5.27 quantum at 12.40 px past that; plus 1 px of stencil gives a required
// half-width of 67.3. WX = WY = 144 (hw = 72) clears it with 4.7 px of drift
// margin. 128 would clip the phi > 1e-5 support (68.6 px) and is left as a
// post-validation tuning flag only.
//
// The SHORT side of the elongated classes is 144, not less: a class change must
// never shrink the window on the axis that did not trigger it. The elongated
// classes used to be 208x112 / 112x208, and 112 is below the required window
// (2*67.3 = 134.6) and below the phi > 1e-5 support of a relaxed cell
// (2*63.7 = 127.4), so ANY promotion truncated phi on the perpendicular axis,
// cut a ~6e-3 step into the interface profile and destroyed phi^2 mass that
// the volume term then fought to restore. Enforced by static_assert below.
//
// ONE class -- the LARGE one, index kClassLarge -- is exempt from the "phi AND
// S both resident" rule that bounds all the others. It stages phi only: S is
// read pointwise from global, and phi^{n+1} is written straight to global from
// the P2 sweep instead of being staged in S_s (see the
// `if constexpr (kStagesS<CLS>)` branches of process_cell<CLS> in kernels.cu).
// Dropping s_bytes(W,W) from its budget is the ONLY reason a class larger than
// 144 in both axes is possible at all; it fits inside the budget the staged
// classes already set, so it costs zero extra shared memory (static_assert'd
// below). A sixth, rare fallback class uses the complete native tile without
// staging phi or S. Only contact with that physical boundary is exhaustion.
// ---------------------------------------------------------------------------
#if defined(PF_GH200_CMAKE_BUILD) && !defined(PF_EXTENDED_SUPPORT_LAYOUT)
#error "CMake target omitted pf_shape_config: PF_EXTENDED_SUPPORT_LAYOUT is undefined"
#endif
#ifndef PF_EXTENDED_SUPPORT_LAYOUT
#define PF_EXTENDED_SUPPORT_LAYOUT 1
#endif
#if PF_EXTENDED_SUPPORT_LAYOUT != 0 && PF_EXTENDED_SUPPORT_LAYOUT != 1
#error "PF_EXTENDED_SUPPORT_LAYOUT must be exactly 0 or 1"
#endif

// The EXTENDED pair (tile 288, shared-phi class 224) is the default. Class 5
// adds a 286x286 global-memory fallback inside the same tile; it does not alter
// a physical parameter or allocate a larger per-cell field.
//
// Permitting either quantity to change alone would break the aligned-origin /
// zero-ring contract, so the two are selected as one audited pair. This is a
// representation choice only: no physical parameter or update equation depends
// on it.
//
// GPU evidence for the extended pair, Roihu job 687115 (gputest, free queue,
// 2026-08-16), receipt schema pf-n800-extended-gate-v7:
//   - device probe reports tile=288 shared-phi edge=224 on a real GH200;
//   - synthetic supports 201..216 select class 4 with class_exhausted=0;
//   - restart cuts at 1/10/100 steps reload PASS_EXACT on both the ctrl and
//     soft N=800 branches, max_phi_abs_difference 0.0, checkpoint SHA equal.
// NOT established by that gate: a full-length production segment, and general
// (unconstrained) restart parity -- see promote_ctr below and
// EXTENDED_SUPPORT_LAYOUT.md. Build the compact pair with
// -DPF_EXTENDED_SUPPORT_LAYOUT=OFF to reproduce a pre-2026-08-16 geometry.
constexpr bool kExtendedSupportLayout = PF_EXTENDED_SUPPORT_LAYOUT != 0;
constexpr int kTilePitch = kExtendedSupportLayout ? 288 : 256;
constexpr int kLargeClassEdge = kExtendedSupportLayout ? 224 : 208;
constexpr int kTileArea  = kTilePitch * kTilePitch;
static_assert((kTilePitch == 256 && kLargeClassEdge == 208) ||
              (kTilePitch == 288 && kLargeClassEdge == 224),
              "support layout must be one of the two audited tile/class pairs");
static_assert(kTilePitch % 32 == 0,
              "tile rows must remain 128-byte aligned in float storage");

struct ShapeClass {
    int wx, wy, tx0, ty0;
};

// Class 3 (160x160) exists because 1 and 2 each shrink the axis that did NOT
// overflow back to 144, so a cell that grows moderately in BOTH axes fits
// neither. That is not a corner case: at rho=0.89 the hexagonal Voronoi
// circumradius is 57.1 px and the phi>1e-5 tail adds 14.7, so the jammed
// equilibrium support is ~144 -- exactly the promote threshold. It costs
// 213,440 B of shared memory, and it is what sets kSmemRaw.
//
// Class 3 is nevertheless NEVER SELECTED in practice, measured: a cell
// elongates along ONE axis first, promotes to tall (144x176), grows ey freely
// to 168 there, and when ex later crosses 136 it needs wx >= 144 AND wy >= 176
// simultaneously. 160x160 is too short in the long axis, 144x176 too narrow in
// the short one, and 176x176 needs 254,912 B -- over the 232,448 B sm_90
// per-block opt-in maximum. Measured with gamma=0.35 everywhere, N=396,
// rho=0.89, over 1 tau: cls 163/76/157/0, support_clip 2.33% and
// class_exhausted 0.17% of cell-steps, i.e. INVALID. See RESULTS.md 7d.
//
// Class 4 is the largest shared-phi class. It is 208x208 in the compact layout
// and 224x224 in the extended candidate. Their raw footprints are 183,616 B
// and 211,904 B, both below the staged-class maximum of 213,440 B, so neither
// changes the fused launch request (static_assert'd below).
//
// tx0 = ty0 = 32 is forced, not chosen: class_ok() requires the origin to be a
// multiple of 32 (128 B aligned tile rows), at least 1 (the tile's zero ring)
// and to satisfy tx0 + wx <= kTilePitch - 1. The audited pairs are 256/208
// (32 pixels left, 16 right) and 288/224 (32 pixels on both sides).
//
// A 224x224 class is illegal in the compact tile because its aligned origin
// cannot retain the zero ring. Enlarging the tile to 288 makes the pair legal
// without relaxing any execution rule. It raises containable support extent
// after kPromoteSlack from 200 to 216 pixels. Wider supports use class 5's
// global-memory path within the already allocated tile.
constexpr int kNumClasses = 6;
constexpr ShapeClass kClasses[kNumClasses] = {
    {144, 144, 64, 64},   // 0: round
    {176, 144, 32, 64},   // 1: wide
    {144, 176, 64, 32},   // 2: tall
    {160, 160, 32, 32},   // 3: big    (larger than round in BOTH axes)
    {kLargeClassEdge, kLargeClassEdge, 32, 32},
                           // 4: large  (phi only in smem; S read from global)
    {kTilePitch - 2, kTilePitch - 2, 1, 1},
                           // 5: fallback (phi and S read from global)
};

constexpr int kClassRound = 0;
constexpr int kClassWide  = 1;
constexpr int kClassTall  = 2;
constexpr int kClassBig   = 3;
constexpr int kClassLarge = 4;
constexpr int kClassFallback = 5;
static_assert(kNumClasses == 6, "class_of() below enumerates exactly 6 classes");
// k_step's dispatch has an explicit case for every class and a `default:`
// that refuses everything else. `default:` is unreachable by construction and
// exists only to make a corrupt class id a counted refusal rather than a wrong
// geometry.
static_assert(kClassFallback == kNumClasses - 1,
              "the global fallback must be the final shape class");

// The selector may alter only the storage tile and shared-phi edge. Classes 0..3
// and the largest shared-phi class's aligned origin remain pinned.
static_assert(kClasses[0].wx == 144 && kClasses[0].wy == 144 &&
              kClasses[0].tx0 == 64 && kClasses[0].ty0 == 64,
              "support-layout selector changed class 0 geometry");
static_assert(kClasses[1].wx == 176 && kClasses[1].wy == 144 &&
              kClasses[1].tx0 == 32 && kClasses[1].ty0 == 64,
              "support-layout selector changed class 1 geometry");
static_assert(kClasses[2].wx == 144 && kClasses[2].wy == 176 &&
              kClasses[2].tx0 == 64 && kClasses[2].ty0 == 32,
              "support-layout selector changed class 2 geometry");
static_assert(kClasses[3].wx == 160 && kClasses[3].wy == 160 &&
              kClasses[3].tx0 == 32 && kClasses[3].ty0 == 32,
              "support-layout selector changed class 3 geometry");
static_assert(kClasses[kClassLarge].wx == kLargeClassEdge &&
              kClasses[kClassLarge].wy == kLargeClassEdge &&
              kClasses[kClassLarge].tx0 == 32 &&
              kClasses[kClassLarge].ty0 == 32,
              "largest shared-phi class must retain its aligned origin");
static_assert(kClasses[kClassFallback].wx == kTilePitch - 2 &&
              kClasses[kClassFallback].wy == kTilePitch - 2 &&
              kClasses[kClassFallback].tx0 == 1 &&
              kClasses[kClassFallback].ty0 == 1,
              "the fallback must retain the tile's one-pixel stencil ring");

// Does class `c` stage S (and, in P2, phi^{n+1}) in shared memory?
//
// This is THE predicate that selects between the two bodies of process_cell.
// It is a compile-time property of the class, never a runtime branch: the fused
// path takes it through `if constexpr (kStagesS<CLS>)`, so classes 0..3 compile
// to exactly the code they compiled to before class 4 existed.
constexpr bool class_stages_S(int c) { return c >= 0 && c <= kClassBig; }
constexpr bool class_stages_phi(int c) { return c != kClassFallback; }
template <int CLS>
inline constexpr bool kStagesS = class_stages_S(CLS);

// Runtime-indexed access to the class table from DEVICE code. A constexpr
// namespace-scope array cannot be indexed with a runtime value on the device,
// but every member read below has a literal index and is therefore an integral
// constant expression, so the whole thing folds into six immediates.
__host__ __device__ __forceinline__ constexpr ShapeClass class_of(int c) {
    if (c == kClassFallback) return kClasses[kClassFallback];
    return ShapeClass{
        c == 1 ? kClasses[1].wx  : c == 2 ? kClasses[2].wx
               : c == 3 ? kClasses[3].wx  : c == 4 ? kClasses[4].wx
               : kClasses[0].wx,
        c == 1 ? kClasses[1].wy  : c == 2 ? kClasses[2].wy
               : c == 3 ? kClasses[3].wy  : c == 4 ? kClasses[4].wy
               : kClasses[0].wy,
        c == 1 ? kClasses[1].tx0 : c == 2 ? kClasses[2].tx0
               : c == 3 ? kClasses[3].tx0 : c == 4 ? kClasses[4].tx0
               : kClasses[0].tx0,
        c == 1 ? kClasses[1].ty0 : c == 2 ? kClasses[2].ty0
               : c == 3 ? kClasses[3].ty0 : c == 4 ? kClasses[4].ty0
               : kClasses[0].ty0};
}

// All classes use the same safety margin during ordinary promotion.
__host__ __device__ __forceinline__ constexpr bool class_contains_support(
    int c, int ex, int ey, int slack) {
    const ShapeClass sc = class_of(c);
    return ex + slack <= sc.wx && ey + slack <= sc.wy;
}

__host__ __device__ __forceinline__ constexpr int class_support_capacity(
    int c, int slack) {
    return class_of(c).wx - slack;
}

// Smallest-window shape class that contains an (ex, ey) support with `slack`
// pixels to spare on both axes; -1 if even the fallback lacks that margin.
//
// Containment is tested on both axes, never inferred from which extent is
// larger. Choosing the destination from ex >= ey alone moves a cell into a
// class that may be narrower on the axis that did NOT overflow, and P3's store
// then writes 0.0f for every source row/column outside the destination window:
// phi is truncated on a live face, a step discontinuity is cut into the
// interface profile, and the phi^2 mass the volume term is holding at A0
// disappears. When nothing contains the support the caller must report it
// The caller may keep an already-active fallback in its measured no-margin
// regime; it must never clip the field into a smaller class.
//
// __host__ too, and constexpr, because the checkpoint reader has to make
// EXACTLY this decision when it repacks a foreign tile into a native one. Two
// copies of this rule -- one on the device, one in the loader -- would be the
// same class of duplication that put a factor of 2 in the repulsion
// coefficient for eight months.
__host__ __device__ __forceinline__ constexpr int class_containing(int ex, int ey,
                                                                   int slack) {
    int best = -1, best_area = 0;
    for (int c = 0; c < kNumClasses; ++c) {
        const ShapeClass sc = class_of(c);
        const int area = sc.wx * sc.wy;
        if (class_contains_support(c, ex, ey, slack) &&
            (best < 0 || area < best_area)) {
            best = c;
            best_area = area;
        }
    }
    return best;
}

// Checkpoint recovery and an already-active fallback may use the fallback's
// remaining safety margin instead of refusing a representable field. This does
// not enlarge the numerical window: the 286x286 interior and its one-pixel
// tile ring remain unchanged. A support wider than 286 on either axis is true
// storage exhaustion and is still refused.
__host__ __device__ __forceinline__ constexpr int class_containing_storage(
    int ex, int ey, int slack) {
    const int normal = class_containing(ex, ey, slack);
    if (normal >= 0) return normal;
    const ShapeClass fb = class_of(kClassFallback);
    return ex <= fb.wx && ey <= fb.wy ? kClassFallback : -1;
}

// Smallest native class that contains the measured support and every nonzero
// source pixel at the class's canonical tile offset. The checkpoint reader uses
// this to decide whether a foreign tile can be copied without moving a global
// coordinate or discarding even a sub-threshold tail. Source coordinates beyond
// a smaller on-disk tile are exactly zero and need not be represented here.
__host__ __device__ __forceinline__ constexpr int class_preserving_nonzero(
    int ex, int ey, int slack,
    int nz_lo_x, int nz_hi_x, int nz_lo_y, int nz_hi_y) {
    int best = -1, best_area = 0;
    for (int c = 0; c < kNumClasses; ++c) {
        const ShapeClass sc = class_of(c);
        const int area = sc.wx * sc.wy;
        const bool support_fits = c == kClassFallback
            ? ex <= sc.wx && ey <= sc.wy
            : class_contains_support(c, ex, ey, slack);
        const bool nonzero_fits =
            nz_lo_x >= sc.tx0 && nz_hi_x < sc.tx0 + sc.wx &&
            nz_lo_y >= sc.ty0 && nz_hi_y < sc.ty0 + sc.wy;
        if (support_fits && nonzero_fits &&
            (best < 0 || area < best_area)) {
            best = c;
            best_area = area;
        }
    }
    return best;
}

// Class-change thresholds (growth-only hysteresis: promote fast, demote slow).
// Both are margins against the CANDIDATE class's own window, not against a
// fixed 144: the destination is chosen by containment on both axes (see
// class_containing() in kernels.cu), so these read as "the window must hold the
// support with this many pixels to spare".
constexpr int kPromoteSlack   = 8;     // leave a class when extent + 8 > its W
                                       // (= 2 * kMaxShiftPerStep: one worst-case
                                       //  recentring shift on each side)
constexpr int kDemoteSlack    = 20;    // enter a SMALLER class only at +20
constexpr int kDemoteDwell    = 200;   // ... for this many consecutive checks
constexpr int kMaxShiftPerStep = 4;    // |recentring shift| cap on normal steps

// ---------------------------------------------------------------------------
// Block shape and shared-memory layout.
//
// 768 threads = 24 warps, one CTA per cell, grid = numSMs, 1 CTA/SM. The
// memory-level parallelism comes from the 3-stage cp.async pipeline (55.3 KB
// in flight vs the 19.7 KB Little's-law requirement), not from warp count.
//
// smem layout, in order:
//   [0)                 red_s   : double[24][8]   fp64 reduction slots
//                                 (slots 4..7 of each warp are aliased as
//                                  int[8] for the bbox / phi_max reduction)
//   [kRedBytes)         bcast_s : 128 words       CTA-wide scalar broadcast
//   [+kBcastBytes)      (reserved 128 B, formerly mbarriers)
//   [kScalarBytes)      phi_s   : float (WY+2) rows x phi_pitch(WX)
//   [+phi bytes)        S_s     : uint32 WY rows x WX  (no halo: S is
//                                 pointwise, never in a stencil)
//
// The LARGE class (kClassLarge) allocates the first two regions only: it has no
// S_s at all, so its footprint is kScalarBytes + phi_bytes(WX, WY). Its S_s
// pointer is nullptr in that instantiation, so a missed `if constexpr` guard is
// an immediate null dereference rather than a silent overwrite of red_s.
//
// phi_s carries a 4-float left pad rather than the 1 float the stencil needs,
// because 16 B cp.async requires the destination of every copied row to be
// 16 B aligned and a 1-float pad leaves it at +4 B. Rect pixel (x, y) lives at
// phi_s[(y+1)*PX + (kPhiPadLeft + x)]; the synthesised ring occupies row 0,
// row WY+1, column kPhiPadLeft-1 and column kPhiPadLeft+WX.
// ---------------------------------------------------------------------------
constexpr int kBlockThreads  = 768;
constexpr int kWarpsPerBlock = kBlockThreads / 32;    // 24
constexpr int kStripRows     = 16;
constexpr int kPipeStages    = 3;
constexpr int kPhiPadLeft    = 4;

// Broadcast-word assignments. Words 0..23 belong to process_cell's P0; the two
// loop-control words live at the far end so k_step needs no STATIC shared
// memory at all. That matters: with zero static shared allocations the dynamic
// region starts at the shared-memory base and its 16 B alignment (required by
// the cp.async destinations) is guaranteed rather than merely conventional.
constexpr int kBcastCtrlN   = 124;
constexpr int kBcastCtrlCls = 125;

constexpr int kRedSlots   = 8;
constexpr int kRedBytes   = kWarpsPerBlock * kRedSlots * (int)sizeof(double);  // 1536
constexpr int kBcastWords = 128;
constexpr int kBcastBytes = kBcastWords * 4;                                   // 512
constexpr int kMbarBytes  = 128;                                               // reserved
constexpr int kScalarBytes = kRedBytes + kBcastBytes + kMbarBytes;             // 2176

static_assert(kScalarBytes % 16 == 0, "phi_s must land 16 B aligned");

constexpr int align_up(int v, int a) { return ((v + a - 1) / a) * a; }

constexpr int phi_pitch(int wx) { return align_up(wx + kPhiPadLeft + 1, 4); }
constexpr int phi_bytes(int wx, int wy) { return (wy + 2) * phi_pitch(wx) * 4; }
constexpr int s_bytes(int wx, int wy) { return wy * wx * 4; }
constexpr int class_smem(int wx, int wy) {
    return kScalarBytes + phi_bytes(wx, wy) + s_bytes(wx, wy);
}
// The large class's footprint: phi_s and the scalar/reduction region, nothing
// else.
constexpr int class_smem_large(int wx, int wy) {
    return kScalarBytes + phi_bytes(wx, wy);
}
constexpr int class_smem_of(int c) {
    return !class_stages_phi(c) ? kScalarBytes
         : class_stages_S(c) ? class_smem(kClasses[c].wx, kClasses[c].wy)
                             : class_smem_large(kClasses[c].wx, kClasses[c].wy);
}

constexpr int cmax(int a, int b) { return a > b ? a : b; }

// kSmemRaw is the max over ALL classes, and the loop is generic so that adding
// a class cannot silently leave a hand-written cmax() chain behind.
constexpr int smem_raw_all() {
    int m = 0;
    for (int c = 0; c < kNumClasses; ++c) m = cmax(m, class_smem_of(c));
    return m;
}
constexpr int smem_raw_staged_only() {
    int m = 0;
    for (int c = 0; c < kNumClasses; ++c)
        if (class_stages_S(c)) m = cmax(m, class_smem_of(c));
    return m;
}
constexpr int kSmemRaw = smem_raw_all();
constexpr int kSmemBytes = align_up(kSmemRaw, 128);
constexpr int kLargeClassSmemRaw = class_smem_of(kClassLarge);
constexpr int kLargeClassSmemBytes = align_up(kLargeClassSmemRaw, 128);
constexpr int kExpectedLargeClassSmemRaw =
    kExtendedSupportLayout ? 211904 : 183616;

static_assert(kLargeClassSmemRaw == kExpectedLargeClassSmemRaw,
              "largest shared-phi class memory calculation changed");
static_assert(kLargeClassSmemBytes ==
                  (kExtendedSupportLayout ? 211968 : 183680),
              "largest shared-phi class alignment changed");
static_assert(kSmemRaw == 213440 && kSmemBytes == 213504,
              "support-layout selector must not change the staged-class launch "
              "shared-memory request");

// THE load-bearing assertion for the large class. The whole point of not
// staging S is that the largest-window class costs LESS than the staged classes
// already do, so the per-CTA shared-memory request -- and therefore the 1 CTA/SM
// occupancy, the cudaFuncSetAttribute opt-in and every measured timing above --
// is untouched by its existence. If a future edit makes the large class the
// binding constraint, this fires instead of silently costing occupancy.
//   staged max : class 3, 160x160 -> 2,176 + 108,864 + 102,400 = 213,440 B
//   compact    : class 4, 208x208 -> 2,176 + 181,440           = 183,616 B
//   extended   : class 4, 224x224 -> 2,176 + 209,728           = 211,904 B
static_assert(kSmemRaw == smem_raw_staged_only(),
              "the large class raised kSmemRaw: it must fit inside the budget "
              "the staged classes already set, or it is not free");
static_assert(class_smem_of(kClassLarge) < smem_raw_staged_only(),
              "the large class must be strictly cheaper than the staged "
              "classes, otherwise there is no reason for it to skip S_s");
static_assert(class_smem_of(kClassFallback) == kScalarBytes,
              "the global fallback must not reserve a shared phi or S field");

// True per-BLOCK opt-in maximum on sm_90 (cudaDevAttrMaxSharedMemoryPerBlockOptin).
// 233472 B is the per-SM figure and must not be budgeted against.
constexpr int kSmemPerBlockOptinSm90 = 232448;
static_assert(kSmemBytes <= kSmemPerBlockOptinSm90,
              "dynamic shared memory request exceeds the sm_90 per-block "
              "opt-in maximum");
constexpr int kSmemLaunchMarginSm90 = kSmemPerBlockOptinSm90 - kSmemBytes;
constexpr int kLargeClassMarginToStaged = kSmemRaw - kLargeClassSmemRaw;
static_assert(kSmemLaunchMarginSm90 == 18944,
              "unexpected sm_90 per-block shared-memory margin");
static_assert(kLargeClassMarginToStaged ==
                  (kExtendedSupportLayout ? 1536 : 29824),
              "unexpected large-to-staged shared-memory margin");
static_assert(kLargeClassSmemBytes <= kSmemPerBlockOptinSm90,
              "largest shared-phi class exceeds the sm_90 opt-in maximum");

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// sm_90 hardware constants, used by the startup occupancy report.
//
// A --split execution path (k_step_rhs + k_step_post at 512 threads) used to
// live here, trading a third HBM pass over phi for 2-3 CTAs/SM instead of 1.
// It was removed after being measured 46-95% SLOWER than the fused kernel at
// every N from 132 to 2112 on a GH200 (Roihu job 689689, free gputest queue),
// in the ONLY regime it could run: it refused the largest shared-phi class, which
// is exactly where soft cells end up, and its 512-thread reduction made it a
// different trajectory that could not be compared against the fused path
// anyway. `git log` has it if the occupancy idea is ever revisited.
// ---------------------------------------------------------------------------
constexpr int kSmemPerSmSm90       = 233472;
constexpr int kRegsPerSmSm90       = 65536;
constexpr int kMaxThreadsPerSmSm90 = 2048;

// --- per-class structural invariants ---------------------------------------
namespace detail {
constexpr bool class_ok(ShapeClass c) {
    return c.wy % kStripRows == 0            // whole strips, no partial tail
        && c.wx % 4 == 0                     // 16 B cp.async chunks on phi
        && c.tx0 % 32 == 0 && c.ty0 % 32 == 0  // 128 B aligned tile rows
        && c.tx0 >= 1 && c.ty0 >= 1          // room for the zero ring
        && c.tx0 + c.wx <= kTilePitch - 1
        && c.ty0 + c.wy <= kTilePitch - 1
        && phi_pitch(c.wx) % 4 == 0;
}
// Every rule applies to the large class too, unchanged and unrelaxed. In
// particular the 32-alignment of tx0/ty0 is what keeps the cp.async SOURCE
// rows 128 B aligned; the DESTINATION alignment the 16 B copies need comes from
// kScalarBytes % 16 == 0 and phi_pitch(wx) % 4 == 0 (checked below), which hold
// for both audited shared-phi layouts exactly as they do for 144.
static_assert(class_ok(kClasses[0]), "shape class 0 violates a layout rule");
static_assert(class_ok(kClasses[1]), "shape class 1 violates a layout rule");
static_assert(class_ok(kClasses[2]), "shape class 2 violates a layout rule");
static_assert(class_ok(kClasses[3]), "shape class 3 violates a layout rule");
static_assert(class_ok(kClasses[4]), "shape class 4 violates a layout rule");
static_assert(kNumClasses == 6, "add a layout static_assert for the new class");
constexpr bool fallback_class_ok(ShapeClass c) {
    return c.wx == kTilePitch - 2 && c.wy == kTilePitch - 2
        && c.tx0 == 1 && c.ty0 == 1;
}
static_assert(fallback_class_ok(kClasses[kClassFallback]),
              "fallback must leave a one-pixel ring in the native tile");
static_assert(kClasses[0].wx == kClasses[0].wy, "class 0 must be the square one");

// 16 B cp.async destinations: every copied row starts at
// smem + kScalarBytes + ((y+1)*phi_pitch(wx) + kPhiPadLeft)*4, so the byte
// offset is 16 B aligned iff kScalarBytes, phi_pitch(wx)*4 and kPhiPadLeft*4
// all are. Asserted per class rather than argued per class.
constexpr bool cpasync_dst_ok(ShapeClass c) {
    return (kScalarBytes % 16 == 0) && ((phi_pitch(c.wx) * 4) % 16 == 0)
        && ((kPhiPadLeft * 4) % 16 == 0);
}
static_assert(cpasync_dst_ok(kClasses[0]), "class 0 breaks cp.async dst alignment");
static_assert(cpasync_dst_ok(kClasses[1]), "class 1 breaks cp.async dst alignment");
static_assert(cpasync_dst_ok(kClasses[2]), "class 2 breaks cp.async dst alignment");
static_assert(cpasync_dst_ok(kClasses[3]), "class 3 breaks cp.async dst alignment");
static_assert(cpasync_dst_ok(kClasses[4]), "class 4 breaks cp.async dst alignment");

// No class may be SMALLER than the round class in EITHER axis. This is the
// invariant that makes a class change safe: whichever axis triggered the
// change, the other one cannot shrink below the window that gate 5 validates
// against the phi > 1e-5 support extent. Violating it truncates phi on a live
// face -- which is exactly what 208x112 / 112x208 did.
constexpr bool class_not_narrower(ShapeClass c) {
    return c.wx >= kClasses[0].wx && c.wy >= kClasses[0].wy;
}
static_assert(class_not_narrower(kClasses[1]),
              "shape class 1 is narrower than class 0 on one axis: promoting "
              "into it would truncate the support on that axis");
static_assert(class_not_narrower(kClasses[2]),
              "shape class 2 is narrower than class 0 on one axis: promoting "
              "into it would truncate the support on that axis");
static_assert(class_not_narrower(kClasses[3]),
              "shape class 3 is narrower than class 0 on one axis: promoting "
              "into it would truncate the support on that axis");
static_assert(class_not_narrower(kClasses[4]),
              "shape class 4 is narrower than class 0 on one axis: promoting "
              "into it would truncate the support on that axis");
static_assert(class_not_narrower(kClasses[5]),
              "fallback is narrower than class 0");

// The large shared-phi class must dominate classes 0--3; the global fallback
// separately dominates it below.
constexpr bool dominated_by_large(ShapeClass c) {
    return c.wx <= kClasses[kClassLarge].wx && c.wy <= kClasses[kClassLarge].wy;
}
static_assert(dominated_by_large(kClasses[0]), "class 0 is not covered by the large class");
static_assert(dominated_by_large(kClasses[1]), "class 1 is not covered by the large class");
static_assert(dominated_by_large(kClasses[2]), "class 2 is not covered by the large class");
static_assert(dominated_by_large(kClasses[3]), "class 3 is not covered by the large class");

constexpr bool dominated_by_fallback(ShapeClass c) {
    return c.wx <= kClasses[kClassFallback].wx
        && c.wy <= kClasses[kClassFallback].wy;
}
static_assert(dominated_by_fallback(kClasses[kClassLarge]),
              "fallback must contain the largest shared-memory class");

// The large class must also be the LARGEST by area, because class_containing()
// picks the smallest containing class: if some other class had a bigger area,
// the large one would be preferred even where a cheaper class fits, and every
// promoted cell would pay the large path's extra HBM read for nothing.
constexpr bool smaller_area_than_large(ShapeClass c) {
    return c.wx * c.wy < kClasses[kClassLarge].wx * kClasses[kClassLarge].wy;
}
static_assert(smaller_area_than_large(kClasses[0]), "class 0 is not smaller in area than the large class");
static_assert(smaller_area_than_large(kClasses[1]), "class 1 is not smaller in area than the large class");
static_assert(smaller_area_than_large(kClasses[2]), "class 2 is not smaller in area than the large class");
static_assert(smaller_area_than_large(kClasses[3]), "class 3 is not smaller in area than the large class");
static_assert(kClasses[kClassLarge].wx * kClasses[kClassLarge].wy
                  < kClasses[kClassFallback].wx * kClasses[kClassFallback].wy,
              "fallback must be more expensive than the shared-phi classes");
}  // namespace detail

// ---------------------------------------------------------------------------
// Sticky device alarm flags. Any nonzero entry means the run is INVALID, not
// merely slow. There is deliberately no silent clamp behind any of these.
// ---------------------------------------------------------------------------
enum : int {
    FLAG_S_OVERFLOW = 0,   // Q5.27 accumulator wrapped
    FLAG_Q_CLAMP    = 1,   // a single contribution hit phi^2 > 4
    FLAG_SUPPORT_CLIP = 2, // support bbox touched the rect edge
    FLAG_CLASS_EXHAUSTED = 3,  // a cell outgrew the widest shape class
    FLAG_S_NEGATIVE = 4,   // q_S < q_of(phi_n): counted, never floored silently
    FLAG_NONFINITE  = 5,   // phi went non-finite
    FLAG_V_NONPOS   = 6,   // carried V <= 0, recentring skipped
    // A CTA was handed a shape class its execution path cannot process, and
    // SKIPPED the cell rather than running it with the wrong geometry. Raised
    // by the dispatch
    // default, which is unreachable by construction. Counted, and the run is
    // reported INVALID -- the previous behaviour was to fall through `default:`
    // into process_cell<kClassTall>, i.e. to read a 160x160 window as 144x176.
    FLAG_CLASS_UNSUPPORTED = 7,
    FLAG_COUNT      = 8
};
// ---------------------------------------------------------------------------
// Fatal alarms are always enabled. Only the high-frequency support_clip
// advisory is optional; opt it in with -DPF_ALARMS.
//
// support_clip fires when the phi > kSupportEps bounding box merely touches a
// window edge. Direct border-ring measurements put the affected phi^2 mass at
// ~1e-13 of the total, so it remains advisory. class_exhausted is different: no
// available class can contain the support, which means real field truncation.
// It is therefore an always-on sticky atomicOr. Every other non-advisory flag
// is likewise always compiled in and fatal.
// ---------------------------------------------------------------------------
#if defined(PF_ALARMS) && defined(PF_NO_ALARMS)
#error "PF_ALARMS and PF_NO_ALARMS are mutually exclusive; pass at most one."
#endif

#define PF_FATAL_ADD(flags, idx) atomicAdd(&(flags)[idx], 1u)
#define PF_FATAL_OR(flags, idx)  atomicOr(&(flags)[idx], 1u)

#if defined(PF_ALARMS)
#define PF_ADVISORY_ADD(flags, idx) atomicAdd(&(flags)[idx], 1u)
#define PF_SUPPORT_CLIP_ENABLED 1
#else
#define PF_ADVISORY_ADD(flags, idx) ((void)0)
#define PF_SUPPORT_CLIP_ENABLED 0
#endif

// FLAG_COUNT is pinned at 8 because DumpHeader carries uint32_t flags[FLAG_COUNT]
// inline: changing it changes the on-disk dump layout that dump_phi and the
// Python oracle parse. Bump kDumpVersion if that ever has to move.
static_assert(FLAG_CLASS_UNSUPPORTED < FLAG_COUNT, "flag index out of range");

constexpr bool flag_is_fatal(int i) {
    return i >= 0 && i < FLAG_COUNT && i != FLAG_SUPPORT_CLIP;
}
static_assert(!flag_is_fatal(FLAG_SUPPORT_CLIP),
              "support_clip is the sole advisory flag");
static_assert(flag_is_fatal(FLAG_S_OVERFLOW), "S_overflow must stop production");
static_assert(flag_is_fatal(FLAG_Q_CLAMP), "q_clamp must stop production");
static_assert(flag_is_fatal(FLAG_CLASS_EXHAUSTED),
              "class_exhausted must stop production");
static_assert(flag_is_fatal(FLAG_S_NEGATIVE),
              "S_other_negative must stop production");
static_assert(flag_is_fatal(FLAG_NONFINITE), "phi_nonfinite must stop production");
static_assert(flag_is_fatal(FLAG_V_NONPOS), "V_nonpositive must stop production");
static_assert(flag_is_fatal(FLAG_CLASS_UNSUPPORTED),
              "class_unsupported must stop production");

inline const char* flag_name(int i) {
    switch (i) {
        case FLAG_S_OVERFLOW:      return "S_overflow";
        case FLAG_Q_CLAMP:         return "q_clamp";
        case FLAG_SUPPORT_CLIP:    return "support_clip";
        case FLAG_CLASS_EXHAUSTED: return "class_exhausted";
        case FLAG_S_NEGATIVE:      return "S_other_negative";
        case FLAG_NONFINITE:       return "phi_nonfinite";
        case FLAG_V_NONPOS:        return "V_nonpositive";
        case FLAG_CLASS_UNSUPPORTED: return "class_unsupported";
        default:                   return "unused";
    }
}

// ---------------------------------------------------------------------------
// Simulation parameters. Doubles on the host so coefficient derivation and
// time accumulation never lose precision; the kernel takes float copies.
// ---------------------------------------------------------------------------
struct SimParams {
    int    Nx = 0, Ny = 0;          // domain side, pixels (must be equal)
    double dx = 1.0, dy = 1.0;      // must both be exactly 1
    int    num_cells = 288;
    double rho = 0.90;              // packing fraction, sets Nx from N and R
    double dt = 0.01;
    double t_end = 100.0;
    double lambda = 7.0;
    double target_radius = 49.0;
    double kappa = 10.0;
    double mu = 1.0;
    double xi = 1500.0;
    double tau = 1.0e4;
    double v_A = 1.0e-2;
    double gamma_normal = 1.0;
    double gamma_cancer = 0.35;
    double cancer_fraction = 0.0;   // fraction of cells given gamma_cancer
    double v_A_sigma = 0.0;         // lognormal spread on per-cell v_A
    unsigned long long seed = 1234;
    // Polarity RNG stream, INDEPENDENT of the placement seed. 0 = follow `seed`.
    //
    // The matched-pair protocol needs the control and soft branches to share the
    // reorientation sequence while their configurations may differ, and the
    // converse experiment (same packing, different motility noise) needs the
    // opposite. Deriving both from one seed makes those two impossible to
    // separate and silently confounds structural variance with noise variance.
    unsigned long long polarity_seed = 0;
    int    print_interval = 100;
    int    full_moment_every = 100;
    int    verify_every = 4096;

    // Resolved stream for everything polarity: initial theta and every tumble.
    __host__ __device__ unsigned long long polarity_stream() const {
        return polarity_seed ? polarity_seed : seed;
    }

    __host__ __device__ double area0() const {
        return target_area(target_radius);
    }
    __host__ __device__ double bulk() const { return bulk_coeff(lambda); }
    __host__ __device__ double interaction() const {
        return interaction_coeff(kappa, lambda);
    }
    __host__ __device__ double motility() const {
        return motility_coeff(kappa, xi, lambda);
    }
    __host__ __device__ double volume() const { return volume_coeff(mu, area0()); }
    __host__ __device__ double dA() const { return dx * dy; }

    // Per-step Bernoulli tumble probability from P(t_r) = (1/tau) exp(-t_r/tau).
    // MUST be expm1: at tau = 1e4, dt = 0.01, "1 - expf(-dt/tau)" in fp32
    // returns 1.013279e-06 instead of 9.999995e-07 -- a +1.33% bias that turns
    // tau_eff = 10000 into 9869. Computed here in double, on the host, and
    // shipped to the kernel as a double argument.
    double p_tumble() const { return -std::expm1(-dt / tau); }

    // 64-bit, and range-checked: the CLI accepts t_end up to 1e12 and dt down
    // to 1e-12, so t_end/dt reaches 1e24. A float-to-int conversion out of
    // range is undefined -- measured, aarch64 saturates (--t-end 1e9 --dt 0.01
    // silently became 2147483647 steps = t_end 2.1e7), x86 gives INT_MIN and
    // run() then does nothing while reporting "alarms: all clear". Returns -1
    // when the count does not fit; validate() rejects that.
    long long total_steps() const {
        const double n = t_end / dt + 0.5;
        return (n >= 0.0 && n <= 9.0e18) ? (long long)n : -1LL;
    }
};

// Domain side for N cells of area A0 at packing fraction rho.
inline int domain_side_for(int n, double radius, double rho) {
    const double a = (double)n * target_area(radius) / rho;
    return (int)std::ceil(std::sqrt(a));
}

// Row pitch of an S buffer, in uint32: 32-word (128 B) aligned rows.
inline int s_pitch_for(int side) { return 32 * ((side + 31) / 32); }

// ---------------------------------------------------------------------------
// Startup validation. Returns false and prints an actionable message rather
// than asserting, so a bad CLI never silently produces plausible garbage.
// ---------------------------------------------------------------------------
inline bool validate(const SimParams& p) {
    bool ok = true;
    auto fail = [&](const char* msg) {
        std::fprintf(stderr, "[fatal] %s\n", msg);
        ok = false;
    };

    if (!(p.dx == 1.0 && p.dy == 1.0)) {
        std::fprintf(stderr,
            "[fatal] dx and dy must both be exactly 1.0 (got %.17g, %.17g).\n"
            "        The 9-point Laplacian, the 1/(2h) gradients and the\n"
            "        dA = 1 quadrature are all hard-coded for h = 1. Rescale\n"
            "        lambda and R instead of changing h.\n", p.dx, p.dy);
        ok = false;
    }
    if (p.Nx != p.Ny) {
        std::fprintf(stderr,
            "[fatal] Nx must equal Ny (got %d x %d). The shared field S uses a\n"
            "        single side length L for both axes.\n", p.Nx, p.Ny);
        ok = false;
    }
    if (p.num_cells <= 0) fail("num_cells must be positive");
    if (!(p.lambda > 0.0)) fail("lambda must be positive");
    if (!(p.target_radius > 0.0)) fail("radius must be positive");
    if (!(p.dt > 0.0)) fail("dt must be positive");
    if (p.total_steps() < 0) {
        std::fprintf(stderr,
            "[fatal] t_end/dt = %.6g/%.6g does not fit in a 64-bit step count.\n"
            "        Reduce t_end or increase dt (the limit is 9e18 steps).\n",
            p.t_end, p.dt);
        ok = false;
    }
    if (!(p.xi > 0.0)) fail("xi must be positive");
    if (!(p.tau > 0.0)) fail("tau must be positive");
    if (p.kappa < 0.0) fail("kappa must be non-negative");
    if (p.mu < 0.0) fail("mu must be non-negative");
    if (!(p.gamma_normal > 0.0)) fail("gamma must be positive");
    if (!(p.rho > 0.0 && p.rho < 1.0)) fail("rho must lie in (0,1)");

    // The rect must fit strictly inside the domain, otherwise two rect pixels
    // alias onto one global pixel and q_S - q_of(phi_n) stops being an exact
    // self-subtraction.
    int wmax = 0;
    for (int c = 0; c < kNumClasses; ++c) {
        wmax = wmax > kClasses[c].wx ? wmax : kClasses[c].wx;
        wmax = wmax > kClasses[c].wy ? wmax : kClasses[c].wy;
    }
    if (p.Nx <= wmax) {
        std::fprintf(stderr,
            "[fatal] domain side %d must exceed the largest rect dimension %d.\n"
            "        Increase N or rho, or shrink the radius.\n", p.Nx, wmax);
        ok = false;
    }

    // Explicit Euler on the diffusion part: dt * gamma * rho(lap) < 2.
    const double gmax = p.gamma_normal > p.gamma_cancer ? p.gamma_normal
                                                        : p.gamma_cancer;
    const double cfl = p.dt * gmax * kLapSpectralRadius;
    if (cfl >= 1.0) {
        std::fprintf(stderr,
            "[fatal] dt*gamma*rho(lap) = %.4f >= 1; explicit Euler is unstable "
            "(limit 2, refuse above 1). Reduce dt below %.5g.\n",
            cfl, 1.0 / (gmax * kLapSpectralRadius));
        ok = false;
    } else if (cfl > 0.5) {
        std::fprintf(stderr,
            "[warn] dt*gamma*rho(lap) = %.4f; margin to the stability limit is "
            "under 4x.\n", cfl);
    }

    // Convention-free coefficient invariant, re-checked at runtime with the
    // actual parameters (the static_asserts only cover fixed test points).
    const double ratio = p.interaction() / p.motility();
    if (!(std::fabs(ratio - p.xi) <= 1e-9 * p.xi)) {
        std::fprintf(stderr,
            "[fatal] interaction_coeff/motility_coeff = %.17g but xi = %.17g.\n"
            "        This invariant is convention-free: if it fails, one of the\n"
            "        two coefficients has the wrong numerical factor.\n",
            ratio, p.xi);
        ok = false;
    }
    return ok;
}

inline void print_params(const SimParams& p, int side, int pitch) {
    std::printf("--- FUSE-1R configuration ---\n");
    std::printf("  cells            %d\n", p.num_cells);
    std::printf("  domain           %d x %d  (rho = %.4f)\n", side, side, p.rho);
    std::printf("  S pitch          %d uint32  (%.2f MB/buffer, x3)\n",
                pitch, (double)pitch * side * 4.0 / 1048576.0);
    std::printf("  R, lambda        %.4g, %.4g   (init R_eff = %.6g)\n",
                p.target_radius, p.lambda,
                init_radius(p.target_radius, p.lambda));
    std::printf("  dt, t_end        %.6g, %.6g  (%lld steps)\n",
                p.dt, p.t_end, p.total_steps());
    std::printf("  kappa, mu, xi    %.6g, %.6g, %.6g\n", p.kappa, p.mu, p.xi);
    std::printf("  gamma n/c        %.6g / %.6g  (cancer fraction %.4g)\n",
                p.gamma_normal, p.gamma_cancer, p.cancer_fraction);
    std::printf("  v_A, tau         %.6g, %.6g\n", p.v_A, p.tau);
    std::printf("  A0               %.6f\n", p.area0());
    std::printf("  bulk  30/l^2     %.9g\n", p.bulk());
    std::printf("  inter 60k/l^2    %.9g\n", p.interaction());
    std::printf("  motil 60k/(xl^2) %.9g\n", p.motility());
    std::printf("  inter/motil      %.9g   (must equal xi)\n",
                p.interaction() / p.motility());
    std::printf("  vol   2mu/A0     %.9g\n", p.volume());
    std::printf("  p_tumble         %.9e   (-expm1(-dt/tau))\n", p.p_tumble());
    std::printf("  fatal alarms     ENABLED (all non-advisory flags)\n");
    std::printf("  support_clip     %s\n",
                PF_SUPPORT_CLIP_ENABLED
                    ? "ENABLED (-DPF_ALARMS; advisory only)"
                    : "NOT INSTRUMENTED (default; advisory only)");
    std::printf("  support layout   %s  (tile %d, guarded support %d px/axis, "
                "physical interior %d)\n",
                kExtendedSupportLayout
                    ? "EXTENDED (default; support+restart GPU-gated 687115)"
                    : "COMPACT LEGACY (pre-2026-08-16 geometry)",
                kTilePitch,
                class_support_capacity(kClassFallback, kPromoteSlack),
                kClasses[kClassFallback].wx);
    std::printf("  smem/CTA         %d B of %d B opt-in max\n",
                kSmemBytes, kSmemPerBlockOptinSm90);
    std::printf("  large class      %d: %d x %d, %d B raw / %d B aligned "
                "(phi only, S from global) -- staged max %d B, "
                "sm_90 launch margin %d B\n",
                kClassLarge, kClasses[kClassLarge].wx, kClasses[kClassLarge].wy,
                kLargeClassSmemRaw, kLargeClassSmemBytes,
                smem_raw_staged_only(), kSmemLaunchMarginSm90);
    std::printf("  fallback class   %d: %d x %d at (1,1), phi+S from global, "
                "%d B smem/CTA\n",
                kClassFallback, kClasses[kClassFallback].wx,
                kClasses[kClassFallback].wy,
                class_smem_of(kClassFallback));
    std::printf("  exec path        k_step + sparse fallback filter "
                "(%d threads; %d / %d B smem)\n",
                kBlockThreads, kSmemBytes, kScalarBytes);
}

}  // namespace pf
