#pragma once
// ===========================================================================
// FUSE-1R kernel ABI: per-cell state, the single per-step argument block, the
// counter-based RNG, and the kernel / launcher declarations.
//
// One kernel per step. One CTA per cell. The cell's whole rect is resident in
// shared memory, so the EXACT (non-lagged) interaction velocity is available to
// the RHS in the same pass -- the interaction velocity is ~7x v_A, so lagging
// it by a step would change the integrator for the dominant transport term.
// ===========================================================================

#include "params.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) && defined(CUDART_VERSION) && (CUDART_VERSION >= 11070)
#define PF_GRID_CONSTANT __grid_constant__
#else
#define PF_GRID_CONSTANT
#endif

namespace pf {

// ---------------------------------------------------------------------------
// Per-cell state, AoS, one 192 B record per cell.
//
// V, Cx, Cy and the bbox describe the field ABOUT TO BE READ: they were
// accumulated while last step's result was being stored. They are not stale --
// V(phi^n) is exactly what the volume term wants, and the centre of mass is
// exactly current.
//
// alignas is 64, not 128: 192 = 3 x 64, so each record is 64 B aligned and
// spans exactly three 64 B sectors. alignas(128) would round sizeof up to
// 256 B and add 33% to the (small) per-cell traffic for nothing.
// ---------------------------------------------------------------------------
struct alignas(64) CellState {
    int32_t  gx0, gy0;            // global coords of rect pixel (0,0), in [0,L)
    uint8_t  cls;                 // shape class of the field to be read
    uint8_t  cls_written[2];      // class last written into phi[parity]
    uint8_t  pad0;
    int32_t  global_id;           // RNG key; stable across any reordering
    float    gamma, v_A, R_tgt;   // per-cell stiffness and motility (required)
    float    theta;               // polarity angle; p_hat = (cos,sin) => |p|==1
    double   V;                   // sum(phi^2) of the field about to be read
    double   Cx, Cy;              // sum(phi^2 x), sum(phi^2 y), rect coords
    double   perim;               // full-moment steps only
    int32_t  bb_lo_x, bb_hi_x, bb_lo_y, bb_hi_y;   // support bbox, rect coords
    float    vx, vy;              // published, diagnostic
    double   Ix, Iy;              // published, diagnostic
    uint32_t promote_ctr;         // demote-hysteresis dwell counter
    uint32_t shift_ctr;           // number of recentring shifts applied
    uint32_t tumble_ctr;          // number of tumbles
    float    phi_max;             // max |phi| over the stored frame
    uint32_t reserved[18];
};
static_assert(sizeof(CellState) == 192, "CellState must be exactly 192 B");
static_assert(alignof(CellState) == 64, "CellState must be 64 B aligned");

// ---------------------------------------------------------------------------
// Packed observables, written by the GPU straight into host-pinned memory over
// the coherent NVLink-C2C link: no D2H copy, no stream sync on the hot path.
// ---------------------------------------------------------------------------
struct TrajPackedCell {
    int32_t global_id;
    int32_t cls;
    float   cx, cy;          // global centre of mass (periodic-unwrapped rect)
    float   vx, vy;
    float   theta;
    float   volume;
    float   perim;
    float   gamma, v_A;
    float   phi_max;
};

// ---------------------------------------------------------------------------
// The single per-step argument block. Passed by value as a __grid_constant__
// const parameter so it lives in the immediate constant bank and the compiler
// may assume it is not aliased or mutated.
//
// `step` is NOT here: with a CUDA graph replaying a fixed 6-step body forever,
// a baked step number would freeze. Cadence predicates read the device step
// counter *step_rd, which the previous launch wrote; this launch writes
// *step_wr (the other slot), so the read value is stable for the whole kernel.
// ---------------------------------------------------------------------------
struct StepArgs {
    // --- pointers -----------------------------------------------------------
    const float*    phi_in;
    float*          phi_out;
    const uint32_t* S_rd;          // read this step
    uint32_t*       S_sc;          // scatter into, for next step
    uint32_t*       S_cl;          // clear-ahead target
    CellState*      cell;
    uint8_t*        cell_cls;      // shape class per cell, hot 1 B copy
    const uint32_t* perm;          // Morton visit order (identity if disabled)
    unsigned long long* cursor_use;
    unsigned long long* cursor_clear;
    const unsigned long long* step_rd;
    unsigned long long* step_wr;
    uint32_t*       flags;

    // --- geometry -----------------------------------------------------------
    int N;
    int L;                         // domain side
    int P;                         // S row pitch, uint32
    int parity_out;                // which phi buffer is being written (0/1)

    // --- physics (already includes mobility M = 1/2) ------------------------
    float  dt;
    // A0 and vol_scale are double because A0 - V is a difference of two ~7543
    // quantities whose fp32 ulp is 4.9e-4: in fp32 the volume restoring force
    // would be quantised at the level of its own steady-state value.
    double A0;
    double vol_scale;              // 2*mu/A0        -> volC = vol_scale*(A0-V)
    float  bulk_scale;              // 30/lambda^2    -> dwC  = bulk_scale*gamma
    float  rep_coeff;               // 60*kappa/lambda^2
    float  mot_coeff;               // rep_coeff/xi

    // --- rng / cadence ------------------------------------------------------
    unsigned long long seed;
    // Tumble stream. Set from SimParams::polarity_stream(), NOT from `seed`, so
    // two runs can share a reorientation sequence while differing elsewhere --
    // exactly what the matched-pair protocol requires.
    unsigned long long polarity_seed;
    double p_tumble;               // -expm1(-dt/tau), computed in double, host
    int    full_moment_every;
    unsigned long long clear_ahead_words;   // P*L, or 0 to skip phase C
};

// ---------------------------------------------------------------------------
// Philox4x32-10, counter based. Key (seed, global_id), counter (step, 0, 0, 0).
// No RNG state to checkpoint, and the stream is invariant under any reordering
// of cells because the key is the stable global id, not the array index.
// ---------------------------------------------------------------------------
struct Philox4 { uint32_t v[4]; };

__host__ __device__ __forceinline__ uint32_t pf_mulhi(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
    return __umulhi(a, b);
#else
    return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
#endif
}

__host__ __device__ __forceinline__ Philox4 philox4x32_10(
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t k0, uint32_t k1)
{
    const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    const uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    for (int r = 0; r < 10; ++r) {
        if (r > 0) { k0 += W0; k1 += W1; }
        const uint32_t hi0 = pf_mulhi(M0, c0), lo0 = M0 * c0;
        const uint32_t hi1 = pf_mulhi(M1, c2), lo1 = M1 * c2;
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    Philox4 out; out.v[0] = c0; out.v[1] = c1; out.v[2] = c2; out.v[3] = c3;
    return out;
}

// 53-bit uniform on [0,1) -- enough resolution that a p_tumble of 1e-6 is
// resolved to 9 significant digits.
__host__ __device__ __forceinline__ double philox_uniform53(uint32_t a, uint32_t b) {
    const uint64_t m = (((uint64_t)a << 32) | (uint64_t)b) >> 11;
    return (double)m * (1.0 / 9007199254740992.0);   // 2^-53
}

// ---------------------------------------------------------------------------
// Per-cell initial-condition draws.
//
// Counter domains, fixed forever: 1 = placement jitter, 2 = polarity angle,
// 3 = v_A lognormal disorder. The key is the cell's GLOBAL id, never its array
// index, so the draw is identical at fresh init and when a resume has to
// RE-DERIVE it because the user overrode the policy on the command line and
// the corresponding sidecar was therefore discarded. Two copies of these two
// formulas -- one in seed_positions(), one in the checkpoint loader -- would
// silently give a resumed leg a different disorder realisation from the leg it
// continues.
// ---------------------------------------------------------------------------
constexpr uint32_t kIcDomainJitter   = 1u;
constexpr uint32_t kIcDomainPolarity = 2u;
constexpr uint32_t kIcDomainVA       = 3u;

// Host-only on purpose: both are initial-condition policy, evaluated once per
// cell before any launch. Keeping them off the device sidesteps the
// std::sqrt-in-device-code portability question entirely.
// `stream` must be SimParams::polarity_stream(), not the placement seed.
inline float ic_theta(int32_t gid, unsigned long long stream) {
    const Philox4 r = philox4x32_10((uint32_t)gid, kIcDomainPolarity, 0u, 0u,
                                    (uint32_t)(stream & 0xFFFFFFFFull),
                                    (uint32_t)(stream >> 32));
    return (float)(2.0 * kPi * philox_uniform53(r.v[0], r.v[1]));
}

// Lognormal with median v_A: exp(sigma * z), z ~ N(0,1) by Box-Muller. Returns
// v_A unchanged when sigma <= 0, without consuming the draw, exactly as
// seed_positions does.
inline double ic_v_A(int32_t gid, unsigned long long seed,
                     double v_A, double sigma) {
    if (!(sigma > 0.0)) return v_A;
    const Philox4 r = philox4x32_10((uint32_t)gid, kIcDomainVA, 0u, 0u,
                                    (uint32_t)(seed & 0xFFFFFFFFull),
                                    (uint32_t)(seed >> 32));
    const double u1 = std::max(1e-300, philox_uniform53(r.v[0], r.v[1]));
    const double u2 = philox_uniform53(r.v[2], r.v[3]);
    const double z = std::sqrt(-2.0 * std::log(u1))
                   * std::cos(2.0 * kPi * u2);
    return v_A * std::exp(sigma * z);
}

// ---------------------------------------------------------------------------
// State dump format (--dump-state). Self-describing, little-endian, so the
// Python oracle at cpp/simulation/tests/python/cpu_reference.py can paint each
// cell's rect into a full periodic domain exactly as cells_from_checkpoint
// does for the production checkpoints.
// ---------------------------------------------------------------------------
constexpr uint32_t kDumpMagic   = 0x46523152u;   // "FR1R"
constexpr uint32_t kDumpVersion = 1u;

struct DumpHeader {
    uint32_t magic;
    uint32_t version;
    int32_t  num_cells;
    int32_t  domain_side;
    int32_t  tile_pitch;
    int32_t  num_classes;
    int32_t  cls_wx[8];
    int32_t  cls_wy[8];
    int32_t  cls_tx0[8];
    int32_t  cls_ty0[8];
    double   dx, dy, dt, t_now;
    double   lambda, radius, kappa, mu, xi, tau, v_A;
    double   gamma_normal, gamma_cancer;
    double   p_tumble;
    int64_t  step;
    uint32_t flags[FLAG_COUNT];
    uint32_t pad[8];
};

// Each cell then contributes one DumpCell followed by wx*wy float32 in rect
// row-major order (x fastest). Global pixel of rect (a,b) is
// ((gx0 + a) mod L, (gy0 + b) mod L).
struct DumpCell {
    int32_t global_id;
    int32_t cls;
    int32_t gx0, gy0;
    int32_t wx, wy;
    int32_t bb_lo_x, bb_hi_x, bb_lo_y, bb_hi_y;
    float   gamma, v_A, theta, vx, vy, phi_max;
    double  V, Cx, Cy, perim, Ix, Iy;
};

// ---------------------------------------------------------------------------
// Kernels (definitions in kernels.cu).
// ---------------------------------------------------------------------------
// 768 threads, 1 CTA/SM. The launch bounds are load-bearing: they set the
// 85-register budget the peak live set (~45) was sized against.
__global__ __launch_bounds__(kBlockThreads, 1)
void k_step(PF_GRID_CONSTANT const StepArgs A);

// ---------------------------------------------------------------------------
// SPLIT path (--split). Same StepArgs, same physics, two kernels per step.
//
// k_step_rhs  : P0, P1, P1b, P2 + the shifted store of phi^{n+1} STRAIGHT TO
//               GLOBAL. Shared memory holds phi_s and the scalar/reduction
//               region only -- S is read pointwise from global. Persistent
//               grid + work cursor, exactly like k_step; also carries phase C
//               (clear-ahead) and the step-counter/cursor bookkeeping.
// k_step_post : P3, P3b. One CTA per cell, grid = N. Re-reads phi^{n+1} from
//               global, scatters into S_next, accumulates V/Cx/Cy/perim, the
//               integer bbox and phi_max, and writes those CellState fields.
//
// The launch-bounds min-blocks arguments are load-bearing: they are what force
// ptxas to a register budget that permits the target CTAs/SM (see params.cuh).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kSplitBlockThreads, kSplitRhsCtasPerSm)
void k_step_rhs(PF_GRID_CONSTANT const StepArgs A);

__global__ __launch_bounds__(kSplitBlockThreads, kSplitPostCtasPerSm)
void k_step_post(PF_GRID_CONSTANT const StepArgs A);

__global__ void k_init_tiles(float* phi_a, float* phi_b, CellState* cell,
                             const uint8_t* cls, int N, int L,
                             const float* seed_cx, const float* seed_cy,
                             float radius_eff, float kappa_iface);

__global__ __launch_bounds__(kBlockThreads, 1)
void k_init_moments(CellState* cell, const uint8_t* cls,
                    const float* phi, int N);

__global__ void k_scatter_all(const float* phi, const CellState* cell,
                              const uint8_t* cls, uint32_t* S,
                              int N, int L, int P, uint32_t* flags);

__global__ void k_zero_u32(uint32_t* p, size_t n);

// Debug cadence: recompute V from scratch and measure max|phi| outside the
// window (invariant I1). Host compares; no float atomics, so the result is
// reduction-order independent.
__global__ void k_verify_cells(const float* phi, const CellState* cell,
                               const uint8_t* cls, int N,
                               double* out_V, float* out_outside_max);
__global__ void k_verify_S(const uint32_t* S, size_t n, uint32_t* out_max);

__global__ void k_pack_traj(const CellState* cell, const uint8_t* cls,
                            TrajPackedCell* out, int N, int L);

// Single-CTA bitonic sort of (morton(COM) << 32 | index). M must be a power of
// two >= N and M*8 bytes of dynamic shared memory must be opted in.
__global__ void k_morton_sort(const CellState* cell, uint32_t* perm,
                              int N, int M, int L);

// ---------------------------------------------------------------------------
// Host-side launchers.
// ---------------------------------------------------------------------------
void configure_k_step_smem();                // cudaFuncSetAttribute opt-in
void configure_split_smem();                 // opt-in for k_step_rhs
void configure_morton_smem(int smem_bytes);
int  k_step_grid(int device);                 // = numSMs

void launch_step(const StepArgs& A, int grid, cudaStream_t stream,
                 const void* l2_base, size_t l2_bytes, float l2_hit_ratio);

// Two launches on `stream`: k_step_rhs on `grid` persistent CTAs, then
// k_step_post on A.N CTAs. Stream-ordered, so the kernel boundary is the only
// synchronisation the split needs.
void launch_step_split(const StepArgs& A, int grid, cudaStream_t stream,
                       const void* l2_base, size_t l2_bytes, float l2_hit_ratio);

// ---------------------------------------------------------------------------
// Runtime occupancy report. `regs` and `local_bytes` come from
// cudaFuncGetAttributes; `ctas_per_sm` is what the driver's occupancy
// calculator says is actually achievable with the given block size and dynamic
// shared-memory request (so it accounts for registers AND shared memory AND the
// MaxDynamicSharedMemorySize opt-in -- call the configure_* helpers first).
// ---------------------------------------------------------------------------
struct KernelStats {
    int    regs         = 0;
    size_t local_bytes  = 0;      // > 0 means ptxas spilled
    int    static_smem  = 0;
    int    dynamic_smem = 0;
    int    ctas_per_sm  = 0;      // measured, from the occupancy calculator
    int    reg_limited_ctas = 0;  // 65536 / (regs * threads), the reg ceiling
    int    warps_per_sm  = 0;
    double occupancy     = 0.0;   // resident threads / max resident threads
};

bool query_kernel_stats(const void* fn, int block_threads, int dynamic_smem,
                        int device, KernelStats* out);

}  // namespace pf
