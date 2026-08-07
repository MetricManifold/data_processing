// ---------------------------------------------------------------------------
// checkpoint_format.h — THE single source of truth for the binary checkpoint
// layout. Plain C++17, no CUDA: it is included from both simulator trees.
//
//   consumers (C++):
//     cpp/simulation/src/sim.cu      writer (save_checkpoint), reader
//                                    (init_from_checkpoint), peek
//     cpp/gh200_sim/src/checkpoint.cu  writer + reader for the GH200 engine
//
//   MIRRORS THAT MUST BE KEPT IN SYNC BY HAND (Rust, independent parsers):
//     rust/cpu_ref/src/checkpoint.rs
//     rust/cell_analyze/src/analysis/checkpoint.rs
//     rust/cell_analyze/src/analysis/merge_checkpoint.rs
//   All three read tile_t from the file rather than assuming a build-time
//   constant, so a writer may emit any tile edge it likes. If you change
//   anything below, change those three too and bump VERSION_CURRENT here.
//
// Bump VERSION_CURRENT here and only here when changing format.
//
// ---- DIMENSIONALITY -------------------------------------------------------
// The v8 format is 2-D ONLY, in three places, all of them structural:
//   1. CellRecordHeader carries exactly (origin_x, origin_y), (cx, cy),
//      (vx, vy) — no z.
//   2. the per-cell payload is exactly tile_t*tile_t floats, and its ordering
//      (y outer, x fastest) is implicit rather than described by the file.
//   3. `tile_t` is a single scalar, so a tile cannot be anisotropic either.
// A 3-D successor must therefore change the CELL RECORD and the payload
// shape, i.e. it is a new version, not an extension. The FixedPrefix,
// RankTrailer and SidecarBlockHeader records are all dimension-agnostic and
// can be carried over unchanged; so can every sidecar magic, since sidecars
// are per-CELL scalars, not per-pixel data. The cleanest v9 would replace the
// bare `int32 tile_t` with `int32 ndim` followed by ndim tile extents, and
// widen CellRecordHeader's three vectors to ndim components.
// ---------------------------------------------------------------------------
#pragma once
#include <cstdint>
#include <cstddef>

namespace ckpt {

constexpr uint32_t MAGIC           = 0x43454C4C;  // 'CELL'
constexpr uint32_t VERSION_CURRENT = 8;
constexpr uint32_t VERSION_MIN     = 3;

// Sidecar block magics (appear after per-cell records, in any order).
constexpr uint32_t MAGIC_VA_A = 0x56415F41;  // 'VA_A' — per-cell v_A
constexpr uint32_t MAGIC_GAMA = 0x47414D41;  // 'GAMA' — per-cell gamma
constexpr uint32_t MAGIC_RADI = 0x52414449;  // 'RADI' — per-cell target radius
constexpr uint32_t MAGIC_POLR = 0x504F4C52;  // 'POLR' — per-cell polarity theta
constexpr uint32_t MAGIC_RNGS = 0x53474E52;  // 'RNGS' — per-cell curandState blob

// ---------------------------------------------------------------------------
// On-disk records. Packed and little-endian (every supported host platform
// writes LE natively). The structs ARE the schema: writer/reader/peek all
// fread/fwrite these as units, so layout drift is a compile error rather
// than a silent miscount.
//
// Whole-file order for v8:
//   FixedPrefix (44 B)
//   SimParams blob (prefix.sp_sz bytes; see SimParamsV8 below)
//   int32 tile_t
//   RankTrailer (12 B)                       [v8+ only]
//   num_cells_local x { CellRecordHeader (32 B), tile_t*tile_t float32 }
//   0..N x { SidecarBlockHeader (8 B), payload }   until EOF/unknown magic
// ---------------------------------------------------------------------------
#pragma pack(push, 1)

// File bytes 0..43. SimParams blob (sp_sz bytes) follows, then tile_t (i32),
// then RankTrailer (v8+ only), then per-cell records, then sidecar blocks.
struct FixedPrefix {
    uint32_t magic;               // = MAGIC
    uint32_t version;             // >= VERSION_MIN, <= VERSION_CURRENT
    int32_t  step;
    double   cur_time;            // v5+; v3-v4 used f32 — legacy reader handles
    int32_t  num_cells_local;
    int32_t  save_interval;
    int32_t  reserved;
    int32_t  trajectory_samples;
    uint8_t  bools[4];
    uint32_t sp_sz;
};

// v8+ only.  Lets per-rank files of a multi-GPU run be merged unambiguously.
struct RankTrailer {
    int32_t num_ranks;
    int32_t rank_id;
    int32_t num_cells_global;
};

// One per cell.  Followed by tile_t*tile_t f32s of phi data, row-major with
// y outer and x fastest. (origin_x, origin_y) are the GLOBAL coordinates of
// tile pixel (0, 0); they may be negative and are applied modulo the domain
// side. cx/cy are the global centre of mass, `volume` is sum(phi^2)*dA.
struct CellRecordHeader {
    int32_t cell_id;              // global id in v8+, local index pre-v8
    int32_t origin_x;
    int32_t origin_y;
    float   cx;
    float   cy;
    float   vx;
    float   vy;
    float   volume;
};

// Each sidecar block starts with this, followed by count*dtype-sized values.
// dtype is implied by magic (float for VA_A/GAMA/RADI/POLR, curandState bytes
// for RNGS).
struct SidecarBlockHeader {
    uint32_t magic;
    int32_t  count;
};

// ---------------------------------------------------------------------------
// The SimParams blob, as v6/v7/v8 write it: cpp/simulation's live `SimParams`
// (include/types.cuh) fwritten verbatim, 144 bytes.
//
// This mirror exists so that a reader/writer which does NOT share that live
// struct — the GH200 engine, whose SimParams is a different type entirely —
// still has ONE authoritative statement of the on-disk field order. The old
// tree static_asserts its live struct against this mirror (see the top of
// cpp/simulation/src/sim.cu), so the two cannot drift silently: any edit to
// cpp/simulation/include/types.cuh that moves a field is a compile error
// until this mirror and VERSION_CURRENT are updated.
//
// The trailing 7 bytes are the tail padding of the natural layout. The writer
// MUST zero them (memset before field-by-field copy) or checkpoint files stop
// being bit-reproducible between runs.
// ---------------------------------------------------------------------------
struct SimParamsV8 {
    int32_t  Nx, Ny;                 //   0,   4
    double   dx, dy;                 //   8,  16
    double   dt;                     //  24
    double   t_end;                  //  32
    double   lambda;                 //  40
    double   gamma;                  //  48   (ONE gamma; per-cell in GAMA)
    double   kappa;                  //  56
    double   target_radius;          //  64
    double   mu;                     //  72
    double   v_A;                    //  80
    double   xi;                     //  88
    double   tau;                    //  96
    double   subdomain_padding;      // 104
    int32_t  halo;                   // 112   (legacy; 0 for v7+)
    int32_t  save_interval;          // 116
    int32_t  print_interval;         // 120
    int32_t  trajectory_samples;     // 124
    uint32_t seed;                   // 128
    uint32_t polarity_seed;          // 132
    uint8_t  abp;                    // 136
    uint8_t  tail_pad[7];            // 137..143, MUST be written as zero
};

#pragma pack(pop)

static_assert(sizeof(FixedPrefix)       == 44, "FixedPrefix layout drift");
static_assert(sizeof(RankTrailer)       == 12, "RankTrailer layout drift");
static_assert(sizeof(CellRecordHeader)  == 32, "CellRecordHeader layout drift");
static_assert(sizeof(SidecarBlockHeader) == 8, "SidecarBlockHeader layout drift");
static_assert(sizeof(SimParamsV8)      == 144, "SimParamsV8 layout drift");

// Spot-checked against the three independent Rust parsers, which hard-code
// these byte offsets (see rust/cell_analyze/src/analysis/checkpoint.rs, the
// `sim_params_size == 144` branch).
static_assert(offsetof(SimParamsV8, lambda)        ==  40, "lambda offset");
static_assert(offsetof(SimParamsV8, gamma)         ==  48, "gamma offset");
static_assert(offsetof(SimParamsV8, target_radius) ==  64, "radius offset");
static_assert(offsetof(SimParamsV8, v_A)           ==  80, "v_A offset");
static_assert(offsetof(SimParamsV8, tau)           ==  96, "tau offset");
static_assert(offsetof(SimParamsV8, halo)          == 112, "halo offset");
static_assert(offsetof(SimParamsV8, seed)          == 128, "seed offset");
static_assert(offsetof(SimParamsV8, polarity_seed) == 132, "polarity offset");

}  // namespace ckpt
