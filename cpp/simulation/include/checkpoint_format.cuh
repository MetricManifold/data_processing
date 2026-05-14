// ---------------------------------------------------------------------------
// checkpoint_format.cuh — single source of truth for the binary checkpoint
// layout.  Writer (save_checkpoint), reader (init_from_checkpoint), and
// peek (peek_v8_rank_header) all read/write through the POD records below.
// Bump VERSION_CURRENT here and only here when changing format.
// ---------------------------------------------------------------------------
#pragma once
#include <cstdint>

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
// ---------------------------------------------------------------------------
#pragma pack(push, 1)

// File bytes 0..39. SimParams blob (sp_sz bytes) follows, then tile_t (i32),
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

// One per cell.  Followed by tile_t*tile_t f32s of phi data.
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

#pragma pack(pop)

static_assert(sizeof(FixedPrefix)       == 44, "FixedPrefix layout drift");
static_assert(sizeof(RankTrailer)       == 12, "RankTrailer layout drift");
static_assert(sizeof(CellRecordHeader)  == 32, "CellRecordHeader layout drift");
static_assert(sizeof(SidecarBlockHeader) == 8, "SidecarBlockHeader layout drift");

}  // namespace ckpt
