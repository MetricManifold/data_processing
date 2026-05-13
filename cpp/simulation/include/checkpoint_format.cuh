// ---------------------------------------------------------------------------
// checkpoint_format.cuh — single source of truth for the binary checkpoint
// header. C++ side mirrors rust/cell_data/src/checkpoint_format.rs (when it
// exists). Bump CKPT_VERSION_CURRENT here and only here when changing format.
// ---------------------------------------------------------------------------
#pragma once
#include <cstdint>

namespace ckpt {

constexpr uint32_t MAGIC           = 0x43454C4C;  // 'CELL'
constexpr uint32_t VERSION_CURRENT = 8;
constexpr uint32_t VERSION_MIN     = 3;

// Sidecar block magics (appear after per-cell records, in any order).
constexpr uint32_t MAGIC_VA_A = 0x56415F41;  // 'VA_A'
constexpr uint32_t MAGIC_GAMA = 0x47414D41;  // 'GAMA'
constexpr uint32_t MAGIC_RADI = 0x52414449;  // 'RADI'
constexpr uint32_t MAGIC_POLR = 0x504F4C52;  // 'POLR'
constexpr uint32_t MAGIC_RNGS = 0x53474E52;  // 'RNGS'

// v8 layout (all little-endian, packed):
//   magic(u32)=MAGIC, version(u32)=8,
//   step(i32), cur_time(f64),
//   num_cells_local(i32), save_interval(i32), reserved(i32),
//   trajectory_samples(i32),
//   bools[4](u8),
//   sp_sz(u32), SimParams(sp_sz bytes),
//   tile_t(i32),
//   num_ranks(i32), rank_id(i32), num_cells_global(i32),     <-- NEW in v8
//   per local cell:
//     cell_id(i32) = GLOBAL id,
//     origin_x(i32), origin_y(i32),
//     cx(f32), cy(f32), vx(f32), vy(f32), volume(f32),
//     phi[tile_t*tile_t](f32)
//   sidecar arrays: VA_A, GAMA, RADI, POLR, RNGS (each: magic, count, data)
//
// v7 is identical except the three NEW i32s are absent and `cell_id` is the
// local array index (not stable across rank counts).
//
// For G=1 runs, v8 still writes num_ranks=1, rank_id=0, num_cells_global=N.
// This makes v8 self-describing for any rank count and lets a multi-rank
// run be reassembled from the per-rank files alone (each cell carries its
// global id, so cross-rank join is unambiguous).

}  // namespace ckpt
