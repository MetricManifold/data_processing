#pragma once
// ===========================================================================
// FUSE-1R checkpoint I/O — v8 format, byte-compatible with the production
// simulator at cpp/simulation.
//
// This is the ONLY translation unit that knows about the on-disk layout; the
// record definitions themselves live one level up again, in
// cpp/common/checkpoint_format.h, which cpp/simulation includes verbatim.
// Nothing here belongs in sim.cu: checkpointing is pure I/O on a cadence and
// must stay off the step loop's critical path and out of its source file.
//
// ---- what this layer has to reconcile -------------------------------------
// The v8 format was written by a solver with ONE uniform tile edge (TILE_T,
// 320 by default) per cell. This engine uses a compile-time kTilePitch of 256
// for the compact production layout or 288 for the extended candidate, with per-cell
// SHAPE CLASSES and a window offset (tx0, ty0) inside the tile, under the
// invariant that phi is exactly 0.0f outside the window (I1).
//
//   WRITE  emits tile_t = kTilePitch (256 or 288) and the tile verbatim, zeros and
//          all. Compact and lossless — see checkpoint.cu for why 256 rather
//          than 320 is safe for every known consumer.
//   READ   accepts any tile_t, locates the phi > kSupportEps support bbox,
//          picks the smallest shape class that CONTAINS it via
//          class_containing_storage(), and re-centres it in that class's
//          window. A support wider than the fixed tile interior is refused.
//
// ---- dimensionality -------------------------------------------------------
// The v8 FORMAT is 2-D only; cpp/common/checkpoint_format.h says exactly
// where and what a 3-D successor must change. The interfaces below are kept
// dimension-agnostic wherever that is free: geometry travels as kCkptDims-
// wide arrays (`origin[]`, `extent[]`) and the placement arithmetic is a loop
// over axes, so a 3-D port touches the tile scan and the record encode/decode
// and nothing else.
// ===========================================================================

#include "kernels.cuh"
#include "params.cuh"

#include "checkpoint_format.h"

#include <string>
#include <vector>

namespace pf {

// Number of spatial dimensions this build's checkpoint layer handles. The
// per-axis helpers below are written against this rather than against a
// hard-coded 2, so the 3-D work is confined to the tile scan and the record
// codec.
constexpr int kCkptDims = 2;

// ---------------------------------------------------------------------------
// Which SimParams fields did the user explicitly set on the command line?
//
// This mask is load-bearing, not a convenience: on resume the file's SimParams
// are adopted wholesale and then ONLY the flagged fields are overwritten from
// the CLI. That is what makes the two-phase protocol work — equilibrate at
// gamma = 1, then resume the SAME microstate with a changed gamma and nothing
// else disturbed. Modelled on SimOverrides in cpp/simulation/include/sim.cuh.
// ---------------------------------------------------------------------------
struct SimOverrides {
    bool t_end = false, dt = false, v_A = false, tau = false;
    bool gamma = false, gamma_cancer = false, cancer_fraction = false;
    bool kappa = false, mu = false, xi = false, lambda = false;
    bool target_radius = false, v_A_sigma = false;
    bool seed = false, polarity_seed = false;
    bool print_interval = false, full_moment = false;
    bool verify_every = false;
    // Geometry. Accepted on a fresh run, REFUSED on resume: both change the
    // domain side, and every stored origin and phi tile is expressed in it.
    bool num_cells = false, rho = false;

    // True when the user asked for a different per-cell gamma assignment, so
    // the GAMA sidecar must be discarded and the assignment re-derived.
    bool gamma_policy_changed() const {
        return gamma || gamma_cancer || cancer_fraction;
    }
    // Likewise for v_A: either the median or the disorder width moved.
    bool v_A_policy_changed() const { return v_A || v_A_sigma; }

    // Overwrite the flagged fields of `p` (adopted from the file) with `cli`.
    void apply(SimParams& p, const SimParams& cli) const;
};

// ---------------------------------------------------------------------------
// Host-side per-cell state recovered from a checkpoint, already translated
// into this engine's geometry.
// ---------------------------------------------------------------------------
struct CkptCell {
    int32_t global_id = 0;
    // Global coordinates of WINDOW pixel 0, wrapped into [0, L). This is
    // CellState::gx0/gy0, NOT the file's tile origin — the two differ by the
    // class's (tx0, ty0) plus whatever re-centring the repack applied.
    int32_t origin[kCkptDims] = {0, 0};
    uint8_t cls   = 0;
    float   gamma = 0.0f, v_A = 0.0f, R_tgt = 0.0f, theta = 0.0f;
    float   vx = 0.0f, vy = 0.0f;
};

struct CheckpointData {
    SimParams params{};          // adopted from the file, before CLI overrides
    uint32_t  version = 0;
    long long step    = 0;
    double    t       = 0.0;
    int       n       = 0;
    int       file_tile_t = 0;
    int32_t   num_ranks = 1, rank_id = 0, n_global = 0;
    // Which per-cell sidecars were actually present. Absent ones leave the
    // corresponding CkptCell field at the params-derived default, which is
    // the "sidecar > params" half of the CLI > sidecar > params precedence.
    bool had_gamma = false, had_vA = false, had_radius = false, had_polr = false;
    // An RNGS block was present and skipped. Recorded rather than ignored: it
    // means the file was written by a solver with a mutable cuRAND stream, so
    // the tumble SEQUENCE across the join is this engine's Philox sequence, not
    // the writer's. Reported at load so it is never inferred later.
    bool had_rngs = false;

    std::vector<CkptCell> cells;      // n entries
    std::vector<float>    phi;        // n * kTileArea, native build tiles
};

// Read `path` and repack it into this engine's geometry. Returns false and
// prints an actionable message on any failure — including a support that fits
// no shape class, which is fatal by design rather than clipped.
bool checkpoint_read(const std::string& path, CheckpointData* out);

// Resolve the per-cell scalars in `d` under the CLI > sidecar > params
// precedence, using `p` = the file's SimParams with `ov` already applied.
//
// The rule per field: an explicit CLI flag that changes the ASSIGNMENT POLICY
// discards the corresponding sidecar and re-derives every cell's value; a
// missing sidecar likewise falls back to the params; otherwise the sidecar
// wins, because it is the microstate. Called by the driver between
// checkpoint_read() and Sim::init_from_checkpoint().
void resolve_per_cell_scalars(const SimParams& p, const SimOverrides& ov,
                              CheckpointData* d);

// ---------------------------------------------------------------------------
// Everything the writer needs, gathered by the caller. Deliberately a view:
// the checkpoint layer owns no simulation state and never reaches into Sim.
//
// `d_phi` is a DEVICE pointer to the pool half holding phi^step; the writer
// streams it to the file through a bounded host staging buffer, so peak host
// memory is independent of N. The caller must have synchronised the stream
// before calling — this function does no synchronisation of its own and does
// not know which stream produced the data.
// ---------------------------------------------------------------------------
struct CheckpointWriteView {
    const SimParams* p    = nullptr;
    long long        step = 0;
    double           t    = 0.0;
    int              N    = 0;
    int              L    = 0;
    const CellState* cell = nullptr;   // host copy, N entries
    const uint8_t*   cls  = nullptr;   // host copy, N entries
    const float*     d_phi = nullptr;  // device, N * kTileArea floats
    int              trajectory_samples = 0;
    int              save_interval      = 0;
};

// Write the same bytes to every path in `paths`, each via <path>.tmp +
// fclose-check + rename. Several paths in one call because the rolling and
// the tagged checkpoint frequently fall due on the same step and the D2H
// traffic is the expensive part: gathering it twice would double the cost of
// the checkpoint for nothing.
bool checkpoint_write(const CheckpointWriteView& v,
                      const std::vector<std::string>& paths);

}  // namespace pf
