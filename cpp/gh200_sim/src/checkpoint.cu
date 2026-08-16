// ===========================================================================
// FUSE-1R checkpoint I/O — v8 format, byte-compatible with cpp/simulation.
//
// Record definitions come from cpp/common/checkpoint_format.h, which the old
// tree includes too. Nothing in this file duplicates a layout constant.
//
// ---- WHY THIS FILE EMITS ITS NATIVE tile_t AND NOT 320 --------------------
// The v8 format stores the tile edge in the file (int32 tile_t, right after
// the SimParams blob). Every consumer that matters reads it from there rather
// than assuming a build-time constant, verified by reading each parser:
//
//   rust/cpu_ref/src/checkpoint.rs                 :170  reads tile_t, tt = tile_t^2
//   rust/cell_analyze/.../checkpoint.rs            :300  reads tile_t, bbox
//                                                        spans ox..ox+tile_t
//   rust/cell_analyze/.../merge_checkpoint.rs      :95   reads tile_t,
//                                                        record_size = 32 + tile_t^2*4
//   cpp/simulation/src/sim.cu (decode_cell_records) :990 reads T_in and
//                                                        re-tiles when it
//                                                        differs from TILE_T
//
// Both native values (compact 256 and extended 288) are therefore read from the
// file rather than inferred. Writing costs one contiguous D2H and zero
// repacking, and it is lossless because invariant I1 guarantees the tile is
// exactly 0.0f outside the active window.
//
// The old simulator, built with TILE_T=320, takes its existing re-tile path on
// load. Its offset is computed from the file's tile_t, not hard-coded to 256.
//
// ---- RNGS ------------------------------------------------------------------
// This engine has NO RNG state: tumbles come from counter-based Philox keyed
// on (seed, global_id, step), all three of which are restored. There is
// therefore nothing to put in an RNGS sidecar and this writer emits none.
// The old reader treats sidecars as optional and simply re-seeds its curand
// states, which is correct — its stream was never continuous with ours.
// On READ an RNGS block is recognised and skipped (see read_sidecars).
// ===========================================================================

#include "../include/checkpoint.cuh"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace pf {
namespace {

// ---------------------------------------------------------------------------
// stdio helpers. Every read and write is checked; a short count is an error,
// never a silently truncated file.
// ---------------------------------------------------------------------------
bool read_exact(std::FILE* f, void* dst, size_t bytes, const char* what) {
    if (bytes == 0) return true;
    if (std::fread(dst, 1, bytes, f) != bytes) {
        std::fprintf(stderr, "[ckpt] short read of %s (%zu B)\n", what, bytes);
        return false;
    }
    return true;
}

int wrapi_h(int v, int L) {
    v %= L;
    return v < 0 ? v + L : v;
}

double wrapd_h(double v, double L) {
    double m = std::fmod(v, L);
    if (m < 0.0) m += L;
    return m;
}

// ---------------------------------------------------------------------------
// SimParams <-> the v8 on-disk blob.
//
// The two structs describe different solvers, so the mapping is partial in
// BOTH directions and every gap is listed here rather than discovered later:
//
//   v8 has, we do not:  gamma (single-valued), subdomain_padding, halo,
//                       polarity_seed, abp. Written as gamma_normal / 2.0 /
//                       0 / 0 / false respectively.
//   we have, v8 does not:
//                       gamma_cancer, cancer_fraction, v_A_sigma,
//                       full_moment_every, verify_every, rho.
//
// None of the second group is needed to REPRODUCE a state, because the
// per-cell consequences of all three physics ones (gamma_cancer +
// cancer_fraction -> per-cell gamma, v_A_sigma -> per-cell v_A) are persisted
// exactly in the GAMA and VA_A sidecars. They matter only if the user asks for
// a NEW assignment on resume, which is precisely when they must pass the flag
// again — and passing it is what sets the override bit that discards the
// sidecar. rho is back-derived from the stored domain side for reporting.
// full_moment_every / verify_every are cadences, not state.
// ---------------------------------------------------------------------------
void params_from_v8(const ckpt::SimParamsV8& f, int num_cells, SimParams* p) {
    p->Nx = f.Nx;
    p->Ny = f.Ny;
    p->dx = f.dx;
    p->dy = f.dy;
    p->dt = f.dt;
    p->t_end = f.t_end;
    p->lambda = f.lambda;
    p->gamma_normal = f.gamma;
    p->kappa = f.kappa;
    p->target_radius = f.target_radius;
    p->mu = f.mu;
    p->v_A = f.v_A;
    p->xi = f.xi;
    p->tau = f.tau;
    p->num_cells = num_cells;
    p->seed = (unsigned long long)f.seed;
    // Restore the polarity stream so a production leg resumed from an
    // equilibration checkpoint keeps reorienting on the same schedule. 0 in the
    // file means "follow seed", which polarity_stream() already resolves.
    p->polarity_seed = (unsigned long long)f.polarity_seed;
    if (f.print_interval > 0) p->print_interval = f.print_interval;
    // Reporting only: the authority on the domain is Nx/Ny, which we just
    // adopted. Recomputing Nx from rho on resume would silently move every
    // stored origin, so nothing downstream is allowed to do it.
    const double area = (double)f.Nx * (double)f.Ny;
    if (area > 0.0)
        p->rho = (double)num_cells * target_area(p->target_radius) / area;
}

void params_to_v8(const SimParams& p, int trajectory_samples,
                  int save_interval, ckpt::SimParamsV8* f) {
    std::memset(f, 0, sizeof(*f));   // incl. tail_pad: bit-reproducible files
    f->Nx = p.Nx;
    f->Ny = p.Ny;
    f->dx = p.dx;
    f->dy = p.dy;
    f->dt = p.dt;
    f->t_end = p.t_end;
    f->lambda = p.lambda;
    f->gamma = p.gamma_normal;      // per-cell heterogeneity -> GAMA sidecar
    f->kappa = p.kappa;
    f->target_radius = p.target_radius;
    f->mu = p.mu;
    f->v_A = p.v_A;
    f->xi = p.xi;
    f->tau = p.tau;
    // The old tree's live default. It is that solver's adaptive-rect padding
    // and has no meaning here, but writing 0 would give a resumed old run a
    // degenerate rect. Do not "helpfully" write anything else.
    f->subdomain_padding = 2.0;
    f->halo = 0;                    // v7+ tiles carry no halo
    f->save_interval = save_interval;
    f->print_interval = p.print_interval;
    f->trajectory_samples = trajectory_samples;
    f->seed = (uint32_t)(p.seed & 0xFFFFFFFFull);
    // Persist the RESOLVED polarity stream, not the raw field. The matched-pair
    // protocol equilibrates and then resumes into production, and both branches
    // must reorient at the same times and angles across that boundary. Writing 0
    // would silently re-derive the stream from `seed` on resume and break the
    // pairing for any run whose polarity seed differs from its placement seed.
    f->polarity_seed = (uint32_t)(p.polarity_stream() & 0xFFFFFFFFull);
    // Truthful, not a coercion: this engine integrates run-and-tumble only, so
    // the state it just produced IS abp = 0. The READ side refuses a nonzero
    // abp rather than silently reinterpreting it (see checkpoint_read).
    f->abp = 0;
}

// ---------------------------------------------------------------------------
// Two bounding boxes of one foreign tile, in that tile's local coordinates,
// from a single pass:
//
//   sup[]  pixels above kSupportEps. This is the SUPPORT, and it is what
//          decides the shape class — the same threshold and the same meaning
//          the device's own bbox reduction uses.
//   nz[]   pixels that are merely nonzero. Everything between the two boxes is
//          exponential tail below 1e-5; it does not affect the class, but a
//          window that also contains it is a LOSSLESS placement, which is what
//          makes reloading our own checkpoints exact rather than merely
//          equivalent.
//
// Per-axis lo/hi pairs, so the placement arithmetic below is a loop over axes;
// only this scan is inherently 2-D, because it is what indexes a 2-D array.
// Returns false when nothing exceeds kSupportEps — a cell with no support
// cannot be centred, cannot be classified, and is not a cell.
// ---------------------------------------------------------------------------
bool tile_bboxes(const float* tile, int T, int sup_lo[kCkptDims],
                 int sup_hi[kCkptDims], int nz_lo[kCkptDims],
                 int nz_hi[kCkptDims]) {
    for (int d = 0; d < kCkptDims; ++d) {
        sup_lo[d] = nz_lo[d] = T;
        sup_hi[d] = nz_hi[d] = -1;
    }
    for (int y = 0; y < T; ++y) {
        const float* row = tile + (size_t)y * T;
        for (int x = 0; x < T; ++x) {
            const float v = row[x];
            if (v == 0.0f) continue;
            if (x < nz_lo[0]) nz_lo[0] = x;
            if (x > nz_hi[0]) nz_hi[0] = x;
            if (y < nz_lo[1]) nz_lo[1] = y;
            if (y > nz_hi[1]) nz_hi[1] = y;
            if (v <= kSupportEps) continue;
            if (x < sup_lo[0]) sup_lo[0] = x;
            if (x > sup_hi[0]) sup_hi[0] = x;
            if (y < sup_lo[1]) sup_lo[1] = y;
            if (y > sup_hi[1]) sup_hi[1] = y;
        }
    }
    return sup_hi[0] >= 0;
}

// ---------------------------------------------------------------------------
// Repack one foreign tile_t x tile_t tile into one native kTilePitch tile.
//
// Chooses the smallest shape class that CONTAINS the support with
// kPromoteSlack pixels to spare on both axes — the SAME predicate, from the
// same definition in params.cuh, that the device applies every step. Using
// kPromoteSlack rather than 0 is what makes the placement stable: the device
// promotes when extent + kPromoteSlack > W, so a cell placed under this rule
// is never promoted on its first step purely because it was just loaded. Using
// kDemoteSlack (20) instead would push an ordinary relaxed cell (extent ~127)
// out of the round class and into class 3, which is measured never to be
// selected in practice and costs the large-window path for nothing.
//
// The support is then CENTRED in the chosen window. Source pixels outside the
// foreign tile read as 0.0f, which is exact: the support is strictly inside
// that tile (we just measured it), so nothing outside it was ever nonzero.
//
// An EXACT fast path runs first whenever the tile's whole nonzero content sits
// inside some native class at that class's canonical offset. Equality of the
// file and build tile edges is unnecessary: the file records the global origin
// of pixel (0,0), and source coordinates outside a smaller tile are exactly
// zero. Adopting the canonical source offset therefore preserves every stored
// value and its global coordinate. Besides keeping save -> load -> save
// idempotent, this makes a compatible 256-pixel checkpoint load losslessly into
// the opt-in 288-pixel layout.
//
// Returns false, loudly, when no class contains the support. There is
// deliberately no clipping fallback anywhere on this path: silently truncating
// phi is the failure mode the shape-class machinery exists to make impossible,
// and a loader is not allowed to reintroduce it.
//
// `out_dropped` receives the largest |phi| that fell outside the chosen
// window. It is zero on the exact path and bounded by kSupportEps on the
// general one (by construction: the support is inside), and it is reported so
// the magnitude is a measured number rather than an argument.
// ---------------------------------------------------------------------------
bool repack_tile(const float* src, int T, int cell_id,
                 float* dst_tile, int* out_cls, int off[kCkptDims],
                 float* out_dropped, bool* out_exact, int out_ext[kCkptDims]) {
    int lo[kCkptDims], hi[kCkptDims], nz_lo[kCkptDims], nz_hi[kCkptDims];
    if (!tile_bboxes(src, T, lo, hi, nz_lo, nz_hi)) {
        std::fprintf(stderr,
            "[ckpt] cell %d has no phi > %.1e anywhere in its %dx%d tile.\n"
            "       That is a vanished cell, not a state this engine can "
            "resume from.\n", cell_id, (double)kSupportEps, T, T);
        return false;
    }

    int ext[kCkptDims];
    for (int d = 0; d < kCkptDims; ++d) {
        ext[d] = hi[d] - lo[d] + 1;
        out_ext[d] = ext[d];
    }

    int cls = class_containing(ext[0], ext[1], kPromoteSlack);
    if (cls < 0) {
        std::fprintf(stderr,
            "[ckpt] cell %d has a phi > %.1e support of %d x %d px, which fits "
            "NO shape class with the %d px promote slack.\n"
            "       Widest representable support is %d x %d px. Refusing to "
            "load: clipping phi here would cut a step discontinuity into the\n"
            "       interface and destroy the phi^2 mass the volume term is "
            "holding at A0. Add a larger shape class (see the table in\n"
            "       include/params.cuh) or resume this state with the solver "
            "that wrote it.\n"
            "       NOTE: a relaxed cell at R = 49, lambda = 7 has a phi > 1e-5\n"
            "       support of ~127 px, because the stationary interface of the\n"
            "       shared 30/lambda^2 bulk term decays as exp(-2*sqrt(7.5)/"
            "lambda * d). A support far wider than that means the field in the\n"
            "       file is NOT on that stationary profile -- an unrelaxed "
            "initial condition, or a different interface normalisation.\n",
            cell_id, (double)kSupportEps, ext[0], ext[1], kPromoteSlack,
            kClasses[kClassLarge].wx - kPromoteSlack,
            kClasses[kClassLarge].wy - kPromoteSlack);
        return false;
    }

    // --- exact path -------------------------------------------------------
    // Smallest area wins, exactly as class_containing() decides. The shared
    // helper is exercised by both CPU layout contracts and proves every
    // nonzero source pixel lies inside the rectangle copied below.
    const int exact_cls = class_preserving_nonzero(
        ext[0], ext[1], kPromoteSlack,
        nz_lo[0], nz_hi[0], nz_lo[1], nz_hi[1]);
    const bool exact = exact_cls >= 0;
    if (exact) {
        cls = exact_cls;
        const ShapeClass s = class_of(cls);
        off[0] = s.tx0;
        off[1] = s.ty0;
    }

    const ShapeClass sc = class_of(cls);
    const int w[kCkptDims]   = {sc.wx, sc.wy};
    const int tz0[kCkptDims] = {sc.tx0, sc.ty0};
    if (!exact) {
        // Centre the support in the chosen window. Source pixels outside the
        // foreign tile read as 0.0f, which is exact: the support is strictly
        // inside that tile, so nothing outside it was ever above kSupportEps.
        for (int d = 0; d < kCkptDims; ++d)
            off[d] = lo[d] - (w[d] - ext[d]) / 2;
    }

    std::memset(dst_tile, 0, (size_t)kTileArea * sizeof(float));
    for (int b = 0; b < w[1]; ++b) {
        const int sy = off[1] + b;
        if (sy < 0 || sy >= T) continue;
        const float* srow = src + (size_t)sy * T;
        float* drow = dst_tile + (size_t)(tz0[1] + b) * kTilePitch + tz0[0];
        for (int a = 0; a < w[0]; ++a) {
            const int sx = off[0] + a;
            if (sx < 0 || sx >= T) continue;
            drow[a] = srow[sx];
        }
    }

    // Measure, do not assume. The general path drops only sub-kSupportEps
    // tail, but "only" is a claim, and this is the number that substantiates
    // it for the actual file in front of us.
    float dropped = 0.0f;
    if (!exact) {
        for (int y = nz_lo[1]; y <= nz_hi[1]; ++y) {
            const bool row_in = (y >= off[1] && y < off[1] + w[1]);
            const float* srow = src + (size_t)y * T;
            for (int x = nz_lo[0]; x <= nz_hi[0]; ++x) {
                if (row_in && x >= off[0] && x < off[0] + w[0]) continue;
                const float a = std::fabs(srow[x]);
                if (a > dropped) dropped = a;
            }
        }
    }
    *out_dropped = dropped;
    *out_exact = exact;
    *out_cls = cls;
    return true;
}

// ---------------------------------------------------------------------------
// Sidecars. Stops at EOF, at an unknown magic (rewinding so nothing is lost),
// or at RNGS — see the note at the top of the file: this engine keeps no RNG
// state, and sizeof(curandState) is not knowable here, so the block cannot be
// skipped over. RNGS is the LAST block the old writer emits, so stopping there
// loses nothing; say so if it happens rather than failing silently.
// ---------------------------------------------------------------------------
bool read_sidecars(std::FILE* f, int n, CheckpointData* out) {
    std::vector<float> buf;
    for (;;) {
        const long pos = std::ftell(f);
        ckpt::SidecarBlockHeader sh{};
        const size_t got = std::fread(&sh, 1, sizeof(sh), f);
        if (got == 0) break;                       // clean EOF
        if (got != sizeof(sh)) {
            std::fprintf(stderr, "[ckpt] truncated sidecar header\n");
            return false;
        }
        if (sh.magic == ckpt::MAGIC_RNGS) {
            // SKIP BY SIZE, do not break.
            //
            // This used to `break`, which was correct only by accident: the
            // historical writer happens to emit RNGS last
            // (cpp/simulation/src/sim.cu, after per_cell_float_state()). That
            // made the reader silently dependent on the block ORDER of a writer
            // in a different tree, and it meant nothing past that point was ever
            // validated -- a truncated or corrupt tail read as a clean file.
            //
            // sizeof(curandStateXORWOW_t) is 48 and is fixed by the cuRAND ABI,
            // so the payload length is knowable without linking cuRAND. Skipping
            // it lets the loop continue to EOF and lets any later sidecar be
            // read, in any order.
            constexpr long kCurandStateBytes = 48;
            const long payload = (long)sh.count * kCurandStateBytes;
            if (std::fseek(f, payload, SEEK_CUR) != 0) {
                std::fprintf(stderr,
                    "[ckpt] RNGS sidecar claims %d entries (%ld B) but the file "
                    "ends before that. The file is truncated or its RNGS "
                    "element size is not the 48 B cuRAND XORWOW state.\n",
                    sh.count, payload);
                return false;
            }
            std::printf("[ckpt] RNGS sidecar (%d entries, %ld B) skipped: this "
                        "engine's tumble stream is counter-based Philox keyed "
                        "on (polarity_seed, global_id, step), all restored, so "
                        "there is no curand state to continue.\n",
                        sh.count, payload);
            out->had_rngs = true;
            continue;
        }
        if (sh.magic != ckpt::MAGIC_VA_A && sh.magic != ckpt::MAGIC_GAMA &&
            sh.magic != ckpt::MAGIC_RADI && sh.magic != ckpt::MAGIC_POLR) {
            std::fseek(f, pos, SEEK_SET);
            break;
        }
        if (sh.count < 0) {
            std::fprintf(stderr, "[ckpt] negative sidecar count %d\n", sh.count);
            return false;
        }
        buf.assign((size_t)sh.count, 0.0f);
        if (!read_exact(f, buf.data(), (size_t)sh.count * sizeof(float),
                        "sidecar payload"))
            return false;
        if (sh.count != n) {
            std::fprintf(stderr,
                "[ckpt] sidecar 0x%08X has %d entries but the file has %d "
                "cells; ignoring the block.\n", sh.magic, sh.count, n);
            continue;
        }
        switch (sh.magic) {
            case ckpt::MAGIC_VA_A:
                for (int i = 0; i < n; ++i) out->cells[(size_t)i].v_A = buf[(size_t)i];
                out->had_vA = true;
                break;
            case ckpt::MAGIC_GAMA:
                for (int i = 0; i < n; ++i) out->cells[(size_t)i].gamma = buf[(size_t)i];
                out->had_gamma = true;
                break;
            case ckpt::MAGIC_RADI:
                for (int i = 0; i < n; ++i) out->cells[(size_t)i].R_tgt = buf[(size_t)i];
                out->had_radius = true;
                break;
            default:   // MAGIC_POLR
                for (int i = 0; i < n; ++i) out->cells[(size_t)i].theta = buf[(size_t)i];
                out->had_polr = true;
                break;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Fan-out writer: one logical byte stream, several files.
//
// The rolling and the tagged checkpoint routinely fall due on the same step,
// and the D2H of the phi pool dominates the cost of a checkpoint. Writing both
// from one gather keeps that cost at one pass. Each file lands via <path>.tmp
// + checked fclose + rename, so a SIGKILL between the two never leaves a
// half-written checkpoint where a good one used to be.
// ---------------------------------------------------------------------------
class FanOutWriter {
public:
    bool open(const std::vector<std::string>& paths) {
        final_ = paths;
        for (const std::string& p : paths) {
            const std::string t = p + ".tmp";
            std::FILE* f = std::fopen(t.c_str(), "wb");
            if (!f) {
                std::fprintf(stderr, "[ckpt] cannot open %s: %s\n", t.c_str(),
                             std::strerror(errno));
                discard();
                return false;
            }
            tmp_.push_back(t);
            fps_.push_back(f);
        }
        return !fps_.empty();
    }

    bool write(const void* src, size_t bytes, const char* what) {
        if (bytes == 0) return true;
        for (size_t i = 0; i < fps_.size(); ++i) {
            if (std::fwrite(src, 1, bytes, fps_[i]) != bytes) {
                std::fprintf(stderr, "[ckpt] short write of %s to %s: %s\n",
                             what, tmp_[i].c_str(), std::strerror(errno));
                return false;
            }
        }
        return true;
    }

    // fclose is where a full filesystem finally reports itself, so its return
    // value is the one that decides whether the rename may happen at all.
    bool commit() {
        bool ok = true;
        for (size_t i = 0; i < fps_.size(); ++i) {
            if (std::fclose(fps_[i]) != 0) {
                std::fprintf(stderr, "[ckpt] failed to close %s: %s\n",
                             tmp_[i].c_str(), std::strerror(errno));
                ok = false;
            }
            fps_[i] = nullptr;
        }
        fps_.clear();
        if (!ok) { unlink_tmps(); return false; }
        for (size_t i = 0; i < tmp_.size(); ++i) {
            std::error_code ec;
            std::filesystem::rename(tmp_[i], final_[i], ec);
            if (ec) {
                std::fprintf(stderr, "[ckpt] rename %s -> %s failed: %s\n",
                             tmp_[i].c_str(), final_[i].c_str(),
                             ec.message().c_str());
                ok = false;
            }
        }
        return ok;
    }

    void discard() {
        for (std::FILE* f : fps_) if (f) std::fclose(f);
        fps_.clear();
        unlink_tmps();
    }

    ~FanOutWriter() { for (std::FILE* f : fps_) if (f) std::fclose(f); }

private:
    void unlink_tmps() {
        std::error_code ec;
        for (const std::string& t : tmp_) std::filesystem::remove(t, ec);
    }
    std::vector<std::FILE*>  fps_;
    std::vector<std::string> tmp_, final_;
};

// Cells per staged D2H. The pool is [N][T][T] and contiguous, so a run of
// cells is one contiguous copy. 64 * 256 KB = 16 MB of host staging, which
// bounds peak host memory independently of N while keeping each transfer far
// above the size where PCIe/C2C latency matters.
constexpr int kStageCells = 64;

}  // namespace

// ---------------------------------------------------------------------------
void SimOverrides::apply(SimParams& p, const SimParams& cli) const {
    if (t_end)           p.t_end = cli.t_end;
    if (dt)              p.dt = cli.dt;
    if (v_A)             p.v_A = cli.v_A;
    if (v_A_sigma)       p.v_A_sigma = cli.v_A_sigma;
    if (tau)             p.tau = cli.tau;
    if (gamma)           p.gamma_normal = cli.gamma_normal;
    if (gamma_cancer)    p.gamma_cancer = cli.gamma_cancer;
    if (cancer_fraction) p.cancer_fraction = cli.cancer_fraction;
    if (kappa)           p.kappa = cli.kappa;
    if (mu)              p.mu = cli.mu;
    if (xi)              p.xi = cli.xi;
    if (lambda)          p.lambda = cli.lambda;
    if (target_radius)   p.target_radius = cli.target_radius;
    if (seed)            p.seed = cli.seed;
    if (polarity_seed)   p.polarity_seed = cli.polarity_seed;
    if (print_interval)  p.print_interval = cli.print_interval;
    if (full_moment)     p.full_moment_every = cli.full_moment_every;
    if (verify_every)    p.verify_every = cli.verify_every;
}

// ---------------------------------------------------------------------------
bool checkpoint_read(const std::string& path, CheckpointData* out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[ckpt] cannot open %s: %s\n", path.c_str(),
                     std::strerror(errno));
        return false;
    }
    struct Closer {
        std::FILE* f;
        ~Closer() { if (f) std::fclose(f); }
    } closer{f};

    ckpt::FixedPrefix pre{};
    if (!read_exact(f, &pre, sizeof(pre), "fixed prefix")) return false;
    if (pre.magic != ckpt::MAGIC) {
        std::fprintf(stderr, "[ckpt] %s is not a cell checkpoint "
                     "(magic 0x%08X, expected 0x%08X)\n",
                     path.c_str(), pre.magic, ckpt::MAGIC);
        return false;
    }
    // v3..v6 predate the uniform-tile layout: their per-cell records carry a
    // bounding box and a variable-size phi block, and three mutually
    // incompatible SimParams encodings live in the wild. Re-implementing that
    // archaeology here would double the size of this file to serve files the
    // production simulator can already upconvert in one pass.
    if (pre.version < 7 || pre.version > ckpt::VERSION_CURRENT) {
        std::fprintf(stderr,
            "[ckpt] %s is v%u; this engine reads v7 and v8 only.\n"
            "       Load it with cpp/simulation's cell_sim (which reads v3+) "
            "and save once to upconvert, or use `cell_analyze`.\n",
            path.c_str(), pre.version);
        return false;
    }
    if (pre.sp_sz != sizeof(ckpt::SimParamsV8)) {
        std::fprintf(stderr,
            "[ckpt] v%u SimParams blob is %u B, expected %zu B. The file was "
            "written by a build whose SimParams layout has drifted from\n"
            "       cpp/common/checkpoint_format.h; it cannot be decoded "
            "without knowing which layout it is.\n",
            pre.version, pre.sp_sz, sizeof(ckpt::SimParamsV8));
        return false;
    }
    if (pre.num_cells_local <= 0) {
        std::fprintf(stderr, "[ckpt] implausible cell count %d\n",
                     pre.num_cells_local);
        return false;
    }

    ckpt::SimParamsV8 sp{};
    if (!read_exact(f, &sp, sizeof(sp), "SimParams blob")) return false;

    int32_t tile_t = 0;
    if (!read_exact(f, &tile_t, sizeof(tile_t), "tile_t")) return false;
    if (tile_t <= 0 || tile_t > 4096) {
        std::fprintf(stderr, "[ckpt] tile_t = %d is out of range (1..4096); "
                     "the file is corrupt or foreign\n", tile_t);
        return false;
    }

    out->version = pre.version;
    out->num_ranks = 1;
    out->rank_id = 0;
    out->n_global = pre.num_cells_local;
    if (pre.version >= 8) {
        ckpt::RankTrailer tr{};
        if (!read_exact(f, &tr, sizeof(tr), "rank trailer")) return false;
        out->num_ranks = tr.num_ranks;
        out->rank_id = tr.rank_id;
        out->n_global = tr.num_cells_global;
        if (tr.num_ranks > 1) {
            std::fprintf(stderr,
                "[ckpt] %s is rank %d of a %d-rank multi-GPU checkpoint and "
                "holds only that rank's cells. This engine is single-GPU;\n"
                "       consolidate first with `cell_analyze merge-ckpt %s`.\n",
                path.c_str(), tr.rank_id, tr.num_ranks, path.c_str());
            return false;
        }
    }

    const int n = pre.num_cells_local;
    out->n = n;
    out->step = pre.step;
    out->t = pre.cur_time;
    out->file_tile_t = tile_t;
    out->params = SimParams{};
    params_from_v8(sp, n, &out->params);

    // ABP: REFUSE, do not coerce.
    //
    // The v8 blob carries an `abp` flag selecting active-Brownian-particle
    // polarity dynamics (a rotational-diffusion update) instead of run and
    // tumble. This engine implements run and tumble ONLY. Silently loading an
    // ABP state and integrating it as RTP changes the polarity model without
    // changing a single line of the log, which is the exact class of failure
    // the rest of this engine refuses to allow (no silent clamps, no silent
    // truncation). So it is a hard error, not a warning.
    if (sp.abp != 0) {
        std::fprintf(stderr,
            "[ckpt] %s was written with abp = %u (active Brownian particle "
            "polarity). This engine implements run-and-tumble only:\n"
            "       theta is resampled uniformly with probability "
            "-expm1(-dt/tau) per step, not rotationally diffused.\n"
            "       Loading it here would silently change the polarity model. "
            "Refusing. Continue this branch with the solver that wrote it, or\n"
            "       start a new run and label it as a different polarity "
            "convention.\n",
            path.c_str(), (unsigned)sp.abp);
        return false;
    }

    if (out->params.Nx <= 0 || out->params.Nx != out->params.Ny) {
        std::fprintf(stderr, "[ckpt] checkpoint domain %d x %d is not a valid "
                     "square\n", out->params.Nx, out->params.Ny);
        return false;
    }
    const int L = out->params.Nx;

    // Defaults BEFORE the sidecars: this is the "params" tier of the
    // CLI > sidecar > params precedence. A file with no GAMA block therefore
    // yields the uniform gamma its SimParams recorded, not zero.
    out->cells.assign((size_t)n, CkptCell{});
    for (int i = 0; i < n; ++i) {
        CkptCell& c = out->cells[(size_t)i];
        c.gamma = (float)out->params.gamma_normal;
        c.v_A   = (float)out->params.v_A;
        c.R_tgt = (float)out->params.target_radius;
    }
    out->phi.assign((size_t)n * (size_t)kTileArea, 0.0f);

    std::vector<float> src((size_t)tile_t * (size_t)tile_t);
    int cls_hist[kNumClasses] = {};
    int n_exact = 0;
    int max_ext[kCkptDims] = {0, 0};
    float worst_dropped = 0.0f;
    for (int i = 0; i < n; ++i) {
        ckpt::CellRecordHeader rec{};
        if (!read_exact(f, &rec, sizeof(rec), "cell record header")) return false;
        if (!read_exact(f, src.data(), src.size() * sizeof(float), "phi tile"))
            return false;

        int cls = 0, off[kCkptDims] = {0, 0}, ext[kCkptDims] = {0, 0};
        float dropped = 0.0f;
        bool exact = false;
        if (!repack_tile(src.data(), tile_t, rec.cell_id,
                         out->phi.data() + (size_t)i * kTileArea, &cls, off,
                         &dropped, &exact, ext))
            return false;
        n_exact += exact ? 1 : 0;
        if (dropped > worst_dropped) worst_dropped = dropped;
        for (int d = 0; d < kCkptDims; ++d)
            if (ext[d] > max_ext[d]) max_ext[d] = ext[d];

        CkptCell& c = out->cells[(size_t)i];
        // v8 cell_id is the GLOBAL id; v7 stored the local index, which for a
        // single-rank file is the same thing.
        c.global_id = rec.cell_id;
        c.cls = (uint8_t)cls;
        c.vx = rec.vx;
        c.vy = rec.vy;
        // origin_x/origin_y are the GLOBAL coordinate of the foreign tile's
        // pixel (0,0) and may be negative; CellState::gx0/gy0 is the global
        // coordinate of the WINDOW's pixel (0,0), in [0, L). off[] carries the
        // re-centring the repack applied, so the conversion is one add and one
        // wrap per axis and does not go through the class's (tx0, ty0) at all.
        const int forigin[kCkptDims] = {rec.origin_x, rec.origin_y};
        for (int d = 0; d < kCkptDims; ++d)
            c.origin[d] = wrapi_h(forigin[d] + off[d], L);

        ++cls_hist[cls];
    }

    if (!read_sidecars(f, n, out)) return false;

    char hist[16 * kNumClasses];
    int hl = 0;
    for (int c = 0; c < kNumClasses; ++c)
        hl += std::snprintf(hist + hl, sizeof(hist) - (size_t)hl,
                            c ? "/%d" : "%d", cls_hist[c]);
    std::printf("[ckpt] %s: v%u  step %d  t %.4f  %d cells  L %d  "
                "tile_t %d -> %d\n"
                "       shape classes %s   sidecars:%s%s%s%s\n",
                path.c_str(), pre.version, pre.step, pre.cur_time, n, L,
                tile_t, kTilePitch, hist,
                out->had_polr ? " POLR" : "", out->had_gamma ? " GAMA" : "",
                out->had_vA ? " VA_A" : "", out->had_radius ? " RADI" : "");
    // The single number that says whether this file's fields are shaped the way
    // this engine's windows were sized for. A relaxed R=49, lambda=7 cell has a
    // phi > 1e-5 support of ~127 px; anything much wider is not on the
    // stationary profile of the 30/lambda^2 bulk term, and will burn the large
    // class (or be refused) for reasons that have nothing to do with crowding.
    std::printf("       widest phi > %.0e support %d x %d px  "
                "(round class holds %d, the largest class %d before refusal)\n",
                (double)kSupportEps, max_ext[0], max_ext[1],
                kClasses[kClassRound].wx - kPromoteSlack,
                kClasses[kClassLarge].wx - kPromoteSlack);
    std::printf("       %d of %d cells reloaded EXACTLY (whole nonzero tile "
                "inside a class window, copied pixel for pixel)\n",
                n_exact, n);
    if (n_exact < n)
        std::printf("       the other %d were re-centred; largest |phi| that "
                    "fell outside the new window was %.3e, against a support "
                    "threshold of %.1e\n",
                    n - n_exact, (double)worst_dropped, (double)kSupportEps);
    if (worst_dropped > kSupportEps) {
        // Cannot happen by construction (the support is contained), so if it
        // does, the scan and the placement disagree and the state is not the
        // one on disk. Refuse rather than run a silently truncated field.
        std::fprintf(stderr,
            "[ckpt] INTERNAL: dropped a pixel of %.3e, above the support "
            "threshold %.1e. Refusing to continue.\n",
            (double)worst_dropped, (double)kSupportEps);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
void resolve_per_cell_scalars(const SimParams& p, const SimOverrides& ov,
                              CheckpointData* d) {
    const int n = d->n;

    // ---- gamma ------------------------------------------------------------
    // Only an explicit change to the ASSIGNMENT (--gamma, --gamma-cancer or
    // --cancer-fraction) discards GAMA. This is the load-bearing case: the
    // two-phase protocol equilibrates at gamma = 1 and then resumes the SAME
    // microstate with a changed gamma, and it works precisely because passing
    // --gamma is what says "re-derive", while passing nothing says "keep the
    // per-cell values that came out of the equilibration".
    if (!d->had_gamma || ov.gamma_policy_changed()) {
        const int n_cancer = (int)std::llround(p.cancer_fraction * (double)n);
        for (int i = 0; i < n; ++i) {
            CkptCell& c = d->cells[(size_t)i];
            c.gamma = (float)((c.global_id < n_cancer) ? p.gamma_cancer
                                                       : p.gamma_normal);
        }
        std::printf("[ckpt] per-cell gamma re-derived from the CLI: "
                    "%d of %d cells at gamma_cancer = %.6g, the rest at "
                    "%.6g%s\n", n_cancer, n, p.gamma_cancer, p.gamma_normal,
                    d->had_gamma ? " (GAMA sidecar discarded)"
                                 : " (no GAMA sidecar in the file)");
    }

    // ---- v_A --------------------------------------------------------------
    bool redo_vA = (!d->had_vA || ov.v_A_policy_changed());
    // The all-zero-sidecar guard, ported from cpp/simulation/src/sim.cu:1379:
    // an equilibration run at v_A = 0 still emits a VA_A block, full of zeros.
    // Resuming that with a nonzero v_A must use the v_A now in force, not the
    // zeros. The epsilon distinguishes genuine all-zero data from float noise.
    if (!redo_vA && d->had_vA && p.v_A > 0.0) {
        constexpr double kZeroSidecarSumEpsilon = 1e-12;
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += std::fabs((double)d->cells[(size_t)i].v_A);
        if (sum <= kZeroSidecarSumEpsilon) {
            std::printf("[ckpt] VA_A sidecar is identically zero but v_A = %.6g "
                        "is in force; discarding the sidecar.\n", p.v_A);
            redo_vA = true;
        }
    }
    if (redo_vA) {
        for (int i = 0; i < n; ++i) {
            CkptCell& c = d->cells[(size_t)i];
            c.v_A = (float)ic_v_A(c.global_id, p.seed, p.v_A, p.v_A_sigma);
        }
        std::printf("[ckpt] per-cell v_A re-derived from the CLI: median %.6g, "
                    "lognormal sigma %.6g\n", p.v_A, p.v_A_sigma);
    }

    // ---- target radius ----------------------------------------------------
    if (!d->had_radius || ov.target_radius) {
        for (int i = 0; i < n; ++i)
            d->cells[(size_t)i].R_tgt = (float)p.target_radius;
    }

    // ---- polarity ---------------------------------------------------------
    // theta has no CLI override: it IS the microstate, not a policy. A missing
    // POLR block therefore means the polarity is discontinuous across the
    // resume, which is worth saying out loud -- run-and-tumble statistics
    // measured across such a join have a seam in them.
    if (!d->had_polr) {
        for (int i = 0; i < n; ++i) {
            CkptCell& c = d->cells[(size_t)i];
            c.theta = ic_theta(c.global_id, p.polarity_stream());
        }
        std::fprintf(stderr,
            "[ckpt] warning: no POLR sidecar; polarity angles were re-drawn "
            "from seed %llu. The polarity is NOT continuous across this "
            "resume.\n", (unsigned long long)p.seed);
    }
}

// ---------------------------------------------------------------------------
bool checkpoint_write(const CheckpointWriteView& v,
                      const std::vector<std::string>& paths) {
    if (paths.empty()) return true;
    if (!v.p || !v.cell || !v.cls || !v.d_phi || v.N <= 0 || v.L <= 0) {
        std::fprintf(stderr, "[ckpt] incomplete write view\n");
        return false;
    }
    // v8 stores the step as int32. At dt = 0.01 that runs out at t = 2.1e7,
    // which the FSS campaigns are within an order of magnitude of, so say so
    // instead of wrapping the field into a negative step silently.
    if (v.step < 0 || v.step > 2147483647LL) {
        std::fprintf(stderr,
            "[ckpt] step %lld does not fit the v8 int32 step field. Refusing "
            "to write a checkpoint that would resume at the wrong time.\n",
            v.step);
        return false;
    }
    if (v.p->seed > 0xFFFFFFFFull)
        std::fprintf(stderr,
            "[ckpt] warning: seed %llu is stored truncated to its low 32 bits "
            "(0x%08X) — the v8 field is uint32. A run resumed from this file "
            "without an explicit --seed gets a DIFFERENT Philox key.\n",
            (unsigned long long)v.p->seed,
            (unsigned)(v.p->seed & 0xFFFFFFFFull));

    // ---- restart-exactness, decided per FILE rather than assumed -----------
    //
    // The only piece of CellState that a reload cannot reconstruct is
    // promote_ctr, the shape-class DEMOTE DWELL counter. Everything else is
    // either stored (origin, theta, gamma, v_A, R_tgt, the field itself),
    // recomputed exactly from the stored field (V, Cx, Cy, bbox, phi_max), or a
    // pure diagnostic the RHS never reads (shift_ctr, tumble_ctr, perim,
    // Ix, Iy). The Philox counter is the step number, which is stored.
    //
    // So the correct statement is not "restarts are unproven pending a new
    // sidecar". It is: a reload is EXACT whenever no cell was mid-dwell at save
    // time, and that is a property of this file which we can simply check.
    // promote_ctr is zero except while a cell is continuously eligible to demote
    // into a strictly smaller class, so it is zero for the overwhelming majority
    // of steps. Checking costs one pass over an array already in host memory and
    // needs no format change and no per-step tracking.
    int mid_dwell = 0;
    uint32_t max_dwell = 0u;
    for (int i = 0; i < v.N; ++i) {
        const uint32_t d = v.cell[(size_t)i].promote_ctr;
        if (d != 0u) { ++mid_dwell; if (d > max_dwell) max_dwell = d; }
    }
    if (mid_dwell == 0) {
        std::printf("[ckpt] restart-exact: no cell is mid-demote-dwell, so a "
                    "resume from this file reproduces the uninterrupted "
                    "trajectory bit for bit.\n");
    } else {
        std::fprintf(stderr,
            "[ckpt] warning: %d of %d cells are mid-demote-dwell (max %u of %d "
            "checks). promote_ctr is not carried by the v8 format, so a resume\n"
            "       from this file restarts those dwell counters at zero. The "
            "physics state is unaffected; the only consequence is that a\n"
            "       demotion those cells were %u/%d of the way toward is "
            "deferred. Trajectories will differ from the uninterrupted run in\n"
            "       the last bits from the first deferred demotion onward. Save "
            "one step later for an exact-restart file.\n",
            mid_dwell, v.N, max_dwell, kDemoteDwell, max_dwell, kDemoteDwell);
    }

    FanOutWriter w;
    if (!w.open(paths)) return false;

    ckpt::FixedPrefix pre{};
    pre.magic = ckpt::MAGIC;
    pre.version = ckpt::VERSION_CURRENT;
    pre.step = (int32_t)v.step;
    pre.cur_time = v.t;
    pre.num_cells_local = v.N;
    pre.save_interval = v.save_interval;
    pre.reserved = 0;
    pre.trajectory_samples = v.trajectory_samples;
    pre.sp_sz = (uint32_t)sizeof(ckpt::SimParamsV8);   // bools[] stay zero

    ckpt::SimParamsV8 sp{};
    params_to_v8(*v.p, v.trajectory_samples, v.save_interval, &sp);

    const int32_t tile_t = kTilePitch;
    ckpt::RankTrailer tr{};
    tr.num_ranks = 1;
    tr.rank_id = 0;
    tr.num_cells_global = v.N;

    if (!w.write(&pre, sizeof(pre), "fixed prefix") ||
        !w.write(&sp, sizeof(sp), "SimParams blob") ||
        !w.write(&tile_t, sizeof(tile_t), "tile_t") ||
        !w.write(&tr, sizeof(tr), "rank trailer")) {
        w.discard();
        return false;
    }

    // ---- per-cell records, streamed ---------------------------------------
    const double dA = v.p->dA();
    std::vector<float> stage((size_t)kStageCells * (size_t)kTileArea);
    for (int base = 0; base < v.N; base += kStageCells) {
        const int cnt = (v.N - base) < kStageCells ? (v.N - base) : kStageCells;
        const size_t words = (size_t)cnt * (size_t)kTileArea;
        const cudaError_t e = cudaMemcpy(stage.data(),
                                         v.d_phi + (size_t)base * kTileArea,
                                         words * sizeof(float),
                                         cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "[ckpt] D2H of cells %d..%d failed: %s\n",
                         base, base + cnt - 1, cudaGetErrorString(e));
            w.discard();
            return false;
        }
        for (int k = 0; k < cnt; ++k) {
            const int i = base + k;
            const CellState& c = v.cell[(size_t)i];
            const int cls = (int)c.cls;
            if (cls < 0 || cls >= kNumClasses || (int)v.cls[(size_t)i] != cls) {
                std::fprintf(stderr,
                    "[ckpt] cell %d carries shape class %d (cell_cls says %d); "
                    "that is memory corruption, not a shape. Refusing to "
                    "write.\n", i, cls, (int)v.cls[(size_t)i]);
                w.discard();
                return false;
            }
            const ShapeClass sc = class_of(cls);

            ckpt::CellRecordHeader rec{};
            rec.cell_id = c.global_id;
            // The format's origin is the TILE's pixel (0,0); CellState carries
            // the WINDOW's. They differ by the class offset, which is why the
            // class must be written into the geometry rather than the file.
            // Wrapped into [0, L): negatives are legal in the format but every
            // consumer applies the coordinate modulo L anyway, and not every
            // one of them is careful about the sign of the remainder.
            rec.origin_x = wrapi_h(c.gx0 - sc.tx0, v.L);
            rec.origin_y = wrapi_h(c.gy0 - sc.ty0, v.L);
            const double invV = (c.V > 0.0) ? 1.0 / c.V : 0.0;
            rec.cx = (float)wrapd_h((double)c.gx0 + c.Cx * invV, (double)v.L);
            rec.cy = (float)wrapd_h((double)c.gy0 + c.Cy * invV, (double)v.L);
            rec.vx = c.vx;
            rec.vy = c.vy;
            // Same definition as the old writer's: V is sum(phi^2) over the
            // active region in both engines (see k_reduce_mb_fast's sV += c*c),
            // so `volume` means the same number in both files.
            rec.volume = (float)(c.V * dA);

            if (!w.write(&rec, sizeof(rec), "cell record") ||
                !w.write(stage.data() + (size_t)k * kTileArea,
                         (size_t)kTileArea * sizeof(float), "phi tile")) {
                w.discard();
                return false;
            }
        }
    }

    // ---- sidecars, in the old writer's order (per_cell_float_state) --------
    std::vector<float> col((size_t)v.N);
    auto emit = [&](uint32_t magic, float CellState::*field) -> bool {
        for (int i = 0; i < v.N; ++i) col[(size_t)i] = v.cell[(size_t)i].*field;
        ckpt::SidecarBlockHeader sh{magic, v.N};
        return w.write(&sh, sizeof(sh), "sidecar header") &&
               w.write(col.data(), col.size() * sizeof(float), "sidecar payload");
    };
    if (!emit(ckpt::MAGIC_POLR, &CellState::theta) ||
        !emit(ckpt::MAGIC_GAMA, &CellState::gamma) ||
        !emit(ckpt::MAGIC_VA_A, &CellState::v_A) ||
        !emit(ckpt::MAGIC_RADI, &CellState::R_tgt)) {
        w.discard();
        return false;
    }
    // No RNGS: see the header comment.

    if (!w.commit()) return false;

    std::printf("[ckpt] saved step %lld  t %.4f  %d cells  tile_t %d ->", v.step,
                v.t, v.N, (int)tile_t);
    for (const std::string& p : paths) std::printf(" %s", p.c_str());
    std::printf("\n");
    std::fflush(stdout);
    return true;
}

}  // namespace pf
