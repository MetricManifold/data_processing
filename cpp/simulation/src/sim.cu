// Host-side glue — allocation, init, step loop, checkpoint I/O.
//
// Architecture: fixed-tile unified-pool. Each cell owns a TILE_T x TILE_T
// phi buffer in a contiguous device array; an active rect inside the tile
// tracks the cell's current extent.
//
// Checkpoint compatibility:
//   * Reads:  v3, v4, v5, v6 (legacy variable-W/H tiles) AND v7 (TILE_T
//             uniform tiles). On legacy load we re-tile each cell into a
//             centred TILE_T x TILE_T buffer.
//   * Writes: v7 only.

#include "sim.cuh"
#include "checkpoint_format.cuh"
#include "multi_gpu.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

static bool sim_env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    return std::strcmp(value, "0") != 0
        && std::strcmp(value, "false") != 0
        && std::strcmp(value, "FALSE") != 0
        && std::strcmp(value, "off") != 0
        && std::strcmp(value, "OFF") != 0;
}

static int sim_env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < 0 || parsed > 1000000000L) return fallback;
    return (int)parsed;
}

#define CK(call) do {                                                      \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,            \
                cudaGetErrorString(e)); exit(1);                           \
    }                                                                      \
} while(0)

// ---------------------------------------------------------------------------
// place_cells — rejection sampling with periodic distance.
//
// Multi-GPU semantics: `n` is always the GLOBAL cell count. Every rank
// runs the identical RNG-driven placement (same params.seed) so the
// generated cells are bit-identical across ranks. h_cells is left
// containing the full GLOBAL vector after this call; slice_cells_to_local
// (called after apply_gamma_spec / apply_v_A_disorder) trims it to this
// rank's slice. apply_gamma_spec / apply_v_A_disorder need the global
// vector because spec selectors like `cluster(p%, x, y)` and `nearest(x, y)`
// depend on the global population layout.
// ---------------------------------------------------------------------------
void Simulation::place_cells(int n, double R) {
    unsigned s = params.seed ? params.seed : 42;
    srand(s);
    cells_global = n;
    if (gpus <= 1) cell_offset = 0;
    h_cells.resize(n);
    if (n == 1) {
        h_cells[0] = {(double)(params.Nx / 2), (double)(params.Ny / 2),
                       R, params.gamma, params.v_A, 0, 0};
        return;
    }
    double area = (double)params.Nx * params.Ny;
    double spacing = std::fmax(2.0 * R, std::sqrt(area / n) * 0.8);
    int placed = 0;
    while (placed < n) {
        bool ok = false;
        for (int att = 0; att < 10000 && !ok; att++) {
            double cx = (double)rand() / RAND_MAX * params.Nx;
            double cy = (double)rand() / RAND_MAX * params.Ny;
            bool good = true;
            for (int j = 0; j < placed && good; j++) {
                double dx = std::fabs(cx - h_cells[j].cx);
                double dy = std::fabs(cy - h_cells[j].cy);
                if (dx > params.Nx * 0.5) dx = params.Nx - dx;
                if (dy > params.Ny * 0.5) dy = params.Ny - dy;
                if (std::sqrt(dx * dx + dy * dy) < spacing) good = false;
            }
            if (good) {
                h_cells[placed] = {cx, cy, R, params.gamma, params.v_A, 0, 0};
                rand(); // consume one to match historical Cell ctor RNG draw
                ok = true;
                placed++;
            }
        }
        if (!ok) {
            spacing *= 0.95;
            if (spacing < R) {
                fprintf(stderr, "Warning: placed %d/%d\n", placed, n);
                h_cells.resize(placed);
                cells_global = placed;
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// compute_origins — set tile origin so cell COM lands at (T/2, T/2).
// ---------------------------------------------------------------------------
void Simulation::compute_origins() {
    for (auto& c : h_cells) {
        c.ox = (int)std::floor(c.cx) - TILE_T / 2;
        c.oy = (int)std::floor(c.cy) - TILE_T / 2;
    }
}

// ---------------------------------------------------------------------------
// PartitionLayout::for_rank — single formula for slab geometry. Used by
// fresh-init (slice_cells_to_local) and resume (init_from_checkpoint) to
// keep their slab math in sync.
// ---------------------------------------------------------------------------
PartitionLayout PartitionLayout::for_rank(int gpus, int rank, int Ny, int halo_h) {
    PartitionLayout L{};
    L.Ny = Ny;
    if (gpus <= 1) {
        L.slab_y_lo    = 0;
        L.slab_y_hi    = Ny;
        L.S_halo_h     = 0;
        L.S_ext_height = Ny;
    } else {
        L.slab_y_lo    = (int)((long long)rank       * Ny / gpus);
        L.slab_y_hi    = (int)((long long)(rank + 1) * Ny / gpus);
        L.S_halo_h     = halo_h;
        L.S_ext_height = (L.slab_y_hi - L.slab_y_lo) + 2 * halo_h;
    }
    return L;
}

void Simulation::build_layout() {
    layout = PartitionLayout::for_rank(gpus, rank, params.Ny, HALO_H);
}

// ---------------------------------------------------------------------------
// Per-cell float-state registry — see PerCellField doc in sim.cuh.
// Single source of truth for every per-cell float scalar's lifecycle.
// ---------------------------------------------------------------------------
std::vector<Simulation::PerCellField> Simulation::per_cell_float_state() {
    return {
        {"polar_theta", ckpt::MAGIC_POLR, &cells.polar_theta, &d_polar_theta_scratch},
        {"polar_x",     0,                &cells.polar_x,     &d_polar_x_scratch},
        {"polar_y",     0,                &cells.polar_y,     &d_polar_y_scratch},
        {"gamma",       ckpt::MAGIC_GAMA, &cells.gamma_cell,  &d_gamma_scratch},
        {"v_A",         ckpt::MAGIC_VA_A, &cells.v_A_cell,    &d_v_A_scratch},
        {"tgt_radius",  ckpt::MAGIC_RADI, &cells.tgt_radius,  &d_tgt_R_scratch},
    };
}

// ---------------------------------------------------------------------------
// GPU allocation. Per-cell arrays are allocated to capacity, not to the
// initial num_cells, so that cell migration between ranks (at rebind
// boundaries) can grow num_cells without realloc. For G=1, capacity ==
// num_cells (no migration possible). For G>1, capacity = max(initial_local,
// 2 * (N_global / G)) — generous enough that even pathological imbalance
// (one rank ending up with all the cells of one neighbour pushed in)
// doesn't realloc. Phi pool is the only big buffer that actually scales
// with capacity (TILE_AREA*4B per cell per pool half).
// ---------------------------------------------------------------------------
void Simulation::alloc_gpu() {
    const int n = (int)h_cells.size();
    cells.num_cells = n;
    if (gpus <= 1) {
        cells.capacity = n;
    } else {
        // 2x over-allocation. At G=4 N_global=12800: each rank starts with
        // ~3200 cells, capacity = 6400 — never reached in practice but
        // guarantees migration cannot OOM the per-cell arrays.
        int per_rank_floor = (cells_global > 0)
            ? (2 * cells_global / gpus + 16)
            : (2 * n + 16);
        cells.capacity = std::max(n, per_rank_floor);
    }
    const int cap = cells.capacity;

    const size_t pool_bytes = 2ULL * cap * TILE_AREA * sizeof(float);
    CK(cudaMalloc(&cells.phi_pool, pool_bytes));
    CK(cudaMemset(cells.phi_pool, 0, pool_bytes));
    cells.phi_in  = cells.phi_pool;
    cells.phi_out = cells.phi_pool + (size_t)cap * TILE_AREA;
    // Persistent half-pointers used by graph capture (parity-aware).
    phi_A = cells.phi_pool;
    phi_B = cells.phi_pool + (size_t)cap * TILE_AREA;

    // ----- Slab partition for S -----
    // For G == 1: slab covers the whole grid (y_lo=0, halo=0, ext_height=Ny).
    // For G  > 1: slab geometry was set by build_layout (called from
    //             slice_cells_to_local for fresh init, or
    //             init_from_checkpoint for resume).
    //             Each rank's S buffer is (slab_height + 2*HALO_H) x Nx
    //             floats. Pixels outside [y_lo - HALO_H, y_hi + HALO_H)
    //             (with periodic wrap) are NOT addressable on this rank;
    //             cells whose tiles extend beyond that window violate the
    //             slab contract and require migration to a neighbour rank.
    cells.S_y_lo       = layout.slab_y_lo;
    cells.S_halo_h     = layout.S_halo_h;
    cells.S_ext_height = layout.S_ext_height;
    // Halos must not overlap themselves around the periodic wrap. This
    // requires slab_height + 2*HALO_H <= Ny, i.e. ext_height <= Ny.
    // Hits when (Ny / G) is too small relative to HALO_H — for our target
    // (Ny=10412, G=4, HALO_H=159) ext_height = 2603+318 = 2921 << 10412.
    if (cells.S_ext_height > params.Ny) {
        fprintf(stderr,
            "[FATAL] alloc_gpu: slab ext_height (%d) > Ny (%d). "
            "G=%d may be too high for Ny=%d (slab_h=%d, HALO_H=%d).\n",
            cells.S_ext_height, params.Ny, gpus, params.Ny,
            layout.slab_height(), HALO_H);
        std::exit(1);
    }
    const size_t S_bytes = (size_t)cells.S_ext_height * params.Nx * sizeof(float);
    CK(cudaMalloc(&cells.S, S_bytes));
    CK(cudaMemset(cells.S, 0, S_bytes));

    auto ai = [&](int*&   p, size_t k) { CK(cudaMalloc(&p, k * sizeof(int))); };
    auto af = [&](float*& p, size_t k) { CK(cudaMalloc(&p, k * sizeof(float))); };

    // All per-cell arrays sized by capacity. Slots [num_cells, capacity)
    // are uninitialised junk; kernels never read them because they iterate
    // n < num_cells. Migration (when added) writes into these slots and
    // bumps num_cells.
    ai(cells.origin, 2 * cap);
    ai(cells.rect,   4 * cap);

    af(cells.volumes,      cap);
    af(cells.Ix,           cap);
    af(cells.Iy,           cap);
    af(cells.Cx,           cap);
    af(cells.Cy,           cap);
    af(cells.Cxx,          cap);
    af(cells.Cyy,          cap);
    af(cells.perimeters,   cap);
    af(cells.velocities_x, cap);
    af(cells.velocities_y, cap);

    for (auto& f : per_cell_float_state()) {
        af(*f.dev_ptr, cap);
    }

    CK(cudaMalloc(&cells.rng_states, cap * sizeof(curandState)));

    CK(cudaMemset(cells.velocities_x, 0, cap * sizeof(float)));
    CK(cudaMemset(cells.velocities_y, 0, cap * sizeof(float)));

    printf("[GPU] %d cells (cap %d), T=%d, pool=%.1f MB, S=%.1f MB (%dx%d)\n",
           n, cap, TILE_T, pool_bytes / 1e6, S_bytes / 1e6,
           params.Nx, params.Ny);

    // Migration buffers (G > 1 only). Sized once at init; never reallocated.
    if (gpus > 1) {
        // Per-direction migration capacity scales with cell capacity so a
        // full boundary "row" can migrate in one rebind cycle even at
        // high N or G. capacity/16 gives ~3× headroom over realistic
        // boundary-row counts (~120 cells at N=12800,G=4) while keeping
        // the pack buffer footprint tight: 4 buffers × capacity/16 ×
        // ~410 KB/cell ≈ 656 MB per rank at N=12800.
        max_migrants_per_dir = std::max(MAX_MIGRANTS_DEFAULT, cap / 16);
        // 5 migration counters in one contiguous buffer so the host can
        // download them with a single cudaMemcpyAsync per phase (was
        // 3+2 separate copies; the nsys profile showed cudaMemcpyAsync
        // was 84% of API time, dominated by these per-rebind syncs).
        //   layout: [stay, up, down, in_prev, in_next]
        CK(cudaMalloc(&d_mig_counts, 5 * sizeof(int)));
        d_n_stay    = d_mig_counts + 0;
        d_n_up      = d_mig_counts + 1;
        d_n_down    = d_mig_counts + 2;
        d_n_in_prev = d_mig_counts + 3;
        d_n_in_next = d_mig_counts + 4;
        CK(cudaMalloc(&d_stay_idx,  cap * sizeof(int)));
        CK(cudaMalloc(&d_up_idx,    cap * sizeof(int)));
        CK(cudaMalloc(&d_down_idx,  cap * sizeof(int)));

        const size_t pack_buf_bytes =
            (size_t)max_migrants_per_dir * CELL_PACK_BYTES;
        CK(cudaMalloc(&d_pack_up,      pack_buf_bytes));
        CK(cudaMalloc(&d_pack_down,    pack_buf_bytes));
        CK(cudaMalloc(&d_pack_in_prev, pack_buf_bytes));
        CK(cudaMalloc(&d_pack_in_next, pack_buf_bytes));

        // Scratch arrays for compaction. One mirror per relevant per-cell
        // field. We do NOT scratch volumes/Ix/Iy/etc — those are reduction
        // outputs that the next step recomputes from scratch anyway.
        CK(cudaMalloc(&d_origin_scratch,      2 * cap * sizeof(int)));
        CK(cudaMalloc(&d_rect_scratch,        4 * cap * sizeof(int)));
        for (auto& f : per_cell_float_state()) {
            if (f.scratch_ptr) {
                CK(cudaMalloc(f.scratch_ptr, cap * sizeof(float)));
            }
        }
        CK(cudaMalloc(&d_rng_scratch,         cap * sizeof(curandState)));
        // Persistent global-id buffers (replaces per-migration cudaMallocAsync).
        CK(cudaMalloc(&d_gid_src,             cap * sizeof(int)));
        CK(cudaMalloc(&d_gid_arr,             cap * sizeof(int)));

        printf("[migration] alloc: max %d migrants/dir (capacity=%d), "
               "pack_bytes=%zu, total %zu MB per-rank\n",
               max_migrants_per_dir, cap, CELL_PACK_BYTES,
               4 * pack_buf_bytes / (1024 * 1024));
    }

    configure_l2_persistence();
}

// ---------------------------------------------------------------------------
// configure_l2_persistence — pin the global S field in L2 cache via the
// CUDA access-policy-window mechanism. S is read once per pixel by every
// cell's tile (k_evolve_l1 pass-1 reads + pass-2 reads), so it gets reused
// O(N_cells_overlapping) times per step. Pinning it as "persisting" tells
// the L2 controller to prefer it over the streaming reads of phi tiles.
//
// On Ada (RTX 4090, sm_8.9):
//   - Max persisting carveout ~ 32 MB (half of 64 MB total L2)
// On Hopper (H100, sm_9.0):
//   - Max persisting carveout ~ 44 MB
//
// Guard: if S exceeds the carveout, persistence will thrash the cache
// (hitRatio drops, evictions overwhelm the gain). We skip in that case
// and let the default LRU policy run.
// ---------------------------------------------------------------------------
void Simulation::configure_l2_persistence() {
    if (cells.S == nullptr || cells.num_cells == 0) return;
    if (sim_env_flag_enabled("CELL_SIM_DISABLE_L2_PERSIST")) {
        printf("[L2] persistence disabled by CELL_SIM_DISABLE_L2_PERSIST.\n");
        return;
    }

    int dev = 0;
    cudaGetDevice(&dev);

    int max_persist_bytes = 0;
    cudaDeviceGetAttribute(&max_persist_bytes,
                           cudaDevAttrMaxPersistingL2CacheSize, dev);
    if (max_persist_bytes <= 0) {
        printf("[L2] persistence not supported on this device; skipping.\n");
        return;
    }

    // S is sized by the slab's extended height, not Ny (== Ny for G=1).
    const size_t S_bytes = (size_t)cells.S_ext_height * params.Nx * sizeof(float);

    // Use the full available carveout. When S fits, pin all of it; when
    // S exceeds the carveout, pin the leading carveout-worth of bytes
    // (testsim approach). Even partial pinning gives a measurable win
    // because the hot rebind / reduce / RHS reads of S are spatially
    // localized to whichever cells happen to be in flight.
    const size_t persist_size = std::min((size_t)max_persist_bytes, S_bytes);
    const size_t window_bytes = persist_size;

    // Reserve the carveout. cudaDeviceSetLimit grows the carveout to the
    // requested size if larger than current.
    cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                         persist_size);
    if (err != cudaSuccess) {
        printf("[L2] cudaDeviceSetLimit failed: %s; skipping persistence.\n",
               cudaGetErrorString(err));
        return;
    }

    // Attach the access policy window to the default (legacy) stream.
    // hitRatio = 1.0 means every access in the window is a candidate for
    // persistence; hitProp = persisting (kept), missProp = streaming
    // (don't pollute L2).
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = cells.S;
    attr.accessPolicyWindow.num_bytes = window_bytes;
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    err = cudaStreamSetAttribute(cudaStreamDefault,
                                 cudaStreamAttributeAccessPolicyWindow,
                                 &attr);
    if (err != cudaSuccess) {
        printf("[L2] cudaStreamSetAttribute failed: %s; skipping persistence.\n",
               cudaGetErrorString(err));
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
        return;
    }

    if (S_bytes <= (size_t)max_persist_bytes) {
        printf("[L2] persisting full S (%.1f MB) in carveout (max %.1f MB)\n",
               S_bytes / 1e6, max_persist_bytes / 1e6);
    } else {
        printf("[L2] persisting first %.1f MB of S=%.1f MB (carveout max %.1f MB)\n",
               window_bytes / 1e6, S_bytes / 1e6, max_persist_bytes / 1e6);
    }
}

// ---------------------------------------------------------------------------
// upload_initial_state — push origins, per-cell scalars, and initial polar
// angles to the device for a *fresh* init. phi tiles are filled by
// k_init_phi (host -> small d_cx/d_cy temporaries -> device kernel).
// ---------------------------------------------------------------------------
void Simulation::upload_initial_state() {
    const int n = (int)h_cells.size();

    std::vector<int>   h_origin(2 * n);
    std::vector<float> h_g(n), h_vA(n), h_tr(n);
    std::vector<float> h_th(n), h_px(n), h_py(n);
    std::vector<float> h_cx(n), h_cy(n);

    // Deterministic per-cell theta seeded by the GLOBAL cell id, not the
    // local iteration index, so the same physical cell gets the same theta
    // regardless of GPU count. The previous version called rand() per
    // local cell, which produced different thetas in G=4 vs G=1 (rand()
    // is process-global and slice_cells_to_local left only this rank's
    // cells in h_cells, so iteration order was rank-dependent).
    auto theta_for = [&](int gid) {
        // splitmix-style 64-bit hash → float in [0, 2*pi)
        uint64_t z = (uint64_t)params.polarity_seed * 0x9E3779B97F4A7C15ULL
                   + (uint64_t)gid * 0xBF58476D1CE4E5B9ULL
                   + 0x94D049BB133111EBULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z =  z ^ (z >> 31);
        // Use top 24 bits for f32 mantissa precision.
        float u = (float)((z >> 40) & ((1u<<24)-1)) / (float)(1u<<24);
        return u * 2.0f * (float)M_PI;
    };

    for (int i = 0; i < n; i++) {
        const auto& c = h_cells[i];
        h_origin[2*i + 0] = c.ox;
        h_origin[2*i + 1] = c.oy;
        h_g [i] = (float)c.gamma;
        h_vA[i] = (float)c.v_A;
        h_tr[i] = (float)c.radius;
        int gid = (gpus > 1 && (int)h_global_id.size() > i) ? h_global_id[i] : i;
        float theta = theta_for(gid);
        h_th[i] = theta;
        h_px[i] = cosf(theta);
        h_py[i] = sinf(theta);
        h_cx[i] = (float)c.cx;
        h_cy[i] = (float)c.cy;
    }

    CK(cudaMemcpy(cells.origin,      h_origin.data(), 2*n*sizeof(int),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.gamma_cell,  h_g.data(),      n*sizeof(float),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.v_A_cell,    h_vA.data(),     n*sizeof(float),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.tgt_radius,  h_tr.data(),     n*sizeof(float),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_theta, h_th.data(),     n*sizeof(float),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_x,     h_px.data(),     n*sizeof(float),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_y,     h_py.data(),     n*sizeof(float),    cudaMemcpyHostToDevice));

    // Initial rect: full tile minus 1px stencil halo. The first rebind
    // (after REBIND_EVERY-1 evolve steps) will shrink/grow it from sigma.
    std::vector<int> h_rect(4 * n);
    for (int i = 0; i < n; i++) {
        h_rect[4*i + 0] = 1;
        h_rect[4*i + 1] = 1;
        h_rect[4*i + 2] = TILE_T - 2;
        h_rect[4*i + 3] = TILE_T - 2;
    }
    CK(cudaMemcpy(cells.rect, h_rect.data(), 4*n*sizeof(int), cudaMemcpyHostToDevice));

    // Stage cx/cy on the device just for k_init_phi (freed immediately).
    float *d_cx = nullptr, *d_cy = nullptr;
    CK(cudaMalloc(&d_cx, n * sizeof(float)));
    CK(cudaMalloc(&d_cy, n * sizeof(float)));
    CK(cudaMemcpy(d_cx, h_cx.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cy, h_cy.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    launch_init_phi(cells, params, d_cx, d_cy);

    cudaFree(d_cx);
    cudaFree(d_cy);
}

// ---------------------------------------------------------------------------
// apply_gamma_spec — parse --gamma string into per-cell h_cells[i].gamma.
// Supported syntaxes:
//   "<f>"                          all cells
//   "<f>:cell<k>"                  only cell k
//   "<f>:nearest(x,y)"             cell closest to (x,y)
//   "<f>:cluster(p%,x,y)"          floor(p% * N) cells nearest to (x,y)
//   "<f>:<p>%"                     floor(p% * N) cells (lowest-id first)
// Multiple segments may be chained with ';' (composable --gamma flags):
//   "1.0;0.35:cell0;0.35:nearest(500,500)" sets baseline 1.0, then
//   overrides cell 0 and the cell nearest (500,500) to 0.35.
// Unrecognised strings fall back to "all cells = parsed scalar".
// ---------------------------------------------------------------------------
static void apply_one_gamma_segment(std::vector<CellHost>& h_cells,
                                    const std::string& spec) {
    if (spec.empty()) return;
    const int n = (int)h_cells.size();
    if (n == 0) return;

    double soft_gamma = atof(spec.c_str());
    size_t colon = spec.find(':');
    if (colon == std::string::npos) {
        for (auto& c : h_cells) c.gamma = soft_gamma;
        return;
    }
    std::string sel = spec.substr(colon + 1);

    auto pct_to_count = [&](double pct) {
        int k = (int)std::floor(pct * 0.01 * n);
        if (k < 1) k = 1;
        if (k > n) k = n;
        return k;
    };

    if (sel.rfind("cell", 0) == 0) {
        int k = atoi(sel.c_str() + 4);
        if (k >= 0 && k < n) h_cells[k].gamma = soft_gamma;
    } else if (sel.rfind("nearest(", 0) == 0) {
        double x = 0, y = 0;
        sscanf(sel.c_str() + 8, "%lf,%lf", &x, &y);
        int best = 0; double bd = 1e300;
        for (int i = 0; i < n; i++) {
            double dx = h_cells[i].cx - x;
            double dy = h_cells[i].cy - y;
            double d = dx*dx + dy*dy;
            if (d < bd) { bd = d; best = i; }
        }
        h_cells[best].gamma = soft_gamma;
    } else if (sel.rfind("cluster(", 0) == 0) {
        double pct = 0, x = 0, y = 0;
        sscanf(sel.c_str() + 8, "%lf%%,%lf,%lf", &pct, &x, &y);
        int k = pct_to_count(pct);
        std::vector<std::pair<double,int>> ranked(n);
        for (int i = 0; i < n; i++) {
            double dx = h_cells[i].cx - x;
            double dy = h_cells[i].cy - y;
            ranked[i] = {dx*dx + dy*dy, i};
        }
        std::sort(ranked.begin(), ranked.end());
        for (int i = 0; i < k; i++) h_cells[ranked[i].second].gamma = soft_gamma;
    } else if (sel.find('%') != std::string::npos) {
        double pct = atof(sel.c_str());
        int k = pct_to_count(pct);
        for (int i = 0; i < k; i++) h_cells[i].gamma = soft_gamma;
    } else {
        // Fallback: treat the whole spec as a bare number.
        for (auto& c : h_cells) c.gamma = soft_gamma;
    }
}

void Simulation::apply_gamma_spec() {
    if (gamma_spec.empty()) return;
    // Split on ';' so multiple --gamma flags compose. Segments apply in
    // CLI order; later segments overwrite earlier ones for overlapping
    // cells, which is intuitive: `--gamma 1.0 --gamma 0.35:cell0` leaves
    // every cell at 1.0 except cell 0 at 0.35.
    size_t start = 0;
    while (start <= gamma_spec.size()) {
        size_t sep = gamma_spec.find(';', start);
        std::string seg = (sep == std::string::npos)
            ? gamma_spec.substr(start)
            : gamma_spec.substr(start, sep - start);
        if (!seg.empty()) apply_one_gamma_segment(h_cells, seg);
        if (sep == std::string::npos) break;
        start = sep + 1;
    }
}

// ---------------------------------------------------------------------------
// apply_v_A_disorder — log-normal disorder on v_A (fresh init only).
// Box-Muller draws are parameterised so the *output std-dev* matches the
// requested sigma when v_A_sigma != 0. Re-uses the placement seed XOR'd
// with a fixed mixer to keep results reproducible.
// ---------------------------------------------------------------------------
void Simulation::apply_v_A_disorder() {
    if (v_A_sigma <= 0.0 || params.v_A <= 0.0) return;
    const int n = (int)h_cells.size();
    if (n == 0) return;

    double cv = v_A_sigma / params.v_A;
    double sigma_ln = std::sqrt(std::log(1.0 + cv * cv));
    double mu_ln    = std::log(params.v_A) - 0.5 * sigma_ln * sigma_ln;

    unsigned seed = (params.seed ? params.seed : 42u) ^ 0x9E3779B9u;
    srand(seed);
    auto u01 = []() { return ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0); };

    for (int i = 0; i < n; i++) {
        double u1 = u01(), u2 = u01();
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        h_cells[i].v_A = std::exp(mu_ln + sigma_ln * z);
    }
}

// ---------------------------------------------------------------------------
// finalize_init — RNG, initial scatter+velocity reduction, sync.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// finalize_init — three single-purpose steps, sequenced.
// ---------------------------------------------------------------------------
void Simulation::seed_rng_if_fresh() {
    // Preserve random-stream continuity on resume: only initialize the
    // per-cell curandStates here when they were NOT restored from the
    // checkpoint's RNGS sidecar. (Without this guard, a chained run with
    // the same polarity_seed re-seeds each cell back to offset 0 and
    // replays the same tumble decisions across resume.)
    if (rng_restored_from_ckpt) return;
    const unsigned long polar_seed = params.polarity_seed
                                     ? params.polarity_seed
                                     : (params.seed ? params.seed : 1234u);
    // For multi-GPU: upload h_global_id to d_gid_src so each cell's
    // curand sequence number matches its global ID, not its local index.
    // This ensures the polarity noise stream for a given physical cell
    // is identical regardless of GPU count.
    const int* d_ids = nullptr;
    if (gpus > 1 && !h_global_id.empty() && d_gid_src) {
        CK(cudaMemcpy(d_gid_src, h_global_id.data(),
                      h_global_id.size() * sizeof(int),
                      cudaMemcpyHostToDevice));
        d_ids = d_gid_src;
    }
    launch_rng_init(cells, polar_seed, d_ids);
}

void Simulation::compute_initial_velocities() {
    launch_initial_velocity(cells, params);
}

void Simulation::setup_step_stream() {
    if (step_stream) return;  // idempotent — only runs once per Simulation
    // Dedicated stream for the captured step pipeline. Non-blocking so it
    // doesn't serialize against the default stream used by I/O paths.
    CK(cudaStreamCreateWithFlags(&step_stream, cudaStreamNonBlocking));
    if (sim_env_flag_enabled("CELL_SIM_DISABLE_L2_PERSIST")) return;
    // Mirror the L2 access-policy-window onto step_stream so the
    // S-field benefits from persistence on the captured/replayed
    // kernels too. We size the window at min(S_bytes, carveout) so
    // even oversize S gets the leading slice pinned.
    int max_persist_bytes = 0;
    cudaDeviceGetAttribute(&max_persist_bytes,
                           cudaDevAttrMaxPersistingL2CacheSize, 0);
    if (max_persist_bytes <= 0) return;
    const size_t S_bytes = (size_t)cells.S_ext_height * params.Nx * sizeof(float);
    const size_t window_bytes = std::min((size_t)max_persist_bytes, S_bytes);
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = cells.S;
    attr.accessPolicyWindow.num_bytes = window_bytes;
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(step_stream,
                           cudaStreamAttributeAccessPolicyWindow, &attr);
}

void Simulation::finalize_init() {
    seed_rng_if_fresh();
    compute_initial_velocities();
    setup_step_stream();
    CK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// init — fresh start.
// ---------------------------------------------------------------------------
void Simulation::init(const SimParams& p, int n_cells) {
    params = p;

    // If no seed was supplied, draw a non-deterministic one. Storing it
    // back into params makes the actual seed reproducible (it gets written
    // into the checkpoint SimParams blob) and avoids the historical bug
    // where every "unseeded" run silently used seed=42.
    if (params.seed == 0) {
        std::random_device rd;
        unsigned s = rd();
        if (s == 0) s = 0xC0FFEEu;  // re-roll the unlikely 0
        params.seed = s;
    }
    if (params.polarity_seed == 0) {
        std::random_device rd;
        unsigned s = rd();
        if (s == 0) s = 0xDECAFBADu;
        params.polarity_seed = s;
    }
    printf("[SIM] seed=%u polarity_seed=%u\n",
           params.seed, params.polarity_seed);

    if (params.Nx < TILE_T || params.Ny < TILE_T) {
        fprintf(stderr,
                "[FATAL] domain (%d x %d) smaller than TILE_T=%d. "
                "Increase -N or reduce confluence so L >= %d.\n",
                params.Nx, params.Ny, TILE_T, TILE_T);
        std::exit(1);
    }

    // Default trajectory cadence: traj_every from trajectory_samples.
    if (params.trajectory_samples > 0 && params.t_end > 0 && params.dt > 0) {
        long long total_steps = (long long)(params.t_end / params.dt + 0.5);
        traj_every = (int)((total_steps + params.trajectory_samples - 1)
                            / params.trajectory_samples);
        if (traj_every < 1) traj_every = 1;
    }

    place_cells(n_cells, params.target_radius);
    apply_gamma_spec();
    apply_v_A_disorder();
    slice_cells_to_local();
    compute_origins();
    alloc_gpu();
    upload_initial_state();
    finalize_init();
    if (gpus > 1) {
        printf("[SIM] init rank=%d/%d: %d/%d cells (offset=%d), "
               "t_end=%.1f, dt=%.4f, traj_every=%d\n",
               rank, gpus, (int)h_cells.size(), cells_global, cell_offset,
               params.t_end, params.dt, traj_every);
    } else {
        printf("[SIM] init: %d cells, t_end=%.1f, dt=%.4f, traj_every=%d\n",
               n_cells, params.t_end, params.dt, traj_every);
    }
}

// ---------------------------------------------------------------------------
// init_from_checkpoint — versions 3..7. v3-v6 use variable W/H tiles
// (legacy format); we re-tile them into TILE_T uniform buffers on load.
// v7 is the native format produced by save_checkpoint() below.
// v8 adds (num_ranks, rank_id, num_cells_global) immediately after T_w.
// For v8 multi-rank checkpoints, each rank reads its own per-rank file
// (path = <dir>/rank{K}/checkpoint.bin for K>0; rank 0 reuses the supplied
// path). Resuming with a different --gpus is unsupported in C++; use
// `cell_analyze merge-ckpt` to consolidate first.
// ---------------------------------------------------------------------------
namespace {
// Peek the v8 header fields without disturbing the main reader's logic.
// Returns true if the file is v8 and the three new fields were populated.
// On v3..v7 or on any error, returns false with outputs left at defaults.
bool peek_v8_rank_header(const std::string& path,
                         int32_t& num_ranks, int32_t& rank_id,
                         int32_t& num_cells_global) {
    num_ranks = 1; rank_id = 0; num_cells_global = 0;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    ckpt::FixedPrefix prefix{};
    if (fread(&prefix, sizeof(prefix), 1, f) != 1 ||
        prefix.magic != ckpt::MAGIC || prefix.version < 8) {
        fclose(f); return false;
    }
    // Skip SimParams blob + tile_t(4). Now at the v8 trailer.
    if (fseek(f, (long)prefix.sp_sz + (long)sizeof(int32_t), SEEK_CUR) != 0) {
        fclose(f); return false;
    }
    ckpt::RankTrailer trailer{};
    if (fread(&trailer, sizeof(trailer), 1, f) != 1) {
        fclose(f); return false;
    }
    num_ranks        = trailer.num_ranks;
    rank_id          = trailer.rank_id;
    num_cells_global = trailer.num_cells_global;
    fclose(f);
    return true;
}

// ----------------------------------------------------------------------------
// CheckpointPayload — pure-data result of read_checkpoint.
// Holds everything we read from disk. No GPU pointers, no Simulation state.
// apply_checkpoint() consumes this to populate the Simulation.
// ----------------------------------------------------------------------------
struct CheckpointPayload {
    uint32_t version = 0;
    SimParams params{};       // decoded; legacy fixups already applied
    int32_t   step_count = 0;
    double    cur_time   = 0.0;
    int32_t   num_cells  = 0;
    // v8 multi-rank trailer (defaults match single-rank for v3..v7).
    int32_t   v8_num_ranks = 1;
    int32_t   v8_rank_id   = 0;
    int32_t   v8_n_global  = 0;
    // Per-cell records (size = num_cells).
    std::vector<float>              cx, cy, vx, vy, vol;
    std::vector<int>                ox, oy;
    std::vector<int>                gid;       // global ids; local index for pre-v8
    std::vector<std::vector<float>> phi_tiles; // each TILE_AREA floats
    // Optional sidecars (empty if absent on disk).
    std::vector<float>   per_vA, per_gamma, per_radius, per_polar_theta;
    std::vector<uint8_t> per_rng_bytes;
};

// Resolve `path_in` to a per-rank file. For v8 multi-rank checkpoints with
// num_ranks > 1, rank K>0 reads <dir>/rank{K}/<basename>. Validates that
// --gpus matches the checkpoint layout (same-G resumes only in C++).
bool resolve_checkpoint_path(const std::string& path_in,
                             int gpus, int rank,
                             std::string& out_path) {
    out_path = path_in;
    int32_t ck_num_ranks = 1, ck_rank_id = 0, ck_n_global = 0;
    if (!peek_v8_rank_header(path_in, ck_num_ranks, ck_rank_id, ck_n_global) ||
        ck_num_ranks <= 1) {
        return true;  // not a multi-rank checkpoint; nothing to redirect
    }
    if (gpus != ck_num_ranks) {
        fprintf(stderr,
            "[ckpt] checkpoint was saved with --gpus %d but you "
            "requested --gpus %d. Use `cell_analyze merge-ckpt %s` "
            "to consolidate into a single-rank checkpoint, then "
            "resume with any --gpus value.\n",
            ck_num_ranks, gpus, path_in.c_str());
        return false;
    }
    if (rank > 0) {
        size_t slash = path_in.find_last_of('/');
        std::string dir  = (slash == std::string::npos) ? "." : path_in.substr(0, slash);
        std::string base = (slash == std::string::npos) ? path_in : path_in.substr(slash + 1);
        out_path = dir + "/rank" + std::to_string(rank) + "/" + base;
    }
    return true;
}

// Decode the SimParams blob according to its on-disk variant.
// v6/v7/v8: native layout, blit straight into the struct.
// v3/v4/v5: packed f32 layout with two sub-variants keyed by sp_sz.
// Returns false if the v7/v8 SimParams size doesn't match this build's
// sizeof(SimParams) — that's a hard incompatibility we can't recover from.
bool decode_simparams(FILE* f, uint32_t ver, uint32_t sp_sz, SimParams& params) {
    if (sp_sz == sizeof(SimParams) && (ver == 6 || ver == 7 || ver == 8)) {
        fread(&params, sp_sz, 1, f);
        // subdomain_padding was a dead field in v6/v7 ckpts; reset only
        // those legacy versions. v8 writes the live adaptive-rect value.
        if (ver < 8) params.subdomain_padding = SimParams{}.subdomain_padding;
        return true;
    }
    if (ver == 7 || ver == 8) {
        fprintf(stderr, "v%u checkpoint with foreign SimParams (sp=%u, ours=%zu)\n",
                ver, sp_sz, sizeof(SimParams));
        return false;
    }
    // Legacy v3/v4/v5: SimParams was a packed f32 layout. Two known
    // sub-variants live in the wild; we key off sp_sz:
    //
    //   sp_sz == 72 (or 76 with abp): old pre-cutover baseline.
    //     Nx Ny dx dy dt t_end λ γ κ R μ v_A ξ τ halo
    //     offsets: 0 4 8 12 16 20 24 28 32 36 40 44 48 52 60
    //
    //   sp_sz == 92: v4 produced by the roihu binary (May 2026).
    //     Nx Ny dx dy dt t_end N_cells λ γ κ R μ v_A ξ τ halo print_int sub_pad ... gamma_soft
    //     offsets: 0 4 8 12 16 20  24    28 32 36 40 44 48 52 56 60 64 68 ... 84
    //
    // The v4 layout inserts an int32 cell-count N at @24 that shifts all
    // the f32 physics scalars (λ..τ) by +4. `halo` sits at @60 in both
    // layouts. Mis-reading this caused the τ=1500 silent corruption in
    // FSS pipeline runs resumed from May-6 roihu equilibrations.
    std::vector<uint8_t> sp(sp_sz);
    fread(sp.data(), 1, sp_sz, f);
    auto u_i32 = [&](size_t off) {
        int32_t v; std::memcpy(&v, sp.data() + off, 4); return v;
    };
    auto u_f32 = [&](size_t off) {
        float v; std::memcpy(&v, sp.data() + off, 4); return v;
    };
    params = SimParams{};
    params.Nx    = u_i32(0);
    params.Ny    = u_i32(4);
    params.dx    = u_f32(8);
    params.dy    = u_f32(12);
    params.dt    = u_f32(16);
    params.t_end = u_f32(20);
    const bool is_v4_roihu = (sp_sz == 92);
    const size_t shift = is_v4_roihu ? 4 : 0;
    params.lambda        = u_f32(24 + shift);
    params.gamma         = u_f32(28 + shift);
    params.kappa         = u_f32(32 + shift);
    params.target_radius = u_f32(36 + shift);
    params.mu            = u_f32(40 + shift);
    params.v_A           = u_f32(44 + shift);
    params.xi            = u_f32(48 + shift);
    params.tau           = u_f32(52 + shift);
    params.halo          = u_i32(60);
    if (is_v4_roihu) {
        if (sp_sz >= 68) params.print_interval = u_i32(64);
    } else if (sp_sz >= 76) {
        params.abp = (u_i32(72) == 1);
    }
    params.subdomain_padding = SimParams{}.subdomain_padding;
    if (params.print_interval == 0) params.print_interval = 100;
    params.trajectory_samples = 0;
    fprintf(stderr,
            "[ckpt] legacy v%u SimParams: sp_sz=%u → %s layout "
            "(τ=%.1f, ξ=%.1f, λ=%.1f, γ=%.2f, halo=%d)\n",
            ver, sp_sz,
            (is_v4_roihu ? "v4-roihu (shifted)" : "baseline-72"),
            params.tau, params.xi, params.lambda, params.gamma,
            params.halo);
    return true;
}

// Read per-cell records into the payload. Handles v7/v8 native (uniform
// TILE_T tiles) and legacy v3-v6 (variable W/H tiles, re-tile centred).
void decode_cell_records(FILE* f, uint32_t ver, int Tin, int halo,
                         CheckpointPayload& p) {
    const int nc = p.num_cells;
    p.cx.resize(nc); p.cy.resize(nc); p.vx.resize(nc); p.vy.resize(nc); p.vol.resize(nc);
    p.ox.resize(nc); p.oy.resize(nc);
    p.gid.resize(nc);
    p.phi_tiles.resize(nc);
    for (int i = 0; i < nc; ++i) p.gid[i] = i;

    if (ver == 7 || ver == 8) {
        const size_t Tin_area = (size_t)Tin * Tin;
        for (int i = 0; i < nc; ++i) {
            ckpt::CellRecordHeader rec{};
            fread(&rec, sizeof(rec), 1, f);
            if (ver == 8) p.gid[i] = rec.cell_id;
            p.cx[i] = rec.cx;  p.cy[i] = rec.cy;
            p.vx[i] = rec.vx;  p.vy[i] = rec.vy;
            p.vol[i] = rec.volume;
            p.ox[i] = rec.origin_x;  p.oy[i] = rec.origin_y;
            p.phi_tiles[i].assign(TILE_AREA, 0.0f);
            if (Tin == TILE_T) {
                fread(p.phi_tiles[i].data(), sizeof(float), TILE_AREA, f);
            } else {
                std::vector<float> tile_in(Tin_area);
                fread(tile_in.data(), sizeof(float), Tin_area, f);
                int dx = (TILE_T - Tin) / 2;
                int dy = (TILE_T - Tin) / 2;
                p.ox[i] = rec.origin_x - dx;
                p.oy[i] = rec.origin_y - dy;
                for (int ly = 0; ly < Tin; ++ly) {
                    int dst_y = ly + dy;
                    if (dst_y < 0 || dst_y >= TILE_T) continue;
                    for (int lx = 0; lx < Tin; ++lx) {
                        int dst_x = lx + dx;
                        if (dst_x < 0 || dst_x >= TILE_T) continue;
                        p.phi_tiles[i][dst_y * TILE_T + dst_x] = tile_in[ly * Tin + lx];
                    }
                }
            }
        }
        return;
    }
    // Legacy v3-v6: variable W/H tiles. Re-tile each cell into a
    // centred TILE_T x TILE_T buffer.
    for (int i = 0; i < nc; ++i) {
        int32_t cid;
        int32_t x0, y0, x1, y1;
        fread(&cid, 4, 1, f);
        fread(&x0, 4, 1, f); fread(&y0, 4, 1, f);
        fread(&x1, 4, 1, f); fread(&y1, 4, 1, f);
        fread(&p.cx[i],  4, 1, f); fread(&p.cy[i],  4, 1, f);
        fread(&p.vx[i],  4, 1, f); fread(&p.vy[i],  4, 1, f);
        fread(&p.vol[i], 4, 1, f);

        int w = (x1 - x0) + 2 * halo;
        int h = (y1 - y0) + 2 * halo;
        int ox_legacy = x0 - halo;
        int oy_legacy = y0 - halo;
        std::vector<float> tile_legacy((size_t)w * h);
        fread(tile_legacy.data(), sizeof(float), (size_t)w * h, f);

        int dx = (TILE_T - w) / 2;
        int dy = (TILE_T - h) / 2;
        p.ox[i] = ox_legacy - dx;
        p.oy[i] = oy_legacy - dy;
        p.phi_tiles[i].assign(TILE_AREA, 0.0f);
        for (int ly = 0; ly < h; ++ly) {
            int dst_y = ly + dy;
            if (dst_y < 0 || dst_y >= TILE_T) continue;
            for (int lx = 0; lx < w; ++lx) {
                int dst_x = lx + dx;
                if (dst_x < 0 || dst_x >= TILE_T) continue;
                p.phi_tiles[i][dst_y * TILE_T + dst_x] = tile_legacy[ly * w + lx];
            }
        }
    }
}

// Read magic-tagged per-cell sidecar blocks (VA_A/GAMA/RADI/POLR/RNGS) into
// the payload. Stops at the first unrecognized magic (or EOF) and rewinds
// so callers don't lose position.
void decode_sidecars(FILE* f, CheckpointPayload& p) {
    while (true) {
        long pos = ftell(f);
        ckpt::SidecarBlockHeader sh{};
        if (fread(&sh, sizeof(sh), 1, f) != 1) break;
        if (sh.magic == ckpt::MAGIC_VA_A || sh.magic == ckpt::MAGIC_GAMA ||
            sh.magic == ckpt::MAGIC_RADI || sh.magic == ckpt::MAGIC_POLR) {
            std::vector<float> data(sh.count);
            fread(data.data(), sizeof(float), sh.count, f);
            if      (sh.magic == ckpt::MAGIC_VA_A) p.per_vA          = std::move(data);
            else if (sh.magic == ckpt::MAGIC_GAMA) p.per_gamma       = std::move(data);
            else if (sh.magic == ckpt::MAGIC_RADI) p.per_radius      = std::move(data);
            else                                   p.per_polar_theta = std::move(data);
        } else if (sh.magic == ckpt::MAGIC_RNGS) {
            const size_t payload = (size_t)sh.count * sizeof(curandState);
            p.per_rng_bytes.assign(payload, 0);
            if (payload > 0) fread(p.per_rng_bytes.data(), 1, payload, f);
        } else {
            fseek(f, pos, SEEK_SET);
            break;
        }
    }
}

// Top-level pure file I/O: read whatever is on disk into a CheckpointPayload.
// Touches NO Simulation state. Returns false on missing file, bad magic, or
// unrecoverable SimParams layout mismatch.
bool read_checkpoint(const std::string& path, CheckpointPayload& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open checkpoint %s\n", path.c_str());
        return false;
    }

    uint32_t magic = 0, ver = 0;
    fread(&magic, 4, 1, f);
    fread(&ver,   4, 1, f);
    if (magic != ckpt::MAGIC) {
        fprintf(stderr, "Not a cell_sim checkpoint (magic=0x%08x)\n", magic);
        fclose(f); return false;
    }
    out.version = ver;

    int32_t cs = 0; double ct_f64 = 0.0; int32_t nc = 0;
    int32_t si = 0, ci2 = 0, ts = 0; uint8_t bools[4]{}; uint32_t sp_sz = 0;
    if (ver >= 5) {
        ckpt::FixedPrefix prefix{};
        prefix.magic   = magic;
        prefix.version = ver;
        const size_t tail_bytes = sizeof(prefix) - 8;
        fread(reinterpret_cast<uint8_t*>(&prefix) + 8, tail_bytes, 1, f);
        cs     = prefix.step;
        ct_f64 = prefix.cur_time;
        nc     = prefix.num_cells_local;
        si     = prefix.save_interval;
        ci2    = prefix.reserved;
        ts     = prefix.trajectory_samples;
        std::memcpy(bools, prefix.bools, 4);
        sp_sz  = prefix.sp_sz;
    } else {
        // v3/v4: cur_time was f32 (4 bytes), 4-byte-shorter prefix.
        fread(&cs, 4, 1, f);
        float ct_f32 = 0.0f;
        fread(&ct_f32, 4, 1, f);
        ct_f64 = ct_f32;
        fread(&nc, 4, 1, f);
        fread(&si, 4, 1, f);
        fread(&ci2, 4, 1, f);
        fread(&ts, 4, 1, f);
        fread(bools, 1, 4, f);
        fread(&sp_sz, 4, 1, f);
    }
    (void)si; (void)ci2; (void)ts; (void)bools;

    if (!decode_simparams(f, ver, sp_sz, out.params)) {
        fclose(f); return false;
    }
    out.step_count = cs;
    out.cur_time   = ct_f64;
    out.num_cells  = nc;
    out.v8_n_global = nc;  // sensible default; v8 reader overrides below

    int Tin = TILE_T;
    int halo_legacy = out.params.halo;
    if (ver == 7 || ver == 8) {
        int32_t T_in = 0;
        fread(&T_in, 4, 1, f);
        Tin = T_in;
        if (Tin != TILE_T) {
            fprintf(stderr, "[ckpt] v%u TILE_T re-tile: file=%d build=%d\n",
                    ver, Tin, TILE_T);
        }
        if (ver == 8) {
            ckpt::RankTrailer trailer{};
            fread(&trailer, sizeof(trailer), 1, f);
            out.v8_num_ranks = trailer.num_ranks;
            out.v8_rank_id   = trailer.rank_id;
            out.v8_n_global  = trailer.num_cells_global;
        }
    }
    decode_cell_records(f, ver, Tin, halo_legacy, out);
    if (ver < 7) {
        // Legacy format had no halo concept after re-tile; record 0.
        out.params.halo = 0;
    }
    decode_sidecars(f, out);
    fclose(f);
    return true;
}
}  // namespace

// ---------------------------------------------------------------------------
// init_from_checkpoint — thin orchestrator.
//
// 1. resolve path (v8 multi-rank dispatch)
// 2. read_checkpoint(path) — pure file I/O, returns a CheckpointPayload
// 3. adopt payload into this Simulation (params, step_count, cur_time, h_cells)
// 4. apply CLI overrides, recompute traj cadence, build slab layout
// 5. alloc_gpu, resolve per-cell scalar policy (gamma_spec, v_A_sigma, sidecars)
// 6. upload to GPU, restore RNG if RNGS sidecar present
// 7. finalize_init
// ---------------------------------------------------------------------------
bool Simulation::init_from_checkpoint(const std::string& path_in,
                                      const SimParams& cli,
                                      const SimOverrides& ov) {
    std::string path;
    if (!resolve_checkpoint_path(path_in, gpus, rank, path)) return false;

    CheckpointPayload payload;
    if (!read_checkpoint(path, payload)) return false;

    // ---- Adopt the payload into Simulation state ----
    params      = payload.params;
    step_count  = payload.step_count;
    cur_time    = payload.cur_time;
    const int nc = payload.num_cells;

    // ---- CLI overrides on top of the loaded params ----
    if (ov.t_end)              params.t_end = cli.t_end;
    if (ov.dt)                 params.dt = cli.dt;
    if (ov.v_A)                params.v_A = cli.v_A;
    if (ov.tau)                params.tau = cli.tau;
    if (ov.gamma)              params.gamma = cli.gamma;
    if (ov.kappa)              params.kappa = cli.kappa;
    if (ov.mu)                 params.mu = cli.mu;
    if (ov.xi)                 params.xi = cli.xi;
    if (ov.lambda)             params.lambda = cli.lambda;
    if (ov.target_radius)      params.target_radius = cli.target_radius;
    if (ov.subdomain_padding)  params.subdomain_padding = cli.subdomain_padding;
    if (ov.save_interval)      params.save_interval = cli.save_interval;
    if (ov.print_interval)     params.print_interval = cli.print_interval;
    if (ov.trajectory_samples) params.trajectory_samples = cli.trajectory_samples;
    if (ov.seed)               params.seed = cli.seed;
    if (ov.polarity_seed)      params.polarity_seed = cli.polarity_seed;
    if (ov.abp)                params.abp = cli.abp;

    // Legacy checkpoints (or fresh CLI without --seed) may leave seed=0.
    // Promote to a non-deterministic value so resumes never silently
    // collide on the historical seed=42 fallback.
    if (params.seed == 0) {
        std::random_device rd;
        unsigned s = rd();
        if (s == 0) s = 0xC0FFEEu;
        params.seed = s;
    }
    if (params.polarity_seed == 0) {
        std::random_device rd;
        unsigned s = rd();
        if (s == 0) s = 0xDECAFBADu;
        params.polarity_seed = s;
    }
    printf("[SIM] resume seed=%u polarity_seed=%u\n",
           params.seed, params.polarity_seed);

    if (params.trajectory_samples > 0 && params.t_end > 0 && params.dt > 0) {
        long long total_steps = (long long)(params.t_end / params.dt + 0.5);
        traj_every = (int)((total_steps + params.trajectory_samples - 1)
                            / params.trajectory_samples);
        if (traj_every < 1) traj_every = 1;
    }

    // ---- Build host cell vector from payload ----
    h_cells.resize(nc);
    for (int i = 0; i < nc; ++i) {
        h_cells[i] = {payload.cx[i], payload.cy[i],
                      params.target_radius, params.gamma, params.v_A,
                      payload.ox[i], payload.oy[i]};
    }

    if (params.Nx < TILE_T || params.Ny < TILE_T) {
        fprintf(stderr,
                "[FATAL] checkpoint domain (%d x %d) smaller than TILE_T=%d.\n",
                params.Nx, params.Ny, TILE_T);
        std::exit(1);
    }

    // ---- Multi-GPU slab geometry ----
    // alloc_gpu() needs the slab geometry set before it allocates S, so
    // build the layout here. Three resume cases:
    //  (a) gpus == 1: trivial single-GPU.
    //  (b) gpus > 1 with a v8 multi-rank checkpoint matching gpus: each
    //      rank's file already contains only its own slab cells.
    //  (c) gpus > 1 with v7 or v8-single-rank: not supported by the C++
    //      loader; use cell_analyze to consolidate or split first.
    build_layout();
    if (gpus > 1) {
        if (payload.version == 8 && payload.v8_num_ranks == gpus) {
            cells_global = payload.v8_n_global;
            h_global_id  = payload.gid;
        } else {
            std::string layout_desc = (payload.version == 8)
                ? (std::to_string(payload.v8_num_ranks) + "-rank")
                : "single-rank";
            fprintf(stderr,
                "[ckpt] cannot resume a v%u %s checkpoint with --gpus %d. "
                "Save with the same --gpus, or use `cell_analyze merge-ckpt`/"
                "`split-ckpt` to consolidate or repartition the per-rank "
                "files first.\n",
                payload.version, layout_desc.c_str(), gpus);
            return false;
        }
    } else {
        cells_global = nc;
        h_global_id  = payload.gid;
    }

    alloc_gpu();
    const int n = cells.num_cells;

    // ---- Resolve per-cell scalars: CLI > sidecar > params ----
    if (!gamma_spec.empty()) {
        apply_gamma_spec();           // writes into h_cells[i].gamma
        payload.per_gamma.clear();
    }
    if (v_A_sigma > 0.0) payload.per_vA.clear();
    // Guard the "all-zero sidecar" case: equilibrations with v_A=0 still
    // emit a VA_A sidecar (all zeros). Resuming with --v-A>0 must use the
    // CLI value, not the zero sidecar.
    if (ov.v_A && !payload.per_vA.empty() && cli.v_A > 0.0) {
        double sum = 0.0;
        for (float v : payload.per_vA) sum += std::fabs(v);
        if (sum <= 1e-12) payload.per_vA.clear();
    }

    // Polarity RNG for legacy resumes that lack a POLR block.
    srand(params.seed ? params.seed : 42);

    std::vector<int>   h_origin(2 * n);
    std::vector<float> h_g(n), h_vA(n), h_tr(n);
    std::vector<float> h_th(n), h_px(n), h_py(n);
    for (int i = 0; i < n; ++i) {
        auto& c = h_cells[i];
        double g  = (i < (int)payload.per_gamma.size())  ? (double)payload.per_gamma[i]  : c.gamma;
        double R  = (i < (int)payload.per_radius.size()) ? (double)payload.per_radius[i] : c.radius;
        double vA = (i < (int)payload.per_vA.size())     ? (double)payload.per_vA[i]     : c.v_A;
        c.gamma = g; c.radius = R; c.v_A = vA;

        h_origin[2*i + 0] = payload.ox[i];
        h_origin[2*i + 1] = payload.oy[i];
        h_g [i] = (float)g;
        h_vA[i] = (float)vA;
        h_tr[i] = (float)R;

        float theta;
        if (i < (int)payload.per_polar_theta.size()) {
            theta = payload.per_polar_theta[i];
        } else {
            theta = (float)(rand() % 10000) / 10000.0f * 2.0f * (float)M_PI;
        }
        h_th[i] = theta;
        h_px[i] = cosf(theta);
        h_py[i] = sinf(theta);
    }

    CK(cudaMemcpy(cells.origin,      h_origin.data(), 2*n*sizeof(int),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.gamma_cell,  h_g.data(),      n*sizeof(float),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.v_A_cell,    h_vA.data(),     n*sizeof(float),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.tgt_radius,  h_tr.data(),     n*sizeof(float),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_theta, h_th.data(),     n*sizeof(float),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_x,     h_px.data(),     n*sizeof(float),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_y,     h_py.data(),     n*sizeof(float),    cudaMemcpyHostToDevice));

    // Initial rect on resume: full minus 1px halo. First rebind shrinks it.
    {
        std::vector<int> h_rect(4 * n);
        for (int i = 0; i < n; ++i) {
            h_rect[4*i + 0] = 1;
            h_rect[4*i + 1] = 1;
            h_rect[4*i + 2] = TILE_T - 2;
            h_rect[4*i + 3] = TILE_T - 2;
        }
        CK(cudaMemcpy(cells.rect, h_rect.data(), 4*n*sizeof(int), cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < n; ++i) {
        CK(cudaMemcpy(cells.phi_in + (size_t)i * TILE_AREA,
                      payload.phi_tiles[i].data(), TILE_AREA * sizeof(float),
                      cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(cells.velocities_x, payload.vx.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.velocities_y, payload.vy.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    // Restore per-cell curandState from the RNGS sidecar (when present and
    // sized as expected). A user-supplied --polarity-seed on resume defeats
    // the restore so they get a fresh independent stream. Without restore,
    // finalize_init() re-seeds and the resumed run replays correlated RNG.
    if (!ov.polarity_seed &&
        !payload.per_rng_bytes.empty() &&
        payload.per_rng_bytes.size() == (size_t)n * sizeof(curandState)) {
        CK(cudaMemcpy(cells.rng_states, payload.per_rng_bytes.data(),
                      payload.per_rng_bytes.size(), cudaMemcpyHostToDevice));
        rng_restored_from_ckpt = true;
        printf("[SIM] restored RNG state for %d cells from checkpoint\n", n);
    } else if (!payload.per_rng_bytes.empty()) {
        fprintf(stderr,
                "[SIM] WARNING: RNGS sidecar present (%zu bytes) but ignored "
                "(--polarity-seed override or size mismatch; expected %zu).\n",
                payload.per_rng_bytes.size(), (size_t)n * sizeof(curandState));
    }

    finalize_init();

    if (gpus > 1) {
        printf("[multi-gpu] rank %d/%d resumed from per-rank checkpoint: "
               "slab y in [%d,%d), %d/%d cells\n",
               rank, gpus, slab_y_lo(), slab_y_hi(), n, cells_global);
    }
    printf("[SIM] resumed from %s: step=%d, t=%.4f, %d cells, %dx%d, t_end=%.1f\n",
           path.c_str(), step_count, cur_time, n,
           params.Nx, params.Ny, params.t_end);
    return true;
}

// ---------------------------------------------------------------------------
// slice_cells_to_local — multi-GPU helper.
//
// After place_cells + apply_gamma_spec + apply_v_A_disorder have produced
// the GLOBAL cell vector (h_cells.size() == cells_global), partition the
// vector by spatial Y position: rank g keeps cells whose COM cy lies in
// [slab_y_lo[g], slab_y_hi[g]). Slab bounds are stored on the Simulation
// for later use (alloc_gpu, halo exchange, migration). h_global_id is
// populated so we can recover the original cell ids for trajectory I/O.
// For gpus == 1 this is a no-op (slab covers the full grid).
// ---------------------------------------------------------------------------
void Simulation::slice_cells_to_local() {
    const int n_global = (int)h_cells.size();
    cells_global = n_global;
    build_layout();
    if (gpus <= 1) {
        cell_offset = 0;
        h_global_id.resize(n_global);
        for (int i = 0; i < n_global; ++i) h_global_id[i] = i;
        return;
    }
    const int y_lo = layout.slab_y_lo;
    const int y_hi = layout.slab_y_hi;
    std::vector<CellHost> kept;
    kept.reserve(n_global / gpus + 16);
    h_global_id.clear();
    h_global_id.reserve(n_global / gpus + 16);
    for (int i = 0; i < n_global; ++i) {
        // Wrap cy into [0, Ny) for partition decision.
        double cy = h_cells[i].cy;
        cy = std::fmod(cy, (double)params.Ny);
        if (cy < 0) cy += params.Ny;
        int cy_int = (int)std::floor(cy);
        if (cy_int >= y_lo && cy_int < y_hi) {
            kept.push_back(h_cells[i]);
            h_global_id.push_back(i);
        }
    }
    cell_offset = -1;  // not meaningful for spatial partition
    h_cells.swap(kept);
    fprintf(stderr,
        "[multi-gpu] rank %d/%d: spatial slab y in [%d,%d), %d/%d cells\n",
        rank, gpus, y_lo, y_hi, (int)h_cells.size(), n_global);
}

// ---------------------------------------------------------------------------
// load_scripted_events — owns the invariant that scripted_active is true
// iff the host/device event arrays are populated and sorted by step_count.
// Replaces the inline parser in main.cu which had to poke 7 public fields
// in the right order; getting any of them out of sync (e.g. forgetting
// cudaMalloc) was a silent-failure bug.
//
// File format (cpu_ref --events): `# t cid [old_theta] new_theta` per line.
// `t` is converted to step_count using sim.cur_time + sim.params.dt; events
// with t <= cur_time are rejected.
// ---------------------------------------------------------------------------
bool Simulation::load_scripted_events(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "Error: cannot open --scripted-events file '%s'\n",
                path.c_str());
        return false;
    }
    const int N = cells.num_cells;
    const double dt = params.dt;
    const double start_t = cur_time;
    const int    start_step = step_count;
    struct Evt { int step_at_start; int cid; float theta; };
    std::vector<Evt> evs;
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        size_t p0 = line.find_first_not_of(" \t\r\n");
        if (p0 == std::string::npos) continue;
        if (line[p0] == '#') continue;
        std::istringstream is(line);
        std::vector<double> toks;
        double v;
        while (is >> v) toks.push_back(v);
        double t, new_theta; int cid;
        if (toks.size() == 3) {
            t = toks[0]; cid = (int)toks[1]; new_theta = toks[2];
        } else if (toks.size() == 4) {
            t = toks[0]; cid = (int)toks[1]; new_theta = toks[3];
        } else {
            fprintf(stderr,
                "Error: %s line %d: expected 3 or 4 cols, got %zu\n",
                path.c_str(), lineno, toks.size());
            return false;
        }
        if (cid < 0 || cid >= N) {
            fprintf(stderr,
                "Error: %s line %d: cid %d out of range (n_cells=%d)\n",
                path.c_str(), lineno, cid, N);
            return false;
        }
        if (t <= start_t) {
            fprintf(stderr,
                "Error: %s line %d: t=%.6f <= start_t=%.6f\n",
                path.c_str(), lineno, t, start_t);
            return false;
        }
        int step_idx = (int)std::llround((t - start_t) / dt);
        int step_at_start = start_step + step_idx - 1;
        evs.push_back({step_at_start, cid, (float)new_theta});
    }
    std::sort(evs.begin(), evs.end(), [](const Evt& a, const Evt& b) {
        if (a.step_at_start != b.step_at_start) return a.step_at_start < b.step_at_start;
        return a.cid < b.cid;
    });
    scripted_active = !evs.empty();
    scripted_cursor = 0;
    h_scripted_step.clear();
    h_scripted_cid.clear();
    h_scripted_theta.clear();
    h_scripted_step.reserve(evs.size());
    h_scripted_cid.reserve(evs.size());
    h_scripted_theta.reserve(evs.size());
    for (const auto& e : evs) {
        h_scripted_step.push_back(e.step_at_start);
        h_scripted_cid.push_back(e.cid);
        h_scripted_theta.push_back(e.theta);
    }
    if (!evs.empty()) {
        CK(cudaMalloc(&d_scripted_cid,   evs.size() * sizeof(int)));
        CK(cudaMalloc(&d_scripted_theta, evs.size() * sizeof(float)));
        CK(cudaMemcpy(d_scripted_cid,   h_scripted_cid.data(),
                      evs.size() * sizeof(int),   cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_scripted_theta, h_scripted_theta.data(),
                      evs.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    printf("[scripted] %zu events loaded from %s (PRNG tumble path disabled)\n",
           evs.size(), path.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// step — one integration step.
// ---------------------------------------------------------------------------
// Hot path (the vast majority of steps): polar + scatter + fast-reduce + RHS,
// captured into a CUDA Graph (one per pool-parity) and replayed each step.
// Replay is a single host->driver call vs. ~5 launches, saving ~3-5 us/step
// of API overhead. Slow path (rebind, output, scripted, first encounter):
// direct launches on the same step_stream.
// ---------------------------------------------------------------------------
Simulation::StepFlags Simulation::compute_step_flags(int next_step) const {
    StepFlags f{};
    f.will_rebind = (next_step % REBIND_EVERY) == 0;
    f.will_traj   = (traj_fp && traj_every > 0 && next_step % traj_every == 0);
    f.will_save   = (params.save_interval > 0 && next_step % params.save_interval == 0);
    f.will_ckpt   = (checkpoint_interval  > 0 && next_step % checkpoint_interval == 0);
    f.will_vtk    = (vtk_interval > 0 && next_step % vtk_interval == 0);
    f.need_full_red = f.will_rebind || f.will_traj || f.will_save || f.will_ckpt || f.will_vtk;
    return f;
}

void Simulation::apply_scripted_events_for_step() {
    if (!scripted_active) return;
    int begin = scripted_cursor;
    const int total = (int)h_scripted_step.size();
    while (scripted_cursor < total
           && h_scripted_step[scripted_cursor] == step_count) {
        scripted_cursor++;
    }
    int count = scripted_cursor - begin;
    if (count > 0) {
        launch_apply_scripted(cells,
                              d_scripted_cid + begin,
                              d_scripted_theta + begin,
                              count, step_stream);
    }
}

void Simulation::step() {
    int next_step = step_count + 1;
    const StepFlags f = compute_step_flags(next_step);
    const bool fast_path = !scripted_active && !f.will_rebind && !f.need_full_red
                           && (params.v_A != 0.0) && (params.tau > 0.0);

    // Keep cells.phi_in / phi_out in sync with parity, so any direct kernel
    // launch (output, rebind path) reads the right buffer.
    sync_pool_to_parity();

    if (fast_path) {
        if (!step_graph_built[parity]) {
            // Capture once per parity. cudaStreamCaptureModeThreadLocal so
            // unrelated CUDA calls on the host (e.g. cudaMalloc) don't
            // accidentally get pulled into the capture.
            cudaGraph_t graph = nullptr;
            CK(cudaStreamBeginCapture(step_stream,
                                      cudaStreamCaptureModeThreadLocal));
            launch_polar(cells, params, step_stream);
            launch_scatter_S(cells, params, step_stream);
            launch_evolve(cells, params, /*need_full_reduce=*/false, step_stream);
            CK(cudaStreamEndCapture(step_stream, &graph));
            CK(cudaGraphInstantiate(&step_graph[parity], graph, nullptr, nullptr, 0));
            cudaGraphDestroy(graph);
            step_graph_built[parity] = true;
        }
        CK(cudaGraphLaunch(step_graph[parity], step_stream));
        flip_parity();
    } else {
        // Slow path: direct launches on step_stream so ordering matches the
        // graph path (no cross-stream sync needed).
        if (scripted_active) {
            apply_scripted_events_for_step();
        } else {
            launch_polar(cells, params, step_stream);
        }
        launch_scatter_S(cells, params, step_stream);
        launch_evolve(cells, params, f.need_full_red, step_stream);
        flip_parity();

        if (f.will_rebind) {
            launch_rebind(cells,
                          (float)params.subdomain_padding,
                          (float)params.gamma, step_stream);
            flip_parity();
        }
    }

#ifndef NDEBUG
    // Per-step launch error check: useful in Debug builds, off in Release
    // (NDEBUG is set automatically by CMake for Release). Saves one host
    // API call per step on the hot path.
    {
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA] step %d (peek): %s\n", step_count, cudaGetErrorString(err));
            fflush(stderr);
            cudaError_t serr = cudaDeviceSynchronize();
            if (serr != cudaSuccess) {
                fprintf(stderr, "[CUDA] step %d (sync): %s\n", step_count, cudaGetErrorString(serr));
                fflush(stderr);
            }
            exit(1);
        }
    }
#endif
    step_count++;
    cur_time += params.dt;
}

// ---------------------------------------------------------------------------
// step_pre_reduce / step_post_reduce — multi-GPU step decomposition.
// ---------------------------------------------------------------------------
// The orchestrator drives one step like this, for each rank g in lockstep:
//
//   for g: sim[g].step_pre_reduce();          // polar + scatter to LOCAL S
//   ncclGroupStart();
//   for g: ncclAllReduce(S, sum, comm[g], stream[g]);
//   ncclGroupEnd();
//   for g: sim[g].step_post_reduce();         // evolve + maybe rebind
//
// Graph capture is NOT used on this path — capturing across an external
// NCCL collective is fragile, and the per-step graph savings (~3-5us) are
// dwarfed by the all-reduce. The graph fast path remains in step() above
// and is what --gpus 1 always uses.
//
// step_pre_reduce / step_post_reduce together advance step_count and
// cur_time by exactly one step (post_reduce does the increment). It is
// the orchestrator's responsibility to issue them in pairs.
// ---------------------------------------------------------------------------
void Simulation::step_pre_reduce() {
    sync_pool_to_parity();

    if (scripted_active) {
        apply_scripted_events_for_step();
    } else {
        launch_polar(cells, params, step_stream);
    }
    launch_scatter_S(cells, params, step_stream);
}

void Simulation::step_post_reduce() {
    int next_step = step_count + 1;
    const StepFlags f = compute_step_flags(next_step);

    launch_evolve(cells, params, f.need_full_red, step_stream);
    flip_parity();

    if (f.will_rebind) {
        launch_rebind(cells,
                      (float)params.subdomain_padding,
                      (float)params.gamma, step_stream);
        flip_parity();
    }

#ifndef NDEBUG
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] step %d (peek): %s\n", step_count,
                cudaGetErrorString(err));
        fflush(stderr);
        cudaError_t serr = cudaDeviceSynchronize();
        if (serr != cudaSuccess) {
            fprintf(stderr, "[CUDA] step %d (sync): %s\n", step_count,
                    cudaGetErrorString(serr));
            fflush(stderr);
        }
        exit(1);
    }
#endif
    step_count++;
    cur_time += params.dt;
}

// ---------------------------------------------------------------------------
// Multi-GPU step pieces, factored out so the orchestrator can issue them
// directly or capture them into a CUDA graph.
// ---------------------------------------------------------------------------
void Simulation::launch_halo_exchange(MgWorld& world, int my_rank,
                                      int prev_rank, int next_rank,
                                      float* my_top_band, float* my_bot_band,
                                      float* halo_top_recv, float* halo_bot_recv,
                                      size_t halo_band_floats) {
    if (gpus < 2) return;
    cudaStream_t s = step_stream;
    if (gpus == 2) {
        // For G=2, prev == next; NCCL pair-by-order requires a 4-issue
        // ordering (Send_top, Recv_top, Send_bot, Recv_bot) so the two
        // boundaries are disambiguated. See orchestrator comments for the
        // full derivation. After the exchange, halo_top_recv holds peer's
        // top band — which is OUR bot boundary — so it adds into my_bot.
        mg_group_start();
        mg_send_bytes(world.comms[my_rank], my_top_band, prev_rank,
                      halo_band_floats * sizeof(float), s);
        mg_recv_bytes(world.comms[my_rank], halo_top_recv, prev_rank,
                      halo_band_floats * sizeof(float), s);
        mg_send_bytes(world.comms[my_rank], my_bot_band, next_rank,
                      halo_band_floats * sizeof(float), s);
        mg_recv_bytes(world.comms[my_rank], halo_bot_recv, next_rank,
                      halo_band_floats * sizeof(float), s);
        mg_group_end();
        launch_halo_add_pair(my_bot_band, halo_top_recv,
                             my_top_band, halo_bot_recv,
                             halo_band_floats, s);
    } else {
        // G >= 3: standard pair-by-peer routing.
        mg_group_start();
        mg_send_recv_f32(world.comms[my_rank],
                         my_top_band, prev_rank,
                         halo_top_recv, prev_rank,
                         halo_band_floats, s);
        mg_send_recv_f32(world.comms[my_rank],
                         my_bot_band, next_rank,
                         halo_bot_recv, next_rank,
                         halo_band_floats, s);
        mg_group_end();
        launch_halo_add_pair(my_top_band, halo_top_recv,
                             my_bot_band, halo_bot_recv,
                             halo_band_floats, s);
    }
}

bool Simulation::mg_step_is_fast_path() const {
    if (scripted_active) return false;
    int next_step = step_count + 1;
    if ((next_step % REBIND_EVERY) == 0) return false;
    if (traj_fp && traj_every > 0 && (next_step % traj_every) == 0) return false;
    if (params.save_interval > 0 && (next_step % params.save_interval) == 0) return false;
    if (checkpoint_interval  > 0 && (next_step % checkpoint_interval)  == 0) return false;
    if (vtk_interval         > 0 && (next_step % vtk_interval)         == 0) return false;
    return true;
}

void Simulation::invalidate_mg_step_graph() {
    for (int i = 0; i < 2; ++i) {
        if (mg_step_graph[i]) {
            cudaGraphExecDestroy(mg_step_graph[i]);
            mg_step_graph[i] = nullptr;
        }
        mg_step_graph_built[i] = false;
    }
}

void Simulation::launch_mg_fast_step_kernels(MgWorld& world, int my_rank,
                                             int prev_rank, int next_rank,
                                             float* my_top_band, float* my_bot_band,
                                             float* halo_top_recv, float* halo_bot_recv,
                                             size_t halo_band_floats) {
    // Fast path only: no scripted events (polar runs normally), no rebind,
    // no full reduction. Pointer state is owned by the parity invariant
    // (sync_pool_to_parity has already been called by the caller before
    // we get here — graph capture freezes the pointers seen at this point).
    launch_polar(cells, params, step_stream);
    launch_scatter_S(cells, params, step_stream);
    if (gpus > 1) {
        launch_halo_exchange(world, my_rank, prev_rank, next_rank,
                             my_top_band, my_bot_band,
                             halo_top_recv, halo_bot_recv,
                             halo_band_floats);
    }
    launch_evolve(cells, params, /*need_full_reduce=*/false, step_stream);
    // NOTE: flip_parity is done by the caller, *outside* the captured
    // region. That keeps host-side parity state consistent across graph
    // launches and avoids re-capturing the graph on every parity flip
    // (we already key the graph cache on parity).
}

// ---------------------------------------------------------------------------
// run loop
// ---------------------------------------------------------------------------
#ifdef ENABLE_VISUALIZER
#include "visualizer.cuh"
#endif

void Simulation::run() {
    int target_step = std::max(step_count, (int)(params.t_end / params.dt));
    int total = target_step - step_count;
    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_VISUALIZER
    cellsim::Visualizer viz;
    bool viz_active = false;
    if (live_view) {
        viz_active = viz.init(params.Nx, params.Ny);
        if (!viz_active) {
            fprintf(stderr, "[viz] init failed; continuing headless\n");
        }
    }
#endif

    if (traj_every > 0) {
        std::string tp = out_dir + "/trajectory.txt";
        traj_fp = fopen(tp.c_str(), "a");
        if (traj_fp) {
            setvbuf(traj_fp, nullptr, _IOFBF, 1 << 20);
            fseek(traj_fp, 0, SEEK_END);
            long pos = ftell(traj_fp);
            if (pos == 0) {
                fprintf(traj_fp, "# Trajectory data\n");
                fprintf(traj_fp, "# Format: time cell_id x y vx vy px py theta v_A_i L_n volume\n");
                fprintf(traj_fp, "# v_A=%.6f N=%d Lx=%d Ly=%d dim=2 dt=%.6f tau=%.4f\n",
                        params.v_A, cells.num_cells, params.Nx, params.Ny,
                        params.dt, params.tau);
            }
        }
    }

    while (step_count < target_step) {
        step();
        bool wrote_output = false;
        if (params.save_interval > 0 && step_count % params.save_interval == 0) {
            char tag[32]; snprintf(tag, sizeof(tag), "%08d", step_count);
            save_checkpoint(out_dir, tag);
            wrote_output = true;
        }
        if (checkpoint_interval > 0 && step_count % checkpoint_interval == 0) {
            save_checkpoint(out_dir);
            wrote_output = true;
        }
        if (traj_fp && traj_every > 0 && step_count % traj_every == 0) {
            write_trajectory();
            wrote_output = true;
        }
        if (vtk_interval > 0 && step_count % vtk_interval == 0) {
            write_vtk();
            wrote_output = true;
        }
        // Status line: print on cadence regardless of whether trajectory/VTK
        // was written this step. The previous `wrote_output &&` gate caused
        // long compute phases (e.g. equilibration with --trajectory-samples 0)
        // to appear silent, which trips agent-terminal idle-kill (Ctrl+C
        // after ~30-60 s of no stdout). --print-interval 0 still disables.
        if (params.print_interval > 0
            && step_count % params.print_interval == 0)
            print_status();

#ifdef ENABLE_VISUALIZER
        if (viz_active && (step_count % live_view_interval == 0)) {
            if (viz.should_close()) {
                viz.shutdown();
                viz_active = false;
                // When the window is closed, exit the sim instead of
                // silently continuing headless. Live-view runs are
                // exploratory; the user closing the window means "stop".
                printf("[viz] window closed; stopping sim\n");
                break;
            } else {
                // Viz reads sim buffers; sim runs on step_stream so we must
                // wait for the current step to finish before sampling. Without
                // this the colormap shows half-applied scatter/evolve state
                // and looks like flashing/tearing.
                cudaStreamSynchronize(step_stream);
                viz.update(cells.S, cells.phi_in, cells.origin, cells.rect,
                           cells.Cx, cells.Cy, cells.Cxx, cells.Cyy,
                           cells.volumes,
                           cells.velocities_x, cells.velocities_y,
                           cells.gamma_cell, cells.tgt_radius,
                           cells.num_cells, params.Nx, params.Ny, cur_time,
                           (float)params.subdomain_padding,
                           (float)params.lambda);
            }
        }
#endif
    }
    if (traj_fp) {
        finish_trajectory_writer();
        fclose(traj_fp);
        traj_fp = nullptr;
    }

    if (save_final_checkpoint) save_checkpoint(out_dir);

    CK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();
    int denom = std::max(1, total);
    printf("[SIM] Done: %d steps, t=%.2f, wall=%.3fs (%.3f ms/step)\n",
           total, cur_time, wall, wall * 1000.0 / denom);
#ifdef CELL_SIM_BBOX_TELEMETRY
    extern __device__ int g_bbox_max_raw_hw;
    extern __device__ int g_bbox_clamp_events;
    int max_raw = 0, clamps = 0;
    cudaMemcpyFromSymbol(&max_raw, g_bbox_max_raw_hw, sizeof(int));
    cudaMemcpyFromSymbol(&clamps,  g_bbox_clamp_events, sizeof(int));
    int ceiling = (TILE_T >> 1) - 1;
    printf("[SIM] bbox-telemetry: lifetime max_raw_hw=%d (ceiling=%d, %.0f%% margin) total_clamp_events=%d\n",
           max_raw, ceiling, 100.0 * (1.0 - (double)max_raw / ceiling), clamps);
#endif
}

// ---------------------------------------------------------------------------
// print_status — average volume vs target.
// ---------------------------------------------------------------------------
void Simulation::print_status() {
    if (step_stream) CK(cudaStreamSynchronize(step_stream));
    int n = cells.num_cells;
    std::vector<float> vols(n);
    CK(cudaMemcpy(vols.data(), cells.volumes, n * sizeof(float), cudaMemcpyDeviceToHost));
    double avg = 0; for (float v : vols) avg += v; avg /= n;
    double tgt = params.target_area();
    // Free NaN tripwire: avg already computed; if it's not finite,
    // physics has gone off the rails (typical cause: a kernel produced
    // Inf/NaN values that propagated through the reduction). Abort
    // immediately so we don't burn cluster hours producing garbage
    // trajectories. Exit code 2 distinguishes from normal failures.
    if (!std::isfinite(avg)) {
        fprintf(stderr,
                "[FATAL] non-finite avg_vol=%g at step=%d t=%.4f — "
                "aborting before more garbage data is written\n",
                avg, step_count, cur_time);
        std::exit(2);
    }
    printf("step=%d t=%.2f avg_vol=%.1f (target=%.1f, err=%.2f%%)\n",
           step_count, cur_time, avg, tgt, 100.0 * (avg - tgt) / tgt);
#ifdef CELL_SIM_BBOX_TELEMETRY
    // Read the device counters set in k_rebind. Off-default; see comment
    // at the top of kernels.cu for why this should not be enabled in
    // general production runs.
    extern __device__ int g_bbox_max_raw_hw;
    extern __device__ int g_bbox_clamp_events;
    int max_raw = 0, clamps = 0;
    cudaMemcpyFromSymbol(&max_raw, g_bbox_max_raw_hw, sizeof(int));
    cudaMemcpyFromSymbol(&clamps,  g_bbox_clamp_events, sizeof(int));
    int ceiling = (TILE_T >> 1) - 1;
    if (clamps > 0 || max_raw >= (int)(0.9f * ceiling)) {
        printf("  [bbox] max_raw_hw=%d (ceiling=%d, %.0f%% margin) clamp_events=%d\n",
               max_raw, ceiling, 100.0 * (1.0 - (double)max_raw / ceiling), clamps);
    }
#endif
}

// ---------------------------------------------------------------------------
// write_trajectory — stable CSV format consumed by the Rust analyzer
// and Python tooling. Centroids are computed on the host from packed
// per-cell fields (origin, Cx, Cy, V) that are gathered on-device by
// k_pack_traj and shipped via one cudaMemcpyAsync to a pinned host ring.
// The producer (step thread) never calls cudaStreamSynchronize; the
// writer thread waits on a per-snapshot cudaEvent.
// ---------------------------------------------------------------------------
void Simulation::init_trajectory_ring() {
    if (traj_ring_init) return;
    const int cap = (cells.capacity > 0) ? cells.capacity : cells.num_cells;
    if (cap <= 0) return;
    CK(cudaMalloc(&d_traj_pack, (size_t)cap * sizeof(TrajPackedCell)));
    // CELL_SIM_TRAJ_RING tunes how many in-flight snapshots are allowed.
    // Depth 4 is enough to hide ~4x the writer's host time; raising it
    // only matters if the formatter+disk get slower than the GPU step.
    const int depth = std::max(2, sim_env_int("CELL_SIM_TRAJ_RING", 4));
    traj_ring.resize(depth);
    for (auto& slot : traj_ring) {
        CK(cudaMallocHost((void**)&slot.host_packed,
                          (size_t)cap * sizeof(TrajPackedCell)));
        CK(cudaEventCreateWithFlags(&slot.ready_event, cudaEventDisableTiming));
        traj_writer_free.push_back(&slot);
    }
    traj_ring_capacity = cap;
    traj_ring_init = true;
}

void Simulation::free_trajectory_ring() {
    if (!traj_ring_init) return;
    for (auto& slot : traj_ring) {
        if (slot.host_packed) { cudaFreeHost(slot.host_packed); slot.host_packed = nullptr; }
        if (slot.ready_event) { cudaEventDestroy(slot.ready_event); slot.ready_event = nullptr; }
    }
    traj_ring.clear();
    traj_writer_free.clear();
    if (d_traj_pack) { cudaFree(d_traj_pack); d_traj_pack = nullptr; }
    traj_ring_capacity = 0;
    traj_ring_init = false;
}

void Simulation::start_trajectory_writer() {
    if (traj_writer_started || !traj_fp) return;
    init_trajectory_ring();
    traj_writer_stop = false;
    traj_writer_thread = std::thread(&Simulation::trajectory_writer_loop, this);
    traj_writer_started = true;
}

void Simulation::finish_trajectory_writer() {
    if (!traj_writer_started) return;
    {
        std::lock_guard<std::mutex> lock(traj_writer_mutex);
        traj_writer_stop = true;
    }
    traj_writer_cv.notify_all();
    if (traj_writer_thread.joinable()) traj_writer_thread.join();
    traj_writer_started = false;
    traj_writer_stop = false;
    traj_writer_queue.clear();
    traj_writer_free.clear();
    if (traj_fp) fflush(traj_fp);
    free_trajectory_ring();
}

void Simulation::trajectory_writer_loop() {
    for (;;) {
        TrajectorySnapshot* slot = nullptr;
        {
            std::unique_lock<std::mutex> lock(traj_writer_mutex);
            traj_writer_cv.wait(lock, [&] {
                return traj_writer_stop || !traj_writer_queue.empty();
            });
            if (traj_writer_queue.empty()) {
                if (traj_writer_stop) break;
                continue;
            }
            slot = traj_writer_queue.front();
            traj_writer_queue.pop_front();
        }
        // Block here, not on the producer: wait for the async D2H copy to
        // land in the pinned host buffer before formatting.
        CK(cudaEventSynchronize(slot->ready_event));
        write_trajectory_snapshot(*slot);
        {
            std::lock_guard<std::mutex> lock(traj_writer_mutex);
            traj_writer_free.push_back(slot);
        }
        traj_writer_cv.notify_all();
    }
}

void Simulation::write_trajectory_snapshot(const TrajectorySnapshot& snap) {
    if (!traj_fp) return;
    const int n = snap.n;
    const int Nx = params.Nx, Ny = params.Ny;
    const double dA = params.dA();
    const double tgt_r = params.target_radius;

    auto wrap_d = [](double v, int L) {
        double m = std::fmod(v, (double)L);
        if (m < 0) m += L;
        return m;
    };

    int skipped_v0 = 0;
    for (int i = 0; i < n; i++) {
        const TrajPackedCell& c = snap.host_packed[i];
        // Skip cells whose volume hasn't been refreshed yet — happens for
        // arrival slots in the rebind step that triggered a migration: the
        // tile is copied over but volumes/Cx/Cy aren't refilled until the
        // next reduce. Without this guard, Cx/V=0 emits the bare origin
        // and produces a fake L/2-sized cell jump for one frame.
        if (c.V < 1e-6f) { ++skipped_v0; continue; }
        double invV = 1.0 / c.V;
        double cx_g = wrap_d(c.ox + c.Cx * invV, Nx);
        double cy_g = wrap_d(c.oy + c.Cy * invV, Ny);
        // Free NaN tripwire: cx_g/cy_g already computed on host. If
        // either is non-finite, the cell's reductions are corrupted
        // (typical: NaN propagating from a previous step's RHS). Abort
        // before we write a garbage trajectory row.
        if (!std::isfinite(cx_g) || !std::isfinite(cy_g)) {
            fprintf(stderr,
                    "[FATAL] non-finite centroid for cell %d at step=%d "
                    "t=%.4f: cx=%g cy=%g — aborting\n",
                    i, snap.step, snap.time, cx_g, cy_g);
            fflush(traj_fp);
            std::exit(2);
        }
        float theta = atan2f(c.py, c.px);
        double perim = c.per * dA;
        double Ln = perim / (2.0 * M_PI * tgt_r);
        double vol = c.V * dA;
        int gid = ((int)snap.global_id.size() > i) ? snap.global_id[i] : i;
        fprintf(traj_fp,
                "%.6f %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                snap.time, gid, cx_g, cy_g, c.vx, c.vy,
                c.px, c.py, theta, c.vA, Ln, vol);
    }
    ++traj_flush_counter;
    const int flush_every = sim_env_int("CELL_SIM_TRAJ_FLUSH_EVERY", 100);
    if (flush_every <= 1 || (traj_flush_counter % flush_every) == 0) {
        fflush(traj_fp);
    }
    if (skipped_v0 > 0) {
        fprintf(stderr,
            "[traj] rank %d t=%.4f: skipped %d cell(s) with V=0 "
            "(post-migration pre-reduce); next sample will include them\n",
            rank, snap.time, skipped_v0);
    }
}

void Simulation::write_trajectory() {
    if (!traj_fp) return;
    start_trajectory_writer();
    const int n = cells.num_cells;
    if (n > traj_ring_capacity) {
        // Should never happen: ring was sized to cells.capacity which
        // includes the multi-GPU migration headroom. If it does, the
        // contract is broken — abort rather than silently truncate.
        fprintf(stderr,
                "[FATAL] write_trajectory: num_cells %d > ring capacity %d\n",
                n, traj_ring_capacity);
        std::exit(2);
    }

    // Acquire a free slot. Blocks only if the writer is genuinely behind
    // (ring full of in-flight snapshots) — with depth 4 and the writer
    // doing ~50us/snapshot vs ~500us per step, never in steady state.
    TrajectorySnapshot* slot = nullptr;
    {
        std::unique_lock<std::mutex> lock(traj_writer_mutex);
        traj_writer_cv.wait(lock, [&] {
            return traj_writer_stop || !traj_writer_free.empty();
        });
        if (traj_writer_stop) return;
        slot = traj_writer_free.front();
        traj_writer_free.pop_front();
    }

    slot->time = cur_time;
    slot->step = step_count;
    slot->n = n;
    if (gpus > 1 && (int)h_global_id.size() >= n) {
        slot->global_id.assign(h_global_id.begin(), h_global_id.begin() + n);
    } else {
        slot->global_id.clear();
    }

    // Pack on device, single async D2H, then mark ready. All three live on
    // step_stream so they implicitly serialise after the step's evolve.
    launch_pack_traj(cells, d_traj_pack, n, step_stream);
    CK(cudaMemcpyAsync(slot->host_packed, d_traj_pack,
                       (size_t)n * sizeof(TrajPackedCell),
                       cudaMemcpyDeviceToHost, step_stream));
    CK(cudaEventRecord(slot->ready_event, step_stream));

    {
        std::lock_guard<std::mutex> lock(traj_writer_mutex);
        traj_writer_queue.push_back(slot);
    }
    traj_writer_cv.notify_one();
}

static void composite_vtk_cells(Simulation& sim,
                                std::vector<float>& grid,
                                std::vector<int>& h_or,
                                std::vector<float>& tile) {
    const int n = sim.cells.num_cells;
    if (n <= 0) return;
    const int Nx = sim.params.Nx, Ny = sim.params.Ny;
    h_or.resize(2 * n);
    tile.resize(TILE_AREA);
    CK(cudaMemcpy(h_or.data(), sim.cells.origin, 2*n*sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        CK(cudaMemcpy(tile.data(),
                      sim.cells.phi_in + (size_t)i * TILE_AREA,
                      TILE_AREA * sizeof(float), cudaMemcpyDeviceToHost));
        int ox = h_or[2*i + 0];
        int oy = h_or[2*i + 1];
        for (int ly = 0; ly < TILE_T; ly++) {
            int gy = ((oy + ly) % Ny + Ny) % Ny;
            for (int lx = 0; lx < TILE_T; lx++) {
                int gx = ((ox + lx) % Nx + Nx) % Nx;
                float v = tile[(size_t)ly * TILE_T + lx];
                float& g = grid[(size_t)gy * Nx + gx];
                if (v > g) g = v;
            }
        }
    }
}

static void write_vtk_grid_file(const std::string& out_dir,
                                int step_count,
                                double cur_time,
                                int Nx, int Ny,
                                const std::vector<float>& grid) {
    auto swap_f32 = [](float fv) {
        uint32_t u; std::memcpy(&u, &fv, 4);
        u = ((u & 0x000000FFu) << 24) |
            ((u & 0x0000FF00u) << 8)  |
            ((u & 0x00FF0000u) >> 8)  |
            ((u & 0xFF000000u) >> 24);
        std::memcpy(&fv, &u, 4); return fv;
    };
    std::vector<float> be(grid.size());
    for (size_t k = 0; k < grid.size(); k++) be[k] = swap_f32(grid[k]);

    char fn[512];
    snprintf(fn, sizeof(fn), "%s/output_%06d.vtk", out_dir.c_str(), step_count);
    FILE* f = fopen(fn, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", fn); return; }
    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "cell_sim phase-field composite step=%d t=%.6f\n",
            step_count, cur_time);
    fprintf(f, "BINARY\n");
    fprintf(f, "DATASET STRUCTURED_POINTS\n");
    fprintf(f, "DIMENSIONS %d %d 1\n", Nx, Ny);
    fprintf(f, "ORIGIN 0 0 0\n");
    fprintf(f, "SPACING 1 1 1\n");
    fprintf(f, "POINT_DATA %lld\n", (long long)Nx * Ny);
    fprintf(f, "SCALARS phi float 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");
    fwrite(be.data(), sizeof(float), be.size(), f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// write_vtk — composite max(phi) field, big-endian binary VTK.
// ---------------------------------------------------------------------------
void Simulation::write_vtk() {
    CK(cudaDeviceSynchronize());
    const int Nx = params.Nx, Ny = params.Ny;
    std::vector<float> grid((size_t)Nx * Ny, 0.0f);
    std::vector<int> h_or;
    std::vector<float> tile;
    composite_vtk_cells(*this, grid, h_or, tile);
    write_vtk_grid_file(out_dir, step_count, cur_time, Nx, Ny, grid);
}

// ---------------------------------------------------------------------------
// save_checkpoint — writes the current v8 format.
//
// Schema and POD records live in include/checkpoint_format.cuh. Bump
// ckpt::VERSION_CURRENT there to start a new version and add a branch
// here + in init_from_checkpoint.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
void Simulation::save_checkpoint(const std::string& dir, const std::string& tag) {
    CK(cudaDeviceSynchronize());
    int n = cells.num_cells;

    std::vector<int>   h_or(2 * n);
    std::vector<float> V(n), Cx(n), Cy(n);
    std::vector<float> vx(n), vy(n);
    CK(cudaMemcpy(h_or.data(), cells.origin,       2*n*sizeof(int),    cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(V.data(),    cells.volumes,      n*sizeof(float),    cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Cx.data(),   cells.Cx,           n*sizeof(float),    cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Cy.data(),   cells.Cy,           n*sizeof(float),    cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vx.data(),   cells.velocities_x, n*sizeof(float),    cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vy.data(),   cells.velocities_y, n*sizeof(float),    cudaMemcpyDeviceToHost));

    char fn[512];
    if (tag.empty())
        snprintf(fn, sizeof(fn), "%s/checkpoint.bin", dir.c_str());
    else
        snprintf(fn, sizeof(fn), "%s/checkpoint_%s.bin", dir.c_str(), tag.c_str());
    // C3: atomic write. Write to <fn>.tmp and rename on close. Without this,
    // a SLURM SIGTERM-on-timeout mid-write truncates checkpoint.bin and
    // overwrites the previous good copy, losing the run.
    char fn_tmp[520];
    snprintf(fn_tmp, sizeof(fn_tmp), "%s.tmp", fn);
    FILE* f = fopen(fn_tmp, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", fn_tmp); return; }

    uint32_t sp_sz = sizeof(SimParams);
    int32_t T_w = TILE_T;
    // v8 multi-rank header: every checkpoint carries (num_ranks, rank_id,
    // num_cells_global). Single-GPU runs write (1, 0, nc). Multi-rank
    // runs write the per-rank coordinates, and each cell record uses its
    // GLOBAL id so the per-rank files can be merged unambiguously.
    ckpt::FixedPrefix prefix{};
    prefix.magic              = ckpt::MAGIC;
    prefix.version            = ckpt::VERSION_CURRENT;
    prefix.step               = step_count;
    prefix.cur_time           = cur_time;
    prefix.num_cells_local    = n;
    prefix.save_interval      = params.save_interval;
    prefix.reserved           = 0;
    prefix.trajectory_samples = params.trajectory_samples;
    prefix.sp_sz              = sp_sz;
    fwrite(&prefix, sizeof(prefix), 1, f);
    fwrite(&params, sp_sz, 1, f);
    fwrite(&T_w, sizeof(T_w), 1, f);

    ckpt::RankTrailer trailer{};
    trailer.num_ranks        = (gpus > 1) ? gpus : 1;
    trailer.rank_id          = (gpus > 1) ? rank : 0;
    trailer.num_cells_global = (gpus > 1) ? cells_global : n;
    fwrite(&trailer, sizeof(trailer), 1, f);

    auto wrap_d = [](double v, int L) {
        double m = std::fmod(v, (double)L);
        if (m < 0) m += L;
        return m;
    };
    const int Nx = params.Nx, Ny = params.Ny;

    std::vector<float> tile(TILE_AREA);
    for (int i = 0; i < n; i++) {
        double invV = (V[i] > 1e-6f) ? 1.0 / V[i] : 0.0;
        float cx = (float)wrap_d(h_or[2*i + 0] + Cx[i] * invV, Nx);
        float cy = (float)wrap_d(h_or[2*i + 1] + Cy[i] * invV, Ny);
        float vol = V[i] * (float)params.dA();

        ckpt::CellRecordHeader rec{};
        rec.cell_id  = (gpus > 1 && (int)h_global_id.size() > i) ? h_global_id[i] : i;
        rec.origin_x = h_or[2*i + 0];
        rec.origin_y = h_or[2*i + 1];
        rec.cx       = cx;
        rec.cy       = cy;
        rec.vx       = vx[i];
        rec.vy       = vy[i];
        rec.volume   = vol;
        fwrite(&rec, sizeof(rec), 1, f);

        CK(cudaMemcpy(tile.data(),
                      cells.phi_in + (size_t)i * TILE_AREA,
                      TILE_AREA * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(tile.data(), sizeof(float), TILE_AREA, f);
    }

    auto write_per_cell = [&](uint32_t m, const float* dev_ptr) {
        std::vector<float> h(n);
        CK(cudaMemcpy(h.data(), dev_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
        ckpt::SidecarBlockHeader sh{m, n};
        fwrite(&sh, sizeof(sh), 1, f);
        fwrite(h.data(), sizeof(float), n, f);
    };
    for (auto& fld : per_cell_float_state()) {
        if (fld.sidecar_magic != 0) write_per_cell(fld.sidecar_magic, *fld.dev_ptr);
    }

    // C2: RNGS sidecar — per-cell curandState bytes so resume preserves
    // the random-stream continuity. Without this, chained jobs replay the
    // same tumble decisions from offset 0 every time finalize_init runs.
    {
        const size_t bytes = (size_t)n * sizeof(curandState);
        std::vector<uint8_t> h_rng(bytes);
        CK(cudaMemcpy(h_rng.data(), cells.rng_states, bytes, cudaMemcpyDeviceToHost));
        ckpt::SidecarBlockHeader sh{ckpt::MAGIC_RNGS, n};
        fwrite(&sh, sizeof(sh), 1, f);
        fwrite(h_rng.data(), 1, bytes, f);
    }

    fclose(f);

    // Atomic rename: replace existing checkpoint.bin with the .tmp we just
    // wrote. std::filesystem::rename replaces on both Windows and POSIX.
    try {
        std::filesystem::rename(fn_tmp, fn);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to rename %s -> %s: %s\n", fn_tmp, fn, e.what());
        return;
    }
    printf("Saved checkpoint: step=%d, t=%.4f, cells=%d (%s)\n", step_count, cur_time, n, fn);
}

// ---------------------------------------------------------------------------
// cleanup
// ---------------------------------------------------------------------------
void Simulation::cleanup() {
    finish_trajectory_writer();
    for (int i = 0; i < 2; ++i) {
        if (step_graph[i])    { cudaGraphExecDestroy(step_graph[i]);    step_graph[i] = nullptr; }
        if (mg_step_graph[i]) { cudaGraphExecDestroy(mg_step_graph[i]); mg_step_graph[i] = nullptr; }
        step_graph_built[i]    = false;
        mg_step_graph_built[i] = false;
    }
    if (step_stream) { cudaStreamDestroy(step_stream); step_stream = nullptr; }
    auto cf = [](auto& p) { if (p) { cudaFree(p); p = nullptr; } };
    cf(cells.phi_pool);
    cells.phi_in = cells.phi_out = nullptr;
    phi_A = phi_B = nullptr;
    cf(cells.S);
    cf(cells.origin);
    cf(cells.rect);
    cf(cells.volumes); cf(cells.Ix); cf(cells.Iy);
    cf(cells.Cx); cf(cells.Cy);
    cf(cells.Cxx); cf(cells.Cyy);
    cf(cells.perimeters);
    cf(cells.velocities_x); cf(cells.velocities_y);
    for (auto& f : per_cell_float_state()) {
        cf(*f.dev_ptr);
        if (f.scratch_ptr) cf(*f.scratch_ptr);
    }
    cf(cells.rng_states);
    cf(d_scripted_cid);
    cf(d_scripted_theta);
    // Migration buffers (multi-GPU only; nullptr for G=1).
    // d_mig_counts is the real allocation; d_n_* are aliases into it.
    cf(d_mig_counts);
    d_n_stay = d_n_up = d_n_down = d_n_in_prev = d_n_in_next = nullptr;
    cf(d_stay_idx);  cf(d_up_idx);    cf(d_down_idx);
    auto cfv = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    cfv(d_pack_up);      cfv(d_pack_down);
    cfv(d_pack_in_prev); cfv(d_pack_in_next);
    cf(d_origin_scratch); cf(d_rect_scratch);
    cfv(d_rng_scratch);
    cf(d_gid_src); cf(d_gid_arr);
}

// ===========================================================================
// Multi-GPU orchestrator (single process, one host thread, NCCL).
// Compiled only when ENABLE_MULTI_GPU is ON. main.cu always sees the
// declaration (in sim.cuh) but only invokes it under mg_available().
// ===========================================================================

#include "multi_gpu.cuh"
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#ifdef ENABLE_MULTI_GPU

#ifdef _WIN32
  #include <direct.h>
  #define MG_MKDIR(d) _mkdir(d)
#else
  #include <sys/stat.h>
  #define MG_MKDIR(d) mkdir(d, 0755)
#endif

// ---------------------------------------------------------------------------
// MgBarrier — reusable cyclic barrier. Required because we drive the multi-
// GPU step pipeline from one persistent host thread per rank: each step has
// distinct phases (kernel-launch vs I/O) that must not overlap. A barrier
// here lets all G workers finish a phase before the main thread observes the
// per-rank state for I/O / termination.
// ---------------------------------------------------------------------------
struct MgBarrier {
    std::mutex              m;
    std::condition_variable cv;
    int                     n;            // total participants (workers + main)
    int                     arrived = 0;
    uint64_t                gen     = 0;

    explicit MgBarrier(int participants) : n(participants) {}

    void wait() {
        std::unique_lock<std::mutex> lk(m);
        uint64_t my_gen = gen;
        if (++arrived == n) {
            arrived = 0;
            ++gen;
            cv.notify_all();
        } else {
            cv.wait(lk, [&]{ return gen != my_gen; });
        }
    }
};

static void write_multi_gpu_vtk(const std::vector<std::unique_ptr<Simulation>>& sims,
                                const MgWorld& world) {
    if (sims.empty()) return;
    Simulation& root = *sims[0];
    const int Nx = root.params.Nx, Ny = root.params.Ny;
    std::vector<float> grid((size_t)Nx * Ny, 0.0f);
    std::vector<int> h_or;
    std::vector<float> tile;

    for (int g = 0; g < (int)sims.size(); ++g) {
        CK(cudaSetDevice(world.devices[g]));
        CK(cudaDeviceSynchronize());
        composite_vtk_cells(*sims[g], grid, h_or, tile);
    }
    write_vtk_grid_file(root.out_dir, root.step_count, root.cur_time, Nx, Ny, grid);
}

// ---------------------------------------------------------------------------
// Simulation::migrate_cells — implementation under ENABLE_MULTI_GPU.
//
// Called from the orchestrator's main thread between barrier sync points
// on rebind boundaries (step_count % REBIND_EVERY == 0). The full
// sequence is:
//
//   1. Zero counters, classify all local cells -> stay/up/down lists.
//   2. Download n_up/n_down to host. Range-check vs max_migrants_per_dir.
//   3. NCCL-exchange the four counts (n_up to prev, n_down to next;
//      recv n_in_prev, n_in_next).
//   4. Pack outgoing into d_pack_up / d_pack_down.
//   5. NCCL-exchange the pack bytes.
//   6. Compact stays from current cell arrays into scratch arrays
//      (using d_stay_idx as the gather permutation), and compact phi
//      from phi_in into phi_out.
//   7. Unpack arrivals into the freshly-compacted arrays at slot
//      [n_stay, n_stay + n_in_prev + n_in_next), writing phi tiles into
//      phi_out (which becomes phi_in after the swap below).
//   8. Swap the scratch arrays into CellArrays (and parity-flip phi) so
//      the freshly-arranged data becomes "current" for the next step.
//   9. Update num_cells, the host-side h_global_id vector.
//
// Returns the new local cell count.
// ---------------------------------------------------------------------------
int Simulation::migrate_cells(MgWorld& world, int my_rank) {
    if (gpus <= 1) return cells.num_cells;
    cudaStream_t s = step_stream;

    const int prev_rank = (my_rank - 1 + gpus) % gpus;
    const int next_rank = (my_rank + 1) % gpus;

    // 1. Zero counters, classify.
    CK(cudaMemsetAsync(d_mig_counts, 0, 5 * sizeof(int), s));
    launch_classify_migrants(
        cells.origin, cells.num_cells,
        slab_y_lo(), slab_y_hi(), params.Ny,
        my_rank, gpus,
        d_n_stay, d_n_up, d_n_down,
        d_stay_idx, d_up_idx, d_down_idx, s);

    // 2. NCCL count exchange (device-to-device, no host involvement).
    //    Issue immediately after classify so we only need one host sync
    //    afterwards to download all 5 counts at once.
    mg_group_start();
    mg_send_recv_i32(world.comms[my_rank], d_n_up,   prev_rank, d_n_in_prev, prev_rank, s);
    mg_send_recv_i32(world.comms[my_rank], d_n_down, next_rank, d_n_in_next, next_rank, s);
    mg_group_end();

    // 3. Single host download of all 5 counts (was 3+2 separate copies).
    int h_counts[5] = {0, 0, 0, 0, 0};
    CK(cudaMemcpyAsync(h_counts, d_mig_counts, 5 * sizeof(int),
                       cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    int h_n_stay    = h_counts[0];
    int h_n_up      = h_counts[1];
    int h_n_down    = h_counts[2];
    int h_n_in_prev = h_counts[3];
    int h_n_in_next = h_counts[4];

    if (h_n_up > max_migrants_per_dir ||
        h_n_down > max_migrants_per_dir) {
        fprintf(stderr,
            "[FATAL] migration: rank %d exceeded max_migrants_per_dir=%d "
            "(up=%d down=%d). The per-direction pack buffer is sized at "
            "alloc time to max(128, capacity/16). Raise it by increasing "
            "cell capacity (currently %d) or by hand-editing alloc_gpu().\n",
            my_rank, max_migrants_per_dir, h_n_up, h_n_down, cells.capacity);
        std::exit(1);
    }

    int h_n_in_total = h_n_in_prev + h_n_in_next;
    int new_num_cells = h_n_stay + h_n_in_total;
    if (new_num_cells > cells.capacity) {
        fprintf(stderr,
            "[FATAL] migration: rank %d would have %d cells > capacity %d\n",
            my_rank, new_num_cells, cells.capacity);
        std::exit(1);
    }

    // Fast exit if no migration on either side.
    if (h_n_up == 0 && h_n_down == 0 &&
        h_n_in_prev == 0 && h_n_in_next == 0) {
        return cells.num_cells;
    }

    // 4. Pack outgoing.
    //    Upload h_global_id into the persistent d_gid_src buffer (allocated
    //    once in alloc_gpu). Pack reads gid from there; unpack writes the
    //    arrival cells' gids into d_gid_arr at the trailing slots so the
    //    host post-step code can read them back at the right offset.
    if (cells.num_cells > 0) {
        CK(cudaMemcpyAsync(d_gid_src, h_global_id.data(),
                           (size_t)cells.num_cells * sizeof(int),
                           cudaMemcpyHostToDevice, s));
    }
    if (h_n_up   > 0) launch_pack_migrants(cells, d_up_idx,   h_n_up,
                                           d_gid_src, d_pack_up,   s);
    if (h_n_down > 0) launch_pack_migrants(cells, d_down_idx, h_n_down,
                                           d_gid_src, d_pack_down, s);

    // 5. NCCL pack-bytes exchange. Each direction's send/recv are issued
    //    independently inside one group bracket so NCCL can pair them
    //    across ranks. n=0 calls are skipped (NCCL's 0-byte messages
    //    work but it's cleaner to skip).
    const std::size_t pack = CELL_PACK_BYTES;
    mg_group_start();
    mg_send_bytes(world.comms[my_rank], d_pack_up,
                  prev_rank, (size_t)h_n_up   * pack, s);
    mg_send_bytes(world.comms[my_rank], d_pack_down,
                  next_rank, (size_t)h_n_down * pack, s);
    mg_recv_bytes(world.comms[my_rank], d_pack_in_prev,
                  prev_rank, (size_t)h_n_in_prev * pack, s);
    mg_recv_bytes(world.comms[my_rank], d_pack_in_next,
                  next_rank, (size_t)h_n_in_next * pack, s);
    mg_group_end();

    // 6. Compact stays into scratch arrays + phi_out.
    if (h_n_stay > 0) {
        launch_compact_stays(
            cells, d_stay_idx, h_n_stay,
            cells.phi_out,
            d_origin_scratch, d_rect_scratch,
            d_gamma_scratch, d_v_A_scratch, d_tgt_R_scratch,
            d_polar_theta_scratch, d_polar_x_scratch, d_polar_y_scratch,
            d_rng_scratch, s);
    }

    // 7. Unpack arrivals into [h_n_stay, h_n_stay + h_n_in_total).
    //    Phi tiles land in phi_out; scalars land directly in CellArrays
    //    fields (NOT scratch — we'll swap scratch into them after).
    //    But arrivals' scalars need to ALSO land in scratch so the swap
    //    leaves a consistent state. So unpack should write to scratch.
    //
    //    Simpler approach: unpack into scratch arrays at the trailing
    //    slots. We adjust k_unpack_migrants's output pointers to point
    //    at scratch instead of cells.* by temporarily aliasing.
    //
    //    For phi the unpack already writes to phi_out, which is correct.

    auto unpack_into_scratch = [&](void* pack_buf, int n_in, int dst_offset) {
        if (n_in <= 0) return;
        // Build a temp CellArrays-like view that aims unpack at scratch.
        // k_unpack_migrants writes phi_out (cells.phi_out — correct) and
        // origin/rect/scalars (we redirect those to scratch arrays).
        CellArrays tmp = cells;       // shallow copy
        tmp.origin       = d_origin_scratch;
        tmp.rect         = d_rect_scratch;
        tmp.polar_theta  = d_polar_theta_scratch;
        tmp.polar_x      = d_polar_x_scratch;
        tmp.polar_y      = d_polar_y_scratch;
        tmp.gamma_cell   = d_gamma_scratch;
        tmp.v_A_cell     = d_v_A_scratch;
        tmp.tgt_radius   = d_tgt_R_scratch;
        tmp.rng_states   = d_rng_scratch;
        // phi_out, capacity left as-is.
        launch_unpack_migrants(tmp, pack_buf, n_in, dst_offset,
                               d_gid_arr, s);
    };
    unpack_into_scratch(d_pack_in_prev, h_n_in_prev, h_n_stay);
    unpack_into_scratch(d_pack_in_next, h_n_in_next, h_n_stay + h_n_in_prev);

    // 8. Swap scratch into CellArrays. After this:
    //      cells.origin etc. = the freshly compacted+arrived data
    //      d_*_scratch = the old (stale) arrays, available for the next
    //                    migration round
    //    For phi: swap phi_in / phi_out via parity flip so the freshly
    //    compacted+arrived tiles (currently in phi_out) become phi_in.
    std::swap(cells.origin,       d_origin_scratch);
    std::swap(cells.rect,         d_rect_scratch);
    for (auto& f : per_cell_float_state()) {
        if (f.scratch_ptr) std::swap(*f.dev_ptr, *f.scratch_ptr);
    }
    {
        // rng_states is void* in CellArrays; do an unstructured swap.
        void* tmp = cells.rng_states;
        cells.rng_states = d_rng_scratch;
        d_rng_scratch = tmp;
    }
    // Phi parity flip: the new state lives in phi_out, swap it to phi_in.
    flip_parity();

    // 9. Update num_cells and host-side h_global_id. Stays inherit
    //    their old gid via stay_idx; arrivals' gids are read back from
    //    d_gid_arr (filled by the unpack kernels).
    cells.num_cells = new_num_cells;
    {
        std::vector<int> old_gid(h_global_id);
        std::vector<int> h_stay_idx(h_n_stay);
        if (h_n_stay > 0) {
            CK(cudaMemcpyAsync(h_stay_idx.data(), d_stay_idx,
                               h_n_stay * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
        }
        h_global_id.assign(new_num_cells, -1);
        if (h_n_in_total > 0) {
            CK(cudaMemcpyAsync(h_global_id.data() + h_n_stay,
                               d_gid_arr + h_n_stay,
                               (size_t)h_n_in_total * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
        }
        CK(cudaStreamSynchronize(s));
        for (int k = 0; k < h_n_stay; ++k) {
            h_global_id[k] = old_gid[h_stay_idx[k]];
        }
    }

    return cells.num_cells;
}

int run_multi_gpu(const MultiGpuRunArgs& args) {
    MgWorld world;
    if (!mg_init_world(args.gpus, world)) return 1;

    // ---- Single source of truth for seeds across ranks ----
    // Each rank's Simulation::init() will draw a random_device seed if
    // params.seed == 0. Letting that happen per-rank produces different
    // populations on each rank → place_cells generates different cell
    // layouts → slice_cells_to_local keeps a random subset, and the
    // summed counts drift below N (this was the 5/3200 loss bug in
    // eq_3200_va015_v2). Resolve once here so every rank gets identical
    // seeds and therefore identical h_cells before slicing.
    SimParams seeded_params = args.params;
    if (args.ckpt_path.empty()) {
        if (seeded_params.seed == 0) {
            std::random_device rd;
            unsigned s = rd();
            if (s == 0) s = 0xC0FFEEu;
            seeded_params.seed = s;
        }
        if (seeded_params.polarity_seed == 0) {
            std::random_device rd;
            unsigned s = rd();
            if (s == 0) s = 0xDECAFBADu;
            seeded_params.polarity_seed = s;
        }
    }

    // ---- Per-rank Simulation construction & init ----
    std::vector<std::unique_ptr<Simulation>> sims;
    sims.reserve(args.gpus);
    for (int g = 0; g < args.gpus; ++g) {
        CK(cudaSetDevice(world.devices[g]));
        auto sim = std::make_unique<Simulation>();
        sim->gpus    = args.gpus;
        sim->rank    = g;
        sim->device  = world.devices[g];
        // Per-rank output dir. Rank 0 writes to the user-supplied --output
        // path (so the canonical files land where users expect); other
        // ranks write to {outdir}/rank{g}/ to keep their slice
        // checkpoints / trajectories segregated.
        sim->out_dir = (g == 0) ? args.outdir
                                : args.outdir + "/rank" + std::to_string(g);
        MG_MKDIR(sim->out_dir.c_str());

        sim->save_final_checkpoint = args.save_final;
        sim->checkpoint_interval   = args.checkpoint_interval;
        sim->gamma_spec            = args.gamma_spec;
        sim->v_A_sigma             = args.v_A_sigma;
        sim->vtk_interval          = args.vtk_interval;

        if (!args.ckpt_path.empty()) {
            if (!sim->init_from_checkpoint(args.ckpt_path,
                                           args.params, args.ov)) {
                mg_finalize_world(world);
                return 1;
            }
        } else {
            if (g == 0) {
                printf("=== Phase-Field Cell Simulation (v2, multi-GPU --gpus %d) ===\n",
                       args.gpus);
                printf("Cells (global): %d, R=%.1f, Domain: %dx%d\n",
                       args.ncells_global, args.params.target_radius,
                       args.params.Nx, args.params.Ny);
                printf("gamma=%.2f, kappa=%.2f, mu=%.2f, lambda=%.2f\n",
                       args.params.gamma, args.params.kappa,
                       args.params.mu, args.params.lambda);
                printf("v_A=%.4f, xi=%.1f, tau=%.1f, dt=%.4f, t_end=%.1f\n",
                       args.params.v_A, args.params.xi, args.params.tau,
                       args.params.dt, args.params.t_end);
            }
            sim->init(seeded_params, args.ncells_global);
        }
        sims.push_back(std::move(sim));
    }

    // After init, every rank has its own step_stream, its own NCCL comm,
    // and its own (replicated) cells.S of size Nx*Ny floats. The S values
    // are stale until the first step's pre-reduce fills them; the initial
    // velocity is computed inside finalize_init using each rank's LOCAL
    // S (sum of its own cells only). For motility-driven runs that is a
    // small first-step transient that washes out within ~tau / dt steps,
    // so we accept it for first bring-up. (Phase B: do an initial NCCL
    // all-reduce on S right after finalize_init and re-run
    // launch_initial_velocity to seed the first trajectory write.)

    Simulation& s0 = *sims[0];
    int target_step = std::max(s0.step_count, (int)(s0.params.t_end / s0.params.dt));
    int total       = target_step - s0.step_count;
    auto t0         = std::chrono::high_resolution_clock::now();

    // Open per-rank trajectory files.
    for (int g = 0; g < args.gpus; ++g) {
        Simulation& s = *sims[g];
        if (s.traj_every <= 0) continue;
        std::string tp = s.out_dir + "/trajectory.txt";
        s.traj_fp = fopen(tp.c_str(), "a");
        if (!s.traj_fp) continue;
        setvbuf(s.traj_fp, nullptr, _IOFBF, 1 << 20);
        fseek(s.traj_fp, 0, SEEK_END);
        if (ftell(s.traj_fp) == 0) {
            fprintf(s.traj_fp,
                "# Trajectory data (rank %d, cells [%d, %d) of %d global)\n",
                g, s.cell_offset,
                s.cell_offset + (int)s.h_cells.size(), s.cells_global);
            fprintf(s.traj_fp,
                "# Format: time cell_id x y vx vy px py theta v_A_i L_n volume\n");
            fprintf(s.traj_fp,
                "# v_A=%.6f N_global=%d N_local=%d Lx=%d Ly=%d dim=2 dt=%.6f tau=%.4f\n",
                s.params.v_A, s.cells_global, (int)s.h_cells.size(),
                s.params.Nx, s.params.Ny, s.params.dt, s.params.tau);
        }
    }

    const size_t S_floats = (size_t)s0.params.Nx * s0.params.Ny;

    // ---------------------------------------------------------------------
    // Halo exchange staging buffers. Each rank's S buffer holds rows
    // [y_lo - HALO_H, y_hi + HALO_H), with two boundary "bands" of size
    // 2*HALO_H rows each: one at the top (local rows [0, 2*HALO_H)) and
    // one at the bottom (local rows [slab_h, slab_h + 2*HALO_H)). Each
    // exchange step: send my band to neighbour, recv neighbour's band
    // into staging, kernel-add staging into my band. After: both ranks'
    // bands hold the sum (the correct global S contribution from both).
    //
    // For G == 1 these stay at size 0 / nullptr; halo exchange is a no-op.
    // ---------------------------------------------------------------------
    const size_t halo_band_floats = (args.gpus > 1)
        ? (size_t)(2 * HALO_H) * s0.params.Nx
        : 0;
    std::vector<float*> halo_top_recv(args.gpus, nullptr);
    std::vector<float*> halo_bot_recv(args.gpus, nullptr);
    if (args.gpus > 1) {
        for (int g = 0; g < args.gpus; ++g) {
            CK(cudaSetDevice(world.devices[g]));
            CK(cudaMalloc(&halo_top_recv[g], halo_band_floats * sizeof(float)));
            CK(cudaMalloc(&halo_bot_recv[g], halo_band_floats * sizeof(float)));
        }

        // Init-time halo exchange + velocity recompute. Without this the
        // initial S is scatter-from-local-cells-only on each rank, so
        // cells near slab boundaries see an incomplete S and start with
        // slightly wrong velocities. After this fix, step-0 velocities
        // match G=1 exactly (modulo float-atomic noise in scatter).
        // Default streams are used here \u2014 init is one-shot, no need
        // to optimise.
        std::vector<std::thread> init_workers;
        init_workers.reserve(args.gpus);
        for (int g = 0; g < args.gpus; ++g) {
            init_workers.emplace_back([&, g] {
                CK(cudaSetDevice(world.devices[g]));
                Simulation& sim   = *sims[g];
                const int prev_r  = (g - 1 + args.gpus) % args.gpus;
                const int next_r  = (g + 1) % args.gpus;
                const int slab_h  = sim.cells.S_ext_height
                                    - 2 * sim.cells.S_halo_h;
                float* const S          = sim.cells.S;
                float* const my_top     = S;
                float* const my_bot     = S + (size_t)slab_h * s0.params.Nx;
                // S was scatter-only-local from finalize_init. Exchange
                // halo bands so cells near slab boundaries see neighbour
                // contributions, then recompute initial velocities.
                sim.launch_halo_exchange(world, g, prev_r, next_r,
                                         my_top, my_bot,
                                         halo_top_recv[g], halo_bot_recv[g],
                                         halo_band_floats);
                // Make sure the halo-add is visible to the default stream
                // before the velocity reduce reads S.
                CK(cudaStreamSynchronize(sim.step_stream));
                launch_initial_velocity_reduce(sim.cells, sim.params);
                CK(cudaDeviceSynchronize());
            });
        }
        for (auto& t : init_workers) t.join();
    }

    // ---------------------------------------------------------------------
    // Persistent worker threads, one per rank. Each thread:
    //   1. cudaSetDevice(g) once at startup (thread-local context).
    //   2. Loops:
    //        wait at start_barrier (woken by main thread once per step)
    //        if shutdown: break
    //        step_pre_reduce(); ncclGroupStart/AllReduce/GroupEnd;
    //        step_post_reduce();
    //        wait at end_barrier (releases main thread for I/O)
    //
    // Why threads (vs the previous single-host-thread loop):
    //   The previous orchestrator issued every kernel launch sequentially
    //   on one host thread, with cudaSetDevice() context flips between
    //   ranks. With G ranks each issuing ~7 launches/step, the host-side
    //   serialization added ~G * 7 * ~10us = ~280us/step at G=4 — enough
    //   to dominate the ~1ms/step compute at N=1152 and turn 4-GPU into
    //   a 3.15x slowdown vs single-GPU. Per-rank threads parallelize the
    //   launches, the GPUs run concurrently, and ncclGroupEnd
    //   synchronizes the all-reduce across threads at the right moment.
    // ---------------------------------------------------------------------
    MgBarrier start_barrier(args.gpus + 1);  // workers + main
    MgBarrier end_barrier(args.gpus + 1);
    std::atomic<bool> shutdown{false};

    std::vector<std::thread> workers;
    workers.reserve(args.gpus);
    for (int g = 0; g < args.gpus; ++g) {
        workers.emplace_back([&, g] {
            cudaSetDevice(world.devices[g]);
            // Stick this thread to its device for its entire lifetime.
            // No further cudaSetDevice calls in the hot loop.
            const int prev_rank = (g - 1 + args.gpus) % args.gpus;
            const int next_rank = (g + 1) % args.gpus;
            const int slab_h    = sims[g]->cells.S_ext_height
                                  - 2 * sims[g]->cells.S_halo_h;
            float* const S          = sims[g]->cells.S;
            float* const my_top_band = S;                                  // rows [0, 2H)
            float* const my_bot_band = S + (size_t)slab_h * s0.params.Nx;  // rows [slab_h, slab_h + 2H)

            while (true) {
                start_barrier.wait();
                if (shutdown.load(std::memory_order_acquire)) break;

                // ---- Per-step body ----
                // Fast path: capture polar + scatter + halo + evolve into
                // a CUDA graph keyed on parity. Eligible when next step
                // doesn't hit rebind, full-reduce, scripted events. The
                // captured region runs entirely on step_stream; flip_parity
                // and step_count/cur_time updates happen on the host
                // *after* the graph launch to keep host-side state
                // consistent and avoid re-capturing on every parity flip.
                //
                // Slow path: original direct-launch sequence below
                // (step_pre_reduce → halo → step_post_reduce). Used for
                // rebind steps, output steps, and scripted events. The
                // graph is invalidated after migration since pointers
                // and per-cell counts may have shifted.
                Simulation& sim = *sims[g];
                // Env-guarded graph-capture disable. Useful when a kernel
                // change interacts poorly with capture (see Win A / fused
                // halo_add investigation).
                static const bool no_mg_graph = []() {
                    const char* e = std::getenv("CELL_SIM_NO_MG_GRAPH");
                    return e && e[0] == '1';
                }();
                const bool fast = (args.gpus > 1)
                                  && sim.mg_step_is_fast_path()
                                  && !no_mg_graph;
                if (fast) {
                    int p = sim.parity;
                    sim.sync_pool_to_parity();
                    if (!sim.mg_step_graph_built[p]) {
                        cudaGraph_t graph = nullptr;
                        CK(cudaStreamBeginCapture(
                            sim.step_stream,
                            cudaStreamCaptureModeThreadLocal));
                        sim.launch_mg_fast_step_kernels(
                            world, g, prev_rank, next_rank,
                            my_top_band, my_bot_band,
                            halo_top_recv[g], halo_bot_recv[g],
                            halo_band_floats);
                        CK(cudaStreamEndCapture(sim.step_stream, &graph));
                        CK(cudaGraphInstantiate(
                            &sim.mg_step_graph[p], graph, nullptr, nullptr, 0));
                        cudaGraphDestroy(graph);
                        sim.mg_step_graph_built[p] = true;
                    }
                    CK(cudaGraphLaunch(sim.mg_step_graph[p], sim.step_stream));
                    sim.flip_parity();
                    sim.step_count++;
                    sim.cur_time += sim.params.dt;
                } else {
                    // ---- Slow path: direct launches (rebind / output / scripted) ----
                    sim.step_pre_reduce();
                    if (args.gpus > 1) {
                        sim.launch_halo_exchange(
                            world, g, prev_rank, next_rank,
                            my_top_band, my_bot_band,
                            halo_top_recv[g], halo_bot_recv[g],
                            halo_band_floats);
                    }
                    sim.step_post_reduce();
                }

                // Migration on rebind cadence (only when G > 1; for G=1
                // migrate_cells is a no-op early-out). step_post_reduce
                // ran k_rebind on this step iff (step_count is now a
                // multiple of REBIND_EVERY) — it does the increment last.
                //
                // Diagnostic: CELL_SIM_SKIP_MIGRATION=1 disables migration
                // entirely. Useful for short runs where no cell crosses a
                // slab boundary, isolating the per-step halo cost from
                // the migration host-sync cost. Does NOT disable rebind.
                static const bool skip_migration = []() {
                    const char* e = std::getenv("CELL_SIM_SKIP_MIGRATION");
                    return e && e[0] == '1';
                }();
                if (!skip_migration
                    && args.gpus > 1
                    && sim.step_count > 0
                    && (sim.step_count % REBIND_EVERY) == 0)
                {
                    sim.migrate_cells(world, g);
                    // Migration may have moved cells between ranks, shifting
                    // pointers and per-cell counts. Drop the cached graph
                    // so the next fast step rebuilds with fresh state.
                    sim.invalidate_mg_step_graph();

                    // Cell-loss audit (off by default; env-guarded). When
                    // CELL_SIM_AUDIT_CELLS=1 we sum num_cells across ranks
                    // via NCCL after every migration and FATAL on drift.
                    // Off path: a single getenv check at startup; cost
                    // amortised to zero.
                    static const bool audit_cells = []() {
                        const char* e = std::getenv("CELL_SIM_AUDIT_CELLS");
                        return e && e[0] == '1';
                    }();
                    if (audit_cells) {
                        int32_t* d_local = sim.d_n_in_prev;   // reuse 1-int scratch
                        int32_t  h_local = sim.cells.num_cells;
                        CK(cudaMemcpyAsync(d_local, &h_local, sizeof(int32_t),
                                           cudaMemcpyHostToDevice,
                                           sim.step_stream));
                        mg_allreduce_sum_i32(world.comms[g], d_local, 1,
                                             sim.step_stream);
                        int32_t h_total = 0;
                        CK(cudaMemcpyAsync(&h_total, d_local, sizeof(int32_t),
                                           cudaMemcpyDeviceToHost,
                                           sim.step_stream));
                        CK(cudaStreamSynchronize(sim.step_stream));
                        if (h_total != sim.cells_global) {
                            fprintf(stderr,
                                "[FATAL] cell-loss audit: rank %d step %d "
                                "sum=%d expected cells_global=%d "
                                "(local=%d)\n",
                                g, sim.step_count, h_total,
                                sim.cells_global, sim.cells.num_cells);
                            std::exit(1);
                        }
                    }
                }

                end_barrier.wait();
            }
        });
    }

    while (sims[0]->step_count < target_step) {
        // Wake all workers; they advance the step count atomically.
        start_barrier.wait();
        // Block until every worker has finished post_reduce.
        end_barrier.wait();

        // ---- I/O on cadence (per-rank, segregated dirs) ----
        int sc = sims[0]->step_count;
        bool wrote_any = false;
        if (s0.params.save_interval > 0 && sc % s0.params.save_interval == 0) {
            char tag[32]; snprintf(tag, sizeof(tag), "%08d", sc);
            for (int g = 0; g < args.gpus; ++g) {
                CK(cudaSetDevice(world.devices[g]));
                sims[g]->save_checkpoint(sims[g]->out_dir, tag);
            }
            wrote_any = true;
        }
        if (s0.checkpoint_interval > 0 && sc % s0.checkpoint_interval == 0) {
            for (int g = 0; g < args.gpus; ++g) {
                CK(cudaSetDevice(world.devices[g]));
                sims[g]->save_checkpoint(sims[g]->out_dir);
            }
            wrote_any = true;
        }
        if (sims[0]->traj_every > 0 && sc % sims[0]->traj_every == 0) {
            for (int g = 0; g < args.gpus; ++g) {
                CK(cudaSetDevice(world.devices[g]));
                sims[g]->write_trajectory();
            }
            wrote_any = true;
        }
        if (s0.vtk_interval > 0 && sc % s0.vtk_interval == 0) {
            write_multi_gpu_vtk(sims, world);
            wrote_any = true;
        }
        (void)wrote_any;
        if (s0.params.print_interval > 0 && sc % s0.params.print_interval == 0) {
            for (int g = 0; g < args.gpus; ++g) {
                CK(cudaSetDevice(world.devices[g]));
                printf("[rank %d] ", g);
                sims[g]->print_status();
            }
        }
    }

    // Tell workers to exit, then wake them so they observe the flag.
    shutdown.store(true, std::memory_order_release);
    start_barrier.wait();
    for (auto& t : workers) t.join();

    for (int g = 0; g < args.gpus; ++g) {
        if (sims[g]->traj_fp) {
            sims[g]->finish_trajectory_writer();
            fclose(sims[g]->traj_fp);
            sims[g]->traj_fp = nullptr;
        }
    }

    if (s0.save_final_checkpoint) {
        for (int g = 0; g < args.gpus; ++g) {
            CK(cudaSetDevice(world.devices[g]));
            sims[g]->save_checkpoint(sims[g]->out_dir);
        }
    }

    for (int g = 0; g < args.gpus; ++g) {
        CK(cudaSetDevice(world.devices[g]));
        CK(cudaDeviceSynchronize());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();
    int denom = std::max(1, total);
    printf("[SIM] Done (multi-GPU --gpus %d): %d steps, t=%.2f, wall=%.3fs (%.3f ms/step)\n",
           args.gpus, total, sims[0]->cur_time, wall, wall * 1000.0 / denom);

    for (int g = 0; g < args.gpus; ++g) {
        CK(cudaSetDevice(world.devices[g]));
        sims[g]->cleanup();
    }
    if (args.gpus > 1) {
        for (int g = 0; g < args.gpus; ++g) {
            cudaSetDevice(world.devices[g]);
            if (halo_top_recv[g]) cudaFree(halo_top_recv[g]);
            if (halo_bot_recv[g]) cudaFree(halo_bot_recv[g]);
        }
    }
    mg_finalize_world(world);
    return 0;
}

#endif  // ENABLE_MULTI_GPU
