// ===========================================================================
// FUSE-1R host side: allocation, initialisation, the per-step launch sequence,
// L2 residency policy, CUDA graph capture, diagnostics and state dump.
//
// Steady state is exactly ONE kernel launch per step. All cadence decisions
// (full-moment steps, shape-class changes, recentring) are device-side
// predicates, so the host never branches per step and a 6-step CUDA graph can
// be replayed unchanged.
// ===========================================================================

#include "../include/sim.cuh"
#include "../include/palmieri_initializer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace pf {

// ---------------------------------------------------------------------------
#define CU_CHECK(expr)                                                        \
    do {                                                                      \
        const cudaError_t _e = (expr);                                        \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[cuda] %s:%d %s -> %s\n", __FILE__,         \
                         __LINE__, #expr, cudaGetErrorString(_e));            \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define CU_WARN(expr)                                                         \
    do {                                                                      \
        const cudaError_t _e = (expr);                                        \
        if (_e != cudaSuccess)                                                \
            std::fprintf(stderr, "[cuda-warn] %s:%d %s -> %s\n", __FILE__,    \
                         __LINE__, #expr, cudaGetErrorString(_e));            \
    } while (0)

Sim::~Sim() {
    if (dual_centroid_fp_) std::fclose(dual_centroid_fp_);
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_)      cudaGraphDestroy(graph_);
    if (stream_)     cudaStreamDestroy(stream_);
    cudaFree(d_phi_[0]);
    cudaFree(d_phi_[1]);
    cudaFree(d_S_);
    cudaFree(d_cell_);
    cudaFree(d_cls_);
    cudaFree(d_perm_);
    cudaFree(d_cursor_);
    cudaFree(d_step_);
    cudaFree(d_flags_);
    cudaFree(d_vchk_);
    cudaFree(d_ochk_);
    cudaFree(d_smax_);
    if (h_traj_) cudaFreeHost(h_traj_);
    if (h_dual_centroid_) cudaFreeHost(h_dual_centroid_);
}

// ---------------------------------------------------------------------------
// Initial condition. The historical default is the grid+jitter layout. A
// fresh run may instead load one pre-generated Palmieri accepted-centre table;
// paired branches then consume the same bytes rather than merely regenerating
// from equal integer seeds.
// ---------------------------------------------------------------------------
bool Sim::seed_positions(std::vector<float>& cx, std::vector<float>& cy,
                         std::vector<float>& gam, std::vector<float>& va,
                         std::vector<int32_t>& gid)
{
    const int N = p_.num_cells;
    cx.resize(N); cy.resize(N); gam.resize(N); va.resize(N); gid.resize(N);

    if (!opt_.initial_centres_path.empty()) {
        PalmieriCentresCsvDiagnostics diag{};
        std::string error;
        if (!palmieri_read_centres_csv(opt_.initial_centres_path, N,
                                        (double)side_, p_.target_radius,
                                        &cx, &cy, &diag, &error)) {
            std::fprintf(stderr, "[fatal] invalid --initial-centres '%s': %s\n",
                         opt_.initial_centres_path.c_str(), error.c_str());
            return false;
        }
        std::printf("  initializer      %s\n", kPalmieriInitializerMethod);
        std::printf("  initial centres  %s  (%zu rows, min distance %.9g, "
                    "table FNV-1a %016llx)\n",
                    opt_.initial_centres_path.c_str(), diag.accepted_count,
                    diag.minimum_periodic_distance,
                    (unsigned long long)diag.table_fnv1a64);
    } else {
        const int nx = (int)std::ceil(std::sqrt((double)N));
        const double sp = (double)side_ / (double)nx;
        for (int i = 0; i < N; ++i) {
            const int gxi = i % nx;
            const int gyi = i / nx;
            // Counter domains are declared in kernels.cuh; the polarity and
            // v_A draws go through shared helpers so checkpoint re-derivation
            // lands on the same numbers.
            const Philox4 r = philox4x32_10(
                (uint32_t)i, kIcDomainJitter, 0u, 0u,
                (uint32_t)(p_.seed & 0xFFFFFFFFull),
                (uint32_t)(p_.seed >> 32));
            const double jx =
                (philox_uniform53(r.v[0], r.v[1]) - 0.5) * 0.10 * sp;
            const double jy =
                (philox_uniform53(r.v[2], r.v[3]) - 0.5) * 0.10 * sp;
            double x = ((double)gxi + 0.5 + ((gyi & 1) ? 0.5 : 0.0))
                       * sp + jx;
            double y = ((double)gyi + 0.5) * sp + jy;
            x -= std::floor(x / (double)side_) * (double)side_;
            y -= std::floor(y / (double)side_) * (double)side_;
            cx[(std::size_t)i] = (float)x;
            cy[(std::size_t)i] = (float)y;
        }
        std::printf("  initializer      grid+jitter (historical default)\n");
    }

    const int n_cancer = (int)std::llround(p_.cancer_fraction * (double)N);
    for (int i = 0; i < N; ++i) {
        gam[(std::size_t)i] =
            (float)((i < n_cancer) ? p_.gamma_cancer : p_.gamma_normal);
        // Per-cell v_A disorder: lognormal, median p_.v_A.
        va[(std::size_t)i] =
            (float)ic_v_A(i, p_.seed, p_.v_A, p_.v_A_sigma);
        gid[(std::size_t)i] = i;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Device query, allocation, stream and zeroing. Shared by the fresh and the
// resumed paths: neither is allowed to have its own copy of this.
// ---------------------------------------------------------------------------
bool Sim::alloc_device(const SimParams& p, const RunOptions& opt, int device) {
    p_ = p;
    opt_ = opt;
    device_ = device;

    CU_CHECK(cudaSetDevice(device_));

    cudaDeviceProp prop{};
    CU_CHECK(cudaGetDeviceProperties(&prop, device_));
    grid_ = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 132;
    l2_persist_max_ = (size_t)prop.persistingL2CacheMaxSize;
    l2_window_max_  = (size_t)prop.accessPolicyMaxWindowSize;

    side_  = p_.Nx;
    pitch_ = s_pitch_for(side_);
    s_buf_words_ = (size_t)pitch_ * (size_t)side_;

    const int N = p_.num_cells;
    const size_t pool_words = (size_t)N * (size_t)kTileArea;

    std::printf("--- device ---\n");
    std::printf("  %s  cc %d.%d  %d SMs  %.1f GiB  L2 %.1f MB "
                "(persist max %.1f MB, window max %.1f MB)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                (double)prop.totalGlobalMem / 1073741824.0,
                (double)prop.l2CacheSize / 1048576.0,
                (double)l2_persist_max_ / 1048576.0,
                (double)l2_window_max_ / 1048576.0);
    std::printf("  phi pool  2 x %.2f MB    S  3 x %.2f MB    cells %.2f MB\n",
                (double)pool_words * 4.0 / 1048576.0,
                (double)s_buf_words_ * 4.0 / 1048576.0,
                (double)N * sizeof(CellState) / 1048576.0);

    if (prop.major < 9)
        std::fprintf(stderr, "[warn] built for sm_90; this device is sm_%d%d\n",
                     prop.major, prop.minor);

    // --- allocation --------------------------------------------------------
    CU_CHECK(cudaMalloc(&d_phi_[0], pool_words * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_phi_[1], pool_words * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_S_, 3 * s_buf_words_ * sizeof(uint32_t)));
    CU_CHECK(cudaMalloc(&d_cell_, (size_t)N * sizeof(CellState)));
    CU_CHECK(cudaMalloc(&d_cls_, (size_t)N));
    CU_CHECK(cudaMalloc(&d_perm_, (size_t)N * sizeof(uint32_t)));
    CU_CHECK(cudaMalloc(&d_cursor_, 2 * sizeof(unsigned long long)));
    CU_CHECK(cudaMalloc(&d_step_, 2 * sizeof(unsigned long long)));
    CU_CHECK(cudaMalloc(&d_flags_, FLAG_COUNT * sizeof(uint32_t)));
    CU_CHECK(cudaMalloc(&d_vchk_, (size_t)N * sizeof(double)));
    CU_CHECK(cudaMalloc(&d_ochk_, (size_t)N * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_smax_, sizeof(uint32_t)));

    // Trajectory staging in mapped pinned memory: the GPU writes it directly
    // over the coherent NVLink-C2C link, so no D2H copy and no stream sync
    // ever appears on the critical path.
    CU_CHECK(cudaHostAlloc((void**)&h_traj_, (size_t)N * sizeof(TrajPackedCell),
                           cudaHostAllocMapped));
    CU_CHECK(cudaHostGetDevicePointer((void**)&d_traj_, h_traj_, 0));
    if (!opt_.dual_centroid_path.empty()) {
        CU_CHECK(cudaHostAlloc((void**)&h_dual_centroid_,
                               (size_t)N * sizeof(ValidationCentroidCell),
                               cudaHostAllocMapped));
        CU_CHECK(cudaHostGetDevicePointer((void**)&d_dual_centroid_,
                                           h_dual_centroid_, 0));
    }

    CU_CHECK(cudaStreamCreate(&stream_));

    CU_CHECK(cudaMemset(d_cursor_, 0, 2 * sizeof(unsigned long long)));
    CU_CHECK(cudaMemset(d_step_, 0, 2 * sizeof(unsigned long long)));
    CU_CHECK(cudaMemset(d_flags_, 0, FLAG_COUNT * sizeof(uint32_t)));
    CU_CHECK(cudaMemset(d_S_, 0, 3 * s_buf_words_ * sizeof(uint32_t)));
    return true;
}

// ---------------------------------------------------------------------------
// Everything after the initial condition exists: shared-memory opt-in,
// occupancy report, L2 carve-out, Morton sizing, graph capture. Shared by the
// fresh and the resumed paths.
// ---------------------------------------------------------------------------
bool Sim::configure_and_capture() {
    const int N = p_.num_cells;

    // Both opt-ins are done unconditionally and before ANY step launch: the
    // MaxDynamicSharedMemorySize attribute is per-kernel and a missing opt-in
    // is a launch failure, not a slowdown.
    configure_k_step_smem();
    CU_CHECK(cudaGetLastError());

    cudaFuncAttributes fa{};
    CU_CHECK(cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(k_step)));
    std::printf("  k_step: %d regs, %zu B local (spill), %d B static smem, "
                "%d B dynamic smem requested\n",
                fa.numRegs, (size_t)fa.localSizeBytes,
                (int)fa.sharedSizeBytes, kSmemBytes);
    // cudaFuncGetAttributes reports LOCAL MEMORY PER THREAD, which is the whole
    // stack frame, not the spill traffic: ptxas alone separates the two. A small
    // ABI frame is normal and zero-cost, so warning on any nonzero value cried
    // wolf on a kernel that had already been made spill-free. Gate on a budget
    // and say what the number actually is.
    if (fa.localSizeBytes > kLocalBytesBudget)
        std::fprintf(stderr,
            "[warn] k_step uses %zu B/thread of local memory, over the %zu B "
            "budget. That is the stack frame INCLUDING any spill; grep the "
            "build log for 'spill stores' for the breakdown.\n",
            (size_t)fa.localSizeBytes, kLocalBytesBudget);

    cudaFuncAttributes fallback_fa{};
    CU_CHECK(cudaFuncGetAttributes(
        &fallback_fa, reinterpret_cast<const void*>(k_step_fallback)));
    std::printf("  k_step_fallback: %d regs, %zu B local, %d B static smem, "
                "%d B dynamic smem requested\n",
                fallback_fa.numRegs, (size_t)fallback_fa.localSizeBytes,
                (int)fallback_fa.sharedSizeBytes, kScalarBytes);

    print_path_report();

    // --- L2 persistence carve-out -----------------------------------------
    // Only carve out when the two hot S buffers actually FIT. If they do not,
    // reserving the space still evicts phi from the rest of L2 while failing to
    // hold S. Measured on GH200 (60 MB L2, 37.5 MB persist max): forcing it on is
    // -4.3% at N=132 where it fits, but +7.2% at N=1056 and +6.5% at N=2112 where
    // it does not. Crossover is near N=500. Clearing the flag keeps this decision
    // consistent with apply_l2_window(), which gates on the same flag.
    if (opt_.l2_persist && l2_persist_max_ > 0) {
        const size_t budget = (size_t)(0.85 * (double)l2_persist_max_);
        const size_t need   = 2 * s_buf_words_ * sizeof(uint32_t);
        if (need <= budget) {
            CU_WARN(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, need));
            std::printf("  L2 persisting carve-out %.2f MB (S buffer %.2f MB each)\n",
                        (double)need / 1048576.0,
                        (double)s_buf_words_ * 4.0 / 1048576.0);
        } else {
            opt_.l2_persist = false;
            std::printf("  L2 persisting carve-out DISABLED: 2 x S = %.2f MB exceeds "
                        "the %.2f MB budget\n"
                        "      (reserving it would evict phi without holding S)\n",
                        (double)need / 1048576.0, (double)budget / 1048576.0);
        }
    }

    if (opt_.morton) {
        int M = 1;
        while (M < N) M <<= 1;
        const size_t sm = (size_t)M * sizeof(unsigned long long);
        if (sm > (size_t)kSmemPerBlockOptinSm90) {
            std::fprintf(stderr,
                "[warn] Morton sort needs %.1f KB of shared memory for N=%d; "
                "over the per-block limit. Falling back to identity order.\n",
                (double)sm / 1024.0, N);
            opt_.morton = false;
        } else {
            configure_morton_smem((int)sm);
            CU_CHECK(cudaGetLastError());
        }
    }

    if (opt_.use_graph) {
        if (!build_graph())
            std::fprintf(stderr, "[warn] graph capture failed; using per-step "
                                 "launches.\n");
    }
    return true;
}

// ---------------------------------------------------------------------------
bool Sim::init(const SimParams& p, const RunOptions& opt, int device) {
    // Prepare and validate the host initial condition before cudaSetDevice or
    // any allocation. A malformed shared table therefore fails without
    // opening a GPU context or consuming simulation time.
    p_ = p;
    opt_ = opt;
    side_ = p.Nx;
    std::vector<float> cx, cy, gam, va;
    std::vector<int32_t> gid;
    if (!seed_positions(cx, cy, gam, va, gid)) return false;

    if (!alloc_device(p, opt, device)) return false;
    const int N = p_.num_cells;

    // --- per-cell state ----------------------------------------------------

    std::vector<CellState> h_cell((size_t)N);
    std::memset(h_cell.data(), 0, h_cell.size() * sizeof(CellState));
    std::vector<uint8_t> h_cls((size_t)N, (uint8_t)kClassRound);
    std::vector<uint32_t> h_perm((size_t)N);
    for (int i = 0; i < N; ++i) {
        CellState& c = h_cell[(size_t)i];
        c.global_id = gid[i];
        c.gamma = gam[i];
        c.v_A   = va[i];
        c.R_tgt = (float)p_.target_radius;
        c.theta = ic_theta(gid[i], p_.polarity_stream());
        c.cls = (uint8_t)kClassRound;
        c.cls_written[0] = (uint8_t)kClassRound;
        c.cls_written[1] = (uint8_t)kClassRound;
        h_perm[(size_t)i] = (uint32_t)i;
    }
    CU_CHECK(cudaMemcpy(d_cell_, h_cell.data(), h_cell.size() * sizeof(CellState),
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_cls_, h_cls.data(), h_cls.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_perm_, h_perm.data(), h_perm.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

    float *d_cx = nullptr, *d_cy = nullptr;
    CU_CHECK(cudaMalloc(&d_cx, (size_t)N * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_cy, (size_t)N * sizeof(float)));
    CU_CHECK(cudaMemcpy(d_cx, cx.data(), (size_t)N * sizeof(float),
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_cy, cy.data(), (size_t)N * sizeof(float),
                        cudaMemcpyHostToDevice));

    k_init_tiles<<<N, 256>>>(d_phi_[0], d_phi_[1], d_cell_, d_cls_, N, side_,
                             d_cx, d_cy,
                             (float)init_radius(p_.target_radius, p_.lambda),
                             (float)interface_k(p_.lambda));
    CU_CHECK(cudaGetLastError());
    k_init_moments<<<N, kBlockThreads>>>(d_cell_, d_cls_, d_phi_[0], N);
    CU_CHECK(cudaGetLastError());
    k_scatter_all<<<N, 256>>>(d_phi_[0], d_cell_, d_cls_, d_S_, N, side_,
                              pitch_, d_flags_);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
    CU_WARN(cudaFree(d_cx));
    CU_WARN(cudaFree(d_cy));

    return configure_and_capture();
}

// ---------------------------------------------------------------------------
// Resume from a checkpoint.
//
// Mirrors init() exactly, with the initial condition read from disk instead of
// synthesised. The three things that make a resume different from a restart --
// and that are easy to get silently wrong -- are handled explicitly:
//
//  1. PHI PARITY. The step loop reads d_phi_[steps_done_ % 2], so the loaded
//     field goes into that half and the other is zeroed. (k_init_tiles zeroes
//     both on a fresh run; we do the same, so the two paths hand the first
//     step an identically-shaped world.)
//  2. S ROTATION. args_for_slot(slot) reads S rotation slot slot % 3, and the
//     loop enters at slot = steps_done_ % kGraphBody, so the scatter must land
//     in S[steps_done_ % 3], not S[0].
//  3. THE DEVICE STEP COUNTER *IS* THE RNG COUNTER. Philox is keyed on
//     (seed, global_id) with the counter (step, 0, 0, 0), so leaving d_step_
//     at zero would replay the first N steps of every cell's tumble stream on
//     every resume -- a systematic, silent bias in the persistence time. The
//     counter slots follow the phi parity, so slot [S%2] holds S and the other
//     holds S+1, which is exactly what the previous launch would have left.
//
// The graph needs no special handling: run() only replays it when
// steps_done_ % kGraphBody == 0, so a resume at an arbitrary step walks
// per-step launches until it reaches a body boundary.
// ---------------------------------------------------------------------------
bool Sim::init_from_checkpoint(const SimParams& p, const CheckpointData& d,
                               const RunOptions& opt, int device) {
    if ((int)d.cells.size() != d.n || d.phi.size() != (size_t)d.n * kTileArea) {
        std::fprintf(stderr, "[ckpt] payload is internally inconsistent "
                     "(%zu cells, %zu phi words, n=%d)\n",
                     d.cells.size(), d.phi.size(), d.n);
        return false;
    }
    if (p.num_cells != d.n) {
        std::fprintf(stderr, "[ckpt] params say %d cells, the file has %d\n",
                     p.num_cells, d.n);
        return false;
    }
    if (d.step < 0) {
        std::fprintf(stderr, "[ckpt] negative step %lld\n", d.step);
        return false;
    }
    if (!alloc_device(p, opt, device)) return false;
    const int N = p_.num_cells;

    steps_done_ = d.step;
    const int pin = (int)(steps_done_ % 2);
    const int rot = (int)(steps_done_ % 3);

    std::vector<CellState> h_cell((size_t)N);
    std::memset(h_cell.data(), 0, h_cell.size() * sizeof(CellState));
    std::vector<uint8_t>  h_cls((size_t)N);
    std::vector<uint32_t> h_perm((size_t)N);
    for (int i = 0; i < N; ++i) {
        const CkptCell& s = d.cells[(size_t)i];
        CellState& c = h_cell[(size_t)i];
        c.global_id = s.global_id;
        c.gx0 = s.origin[0];
        c.gy0 = s.origin[1];
        c.gamma = s.gamma;
        c.v_A   = s.v_A;
        c.R_tgt = s.R_tgt;
        c.theta = s.theta;
        c.vx = s.vx;
        c.vy = s.vy;
        c.cls = s.cls;
        // Both halves: the unloaded half is all zeros, so declaring it "already
        // written with this class" is true and spares the first step the frame
        // -zeroing pass a class change would otherwise trigger.
        c.cls_written[0] = s.cls;
        c.cls_written[1] = s.cls;
        h_cls[(size_t)i]  = s.cls;
        h_perm[(size_t)i] = (uint32_t)i;
    }
    CU_CHECK(cudaMemcpy(d_cell_, h_cell.data(), h_cell.size() * sizeof(CellState),
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_cls_, h_cls.data(), h_cls.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_perm_, h_perm.data(), h_perm.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

    const size_t pool_bytes = (size_t)N * kTileArea * sizeof(float);
    CU_CHECK(cudaMemcpy(d_phi_[pin], d.phi.data(), pool_bytes,
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(d_phi_[1 - pin], 0, pool_bytes));

    const unsigned long long h_step[2] = {
        (unsigned long long)(pin == 0 ? steps_done_ : steps_done_ + 1),
        (unsigned long long)(pin == 1 ? steps_done_ : steps_done_ + 1)};
    CU_CHECK(cudaMemcpy(d_step_, h_step, sizeof(h_step), cudaMemcpyHostToDevice));

    // V / Cx / Cy / bbox / phi_max are RECOMPUTED from the loaded field rather
    // than restored from the file's per-cell record: the record's cx/cy/volume
    // are f32 diagnostics of a different tiling, and the bbox is what drives
    // the shape-class decision. Recomputing costs one kernel launch, once.
    k_init_moments<<<N, kBlockThreads>>>(d_cell_, d_cls_, d_phi_[pin], N);
    CU_CHECK(cudaGetLastError());
    k_scatter_all<<<N, 256>>>(d_phi_[pin], d_cell_, d_cls_,
                              d_S_ + (size_t)rot * s_buf_words_, N, side_,
                              pitch_, d_flags_);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());

    // This engine derives time as step * dt rather than carrying an
    // accumulator, so a resume that changes --dt rescales the whole time axis
    // rather than continuing from the stored instant. That is a legitimate
    // thing to want and a very easy thing to do by accident, so say it out
    // loud instead of letting a trajectory silently start at the wrong t.
    if (std::fabs(d.t - time()) > 1e-6 * std::max(1.0, std::fabs(d.t)))
        std::fprintf(stderr,
            "[ckpt] warning: the file records t = %.6f at step %lld, but "
            "step * dt = %.6f with the dt now in force. Time is derived from "
            "the step count here, so the run continues at %.6f.\n",
            d.t, steps_done_, time(), time());

    std::printf("  resumed at step %lld (t = %.4f), phi parity %d, "
                "S rotation slot %d\n", steps_done_, time(), pin, rot);

    return configure_and_capture();
}

// ---------------------------------------------------------------------------
// Checkpoint output.
//
// Nothing here runs per step. save_checkpoint() is called only from the
// cadence branch in run(), which the step loop has already synchronised for
// its own print/trajectory work, so a checkpoint adds no synchronisation the
// loop was not already paying for. The D2H of the phi pool is streamed inside
// checkpoint_write() through a bounded staging buffer.
// ---------------------------------------------------------------------------
std::vector<std::string> Sim::checkpoint_paths(bool rolling, bool tagged) const {
    std::vector<std::string> paths;
    if (opt_.ckpt_dir.empty()) return paths;
    char buf[1100];
    if (rolling) {
        std::snprintf(buf, sizeof(buf), "%s/checkpoint.bin",
                      opt_.ckpt_dir.c_str());
        paths.emplace_back(buf);
    }
    if (tagged) {
        std::snprintf(buf, sizeof(buf), "%s/checkpoint_%08lld.bin",
                      opt_.ckpt_dir.c_str(), steps_done_);
        paths.emplace_back(buf);
    }
    return paths;
}

bool Sim::save_checkpoint(const std::vector<std::string>& paths) {
    if (paths.empty()) return true;
    const int N = p_.num_cells;
    const int pin = (int)(steps_done_ % 2);   // the buffer holding phi^step

    std::vector<CellState> h((size_t)N);
    std::vector<uint8_t>   hc((size_t)N);
    CU_CHECK(cudaMemcpy(h.data(), d_cell_, h.size() * sizeof(CellState),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(hc.data(), d_cls_, hc.size(), cudaMemcpyDeviceToHost));

    CheckpointWriteView v;
    v.p    = &p_;
    v.step = steps_done_;
    v.t    = time();
    v.N    = N;
    v.L    = side_;
    v.cell = h.data();
    v.cls  = hc.data();
    v.d_phi = d_phi_[pin];
    v.trajectory_samples = opt_.traj_samples;
    // The v8 field is int32; the CLI accepts up to 1e12. Clamp rather than
    // wrap: this value is metadata a later leg reads back as a cadence.
    v.save_interval = (int)std::min<long long>(opt_.save_interval, 2147483647LL);
    return checkpoint_write(v, paths);
}

// ---------------------------------------------------------------------------
// Which path is active, and what the hardware actually gives each kernel.
//
// The measured CTAs/SM is the number that matters: on sm_90 BOTH the register
// file and shared memory can pin a kernel to 1 CTA/SM independently, and the
// fused kernel is capped by both (80 regs x 768 threads = 61,440 of 65,536
// registers; 211,840 of 233,472 B of shared memory). Freeing only one of the
// two would change nothing.
// ---------------------------------------------------------------------------
void Sim::print_path_report() const {
    auto report = [&](const char* name, const void* fn, int threads,
                      int dyn_smem, int target_ctas, int reg_budget) {
        KernelStats s{};
        if (!query_kernel_stats(fn, threads, dyn_smem, device_, &s)) {
            std::fprintf(stderr, "[warn] occupancy query failed for %s\n", name);
            return;
        }
        std::printf("  %-12s %3d thr  %3d regs (budget %d)  %6d B smem "
                    "(%d static + %d dyn)  %zu B spill\n",
                    name, threads, s.regs, reg_budget,
                    s.static_smem + s.dynamic_smem, s.static_smem,
                    s.dynamic_smem, s.local_bytes);
        std::printf("  %-12s ACHIEVED %d CTAs/SM  (%d warps/SM, %.1f%% "
                    "occupancy);  register ceiling %d CTAs/SM;  target %d\n",
                    "", s.ctas_per_sm, s.warps_per_sm, 100.0 * s.occupancy,
                    s.reg_limited_ctas, target_ctas);
        if (s.local_bytes > kLocalBytesBudget)
            std::fprintf(stderr,
                "[warn] %s uses %zu B/thread of local memory to meet its "
                "register budget. Spilling to buy occupancy has already been\n"
                "       measured SLOWER on this kernel (the 1024-thread "
                "experiment). Compare against the fused path before trusting "
                "this build.\n", name, s.local_bytes);
        if (s.ctas_per_sm < target_ctas)
            std::fprintf(stderr,
                "[warn] %s reached only %d CTAs/SM, not the %d it was built "
                "for (%s). The occupancy win this path exists for was NOT "
                "obtained.\n", name, s.ctas_per_sm, target_ctas,
                s.reg_limited_ctas < target_ctas ? "register-limited"
                                                 : "shared-memory-limited");
    };

    std::printf("  exec path: k_step + sparse global-fallback filter, "
                "2 ordered launches/step\n");
    report("k_step", reinterpret_cast<const void*>(k_step),
           kBlockThreads, kSmemBytes, 1,
           kRegsPerSmSm90 / kBlockThreads);   // __launch_bounds__(768, 1)
    report("fallback", reinterpret_cast<const void*>(k_step_fallback),
           kBlockThreads, kScalarBytes, 1,
           kRegsPerSmSm90 / kBlockThreads);
}

// ---------------------------------------------------------------------------
// Per-slot argument baking. slot in [0, 6): phi parity = slot % 2, S rotation
// slot = slot % 3, cursor and step-counter slots follow the phi parity.
// ---------------------------------------------------------------------------
StepArgs Sim::args_for_slot(int slot) const {
    const int pin  = slot % 2;
    const int pout = 1 - pin;
    const int rs   = slot % 3;

    StepArgs A{};
    A.phi_in  = d_phi_[pin];
    A.phi_out = d_phi_[pout];
    A.S_rd = d_S_ + (size_t)rs * s_buf_words_;
    A.S_sc = d_S_ + (size_t)((rs + 1) % 3) * s_buf_words_;
    A.S_cl = d_S_ + (size_t)((rs + 2) % 3) * s_buf_words_;
    A.cell = d_cell_;
    A.cell_cls = d_cls_;
    A.perm = d_perm_;
    A.cursor_use   = d_cursor_ + pin;
    A.cursor_clear = d_cursor_ + pout;
    A.step_rd = d_step_ + pin;
    A.step_wr = d_step_ + pout;
    A.flags = d_flags_;

    A.N = p_.num_cells;
    A.L = side_;
    A.P = pitch_;
    A.parity_out = pout;

    A.dt         = (float)p_.dt;
    A.A0         = p_.area0();
    A.vol_scale  = p_.volume();
    A.bulk_scale = (float)p_.bulk();
    A.rep_coeff  = (float)p_.interaction();
    A.mot_coeff  = (float)p_.motility();

    A.seed = p_.seed;
    A.polarity_seed = p_.polarity_stream();
    A.p_tumble = p_.p_tumble();
    A.full_moment_every = p_.full_moment_every;
    A.clear_ahead_words = (unsigned long long)s_buf_words_;
    return A;
}

// The clear-ahead buffer is write-only streaming and must NOT be pinned: that
// is what leaves L2 room for phi. Pin the read buffer, and the scatter buffer
// too when the pair happens to be contiguous and the carve-out allows it.
void Sim::l2_window_for_slot(int slot, const void** base, size_t* bytes,
                             float* hit) const {
    *base = nullptr; *bytes = 0; *hit = 0.0f;
    if (!opt_.l2_persist || l2_persist_max_ == 0) return;

    const size_t buf = s_buf_words_ * sizeof(uint32_t);
    const int rs = slot % 3;
    const int ss = (rs + 1) % 3;

    size_t nb = buf;
    int    b0 = rs;
    if (ss == rs + 1) { nb = 2 * buf; b0 = rs; }      // contiguous read+scatter

    const size_t cap = std::min((size_t)(0.85 * (double)l2_persist_max_),
                                l2_window_max_);
    if (cap == 0) return;
    if (nb > cap) { nb = buf; }                        // read buffer only
    if (nb > cap) { nb = cap; }                        // still too big: clip

    *base  = (const void*)(d_S_ + (size_t)b0 * s_buf_words_);
    *bytes = nb;
    const size_t resident = std::min(nb, cap);
    *hit = (float)((double)resident / (double)nb);
    return;
}

void Sim::launch_one(int slot) {
    if (opt_.morton && (slot % kMortonEvery) == 0) {
        int M = 1;
        while (M < p_.num_cells) M <<= 1;
        k_morton_sort<<<1, 1024, (size_t)M * sizeof(unsigned long long),
                        stream_>>>(d_cell_, d_perm_, p_.num_cells, M, side_);
    }
    const void* base = nullptr; size_t nb = 0; float hit = 0.0f;
    l2_window_for_slot(slot, &base, &nb, &hit);
    launch_step(args_for_slot(slot), grid_, stream_, base, nb, hit);
}

bool Sim::build_graph() {
    CU_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    for (int s = 0; s < kGraphBody; ++s) launch_one(s);
    CU_CHECK(cudaStreamEndCapture(stream_, &graph_));
    CU_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_, graph_, 0));
    graph_ready_ = true;
    std::printf("  CUDA graph: %d-step body captured, %d kernel nodes (lcm of 2 "
                "phi parities and 3 S rotation slots)%s\n", kGraphBody,
                kGraphBody,
                opt_.morton ? " + Morton sort at slot 0" : "");
    return true;
}

// ---------------------------------------------------------------------------
void Sim::print_line() {
    std::vector<CellState> h((size_t)p_.num_cells);
    if (cudaMemcpy(h.data(), d_cell_, h.size() * sizeof(CellState),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return;
    double vsum = 0.0, vmin = 1e300, vmax = -1e300, spd = 0.0, pmax = 0.0;
    long long shifts = 0, tumbles = 0;
    int cls_count[kNumClasses] = {};
    int cls_bad = 0;
    int fallback_seen = 0;
    unsigned long long no_margin_steps = 0;
    for (const CellState& c : h) {
        vsum += c.V;
        vmin = std::min(vmin, c.V);
        vmax = std::max(vmax, c.V);
        spd += std::sqrt((double)c.vx * c.vx + (double)c.vy * c.vy);
        pmax = std::max(pmax, (double)c.phi_max);
        shifts += c.shift_ctr;
        tumbles += c.tumble_ctr;
        if (c.cls < kNumClasses) cls_count[c.cls]++; else cls_bad++;
        fallback_seen += c.reserved[0] != 0u;
        no_margin_steps += c.reserved[1];
    }
    const double n = (double)p_.num_cells;
    // cls prints round/wide/tall/big/large -- one field per class, generated
    // from kNumClasses rather than from a fixed format string, so adding a
    // class cannot silently hide its usage the way the old 3-field format did.
    // Any cell whose class is out of range is counted separately and printed
    // as a trailing "+K": that is a corrupt record, not a shape.
    char cls_str[16 * kNumClasses + 24];
    int cls_len = 0;
    for (int c = 0; c < kNumClasses; ++c)
        cls_len += std::snprintf(cls_str + cls_len,
                                 sizeof(cls_str) - (size_t)cls_len,
                                 c ? "/%d" : "%d", cls_count[c]);
    if (cls_bad)
        cls_len += std::snprintf(cls_str + cls_len,
                                 sizeof(cls_str) - (size_t)cls_len,
                                 "+%d BAD", cls_bad);
    std::printf("step %8lld  t %10.3f  <V>/A0 %.5f  V range [%.1f %.1f]  "
                "<|v|> %.4e  max|phi| %.5f  cls %s  shifts %lld  "
                "tumbles %lld\n",
                steps_done_, time(), vsum / n / p_.area0(), vmin, vmax,
                spd / n, pmax, cls_str, shifts, tumbles);
    if (fallback_seen && !fallback_reported_) {
        std::printf("[geometry] global fallback used by %d cell(s): "
                    "%dx%d interior in the %dx%d tile\n",
                    fallback_seen, kClasses[kClassFallback].wx,
                    kClasses[kClassFallback].wy, kTilePitch, kTilePitch);
        fallback_reported_ = true;
    }
    if (no_margin_steps && !fallback_no_margin_reported_) {
        std::printf("[geometry] fallback margin/boundary reached; output is "
                    "retained, but boundary dynamics may be clipped and require "
                    "review (cell-steps %llu)\n",
                    no_margin_steps);
        fallback_no_margin_reported_ = true;
    }
    std::fflush(stdout);
}

// Called on a bounded cadence independent of output cadence. Every flag except
// support_clip is production-fatal. A failed D2H read is also fatal because the
// host can no longer prove that the run remains valid.
bool Sim::fatal_flag_set() {
    uint32_t f[FLAG_COUNT] = {0};
    const cudaError_t e =
        cudaMemcpy(f, d_flags_, sizeof(f), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(stderr,
            "\n*** STOPPING AT STEP %lld: fatal-alarm readback failed: %s ***\n"
            "    Run validity can no longer be established.\n",
            steps_done_, cudaGetErrorString(e));
        return true;
    }

    bool fatal = false;
    for (int i = 0; i < FLAG_COUNT; ++i)
        fatal = fatal || (f[i] != 0u && flag_is_fatal(i));
    if (!fatal) return false;

    std::fprintf(stderr,
        "\n*** STOPPING AT STEP %lld: PRODUCTION-FATAL ALARM ***\n",
        steps_done_);
    for (int i = 0; i < FLAG_COUNT; ++i)
        if (f[i] != 0u && flag_is_fatal(i))
            std::fprintf(stderr, "    %-18s %u\n", flag_name(i), f[i]);
    if (f[FLAG_CLASS_EXHAUSTED] != 0u)
        std::fprintf(stderr,
            "    A cell outgrew every available shape class; its field can no "
            "longer be represented without truncation.\n");
    if (f[FLAG_CLASS_UNSUPPORTED] != 0u)
        std::fprintf(stderr, "    A CellState carried a shape class "
                             "outside the supported range.\n");
    std::fprintf(stderr,
        "    THE TRAJECTORY IS INVALID FROM THE FIRST SUCH STEP. A final "
        "checkpoint will be attempted for diagnosis, then the process will "
        "return nonzero.\n");
    return true;
}

void Sim::report_flags() const {
    uint32_t f[FLAG_COUNT] = {0};
    const cudaError_t e =
        cudaMemcpy(f, d_flags_, sizeof(f), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[alarms] readback failed: %s\n",
                     cudaGetErrorString(e));
        return;
    }
    // FLAG_SUPPORT_CLIP is ADVISORY, everything else is FATAL.
    //
    // It fires when the phi > kSupportEps (1e-5) bounding box merely TOUCHES the
    // window edge. That is a deliberately sensitive tripwire on the far
    // exponential tail, not a measure of lost mass. Measured directly at
    // t = 5000, N = 396, rho = 0.89 by dumping the state and summing phi^2 over
    // the window border ring:
    //
    //            cells with phi>1e-5 on border   max border phi   border phi^2 / total
    //   gamma=1        0 / 396                     4.05e-06            7.2e-17
    //   gamma=0.35     5 / 396                     3.57e-04            3.8e-13
    //
    // i.e. six to ten orders of magnitude below fp32 epsilon. Reporting that as
    // "THE RUN IS INVALID" cried wolf and would have caused good runs to be
    // discarded. FLAG_CLASS_EXHAUSTED is the flag that means real truncation --
    // no window could contain the cell at all -- and it stays fatal.
    bool fatal = false, advisory = false;
    for (int i = 0; i < FLAG_COUNT; ++i) {
        if (!f[i]) continue;
        if (i == FLAG_SUPPORT_CLIP) advisory = true;
        else if (flag_is_fatal(i)) fatal = true;
    }
    if (!fatal && !advisory) { std::printf("alarms: all clear\n"); return; }

    if (fatal) {
        std::printf("*** ALARMS SET -- THE RUN IS INVALID, NOT MERELY SLOW ***\n");
        for (int i = 0; i < FLAG_COUNT; ++i)
            if (f[i] && flag_is_fatal(i))
                std::printf("  %-18s %u\n", flag_name(i), f[i]);
    } else {
        std::printf("alarms: no fatal flags\n");
    }
    if (advisory) {
        const double frac = (double)f[FLAG_SUPPORT_CLIP]
                          / ((double)p_.num_cells * (double)(steps_done_ ? steps_done_ : 1));
        std::printf("  advisory: %-18s %u  (%.3f%% of cell-steps)\n"
                    "    the phi>1e-5 support bbox touched the window edge. This is the far\n"
                    "    tail, not the interface; measured border phi^2 is ~1e-13 of the total.\n"
                    "    Dump a state and check the border ring if you need the magnitude.\n",
                    flag_name(FLAG_SUPPORT_CLIP), f[FLAG_SUPPORT_CLIP], 100.0 * frac);
    }
#if !PF_SUPPORT_CLIP_ENABLED
    std::printf("  advisory: support_clip NOT INSTRUMENTED in this build\n");
#endif
}

bool Sim::verify(double* max_rel_V, float* max_outside, uint32_t* max_S) {
    const int N = p_.num_cells;
    const int pin = (int)(steps_done_ % 2);
    CU_CHECK(cudaMemset(d_smax_, 0, sizeof(uint32_t)));
    k_verify_cells<<<N, kBlockThreads, 0, stream_>>>(d_phi_[pin], d_cell_,
                                                     d_cls_, N, d_vchk_, d_ochk_);
    k_verify_S<<<256, 256, 0, stream_>>>(d_S_ + (size_t)(steps_done_ % 3)
                                             * s_buf_words_,
                                         s_buf_words_, d_smax_);
    CU_CHECK(cudaStreamSynchronize(stream_));

    std::vector<double> v((size_t)N);
    std::vector<float>  o((size_t)N);
    std::vector<CellState> h((size_t)N);
    CU_CHECK(cudaMemcpy(v.data(), d_vchk_, v.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(o.data(), d_ochk_, o.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(h.data(), d_cell_, h.size() * sizeof(CellState),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(max_S, d_smax_, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    double mr = 0.0;
    float mo = 0.0f;
    for (int i = 0; i < N; ++i) {
        const double den = std::max(1e-12, std::fabs(h[(size_t)i].V));
        mr = std::max(mr, std::fabs(v[(size_t)i] - h[(size_t)i].V) / den);
        mo = std::max(mo, o[(size_t)i]);
    }
    *max_rel_V = mr;
    *max_outside = mo;
    return true;
}

// ---------------------------------------------------------------------------
bool Sim::run() {
    const long long total = p_.total_steps();
    const long long pi = p_.print_interval > 0 ? p_.print_interval : total;
    bool run_failed = false;

    // The verify cadence must be a THRESHOLD, not a divisibility test.
    // steps_done_ advances in jumps of kGraphBody = 6 on a graph replay, so it
    // steps straight over an exact multiple: at the default verify_every = 4096
    // the sequence runs ...4092, 4098... and at the 512 the README documents it
    // runs ...504, 510, 516..., so `steps_done_ % verify_every == 0` never fired
    // once in a whole run and gates 6, 7 and 8 silently never executed.
    long long next_verify = p_.verify_every > 0 ? (long long)p_.verify_every
                                               : total + 1;

    // Trajectory cadence, independent of the print cadence. Sampling used to be
    // slaved to --print-interval, so `--print-interval 1000000` silently wrote a
    // ONE-FRAME trajectory -- syntactically valid, and useless for MSD.
    //
    // Both quantities below are measured against the span this invocation will
    // actually traverse, NOT against `total`, which is absolute. On a resume
    // (steps_done_ > 0) slaving them to `total` made --trajectory-samples 20000
    // yield 19231 frames at dt = 104: the 208 tau production divides the
    // ABSOLUTE 208e6 steps by 20000 to get 10400, then walks only the 200e6
    // steps remaining after the 8 tau equilibration. Uniform, so the MSD was
    // unharmed, but it silently under-delivered the requested resolution.
    const long long traj_span = total - steps_done_;
    const long long traj_every =
        opt_.out_path.empty() ? total + 1
        : (opt_.traj_interval > 0
               ? opt_.traj_interval
               : std::max<long long>(1, traj_span / std::max(1, opt_.traj_samples)));
    // Offset by steps_done_ so the first frame of a resumed leg lands one full
    // interval in, rather than firing on the very first step-check because the
    // absolute counter is already past a span-relative threshold.
    long long next_traj = steps_done_ + traj_every;

    // Checkpoint cadences. Both are thresholds, both are folded into next_stop
    // so the graph replay cannot jump over one, and both cost exactly two
    // integer compares per step when disabled.
    const bool ckpt_on = !opt_.ckpt_dir.empty();
    const long long ckpt_every = ckpt_on ? opt_.ckpt_interval : 0;
    const long long save_every = ckpt_on ? opt_.save_interval : 0;
    long long next_ckpt = ckpt_every > 0 ? steps_done_ + ckpt_every : total + 1;
    long long next_save = save_every > 0 ? steps_done_ + save_every : total + 1;
    long long next_fatal_poll = steps_done_ + kFatalPollEvery;

    if (!open_trajectory(opt_.out_path)) return false;
    if (!opt_.dual_centroid_path.empty()) {
        if (!open_dual_centroid(opt_.dual_centroid_path)) {
            close_trajectory();
            return false;
        }
    }

    while (steps_done_ < total) {
        const long long next_stop =
            std::min({total, ((steps_done_ / pi) + 1) * pi, next_traj,
                      next_ckpt, next_save, next_fatal_poll});

        if (graph_ready_ && (steps_done_ % kGraphBody) == 0 &&
            steps_done_ + kGraphBody <= next_stop) {
            CU_WARN(cudaGraphLaunch(graph_exec_, stream_));
            steps_done_ += kGraphBody;
        } else {
            launch_one((int)(steps_done_ % kGraphBody));
            steps_done_ += 1;
        }

        const bool do_print = (steps_done_ % pi == 0) || (steps_done_ >= total);
        const bool do_traj  = !opt_.out_path.empty() && steps_done_ >= next_traj;
        const bool do_ckpt  = ckpt_every > 0 && steps_done_ >= next_ckpt;
        const bool do_save  = save_every > 0 && steps_done_ >= next_save;
        const bool do_fatal_poll = steps_done_ >= next_fatal_poll;

        if (do_print || do_traj || do_ckpt || do_save || do_fatal_poll) {
            CU_WARN(cudaStreamSynchronize(stream_));
            if (fatal_flag_set()) {
                run_failed = true;
                break;
            }
            if (do_fatal_poll)
                next_fatal_poll = steps_done_ + kFatalPollEvery;
            if (do_print) print_line();
            if (do_traj) {
                const int N = p_.num_cells;
                k_pack_traj<<<(N + 127) / 128, 128, 0, stream_>>>(d_cell_, d_cls_,
                                                                  d_traj_, N, side_);
                if (d_dual_centroid_) {
                    const int pin = (int)(steps_done_ & 1LL);
                    launch_validation_centroids(d_phi_[pin], d_cell_, d_cls_,
                                                d_dual_centroid_, N, side_, stream_);
                    CU_WARN(cudaGetLastError());
                }
                CU_WARN(cudaStreamSynchronize(stream_));
                append_trajectory_frame(steps_done_);
                if (d_dual_centroid_)
                    append_dual_centroid_frame(steps_done_);
                next_traj = steps_done_ + traj_every;
            }
            // One gather feeds both files when both fall due on the same step.
            if (do_ckpt || do_save) {
                if (!save_checkpoint(checkpoint_paths(do_ckpt, do_save))) {
                    std::fprintf(stderr,
                        "[ckpt] checkpoint failed at step %lld; stopping to "
                        "avoid an unresumable allocation.\n",
                        steps_done_);
                    run_failed = true;
                    break;
                }
                if (do_ckpt) next_ckpt = steps_done_ + ckpt_every;
                if (do_save) next_save = steps_done_ + save_every;
            }
        }

        // Cooperative shutdown. SLURM sends SIGTERM before SIGKILL at walltime;
        // breaking here means everything sampled so far is already on disk.
        if (s_terminate) {
            CU_WARN(cudaStreamSynchronize(stream_));
            std::printf("\n[signal] termination requested at step %lld (t = %.3f); "
                        "%lld trajectory frames already written\n",
                        steps_done_, time(), traj_frames_);
            std::fflush(stdout);
            break;
        }
        if (opt_.strict && (p_.verify_every > 0) && steps_done_ >= next_verify) {
            double mr = 0.0; float mo = 0.0f; uint32_t ms = 0;
            if (verify(&mr, &mo, &ms))
                std::printf("  [verify] step %lld  max rel V error %.3e   "
                            "max|phi| outside window %.3e   max S %.6f\n",
                            steps_done_, mr, (double)mo, (double)ms * kQInvD);
            next_verify = steps_done_ + p_.verify_every;
        }
    }
    CU_WARN(cudaStreamSynchronize(stream_));
    if (!run_failed && fatal_flag_set())
        run_failed = true;
    if (traj_fp_)
        std::printf("trajectory -> %lld frames x %d cells (streamed)\n",
                    traj_frames_, p_.num_cells);
    if (dual_centroid_fp_)
        std::printf("dual centroids -> %lld frames x %d cells (streamed)\n",
                    dual_centroid_frames_, p_.num_cells);
    close_trajectory();
    if (dual_centroid_fp_) close_dual_centroid();
    // A valid exit advances checkpoint.bin. A fatal state is written under a
    // distinct diagnostic name: overwriting the last accepted rolling state
    // with the first invalid frame would destroy the recovery point.
    if (ckpt_on && (opt_.final_checkpoint || run_failed)) {
        std::vector<std::string> final_paths;
        if (run_failed) {
            final_paths.emplace_back(opt_.ckpt_dir + "/checkpoint_failed.bin");
        } else {
            final_paths = checkpoint_paths(true, false);
        }
        if (!save_checkpoint(final_paths)) {
            std::fprintf(stderr, "[ckpt] FINAL CHECKPOINT FAILED. The run "
                         "cannot be resumed from where it stopped.\n");
            run_failed = true;
        }
    } else if (run_failed && !ckpt_on) {
        std::fprintf(stderr,
            "[ckpt] fatal run has no checkpoint directory; no diagnostic "
            "checkpoint can be written.\n");
    }
    report_flags();
    return !run_failed;
}

bool Sim::bench(int steps, double* ms_per_step) {
    // Warm-up: fill caches, resolve the first graph upload, settle clocks.
    const int warm = std::min(steps, 200);
    for (int i = 0; i < warm; ++i) { launch_one((int)(steps_done_ % kGraphBody)); ++steps_done_; }
    CU_CHECK(cudaStreamSynchronize(stream_));

    cudaEvent_t e0, e1;
    CU_CHECK(cudaEventCreate(&e0));
    CU_CHECK(cudaEventCreate(&e1));

    // Align to a graph-body boundary so the timed region is pure graph replay.
    while ((steps_done_ % kGraphBody) != 0) {
        launch_one((int)(steps_done_ % kGraphBody));
        ++steps_done_;
    }
    CU_CHECK(cudaStreamSynchronize(stream_));

    const long long timed_from = steps_done_;
    CU_CHECK(cudaEventRecord(e0, stream_));
    while (steps_done_ - timed_from < steps) {
        if (graph_ready_) {
            CU_CHECK(cudaGraphLaunch(graph_exec_, stream_));
            steps_done_ += kGraphBody;
        } else {
            launch_one((int)(steps_done_ % kGraphBody));
            steps_done_ += 1;
        }
    }
    CU_CHECK(cudaEventRecord(e1, stream_));
    CU_CHECK(cudaEventSynchronize(e1));

    float ms = 0.0f;
    CU_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const long long done = steps_done_ - timed_from;
    *ms_per_step = (double)ms / (double)done;
    CU_WARN(cudaEventDestroy(e0));
    CU_WARN(cudaEventDestroy(e1));

    std::printf("bench: %lld steps in %.3f ms -> %.6f ms/step "
                "(%.2f us/step, %d cells, L=%d, %s, %s)\n",
                done, (double)ms, *ms_per_step, *ms_per_step * 1000.0,
                p_.num_cells, side_, "fused",
                graph_ready_ ? "graph" : "per-step launch");
    report_flags();
    return true;
}

// ---------------------------------------------------------------------------
volatile std::sig_atomic_t Sim::s_terminate = 0;

// --- streaming trajectory writer -------------------------------------------
// Append mode with the header written only when the file is empty, so a resumed
// or restarted leg extends the same file instead of truncating it.
bool Sim::open_trajectory(const std::string& path) {
    if (path.empty()) return true;
    traj_fp_ = std::fopen(path.c_str(), "a");
    if (!traj_fp_) {
        std::fprintf(stderr, "[error] cannot open %s for append\n", path.c_str());
        return false;
    }
    std::fseek(traj_fp_, 0, SEEK_END);
    traj_header_written_ = std::ftell(traj_fp_) > 0;
    if (!traj_header_written_) {
        std::fprintf(traj_fp_, "# Trajectory data\n");
        std::fprintf(traj_fp_,
            "# Format: time cell_id x y vx vy px py theta v_A_i L_n volume\n");
        std::fprintf(traj_fp_,
            "# v_A=%.6f N=%d Lx=%d Ly=%d dim=2 dt=%.6f tau=%.4f "
            "lambda=%.6f R=%.6f perim_offset=%.6f\n",
            p_.v_A, p_.num_cells, side_, side_, p_.dt, p_.tau,
            p_.lambda, p_.target_radius, kPi / interface_k(p_.lambda));
        // lambda / R / perim_offset are emitted so the L_n column can be
        // corrected downstream WITHOUT changing its definition here -- the
        // column stays byte-compatible with every run already on disk.
        //
        // L_n below is perim / (2 pi R). The numerator is the co-area integral
        // of |grad phi|, which is the perimeter of the phi = 1/2 contour; that
        // contour sits at R_eff = R + 1/(2k), because init_radius offsets it by
        // exactly that so integral(phi^2 dA) lands on pi R^2 (params.cuh:90-98).
        // Numerator and denominator therefore describe different circles and
        // L_n carries a fixed +1/(2kR) = +2.6% offset for a relaxed round cell.
        // Harmless for D_eff and MSD, which never touch it, but it shifts the
        // shape index p_eff = L_n * 2 sqrt(pi) by ~0.08 against the Bi/Manning
        // rigidity threshold p0* = 3.81 -- about a third of the whole distance
        // from a circle (3.545) to that threshold.
        //
        // The exact correction is a constant shift of the perimeter, since
        // closed-curve Steiner gives P(d) = P_half + 2 pi d:
        //     p_eff_corrected = (L_n * 2 pi R - perim_offset) / sqrt(volume)
        // with perim_offset = pi / k = pi lambda / sqrt(7.5) = 8.0300 px at
        // lambda = 7. Verified on the discrete production profile: 3.5423 vs a
        // geometric 2 sqrt(pi) = 3.54491, a -0.07% residual.
        traj_header_written_ = true;
    }
    return true;
}

void Sim::append_trajectory_frame(long long step_at) {
    if (!traj_fp_) return;
    const int N = p_.num_cells;
    const double tgt_r = p_.target_radius;
    const double Lw = (double)side_;
    auto wrap_d = [Lw](double v) {
        double m = std::fmod(v, Lw);
        if (m < 0.0) m += Lw;
        return m;
    };
    for (int i = 0; i < N; ++i) {
        const TrajPackedCell& c = h_traj_[i];
        const double th = (double)c.theta;
        const double l_n = (double)c.perim / (2.0 * kPi * tgt_r);
        std::fprintf(traj_fp_,
            "%.6f %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
            (double)step_at * p_.dt, c.global_id,
            wrap_d((double)c.cx), wrap_d((double)c.cy),
            (double)c.vx, (double)c.vy,
            std::cos(th), std::sin(th), th,
            (double)c.v_A, l_n, (double)c.volume);
    }
    ++traj_frames_;
    // Flush every frame: the cost is negligible against a step budget of
    // thousands of steps per frame, and it is what makes a killed run useful.
    std::fflush(traj_fp_);
}

void Sim::close_trajectory() {
    if (!traj_fp_) return;
    std::fflush(traj_fp_);
    std::fclose(traj_fp_);
    traj_fp_ = nullptr;
}

// --- validation-only dual-centroid writer ---------------------------------
bool Sim::open_dual_centroid(const std::string& path) {
    if (path.empty()) return true;
    dual_centroid_fp_ = std::fopen(path.c_str(), "a");
    if (!dual_centroid_fp_) {
        std::fprintf(stderr, "[error] cannot open %s for append\n", path.c_str());
        return false;
    }
    std::fseek(dual_centroid_fp_, 0, SEEK_END);
    if (std::ftell(dual_centroid_fp_) == 0) {
        std::fprintf(dual_centroid_fp_, "# Validation-only dual centroids\n");
        std::fprintf(dual_centroid_fp_,
            "# Format: time cell_id x_phi y_phi x_phi2_scan y_phi2_scan "
            "sum_phi sum_phi2_scan valid_phi valid_phi2_scan\n");
        std::fprintf(dual_centroid_fp_,
            "# periodic_lift=cell_rect phi_buffer=current "
            "phi2_source=independent_field_scan reduction=fixed_warp_order "
            "N=%d L=%d dt=%.17g\n",
            p_.num_cells, side_, p_.dt);
        std::fprintf(dual_centroid_fp_,
            "# x_phi/y_phi use weights phi; *_phi2_scan use independently "
            "rescanned phi^2, not legacy packed moments. Sums are unscaled "
            "lattice sums. Invalid centroids are nan.\n");
    }
    return true;
}

void Sim::append_dual_centroid_frame(long long step_at) {
    if (!dual_centroid_fp_ || !h_dual_centroid_) return;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < p_.num_cells; ++i) {
        const ValidationCentroidCell& c = h_dual_centroid_[i];
        const bool valid_phi = (c.valid_mask & kCentroidPhiValid) != 0u;
        const bool valid_phi2 = (c.valid_mask & kCentroidPhi2Valid) != 0u;
        std::fprintf(dual_centroid_fp_,
            "%.6f %d %.17g %.17g %.17g %.17g %.17g %.17g %d %d\n",
            (double)step_at * p_.dt, c.global_id,
            valid_phi ? c.cx_phi : nan, valid_phi ? c.cy_phi : nan,
            valid_phi2 ? c.cx_phi2 : nan, valid_phi2 ? c.cy_phi2 : nan,
            c.sum_phi, c.sum_phi2, valid_phi ? 1 : 0, valid_phi2 ? 1 : 0);
    }
    ++dual_centroid_frames_;
    std::fflush(dual_centroid_fp_);
}

void Sim::close_dual_centroid() {
    if (!dual_centroid_fp_) return;
    std::fflush(dual_centroid_fp_);
    std::fclose(dual_centroid_fp_);
    dual_centroid_fp_ = nullptr;
}

bool Sim::dump_state(const std::string& path) {
    const int N = p_.num_cells;
    const int pin = (int)(steps_done_ % 2);   // the buffer holding phi^step

    std::vector<CellState> h((size_t)N);
    std::vector<uint8_t> hc((size_t)N);
    CU_CHECK(cudaMemcpy(h.data(), d_cell_, h.size() * sizeof(CellState),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(hc.data(), d_cls_, hc.size(), cudaMemcpyDeviceToHost));

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "[error] cannot write %s\n", path.c_str()); return false; }

    DumpHeader hdr{};
    hdr.magic = kDumpMagic;
    hdr.version = kDumpVersion;
    hdr.num_cells = N;
    hdr.domain_side = side_;
    hdr.tile_pitch = kTilePitch;
    hdr.num_classes = kNumClasses;
    for (int c = 0; c < kNumClasses; ++c) {
        hdr.cls_wx[c] = kClasses[c].wx;
        hdr.cls_wy[c] = kClasses[c].wy;
        hdr.cls_tx0[c] = kClasses[c].tx0;
        hdr.cls_ty0[c] = kClasses[c].ty0;
    }
    hdr.dx = p_.dx; hdr.dy = p_.dy; hdr.dt = p_.dt; hdr.t_now = time();
    hdr.lambda = p_.lambda; hdr.radius = p_.target_radius;
    hdr.kappa = p_.kappa; hdr.mu = p_.mu; hdr.xi = p_.xi; hdr.tau = p_.tau;
    hdr.v_A = p_.v_A;
    hdr.gamma_normal = p_.gamma_normal; hdr.gamma_cancer = p_.gamma_cancer;
    hdr.p_tumble = p_.p_tumble();
    hdr.step = steps_done_;
    CU_WARN(cudaMemcpy(hdr.flags, d_flags_, sizeof(hdr.flags),
                       cudaMemcpyDeviceToHost));
    std::fwrite(&hdr, sizeof(hdr), 1, f);

    std::vector<float> win;
    for (int i = 0; i < N; ++i) {
        const int cls = (int)hc[(size_t)i];
        const int wx = kClasses[cls].wx, wy = kClasses[cls].wy;
        const int tx0 = kClasses[cls].tx0, ty0 = kClasses[cls].ty0;
        const CellState& c = h[(size_t)i];

        DumpCell dc{};
        dc.global_id = c.global_id;
        dc.cls = cls;
        dc.gx0 = c.gx0; dc.gy0 = c.gy0;
        dc.wx = wx; dc.wy = wy;
        dc.bb_lo_x = c.bb_lo_x; dc.bb_hi_x = c.bb_hi_x;
        dc.bb_lo_y = c.bb_lo_y; dc.bb_hi_y = c.bb_hi_y;
        dc.gamma = c.gamma; dc.v_A = c.v_A; dc.theta = c.theta;
        dc.vx = c.vx; dc.vy = c.vy; dc.phi_max = c.phi_max;
        dc.V = c.V; dc.Cx = c.Cx; dc.Cy = c.Cy; dc.perim = c.perim;
        dc.Ix = c.Ix; dc.Iy = c.Iy;
        std::fwrite(&dc, sizeof(dc), 1, f);

        win.assign((size_t)wx * (size_t)wy, 0.0f);
        const float* src = d_phi_[pin] + (size_t)i * kTileArea
                         + (size_t)ty0 * kTilePitch + tx0;
        CU_CHECK(cudaMemcpy2D(win.data(), (size_t)wx * sizeof(float),
                              src, (size_t)kTilePitch * sizeof(float),
                              (size_t)wx * sizeof(float), (size_t)wy,
                              cudaMemcpyDeviceToHost));
        std::fwrite(win.data(), sizeof(float), win.size(), f);
    }
    std::fclose(f);
    std::printf("state dump -> %s (step %lld, phi buffer parity %d)\n",
                path.c_str(), steps_done_, pin);
    return true;
}

}  // namespace pf
