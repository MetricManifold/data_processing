#pragma once
// ===========================================================================
// FUSE-1R host-side simulation object: allocation, initialisation, the
// per-step launch sequence (one kernel per step, optionally as a replayed
// 6-step CUDA graph), L2 residency policy, diagnostics and state dump.
// ===========================================================================

#include "checkpoint.cuh"
#include "kernels.cuh"
#include "params.cuh"

#include <csignal>
#include <cstdio>
#include <string>
#include <vector>

namespace pf {

// lcm(2 phi parities, 3 S rotation slots). A captured body of this length
// replays forever with every pointer and parity baked per node.
//
// This is a property of the ARGUMENT rotation, not of the launch count, so it
// is 6 for both paths: the split path just puts 12 kernel nodes in the captured
// body instead of 6. Graph capture is fully supported for --split.
constexpr int kGraphBody = 6;
constexpr int kMortonEvery = kGraphBody;   // aligned to the graph body

struct RunOptions {
    bool use_graph = true;
    bool morton = false;
    bool l2_persist = true;
    bool strict = false;          // run k_verify every step
    bool split = false;           // --split: k_step_rhs + k_step_post per step
    int  bench_steps = 0;         // >0: timed benchmark, no I/O
    // Trajectory sampling is INDEPENDENT of --print-interval. Tying the two
    // together meant a large print interval silently produced a single-frame
    // trajectory, which is useless for MSD. Legacy names and default (100).
    int  traj_samples  = 100;     // evenly spaced samples across the run
    long long traj_interval = 0;  // steps between samples; overrides traj_samples
    std::string out_path;
    std::string dump_path;

    // ---- checkpointing -----------------------------------------------------
    // Empty ckpt_dir disables checkpointing entirely, including the final one.
    // Both cadences are THRESHOLDS, not divisibility tests, for the same reason
    // the verify cadence is: steps_done_ advances in jumps of kGraphBody on a
    // graph replay and steps straight over an exact multiple.
    std::string ckpt_dir;
    long long ckpt_interval = 0;  // steps between rolling <dir>/checkpoint.bin
    long long save_interval = 0;  // steps between tagged checkpoint_%08d.bin
    bool      final_checkpoint = true;   // on normal exit AND on SIGTERM
};

class Sim {
public:
    Sim() = default;
    ~Sim();
    Sim(const Sim&) = delete;
    Sim& operator=(const Sim&) = delete;

    bool init(const SimParams& p, const RunOptions& opt, int device);
    // Resume. `p` is the checkpoint's SimParams with the CLI overrides already
    // applied (main.cu owns that policy); `d` supplies the microstate. Every
    // per-cell scalar in `d.cells` is taken as-is, so the CLI > sidecar >
    // params precedence is likewise resolved before we get here.
    bool init_from_checkpoint(const SimParams& p, const CheckpointData& d,
                              const RunOptions& opt, int device);
    void run();
    bool bench(int steps, double* ms_per_step);
    bool dump_state(const std::string& path);
    // Gather the current state and write it to every path given. Public so a
    // caller can force one outside the cadence; run() drives the normal case.
    bool save_checkpoint(const std::vector<std::string>& paths);
    void report_flags() const;

    // Set from a SIGTERM/SIGINT handler. The step loop polls it and exits
    // cleanly, so a walltime kill yields a short trajectory rather than none.
    // sig_atomic_t + volatile is the only thing a signal handler may touch.
    static volatile std::sig_atomic_t s_terminate;
    static void request_termination(int) { s_terminate = 1; }
    bool verify(double* max_rel_V, float* max_outside, uint32_t* max_S);

    long long step() const { return steps_done_; }
    double    time() const { return (double)steps_done_ * p_.dt; }
    int       side() const { return side_; }

private:
    // init() and init_from_checkpoint() differ only in where the initial phi
    // and per-cell scalars come from. Everything before that (device query,
    // allocation, stream, zeroing) and everything after it (shared-memory
    // opt-in, occupancy report, L2 carve-out, Morton, graph capture) is shared.
    bool alloc_device(const SimParams& p, const RunOptions& opt, int device);
    bool configure_and_capture();
    // Rolling / tagged / final paths that are due at the current step.
    std::vector<std::string> checkpoint_paths(bool rolling, bool tagged) const;

    StepArgs args_for_slot(int slot) const;
    void     l2_window_for_slot(int slot, const void** base, size_t* bytes,
                                float* hit) const;
    void     launch_one(int slot);
    void     print_path_report() const;
    bool     build_graph();
    void     seed_positions(std::vector<float>& cx, std::vector<float>& cy,
                            std::vector<float>& gam, std::vector<float>& va,
                            std::vector<int32_t>& gid);
    void     print_line();
    // True when a flag is set that means the remaining steps cannot produce a
    // valid result, so the run should stop NOW rather than at t_end. Only
    // FLAG_CLASS_UNSUPPORTED qualifies: the other alarms are counters whose
    // magnitude is the point, and stopping would destroy the measurement.
    bool     fatal_flag_set();

    SimParams  p_{};
    RunOptions opt_{};
    int device_ = 0;
    int grid_   = 132;
    int side_   = 0;      // L
    int pitch_  = 0;      // P

    float*      d_phi_[2] = {nullptr, nullptr};
    uint32_t*   d_S_      = nullptr;     // 3 * pitch_ * side_ uint32
    size_t      s_buf_words_ = 0;
    CellState*  d_cell_   = nullptr;
    uint8_t*    d_cls_    = nullptr;
    uint32_t*   d_perm_   = nullptr;
    unsigned long long* d_cursor_ = nullptr;   // 2 slots
    unsigned long long* d_step_   = nullptr;   // 2 slots
    uint32_t*   d_flags_  = nullptr;
    double*     d_vchk_   = nullptr;
    float*      d_ochk_   = nullptr;
    uint32_t*   d_smax_   = nullptr;
    TrajPackedCell* h_traj_ = nullptr;         // pinned
    TrajPackedCell* d_traj_ = nullptr;         // device alias of h_traj_

    cudaStream_t stream_ = nullptr;
    cudaGraph_t  graph_  = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_ready_ = false;

    long long steps_done_ = 0;
    size_t l2_persist_max_ = 0;
    size_t l2_window_max_  = 0;

    // Trajectory is streamed to disk as it is sampled, NOT accumulated. Holding
    // it in RAM and writing after run() returns meant a walltime kill produced a
    // zero-byte file -- the whole run lost, with no partial result.
    std::FILE* traj_fp_ = nullptr;
    bool       traj_header_written_ = false;
    long long  traj_frames_ = 0;
    bool open_trajectory(const std::string& path);
    void append_trajectory_frame(long long step_at);
    void close_trajectory();
};

}  // namespace pf
