#pragma once
#include "types.cuh"
#include "kernels.cuh"
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

// Which SimParams fields did the user explicitly override on the CLI?
// On resume from checkpoint, set fields override the loaded values.
struct SimOverrides {
    bool t_end = false, dt = false, v_A = false, tau = false;
    bool gamma = false, kappa = false, mu = false, xi = false, lambda = false;
    bool target_radius = false, subdomain_padding = false;
    bool save_interval = false, print_interval = false, trajectory_samples = false;
    bool seed = false, polarity_seed = false, abp = false;
};

// ---------------------------------------------------------------------------
// PartitionLayout — multi-GPU slab geometry (single source of truth).
//
// Built once from (gpus, rank, Ny) via PartitionLayout::for_rank. Holds the
// y-slab bounds plus the halo geometry of the global S field on this rank.
// Two formulas used to be hand-duplicated in slice_cells_to_local and
// init_from_checkpoint; the same numbers were also mirrored on CellArrays
// via S_y_lo/S_halo_h/S_ext_height. This struct collapses both paths to
// one constructor and one assignment site (alloc_gpu).
//
//   for_rank(gpus=1, rank=0, Ny): trivial single-GPU (whole grid, no halo).
//   for_rank(gpus=G, rank=g, Ny): rank g's slab [floor(g*Ny/G), floor((g+1)*Ny/G))
//                                  with HALO_H rows on each side.
//
// Boundaries use floor so the partition is exact and reproducible across
// invocations; rounding the same way in two places was the legacy bug
// risk the layout eliminates.
// ---------------------------------------------------------------------------
struct PartitionLayout {
    int Ny           = 0;   // global domain height (>= slab_y_hi)
    int slab_y_lo    = 0;   // rank-local slab covers global rows [slab_y_lo, slab_y_hi)
    int slab_y_hi    = 0;
    int S_halo_h     = 0;   // halo rows on each side of the slab for global S
    int S_ext_height = 0;   // (slab_y_hi - slab_y_lo) + 2*S_halo_h

    static PartitionLayout for_rank(int gpus, int rank, int Ny, int halo_h);
    int slab_height() const { return slab_y_hi - slab_y_lo; }
};

// ---------------------------------------------------------------------------
// sim_v3 Simulation — same external API as sim_v2 (CLI surface, checkpoint
// I/O, trajectory format) but a completely different internal architecture:
// fixed-T unified phi pool + global S field + COM rebind, no neighbour list.
// ---------------------------------------------------------------------------
struct Simulation {
    SimParams params;
    CellArrays cells;
    std::vector<CellHost> h_cells;
    int step_count = 0;
    double cur_time = 0.0;        // f64: f32 stops advancing past 2^18 * dt
    std::string out_dir = "./output";
    FILE* traj_fp = nullptr;
    int traj_every = 0;
    int traj_flush_counter = 0;
    // Async trajectory writer:
    //   producer (step thread) launches k_pack_traj into a device buffer,
    //   issues one cudaMemcpyAsync to a pinned host slot in `traj_ring`,
    //   records `ready_event` on step_stream, pushes the slot onto
    //   `traj_writer_queue`. Writer thread pops, cudaEventSynchronize,
    //   formats fprintf, returns the slot to `traj_writer_free`. Producer
    //   never calls cudaStreamSynchronize.
    struct TrajectorySnapshot {
        double time = 0.0;
        int step = 0;
        int n = 0;
        TrajPackedCell* host_packed = nullptr;  // pinned, capacity slots
        cudaEvent_t     ready_event = nullptr;
        std::vector<int> global_id;
    };
    TrajPackedCell* d_traj_pack = nullptr;          // device-side pack buffer
    int             traj_ring_capacity = 0;         // = cells.capacity at init
    std::vector<TrajectorySnapshot> traj_ring;      // owns pinned bufs + events
    std::thread traj_writer_thread;
    std::mutex traj_writer_mutex;
    std::condition_variable traj_writer_cv;
    std::deque<TrajectorySnapshot*> traj_writer_queue;  // ready-to-write slots
    std::deque<TrajectorySnapshot*> traj_writer_free;   // available slots
    bool traj_ring_init = false;
    bool traj_writer_started = false;
    bool traj_writer_stop = false;
    bool save_final_checkpoint = true;
    int checkpoint_interval = 0;  // steps; 0 = disabled
    std::string gamma_spec;       // e.g. "0.35", "0.35:cell0", "0.35:20%"
    // Log-normal disorder σ on v_A, applied at fresh init only. Per-cell
    // values are persisted in the VA_A checkpoint sidecar.
    double v_A_sigma = 0.0;
    // Steps between binary VTK composite-field dumps. 0 = disabled (default).
    int vtk_interval = 0;

    // Live CUDA-OpenGL viewer. Only honoured when the binary was compiled
    // with -DENABLE_VISUALIZER=ON. Updates the window every
    // `live_view_interval` steps (default: every step). Closing the window
    // or pressing ESC tears down the viewer and the sim continues headless.
    bool live_view = false;
    int  live_view_interval = 1;

    // ---- Scripted (pre-determined) tumble events for deterministic replay.
    // When `scripted_active` is true, the per-step PRNG-driven polarity
    // update is skipped; instead the events listed here fire at the
    // matching step_count value. Sorted ascending by step_count.
    bool scripted_active = false;
    std::vector<int>   h_scripted_step;   // length = total events
    std::vector<int>   h_scripted_cid;
    std::vector<float> h_scripted_theta;
    int*   d_scripted_cid   = nullptr;    // device mirror of h_scripted_cid
    float* d_scripted_theta = nullptr;
    int    scripted_cursor  = 0;          // first unprocessed event idx

    // Set true when init_from_checkpoint successfully restored RNG state
    // from a checkpoint sidecar; finalize_init() then skips launch_rng_init
    // to preserve the random-stream continuity across resume.
    bool   rng_restored_from_ckpt = false;

    // Load scripted events from `path` (cpu_ref --events format) and
    // populate scripted_active / cursor / host vectors / device mirrors.
    // Must be called AFTER init() or init_from_checkpoint(), since events
    // are validated against the current cell count and `start_t`.
    // Returns false (and prints to stderr) on parse error.
    bool load_scripted_events(const std::string& path);

    // ---- CUDA Graph capture for the hot step pipeline.
    // step_stream is a non-default stream so launches can be captured.
    // step_graph[parity] is the cached executable graph for the single-GPU
    // "regular fast" step (polar + scatter + fast-reduce + RHS) with the
    // pool half-pointers baked in by parity.
    //
    // mg_step_graph[parity] is the analogous cache for the multi-GPU fast
    // step: it captures polar + scatter + (NCCL halo Send/Recv pairs) +
    // halo_add + zero + fast-reduce + RHS in one launch. NCCL 2.18+
    // supports kernel capture, and our build (2.29.7) is well above that.
    // Both graphs are invalidated and rebuilt at every migration round
    // because pointers and per-cell counts may shift.
    //
    // Output / rebind / scripted / first-step paths fall back to direct
    // launches on the same stream.
    cudaStream_t    step_stream            = nullptr;
    cudaGraphExec_t step_graph[2]          = {nullptr, nullptr};
    bool            step_graph_built[2]    = {false, false};
    cudaGraphExec_t mg_step_graph[2]       = {nullptr, nullptr};
    bool            mg_step_graph_built[2] = {false, false};
    int             parity                 = 0;
    float*          phi_A                  = nullptr;  // phi_pool half 0
    float*          phi_B                  = nullptr;  // phi_pool half 1

    // Owned invariant: phi_in/phi_out aliases inside CellArrays must
    // always match the current parity (0 -> phi_in=A, phi_out=B; 1 -> swap).
    // Call `sync_pool_to_parity()` after setting `parity` explicitly,
    // or `flip_parity()` to advance one step. Five sites used to inline
    // the 3-line ritual; the method enforces the invariant in one place.
    void sync_pool_to_parity() {
        cells.phi_in  = (parity == 0) ? phi_A : phi_B;
        cells.phi_out = (parity == 0) ? phi_B : phi_A;
    }
    void flip_parity() {
        parity ^= 1;
        sync_pool_to_parity();
    }

    // Flags for what side-effects fire on the next step. Used by step()
    // and step_post_reduce — the predicate was inlined and duplicated;
    // one helper now keeps the cadence rules in a single place.
    struct StepFlags {
        bool will_rebind;
        bool will_traj;
        bool will_save;
        bool will_ckpt;
        bool will_vtk;
        bool need_full_red;
    };
    StepFlags compute_step_flags(int next_step) const;

    // Advance the scripted-events cursor through events whose step matches
    // step_count and launch them as a batch. No-op when scripted_active
    // is false. Called from step() (slow path) and step_pre_reduce().
    void apply_scripted_events_for_step();

    // ---- Multi-GPU partitioning (single-GPU defaults are: gpus=1, rank=0,
    // device=0, cells_global = cells.num_cells, cell_offset = 0).
    //
    // In a multi-GPU run (--gpus G), one Simulation instance is created
    // per rank by the orchestrator in src/multi_gpu.cu. Each instance
    // performs the full deterministic GLOBAL cell placement (using
    // params.seed) so per-cell scalars and origins are bit-identical
    // across ranks before slicing, then keeps only its local slice
    // [cell_offset, cell_offset + cells.num_cells) inside h_cells and
    // GPU buffers. The global S(x,y) is SLAB-DECOMPOSED across ranks:
    // each rank owns a y-stripe plus 2*HALO_H halo on each side. Halo
    // strips are exchanged + summed between neighbours via ncclSend/Recv
    // pairs in step_pre_reduce/step_post_reduce. No full-S all-reduce.
    int gpus           = 1;
    int rank           = 0;
    int device         = 0;
    int cells_global   = 0;   // total cells across the world
    int cell_offset    = 0;   // first global cell id (B0 only — kept for
                              // single-GPU and contiguous-id slicing)
    // Slab geometry. Single source of truth: built once from (gpus, rank, Ny)
    // and applied to CellArrays.S_* by alloc_gpu. Anyone wanting the slab
    // bounds should read from `layout`, NOT recompute the formula.
    //
    // ┌────────────────────┐
    // │  PartitionLayout   │   ← built by Simulation::build_layout()
    // │  (gpus, rank, Ny,  │      from (gpus, rank, params.Ny)
    // │   halo, slab_y_lo, │
    // │   slab_y_hi, S_*)  │
    // └─────────┬──────────┘
    //           │ alloc_gpu copies S_* fields onto CellArrays
    //           ▼
    // ┌─────────────────────┐
    // │     CellArrays      │   ← read-only view used by kernels
    // │  (S_y_lo, S_halo_h, │
    // │   S_ext_height)     │
    // └─────────────────────┘
    PartitionLayout layout;
    // Build `layout` from current (gpus, rank, params.Ny). Single
    // formula; called from both slice_cells_to_local (fresh init) and
    // init_from_checkpoint (resume). A v8 multi-rank resume hands in
    // the file-stored num_ranks/rank_id which must equal (gpus, rank);
    // mismatch is an error caught by the caller.
    void build_layout();
    // Spatial partition along y. For G==1 these stay 0..Ny / no halo.
    // For G>1 they are set by slice_cells_to_local() before alloc_gpu().
    // Kept as accessor-shaped aliases into `layout` for the small number
    // of callers that still touch them directly.
    int slab_y_lo()    const { return layout.slab_y_lo; }
    int slab_y_hi()    const { return layout.slab_y_hi; }
    // Global cell id of each local cell, length == h_cells.size(). For
    // G=1 this is just [0, n_cells). For G>1 it is the spatial-partition
    // permutation of [0, n_global) and is NOT contiguous. Used for
    // trajectory/checkpoint output and for cell migration tracking.
    std::vector<int> h_global_id;

    void init(const SimParams& p, int n_cells);
    bool init_from_checkpoint(const std::string& path,
                              const SimParams& cli_params,
                              const SimOverrides& ov);
    void run();
    void cleanup();

    // internal
    void place_cells(int n, double R);
    // After place_cells + apply_gamma_spec + apply_v_A_disorder have run
    // on the GLOBAL cell vector (h_cells), trim h_cells to this rank's
    // slice [cell_offset, cell_offset + count). No-op when gpus <= 1.
    void slice_cells_to_local();
    void compute_origins();
    void alloc_gpu();
    void configure_l2_persistence();
    void upload_initial_state();
    void apply_gamma_spec();
    void apply_v_A_disorder();
    // finalize_init = setup_step_stream + seed_rng_if_fresh
    //                + compute_initial_velocities + cudaDeviceSynchronize.
    // The three sub-steps were inlined; splitting them keeps each
    // function single-purpose. setup_step_stream is one-time infra
    // (created once per Simulation lifetime, even across re-init);
    // the other two are state that may need to re-run after migration.
    void finalize_init();
    void setup_step_stream();
    void seed_rng_if_fresh();
    void compute_initial_velocities();
    void step();
    // Multi-GPU step decomposition. The orchestrator drives:
    //   for each rank g: sim[g].step_pre_reduce()       // polar + scatter_S
    //   ncclGroupStart(); for each g: halo Send/Recv pairs; ncclGroupEnd()
    //   for each rank g: launch_halo_add_pair (fold neighbour bands in)
    //   for each rank g: sim[g].step_post_reduce()      // evolve [+ rebind]
    // step() (single-GPU monolithic, with graph fast path) calls neither
    // of these — it remains the hot path for --gpus 1.
    void step_pre_reduce();
    void step_post_reduce();

    // ---- Multi-GPU per-step pieces, factored out so the orchestrator can
    // either issue them directly (slow path) or capture them into a CUDA
    // graph (fast path).
    //
    // launch_halo_exchange enqueues the per-rank halo NCCL pairs plus the
    // two halo_add kernels onto step_stream. Handles G=2 and G>=3 routing.
    // All NCCL calls happen inside an mg_group_start/end pair so they are
    // capturable into the surrounding cudaStreamBeginCapture region.
    void launch_halo_exchange(struct MgWorld& world, int my_rank,
                              int prev_rank, int next_rank,
                              float* my_top_band, float* my_bot_band,
                              float* halo_top_recv, float* halo_bot_recv,
                              size_t halo_band_floats);

    // Returns true if the next step is eligible for the multi-GPU graph
    // fast path (no rebind, no full reduce, no scripted events). The
    // orchestrator decides which path to take before issuing the step.
    bool mg_step_is_fast_path() const;

    // Drop any cached mg_step_graph[]. Called after migration because
    // pointers / per-cell counts may have shifted.
    void invalidate_mg_step_graph();

    // Rebuild the deterministic scatter tile schedule. Pulls origin/rect
    // from device, computes the per-tile cell-overlap list with periodic
    // wrap, sorts each tile's entries by ascending cell id, and uploads
    // the CSR (d_scatter_tile_off, d_scatter_tile_entries) plus tile
    // counts. Called after init, after every k_rebind, and after
    // multi-GPU migration. Also invalidates step_graph[] because the
    // scatter kernel grid shape may have changed.
    void rebuild_scatter_schedule();

    // Issue the multi-GPU fast-step kernel sequence (polar + scatter +
    // halo + zero + fast-reduce + RHS) onto step_stream. Used both as
    // the inner work for graph capture and as the slow-path equivalent.
    void launch_mg_fast_step_kernels(struct MgWorld& world, int my_rank,
                                     int prev_rank, int next_rank,
                                     float* my_top_band, float* my_bot_band,
                                     float* halo_top_recv, float* halo_bot_recv,
                                     size_t halo_band_floats);

    // ---- Migration state (multi-GPU only). Allocated by alloc_gpu when
    // gpus > 1. Used by migrate_cells() (called from the orchestrator at
    // rebind cadence) to move cells between ranks when their rebound COM
    // crosses a slab boundary. All pointers are device memory.
    // Migration counters live in one contiguous device buffer so the
    // host can download all 5 with a single cudaMemcpyAsync per phase.
    // d_n_* aliases point into d_mig_counts; do not free them separately.
    int*   d_mig_counts    = nullptr;     // [5] = {stay, up, down, in_prev, in_next}
    int*   d_n_stay        = nullptr;     // alias into d_mig_counts[0]
    int*   d_n_up          = nullptr;     // alias into d_mig_counts[1]
    int*   d_n_down        = nullptr;     // alias into d_mig_counts[2]
    int*   d_n_in_prev     = nullptr;     // alias into d_mig_counts[3]
    int*   d_n_in_next     = nullptr;     // alias into d_mig_counts[4]
    int*   d_stay_idx      = nullptr;     // [capacity]
    int*   d_up_idx        = nullptr;     // [capacity]
    int*   d_down_idx      = nullptr;     // [capacity]
    void*  d_pack_up       = nullptr;     // [max_migrants_per_dir * CELL_PACK_BYTES]
    void*  d_pack_down     = nullptr;
    void*  d_pack_in_prev  = nullptr;
    void*  d_pack_in_next  = nullptr;
    // Per-direction migration capacity. Sized in alloc_gpu as
    // max(MAX_MIGRANTS_DEFAULT, capacity/4) so a full boundary "row"
    // can always migrate at once even at large N or G. The four pack
    // buffers above are each this many slots; the host classify-check
    // bails fatally on overflow with a clear "raise this and rebuild"
    // message.
    int    max_migrants_per_dir = 0;
    // Scratch arrays for compaction. Same layout as the corresponding
    // CellArrays fields, allocated to capacity. Used to gather stays via
    // a kernel, then we swap pointers with the originals.
    int*   d_origin_scratch     = nullptr;  // [2*capacity]
    int*   d_rect_scratch       = nullptr;  // [4*capacity]
    float* d_gamma_scratch      = nullptr;
    float* d_v_A_scratch        = nullptr;
    float* d_tgt_R_scratch      = nullptr;
    float* d_polar_theta_scratch = nullptr;
    float* d_polar_x_scratch    = nullptr;
    float* d_polar_y_scratch    = nullptr;
    void*  d_rng_scratch        = nullptr;  // curandState array

    // Per-cell float-state registry.
    //
    // Every per-cell float scalar that is BOTH state (carries across
    // steps, not a per-step reduction output) AND uniformly-shaped
    // (one f32 per cell, sized to capacity) lives here. The registry
    // declares each field once and the lifecycle code (alloc, free,
    // migration scratch, swap, sidecar save/load) iterates it.
    //
    // To add a new per-cell float field:
    //   1. Add the device pointer to `CellArrays` (types.cuh).
    //   2. Add a scratch pointer above (if it needs migration support).
    //   3. Add one row to `per_cell_float_state()` (sim.cu).
    // No other site needs editing. Adding the previous 6 fields took
    // ~10 site edits each; now it takes 1.
    //
    // Notes:
    //   - sidecar_magic == 0  -> field is not serialized (derived state,
    //                            e.g. polar_x/polar_y rebuilt from theta).
    //   - scratch_ptr == nullptr -> field has no migration scratch.
    //                               Currently every registered field has
    //                               one, but the API allows G=1-only
    //                               fields without the cost.
    struct PerCellField {
        const char* name;
        uint32_t    sidecar_magic;   // ckpt::MAGIC_* or 0
        float**     dev_ptr;         // &CellArrays.<field>
        float**     scratch_ptr;     // &Simulation.<field>_scratch, or nullptr
    };
    std::vector<PerCellField> per_cell_float_state();
    // Per-cell global ids in device memory. Two persistent buffers sized to
    // capacity (allocated once in alloc_gpu, freed in cleanup). Replaces
    // per-migration cudaMallocAsync/Free which the nsys profile showed
    // accounted for ~20% of host API time on the multi-GPU path.
    int*   d_gid_src           = nullptr;  // [capacity] — gid for current cell layout
    int*   d_gid_arr           = nullptr;  // [capacity] — gid scratch for arrivals

    // Migrate cells whose rebound COM crossed a slab boundary. Called
    // from run_multi_gpu's main thread between barrier sync points, and
    // ONLY on rebind boundaries (step_count % REBIND_EVERY == 0). For
    // gpus == 1 this is a no-op.
    //
    // NOTE: this function issues NCCL calls and so must be invoked while
    // holding the per-rank stream + comm; it expects the caller to do
    // appropriate synchronisation around it. Implemented in sim.cu under
    // ENABLE_MULTI_GPU.
    int migrate_cells(struct MgWorld& world, int rank);
    void print_status();
    void write_trajectory();
    void start_trajectory_writer();
    void finish_trajectory_writer();
    void trajectory_writer_loop();
    void init_trajectory_ring();
    void free_trajectory_ring();
    void write_trajectory_snapshot(const TrajectorySnapshot& snap);
    void write_vtk();
    void save_checkpoint(const std::string& dir, const std::string& tag = "");

    static int domain_for(int n, double R, double rho) {
        return (int)std::ceil(std::sqrt((double)n * M_PI * R * R / rho));
    }
};

// ---------------------------------------------------------------------------
// Multi-GPU orchestrator. Defined in src/multi_gpu.cu when ENABLE_MULTI_GPU
// is ON. Returns 0 on success. Handles cudaSetDevice, NCCL world setup,
// per-rank Simulation init (fresh or from checkpoint), the lockstep step
// loop with slab halo exchange on S, periodic cell migration, and
// rank-0-driven I/O. Caller owns nothing — full lifecycle is managed inside.
//
// When ENABLE_MULTI_GPU is OFF this function is not defined; main.cu
// guards the call with mg_available() and never reaches it.
// ---------------------------------------------------------------------------
struct MultiGpuRunArgs {
    SimParams     params;
    SimOverrides  ov;
    int           ncells_global  = 0;
    int           gpus           = 1;
    std::string   outdir         = "./output";
    std::string   ckpt_path;          // empty -> fresh init
    std::string   gamma_spec;
    double        v_A_sigma      = 0.0;
    int           checkpoint_interval = 0;
    int           vtk_interval        = 0;
    bool          save_final          = true;
};
int run_multi_gpu(const MultiGpuRunArgs& args);

// ---------------------------------------------------------------------------
// Cooperative termination. main.cu installs SIGTERM/SIGINT handlers that
// call request_termination(); the step loop in Simulation::run() (and the
// multi-GPU orchestrator) polls termination_requested() once per step and
// breaks out before the next step if set. This lets the normal cleanup
// path run on signal: trajectory writer drains, final checkpoint writes,
// then the process exits. Without this, SIGTERM kills mid-fprintf and
// trajectory.txt is truncated mid-line.
void request_termination();
bool termination_requested();

