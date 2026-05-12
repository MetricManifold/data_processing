#pragma once
#include "types.cuh"
#include "kernels.cuh"
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>

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

    // ---- CUDA Graph capture for the hot step pipeline.
    // step_stream is a non-default stream so launches can be captured.
    // step_graph[parity] is the cached executable graph for the
    // "regular fast" step (polar + scatter + fast-reduce + RHS) with the
    // pool half-pointers baked in by parity. Output / rebind / scripted
    // / first-step paths fall back to direct launches on the same stream.
    cudaStream_t    step_stream            = nullptr;
    cudaGraphExec_t step_graph[2]          = {nullptr, nullptr};
    bool            step_graph_built[2]    = {false, false};
    int             parity                 = 0;
    float*          phi_A                  = nullptr;  // phi_pool half 0
    float*          phi_B                  = nullptr;  // phi_pool half 1

    // ---- Multi-GPU partitioning (single-GPU defaults are: gpus=1, rank=0,
    // device=0, cells_global = cells.num_cells, cell_offset = 0).
    //
    // In a multi-GPU run (--gpus G), one Simulation instance is created
    // per rank by the orchestrator in src/multi_gpu.cu. Each instance
    // performs the full deterministic GLOBAL cell placement (using
    // params.seed) so per-cell scalars and origins are bit-identical
    // across ranks before slicing, then keeps only its local slice
    // [cell_offset, cell_offset + cells.num_cells) inside h_cells and
    // GPU buffers. The global S(x,y) is REPLICATED on every rank and
    // kept consistent by an NCCL all-reduce sandwiched between the
    // pre-reduce and post-reduce step phases (see step_pre_reduce /
    // step_post_reduce below).
    int gpus           = 1;
    int rank           = 0;
    int device         = 0;
    int cells_global   = 0;   // total cells across the world
    int cell_offset    = 0;   // first global cell id (B0 only — kept for
                              // single-GPU and contiguous-id slicing)
    // Spatial partition along y. For G==1 these stay 0..Ny / no halo.
    // For G>1 they are set by slice_cells_to_local() before alloc_gpu().
    int slab_y_lo      = 0;
    int slab_y_hi      = 0;
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
    void finalize_init();
    void step();
    // Multi-GPU step decomposition. The orchestrator drives:
    //   for each rank g: sim[g].step_pre_reduce()
    //   ncclGroupStart(); for each g: ncclAllReduce(S); ncclGroupEnd()
    //   for each rank g: sim[g].step_post_reduce()
    // step() (single-GPU monolithic, with graph fast path) calls neither
    // of these — it remains the hot path for --gpus 1.
    void step_pre_reduce();
    void step_post_reduce();

    // ---- Migration state (multi-GPU only). Allocated by alloc_gpu when
    // gpus > 1. Used by migrate_cells() (called from the orchestrator at
    // rebind cadence) to move cells between ranks when their rebound COM
    // crosses a slab boundary. All pointers are device memory.
    int*   d_n_stay        = nullptr;     // single int
    int*   d_n_up          = nullptr;     // single int
    int*   d_n_down        = nullptr;     // single int
    int*   d_n_in_prev     = nullptr;     // single int (recv'd from prev)
    int*   d_n_in_next     = nullptr;     // single int (recv'd from next)
    int*   d_stay_idx      = nullptr;     // [capacity]
    int*   d_up_idx        = nullptr;     // [capacity]
    int*   d_down_idx      = nullptr;     // [capacity]
    void*  d_pack_up       = nullptr;     // [MAX_MIGRANTS_PER_DIR * CELL_PACK_BYTES]
    void*  d_pack_down     = nullptr;
    void*  d_pack_in_prev  = nullptr;
    void*  d_pack_in_next  = nullptr;
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
// loop with NCCL all-reduce on S, and rank-0-driven I/O. Caller owns
// nothing — full lifecycle is managed inside.
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
