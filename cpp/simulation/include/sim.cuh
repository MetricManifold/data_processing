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

    void init(const SimParams& p, int n_cells);
    bool init_from_checkpoint(const std::string& path,
                              const SimParams& cli_params,
                              const SimOverrides& ov);
    void run();
    void cleanup();

    // internal
    void place_cells(int n, double R);
    void compute_origins();
    void alloc_gpu();
    void upload_initial_state();
    void apply_gamma_spec();
    void apply_v_A_disorder();
    void finalize_init();
    void step();
    void print_status();
    void write_trajectory();
    void write_vtk();
    void save_checkpoint(const std::string& dir, const std::string& tag = "");

    static int domain_for(int n, double R, double rho) {
        return (int)std::ceil(std::sqrt((double)n * M_PI * R * R / rho));
    }
};
