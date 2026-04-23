#pragma once
#include "types.cuh"
#include "kernels.cuh"
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>

// Which SimParams fields did the user explicitly override on the CLI?
// Used on resume to decide what to keep from the checkpoint vs the flags.
struct SimOverrides {
    bool t_end = false, dt = false, v_A = false, tau = false;
    bool gamma = false, kappa = false, mu = false, xi = false, lambda = false;
    bool target_radius = false, subdomain_padding = false;
    bool save_interval = false, print_interval = false, trajectory_samples = false;
    bool seed = false, polarity_seed = false, abp = false;
};

struct Simulation {
    SimParams params;
    CellArrays cells;
    std::vector<CellHost> h_cells;
    int step_count = 0;
    double cur_time = 0.0;  // double: float32 only supports t<=2^18 before += dt stops advancing
    int cache_w = 0, cache_h = 0;
    std::string out_dir = "./output";
    FILE* traj_fp = nullptr;
    int traj_every = 0;
    bool save_final_checkpoint = true;
    int checkpoint_interval = 0;  // steps; 0 = disabled
    std::string gamma_spec;       // e.g. "0.35", "0.35:cell0", "0.35:20%"

    void init(const SimParams& p, int n_cells);
    // Resume from checkpoint. Loads SimParams, per-cell state, phi, step, time.
    // Any field in `ov` with its flag set overrides the loaded value.
    // Returns false on failure.
    bool init_from_checkpoint(const std::string& path,
                              const SimParams& cli_params,
                              const SimOverrides& ov);
    void run();
    void cleanup();

    // internal
    void place_cells(int n, double R);
    void compute_bboxes();
    void alloc_gpu();
    void upload_phi();
    void apply_gamma_spec();  // parses gamma_spec, writes per-cell gamma into h_cells
    void finalize_init();  // rng, hash, ref, initial reduce (used by both init paths)
    void step();
    void print_status();
    void write_trajectory();
    void save_checkpoint(const std::string& dir);

    size_t max_slot() const {
        double sigma = params.target_radius + 3.0 * params.lambda + 1.0;
        double pad = params.subdomain_padding * params.target_radius + 10.0;
        int half = (int)(2.0 * sigma + pad);
        int side = 2 * half + 2 * params.halo;
        side = (side + 1) & ~1;
        return (size_t)side * side;
    }

    static int domain_for(int n, double R, double rho) {
        return (int)std::ceil(std::sqrt((double)n * M_PI * R * R / rho));
    }
};
