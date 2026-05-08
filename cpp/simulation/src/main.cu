// main.cu — CLI entry point
#include "sim.cuh"
#include "multi_gpu.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(d) _mkdir(d)
#else
#include <sys/stat.h>
#define MKDIR(d) mkdir(d, 0755)
#endif

static void usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  -n <num>              Number of cells (default: 8)\n");
    printf("  -r <radius>           Target radius (default: 20)\n");
    printf("  --radius <f>          Target radius (alias for -r)\n");
    printf("  -N <size>             Domain size LxL\n");
    printf("  --confluence <f>      Packing fraction -> auto domain size\n");
    printf("  -t <time>             End time (default: 100)\n");
    printf("  --dt <step>           Time step (default: 0.01)\n");
    printf("  -dt <step>            Time step (alias for --dt)\n");
    printf("  --v-A <f>             Active motility speed\n");
    printf("  --v-A-sigma <f>       Log-normal disorder σ on v_A (fresh init only)\n");
    printf("  --tau <f>             Reorientation time (default: 10000; <=0 disables tumbling)\n");
    printf("  --gamma <spec>        Surface tension. <spec> = <f> | <f>:cell<k> | <f>:<p>%%\n");
    printf("  --kappa <f>           Interaction strength (default: 10.0)\n");
    printf("  --mu <f>              Volume constraint (default: 1.0)\n");
    printf("  --xi <f>              Friction (default: 1500)\n");
    printf("  --lambda <f>          Interface width (default: 7.0)\n");
    printf("  -l <f>                Interface width (alias for --lambda)\n");
    printf("  --subdomain-padding <f>  Adaptive-rect K: hw = ceil(2*sigma + K*R/4*sqrt(gamma_ref/gamma)) (default: 2.0)\n");
    printf("  --abp                 Use ABP instead of run-and-tumble\n");
    printf("  -o <dir>              Output directory (default: ./output)\n");
    printf("  -c <path>             Resume from checkpoint\n");
    printf("  --save-interval <n>   Save checkpoint every N steps (0=off)\n");
    printf("  --checkpoint-interval <n>  Alias for --save-interval\n");
    printf("  --save-final-checkpoint    Save final checkpoint at t_end (default: on)\n");
    printf("  --no-save-final-checkpoint Disable final checkpoint save\n");
    printf("  --print-interval <n>  Console print interval (default: 100)\n");
    printf("  --trajectory-samples <n>  Number of trajectory snapshots (default 100; 0=off)\n");
    printf("  --trajectory-interval <n> Steps between trajectory saves (alt. to --trajectory-samples)\n");
    printf("  --vtk-interval <n>    Write binary VTK composite field every N steps (0=off, default)\n");
    printf("  --live-view           Open live CUDA-OpenGL viewer (requires ENABLE_VISUALIZER build)\n");
    printf("  --live-view-tu <f>        Frame interval in time units (default: 5.0)\n");
    printf("  --live-view-interval <n>  Steps between viewer updates (overrides --live-view-tu)\n");
    printf("  --seed <n>            Placement RNG seed\n");
    printf("  --polarity-seed <n>   Polarity RNG seed\n");
    printf("  --scripted-events <f> Pre-determined tumble events for deterministic replay\n");
    printf("                        (file format: lines `t cid [old_theta] new_theta`,\n");
    printf("                         3- or 4-col; '#' header lines ignored). Disables\n");
    printf("                         the per-step PRNG tumble path entirely.\n");
    printf("  --gpus <n>            Run on N GPUs (default: 1). N>1 requires a build with\n");
    printf("                        -DENABLE_MULTI_GPU=ON (NCCL). Single-process,\n");
    printf("                        one-thread, multi-device. Cells are partitioned\n");
    printf("                        across GPUs; the global S(x,y) field is replicated\n");
    printf("                        and kept in sync via NCCL all-reduce per step.\n");
    printf("  -h, --help            Show this help\n");
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    SimParams p;
    SimOverrides ov;
    int ncells = 8;
    bool ncells_set = false;
    bool nx_set = false;          // Track explicit -N for conflict validation.
    std::string outdir = "./output";
    std::string ckpt_path;
    std::string gamma_spec;
    float confluence = -1.0f;
    bool save_final = true;
    int checkpoint_interval = 0;
    int trajectory_interval = 0;  // Alt. to --trajectory-samples; >0 enables.
    double v_A_sigma = 0.0;       // Log-normal disorder σ on v_A at fresh init.
    int vtk_interval = 0;         // Steps between binary VTK dumps; 0 = off.
    bool live_view = false;       // Open CUDA-GL live window. Requires
                                  // ENABLE_VISUALIZER build; otherwise warned + ignored.
    int  live_view_interval = 0;  // Step cadence for viewer updates. 0 = derive
                                  // from --live-view-tu after dt is known.
    bool live_view_interval_set = false;
    double live_view_tu = 5.0;    // Default frame interval in time units.
    std::string scripted_events_path;
    int gpus = 1;                 // --gpus N. >1 requires ENABLE_MULTI_GPU.

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-n") && i+1<argc) { ncells = atoi(argv[++i]); ncells_set = true; }
        else if (!strcmp(argv[i], "-r") && i+1<argc) { p.target_radius = atof(argv[++i]); ov.target_radius = true; }
        else if (!strcmp(argv[i], "--radius") && i+1<argc) { p.target_radius = atof(argv[++i]); ov.target_radius = true; }
        else if (!strcmp(argv[i], "-N") && i+1<argc) { p.Nx = atoi(argv[++i]); p.Ny = p.Nx; nx_set = true; }
        else if (!strcmp(argv[i], "--confluence") && i+1<argc) confluence = atof(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i+1<argc) { p.t_end = atof(argv[++i]); ov.t_end = true; }
        else if (!strcmp(argv[i], "--dt") && i+1<argc) { p.dt = atof(argv[++i]); ov.dt = true; }
        else if (!strcmp(argv[i], "-dt") && i+1<argc) { p.dt = atof(argv[++i]); ov.dt = true; }
        else if (!strcmp(argv[i], "--v-A") && i+1<argc) { p.v_A = atof(argv[++i]); ov.v_A = true; }
        else if (!strcmp(argv[i], "--v-A-sigma") && i+1<argc) { v_A_sigma = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--tau") && i+1<argc) { p.tau = atof(argv[++i]); ov.tau = true; }
        else if (!strcmp(argv[i], "--gamma") && i+1<argc) {
            gamma_spec = argv[++i];
            // If purely numeric (no colon, no %), also update scalar + override flag
            if (gamma_spec.find(':') == std::string::npos && gamma_spec.find('%') == std::string::npos) {
                p.gamma = atof(gamma_spec.c_str());
                ov.gamma = true;
            }
        }
        else if (!strcmp(argv[i], "--kappa") && i+1<argc) { p.kappa = atof(argv[++i]); ov.kappa = true; }
        else if (!strcmp(argv[i], "--mu") && i+1<argc) { p.mu = atof(argv[++i]); ov.mu = true; }
        else if (!strcmp(argv[i], "--xi") && i+1<argc) { p.xi = atof(argv[++i]); ov.xi = true; }
        else if (!strcmp(argv[i], "--lambda") && i+1<argc) { p.lambda = atof(argv[++i]); ov.lambda = true; }
        else if (!strcmp(argv[i], "-l") && i+1<argc) { p.lambda = atof(argv[++i]); ov.lambda = true; }
        else if (!strcmp(argv[i], "--subdomain-padding") && i+1<argc) {
            p.subdomain_padding = atof(argv[++i]); ov.subdomain_padding = true;
        }
        else if (!strcmp(argv[i], "--abp")) { p.abp = true; ov.abp = true; }
        else if (!strcmp(argv[i], "-o") && i+1<argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "-c") && i+1<argc) ckpt_path = argv[++i];
        else if (!strcmp(argv[i], "--save-interval") && i+1<argc) {
            p.save_interval = atoi(argv[++i]); ov.save_interval = true;
        }
        else if (!strcmp(argv[i], "--checkpoint-interval") && i+1<argc) {
            checkpoint_interval = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--save-final-checkpoint")) save_final = true;
        else if (!strcmp(argv[i], "--no-save-final-checkpoint")) save_final = false;
        else if (!strcmp(argv[i], "--print-interval") && i+1<argc) {
            p.print_interval = atoi(argv[++i]); ov.print_interval = true;
        }
        else if (!strcmp(argv[i], "--trajectory-samples") && i+1<argc) {
            p.trajectory_samples = atoi(argv[++i]); ov.trajectory_samples = true;
        }
        // Accept-and-translate: --trajectory-interval is converted to --trajectory-samples
        // once we know t_end and dt (after parsing finishes).
        else if (!strcmp(argv[i], "--trajectory-interval") && i+1<argc) {
            trajectory_interval = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--vtk-interval") && i+1<argc) {
            vtk_interval = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--live-view")) { live_view = true; }
        else if (!strcmp(argv[i], "--live-view-interval") && i+1<argc) {
            live_view_interval = atoi(argv[++i]);
            live_view_interval_set = true;
        }
        else if (!strcmp(argv[i], "--live-view-tu") && i+1<argc) {
            live_view_tu = atof(argv[++i]);
        }
        // TODO(observables): re-introduce a `--observables [interval=<n>]` CLI
        // surface that drives an on-GPU per-cell measurement pass (volume,
        // perimeter, shape index, stress/strain tensors, neighbor contacts)
        // and writes observables.csv. Historic flags `--use-diagnostics` and
        // `--observable-interval` are removed here; restore as a single,
        // properly-implemented `--observables` flag once the kernel is ready.
        else if (!strcmp(argv[i], "--seed") && i+1<argc) { p.seed = atoi(argv[++i]); ov.seed = true; }
        else if (!strcmp(argv[i], "--polarity-seed") && i+1<argc) {
            p.polarity_seed = atoi(argv[++i]); ov.polarity_seed = true;
        }
        else if (!strcmp(argv[i], "--scripted-events") && i+1<argc) {
            scripted_events_path = argv[++i];
        }
        else if (!strcmp(argv[i], "--gpus") && i+1<argc) {
            gpus = atoi(argv[++i]);
            if (gpus < 1) {
                fprintf(stderr, "Error: --gpus must be >= 1 (got %d)\n", gpus);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }

    // Mutually-exclusive domain specification: -N OR --confluence, not both.
    if (confluence > 0 && nx_set) {
        fprintf(stderr,
                "Error: -N and --confluence are mutually exclusive. "
                "Pick one.\n");
        return 1;
    }
    // If neither -N nor --confluence is given, fall back to confluence=0.85.
    // This keeps single-line invocations like `cell_sim -n 8 -r 20` runnable
    // without forcing the user to pick a domain size every time.
    if (confluence <= 0 && !nx_set) {
        confluence = 0.85f;
        fprintf(stderr,
                "[warn] No -N or --confluence specified; "
                "auto-setting --confluence=0.85.\n");
    }
    if (confluence > 0) {
        int L = Simulation::domain_for(ncells_set ? ncells : 8,
                                       p.target_radius, confluence);
        // Domain must be >= TILE_T to avoid tile self-wrap. For tiny
        // (n*r²) inputs the confluence formula gives L < TILE_T; bump it
        // up so the sim still runs (the actual confluence will be lower
        // than requested in that case, which is the right trade-off).
        if (L < TILE_T) L = TILE_T;
        p.Nx = L; p.Ny = L;
    }
    // Translate --trajectory-interval into --trajectory-samples now that
    // dt and t_end are known. We round up so the user gets at least one
    // sample per requested interval.
    if (trajectory_interval > 0) {
        long long total_steps = (long long)(p.t_end / p.dt + 0.5);
        int samples = (int)((total_steps + trajectory_interval - 1) /
                            trajectory_interval);
        if (samples < 1) samples = 1;
        p.trajectory_samples = samples;
        ov.trajectory_samples = true;
    }

    MKDIR(outdir.c_str());

    // ---- Multi-GPU dispatch ----
    // --gpus 1 always falls through to the single-GPU path below
    // (preserves the captured-graph fast path and is bit-identical to a
    // no-NCCL build). --gpus > 1 routes to the orchestrator in sim.cu;
    // requires ENABLE_MULTI_GPU=ON at build time.
    if (gpus > 1) {
        if (!mg_available()) {
            fprintf(stderr,
                "Error: --gpus %d requested but this binary was built "
                "without ENABLE_MULTI_GPU. Rebuild with cmake "
                "-DENABLE_MULTI_GPU=ON to enable multi-GPU.\n", gpus);
            return 1;
        }
        if (!scripted_events_path.empty()) {
            fprintf(stderr,
                "Error: --scripted-events is not yet supported on --gpus>1.\n");
            return 1;
        }
        if (live_view) {
            fprintf(stderr,
                "Error: --live-view is incompatible with --gpus>1.\n");
            return 1;
        }
        MultiGpuRunArgs args;
        args.params              = p;
        args.ov                  = ov;
        args.ncells_global       = ncells;
        args.gpus                = gpus;
        args.outdir              = outdir;
        args.ckpt_path           = ckpt_path;
        args.gamma_spec          = gamma_spec;
        args.v_A_sigma           = v_A_sigma;
        args.checkpoint_interval = checkpoint_interval;
        args.vtk_interval        = vtk_interval;
        args.save_final          = save_final;
        return run_multi_gpu(args);
    }

    Simulation sim;
    sim.out_dir = outdir;
    sim.save_final_checkpoint = save_final;
    sim.checkpoint_interval = checkpoint_interval;
    sim.gamma_spec = gamma_spec;
    sim.v_A_sigma = v_A_sigma;
    sim.vtk_interval = vtk_interval;
    sim.live_view = live_view;
    if (!live_view_interval_set) {
        // Derive from --live-view-tu (default 5 tau): one frame per 5 tu.
        live_view_interval = (int)(live_view_tu / p.dt + 0.5);
        if (live_view_interval < 1) live_view_interval = 1;
    }
    sim.live_view_interval = (live_view_interval > 0) ? live_view_interval : 1;
#ifndef ENABLE_VISUALIZER
    if (live_view) {
        fprintf(stderr, "[warn] --live-view ignored: binary not built with "
                        "ENABLE_VISUALIZER. Rebuild with cmake -DENABLE_VISUALIZER=ON.\n");
        sim.live_view = false;
    }
#endif

    if (!ckpt_path.empty()) {
        if (!sim.init_from_checkpoint(ckpt_path, p, ov)) return 1;
    } else {
        printf("=== Phase-Field Cell Simulation (v2) ===\n");
        printf("Cells: %d, R=%.1f, Domain: %dx%d\n", ncells, p.target_radius, p.Nx, p.Ny);
        printf("gamma=%.2f, kappa=%.2f, mu=%.2f, lambda=%.2f\n",
               p.gamma, p.kappa, p.mu, p.lambda);
        printf("v_A=%.4f, xi=%.1f, tau=%.1f, dt=%.4f, t_end=%.1f\n",
               p.v_A, p.xi, p.tau, p.dt, p.t_end);
        sim.init(p, ncells);
    }

    // ---- Optional: load scripted (pre-determined) tumble events.
    // File format mirrors cpu_ref's --events output:
    //   `# t cid [old_theta] new_theta`  (3- or 4-col; '#' lines ignored).
    // Each event's `t` is converted to the step_count value at which
    // Simulation::step() should fire it: step_count_at_start = round(t/dt) - 1.
    if (!scripted_events_path.empty()) {
        std::ifstream f(scripted_events_path);
        if (!f) {
            fprintf(stderr, "Error: cannot open --scripted-events file '%s'\n",
                    scripted_events_path.c_str());
            return 1;
        }
        std::string line;
        const int N = sim.cells.num_cells;
        const double dt = sim.params.dt;
        // start_t = current cur_time at end of init (== ckpt.t for resume,
        // 0 for fresh). Events at t <= start_t are rejected.
        const double start_t = sim.cur_time;
        const int    start_step = sim.step_count;
        struct Evt { int step_at_start; int cid; float theta; };
        std::vector<Evt> evs;
        int lineno = 0;
        while (std::getline(f, line)) {
            ++lineno;
            // strip leading whitespace
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
                    scripted_events_path.c_str(), lineno, toks.size());
                return 1;
            }
            if (cid < 0 || cid >= N) {
                fprintf(stderr,
                    "Error: %s line %d: cid %d out of range (n_cells=%d)\n",
                    scripted_events_path.c_str(), lineno, cid, N);
                return 1;
            }
            if (t <= start_t) {
                fprintf(stderr,
                    "Error: %s line %d: t=%.6f <= start_t=%.6f\n",
                    scripted_events_path.c_str(), lineno, t, start_t);
                return 1;
            }
            int step_idx = (int)std::llround((t - start_t) / dt);
            int step_at_start = start_step + step_idx - 1;
            evs.push_back({step_at_start, cid, (float)new_theta});
        }
        std::sort(evs.begin(), evs.end(), [](const Evt& a, const Evt& b) {
            if (a.step_at_start != b.step_at_start) return a.step_at_start < b.step_at_start;
            return a.cid < b.cid;
        });
        // Hand off to sim.
        sim.scripted_active = !evs.empty();
        sim.scripted_cursor = 0;
        sim.h_scripted_step.reserve(evs.size());
        sim.h_scripted_cid.reserve(evs.size());
        sim.h_scripted_theta.reserve(evs.size());
        for (const auto& e : evs) {
            sim.h_scripted_step.push_back(e.step_at_start);
            sim.h_scripted_cid.push_back(e.cid);
            sim.h_scripted_theta.push_back(e.theta);
        }
        if (!evs.empty()) {
            cudaMalloc(&sim.d_scripted_cid,   evs.size() * sizeof(int));
            cudaMalloc(&sim.d_scripted_theta, evs.size() * sizeof(float));
            cudaMemcpy(sim.d_scripted_cid,   sim.h_scripted_cid.data(),
                       evs.size() * sizeof(int),   cudaMemcpyHostToDevice);
            cudaMemcpy(sim.d_scripted_theta, sim.h_scripted_theta.data(),
                       evs.size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        printf("[scripted] %zu events loaded from %s (PRNG tumble path disabled)\n",
               evs.size(), scripted_events_path.c_str());
    }

    sim.run();
    sim.cleanup();
    return 0;
}
