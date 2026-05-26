// main.cu — CLI entry point
#include "sim.cuh"
#include "multi_gpu.cuh"
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <csignal>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(d) _mkdir(d)
#else
#include <sys/stat.h>
#define MKDIR(d) mkdir(d, 0755)
#endif

static bool cli_bad_value(const char* flag, const char* value, const char* reason) {
    fprintf(stderr, "[cli] bad value for %s: '%s'", flag, value ? value : "");
    if (reason && reason[0]) fprintf(stderr, " (%s)", reason);
    fprintf(stderr, "\n");
    return false;
}

static bool parse_int_range(const char* flag, const char* value,
                            long min_value, long max_value, int& out) {
    if (!value || !*value) return cli_bad_value(flag, value, "expected integer");
    char* end = nullptr;
    errno = 0;
    long parsed = std::strtol(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0') {
        return cli_bad_value(flag, value, "expected integer");
    }
    if (parsed < min_value || parsed > max_value) {
        return cli_bad_value(flag, value, "out of range");
    }
    out = (int)parsed;
    return true;
}

static bool parse_uint_value(const char* flag, const char* value, unsigned int& out) {
    if (!value || !*value) return cli_bad_value(flag, value, "expected unsigned integer");
    if (value[0] == '-') return cli_bad_value(flag, value, "must be >= 0");
    char* end = nullptr;
    errno = 0;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' || parsed > UINT_MAX) {
        return cli_bad_value(flag, value, "expected unsigned integer");
    }
    out = (unsigned int)parsed;
    return true;
}

static bool parse_double_value(const char* flag, const char* value, double& out) {
    if (!value || !*value) return cli_bad_value(flag, value, "expected finite number");
    char* end = nullptr;
    errno = 0;
    double parsed = std::strtod(value, &end);
    if (errno == ERANGE || end == value || *end != '\0' || !std::isfinite(parsed)) {
        return cli_bad_value(flag, value, "expected finite number");
    }
    out = parsed;
    return true;
}

static bool parse_positive_double(const char* flag, const char* value, double& out) {
    if (!parse_double_value(flag, value, out)) return false;
    if (out <= 0.0) return cli_bad_value(flag, value, "must be > 0");
    return true;
}

static bool parse_nonnegative_double(const char* flag, const char* value, double& out) {
    if (!parse_double_value(flag, value, out)) return false;
    if (out < 0.0) return cli_bad_value(flag, value, "must be >= 0");
    return true;
}

static bool parse_selector_pair(const char* flag, const std::string& value,
                                const std::string& body) {
    size_t comma = body.find(',');
    if (comma == std::string::npos || body.find(',', comma + 1) != std::string::npos) {
        return cli_bad_value(flag, value.c_str(), "expected selector(x,y)");
    }
    double x = 0.0, y = 0.0;
    return parse_double_value(flag, body.substr(0, comma).c_str(), x) &&
           parse_double_value(flag, body.substr(comma + 1).c_str(), y);
}

static bool validate_gamma_selector(const char* flag, const std::string& spec,
                                    const std::string& selector) {
    if (selector.empty()) return cli_bad_value(flag, spec.c_str(), "missing selector");
    if (selector.rfind("cell", 0) == 0) {
        int cid = 0;
        return parse_int_range(flag, selector.c_str() + 4, 0, INT_MAX, cid);
    }
    if (selector.rfind("nearest(", 0) == 0 && selector.back() == ')') {
        return parse_selector_pair(flag, spec, selector.substr(8, selector.size() - 9));
    }
    if (selector.rfind("cluster(", 0) == 0 && selector.back() == ')') {
        std::string body = selector.substr(8, selector.size() - 9);
        size_t pct_mark = body.find('%');
        size_t comma = body.find(',', pct_mark == std::string::npos ? 0 : pct_mark + 1);
        if (pct_mark == std::string::npos || comma == std::string::npos) {
            return cli_bad_value(flag, spec.c_str(), "expected cluster(p%,x,y)");
        }
        double pct = 0.0;
        if (!parse_positive_double(flag, body.substr(0, pct_mark).c_str(), pct)) return false;
        if (pct > 100.0) return cli_bad_value(flag, spec.c_str(), "percentage must be <= 100");
        return parse_selector_pair(flag, spec, body.substr(pct_mark + 2));
    }
    size_t pct_mark = selector.find('%');
    if (pct_mark != std::string::npos && pct_mark == selector.size() - 1) {
        double pct = 0.0;
        if (!parse_positive_double(flag, selector.substr(0, pct_mark).c_str(), pct)) return false;
        if (pct > 100.0) return cli_bad_value(flag, spec.c_str(), "percentage must be <= 100");
        return true;
    }
    return cli_bad_value(flag, spec.c_str(), "unknown gamma selector");
}

static bool validate_gamma_spec_arg(const char* flag, const std::string& spec,
                                    double* bare_gamma_out = nullptr) {
    if (spec.empty()) return cli_bad_value(flag, spec.c_str(), "empty gamma spec");
    size_t start = 0;
    bool first = true;
    while (start <= spec.size()) {
        size_t sep = spec.find(';', start);
        std::string segment = (sep == std::string::npos)
            ? spec.substr(start)
            : spec.substr(start, sep - start);
        if (segment.empty()) return cli_bad_value(flag, spec.c_str(), "empty gamma segment");
        size_t colon = segment.find(':');
        std::string gamma_text = (colon == std::string::npos) ? segment : segment.substr(0, colon);
        double gamma = 0.0;
        if (!parse_positive_double(flag, gamma_text.c_str(), gamma)) return false;
        if (colon != std::string::npos &&
            !validate_gamma_selector(flag, segment, segment.substr(colon + 1))) return false;
        if (first && sep == std::string::npos && colon == std::string::npos && bare_gamma_out) {
            *bare_gamma_out = gamma;
        }
        if (sep == std::string::npos) break;
        start = sep + 1;
        first = false;
    }
    return true;
}

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
    printf("  --tau <f>             Reorientation time (default: 10000; 0 disables tumbling)\n");
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
    printf("                        across GPUs; the global S(x,y) field is slab-\n");
    printf("                        decomposed, with halo Send/Recv pairs between\n");
    printf("                        neighbours each step.\n");
    printf("  -h, --help            Show this help\n");
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    // Cooperative SIGTERM/SIGINT: ask the step loop to break, so trajectory
    // writer drains and the final checkpoint is saved instead of dying
    // mid-fprintf. A second signal of the same kind falls through to the
    // default handler (immediate terminate), so the user can always abort
    // a hung shutdown with a second Ctrl+C / kill. std::signal is portable
    // across POSIX and Windows; both platforms self-reset to SIG_DFL on
    // some signals already, but we re-arm SIG_DFL explicitly to be sure.
    auto handler = [](int) {
        request_termination();
        std::signal(SIGTERM, SIG_DFL);
        std::signal(SIGINT,  SIG_DFL);
    };
    std::signal(SIGTERM, handler);
    std::signal(SIGINT,  handler);
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
        if      (!strcmp(argv[i], "-n") && i+1<argc) {
            if (!parse_int_range("-n", argv[++i], 1, INT_MAX, ncells)) return 1;
            ncells_set = true;
        }
        else if (!strcmp(argv[i], "-r") && i+1<argc) {
            if (!parse_positive_double("-r", argv[++i], p.target_radius)) return 1;
            ov.target_radius = true;
        }
        else if (!strcmp(argv[i], "--radius") && i+1<argc) {
            if (!parse_positive_double("--radius", argv[++i], p.target_radius)) return 1;
            ov.target_radius = true;
        }
        else if (!strcmp(argv[i], "-N") && i+1<argc) { p.Nx = atoi(argv[++i]); p.Ny = p.Nx; nx_set = true; }
        else if (!strcmp(argv[i], "--confluence") && i+1<argc) {
            double parsed = 0.0;
            if (!parse_positive_double("--confluence", argv[++i], parsed)) return 1;
            confluence = (float)parsed;
        }
        else if (!strcmp(argv[i], "-t") && i+1<argc) {
            if (!parse_nonnegative_double("-t", argv[++i], p.t_end)) return 1;
            ov.t_end = true;
        }
        else if (!strcmp(argv[i], "--dt") && i+1<argc) {
            if (!parse_positive_double("--dt", argv[++i], p.dt)) return 1;
            ov.dt = true;
        }
        else if (!strcmp(argv[i], "-dt") && i+1<argc) {
            if (!parse_positive_double("-dt", argv[++i], p.dt)) return 1;
            ov.dt = true;
        }
        else if (!strcmp(argv[i], "--v-A") && i+1<argc) {
            if (!parse_nonnegative_double("--v-A", argv[++i], p.v_A)) return 1;
            ov.v_A = true;
        }
        else if (!strcmp(argv[i], "--v-A-sigma") && i+1<argc) {
            if (!parse_nonnegative_double("--v-A-sigma", argv[++i], v_A_sigma)) return 1;
        }
        else if (!strcmp(argv[i], "--tau") && i+1<argc) {
            if (!parse_nonnegative_double("--tau", argv[++i], p.tau)) return 1;
            ov.tau = true;
        }
        else if (!strcmp(argv[i], "--gamma") && i+1<argc) {
            // Composable: multiple --gamma flags accumulate, separated by ';'.
            // apply_gamma_spec() in sim.cu walks the segments in order.
            // The FIRST segment, if it's a bare scalar (no ':' / '%'), also
            // updates p.gamma globally (sets the baseline for all cells).
            std::string new_spec = argv[++i];
            double bare_gamma = 0.0;
            if (!validate_gamma_spec_arg("--gamma", new_spec, &bare_gamma)) return 1;
            if (gamma_spec.empty()) {
                gamma_spec = new_spec;
                if (new_spec.find(':') == std::string::npos &&
                    new_spec.find('%') == std::string::npos) {
                    p.gamma = bare_gamma;
                    ov.gamma = true;
                }
            } else {
                gamma_spec += ";";
                gamma_spec += new_spec;
            }
        }
        else if (!strcmp(argv[i], "--kappa") && i+1<argc) {
            if (!parse_nonnegative_double("--kappa", argv[++i], p.kappa)) return 1;
            ov.kappa = true;
        }
        else if (!strcmp(argv[i], "--mu") && i+1<argc) {
            if (!parse_nonnegative_double("--mu", argv[++i], p.mu)) return 1;
            ov.mu = true;
        }
        else if (!strcmp(argv[i], "--xi") && i+1<argc) {
            if (!parse_positive_double("--xi", argv[++i], p.xi)) return 1;
            ov.xi = true;
        }
        else if (!strcmp(argv[i], "--lambda") && i+1<argc) {
            if (!parse_positive_double("--lambda", argv[++i], p.lambda)) return 1;
            ov.lambda = true;
        }
        else if (!strcmp(argv[i], "-l") && i+1<argc) {
            if (!parse_positive_double("-l", argv[++i], p.lambda)) return 1;
            ov.lambda = true;
        }
        else if (!strcmp(argv[i], "--subdomain-padding") && i+1<argc) {
            if (!parse_nonnegative_double("--subdomain-padding", argv[++i], p.subdomain_padding)) return 1;
            ov.subdomain_padding = true;
        }
        else if (!strcmp(argv[i], "--abp")) { p.abp = true; ov.abp = true; }
        else if (!strcmp(argv[i], "-o") && i+1<argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "-c") && i+1<argc) ckpt_path = argv[++i];
        else if (!strcmp(argv[i], "--save-interval") && i+1<argc) {
            if (!parse_int_range("--save-interval", argv[++i], 0, INT_MAX, p.save_interval)) return 1;
            ov.save_interval = true;
        }
        else if (!strcmp(argv[i], "--checkpoint-interval") && i+1<argc) {
            if (!parse_int_range("--checkpoint-interval", argv[++i], 0, INT_MAX, checkpoint_interval)) return 1;
        }
        else if (!strcmp(argv[i], "--save-final-checkpoint")) save_final = true;
        else if (!strcmp(argv[i], "--no-save-final-checkpoint")) save_final = false;
        else if (!strcmp(argv[i], "--print-interval") && i+1<argc) {
            if (!parse_int_range("--print-interval", argv[++i], 0, INT_MAX, p.print_interval)) return 1;
            ov.print_interval = true;
        }
        else if (!strcmp(argv[i], "--trajectory-samples") && i+1<argc) {
            if (!parse_int_range("--trajectory-samples", argv[++i], 0, INT_MAX, p.trajectory_samples)) return 1;
            ov.trajectory_samples = true;
        }
        // Accept-and-translate: --trajectory-interval is converted to --trajectory-samples
        // once we know t_end and dt (after parsing finishes).
        else if (!strcmp(argv[i], "--trajectory-interval") && i+1<argc) {
            if (!parse_int_range("--trajectory-interval", argv[++i], 0, INT_MAX, trajectory_interval)) return 1;
        }
        else if (!strcmp(argv[i], "--vtk-interval") && i+1<argc) {
            if (!parse_int_range("--vtk-interval", argv[++i], 0, INT_MAX, vtk_interval)) return 1;
        }
        else if (!strcmp(argv[i], "--live-view")) { live_view = true; }
        else if (!strcmp(argv[i], "--live-view-interval") && i+1<argc) {
            if (!parse_int_range("--live-view-interval", argv[++i], 1, INT_MAX, live_view_interval)) return 1;
            live_view_interval_set = true;
        }
        else if (!strcmp(argv[i], "--live-view-tu") && i+1<argc) {
            if (!parse_nonnegative_double("--live-view-tu", argv[++i], live_view_tu)) return 1;
        }
        // TODO(observables): re-introduce a `--observables [interval=<n>]` CLI
        // surface that drives an on-GPU per-cell measurement pass (volume,
        // perimeter, shape index, stress/strain tensors, neighbor contacts)
        // and writes observables.csv. Historic flags `--use-diagnostics` and
        // `--observable-interval` are removed here; restore as a single,
        // properly-implemented `--observables` flag once the kernel is ready.
        else if (!strcmp(argv[i], "--seed") && i+1<argc) {
            if (!parse_uint_value("--seed", argv[++i], p.seed)) return 1;
            ov.seed = true;
        }
        else if (!strcmp(argv[i], "--polarity-seed") && i+1<argc) {
            if (!parse_uint_value("--polarity-seed", argv[++i], p.polarity_seed)) return 1;
            ov.polarity_seed = true;
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
    // The parser + device upload + invariant maintenance all live on
    // Simulation now (load_scripted_events). Earlier this was inlined
    // and poked 7 public Simulation fields by hand.
    if (!scripted_events_path.empty()) {
        if (!sim.load_scripted_events(scripted_events_path)) return 1;
    }

    sim.run();
    sim.cleanup();
    return 0;
}
