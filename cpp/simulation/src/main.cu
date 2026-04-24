// main.cu — CLI entry point
#include "sim.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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
    printf("  --tau <f>             Reorientation time (default: 10000)\n");
    printf("  --gamma <spec>        Surface tension. <spec> = <f> | <f>:cell<k> | <f>:<p>%%\n");
    printf("  --kappa <f>           Interaction strength (default: 10.0)\n");
    printf("  --mu <f>              Volume constraint (default: 1.0)\n");
    printf("  --xi <f>              Friction (default: 1500)\n");
    printf("  --lambda <f>          Interface width (default: 7.0)\n");
    printf("  -l <f>                Interface width (alias for --lambda)\n");
    printf("  --subdomain-padding <f>  Bbox padding as fraction of R (default: 0.6)\n");
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
    printf("  --save-individual-fields  Write per-cell phi (accepted, no-op stub in v2)\n");
    printf("  --use-diagnostics         Enable diagnostic outputs (accepted, no-op stub in v2)\n");
    printf("  --observable-interval <n> Diagnostic cadence (accepted, no-op stub in v2)\n");
    printf("  --stress-fields           Include stress tensor in VTK (accepted, no-op stub in v2)\n");
    printf("  --safe-mode               Cap GPU memory at 1 GB (accepted, no-op stub in v2)\n");
    printf("  --seed <n>            Placement RNG seed\n");
    printf("  --polarity-seed <n>   Polarity RNG seed\n");
    printf("  -h, --help            Show this help\n");
}

int main(int argc, char** argv) {
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
        // Accept-and-ignore stubs: baseline-compatible flags whose payloads
        // (per-cell VTK, GPU diagnostics, stress tensor, mem cap) are not yet
        // implemented in v2. Keeping the CLI surface intact prevents script
        // breakage and lets test_features.py exercise the migration path.
        else if (!strcmp(argv[i], "--save-individual-fields")) { /* stub */ }
        else if (!strcmp(argv[i], "--use-diagnostics"))        { /* stub */ }
        else if (!strcmp(argv[i], "--observable-interval") && i+1<argc) { ++i; /* stub */ }
        else if (!strcmp(argv[i], "--stress-fields"))          { /* stub */ }
        else if (!strcmp(argv[i], "--safe-mode"))              { /* stub */ }
        else if (!strcmp(argv[i], "--seed") && i+1<argc) { p.seed = atoi(argv[++i]); ov.seed = true; }
        else if (!strcmp(argv[i], "--polarity-seed") && i+1<argc) {
            p.polarity_seed = atoi(argv[++i]); ov.polarity_seed = true;
        }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }

    // Mutually-exclusive domain specification: -N OR --confluence, not both.
    // Baseline cell_sim rejects this combination; v2 must match.
    if (confluence > 0 && nx_set) {
        fprintf(stderr,
                "Error: -N and --confluence are mutually exclusive. "
                "Pick one.\n");
        return 1;
    }
    if (confluence > 0) {
        int L = Simulation::domain_for(ncells_set ? ncells : 8,
                                       p.target_radius, confluence);
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

    Simulation sim;
    sim.out_dir = outdir;
    sim.save_final_checkpoint = save_final;
    sim.checkpoint_interval = checkpoint_interval;
    sim.gamma_spec = gamma_spec;
    sim.v_A_sigma = v_A_sigma;
    sim.vtk_interval = vtk_interval;

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

    sim.run();
    sim.cleanup();
    return 0;
}
