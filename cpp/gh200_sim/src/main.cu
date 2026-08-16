// ===========================================================================
// FUSE-1R CLI driver.
// ===========================================================================

#include "../include/checkpoint.cuh"
#include "../include/params.cuh"
#include "../include/sim.cuh"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace pf;

// ---------------------------------------------------------------------------
static void usage(const char* argv0) {
    std::printf(
"FUSE-1R -- GH200 phase-field cell simulation (Palmieri et al. 2015, Eq. S15)\n"
"\n"
"usage: %s [options]\n"
"\n"
"physics\n"
"  --N <int>              number of cells                    (288)\n"
"  --radius <f>           target radius R                    (49)\n"
"  --rho <f>              packing fraction, sets the domain  (0.90)\n"
"  --lambda <f>           interface width parameter          (7)\n"
"  --kappa <f>            repulsion strength                 (10)\n"
"  --mu <f>               volume-constraint strength         (1)\n"
"  --xi <f>               friction                           (1500)\n"
"  --tau <f>              run-and-tumble persistence time    (1e4)\n"
"  --v-A <f>              active self-propulsion speed       (1e-2)\n"
"  --v-A-sigma <f>        lognormal spread on per-cell v_A   (0)\n"
"  --gamma <f>            gamma for normal cells             (1.0)\n"
"  --gamma-cancer <f>     gamma for cancer cells             (0.35)\n"
"  --cancer-fraction <f>  fraction of cells given the soft gamma (0)\n"
"\n"
"numerics / run control\n"
"  --dt <f>               time step                          (0.01)\n"
"  --t-end <f>            end time                           (100)\n"
"  --seed <u64>           grid-placement/per-cell RNG seed   (1234)\n"
"  --polarity-seed <u64>  RNG stream for initial polarity AND every\n"
"                         tumble, independent of --seed. 0 = follow --seed.\n"
"                         Matched pairs share this so both branches\n"
"                         reorient at the same times and angles.\n"
"  --initial-centres <csv>  fresh-start accepted centres, exact header\n"
"                         global_id,x,y. Strictly validated; when absent the\n"
"                         historical grid+jitter initializer remains active.\n"
"                         Matched branches must load the same CSV bytes.\n"
"  --print-interval <int> steps between status lines         (100)\n"
"  --full-moment <int>    steps between perimeter updates    (100)\n"
"  --verify-every <int>   steps between strict-mode checks    (4096)\n"
"  --device <int>         CUDA device                        (0)\n"
"\n"
"performance switches (A/B matrix -- sweep these inside ONE job)\n"
"  --bench <int>          run <int> timed steps and exit\n"
"  --split                two kernels per step (k_step_rhs + k_step_post) at\n"
"                         512 threads instead of the fused k_step at 768. Frees\n"
"                         S_s from shared memory and lowers the register budget\n"
"                         so more than one CTA can share an SM. NOT the default,\n"
"                         and NOT bit-comparable with the fused path: the fp64\n"
"                         reductions run over 16 warp slots instead of 24, so\n"
"                         the two are different (equally valid, individually\n"
"                         reproducible) trajectories. Do not cross-compare dumps.\n"
"  --no-graph             per-step launches instead of a 6-step CUDA graph\n"
"  --morton               Morton-order cell traversal (helps only at large N)\n"
"  --no-l2                do not pin S in the L2 persisting carve-out\n"
"  --strict               run the invariant checks on the verify cadence\n"
"\n"
"output\n"
"  --out <path>           trajectory.txt, legacy cell_sim format (cell_analyze reads it)\n"
"  --trajectory-samples <int>  evenly spaced trajectory frames         (100)\n"
"  --trajectory-interval <int> steps between frames (overrides samples)\n"
"  --dual-centroid-out <path>  validation-only sidecar: phi and phi^2 periodic\n"
"                              centroids on each legacy trajectory frame\n"
"  --dump-state <path>    binary state dump for the CPU-reference comparison\n"
"  --self-test            run the coefficient / RNG / sign validation gates\n"
"  -h, --help             this message\n"
"\n"
"checkpointing (v8 format, byte-compatible with cpp/simulation's cell_sim)\n"
"  -c, --checkpoint <path>     RESUME from this checkpoint. The file's\n"
"                              SimParams are adopted wholesale and then ONLY\n"
"                              the flags you passed explicitly override them,\n"
"                              so `-c eq.bin --gamma 0.35 --t-end 2e6` keeps\n"
"                              the equilibrated microstate and changes gamma\n"
"                              and nothing else. --N and --rho are REFUSED on\n"
"                              resume: both change the domain side, which every\n"
"                              stored origin and phi tile is expressed in.\n"
"  --checkpoint-interval <int> steps between rolling <dir>/checkpoint.bin\n"
"                              (overwritten in place, atomically)\n"
"  --save-interval <int>       steps between tagged <dir>/checkpoint_%%08d.bin\n"
"  --checkpoint-dir <path>     where they go. Defaults to the directory of\n"
"                              --out, else of -c, else '.'. Checkpointing is\n"
"                              OFF unless one of these four flags is given.\n"
"  --no-final-checkpoint       do not write <dir>/checkpoint.bin at exit.\n"
"                              By default one is written on normal completion\n"
"                              AND on SIGTERM, so a walltime kill leaves a\n"
"                              file the next leg can resume from.\n",
    argv0);
}

static bool parse_d(const char* flag, const char* s, double* out,
                    double lo, double hi) {
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !(v >= lo && v <= hi)) {
        std::fprintf(stderr, "[fatal] %s: expected a number in [%g, %g], got '%s'\n",
                     flag, lo, hi, s);
        return false;
    }
    *out = v;
    return true;
}

static bool parse_i(const char* flag, const char* s, long long* out,
                    long long lo, long long hi) {
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0' || v < lo || v > hi) {
        std::fprintf(stderr, "[fatal] %s: expected an integer in [%lld, %lld], "
                     "got '%s'\n", flag, lo, hi, s);
        return false;
    }
    *out = v;
    return true;
}

static bool parse_u64(const char* flag, const char* s, unsigned long long* out) {
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') {
        std::fprintf(stderr, "[fatal] %s: expected an unsigned integer, got '%s'\n",
                     flag, s);
        return false;
    }
    *out = v;
    return true;
}

static bool same_output_path(const std::string& lhs, const std::string& rhs) {
    std::error_code lhs_ec, rhs_ec;
    const std::filesystem::path lhs_path =
        std::filesystem::weakly_canonical(std::filesystem::path(lhs), lhs_ec);
    const std::filesystem::path rhs_path =
        std::filesystem::weakly_canonical(std::filesystem::path(rhs), rhs_ec);
    if (!lhs_ec && !rhs_ec) return lhs_path == rhs_path;
    return std::filesystem::path(lhs).lexically_normal()
        == std::filesystem::path(rhs).lexically_normal();
}

// ===========================================================================
// Validation gates. These are the checks that would have caught the two known
// historical bugs in this physics, so they run on demand and cost nothing.
// ===========================================================================
static int gate_laplacian() {
    // lap(x^2) must be exactly 2 at h = 1, and the weights must sum to 0.
    // f(x,y) = x^2 about the origin, so every row is (1, 0, 1): f[j][i] with
    // i = x+1, j = y+1.
    const double f[3][3] = {
        {1.0, 0.0, 1.0},                  // y = -1
        {1.0, 0.0, 1.0},                  // y =  0
        {1.0, 0.0, 1.0},                  // y = +1
    };
    const double lap =
        (kLapEdgeW * (f[2][1] + f[0][1] + f[1][2] + f[1][0])
         + kLapDiagW * (f[2][2] + f[2][0] + f[0][2] + f[0][0])
         + kLapCentreW * f[1][1]) / kLapDenom;
    const double wsum = 4.0 * kLapEdgeW + 4.0 * kLapDiagW + kLapCentreW;
    const bool ok = (lap == 2.0) && (wsum == 0.0);
    std::printf("  [%s] gate 1  lap(x^2) = %.17g (want exactly 2), "
                "weight sum = %.17g\n", ok ? "PASS" : "FAIL", lap, wsum);
    return ok ? 0 : 1;
}

static int gate_coefficients(const SimParams& p) {
    const double bulk = p.bulk();
    const double rep  = p.interaction();
    const double mot  = p.motility();
    const double ratio = rep / mot;
    const bool ok = (kNumerInteraction == 2 * kNumerBulk)
                 && std::fabs(ratio - p.xi) <= 1e-12 * p.xi;
    std::printf("  [%s] gate 2  30/l^2 = %.9g   60k/l^2 = %.9g   "
                "60k/(xi l^2) = %.9g   ratio = %.12g (want xi = %.12g)\n",
                ok ? "PASS" : "FAIL", bulk, rep, mot, ratio, p.xi);
    std::printf("            repulsion numerator is %d (2x the bulk %d): "
                "Eq.(10) sums over ordered pairs\n",
                kNumerInteraction, kNumerBulk);
    return ok ? 0 : 1;
}

static int gate_velocity_sign(const SimParams& p) {
    // Cell n to the LEFT of cell m. In the overlap grad_x(phi_n) < 0, so
    // Ix = integral(phi_n grad_x(phi_n) phi_m^2) < 0, so vx < 0: n accelerates
    // AWAY from m. If this comes out positive the interaction is attractive
    // and the sign convention is wrong somewhere.
    const int L = 192;
    const double R = 20.0, lam = p.lambda;
    const double k = interface_k(lam);
    const double mot = p.motility();
    int bad = 0;
    for (int sep = 26; sep <= 40; sep += 4) {
        const double nx = 0.5 * L - 0.5 * sep, mx = 0.5 * L + 0.5 * sep;
        std::vector<double> pn((size_t)L * L), pm((size_t)L * L);
        for (int y = 0; y < L; ++y)
            for (int x = 0; x < L; ++x) {
                const double dyn = y - 0.5 * L;
                const double rn = std::sqrt((x - nx) * (x - nx) + dyn * dyn);
                const double rm = std::sqrt((x - mx) * (x - mx) + dyn * dyn);
                pn[(size_t)y * L + x] = 0.5 * (1.0 - std::tanh(k * (rn - R)));
                pm[(size_t)y * L + x] = 0.5 * (1.0 - std::tanh(k * (rm - R)));
            }
        double Ix = 0.0;
        for (int y = 0; y < L; ++y)
            for (int x = 1; x < L - 1; ++x) {
                const size_t i = (size_t)y * L + x;
                const double gx = 0.5 * (pn[i + 1] - pn[i - 1]);
                Ix += pn[i] * gx * pm[i] * pm[i];
            }
        const double vx = mot * Ix;
        const bool ok = (Ix < 0.0);
        if (!ok) ++bad;
        std::printf("  [%s] gate 3  sep %2d  Ix = %+.6e  vx = %+.6e  "
                    "(must be negative: n moves away from m)\n",
                    ok ? "PASS" : "FAIL", sep, Ix, vx);
    }
    return bad ? 1 : 0;
}

static int gate_tumble(const SimParams& p) {
    // The expm1 requirement is an exact-arithmetic property, so test it
    // exactly -- that is strictly stronger than any finite sample.
    const double dt = p.dt, tau = p.tau;
    const double p_good = -std::expm1(-dt / tau);
    const float  p_bad  = 1.0f - std::exp((float)(-dt / tau));
    const double tau_good = -dt / std::log1p(-p_good);
    const double tau_bad  = -dt / std::log1p(-(double)p_bad);
    const bool ok = std::fabs(tau_good - tau) <= 1e-9 * tau;
    std::printf("  [%s] gate 4a -expm1(-dt/tau)  = %.9e -> tau_eff = %.4f\n",
                ok ? "PASS" : "FAIL", p_good, tau_good);
    std::printf("            1-expf(-dt/tau)   = %.9e -> tau_eff = %.4f "
                "(the historical bug: +%.3f%% bias)\n",
                (double)p_bad, tau_bad,
                100.0 * ((double)p_bad / p_good - 1.0));

    // Sampling gate on the Philox stream itself, at a p large enough for the
    // statistics to be cheap: measures uniformity, not the expm1 arithmetic.
    const double tau_s = 10.0;
    const double ps = -std::expm1(-dt / tau_s);
    const long long trials = 5000000LL;
    long long hits = 0;
    for (long long i = 0; i < trials; ++i) {
        const Philox4 r = philox4x32_10((uint32_t)(i & 0xFFFFFFFF),
                                        (uint32_t)(i >> 32), 7u, 0xA5A5A5A5u,
                                        (uint32_t)p.seed, 0u);
        if (philox_uniform53(r.v[0], r.v[1]) < ps) ++hits;
    }
    const double phat = (double)hits / (double)trials;
    const double tau_meas = -dt / std::log1p(-phat);
    const double sigma = std::sqrt(ps * (1.0 - ps) / (double)trials) / ps;
    const bool ok2 = std::fabs(tau_meas - tau_s) <= 5.0 * sigma * tau_s;
    std::printf("  [%s] gate 4b %lld Philox samples at p = %.6e: "
                "tau_eff = %.3f (want %.1f +- %.3f at 5 sigma)\n",
                ok2 ? "PASS" : "FAIL", trials, ps, tau_meas, tau_s,
                5.0 * sigma * tau_s);
    return (ok && ok2) ? 0 : 1;
}

static int gate_geometry(const SimParams& p) {
    // Interface decay and the rect-sizing argument the W = 144 choice rests on,
    // evaluated for THIS run's lambda and R, not for the paper's.
    const double lam = p.lambda, R = p.target_radius;
    const double k = interface_k(lam);
    const double A0 = target_area(R);
    const double circum = std::sqrt(A0 / (1.5 * std::sqrt(3.0)));
    // phi(d) = 1/(1 + exp(2 k d)) so the phi tail is exp(-2 k d) and the phi^2
    // tail is exp(-4 k d). Half a Q5.27 quantum is 0.5 * 2^-27. The support
    // threshold is kSupportEps -- the same constant the bbox that drives the
    // shape-class decision is measured at, not a literal copy of it.
    const double eps = (double)kSupportEps;
    const double d_supp = std::log(1.0 / eps - 1.0) / (2.0 * k);
    const double d_q = std::log(1.0 / (0.5 * kQInvD)) / (4.0 * k);
    const double need = circum + d_q + 1.0;

    // EVERY class must clear the sizing argument on BOTH axes, not just the
    // round one. Testing only kClassRound is what hid the 208x112 / 112x208
    // classes, whose 112 px short side was below both this required window and
    // the phi > 1e-5 support extent, so any promotion truncated phi.
    bool ok = true;
    for (int c = 0; c < kNumClasses; ++c) {
        const int wmin = std::min(kClasses[c].wx, kClasses[c].wy);
        if ((double)(wmin / 2) < need) ok = false;
    }
    std::printf("  [%s] gate 5  k = %.6f  hex circumradius %.2f  "
                "phi^2 < half a quantum at %.2f  phi < %.0e at %.2f\n",
                ok ? "PASS" : "FAIL", k, circum, d_q, eps, d_supp);
    std::printf("            required half-width %.2f px on both axes of every "
                "class; support extent of a relaxed cell %.2f px\n",
                need, 2.0 * (R + d_supp));
    // Per class: the sizing margin, the shared-memory cost, whether it stages S
    // and the largest support extent it can be promoted INTO. That last column
    // is the one the soft-gamma campaign cares about: FLAG_CLASS_EXHAUSTED
    // means "bigger than the largest number in it, on some axis".
    int max_ex = 0, max_ey = 0;
    for (int c = 0; c < kNumClasses; ++c) {
        const int wmin = std::min(kClasses[c].wx, kClasses[c].wy);
        const int hw = wmin / 2;
        max_ex = std::max(max_ex, kClasses[c].wx - kPromoteSlack);
        max_ey = std::max(max_ey, kClasses[c].wy - kPromoteSlack);
        std::printf("            class %d  %3d x %3d  short half-width %d, "
                    "drift margin %+.2f px   smem %6d B  %s   holds extent "
                    "<= %d x %d%s\n",
                    c, kClasses[c].wx, kClasses[c].wy, hw,
                    (double)hw - need, class_smem_of(c),
                    class_stages_S(c) ? "phi+S " : "phi   (S from global)",
                    kClasses[c].wx - kPromoteSlack,
                    kClasses[c].wy - kPromoteSlack,
                    ((double)hw >= need) ? "" : "   <-- TOO SMALL, would clip");
    }
    std::printf("            widest representable support %d x %d px "
                "(kPromoteSlack = %d); anything past that is "
                "FLAG_CLASS_EXHAUSTED, never clipped\n",
                max_ex, max_ey, kPromoteSlack);
    std::printf("            shared memory per CTA %d B of %d B opt-in max "
                "(%.1f%%);  set by the staged classes at %d B, NOT by the "
                "large class at %d B\n",
                kSmemBytes, kSmemPerBlockOptinSm90,
                100.0 * kSmemBytes / kSmemPerBlockOptinSm90,
                smem_raw_staged_only(), class_smem_of(kClassLarge));
    // The split path's budget is a compile-time property, so it is reported by
    // the same gate whether or not --split was passed. static_asserts in
    // params.cuh already make a violation a build error; this is the readout.
    std::printf("            split path: k_step_rhs %d B/CTA of %d B budget "
                "for %d CTAs/SM (%.1f%%), k_step_post %d B static; "
                "REFUSES class %d (%d B > budget)\n",
                kSmemRhsBytes, kSplitSmemBudget, kSplitRhsCtasPerSm,
                100.0 * kSmemRhsBytes / kSplitSmemBudget, kSmemPostBytes,
                kClassLarge,
                class_smem_rhs(kClasses[kClassLarge].wx,
                               kClasses[kClassLarge].wy));
    return ok ? 0 : 1;
}

static int run_self_test(const SimParams& p) {
    std::printf("--- validation gates ---\n");
    int bad = 0;
    bad += gate_laplacian();
    bad += gate_coefficients(p);
    bad += gate_velocity_sign(p);
    bad += gate_tumble(p);
    bad += gate_geometry(p);
    std::printf("--- %s ---\n", bad ? "SELF-TEST FAILED" : "self-test passed");
    return bad ? 1 : 0;
}

// ===========================================================================
int main(int argc, char** argv) {
    SimParams p;
    RunOptions opt;
    // Which fields the user set EXPLICITLY. Only meaningful on resume, where
    // it decides what survives from the checkpoint and what the CLI replaces.
    SimOverrides ov;
    long long device = 0;
    bool self_test = false;
    std::string ckpt_in;
    std::string ckpt_dir_flag;
    bool no_final_ckpt = false;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto need = [&](void) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[fatal] %s requires a value\n", a);
                std::exit(2);
            }
            return argv[++i];
        };
        long long iv = 0;

        if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else if (!std::strcmp(a, "--N")) {
            if (!parse_i(a, need(), &iv, 1, 4000000)) return 2;
            p.num_cells = (int)iv;
            ov.num_cells = true;
        }
        else if (!std::strcmp(a, "--radius"))  { if (!parse_d(a, need(), &p.target_radius, 1e-6, 1e6)) return 2; ov.target_radius = true; }
        else if (!std::strcmp(a, "--rho"))     { if (!parse_d(a, need(), &p.rho, 1e-6, 0.999999)) return 2; ov.rho = true; }
        else if (!std::strcmp(a, "--lambda"))  { if (!parse_d(a, need(), &p.lambda, 1e-6, 1e6)) return 2; ov.lambda = true; }
        else if (!std::strcmp(a, "--kappa"))   { if (!parse_d(a, need(), &p.kappa, 0.0, 1e9)) return 2; ov.kappa = true; }
        else if (!std::strcmp(a, "--mu"))      { if (!parse_d(a, need(), &p.mu, 0.0, 1e9)) return 2; ov.mu = true; }
        else if (!std::strcmp(a, "--xi"))      { if (!parse_d(a, need(), &p.xi, 1e-9, 1e12)) return 2; ov.xi = true; }
        else if (!std::strcmp(a, "--tau"))     { if (!parse_d(a, need(), &p.tau, 1e-9, 1e12)) return 2; ov.tau = true; }
        else if (!std::strcmp(a, "--v-A"))     { if (!parse_d(a, need(), &p.v_A, 0.0, 1e6)) return 2; ov.v_A = true; }
        else if (!std::strcmp(a, "--v-A-sigma")) { if (!parse_d(a, need(), &p.v_A_sigma, 0.0, 10.0)) return 2; ov.v_A_sigma = true; }
        else if (!std::strcmp(a, "--gamma"))   { if (!parse_d(a, need(), &p.gamma_normal, 1e-9, 1e6)) return 2; ov.gamma = true; }
        else if (!std::strcmp(a, "--gamma-cancer")) { if (!parse_d(a, need(), &p.gamma_cancer, 1e-9, 1e6)) return 2; ov.gamma_cancer = true; }
        else if (!std::strcmp(a, "--cancer-fraction")) { if (!parse_d(a, need(), &p.cancer_fraction, 0.0, 1.0)) return 2; ov.cancer_fraction = true; }
        else if (!std::strcmp(a, "--dt"))      { if (!parse_d(a, need(), &p.dt, 1e-12, 1e6)) return 2; ov.dt = true; }
        else if (!std::strcmp(a, "--t-end"))   { if (!parse_d(a, need(), &p.t_end, 0.0, 1e12)) return 2; ov.t_end = true; }
        else if (!std::strcmp(a, "--seed"))    { if (!parse_u64(a, need(), &p.seed)) return 2; ov.seed = true; }
        else if (!std::strcmp(a, "--polarity-seed")) {
            if (!parse_u64(a, need(), &p.polarity_seed)) return 2;
            ov.polarity_seed = true;
        }
        else if (!std::strcmp(a, "--initial-centres")) {
            opt.initial_centres_path = need();
            if (opt.initial_centres_path.empty()) {
                std::fprintf(stderr, "[fatal] --initial-centres requires a non-empty path\n");
                return 2;
            }
        }
        else if (!std::strcmp(a, "--print-interval")) {
            if (!parse_i(a, need(), &iv, 0, 1000000000)) return 2;
            p.print_interval = (int)iv;
            ov.print_interval = true;
        }
        else if (!std::strcmp(a, "-c") || !std::strcmp(a, "--checkpoint")) {
            ckpt_in = need();
        }
        else if (!std::strcmp(a, "--checkpoint-interval")) {
            if (!parse_i(a, need(), &iv, 1, 1000000000000LL)) return 2;
            opt.ckpt_interval = iv;
        }
        else if (!std::strcmp(a, "--save-interval")) {
            if (!parse_i(a, need(), &iv, 1, 1000000000000LL)) return 2;
            opt.save_interval = iv;
        }
        else if (!std::strcmp(a, "--checkpoint-dir")) { ckpt_dir_flag = need(); }
        else if (!std::strcmp(a, "--no-final-checkpoint")) { no_final_ckpt = true; }
        else if (!std::strcmp(a, "--trajectory-samples")) {
            if (!parse_i(a, need(), &iv, 1, 1000000000)) return 2;
            opt.traj_samples = (int)iv;
        }
        else if (!std::strcmp(a, "--trajectory-interval")) {
            if (!parse_i(a, need(), &iv, 1, 1000000000)) return 2;
            opt.traj_interval = (long long)iv;
        }
        else if (!std::strcmp(a, "--full-moment")) {
            if (!parse_i(a, need(), &iv, 0, 1000000000)) return 2;
            p.full_moment_every = (int)iv;
            ov.full_moment = true;
        }
        else if (!std::strcmp(a, "--verify-every")) {
            if (!parse_i(a, need(), &iv, 1, 1000000000)) return 2;
            p.verify_every = (int)iv;
            ov.verify_every = true;
        }
        else if (!std::strcmp(a, "--device")) { if (!parse_i(a, need(), &device, 0, 63)) return 2; }
        else if (!std::strcmp(a, "--bench")) {
            if (!parse_i(a, need(), &iv, 1, 100000000)) return 2;
            opt.bench_steps = (int)iv;
        }
        else if (!std::strcmp(a, "--split"))    opt.split = true;
        else if (!std::strcmp(a, "--no-graph")) opt.use_graph = false;
        else if (!std::strcmp(a, "--morton"))   opt.morton = true;
        else if (!std::strcmp(a, "--no-l2"))    opt.l2_persist = false;
        else if (!std::strcmp(a, "--strict"))   opt.strict = true;
        else if (!std::strcmp(a, "--out"))        opt.out_path = need();
        else if (!std::strcmp(a, "--dual-centroid-out")) opt.dual_centroid_path = need();
        else if (!std::strcmp(a, "--dump-state")) opt.dump_path = need();
        else if (!std::strcmp(a, "--self-test"))  self_test = true;
        else {
            std::fprintf(stderr, "[fatal] unknown option '%s' (try --help)\n", a);
            return 2;
        }
    }
    if (!opt.dual_centroid_path.empty()) {
        if (opt.out_path.empty()) {
            std::fprintf(stderr,
                "[fatal] --dual-centroid-out requires --out: the validation "
                "sidecar is defined only on legacy trajectory sampling frames\n");
            return 2;
        }
        if (opt.bench_steps > 0) {
            std::fprintf(stderr,
                "[fatal] --dual-centroid-out cannot be combined with --bench, "
                "which intentionally performs no trajectory I/O\n");
            return 2;
        }
        if (same_output_path(opt.out_path, opt.dual_centroid_path)) {
            std::fprintf(stderr,
                "[fatal] --dual-centroid-out must differ from --out; the "
                "validation rows cannot be mixed into the legacy file\n");
            return 2;
        }
        std::printf("  dual centroids   %s  (validation-only; trajectory cadence)\n",
                    opt.dual_centroid_path.c_str());
    }
    // ---- resume, or fresh -------------------------------------------------
    // The two differ in exactly one place: where SimParams and the initial
    // condition come from. Everything downstream is identical, which is why
    // the override mask exists rather than a second code path.
    CheckpointData ckpt;
    if (!ckpt_in.empty()) {
        if (!opt.initial_centres_path.empty()) {
            std::fprintf(stderr,
                "[fatal] --initial-centres is valid only for a fresh start. "
                "A checkpoint already contains the complete initialised microstate.\n");
            return 2;
        }
        if (ov.num_cells || ov.rho) {
            std::fprintf(stderr,
                "[fatal] --N and --rho cannot be overridden on resume: they set "
                "the domain side, and every origin and phi tile in the "
                "checkpoint\n        is expressed in it. Drop the flag, or "
                "start a fresh run.\n");
            return 2;
        }
        if (!checkpoint_read(ckpt_in, &ckpt)) return 8;
        SimParams cli = p;          // what the user typed
        p = ckpt.params;            // what the file says
        ov.apply(p, cli);           // ... then only the flags they typed
        resolve_per_cell_scalars(p, ov, &ckpt);
    } else {
        // The domain is square by construction: L = ceil(sqrt(N*A0/rho)).
        p.Nx = p.Ny = domain_side_for(p.num_cells, p.target_radius, p.rho);
    }

    if (!validate(p)) return 3;
    const int pitch = s_pitch_for(p.Nx);
    print_params(p, p.Nx, pitch, opt.split);

    if (self_test) {
        const int rc = run_self_test(p);
        if (rc) return 4;
    }

    // ---- where checkpoints go ---------------------------------------------
    // Off entirely unless the user asked for it in one of four ways. Preference
    // order for the directory: the explicit flag, then this leg's trajectory
    // (new output belongs next to it), then the file we resumed from, then cwd.
    if (!ckpt_dir_flag.empty() || opt.ckpt_interval > 0 ||
        opt.save_interval > 0 || !ckpt_in.empty()) {
        if (!ckpt_dir_flag.empty()) {
            opt.ckpt_dir = ckpt_dir_flag;
        } else {
            const std::filesystem::path src =
                !opt.out_path.empty() ? std::filesystem::path(opt.out_path)
                                      : std::filesystem::path(ckpt_in);
            const std::filesystem::path dir = src.parent_path();
            opt.ckpt_dir = dir.empty() ? std::string(".") : dir.string();
        }
        std::error_code ec;
        std::filesystem::create_directories(opt.ckpt_dir, ec);
        if (ec) {
            std::fprintf(stderr, "[fatal] cannot create checkpoint directory "
                         "%s: %s\n", opt.ckpt_dir.c_str(), ec.message().c_str());
            return 2;
        }
        opt.final_checkpoint = !no_final_ckpt;
        std::printf("  checkpoints      %s  (rolling every %lld steps, tagged "
                    "every %lld, final %s)\n",
                    opt.ckpt_dir.c_str(), opt.ckpt_interval, opt.save_interval,
                    opt.final_checkpoint ? "yes" : "no");
    }

    Sim sim;
    const bool ok = ckpt_in.empty()
        ? sim.init(p, opt, (int)device)
        : sim.init_from_checkpoint(p, ckpt, opt, (int)device);
    if (!ok) {
        std::fprintf(stderr, "[fatal] initialisation failed\n");
        return 5;
    }
    // The repacked tiles are on the GPU now; do not keep a second copy of the
    // whole pool alive for the rest of the run.
    ckpt = CheckpointData{};

    if (opt.bench_steps > 0) {
        double ms = 0.0;
        if (!sim.bench(opt.bench_steps, &ms)) return 6;
    } else {
        // SLURM sends SIGTERM before SIGKILL at walltime. run() streams the
        // trajectory as it samples and polls this flag, so a killed job keeps
        // everything written up to that point instead of losing the whole run.
        std::signal(SIGTERM, Sim::request_termination);
        std::signal(SIGINT,  Sim::request_termination);
        if (!sim.run()) return 9;
    }

    if (!opt.dump_path.empty()) {
        if (!sim.dump_state(opt.dump_path)) return 7;
    }
    return 0;
}
