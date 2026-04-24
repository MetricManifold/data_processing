// sim.cu — Allocation, initialization, stepping loop, I/O
#include "sim.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdint>

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Place cells — rejection sampling with periodic distance
// ---------------------------------------------------------------------------
void Simulation::place_cells(int n, double R) {
    unsigned s = params.seed ? params.seed : 42;
    srand(s);
    h_cells.resize(n);
    if (n == 1) {
        h_cells[0] = {(double)(params.Nx / 2), (double)(params.Ny / 2),
                       R, params.gamma, params.v_A, 0, 0, 0, 0};
        return;
    }
    double area = (double)params.Nx * params.Ny;
    double spacing = std::fmax(2.0 * R, std::sqrt(area / n) * 0.8);
    int placed = 0;
    while (placed < n) {
        bool ok = false;
        for (int att = 0; att < 10000 && !ok; att++) {
            double cx = (double)rand() / RAND_MAX * params.Nx;
            double cy = (double)rand() / RAND_MAX * params.Ny;
            bool good = true;
            for (int j = 0; j < placed && good; j++) {
                double dx = std::fabs(cx - h_cells[j].cx);
                double dy = std::fabs(cy - h_cells[j].cy);
                if (dx > params.Nx * 0.5) dx = params.Nx - dx;
                if (dy > params.Ny * 0.5) dy = params.Ny - dy;
                if (std::sqrt(dx * dx + dy * dy) < spacing) good = false;
            }
            if (good) {
                h_cells[placed] = {cx, cy, R, params.gamma, params.v_A,
                                   0, 0, 0, 0};
                rand(); // consume 1 to match baseline Cell ctor
                ok = true;
                placed++;
            }
        }
        if (!ok) {
            spacing *= 0.95;
            if (spacing < R) {
                fprintf(stderr, "Warning: placed %d/%d\n", placed, n);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compute initial bounding boxes
// ---------------------------------------------------------------------------
void Simulation::compute_bboxes() {
    for (auto& c : h_cells) {
        int extent = (int)(c.radius + 3.0 * params.lambda);
        int pad = (int)(params.subdomain_padding * params.target_radius);
        int margin = extent + pad;
        int side = 2 * margin + 2 * params.halo;
        side = (side + 1) & ~1;  // round up to even
        c.w = side; c.h = side;
        c.ox = (int)c.cx - side / 2;
        c.oy = (int)c.cy - side / 2;
    }
}

// ---------------------------------------------------------------------------
// GPU allocation
// ---------------------------------------------------------------------------
void Simulation::alloc_gpu() {
    int n = (int)h_cells.size();
    cells.num_cells = n;
    size_t page = max_slot();
    cells.slot_size = page;
    cells.max_side = (int)sqrtf((float)page);

    // Phi pool: 2N slots (double buffer)
    size_t pool_bytes = 2 * n * page * sizeof(float);
    CK(cudaMalloc(&cells.phi_pool, pool_bytes));
    CK(cudaMemset(cells.phi_pool, 0, pool_bytes));

    // Phi pointer arrays
    float** h_phi = new float*[n];
    float** h_out = new float*[n];
    for (int i = 0; i < n; i++) {
        h_phi[i] = cells.phi_pool + (size_t)i * page;
        h_out[i] = cells.phi_pool + (size_t)(n + i) * page;
    }
    CK(cudaMalloc(&cells.phi_ptrs, n * sizeof(float*)));
    CK(cudaMalloc(&cells.phi_out_ptrs, n * sizeof(float*)));
    CK(cudaMemcpy(cells.phi_ptrs, h_phi, n * sizeof(float*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.phi_out_ptrs, h_out, n * sizeof(float*), cudaMemcpyHostToDevice));
    delete[] h_phi; delete[] h_out;

    // Helper lambdas
    auto ai = [&](int*& p) { CK(cudaMalloc(&p, n * sizeof(int))); };
    auto af = [&](float*& p) { CK(cudaMalloc(&p, n * sizeof(float))); };

    ai(cells.offsets_x); ai(cells.offsets_y);
    ai(cells.widths); ai(cells.heights);
    ai(cells.old_widths); ai(cells.old_heights);
    ai(cells.shift_x); ai(cells.shift_y);
    af(cells.velocities_x); af(cells.velocities_y);
    af(cells.volumes); af(cells.volume_devs);
    af(cells.centroids_x); af(cells.centroids_y);
    af(cells.ref_x); af(cells.ref_y);
    af(cells.perimeters);
    af(cells.moment_x); af(cells.moment_y);
    af(cells.polar_x); af(cells.polar_y);
    af(cells.polar_theta);
    af(cells.two_gamma); af(cells.two_gamma_bulk);
    af(cells.vol_coeff); af(cells.tgt_area);
    af(cells.tgt_radius); af(cells.v_A_cell);

    CK(cudaMalloc(&cells.nbr_list, n * K_MAX * sizeof(NeighborEntry)));
    ai(cells.nbr_count);
    CK(cudaMalloc(&cells.d_max_wh, 2 * sizeof(int)));
    CK(cudaMalloc(&cells.rng_states, n * sizeof(curandState)));

    // Zero dynamics arrays
    CK(cudaMemset(cells.velocities_x, 0, n * sizeof(float)));
    CK(cudaMemset(cells.velocities_y, 0, n * sizeof(float)));
    CK(cudaMemset(cells.volume_devs, 0, n * sizeof(float)));
    CK(cudaMemset(cells.shift_x, 0, n * sizeof(int)));
    CK(cudaMemset(cells.shift_y, 0, n * sizeof(int)));
    CK(cudaMemset(cells.moment_x, 0, n * sizeof(float)));
    CK(cudaMemset(cells.moment_y, 0, n * sizeof(float)));

    printf("[GPU] %d cells, pool=%.1f MB, slot=%zu (%dx%d)\n",
           n, pool_bytes / 1e6, page, cells.max_side, cells.max_side);
}

// ---------------------------------------------------------------------------
// Upload phi fields (tanh profile) and per-cell constants
// ---------------------------------------------------------------------------
void Simulation::upload_phi() {
    int n = (int)h_cells.size();
    size_t page = cells.slot_size;

    std::vector<int> h_ox(n), h_oy(n), h_w(n), h_h(n);
    std::vector<float> h_tg(n), h_tgb(n), h_ta(n), h_vc(n), h_tr(n);
    std::vector<float> h_vA(n), h_px(n), h_py(n), h_th(n);

    for (int i = 0; i < n; i++) {
        auto& c = h_cells[i];
        h_ox[i] = c.ox; h_oy[i] = c.oy;
        h_w[i] = c.w; h_h[i] = c.h;
        h_tg[i] = (float)(2.0 * c.gamma);
        h_tgb[i] = (float)(2.0 * c.gamma * params.bulk_coeff());
        h_ta[i] = (float)(M_PI * c.radius * c.radius);
        h_vc[i] = (float)(params.mu / (M_PI * c.radius * c.radius));
        h_tr[i] = (float)c.radius;
        h_vA[i] = (float)c.v_A;
        float theta = (float)(rand() % 10000) / 10000.0f * 2.0f * (float)M_PI;
        h_th[i] = theta;
        h_px[i] = cosf(theta); h_py[i] = sinf(theta);
    }

    CK(cudaMemcpy(cells.offsets_x, h_ox.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.offsets_y, h_oy.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.widths, h_w.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.heights, h_h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.two_gamma, h_tg.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.two_gamma_bulk, h_tgb.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.tgt_area, h_ta.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.vol_coeff, h_vc.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.tgt_radius, h_tr.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.v_A_cell, h_vA.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_x, h_px.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_y, h_py.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_theta, h_th.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Generate phi fields on host, upload per cell
    for (int i = 0; i < n; i++) {
        auto& c = h_cells[i];
        int w = c.w, h = c.h;
        std::vector<float> phi(page, 0.0f);
        double R = c.radius, lam = params.lambda;
        double R_eff = R + 0.7088 * lam - 0.5887 * lam * lam / R;
        double iw = std::sqrt(2.0) * lam;
        int Nx = params.Nx, Ny = params.Ny;

        for (int ly = 0; ly < h; ly++) {
            for (int lx = 0; lx < w; lx++) {
                double gx = (double)(c.ox + lx);
                double gy = (double)(c.oy + ly);
                double dx = gx - c.cx, dy = gy - c.cy;
                if (dx >  Nx * 0.5) dx -= Nx;
                if (dx < -Nx * 0.5) dx += Nx;
                if (dy >  Ny * 0.5) dy -= Ny;
                if (dy < -Ny * 0.5) dy += Ny;
                double r = std::sqrt(dx * dx + dy * dy);
                phi[ly * w + lx] = (float)(0.5 * (1.0 - std::tanh((r - R_eff) / iw)));
            }
        }
        // Zero halo
        for (int ly = 0; ly < h; ly++)
            for (int lx = 0; lx < w; lx++)
                if (lx < params.halo || lx >= w - params.halo ||
                    ly < params.halo || ly >= h - params.halo)
                    phi[ly * w + lx] = 0.0f;

        CK(cudaMemcpy(cells.phi_pool + (size_t)i * page,
                       phi.data(), w * h * sizeof(float), cudaMemcpyHostToDevice));
    }

    cache_w = cache_h = 0;
    for (auto& c : h_cells) {
        if (c.w > cache_w) cache_w = c.w;
        if (c.h > cache_h) cache_h = c.h;
    }
}

// ---------------------------------------------------------------------------
// Apply log-normal disorder to per-cell v_A (fresh init only).
// Matches baseline semantics: `v_A_sigma` is the DESIRED STD of the output
// distribution (not log-space sigma). We back-solve the log-normal params:
//     cv        = v_A_sigma / v_A                     (coefficient of variation)
//     sigma_ln  = sqrt(log(1 + cv²))
//     mu_ln     = log(v_A) - ½ sigma_ln²
// Then sample v_A_i = exp(mu_ln + sigma_ln · Z), Z ~ N(0,1) via Box-Muller.
// This gives E[v_A_i] = v_A and Std[v_A_i] = v_A_sigma exactly in the limit.
// Deterministic in params.seed. Does nothing when v_A_sigma <= 0 or v_A == 0.
// On resume, per-cell v_A is loaded from the VA_A sidecar and this function
// is NOT called — existing disorder is preserved bit-for-bit.
// ---------------------------------------------------------------------------
void Simulation::apply_v_A_disorder() {
    if (v_A_sigma <= 0.0) return;
    if (params.v_A <= 0.0) return;
    int n = (int)h_cells.size();
    if (n == 0) return;
    double cv       = v_A_sigma / params.v_A;
    double sigma_ln = std::sqrt(std::log(1.0 + cv * cv));
    double mu_ln    = std::log(params.v_A) - 0.5 * sigma_ln * sigma_ln;
    // Distinct RNG stream from placement (seed XOR golden ratio) for
    // reproducibility without correlation with cell positions.
    unsigned s = (params.seed ? params.seed : 42) ^ 0x9E3779B9u;
    srand(s);
    for (int i = 0; i < n; i++) {
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double z  = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        h_cells[i].v_A = std::exp(mu_ln + sigma_ln * z);
    }
}

// ---------------------------------------------------------------------------
// Parse gamma_spec: "<f>", "<f>:cell<k>", or "<f>:<p>%" and apply to h_cells.
// Default (empty spec): every cell gets params.gamma.
// Bare number: every cell gets that value AND params.gamma is already updated
//              in main.cu so this is redundant-safe.
// Selector: only matching cells take the value; others get params.gamma.
// ---------------------------------------------------------------------------
void Simulation::apply_gamma_spec() {
    int n = (int)h_cells.size();
    // baseline: every cell gets params.gamma
    for (auto& c : h_cells) c.gamma = params.gamma;
    if (gamma_spec.empty()) return;

    size_t colon = gamma_spec.find(':');
    double value = atof(gamma_spec.substr(0, colon).c_str());

    if (colon == std::string::npos) {
        for (auto& c : h_cells) c.gamma = value;
        return;
    }
    std::string sel = gamma_spec.substr(colon + 1);
    if (sel.rfind("cell", 0) == 0) {
        int k = atoi(sel.c_str() + 4);
        if (k >= 0 && k < n) h_cells[k].gamma = value;
    } else if (sel.rfind("nearest(", 0) == 0) {
        // :nearest(x,y) — set the single cell whose center is closest to (x,y).
        double xq = 0, yq = 0;
        if (sscanf(sel.c_str() + 8, "%lf,%lf", &xq, &yq) == 2 && n > 0) {
            int best = 0;
            double best_d2 = (h_cells[0].cx - xq) * (h_cells[0].cx - xq) +
                             (h_cells[0].cy - yq) * (h_cells[0].cy - yq);
            for (int i = 1; i < n; i++) {
                double dx = h_cells[i].cx - xq, dy = h_cells[i].cy - yq;
                double d2 = dx*dx + dy*dy;
                if (d2 < best_d2) { best_d2 = d2; best = i; }
            }
            h_cells[best].gamma = value;
        } else {
            fprintf(stderr, "Bad nearest() spec: %s\n", sel.c_str());
        }
    } else if (sel.rfind("cluster(", 0) == 0) {
        // :cluster(p%,x,y) — set the p% of cells nearest to (x,y).
        double pct = 0, xq = 0, yq = 0;
        if (sscanf(sel.c_str() + 8, "%lf%%,%lf,%lf", &pct, &xq, &yq) == 3 && n > 0) {
            int k = (int)(n * pct / 100.0 + 0.5);
            if (k < 1) k = 1;
            if (k > n) k = n;
            // Build (idx, d2) pairs and partial-sort.
            std::vector<std::pair<double,int>> dists;
            dists.reserve(n);
            for (int i = 0; i < n; i++) {
                double dx = h_cells[i].cx - xq, dy = h_cells[i].cy - yq;
                dists.push_back({dx*dx + dy*dy, i});
            }
            std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
            for (int j = 0; j < k; j++) h_cells[dists[j].second].gamma = value;
        } else {
            fprintf(stderr, "Bad cluster() spec: %s\n", sel.c_str());
        }
    } else if (!sel.empty() && sel.back() == '%') {
        double pct = atof(sel.substr(0, sel.size() - 1).c_str());
        int k = (int)(n * pct / 100.0 + 0.5);
        for (int i = 0; i < k && i < n; i++) h_cells[i].gamma = value;
    } else {
        fprintf(stderr, "Unknown gamma selector: %s\n", sel.c_str());
    }
}

// ---------------------------------------------------------------------------
// Initialize simulation (fresh start)
// ---------------------------------------------------------------------------
void Simulation::init(const SimParams& p, int n_cells) {
    params = p;
    if (params.Nx <= 0) {
        int L = domain_for(n_cells, params.target_radius, 0.85);
        params.Nx = L; params.Ny = L;
    }

    place_cells(n_cells, params.target_radius);
    apply_gamma_spec();
    apply_v_A_disorder();
    compute_bboxes();
    alloc_gpu();
    upload_phi();

    // Set reference points from bbox centers
    {
        int n = cells.num_cells;
        std::vector<float> rx(n), ry(n);
        for (int i = 0; i < n; i++) {
            auto& c = h_cells[i];
            rx[i] = fmodf(fmodf((float)c.ox + c.w * 0.5f, (float)params.Nx) + params.Nx,
                           (float)params.Nx);
            ry[i] = fmodf(fmodf((float)c.oy + c.h * 0.5f, (float)params.Ny) + params.Ny,
                           (float)params.Ny);
        }
        CK(cudaMemcpy(cells.ref_x, rx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cells.ref_y, ry.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }

    finalize_init();
    launch_initial_velocity(cells, params, cache_w, cache_h);
    CK(cudaDeviceSynchronize());

    printf("[SIM] %d cells, %dx%d, dt=%.4f, t_end=%.1f\n",
           n_cells, params.Nx, params.Ny, params.dt, params.t_end);
}

// ---------------------------------------------------------------------------
// Shared init tail: RNG, hash, ref safety, initial reduce, trajectory cadence.
// Does NOT run initial_velocity (state comes from checkpoint on resume).
// ---------------------------------------------------------------------------
void Simulation::finalize_init() {
    launch_rng_init(cells, params.polarity_seed ? params.polarity_seed : 12345);
    launch_hash_build(cells, params.Nx, params.Ny);

    cache_w = std::max(cache_w, 32);
    cache_h = std::max(cache_h, 32);

    launch_initial_reduce(cells, params, cache_w, cache_h);

    int total = std::max(1, (int)(params.t_end / params.dt));
    traj_every = params.trajectory_samples > 0
                 ? std::max(1, total / params.trajectory_samples) : 0;
}

// ---------------------------------------------------------------------------
// Resume from checkpoint
// ---------------------------------------------------------------------------
bool Simulation::init_from_checkpoint(const std::string& path,
                                      const SimParams& cli,
                                      const SimOverrides& ov) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open checkpoint %s\n", path.c_str()); return false; }

    // Header
    uint32_t magic, ver;
    int32_t cs, nc;
    float ct_f32 = 0.0f;
    double ct_f64 = 0.0;
    int32_t si, ci2, ts;
    uint8_t bools[4];
    uint32_t sp_sz;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x43454C4C) {
        fprintf(stderr, "Bad checkpoint magic: 0x%x\n", magic); fclose(f); return false;
    }
    fread(&ver, 4, 1, f);
    fread(&cs, 4, 1, f);
    // v5+: cur_time is f64 (8 bytes). v2-v4: cur_time is f32 (4 bytes).
    if (ver >= 5) {
        fread(&ct_f64, 8, 1, f);
    } else {
        fread(&ct_f32, 4, 1, f);
        ct_f64 = (double)ct_f32;
    }
    fread(&nc, 4, 1, f);
    fread(&si, 4, 1, f);
    fread(&ci2, 4, 1, f);
    fread(&ts, 4, 1, f);
    fread(bools, 1, 4, f);
    fread(&sp_sz, 4, 1, f);
    // Read SimParams block, upconverting pre-v6 layouts (f32 scalars) to f64
    // in-place. Supported layouts:
    //   v6 (sp_sz=144): native f64 struct — fread straight into params.
    //   v5 (sp_sz=88):  sim_v2 pre-f64 layout with int ordering quirks.
    //   v4 (sp_sz=72 or 92): baseline cell_sim layout. Includes a few extra
    //                        fields (soft_cell, v_A_sigma, adhesion_J) that
    //                        sim_v2 currently ignores or stores via per-cell
    //                        sidecar arrays after the cell block.
    //   v3 (sp_sz=72):  old baseline without the motility enum. Same field
    //                   layout up through subdomain_padding.
    // If the buffer is a known size we unpack by offset. Otherwise we error.
    std::vector<uint8_t> sp_buf(sp_sz);
    if (sp_sz > 0) fread(sp_buf.data(), 1, sp_sz, f);

    auto unpack_i32 = [&](size_t off) -> int32_t {
        int32_t v; std::memcpy(&v, sp_buf.data() + off, 4); return v;
    };
    auto unpack_u32 = [&](size_t off) -> uint32_t {
        uint32_t v; std::memcpy(&v, sp_buf.data() + off, 4); return v;
    };
    auto unpack_f32 = [&](size_t off) -> float {
        float v; std::memcpy(&v, sp_buf.data() + off, 4); return v;
    };

    if (ver >= 6 && sp_sz == sizeof(SimParams)) {
        // Native layout — direct copy.
        std::memcpy(&params, sp_buf.data(), sp_sz);
    } else if (ver == 5 && sp_sz == 88) {
        // sim_v2 v5 (all f32 scalars).
        // Layout: Nx,Ny (i32), dx,dy,dt,t_end,lambda,gamma,kappa,target_radius,
        //         mu,v_A,xi,tau,subdomain_padding (f32), halo (i32),
        //         save_interval, print_interval, trajectory_samples (i32),
        //         seed, polarity_seed (u32), abp (bool + pad).
        params.Nx = unpack_i32(0);
        params.Ny = unpack_i32(4);
        params.dx = unpack_f32(8);
        params.dy = unpack_f32(12);
        params.dt = unpack_f32(16);
        params.t_end = unpack_f32(20);
        params.lambda = unpack_f32(24);
        params.gamma = unpack_f32(28);
        params.kappa = unpack_f32(32);
        params.target_radius = unpack_f32(36);
        params.mu = unpack_f32(40);
        params.v_A = unpack_f32(44);
        params.xi = unpack_f32(48);
        params.tau = unpack_f32(52);
        params.subdomain_padding = unpack_f32(56);
        params.halo = unpack_i32(60);
        params.save_interval = unpack_i32(64);
        params.print_interval = unpack_i32(68);
        params.trajectory_samples = unpack_i32(72);
        params.seed = unpack_u32(76);
        params.polarity_seed = unpack_u32(80);
        params.abp = sp_buf[84] != 0;
    } else if ((ver == 3 || ver == 4) && (sp_sz == 72 || sp_sz == 92)) {
        // Baseline cell_sim layout (production / cluster checkpoints).
        // Fixed offsets through subdomain_padding; fields after that
        // (motility_model, v_A_sigma, soft_cell_id, gamma_soft, adhesion_J)
        // are baseline-only and handled below.
        params.Nx = unpack_i32(0);
        params.Ny = unpack_i32(4);
        params.dx = unpack_f32(8);
        params.dy = unpack_f32(12);
        params.dt = unpack_f32(16);
        params.t_end = unpack_f32(20);
        // offset 24 is baseline's SimParams.save_interval (an int) — we read
        // it so the simulator can honour the stored cadence, though CLI
        // overrides take precedence below.
        params.save_interval = unpack_i32(24);
        params.lambda = unpack_f32(28);
        params.gamma = unpack_f32(32);
        params.kappa = unpack_f32(36);
        params.target_radius = unpack_f32(40);
        params.mu = unpack_f32(44);
        params.v_A = unpack_f32(48);
        params.xi = unpack_f32(52);
        params.tau = unpack_f32(56);
        params.halo = unpack_i32(60);
        // offset 64 is baseline's min_subdomain_size — ignored.
        // offset 68 is baseline's subdomain_padding. DO NOT trust it: production
        // cluster checkpoints store stale/wrong values (e.g. 2.5) that don't
        // match the effective bbox sizes in the checkpoint. Interpreting 2.5 as
        // a fraction-of-R would yield pad = 2.5 * 49 = 122 pixels, causing the
        // first pre_step resize (step%10==0) to catastrophically expand every
        // tile and corrupt the loaded phi field (cells dewet and merge within
        // ~0.02τ). Keep sim_v2's default (0.6) or respect --subdomain-padding.
        float baseline_subdom_pad = unpack_f32(68);
        (void)baseline_subdom_pad;
        if (sp_sz >= 76) {
            // motility_model: enum class (int32). 0 = RunAndTumble, 1 = ABP.
            params.abp = (unpack_i32(72) == 1);
        }
        // v_A_sigma (offset 76), soft_cell_id (80), gamma_soft (84),
        // adhesion_J (88) are baseline features not yet in sim_v2.
        // They are ignored here; if present (sp_sz == 92) they are silently
        // dropped. The per-cell v_A sidecar (VA_A block after cells) still
        // carries real per-cell values when the run used --v-A-sigma.
        params.print_interval = 100;      // default; CLI override applied below
        params.trajectory_samples = 0;    // default; CLI override applied below
    } else {
        fprintf(stderr, "Unsupported checkpoint: v%u, SimParams size %u "
                        "(expected v6 sp=%zu, v5 sp=88, or v3/v4 sp=72|92)\n",
                ver, sp_sz, sizeof(SimParams));
        fclose(f); return false;
    }

    step_count = cs;
    cur_time = ct_f64;

    // Apply CLI overrides on top of the loaded params
    if (ov.t_end)              params.t_end = cli.t_end;
    if (ov.dt)                 params.dt = cli.dt;
    if (ov.v_A)                params.v_A = cli.v_A;
    if (ov.tau)                params.tau = cli.tau;
    if (ov.gamma)              params.gamma = cli.gamma;
    if (ov.kappa)              params.kappa = cli.kappa;
    if (ov.mu)                 params.mu = cli.mu;
    if (ov.xi)                 params.xi = cli.xi;
    if (ov.lambda)             params.lambda = cli.lambda;
    if (ov.target_radius)      params.target_radius = cli.target_radius;
    if (ov.subdomain_padding)  params.subdomain_padding = cli.subdomain_padding;
    if (ov.save_interval)      params.save_interval = cli.save_interval;
    if (ov.print_interval)     params.print_interval = cli.print_interval;
    if (ov.trajectory_samples) params.trajectory_samples = cli.trajectory_samples;
    if (ov.seed)               params.seed = cli.seed;
    if (ov.polarity_seed)      params.polarity_seed = cli.polarity_seed;
    if (ov.abp)                params.abp = cli.abp;

    // Populate h_cells from checkpoint cells.
    // Checkpoint stores (x0,y0,x1,y1) = inner region without halo.
    // sim_v2 uses (ox, oy, w, h) = full tile including halo.
    int halo = params.halo;
    h_cells.resize(nc);
    std::vector<float> ck_cx(nc), ck_cy(nc), ck_vx(nc), ck_vy(nc), ck_vol(nc);
    std::vector<int> ck_w(nc), ck_h(nc);
    std::vector<std::vector<float>> ck_phi(nc);

    for (int i = 0; i < nc; i++) {
        int32_t cid; int32_t x0, y0, x1, y1;
        float cx, cy, vx, vy, vol;
        fread(&cid, 4, 1, f);
        fread(&x0, 4, 1, f); fread(&y0, 4, 1, f);
        fread(&x1, 4, 1, f); fread(&y1, 4, 1, f);
        fread(&cx, 4, 1, f); fread(&cy, 4, 1, f);
        fread(&vx, 4, 1, f); fread(&vy, 4, 1, f);
        fread(&vol, 4, 1, f);

        int w = (x1 - x0) + 2 * halo;
        int h = (y1 - y0) + 2 * halo;
        int ox = x0 - halo;
        int oy = y0 - halo;

        h_cells[i].cx = cx; h_cells[i].cy = cy;
        h_cells[i].radius = params.target_radius;
        h_cells[i].gamma = params.gamma;
        h_cells[i].v_A = params.v_A;
        h_cells[i].ox = ox; h_cells[i].oy = oy;
        h_cells[i].w = w;   h_cells[i].h = h;

        ck_cx[i] = cx; ck_cy[i] = cy; ck_vx[i] = vx; ck_vy[i] = vy;
        ck_vol[i] = vol; ck_w[i] = w; ck_h[i] = h;
        ck_phi[i].resize((size_t)w * h);
        fread(ck_phi[i].data(), sizeof(float), (size_t)w * h, f);
    }

    // Optional per-cell magic-tagged arrays
    std::vector<float> per_vA, per_gamma, per_radius, per_polar_theta;
    while (true) {
        long pos = ftell(f);
        uint32_t m;
        if (fread(&m, 4, 1, f) != 1) break;
        int32_t count = 0;
        if (m == 0x56415F41 /* 'VA_A' */ ||
            m == 0x47414D41 /* 'GAMA' */ ||
            m == 0x52414449 /* 'RADI' */ ||
            m == 0x504F4C52 /* 'POLR' */) {
            fread(&count, 4, 1, f);
            std::vector<float> data(count);
            fread(data.data(), sizeof(float), count, f);
            if      (m == 0x56415F41) per_vA = std::move(data);
            else if (m == 0x47414D41) per_gamma = std::move(data);
            else if (m == 0x52414449) per_radius = std::move(data);
            else                      per_polar_theta = std::move(data);
        } else {
            fseek(f, pos, SEEK_SET);
            break;
        }
    }
    fclose(f);

    // Allocate and upload
    alloc_gpu();

    int n = cells.num_cells;
    size_t page = cells.slot_size;
    std::vector<int> h_ox(n), h_oy(n), h_w(n), h_h(n);
    std::vector<float> h_tg(n), h_tgb(n), h_ta(n), h_vc(n), h_tr(n);
    std::vector<float> h_vA(n), h_px(n), h_py(n), h_th(n);

    // RNG for init polarity angle (CPU) — used as fallback when the
    // checkpoint lacks a POLR block (legacy v6 checkpoints).
    srand(params.seed ? params.seed : 42);

    // If user supplied --gamma on resume, it overrides the checkpoint's per-cell array.
    // Bare numeric spec already updated params.gamma; apply_gamma_spec writes per-cell.
    bool user_set_gamma = !gamma_spec.empty();
    if (user_set_gamma) {
        apply_gamma_spec();  // writes h_cells[i].gamma
        per_gamma.clear();   // discard checkpoint's per-cell gamma
    }

    for (int i = 0; i < n; i++) {
        auto& c = h_cells[i];
        double g = (i < (int)per_gamma.size()) ? (double)per_gamma[i] : c.gamma;
        double R = (i < (int)per_radius.size()) ? (double)per_radius[i] : c.radius;
        double vA = (i < (int)per_vA.size()) ? (double)per_vA[i] : c.v_A;
        c.gamma = g; c.radius = R; c.v_A = vA;
        h_ox[i] = c.ox; h_oy[i] = c.oy;
        h_w[i] = c.w; h_h[i] = c.h;
        h_tg[i] = (float)(2.0 * g);
        h_tgb[i] = (float)(2.0 * g * params.bulk_coeff());
        h_ta[i] = (float)(M_PI * R * R);
        h_vc[i] = (float)(params.mu / (M_PI * R * R));
        h_tr[i] = (float)R;
        h_vA[i] = (float)vA;
        // Polarity: prefer checkpoint's persisted theta (POLR block). If the
        // checkpoint predates POLR (legacy v6), fall back to a fresh random
        // angle — same behaviour as pre-fix sim_v2.
        float theta;
        if (i < (int)per_polar_theta.size()) {
            theta = per_polar_theta[i];
        } else {
            theta = (float)(rand() % 10000) / 10000.0f * 2.0f * (float)M_PI;
        }
        h_th[i] = theta;
        h_px[i] = cosf(theta); h_py[i] = sinf(theta);
    }

    CK(cudaMemcpy(cells.offsets_x, h_ox.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.offsets_y, h_oy.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.widths, h_w.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.heights, h_h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.two_gamma, h_tg.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.two_gamma_bulk, h_tgb.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.tgt_area, h_ta.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.vol_coeff, h_vc.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.tgt_radius, h_tr.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.v_A_cell, h_vA.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_x, h_px.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_y, h_py.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.polar_theta, h_th.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Upload phi fields + dynamics (centroids, velocities, volumes)
    for (int i = 0; i < n; i++) {
        CK(cudaMemcpy(cells.phi_pool + (size_t)i * page,
                      ck_phi[i].data(), (size_t)ck_w[i] * ck_h[i] * sizeof(float),
                      cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(cells.centroids_x, ck_cx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.centroids_y, ck_cy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.velocities_x, ck_vx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.velocities_y, ck_vy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.volumes, ck_vol.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cache_w = cache_h = 0;
    for (auto& c : h_cells) {
        if (c.w > cache_w) cache_w = c.w;
        if (c.h > cache_h) cache_h = c.h;
    }

    // Set ref = current centroid (no drift accumulated across resume)
    CK(cudaMemcpy(cells.ref_x, ck_cx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cells.ref_y, ck_cy.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    finalize_init();
    CK(cudaDeviceSynchronize());

    printf("[SIM] resumed from %s: step=%d, t=%.4f, %d cells, %dx%d, t_end=%.1f\n",
           path.c_str(), step_count, cur_time, n, params.Nx, params.Ny, params.t_end);
    return true;
}

// ---------------------------------------------------------------------------
// Single step
// ---------------------------------------------------------------------------
void Simulation::step() {
    launch_polar(cells, params);
    launch_pre_step(cells, params, step_count, cache_w, cache_h);
    if (step_count % 10 == 0)
        launch_hash_build(cells, params.Nx, params.Ny);
    launch_fused(cells, params, cache_w, cache_h, step_count);
    launch_swap(cells, params.Nx, params.Ny);
#ifdef DEBUG_CUDA
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] step %d: %s\n", step_count, cudaGetErrorString(err));
        exit(1);
    }
#endif
    step_count++;
    cur_time += params.dt;
}

// ---------------------------------------------------------------------------
// Run loop
// ---------------------------------------------------------------------------
void Simulation::run() {
    int target_step = std::max(step_count, (int)(params.t_end / params.dt));
    int total = target_step - step_count;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Open trajectory file. Header is written when the file is empty, regardless
    // of step_count — resumes into a fresh output_dir still need the header so
    // downstream tools (cell_analyze) can parse domain/N/tau/v_A.
    // Skipped entirely when traj_every==0 (user passed --trajectory-samples 0)
    // so disabled runs leave no trace on disk.
    if (traj_every > 0) {
        std::string tp = out_dir + "/trajectory.txt";
        traj_fp = fopen(tp.c_str(), "a");
        if (traj_fp) {
            fseek(traj_fp, 0, SEEK_END);
            long pos = ftell(traj_fp);
            if (pos == 0) {
                fprintf(traj_fp, "# Trajectory data\n");
                fprintf(traj_fp, "# Format: time cell_id x y vx vy px py theta v_A_i L_n volume\n");
                fprintf(traj_fp, "# v_A=%.6f N=%d Lx=%d Ly=%d dim=2 dt=%.6f tau=%.4f\n",
                        params.v_A, cells.num_cells, params.Nx, params.Ny, params.dt, params.tau);
            }
        }
    }

    while (step_count < target_step) {
        step();
        if (params.print_interval > 0 && step_count % params.print_interval == 0)
            print_status();
        if (params.save_interval > 0 && step_count % params.save_interval == 0)
            save_checkpoint(out_dir);
        if (checkpoint_interval > 0 && step_count % checkpoint_interval == 0)
            save_checkpoint(out_dir);
        if (traj_fp && traj_every > 0 && step_count % traj_every == 0)
            write_trajectory();
        if (vtk_interval > 0 && step_count % vtk_interval == 0)
            write_vtk();
    }
    if (traj_fp) { fclose(traj_fp); traj_fp = nullptr; }

    if (save_final_checkpoint) save_checkpoint(out_dir);

    CK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();
    int denom = std::max(1, total);
    printf("[SIM] Done: %d steps, t=%.2f, wall=%.3fs (%.3f ms/step)\n",
           total, cur_time, wall, wall * 1000.0 / denom);
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------
void Simulation::print_status() {
    int n = cells.num_cells;
    std::vector<float> vols(n);
    CK(cudaMemcpy(vols.data(), cells.volumes, n * sizeof(float), cudaMemcpyDeviceToHost));
    float avg = 0; for (float v : vols) avg += v; avg /= n;
    float tgt = params.target_area();
    printf("step=%d t=%.2f avg_vol=%.1f (target=%.1f, err=%.2f%%)\n",
           step_count, cur_time, avg, tgt, 100.0f * (avg - tgt) / tgt);
}

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------
void Simulation::write_trajectory() {
    if (!traj_fp) return;
    int n = cells.num_cells;
    std::vector<float> cx(n), cy(n), vx(n), vy(n), px(n), py(n), vA(n), vol(n), per(n);
    CK(cudaMemcpy(cx.data(), cells.centroids_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cy.data(), cells.centroids_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vx.data(), cells.velocities_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vy.data(), cells.velocities_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(px.data(), cells.polar_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(py.data(), cells.polar_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vA.data(), cells.v_A_cell, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vol.data(), cells.volumes, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(per.data(), cells.perimeters, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        float theta = atan2f(py[i], px[i]);
        float Ln = per[i] / (2.0f * (float)M_PI * params.target_radius);
        fprintf(traj_fp, "%.6f %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                cur_time, i, cx[i], cy[i], vx[i], vy[i],
                px[i], py[i], theta, vA[i], Ln, vol[i]);
    }
    fflush(traj_fp);
}

// ---------------------------------------------------------------------------
// VTK output (legacy BINARY STRUCTURED_POINTS)
// ---------------------------------------------------------------------------
// Composite phase-field grid: per voxel, max(φ_i) across all cells.
// Binary format chosen over ASCII for ~10× smaller files and ~5× faster
// writes. Consumers: ParaView (native), pyvista, vtk_viewer.
//
// VTK legacy binary requires BIG-ENDIAN f32 payloads regardless of host
// byte order (see https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
// §6). We byteswap on Windows/Linux x86 hosts before writing.
//
// Scatter is host-side: phi pool is downloaded tile-by-tile and composited
// onto an Nx*Ny float buffer. This is intentionally simple — VTK output is
// off by default (vtk_interval=0), so per-frame cost here is not on the
// critical path. When enabled for visualisation it's still << one sim step.
// ---------------------------------------------------------------------------
void Simulation::write_vtk() {
    CK(cudaDeviceSynchronize());
    int n = cells.num_cells;
    int Nx = params.Nx, Ny = params.Ny;
    int halo = params.halo;

    std::vector<int> h_ox(n), h_oy(n), h_w(n), h_h(n);
    CK(cudaMemcpy(h_ox.data(), cells.offsets_x, n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_oy.data(), cells.offsets_y, n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_w.data(),  cells.widths,    n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_h.data(),  cells.heights,   n * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<float*> h_phi(n);
    CK(cudaMemcpy(h_phi.data(), cells.phi_ptrs, n * sizeof(float*),
                  cudaMemcpyDeviceToHost));

    // Global composite buffer (zero-initialised).
    std::vector<float> grid((size_t)Nx * Ny, 0.0f);

    for (int i = 0; i < n; i++) {
        int w = h_w[i], h = h_h[i];
        int ox = h_ox[i], oy = h_oy[i];
        std::vector<float> tile((size_t)w * h);
        CK(cudaMemcpy(tile.data(), h_phi[i], (size_t)w * h * sizeof(float),
                      cudaMemcpyDeviceToHost));
        // Scatter inner region (strip halo) with periodic wrap, taking max.
        for (int ly = halo; ly < h - halo; ly++) {
            int gy = ((oy + ly) % Ny + Ny) % Ny;
            for (int lx = halo; lx < w - halo; lx++) {
                int gx = ((ox + lx) % Nx + Nx) % Nx;
                float v = tile[(size_t)ly * w + lx];
                float& g = grid[(size_t)gy * Nx + gx];
                if (v > g) g = v;
            }
        }
    }

    // Byte-swap f32 → big-endian for VTK legacy payload.
    auto swap_f32 = [](float f) {
        uint32_t u; std::memcpy(&u, &f, 4);
        u = ((u & 0x000000FFu) << 24) |
            ((u & 0x0000FF00u) << 8)  |
            ((u & 0x00FF0000u) >> 8)  |
            ((u & 0xFF000000u) >> 24);
        std::memcpy(&f, &u, 4); return f;
    };
    std::vector<float> be(grid.size());
    for (size_t k = 0; k < grid.size(); k++) be[k] = swap_f32(grid[k]);

    char fn[512];
    snprintf(fn, sizeof(fn), "%s/output_%06d.vtk", out_dir.c_str(), step_count);
    FILE* f = fopen(fn, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", fn); return; }
    // Header: ASCII (newline-terminated lines), no trailing spaces.
    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "cell_sim phase-field composite step=%d t=%.6f\n",
            step_count, cur_time);
    fprintf(f, "BINARY\n");
    fprintf(f, "DATASET STRUCTURED_POINTS\n");
    fprintf(f, "DIMENSIONS %d %d 1\n", Nx, Ny);
    fprintf(f, "ORIGIN 0 0 0\n");
    fprintf(f, "SPACING 1 1 1\n");
    fprintf(f, "POINT_DATA %lld\n", (long long)Nx * Ny);
    fprintf(f, "SCALARS phi float 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");
    fwrite(be.data(), sizeof(float), be.size(), f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Checkpoint (binary)
// ---------------------------------------------------------------------------
void Simulation::save_checkpoint(const std::string& dir) {
    CK(cudaDeviceSynchronize());
    int n = cells.num_cells;
    int halo = params.halo;

    std::vector<int> h_ox(n), h_oy(n), h_w(n), h_h(n);
    std::vector<float> h_cx(n), h_cy(n), h_vx(n), h_vy(n), h_vol(n);
    CK(cudaMemcpy(h_ox.data(), cells.offsets_x, n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_oy.data(), cells.offsets_y, n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_w.data(), cells.widths, n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_h.data(), cells.heights, n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_cx.data(), cells.centroids_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_cy.data(), cells.centroids_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_vx.data(), cells.velocities_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_vy.data(), cells.velocities_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_vol.data(), cells.volumes, n * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float*> h_phi(n);
    CK(cudaMemcpy(h_phi.data(), cells.phi_ptrs, n * sizeof(float*), cudaMemcpyDeviceToHost));

    char fn[512];
    snprintf(fn, sizeof(fn), "%s/checkpoint.bin", dir.c_str());
    FILE* f = fopen(fn, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", fn); return; }

    // Header (v6: cur_time is f64, SimParams scalars are f64)
    uint32_t magic = 0x43454C4C;
    uint32_t ver = 6;
    int32_t cs = step_count;
    double ct = cur_time;
    int32_t nc = n;
    int32_t si = params.save_interval;
    int32_t ci2 = 0;
    int32_t ts = params.trajectory_samples;
    uint8_t bools[4] = {0, 0, 0, 0};
    uint32_t sp_sz = sizeof(SimParams);

    fwrite(&magic, 4, 1, f);
    fwrite(&ver, 4, 1, f);
    fwrite(&cs, 4, 1, f);
    fwrite(&ct, 8, 1, f);  // v5: double
    fwrite(&nc, 4, 1, f);
    fwrite(&si, 4, 1, f);
    fwrite(&ci2, 4, 1, f);
    fwrite(&ts, 4, 1, f);
    fwrite(bools, 1, 4, f);
    fwrite(&sp_sz, 4, 1, f);
    fwrite(&params, sp_sz, 1, f);

    for (int i = 0; i < n; i++) {
        int bx0 = h_ox[i] + halo, by0 = h_oy[i] + halo;
        int bx1 = h_ox[i] + h_w[i] - halo, by1 = h_oy[i] + h_h[i] - halo;
        int fsz = h_w[i] * h_h[i];

        int32_t cid = i;
        fwrite(&cid, 4, 1, f);
        fwrite(&bx0, 4, 1, f); fwrite(&by0, 4, 1, f);
        fwrite(&bx1, 4, 1, f); fwrite(&by1, 4, 1, f);
        fwrite(&h_cx[i], 4, 1, f); fwrite(&h_cy[i], 4, 1, f);
        fwrite(&h_vx[i], 4, 1, f); fwrite(&h_vy[i], 4, 1, f);
        fwrite(&h_vol[i], 4, 1, f);

        std::vector<float> buf(fsz);
        CK(cudaMemcpy(buf.data(), h_phi[i], fsz * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(buf.data(), sizeof(float), fsz, f);
    }

    // Per-cell arrays (magic-tagged, same format as baseline)
    auto write_per_cell = [&](uint32_t m, const float* dev_ptr) {
        std::vector<float> h(n);
        CK(cudaMemcpy(h.data(), dev_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(&m, 4, 1, f);
        int32_t count = n;
        fwrite(&count, 4, 1, f);
        fwrite(h.data(), sizeof(float), n, f);
    };
    // two_gamma stores 2*gamma; convert back to gamma for round-trip
    {
        std::vector<float> h_tg(n), h_g(n);
        CK(cudaMemcpy(h_tg.data(), cells.two_gamma, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) h_g[i] = 0.5f * h_tg[i];
        uint32_t m = 0x47414D41; // 'GAMA'
        fwrite(&m, 4, 1, f);
        int32_t count = n;
        fwrite(&count, 4, 1, f);
        fwrite(h_g.data(), sizeof(float), n, f);
    }
    write_per_cell(0x52414449 /* 'RADI' */, cells.tgt_radius);
    write_per_cell(0x56415F41 /* 'VA_A' */, cells.v_A_cell);
    // 'POLR' — persisted polarity angle (rad). Added post-cutover so that
    // resumes preserve the motility state rather than re-seeding to random
    // angles (which would scramble the first ~τ of any motile resume).
    // Legacy v6 checkpoints without POLR fall back to the random init
    // path in init_from_checkpoint(), preserving backward compatibility.
    write_per_cell(0x504F4C52 /* 'POLR' */, cells.polar_theta);

    fclose(f);
    printf("Saved checkpoint: step=%d, t=%.4f, cells=%d (%s)\n", cs, ct, n, fn);
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------
void Simulation::cleanup() {
    auto cf = [](auto& p) { if (p) { cudaFree(p); p = nullptr; } };
    cf(cells.phi_pool); cf(cells.phi_ptrs); cf(cells.phi_out_ptrs);
    cf(cells.offsets_x); cf(cells.offsets_y);
    cf(cells.widths); cf(cells.heights);
    cf(cells.old_widths); cf(cells.old_heights);
    cf(cells.shift_x); cf(cells.shift_y);
    cf(cells.velocities_x); cf(cells.velocities_y);
    cf(cells.volumes); cf(cells.volume_devs);
    cf(cells.centroids_x); cf(cells.centroids_y);
    cf(cells.ref_x); cf(cells.ref_y);
    cf(cells.perimeters);
    cf(cells.moment_x); cf(cells.moment_y);
    cf(cells.polar_x); cf(cells.polar_y); cf(cells.polar_theta);
    cf(cells.two_gamma); cf(cells.two_gamma_bulk);
    cf(cells.vol_coeff); cf(cells.tgt_area);
    cf(cells.tgt_radius); cf(cells.v_A_cell);
    cf(cells.nbr_list); cf(cells.nbr_count);
    cf(cells.hash_ids); cf(cells.hash_counts);
    cf(cells.d_max_wh); cf(cells.rng_states);
}
