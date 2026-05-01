//! Phase-field cell simulator — CPU reference, f64 throughout.
//!
//! Ported line-for-line from `cpp/simulation/tests/python/cpu_reference.py`.
//! Correctness is the priority. Each cell owns a full `Ny × Nx` periodic
//! field; no tiles, no halos. The step has TWO passes:
//!
//!   1. Compute v^n for all cells from φ^n (uses S_total − φᵢ²).
//!   2. Update φᵢ for all cells using the v^n from pass 1.
//!
//! Stencils (periodic, central / 9-point isotropic) match the Python ref:
//!   E[iy,ix] = φ[iy, ix+1],  W = φ[iy, ix−1]
//!   N[iy,ix] = φ[iy+1, ix],  S = φ[iy−1, ix]
//!   NE/NW/SE/SW correspondingly.
//!   lap = (4·(E+W+N+S) + (NE+NW+SE+SW) − 20·φ) / (6·h·h)
//!   gx  = (E − W) / (2·dx)
//!   gy  = (N − S) / (2·dy)
//!
//! Update:
//!   var_deriv = −2γ·lap + 60γ/λ²·φ(1−φ)(1−2φ) − 4(μ/A₀)(A₀−V)·φ + 60κ/λ²·φ·S_local
//!   v_n_x     = mc·Σ(φ·gx·S_local)·dA + v_A·pₓ,   mc = 60κ/(ξ·λ²)
//!   φ_new     = φ + dt·(−0.5·var_deriv − (v_n_x·gx + v_n_y·gy))
//!   vol_new   = Σ φ_new² · dA

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Per-cell PRNG: xoshiro256+ seeded via splitmix64 from (polarity_seed, cid).
// Deterministic and stable across platforms / Rust versions.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Xoshiro256Plus {
    s: [u64; 4],
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl Xoshiro256Plus {
    /// Seed for cell `cid` from a master `polarity_seed` (mirrors common
    /// "XOR + splitmix" practice). Each cell gets four splitmix64 outputs
    /// as its state, so streams are independent across cells.
    pub fn seed_for_cell(polarity_seed: u64, cid: u64) -> Self {
        let mut sm = polarity_seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(cid.wrapping_mul(0xBF58_476D_1CE4_E5B9));
        let s0 = splitmix64(&mut sm);
        let s1 = splitmix64(&mut sm);
        let s2 = splitmix64(&mut sm);
        let s3 = splitmix64(&mut sm);
        let mut me = Self { s: [s0, s1, s2, s3] };
        // Avoid all-zero state (xoshiro requires at least one non-zero).
        if me.s == [0; 4] { me.s[0] = 1; }
        // Burn a few values (xoshiro authors recommend it after splitmix).
        for _ in 0..4 { let _ = me.next_u64(); }
        me
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0].wrapping_add(self.s[3]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform double in [0, 1) with 53-bit mantissa.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Top 53 bits of a 64-bit random.
        let x = self.next_u64() >> 11;
        x as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }
}

#[derive(Debug, Clone)]
pub struct Params {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub dt: f64,
    pub lambd: f64,
    pub gamma: f64,
    pub kappa: f64,
    pub mu: f64,
    pub xi: f64,
    pub target_radius: f64,
}

impl Params {
    #[inline]
    pub fn target_area(&self) -> f64 {
        std::f64::consts::PI * self.target_radius * self.target_radius
    }
    #[inline]
    pub fn d_a(&self) -> f64 {
        self.dx * self.dy
    }
    /// `60·κ / (ξ·λ²)`
    #[inline]
    pub fn motility_coeff(&self) -> f64 {
        60.0 * self.kappa / (self.xi * self.lambd * self.lambd)
    }
}

#[derive(Debug, Clone)]
pub struct Cell {
    /// Full domain field, row-major (iy, ix), length `nx * ny`.
    pub phi: Vec<f64>,
    pub vx: f64,
    pub vy: f64,
    pub vol: f64,
    pub v_a: f64,
    /// Per-cell interface tension. Defaults to `Params.gamma` when no
    /// GAMA sidecar is present in the checkpoint.
    pub gamma: f64,
    /// Run-and-tumble polar angle (radians). `(px, py) = (cosθ, sinθ)` is
    /// recomputed from `theta` after every tumble.
    pub theta: f64,
    pub px: f64,
    pub py: f64,
    /// Per-cell independent PRNG.
    pub rng: Xoshiro256Plus,
}

/// Workspace buffers reused across steps. Allocate once.
pub struct Workspace {
    pub psq: Vec<Vec<f64>>,        // [n_cells][nx*ny]
    pub s_total: Vec<f64>,         // [nx*ny]
    pub phi_new: Vec<Vec<f64>>,    // [n_cells][nx*ny]
    pub v_n: Vec<(f64, f64)>,      // [n_cells]
    pub xm1: Vec<usize>,           // [nx]
    pub xp1: Vec<usize>,           // [nx]
    pub ym1: Vec<usize>,           // [ny]
    pub yp1: Vec<usize>,           // [ny]
}

impl Workspace {
    pub fn new(n_cells: usize, nx: usize, ny: usize) -> Self {
        let n = nx * ny;
        let xm1 = (0..nx).map(|i| (i + nx - 1) % nx).collect();
        let xp1 = (0..nx).map(|i| (i + 1) % nx).collect();
        let ym1 = (0..ny).map(|i| (i + ny - 1) % ny).collect();
        let yp1 = (0..ny).map(|i| (i + 1) % ny).collect();
        Self {
            psq: (0..n_cells).map(|_| vec![0.0; n]).collect(),
            s_total: vec![0.0; n],
            phi_new: (0..n_cells).map(|_| vec![0.0; n]).collect(),
            v_n: vec![(0.0, 0.0); n_cells],
            xm1, xp1, ym1, yp1,
        }
    }
}

/// A run-and-tumble event, recorded once per tumble for downstream analysis.
#[derive(Debug, Clone, Copy)]
pub struct TumbleEvent {
    pub t: f64,
    pub cid: u32,
    pub old_theta: f64,
    pub new_theta: f64,
}

/// Advance all cells by one `dt`. Mutates `cells` in place via buffer swap.
///
/// `tau` is the run-and-tumble persistence time. If `tau <= 0` or
/// `v_a == 0` for a given cell, no tumble check is performed for that cell
/// (matches the GPU sentinel).
///
/// Tumble events are appended to `events` (caller decides whether to keep
/// them). The PRNG draws happen BEFORE the velocity / phi passes, so the
/// `(px, py)` used in the step are the post-tumble values.
pub fn step(
    cells: &mut [Cell],
    p: &Params,
    w: &mut Workspace,
    tau: f64,
    sim_time_after_step: f64,
    events: &mut Vec<TumbleEvent>,
) {
    let nx = p.nx;
    let ny = p.ny;
    let n_pix = nx * ny;
    let n_cells = cells.len();

    // Pre-computed coefficients (global; gamma factors are per-cell).
    let inv_lambda2 = 1.0 / (p.lambd * p.lambd);
    let two_keff = 60.0 * p.kappa / (p.lambd * p.lambd);
    let area_target = p.target_area();
    let vc = p.mu / area_target;
    let mc = p.motility_coeff();
    let inv_2dx = 1.0 / (2.0 * p.dx);
    let inv_2dy = 1.0 / (2.0 * p.dy);
    let lap_inv = 1.0 / (6.0 * p.dx * p.dx);
    let dt = p.dt;
    let d_a = p.d_a();

    // ---- 0. Run-and-tumble polarity update (sequential, deterministic) ----
    // Per-cell independent stream; sequential over cells in checkpoint
    // order so tumble logs are reproducible. Two PRNG draws on tumble
    // (one to decide, one for the new angle) — same as the GPU kernel.
    if tau > 0.0 {
        let p_tumble = 1.0 - (-dt / tau).exp();
        for (i, c) in cells.iter_mut().enumerate() {
            if c.v_a == 0.0 { continue; }
            let u1 = c.rng.next_f64();
            if u1 < p_tumble {
                let u2 = c.rng.next_f64();
                let old_theta = c.theta;
                let new_theta = u2 * std::f64::consts::TAU;
                c.theta = new_theta;
                c.px = new_theta.cos();
                c.py = new_theta.sin();
                events.push(TumbleEvent {
                    t: sim_time_after_step,
                    cid: i as u32,
                    old_theta,
                    new_theta,
                });
            }
        }
    }

    // ---- 1. psq[i] = phi[i]² (parallel over cells) ----
    w.psq.par_iter_mut()
        .zip(cells.par_iter())
        .for_each(|(psq_i, c)| {
            for k in 0..n_pix {
                let v = c.phi[k];
                psq_i[k] = v * v;
            }
        });

    // ---- 2. S_total[k] = Σ_i psq[i][k] (parallel over chunks of pixels) ----
    // Order: cell-id ascending. Identical across runs.
    let psq = &w.psq;
    w.s_total
        .par_chunks_mut(4096)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let off = ci * 4096;
            for (j, slot) in chunk.iter_mut().enumerate() {
                let k = off + j;
                let mut s = 0.0_f64;
                for i in 0..n_cells {
                    s += psq[i][k];
                }
                *slot = s;
            }
        });

    // Borrow workspace pieces immutably for the rest of the step.
    let s_total = &w.s_total;
    let xm1 = &w.xm1;
    let xp1 = &w.xp1;
    let ym1 = &w.ym1;
    let yp1 = &w.yp1;

    // ---- 3. v_n[i] = mc · ∫(φᵢ · ∇φᵢ · S_local) dA + v_A · p̂ᵢ ----
    w.v_n
        .par_iter_mut()
        .zip(cells.par_iter())
        .zip(psq.par_iter())
        .for_each(|((vn, c), psq_i)| {
            let phi = &c.phi;
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            for iy in 0..ny {
                let row = iy * nx;
                let row_n = yp1[iy] * nx;
                let row_s = ym1[iy] * nx;
                for ix in 0..nx {
                    let k = row + ix;
                    let e = phi[row + xp1[ix]];
                    let v_w = phi[row + xm1[ix]];
                    let nn = phi[row_n + ix];
                    let ss = phi[row_s + ix];
                    let gx = (e - v_w) * inv_2dx;
                    let gy = (nn - ss) * inv_2dy;
                    let s_local = s_total[k] - psq_i[k];
                    let phi_k = phi[k];
                    sx += phi_k * gx * s_local;
                    sy += phi_k * gy * s_local;
                }
            }
            let vx_int = mc * sx * d_a;
            let vy_int = mc * sy * d_a;
            *vn = (vx_int + c.v_a * c.px, vy_int + c.v_a * c.py);
        });

    // ---- 4. φ_new[i] using v^n[i] (parallel over cells, fused stencil) ----
    let v_n = &w.v_n;
    w.phi_new
        .par_iter_mut()
        .zip(cells.par_iter())
        .zip(psq.par_iter())
        .enumerate()
        .for_each(|(i, ((phi_new_i, c), psq_i))| {
            let phi = &c.phi;
            let vd = area_target - c.vol;
            let (vxn, vyn) = v_n[i];
            // Per-cell gamma factors.
            let two_g = 2.0 * c.gamma;
            let tgb = 60.0 * c.gamma * inv_lambda2;
            for iy in 0..ny {
                let row = iy * nx;
                let row_n = yp1[iy] * nx;
                let row_s = ym1[iy] * nx;
                for ix in 0..nx {
                    let k = row + ix;
                    let xp = xp1[ix];
                    let xm = xm1[ix];
                    let e = phi[row + xp];
                    let v_w = phi[row + xm];
                    let nn = phi[row_n + ix];
                    let ss = phi[row_s + ix];
                    let ne = phi[row_n + xp];
                    let nw = phi[row_n + xm];
                    let se = phi[row_s + xp];
                    let sw = phi[row_s + xm];
                    let phi_k = phi[k];
                    let lap =
                        (4.0 * (e + v_w + nn + ss) + (ne + nw + se + sw) - 20.0 * phi_k) * lap_inv;
                    let gx = (e - v_w) * inv_2dx;
                    let gy = (nn - ss) * inv_2dy;
                    let one_minus = 1.0 - phi_k;
                    let one_minus_2 = 1.0 - 2.0 * phi_k;
                    let bulk = tgb * phi_k * one_minus * one_minus_2;
                    let constraint = -4.0 * vc * vd * phi_k;
                    let s_local = s_total[k] - psq_i[k];
                    let repulsion = two_keff * phi_k * s_local;
                    let var_deriv = -two_g * lap + bulk + constraint + repulsion;
                    let advection = vxn * gx + vyn * gy;
                    phi_new_i[k] = phi_k + dt * (-0.5 * var_deriv - advection);
                }
            }
        });

    // ---- 5. Swap buffers, recompute volume from φ_new² ----
    for i in 0..n_cells {
        std::mem::swap(&mut cells[i].phi, &mut w.phi_new[i]);
        let (vxn, vyn) = w.v_n[i];
        cells[i].vx = vxn;
        cells[i].vy = vyn;
    }

    // Volume update — parallel over cells.
    cells.par_iter_mut().for_each(|c| {
        let mut s = 0.0_f64;
        for &v in &c.phi {
            s += v * v;
        }
        c.vol = s * d_a;
    });
}
