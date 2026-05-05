//! Observable computation for cell simulation trajectories.
//!
//! All functions take `UnwrappedPositions` (or subsets) and return
//! structured results suitable for JSON serialization.

use super::io::UnwrappedPositions;
use serde::Serialize;
use std::f64::consts::PI;

// ============================================================================
// MSD
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct MsdResult {
    pub lag_times: Vec<f64>,
    pub values: Vec<f64>,
    /// Cell-0 MSD(Δt) — the single soft cell in Palmieri runs.
    pub cell0_values: Vec<f64>,
}

/// Compute ensemble-averaged MSD from unwrapped positions.
///
/// Uses multi-origin averaging over the first half of the trajectory.
/// Also computes cell-0 MSD separately for Palmieri soft/ctrl comparison.
pub fn compute_msd(pos: &UnwrappedPositions) -> MsdResult {
    let n_times = pos.n_times;
    let n_cells = pos.n_cells;
    if n_times < 2 {
        return MsdResult {
            lag_times: vec![],
            values: vec![],
            cell0_values: vec![],
        };
    }

    let max_lag = n_times / 2;
    let dt = if n_times > 1 {
        pos.times[1] - pos.times[0]
    } else {
        1.0
    };

    // Find cell-0 index (by cell_id == 0)
    let cell0_idx = pos.cell_ids.iter().position(|&id| id == 0).unwrap_or(0);

    // Use all origins in the first half
    let n_origins = max_lag;
    let mut msd_sum = vec![0.0f64; max_lag];
    let mut msd_count = vec![0u64; max_lag];
    let mut cell0_msd_sum = vec![0.0f64; max_lag];

    for t0 in 0..n_origins {
        for lag in 1..max_lag {
            let ti = t0 + lag;
            if ti >= n_times {
                break;
            }
            let mut sum_dsq = 0.0;
            for i in 0..n_cells {
                let dx = pos.positions[ti][i][0] - pos.positions[t0][i][0];
                let dy = pos.positions[ti][i][1] - pos.positions[t0][i][1];
                let dz = pos.positions[ti][i][2] - pos.positions[t0][i][2];
                let dsq = dx * dx + dy * dy + dz * dz;
                sum_dsq += dsq;
                if i == cell0_idx {
                    cell0_msd_sum[lag] += dsq;
                }
            }
            msd_sum[lag] += sum_dsq / n_cells as f64;
            msd_count[lag] += 1;
        }
    }

    let mut lag_times = Vec::with_capacity(max_lag - 1);
    let mut values = Vec::with_capacity(max_lag - 1);
    let mut cell0_values = Vec::with_capacity(max_lag - 1);
    for lag in 1..max_lag {
        if msd_count[lag] > 0 {
            lag_times.push(lag as f64 * dt);
            values.push(msd_sum[lag] / msd_count[lag] as f64);
            cell0_values.push(cell0_msd_sum[lag] / msd_count[lag] as f64);
        }
    }

    MsdResult { lag_times, values, cell0_values }
}

// ============================================================================
// Diffusion coefficient
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct DiffusionResult {
    pub d_eff: f64,
    pub fit_r2: f64,
}

/// Compute effective diffusion coefficient from long-time MSD slope.
///
/// D_eff = slope / 4 in 2D, fitted to the last `fit_frac` of the MSD curve.
pub fn compute_diffusion(msd: &MsdResult, fit_frac: f64) -> DiffusionResult {
    let n = msd.lag_times.len();
    if n < 5 {
        return DiffusionResult {
            d_eff: 0.0,
            fit_r2: 0.0,
        };
    }

    let start = ((n as f64 * (1.0 - fit_frac)).ceil() as usize).max(1);
    let t_fit = &msd.lag_times[start..];
    let y_fit = &msd.values[start..];
    let n_fit = t_fit.len();
    if n_fit < 2 {
        return DiffusionResult {
            d_eff: 0.0,
            fit_r2: 0.0,
        };
    }

    // Linear regression: y = a*t + b
    let sum_t: f64 = t_fit.iter().sum();
    let sum_y: f64 = y_fit.iter().sum();
    let sum_tt: f64 = t_fit.iter().map(|t| t * t).sum();
    let sum_ty: f64 = t_fit.iter().zip(y_fit.iter()).map(|(t, y)| t * y).sum();
    let nf = n_fit as f64;

    let denom = nf * sum_tt - sum_t * sum_t;
    if denom.abs() < 1e-30 {
        return DiffusionResult {
            d_eff: 0.0,
            fit_r2: 0.0,
        };
    }

    let slope = (nf * sum_ty - sum_t * sum_y) / denom;
    let intercept = (sum_y - slope * sum_t) / nf;
    let d_eff = (slope / 4.0).max(0.0);

    // R² goodness of fit
    let y_mean = sum_y / nf;
    let ss_tot: f64 = y_fit.iter().map(|y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = t_fit
        .iter()
        .zip(y_fit.iter())
        .map(|(t, y)| {
            let pred = slope * t + intercept;
            (y - pred).powi(2)
        })
        .sum();
    let fit_r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    DiffusionResult { d_eff, fit_r2 }
}

// ============================================================================
// MSD log-slope Δ(t)
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct LogSlopeResult {
    pub times: Vec<f64>,
    pub delta: Vec<f64>,
}

/// Compute Δ(t) = d(ln MSD)/d(ln t), the instantaneous diffusion exponent.
///
/// Δ = 2 → ballistic, Δ = 1 → diffusive, Δ < 1 → subdiffusive/caged.
pub fn msd_log_slope(msd: &MsdResult) -> LogSlopeResult {
    let mut times = Vec::new();
    let mut delta = Vec::new();

    let valid: Vec<(f64, f64)> = msd
        .lag_times
        .iter()
        .zip(msd.values.iter())
        .filter(|(&t, &v)| t > 0.0 && v > 0.0)
        .map(|(&t, &v)| (t.ln(), v.ln()))
        .collect();

    for i in 1..valid.len() {
        let dt = valid[i].0 - valid[i - 1].0;
        if dt.abs() < 1e-30 {
            continue;
        }
        let d = (valid[i].1 - valid[i - 1].1) / dt;
        let t_mid = ((valid[i].0 + valid[i - 1].0) / 2.0).exp();
        times.push(t_mid);
        delta.push(d);
    }

    LogSlopeResult { times, delta }
}

// ============================================================================
// Cage length
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct CageLengthResult {
    pub l_c: f64,
    pub t_star: f64,
}

/// Extract cage length from MSD plateau (minimum of Δ(t) near τ).
pub fn cage_length(msd: &MsdResult, tau: f64) -> CageLengthResult {
    let ls = msd_log_slope(msd);
    if ls.delta.is_empty() {
        return CageLengthResult {
            l_c: f64::NAN,
            t_star: f64::NAN,
        };
    }

    // Find minimum Δ in window [0.1τ, 5τ]
    let mut best_idx = 0;
    let mut best_val = f64::INFINITY;
    for (i, (&t, &d)) in ls.times.iter().zip(ls.delta.iter()).enumerate() {
        if t > 0.1 * tau && t < 5.0 * tau && d < best_val {
            best_val = d;
            best_idx = i;
        }
    }

    // Fallback: global minimum
    if best_val == f64::INFINITY {
        for (i, &d) in ls.delta.iter().enumerate() {
            if d < best_val {
                best_val = d;
                best_idx = i;
            }
        }
    }

    let t_star = ls.times[best_idx];

    // Interpolate MSD at t_star
    let l_c_sq = interp(t_star, &msd.lag_times, &msd.values);
    CageLengthResult {
        l_c: l_c_sq.sqrt(),
        t_star,
    }
}

// ============================================================================
// Non-Gaussian parameter α₂(t)
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct Alpha2Result {
    pub lag_times: Vec<f64>,
    pub values: Vec<f64>,
}

/// Compute α₂(Δt) = <Δr⁴> / (2<Δr²>²) - 1 (2D).
pub fn non_gaussian_parameter(pos: &UnwrappedPositions) -> Alpha2Result {
    let n_times = pos.n_times;
    let n_cells = pos.n_cells;
    if n_times < 2 || n_cells == 0 {
        return Alpha2Result {
            lag_times: vec![],
            values: vec![],
        };
    }

    let max_lag = n_times / 2;
    let dt = if n_times > 1 {
        pos.times[1] - pos.times[0]
    } else {
        1.0
    };
    let n_origins = max_lag;

    let mut r2_sum = vec![0.0f64; max_lag];
    let mut r4_sum = vec![0.0f64; max_lag];
    let mut count = vec![0u64; max_lag];

    for t0 in 0..n_origins {
        for lag in 1..max_lag {
            let ti = t0 + lag;
            if ti >= n_times {
                break;
            }
            for i in 0..n_cells {
                let dx = pos.positions[ti][i][0] - pos.positions[t0][i][0];
                let dy = pos.positions[ti][i][1] - pos.positions[t0][i][1];
                let dz = pos.positions[ti][i][2] - pos.positions[t0][i][2];
                let r2 = dx * dx + dy * dy + dz * dz;
                r2_sum[lag] += r2;
                r4_sum[lag] += r2 * r2;
                count[lag] += 1;
            }
        }
    }

    let mut lag_times = Vec::with_capacity(max_lag - 1);
    let mut values = Vec::with_capacity(max_lag - 1);
    for lag in 1..max_lag {
        if count[lag] > 0 {
            let mean_r2 = r2_sum[lag] / count[lag] as f64;
            let mean_r4 = r4_sum[lag] / count[lag] as f64;
            let denom = 2.0 * mean_r2 * mean_r2; // factor 2 for 2D
            let a2 = if denom > 0.0 {
                mean_r4 / denom - 1.0
            } else {
                0.0
            };
            lag_times.push(lag as f64 * dt);
            values.push(a2);
        }
    }

    Alpha2Result { lag_times, values }
}

// ============================================================================
// Self-overlap Q(t) and four-point susceptibility χ₄(t)
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct OverlapResult {
    pub lag_times: Vec<f64>,
    pub q_mean: Vec<f64>,
    pub chi4: Vec<f64>,
    pub tau_alpha: f64,
    pub beta: f64,
    pub fit_r2: f64,
}

/// Compute Q(t), χ₄(t), and stretched-exponential fit.
pub fn overlap_and_chi4(pos: &UnwrappedPositions, cage_radius: f64) -> OverlapResult {
    let n_times = pos.n_times;
    let n_cells = pos.n_cells;
    if n_times < 2 || n_cells == 0 {
        return OverlapResult {
            lag_times: vec![],
            q_mean: vec![],
            chi4: vec![],
            tau_alpha: f64::NAN,
            beta: f64::NAN,
            fit_r2: 0.0,
        };
    }

    let max_lag = n_times / 2;
    let dt = if n_times > 1 {
        pos.times[1] - pos.times[0]
    } else {
        1.0
    };
    let n_origins = max_lag;

    // Collect Q values per origin to compute variance for χ₄
    let mut q_per_origin: Vec<Vec<f64>> = vec![Vec::new(); max_lag];

    for t0 in 0..n_origins {
        for lag in 0..max_lag {
            let ti = t0 + lag;
            if ti >= n_times {
                break;
            }
            let mut overlap_count = 0u32;
            for i in 0..n_cells {
                let dx = pos.positions[ti][i][0] - pos.positions[t0][i][0];
                let dy = pos.positions[ti][i][1] - pos.positions[t0][i][1];
                let dz = pos.positions[ti][i][2] - pos.positions[t0][i][2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < cage_radius {
                    overlap_count += 1;
                }
            }
            q_per_origin[lag].push(overlap_count as f64 / n_cells as f64);
        }
    }

    let mut lag_times = Vec::with_capacity(max_lag);
    let mut q_mean = Vec::with_capacity(max_lag);
    let mut chi4 = Vec::with_capacity(max_lag);

    for lag in 0..max_lag {
        let vals = &q_per_origin[lag];
        if vals.is_empty() {
            continue;
        }
        let n = vals.len() as f64;
        let mean: f64 = vals.iter().sum::<f64>() / n;
        let var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

        lag_times.push(lag as f64 * dt);
        q_mean.push(mean);
        chi4.push(n_cells as f64 * var);
    }

    // Fit stretched exponential to Q(t)
    let (tau_alpha, beta, fit_r2) = fit_stretched_exponential(&lag_times, &q_mean);

    OverlapResult {
        lag_times,
        q_mean,
        chi4,
        tau_alpha,
        beta,
        fit_r2,
    }
}

// ============================================================================
// Static structure factor S(q)
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct StructureFactorResult {
    pub q_bins: Vec<f64>,
    pub s_q: Vec<f64>,
    pub q_star: f64,
}

/// Compute angularly-averaged S(q) from wrapped positions.
///
/// Averages over `n_frames` frames from the second half of the trajectory.
pub fn structure_factor(pos: &UnwrappedPositions, n_bins: usize, n_frames: usize) -> StructureFactorResult {
    let lx = pos.lx;
    let ly = pos.ly;
    let l_min = lx.min(ly);
    let q_max = 2.0 * PI * 20.0 / l_min;
    let dq = q_max / n_bins as f64;
    let dqx = 2.0 * PI / lx;
    let dqy = 2.0 * PI / ly;
    let nx_max = (q_max / dqx).ceil() as i32;
    let ny_max = (q_max / dqy).ceil() as i32;

    let mut s_sum = vec![0.0f64; n_bins];
    let mut counts = vec![0u64; n_bins];

    // Use frames from the second half
    let n_sq = n_frames.min(pos.n_times);
    let start = pos.n_times / 2;
    let step = if n_sq > 1 {
        (pos.n_times - start).max(1) / n_sq
    } else {
        1
    };

    let mut frames_used = 0;
    let mut t_idx = start;
    while t_idx < pos.n_times && frames_used < n_sq {
        // Wrap positions into [0, L)
        let n = pos.n_cells;
        let wrapped: Vec<[f64; 2]> = (0..n).map(|i| pos.wrapped(t_idx, i)).collect();

        for nx in -nx_max..=nx_max {
            let qx = nx as f64 * dqx;
            for ny in -ny_max..=ny_max {
                if nx == 0 && ny == 0 {
                    continue;
                }
                let qy = ny as f64 * dqy;
                let q_mag = (qx * qx + qy * qy).sqrt();
                if q_mag > q_max {
                    continue;
                }
                let b = (q_mag / dq) as usize;
                if b >= n_bins {
                    continue;
                }
                let mut rho_re = 0.0f64;
                let mut rho_im = 0.0f64;
                for p in &wrapped {
                    let phase = qx * p[0] + qy * p[1];
                    rho_re += phase.cos();
                    rho_im += phase.sin();
                }
                s_sum[b] += (rho_re * rho_re + rho_im * rho_im) / n as f64;
                counts[b] += 1;
            }
        }

        frames_used += 1;
        t_idx += step.max(1);
    }

    let q_bins: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * dq).collect();
    let mut s_q = vec![0.0; n_bins];
    for i in 0..n_bins {
        if counts[i] > 0 {
            s_q[i] = s_sum[i] / counts[i] as f64;
        }
    }

    // Find q* (first peak above q_min)
    let q_min = 0.02;
    let q_star = q_bins
        .iter()
        .zip(s_q.iter())
        .filter(|(&q, _)| q > q_min)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0.0, |(&q, _)| q);

    StructureFactorResult {
        q_bins,
        s_q,
        q_star,
    }
}

// ============================================================================
// Self-intermediate scattering function F_s(q*, t)
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct ScatteringResult {
    pub lag_times: Vec<f64>,
    pub fs: Vec<f64>,
    pub tau_alpha: f64,
    pub beta: f64,
    pub fit_r2: f64,
}

/// Compute F_s(q*, Δt) averaged over 4 q-vector orientations and multiple origins.
pub fn self_intermediate_scattering(pos: &UnwrappedPositions, q_star: f64) -> ScatteringResult {
    let n_times = pos.n_times;
    let n_cells = pos.n_cells;
    if n_times < 2 || n_cells == 0 || q_star <= 0.0 {
        return ScatteringResult {
            lag_times: vec![],
            fs: vec![],
            tau_alpha: f64::NAN,
            beta: f64::NAN,
            fit_r2: 0.0,
        };
    }

    let max_lag = n_times / 2;
    let dt = if n_times > 1 {
        pos.times[1] - pos.times[0]
    } else {
        1.0
    };

    // 4 q-vector orientations at |q| = q_star
    let q_vectors: [[f64; 2]; 4] = [
        [q_star, 0.0],
        [0.0, q_star],
        [q_star * (PI / 4.0).cos(), q_star * (PI / 4.0).sin()],
        [
            q_star * (3.0 * PI / 4.0).cos(),
            q_star * (3.0 * PI / 4.0).sin(),
        ],
    ];

    let n_origins = max_lag;
    let mut fs_sum = vec![0.0f64; max_lag];
    let mut fs_count = vec![0u64; max_lag];

    for t0 in 0..n_origins {
        for lag in 0..max_lag {
            let ti = t0 + lag;
            if ti >= n_times {
                break;
            }
            let mut fs_val = 0.0;
            for qv in &q_vectors {
                let mut cos_sum = 0.0;
                for i in 0..n_cells {
                    let dx = pos.positions[ti][i][0] - pos.positions[t0][i][0];
                    let dy = pos.positions[ti][i][1] - pos.positions[t0][i][1];
                    let phase = qv[0] * dx + qv[1] * dy;
                    cos_sum += phase.cos();
                }
                fs_val += cos_sum / n_cells as f64;
            }
            fs_sum[lag] += fs_val / q_vectors.len() as f64;
            fs_count[lag] += 1;
        }
    }

    let mut lag_times = Vec::with_capacity(max_lag);
    let mut fs = Vec::with_capacity(max_lag);
    for lag in 0..max_lag {
        if fs_count[lag] > 0 {
            lag_times.push(lag as f64 * dt);
            fs.push(fs_sum[lag] / fs_count[lag] as f64);
        }
    }

    let (tau_alpha, beta, fit_r2) = fit_stretched_exponential(&lag_times, &fs);

    ScatteringResult {
        lag_times,
        fs,
        tau_alpha,
        beta,
        fit_r2,
    }
}

// ============================================================================
// van Hove self-correlation G_s(Δx, t)
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct VanHoveResult {
    pub dx_bins: Vec<f64>,
    /// Map from lag_time → histogram values
    pub distributions: Vec<VanHoveLag>,
}

#[derive(Serialize, Clone, Debug)]
pub struct VanHoveLag {
    pub lag_time: f64,
    pub histogram: Vec<f64>,
}

/// Compute van Hove G_s(Δx, t) at selected lag fractions of τ.
pub fn van_hove(pos: &UnwrappedPositions, tau: f64, n_bins: usize) -> VanHoveResult {
    let n_times = pos.n_times;
    let n_cells = pos.n_cells;
    let dt = if n_times > 1 {
        pos.times[1] - pos.times[0]
    } else {
        1.0
    };

    if n_times < 2 || n_cells == 0 {
        return VanHoveResult {
            dx_bins: vec![],
            distributions: vec![],
        };
    }

    // Target lags: 0.1τ, τ, 3τ
    let target_lags: Vec<f64> = vec![0.1 * tau, tau, 3.0 * tau];
    let lag_indices: Vec<usize> = target_lags
        .iter()
        .map(|&t| ((t / dt).round() as usize).max(1).min(n_times - 1))
        .collect();

    // Determine r_max from the largest lag displacement
    let max_lag = *lag_indices.iter().max().unwrap_or(&1);
    let mut max_dx = 1.0f64;
    let n_sample = (n_times - max_lag).min(50);
    for t0_idx in 0..n_sample {
        let t0 = t0_idx * (n_times - max_lag) / n_sample.max(1);
        let ti = t0 + max_lag;
        if ti >= n_times {
            break;
        }
        for i in 0..n_cells {
            let dx = (pos.positions[ti][i][0] - pos.positions[t0][i][0]).abs();
            if dx > max_dx {
                max_dx = dx;
            }
        }
    }
    let r_max = max_dx * 1.5;
    let bin_width = 2.0 * r_max / n_bins as f64;
    let dx_bins: Vec<f64> = (0..n_bins)
        .map(|i| -r_max + (i as f64 + 0.5) * bin_width)
        .collect();

    let mut distributions = Vec::new();

    for &lag in &lag_indices {
        if lag >= n_times {
            continue;
        }
        let mut hist = vec![0.0f64; n_bins];
        let mut total = 0u64;

        let n_origins = n_times - lag;
        for t0 in 0..n_origins {
            let ti = t0 + lag;
            for i in 0..n_cells {
                let dx = pos.positions[ti][i][0] - pos.positions[t0][i][0];
                let b = ((dx + r_max) / bin_width) as usize;
                if b < n_bins {
                    hist[b] += 1.0;
                    total += 1;
                }
            }
        }

        // Normalize to probability density
        if total > 0 {
            let norm = total as f64 * bin_width;
            for v in &mut hist {
                *v /= norm;
            }
        }

        distributions.push(VanHoveLag {
            lag_time: lag as f64 * dt,
            histogram: hist,
        });
    }

    VanHoveResult {
        dx_bins,
        distributions,
    }
}

// ============================================================================
// Per-cell diffusion coefficient
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct PerCellDiffusionResult {
    pub cell_ids: Vec<u32>,
    pub d_values: Vec<f64>,
    pub d_mean: f64,
    pub d_std: f64,
    pub cv: f64,
}

/// Compute per-cell long-time diffusion coefficient.
pub fn per_cell_diffusion(pos: &UnwrappedPositions, _fit_frac: f64, tau: f64) -> PerCellDiffusionResult {
    let n_times = pos.n_times;
    let n_cells = pos.n_cells;
    let dt = if n_times > 1 { pos.times[1] - pos.times[0] } else { 1.0 };

    if n_times < 10 || n_cells == 0 {
        return PerCellDiffusionResult {
            cell_ids: pos.cell_ids.clone(),
            d_values: vec![0.0; n_cells],
            d_mean: 0.0,
            d_std: 0.0,
            cv: 0.0,
        };
    }

    let max_lag = n_times / 2;

    // Palmieri-style D_eff extraction: MSD(Δt)/(2*d*Δt) evaluated at Δt = 8τ.
    // At 8τ the MSD/4t curve has plateaued past the ballistic regime
    // but before long-lag noise dominates.
    // Einstein relation: D = MSD / (2*d*t) where d is spatial dimension
    let d_divisor = if pos.dim == 3 { 6.0 } else { 4.0 };
    let lag_8tau = ((8.0 * tau / dt).round() as usize).min(max_lag).max(1);

    let mut d_values = vec![0.0f64; n_cells];

    for i in 0..n_cells {
        let n_origins = n_times - lag_8tau;
        if n_origins < 2 {
            continue;
        }
        let lag_time = lag_8tau as f64 * dt;
        let mut msd_sum = 0.0f64;
        for t0 in 0..n_origins {
            let ti = t0 + lag_8tau;
            let dx = pos.positions[ti][i][0] - pos.positions[t0][i][0];
            let dy = pos.positions[ti][i][1] - pos.positions[t0][i][1];
            let dz = pos.positions[ti][i][2] - pos.positions[t0][i][2];
            msd_sum += dx * dx + dy * dy + dz * dz;
        }
        let msd = msd_sum / n_origins as f64;
        d_values[i] = msd / (d_divisor * lag_time);
    }

    let d_mean = d_values.iter().sum::<f64>() / n_cells as f64;
    let d_std = (d_values.iter().map(|d| (d - d_mean).powi(2)).sum::<f64>() / n_cells as f64)
        .sqrt();

    PerCellDiffusionResult {
        cell_ids: pos.cell_ids.clone(),
        d_values,
        d_mean,
        d_std,
        cv: if d_mean > 0.0 { d_std / d_mean } else { 0.0 },
    }
}

// ============================================================================
// Displacement (for Phase 0 quench analysis)
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct DisplacementResult {
    pub mean_dr: f64,
    pub rms_dr: f64,
    pub max_dr: f64,
    /// Normalized by cell radius R
    pub mean_dr_over_r: f64,
}

/// Compute displacement statistics between first and last frame.
pub fn compute_displacement(pos: &UnwrappedPositions, cell_radius: f64) -> DisplacementResult {
    let n_cells = pos.n_cells;
    if pos.n_times < 2 || n_cells == 0 {
        return DisplacementResult {
            mean_dr: 0.0,
            rms_dr: 0.0,
            max_dr: 0.0,
            mean_dr_over_r: 0.0,
        };
    }

    let t_last = pos.n_times - 1;
    let mut sum_dr = 0.0f64;
    let mut sum_dr2 = 0.0f64;
    let mut max_dr = 0.0f64;

    for i in 0..n_cells {
        let dx = pos.positions[t_last][i][0] - pos.positions[0][i][0];
        let dy = pos.positions[t_last][i][1] - pos.positions[0][i][1];
        let dz = pos.positions[t_last][i][2] - pos.positions[0][i][2];
        let dr = (dx * dx + dy * dy + dz * dz).sqrt();
        sum_dr += dr;
        sum_dr2 += dr * dr;
        if dr > max_dr {
            max_dr = dr;
        }
    }

    let mean_dr = sum_dr / n_cells as f64;
    let rms_dr = (sum_dr2 / n_cells as f64).sqrt();

    DisplacementResult {
        mean_dr,
        rms_dr,
        max_dr,
        mean_dr_over_r: mean_dr / cell_radius,
    }
}

// ============================================================================
// Stokes-Einstein product
// ============================================================================

/// Compute D × τ_α. Constant in equilibrium; growth indicates heterogeneity.
pub fn stokes_einstein(d_eff: f64, tau_alpha: f64) -> f64 {
    if d_eff.is_nan() || tau_alpha.is_nan() || tau_alpha <= 0.0 {
        f64::NAN
    } else {
        d_eff * tau_alpha
    }
}

// ============================================================================
// Stretched exponential fitting (used by Q(t) and F_s)
// ============================================================================

/// Fit y(t) to exp(-(t/τ)^β) using iterative Gauss-Newton.
///
/// Returns (tau, beta, R²).
fn fit_stretched_exponential(lag_times: &[f64], y: &[f64]) -> (f64, f64, f64) {
    // Filter to (t > 0, 0.05 < y < 0.99)
    let data: Vec<(f64, f64)> = lag_times
        .iter()
        .zip(y.iter())
        .filter(|(&t, &v)| t > 0.0 && v > 0.05 && v < 0.99)
        .map(|(&t, &v)| (t, v))
        .collect();

    if data.len() < 5 {
        return (f64::NAN, f64::NAN, 0.0);
    }

    // Initial guess: τ from 1/e crossing
    let e_inv = 1.0 / std::f64::consts::E;
    let tau0 = data
        .iter()
        .min_by(|(_, a), (_, b)| {
            (a - e_inv)
                .abs()
                .partial_cmp(&(b - e_inv).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(data[data.len() / 2].0, |&(t, _)| t)
        .max(1e-3);

    let mut tau = tau0;
    let mut beta = 0.8f64;

    // Simple Gauss-Newton iterations
    for _ in 0..100 {
        let mut jtj = [[0.0f64; 2]; 2];
        let mut jtr = [0.0f64; 2];

        for &(t, y_obs) in &data {
            let ratio = t / tau;
            let rb = ratio.powf(beta);
            let y_pred = (-rb).exp();
            let residual = y_obs - y_pred;

            // Partials: dy/dtau = y_pred * beta * ratio^beta / tau
            let dy_dtau = y_pred * beta * rb / tau;
            // dy/dbeta = -y_pred * ratio^beta * ln(ratio)
            let ln_ratio = ratio.ln();
            let dy_dbeta = -y_pred * rb * ln_ratio;

            jtj[0][0] += dy_dtau * dy_dtau;
            jtj[0][1] += dy_dtau * dy_dbeta;
            jtj[1][0] += dy_dbeta * dy_dtau;
            jtj[1][1] += dy_dbeta * dy_dbeta;
            jtr[0] += dy_dtau * residual;
            jtr[1] += dy_dbeta * residual;
        }

        // Solve 2x2 system with Levenberg-Marquardt damping
        let lambda = 1e-4;
        jtj[0][0] += lambda;
        jtj[1][1] += lambda;

        let det = jtj[0][0] * jtj[1][1] - jtj[0][1] * jtj[1][0];
        if det.abs() < 1e-30 {
            break;
        }

        let d_tau = (jtj[1][1] * jtr[0] - jtj[0][1] * jtr[1]) / det;
        let d_beta = (jtj[0][0] * jtr[1] - jtj[1][0] * jtr[0]) / det;

        tau = (tau + d_tau).max(1e-3);
        beta = (beta + d_beta).clamp(0.01, 2.0);

        if (d_tau / tau).abs() < 1e-8 && (d_beta / beta).abs() < 1e-8 {
            break;
        }
    }

    // R²
    let y_mean: f64 = data.iter().map(|&(_, y)| y).sum::<f64>() / data.len() as f64;
    let ss_tot: f64 = data.iter().map(|&(_, y)| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = data
        .iter()
        .map(|&(t, y)| {
            let pred = (-(t / tau).powf(beta)).exp();
            (y - pred).powi(2)
        })
        .sum();
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    (tau, beta, r2)
}

// ============================================================================
// Linear interpolation helper
// ============================================================================

fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if xs.is_empty() || ys.is_empty() {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    for i in 1..xs.len() {
        if xs[i] >= x {
            let t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
            return ys[i - 1] + t * (ys[i] - ys[i - 1]);
        }
    }
    ys[ys.len() - 1]
}

// ============================================================================
// v_A – mobility Pearson correlation
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct VaMobilityCorrelationResult {
    pub pearson_r: f64,
    pub n_cells: usize,
    /// Per-cell time-averaged speed
    pub cell_speeds: Vec<f64>,
    /// Per-cell inherent v_A
    pub cell_va: Vec<f64>,
}

/// Compute Pearson correlation between inherent v_A and time-averaged speed.
///
/// Requires per-cell v_A from the 10th trajectory column.
pub fn va_mobility_correlation(pos: &UnwrappedPositions) -> VaMobilityCorrelationResult {
    let n_cells = pos.n_cells;
    let n_times = pos.n_times;

    if pos.inherent_v_a.is_empty() || n_times < 2 || n_cells == 0 {
        return VaMobilityCorrelationResult {
            pearson_r: f64::NAN,
            n_cells,
            cell_speeds: vec![],
            cell_va: vec![],
        };
    }

    let dt = if n_times > 1 {
        pos.times[1] - pos.times[0]
    } else {
        1.0
    };

    // Compute time-averaged speed per cell
    let mut speeds = vec![0.0f64; n_cells];
    for i in 0..n_cells {
        let mut total_dist = 0.0;
        for t in 1..n_times {
            let dx = pos.positions[t][i][0] - pos.positions[t - 1][i][0];
            let dy = pos.positions[t][i][1] - pos.positions[t - 1][i][1];
            let dz = pos.positions[t][i][2] - pos.positions[t - 1][i][2];
            total_dist += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        speeds[i] = total_dist / ((n_times - 1) as f64 * dt);
    }

    // Pearson correlation
    let va = &pos.inherent_v_a;
    let mean_va = va.iter().sum::<f64>() / n_cells as f64;
    let mean_sp = speeds.iter().sum::<f64>() / n_cells as f64;

    let mut cov = 0.0;
    let mut var_va = 0.0;
    let mut var_sp = 0.0;
    for i in 0..n_cells {
        let dv = va[i] - mean_va;
        let ds = speeds[i] - mean_sp;
        cov += dv * ds;
        var_va += dv * dv;
        var_sp += ds * ds;
    }

    let denom = (var_va * var_sp).sqrt();
    let pearson_r = if denom > 0.0 { cov / denom } else { 0.0 };

    VaMobilityCorrelationResult {
        pearson_r,
        n_cells,
        cell_speeds: speeds,
        cell_va: va.clone(),
    }
}

// ============================================================================
// Spatial autocorrelation C(r)
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct SpatialCorrelationResult {
    pub r_bins: Vec<f64>,
    pub c_r: Vec<f64>,
    /// Correlation length: r where C(r) = 1/e
    pub xi: f64,
}

/// Compute spatial autocorrelation of instantaneous cell mobility vs distance.
///
/// C(r) = <δm_i δm_j> for |r_i - r_j| ∈ [r, r+dr], normalized by variance.
/// Extract correlation length ξ where C(r) = 1/e.
pub fn spatial_correlation(pos: &UnwrappedPositions, n_bins: usize) -> SpatialCorrelationResult {
    let n_cells = pos.n_cells;
    let n_times = pos.n_times;
    let lx = pos.lx;
    let ly = pos.ly;

    if n_times < 2 || n_cells < 2 {
        return SpatialCorrelationResult {
            r_bins: vec![],
            c_r: vec![],
            xi: f64::NAN,
        };
    }

    let r_max = (lx.min(ly)) / 2.0;
    let dr = r_max / n_bins as f64;

    let mut c_sum = vec![0.0f64; n_bins];
    let mut counts = vec![0u64; n_bins];

    // Sample frames from the second half
    let n_sample = 20.min(n_times);
    let step = (n_times / 2).max(1) / n_sample.max(1);

    for f_idx in 0..n_sample {
        let t = n_times / 2 + f_idx * step.max(1);
        if t >= n_times || t == 0 {
            continue;
        }

        // Compute per-cell displacement magnitude (instantaneous speed proxy)
        let mut mobility = vec![0.0f64; n_cells];
        for i in 0..n_cells {
            let dx = pos.positions[t][i][0] - pos.positions[t - 1][i][0];
            let dy = pos.positions[t][i][1] - pos.positions[t - 1][i][1];
            let dz = pos.positions[t][i][2] - pos.positions[t - 1][i][2];
            mobility[i] = (dx * dx + dy * dy + dz * dz).sqrt();
        }

        let mean_m = mobility.iter().sum::<f64>() / n_cells as f64;
        let var_m: f64 = mobility.iter().map(|m| (m - mean_m).powi(2)).sum::<f64>() / n_cells as f64;
        if var_m < 1e-30 {
            continue;
        }

        // Wrapped positions for distance calculation
        let wrapped: Vec<[f64; 2]> = (0..n_cells).map(|i| pos.wrapped(t, i)).collect();

        // Pairwise correlation
        for i in 0..n_cells {
            for j in (i + 1)..n_cells {
                let mut dx = wrapped[i][0] - wrapped[j][0];
                let mut dy = wrapped[i][1] - wrapped[j][1];
                // Minimum image
                if dx > lx / 2.0 { dx -= lx; } else if dx < -lx / 2.0 { dx += lx; }
                if dy > ly / 2.0 { dy -= ly; } else if dy < -ly / 2.0 { dy += ly; }
                let r = (dx * dx + dy * dy).sqrt();
                let b = (r / dr) as usize;
                if b < n_bins {
                    let dm_i = mobility[i] - mean_m;
                    let dm_j = mobility[j] - mean_m;
                    c_sum[b] += dm_i * dm_j / var_m;
                    counts[b] += 1;
                }
            }
        }
    }

    let r_bins: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * dr).collect();
    let mut c_r = vec![0.0; n_bins];
    for i in 0..n_bins {
        if counts[i] > 0 {
            c_r[i] = c_sum[i] / counts[i] as f64;
        }
    }

    // Find ξ where C(r) = 1/e (first crossing from above)
    let threshold = 1.0 / std::f64::consts::E;
    let xi = r_bins
        .iter()
        .zip(c_r.iter())
        .skip(1) // skip r=0 bin
        .find(|(_, &c)| c < threshold)
        .map_or(r_max, |(&r, _)| r);

    SpatialCorrelationResult { r_bins, c_r, xi }
}

// ============================================================================
// Shape index from L_n (normalized perimeter)
// ============================================================================

/// Shape index result: p_eff = L_n × 2√π.
/// The vertex model transition occurs at p₀* ≈ 3.81.
#[derive(Serialize, Clone, Debug)]
pub struct ShapeIndexResult {
    /// Mean shape index ⟨p⟩ over all cells in the last frame
    pub mean_p: f64,
    /// Std dev of p across cells in the last frame
    pub std_p: f64,
    /// Per-cell shape index in the last frame (indexed by cell_id order)
    pub per_cell_p: Vec<f64>,
    /// Time series: mean p per frame (average over all cells)
    pub p_vs_time: Vec<f64>,
    /// Time series: cell 0's p per frame (for Palmieri Fig 3A)
    pub cell0_p_vs_time: Vec<f64>,
    /// Corresponding times for p_vs_time
    pub times: Vec<f64>,
    /// Number of frames with valid L_n data
    pub n_frames: usize,
}

/// Compute shape index from trajectory L_n values.
/// p_eff = L_n × 2√π ≈ L_n × 3.5449
pub fn shape_index(traj: &super::io::Trajectory) -> ShapeIndexResult {
    let factor = 2.0 * std::f64::consts::PI.sqrt(); // 2√π ≈ 3.5449

    let mut times = Vec::new();
    let mut p_vs_time = Vec::new();

    let cell_ids = traj.cell_ids();
    let cell0_id = cell_ids.first().copied().unwrap_or(0);

    let mut cell0_p_vs_time = Vec::new();

    for &(t, ref frame) in &traj.frames {
        let mut sum_p = 0.0;
        let mut count = 0;
        for &cid in &cell_ids {
            if let Some(snap) = frame.get(&cid) {
                if snap.l_n > 0.0 {
                    sum_p += snap.l_n * factor;
                    count += 1;
                }
            }
        }
        if count > 0 {
            times.push(t);
            p_vs_time.push(sum_p / count as f64);
            // Cell 0 L_n for this frame
            let c0_p = frame.get(&cell0_id).map_or(0.0, |s| s.l_n * factor);
            cell0_p_vs_time.push(c0_p);
        }
    }

    // Last frame per-cell values
    let mut per_cell_p = Vec::new();
    let mut last_sum = 0.0;
    let mut last_sum_sq = 0.0;
    let mut last_count = 0usize;

    if let Some(&(_, ref frame)) = traj.frames.last() {
        for &cid in &cell_ids {
            if let Some(snap) = frame.get(&cid) {
                let p = snap.l_n * factor;
                per_cell_p.push(p);
                if snap.l_n > 0.0 {
                    last_sum += p;
                    last_sum_sq += p * p;
                    last_count += 1;
                }
            }
        }
    }

    let mean_p = if last_count > 0 { last_sum / last_count as f64 } else { 0.0 };
    let var = if last_count > 1 {
        (last_sum_sq / last_count as f64) - mean_p * mean_p
    } else {
        0.0
    };
    let std_p = if var > 0.0 { var.sqrt() } else { 0.0 };

    let n_frames = times.len();

    ShapeIndexResult {
        mean_p,
        std_p,
        per_cell_p,
        p_vs_time,
        cell0_p_vs_time,
        times,
        n_frames,
    }
}

// ============================================================================
// Velocity autocorrelation C_v(τ) = ⟨v(t)·v(t+τ)⟩ / ⟨v²⟩
// ============================================================================

/// Velocity autocorrelation result.
#[derive(Serialize, Clone, Debug)]
pub struct VelocityAutocorrelationResult {
    pub lag_times: Vec<f64>,
    /// C_v(τ) normalized: C_v(0) = 1
    pub cv: Vec<f64>,
    /// Stretching exponent β from fit C_v ~ exp(-(τ/τ_c)^β)
    pub beta: f64,
    /// Correlation time τ_c
    pub tau_c: f64,
}

/// Compute velocity autocorrelation from unwrapped positions.
/// Velocities are finite-difference: v(t) = [x(t+dt) - x(t-dt)] / (2*dt_save).
pub fn velocity_autocorrelation(pos: &UnwrappedPositions) -> VelocityAutocorrelationResult {
    let n = pos.n_times;
    let nc = pos.n_cells;
    if n < 3 {
        return VelocityAutocorrelationResult {
            lag_times: vec![], cv: vec![], beta: 1.0, tau_c: 0.0,
        };
    }

    let dt_save = if n >= 2 { pos.times[1] - pos.times[0] } else { 1.0 };

    // Compute velocities via central differences (skip first/last)
    let n_vel = n - 2;
    let mut vx = vec![vec![0.0_f64; nc]; n_vel];
    let mut vy = vec![vec![0.0_f64; nc]; n_vel];
    let mut vz = vec![vec![0.0_f64; nc]; n_vel];
    for t in 0..n_vel {
        let dt2 = pos.times[t + 2] - pos.times[t];
        if dt2 <= 0.0 { continue; }
        for i in 0..nc {
            vx[t][i] = (pos.positions[t + 2][i][0] - pos.positions[t][i][0]) / dt2;
            vy[t][i] = (pos.positions[t + 2][i][1] - pos.positions[t][i][1]) / dt2;
            vz[t][i] = (pos.positions[t + 2][i][2] - pos.positions[t][i][2]) / dt2;
        }
    }

    // Compute C_v(lag) = ⟨v(t)·v(t+lag)⟩ averaged over cells and time origins
    let max_lag = n_vel / 2;
    let mut lag_times = Vec::with_capacity(max_lag);
    let mut cv = Vec::with_capacity(max_lag);

    // Normalization: C_v(0) = ⟨v²⟩
    let mut v_sq_sum = 0.0;
    let mut v_sq_count = 0usize;
    for t in 0..n_vel {
        for i in 0..nc {
            v_sq_sum += vx[t][i] * vx[t][i] + vy[t][i] * vy[t][i] + vz[t][i] * vz[t][i];
            v_sq_count += 1;
        }
    }
    let v_sq_mean = if v_sq_count > 0 { v_sq_sum / v_sq_count as f64 } else { 1.0 };

    for lag in 0..max_lag {
        let mut dot_sum = 0.0;
        let mut count = 0usize;
        for t in 0..(n_vel - lag) {
            for i in 0..nc {
                dot_sum += vx[t][i] * vx[t + lag][i] + vy[t][i] * vy[t + lag][i] + vz[t][i] * vz[t + lag][i];
                count += 1;
            }
        }
        let cv_val = if count > 0 && v_sq_mean > 0.0 {
            (dot_sum / count as f64) / v_sq_mean
        } else {
            0.0
        };
        lag_times.push(lag as f64 * dt_save);
        cv.push(cv_val);
    }

    // Simple estimate of tau_c: time where C_v first drops below 1/e
    let e_inv = 1.0 / std::f64::consts::E;
    let mut tau_c = lag_times.last().copied().unwrap_or(0.0);
    for (i, &c) in cv.iter().enumerate() {
        if c < e_inv {
            tau_c = lag_times[i];
            break;
        }
    }

    VelocityAutocorrelationResult {
        lag_times,
        cv,
        beta: 1.0, // placeholder — full stretched-exp fit could be added
        tau_c,
    }
}

// ============================================================================
// Burst detection: events where |v| > μ_v + k*σ_v for consecutive frames
// ============================================================================

/// A single burst event for a cell.
#[derive(Serialize, Clone, Debug)]
pub struct BurstEvent {
    pub cell_id: u32,
    pub t_start: f64,
    pub t_end: f64,
    pub duration: f64,
    pub peak_speed: f64,
    pub mean_speed: f64,
}

/// Burst detection result for the full trajectory.
#[derive(Serialize, Clone, Debug)]
pub struct BurstDetectionResult {
    /// Total bursts detected across all cells
    pub total_bursts: usize,
    /// Bursts per cell (mean across cells)
    pub mean_bursts_per_cell: f64,
    /// Mean burst duration (in time units)
    pub mean_duration: f64,
    /// Mean peak speed during bursts
    pub mean_peak_speed: f64,
    /// Individual burst events (capped at 10000 to avoid huge JSON)
    pub events: Vec<BurstEvent>,
    /// Threshold used: μ + k*σ
    pub threshold: f64,
    /// Speed statistics
    pub speed_mean: f64,
    pub speed_std: f64,
}

/// Detect speed bursts in trajectory.
/// A burst is defined as |v| > μ_v + k*σ_v for at least `min_frames` consecutive frames.
/// `k` defaults to 3.0 (3-sigma), `min_frames` defaults to 1.
pub fn detect_bursts(
    pos: &UnwrappedPositions,
    _traj: &super::io::Trajectory,
    k_sigma: f64,
    min_frames: usize,
) -> BurstDetectionResult {
    let n = pos.n_times;
    let nc = pos.n_cells;
    let cell_ids = &pos.cell_ids;
    let empty = BurstDetectionResult {
        total_bursts: 0, mean_bursts_per_cell: 0.0, mean_duration: 0.0,
        mean_peak_speed: 0.0, events: vec![], threshold: 0.0,
        speed_mean: 0.0, speed_std: 0.0,
    };
    if n < 2 { return empty; }

    // Compute speeds from finite differences
    let mut speeds: Vec<Vec<f64>> = Vec::with_capacity(n - 1);
    for t in 0..(n - 1) {
        let dt = pos.times[t + 1] - pos.times[t];
        if dt <= 0.0 {
            speeds.push(vec![0.0; nc]);
            continue;
        }
        let mut frame_speeds = vec![0.0; nc];
        for i in 0..nc {
            let dx = pos.positions[t + 1][i][0] - pos.positions[t][i][0];
            let dy = pos.positions[t + 1][i][1] - pos.positions[t][i][1];
            let dz = pos.positions[t + 1][i][2] - pos.positions[t][i][2];
            frame_speeds[i] = (dx * dx + dy * dy + dz * dz).sqrt() / dt;
        }
        speeds.push(frame_speeds);
    }

    // Global speed statistics (all cells, all times)
    let mut all_speeds: Vec<f64> = Vec::new();
    for frame in &speeds {
        for &s in frame {
            all_speeds.push(s);
        }
    }
    if all_speeds.is_empty() { return empty; }

    let speed_mean = all_speeds.iter().sum::<f64>() / all_speeds.len() as f64;
    let speed_var = all_speeds.iter().map(|s| (s - speed_mean).powi(2)).sum::<f64>()
        / all_speeds.len() as f64;
    let speed_std = speed_var.sqrt();
    let threshold = speed_mean + k_sigma * speed_std;

    // Detect bursts per cell
    let mut events = Vec::new();
    let n_speed = speeds.len();

    for ci in 0..nc {
        let cid = cell_ids[ci];
        let mut t = 0;
        while t < n_speed {
            if speeds[t][ci] > threshold {
                let t_start_idx = t;
                let mut peak = speeds[t][ci];
                let mut sum_speed = speeds[t][ci];
                let mut count = 1usize;
                t += 1;
                while t < n_speed && speeds[t][ci] > threshold {
                    if speeds[t][ci] > peak { peak = speeds[t][ci]; }
                    sum_speed += speeds[t][ci];
                    count += 1;
                    t += 1;
                }
                if count >= min_frames {
                    events.push(BurstEvent {
                        cell_id: cid,
                        t_start: pos.times[t_start_idx],
                        t_end: pos.times[(t_start_idx + count).min(n - 1)],
                        duration: pos.times[(t_start_idx + count).min(n - 1)] - pos.times[t_start_idx],
                        peak_speed: peak,
                        mean_speed: sum_speed / count as f64,
                    });
                }
            } else {
                t += 1;
            }
        }
    }

    let total = events.len();
    let mean_dur = if total > 0 { events.iter().map(|e| e.duration).sum::<f64>() / total as f64 } else { 0.0 };
    let mean_peak = if total > 0 { events.iter().map(|e| e.peak_speed).sum::<f64>() / total as f64 } else { 0.0 };

    // Cap events list to avoid huge JSON
    if events.len() > 10000 {
        events.truncate(10000);
    }

    BurstDetectionResult {
        total_bursts: total,
        mean_bursts_per_cell: total as f64 / nc as f64,
        mean_duration: mean_dur,
        mean_peak_speed: mean_peak,
        events,
        threshold,
        speed_mean,
        speed_std,
    }
}

// ============================================================================
// Velocity distribution — P(v_x), kurtosis, for cell 0 and population
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct VelocityDistributionResult {
    /// Histogram bin edges for v_x
    pub bin_edges: Vec<f64>,
    /// P(v_x) for cell 0
    pub cell0_hist: Vec<f64>,
    /// P(v_x) for population (all cells)
    pub pop_hist: Vec<f64>,
    /// Cell 0 velocity stats
    pub cell0_mean_speed: f64,
    pub cell0_kurtosis: f64,
    /// Population velocity stats
    pub pop_mean_speed: f64,
    pub pop_kurtosis: f64,
    /// Gaussian std dev (for reference curve)
    pub pop_sigma_vx: f64,
    pub cell0_sigma_vx: f64,
    /// Raw velocity samples (kept in-memory for downstream panels such as
    /// `panels::draw_gvi_panel`; skipped during JSON serialization to avoid
    /// bloating run_result.json).
    #[serde(skip, default)]
    pub pop_vx: Vec<f64>,
    #[serde(skip, default)]
    pub pop_vy: Vec<f64>,
    #[serde(skip, default)]
    pub cell0_vx: Vec<f64>,
    #[serde(skip, default)]
    pub cell0_vy: Vec<f64>,
}

/// Compute velocity distribution P(v_x) for cell 0 and population.
/// Velocities computed as centroid finite-differences with periodic unwrapping.
pub fn velocity_distribution(pos: &UnwrappedPositions, n_bins: usize) -> VelocityDistributionResult {
    let n = pos.n_times;
    let nc = pos.n_cells;
    let dt = if n > 1 { pos.times[1] - pos.times[0] } else { 1.0 };

    let empty = VelocityDistributionResult {
        bin_edges: vec![], cell0_hist: vec![], pop_hist: vec![],
        cell0_mean_speed: 0.0, cell0_kurtosis: 0.0,
        pop_mean_speed: 0.0, pop_kurtosis: 0.0,
        pop_sigma_vx: 0.0, cell0_sigma_vx: 0.0,
        pop_vx: vec![], pop_vy: vec![], cell0_vx: vec![], cell0_vy: vec![],
    };
    if n < 3 || nc == 0 { return empty; }

    // Compute v_x for all cells at all times
    let n_vel = n - 1;
    let mut cell0_vx: Vec<f64> = Vec::with_capacity(n_vel);
    let mut cell0_vy: Vec<f64> = Vec::with_capacity(n_vel);
    let mut pop_vx: Vec<f64> = Vec::with_capacity(n_vel * nc);
    let mut pop_vy: Vec<f64> = Vec::with_capacity(n_vel * nc);
    let mut cell0_speeds: Vec<f64> = Vec::with_capacity(n_vel);
    let mut pop_speeds: Vec<f64> = Vec::with_capacity(n_vel * nc);

    for t in 0..n_vel {
        for i in 0..nc {
            let dx = pos.positions[t + 1][i][0] - pos.positions[t][i][0];
            let dy = pos.positions[t + 1][i][1] - pos.positions[t][i][1];
            let vx = dx / dt;
            let vy = dy / dt;
            let speed = (vx * vx + vy * vy).sqrt();

            pop_vx.push(vx);
            pop_vy.push(vy);
            pop_speeds.push(speed);

            if pos.cell_ids[i] == 0 {
                cell0_vx.push(vx);
                cell0_vy.push(vy);
                cell0_speeds.push(speed);
            }
        }
    }

    // Statistics
    let mean_f = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let std_f = |v: &[f64], m: f64| (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    let kurtosis_f = |v: &[f64], m: f64, s: f64| {
        if s < 1e-30 { return 0.0; }
        let n = v.len() as f64;
        let m4 = v.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n;
        m4 - 3.0 // excess kurtosis (0 for Gaussian)
    };

    let c0_vx_mean = mean_f(&cell0_vx);
    let c0_vx_std = std_f(&cell0_vx, c0_vx_mean);
    let c0_kurt = kurtosis_f(&cell0_vx, c0_vx_mean, c0_vx_std);
    let c0_mean_speed = mean_f(&cell0_speeds);

    let pop_vx_mean = mean_f(&pop_vx);
    let pop_vx_std = std_f(&pop_vx, pop_vx_mean);
    let pop_kurt = kurtosis_f(&pop_vx, pop_vx_mean, pop_vx_std);
    let pop_mean_speed = mean_f(&pop_speeds);

    // Histogram
    let v_max = pop_vx.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    let bin_width = 2.0 * v_max / n_bins as f64;
    let bin_edges: Vec<f64> = (0..=n_bins).map(|i| -v_max + i as f64 * bin_width).collect();

    let histogramize = |data: &[f64]| -> Vec<f64> {
        let mut counts = vec![0u64; n_bins];
        for &vx in data {
            let idx = ((vx + v_max) / bin_width) as usize;
            let idx = idx.min(n_bins - 1);
            counts[idx] += 1;
        }
        let total = data.len() as f64;
        counts.iter().map(|&c| c as f64 / (total * bin_width)).collect()
    };

    let cell0_hist = histogramize(&cell0_vx);
    let pop_hist = histogramize(&pop_vx);

    VelocityDistributionResult {
        bin_edges,
        cell0_hist,
        pop_hist,
        cell0_mean_speed: c0_mean_speed,
        cell0_kurtosis: c0_kurt,
        pop_mean_speed: pop_mean_speed,
        pop_kurtosis: pop_kurt,
        pop_sigma_vx: pop_vx_std,
        cell0_sigma_vx: c0_vx_std,
        pop_vx,
        pop_vy,
        cell0_vx,
        cell0_vy,
    }
}

// ============================================================================
// Observable registry
// ============================================================================

/// All available observable names.
// ============================================================================
// Polarity τ estimation
// ============================================================================

/// Result of polarity autocorrelation analysis.
/// Estimates persistence time τ from ⟨p̂(t+Δt)·p̂(t)⟩ = exp(-Δt/τ).
#[derive(Serialize, Clone, Debug)]
pub struct PolarityTauResult {
    /// Estimated persistence time τ (in simulation time units)
    pub tau: f64,
    /// R² goodness of fit for the exponential decay
    pub fit_r2: f64,
    /// Lag times used for the fit (in τ units)
    pub lag_times: Vec<f64>,
    /// Polarity autocorrelation C_p(Δt) at each lag
    pub correlation: Vec<f64>,
    /// Per-cell τ estimates (cell_id, tau)
    pub per_cell_tau: Vec<(u32, f64)>,
    /// Population mean τ
    pub tau_mean: f64,
    /// Population std τ
    pub tau_std: f64,
}

/// Estimate persistence time τ from polarity autocorrelation.
///
/// For run-and-tumble: ⟨p̂(t+Δt)·p̂(t)⟩ = exp(-Δt/τ).
/// For ABP: ⟨p̂(t+Δt)·p̂(t)⟩ = exp(-Δt·D_r) where D_r = 1/τ.
///
/// Uses ALL cells (population average) for the main fit,
/// and also computes per-cell τ estimates.
pub fn polarity_tau(traj: &super::io::Trajectory) -> PolarityTauResult {
    let frames = &traj.frames;
    let n_frames = frames.len();

    if n_frames < 10 {
        return PolarityTauResult {
            tau: 0.0, fit_r2: 0.0,
            lag_times: vec![], correlation: vec![],
            per_cell_tau: vec![], tau_mean: 0.0, tau_std: 0.0,
        };
    }

    // Get frame times
    let times: Vec<f64> = frames.iter().map(|(t, _)| *t).collect();
    let dt_frame = if n_frames >= 2 { times[1] - times[0] } else { 1.0 };

    // Get all cell IDs from first frame
    let cell_ids: Vec<u32> = {
        let mut ids: Vec<u32> = frames[0].1.keys().copied().collect();
        ids.sort();
        ids
    };
    let n_cells = cell_ids.len();

    // Check if polarity is non-trivial (v_A=0 equilibrations have px=py=0)
    let has_polarity = frames.iter().any(|(_, cells)| {
        cells.values().any(|s| s.px.abs() > 1e-10 || s.py.abs() > 1e-10)
    });
    if !has_polarity {
        return PolarityTauResult {
            tau: 0.0, fit_r2: 0.0,
            lag_times: vec![], correlation: vec![],
            per_cell_tau: vec![], tau_mean: 0.0, tau_std: 0.0,
        };
    }

    // Build polarity arrays: polarity[cell_idx][frame] = (px, py)
    let mut pol: Vec<Vec<(f64, f64)>> = vec![Vec::with_capacity(n_frames); n_cells];
    for (_, cells) in frames {
        for (ci, &cid) in cell_ids.iter().enumerate() {
            if let Some(snap) = cells.get(&cid) {
                pol[ci].push((snap.px, snap.py));
            } else {
                pol[ci].push((0.0, 0.0));
            }
        }
    }

    // Compute population-averaged polarity autocorrelation for log-spaced lags
    let max_lag = n_frames / 4;
    if max_lag < 3 {
        return PolarityTauResult {
            tau: 0.0, fit_r2: 0.0,
            lag_times: vec![], correlation: vec![],
            per_cell_tau: vec![], tau_mean: 0.0, tau_std: 0.0,
        };
    }

    let mut lag_indices: Vec<usize> = Vec::new();
    let mut lag = 1usize;
    while lag <= max_lag {
        lag_indices.push(lag);
        lag = std::cmp::max(lag + 1, (lag as f64 * 1.2).ceil() as usize);
    }
    lag_indices.dedup();

    let mut lag_times_out: Vec<f64> = Vec::new();
    let mut corr_out: Vec<f64> = Vec::new();

    for &lag in &lag_indices {
        let mut dot_sum = 0.0;
        let mut count = 0usize;
        for ci in 0..n_cells {
            for t in 0..(n_frames - lag) {
                let (px0, py0) = pol[ci][t];
                let (px1, py1) = pol[ci][t + lag];
                let norm0 = (px0 * px0 + py0 * py0).sqrt();
                let norm1 = (px1 * px1 + py1 * py1).sqrt();
                if norm0 > 1e-10 && norm1 > 1e-10 {
                    dot_sum += (px0 * px1 + py0 * py1) / (norm0 * norm1);
                    count += 1;
                }
            }
        }
        if count > 0 {
            let c = dot_sum / count as f64;
            if c > 0.01 {
                lag_times_out.push(lag as f64 * dt_frame);
                corr_out.push(c);
            }
        }
    }

    if corr_out.len() < 3 {
        return PolarityTauResult {
            tau: 0.0, fit_r2: 0.0,
            lag_times: lag_times_out, correlation: corr_out,
            per_cell_tau: vec![], tau_mean: 0.0, tau_std: 0.0,
        };
    }

    // Fit ln(C) = -Δt / τ  →  slope = -1/τ
    let (tau_pop, r2_pop) = fit_exp_decay(&lag_times_out, &corr_out);

    // Per-cell τ estimates
    let mut per_cell_tau: Vec<(u32, f64)> = Vec::new();
    for (ci, &cid) in cell_ids.iter().enumerate() {
        let mut cell_lags: Vec<f64> = Vec::new();
        let mut cell_corr: Vec<f64> = Vec::new();
        for &lag in &lag_indices {
            let mut dot_sum = 0.0;
            let mut count = 0usize;
            for t in 0..(n_frames - lag) {
                let (px0, py0) = pol[ci][t];
                let (px1, py1) = pol[ci][t + lag];
                let norm0 = (px0 * px0 + py0 * py0).sqrt();
                let norm1 = (px1 * px1 + py1 * py1).sqrt();
                if norm0 > 1e-10 && norm1 > 1e-10 {
                    dot_sum += (px0 * px1 + py0 * py1) / (norm0 * norm1);
                    count += 1;
                }
            }
            if count > 10 {
                let c = dot_sum / count as f64;
                if c > 0.01 {
                    cell_lags.push(lag as f64 * dt_frame);
                    cell_corr.push(c);
                }
            }
        }
        if cell_corr.len() >= 3 {
            let (cell_tau, _) = fit_exp_decay(&cell_lags, &cell_corr);
            if cell_tau > 0.0 && cell_tau.is_finite() {
                per_cell_tau.push((cid, cell_tau));
            }
        }
    }

    let tau_mean = if per_cell_tau.is_empty() { tau_pop } else {
        per_cell_tau.iter().map(|(_, t)| t).sum::<f64>() / per_cell_tau.len() as f64
    };
    let tau_std = if per_cell_tau.len() < 2 { 0.0 } else {
        let var = per_cell_tau.iter().map(|(_, t)| (t - tau_mean).powi(2)).sum::<f64>()
            / (per_cell_tau.len() - 1) as f64;
        var.sqrt()
    };

    PolarityTauResult {
        tau: tau_pop,
        fit_r2: r2_pop,
        lag_times: lag_times_out,
        correlation: corr_out,
        per_cell_tau,
        tau_mean,
        tau_std,
    }
}

/// Fit C(Δt) = exp(-Δt/τ) via linear regression on ln(C) vs Δt.
/// Returns (τ, R²).
fn fit_exp_decay(lag_times: &[f64], corr: &[f64]) -> (f64, f64) {
    let n = lag_times.len() as f64;
    if n < 2.0 { return (0.0, 0.0); }

    let ln_corr: Vec<f64> = corr.iter().map(|c| c.ln()).collect();

    let sum_x: f64 = lag_times.iter().sum();
    let sum_y: f64 = ln_corr.iter().sum();
    let sum_xy: f64 = lag_times.iter().zip(ln_corr.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = lag_times.iter().map(|x| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 { return (0.0, 0.0); }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    if slope >= 0.0 { return (0.0, 0.0); } // correlation must decay

    let tau = -1.0 / slope;

    // R²
    let y_mean = sum_y / n;
    let ss_tot: f64 = ln_corr.iter().map(|y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = lag_times.iter().zip(ln_corr.iter())
        .map(|(x, y)| { let yp = slope * x + intercept; (y - yp).powi(2) }).sum();
    let r2 = if ss_tot > 1e-30 { 1.0 - ss_res / ss_tot } else { 0.0 };

    (tau, r2)
}

// ============================================================================
// Hexatic order parameter ψ₆
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct HexaticOrderResult {
    /// Mean |ψ₆| across all cells (time-averaged).
    pub psi6_mean: f64,
    /// Std of |ψ₆| across cells.
    pub psi6_std: f64,
    /// Per-cell time-averaged |ψ₆|.
    pub psi6_per_cell: Vec<f64>,
    /// g₆(r) orientational correlation function: radii (in pixels).
    pub g6_r: Vec<f64>,
    /// g₆(r) values.
    pub g6_values: Vec<f64>,
}

/// Compute hexatic order ψ₆ from cell centroids.
///
/// For each cell i, find Delaunay-style neighbors (cells within cutoff = 3R),
/// then ψ₆ᵢ = |1/nᵢ Σⱼ exp(6i·θᵢⱼ)| where θᵢⱼ = atan2(yⱼ-yᵢ, xⱼ-xᵢ).
/// Also computes g₆(r) = ⟨ψ₆*(rᵢ)·ψ₆(rⱼ)⟩ vs |rᵢ-rⱼ|.
pub fn compute_hexatic_order(pos: &UnwrappedPositions, cell_radius: f64) -> HexaticOrderResult {
    let n_cells = pos.n_cells;
    let n_times = pos.n_times;
    let cutoff = 3.0 * cell_radius;
    let cutoff2 = cutoff * cutoff;
    let lx = pos.lx;
    let ly = pos.ly;

    // Per-cell accumulator for time-averaged |ψ₆|
    let mut psi6_accum = vec![0.0f64; n_cells];
    // For g₆(r): accumulate Re(ψ₆*(i)·ψ₆(j)) in radial bins
    let n_bins = 40;
    let bin_width = cutoff * 2.0 / n_bins as f64;
    let mut g6_sum = vec![0.0f64; n_bins];
    let mut g6_count = vec![0u64; n_bins];

    for t in 0..n_times {
        // Wrapped positions for this frame
        let wx: Vec<f64> = (0..n_cells).map(|i| pos.positions[t][i][0].rem_euclid(lx)).collect();
        let wy: Vec<f64> = (0..n_cells).map(|i| pos.positions[t][i][1].rem_euclid(ly)).collect();

        // Compute ψ₆ for each cell (complex: re + im)
        let mut psi6_re = vec![0.0f64; n_cells];
        let mut psi6_im = vec![0.0f64; n_cells];
        let mut n_nbr = vec![0u32; n_cells];

        for i in 0..n_cells {
            for j in (i + 1)..n_cells {
                let mut dx = wx[j] - wx[i];
                let mut dy = wy[j] - wy[i];
                if dx > lx * 0.5 { dx -= lx; }
                if dx < -lx * 0.5 { dx += lx; }
                if dy > ly * 0.5 { dy -= ly; }
                if dy < -ly * 0.5 { dy += ly; }
                let r2 = dx * dx + dy * dy;
                if r2 < cutoff2 && r2 > 1e-10 {
                    let theta = dy.atan2(dx);
                    let c6 = (6.0 * theta).cos();
                    let s6 = (6.0 * theta).sin();
                    psi6_re[i] += c6;
                    psi6_im[i] += s6;
                    n_nbr[i] += 1;
                    // Reverse direction for j
                    let c6r = (6.0 * (theta + PI)).cos();
                    let s6r = (6.0 * (theta + PI)).sin();
                    psi6_re[j] += c6r;
                    psi6_im[j] += s6r;
                    n_nbr[j] += 1;
                }
            }
        }

        // Normalize and accumulate
        for i in 0..n_cells {
            if n_nbr[i] > 0 {
                let n = n_nbr[i] as f64;
                psi6_re[i] /= n;
                psi6_im[i] /= n;
            }
            let mag = (psi6_re[i] * psi6_re[i] + psi6_im[i] * psi6_im[i]).sqrt();
            psi6_accum[i] += mag;
        }

        // g₆(r): correlate ψ₆ between all pairs
        for i in 0..n_cells {
            for j in (i + 1)..n_cells {
                let mut dx = wx[j] - wx[i];
                let mut dy = wy[j] - wy[i];
                if dx > lx * 0.5 { dx -= lx; }
                if dx < -lx * 0.5 { dx += lx; }
                if dy > ly * 0.5 { dy -= ly; }
                if dy < -ly * 0.5 { dy += ly; }
                let r = (dx * dx + dy * dy).sqrt();
                let bin = (r / bin_width) as usize;
                if bin < n_bins {
                    // Re(ψ₆*(i)·ψ₆(j))
                    let dot = psi6_re[i] * psi6_re[j] + psi6_im[i] * psi6_im[j];
                    g6_sum[bin] += dot;
                    g6_count[bin] += 1;
                }
            }
        }
    }

    let nt = n_times as f64;
    let psi6_per_cell: Vec<f64> = psi6_accum.iter().map(|&v| v / nt).collect();
    let psi6_mean = psi6_per_cell.iter().sum::<f64>() / n_cells as f64;
    let psi6_var = psi6_per_cell.iter().map(|&v| (v - psi6_mean).powi(2)).sum::<f64>() / n_cells as f64;
    let psi6_std = psi6_var.sqrt();

    let g6_r: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * bin_width).collect();
    let g6_values: Vec<f64> = (0..n_bins)
        .map(|i| if g6_count[i] > 0 { g6_sum[i] / g6_count[i] as f64 } else { 0.0 })
        .collect();

    HexaticOrderResult {
        psi6_mean,
        psi6_std,
        psi6_per_cell,
        g6_r,
        g6_values,
    }
}

// ============================================================================
// Voronoi shape index q = P/√A
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct VoronoiShapeResult {
    /// Mean Voronoi shape index q = P/√A across all cells (time-averaged).
    pub q_mean: f64,
    /// Std of q.
    pub q_std: f64,
    /// Per-cell time-averaged q.
    pub q_per_cell: Vec<f64>,
}

/// Compute Voronoi shape index from cell centroids.
///
/// Uses a simple geometric construction: for each cell, find neighbors within
/// cutoff, build the Voronoi polygon by intersecting perpendicular bisectors,
/// compute P/√A.  Falls back to a nearest-neighbor polygon if Delaunay is
/// too complex (we don't pull in a full Voronoi library).
///
/// Simpler approximation used here: for each cell i, find all neighbors j
/// within cutoff. The Voronoi polygon vertex between neighbors j and k
/// (adjacent in angular order) is the circumcenter of (i, j, k). Polygon
/// P and A are computed from these vertices.
pub fn compute_voronoi_shape(pos: &UnwrappedPositions, cell_radius: f64) -> VoronoiShapeResult {
    let n_cells = pos.n_cells;
    let n_times = pos.n_times;
    let cutoff = 4.0 * cell_radius;
    let cutoff2 = cutoff * cutoff;
    let lx = pos.lx;
    let ly = pos.ly;

    let mut q_accum = vec![0.0f64; n_cells];
    let mut q_count = vec![0u32; n_cells];

    for t in 0..n_times {
        let wx: Vec<f64> = (0..n_cells).map(|i| pos.positions[t][i][0].rem_euclid(lx)).collect();
        let wy: Vec<f64> = (0..n_cells).map(|i| pos.positions[t][i][1].rem_euclid(ly)).collect();

        for i in 0..n_cells {
            // Find neighbors, sorted by angle
            let mut nbrs: Vec<(f64, f64, f64)> = Vec::new(); // (angle, dx, dy)
            for j in 0..n_cells {
                if j == i { continue; }
                let mut dx = wx[j] - wx[i];
                let mut dy = wy[j] - wy[i];
                if dx > lx * 0.5 { dx -= lx; }
                if dx < -lx * 0.5 { dx += lx; }
                if dy > ly * 0.5 { dy -= ly; }
                if dy < -ly * 0.5 { dy += ly; }
                let r2 = dx * dx + dy * dy;
                if r2 < cutoff2 {
                    nbrs.push((dy.atan2(dx), dx, dy));
                }
            }
            if nbrs.len() < 3 { continue; }
            nbrs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Build Voronoi polygon: vertex between consecutive neighbors
            // j and k is the circumcenter of triangle (0,0), (dxⱼ,dyⱼ), (dxₖ,dyₖ)
            let nn = nbrs.len();
            let mut verts: Vec<(f64, f64)> = Vec::with_capacity(nn);
            for idx in 0..nn {
                let (_, ax, ay) = nbrs[idx];
                let (_, bx, by) = nbrs[(idx + 1) % nn];
                // Circumcenter of (0,0), (ax,ay), (bx,by)
                let d = 2.0 * (ax * by - ay * bx);
                if d.abs() < 1e-12 {
                    // Degenerate — use midpoint of perpendicular bisectors
                    verts.push(((ax + bx) * 0.25, (ay + by) * 0.25));
                } else {
                    let a2 = ax * ax + ay * ay;
                    let b2 = bx * bx + by * by;
                    let cx = (a2 * by - b2 * ay) / d;
                    let cy = (bx * a2 - ax * b2) / d;
                    verts.push((cx, cy));
                }
            }

            // Polygon perimeter and area (shoelace)
            let nv = verts.len();
            let mut perim = 0.0;
            let mut area = 0.0;
            for vi in 0..nv {
                let (x0, y0) = verts[vi];
                let (x1, y1) = verts[(vi + 1) % nv];
                perim += ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
                area += x0 * y1 - x1 * y0;
            }
            area = area.abs() * 0.5;
            if area > 1e-10 {
                let q = perim / area.sqrt();
                q_accum[i] += q;
                q_count[i] += 1;
            }
        }
    }

    let q_per_cell: Vec<f64> = (0..n_cells)
        .map(|i| if q_count[i] > 0 { q_accum[i] / q_count[i] as f64 } else { 0.0 })
        .collect();
    let valid: Vec<f64> = q_per_cell.iter().filter(|&&v| v > 0.0).copied().collect();
    let q_mean = if valid.is_empty() { 0.0 } else { valid.iter().sum::<f64>() / valid.len() as f64 };
    let q_var = if valid.len() < 2 { 0.0 } else {
        valid.iter().map(|&v| (v - q_mean).powi(2)).sum::<f64>() / valid.len() as f64
    };

    VoronoiShapeResult {
        q_mean,
        q_std: q_var.sqrt(),
        q_per_cell,
    }
}

// ============================================================================
// Kinetic energy time series
// ============================================================================

#[derive(Serialize, Clone, Debug)]
pub struct KineticEnergyResult {
    /// Time points.
    pub times: Vec<f64>,
    /// KE per cell = ½⟨v²⟩ at each time (averaged over all cells).
    pub ke_per_cell: Vec<f64>,
    /// Total KE = ½Σv² at each time.
    pub ke_total: Vec<f64>,
    /// Time-averaged KE per cell.
    pub ke_mean: f64,
}

/// Compute kinetic energy time series from displacement velocities.
pub fn compute_kinetic_energy(pos: &UnwrappedPositions) -> KineticEnergyResult {
    let n_times = pos.n_times;
    let n_cells = pos.n_cells;
    if n_times < 2 {
        return KineticEnergyResult {
            times: vec![],
            ke_per_cell: vec![],
            ke_total: vec![],
            ke_mean: 0.0,
        };
    }

    let mut times = Vec::with_capacity(n_times - 1);
    let mut ke_per_cell = Vec::with_capacity(n_times - 1);
    let mut ke_total = Vec::with_capacity(n_times - 1);

    for t in 1..n_times {
        let dt = pos.times[t] - pos.times[t - 1];
        if dt < 1e-30 { continue; }
        let inv_dt = 1.0 / dt;
        let mut sum_v2 = 0.0;
        for i in 0..n_cells {
            let dx = pos.positions[t][i][0] - pos.positions[t - 1][i][0];
            let dy = pos.positions[t][i][1] - pos.positions[t - 1][i][1];
            let v2 = (dx * inv_dt).powi(2) + (dy * inv_dt).powi(2);
            sum_v2 += v2;
        }
        times.push((pos.times[t] + pos.times[t - 1]) * 0.5);
        ke_total.push(0.5 * sum_v2);
        ke_per_cell.push(0.5 * sum_v2 / n_cells as f64);
    }

    let ke_mean = if ke_per_cell.is_empty() { 0.0 } else {
        ke_per_cell.iter().sum::<f64>() / ke_per_cell.len() as f64
    };

    KineticEnergyResult {
        times,
        ke_per_cell,
        ke_total,
        ke_mean,
    }
}

pub const ALL_OBSERVABLES: &[&str] = &[
    "msd",
    "diffusion",
    "log_slope",
    "cage",
    "alpha2",
    "overlap",
    "structure",
    "scattering",
    "van_hove",
    "per_cell_diffusion",
    "displacement",
    "va_mobility_correlation",
    "spatial_correlation",
    "shape_index",
    "velocity_autocorrelation",
    "burst_detection",
    "velocity_distribution",
    "polarity_tau",
    "hexatic_order",
    "voronoi_shape",
    "kinetic_energy",
];

/// Check if an observable name is valid.
pub fn is_valid_observable(name: &str) -> bool {
    ALL_OBSERVABLES.contains(&name)
}
