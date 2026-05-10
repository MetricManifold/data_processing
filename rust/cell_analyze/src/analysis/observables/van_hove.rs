//! van Hove self-correlation G_s(Δx, t).
//!
//! Histogram of single-cell x-displacements at lag times {0.1τ, τ, 3τ}.
//! Gaussian for normal diffusion; exponential tails are the signature
//! of dynamic heterogeneity / hopping.
//!
//! Ported from legacy `observables.rs::van_hove`.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct VanHove {
    pub n_bins: usize,
}

impl Default for VanHove {
    fn default() -> Self { Self { n_bins: 80 } }
}

#[derive(Clone, Debug, Serialize)]
pub struct VanHoveLag {
    pub lag_time: f64,
    pub histogram: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct VanHoveOutput {
    pub dx_bins: Vec<f64>,
    pub distributions: Vec<VanHoveLag>,
}

impl Observable for VanHove {
    type Output = VanHoveOutput;

    fn id(&self) -> &'static str { "van_hove" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let n_bins = self.n_bins;
        let n_times = pos.n_times;
        let n_cells = pos.n_cells;
        let tau = ctx.params.tau;
        let dt = if n_times > 1 { pos.times[1] - pos.times[0] } else { 1.0 };
        if n_times < 2 || n_cells == 0 {
            return Ok(VanHoveOutput { dx_bins: vec![], distributions: vec![] });
        }
        let target_lags: Vec<f64> = vec![0.1 * tau, tau, 3.0 * tau];
        let lag_indices: Vec<usize> = target_lags.iter()
            .map(|&t| ((t / dt).round() as usize).max(1).min(n_times - 1))
            .collect();
        let max_lag = *lag_indices.iter().max().unwrap_or(&1);
        let mut max_dx = 1.0_f64;
        let n_sample = (n_times - max_lag).min(50);
        for t0_idx in 0..n_sample {
            let t0 = t0_idx * (n_times - max_lag) / n_sample.max(1);
            let ti = t0 + max_lag;
            if ti >= n_times { break; }
            for i in 0..n_cells {
                let dx = (pos.positions[ti][i][0] - pos.positions[t0][i][0]).abs();
                if dx > max_dx { max_dx = dx; }
            }
        }
        let r_max = max_dx * 1.5;
        let bin_width = 2.0 * r_max / n_bins as f64;
        let dx_bins: Vec<f64> = (0..n_bins).map(|i| -r_max + (i as f64 + 0.5) * bin_width).collect();
        let mut distributions = Vec::new();
        for &lag in &lag_indices {
            if lag >= n_times { continue; }
            let mut hist = vec![0.0_f64; n_bins];
            let mut total = 0_u64;
            let n_origins = n_times - lag;
            for t0 in 0..n_origins {
                let ti = t0 + lag;
                for i in 0..n_cells {
                    let dx = pos.positions[ti][i][0] - pos.positions[t0][i][0];
                    let b = ((dx + r_max) / bin_width) as usize;
                    if b < n_bins { hist[b] += 1.0; total += 1; }
                }
            }
            if total > 0 {
                let norm = total as f64 * bin_width;
                for v in &mut hist { *v /= norm; }
            }
            distributions.push(VanHoveLag { lag_time: lag as f64 * dt, histogram: hist });
        }
        Ok(VanHoveOutput { dx_bins, distributions })
    }
}
