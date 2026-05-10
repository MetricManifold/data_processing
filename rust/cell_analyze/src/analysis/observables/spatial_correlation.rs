//! Spatial autocorrelation C(r) of instantaneous mobility.
//!
//! For each pair (i, j), correlate δm_i δm_j vs |r_i − r_j|, where m
//! is per-cell single-frame displacement magnitude (mobility proxy).
//! Extracts ξ where C(r) = 1/e.
//!
//! Ported from legacy `observables.rs::spatial_correlation`.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct SpatialCorrelation {
    pub n_bins: usize,
}

impl Default for SpatialCorrelation {
    fn default() -> Self { Self { n_bins: 60 } }
}

#[derive(Clone, Debug, Serialize)]
pub struct SpatialCorrelationOutput {
    pub r_bins: Vec<f64>,
    pub c_r: Vec<f64>,
    pub xi: f64,
}

impl Observable for SpatialCorrelation {
    type Output = SpatialCorrelationOutput;

    fn id(&self) -> &'static str { "spatial_correlation" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let n_bins = self.n_bins;
        let n_cells = pos.n_cells;
        let n_times = pos.n_times;
        let lx = pos.lx;
        let ly = pos.ly;
        if n_times < 2 || n_cells < 2 {
            return Ok(SpatialCorrelationOutput { r_bins: vec![], c_r: vec![], xi: f64::NAN });
        }
        let r_max = lx.min(ly) / 2.0;
        let dr = r_max / n_bins as f64;
        let mut c_sum = vec![0.0_f64; n_bins];
        let mut counts = vec![0_u64; n_bins];
        let n_sample = 20.min(n_times);
        let step = (n_times / 2).max(1) / n_sample.max(1);
        for f_idx in 0..n_sample {
            let t = n_times / 2 + f_idx * step.max(1);
            if t >= n_times || t == 0 { continue; }
            let mut mobility = vec![0.0_f64; n_cells];
            for i in 0..n_cells {
                let dx = pos.positions[t][i][0] - pos.positions[t - 1][i][0];
                let dy = pos.positions[t][i][1] - pos.positions[t - 1][i][1];
                let dz = pos.positions[t][i][2] - pos.positions[t - 1][i][2];
                mobility[i] = (dx * dx + dy * dy + dz * dz).sqrt();
            }
            let mean_m = mobility.iter().sum::<f64>() / n_cells as f64;
            let var_m: f64 = mobility.iter().map(|m| (m - mean_m).powi(2)).sum::<f64>() / n_cells as f64;
            if var_m < 1e-30 { continue; }
            let wrapped: Vec<[f64; 2]> = (0..n_cells).map(|i| pos.wrapped(t, i)).collect();
            for i in 0..n_cells {
                for j in (i + 1)..n_cells {
                    let mut dx = wrapped[i][0] - wrapped[j][0];
                    let mut dy = wrapped[i][1] - wrapped[j][1];
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
            if counts[i] > 0 { c_r[i] = c_sum[i] / counts[i] as f64; }
        }
        let threshold = 1.0 / std::f64::consts::E;
        let xi = r_bins.iter().zip(c_r.iter()).skip(1)
            .find(|(_, &c)| c < threshold)
            .map_or(r_max, |(&r, _)| r);
        Ok(SpatialCorrelationOutput { r_bins, c_r, xi })
    }
}
