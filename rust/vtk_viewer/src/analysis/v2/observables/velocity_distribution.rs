//! Velocity distribution observable: histogram + raw (vx, vy) samples
//! per cell, plus second-moment σ and excess kurtosis.
//!
//! Velocities are computed as centroid finite-differences over
//! successive frames of `UnwrappedPositions` (so periodic-image
//! crossings are already handled by the unwrap step).

use anyhow::Result;
use serde::Serialize;

use crate::analysis::io::UnwrappedPositions;
use crate::analysis::v2::observable::{Context, Observable, Requirements};

pub struct VelocityDistribution {
    pub n_bins: usize,
}

impl Default for VelocityDistribution {
    fn default() -> Self {
        Self { n_bins: 80 }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VelocityDistributionOutput {
    pub bin_edges: Vec<f64>,
    pub cell0_hist: Vec<f64>,
    pub pop_hist: Vec<f64>,
    pub cell0_sigma_vx: f64,
    pub pop_sigma_vx: f64,
    pub cell0_kurtosis: f64,
    pub pop_kurtosis: f64,
    pub cell0_mean_speed: f64,
    pub pop_mean_speed: f64,
    /// Raw vx samples for cell 0 (kept in-memory for the G(v_i) panel).
    /// Skipped during JSON serialization to keep `RunAnalysis.json` small.
    #[serde(skip, default)]
    pub cell0_vx: Vec<f64>,
    #[serde(skip, default)]
    pub cell0_vy: Vec<f64>,
    #[serde(skip, default)]
    pub pop_vx: Vec<f64>,
    #[serde(skip, default)]
    pub pop_vy: Vec<f64>,
}

/// Stand-alone compute, public for sites that have positions in hand
/// but don't want to round-trip through the full `Observable` machinery.
pub fn compute_velocity_distribution(
    pos: &UnwrappedPositions,
    n_bins: usize,
) -> VelocityDistributionOutput {
    let n = pos.n_times;
    let nc = pos.n_cells;
    let dt = if n > 1 { pos.times[1] - pos.times[0] } else { 1.0 };

    let empty = VelocityDistributionOutput {
        bin_edges: vec![],
        cell0_hist: vec![],
        pop_hist: vec![],
        cell0_sigma_vx: 0.0,
        pop_sigma_vx: 0.0,
        cell0_kurtosis: 0.0,
        pop_kurtosis: 0.0,
        cell0_mean_speed: 0.0,
        pop_mean_speed: 0.0,
        cell0_vx: vec![],
        cell0_vy: vec![],
        pop_vx: vec![],
        pop_vy: vec![],
    };
    if n < 3 || nc == 0 {
        return empty;
    }

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

    let mean_f = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let std_f = |v: &[f64], m: f64| {
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };
    let kurtosis_f = |v: &[f64], m: f64, s: f64| {
        if s < 1e-30 {
            return 0.0;
        }
        let nn = v.len() as f64;
        let m4 = v.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / nn;
        m4 - 3.0 // excess kurtosis
    };

    let c0_vx_mean = mean_f(&cell0_vx);
    let c0_vx_std = std_f(&cell0_vx, c0_vx_mean);
    let c0_kurt = kurtosis_f(&cell0_vx, c0_vx_mean, c0_vx_std);
    let c0_mean_speed = mean_f(&cell0_speeds);

    let pop_vx_mean = mean_f(&pop_vx);
    let pop_vx_std = std_f(&pop_vx, pop_vx_mean);
    let pop_kurt = kurtosis_f(&pop_vx, pop_vx_mean, pop_vx_std);
    let pop_mean_speed = mean_f(&pop_speeds);

    let v_max = pop_vx.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    let bin_width = 2.0 * v_max / n_bins as f64;
    let bin_edges: Vec<f64> = (0..=n_bins)
        .map(|i| -v_max + i as f64 * bin_width)
        .collect();

    let histogramize = |data: &[f64]| -> Vec<f64> {
        let mut counts = vec![0u64; n_bins];
        for &vx in data {
            let idx = ((vx + v_max) / bin_width) as usize;
            let idx = idx.min(n_bins - 1);
            counts[idx] += 1;
        }
        let total = data.len() as f64;
        counts
            .iter()
            .map(|&c| c as f64 / (total * bin_width))
            .collect()
    };

    let cell0_hist = histogramize(&cell0_vx);
    let pop_hist = histogramize(&pop_vx);

    VelocityDistributionOutput {
        bin_edges,
        cell0_hist,
        pop_hist,
        cell0_sigma_vx: c0_vx_std,
        pop_sigma_vx: pop_vx_std,
        cell0_kurtosis: c0_kurt,
        pop_kurtosis: pop_kurt,
        cell0_mean_speed: c0_mean_speed,
        pop_mean_speed: pop_mean_speed,
        cell0_vx,
        cell0_vy,
        pop_vx,
        pop_vy,
    }
}

impl Observable for VelocityDistribution {
    type Output = VelocityDistributionOutput;

    fn id(&self) -> &'static str {
        "velocity_distribution"
    }

    fn requires(&self) -> Requirements {
        Requirements::POSITIONS
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        Ok(compute_velocity_distribution(&ctx.positions, self.n_bins))
    }
}
