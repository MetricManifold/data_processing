//! Non-Gaussian parameter α₂(Δt) = ⟨Δr⁴⟩ / (2⟨Δr²⟩²) − 1 (2D form).
//!
//! α₂ ≈ 0 in equilibrium / homogeneous diffusion. Peaks near the
//! cage-breaking timescale in glass-formers.
//!
//! Ported from legacy `observables.rs::non_gaussian_parameter`.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct NonGaussian;

#[derive(Clone, Debug, Serialize)]
pub struct NonGaussianOutput {
    pub lag_times: Vec<f64>,
    pub values: Vec<f64>,
}

impl Observable for NonGaussian {
    type Output = NonGaussianOutput;

    fn id(&self) -> &'static str { "alpha2" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let n_times = pos.n_times;
        let n_cells = pos.n_cells;
        if n_times < 2 || n_cells == 0 {
            return Ok(NonGaussianOutput { lag_times: vec![], values: vec![] });
        }
        let max_lag = n_times / 2;
        let dt = if n_times > 1 { pos.times[1] - pos.times[0] } else { 1.0 };
        let n_origins = max_lag;

        let mut r2_sum = vec![0.0_f64; max_lag];
        let mut r4_sum = vec![0.0_f64; max_lag];
        let mut count = vec![0_u64; max_lag];

        for t0 in 0..n_origins {
            for lag in 1..max_lag {
                let ti = t0 + lag;
                if ti >= n_times { break; }
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
                let denom = 2.0 * mean_r2 * mean_r2;
                let a2 = if denom > 0.0 { mean_r4 / denom - 1.0 } else { 0.0 };
                lag_times.push(lag as f64 * dt);
                values.push(a2);
            }
        }
        Ok(NonGaussianOutput { lag_times, values })
    }
}
