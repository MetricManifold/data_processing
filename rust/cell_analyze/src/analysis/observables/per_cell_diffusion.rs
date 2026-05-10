//! Per-cell long-time diffusion coefficient.
//!
//! D_i = MSD_i(8τ) / (4·8τ) for each cell. Returns mean, std, CV across
//! cells. Useful for soft-vs-hard contrast and for D_i histograms.
//!
//! Ported from legacy `observables.rs::per_cell_diffusion`.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct PerCellDiffusion;

#[derive(Clone, Debug, Serialize)]
pub struct PerCellDiffusionOutput {
    pub cell_ids: Vec<u32>,
    pub d_values: Vec<f64>,
    pub d_mean: f64,
    pub d_std: f64,
    pub cv: f64,
}

impl Observable for PerCellDiffusion {
    type Output = PerCellDiffusionOutput;

    fn id(&self) -> &'static str { "per_cell_diffusion" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let tau = ctx.params.tau;
        let n_times = pos.n_times;
        let n_cells = pos.n_cells;
        let dt = if n_times > 1 { pos.times[1] - pos.times[0] } else { 1.0 };
        if n_times < 10 || n_cells == 0 {
            return Ok(PerCellDiffusionOutput {
                cell_ids: pos.cell_ids.clone(),
                d_values: vec![0.0; n_cells],
                d_mean: 0.0, d_std: 0.0, cv: 0.0,
            });
        }
        let max_lag = n_times / 2;
        let d_divisor = if pos.dim == 3 { 6.0 } else { 4.0 };
        let lag_8tau = ((8.0 * tau / dt).round() as usize).min(max_lag).max(1);
        let mut d_values = vec![0.0_f64; n_cells];
        for i in 0..n_cells {
            let n_origins = n_times - lag_8tau;
            if n_origins < 2 { continue; }
            let lag_time = lag_8tau as f64 * dt;
            let mut msd_sum = 0.0_f64;
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
        let d_std = (d_values.iter().map(|d| (d - d_mean).powi(2)).sum::<f64>()
            / n_cells as f64).sqrt();
        Ok(PerCellDiffusionOutput {
            cell_ids: pos.cell_ids.clone(),
            d_values, d_mean, d_std,
            cv: if d_mean > 0.0 { d_std / d_mean } else { 0.0 },
        })
    }
}
