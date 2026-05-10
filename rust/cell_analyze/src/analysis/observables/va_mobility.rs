//! Pearson correlation between inherent v_A and time-averaged speed.
//!
//! Tests whether per-cell heterogeneity in the active speed (sampled
//! at init from the v_A_sigma log-normal) is reflected in the cell's
//! actual time-averaged motility — important sanity check for v_A
//! disorder runs.
//!
//! Ported from legacy `observables.rs::va_mobility_correlation`.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct VaMobilityCorrelation;

#[derive(Clone, Debug, Serialize)]
pub struct VaMobilityCorrelationOutput {
    pub pearson_r: f64,
    pub n_cells: usize,
    pub cell_speeds: Vec<f64>,
    pub cell_va: Vec<f64>,
}

impl Observable for VaMobilityCorrelation {
    type Output = VaMobilityCorrelationOutput;

    fn id(&self) -> &'static str { "va_mobility_correlation" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let n_cells = pos.n_cells;
        let n_times = pos.n_times;
        if pos.inherent_v_a.is_empty() || n_times < 2 || n_cells == 0 {
            return Ok(VaMobilityCorrelationOutput {
                pearson_r: f64::NAN, n_cells,
                cell_speeds: vec![], cell_va: vec![],
            });
        }
        let dt = if n_times > 1 { pos.times[1] - pos.times[0] } else { 1.0 };
        let mut speeds = vec![0.0_f64; n_cells];
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
        Ok(VaMobilityCorrelationOutput {
            pearson_r, n_cells,
            cell_speeds: speeds,
            cell_va: va.clone(),
        })
    }
}
