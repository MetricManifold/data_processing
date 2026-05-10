//! Kinetic energy time series ½Σv² (per-cell average and total).
//!
//! Velocity from forward finite differences. Useful as an equilibration
//! diagnostic.
//!
//! Ported from legacy `observables.rs::compute_kinetic_energy`.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct KineticEnergy;

#[derive(Clone, Debug, Serialize)]
pub struct KineticEnergyOutput {
    pub times: Vec<f64>,
    pub ke_per_cell: Vec<f64>,
    pub ke_total: Vec<f64>,
    pub ke_mean: f64,
}

impl Observable for KineticEnergy {
    type Output = KineticEnergyOutput;

    fn id(&self) -> &'static str { "kinetic_energy" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let n_times = pos.n_times;
        let n_cells = pos.n_cells;
        if n_times < 2 {
            return Ok(KineticEnergyOutput {
                times: vec![], ke_per_cell: vec![], ke_total: vec![], ke_mean: 0.0,
            });
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
                let dx = (pos.positions[t][i][0] - pos.positions[t - 1][i][0]) * inv_dt;
                let dy = (pos.positions[t][i][1] - pos.positions[t - 1][i][1]) * inv_dt;
                let dz = (pos.positions[t][i][2] - pos.positions[t - 1][i][2]) * inv_dt;
                sum_v2 += dx * dx + dy * dy + dz * dz;
            }
            let total = 0.5 * sum_v2;
            times.push(pos.times[t]);
            ke_per_cell.push(total / n_cells.max(1) as f64);
            ke_total.push(total);
        }
        let ke_mean = if ke_per_cell.is_empty() {
            0.0
        } else {
            ke_per_cell.iter().sum::<f64>() / ke_per_cell.len() as f64
        };
        Ok(KineticEnergyOutput { times, ke_per_cell, ke_total, ke_mean })
    }
}
