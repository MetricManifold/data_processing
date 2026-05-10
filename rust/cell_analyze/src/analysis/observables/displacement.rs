//! Net displacement statistics from first to last frame.
//!
//! Coarse summary used by Phase-0 quench analysis: mean Δr, RMS Δr,
//! max Δr, mean Δr / R.
//!
//! Ported from legacy `observables.rs::compute_displacement`.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct Displacement;

#[derive(Clone, Debug, Serialize)]
pub struct DisplacementOutput {
    pub mean_dr: f64,
    pub rms_dr: f64,
    pub max_dr: f64,
    pub mean_dr_over_r: f64,
}

impl Observable for Displacement {
    type Output = DisplacementOutput;

    fn id(&self) -> &'static str { "displacement" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let cell_radius = ctx.params.cell_radius;
        let n_cells = pos.n_cells;
        if pos.n_times < 2 || n_cells == 0 {
            return Ok(DisplacementOutput {
                mean_dr: 0.0, rms_dr: 0.0, max_dr: 0.0, mean_dr_over_r: 0.0,
            });
        }
        let t_last = pos.n_times - 1;
        let mut sum_dr = 0.0_f64;
        let mut sum_dr2 = 0.0_f64;
        let mut max_dr = 0.0_f64;
        for i in 0..n_cells {
            let dx = pos.positions[t_last][i][0] - pos.positions[0][i][0];
            let dy = pos.positions[t_last][i][1] - pos.positions[0][i][1];
            let dz = pos.positions[t_last][i][2] - pos.positions[0][i][2];
            let dr = (dx * dx + dy * dy + dz * dz).sqrt();
            sum_dr += dr;
            sum_dr2 += dr * dr;
            if dr > max_dr { max_dr = dr; }
        }
        let mean_dr = sum_dr / n_cells as f64;
        let rms_dr = (sum_dr2 / n_cells as f64).sqrt();
        Ok(DisplacementOutput {
            mean_dr, rms_dr, max_dr,
            mean_dr_over_r: mean_dr / cell_radius,
        })
    }
}
