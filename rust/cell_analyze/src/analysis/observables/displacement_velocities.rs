//! Displacement-velocity time series for the tagged cell.
//!
//! Computes per-frame velocity (vx, vy) and speed |v| from successive
//! unwrapped positions, plus the mean speed. Used by the speed-bursts
//! panel and the velocity distribution / G(v) panels.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct DisplacementVelocities;

#[derive(Clone, Debug, Serialize)]
pub struct DisplacementVelocitiesOutput {
    /// Mid-point time of each velocity sample, in τ units.
    pub t_tau: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub speeds: Vec<f64>,
    pub mean_speed: f64,
    /// Standard deviation of speed (used to set the burst threshold).
    pub std_speed: f64,
}

impl Observable for DisplacementVelocities {
    type Output = DisplacementVelocitiesOutput;

    fn id(&self) -> &'static str {
        "displacement_velocities"
    }

    fn requires(&self) -> Requirements {
        Requirements::POSITIONS
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let tau = ctx.params.tau;
        let n = pos.n_times;
        let dt = if n > 1 { pos.times[1] - pos.times[0] } else { 1.0 };
        let tagged: u32 = ctx.params.tagged_cells.first().copied().unwrap_or(0);
        let idx = pos
            .cell_ids
            .iter()
            .position(|&c| c == tagged)
            .unwrap_or(0);

        let mut vx = Vec::with_capacity(n.saturating_sub(1));
        let mut vy = Vec::with_capacity(n.saturating_sub(1));
        let mut speeds = Vec::with_capacity(n.saturating_sub(1));
        for i in 1..n {
            let dx = pos.positions[i][idx][0] - pos.positions[i - 1][idx][0];
            let dy = pos.positions[i][idx][1] - pos.positions[i - 1][idx][1];
            vx.push(dx / dt);
            vy.push(dy / dt);
            speeds.push((dx * dx + dy * dy).sqrt() / dt);
        }
        let t_tau: Vec<f64> = (0..speeds.len())
            .map(|i| (i as f64 + 0.5) * dt / tau)
            .collect();

        let mean_speed = if speeds.is_empty() {
            0.0
        } else {
            speeds.iter().sum::<f64>() / speeds.len() as f64
        };
        let std_speed = if speeds.len() > 1 {
            let var: f64 = speeds
                .iter()
                .map(|v| (v - mean_speed).powi(2))
                .sum::<f64>()
                / (speeds.len() as f64 - 1.0);
            var.sqrt()
        } else {
            0.0
        };

        Ok(DisplacementVelocitiesOutput {
            t_tau,
            vx,
            vy,
            speeds,
            mean_speed,
            std_speed,
        })
    }
}
