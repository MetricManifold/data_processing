//! Speed-burst detection: identifies frames where |v| exceeds
//! `μ + k·σ` for at least `min_frames` consecutive samples.
//!
//! Wraps the legacy
//! [`crate::analysis::observables::detect_bursts`] for v2.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observables as legacy;
use crate::analysis::v2::observable::{Context, Observable, Requirements};

pub struct Bursts {
    pub k_sigma: f64,
    pub min_frames: usize,
}

impl Default for Bursts {
    fn default() -> Self {
        Self {
            k_sigma: 3.0,
            min_frames: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BurstEvent {
    pub cell_id: u32,
    pub t_start: f64,
    pub t_end: f64,
    pub duration: f64,
    pub peak_speed: f64,
    pub mean_speed: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct BurstsOutput {
    pub threshold: f64,
    pub speed_mean: f64,
    pub speed_std: f64,
    pub total_bursts: usize,
    pub mean_bursts_per_cell: f64,
    pub mean_duration: f64,
    pub mean_peak_speed: f64,
    pub events: Vec<BurstEvent>,
}

impl Observable for Bursts {
    type Output = BurstsOutput;

    fn id(&self) -> &'static str {
        "bursts"
    }

    fn requires(&self) -> Requirements {
        Requirements::POSITIONS | Requirements::TRAJECTORY
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let traj = ctx
            .trajectory
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("bursts requires trajectory"))?;
        let r = legacy::detect_bursts(&ctx.positions, traj, self.k_sigma, self.min_frames);
        Ok(BurstsOutput {
            threshold: r.threshold,
            speed_mean: r.speed_mean,
            speed_std: r.speed_std,
            total_bursts: r.total_bursts,
            mean_bursts_per_cell: r.mean_bursts_per_cell,
            mean_duration: r.mean_duration,
            mean_peak_speed: r.mean_peak_speed,
            events: r
                .events
                .into_iter()
                .map(|e| BurstEvent {
                    cell_id: e.cell_id,
                    t_start: e.t_start,
                    t_end: e.t_end,
                    duration: e.duration,
                    peak_speed: e.peak_speed,
                    mean_speed: e.mean_speed,
                })
                .collect(),
        })
    }
}
