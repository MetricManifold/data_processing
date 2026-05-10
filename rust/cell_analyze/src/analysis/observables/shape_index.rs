//! Shape index from L_n column (perimeter / 2πR).
//!
//! p_eff = L_n × 2√π. Bi/Manning vertex-model glass-fluid transition
//! is at p₀* ≈ 3.81 in 2D. Reads L_n directly from the trajectory file
//! (column 11 in the 12-col 2D schema) — needs the raw `Trajectory`,
//! not just unwrapped positions.
//!
//! Ported from legacy `observables.rs::shape_index`.

use anyhow::{anyhow, Result};
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct ShapeIndex;

#[derive(Clone, Debug, Serialize)]
pub struct ShapeIndexOutput {
    pub mean_p: f64,
    pub std_p: f64,
    pub per_cell_p: Vec<f64>,
    pub p_vs_time: Vec<f64>,
    pub cell0_p_vs_time: Vec<f64>,
    pub times: Vec<f64>,
    pub n_frames: usize,
}

impl Observable for ShapeIndex {
    type Output = ShapeIndexOutput;

    fn id(&self) -> &'static str { "shape_index" }
    fn requires(&self) -> Requirements { Requirements::TRAJECTORY }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let traj = ctx.trajectory.as_ref()
            .ok_or_else(|| anyhow!("shape_index requires trajectory but none loaded"))?;
        let factor = 2.0 * std::f64::consts::PI.sqrt();
        let mut times = Vec::new();
        let mut p_vs_time = Vec::new();
        let cell_ids = traj.cell_ids();
        let cell0_id = cell_ids.first().copied().unwrap_or(0);
        let mut cell0_p_vs_time = Vec::new();

        for (t, frame) in &traj.frames {
            let mut sum_p = 0.0;
            let mut count = 0usize;
            for &cid in &cell_ids {
                if let Some(snap) = frame.get(&cid) {
                    if snap.l_n > 0.0 {
                        sum_p += snap.l_n * factor;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                times.push(*t);
                p_vs_time.push(sum_p / count as f64);
                let c0_p = frame.get(&cell0_id).map_or(0.0, |s| s.l_n * factor);
                cell0_p_vs_time.push(c0_p);
            }
        }

        let mut per_cell_p = Vec::new();
        let mut last_sum = 0.0;
        let mut last_sum_sq = 0.0;
        let mut last_count = 0usize;
        if let Some((_, frame)) = traj.frames.last() {
            for &cid in &cell_ids {
                if let Some(snap) = frame.get(&cid) {
                    let p = snap.l_n * factor;
                    per_cell_p.push(p);
                    if snap.l_n > 0.0 {
                        last_sum += p;
                        last_sum_sq += p * p;
                        last_count += 1;
                    }
                }
            }
        }
        let mean_p = if last_count > 0 { last_sum / last_count as f64 } else { 0.0 };
        let var = if last_count > 1 { (last_sum_sq / last_count as f64) - mean_p * mean_p } else { 0.0 };
        let std_p = if var > 0.0 { var.sqrt() } else { 0.0 };
        let n_frames = times.len();
        Ok(ShapeIndexOutput {
            mean_p, std_p, per_cell_p,
            p_vs_time, cell0_p_vs_time, times, n_frames,
        })
    }
}
