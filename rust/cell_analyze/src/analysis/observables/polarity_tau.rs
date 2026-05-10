//! Persistence time τ from polarity autocorrelation.
//!
//! ⟨p̂(t+Δt)·p̂(t)⟩ = exp(−Δt / τ). Reads polarity (px, py) from the
//! raw trajectory file (cols 6,7 in the 12-col 2D schema), so requires
//! `Trajectory`, not just unwrapped positions.
//!
//! Returns the population fit + per-cell τ array + mean/std across cells.
//!
//! Ported from legacy `observables.rs::polarity_tau`.

use anyhow::{anyhow, Result};
use serde::Serialize;

use super::fits::fit_exp_decay;
use crate::analysis::observable::{Context, Observable, Requirements};

pub struct PolarityTau;

#[derive(Clone, Debug, Serialize)]
pub struct PolarityTauOutput {
    pub tau: f64,
    pub fit_r2: f64,
    pub lag_times: Vec<f64>,
    pub correlation: Vec<f64>,
    pub per_cell_tau: Vec<(u32, f64)>,
    pub tau_mean: f64,
    pub tau_std: f64,
}

impl Observable for PolarityTau {
    type Output = PolarityTauOutput;

    fn id(&self) -> &'static str { "polarity_tau" }
    fn requires(&self) -> Requirements { Requirements::TRAJECTORY }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let traj = ctx.trajectory.as_ref()
            .ok_or_else(|| anyhow!("polarity_tau requires trajectory but none loaded"))?;
        let frames = &traj.frames;
        let n_frames = frames.len();
        let empty = PolarityTauOutput {
            tau: 0.0, fit_r2: 0.0,
            lag_times: vec![], correlation: vec![],
            per_cell_tau: vec![], tau_mean: 0.0, tau_std: 0.0,
        };
        if n_frames < 10 { return Ok(empty); }
        let times: Vec<f64> = frames.iter().map(|(t, _)| *t).collect();
        let dt_frame = if n_frames >= 2 { times[1] - times[0] } else { 1.0 };
        let cell_ids: Vec<u32> = {
            let mut ids: Vec<u32> = frames[0].1.keys().copied().collect();
            ids.sort();
            ids
        };
        let n_cells = cell_ids.len();
        let has_polarity = frames.iter().any(|(_, cells)| {
            cells.values().any(|s| s.px.abs() > 1e-10 || s.py.abs() > 1e-10)
        });
        if !has_polarity { return Ok(empty); }
        let mut pol: Vec<Vec<(f64, f64)>> = vec![Vec::with_capacity(n_frames); n_cells];
        for (_, cells) in frames {
            for (ci, &cid) in cell_ids.iter().enumerate() {
                if let Some(snap) = cells.get(&cid) {
                    pol[ci].push((snap.px, snap.py));
                } else {
                    pol[ci].push((0.0, 0.0));
                }
            }
        }
        let max_lag = n_frames / 4;
        if max_lag < 3 { return Ok(empty); }
        let mut lag_indices: Vec<usize> = Vec::new();
        let mut lag = 1usize;
        while lag <= max_lag {
            lag_indices.push(lag);
            lag = std::cmp::max(lag + 1, (lag as f64 * 1.2).ceil() as usize);
        }
        lag_indices.dedup();

        let mut lag_times_out: Vec<f64> = Vec::new();
        let mut corr_out: Vec<f64> = Vec::new();
        for &lag in &lag_indices {
            let mut dot_sum = 0.0;
            let mut count = 0usize;
            for ci in 0..n_cells {
                for t in 0..(n_frames - lag) {
                    let (px0, py0) = pol[ci][t];
                    let (px1, py1) = pol[ci][t + lag];
                    let norm0 = (px0 * px0 + py0 * py0).sqrt();
                    let norm1 = (px1 * px1 + py1 * py1).sqrt();
                    if norm0 > 1e-10 && norm1 > 1e-10 {
                        dot_sum += (px0 * px1 + py0 * py1) / (norm0 * norm1);
                        count += 1;
                    }
                }
            }
            if count > 0 {
                let c = dot_sum / count as f64;
                if c > 0.01 {
                    lag_times_out.push(lag as f64 * dt_frame);
                    corr_out.push(c);
                }
            }
        }
        if corr_out.len() < 3 {
            return Ok(PolarityTauOutput {
                tau: 0.0, fit_r2: 0.0,
                lag_times: lag_times_out, correlation: corr_out,
                per_cell_tau: vec![], tau_mean: 0.0, tau_std: 0.0,
            });
        }
        let (tau_pop, r2_pop) = fit_exp_decay(&lag_times_out, &corr_out);
        let mut per_cell_tau: Vec<(u32, f64)> = Vec::new();
        for (ci, &cid) in cell_ids.iter().enumerate() {
            let mut cell_lags: Vec<f64> = Vec::new();
            let mut cell_corr: Vec<f64> = Vec::new();
            for &lag in &lag_indices {
                let mut dot_sum = 0.0;
                let mut count = 0usize;
                for t in 0..(n_frames - lag) {
                    let (px0, py0) = pol[ci][t];
                    let (px1, py1) = pol[ci][t + lag];
                    let norm0 = (px0 * px0 + py0 * py0).sqrt();
                    let norm1 = (px1 * px1 + py1 * py1).sqrt();
                    if norm0 > 1e-10 && norm1 > 1e-10 {
                        dot_sum += (px0 * px1 + py0 * py1) / (norm0 * norm1);
                        count += 1;
                    }
                }
                if count > 10 {
                    let c = dot_sum / count as f64;
                    if c > 0.01 {
                        cell_lags.push(lag as f64 * dt_frame);
                        cell_corr.push(c);
                    }
                }
            }
            if cell_corr.len() >= 3 {
                let (cell_tau, _) = fit_exp_decay(&cell_lags, &cell_corr);
                if cell_tau > 0.0 && cell_tau.is_finite() {
                    per_cell_tau.push((cid, cell_tau));
                }
            }
        }
        let tau_mean = if per_cell_tau.is_empty() { tau_pop } else {
            per_cell_tau.iter().map(|(_, t)| t).sum::<f64>() / per_cell_tau.len() as f64
        };
        let tau_std = if per_cell_tau.len() < 2 { 0.0 } else {
            let var = per_cell_tau.iter().map(|(_, t)| (t - tau_mean).powi(2)).sum::<f64>()
                / (per_cell_tau.len() - 1) as f64;
            var.sqrt()
        };
        Ok(PolarityTauOutput {
            tau: tau_pop, fit_r2: r2_pop,
            lag_times: lag_times_out, correlation: corr_out,
            per_cell_tau, tau_mean, tau_std,
        })
    }
}
