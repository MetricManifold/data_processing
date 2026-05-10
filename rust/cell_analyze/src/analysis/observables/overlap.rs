//! Self-overlap Q(t) and four-point susceptibility χ₄(t).
//!
//! Q(t) = ⟨ N⁻¹ Σᵢ Θ(a − |Δrᵢ(t)|) ⟩ — fraction of cells that have NOT
//! moved further than the cage radius `a` at time `t`.
//! χ₄(t) = N · Var(Q(t)) — dynamic heterogeneity (peaks near τ_α).
//!
//! Also fits the Q(t) decay to a stretched exponential giving (τ_α, β, R²).
//!
//! Default cage radius is the cell's effective radius (params.cell_radius);
//! struct field `cage_radius` overrides if non-NaN.
//!
//! Ported from legacy `observables.rs::overlap_and_chi4`.

use anyhow::Result;
use serde::Serialize;

use super::fits::fit_stretched_exponential;
use crate::analysis::observable::{Context, Observable, Requirements};

pub struct Overlap {
    /// Cage radius `a`. NaN → use ctx.params.cell_radius.
    pub cage_radius: f64,
}

impl Default for Overlap {
    fn default() -> Self { Self { cage_radius: f64::NAN } }
}

#[derive(Clone, Debug, Serialize)]
pub struct OverlapOutput {
    pub lag_times: Vec<f64>,
    pub q_mean: Vec<f64>,
    pub chi4: Vec<f64>,
    pub tau_alpha: f64,
    pub beta: f64,
    pub fit_r2: f64,
}

impl Observable for Overlap {
    type Output = OverlapOutput;

    fn id(&self) -> &'static str { "overlap_chi4" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let n_times = pos.n_times;
        let n_cells = pos.n_cells;
        let cage_radius = if self.cage_radius.is_nan() {
            ctx.params.cell_radius
        } else {
            self.cage_radius
        };
        let empty = OverlapOutput {
            lag_times: vec![], q_mean: vec![], chi4: vec![],
            tau_alpha: f64::NAN, beta: f64::NAN, fit_r2: 0.0,
        };
        if n_times < 2 || n_cells == 0 { return Ok(empty); }
        let max_lag = n_times / 2;
        let dt = if n_times > 1 { pos.times[1] - pos.times[0] } else { 1.0 };
        let n_origins = max_lag;

        let mut q_per_origin: Vec<Vec<f64>> = vec![Vec::new(); max_lag];
        for t0 in 0..n_origins {
            for lag in 0..max_lag {
                let ti = t0 + lag;
                if ti >= n_times { break; }
                let mut overlap_count = 0_u32;
                for i in 0..n_cells {
                    let dx = pos.positions[ti][i][0] - pos.positions[t0][i][0];
                    let dy = pos.positions[ti][i][1] - pos.positions[t0][i][1];
                    let dz = pos.positions[ti][i][2] - pos.positions[t0][i][2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < cage_radius { overlap_count += 1; }
                }
                q_per_origin[lag].push(overlap_count as f64 / n_cells as f64);
            }
        }
        let mut lag_times = Vec::with_capacity(max_lag);
        let mut q_mean = Vec::with_capacity(max_lag);
        let mut chi4 = Vec::with_capacity(max_lag);
        for lag in 0..max_lag {
            let vals = &q_per_origin[lag];
            if vals.is_empty() { continue; }
            let n = vals.len() as f64;
            let mean: f64 = vals.iter().sum::<f64>() / n;
            let var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            lag_times.push(lag as f64 * dt);
            q_mean.push(mean);
            chi4.push(n_cells as f64 * var);
        }
        let (tau_alpha, beta, fit_r2) = fit_stretched_exponential(&lag_times, &q_mean);
        Ok(OverlapOutput { lag_times, q_mean, chi4, tau_alpha, beta, fit_r2 })
    }
}
