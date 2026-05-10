//! Self-intermediate scattering function F_s(q*, t).
//!
//! F_s(q, t) = ⟨ N⁻¹ Σⱼ cos(q · Δrⱼ(t)) ⟩, averaged over 4 q-vector
//! orientations at |q| = q*. Decay gives the structural relaxation time
//! τ_α via stretched-exponential fit.
//!
//! `q_star` field defaults to NaN → look up StructureFactor in the bag
//! (computes one transparently if missing).
//!
//! Ported from legacy `observables.rs::self_intermediate_scattering`.

use anyhow::Result;
use serde::Serialize;
use std::f64::consts::PI;

use super::fits::fit_stretched_exponential;
use super::structure_factor::StructureFactor;
use crate::analysis::observable::{Context, Observable, Requirements};

pub struct SelfScattering {
    /// |q*|. NaN → derive from StructureFactor.
    pub q_star: f64,
}

impl Default for SelfScattering {
    fn default() -> Self { Self { q_star: f64::NAN } }
}

#[derive(Clone, Debug, Serialize)]
pub struct SelfScatteringOutput {
    pub q_star: f64,
    pub lag_times: Vec<f64>,
    pub fs: Vec<f64>,
    pub tau_alpha: f64,
    pub beta: f64,
    pub fit_r2: f64,
}

impl Observable for SelfScattering {
    type Output = SelfScatteringOutput;

    fn id(&self) -> &'static str { "fs_qstar" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let n_times = pos.n_times;
        let n_cells = pos.n_cells;
        let q_star = if self.q_star.is_nan() {
            StructureFactor::default().compute(ctx)?.q_star
        } else {
            self.q_star
        };
        let empty = SelfScatteringOutput {
            q_star, lag_times: vec![], fs: vec![],
            tau_alpha: f64::NAN, beta: f64::NAN, fit_r2: 0.0,
        };
        if n_times < 2 || n_cells == 0 || q_star <= 0.0 { return Ok(empty); }

        let max_lag = n_times / 2;
        let dt = if n_times > 1 { pos.times[1] - pos.times[0] } else { 1.0 };
        let q_vectors: [[f64; 2]; 4] = [
            [q_star, 0.0], [0.0, q_star],
            [q_star * (PI / 4.0).cos(), q_star * (PI / 4.0).sin()],
            [q_star * (3.0 * PI / 4.0).cos(), q_star * (3.0 * PI / 4.0).sin()],
        ];
        let n_origins = max_lag;
        let mut fs_sum = vec![0.0_f64; max_lag];
        let mut fs_count = vec![0_u64; max_lag];

        for t0 in 0..n_origins {
            for lag in 0..max_lag {
                let ti = t0 + lag;
                if ti >= n_times { break; }
                let mut fs_val = 0.0;
                for qv in &q_vectors {
                    let mut cos_sum = 0.0;
                    for i in 0..n_cells {
                        let dx = pos.positions[ti][i][0] - pos.positions[t0][i][0];
                        let dy = pos.positions[ti][i][1] - pos.positions[t0][i][1];
                        let phase = qv[0] * dx + qv[1] * dy;
                        cos_sum += phase.cos();
                    }
                    fs_val += cos_sum / n_cells as f64;
                }
                fs_sum[lag] += fs_val / q_vectors.len() as f64;
                fs_count[lag] += 1;
            }
        }
        let mut lag_times = Vec::with_capacity(max_lag);
        let mut fs = Vec::with_capacity(max_lag);
        for lag in 0..max_lag {
            if fs_count[lag] > 0 {
                lag_times.push(lag as f64 * dt);
                fs.push(fs_sum[lag] / fs_count[lag] as f64);
            }
        }
        let (tau_alpha, beta, fit_r2) = fit_stretched_exponential(&lag_times, &fs);
        Ok(SelfScatteringOutput { q_star, lag_times, fs, tau_alpha, beta, fit_r2 })
    }
}
