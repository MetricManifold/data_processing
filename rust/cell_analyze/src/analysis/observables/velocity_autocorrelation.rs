//! Velocity autocorrelation C_v(τ) = ⟨v(t)·v(t+τ)⟩ / ⟨v²⟩.
//!
//! Velocities from central finite differences. Estimates correlation
//! time τ_c via first 1/e crossing.
//!
//! Ported from legacy `observables.rs::velocity_autocorrelation`.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct VelocityAutocorrelation;

#[derive(Clone, Debug, Serialize)]
pub struct VelocityAutocorrelationOutput {
    pub lag_times: Vec<f64>,
    pub cv: Vec<f64>,
    pub beta: f64,
    pub tau_c: f64,
}

impl Observable for VelocityAutocorrelation {
    type Output = VelocityAutocorrelationOutput;

    fn id(&self) -> &'static str { "velocity_autocorrelation" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let n = pos.n_times;
        let nc = pos.n_cells;
        if n < 3 {
            return Ok(VelocityAutocorrelationOutput {
                lag_times: vec![], cv: vec![], beta: 1.0, tau_c: 0.0,
            });
        }
        let dt_save = if n >= 2 { pos.times[1] - pos.times[0] } else { 1.0 };
        let n_vel = n - 2;
        let mut vx = vec![vec![0.0_f64; nc]; n_vel];
        let mut vy = vec![vec![0.0_f64; nc]; n_vel];
        let mut vz = vec![vec![0.0_f64; nc]; n_vel];
        for t in 0..n_vel {
            let dt2 = pos.times[t + 2] - pos.times[t];
            if dt2 <= 0.0 { continue; }
            for i in 0..nc {
                vx[t][i] = (pos.positions[t + 2][i][0] - pos.positions[t][i][0]) / dt2;
                vy[t][i] = (pos.positions[t + 2][i][1] - pos.positions[t][i][1]) / dt2;
                vz[t][i] = (pos.positions[t + 2][i][2] - pos.positions[t][i][2]) / dt2;
            }
        }
        let max_lag = n_vel / 2;
        let mut lag_times = Vec::with_capacity(max_lag);
        let mut cv = Vec::with_capacity(max_lag);
        let mut v_sq_sum = 0.0;
        let mut v_sq_count = 0usize;
        for t in 0..n_vel {
            for i in 0..nc {
                v_sq_sum += vx[t][i] * vx[t][i] + vy[t][i] * vy[t][i] + vz[t][i] * vz[t][i];
                v_sq_count += 1;
            }
        }
        let v_sq_mean = if v_sq_count > 0 { v_sq_sum / v_sq_count as f64 } else { 1.0 };
        for lag in 0..max_lag {
            let mut dot_sum = 0.0;
            let mut count = 0usize;
            for t in 0..(n_vel - lag) {
                for i in 0..nc {
                    dot_sum += vx[t][i] * vx[t + lag][i]
                            + vy[t][i] * vy[t + lag][i]
                            + vz[t][i] * vz[t + lag][i];
                    count += 1;
                }
            }
            let cv_val = if count > 0 && v_sq_mean > 0.0 {
                (dot_sum / count as f64) / v_sq_mean
            } else { 0.0 };
            lag_times.push(lag as f64 * dt_save);
            cv.push(cv_val);
        }
        let e_inv = 1.0 / std::f64::consts::E;
        let mut tau_c = lag_times.last().copied().unwrap_or(0.0);
        for (i, &c) in cv.iter().enumerate() {
            if c < e_inv { tau_c = lag_times[i]; break; }
        }
        Ok(VelocityAutocorrelationOutput {
            lag_times, cv, beta: 1.0, tau_c,
        })
    }
}
