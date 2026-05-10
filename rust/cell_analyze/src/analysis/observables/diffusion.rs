//! Effective diffusion coefficient + R² fit from MSD long-time slope.
//!
//! D_eff = slope / 4 in 2D, fitted to the last `fit_frac` of the MSD curve.
//! Distinct from `msd_palmieri` which extracts D_eff at a single fixed lag
//! (8τ); this one fits a line over the long-time portion.
//!
//! Ported from legacy `observables.rs::compute_diffusion` (commit 45ce569^).

use anyhow::Result;
use serde::Serialize;

use super::msd::Msd;
use crate::analysis::observable::{Context, Observable, Requirements};

/// Default fraction of the MSD curve (from the right) used in the fit.
pub const DEFAULT_FIT_FRAC: f64 = 0.5;

pub struct Diffusion {
    pub fit_frac: f64,
}

impl Default for Diffusion {
    fn default() -> Self {
        Self { fit_frac: DEFAULT_FIT_FRAC }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DiffusionOutput {
    pub d_eff: f64,
    pub fit_r2: f64,
}

impl Observable for Diffusion {
    type Output = DiffusionOutput;

    fn id(&self) -> &'static str { "diffusion" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        // Re-compute MSD locally so this observable is self-contained
        // (Msd may not have been requested in `compute = [...]`).
        let msd = Msd.compute(ctx)?;
        let n = msd.lag_times.len();
        if n < 5 {
            return Ok(DiffusionOutput { d_eff: 0.0, fit_r2: 0.0 });
        }
        let start = ((n as f64 * (1.0 - self.fit_frac)).ceil() as usize).max(1);
        let t = &msd.lag_times[start..];
        let y = &msd.values[start..];
        let nf = t.len();
        if nf < 2 {
            return Ok(DiffusionOutput { d_eff: 0.0, fit_r2: 0.0 });
        }
        let nf_f = nf as f64;
        let sum_t: f64 = t.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_tt: f64 = t.iter().map(|x| x * x).sum();
        let sum_ty: f64 = t.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let denom = nf_f * sum_tt - sum_t * sum_t;
        if denom.abs() < 1e-30 {
            return Ok(DiffusionOutput { d_eff: 0.0, fit_r2: 0.0 });
        }
        let slope = (nf_f * sum_ty - sum_t * sum_y) / denom;
        let intercept = (sum_y - slope * sum_t) / nf_f;
        let d_eff = (slope / 4.0).max(0.0);
        let y_mean = sum_y / nf_f;
        let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
        let ss_res: f64 = t.iter().zip(y.iter())
            .map(|(a, b)| (b - (slope * a + intercept)).powi(2)).sum();
        let fit_r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
        Ok(DiffusionOutput { d_eff, fit_r2 })
    }
}
