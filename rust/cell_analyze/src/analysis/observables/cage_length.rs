//! Cage length L_c from the MSD plateau (minimum of Δ(t) near τ).
//!
//! Ported from legacy `observables.rs::cage_length`.

use anyhow::Result;
use serde::Serialize;

use super::msd::Msd;
use super::msd_log_slope::MsdLogSlope;
use crate::analysis::observable::{Context, Observable, Requirements};

pub struct CageLength;

#[derive(Clone, Debug, Serialize)]
pub struct CageLengthOutput {
    pub l_c: f64,
    pub t_star: f64,
}

fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if xs.is_empty() || ys.is_empty() { return 0.0; }
    if x <= xs[0] { return ys[0]; }
    if x >= xs[xs.len() - 1] { return ys[ys.len() - 1]; }
    for i in 1..xs.len() {
        if xs[i] >= x {
            let t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
            return ys[i - 1] + t * (ys[i] - ys[i - 1]);
        }
    }
    ys[ys.len() - 1]
}

impl Observable for CageLength {
    type Output = CageLengthOutput;

    fn id(&self) -> &'static str { "cage_length" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let msd = Msd.compute(ctx)?;
        let ls = MsdLogSlope.compute(ctx)?;
        let tau = ctx.params.tau;
        if ls.delta.is_empty() {
            return Ok(CageLengthOutput { l_c: f64::NAN, t_star: f64::NAN });
        }
        // Minimum Δ in window [0.1τ, 5τ]
        let mut best_idx = 0usize;
        let mut best_val = f64::INFINITY;
        for (i, (&t, &d)) in ls.times.iter().zip(ls.delta.iter()).enumerate() {
            if t > 0.1 * tau && t < 5.0 * tau && d < best_val {
                best_val = d;
                best_idx = i;
            }
        }
        if best_val == f64::INFINITY {
            for (i, &d) in ls.delta.iter().enumerate() {
                if d < best_val { best_val = d; best_idx = i; }
            }
        }
        let t_star = ls.times[best_idx];
        let l_c_sq = interp(t_star, &msd.lag_times, &msd.values);
        Ok(CageLengthOutput { l_c: l_c_sq.sqrt(), t_star })
    }
}
