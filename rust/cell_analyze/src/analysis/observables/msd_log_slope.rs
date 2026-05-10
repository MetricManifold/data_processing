//! MSD log-slope Δ(t) = d log MSD / d log t.
//!
//! Δ ≈ 2 → ballistic, Δ ≈ 1 → diffusive, Δ < 1 → subdiffusive/caged.
//! Ported from legacy `observables.rs::msd_log_slope`.

use anyhow::Result;
use serde::Serialize;

use super::msd::Msd;
use crate::analysis::observable::{Context, Observable, Requirements};

pub struct MsdLogSlope;

#[derive(Clone, Debug, Serialize)]
pub struct MsdLogSlopeOutput {
    pub times: Vec<f64>,
    pub delta: Vec<f64>,
}

impl Observable for MsdLogSlope {
    type Output = MsdLogSlopeOutput;

    fn id(&self) -> &'static str { "msd_log_slope" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let msd = Msd.compute(ctx)?;
        let valid: Vec<(f64, f64)> = msd.lag_times.iter().zip(msd.values.iter())
            .filter(|(&t, &v)| t > 0.0 && v > 0.0)
            .map(|(&t, &v)| (t.ln(), v.ln()))
            .collect();
        let mut times = Vec::new();
        let mut delta = Vec::new();
        for i in 1..valid.len() {
            let dt = valid[i].0 - valid[i - 1].0;
            if dt.abs() < 1e-30 { continue; }
            let d = (valid[i].1 - valid[i - 1].1) / dt;
            let t_mid = ((valid[i].0 + valid[i - 1].0) / 2.0).exp();
            times.push(t_mid);
            delta.push(d);
        }
        Ok(MsdLogSlopeOutput { times, delta })
    }
}
