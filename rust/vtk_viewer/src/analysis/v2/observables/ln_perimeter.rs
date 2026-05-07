//! L_n perimeter time series for the tagged cell.
//!
//! Reads the `l_n` column out of trajectory snapshots aligned to
//! `UnwrappedPositions` time indices. Provides the data used by both
//! `ln_timeseries` and `ln_histogram` pair panels.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::v2::observable::{Context, Observable, Requirements};

pub struct LnPerimeter;

#[derive(Clone, Debug, Serialize)]
pub struct LnPerimeterOutput {
    /// Time in units of τ for each frame.
    pub t_tau: Vec<f64>,
    /// L_n(t) for the tagged cell. Frames missing the tagged cell are
    /// emitted as NaN so the array is index-aligned with `t_tau`.
    pub series: Vec<f64>,
    /// Mean L_n over non-NaN frames.
    pub mean: f64,
}

impl Observable for LnPerimeter {
    type Output = LnPerimeterOutput;

    fn id(&self) -> &'static str {
        "ln_perimeter"
    }

    fn requires(&self) -> Requirements {
        Requirements::POSITIONS | Requirements::TRAJECTORY
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let traj = ctx
            .trajectory
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ln_perimeter requires trajectory"))?;
        let tau = ctx.params.tau;
        let dt = if pos.n_times > 1 {
            pos.times[1] - pos.times[0]
        } else {
            1.0
        };
        let tagged: u32 = ctx
            .params
            .tagged_cells
            .first()
            .copied()
            .unwrap_or(0);

        // Index trajectory by integer time key (microseconds-ish) so we
        // can align to `pos.times` even when frames were dropped.
        let traj_map: std::collections::HashMap<i64, &std::collections::HashMap<u32, _>> =
            traj.frames.iter().map(|(t, c)| ((*t * 1e6) as i64, c)).collect();

        let mut series = Vec::with_capacity(pos.n_times);
        for &t in &pos.times {
            let key = (t * 1e6) as i64;
            let v = traj_map
                .get(&key)
                .and_then(|cells| cells.get(&tagged))
                .map(|s| s.l_n)
                .unwrap_or(f64::NAN);
            series.push(v);
        }
        let t_tau: Vec<f64> = (0..series.len()).map(|i| i as f64 * dt / tau).collect();
        let valid: Vec<f64> = series.iter().copied().filter(|v| v.is_finite()).collect();
        let mean = if valid.is_empty() {
            f64::NAN
        } else {
            valid.iter().sum::<f64>() / valid.len() as f64
        };

        Ok(LnPerimeterOutput { t_tau, series, mean })
    }
}
