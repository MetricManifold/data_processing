//! Mean-squared-displacement observable.
//!
//! Multi-origin averaging over the first half of the trajectory plus a
//! cell-0-specific MSD for the Palmieri convention. The implementation
//! is owned here (this is the canonical home).

use anyhow::Result;
use serde::Serialize;

use crate::analysis::io::UnwrappedPositions;
use crate::analysis::observable::{Context, Observable, Requirements};

/// Ensemble-averaged mean-squared displacement.
pub struct Msd;

#[derive(Clone, Debug, Serialize)]
pub struct MsdOutput {
    pub lag_times: Vec<f64>,
    pub values: Vec<f64>,
    /// MSD of cell 0 only (the conventional Palmieri "soft cell").
    pub cell0_values: Vec<f64>,
}

/// Stand-alone MSD computation. Public so other modules (and the legacy
/// shim) can call it; the trait `Observable for Msd` just forwards.
pub fn compute_msd(pos: &UnwrappedPositions) -> MsdOutput {
    let n_times = pos.n_times;
    let n_cells = pos.n_cells;
    if n_times < 2 {
        return MsdOutput {
            lag_times: vec![],
            values: vec![],
            cell0_values: vec![],
        };
    }
    let max_lag = n_times / 2;
    let dt = if n_times > 1 { pos.times[1] - pos.times[0] } else { 1.0 };
    let cell0_idx = pos.cell_ids.iter().position(|&id| id == 0).unwrap_or(0);

    let n_origins = max_lag;
    let mut msd_sum = vec![0.0f64; max_lag];
    let mut msd_count = vec![0u64; max_lag];
    let mut cell0_msd_sum = vec![0.0f64; max_lag];

    for t0 in 0..n_origins {
        for lag in 1..max_lag {
            let ti = t0 + lag;
            if ti >= n_times {
                break;
            }
            let mut sum_dsq = 0.0;
            for i in 0..n_cells {
                let dx = pos.positions[ti][i][0] - pos.positions[t0][i][0];
                let dy = pos.positions[ti][i][1] - pos.positions[t0][i][1];
                let dz = pos.positions[ti][i][2] - pos.positions[t0][i][2];
                let dsq = dx * dx + dy * dy + dz * dz;
                sum_dsq += dsq;
                if i == cell0_idx {
                    cell0_msd_sum[lag] += dsq;
                }
            }
            msd_sum[lag] += sum_dsq / n_cells as f64;
            msd_count[lag] += 1;
        }
    }

    let mut lag_times = Vec::with_capacity(max_lag.saturating_sub(1));
    let mut values = Vec::with_capacity(max_lag.saturating_sub(1));
    let mut cell0_values = Vec::with_capacity(max_lag.saturating_sub(1));
    for lag in 1..max_lag {
        if msd_count[lag] > 0 {
            lag_times.push(lag as f64 * dt);
            values.push(msd_sum[lag] / msd_count[lag] as f64);
            cell0_values.push(cell0_msd_sum[lag] / msd_count[lag] as f64);
        }
    }
    MsdOutput { lag_times, values, cell0_values }
}

impl Observable for Msd {
    type Output = MsdOutput;

    fn id(&self) -> &'static str {
        "msd"
    }

    fn requires(&self) -> Requirements {
        Requirements::POSITIONS
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        Ok(compute_msd(&ctx.positions))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::observable::{ObservableBag, RunParams};
    use std::sync::Arc;

    fn synthetic_pos(n_times: usize, dt: f64, v: f64) -> UnwrappedPositions {
        let times: Vec<f64> = (0..n_times).map(|i| i as f64 * dt).collect();
        let cell_ids = vec![0u32, 1];
        let positions: Vec<Vec<[f64; 3]>> = (0..n_times)
            .map(|t| {
                vec![
                    [v * times[t], 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            })
            .collect();
        UnwrappedPositions {
            times,
            cell_ids,
            positions,
            lx: 1000.0,
            ly: 1000.0,
            lz: 0.0,
            dim: 2,
            n_cells: 2,
            n_times,
            inherent_v_a: vec![],
        }
    }

    #[test]
    fn msd_drift_matches_expected() {
        let pos = synthetic_pos(20, 1.0, 0.1);
        let ctx = Context {
            positions: Arc::new(pos),
            trajectory: None,
            checkpoint: None,
            params: RunParams::default(),
        };
        let out = Msd.compute(&ctx).expect("compute msd");
        assert!(!out.lag_times.is_empty());
        assert!((out.cell0_values[0] - 0.01).abs() < 1e-9);
        assert!((out.values[0] - 0.005).abs() < 1e-9);
    }

    #[test]
    fn msd_round_trips_through_bag() {
        let pos = synthetic_pos(10, 1.0, 0.1);
        let ctx = Context {
            positions: Arc::new(pos),
            trajectory: None,
            checkpoint: None,
            params: RunParams::default(),
        };
        let out = Msd.compute(&ctx).expect("compute");
        let mut bag = ObservableBag::new();
        bag.insert::<Msd>(out);
        assert!(bag.get::<Msd>().is_some());
    }
}
