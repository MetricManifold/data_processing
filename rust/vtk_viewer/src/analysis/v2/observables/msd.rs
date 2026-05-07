//! Mean-squared-displacement observable.
//!
//! v2 wrapper around the existing [`crate::analysis::observables::compute_msd`].
//! The implementation is *not* duplicated here — phase 9 (cutover) will
//! move the body of the compute function into this file and delete the
//! legacy entry point. Until then both paths share one implementation.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observables as legacy;
use crate::analysis::v2::observable::{Context, Observable, Requirements};

/// Ensemble-averaged mean-squared displacement.
///
/// Multi-origin averaging over the first half of the trajectory, plus a
/// cell-0-specific MSD for the Palmieri soft-cell convention. See
/// [`MsdOutput`] for what's returned.
pub struct Msd;

#[derive(Clone, Debug, Serialize)]
pub struct MsdOutput {
    pub lag_times: Vec<f64>,
    pub values: Vec<f64>,
    /// MSD of cell 0 only (the conventional Palmieri "soft cell").
    pub cell0_values: Vec<f64>,
}

impl From<legacy::MsdResult> for MsdOutput {
    fn from(r: legacy::MsdResult) -> Self {
        Self {
            lag_times: r.lag_times,
            values: r.values,
            cell0_values: r.cell0_values,
        }
    }
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
        Ok(legacy::compute_msd(&ctx.positions).into())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::io::UnwrappedPositions;
    use crate::analysis::v2::observable::{ObservableBag, RunParams};
    use std::sync::Arc;

    /// Build a tiny synthetic trajectory: one cell drifting with constant
    /// velocity, plus a second cell stationary. MSD(Δt) of cell 0 should
    /// equal `(v Δt)²`; ensemble MSD = half of that.
    fn synthetic_pos(n_times: usize, dt: f64, v: f64) -> UnwrappedPositions {
        let times: Vec<f64> = (0..n_times).map(|i| i as f64 * dt).collect();
        let cell_ids = vec![0u32, 1];
        let positions: Vec<Vec<[f64; 3]>> = (0..n_times)
            .map(|t| {
                vec![
                    [v * times[t], 0.0, 0.0], // cell 0 drifts in x
                    [0.0, 0.0, 0.0],          // cell 1 stays
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
        // First lag = dt = 1.0; cell-0 MSD = (v dt)^2 = 0.01
        assert!(out.lag_times.len() >= 1);
        assert!((out.cell0_values[0] - 0.01).abs() < 1e-9,
                "cell0 msd at dt=1: got {}", out.cell0_values[0]);
        // Ensemble MSD averages cell 0 (drift) and cell 1 (static):
        //   ((v dt)^2 + 0) / 2 = 0.005
        assert!((out.values[0] - 0.005).abs() < 1e-9,
                "ensemble msd at dt=1: got {}", out.values[0]);
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
        let back = bag.get::<Msd>().expect("msd missing from bag");
        assert!(!back.lag_times.is_empty());
    }
}
