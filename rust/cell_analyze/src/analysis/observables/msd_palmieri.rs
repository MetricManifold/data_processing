//! Palmieri-style MSD/Δt curve over the 0..8τ window plus the
//! D_eff = MSD(8τ)/(4·8τ) read-off.
//!
//! Differs from the general [`crate::analysis::observables::msd::Msd`]
//! observable in two ways:
//!   1. lags are sampled densely in the visible 0..8τ window (200
//!      points), so the MSD/Δt curve is smooth on the panel,
//!   2. the 8τ lag is always included so D_eff is read from the same
//!      time point used by Palmieri (2015).

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct MsdPalmieri;

#[derive(Clone, Debug, Serialize)]
pub struct MsdPalmieriOutput {
    /// Lag time in τ units for each sampled point.
    pub lag_tau: Vec<f64>,
    /// MSD(Δt)/Δt for the tagged cell.
    pub msd_t_cell: Vec<f64>,
    /// MSD(Δt)/Δt for the population.
    pub msd_t_pop: Vec<f64>,
    /// D_eff for the tagged cell evaluated at the 8τ lag.
    pub d_eff_cell: f64,
    /// D_eff for the population evaluated at the 8τ lag.
    pub d_eff_pop: f64,
}

impl Observable for MsdPalmieri {
    type Output = MsdPalmieriOutput;

    fn id(&self) -> &'static str {
        "msd_palmieri"
    }

    fn requires(&self) -> Requirements {
        Requirements::POSITIONS
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let tau = ctx.params.tau;
        let n = pos.n_times;
        if n < 4 {
            return Ok(MsdPalmieriOutput {
                lag_tau: vec![],
                msd_t_cell: vec![],
                msd_t_pop: vec![],
                d_eff_cell: f64::NAN,
                d_eff_pop: f64::NAN,
            });
        }
        let dt = pos.times[1] - pos.times[0];
        let tagged: u32 = ctx.params.tagged_cells.first().copied().unwrap_or(0);
        let cell_idx = pos
            .cell_ids
            .iter()
            .position(|&c| c == tagged)
            .unwrap_or(0);

        let max_lag = n / 2;
        let lag_8tau = ((8.0 * tau / dt).round() as usize).min(max_lag);
        if lag_8tau < 2 {
            return Ok(MsdPalmieriOutput {
                lag_tau: vec![],
                msd_t_cell: vec![],
                msd_t_pop: vec![],
                d_eff_cell: f64::NAN,
                d_eff_pop: f64::NAN,
            });
        }

        // 200-point dense sampling of 0..8τ, plus the first 10 lags
        // (so the curve has good resolution near the origin) and 8τ
        // itself.
        let n_visible = 200usize;
        let stride = (lag_8tau / n_visible).max(1);
        let mut lags: std::collections::BTreeSet<usize> = (1..=10.min(lag_8tau)).collect();
        let mut l = stride;
        while l <= lag_8tau {
            lags.insert(l);
            l += stride;
        }
        lags.insert(lag_8tau);

        let mut lag_tau = Vec::with_capacity(lags.len());
        let mut msd_t_cell = Vec::with_capacity(lags.len());
        let mut msd_t_pop = Vec::with_capacity(lags.len());
        let mut d_eff_cell = f64::NAN;
        let mut d_eff_pop = f64::NAN;
        let n_cells = pos.n_cells;
        for &lag in &lags {
            let lag_time = lag as f64 * dt;
            let n_origins = n - lag;
            if n_origins < 2 {
                break;
            }
            let mut cell_sum = 0.0f64;
            let mut pop_sum = 0.0f64;
            for t0 in 0..n_origins {
                let ti = t0 + lag;
                let dx = pos.positions[ti][cell_idx][0] - pos.positions[t0][cell_idx][0];
                let dy = pos.positions[ti][cell_idx][1] - pos.positions[t0][cell_idx][1];
                cell_sum += dx * dx + dy * dy;
                let mut s = 0.0;
                for ci in 0..n_cells {
                    let ddx = pos.positions[ti][ci][0] - pos.positions[t0][ci][0];
                    let ddy = pos.positions[ti][ci][1] - pos.positions[t0][ci][1];
                    s += ddx * ddx + ddy * ddy;
                }
                pop_sum += s / n_cells as f64;
            }
            let msd_cell = cell_sum / n_origins as f64;
            let msd_p = pop_sum / n_origins as f64;
            lag_tau.push(lag_time / tau);
            msd_t_cell.push(msd_cell / lag_time);
            msd_t_pop.push(msd_p / lag_time);
            if lag == lag_8tau {
                d_eff_cell = msd_cell / (4.0 * lag_time);
                d_eff_pop = msd_p / (4.0 * lag_time);
            }
        }

        Ok(MsdPalmieriOutput {
            lag_tau,
            msd_t_cell,
            msd_t_pop,
            d_eff_cell,
            d_eff_pop,
        })
    }
}
