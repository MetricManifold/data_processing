//! Velocity distribution observable: histograms + raw arrays of
//! (vx, vy) per cell, plus second-moment σ and kurtosis.
//!
//! Wraps the existing
//! [`crate::analysis::observables::velocity_distribution`] but exposes
//! it through the v2 trait. Phase 9 (cutover) will inline the body.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observables as legacy;
use crate::analysis::v2::observable::{Context, Observable, Requirements};

/// Histogram bin count for the binned velocity distribution.
pub struct VelocityDistribution {
    pub n_bins: usize,
}

impl Default for VelocityDistribution {
    fn default() -> Self {
        Self { n_bins: 80 }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VelocityDistributionOutput {
    pub bin_edges: Vec<f64>,
    pub cell0_hist: Vec<f64>,
    pub pop_hist: Vec<f64>,
    pub cell0_sigma_vx: f64,
    pub pop_sigma_vx: f64,
    pub cell0_kurtosis: f64,
    pub pop_kurtosis: f64,
    pub cell0_mean_speed: f64,
    pub pop_mean_speed: f64,
    /// Raw vx samples for cell 0 (used by G(v_i) panel).
    pub cell0_vx: Vec<f64>,
    pub cell0_vy: Vec<f64>,
    /// Raw vx samples for the population.
    pub pop_vx: Vec<f64>,
    pub pop_vy: Vec<f64>,
}

impl Observable for VelocityDistribution {
    type Output = VelocityDistributionOutput;

    fn id(&self) -> &'static str {
        "velocity_distribution"
    }

    fn requires(&self) -> Requirements {
        Requirements::POSITIONS
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let r = legacy::velocity_distribution(&ctx.positions, self.n_bins);
        Ok(VelocityDistributionOutput {
            bin_edges: r.bin_edges,
            cell0_hist: r.cell0_hist,
            pop_hist: r.pop_hist,
            cell0_sigma_vx: r.cell0_sigma_vx,
            pop_sigma_vx: r.pop_sigma_vx,
            cell0_kurtosis: r.cell0_kurtosis,
            pop_kurtosis: r.pop_kurtosis,
            cell0_mean_speed: r.cell0_mean_speed,
            pop_mean_speed: r.pop_mean_speed,
            cell0_vx: r.cell0_vx,
            cell0_vy: r.cell0_vy,
            pop_vx: r.pop_vx,
            pop_vy: r.pop_vy,
        })
    }
}
