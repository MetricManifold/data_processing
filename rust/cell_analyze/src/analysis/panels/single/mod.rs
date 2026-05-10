//! Single-run panels: render observables for one selected
//! [`RunAnalysis`].
//!
//! Each panel consumes a [`SingleRunData`] borrowed reference + the
//! standard [`PanelOpts`]. New single-run panels go in their own files
//! here and are wired into the `studies.rs` panel dispatcher.

use plotters::style::RGBAColor;

use crate::analysis::analyze_run::RunAnalysis;

pub mod gvi;
pub mod ln_timeseries;
pub mod msd;
pub mod speed_bursts;

/// Borrowed reference to one run for a single-run panel.
pub struct SingleRunData<'a> {
    pub run: &'a RunAnalysis,
    pub label: &'a str,
}

/// Default colour for the single series.
pub const SINGLE_COLOR: RGBAColor = RGBAColor(52, 130, 200, 1.0);
