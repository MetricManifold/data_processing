//! Overlay panels: N runs rendered as colored series on shared axes.
//! Colour cycles through [`crate::analysis::v2::panels::layout::PALETTE`].

use crate::analysis::v2::analyze_run::RunAnalysis;

pub mod gvi;
pub mod ln_timeseries;
pub mod msd;

/// Borrowed reference to N runs.
pub struct OverlayData<'a> {
    pub series: Vec<OverlaySeries<'a>>,
}

pub struct OverlaySeries<'a> {
    pub run: &'a RunAnalysis,
    pub label: &'a str,
}
