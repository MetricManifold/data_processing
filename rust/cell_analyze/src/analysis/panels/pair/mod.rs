//! Pair panels: render diagnostic comparisons between two
//! [`RunAnalysis`] runs (typically soft vs ctrl). All pair panels share
//! the [`PairPanelData`] input struct, which carries borrowed
//! references to both runs plus their human labels.

use plotters::style::RGBAColor;

use crate::analysis::analyze_run::RunAnalysis;

pub mod deff_bar;
pub mod gvi;
pub mod ln_histogram;
pub mod ln_timeseries;
pub mod msd_t;
pub mod speed_bursts;
pub mod summary;
pub mod trajectory_xy;

/// Shared data shape consumed by every pair panel.
pub struct PairPanelData<'a> {
    pub numerator: &'a RunAnalysis,
    pub denominator: &'a RunAnalysis,
    pub numerator_label: &'a str,
    pub denominator_label: &'a str,
}

/// Default soft (numerator) color: red.
pub const SOFT_COLOR: RGBAColor = RGBAColor(220, 60, 60, 1.0);
/// Default ctrl (denominator) color: blue.
pub const CTRL_COLOR: RGBAColor = RGBAColor(52, 152, 219, 1.0);
/// 50%-alpha variants for histogram fills + light overlays.
pub const SOFT_ALPHA: RGBAColor = RGBAColor(220, 60, 60, 0.5);
pub const CTRL_ALPHA: RGBAColor = RGBAColor(52, 152, 219, 0.5);
pub const POP_DASH: RGBAColor = RGBAColor(120, 120, 120, 0.4);
