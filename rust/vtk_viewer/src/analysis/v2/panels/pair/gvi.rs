//! G(v_i) Palmieri velocity-distribution panel for a pair of runs.
//!
//! Thin orchestration: builds two `GviSeries` from the run bags and
//! delegates to [`crate::analysis::v2::panels::gvi_core::draw_gvi_panel`].

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;

use crate::analysis::v2::observables::velocity_distribution::VelocityDistribution;
use crate::analysis::v2::panels::gvi_core::{
    compute_gvi, draw_gvi_panel, GviMarker, GviPanelOpts, GviSeries,
};
use crate::analysis::v2::panels::{Panel, PanelOpts};

use super::{PairPanelData, CTRL_COLOR, SOFT_COLOR};

pub struct GviPair {
    pub x_max: f64,
    pub v_a: f64,
}

impl Default for GviPair {
    fn default() -> Self {
        Self {
            x_max: 0.022,
            v_a: 0.01,
        }
    }
}

impl<'a, 'b> Panel<'a, 'b> for GviPair {
    type Data = PairPanelData<'a>;

    fn id(&self) -> &'static str {
        "gvi_pair"
    }

    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()> {
        let num = data
            .numerator
            .bag
            .get::<VelocityDistribution>()
            .ok_or_else(|| anyhow!("gvi_pair: numerator missing velocity_distribution"))?;
        let den = data
            .denominator
            .bag
            .get::<VelocityDistribution>()
            .ok_or_else(|| anyhow!("gvi_pair: denominator missing velocity_distribution"))?;

        // Gaussian reference σ from the denominator (control) so soft
        // tails appear as deviations from a matched-second-moment
        // baseline.
        let (_, _, ctrl_sigma) = compute_gvi(&den.cell0_vx, &den.cell0_vy);

        let series = vec![
            GviSeries {
                label: data.numerator_label.to_string(),
                vx: &num.cell0_vx,
                vy: &num.cell0_vy,
                color: SOFT_COLOR,
                marker: GviMarker::Triangle,
            },
            GviSeries {
                label: data.denominator_label.to_string(),
                vx: &den.cell0_vx,
                vy: &den.cell0_vy,
                color: CTRL_COLOR,
                marker: GviMarker::Circle,
            },
        ];

        let title = opts.title.clone().unwrap_or_else(|| "G(v_i)".into());
        let gvi_opts = GviPanelOpts {
            title,
            panel_label: None,
            x_max: opts.x_range.map(|r| r.1).unwrap_or(self.x_max),
            y_range: opts.y_range,
            gaussian_ref_sigma: Some(ctrl_sigma),
            palmieri_fit_index: Some(0),
            v_a: self.v_a,
            palmieri_fit_min_v: None,
            gaussian_sigma_sweep: Vec::new(),
        };
        draw_gvi_panel(area, &series, &gvi_opts)
    }
}
