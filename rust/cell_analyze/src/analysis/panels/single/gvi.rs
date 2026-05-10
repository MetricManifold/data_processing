//! Single-run G(v_i) panel: one velocity series with a Gaussian
//! reference and an optional Palmieri Eq. 5 fit.

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;

use crate::analysis::observables::velocity_distribution::VelocityDistribution;
use crate::analysis::panels::gvi_core::{
    draw_gvi_panel, GviMarker, GviPanelOpts, GviSeries,
};
use crate::analysis::panels::{Panel, PanelOpts};

use super::{SingleRunData, SINGLE_COLOR};

pub struct GviSingle {
    pub x_max: f64,
    pub fit_eq5: bool,
}

impl Default for GviSingle {
    fn default() -> Self {
        Self {
            x_max: 0.022,
            fit_eq5: true,
        }
    }
}

impl<'a, 'b> Panel<'a, 'b> for GviSingle {
    type Data = SingleRunData<'a>;

    fn id(&self) -> &'static str {
        "gvi_single"
    }

    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()> {
        let v = data
            .run
            .bag
            .get::<VelocityDistribution>()
            .ok_or_else(|| anyhow!("gvi_single: missing velocity_distribution"))?;
        let series = vec![GviSeries {
            label: data.label.to_string(),
            vx: &v.cell0_vx,
            vy: &v.cell0_vy,
            color: SINGLE_COLOR,
            marker: GviMarker::Triangle,
        }];
        let title = opts.title.clone().unwrap_or_else(|| "G(v_i)".into());
        let gvi_opts = GviPanelOpts {
            title,
            panel_label: None,
            x_max: opts.x_range.map(|r| r.1).unwrap_or(self.x_max),
            y_range: opts.y_range,
            gaussian_ref_sigma: None, // use series 0's σ
            palmieri_fit_index: if self.fit_eq5 { Some(0) } else { None },
            v_a: data.run.params.v_a,
            palmieri_fit_min_v: None,
            gaussian_sigma_sweep: Vec::new(),
        };
        draw_gvi_panel(area, &series, &gvi_opts)
    }
}
