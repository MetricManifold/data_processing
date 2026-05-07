//! Overlay G(v_i): N velocity series on the same Palmieri panel.

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;

use crate::analysis::v2::observables::velocity_distribution::VelocityDistribution;
use crate::analysis::v2::panels::gvi_core::{
    draw_gvi_panel, GviMarker, GviPanelOpts, GviSeries,
};
use crate::analysis::v2::panels::layout::PALETTE;
use crate::analysis::v2::panels::{Panel, PanelOpts};

use super::OverlayData;

pub struct GviOverlay {
    pub x_max: f64,
}

impl Default for GviOverlay {
    fn default() -> Self {
        Self { x_max: 0.022 }
    }
}

const MARKERS: &[GviMarker] = &[
    GviMarker::Triangle,
    GviMarker::Circle,
    GviMarker::Square,
    GviMarker::Diamond,
];

impl<'a, 'b> Panel<'a, 'b> for GviOverlay {
    type Data = OverlayData<'a>;

    fn id(&self) -> &'static str {
        "gvi_overlay"
    }

    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()> {
        if data.series.is_empty() {
            return Err(anyhow!("gvi_overlay: empty series list"));
        }
        // Collect raw vx/vy + label refs first so we can build the
        // GviSeries borrows at the right scope.
        let mut raw: Vec<(String, &[f64], &[f64])> = Vec::with_capacity(data.series.len());
        for s in &data.series {
            let v = s
                .run
                .bag
                .get::<VelocityDistribution>()
                .ok_or_else(|| {
                    anyhow!("gvi_overlay: series `{}` missing velocity_distribution", s.label)
                })?;
            raw.push((s.label.to_string(), v.cell0_vx.as_slice(), v.cell0_vy.as_slice()));
        }
        let series: Vec<GviSeries> = raw
            .iter()
            .enumerate()
            .map(|(i, (label, vx, vy))| GviSeries {
                label: label.clone(),
                vx,
                vy,
                color: PALETTE[i % PALETTE.len()],
                marker: MARKERS[i % MARKERS.len()],
            })
            .collect();

        let title = opts.title.clone().unwrap_or_else(|| "G(v_i)".into());
        let gvi_opts = GviPanelOpts {
            title,
            panel_label: None,
            x_max: opts.x_range.map(|r| r.1).unwrap_or(self.x_max),
            y_range: opts.y_range,
            gaussian_ref_sigma: None,
            // No Eq.5 fit by default for arbitrary-N overlays; per-series
            // ζ doesn't generalize cleanly. Single/pair panels still fit.
            palmieri_fit_index: None,
            v_a: data.series[0].run.params.v_a,
            palmieri_fit_min_v: None,
            gaussian_sigma_sweep: Vec::new(),
        };
        draw_gvi_panel(area, &series, &gvi_opts)
    }
}
