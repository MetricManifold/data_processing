//! Overlay MSD/Δt: N runs colored by series on shared axes.

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::observables::msd_palmieri::MsdPalmieri;
use crate::analysis::panels::layout::PALETTE;
use crate::analysis::panels::{Panel, PanelOpts};

use super::OverlayData;

pub struct MsdOverlay {
    pub msd_lag_max: f64,
}

impl Default for MsdOverlay {
    fn default() -> Self {
        Self { msd_lag_max: 8.0 }
    }
}

impl<'a, 'b> Panel<'a, 'b> for MsdOverlay {
    type Data = OverlayData<'a>;

    fn id(&self) -> &'static str {
        "msd_overlay"
    }

    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()> {
        if data.series.is_empty() {
            return Err(anyhow!("msd_overlay: empty series list"));
        }
        let x_max = opts.x_range.map(|r| r.1).unwrap_or(self.msd_lag_max);
        let mut all_series: Vec<(String, Vec<(f64, f64)>, f64)> = Vec::new();
        for s in &data.series {
            let m = s
                .run
                .bag
                .get::<MsdPalmieri>()
                .ok_or_else(|| anyhow!("msd_overlay: series `{}` missing msd_palmieri", s.label))?;
            let pts: Vec<(f64, f64)> = m
                .lag_tau
                .iter()
                .zip(m.msd_t_cell.iter())
                .filter(|(x, y)| **x <= x_max && **x > 0.0 && **y > 0.0)
                .map(|(&x, &y)| (x, y))
                .collect();
            all_series.push((s.label.to_string(), pts, m.d_eff_cell));
        }

        let all_y: Vec<f64> = all_series
            .iter()
            .flat_map(|(_, pts, _)| pts.iter().map(|p| p.1))
            .collect();
        let y_lo = all_y.iter().copied().fold(f64::INFINITY, f64::min);
        let y_hi = all_y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let y_pad = (y_hi - y_lo).max(1e-6) * 0.1;
        let y_max = opts.y_range.map(|r| r.1).unwrap_or(y_hi + y_pad);
        let y_min = opts
            .y_range
            .map(|r| r.0)
            .unwrap_or((y_lo - y_pad).max(0.0));

        let title = opts.title.clone().unwrap_or_else(|| "MSD/Δt".into());
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 16))
            .margin(8)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..x_max, y_min..y_max)?;
        chart
            .configure_mesh()
            .x_desc("Δt (τ)")
            .y_desc("MSD/Δt")
            .x_label_style(("sans-serif", 14))
            .y_label_style(("sans-serif", 14))
            .light_line_style(TRANSPARENT)
            .bold_line_style(RGBAColor(200, 200, 200, 0.3))
            .draw()?;

        for (i, (label, pts, d_eff)) in all_series.into_iter().enumerate() {
            let color = PALETTE[i % PALETTE.len()];
            chart
                .draw_series(LineSeries::new(pts, color.stroke_width(2)))?
                .label(format!("{} (D={:.4})", label, d_eff))
                .legend(move |(x, y)| {
                    Rectangle::new([(x, y - 2), (x + 12, y + 2)], color.filled())
                });
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::LowerRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 9))
            .draw()?;
        chart.plotting_area().draw(&PathElement::new(
            vec![(0.0, y_max), (x_max, y_max), (x_max, y_min)],
            BLACK.mix(0.5).stroke_width(1),
        ))?;
        Ok(())
    }
}
