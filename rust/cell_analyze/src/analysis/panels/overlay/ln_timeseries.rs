//! Overlay L_n(t): N runs colored on shared time axis.

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::observables::ln_perimeter::LnPerimeter;
use crate::analysis::panels::layout::PALETTE;
use crate::analysis::panels::{Panel, PanelOpts};

use super::OverlayData;

pub struct LnTimeseriesOverlay;

impl<'a, 'b> Panel<'a, 'b> for LnTimeseriesOverlay {
    type Data = OverlayData<'a>;

    fn id(&self) -> &'static str {
        "ln_timeseries_overlay"
    }

    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()> {
        if data.series.is_empty() {
            return Err(anyhow!("ln_timeseries_overlay: empty series list"));
        }
        let mut all: Vec<(String, &Vec<f64>, &Vec<f64>, f64)> = Vec::new();
        for s in &data.series {
            let ln = s
                .run
                .bag
                .get::<LnPerimeter>()
                .ok_or_else(|| {
                    anyhow!("ln_timeseries_overlay: series `{}` missing ln_perimeter", s.label)
                })?;
            all.push((s.label.to_string(), &ln.t_tau, &ln.series, ln.mean));
        }
        let valid: Vec<f64> = all
            .iter()
            .flat_map(|(_, _, series, _)| series.iter().copied().filter(|v| v.is_finite()))
            .collect();
        let data_min = valid.iter().copied().fold(f64::INFINITY, f64::min);
        let data_max = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (y_min, y_max) = opts.y_range.unwrap_or_else(|| {
            let lo = ((data_min - 0.03) * 20.0).floor() / 20.0;
            let hi = ((data_max + 0.03) * 20.0).ceil() / 20.0;
            (0.98_f64.min(lo), 1.50_f64.max(hi))
        });
        let x_max = all
            .iter()
            .map(|(_, t, _, _)| t.last().copied().unwrap_or(1.0))
            .fold(0.0_f64, f64::max);

        let title = opts.title.clone().unwrap_or_else(|| "L_n(t)".into());
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 16))
            .margin(8)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..x_max, y_min..y_max)?;
        chart
            .configure_mesh()
            .x_desc("t (τ)")
            .y_desc("L_n")
            .x_label_style(("sans-serif", 14))
            .y_label_style(("sans-serif", 14))
            .light_line_style(TRANSPARENT)
            .bold_line_style(RGBAColor(200, 200, 200, 0.3))
            .draw()?;

        for (i, (label, ts, series, mean)) in all.into_iter().enumerate() {
            let color = PALETTE[i % PALETTE.len()];
            let step = (ts.len() / 1000).max(1);
            chart
                .draw_series(LineSeries::new(
                    ts.iter()
                        .step_by(step)
                        .zip(series.iter().step_by(step))
                        .map(|(&t, &l)| (t, l.min(y_max).max(y_min))),
                    color.mix(0.6).stroke_width(1),
                ))?
                .label(format!("{} (μ={:.3})", label, mean))
                .legend(move |(x, y)| Rectangle::new([(x, y - 2), (x + 12, y + 2)], color.filled()));
        }
        chart.draw_series(LineSeries::new(
            vec![(0.0, 1.0), (x_max, 1.0)],
            BLACK.mix(0.2).stroke_width(1),
        ))?;
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 10))
            .draw()?;
        chart.plotting_area().draw(&PathElement::new(
            vec![(0.0, y_max), (x_max, y_max), (x_max, y_min)],
            BLACK.mix(0.5).stroke_width(1),
        ))?;
        Ok(())
    }
}
