//! Sweep panels: `axis variable` (x) versus `metric` (y) with error
//! bars. Reusable across FSS (x=N), pairwise (x=d), percolation (x=fc).

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use super::super::aggregate::SweepCurve;
use super::layout::{padded, PALETTE};
use super::{Panel, PanelOpts};

/// Plot one metric as a function of the sweep axis with stderr bars.
pub struct MetricVsX<'a> {
    /// Name of the metric (must exist in every `point.metrics`).
    pub metric: &'a str,
    /// Optional reference horizontal line (e.g. ratio = 1).
    pub h_line: Option<f64>,
}

impl<'a, 'b, 's> Panel<'a, 'b> for MetricVsX<'s> {
    type Data = SweepCurve;

    fn id(&self) -> &'static str {
        "metric_vs_x"
    }

    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()> {
        if data.points.is_empty() {
            return Err(anyhow!("metric_vs_x: empty sweep curve"));
        }

        // Collect (x, mean, stderr) tuples.
        let mut xs: Vec<f64> = Vec::with_capacity(data.points.len());
        let mut ys: Vec<f64> = Vec::with_capacity(data.points.len());
        let mut errs: Vec<f64> = Vec::with_capacity(data.points.len());
        for p in &data.points {
            let m = p
                .metrics
                .get(self.metric)
                .ok_or_else(|| anyhow!("metric `{}` missing at x={}", self.metric, p.x))?;
            xs.push(p.x);
            ys.push(m.mean);
            errs.push(m.stderr);
        }

        // Axis ranges.
        let (x_min, x_max) = opts.x_range.unwrap_or_else(|| {
            let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            padded(lo, hi, 0.10)
        });
        let (y_min, y_max) = opts.y_range.unwrap_or_else(|| {
            let lo = ys
                .iter()
                .zip(&errs)
                .map(|(y, e)| y - e)
                .chain(self.h_line.into_iter())
                .fold(f64::INFINITY, f64::min);
            let hi = ys
                .iter()
                .zip(&errs)
                .map(|(y, e)| y + e)
                .chain(self.h_line.into_iter())
                .fold(f64::NEG_INFINITY, f64::max);
            padded(lo, hi, 0.10)
        });

        // Build chart.
        let title = opts.title.clone().unwrap_or_else(|| self.metric.to_string());
        let x_label = opts.x_label.clone().unwrap_or_else(|| data.axis.clone());
        let y_label = opts.y_label.clone().unwrap_or_else(|| self.metric.to_string());

        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 16))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(60)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .x_label_style(("sans-serif", 12))
            .y_label_style(("sans-serif", 12))
            .light_line_style(TRANSPARENT)
            .bold_line_style(RGBAColor(200, 200, 200, 0.3))
            .draw()?;

        // Optional horizontal reference.
        if let Some(h) = self.h_line {
            chart.draw_series(LineSeries::new(
                vec![(x_min, h), (x_max, h)],
                BLACK.mix(0.4).stroke_width(1),
            ))?;
        }

        // Error bars.
        let bar_color = PALETTE[0];
        for ((x, y), e) in xs.iter().zip(&ys).zip(&errs) {
            chart.draw_series(LineSeries::new(
                vec![(*x, y - e), (*x, y + e)],
                bar_color.stroke_width(1),
            ))?;
        }

        // Connecting line.
        chart.draw_series(LineSeries::new(
            xs.iter().zip(&ys).map(|(x, y)| (*x, *y)),
            bar_color.stroke_width(2),
        ))?;

        // Markers.
        chart.draw_series(
            xs.iter()
                .zip(&ys)
                .map(|(x, y)| Circle::new((*x, *y), 4, bar_color.filled())),
        )?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests: render to an in-memory SVG buffer and check that something is
// drawn (we don't pixel-diff; just smoke).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::v2::aggregate::{MetricValue, SweepCurve, SweepPoint};
    use std::collections::BTreeMap;

    fn fake_curve() -> SweepCurve {
        let mut points = vec![];
        for x in [2.0_f64, 4.0, 8.0, 20.0] {
            let mut metrics = BTreeMap::new();
            metrics.insert(
                "d_eff".into(),
                MetricValue {
                    mean: 0.05 * (1.0 - (-x / 6.0).exp()),
                    stderr: 0.005,
                    values: vec![],
                },
            );
            points.push(SweepPoint {
                x,
                variables: BTreeMap::new(),
                n: 3,
                metrics,
            });
        }
        SweepCurve {
            axis: "d".into(),
            points,
        }
    }

    #[test]
    fn metric_vs_x_renders_to_svg_buffer() {
        let mut buf = String::new();
        {
            let backend = SVGBackend::with_string(&mut buf, (400, 300));
            let area = backend.into_drawing_area();
            area.fill(&WHITE).unwrap();
            let panel = MetricVsX {
                metric: "d_eff",
                h_line: None,
            };
            let opts = PanelOpts {
                title: Some("D_eff vs d".into()),
                ..Default::default()
            };
            panel.render(&area, &fake_curve(), &opts).expect("render");
        }
        // SVG should contain at least one circle (marker) and the title.
        assert!(buf.contains("<circle"), "expected markers in SVG");
        assert!(buf.contains("D_eff vs d"), "expected title in SVG");
    }
}
