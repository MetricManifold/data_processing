//! L_n(t) time series for the tagged cell, both runs overlaid.

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::observables::ln_perimeter::LnPerimeter;
use crate::analysis::panels::{Panel, PanelOpts};

use super::{PairPanelData, CTRL_ALPHA, CTRL_COLOR, SOFT_ALPHA, SOFT_COLOR};

/// Pair-comparison L_n(t) renderer. `decimate_max = Some(N)` thins
/// each trace to at most ~N points via step_by; default None preserves
/// every frame for full-fidelity rendering.
#[derive(Default)]
pub struct LnTimeseriesPair {
    pub decimate_max: Option<usize>,
}

impl<'a, 'b> Panel<'a, 'b> for LnTimeseriesPair {
    type Data = PairPanelData<'a>;

    fn id(&self) -> &'static str {
        "ln_timeseries_pair"
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
            .get::<LnPerimeter>()
            .ok_or_else(|| anyhow!("ln_timeseries_pair: numerator missing ln_perimeter"))?;
        let den = data
            .denominator
            .bag
            .get::<LnPerimeter>()
            .ok_or_else(|| anyhow!("ln_timeseries_pair: denominator missing ln_perimeter"))?;

        let valid: Vec<f64> = num
            .series
            .iter()
            .chain(den.series.iter())
            .copied()
            .filter(|v| v.is_finite() && *v > 0.5)
            .collect();
        let data_min = valid.iter().copied().fold(f64::INFINITY, f64::min);
        let data_max = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let (y_min, y_max) = if let Some((lo, hi)) = opts.y_range {
            (lo, hi)
        } else {
            let auto_min = ((data_min - 0.03) * 20.0).floor() / 20.0;
            let auto_max = ((data_max + 0.03) * 20.0).ceil() / 20.0;
            (0.98_f64.min(auto_min), 1.50_f64.max(auto_max))
        };
        let x_max = num
            .t_tau
            .last()
            .copied()
            .unwrap_or(1.0)
            .max(den.t_tau.last().copied().unwrap_or(1.0));

        let title = opts.title.clone().unwrap_or_else(|| "Cell 0 L_n(t)".into());
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

        let step_n = self.decimate_max
            .map(|m| (num.t_tau.len() / m.max(1)).max(1))
            .unwrap_or(1);
        let step_d = self.decimate_max
            .map(|m| (den.t_tau.len() / m.max(1)).max(1))
            .unwrap_or(1);
        chart
            .draw_series(LineSeries::new(
                num.t_tau
                    .iter()
                    .step_by(step_n)
                    .zip(num.series.iter().step_by(step_n))
                    .map(|(&t, &l)| (t, l.min(y_max).max(y_min))),
                SOFT_ALPHA.stroke_width(1),
            ))?
            .label(format!("{} (μ={:.3})", data.numerator_label, num.mean))
            .legend(move |(x, y)| Rectangle::new([(x, y - 2), (x + 12, y + 2)], SOFT_COLOR.filled()));
        chart
            .draw_series(LineSeries::new(
                den.t_tau
                    .iter()
                    .step_by(step_d)
                    .zip(den.series.iter().step_by(step_d))
                    .map(|(&t, &l)| (t, l.min(y_max).max(y_min))),
                CTRL_ALPHA.stroke_width(1),
            ))?
            .label(format!("{} (μ={:.3})", data.denominator_label, den.mean))
            .legend(move |(x, y)| Rectangle::new([(x, y - 2), (x + 12, y + 2)], CTRL_COLOR.filled()));
        // Reference line at L_n = 1
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
