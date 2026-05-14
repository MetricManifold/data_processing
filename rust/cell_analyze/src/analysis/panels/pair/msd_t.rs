//! MSD/Δt → 4·D_eff curves for both runs.

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::observables::msd_palmieri::MsdPalmieri;
use crate::analysis::panels::{Panel, PanelOpts};

use super::{PairPanelData, CTRL_COLOR, POP_DASH, SOFT_COLOR};

pub struct MsdTPair {
    pub msd_lag_max: f64,
}

impl Default for MsdTPair {
    fn default() -> Self {
        Self { msd_lag_max: 8.0 }
    }
}

impl<'a, 'b> Panel<'a, 'b> for MsdTPair {
    type Data = PairPanelData<'a>;

    fn id(&self) -> &'static str {
        "msd_t_pair"
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
            .get::<MsdPalmieri>()
            .ok_or_else(|| anyhow!("msd_t_pair: numerator missing msd_palmieri"))?;
        let den = data
            .denominator
            .bag
            .get::<MsdPalmieri>()
            .ok_or_else(|| anyhow!("msd_t_pair: denominator missing msd_palmieri"))?;

        let x_max_raw = opts.x_range.map(|r| r.1).unwrap_or(self.msd_lag_max);

        let collect_pts = |lags: &[f64], ys: &[f64]| -> Vec<(f64, f64)> {
            lags.iter()
                .zip(ys.iter())
                .filter(|(x, y)| **x <= x_max_raw && **x > 0.0 && **y > 0.0)
                .map(|(&x, &y)| (x, y))
                .collect()
        };

        let soft_pts = collect_pts(&num.lag_tau, &num.msd_t_cell);
        let ctrl_pts = collect_pts(&den.lag_tau, &den.msd_t_cell);
        let pop_pts = collect_pts(&num.lag_tau, &num.msd_t_pop);

        let all_y: Vec<f64> = soft_pts
            .iter()
            .chain(ctrl_pts.iter())
            .chain(pop_pts.iter())
            .map(|p| p.1)
            .collect();
        // If the filter dropped everything (all-NaN data, all-zero MSD,
        // wrong τ in trajectory header so x-axis range excluded every
        // point, etc.) we cannot build a chart at all. Plotters'
        // build_cartesian_2d will lock up on NaN axis bounds, so bail
        // explicitly with a clear empty-panel render instead.
        if all_y.is_empty() {
            let title = opts.title.clone().unwrap_or_else(|| "MSD/Δt → 4D_eff".into());
            let _ = ChartBuilder::on(area)
                .caption(format!("{} (no data)", title), ("sans-serif", 16))
                .margin(8)
                .x_label_area_size(30)
                .y_label_area_size(50)
                .build_cartesian_2d(0.0..1.0, 0.0..1.0)?
                .configure_mesh().draw();
            eprintln!("[msd_t_pair] no plottable points after filter — \
                       check that trajectory τ matches expected scale \
                       (num.lag_tau range: {:?})",
                      num.lag_tau.first().zip(num.lag_tau.last()));
            return Ok(());
        }
        let y_lo = all_y.iter().copied().fold(f64::INFINITY, f64::min);
        let y_hi = all_y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Even with non-empty data, guard against degenerate ranges
        // (single point, or values all equal) which produce zero span
        // and break axis tick generation in plotters.
        let y_span = (y_hi - y_lo).max(1e-6);
        let y_pad = y_span * 0.1;
        let y_max = opts.y_range.map(|r| r.1).unwrap_or(y_hi + y_pad);
        let y_min = if opts.y_range.is_some() {
            opts.y_range.unwrap().0
        } else {
            (y_lo - y_pad).max(0.0)
        };
        // Final sanity: if y_max <= y_min (range provided is degenerate
        // or padding produced NaN), Plotters will hang on axis math.
        if !y_max.is_finite() || !y_min.is_finite() || y_max <= y_min {
            eprintln!("[msd_t_pair] degenerate y-axis range \
                       [{:?}, {:?}] — skipping panel.", y_min, y_max);
            return Ok(());
        }

        let title = opts
            .title
            .clone()
            .unwrap_or_else(|| "MSD/Δt → 4D_eff".into());
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 16))
            .margin(8)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..x_max_raw, y_min..y_max)?;
        chart
            .configure_mesh()
            .x_desc("Δt (τ)")
            .y_desc("MSD/Δt")
            .x_label_style(("sans-serif", 14))
            .y_label_style(("sans-serif", 14))
            .light_line_style(TRANSPARENT)
            .bold_line_style(RGBAColor(200, 200, 200, 0.3))
            .draw()?;

        chart
            .draw_series(LineSeries::new(soft_pts, SOFT_COLOR.stroke_width(2)))?
            .label(format!(
                "{} c0 (D={:.4})",
                data.numerator_label, num.d_eff_cell
            ))
            .legend(move |(x, y)| Rectangle::new([(x, y - 2), (x + 12, y + 2)], SOFT_COLOR.filled()));
        chart
            .draw_series(LineSeries::new(ctrl_pts, CTRL_COLOR.stroke_width(2)))?
            .label(format!(
                "{} c0 (D={:.4})",
                data.denominator_label, den.d_eff_cell
            ))
            .legend(move |(x, y)| Rectangle::new([(x, y - 2), (x + 12, y + 2)], CTRL_COLOR.filled()));
        chart
            .draw_series(LineSeries::new(pop_pts, POP_DASH.stroke_width(1)))?
            .label("Population");

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::LowerRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 10))
            .draw()?;
        chart.plotting_area().draw(&PathElement::new(
            vec![(0.0, y_max), (x_max_raw, y_max), (x_max_raw, y_min)],
            BLACK.mix(0.5).stroke_width(1),
        ))?;
        Ok(())
    }
}
