//! Single-run MSD/Δt → 4·D_eff curve.

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::observables::msd_palmieri::MsdPalmieri;
use crate::analysis::panels::{Panel, PanelOpts};

use super::{SingleRunData, SINGLE_COLOR};

pub struct MsdSingle {
    pub msd_lag_max: f64,
    pub show_population: bool,
}

impl Default for MsdSingle {
    fn default() -> Self {
        Self {
            msd_lag_max: 8.0,
            show_population: true,
        }
    }
}

impl<'a, 'b> Panel<'a, 'b> for MsdSingle {
    type Data = SingleRunData<'a>;

    fn id(&self) -> &'static str {
        "msd_single"
    }

    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()> {
        let m = data
            .run
            .bag
            .get::<MsdPalmieri>()
            .ok_or_else(|| anyhow!("msd_single: missing msd_palmieri"))?;

        let x_max = opts.x_range.map(|r| r.1).unwrap_or(self.msd_lag_max);
        let cell_pts: Vec<(f64, f64)> = m
            .lag_tau
            .iter()
            .zip(m.msd_t_cell.iter())
            .filter(|(x, y)| **x <= x_max && **x > 0.0 && **y > 0.0)
            .map(|(&x, &y)| (x, y))
            .collect();
        let pop_pts: Vec<(f64, f64)> = if self.show_population {
            m.lag_tau
                .iter()
                .zip(m.msd_t_pop.iter())
                .filter(|(x, y)| **x <= x_max && **x > 0.0 && **y > 0.0)
                .map(|(&x, &y)| (x, y))
                .collect()
        } else {
            vec![]
        };

        let all_y: Vec<f64> = cell_pts
            .iter()
            .chain(pop_pts.iter())
            .map(|p| p.1)
            .collect();
        let y_lo = all_y.iter().copied().fold(f64::INFINITY, f64::min);
        let y_hi = all_y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let y_pad = (y_hi - y_lo).max(1e-6) * 0.1;
        let y_max = opts.y_range.map(|r| r.1).unwrap_or(y_hi + y_pad);
        let y_min = opts
            .y_range
            .map(|r| r.0)
            .unwrap_or((y_lo - y_pad).max(0.0));

        let title = opts
            .title
            .clone()
            .unwrap_or_else(|| "MSD/Δt → 4D_eff".into());
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

        chart
            .draw_series(LineSeries::new(cell_pts, SINGLE_COLOR.stroke_width(2)))?
            .label(format!(
                "{} c0 (D={:.4})",
                data.label, m.d_eff_cell
            ))
            .legend(move |(x, y)| Rectangle::new([(x, y - 2), (x + 12, y + 2)], SINGLE_COLOR.filled()));
        if !pop_pts.is_empty() {
            chart
                .draw_series(LineSeries::new(
                    pop_pts,
                    RGBAColor(120, 120, 120, 0.4).stroke_width(1),
                ))?
                .label("Population");
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::LowerRight)
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
