//! D_eff bar chart: 4 bars showing soft pop, soft c0, ctrl pop, ctrl c0.

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::observables::msd_palmieri::MsdPalmieri;
use crate::analysis::panels::{Panel, PanelOpts};

use super::{PairPanelData, CTRL_ALPHA, CTRL_COLOR, SOFT_ALPHA, SOFT_COLOR};

pub struct DeffBarPair;

impl<'a, 'b> Panel<'a, 'b> for DeffBarPair {
    type Data = PairPanelData<'a>;

    fn id(&self) -> &'static str {
        "deff_bar_pair"
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
            .ok_or_else(|| anyhow!("deff_bar_pair: numerator missing msd_palmieri"))?;
        let den = data
            .denominator
            .bag
            .get::<MsdPalmieri>()
            .ok_or_else(|| anyhow!("deff_bar_pair: denominator missing msd_palmieri"))?;

        let vals = [num.d_eff_pop, num.d_eff_cell, den.d_eff_pop, den.d_eff_cell];
        let y_max = vals
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0f64, f64::max)
            * 1.3;
        let labels = [
            format!("{}\npop", data.numerator_label),
            format!("{}\nc0", data.numerator_label),
            format!("{}\npop", data.denominator_label),
            format!("{}\nc0", data.denominator_label),
        ];

        let title = opts.title.clone().unwrap_or_else(|| "D_eff at 8τ".into());
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 14))
            .margin(8)
            .x_label_area_size(45)
            .y_label_area_size(50)
            .build_cartesian_2d(-0.5_f64..3.5_f64, 0.0..y_max)?;
        chart
            .configure_mesh()
            .y_desc("D_eff")
            .light_line_style(TRANSPARENT)
            .bold_line_style(TRANSPARENT)
            .x_labels(4)
            .x_label_style(("sans-serif", 8))
            .y_label_style(("sans-serif", 9))
            .x_label_formatter(&|x| {
                let idx = x.round() as usize;
                if idx < 4 {
                    labels[idx].clone()
                } else {
                    String::new()
                }
            })
            .draw()?;

        let bar_w = 0.35;
        let colors = [SOFT_ALPHA, SOFT_COLOR, CTRL_ALPHA, CTRL_COLOR];
        for (i, (&v, c)) in vals.iter().zip(colors.iter()).enumerate() {
            if !v.is_finite() {
                continue;
            }
            let x = i as f64;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x - bar_w, 0.0), (x + bar_w, v)],
                c.filled(),
            )))?;
            chart.draw_series(std::iter::once(Text::new(
                format!("{:.4}", v),
                (x, v + y_max * 0.02),
                ("sans-serif", 8).into_font(),
            )))?;
        }
        chart.plotting_area().draw(&PathElement::new(
            vec![(-0.5, y_max), (3.5, y_max), (3.5, 0.0)],
            BLACK.mix(0.5).stroke_width(1),
        ))?;
        Ok(())
    }
}
