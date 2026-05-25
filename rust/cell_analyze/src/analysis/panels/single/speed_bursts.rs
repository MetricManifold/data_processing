//! Single-run speed bursts |v|(t).

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::observables::displacement_velocities::DisplacementVelocities;
use crate::analysis::panels::{Panel, PanelOpts};

use super::{SingleRunData, SINGLE_COLOR};

pub struct SpeedBurstsSingle {
    pub speed_max: f64,
}

impl Default for SpeedBurstsSingle {
    fn default() -> Self {
        Self { speed_max: 0.02 }
    }
}

impl<'a, 'b> Panel<'a, 'b> for SpeedBurstsSingle {
    type Data = SingleRunData<'a>;

    fn id(&self) -> &'static str {
        "speed_bursts_single"
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
            .get::<DisplacementVelocities>()
            .ok_or_else(|| anyhow!("speed_bursts_single: missing displacement_velocities"))?;

        let x_max = opts
            .x_range
            .map(|r| r.1)
            .unwrap_or_else(|| v.t_tau.last().copied().unwrap_or(1.0));
        let y_max = opts.y_range.map(|r| r.1).unwrap_or(self.speed_max);
        let burst_thresh = v.mean_speed + 3.0 * v.std_speed;
        let v_a = data.run.params.v_a;

        let title = opts.title.clone().unwrap_or_else(|| "|v|(t)".into());
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 16))
            .margin(8)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..x_max, 0.0..y_max)?;
        chart
            .configure_mesh()
            .x_desc("t (τ)")
            .y_desc("|v|")
            .x_label_style(("sans-serif", 14))
            .y_label_style(("sans-serif", 14))
            .light_line_style(TRANSPARENT)
            .bold_line_style(RGBAColor(200, 200, 200, 0.3))
            .draw()?;

        let step = (v.t_tau.len() / 1000).max(1);
        chart
            .draw_series(LineSeries::new(
                v.t_tau
                    .iter()
                    .step_by(step)
                    .zip(v.speeds.iter().step_by(step))
                    .map(|(&t, &s)| (t, s.min(y_max))),
                SINGLE_COLOR.stroke_width(1),
            ))?
            .label(data.label)
            .legend(move |(x, y)| Rectangle::new([(x, y - 2), (x + 12, y + 2)], SINGLE_COLOR.filled()));
        chart
            .draw_series(LineSeries::new(
                vec![(0.0, burst_thresh), (x_max, burst_thresh)],
                BLACK.mix(0.4).stroke_width(1),
            ))?
            .label(format!("μ+3σ={:.4}", burst_thresh));
        chart.draw_series(LineSeries::new(
            vec![(0.0, v_a), (x_max, v_a)],
            BLACK.mix(0.2).stroke_width(1),
        ))?;
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 9))
            .draw()?;
        chart.plotting_area().draw(&PathElement::new(
            vec![(0.0, y_max), (x_max, y_max), (x_max, 0.0)],
            BLACK.mix(0.5).stroke_width(1),
        ))?;
        Ok(())
    }
}
