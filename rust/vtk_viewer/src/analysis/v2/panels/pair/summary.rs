//! Summary text panel: parameters + ratios in a monospace block.

use anyhow::Result;
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::v2::observables::displacement_velocities::DisplacementVelocities;
use crate::analysis::v2::observables::ln_perimeter::LnPerimeter;
use crate::analysis::v2::observables::msd_palmieri::MsdPalmieri;
use crate::analysis::v2::panels::{Panel, PanelOpts};

use super::PairPanelData;

pub struct SummaryPair;

impl<'a, 'b> Panel<'a, 'b> for SummaryPair {
    type Data = PairPanelData<'a>;

    fn id(&self) -> &'static str {
        "summary_pair"
    }

    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()> {
        area.fill(&WHITE)?;

        let num_msd = data.numerator.bag.get::<MsdPalmieri>();
        let den_msd = data.denominator.bag.get::<MsdPalmieri>();
        let num_ln = data.numerator.bag.get::<LnPerimeter>();
        let den_ln = data.denominator.bag.get::<LnPerimeter>();
        let num_v = data.numerator.bag.get::<DisplacementVelocities>();
        let den_v = data.denominator.bag.get::<DisplacementVelocities>();

        let mut lines: Vec<String> = vec![
            "--- Parameters ---".into(),
            format!(
                "v_A={:.3} τ={:.0} R={:.0}",
                data.numerator.params.v_a,
                data.numerator.params.tau,
                data.numerator.params.cell_radius
            ),
        ];
        // Variables shared by both runs (residual, identifies the pair).
        if !data.numerator.variables.is_empty() {
            let kvs: Vec<String> = data
                .numerator
                .variables
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            lines.push(kvs.join(" "));
        }
        lines.extend([String::new(), "--- Observables ---".into()]);
        if let (Some(n), Some(d)) = (num_msd, den_msd) {
            let r = if d.d_eff_cell.abs() > 1e-30 {
                n.d_eff_cell / d.d_eff_cell
            } else {
                f64::NAN
            };
            lines.push(format!(
                "D_eff c0: {:.4}/{:.4}={:.2}",
                n.d_eff_cell, d.d_eff_cell, r
            ));
            lines.push(format!("D_eff pop: {:.4}/{:.4}", n.d_eff_pop, d.d_eff_pop));
        }
        if let (Some(n), Some(d)) = (num_ln, den_ln) {
            let r = if d.mean.abs() > 1e-30 {
                n.mean / d.mean
            } else {
                f64::NAN
            };
            lines.push(format!("L_n c0: {:.3}/{:.3}={:.2}", n.mean, d.mean, r));
        }
        if let (Some(n), Some(d)) = (num_v, den_v) {
            let r = if d.mean_speed.abs() > 1e-30 {
                n.mean_speed / d.mean_speed
            } else {
                f64::NAN
            };
            lines.push(format!(
                "Speed c0: {:.5}/{:.5}={:.2}",
                n.mean_speed, d.mean_speed, r
            ));
        }
        lines.push(format!(
            "{} vs {}",
            data.numerator_label, data.denominator_label
        ));

        let title = opts.title.clone().unwrap_or_else(|| "Diagnostics".into());
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 14))
            .margin(5)
            .x_label_area_size(0)
            .y_label_area_size(0)
            .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;

        for (i, line) in lines.iter().enumerate() {
            if line.is_empty() {
                continue;
            }
            let y = 0.92 - i as f64 * 0.085;
            let font_size = if line.starts_with("---") { 10 } else { 9 };
            chart.draw_series(std::iter::once(Text::new(
                line.clone(),
                (0.03, y),
                ("monospace", font_size).into_font(),
            )))?;
        }
        Ok(())
    }
}
