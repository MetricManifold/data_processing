//! Summary text panel: parameters + ratios in a monospace block.

use anyhow::Result;
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::observables::displacement_velocities::DisplacementVelocities;
use crate::analysis::observables::ln_perimeter::LnPerimeter;
use crate::analysis::observables::msd_palmieri::MsdPalmieri;
use crate::analysis::panels::{Panel, PanelOpts};

use super::PairPanelData;

pub struct SummaryPair;

fn ratio(num: f64, den: f64) -> f64 {
    if den.abs() > 1e-30 {
        num / den
    } else {
        f64::NAN
    }
}

fn fmt_opt(v: Option<f64>, prec: usize) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.prec$}"),
        _ => "NA".to_string(),
    }
}

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

        let n = &data.numerator.metadata;
        let d = &data.denominator.metadata;

        let mut lines: Vec<String> = vec![
            "--- Parameters ---".into(),
            format!(
                "v_A={:.3} τ={:.0} R={:.0}",
                data.numerator.params.v_a,
                data.numerator.params.tau,
                data.numerator.params.cell_radius
            ),
            format!("N={} dim={} L=({:.1},{:.1})", n.n_cells, n.dim, n.lx, n.ly),
            format!(
                "dt(n/d)={}/{} tau(n/d)={}/{}",
                fmt_opt(n.dt, 4),
                fmt_opt(d.dt, 4),
                fmt_opt(n.tau, 1),
                fmt_opt(d.tau, 1)
            ),
            format!("tau source n/d: {}/{}", n.tau_source, d.tau_source),
            format!(
                "t_n=[{},{}] Δ={} f={} (sub={})",
                fmt_opt(n.time_start, 1),
                fmt_opt(n.time_end, 1),
                fmt_opt(n.duration, 1),
                n.frame_count,
                n.subsample
            ),
            format!(
                "t_d=[{},{}] Δ={} f={} (sub={})",
                fmt_opt(d.time_start, 1),
                fmt_opt(d.time_end, 1),
                fmt_opt(d.duration, 1),
                d.frame_count,
                d.subsample
            ),
            format!(
                "ckpt n/d: tτ={}/{} step={:?}/{:?}",
                fmt_opt(n.checkpoint_tau, 2),
                fmt_opt(d.checkpoint_tau, 2),
                n.checkpoint_step,
                d.checkpoint_step
            ),
            format!(
                "tagged γ n/d: {}/{}; tagged v_A n/d: {}/{}",
                fmt_opt(n.tagged_gamma, 3),
                fmt_opt(d.tagged_gamma, 3),
                fmt_opt(n.tagged_v_a, 3),
                fmt_opt(d.tagged_v_a, 3)
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
            let r = ratio(n.d_eff_cell, d.d_eff_cell);
            lines.push(format!(
                "D_eff c0: {:.4}/{:.4}={:.2}",
                n.d_eff_cell, d.d_eff_cell, r
            ));
            lines.push(format!(
                "D_eff pop: {:.4}/{:.4}={:.2}",
                n.d_eff_pop,
                d.d_eff_pop,
                ratio(n.d_eff_pop, d.d_eff_pop)
            ));
        }
        if let (Some(n), Some(d)) = (num_ln, den_ln) {
            let r = ratio(n.mean, d.mean);
            lines.push(format!("L_n c0: {:.3}/{:.3}={:.2}", n.mean, d.mean, r));
        }
        if let (Some(n), Some(d)) = (num_v, den_v) {
            let r = ratio(n.mean_speed, d.mean_speed);
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

        let n_nonempty = lines.iter().filter(|s| !s.is_empty()).count().max(1) as f64;
        let step = (0.86 / n_nonempty).max(0.032);
        let mut draw_idx = 0usize;
        for line in lines.iter() {
            if line.is_empty() {
                continue;
            }
            let y = 0.95 - draw_idx as f64 * step;
            draw_idx += 1;
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
