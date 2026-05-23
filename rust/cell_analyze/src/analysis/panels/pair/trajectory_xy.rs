//! Pair-comparison XY trajectory panel.
//!
//! Soft vs ctrl: draws both cells' (x, y) paths on the same axes using
//! the pair color palette (soft red, ctrl blue). Same view + draw_box
//! options as the single panel. Annotates net |Δr| ratio (soft / ctrl)
//! in the corner for quick reading.

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::observables::trajectory_xy::TrajectoryXy;
use crate::analysis::panels::single::trajectory_xy::{ViewMode, wrap_segment};
use crate::analysis::panels::{Panel, PanelOpts};

use super::{PairPanelData, CTRL_COLOR, SOFT_COLOR};

pub struct TrajectoryXyPair {
    pub view: ViewMode,
    pub draw_box: bool,
}

impl Default for TrajectoryXyPair {
    fn default() -> Self {
        Self {
            view: ViewMode::Unwrapped,
            draw_box: true,
        }
    }
}

/// Net displacement of the (single) recorded path in a TrajectoryXy
/// observable output. Returns 0 if there are no paths.
fn net_disp(xy: &[[f64; 2]]) -> f64 {
    if xy.len() < 2 { return 0.0; }
    let s = xy[0];
    let e = xy[xy.len() - 1];
    let dx = e[0] - s[0];
    let dy = e[1] - s[1];
    (dx * dx + dy * dy).sqrt()
}

impl<'a, 'b> Panel<'a, 'b> for TrajectoryXyPair {
    type Data = PairPanelData<'a>;

    fn id(&self) -> &'static str {
        "trajectory_xy_pair"
    }

    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()> {
        let soft = data
            .numerator
            .bag
            .get::<TrajectoryXy>()
            .ok_or_else(|| anyhow!("trajectory_xy_pair: numerator missing trajectory_xy"))?;
        let ctrl = data
            .denominator
            .bag
            .get::<TrajectoryXy>()
            .ok_or_else(|| anyhow!("trajectory_xy_pair: denominator missing trajectory_xy"))?;

        if soft.paths.is_empty() || ctrl.paths.is_empty() {
            let title = opts.title.clone().unwrap_or_else(|| "trajectory pair".into());
            let _ = ChartBuilder::on(area)
                .caption(format!("{} (no paths)", title), ("sans-serif", 16))
                .margin(8)
                .build_cartesian_2d(0.0..1.0, 0.0..1.0)?
                .configure_mesh()
                .draw();
            return Ok(());
        }

        // Both observable outputs should share lx/ly (same study). Use
        // numerator's; mismatch is unlikely but we'd just visualize the
        // soft path's box.
        let lx = soft.lx;
        let ly = soft.ly;

        let (mut x_min, mut x_max, mut y_min, mut y_max) = match self.view {
            ViewMode::Box | ViewMode::BoxUnwrapped => (0.0, lx, 0.0, ly),
            ViewMode::Unwrapped => {
                let mut x_lo = f64::INFINITY;
                let mut x_hi = f64::NEG_INFINITY;
                let mut y_lo = f64::INFINITY;
                let mut y_hi = f64::NEG_INFINITY;
                for txy in [soft, ctrl] {
                    for path in &txy.paths {
                        for [x, y] in &path.xy {
                            if *x < x_lo { x_lo = *x; }
                            if *x > x_hi { x_hi = *x; }
                            if *y < y_lo { y_lo = *y; }
                            if *y > y_hi { y_hi = *y; }
                        }
                    }
                }
                let pad_x = ((x_hi - x_lo).abs() * 0.05).max(lx * 0.02);
                let pad_y = ((y_hi - y_lo).abs() * 0.05).max(ly * 0.02);
                (x_lo - pad_x, x_hi + pad_x, y_lo - pad_y, y_hi + pad_y)
            }
        };
        if let Some((lo, hi)) = opts.x_range { x_min = lo; x_max = hi; }
        if let Some((lo, hi)) = opts.y_range { y_min = lo; y_max = hi; }

        // Net-Δr ratio for the corner annotation.
        let soft_net = soft.paths.first().map(|p| net_disp(&p.xy)).unwrap_or(0.0);
        let ctrl_net = ctrl.paths.first().map(|p| net_disp(&p.xy)).unwrap_or(0.0);
        let ratio = if ctrl_net > 1e-9 {
            format!("{:.2}", soft_net / ctrl_net)
        } else {
            "—".into()
        };

        let title = opts.title.clone().unwrap_or_else(|| {
            format!(
                "trajectory ({} vs {})",
                data.numerator_label, data.denominator_label
            )
        });

        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 16))
            .margin(8)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
        chart
            .configure_mesh()
            .x_desc("x")
            .y_desc("y")
            .x_label_style(("sans-serif", 14))
            .y_label_style(("sans-serif", 14))
            .light_line_style(TRANSPARENT)
            .bold_line_style(RGBAColor(200, 200, 200, 0.3))
            .draw()?;

        if self.draw_box {
            chart.draw_series(LineSeries::new(
                vec![(0.0, 0.0), (lx, 0.0), (lx, ly), (0.0, ly), (0.0, 0.0)],
                BLACK.mix(0.25).stroke_width(1),
            ))?;
        }

        // Draw a single (soft, color, label) entry.
        let draw_path = |chart: &mut ChartContext<'_, SVGBackend<'b>, _>,
                         path_xy: &[[f64; 2]],
                         color: RGBAColor|
         -> Result<()> {
            let n = path_xy.len();
            if n < 2 { return Ok(()); }
            match self.view {
                ViewMode::Unwrapped | ViewMode::BoxUnwrapped => {
                    let pts: Vec<(f64, f64)> = path_xy.iter().map(|p| (p[0], p[1])).collect();
                    chart.draw_series(LineSeries::new(pts, color.stroke_width(2)))?;
                }
                ViewMode::Box => {
                    for i in 0..(n - 1) {
                        let sub = wrap_segment(path_xy[i], path_xy[i + 1], lx, ly);
                        for (a, b) in sub {
                            chart.draw_series(LineSeries::new(
                                vec![(a[0], a[1]), (b[0], b[1])],
                                color.stroke_width(2),
                            ))?;
                        }
                    }
                }
            }
            // Start dot + end square.
            let (s_xy, e_xy) = match self.view {
                ViewMode::Box => {
                    let s = path_xy.first().copied().unwrap_or([0.0, 0.0]);
                    let e = path_xy.last().copied().unwrap_or([0.0, 0.0]);
                    (
                        [s[0].rem_euclid(lx), s[1].rem_euclid(ly)],
                        [e[0].rem_euclid(lx), e[1].rem_euclid(ly)],
                    )
                }
                _ => (
                    path_xy.first().copied().unwrap_or([0.0, 0.0]),
                    path_xy.last().copied().unwrap_or([0.0, 0.0]),
                ),
            };
            chart.draw_series(std::iter::once(Circle::new(
                (s_xy[0], s_xy[1]),
                4,
                color.filled(),
            )))?;
            let mw = (x_max - x_min) * 0.008;
            let mh = (y_max - y_min) * 0.008;
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (e_xy[0] - mw, e_xy[1] - mh),
                    (e_xy[0] + mw, e_xy[1] + mh),
                ],
                color.stroke_width(2),
            )))?;
            Ok(())
        };

        // Draw ctrl first so soft sits on top (the soft path is usually
        // the interesting one to inspect).
        for path in &ctrl.paths {
            draw_path(&mut chart, &path.xy, CTRL_COLOR)?;
        }
        for path in &soft.paths {
            draw_path(&mut chart, &path.xy, SOFT_COLOR)?;
        }

        // Legend.
        chart.draw_series(std::iter::once(Text::new(
            format!(
                "{} (|Δr|={:.0})",
                data.numerator_label, soft_net
            ),
            (x_min + (x_max - x_min) * 0.02, y_max - (y_max - y_min) * 0.05),
            ("sans-serif", 14).into_font().color(&SOFT_COLOR),
        )))?;
        chart.draw_series(std::iter::once(Text::new(
            format!(
                "{} (|Δr|={:.0})",
                data.denominator_label, ctrl_net
            ),
            (x_min + (x_max - x_min) * 0.02, y_max - (y_max - y_min) * 0.10),
            ("sans-serif", 14).into_font().color(&CTRL_COLOR),
        )))?;
        chart.draw_series(std::iter::once(Text::new(
            format!("|Δr| ratio = {}", ratio),
            (x_min + (x_max - x_min) * 0.02, y_max - (y_max - y_min) * 0.15),
            ("sans-serif", 14).into_font(),
        )))?;

        Ok(())
    }
}
