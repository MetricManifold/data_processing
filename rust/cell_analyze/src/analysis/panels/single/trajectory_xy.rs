//! Single-run XY trajectory path panel.
//!
//! Reads `TrajectoryXy` observable output and draws each requested
//! cell's path. Three view modes:
//! - `Unwrapped` (default): axes auto-fit to the unwrapped path extent;
//!   a faint domain box at (0,0)-(Lx,Ly) is drawn for context.
//! - `Box`: axes are exactly (0,0)-(Lx,Ly) and the path is wrapped into
//!   the periodic box, splitting segments that cross a boundary so we
//!   can see periodicity rendered correctly.
//! - `BoxUnwrapped`: axes are exactly (0,0)-(Lx,Ly) but path is drawn
//!   unwrapped — useful for cells that stay inside the original tile.
//!
//! Color modes: `Time` (viridis-by-time-fraction) or `Solid` (one
//! palette color per cell).

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::observables::trajectory_xy::TrajectoryXy;
use crate::analysis::panels::{Panel, PanelOpts};

use super::SingleRunData;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    Unwrapped,
    Box,
    BoxUnwrapped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    Time,
    Solid,
}

pub struct TrajectoryXySingle {
    pub view: ViewMode,
    pub color: ColorMode,
    /// Draw the (0,0)-(Lx,Ly) periodic-box outline (default true).
    pub draw_box: bool,
}

impl Default for TrajectoryXySingle {
    fn default() -> Self {
        Self {
            view: ViewMode::Unwrapped,
            color: ColorMode::Time,
            draw_box: true,
        }
    }
}

/// Per-cell palette for `solid` mode.
const SOLID_PALETTE: &[(u8, u8, u8)] = &[
    (33, 144, 141),
    (200, 50, 80),
    (90, 70, 200),
    (220, 150, 30),
];

/// Sample a viridis-like color at u in [0, 1].
fn viridis_at(u: f64) -> RGBColor {
    let u = u.clamp(0.0, 1.0);
    let stops: &[(f64, (u8, u8, u8))] = &[
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 144, 141)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ];
    for w in stops.windows(2) {
        let (a, ca) = w[0];
        let (b, cb) = w[1];
        if u <= b {
            let t = if b > a { (u - a) / (b - a) } else { 0.0 };
            let r = (ca.0 as f64 + t * (cb.0 as f64 - ca.0 as f64)) as u8;
            let g = (ca.1 as f64 + t * (cb.1 as f64 - ca.1 as f64)) as u8;
            let b = (ca.2 as f64 + t * (cb.2 as f64 - ca.2 as f64)) as u8;
            return RGBColor(r, g, b);
        }
    }
    let (_, c) = stops[stops.len() - 1];
    RGBColor(c.0, c.1, c.2)
}

/// Wrap a (p0 -> p1) segment into [0, Lx) x [0, Ly), splitting at any
/// boundary crossings so visual periodicity is correct.
///
/// Algorithm: reduce p1 to its minimum-image displacement from p0,
/// then walk along the (potentially short) segment, inserting a break
/// every time it would cross x=0, x=Lx, y=0, or y=Ly. Each subsegment
/// is shifted by ±Lx / ±Ly so it sits inside the canonical box.
pub(crate) fn wrap_segment(
    mut p0: [f64; 2],
    p1_raw: [f64; 2],
    lx: f64,
    ly: f64,
) -> Vec<([f64; 2], [f64; 2])> {
    // Canonicalise p0 into [0, Lx) x [0, Ly).
    p0[0] = p0[0].rem_euclid(lx);
    p0[1] = p0[1].rem_euclid(ly);
    // Minimum-image displacement to p1.
    let mut dx = p1_raw[0] - p0[0];
    let mut dy = p1_raw[1] - p0[1];
    dx -= (dx / lx).round() * lx;
    dy -= (dy / ly).round() * ly;

    let mut out = Vec::new();
    let mut cur = p0;
    let mut t = 0.0_f64;
    // Cap iterations so a degenerate input can't loop forever.
    for _ in 0..32 {
        if t >= 1.0 - 1e-12 { break; }
        // Find the next boundary crossing time, if any.
        let mut t_next = 1.0;
        let mut shift: [f64; 2] = [0.0, 0.0];
        if dx > 1e-12 {
            let tx = ((lx - cur[0]) / dx).clamp(0.0, 1.0 - t) + t;
            if tx < t_next - 1e-12 { t_next = tx; shift = [-lx, 0.0]; }
        } else if dx < -1e-12 {
            let tx = (-cur[0] / dx).clamp(0.0, 1.0 - t) + t;
            if tx < t_next - 1e-12 { t_next = tx; shift = [lx, 0.0]; }
        }
        if dy > 1e-12 {
            let ty = ((ly - cur[1]) / dy).clamp(0.0, 1.0 - t) + t;
            if ty < t_next - 1e-12 { t_next = ty; shift = [0.0, -ly]; }
        } else if dy < -1e-12 {
            let ty = (-cur[1] / dy).clamp(0.0, 1.0 - t) + t;
            if ty < t_next - 1e-12 { t_next = ty; shift = [0.0, ly]; }
        }
        // Endpoint of this subsegment (in pre-shift coords).
        let end = [
            p0[0] + (t_next) * dx,
            p0[1] + (t_next) * dy,
        ];
        out.push((cur, end));
        if t_next >= 1.0 - 1e-12 { break; }
        // Hop to the other side of the box.
        cur = [end[0] + shift[0], end[1] + shift[1]];
        // Snap to exactly 0 or Lx/Ly along whichever axis we just
        // crossed, so floating-point noise doesn't leave a one-pixel
        // gap on the boundary. shift = -Lx means we crossed x=Lx, so
        // the wrapped position is x=0 (and symmetrically).
        if shift[0] < 0.0 { cur[0] = 0.0; }
        if shift[0] > 0.0 { cur[0] = lx; }
        if shift[1] < 0.0 { cur[1] = 0.0; }
        if shift[1] > 0.0 { cur[1] = ly; }
        // We're starting a new subsegment from `cur`. To stay
        // consistent we redefine the parametric form: re-anchor p0=cur
        // and shrink the remaining (dx, dy) by the fraction we've used.
        // Equivalently: rebuild p0/dx/dy for the residual segment.
        let remaining = 1.0 - t_next;
        let new_dx = dx * remaining;
        let new_dy = dy * remaining;
        p0 = cur;
        dx = new_dx;
        dy = new_dy;
        t = 0.0;
    }
    out
}

impl<'a, 'b> Panel<'a, 'b> for TrajectoryXySingle {
    type Data = SingleRunData<'a>;

    fn id(&self) -> &'static str {
        "trajectory_xy_single"
    }

    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()> {
        let txy = data
            .run
            .bag
            .get::<TrajectoryXy>()
            .ok_or_else(|| anyhow!("trajectory_xy_single: missing trajectory_xy observable"))?;

        if txy.paths.is_empty() {
            let title = opts.title.clone().unwrap_or_else(|| "trajectory (x, y)".into());
            let _ = ChartBuilder::on(area)
                .caption(format!("{} (no paths)", title), ("sans-serif", 16))
                .margin(8)
                .build_cartesian_2d(0.0..1.0, 0.0..1.0)?
                .configure_mesh()
                .draw();
            return Ok(());
        }

        // Axes.
        let (mut x_min, mut x_max, mut y_min, mut y_max) = match self.view {
            ViewMode::Box | ViewMode::BoxUnwrapped => (0.0, txy.lx, 0.0, txy.ly),
            ViewMode::Unwrapped => {
                let mut x_lo = f64::INFINITY;
                let mut x_hi = f64::NEG_INFINITY;
                let mut y_lo = f64::INFINITY;
                let mut y_hi = f64::NEG_INFINITY;
                for path in &txy.paths {
                    for [x, y] in &path.xy {
                        if *x < x_lo { x_lo = *x; }
                        if *x > x_hi { x_hi = *x; }
                        if *y < y_lo { y_lo = *y; }
                        if *y > y_hi { y_hi = *y; }
                    }
                }
                let pad_x = ((x_hi - x_lo).abs() * 0.05).max(txy.lx * 0.02);
                let pad_y = ((y_hi - y_lo).abs() * 0.05).max(txy.ly * 0.02);
                (x_lo - pad_x, x_hi + pad_x, y_lo - pad_y, y_hi + pad_y)
            }
        };
        if let Some((lo, hi)) = opts.x_range { x_min = lo; x_max = hi; }
        if let Some((lo, hi)) = opts.y_range { y_min = lo; y_max = hi; }

        let title = opts
            .title
            .clone()
            .unwrap_or_else(|| format!("trajectory ({})", data.label));

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
                vec![
                    (0.0, 0.0),
                    (txy.lx, 0.0),
                    (txy.lx, txy.ly),
                    (0.0, txy.ly),
                    (0.0, 0.0),
                ],
                BLACK.mix(0.25).stroke_width(1),
            ))?;
        }

        for (ci, path) in txy.paths.iter().enumerate() {
            let n = path.xy.len();
            if n < 2 { continue; }
            let solid = {
                let (r, g, b) = SOLID_PALETTE[ci % SOLID_PALETTE.len()];
                RGBColor(r, g, b)
            };
            let stroke_for = |i: usize| -> RGBColor {
                match self.color {
                    ColorMode::Solid => solid,
                    ColorMode::Time => {
                        let u = i as f64 / (n - 1).max(1) as f64;
                        viridis_at(u)
                    }
                }
            };

            match self.view {
                ViewMode::Unwrapped | ViewMode::BoxUnwrapped => {
                    if self.color == ColorMode::Solid {
                        let pts: Vec<(f64, f64)> =
                            path.xy.iter().map(|p| (p[0], p[1])).collect();
                        chart.draw_series(LineSeries::new(pts, solid.stroke_width(2)))?;
                    } else {
                        for i in 0..(n - 1) {
                            let p0 = (path.xy[i][0], path.xy[i][1]);
                            let p1 = (path.xy[i + 1][0], path.xy[i + 1][1]);
                            chart.draw_series(LineSeries::new(
                                vec![p0, p1],
                                stroke_for(i).stroke_width(2),
                            ))?;
                        }
                    }
                }
                ViewMode::Box => {
                    for i in 0..(n - 1) {
                        let sub = wrap_segment(path.xy[i], path.xy[i + 1], txy.lx, txy.ly);
                        let c = stroke_for(i);
                        for (a, b) in sub {
                            chart.draw_series(LineSeries::new(
                                vec![(a[0], a[1]), (b[0], b[1])],
                                c.stroke_width(2),
                            ))?;
                        }
                    }
                }
            }

            // Start + end markers.
            let (start_xy, end_xy) = match self.view {
                ViewMode::Box => {
                    let s = path.xy.first().copied().unwrap_or([0.0, 0.0]);
                    let e = path.xy.last().copied().unwrap_or([0.0, 0.0]);
                    (
                        [s[0].rem_euclid(txy.lx), s[1].rem_euclid(txy.ly)],
                        [e[0].rem_euclid(txy.lx), e[1].rem_euclid(txy.ly)],
                    )
                }
                _ => (
                    path.xy.first().copied().unwrap_or([0.0, 0.0]),
                    path.xy.last().copied().unwrap_or([0.0, 0.0]),
                ),
            };
            chart.draw_series(std::iter::once(Circle::new(
                (start_xy[0], start_xy[1]),
                4,
                BLACK.filled(),
            )))?;
            let mw = (x_max - x_min) * 0.008;
            let mh = (y_max - y_min) * 0.008;
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (end_xy[0] - mw, end_xy[1] - mh),
                    (end_xy[0] + mw, end_xy[1] + mh),
                ],
                BLACK.stroke_width(2),
            )))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_no_crossing() {
        let segs = wrap_segment([1.0, 1.0], [3.0, 4.0], 10.0, 10.0);
        assert_eq!(segs.len(), 1);
        assert!((segs[0].0[0] - 1.0).abs() < 1e-9);
        assert!((segs[0].1[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn wrap_crosses_right_edge() {
        // p0 just left of x=Lx, p1 just past — split into two.
        let segs = wrap_segment([9.5, 5.0], [10.5, 5.0], 10.0, 10.0);
        assert!(segs.len() >= 2);
        // First subsegment ends at x ≈ 10 (the right boundary).
        assert!((segs[0].1[0] - 10.0).abs() < 1e-6);
        // Second subsegment starts at x ≈ 0 (wrap to left edge).
        assert!((segs[1].0[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn wrap_minimum_image_short_path() {
        // p0 = 0.5, p1 = 9.5 — minimum image goes left across x=0,
        // not right across the interior.
        let segs = wrap_segment([0.5, 5.0], [9.5, 5.0], 10.0, 10.0);
        assert!(segs.len() >= 2);
        // First subsegment should hit x ≈ 0.
        assert!((segs[0].1[0] - 0.0).abs() < 1e-6);
    }
}
