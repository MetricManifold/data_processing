//! Layout helpers: split a figure into a grid of panels, compute axis
//! ranges with sensible padding, default color palette.

use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::style::RGBAColor;

/// Default color palette (max-contrast 8-color set tuned for reds/blues
/// against the soft-vs-hard convention used elsewhere in the codebase).
pub const PALETTE: &[RGBAColor] = &[
    RGBAColor(220, 60, 60, 1.0),    // red — soft
    RGBAColor(52, 152, 219, 1.0),   // blue — ctrl
    RGBAColor(46, 204, 113, 1.0),   // green
    RGBAColor(241, 196, 15, 1.0),   // yellow
    RGBAColor(155, 89, 182, 1.0),   // purple
    RGBAColor(230, 126, 34, 1.0),   // orange
    RGBAColor(26, 188, 156, 1.0),   // teal
    RGBAColor(149, 165, 166, 1.0),  // gray
];

/// Split a figure area into (rows × cols) panels with a title strip on
/// top. Returns (title_area, panels) where `panels` is a row-major Vec.
pub fn grid<'b>(
    root: &DrawingArea<SVGBackend<'b>, Shift>,
    rows: usize,
    cols: usize,
    title_height: u32,
) -> (
    DrawingArea<SVGBackend<'b>, Shift>,
    Vec<DrawingArea<SVGBackend<'b>, Shift>>,
) {
    let (title_area, chart_area) = root.split_vertically(title_height);
    let panels = chart_area.split_evenly((rows, cols));
    (title_area, panels)
}

/// Pad a [min, max] range by a fractional amount (default 5%) so points
/// near the edge aren't clipped against the axis.
pub fn padded(min: f64, max: f64, frac: f64) -> (f64, f64) {
    if !min.is_finite() || !max.is_finite() {
        return (-1.0, 1.0);
    }
    let span = (max - min).abs().max(max.abs() * 0.01).max(1e-12);
    (min - span * frac, max + span * frac)
}
