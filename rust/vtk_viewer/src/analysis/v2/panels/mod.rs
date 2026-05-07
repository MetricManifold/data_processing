//! Panels: typed renderers that paint a single chart inside a
//! `plotters::DrawingArea`. Each panel consumes a typed `Data` (a
//! sweep, a pair, a single run, ...) so that adding a new panel is just
//! a new file with an `impl Panel`.
//!
//! Layout helpers (multi-panel composition, color palette, default
//! styles) live in [`layout`].

pub mod gvi_core;
pub mod layout;
pub mod overlay;
pub mod pair;
pub mod single;
pub mod sweep;

use anyhow::Result;
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;

/// Common panel options. Concrete panels may extend this with their own
/// struct, but the basics live here so the TOML can configure them
/// uniformly.
#[derive(Clone, Debug, Default)]
pub struct PanelOpts {
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub x_range: Option<(f64, f64)>,
    pub y_range: Option<(f64, f64)>,
    pub log_x: bool,
    pub log_y: bool,
}

/// A panel that knows how to render itself into a drawing area given
/// its typed input data.
///
/// Two lifetimes:
/// - `'a` flows into the associated `Data` type (so panels can borrow
///   from outer state, e.g. a slice of `RunAnalysis`).
/// - `'b` is the SVG backend lifetime.
pub trait Panel<'a, 'b> {
    type Data;
    fn id(&self) -> &'static str;
    fn render(
        &self,
        area: &DrawingArea<SVGBackend<'b>, Shift>,
        data: &Self::Data,
        opts: &PanelOpts,
    ) -> Result<()>;
}
