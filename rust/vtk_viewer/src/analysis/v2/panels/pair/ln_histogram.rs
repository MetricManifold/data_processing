//! L_n histogram for the tagged cell across both runs.

use anyhow::{anyhow, Result};
use plotters::backend::SVGBackend;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

use crate::analysis::v2::observables::ln_perimeter::LnPerimeter;
use crate::analysis::v2::panels::{Panel, PanelOpts};

use super::{PairPanelData, CTRL_ALPHA, SOFT_ALPHA};

pub struct LnHistogramPair {
    pub n_bins: usize,
}

impl Default for LnHistogramPair {
    fn default() -> Self {
        Self { n_bins: 40 }
    }
}

impl<'a, 'b> Panel<'a, 'b> for LnHistogramPair {
    type Data = PairPanelData<'a>;

    fn id(&self) -> &'static str {
        "ln_histogram_pair"
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
            .get::<LnPerimeter>()
            .ok_or_else(|| anyhow!("ln_histogram_pair: numerator missing ln_perimeter"))?;
        let den = data
            .denominator
            .bag
            .get::<LnPerimeter>()
            .ok_or_else(|| anyhow!("ln_histogram_pair: denominator missing ln_perimeter"))?;

        let mut all_ln: Vec<f64> = num
            .series
            .iter()
            .chain(den.series.iter())
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        all_ln.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let (ln_min, ln_max) = if let Some((lo, hi)) = opts.x_range {
            (lo, hi)
        } else {
            let auto_min = all_ln.first().copied().unwrap_or(0.98_f64).max(0.98);
            let auto_max = all_ln.last().copied().unwrap_or(1.5_f64) + 0.02;
            (auto_min, auto_max.max(1.5))
        };
        let bw = (ln_max - ln_min) / self.n_bins as f64;

        let mut s_hist = vec![0u32; self.n_bins];
        let mut c_hist = vec![0u32; self.n_bins];
        for &v in &num.series {
            if !v.is_finite() {
                continue;
            }
            let b = ((v - ln_min) / bw).floor() as i64;
            if b >= 0 && (b as usize) < self.n_bins {
                s_hist[b as usize] += 1;
            }
        }
        for &v in &den.series {
            if !v.is_finite() {
                continue;
            }
            let b = ((v - ln_min) / bw).floor() as i64;
            if b >= 0 && (b as usize) < self.n_bins {
                c_hist[b as usize] += 1;
            }
        }
        let s_total = num.series.iter().filter(|v| v.is_finite()).count().max(1) as f64;
        let c_total = den.series.iter().filter(|v| v.is_finite()).count().max(1) as f64;
        let s_density: Vec<f64> = s_hist.iter().map(|&c| c as f64 / (s_total * bw)).collect();
        let c_density: Vec<f64> = c_hist.iter().map(|&c| c as f64 / (c_total * bw)).collect();
        let y_max = s_density
            .iter()
            .chain(c_density.iter())
            .copied()
            .fold(0.0f64, f64::max)
            * 1.2;

        let title = opts
            .title
            .clone()
            .unwrap_or_else(|| "Cell 0 L_n distribution".into());
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 16))
            .margin(8)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(ln_min..ln_max, 0.0..y_max)?;
        chart
            .configure_mesh()
            .x_desc("L_n")
            .y_desc("Density")
            .x_label_style(("sans-serif", 14))
            .y_label_style(("sans-serif", 14))
            .light_line_style(TRANSPARENT)
            .bold_line_style(RGBAColor(200, 200, 200, 0.3))
            .draw()?;

        chart
            .draw_series(s_density.iter().enumerate().map(|(i, &d)| {
                let x0 = ln_min + i as f64 * bw;
                Rectangle::new([(x0, 0.0), (x0 + bw, d)], SOFT_ALPHA.filled())
            }))?
            .label(data.numerator_label)
            .legend(move |(x, y)| Rectangle::new([(x, y - 3), (x + 12, y + 3)], SOFT_ALPHA.filled()));
        chart
            .draw_series(c_density.iter().enumerate().map(|(i, &d)| {
                let x0 = ln_min + i as f64 * bw;
                Rectangle::new([(x0, 0.0), (x0 + bw, d)], CTRL_ALPHA.filled())
            }))?
            .label(data.denominator_label)
            .legend(move |(x, y)| Rectangle::new([(x, y - 3), (x + 12, y + 3)], CTRL_ALPHA.filled()));

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 10))
            .draw()?;
        chart.plotting_area().draw(&PathElement::new(
            vec![(ln_min, y_max), (ln_max, y_max), (ln_max, 0.0)],
            BLACK.mix(0.5).stroke_width(1),
        ))?;
        Ok(())
    }
}
