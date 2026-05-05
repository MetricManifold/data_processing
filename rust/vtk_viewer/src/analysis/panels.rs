//! Reusable plot panels.
//!
//! These functions take a generic plotters drawing area plus a list of
//! "series" and render a single chart. They are intentionally agnostic
//! about what kind of run produced the data — the same function is used
//! by the `study` paired-comparison panel grid and by single-run plots
//! produced by the `run` subcommand.

use anyhow::Result;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::RGBAColor;

// ===========================================================================
// G(v_i) — Palmieri velocity-distribution panel
// ===========================================================================

#[derive(Clone, Copy)]
pub enum GviMarker {
    Triangle,
    Circle,
}

/// One series of velocity samples to be plotted on a G(v_i) panel.
pub struct GviSeries<'a> {
    pub label: String,
    /// Cartesian velocity components. `vx` and `vy` are concatenated to form
    /// |v_i| samples, matching Palmieri's convention.
    pub vx: &'a [f64],
    pub vy: &'a [f64],
    pub color: RGBAColor,
    pub marker: GviMarker,
}

pub struct GviPanelOpts {
    pub title: String,
    /// Optional "(a)" / "(b)" prefix.
    pub panel_label: Option<char>,
    pub x_max: f64,
    pub y_range: Option<(f64, f64)>,
    /// σ for the Gaussian reference line G = -|v|/(σ√2). If `None` the σ of
    /// the first series is used.
    pub gaussian_ref_sigma: Option<f64>,
    /// If `Some(i)`, fit Palmieri Eq. 5 (ζ via grid search) to series `i` and
    /// draw the resulting curve. The σ used for the fit is `gaussian_ref_sigma`
    /// (or series 0's σ).
    pub palmieri_fit_index: Option<usize>,
    /// Burst velocity v_A used by Eq. 5 (Palmieri 2015 uses v_A = 0.01).
    pub v_a: f64,
    /// Lower |v| bound for the Eq. 5 fit window. Points with |v| below
    /// this threshold are excluded from the SSE so the fit is driven by
    /// the tail rather than the dense bulk near zero. `None` = use `v_a`.
    pub palmieri_fit_min_v: Option<f64>,
    /// If non-empty, draw thin Gaussian reference curves at each listed
    /// σ (in addition to the primary one). Useful for visually exploring
    /// what σ best matches the data.
    pub gaussian_sigma_sweep: Vec<f64>,
}

impl Default for GviPanelOpts {
    fn default() -> Self {
        Self {
            title: "G(v_i)".to_string(),
            panel_label: None,
            x_max: 0.022,
            y_range: None,
            gaussian_ref_sigma: None,
            palmieri_fit_index: None,
            v_a: 0.01,
            palmieri_fit_min_v: None,
            gaussian_sigma_sweep: Vec::new(),
        }
    }
}

/// Empirical CCDF-based G(v_i) = -sqrt(|ln CCDF(|v|)|).
/// Returns (|v|, G) sample pairs (subsampled to ~200) and the moment-based
/// std-deviation σ of (vx ⊕ vy) (after subtracting the mean).
///
/// Note on what the "Gaussian reference" means here: the line drawn with
/// this σ is the analytic G(v) for a centred Gaussian with the same
/// variance as the data. If the empirical distribution actually were
/// Gaussian, the data would lie on the line. Visible deviations
/// (fat-tail bowing, concave bulk, etc.) are exactly the non-Gaussian
/// burst/active signal — that's the whole point of the Palmieri plot.
/// In Palmieri 2015 the reference σ is taken from a passive control run;
/// when only one (active) trajectory is plotted, the same-σ Gaussian is
/// the natural visual baseline.
pub fn compute_gvi(vx: &[f64], vy: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let mut v_abs: Vec<f64> = vx.iter().chain(vy.iter()).map(|v| v.abs()).filter(|v| v.is_finite()).collect();
    if v_abs.is_empty() { return (vec![], vec![], 0.0); }

    // Moment-based σ (centred). Robust enough now that the unwrap MIC
    // bug is fixed (no more 2L-jump outliers in pop_vx).
    let all: Vec<f64> = vx.iter().chain(vy.iter()).copied().filter(|v| v.is_finite()).collect();
    let n_all = all.len() as f64;
    let mean = all.iter().sum::<f64>() / n_all;
    let sigma = (all.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_all).sqrt().max(1e-30);

    v_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v_abs.len();
    let step = (n / 200).max(1);
    let mut centers = Vec::new();
    let mut gvi = Vec::new();
    let mut last_idx: Option<usize> = None;
    let push_idx = |i: usize, centers: &mut Vec<f64>, gvi: &mut Vec<f64>, last_idx: &mut Option<usize>| {
        if last_idx.map_or(false, |li| li == i) { return; }
        let v = v_abs[i];
        let ccdf = (n - i) as f64 / n as f64;
        if ccdf > 0.0 && ccdf < 1.0 {
            centers.push(v);
            gvi.push(-((-ccdf.ln()).sqrt()));
            *last_idx = Some(i);
        }
    };
    // Bulk: linear step through sorted samples for the body of the
    // distribution.
    for i in (0..n).step_by(step) {
        push_idx(i, &mut centers, &mut gvi, &mut last_idx);
    }
    // Tail: log-spaced indices so the high-|v| burst tail is shown
    // (linear step skips the last `step` samples = top 0.5% of values).
    // Halve the remaining-CCDF count each iteration: ccdf = k/n, k =
    // n-i, so i = n - k. k_start ~= step (resume where bulk ended).
    let mut k = step;
    while k > 1 {
        k = (k / 2).max(1);
        let i = n.saturating_sub(k);
        if i < n { push_idx(i, &mut centers, &mut gvi, &mut last_idx); }
        if k == 1 { break; }
    }
    (centers, gvi, sigma)
}

/// Abramowitz & Stegun 7.1.26 erfc, valid for |z| with reflection.
fn erfc_approx(z: f64) -> f64 {
    let z = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let val = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))))
        * (-z * z).exp();
    val.max(1e-15)
}

fn g_from_ccdf(ccdf: f64) -> f64 {
    let ccdf = ccdf.max(1e-15);
    -((-ccdf.ln()).sqrt())
}

/// Palmieri Eq. 5: mixture of Gaussian and burst (arcsine ⊛ Gaussian) CCDF.
fn palmieri_ccdf(v: f64, sigma: f64, zeta: f64, v_a: f64) -> f64 {
    let sqrt2 = std::f64::consts::SQRT_2;
    let ccdf_gauss = erfc_approx(v / (sigma * sqrt2));
    let ccdf_burst_plus = erfc_approx((v - v_a) / (sigma * sqrt2));
    let ccdf_burst_minus = erfc_approx((v + v_a) / (sigma * sqrt2));
    let ccdf_burst = (ccdf_burst_plus + ccdf_burst_minus) / 2.0;
    (1.0 - zeta) * ccdf_gauss + zeta * ccdf_burst
}

/// Render a G(v_i) panel containing any number of velocity series.
///
/// Each series gets its own marker (triangle/circle) in its own colour. A
/// Gaussian reference line and (optionally) a Palmieri Eq. 5 fit are
/// overlaid. Returns `Ok(())` once the chart is committed to `area`.
pub fn draw_gvi_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    series: &[GviSeries],
    opts: &GviPanelOpts,
) -> Result<()> {
    // Compute (|v|, G, σ) per series.
    let computed: Vec<(Vec<f64>, Vec<f64>, f64)> =
        series.iter().map(|s| compute_gvi(s.vx, s.vy)).collect();

    let x_max = opts.x_max;
    let y_min_data = computed.iter()
        .flat_map(|(_, g, _)| g.iter().copied())
        .fold(0.0_f64, f64::min);
    let (y_min, y_max) = opts.y_range
        .unwrap_or(((y_min_data - 0.3).min(-3.5), 0.5));

    let caption = match opts.panel_label {
        Some(c) => format!("({}) {}", c, opts.title),
        None => opts.title.clone(),
    };

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 16))
        .margin(8).x_label_area_size(30).y_label_area_size(50)
        .build_cartesian_2d(0.0..x_max, y_min..y_max)?;
    chart.configure_mesh()
        .x_desc("|v_i|").y_desc("G(v_i)")
        .x_label_style(("sans-serif", 14)).y_label_style(("sans-serif", 14))
        .light_line_style(TRANSPARENT)
        .bold_line_style(RGBAColor(200, 200, 200, 0.3))
        .draw()?;

    // Series scatter
    for (s, (xs, gs, sigma)) in series.iter().zip(computed.iter()) {
        let label = format!("{} (σ={:.4})", s.label, sigma);
        let color = s.color;
        match s.marker {
            GviMarker::Triangle => {
                chart.draw_series(xs.iter().zip(gs.iter()).map(|(&x, &y)| {
                    TriangleMarker::new((x, y), 4, color.filled())
                }))?.label(label)
                    .legend(move |(x, y)| TriangleMarker::new((x + 6, y), 4, color.filled()));
            }
            GviMarker::Circle => {
                chart.draw_series(xs.iter().zip(gs.iter()).map(|(&x, &y)| {
                    Circle::new((x, y), 3, color.filled())
                }))?.label(label)
                    .legend(move |(x, y)| Circle::new((x + 6, y), 3, color.filled()));
            }
        }
    }

    // Gaussian reference: use moment-based σ of the first series. This
    // matches the data's second moment, which is the textbook "Gaussian
    // with the same variance" baseline (Palmieri-style). Any visible
    // deviation between this line and the data is the non-Gaussian /
    // active signal — fitting σ to the curve shape is misleading because
    // no Gaussian σ actually matches a non-Gaussian distribution.
    let ref_sigma = opts.gaussian_ref_sigma
        .or_else(|| computed.first().map(|(_, _, s)| *s))
        .unwrap_or(1.0)
        .max(1e-6);
    chart.draw_series(LineSeries::new(
        (1..200).map(|i| {
            let v = i as f64 * x_max / 200.0;
            let ccdf = erfc_approx(v / (ref_sigma * std::f64::consts::SQRT_2));
            (v, g_from_ccdf(ccdf))
        }),
        BLACK.mix(0.5).stroke_width(2),
    ))?.label(format!("Gaussian (σ={:.4})", ref_sigma))
        .legend(|(x, y)| PathElement::new(
            vec![(x, y), (x + 15, y)],
            BLACK.mix(0.5).stroke_width(2),
        ));

    // Optional σ-sweep: draw thin coloured Gaussian curves at user-supplied
    // σ values for visual exploration.
    let sweep_palette = [
        RGBAColor(220, 50, 50, 0.9),
        RGBAColor(50, 130, 220, 0.9),
        RGBAColor(50, 170, 80, 0.9),
        RGBAColor(200, 130, 30, 0.9),
        RGBAColor(150, 60, 200, 0.9),
        RGBAColor(20, 160, 180, 0.9),
    ];
    for (i, &s) in opts.gaussian_sigma_sweep.iter().enumerate() {
        if s <= 0.0 { continue; }
        let color = sweep_palette[i % sweep_palette.len()];
        chart.draw_series(LineSeries::new(
            (1..200).map(|k| {
                let v = k as f64 * x_max / 200.0;
                let ccdf = erfc_approx(v / (s * std::f64::consts::SQRT_2));
                (v, g_from_ccdf(ccdf))
            }),
            color.stroke_width(1),
        ))?.label(format!("σ={:.4}", s))
            .legend(move |(x, y)| PathElement::new(
                vec![(x, y), (x + 15, y)],
                color.stroke_width(1),
            ));
    }

    // Optional Palmieri Eq. 5 fit. If `palmieri_fit_index` is set, fit
    // Eq.5 to that series' G(v) curve and to *every* later series too,
    // so that e.g. cell 0 gets its own fit overlay with its own ζ
    // (cell 0 has its own intrinsic burst statistics that differ from
    // the population). Each fit is drawn in the matching series colour.
    if let Some(idx) = opts.palmieri_fit_index {
        for (s_i, ((xs, gs, _), s)) in computed.iter().zip(series.iter()).enumerate().skip(idx) {
            // Trapezoidal weighting (≈ ∫ dv) so dense bulk samples near
            // v=0 don't swamp the sparse tail.
            let v_min = opts.palmieri_fit_min_v.unwrap_or(0.0);
            let mut best_zeta = 0.05_f64;
            let mut best_sse = f64::INFINITY;
            for zi in 1..200 {
                let zeta_try = zi as f64 * 0.0025;
                let mut sse = 0.0;
                let mut count = 0;
                for (k, (&v, &g_data)) in xs.iter().zip(gs.iter()).enumerate() {
                    if v >= v_min && v < x_max {
                        let v_prev = if k > 0 { xs[k - 1] } else { v };
                        let v_next = if k + 1 < xs.len() { xs[k + 1] } else { v };
                        let dv = ((v_next - v_prev) * 0.5).max(0.0);
                        let ccdf = palmieri_ccdf(v, ref_sigma, zeta_try, opts.v_a);
                        let g_model = g_from_ccdf(ccdf);
                        sse += (g_data - g_model).powi(2) * dv;
                        count += 1;
                    }
                }
                if count > 0 && sse < best_sse {
                    best_sse = sse;
                    best_zeta = zeta_try;
                }
            }

            let palmieri_curve: Vec<(f64, f64)> = (1..200).filter_map(|i| {
                let v = i as f64 * x_max / 200.0;
                let ccdf = palmieri_ccdf(v, ref_sigma, best_zeta, opts.v_a);
                if ccdf > 1e-15 { Some((v, g_from_ccdf(ccdf))) } else { None }
            }).collect();
            // First fit (population) gets the standard pinky-red colour;
            // additional fits inherit their series colour.
            let eq5_color = if s_i == idx {
                RGBAColor(200, 100, 100, 0.8)
            } else {
                s.color
            };
            chart.draw_series(LineSeries::new(palmieri_curve, eq5_color.stroke_width(2)))?
                .label(format!("Eq.5 [{}] ζ={:.1}%", s.label, best_zeta * 100.0))
                .legend(move |(x, y)| PathElement::new(
                    vec![(x, y), (x + 15, y)],
                    eq5_color.stroke_width(2),
                ));
        }
    }

    chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8)).border_style(BLACK.mix(0.3))
        .label_font(("sans-serif", 9)).draw()?;
    chart.plotting_area().draw(&PathElement::new(
        vec![(0.0, y_max), (x_max, y_max), (x_max, y_min)],
        BLACK.mix(0.5).stroke_width(1),
    ))?;
    Ok(())
}
