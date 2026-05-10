//! Hexatic order parameter ψ₆ + g₆(r).
//!
//! ψ₆ᵢ = |1/nᵢ Σⱼ exp(6 i·θᵢⱼ)| over near-neighbors (within 3R).
//! Time-averaged per-cell, plus the orientational correlation g₆(r).
//!
//! Ported from legacy `observables.rs::compute_hexatic_order`.

use anyhow::Result;
use serde::Serialize;
use std::f64::consts::PI;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct HexaticOrder;

#[derive(Clone, Debug, Serialize)]
pub struct HexaticOrderOutput {
    pub psi6_mean: f64,
    pub psi6_std: f64,
    pub psi6_per_cell: Vec<f64>,
    pub g6_r: Vec<f64>,
    pub g6_values: Vec<f64>,
}

impl Observable for HexaticOrder {
    type Output = HexaticOrderOutput;

    fn id(&self) -> &'static str { "hexatic_order" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let cell_radius = ctx.params.cell_radius;
        let n_cells = pos.n_cells;
        let n_times = pos.n_times;
        let cutoff = 3.0 * cell_radius;
        let cutoff2 = cutoff * cutoff;
        let lx = pos.lx;
        let ly = pos.ly;
        let mut psi6_accum = vec![0.0_f64; n_cells];
        let n_bins = 40;
        let bin_width = cutoff * 2.0 / n_bins as f64;
        let mut g6_sum = vec![0.0_f64; n_bins];
        let mut g6_count = vec![0_u64; n_bins];

        for t in 0..n_times {
            let wx: Vec<f64> = (0..n_cells).map(|i| pos.positions[t][i][0].rem_euclid(lx)).collect();
            let wy: Vec<f64> = (0..n_cells).map(|i| pos.positions[t][i][1].rem_euclid(ly)).collect();
            let mut psi6_re = vec![0.0_f64; n_cells];
            let mut psi6_im = vec![0.0_f64; n_cells];
            let mut n_nbr = vec![0_u32; n_cells];

            for i in 0..n_cells {
                for j in (i + 1)..n_cells {
                    let mut dx = wx[j] - wx[i];
                    let mut dy = wy[j] - wy[i];
                    if dx > lx * 0.5 { dx -= lx; }
                    if dx < -lx * 0.5 { dx += lx; }
                    if dy > ly * 0.5 { dy -= ly; }
                    if dy < -ly * 0.5 { dy += ly; }
                    let r2 = dx * dx + dy * dy;
                    if r2 < cutoff2 && r2 > 1e-10 {
                        let theta = dy.atan2(dx);
                        let c6 = (6.0 * theta).cos();
                        let s6 = (6.0 * theta).sin();
                        psi6_re[i] += c6; psi6_im[i] += s6; n_nbr[i] += 1;
                        let c6r = (6.0 * (theta + PI)).cos();
                        let s6r = (6.0 * (theta + PI)).sin();
                        psi6_re[j] += c6r; psi6_im[j] += s6r; n_nbr[j] += 1;
                    }
                }
            }
            for i in 0..n_cells {
                if n_nbr[i] > 0 {
                    let n = n_nbr[i] as f64;
                    psi6_re[i] /= n;
                    psi6_im[i] /= n;
                }
                let mag = (psi6_re[i] * psi6_re[i] + psi6_im[i] * psi6_im[i]).sqrt();
                psi6_accum[i] += mag;
            }
            for i in 0..n_cells {
                for j in (i + 1)..n_cells {
                    let mut dx = wx[j] - wx[i];
                    let mut dy = wy[j] - wy[i];
                    if dx > lx * 0.5 { dx -= lx; }
                    if dx < -lx * 0.5 { dx += lx; }
                    if dy > ly * 0.5 { dy -= ly; }
                    if dy < -ly * 0.5 { dy += ly; }
                    let r = (dx * dx + dy * dy).sqrt();
                    let bin = (r / bin_width) as usize;
                    if bin < n_bins {
                        let dot = psi6_re[i] * psi6_re[j] + psi6_im[i] * psi6_im[j];
                        g6_sum[bin] += dot;
                        g6_count[bin] += 1;
                    }
                }
            }
        }
        let nt = n_times as f64;
        let psi6_per_cell: Vec<f64> = psi6_accum.iter().map(|&v| v / nt).collect();
        let psi6_mean = psi6_per_cell.iter().sum::<f64>() / n_cells.max(1) as f64;
        let psi6_var = psi6_per_cell.iter().map(|&v| (v - psi6_mean).powi(2)).sum::<f64>()
            / n_cells.max(1) as f64;
        let psi6_std = psi6_var.sqrt();
        let g6_r: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * bin_width).collect();
        let g6_values: Vec<f64> = (0..n_bins)
            .map(|i| if g6_count[i] > 0 { g6_sum[i] / g6_count[i] as f64 } else { 0.0 })
            .collect();
        Ok(HexaticOrderOutput {
            psi6_mean, psi6_std, psi6_per_cell, g6_r, g6_values,
        })
    }
}
