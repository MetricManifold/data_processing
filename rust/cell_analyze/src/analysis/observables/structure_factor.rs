//! Static structure factor S(q), angularly averaged.
//!
//! S(q) = ⟨ |ρ(q)|² / N ⟩ where ρ(q) = Σⱼ exp(i q·rⱼ),
//! averaged over N_frames frames from the second half of the trajectory
//! and over a discrete grid of q-vectors with |q| ≤ q_max.
//!
//! Returns S(q) in radial bins, plus q* = location of the first peak
//! (used downstream by self_intermediate_scattering).
//!
//! Ported from legacy `observables.rs::structure_factor`.

use anyhow::Result;
use serde::Serialize;
use std::f64::consts::PI;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct StructureFactor {
    pub n_bins: usize,
    pub n_frames: usize,
}

impl Default for StructureFactor {
    fn default() -> Self { Self { n_bins: 80, n_frames: 8 } }
}

#[derive(Clone, Debug, Serialize)]
pub struct StructureFactorOutput {
    pub q_bins: Vec<f64>,
    pub s_q: Vec<f64>,
    pub q_star: f64,
}

impl Observable for StructureFactor {
    type Output = StructureFactorOutput;

    fn id(&self) -> &'static str { "structure_factor" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let n_bins = self.n_bins;
        let n_frames = self.n_frames;
        let lx = pos.lx;
        let ly = pos.ly;
        let l_min = lx.min(ly);
        let q_max = 2.0 * PI * 20.0 / l_min;
        let dq = q_max / n_bins as f64;
        let dqx = 2.0 * PI / lx;
        let dqy = 2.0 * PI / ly;
        let nx_max = (q_max / dqx).ceil() as i32;
        let ny_max = (q_max / dqy).ceil() as i32;

        let mut s_sum = vec![0.0_f64; n_bins];
        let mut counts = vec![0_u64; n_bins];

        let n_sq = n_frames.min(pos.n_times);
        let start = pos.n_times / 2;
        let step = if n_sq > 1 { (pos.n_times - start).max(1) / n_sq } else { 1 };

        let mut frames_used = 0;
        let mut t_idx = start;
        while t_idx < pos.n_times && frames_used < n_sq {
            let n = pos.n_cells;
            let wrapped: Vec<[f64; 2]> = (0..n).map(|i| pos.wrapped(t_idx, i)).collect();
            for nx in -nx_max..=nx_max {
                let qx = nx as f64 * dqx;
                for ny in -ny_max..=ny_max {
                    if nx == 0 && ny == 0 { continue; }
                    let qy = ny as f64 * dqy;
                    let q_mag = (qx * qx + qy * qy).sqrt();
                    if q_mag > q_max { continue; }
                    let b = (q_mag / dq) as usize;
                    if b >= n_bins { continue; }
                    let mut rho_re = 0.0_f64;
                    let mut rho_im = 0.0_f64;
                    for p in &wrapped {
                        let phase = qx * p[0] + qy * p[1];
                        rho_re += phase.cos();
                        rho_im += phase.sin();
                    }
                    s_sum[b] += (rho_re * rho_re + rho_im * rho_im) / n as f64;
                    counts[b] += 1;
                }
            }
            frames_used += 1;
            t_idx += step.max(1);
        }
        let q_bins: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * dq).collect();
        let mut s_q = vec![0.0; n_bins];
        for i in 0..n_bins {
            if counts[i] > 0 { s_q[i] = s_sum[i] / counts[i] as f64; }
        }
        let q_min = 0.02;
        let q_star = q_bins.iter().zip(s_q.iter())
            .filter(|(&q, _)| q > q_min)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0.0, |(&q, _)| q);
        Ok(StructureFactorOutput { q_bins, s_q, q_star })
    }
}
