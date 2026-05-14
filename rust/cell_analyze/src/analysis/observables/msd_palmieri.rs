//! Palmieri-style MSD/Δt curve over the 0..8τ window plus the
//! D_eff = MSD(8τ)/(4·8τ) read-off.
//!
//! Differs from the general [`crate::analysis::observables::msd::Msd`]
//! observable in two ways:
//!   1. lags are sampled densely in the visible 0..8τ window (200
//!      points), so the MSD/Δt curve is smooth on the panel,
//!   2. the 8τ lag is always included so D_eff is read from the same
//!      time point used by Palmieri (2015).

use anyhow::Result;
use rayon::prelude::*;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct MsdPalmieri;

#[derive(Clone, Debug, Serialize)]
pub struct MsdPalmieriOutput {
    /// Lag time in τ units for each sampled point.
    pub lag_tau: Vec<f64>,
    /// MSD(Δt)/Δt for the tagged cell.
    pub msd_t_cell: Vec<f64>,
    /// MSD(Δt)/Δt for the population.
    pub msd_t_pop: Vec<f64>,
    /// D_eff for the tagged cell evaluated at the 8τ lag.
    pub d_eff_cell: f64,
    /// D_eff for the population evaluated at the 8τ lag.
    pub d_eff_pop: f64,
}

impl Observable for MsdPalmieri {
    type Output = MsdPalmieriOutput;

    fn id(&self) -> &'static str {
        "msd_palmieri"
    }

    fn requires(&self) -> Requirements {
        Requirements::POSITIONS
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let tau = ctx.params.tau;
        let n = pos.n_times;
        if n < 4 {
            return Ok(MsdPalmieriOutput {
                lag_tau: vec![],
                msd_t_cell: vec![],
                msd_t_pop: vec![],
                d_eff_cell: f64::NAN,
                d_eff_pop: f64::NAN,
            });
        }
        let dt = pos.times[1] - pos.times[0];
        let tagged: u32 = ctx.params.tagged_cells.first().copied().unwrap_or(0);
        let cell_idx = pos
            .cell_ids
            .iter()
            .position(|&c| c == tagged)
            .unwrap_or(0);

        let max_lag = n / 2;
        let lag_8tau = ((8.0 * tau / dt).round() as usize).min(max_lag);
        if lag_8tau < 2 {
            return Ok(MsdPalmieriOutput {
                lag_tau: vec![],
                msd_t_cell: vec![],
                msd_t_pop: vec![],
                d_eff_cell: f64::NAN,
                d_eff_pop: f64::NAN,
            });
        }

        // 200-point dense sampling of 0..8τ, plus the first 10 lags
        // (so the curve has good resolution near the origin) and 8τ
        // itself.
        let n_visible = 200usize;
        let stride = (lag_8tau / n_visible).max(1);
        let mut lags: std::collections::BTreeSet<usize> = (1..=10.min(lag_8tau)).collect();
        let mut l = stride;
        while l <= lag_8tau {
            lags.insert(l);
            l += stride;
        }
        lags.insert(lag_8tau);
        let lags: Vec<usize> = lags.into_iter().collect();

        // Materialize positions as a contiguous (n_times * n_cells * 2)
        // row-major buffer once, instead of chasing the Vec<Vec<[f64;3]>>
        // double indirection inside the triple-nested kernel below. For
        // the production case (n_times ~ 20k, n_cells = 100) this is
        // ~32 MB and turns the hot inner loop into a sequential f64 read.
        let n_cells = pos.n_cells;
        let stride_t = n_cells * 2;
        let mut flat: Vec<f64> = Vec::with_capacity(n * stride_t);
        for t in 0..n {
            for ci in 0..n_cells {
                let p = &pos.positions[t][ci];
                flat.push(p[0]);
                flat.push(p[1]);
            }
        }
        // Tagged-cell positions, packed contiguously across time.
        let mut tagged_x: Vec<f64> = Vec::with_capacity(n);
        let mut tagged_y: Vec<f64> = Vec::with_capacity(n);
        for t in 0..n {
            tagged_x.push(pos.positions[t][cell_idx][0]);
            tagged_y.push(pos.positions[t][cell_idx][1]);
        }

        // Compute (cell_sum, pop_sum) per lag in parallel. Each lag is
        // independent so rayon's `par_iter` scales linearly with cores.
        let inv_n_cells = 1.0 / n_cells as f64;
        let per_lag: Vec<(usize, f64, f64, usize)> = lags
            .par_iter()
            .filter_map(|&lag| {
                let n_origins = n.saturating_sub(lag);
                if n_origins < 2 {
                    return None;
                }
                let mut cell_sum = 0.0f64;
                let mut pop_sum = 0.0f64;
                for t0 in 0..n_origins {
                    let ti = t0 + lag;
                    let dx = tagged_x[ti] - tagged_x[t0];
                    let dy = tagged_y[ti] - tagged_y[t0];
                    cell_sum += dx * dx + dy * dy;

                    // Tight contiguous-memory inner loop. Two slices of
                    // `flat` cover all cells at ti and t0; iterate pairs
                    // and accumulate squared displacement.
                    let base_i = ti * stride_t;
                    let base_0 = t0 * stride_t;
                    let row_i = &flat[base_i..base_i + stride_t];
                    let row_0 = &flat[base_0..base_0 + stride_t];
                    let mut s = 0.0f64;
                    // Manual chunked iteration so the compiler can
                    // vectorize. Each cell has two consecutive doubles.
                    let mut k = 0;
                    while k < stride_t {
                        let ddx = row_i[k] - row_0[k];
                        let ddy = row_i[k + 1] - row_0[k + 1];
                        s += ddx * ddx + ddy * ddy;
                        k += 2;
                    }
                    pop_sum += s * inv_n_cells;
                }
                Some((lag, cell_sum, pop_sum, n_origins))
            })
            .collect();

        let mut lag_tau = Vec::with_capacity(per_lag.len());
        let mut msd_t_cell = Vec::with_capacity(per_lag.len());
        let mut msd_t_pop = Vec::with_capacity(per_lag.len());
        let mut d_eff_cell = f64::NAN;
        let mut d_eff_pop = f64::NAN;
        for (lag, cell_sum, pop_sum, n_origins) in per_lag {
            let lag_time = lag as f64 * dt;
            let inv_no = 1.0 / n_origins as f64;
            let msd_cell = cell_sum * inv_no;
            let msd_p = pop_sum * inv_no;
            lag_tau.push(lag_time / tau);
            msd_t_cell.push(msd_cell / lag_time);
            msd_t_pop.push(msd_p / lag_time);
            if lag == lag_8tau {
                d_eff_cell = msd_cell / (4.0 * lag_time);
                d_eff_pop = msd_p / (4.0 * lag_time);
            }
        }

        Ok(MsdPalmieriOutput {
            lag_tau,
            msd_t_cell,
            msd_t_pop,
            d_eff_cell,
            d_eff_pop,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::io::UnwrappedPositions;
    use crate::analysis::observable::{Context, RunParams};
    use std::sync::Arc;

    fn ctx_with_positions(positions: Vec<Vec<[f64; 3]>>, times: Vec<f64>, n_cells: usize, tau: f64) -> Context {
        let n_times = times.len();
        let pos = UnwrappedPositions {
            times,
            cell_ids: (0..n_cells as u32).collect(),
            positions,
            lx: 1000.0,
            ly: 1000.0,
            lz: 0.0,
            dim: 2,
            n_cells,
            n_times,
            inherent_v_a: vec![],
        };
        Context {
            positions: Arc::new(pos),
            trajectory: None,
            checkpoint: None,
            params: RunParams { tau, ..Default::default() },
        }
    }

    /// Diffusive ballistic test: one cell moving at constant velocity v=1
    /// in x direction. MSD(Δt) = (v·Δt)^2 = Δt^2, so MSD/Δt = Δt linearly.
    /// We verify the D_eff readoff at 8τ matches MSD(8τ)/(4·8τ).
    #[test]
    fn ballistic_one_cell() {
        let tau = 10.0;
        let dt = 0.1;
        let n_times = 2000;
        let times: Vec<f64> = (0..n_times).map(|i| i as f64 * dt).collect();
        let positions: Vec<Vec<[f64; 3]>> = (0..n_times)
            .map(|i| vec![[i as f64 * dt, 0.0, 0.0]])
            .collect();
        let ctx = ctx_with_positions(positions, times, 1, tau);
        let out = MsdPalmieri.compute(&ctx).unwrap();
        // 8τ = 80 sim time, dt=0.1, lag_8tau = 800 idx.
        // MSD(8τ) = 80^2 = 6400. D_eff = 6400 / (4·80) = 20.
        assert!((out.d_eff_cell - 20.0).abs() < 0.01, "D_eff_cell = {}", out.d_eff_cell);
        assert!((out.d_eff_pop - 20.0).abs() < 0.01, "D_eff_pop = {}", out.d_eff_pop);
        // MSD/Δt should grow linearly with Δt. Check first and last lag.
        let last = out.msd_t_cell.last().copied().unwrap();
        assert!((last - 80.0).abs() < 0.5, "msd_t_cell at 8τ = {}", last);
    }

    /// Population MSD: two cells with opposite velocities cancel out
    /// to zero net population displacement... no wait, MSD is squared so
    /// they add. v=1 vs v=-1 → both contribute v²Δt² each → population
    /// average is the same as one-cell. Same D_eff_pop expected.
    #[test]
    fn population_two_cells_same_speed() {
        let tau = 10.0;
        let dt = 0.1;
        let n_times = 2000;
        let times: Vec<f64> = (0..n_times).map(|i| i as f64 * dt).collect();
        let positions: Vec<Vec<[f64; 3]>> = (0..n_times)
            .map(|i| vec![
                [i as f64 * dt, 0.0, 0.0],
                [-(i as f64 * dt), 0.0, 0.0],
            ])
            .collect();
        let ctx = ctx_with_positions(positions, times, 2, tau);
        let out = MsdPalmieri.compute(&ctx).unwrap();
        // Both cells: MSD = (1·Δt)² for each, mean = same. D_eff_pop = 20.
        assert!((out.d_eff_pop - 20.0).abs() < 0.01, "D_eff_pop = {}", out.d_eff_pop);
    }
    /// Smoke benchmark mirroring the production hang case: N=100 cells,
    /// 19000 frames, single run. Pre-fix this took ~minutes; post-fix
    /// it should be a few seconds at most. We assert it finishes in
    /// under 30 seconds so a regression surfaces here, not on the cluster.
    #[test]
    fn bench_production_size() {
        let tau = 10000.0;
        let dt = 104.01;
        let n_cells = 100usize;
        let n_times = 19000usize;
        let times: Vec<f64> = (0..n_times).map(|i| i as f64 * dt).collect();
        let positions: Vec<Vec<[f64; 3]>> = (0..n_times)
            .map(|t| {
                let mut row = Vec::with_capacity(n_cells);
                for ci in 0..n_cells {
                    // Light random-walk-ish placement that doesn't matter
                    // numerically for the perf test.
                    let phase = (t as f64) * 0.01 + (ci as f64) * 0.1;
                    row.push([phase.sin() * 100.0, phase.cos() * 100.0, 0.0]);
                }
                row
            })
            .collect();
        let ctx = ctx_with_positions(positions, times, n_cells, tau);
        let t0 = std::time::Instant::now();
        let out = MsdPalmieri.compute(&ctx).unwrap();
        let elapsed = t0.elapsed();
        eprintln!("msd_palmieri N=100 T=19000 took {:?}, {} lags", elapsed, out.lag_tau.len());
        assert!(out.lag_tau.len() > 100, "expected dense lag sampling");
        assert!(elapsed.as_secs() < 30,
                "msd_palmieri took {:?} — perf regression", elapsed);
       }
}

