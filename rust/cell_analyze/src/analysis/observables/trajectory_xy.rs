//! Per-cell XY trajectory paths over time. Outputs the (x, y) positions
//! of one or more tagged cells across the trajectory window, optionally
//! decimated to bound JSON size, and optionally wrapped back into the
//! periodic box for snapshot-style rendering.
//!
//! Used by the `trajectory_xy` panels (single / pair / overlay) to draw
//! cell paths. Reusing the existing `UnwrappedPositions` (already loaded
//! for every run) keeps this nearly free — the only cost is the per-cell
//! Vec stored in the bag.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

/// One cell's path through the trajectory window.
#[derive(Clone, Debug, Serialize)]
pub struct CellPath {
    pub cell_id: u32,
    /// Times (sim units) at the sampled points.
    pub t: Vec<f64>,
    /// Unwrapped (x, y) positions. Length == `t.len()`. Z is dropped;
    /// 2D-only panel.
    pub xy: Vec<[f64; 2]>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrajectoryXyOutput {
    /// Domain box. Echoed for panels that draw the box outline.
    pub lx: f64,
    pub ly: f64,
    /// One entry per requested cell (in the order they were requested).
    /// Cells whose id is missing from the trajectory are silently skipped
    /// so the panel doesn't error out on stale TOML.
    pub paths: Vec<CellPath>,
    /// True if the original trajectory was decimated to fit `max_points`.
    pub decimated: bool,
    /// Stride used (1 = no decimation).
    pub stride: usize,
}

/// `trajectory_xy` observable. Default: cell 0 only, up to 2000 points.
pub struct TrajectoryXy {
    /// Cells to record. Defaults to `ctx.params.tagged_cells` when empty.
    pub cells: Vec<u32>,
    /// Cap on points per cell. Stride is `ceil(n_times / max_points)`.
    /// 0 means no decimation.
    pub max_points: usize,
}

impl Default for TrajectoryXy {
    fn default() -> Self {
        Self {
            cells: Vec::new(),
            max_points: 2000,
        }
    }
}

impl Observable for TrajectoryXy {
    type Output = TrajectoryXyOutput;

    fn id(&self) -> &'static str {
        "trajectory_xy"
    }

    fn requires(&self) -> Requirements {
        Requirements::POSITIONS
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let lx = pos.lx;
        let ly = pos.ly;

        // Resolve which cells to record: explicit `cells` overrides the
        // study's tagged_cells; otherwise default to all tagged.
        let requested: Vec<u32> = if self.cells.is_empty() {
            ctx.params.tagged_cells.clone()
        } else {
            self.cells.clone()
        };

        // Decimation stride. With max_points=0 we keep every frame.
        let n_times = pos.n_times;
        let stride: usize = if self.max_points == 0 || n_times <= self.max_points {
            1
        } else {
            (n_times + self.max_points - 1) / self.max_points
        };
        let decimated = stride > 1;

        // Map cell_id -> column index in UnwrappedPositions.positions.
        // The trajectory loader sorts cell_ids, so we look up linearly
        // (N requested cells is tiny — usually 1 or 2).
        let mut paths: Vec<CellPath> = Vec::with_capacity(requested.len());
        for &cid in &requested {
            let Some(col) = pos.cell_ids.iter().position(|&c| c == cid) else {
                eprintln!(
                    "[trajectory_xy] cell {} not in trajectory (have {} cells); skipping",
                    cid, pos.n_cells
                );
                continue;
            };
            let cap = (n_times + stride - 1) / stride;
            let mut t = Vec::with_capacity(cap);
            let mut xy = Vec::with_capacity(cap);
            let mut i = 0usize;
            while i < n_times {
                t.push(pos.times[i]);
                let p = pos.positions[i][col];
                xy.push([p[0], p[1]]);
                i += stride;
            }
            paths.push(CellPath { cell_id: cid, t, xy });
        }

        Ok(TrajectoryXyOutput {
            lx,
            ly,
            paths,
            decimated,
            stride,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::io::UnwrappedPositions;
    use crate::analysis::observable::{Context, RunParams};
    use std::sync::Arc;

    fn synthetic_pos(n_cells: usize, n_times: usize) -> UnwrappedPositions {
        let cell_ids: Vec<u32> = (0..n_cells as u32).collect();
        let times: Vec<f64> = (0..n_times).map(|t| t as f64).collect();
        let positions: Vec<Vec<[f64; 3]>> = (0..n_times)
            .map(|t| {
                (0..n_cells)
                    .map(|i| [i as f64 * 10.0 + t as f64, t as f64, 0.0])
                    .collect()
            })
            .collect();
        UnwrappedPositions {
            times,
            cell_ids,
            positions,
            lx: 1000.0,
            ly: 1000.0,
            lz: 0.0,
            n_cells,
            n_times,
            dim: 2,
            inherent_v_a: vec![0.0; n_cells],
        }
    }

    fn ctx(pos: UnwrappedPositions, tagged: Vec<u32>) -> Context {
        Context {
            positions: Arc::new(pos),
            trajectory: None,
            checkpoint: None,
            params: RunParams {
                tau: 10000.0,
                cell_radius: 49.0,
                v_a: 0.01,
                tagged_cells: tagged,
                soft_cells: vec![],
            },
        }
    }

    #[test]
    fn default_records_tagged_cell() {
        let p = synthetic_pos(3, 100);
        let c = ctx(p, vec![1]);
        let out = TrajectoryXy::default().compute(&c).unwrap();
        assert_eq!(out.paths.len(), 1);
        assert_eq!(out.paths[0].cell_id, 1);
        assert_eq!(out.paths[0].t.len(), 100);
        assert_eq!(out.paths[0].xy[0], [10.0, 0.0]);
        assert_eq!(out.paths[0].xy[99], [109.0, 99.0]);
        assert!(!out.decimated);
        assert_eq!(out.stride, 1);
    }

    #[test]
    fn explicit_cells_override_tagged() {
        let p = synthetic_pos(3, 50);
        let c = ctx(p, vec![0]);
        let out = TrajectoryXy {
            cells: vec![0, 2],
            max_points: 0,
        }
        .compute(&c)
        .unwrap();
        assert_eq!(out.paths.len(), 2);
        assert_eq!(out.paths[0].cell_id, 0);
        assert_eq!(out.paths[1].cell_id, 2);
    }

    #[test]
    fn decimation_caps_point_count() {
        let p = synthetic_pos(1, 10000);
        let c = ctx(p, vec![0]);
        let out = TrajectoryXy {
            cells: vec![],
            max_points: 500,
        }
        .compute(&c)
        .unwrap();
        assert!(out.decimated);
        assert_eq!(out.stride, 20);
        assert!(out.paths[0].t.len() <= 500);
        // First sample is t=0, last sample at stride*(k-1) for some k.
        assert_eq!(out.paths[0].t[0], 0.0);
    }

    #[test]
    fn missing_cell_is_skipped() {
        let p = synthetic_pos(2, 20);
        let c = ctx(p, vec![0]);
        let out = TrajectoryXy {
            cells: vec![0, 99],
            max_points: 0,
        }
        .compute(&c)
        .unwrap();
        assert_eq!(out.paths.len(), 1);
        assert_eq!(out.paths[0].cell_id, 0);
    }
}
