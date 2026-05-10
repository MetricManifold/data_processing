//! Voronoi shape index q = P/√A from a simple geometric construction.
//!
//! For each cell i, finds neighbors within 4R and builds the Voronoi
//! polygon by connecting circumcenters of consecutive (i, j, k)
//! triangles. Computes shoelace area and perimeter, returns q = P/√A.
//!
//! Ported from legacy `observables.rs::compute_voronoi_shape`.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::observable::{Context, Observable, Requirements};

pub struct VoronoiShape;

#[derive(Clone, Debug, Serialize)]
pub struct VoronoiShapeOutput {
    pub q_mean: f64,
    pub q_std: f64,
    pub q_per_cell: Vec<f64>,
}

impl Observable for VoronoiShape {
    type Output = VoronoiShapeOutput;

    fn id(&self) -> &'static str { "voronoi_shape" }
    fn requires(&self) -> Requirements { Requirements::POSITIONS }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let cell_radius = ctx.params.cell_radius;
        let n_cells = pos.n_cells;
        let n_times = pos.n_times;
        let cutoff = 4.0 * cell_radius;
        let cutoff2 = cutoff * cutoff;
        let lx = pos.lx;
        let ly = pos.ly;
        let mut q_accum = vec![0.0_f64; n_cells];
        let mut q_count = vec![0_u32; n_cells];

        for t in 0..n_times {
            let wx: Vec<f64> = (0..n_cells).map(|i| pos.positions[t][i][0].rem_euclid(lx)).collect();
            let wy: Vec<f64> = (0..n_cells).map(|i| pos.positions[t][i][1].rem_euclid(ly)).collect();
            for i in 0..n_cells {
                let mut nbrs: Vec<(f64, f64, f64)> = Vec::new();
                for j in 0..n_cells {
                    if j == i { continue; }
                    let mut dx = wx[j] - wx[i];
                    let mut dy = wy[j] - wy[i];
                    if dx > lx * 0.5 { dx -= lx; }
                    if dx < -lx * 0.5 { dx += lx; }
                    if dy > ly * 0.5 { dy -= ly; }
                    if dy < -ly * 0.5 { dy += ly; }
                    let r2 = dx * dx + dy * dy;
                    if r2 < cutoff2 {
                        nbrs.push((dy.atan2(dx), dx, dy));
                    }
                }
                if nbrs.len() < 3 { continue; }
                nbrs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let nn = nbrs.len();
                let mut verts: Vec<(f64, f64)> = Vec::with_capacity(nn);
                for idx in 0..nn {
                    let (_, ax, ay) = nbrs[idx];
                    let (_, bx, by) = nbrs[(idx + 1) % nn];
                    let d = 2.0 * (ax * by - ay * bx);
                    if d.abs() < 1e-12 {
                        verts.push(((ax + bx) * 0.25, (ay + by) * 0.25));
                    } else {
                        let a2 = ax * ax + ay * ay;
                        let b2 = bx * bx + by * by;
                        let cx = (a2 * by - b2 * ay) / d;
                        let cy = (bx * a2 - ax * b2) / d;
                        verts.push((cx, cy));
                    }
                }
                let nv = verts.len();
                let mut perim = 0.0;
                let mut area = 0.0;
                for vi in 0..nv {
                    let (x0, y0) = verts[vi];
                    let (x1, y1) = verts[(vi + 1) % nv];
                    perim += ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
                    area += x0 * y1 - x1 * y0;
                }
                area = area.abs() * 0.5;
                if area > 1e-10 {
                    let q = perim / area.sqrt();
                    q_accum[i] += q;
                    q_count[i] += 1;
                }
            }
        }
        let q_per_cell: Vec<f64> = (0..n_cells)
            .map(|i| if q_count[i] > 0 { q_accum[i] / q_count[i] as f64 } else { 0.0 })
            .collect();
        let valid: Vec<f64> = q_per_cell.iter().filter(|&&v| v > 0.0).copied().collect();
        let q_mean = if valid.is_empty() { 0.0 } else { valid.iter().sum::<f64>() / valid.len() as f64 };
        let q_var = if valid.len() < 2 { 0.0 } else {
            valid.iter().map(|&v| (v - q_mean).powi(2)).sum::<f64>() / valid.len() as f64
        };
        Ok(VoronoiShapeOutput {
            q_mean, q_std: q_var.sqrt(), q_per_cell,
        })
    }
}
