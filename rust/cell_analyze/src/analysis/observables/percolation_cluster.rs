//! Percolation cluster analysis. Phase 3B — Claim 2 (cancer-cell
//! fluidisation as a percolation transition).
//!
//! Per trajectory:
//!   1. compute per-cell D_eff via per_cell_diffusion,
//!   2. flag mobile cells (D_eff > threshold OR percentile cut),
//!   3. build adjacency graph among mobile cells (Voronoi neighbours when
//!      voronoi_shape exposes them, else distance cutoff),
//!   4. find connected components via union-find,
//!   5. emit order parameters: S_max, P_inf, chi, size histogram.
//!
//! See `transfer/percolation_observable_SPEC.md` on nibi for the full
//! spec. v1 ships distance-cutoff adjacency; the Voronoi path will land
//! once `voronoi_shape` exposes its neighbour list.

use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;

use super::per_cell_diffusion::PerCellDiffusion;
use crate::analysis::observable::{Context, Observable, Requirements};

/// Default mobility cutoff (D_eff). Tunable per-system via TOML once
/// study config gains a percolation block; for v1 the default of 1e-4
/// matches the SPEC.
const DEFAULT_MOBILE_THRESHOLD: f64 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MobilityMetric { DEff }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdMode { Absolute, Percentile }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Adjacency { Distance, Voronoi }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancerLabel { Include, Exclude }

pub struct PercolationCluster {
    pub mobility_metric: MobilityMetric,
    pub threshold_mode: ThresholdMode,
    pub mobile_threshold: f64,
    pub mobile_threshold_pct: f64,
    pub adjacency: Adjacency,
    pub adj_cutoff_factor: f64,  // multiplied by cell_radius
    pub cancer_label: CancerLabel,
    pub cancer_gamma_max: f64,   // cells with gamma <= this are "cancer/soft"
}

impl Default for PercolationCluster {
    fn default() -> Self {
        Self {
            mobility_metric: MobilityMetric::DEff,
            threshold_mode: ThresholdMode::Percentile,
            mobile_threshold: DEFAULT_MOBILE_THRESHOLD,
            mobile_threshold_pct: 50.0,
            adjacency: Adjacency::Voronoi,
            adj_cutoff_factor: 2.0,
            cancer_label: CancerLabel::Include,
            cancer_gamma_max: 0.5,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PercolationClusterOutput {
    pub n_cells_total: usize,
    pub n_cells_mobile: usize,
    pub frac_mobile: f64,
    pub n_clusters: usize,
    pub s_max: usize,
    pub p_inf: f64,
    pub chi: f64,
    pub cluster_size_histogram: BTreeMap<String, usize>,
    pub cluster_sizes_sorted: Vec<usize>,
    pub mean_cluster_mass_excl_max: f64,
    pub sum_s2_excl_max: usize,
    pub sum_s_excl_max: usize,
    pub mobile_threshold_used: f64,
    pub mobility_metric_used: String,
    pub adjacency_used: String,
    pub cancer_label_used: String,
    pub n_cancer_cells_in_cluster: usize,
}

// ---- Union-find ------------------------------------------------------------
struct DSU { parent: Vec<usize>, size: Vec<usize> }
impl DSU {
    fn new(n: usize) -> Self { Self { parent: (0..n).collect(), size: vec![1; n] } }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb { return; }
        let (big, small) = if self.size[ra] >= self.size[rb] { (ra, rb) } else { (rb, ra) };
        self.parent[small] = big;
        self.size[big] += self.size[small];
    }
}

// Voronoi neighbour list for all `n` cells at the given positions on a
// periodic Lx×Ly box. Construction mirrors `voronoi_shape::compute`: take
// cells within cutoff (4R), sort by polar angle, walk consecutive triangles.
// Two cells are Voronoi-neighbours iff they contribute a finite Voronoi
// edge segment between them; for the periodic-disk-packing geometry this
// reduces to "j is one of cell i's angularly-consecutive neighbours within
// the cutoff". Returns per-cell sorted neighbour indices.
fn periodic_delta(ax: f64, ay: f64, bx: f64, by: f64, lx: f64, ly: f64) -> (f64, f64) {
    let mut dx = bx - ax;
    let mut dy = by - ay;
    if dx > lx * 0.5 { dx -= lx; } else if dx < -lx * 0.5 { dx += lx; }
    if dy > ly * 0.5 { dy -= ly; } else if dy < -ly * 0.5 { dy += ly; }
    (dx, dy)
}

/// Gabriel-graph neighbours in a periodic Lx×Ly box.
///
/// `i` and `j` are neighbours iff no third cell lies inside the circle having
/// segment `ij` as its diameter. The Gabriel graph is a strict subgraph of the
/// Delaunay triangulation, so every edge is genuinely local: a pair separated
/// by an intervening cell is never joined. `cutoff` only bounds the candidate
/// search; it does not itself create edges.
///
/// This replaces an earlier function that sorted candidates by polar angle and
/// then returned all of them, which made it a plain distance cutoff (the sort
/// had no effect on the result) while reporting itself as Voronoi adjacency.
fn gabriel_neighbours(
    wx: &[f64], wy: &[f64], lx: f64, ly: f64, cutoff: f64,
) -> Vec<Vec<usize>> {
    let n = wx.len();
    let cutoff2 = cutoff * cutoff;
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (dx, dy) = periodic_delta(wx[i], wy[i], wx[j], wy[j], lx, ly);
            let d2 = dx * dx + dy * dy;
            if d2 >= cutoff2 || d2 == 0.0 { continue; }
            // Circle with ij as diameter: centre at i + d/2, radius |d|/2.
            let (mx, my) = (wx[i] + 0.5 * dx, wy[i] + 0.5 * dy);
            let r2 = 0.25 * d2;
            let mut blocked = false;
            for k in 0..n {
                if k == i || k == j { continue; }
                let (kx, ky) = periodic_delta(mx, my, wx[k], wy[k], lx, ly);
                if kx * kx + ky * ky < r2 { blocked = true; break; }
            }
            if !blocked {
                out[i].push(j);
                out[j].push(i);
            }
        }
    }
    for v in out.iter_mut() { v.sort_unstable(); v.dedup(); }
    out
}

impl Observable for PercolationCluster {
    type Output = PercolationClusterOutput;

    fn id(&self) -> &'static str { "percolation_cluster" }
    fn requires(&self) -> Requirements {
        // POSITIONS for D_eff + adjacency; CHECKPOINT for per-cell gamma
        // (needed to tag cancer cells when cancer_label != Include or
        // to count cancer cells in the giant cluster).
        Requirements::POSITIONS | Requirements::CHECKPOINT
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        let pos = &ctx.positions;
        let n_cells = pos.n_cells;
        let cell_radius = ctx.params.cell_radius;

        // 1. Mobility per cell.
        let d_eff = PerCellDiffusion.compute(ctx)?;
        let metric = &d_eff.d_values;

        // 2. Threshold.
        let threshold = match self.threshold_mode {
            ThresholdMode::Absolute => self.mobile_threshold,
            ThresholdMode::Percentile => {
                let mut sorted: Vec<f64> = metric.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                // Index over (len-1), not len: with len as the scale a "50th
                // percentile" cut left 49/100 cells mobile, a systematic ~1/N
                // offset in the percolation control parameter.
                let idx = ((self.mobile_threshold_pct / 100.0)
                    * ((sorted.len() - 1) as f64))
                    .round()
                    .clamp(0.0, (sorted.len() - 1) as f64) as usize;
                sorted[idx]
            }
        };
        let mut is_mobile: Vec<bool> = metric.iter().map(|&m| m > threshold).collect();

        // Cancer-cell handling. A cell is "cancer/soft" if its per-cell
        // gamma is below `cancer_gamma_max`. Mapping checkpoint cells to
        // position cell_ids by id.
        let mut is_cancer: Vec<bool> = vec![false; n_cells];
        if let Some(ck) = ctx.checkpoint.as_deref() {
            let gammas = &ck.per_cell_gamma;
            let cells = &ck.cells;
            if gammas.len() == cells.len() {
                // Build id -> gamma map, then resolve to position order.
                let mut by_id: std::collections::HashMap<i32, f64> =
                    std::collections::HashMap::with_capacity(cells.len());
                for (c, &g) in cells.iter().zip(gammas.iter()) {
                    by_id.insert(c.id, g as f64);
                }
                for (i, &cid) in pos.cell_ids.iter().enumerate() {
                    if let Some(&g) = by_id.get(&(cid as i32)) {
                        is_cancer[i] = g <= self.cancer_gamma_max;
                    }
                }
            }
        }
        if matches!(self.cancer_label, CancerLabel::Exclude) {
            for i in 0..n_cells {
                if is_cancer[i] { is_mobile[i] = false; }
            }
        }

        // 3. Adjacency among mobile cells (last frame). Voronoi uses the
        // angular-sort approach from voronoi_shape; distance is a simple
        // cutoff. Both wrap positions into [0, L) once for the periodic
        // distance computation.
        let cutoff = self.adj_cutoff_factor * cell_radius;
        let cutoff2 = cutoff * cutoff;
        let t_last = if pos.n_times > 0 { pos.n_times - 1 } else { 0 };
        let lx = pos.lx;
        let ly = pos.ly;

        let mobile_idx: Vec<usize> = (0..n_cells).filter(|&i| is_mobile[i]).collect();
        let m = mobile_idx.len();
        // Position within `mobile_idx` for each global cell id, or usize::MAX.
        let mut slot = vec![usize::MAX; n_cells];
        for (a, &i) in mobile_idx.iter().enumerate() { slot[i] = a; }
        let mut dsu = DSU::new(m);
        if pos.n_times > 0 && m > 1 {
            // Adjacency is built over ALL cells and then induced on the mobile
            // subset. Building it from mobile coordinates alone would let two
            // mobile cells be joined straight through the immobile cells that
            // physically separate them, inflating cluster sizes and pushing the
            // apparent threshold to a lower mobile fraction.
            let wx: Vec<f64> = (0..n_cells)
                .map(|i| pos.positions[t_last][i][0].rem_euclid(lx)).collect();
            let wy: Vec<f64> = (0..n_cells)
                .map(|i| pos.positions[t_last][i][1].rem_euclid(ly)).collect();
            match self.adjacency {
                Adjacency::Distance => {
                    for i in 0..n_cells {
                        if slot[i] == usize::MAX { continue; }
                        for j in (i + 1)..n_cells {
                            if slot[j] == usize::MAX { continue; }
                            let (dx, dy) =
                                periodic_delta(wx[i], wy[i], wx[j], wy[j], lx, ly);
                            if dx * dx + dy * dy < cutoff2 {
                                dsu.union(slot[i], slot[j]);
                            }
                        }
                    }
                }
                Adjacency::Voronoi => {
                    let nbrs = gabriel_neighbours(&wx, &wy, lx, ly, cutoff);
                    for i in 0..n_cells {
                        if slot[i] == usize::MAX { continue; }
                        for &j in &nbrs[i] {
                            if j > i && slot[j] != usize::MAX {
                                dsu.union(slot[i], slot[j]);
                            }
                        }
                    }
                }
            }
        }

        // 4. Cluster sizes.
        let mut sizes_by_root: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for a in 0..m {
            let r = dsu.find(a);
            *sizes_by_root.entry(r).or_insert(0) += 1;
        }
        let mut cluster_sizes_sorted: Vec<usize> =
            sizes_by_root.values().copied().collect();
        cluster_sizes_sorted.sort_unstable_by(|a, b| b.cmp(a));
        let n_clusters = cluster_sizes_sorted.len();
        let s_max = *cluster_sizes_sorted.first().unwrap_or(&0);

        // chi = Σ s² / Σ s, excluding the largest cluster.
        let (sum_s2_excl_max, sum_s_excl_max): (usize, usize) = cluster_sizes_sorted
            .iter().skip(1).fold((0usize, 0usize), |(a, b), &s| (a + s * s, b + s));
        let chi = if sum_s_excl_max > 0 {
            sum_s2_excl_max as f64 / sum_s_excl_max as f64
        } else { 0.0 };
        let mean_cluster_mass_excl_max = if cluster_sizes_sorted.len() > 1 {
            sum_s_excl_max as f64 / (cluster_sizes_sorted.len() - 1) as f64
        } else { 0.0 };

        // Largest-cluster cancer-cell count.
        let n_cancer_cells_in_cluster = if !cluster_sizes_sorted.is_empty() {
            // Find root of largest, then count cancer-tagged mobile_idx
            // entries that resolve to it.
            let mut largest_root = 0usize;
            let mut largest_size = 0usize;
            for (&r, &sz) in &sizes_by_root {
                if sz > largest_size { largest_size = sz; largest_root = r; }
            }
            (0..m).filter(|&a| dsu.find(a) == largest_root)
                  .filter(|&a| is_cancer[mobile_idx[a]])
                  .count()
        } else { 0 };

        let mut hist: BTreeMap<String, usize> = BTreeMap::new();
        for &s in &cluster_sizes_sorted {
            *hist.entry(s.to_string()).or_insert(0) += 1;
        }

        Ok(PercolationClusterOutput {
            n_cells_total: n_cells,
            n_cells_mobile: m,
            frac_mobile: if n_cells > 0 { m as f64 / n_cells as f64 } else { 0.0 },
            n_clusters,
            s_max,
            p_inf: if n_cells > 0 { s_max as f64 / n_cells as f64 } else { 0.0 },
            chi,
            cluster_size_histogram: hist,
            cluster_sizes_sorted,
            mean_cluster_mass_excl_max,
            sum_s2_excl_max,
            sum_s_excl_max,
            mobile_threshold_used: threshold,
            mobility_metric_used: match self.mobility_metric {
                MobilityMetric::DEff => "d_eff".to_string(),
            },
            adjacency_used: match self.adjacency {
                Adjacency::Distance => format!("distance(cutoff={:.3})", cutoff),
                Adjacency::Voronoi  => format!("gabriel(cutoff={:.3})", cutoff),
            },
            cancer_label_used: match self.cancer_label {
                CancerLabel::Include => "include".to_string(),
                CancerLabel::Exclude => "exclude".to_string(),
            },
            n_cancer_cells_in_cluster,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsu_basic() {
        let mut d = DSU::new(5);
        d.union(0, 1);
        d.union(1, 2);
        d.union(3, 4);
        assert_eq!(d.find(0), d.find(2));
        assert_ne!(d.find(0), d.find(3));
    }
}
