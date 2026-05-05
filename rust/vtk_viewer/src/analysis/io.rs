//! Trajectory I/O and periodic boundary unwrapping.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Per-cell snapshot at a single time point.
#[derive(Clone, Copy, Debug)]
pub struct CellSnapshot {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub px: f64,
    pub py: f64,
    pub pz: f64,
    pub theta: f64,
    /// Per-cell inherent v_A (10th column in 2D, 14th in 3D)
    pub v_a_inherent: f64,
    /// Normalized perimeter L_n (11th column in 2D, 0.0 in 3D)
    pub l_n: f64,
}

/// How tau was determined
#[derive(Clone, Debug, PartialEq)]
pub enum TauSource {
    /// Parsed from trajectory header ("tau=...")
    Header,
    /// Read from checkpoint.bin SimParams
    Checkpoint,
    /// Unknown — no source available
    Unknown,
}

impl std::fmt::Display for TauSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TauSource::Header => write!(f, "header"),
            TauSource::Checkpoint => write!(f, "checkpoint"),
            TauSource::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Parameters parsed from the trajectory header.
#[derive(Clone, Debug)]
pub struct TrajectoryParams {
    pub v_a: f64,
    pub n_cells: usize,
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
    /// 2 or 3
    pub dim: usize,
    /// Persistence time τ — None until resolved from header, checkpoint, or polarity
    pub tau: Option<f64>,
    /// How tau was determined
    pub tau_source: TauSource,
    /// Extra key-value pairs from header (e.g. adhesion_J)
    pub extra: HashMap<String, String>,
}

impl Default for TrajectoryParams {
    fn default() -> Self {
        Self {
            v_a: 0.0,
            n_cells: 0,
            lx: 1600.0,
            ly: 1600.0,
            lz: 0.0,
            dim: 2,
            tau: None,
            tau_source: TauSource::Unknown,
            extra: HashMap::new(),
        }
    }
}

/// Full loaded trajectory: params + time-ordered frames.
pub struct Trajectory {
    pub params: TrajectoryParams,
    /// Sorted by time: (time, cell_id -> snapshot)
    pub frames: Vec<(f64, HashMap<u32, CellSnapshot>)>,
}

impl Trajectory {
    /// Number of unique cells across all frames (from first frame).
    pub fn n_cells(&self) -> usize {
        self.frames.first().map_or(0, |(_, m)| m.len())
    }

    /// Sorted unique cell IDs from the first frame.
    pub fn cell_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .frames
            .first()
            .map_or(Vec::new(), |(_, m)| m.keys().cloned().collect());
        ids.sort();
        ids
    }
}

/// Load trajectory from file. Parses header for v_A, N, Lx, Ly, Lz, dim.
///
/// Handles chain-job overlaps automatically:
/// - **Independent restart** (backward jump > 10000 TU, start positions match):
///   keep the longest monotonic segment.
/// - **Chain continuation** (backward jump < 1000 TU, positions continuous):
///   stitch segments, trimming the overlap.
pub fn load_trajectory(path: &Path) -> Result<Trajectory> {
    load_trajectory_subsample(path, 1)
}

/// Load a trajectory file with optional subsampling.
/// `subsample` = 1 means keep every frame, 10 means keep every 10th, etc.
pub fn load_trajectory_subsample(path: &Path, subsample: usize) -> Result<Trajectory> {
    let file = fs::File::open(path).context("Opening trajectory file")?;
    let reader = BufReader::new(file);

    let mut params = TrajectoryParams::default();
    let subsample = subsample.max(1);

    // Phase 1: Parse all rows in file order, tracking cell-0 timestamps
    struct RawRow {
        time: f64,
        cell_id: u32,
        snap: CellSnapshot,
    }
    let mut rows: Vec<RawRow> = Vec::new();
    let mut cell0_count: usize = 0;
    let mut keep_frame = true;

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') {
            for tok in line.split_whitespace() {
                if let Some((k, v)) = tok.split_once('=') {
                    match k {
                        "v_A" => params.v_a = v.parse().unwrap_or(0.0),
                        "N" => params.n_cells = v.parse().unwrap_or(0),
                        "Lx" => params.lx = v.parse().unwrap_or(1600.0),
                        "Ly" => params.ly = v.parse().unwrap_or(1600.0),
                        "Lz" => {
                            params.lz = v.parse().unwrap_or(0.0);
                            if params.lz > 0.0 { params.dim = 3; }
                        }
                        "dim" => params.dim = v.parse().unwrap_or(2),
                        "tau" => {
                            params.tau = v.parse().ok();
                            if params.tau.is_some() {
                                params.tau_source = TauSource::Header;
                            }
                        }
                        _ => {
                            params.extra.insert(k.to_string(), v.to_string());
                        }
                    }
                }
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 9 {
            continue;
        }

        let t: f64 = parts[0].parse().unwrap_or(0.0);
        let cid: u32 = parts[1].parse().unwrap_or(0);

        // Subsampling: count cell-0 appearances to track frame number
        if cid == 0 {
            keep_frame = cell0_count % subsample == 0;
            cell0_count += 1;
        }
        if !keep_frame {
            continue;
        }

        let snap = if parts.len() >= 14 || params.dim == 3 {
            if parts.len() < 14 { continue; }
            CellSnapshot {
                x: parts[2].parse().unwrap_or(0.0),
                y: parts[3].parse().unwrap_or(0.0),
                z: parts[4].parse().unwrap_or(0.0),
                vx: parts[5].parse().unwrap_or(0.0),
                vy: parts[6].parse().unwrap_or(0.0),
                vz: parts[7].parse().unwrap_or(0.0),
                px: parts[8].parse().unwrap_or(0.0),
                py: parts[9].parse().unwrap_or(0.0),
                pz: parts[10].parse().unwrap_or(0.0),
                theta: parts[11].parse().unwrap_or(0.0),
                v_a_inherent: parts[13].parse().unwrap_or(0.0),
                l_n: 0.0,
            }
        } else {
            CellSnapshot {
                x: parts[2].parse().unwrap_or(0.0),
                y: parts[3].parse().unwrap_or(0.0),
                z: 0.0,
                vx: parts[4].parse().unwrap_or(0.0),
                vy: parts[5].parse().unwrap_or(0.0),
                vz: 0.0,
                px: parts[6].parse().unwrap_or(0.0),
                py: parts[7].parse().unwrap_or(0.0),
                pz: 0.0,
                theta: parts[8].parse().unwrap_or(0.0),
                v_a_inherent: if parts.len() > 9 { parts[9].parse().unwrap_or(0.0) } else { 0.0 },
                l_n: if parts.len() > 10 { parts[10].parse().unwrap_or(0.0) } else { 0.0 },
            }
        };

        rows.push(RawRow { time: t, cell_id: cid, snap });
    }

    // Phase 2: Detect chain-job overlaps via cell-0 backward time jumps
    let cell0_indices: Vec<usize> = rows.iter().enumerate()
        .filter(|(_, r)| r.cell_id == 0)
        .map(|(i, _)| i)
        .collect();

    let mut backward_jumps: Vec<usize> = Vec::new(); // indices into cell0_indices
    for i in 1..cell0_indices.len() {
        let prev_t = rows[cell0_indices[i - 1]].time;
        let curr_t = rows[cell0_indices[i]].time;
        if curr_t < prev_t {
            backward_jumps.push(i);
        }
    }

    if !backward_jumps.is_empty() {
        // Split into segments at each backward jump
        let n_cells_est = if params.n_cells > 0 { params.n_cells } else {
            // Estimate from first frame
            rows.iter().take_while(|r| r.time == rows[0].time).count()
        };

        let mut seg_boundaries: Vec<usize> = vec![0]; // row indices into `rows`
        for &bj in &backward_jumps {
            // The backward jump is between cell0_indices[bj-1] and cell0_indices[bj].
            // The segment boundary in rows is at the first row of the new segment.
            // We need to find the start of the frame containing cell0_indices[bj].
            // Approximate: the row index minus n_cells * cell_id (cell 0 is first in frame)
            seg_boundaries.push(cell0_indices[bj]);
        }
        seg_boundaries.push(rows.len());

        let mut segments: Vec<(usize, usize, f64, f64, f64, f64)> = Vec::new(); // (start, end, t_first, t_last, x0_first, y0_first)
        for i in 0..seg_boundaries.len() - 1 {
            let start = seg_boundaries[i];
            let end = seg_boundaries[i + 1];
            let seg_rows = &rows[start..end];
            let first_c0 = seg_rows.iter().find(|r| r.cell_id == 0);
            let last_c0 = seg_rows.iter().rev().find(|r| r.cell_id == 0);
            if let (Some(first), Some(last)) = (first_c0, last_c0) {
                segments.push((start, end, first.time, last.time, first.snap.x, first.snap.y));
            }
        }

        // Classify each backward jump
        let mut keep_ranges: Vec<(usize, usize)> = Vec::new();

        if segments.len() == 2 {
            let (s0_start, s0_end, s0_t0, s0_t1, s0_x, s0_y) = segments[0];
            let (s1_start, s1_end, s1_t0, s1_t1, s1_x, s1_y) = segments[1];
            let time_jump = s0_t1 - s1_t0;
            let pos_dist = ((s0_x - s1_x).powi(2) + (s0_y - s1_y).powi(2)).sqrt();

            if time_jump > 10000.0 && pos_dist < 50.0 {
                // Type 1: Independent restart from same checkpoint
                // Keep the longer segment
                let s0_frames = rows[s0_start..s0_end].iter().filter(|r| r.cell_id == 0).count();
                let s1_frames = rows[s1_start..s1_end].iter().filter(|r| r.cell_id == 0).count();
                if s0_frames >= s1_frames {
                    keep_ranges.push((s0_start, s0_end));
                    eprintln!("  Chain overlap: independent restart, keeping segment 0 ({} frames)", s0_frames);
                } else {
                    keep_ranges.push((s1_start, s1_end));
                    eprintln!("  Chain overlap: independent restart, keeping segment 1 ({} frames)", s1_frames);
                }
            } else if time_jump < 1000.0 {
                // Type 2: Chain continuation
                // Stitch: keep segment 0 up to the overlap, then segment 1 from after overlap
                // Find last time in seg0 that's before seg1's first time
                let trim_time = s1_t0;
                let trim_end = rows[s0_start..s0_end].iter()
                    .rposition(|r| r.time < trim_time)
                    .map(|i| s0_start + i + 1)
                    .unwrap_or(s0_end);
                keep_ranges.push((s0_start, trim_end));
                keep_ranges.push((s1_start, s1_end));
                let kept_s0 = rows[s0_start..trim_end].iter().filter(|r| r.cell_id == 0).count();
                let kept_s1 = rows[s1_start..s1_end].iter().filter(|r| r.cell_id == 0).count();
                eprintln!("  Chain overlap: continuation, stitching ({} + {} frames)", kept_s0, kept_s1);
            } else {
                // Ambiguous: keep longest
                let s0_frames = rows[s0_start..s0_end].iter().filter(|r| r.cell_id == 0).count();
                let s1_frames = rows[s1_start..s1_end].iter().filter(|r| r.cell_id == 0).count();
                if s0_frames >= s1_frames {
                    keep_ranges.push((s0_start, s0_end));
                } else {
                    keep_ranges.push((s1_start, s1_end));
                }
                eprintln!("  Chain overlap: ambiguous (time_jump={:.0}, pos_dist={:.1}), keeping longest", time_jump, pos_dist);
            }
        } else {
            // Multiple overlaps: keep longest segment
            let best = segments.iter().enumerate()
                .max_by_key(|(_, s)| rows[s.0..s.1].iter().filter(|r| r.cell_id == 0).count())
                .map(|(_, s)| (s.0, s.1));
            if let Some((start, end)) = best {
                keep_ranges.push((start, end));
                eprintln!("  Chain overlap: {} jumps, keeping longest segment", backward_jumps.len());
            }
        }

        // Rebuild rows from keep_ranges
        let mut keep_indices: Vec<bool> = vec![false; rows.len()];
        for &(start, end) in &keep_ranges {
            for i in start..end {
                keep_indices[i] = true;
            }
        }
        let mut idx = 0;
        rows.retain(|_| { let keep = keep_indices[idx]; idx += 1; keep });
    }

    // Phase 3: Build frame hashmap from (possibly cleaned) rows
    let mut by_time: HashMap<i64, HashMap<u32, CellSnapshot>> = HashMap::new();
    let mut time_order: Vec<(i64, f64)> = Vec::new();

    for row in &rows {
        // Group rows by frame: round time to nearest 1 TU.
        // This handles floating-point drift (e.g., 80001.005625 vs 80001.005603)
        // while keeping distinct frames separate (dt_save >= 10 TU typically).
        let key = (row.time + 0.5) as i64;
        let frame = by_time.entry(key).or_insert_with(|| {
            time_order.push((key, row.time));
            HashMap::new()
        });
        frame.insert(row.cell_id, row.snap);
    }

    time_order.sort_by_key(|&(k, _)| k);
    let frames: Vec<(f64, HashMap<u32, CellSnapshot>)> = time_order
        .into_iter()
        .map(|(k, t)| (t, by_time.remove(&k).unwrap()))
        .collect();

    if params.n_cells == 0 && !frames.is_empty() {
        params.n_cells = frames[0].1.len();
    }

    eprintln!(
        "Loaded trajectory: {} frames, {} cells, v_A={}, dim={}, Lx={}, Ly={}{}",
        frames.len(),
        params.n_cells,
        params.v_a,
        params.dim,
        params.lx,
        params.ly,
        if params.dim == 3 { format!(", Lz={}", params.lz) } else { String::new() },
    );

    // Resolve tau: header → checkpoint only. No guessing.
    if params.tau.is_none() {
        // Try checkpoint.bin in the same directory
        if let Some(parent) = path.parent() {
            let ckpt_path = parent.join("checkpoint.bin");
            if ckpt_path.exists() {
                if let Ok(ckpt) = super::checkpoint::load_checkpoint(&ckpt_path) {
                    let ckpt_tau = ckpt.params.tau;
                    if ckpt_tau > 0.0 {
                        eprintln!("  tau={} from checkpoint.bin", ckpt_tau);
                        params.tau = Some(ckpt_tau as f64);
                        params.tau_source = TauSource::Checkpoint;
                    }
                }
            }
        }
    }

    if params.tau.is_none() {
        eprintln!("  WARNING: tau=UNKNOWN (not in header, no checkpoint found)");
    }

    Ok(Trajectory { params, frames })
}

/// Unwrapped position arrays for a set of cells.
///
/// Shape: `positions[time_idx][cell_idx] = [x, y, z]` (unwrapped, z=0 for 2D).
/// `cell_ids` gives the mapping from cell_idx to cell_id.
/// `times[time_idx]` gives the simulation time.
pub struct UnwrappedPositions {
    pub times: Vec<f64>,
    pub cell_ids: Vec<u32>,
    /// positions\[t\]\[i\] = [x, y, z] unwrapped for cell_ids\[i\]
    pub positions: Vec<Vec<[f64; 3]>>,
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
    pub dim: usize,
    pub n_cells: usize,
    pub n_times: usize,
    /// Per-cell inherent v_A (from 10th column of trajectory). Empty if not available.
    pub inherent_v_a: Vec<f64>,
}

impl UnwrappedPositions {
    /// Wrapped positions for cell i at time t (mod Lx, Ly, Lz).
    pub fn wrapped(&self, t: usize, i: usize) -> [f64; 2] {
        let p = self.positions[t][i];
        [p[0].rem_euclid(self.lx), p[1].rem_euclid(self.ly)]
    }
}

/// Build unwrapped position arrays from a trajectory.
///
/// Handles periodic boundary crossings by detecting jumps > L/2.
/// Discards incomplete time points (fewer cells than expected).
pub fn unwrap_trajectory(traj: &Trajectory) -> UnwrappedPositions {
    let cell_ids = traj.cell_ids();
    let n_cells = cell_ids.len();
    let id_to_idx: HashMap<u32, usize> = cell_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let lx = traj.params.lx;
    let ly = traj.params.ly;
    let lz = traj.params.lz;
    let dim = traj.params.dim;

    let mut times = Vec::with_capacity(traj.frames.len());
    let mut positions: Vec<Vec<[f64; 3]>> = Vec::with_capacity(traj.frames.len());
    let mut skipped_incomplete = 0usize;

    for (t, cells) in &traj.frames {
        // Skip incomplete time points
        if cells.len() < n_cells {
            skipped_incomplete += 1;
            continue;
        }

        let mut frame = vec![[0.0, 0.0, 0.0]; n_cells];
        for (&cid, snap) in cells {
            if let Some(&idx) = id_to_idx.get(&cid) {
                frame[idx] = [snap.x, snap.y, snap.z];
            }
        }

        // Unwrap periodic boundaries relative to previous frame.
        // Note: `prev` holds *unwrapped* coordinates that may be many box-
        // lengths away from the raw read-from-file value, so we need
        // multi-wrap minimum-image (round-based), not a single ±L correction.
        if let Some(prev) = positions.last() {
            for i in 0..n_cells {
                let mut dx = frame[i][0] - prev[i][0];
                let mut dy = frame[i][1] - prev[i][1];
                if lx > 0.0 { dx -= lx * (dx / lx).round(); }
                if ly > 0.0 { dy -= ly * (dy / ly).round(); }
                frame[i][0] = prev[i][0] + dx;
                frame[i][1] = prev[i][1] + dy;

                if dim == 3 && lz > 0.0 {
                    let mut dz = frame[i][2] - prev[i][2];
                    dz -= lz * (dz / lz).round();
                    frame[i][2] = prev[i][2] + dz;
                }
            }
        }

        times.push(*t);
        positions.push(frame);
    }

    let n_times = times.len();

    // Extract per-cell inherent v_A from second time step (avoids checkpoint artifact at t=0)
    let inherent_v_a = if traj.frames.len() >= 2 {
        let t_idx = 1.min(traj.frames.len() - 1);
        let (_, cells) = &traj.frames[t_idx];
        let mut va = vec![0.0; n_cells];
        for (&cid, snap) in cells {
            if let Some(&idx) = id_to_idx.get(&cid) {
                va[idx] = snap.v_a_inherent;
            }
        }
        // Only keep if at least one nonzero (otherwise trajectory has no per-cell v_A column)
        if va.iter().any(|&v| v > 0.0) { va } else { vec![] }
    } else {
        vec![]
    };

    if skipped_incomplete > 0 {
        eprintln!(
            "  Warning: {} frames skipped (incomplete: expected {} cells per frame)",
            skipped_incomplete, n_cells
        );
    }

    UnwrappedPositions {
        times,
        cell_ids,
        positions,
        lx,
        ly,
        lz,
        dim,
        n_cells,
        n_times,
        inherent_v_a,
    }
}
