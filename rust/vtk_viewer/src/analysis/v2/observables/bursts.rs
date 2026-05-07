//! Speed-burst detection: identifies frames where |v| exceeds
//! `μ + k·σ` for at least `min_frames` consecutive samples.

use anyhow::Result;
use serde::Serialize;

use crate::analysis::io::UnwrappedPositions;
use crate::analysis::v2::observable::{Context, Observable, Requirements};

pub struct Bursts {
    pub k_sigma: f64,
    pub min_frames: usize,
}

impl Default for Bursts {
    fn default() -> Self {
        Self {
            k_sigma: 3.0,
            min_frames: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BurstEvent {
    pub cell_id: u32,
    pub t_start: f64,
    pub t_end: f64,
    pub duration: f64,
    pub peak_speed: f64,
    pub mean_speed: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct BurstsOutput {
    pub threshold: f64,
    pub speed_mean: f64,
    pub speed_std: f64,
    pub total_bursts: usize,
    pub mean_bursts_per_cell: f64,
    pub mean_duration: f64,
    pub mean_peak_speed: f64,
    pub events: Vec<BurstEvent>,
}

/// Stand-alone burst detection. Same algorithm as the legacy
/// `detect_bursts`. The `_traj` parameter from the legacy signature is
/// dropped (never read).
pub fn compute_bursts(
    pos: &UnwrappedPositions,
    k_sigma: f64,
    min_frames: usize,
) -> BurstsOutput {
    let n = pos.n_times;
    let nc = pos.n_cells;
    let cell_ids = &pos.cell_ids;
    let empty = BurstsOutput {
        threshold: 0.0,
        speed_mean: 0.0,
        speed_std: 0.0,
        total_bursts: 0,
        mean_bursts_per_cell: 0.0,
        mean_duration: 0.0,
        mean_peak_speed: 0.0,
        events: vec![],
    };
    if n < 2 || nc == 0 {
        return empty;
    }

    // Speeds from finite differences (one row per (n-1, nc)).
    let mut speeds: Vec<Vec<f64>> = Vec::with_capacity(n - 1);
    for t in 0..(n - 1) {
        let dt = pos.times[t + 1] - pos.times[t];
        if dt <= 0.0 {
            speeds.push(vec![0.0; nc]);
            continue;
        }
        let mut frame = vec![0.0; nc];
        for i in 0..nc {
            let dx = pos.positions[t + 1][i][0] - pos.positions[t][i][0];
            let dy = pos.positions[t + 1][i][1] - pos.positions[t][i][1];
            let dz = pos.positions[t + 1][i][2] - pos.positions[t][i][2];
            frame[i] = (dx * dx + dy * dy + dz * dz).sqrt() / dt;
        }
        speeds.push(frame);
    }

    let all_speeds: Vec<f64> = speeds.iter().flat_map(|f| f.iter().copied()).collect();
    if all_speeds.is_empty() {
        return empty;
    }
    let speed_mean = all_speeds.iter().sum::<f64>() / all_speeds.len() as f64;
    let speed_var = all_speeds
        .iter()
        .map(|s| (s - speed_mean).powi(2))
        .sum::<f64>()
        / all_speeds.len() as f64;
    let speed_std = speed_var.sqrt();
    let threshold = speed_mean + k_sigma * speed_std;

    let mut events = Vec::new();
    let n_speed = speeds.len();
    for ci in 0..nc {
        let cid = cell_ids[ci];
        let mut t = 0usize;
        while t < n_speed {
            if speeds[t][ci] > threshold {
                let t_start_idx = t;
                let mut peak = speeds[t][ci];
                let mut sum_speed = speeds[t][ci];
                let mut count = 1usize;
                t += 1;
                while t < n_speed && speeds[t][ci] > threshold {
                    if speeds[t][ci] > peak {
                        peak = speeds[t][ci];
                    }
                    sum_speed += speeds[t][ci];
                    count += 1;
                    t += 1;
                }
                if count >= min_frames {
                    let end_t = pos.times[(t_start_idx + count).min(n - 1)];
                    events.push(BurstEvent {
                        cell_id: cid,
                        t_start: pos.times[t_start_idx],
                        t_end: end_t,
                        duration: end_t - pos.times[t_start_idx],
                        peak_speed: peak,
                        mean_speed: sum_speed / count as f64,
                    });
                }
            } else {
                t += 1;
            }
        }
    }

    let total = events.len();
    let mean_dur = if total > 0 {
        events.iter().map(|e| e.duration).sum::<f64>() / total as f64
    } else {
        0.0
    };
    let mean_peak = if total > 0 {
        events.iter().map(|e| e.peak_speed).sum::<f64>() / total as f64
    } else {
        0.0
    };
    if events.len() > 10000 {
        events.truncate(10000);
    }

    BurstsOutput {
        threshold,
        speed_mean,
        speed_std,
        total_bursts: total,
        mean_bursts_per_cell: total as f64 / nc as f64,
        mean_duration: mean_dur,
        mean_peak_speed: mean_peak,
        events,
    }
}

impl Observable for Bursts {
    type Output = BurstsOutput;

    fn id(&self) -> &'static str {
        "bursts"
    }

    fn requires(&self) -> Requirements {
        Requirements::POSITIONS
    }

    fn compute(&self, ctx: &Context) -> Result<Self::Output> {
        Ok(compute_bursts(&ctx.positions, self.k_sigma, self.min_frames))
    }
}
