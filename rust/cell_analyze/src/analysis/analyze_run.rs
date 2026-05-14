//! Per-run analysis: load data once, run requested observables, return
//! a typed [`RunAnalysis`] that serializes to JSON.
//!
//! This is the heart of the v2 pipeline. Studies discover [`RunSpec`]s,
//! call [`analyze_run`] on each (in parallel via rayon), and feed the
//! resulting [`RunAnalysis`] vector to the aggregation layer.

use anyhow::{Context as _, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::analysis::checkpoint::{load_checkpoint, Checkpoint};
use crate::analysis::io::{load_trajectory_subsample, unwrap_trajectory, Trajectory};

use super::discovery::{RunSpec, ScalarValue};
use super::observable::{Context, ObservableBag, Requirements, RunParams};
use super::observables::ErasedObservable;

// ---------------------------------------------------------------------------
// RunAnalysis
// ---------------------------------------------------------------------------
/// Output of [`analyze_run`]: typed observable bag plus enough context
/// (variables, params) to drive aggregation and plotting.
///
/// Serializable so the entire result can be dumped to
/// `run_analysis.json` and re-loaded by downstream tooling on a
/// different machine.
pub struct RunAnalysis {
    pub directory: std::path::PathBuf,
    pub variables: BTreeMap<String, ScalarValue>,
    pub params: RunParams,
    pub metadata: RunMetadata,
    pub bag: ObservableBag,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunMetadata {
    pub n_cells: usize,
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
    pub dim: usize,
    pub dt: Option<f64>,
    pub tau: Option<f64>,
    pub tau_source: String,
    pub time_start: Option<f64>,
    pub time_end: Option<f64>,
    pub duration: Option<f64>,
    pub frame_count: usize,
    pub subsample: usize,
    pub checkpoint_time: Option<f64>,
    pub checkpoint_step: Option<i32>,
    pub checkpoint_tau: Option<f64>,
    pub tagged_gamma: Option<f64>,
    pub tagged_v_a: Option<f64>,
}

/// JSON-serializable view of a [`RunAnalysis`].
#[derive(Debug, Serialize)]
pub struct RunAnalysisJson {
    pub directory: String,
    pub variables: BTreeMap<String, ScalarValue>,
    pub params: RunParams,
    pub metadata: RunMetadata,
    /// Map observable id → its serialized output (a serde_json::Value).
    pub observables: BTreeMap<String, serde_json::Value>,
}

fn infer_dt_from_frames(frames: &[(f64, std::collections::HashMap<u32, crate::analysis::io::CellSnapshot>)]) -> Option<f64> {
    let mut prev = None;
    for (t, _) in frames {
        if let Some(p) = prev {
            let dt = *t - p;
            if dt > 0.0 {
                return Some(dt);
            }
        }
        prev = Some(*t);
    }
    None
}

// ---------------------------------------------------------------------------
// analyze_run
// ---------------------------------------------------------------------------
/// Plan for running observables: which to compute + per-run params.
pub struct AnalyzePlan<'a> {
    pub observables: &'a [Box<dyn ErasedObservable>],
    pub params: RunParams,
    pub subsample: usize,
}

/// Load data, run observables, return a typed [`RunAnalysis`].
///
/// Loads only what's needed: the unwrapped positions are always loaded
/// (most observables need them); the raw trajectory and checkpoint are
/// loaded only if some observable's `requires()` says so.
///
/// `prevalidated` lets callers pass a [`Trajectory`] that was already
/// parsed by the validation pre-pass — used by `study` to avoid loading
/// the trajectory twice. Only consumed when `plan.subsample <= 1`
/// (subsampled loads need a fresh parse).
pub fn analyze_run(
    spec: &RunSpec,
    plan: &AnalyzePlan<'_>,
    prevalidated: Option<Arc<Trajectory>>,
) -> Result<RunAnalysis> {
    // 1. Compute the union of requirements.
    let needed = plan
        .observables
        .iter()
        .fold(Requirements::POSITIONS, |acc, o| acc | o.requires());

    // 2. Load only what's needed. Reuse the pre-validated trajectory
    //    when present and the caller didn't ask for subsampling.
    let traj_arc: Arc<Trajectory> = match prevalidated {
        Some(t) if plan.subsample <= 1 => t,
        _ => Arc::new(
            load_trajectory_subsample(&spec.trajectory, plan.subsample)
                .with_context(|| format!("load trajectory {}", spec.trajectory.display()))?,
        ),
    };
    let traj: &Trajectory = &traj_arc;
    let n_cells = traj.params.n_cells;
    let lx = traj.params.lx;
    let ly = traj.params.ly;
    let lz = traj.params.lz;
    let dim = traj.params.dim;
    let tau = traj.params.tau;
    let tau_source = traj.params.tau_source.to_string();
    let time_start = traj.frames.first().map(|(t, _)| *t);
    let time_end = traj.frames.last().map(|(t, _)| *t);
    let dt_from_traj = infer_dt_from_frames(&traj.frames);
    let frame_count = traj.frames.len();
    let positions = Arc::new(unwrap_trajectory(traj));
    let trajectory = if needed.contains(Requirements::TRAJECTORY) {
        Some(traj_arc.clone())
    } else {
        None
    };
    let checkpoint: Option<Arc<Checkpoint>> = if needed.contains(Requirements::CHECKPOINT) {
        let path = spec
            .checkpoint
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "observable requires checkpoint but spec has none for {}",
                    spec.directory.display()
                )
            })?;
        Some(Arc::new(load_checkpoint(path).with_context(|| {
            format!("load checkpoint {}", path.display())
        })?))
    } else {
        None
    };

    // 2b. Capture immutable run provenance so figures/JSON can include
    // full metadata (domain, time span, checkpoint context).
    let dt_from_ckpt = checkpoint.as_ref().map(|ck| ck.params.dt as f64);
    let checkpoint_time = checkpoint.as_ref().map(|ck| ck.header.time);
    let checkpoint_step = checkpoint.as_ref().map(|ck| ck.header.step);
    let checkpoint_tau = checkpoint.as_ref().map(|ck| ck.header.time / 10000.0);
    let tagged_gamma = checkpoint
        .as_ref()
        .and_then(|ck| ck.per_cell_gamma.first().copied())
        .map(|g| g as f64);
    let tagged_v_a = checkpoint
        .as_ref()
        .and_then(|ck| ck.per_cell_v_a.first().copied())
        .map(|v| v as f64);
    let metadata = RunMetadata {
        n_cells,
        lx,
        ly,
        lz,
        dim,
        dt: dt_from_ckpt.or(dt_from_traj),
        tau,
        tau_source,
        time_start,
        time_end,
        duration: match (time_start, time_end) {
            (Some(t0), Some(t1)) => Some(t1 - t0),
            _ => None,
        },
        frame_count,
        subsample: plan.subsample,
        checkpoint_time,
        checkpoint_step,
        checkpoint_tau,
        tagged_gamma,
        tagged_v_a,
    };

    // 3. Build context.
    let ctx = Context {
        positions,
        trajectory,
        checkpoint,
        params: plan.params.clone(),
    };

    // 4. Run each observable in turn.
    let mut bag = ObservableBag::new();
    for obs in plan.observables {
        obs.compute_into_bag(&ctx, &mut bag).with_context(|| {
            format!(
                "observable `{}` failed for {}",
                obs.id(),
                spec.directory.display()
            )
        })?;
    }

    Ok(RunAnalysis {
        directory: spec.directory.clone(),
        variables: spec.variables.clone(),
        params: plan.params.clone(),
        metadata,
        bag,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::observables::{msd::Msd, register_builtin};
    use std::fs;

    /// Write a minimal v2-format trajectory file: header line + a few
    /// frames of one cell drifting in x.
    fn write_synthetic_traj(path: &std::path::Path) -> Result<()> {
        let mut s = String::new();
        s.push_str("# v_A=0.01 N=1 Lx=1000.0 Ly=1000.0 dim=2 tau=10000.0\n");
        // Schema (12 cols, 2D): time cell_id x y vx vy px py theta v_a L_n volume
        for t in 0..20 {
            let time = t as f64;
            let x = 0.1 * time;
            s.push_str(&format!(
                "{:.6}\t0\t{:.6}\t0.0\t0.1\t0.0\t1.0\t0.0\t0.0\t0.01\t1.0\t100.0\n",
                time, x
            ));
        }
        fs::write(path, s)?;
        Ok(())
    }

    #[test]
    fn analyze_run_msd_end_to_end() {
        let dir = std::env::temp_dir().join("v2_analyze_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let traj = dir.join("trajectory.txt");
        write_synthetic_traj(&traj).unwrap();

        let spec = RunSpec {
            directory: dir.clone(),
            trajectory: traj,
            checkpoint: None,
            variables: BTreeMap::new(),
        };
        let observables = register_builtin(); // includes Msd
        let plan = AnalyzePlan {
            observables: &observables,
            params: RunParams::default(),
            subsample: 1,
        };
        let res = analyze_run(&spec, &plan, None).expect("analyze_run");
        let msd = res.bag.get::<Msd>().expect("msd missing");
        assert!(!msd.lag_times.is_empty(), "got msd lag times");
        // Cell 0 drifts at v=0.1, so MSD at lag 1 ≈ 0.01.
        assert!((msd.cell0_values[0] - 0.01).abs() < 1e-9,
                "cell0 msd at lag 1 = {}", msd.cell0_values[0]);

        let _ = fs::remove_dir_all(&dir);
    }
}
