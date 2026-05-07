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
use crate::analysis::io::{load_trajectory_subsample, unwrap_trajectory};

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
    pub bag: ObservableBag,
}

/// JSON-serializable view of a [`RunAnalysis`]. The bag is dropped (it
/// contains type-erased values that can't be Serialize'd directly);
/// observable results are inserted into `observables` as a separately
/// serialized map by the caller — see [`RunAnalysis::to_json`].
#[derive(Debug, Serialize)]
pub struct RunAnalysisJson {
    pub directory: String,
    pub variables: BTreeMap<String, ScalarValue>,
    pub params: RunParams,
    /// Map observable id → its serialized output (a serde_json::Value).
    pub observables: BTreeMap<String, serde_json::Value>,
}

impl RunAnalysis {
    /// Build a JSON-serializable view. The caller must provide a
    /// closure that knows how to serialize each observable's output;
    /// this is delegated because `ObservableBag` is type-erased.
    pub fn to_json(
        &self,
        serializers: &[(
            &'static str,
            Box<dyn Fn(&ObservableBag) -> Option<serde_json::Value>>,
        )],
    ) -> RunAnalysisJson {
        let mut observables = BTreeMap::new();
        for (id, ser) in serializers {
            if let Some(v) = ser(&self.bag) {
                observables.insert((*id).to_string(), v);
            }
        }
        RunAnalysisJson {
            directory: self.directory.to_string_lossy().to_string(),
            variables: self.variables.clone(),
            params: self.params.clone(),
            observables,
        }
    }
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
pub fn analyze_run(spec: &RunSpec, plan: &AnalyzePlan<'_>) -> Result<RunAnalysis> {
    // 1. Compute the union of requirements.
    let needed = plan
        .observables
        .iter()
        .fold(Requirements::POSITIONS, |acc, o| acc | o.requires());

    // 2. Load only what's needed.
    let traj = load_trajectory_subsample(&spec.trajectory, plan.subsample)
        .with_context(|| format!("load trajectory {}", spec.trajectory.display()))?;
    let positions = Arc::new(unwrap_trajectory(&traj));
    let trajectory = if needed.contains(Requirements::TRAJECTORY) {
        Some(Arc::new(traj))
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
        bag,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::v2::observables::{msd::Msd, register_builtin};
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
        let res = analyze_run(&spec, &plan).expect("analyze_run");
        let msd = res.bag.get::<Msd>().expect("msd missing");
        assert!(!msd.lag_times.is_empty(), "got msd lag times");
        // Cell 0 drifts at v=0.1, so MSD at lag 1 ≈ 0.01.
        assert!((msd.cell0_values[0] - 0.01).abs() < 1e-9,
                "cell0 msd at lag 1 = {}", msd.cell0_values[0]);

        let _ = fs::remove_dir_all(&dir);
    }
}
