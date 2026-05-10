//! Convenience helpers for the runnable examples in `examples/`.
//!
//! Each per-observable example loads a trajectory directory, builds an
//! `analysis::observable::Context`, runs ONE observable, and prints
//! the headline numbers. To keep each example tiny, the boilerplate
//! lives here.

use anyhow::{Context as _, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::analysis::checkpoint::load_checkpoint;
use crate::analysis::io::{load_trajectory_subsample, unwrap_trajectory};
use crate::analysis::observable::{Context, RunParams};

/// Path to the local 100c-rho90 ctrl run on this workspace, used as
/// the default in examples that don't take CLI args.
pub fn default_ctrl_dir() -> PathBuf {
    PathBuf::from("results/local_test/100c_rho90_ctrl/run_01")
}

/// Path to the local 100c-rho90 soft run.
pub fn default_soft_dir() -> PathBuf {
    PathBuf::from("results/local_test/100c_rho90_soft/run_01")
}

/// Load a trajectory directory into a fully-populated `Context`.
///
/// Reads `<dir>/trajectory.txt` (required) and `<dir>/checkpoint.bin`
/// (loaded if present). All observable Requirements (POSITIONS,
/// TRAJECTORY, CHECKPOINT) are satisfied — examples don't need to
/// reason about which fields they need.
pub fn load_run(dir: &Path) -> Result<Context> {
    let traj_path = dir.join("trajectory.txt");
    let traj = load_trajectory_subsample(&traj_path, 1)
        .with_context(|| format!("load trajectory {}", traj_path.display()))?;
    let positions = Arc::new(unwrap_trajectory(&traj));
    let trajectory = Some(Arc::new(traj));
    let ckpt_path = dir.join("checkpoint.bin");
    let checkpoint = if ckpt_path.exists() {
        Some(Arc::new(load_checkpoint(&ckpt_path)?))
    } else {
        None
    };
    Ok(Context {
        positions,
        trajectory,
        checkpoint,
        params: default_params(),
    })
}

/// Default scientific parameters used by all examples.
/// Matches the `local_smoke.toml`: τ=10000, R=49, v_A=0.01, tagged=cell0.
pub fn default_params() -> RunParams {
    RunParams {
        tau: 10000.0,
        cell_radius: 49.0,
        v_a: 0.01,
        tagged_cells: vec![0],
        soft_cells: vec![],
    }
}

/// Resolve the trajectory dir from CLI argv[1] (if given) or fall back
/// to `default_ctrl_dir()`. Examples invoke this so users can override
/// the dataset.
pub fn run_dir_from_args() -> PathBuf {
    std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_ctrl_dir)
}

/// Pretty-print a label/value pair right-aligned for the example
/// output. Keeps each example's print code one line per metric.
pub fn print_kv(label: &str, value: impl std::fmt::Display) {
    println!("  {:<28} {}", label, value);
}

/// Pretty-print an Observable's output as JSON, truncating long arrays
/// to first/last `head` elements each so per-frame timeseries don't
/// flood the console.
pub fn print_output<T: serde::Serialize>(out: &T) -> Result<()> {
    let value = serde_json::to_value(out)?;
    let truncated = truncate_arrays(value, 5);
    println!("{}", serde_json::to_string_pretty(&truncated)?);
    Ok(())
}

fn truncate_arrays(value: serde_json::Value, head: usize) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Array(arr) if arr.len() > head * 2 + 1 => {
            let mut out = Vec::with_capacity(head * 2 + 1);
            for v in arr.iter().take(head) {
                out.push(truncate_arrays(v.clone(), head));
            }
            out.push(Value::String(format!("... ({} elided) ...", arr.len() - head * 2)));
            for v in arr.iter().skip(arr.len() - head) {
                out.push(truncate_arrays(v.clone(), head));
            }
            Value::Array(out)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(|v| truncate_arrays(v, head)).collect()),
        Value::Object(map) => {
            Value::Object(map.into_iter().map(|(k, v)| (k, truncate_arrays(v, head))).collect())
        }
        other => other,
    }
}
