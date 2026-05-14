//! Run pre-validation.
//!
//! Single source of truth for "is this run analyzable" checks. Called
//! by:
//!   - `cell_analyze check`     — the user-facing diagnostic command.
//!   - `cell_analyze study`     — gates each discovered run before
//!                                figure rendering.
//!   - `cell_analyze snapshot`  — gates the checkpoint/trajectory load
//!                                before rendering.
//!
//! All callers print the same banner, fail with the same exit code
//! discipline (0 = all pass, 1 = any fail), and reuse the same parsing
//! logic. Adding a new check belongs here — never as a fresh
//! `if data.is_nan()` guard in a renderer.
//!
//! The pre-pass also pre-loads the [`Trajectory`] when one is needed so
//! callers can reuse it without paying the parse cost twice.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::analysis::checkpoint::{self, Checkpoint};
use crate::analysis::io::{load_trajectory, unwrap_trajectory, Trajectory};
use crate::analysis::observable::{Context as ObsContext, Observable, RunParams};
use crate::analysis::observables::displacement_velocities::DisplacementVelocities;
use crate::analysis::observables::ln_perimeter::LnPerimeter;
use crate::analysis::observables::msd_palmieri::MsdPalmieri;

/// One named check result. Serialized in the JSON report.
#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Aggregated report for a single run directory.
#[derive(Serialize)]
pub struct ValidationReport {
    pub dir: PathBuf,
    pub findings: Vec<Finding>,
    pub rows_parsed: usize,
    /// Pre-loaded trajectory, populated when `with_observables=true` and
    /// the trajectory parsed successfully. Callers can reuse it instead
    /// of re-loading. Skipped from JSON output (not serializable in a
    /// useful way and large).
    #[serde(skip)]
    pub trajectory: Option<Arc<Trajectory>>,
}

impl ValidationReport {
    pub fn all_pass(&self) -> bool {
        self.findings.iter().all(|f| f.passed)
    }
    pub fn passed_count(&self) -> usize {
        self.findings.iter().filter(|f| f.passed).count()
    }
    pub fn total(&self) -> usize {
        self.findings.len()
    }
}

/// Expectations a caller can impose. `None` means "skip this check".
#[derive(Default, Clone)]
pub struct Expectations {
    pub n_cells: Option<usize>,
    pub frames: Option<usize>,
    pub t_start: Option<f64>,
    pub t_end: Option<f64>,
}

/// Run the full pre-validation pass on `dir`. The directory must
/// contain `trajectory.txt`; `checkpoint.bin` is optional (skipped if
/// absent). When `with_observables=true`, also computes msd_palmieri,
/// displacement_velocities, ln_perimeter and verifies their outputs
/// aren't NaN/Inf/empty.
pub fn validate_run(
    dir: &Path,
    expectations: &Expectations,
    with_observables: bool,
) -> Result<ValidationReport> {
    let mut findings: Vec<Finding> = Vec::new();
    let mut push = |name: &str, passed: bool, detail: String| {
        findings.push(Finding { name: name.to_string(), passed, detail });
    };

    let traj_path = dir.join("trajectory.txt");
    if !traj_path.exists() {
        push("trajectory_exists", false,
             format!("trajectory.txt not found in {}", dir.display()));
        return Ok(ValidationReport {
            dir: dir.to_path_buf(), findings, rows_parsed: 0, trajectory: None,
        });
    }

    // Multi-GPU runs: union rank-0 trajectory with sibling rankN/.
    let mut traj_paths: Vec<PathBuf> = vec![traj_path.clone()];
    for k in 1.. {
        let candidate = dir.join(format!("rank{}", k)).join("trajectory.txt");
        if candidate.exists() {
            traj_paths.push(candidate);
        } else {
            break;
        }
    }
    if traj_paths.len() > 1 {
        eprintln!("validate_run: detected multi-GPU run, reading {} rank trajectories",
                  traj_paths.len());
    }

    // -------- Pass 1: scan trajectory rows for header + sanity --------
    let mut header_fields: HashMap<String, String> = HashMap::new();
    let mut timestamps: Vec<f64> = Vec::new();
    let mut rows_per_t: HashMap<u64, usize> = HashMap::new();
    let mut any_nan = false;
    let mut any_non_numeric = false;
    let mut row_count: usize = 0;

    for tp in &traj_paths {
        let f = std::fs::File::open(tp)
            .with_context(|| format!("opening {}", tp.display()))?;
        for line in BufReader::new(f).lines() {
            let line = match line { Ok(l) => l, Err(_) => continue };
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }
            if trimmed.starts_with('#') {
                for tok in trimmed.split_whitespace() {
                    if let Some((k, v)) = tok.split_once('=') {
                        // Multi-GPU writes N_global / N_local; normalize to N.
                        let key = if k == "N_global" { "N".to_string() } else { k.to_string() };
                        header_fields.entry(key).or_insert_with(|| v.to_string());
                    }
                }
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 4 { continue; }
            let t = match parts[0].parse::<f64>() {
                Ok(v) => v,
                Err(_) => { any_non_numeric = true; continue; }
            };
            if !t.is_finite() { any_nan = true; continue; }
            for idx in [2usize, 3] {
                if idx < parts.len() {
                    match parts[idx].parse::<f64>() {
                        Ok(v) if !v.is_finite() => any_nan = true,
                        Err(_) => any_non_numeric = true,
                        _ => {}
                    }
                }
            }
            let t_bits = t.to_bits();
            *rows_per_t.entry(t_bits).or_insert(0) += 1;
            if rows_per_t[&t_bits] == 1 {
                timestamps.push(t);
            }
            row_count += 1;
        }
    }
    timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // -------- Structural checks --------
    let required_keys = ["N", "Lx", "Ly", "dim", "tau", "v_A"];
    let missing: Vec<&str> = required_keys
        .iter()
        .filter(|k| !header_fields.contains_key(**k))
        .copied()
        .collect();
    push("trajectory_header",
         missing.is_empty(),
         if missing.is_empty() {
             format!("all keys present: {}", required_keys.join(", "))
         } else {
             format!("MISSING keys: {}", missing.join(", "))
         });

    push("trajectory_no_nan",
         !any_nan && !any_non_numeric,
         if any_nan { "NaN/Inf found in data".to_string() }
         else if any_non_numeric { "non-numeric tokens found".to_string() }
         else { "all values finite".to_string() });

    let mut monotonic = true;
    let mut first_bad: Option<(usize, f64, f64)> = None;
    for i in 1..timestamps.len() {
        if timestamps[i] <= timestamps[i - 1] {
            monotonic = false;
            if first_bad.is_none() {
                first_bad = Some((i, timestamps[i - 1], timestamps[i]));
            }
        }
    }
    push("timestamps_monotonic",
         monotonic,
         if monotonic {
             format!("{} unique timestamps strictly increasing", timestamps.len())
         } else {
             let (i, prev, curr) = first_bad.unwrap();
             format!("NON-MONOTONIC at frame {}: {:.6} → {:.6}", i, prev, curr)
         });

    let header_n: Option<usize> = header_fields.get("N").and_then(|v| v.parse().ok());
    let expected_rows_per_frame = expectations.n_cells.or(header_n);
    let mut bad_frame: Option<(f64, usize)> = None;
    if let Some(n) = expected_rows_per_frame {
        for &t in &timestamps {
            let c = rows_per_t[&t.to_bits()];
            if c != n {
                bad_frame = Some((t, c));
                break;
            }
        }
    }
    push("rows_per_frame_consistent",
         bad_frame.is_none(),
         match (expected_rows_per_frame, bad_frame) {
             (None, _) => "skipped (no expected N)".to_string(),
             (Some(n), None) => format!("every frame has {} rows", n),
             (Some(n), Some((t, c))) => {
                 format!("frame t={:.3} has {} rows (expected {})", t, c, n)
             }
         });

    if let Some(ef) = expectations.frames {
        let tol = (ef as f64 * 0.02).max(2.0);
        let diff = (timestamps.len() as f64 - ef as f64).abs();
        push("frame_count",
             diff <= tol,
             format!("got {}, expected {} (tol ±{:.0})", timestamps.len(), ef, tol));
    } else {
        push("frame_count", true, format!("{} frames (no expectation)", timestamps.len()));
    }

    if timestamps.len() >= 3 {
        let intervals: Vec<f64> = (1..timestamps.len())
            .map(|i| timestamps[i] - timestamps[i - 1])
            .collect();
        let mean: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let max_dev = intervals.iter().map(|&x| (x - mean).abs()).fold(0.0f64, f64::max);
        let rel_dev = if mean > 0.0 { max_dev / mean } else { 1.0 };
        push("frame_interval_uniform",
             rel_dev < 0.10,
             format!("mean Δt = {:.3}, max deviation {:.1}%", mean, 100.0 * rel_dev));
    }

    if let Some(ts_exp) = expectations.t_start {
        if let Some(&ts_got) = timestamps.first() {
            let tol = (ts_exp.abs() * 0.01).max(1.0);
            push("t_start",
                 (ts_got - ts_exp).abs() <= tol,
                 format!("got {:.3}, expected {:.3} (tol ±{:.1})", ts_got, ts_exp, tol));
        }
    }
    if let Some(te_exp) = expectations.t_end {
        if let Some(&te_got) = timestamps.last() {
            let tol = (te_exp.abs() * 0.01).max(1.0);
            push("t_end",
                 (te_got - te_exp).abs() <= tol,
                 format!("got {:.3}, expected {:.3} (tol ±{:.1})", te_got, te_exp, tol));
        }
    }

    // -------- Checkpoint --------
    let mut ckpt_tau_from_file: Option<f64> = None;
    let mut ckpt_dt_from_file: Option<f64> = None;
    let ckpt_path = dir.join("checkpoint.bin");
    if ckpt_path.exists() {
        match checkpoint::load_checkpoint(&ckpt_path) {
            Ok(ckpt) => {
                ckpt_tau_from_file = Some(ckpt.params.tau as f64);
                ckpt_dt_from_file = Some(ckpt.params.dt as f64);
                report_checkpoint(&ckpt, &timestamps, expected_rows_per_frame, &mut push);
            }
            Err(e) => push("checkpoint_consistency", false,
                           format!("failed to parse checkpoint: {}", e)),
        }
    } else {
        push("checkpoint_consistency", true, "no checkpoint.bin (skipped)".to_string());
    }

    // Cross-checks: trajectory header vs checkpoint scalars.
    let traj_tau: Option<f64> = header_fields.get("tau").and_then(|v| v.parse().ok());
    let traj_dt: Option<f64> = header_fields.get("dt").and_then(|v| v.parse().ok());
    if let (Some(tt), Some(ct)) = (traj_tau, ckpt_tau_from_file) {
        let rel = ((tt - ct).abs() / ct.max(1e-9)) * 100.0;
        push("tau_traj_vs_ckpt",
             rel < 0.1,
             if rel < 0.1 {
                 format!("traj τ={:.4} matches ckpt τ={:.4}", tt, ct)
             } else {
                 format!("MISMATCH: traj τ={:.4} vs ckpt τ={:.4} ({:.1}% diff) — \
                          likely --tau passed wrong on a chain step",
                         tt, ct, rel)
             });
    }
    if let (Some(td), Some(cd)) = (traj_dt, ckpt_dt_from_file) {
        let rel = ((td - cd).abs() / cd.max(1e-9)) * 100.0;
        push("dt_traj_vs_ckpt",
             rel < 0.1,
             if rel < 0.1 {
                 format!("traj dt={:.6} matches ckpt dt={:.6}", td, cd)
             } else {
                 format!("MISMATCH: traj dt={:.6} vs ckpt dt={:.6} ({:.1}% diff)",
                         td, cd, rel)
             });
    }

    // Metadata banner.
    {
        let traj_n = header_fields.get("N").map(|s| s.as_str()).unwrap_or("?");
        let traj_va = header_fields.get("v_A").map(|s| s.as_str()).unwrap_or("?");
        let traj_lx = header_fields.get("Lx").map(|s| s.as_str()).unwrap_or("?");
        let traj_ly = header_fields.get("Ly").map(|s| s.as_str()).unwrap_or("?");
        push("metadata", true,
             format!("traj: v_A={} τ={} dt={} N={} Lx={} Ly={}",
                     traj_va,
                     traj_tau.map(|v| format!("{:.4}", v)).unwrap_or_else(|| "?".into()),
                     traj_dt.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "?".into()),
                     traj_n, traj_lx, traj_ly));
    }
    if let (Some(ct), Some(cd)) = (ckpt_tau_from_file, ckpt_dt_from_file) {
        push("ckpt_metadata", true, format!("ckpt: τ={:.4} dt={:.6}", ct, cd));
    }

    // -------- Observable pass --------
    let trajectory_arc = if with_observables {
        match run_observable_pass(&traj_paths[0], &mut push) {
            Ok(traj) => Some(traj),
            Err(e) => {
                push("observable_pass", false, format!("error: {}", e));
                None
            }
        }
    } else {
        None
    };

    Ok(ValidationReport {
        dir: dir.to_path_buf(),
        findings,
        rows_parsed: row_count,
        trajectory: trajectory_arc,
    })
}

fn report_checkpoint(
    ckpt: &Checkpoint,
    timestamps: &[f64],
    expected_rows_per_frame: Option<usize>,
    push: &mut dyn FnMut(&str, bool, String),
) {
    let ckpt_ver = ckpt.header.version;
    let ckpt_t = ckpt.header.time;
    let ckpt_n_local = ckpt.header.num_cells;
    let ckpt_n_global = ckpt.header.num_cells_global;
    let mut ok = true;
    let mut msgs: Vec<String> = Vec::new();
    let ckpt_n_for_check = if ckpt_ver >= 8 { ckpt_n_global } else { ckpt_n_local };
    if ckpt_ver >= 8 && ckpt_n_local != ckpt_n_global {
        msgs.push(format!("v{} step_t={:.3} N={} (local={}, multi-rank)",
                          ckpt_ver, ckpt_t, ckpt_n_global, ckpt_n_local));
    } else {
        msgs.push(format!("v{} step_t={:.3} N={}", ckpt_ver, ckpt_t, ckpt_n_local));
    }
    if let Some(&last_t) = timestamps.last() {
        let tol = (ckpt_t.abs() * 0.01).max(1.0);
        if (ckpt_t - last_t).abs() > tol {
            ok = false;
            msgs.push(format!("checkpoint t={:.3} disagrees with last trajectory t={:.3}",
                              ckpt_t, last_t));
        }
    }
    if let Some(n) = expected_rows_per_frame {
        if ckpt_n_for_check as usize != n {
            ok = false;
            msgs.push(format!("checkpoint N={} disagrees with expected {}",
                              ckpt_n_for_check, n));
        }
    }
    push("checkpoint_consistency", ok, msgs.join("; "));
}

/// Load trajectory, run msd_palmieri / displacement_velocities /
/// ln_perimeter, report whether outputs are finite + non-empty. Returns
/// the pre-loaded [`Trajectory`] on success so callers can reuse it.
fn run_observable_pass(
    traj_path: &Path,
    push: &mut dyn FnMut(&str, bool, String),
) -> Result<Arc<Trajectory>> {
    let traj = load_trajectory(traj_path)
        .with_context(|| format!("loading {}", traj_path.display()))?;
    let traj_arc = Arc::new(traj);
    let tau = traj_arc.params.tau.unwrap_or(10000.0);
    let positions = Arc::new(unwrap_trajectory(&traj_arc));
    if positions.n_times < 4 {
        push("observable_pass", false,
             format!("trajectory has only {} time points — observables need ≥4",
                     positions.n_times));
        return Ok(traj_arc);
    }
    let ctx = ObsContext {
        positions,
        trajectory: Some(traj_arc.clone()),
        checkpoint: None,
        params: RunParams { tau, ..Default::default() },
    };

    match MsdPalmieri.compute(&ctx) {
        Ok(out) => {
            let n_nan = out.msd_t_cell.iter().filter(|x| !x.is_finite()).count()
                + out.msd_t_pop.iter().filter(|x| !x.is_finite()).count();
            let d_eff_finite = out.d_eff_cell.is_finite() && out.d_eff_pop.is_finite();
            let ok = !out.lag_tau.is_empty() && n_nan == 0 && d_eff_finite;
            push("obs:msd_palmieri", ok,
                 if ok {
                     format!("{} lags, D_eff_c0={:.4e}, D_eff_pop={:.4e}",
                             out.lag_tau.len(), out.d_eff_cell, out.d_eff_pop)
                 } else if out.lag_tau.is_empty() {
                     "empty output (τ ≫ trajectory duration?)".into()
                 } else if !d_eff_finite {
                     format!("D_eff non-finite: cell={}, pop={}",
                             out.d_eff_cell, out.d_eff_pop)
                 } else {
                     format!("{} NaN/Inf entries in MSD curves", n_nan)
                 });
        }
        Err(e) => push("obs:msd_palmieri", false, format!("compute failed: {}", e)),
    }

    match DisplacementVelocities.compute(&ctx) {
        Ok(out) => {
            let mean_speed = if out.speeds.is_empty() {
                f64::NAN
            } else {
                out.speeds.iter().sum::<f64>() / out.speeds.len() as f64
            };
            let n_nan = out.speeds.iter().filter(|x| !x.is_finite()).count();
            let ok = !out.speeds.is_empty() && n_nan == 0 && mean_speed.is_finite();
            push("obs:displacement_velocities", ok,
                 if ok {
                     format!("{} samples, ⟨|v|⟩={:.4e}", out.speeds.len(), mean_speed)
                 } else if out.speeds.is_empty() {
                     "empty output".into()
                 } else {
                     format!("{} NaN/Inf samples, mean={}", n_nan, mean_speed)
                 });
        }
        Err(e) => push("obs:displacement_velocities", false,
                       format!("compute failed: {}", e)),
    }

    match LnPerimeter.compute(&ctx) {
        Ok(out) => {
            let n_nan = out.series.iter().filter(|x| !x.is_finite()).count();
            let mean = if out.series.is_empty() {
                f64::NAN
            } else {
                out.series.iter().sum::<f64>() / out.series.len() as f64
            };
            let ok = !out.series.is_empty() && n_nan == 0 && mean.is_finite() && mean > 1e-6;
            push("obs:ln_perimeter", ok,
                 if ok {
                     format!("{} samples, ⟨L_n⟩={:.4}", out.series.len(), mean)
                 } else if out.series.is_empty() {
                     "empty output".into()
                 } else if mean <= 1e-6 {
                     "all-zero output (perimeter data missing?)".into()
                 } else {
                     format!("{} NaN/Inf samples", n_nan)
                 });
        }
        Err(e) => push("obs:ln_perimeter", false, format!("compute failed: {}", e)),
    }

    Ok(traj_arc)
}

/// Pretty-print a report to stdout. Used by both `check` directly and
/// the auto-validation pre-pass in `study`/`snapshot`.
pub fn print_report(report: &ValidationReport) {
    println!("=== cell_analyze check: {} ===", report.dir.display());
    for f in &report.findings {
        let mark = if f.passed { "PASS" } else { "FAIL" };
        println!("  [{}] {:<28} {}", mark, f.name, f.detail);
    }
    println!("--- rows parsed: {} ---", report.rows_parsed);
    println!("=== {} / {} checks passed ===",
             report.passed_count(), report.total());
}
