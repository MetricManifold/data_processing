//! Declarative study TOML executor.
//!
//! A study TOML wires together:
//!   1. discovery (find runs)
//!   2. observables (compute metrics on each run)
//!   3. aggregate ops (groupby, mean_stderr, sweep, pair_ratio)
//!   4. figures (multi-panel SVGs)
//!
//! The schema is intentionally minimal in this first cut. Phase 6
//! supports the **sweep** workflow needed for FSS and Phase 3A pairwise:
//!
//! ```toml
//! [study]
//! name       = "Phase 3A pairwise"
//! output_dir = "phase3a_results"
//!
//! [discovery]
//! pattern          = "phase3a/d_{d:f64}R/run_{rep:int}"
//! trajectory_name  = "trajectory.txt"
//!
//! [observables]
//! compute     = ["msd"]
//! tau         = 10000.0
//! cell_radius = 49.0
//!
//! [[aggregate]]
//! op   = "groupby"
//! vars = ["d"]
//! into = "by_d"
//!
//! [[aggregate]]
//! op      = "mean_stderr"
//! input   = "by_d"
//! metrics = ["msd_lag1"]
//! into    = "by_d_summary"
//!
//! [[aggregate]]
//! op    = "sweep"
//! axis  = "d"
//! input = "by_d_summary"
//! into  = "phase3a_curve"
//!
//! [[figure]]
//! output = "phase3a_msd_vs_d.svg"
//! width  = 600
//! height = 400
//! layout = [1, 1]
//! panels = [
//!   { type = "metric_vs_x", input = "phase3a_curve",
//!     metric = "msd_lag1", title = "MSD(Δt=1) vs separation" }
//! ]
//! ```
//!
//! Aggregate ops are executed in order. Each writes its output into a
//! named slot in the [`Workspace`]; subsequent ops read from named
//! inputs. Figures consume slots by name and pass typed data to panels.
//!
//! Built-in metric extractors are pre-registered for the observables we
//! ship; new extractors can be added in [`metric_registry`].

use anyhow::{anyhow, Context as _, Result};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use plotters::backend::SVGBackend;
use plotters::prelude::*;

use super::aggregate::{
    GroupBy, GroupSummary, MeanStderr, MetricExtractor, PairRatio, PairResult, Sweep, SweepCurve,
};
use super::analyze_run::{analyze_run, AnalyzePlan, RunAnalysis};
use super::discovery::{discover, DiscoveryRule};
use super::observable::RunParams;
use super::observables::{register_builtin, ErasedObservable};
use super::panels::layout;
use super::panels::sweep::MetricVsX;
use super::panels::{Panel, PanelOpts};

// ---------------------------------------------------------------------------
// TOML schema
// ---------------------------------------------------------------------------
#[derive(Debug, Deserialize)]
pub struct StudyToml {
    pub study: StudyMeta,
    pub discovery: DiscoveryToml,
    pub observables: ObservablesToml,
    #[serde(default, rename = "aggregate")]
    pub aggregates: Vec<AggregateToml>,
    #[serde(default, rename = "figure")]
    pub figures: Vec<FigureToml>,
}

#[derive(Debug, Deserialize)]
pub struct StudyMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub output_dir: String,
}

#[derive(Debug, Deserialize)]
pub struct DiscoveryToml {
    pub pattern: String,
    #[serde(default = "default_traj_name")]
    pub trajectory_name: String,
    #[serde(default)]
    pub checkpoint_name: Option<String>,
}

fn default_traj_name() -> String {
    "trajectory.txt".to_string()
}

#[derive(Debug, Deserialize)]
pub struct ObservablesToml {
    pub compute: Vec<String>,
    #[serde(default = "default_tau")]
    pub tau: f64,
    #[serde(default = "default_cell_radius")]
    pub cell_radius: f64,
    #[serde(default = "default_v_a")]
    pub v_a: f64,
    #[serde(default)]
    pub tagged_cells: Vec<u32>,
    #[serde(default)]
    pub soft_cells: Vec<u32>,
    #[serde(default = "default_subsample")]
    pub subsample: usize,
}

fn default_tau() -> f64 {
    10000.0
}
fn default_cell_radius() -> f64 {
    49.0
}
fn default_v_a() -> f64 {
    0.01
}
fn default_subsample() -> usize {
    1
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op")]
#[serde(rename_all = "snake_case")]
pub enum AggregateToml {
    Groupby {
        vars: Vec<String>,
        into: String,
    },
    MeanStderr {
        input: String,
        metrics: Vec<String>,
        into: String,
    },
    Sweep {
        axis: String,
        input: String,
        into: String,
    },
    PairRatio {
        pair_var: String,
        numerator: String,
        denominator: String,
        input: String,
        into: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct FigureToml {
    pub output: String,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_layout")]
    pub layout: [usize; 2],
    #[serde(default)]
    pub title: Option<String>,
    pub panels: Vec<PanelToml>,
}

fn default_width() -> u32 {
    600
}
fn default_height() -> u32 {
    400
}
fn default_layout() -> [usize; 2] {
    [1, 1]
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum PanelToml {
    MetricVsX {
        input: String,
        metric: String,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        x_label: Option<String>,
        #[serde(default)]
        y_label: Option<String>,
        #[serde(default)]
        x_range: Option<[f64; 2]>,
        #[serde(default)]
        y_range: Option<[f64; 2]>,
        #[serde(default)]
        h_line: Option<f64>,
    },
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------
/// A small typed slot store keyed by name. Each aggregate op writes
/// into one slot; figures and downstream ops read by name. Slots are
/// strongly typed via the enum so panel rendering pulls out the right
/// shape.
#[derive(Default)]
pub struct Workspace {
    slots: BTreeMap<String, Slot>,
}

pub enum Slot {
    Groups(Vec<GroupOwned>),
    Summaries(Vec<GroupSummary>),
    Curve(SweepCurve),
    Pairs(Vec<PairResult>),
}

/// Owned variant of [`super::aggregate::Group`] (the borrowed form
/// can't live in a slot because slots outlive the aggregate call).
pub struct GroupOwned {
    pub key: String,
    pub variables: BTreeMap<String, super::discovery::ScalarValue>,
    pub member_indices: Vec<usize>,
}

impl Workspace {
    pub fn insert(&mut self, name: &str, slot: Slot) {
        self.slots.insert(name.to_string(), slot);
    }
    pub fn get(&self, name: &str) -> Result<&Slot> {
        self.slots
            .get(name)
            .ok_or_else(|| anyhow!("workspace slot `{}` not found", name))
    }
}

// ---------------------------------------------------------------------------
// Metric registry
// ---------------------------------------------------------------------------
/// Map metric name → extractor closure. New observables register their
/// metrics here. For phase 6 we ship just MSD's basic metrics.
pub fn metric_registry() -> BTreeMap<&'static str, MetricExtractor> {
    use crate::analysis::v2::observables::msd::Msd;
    let mut m: BTreeMap<&'static str, MetricExtractor> = BTreeMap::new();
    m.insert("msd_lag1", crate::v2_metric!(Msd, |out| {
        out.cell0_values.first().copied().unwrap_or(f64::NAN)
    }));
    m.insert("msd_pop_lag1", crate::v2_metric!(Msd, |out| {
        out.values.first().copied().unwrap_or(f64::NAN)
    }));
    m
}

// ---------------------------------------------------------------------------
// Observable registry (filter to those named in TOML)
// ---------------------------------------------------------------------------
fn observables_for_request(requested: &[String]) -> Result<Vec<Box<dyn ErasedObservable>>> {
    let all = register_builtin();
    let mut out = Vec::new();
    for name in requested {
        let bare = name.split('(').next().unwrap_or(name).trim();
        let found = all.iter().find(|o| o.id() == bare).is_some();
        if !found {
            return Err(anyhow!("unknown observable `{}`", name));
        }
    }
    for o in all {
        if requested.iter().any(|n| n.split('(').next().unwrap_or(n).trim() == o.id()) {
            out.push(o);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// run_study (top-level)
// ---------------------------------------------------------------------------
pub fn run_study(toml_path: &Path, base_dir: &Path) -> Result<()> {
    let raw = std::fs::read_to_string(toml_path)
        .with_context(|| format!("read TOML {}", toml_path.display()))?;
    let cfg: StudyToml = toml::from_str(&raw).with_context(|| "parse TOML")?;

    eprintln!("Study: {}", cfg.study.name);
    eprintln!("  output_dir: {}", cfg.study.output_dir);

    // 1. Discover runs.
    let rule = DiscoveryRule::new(
        &cfg.discovery.pattern,
        &cfg.discovery.trajectory_name,
        cfg.discovery.checkpoint_name.as_deref(),
    )?;
    let specs = discover(base_dir, &rule)?;
    eprintln!("Discovered {} runs", specs.len());
    if specs.is_empty() {
        return Err(anyhow!(
            "no runs matched `{}` under `{}`",
            cfg.discovery.pattern,
            base_dir.display()
        ));
    }

    // 2. Per-run analysis (parallel).
    let observables = observables_for_request(&cfg.observables.compute)?;
    let params = RunParams {
        tau: cfg.observables.tau,
        cell_radius: cfg.observables.cell_radius,
        v_a: cfg.observables.v_a,
        tagged_cells: if cfg.observables.tagged_cells.is_empty() {
            vec![0]
        } else {
            cfg.observables.tagged_cells.clone()
        },
        soft_cells: cfg.observables.soft_cells.clone(),
    };
    let plan = AnalyzePlan {
        observables: &observables,
        params,
        subsample: cfg.observables.subsample,
    };

    eprintln!("Analyzing runs...");
    let runs: Vec<RunAnalysis> = specs
        .par_iter()
        .map(|spec| analyze_run(spec, &plan))
        .collect::<Result<Vec<_>>>()?;
    eprintln!("  done ({} runs analyzed)", runs.len());

    // 3. Run aggregate pipeline.
    let mut ws = Workspace::default();
    let metrics = metric_registry();
    for op in &cfg.aggregates {
        execute_op(op, &runs, &metrics, &mut ws)?;
    }

    // 4. Render figures.
    let out_dir = Path::new(&cfg.study.output_dir);
    let out_dir = if out_dir.is_absolute() {
        out_dir.to_path_buf()
    } else {
        base_dir.join(out_dir)
    };
    std::fs::create_dir_all(&out_dir).ok();
    for fig in &cfg.figures {
        render_figure(fig, &ws, &out_dir)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// execute_op
// ---------------------------------------------------------------------------
fn execute_op(
    op: &AggregateToml,
    runs: &[RunAnalysis],
    metrics: &BTreeMap<&'static str, MetricExtractor>,
    ws: &mut Workspace,
) -> Result<()> {
    match op {
        AggregateToml::Groupby { vars, into } => {
            let var_refs: Vec<&str> = vars.iter().map(String::as_str).collect();
            let groups = GroupBy { vars: &var_refs }.run(runs);
            // Convert to owned form keyed by member indices.
            let mut owned = Vec::with_capacity(groups.len());
            for g in &groups {
                let indices: Vec<usize> = g
                    .members
                    .iter()
                    .map(|m| {
                        runs.iter()
                            .position(|r| std::ptr::eq(r, *m))
                            .expect("member not in runs slice")
                    })
                    .collect();
                owned.push(GroupOwned {
                    key: g.key.clone(),
                    variables: g.variables.clone(),
                    member_indices: indices,
                });
            }
            ws.insert(into, Slot::Groups(owned));
            Ok(())
        }
        AggregateToml::MeanStderr {
            input,
            metrics: metric_names,
            into,
        } => {
            let groups_owned = match ws.get(input)? {
                Slot::Groups(g) => g,
                _ => return Err(anyhow!("mean_stderr expects `groups`-shaped input `{}`", input)),
            };
            // Reconstruct borrowed groups for the aggregator.
            let groups_borrowed: Vec<super::aggregate::Group<'_>> = groups_owned
                .iter()
                .map(|g| super::aggregate::Group {
                    key: g.key.clone(),
                    variables: g.variables.clone(),
                    members: g.member_indices.iter().map(|&i| &runs[i]).collect(),
                })
                .collect();
            // Build extractor list from registry.
            let mut metric_pairs: Vec<(&str, MetricExtractor)> = Vec::new();
            for name in metric_names {
                let ex = metrics
                    .get(name.as_str())
                    .ok_or_else(|| anyhow!("unknown metric `{}`", name))?;
                metric_pairs.push((name.as_str(), reborrow(ex)));
            }
            let summaries = MeanStderr {
                metrics: &metric_pairs,
            }
            .run(&groups_borrowed);
            ws.insert(into, Slot::Summaries(summaries));
            Ok(())
        }
        AggregateToml::Sweep { axis, input, into } => {
            let summaries = match ws.get(input)? {
                Slot::Summaries(s) => s,
                _ => return Err(anyhow!("sweep expects `summaries` input `{}`", input)),
            };
            let curve = Sweep { axis }.run(summaries)?;
            ws.insert(into, Slot::Curve(curve));
            Ok(())
        }
        AggregateToml::PairRatio {
            pair_var,
            numerator,
            denominator,
            input,
            into,
        } => {
            let summaries = match ws.get(input)? {
                Slot::Summaries(s) => s,
                _ => return Err(anyhow!("pair_ratio expects `summaries` input `{}`", input)),
            };
            let pairs = PairRatio {
                pair_var,
                numerator,
                denominator,
            }
            .run(summaries);
            ws.insert(into, Slot::Pairs(pairs));
            Ok(())
        }
    }
}

/// Re-wrap a borrowed `MetricExtractor` reference back into a Box for
/// the aggregator's API. The closure inside is `Fn`, not `FnMut`, so
/// re-boxing as a thin wrapper is cheap and safe.
fn reborrow(ex: &MetricExtractor) -> MetricExtractor {
    let cloned: &'static MetricExtractor = unsafe {
        // SAFETY: the registry outlives any aggregate call within
        // run_study; we only use the reborrow inside this function's
        // call chain.
        std::mem::transmute::<&MetricExtractor, &'static MetricExtractor>(ex)
    };
    Box::new(move |bag| (cloned)(bag))
}

// ---------------------------------------------------------------------------
// render_figure
// ---------------------------------------------------------------------------
fn render_figure(fig: &FigureToml, ws: &Workspace, out_dir: &Path) -> Result<()> {
    let path: PathBuf = out_dir.join(&fig.output);
    let backend = SVGBackend::new(&path, (fig.width, fig.height));
    let area = backend.into_drawing_area();
    area.fill(&WHITE)?;

    let title_h = if fig.title.is_some() { 30 } else { 0 };
    let (title_area, panels_area) = if title_h > 0 {
        let (t, c) = layout::grid(&area, fig.layout[0], fig.layout[1], title_h);
        if let Some(t_text) = &fig.title {
            t.titled(t_text, ("sans-serif", 18))?;
        }
        (Some(t), c)
    } else {
        let cells = area.split_evenly((fig.layout[0], fig.layout[1]));
        (None, cells)
    };
    let _ = title_area;

    for (idx, panel_cfg) in fig.panels.iter().enumerate() {
        if idx >= panels_area.len() {
            break;
        }
        let cell = &panels_area[idx];
        match panel_cfg {
            PanelToml::MetricVsX {
                input,
                metric,
                title,
                x_label,
                y_label,
                x_range,
                y_range,
                h_line,
            } => {
                let curve = match ws.get(input)? {
                    Slot::Curve(c) => c,
                    _ => return Err(anyhow!("metric_vs_x expects a curve in slot `{}`", input)),
                };
                let panel = MetricVsX {
                    metric,
                    h_line: *h_line,
                };
                let opts = PanelOpts {
                    title: title.clone(),
                    x_label: x_label.clone(),
                    y_label: y_label.clone(),
                    x_range: x_range.map(|r| (r[0], r[1])),
                    y_range: y_range.map(|r| (r[0], r[1])),
                    log_x: false,
                    log_y: false,
                };
                panel.render(cell, curve, &opts)?;
            }
        }
    }
    eprintln!("  wrote {}", path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_synthetic_traj(path: &Path, drift_v: f64) {
        let mut s = String::new();
        s.push_str("# v_A=0.01 N=1 Lx=1000.0 Ly=1000.0 dim=2 tau=10000.0\n");
        for t in 0..20 {
            let time = t as f64;
            let x = drift_v * time;
            s.push_str(&format!(
                "{:.6}\t0\t{:.6}\t0.0\t{:.3}\t0.0\t1.0\t0.0\t0.0\t0.01\t1.0\t100.0\n",
                time, x, drift_v
            ));
        }
        fs::write(path, s).unwrap();
    }

    #[test]
    fn end_to_end_phase3a_skeleton() {
        // 1. Build a phase3a-shaped tree with synthetic trajectories.
        let root = std::env::temp_dir().join("v2_studies_e2e");
        let _ = fs::remove_dir_all(&root);
        for d in &[2.0_f64, 4.0, 8.0] {
            for rep in 1..=3 {
                let dir = root.join("phase3a").join(format!("d_{}R", *d as i64))
                    .join(format!("run_{:02}", rep));
                fs::create_dir_all(&dir).unwrap();
                let drift = 0.01_f64 * (1.0_f64 - (-d / 6.0).exp());
                write_synthetic_traj(&dir.join("trajectory.txt"), drift);
            }
        }

        // 2. Write the TOML.
        let toml_text = r#"
[study]
name = "Phase 3A test"
output_dir = "phase3a_results"

[discovery]
pattern = "phase3a/d_{d:int}R/run_{rep:int}"

[observables]
compute = ["msd"]

[[aggregate]]
op = "groupby"
vars = ["d"]
into = "by_d"

[[aggregate]]
op = "mean_stderr"
input = "by_d"
metrics = ["msd_lag1"]
into = "by_d_summary"

[[aggregate]]
op = "sweep"
axis = "d"
input = "by_d_summary"
into = "phase3a_curve"

[[figure]]
output = "phase3a_msd_vs_d.svg"
width = 600
height = 400
layout = [1, 1]
title = "Phase 3A: MSD(Δt=1) vs separation"
panels = [
  { type = "metric_vs_x", input = "phase3a_curve", metric = "msd_lag1", title = "MSD@1 vs d" }
]
"#;
        let toml_path = root.join("phase3a.toml");
        fs::write(&toml_path, toml_text).unwrap();

        // 3. Run the study.
        run_study(&toml_path, &root).expect("run_study");

        // 4. Verify the figure exists.
        let svg = root.join("phase3a_results").join("phase3a_msd_vs_d.svg");
        assert!(svg.exists(), "figure not produced: {}", svg.display());

        let _ = fs::remove_dir_all(&root);
    }
}
