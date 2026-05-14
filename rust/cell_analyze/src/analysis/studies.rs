//! Declarative study TOML executor.
//!
//! A study TOML wires together:
//!   1. discovery (find runs)
//!   2. observables (compute metrics on each run)
//!   3. aggregate ops (groupby, mean_stderr, sweep, pair_ratio)
//!   4. figures (multi-panel SVGs)
//!
//! The schema is intentionally minimal. The current set of operators
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
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use plotters::backend::SVGBackend;
use plotters::prelude::*;

use super::aggregate::{
    GroupBy, GroupSummary, MeanStderr, MetricExtractor, PairRatio, PairResult, Sweep, SweepCurve,
};
use super::analyze_run::{analyze_run, AnalyzePlan, RunAnalysis, RunAnalysisJson};
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
    /// Pair *individual runs* (not aggregated summaries) by the value
    /// of `pair_var`. Runs that agree on every variable except
    /// `pair_var` form one [`RunPair`]. Used by diagnostic/comparison
    /// figures that need direct access to each run's observables.
    PairRuns {
        pair_var: String,
        numerator: String,
        denominator: String,
        /// Optional filter: only emit pairs whose `pair_var` residual
        /// (e.g. seed) is in this list. Empty = emit all.
        #[serde(default)]
        seeds: Vec<String>,
        /// Variable name used for the seed filter (default: "seed").
        #[serde(default = "default_seed_var")]
        seed_var: String,
        into: String,
    },
    /// Pick a single run that matches a set of variable filters.
    /// Filters are `var = value` equality checks. Errors if zero or
    /// multiple runs match.
    SingleRun {
        /// Map of variable name → required value. Stringly-typed for
        /// TOML simplicity; values are matched against the
        /// `ScalarValue::to_string()` of each run's variables.
        #[serde(default)]
        filter: BTreeMap<String, String>,
        /// Optional display label. Default: stringified filter.
        #[serde(default)]
        label: Option<String>,
        into: String,
    },
    /// Bundle N runs into an Overlay slot for N-way layered plots.
    /// `vary` names the variable whose distinct values become the
    /// series labels (e.g. `vary = "gamma_c"`). `filter` further
    /// constrains which runs are eligible.
    Overlay {
        vary: String,
        #[serde(default)]
        filter: BTreeMap<String, String>,
        into: String,
    },
}

fn default_seed_var() -> String {
    "seed".to_string()
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
    /// Common shape for all pair panels (speed_bursts, gvi, ln_*, msd_t,
    /// deff_bar, summary). The `subtype` discriminates the actual
    /// panel; common knobs (title, ranges) are shared.
    Pair {
        /// `speed_bursts | gvi | ln_timeseries | ln_histogram | msd_t |
        /// deff_bar | summary`.
        subtype: String,
        input: String,
        /// Which RunPair index inside the slot to render (default: 0).
        #[serde(default)]
        pair_index: usize,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        x_range: Option<[f64; 2]>,
        #[serde(default)]
        y_range: Option<[f64; 2]>,
        /// Speed-bursts panel: max speed for the y axis (default 0.02).
        #[serde(default)]
        speed_max: Option<f64>,
        /// MSD/Δt panel: max lag in τ (default 8).
        #[serde(default)]
        msd_lag_max: Option<f64>,
        /// G(v_i) panel: x_max for |v| (default 0.022).
        #[serde(default)]
        gvi_x_max: Option<f64>,
        /// L_n histogram panel: number of bins (default 40).
        #[serde(default)]
        bins: Option<usize>,
    },
    /// Single-run panel. `input` references a `single_run` slot.
    Single {
        /// `msd | gvi | ln_timeseries | speed_bursts`.
        subtype: String,
        input: String,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        x_range: Option<[f64; 2]>,
        #[serde(default)]
        y_range: Option<[f64; 2]>,
        #[serde(default)]
        speed_max: Option<f64>,
        #[serde(default)]
        msd_lag_max: Option<f64>,
        #[serde(default)]
        gvi_x_max: Option<f64>,
        /// G(v_i) panel: enable Eq.5 fit (default true).
        #[serde(default)]
        fit_eq5: Option<bool>,
        /// MSD panel: also draw the population MSD (default true).
        #[serde(default)]
        show_population: Option<bool>,
    },
    /// Overlay panel: N runs colored by series. `input` references an
    /// `overlay` slot.
    Overlay {
        /// `msd | gvi | ln_timeseries`.
        subtype: String,
        input: String,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        x_range: Option<[f64; 2]>,
        #[serde(default)]
        y_range: Option<[f64; 2]>,
        #[serde(default)]
        msd_lag_max: Option<f64>,
        #[serde(default)]
        gvi_x_max: Option<f64>,
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
    /// Run pairs identified by a pair variable (e.g. soft vs ctrl at
    /// the same seed). Stored as indices into the runs slice; the
    /// figure renderer resolves them via the runs vector.
    RunPairs(Vec<RunPair>),
    /// A single run, identified by index. Used by single_run panels
    /// (one trajectory at a time).
    SingleRun(SingleRunRef),
    /// N-way overlay: a list of runs to render on shared axes with one
    /// colored series per run (e.g. γ_c sweep).
    Overlay(OverlayRef),
}

/// Reference to one selected run.
#[derive(Clone, Debug, Serialize)]
pub struct SingleRunRef {
    pub index: usize,
    pub label: String,
}

/// A bundle of N runs to be plotted overlay-style.
#[derive(Clone, Debug, Serialize)]
pub struct OverlayRef {
    pub series: Vec<OverlaySeriesRef>,
}

/// One series in an overlay: index into runs + display label + the
/// variable value that distinguishes it (used for the legend).
#[derive(Clone, Debug, Serialize)]
pub struct OverlaySeriesRef {
    pub run_index: usize,
    pub label: String,
}

/// One paired comparison: numerator and denominator runs identified by
/// indices into the `runs: &[RunAnalysis]` passed to the figure
/// renderer. `key` and `variables` describe the residual variables
/// (everything except the pair variable).
#[derive(Clone, Debug, Serialize)]
pub struct RunPair {
    pub key: String,
    pub variables: BTreeMap<String, super::discovery::ScalarValue>,
    pub numerator_idx: usize,
    pub denominator_idx: usize,
    pub numerator_label: String,
    pub denominator_label: String,
}

/// Owned variant of [`super::aggregate::Group`] (the borrowed form
/// can't live in a slot because slots outlive the aggregate call).
#[derive(Clone, Debug, Serialize)]
pub struct GroupOwned {
    pub key: String,
    pub variables: BTreeMap<String, super::discovery::ScalarValue>,
    pub member_indices: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct StudyResultJson {
    study_name: String,
    study_description: String,
    config_path: String,
    data_dir: String,
    output_dir: String,
    runs: Vec<RunAnalysisJson>,
    slots: BTreeMap<String, serde_json::Value>,
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
/// metrics here.
pub fn metric_registry() -> BTreeMap<&'static str, MetricExtractor> {
    use crate::analysis::observables::msd::Msd;
    use crate::analysis::observables::msd_palmieri::MsdPalmieri;
    let mut m: BTreeMap<&'static str, MetricExtractor> = BTreeMap::new();
    m.insert("msd_lag1", crate::metric!(Msd, |out| {
        out.cell0_values.first().copied().unwrap_or(f64::NAN)
    }));
    m.insert("msd_pop_lag1", crate::metric!(Msd, |out| {
        out.values.first().copied().unwrap_or(f64::NAN)
    }));
    m.insert("deff_palmieri", crate::metric!(MsdPalmieri, |out| {
        out.d_eff_cell
    }));
    m.insert("deff_pop_palmieri", crate::metric!(MsdPalmieri, |out| {
        out.d_eff_pop
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
pub fn run_study(toml_path: &Path, base_dir: &Path, skip_validation: bool) -> Result<()> {
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

    // 1b. Validation pre-pass: each run gets the same checks as
    // `cell_analyze check --with-observables`. Runs that fail the
    // structural or observable checks are dropped from the study with
    // a clear warning; a fully empty result aborts. Pass
    // `--skip-validation` to bypass entirely. The pre-pass also
    // pre-loads each run's trajectory so we don't pay the parse cost
    // twice.
    let validated: Vec<(crate::analysis::discovery::RunSpec, Option<std::sync::Arc<crate::analysis::io::Trajectory>>)> = if skip_validation {
        eprintln!("  (skipping validation pre-pass)");
        specs.into_iter().map(|s| (s, None)).collect()
    } else {
        use crate::analysis::precheck::{validate_run, print_report, Expectations};
        eprintln!("Validating runs...");
        let mut kept = Vec::new();
        let mut dropped = 0usize;
        for spec in specs {
            let report = match validate_run(&spec.directory, &Expectations::default(), true) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  [SKIP] {} — validation crashed: {}",
                              spec.directory.display(), e);
                    dropped += 1;
                    continue;
                }
            };
            if !report.all_pass() {
                eprintln!("  [SKIP] {} — failed validation:", spec.directory.display());
                print_report(&report);
                dropped += 1;
                continue;
            }
            // Reuse the pre-loaded trajectory.
            kept.push((spec, report.trajectory));
        }
        if dropped > 0 {
            eprintln!("  dropped {} run(s); kept {}", dropped, kept.len());
        }
        if kept.is_empty() {
            return Err(anyhow!(
                "all runs failed validation; re-run with --skip-validation to bypass"
            ));
        }
        kept
    };

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
    let runs: Vec<RunAnalysis> = validated
        .par_iter()
        .map(|(spec, prevalidated)| analyze_run(spec, &plan, prevalidated.clone()))
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
        render_figure(fig, &ws, &runs, &out_dir)?;
    }

    // 5. Emit machine-readable raw numbers alongside figures. TOML
    // remains the source of truth for what gets computed; the JSON is
    // the canonical raw-data artifact downstream tools consume.
    let json_path = out_dir.join("study_results.json");
    write_study_json(
        &json_path,
        toml_path,
        base_dir,
        &out_dir,
        &cfg,
        &observables,
        &runs,
        &ws,
    )?;
    eprintln!("  wrote {}", json_path.display());

    Ok(())
}

fn write_study_json(
    path: &Path,
    toml_path: &Path,
    base_dir: &Path,
    out_dir: &Path,
    cfg: &StudyToml,
    observables: &[Box<dyn ErasedObservable>],
    runs: &[RunAnalysis],
    ws: &Workspace,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create parent dir {}", parent.display()))?;
    }

    let runs_json: Vec<RunAnalysisJson> = runs
        .iter()
        .map(|r| {
            let mut obs_json = BTreeMap::new();
            for obs in observables {
                if let Some(v) = obs.serialize_output(&r.bag) {
                    obs_json.insert(obs.id().to_string(), v);
                }
            }
            RunAnalysisJson {
                directory: r.directory.to_string_lossy().to_string(),
                variables: r.variables.clone(),
                params: r.params.clone(),
                metadata: r.metadata.clone(),
                observables: obs_json,
            }
        })
        .collect();

    let mut slots_json = BTreeMap::new();
    for (name, slot) in &ws.slots {
        let value = match slot {
            Slot::Groups(v) => serde_json::json!({"kind": "groups", "data": v}),
            Slot::Summaries(v) => serde_json::json!({"kind": "summaries", "data": v}),
            Slot::Curve(v) => serde_json::json!({"kind": "curve", "data": v}),
            Slot::Pairs(v) => serde_json::json!({"kind": "pairs", "data": v}),
            Slot::RunPairs(v) => serde_json::json!({"kind": "run_pairs", "data": v}),
            Slot::SingleRun(v) => serde_json::json!({"kind": "single_run", "data": v}),
            Slot::Overlay(v) => serde_json::json!({"kind": "overlay", "data": v}),
        };
        slots_json.insert(name.clone(), value);
    }

    let payload = StudyResultJson {
        study_name: cfg.study.name.clone(),
        study_description: cfg.study.description.clone(),
        config_path: toml_path.to_string_lossy().to_string(),
        data_dir: base_dir.to_string_lossy().to_string(),
        output_dir: out_dir.to_string_lossy().to_string(),
        runs: runs_json,
        slots: slots_json,
    };

    let text = serde_json::to_string_pretty(&payload)?;
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))?;
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
            match ws.get(input)? {
                Slot::Summaries(s) => {
                    let curve = Sweep { axis }.run(s)?;
                    ws.insert(into, Slot::Curve(curve));
                }
                Slot::Pairs(pairs) => {
                    let as_summaries: Vec<GroupSummary> = pairs
                        .iter()
                        .map(|p| GroupSummary {
                            key: p.key.clone(),
                            variables: p.variables.clone(),
                            n: p.numerator.n,
                            metrics: p.ratios.clone(),
                        })
                        .collect();
                    let curve = Sweep { axis }.run(&as_summaries)?;
                    ws.insert(into, Slot::Curve(curve));
                }
                _ => return Err(anyhow!("sweep expects `summaries` or `pairs` input `{}`", input)),
            }
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
        AggregateToml::PairRuns {
            pair_var,
            numerator,
            denominator,
            seeds,
            seed_var,
            into,
        } => {
            let pairs = pair_runs(runs, pair_var, numerator, denominator, seeds, seed_var);
            ws.insert(into, Slot::RunPairs(pairs));
            Ok(())
        }
        AggregateToml::SingleRun { filter, label, into } => {
            let matches: Vec<usize> = runs
                .iter()
                .enumerate()
                .filter(|(_, r)| filter_matches(r, filter))
                .map(|(i, _)| i)
                .collect();
            if matches.is_empty() {
                return Err(anyhow!(
                    "single_run: no runs match filter {:?}",
                    filter
                ));
            }
            if matches.len() > 1 {
                return Err(anyhow!(
                    "single_run: filter {:?} matched {} runs (need exactly 1)",
                    filter,
                    matches.len()
                ));
            }
            let idx = matches[0];
            let display_label = label.clone().unwrap_or_else(|| {
                if filter.is_empty() {
                    "run".to_string()
                } else {
                    filter
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join(",")
                }
            });
            ws.insert(
                into,
                Slot::SingleRun(SingleRunRef {
                    index: idx,
                    label: display_label,
                }),
            );
            Ok(())
        }
        AggregateToml::Overlay { vary, filter, into } => {
            let mut series: Vec<OverlaySeriesRef> = runs
                .iter()
                .enumerate()
                .filter(|(_, r)| filter_matches(r, filter))
                .map(|(i, r)| {
                    let val = r
                        .variables
                        .get(vary)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| format!("run{}", i));
                    OverlaySeriesRef {
                        run_index: i,
                        label: format!("{}={}", vary, val),
                    }
                })
                .collect();
            if series.is_empty() {
                return Err(anyhow!(
                    "overlay: no runs match filter {:?}",
                    filter
                ));
            }
            // Sort by the vary variable value (numeric if possible).
            series.sort_by(|a, b| {
                let ax = runs[a.run_index]
                    .variables
                    .get(vary)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::INFINITY);
                let bx = runs[b.run_index]
                    .variables
                    .get(vary)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::INFINITY);
                ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
            });
            ws.insert(into, Slot::Overlay(OverlayRef { series }));
            Ok(())
        }
    }
}

/// `var = value` equality check on a `RunAnalysis`. Used by both
/// `single_run` and `overlay` filter blocks.
fn filter_matches(run: &RunAnalysis, filter: &BTreeMap<String, String>) -> bool {
    for (k, v) in filter {
        let actual = run
            .variables
            .get(k)
            .map(|s| s.to_string())
            .unwrap_or_default();
        if &actual != v {
            return false;
        }
    }
    true
}

/// For each unique residual (every variable except `pair_var`), find
/// the run with `pair_var = numerator` and the run with `pair_var =
/// denominator`. Emit one [`RunPair`] per match.
fn pair_runs(
    runs: &[RunAnalysis],
    pair_var: &str,
    numerator: &str,
    denominator: &str,
    seeds: &[String],
    seed_var: &str,
) -> Vec<RunPair> {
    let mut by_residual: BTreeMap<
        String,
        (
            BTreeMap<String, BTreeMap<String, usize>>,
            BTreeMap<String, super::discovery::ScalarValue>,
        ),
    > = BTreeMap::new();
    for (i, r) in runs.iter().enumerate() {
        let cond = r
            .variables
            .get(pair_var)
            .map(|v| v.to_string())
            .unwrap_or_default();
        let residual_key: String = r
            .variables
            .iter()
            .filter(|(k, _)| k.as_str() != pair_var)
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");
        let entry = by_residual.entry(residual_key).or_insert_with(|| {
            let residual_vars: BTreeMap<_, _> = r
                .variables
                .iter()
                .filter(|(k, _)| k.as_str() != pair_var)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            (BTreeMap::new(), residual_vars)
        });
        entry
            .0
            .entry(cond)
            .or_default()
            .insert(format!("{}", i), i);
    }

    let mut out = Vec::new();
    for (residual, (cond_map, residual_vars)) in by_residual {
        let num_run = cond_map.get(numerator).and_then(|m| m.values().next());
        let den_run = cond_map.get(denominator).and_then(|m| m.values().next());
        let (Some(&num_idx), Some(&den_idx)) = (num_run, den_run) else {
            continue;
        };
        // Filter by seed if requested.
        if !seeds.is_empty() {
            let seed_value = residual_vars
                .get(seed_var)
                .map(|v| v.to_string())
                .unwrap_or_default();
            if !seeds.iter().any(|s| s == &seed_value) {
                continue;
            }
        }
        out.push(RunPair {
            key: residual,
            variables: residual_vars,
            numerator_idx: num_idx,
            denominator_idx: den_idx,
            numerator_label: numerator.to_string(),
            denominator_label: denominator.to_string(),
        });
    }
    out
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
fn render_figure(
    fig: &FigureToml,
    ws: &Workspace,
    runs: &[RunAnalysis],
    out_dir: &Path,
) -> Result<()> {
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
            PanelToml::Pair {
                subtype,
                input,
                pair_index,
                title,
                x_range,
                y_range,
                speed_max,
                msd_lag_max,
                gvi_x_max,
                bins,
            } => {
                let pairs = match ws.get(input)? {
                    Slot::RunPairs(p) => p,
                    _ => return Err(anyhow!("pair panel expects RunPairs slot `{}`", input)),
                };
                let p = pairs.get(*pair_index).ok_or_else(|| {
                    anyhow!("pair_index {} out of bounds for slot `{}`", pair_index, input)
                })?;
                let data = crate::analysis::panels::pair::PairPanelData {
                    numerator: &runs[p.numerator_idx],
                    denominator: &runs[p.denominator_idx],
                    numerator_label: &p.numerator_label,
                    denominator_label: &p.denominator_label,
                };
                let opts = PanelOpts {
                    title: title.clone(),
                    x_label: None,
                    y_label: None,
                    x_range: x_range.map(|r| (r[0], r[1])),
                    y_range: y_range.map(|r| (r[0], r[1])),
                    log_x: false,
                    log_y: false,
                };
                use crate::analysis::panels::pair as pp;
                use crate::analysis::panels::Panel as _Panel;
                match subtype.as_str() {
                    "speed_bursts" => pp::speed_bursts::SpeedBurstsPair {
                        speed_max: speed_max.unwrap_or(0.02),
                    }
                    .render(cell, &data, &opts)?,
                    "ln_timeseries" => pp::ln_timeseries::LnTimeseriesPair.render(cell, &data, &opts)?,
                    "ln_histogram" => pp::ln_histogram::LnHistogramPair {
                        n_bins: bins.unwrap_or(40),
                    }
                    .render(cell, &data, &opts)?,
                    "msd_t" => pp::msd_t::MsdTPair {
                        msd_lag_max: msd_lag_max.unwrap_or(8.0),
                    }
                    .render(cell, &data, &opts)?,
                    "deff_bar" => pp::deff_bar::DeffBarPair.render(cell, &data, &opts)?,
                    "gvi" => pp::gvi::GviPair {
                        x_max: gvi_x_max.unwrap_or(0.022),
                        v_a: data.numerator.params.v_a,
                    }
                    .render(cell, &data, &opts)?,
                    "summary" => pp::summary::SummaryPair.render(cell, &data, &opts)?,
                    other => {
                        return Err(anyhow!("unknown pair subtype `{}`", other));
                    }
                }
            }
            PanelToml::Single {
                subtype,
                input,
                title,
                x_range,
                y_range,
                speed_max,
                msd_lag_max,
                gvi_x_max,
                fit_eq5,
                show_population,
            } => {
                let single = match ws.get(input)? {
                    Slot::SingleRun(r) => r,
                    _ => return Err(anyhow!("single panel expects SingleRun slot `{}`", input)),
                };
                let data = crate::analysis::panels::single::SingleRunData {
                    run: &runs[single.index],
                    label: &single.label,
                };
                let opts = PanelOpts {
                    title: title.clone(),
                    x_label: None,
                    y_label: None,
                    x_range: x_range.map(|r| (r[0], r[1])),
                    y_range: y_range.map(|r| (r[0], r[1])),
                    log_x: false,
                    log_y: false,
                };
                use crate::analysis::panels::single as sp;
                use crate::analysis::panels::Panel as _Panel;
                match subtype.as_str() {
                    "msd" => sp::msd::MsdSingle {
                        msd_lag_max: msd_lag_max.unwrap_or(8.0),
                        show_population: show_population.unwrap_or(true),
                    }
                    .render(cell, &data, &opts)?,
                    "gvi" => sp::gvi::GviSingle {
                        x_max: gvi_x_max.unwrap_or(0.022),
                        fit_eq5: fit_eq5.unwrap_or(true),
                    }
                    .render(cell, &data, &opts)?,
                    "ln_timeseries" => sp::ln_timeseries::LnTimeseriesSingle.render(cell, &data, &opts)?,
                    "speed_bursts" => sp::speed_bursts::SpeedBurstsSingle {
                        speed_max: speed_max.unwrap_or(0.02),
                    }
                    .render(cell, &data, &opts)?,
                    other => {
                        return Err(anyhow!("unknown single subtype `{}`", other));
                    }
                }
            }
            PanelToml::Overlay {
                subtype,
                input,
                title,
                x_range,
                y_range,
                msd_lag_max,
                gvi_x_max,
            } => {
                let overlay = match ws.get(input)? {
                    Slot::Overlay(o) => o,
                    _ => return Err(anyhow!("overlay panel expects Overlay slot `{}`", input)),
                };
                let series: Vec<crate::analysis::panels::overlay::OverlaySeries<'_>> = overlay
                    .series
                    .iter()
                    .map(|s| crate::analysis::panels::overlay::OverlaySeries {
                        run: &runs[s.run_index],
                        label: &s.label,
                    })
                    .collect();
                let data = crate::analysis::panels::overlay::OverlayData { series };
                let opts = PanelOpts {
                    title: title.clone(),
                    x_label: None,
                    y_label: None,
                    x_range: x_range.map(|r| (r[0], r[1])),
                    y_range: y_range.map(|r| (r[0], r[1])),
                    log_x: false,
                    log_y: false,
                };
                use crate::analysis::panels::overlay as op;
                use crate::analysis::panels::Panel as _Panel;
                match subtype.as_str() {
                    "msd" => op::msd::MsdOverlay {
                        msd_lag_max: msd_lag_max.unwrap_or(8.0),
                    }
                    .render(cell, &data, &opts)?,
                    "gvi" => op::gvi::GviOverlay {
                        x_max: gvi_x_max.unwrap_or(0.022),
                    }
                    .render(cell, &data, &opts)?,
                    "ln_timeseries" => op::ln_timeseries::LnTimeseriesOverlay.render(cell, &data, &opts)?,
                    other => {
                        return Err(anyhow!("unknown overlay subtype `{}`", other));
                    }
                }
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
        run_study(&toml_path, &root, true).expect("run_study");

        // 4. Verify the figure exists.
        let svg = root.join("phase3a_results").join("phase3a_msd_vs_d.svg");
        assert!(svg.exists(), "figure not produced: {}", svg.display());

        let _ = fs::remove_dir_all(&root);
    }
}
