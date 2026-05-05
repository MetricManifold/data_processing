//! Study template engine: TOML-driven analysis pipeline.
//!
//! A study config defines:
//! - Discovery: how to find runs via path patterns with named captures
//! - Observables: what to compute per run
//! - Analysis: how to group, pair, and compute metrics across runs
//! - Plots: what SVG plots to generate

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::io::{load_trajectory, load_trajectory_subsample, unwrap_trajectory};
use super::observables::*;
use super::output::RunResult;

// ============================================================================
// TOML config schema
// ============================================================================

/// Top-level study configuration parsed from TOML.
#[derive(Debug, Deserialize)]
pub struct StudyConfig {
    pub study: StudyMeta,
    pub discovery: DiscoveryConfig,
    pub observables: ObservablesConfig,
    pub analysis: AnalysisConfig,
    #[serde(default)]
    pub plots: Vec<PlotConfig>,
    #[serde(default)]
    pub figures: Vec<FigureConfig>,
    #[serde(default)]
    pub diagnostic: Option<DiagnosticConfig>,
}

/// Configuration for diagnostic comparison figures (soft vs ctrl panels).
#[derive(Debug, Deserialize, Clone)]
pub struct DiagnosticConfig {
    /// Which seeds to generate diagnostics for (empty = all)
    #[serde(default)]
    pub seeds: Vec<String>,
    /// Output filename pattern (use {seed} placeholder)
    #[serde(default = "default_diag_output")]
    pub output: String,
    /// L_n y-axis range
    #[serde(default = "default_ln_range")]
    pub ln_range: [f64; 2],
    /// Speed y-axis max
    #[serde(default = "default_speed_max")]
    pub speed_max: f64,
    /// MSD lag max in units of tau
    #[serde(default = "default_msd_lag_max")]
    pub msd_lag_max: f64,
    /// Compare panel configuration (which panels to show and their settings)
    #[serde(default)]
    pub panels: Vec<ComparePanelConfig>,
}

/// Configuration for a single panel in the compare figure.
#[derive(Debug, Deserialize, Clone)]
pub struct ComparePanelConfig {
    /// Panel type: "trajectory", "msd_t", "ln_timeseries", "ln_histogram",
    /// "speed_bursts", "gvi", "deff_bar", "summary"
    #[serde(rename = "type")]
    pub panel_type: String,
    /// Optional title override
    pub title: Option<String>,
    /// X-axis range [min, max] (panel-specific defaults if omitted)
    pub x_range: Option<[f64; 2]>,
    /// Y-axis range [min, max]
    pub y_range: Option<[f64; 2]>,
    /// Log scale on x-axis
    #[serde(default)]
    pub log_x: bool,
    /// Log scale on y-axis
    #[serde(default)]
    pub log_y: bool,
    /// Number of histogram bins (for ln_histogram, gvi)
    pub bins: Option<usize>,
    /// Show population average (dashed) in addition to cell 0
    #[serde(default = "default_true")]
    pub show_population: bool,
    /// Show Gaussian reference line (for gvi)
    #[serde(default = "default_true")]
    pub gaussian_ref: bool,
}

fn default_diag_output() -> String { "diagnostic_{seed}.svg".to_string() }
fn default_ln_range() -> [f64; 2] { [0.98, 1.5] }
fn default_speed_max() -> f64 { 0.02 }
fn default_msd_lag_max() -> f64 { 8.0 }

#[derive(Debug, Deserialize)]
pub struct StudyMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Deserialize)]
pub struct DiscoveryConfig {
    /// Path pattern(s) with named captures in braces: {N}, {rho}, {cond}, {seed}
    /// Accepts a single string or a list of strings. Multiple patterns are tried
    /// in order and all matches are merged (duplicates by path are removed).
    /// Supports two layouts:
    ///   - Directory: "{N}c_rho{rho}_{cond}/run_{seed}" (multi-dir)  
    ///   - File: "fss_{N}c_{rho}{cond}.txt" (single-file per condition)
    #[serde(deserialize_with = "deserialize_string_or_vec")]
    pub pattern: Vec<String>,
    /// Optional: if set, look for trajectory files matching this name pattern
    /// instead of trajectory.txt
    #[serde(default = "default_traj_name")]
    pub trajectory_name: String,
}

/// Deserialize a field that can be either a single string or a list of strings.
fn deserialize_string_or_vec<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct StringOrVec;
    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Vec<String>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a string or list of strings")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Vec<String>, E> {
            Ok(vec![v.to_string()])
        }
        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Vec<String>, A::Error> {
            let mut v = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                v.push(s);
            }
            Ok(v)
        }
    }
    deserializer.deserialize_any(StringOrVec)
}

fn default_traj_name() -> String {
    "trajectory.txt".to_string()
}

#[derive(Debug, Deserialize)]
pub struct ObservablesConfig {
    pub compute: Vec<String>,
    #[serde(default = "default_tau")]
    pub tau: f64,
    #[serde(default = "default_cell_radius")]
    pub cell_radius: f64,
    #[serde(default = "default_fit_frac")]
    pub fit_frac: f64,
}

fn default_tau() -> f64 { 10000.0 }
fn default_cell_radius() -> f64 { 49.0 }
fn default_fit_frac() -> f64 { 0.3 }

#[derive(Debug, Deserialize)]
pub struct AnalysisConfig {
    /// Which cell index is the "tagged" cell of interest (default: 0)
    #[serde(default)]
    pub tagged_cell: usize,
    /// Variables to group by (seeds within a group are averaged)
    pub group_by: Vec<String>,
    /// Variable to pair across (e.g. "cond" to compare soft vs ctrl)
    #[serde(default)]
    pub pair_by: Option<String>,
    /// Value of pair_by for the numerator (e.g. "soft")
    #[serde(default)]
    pub pair_numerator: Option<String>,
    /// Value of pair_by for the denominator (e.g. "ctrl")
    #[serde(default)]
    pub pair_denominator: Option<String>,
    /// Named metrics to compute for each group
    #[serde(default)]
    pub metrics: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct PlotConfig {
    /// Plot title
    pub title: String,
    /// X-axis variable (must be a discovered variable, e.g. "N")
    pub x: String,
    /// Y-axis metric name (must be defined in analysis.metrics or built-in)
    pub y: String,
    /// Output filename (SVG)
    pub output: String,
    /// Optional: x-axis label
    #[serde(default)]
    pub x_label: Option<String>,
    /// Optional: y-axis label
    #[serde(default)]
    pub y_label: Option<String>,
    /// Optional: x-axis transform ("inverse_sqrt" for 1/√x)
    #[serde(default)]
    pub x_transform: Option<String>,
    /// Optional: log scale on x
    #[serde(default)]
    pub x_log: bool,
    /// Optional: include error bars
    #[serde(default = "default_true")]
    pub error_bars: bool,
    /// Optional: y-axis min (auto if not set)
    #[serde(default)]
    pub y_min: Option<f64>,
    /// Optional: y-axis max (auto if not set)
    #[serde(default)]
    pub y_max: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct FigureConfig {
    /// Figure title
    pub title: String,
    /// Output filename (SVG)
    pub output: String,
    /// Layout: [rows, cols]
    pub layout: [usize; 2],
    /// Width in pixels (default: 800)
    #[serde(default = "default_fig_width")]
    pub width: u32,
    /// Height in pixels (default: 600)
    #[serde(default = "default_fig_height")]
    pub height: u32,
    /// Panels (one per subplot)
    pub panels: Vec<PanelConfig>,
}

fn default_fig_width() -> u32 { 900 }
fn default_fig_height() -> u32 { 700 }

#[derive(Debug, Deserialize)]
pub struct PanelConfig {
    /// X-axis variable
    pub x: String,
    /// Y-axis metric name
    pub y: String,
    /// Optional panel title
    #[serde(default)]
    pub title: Option<String>,
    /// Optional: x-axis label
    #[serde(default)]
    pub x_label: Option<String>,
    /// Optional: y-axis label
    #[serde(default)]
    pub y_label: Option<String>,
    /// Optional: x-axis transform
    #[serde(default)]
    pub x_transform: Option<String>,
    /// Optional: include error bars (default: true)
    #[serde(default = "default_true")]
    pub error_bars: bool,
    /// Optional: horizontal reference line
    #[serde(default)]
    pub h_line: Option<f64>,
    /// Optional: y-axis min
    #[serde(default)]
    pub y_min: Option<f64>,
    /// Optional: y-axis max
    #[serde(default)]
    pub y_max: Option<f64>,
}

fn default_true() -> bool { true }

// ============================================================================
// Discovery: pattern → directory walk → classified runs
// ============================================================================

/// A discovered run with extracted variables from the path pattern.
#[derive(Debug, Clone)]
pub struct DiscoveredRun {
    pub path: PathBuf,
    pub trajectory: PathBuf,
    pub variables: BTreeMap<String, String>,
}

impl DiscoveredRun {
    /// Get a variable as string, or "" if missing.
    pub fn var(&self, name: &str) -> &str {
        self.variables.get(name).map(|s| s.as_str()).unwrap_or("")
    }

    /// Get a variable parsed as f64.
    pub fn var_f64(&self, name: &str) -> Option<f64> {
        self.variables.get(name)?.parse().ok()
    }

    /// Get a variable parsed as usize.
    pub fn var_usize(&self, name: &str) -> Option<usize> {
        self.variables.get(name)?.parse().ok()
    }

    /// Build a group key from the specified variables.
    pub fn group_key(&self, group_by: &[String]) -> String {
        group_by
            .iter()
            .map(|v| format!("{}={}", v, self.var(v)))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Convert a pattern like "{N}c_rho{rho}_{cond}/run_{seed}" into a regex
/// with named capture groups, and extract the variable names.
fn pattern_to_regex(pattern: &str) -> Result<(regex::Regex, Vec<String>)> {
    let mut regex_str = String::from("^");
    let mut var_names = Vec::new();
    let mut chars = pattern.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            // Extract variable name
            let mut name = String::new();
            for inner in chars.by_ref() {
                if inner == '}' {
                    break;
                }
                name.push(inner);
            }
            if name.is_empty() {
                anyhow::bail!("Empty variable name in pattern");
            }
            var_names.push(name.clone());
            // Named capture: word chars, digits, dots, hyphens
            regex_str.push_str(&format!("(?P<{}>[-\\w.]+)", name));
        } else if ch == '/' {
            // Path separator — match either / or \ 
            regex_str.push_str("[/\\\\]");
        } else if ".+*?^$|()[]".contains(ch) {
            // Escape regex metacharacters
            regex_str.push('\\');
            regex_str.push(ch);
        } else {
            regex_str.push(ch);
        }
    }
    regex_str.push('$');

    let re = regex::Regex::new(&regex_str)
        .with_context(|| format!("Invalid pattern regex: {}", regex_str))?;
    Ok((re, var_names))
}

/// Discover runs under `base_dir` matching the config pattern(s).
pub fn discover_study_runs(
    base_dir: &Path,
    config: &DiscoveryConfig,
) -> Result<Vec<DiscoveredRun>> {
    let mut runs = Vec::new();
    let traj_name = &config.trajectory_name;
    let mut seen_paths = std::collections::HashSet::new();

    for pattern in &config.pattern {
        let (re, var_names) = pattern_to_regex(pattern)?;
        let depth = pattern.matches('/').count() + 1;
        let mut pattern_runs = Vec::new();
        walk_and_match(base_dir, base_dir, &re, &var_names, traj_name, depth, 0, &mut pattern_runs)?;
        for r in pattern_runs {
            if seen_paths.insert(r.path.clone()) {
                runs.push(r);
            }
        }
    }

    runs.sort_by(|a, b| {
        let ak = format!("{:?}", a.variables);
        let bk = format!("{:?}", b.variables);
        ak.cmp(&bk)
    });

    Ok(runs)
}

fn walk_and_match(
    base_dir: &Path,
    current: &Path,
    re: &regex::Regex,
    var_names: &[String],
    traj_name: &str,
    max_depth: usize,
    current_depth: usize,
    results: &mut Vec<DiscoveredRun>,
) -> Result<()> {
    if current_depth > max_depth {
        return Ok(());
    }

    let entries = match std::fs::read_dir(current) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Try matching the relative path against the pattern
            let rel = path
                .strip_prefix(base_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            if let Some(caps) = re.captures(&rel) {
                // Check for trajectory file
                let traj = path.join(traj_name);
                if traj.exists() {
                    let mut variables = BTreeMap::new();
                    for name in var_names {
                        if let Some(m) = caps.name(name) {
                            variables.insert(name.clone(), m.as_str().to_string());
                        }
                    }
                    results.push(DiscoveredRun {
                        path: path.clone(),
                        trajectory: traj,
                        variables,
                    });
                    continue; // Don't recurse further into matched dirs
                }
            }

            // Recurse deeper
            walk_and_match(base_dir, &path, re, var_names, traj_name, max_depth, current_depth + 1, results)?;
        } else if path.is_file() && current_depth == 0 {
            // For file-level patterns (e.g., fss_{N}c_{rho}{cond}.txt)
            let fname = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if let Some(caps) = re.captures(&fname) {
                let mut variables = BTreeMap::new();
                for name in var_names {
                    if let Some(m) = caps.name(name) {
                        variables.insert(name.clone(), m.as_str().to_string());
                    }
                }
                results.push(DiscoveredRun {
                    path: path.parent().unwrap_or(base_dir).to_path_buf(),
                    trajectory: path.clone(),
                    variables,
                });
            }
        }
    }

    Ok(())
}

// ============================================================================
// Analysis engine: analyze, group, pair, compute metrics
// ============================================================================

/// Result for a single run including extracted variables.
#[derive(Debug, Clone)]
struct AnalyzedRun {
    variables: BTreeMap<String, String>,
    trajectory: PathBuf,
    result: RunResult,
}

/// Aggregated metrics for a group of seeds.
#[derive(Debug, Clone, Serialize)]
pub struct GroupMetrics {
    pub group_key: String,
    pub variables: BTreeMap<String, String>,
    pub n_seeds: usize,
    pub metrics: BTreeMap<String, MetricValue>,
    /// Per-seed raw values for each metric
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub per_seed: BTreeMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricValue {
    pub mean: f64,
    pub stderr: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f64>>,
}

/// Paired group result (e.g., soft vs ctrl at the same N, rho).
#[derive(Debug, Clone, Serialize)]
pub struct PairedGroupMetrics {
    pub group_key: String,
    pub variables: BTreeMap<String, String>,
    pub numerator: GroupMetrics,
    pub denominator: GroupMetrics,
    /// Paired metrics: ratios, differences
    pub paired_metrics: BTreeMap<String, MetricValue>,
}

/// Per-run parameter summary (included in study output).
#[derive(Debug, Clone, Serialize)]
pub struct RunParamsSummary {
    pub path: String,
    pub params: super::output::RunParams,
}

/// Full study output.
#[derive(Debug, Clone, Serialize)]
pub struct StudyResult {
    pub study_name: String,
    pub description: String,
    pub n_runs_total: usize,
    pub n_groups: usize,
    /// Per-run parameters (path, params with bbox_mean/subdomain_padding)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub run_params: Vec<RunParamsSummary>,
    /// Data quality warnings
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    /// Unpaired group metrics (one per group_key)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupMetrics>,
    /// Paired comparisons (when pair_by is specified)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub paired: Vec<PairedGroupMetrics>,
}

/// Run the full study pipeline.
pub fn run_study(
    base_dir: &Path,
    config: &StudyConfig,
    output_dir: &Path,
    subsample: usize,
) -> Result<StudyResult> {
    // 1. Discover runs
    eprintln!("Discovering runs...");
    let discovered = discover_study_runs(base_dir, &config.discovery)?;
    eprintln!("  Found {} runs", discovered.len());

    if discovered.is_empty() {
        anyhow::bail!(
            "No runs found matching pattern(s) {:?} under {}",
            config.discovery.pattern,
            base_dir.display()
        );
    }

    // Print discovery summary
    let mut var_values: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    for run in &discovered {
        for (k, v) in &run.variables {
            var_values.entry(k.clone()).or_default().insert(v.clone());
        }
    }
    for (k, vs) in &var_values {
        let vals: Vec<_> = vs.iter().collect();
        eprintln!("  {}: {:?}", k, vals);
    }

    // 2. Analyze all runs in parallel
    let obs = &config.observables;
    eprintln!("\nAnalyzing {} runs...", discovered.len());

    let analyzed: Vec<Result<AnalyzedRun>> = discovered
        .par_iter()
        .map(|run| {
            let rel = run.trajectory.strip_prefix(base_dir)
                .unwrap_or(&run.trajectory)
                .display()
                .to_string();
            eprintln!("  {}", rel);

            let result = analyze_run_for_study(
                &run.trajectory,
                &run.path,
                &obs.compute,
                obs.tau,
                obs.cell_radius,
                obs.fit_frac,
                subsample,
            )?;

            Ok(AnalyzedRun {
                variables: run.variables.clone(),
                trajectory: run.trajectory.clone(),
                result,
            })
        })
        .collect();

    // Collect successful results
    let mut runs: Vec<AnalyzedRun> = Vec::new();
    let mut errors = Vec::new();
    for r in analyzed {
        match r {
            Ok(ar) => runs.push(ar),
            Err(e) => errors.push(e.to_string()),
        }
    }
    if !errors.is_empty() {
        eprintln!("\n{} runs failed:", errors.len());
        for e in &errors {
            eprintln!("  {}", e);
        }
    }

    let n_runs_total = runs.len();

    // Collect per-run params for output
    let run_params: Vec<RunParamsSummary> = runs.iter().map(|ar| RunParamsSummary {
        path: ar.result.path.clone(),
        params: ar.result.params.clone(),
    }).collect();

    // 3. Group and aggregate
    let ac = &config.analysis;
    let tagged = ac.tagged_cell;

    if let (Some(pair_by), Some(num_val), Some(den_val)) =
        (&ac.pair_by, &ac.pair_numerator, &ac.pair_denominator)
    {
        // Paired analysis mode
        let paired = compute_paired_metrics(&runs, &ac.group_by, pair_by, num_val, den_val, tagged, &ac.metrics);

        // Generate individual plots
        if !config.plots.is_empty() {
            std::fs::create_dir_all(output_dir).ok();
            for plot_cfg in &config.plots {
                if let Err(e) = generate_plot(plot_cfg, &paired, output_dir) {
                    eprintln!("Plot '{}' failed: {}", plot_cfg.title, e);
                }
            }
        }

        // Generate multi-panel figures
        if !config.figures.is_empty() {
            std::fs::create_dir_all(output_dir).ok();
            for fig_cfg in &config.figures {
                if let Err(e) = generate_figure(fig_cfg, &paired, output_dir) {
                    eprintln!("Figure '{}' failed: {}", fig_cfg.title, e);
                }
            }
        }

        // Generate per-seed diagnostic comparisons
        if let Some(ref diag) = config.diagnostic {
            std::fs::create_dir_all(output_dir).ok();
            // Build a compound key from all non-pairing variables (e.g. "rho=85,seed=2")
            // so that each unique combination gets its own diagnostic figure.
            let non_pair_vars: Vec<String> = {
                let mut vars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
                for run in &runs {
                    for k in run.variables.keys() {
                        if k != pair_by { vars.insert(k.clone()); }
                    }
                }
                vars.into_iter().collect()
            };
            // Map compound_key -> (soft_path, ctrl_path, variables)
            let mut soft_map: BTreeMap<String, (PathBuf, BTreeMap<String, String>)> = BTreeMap::new();
            let mut ctrl_map: BTreeMap<String, (PathBuf, BTreeMap<String, String>)> = BTreeMap::new();
            for run in &runs {
                let compound_key: String = non_pair_vars.iter()
                    .map(|v| run.variables.get(v).cloned().unwrap_or_default())
                    .collect::<Vec<_>>().join("_");
                let cond = run.variables.get(pair_by).cloned().unwrap_or_default();
                let path = run.trajectory.parent().unwrap_or(Path::new(".")).to_path_buf();
                if cond == *num_val { soft_map.insert(compound_key.clone(), (path.clone(), run.variables.clone())); }
                if cond == *den_val { ctrl_map.insert(compound_key.clone(), (path, run.variables.clone())); }
            }
            let target_keys: Vec<String> = if diag.seeds.is_empty() {
                soft_map.keys().filter(|k| ctrl_map.contains_key(*k)).cloned().collect()
            } else {
                // Filter compound keys that match any of the target seeds
                soft_map.keys()
                    .filter(|k| ctrl_map.contains_key(*k))
                    .filter(|k| {
                        if let Some((_, vars)) = soft_map.get(*k) {
                            diag.seeds.iter().any(|s| vars.get("seed").map(|v| v == s).unwrap_or(false))
                        } else { false }
                    })
                    .cloned().collect()
            };
            for key in &target_keys {
                if let (Some((sp, vars)), Some((cp, _))) = (soft_map.get(key), ctrl_map.get(key)) {
                    // Substitute all discovered variables in the output filename
                    let mut out_name = diag.output.clone();
                    for (var_name, var_value) in vars {
                        out_name = out_name.replace(&format!("{{{}}}", var_name), var_value);
                    }
                    let out_path = output_dir.join(&out_name);
                    let tau = config.observables.tau;
                    let seed_display = vars.get("seed").cloned().unwrap_or(key.clone());
                    if let Err(e) = generate_comparison_with_config(
                        sp, cp, &out_path, tagged, subsample, tau,
                        num_val, den_val, diag,
                    ) {
                        eprintln!("Diagnostic {} failed: {}", seed_display, e);
                    }
                }
            }
        }

        // Generate data quality warnings
        let mut warnings = Vec::new();
        for pg in &paired {
            let ns = pg.numerator.n_seeds;
            let nd = pg.denominator.n_seeds;
            if ns < 5 || nd < 5 {
                warnings.push(format!(
                    "{}: only {} soft + {} ctrl seeds (recommend >=10)",
                    pg.group_key, ns, nd
                ));
            }
            if ns != nd {
                warnings.push(format!(
                    "{}: unequal seeds ({} soft vs {} ctrl)",
                    pg.group_key, ns, nd
                ));
            }
            // Check D_eff CV
            if let Some(mv) = pg.numerator.metrics.get("d_eff") {
                if mv.stderr > 0.0 && mv.mean > 0.0 && mv.stderr / mv.mean > 0.3 {
                    warnings.push(format!(
                        "{}: high D_eff stderr/mean = {:.0}% (consider more seeds)",
                        pg.group_key, 100.0 * mv.stderr / mv.mean
                    ));
                }
            }
        }
        if !warnings.is_empty() {
            eprintln!("\nWarnings:");
            for w in &warnings {
                eprintln!("  {}", w);
            }
        }

        Ok(StudyResult {
            study_name: config.study.name.clone(),
            description: config.study.description.clone(),
            n_runs_total,
            n_groups: paired.len(),
            run_params,
            warnings,
            groups: Vec::new(),
            paired,
        })
    } else {
        // Unpaired analysis mode
        let groups = compute_group_metrics(&runs, &ac.group_by, tagged, &ac.metrics);

        // Generate per-run diagnostic panels (single-condition mode)
        if let Some(ref diag) = config.diagnostic {
            std::fs::create_dir_all(output_dir).ok();
            for run in &runs {
                let run_path = run.trajectory.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
                let mut out_name = diag.output.clone();
                for (var_name, var_value) in &run.variables {
                    out_name = out_name.replace(&format!("{{{}}}", var_name), var_value);
                }
                let out_path = output_dir.join(&out_name);
                let tau = config.observables.tau;
                // Single-run diagnostic: pass same path as both soft and ctrl
                if let Err(e) = generate_comparison_with_config(
                    &run_path, &run_path, &out_path, tagged, subsample, tau,
                    "run", "_", diag,
                ) {
                    eprintln!("Diagnostic {:?} failed: {}", out_path, e);
                }
            }
        }

        Ok(StudyResult {
            study_name: config.study.name.clone(),
            description: config.study.description.clone(),
            n_runs_total,
            n_groups: groups.len(),
            run_params,
            warnings: Vec::new(),
            groups,
            paired: Vec::new(),
        })
    }
}

/// Analyze a single run for the study pipeline.
fn analyze_run_for_study(
    traj_path: &Path,
    dir: &Path,
    observables: &[String],
    tau: f64,
    _cell_radius: f64,
    fit_frac: f64,
    subsample: usize,
) -> Result<RunResult> {
    let traj = load_trajectory_subsample(traj_path, subsample)?;
    let pos = unwrap_trajectory(&traj);

    let has = |name: &str| observables.iter().any(|s| s == name);

    let cell_spacing = (pos.lx * pos.ly / pos.n_cells as f64).sqrt();
    let cage_radius = cell_spacing * 0.3;

    let msd = if has("msd") || has("diffusion") || has("log_slope") || has("cage") {
        Some(compute_msd(&pos))
    } else {
        None
    };

    let diffusion = if has("diffusion") {
        msd.as_ref().map(|m| compute_diffusion(m, fit_frac))
    } else {
        None
    };

    let pcd = if has("per_cell_diffusion") {
        Some(per_cell_diffusion(&pos, fit_frac, tau))
    } else {
        None
    };

    let shape_idx = if has("shape_index") {
        Some(shape_index(&traj))
    } else {
        None
    };

    let vel_dist = if has("velocity_distribution") {
        Some(velocity_distribution(&pos, 100))
    } else {
        None
    };

    let overlap = if has("overlap") {
        Some(overlap_and_chi4(&pos, cage_radius))
    } else {
        None
    };

    let se = match (&diffusion, &overlap) {
        (Some(d), Some(o)) => {
            let val = stokes_einstein(d.d_eff, o.tau_alpha);
            if val.is_finite() { Some(val) } else { None }
        }
        _ => None,
    };

    let extra: BTreeMap<String, String> = traj
        .params
        .extra
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Read bbox stats from checkpoint (one read for both fields)
    let ckpt_path = dir.join("checkpoint.bin");
    let bbox_stats = if ckpt_path.exists() {
        super::checkpoint::read_bbox_stats(&ckpt_path).ok()
    } else {
        None
    };

    Ok(RunResult {
        path: dir.display().to_string(),
        params: super::output::RunParams {
            v_a: traj.params.v_a,
            n_cells: traj.params.n_cells,
            lx: traj.params.lx,
            ly: traj.params.ly,
            confluence: traj.params.n_cells as f64 * std::f64::consts::PI * _cell_radius * _cell_radius / (traj.params.lx * traj.params.ly),
            subdomain_padding: bbox_stats.as_ref().and_then(|s| s.1),
            bbox_mean: bbox_stats.as_ref().map(|s| s.0),
            extra,
        },
        msd: None, // Skip bulky MSD arrays in study output
        diffusion,
        log_slope: None,
        cage: None,
        alpha2: None,
        overlap,
        structure: None,
        scattering: None,
        van_hove: None,
        per_cell_diffusion: pcd,
        displacement: None,
        stokes_einstein: se,
        va_mobility_correlation: None,
        spatial_correlation: None,
        shape_index: shape_idx,
        velocity_autocorrelation: None,
        burst_detection: None,
        velocity_distribution: vel_dist,
        polarity_tau: None,
        hexatic_order: None,
        voronoi_shape: None,
        kinetic_energy: None,
    })
}

// ============================================================================
// Metric extraction from a single RunResult
// ============================================================================

/// Extract a named metric value from a RunResult.
fn extract_metric(result: &RunResult, metric_expr: &str, tagged_cell: usize) -> Option<f64> {
    match metric_expr {
        "tagged_cell_d_eff" => {
            result.per_cell_diffusion.as_ref()
                .and_then(|pcd| pcd.d_values.get(tagged_cell).copied())
        }
        "population_d_eff" => {
            result.per_cell_diffusion.as_ref()
                .map(|pcd| pcd.d_mean)
        }
        "normal_d_eff" => {
            // Mean D_eff excluding the tagged cell
            result.per_cell_diffusion.as_ref().map(|pcd| {
                let sum: f64 = pcd.d_values.iter().enumerate()
                    .filter(|(i, _)| *i != tagged_cell)
                    .map(|(_, v)| v)
                    .sum();
                let n = pcd.d_values.len().saturating_sub(1);
                if n > 0 { sum / n as f64 } else { 0.0 }
            })
        }
        "tagged_cell_ln" => {
            let factor = 2.0 * std::f64::consts::PI.sqrt();
            result.shape_index.as_ref()
                .and_then(|si| si.per_cell_p.first().copied())
                .map(|p| p / factor)
        }
        "population_ln" => {
            let factor = 2.0 * std::f64::consts::PI.sqrt();
            result.shape_index.as_ref()
                .map(|si| {
                    let sum: f64 = si.per_cell_p.iter().sum();
                    sum / si.per_cell_p.len() as f64 / factor
                })
        }
        "tagged_cell_kurtosis" => {
            result.velocity_distribution.as_ref()
                .map(|vd| vd.cell0_kurtosis)
        }
        "population_kurtosis" => {
            result.velocity_distribution.as_ref()
                .map(|vd| vd.pop_kurtosis)
        }
        "tagged_cell_mean_speed" => {
            result.velocity_distribution.as_ref()
                .map(|vd| vd.cell0_mean_speed)
        }
        "d_eff_cv" => {
            result.per_cell_diffusion.as_ref()
                .map(|pcd| pcd.cv)
        }
        "diffusion_r2" => {
            result.diffusion.as_ref().map(|d| d.fit_r2)
        }
        "stokes_einstein" => {
            result.stokes_einstein
        }
        "tau_alpha" => {
            result.overlap.as_ref().map(|o| o.tau_alpha)
        }
        "population_mean_speed" => {
            result.velocity_distribution.as_ref()
                .map(|vd| vd.pop_mean_speed)
        }
        _ => {
            eprintln!("  Warning: unknown metric expression '{}'", metric_expr);
            None
        }
    }
}

// ============================================================================
// Grouping and pairing
// ============================================================================

fn compute_group_metrics(
    runs: &[AnalyzedRun],
    group_by: &[String],
    tagged_cell: usize,
    metric_defs: &BTreeMap<String, String>,
) -> Vec<GroupMetrics> {
    // Group runs by key
    let mut groups: BTreeMap<String, Vec<&AnalyzedRun>> = BTreeMap::new();
    for run in runs {
        let key = run.variables.iter()
            .filter(|(k, _)| group_by.contains(k))
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");
        groups.entry(key).or_default().push(run);
    }

    groups
        .into_iter()
        .map(|(key, members)| {
            let variables = members[0].variables.clone();
            let n_seeds = members.len();

            let mut metrics = BTreeMap::new();
            let mut per_seed = BTreeMap::new();

            for (metric_name, metric_expr) in metric_defs {
                let values: Vec<f64> = members
                    .iter()
                    .filter_map(|m| extract_metric(&m.result, metric_expr, tagged_cell))
                    .collect();

                if !values.is_empty() {
                    let mv = aggregate_values(&values);
                    per_seed.insert(metric_name.clone(), values);
                    metrics.insert(metric_name.clone(), mv);
                }
            }

            GroupMetrics {
                group_key: key,
                variables,
                n_seeds,
                metrics,
                per_seed,
            }
        })
        .collect()
}

fn compute_paired_metrics(
    runs: &[AnalyzedRun],
    group_by: &[String],
    pair_by: &str,
    num_val: &str,
    den_val: &str,
    tagged_cell: usize,
    metric_defs: &BTreeMap<String, String>,
) -> Vec<PairedGroupMetrics> {
    // Split runs into numerator and denominator sets
    let num_runs: Vec<&AnalyzedRun> = runs.iter()
        .filter(|r| r.variables.get(pair_by).map(|s| s.as_str()) == Some(num_val))
        .collect();
    let den_runs: Vec<&AnalyzedRun> = runs.iter()
        .filter(|r| r.variables.get(pair_by).map(|s| s.as_str()) == Some(den_val))
        .collect();

    // Group each set separately (excluding the pair_by variable from grouping)
    let group_vars: Vec<String> = group_by.iter()
        .filter(|v| v.as_str() != pair_by)
        .cloned()
        .collect();

    let make_key = |r: &AnalyzedRun| -> String {
        group_vars.iter()
            .map(|v| format!("{}={}", v, r.variables.get(v).unwrap_or(&String::new())))
            .collect::<Vec<_>>()
            .join(",")
    };

    let mut num_groups: BTreeMap<String, Vec<&AnalyzedRun>> = BTreeMap::new();
    for r in &num_runs {
        num_groups.entry(make_key(r)).or_default().push(r);
    }

    let mut den_groups: BTreeMap<String, Vec<&AnalyzedRun>> = BTreeMap::new();
    for r in &den_runs {
        den_groups.entry(make_key(r)).or_default().push(r);
    }

    // Match groups
    let all_keys: std::collections::BTreeSet<String> = num_groups.keys()
        .chain(den_groups.keys())
        .cloned()
        .collect();

    all_keys
        .into_iter()
        .filter_map(|key| {
            let num_members = num_groups.get(&key)?;
            let den_members = den_groups.get(&key)?;

            let mut num_vars = num_members[0].variables.clone();
            num_vars.remove(pair_by);

            // Compute per-group metrics
            let num_gm = compute_single_group(&key, num_members, tagged_cell, metric_defs);
            let den_gm = compute_single_group(&key, den_members, tagged_cell, metric_defs);

            // Compute paired metrics (ratios and differences)
            let mut paired_metrics = BTreeMap::new();
            for metric_name in metric_defs.keys() {
                let num_mv = num_gm.metrics.get(metric_name);
                let den_mv = den_gm.metrics.get(metric_name);

                if let (Some(n), Some(d)) = (num_mv, den_mv) {
                    if d.mean.abs() > 1e-30 {
                        // Ratio with propagated error: σ(a/b) = |a/b| * sqrt((σa/a)² + (σb/b)²)
                        let ratio = n.mean / d.mean;
                        let rel_err_n = if n.mean.abs() > 1e-30 { n.stderr / n.mean.abs() } else { 0.0 };
                        let rel_err_d = if d.mean.abs() > 1e-30 { d.stderr / d.mean.abs() } else { 0.0 };
                        let ratio_stderr = ratio.abs() * (rel_err_n.powi(2) + rel_err_d.powi(2)).sqrt();

                        paired_metrics.insert(
                            format!("{}_ratio", metric_name),
                            MetricValue { mean: ratio, stderr: ratio_stderr, values: None },
                        );
                    }

                    // Difference
                    let diff = n.mean - d.mean;
                    let diff_stderr = (n.stderr.powi(2) + d.stderr.powi(2)).sqrt();
                    paired_metrics.insert(
                        format!("{}_diff", metric_name),
                        MetricValue { mean: diff, stderr: diff_stderr, values: None },
                    );
                }
            }

            Some(PairedGroupMetrics {
                group_key: key,
                variables: num_vars,
                numerator: num_gm,
                denominator: den_gm,
                paired_metrics,
            })
        })
        .collect()
}

fn compute_single_group(
    key: &str,
    members: &[&AnalyzedRun],
    tagged_cell: usize,
    metric_defs: &BTreeMap<String, String>,
) -> GroupMetrics {
    let variables = members[0].variables.clone();
    let n_seeds = members.len();

    let mut metrics = BTreeMap::new();
    let mut per_seed = BTreeMap::new();

    for (metric_name, metric_expr) in metric_defs {
        let values: Vec<f64> = members
            .iter()
            .filter_map(|m| extract_metric(&m.result, metric_expr, tagged_cell))
            .collect();

        if !values.is_empty() {
            per_seed.insert(metric_name.clone(), values.clone());
            metrics.insert(metric_name.clone(), aggregate_values(&values));
        }
    }

    GroupMetrics {
        group_key: key.to_string(),
        variables,
        n_seeds,
        metrics,
        per_seed,
    }
}

fn aggregate_values(values: &[f64]) -> MetricValue {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let stderr = if values.len() > 1 {
        let var: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt() / n.sqrt()
    } else {
        0.0
    };
    MetricValue {
        mean,
        stderr,
        values: Some(values.to_vec()),
    }
}

// ============================================================================
// Plotting (SVG via plotters)
// ============================================================================

fn generate_plot(
    cfg: &PlotConfig,
    paired_groups: &[PairedGroupMetrics],
    output_dir: &Path,
) -> Result<()> {
    use plotters::prelude::*;

    // Determine which variables are NOT the x-axis — these become series keys
    let all_vars: std::collections::BTreeSet<String> = paired_groups.iter()
        .flat_map(|pg| pg.variables.keys().cloned())
        .collect();

    let series_vars: Vec<String> = all_vars.iter()
        .filter(|v| v.as_str() != cfg.x)
        .cloned()
        .collect();

    // Group points by series key
    let mut series_map: BTreeMap<String, Vec<(f64, f64, f64)>> = BTreeMap::new();

    for pg in paired_groups {
        let x_val = pg.variables.get(&cfg.x)
            .and_then(|s| s.parse::<f64>().ok());
        let y_metric = pg.paired_metrics.get(&cfg.y)
            .or_else(|| pg.numerator.metrics.get(&cfg.y));

        if let (Some(x), Some(m)) = (x_val, y_metric) {
            let x_transformed = match cfg.x_transform.as_deref() {
                Some("inverse_sqrt") => 1.0 / x.sqrt(),
                Some("log") => x.ln(),
                _ => x,
            };

            let series_key = if series_vars.is_empty() {
                "data".to_string()
            } else {
                series_vars.iter()
                    .map(|v| format!("{}={}", v, pg.variables.get(v).unwrap_or(&"?".to_string())))
                    .collect::<Vec<_>>()
                    .join(", ")
            };

            series_map.entry(series_key).or_default().push((x_transformed, m.mean, m.stderr));
        }
    }

    if series_map.is_empty() {
        anyhow::bail!("No data points for plot '{}' (x={}, y={})", cfg.title, cfg.x, cfg.y);
    }

    // Sort each series by x
    for points in series_map.values_mut() {
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Compute global axis ranges
    let all_points: Vec<&(f64, f64, f64)> = series_map.values().flat_map(|v| v.iter()).collect();
    let x_min = all_points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = all_points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = all_points.iter().map(|p| p.1 - p.2.abs()).fold(f64::INFINITY, f64::min);
    let y_max = all_points.iter().map(|p| p.1 + p.2.abs()).fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min).max(x_max.abs() * 0.01) * 0.15;
    let y_pad = (y_max - y_min).max(y_max.abs() * 0.01) * 0.15;

    // Apply optional axis overrides from TOML
    let plot_y_min = cfg.y_min.unwrap_or(y_min - y_pad);
    let plot_y_max = cfg.y_max.unwrap_or(y_max + y_pad);

    let out_path = output_dir.join(&cfg.output);
    let root = SVGBackend::new(&out_path, (800, 560)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_label = cfg.x_label.as_deref().unwrap_or(&cfg.x);
    let y_label = cfg.y_label.as_deref().unwrap_or(&cfg.y);

    let mut chart = ChartBuilder::on(&root)
        .caption(&cfg.title, ("sans-serif", 26).into_font().color(&BLACK))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(
            (x_min - x_pad)..(x_max + x_pad),
            plot_y_min..plot_y_max,
        )?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_label_style(("sans-serif", 18))
        .y_label_style(("sans-serif", 18))
        .axis_desc_style(("sans-serif", 20))
        .light_line_style(TRANSPARENT)
        .bold_line_style(RGBAColor(200, 200, 200, 0.3))
        .draw()?;

    // Color palette for multiple series
    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK];

    for (i, (series_key, points)) in series_map.iter().enumerate() {
        let color = colors[i % colors.len()];

        // Draw error bars
        if cfg.error_bars {
            for &(x, y, e) in points {
                if e > 0.0 {
                    let cap = (x_max - x_min) * 0.01;
                    chart.draw_series(std::iter::once(
                        PathElement::new(vec![(x, y - e), (x, y + e)], color.mix(0.5)),
                    ))?;
                    // Caps
                    chart.draw_series(std::iter::once(
                        PathElement::new(vec![(x - cap, y - e), (x + cap, y - e)], color.mix(0.5)),
                    ))?;
                    chart.draw_series(std::iter::once(
                        PathElement::new(vec![(x - cap, y + e), (x + cap, y + e)], color.mix(0.5)),
                    ))?;
                }
            }
        }

        // Draw line
        chart.draw_series(LineSeries::new(
            points.iter().map(|&(x, y, _)| (x, y)),
            color.stroke_width(2),
        ))?
        .label(series_key.clone())
        .legend(move |(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
        });

        // Draw points
        chart.draw_series(
            points.iter().map(|&(x, y, _)| Circle::new((x, y), 5, color.filled())),
        )?;
    }

    // Draw legend if multiple series
    if series_map.len() > 1 {
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .label_font(("sans-serif", 14))
            .position(plotters::chart::SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    eprintln!("  Plot saved: {}", out_path.display());
    Ok(())
}

fn generate_figure(
    cfg: &FigureConfig,
    paired_groups: &[PairedGroupMetrics],
    output_dir: &Path,
) -> Result<()> {
    use plotters::prelude::*;

    let [n_rows, n_cols] = cfg.layout;
    let out_path = output_dir.join(&cfg.output);

    let panel_w = cfg.width / n_cols as u32;
    let panel_h = (cfg.height - 40) / n_rows as u32; // 40px for title
    let total_w = cfg.width;
    let total_h = cfg.height;

    let root = SVGBackend::new(&out_path, (total_w, total_h)).into_drawing_area();
    root.fill(&WHITE)?;

    // Draw figure title
    root.titled(&cfg.title, ("sans-serif", 20))?;

    let panels = root.split_evenly((n_rows, n_cols));

    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK];

    for (idx, panel_cfg) in cfg.panels.iter().enumerate() {
        if idx >= panels.len() { break; }
        let area = &panels[idx];

        // Determine series variables (everything except x)
        let all_vars: std::collections::BTreeSet<String> = paired_groups.iter()
            .flat_map(|pg| pg.variables.keys().cloned())
            .collect();
        let series_vars: Vec<String> = all_vars.iter()
            .filter(|v| v.as_str() != panel_cfg.x)
            .cloned()
            .collect();

        // Extract data points grouped by series
        let mut series_map: BTreeMap<String, Vec<(f64, f64, f64)>> = BTreeMap::new();

        for pg in paired_groups {
            let x_val = pg.variables.get(&panel_cfg.x)
                .and_then(|s| s.parse::<f64>().ok());
            let y_metric = pg.paired_metrics.get(&panel_cfg.y)
                .or_else(|| pg.numerator.metrics.get(&panel_cfg.y));

            if let (Some(x), Some(m)) = (x_val, y_metric) {
                let x_t = match panel_cfg.x_transform.as_deref() {
                    Some("inverse_sqrt") => 1.0 / x.sqrt(),
                    Some("log") => x.ln(),
                    _ => x,
                };
                let key = if series_vars.is_empty() {
                    "data".to_string()
                } else {
                    series_vars.iter()
                        .map(|v| format!("{}={}", v, pg.variables.get(v).unwrap_or(&"?".to_string())))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                series_map.entry(key).or_default().push((x_t, m.mean, m.stderr));
            }
        }

        if series_map.is_empty() { continue; }

        for pts in series_map.values_mut() {
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        let all_pts: Vec<&(f64, f64, f64)> = series_map.values().flat_map(|v| v.iter()).collect();
        let x_min = all_pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let x_max = all_pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let y_min = all_pts.iter().map(|p| p.1 - p.2.abs()).fold(f64::INFINITY, f64::min);
        let y_max = all_pts.iter().map(|p| p.1 + p.2.abs()).fold(f64::NEG_INFINITY, f64::max);
        let x_pad = (x_max - x_min).max(x_max.abs() * 0.01) * 0.15;
        let y_pad = (y_max - y_min).max(y_max.abs() * 0.01) * 0.15;

        let plot_y_min = panel_cfg.y_min.unwrap_or(y_min - y_pad);
        let plot_y_max = panel_cfg.y_max.unwrap_or(y_max + y_pad);

        let panel_title = panel_cfg.title.as_deref().unwrap_or(&panel_cfg.y);
        let x_label = panel_cfg.x_label.as_deref().unwrap_or(&panel_cfg.x);
        let y_label = panel_cfg.y_label.as_deref().unwrap_or(&panel_cfg.y);

        let mut chart = ChartBuilder::on(area)
            .caption(panel_title, ("sans-serif", 16))
            .margin(10)
            .margin_left(15)
            .margin_bottom(10)
            .x_label_area_size(40)
            .y_label_area_size(65)
            .build_cartesian_2d(
                (x_min - x_pad)..(x_max + x_pad),
                plot_y_min..plot_y_max,
            )?;

        chart.configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .x_label_style(("sans-serif", 14))
            .y_label_style(("sans-serif", 14))
            .axis_desc_style(("sans-serif", 15))
            .light_line_style(TRANSPARENT)
            .bold_line_style(RGBAColor(200, 200, 200, 0.3))
            .draw()?;

        // Optional horizontal reference line
        if let Some(h) = panel_cfg.h_line {
            chart.draw_series(LineSeries::new(
                vec![(x_min - x_pad, h), (x_max + x_pad, h)],
                BLACK.mix(0.3).stroke_width(1),
            ))?;
        }

        for (i, (key, pts)) in series_map.iter().enumerate() {
            let color = colors[i % colors.len()];

            if panel_cfg.error_bars {
                let cap = (x_max - x_min) * 0.01;
                for &(x, y, e) in pts {
                    if e > 0.0 {
                        chart.draw_series(std::iter::once(
                            PathElement::new(vec![(x, y - e), (x, y + e)], color.mix(0.5)),
                        ))?;
                        chart.draw_series(std::iter::once(
                            PathElement::new(vec![(x - cap, y - e), (x + cap, y - e)], color.mix(0.5)),
                        ))?;
                        chart.draw_series(std::iter::once(
                            PathElement::new(vec![(x - cap, y + e), (x + cap, y + e)], color.mix(0.5)),
                        ))?;
                    }
                }
            }

            chart.draw_series(LineSeries::new(
                pts.iter().map(|&(x, y, _)| (x, y)),
                color.stroke_width(2),
            ))?
            .label(key.clone())
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 15, y)], color.stroke_width(2))
            });

            chart.draw_series(
                pts.iter().map(|&(x, y, _)| Circle::new((x, y), 4, color.filled())),
            )?;
        }

        if series_map.len() > 1 {
            chart.configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .label_font(("sans-serif", 10))
                .draw()?;
        }
    }

    root.present()?;
    eprintln!("  Figure saved: {}", out_path.display());
    Ok(())
}

// ============================================================================
// Diagnostic comparison: side-by-side soft vs ctrl panels
// ============================================================================

/// Generate a multi-row diagnostic comparison of two runs.
/// Rows: L_n time series, displacement speed, trajectory, MSD/4t
pub fn generate_comparison(
    soft_path: &Path,
    ctrl_path: &Path,
    output: &Path,
    tagged_cell: usize,
    subsample: usize,
    soft_label: &str,
    ctrl_label: &str,
    config_path: Option<&Path>,
) -> Result<()> {
    let diag = if let Some(cfg_path) = config_path {
        // Load TOML config
        let toml_str = std::fs::read_to_string(cfg_path)
            .with_context(|| format!("Failed to read compare config: {}", cfg_path.display()))?;
        let cfg: DiagnosticConfig = toml::from_str(&toml_str)
            .with_context(|| format!("Failed to parse compare config: {}", cfg_path.display()))?;
        eprintln!("Loaded compare config: {} ({} panel overrides)", cfg_path.display(), cfg.panels.len());
        cfg
    } else {
        DiagnosticConfig {
            seeds: vec![],
            output: output.display().to_string(),
            ln_range: default_ln_range(),
            speed_max: default_speed_max(),
            msd_lag_max: default_msd_lag_max(),
            panels: vec![],
        }
    };
    generate_comparison_with_config(
        soft_path, ctrl_path, output,
        tagged_cell, subsample, 0.0, // tau=0 signals "read from trajectory"
        soft_label, ctrl_label, &diag,
    )
}

/// Generate diagnostic comparison with full config — overlaid panels.
/// Both soft and ctrl are drawn on the same axes for direct comparison.
/// Layout: 2 rows × 4 columns = 8 panels
///   Row 0: Trajectory | MSD/Δt | L_n(t) | L_n histogram
///   Row 1: Speed bursts | G(v_i) | D_eff bars | Summary text
fn generate_comparison_with_config(
    soft_path: &Path,
    ctrl_path: &Path,
    output: &Path,
    tagged_cell: usize,
    subsample: usize,
    tau: f64,
    soft_label: &str,
    ctrl_label: &str,
    diag: &DiagnosticConfig,
) -> Result<()> {
    use plotters::prelude::*;
    use plotters::style::RGBAColor;

    // Load both trajectories
    eprintln!("Loading soft: {}", soft_path.display());
    let soft_traj_path = if soft_path.is_dir() { soft_path.join("trajectory.txt") } else { soft_path.to_path_buf() };
    let ctrl_traj_path = if ctrl_path.is_dir() { ctrl_path.join("trajectory.txt") } else { ctrl_path.to_path_buf() };

    let soft_traj = super::io::load_trajectory_subsample(&soft_traj_path, subsample)?;
    let ctrl_traj = super::io::load_trajectory_subsample(&ctrl_traj_path, subsample)?;
    let soft_pos = super::io::unwrap_trajectory(&soft_traj);
    let ctrl_pos = super::io::unwrap_trajectory(&ctrl_traj);

    // Use tau from trajectory header/checkpoint if caller passed 0 (auto-detect)
    let tau = if tau <= 0.0 {
        soft_traj.params.tau.unwrap_or_else(|| {
            eprintln!("  WARNING: tau unknown — using 10000 as last resort");
            10000.0
        })
    } else { tau };

    let tc = tagged_cell;
    let soft_color = RGBAColor(220, 60, 60, 1.0);   // red
    let ctrl_color = RGBAColor(52, 152, 219, 1.0);    // blue
    let soft_alpha = RGBAColor(220, 60, 60, 0.5);
    let ctrl_alpha = RGBAColor(52, 152, 219, 0.5);
    let pop_dash = RGBAColor(120, 120, 120, 0.4);

    // Precompute data for both runs
    struct RunData {
        ln_series: Vec<f64>,
        t_tau: Vec<f64>,
        speeds: Vec<f64>,
        t_speed: Vec<f64>,
        inst_speeds: Vec<f64>,      // instantaneous |v| from trajectory vx,vy (per frame)
        raw_x: Vec<f64>,
        raw_y: Vec<f64>,
        uw_x: Vec<f64>,   // unwrapped
        uw_y: Vec<f64>,
        msd_t: Vec<(f64, f64)>,     // (lag_tau, msd/Δt) for cell tc
        msd_pop: Vec<(f64, f64)>,   // population msd/Δt
        vx: Vec<f64>,               // displacement velocities
        vy: Vec<f64>,
        d_cell0: f64,
        d_pop: f64,
        mean_ln: f64,
        mean_speed: f64,
        lx: f64,
        ly: f64,
    }

    fn extract_data(traj: &super::io::Trajectory, pos: &super::io::UnwrappedPositions, tc: usize, tau: f64) -> RunData {
        let n_times = pos.n_times;
        let n_cells = pos.n_cells;
        let dt = if n_times > 1 { pos.times[1] - pos.times[0] } else { 1.0 };
        let cell0_idx = pos.cell_ids.iter().position(|&id| id == tc as u32).unwrap_or(0);

        // Build ln_series aligned to unwrapped positions (pos.times),
        // NOT all traj.frames — unwrap_trajectory may skip incomplete frames.
        let traj_map: std::collections::HashMap<i64, &std::collections::HashMap<u32, super::io::CellSnapshot>> =
            traj.frames.iter().map(|(t, cells)| ((*t * 1e6) as i64, cells)).collect();
        let mut ln_series = Vec::with_capacity(n_times);
        let mut inst_speeds = Vec::with_capacity(n_times);
        for &t in &pos.times {
            let key = (t * 1e6) as i64;
            if let Some(cells) = traj_map.get(&key) {
                if let Some(snap) = cells.get(&(tc as u32)) {
                    ln_series.push(snap.l_n);
                    inst_speeds.push((snap.vx * snap.vx + snap.vy * snap.vy).sqrt());
                } else {
                    ln_series.push(f64::NAN);
                    inst_speeds.push(f64::NAN);
                }
            } else {
                ln_series.push(f64::NAN);
                inst_speeds.push(f64::NAN);
            }
        }
        let t_tau: Vec<f64> = (0..ln_series.len()).map(|i| i as f64 * dt / tau).collect();

        // Displacement speed
        let mut speeds = Vec::new();
        let mut vx_all = Vec::new();
        let mut vy_all = Vec::new();
        for i in 1..n_times {
            let dx = pos.positions[i][cell0_idx][0] - pos.positions[i-1][cell0_idx][0];
            let dy = pos.positions[i][cell0_idx][1] - pos.positions[i-1][cell0_idx][1];
            speeds.push((dx*dx + dy*dy).sqrt() / dt);
            vx_all.push(dx / dt);
            vy_all.push(dy / dt);
        }
        let t_speed: Vec<f64> = (0..speeds.len()).map(|i| (i as f64 + 0.5) * dt / tau).collect();

        // Raw trajectory (wrapped)
        let raw_x: Vec<f64> = traj.frames.iter().filter_map(|(_, c)| c.get(&(tc as u32)).map(|s| s.x)).collect();
        let raw_y: Vec<f64> = traj.frames.iter().filter_map(|(_, c)| c.get(&(tc as u32)).map(|s| s.y)).collect();
        // Unwrapped trajectory
        let uw_x: Vec<f64> = (0..n_times).map(|i| pos.positions[i][cell0_idx][0]).collect();
        let uw_y: Vec<f64> = (0..n_times).map(|i| pos.positions[i][cell0_idx][1]).collect();

        // MSD/Δt for cell tc and population
        let max_lag = n_times / 2;
        let lag_8tau = ((8.0 * tau / dt).round() as usize).min(max_lag);
        // Sample lags densely in the visible 0..8τ window (200 points there)
        // and ensure lag_8tau itself is included for D_eff readout. The
        // earlier scheme spread 100 lags over the full max_lag (~100τ),
        // leaving only ~8 points inside the plotted 0..8τ range.
        let n_lags_visible = 200usize;
        let stride = (lag_8tau / n_lags_visible).max(1);
        let mut lag_set: std::collections::BTreeSet<usize> =
            (1..=10.min(lag_8tau)).collect();
        let mut l = stride;
        while l <= lag_8tau {
            lag_set.insert(l);
            l += stride;
        }
        lag_set.insert(lag_8tau);
        let lags: Vec<usize> = lag_set.into_iter().collect();

        let mut msd_t = Vec::new();
        let mut msd_pop = Vec::new();
        let mut d_cell0 = 0.0;
        let mut d_pop = 0.0;
        for &lag in &lags {
            let lag_time = lag as f64 * dt;
            let n_origins = n_times - lag;
            if n_origins < 2 { break; }
            let mut cell_sum = 0.0f64;
            let mut pop_sum = 0.0f64;
            for t0 in 0..n_origins {
                let ti = t0 + lag;
                let dx = pos.positions[ti][cell0_idx][0] - pos.positions[t0][cell0_idx][0];
                let dy = pos.positions[ti][cell0_idx][1] - pos.positions[t0][cell0_idx][1];
                cell_sum += dx*dx + dy*dy;
                let mut s = 0.0;
                for ci in 0..n_cells {
                    let ddx = pos.positions[ti][ci][0] - pos.positions[t0][ci][0];
                    let ddy = pos.positions[ti][ci][1] - pos.positions[t0][ci][1];
                    s += ddx*ddx + ddy*ddy;
                }
                pop_sum += s / n_cells as f64;
            }
            let msd_cell = cell_sum / n_origins as f64;
            let msd_p = pop_sum / n_origins as f64;
            let lt = lag_time / tau;
            msd_t.push((lt, msd_cell / lag_time));
            msd_pop.push((lt, msd_p / lag_time));
            if lag == lag_8tau {
                d_cell0 = msd_cell / (4.0 * lag_time);
                d_pop = msd_p / (4.0 * lag_time);
            }
        }

        let mean_ln = if ln_series.is_empty() { 0.0 } else { ln_series.iter().sum::<f64>() / ln_series.len() as f64 };
        let mean_speed = if speeds.is_empty() { 0.0 } else { speeds.iter().sum::<f64>() / speeds.len() as f64 };

        RunData { ln_series, t_tau, speeds, t_speed, inst_speeds, raw_x, raw_y, uw_x, uw_y,
                  msd_t, msd_pop, vx: vx_all, vy: vy_all, d_cell0, d_pop, mean_ln, mean_speed,
                  lx: pos.lx, ly: pos.ly }
    }

    let sd = extract_data(&soft_traj, &soft_pos, tc, tau);
    let cd = extract_data(&ctrl_traj, &ctrl_pos, tc, tau);

    // --- Determine panel layout from TOML or default ---
    let default_panels: Vec<ComparePanelConfig> = vec![
        ComparePanelConfig { panel_type: "trajectory".into(), title: None, x_range: None, y_range: None, log_x: false, log_y: false, bins: None, show_population: true, gaussian_ref: true },
        ComparePanelConfig { panel_type: "msd_t".into(), title: None, x_range: None, y_range: None, log_x: false, log_y: false, bins: None, show_population: true, gaussian_ref: true },
        ComparePanelConfig { panel_type: "ln_timeseries".into(), title: None, x_range: None, y_range: None, log_x: false, log_y: false, bins: None, show_population: true, gaussian_ref: true },
        ComparePanelConfig { panel_type: "ln_histogram".into(), title: None, x_range: None, y_range: None, log_x: false, log_y: false, bins: Some(40), show_population: true, gaussian_ref: true },
        ComparePanelConfig { panel_type: "speed_bursts".into(), title: None, x_range: None, y_range: None, log_x: false, log_y: false, bins: None, show_population: true, gaussian_ref: true },
        ComparePanelConfig { panel_type: "gvi".into(), title: None, x_range: None, y_range: None, log_x: false, log_y: false, bins: None, show_population: true, gaussian_ref: true },
        ComparePanelConfig { panel_type: "deff_bar".into(), title: None, x_range: None, y_range: None, log_x: false, log_y: false, bins: None, show_population: true, gaussian_ref: true },
        ComparePanelConfig { panel_type: "summary".into(), title: None, x_range: None, y_range: None, log_x: false, log_y: false, bins: None, show_population: true, gaussian_ref: true },
    ];
    let active_panels = if diag.panels.is_empty() { &default_panels } else { &diag.panels };
    let n_panels = active_panels.len();

    // Dynamic grid: prefer 4 columns, compute rows needed
    let n_cols = if n_panels <= 2 { n_panels as u32 }
                 else if n_panels == 3 || n_panels == 6 || n_panels == 9 { 3 }
                 else if n_panels <= 4 { n_panels as u32 }
                 else { 4 };
    let n_rows = ((n_panels as u32) + n_cols - 1) / n_cols;
    let pw = 340u32;
    let ph = 280u32;
    let title_h = 45u32;
    let total_w = pw * n_cols;
    let total_h = ph * n_rows + title_h;

    let root = SVGBackend::new(output, (total_w, total_h)).into_drawing_area();
    root.fill(&WHITE)?;
    // Reserve a fixed title area then split the remainder into panels
    let (title_area, chart_area) = root.split_vertically(title_h);
    title_area.titled(
        &format!("{} vs {} — Cell {} Diagnostics", soft_label, ctrl_label, tc),
        ("sans-serif", 18),
    )?;
    let panels = chart_area.split_evenly((n_rows as usize, n_cols as usize));

    for (panel_idx, panel_cfg) in active_panels.iter().enumerate() {
        if panel_idx >= panels.len() { break; }
        let area = &panels[panel_idx];
        let panel_label = (b'a' + panel_idx as u8) as char;

        match panel_cfg.panel_type.as_str() {

    // ===== trajectory: Cell 0 wrapped path =====
    "trajectory" => {
        let lx = sd.lx;
        let ly = sd.ly;

        let mut chart = ChartBuilder::on(area)
            .caption(panel_cfg.title.as_deref().unwrap_or(&format!("({}) Cell 0 Trajectory", panel_label)), ("sans-serif", 14))
            .margin(8).x_label_area_size(24).y_label_area_size(40)
            .build_cartesian_2d(0.0..lx, 0.0..ly)?;
        chart.configure_mesh()
            .disable_axes()
            .disable_mesh()
            .draw()?;
        // Custom axes: just the border lines
        chart.plotting_area().draw(&PathElement::new(
            vec![(0.0, 0.0), (lx, 0.0)], BLACK.mix(0.6).stroke_width(1),
        ))?;
        chart.plotting_area().draw(&PathElement::new(
            vec![(0.0, 0.0), (0.0, ly)], BLACK.mix(0.6).stroke_width(1),
        ))?;
        // Custom tick labels using Text elements positioned just outside axes
        // X-axis: "0" at left, domain size at right
        area.draw_text("0", &("sans-serif", 10).into_text_style(area).color(&BLACK),
            (48, area.dim_in_pixel().1 as i32 - 12))?;
        area.draw_text(&format!("{:.0}", lx), &("sans-serif", 10).into_text_style(area).color(&BLACK),
            (area.dim_in_pixel().0 as i32 - 30, area.dim_in_pixel().1 as i32 - 12))?;
        // Y-axis: "0" at bottom, domain size at top
        area.draw_text("0", &("sans-serif", 10).into_text_style(area).color(&BLACK),
            (28, area.dim_in_pixel().1 as i32 - 20))?;
        area.draw_text(&format!("{:.0}", ly), &("sans-serif", 10).into_text_style(area).color(&BLACK),
            (16, 20))?;
        // Draw frame on top and right edges
        chart.plotting_area().draw(&PathElement::new(
            vec![(0.0, ly), (lx, ly), (lx, 0.0)], BLACK.mix(0.6).stroke_width(1),
        ))?;

        // Draw wrapped trajectory: break line at periodic jumps
        for (rx, ry, color, full_color, label) in [
            (&sd.raw_x, &sd.raw_y, soft_alpha, soft_color, soft_label),
            (&cd.raw_x, &cd.raw_y, ctrl_alpha, ctrl_color, ctrl_label),
        ] {
            let fc = full_color;
            let mut seg_start = 0usize;
            let mut first_seg = true;
            for i in 1..rx.len() {
                let dx = (rx[i] - rx[i-1]).abs();
                let dy = (ry[i] - ry[i-1]).abs();
                if dx > lx * 0.4 || dy > ly * 0.4 {
                    if i > seg_start + 1 {
                        let series = chart.draw_series(LineSeries::new(
                            (seg_start..i).map(|j| (rx[j], ry[j])),
                            color.stroke_width(1),
                        ))?;
                        if first_seg {
                            series.label(label)
                                .legend(move |(x,y)| Rectangle::new([(x,y-2),(x+12,y+2)], fc.filled()));
                            first_seg = false;
                        }
                    }
                    seg_start = i;
                }
            }
            if rx.len() > seg_start + 1 {
                let series = chart.draw_series(LineSeries::new(
                    (seg_start..rx.len()).map(|j| (rx[j], ry[j])),
                    color.stroke_width(1),
                ))?;
                if first_seg {
                    series.label(label)
                        .legend(move |(x,y)| Rectangle::new([(x,y-2),(x+12,y+2)], fc.filled()));
                }
            }
        }
        // Start markers (circles)
        if !sd.raw_x.is_empty() {
            chart.draw_series(std::iter::once(
                Circle::new((sd.raw_x[0], sd.raw_y[0]), 4, soft_color.filled()),
            ))?;
        }
        if !cd.raw_x.is_empty() {
            chart.draw_series(std::iter::once(
                Circle::new((cd.raw_x[0], cd.raw_y[0]), 4, ctrl_color.filled()),
            ))?;
        }
        // End markers (triangles)
        if sd.raw_x.len() > 1 {
            let last = sd.raw_x.len() - 1;
            chart.draw_series(std::iter::once(
                TriangleMarker::new((sd.raw_x[last], sd.raw_y[last]), 5, soft_color.filled()),
            ))?;
        }
        if cd.raw_x.len() > 1 {
            let last = cd.raw_x.len() - 1;
            chart.draw_series(std::iter::once(
                TriangleMarker::new((cd.raw_x[last], cd.raw_y[last]), 5, ctrl_color.filled()),
            ))?;
        }
        chart.configure_series_labels().position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.8)).border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 9)).draw()?;
    }

    // ===== msd_t: MSD/Δt curves =====
    "msd_t" => {
        let x_max_raw = panel_cfg.x_range.map(|r| r[1]).unwrap_or(diag.msd_lag_max);
        let use_log_x = panel_cfg.log_x;
        let use_log_y = panel_cfg.log_y;

        // Helper closures for optional log transform
        let tx = |v: f64| if use_log_x && v > 0.0 { v.ln() } else { v };
        let ty = |v: f64| if use_log_y && v > 0.0 { v.ln() } else { v };

        // Filter and transform data
        let soft_pts: Vec<(f64, f64)> = sd.msd_t.iter()
            .filter(|(x, y)| *x <= x_max_raw && *x > 0.0 && *y > 0.0)
            .map(|&(x, y)| (tx(x), ty(y))).collect();
        let ctrl_pts: Vec<(f64, f64)> = cd.msd_t.iter()
            .filter(|(x, y)| *x <= x_max_raw && *x > 0.0 && *y > 0.0)
            .map(|&(x, y)| (tx(x), ty(y))).collect();
        let pop_pts: Vec<(f64, f64)> = sd.msd_pop.iter()
            .filter(|(x, _)| *x <= x_max_raw && *x > 0.0)
            .map(|&(x, y)| (tx(x), ty(y.max(1e-12)))).collect();

        let all_pts: Vec<&(f64, f64)> = soft_pts.iter().chain(ctrl_pts.iter()).chain(pop_pts.iter()).collect();
        let x_lo = all_pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let x_hi = all_pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let y_lo = all_pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let y_hi = all_pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let x_pad = (x_hi - x_lo).max(0.1) * 0.05;
        let y_pad = (y_hi - y_lo).max(0.1) * 0.1;

        let x_min_c = if use_log_x { x_lo - x_pad } else { 0.0 };
        let x_max_c = x_hi + x_pad;
        let y_min_c = if use_log_y { y_lo - y_pad } else { 0.0 };
        let y_max_c = panel_cfg.y_range.map(|r| ty(r[1])).unwrap_or(y_hi + y_pad);

        let x_label = if use_log_x { "ln(Δt/τ)" } else { "Δt (τ)" };
        let y_label = if use_log_y { "ln(MSD/Δt)" } else { "MSD/Δt" };

        let mut chart = ChartBuilder::on(area)
            .caption(&format!("({}) {}", panel_label, panel_cfg.title.as_deref().unwrap_or("MSD/Δt → 4D_eff")), ("sans-serif", 16))
            .margin(8).x_label_area_size(30).y_label_area_size(50)
            .build_cartesian_2d(x_min_c..x_max_c, y_min_c..y_max_c)?;
        chart.configure_mesh().x_desc(x_label).y_desc(y_label)
            .x_label_style(("sans-serif", 14)).y_label_style(("sans-serif", 14)).light_line_style(TRANSPARENT).bold_line_style(RGBAColor(200, 200, 200, 0.3)).draw()?;

        // Soft cell0
        chart.draw_series(LineSeries::new(
            soft_pts.iter().copied(),
            soft_color.stroke_width(2),
        ))?.label(&format!("{} c0 (D={:.4})", soft_label, sd.d_cell0))
            .legend(move |(x,y)| Rectangle::new([(x,y-2),(x+12,y+2)], soft_color.filled()));
        // Ctrl cell0
        chart.draw_series(LineSeries::new(
            ctrl_pts.iter().copied(),
            ctrl_color.stroke_width(2),
        ))?.label(&format!("{} c0 (D={:.4})", ctrl_label, cd.d_cell0))
            .legend(move |(x,y)| Rectangle::new([(x,y-2),(x+12,y+2)], ctrl_color.filled()));
        // Population (dashed, gray)
        chart.draw_series(LineSeries::new(
            pop_pts.iter().copied(),
            pop_dash.stroke_width(1),
        ))?.label("Population");

        chart.configure_series_labels().position(SeriesLabelPosition::LowerRight)
            .background_style(WHITE.mix(0.8)).border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 10)).draw()?;
        // Top+right frame
        chart.plotting_area().draw(&PathElement::new(
            vec![(x_min_c, y_max_c), (x_max_c, y_max_c), (x_max_c, y_min_c)], BLACK.mix(0.5).stroke_width(1)))?;
    }

    // ===== ln_timeseries: L_n(t) overlaid =====
    "ln_timeseries" => {
        // Use TOML ln_range if specified, otherwise auto-scale to data.
        let valid_ln: Vec<f64> = sd.ln_series.iter().chain(cd.ln_series.iter())
            .copied().filter(|&v| v > 0.5).collect();
        let data_min = valid_ln.iter().copied().fold(f64::INFINITY, f64::min);
        let data_max = valid_ln.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Panel y_range overrides DiagnosticConfig ln_range overrides auto-scale
        let (y_min, y_max) = if let Some([lo, hi]) = panel_cfg.y_range {
            (lo, hi)
        } else {
            let auto_min = ((data_min - 0.03) * 20.0).floor() / 20.0;
            let auto_max = ((data_max + 0.03) * 20.0).ceil() / 20.0;
            (diag.ln_range[0].min(auto_min), diag.ln_range[1].max(auto_max))
        };
        let x_max = sd.t_tau.last().copied().unwrap_or(1.0).max(cd.t_tau.last().copied().unwrap_or(1.0));

        let mut chart = ChartBuilder::on(area)
            .caption(&format!("({}) {}", panel_label, panel_cfg.title.as_deref().unwrap_or("Cell 0 L_n(t)")), ("sans-serif", 16))
            .margin(8).x_label_area_size(30).y_label_area_size(50)
            .build_cartesian_2d(0.0..x_max, y_min..y_max)?;
        chart.configure_mesh().x_desc("t (τ)").y_desc("L_n")
            .x_label_style(("sans-serif", 14)).y_label_style(("sans-serif", 14)).light_line_style(TRANSPARENT).bold_line_style(RGBAColor(200, 200, 200, 0.3)).draw()?;

        chart.draw_series(LineSeries::new(
            sd.t_tau.iter().zip(sd.ln_series.iter()).map(|(&t,&l)| (t, l.min(y_max).max(y_min))),
            soft_alpha.stroke_width(1),
        ))?.label(&format!("{} (μ={:.3})", soft_label, sd.mean_ln))
            .legend(move |(x,y)| Rectangle::new([(x,y-2),(x+12,y+2)], soft_color.filled()));
        chart.draw_series(LineSeries::new(
            cd.t_tau.iter().zip(cd.ln_series.iter()).map(|(&t,&l)| (t, l.min(y_max).max(y_min))),
            ctrl_alpha.stroke_width(1),
        ))?.label(&format!("{} (μ={:.3})", ctrl_label, cd.mean_ln))
            .legend(move |(x,y)| Rectangle::new([(x,y-2),(x+12,y+2)], ctrl_color.filled()));
        // Reference line at L_n = 1
        chart.draw_series(LineSeries::new(
            vec![(0.0, 1.0), (x_max, 1.0)],
            BLACK.mix(0.2).stroke_width(1),
        ))?;

        chart.configure_series_labels().position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.8)).border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 10)).draw()?;
        chart.plotting_area().draw(&PathElement::new(
            vec![(0.0, y_max), (x_max, y_max), (x_max, y_min)], BLACK.mix(0.5).stroke_width(1)))?;
    }

    // ===== ln_histogram: L_n distribution =====
    "ln_histogram" => {
        let n_bins = panel_cfg.bins.unwrap_or(40);
        let all_ln: Vec<f64> = sd.ln_series.iter().chain(cd.ln_series.iter()).copied().collect();
        // Use panel x_range, then fall back to diag.ln_range, then auto-scale
        let (ln_min, ln_max) = if let Some([lo, hi]) = panel_cfg.x_range {
            (lo, hi)
        } else {
            let auto_min = all_ln.iter().copied().fold(f64::INFINITY, f64::min).max(diag.ln_range[0]);
            let auto_max = all_ln.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 0.02;
            (auto_min, auto_max.max(diag.ln_range[1]))
        };
        let bw = (ln_max - ln_min) / n_bins as f64;

        let mut s_hist = vec![0u32; n_bins];
        let mut c_hist = vec![0u32; n_bins];
        for &v in &sd.ln_series {
            let b = ((v - ln_min) / bw).floor() as usize;
            if b < n_bins { s_hist[b] += 1; }
        }
        for &v in &cd.ln_series {
            let b = ((v - ln_min) / bw).floor() as usize;
            if b < n_bins { c_hist[b] += 1; }
        }
        let s_total = sd.ln_series.len().max(1) as f64;
        let c_total = cd.ln_series.len().max(1) as f64;
        let s_density: Vec<f64> = s_hist.iter().map(|&c| c as f64 / (s_total * bw)).collect();
        let c_density: Vec<f64> = c_hist.iter().map(|&c| c as f64 / (c_total * bw)).collect();
        let y_max = s_density.iter().chain(c_density.iter()).copied().fold(0.0f64, f64::max) * 1.2;

        let mut chart = ChartBuilder::on(area)
            .caption(&format!("({}) {}", panel_label, panel_cfg.title.as_deref().unwrap_or("Cell 0 L_n Distribution")), ("sans-serif", 16))
            .margin(8).x_label_area_size(30).y_label_area_size(50)
            .build_cartesian_2d(ln_min..ln_max, 0.0..y_max)?;
        chart.configure_mesh().x_desc("L_n").y_desc("Density")
            .x_label_style(("sans-serif", 14)).y_label_style(("sans-serif", 14)).light_line_style(TRANSPARENT).bold_line_style(RGBAColor(200, 200, 200, 0.3)).draw()?;

        chart.draw_series(s_density.iter().enumerate().map(|(i, &d)| {
            let x0 = ln_min + i as f64 * bw;
            Rectangle::new([(x0, 0.0), (x0 + bw, d)], soft_alpha.filled())
        }))?.label(soft_label)
            .legend(move |(x,y)| Rectangle::new([(x,y-3),(x+12,y+3)], soft_alpha.filled()));
        chart.draw_series(c_density.iter().enumerate().map(|(i, &d)| {
            let x0 = ln_min + i as f64 * bw;
            Rectangle::new([(x0, 0.0), (x0 + bw, d)], ctrl_alpha.filled())
        }))?.label(ctrl_label)
            .legend(move |(x,y)| Rectangle::new([(x,y-3),(x+12,y+3)], ctrl_alpha.filled()));
        chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8)).border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 10)).draw()?;
        chart.plotting_area().draw(&PathElement::new(
            vec![(ln_min, y_max), (ln_max, y_max), (ln_max, 0.0)], BLACK.mix(0.5).stroke_width(1)))?;
    }

    // ===== speed_bursts: Speed |v|(t) =====
    "speed_bursts" => {
        let x_max = panel_cfg.x_range.map(|r| r[1]).unwrap_or_else(|| sd.t_speed.last().copied().unwrap_or(1.0).max(cd.t_speed.last().copied().unwrap_or(1.0)));
        let y_max = panel_cfg.y_range.map(|r| r[1]).unwrap_or(diag.speed_max);
        // Burst threshold from ctrl (μ + 3σ)
        let ctrl_mean = cd.mean_speed;
        let ctrl_std = if cd.speeds.len() > 1 {
            let var = cd.speeds.iter().map(|s| (s - ctrl_mean).powi(2)).sum::<f64>() / (cd.speeds.len() - 1) as f64;
            var.sqrt()
        } else { 0.0 };
        let burst_thresh = ctrl_mean + 3.0 * ctrl_std;

        let mut chart = ChartBuilder::on(area)
            .caption(&format!("({}) {}", panel_label, panel_cfg.title.as_deref().unwrap_or("Cell 0 Speed |v|(t)")), ("sans-serif", 16))
            .margin(8).x_label_area_size(30).y_label_area_size(50)
            .build_cartesian_2d(0.0..x_max, 0.0..y_max)?;
        chart.configure_mesh().x_desc("t (τ)").y_desc("|v|")
            .x_label_style(("sans-serif", 14)).y_label_style(("sans-serif", 14)).light_line_style(TRANSPARENT).bold_line_style(RGBAColor(200, 200, 200, 0.3)).draw()?;

        chart.draw_series(LineSeries::new(
            sd.t_speed.iter().zip(sd.speeds.iter()).map(|(&t,&s)| (t, s.min(y_max))),
            soft_alpha.stroke_width(1),
        ))?.label(soft_label)
            .legend(move |(x,y)| Rectangle::new([(x,y-2),(x+12,y+2)], soft_color.filled()));
        chart.draw_series(LineSeries::new(
            cd.t_speed.iter().zip(cd.speeds.iter()).map(|(&t,&s)| (t, s.min(y_max))),
            ctrl_alpha.stroke_width(1),
        ))?.label(ctrl_label)
            .legend(move |(x,y)| Rectangle::new([(x,y-2),(x+12,y+2)], ctrl_color.filled()));
        // Burst threshold
        chart.draw_series(LineSeries::new(
            vec![(0.0, burst_thresh), (x_max, burst_thresh)],
            BLACK.mix(0.4).stroke_width(1),
        ))?.label(&format!("μ+3σ={:.4}", burst_thresh))
            .legend(|(x,y)| PathElement::new(vec![(x,y),(x+12,y)], BLACK.mix(0.4).stroke_width(1)));
        // v_A reference
        chart.draw_series(LineSeries::new(
            vec![(0.0, 0.01), (x_max, 0.01)],
            BLACK.mix(0.2).stroke_width(1),
        ))?;

        chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8)).border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 9)).draw()?;
        chart.plotting_area().draw(&PathElement::new(
            vec![(0.0, y_max), (x_max, y_max), (x_max, 0.0)], BLACK.mix(0.5).stroke_width(1)))?;
    }

    // ===== gvi: velocity distribution G(v_i) =====
    "gvi" => {
        use super::panels::{draw_gvi_panel, GviSeries, GviPanelOpts, GviMarker};

        // σ for the Gaussian reference: use ctrl (so soft tails show as deviations).
        let (_, _, c_sigma) = super::panels::compute_gvi(&cd.vx, &cd.vy);

        let series = vec![
            GviSeries {
                label: soft_label.to_string(),
                vx: &sd.vx, vy: &sd.vy,
                color: soft_color, marker: GviMarker::Triangle,
            },
            GviSeries {
                label: ctrl_label.to_string(),
                vx: &cd.vx, vy: &cd.vy,
                color: ctrl_color, marker: GviMarker::Circle,
            },
        ];

        let opts = GviPanelOpts {
            title: panel_cfg.title.clone().unwrap_or_else(|| "G(v_i)".to_string()),
            panel_label: Some(panel_label),
            x_max: panel_cfg.x_range.map(|r| r[1]).unwrap_or(0.022),
            y_range: panel_cfg.y_range.map(|r| (r[0], r[1])),
            gaussian_ref_sigma: Some(c_sigma),
            palmieri_fit_index: Some(0), // fit Eq. 5 to soft
            v_a: 0.01,
            palmieri_fit_min_v: None,
            gaussian_sigma_sweep: Vec::new(),
        };
        draw_gvi_panel(area, &series, &opts)?;
    }

    // ===== deff_bar: D_eff bar chart =====
    "deff_bar" => {
        let vals = [sd.d_pop, sd.d_cell0, cd.d_pop, cd.d_cell0];
        let y_max = vals.iter().copied().fold(0.0f64, f64::max) * 1.3;
        let labels = [
            format!("{}\npop", soft_label), format!("{}\nc0", soft_label),
            format!("{}\npop", ctrl_label), format!("{}\nc0", ctrl_label),
        ];

        let mut chart = ChartBuilder::on(area)
            .caption(&format!("({}) {}", panel_label, panel_cfg.title.as_deref().unwrap_or("D_eff at 8τ")), ("sans-serif", 14))
            .margin(8).x_label_area_size(45).y_label_area_size(50)
            .build_cartesian_2d((-0.5f64)..3.5f64, 0.0..y_max)?;
        chart.configure_mesh().y_desc("D_eff")
            .light_line_style(TRANSPARENT)
            .bold_line_style(TRANSPARENT)
            .x_labels(4)
            .x_label_style(("sans-serif", 8)).y_label_style(("sans-serif", 9))
            .x_label_formatter(&|x| {
                let idx = x.round() as usize;
                if idx < 4 { labels[idx].clone() } else { String::new() }
            })
            .draw()?;

        let bar_w = 0.35;
        let colors = [soft_alpha, soft_color, ctrl_alpha, ctrl_color];
        for (i, (&v, c)) in vals.iter().zip(colors.iter()).enumerate() {
            let x = i as f64;
            chart.draw_series(std::iter::once(
                Rectangle::new([(x - bar_w, 0.0), (x + bar_w, v)], c.filled()),
            ))?;
            // Value label on top
            chart.draw_series(std::iter::once(
                Text::new(format!("{:.4}", v), (x, v + y_max * 0.02), ("sans-serif", 8).into_font()),
            ))?;
        }
        chart.plotting_area().draw(&PathElement::new(
            vec![(-0.5, y_max), (3.5, y_max), (3.5, 0.0)], BLACK.mix(0.5).stroke_width(1)))?;
    }

    // ===== summary: text summary =====
    "summary" => {
        area.fill(&WHITE)?;

        let ratio_d = if cd.d_cell0 > 0.0 { sd.d_cell0 / cd.d_cell0 } else { f64::NAN };
        let ratio_ln = if cd.mean_ln > 0.0 { sd.mean_ln / cd.mean_ln } else { f64::NAN };
        let ratio_spd = if cd.mean_speed > 0.0 { sd.mean_speed / cd.mean_speed } else { f64::NAN };

        // Data provenance
        let s_t0 = soft_pos.times.first().copied().unwrap_or(0.0);
        let s_tf = soft_pos.times.last().copied().unwrap_or(0.0);
        let s_dur = (s_tf - s_t0) / tau;
        let c_t0 = ctrl_pos.times.first().copied().unwrap_or(0.0);
        let c_tf = ctrl_pos.times.last().copied().unwrap_or(0.0);
        let c_dur = (c_tf - c_t0) / tau;

        // Extract gamma info from trajectory header extras
        let s_gamma = soft_traj.params.extra.get("gamma_n")
            .or_else(|| soft_traj.params.extra.get("gamma"))
            .cloned().unwrap_or_default();
        let c_gamma = ctrl_traj.params.extra.get("gamma_n")
            .or_else(|| ctrl_traj.params.extra.get("gamma"))
            .cloned().unwrap_or_default();

        let mut lines = vec![
            format!("--- Parameters ---"),
            format!("N={}, Lx={:.0}, v_A={:.3}",
                    soft_traj.params.n_cells, soft_pos.lx, soft_traj.params.v_a),
        ];
        // Add dt from trajectory header extras, or frame interval
        let dt_str = soft_traj.params.extra.get("dt")
            .cloned()
            .unwrap_or_else(|| {
                let frame_dt = if soft_pos.n_times > 1 { soft_pos.times[1] - soft_pos.times[0] } else { 0.0 };
                format!("~{:.0} (frame)", frame_dt)
            });
        lines.push(format!("dt={}, τ={:.0}", dt_str, tau));
        // Add gamma info if available
        if !s_gamma.is_empty() || !c_gamma.is_empty() {
            lines.push(format!("{} γ={}, {} γ={}",
                soft_label, if s_gamma.is_empty() { "?" } else { &s_gamma },
                ctrl_label, if c_gamma.is_empty() { "?" } else { &c_gamma }));
        }
        lines.extend([
            format!("{}: t=[{:.0},{:.0}] ({:.0}τ, {}f)",
                soft_label, s_t0, s_tf, s_dur, soft_pos.n_times),
            format!("{}: t=[{:.0},{:.0}] ({:.0}τ, {}f)",
                ctrl_label, c_t0, c_tf, c_dur, ctrl_pos.n_times),
        ]);
        if subsample > 1 { lines.push(format!("Subsample: {}x", subsample)); }
        lines.extend([
            String::new(),
            format!("--- Observables ---"),
            format!("D_eff c0: {:.4}/{:.4}={:.2}", sd.d_cell0, cd.d_cell0, ratio_d),
            format!("D_eff pop: {:.4}/{:.4}", sd.d_pop, cd.d_pop),
            format!("L_n c0: {:.3}/{:.3}={:.2}", sd.mean_ln, cd.mean_ln, ratio_ln),
            format!("Speed c0: {:.5}/{:.5}={:.2}", sd.mean_speed, cd.mean_speed, ratio_spd),
        ]);

        // Use a dummy chart area for text
        let mut chart = ChartBuilder::on(area)
            .caption(&format!("({}) {}", panel_label, panel_cfg.title.as_deref().unwrap_or("Summary")), ("sans-serif", 14))
            .margin(5).x_label_area_size(0).y_label_area_size(0)
            .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;

        for (i, line) in lines.iter().enumerate() {
            if !line.is_empty() {
                let y = 0.92 - i as f64 * 0.085;
                let font_size = if line.starts_with("---") { 10 } else { 9 };
                chart.draw_series(std::iter::once(
                    Text::new(line.clone(), (0.03, y), ("monospace", font_size).into_font()),
                ))?;
            }
        }
    }

    // ===== ln_speed_correlation: Palmieri Fig 3E — ⟨ΔL_n⟩ vs |v| =====
    "ln_speed_correlation" => {
        // Compute ΔL_n paired with displacement speed for each frame transition
        // speeds[i] = displacement speed from frame i to i+1 (total velocity)
        // ΔL_n[i] = ln_series[i+1] - ln_series[i]
        let n_bins = panel_cfg.bins.unwrap_or(20);

        struct BinnedCorr {
            bins: Vec<(f64, f64, usize)>, // (v_center, mean_delta_ln, count)
        }

        fn compute_binned(ln: &[f64], speeds: &[f64], n_bins: usize) -> BinnedCorr {
            let n = speeds.len().min(ln.len().saturating_sub(1));
            if n == 0 { return BinnedCorr { bins: vec![] }; }

            // Collect (speed, delta_ln) pairs
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for i in 0..n {
                let dl = ln[i + 1] - ln[i];
                let v = speeds[i];
                if ln[i] > 0.5 && ln[i + 1] > 0.5 && v.is_finite() && dl.is_finite() {
                    pairs.push((v, dl));
                }
            }
            if pairs.is_empty() { return BinnedCorr { bins: vec![] }; }

            let v_max = pairs.iter().map(|p| p.0).fold(0.0f64, f64::max);
            let bw = v_max / n_bins as f64;
            if bw <= 0.0 { return BinnedCorr { bins: vec![] }; }

            let mut bin_sum = vec![0.0f64; n_bins];
            let mut bin_count = vec![0usize; n_bins];
            for &(v, dl) in &pairs {
                let b = ((v / bw) as usize).min(n_bins - 1);
                bin_sum[b] += dl;
                bin_count[b] += 1;
            }

            let bins: Vec<(f64, f64, usize)> = (0..n_bins)
                .filter(|&b| bin_count[b] > 5) // require minimum samples per bin
                .map(|b| {
                    let center = (b as f64 + 0.5) * bw;
                    let mean = bin_sum[b] / bin_count[b] as f64;
                    (center, mean, bin_count[b])
                })
                .collect();
            BinnedCorr { bins }
        }

        let sb = compute_binned(&sd.ln_series, &sd.speeds, n_bins);
        let cb = compute_binned(&cd.ln_series, &cd.speeds, n_bins);

        // Determine axis ranges
        let all_v: Vec<f64> = sb.bins.iter().chain(cb.bins.iter()).map(|b| b.0).collect();
        let all_dl: Vec<f64> = sb.bins.iter().chain(cb.bins.iter()).map(|b| b.1).collect();
        let x_max = panel_cfg.x_range.map(|r| r[1]).unwrap_or_else(||
            all_v.iter().copied().fold(0.0f64, f64::max) * 1.1);
        let y_min = panel_cfg.y_range.map(|r| r[0]).unwrap_or_else(||
            all_dl.iter().copied().fold(0.0f64, f64::min) * 1.3);
        let y_max = panel_cfg.y_range.map(|r| r[1]).unwrap_or_else(||
            all_dl.iter().copied().fold(0.0f64, f64::max) * 1.3);
        // Ensure range spans zero
        let y_lo = y_min.min(-0.001);
        let y_hi = y_max.max(0.001);

        let mut chart = ChartBuilder::on(area)
            .caption(&format!("({}) {}", panel_label,
                panel_cfg.title.as_deref().unwrap_or("⟨ΔL_n⟩ vs |v| (Fig 3E)")),
                ("sans-serif", 16))
            .margin(8).x_label_area_size(30).y_label_area_size(50)
            .build_cartesian_2d(0.0..x_max, y_lo..y_hi)?;
        chart.configure_mesh().x_desc("|v|").y_desc("⟨ΔL_n⟩")
            .x_label_style(("sans-serif", 14)).y_label_style(("sans-serif", 14))
            .light_line_style(TRANSPARENT).bold_line_style(RGBAColor(200, 200, 200, 0.3)).draw()?;

        // Zero reference line
        chart.draw_series(LineSeries::new(
            vec![(0.0, 0.0), (x_max, 0.0)],
            BLACK.mix(0.3).stroke_width(1),
        ))?;

        // Soft cell 0
        if !sb.bins.is_empty() {
            chart.draw_series(sb.bins.iter().map(|&(v, dl, _)| {
                Circle::new((v, dl), 5, soft_color.filled())
            }))?.label(soft_label)
                .legend(move |(x, y)| Circle::new((x + 6, y), 4, soft_color.filled()));
            chart.draw_series(LineSeries::new(
                sb.bins.iter().map(|&(v, dl, _)| (v, dl)),
                soft_alpha.stroke_width(1),
            ))?;
        }

        // Ctrl cell 0
        if !cb.bins.is_empty() {
            chart.draw_series(cb.bins.iter().map(|&(v, dl, _)| {
                Circle::new((v, dl), 4, ctrl_color.filled())
            }))?.label(ctrl_label)
                .legend(move |(x, y)| Circle::new((x + 6, y), 4, ctrl_color.filled()));
            chart.draw_series(LineSeries::new(
                cb.bins.iter().map(|&(v, dl, _)| (v, dl)),
                ctrl_alpha.stroke_width(1),
            ))?;
        }

        chart.configure_series_labels().position(SeriesLabelPosition::LowerLeft)
            .background_style(WHITE.mix(0.8)).border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 10)).draw()?;
        chart.plotting_area().draw(&PathElement::new(
            vec![(0.0, y_hi), (x_max, y_hi), (x_max, y_lo)], BLACK.mix(0.5).stroke_width(1)))?;
    }

    // Unknown panel type
    other => {
        eprintln!("  Warning: unknown panel type '{}', skipping", other);
    }

    } // match panel_type
    } // for panel_idx

    root.present()?;
    eprintln!("Diagnostic saved: {}", output.display());
    Ok(())
}

// ============================================================================
// FSS: Finite-size scaling plot
// ============================================================================

/// Per-seed observables computed for FSS.
struct FssSeedData {
    n: usize,
    cond: String,
    seed: String,
    d_eff_cell0: f64,
    d_eff_pop: f64,
    mean_ln: f64,
    mean_speed: f64,
    duration_tau: f64,
    n_frames: usize,
}

/// Generate a multi-panel FSS plot: observables vs N across multiple system sizes.
///
/// Expects directory layout: `{N}c_rho{rho}_{cond}/run_{seed}/trajectory.txt`
pub fn generate_fss_plot(
    base_dir: &Path,
    output: &Path,
    pattern: &str,
    tagged_cell: usize,
    subsample: usize,
    tau: f64,
) -> Result<()> {
    use plotters::prelude::*;

    // 1. Discover runs
    let disc_cfg = DiscoveryConfig {
        pattern: vec![pattern.to_string()],
        trajectory_name: "trajectory.txt".to_string(),
    };
    let runs = discover_study_runs(base_dir, &disc_cfg)?;
    if runs.is_empty() {
        anyhow::bail!("No runs found matching '{}' under {}", pattern, base_dir.display());
    }
    eprintln!("Discovered {} runs", runs.len());

    // 2. Process each run in parallel → FssSeedData
    let seed_data: Vec<FssSeedData> = runs.par_iter().filter_map(|run| {
        let n = run.var_usize("N")?;
        let cond = run.var("cond").to_string();
        let seed = run.var("seed").to_string();

        let traj = match super::io::load_trajectory_subsample(&run.trajectory, subsample) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  Warning: failed to load {}: {}", run.trajectory.display(), e);
                return None;
            }
        };
        let pos = super::io::unwrap_trajectory(&traj);
        let tc = tagged_cell;
        let cell0_idx = pos.cell_ids.iter().position(|&id| id == tc as u32).unwrap_or(0);
        let n_times = pos.n_times;
        let n_cells = pos.n_cells;
        if n_times < 10 { return None; }
        let dt = pos.times[1] - pos.times[0];

        // D_eff at lag=8τ
        let lag_8tau = ((8.0 * tau / dt).round() as usize).min(n_times / 2);
        if lag_8tau == 0 { return None; }
        let n_origins = n_times - lag_8tau;
        let mut cell_sum = 0.0f64;
        let mut pop_sum = 0.0f64;
        for t0 in 0..n_origins {
            let ti = t0 + lag_8tau;
            let dx = pos.positions[ti][cell0_idx][0] - pos.positions[t0][cell0_idx][0];
            let dy = pos.positions[ti][cell0_idx][1] - pos.positions[t0][cell0_idx][1];
            cell_sum += dx * dx + dy * dy;
            let mut s = 0.0;
            for ci in 0..n_cells {
                let ddx = pos.positions[ti][ci][0] - pos.positions[t0][ci][0];
                let ddy = pos.positions[ti][ci][1] - pos.positions[t0][ci][1];
                s += ddx * ddx + ddy * ddy;
            }
            pop_sum += s / n_cells as f64;
        }
        let lag_time = lag_8tau as f64 * dt;
        let d_cell0 = cell_sum / n_origins as f64 / (4.0 * lag_time);
        let d_pop = pop_sum / n_origins as f64 / (4.0 * lag_time);

        // Mean L_n (skip init artifacts < 0.5)
        let ln_vals: Vec<f64> = traj.frames.iter()
            .filter_map(|(_, cells)| cells.get(&(tc as u32)).map(|s| s.l_n))
            .filter(|&v| v > 0.5)
            .collect();
        let mean_ln = if ln_vals.is_empty() { 0.0 } else {
            ln_vals.iter().sum::<f64>() / ln_vals.len() as f64
        };

        // Mean displacement speed
        let mut speed_sum = 0.0f64;
        for i in 1..n_times {
            let dx = pos.positions[i][cell0_idx][0] - pos.positions[i-1][cell0_idx][0];
            let dy = pos.positions[i][cell0_idx][1] - pos.positions[i-1][cell0_idx][1];
            speed_sum += (dx * dx + dy * dy).sqrt() / dt;
        }
        let mean_speed = speed_sum / (n_times - 1).max(1) as f64;

        let t0 = pos.times.first().copied().unwrap_or(0.0);
        let tf = pos.times.last().copied().unwrap_or(0.0);
        let duration_tau = (tf - t0) / tau;

        eprintln!("  N={:>4} {:>4} seed={}: D_c0={:.5} D_pop={:.5} L_n={:.3} dur={:.1}τ ({} frames)",
                  n, cond, seed, d_cell0, d_pop, mean_ln, duration_tau, n_times);

        Some(FssSeedData { n, cond, seed, d_eff_cell0: d_cell0, d_eff_pop: d_pop, mean_ln, mean_speed,
                           duration_tau, n_frames: n_times })
    }).collect();

    if seed_data.is_empty() {
        anyhow::bail!("No valid runs processed");
    }

    // 3. Group by (N, cond) → compute mean ± stderr
    #[derive(Debug)]
    struct FssPoint {
        n: usize,
        cond: String,
        d_eff_cell0: MetricValue,
        d_eff_pop: MetricValue,
        mean_ln: MetricValue,
        mean_speed: MetricValue,
        n_seeds: usize,
        min_dur_tau: f64,
        max_dur_tau: f64,
        mean_dur_tau: f64,
        total_frames: usize,
    }

    let mut groups: BTreeMap<(usize, String), Vec<&FssSeedData>> = BTreeMap::new();
    for sd in &seed_data {
        groups.entry((sd.n, sd.cond.clone())).or_default().push(sd);
    }

    fn metric_from(vals: &[f64]) -> MetricValue {
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let stderr = if vals.len() > 1 {
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            var.sqrt() / n.sqrt()
        } else { 0.0 };
        MetricValue { mean, stderr, values: None }
    }

    let points: Vec<FssPoint> = groups.iter().map(|((n, cond), seeds)| {
        let d_c0: Vec<f64> = seeds.iter().map(|s| s.d_eff_cell0).collect();
        let d_pop: Vec<f64> = seeds.iter().map(|s| s.d_eff_pop).collect();
        let ln: Vec<f64> = seeds.iter().map(|s| s.mean_ln).collect();
        let spd: Vec<f64> = seeds.iter().map(|s| s.mean_speed).collect();
        let durs: Vec<f64> = seeds.iter().map(|s| s.duration_tau).collect();
        let min_dur = durs.iter().copied().fold(f64::INFINITY, f64::min);
        let max_dur = durs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_dur = durs.iter().sum::<f64>() / durs.len() as f64;
        let total_frames: usize = seeds.iter().map(|s| s.n_frames).sum();
        FssPoint {
            n: *n, cond: cond.clone(),
            d_eff_cell0: metric_from(&d_c0),
            d_eff_pop: metric_from(&d_pop),
            mean_ln: metric_from(&ln),
            mean_speed: metric_from(&spd),
            n_seeds: seeds.len(),
            min_dur_tau: min_dur, max_dur_tau: max_dur, mean_dur_tau: mean_dur,
            total_frames,
        }
    }).collect();

    // Find unique conditions and Ns
    let conds: Vec<String> = points.iter().map(|p| p.cond.clone())
        .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let ns: Vec<usize> = points.iter().map(|p| p.n)
        .collect::<std::collections::BTreeSet<_>>().into_iter().collect();

    eprintln!("\nFSS Summary:");
    eprintln!("  Sizes: {:?}", ns);
    eprintln!("  Conditions: {:?}", conds);
    for p in &points {
        eprintln!("  N={:>4} {:>4} ({} seeds, {:.0}-{:.0}τ, {}f): D_c0={:.5}±{:.5}  L_n={:.3}±{:.3}",
                  p.n, p.cond, p.n_seeds, p.min_dur_tau, p.max_dur_tau, p.total_frames,
                  p.d_eff_cell0.mean, p.d_eff_cell0.stderr,
                  p.mean_ln.mean, p.mean_ln.stderr);
    }

    // 4. Render 4-panel SVG
    let n_cols = 2u32;
    let n_rows = 2u32;
    let pw = 450u32;
    let ph = 350u32;
    let title_h = 45u32;
    let total_w = pw * n_cols;
    let total_h = ph * n_rows + title_h;

    let root = SVGBackend::new(output, (total_w, total_h)).into_drawing_area();
    root.fill(&WHITE)?;
    let (title_area, chart_area) = root.split_vertically(title_h);
    title_area.titled("Finite-Size Scaling — Cell 0 Observables", ("sans-serif", 20))?;
    let panels = chart_area.split_evenly((n_rows as usize, n_cols as usize));

    let soft_color = RGBAColor(220, 60, 60, 1.0);
    let ctrl_color = RGBAColor(52, 152, 219, 1.0);

    fn cond_color(cond: &str) -> RGBAColor {
        if cond.contains("soft") { RGBAColor(220, 60, 60, 1.0) }
        else { RGBAColor(52, 152, 219, 1.0) }
    }

    fn cond_label(cond: &str) -> &str {
        if cond.contains("soft") { "Soft (γ=0.35)" }
        else { "Ctrl (γ=1.0)" }
    }

    // Helper: draw a panel with metric vs N for each condition
    fn draw_panel(
        area: &DrawingArea<SVGBackend, plotters::coord::Shift>,
        points: &[FssPoint],
        conds: &[String],
        ns: &[usize],
        title: &str,
        y_label: &str,
        metric_fn: &dyn Fn(&FssPoint) -> &MetricValue,
        h_line: Option<f64>,
    ) -> Result<()> {
        let x_min = *ns.first().unwrap() as f64 * 0.8;
        let x_max = *ns.last().unwrap() as f64 * 1.2;

        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for p in points {
            let m = metric_fn(p);
            y_min = y_min.min(m.mean - m.stderr);
            y_max = y_max.max(m.mean + m.stderr);
        }
        if let Some(h) = h_line {
            y_min = y_min.min(h);
            y_max = y_max.max(h);
        }
        let y_pad = (y_max - y_min).max(y_max.abs() * 0.01) * 0.15;
        y_min -= y_pad;
        y_max += y_pad;

        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 16))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(60)
            .build_cartesian_2d((x_min.ln())..(x_max.ln()), y_min..y_max)?;

        // Force tick marks only at the actual N values
        let ns_ln: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
        let ns_owned: Vec<usize> = ns.to_vec();
        chart.configure_mesh()
            .x_desc("N")
            .y_desc(y_label)
            .x_label_style(("sans-serif", 12))
            .y_label_style(("sans-serif", 12))
            .x_labels(ns.len() + 2)
            .set_all_tick_mark_size(3)
            .x_label_formatter(&|v| {
                // Only show label if this tick is close to an actual N value
                for &n in &ns_owned {
                    if (v - (n as f64).ln()).abs() < 0.05 {
                        return format!("{}", n);
                    }
                }
                String::new()
            })
            .light_line_style(RGBAColor(200, 200, 200, 0.3))
            .draw()?;

        // Draw reference line
        if let Some(h) = h_line {
            chart.draw_series(LineSeries::new(
                vec![(x_min.ln(), h), (x_max.ln(), h)],
                BLACK.mix(0.3).stroke_width(1),
            ))?;
        }

        for cond in conds {
            let color = cond_color(cond);
            let label = cond_label(cond);
            let mut cond_points: Vec<(f64, f64, f64)> = points.iter()
                .filter(|p| p.cond == *cond)
                .map(|p| {
                    let m = metric_fn(p);
                    ((p.n as f64).ln(), m.mean, m.stderr)
                })
                .collect();
            cond_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Error bars
            let x_span = x_max.ln() - x_min.ln();
            for &(x, y, e) in &cond_points {
                if e > 0.0 {
                    let cap = x_span * 0.01;
                    chart.draw_series(std::iter::once(
                        PathElement::new(vec![(x, y - e), (x, y + e)], color.mix(0.6).stroke_width(1)),
                    ))?;
                    chart.draw_series(std::iter::once(
                        PathElement::new(vec![(x - cap, y - e), (x + cap, y - e)], color.mix(0.6).stroke_width(1)),
                    ))?;
                    chart.draw_series(std::iter::once(
                        PathElement::new(vec![(x - cap, y + e), (x + cap, y + e)], color.mix(0.6).stroke_width(1)),
                    ))?;
                }
            }

            // Line + points
            chart.draw_series(LineSeries::new(
                cond_points.iter().map(|&(x, y, _)| (x, y)),
                color.stroke_width(2),
            ))?.label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], color.stroke_width(2)));

            chart.draw_series(
                cond_points.iter().map(|&(x, y, _)| Circle::new((x, y), 5, color.filled())),
            )?;
        }

        chart.configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 11))
            .draw()?;

        // Top+right frame
        chart.plotting_area().draw(&PathElement::new(
            vec![(x_min.ln(), y_max), (x_max.ln(), y_max), (x_max.ln(), y_min)],
            BLACK.mix(0.5).stroke_width(1),
        ))?;

        Ok(())
    }

    // Panel 0: D_eff (cell 0) vs N
    draw_panel(&panels[0], &points, &conds, &ns,
        "(a) D_eff (cell 0) vs N", "D_eff",
        &|p: &FssPoint| &p.d_eff_cell0, None)?;

    // Panel 1: D_eff (population) vs N
    draw_panel(&panels[1], &points, &conds, &ns,
        "(b) D_eff (population) vs N", "D_eff",
        &|p: &FssPoint| &p.d_eff_pop, None)?;

    // Panel 2: L_n vs N
    draw_panel(&panels[2], &points, &conds, &ns,
        "(c) Mean L_n (cell 0) vs N", "⟨L_n⟩",
        &|p: &FssPoint| &p.mean_ln, Some(1.0))?;

    // Panel 3: D_eff ratio (soft/ctrl) vs N
    {
        let area = &panels[3];
        // Compute paired ratios
        let mut ratio_points: Vec<(f64, f64, f64)> = Vec::new(); // (ln_N, ratio, stderr)
        for &n_val in &ns {
            let soft = points.iter().find(|p| p.n == n_val && p.cond.contains("soft"));
            let ctrl = points.iter().find(|p| p.n == n_val && !p.cond.contains("soft"));
            if let (Some(s), Some(c)) = (soft, ctrl) {
                if c.d_eff_cell0.mean > 0.0 {
                    let ratio = s.d_eff_cell0.mean / c.d_eff_cell0.mean;
                    // Propagate error: δ(a/b) = (a/b) * sqrt((δa/a)² + (δb/b)²)
                    let rel_err = ((s.d_eff_cell0.stderr / s.d_eff_cell0.mean).powi(2)
                        + (c.d_eff_cell0.stderr / c.d_eff_cell0.mean).powi(2)).sqrt();
                    ratio_points.push(((n_val as f64).ln(), ratio, ratio * rel_err));
                }
            }
        }

        if ratio_points.is_empty() {
            eprintln!("  WARNING: FSS ratio panel empty — no paired soft/ctrl data found at any N");
        }

        if !ratio_points.is_empty() {
            let x_min = (*ns.first().unwrap() as f64 * 0.8).ln();
            let x_max = (*ns.last().unwrap() as f64 * 1.2).ln();
            let y_min_v = ratio_points.iter().map(|p| p.1 - p.2).fold(f64::INFINITY, f64::min);
            let y_max_v = ratio_points.iter().map(|p| p.1 + p.2).fold(f64::NEG_INFINITY, f64::max);
            let y_pad = (y_max_v - y_min_v).max(0.01) * 0.2;
            let y_lo = (y_min_v - y_pad).min(1.0);
            let y_hi = y_max_v + y_pad;

            let mut chart = ChartBuilder::on(area)
                .caption("(d) L_n = D_soft/D_ctrl vs N", ("sans-serif", 16))
                .margin(10).x_label_area_size(35).y_label_area_size(60)
                .build_cartesian_2d(x_min..x_max, y_lo..y_hi)?;
            let ns_owned2: Vec<usize> = ns.to_vec();
            chart.configure_mesh()
                .x_desc("N").y_desc("D_soft/D_ctrl")
                .x_label_style(("sans-serif", 12)).y_label_style(("sans-serif", 12))
                .x_labels(ns.len() + 2)
                .set_all_tick_mark_size(3)
                .x_label_formatter(&|v| {
                    for &n in &ns_owned2 {
                        if (v - (n as f64).ln()).abs() < 0.05 {
                            return format!("{}", n);
                        }
                    }
                    String::new()
                })
                .light_line_style(RGBAColor(200, 200, 200, 0.3))
                .draw()?;

            // Reference at 1.0
            chart.draw_series(LineSeries::new(
                vec![(x_min, 1.0), (x_max, 1.0)], BLACK.mix(0.3).stroke_width(1),
            ))?;

            let ratio_color = RGBAColor(100, 60, 180, 1.0); // purple
            // Error bars
            let x_span = x_max - x_min;
            for &(x, y, e) in &ratio_points {
                if e > 0.0 {
                    let cap = x_span * 0.01;
                    chart.draw_series(std::iter::once(
                        PathElement::new(vec![(x, y - e), (x, y + e)], ratio_color.mix(0.6).stroke_width(1)),
                    ))?;
                    chart.draw_series(std::iter::once(
                        PathElement::new(vec![(x - cap, y - e), (x + cap, y - e)], ratio_color.mix(0.6).stroke_width(1)),
                    ))?;
                    chart.draw_series(std::iter::once(
                        PathElement::new(vec![(x - cap, y + e), (x + cap, y + e)], ratio_color.mix(0.6).stroke_width(1)),
                    ))?;
                }
            }
            chart.draw_series(LineSeries::new(
                ratio_points.iter().map(|&(x, y, _)| (x, y)),
                ratio_color.stroke_width(2),
            ))?.label("D_soft/D_ctrl")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], ratio_color.stroke_width(2)));
            chart.draw_series(
                ratio_points.iter().map(|&(x, y, _)| Circle::new((x, y), 6, ratio_color.filled())),
            )?;

            // Annotate values
            for &(x, y, e) in &ratio_points {
                chart.draw_series(std::iter::once(
                    Text::new(format!("{:.2}±{:.2}", y, e), (x, y + y_pad * 0.3),
                              ("sans-serif", 10).into_font()),
                ))?;
            }

            chart.configure_series_labels()
                .position(SeriesLabelPosition::UpperRight)
                .background_style(WHITE.mix(0.8))
                .border_style(BLACK.mix(0.3))
                .label_font(("sans-serif", 11))
                .draw()?;
            chart.plotting_area().draw(&PathElement::new(
                vec![(x_min, y_hi), (x_max, y_hi), (x_max, y_lo)],
                BLACK.mix(0.5).stroke_width(1),
            ))?;
        }
    }

    root.present()?;
    eprintln!("\nFSS plot saved: {}", output.display());
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_to_regex() {
        let (re, vars) = pattern_to_regex("{N}c_rho{rho}_{cond}/run_{seed}").unwrap();
        assert_eq!(vars, vec!["N", "rho", "cond", "seed"]);

        let caps = re.captures("100c_rho90_soft/run_01").unwrap();
        assert_eq!(caps.name("N").unwrap().as_str(), "100");
        assert_eq!(caps.name("rho").unwrap().as_str(), "90");
        assert_eq!(caps.name("cond").unwrap().as_str(), "soft");
        assert_eq!(caps.name("seed").unwrap().as_str(), "01");
    }

    #[test]
    fn test_pattern_to_regex_file() {
        let (re, vars) = pattern_to_regex("fss_{N}c_{rho}{cond}.txt").unwrap();
        assert_eq!(vars, vec!["N", "rho", "cond"]);

        let caps = re.captures("fss_400c_85s.txt").unwrap();
        assert_eq!(caps.name("N").unwrap().as_str(), "400");
        assert_eq!(caps.name("rho").unwrap().as_str(), "85");
        assert_eq!(caps.name("cond").unwrap().as_str(), "s");
    }

    #[test]
    fn test_aggregate_values() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mv = aggregate_values(&vals);
        assert!((mv.mean - 3.0).abs() < 1e-10);
        assert!(mv.stderr > 0.0);
    }
}
