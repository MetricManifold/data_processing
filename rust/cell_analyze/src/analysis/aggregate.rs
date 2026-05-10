//! Aggregation primitives: composable operators that turn a list of
//! [`RunAnalysis`] into grouped, averaged, paired, or swept tables.
//!
//! Each operator implements a small typed interface. Studies compose
//! them via the TOML's `[[aggregate]]` blocks (see ARCHITECTURE.md).
//! The set of built-ins covers the FSS, soft-vs-hard, and pairwise
//! workflows needed for the manuscript:
//!
//! - [`GroupBy`] — collect runs sharing a set of variable values
//! - [`MeanStderr`] — mean ± stderr of a metric across replicates
//! - [`Sweep`] — order groups along a numeric axis for plotting
//! - [`PairRatio`] — soft/ctrl style ratios with error propagation
//!
//! These are deliberately *small*. New aggregation needs are met by
//! adding new operators, not by extending these.

use anyhow::{anyhow, Result};
use serde::Serialize;
use std::collections::BTreeMap;

use super::analyze_run::RunAnalysis;
use super::discovery::ScalarValue;
use super::observable::ObservableBag;

// ---------------------------------------------------------------------------
// MetricExtractor
// ---------------------------------------------------------------------------
/// Closure that pulls a scalar metric out of an observable bag.
///
/// Constructed with `metric!(MyObservable, |out| out.field)`. The macro
/// keeps the type machinery hidden from the call site.
pub type MetricExtractor = Box<dyn Fn(&ObservableBag) -> Option<f64> + Send + Sync>;

/// Convenience macro: `metric!(Msd, |out| out.cell0_values[0])`.
#[macro_export]
macro_rules! metric {
    ($obs:ty, $extract:expr) => {
        Box::new(|bag: &$crate::analysis::observable::ObservableBag| {
            bag.get::<$obs>().map($extract)
        }) as $crate::analysis::aggregate::MetricExtractor
    };
}

// ---------------------------------------------------------------------------
// MetricValue (mean + stderr)
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Serialize)]
pub struct MetricValue {
    pub mean: f64,
    pub stderr: f64,
    /// Per-replicate raw values (kept for plotting individual points).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<f64>,
}

impl MetricValue {
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len() as f64;
        if n == 0.0 {
            return Self {
                mean: f64::NAN,
                stderr: 0.0,
                values: vec![],
            };
        }
        let mean = samples.iter().sum::<f64>() / n;
        let stderr = if n > 1.0 {
            let var: f64 = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            (var / n).sqrt()
        } else {
            0.0
        };
        Self {
            mean,
            stderr,
            values: samples.to_vec(),
        }
    }

    /// Ratio with propagated error: σ(a/b) ≈ |a/b|·√((σa/a)² + (σb/b)²).
    pub fn ratio(num: &Self, den: &Self) -> Self {
        if den.mean.abs() < 1e-30 {
            return Self {
                mean: f64::NAN,
                stderr: 0.0,
                values: vec![],
            };
        }
        let r = num.mean / den.mean;
        let rel_n = if num.mean.abs() > 1e-30 {
            num.stderr / num.mean.abs()
        } else {
            0.0
        };
        let rel_d = den.stderr / den.mean.abs();
        Self {
            mean: r,
            stderr: r.abs() * (rel_n.powi(2) + rel_d.powi(2)).sqrt(),
            values: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// GroupBy
// ---------------------------------------------------------------------------
/// Group runs that agree on the values of a chosen list of variables.
/// The key is a stringified tuple e.g. `"N=400,rho=90"`.
pub struct GroupBy<'a> {
    pub vars: &'a [&'a str],
}

#[derive(Clone)]
pub struct Group<'a> {
    pub key: String,
    pub variables: BTreeMap<String, ScalarValue>,
    pub members: Vec<&'a RunAnalysis>,
}

impl<'a> GroupBy<'a> {
    pub fn run(&self, runs: &'a [RunAnalysis]) -> Vec<Group<'a>> {
        let mut acc: BTreeMap<String, Group<'a>> = BTreeMap::new();
        for r in runs {
            let key = self
                .vars
                .iter()
                .map(|v| {
                    let value = r
                        .variables
                        .get(*v)
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                    format!("{}={}", v, value)
                })
                .collect::<Vec<_>>()
                .join(",");
            acc.entry(key.clone())
                .or_insert_with(|| Group {
                    key,
                    variables: self
                        .vars
                        .iter()
                        .filter_map(|v| r.variables.get(*v).map(|s| (v.to_string(), s.clone())))
                        .collect(),
                    members: vec![],
                })
                .members
                .push(r);
        }
        acc.into_values().collect()
    }
}

// ---------------------------------------------------------------------------
// MeanStderr
// ---------------------------------------------------------------------------
/// For each group, compute mean±stderr of the named metrics across its
/// member runs.
pub struct MeanStderr<'a> {
    pub metrics: &'a [(&'a str, MetricExtractor)],
}

#[derive(Clone, Debug, Serialize)]
pub struct GroupSummary {
    pub key: String,
    pub variables: BTreeMap<String, ScalarValue>,
    pub n: usize,
    pub metrics: BTreeMap<String, MetricValue>,
}

impl<'a> MeanStderr<'a> {
    pub fn run(&self, groups: &[Group<'a>]) -> Vec<GroupSummary> {
        groups
            .iter()
            .map(|g| {
                let mut metrics = BTreeMap::new();
                for (name, extract) in self.metrics {
                    let samples: Vec<f64> =
                        g.members.iter().filter_map(|r| extract(&r.bag)).collect();
                    metrics.insert(name.to_string(), MetricValue::from_samples(&samples));
                }
                GroupSummary {
                    key: g.key.clone(),
                    variables: g.variables.clone(),
                    n: g.members.len(),
                    metrics,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Sweep
// ---------------------------------------------------------------------------
/// Order group summaries along a numeric axis (the value of one
/// variable) so they're ready for `metric_vs_x` panels.
pub struct Sweep<'a> {
    pub axis: &'a str,
}

#[derive(Clone, Debug, Serialize)]
pub struct SweepCurve {
    pub axis: String,
    pub points: Vec<SweepPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SweepPoint {
    pub x: f64,
    pub variables: BTreeMap<String, ScalarValue>,
    pub n: usize,
    pub metrics: BTreeMap<String, MetricValue>,
}

impl<'a> Sweep<'a> {
    pub fn run(&self, summaries: &[GroupSummary]) -> Result<SweepCurve> {
        let mut points: Vec<SweepPoint> = summaries
            .iter()
            .map(|s| {
                let x = s
                    .variables
                    .get(self.axis)
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| {
                        anyhow!(
                            "axis variable `{}` missing or non-numeric in group `{}`",
                            self.axis,
                            s.key
                        )
                    })?;
                Ok(SweepPoint {
                    x,
                    variables: s.variables.clone(),
                    n: s.n,
                    metrics: s.metrics.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
        Ok(SweepCurve {
            axis: self.axis.to_string(),
            points,
        })
    }
}

// ---------------------------------------------------------------------------
// PairRatio
// ---------------------------------------------------------------------------
/// For each pair of groups (numerator value vs denominator value of a
/// chosen variable, holding all other variables fixed), produce ratio
/// metrics with error propagation.
pub struct PairRatio<'a> {
    pub pair_var: &'a str,
    pub numerator: &'a str,
    pub denominator: &'a str,
}

#[derive(Clone, Debug, Serialize)]
pub struct PairResult {
    pub key: String,
    pub variables: BTreeMap<String, ScalarValue>,
    pub numerator: GroupSummary,
    pub denominator: GroupSummary,
    pub ratios: BTreeMap<String, MetricValue>,
}

impl<'a> PairRatio<'a> {
    pub fn run(&self, summaries: &[GroupSummary]) -> Vec<PairResult> {
        // Build a lookup keyed by everything-except-pair_var → (cond → summary).
        let mut by_residual: BTreeMap<String, BTreeMap<String, &GroupSummary>> = BTreeMap::new();
        for s in summaries {
            let cond = s
                .variables
                .get(self.pair_var)
                .map(|v| v.to_string())
                .unwrap_or_default();
            let residual = s
                .variables
                .iter()
                .filter(|(k, _)| k.as_str() != self.pair_var)
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(",");
            by_residual
                .entry(residual)
                .or_default()
                .insert(cond, s);
        }

        let mut out = Vec::new();
        for (residual, cond_map) in by_residual {
            let (Some(num), Some(den)) =
                (cond_map.get(self.numerator), cond_map.get(self.denominator))
            else {
                continue;
            };
            let mut ratios = BTreeMap::new();
            for (name, m_n) in &num.metrics {
                if let Some(m_d) = den.metrics.get(name) {
                    ratios.insert(name.clone(), MetricValue::ratio(m_n, m_d));
                }
            }
            out.push(PairResult {
                key: residual,
                variables: num
                    .variables
                    .iter()
                    .filter(|(k, _)| k.as_str() != self.pair_var)
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
                numerator: (*num).clone(),
                denominator: (*den).clone(),
                ratios,
            });
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::observable::{ObservableBag, RunParams};
    use crate::analysis::observables::msd::{Msd, MsdOutput};

    fn fake_run(d: f64, rep: i64, msd_at_lag1: f64) -> RunAnalysis {
        let mut variables = BTreeMap::new();
        variables.insert("d".into(), ScalarValue::Float(d));
        variables.insert("rep".into(), ScalarValue::Int(rep));
        let mut bag = ObservableBag::new();
        bag.insert::<Msd>(MsdOutput {
            lag_times: vec![1.0],
            values: vec![msd_at_lag1 / 2.0],
            cell0_values: vec![msd_at_lag1],
        });
        RunAnalysis {
            directory: format!("d_{}/rep_{}", d, rep).into(),
            variables,
            params: RunParams::default(),
            metadata: crate::analysis::analyze_run::RunMetadata {
                n_cells: 0, lx: 0.0, ly: 0.0, lz: 0.0, dim: 2,
                dt: None, tau: None, tau_source: "test".into(),
                time_start: None, time_end: None, duration: None,
                frame_count: 0, subsample: 1,
                checkpoint_time: None, checkpoint_step: None, checkpoint_tau: None,
                tagged_gamma: None, tagged_v_a: None,
            },
            bag,
        }
    }

    #[test]
    fn groupby_partitions_runs() {
        let runs = vec![
            fake_run(2.0, 1, 0.01),
            fake_run(2.0, 2, 0.02),
            fake_run(4.0, 1, 0.03),
        ];
        let g = GroupBy { vars: &["d"] }.run(&runs);
        assert_eq!(g.len(), 2);
        let by_d2: &Group = g.iter().find(|x| x.key.contains("d=2")).unwrap();
        assert_eq!(by_d2.members.len(), 2);
    }

    #[test]
    fn mean_stderr_computes_correctly() {
        let runs = vec![fake_run(2.0, 1, 0.01), fake_run(2.0, 2, 0.03)];
        let groups = GroupBy { vars: &["d"] }.run(&runs);
        let metrics: &[(&str, MetricExtractor)] = &[(
            "msd_lag1",
            metric!(Msd, |out| out.cell0_values[0]),
        )];
        let summaries = MeanStderr { metrics }.run(&groups);
        assert_eq!(summaries.len(), 1);
        let m = summaries[0].metrics.get("msd_lag1").unwrap();
        assert!((m.mean - 0.02).abs() < 1e-12, "mean = {}", m.mean);
        // stderr of (0.01, 0.03) with n=2: σ = 0.01√2, stderr = σ/√n = 0.01
        assert!((m.stderr - 0.01).abs() < 1e-9, "stderr = {}", m.stderr);
    }

    #[test]
    fn sweep_orders_by_axis() {
        let runs = vec![
            fake_run(8.0, 1, 0.05),
            fake_run(2.0, 1, 0.01),
            fake_run(4.0, 1, 0.03),
        ];
        let groups = GroupBy { vars: &["d"] }.run(&runs);
        let metrics: &[(&str, MetricExtractor)] = &[(
            "msd",
            metric!(Msd, |out| out.cell0_values[0]),
        )];
        let summaries = MeanStderr { metrics }.run(&groups);
        let curve = Sweep { axis: "d" }.run(&summaries).unwrap();
        let xs: Vec<f64> = curve.points.iter().map(|p| p.x).collect();
        assert_eq!(xs, vec![2.0, 4.0, 8.0]);
    }
}
