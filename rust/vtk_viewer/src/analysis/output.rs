//! Structured JSON output for analysis results.

use serde::Serialize;
use std::collections::BTreeMap;

use super::observables::*;

/// Full analysis result for a single run.
#[derive(Serialize, Clone, Debug)]
pub struct RunResult {
    pub path: String,
    pub params: RunParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub msd: Option<MsdResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffusion: Option<DiffusionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_slope: Option<LogSlopeResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cage: Option<CageLengthResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha2: Option<Alpha2Result>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlap: Option<OverlapResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structure: Option<StructureFactorResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scattering: Option<ScatteringResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub van_hove: Option<VanHoveResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_cell_diffusion: Option<PerCellDiffusionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub displacement: Option<DisplacementResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stokes_einstein: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub va_mobility_correlation: Option<VaMobilityCorrelationResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spatial_correlation: Option<SpatialCorrelationResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape_index: Option<ShapeIndexResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity_autocorrelation: Option<VelocityAutocorrelationResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub burst_detection: Option<BurstDetectionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity_distribution: Option<VelocityDistributionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polarity_tau: Option<PolarityTauResult>,
}

/// Parameters extracted from the trajectory header.
#[derive(Serialize, Clone, Debug)]
pub struct RunParams {
    pub v_a: f64,
    pub n_cells: usize,
    pub lx: f64,
    pub ly: f64,
    /// Packing fraction ρ = N·π·R² / (Lx·Ly)
    pub confluence: f64,
    /// Subdomain padding parameter (fraction of R)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subdomain_padding: Option<f64>,
    /// Mean bounding box width across all cells (pixels)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox_mean: Option<f64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, String>,
}

/// Aggregated result for a group of runs (e.g. all replicates at one Jk value).
#[derive(Serialize, Clone, Debug)]
pub struct GroupResult {
    pub n_runs: usize,
    pub params: RunParams,
    pub diffusion: Option<AggregatedScalar>,
    pub overlap: Option<AggregatedOverlap>,
    pub displacement: Option<AggregatedScalar>,
}

#[derive(Serialize, Clone, Debug)]
pub struct AggregatedScalar {
    pub mean: f64,
    pub stderr: f64,
    pub values: Vec<f64>,
}

#[derive(Serialize, Clone, Debug)]
pub struct AggregatedOverlap {
    pub tau_alpha_mean: f64,
    pub tau_alpha_stderr: f64,
    pub beta_mean: f64,
}

/// Batch output: groups + optional per-run details.
#[derive(Serialize, Clone, Debug)]
pub struct BatchResult {
    pub batch: bool,
    pub pattern: String,
    pub groups: BTreeMap<String, GroupResult>,
    /// Summary arrays for quick plotting: (group_key, value) pairs.
    pub summary: BatchSummary,
}

#[derive(Serialize, Clone, Debug)]
pub struct BatchSummary {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub d_eff_vs_group: Vec<(String, f64, f64)>, // (key, mean, stderr)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tau_alpha_vs_group: Vec<(String, f64, f64)>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub displacement_vs_group: Vec<(String, f64, f64)>,
}

/// Aggregate per-run results into a GroupResult.
pub fn aggregate_group(runs: &[&RunResult]) -> GroupResult {
    let n_runs = runs.len();
    let params = runs[0].params.clone();

    let diffusion = {
        let vals: Vec<f64> = runs
            .iter()
            .filter_map(|r| r.diffusion.as_ref().map(|d| d.d_eff))
            .filter(|d| d.is_finite())
            .collect();
        if vals.is_empty() {
            None
        } else {
            Some(aggregate_scalar(&vals))
        }
    };

    let overlap = {
        let taus: Vec<f64> = runs
            .iter()
            .filter_map(|r| r.overlap.as_ref().map(|o| o.tau_alpha))
            .filter(|t| t.is_finite())
            .collect();
        let betas: Vec<f64> = runs
            .iter()
            .filter_map(|r| r.overlap.as_ref().map(|o| o.beta))
            .filter(|b| b.is_finite())
            .collect();
        if taus.is_empty() {
            None
        } else {
            let tau_mean = taus.iter().sum::<f64>() / taus.len() as f64;
            let tau_stderr = if taus.len() > 1 {
                let var: f64 =
                    taus.iter().map(|t| (t - tau_mean).powi(2)).sum::<f64>() / taus.len() as f64;
                var.sqrt() / (taus.len() as f64).sqrt()
            } else {
                0.0
            };
            let beta_mean = if betas.is_empty() {
                f64::NAN
            } else {
                betas.iter().sum::<f64>() / betas.len() as f64
            };
            Some(AggregatedOverlap {
                tau_alpha_mean: tau_mean,
                tau_alpha_stderr: tau_stderr,
                beta_mean,
            })
        }
    };

    let displacement = {
        let vals: Vec<f64> = runs
            .iter()
            .filter_map(|r| r.displacement.as_ref().map(|d| d.mean_dr))
            .filter(|d| d.is_finite())
            .collect();
        if vals.is_empty() {
            None
        } else {
            Some(aggregate_scalar(&vals))
        }
    };

    GroupResult {
        n_runs,
        params,
        diffusion,
        overlap,
        displacement,
    }
}

fn aggregate_scalar(vals: &[f64]) -> AggregatedScalar {
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let stderr = if vals.len() > 1 {
        let var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        var.sqrt() / n.sqrt()
    } else {
        0.0
    };
    AggregatedScalar {
        mean,
        stderr,
        values: vals.to_vec(),
    }
}
