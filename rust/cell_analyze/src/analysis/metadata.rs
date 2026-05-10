//! Checkpoint provenance metadata for plot overlays.
//!
//! Reads `.sim_marker.json` (written by compute-canada-mcp) from the
//! checkpoint's parent directory, and computes confluence independently
//! from the checkpoint's own SimParams + cell count.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimMarker {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub study: Option<MarkerStudy>,
    #[serde(default)]
    pub provenance: Option<MarkerProvenance>,
    #[serde(default)]
    pub submission_params: Option<serde_json::Value>,
    #[serde(default)]
    pub status: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarkerStudy {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarkerProvenance {
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub job_ids: Vec<String>,
    #[serde(default)]
    pub seed: Option<serde_json::Value>,
    #[serde(default)]
    pub source_checkpoint: Option<String>,
}

/// Try to load `.sim_marker.json` from the directory containing `checkpoint_path`.
/// Returns None silently if missing or malformed.
pub fn load_marker_for(checkpoint_path: &Path) -> Option<SimMarker> {
    let dir = checkpoint_path.parent()?;
    let marker_path = dir.join(".sim_marker.json");
    let bytes = std::fs::read(&marker_path).ok()?;
    serde_json::from_slice::<SimMarker>(&bytes).ok()
}

/// Extract a scalar field from submission_params as f64.
pub fn marker_param_f64(marker: &SimMarker, key: &str) -> Option<f64> {
    marker
        .submission_params
        .as_ref()?
        .get(key)?
        .as_f64()
}

/// Extract a string field from submission_params.
pub fn marker_param_str(marker: &SimMarker, key: &str) -> Option<String> {
    marker
        .submission_params
        .as_ref()?
        .get(key)?
        .as_str()
        .map(|s| s.to_string())
}

/// Compute confluence φ = N · π · R² / (Lx · Ly) from checkpoint params + cell count.
/// `lx`, `ly` are domain extents in physical units (Nx*dx, Ny*dy).
pub fn compute_confluence(num_cells: i32, radius: f32, lx: f32, ly: f32) -> f64 {
    let area_cells = (num_cells as f64) * std::f64::consts::PI * (radius as f64).powi(2);
    let area_box = (lx as f64) * (ly as f64);
    if area_box > 0.0 {
        area_cells / area_box
    } else {
        0.0
    }
}
