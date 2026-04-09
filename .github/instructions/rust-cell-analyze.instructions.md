---
applyTo: "rust/vtk_viewer/src/analysis/**,rust/vtk_viewer/src/cell_analyze.rs"
---

# Rust cell_analyze CLI Tool — Copilot Instructions

> **When to consult this file:** You are running, modifying, or debugging the Rust-based `cell_analyze` binary. This covers the CLI interface, analysis observables, JSON output format, trajectory I/O with chain-overlap handling, key algorithms, Python wrapper scripts, and how to rebuild the binary. For the VTK *viewer* GUI, see [vtk-viewer.instructions.md](vtk-viewer.instructions.md). For simulation physics & parameters, see [cell-simulation.instructions.md](cell-simulation.instructions.md).

## ⚠️ MANDATORY: Keep This File Updated

**Whenever the Rust `cell_analyze` tool is modified** — new observables added, CLI flags changed, JSON output format altered, algorithms updated, or Python wrapper scripts changed — **this instruction file MUST be updated in the same session** to reflect those changes. This ensures future sessions have accurate context about the tool without needing to re-read source code.

## ⚠️ Related Instructions

| Task | Instruction File |
|------|-----------------|
| Cell simulation parameters/physics | [cell-simulation.instructions.md](cell-simulation.instructions.md) |
| VTK viewer (rendering, colormaps, glow) | [vtk-viewer.instructions.md](vtk-viewer.instructions.md) |
| Cluster job submission / MCP tools | [cluster-operations.instructions.md](cluster-operations.instructions.md) |
| Palmieri study design / FSS analysis | [palmieri-extension.instructions.md](palmieri-extension.instructions.md) |

---

## 1. What It Is

`cell_analyze` is a **Rust CLI binary** that reads simulation trajectory files, computes physics observables (MSD, diffusion, shape index, velocity distributions, etc.), and outputs structured JSON. It is the primary analysis engine for the Palmieri validation and finite-size scaling (FSS) studies.

## 2. Location & Building

### Source
```
rust/vtk_viewer/src/cell_analyze.rs       # CLI entry point (clap-driven)
rust/vtk_viewer/src/analysis/mod.rs       # Module root
rust/vtk_viewer/src/analysis/io.rs        # Trajectory I/O + chain-overlap handling
rust/vtk_viewer/src/analysis/observables.rs  # All observable computations
rust/vtk_viewer/src/analysis/output.rs    # JSON serialization types
rust/vtk_viewer/src/analysis/batch.rs     # Batch (parallel) mode over multiple runs
rust/vtk_viewer/src/analysis/study.rs     # TOML-driven study pipeline + SVG plotting
rust/vtk_viewer/src/analysis/checkpoint.rs # Checkpoint binary reader (v2–v5)
```

### Build
```bash
cd rust/vtk_viewer
cargo build --release --bin cell_analyze
```
Binary appears at: `rust/vtk_viewer/target/release/cell_analyze.exe` (Windows) or `cell_analyze` (Linux).

### Cross-compiling for Linux (cluster deployment)
Use the `sync_analysis` MCP tool — it builds in WSL and uploads via the existing SSH connection:
```
sync_analysis(cluster="nibi", confirm=true)
```
This builds `cell_analyze` for `x86_64-unknown-linux-musl` (static binary, no glibc dependency) and deploys to `~/cell_simulation/bin/cell_analyze` on the cluster.

**Manual build (if MCP unavailable):**
```bash
# In WSL:
cd /mnt/c/Users/stevensilber/source/repos/data_processing/rust/vtk_viewer
CARGO_TARGET_DIR=target_mcp cargo build --release --target x86_64-unknown-linux-musl --bin cell_analyze
# Upload manually via scp
```

**Note:** The `vtk_viewer` GUI binary requires `eframe` (system graphics) and cannot cross-compile for musl. Only `cell_analyze` is deployed to the cluster. The Cargo.toml has `required-features = ["gui"]` on vtk_viewer to prevent build failures.

### Dependencies (Cargo.toml)
- **rayon** — parallel batch processing
- **serde + serde_json** — JSON serialization
- **clap 4** (derive) — CLI argument parsing
- **anyhow** — error handling
- **toml** — study config parsing
- **regex** — pattern-based discovery
- **plotters** (svg_backend) — SVG plot generation
- **png** — PNG encoding for `snapshot` subcommand

---

## 3. CLI Interface

### Four Subcommands

#### `cell_analyze run <dir>` — Single simulation analysis
```bash
cell_analyze run /path/to/simulation -o results.json \
  --observables msd,diffusion,per_cell_diffusion,shape_index,velocity_distribution \
  --tau 10000 --cell_radius 49 --fit_frac 0.3
```
- `<dir>` must contain `trajectory.txt`
- `-o FILE` — output JSON file (default: stdout)
- `--observables CSV` — comma-separated list (default: all)
- `--tau FLOAT` — persistence time τ (default: 10000)
- `--cell_radius FLOAT` — cell radius R (default: 49)
- `--fit_frac FLOAT` — fraction of MSD curve for D_eff fit (default: 0.3)

#### `cell_analyze study <config.toml> -d <data_dir>` — TOML-driven analysis pipeline
```bash
cell_analyze study fss.toml -d /path/to/data -o results.json --plot-dir plots/
cell_analyze study fss.toml -d /path/to/data --dry-run          # preview discovery
cell_analyze study fss.toml -d /path/to/data --subsample 100    # keep every 100th frame
```
- The TOML config defines: discovery pattern, observables, grouping, pairing, metrics, plots
- This is the general-purpose pipeline: it replaces the old `batch`, `compare`, and `fss` subcommands
- For batch analysis: use a study TOML with no pairing/plots
- For two-run comparison: use a study TOML with `diagnostic` config and `pair_by`
- For FSS plots: use a study TOML with `group_by = ["N"]` and appropriate plot config
- `--subsample N` — keep every Nth frame (default: 1 = all). **Prefer no subsampling.**
- `--threads N` — limit parallelism (use `--threads 1` on login nodes)
- `--dry-run` — show discovered runs without analyzing
- Study configs live in `cpp/simulation/study/palmieri_extension/*.toml`

#### `cell_analyze snapshot <input> [-o output.png]` — Phase-field rendering & movies
```bash
# Single file (checkpoint or VTK frame):
cell_analyze snapshot checkpoint.bin -o snap.png --label-cells
cell_analyze snapshot frame_100000.vtk -o snap.png

# Directory mode (all VTK frames → PNG sequence):
cell_analyze snapshot /path/to/sim_output/ -o frames/ --skip 5

# Movie mode (render + pipe directly to ffmpeg — no intermediate PNGs):
cell_analyze snapshot /path/to/sim_output/ -o frames/ --movie --fps 15 --skip 5

# Per-cell rendering (watershed segmentation + colored contours):
cell_analyze snapshot /path/to/sim_output/ --movie --color-by gamma --shade-speed --fps 15
cell_analyze snapshot /path/to/sim_output/ --movie --color-by v_a --shade-speed   # Griffiths disorder
cell_analyze snapshot /path/to/sim_output/ --movie --color-by cell_id             # Track individual cells
```
- Accepts: checkpoint.bin, frame_NNNNNN.vtk, or directory containing VTK frames
- `--label-cells` — draw cell IDs at centroids. Green = soft cells (auto-detected from gamma). Works with all modes
- `--movie` — pipe raw RGB frames directly to ffmpeg (fast, no disk I/O). Falls back to PNG if ffmpeg unavailable
- `--skip N` — render every Nth frame in directory/movie mode (default: 1)
- `--fps N` — movie framerate (default: 15)
- `--color-by MODE` — color cell contours by property (requires checkpoint.bin in same directory):
  - `auto` (default) — detects from checkpoint: uses `v_a` if per-cell v_A varies, `gamma` if gamma varies, else `cell_id`
  - `v_a` — coolwarm colormap by per-cell v_A (blue=low, red=high). Best for Griffiths disorder
  - `gamma` — coolwarm by per-cell gamma (blue=soft, red=stiff). Best for Palmieri soft-cell
  - `cell_id` — unique HSV hue per cell (golden-ratio rotation). Best for tracking
  - `none` — plain phi heatmap (no watershed)
- `--shade-speed` — shade cell interiors by displacement speed (grayscale: brighter=faster). Requires trajectory.txt
- `--speed-window N` — frames to average for speed computation (default: 5)

**Per-cell rendering pipeline:** When `--color-by` is not `none`, the tool performs watershed segmentation from cell centroids, erodes the mask to create boundary rings, and colors contours/interiors independently. The speed normalization is fixed globally (P95 across the trajectory) to prevent frame-to-frame flashing.

**Output:** PNG at native domain resolution. Movies use libx264 CRF 18.

#### `cell_analyze list` — Show all available observables with descriptions

### Global flags
- `--subsample N` — available on `run` and `study` subcommands. Filters frames during I/O.

---

## 4. Input Format: `trajectory.txt`

### Header
Lines starting with `#` contain metadata key=value pairs:
```
# v_A=0.01 N=72 Lx=776.0 Ly=776.0 dt=0.01 ...
```
Keys used: `v_A`, `N`, `Lx`, `Ly`, `Lz`, `dim`, plus any extras stored in `params.extra`.

### Data Rows (2D)
```
time cell_id x y vx vy px py theta [v_A_inherent] [L_n]
```
- Column 0: time (simulation time units)
- Column 1: cell_id (0-indexed)
- Columns 2-3: x, y positions (unwrapped by loader — handles periodic jumps)
- Columns 4-5: vx, vy velocities
- Columns 6-7: px, py polarity
- Column 8: theta (angle)
- Column 9 (optional): per-cell inherent v_A
- Column 10 (optional): L_n normalized perimeter (for shape index)

### 3D Data Rows (14+ columns)
Same pattern with z-component for position, velocity, polarity.

---

## 5. Chain-Overlap Handling (io.rs)

When chain jobs produce overlapping trajectory segments, the loader detects backward time jumps in cell 0's data and classifies:

| Type | Condition | Action |
|------|-----------|--------|
| **Independent restart** | time jump > 10000 TU | Keep longest segment |
| **Continuation** | time jump < 1000 TU | Stitch segments, trim overlap |
| **Ambiguous** | 1000–10000 TU jump | Keep longest segment |
| **Multiple overlaps** | >2 segments | Keep longest single segment |

Position unwrapping (removing periodic boundary jumps) is also handled automatically by `io.rs`.

---

## 6. Output JSON Structure

### Single Run (`cell_analyze run`)
```json
{
  "path": "/path/to/run",
  "params": {
    "v_a": 0.01, "n_cells": 72,
    "lx": 776.0, "ly": 776.0,
    "extra": { "gamma_n": "1.0", ... }
  },
  "msd": { "lag_times": [...], "values": [...] },
  "diffusion": { "d_eff": 5.6e-7, "fit_r2": 0.99 },
  "per_cell_diffusion": {
    "cell_ids": [0,1,...], "d_values": [...],
    "d_mean": 5.3e-7, "d_std": 1.9e-7, "cv": 0.36
  },
  "shape_index": {
    "mean_p": 3.52, "std_p": 0.18,
    "per_cell_p": [...],
    "p_vs_time": [...], "cell0_p_vs_time": [...],
    "times": [...], "n_frames": 520
  },
  "velocity_distribution": {
    "bin_edges": [...], "cell0_hist": [...], "pop_hist": [...],
    "cell0_kurtosis": 1.2, "pop_kurtosis": 0.4,
    "cell0_mean_speed": 0.003, "cell0_sigma_vx": 0.0026,
    "pop_mean_speed": 0.003, "pop_sigma_vx": 0.0026
  }
}
```

### Batch (`cell_analyze batch`)
Groups results by pattern match level, computes mean/stderr per group:
```json
{
  "batch": true, "pattern": "...",
  "groups": {
    "group_name": {
      "n_runs": 5,
      "diffusion": { "mean": ..., "stderr": ..., "values": [...] },
      ...
    }
  },
  "summary": { "d_eff_vs_group": [...] }
}
```

---

## 7. Available Observables

| Observable | What It Computes |
|-----------|-----------------|
| `msd` | Mean-squared displacement ⟨Δr²(Δt)⟩, ensemble + time-averaged |
| `diffusion` | D_eff from linear fit to long-time MSD; D = slope/4 (2D) |
| `per_cell_diffusion` | Per-cell D_i from MSD/(4Δt) evaluated at lag = 8τ (Palmieri convention) |
| `shape_index` | p_eff = L_n × 2√π; vertex model shape parameter; cell0 time series |
| `velocity_distribution` | P(v_x) histogram, kurtosis, σ_v for cell 0 and population |
| `log_slope` | Δ(t) = d(ln MSD)/d(ln t); diffusion exponent |
| `cage` | Cage length l_c from MSD plateau |
| `alpha2` | Non-Gaussian parameter α₂ |
| `overlap` | Self-overlap Q(t), χ₄, stretched-exp fit: τ_α, β |
| `structure` | Static structure factor S(q), peak q* |
| `scattering` | Self-intermediate scattering F_s(q*, t) |
| `van_hove` | van Hove correlation G_s(Δx, t) at multiple lags |
| `displacement` | Total displacement stats (mean, RMS, max) |
| `stokes_einstein` | D × τ_α product (derived — requires both `diffusion` and `overlap`; not independently selectable) |
| `velocity_autocorrelation` | C_v(τ) and correlation time τ_c |
| `burst_detection` | Speed burst events where |v| > μ + 3σ |
| `va_mobility_correlation` | Correlation between inherent v_A and measured speed |
| `spatial_correlation` | Spatial autocorrelation of mobility, correlation length ξ |

### Standard Palmieri Analysis Set
```
msd,diffusion,per_cell_diffusion,shape_index,velocity_distribution
```
This is what `fss_quick.py` requests by default.

---

## 8. Key Algorithms

### D_eff Extraction (per_cell_diffusion)
- **Method:** MSD/(4Δt) evaluated at lag = 8τ (Palmieri convention), NOT averaged over a range
- For each cell, compute MSD(Δt) using time-averaged sliding window, then evaluate MSD/(4Δt) at exactly Δt = 8τ
- This gives D_i for that cell at the Palmieri plateau
- CV (coefficient of variation) = std/mean; typical good run has CV ≈ 0.28–0.37

### Shape Index
- p_eff = L_n × 2√π ≈ L_n × 3.5449
- Vertex model transition at p ≈ 3.81 (solid-like below, fluid-like above)
- `cell0_p_vs_time` tracks cell 0's individual shape trajectory (for Palmieri Fig 3A)

### Velocity Distribution
- Computes v_x = Δx/Δt for cell 0 and all cells
- Histogram with 100 bins
- Reports excess kurtosis (K > 0 = non-Gaussian tails = bursty motion)
- Python wrapper `fss_quick.py` transforms to G(v_i) = -√|log[P(v)/P(0)]| for Palmieri Fig 4

### Palmieri Eq. 5 (Two-Regime Velocity Fit)
For soft cells, `fss_quick.py` fits Palmieri's Eq. 5:
```
P_soft(v) = (1-ζ) P_G(v) + ζ ∫ P_iso(v') P_G(v-v') dv'
```
where:
- P_G(v) = Gaussian with σ_G (from normal/ctrl cell data)
- P_iso(v') = 1/(π √(v_A² - v'²)) for |v'| < v_A (arcsine/self-propulsion distribution)
- ζ = single fitting parameter = fraction of time in "bursty" regime
- The integral is a convolution computed via FFT
- Palmieri found ζ ≈ 0.038 (3-4% bursty) for 72-cell, ρ=0.90

### ⚠️ CRITICAL: Velocity Definition for Palmieri Analysis
**Use DISPLACEMENT-BASED velocity (Δx/Δt), NOT the EoM velocity stored in trajectory columns 4-5.**

The trajectory file stores two different velocity quantities:
- **Columns 4-5 (vx, vy):** The equation-of-motion velocity v = v_I + v_A from `kernel_compute_velocities`. This is the phase-field advection velocity. Its distribution has σ ≈ v_A/√2 ≈ 0.007 and is **sub-Gaussian** (kurtosis < 0). This is NOT what Palmieri plots.
- **Displacement velocity (Δx/Δt):** Computed from centroid position differences between frames. Its distribution has σ ≈ 0.003 (matching Palmieri's σ_G = 0.0029) and shows **non-Gaussian tails** (positive kurtosis) with burst events reaching 1.5-1.9× v_A. This IS what Palmieri calls "instantaneous velocity."

The reason these differ: The EoM velocity drives the advection ∂φ/∂t + v·∇φ = -Γ δF/δφ, but the centroid displacement is the NET result of advection minus shape relaxation. The centroid (∫x·φ²dA / ∫φ²dA) moves less than v because the restoring force partially counteracts advection. During bursts, the elastic relaxation ADDS to the displacement, producing velocities exceeding v_A.

`fss_quick.py` correctly computes displacement velocity from trajectory positions (columns 2-3). The Rust `velocity_distribution` observable also uses displacement-based velocity (Δx/Δt from unwrapped positions).

---

## 10. Modifying Observables

### Adding a New Observable
1. In `observables.rs`, add a new function `compute_your_observable(data, params) -> YourResult`
2. Create a result struct with `#[derive(Serialize)]`
3. In `cell_analyze.rs` or `mod.rs`, add the observable name to the dispatch logic
4. Add to the `list` subcommand help text
5. Rebuild: `cargo build --release --bin cell_analyze`

### Common Gotchas
- Trajectory positions are **unwrapped** by `io.rs`; don't re-unwrap
- Frame indexing: `data[frame_idx][cell_idx]` gives `(x, y, vx, vy, ...)`
- `v_A` is read from trajectory header, not hardcoded
- For per-cell analysis, cell 0 has special meaning (the "tagged" cell in Palmieri)

---

## 11. Typical Workflow

### Cluster analysis (preferred — no download needed)
```
1. Sync binary: sync_analysis(confirm=true)
2. Upload TOML: sync_study_config(local_path="cpp/.../fss.toml", remote_path="~/fss.toml")
3. Submit SLURM analysis job:
     run_analysis(data_dir="/scratch/.../fss_reps", study_config="~/fss.toml")
     → Runs full-resolution (no subsampling) on a CPU compute node
4. Download JSON: download_results(remote_path="/scratch/.../analysis_results.json")

Do NOT subsample when using run_analysis — the SLURM job has enough
time and memory to process full-resolution trajectories directly.
```

### Local analysis (when data is already downloaded)
```
1. Download subsampled trajectories to cluster_results/fss_reps_sub/
2. Restructure: each run needs a trajectory.txt inside its directory
     e.g., 100c_rho90_soft/run_01/trajectory.txt
3. Run: cell_analyze study fss.toml -d cluster_results/fss_reps_sub -o results.json
4. Results appear in JSON + SVG plots
```

### D_eff extraction method
The `per_cell_diffusion` observable extracts D_eff at lag = 8τ (the Palmieri plateau),
NOT averaged over a wide lag range. This was validated against Palmieri Fig 5.
MSD is computed using the time-averaged sliding-window method: for each lag Δt,
average |r(t₀+Δt) - r(t₀)|² over all starting times t₀.

---

## 12. TOML Study Config Schema

Study configs live in `cpp/simulation/study/palmieri_extension/*.toml`.

### Complete TOML structure

```toml
[study]
name = "Study Name"            # Display name
description = "Description"    # Optional description

[discovery]
pattern = "{N}c_rho{rho}_{cond}/run_{seed}"   # Path pattern with {var} captures
trajectory_name = "trajectory.txt"             # Default: "trajectory.txt"

[observables]
compute = ["per_cell_diffusion", "shape_index", "velocity_distribution"]
tau = 10000.0               # Persistence time τ (default: 10000)
cell_radius = 49.0          # Cell radius R (default: 49)
fit_frac = 0.3              # Fraction of MSD for D_eff fit (default: 0.3)

[analysis]
tagged_cell = 0             # Tagged cell index (default: 0)
group_by = ["N", "rho", "cond"]   # Variables to group runs by
pair_by = "cond"            # Variable to pair conditions on
pair_numerator = "soft"     # Value of pair_by for numerator
pair_denominator = "ctrl"   # Value of pair_by for denominator

[analysis.metrics]          # Map of metric_name → expression string
d_eff = "tagged_cell_d_eff"
d_eff_pop = "population_d_eff"
d_eff_normal = "normal_d_eff"
ln = "tagged_cell_ln"
ln_pop = "population_ln"
kurtosis = "tagged_cell_kurtosis"
mean_speed = "tagged_cell_mean_speed"

# ── Single-panel plots ──
[[plots]]
title = "D_eff ratio vs 1/sqrt(N)"
x = "N"                     # Variable for x-axis
y = "d_eff_ratio"           # Metric for y-axis (auto-appended _ratio for paired)
output = "fss_deff_ratio.svg"
x_label = "1/sqrt(N)"
y_label = "D_eff(soft) / D_eff(ctrl)"
x_transform = "inverse_sqrt"  # Options: "inverse_sqrt", "log"
x_log = false               # Optional: log scale on x-axis
error_bars = true
y_min = 0.5                 # Optional axis limits
y_max = 2.0

# ── Multi-panel figures ──
[[figures]]
title = "FSS Overview"
output = "fss_overview.svg"
layout = [2, 2]             # [rows, cols]
width = 900                 # Default: 900
height = 700                # Default: 700

[[figures.panels]]
x = "N"
y = "d_eff_ratio"
title = "D_eff ratio"
x_label = "N"
y_label = "ratio"
x_transform = "inverse_sqrt"
error_bars = true
h_line = 1.0                # Optional horizontal reference line
y_min = 0.5
y_max = 2.0

# ── Per-seed diagnostic comparison ──
[diagnostic]
seeds = [1, 2, 3]           # Seeds to generate diagnostics for
output = "diag_seed{seed}.svg"  # {seed} placeholder
ln_range = [0.9, 1.3]       # Optional axis ranges
speed_max = 0.02
msd_lag_max = 100000
```

### Valid metric expression strings
- `tagged_cell_d_eff`, `population_d_eff`, `normal_d_eff`
- `tagged_cell_ln`, `population_ln`
- `tagged_cell_kurtosis`, `population_kurtosis`
- `tagged_cell_mean_speed`, `population_mean_speed`
- `d_eff_cv`, `diffusion_r2`
- `stokes_einstein`, `tau_alpha`

### Study JSON output structure
```json
{
  "study_name": "...", "description": "...",
  "n_runs_total": 20, "n_groups": 4,
  "warnings": ["..."],
  "groups": [
    {
      "group_key": "N=100,rho=90,cond=soft",
      "variables": {"N": "100", "rho": "90", "cond": "soft"},
      "n_seeds": 5,
      "metrics": {
        "d_eff": {"mean": 5.6e-7, "stderr": 1.2e-7, "values": [...]}
      }
    }
  ],
  "paired": [
    {
      "group_key": "N=100,rho=90",
      "variables": {"N": "100", "rho": "90"},
      "numerator": { "metrics": {...} },
      "denominator": { "metrics": {...} },
      "paired_metrics": {
        "d_eff_ratio": {"mean": 1.34, "stderr": 0.13, "values": [...]},
        "d_eff_diff": {"mean": 1.2e-7, "stderr": 5e-8, "values": [...]}
      }
    }
  ]
}
```

---

## 13. Checkpoint Binary Format

The `snapshot` subcommand and `checkpoint.rs` module read checkpoint.bin files.

### Header (40 bytes, v4+)
| Offset | Type | Field |
|--------|------|-------|
| 0 | u32 | Magic: `0x43454C4C` ("CELL") |
| 4 | u32 | Version (2–5 supported) |
| 8 | i32 | Step number |
| 12 | f32 | Simulation time |
| 16 | i32 | Number of cells |
| 20 | i32 | Save interval |
| 24 | i32 | Checkpoint interval |
| 28 | i32 | Trajectory samples |
| 32 | 4×bool | save_vtk, save_tracking, compute_diagnostics, save_individual_fields |
| 36 | u32 | sim_params_size |

### SimParams block
Nx, Ny, dx, dy, dt, t_end, save_interval, lambda, gamma, kappa, target_radius, mu, v_A, xi, tau, halo_width, min_subdomain_size, subdomain_padding (all f32/i32).

### Per-cell data (repeated num_cells times)
| Field | Type | Notes |
|-------|------|-------|
| cell_id | i32 | |
| bbox | 4×i32 | x, y, width, height |
| centroid | 2×f32 | x, y |
| velocity | 2×f32 | vx, vy |
| volume | f32 | |
| phi field | f32[w×h] | Full phi data within bbox (with halo) |

### Optional tagged arrays (after all cells)
| Magic | Tag | Content |
|-------|-----|---------|
| `0x56415F41` | "VA_A" | Per-cell v_A array (f32[N]) |
| `0x47414D41` | "GAMA" | Per-cell gamma array (f32[N]) |
| `0x52414449` | "RADI" | Per-cell radius array (f32[N]) |

### Version differences
- **v2–v3:** Older bbox format, no halo field
- **v4:** Inner bbox + halo; sim_params_size auto-detection heuristic
- **v5:** bbox_with_halo field replaces inner bbox

---

## 14. Observable JSON Output Reference

### Full RunResult fields

| Observable | Key JSON fields |
|-----------|----------------|
| `msd` | `lag_times[]`, `values[]` |
| `diffusion` | `d_eff`, `fit_r2` |
| `per_cell_diffusion` | `cell_ids[]`, `d_values[]`, `d_mean`, `d_std`, `cv` |
| `shape_index` | `mean_p`, `std_p`, `per_cell_p[]`, `p_vs_time[]`, `cell0_p_vs_time[]`, `times[]` |
| `velocity_distribution` | `bin_edges[]`, `cell0_hist[]`, `pop_hist[]`, `cell0_kurtosis`, `pop_kurtosis`, `cell0_mean_speed`, `cell0_sigma_vx` |
| `log_slope` | `times[]`, `delta[]` |
| `cage` | `l_c`, `t_star` |
| `alpha2` | `lag_times[]`, `values[]` |
| `overlap` | `lag_times[]`, `q_mean[]`, `chi4[]`, `tau_alpha`, `beta`, `fit_r2` |
| `structure` | `q_bins[]`, `s_q[]`, `q_star` |
| `scattering` | `lag_times[]`, `fs[]`, `tau_alpha`, `beta`, `fit_r2` |
| `van_hove` | `dx_bins[]`, `distributions[{lag_time, histogram[]}]` |
| `displacement` | `mean_dr`, `rms_dr`, `max_dr`, `mean_dr_over_r` |
| `burst_detection` | `total_bursts`, `mean_bursts_per_cell`, `mean_duration`, `mean_peak_speed`, `events[]`, `threshold`, `speed_mean`, `speed_std` |
| `velocity_autocorrelation` | `lag_times[]`, `cv[]`, `beta` (stub=1.0), `tau_c` |
| `va_mobility_correlation` | `pearson_r`, `n_cells`, `cell_speeds[]`, `cell_va[]` |
| `spatial_correlation` | `r_bins[]`, `c_r[]`, `xi` |

---

## 13. Diagnostic Panel Configuration

The `[diagnostic]` section in study TOMLs generates per-seed SVG comparison figures (soft vs ctrl). Panels are fully configurable:

### Adding panels
```toml
[diagnostic]
output = "diag_rho{rho}_{seed}.svg"   # All {var} placeholders are substituted
ln_range = [0.98, 1.5]                # Default y-range for L_n panels
speed_max = 0.02                      # Default y-max for speed panel
msd_lag_max = 8.0                     # Default x-max for MSD panel (in τ)

[[diagnostic.panels]]
type = "trajectory"          # Cell 0 wrapped path (x,y)

[[diagnostic.panels]]
type = "msd_t"               # MSD/Δt curves (→ 4D_eff plateau)
log_x = true                 # Log-transform axes
log_y = true
x_range = [0.01, 10.0]       # Override axis range [min, max]

[[diagnostic.panels]]
type = "ln_timeseries"       # L_n(t) time series
y_range = [0.95, 1.4]        # Override y-axis (takes priority over ln_range)

[[diagnostic.panels]]
type = "ln_histogram"        # L_n distribution (overlaid bars)
bins = 60                    # Number of histogram bins
x_range = [0.98, 1.3]        # Override x-axis (takes priority over ln_range)

[[diagnostic.panels]]
type = "speed_bursts"        # Cell 0 speed |v|(t) with burst threshold

[[diagnostic.panels]]
type = "gvi"                 # Palmieri G(v_i) with Eq.5 fit
x_range = [0.0, 0.025]       # Override velocity range

[[diagnostic.panels]]
type = "deff_bar"            # 4-bar D_eff comparison

[[diagnostic.panels]]
type = "summary"             # Text summary of parameters + observables
```

### Available panel types
| Type | Renders | Key options |
|------|---------|-------------|
| `trajectory` | Cell 0 wrapped path with start/end markers | — |
| `msd_t` | MSD/Δt vs lag (→ D_eff plateau) | `log_x`, `log_y`, `x_range`, `y_range` |
| `ln_timeseries` | Cell 0 L_n(t) with mean overlay | `y_range` (or `ln_range` from diagnostic) |
| `ln_histogram` | L_n distribution as density bars | `bins`, `x_range` (or `ln_range`) |
| `speed_bursts` | Cell 0 displacement speed with μ+3σ threshold | `x_range`, `y_range` (or `speed_max`) |
| `gvi` | Palmieri G(v_i) + Gaussian ref + Eq.5 ζ fit | `x_range`, `y_range` |
| `deff_bar` | 4-bar chart: soft_pop, soft_c0, ctrl_pop, ctrl_c0 | — |
| `summary` | Text: N, Lx, v_A, dt, γ, time ranges, D_eff ratios | — |

### Panel layout
- Panels render left-to-right, top-to-bottom in a dynamic grid
- ≤2 panels → 1 row; ≤4 → 1 or 2 rows; >4 → 4 columns, as many rows as needed
- Panel order follows the TOML `[[diagnostic.panels]]` array order
- If no panels are specified, defaults to all 8 panel types

### Template variables in output filename
The `output` field substitutes ALL discovered variables:
```toml
output = "diag_{N}c_rho{rho}_{seed}.svg"
# Produces: diag_100c_rho90_2.svg, diag_100c_rho85_3.svg, etc.
```

---

## 14. Case Study: Palmieri Validation (100-cell)

**Goal:** Reproduce Palmieri et al. (2015) Fig 2–5 for 100 cells at ρ=0.85 and ρ=0.90.

### Data layout on cluster
```
/scratch/ssilber/palmieri_ext/
├── prod_100c_rho85_soft_v3/   # --gamma 0.35:cell0
│   ├── trajectory.txt
│   └── checkpoint.bin
├── prod_100c_rho85_ctrl_v3/   # uniform gamma=1.0
│   ├── trajectory.txt
│   └── checkpoint.bin
├── prod_100c_rho90_soft_v3/
└── prod_100c_rho90_ctrl_v3/
```

### Study TOML
```toml
[study]
name = "palmieri_ext_100c"
description = "Palmieri extension: soft cell 0 vs control"

[discovery]
pattern = "prod_100c_rho{rho}_{cond}_v{seed}"

[observables]
compute = ["msd", "diffusion"]
tau = 10000.0
cell_radius = 49.0

[analysis]
pair_by = "cond"
pair_numerator = "soft"
pair_denominator = "ctrl"
group_by = ["rho"]

[diagnostic]
output = "diag_100c_rho{rho}_{seed}.svg"

[[diagnostic.panels]]
type = "trajectory"

[[diagnostic.panels]]
type = "msd_t"
log_x = true
log_y = true

[[diagnostic.panels]]
type = "deff_bar"

[[diagnostic.panels]]
type = "summary"
```

### Run
```bash
cell_analyze study ~/study/palmieri_val.toml \
  -d /scratch/ssilber/palmieri_ext \
  --plot-dir /scratch/ssilber/palmieri_ext/plots \
  -o /scratch/ssilber/palmieri_ext/results.json
```

### What it produces
- `results.json` — grouped metrics per (rho) with paired soft/ctrl comparisons
- `diag_100c_rho85_2.svg`, `diag_100c_rho90_3.svg`, etc. — per-seed diagnostic panels
- Console output: run counts, warnings (low seed count), D_eff ratios

---

## 15. Case Study: Finite-Size Scaling (FSS)

**Goal:** Test whether D_eff(soft)/D_eff(ctrl) converges as N→∞.

### Data layout
```
/scratch/ssilber/palmieri_ext/
├── 100c_rho90_soft/run_01/trajectory.txt
├── 100c_rho90_soft/run_02/trajectory.txt
├── 100c_rho90_ctrl/run_01/trajectory.txt
├── 200c_rho90_soft/run_01/trajectory.txt
├── ...
└── 6400c_rho90_ctrl/run_03/trajectory.txt
```

### Study TOML (fss.toml)
```toml
[study]
name = "Palmieri FSS"
description = "D_eff ratio vs system size N"

[discovery]
pattern = "{N}c_rho{rho}_{cond}/run_{seed}"

[observables]
compute = ["per_cell_diffusion", "shape_index", "velocity_distribution"]
tau = 10000.0
cell_radius = 49.0

[analysis]
tagged_cell = 0
group_by = ["N", "rho", "cond"]
pair_by = "cond"
pair_numerator = "soft"
pair_denominator = "ctrl"

[analysis.metrics]
d_eff = "tagged_cell_d_eff"
ln = "tagged_cell_ln"
kurtosis = "tagged_cell_kurtosis"

[[plots]]
title = "D_eff ratio vs 1/sqrt(N)"
x = "N"
y = "d_eff_ratio"
output = "fss_deff_ratio.svg"
x_transform = "inverse_sqrt"
h_line = 1.0

[[figures]]
title = "FSS Overview"
output = "fss_overview.svg"
layout = [2, 2]

[[figures.panels]]
x = "N"
y = "d_eff"
title = "D_eff (cell 0)"
x_transform = "inverse_sqrt"

[[figures.panels]]
x = "N"
y = "d_eff_pop"
title = "D_eff (population)"
x_transform = "inverse_sqrt"

[[figures.panels]]
x = "N"
y = "ln"
title = "Mean L_n"
x_transform = "inverse_sqrt"
h_line = 1.0

[[figures.panels]]
x = "N"
y = "d_eff_ratio"
title = "D_eff ratio"
x_transform = "inverse_sqrt"
h_line = 1.0
```

### Run
```bash
cell_analyze study ~/study/fss.toml \
  -d /scratch/ssilber/palmieri_ext \
  --plot-dir /scratch/ssilber/palmieri_ext/fss_plots \
  -o /scratch/ssilber/palmieri_ext/fss_results.json \
  --threads 4
```

### Expected output
- `fss_deff_ratio.svg` — single plot: D_eff ratio vs 1/√N with error bars
- `fss_overview.svg` — 4-panel figure: D_eff(cell0), D_eff(pop), L_n, ratio
- `fss_results.json` — full numerical results for all groups
- If N→∞ extrapolation shows ratio→1, the Palmieri result is a finite-size artifact

---

## 16. Snapshot Rendering for Publications

### Generating a labeled snapshot from a checkpoint
```bash
# Basic phi heatmap with cell IDs (green = soft cells)
cell_analyze snapshot checkpoint.bin -o figure_1a.png --label-cells

# Per-cell rendering with gamma-colored contours (for Palmieri soft-cell papers)
cell_analyze snapshot /path/to/vtk_dir/ --color-by gamma --label-cells --skip 1000 -o fig1/
```

### Generating a movie from VTK frames
```bash
# Standard phi movie
cell_analyze snapshot /path/to/vtk_dir/ --movie --skip 5 --fps 15 -o movie_dir/

# Griffiths disorder visualization (v_A contours + speed-shaded interiors)
cell_analyze snapshot /path/to/vtk_dir/ --movie --color-by v_a --shade-speed --fps 15 -o movie_dir/

# Track individual cells
cell_analyze snapshot /path/to/vtk_dir/ --movie --color-by cell_id --shade-speed --fps 10 -o movie_dir/
```

### What `--color-by auto` does
1. Loads `checkpoint.bin` from the VTK directory
2. Reads per-cell v_A array — if values vary (range > 1% of max), uses coolwarm by v_A
3. Otherwise reads per-cell gamma — if values vary, uses coolwarm by gamma
4. Otherwise falls back to HSV by cell_id
