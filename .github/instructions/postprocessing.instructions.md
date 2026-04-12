---
applyTo: "cpp/simulation/postprocessing/**"
---

# Cell Simulation Post-Processing - Copilot Instructions

> **When to consult this file:** You are visualizing, analyzing, or plotting simulation output (VTK frames, trajectories, observables). This covers Python scripts in `postprocessing/` for visualization and one-off plotting. For production analysis (MSD, diffusion, structure factor, etc.), use the Rust `cell_analyze` tool — see [rust-cell-analyze.instructions.md](rust-cell-analyze.instructions.md). For running simulations, see [cell-simulation.instructions.md](cell-simulation.instructions.md).

> **⚠️ Primary analysis tool:** `cell_analyze` (Rust) handles all standard observables. The Python and C scripts in this directory are legacy — use them only for custom visualization or plotting. `read_checkpoint.py` has been removed; use `cell_analyze run <dir>` or `cell_analyze study` for checkpoint/trajectory analysis.

## ⚠️ Related Instructions - Read Before Proceeding

| Task | Instruction File |
|------|-----------------|
| **Building & running simulations** | [cell-simulation.instructions.md](cell-simulation.instructions.md) |
| **Running simulations on cluster** | [cluster-operations.instructions.md](cluster-operations.instructions.md) |
| **Production runs (use `start_simulation` MCP tool)** | [cluster-operations.instructions.md](cluster-operations.instructions.md) - see "Jamming Study Production" |
| **Developing analysis tools for cluster** | [cluster-postprocessing.instructions.md](cluster-postprocessing.instructions.md) |

**Quick-reference READMEs:**
- [cluster/README.md](cpp/simulation/cluster/README.md) - Cluster submission quick start
- [postprocessing/README.md](cpp/simulation/postprocessing/README.md) - Analysis scripts (this folder)

**Important:** For large-scale batch processing on cluster data, see [cluster-postprocessing.instructions.md](cluster-postprocessing.instructions.md) for the proper workflow (develop locally → submit SLURM jobs).

## Directory Structure

```
cpp/simulation/
├── postprocessing/           # ← Main analysis scripts (THIS FOLDER)
│   ├── output/               # ← ALL GENERATED PLOTS/IMAGES GO HERE
│   │   ├── energy_comparison_YYYYMMDD.png
│   │   ├── jamming_transition_phi85_YYYYMMDD.png
│   │   └── msd_analysis_v0.008_YYYYMMDD.png
│   ├── visualize.py          # 2D field visualization & movies
│   ├── visualize_3d.py       # 3D volume rendering
│   ├── visualize_stress.py   # Stress tensor visualization
│   ├── visualize_combined.py # Synchronized movie + observables
│   ├── analyze_trajectory.py # MSD, autocorrelations, diffusion
│   ├── plot_observables.py   # Energy/stress/coordination plots
│   ├── plot_energy.py        # Equilibration energy vs time
│   ├── plot_jamming_transition.py  # D(v_A) jamming curve
│   └── stack_videos.py       # Video grid layout
│
├── cluster/                  # Cluster-specific tools
│   ├── msd_calculator.c      # Fast C MSD calculator
│   ├── energy_analyzer.c     # Equilibration energy analyzer
│   ├── msd_analysis.py       # MSD from tracking.txt format
│   ├── plot_msd.py           # Quick MSD visualization
│   └── debug/                # Debug/validation scripts (not for regular use)
│       ├── debug_msd.py
│       ├── validate_msd_python.py
│       ├── analyze_msd_detailed.py
│       ├── analyze_trajectories.py
│       ├── check_crossover.py
│       └── plot_D_vs_vA.py
│
└── agent_test_runs/          # Local test output (gitignored)
```

---

## ⚠️ Output File Conventions

**ALL generated plots, images, and analysis outputs MUST be saved to `postprocessing/output/`.**

This maintains a track record of all generated results and keeps the codebase organized.

### Output Directory
```
cpp/simulation/postprocessing/output/
```

Create this directory if it doesn't exist. It should be gitignored (add to `.gitignore` if not present).

### Naming Convention

Use descriptive names with dates for traceability:

```
<analysis_type>_<parameters>_YYYYMMDD.png
```

**Examples:**
| Analysis | Filename |
|----------|----------|
| Energy comparison (phi85 vs phi89) | `energy_comparison_phi85_phi89_20260110.png` |
| Jamming transition at 85% packing | `jamming_transition_phi85_20260110.png` |
| MSD analysis for v_A=0.008 | `msd_analysis_vA0.008_20260110.png` |
| Equilibration energy at 89% | `energy_equilibration_phi89_20260110.png` |
| Stress visualization | `stress_combined_t80000_20260110.png` |

### Usage in Scripts

When running postprocessing scripts, always specify the output path:

```powershell
# Good - saves to output folder with descriptive name
python plot_energy.py ..\cluster\energy_phi89.txt -o output\energy_phi89_20260110.png

# Good - comparison plot
python plot_energy.py ..\cluster\energy_phi85.txt ..\cluster\energy_phi89.txt -o output\energy_comparison_20260110.png

# Good - jamming transition
python plot_jamming_transition.py results.txt -o output\jamming_transition_phi85_20260110.png
```

### Agent Instructions

When generating plots or analysis outputs:
1. **Always save to `postprocessing/output/`** - never to random locations
2. **Use descriptive filenames** with relevant parameters (phi, v_A, etc.)
3. **Include the date** (YYYYMMDD format) for version tracking
4. **Don't overwrite previous results** - use unique names or dates
5. **Report the full path** to the user after saving

## Overview

Post-processing scripts are located in `cpp/simulation/postprocessing/`. They analyze trajectory data, VTK field output, and diagnostic observables to produce plots and movies for paper figures.

## Output Files from Simulation

### 2D Simulation Output
| File | Format | Contents | When Generated |
|------|--------|----------|----------------|
| `frame_*.vtk` | VTK structured points | 2D phase field φ(x,y) | Every `save_interval` steps |
| `trajectory.txt` | Text | Cell positions, velocities, polarization | Every `trajectory_interval` steps |
| `checkpoint.bin` | Binary | Full 2D simulation state for restart | Every `checkpoint_interval` steps |
| `tracking.txt` | Text | Simple cell tracking (time, x, y, cell_id) | If tracking enabled |

### 3D Simulation Output
| File | Format | Contents | When Generated |
|------|--------|----------|----------------|
| `checkpoint_3d_*.bin` | Binary | 3D cell phase fields + metadata | Every `save_interval` steps |
| `trajectory.txt` | Text | Cell centroids, polarization (3D) | Every `trajectory_interval` steps |

### Trajectory File Format
```
# Trajectory data for MSD computation
# Format: time cell_id x y vx vy px py theta
# v_A=0.004 N=288 Lx=1600 Ly=1600
80000.000000 0 1361.088 614.591 -0.000001 -0.000006 0.246579 -0.969123 4.961538
```

**Header line** contains simulation parameters:
- `v_A` - Active velocity
- `N` - Number of cells
- `Lx`, `Ly` - Domain size (for periodic boundary unwrapping)

## Python Scripts Reference

### `visualize.py` - 2D Field Visualization
**Purpose:** Plot 2D phase field frames from VTK files, generate movies, visualize energy

**Location:** `cpp/simulation/postprocessing/visualize.py`

```powershell
# Plot last frame
python visualize.py agent_test_runs/my_sim --last 1

# Generate movie from images
python visualize.py agent_test_runs/my_sim --movie

# Create movie from existing images (skip regenerating PNGs)
python visualize.py agent_test_runs/my_sim --movie-only

# Frame range
python visualize.py agent_test_runs/my_sim --start 0 --end 100

# Single specific frame
python visualize.py agent_test_runs/my_sim -f 500

# Process in reverse order (last to first)
python visualize.py agent_test_runs/my_sim -r

# Show polarization arrows on cells
python visualize.py agent_test_runs/my_sim --use-arrows --arrow-scale 15

# Energy visualization mode (stress heatmap + energy time series)
python visualize.py agent_test_runs/my_sim --energy

# Control FPS for movies
python visualize.py agent_test_runs/my_sim --movie --fps 15
```

**Command-line options:**
| Flag | Description |
|------|-------------|
| `--last N` | Process only last N frames |
| `--start`, `--end` | Frame range to process |
| `-f`, `--frame` | Single specific frame number |
| `-r`, `--reverse` | Process frames in reverse order |
| `--movie` | Create movie after generating images |
| `--movie-only` | Create movie from existing images |
| `--fps N` | Frames per second for movie (default: 10) |
| `--use-arrows` | Show polarization arrows on cells |
| `--arrow-scale` | Scale factor for arrows (default: 15) |
| `--energy` | Energy visualization mode with stress heatmap |
| `--energy-only` | Energy movie from existing data (skip recomputation) |
| `--no-save` | Don't save PNG images |

**Output:** 
- `images/frame_*.png` - Individual frame images
- `simulation.mp4` - Animation movie
- `energy_images/frame_*.png` - Energy visualization frames (with `--energy`)
- `energy_movie.mp4` - Energy visualization movie
- `energy.txt` - Energy time series data

---

### `visualize_3d.py` - 3D Volume Rendering
**Purpose:** Render 3D cell volumes from binary checkpoint files using PyVista

**Location:** `cpp/simulation/postprocessing/visualize_3d.py`

**Note:** This script reads **binary checkpoint files** (`checkpoint_3d_*.bin`), NOT VTK files.

```powershell
# Volume rendering (semi-transparent cells showing interiors)
python visualize_3d.py agent_test_runs/my_3d_sim --volume

# Volume rendering with boundary-only mode (soap bubble effect - DEFAULT)
python visualize_3d.py agent_test_runs/my_3d_sim --volume

# Solid cells (no transparency at center)
python visualize_3d.py agent_test_runs/my_3d_sim --volume --no-boundary

# Isosurface rendering (shows cell boundaries as surfaces)
python visualize_3d.py agent_test_runs/my_3d_sim --iso 0.5

# Generate 3D movie
python visualize_3d.py agent_test_runs/my_3d_sim --movie --volume

# Show specific cells only
python visualize_3d.py agent_test_runs/my_3d_sim --volume --cells "0,1,5"

# Custom colormap for volume rendering
python visualize_3d.py agent_test_runs/my_3d_sim --volume --cmap plasma

# Single frame
python visualize_3d.py agent_test_runs/my_3d_sim -f 100

# Single checkpoint file (interactive)
python visualize_3d.py agent_test_runs/checkpoint_3d_000500.bin --volume
```

**Command-line options:**
| Flag | Description |
|------|-------------|
| `--volume` | Use volume rendering (3D heatmap style) |
| `--iso VALUE` | Isosurface value for surface extraction (default: 0.5) |
| `--no-boundary` | Disable boundary mode (show solid cells instead of soap-bubble effect) |
| `--cells "0,1,5"` | Show only specific cell IDs |
| `--cmap NAME` | Colormap for volume rendering (default: viridis) |
| `--movie` | Create movie after generating images |
| `--movie-only` | Create movie from existing images |
| `--fps N` | Frames per second for movie (default: 10) |
| `-f`, `--frame` | Single specific frame number |
| `--start`, `--end` | Frame range to process |
| `--grid` | Show grid lines |
| `--screenshot FILE` | Save single screenshot to file |

**Rendering modes:**
- **Volume (default)**: Semi-transparent rendering showing cell interiors, with boundary mode making cell centers nearly invisible (soap-bubble effect)
- **Isosurface**: Extracts surface mesh at φ=0.5, colored per cell

**Output:**
- `images/checkpoint_3d_*.png` - Individual frame screenshots
- `simulation_3d.mp4` - Animation movie

---

### `analyze_trajectory.py` - Trajectory Analysis
**Purpose:** Compute MSD, velocity/polarization autocorrelation, diffusion coefficient

**Location:** `cpp/simulation/postprocessing/analyze_trajectory.py`

```powershell
# Full analysis with plots
python analyze_trajectory.py agent_test_runs/my_sim

# Save plots without display
python analyze_trajectory.py agent_test_runs/my_sim --no-show
```

**Computes:**
- Mean Squared Displacement (MSD) vs time lag (with periodic boundary unwrapping)
- Velocity autocorrelation function C_v(t)
- Polarization autocorrelation function C_p(t)
- Effective diffusion coefficient D_eff from MSD slope (long-time fit)
- Persistence time τ_p from exponential fit to polarization decay

**Output:** 
- `msd.png` - MSD plots (log-log and linear scale)
- `autocorrelations.png` - Velocity and polarization autocorrelation
- `trajectories.png` - 2D cell trajectories (split at periodic boundaries)

**Limitations:**
- Single-threaded Python, slow for large datasets (>10k time points)
- For batch processing (many replicates), use `msd_calculator` (see below)

---

### `stack_videos.py` - Video Grid Layout
**Purpose:** Combine multiple videos into side-by-side or grid layout

**Location:** `cpp/simulation/postprocessing/stack_videos.py`

```powershell
# Side by side (default 1x2 grid)
python stack_videos.py video1.mp4 video2.mp4 -o comparison.mp4

# 2x2 grid
python stack_videos.py v1.mp4 v2.mp4 v3.mp4 v4.mp4 -o grid.mp4 --grid 2x2

# Vertical stack (2x1)
python stack_videos.py top.mp4 bottom.mp4 -o stacked.mp4 --grid 2x1

# With labels
python stack_videos.py low_vA.mp4 high_vA.mp4 -o comparison.mp4 --labels "v_A=0.004" "v_A=0.012"

# Custom label size
python stack_videos.py v1.mp4 v2.mp4 -o out.mp4 --labels "A" "B" --label-size 36
```

**Command-line options:**
| Flag | Description |
|------|-------------|
| `-o`, `--output` | Output video file (required) |
| `--grid ROWSxCOLS` | Grid layout (default: 1x2) |
| `--labels "A" "B"` | Labels for each video |
| `--label-size N` | Font size for labels (default: 24) |
| `--fps N` | Output frame rate |

**Requires:** `moviepy` (`pip install moviepy`)

**Output:** Combined video file

---

### `plot_jamming_transition.py` - Jamming Transition Plot
**Purpose:** Plot D(v_A) curve from batch MSD analysis to visualize jamming transition

**Location:** `cpp/simulation/postprocessing/plot_jamming_transition.py`

```powershell
# Basic usage
python plot_jamming_transition.py diffusion_results.txt

# Custom output path
python plot_jamming_transition.py diffusion_results.txt -o jamming_curve.png

# Save without displaying
python plot_jamming_transition.py diffusion_results.txt --no-show
```

**Input:** Output from `msd_calculator --batch` (see below)

**Output:** `jamming_transition.png` with:
- D vs v_A (linear scale) with error bars and jammed region shading
- D vs v_A (log-log scale) with D ∝ v_A² reference line
- Estimated critical velocity v_A^c
- Summary statistics printed to console

---

### `cluster/msd_analysis.py` - MSD from Tracking Data
**Purpose:** Compute MSD from tracking.txt file format (alternative to trajectory.txt)

**Location:** `cpp/simulation/cluster/msd_analysis.py`

```powershell
# Basic MSD analysis
python msd_analysis.py tracking.txt

# With plot
python msd_analysis.py tracking.txt --plot
```

**Input format:** `time x y cell_id` columns

**Output:**
- `<filename>_msd.txt` - MSD data (lag_time, msd, msd_err)
- `<filename>_msd.png` - MSD plots (if `--plot` specified)

**Note:** For trajectory.txt format (with velocities and polarization), use `analyze_trajectory.py` instead.

---

### `plot_observables.py` - Diagnostic Observables
**Purpose:** Plot energy, stress, pressure, coordination from GPU diagnostics CSV

**Location:** `cpp/simulation/postprocessing/plot_observables.py`

```powershell
# Basic usage (auto-finds observables.csv in current dir)
python plot_observables.py

# Specify CSV file
python plot_observables.py path/to/observables.csv

# Specify output directory
python plot_observables.py path/to/observables.csv path/to/output_dir
```

**Requires:** Simulation built with `-DENABLE_DIAGNOSTICS=ON`

**Input CSV columns:** `step,time,E_grad,E_bulk,E_int,E_total,sigma_xx,sigma_yy,sigma_xy,pressure,z_mean,z_std,z_min,z_max`

**Plots (2×3 grid):**
1. Energy components (E_grad, E_bulk, E_int)
2. Total energy vs time
3. Stress tensor components (σ_xx, σ_yy, σ_xy)
4. Pressure vs time
5. Mean coordination with ±1σ band
6. Coordination range (z_min, z_max, z_mean)

**Output:** 
- `observables_plot.png`
- Summary statistics printed to console

**Requires:** `pandas` (`pip install pandas`)

---

### `visualize_stress.py` - Stress Field Visualization
**Purpose:** Visualize stress tensor fields from VTK output with beautiful colormaps and glow effects

**Location:** `cpp/simulation/postprocessing/visualize_stress.py`

```powershell
# Visualize last frame with all stress fields (2×3 grid)
python visualize_stress.py agent_test_runs/my_sim

# Specific frame
python visualize_stress.py agent_test_runs/my_sim -f 500

# Single stress field
python visualize_stress.py agent_test_runs/my_sim --field von_mises

# All fields in one figure
python visualize_stress.py agent_test_runs/my_sim -f 500 --all-fields

# Generate single-field movie
python visualize_stress.py agent_test_runs/my_sim --movie --field tau_max

# Generate combined 6-panel movie (all fields)
python visualize_stress.py agent_test_runs/my_sim --combined

# Disable glow effects
python visualize_stress.py agent_test_runs/my_sim --combined --no-glow

# Movie from existing images
python visualize_stress.py agent_test_runs/my_sim --movie-only
```

**Command-line options:**
| Flag | Description |
|------|-------------|
| `-f`, `--frame` | Specific frame number |
| `--field NAME` | Stress field to show: `von_mises`, `tau_max`, `pressure`, `sigma_xx`, `sigma_yy`, `sigma_xy`, `sigma_1`, `sigma_2` |
| `--all-fields` | Show all 6 fields in single figure |
| `--movie` | Generate single-field movie |
| `--combined` | Generate 6-panel combined movie |
| `--movie-only` | Use existing images (skip regeneration) |
| `--fps N` | Frames per second (default: 10) |
| `--last N` | Only process last N frames |
| `--no-glow` | Disable glow effects on high-stress regions |

**Requires:** Simulation built with stress output (VTK must contain `sigma_xx`, `sigma_yy`, `sigma_xy` fields)

**Stress quantities computed:**
- Von Mises stress: σ_vm = √(σ₁² - σ₁σ₂ + σ₂²)
- Max shear stress: τ_max = (σ₁ - σ₂)/2
- Principal stresses: σ₁, σ₂ (eigenvalues)
- Pressure: p = -(σ_xx + σ_yy)/2

**Output:**
- `stress_all_NNNNNN.png` - All fields (with `--all-fields`)
- `stress_{field}_NNNNNN.png` - Single field
- `stress_images_{field}/` - Movie frame images
- `stress_{field}_movie.mp4` - Single field movie
- `stress_images_combined/` - Combined movie frames
- `stress_combined_movie.mp4` - 6-panel movie

**Features:**
- Custom colormaps optimized for stress visualization
- Power-law normalization to show low-stress regions
- Optional glow effects on high-stress regions
- Dark background for better contrast

---

### `visualize_combined.py` - Synchronized Movie + Observables
**Purpose:** Create video with simulation frames synced to observable traces

**Location:** `cpp/simulation/postprocessing/visualize_combined.py`

```powershell
# Basic usage
python visualize_combined.py agent_test_runs/my_sim

# Custom output path
python visualize_combined.py agent_test_runs/my_sim -o combined_movie.mp4

# Custom frame rate
python visualize_combined.py agent_test_runs/my_sim --fps 15
```

**Requires:** 
- Both VTK frames (`frame_*.vtk`) and `observables.csv` in the directory
- Simulation built with `-DENABLE_DIAGNOSTICS=ON`
- `vtk` package (`pip install vtk`)

**Layout:**
- Top left: Simulation frame (cell phase field)
- Right side: Energy components and stress traces
- Bottom: Pressure and coordination number traces

All traces show data progressively synced to the current simulation frame, with markers indicating the current time point.

**Output:** `combined_movie.mp4` (or custom path with `-o`)

**Requires:** `pandas`, `vtk`, `imageio`

---

## High-Performance Analysis Tools

For large-scale analysis (e.g., processing hundreds of trajectory files), use compiled C tools instead of Python.

### `cluster/msd_calculator.c` - Fast MSD/Diffusion Calculator
**Purpose:** Compute diffusion coefficients from trajectory data at scale

**Compile (on cluster):**
```bash
module load gcc/12.3
gcc -O3 -o msd_calculator msd_calculator.c -lm
```

**Usage:**
```bash
# Single file
./msd_calculator trajectory.txt 96000 msd_output.txt

# Batch processing (all replicates)
./msd_calculator --batch /path/to/production 96000 diffusion_results.txt
```

**Output format:**
```
# Diffusion coefficients from jamming study
# v_A D D_stderr n_replicates
0.004000 1.234567e-02 5.678901e-04 100
0.005000 2.345678e-02 6.789012e-04 99
...
```

**Note:** For cluster usage, submit via SLURM - see `cluster/msd_job.sh`

---

## Jamming Transition Analysis

### Goal
Reproduce Palmieri et al. jamming transition: D(v_A) showing:
- D ≈ 0 below critical v_A (jammed phase)
- D increases sharply above v_A^c (unjammed phase)

### Data Requirements
- Multiple velocity values (v_A sweep)
- Many replicates per velocity (50-100) for statistics
- Long trajectories (t ~ 10^4 - 10^5) for reliable MSD slope

### Analysis Workflow
1. **Run production simulations** on cluster with velocity sweep
2. **Compute D(v_A)** using `msd_calculator --batch`
3. **Plot jamming transition** - D vs v_A curve

### Plotting Diffusion Results

Use `plot_jamming_transition.py` for publication-ready plots:

```powershell
# From cluster results
python plot_jamming_transition.py diffusion_results.txt -o jamming_transition.png
```

Or manually with matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results from msd_calculator
data = np.loadtxt('diffusion_results.txt', comments='#')
v_A = data[:, 0]
D = data[:, 1]
D_err = data[:, 2]

plt.figure(figsize=(8, 6))
plt.errorbar(v_A, D, yerr=D_err, fmt='o-', capsize=3)
plt.xlabel('Active velocity $v_A$')
plt.ylabel('Diffusion coefficient $D$')
plt.title('Jamming Transition')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.savefig('jamming_transition.png', dpi=150)
plt.show()
```

---

## Key Physics Quantities

| Quantity | Symbol | How to Compute | Script |
|----------|--------|----------------|--------|
| Mean Squared Displacement | MSD(t) | ⟨\|r(t) - r(0)\|²⟩ | `analyze_trajectory.py`, `msd_calculator` |
| Diffusion coefficient | D | MSD ~ 4Dt (2D) at long times | `analyze_trajectory.py`, `msd_calculator` |
| Velocity autocorrelation | C_v(t) | ⟨v(0)·v(t)⟩/⟨v²⟩ | `analyze_trajectory.py` |
| Polarization autocorrelation | C_p(t) | ⟨p(0)·p(t)⟩ | `analyze_trajectory.py` |
| Persistence time | τ_p | Decay time of C_p(t) | `analyze_trajectory.py` |
| Gradient energy | E_grad | γ ∫ \|∇φ\|² dx | `visualize.py --energy` |
| Bulk energy | E_bulk | (30/λ²) ∫ φ²(1-φ)² dx | `visualize.py --energy` |
| Interaction energy | E_int | From overlap regions (φ > 1) | `visualize.py --energy` |

---

## Adding New Analysis

When adding new post-processing:

1. **Python for prototyping**: Write initial version in Python
2. **C for production**: Port to C if processing many files
3. **SLURM for cluster**: Create job script for batch processing
4. **Test locally first**: Download small sample, verify results
5. **Document here**: Add to this instructions file

### Template for new analysis script
```python
#!/usr/bin/env python3
"""Brief description of analysis."""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('input_dir', help='Directory containing data')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    
    # Load data
    # Compute metrics
    # Plot results
    # Save output
    
if __name__ == '__main__':
    main()
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `PyVista not available` | `pip install pyvista` |
| `moviepy` import error | `pip install moviepy` |
| Plots don't display | Add `--no-show` flag, check saved files |
| VTK read error | Check file format (legacy vs XML), ensure DIMENSIONS line present |
| 3D checkpoint read error | Verify file is `checkpoint_3d_*.bin`, not VTK |
| MSD negative slope | Trajectory too short for diffusive regime, or cells fully caged |
| D ≈ 0 for all velocities | System is jammed - try higher v_A values |
| D increases then plateaus | Finite-size effects - cells hitting periodic boundaries |
| `--energy` mode missing data | Need `trajectory.txt` for polarization arrows |
| Movie creation fails | Install `imageio imageio-ffmpeg` or ensure ffmpeg in PATH |

---

## Debug Scripts (cluster/debug/)

These scripts are for debugging and validation during development. **Not for regular use.**

| Script | Purpose |
|--------|---------|
| `debug_msd.py` | Verbose MSD debugging with per-cell output |
| `validate_msd_python.py` | Compare Python vs C MSD implementation |
| `analyze_msd_detailed.py` | Analyze MSD for jamming signatures |
| `analyze_trajectories.py` | Visualize raw cell trajectories |
| `check_crossover.py` | Detect ballistic→diffusive crossover |
| `plot_D_vs_vA.py` | Quick D(v_A) plot with hardcoded data |

**Note:** These were created during algorithm development and troubleshooting. For actual analysis, use the main scripts in `postprocessing/`.

---

## Script Consolidation Notes

The following redundant scripts were **deleted** during organization:

| Deleted Script | Reason | Use Instead |
|----------------|--------|-------------|
| `compute_diffusion.py` | Duplicate of `analyze_trajectory.py` | `postprocessing/analyze_trajectory.py` |

The `analyze_trajectory.py` script provides all MSD functionality plus:
- Velocity and polarization autocorrelations
- Persistence time fitting
- Publication-quality plots

---

## Dependencies

Required Python packages:
```bash
pip install numpy matplotlib
pip install pyvista        # For 3D visualization
pip install moviepy        # For video stacking
pip install imageio imageio-ffmpeg  # For movie creation (alternative)
```

For cluster C code:
```bash
module load gcc/12.3
```
