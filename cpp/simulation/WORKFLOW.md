# Cell Simulation Workflow

## Overview

This document describes the workflow for running cell simulations to measure diffusion and MSD (Mean Squared Displacement) for studying jamming transitions.

## Two-Phase Workflow

### Phase 1: Equilibration (v_A = 0)

First, equilibrate the system without active motility to let cells relax to a stable configuration:

```powershell
.\cell_sim.exe -n 12 -t 100 --v-A 0 -o output_equilibrated --save-interval 1000 --save-final-checkpoint
```

**Parameters:**
- `-n 12` — Number of cells
- `-t 100` — Run until t=100 (adjust based on system size)
- `--v-A 0` — No active self-propulsion during equilibration
- `-o output_equilibrated` — Output directory
- `--save-interval 1000` — Save VTK every 1000 steps (reduces I/O)
- `--save-final-checkpoint` — Save checkpoint at end for Phase 2

**Equilibration criteria:**
- Volume should stabilize near target (1256.6 for R=20)
- Shape factor should stabilize (~0.89-0.90)
- No significant cell rearrangements

### Phase 2: Production Run with Motility

Load the equilibrated checkpoint and enable active motility:

```powershell
.\cell_sim.exe -c output_equilibrated/checkpoint.bin -o output_motile -t 200 --v-A 0.5 --save-interval 100 --trajectory-samples 100
```

**Parameters:**
- `-c output_equilibrated/checkpoint.bin` — Load equilibrated state
- `-t 200` — New end time (100 time units more)
- `--v-A 0.5` — Active velocity (vary this for D vs v_A measurements)
- `--trajectory-samples 100` — Number of trajectory points for MSD
- `--save-interval 100` — VTK save frequency

## Trajectory Output

The simulation produces `trajectory.txt` with format:
```
# Trajectory data for MSD computation
# Format: time cell_id x y vx vy px py theta
# v_A=0.5 N=12
100.002953 0 206.943817 29.227455 0.188316 -0.463181 0.376633 -0.926363 5.098548
...
```

**Columns:**
1. `time` — Simulation time
2. `cell_id` — Cell identifier
3. `x, y` — Centroid position
4. `vx, vy` — Velocity components
5. `px, py` — Polarization unit vector
6. `theta` — Polarization angle (radians)

## Ensemble Runs

For statistical averaging, run multiple independent simulations with different seeds:

```powershell
# Equilibration with seed
.\cell_sim.exe -n 12 -t 100 --v-A 0 --seed 42 -o run_42/equilibrated --save-final-checkpoint

# Production with same seed (for reproducibility)
.\cell_sim.exe -c run_42/equilibrated/checkpoint.bin -o run_42/motile -t 200 --v-A 0.5 --trajectory-samples 200
```

## Parameter Sweep for D vs v_A

To measure diffusion coefficient as a function of active velocity:

```powershell
# Array of v_A values to test
$vA_values = @(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

foreach ($vA in $vA_values) {
    $outdir = "sweep/vA_$vA"
    .\cell_sim.exe -c equilibrated/checkpoint.bin -o $outdir -t 500 --v-A $vA --trajectory-samples 500
}
```

## Visualization

Generate PNG images from VTK output:

```powershell
python visualize.py output_motile
```

Images are saved to `output_motile/images/`.

Create a movie from frames:

```powershell
python visualize.py output_motile --movie --fps 15
```

## Resuming Interrupted Simulations

If a simulation is interrupted, you can resume from the last checkpoint:

```powershell
# Resume and continue to new end time
.\cell_sim.exe -c output_dir/checkpoint.bin -o output_dir -t <new_end_time> [other options]
```

**Example:** If your simulation was running to t=8000 but stopped at t=400 (saved in checkpoint):

```powershell
.\cell_sim.exe -c output_n72_c89_eq/checkpoint.bin -o output_n72_c89_eq -t 8000 --save-interval 4000
```

**Notes:**
- Checkpoints are saved periodically (every `checkpoint_interval` steps, default: `save_interval × 10`)
- The `-t` parameter is the absolute end time, not additional time
- You can change parameters like `--v-A` when resuming (e.g., to start motile phase)
- Output files will be appended to the same directory

## Command Reference

| Option | Description | Default |
|--------|-------------|---------|
| `-n <num>` | Number of cells | 8 |
| `-r <radius>` | Cell radius | 20 |
| `-N <size>` | Domain size NxN | 256 |
| `-t <time>` | End time | 100 |
| `-dt <step>` | Time step | 0.01 |
| `-o <dir>` | Output directory | ./output |
| `-c <file>` | Load checkpoint | — |
| `--v-A <f>` | Active velocity | from params |
| `--save-interval <n>` | Steps between VTK saves | 100 |
| `--trajectory-samples <n>` | Trajectory points to save | 100 |
| `--checkpoint-interval <n>` | Steps between checkpoints | save_interval×10 |
| `--save-final-checkpoint` | Save checkpoint at end | false |
| `--seed <n>` | Random seed | — |
| `--no-vtk` | Disable VTK output | false |
| `--no-diagnostics` | Skip diagnostics for speed | false |

## Post-Processing

The trajectory data can be analyzed with Python/NumPy to compute:

1. **MSD(τ)** = ⟨|r(t+τ) - r(t)|²⟩
2. **Diffusion coefficient** D = lim(τ→∞) MSD(τ)/(4τ)
3. **Velocity autocorrelation** ⟨v(t)·v(t+τ)⟩

See `analysis/` directory for post-processing scripts.

---

## 3D Simulations

### Quick Start - 3D

```powershell
# 8-cell 3D simulation (small test)
.\cell_sim.exe --3d -n 8 --size 100 -t 10 -dt 0.02 --save-interval 500 -o output3d

# 16-cell 3D simulation  
.\cell_sim.exe --3d -n 16 --size 150 -t 10 -dt 0.02 --save-interval 1000 -o output3d_16

# Large 64-cell 3D simulation (requires ~4GB GPU memory)
.\cell_sim.exe --3d -n 64 --size 200 -t 10 -dt 0.02 --save-interval 2000 -o output3d_64
```

### 3D Timing Guidelines

| Cells | Domain | Typical dt | Time for ~stable | Steps at dt=0.02 |
|-------|--------|------------|------------------|------------------|
| 8     | 100³   | 0.02       | t=5-10           | 250-500          |
| 16    | 150³   | 0.02       | t=5-10           | 250-500          |
| 64    | 200³   | 0.02       | t=5-10           | 250-500          |

**Note:** Don't simulate too far in time! t=10 is usually sufficient for equilibration. 
Check volume stabilization to confirm.

### 3D Memory Requirements

Per-cell memory (at 150³ max bbox):
- phi + dphi: 2 × 150³ × 4 bytes = 27 MB
- Work buffers: 7 × 150³ × 4 bytes = 94.5 MB per cell (fused kernel)

**Current memory model (parallel, N cells, 7 buffers/cell):**
- Work buffer: N × 7 × max_size × 4 bytes
- 64 cells at 150³: 64 × 7 × 3.375M × 4 = 6.0 GB ✓ Fits in 8GB GPU!

**Memory optimization achieved:**
- Fused kernel combines laplacian + bulk + constraint + advection in ONE pass
- Eliminates grad_x, grad_y, grad_z intermediate buffers (3 buffers saved)
- 30% memory savings while maintaining full parallel performance

**DO NOT use serial processing** - it kills performance. Always keep parallel.

### 3D Visualization

```powershell
# Single checkpoint
python visualize_3d.py output3d/checkpoint_001000.bin

# Directory (all checkpoints)
python visualize_3d.py output3d/

# Create movie
python visualize_3d.py output3d/ --movie --fps 10

# Movie only (no individual frames)
python visualize_3d.py output3d/ --movie-only --fps 10

# Specific frame range
python visualize_3d.py output3d/ --start 0 --end 500

# Specific cells only
python visualize_3d.py output3d/checkpoint_001000.bin --cells 0,1,2
```

### 3D-specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--3d` | Enable 3D mode | false |
| `--size <n>` | Domain size N×N×N | 100 |
| `--cell-radius <r>` | 3D cell radius | 15 |

### Physics Parameters (same in 2D and 3D)

| Parameter | Value | Notes |
|-----------|-------|-------|
| mu        | 1.0   | Volume constraint strength (dimensionless) |
| gamma     | 1.0   | Surface tension coefficient |
| kappa     | 0.5   | Gradient penalty |
| lambda    | 0.1   | Cell-cell repulsion |
| v_A       | 0.0   | Active motility (default off) |

**mu scaling:** The volume constraint `E = (μ/V₀)(V - V₀)²` gives relaxation rate 4μ 
regardless of dimension when properly normalized by target volume V₀.

