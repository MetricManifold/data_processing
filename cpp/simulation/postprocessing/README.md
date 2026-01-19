# Postprocessing Scripts

Analysis and visualization scripts for cell simulation output.

---

## Directory Structure

```
postprocessing/
├── README.md                    # This file
├── analyze_trajectory.py        # MSD & autocorrelation analysis
├── plot_energy.py               # Equilibration energy vs time
├── plot_jamming_transition.py   # D vs v_A plots (jamming study)
├── plot_observables.py          # Energy, pressure, etc. traces
├── stack_videos.py              # Grid layout for multiple videos
├── visualize.py                 # 2D VTK visualization
├── visualize_3d.py              # 3D PyVista rendering
├── visualize_combined.py        # Movie + observable sync
└── visualize_stress.py          # Stress tensor visualization
```

---

## Script Reference

### analyze_trajectory.py

**Purpose:** Comprehensive trajectory analysis for active matter physics.

**Computes:**
- Mean Squared Displacement (MSD) with proper periodic boundary unwrapping
- Velocity autocorrelation function C_v(Δt)
- Polarization autocorrelation function C_p(Δt)
- Effective diffusion coefficient D_eff
- Persistence time τ_p

**Usage:**
```bash
python analyze_trajectory.py <output_dir>
python analyze_trajectory.py <output_dir> --no-show  # Save plots only
```

**Input:** `trajectory.txt` in the output directory  
**Output:** `msd.png`, `autocorrelations.png`, `trajectories.png`

---

### plot_jamming_transition.py

**Purpose:** Create publication-ready D vs v_A plots for jamming transition studies.

**Usage:**
```bash
python plot_jamming_transition.py diffusion_results.txt
python plot_jamming_transition.py diffusion_results.txt --output jamming.png
```

**Input:** Output from cluster `msd_calculator` batch mode  
**Output:** Jamming transition plot (PNG)

---

### plot_energy.py

**Purpose:** Plot equilibration energy vs time with error bars from batch analysis.

**Usage:**
```bash
# Single file
python plot_energy.py energy_results.txt

# Compare packing fractions
python plot_energy.py energy_phi85.txt energy_phi89.txt

# Save to file
python plot_energy.py energy_phi89.txt --output energy.png --no-show
```

**Input:** Output from cluster `energy_analyzer` batch mode  
**Output:** Two-panel plot (linear and log scale)

---

### plot_observables.py

**Purpose:** Plot diagnostic observables over time (energy, pressure, etc.).

**Usage:**
```bash
python plot_observables.py observables.csv
python plot_observables.py observables.csv --output diagnostics.png
```

**Input:** `observables.csv` from simulation  
**Output:** Multi-panel figure with energy, pressure, and other quantities

---

### stack_videos.py

**Purpose:** Combine multiple simulation videos into grid layouts for comparison.

**Usage:**
```bash
# Side by side (1x2)
python stack_videos.py video1.mp4 video2.mp4 -o output.mp4

# 2x2 grid
python stack_videos.py v1.mp4 v2.mp4 v3.mp4 v4.mp4 -o output.mp4 --grid 2x2

# With labels
python stack_videos.py v1.mp4 v2.mp4 -o output.mp4 --labels "v_A=0.01" "v_A=0.02"
```

**Requires:** `moviepy`

---

### visualize.py

**Purpose:** 2D visualization of VTK checkpoint files using matplotlib.

**Usage:**
```bash
python visualize.py <output_dir>
python visualize.py <output_dir> --save-frames
python visualize.py <output_dir> --movie output.mp4
```

**Input:** VTK files (`checkpoint_*.vtk`) and optionally `trajectory.txt`  
**Output:** Interactive plot or movie file

---

### visualize_3d.py

**Purpose:** 3D isosurface rendering of cell fields using PyVista.

**Usage:**
```bash
python visualize_3d.py <checkpoint.vtk>
python visualize_3d.py <output_dir> --animate
python visualize_3d.py <output_dir> --save-frames
```

**Input:** 3D checkpoint files (binary format with `CS3D` magic)  
**Requires:** `pyvista`

---

### visualize_combined.py

**Purpose:** Synchronized movie with simulation frames and observable traces.

**Usage:**
```bash
python visualize_combined.py <output_dir> -o combined.mp4
```

**Input:** VTK checkpoints + `observables.csv`  
**Output:** Combined video with simulation on top, data traces below

---

### visualize_stress.py

**Purpose:** Visualize stress tensor fields from simulation output.

**Usage:**
```bash
python visualize_stress.py <stress_field.vtk>
```

**Input:** VTK files containing stress tensor components  
**Output:** Stress visualization with customizable color mapping

---

## Dependencies

| Script                    | Required Packages                          |
|---------------------------|--------------------------------------------|
| analyze_trajectory.py     | numpy, matplotlib                          |
| plot_energy.py            | numpy, matplotlib                          |
| plot_jamming_transition.py| numpy, matplotlib                          |
| plot_observables.py       | pandas, matplotlib                         |
| stack_videos.py           | moviepy                                    |
| visualize.py              | numpy, matplotlib                          |
| visualize_3d.py           | numpy, pyvista                             |
| visualize_combined.py     | numpy, pandas, matplotlib, vtk, imageio    |
| visualize_stress.py       | numpy, matplotlib, scipy                   |

Install all:
```bash
pip install numpy matplotlib pandas scipy pyvista vtk imageio moviepy
```

---

## Common Workflows

### After a simulation run:
```bash
# Quick visualization
python visualize.py agent_test_runs/my_run

# Generate movie
python visualize.py agent_test_runs/my_run --movie my_run.mp4

# Analyze trajectory for MSD and diffusion
python analyze_trajectory.py agent_test_runs/my_run
```

### For cluster results (jamming study):
```bash
# Download results from cluster
scp -r ssilber@nibi:$SCRATCH/cell_sim_results/production ./results/

# Run trajectory analysis on each run
for dir in results/v_*; do
    python analyze_trajectory.py "$dir" --no-show
done

# Plot jamming transition (after running msd_calculator on cluster)
python plot_jamming_transition.py results/diffusion_results.txt
```

### Compare multiple velocities:
```bash
# Make individual movies
python visualize.py results/v_0.004 --movie v004.mp4
python visualize.py results/v_0.008 --movie v008.mp4

# Stack side-by-side
python stack_videos.py v004.mp4 v008.mp4 -o comparison.mp4 --labels "v_A=0.004" "v_A=0.008"
```

---

## Related

- [cluster/README.md](../cluster/README.md) - Cluster submission and batch processing
- [cluster/msd_job.sh](../cluster/msd_job.sh) - SLURM job for batch MSD (uses C calculator)
- [cluster/msd_calculator.c](../cluster/msd_calculator.c) - Fast C implementation for MSD
- [cluster/energy_job.sh](../cluster/energy_job.sh) - SLURM job for equilibration energy analysis
- [cluster/energy_analyzer.c](../cluster/energy_analyzer.c) - Fast C implementation for energy analysis
