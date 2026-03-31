# Simulation Runbook (Quick Reference)

**For AI agents: Use this when you need to build, run, or test the simulation.**

---

## Cluster Connection

**Hostname:** `nibi.alliancecan.ca`  
**Username:** `ssilber`  
**Scratch:** `/scratch/ssilber`

### Cluster Directory Structure

```
/scratch/ssilber/
├── cell_sim/                    # MAIN simulation codebase and job submission scripts
│   ├── build/bin/cell_sim       # Production executable
│   ├── submit_jobs.sh           # Job submission scripts
│   └── ...                      # Full simulation source
│
├── jamming_study/               # ACTIVE production runs (jamming/confluence study)
│   ├── production/              # Running production jobs
│   │   ├── v0_r1/               # 72-cell runs at various v_A values
│   │   │   └── checkpoint.bin   # Equilibrated checkpoint (288 cells, t≈97874)
│   │   ├── v0_r4/               # 288-cell runs
│   │   │   └── checkpoint.bin   # Equilibrated checkpoint
│   │   └── p288_*/              # 288-cell production at different phi values
│   └── ...                      # Job arrays, analysis scripts
│
├── simulation_diag/             # Diagnostic-enabled build (experimental)
│   ├── build/bin/cell_sim       # Executable with -DENABLE_DIAGNOSTICS=ON
│   └── simulation_code.tar      # Uploaded source archive
│
└── diag_test2/                  # Output from diagnostic test runs
    └── observables.csv          # Diagnostic output (energy, stress, coordination)
```

**Current Activity (Dec 2025):**
- Production jobs running in `jamming_study/production/p288_*` (288-cell, various φ values)
- Jobs submitted via `rrg-mkarttu-ab` account (RAC allocation)
- Diagnostic testing in `simulation_diag/` with `def-mkarttu` account

**IMPORTANT:** 
- `/scratch/ssilber/cell_sim/` is the **primary working directory** with all submission infrastructure
- `/scratch/ssilber/simulation_diag/` is for experimental diagnostic builds only
- Always document which cluster directory you're working in when making changes

### Persistent SSH via WSL (Required)

Windows native SSH doesn't support ControlMaster. Use WSL:

```powershell
# 1. Set up persistent connection (authenticate once, lasts 4 hours)
wsl mkdir -p ~/.ssh/sockets
wsl ssh -M -S ~/.ssh/sockets/nibi -o ControlPersist=4h -fN ssilber@nibi.alliancecan.ca

# 2. All subsequent commands use the persistent connection (no MFA prompts)
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "command here"

# 3. File transfers via WSL
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" file.tar ssilber@nibi.alliancecan.ca:/scratch/ssilber/
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" ssilber@nibi.alliancecan.ca:/scratch/ssilber/file.tar /mnt/c/path/

# 4. Check connection status
wsl ssh -S ~/.ssh/sockets/nibi -O check ssilber@nibi.alliancecan.ca

# 5. Close connection when done
wsl ssh -S ~/.ssh/sockets/nibi -O exit ssilber@nibi.alliancecan.ca
```

### Build on Cluster

```bash
# Load required modules
module load cuda/12.6 cmake/3.31.0

# Build (standard)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# Build with diagnostics
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_DIAGNOSTICS=ON
make -j8
```

### Key Cluster Paths
- **Main codebase**: `/scratch/ssilber/cell_sim/` (production builds and job scripts)
- **Active production**: `/scratch/ssilber/jamming_study/` (running 288-cell jobs at various φ)
- **Diagnostic build**: `/scratch/ssilber/simulation_diag/build/bin/cell_sim`
- **72-cell checkpoint**: `/scratch/ssilber/jamming_study/production/v0_r1/checkpoint.bin` (actually 288 cells, t≈97874)
- **288-cell checkpoint**: `/scratch/ssilber/jamming_study/production/v0_r4/checkpoint.bin`

### SLURM Job Submission
```bash
# Required account for GPU jobs
#SBATCH --account=def-mkarttu

# Alternative accounts available:
# - def-mkarttu (default)
# - rrg-mkarttu-ab (RAC allocation)

# Do NOT specify --partition; let SLURM auto-route
```

---

## Build Commands

```powershell
cd c:\Users\stevensilber\source\repos\data_processing\cpp\simulation

# Standard Release build
mkdir build -Force; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Safe mode (use when loading checkpoints or debugging memory)
cmake .. -DCMAKE_BUILD_TYPE=Release -DSAFE_MODE=ON
cmake --build . --config Release

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
```

**Executable location:** `build\bin\Release\cell_sim.exe`

---

## Run Commands

### 2D Simulations

```powershell
# Single cell test (validation)
.\build\bin\Release\cell_sim.exe -n 1 -N 256 -r 49 -t 10 --dt 0.01 -o agent_test_runs/test_2d_single

# Multi-cell test
.\build\bin\Release\cell_sim.exe -n 8 -N 512 -r 49 -t 10 --dt 0.01 -o agent_test_runs/test_2d_multi

# With motility
.\build\bin\Release\cell_sim.exe -n 8 -N 512 -r 49 -t 100 --dt 0.01 --v-A 0.01 --trajectory-samples 200 -o agent_test_runs/test_2d_motile

# Save checkpoint for later
.\build\bin\Release\cell_sim.exe -n 8 -N 512 -r 49 -t 50 --dt 0.01 --save-final-checkpoint -o agent_test_runs/test_checkpoint
```

### 3D Simulations

```powershell
# Single cell test (validation)
.\build\bin\Release\cell_sim.exe --3d -n 1 -N 240 -r 49 -t 10 --dt 0.02 --checkpoint-interval 500 -o agent_test_runs/test_3d_single

# Multi-cell test (16 cells at 85% confluence, R=49)
# Confluence calculation: Domain = cbrt(n_cells * (4/3)*pi*R^3 / confluence)
# For 16 cells, R=49, 85% confluence: N = cbrt(16 * 492807 / 0.85) ≈ 210
.\build\bin\Release\cell_sim.exe --3d -n 16 -N 210 -r 49 -t 10 --dt 0.02 --checkpoint-interval 500 -o agent_test_runs/test_3d_multi
```

**IMPORTANT**:
- Use `--dt 0.02` for 3D (not 0.01) - 3D is more stable and this halves runtime
- Use random initialization (default). Do NOT use --grid or --confluence flags
- Calculate domain size manually: `N = cbrt(n_cells * cell_volume / target_confluence)`

### Resume from Checkpoint

```powershell
.\build\bin\Release\cell_sim.exe -c agent_test_runs/test_checkpoint/checkpoint.bin -t 100 -o agent_test_runs/resumed_run
```

**⚠️ v_A regeneration:** v4 checkpoints store per-cell v_A. When resuming from an equilibration checkpoint (v_A=0) for production, you **must** specify `--v-A` on the command line to regenerate per-cell values. Without it, cells will have v_A=0 from the checkpoint. Specifying `--v-A` or `--v-A-sigma` automatically clears checkpoint v_A values and regenerates fresh ones.

---

## CLI Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `-n <num>` | Number of cells | 8 |
| `-N <size>` | Domain size (2D: NxN) | 256 |
| `--size <n>` | Domain size (3D: NxNxN) | 100 |
| `-r <radius>` | Cell radius | 49 |
| `-t <time>` | End time | 100 |
| `--dt <step>` | Time step | 0.01 |
| `-o <dir>` | Output directory | ./output |
| `-c <file>` | Load checkpoint | — |
| `--v-A <f>` | Active velocity | 0 |
| `--v-A-sigma <f>` | Per-cell v_A disorder std dev (log-normal) | 0 |
| `--3d` | Enable 3D mode | false |
| `--save-interval <n>` | Steps between VTK saves | 100 |
| `--trajectory-samples <n>` | Trajectory points to save | 100 |
| `--checkpoint-interval <n>` | Steps between checkpoints | save_interval×10 |
| `--save-final-checkpoint` | Save checkpoint at end | false |
| `--seed <n>` | Random seed | — |
| `--subdomain-padding <f>` | Cell bbox size as multiple of R | 2.0 |

---

## 3D Optimization Options

For faster 3D simulations:

| Option | Effect | Notes |
|--------|--------|-------|
| `--dt 0.02` | 2× fewer steps | 3D is more stable than 2D |
| `--checkpoint-interval 500` | Fewer disk writes | Checkpoints are large in 3D |
| `--save-interval 0` | No VTK output | Use for benchmarking |
| `--subdomain-padding 1.5` | Smaller cell bboxes | Reduces memory but may need more bbox updates |

**Internal optimizations** (automatic):
- Batched GPU kernels process all cells in parallel
- Centroid sync only every 10 steps (not every step)
- Fused kernel pipeline minimizes memory transfers
- Work buffer: 5 buffers per cell (optimized from 7)

---

## Validation Test Suite

**Run these after any code changes:**

```powershell
# Test 1: 2D single cell
.\build\bin\Release\cell_sim.exe -n 1 -N 256 -r 49 -t 10 --dt 0.01 -o agent_test_runs/validate_2d_single
# Expected: volume ~7543, phi_max ~1.0, no NaN

# Test 2: 2D multi-cell
.\build\bin\Release\cell_sim.exe -n 8 -N 512 -r 49 -t 10 --dt 0.01 -o agent_test_runs/validate_2d_multi
# Expected: cells repel, volumes stable

# Test 3: 3D single cell
.\build\bin\Release\cell_sim.exe --3d -n 1 -N 240 -r 49 -t 10 --dt 0.02 --checkpoint-interval 500 -o agent_test_runs/validate_3d_single
# Expected: volume ~492807, phi_max ~1.0, no NaN

# Test 4: 3D multi-cell (16 cells, 85% confluence)
# N = cbrt(16 * 492807 / 0.85) ≈ 210
.\build\bin\Release\cell_sim.exe --3d -n 16 -N 210 -r 49 -t 10 --dt 0.02 --checkpoint-interval 500 -o agent_test_runs/validate_3d_multi
# Expected: cells interact and repel, no collapse
```

---

## Visualization Commands

```powershell
# 2D: Plot last frame
python visualize.py -d agent_test_runs/my_sim --last

# 2D: Generate movie
python visualize.py -d agent_test_runs/my_sim --movie

# 3D: Isosurface visualization
python visualize_3d.py agent_test_runs/my_3d_sim

# 3D: Volume rendering (preferred for dense packing)
python visualize_3d.py agent_test_runs/my_3d_sim --volume

# 3D: Generate movie with volume rendering
python visualize_3d.py agent_test_runs/my_3d_sim --movie --volume

# Trajectory analysis (MSD, autocorrelations)
python analyze_trajectory.py agent_test_runs/my_sim --no-show
```

**IMPORTANT**: Always use `--volume` flag for 3D visualization to see cell interiors.

---

## Stress Field Visualization

**Prerequisite:** Run simulation with `--stress-fields` flag (requires `-DENABLE_STRESS_FIELDS=ON` build):

```powershell
# Build with stress fields enabled
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_STRESS_FIELDS=ON
cmake --build . --config Release

# Run simulation with stress output
.\build\bin\Release\cell_sim.exe -n 16 -N 512 -r 49 -t 100 --v-A 0.03 --tau 100 --stress-fields -o agent_test_runs/stress_test
```

### visualize_stress.py Usage

```powershell
# Default: show last frame with all 6 stress panels
python visualize_stress.py agent_test_runs/stress_test

# Show specific frame
python visualize_stress.py agent_test_runs/stress_test -f 500

# Single stress field instead of 6-panel
python visualize_stress.py agent_test_runs/stress_test --field von_mises

# Generate combined 6-panel movie (recommended)
python visualize_stress.py agent_test_runs/stress_test --combined

# Regenerate movie from existing images (fast)
python visualize_stress.py agent_test_runs/stress_test --movie-only

# Generate single-field movie
python visualize_stress.py agent_test_runs/stress_test --movie --field von_mises

# Faster movie (higher FPS)
python visualize_stress.py agent_test_runs/stress_test --combined --fps 30

# Last N frames only
python visualize_stress.py agent_test_runs/stress_test --combined --last 200

# Disable glow effects (cleaner but less dramatic)
python visualize_stress.py agent_test_runs/stress_test --combined --no-glow
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `output_dir` | Directory with VTK files | output |
| `-f`, `--frame` | Specific frame number | last |
| `--field` | Stress field to visualize | von_mises |
| `--all-fields` | Show all fields in one figure | false |
| `--movie` | Generate single-field movie | false |
| `--combined` | Generate 6-panel movie | false |
| `--movie-only` | Skip image generation, use existing | false |
| `--fps` | Movie frame rate | 10 |
| `--last` | Only process last N frames | all |
| `--no-glow` | Disable glow effects | false |

### Stress Fields Explained

| Field | Description | Colormap |
|-------|-------------|----------|
| **von_mises** | Equivalent stress (overall stress magnitude) | plasma |
| **tau_max** | Maximum shear stress | custom (black→violet→blue→cyan→lime→yellow→orange→pink→white) |
| **pressure** | Hydrostatic pressure (P = -(σ_xx+σ_yy)/2) | RdBu_r (blue=tension, red=compression) |
| **sigma_xx** | Normal stress in x | RdBu_r |
| **sigma_yy** | Normal stress in y | RdBu_r |
| **sigma_xy** | Shear stress | PuOr_r |

### Output Files

```
stress_test/
├── frame_000000.vtk       # VTK files with stress fields
├── frame_000100.vtk
├── ...
├── stress_combined_movie.mp4   # 6-panel movie (from --combined)
├── stress_images_combined/     # Individual PNG frames
│   ├── frame_000000.png
│   └── ...
└── stress_images_von_mises/    # Single-field images (from --movie)
```

---

## Output Directory Convention

**All test output goes to:** `agent_test_runs/`

Use descriptive names:
```powershell
-o agent_test_runs/feature_name_test
-o agent_test_runs/bugfix_validation
-o agent_test_runs/perf_benchmark_n64
```

---

## Development Workflow

### When adding a feature:
1. Build in Release mode
2. Run validation test suite (all 4 tests)
3. Run your specific test case
4. Visualize results to confirm correctness
5. Build in Debug mode and run again if issues found

### When fixing a bug:
1. Create a minimal reproduction case
2. Build with SAFE_MODE=ON if memory-related
3. Fix the issue
4. Run validation test suite
5. Verify the reproduction case is fixed

### When optimizing performance:
1. Run baseline benchmark: `.\build\bin\Release\cell_sim.exe -n 16 -N 512 -t 100 --no-vtk -o agent_test_runs/baseline`
2. Note the runtime
3. Make changes
4. Run same benchmark
5. Compare runtimes
6. Run validation test suite to ensure correctness

---

## Key Parameters

| Parameter | 2D Value | 3D Value | Notes |
|-----------|----------|----------|-------|
| R (radius) | 49 | 49 | Target area/volume |
| λ (interface) | 7 | 7 | Interface width |
| γ (gradient) | 1 | 1 | Interface energy |
| κ (repulsion) | 10 | 10 | Cell-cell repulsion |
| μ (volume) | 1 | 1 | Volume constraint |
| Target volume | 7543 | 492807 | πR² (2D), 4πR³/3 (3D) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| NaN in output | Reduce dt by 2x, check bounding box updates |
| Volume drift | Check μ parameter, verify volume constraint term |
| OOM on checkpoint load | Build with -DSAFE_MODE=ON |
| Cells disappear | Check bounding box tracking, initialization |
| Build fails | Ensure CUDA 11+, CMake 3.18+, C++17 compiler |

---

*See AGENT_ONBOARDING.md for full documentation.*
