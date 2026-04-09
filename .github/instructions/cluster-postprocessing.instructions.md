````instructions
---
applyTo: "cpp/simulation/cluster/**"
---

# Cluster Post-Processing Development - Copilot Instructions

> **When to consult this file:** You are developing or running analysis tools that process simulation data on Alliance (Compute Canada) clusters. This covers the full develop-locally → test → submit-SLURM workflow, data format parsing, and batch processing patterns. For the Python visualization scripts themselves, see [postprocessing.instructions.md](postprocessing.instructions.md). For submitting simulation jobs, see [cluster-operations.instructions.md](cluster-operations.instructions.md).

This document describes the workflow for developing and running analysis tools on cluster simulation data.

## ⚠️ Related Instructions - Read Before Proceeding

| Task | Instruction File |
|------|-----------------|
| **Building & running simulations** | [cell-simulation.instructions.md](cell-simulation.instructions.md) |
| **Running simulations on cluster** | [cluster-operations.instructions.md](cluster-operations.instructions.md) |
| **Production runs (use `start_simulation` MCP tool)** | [cluster-operations.instructions.md](cluster-operations.instructions.md) - see "Jamming Study Production" |
| **Visualization scripts (Python)** | [postprocessing.instructions.md](postprocessing.instructions.md) |
| **Analysis feedback / known issues** | [FEEDBACK.md](rust/vtk_viewer/FEEDBACK.md) |

**Quick-reference READMEs:**
- [cluster/README.md](cpp/simulation/cluster/README.md) - Submission script usage & examples
- [postprocessing/README.md](cpp/simulation/postprocessing/README.md) - Analysis scripts overview

**Critical:** Prefer submitting analysis via SLURM (`run_analysis` MCP tool) with full-resolution data (no subsampling). Only use `--subsample` or `--threads 1` as a fallback when running on login nodes.

---

## ★ Preferred Workflow: cell_analyze (Rust)

The primary analysis tool is `cell_analyze`, a Rust binary that computes all standard glass/jamming observables from trajectory data. It is **200× faster than Python**, produces structured JSON output, and deploys as a single static binary with no module loads needed on the cluster.

### Cluster Deployment

The binary is deployed at `~/bin/cell_analyze` on nibi. To update:
```bash
# Build in WSL (not Windows — GUI deps block cross-compilation):
cd /mnt/c/Users/stevensilber/source/repos/data_processing/rust/vtk_viewer
cargo build --release --bin cell_analyze
scp target/release/cell_analyze ssilber@nibi.alliancecan.ca:~/bin/cell_analyze
```

### Running on the Cluster

**Study pipeline (preferred — TOML-driven, via MCP tools):**
```
1. sync_analysis(confirm=true)                  # build + deploy binary
2. sync_study_config(local_path="...fss.toml",  # upload TOML
                     remote_path="~/fss.toml")
3. run_analysis(data_dir="/scratch/.../fss_reps",  # submit SLURM job
                study_config="~/fss.toml")          # full resolution, no subsampling
4. download_results(remote_path="/.../analysis_results.json")  # get results
```

**Do NOT subsample when using `run_analysis`** — the SLURM job processes
full-resolution trajectories directly. The `--subsample` flag exists only
as a fallback for login-node usage.

**Key flags (login-node fallback only):**
- `--subsample N` — keep every Nth frame (only for login nodes)
- `--threads 1` — limit CPU usage on login nodes

### Available Observables (14 total)

| Name | What it computes | Key output fields |
|------|-----------------|-------------------|
| `msd` | Mean squared displacement MSD(Δt) | `lag_times[]`, `msd[]` |
| `diffusion` | D_eff from long-time MSD slope | `d_eff`, `r_squared` |
| `log_slope` | Instantaneous diffusion exponent Δ(t) | `lag_times[]`, `slopes[]` |
| `cage` | Cage length l_c from MSD plateau | `cage_length` |
| `alpha2` | Non-Gaussian parameter α₂(t) | `lag_times[]`, `alpha2[]` |
| `overlap` | Self-overlap Q(t), χ₄(t), τ_α, β | `lag_times[]`, `q_mean[]`, `chi4[]`, `tau_alpha`, `beta` |
| `structure` | Static structure factor S(q) + peak q* | `q_values[]`, `s_q[]`, `q_star` |
| `scattering` | Self-intermediate scattering F_s(q*, t) | `lag_times[]`, `fs[]`, `tau_alpha` |
| `van_hove` | van Hove self-correlation G_s(Δx, t) | `displacements[]`, `gs[lag_idx][]` |
| `per_cell_diffusion` | Per-cell D_i + CV | `d_values[]`, `mean`, `std`, `cv` |
| `displacement` | Mean/rms/max displacement | `mean_dr`, `rms_dr`, `max_dr` |
| `va_mobility_correlation` | Pearson r(v_A, speed) for σ>0 runs | `pearson_r`, `p_value` |
| `spatial_correlation` | C(r) mobility autocorrelation + ξ | `radii[]`, `c_r[]`, `xi` |
| `shape_index` | Shape index p = L_n × 2√π from perimeter | `mean_p`, `std_p`, `per_cell_p[]`, `p_vs_time[]`, `times[]` |

**Time series in JSON output:** Many observables return arrays (marked with `[]` above). These are full time series suitable for plotting. For example, `msd.lag_times` + `msd.msd` give you the full MSD curve; `shape_index.times` + `shape_index.p_vs_time` give the shape index evolution.

### Three-Step Cluster Workflow (Use MCP Tools)

**Step 1: Deploy binary to cluster (once after code changes)**
```
sync_analysis(cluster="nibi", confirm=true)
```
Builds in WSL, uploads single binary to `~/cell_simulation/bin/cell_analyze`.

**Step 2: Submit analysis SLURM job**
```
run_analysis(
  cluster="nibi",
  data_dir="/scratch/ssilber/adhesion_study/phase1_motility",
  pattern="Jk_*/run_*",
  observables="msd,overlap,diffusion",
  output="phase1_results.json"
)
```
Generates SLURM script, submits job. Returns job ID.

**Step 3: Monitor and download results**
```
check_progress(cluster="nibi")
# ... when COMPLETED:
download_results(remote_path="~/cell_sim_logs/phase1_results.json")
```
Downloads compact JSON (~10-100 KB). Raw data (GB) stays on cluster.

### Local Usage (for agents running analysis on local data)

The `cell_analyze` binary runs natively on Windows. No WSL needed for local analysis.

```powershell
# Build once (from repo root)
cd rust/vtk_viewer
cargo build --release --bin cell_analyze

# Single run — all observables (default)
.\\target\\release\\cell_analyze.exe run agent_test_runs\\my_sim -o results.json

# Specific observables only (faster)
.\\target\\release\\cell_analyze.exe run agent_test_runs\\my_sim --observables msd,diffusion,shape_index -o results.json

# Batch mode — analyze all runs matching a pattern
.\\target\\release\\cell_analyze.exe batch agent_test_runs\\my_study --pattern \"v*\" -o batch_results.json

# List all 14 available observables
.\\target\\release\\cell_analyze.exe list
```

**Important for agents:** The binary is at `rust/vtk_viewer/target/release/cell_analyze.exe` after building. The output JSON can be loaded directly in Python notebooks:

```python
import json
results = json.load(open('results.json'))

# Scalar values
d_eff = results['diffusion']['d_eff']
mean_p = results['shape_index']['mean_p']

# Time series (for plotting)
import numpy as np
lag = np.array(results['msd']['lag_times'])
msd = np.array(results['msd']['msd'])

# Per-cell arrays
per_cell_d = np.array(results['per_cell_diffusion']['d_values'])
per_cell_p = np.array(results['shape_index']['per_cell_p'])
```

**When to use which tool:**
- **Local data** → run `cell_analyze.exe` directly (no MCP needed)
- **Cluster data** → use `run_analysis` MCP tool (submits SLURM job, runs on cluster)
- **Cluster deploy** → use `sync_analysis` MCP tool (builds + uploads binary)

### JSON Output Format

Single-run output has this structure:
```json
{
  "params": { "v_a": 0.008, "n_cells": 288, "lx": 1562, "ly": 1562 },
  "msd": { "lag_times": [...], "values": [...] },
  "diffusion": { "d_eff": 1.23e-5, "fit_r2": 0.997 },
  "overlap": { "lag_times": [...], "q_mean": [...], "chi4": [...], "tau_alpha": 15234, "beta": 0.72 },
  "displacement": { "mean_dr": 3.2, "mean_dr_over_r": 0.065 }
}
```

Batch-mode output adds `groups` (keyed by directory name) with `mean ± stderr` aggregation, and a `summary` section for quick phase-diagram plotting.

### Source Code Location

```
rust/vtk_viewer/src/
├── cell_analyze.rs          # CLI binary (run, batch, list subcommands)
├── analysis/
│   ├── mod.rs               # Module root
│   ├── io.rs                # Trajectory loading + periodic unwrapping
│   ├── observables.rs       # All observable computations
│   ├── batch.rs             # Directory glob discovery + grouping
│   └── output.rs            # JSON output types + batch aggregation
```

### Plotting from JSON

Plotting stays in Python (local, trivial). Example:
```python
import json, matplotlib.pyplot as plt

with open("phase1_results.json") as f:
    data = json.load(f)

for key, mean, stderr in data["summary"]["d_eff_vs_group"]:
    plt.errorbar(float(key.replace("Jk_",""))/2, mean, yerr=stderr, fmt='ko', capsize=3)

plt.xlabel(r'$\tilde{J}$'); plt.ylabel(r'$D_\mathrm{eff}$')
plt.savefig("fig_phase_diagram.pdf")
```

### When to Use the Legacy C/Python Path Instead

- `msd_calculator.c`: Only if processing 1000+ runs where even Rust batch speed isn't enough (unlikely)
- Python `glass_observables.py`: Only for prototyping new observables before porting to Rust
- Python `analyze_adhesion.py`, `validate_bresler_cluster.py`: Superseded by `cell_analyze batch`

---

## Overview

When computing results from simulation data on the cluster:
1. **Understand the data format** - Know what you're parsing
2. **Develop locally first** - Iterate quickly on small samples
3. **Scale on cluster** - Submit via SLURM, never run on login nodes
4. **Validate incrementally** - Single run → parameter sweep → full dataset

## Data Formats Reference

> **Note:** Trajectory and VTK formats are also documented in [postprocessing.instructions.md](postprocessing.instructions.md). This section is the **authoritative reference for parsing details** (column types, header parsing, data quirks like duplicates and truncation). The postprocessing file documents the same formats from a visualization perspective.

### Trajectory File Format (`trajectory.txt`)
```
# Trajectory data for MSD computation
# Format: time cell_id x y vx vy px py theta
# v_A=0.004 N=288 Lx=1600 Ly=1600
80000.000000 0 1361.088 614.591 -0.000001 -0.000006 0.246579 -0.969123 4.961538
80000.000000 1 523.442 1203.887 0.000003 0.000001 -0.784631 0.619961 2.472787
...
```

**Header line** (starts with `# v_A=`):
| Field | Description |
|-------|-------------|
| `v_A` | Active velocity parameter |
| `N` | Number of cells |
| `Lx`, `Ly` | Domain dimensions (for periodic boundary unwrapping) |

**Data columns:**
| Column | Type | Description |
|--------|------|-------------|
| `time` | float | Simulation time |
| `cell_id` | int | Cell index (0 to N-1) |
| `x`, `y` | float | Cell centroid position |
| `vx`, `vy` | float | Cell velocity |
| `px`, `py` | float | Polarization direction (unit vector) |
| `theta` | float | Polarization angle (radians) |

**Trajectory size expectations (288 cells, production):**
- With `--trajectory-interval 18000`: ~2000 saves → ~576k lines → ~47 MB per run
- With `--trajectory-interval 100` (old default): ~360k saves → ~103M lines → ~8.9 GB per run
- Current production uses interval=18000. If processing older data, files may be much larger.

### Tracking File Format (`tracking.txt`)
Simpler format for basic position tracking:
```
# time x y cell_id
0.000000 512.0 512.0 0
0.100000 512.1 512.0 0
```

### VTK Field Format (`frame_*.vtk`)
Legacy VTK structured points:
```
# vtk DataFile Version 3.0
Cell simulation output
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS 512 512 1
ORIGIN 0 0 0
SPACING 1 1 1
POINT_DATA 262144
SCALARS phi float 1
LOOKUP_TABLE default
0.0 0.0 0.0 ...
```

### Directory Structure (Production Data)
```
/scratch/ssilber/jamming_study/production/
├── v0_r1/               # velocity group 0, replicate 1
│   └── trajectory.txt
├── v0_r2/
│   └── trajectory.txt
...
├── v9_r100/             # velocity group 9, replicate 100
│   └── trajectory.txt
```

Velocity groups map to v_A values (check trajectory headers for exact values).

---

## Development Workflow

### Phase 1: Local Development

#### Step 1.1: Download Sample Data

```powershell
# Download a single small trajectory file for testing
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "head -10000 /scratch/ssilber/jamming_study/production/v0_r1/trajectory.txt" > cpp/simulation/cluster/sample_trajectory.txt

# Or download full file if small
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" ssilber@nibi.alliancecan.ca:/scratch/ssilber/jamming_study/production/v0_r1/trajectory.txt cpp/simulation/cluster/test_traj.txt
```

#### Step 1.2: Create the Analysis Script

**Use C for performance** - Python is too slow for batch processing large files.

Template structure:
```c
/*
 * [Analysis Name] - Brief description
 * 
 * Compile: gcc -O3 -o analyzer analyzer.c -lm
 * Usage: ./analyzer <input_file> [options] <output_file>
 *        ./analyzer --batch <base_dir> [options] <output_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>

#define MAX_CELLS 512
#define MAX_TIMES 50000
#define LINE_BUF 1024

// Data structures
typedef struct {
    double time;
    double x[MAX_CELLS];
    double y[MAX_CELLS];
    // ... additional fields
    int n_cells;
} TimePoint;

typedef struct {
    TimePoint *points;
    int n_points;
    int capacity;
    double v_A;        // From header
    double Lx, Ly;     // Domain size
    int N;             // Cell count
} Trajectory;

// Parse header for parameters
int parse_header(const char *line, Trajectory *traj) {
    if (strstr(line, "v_A=")) {
        sscanf(strstr(line, "v_A="), "v_A=%lf N=%d Lx=%lf Ly=%lf",
               &traj->v_A, &traj->N, &traj->Lx, &traj->Ly);
        return 1;
    }
    return 0;
}

// Main analysis function
double compute_metric(Trajectory *traj) {
    // Your analysis here
    return 0.0;
}

// Single file processing
int process_single(const char *filename, const char *output) {
    // Load, compute, output
}

// Batch processing (all replicates)
int process_batch(const char *base_dir, const char *output) {
    // Loop over velocity groups and replicates
    for (int v = 0; v <= 9; v++) {
        for (int r = 1; r <= 100; r++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/v%d_r%d/trajectory.txt", base_dir, v, r);
            // Process and accumulate statistics
        }
    }
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);  // Disable buffering for progress visibility
    
    if (argc < 3) {
        printf("Usage: %s <input> <output>\n", argv[0]);
        printf("       %s --batch <dir> <output>\n", argv[0]);
        return 1;
    }
    
    if (strcmp(argv[1], "--batch") == 0) {
        return process_batch(argv[2], argv[3]);
    } else {
        return process_single(argv[1], argv[2]);
    }
}
```

#### Step 1.3: Compile and Test Locally

```powershell
# Compile with WSL gcc
cd C:\Users\stevensilber\source\repos\data_processing\cpp\simulation\cluster
wsl gcc -O3 -o analyzer analyzer.c -lm

# Test on sample data
wsl ./analyzer sample_trajectory.txt test_output.txt

# View output
Get-Content test_output.txt
```

#### Step 1.4: Visualize Results

**CRITICAL:** Always visualize results at every step. Numerical output alone is insufficient to catch bugs.

```powershell
# Create visualization for MSD data
python plot_msd.py msd_output.txt -o msd_plot.png

# Open to verify
Start-Process msd_plot.png
```

Example MSD visualization script (`plot_msd.py`):
```python
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_msd(filenames, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for fname in filenames:
        data = np.loadtxt(fname)
        lag, msd = data[:, 0], data[:, 1]
        label = fname.split('/')[-1].replace('.txt', '')
        
        # Linear plot
        axes[0].plot(lag, msd, '-', label=label)
        axes[0].set_xlabel('Lag Time')
        axes[0].set_ylabel('MSD')
        axes[0].set_title('MSD vs Time')
        axes[0].legend()
        
        # Log-log plot (reveals slope/diffusion exponent)
        axes[1].loglog(lag, msd, '-', label=label)
        axes[1].set_xlabel('Lag Time')
        axes[1].set_ylabel('MSD')
        axes[1].set_title('MSD vs Time (log-log)')
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved: {output}")

if __name__ == '__main__':
    plot_msd(sys.argv[1:-2], sys.argv[-1])
```

**What to check:**
- MSD should start near 0 and increase
- Linear MSD vs time indicates normal diffusion (slope = 2D)
- Negative slopes are WRONG - indicates data corruption
- Log-log slope should be ~1 for normal diffusion
- Compare jammed (v0) vs fluid (v9): fluid should have ~10x higher MSD

#### Step 1.5: Create Python Reference Implementation

For complex calculations, create a Python version to validate C output:
```python
# debug_msd.py - Reference implementation
import numpy as np

def compute_msd_python(trajectory_file):
    """
    Compute MSD using numpy for validation.
    Much slower than C, but easier to debug.
    """
    # ... implement same algorithm as C
    # Compare output to C version
```

This catches:
- Off-by-one errors
- Incorrect array indexing
- Boundary condition bugs

#### Step 1.6: Iterate Until Correct

- Check numerical output makes physical sense
- Compare against expected values or literature
- Test edge cases (empty files, single cell, etc.)
- **C and Python implementations should give same result (within floating point tolerance)**

---

### Phase 2: Single Simulation Test on Cluster
- Compare against expected values or literature
- Test edge cases (empty files, single cell, etc.)

---

### Phase 2: Single Simulation Test on Cluster

#### Step 2.1: Upload Code

```powershell
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" analyzer.c ssilber@nibi.alliancecan.ca:~/cell_simulation/cluster/
```

#### Step 2.2: Compile on Cluster (Login Node)

```powershell
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cd ~/cell_simulation/cluster && module load gcc/12.3 && gcc -O3 -o analyzer analyzer.c -lm"
```

**CRITICAL:** Always compile on login node BEFORE submitting job. SLURM jobs should use pre-compiled binaries.

#### Step 2.3: Create SLURM Job Script

Create `analyzer_job.sh`:
```bash
#!/bin/bash
#SBATCH --account=<your-account>   # Check with list_jobs or check_queues
#SBATCH --job-name=my_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

SCRIPT_DIR="$HOME/cell_simulation/cluster"
OUTPUT_DIR="$HOME/cell_sim_logs"

mkdir -p "$OUTPUT_DIR"

echo "Start: $(date)"

# Check binary exists
if [ ! -f "$SCRIPT_DIR/analyzer" ]; then
    echo "ERROR: analyzer not found! Compile first."
    exit 1
fi

# Run analysis
"$SCRIPT_DIR/analyzer" /scratch/ssilber/jamming_study/production/v0_r1/trajectory.txt "$OUTPUT_DIR/test_result.txt"

echo "Done: $(date)"
cat "$OUTPUT_DIR/test_result.txt"
```

#### Step 2.4: Submit Job (NOT Direct Execution!)

```powershell
# Upload job script
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" analyzer_job.sh ssilber@nibi.alliancecan.ca:~/cell_simulation/cluster/

# Submit via SLURM
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cd ~/cell_simulation/cluster && sbatch analyzer_job.sh"
```

#### Step 2.5: Monitor and Validate

```powershell
# Check job status
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "squeue -u ssilber"

# View output when done
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cat ~/cell_sim_logs/test_result.txt"
```

---

### Phase 3: Parameter Sweep

#### Step 3.1: Test Across Different Velocities

Modify job to process multiple velocity groups:
```bash
# In job script
for v in 0 5 9; do
    "$SCRIPT_DIR/analyzer" "/scratch/ssilber/jamming_study/production/v${v}_r1/trajectory.txt" "$OUTPUT_DIR/result_v${v}.txt"
done
```

#### Step 3.2: Validate Trend

Download and plot:
```powershell
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cat ~/cell_sim_logs/result_v*.txt"
```

Check if results show expected physical behavior (e.g., D increases with v_A for jamming transition).

---

### Phase 4: Full Dataset Processing

⚠️ **USER REVIEW CHECKPOINT** ⚠️

**Before running on the full dataset, STOP and ask the user to review:**

1. Show the results from Phase 2-3 (single file and parameter sweep)
2. Ask: "Do these results look physically reasonable?"
3. Present key validation metrics:
   - Expected vs actual values for known cases
   - Physical sanity checks (e.g., D_fluid > D_jammed)
   - Visualization of trends across parameters

**Do NOT proceed to full batch processing until user confirms results are correct.**

This prevents wasting hours of cluster time on buggy code.

---

#### Step 4.1: Update Script for Batch Mode

Add `--batch` flag that processes all replicates:
- Loop over velocity groups (v0-v9)
- Loop over replicates (r1-r100)
- Compute mean ± stderr for each velocity
- Output aggregated results

#### Step 4.2: Update Job Script

Increase resources if needed:
```bash
#SBATCH --time=04:00:00  # More time for full dataset
#SBATCH --mem=16G        # More memory if needed
```

Run batch mode:
```bash
"$SCRIPT_DIR/analyzer" --batch /scratch/ssilber/jamming_study/production "$OUTPUT_DIR/full_results.txt"
```

#### Step 4.3: Final Analysis and Plotting

```powershell
# Download results
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" ssilber@nibi.alliancecan.ca:~/cell_sim_logs/full_results.txt cpp/simulation/cluster/

# Plot
python plot_jamming_transition.py cluster/full_results.txt -o jamming_transition_final.png
```

---

## Data Quality Issues

### Known Issues with Trajectory Files

**1. Duplicate Rows Per Time Step**

Production trajectory files often have multiple rows per cell per time step (simulation outputs at sub-time intervals). Your parser **must handle duplicates**:

```c
// Track which cells we've seen at current time
int cells_seen[MAX_CELLS];
memset(cells_seen, 0, sizeof(cells_seen));

// When reading each row
if (cells_seen[cell_id]) {
    continue;  // Skip duplicate
}
cells_seen[cell_id] = 1;
```

**2. Incomplete Last Time Points**

Trajectory files may be truncated, leaving the last time point with fewer cells than expected. **Always validate**:

```c
// After reading, check last time point
int expected = traj->N;  // From header
int actual = count_cells_at_last_time(traj);
if (actual < expected) {
    printf("Warning: Last time point has %d/%d cells, discarding\n", actual, expected);
    traj->n_points--;  // Remove incomplete time point
}
```

This is critical for MSD calculations - incomplete time points leave positions at (0,0), causing massive spurious displacements.

**3. Zero-Initialize Position Arrays**

Always zero arrays when moving to a new time point:
```c
memset(cells_seen, 0, sizeof(cells_seen));
memset(point->x, 0, sizeof(point->x));
memset(point->y, 0, sizeof(point->y));
```

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Binary not found in job | Didn't pre-compile | Compile on login node before sbatch |
| Job runs forever | Inefficient code or huge dataset | Add progress output, optimize algorithm |
| NaN in results | Division by zero, bad input | Add input validation, check empty files |
| Wrong results | Logic error | Test locally with known data first |
| Permission denied | File not executable | `chmod +x analyzer` |
| Out of memory | Large arrays | Reduce MAX_CELLS/MAX_TIMES or stream processing |
| Results differ local vs cluster | Different data sample | Use same input file |
| Negative D values | Incomplete time points or uninitialized arrays | Handle duplicates, zero-init arrays |
| MSD values huge (~1000s) | Spurious displacements from bad data | Detect/remove incomplete time points |

### Debugging on Cluster

```powershell
# Check job error output
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cat ~/cell_sim_logs/my_analysis_*.err"

# Check job stdout
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cat ~/cell_sim_logs/my_analysis_*.out"
```

### Path Resolution Issues

In SLURM jobs, `BASH_SOURCE` and relative paths may not work. **Always use absolute paths:**
```bash
SCRIPT_DIR="$HOME/cell_simulation/cluster"  # Good
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"  # BAD in SLURM
```

---

## Checklist for New Analysis

- [ ] **Understand data format**: Know columns, headers, directory structure
- [ ] **Download sample**: Get representative test data locally
- [ ] **Write C code**: Use template, compile with `-O3 -lm`
- [ ] **Handle data quirks**: Duplicates, incomplete time points, zero-init arrays
- [ ] **Test locally**: Verify output makes sense
- [ ] **Create/update plotter**: Ensure visualization works
- [ ] **View results**: Check both numbers AND plots
- [ ] **Create Python reference**: For complex algorithms, validate C vs Python
- [ ] **Upload to cluster**: Binary + job script
- [ ] **Pre-compile**: `module load gcc/12.3 && gcc -O3 ...`
- [ ] **Single-file test**: Submit SLURM job for one file
- [ ] **Validate**: Compare cluster output to local
- [ ] **Parameter sweep**: Test across different simulation parameters
- [ ] **⚠️ USER REVIEW**: Ask user to confirm results before full batch
- [ ] **Full batch**: Process complete dataset
- [ ] **Final plots**: Generate publication-ready figures

---

## Example: MSD/Diffusion Analysis

The `msd_calculator.c` is a reference implementation following this workflow:

### Code Structure
- Parses trajectory header for `v_A`, `N`, `Lx`, `Ly`
- Unwraps periodic boundaries for MSD calculation
- Computes diffusion coefficient D from long-time MSD slope
- Batch mode aggregates over replicates with error estimation

### Local Testing
```powershell
# Download samples
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "head -5000 /scratch/ssilber/jamming_study/production/v0_r1/trajectory.txt" > cluster/test_traj_v0.txt
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "head -5000 /scratch/ssilber/jamming_study/production/v9_r1/trajectory.txt" > cluster/test_traj_v9.txt

# Compile and test
wsl gcc -O3 -o cluster/msd_calculator cluster/msd_calculator.c -lm
wsl ./cluster/msd_calculator cluster/test_traj_v0.txt 5000 cluster/msd_v0.txt
wsl ./cluster/msd_calculator cluster/test_traj_v9.txt 5000 cluster/msd_v9.txt

# Check D values - v0 should be ~0 (jammed), v9 should be positive
```

### Cluster Execution
```powershell
# Pre-compile on cluster
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cd ~/cell_simulation/cluster && module load gcc/12.3 && gcc -O3 -o msd_calculator msd_calculator.c -lm"

# Submit job
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cd ~/cell_simulation/cluster && sbatch msd_job.sh"

# Results go to ~/cell_sim_logs/diffusion_results.txt
```

### Visualization
```powershell
# Download and plot
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" ssilber@nibi.alliancecan.ca:~/cell_sim_logs/diffusion_results.txt cpp/simulation/cluster/
python cpp/simulation/plot_jamming_transition.py cpp/simulation/cluster/diffusion_results.txt -o jamming_transition.png
```

---

## Lessons Learned from MSD Development

### Process Lessons
1. **BASH_SOURCE fails in SLURM** - Use hardcoded `$HOME/...` paths
2. **Compile on login node** - Jobs should only run pre-compiled binaries
3. **Add progress output** - `setbuf(stdout, NULL)` for unbuffered progress
4. **Test locally first** - Catch bugs before wasting cluster time
5. **Validate incrementally** - Single file → few files → all files
6. **Physical sanity checks** - D < 0 means jammed, D > 0 means unjammed
7. **Error handling** - Check file existence before processing
8. **Memory limits** - Use `#define MAX_*` constants that fit in SLURM allocation

### Data Quality Lessons (Critical!)
9. **Production data has duplicates** - Multiple rows per cell per time step
10. **Files may be truncated** - Last time point often incomplete
11. **Zero-initialize arrays** - Garbage data causes spurious results
12. **Visualize at every step** - Numerical output alone misses bugs
13. **Create reference implementation** - Python version validates C code
14. **Compare across parameters** - v9 should differ from v0 in predictable ways

### Validation Criteria for MSD
- MSD values should be small (10⁻⁵ to 10⁻² range for our simulations)
- MSD should increase monotonically with lag time
- D_fluid > D_jammed (typically ~10x difference)
- Negative D values indicate bugs, not physical jamming
- Log-log slope of MSD vs time should be ~1 for normal diffusion

---

## File Organization

```
cpp/simulation/cluster/
├── msd_calculator.c       # Analysis source code
├── msd_calculator         # Pre-compiled binary (on cluster)
├── msd_job.sh             # SLURM job script
├── sample_trajectory.txt  # Test data (local)
├── test_*.txt             # Local test outputs
└── ...

~/cell_sim_logs/           # On cluster
├── diffusion_results.txt  # Analysis output
├── *.out, *.err           # Job logs
└── ...
```

````