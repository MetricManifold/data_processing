---
applyTo: "cpp/simulation/**"
---

# Cell Simulation Project Instructions

> **When to consult this file:** You are building, running, testing, or modifying the cell simulation code (CUDA). This covers local builds, CLI reference, physics parameters, and validation tests. For cluster job submission, see [cluster-operations.instructions.md](cluster-operations.instructions.md). For analyzing simulation output and rendering snapshots, use the `cell_analyze` Rust binary (`cell_analyze list` / `cell_analyze --help`).

## Key Project Locations

### CUDA Version (Primary)
- **Source code**: `cpp/simulation/src/` and `cpp/simulation/include/`
- **Build directory**: `cpp/simulation/build/`
- **Executable**: `cpp/simulation/build/bin/cell_sim.exe` (Windows) or `~/cell_simulation/build/bin/cell_sim` (cluster)
- **Test output**: `cpp/simulation/agent_test_runs/`

### Shared Resources
- **Visualization & analysis**: `cell_analyze` (Rust binary in `rust/cell_analyze/`); 3D-only Python helpers live in `cpp/simulation/postprocessing/`
- **Cluster scripts**: `cpp/simulation/cluster/`

## Build Commands

### Prerequisites
- **CUDA Toolkit** (v12+ recommended, v13 tested)
- **Visual Studio** with Desktop C++ workload (VS 2022 v18 tested)
- **CMake** 3.18+

The build uses Ninja (bundled with VS) and MSVC. Since `cl.exe` and `ninja` are
not on PATH in a normal PowerShell, every build command must run inside the
**vcvars64** environment. The one-liner below does this automatically:

```powershell
# Helper: build from any PowerShell prompt (Release)
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$installPath = (& $vsWhere -latest -property installationPath)
cmd /c "`"$installPath\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && cd /d c:\Users\stevensilber\source\repos\data_processing\cpp\simulation\build && cmake --build . --config Release 2>&1"
```

### First-Time Setup (configure + build)
```powershell
cd c:\Users\stevensilber\source\repos\data_processing\cpp\simulation
mkdir build -Force; cd build

# Run inside a VS Developer PowerShell / Command Prompt, OR wrap with vcvars64:
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$installPath = (& $vsWhere -latest -property installationPath)
cmd /c "`"$installPath\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && cd /d c:\Users\stevensilber\source\repos\data_processing\cpp\simulation\build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release 2>&1"
```

### Incremental Rebuild (after code changes)
```powershell
# Only recompiles changed files — fast (~10-30s depending on which .cu changed)
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$installPath = (& $vsWhere -latest -property installationPath)
cmd /c "`"$installPath\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && cd /d c:\Users\stevensilber\source\repos\data_processing\cpp\simulation\build && cmake --build . --config Release 2>&1"
```

> **IMPORTANT**: If the linker fails with `LNK1104: cannot open file 'bin\cell_sim.exe'`,
> a running `cell_sim.exe` process is locking the file. Stop it first:
> `Get-Process cell_sim -ErrorAction SilentlyContinue | Stop-Process -Force`

### Standard Release Build
The following commands assume you are already inside a vcvars64 environment
(Developer PowerShell, or using the `cmd /c` wrapper above):
```powershell
cd c:\Users\stevensilber\source\repos\data_processing\cpp\simulation\build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Safe Mode (for checkpoint loading or memory debugging)
```powershell
cmake .. -DCMAKE_BUILD_TYPE=Release -DSAFE_MODE=ON
cmake --build . --config Release
```

### With Stress Fields Enabled
```powershell
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_STRESS_FIELDS=ON
cmake --build . --config Release
```

### With Diagnostics Enabled
```powershell
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_DIAGNOSTICS=ON
cmake --build . --config Release
```

### Debug Build
```powershell
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
```

### Build Option Reference
| CMake Flag | Purpose |
|------------|---------|
| `-DSAFE_MODE=ON` | Enable GPU memory tracking with 1GB limit |
| `-DENABLE_DIAGNOSTICS=ON` | Enable GPU-side diagnostics (energy, stress, contacts) |
| `-DENABLE_STRESS_FIELDS=ON` | Enable stress tensor field output in VTK files |

---

## Running Simulations (CUDA)

### 2D Simulations

```powershell
# Single cell test
.\build\bin\Release\cell_sim.exe -n 1 -N 256 -r 49 -t 10 --dt 0.01 -o agent_test_runs/test_2d_single

# Multi-cell test
.\build\bin\Release\cell_sim.exe -n 8 -N 512 -r 49 -t 10 --dt 0.01 -o agent_test_runs/test_2d_multi

# With motility
.\build\bin\Release\cell_sim.exe -n 8 -N 512 -r 49 -t 100 --dt 0.01 --v-A 0.01 --tau 100 --trajectory-samples 200 -o agent_test_runs/test_2d_motile
```
### Long-Running Motility Simulations

For motility studies, use `--print-interval` to monitor progress independently of VTK saves:

**⚠️ NEVER hardcode domain size `-N`.** Always compute from the target confluence:

$$L = \lceil\sqrt{N \pi R^2 / \rho}\rceil$$

When using the MCP tools, pass `--confluence` and the tool computes $L$ automatically. When running locally, compute $L$ in the command:

```powershell
# Compute domain size for target confluence, then run
$n=72; $r=49; $rho=0.90; $L=[math]::Ceiling([math]::Sqrt($n * [math]::PI * $r*$r / $rho))
Write-Host "Domain size L=$L for n=$n r=$r rho=$rho"

# Equilibration (v_A=0)
.\build\bin\Release\cell_sim.exe -n $n -N $L -r $r -t 8000 --dt 0.01 --v-A 0 --print-interval 10000 --save-interval 100000 -o agent_test_runs/equilibration_72

# Motility run
.\build\bin\Release\cell_sim.exe -n $n -N $L -r $r -t 8000 --dt 0.01 --v-A 0.004 --tau 10000 --print-interval 10000 --save-interval 100000 -o agent_test_runs/motility_vA0.004
```

**CRITICAL**: Always use `--print-interval` for long runs!
- Without it, progress only prints at VTK save points
- With `--save-interval 100000`, first output is after ~3.5 minutes
- This makes simulations appear to "hang" or "crash" when they're actually running fine
- Recommended: `--print-interval 10000` for updates every ~20 seconds

**Typical timing** (dt=0.01):

| Cells | Domain | GPU | ~t/s | Time Units/Hour | Per 3h Job | Notes |
|------:|-------:|-----|-----:|----------------:|-----------:|-------|
| 288 | 1600² | RTX 4090 Laptop | ~4.9 | 17,484 | ~52,000 | Local dev, older binary (measured) |
| 288 | 1562² | H100 MIG 1g.10gb | ~10 | 36,000 | ~108,000 | Cluster, Feb 2025 (measured) |
| 288 | 1562² | H100 Full | ~59 | 212,400 | ~637,000 | When `gres/gpu:1` lands on full-GPU node (measured) |
| 1152 | 3124² | H100 Full | 17.3 | 62,280 | ~187,000 | Cluster (measured) |
| 4608 | 6249² | H100 Full | 4.7 | 16,920 | ~203,000¹ | Cluster, steady state (measured) |
| 4608 | 6249² | H100 MIG 1g.10gb | ~0.8² | ~2,900 | ~8,600 | **Extrapolated** from SM ratio; impractical |
| 4608 | 6400² | RTX 4090 (16 GB) | — | — | — | OOM locally |

¹ 4608c uses a longer walltime (auto-selected from calibration), giving ~203k per job.
² Extrapolated (MIG/Full ratio ≈ 1:6 from 288c measurements). Never measured.

**VRAM usage (measured values — re-measure after significant code changes):**

| Cells | Domain | Phi Pool | Peak VRAM Used | Fits in | Provenance |
|------:|-------:|----------|---------------:|--------:|------------|
| 288 | 1562² | 101 MB | ~170 MB | Any MIG (10 GB) | Measured |
| 1152 | 3124² | 440 MB | ~790 MB | Any MIG (10 GB) | Measured |
| 4608 | 6249² | 2,152 MB | ~3,660 MB | MIG 1g.10gb (10 GB) | Measured |
| 18432 | 12496² | ~8.6 GB | ~14 GB | MIG 2g.20gb (20 GB) | Extrapolated |
| 73728 | 24992² | ~34 GB | ~58 GB | Full H100 (80 GB) only | Extrapolated |

³ Extrapolated linearly from 4608c measurements. Never measured.

**Full GPU is needed for COMPUTE, not memory.** Even 4608c fits in a 10 GB MIG slice,
but runs ~6× slower due to fewer SMs. Request full H100 only when the MIG
compute rate makes chaining impractical (>~20 chains).

> **Chain calculation details**: The MCP `start_simulation` / `resume_simulation` tools auto-calculate chain counts from these rates. See [cluster-operations.instructions.md → Performance & Chain Calculations](cluster-operations.instructions.md).

⚠️ **GPU selection and chaining are handled automatically by the MCP tools** (`start_simulation` / `resume_simulation`). The tools select the appropriate GPU type and walltime partition per cluster and cell count. Do not hardcode GPU GRES values — they vary by cluster (see `tools/compute_canada_mcp/DESIGN.md` for the authoritative reference). Use `estimate_cost` to preview before submitting.

**IMPORTANT 2D Notes:**
- Use `--dt 0.01` for 2D simulations (same as 3D; stable for R=49)
- Use radius `-r 49` for standard tests (matches paper parameters)
- CLI default radius is 20, but always specify `-r 49` explicitly

### 3D Simulations

```powershell
# Single cell test
.\build\bin\Release\cell_sim.exe --3d -n 1 -N 240 -r 49 -t 10 --dt 0.01 --checkpoint-interval 500 -o agent_test_runs/test_3d_single

# Multi-cell (16 cells at 85% confluence)
# Domain calculation: N = cbrt(n_cells * cell_volume / target_confluence)
# For r=49: V = (4/3)πr³ ≈ 492807, so N ≈ cbrt(16 * 492807 / 0.85) ≈ 210
.\build\bin\Release\cell_sim.exe --3d -n 16 -N 210 -r 49 -t 10 --dt 0.01 --checkpoint-interval 500 -o agent_test_runs/test_3d_multi
```

**IMPORTANT 3D Notes:**
- Use `--dt 0.01` for 3D simulations
- Use random initialization (default). Use `--confluence` to auto-compute domain size
- Larger `--checkpoint-interval` (500+) recommended to reduce I/O

### Resume from Checkpoint
```powershell
.\build\bin\Release\cell_sim.exe -c agent_test_runs/test_checkpoint/checkpoint.bin -t 100 -o agent_test_runs/resumed_run
```

#### v_A Regeneration on Resume

**IMPORTANT:** v4 checkpoints store per-cell v_A values. When resuming from a checkpoint, the integrator restores these stored values by default. This means:

- Resuming an **equilibration checkpoint** (where all v_A=0) with `--v-A 0.008` would silently keep all cells at v_A=0 if the checkpoint's per-cell values aren't cleared.
- To fix this, specifying `--v-A` or `--v-A-sigma` on the command line when resuming automatically **clears the checkpoint's per-cell v_A** and regenerates fresh values from the command-line parameters.

```powershell
# Start production from equilibration checkpoint — v_A is regenerated correctly
.\build\bin\Release\cell_sim.exe -c eq/checkpoint.bin -t 40000 --v-A 0.008 -o production_run

# With per-cell disorder (Griffiths study)
.\build\bin\Release\cell_sim.exe -c eq/checkpoint.bin -t 40000 --v-A 0.008 --v-A-sigma 0.006 -o griffiths_run
```

The console will print `Per-cell v_A will be regenerated (--v-A or --v-A-sigma specified)` to confirm.

**Priority order for per-cell v_A initialization:**
1. Checkpoint values (if present AND `--v-A`/`--v-A-sigma` NOT specified on command line)
2. Log-normal distribution around `--v-A` with std dev `--v-A-sigma` (if `--v-A-sigma > 0`)
3. Uniform value from `--v-A` (if no sigma)

**Note:** v3 (older) checkpoints do not store per-cell v_A, so this issue only affects v4 checkpoints. Cluster 288-cell equilibration checkpoints created before the v4 format are unaffected.

## CLI Options Reference

**Run `cell_sim -h` for the full CLI reference with current defaults.** The binary prints all available options, defaults from the actual code, and parameter set examples. Do not hardcode CLI flags or defaults from this document — always use `-h` as the source of truth.

Key options (see `-h` for the complete list and defaults):
- Geometry: `-n`, `-r`/`--radius`, `-N` OR `--confluence` (mutually exclusive)
- Time: `-t`, `-dt`
- Physics: `--v-A`, `--v-A-sigma`, `--gamma`, `--kappa`, `--mu`, `--xi`, `--adhesion`, `--tau`, `--abp`
- Bbox: `--subdomain-padding` (buffer beyond cell extent, in units of R; default 0.6)
- RNG: `--seed` (cell placement), `--polarity-seed` (velocity/polarity GPU RNG; default: random)
- I/O: `--save-interval`, `--trajectory-samples` OR `--trajectory-interval` (mutually exclusive), `--checkpoint-interval`, `--save-final-checkpoint`
- Resume: `-c` (checkpoint file)

**⚠️ Checkpoint resume behavior:** On resume from checkpoint (`-c`), physics parameters (dt, gamma, kappa, mu, xi, lambda, tau, etc.) are **preserved from the checkpoint** unless explicitly overridden on the CLI. Only parameters with corresponding `--flag` actually specified on the command line will override the checkpoint values. This prevents accidental physics changes from binary default drift.

**⚠️ Production I/O guidance:** For long cluster runs (t~880,000), use `--save-interval 0` (no VTK) and `--trajectory-interval 18000` (~2000 data points over 720k production time, ~47 MB/run). The default interval of 100 steps produces ~8.9 GB/run and will exceed scratch quota. See [cluster-operations.instructions.md](cluster-operations.instructions.md) for the full data budget calculation.

**⚠️ Equilibration I/O guidance:** For equilibration runs (`--v-A 0`), save only ~10 VTK frames and disable trajectory/tracking — these aren't needed. The MCP `start_simulation` tool with equilibration settings handles this automatically. Checkpoints must be frequent enough (≥3 per chain job) to survive mid-job crashes.

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
.\build\bin\Release\cell_sim.exe --3d -n 1 -N 240 -r 49 -t 10 --dt 0.01 --checkpoint-interval 500 -o agent_test_runs/validate_3d_single
# Expected: volume ~492807, phi_max ~1.0, no NaN

# Test 4: 3D multi-cell (16 cells, 85% confluence)
.\build\bin\Release\cell_sim.exe --3d -n 16 -N 210 -r 49 -t 10 --dt 0.01 --checkpoint-interval 500 -o agent_test_runs/validate_3d_multi
# Expected: cells interact and repel, no collapse
```

## Visualization & Analysis

All 2D analysis and snapshot rendering goes through the `cell_analyze` binary (`rust/cell_analyze/`). It is TOML-driven for studies and has dedicated subcommands for one-off snapshots, movies, and trajectory integrity checks. Discover everything via:

```powershell
cell_analyze --help          # subcommands: study, snapshot, check, list
cell_analyze list            # all observables, aggregators, panel types, templates
cell_analyze study --help    # TOML pipeline (figures + raw study_results.json)
cell_analyze snapshot --help # PNGs/movies from checkpoints or VTK frames
```

Reference study TOMLs live in `cpp/simulation/study/templates/`. 3D output (which `cell_analyze` does not yet handle) uses the Python helpers in `cpp/simulation/postprocessing/`; see that directory's README.

## Key Physical Parameters

The phase field model has the following physics parameters. The **binary defaults correspond to the Palmieri et al. (2015) parameter set** — run `cell_sim -h` for current default values.

| Parameter | Symbol | Physical meaning |
|-----------|--------|------------------|
| Gradient energy | $\gamma$ | Interfacial stiffness / surface tension |
| Repulsion | $\kappa$ | Cell-cell repulsion strength |
| Volume constraint | $\mu$ | Resistance to area/volume changes |
| Friction | $\xi$ | Dissipation rate; sets velocity scale via $v_I \propto \kappa/\xi$ |
| Interface width | $\lambda$ | Controls cell boundary sharpness (not a CLI flag) |
| Cell radius | $R$ | Target area $A_0 = \pi R^2$ (2D) or volume $V_0 = \frac{4}{3}\pi R^3$ (3D) |
| Adhesion | $J$ | Cell-cell adhesion (gradient coupling); 0 = disabled |

### Parameter Sets Used in Production

Different studies use different calibrations. **Each study-specific instruction file specifies which set to use.**

- **Palmieri (2015):** The binary defaults. No parameter overrides needed. Run `cell_sim -h` to see current values.
- **Bresler (2018):** Requires overrides: `gamma=3.75`, `mu=0.5`, `xi=1000`. All other parameters use binary defaults.

The Bresler calibration was developed for the sharp-interface limit analysis. The two sets produce **different physics** (different length/time/energy scales) and are not interchangeable. When in doubt, check the study-specific instructions.

These are defined in `include/types.cuh` (2D) and `include/types3d.cuh` (3D).

## Development Workflow

### 3D Testing Requirements

**IMPORTANT: All 3D test simulations must use 85% confluence.**

Domain size $L$ for 3D must be computed from the target confluence $\rho$:
$$L = \lceil\sqrt[3]{n_{\text{cells}} \times \frac{4}{3}\pi R^3 / \rho}\rceil$$

Never hardcode $L$ — always compute at run time. Example:
```powershell
$n=64; $r=10; $rho=0.85
$V=[math]::PI * 4.0/3.0 * [math]::Pow($r, 3)
$L=[math]::Ceiling([math]::Pow($n * $V / $rho, 1.0/3.0))
Write-Host "3D domain L=$L for n=$n r=$r rho=$rho"
.\build\bin\Release\cell_sim.exe --3d -n $n -N $L -r $r -t 100 --dt 0.01 --seed 12345 -o agent_test_runs/test_3d
```

### Adding a Feature
1. Build in Release mode
2. Run all 4 validation tests
3. Run your specific test case
4. Visualize results to confirm correctness (use `--volume` for 3D)
5. Build in Debug mode if issues found

### Fixing a Bug
1. Create a minimal reproduction case
2. Build with SAFE_MODE=ON if memory-related
3. Fix the issue
4. Run validation test suite
5. Verify reproduction case is fixed

### Optimizing Performance
1. Run baseline: `.\build\bin\Release\cell_sim.exe -n 16 -N 512 -t 100 --save-interval 0 -o agent_test_runs/baseline`
2. Note runtime
3. Make changes
4. Run same benchmark, compare
5. Run validation test suite

### Reproducibility Testing for Optimization Work

**CRITICAL: When optimizing GPU kernels, establish a deterministic baseline first.**

Use `--seed 42` (or any fixed seed) to make runs reproducible:

```powershell
# Standard 3D optimization baseline (32 cells, fixed seed)
.\build\bin\Release\cell_sim.exe --3d -n 32 -N 200 -r 36 -t 5 --dt 0.01 --seed 42 --v-A 0 --save-interval 0 -o agent_test_runs/optim_baseline

# Record the baseline metrics:
# - Total wall time
# - Final cell positions (if sync_centroids enabled)
# - Per-phase timing (if ENABLE_KERNEL_PROFILING build)
```

**After optimization:**
```powershell
# Run same test with optimized code
.\build\bin\Release\cell_sim.exe --3d -n 32 -N 200 -r 36 -t 5 --dt 0.01 --seed 42 --v-A 0 --save-interval 0 -o agent_test_runs/optim_test

# Verify correctness (outputs should match baseline):
# - Cell trajectories/positions must be identical (or within FP tolerance)
# - Volume conservation must be preserved
# - No NaN or instabilities introduced
```

**Kernel profiling build:**
```powershell
cmake .. -DENABLE_KERNEL_PROFILING=ON
cmake --build . --config Release

# Profile run shows per-phase timing breakdown
.\build\bin\Release\cell_sim.exe --3d -n 32 -N 200 -r 36 -t 5 --dt 0.01 --seed 42 --save-interval 0 -o agent_test_runs/profile_test
```

**Validation criteria for optimizations:**
1. Runtime must decrease (or at least not increase significantly)
2. Final cell positions must match baseline (same seed → same result)
3. No new errors in `get_errors` output
4. Validation test suite still passes

## Output Directory Convention

**All test output goes to:** `agent_test_runs/`

Use descriptive names:
```powershell
-o agent_test_runs/feature_name_test
-o agent_test_runs/bugfix_validation  
-o agent_test_runs/perf_benchmark_n64
```

## Cluster Operations

See [cluster-operations.instructions.md](cluster-operations.instructions.md) for comprehensive cluster documentation including:
- SSH persistent connection setup (avoiding MFA)
- Building on cluster (modules, cmake)
- Job submission (presets, SLURM)
- Submission log tracking
- Status monitoring
- File transfer

### Wall Time and Job Chaining

The MCP `start_simulation` / `resume_simulation` tools automatically compute walltime, chain count, and GPU resources from **per-cluster calibration data** (`~/cell_sim_calibration.json`). Do not manually specify walltime, partition, account, or chain count — the tools probe the live scheduler state and benchmark data to make optimal choices.

For long simulations, the tools handle **job chaining automatically**:
- Each job runs for the calibration-computed walltime and saves a checkpoint
- The next job depends on the previous via `--dependency=afterany:PREV_JOB_ID`
- Chain count is auto-calculated from calibration rate × remaining simulation time

Use `estimate_cost` to preview walltime and chain counts before submitting.
Use `benchmark_cluster(action='submit')` then `benchmark_cluster(action='collect')` to calibrate a new cluster.

**Submitting 3D jobs with auto-chaining (use `start_simulation` MCP tool):**

- **100-cell equilibration** (auto-chains ~4 jobs): cluster=nibi, cells=100, t_end=60000, three_d=true
- **400-cell equilibration** (auto-chains ~8 jobs): cluster=nibi, cells=400, t_end=60000, three_d=true
- **Preview**: use `estimate_cost` MCP tool to preview before submitting

The script auto-calculates domain size for 85% confluence and chains jobs appropriately.

## Code Architecture Overview

### Source Files (`src/`)
| File | Purpose |
|------|---------|
| `main.cu` | Entry point, CLI parsing, simulation orchestration |
| `kernels_shared.cu` | Shared memory CUDA kernels for 2D |
| `kernels_solver.cu` | Time integration kernels |
| `kernels3d.cu` | 3D-specific CUDA kernels |
| `integrator.cu` | 2D time stepping logic |
| `io.cu` | 2D VTK output, checkpoints |
| `io3d.cu` | 3D VTK output, checkpoints |

### Header Files (`include/`)
| File | Purpose |
|------|---------|
| `types.cuh` | 2D parameter structs, `SimParams` |
| `types3d.cuh` | 3D parameter structs, `SimParams3D` |
| `physics.cuh` | Physics device functions (bulk term, repulsion, advection) |
| `cell.cuh` / `cell3d.cuh` | Cell data structures |
| `domain.cuh` / `domain3d.cuh` | Domain management |
| `kernels.cuh` / `kernels3d.cuh` | Kernel declarations |
| `simulation.cuh` / `simulation3d.cuh` | High-level simulation interface |

### Physics Equations
The simulation solves:
```
dφ/dt = -v·∇φ - 0.5 * δF/δφ
```
Where the functional derivative `δF/δφ` is:
```
δF/δφ = -2γ∇²φ + γ(60/λ²)φ(1-φ)(1-2φ) + volume_constraint + repulsion + adhesion
```
Note: γ multiplies the **entire elastic bracket** (both gradient energy and double-well bulk potential), per Palmieri Eq. (7) / Eq. (S15). This is the convention where γ controls overall cell stiffness.

Term details:
- Laplacian: `−2γ∇²φ` (surface tension / interface restoring)
- Bulk: `γ(60/λ²)φ(1-φ)(1-2φ)` (double-well enforcing φ ∈ [0,1])
- Volume constraint: `−4(μ/V₀)(V₀ − V)φ`
- Repulsion: `(60κ/λ²) × φᵢ × Σⱼφⱼ²`
- Adhesion: `−J × Σⱼ∇²φⱼ` (only when `--adhesion J` > 0; zero overhead when disabled)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Simulation appears to hang after "Starting simulation" | Add `--print-interval 10000` to see progress; without it, output only at VTK saves |
| NaN in output | Reduce dt by 2x, check bounding box updates |
| Volume drift | Check μ parameter, verify volume constraint term |
| OOM on checkpoint load | Build with `-DSAFE_MODE=ON` |
| Cells disappear | Check bounding box tracking, increase subdomain-padding |
| Build fails | Ensure CUDA 11+, CMake 3.18+, C++17 compiler |
| Cells overlap excessively | Increase κ (repulsion), check initialization spacing |
| Slow performance | Use `--save-interval 0` during benchmarks |

## Additional Documentation

- [AGENT_ONBOARDING.md](cpp/simulation/AGENT_ONBOARDING.md) - Full onboarding and physics background
- [RUNBOOK.md](cpp/simulation/RUNBOOK.md) - Detailed operational runbook

**Note**: For physics details and implementation specifics, refer to inline documentation in `include/physics.cuh`. The code comments are authoritative for implementation details.

---

## Related Instruction Files

**⚠️ ALWAYS consult the relevant instruction file before performing operations:**

| Task | Instruction File |
|------|-----------------|
| Running jobs on cluster (Nibi) | [cluster-operations.instructions.md](cluster-operations.instructions.md) |
| Production runs with job chaining | [cluster-operations.instructions.md](cluster-operations.instructions.md) - see "Jamming Study Production" |
| Analyzing simulation output | `cell_analyze --help` and `cell_analyze list` (no instruction file — the CLI is self-documenting) |
| VTK viewer (Rust GUI) | [vtk-viewer.instructions.md](vtk-viewer.instructions.md) |
**Quick-reference READMEs:**
- [cluster/README.md](cpp/simulation/cluster/README.md) - Cluster submission quick start
- [postprocessing/README.md](cpp/simulation/postprocessing/README.md) - 3D Python helpers
**Critical reminders:**
- For cluster production runs, use `start_simulation` or `resume_simulation` MCP tools (auto-chains jobs)
- Never run compute on cluster login nodes - always submit via SLURM
- Test analysis code locally before running on cluster
