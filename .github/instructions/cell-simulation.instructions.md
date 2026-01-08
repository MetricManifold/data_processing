---
applyTo: "cpp/simulation/**"
---

# Cell Simulation Project Instructions

This is a CUDA-based cell simulation project. The simulation models phase field cells in 2D and 3D using GPU computation.

## Key Project Locations

- **Simulation source code**: `cpp/simulation/src/` and `cpp/simulation/include/`
- **Build directory**: `cpp/simulation/build/`
- **Executable**: `cpp/simulation/build/bin/Release/cell_sim.exe`
- **Test output**: `cpp/simulation/agent_test_runs/`
- **Visualization scripts**: `cpp/simulation/*.py`
- **Cluster scripts**: `cpp/simulation/cluster/`

## Build Commands

### Standard Release Build
```powershell
cd c:\Users\stevensilber\source\repos\data_processing\cpp\simulation
mkdir build -Force; cd build
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

## Running Simulations

### 2D Simulations

```powershell
# Single cell test
.\build\bin\Release\cell_sim.exe -n 1 -N 256 -r 49 -t 10 --dt 0.01 -o agent_test_runs/test_2d_single

# Multi-cell test
.\build\bin\Release\cell_sim.exe -n 8 -N 512 -r 49 -t 10 --dt 0.01 -o agent_test_runs/test_2d_multi

# With motility
.\build\bin\Release\cell_sim.exe -n 8 -N 512 -r 49 -t 100 --dt 0.01 --v-A 0.01 --tau 100 --trajectory-samples 200 -o agent_test_runs/test_2d_motile
```

**IMPORTANT 2D Notes:**
- Use `--dt 0.01` for 2D simulations
- Use radius `-r 49` for standard tests (matches paper parameters)
- CLI default radius is 20, but always specify `-r 49` explicitly

### 3D Simulations

```powershell
# Single cell test
.\build\bin\Release\cell_sim.exe --3d -n 1 -N 240 -r 49 -t 10 --dt 0.02 --checkpoint-interval 500 -o agent_test_runs/test_3d_single

# Multi-cell (16 cells at 85% confluence)
# Domain calculation: N = cbrt(n_cells * cell_volume / target_confluence)
# For r=49: V = (4/3)πr³ ≈ 492807, so N ≈ cbrt(16 * 492807 / 0.85) ≈ 210
.\build\bin\Release\cell_sim.exe --3d -n 16 -N 210 -r 49 -t 10 --dt 0.02 --checkpoint-interval 500 -o agent_test_runs/test_3d_multi
```

**IMPORTANT 3D Notes:**
- Use `--dt 0.02` for 3D simulations (can use 0.01 if unstable)
- Use random initialization (default). Do NOT use `--grid` or `--confluence` flags for 3D
- Larger `--checkpoint-interval` (500+) recommended to reduce I/O
- Calculate domain size manually for desired confluence

### Resume from Checkpoint
```powershell
.\build\bin\Release\cell_sim.exe -c agent_test_runs/test_checkpoint/checkpoint.bin -t 100 -o agent_test_runs/resumed_run
```

## CLI Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--3d` | Enable 3D mode | false |
| `-n <num>` | Number of cells | 8 |
| `-N <size>` | Domain size (NxN for 2D, NxNxN for 3D) | 256 |
| `-Nz <size>` | Z dimension for 3D (if different from N) | same as N |
| `-r <radius>` | Cell radius | 20 (use 49 for production) |
| `-t <time>` | End time | 100 |
| `--dt <step>` | Time step | 0.01 |
| `-o <dir>` | Output directory | ./output |
| `-c <file>` | Load checkpoint | — |
| `--v-A <f>` | Active motility velocity | 0 |
| `--tau <f>` | Persistence/reorientation time | 100 |
| `--abp` | Use Active Brownian Particle model (vs Run-and-Tumble) | false |
| `--save-interval <n>` | Steps between VTK saves (0=none) | 100 |
| `--trajectory-samples <n>` | Trajectory points to save | 100 |
| `--trajectory-interval <n>` | Steps between trajectory saves (-1=use save_interval) | -1 |
| `--checkpoint-interval <n>` | Steps between checkpoints | save_interval×10 |
| `--save-final-checkpoint` | Save checkpoint at end | false |
| `--seed <n>` | Random seed | time-based |
| `--subdomain-padding <f>` | Cell bbox size as multiple of R | 2.0 |
| `--stress-fields` | Output stress tensor fields (requires `ENABLE_STRESS_FIELDS` build) | false |
| `--observable-interval <n>` | Steps between GPU diagnostics (requires `ENABLE_DIAGNOSTICS` build) | 0 (disabled) |
| `--use-diagnostics` | Enable volume/shape computation | false |
| `--grid` | Grid-based cell initialization (for high confluence 2D) | false |
| `--confluence <f>` | Target confluence 0-1 (implies --grid) | 0.85 |
| `-l, --lambda <f>` | Interface width λ | 7.0 |

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
.\build\bin\Release\cell_sim.exe --3d -n 16 -N 210 -r 49 -t 10 --dt 0.02 --checkpoint-interval 500 -o agent_test_runs/validate_3d_multi
# Expected: cells interact and repel, no collapse
```

## Visualization Commands

```powershell
# 2D: Plot last frame
python visualize.py agent_test_runs/my_sim --last 1

# 2D: Generate movie
python visualize.py agent_test_runs/my_sim --movie

# 2D: Specific frame range
python visualize.py agent_test_runs/my_sim --start 0 --end 100

# 3D: Volume rendering (REQUIRED to see cell interiors)
python visualize_3d.py agent_test_runs/my_3d_sim --volume

# 3D: Generate movie with volume rendering
python visualize_3d.py agent_test_runs/my_3d_sim --movie --volume

# 3D: Isosurface rendering (shows cell boundaries only)
python visualize_3d.py agent_test_runs/my_3d_sim --iso 0.5

# Trajectory analysis (MSD, velocity autocorrelation)
python analyze_trajectory.py agent_test_runs/my_sim --no-show
```

**IMPORTANT Visualization Notes:**
- Always use `--volume` flag for 3D to see cell interiors, not just surfaces
- Pass directory as positional argument (not with `-d` flag)
- `--last N` shows last N frames

## Key Physical Parameters

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| Radius | R | 49 | Target area/volume (use `-r` to set) |
| Interface width | λ | 7 | Controls interface sharpness |
| Gradient energy | γ | 1 | Interface energy coefficient |
| Repulsion | κ | 10 | Cell-cell repulsion strength |
| Volume constraint | μ | 1 | Strength of volume conservation |
| Target volume (2D) | V₀ | πR² ≈ 7543 | For R=49 |
| Target volume (3D) | V₀ | (4/3)πR³ ≈ 492807 | For R=49 |

These are defined in `include/types.cuh` (2D) and `include/types3d.cuh` (3D).

## Development Workflow

### Adding a Feature
1. Build in Release mode
2. Run all 4 validation tests
3. Run your specific test case
4. Visualize results to confirm correctness
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

## Output Directory Convention

**All test output goes to:** `agent_test_runs/`

Use descriptive names:
```powershell
-o agent_test_runs/feature_name_test
-o agent_test_runs/bugfix_validation  
-o agent_test_runs/perf_benchmark_n64
```

## Cluster Operations

See [cluster-operations.instructions.md](.github/instructions/cluster-operations.instructions.md) for comprehensive cluster documentation including:
- SSH persistent connection setup (avoiding MFA)
- Building on cluster (modules, cmake)
- Job submission (presets, SLURM)
- Submission log tracking
- Status monitoring
- File transfer

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
dφ/dt = -v·∇φ - 0.5 * (-2γ∇²φ + f'(φ) + volume_constraint + repulsion)
```
Where:
- `f'(φ) = (60/λ²) * φ(1-φ)(1-2φ)` - bulk potential enforcing φ ∈ [0,1]
- Volume constraint: `δE/δφ = -4(μ/V₀)(V₀ - V)φ`
- Repulsion: `δE/δφᵢ = (60κ/λ²) * φᵢ * Σⱼφⱼ²`

See `include/physics.cuh` for detailed inline documentation.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| NaN in output | Reduce dt by 2x, check bounding box updates |
| Volume drift | Check μ parameter, verify volume constraint term |
| OOM on checkpoint load | Build with `-DSAFE_MODE=ON` |
| Cells disappear | Check bounding box tracking, increase subdomain-padding |
| Build fails | Ensure CUDA 11+, CMake 3.18+, C++17 compiler |
| Cells overlap excessively | Increase κ (repulsion), check initialization spacing |
| Slow performance | Use `--save-interval 0` during benchmarks |

## Additional Documentation

- [AGENT_ONBOARDING.md](cpp/simulation/AGENT_ONBOARDING.md) - Full onboarding and physics background
- [DIAGNOSTICS_DESIGN.md](cpp/simulation/DIAGNOSTICS_DESIGN.md) - GPU diagnostics system design
- [RUNBOOK.md](cpp/simulation/RUNBOOK.md) - Detailed operational runbook

**Note**: For physics details and implementation specifics, refer to inline documentation in `include/physics.cuh`. The code comments are authoritative for implementation details.
