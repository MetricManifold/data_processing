# Simulation Runbook (Quick Reference)

**For AI agents: Use this when you need to build, run, or test the simulation.**

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
.\build\bin\Release\cell_sim.exe --3d -n 1 --size 240 -r 49 -t 10 --dt 0.01 -o agent_test_runs/test_3d_single

# Multi-cell test
.\build\bin\Release\cell_sim.exe --3d -n 8 --size 150 -r 49 -t 10 --dt 0.02 -o agent_test_runs/test_3d_multi
```

### Resume from Checkpoint

```powershell
.\build\bin\Release\cell_sim.exe -c agent_test_runs/test_checkpoint/checkpoint.bin -t 100 -o agent_test_runs/resumed_run
```

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
| `--3d` | Enable 3D mode | false |
| `--save-interval <n>` | Steps between VTK saves | 100 |
| `--trajectory-samples <n>` | Trajectory points to save | 100 |
| `--checkpoint-interval <n>` | Steps between checkpoints | save_interval×10 |
| `--save-final-checkpoint` | Save checkpoint at end | false |
| `--seed <n>` | Random seed | — |
| `--no-vtk` | Disable VTK output | false |
| `--no-diagnostics` | Skip diagnostics | false |

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
.\build\bin\Release\cell_sim.exe --3d -n 1 --size 240 -r 49 -t 10 --dt 0.01 -o agent_test_runs/validate_3d_single
# Expected: volume ~492807, phi_max ~1.0, no NaN

# Test 4: 3D multi-cell
.\build\bin\Release\cell_sim.exe --3d -n 2 --size 240 -r 40 -t 10 --dt 0.01 -o agent_test_runs/validate_3d_multi
# Expected: cells interact, no collapse
```

---

## Visualization Commands

```powershell
# 2D: Plot last frame
python visualize.py -d agent_test_runs/my_sim --last

# 2D: Generate movie
python visualize.py -d agent_test_runs/my_sim --movie

# 3D: Isosurface
python visualize_3d.py agent_test_runs/my_3d_sim --iso

# 3D: Generate movie
python visualize_3d.py agent_test_runs/my_3d_sim --movie --fps 10

# Trajectory analysis (MSD, autocorrelations)
python analyze_trajectory.py agent_test_runs/my_sim --no-show
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
