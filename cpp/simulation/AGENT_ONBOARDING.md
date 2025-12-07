# Cellular Migration Simulation Runbook

## 1. Project Overview

**Purpose:**  
This project simulates cellular migration using a **phase-field model** accelerated by CUDA. It models cell membranes as continuous fields that evolve according to a free energy functional, supporting both 2D and 3D environments.

**What is a Phase-Field Model?**  
Instead of tracking discrete cell boundaries, each cell is represented by a smooth field φₙ(x) where:
- φ ≈ 1 inside the cell
- φ ≈ 0 outside the cell  
- Smooth transition (tanh profile) at the interface with width λ

This approach naturally handles:
- Cell deformation and shape changes
- Cell-cell repulsion (overlapping fields increase energy)
- Volume conservation (soft constraint on ∫φ² dV)
- Active motility and migration

**Key Goals:**  
- Achieve efficient, large-scale cell migration simulations in 2D and 3D
- Optimize memory usage per cell to enable higher cell counts
- Maintain or improve computational performance as memory optimizations are applied
- Study jamming transitions and collective cell behavior

**Scientific Objective:**  
This simulation aims to reproduce and extend results from experimental and theoretical studies of cell jamming:
- Bresler et al., "Near the jamming transition of elastic active cells" (arXiv:1807.10318)
- Palmieri et al., "Multiple scale model for cell migration in monolayers" (Nature Sci. Rep. 2015, DOI:10.1038/srep11745)

**Key phenomena to capture:**
- Transition from fluid-like (diffusive) to solid-like (jammed) collective behavior
- Mean squared displacement (MSD) crossover: ballistic → diffusive → caged
- Effective diffusion coefficient D_eff as function of cell density and motility
- Voronoi cell shape as order parameter near jamming

**Current Development Priority:** Performance and memory optimization to enable larger cell counts (scaling).

---

## 2. Codebase Structure

```
cpp/simulation/
├── CMakeLists.txt              # Build system (CUDA 11+, C++17)
├── AGENT_ONBOARDING.md         # This file (main runbook)
├── visualize.py                # 2D visualization (VTK → PNG/movie)
├── visualize_3d.py             # 3D visualization (checkpoints → isosurface)
├── analyze_trajectory.py       # MSD, autocorrelations, diffusion analysis
├── include/
│   ├── types.cuh               # SimParams, BoundingBox, Vec2 (2D)
│   ├── types3d.cuh             # SimParams3D, BoundingBox3D, Vec3, PhysicsCoeffs
│   ├── physics.cuh             # Shared physics helpers (tanh_profile, etc.)
│   ├── cell.cuh                # Cell class (2D phase field on subdomain)
│   ├── cell3d.cuh              # Cell3D class
│   ├── domain.cuh              # Domain class (cell collection, overlap pairs)
│   ├── domain3d.cuh            # Domain3D class
│   ├── kernels.cuh             # 2D kernel declarations
│   ├── kernels3d.cuh           # 3D kernel declarations
│   ├── integrator.cuh          # Integrator class (2D, memory management)
│   ├── integrator3d.cuh        # Integrator3D class
│   ├── io.cuh                  # I/O functions (VTK, checkpoints, tracking)
│   ├── simulation.cuh          # Simulation class (2D top-level controller)
│   └── simulation3d.cuh        # Simulation3D class
├── src/
│   ├── main.cu                 # Entry point (CLI parsing, runs 2D or 3D)
│   ├── kernels_optimized_v2.cu # Fused 2D kernels (main path)
│   ├── kernels_optimized_v4.cu # 2D kernels with neighbor-list optimization
│   ├── kernels3d.cu            # 3D kernel implementations
│   ├── integrator.cu           # 2D integrator implementation
│   ├── io.cu                   # 2D I/O implementation
│   └── io3d.cu                 # 3D I/O implementation
```

**Related directories (outside simulation/):**
- `cudatest/` — CUDA test code and experiments

---

## 3. Physics Model (Complete Reference)

### Free Energy Functional

The system minimizes a free energy:

$$\mathcal{F} = \mathcal{F}_0 + \mathcal{F}_{int}$$

**Single-cell free energy:**
$$\mathcal{F}_0 = \sum_n \left[ \gamma_n \int dx \int dy \left( (\nabla\phi_n)^2 + \frac{30}{\lambda^2}\phi_n^2(1-\phi_n)^2 \right) + \frac{\mu_n}{\pi R^2}\left(\pi R^2 - \int dx \int dy \, \phi_n^2 \right)^2 \right]$$

**Interaction energy (cell-cell repulsion):**
$$\mathcal{F}_{int} = \frac{30\kappa}{\lambda^2} \int dx \int dy \sum_{n,m \neq n} \phi_n^2 \phi_m^2$$

### Physical Parameters

| Symbol | Name | Value | Notes |
|--------|------|-------|-------|
| λ | Interface width | 7 | Controls interface sharpness (grid points) |
| γ | Gradient coefficient | 1 | Interface energy |
| κ | Interaction strength | 10 | Cell-cell repulsion |
| μ | Volume constraint strength | 1 | Soft area/volume constraint |
| R | Target cell radius | 49 | Target area = πR² ≈ 7543 (2D), volume = (4/3)πR³ (3D) |
| ξ | Friction coefficient | 1.5 × 10³ | Drag on cell motion |
| τ | Reorientation time | 10⁴ | Polarization persistence |
| v_A | Active motility speed | 10⁻² | Self-propulsion magnitude |

### Derived Coefficients

| Coefficient | Formula | Value | Used in |
|-------------|---------|-------|---------|
| bulk_coeff | 30/λ² | 30/49 ≈ 0.612 | Bulk potential f'(φ) |
| interaction_coeff | 30κ/λ² | 300/49 ≈ 6.122 | Cell-cell repulsion |
| volume_coeff | μ/(πR²) | 1/7543 ≈ 1.33×10⁻⁴ | Volume constraint (2D) |
| motility_coeff | 60κ/(ξλ²) | 600/(1500×49) ≈ 8.16×10⁻³ | Velocity from interactions |

### Equation of Motion

$$\frac{\partial \phi_n}{\partial t} + \mathbf{v}_n \cdot \nabla\phi_n = -\frac{1}{2}\frac{\delta \mathcal{F}}{\delta \phi_n}$$

Where the **motility/advection term** $\mathbf{v}_n \cdot \nabla\phi_n$ represents active cell motion.

### Variational Derivative (Complete Derivation)

The variational derivative $\frac{\delta \mathcal{F}}{\delta \phi_n}$ is computed term by term:

**1. Gradient term:**
$$\frac{\delta}{\delta \phi_n}\left[\gamma \int (\nabla\phi_n)^2\right] = -2\gamma \nabla^2\phi_n$$

**2. Bulk potential:** $f(\phi) = \frac{30}{\lambda^2}\phi^2(1-\phi)^2$
$$f'(\phi) = \frac{\delta}{\delta \phi}\left[\frac{30}{\lambda^2}\phi^2(1-\phi)^2\right] = \frac{60}{\lambda^2}\phi(1-\phi)(1-2\phi)$$

**3. Volume constraint:**
$$\frac{\delta}{\delta \phi_n}\left[\frac{\mu}{\pi R^2}\left(\pi R^2 - \int\phi_n^2\right)^2\right]$$

Let $V = \int \phi^2 \, dx$ and $A = \pi R^2$. Then:
$$\frac{\delta}{\delta \phi}[(A-V)^2] = 2(A-V) \cdot \frac{\delta(A-V)}{\delta\phi} = 2(A-V)(-2\phi) = -4(A-V)\phi$$

So: $\frac{\delta F_{vol}}{\delta \phi} = -\frac{4\mu}{\pi R^2}\left(\pi R^2 - \int\phi^2\right)\phi$

**4. Interaction term:**
$$\frac{\delta}{\delta \phi_n}\left[\frac{30\kappa}{\lambda^2} \int \sum_{m \neq n} \phi_n^2 \phi_m^2\right] = \frac{60\kappa}{\lambda^2}\phi_n \sum_{m \neq n}\phi_m^2$$

### Combined Variational Derivative

$$\frac{\delta \mathcal{F}}{\delta \phi_n} = -2\gamma \nabla^2\phi_n + \frac{60}{\lambda^2}\phi_n(1-\phi_n)(1-2\phi_n) - \frac{4\mu}{\pi R^2}\left(\pi R^2 - \int\phi_n^2\right)\phi_n + \frac{60\kappa}{\lambda^2}\phi_n\sum_{m\neq n}\phi_m^2$$

### Expanded Equation of Motion

$$\frac{\partial \phi_n}{\partial t} = -\mathbf{v}_n \cdot \nabla\phi_n - \frac{1}{2}\left[ -2\gamma \nabla^2\phi_n + \frac{60}{\lambda^2}\phi_n(1-\phi_n)(1-2\phi_n) - \frac{4\mu}{\pi R^2}\left(\pi R^2 - \int\phi_n^2\right)\phi_n + \frac{60\kappa}{\lambda^2}\phi_n\sum_{m\neq n}\phi_m^2 \right]$$

### 2D vs 3D Differences

The formulation is dimension-independent! Only the geometric quantities change:

| Quantity | 2D | 3D |
|----------|-----|-----|
| Target volume | $V_{target} = \pi R^2$ | $V_{target} = \frac{4}{3}\pi R^3$ |
| Volume integral | $V = \int \phi^2 \, dx\,dy$ | $V = \int \phi^2 \, dx\,dy\,dz$ |
| Laplacian | $\nabla^2\phi = \phi_{xx} + \phi_{yy}$ (5-point stencil) | $\nabla^2\phi = \phi_{xx} + \phi_{yy} + \phi_{zz}$ (7-point stencil) |

### Motility Model

Total velocity: **v_n = v_{n,I} + v_{n,A}**

**v_{n,I} - Interaction velocity (implemented):**
$$\mathbf{v}_{n,I} = \frac{60\kappa}{\xi\lambda^2} \int \phi_n (\nabla\phi_n) \sum_{m \neq n} \phi_m^2 \, dx$$

**v_{n,A} - Active motility:**
- Constant speed: |v_{n,A}| = v_A
- Persistent direction θ_n per cell
- Random reorientation with exponential waiting time: P(t_r) = (1/τ)exp(-t_r/τ)
- Run-and-tumble dynamics → effective diffusion D_eff = v_A²τ/2

---

## 4. Architecture Deep Dive

### Subdomain Optimization

**Key insight:** Each cell only occupies a small region of the domain. Instead of storing N×N (or N×N×N) fields per cell, each cell has a **bounding box** that tracks only the relevant region.

```
Global Domain (800×800)
┌─────────────────────────────────────┐
│                                     │
│    ┌───────┐                        │
│    │ Cell 0│  ← Subdomain ~200×200  │
│    │ φ≈1   │                        │
│    └───────┘                        │
│              ┌───────┐              │
│              │ Cell 1│              │
│              │ φ≈1   │              │
│              └───────┘              │
│                                     │
└─────────────────────────────────────┘
```

**BoundingBox struct:**
- `x0, y0` — lower-left corner (can be negative for periodic wrapping)
- `x1, y1` — upper-right corner (can exceed domain size)
- `halo` — ghost cells for stencil operations

**Periodic Boundary Conditions:**
The subdomain can wrap around domain edges. The `global_to_local()` and `local_to_global()` functions handle coordinate conversion with periodic wrapping.

### Memory Layout

**Per-cell GPU memory (2D):**
- `d_phi` — Phase field φ (subdomain_size floats)
- `d_dphi_dt` — Time derivative (subdomain_size floats)

**Shared work buffers (Integrator):**
- Single large buffer `d_work_buffer` used for all intermediate computations
- Sized for largest cell × 9 slots (see buffer slot allocation below)

### Buffer Slot Allocation (2D Kernels)

The fused kernels use 9 work buffer slots per cell:

| Slot | Name | Purpose |
|------|------|---------|
| 0 | laplacian | ∇²φ |
| 1 | bulk_term | f'(φ) = 60/λ² · φ(1-φ)(1-2φ) |
| 2 | constraint_term | Volume constraint contribution |
| 3 | repulsion_term | Cell-cell repulsion: κφΣφ_m² |
| 4 | grad_x | ∂φ/∂x for advection |
| 5 | grad_y | ∂φ/∂y for advection |
| 6 | advection_term | v·∇φ |
| 7 | interaction_sum | Σ_m φ_m² from other cells |
| 8 | phi_sq | φ² for volume integral |

### Kernel Execution Flow (step_fused_v2)

```
1. Upload cell pointers/metadata to GPU
2. For each cell (parallel):
   a. Compute laplacian (5-point stencil)
   b. Compute bulk potential f'(φ)
   c. Compute interaction sum Σφ_m² (reads overlapping cells)
   d. Compute repulsion term
   e. Compute volume integral ∫φ² (reduction)
   f. Compute volume constraint term
   g. Compute gradient ∇φ
   h. Compute advection v·∇φ
   i. Combine all terms into dφ/dt
3. Download computed volumes/centroids
4. For each cell: φ += dt · dφ/dt (Euler step)
5. Update cell velocities from motility integrals
6. Periodically update bounding boxes (every N steps)
```

---

## 5. Building and Running

### Prerequisites
- CUDA Toolkit 11.0+ (tested with 12.x)
- CMake 3.18+
- C++17 compiler
- GPU with compute capability 7.5+ (Turing or newer)

### Build Commands

**Standard Release build:**
```powershell
cd cpp/simulation
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

**Safe mode build (CRITICAL for experimental work):**
```powershell
cmake .. -DCMAKE_BUILD_TYPE=Release -DSAFE_MODE=ON
cmake --build . --config Release
```

⚠️ **SAFE_MODE enables GPU memory tracking macros that:**
- Track total GPU memory allocation
- Prevent runaway allocations that could exhaust GPU memory
- **Essential when loading checkpoints** — a corrupted or incorrect checkpoint read can attempt to allocate many terabytes of memory, crashing the system
- Adds small overhead, so disable for production runs

**When to use SAFE_MODE:**
- Loading checkpoints from unknown sources
- Testing new I/O code
- Debugging memory issues
- Experimental development work

**When to disable SAFE_MODE:**
- Production runs after validation
- Performance benchmarking

### Running Simulations

**2D simulation:**
```powershell
.\build\bin\Release\cell_sim.exe -n 16 -N 512 -r 49 -t 100 --dt 0.01 -o agent_test_runs/test_2d
```

**3D simulation:**
```powershell
.\build\bin\Release\cell_sim.exe --3d -n 8 --size 240 -r 49 -t 10 --dt 0.01 -o agent_test_runs/test_3d
```

### CLI Reference

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
| `--no-diagnostics` | Skip diagnostics for speed | false |

---

## 6. Output Directory Convention

### Agent Test Runs

**All agent test output should go to:** `agent_test_runs/`

This keeps experimental runs separate from production data. Use descriptive subdirectory names:

```powershell
# Good examples:
-o agent_test_runs/single_cell_3d_test
-o agent_test_runs/memory_opt_v2_benchmark
-o agent_test_runs/equilibration_n16

# Bad examples (pollute workspace):
-o output
-o test
-o .
```

### Output Directory Structure

After a simulation, the output directory contains:

```
agent_test_runs/my_simulation/
├── phi_000000.vtk          # VTK frame at step 0
├── phi_000100.vtk          # VTK frame at step 100
├── ...
├── checkpoint.bin          # Latest checkpoint (if --save-final-checkpoint)
├── checkpoint_001000.bin   # Periodic checkpoint at step 1000
├── trajectory.txt          # Cell positions/velocities over time
└── images/                 # Generated by visualize.py
    ├── frame_000000.png
    └── ...
```

---

## 7. Simulation Workflow (How to Run Experiments)

### Two-Phase Workflow for MSD Studies

#### Phase 1: Equilibration (v_A = 0)

First, equilibrate the system without active motility to let cells relax to a stable configuration:

```powershell
.\build\bin\Release\cell_sim.exe -n 12 -t 100 --v-A 0 -o agent_test_runs/equilibrated --save-interval 1000 --save-final-checkpoint
```

**Equilibration criteria:**
- Volume should stabilize near target (7543 for R=49 in 2D)
- Shape factor should stabilize (~0.89-0.90)
- No significant cell rearrangements

#### Phase 2: Production Run with Motility

Load the equilibrated checkpoint and enable active motility:

```powershell
.\build\bin\Release\cell_sim.exe -c agent_test_runs/equilibrated/checkpoint.bin -o agent_test_runs/motile -t 200 --v-A 0.5 --trajectory-samples 100
```

### Resuming Interrupted Simulations

```powershell
# Resume from checkpoint to new end time
.\build\bin\Release\cell_sim.exe -c agent_test_runs/output_dir/checkpoint.bin -o agent_test_runs/output_dir -t <new_end_time>
```

**Notes:**
- The `-t` parameter is the **absolute** end time, not additional time
- You can change parameters like `--v-A` when resuming

### 3D Simulation Guidelines

**Recommended parameters:**

| Cells | Domain | dt | Time for stable | Notes |
|-------|--------|-----|-----------------|-------|
| 1 | 240³ | 0.01 | t=10 | Single cell validation |
| 8 | 150³ | 0.02 | t=10 | Small multi-cell |
| 16 | 200³ | 0.02 | t=10 | Medium run |
| 64 | 250³ | 0.02 | t=10 | Large (needs ~6GB VRAM) |

**3D test commands:**
```powershell
# Single cell validation (ALWAYS start here)
.\build\bin\Release\cell_sim.exe --3d -n 1 --size 240 -r 49 --dt 0.01 -t 10 --save-interval 100 -o agent_test_runs/3d_single

# Multi-cell test
.\build\bin\Release\cell_sim.exe --3d -n 8 --size 150 -r 49 --dt 0.02 -t 10 --save-interval 500 -o agent_test_runs/3d_multi
```

### Memory Requirements (3D)

Per-cell memory at 150³ max bbox:
- phi + dphi: 2 × 150³ × 4 bytes = 27 MB
- Work buffers: 7 × 150³ × 4 bytes = 94.5 MB per cell

**64 cells at 150³: ~6.0 GB** ✓ Fits in 8GB GPU

⚠️ **DO NOT use serial processing** — it destroys performance. Always keep parallel.

---

## 8. Common Pitfalls and Debugging

### NaN Explosion

**Symptoms:** φ suddenly goes to ±infinity, then NaN

**Causes & Solutions:**
1. **Bounding box update bug** — Fixed in 3D (see bug history below)
2. **Time step too large** — Try reducing dt by 2x
3. **Cell near domain boundary** — Check initialization places cells away from edges

**Diagnosis:**
```cpp
// Add to kernel to check for NaN
if (isnan(phi[idx]) || fabsf(phi[idx]) > 100.0f) {
    printf("NaN detected at idx=%d, step=%d\n", idx, step);
}
```

### Volume Drift

**Symptoms:** Cell volume slowly drifts from target

**Causes:**
1. Volume constraint too weak (increase μ)
2. Numerical discretization error
3. Interface width too small for grid resolution

### Cells Disappear

**Symptoms:** Cell vanishes or shrinks rapidly

**Causes:**
1. Bounding box not tracking cell properly
2. Cell pushed outside subdomain
3. Strong repulsion causing collapse

### Checkpoint Loading Crashes

**Symptoms:** System hangs or OOM when loading checkpoint

**Cause:** Corrupted checkpoint attempts to allocate huge memory

**Solution:** Build with `-DSAFE_MODE=ON` for memory tracking

### Historical Bug: 3D Bounding Box Update (FIXED 2025-12-05)

**Root cause:** `Cell3D::update_bounding_box()` had several issues:
- Used min/max local coords instead of centroid + max periodic distance
- Iterated NEW coords → OLD (backward)
- `global_to_local()` could return negative indices

**Fix applied:** Rewrote to match working 2D algorithm.

**Verification test:**
```powershell
.\build\bin\Release\cell_sim.exe --3d -n 1 --size 240 -r 49 --dt 0.01 -t 100 -o agent_test_runs/3d_bbox_test
# Expected: volume ~492807 (99%+), no NaN
```

---

## 9. Memory Optimization History

### 2D Optimization
- **Initial:** 12 work buffers per cell
- **Current:** 9 work buffers per cell
- **Method:** Eliminated unused buffers, remapped kernel accesses
- **Validated:** All kernel paths (V2 and V4) tested and confirmed

### 3D Optimization
- **Initial:** 7 work buffers per cell
- **Current:** Fused kernel combines laplacian + bulk + constraint + advection in ONE pass
- **Result:** Eliminated grad_x, grad_y, grad_z intermediate buffers (3 buffers saved)
- **Impact:** ~30% memory savings, enables 64 cells at 150³ subdomain in 8GB GPU

---

## 10. Performance Considerations

### Critical Rules
1. **Always benchmark after changes** — Use consistent test case (e.g., 16 cells, 1000 steps)
2. **Never use serial processing** — Parallel cell processing is essential
3. **Memory bandwidth vs compute** — Reducing buffers helps if memory-bound
4. **Minimize GPU↔CPU transfers** — Batch operations, reduce sync frequency

### Tuning Parameters
- `bbox_update_interval` (default 10) — How often to update bounding boxes
- `use_fused_v2` vs `use_fused_v4` — V4 uses neighbor lists for O(N) interactions
- CUDA block size: 16×16 (2D), 8×8×8 (3D)

### Profiling
```powershell
nsys profile .\build\bin\Release\cell_sim.exe -n 16 -t 10 -o agent_test_runs/profile_output
```

---

## 11. Iterating on Changes (Development Workflow)

### Before Making Changes
1. Understand the affected code path (V2 vs V4, 2D vs 3D)
2. Create a test case that exercises the change
3. Record baseline performance metrics

### Validation Test Suite

**Run these tests after any kernel/physics changes:**

```powershell
# Test 1: Single cell 2D (basic sanity)
.\build\bin\Release\cell_sim.exe -n 1 -N 256 -r 49 -t 10 --dt 0.01 -o agent_test_runs/validate_2d_single
# Expected: volume ~7543, phi_max ~1.0, no NaN

# Test 2: Multi-cell 2D
.\build\bin\Release\cell_sim.exe -n 8 -N 512 -r 49 -t 10 --dt 0.01 -o agent_test_runs/validate_2d_multi
# Expected: cells repel, volumes stable

# Test 3: Single cell 3D
.\build\bin\Release\cell_sim.exe --3d -n 1 --size 240 -r 49 -t 10 --dt 0.01 -o agent_test_runs/validate_3d_single
# Expected: volume ~492807, phi_max ~1.0, no NaN

# Test 4: Multi-cell 3D
.\build\bin\Release\cell_sim.exe --3d -n 2 --size 240 -r 40 -t 10 --dt 0.01 -o agent_test_runs/validate_3d_multi
# Expected: cells interact, no collapse
```

### After Making Changes
1. Build in both Debug and Release modes
2. Run validation tests above
3. Benchmark and compare to baseline
4. Update this document if behavior changes
5. Document the change rationale in commit message

### Code Style
- Use `__host__ __device__` for functions callable from both CPU and GPU
- Prefix device-only pointers with `d_`
- Use `#pragma once` for header guards
- Keep kernels focused — one kernel, one responsibility

---

## 12. Visualization and Post-Processing

Visualization is **vital for collecting and validating results**.

### 12.1 2D Visualization (visualize.py)

**Dependencies:** matplotlib, numpy, imageio

```powershell
# Plot last frame
python visualize.py -d agent_test_runs/my_sim --last

# Generate movie
python visualize.py -d agent_test_runs/my_sim --movie

# Energy analysis
python visualize.py -d agent_test_runs/my_sim --energy

# With trajectory arrows
python visualize.py -d agent_test_runs/my_sim --frame 100 --use-arrows
```

### 12.2 3D Visualization (visualize_3d.py)

**Dependencies:** numpy, pyvista

```powershell
# Isosurface visualization
python visualize_3d.py agent_test_runs/my_3d_sim --iso

# Volume rendering
python visualize_3d.py agent_test_runs/my_3d_sim --volume

# Generate movie
python visualize_3d.py agent_test_runs/my_3d_sim --movie --fps 10
```

### 12.3 Trajectory Analysis (analyze_trajectory.py)

**Dependencies:** numpy, matplotlib

```powershell
# Full analysis (MSD, autocorrelations, trajectories)
python analyze_trajectory.py agent_test_runs/my_sim

# Headless mode (no display)
python analyze_trajectory.py agent_test_runs/my_sim --no-show
```

**Output files:**
- `msd.png` — Mean squared displacement (log-log + linear)
- `autocorrelations.png` — Velocity and polarization autocorrelation
- `trajectories.png` — 2D cell paths

**Physical interpretation:**
| Quantity | Free Cell | Jammed |
|----------|-----------|--------|
| MSD slope | ~2→~1 (ballistic→diffusive) | ~0 (caged) |
| D_eff | v_A²τ/2 | → 0 |
| τ_p | ≈ τ (input) | < τ (collisions) |

---

## 13. Key Files to Understand

**Start here (in order):**
1. [types.cuh](include/types.cuh) — Parameters and data structures
2. [cell.cuh](include/cell.cuh) — Cell class and phase field initialization
3. [kernels.cuh](include/kernels.cuh) — Kernel function signatures
4. [kernels_optimized_v2.cu](src/kernels_optimized_v2.cu) — Main kernel implementations
5. [integrator.cuh](include/integrator.cuh) — Memory management and stepping

**For 3D work:**
1. [types3d.cuh](include/types3d.cuh) — 3D parameter extensions
2. [cell3d.cuh](include/cell3d.cuh) — 3D cell with bounding box
3. [kernels3d.cu](src/kernels3d.cu) — 3D kernel implementations

---

## 14. Onboarding Checklist

- [ ] Read this document completely
- [ ] Build the project successfully (Release mode)
- [ ] Build with SAFE_MODE=ON for experimental work
- [ ] Run single-cell 2D test, verify output in `agent_test_runs/`
- [ ] Run single-cell 3D test, verify no NaN
- [ ] Run visualize.py on 2D output
- [ ] Run visualize_3d.py on 3D output
- [ ] Run analyze_trajectory.py on output with motility
- [ ] Understand buffer slot allocation in kernels
- [ ] Run validation test suite before making changes

---

## 15. Future Work / Open Questions

1. **Active motility (v_A):** Fully implemented in 2D, basic in 3D. MSD analysis pending.
2. **Heterogeneous cell types:** Allow assigning different parameter sets to different cells when loading from checkpoint. Tunable parameters would include:
   - λ (interface width) — affects cell stiffness/deformability
   - κ (interaction strength) — affects repulsion strength
   - μ (volume constraint) — affects compressibility
   - v_A (active velocity) — affects motility
   - τ (reorientation time) — affects persistence
   
   This enables studying binary mixtures of soft/stiff cells (as in Bresler et al.) and elastic mismatch effects (as in Palmieri et al.).

3. **Performance optimization:** Ongoing work to reduce memory per cell and increase maximum cell count. Target: 100+ cells in 3D, 500+ cells in 2D.

---

## 16. End-to-End Workflow Example

This section demonstrates a complete experiment: studying MSD as a function of cell density.

### Step 1: Define Experiment Parameters

**Goal:** Compare MSD for low-density (fluid) vs high-density (jammed) configurations.

| Run | Cells | Domain | Packing Fraction | Expected Behavior |
|-----|-------|--------|------------------|-------------------|
| A | 8 | 512² | ~7% (φ << φ_c) | Diffusive |
| B | 32 | 512² | ~28% (φ ≈ φ_c) | Near jamming |
| C | 64 | 512² | ~56% (φ > φ_c) | Jammed/caged |

*Packing fraction: φ = N × πR² / L² where N=cells, R=49, L=512*

### Step 2: Equilibration (No Motility)

```powershell
# Run A: Low density
.\build\bin\Release\cell_sim.exe -n 8 -N 512 -r 49 -t 50 --v-A 0 --dt 0.01 --save-interval 500 --save-final-checkpoint -o agent_test_runs/jamming_study/run_A_equil

# Run B: Medium density
.\build\bin\Release\cell_sim.exe -n 32 -N 512 -r 49 -t 50 --v-A 0 --dt 0.01 --save-interval 500 --save-final-checkpoint -o agent_test_runs/jamming_study/run_B_equil

# Run C: High density
.\build\bin\Release\cell_sim.exe -n 64 -N 512 -r 49 -t 50 --v-A 0 --dt 0.01 --save-interval 500 --save-final-checkpoint -o agent_test_runs/jamming_study/run_C_equil
```

**Validation:** Check that volumes stabilized (~7543 each), no overlapping cells exploded.

### Step 3: Production Run with Motility

```powershell
# Run A: Low density with motility
.\build\bin\Release\cell_sim.exe -c agent_test_runs/jamming_study/run_A_equil/checkpoint.bin -t 200 --v-A 0.01 --trajectory-samples 500 -o agent_test_runs/jamming_study/run_A_motile

# Run B: Medium density with motility
.\build\bin\Release\cell_sim.exe -c agent_test_runs/jamming_study/run_B_equil/checkpoint.bin -t 200 --v-A 0.01 --trajectory-samples 500 -o agent_test_runs/jamming_study/run_B_motile

# Run C: High density with motility
.\build\bin\Release\cell_sim.exe -c agent_test_runs/jamming_study/run_C_equil/checkpoint.bin -t 200 --v-A 0.01 --trajectory-samples 500 -o agent_test_runs/jamming_study/run_C_motile
```

### Step 4: Analyze Results

```powershell
# Generate MSD plots for each run
python analyze_trajectory.py agent_test_runs/jamming_study/run_A_motile --no-show
python analyze_trajectory.py agent_test_runs/jamming_study/run_B_motile --no-show
python analyze_trajectory.py agent_test_runs/jamming_study/run_C_motile --no-show
```

### Step 5: Interpret Results

**Expected MSD behavior:**

| Run | Short time (t < τ) | Long time (t >> τ) | D_eff |
|-----|-------------------|-------------------|-------|
| A (fluid) | MSD ~ t² (ballistic) | MSD ~ t (diffusive) | D ≈ v_A²τ/2 |
| B (near jamming) | MSD ~ t² | MSD ~ t^α, α < 1 | D reduced |
| C (jammed) | MSD ~ t² briefly | MSD → plateau (caged) | D → 0 |

**Success criteria:**
- Clear difference in long-time MSD slopes between runs
- Run A shows linear MSD growth (diffusive)
- Run C shows MSD plateau (caging)
- Can extract D_eff from linear fit to long-time MSD

### Step 6: Visualize

```powershell
# Generate movies to visually confirm behavior
python visualize.py -d agent_test_runs/jamming_study/run_A_motile --movie
python visualize.py -d agent_test_runs/jamming_study/run_C_motile --movie
```

---

## 17. Glossary of Terms

| Term | Definition |
|------|------------|
| **Phase field (φ)** | Continuous scalar field representing cell presence. φ≈1 inside cell, φ≈0 outside, smooth transition at interface. |
| **Variational derivative** | δF/δφ — the functional derivative of free energy F with respect to field φ. Gives the "force" driving field evolution. |
| **Subdomain / Bounding box** | Region of the global domain where a cell's phase field is non-negligible. Optimization to avoid storing full N×N grid per cell. |
| **Ghost cells / Halo** | Extra grid points around subdomain boundary needed for stencil operations (e.g., Laplacian needs neighbors). |
| **Stencil** | Local pattern of grid points used in finite difference operations. 5-point stencil (2D) or 7-point stencil (3D) for Laplacian. |
| **VTK** | Visualization Toolkit file format. Stores 3D scalar/vector fields. Readable by ParaView, PyVista, etc. |
| **MSD** | Mean Squared Displacement: ⟨|r(t) - r(0)|²⟩ averaged over cells and time origins. Key metric for diffusion/jamming. |
| **Jamming transition** | Phase transition from fluid-like (cells can rearrange) to solid-like (cells trapped by neighbors) behavior. |
| **Packing fraction (φ)** | Area (or volume) fraction occupied by cells: φ = N×πR²/L² (2D). Jamming typically occurs at φ_c ≈ 0.84 for disks. |
| **D_eff** | Effective diffusion coefficient. In diffusive regime: MSD = 4 D_eff t (2D) or 6 D_eff t (3D). |
| **Ballistic regime** | Short-time behavior where MSD ~ t² (cell moves in straight line before reorienting). |
| **Diffusive regime** | Long-time behavior where MSD ~ t (random walk after many reorientations). |
| **Caged dynamics** | When MSD plateaus — cell oscillates in "cage" formed by neighbors but cannot escape. |
| **Run-and-tumble** | Motility model where cell moves straight for random time τ, then picks new random direction. |
| **Laplacian (∇²)** | Second spatial derivative: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² (+ ∂²φ/∂z² in 3D). Measures local curvature. |
| **Euler step** | Simplest time integration: φ(t+dt) = φ(t) + dt × dφ/dt. First-order accurate. |
| **Checkpoint** | Binary file containing complete simulation state (all φ fields, positions, velocities). Enables restart. |
| **CUDA kernel** | GPU function executed in parallel across many threads. One thread typically handles one grid point. |
| **Reduction** | Parallel algorithm to compute sum/max/min across array. Used for volume integrals ∫φ² dx. |
| **Fused kernel** | Single GPU kernel that combines multiple operations to reduce memory bandwidth usage. |
| **Work buffer** | Temporary GPU memory for intermediate calculations (gradients, interaction sums, etc.). |

---

## 18. Useful References

### Primary References (Target Papers)
- **Bresler, Palmieri, Grant (2018):** "Near the jamming transition of elastic active cells: A sharp-interface approach" — [arXiv:1807.10318](https://arxiv.org/abs/1807.10318)
- **Palmieri et al. (2015):** "Multiple scale model for cell migration in monolayers: Elastic mismatch between cells enhances motility" — [DOI:10.1038/srep11745](https://doi.org/10.1038/srep11745)

### Background Reading
- **Phase-field models:** Provatas & Elder, "Phase-Field Methods in Materials Science"
- **CUDA optimization:** NVIDIA CUDA Best Practices Guide
- **Cell migration physics:** Ziebert & Aranson, "Computational approaches to substrate-based cell motility"

### Tools Documentation
- **PyVista:** https://docs.pyvista.org/
- **VTK file formats:** https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf

---

*Last updated: 2025-12-07*
*This document is intended for AI agent onboarding.*
