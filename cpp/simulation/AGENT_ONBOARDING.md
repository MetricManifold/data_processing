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

---

## 2. Codebase Structure

```
cpp/simulation/
├── CMakeLists.txt              # Build system (CUDA 11+, C++17)
├── AGENT_ONBOARDING.md         # This file
├── IMPLEMENTATION_PLAN.md      # Physics equations and implementation phases
├── TEST_PLAN_3D.md             # 3D testing log and bug fixes
├── WORKFLOW.md                 # User workflow for running simulations
├── visualize.py                # 2D visualization (VTK → PNG/movie)
├── visualize_3d.py             # 3D visualization
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
- `rust/cell_data/` — Rust-based MSD analysis tool
- `web/symphas-viewer/` — React web viewer for simulation output

---

## 3. Physics Model (Essential Understanding)

### Free Energy Functional

The system minimizes a free energy:

$$\mathcal{F} = \sum_n \mathcal{F}_n^{single} + \mathcal{F}_{interaction}$$

**Single-cell terms:**
1. **Interface energy:** $\gamma \int |\nabla\phi_n|^2 dx$ — penalizes sharp gradients
2. **Bulk potential:** $\frac{30}{\lambda^2} \int \phi_n^2(1-\phi_n)^2 dx$ — double-well keeps φ near 0 or 1
3. **Volume constraint:** $\frac{\mu}{V_0}(V_0 - \int\phi_n^2 dx)^2$ — soft area/volume conservation

**Interaction term:**
$$\mathcal{F}_{int} = \frac{30\kappa}{\lambda^2} \int \sum_{n,m \neq n} \phi_n^2 \phi_m^2 dx$$

This creates repulsion when cells overlap.

### Equation of Motion

$$\frac{\partial \phi_n}{\partial t} = -\mathbf{v}_n \cdot \nabla\phi_n - \frac{1}{2}\frac{\delta \mathcal{F}}{\delta \phi_n}$$

The variational derivative expands to:
$$\frac{\delta \mathcal{F}}{\delta \phi_n} = -2\gamma \nabla^2\phi_n + f'(\phi_n) + \text{volume term} + \text{repulsion term}$$

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for complete derivation.

### Key Parameters (types.cuh / types3d.cuh)

| Symbol | Name | Default | Description |
|--------|------|---------|-------------|
| λ | `lambda` | 7 | Interface width (grid points) |
| γ | `gamma` | 1 | Gradient coefficient |
| κ | `kappa` | 10 | Cell-cell repulsion strength |
| μ | `mu` | 1 | Volume constraint strength |
| R | `target_radius` | 49 (2D), 20 (3D) | Target cell radius |
| v_A | `v_A` | 0 | Active motility speed |
| ξ | `xi` | 1500 | Friction coefficient |
| τ | `tau` | 10000 | Reorientation time |

### Motility Models

1. **Run-and-Tumble:** Cells move with constant speed |v_A|, random direction. Direction changes with Poisson rate 1/τ.
2. **Active Brownian Particle (ABP):** Continuous rotational diffusion of polarization direction.

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

```powershell
cd cpp/simulation
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

**Safe mode (GPU memory tracking):**
```powershell
cmake .. -DSAFE_MODE=ON
```

### Running Simulations

**2D simulation:**
```powershell
.\build\bin\Release\cell_sim.exe -n 16 -N 512 -r 30 -t 100 --dt 0.01 -o output_2d
```

**3D simulation:**
```powershell
.\build\bin\Release\cell_sim.exe --3d -n 8 --size 150 -r 20 -t 10 --dt 0.02 -o output_3d
```

**Key CLI options:**
| Option | Description |
|--------|-------------|
| `-n <num>` | Number of cells |
| `-N <size>` | Domain size (2D: NxN) |
| `--size <n>` | Domain size (3D: NxNxN) |
| `-r <radius>` | Cell radius |
| `-t <time>` | End time |
| `--dt <step>` | Time step |
| `-o <dir>` | Output directory |
| `-c <file>` | Load checkpoint |
| `--v-A <f>` | Active motility speed |
| `--3d` | Enable 3D mode |
| `--save-interval <n>` | Steps between VTK saves |

See [WORKFLOW.md](WORKFLOW.md) for complete workflow documentation.

---

## 6. Memory Optimization History

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

### Current Focus
- Further memory reduction by computing integrand_x/y on-the-fly in reduction kernels
- Evaluating tradeoffs between memory savings and computational overhead
- Kernel fusion strategies for 2D (similar to 3D success)

---

## 7. Performance Considerations

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
# NVIDIA Nsight profiling
nsys profile .\cell_sim.exe -n 16 -t 10 -o profile_output
```

---

## 8. 3D Extension Status

### Completed ✅
- Core data structures (types3d.cuh, cell3d.cuh, domain3d.cuh)
- 3D kernels (laplacian, bulk, volume, interaction)
- Integrator3D with fused kernels
- I/O: checkpoints, VTK output
- Bounding box update (fixed critical bug — see TEST_PLAN_3D.md)

### Known Issues
- Large cell counts (>64) require significant GPU memory
- 3D visualization can be slow for large domains

### Testing Results (from TEST_PLAN_3D.md)
| Test | Parameters | Result |
|------|------------|--------|
| Single cell, t=10 | N=240, r=49 | ✅ PASS |
| Single cell, t=100 | N=240, r=49 | ✅ PASS |
| Two cells, t=10 | N=240, r=40 | ✅ PASS |

---

## 9. Common Debugging Scenarios

### NaN Explosion
**Symptoms:** φ suddenly goes to ±infinity, then NaN
**Causes:**
1. Bounding box update bug (fixed in 3D, see TEST_PLAN_3D.md)
2. Time step too large (try reducing dt by 2x)
3. Cell touching domain boundary incorrectly

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

---

## 10. Key Files to Understand

**Start here (in order):**
1. [types.cuh](include/types.cuh) — Parameters and data structures
2. [cell.cuh](include/cell.cuh) — Cell class and phase field initialization
3. [kernels.cuh](include/kernels.cuh) — Kernel function signatures
4. [kernels_optimized_v2.cu](src/kernels_optimized_v2.cu) — Main kernel implementations
5. [integrator.cuh](include/integrator.cuh) — Memory management and stepping

**For 3D work:**
1. [types3d.cuh](include/types3d.cuh) — 3D parameter extensions
2. [cell3d.cuh](include/cell3d.cuh) — 3D cell with bounding box (note: fixed bug here)
3. [kernels3d.cu](src/kernels3d.cu) — 3D kernel implementations

---

## 11. Onboarding Checklist

- [ ] Read this document completely
- [ ] Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for physics equations
- [ ] Build the project successfully
- [ ] Run a simple 2D simulation (1 cell, t=10)
- [ ] Run a simple 3D simulation (1 cell, t=10)
- [ ] Understand buffer slot allocation in kernels
- [ ] Review TEST_PLAN_3D.md for bug history
- [ ] Benchmark baseline performance before making changes

---

## 12. Development Guidelines

### Before Making Changes
1. Understand the affected code path (V2 vs V4, 2D vs 3D)
2. Create a test case that exercises the change
3. Record baseline performance metrics

### After Making Changes
1. Build in both Debug and Release modes
2. Run validation tests (single cell, multi-cell)
3. Benchmark and compare to baseline
4. Update relevant documentation
5. Document the change rationale in commit message

### Code Style
- Use `__host__ __device__` for functions callable from both CPU and GPU
- Prefix device-only pointers with `d_`
- Use `#pragma once` for header guards
- Keep kernels focused — one kernel, one responsibility

---

## 13. Future Work / Open Questions

1. **Active motility (v_A):** Fully implemented in 2D, basic in 3D. MSD analysis pending.
2. **Cell division:** Framework exists (CellState::Dividing) but not implemented.
3. **Adaptive time stepping:** Currently fixed dt, could improve stability.
4. **Multi-GPU:** Not implemented, would require domain decomposition.
5. **Spectral methods:** cuFFT integration for faster Laplacian computation.

---

## 14. Useful References

- **Paper:** (Ask user for the specific paper this model is based on)
- **Phase-field models:** Provatas & Elder, "Phase-Field Methods in Materials Science"
- **CUDA optimization:** NVIDIA CUDA Best Practices Guide
- **Cell migration physics:** Ziebert & Aranson, "Computational approaches to substrate-based cell motility"

---

*Last updated: 2025-12-07*
*For questions, refer to the conversation history or ask the user directly.*
