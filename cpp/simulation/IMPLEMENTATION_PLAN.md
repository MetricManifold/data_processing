# Phase-Field Cell Simulation - Implementation Plan

## Overview
Multi-cell phase-field simulation with subdomain optimization for CUDA acceleration.

## Physics Model

### Free Energy Functional

$$\mathcal{F} = \mathcal{F}_0 + \mathcal{F}_{int}$$

**Single-cell free energy:**
$$\mathcal{F}_0 = \sum_n \left[ \gamma_n \int dx \int dy \left( (\nabla\phi_n)^2 + \frac{30}{\lambda^2}\phi_n^2(1-\phi_n)^2 \right) + \frac{\mu_n}{\pi R^2}\left(\pi R^2 - \int dx \int dy \, \phi_n^2 \right)^2 \right]$$

**Interaction energy (cell-cell repulsion):**
$$\mathcal{F}_{int} = \frac{30\kappa}{\lambda^2} \int dx \int dy \sum_{n,m \neq n} \phi_n^2 \phi_m^2$$

### Parameters (from paper Table 1)

| Symbol | Name | Value | Notes |
|--------|------|-------|-------|
| λ | Interface width | 7 | Controls interface sharpness |
| γ | Gradient coefficient | 1 | Interface energy |
| κ | Interaction strength | 10 | Cell-cell repulsion |
| μ | Volume constraint strength | 1 | Soft area constraint |
| R | Target cell radius | 49 | Target area = πR² ≈ 7543 |
| ξ | Friction coefficient | 1.5 × 10³ | Drag on cell motion |
| τ | Reorientation time | 10⁴ | Polarization persistence |
| v_A | Active motility speed | 10⁻² | Self-propulsion magnitude |

### Derived Coefficients

| Coefficient | Formula | Value | Used in |
|-------------|---------|-------|---------|
| bulk_coeff | 30/λ² | 30/49 ≈ 0.612 | Bulk potential f'(φ) |
| interaction_coeff | 30κ/λ² | 300/49 ≈ 6.122 | Cell-cell repulsion |
| volume_coeff | μ/(πR²) | 1/7543 ≈ 1.33×10⁻⁴ | Volume constraint |
| motility_coeff | 60κ/(ξλ²) | 600/(1500×49) ≈ 8.16×10⁻³ | Velocity from interactions |

### Equation of Motion

$$\frac{\partial \phi_n}{\partial t} + \mathbf{v}_n \cdot \nabla\phi_n = -\frac{1}{2}\frac{\delta \mathcal{F}}{\delta \phi_n}$$

Where the **motility/advection term** $\mathbf{v}_n \cdot \nabla\phi_n$ represents active cell motion.

### Variational Derivative

The variational derivative $\frac{\delta \mathcal{F}}{\delta \phi_n}$ is computed term by term:

**1. Gradient term:** $\frac{\delta}{\delta \phi_n}\left[\gamma \int (\nabla\phi_n)^2\right] = -2\gamma \nabla^2\phi_n$

**2. Bulk potential:** $f(\phi) = \frac{30}{\lambda^2}\phi^2(1-\phi)^2$

$$f'(\phi) = \frac{\delta}{\delta \phi}\left[\frac{30}{\lambda^2}\phi^2(1-\phi)^2\right] = \frac{60}{\lambda^2}\phi(1-\phi)(1-2\phi)$$

**3. Volume constraint:** $\frac{\delta}{\delta \phi_n}\left[\frac{\mu}{\pi R^2}\left(\pi R^2 - \int\phi_n^2\right)^2\right]$

Let $V = \int \phi^2 \, dx$ and $A = \pi R^2$. Then:
$$\frac{\delta}{\delta \phi}[(A-V)^2] = 2(A-V) \cdot \frac{\delta(A-V)}{\delta\phi} = 2(A-V)(-2\phi) = -4(A-V)\phi$$

So: $\frac{\delta F_{vol}}{\delta \phi} = -\frac{4\mu}{\pi R^2}\left(\pi R^2 - \int\phi^2\right)\phi$

**4. Interaction term:** $\frac{\delta}{\delta \phi_n}\left[\frac{30\kappa}{\lambda^2} \int \sum_{m \neq n} \phi_n^2 \phi_m^2\right] = \frac{60\kappa}{\lambda^2}\phi_n \sum_{m \neq n}\phi_m^2$

### Combined Variational Derivative

$$\frac{\delta \mathcal{F}}{\delta \phi_n} = -2\gamma \nabla^2\phi_n + \frac{60}{\lambda^2}\phi_n(1-\phi_n)(1-2\phi_n) - \frac{4\mu}{\pi R^2}\left(\pi R^2 - \int\phi_n^2\right)\phi_n + \frac{60\kappa}{\lambda^2}\phi_n\sum_{m\neq n}\phi_m^2$$

### Expanded Equation of Motion

$$\frac{\partial \phi_n}{\partial t} = -\mathbf{v}_n \cdot \nabla\phi_n - \frac{1}{2}\left[ -2\gamma \nabla^2\phi_n + \frac{60}{\lambda^2}\phi_n(1-\phi_n)(1-2\phi_n) - \frac{4\mu}{\pi R^2}\left(\pi R^2 - \int\phi_n^2\right)\phi_n + \frac{60\kappa}{\lambda^2}\phi_n\sum_{m\neq n}\phi_m^2 \right]$$

---

## Implementation Phases

### Phase 1: Core Data Structures ✅
**Files:** `types.cuh`, `cell.cuh`, `domain.cuh`

- [ ] `BoundingBox` - subdomain bounds with periodic wrapping
- [ ] `Cell` - phase field on subdomain, centroid, volume integral
- [ ] `Domain` - global domain, cell collection, neighbor detection

### Phase 2: Memory Management
**Files:** `memory.cuh`, `memory.cu`

- [ ] Device memory pools for cell fields
- [ ] Dynamic subdomain resizing
- [ ] Halo/ghost cell management for periodic BC

### Phase 3: CUDA Kernels - Local Terms
**Files:** `kernels_local.cu`

- [ ] Laplacian kernel (5-point stencil, periodic BC within subdomain)
- [ ] Bulk potential: f'(φ) = 4aφ³ + 2bφ - 3cφ²
- [ ] Volume integral reduction (∫φ² dx)
- [ ] Volume constraint contribution: 2λ₁(∫φ²)·φ + λ₂·φ

### Phase 4: CUDA Kernels - Interaction Terms  
**Files:** `kernels_interaction.cu`

- [ ] Bounding box overlap detection
- [ ] Cell-cell repulsion: κφᵢΣⱼφⱼ² (only in overlap regions)
- [ ] Sparse interaction list management

### Phase 5: Time Integration
**Files:** `integrator.cuh`, `integrator.cu`

- [ ] Forward Euler (initial)
- [ ] Semi-implicit scheme (for stability)
- [ ] Adaptive subdomain resizing after each step

### Phase 6: Motility/Velocity Coupling
**Files:** `kernels_interaction.cu` (added), `integrator.cu`

- [x] Cell velocity computation: $\mathbf{v}_n = \frac{60\kappa}{\xi\lambda^2} \int \phi_n (\nabla\phi_n) \sum_{m\neq n} \phi_m^2 \, dx$
- [x] Advection term: vₙ · ∇φₙ (upwind scheme for stability)
- [x] Integration with time stepping

### Phase 7: I/O and Visualization
**Files:** `io.cuh`, `io.cu`

- [ ] Checkpoint save/load
- [ ] VTK output for ParaView
- [ ] Cell tracking data output

### Phase 8: Main Driver
**Files:** `main.cu`, `simulation.cuh`

- [ ] Configuration parsing
- [ ] Initialization (random or from file)
- [ ] Main time loop
- [ ] Diagnostics (energy, cell count, etc.)

---

## Directory Structure
```
cpp/simulation/
├── CMakeLists.txt          # Build system with CUDA
├── IMPLEMENTATION_PLAN.md  # This file
├── include/
│   ├── types.cuh           # Basic types, parameters
│   ├── cell.cuh            # Cell class
│   ├── domain.cuh          # Domain class
│   ├── memory.cuh          # Memory management
│   ├── integrator.cuh      # Time stepping
│   ├── io.cuh              # File I/O
│   └── simulation.cuh      # Top-level simulation
├── src/
│   ├── main.cu             # Entry point
│   ├── memory.cu           # Memory implementation
│   ├── kernels_local.cu    # Local term kernels
│   ├── kernels_interaction.cu  # Cell-cell interaction
│   ├── integrator.cu       # Time integration
│   ├── motility.cu         # Motility terms
│   └── io.cu               # I/O implementation
└── config/
    └── default.json        # Default parameters
```

---

## Build Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17
- (Optional) cuFFT for spectral methods

---

## Current Progress

| Phase | Status | Notes |
|-------|--------|-------|
| 1 | ✅ Complete | types.cuh, cell.cuh, domain.cuh |
| 2 | ⏸️ Deferred | Basic memory in Phase 1, advanced pooling later |
| 3 | ✅ Complete | kernels_local.cu - Laplacian, bulk, volume, gradient |
| 4 | ✅ Complete | kernels_interaction.cu - cell-cell repulsion |
| 5 | ✅ Complete | integrator.cu - Forward Euler |
| 6 | 🟡 Partial | v_I done, v_A (active motility) pending |
| 7 | ✅ Complete | io.cu - VTK, tracking, checkpoints |
| 8 | ✅ Complete | main.cu, simulation.cuh |

## Motility Model

Total velocity: **v_n = v_{n,I} + v_{n,A}**

### v_{n,I} - Interaction velocity ✅
$$\mathbf{v}_{n,I} = \frac{60\kappa}{\xi\lambda^2} \int \phi_n (\nabla\phi_n) \sum_{m \neq n} \phi_m^2 \, dx$$

### v_{n,A} - Active motility (TODO)
- Constant speed: |v_{n,A}| = v_A
- Persistent direction θ_n per cell
- Random reorientation with exponential waiting time: P(t_r) = (1/τ)exp(-t_r/τ)
- Run-and-tumble dynamics → effective diffusion D_eff = v_A²τ/2
- Implementation needs:
  1. Per-cell direction angle θ_n
  2. Per-cell time since last reorientation
  3. Exponential random sampling for reorientation events
  4. cuRAND for stochastic dynamics

---

## Next Steps

1. **Start with Phase 1**: Define core data structures
2. Build minimal working example with 1-2 cells
3. Add complexity incrementally
4. Validate against your existing SymPhas results

Ready to begin Phase 1?
