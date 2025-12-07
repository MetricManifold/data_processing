# Cellular Migration Simulation Runbook

## 1. Project Overview

**Purpose:**  
This project simulates cellular migration using CUDA-accelerated kernels, supporting both 2D and 3D environments. The simulation models cell motility, interactions, and physical constraints, with a focus on high performance and scalability to large cell counts.

**Key Goals:**  
- Achieve efficient, large-scale cell migration simulations in 2D and 3D.
- Optimize memory usage per cell to enable higher cell counts.
- Maintain or improve computational performance as memory optimizations are applied.
- Extend the codebase to robust 3D support.

---

## 2. Codebase Structure

- `cpp/` — Main C++/CUDA source code.
  - `main.cpp` — Entry point for simulation.
  - `kernels_optimized_v2.cu`, `kernels_optimized_v4.cu` — CUDA kernels for cell updates and interactions.
  - `integrator.cu` — Buffer allocation and integration logic.
  - `writemat.h` — Matrix output utilities.
- `cudatest/` — CUDA test code and headers.
- `build/` — Build artifacts and project files.
- `data/`, `output/`, `figures/` — Simulation input/output and results.
- `algorithms.ipynb`, `tests.ipynb` — Analysis and testing notebooks.

---

## 3. Simulation Logic

- Each cell is represented by a set of state variables and work buffers.
- Kernels update cell states, compute interactions, and perform reductions.
- Motility models include Run-and-Tumble and Active Brownian Particle (ABP).
- Buffer slots store intermediate results (laplacian, gradients, constraints, etc.).

---

## 4. Memory Optimization History

### Initial State
- 2D simulation used 12 work buffers per cell.
- 3D simulation used 7 work buffers per cell.

### Recent Optimizations
- 2D buffer reduced from 12 to 9 slots by eliminating unused buffers and remapping kernel accesses.
- All relevant kernel code updated and validated (V2 and V4 paths).
- Simulation runs confirmed correct output and performance.

### Current Focus
- Further memory reduction by eliminating integrand_x/y and phi_sq buffers.
- Considering on-the-fly computation in reduction kernels to save memory.
- Evaluating tradeoffs between memory savings and computational overhead.

---

## 5. Performance Considerations

- Performance is critical; all changes must be benchmarked.
- Memory bandwidth vs. compute tradeoff: Reducing buffers may help if memory-bound, but hurt if compute-bound.
- Kernel fusion and register pressure must be considered when refactoring.
- Always validate changes with both build and simulation runs.

---

## 6. 3D Extension Challenges

- 3D simulations require more memory per cell due to additional state variables.
- Extending to higher cell counts is limited by GPU memory.
- Buffer layout and kernel logic must be carefully adapted for 3D.
- Planned reduction: 3D buffer from 7 to 5 slots (not yet implemented).

---

## 7. Onboarding Checklist

1. **Understand the simulation logic and buffer usage.**
2. **Review recent code changes for buffer reduction (see `integrator.cu`, `kernels_optimized_v2.cu`, `kernels_optimized_v4.cu`).**
3. **Benchmark performance after any change.**
4. **Trace buffer slot usage before making further reductions.**
5. **Coordinate with the team on 3D extension and memory optimization strategies.**
6. **Document all changes and rationale for future agents.**

---

## 8. Next Steps

- Prototype kernel fusion/on-the-fly computation to eliminate more buffers.
- Benchmark and compare performance/memory usage.
- Implement 3D buffer reduction and validate with large cell counts.
- Continue to document all findings and decisions.

---

## 9. Key Takeaways

- Memory optimization is ongoing and vital for scaling.
- Performance must be measured after every change.
- 3D support is a major goal, with memory as the main bottleneck.
- All buffer changes require careful kernel code updates and validation.

---

This runbook should enable a new agent to quickly understand the project’s purpose, current state, and priorities. For any code changes, always benchmark and document thoroughly.
