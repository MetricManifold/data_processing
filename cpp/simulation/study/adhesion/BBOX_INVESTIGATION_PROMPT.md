# Bbox Pool Slot Overflow Investigation

## The Problem

When running the cell simulation with gradient-coupling adhesion (`--adhesion J` where J > 0), many cells emit this warning every timestep:

```
WARNING: Cell NNN bbox capped to 214x214 (45796) to fit pool slot 45796.
Consider increasing safety margin in compute_max_page_size().
```

This means **every cell's bounding box is being clamped** to fit the GPU memory pool slot. The capping is soft (no crash, no reallocation), but it means the per-cell phi field is computed on a truncated bounding box. This could cause:

1. Interface gradient artifacts at the bbox boundary (adhesion force truncated)
2. Slow accumulation of mass conservation error
3. Performance overhead from the capping path (branches, fprintf per step)

Despite the warnings, **sanity checks show no visible pathology**: cells remain well-separated, phi stays bounded [0,1] with zero overlap, cell count is conserved at 288 over 50,000 TU of production time. But the bbox capping is systematic, not occasional — it hits nearly all 288 cells at every step.

## Root Cause

The function `compute_max_page_size()` in `integrator.cu` (line 316) computes the maximum possible bounding box side length:

```cpp
int halo = params.halo_width;                                   // = 4
int adaptive_margin = static_cast<int>(2.0f * params.lambda) + halo;  // = 18
int overshoot = static_cast<int>(0.25f * adaptive_margin);            // = 4
int max_dist = static_cast<int>(params.target_radius + 3.0f * params.lambda) + 1; // = 71
int max_half = max_dist + adaptive_margin + overshoot + 10;           // = 103
int max_side = 2 * max_half + 2 * halo;                              // = 214
// pool_slot_size = 214 * 214 = 45796
```

With R=49, λ=7, halo=4, this gives **exactly 214×214 = 45796** — zero headroom.

The problem: **this formula doesn't account for adhesion**. When adhesion is active ($J > 0$), the effective interface stiffness at shared contacts is reduced from $\gamma$ to $\gamma - J/2$. This widens the interface:

$$\lambda_\text{eff} = \lambda \sqrt{\frac{\gamma}{\gamma - J/2}}$$

| J (--adhesion) | $\tilde{J}$ | $\lambda_\text{eff}$ | max_dist needed |
|---|---|---|---|
| 0 | 0 | 7.0 | 70 |
| 0.5 | 0.25 | 8.1 | 73 |
| 1.0 | 0.50 | 9.9 | 79 |
| 1.5 | 0.75 | 14.0 | 91 |

At J=1.5, `max_dist` should be ~91, not 71. That's 20 extra pixels on each side → the pool slot should be ~254×254 = 64516, not 214×214 = 45796. The current formula under-allocates by ~41%.

## Files to Investigate

| File | Lines | What |
|------|-------|------|
| `src/integrator.cu` | 316-335 | `compute_max_page_size()` — **the fix goes here** |
| `src/kernels_solver.cu` | 1225-1245 | Bbox capping logic (where warning is emitted) |
| `include/types.cuh` | ~63 | `SimParams` struct — check if `adhesion_J` is accessible |
| `include/cell.cuh` | 296+ | Bbox update logic in `update_bbox()` |

## Proposed Fix

Modify `compute_max_page_size()` to account for adhesion-widened interfaces:

```cpp
size_t Integrator::compute_max_page_size(const SimParams &params) {
  int halo = params.halo_width;
  
  // Effective interface width: adhesion reduces gradient stiffness at shared contacts
  float lambda_eff = params.lambda;
  if (params.adhesion_J > 0.0f && params.gamma > params.adhesion_J / 2.0f) {
    lambda_eff = params.lambda * sqrtf(params.gamma / (params.gamma - params.adhesion_J / 2.0f));
  }
  
  int adaptive_margin = static_cast<int>(2.0f * lambda_eff) + halo;
  int overshoot = static_cast<int>(0.25f * adaptive_margin);
  int max_dist = static_cast<int>(params.target_radius + 3.0f * lambda_eff) + 1;
  int max_half = max_dist + adaptive_margin + overshoot + 10;
  int max_side = 2 * max_half + 2 * halo;
  return static_cast<size_t>(max_side) * max_side;
}
```

This increases GPU memory usage per cell proportional to $\lambda_\text{eff}^2$, but since it's allocated once and never reallocated, the cost is just ~40% more memory at the strongest adhesion (J=1.5 with γ=1).

## Validation

After the fix:
1. Run the two-cell test at J=1.5 (strongest adhesion within stability bound) and confirm zero warnings
2. Run a 288-cell Phase 0 at Jk_0.15 and confirm zero warnings
3. Compare mass conservation before/after fix
4. Check that the Phase 1/2 production runs are faster (no capping overhead, no fprintf per step)

## Context

- Current adhesion study parameters: N=288, R=49, L=1562, γ=1, κ=10, λ=7
- Adhesion range: J ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5} (Jtilde ∈ {0, 0.125, ..., 0.75})
- The warning appears for ALL J > 0 values to varying degrees
- Production runs (Phase 1/2) on narval with J > 0 are ~2-10× slower than J=0; part of this is inherent (Laplacian coupling cost), but the bbox capping overhead may contribute
- Full adhesion study docs: `study/adhesion/` + `.github/instructions/adhesion-study.instructions.md`
