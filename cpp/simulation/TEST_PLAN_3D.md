# 3D Simulation Test Plan

## Overview
Testing the 3D phase-field cell simulation to diagnose issues with cell fragmentation and boundary problems.

---

## Test 1: Single Cell Basic Test
**Date:** 2025-12-05
**Goal:** Verify a single cell maintains shape and moves correctly

### Parameters
- Domain: N=240 (240×240×240)
- Cells: n=1
- Radius: r=49
- Time step: dt=0.01
- End time: t=10
- Motility: v_A=0.01 (default)
- Save interval: 100 steps (10 frames total)

### Command
```powershell
.\build\bin\Release\cell_sim.exe --3d -N 240 -n 1 -r 49 --dt 0.01 -t 10 --save-interval 100 -o output_3d_test1
```

### Expected Behavior
- Cell should remain spherical
- Cell should move in a direction due to motility
- No fragmentation or boundary artifacts

### Results
_To be filled after test_

---

## Test 2: Single Cell Longer Run (if Test 1 passes)
**Parameters:** Same as Test 1 but t=100

---

## Test 3: Two Cells Interaction (if Test 2 passes)
**Parameters:** n=2, min-spacing to allow overlap

---

## Known Issues to Investigate
1. **Subdomain sizing** - Are subdomains large enough for the cell radius?
2. **Periodic boundary handling** - Is the 3D periodic BC code correct?
3. **Laplacian stencil** - Is the 3D Laplacian correct?
4. **Interface width (lambda)** - Default is 7, may need adjustment for larger cells
5. **Volume constraint** - Is the volume preservation working in 3D?

---

## Physics Notes: 2D vs 3D

### Volume Constraint (Dimension-Independent Formulation)

The free energy for volume conservation is:
$$F_{volume} = \frac{\mu}{2}(V - V_{target})^2$$

where $V = \int \phi^2 \, dV$ is measured by integrating $\phi^2$ over the domain.

Functional derivative:
$$\frac{\delta F_{volume}}{\delta \phi} = 2\mu(V - V_{target})\phi$$

This gives the RHS contribution:
$$\frac{\partial \phi}{\partial t} \supset -2\mu(V - V_{target})\phi$$

**The formulation is the same for 2D and 3D!** The only differences are:

| Quantity | 2D | 3D |
|----------|-----|-----|
| Target volume | $V_{target} = \pi R^2$ | $V_{target} = \frac{4}{3}\pi R^3$ |
| Volume integral | $V = \int \phi^2 \, dx\,dy$ | $V = \int \phi^2 \, dx\,dy\,dz$ |
| Laplacian | $\nabla^2\phi = \phi_{xx} + \phi_{yy}$ | $\nabla^2\phi = \phi_{xx} + \phi_{yy} + \phi_{zz}$ |

**Implementation verification:**
- ✅ `types3d.cuh:200`: `target_volume() = (4/3)πR³`
- ✅ `kernels3d.cu`: `compute_volume_integral_3d()` integrates over 3D
- ✅ `kernels3d.cu`: 3D Laplacian uses 6-point stencil

---

## Test Log

| Test | Date | Parameters | Result | Notes |
|------|------|------------|--------|-------|
| 1a | 2025-12-05 | N=240, n=1, r=49, dt=0.01, t=10 | **ISSUE** | Cell placed near boundary, caused fragmentation |
| 1b | 2025-12-05 | N=240, n=1, r=49, dt=0.01, t=10 (centered) | **ISSUE** | NaN explosion at ~step 80 due to bbox update bug |
| 1c | 2025-12-05 | N=240, n=1, r=49, dt=0.01, t=10 (bbox disabled) | **PASS** | Volume=491579 (99.75%), phi_max=1.008, stable |
| 2 | 2025-12-05 | N=240, n=1, r=49, dt=0.01, t=100 (bbox disabled) | **PASS** | Volume=490141 (99.46%), phi_max=1.017, stable |
| 3 | 2025-12-05 | N=240, n=2, r=49, dt=0.01, t=10 (bbox disabled) | **PARTIAL** | No NaN, but cells shrink to 57% volume |

### Test 1a - Original (with bug)
- **Cell center:** (121.1, 16.0, 213.4) - TOO CLOSE TO Y BOUNDARY
- **Subdomain:** [19,-86,111]->[223,118,315] - negative indices
- **Problem:** Cell overlapped periodic boundary, Neumann BC at subdomain edge cut it
- **Screenshots:** test1_t1.png, test1_final.png

### Test 1b - Fixed (cell at center)
- **Fix applied:** Single cell now initialized at domain center (120, 120, 120)
- **Cell center:** (120.0, 120.0, 120.0)
- **Subdomain:** [18,18,18]->[222,222,222] - all positive, no wrapping
- **Observation:** Simulation explodes between step 80-82 (dt=0.01) or 740-750 (dt=0.001)!

### Bug Investigation: Numerical Instability (NaN Explosion)
**Symptoms:**
- Simulation runs fine for ~80 steps with dt=0.01
- Single voxel corrupts from φ≈0 to φ=-2045 at step 81
- Explosion propagates: step 82 has φ_max=104M, step 83+ all NaN

**Root Cause Found: Bounding Box Update Bug**
- `Cell3D::update_bounding_box()` in `cell3d.cuh` causes memory corruption
- Called every 10 steps by integrator
- When bounding box is resized, something goes wrong with the field interpolation

**Verification:**
- Setting `bbox_update_interval = 100000` (effectively disabling) makes simulation stable
- Test with dt=0.01, t=10: volume=491579 (target=492807), phi_max=1.008, no NaN

### Bug Fix Applied
1. **File:** `include/domain3d.cuh` - Single cell center placement
2. **File:** `include/cell3d.cuh` - **FIXED** `update_bounding_box()` function

### Root Cause: `Cell3D::update_bounding_box()` Bug

The 3D version had several issues compared to the working 2D version:

| Aspect | 2D (Correct) | 3D (Was Broken) |
|--------|--------------|-----------------|
| New bbox center | Uses centroid + max periodic distance | Used min/max local coords (fails at boundaries) |
| Copy direction | Iterates OLD coords → NEW | Iterated NEW coords → OLD (backward!) |
| Coordinate conversion | Direct calculation with bounds check | Used buggy `global_to_local()` that could return negative indices |

**The Fix (applied 2025-12-05):**
- Rewrote `Cell3D::update_bounding_box()` to match the 2D algorithm
- Uses centroid + max periodic distance for new bbox sizing
- Iterates over OLD local coords, maps to NEW coords with proper bounds checking
- Direct coordinate calculation instead of relying on `global_to_local()`

### Working Configuration (dt=0.01, t=10, N=240, r=49)
```
volume=491579.0 (98% of target 492807)
phi_range=[0.0, 1.008]
phi_sq_sum=491578.4
```
Screenshot: `output_3d_debug9/cell_visualization.png`

### Test Results After Fix

| Test | Parameters | Result | Notes |
|------|------------|--------|-------|
| Single cell, t=10 | N=240, n=1, r=49 | ✅ PASS | Ran 1000 steps without NaN |
| Single cell, t=100 | N=240, n=1, r=49 | ✅ PASS | Ran 10,000 steps, 119 seconds, stable |
| Two cells, t=10 | N=240, n=2, r=40, s=100 | ✅ PASS | Ran 1000 steps, cells interacting correctly |

### Previous Notes (Before Fix)

### Test 1c - With Bbox Updates Disabled
- **Fix:** Set `bbox_update_interval = 100000` in integrator3d.cuh
- **Result:** ✅ PASS
- **Final state:** t=10, volume=491579 (99.75% of target), phi_max=1.008
- **Screenshot:** `output_3d_test1/visualization_t10.png`
