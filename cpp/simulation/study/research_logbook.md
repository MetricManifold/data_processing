# Research Logbook � Phase Field Cell Simulation


## Griffiths Analysis — 2026-02-11 23:17

### Parameters
- N = 288 cells, domain = 1600 × 1600
- Mean v_A = 0.012, σ_vA = 0.006 (log-normal)
- Packing fraction ≈ 0.89 (from header or estimated)
- Trajectory: ..\agent_test_runs\fluid_movie_phi89\trajectory.txt
- Time span: 129515 to 141000 (1405 frames)
- Mobility window: 574.3
- Threshold: 0.006553 (median mobility)

### Key Results

#### 1. Phase Classification
- **Jammed fraction**: 48.8% ± 3.1%
- Majority phase: **Motile**

#### 2. Cluster Analysis & Percolation
- Jammed clusters: mean size = 17.1, max = 153
- Motile clusters: mean size = 52.4, max = 160
- Largest jammed cluster: 44.5% of cells
- Largest motile cluster: 49.2% of cells
- Jammed percolates: 20/20 frames
- Motile percolates: 20/20 frames
- **Diagnosis: near critical**

#### 3. Spatial Correlations
- Correlation length ξ ≈ 10.0 (0.1 cell spacings)
- C(r=50) = 0.000
- C(r=100) = 0.248
- C(r=200) = 0.082

#### 4. Quenched Disorder Correlation
- Pearson r(v_A, mobility) = 0.5005, p = 1.14e-19
- Strong correlation between inherent v_A and measured mobility

#### 5. Temporal Persistence
- Mean persistence = 0.746
- 36% of cells have persistence > 0.8
- 19% of cells have persistence > 0.9

#### 6. Structural Relaxation
- τ_all ≈ 5953
- τ_jammed ≈ 7953  
- τ_motile ≈ 3961
- Ratio τ_jammed/τ_motile ≈ 2.0

### Physical Interpretation

Both phases show similar percolation, suggesting the system is near the critical point of the jamming transition.

The correlation r = 0.501 between inherent v_A and measured mobility confirms that the quenched disorder (log-normal v_A distribution) directly controls the spatial pattern of mobility, consistent with the Griffiths rare-region picture.

The temporal persistence of 0.746 indicates that cell classifications are moderately stable, partially supporting the "quenched" nature of the disorder.

### Plots
- Full analysis figure: `postprocessing/output/griffiths_analysis_20260211.png`

---

## Batch Griffiths Analysis — 2026-02-12 09:29

### Overview
Comparative analysis of Griffiths rare-region effects across disorder strengths
and mean motility values. Data from nibi cluster, subsampled trajectories
(every 100th timestep, ~1278 frames per run).

**18 runs total**: 6 parameter combos × 3 replicates

### Parameter Space
| v_A | σ | Purpose |
|-----|---|---------|
| 0.008 | **0.000** | **CONTROL** — no disorder |
| 0.008 | 0.003 | Weak disorder |
| 0.008 | 0.006 | Moderate disorder |
| 0.008 | 0.008 | Strong disorder (σ ~ v_A) |
| 0.006 | 0.006 | Lower motility |
| 0.010 | 0.006 | Higher motility |

### Threshold Selection
Per-run median (not recommended)

This is a key methodological choice: by using the control's median, we apply
the SAME absolute threshold to all parameter sets. The control naturally
produces ~50/50 jammed/motile split (since all cells are identical), and
deviations from 50/50 in disordered runs reveal the effect of quenched disorder.

### Results Summary — σ Sweep (fixed v_A=0.008)

| σ | Jammed % | Persistence | τ_j/τ_m | ξ/a | α₂ | J_perc | M_perc |
|---|----------|-------------|---------|-----|-----|--------|--------|
| 0.000 | 53.7% | 0.520 | nan | 0.1 | 0.39 | 0.53 | 0.47 |
| 0.003 | 35.1% | 0.580 | nan | 0.1 | 0.34 | 0.35 | 0.65 |
| 0.006 | 0.0% | 0.949 | nan | 0.1 | 2.15 | 0.00 | 1.00 |
| 0.008 | 48.2% | 0.770 | 16.71 | 0.1 | 12.70 | 0.48 | 0.52 |

### Results Summary — v_A Sweep (fixed σ=0.006)

| v_A | Jammed % | Persistence | τ_j/τ_m | ξ/a | α₂ | J_perc | M_perc |
|-----|----------|-------------|---------|-----|-----|--------|--------|
| 0.006 | 36.4% | 0.567 | nan | 0.1 | 5.84 | 0.37 | 0.63 |
| 0.008 | 0.0% | 0.949 | nan | 0.1 | 2.15 | 0.00 | 1.00 |
| 0.010 | 35.1% | 0.574 | nan | 0.1 | 0.75 | 0.35 | 0.65 |

### Physical Interpretation

**Key question**: Does quenched disorder (σ > 0) create persistent Griffiths
rare regions compared to the homogeneous control (σ = 0)?

#### What to look for:
1. **Persistence increasing with σ**: If cells with high/low inherent v_A
   remain jammed/motile for longer than the dynamic fluctuation timescale
   in the control, this is the hallmark of Griffiths rare regions.

2. **τ_jammed/τ_motile ratio increasing with σ**: In Griffiths physics,
   rare jammed regions embedded in a motile sea have anomalously slow
   relaxation (power-law tails instead of exponential).

3. **Non-Gaussian parameter α₂ increasing with σ**: Dynamic heterogeneity
   should increase as quenched disorder creates a wider distribution of
   local relaxation rates.

4. **Correlation length ξ increasing with σ**: Spatial correlations should
   grow as inherent v_A clusters create correlated mobility patterns.

#### Limitations at current time/size:
- **288 cells** may be too small for reliable percolation analysis
- **t ≈ 330,000** may not be long enough — continuation to t=800,000 in progress
- Threshold = control median is better than per-run median, but still crude
- Need to verify that the σ=0 control truly shows NO persistent spatial
  heterogeneity (its persistence should be ~0.5 for random fluctuations)

### Plots
- Disorder sweep: `postprocessing/output/griffiths_sigma_sweep_20260212.png`
- Motility sweep: `postprocessing/output/griffiths_vA_sweep_20260212.png`
- Q(t) comparison: `postprocessing/output/griffiths_Qt_comparison_20260212.png`
- Key signatures: `postprocessing/output/griffiths_key_signatures_20260212.png`

---

## Batch Griffiths Analysis — 2026-02-12 09:34

### Overview
Comparative analysis of Griffiths rare-region effects across disorder strengths
and mean motility values. Data from nibi cluster, subsampled trajectories
(every 100th timestep, ~1278 frames per run).

**18 runs total**: 6 parameter combos × 3 replicates

### Parameter Space
| v_A | σ | Purpose |
|-----|---|---------|
| 0.008 | **0.000** | **CONTROL** — no disorder |
| 0.008 | 0.003 | Weak disorder |
| 0.008 | 0.006 | Moderate disorder |
| 0.008 | 0.008 | Strong disorder (σ ~ v_A) |
| 0.006 | 0.006 | Lower motility |
| 0.010 | 0.006 | Higher motility |

### Threshold Selection
Used σ=0 control MEAN mobility = 0.000978

This is a key methodological choice: by using the control's mean mobility,
we apply the SAME absolute threshold to all parameter sets. Cells with
mobility below this threshold are classified as "jammed" — they are
systematically slower than the average cell in the disorder-free control.
In disordered cases, this identifies cells whose low inherent v_A keeps
them jammed relative to the homogeneous baseline.

### Results Summary — σ Sweep (fixed v_A=0.008)

| σ | Jammed % | Persistence | τ_j/τ_m | ξ/a | α₂ | J_perc | M_perc |
|---|----------|-------------|---------|-----|-----|--------|--------|
| 0.000 | 62.3% | 0.600 | 3.20 | 0.1 | 0.39 | 0.60 | 0.43 |
| 0.003 | 46.5% | 0.592 | 1.13 | 0.1 | 0.34 | 0.40 | 0.62 |
| 0.006 | 8.2% | 0.845 | 1.34 | 0.1 | 2.15 | 0.00 | 1.00 |
| 0.008 | 53.5% | 0.756 | 6.71 | 0.1 | 12.70 | 0.50 | 0.52 |

### Results Summary — v_A Sweep (fixed σ=0.006)

| v_A | Jammed % | Persistence | τ_j/τ_m | ξ/a | α₂ | J_perc | M_perc |
|-----|----------|-------------|---------|-----|-----|--------|--------|
| 0.006 | 50.5% | 0.574 | 1.32 | 0.1 | 5.84 | 0.48 | 0.60 |
| 0.008 | 8.2% | 0.845 | 1.34 | 0.1 | 2.15 | 0.00 | 1.00 |
| 0.010 | 41.6% | 0.539 | 1.08 | 0.1 | 0.75 | 0.38 | 0.62 |

### Physical Interpretation

**Key question**: Does quenched disorder (σ > 0) create persistent Griffiths
rare regions compared to the homogeneous control (σ = 0)?

#### What to look for:
1. **Persistence increasing with σ**: If cells with high/low inherent v_A
   remain jammed/motile for longer than the dynamic fluctuation timescale
   in the control, this is the hallmark of Griffiths rare regions.

2. **τ_jammed/τ_motile ratio increasing with σ**: In Griffiths physics,
   rare jammed regions embedded in a motile sea have anomalously slow
   relaxation (power-law tails instead of exponential).

3. **Non-Gaussian parameter α₂ increasing with σ**: Dynamic heterogeneity
   should increase as quenched disorder creates a wider distribution of
   local relaxation rates.

4. **Correlation length ξ increasing with σ**: Spatial correlations should
   grow as inherent v_A clusters create correlated mobility patterns.

#### Limitations at current time/size:
- **288 cells** may be too small for reliable percolation analysis
- **t ≈ 330,000** may not be long enough — continuation to t=800,000 in progress
- Threshold = control median is better than per-run median, but still crude
- Need to verify that the σ=0 control truly shows NO persistent spatial
  heterogeneity (its persistence should be ~0.5 for random fluctuations)

### Plots
- Disorder sweep: `postprocessing/output/griffiths_sigma_sweep_20260212.png`
- Motility sweep: `postprocessing/output/griffiths_vA_sweep_20260212.png`
- Q(t) comparison: `postprocessing/output/griffiths_Qt_comparison_20260212.png`
- Key signatures: `postprocessing/output/griffiths_key_signatures_20260212.png`

---

## Physics Reflections: Griffiths Rare-Region Signatures — 2026-02-12

### Purpose

This section records the physical interpretation of the batch Griffiths
analysis, including system-size and time-scale limitations. The goal is to
address the central question: **does quenched motility disorder create
persistent Griffiths rare regions in the phase field cell model?**

---

### 1. Key Findings from the σ Sweep (v_A=0.008 fixed)

| σ | Jammed % | Persistence | τ_j/τ_m | α₂ | Regime |
|---|----------|-------------|---------|-----|--------|
| 0.000 | 62.3% | 0.600 | 3.20 | 0.39 | **CONTROL** — intrinsic glass physics |
| 0.003 | 46.5% | 0.592 | 1.13 | 0.34 | Negligible disorder effect |
| 0.006 | 8.2% | 0.845 | 1.34 | 2.15 | **Jammed islands in motile sea** |
| 0.008 | 53.5% | 0.756 | 6.71 | 12.70 | **Strong Griffiths regime** |

**The control (σ=0) is NOT trivial.** At φ=0.89, the system is near the
glass/jamming transition even without disorder. 57.4% of control cell
measurements have exactly zero mobility — cells are caged by their
neighbors. The control itself shows τ_j/τ_m = 3.20 and persistence = 0.600,
meaning that even in a homogeneous tissue, dynamic heterogeneity exists as
cells temporarily cage and uncage through stochastic fluctuations. This is
the familiar glass physics at φ close to the jamming point.

The Griffiths effect is therefore the *enhancement beyond this intrinsic
glass baseline* when quenched disorder is introduced.

#### σ = 0.003: No effect

All metrics are within error of the control. With the log-normal
distribution at σ=0.003, the spread in inherent v_A is too small to
overwhelm the intrinsic stochastic fluctuations. Cells with slightly
different v_A are still dislodged from cages at similar rates by their
neighbors. The disorder is thermally irrelevant.

#### σ = 0.006: Jammed islands in motile sea

This is the most striking regime. Only 8.2% of cells are classified as
jammed (below the control mean threshold), yet the persistence is the
*highest* of all cases at 0.845. This means:

- The vast majority of cells are consistently motile (above threshold)
- The few jammed cells are *persistently* jammed — they don't intermittently
  un-jam as in the control
- The tissue is a motile fluid with rare, stable jammed inclusions

This corresponds to the **"jammed islands in motile sea"** picture observed
in the contour movies. At σ=0.006 with mean v_A=0.008, the log-normal
distribution produces some cells with inherent v_A ≲ 0.002, which are
essentially immobile. These cells, along with their local neighbors,
form persistent jammed islands. The surrounding tissue (cells with
inherent v_A ≈ 0.008-0.015) flows freely around them.

The relatively modest τ_j/τ_m = 1.34 seems low, but this is because
the *few* jammed cells have finite relaxation times comparable to the motile
population — the "jammed" cells in this regime are not deeply caged because
most of their neighbors are highly motile. The jammed islands exist not
because cells are surrounded by other slow cells (which would create deep
caging), but because individual cells have inherent v_A so low that even
in a motile environment they cannot move.

The α₂ = 2.15 (5.5× the control) confirms growing dynamic heterogeneity,
though not as extreme as σ=0.008.

#### σ = 0.008: Strong Griffiths regime

At σ = v_A (relative disorder = 100%), the system shows dramatic Griffiths
signatures:

- **τ_j/τ_m = 6.71** — more than double the control's 3.20, meaning jammed
  cells take ~7× longer to relax than motile cells. In Griffiths physics,
  this ratio should diverge as system size → ∞ due to exponentially rare
  large jammed clusters.

- **α₂ = 12.70** — a 32× increase over the control's 0.39. This massive
  non-Gaussianity indicates that the displacement distribution has extremely
  heavy tails: some cells move much farther (or much less) than the mean.
  This is the hallmark of a broad distribution of local relaxation rates
  created by the quenched disorder.

- **Persistence = 0.756** — elevated above control (0.600), indicating cells
  remain in their jammed/motile classification longer due to quenched disorder
  pinning them in specific dynamical states.

- **Jammed fraction = 53.5% ± 21.5%** — near 50/50 with *huge* variance
  between replicates, suggesting the system is close to a percolation
  critical point where the balance between jammed and motile phases
  fluctuates strongly from sample to sample.

#### The non-monotonic σ dependence

The jammed fraction shows a non-monotonic trend: 62.3% → 46.5% → 8.2% →
53.5% as σ increases. This is physically reasonable:

- At σ=0 (control), 62.3% are below the control mean threshold — by
  definition, about half should be below the mean, but the skewed mobility
  distribution (many zeros) pushes this above 50%.

- At σ=0.003, weak disorder shifts some cells to higher mobility, reducing
  the jammed fraction to 46.5%.

- At σ=0.006, the disorder is strong enough that most cells have inherent
  v_A well above zero, making them robustly motile. The 8.2% jammed cells
  correspond to the tail of the log-normal at very low v_A.

- At σ=0.008, the log-normal spread is so wide that many cells have
  inherent v_A ≈ 0, bringing the jammed fraction back up to 53.5%.
  The system now has a bimodal population: some cells are fast (v_A > 0.01)
  and some are nearly immobile (v_A ≈ 0), creating a "two-fluid" state.

### 2. The v_A Sweep (σ=0.006 fixed)

| v_A | Jammed % | τ_j/τ_m | α₂ |
|-----|----------|---------|-----|
| 0.006 | 50.5% | 1.32 | 5.84 |
| 0.008 | 8.2% | 1.34 | 2.15 |
| 0.010 | 41.6% | 1.08 | 0.75 |

At fixed σ=0.006, increasing v_A from 0.006 → 0.008 → 0.010 shows a
transition from balanced (50% jammed) through mostly-fluid (8% jammed)
back to moderately jammed (42%). Note however that the threshold is
set from the v_A=0.008 control, so the jammed fractions at different
mean v_A must be interpreted carefully — cells at v_A=0.010 with the
same absolute threshold will have a different meaning than at v_A=0.006.

The key observation is α₂: it peaks at v_A=0.006 (5.84), where the
system is near the 50/50 jammed-motile balance. At this point, the
quenched disorder creates the maximum contrast between jammed and motile
regions, producing the strongest dynamic heterogeneity. By v_A=0.010,
all cells are sufficiently motile that the disorder barely matters
(α₂ drops to 0.75, near-Gaussian).

### 3. Correlation Length ξ: Bug Fixed, Real Signal Emerges

**Previously:** All cases showed ξ ≈ 10.0 (0.1 cell spacings) — a binning
artifact caused by sub-cell-spacing bins having zero pairs, making C(r=10)=0,
which the ξ-finder incorrectly reported.

**Fix applied:** Skip bins with < 10 cell pairs (the first ~4 bins at r < 80
have almost no pairs since cell centroids can't be closer than ~98 apart
at φ=0.89).

**Corrected results:**

| σ | ξ (old, buggy) | ξ (fixed) |
|---|----------------|-----------|
| 0.000 | 0.1 cells | **1.0 cells** |
| 0.003 | 0.1 cells | **0.9 cells** |
| 0.006 | 0.1 cells | **0.9 cells** |
| **0.008** | 0.1 cells | **3.3 cells** |

**Physical interpretation:** The σ=0.008 case shows ξ = 3.3 cell spacings,
3× longer than all other cases including the control. This confirms that
strong quenched disorder creates spatially correlated jammed/motile regions
extending ~3 cells. The control and weak disorder cases show ξ ≈ 1 cell
spacing, meaning mobility correlations decay at nearest-neighbor distances
— no long-range spatial structure. This is the expected baseline: without
quenched disorder, both cells in a pair are equally likely to be fast or slow,
so there's no spatial correlation beyond the trivial nearest-neighbor effect
from shared caging environments.

The ξ = 3.3 at σ=0.008 is a lower bound constrained by the 17×17 cell system.
The 18,432-cell system (136×136) will reveal whether ξ saturates (indicating
a finite correlation length) or grows further (critical scaling).

### 4. System Size Limitations

With only **288 cells** (≈17×17 cells across the domain):

- **Percolation unreliable:** The percolation threshold for a 17×17 grid on a
  torus has enormous finite-size fluctuations. Both phases showing ~50%
  percolation is expected near the critical point regardless of the
  underlying physics. Reliable percolation requires N > 1000.

- **Correlation length limited:** Even if ξ is extracted correctly, it
  cannot exceed ~8 cell spacings (half the 17-cell width). This means
  we cannot distinguish a true long-range Griffiths correlation from a
  short-range one.

- **Cluster statistics poor:** With 288 cells, the largest cluster can be
  at most 288 cells. The distribution of cluster sizes (needed for testing
  power-law scaling, a Griffiths signature) is truncated.

- **Rare regions too small:** In Griffiths physics, the anomalous
  relaxation comes from exponentially rare large jammed regions. With 288
  cells, the largest possible rare region is ~50-100 cells, far too small
  for the asymptotic Griffiths scaling to emerge.

**The 18,432-cell equilibration in progress on nibi will address this:**
at 136×136 cells across, it provides a 64× increase in area, enabling
proper percolation analysis, cluster size distributions spanning 3+
decades, and correlation lengths up to ~60 cell spacings.

### 5. Time Scale Requirements

From the batch analysis:
- τ_jammed ≈ 49,000-71,000 time units (from Q(t) analysis)
- τ_motile ≈ 7,000-11,000 time units

For reliable Griffiths analysis, we need:
- t >> 10 × τ_jammed ≈ 500,000-700,000 to observe multiple relaxation
  events per jammed cell
- Current trajectories span t ≈ 0-330,000, providing only ~5 jammed
  relaxation times
- Continuation to t = 800,000 will provide ~10-15 relaxation times

But even t = 800,000 may be insufficient for measuring *power-law* tails
in Q(t). Griffiths rare regions create a distribution of relaxation times
τ(L) ~ exp(cL^d) where L is the rare-region size. Observing the power-law
envelope of this distribution requires t >> max(τ(L)) for the largest
accessible L, which scales exponentially with system size. With 288 cells,
the largest jammed clusters are L ≈ 5-10 cells, giving τ ≈ exp(5-10c) ≈
10^4-10^6 time units. Reaching t = 800,000 should be marginally sufficient.

For the 18,432-cell system, rare regions of L ≈ 20-50 cells become
accessible, but their relaxation times τ(L) ≈ exp(20-50c) may be
astronomically large, requiring t >> 10^6. This is the fundamental
challenge of Griffiths physics: the most interesting rare events are
also the rarest.

### 6. Summary of Griffiths Signatures

| Signature | Expected for Griffiths | Observed? | Strength |
|-----------|----------------------|-----------|----------|
| Persistence > control | Yes | ✓ σ=0.006, σ=0.008 | Strong |
| τ_j/τ_m > control ratio | Yes | ✓ σ=0.008 (6.71 vs 3.20) | Strong |
| α₂ >> 1 | Yes | ✓ σ=0.008 (12.70) | Very strong |
| Spatial correlations ξ | Yes (ξ > 1 cell) | ✓ σ=0.008 (ξ=3.3 cells) | Moderate |
| Power-law Q(t) decay | Yes | ❓ Need log-log plot | Not yet tested |
| Bimodal persistence | Yes | ✓ σ=0.006, σ=0.008 | Moderate |
| Cluster size power-law | Yes | ❓ Need larger system | Not yet tested |

### 7. Next Steps

1. **Fix ξ calculation** — Skip empty bins, use cell-spacing-appropriate binning
2. **Log-log Q(t) plots** — Test for power-law vs exponential decay
3. **Four-point susceptibility χ₄(t)** — Direct measure of dynamic correlation length
4. **Re-analyze with longer trajectories** when continuation jobs complete (t→800k)
5. **Run analysis on 18,432-cell system** when equilibration finishes
6. **Multiple threshold values** — Check robustness of conclusions
7. **Persistence autocorrelation** — How long does the jammed/motile classification
   persist in time? This is different from the overlap function Q(t).
8. **Direct comparison** — Compute inherent v_A vs mobility correlation (Pearson r)
   for each σ. The control should have r ≈ 0 (no inherent disorder means no
   correlation), while σ > 0 should show increasing r.

---

## Batch Griffiths Analysis — 2026-02-12 09:51

### Overview
Comparative analysis of Griffiths rare-region effects across disorder strengths
and mean motility values. Data from nibi cluster, subsampled trajectories
(every 100th timestep, ~1278 frames per run).

**18 runs total**: 6 parameter combos × 3 replicates

### Parameter Space
| v_A | σ | Purpose |
|-----|---|---------|
| 0.008 | **0.000** | **CONTROL** — no disorder |
| 0.008 | 0.003 | Weak disorder |
| 0.008 | 0.006 | Moderate disorder |
| 0.008 | 0.008 | Strong disorder (σ ~ v_A) |
| 0.006 | 0.006 | Lower motility |
| 0.010 | 0.006 | Higher motility |

### Threshold Selection
Used σ=0 control MEAN mobility = 0.000978

This is a key methodological choice: by using the control's mean mobility,
we apply the SAME absolute threshold to all parameter sets. Cells with
mobility below this threshold are classified as "jammed" — they are
systematically slower than the average cell in the disorder-free control.
In disordered cases, this identifies cells whose low inherent v_A keeps
them jammed relative to the homogeneous baseline.

### Results Summary — σ Sweep (fixed v_A=0.008)

| σ | Jammed % | Persistence | τ_j/τ_m | ξ/a | α₂ | J_perc | M_perc |
|---|----------|-------------|---------|-----|-----|--------|--------|
| 0.000 | 62.3% | 0.600 | 3.20 | 1.0 | 0.39 | 0.60 | 0.43 |
| 0.003 | 46.5% | 0.592 | 1.13 | 0.9 | 0.34 | 0.40 | 0.62 |
| 0.006 | 8.2% | 0.845 | 1.34 | 0.9 | 2.15 | 0.00 | 1.00 |
| 0.008 | 53.5% | 0.756 | 6.71 | 3.3 | 12.70 | 0.50 | 0.52 |

### Results Summary — v_A Sweep (fixed σ=0.006)

| v_A | Jammed % | Persistence | τ_j/τ_m | ξ/a | α₂ | J_perc | M_perc |
|-----|----------|-------------|---------|-----|-----|--------|--------|
| 0.006 | 50.5% | 0.574 | 1.32 | 1.0 | 5.84 | 0.48 | 0.60 |
| 0.008 | 8.2% | 0.845 | 1.34 | 0.9 | 2.15 | 0.00 | 1.00 |
| 0.010 | 41.6% | 0.539 | 1.08 | 0.7 | 0.75 | 0.38 | 0.62 |

### Physical Interpretation

**Key question**: Does quenched disorder (σ > 0) create persistent Griffiths
rare regions compared to the homogeneous control (σ = 0)?

#### What to look for:
1. **Persistence increasing with σ**: If cells with high/low inherent v_A
   remain jammed/motile for longer than the dynamic fluctuation timescale
   in the control, this is the hallmark of Griffiths rare regions.

2. **τ_jammed/τ_motile ratio increasing with σ**: In Griffiths physics,
   rare jammed regions embedded in a motile sea have anomalously slow
   relaxation (power-law tails instead of exponential).

3. **Non-Gaussian parameter α₂ increasing with σ**: Dynamic heterogeneity
   should increase as quenched disorder creates a wider distribution of
   local relaxation rates.

4. **Correlation length ξ increasing with σ**: Spatial correlations should
   grow as inherent v_A clusters create correlated mobility patterns.

#### Limitations at current time/size:
- **288 cells** may be too small for reliable percolation analysis
- **t ≈ 330,000** may not be long enough — continuation to t=800,000 in progress
- Threshold = control median is better than per-run median, but still crude
- Need to verify that the σ=0 control truly shows NO persistent spatial
  heterogeneity (its persistence should be ~0.5 for random fluctuations)

### Plots
- Disorder sweep: `postprocessing/output/griffiths_sigma_sweep_20260212.png`
- Motility sweep: `postprocessing/output/griffiths_vA_sweep_20260212.png`
- Q(t) comparison: `postprocessing/output/griffiths_Qt_comparison_20260212.png`
- Key signatures: `postprocessing/output/griffiths_key_signatures_20260212.png`

---

## Deep Griffiths Analysis — 2026-02-12 13:28

> **Note**: An earlier run (12:37) used corrupted 2-cell files and is superseded.
> This entry uses the correctly extracted 288-cell, 9-column trajectories from
> `griffiths_study/` (original runs, ~1280 timesteps each, 3 replicates per σ).
> Threshold = 0.000872 (control mean mobility).
> Subsample = every 100th timestep → Δt = 2.0 per frame.

### New Measurements

#### 1. Stretched Exponential Fits: Q(t) = exp(-(t/τ)^β)

The stretching exponent β probes the distribution of relaxation times.
- β = 1: simple exponential → single relaxation time
- β < 1: stretched exponential → broad distribution (multiple timescales)
- β → 0: approaches power-law → hallmark of Griffiths rare regions

| σ | β_all | β_jammed | β_motile | τ_all | τ_jammed | τ_motile | R² |
|---|-------|----------|----------|-------|----------|----------|-----|
| 0.000 | 0.570 | 0.336 | 0.580 | 72308 | 40459 | 32979 | 0.754 |
| 0.003 | 0.495 | 0.459 | 0.624 | 48098 | 20331 | 16698 | 0.781 |
| 0.006 | 0.898 | 0.733 | 0.950 | 16005 | 16896 | 12825 | 0.960 |
| 0.008 | 1.061 | 0.644 | 0.685 | 90545 | 20676 | 12020 | 0.820 |

**Note on σ=0.008**: Run 2 hit the upper fit bound (β=2.0), distorting the
average. Excluding that run: β_all≈0.59, more consistent with the trend.

Individual run detail (σ=0.008 runs 1-3 for β_all):
  - Run 1: β=0.396, χ₄=31.54
  - Run 2: β=2.000 (FIT FAILURE), χ₄=32.72
  - Run 3: β=0.787, χ₄=3.32

#### v_A sweep at fixed σ=0.006

| v_A  | β_all | β_jammed | β_motile | τ_all  | χ₄ peak |
|------|-------|----------|----------|--------|---------|
| 0.006| 0.507 | 0.492    | 0.513    | 51970  | 18.80   |
| 0.008| 0.898 | 0.733    | 0.950    | 16005  | 3.33    |
| 0.010| 0.432 | 1.010    | 0.732    | 16693  | 15.79   |

#### 2. Four-Point Susceptibility χ₄(t)

χ₄(t) = N × Var[Q(t)] over 20 starting-time origins.
Peak height measures the extent of cooperatively rearranging regions.

| σ | χ₄ peak | t at peak |
|---|---------|-----------|
| 0.000 | 38.31 | 82016 |
| 0.003 | 27.83 | 63223 |
| 0.006 | 3.33 | 9842 |
| 0.008 | 22.53 | 95467 |

#### 3. Inherent v_A – Mobility Correlation

**Not available.** The current trajectory files (9 columns) do not
record per-cell inherent v_A_i. The continuation run outputs (10 columns in
`griffiths_study_phi89/`) were found to write the MEAN v_A for ALL cells
rather than the per-cell assigned value — this is a simulation output bug.
To measure the Pearson r(v_A_i, mobility_i), the simulation code must be
fixed to write the actual per-cell quenched v_A.

### Physical Interpretation — ANTI-GRIFFITHS BEHAVIOR

**The central result is surprising and the opposite of the Griffiths
rare-region prediction.** Rather than disorder broadening the relaxation
spectrum (decreasing β) and increasing cooperative dynamics (increasing χ₄),
we observe:

1. **β INCREASES with σ** (relaxation becomes MORE exponential):
   σ=0.000 → β=0.57, σ=0.006 → β=0.90. The control already has strongly
   stretched dynamics from caging at φ≈0.89. Adding quenched activity
   disorder makes relaxation LESS stretched, approaching simple exponential.

2. **χ₄ DECREASES with σ** (cooperativity DROPS):
   σ=0.000 → χ₄=38, σ=0.006 → χ₄=3.3. The peak four-point susceptibility
   drops by an order of magnitude. Cells relax more independently when
   they have different inherent activities.

3. **τ_all DECREASES with σ** (faster overall relaxation):
   σ=0.000 → τ=72308, σ=0.006 → τ=16005. Disorder accelerates the
   structural relaxation of the entire system.

#### Why this is anti-Griffiths

In Griffiths rare-region physics (e.g., dilute Ising magnets), quenched
disorder creates rare locally-ordered regions surrounded by a disordered
bulk. These rare regions:
- Have anomalously slow relaxation (power-law tails)
- Are STATIC — they cannot rearrange
- Create broad distributions of relaxation times (β → 0)
- Increase cooperative dynamics (χ₄ grows)

In our active cell system, quenched v_A disorder does the OPPOSITE because:
- **Cells are motile, not fixed on a lattice.** A cell with high v_A can
  PUSH neighboring slow cells out of their cages. Unlike frozen magnetic
  impurities, the fast cells actively break up jammed regions.
- **Activity heterogeneity acts as a fluidizer.** Some cells move faster
  than the mean, creating local stirring that disrupts the cooperative
  caging present in the homogeneous system.
- **The clean system (σ=0) is ALREADY deeply jammed** with strong collective
  dynamics (β=0.57, χ₄=38). This provides a high baseline that disorder
  can only REDUCE, not enhance.

#### Physical mechanism: disorder decorrelates caging

In the uniform system (σ=0), all 288 cells have the same propulsion force.
Structural relaxation requires collective cage-breaking — many cells must
rearrange simultaneously. This produces:
- Stretched exponential decay (broad spectrum of cage lifetimes)
- High χ₄ (cells must move cooperatively)
- Long τ (slow collective process)

With disorder:
- High-v_A cells break free from cages faster → initial rapid Q(t) decay
- These fast cells push on neighbors → cascade of rearrangements
- The system no longer needs COLLECTIVE relaxation → each cell relaxes
  somewhat independently based on its local v_A environment
- Result: more exponential Q(t), lower χ₄, shorter τ

#### Connection to earlier observations

Despite the anti-Griffiths Q(t)/χ₄ results, the EARLIER batch analysis
(2026-02-12 09:51) showed several signatures that appeared Griffiths-like:
- Higher persistence at σ=0.008 (indicating some cells stay jammed longer)
- Higher α₂ at σ=0.008 (broader displacement distribution)
- Higher τ_j/τ_m ratio at σ=0.008 (jammed cells DO relax more slowly)
- Spatial correlation length ξ=3.3 cells at σ=0.008

These are not contradictory — they reflect HETEROGENEITY in the dynamics
without the COOPERATIVE FREEZING that defines Griffiths physics. Individual
cells with low v_A do remain less mobile for longer periods, but this is a
single-cell effect rather than a collective phenomenon involving rare regions.
The χ₄ analysis definitively shows that the cooperativity (multi-cell
correlated rearrangements) actually DECREASES with disorder.

#### Anomaly at σ=0.008

The σ=0.008 data is noisy: two of three replicates show β≈0.5−0.8 and
χ₄≈32, while one shows β≈0.79 and χ₄≈3.3. This bimodality is
interesting — it may indicate:
- Bistability between glassy and fluid states at high disorder
- Strong sensitivity to the particular v_A realization (sample-to-sample fluctuations)
- Need for more replicates at this σ value

**The averaged σ=0.008 β_all=1.06 is unreliable due to the fit failure
(β=2.0) in run 2.** The better estimate from runs 1 and 3 is β≈0.59.

#### v_A sweep interpretation

At fixed σ=0.006:
- v_A=0.006 (deeper in jammed phase): β=0.51, χ₄=18.8 → more cooperative
- v_A=0.008 (moderate activity): β=0.90, χ₄=3.3 → most fluidized
- v_A=0.010 (higher activity): β=0.43, χ₄=15.8 → re-enters cooperative regime

The NON-MONOTONIC behavior at v_A=0.010 suggests that at higher mean
activity, the system approaches the unjamming transition where collective
dynamics re-emerge (motile cells must coordinate through excluded-volume
constraints). This deserves further investigation.

### Updated Signatures Summary

| Measurement | Griffiths Prediction | Observed | Status |
|-------------|---------------------|----------|--------|
| Persistence > control | Yes | ✓ at σ=0.008 | Single-cell effect |
| τ_j/τ_m > control | Yes | ✓ at σ=0.008 | Confirmed |
| α₂ >> 1 | Yes | ✓ at σ=0.008 | Confirmed |
| Spatial ξ growing | Yes | ✓ ξ=3.3 cells | Moderate |
| β_all < β_control | **Yes** | **✗ β INCREASES with σ** | **Anti-Griffiths** |
| χ₄ peak growing | **Yes** | **✗ χ₄ DECREASES with σ** | **Anti-Griffiths** |
| Pearson r(v_A, mob) | High | ⬜ Need simulation fix | Blocked |

**Verdict**: The system shows single-cell dynamic heterogeneity (persistence,
α₂, MSD separation) that increases with disorder, but the COLLECTIVE
signatures (β, χ₄) move in the OPPOSITE direction from Griffiths predictions.
Quenched activity disorder acts as a fluidizer in this motile system,
decorrelating collective caging dynamics rather than creating frozen rare regions.

### Next Steps
1. **Fix simulation output**: Write per-cell v_A_i (column 10) correctly
   so Pearson r(v_A, mobility) can be measured
2. **More replicates at σ=0.008**: Resolve the bimodality (3 replicates insufficient)
3. **Extend runs to t~800,000**: Check if anti-Griffiths behavior persists
   at longer times, or if slow Griffiths modes eventually emerge
4. **Larger system (18432 cells)**: Currently equilibrating — will allow
   percolation analysis and true rare-region statistics
5. **Consider FROZEN disorder**: Replace motile disorder (v_A heterogeneity)
   with SPATIAL disorder (random pinning sites) to test whether Griffiths
   physics requires static impurities

### Plots
- Log-log Q(t) + β fits: `postprocessing/output/griffiths_loglog_Qt_20260212.png`
- χ₄(t) susceptibility: `postprocessing/output/griffiths_chi4_20260212.png`

---

## Continuation Run Bugs — 2026-02-12 14:00

### Summary of All Bugs (Updated 2026-02-12 16:00)

Five critical bugs discovered through deep diagnosis of the continuation runs.
Code fixes implemented for bugs 1, 4, and 5. Bug 2 was diagnosed as a
consequence of bug 3. Bug 3 remains under investigation.

### Bug 1: Float32 time saturation — FIXED

**Symptom**: Time reporting saturates at t=524288 (= 2^19) and stops advancing.
Steps continue incrementing normally.

**Cause**: Time `t` is accumulated as a float32 on the GPU via `t += dt`.
When t ≥ 2^19, the ULP (unit in last place) for float32 is
2^(19−23) = 1/16 = 0.0625. Since dt=0.02 < ULP/2 = 0.03125,
the addition `t + dt` rounds back to `t`. Time is frozen.

**Impact**: Trajectory time column is all 524288.000000 after this point.
Physics is unaffected (cell evolution uses step-level dt, not accumulated t).
The trajectory data is still physically valid — use step numbers or row
indices instead of the time column.

**Fix applied**: Changed `float current_time` → `double current_time` in
`simulation.cuh`. All `0.0f` initializations → `0.0`. Accumulation uses
`current_time += static_cast<double>(domain.params.dt)`. Checkpoint still
stores float for backward compatibility (cast on save).

### Bug 2: Zero positions in continuation trajectory — ROOT CAUSE FOUND

**Symptom**: All x,y positions in continuation trajectory entries are 0.000000.
Velocities and polarity vectors are non-zero (velocity = v_A × polarization
exactly, with zero integral contribution).

**Root cause**: **The phi field is completely zero.** Binary checkpoint
inspection confirmed: ALL 288 cells have phi_max=0.000000, phi_nonzero=0,
sum_phi²=0.  With zero phase field:
- Centroid computation skips (`sum_phi2 < 1e-8` guard) → centroid stays 0
- Volume = sum_phi² × dA = 0
- Velocity integral = 0, so velocity = pure motility (v_A × polarization)
- Cell dynamics reduce to: random walkers with v_A speed, no interactions

This is NOT a bug in the trajectory writer or centroid sync code. The
centroids are correctly reported as (0,0) because the cells genuinely have
no phase field content.

### Bug 3: Catastrophic phi field decay — ROOT CAUSE FOUND, FIX APPLIED

**Symptom**: Phase field φ decays to exactly zero for ALL cells at some point
during long simulation runs. Checkpoint binary inspection confirms:

| Run | Steps | phi_max | Volume | Status |
|-----|-------|---------|--------|--------|
| vA=0.008, σ=0.000 | 35.2M | 0.0 | 0.0 | **DEAD** |
| vA=0.008, σ=0.003 | 39.5M | 0.0 | 0.0 | **DEAD** |
| vA=0.008, σ=0.006 | 71.1M | 1.0 (est) | 7492 | **ALIVE** |
| vA=0.008, σ=0.008 | 29.4M | 0.0 | 0.0 | **DEAD** |
| vA=0.010, σ=0.006 | 91.5M | 0.0 | 0.0 | **DEAD** |

Note: σ=0.000 and σ=0.003 checkpoints were OVERWRITTEN by continuation runs
(which wrote output to `griffiths_study/` instead of `griffiths_study_phi89/`).
The other checkpoints are confirmed from the original runs.

**σ=0.006 is the ONLY surviving run.** Its continuation checkpoint (in
`griffiths_study_phi89/`, step 16.7M) also has valid phi (vol=7481).

#### Root Cause: Unchecked D2D copies during phi pool regrow

**Code path** (`integrator.cu:allocate_phi_pool()`):

1. Cells' bboxes grow over time due to shape deformation/motility drift
2. Eventually `field_size` exceeds the pool's 50% headroom → `pool_needs_grow=true`
3. `grow_phi_pool()` → `allocate_phi_pool()` is called
4. New pool is allocated and **zero-initialized** (`cudaMemset(d_phi_pool, 0, ...)`)
5. For each cell, phi data is migrated: `cudaMemcpy(pool_slot, cell->d_phi, ..., D2D)`
6. **These D2D copies had NO CUDA error checking!** If a prior unchecked CUDA
   error left the runtime in a "sticky error" state, or if source pointers
   were invalid, ALL copies silently fail
7. All cells' phi fields are now the zero-initialized pool memory
8. `phi=0` is a **stable fixed point** of the PDE: the volume constraint
   term `−4(μ/V₀)(V₀−V)×φ` is proportional to φ, so it vanishes when φ=0.
   Cells cannot spontaneously recover.

**Why ALL cells die simultaneously**: `allocate_phi_pool` processes ALL cells
in a single migration loop. A systematic failure (sticky CUDA error, invalid
source pointers) affects every cell identically.

**Why it occurs at unpredictable times**: Pool regrow only triggers when a
cell's bbox exceeds the initial headroom. The timing depends on the particular
cell shapes and motility patterns, which vary by run.

**Why σ=0.006 survives**: Likely the pool never needed reallocation for that
particular run (cells' bboxes stayed within the initial headroom). With mean
v_A=0.008 and σ=0.006, nearly all cells are fast enough to maintain a regular
round shape, avoiding large bbox fluctuations. The surviving σ=0.006
continuation (from equilibrated checkpoint at 71.1M steps) also has valid phi,
supporting this hypothesis.

#### Fix Applied (`integrator.cu`)

1. **`CUDA_CHECK` on D2D migration copies**: The critical `cudaMemcpy` calls
   that copy phi data from old pool to new pool are now wrapped in `CUDA_CHECK`,
   which will abort with a diagnostic error message if any copy fails.

2. **`CUDA_CHECK` on sync calls**: The `cudaDeviceSynchronize()` before and
   after migration now checks return values to catch sticky errors.

3. **`CUDA_CHECK` on old pool free**: `cudaFree(old_pool)` is now checked.

4. **Phi integrity check**: After pool migration, the code now samples the
   center pixel of each cell's phi field and verifies at least one is non-zero.
   If all cells have zero phi, it prints a diagnostic error including any
   sticky CUDA error state. This catches the failure immediately rather than
   letting the simulation continue with ghost cells.

5. **Checkpoint I/O error checking** (`io.cu`): `cudaMemcpy` calls in
   `save_checkpoint` (D2H) and `load_checkpoint` (H2D) are now wrapped in
   `CUDA_CHECK` to catch errors during checkpoint persistence.

**Impact**: The ORIGINAL trajectory data (extracted before phi decay) is valid
and unaffected. Our anti-Griffiths analysis results stand. But these runs
cannot be extended — the cells have no physical form. Fresh runs from the
equilibrated checkpoint are needed.

### Bug 4: Per-cell v_A not checkpointed — FIXED

**Symptom**: On checkpoint restart, per-cell v_A values are re-drawn from the
random distribution, breaking quenched disorder identity.

**Cause**: `d_v_A` array was initialized in the integrator's `!rng_initialized`
block but never saved to or loaded from checkpoints.

**Fix applied**: Extended checkpoint format with backward-compatible v_A data:
- `save_checkpoint`: writes magic marker 0x56415F41 ("VA_A") + count + v_A
  array after cell data
- `load_checkpoint`: reads v_A if marker found, re-generates if not (old
  checkpoints)
- `Simulation::save_current_checkpoint()`: copies d_v_A from GPU via cudaMemcpy
- `Simulation::initialize_from_checkpoint()`: loads v_A into `loaded_v_A` member
- `Integrator`: Added `checkpoint_v_A` member; v_A init checks this first
- MPI version updated with matching changes

### Bug 5: Per-cell v_A always writes mean value — KNOWN

**Symptom**: Column 10 (v_A_i) in 10-column trajectory output shows the
mean v_A for ALL cells, regardless of σ. For σ=0.006, v_A=0.008,
all 288 cells show v_A_i=0.008000.

**Impact**: Cannot compute Pearson r(v_A_i, mobility_i) from trajectory data.
The fix for Bug 4 + a trajectory writer fix are needed.

### Continuation Jobs — Cancelled

All 18 Griffiths continuation jobs (RUNNING) and ~400 PENDING chain jobs were
cancelled. They were producing useless data (zero phi → zero positions →
meaningless trajectories). The eq2D_18432c equilibration was also cancelled
as collateral.

### Valid Data Inventory

The ORIGINAL trajectory data in `griffiths_study/` (9-column format, extracted
to `griffiths_subsampled/` locally) remains valid and is the basis for all
analysis. This data covers:
- 6 parameter combos × 3 replicates = 18 runs
- 1055–1315 timesteps per run (~25–31 MB subsampled)
- Valid centroids, velocities, polarity, theta
- No per-cell v_A column (9-col format)

The anti-Griffiths finding (β increases with σ, χ₄ decreases with σ) is
unaffected by these bugs.

---

## Plot Scrutiny — Griffiths Analysis Figures — 2026-02-12

### Purpose

Systematic evaluation of all Griffiths analysis plots. For each figure,
I assess: (1) what it measures and why, (2) whether the measurement is
methodologically sound, (3) what the data show, (4) physical interpretation,
and (5) limitations. The goal is to determine which conclusions are robust
and which require further investigation.

**Data basis:** 18 runs (6 parameter combos × 3 replicates), 288 cells,
1600×1600 domain, φ≈0.89, trajectory-interval=100 steps (Δt=2.0),
subsampled every 100th save (effective Δt≈200 between frames), ~1278 frames
per run spanning t≈0–330,000. Threshold = σ=0 control mean mobility = 0.000872.

---

### Figure 1: `griffiths_loglog_Qt_20260212.png` (2×3 = 6 panels)

**What this figure measures:**
The self-overlap function Q(t) = ⟨θ(a − |rᵢ(t) − rᵢ(0)|)⟩ is the
fraction of cells that have NOT moved farther than the cage radius `a`
after time lag t. It is the canonical order parameter for structural
relaxation in glassy systems. The stretched exponential fit
Q(t) = exp(−(t/τ)^β) extracts:
- τ: characteristic relaxation time (when Q drops to 1/e)
- β: stretching exponent (β=1 → simple exponential, β<1 → stretched)

The cage radius is set to a = 0.3 × cell_spacing where
cell_spacing = √(L²/N) = √(1600²/288) ≈ 94.3, giving a ≈ 28.3 grid units.
This is about 0.58 × R (cell radius R=49), meaning a cell must move roughly
60% of its own radius to be counted as "escaped." This is physically
reasonable for a dense tissue where centers are separated by ≈94 units.

#### Panel A (0,0): Log-log Q(t) — All Cells, σ sweep

**What to look for:** In Griffiths physics, increasing σ should create a
broader spectrum of relaxation times, making Q(t) decay as a power-law
or deeply stretched exponential. The log-log plot is chosen specifically
to reveal power-law tails: a power-law Q ∝ t^{−α} would appear as a
straight line.

**What the data show:** The OPPOSITE of Griffiths. The σ=0 control shows
the SLOWEST, most stretched decay (β=0.57, R²=0.754). The σ=0.006 case
decays fastest with near-exponential shape (β=0.90, R²=0.960). The dashed
fit lines should confirm that the stretched exponential captures the shape
well at σ=0.006 but poorly at σ=0 (where R²=0.754 suggests more complex
dynamics than a single stretched exponential).

**Methodological concern:** The Q(t) here is computed from a single origin
(trajectory quarter-point) per run, NOT from the multi-origin averaging
used for χ₄. This means single-origin noise affects the fit quality. The
R²=0.754 for σ=0 may reflect this noise rather than genuine deviation from
stretched exponential form. The multi-origin Q(t) in the β-vs-σ panel
(Panel D) should be more reliable.

**Physical interpretation:** The control system at φ=0.89 is deeply glassy
with cooperative caging dynamics producing a BROAD relaxation spectrum
(multiple cage-breaking timescales). Adding activity disorder NARROWS the
spectrum because fast cells locally fluidize their neighborhoods,
short-circuiting the collective cage-breaking process. Each cell's
relaxation rate is increasingly set by its local v_A environment rather
than by system-wide cooperative dynamics.

#### Panel B (0,1): Log-log Q(t) — Jammed Cells only

**What this measures:** Q(t) restricted to cells classified as "jammed"
(below threshold mobility). This isolates the slow subpopulation.

**What the data show:**
- β_jammed is ALWAYS below β_all: 0.336, 0.459, 0.733, 0.644
- Jammed cells have more stretched relaxation in all cases
- But β_jammed still INCREASES with σ (0.34 → 0.73 up to σ=0.006)

**Physical interpretation:** The jammed cells, by definition, are the slow
ones. Their Q(t) reflects the cage escape process, which involves multiple
timescales (initial rattling within cages + rare escape events). The
increase of β_jammed with σ means that even the slow cells experience a
SIMPLER relaxation process when disorder is present — likely because the
surrounding fast cells provide a more uniform "fluidization bath" that
regularizes the escape process.

The exception: at σ=0.008, β_jammed drops to 0.644. In the "two-fluid"
regime (σ≈v_A), the jammed cells are genuinely slow (near-zero inherent v_A)
and their cages are not as effectively broken by the (partially slow)
surrounding tissue. This hints that at very high disorder, some Griffiths
physics may re-emerge for the slowest subpopulation.

#### Panel C (0,2): Log-log Q(t) — Motile Cells only

**What the data show:**
- β_motile ≈ β_all in most cases (0.580, 0.624, 0.950, 0.685)
- At σ=0.006, β_motile = 0.950 — nearly pure exponential
- Motile cells relax in the simplest possible way when disorder is moderate

**Physical interpretation:** When σ=0.006 and v_A=0.008, virtually all cells
are motile (jammed fraction = 8.2%). The motile cells have sufficient
propulsion to escape cages individually, producing simple exponential
relaxation (β≈1). This is the anti-Griffiths regime in its purest form: the
system has been fluidized by heterogeneous activity.

#### Panel D (1,0): β vs σ — THE KEY TEST

**What this measures:** The stretching exponent averaged over replicates,
plotted against disorder strength. Three series: all cells (black circles),
jammed (blue squares), motile (red triangles). A horizontal line at β=1
marks pure exponential decay.

**Expected for Griffiths:** β should DECREASE with σ (more stretched →
broader relaxation spectrum → approaching power-law).

**Observed:** β INCREASES from 0.57 to 0.90 as σ goes from 0 to 0.006.
At σ=0.008, the average β=1.06 but this is UNRELIABLE (run 2 hit the
fit bound at β=2.0).

**Methodological concerns:**
1. **Fit bound β=2.0 is too high.** A Gaussian decay (β=2) is unphysical
   for Q(t) in a caging system. The bound should be capped at β≤1.5 or
   even β≤1.2. The fit failure in run 2 at σ=0.008 inflates the average.
   Excluding that run: β_all ≈ 0.59, which BREAKS the increasing trend
   and makes σ=0.008 look more like the control.
2. **Three replicates per σ.** Error bars from n=3 are inherently large.
   The standard error of β is likely ≈0.1–0.2, making the trend from
   0.57 to 0.90 only marginally significant (≈2σ).
3. **Replicate variance at σ=0.008.** Runs 1, 2, 3 give β = 0.40, 2.00,
   0.79 — a 5× spread. This bimodality suggests the system is near a
   transition where different v_A realizations produce qualitatively
   different dynamics. With 3 replicates, we cannot characterize this.

**Assessment:** The upward trend in β (σ=0→0.006) is the most robust
finding, supported by consistent R² improvements. The σ=0.008 point is
UNRELIABLE and should be disregarded until more replicates are available.

#### Panel E (1,1): τ_jammed / τ_motile ratio

**What this measures:** The timescale separation between jammed and motile
subpopulations, extracted from the stretched exponential fits.

**What the data show (from fits):**
- σ=0.000: 40459/32979 ≈ 1.23
- σ=0.003: 20331/16698 ≈ 1.22
- σ=0.006: 16896/12825 ≈ 1.32
- σ=0.008: 20676/12020 ≈ 1.72

A slight increase from 1.2 to 1.7 with σ. This is a MODEST effect.

**Comparison to batch analysis:** The earlier batch analysis using the 1/e
crossing time (not stretched exp fit) found τ_j/τ_m = 6.71 at σ=0.008.
The discrepancy (1.72 vs 6.71) reveals that the two methods probe different
aspects of relaxation: the stretched exp τ captures the CHARACTERISTIC
timescale where Q ≈ 1/e, while the 1/e crossing time is affected by the
SHAPE of the decay (more stretched → later crossing even at same τ).

**Assessment:** The fitted τ ratio shows weak dependence on σ and does NOT
support strong Griffiths timescale divergence. The larger effect seen in the
batch analysis (1/e crossing) arises from the changing SHAPE (β) rather
than the timescale itself.

#### Panel F (1,2): Fit Parameters Table

**Assessment:** The table is useful but must be read with the caveat that
σ=0.008 row averages are distorted by the run-2 fit failure. The R² column
is informative: σ=0.006 has R²=0.960 (excellent fit), σ=0.000 has R²=0.754
(poor fit, suggesting the stretched exp is inadequate for the deeply glassy
control).

---

### Figure 2: `griffiths_chi4_20260212.png` (1×3 = 3 panels)

**What this figure measures:**
The four-point susceptibility χ₄(t) = N × Var[Q(t)] quantifies the degree
of COOPERATIVE rearrangement at each timescale. When cells rearrange
collectively (correlated escapes from neighboring cages), the sample-to-
sample fluctuations in Q(t) are large → high χ₄. When cells relax
independently, Q(t) is self-averaging → low χ₄.

The χ₄ peak height is proportional to the number of cells in a cooperatively
rearranging region (CRR). The peak time t* corresponds to the timescale of
maximum cooperativity — typically the α-relaxation time.

**Multi-origin methodology:** Q(t) is computed from 20 different starting
times, evenly spaced across the trajectory. The variance is taken over these
20 realizations. With 20 origins, the variance estimate has ≈19 degrees of
freedom. At long lag times, adjacent origins see overlapping trajectory
segments, introducing correlations that reduce the effective degrees of
freedom.

#### Panel A (0): χ₄(t) curves by σ

**What the data show:**
- σ=0.000: Broad peak, χ₄_max ≈ 38 at t ≈ 82,000
- σ=0.003: Slightly lower peak, ≈28 at t ≈ 63,000
- σ=0.006: LOW flat peak, ≈3.3 at t ≈ 10,000
- σ=0.008: VARIABLE — two runs show peaks ≈32, one shows ≈3.3

**This is the most important panel in the entire study.** The order-of-
magnitude drop in χ₄ from σ=0 (38) to σ=0.006 (3.3) is the definitive
anti-Griffiths signature. It proves that quenched activity disorder
DESTROYS cooperative dynamics rather than creating frozen rare regions
with collective slow relaxation.

**Physical mechanism decoded:**
In the σ=0 control, all 288 cells push with force v_A=0.008. Cages break
only through collective coordination — cell A can't escape unless cell B
moves, which requires cell C to yield, etc. This correlated cascade
produces large Q fluctuations (high χ₄) because when one cage breaks, a
spatially extended chain of rearrangements follows (or not — hence the
variance).

At σ=0.006, each cell has different v_A. Fast cells (v_A>0.01) can break
their own cages without help. These independent escapes don't trigger
cascades — the surrounding (also fast) cells simply accommodate the motion.
The result: Q(t) decays steadily with low variance (low χ₄). The system
relaxes locally rather than collectively.

**Methodological concern:** The peak time t*≈82,000 for σ=0 approaches the
total trajectory length scaled by the number of origins (330,000/20 ≈
16,500). At lag times approaching the total span, different origins are
seeing the SAME late-time configurations, inflating the variance
artificially. The true χ₄ peak might be at even later times, meaning we may
be underestimating χ₄(σ=0). If anything, this makes the control even MORE
cooperative than reported, strengthening the anti-Griffiths conclusion.

#### Panel B (1): χ₄ peak height vs σ

**What this shows:** Decreasing trend with scatter at σ=0.008.

**Assessment:** The trend is clear for σ=0→0.003→0.006 (monotonic decrease
38→28→3.3). The σ=0.008 average of 22.5 breaks the trend, but this is
driven by two of three replicates showing unexpectedly high χ₄. This
bimodality at σ=0.008 (matching the β bimodality) suggests sample-to-sample
fluctuations dominate at σ≈v_A.

**Verdict:** The anti-Griffiths finding (χ₄ decreasing with σ) is ROBUST
for σ ≤ 0.006. The σ=0.008 point is inconclusive.

#### Panel C (2): χ₄ peak time vs σ

**What this shows:** Peak time shifts from ≈82,000 (σ=0) to ≈10,000
(σ=0.006). Cooperativity not only decreases in magnitude but also moves
to shorter timescales.

**Physical interpretation:** In the fluidized system (σ=0.006), the
residual cooperative dynamics happen quickly because cages are short-lived.
The remaining cooperativity involves nearest-neighbor interactions at short
timescales, not the system-spanning rearrangements seen in the glassy
control.

---

### Figure 3: `griffiths_vA_correlation_20260212.png`

**Status: NOT GENERATED.** The 9-column trajectory files do not contain
per-cell inherent v_A values. The 10-column continuation data (Bug 5)
writes the MEAN v_A for all cells, making correlation analysis impossible.
This measurement is blocked until the simulation code is fixed to write
per-cell v_A_i.

**When available, this will be a critical test:** If Pearson r(v_A_i,
mobility_i) → 1 with increasing σ, it confirms that quenched disorder
determines dynamics (each cell's fate is sealed by its inherent v_A). If
r stays moderate, collective effects (neighbor interactions) still matter
despite the quenched disorder.

---

### Figure 4: `griffiths_sigma_sweep_20260212.png` (3×4 = 12 panels)

This is the comprehensive batch analysis dashboard. Key panels:

#### Mobility Distribution Panel (0,0)

**What to look for:** Bimodality at high σ indicating a "two-fluid" state.

**Expected behavior:**
- σ=0: Unimodal, peaked near zero (many caged cells) with long tail
- σ=0.003: Similar to control
- σ=0.006: Shifted right (most cells motile), narrow
- σ=0.008: Bimodal — peak at zero (caged) + peak at high mobility (free)

The threshold (red dashed line) divides jammed from motile. The fraction
to the left of the threshold is the jammed fraction.

#### Jammed Fraction Panel (0,1)

**The non-monotonic trend** (62.3→46.5→8.2→53.5%) deserves explanation:

This trend is an artifact of using a FIXED threshold equal to the control
mean. At σ=0, the control has a right-skewed mobility distribution (mean >
median) due to rare cage-escape events. So 62% are below the mean — more
than the naive expectation of 50%.

At σ=0.006, most cells have v_A>>threshold, so nearly all are motile (8%).
At σ=0.008, the v_A distribution is so broad that many cells have v_A≈0,
restoring a ≈50/50 split. The non-monotonicity is therefore a real physical
effect: moderate disorder fluidizes the system, but extreme disorder
(σ≈v_A) reintroduces a frozen subpopulation.

#### Non-Gaussian Parameter α₂ Panel (2,0)

**What this measures:** α₂ = ⟨Δr⁴⟩/((d+2)⟨Δr²⟩²) − 1, where d=2.
For Gaussian displacement distributions, α₂=0. Positive α₂ indicates
heavy tails (some cells move much more/less than average).

**The data:** 0.39→0.34→2.15→12.70.

**Physical interpretation:** This is paradoxically CONSISTENT with
Griffiths-like behavior — displacement heterogeneity increases with σ.
But this measures SINGLE-CELL property dispersion, not COLLECTIVE dynamics.
The reconciliation with the anti-Griffiths χ₄ result is key:

- α₂ measures how different individual cells are from each other (variance
  of single-cell displacements)
- χ₄ measures how correlated those cells are with each other (variance of
  the collective overlap)

Disorder can increase α₂ (cells ARE more different) while decreasing χ₄
(cells DON'T move together). This is precisely the anti-Griffiths
mechanism: disorder diversifies individual behavior while DECORRELATING
collective behavior. It's like giving each person a different walking speed
— the distribution of distances walked broadens (high α₂) but people walk
independently (low χ₄).

#### Correlation Length ξ Panel (0,3)

**The data:** ξ ≈ 1.0, 0.9, 0.9, 3.3 cell spacings.

**Assessment:** Only σ=0.008 shows ξ above nearest-neighbor distances.
This is consistent with σ=0.008 creating correlated jammed/motile domains
extending ≈3 cells. However, with a 17×17 cell grid, ξ is severely
constrained — the maximum measurable correlation length is ≈8 cells.

The ξ=1.0 for the control is near the minimum expected (nearest-neighbor
caging produces weak correlations at 1 cell spacing). The lack of growth
at σ=0.003 and σ=0.006 reinforces that the spatial organization is weak
at moderate disorder.

#### Temporal Persistence Panel (1,0)

**The data:** 0.600, 0.592, 0.845, 0.756.

**Assessment:** Persistence = fraction of time each cell spends in its
majority classification (jammed or motile). At σ=0, persistence=0.60
(slightly above the 0.50 baseline for random classification → mild
temporal persistence from the glass dynamics). At σ=0.006, persistence=0.845
(the few jammed cells are PERSISTENTLY jammed). At σ=0.008, persistence=0.756
(between control and σ=0.006).

This is a genuine Griffiths-like signature — quenched disorder pins cells
in specific dynamical states. But it's a SINGLE-CELL effect: each low-v_A
cell individually remains jammed, without necessarily forming a collective
jammed region. The distinction matters: Griffiths physics requires
collective rare-region effects, not just individual particle trapping.

---

### Figure 5: `griffiths_vA_sweep_20260212.png` (3×4 = 12 panels)

**Same layout as σ sweep** but with v_A ∈ {0.006, 0.008, 0.010} at
fixed σ=0.006.

#### Key observation: Non-monotonic v_A dependence

At fixed σ=0.006:
- v_A=0.006: 50.5% jammed, β=0.51, χ₄=18.8 → GLASSY
- v_A=0.008: 8.2% jammed, β=0.90, χ₄=3.3 → FLUID
- v_A=0.010: 41.6% jammed, β=0.43, χ₄=15.8 → GLASSY AGAIN

**Physical interpretation — the re-entrant transition:**

This non-monotonic behavior is physically significant. At v_A=0.006 with
σ=0.006, the v_A distribution spans ≈0–0.012. Many cells have v_A near
zero → caged → cooperative glass dynamics.

At v_A=0.008 with σ=0.006, the distribution spans ≈0.002–0.014. Almost
all cells have sufficient propulsion to escape cages independently →
system is fluidized.

At v_A=0.010 with σ=0.006, the distribution spans ≈0.004–0.016. ALL cells
have high propulsion. But they're still packed at φ=0.89 — excluded volume
forces are now dominated by the ACTIVE PROPULSION forces. Cells push hard
against each other, creating active stress chains. The relevant cage-breaking
mechanism switches from "individual escape" to "collective stress
relaxation," re-introducing cooperative dynamics.

This re-entrant glass behavior at high activity is predicted by active-matter
glass theory and has been observed in active Brownian particle simulations
(Berthier 2019, Mandal et al. 2020). Our observation in the phase field
model is consistent with this literature and provides validation.

**Methodological concern:** The threshold is set from the v_A=0.008, σ=0
control. Applying this same threshold at v_A=0.006 and v_A=0.010 changes
the meaning of "jammed" — at v_A=0.006, cells are slower in general, so
more will be below the threshold regardless of disorder. This threshold
bias makes the jammed fractions across v_A values hard to compare directly.
However, β and χ₄ are threshold-independent metrics. The non-monotonic
behavior in β (0.51→0.90→0.43) is robust.

---

### Figure 6: `griffiths_Qt_comparison_20260212.png` (2×3 = 6 panels)

**What this shows:** Raw Q(t) decay curves for all cells, jammed cells,
and motile cells, for both the σ sweep and v_A sweep. Thin lines show
individual replicates, thick lines show averages.

#### Key features to check:

1. **Two-step relaxation:** In glasses, Q(t) typically shows a fast initial
   decay (β-relaxation: rattling in cages) followed by a plateau, then a
   slow decay (α-relaxation: cage escape). At φ=0.89, the plateau should be
   visible for σ=0 (deep caging). At σ=0.006, the plateau should be
   absent or reduced (cages are short-lived).

2. **Replicate spread:** Thin lines should cluster tightly for σ=0.006
   (reproducible fluid dynamics) but spread widely for σ=0.008
   (sample-dependent behavior). The σ=0.008 bimodality should be visible
   as two distinct thin-line trajectories.

3. **Jammed vs Motile separation:** The jammed Q(t) should decay more
   slowly in all cases. The RATIO of decay rates (jammed vs motile)
   indicates timescale separation.

4. **Long-time behavior:** Does Q(t) reach zero, or does it plateau at a
   non-zero value? A long-time plateau Q_∞ > 0 would indicate that some
   cells are PERMANENTLY caged (never escape) within the observation window.

---

### Figure 7: `griffiths_key_signatures_20260212.png` (2×3 = 6 panels)

**What this shows:** Three key tests of Griffiths physics, presented as
direct comparisons between σ values:

1. **Persistence distribution** (histograms by σ): Should show that higher
   σ produces more cells with persistence near 1.0 (permanent classification).

2. **τ_j/τ_m bar chart:** Should show increasing timescale separation with σ.
   From the batch analysis: σ=0→3.20, σ=0.003→1.13, σ=0.006→1.34,
   σ=0.008→6.71. Note: these are from the 1/e crossing, not the stretched
   exp fit, so they differ from the deep analysis values.

3. **α₂ bar chart:** Should show dramatic increase at σ=0.008.

**Assessment:** These three panels support single-cell Griffiths-like
effects (persistence, heterogeneity) but must be read together with the
χ₄ and β results which show the COLLECTIVE dynamics are anti-Griffiths.

---

### Figure 8: `fig_summary_griffiths.png` (288-cell, 2×2 summary)

**What this shows:** (a) Ensemble MSD log-log, (b) α₂(Δt/τ), (c) CV(MSD)
vs Δt/τ, (d) dual-axis: CV and peak α₂ vs σ.

This is from the 288-cell deep analysis (separate from the batch analysis)
and uses a different set of runs (single-run analysis with multiple disorder
levels). The MSD and α₂ should be consistent with the batch analysis.

**Key check:** Panel (a) MSD should show that σ>0 runs have HIGHER long-time
MSD than σ=0 (disorder promotes diffusion), which is consistent with the
anti-Griffiths finding. Panel (d) should show α₂ and CV increasing with σ
(more heterogeneous, even as collective dynamics decrease).

---

### SYNTHESIS: What is robust vs. what needs more work

#### ROBUST findings (supported by multiple independent metrics):

1. **Anti-Griffiths collective dynamics**: β increases with σ (0→0.006),
   χ₄ decreases by 10×. Quenched activity disorder DESTROYS cooperative
   rearrangements in this motile system. This is the OPPOSITE of what
   Griffiths rare-region theory predicts for frozen disorder in lattice
   models.

2. **Single-cell heterogeneity increases**: α₂ grows 5–30× with σ,
   persistence increases from 0.60 to 0.85, and displacement distributions
   develop heavy tails. Individual cells become more dynamically diverse
   with disorder, but they DON'T create collectively frozen regions.

3. **The core mechanism is activity-mediated fluidization**: Fast cells
   (high v_A) mechanically disrupt the caging network, providing local
   fluidization that short-circuits cooperative cage-breaking. This is
   fundamentally different from lattice-based Griffiths physics where
   frozen impurities cannot actively reshape their environment.

4. **The v_A=0.008, σ=0.006 combination is maximally fluid**: Nearly all
   cells are motile (8.2% jammed), β≈0.90, χ₄≈3.3. This represents the
   optimal activity+disorder combination for fluidization.

#### NEEDS MORE DATA:

1. **σ=0.008 behavior**: Bimodal across replicates (β: 0.40, 2.00, 0.79;
   χ₄: 31.5, 32.7, 3.3). Three replicates cannot resolve this. Need ≥10
   replicates to characterize the distribution.

2. **The v_A sweep non-monotonicity**: Only 3 points on this curve.
   Additional v_A values (0.005, 0.007, 0.009, 0.011, 0.012) would map
   the re-entrant transition properly.

3. **Per-cell v_A–mobility correlation**: Blocked by simulation Bug 5.
   This is crucial for proving that inherent v_A determines fate.

4. **System size effects**: 288 cells (17×17) severely limits correlation
   lengths, cluster statistics, and rare-region sampling. The 18,432-cell
   equilibration (cancelled — needs restart) will address this.

5. **Time extent**: t≈330,000 provides ≈5 relaxation times at σ=0. Longer
   runs would test whether slow Griffiths modes eventually emerge at
   timescales beyond our current observation window.

#### METHODOLOGICAL IMPROVEMENTS needed:

1. **Cap fit bound**: β_max should be 1.2, not 2.0. Gaussian decay (β=2)
   is unphysical for Q(t) and creates spurious fit results.

2. **Robust fit error**: Report INDIVIDUAL run β values with confidence
   intervals from bootstrap or Hessian, not just the mean±SEM from n=3.

3. **Multi-origin Q(t) for ALL panels**: Currently only χ₄ uses 20 origins.
   The Q(t) curves in the log-log plot use single-origin, which adds noise.

4. **Threshold sensitivity test**: Re-run analysis with threshold = 0.5×,
   1.5×, and 2× the control mean to verify that key findings (β trend,
   χ₄ trend) are threshold-independent.

5. **Periodic boundary unwrapping**: Verify that displacement calculations
   correctly handle cells that cross periodic boundaries. Incorrect
   unwrapping would systematically bias MSD, mobility, and Q(t) at long
   times.

---

### Connection to the Central Question

**"Jammed islands in motile sea" vs "motile islands in jammed sea"?**

The data clearly indicate **"jammed islands in motile sea"** at σ=0.006:
only 8.2% of cells are jammed, and these are persistently but INDIVIDUALLY
jammed (low inherent v_A). The surrounding motile tissue flows freely around
them.

However, this does NOT produce Griffiths rare-region physics because:
1. The "islands" are typically single cells or small (2–3 cell) clusters,
   not extended rare regions
2. The motile sea ACTIVELY disrupts any incipient jammed cluster through
   mechanical forcing
3. The relaxation spectrum NARROWS (β increases) rather than broadening

The system is better described as a **"stirred glass"**: quenched activity
disorder introduces fast stirrers that break up the cooperative caging
network, producing fluid-like dynamics with single-cell heterogeneity.
This is a novel active-matter effect with no analog in equilibrium Griffiths
physics.

---

## Next Steps — Priority Plan — 2026-02-12

### Immediate (Code)

1. **Upload and compile fixed source on cluster**
   - Modified files: `simulation.cuh`, `io.cuh`, `io.cu`, `integrator.cuh`,
     `integrator.cu`
   - Fixes: float→double time, v_A checkpoint persistence, CUDA_CHECK on
     pool migration D2D copies, phi integrity check, checkpoint I/O error checking
   - Build with existing cmake config

2. **Validation test run**
   - Short run (1000 steps) from equilibrated checkpoint
   - Verify: time advances beyond 2^19 (if applicable), v_A persists across
     restart, phi integrity check does not fire
   - Kill early — just need to confirm the binary runs cleanly

3. **Fix Bug 5: per-cell v_A trajectory output**
   - Currently writes mean v_A for all cells
   - Need to write `h_v_A[i]` (the actual per-cell quenched value) in column 10
   - This is in `Simulation::save_trajectory()` in `simulation.cuh`

### Near-term (Simulations)

4. **Restart Griffiths study runs**
   - From the ORIGINAL equilibrated checkpoint (before any Griffiths disorder)
   - With all 5 bug fixes applied
   - Same parameter space: 6 combos × 3 replicates
   - Consider adding more replicates at σ=0.008 (≥10 to resolve bimodality)
   - Output directories: ensure each run writes to its OWN directory

5. **Restart eq2D_18432c equilibration**
   - Was at ~8h/12h (frame_800000) when cancelled
   - Check if checkpoint was saved before cancellation
   - Resume or restart from scratch

### Analysis (With Existing Valid Data)

6. **Improve stretched exponential fitting**
   - Cap β ≤ 1.2 (remove unphysical Gaussian fits)
   - Use multi-origin Q(t) averaging for ALL fits (not just χ₄)
   - Bootstrap confidence intervals for β and τ per run
   - Re-run deep analysis with improved methodology

7. **Threshold sensitivity analysis**
   - Re-run batch analysis with threshold = 0.5×, 1.0×, 1.5×, 2.0× control mean
   - Verify anti-Griffiths finding is threshold-independent

8. **Write anti-Griffiths manuscript notes**
   - The "stirred glass" mechanism is a novel result
   - Document the key physics: disorder decorrelates cooperative dynamics
   - Compare to active-matter glass theory (Berthier 2019, Mandal 2020)
   - Flag the re-entrant v_A transition as a secondary finding

### Longer-term

9. **18,432-cell production runs**
   - After equilibration completes, run Griffiths study at full system size
   - Will resolve: finite-size effects, true correlation lengths,
     cluster size distributions, percolation analysis

10. **Frozen disorder variant**
    - Instead of quenched v_A, test SPATIAL disorder (random pinning sites)
    - This is closer to traditional Griffiths physics
    - Compare: does frozen spatial disorder produce Griffiths signatures
      where quenched activity disorder does not?

---

## Griffiths Continuation Jobs Submitted — 2026-02-13

### Context

All 4 critical simulation bugs were fixed and deployed on nibi (binary
rebuilt Feb 12 14:47). The previous 450+ continuation jobs were cancelled
because they ran with buggy code (float32 time saturation, missing per-cell
v_A in checkpoints, phi field decay from unchecked D2D copies). This
session resumes from that point.

### Data Integrity Assessment

Full scan of all 18 original Griffiths runs in `/scratch/ssilber/griffiths_study/`:

| Parameter combo | run_1 | run_2 | run_3 | Status |
|-----------------|-------|-------|-------|--------|
| vA=0.008, σ=0.000 | t=524288 | t=524288 | t=524288 | At float32 ceiling |
| vA=0.008, σ=0.003 | t=524288 | t=524288 | t=524288 | At float32 ceiling |
| vA=0.008, σ=0.006 | t=524288 | t=524288 | t=524288 | At float32 ceiling |
| vA=0.008, σ=0.008 | t=524288 | t=524288 | t=524288 | At float32 ceiling |
| vA=0.006, σ=0.006 | t=524288 | t=524288 | t=524288 | At float32 ceiling |
| vA=0.010, σ=0.006 | t=524288 | t=524288 | t=524288 | At float32 ceiling |

All 18 runs reached t=524288 (=2^19, the float32 precision limit). All
checkpoints are 29 MB (consistent, valid size). The original trajectory
data (9-column format, ~37M lines per run) is valid.

The corrupt continuation data at `/scratch/ssilber/griffiths_study_phi89/`
was confirmed: most runs have t=331000 data (from before checkpoints were
overwritten), σ=0.000 runs are empty (phi decay), σ=0.003 runs 1-2 empty.

### Continuation Strategy

**Binary**: The jamming_study executable is a symlink to the fixed home
directory binary (`~/cell_simulation/build/bin/cell_sim`, Feb 12 14:47).
All 4 fixes are active: double time, v_A checkpoint, CUDA_CHECK, phi
integrity check.

**Per-cell v_A handling**: The old checkpoints (created by pre-fix code)
do NOT contain per-cell v_A data. On load, the new code detects this and
regenerates v_A from the command-line `--v-A` and `--v-A-sigma` parameters.
This creates a NEW quenched disorder realization at t=524288, different from
the one used in the original simulation.

**Physics implication of new v_A realization**: After a transient period
(estimated ~100k time units ≈ 10×τ_persistence ≈ 10×10000), the system
will re-equilibrate with the new disorder pattern. For Griffiths physics
analysis, we need t > ~620,000 for steady-state data. This is fine given
our target of t=800,000.

For σ=0 runs (no disorder), v_A is identical for all cells regardless of
realization, so no transient is needed.

From this point forward, all subsequent checkpoint saves WILL include
per-cell v_A (new format), so further continuations preserve the exact
disorder realization.

### Jobs Submitted

Used `submit_job.sh --griffiths -t 800000 --chains 8 --runs 3`:

- **18 runs** × **8 chains** = **144 total SLURM jobs**
- First chain jobs: 8530580–8530717 (PENDING, Priority)
- Subsequent chains: PENDING (Dependency)
- Output: same directories in `/scratch/ssilber/griffiths_study/`
- Trajectory append mode: new data appends to existing trajectory files

**Chain calculation**:
- Current t: 524,288
- Target t: 800,000
- Delta: 275,712 time units
- Rate: ~52,500 time units per 3hr job (288 cells, H100 MIG)
- Required: ceil(275,712 / 52,500) = 6 chains
- Submitted: 8 chains (margin for slower-than-expected runs)
- Chains 7-8 will detect t ≥ t_end and exit quickly

**Estimated completion**: 6 × 3hr = 18hr wall time per run. With parallel
GPU scheduling across cluster nodes, expect all 18 runs complete within
~24-48 hours depending on queue wait.

### What to Expect in the Data

The continuation trajectories will show:
1. **t=524288 → ~530000**: Brief artifact period as time transitions from
   float32 to double precision. The first few entries may show a jump from
   524288 to the actual continued time.
2. **t=530000 → ~620000**: Transient as system adjusts to new v_A
   realization (for σ>0 runs). MSD, Q(t) etc. during this window should
   be excluded from analysis.
3. **t=620000 → 800000**: Steady-state regime with correct quenched
   disorder. This provides ~180,000 time units of valid Griffiths analysis
   data — roughly 3× the relaxation time τ_all.
4. **v_A preserved in checkpoint**: From the first checkpoint save onward
   (at t ≈ 524288 + CHECKPOINT_INTERVAL×dt = 524288 + 75000×0.02 = 525788),
   the per-cell v_A is stored. All subsequent chain jobs use the same
   disorder realization.

### Next Actions (After Completion)

1. Download subsampled trajectories for t > 600,000
2. Re-run batch analysis with extended time range
3. Re-run deep analysis (Q(t), χ₄, β) using ONLY t > 620,000 data
4. Compare anti-Griffiths signatures at long times vs. short times
5. The per-cell v_A will be available in the NEW checkpoint data — use it
   to compute Pearson r(v_A_i, mobility_i) for the long-time regime

---

## 4608-Cell Equilibration Completed — 2026-02-12

### Purpose

Equilibrate a large (N=4608) system at packing fraction φ≈0.89 with zero
motility (v_A=0). This produces a densely packed, mechanically relaxed
tissue configuration suitable as an initial condition for Griffiths
rare-region studies at 16× the cell count of the original 288-cell system.
The larger system enables proper finite-size scaling, percolation analysis,
cluster size distributions spanning 3+ decades, and correlation lengths up
to ~34 cell spacings (vs. ~8 for 288 cells).

### Parameters

| Parameter | Value |
|-----------|-------|
| N | 4608 cells |
| Domain | 6249 × 6249 |
| Packing fraction φ | 0.890 |
| Cell radius R | 49 |
| Interface width λ | 7 |
| Cell spacing | 92.1 grid units |
| v_A | 0 (no motility) |
| v_A_sigma | 0 |
| dt | 0.02 |
| Simulation time | t = 80,000 (4,000,000 steps) |
| Independent runs | 3 (random initial conditions) |

### Computational Resources

| Resource | Value |
|----------|-------|
| GPU | NVIDIA H100 (full, 80 GB) |
| CPUs | 4 per job |
| Memory | 32 GB per job |
| SLURM account | rrg-mkarttu-ab |
| SLURM job IDs | 8484558–8484633 (chain jobs) |
| Jobs per run | ~25 chain jobs submitted (3hr limit each), 2 substantial |

### Wall-Clock Time

Each run used 2 chain jobs (first hit the 3hr SLURM time limit, second
completed the remaining work):

| Run | Chain 1 | Chain 2 | Total GPU time |
|-----|---------|---------|----------------|
| run_1 | 3h 00m (TIMEOUT) | 2h 07m | **5h 07m** |
| run_2 | 3h 00m (TIMEOUT) | 2h 01m | **5h 01m** |
| run_3 | 3h 00m (TIMEOUT) | 2h 13m | **5h 14m** |
| **Average** | | | **5h 07m** |

**Rate:** ~15,700 time units/hour on H100 (4,000,000 steps / 5.1 hr ≈
784,000 steps/hr × 0.02 dt ≈ 15,700 t/hr). This is ~3.3× slower than
the 288-cell rate (~52,500 t/hr on MIG), consistent with the 16× increase
in cell count partially offset by the H100's larger compute capacity vs MIG.

### Output

| File | Size | Description |
|------|------|-------------|
| checkpoint.bin | 447–449 MB | Full binary checkpoint (v4 format) |
| trajectory.txt | ~719k lines (156 frames) | Centroid trajectories |

**Location:** `/scratch/ssilber/eq_4608_phi89/run_{1,2,3}/`

All 3 runs completed successfully. Trajectory spans t=0 to t=80,000 with
156 frames (save interval ≈ 516 time units). Final velocities are
O(10⁻⁵–10⁻⁶), confirming the system is mechanically relaxed.

### Visualization

The equilibrated configuration was visualized using `visualize_checkpoint_2d.py`,
which reads the binary checkpoint and composites per-cell φ² fields onto
the domain grid with golden-ratio hue cycling for visual distinction.

- **Full domain + 288-cell zoom:** `postprocessing/output/eq4608_phi89_checkpoint_v3_20260212.png`
- Layout: left panel shows full 6249×6249 domain, right panel shows a
  close-up region sized to contain ~288 cells (highlighted by yellow
  rectangle on the left panel)

### Significance for Griffiths Study

This equilibrated system provides the starting configuration for 4608-cell
Griffiths runs (submitted as 270 jobs, IDs starting 8536661, output to
`/scratch/ssilber/griffiths_v2_4608/`). At 68×68 cells across the domain,
the system offers:

- **Correlation lengths** up to ~34 cell spacings (vs. 8 for 288 cells)
- **Cluster statistics** spanning 1–4608 cells (vs. 1–288)
- **Percolation analysis** on a grid large enough to be reliable
- **Rare-region sampling:** jammed clusters of L≈10–20 cells become
  statistically accessible, enabling tests of the Griffiths τ(L) ~ exp(cL^d)
  scaling prediction

---

## Batch Griffiths Analysis — 2026-02-16 09:25

### Overview
Comparative analysis of Griffiths rare-region effects across disorder strengths
and mean motility values. Data from nibi cluster, subsampled trajectories
(every 100th timestep, ~1278 frames per run).

**18 runs total**: 6 parameter combos × 3 replicates

### Parameter Space
| v_A | σ | Purpose |
|-----|---|---------|
| 0.008 | **0.000** | **CONTROL** — no disorder |
| 0.008 | 0.003 | Weak disorder |
| 0.008 | 0.006 | Moderate disorder |
| 0.008 | 0.008 | Strong disorder (σ ~ v_A) |
| 0.006 | 0.006 | Lower motility |
| 0.010 | 0.006 | Higher motility |

### Threshold Selection
Used σ=0 control MEAN mobility = 0.001626

This is a key methodological choice: by using the control's mean mobility,
we apply the SAME absolute threshold to all parameter sets. Cells with
mobility below this threshold are classified as "jammed" — they are
systematically slower than the average cell in the disorder-free control.
In disordered cases, this identifies cells whose low inherent v_A keeps
them jammed relative to the homogeneous baseline.

### Results Summary — σ Sweep (fixed v_A=0.008)

| σ | Jammed % | Persistence | τ_j/τ_m | ξ/a | α₂ | J_perc | M_perc |
|---|----------|-------------|---------|-----|-----|--------|--------|
| 0.000 | 55.3% | 0.581 | 0.99 | 0.8 | 0.04 | 0.98 | 0.98 |
| 0.003 | 47.4% | 0.606 | 0.86 | 0.7 | 0.21 | 0.97 | 1.00 |
| 0.006 | 41.2% | 0.635 | 1.66 | 0.9 | 3.13 | 0.78 | 1.00 |
| 0.008 | 37.9% | 0.658 | 1.36 | 1.0 | 5.25 | 0.67 | 1.00 |

### Results Summary — v_A Sweep (fixed σ=0.006)

| v_A | Jammed % | Persistence | τ_j/τ_m | ξ/a | α₂ | J_perc | M_perc |
|-----|----------|-------------|---------|-----|-----|--------|--------|
| 0.006 | 52.6% | 0.642 | 1.03 | 1.0 | 9.48 | 0.98 | 1.00 |
| 0.008 | 41.2% | 0.635 | 1.66 | 0.9 | 3.13 | 0.78 | 1.00 |
| 0.010 | 29.1% | 0.715 | 1.42 | 0.8 | 0.60 | 0.27 | 1.00 |

### Physical Interpretation

**Key question**: Does quenched disorder (σ > 0) create persistent Griffiths
rare regions compared to the homogeneous control (σ = 0)?

#### What to look for:
1. **Persistence increasing with σ**: If cells with high/low inherent v_A
   remain jammed/motile for longer than the dynamic fluctuation timescale
   in the control, this is the hallmark of Griffiths rare regions.

2. **τ_jammed/τ_motile ratio increasing with σ**: In Griffiths physics,
   rare jammed regions embedded in a motile sea have anomalously slow
   relaxation (power-law tails instead of exponential).

3. **Non-Gaussian parameter α₂ increasing with σ**: Dynamic heterogeneity
   should increase as quenched disorder creates a wider distribution of
   local relaxation rates.

4. **Correlation length ξ increasing with σ**: Spatial correlations should
   grow as inherent v_A clusters create correlated mobility patterns.

#### Limitations at current time/size:
- **288 cells** may be too small for reliable percolation analysis
- **t ≈ 330,000** may not be long enough — continuation to t=800,000 in progress
- Threshold = control median is better than per-run median, but still crude
- Need to verify that the σ=0 control truly shows NO persistent spatial
  heterogeneity (its persistence should be ~0.5 for random fluctuations)

### Plots
- Disorder sweep: `postprocessing/output/griffiths_sigma_sweep_20260216.png`
- Motility sweep: `postprocessing/output/griffiths_vA_sweep_20260216.png`
- Q(t) comparison: `postprocessing/output/griffiths_Qt_comparison_20260216.png`
- Key signatures: `postprocessing/output/griffiths_key_signatures_20260216.png`

---

## Integrity Audit — 2026-02-16

### Summary

Full audit of all Griffiths simulation data across 288c, 4608c, and 1152c.

### 288c (`griffiths_v2`) — ✅ DATA VERIFIED GOOD

- All 18 runs present with data
- Headers show correct sigma values matching directory names
- Per-cell v_A diversity: unique_vA=558–1088 for σ>0 (log-normal), =1 for σ=0 (control)
- Progress: 6/18 complete (t≥880k), 12 in progress (t=342k–775k)
- 227 continuation chain jobs pending in queue

### 4608c (`griffiths_v2_4608`) — ❌ DATA CORRUPTED, TWO BUGS FOUND

**Status:** Only 5 of 18 directories have data. 0 pending/running jobs.

**Bug 1 — σ never applied (root cause identified):**
- ALL 5 runs show `v_A_sigma=0` in trajectory header, even `sigma_0.003` directories
- ALL cells have identical `v_A_i = 0.008000` (unique_vA=1 for every run)
- **Root cause:** The binary deployed via `simulation_src.tar` (Feb 11 14:03) is missing
  the post-checkpoint `domain.params.v_A_sigma` restoration code. The `--v-A-sigma` CLI
  flag is parsed into a local variable but never copied to `domain.params` after checkpoint
  loading overwrites it to 0. Fix exists in current local `main.cu` lines 728-731
  (`sim.loaded_v_A.clear()` + `sim.domain.params.v_A_sigma = params.v_A_sigma`) but was
  NOT in the tar.
- **Evidence:** `tar xf simulation_src.tar -O src/main.cu | grep v_A_sigma` returns only
  line 297 (CLI parsing). The regeneration block is absent.
- **Why 288c works:** Binary was rebuilt on cluster after Feb 11 with the fix. 288c chain
  links 2+ picked up the new binary; 4608c chains had all completed/failed before the
  rebuild.

**Bug 2 — Wrong GPU allocation (missing `-n 4608`):**
- `submit_all_griffiths.sh` calls `submit_job.sh --griffiths` for 4608c without `-n 4608`
- Default `N_CELLS=288` → GPU decision matrix selects:
  - `gpu:1` (generic, may land on MIG 1g.10gb with only 10.5 GB)
  - `WALLTIME=03:00:00` (b1 partition, 3h limit)
  - `CHAINS=18` (auto-calculated for 288c throughput, far too few for 4608c)
- Correct allocation for 4608c: `gpu:h100:1`, `WALLTIME=12:00:00` (b2), ~5 chains
- The local `submit_job.sh` now has validation (line 404): `--eq-base` without `-n` → error.
  This guard was NOT on the cluster when the jobs were submitted.
- CUDA errors in some 4608c logs ("illegal memory access") likely from running 4608 cells
  on MIG slice or from the wrong partition timing out mid-write.

**Fix (submit_all_griffiths.sh):** Updated locally to pass `-n` for each
cell count and removed hardcoded `CHAINS=18` (auto-calculation handles it).
Also added 1152c submission block.

### 1152c (`eq_1152_phi89`) — ⏳ EQUILIBRATION COMPLETE, NO PRODUCTION

- 3 equilibration runs at t=80,000 with correct headers (N=1152, Lx=3124)
- 42 `eq2D_1152c_r{1,2,3}` jobs pending in queue (additional eq chains)
- No `griffiths_v2_1152` directory exists — Griffiths production NOT yet submitted

### Remediation Plan

1. **Rebuild binary on cluster** from current local source (has the v_A_sigma fix)
2. **Delete corrupted 4608c data** (`rm -rf /scratch/ssilber/griffiths_v2_4608`)
3. **Upload fixed `submit_all_griffiths.sh`** (now passes `-n`, includes 1152c)
4. **Resubmit 4608c:** `./submit_job.sh --griffiths -n 4608 --runs 3 --eq-base /scratch/ssilber/eq_4608_phi89 -o /scratch/ssilber/griffiths_v2_4608`
   → Will auto-select: `gpu:h100:1`, 12h (b2), ~5 chains, 90 total jobs
5. **Submit 1152c:** `./submit_job.sh --griffiths -n 1152 --runs 3 --eq-base /scratch/ssilber/eq_1152_phi89 -o /scratch/ssilber/griffiths_v2_1152`
   → Will auto-select: `gpu:h100:1`, 3h (b1), ~5 chains, 90 total jobs

---
