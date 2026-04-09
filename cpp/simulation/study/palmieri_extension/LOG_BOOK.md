# Palmieri Extension Study — Lab Notebook

**Project:** Phase-Field Model — Multi-Cell Populations: Finite-Size Scaling, Polydispersity, and Elastic Mismatch  
**System:** 72 / 500 / 2000 cells, 2D, various φ, R = 49, τ = 10,000  
**Started:** February 2026  

---

## Motivation

Palmieri et al. (Sci. Rep. 5, 11745, 2015) demonstrated that elastic mismatch alone can enhance the motility of a soft cancer cell embedded in a stiff monolayer — only 72 cells, a single cancer cell, monodisperse populations, purely repulsive interactions, and minimal statistical analysis. It did not engage with the jamming transition or connect to the vertex model literature.

This study extends the Palmieri model across three axes:

1. **Finite-size scaling** — Scale from 72 to 500 to 2000 cells to determine whether the Palmieri results are robust or finite-size artefacts.
2. **Polydispersity** — Gaussian radius distributions (CV_R = 0–0.20) and lognormal stiffness distributions (CV_γ = 0–0.50) to suppress crystallisation and test the motility enhancement threshold.
3. **Multiple cancer cells** — Vary cancer cell fraction f_c from 0.01 to 0.50 to find the percolation threshold for system-spanning fluidisation.

The study is guided by the supervisor's research plan (`palmieri_model_2d_extensions.pdf`), focusing on Phases 1, 3, and 5 of that document.

---

## Relationship to Other Studies

| Study | Focus | Status | Connection |
|-------|-------|--------|------------|
| Adhesion (`study/adhesion/`) | Gradient-coupling adhesion and the rigidity transition | In progress | Phase 4 of the supervisor's plan; adhesion results feed into Phase 4 here |
| Griffiths (`study/griffiths/`) | Quenched motility disorder | In progress | Disorder framework applies to polydispersity |
| **This study** | Finite-size, polydispersity, cancer cell populations | **Starting** | Phases 1, 3, 5 of the supervisor's plan |

---

## Model Specification

### Base model (Palmieri et al.)

- Each cell $n$: scalar field $\phi_n(\mathbf{x}, t)$ with $\phi_n = 1$ inside, $\phi_n = 0$ outside.
- Single-cell free energy $F_n[\phi_n]$ with elastic parameter $\gamma_n$, preferred radius $R_n$, area constraint $\mu_n$, interface width $\lambda$.
- Cell-cell interaction: repulsive overlap $F_\text{rep} = \kappa \sum_{n<m} \int \phi_n^2 \phi_m^2 \, dA$.
- Dynamics: Allen-Cahn with advection, run-and-tumble motility.
- Parameters: **Palmieri et al. (2015)** — $\gamma = 1$, $\kappa = 10$, $\mu = 1$, $\xi = 1500$, $R = 49$, $\lambda = 7$. These are the binary defaults.

> **Note:** The adhesion study uses Bresler parameters ($\gamma = 3.75$, $\mu = 0.5$, $\xi = 1000$) because its stability bound benefits from higher $\gamma$. This study uses Palmieri parameters for direct comparison with the original paper. The two sets are NOT interchangeable.

### Extension 1: Stiffness heterogeneity (existing)

Per-cell $\gamma_n$ via `--gamma V:selector` syntax. Cancer cells defined as cells with $\gamma_c < \gamma_n$ (e.g., `--gamma 0.35:cell0` for a single soft cancer cell, or `--gamma 1.3125:20%` for 20% of cells at $\gamma_c/\gamma_n = 0.35$).

### Extension 2: Radius polydispersity (needs implementation)

Per-cell $R_n$ drawn from Gaussian(⟨R⟩, CV_R × ⟨R⟩). Requires `--radius V:selector` with `cv` selector type. See FEEDBACK.md for implementation request.

### Extension 3: Adhesion (from adhesion study)

Gradient-coupling adhesion $F_\text{adh} = J\sum_{i<j}\int\nabla\phi_i\cdot\nabla\phi_j\,dA$ with stability bound $J < 2\gamma$. Already implemented.

---

## Experimental Design

### Phase 1: Validation and Baseline (supervisor's Phase 1)

**Goal:** Reproduce Palmieri et al. results at larger system size with proper ensemble statistics.

| Label | N_cells | Cancer | γ_c/γ_n | ρ | Adhesion | Realisations |
|-------|---------|--------|---------|---|----------|-------------|
| 1A | 72 | 1 | 0.35 | 0.85, 0.90 | None | 20 |
| 1B | 72 | 0 | — | 0.85, 0.90 | None | 20 |
| 1C | 500 | 1 | 0.35 | 0.85, 0.90 | None | 20 |
| 1D | 500 | 0 | — | 0.85, 0.90 | None | 20 |
| 1E | 2000 | 1 | 0.35 | 0.85, 0.90 | None | 10 |
| 1F | 2000 | 0 | — | 0.85, 0.90 | None | 10 |

**Key analyses:**
- D_eff of cancer cell and mean D_eff of normal cells (with error bars from ensemble averaging)
- Burst identification: |v_n| > μ_v + 3σ_v for ≥ T_burst consecutive steps
- Velocity distributions P(v_x), P(v_y) — fit to two-regime model and Student-t
- Finite-size scaling: all quantities vs 1/√N

### Phase 3: Multiple Cancer Cells and Percolation (supervisor's Phase 3)

**Goal:** Find the percolation threshold f_c* for system-spanning unjamming.

| Label | N_cells | Cancer cells | γ_c/γ_n | ρ | Configuration | Realisations |
|-------|---------|-------------|---------|---|---------------|-------------|
| 3A | 2000 | 2 (separation d) | 0.35 | 0.90 | d = 2R, 4R, 6R, 8R, 12R, 20R | 10/d |
| 3B | 4000 | f_c × N | 0.35 | 0.90 | Random dispersal | 10/f_c |
| | | f_c = 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50 | | | | |
| 3C | 4000 | f_c × N | 0.35 | 0.90 | Compact cluster | 10/f_c |
| 3D | 6000 | f_c × N | 0.35 | 0.85, 0.90, 0.95 | Random dispersal | 5/(f_c,ρ) |

**Key analyses:**
- Pairwise cooperativity: excess T1 rate vs separation d
- Percolation scan: f_mobile, cluster size distribution P(s), mean cluster size ⟨s⟩
- Clustered vs dispersed cancer cells: D_eff for edge/interior/isolated
- Finite-size scaling with 2000, 4000, 6000 cells

### Phase 5: Polydispersity (supervisor's Phase 5) — BLOCKED on `--radius V:selector`

**Goal:** Suppress crystallisation, test motility enhancement threshold, establish realistic baseline.

| Label | N_cells | Cancer | CV_R | CV_γ | γ_c/⟨γ⟩ | ρ | Adhesion | Realisations |
|-------|---------|--------|------|------|---------|---|----------|-------------|
| 5A | 2000 | 0 | 0–0.20 | 0 | — | 0.85, 0.90 | None | 10 |
| 5B | 2000 | 0 | 0.10 | 0–0.50 | — | 0.85, 0.90 | None | 10 |
| 5C | 2000 | 1 | 0.10 | 0.20 | 0.10–0.90 | 0.90 | None | 10 |
| 5D | 4000 | 0 | 0.10 | 0.20 | — | 0.90 | None, J_0 | 10 |
| 5E | 4000 | f_c×N | 0.10 | 0.20 | 0.35 | 0.90 | J_0 | 10/f_c |

**Key analyses:**
- Hexagonal order parameter ψ₆ vs CV_R (crystallisation suppression)
- Mobility landscape: D_eff(R_n, γ_n) contour plot
- Threshold vs continuous: D_eff vs γ_c/⟨γ⟩ at fixed CV_γ
- Soft-spot prediction: which cells undergo T1 rearrangements?

---

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Per-cell γ (`--gamma V:selector`) | ✅ Implemented | Supports bare, fraction, cell selectors |
| Per-cell R (`--radius V:selector`) | ❌ Not implemented | Requested in FEEDBACK.md; blocks Phase 5 |
| Adhesion (`--adhesion J`) | ✅ Implemented | Gradient coupling; needed for Phase 5D/5E |
| N = 100–12800 cells | ✅ Available | Up to ~6400 on MIG; 12800 needs V100+ |
| Multiple cancer cells (`--gamma V:N%`) | ✅ Available | Via fraction selector (5%+) |
| Palmieri defaults | ✅ Binary defaults | No physics param overrides needed |

---

## What Can Start Now (without `--radius`)

**Phase 1** (finite-size scaling) and **Phase 3** (multiple cancer cells) require only:
- Per-cell stiffness heterogeneity (`--gamma V:selector`) — already implemented
- Large system sizes (up to 12800 cells) — cluster
- Multiple realisations — different random seeds
- Palmieri parameters — binary defaults, no overrides needed
- For populations: start with 1-in-other, then fractions from 5% upward

These are unblocked and can proceed immediately.

**Phase 5** is blocked on `--radius V:selector` implementation.

---

## Log

### 2026-02-25

- Created study folder and LOG_BOOK
- Reviewed supervisor's extension plan (palmieri_model_2d_extensions.pdf)
- Filed `--radius V:selector` feature request in FEEDBACK.md
- Created agent instructions (.github/instructions/palmieri-extension.instructions.md)
- **Next:** Begin Phase 1 equilibration campaign (72, 500, 2000 cells at ρ = 0.85 and 0.90)
