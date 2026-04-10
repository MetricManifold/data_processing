---
applyTo: "cpp/simulation/study/palmieri_extension/**"
---

# Palmieri Extension Study — Agent Instructions

> **When to consult this file:** You are running, analyzing, designing experiments, or writing for the Palmieri extension study — finite-size scaling, stiffness heterogeneity populations, radius polydispersity, or multiple cancer cell fraction experiments. For HOW to submit jobs, build binaries, or use CLI flags, see [cell-simulation.instructions.md](cell-simulation.instructions.md) and [cluster-operations.instructions.md](cluster-operations.instructions.md). This file defines **WHAT** to run and **WHY**.

---

## Related Files

> **NOTE:** Study files (`cpp/simulation/study/`) are NOT tracked in git. They are gitignored and backed up to Nibi at `~/cell_simulation/study/`. After editing TOMLs or manuscripts, sync with `sync_study_config` MCP tool or rsync.

| File | Purpose |
|------|---------|
| `cpp/simulation/study/palmieri_extension/LOG_BOOK.md` | Primary logbook — all results, decisions, progress |
| `palmieri_model_2d_extensions.pdf` | Supervisor's full research plan (root of repo) |
| `cpp/simulation/cluster/references.md` | Literature summaries |

### Companion Studies

| Study | Instruction File | Connection |
|-------|-----------------|------------|
| Adhesion | [adhesion-study.instructions.md](adhesion-study.instructions.md) | Adhesion term for combined experiments |
| Griffiths | [griffiths-study.instructions.md](griffiths-study.instructions.md) | Quenched disorder baseline; methodology |

---

## Study Overview

**Goal:** Extend the Palmieri et al. (2015) single-soft-cell-in-monolayer result along three axes:

1. **Finite-size scaling** — Does the motility enhancement survive as $N$ increases from 100 to 12,800? Or are the original 72-cell results finite-size artefacts?
2. **Multiple cancer cells and percolation** — At what cancer cell fraction $f_c^*$ does fluidisation become system-spanning? Is the transition in the 2D percolation universality class?
3. **Polydispersity** — Does size/stiffness heterogeneity suppress crystallisation artefacts? Is cancer cell motility enhancement a sharp threshold or continuous in the stiffness contrast?

**Guided by:** Phases 1, 3, and 5 of the supervisor's research plan (`palmieri_model_2d_extensions.pdf`).

---

## Physics Background

### The Palmieri Result

Palmieri et al. (2015) showed that a single soft cell ($\gamma_c / \gamma_n = 0.35$) in a stiff monolayer (72 cells, $\rho = 0.85$–$0.90$) exhibits enhanced motility: intermittent speed bursts, non-Gaussian velocity distributions, and elevated $D_\text{eff}$. The mechanism is elastic mismatch — the soft cell deforms more easily through the cage of stiff neighbours.

**Limitations not addressed in the original study:**
- Only 72 cells (finite-size effects unknown)
- Single cancer cell only (no collective fluidisation)
- Monodisperse (hexagonal ordering artefacts)
- No adhesion, no ensemble averages

### The Three Extensions

**Finite-size scaling (Phase 1):** If $D_\text{eff}$, burst frequency, and velocity distribution parameters don't converge by $N = 72$, the original results are quantitatively unreliable. We use $N \in \{100, 200, 400, 800, 1600, 3200, 6400, 12800\}$ to span from near-original to thermodynamic limit.

**Percolation of fluidisation (Phase 3):** Cancer cells at fraction $f_c$ create local mobile regions. At $f_c^*$, these connect into a system-spanning cluster. The cluster size distribution should follow $P(s) \sim s^{-\tau_p}$ with $\tau_p \approx 187/91 \approx 2.05$ (2D percolation). Above $f_c^*$, a macroscopic fraction of cells is mobile.

**Polydispersity (Phase 5):** Size polydispersity (CV_R ≈ 0.10–0.15) suppresses hexagonal ordering. Stiffness polydispersity (CV_γ) creates a landscape of soft spots. The critical question: measuring $D_\text{eff}$ vs $\gamma_c/\langle\gamma\rangle$ at fixed CV_γ reveals whether motility enhancement is a sharp **threshold** (step function) or **continuous** (smooth increase). This has implications for mechanical phenotyping of metastatically competent cells.

---

## Parameters

### Palmieri Parameters (Source of Truth)

This study uses the **original Palmieri et al. (2015) parameters**, which are the binary defaults. **No physics parameter overrides are needed** when submitting via the MCP tools.

Run `cell_sim -h` to see current default values. The key point: the Palmieri parameters ARE the binary defaults, so no `gamma`, `mu`, `xi`, or other physics flags need to be specified.

> **Why Palmieri, not Bresler?** The Bresler et al. (2018) sharp-interface extension reparametrised $\gamma$, $\mu$, $\xi$ for analytical convenience. The adhesion study uses Bresler parameters because the stability bound $J < 2\gamma$ benefits from higher $\gamma$. This study directly extends the Palmieri model and must use Palmieri parameters for valid comparison with the original paper. The two parameter sets produce different physics — they are NOT interchangeable.

### Packing Fraction as a Study Variable

Unlike the adhesion study (fixed $\rho = 0.89$), this study treats **packing fraction $\rho$ as a primary variable**. The supervisor's plan sweeps $\rho$ from 0.70 to 1.00. We focus on:

$$\rho \in \{0.70,\; 0.75,\; 0.80,\; 0.85,\; 0.90,\; 0.95,\; 1.00\}$$

Palmieri et al. used $\rho = 0.85$ and $0.90$. Adding lower $\rho$ probes the non-confluent regime (inaccessible to vertex models); higher $\rho$ approaches full confluence.

### System Size Table

Domain side $L = \lceil\sqrt{N \pi R^2 / \rho}\rceil$ for $R = 49$:

| $N$ | $\rho = 0.70$ | $0.75$ | $0.80$ | $0.85$ | $0.90$ | $0.95$ | $1.00$ |
|-----|------|------|------|------|------|------|------|
| 100 | 1039 | 1003 | 972 | 943 | 916 | 892 | 869 |
| 200 | 1469 | 1419 | 1374 | 1333 | 1295 | 1261 | 1229 |
| 400 | 2077 | 2006 | 1943 | 1885 | 1831 | 1783 | 1738 |
| 800 | 2937 | 2837 | 2747 | 2665 | 2590 | 2521 | 2457 |
| 1600 | 4153 | 4012 | 3885 | 3769 | 3662 | 3565 | 3475 |
| 3200 | 5873 | 5674 | 5493 | 5329 | 5179 | 5041 | 4913 |
| 6400 | 8305 | 8023 | 7769 | 7537 | 7324 | 7129 | 6949 |
| 12800 | 11745 | 11347 | 10986 | 10658 | 10358 | 10082 | 9826 |

All sizes fit on MIG 10 GB slices up to $N = 6400$. $N = 12{,}800$ needs V100 16 GB or larger.

### Stiffness Ratios

The Palmieri ratio $\gamma_c / \gamma_n = 0.35$ maps to $\gamma_c = 0.35 \times 1.0 = 0.35$ at B Palmieri parameters (this is the original paper's value).

---

## Experimental Phases

### Phase 1: Validation and Finite-Size Scaling

**Goal:** Reproduce Palmieri et al. with proper statistics and test convergence with $N$.

**Status:** ✅ Unblocked — uses only existing per-cell stiffness overrides.

**Design:**

For each $N \in \{100, 200, 400, 800, 1600, 3200, 6400, 12800\}$ and $\rho \in \{0.85, 0.90\}$ (extend to full $\rho$ range once convergence is established):
- **1-in-other (soft-in-normal):** 1 cancer cell ($\gamma_c/\gamma_n = 0.35$) + $(N-1)$ normal cells
- **All-normal control:** $N$ normal cells

| Parameter | Value |
|-----------|-------|
| Motility | $v_A = 0.01$ (Palmieri value) |
| Production time | $t = 100{,}000$ ($10\tau$) |
| Replicates | 20 ($N \leq 400$), 10 ($N \leq 1600$), 5 ($N \leq 6400$), 3 ($N = 12800$) |

**Analyses:**
1. $D_\text{eff}$ of the cancer cell and mean $D_\text{eff}$ of normal cells (with ensemble error bars)
2. Burst identification: $|v_n| > \mu_v + 3\sigma_v$ for $\geq T_\text{burst}$ consecutive steps; measure burst frequency, duration, amplitude
3. Velocity distributions $P(v_x)$, $P(v_y)$ — fit to two-regime model (Palmieri Eq. 5) and Student-t
4. Shape index $p_\text{eff} = L_n \times 2\sqrt{\pi}$ from trajectory column 11
5. **Finite-size scaling:** All quantities plotted vs $1/\sqrt{N}$ to extrapolate to thermodynamic limit

**Key question:** Does $D_\text{eff}^\text{cancer} / D_\text{eff}^\text{normal}$ converge as $N \to \infty$, or does it vanish (the single cancer cell becomes negligible in a large system)?

**Confluence dependence:** Once finite-size convergence is established at $\rho = 0.85$ and $0.90$, extend the study to the full confluence range $\rho \in \{0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00\}$ at the converged system size. This maps the jamming phase diagram as a function of both $N$ and $\rho$.

### Phase 3: Multiple Cancer Cells and Percolation

**Goal:** Find the percolation threshold $f_c^*$ for system-spanning unjamming.

**Status:** ✅ Unblocked — uses fraction selector for cancer cell population.

**Sub-phase 3A — Pairwise cooperativity:**

Two cancer cells at separation $d \in \{2R, 4R, 6R, 8R, 12R, 20R\}$ in an $N = 2000$ monolayer at $\rho = 0.90$. Measure the T1 rate, local relaxation time, and local shape index between them as a function of $d$. The excess T1 rate quantifies cooperative enhancement. If it decays exponentially → finite interaction range; algebraically → long-ranged via the elastic field.

**Sub-phase 3B — Percolation scan:**

Cancer cell fraction $f_c \in \{0.05, 0.10, 0.15, 0.20, 0.30, 0.50\}$ in $N = 4000$ at $\rho = 0.90$, randomly dispersed. 10 realisations per $f_c$.

**Note on population construction:** Always start with the **1-in-other** case (Phase 1) before moving to fractions. Fraction populations begin at 5% — below this, the cancer cell count is too low for meaningful statistics (at $N = 4000$, 1% is only 40 cells; 2% is 80).

**Percolation observables:**
- $f_\text{mobile}$: fraction of cells whose cage-relative displacement exceeds threshold over $T_\text{obs}$
- $P(s)$: cluster size distribution of connected mobile regions (Voronoi adjacency)
- $\langle s \rangle$ (excluding largest cluster): diverges at $f_c^*$
- $P_\infty = s_\text{max}/N$: order parameter
- $\chi = \langle s^2 \rangle / \langle s \rangle$: susceptibility (peaks at $f_c^*$)
- $D_\text{eff}$ of cancer cells vs $f_c$
- Cancer cell $D_\text{eff}$ by environment: isolated (no cancer neighbour within $4R$), paired (one neighbour), clustered ($\geq 2$ neighbours)

**Sub-phase 3C — Clustered vs dispersed:**

Same $f_c$ values but cancer cells initialised as a compact cluster. Prediction: interior cancer cells are less motile than edge/isolated cells because they lack elastic mismatch contact with stiff tissue.

**Sub-phase 3D — Finite-size scaling of percolation:**

Repeat at $N \in \{2000, 4000, 6000\}$ and $\rho \in \{0.80, 0.85, 0.90, 0.95\}$ for $f_c \in \{0.05, 0.10, 0.15, 0.20\}$. Extract $f_c^*$ and critical exponents ($\nu$, $\beta$, $\gamma_p$) via finite-size scaling collapse.

### Phase 5: Polydispersity

**Status:** ⚠️ BLOCKED on `--radius V:selector` implementation (see FEEDBACK.md). Stiffness polydispersity via a `cv` selector is also not yet implemented.

**When unblocked:**

| Sub-phase | What varies | Key observable |
|-----------|------------|----------------|
| 5A | CV_R = 0–0.20 (size polydispersity) | $\psi_6$ (hexagonal order suppression), $\rho_J$ |
| 5B | CV_γ = 0–0.50 (stiffness polydispersity) | $\xi_\text{dyn}$ (dynamic heterogeneity length) |
| 5C | $\gamma_c/\langle\gamma\rangle = 0.10$–$0.90$ in polydisperse background | $D_\text{eff}$: threshold vs continuous |
| 5D | Adhesion $J_0$ (fully realistic: polydispersity + adhesion) | Reference baseline |
| 5E | $f_c = 0.05$–$0.20$ (polydispersity + adhesion + multiple cancer cells) | Percolation in realistic tissue |

---

## Equilibration Campaign

Every $(N, \rho)$ combination requires dedicated equilibrated checkpoints before production runs.

**Protocol:** $v_A = 0$, $J = 0$, grid init, $t = 80{,}000$ ($8\tau$), save final checkpoint only (no VTK, no trajectory). Uses Palmieri parameters (binary defaults — no physics parameter overrides needed).

Equilibrations are needed for every unique $(N, \rho)$ combination. Start with $\rho \in \{0.85, 0.90\}$ (the Palmieri paper values), then extend to the full confluence range once the pipeline is validated.

| $N$ | $\rho$ | Replicates |
|-----|--------|------------|
| 100 | 0.85, 0.90 | 20 |
| 200 | 0.85, 0.90 | 20 |
| 400 | 0.85, 0.90 | 20 |
| 800 | 0.85, 0.90 | 10 |
| 1600 | 0.85, 0.90 | 10 |
| 3200 | 0.85, 0.90 | 5 |
| 6400 | 0.85, 0.90 | 5 |
| 12800 | 0.85, 0.90 | 3 |

Use `estimate_cost` to preview walltime, chain count, and storage for each $(N, \rho)$ combination before submitting. The MCP tools auto-compute these from per-cluster calibration data.

Look up $L$ from the system size table above for the corresponding $(N, \rho)$.

---

## Analysis Pipeline

### Existing Tools

| Tool | Purpose |
|------|---------|
| `cell_analyze` (Rust binary) | MSD, $D_\text{eff}$, $Q(t)$, $\alpha_2$, $\chi_4$, shape index |
| `analyze_trajectory.py` | General trajectory analysis |
| `visualize.py` | 2D VTK visualization |

### New Analysis Needed

| Analysis | Purpose | Phase |
|----------|---------|-------|
| Burst detection | Velocity bursts with Palmieri criteria | 1 |
| Velocity distribution fitting | Two-regime model + Student-t | 1 |
| Finite-size extrapolation | Observables vs $1/\sqrt{N}$ | 1 |
| Percolation analysis | Cluster ID, $P(s)$, $\langle s \rangle$, $P_\infty$, $\chi$ | 3 |
| Voronoi tessellation | Neighbour identification for clusters | 3 |
| Mobility profiles | $D_\text{eff}(r \mid \text{cancer})$ | 3 |
| Hexagonal order $\psi_6$ | Bond-orientational order vs CV_R | 5 |
| Mobility landscape | $D_\text{eff}(R_n, \gamma_n)$ contour | 5 |
| Soft-spot prediction | Predict T1 sites from $(\gamma_n, z_n)$ | 5 |

---

## Key References

| Paper | Key result | Relevance |
|-------|-----------|-----------|
| Palmieri et al. (2015), Sci. Rep. 5:11745 | Elastic mismatch enhances cancer cell motility | **Base model and parameter source** |
| Bresler, Palmieri & Grant (2018), arXiv:1807.10318 | Sharp-interface limit of the Palmieri model | Theoretical context (not used for parameters) |
| Bi et al. (2015), Nat. Phys. 11:1074 | $p_0^* \approx 3.81$ rigidity transition | Shape index target |
| Bi et al. (2016), Phys. Rev. X 6:021011 | $(p_0, v_0)$ phase diagram | Phase diagram template |
| Park et al. (2015), Nat. Mater. 14:1040 | Experimental shape index in asthmatic epithelia | Experimental context |
| Lee et al. (2012), Biophys. J. 102:2731 | Cancer cell migration experiments | Palmieri's experimental reference |
| Loewe et al. (2020), PRL 125:038003 | Solid-liquid transition via deformability | PFM jamming |
| Saito & Ishihara (2024), Sci. Adv. 10:eadi8433 | Deformability-driven fluid-fluid transition | Novel PFM result |
