# Phase Field Tissue Mechanics — Research Agent Instructions

> **When to consult this file:** You are reasoning about *research questions* — what physics to probe, what observables matter, how to interpret results in light of the vertex-model and active-matter literature. For *how* to build / run / analyze, use the relevant tooling instructions: simulation builds and CLI in [cell-simulation.instructions.md](cell-simulation.instructions.md), cluster jobs in [cluster-operations.instructions.md](cluster-operations.instructions.md), analysis via `cell_analyze --help` and `cell_analyze list`. Study-specific physics and protocol live in the per-study files (`adhesion-study`, `griffiths-study`, `palmieri-extension`).

---

## Mission

Use a multi-cell **phase field model** (PFM) to study glass/jamming and active-matter physics in confluent and near-confluent tissues, focusing on questions that *vertex models* and *cellular Potts models* cannot answer cleanly.

Each cell is a continuous field $\phi_i(\mathbf{r},t)$ with:

- Cahn–Hilliard interface energy ($\gamma |\nabla\phi|^2$ + double-well bulk)
- Quartic steric repulsion ($\kappa \sum_{i<j}\phi_i^2\phi_j^2$)
- Soft volume constraint ($\mu (V_i-A_0)^2$)
- Optional gradient-coupling adhesion ($J \sum_{i<j}\int \nabla\phi_i\cdot\nabla\phi_j\,dA$)
- Self-propulsion via run-and-tumble polarity dynamics ($v_A$, $\tau$)

The simulation is GPU-resident (CUDA) and supports 2D and 3D up to ~10⁴ cells. Detailed dynamics, parameter sets (Palmieri vs Bresler), and the EOM live in the simulation source and study-specific files — do not re-derive them here.

---

## Where the PFM has an edge over vertex models

| Vertex model assumption | PFM relaxes it via |
|---|---|
| Straight polygonal edges | Curved interfaces with finite width $\lambda$ |
| Discrete T1 events (instantaneous topology flips) | Continuous interface evolution; rearrangements have a finite trajectory |
| 100% confluence | Free surfaces and gaps are natural ($\rho < 1$) |
| No overlap | Compression / interpenetration accessible (limited by $\kappa$) |
| Adhesion as one line-tension parameter $\Lambda$ | Adhesion is a variational gradient coupling with a closed-form stability bound $J < 2\gamma$ |
| Motility as an external force | Active velocity couples directly to the field via advection $-v\cdot\nabla\phi$ |
| Frozen at $T = 0$, $v_0 = 0$ | Pure gradient-descent quench is meaningful (the "adhesion quench" protocol uses this) |

Research-question priority should weight these gaps. Vertex-model reproductions are useful as validation; novelty lies in regimes vertex models cannot touch.

---

## Open research directions

Listed by where the PFM advantage is strongest. Each line is one phrasing; the actual hypothesis lives in the study TOML / LOG_BOOK once a direction is picked up.

### Interface and shape dynamics
- Boundary fluctuation spectrum $S(k) = \langle |\hat\phi(k)|^2\rangle$ across the jamming transition — do they diverge or change scaling?
- Continuous-vs-discrete rearrangements: time-resolved contact-area trajectory during would-be T1 events. Is the "T1" a singular event or a finite-width process?
- Effective shape index $p_\mathrm{eff} = L_n \cdot 2\sqrt{\pi}$ extracted from $\phi = 0.5$ contours, compared to the vertex-model $p_0^* \approx 3.81$.

### Non-confluent and compressible regimes
- Second jamming transition at intermediate $\phi$ (geometric caging vs shape transition).
- Gap-size distribution $P(A_\mathrm{gap})$ and its percolation threshold.
- Overlap-dominated regime under compression: where does cell-shape control give way to overlap control?

### Adhesion-controlled rigidity (active study)
- $(\tilde J, v_A)$ phase diagram. Adhesion lowers the motility threshold for unjamming — by how much, and does $\langle p_\mathrm{eff}\rangle$ at the boundary land near 3.81?
- Adhesion quench at $v_A = 0$: continuous relaxation path to local minima, inaccessible to vertex models.
- See [adhesion-study.instructions.md](adhesion-study.instructions.md) for protocol, parameters, manuscript.

### Quenched motility disorder (active study)
- Griffiths rare-region prediction vs "stirred glass" prediction. Single-cell observables ($\alpha_2$, CV of $D_i$) are ambiguous; collective observables ($\beta$ from $Q(t)$ fits, $\chi_4$ peak) are decisive.
- Non-monotone $\bar v_A$ dependence of fluidization (testable signature of the Debets cage-scanning picture).
- See [griffiths-study.instructions.md](griffiths-study.instructions.md).

### Heterogeneity beyond a single soft cell (active study)
- Finite-size scaling of Palmieri's single-soft-cell motility enhancement — does the headline ratio $D_\mathrm{eff}^c/D_\mathrm{eff}^n$ survive $N \to \infty$?
- Percolation of fluidization as cancer-cell fraction $f_c$ grows.
- Polydispersity (size $R$ and stiffness $\gamma$) and its effect on shape-index transitions.
- See [palmieri-extension.instructions.md](palmieri-extension.instructions.md).

### 3D-specific
- Surface-vs-bulk dynamics in finite 3D tissues (layer-resolved MSD, T1 gradient).
- 3D analog of the shape-index transition (surface-area-to-volume scaling, asphericity).

---

## Analysis workflow

All quantitative observables are computed by the `cell_analyze` Rust binary. Do not reinvent them in Python.

- `cell_analyze list` enumerates every observable, aggregator, panel, and template.
- TOML studies in `cpp/simulation/study/templates/` show the canonical pipelines (FSS sweep, soft/ctrl pair, overlay sweep, single-run, pairwise separation).
- Each study run produces both figures and a `study_results.json` with full raw numbers — feed that into manuscript plotting / hypothesis checks.

If a new observable or aggregator is needed: drop a file in `rust/cell_analyze/src/analysis/observables/` (or `aggregate.rs`), register it, and rebuild. The crate is set up so adding metrics does not require touching the pipeline.

---

## How to design and report an experiment

1. **Pose the question physically.** What null hypothesis does the existing literature predict? What signature distinguishes alternatives?
2. **Identify the decisive observable.** Prefer collective / disorder-distinguishing observables when single-cell ones are ambiguous (e.g. $\chi_4$ over $\alpha_2$).
3. **Author the study TOML.** Discovery pattern, observables, aggregators, figures. Drop figures if you only want raw numbers — `study_results.json` is always written.
4. **Run.** Local for smoke; cluster (via MCP `start_simulation` / `resume_simulation` / `run_analysis`) for production. See [cluster-operations.instructions.md](cluster-operations.instructions.md).
5. **Interpret with literature in hand.** State explicitly what's reproduced, what's new, and which PFM affordance enables the new part.
6. **Log in the study LOG_BOOK.md.** That file is the source of truth for current status, not this instruction file.

---

## Terminology

- *Jammed* — solid-like, non-diffusive, caged dynamics.
- *Unjammed / fluid* — diffusive, free cell rearrangements.
- *Confluent* — no gaps ($\phi \to 1$); vertex-model regime.
- *Shape index* — $p = P/\sqrt{A}$ (measured) or $p_0$ (vertex-model target).
- *T1 transition* — neighbor exchange; in PFM it's a continuous process, not an instantaneous flip.
- *Tagged cell* — cell 0 by convention; the cancer / soft cell in Palmieri-style runs.

Stick to this vocabulary in writing and code so observables stay searchable.

---

## What this file does *not* contain

- File paths to scripts, data, or checkpoints (they drift).
- Production-run status (lives in study `LOG_BOOK.md`).
- CLI flags or build commands (use `--help` and the simulation instruction file).
- Manuscript style rules (see [manuscript-writing.instructions.md](manuscript-writing.instructions.md)).
