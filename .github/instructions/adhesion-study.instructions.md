---
applyTo: "cpp/simulation/agent_test_runs/adhesion_study/**,cpp/simulation/study/adhesion/**"
---

# Adhesion Study — Agent Instructions

> **When to consult this file:** You are running, analyzing, designing experiments, or writing for the gradient-coupling adhesion study. This covers the physics of gradient-coupling adhesion, the stability bound derivation, the three experimental phases, data requirements, the connection to vertex model physics, interpretation guidance, and the manuscript. For simulation CLI or builds, see [cell-simulation.instructions.md](cell-simulation.instructions.md).

---

## Related Files

### Study Directory (`cpp/simulation/study/adhesion/`)

| File | Purpose |
|------|---------|
| `LITERATURE_REVIEW.md` | Comprehensive lit review (audited Feb 2026) |
| `LOG_BOOK.md` | Primary logbook — all results, decisions, progress |
| `manuscript.tex` | PRE manuscript (in progress) |
| `analyze_phase0.py` | Phase 0 quench analysis |

### Test Directory (`cpp/simulation/agent_test_runs/adhesion_study/`)

| File | Purpose |
|------|---------|
| `analyze_phase0_grad.py` | Phase 0 analysis for gradient coupling |
| `analyze_two_cell.py` | Two-cell equilibrium sweep analysis |

### Instruction Files

| Task | Instruction File |
|------|-----------------|
| Building & running simulations | [cell-simulation.instructions.md](cell-simulation.instructions.md) |
| Cluster operations & job submission | [cluster-operations.instructions.md](cluster-operations.instructions.md) |
| Post-processing & visualization | [postprocessing.instructions.md](postprocessing.instructions.md) |
| Griffiths study (companion) | [griffiths-study.instructions.md](griffiths-study.instructions.md) |

---

## ⚠️ MANDATORY: Use MCP Tools for ALL Cluster Submissions

Never create one-off submission scripts or use `run_command` with `sbatch`. All adhesion study jobs MUST use the `start_simulation` or `resume_simulation` MCP tools. See [cluster-operations.instructions.md](cluster-operations.instructions.md) for the full tool reference.

---

## Study Overview

**Title:** Adhesion-controlled rigidity transition in a multi-cell phase field model

**Goal:** Demonstrate that gradient-coupling adhesion $F_\text{adh} = J\sum_{i<j}\int\nabla\phi_i\cdot\nabla\phi_j\,dA$ drives a rigidity transition analogous to the vertex model shape index transition at $p_0^* \approx 3.81$, and map the $(\tilde{J}, v_A)$ phase diagram.

**Target journal:** Physical Review E

**What makes this novel:**
1. No multi-cell phase field model has systematically varied adhesion across a rigidity transition
2. The adhesion quench ($v_A = 0$) is impossible in vertex models — it reveals the continuous relaxation path
3. The gradient coupling is the simplest variational member of the Nonomura family, with an analytically sharp stability bound $J < 2\gamma$

---

## Physics of Gradient-Coupling Adhesion

### The Adhesion Energy

$$F_\text{adh} = J \sum_{i<j} \int \nabla\phi_i \cdot \nabla\phi_j \, dA$$

At a shared interface: $\nabla\phi_i \approx -\nabla\phi_j$ (anti-parallel gradients), so $\nabla\phi_i \cdot \nabla\phi_j < 0$ → energy lowered. Away from interfaces: both gradients vanish → zero contribution. The adhesion is **strictly surface-localized**.

### Variational Derivative: Laplacian Coupling

$$\frac{\delta F_\text{adh}}{\delta \phi_i} = -J \sum_{j \neq i} \nabla^2 \phi_j$$

Integration by parts converts the gradient coupling into a Laplacian coupling. Since $\nabla^2\phi_j$ is localized at cell $j$'s interface (zero in the flat interior/exterior), the adhesion force is automatically surface-localized.

**Implementation:** Scatter all fields into $S(\mathbf{r}) = \sum_k\phi_k(\mathbf{r})$, then compute $\nabla^2 S - \nabla^2\phi_i$ via a five-point stencil. No pairwise loops; $O(NL^2)$ scaling. When $J = 0$, no extra memory is allocated and zero overhead is incurred.

### Stability Bound: $J < 2\gamma$

The total gradient-type energy at a shared interface:

$$E_\text{shared} = \int[\gamma|\nabla\phi_1|^2 + \gamma|\nabla\phi_2|^2 + J\nabla\phi_1\cdot\nabla\phi_2]\,dx = (2\gamma - J)\int|\nabla\phi|^2\,dx$$

This requires $2\gamma - J > 0$, i.e., $J < 2\gamma$.

**Positive-definiteness proof:** Define $S = \phi_1 + \phi_2$. Using $\nabla\phi_1\cdot\nabla\phi_2 = \frac{1}{2}(|\nabla S|^2 - |\nabla\phi_1|^2 - |\nabla\phi_2|^2)$, the energy becomes $(\gamma - J/2)|\nabla\phi_1|^2 + (\gamma - J/2)|\nabla\phi_2|^2 + (J/2)|\nabla S|^2$. Positive-definite iff $\gamma > J/2$, i.e., $J < 2\gamma$. This extends to $N$ cells and holds in any spatial dimension.

### The Dimensionless Control Parameter

$$\tilde{J} = \frac{J}{2\gamma}$$

measures the fraction of surface energy removed at shared interfaces:

| $\tilde{J}$ | $J$ ($\gamma=3.75$) | Surface energy reduction | Regime |
|---|---|---|---|
| 0 | 0 | 0% | Pure repulsion |
| 0.125 | 0.9375 | 12.5% | Very weak adhesion |
| 0.25 | 1.875 | 25% | Weak adhesion |
| 0.375 | 2.8125 | 37.5% | Moderate-weak adhesion |
| 0.50 | 3.75 | 50% | Moderate adhesion |
| 0.625 | 4.6875 | 62.5% | Moderate-strong adhesion |
| 0.75 | 5.625 | 75% | Strong adhesion (safe limit) |
| **1.0** | **7.5** | **100%** | **Critical: zero interface cost → instability** |
| >1 | >7.5 | — | Cell merger |

### Sharp-Interface Limit: Recovery of Vertex Model Adhesion

Using $\tanh$-profile interfaces of width $\lambda$ at a flat shared boundary of length $\ell_{ij}$:

$$\int_{-\infty}^{\infty} \nabla\phi_1 \cdot \nabla\phi_2 \, dx = -\frac{1}{3\lambda}$$

(via $\int \text{sech}^4 u\,du = 4/3$). Summing over all contacts:

$$F_\text{adh} \sim -\frac{J}{3\lambda}\sum_{i<j}\ell_{ij}$$

This matches the vertex model adhesion $-\gamma_\text{vm}\sum\ell_{ij}$ with identification $\gamma_\text{vm} = J/(3\lambda)$.

### Comparison with Nonomura (2012)

Nonomura uses $\nabla h(\phi_i) \cdot \nabla h(\phi_j)$ with $h(\phi) = \phi^2(3-2\phi)$ plus a regularization term $c\sum_i|\nabla h(\phi_i)|^2$. The regularization adds gradient stiffness proportional to $c\langle h'^2\rangle$ (where $\langle h'^2\rangle \approx 0.77$), raising the stability bound ~$8.7\times$ but adding a second free parameter. At Nonomura's strongest adhesion ($\gamma_N/D_0 = 6.5$), surface energy reduction is ~36% — comparable to our $\tilde{J} \approx 0.36$.

Our single-parameter model trades Nonomura's extended stability range for analytical transparency.

### Why Previous Adhesion Forms Failed

Six adhesion forms were tested before reaching gradient coupling. The fundamental lesson:

| Form | Problem |
|------|---------|
| Bilinear $-J\phi_i\phi_j$ | $\delta F/\delta\phi_i = -J\phi_j$ has no $\phi_i$ factor → nucleation in empty space |
| Smooth step $-Jg(\phi_i)g(\phi_j)$ | First-order: cells either repel or merge; no crossover |
| Reduced-$\kappa$ $(\kappa-J)\phi_i^2\phi_j^2$ | Same spatial profile as repulsion; no equilibrium |
| h(φ)·φ² coupling | Non-variational bulk force squishes cells |
| h(φ)·h(φ) coupling | $h'(\phi)$ changes sign → repulsive at interface midpoint |
| **Gradient coupling** $J\nabla\phi_i\cdot\nabla\phi_j$ | **Works.** Surface-localized, variational, single parameter |

The gradient coupling succeeds because it acts only where interfaces overlap, with a force ($-J\nabla^2\phi_j$) that alternates in sign across the interface and integrates to zero — preventing the nucleation/source behavior that destroyed bilinear forms.

---

## Equation of Motion

$$\frac{\partial\phi_i}{\partial t} = -M\frac{\delta F}{\delta \phi_i} - v_{A,i}\hat{\mathbf{p}}_i \cdot \nabla\phi_i$$

**Note on the advection sign:** The term $-\mathbf{v}\cdot\nabla\phi$ translates the cell in the direction of $\mathbf{v}$. This matches the kernel implementation (`dphi_dt = -0.5*var_deriv - advection` where `advection = v·∇φ`). The **manuscript had a sign error** ($+\mathbf{v}\cdot\nabla\phi$) that was corrected to $-\mathbf{v}\cdot\nabla\phi$ in the Feb 2026 audit.

The full variational derivative:

$$\frac{\delta F}{\delta\phi_i} = -2\gamma\nabla^2\phi_i + \frac{30\gamma}{\lambda^2}\phi_i(1-\phi_i)(1-2\phi_i) + 4\mu(V_i - A_0)\phi_i + 2\kappa\phi_i(S_2 - \phi_i^2) - J(\nabla^2 S - \nabla^2\phi_i)$$

where $S = \sum_k\phi_k$, $S_2 = \sum_k\phi_k^2$.

---

## Experimental Design

### Phase 0: Adhesion Quench (Diagnostic — UNIQUE TO PHASE FIELD)

**Purpose:** Determine if adhesion induces spontaneous rearrangements via pure gradient descent ($v_A = 0$). This experiment has no vertex model analog — in the vertex model at $T = 0$, $v_0 = 0$, changing $p_0$ relabels the energy without inducing motion. The phase field dynamics trace out the continuous relaxation path.

**Protocol:**
1. Start from equilibrated 288-cell checkpoint ($J = 0$, $v_A = 0$, $t \geq 80{,}000$)
2. Instantaneously set $J > 0$ while keeping $v_A = 0$
3. Evolve for $t = 20{,}000$ (2τ)
4. **Mandatory:** Include $J = 0$ control with identical protocol

**Parameter values (extended 7-value grid, $\Delta\tilde{J} = 0.125$, Bresler $\gamma = 3.75$):**

| $J/\kappa$ | $J$ | $\tilde{J}$ | CLI flag |
|-----------|-----|-------------|----------|
| 0.000 | 0.0 | 0.000 | (omit `--adhesion`) |
| 0.09375 | 0.9375 | 0.125 | `--adhesion 0.9375` |
| 0.1875 | 1.875 | 0.250 | `--adhesion 1.875` |
| 0.28125 | 2.8125 | 0.375 | `--adhesion 2.8125` |
| 0.375 | 3.75 | 0.500 | `--adhesion 3.75` |
| 0.46875 | 4.6875 | 0.625 | `--adhesion 4.6875` |
| 0.5625 | 5.625 | 0.750 | `--adhesion 5.625` |

**Do not exceed $\tilde{J} = 0.75$ ($J = 5.625$).** Values $J \geq 7.5$ ($\tilde{J} \geq 1$) are past the stability bound $J < 2\gamma = 7.5$.

**What to expect (confirmed by Phase 0 data):**

At $\phi = 0.89$ with $v_A = 0$, the quench does **not** produce neighbor exchanges (T1-like events). Adhesion lowers the energy of increased-contact configurations but does **not** lower the saddle-point energy for topology changes — the saddle is dominated by the cell-squeezing cost ($\kappa$ repulsion + $\gamma$ gradient energy required to push one cell past another). At $v_A = 0$, gradient descent cannot overcome these barriers regardless of $\tilde{J}$.

**Phase 0 gradient-coupling results (288 cells, rorqual run_01, 2τ):**

| $\tilde{J}$ | Mean $\Delta r / R$ | RMS displacement | Regime |
|---|---|---|---|
| 0.00 | 0.025 | 1.2 | Control (confirms equilibration) |
| 0.25 | 0.058 | 2.8 | Interface adjustment only |
| 0.50 | 0.086 | 4.2 | Interface adjustment only |
| 0.75 | 0.132 | 6.5 | Interface adjustment, v_rms still growing |

All displacements are sub-cell ($\ll R$). The smooth, monotonic increase reflects interface reshaping (contact angles adjusting, contact areas growing) — not topology change. This is physically correct: the quench traces the continuous relaxation path to the nearest local minimum, which preserves neighbor topology.

**Phase 0's scientific value** is therefore in **static equilibrium measurements**, not displacement:
- **Effective shape index** $p_\text{eff}(\tilde{J})$ from $\phi = 0.5$ contours — does adhesion raise $\langle p \rangle$ toward 3.81?
- **Contact angle** from interface geometry — test the Young-Dupré prediction $\cos\alpha = 1 - \tilde{J}$
- **Energy decomposition** — how much energy is released by interface relaxation vs. stored in elastic deformation?
- **Relaxation timescale** vs $\tilde{J}$ — does it diverge, suggesting proximity to a transition?

**I/O settings:**
- Trajectory: every 50 steps (= 1 TU) → high resolution for relaxation dynamics
- VTK: ~10 frames total → visual inspection
- Checkpoint: end-of-run only

### Phase 1: Motility Probe Sweep (MAIN EXPERIMENT)

**Purpose:** Map the fluid-solid boundary at nonzero motility. This is the **primary experiment** of the study. At $v_A = 0$ (Phase 0), cells are trapped in local energy minima by topological barriers that adhesion alone cannot remove. Motility provides the "thermal" energy that kicks cells over these barriers, analogous to $v_0$ in the vertex model. The adhesion-motility phase diagram $(\tilde{J}, v_A)$ is the paper's central result — the phase field analog of Bi et al.'s $(p_0, v_0)$ diagram.

Sweep $\tilde{J}$ at fixed small $v_A = 0.002$.

**Expected behavior:**
- $\tilde{J} = 0$, $v_A = 0.002$: jammed (strong caging, MSD plateau). From Griffiths data, $v_A = 0.008$ is near the transition at $J = 0$, so $v_A = 0.002$ is well below.
- Increasing $\tilde{J}$: adhesion reduces effective surface tension → cells deform and rearrange more easily → caging weakens, eventually diffusive regime emerges
- Transition at $\tilde{J}^*(v_A = 0.002)$ locates the adhesion-controlled unjamming boundary

**Parameters:** $\tilde{J} \in \{0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75\}$, $v_A = 0.002$, 3 replicates, $t = 50{,}000$ (5τ).

**Key observables:** MSD, $Q(t)$ (fit $\exp[-(t/\tau_\alpha)^\beta]$), $\alpha_2(t)$, $\chi_4(t)$, $p_\text{eff}$.

### Phase 2: Full $(\tilde{J}, v_A)$ Phase Diagram

**Purpose:** Map the complete two-parameter phase diagram for comparison with Bi et al. (2016).

**Grid:** $\tilde{J} \in \{0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75\}$ × $v_A \in \{0.002, 0.004, 0.006, 0.008, 0.010, 0.012\}$ × 3 replicates = 126 runs at $t = 50{,}000$ (5τ).

**Key predictions:**
- At $v_A = 0$: no T1 events (Phase 0 established that adhesion alone cannot overcome barriers at $\phi = 0.89$); Phase 0 provides shape/contact-angle baseline
- At $\tilde{J} = 0$: transition from Griffiths study clean baseline ($v_A^* \approx 0.008$; see companion study)
- Phase boundary should connect the Griffiths $v_A^*$ at $\tilde{J} = 0$ to a lower $v_A^*$ at finite $\tilde{J}$ — adhesion lowers the motility threshold for unjamming

---

## Mandatory Experimental Standards

### 1. Equilibration

All production runs start from equilibrated checkpoints: $t \geq 80{,}000$, $J = 0$, $v_A = 0$, $L = 1562$, $\phi = 0.89$. Uses Bresler et al. (2018) parameters ($\gamma = 3.75$, $\kappa = 10$, $\mu = 0.5$, $\xi = 1000$). Checkpoints: `/scratch/ssilber/eq_bresler_phi89/run_{01..10}/` on nibi, narval, rorqual.

**Why $\phi = 0.89$?** Confluent regime where vertex model operates. Three reasons to keep this packing fraction despite Phase 0 showing no T1 events:
1. **Vertex model comparison requires confluence** — Bi et al. (2015, 2016) operate at $\phi = 1$ (perfect tiling). Our $\phi = 0.89$ is the closest achievable in a phase field model with finite interface width.
2. **Lower $\phi$ tests a different transition** — reducing packing fraction would test the density-driven jamming transition (geometric caging → free space), not the adhesion-controlled rigidity transition we study.
3. **Experimental tissues are confluent** — in vivo epithelia (Park et al. 2015, Mongera et al. 2018) have $\phi \approx 1$. The biologically relevant regime is high confluence.

The consequence is that at $v_A = 0$, cells are too tightly packed for adhesion alone to drive T1-like rearrangements — the saddle-point energy for topology changes is dominated by the cell-squeezing cost ($\kappa$ repulsion + $\gamma$ gradient energy). This is why Phase 1 (motility) is the main experiment: motility provides the kinetic energy to overcome these barriers, and adhesion lowers the motility threshold needed.

### 2. Controls

Every experiment includes $J = 0$ control from the same checkpoint. Control displacement must be < $0.05R$.

### 3. Run Duration

| Phase | Minimum | Recommended | Rationale |
|-------|---------|-------------|-----------|
| Phase 0 ($v_A = 0$) | $t = 10{,}000$ (1τ) | $t = 20{,}000$ (2τ) | Critical slowing down near transition |
| Phase 1 | $t = 30{,}000$ (3τ) | $t = 50{,}000$ (5τ) | $\alpha_2$ peaks at ~2τ |
| Phase 2 | $t = 50{,}000$ (5τ) | $t = 50{,}000$ (5τ) | Matches Griffiths study |

### 4. Parameter Mapping

The scan parameter is $J/\kappa$ but the physics parameter is $\tilde{J} = J/(2\gamma)$:

$$\tilde{J} = \frac{J}{2\gamma} = \frac{J}{7.5} \quad \text{for Bresler } \gamma = 3.75$$

$$J/\kappa = \frac{J}{10} = \frac{7.5\tilde{J}}{10} = 0.75\tilde{J}$$

### 5. Fixed Parameters (Bresler et al. 2018)

This study uses **Bresler parameters**, which differ from the binary defaults. The non-default overrides that must be specified when submitting via MCP tools:

| Parameter | Value | Notes |
|-----------|-------|-------|
| $\gamma$ | 3.75 | Bresler (binary default: see `cell_sim -h`) |
| $\mu$ | 0.5 | Bresler (binary default: see `cell_sim -h`) |
| $\xi$ | 1000 | Bresler (binary default: see `cell_sim -h`) |

All other parameters ($\kappa$, $\tau$, $dt$, $R$, $\lambda$, $M$) use **binary defaults** — no overrides needed. Run `cell_sim -h` for current default values.

Geometry: $N = 288$ (or 1152), $\phi = 0.89$.

> **Note:** These are the Bresler, Palmieri & Grant (2018, arXiv:1807.10318) parameters. The shift from $\gamma=1$ to $\gamma=3.75$ moves the stability bound from $J < 2$ to $J < 7.5$. When submitting via MCP tools, use the tool's native `gamma`, `mu`, `xi` parameters — do NOT use `extra_cli_flags` for physics parameters that the tool schema supports.

---

## Two-Cell Validation

Two isolated cells at close range, $v_A = 0$, swept across $\tilde{J}$ values from 0 to beyond the stability bound. This validates mass conservation, the stability bound, cell shape preservation, and the absence of cell-shape artifacts. Results are in the development `LOG_BOOK.md` and `manuscript.tex` Table I.

**Key validation criteria:**
- Mass conservation < 1.2% across stable range
- Circular cell shape preserved (no squishing artifact)
- Critical point confirmed at $\tilde{J} = 1$ (interface dissolves)
- Cells merge at $\tilde{J} > 1$ (past stability bound)

---

## Connection to Vertex Model

| Vertex model | Phase field model |
|---|---|
| Line tension $\Lambda_{ij}$ at shared edges | Effective surface tension $(2\gamma - J)$ at shared interfaces |
| Adhesion reduces $\Lambda_{ij}$, raises $p_0$ | Adhesion reduces interface energy, favors elongated shapes |
| Control parameter: $p_0 = P_0/\sqrt{A_0}$ | Control parameter: $\tilde{J} = J/(2\gamma)$ |
| Transition at $p_0^* \approx 3.81$ | Transition at $\tilde{J}^*$ (to be determined from data) |
| At $T = 0$, $v_0 = 0$: frozen regardless of $p_0$ | At $v_A = 0$: relaxes via gradient descent → **quench possible** |
| $v_0$ provides effective temperature | $v_A$ provides kinetic energy to overcome topological barriers |
| $(p_0, v_0)$ phase diagram (Bi et al. 2016) | $(\tilde{J}, v_A)$ phase diagram (this study) |

**The open question:** Does $\langle p_\text{eff}\rangle \approx 3.81$ at the transition? Extract from $\phi = 0.5$ contours in Phase 1/2.

---

## Cluster Execution

Cluster data lives under `/scratch/ssilber/` on Nibi and other Alliance clusters. Equilibration checkpoints follow the pattern `eq_bresler_phi89/run_{NN}/checkpoint.bin` (Bresler parameters). Adhesion data follows `adhesion_bresler/phase{0,2}/Jt{X.XXX}_vA{Y.YYY}/run_NN/`. Use the **compute-canada MCP tool** (`list_jobs`, `check_progress`, `discover`) to check current job status and data availability.

### Submission Example

Use the `resume_simulation` MCP tool (from an equilibrated checkpoint):

```
resume_simulation(
  cluster="nibi",
  checkpoints=["/scratch/ssilber/eq_bresler_phi89/run_01/checkpoint.bin"],
  t_end=100000,
  adhesion_J=1.0,
  gamma=3.75,
  mu=0.5,
  xi=1000,
  trajectory_samples=2000,
  output_dir="/scratch/ssilber/adhesion_bresler/phase0/Jt0.133_vA0.000/run_01"
)
```

The MCP tool handles GPU selection, SLURM account, walltime, job chaining, and checkpoint management automatically. Do not manually specify account, walltime, or partition — the tool auto-selects these from calibration data and scheduler state.

---

## Discovering What Needs Doing

The study's progress is tracked in `LOG_BOOK.md` (both development and study versions) and the manuscript TODO markers. To determine what work remains:

1. **Read `LOG_BOOK.md`** — the latest entries describe completed runs and next steps
2. **Check `manuscript.tex`** — search for `\TODO` markers to find data and figure gaps
3. **Use the compute-canada MCP tool** — `list_jobs` and `check_progress` show what's running/pending
4. **Check `LITERATURE_REVIEW.md`** Sec. 8 — lists predicted observables and expected signals

### Analyses a Referee Would Likely Request

1. **Finer $\tilde{J}$ resolution** near the transition to establish sharpness
2. **Error bars / multiple replicates** for quench displacement
3. **Shape index extraction** ($p_\text{eff}$ from $\phi = 0.5$ contours) — the claimed vertex model connection needs this
4. **Energy decomposition** during quench to distinguish interface relaxation from rearrangement
5. **Finite-size check** — at least one $\tilde{J}$ compared between $N = 288$ and $N = 1152$
6. **Contact angle measurement** from $\phi = 0.5$ contours to test the Young-Dupré prediction

---

## Danger Signals and Pitfalls

1. **Cell merger at $\tilde{J} \geq 1$:** Monitor mass per cell; > $1.5V_0$ indicates dissolution.
2. **Mass drift:** Laplacian integrates to zero over periodic domain; any drift is discretization artifact. Track per-cell mass; < 1% acceptable.
3. **Insufficient equilibration:** Early Phase 0 used 2,000 TU at $\phi = 0.85$ — all discarded. Minimum: $t = 80{,}000$ at $\phi = 0.89$.
4. **$J/\kappa$ vs $\tilde{J}$ confusion:** $\tilde{J} = J/(2\gamma) = J/7.5$ for Bresler params. Always convert.
5. **Past-stability values:** Safe range: $J < 7.5$ ($\tilde{J} < 1$). Production limit: $\tilde{J} \leq 0.75$ ($J \leq 5.625$).
6. **Missing `--adhesion` flag:** Defaults to $J = 0$. Verify in job log output.
7. **Trajectory sampling bug (fixed Feb 18):** Binary ignored `--trajectory-samples` on checkpoint resume. Fixed; rebuilt on all clusters.
8. **Wrong packing fraction:** Use $L = 1562$ ($\phi = 0.89$), not $L = 1600$ ($\phi = 0.85$).

---

## Implementation History

The gradient coupling is the **seventh** adhesion form tested. The development history (including h(φ) coupling, bilinear variants, and normalization calibration) is in `agent_test_runs/adhesion_study/LOG_BOOK.md`.

---

## Key References

| Paper | Relevance |
|-------|-----------|
| Nonomura 2012, PLoS ONE 7, e33501 | Ancestor of gradient coupling; regularization comparison |
| Palmieri et al. 2015, Sci. Rep. 5, 11745 | 8τ equilibration protocol |
| Najem & Grant 2016, PRE 93, 052405 | Alternative range-field adhesion |
| Löber et al. 2015, Sci. Rep. 5, 9172 | Non-variational gradient-type adhesion |
| Bi et al. 2015, Nat. Phys. 11, 1074 | Shape index transition $p_0^* \approx 3.81$ |
| Bi et al. 2016, PRX 6, 021011 | $(p_0, v_0)$ phase diagram — comparison target |
| Moshe et al. 2018, PRL 120, 268105 | Geometric frustration explains $p_0^* = 3.81$ |
| Mongera et al. 2018, Nature 561, 401 | In vivo adhesion-controlled jamming |
| Loewe et al. 2020, PRL 125, 038003 | PFM solid-liquid transition (no adhesion) |

---

## Logbook Protocol

The primary logbook is `cpp/simulation/study/adhesion/LOG_BOOK.md`.

### Entry Template

```markdown
### YYYY-MM-DD — [Title]
**Runs:** N=288, L=1562, φ=0.89, J/κ=[values] (Jtilde=[values]), v_A=[value]
**Key results:** [quantitative]
**Next:** [action items]
```

### Output

Figures: `postprocessing/output/adhesion_<analysis>_<params>_YYYYMMDD.png`

---

## Quality Checklist

- [ ] Checkpoint: $t \geq 80{,}000$, $J = 0$, $v_A = 0$, $L = 1562$
- [ ] $J = 0$ control with identical protocol
- [ ] Control displacement < $0.05R$
- [ ] $\tilde{J} \leq 0.75$ (within stability bound)
- [ ] Duration meets phase minimum
- [ ] $J/\kappa$ converted to $\tilde{J}$ for interpretation
- [ ] Figures saved to `postprocessing/output/`
- [ ] Logbook updated with quantitative results
- [ ] Mass conservation verified (< 1%)

---

*Last updated: February 20, 2026 — Updated for Bresler parameters (γ=3.75, κ=10, μ=0.5, ξ=1000)*
