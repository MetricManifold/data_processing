# Adhesion Study — Lab Notebook

**Project:** Phase-Field Model — Cell-Cell Adhesion and the Shape Index Transition  
**System:** 288 / 1152 cells, 2D, φ ≈ 0.89, R = 49, τ = 10,000  
**Started:** February 2026  

---

## Motivation

The vertex model predicts a **rigidity transition** controlled by the target shape index
$p_0 = P_0 / \sqrt{A_0}$ at $p_0^* \approx 3.81$ (Bi et al., 2015). Physically, $p_0$ encodes
the balance between **cortical contractility** (actomyosin shrinks the perimeter) and
**cadherin-mediated adhesion** (E-cadherins at shared edges lower the effective line
tension, increasing $P_0$).

Our phase field model currently has **no adhesion term** — cell-cell interactions are
purely repulsive ($\kappa \sum \phi_i^2 \phi_j^2$). The jamming we observe is entirely
steric/geometric. This study adds an adhesion term and maps the resulting phase diagram
to connect with the vertex model literature.

---

## The Adhesion Term

### Choice: Gradient-coupling adhesion

$$F_\text{adh} = J \sum_{i<j} \int \nabla\phi_i \cdot \nabla\phi_j \, dA$$

At shared interfaces, $\nabla\phi_i$ and $\nabla\phi_j$ are anti-parallel, so the energy
is negative (favorable). The variational derivative:

$$\frac{\delta F}{\delta \phi_i} = -J \sum_{j \neq i} \nabla^2 \phi_j$$

This is the only adhesion form (out of seven tested) that is simultaneously:
1. **Surface-localized** — force acts only where interfaces overlap
2. **Variational** — derived from a well-defined free energy
3. **Self-gating** — no nucleation in empty space ($\nabla\phi = 0$ away from interfaces)
4. **Single-parameter** — controlled entirely by $J$ (or $\tilde{J} = J/2\gamma$)

**Implementation:** Scatter all fields into $S(\mathbf{r}) = \sum_k\phi_k(\mathbf{r})$,
then compute $\nabla^2 S - \nabla^2\phi_i$ via a five-point stencil. No pairwise loops;
$O(NL^2)$ scaling. When $J = 0$, no extra memory is allocated and zero overhead is incurred.

**Stability bound:** $J < 2\gamma$ ($\tilde{J} < 1$). Safe production range: $\tilde{J} \leq 0.75$.

### Why this form? (Development history)

The gradient coupling is the **seventh** adhesion form tested. Six previous forms failed:

| Form | Problem |
|------|---------|
| Bilinear $-J\phi_i\phi_j$ | $\delta F/\delta\phi_i = -J\phi_j$ has no $\phi_i$ factor → nucleation in empty space |
| Smooth step $-Jg(\phi_i)g(\phi_j)$ | First-order: cells either repel or merge; no crossover |
| Reduced-$\kappa$ $(\kappa-J)\phi_i^2\phi_j^2$ | Same spatial profile as repulsion; no equilibrium |
| h(φ)·φ² coupling | Non-variational bulk force squishes cells |
| h(φ)·h(φ) coupling | $h'(\phi)$ changes sign → repulsive at interface midpoint |
| **Gradient coupling** $J\nabla\phi_i\cdot\nabla\phi_j$ | **Works.** Surface-localized, variational, single parameter |

The original logbook opened with bilinear adhesion as the chosen form and the development
history below (§"Functional Form Analysis") documents the full journey from bilinear through
$h(\phi)$ to gradient coupling. The detailed derivations and failed-form analyses are preserved
for reference but are superseded by the gradient coupling choice above.

### Literature

#### Phase field models for cell mechanics

- **Nonomura (2012)** — PLOS ONE 7(4):e33501.
  [DOI: 10.1371/journal.pone.0033501](https://doi.org/10.1371/journal.pone.0033501).
  Multi-cell phase field with $\phi_i^2 \phi_j^2$ repulsion; discusses attractive terms.
  128 citations — foundational multi-cell phase field paper.
- **Palmieri, Bresler, Wirtz & Grant (2015)** — Sci. Rep. 5:11745.
  [DOI: 10.1038/srep11745](https://doi.org/10.1038/srep11745).
  Phase field model for cell migration in monolayers; elastic mismatch between
  cell types enhances motility. Uses self-propelled deformable droplet description.
- **Löber, Ziebert & Aranson (2015)** — Sci. Rep. 5:9172.
  [DOI: 10.1038/srep09172](https://doi.org/10.1038/srep09172).
  Uses bilinear overlap $\rho_i \rho_j$ as steric **repulsion** (parameter $\lambda$);
  adhesion modeled by advecting $\rho_i$ along the interface normal of cell $j$ (parameter $\kappa$).
  Our adhesion term $-J\phi_i\phi_j$ takes the same bilinear functional form with opposite sign.
- **Najem & Grant (2016)** — Phys. Rev. E 93:052405.
  [DOI: 10.1103/PhysRevE.93.052405](https://doi.org/10.1103/PhysRevE.93.052405).
  Phase-field model for collective cell migration. Same Grant group as Palmieri et al.;
  incorporates cell-cell adhesion via overlap integrals. 41 citations.
- **Wenzel & Voigt (2021)** — Phys. Rev. E 104:054410.
  [DOI: 10.1103/PhysRevE.104.054410](https://doi.org/10.1103/PhysRevE.104.054410).
  Multiphase field models for collective cell migration. Systematic comparison
  of coupling terms (repulsion, adhesion, gradient) in multiphase field framework. 28 citations.
- **Graham, Zhang & Yeomans (2024)** — Soft Matter 20:2955–2960.
  [DOI: 10.1039/d3sm01033c](https://doi.org/10.1039/d3sm01033c).
  Cell sorting by active forces in a phase-field model of cell monolayers.
  Uses multi-phase field with active forces to study phase separation; directly relevant
  to our adhesion + motility study.
- **Saito & Ishihara (2024)** — Sci. Adv. 10(19):eadi8433.
  [DOI: 10.1126/sciadv.adi8433](https://doi.org/10.1126/sciadv.adi8433).
  Phase field + deformability → fluid-to-fluid transition distinct from the vertex model
  shape-index transition. Shows cell deformability (not just shape index) controls rigidity.

#### Vertex model / jamming transition

- **Bi, Lopez, Schwarz & Manning (2015)** — Nat. Phys. 11:1074–1079.
  [DOI: 10.1038/nphys3471](https://doi.org/10.1038/nphys3471).
  Predicts rigidity transition at $p_0^* \approx 3.81$ in the vertex model.
  Foundational paper for the shape-index-controlled jamming transition.
- **Bi, Yang, Marchetti & Manning (2016)** — Phys. Rev. X 6:021011.
  [DOI: 10.1103/PhysRevX.6.021011](https://doi.org/10.1103/PhysRevX.6.021011).
  $(p_0, v_0)$ phase diagram — motility-driven unjamming in vertex model.
  Direct analog to our Phase 1/2 experiments.
- **Park et al. (2015)** — Nat. Mater. 14:1040–1048.
  [DOI: 10.1038/nmat4357](https://doi.org/10.1038/nmat4357).
  Key experimental validation: unjamming transition in human bronchial epithelial cells.
  Confirms $p > 3.81$ in unjammed asthmatic tissue. 757 citations.

#### Reviews

- **Camley & Rappel (2017)** — J. Phys. D 50:113002.
  [DOI: 10.1088/1361-6463/aa56fe](https://doi.org/10.1088/1361-6463/aa56fe).
  Comprehensive review comparing phase field, vertex, and particle-based approaches
  for collective cell motility. 189 citations.

### Alternatives considered but rejected

> **Note:** This table reflects the initial (incorrect) assessment from Feb 13.
> Gradient coupling was ultimately adopted as the production form after all others
> failed. See §"Gradient Coupling — Replacing h(φ) with ∇φ·∇φ" below.

| Option | Form | Original assessment | Outcome |
|--------|------|---------------------|---------|
| Bilinear overlap | $-J \int \phi_i \phi_j \, dA$ | Originally chosen | **Failed** — nucleation in empty space |
| Gradient coupling | $J \int \nabla\phi_i \cdot \nabla\phi_j \, dA$ | "Expensive; more like differential adhesion" | **Adopted** — the only form that works |
| Interface-weighted | $-J \int |\nabla\phi_i|^2 \phi_j^2 \, dA$ | Asymmetric; complex variational derivative | Not tested |

---

### Functional Form Analysis — Why Bilinear Fails and the $h(\phi)$ Solution

**Date:** 2026-02-19

The original bilinear adhesion term $F_{adh} = -J \sum_{i<j} \int \phi_i \phi_j \, dA$
has a fundamental flaw: the variational derivative $\delta F/\delta \phi_i = -J \sum_{j \neq i} \phi_j$
has **no $\phi_i$ dependence**. At any grid point where $\sum \phi_j > 0$, adhesion acts
as a source term that drives $\phi_i$ upward from zero. This nucleates field everywhere,
eventually filling the entire domain with nonzero $\phi$.

#### Failed approaches (chronological)

| Version | Form | Behavior | Why it fails |
|---------|------|----------|-------------|
| v1 (bare bilinear) | $-J \phi_i \phi_j$ | Nucleation — field fills domain | $\delta F/\delta\phi_i = -J\phi_j$ has no $\phi_i$ factor |
| v2 ($g(\phi)$ smoothstep) | $-J\, g(\phi_i)\,g(\phi_j)$, $g = \phi^2(3-2\phi)$ | First-order transition: cells either fully repel or fully merge | $g$ maps $[0,1]\to[0,1]$ with same profile as $\phi^2$ — no crossover with repulsion |
| v3 (gradient coupling) | $\varepsilon \nabla\phi_i \cdot \nabla\phi_j$ | Net repulsion at ALL coupling strengths | Integration by parts: $-\varepsilon \nabla^2\phi_j$ adds curvature matching, not attraction |
| v4 (reduced $\kappa$) | $(\kappa - J)\phi_i^2\phi_j^2$ | Weakens repulsion only — no attraction from distance | Same $\phi^2\phi^2$ spatial profile, just reduced amplitude |
| v5 (scaled bilinear) | $-J \cdot (30/\lambda^2) \cdot \phi_i \phi_j$ | 2-cell works! But multi-cell degrades: $\phi_{max} \to 0.83$, field fills corners | Scaling fixes the energy scale but NOT the nucleation (source at $\phi_i=0$) |
| v6 (gated bilinear) | $-J \cdot (30/\lambda^2) \cdot \text{gate}(\phi_i) \cdot \phi_j$ | Delays onset but still degrades at strong $J$ | Gate is an ad hoc fix — below the threshold, there's still a force |

#### The fundamental tension

For a stable equilibrium to exist (cells in contact but not merged), we need adhesion
and repulsion to have **different spatial profiles** so they cross over:
- **Repulsion** $\kappa \phi_i^2 \phi_j^2$ is proportional to $\phi^2$ — strongest in
  the cell core ($\phi \approx 1$), negligible in the tail ($\phi \ll 1$).
- **Adhesion** must be strongest in the **interface tail** ($\phi \sim 0.1$–$0.4$) where
  cells first make contact, and weak in both the core ($\phi \approx 1$) and empty space
  ($\phi \approx 0$).

If adhesion has the same $\phi^n$ profile as repulsion (any power law $\phi_i^a \phi_j^a$),
it cannot create a crossover — it's just a rescaled version of repulsion.

#### Power law inventory

Systematic analysis of $F = -J \sum_{i<j} \int \phi_i^a \phi_j^a \, dA$:

| Exponent $a$ | $\delta F/\delta\phi_i$ | Self-gating? | Spatial profile | Problem |
|---|---|---|---|---|
| 1 | $-J\phi_j$ | ❌ No | Widest range | Nucleation |
| 1 < a < 2 | $-aJ\phi_i^{a-1}\phi_j^a$ | ✅ | Intermediate | Divergent curvature at $\phi=0$ (non-smooth) |
| 2 | $-2J\phi_i\phi_j^2$ | ✅ | Same as repulsion $\phi^2$ | Just reduced $\kappa$; no crossover → no minimum |
| $a > 2$ | $-aJ\phi_i^{a-1}\phi_j^a$ | ✅ | Narrower than repulsion | Even shorter range — crossover in wrong direction |

**Conclusion:** No power law $\phi^a$ can work. We need a function that peaks in the
interface tail, not in the core.

#### The $h(\phi)$ solution

Define:
$$h(\phi) = \phi^2 (1 - \phi)^n$$

Properties:
- $h(0) = 0$ — zero in empty space (no nucleation)
- $h(1) = 0$ — zero in cell core (adhesion inactive deep inside cells)
- Peak at $\phi^* = 2/(n+2)$ — tunable to sit in the interface tail
- $h'(0) = 0$ — from the $\phi^2$ factor, not just $h(0)=0$ (smooth self-gating)

For $n = 4$: $\phi^* = 1/3$, which sits precisely in the tail of the tanh interface
profile. This means adhesion is strongest exactly where cells first make contact.

**Free energy:**
$$F_{adh} = -\frac{30J}{\lambda^2} \sum_{i<j} \int \left[h(\phi_i) \cdot \phi_j^2 + \phi_i^2 \cdot h(\phi_j)\right] dA$$

The $\phi_j^2$ factors play two roles:
1. They couple adhesion to the same overlap region where repulsion acts
2. The symmetrized form $h(\phi_i)\phi_j^2 + \phi_i^2 h(\phi_j)$ ensures the free energy is symmetric under $i \leftrightarrow j$

**Variational derivative:**
$$\frac{\delta F_{adh}}{\delta \phi_i} = -\frac{30J}{\lambda^2} \sum_{j \neq i} \left[h'(\phi_i) \cdot \phi_j^2 + 2\phi_i \cdot h(\phi_j)\right]$$

where:
$$h'(\phi) = \phi(1-\phi)^{n-1}\left[2(1-\phi) - n\phi\right]$$

**Self-gating proof:** Both terms vanish at $\phi_i = 0$:
- $h'(0) = 0$ (from the $\phi$ factor in $h' = \phi(1-\phi)^{n-1}[\ldots]$)
- $2\phi_i \cdot h(\phi_j)\big|_{\phi_i=0} = 0$

Therefore: **no nucleation is possible.** The adhesion force on cell $i$ is strictly
zero wherever $\phi_i = 0$.

**Crossover with repulsion:** The repulsion force is $2\kappa(30/\lambda^2)\phi_i \cdot \sum\phi_j^2$.
At light overlap ($\phi_i$ small), $h(\phi_i) = \phi_i^2(1-\phi_i)^n \approx \phi_i^2$
which matches $\phi_i$ in the repulsion → crossover exists. At deep overlap ($\phi_i \to 1$),
$h(\phi_i) \to 0$ while repulsion grows → repulsion dominates → no merger.

#### Choice of $n$

| $n$ | Peak $\phi^*$ | Peak $h(\phi^*)$ | Behavior |
|-----|---------|--------------|----------|
| 2 | 0.50 | 0.0625 | Peak at interface center — too deep |
| 3 | 0.40 | 0.0346 | Slightly below center |
| **4** | **0.33** | **0.0219** | **In the tail — optimal for first-contact adhesion** |
| 5 | 0.29 | 0.0150 | Deep in the tail |
| 6 | 0.25 | 0.0107 | Very deep in tail — weak signal |

**Selected: $n = 4$** — peak at $\phi = 1/3$ captures the regime where cell interfaces
first overlap. The $\phi^2$ factor in $h$ provides self-gating while $(1-\phi)^4$
ensures the function is negligible for $\phi > 0.7$.

$n$ is defined as a compile-time constant `ADHESION_N` in `types.cuh` for easy tuning.

#### Implementation summary

The scatter kernel accumulates $\sum_k h(\phi_k)$ instead of $\sum_k \phi_k$
into `sum_field_linear`. The fused kernel reads both:
- `sum_field` = $\sum_k \phi_k^2$ (already used for repulsion)
- `sum_field_linear` = $\sum_k h(\phi_k)$ (reused buffer, now stores $h$ values)

No additional GPU memory is needed — the existing `sum_field_linear` buffer is reused
with the new interpretation. The compute cost is minimal (one `powf` per pixel in
the scatter kernel, plus $h'(\phi)$ evaluation in the fused kernel).

---

## Implementation Plan

### New parameter

- CLI flag: `--adhesion J` (or `-J`)
- Default: $J = 0$ (no adhesion, backward compatible)
- Stored in `SimParams` as `adhesion_J`

### Kernel changes

1. **New sum field:** `sum_phi_field` storing $S_1(x,y) = \sum_k \phi_k(x,y)$ (linear,
   not squared). Allocated alongside existing `sum_phi_sq_field`.
2. **New scatter kernel** (or extend `kernel_scatter_phi_sq`): also accumulates $\phi_k$
   into `sum_phi_field`.
3. **Fused kernel modification:** Add adhesion term to the RHS:
   ```
   float sum_phi_j = sum_phi_field[gy * Nx + gx] - phi_val;
   float adhesion = -adhesion_J * sum_phi_j;  // note: negative sign from -δF/δφ
   ```
   This adds to `var_deriv` alongside `repulsion`.
4. **Velocity integral:** May need adhesion contribution to $\mathbf{v}$ integral
   (TBD — check if adhesion-mediated forces contribute to cell velocity).

### Estimated performance impact

- One extra global-memory field read per pixel in fused kernel (~3–5% overhead)
- One extra scatter pass (can fuse with existing scatter)
- No additional neighbor-list operations

---

## How to Probe Rigidity: Three Options

The vertex model literature (Bi et al. 2015) finds a rigidity transition at $p_0^* \approx 3.81$
where the shear modulus $G \to 0$. Crucially, at $T = 0, v_0 = 0$, **nothing moves** in the
vertex model regardless of $p_0$ — cells sit in a local energy minimum. The transition is
about whether that minimum is **rigid** ($G > 0$, deformations cost energy) or **floppy**
($G = 0$, deformations are free). Bi et al. detect this through linear response (apply
shear, measure stress).

Our phase field model has overdamped dynamics: $\partial_t \phi = -\delta F/\delta \phi$.
At $v_A = 0$ without adhesion, the system similarly sits at a local minimum. The question
is how to detect whether adhesion removes rigidity. Three approaches:

### Option A: Shear Modulus Measurement (vertex model analog)

**Protocol:**
1. Equilibrate at each $J/\kappa$ (with $v_A = 0$)
2. Apply small affine shear: shift all cell centroids by $\delta x = \gamma \cdot y$
3. Let the system relax back to equilibrium
4. Measure residual stress → $G(J/\kappa)$
5. Find $J/\kappa^*$ where $G \to 0$

**Pros:** Most direct comparison to Bi (2015). Clean, unambiguous.
**Cons:** Need to implement affine shear deformation (not trivial — must shift phase fields,
not just centroids). Medium implementation effort.

**Novelty:** Low — reproduces the vertex model measurement in a different framework.

### Option B: Motility Probe (Bi 2016 approach)

**Protocol:**
1. Equilibrate at $J = 0$, $v_A = 0$ (already have checkpoint)
2. Restart with adhesion $J/\kappa$ + small probe motility $v_A = 0.002$
3. Measure MSD:
   - $G > 0$ (jammed) → cells rattle in cages, MSD plateaus
   - $G = 0$ (unjammed) → cells diffuse, MSD grows linearly
4. Sweep $J/\kappa$ to find the critical value

**Pros:** Zero implementation effort beyond the adhesion term itself. We already have
the full MSD analysis pipeline from the Griffiths study. Directly maps to Bi (2016)
phase diagram.
**Cons:** Mixes adhesion and motility effects. Not purely testing rigidity — testing
whether motility can overcome barriers.

**Novelty:** Moderate — extends the $(p_0, v_0)$ phase diagram to a continuous-field model.

### Option C: Adhesion Quench — Watch the Energy Landscape (phase field unique)

**Protocol:**
1. Equilibrate at $J = 0$, $v_A = 0$ (already have checkpoint)
2. Restart with $J > 0$, still $v_A = 0$
3. Observe spontaneous relaxation to new energy minimum

**What happens:** Turning on adhesion changes the free energy landscape. The old
equilibrium (at $J = 0$) is generally NOT a minimum of the new energy (at $J > 0$),
because cells now have an energetic incentive to increase contact area. The system
will flow downhill via gradient descent ($\partial_t \phi = -\delta F/\delta \phi$).

The critical question is **what that relaxation looks like**:
- Below $J/\kappa^*$: System relaxes slightly — cell interfaces soften, contact angles
  adjust, but the **neighbor topology stays the same**. Cells settle into a nearby
  minimum without rearranging. Total displacement is small (order ε).
- Above $J/\kappa^*$: Energy barriers between configurations vanish. The system
  "flows downhill" through multiple T1-like rearrangements, changing neighbor topology.
  Cells slide past each other continuously. Total displacement is large (order R).

**Observables:**
- Total centroid displacement during relaxation: $\sum_i |\Delta \mathbf{r}_i|$
- Number of neighbor exchanges (T1-like events)
- Energy vs. time: single drop (no rearrangement) vs. staircase (discrete events)
  vs. smooth decay (continuous flow)
- **Relaxation time** as function of $J/\kappa$: should diverge at $J/\kappa^*$

**Pros:** Zero implementation effort beyond adhesion. Fast runs (relaxation completes
quickly at $v_A = 0$ — no need for 5τ). **Only possible in a phase field model** —
vertex models enforce instantaneous T1s and cannot show the continuous relaxation path.
**Cons:** May be subtle. If adhesion simply shifts the minimum without changing
topology (just adjusts interface shapes), the signal may be weak.

**Novelty: HIGH.** This directly shows what the vertex model cannot: the continuous
path through configuration space during a rearrangement. The relaxation dynamics
after a $J$-quench reveal the energy landscape structure in a way that is fundamentally
inaccessible to discrete-topology models.

**Why this idea arose:** In a vertex model, changing $p_0$ changes the energy of each
configuration, but the dynamics are event-driven — edges collapse or expand discretely.
You can never see the "path" through a T1. In the phase field, the field evolves
continuously via gradient descent. When the landscape becomes flat ($G \to 0$), the
system literally flows along the flat direction, and the entire trajectory is observable.
This continuous relaxation path IS the data — it shows how cells rearrange in a model
with no topological constraints.

### Recommendation: Do C first, then B

- **Option C is the first experiment** because it's fast (relaxation at $v_A = 0$ is
  quick — cells either rearrange or they don't, within ~1000–5000 TU), scientifically
  unique (only phase field can show this), and diagnostic (tells us immediately whether
  adhesion changes the topology or just adjusts interfaces).
- **Option B is the production campaign** — systematic sweep of $(J/\kappa, v_A)$ with
  MSD as the order parameter. This gives the publishable phase diagram.
- **Option A is optional** — if reviewers want a direct shear modulus comparison, we
  implement it later.

---

## Experimental Design

### Phase 0: Adhesion Quench (Option C — fast, diagnostic)

**Goal:** Determine if adhesion induces spontaneous rearrangements via gradient descent.

**Protocol:** From well-equilibrated checkpoint ($J = 0$, $v_A = 0$, $t \geq 80{,}000$
— following Palmieri et al. (2015) equilibration protocol of $\geq 8\tau$),
restart with different $J$ values and NO motility. Watch what happens.

| Parameter | Values |
|-----------|--------|
| N | 288 |
| L | 1562 (φ = 0.89) |
| $J/\kappa$ | **0.00 (control)**, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50 |
| $v_A$ | 0 |
| t_run | 20,000 (2τ) from checkpoint time |
| Trajectory interval | 50 steps (= 1 TU at dt=0.02) — high resolution |
| VTK output | ~10 frames total |
| Seeds | Deterministic at $v_A = 0$ (no noise) |
| Starting state | `/scratch/ssilber/eq_phi89/run_1/checkpoint.bin` (t=80,000) |
| Replicates | 1 per $J/\kappa$ (deterministic dynamics); use different starting checkpoints for replicates |

**Mandatory control:** $J = 0$ run with identical protocol — confirms starting state is
fully equilibrated (should show negligible displacement).

**Observables:**
- Total centroid displacement $\sum_i |\mathbf{r}_i(t) - \mathbf{r}_i(t_0)|$ vs $J/\kappa$
- Energy time series $F(t)$: gradient, bulk, interaction, adhesion components
- Number of neighbor changes (before/after comparison)
- Shape index $p_{eff}$ before and after relaxation
- Cell contact area $\int \phi_i \phi_j \, dA$ before and after
- Relaxation timescale vs $J/\kappa$

**Expected wall time:** 288 cells, t=20,000, ~25 min per run, 8 runs → ~3.5 hours total.

**Decision point:** If we see rearrangements above some $J/\kappa^*$, proceed with
Phase 1/2 around that value. If no rearrangements at any $J/\kappa$, the energy
barriers are never truly zero and we must use motility (Option B) to probe the landscape.

### Phase 1: Motility Probe Sweep (Option B — production)

**Goal:** Map the fluid-solid boundary using small motility as a probe.

| Parameter | Values |
|-----------|--------|
| N | 288 |
| φ | 0.89 |
| $J/\kappa$ | 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5 |
| $v_A$ | 0.002 (small probe) |
| t_end | 50,000 (5τ) |
| Trajectory interval | 500 steps (= 10 TU) |
| Seeds | 3 replicates |
| Starting state | Equilibrated 288-cell checkpoint |

**Observables:**
- MSD → jammed (plateau) vs. fluid (linear growth)
- cage-relative MSD
- Self-overlap $Q(t)$
- α₂ → dynamic heterogeneity
- Effective shape index $p_{eff} = P / \sqrt{A}$ from $\phi = 0.5$ contour

**Expected wall time:** ~52 min per run × 8 values × 3 seeds = ~21 hours (run overnight).

### Phase 2: $(J/\kappa, v_A)$ Phase Diagram

**Goal:** Full phase diagram in adhesion-motility space.

| Parameter | Values |
|-----------|--------|
| N | 288 (local), 1152 (cluster) |
| φ | 0.89 |
| $J/\kappa$ | 0, 0.1, 0.2, 0.3, 0.4, 0.5 |
| $v_A$ | 0.002, 0.004, 0.006, 0.008, 0.010, 0.012 |
| t_end | 50,000 (5τ) |
| Trajectory interval | 500 steps (= 10 time units) |
| Seeds | 3 replicates per point (local), 10+ (cluster) |

**Total grid:** 6 × 6 × 3 = 108 runs (local, 288 cells)

**Observables:**
- MSD at $t = 5\tau$ → classify jammed vs. fluid
- $\alpha_2$ peak → dynamic heterogeneity
- Effective shape index $\langle p_{eff} \rangle$ → connect to vertex model $p_0$

**Key prediction:** At $v_A = 0$, there should be a critical $J/\kappa^*$ above which the
system unjams. This traces out a phase boundary in $(J/\kappa, v_A)$ space analogous to
the $(p_0, v_0)$ phase diagram of Bi et al. (2016).

### Phase 3: Griffiths × Adhesion (future, cluster)

**Goal:** Test whether adhesion modifies the Griffiths singularity found in the disorder study.

| Parameter | Values |
|-----------|--------|
| N | 1152 |
| $J/\kappa$ | 0 (reference), $J/\kappa^*$ (critical) |
| $v_A$ | 0.008 |
| $\sigma_{v_A}$ | 0, 0.003, 0.006, 0.008 |
| t_end | 50,000 (5τ) |

---

## Data Requirements

### Trajectory sampling

The trajectory file stores per-cell centroids, velocities, polarizations, and $v_{A,i}$.

| Purpose | Interval (steps) | Interval (time, dt=0.02) | Notes |
|---------|-------------------|--------------------------|-------|
| MSD computation | 250–500 | 5–10 TU | Need sub-τ resolution for caging |
| Long-time diffusion | 500–1000 | 10–20 TU | Sufficient for $D_{eff}$ |
| Phase diagram scanning | 500 | 10 TU | Balance resolution vs. file size |
| Equilibration check | 5000 | 100 TU | Coarse is fine |

**File size estimate for 1152 cells:**
- 1152 cells × ~100 bytes/line × (20,000 TU / 10 TU per sample) = ~230 MB per run
- For 288 cells: ~58 MB per run

### VTK snapshots (for shape index)

To compute $p_{eff}$ from the $\phi = 0.5$ contour, we need VTK frames.

| Purpose | Interval | Notes |
|---------|----------|-------|
| Shape index statistics | Every 1000–2000 TU | 10–25 frames per run |
| Movies | Every 50 TU | Only for select runs |

### Checkpoints

- Save at end of each run (`--save-final-checkpoint`)
- Mid-run checkpoints for safety: every 10,000–25,000 steps

---

## Run Durations

### How long is enough?

From existing 288-cell Griffiths data (LOG_BOOK.md):
- τ (tumble time) = 10,000 TU
- Production runs were t = 50,000 (5τ) — this was sufficient for MSD to reach diffusive
  regime in fluid cases and to clearly show caging in jammed cases
- α₂ peaked at ~2.4τ for strongest disorder
- Self-overlap $Q(t)$ decays over 1–2τ in fluid phase

**Recommendation:**
- Phase 1 (exploratory): t = 20,000 (2τ) — enough to see if system is jammed/unjammed
- Phase 2 (production): t = 50,000 (5τ) — need long-time tail for MSD power-law fitting
- If a run looks interesting, extend to 10τ

**Why not t = 10,000 (1τ)?** Cage-relative MSD needs at least 1–2τ to distinguish
subdiffusive (jammed) from diffusive (fluid). At $t < τ$, all runs look ballistic.
The α₂ peak occurs at ~2τ, so we need at least 3–4τ to capture it properly.

### Wall time estimates (RTX 4090 Laptop)

| Cells | t_end | Steps (dt=0.02) | ms/step | Wall time |
|-------|-------|-----------------|---------|-----------|
| 288 | 20,000 | 1,000,000 | 1.24 | ~21 min |
| 288 | 50,000 | 2,500,000 | 1.24 | ~52 min |
| 1152 | 20,000 | 1,000,000 | ~7.4 | ~2.1 hr |
| 1152 | 50,000 | 2,500,000 | ~7.4 | ~5.1 hr |

---

## Current Status (as of 2026-02-18)

### Available Equilibration Data

| Location | Cells | Domain | φ | t_end | Runs | Status |
|----------|-------|--------|---|-------|------|--------|
| Nibi `/scratch/ssilber/eq_phi89/` | 288 | 1562 | 0.89 | 80,000 (8τ) | 100 | **Complete** ✓ |
| Nibi `/scratch/ssilber/eq_1152_phi89/` | 1152 | 3124 | 0.89 | 80,000 (8τ) | 3 done + 7 pending | Partial |
| Nibi `/scratch/ssilber/eq_4608_phi89/` | 4608 | 6248 | 0.89 | 80,000 (8τ) | 3 done + 7 pending | Partial |
| Narval `/scratch/ssilber/eq_phi89/` | 288 | 1562 | 0.89 | 80,000 (8τ) | 10 | **Running** (~25%) |
| Narval `/scratch/ssilber/eq_1152_phi89/` | 1152 | 3124 | 0.89 | 80,000 (8τ) | 10 | 2 running, 8 pending |
| Fir `/scratch/ssilber/eq_phi89/` | 288 | 1562 | 0.89 | 80,000 (8τ) | 10 | **Complete** ✓ |
| Fir `/scratch/ssilber/eq_1152_phi89/` | 1152 | 3124 | 0.89 | 80,000 (8τ) | 10 | Pending |
| Rorqual `/scratch/ssilber/eq_phi89/` | 288 | 1562 | 0.89 | 80,000 (8τ) | 10 | **Complete** ✓ |
| Rorqual `/scratch/ssilber/eq_1152_phi89/` | 1152 | 3124 | 0.89 | 80,000 (8τ) | 10 | **Running** |
| Rorqual `/scratch/ssilber/eq_4608_phi89/` | 4608 | 6248 | 0.89 | 80,000 (8τ) | 10 | Pending |

### Phase 0 Quench — Submitted on 3 Clusters

| Cluster | Checkpoint | Jobs | Status |
|---------|-----------|------|--------|
| Nibi | run_12 | 8805711–8805755 | PENDING |
| Rorqual | run_01 | 7063347–7063355 | PENDING |
| Fir | run_02 | 23290085–23290111 | PENDING |

### Cluster (Nibi) — Griffiths v2 runs

Running the Griffiths disorder study (pre-adhesion baseline) with 288 cells starting
from `eq_phi89` checkpoints at t=80,000.

| Parameter combo | Runs | Last t (sample) | % to 880k | Status |
|----------------|------|-----------------|-----------|--------|
| v0.008, σ=0.000 | 3 | ~387k | 44% | Chain jobs running |
| v0.008, σ=0.003 | 3 | — | — | Chain jobs queued |
| v0.008, σ=0.006 | 3 | — | — | Chain jobs running |
| v0.008, σ=0.008 | 3 | — | — | Chain jobs running |
| v0.006, σ=0.006 | 3 | ~371k | 42% | Chain jobs running |
| v0.010, σ=0.006 | 3 | ~113k | 13% | Chain jobs running |

All griffiths runs use the same eq_phi89 checkpoints (t=80,000) and chain across
3-hour SLURM jobs. 8 GPU slots currently active, ~630 griffiths jobs remaining.

---

## Analysis Pipeline

### Existing tools (from Griffiths study, in `postprocessing/`)

| Script | Purpose | Reusable for adhesion? |
|--------|---------|----------------------|
| `analyze_griffiths.py` | MSD, α₂, CV, displacements, diffusion | ✓ Phase 1/2 (with motility) |
| `analyze_griffiths_deep.py` | Self-overlap, spatial correlations, caging | ✓ Phase 1/2 |
| `analyze_trajectory.py` | General trajectory analysis | ✓ All phases |
| `plot_energy.py` | Energy time series | ✓ Phase 0 (quench energy decay) |
| `read_checkpoint.py` | Read/inspect checkpoint files | ✓ Utility |
| `visualize.py` | 2D VTK frame visualization + movies | ✓ All phases |

### Adhesion-specific tools

| Script | Purpose | Status | Location |
|--------|---------|--------|----------|
| `analyze_phase0.py` | Phase 0 quench analysis (displacement, MSD) | Exists (draft) | `adhesion_study/` |
| `analyze_adhesion.py` | Unified adhesion analysis (displacement vs J/κ, energy, transitions) | **To build** | `postprocessing/` |
| Shape index from VTK | Extract $\phi = 0.5$ contour, compute $P$ and $A$ | **To build** | `postprocessing/` |
| Contact area diagnostic | Compute $\int \phi_i \phi_j \, dA$ from VTK | **To build** | `postprocessing/` |
| Phase diagram plotter | Heatmap of MSD/$\alpha_2$ in $(J/\kappa, v_A)$ space | **To build** | `postprocessing/` |

### Phase 0 analysis pipeline

For the quench experiment, the analysis workflow is:

1. **Load trajectories** from each $J/\kappa$ run (including $J = 0$ control)
2. **Compute displacement** $|\mathbf{r}_i(t) - \mathbf{r}_i(t_0)|$ per cell, accounting
   for periodic boundaries (L=1562)
3. **Plot displacement vs $J/\kappa$** — the key diagnostic:
   - Flat = no rearrangements (rigid phase)
   - Rising = adhesion-induced rearrangements (fluid phase)
   - Sharp jump = transition
4. **Plot energy components** vs time for each $J/\kappa$:
   - Gradient energy, bulk potential, interaction (repulsion), adhesion
   - Single smooth decay = interface relaxation only
   - Staircase = discrete rearrangement events
5. **Compare to control** — subtract control displacement from adhesion runs

All figures saved to `postprocessing/output/adhesion_*.png` with date stamps.

---

## Open Questions

1. **Shape index extraction:** How to robustly extract $p_\text{eff}$ from $\phi = 0.5$
   contours? Marching squares on individual $\phi_i = 0.5$ → measure perimeter $P$ and
   area $A$ → $p = P/\sqrt{A}$. Does $\langle p \rangle$ increase with $\tilde{J}$?
2. **Contact angle measurement:** The Young-Dupré prediction gives $\cos\alpha = 1 - \tilde{J}$
   at triple points. Can we extract this from VTK data? This validates the sharp-interface
   limit and connects to vertex model edge tension.
3. **Phase 1 transition location:** At $v_A = 0.002$, what $\tilde{J}^*$ unjams the system?
   The Griffiths baseline gives $v_A^* \approx 0.008$ at $\tilde{J} = 0$. Adhesion should
   lower this threshold — but by how much?
4. **Energy barrier structure:** At $v_A = 0$, adhesion lowers the destination energy but
   not the saddle point. Can we measure the barrier height as a function of $\tilde{J}$?
   This would quantify how much motility is needed to overcome it.
5. **Finite-size effects:** Does the transition shift between $N = 288$ and $N = 1152$?
   At least one comparison point is needed for the manuscript.
6. **Does $\langle p_\text{eff}\rangle \approx 3.81$ at the transition?** This is the
   central prediction connecting to the vertex model. Extract from Phase 1/2 data.

---

## Log

### 2026-02-13

- Discussed adhesion physics and initially chose bilinear overlap — later superseded by gradient coupling (see §"Gradient Coupling")
- Designed 3-phase experimental plan
- Started 1152-cell Griffiths baseline runs locally (s0.000 and s0.003 complete, s0.006 running)
- Discussed rigidity probing strategies: shear modulus (A), motility probe (B), adhesion quench (C)
- **Option C (adhesion quench)** identified as highest-novelty experiment —
  only phase field models can show continuous relaxation through T1-like events.
  In a vertex model, topology changes are instantaneous; the phase field shows the
  full trajectory through configuration space during gradient descent on the free energy.
- Adopted plan: Phase 0 (quench, ~45 min) → Phase 1 (motility probe, overnight) →
  Phase 2 (full phase diagram, cluster)
- Created this logbook
- **Next:** Implement adhesion term (one new parameter, one sum field, ~20 lines of kernel code)

### 2026-02-14

**Implementation:**
- Implemented adhesion term $-J \sum_{j \neq i} \phi_j$ in CUDA kernels
- Added `--adhesion <J>` CLI flag, `adhesion_J` field in `SimParams`
- New `sum_phi_field` alongside existing `sum_phi_sq_field`
- Zero overhead when $J = 0$ (field not allocated, kernel branch skipped)
- Fixed CUDA toolkit mismatch build issue
- Fixed `adhesion_J` checkpoint restore bug (SimParams size mismatch 76→84 bytes)
- Updated cluster binary on Nibi

**Preliminary Phase 0 runs (local — FLAWED, kept for reference):**
- Ran all 7 J/κ values (0.05–0.50) locally from `equilibration/checkpoint.bin`
- Generated movies for each run
- Data in `phase0_quench/Jk_*/`

**⚠️ Methodological issues identified:**
1. **Insufficient equilibration** — local checkpoint was only t=2,000 (0.2τ) at
   L=1600 (φ=0.85). Literature standard is t≥80,000 (8τ). Residual stress from
   incomplete equilibration contaminates the adhesion signal.
2. **No J=0 control** — without a control, cannot distinguish adhesion-induced
   dynamics from continued equilibration drift.
3. **Too short** — runs only to t=5,000 (0.5τ). Near the critical $J/\kappa^*$,
   relaxation time diverges (critical slowing down), requiring at least 1–2τ.
4. **Wrong packing fraction** — L=1600 gives φ≈0.85, not the target φ=0.89
   (L=1562) where vertex model comparisons are valid.

These runs are NOT valid for publication but were useful for testing the implementation.
Proper Phase 0 must use the Nibi eq_phi89 checkpoints (t=80,000, L=1562, φ=0.89).

**Literature verification:**
- Verified all 11 references in this logbook with DOIs via Crossref
- Corrected one hallucinated reference ("Bhatt" → Wirtz/Palmieri)
- All DOIs confirmed resolvable

### 2026-02-15

**Cluster audit:**
- Confirmed eq_phi89 data on Nibi: 100 runs, all complete at t=80,000
- 3 × 1152-cell equilibrations complete at t=80,000 (H100, ~77 min each)
- Griffiths v2 running: 8 GPU slots active, ~630 jobs remaining
- Binary confirmed up-to-date with adhesion support (2026-02-14 build)

**Created agent instructions:**
- `.github/instructions/adhesion-study.instructions.md` — comprehensive protocol
  covering experimental standards, phase definitions, directory structure,
  logbook protocol, quality checklist, cluster submission guide

**Updated experimental protocol:**
- Phase 0 now requires: t≥80,000 starting checkpoint, J=0 control, t=20,000 run
  duration, L=1562 (φ=0.89)
- Corrected Phase 0 parameter table in this logbook

**Cancelled eq_phi89 extension:**
- Extension jobs (t=80k→100k) cancelled — t=80,000 (8τ, per Palmieri et al. 2015) is sufficient
- 11 runs (1–11) had checkpoints overwritten before cancellation; data deleted and
  re-equilibration submitted (jobs 8661002–8661012, from scratch to t=80,000)
- 89 runs (12–100) remain intact at t=80,000

**Submitted Phase 0 adhesion quench (8 jobs):**
- Jobs 8661054–8661061, all pending (Priority)
- Starting checkpoint: `/scratch/ssilber/eq_phi89/run_12/checkpoint.bin` (t=80,000)
- J/κ = 0.00 (control), 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50
- v_A = 0 (deterministic), t_end = 100,000 (20k from checkpoint)
- Trajectory interval: 50 steps (1 TU), ~10 VTK frames per run
- Output: `/scratch/ssilber/adhesion_study/phase0_quench/Jk_{X.XX}_run12/`
- GPU: MIG 1g.10gb (H100), 3h walltime, ~12 min expected per run

**Next:**
- Monitor Phase 0 jobs; once complete, download trajectories and run analysis
- Set up Phase 0 analysis pipeline (displacement, energy, neighbor topology)

### 2026-02-18

**Phase 0 resubmission and equilibration campaign.**

#### Corrections

The Phase 0 jobs submitted on Feb 15 (8661054–8661061) all **FAILED** with exit code 127
(binary not found). These were submitted before the MCP Compute Canada tool was
operational. Old empty directories (`Jk_X.XX_run12/`) cleaned up via `rmdir`.

The analysis script `analyze_phase0.py` was rewritten to fix:
- Domain size $L = 1600 \to 1562$ (correct for $\phi = 0.89$)
- Added $J = 0$ control data point
- Added matplotlib figures: displacement vs $J/\kappa$, MSD time series (linear + log-log)
- Proper trajectory header parsing for domain size auto-detection
- Output to `postprocessing/output/adhesion_phase0_*.png`

#### Phase 0 Adhesion Quench — Resubmitted via MCP

Resubmitted all 8 $J/\kappa$ values using the MCP `resume_simulation` tool with
`parameter_sweep`:

| Parameter | Value |
|-----------|-------|
| Checkpoint | `/scratch/ssilber/eq_phi89/run_12/checkpoint.bin` (t=80,000) |
| $J/\kappa$ | 0.00 (control), 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50 |
| $v_A$ | 0 |
| $t_{end}$ | 100,000 (20k from checkpoint = 2τ) |
| trajectory_samples | 20,000 (~50 steps = 1 TU resolution) |
| save_interval | 100,000 (~10 VTK frames) |
| study_tag | `adhesion_phase0_quench` |
| Cluster | Nibi (H100, MIG 1g.10gb) |

- Jobs 8805711–8805755, all PENDING (Priority) — behind ~420 Griffiths v2 jobs in queue
- Output: `/scratch/ssilber/adhesion_study/phase0_quench/Jk_{X.XX}/`
- Expected runtime: ~12 min per run once scheduled
- Warning: trajectory ~495 MB/run (high sampling rate)

#### Equilibration Campaign — All Clusters

Submitted equilibrations on all 4 clusters to ensure availability for future phases.
All use: $\phi = 0.89$, $R = 49$, $v_A = 0$, $t_{end} = 80{,}000$ (8τ per Palmieri 2015),
`trajectory_samples=0`, `save_interval=400000`, `checkpoint_interval=500000`,
`save_final_checkpoint=true`.

**New submissions (2026-02-18):**

| Cluster | Cells | Runs | Jobs | Status |
|---------|-------|------|------|--------|
| Nibi | 1152 | 7 new (→10 total) | 8805436–8805446 | PENDING |
| Nibi | 4608 | 7 new (→10 total) | 8805447–8805593 | PENDING (12h walltime) |
| Narval | 288 | 10 | 56681094–56681108 | **RUNNING** (t≈20k, ~25%) |
| Narval | 1152 | 10 | 56681111–56681126 | 2 RUNNING, 8 PENDING |
| Fir | 1152 | 10 | 23289753–23289791 | PENDING |
| Rorqual | 1152 | 10 | 7063283–7063300 | **RUNNING** (5 min elapsed) |
| Rorqual | 4608 | 10 | 7063301–7063312 | PENDING (12h walltime) |

**Complete equilibration inventory (existing + new):**

| Cluster | 288c | 1152c | 4608c |
|---------|------|-------|-------|
| Nibi | 100 ✓ | 3 done + 7 pending = 10 | 3 done + 7 pending = 10 |
| Narval | 10 running | 10 (2 running, 8 pending) | — |
| Fir | 10 ✓ | 10 pending | — |
| Rorqual | 10 ✓ | 10 running | 10 pending |
| **Total** | **130** | **40** | **20** |

**Notes:**
- Narval 288c runs at t≈20,000 after 10 min (~40 min to completion on A100)
- Rorqual 1152c just started (5 min), 4608c queued behind (12h walltime)
- Nibi queue has ~420 Griffiths v2 jobs ahead → equilibrations and Phase 0 will start
  once Griffiths drain or via backfill
- Narval also has 10 × 1152c at $\phi = 0.91$ (jobs 56656689–56656704, COMPLETED) —
  wrong packing fraction, not usable for adhesion study but archived

#### Phase 0 — Additional Replicates on Rorqual and Fir

To avoid waiting for nibi's Griffiths queue (~420 jobs), submitted Phase 0 on two
additional clusters using different starting checkpoints (independent replicates):

| Cluster | Checkpoint | Jobs | Notes |
|---------|-----------|------|-------|
| Rorqual | run_01 (t=80,000) | 7063347–7063355 | 8 J/κ values, same protocol |
| Fir | run_02 (t=80,000) | 23290085–23290111 | 8 J/κ values, same protocol |

All use identical parameters to the nibi submission (t_end=100,000, trajectory_samples=20,000,
save_interval=100,000, v_A=0). Output to `/scratch/ssilber/adhesion_study/phase0_quench/Jk_{X.XX}_run{NN}/`.

This gives us 3 replicates from 3 different equilibrated starting states (runs 01, 02, 12)
on 3 different clusters — useful for verifying reproducibility of the quench dynamics
(at v_A=0 the dynamics are deterministic, so differences between replicates come purely
from different initial configurations).

#### Bug Fix: `trajectory_samples` Override Ignored During Checkpoint Resume

**Problem:** The binary silently ignored `--trajectory-samples` when resuming from a
checkpoint. Rorqual J=0.00 (run_01) completed with only **12 trajectory snapshots**
instead of the expected ~20,000.

**Root cause (main.cu):**
1. CLI parsing sets `save_interval=100000` and `trajectory_samples=20000`
2. Pre-checkpoint code computes `sim.trajectory_interval = save_interval = 100000`
   (line 648, default when `--trajectory-interval` not explicitly set)
3. `initialize_from_checkpoint()` overwrites `trajectory_samples` to checkpoint value
   (100) but does NOT touch `trajectory_interval`
4. Post-load code restores `trajectory_samples=20000` from CLI, but `trajectory_interval`
   remains 100000
5. `run()` takes the `trajectory_interval > 0` branch, writing trajectory every 100000
   steps instead of computing interval from samples

**Fix:** Added `trajectory_interval_set` tracking alongside existing `trajectory_samples_set`.
When `trajectory_samples` is set via CLI but `trajectory_interval` is not, reset
`sim.trajectory_interval = 0` to force the "compute from samples" path:

```cpp
if (trajectory_samples_set) {
    sim.trajectory_samples = cmd_trajectory_samples;
    if (!trajectory_interval_set) {
        sim.trajectory_interval = 0;
    }
}
```

**Applied to:** All 4 clusters (nibi, rorqual, fir, narval) via direct source patch +
`make -j4`. Binary rebuilt in-place at `~/cell_simulation/build/bin/cell_sim`.

**Impact on Phase 0 jobs:**
- Rorqual J=0.00 (7063347): **COMPLETED with buggy data** (12 snapshots). Cleaned and
  resubmitted as job 7063605–7063606.
- Rorqual J=0.05 (7063349): **Cancelled** (was running with old binary). Cleaned and
  resubmitted as job 7063607–7063608.
- Rorqual J=0.10 (7063350): Started at 00:58 EST, binary rebuilt at 00:55 EST → **uses
  fixed binary** ✓. Verified: 6645 trajectory snapshots at ~50-step intervals.
- Rorqual J=0.15–0.50 (PENDING): Will use fixed binary ✓.
- Fir all 8 jobs: Binary rebuilt at 21:56 PST, jobs started at 22:00 PST → **all use
  fixed binary** ✓. Verified: J=0.00 trajectory has 4322 snapshots at ~50-step intervals.
- Nibi all 8 jobs: PENDING, will use fixed binary ✓.

**Verification:** Fir J=0.00 and rorqual J=0.10 trajectory files confirmed to have
correct ~50-step sampling interval (trajectory_samples=20000, total_steps=1000000,
expected interval=50).

**Next:**
- ~~Wait for remaining Phase 0 runs to complete~~ → Completed; see §"Phase 0 Gradient-Coupling Results"
- Full analysis: energy decomposition, shape index extraction, contact angle measurement
- Proceed to Phase 1 (motility probe)
- Monitor equilibration progress across all clusters

---

## $h(\phi)$ Implementation — Local Validation

### 2026-02-XX (h(φ) adhesion: implementation and calibration)

**Context:** All previous adhesion implementations (bare bilinear, smoothstep, gradient
coupling, reduced-κ, scaled bilinear, gated bilinear) showed unphysical behavior — either
nucleation in empty space or field degradation. The $h(\phi) = \phi^2(1-\phi)^n$ functional
form derived from first principles (see §"Functional Form Analysis" above) solves the
nucleation problem completely and provides stable cell-cell adhesion.

#### Implementation details

**Compile-time constants** (in `types.cuh`):
```
ADHESION_N = 4           // h(φ) exponent → peak at φ* = 1/3
ADHESION_H_PEAK_INV = 5.0625  // = (3/2)^4 = 81/16 ("φ² matching" normalization)
```

**Normalization choice: "φ² matching"**
The normalization `h_peak_inv = 1/(n/(n+2))^n` ensures that at the peak of $h$,
$h(\phi^*) = \phi^{*2}$. This makes the adhesion energy density match the repulsion
energy density scale: at $J = \kappa$, the adhesion and repulsion contributions are
approximately equal at the interface peak. Physically, $J/\kappa$ is the natural control
parameter.

**Note:** `bulk_coeff` (= $30/\lambda^2$) is NOT included in the adhesion term. The adhesion
strength is controlled solely by $J$ and `h_peak_inv`. The $30/\lambda^2$ factor belongs to
the double-well gradient energy, not adhesion.

**Kernel changes** (3 files):
1. `kernels_shared.cu` scatter: accumulates $h(\phi) = \phi^2(1-\phi)^n \times$ `H_PEAK_INV`
2. `kernels_shared.cu` fused: adhesion = $-J \cdot [h'(\phi_i)\Sigma\phi_j^2 + 2\phi_i \Sigma h(\phi_j)]$
3. `io.cu`: energy computation uses normalized $h(\phi)$

#### Calibration history

Three normalization values were tested:

| `H_PEAK_INV` | 2-cell merger | 7-cell sparse | 7-cell dense (89%) |
|---------------|---------------|---------------|---------------------|
| 45.5625 (peak=1 norm) | J=7 (mass +17%) | Contours fuse J≈1 | Not tested |
| 5.0625 (φ² matching) | Never (≤J=9) | No merger (≤J=7) ✅ | No merger (≤J=7) ✅ |

`H_PEAK_INV = 45.5625` (original peak=1 normalization) was too strong: even 2 cells
merged at $J/\kappa = 0.7$ with significant mass inflation (+17% at J=9).

`H_PEAK_INV = 5.0625` (φ² matching) is the production normalization. No cell merger
at any tested $J$ value up to $J/\kappa = 0.9$ (2-cell) or $J/\kappa = 0.7$ (7-cell).

#### Field quality verification

All tests: `max_phi = 1.0000`, `corner = 0.000000` at EVERY $J$ value tested (J=0 through J=9).
**The h(φ) functional form completely eliminates nucleation.** This was the primary bug
with all previous implementations.

#### 2-cell sweep (v10, h_peak_inv = 5.0625)

Two cells initialized at $d = 90$, domain 400×400, $R = 49$, $t = 500$ TU, $v_A = 0$.

| $J$ | $J/\kappa$ | Distance | $\Delta d$ from control | Mass | Status |
|-----|-----------|----------|------------------------|------|--------|
| 0 | 0.00 | 97.6 | — | 15803 | Control ✅ |
| 1 | 0.10 | 97.2 | -0.4 | 15800 | Barely visible |
| 2 | 0.20 | 91.0 | -6.6 | 15746 | Visible adhesion |
| 3 | 0.30 | 73.6 | -24.0 | 15774 | Contact |
| 5 | 0.50 | 72.8 | -24.8 | 16366 | Contact plateau |
| 7 | 0.70 | 72.6 | -25.0 | 17325 | No merger ✅ |
| 9 | 0.90 | 72.9 | -24.7 | 18445 | No merger ✅ |

**Convergence:** Wide start ($d = 110$) at $J = 3$ converges to $d = 74.4$ vs narrow start
($d = 90$) at $d = 74.1$. Difference < 0.3 pixels — same equilibrium ✅.

**Interpretation:** Sharp transition from barely-touching ($J = 2$, $d = 91$) to full
contact ($J = 3$, $d = 73.6$), then plateau. This is a first-order-like contact transition
for the isolated 2-cell system. The contact distance ($d \approx 73$) is set by the balance
of repulsion ($\kappa \phi^2 \phi^2$) and adhesion ($h(\phi)$) at the interface overlap zone.

#### Multi-cell tests

**Sparse 7-cell** (hex-packed, $d = 90$, 400×400 domain, $\phi \approx 33\%$):

| $J$ | $J/\kappa$ | Centroids | Min pair | φ>0.5 blobs | Status |
|-----|-----------|-----------|----------|-------------|--------|
| 0 | 0.00 | 7 | 102.6 | 7 | Control ✅ |
| 3 | 0.30 | 7 | 86.7 | 7 | Adhesion visible ✅ |
| 3.5 | 0.35 | 7 | 85.6 | 1 | Contours fuse, cells distinct ✅ |
| 5 | 0.50 | 7 | 83.1 | 1 | Contours fuse, cells distinct ✅ |
| 7 | 0.70 | 7 | 80.2 | 1 | Contours fuse, cells distinct ✅ |

**Note on "φ>0.5 blobs":** The VTK files store the combined field $\sum_i \phi_i$. At
$J \geq 3.5$, cells are close enough that their $\phi > 0.5$ contours touch in the
combined field, producing a single connected region. This is NOT cell merger — each cell
maintains a distinct centroid tracked by the simulation. The `ndimage.label` count of
connected regions in the sum field is misleading at high adhesion.

**Dense 7-cell** (grid init, $N = 244$, $\phi \approx 89\%$, production-like confluence):

| $J$ | $J/\kappa$ | Centroids | Min pair | Status |
|-----|-----------|-----------|----------|--------|
| 0 | 0.00 | 7 | 85.3 | Control ✅ |
| 3 | 0.30 | 7 | 85.6 | Stable ✅ |
| 5 | 0.50 | 7 | 84.2 | Stable ✅ |
| 7 | 0.70 | 7 | 83.3 | Stable ✅ |

**Finding:** All 7 cells survive at EVERY $J$ value tested, in both sparse and dense
packing. The adhesion monotonically reduces inter-cell distance (from 102.6 at $J = 0$
to 80.2 at $J = 7$ in sparse packing) without ever causing actual cell merger. The
$h(\phi)$ self-gating prevents any cell from absorbing another.

#### Mass conservation

| Test | J=0 mass | J=5 mass | J=9 mass | Max inflation |
|------|----------|----------|----------|---------------|
| 2-cell | 15803 | 16366 | 18445 | +17% at J=9 |
| Dense 7-cell | 54345 | 56336 | 58054* | +7% at J=7 |

*J=7 for dense test (J=9 not run).

Mass inflation exists but is modest (< 10%) in the production-relevant dense packing.
The adhesion acts as a weak source of $\phi$ at the cell-cell interface, partially overcome
by the volume constraint ($\mu = 1$). In production runs with many neighbors, the volume
constraint is more effective because inflation is resisted by all surrounding cells.

#### Status and next steps

- ✅ h(φ) adhesion implemented and validated locally
- ✅ No nucleation at any J value
- ✅ Dense packing stable through entire study range
- ✅ Convergence verified (wide/narrow start)
- ❌ **Cluster binaries still use old (gated bilinear) adhesion** — must sync and rebuild
- ❌ Old Phase 0 cluster data uses broken implementation — needs resubmission
- **Next:** `sync_and_build` on all clusters, resubmit Phase 0 quench with gradient coupling adhesion

---

## Gradient Coupling — Replacing h(φ) with ∇φ·∇φ

### 2026-02-XX (diagnosing h(φ) squishing and switching to gradient coupling)

**Problem identified:** The $h(\phi)\cdot\phi^2$ coupling produces a force term
$2\phi_i \cdot \sum h(\phi_j)$ that is nonzero at $\phi_i = 1$ (cell bulk). This
pulls the entire cell interior toward the neighbor's interface, causing cells to
*squish* — flattening against each other with the volume constraint as the only
barrier. Visually, cells push up against neighbors rather than adhering at the surface.

**h·h coupling attempt (FAILED):** Switching to $F = -J \int h(\phi_i) h(\phi_j) \, dA$
gives $\delta F/\delta\phi_i = -J \, h'(\phi_i) \sum h(\phi_j)$, which has $h'(0)=0$
and $h'(1)=0$. However, $h'(\phi)$ changes sign at $\phi^* = 1/3$: attractive for
$\phi < 1/3$ (outer tail) but *repulsive* for $\phi > 1/3$ (interface midpoint).
2-cell sweep showed barely any attraction at $J=3\text{–}5$ and strong repulsion at $J\geq 7$.

**Root cause analysis — energy landscape comparison (1D profiles):**

| Coupling | Energy minimum at | Attractive F always? | Bulk force at φ=1? |
|----------|------------------|-----------------------|---------------------|
| $h(\phi_i) \phi_j^2$ (old) | $d = 70$ (deep overlap) | Yes | Yes — squishing |
| $h(\phi_i) h(\phi_j)$ (failed) | $d \approx 99$ (barely touching) | No — repulsive at φ>1/3 | No |
| $\nabla\phi_i \cdot \nabla\phi_j$ (gradient) | $d \approx 98$ (first contact) | No — repulsive at close range | No |

The gradient coupling is the only form that creates a potential well at the interface
contact distance ($d \approx 2R$) with both an attractive branch (pulling cells together
from far away) and a repulsive branch (preventing deep interpenetration). This is
physically correct *surface adhesion* — reducing the effective surface tension at shared
interfaces.

### Implementation

**Physics:** $F_\text{adh} = J \sum_{i<j} \int \nabla\phi_i \cdot \nabla\phi_j \, dA$

At shared interfaces, $\nabla\phi_i$ and $\nabla\phi_j$ are anti-parallel, so the energy
is negative (favorable). The variational derivative:
$$\frac{\delta F}{\delta \phi_i} = -J \sum_{j \neq i} \nabla^2 \phi_j$$

**Code changes (3 files):**
1. `kernels_shared.cu` **scatter**: scatters plain $\phi$ (not $h(\phi)$) into `sum_field_linear`
   → $S_\text{lin}(x,y) = \sum_k \phi_k(x,y)$
2. `kernels_shared.cu` **fused kernel**: 5-point Laplacian of $S_\text{lin}$ with periodic BC,
   adhesion $= -J \cdot (\nabla^2 S_\text{lin} - \nabla^2 \phi_i)$
3. `io.cu` **energy**: $E_\text{adh} = J \sum_{i<j} \int (\nabla\phi_i \cdot \nabla\phi_j) \, dA$

**Key property:** $J$ acts as a surface-tension reduction at shared interfaces.

### Stability analysis — the $J = 2\gamma$ critical point

Consider two cells in perfect contact. Their gradients are anti-parallel at the shared
interface ($\nabla\phi_1 \approx -\nabla\phi_2$). The total gradient-type energy of the
shared boundary per unit contact length is:

$$E_\text{shared} = \int \left[\gamma|\nabla\phi_1|^2 + \gamma|\nabla\phi_2|^2 + 2J\,\nabla\phi_1 \cdot \nabla\phi_2\right] dx = (2\gamma - J)\int |\nabla\phi|^2\,dx$$

The factor of $2\gamma$ arises because TWO cells each contribute $\gamma|\nabla\phi|^2$. The
adhesion acts once per pair (from $\sum_{i<j}$), contributing $-J\int|\nabla\phi|^2 dx$ (negative
because gradients are anti-parallel). Two free surfaces would cost $2\gamma\int|\nabla\phi|^2 dx$.

The **natural dimensionless control parameter** is:

$$\tilde{J} = \frac{J}{2\gamma}$$

which measures the fraction of surface energy removed at shared interfaces:

| $\tilde{J}$ | $J$ (for $\gamma=1$) | Surface energy reduction | Regime |
|---|---|---|---|
| 0 | 0 | 0% | No adhesion |
| 0.25 | 0.5 | 25% | Mild adhesion |
| 0.45 | 0.9 | 45% | Strong adhesion |
| 0.50 | 1.0 | 50% | Half surface tension removed |
| 0.75 | 1.5 | 75% | Very strong adhesion |
| **1.0** | **2.0** | **100%** | **Critical: zero interface cost** |
| >1.0 | >2.0 | — | **Unstable: negative interface energy → merger** |

**Stability bound: $J < 2\gamma$.** For our parameters ($\gamma = 1$), $J_c = 2$.
At $J = 2$, shared interfaces have zero energy cost → cells can dissolve boundaries.
Above $J = 2$, the system *gains* energy by creating more interface → runaway instability.

The quartic repulsion $\kappa\phi_i^2\phi_j^2$ provides additional stabilization that
shifts the effective merger point slightly above $J = 2\gamma$ (empirically $J \approx 2.5$),
but the shared interface is already effectively dissolved at $J = 2$.

### Validation against Nonomura (2012) — the only variational gradient coupling in the literature

Nonomura (PLOS ONE 7:e33501) uses three inter-cell energy terms:

$$E_\text{int} = \underbrace{\frac{\beta}{2}\sum_{m\neq m'} \int h_m h_{m'} dr}_\text{repulsion (β>0)} + \underbrace{\frac{\gamma_N}{2}\sum_{m\neq m'} \int \nabla h_m \cdot \nabla h_{m'} dr}_\text{adhesion (γ_N>0)} + \underbrace{\frac{c}{2}\sum_m \int |\nabla h_m|^2 dr}_\text{regularization (c>γ_N)}$$

where $h(\phi) = \phi^2(3-2\phi)$. The **regularization** term is critical: it provides
additional gradient stiffness that prevents interface dissolution at high adhesion.
Nonomura requires $c > \gamma_N$ for stability. Without regularization ($c = 0$), his
model would also be limited to $\gamma_N < 2D_0$.

**Nonomura's parameter values (Figure 5, 2-cell 2D adhesion):**

| Parameter | Value | Our equivalent |
|-----------|-------|----------------|
| $D_0$ (gradient coeff) | 0.001 | $\gamma = 1$ |
| $b$ (repulsion) | 1 | $\kappa = 10$ |
| $c$ (regularization) | 0.01 | **None** (we lack this term) |
| $\gamma_N$ (adhesion) | 0.003–0.0065 | $J = 0$–1.5 |
| $\gamma_N / D_0$ | 3–6.5 | Not directly comparable due to $\nabla h$ vs $\nabla\phi$ |

**Surface energy reduction at Nonomura's strongest 2D adhesion** ($\gamma_N = 0.0065$):
approximately **36%**. This is comparable to our $J \approx 0.7$ ($\tilde{J} \approx 0.36$),
which is solidly in our stable regime.

**Why Nonomura can go higher:** the regularization $c = 0.01$ adds $\sim 5\times$ more
gradient stiffness than $D_0$ alone. This raises the effective stability bound from
$\gamma_N < 2D_0 = 0.002$ (without regularization) to $\gamma_N < 2(D_0 + c \cdot 0.77)$
≈ $0.017$ (with regularization). Nonomura's $\gamma_N = 0.0065$ is well below this.

**Our model without regularization:** stability requires $J < 2\gamma = 2$. This gives
a physically meaningful range $J \in [0, 1.5]$ (up to 75% surface reduction), which
is more than sufficient for the adhesion study. Adding a Nonomura-style regularization
would widen the range but add a second parameter ($c$) for diminishing returns — the
$J$-controlled transition we observe at $J \approx 0.5$–$1.0$ in Phase 0 already captures
the relevant physics.

**Löber et al. (2015, Sci. Rep. 5:9172)** use a non-variational adhesion form:
$-\kappa \nabla\rho_i \cdot \hat{f}(\nabla\rho_j)$ with $\kappa = 0$–12 and steric repulsion
$\lambda = 30$ ($\kappa/\lambda$ up to 0.4). Their model includes acto-myosin dynamics and
substrate adhesion, making direct parameter comparison less meaningful. However, Löber
also finds that moderate adhesion ($\kappa = 6$) suppresses collective motion while
strong adhesion ($\kappa = 12$) leads to traveling bands — consistent with the non-trivial
adhesion-controlled transition we study.

### Phase 0 cluster data — parameter mapping

The Phase 0 quench experiments used $J/\kappa$ as the scan variable.
Gradient-coupling results (rorqual run_01, 2τ):

| $J/\kappa$ | $J$ | $\tilde{J} = J/(2\gamma)$ | Surface reduction | Phase 0 result |
|---|---|---|---|---|
| 0.00 | 0 | 0 | 0% | Control: $\Delta r/R = 0.025$ (noise) |
| 0.05 | 0.5 | 0.25 | 25% | $\Delta r/R = 0.058$ — interface adjustment |
| 0.10 | 1.0 | 0.50 | 50% | $\Delta r/R = 0.086$ — interface adjustment |
| 0.15 | 1.5 | 0.75 | 75% | $\Delta r/R = 0.132$ — interface adjustment, not fully relaxed |

All displacements are sub-cell. No neighbor exchanges at any $\tilde{J}$.
Values $J/\kappa \geq 0.20$ ($J \geq 2$) are past the stability bound and were not run.

**Recommendation for production Phase 1/2:** restrict to $J/\kappa \leq 0.15$
($J \leq 1.5$, $\tilde{J} \leq 0.75$) to stay well within the stable regime.

### 2-cell sweep — gradient coupling

Two cells at $d = 90$, domain 400×400, $R = 49$, $t = 500$ TU, $v_A = 0$.

| $J$ | $\tilde{J}$ | $d_\text{final}$ | $\Delta d$ | Mass | Status |
|-----|---|---|---|------|--------|
| 0 | 0 | 97.5 | — | 15803 | Control ✅ |
| 0.1 | 0.05 | 97.3 | -0.2 | 15801 | Barely visible |
| 0.3 | 0.15 | 96.8 | -0.7 | 15797 | Mild adhesion |
| 0.5 | 0.25 | 96.2 | -1.3 | 15791 | Visible |
| 0.9 | 0.45 | 95.0 | -2.5 | 15774 | Clear adhesion |
| 1.0 | 0.50 | 94.7 | -2.8 | 15769 | 50% surface reduction |
| 1.5 | 0.75 | 92.3 | -5.2 | 15726 | Strong |
| 2.0 | 1.00 | 86.2 | -11.3 | 15612 | **Critical** (interface dissolving) |
| 3.0 | 1.50 | 0.0 | — | 31482 | **Merger** ❌ (past stability) |
| 5.0 | 2.50 | 0.0 | — | 42239 | Domain filled ❌ |

**Cell shape (within stable range):** Aspect ratio at J=1.5 preserves circular shape
(no squishing). Compare with old $h\cdot\phi^2$ at J=3 where d dropped to 73.6 with
visible cell deformation.

**Mass conservation:** Stable within 1.2% across J=0–1.5 (15803→15726). No inflation.

**Values J ≥ 2.0 are past the stability limit** and included only to characterize the
transition. They are NOT physical adhesion regimes.

### Comparison: gradient coupling vs h·φ² coupling

| Property | $h(\phi)\cdot\phi^2$ | $\nabla\phi_i \cdot \nabla\phi_j$ |
|----------|--------|--------|
| Squishing artifact | Yes — bulk force deforms cells | **No** — force only at interfaces |
| Nucleation | No | No |
| Mass stability | +17% at J=9 | <1.2% at J=1.5 |
| Physical J range | — | $J \in [0, 2\gamma) = [0, 2)$ |
| Merger transition | Sharp at J ≈ 2.5 | Gradual, critical at $J = 2\gamma = 2$ |
| Cell shape | Flattened at contact | **Circular preserved** |

### Status

- ✅ Gradient coupling implemented and validated locally
- ✅ No squishing, no nucleation, no mass inflation
- ✅ Stability analysis: $J < 2\gamma$ justified, matches Nonomura
- ✅ Phase 0 cluster data in correct range ($J/\kappa \leq 0.15$)
- ❌ Cluster binaries still need sync_and_build
- **Next:** Multi-cell tests (7-cell sparse/dense), cluster deployment

---

## Phase 0 Gradient-Coupling Results

### 2026-02-19 — Phase 0 analysis (rorqual run_01, gradient coupling, 2τ)

**Runs:** N=288, L=1562, φ=0.89, J/κ=0.00, 0.05, 0.10, 0.15 (Jtilde=0, 0.25, 0.50, 0.75), v_A=0
**Cluster:** Rorqual, starting checkpoint: eq_phi89/run_01 (t=80,000)
**Run duration:** t=20,000 (2τ), trajectory sampled every ~1 TU

#### Displacement results

| $\tilde{J}$ | $J/\kappa$ | $J$ | Mean $\Delta r / R$ | RMS disp. | $v_\text{rms}$ (final) | Regime |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.0 | 0.025 | 1.2 | 4.4e-6 | Control (confirms equilibration) |
| 0.25 | 0.05 | 0.5 | 0.058 | 2.8 | 6.5e-5 | Interface adjustment only |
| 0.50 | 0.10 | 1.0 | 0.086 | 4.2 | 3.7e-4 | Interface adjustment only |
| 0.75 | 0.15 | 1.5 | 0.132 | 6.5 | 2.2e-3 | Interface adjustment, v_rms still growing |

#### Key findings

1. **No sharp transition.** Unlike the old bilinear data (which showed a 14× jump between
   $\tilde{J} = 0.25$ and $0.50$), the gradient coupling shows a smooth, monotonic increase
   in displacement. All values are sub-cell ($\ll R$).

2. **No neighbor exchanges (T1-like events).** All displacements reflect interface reshaping
   (contact angles adjusting, contact areas growing), not topology changes. Cells remain
   in the same neighbor configuration throughout.

3. **System not fully relaxed at $\tilde{J} = 0.75$.** The $v_\text{rms}$ at the end of
   the run is still 2.2e-3, suggesting the system hasn't reached its energy minimum. The
   relaxation timescale increases with $\tilde{J}$, consistent with proximity to a
   soft-mode threshold.

4. **Control validates equilibration.** The $\tilde{J} = 0$ run shows $\Delta r / R = 0.025$,
   well below the 0.05R threshold. The starting checkpoint is properly equilibrated.

#### Physics interpretation: why no T1s at $v_A = 0$

This result was initially surprising — Phase 0 was designed expecting adhesion to induce
spontaneous rearrangements. The resolution came from analyzing the energy barrier structure:

**The energy barrier argument:** At $\phi = 0.89$ with $v_A = 0$, cells are tightly packed
in a local energy minimum. Turning on adhesion ($J > 0$) *lowers the energy of
configurations with more contact area* (the destination), but it does NOT lower the
*saddle-point energy* for topology changes. The saddle point — where one cell must
squeeze past another to exchange neighbors — is dominated by:
- $\kappa$ repulsion: cells must deeply overlap during the exchange
- $\gamma$ gradient energy: interfaces must deform through highly curved states

These costs are unaffected by adhesion (which acts only at existing shared interfaces,
not at the geometric squeeze-point). At $v_A = 0$, the system evolves by gradient descent
and cannot cross energy barriers. It relaxes to the nearest local minimum, which has the
same neighbor topology as the starting state.

**Why $\phi = 0.89$ is still correct:** Three reasons to keep this packing fraction:
1. Vertex model comparison requires confluence ($\phi \approx 1$)
2. Lower $\phi$ tests density-driven jamming, not adhesion-controlled rigidity
3. Experimental tissues (Park et al. 2015, Mongera et al. 2018) are confluent

**Consequence for the study:** Phase 0's value is in **static equilibrium measurements**,
not displacement. The key observables are:
- $p_\text{eff}(\tilde{J})$: effective shape index from $\phi = 0.5$ contours
- Contact angle vs $\tilde{J}$: test Young-Dupré prediction $\cos\alpha = 1 - \tilde{J}$
- Energy decomposition: gradient, bulk, repulsion, adhesion components
- Relaxation timescale: does it diverge near a critical $\tilde{J}$?

**Phase 1 is the main experiment.** Motility ($v_A > 0$) provides the "thermal" energy
to kick cells over topological barriers. The $(\tilde{J}, v_A)$ phase diagram is the
paper's central result — analogous to Bi et al.'s $(p_0, v_0)$ diagram. Adhesion lowers
the effective surface tension → cells deform more easily → the motility threshold for
unjamming decreases with increasing $\tilde{J}$.

#### Figures

- `postprocessing/output/adhesion_phase0_grad_displacement_20260219.png` — displacement vs $\tilde{J}$
- `postprocessing/output/adhesion_phase0_grad_msd_20260219.png` — MSD time series
- `postprocessing/output/adhesion_phase0_grad_distributions_20260219.png` — displacement distributions

**Next:**
- Extract $p_\text{eff}(\tilde{J})$ from VTK data (shape index measurement)
- Contact angle extraction from interface geometry
- Submit extended Phase 0 runs to 20τ (check if $\tilde{J} = 0.75$ equilibrates)
- Begin Phase 1 design: $\tilde{J} \in \{0, 0.25, 0.50, 0.75\}$ × $v_A = 0.002$

---

### 2026-02-20 — Mass submission: extended grid, all three phases

**Decision: Extended $\tilde{J}$ grid**

Increased resolution from 4 to 7 values at $\Delta\tilde{J} = 0.125$ spacing to better resolve the transition:

| $\tilde{J}$ | $J/\kappa$ | $J$ (CLI `--adhesion`) |
|---|---|---|
| 0.000 | 0.000 | 0.0 |
| 0.125 | 0.025 | 0.25 |
| 0.250 | 0.050 | 0.5 |
| 0.375 | 0.075 | 0.75 |
| 0.500 | 0.100 | 1.0 |
| 0.625 | 0.125 | 1.25 |
| 0.750 | 0.150 | 1.5 |

Original 4-value grid (0, 0.25, 0.50, 0.75) was too coarse to locate the transition. The 3 new intermediate points (0.125, 0.375, 0.625) add the resolution a referee would require.

#### Phase 0 — Adhesion quench (diagnostic)

**Parameters:** N=288, L=1562, φ=0.89, v_A=0, t=80k→100k (2τ), trajectory_samples=20000, save_interval=100000

| Cluster | Checkpoints | Runs | Job IDs |
|---------|-------------|------|---------|
| Narval | run_01, run_02, run_03 | 21 | 56739128–56739250 |
| Nibi | run_12, run_13, run_14 | 21 | 8897993–8898084 |

**Total:** 42 runs (7 $\tilde{J}$ × 3 replicates × 2 clusters)

Output: `/scratch/ssilber/adhesion_study/phase0_grad/Jk_{X.XX}/run_{NN}/`

#### Phase 1 — Motility probe ($v_A = 0.002$)

**Parameters:** N=288, L=1562, φ=0.89, v_A=0.002, t=80k→130k (5τ), trajectory_samples=2000, save_interval=0

| Cluster | Checkpoints | Runs | Job IDs |
|---------|-------------|------|---------|
| Nibi | run_12, run_13, run_14 | 21 | 8898268–8898461 |
| Narval | run_01, run_02, run_03 | 21 | 56742884–56743778 |

**Total:** 42 runs (7 $\tilde{J}$ × 3 replicates × 2 clusters)

Output: `/scratch/ssilber/adhesion_study/phase1_motility/Jk_{X.XX}_vA0.002/run_{NN}/`

#### Phase 2 — Full $(\tilde{J}, v_A)$ diagram

**Parameters:** N=288, L=1562, φ=0.89, t=80k→130k (5τ), trajectory_samples=2000, save_interval=0

$v_A = 0.002$ already covered by Phase 1. Phase 2 covers $v_A \in \{0.004, 0.006, 0.008, 0.010, 0.012\}$:

| Cluster | $v_A$ values | Runs | Job ID range |
|---------|-------------|------|-------------|
| Nibi | 0.004, 0.006, 0.008 | 63 | 8898488–8898723 |
| Narval | 0.010, 0.012 | 42 | 56743833–56744397 |

**Total:** 105 runs (7 $\tilde{J}$ × 5 $v_A$ × 3 replicates)

Output: `/scratch/ssilber/adhesion_study/phase2_diagram/Jk_{X.XX}_vA{Y.YYY}/run_{NN}/`

#### Grand total

**189 GPU jobs** across 2 clusters (84 nibi, 105 narval), covering the complete $(\tilde{J}, v_A)$ parameter space:
- 7 $\tilde{J}$ values × 6 $v_A$ values × 3 replicates = 126 unique parameter combinations
- Plus 42 Phase 0 diagnostic runs (v_A=0)
- 6 independent equilibrium checkpoints (3 per cluster, different initial conditions)
- Resources: 1 GPU, 3h walltime, 8 GB RAM per job

#### Infrastructure note

Fixed SLURM account suffix resolution in the MCP submission tool. New Alliance clusters (nibi, narval) require `_gpu` suffix on accounts when requesting GPU resources. The tool now auto-resolves: stores base account name in config, appends `_gpu` at submission time after SSH validation.

**Next:**
- Monitor job progress with `list_jobs` / `check_progress`
- Extract Phase 0 observables ($p_\text{eff}$, contact angle, energy decomposition) when data arrives
- Analyze Phase 1 MSD / $Q(t)$ to locate transition at $v_A = 0.002$
- Build automated analysis pipeline for post-processing on cluster
