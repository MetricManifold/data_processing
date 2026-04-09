# Literature Review: Gradient-Coupling Adhesion and the Rigidity Transition in Phase Field Models

> **Purpose:** Comprehensive review of cell-cell adhesion modeling in multi-cell phase field frameworks, the vertex model adhesion–rigidity connection, experimental constraints, and predicted observable signatures for each phase of our study. This document supports the manuscript and guides simulation analysis.

---

## Table of Contents

1. [Tissue Jamming and the Vertex Model](#1-tissue-jamming-and-the-vertex-model)
2. [Multi-Cell Phase Field Models](#2-multi-cell-phase-field-models)
3. [Summary: How Adhesion Has Been Modeled](#3-summary-how-adhesion-has-been-modeled)
4. [Gradient-Coupling Adhesion: Physics and Justification](#4-gradient-coupling-adhesion-physics-and-justification)
5. [Connection to the Vertex Model Shape Index](#5-connection-to-the-vertex-model-shape-index)
6. [Experimental Context](#6-experimental-context)
7. [The Geometric Frustration Picture](#7-the-geometric-frustration-picture)
8. [Predicted Observables and Expected Signals](#8-predicted-observables-and-expected-signals)
9. [Danger Signals and Pitfalls](#9-danger-signals-and-pitfalls)
10. [Extensions and Outlook](#10-extensions-and-outlook)
11. [Reviews](#11-reviews)

---

## 1. Tissue Jamming and the Vertex Model

### 1.1 Experimental evidence for the jamming transition

Dense epithelial monolayers exhibit a transition between a solid-like, non-rearranging state and a fluid-like, collectively flowing state. Key experimental milestones:

- **Angelini et al. (2011)** [PNAS 108, 4714]: Glass-like dynamics in wound-edge MDCK monolayers — caging, subdiffusive MSD, dynamical heterogeneity at high confluence. First quantitative evidence that cell monolayers jam.
- **Park et al. (2015)** [Nat. Mater. 14, 1040]: Bronchial epithelial cells from asthmatic donors are unjammed with elongated shapes ($p > 3.81$); healthy-donor cells remain jammed and compact. Established the shape index as the experimental order parameter.
- **Garcia et al. (2015)** [PNAS 112, 15314]: Active jamming in expanding MDCK monolayers driven by contact inhibition of locomotion. Demonstrates that jamming can be actively maintained, not just a passive crowding effect.
- **Malinverno et al. (2017)** [Nat. Mater. 16, 587]: Rab5-mediated endocytic reawakening fluidizes MDCK-II monolayers *without* changing cadherin expression — effective adhesion is modulated by endocytic recycling of surface receptors rather than transcriptional changes.
- **Mongera et al. (2018)** [Nature 561, 401]: Direct measurement of a fluid-to-solid jamming transition along the zebrafish tailbud axis, governed by changes in **cell-cell adhesion**. The first in vivo demonstration that adhesion, not just motility, controls tissue fluidity. Surface tension measurements correlate with the transition.
- **Atia et al. (2018)** [Nat. Phys. 14, 613]: Cell shape distributions across diverse epithelia (MDCK, HBE, MCF-10A) collapse onto a universal curve predicted by the vertex model, with the shape index $p$ as the unifying variable.
- **Grosser et al. (2021)** [Phys. Rev. X 11, 011033 — DOI: 10.1103/PhysRevX.11.011033]: Cell and nucleus shape as indicators of tissue fluidity in carcinoma — elongated nuclear shape correlates with unjamming, suggesting the shape index criterion extends to 3D.

**Synthesis:** The experimental picture converges on cell shape (quantified by $p = P/\sqrt{A}$) as the order parameter for tissue jamming, with $p \approx 3.81$ as the critical value. Both motility and adhesion can drive the transition, via distinct biological pathways (Sec. 6.3).

### 1.2 The vertex model framework

The dominant theoretical explanation comes from the **vertex model**, where cells are polygons with energy:

$$E = \sum_i \left[ K_A(A_i - A_0)^2 + K_P(P_i - P_0)^2 \right]$$

The target shape index $p_0 = P_0/\sqrt{A_0}$ serves as the control parameter:

- **Nagai & Honda (2001)** [Philos. Mag. B 81, 699]: Original vertex model formulation.
- **Farhadifar et al. (2007)** [Curr. Biol. 17, 2095]: Systematic study of epithelial packing with area and perimeter elasticity.
- **Staple et al. (2010)** [Eur. Phys. J. E 33, 117]: Mechanics and remodelling in vertex model packings.
- **Fletcher et al. (2014)** [Biophys. J. 106, 2291]: Review of vertex model variants.

**Bi et al. (2015)** [Nat. Phys. 11, 1074]: At zero temperature and motility, the vertex model exhibits a density-independent rigidity transition at $p_0^* \approx 3.81$. Below this value, the tissue has a finite shear modulus (rigid); above it, a continuous family of zero-energy configurations exists (floppy).

**Bi et al. (2016)** [Phys. Rev. X 6, 021011]: Added self-propulsion via the Self-Propelled Voronoi model. The $(p_0, v_0)$ phase diagram shows that motility can fluidize the tissue even when $p_0 < p_0^*$, establishing two independent routes to unjamming.

Extensions: active polarity (Barton et al. 2017), multicellular rosettes (Yan & Bi 2019), 3D shape criteria (Merkel & Manning 2018), cell turnover (Czajkowski et al. 2019).

### 1.3 How adhesion enters the vertex model

In the vertex model, cell-cell adhesion enters through the **line tension** $\Lambda_{ij}$ at shared edges. The effective line tension at a shared edge is $\Lambda_{ij} = \Lambda_0 - \gamma_\text{adh}$, where $\Lambda_0$ is the bare contractile line tension (from the actomyosin cortex) and $\gamma_\text{adh}$ is the adhesive contribution (from cadherins bridging adjacent cells). The perimeter energy $K_P(P_i - P_0)^2$ contains a target perimeter $P_0$ that encodes this balance: cortical contractility shrinks $P_0$ (higher line tension favors compact cells), while cadherin-mediated adhesion increases $P_0$ (lower effective tension at shared edges favors elongated cells with more contact).

Critically, $p_0 = P_0/\sqrt{A_0}$ encodes the **ratio** of adhesion to contractility, not adhesion alone. Both increasing adhesion and decreasing contractility raise $p_0$ and push the tissue toward the fluid phase. This means the vertex model cannot distinguish these two biological mechanisms — they map to the same parameter.

In our phase field model, adhesion strength $\tilde{J}$ is an independent parameter, separate from the gradient energy $\gamma$ (which plays the role of interfacial stiffness / cortical tension) and the volume penalty $\mu$. This separation allows us, in principle, to independently tune adhesion and stiffness, exploring whether the mapping $\tilde{J} \to p_\text{eff}$ holds across different values of $\gamma$.

### 1.4 Constraints of the vertex model

Vertex models enforce several assumptions that are not biological requirements:
1. Cell boundaries are **straight shared edges** (no curves).
2. Topology changes (T1 transitions) are **instantaneous** operations when an edge falls below a threshold length — there is no continuous deformation path.
3. Cells tile the plane **without gaps** (strictly confluent).
4. All cells have the **same** target area and perimeter (heterogeneity requires explicit polydispersity).

Phase field models remove all of these: boundaries are smooth and curved, rearrangements evolve continuously in time, packing fraction is a tunable parameter, and cells naturally adopt heterogeneous shapes.

---

## 2. Multi-Cell Phase Field Models

### 2.1 Nonomura (2012) — foundational multi-cell PFM

[PLoS ONE 7, e33501 — DOI: 10.1371/journal.pone.0033501, 128 citations]

The first multi-cell phase field model. Nonomura built the model around the smooth step function $h(\phi) = \phi^2(3 - 2\phi)$, interpreting its gradient $\nabla h(\phi_i)$ as the cell cortex. The interaction energy is:

$$E_\text{int} = \sum_{m \neq m'} \frac{\beta}{2} \int h(\phi_m) h(\phi_{m'}) \, dr + \sum_{m \neq m'} \frac{\gamma_N}{2} \int \nabla h(\phi_m) \cdot \nabla h(\phi_{m'}) \, dr + \sum_m \frac{c}{2} \int |\nabla h(\phi_m)|^2 \, dr$$

- First term: excluded volume repulsion (smooth-step overlap).
- Second term ($\gamma_N > 0$): **gradient-overlap adhesion** — cortex–cortex coupling, strictly interface-localized.
- Third term ($c > \gamma_N$): regularization that prevents adhesion from destabilizing interfaces by adding extra gradient stiffness.

Applied to cell sorting (differential adhesion) with 2 cell types, not to rigidity transitions. Parameters: $D_0 = 0.001$ (gradient energy), $\gamma_N = 0.003$–$0.0065$, $c = 0.01$. Without regularization, stability would require $\gamma_N < 2D_0 = 0.002$; the regularization adds gradient stiffness proportional to $c \langle h'^2 \rangle$ (where $\langle h'^2 \rangle \approx 0.77$ is the profile-averaged square of $h'$), raising the effective bound to $\gamma_N \lesssim 2(D_0 + c \cdot 0.77) \approx 0.017$.

**Relevance to our work:** Our gradient coupling $J\nabla\phi_i \cdot \nabla\phi_j$ is the simplest member of the Nonomura family, using bare gradients $\nabla\phi_i$ instead of $\nabla h(\phi_i)$ and omitting regularization. This trades Nonomura's extended stability range for a single-parameter formulation with an analytically sharp stability bound $J < 2\gamma$.

### 2.2 Palmieri, Bresler, Wirtz & Grant (2015)

[Sci. Rep. 5, 11745 — DOI: 10.1038/srep11745]

Phase field model for cell migration in monolayers using quartic repulsion $\kappa \sum_{i<j} \int \phi_i^2 \phi_j^2 \, dA$ — no adhesion term. Established the equilibration protocol ($8\tau$ at $v_A = 0$ before production runs) that we follow. Studied elastic mismatch between cell types and collective motility.

### 2.3 Löber, Ziebert & Aranson (2015)

[Sci. Rep. 5, 9172 — DOI: 10.1038/srep09172]

Extension of the Ziebert–Aranson single-cell actin-motility model to many cells. Interactions appear directly in the equation of motion (non-variational):
- Repulsion: $-\lambda \rho_i \sum_{j \neq i} \rho_j^2$ (from quartic energy).
- Adhesion: $-\kappa_L \nabla\rho_i \cdot \sum_{j \neq i} \hat{f}(\nabla\rho_j)$, where $\hat{f}$ normalizes the neighbor's gradient to a unit outward normal — advects cell $i$'s boundary toward cell $j$'s. Non-variational; no well-defined energy.

Studied collision outcomes (elastic vs. inelastic) and emergent collective migration. The adhesion parameter $\kappa_L$ controls sticking; it was not scanned systematically through any transition.

**Relevance:** This is the only other gradient-type adhesion in the literature besides Nonomura. However, it is non-variational — energy landscape analysis (quench experiments) is impossible with this formulation. Our variational form ensures well-defined energetics at $v_A = 0$.

### 2.4 Najem & Grant (2016)

[Phys. Rev. E 93, 052405 — DOI: 10.1103/PhysRevE.93.052405, 41 citations]

Introduced a range-field adhesion: each cell has an auxiliary field $C_i$ that defines its adhesive neighborhood independently of the interface width. Adhesion energy: $w\,C_i\,\phi_i^2(1-\phi_i)\sum_{j\neq i}(1-\phi_j)$, which is cortex-localized on cell $i$ and reaches into the intercellular gap via $(1-\phi_j)$, with range set by $C_i$.

Measured tissue surface tension, finding $T_\text{st}/\sigma = 2w/\sigma$ (linear in adhesion), confirming the differential adhesion hypothesis quantitatively. Tested $w/\sigma \in [0.1, 0.5]$.

**Relevance:** Validated adhesion-controlled tissue surface tension in a phase field model, but did not study jamming or rigidity. The independently controllable range is unnecessary for our study, where adhesion *strength* is the scan parameter.

### 2.5 Loewe, Chiang, Marenduzzo & Marchetti (2020)

[Phys. Rev. Lett. 125, 038003 — DOI: 10.1103/PhysRevLett.125.038003]

Multi-phase field model with quartic repulsion only (no adhesion). Studied the solid-liquid transition as a function of **deformability** $d = \varepsilon/\alpha$ and motility (Péclet number). Found continuous and first-order-like transitions depending on deformability, with an intermittent regime near the transition at low $d$.

**Relevance:** Demonstrated that multi-phase field models can produce qualitatively vertex-model-like solid-liquid transitions, establishing the framework we build on.

### 2.6 Wenzel & Voigt (2021)

[Phys. Rev. E 104, 054410 — DOI: 10.1103/PhysRevE.104.054410, 28 citations]

Systematic comparison of four multiphase field models with identical passive energies but different activity mechanisms: (1) random motility (ABP-like), (2) elongation-dependent propulsion, (3) contractile polar gel, (4) traction-based polar gel. All use signed-distance-based repulsion, no adhesion.

Key finding: the microscopic activity mechanism qualitatively changes collective behavior (nematic order, defect dynamics, vorticity). Models 3–4 (with subcellular polarization) best match MDCK data.

**Relevance:** Establishes that activity choice matters as much as interactions. Our study uses ABP-like motility (model 1, simplest), which isolates the adhesion effect.

### 2.7 Graham, Zhang & Yeomans (2024)

[Soft Matter 20, 2955 — DOI: 10.1039/d3sm01033c]

Cell sorting by differential *activity* (not adhesion) in a phase field model with quartic repulsion. Extensile/contractile dipolar forces produce sorting without thermodynamic adhesion.

**Relevance:** Shows that sorting can occur without adhesion — our study asks the complementary question: can adhesion alone produce a rigidity transition without differential activity?

### 2.8 Saito & Ishihara (2024)

[Sci. Adv. 10, eadi8433 — DOI: 10.1126/sciadv.adi8433]

Fourier-contour cell model (not phase field but closely related). Cells interact only through excluded volume. Found a fluid-to-fluid transition controlled by cell deformability: at moderate deformability, vertex-model-like polygonal fluid; at high deformability, "soft fluid" with round, overlapping cells.

**Relevance:** Shows that deformability alone can drive transitions — adds a third axis (adhesion, motility, deformability) to the phase diagram landscape.

---

## 3. Summary: How Adhesion Has Been Modeled

| Reference | Repulsion | Adhesion type | Adhesion form | Range | Scanned through transition? |
|---|---|---|---|---|---|
| Nonomura (2012) | $h(\phi_i) h(\phi_j)$ | Gradient overlap | $-\gamma_N \nabla h(\phi_i) \cdot \nabla h(\phi_j)$ | Interface-localized | No (cell sorting only) |
| Palmieri et al. (2015) | $\kappa \phi_i^2 \phi_j^2$ | None | — | — | — |
| Löber et al. (2015) | $\lambda \rho_i^2 \rho_j^2$ | Advection along normal | $-\kappa_L \nabla\rho_i \cdot \hat{f}(\nabla\rho_j)$ | Interface (non-variational) | No (collision outcomes) |
| Najem & Grant (2016) | $\kappa \phi_i^2 \phi_j^2$ | Range-field cortex | $w C_i \phi_i^2(1-\phi_i)(1-\phi_j)$ | Controlled by $C_i$ | No (surface tension) |
| Loewe et al. (2020) | $\varepsilon \phi_i^2 \phi_j^2$ | None | — | — | — |
| Wenzel & Voigt (2021) | $B(\phi_i) w(d_j)$ signed-dist | None | — | — | — |
| Graham et al. (2024) | $\varepsilon \phi_i^2 \phi_j^2$ | None | — | — | — |
| Saito & Ishihara (2024) | Excluded volume | None | — | — | — |
| **This work** | $\kappa \phi_i^2 \phi_j^2$ | **Gradient coupling** | $J \nabla\phi_i \cdot \nabla\phi_j$ | **Interface-localized** | **Yes** |

**Key observations:**
1. Most recent models include **no adhesion** — only quartic repulsion.
2. The three models with adhesion use interface-localized forms but **none scanned adhesion through a rigidity transition**.
3. Our gradient coupling $J\nabla\phi_i \cdot \nabla\phi_j$ is the simplest member of the Nonomura family: variational, single-parameter, and surface-localized.

---

## 4. Gradient-Coupling Adhesion: Physics and Justification

### 4.1 The adhesion energy

$$F_\text{adh} = J \sum_{i<j} \int \nabla\phi_i \cdot \nabla\phi_j \, dA$$

At a shared interface between cells $i$ and $j$, the fields transition from 1→0 and 0→1 simultaneously: $\nabla\phi_i \approx -\nabla\phi_j$ (anti-parallel gradients). The dot product is **negative**, so $F_\text{adh} < 0$ — the energy is lowered. In the cell interior and exterior, both gradients vanish and the term contributes nothing. The adhesion is therefore **strictly surface-localized**: it acts only where cell boundaries are in contact.

### 4.2 Variational derivative: Laplacian coupling

$$\frac{\delta F_\text{adh}}{\delta \phi_i} = -J \sum_{j \neq i} \nabla^2 \phi_j$$

Integration by parts converts the gradient coupling into a **Laplacian coupling**. Since $\nabla^2\phi_j$ is localized at cell $j$'s interface (zero in the flat interior/exterior), the adhesion force is automatically surface-localized without requiring Nonomura's smooth step function or Najem's auxiliary range field.

**Efficient implementation:** Scatter all fields into a sum field $S(\mathbf{r}) = \sum_k \phi_k(\mathbf{r})$, then compute $\nabla^2 S - \nabla^2\phi_i$ via a five-point stencil. No pairwise loops needed; $O(NL^2)$ scaling, same as the existing repulsion.

### 4.3 Stability bound: $J < 2\gamma$

The gradient coupling has a critical point beyond which shared interfaces become energetically unstable. The derivation proceeds from the total gradient-type energy at a shared interface.

Consider two cells in contact. At the shared boundary, cell 1 drops from $\phi_1 \approx 1$ to $\phi_1 \approx 0$ (inside → outside), while cell 2 rises from $\phi_2 \approx 0$ to $\phi_2 \approx 1$ (outside → inside). Their gradients are therefore anti-parallel: if $\nabla\phi_1 = \mathbf{g}$ (pointing outward from cell 1), then $\nabla\phi_2 \approx -\mathbf{g}$ (pointing inward toward cell 2) at the same spatial location.

The total gradient-type energy includes contributions from each cell's Cahn–Hilliard gradient term ($\gamma|\nabla\phi_i|^2$ per cell) and the adhesion cross-term (one pair in $\sum_{i<j}$, so coefficient $J$, not $2J$):

$$E_\text{shared} = \int \left[\gamma|\nabla\phi_1|^2 + \gamma|\nabla\phi_2|^2 + J\,\nabla\phi_1 \cdot \nabla\phi_2 \right] dx$$

Substituting $\nabla\phi_2 = -\nabla\phi_1 = -\mathbf{g}$:

$$E_\text{shared} = \int \left[\gamma|\mathbf{g}|^2 + \gamma|\mathbf{g}|^2 + J\,\mathbf{g}\cdot(-\mathbf{g}) \right] dx = (2\gamma - J) \int |\mathbf{g}|^2 dx$$

Two **free** surfaces (cells not in contact) would cost $2\gamma \int |\mathbf{g}|^2 dx$ — one $\gamma$ per cell's interface. The shared interface costs $(2\gamma - J)$ of this. The energy reduction from adhesive contact is therefore $J \int|\mathbf{g}|^2 dx$.

The same result follows from a positive-definiteness argument. Define $S = \phi_1 + \phi_2$ and use the identity $\nabla\phi_1\cdot\nabla\phi_2 = \frac{1}{2}(|\nabla S|^2 - |\nabla\phi_1|^2 - |\nabla\phi_2|^2)$. The total gradient-type energy becomes:

$$E = (\gamma - J/2)|\nabla\phi_1|^2 + (\gamma - J/2)|\nabla\phi_2|^2 + (J/2)|\nabla S|^2$$

This is positive-definite if and only if $\gamma > J/2$, i.e., $J < 2\gamma$. This confirms the stability bound without assuming anti-parallel gradients.

The natural dimensionless control parameter is:

$$\tilde{J} = \frac{J}{2\gamma}$$

which measures the **fraction of surface energy removed at shared interfaces** (since the adhesion saves $J/2\gamma$ of the single-surface gradient energy at each cell's boundary).

| $\tilde{J}$ | $J$ ($\gamma=1$) | Physical meaning |
|---|---|---|
| 0 | 0 | Pure repulsion, no adhesion |
| 0.25 | 0.5 | 25% surface energy reduction (weak adhesion) |
| 0.50 | 1.0 | 50% reduction (moderate adhesion) |
| 0.75 | 1.5 | 75% reduction (strong adhesion, still stable) |
| 1.00 | 2.0 | 100% reduction — interface has zero cost → **instability** |
| >1 | >2 | System gains energy from creating interface → cell merger |

The quartic repulsion $\kappa\phi_i^2\phi_j^2$ provides additional stabilization that shifts the effective merger point slightly above $J = 2\gamma$, but the interface is already dissolving at $\tilde{J} = 1$.

### 4.4 Comparison with Nonomura's regularized model

Nonomura's model uses $D_0 = 0.001$ (gradient energy), $\gamma_N = 0.003$–$0.0065$, and regularization $c = 0.01$. Without regularization: $\gamma_N < 2D_0 = 0.002$. The regularization adds gradient stiffness proportional to $c \langle h'^2 \rangle$, where $\langle h'^2 \rangle \approx 0.77$ is the profile-averaged square of $h'(\phi) = 6\phi(1-\phi)$, raising the effective bound to $\gamma_N \lesssim 2(D_0 + c \cdot 0.77) \approx 0.017$ — roughly an $8.7\times$ increase over the unregularized bound, at the cost of a second free parameter.

At Nonomura's strongest two-cell adhesion ($\gamma_N/D_0 = 6.5$), the surface energy reduction is ~36% — comparable to our $\tilde{J} \approx 0.36$.

Our model without regularization: more restricted range ($\tilde{J} < 1$) but single-parameter, analytically transparent, and sufficient for the biologically relevant range (up to 75% surface reduction).

### 4.5 Why this form and not something else

#### The bilinear form fails: a cautionary tale

The most natural first guess for phase field adhesion is the bilinear overlap $F_\text{adh} = -J\sum_{i<j}\int\phi_i\phi_j\,dA$, the scalar field analog of the Ising coupling $-Js_is_j$. This form was the original design for our study and was implemented and tested before being abandoned. The failure is instructive.

The variational derivative of the bilinear adhesion is $\delta F/\delta\phi_i = -J\sum_{j\neq i}\phi_j$. This expression has **no dependence on $\phi_i$**: wherever a neighbor cell has $\phi_j > 0$, the adhesion exerts a constant attractive force on $\phi_i$ regardless of whether $\phi_i$ is 0 (empty space) or 1 (cell interior). In the equation of motion $\partial_t\phi_i = -M\delta F/\delta\phi_i + \ldots$, this acts as a **source term** that nucleates field at every grid point adjacent to any other cell. In multi-cell simulations, this causes $\phi_i$ to grow from zero throughout the domain, destroying the cell structure.

Six progressively sophisticated variants were tested before reaching gradient coupling:

| Version | Form | Behavior | Root cause |
|---|---|---|---|
| v1 (bare bilinear) | $-J\phi_i\phi_j$ | Nucleation: field fills domain | $\delta F/\delta\phi_i = -J\phi_j$ has no $\phi_i$ factor |
| v2 (smooth step) | $-J\,g(\phi_i)g(\phi_j)$ | First-order: cells either repel or merge | No crossover with repulsion: same $\phi^2$ spatial profile |
| v3 (gradient coupling) | $J\nabla\phi_i\cdot\nabla\phi_j$ | Surface tension reduction, not bulk attraction | Same functional form as the final model; initially misinterpreted as "failed" because it reduces surface tension rather than creating a force from distance — which is the correct physics |
| v4 (reduced $\kappa$) | $(\kappa-J)\phi_i^2\phi_j^2$ | No attraction from distance | Same quartic profile, just weaker |
| v5 (scaled bilinear) | $-J(30/\lambda^2)\phi_i\phi_j$ | 2-cell works, multi-cell: $\phi_{\max}\to 0.83$ | Correct energy scale but nucleation persists |
| v6 (gated bilinear) | $-J(30/\lambda^2)\text{gate}(\phi_i)\phi_j$ | Delays onset, still degrades | Ad hoc gate doesn't fully prevent source behavior |

The fundamental lesson is that **self-gating** — a factor of $\phi_i$ (or its derivatives) in $\delta F/\delta\phi_i$ — is essential. Without it, any attractive term acts as a source at $\phi_i = 0$, nucleating field in empty space. The gradient coupling avoids this problem through a different mechanism. The variational derivative $\delta F/\delta\phi_i = -J\sum_{j \neq i}\nabla^2\phi_j$ involves no explicit $\phi_i$ factor, and $\nabla^2\phi_j$ is technically nonzero at cell $j$'s interface even where $\phi_i = 0$. However, $\nabla^2\phi_j$ is localized to the narrow interface region (width $\sim\lambda$), alternates in sign across the interface normal, and integrates to zero over the domain. Any small perturbation $\phi_i = \epsilon > 0$ created by this force at a point far from cell $i$'s body is immediately suppressed by the double-well potential $f'(\phi) \propto \phi(1-\phi)(1-2\phi)$, which provides a strong restoring force toward $\phi = 0$ for isolated perturbations. In contrast, the bilinear adhesion force $-J\phi_j$ is nonzero over the entire volume of cell $j$ (not just its interface), one-signed, and large enough to overwhelm the double-well restoration. The gradient coupling's surface localization and alternating-sign structure thus provide effective self-gating without requiring an explicit $\phi_i$ factor.

For a systematic analysis of why no power law $\phi_i^a\phi_j^a$ can produce a stable crossover with the quartic repulsion $\kappa\phi_i^2\phi_j^2$, see the adhesion study LOG_BOOK ("Functional Form Analysis" section).

#### Why not Najem's range-field?
The auxiliary field $C_i$ provides independently controllable adhesion range — a feature we don't need. Our study varies adhesion *strength* at fixed interface width. The extra field adds computational cost and a second parameter.

#### Why not Löber's non-variational form?
The advection-based adhesion cannot be derived from a free energy, making the quench experiment (Sec. 8.1) impossible. The quench relies on $v_A = 0$ dynamics being purely relaxational (gradient descent on $F$), which requires a well-defined energy functional.

#### Why not Nonomura with regularization?
The regularization adds a second parameter $c > J$ and complicates the stability analysis (the effective bound becomes $J < c + \gamma$ rather than $J < 2\gamma$). For establishing whether an adhesion-controlled rigidity transition exists, a single-parameter model with an analytically sharp stability bound is preferable. The regularization can always be added later to extend the accessible adhesion range.

### 4.6 Sharp-interface limit: recovery of vertex model adhesion

To verify that the gradient coupling encodes the same physics as vertex model adhesion, we compute the adhesion integral in the sharp-interface limit. Consider two cells meeting at a flat shared boundary of length $\ell_{ij}$, with $\tanh$-profile interfaces of width $\lambda$. Taking the interface normal along $\hat{x}$ and the contact along $\hat{y}$, the profiles are:

$$\phi_1(x) = \tfrac{1}{2}[1 - \tanh(x/\lambda)], \qquad \phi_2(x) = \tfrac{1}{2}[1 + \tanh(x/\lambda)]$$

The gradients along $\hat{x}$ are:

$$\nabla\phi_1 = -\frac{1}{2\lambda}\text{sech}^2(x/\lambda)\,\hat{x}, \qquad \nabla\phi_2 = +\frac{1}{2\lambda}\text{sech}^2(x/\lambda)\,\hat{x}$$

The dot product is:

$$\nabla\phi_1 \cdot \nabla\phi_2 = -\frac{1}{4\lambda^2}\text{sech}^4(x/\lambda)$$

Integrating over the interface normal direction using $\int_{-\infty}^{\infty}\text{sech}^4(u)\,du = 4/3$:

$$\int_{-\infty}^{\infty} \nabla\phi_1\cdot\nabla\phi_2\,dx = -\frac{1}{4\lambda^2}\cdot\lambda\cdot\frac{4}{3} = -\frac{1}{3\lambda}$$

Multiplying by the contact length $\ell_{ij}$ and summing over all pairs:

$$F_\text{adh} = J\sum_{i<j}\int\nabla\phi_i\cdot\nabla\phi_j\,dA \sim -\frac{J}{3\lambda}\sum_{i<j}\ell_{ij}$$

In the vertex model, adhesion contributes $-\gamma_\text{vm}\sum_{i<j}\ell_{ij}$ to the energy. The identification $\gamma_\text{vm} = J/(3\lambda)$ connects the two frameworks. The numerical prefactor $1/3$ arises from the $\tanh$ profile; for the double-well $\phi^2(1-\phi)^2$ with prefactor $30\gamma/\lambda^2$, the equilibrium profile differs slightly from $\tanh$ and the prefactor changes, but the **scaling** $F_\text{adh} \propto -(J/\lambda)\sum\ell_{ij}$ is robust.

This confirms that gradient-coupling adhesion encodes the same contact-length-dependent physics as vertex model adhesion: the energy saved by adhesion is proportional to the total shared interface length.

---

## 5. Connection to the Vertex Model Shape Index

### 5.1 The mapping $\tilde{J} \to p_\text{eff}$

In the vertex model, adhesion raises $p_0$ (through reduced line tension). In our model, adhesion ($\tilde{J}$) reduces the effective surface tension at shared interfaces, favoring elongated, high-perimeter shapes. The key question is whether this correspondence is **quantitative**: does the rigidity transition in $\tilde{J}$ coincide with $\langle p_\text{eff} \rangle \approx 3.81$?

The effective shape index $p_\text{eff} = P_i/\sqrt{A_i}$ is measured from the $\phi_i = 0.5$ contour via marching squares. The mapping $\tilde{J} \to \langle p_\text{eff} \rangle$ is a **result** of the simulation, not an assumption.

### 5.2 Why the mapping might work

1. Both models describe confluent monolayers where cells tile the plane.
2. In both, adhesion rewards larger contact area → larger perimeter → higher $p$.
3. The sharp-interface limit (Sec. 4.6) recovers the vertex model adhesion energy.
4. The geometric frustration argument (Sec. 7) applies to both models.

### 5.3 Why the mapping might fail

1. Phase field cells have **curved boundaries** — the $p_0^* = 3.81$ value is derived for polygonal tilings. For smooth contours, the critical shape index may differ.
2. The diffuse interface means cells overlap slightly. The $\phi = 0.5$ contour area may differ from the vertex-model "cell area" (Voronoi cell or polygon).
3. The quartic repulsion $\kappa\phi_i^2\phi_j^2$ creates a repulsive barrier that has no vertex model analog — it could shift the transition.
4. The interface width $\lambda = 7$ is not infinitesimal compared to the cell radius $R = 49$ (ratio $\lambda/R \approx 0.14$). Finite-interface corrections may be significant.

### 5.4 How to test the mapping

At each $\tilde{J}$ value (and each $v_A$ in Phase 2):
1. Extract $\phi_i = 0.5$ contours from VTK snapshots.
2. Compute $P_i$ and $A_i$ for each cell.
3. Compute $p_\text{eff} = P_i/\sqrt{A_i}$.
4. Plot $\langle p_\text{eff} \rangle$ vs. $\tilde{J}$.
5. Compare the transition point to $3.81$.

If $\langle p_\text{eff} \rangle$ at the Phase 0 transition ($\tilde{J} \approx 0.25$–$0.50$) is near $3.81$, the mapping holds. If it differs substantially, the phase field introduces corrections that need to be understood.

---

## 6. Experimental Context

### 6.1 Adhesion controls tissue fluidity in vivo

**Mongera et al. (2018)** [Nature 561, 401] is the single most relevant experiment. They measured tissue surface tension and yield stress along the zebrafish tailbud axis, finding a fluid-to-solid gradient as cells move from posterior (fluid) to anterior (solid). The transition correlates with increasing **N-cadherin expression** from posterior to anterior: anterior tissue has *higher* adhesion and is *solid*; posterior tissue has *lower* adhesion and is *fluid*.

**Direct relevance:** Our model scans adhesion strength $\tilde{J}$ and measures the fluid-solid boundary — directly analogous to the Mongera measurement along the anterior-posterior axis.

**Important nuance: adhesion can both fluidize and rigidify.** Mongera's result — that higher adhesion correlates with *solidity* — appears to contradict the vertex model prediction, where higher adhesion raises $p_0$ toward the *fluid* phase. The resolution involves distinguishing two roles of adhesion:

1. **Energetic/geometric role (vertex model):** Adhesion reduces the effective line tension at shared edges, favoring elongated cell shapes with more contact area. This raises $p_0$ and promotes fluidity. Our gradient coupling captures this mechanism: higher $\tilde{J}$ reduces shared interface energy, which should produce more elongated cells and higher $p_\text{eff}$.

2. **Kinetic/mechanical role (not in the vertex model):** Adhesion creates stronger bonds between cells that must be broken for rearrangements. In 3D tissues like zebrafish tailbud, cells rearrange by sliding past neighbors, and adhesion friction resists this sliding. This kinetic barrier increases with adhesion and *opposes* fluidity.

The vertex model captures only the energetic role. Our phase field model, with its overdamped gradient-descent dynamics at $v_A = 0$, also captures only the energetic role: the system always flows to the nearest energy minimum without thermal activation over barriers. The kinetic barrier effect (bond-breaking friction) would require explicit models of adhesion bond dynamics or thermal noise, which are absent in our formulation. Our Phase 0 quench therefore tests specifically whether the *energetic* effect of adhesion can destabilize confluent packings — the vertex model prediction. If higher $\tilde{J}$ promotes rearrangement in the quench (as our data shows), this confirms that the energetic effect dominates in the overdamped regime. The kinetic effect observed by Mongera et al. in 3D zebrafish tissue likely involves additional physics — cell-cell friction, bond turnover kinetics, or three-dimensional geometric constraints — beyond the scope of this model.

### 6.2 Adhesion modulation at the cell surface

**Malinverno et al. (2017)** [Nat. Mater. 16, 587]: Rab5-mediated endocytosis unjams MDCK-II monolayers by recycling E-cadherin, reducing effective adhesion *without* changing cadherin expression. This shows that effective adhesion is a dynamic quantity that can be experimentally modulated.

**Foty & Steinberg (2005)** [Dev. Biol. 278, 255 — DOI: 10.1016/j.ydbio.2004.11.012]: Quantitative validation of the differential adhesion hypothesis — tissue surface tension $\sigma$ is proportional to cadherin expression level. Our model's prediction that tissue properties change monotonically with $\tilde{J}$ should be consistent with this.

### 6.3 Two routes to unjamming: adhesion vs. motility

**Mitchel et al. (2020)** [Nat. Commun. 11, 5053 — DOI: 10.1038/s41467-020-18841-x]: In primary human bronchial epithelial cells, the unjamming transition (UJT, motility-driven) and the partial epithelial-to-mesenchymal transition (pEMT, adhesion-driven) are **biologically distinct**:
- UJT: cell-cell junctions intact, apico-basal polarity maintained, cells elongate and migrate cooperatively, no mesenchymal markers.
- pEMT: junctions disassemble, mesenchymal markers appear, adhesion is reduced.

**Relevance to our study:** Our Phase 0 (quench at $v_A = 0$) isolates the **adhesion axis** — analogous to pEMT. Our Phase 2 ($\tilde{J}$ × $v_A$ phase diagram) explores both axes simultaneously. We can determine whether the two routes produce distinguishable dynamic signatures (different $\beta$, $\chi_4$, MSD scaling) or merge into a single transition boundary.

### 6.4 Shape as the universal order parameter

**Park et al. (2015)** and **Atia et al. (2018)** established that the shape index $p$ is measurable in real tissues and correlates with jamming state regardless of cell type. Our $p_\text{eff}$ measurement is the direct phase field analog. If our model shows $p_\text{eff} \approx 3.81$ at the transition, this validates the universality hypothesis. If $p_\text{eff}$ at the transition differs, the diffuse interface may introduce corrections that need to be characterized.

### 6.5 Tissue surface tension measurements

Najem & Grant (2016) showed in their phase field model that tissue surface tension scales linearly with adhesion: $T_\text{st}/\sigma = 2w/\sigma$. The prediction from our model is that $T_\text{st} \propto (1 - \tilde{J})$ — surface tension decreases linearly with adhesion until it vanishes at $\tilde{J} = 1$. This is testable against Mongera's zebrafish data if we can establish units.

---

## 7. The Geometric Frustration Picture

**Moshe, Bowick & Marchetti (2018)** [Phys. Rev. Lett. 120, 268105 — DOI: 10.1103/PhysRevLett.120.268105] derived a continuum elasticity limit of the vertex model showing that the rigidity transition at $p_0^* \approx 3.81$ arises from **geometric frustration**: when target area and target perimeter become geometrically incompatible (no polygon can simultaneously satisfy both), the tissue acquires a finite shear modulus. Below $p_0^*$, compatible target shapes exist and the tissue has zero modes (floppy). Above $p_0^*$, the targets are frustrated and all deformations cost energy.

The value $3.81$ is the isoperimetric ratio $P/\sqrt{A}$ of a regular pentagon (5 sides: $P = 5s$, $A = \frac{s^2}{4}\sqrt{25 + 10\sqrt{5}} \approx 1.720\,s^2$, giving $p = 5/\sqrt{1.720} \approx 3.812$). In any 2D tiling, Euler's formula constrains the average coordination number to 6; disordered tilings near the transition contain a mixture of pentagons, hexagons, and heptagons. The pentagon's shape index sets the critical threshold because it represents the crossover between geometrically compatible configurations (hexagons, $p \approx 3.72$) and frustrated ones.

**Application to our model:** The gradient coupling reduces effective surface tension at shared interfaces by $(1 - \tilde{J})$. For cells in a confluent packing, lower interface energy at contacts favors more contact → more elongated shapes → higher $p$. Once $p$ exceeds the geometric frustration threshold, the tissue becomes floppy. This predicts a monotonic relationship between $\tilde{J}$ and $\langle p_\text{eff} \rangle$, with the rigidity transition occurring where $\langle p_\text{eff} \rangle$ crosses $\sim 3.81$.

**Li, Wei, Paoluzzi & Ciamarra (2021)** [Phys. Rev. E 103, 022607 — DOI: 10.1103/PhysRevE.103.022607] showed that cell softness (controlled by $p_0$) qualitatively changes the energy landscape: soft cells ($p_0 \approx 3.81$) have a fractal-like landscape with many near-zero-barrier T1 transitions, producing anomalous subdiffusion. Stiff cells ($p_0 \approx 3.0$) have a conventional glassy landscape with well-defined caging.

**Relevance:** Near our transition ($\tilde{J} \approx 0.25$–$0.50$), we may observe anomalous subdiffusion and fractal-like energy landscapes rather than a clean two-phase separation.

---

## 8. Predicted Observables and Expected Signals

### 8.1 Phase 0: Adhesion quench ($v_A = 0$)

The quench is unique to phase field models — vertex models cannot perform it. Starting from an equilibrated $J = 0$ packing, adhesion is instantaneously turned on and the system relaxes by pure energy minimization.

**Primary observable: total centroid displacement** $\sum_i |\Delta\mathbf{r}_i|$ vs. $\tilde{J}$

**Confirmed signals (gradient-coupling Phase 0, rorqual run_01, 2τ):**

| $\tilde{J}$ | $J/\kappa$ | Mean displacement / $R$ | Regime |
|---|---|---|---|
| 0 | 0 | 0.025 | Control (confirms equilibration) |
| 0.25 | 0.05 | 0.058 | Interface adjustment only |
| 0.50 | 0.10 | 0.086 | Interface adjustment only |
| 0.75 | 0.15 | 0.132 | Interface adjustment, not fully relaxed |

All displacements are sub-cell. No neighbor exchanges (T1-like events) at any $\tilde{J}$.
The quench's scientific value is in static equilibrium measurements ($p_\text{eff}$, contact angle, energy decomposition, relaxation timescale), not displacement.

**Additional Phase 0 observables to extract:**

1. **Energy vs. time:** Below transition — smooth monotonic decay (gradient relaxation). Above transition — stepwise/staircase decay corresponding to discrete rearrangement events resolved continuously.
2. **Neighbor topology changes:** Count new contacts formed and old contacts broken. Plot vs. $\tilde{J}$. Below transition: 0 topology changes. Above: increasing with $\tilde{J}$.
3. **Relaxation timescale:** Time to reach 90% of final displacement. Should diverge (critical slowing down) near transition.
4. **Energy decomposition:** Track gradient, bulk, repulsive, and adhesive energy components separately. The adhesion component should decrease (more favorable contacts); the gradient component may increase (cells deform to create more contact area).
5. **VTK snapshots:** Before/after visual comparison at each $\tilde{J}$. Look for cell elongation, neighbor rearrangements, contact area increase.

**What to watch for:**
- If $\tilde{J} = 0$ control shows displacement > $0.05R$, the starting state is not well-equilibrated. Discard and re-equilibrate.
- If displacement increases gradually rather than sharply, the transition may be continuous (crossover rather than sharp). This is physically interesting but requires finer $\tilde{J}$ sampling to characterize.
- If cells merge at $\tilde{J} = 0.75$ or below, the effective stability bound may be lower than the analytical $\tilde{J} = 1$. Check mass conservation.

### 8.2 Phase 1: Motility probe ($v_A = 0.002$, sweep $\tilde{J}$)

Small motility provides thermal-like fluctuations that let the system explore configuration space. By sweeping $\tilde{J}$ at fixed $v_A$, we locate the fluid-solid boundary at nonzero motility.

**Key observables:**

1. **MSD vs. time:** Jammed → plateau (caging). Fluid → linear growth ($\sim 4D_\text{eff}\,t$). Near transition → two-step: ballistic → plateau → (eventual) escape.
2. **Self-overlap $Q(t)$:** Fraction of cells that haven't moved beyond a cage radius. Fit to stretched exponential $Q(t) = \exp[-(t/\tau_\alpha)^\beta]$. Jammed → $\tau_\alpha$ diverges, $\beta$ decreases. Fluid → fast decay, $\beta \to 1$.
3. **Non-Gaussian parameter $\alpha_2(t)$:** Peaks at $t^* \sim 2\tau$ near the transition, indicating a subpopulation of fast-moving cells coexisting with caged cells. Far from transition (either side), $\alpha_2$ is small.
4. **Four-point susceptibility $\chi_4(t)$:** Peak height measures the number of cooperatively rearranging cells. Should be maximal near the transition. Well into the fluid or jammed phase, $\chi_4$ drops.
5. **Effective shape index $p_\text{eff}$:** Central observable. Plot $\langle p_\text{eff} \rangle$ vs. $\tilde{J}$ and check whether the MSD-identified transition coincides with $\langle p_\text{eff} \rangle \approx 3.81$.

**Expected behavior:**
- At $\tilde{J} = 0$ and $v_A = 0.002$, the system should be jammed (strong caging, plateau MSD). We know from Griffiths study data that $v_A = 0.008$ is near the clean transition at $J = 0$, so $v_A = 0.002$ is well below.
- As $\tilde{J}$ increases, the caging weakens (plateau height increases) and eventually a diffusive regime emerges.
- The motility probe transition at $\tilde{J}^*(v_A = 0.002)$ should occur at **lower $\tilde{J}$** than the quench transition at $\tilde{J}^*(v_A = 0)$, since motility provides additional kinetic energy for barrier crossing.

### 8.3 Phase 2: Phase diagram ($\tilde{J}$ × $v_A$)

The full phase diagram maps the fluid-solid boundary in two-parameter space for comparison with the vertex model $(p_0, v_0)$ diagram of Bi et al. (2016).

**Expected phase diagram topology:**
- At $v_A = 0$: rigidity transition at $\tilde{J}^* \approx 0.25$–$0.50$ (from Phase 0).
- At $\tilde{J} = 0$: motility-driven transition at $v_A^* \approx 0.008$–$0.010$ (from Griffiths study data at $J = 0$).
- Between: a phase boundary connecting these two points, bending such that higher adhesion requires less motility to unjam (and vice versa).
- The vertex model $(p_0, v_0)$ phase boundary has a specific functional form; our boundary should be qualitatively similar if the mapping holds.

**Key observables:**
- MSD at $t = 5\tau$ as a heatmap in $(\tilde{J}, v_A)$ space.
- Phase boundary from MSD threshold or $Q(t)$ decay.
- $\langle p_\text{eff} \rangle$ contours overlaid on the phase diagram.
- System-size comparison (288 vs. 1152 cells) at selected points.

### 8.4 What we expect from the complete study

1. **Sharp quench transition** at $\tilde{J} \approx 0.25$–$0.50$ confirmed. ✓ (Sec. 8.1)
2. **Monotonic $p_\text{eff}$ increase** with $\tilde{J}$: higher adhesion → more elongated cells → higher shape index.
3. **Phase boundary in $(\tilde{J}, v_A)$** qualitatively matching the vertex model $(p_0, v_0)$ diagram.
4. **Quantitative shape index test:** If the phase boundary in our model corresponds to $\langle p_\text{eff} \rangle \approx 3.81$, this validates the vertex model criterion in a continuum framework.
5. **Continuous rearrangement paths** visible in energy time series during quench — something vertex models cannot show.
6. **Energy decomposition during quench:** Below transition, adhesion energy decreases but rearrangement doesn't occur (local minimum). Above transition, adhesion drives rearrangements that lower total energy.

---

## 9. Danger Signals and Pitfalls

### 9.1 Mass conservation

The gradient-coupling adhesion contributes $-J\nabla^2\phi_j$ to $\delta F/\delta\phi_i$. Crucially, the Laplacian integrates to zero over the periodic domain: $\int\nabla^2\phi_j\,dA = 0$ by the divergence theorem with periodic boundary conditions. The adhesion term therefore **cannot cause mass drift at the continuum level**.

Mass changes can only come from the bulk terms in the variational derivative: the double-well potential $\phi(1-\phi)(1-2\phi)$ (whose integral is not generally zero) and the volume constraint. Any observed mass drift is therefore a **numerical discretization artifact** of the explicit Euler scheme, not a consequence of the adhesion physics. All mass-conservation checks should focus on whether the forward-Euler time step $\Delta t = 0.02$ is sufficiently small for the total dynamics, not on the adhesion term specifically.

**Monitor:** total field mass $\sum_{x,y}\phi_i$ for each cell at every VTK frame. Acceptable drift: < 1%. Two-cell validation confirmed < 1.2% drift across $\tilde{J} \in [0, 0.75]$.

### 9.2 Cell merger

At $\tilde{J} \geq 1$, cells merge (confirmed by two-cell validation). But in a many-cell system, local geometry may produce effective merger at lower $\tilde{J}$, especially at triple junctions where three interfaces meet. **Monitor:** check for any cell with mass > $1.5 V_0$ or cells with $\phi_i > 1$ anywhere.

### 9.3 Interface dissolution vs. rearrangement

Large displacements during quench could indicate either (a) genuine cell rearrangements (good) or (b) interface dissolution and spreading (bad). **Distinguish:** check that the $\phi_i = 0.5$ contour remains well-defined (aspect ratio < 3 for circular cells). If cells become diffuse blobs without clear boundaries, $\tilde{J}$ is too high.

### 9.4 Finite-size effects

288 cells on $L = 1562$ gives ~17 cells per linear dimension. Near the transition, cooperative rearrangement regions may span a significant fraction of the system. **Test:** compare 288-cell and 1152-cell results at the same $\tilde{J}$. If the transition shifts, finite-size effects are significant.

### 9.5 Residual equilibration stress

If the starting state is not fully equilibrated, residual stress will produce displacement even at $J = 0$. **Test:** the $J = 0$ control displacement must be < $0.05R$. The initial data shows $0.017R$ — good.

### 9.6 Time step instability

The Laplacian coupling $-J\nabla^2(S - \phi_i)$ adds second-derivative terms to the dynamics. If $J$ is too large relative to $\Delta t$, the explicit Euler scheme becomes unstable. **Signs:** oscillating fields, diverging energy, NaN values. At $\Delta t = 0.02$ and $J = 1.5$ ($\tilde{J} = 0.75$), we have not seen instability; this should be monitored if extending to higher $J$.

### 9.7 Confusing $J/\kappa$ and $\tilde{J}$

The parameter scan uses $J/\kappa$ for historical reasons (it's the CLI flag), but the physics says $\tilde{J} = J/(2\gamma)$. For our parameters ($\gamma = 1$, $\kappa = 10$): $\tilde{J} = 5 \cdot J/\kappa$. The quench table has both columns to avoid confusion.

### 9.8 Contact angle and the Young–Dupré relation

The gradient coupling predicts a specific equilibrium contact angle between two cells. At a triple junction where cells $i$, $j$, and the exterior meet, three surface tensions must balance: the free surface tension $\sigma_f \propto \gamma$ (one cell's interface facing exterior) and the shared interface tension $\sigma_s = (2\gamma - J) \cdot I / \ell_s$ (both cells' gradient energy minus the adhesion, per unit contact length). Since two free surfaces cost $2\gamma I$ and one shared interface costs $(2\gamma - J)I$, the effective tensions satisfy $\sigma_s / (2\sigma_f) = (2\gamma - J)/(2\gamma) = 1 - \tilde{J}$.

At the triple junction, force balance along the shared interface direction gives:

$$\cos\alpha = \frac{\sigma_s}{2\sigma_f} = 1 - \tilde{J}$$

where $\alpha$ is the angle between each cell's free surface and the shared interface. For $\tilde{J} = 0$: $\alpha = 0$ (free surfaces nearly parallel to the shared interface, tangential contact). As $\tilde{J} \to 1$: $\alpha \to 90°$ (free surfaces perpendicular to shared interface, cells flatten maximally against each other). The full contact angle through the exterior is $\theta = \pi - 2\alpha$.

The two-cell validation data (decreasing $d_\text{eq}$ with increasing $\tilde{J}$) is an indirect measurement of this contact angle: stronger adhesion pulls cells closer, consistent with the increasing $\alpha$. Extracting $\alpha$ from the $\phi = 0.5$ contours at the edge of the contact zone would provide a direct test of this force-balance prediction.

---

## 10. Extensions and Outlook

### 10.1 Three-dimensional generalization

The gradient coupling generalizes directly to 3D:

$$F_\text{adh} = J\sum_{i<j}\int\nabla\phi_i\cdot\nabla\phi_j\,dV$$

The variational derivative $-J\sum_{j\neq i}\nabla^2\phi_j$ is unchanged in form. The stability bound $J < 2\gamma$ holds identically (the positive-definiteness argument is dimension-independent). The sum-field implementation ($\nabla^2 S - \nabla^2\phi_i$ via 7-point stencil in 3D) is straightforward. Three-dimensional simulations would enable comparison with Merkel & Manning's (2018) 3D shape criteria and with Mongera's (2018) zebrafish tailbud measurements (which are inherently 3D).

### 10.2 Differential adhesion and cell sorting

Nonomura's (2012) original application was cell sorting driven by differential adhesion between cell types. Our framework naturally supports this by promoting the adhesion coefficient to a type-dependent matrix $J_{\alpha\beta}$, where $\alpha, \beta$ are cell types:

$$F_\text{adh} = \sum_{i<j} J_{\alpha_i \alpha_j}\int\nabla\phi_i\cdot\nabla\phi_j\,dA$$

With $J_{\text{same}} > J_{\text{different}}$, same-type cells would preferentially adhere, driving phase separation. This connects to the differential adhesion hypothesis (Steinberg; Foty & Steinberg 2005) and provides a variational framework for studying sorting dynamics that was Nonomura's original goal.

### 10.3 The cellular Potts model as precursor

The cellular Potts model (CPM), developed by Glazier and Graner (1992) [Phys. Rev. Lett. 69, 2013 — DOI: 10.1103/PhysRevLett.69.2013] from the Potts lattice model, represents cells as connected domains of lattice sites and includes explicit adhesion energy $J_{\tau\tau'}$ between cell types. The CPM predated all continuous phase field approaches and has been widely used for adhesion-driven sorting and tissue mechanics. Chiang & Marenduzzo (2016) [EPL 116, 28009 — DOI: 10.1209/0295-5075/116/28009] studied glass transitions in the CPM, finding glassy arrest controlled by the ratio of cell-cell adhesion to cell-medium adhesion.

The CPM is a natural precursor to our work: it includes explicit adhesion, captures deformable cell shapes, and exhibits jamming. However, its Monte Carlo (Metropolis) dynamics lack a physical time scale, making quantitative comparison with experiments difficult and ruling out quench-type experiments. The phase field formulation provides deterministic, physically motivated dynamics (gradient descent on $F$) that enables the quench protocol and energy landscape analysis central to our study.

---

## 11. Reviews

- **Camley & Rappel (2017)** [J. Phys. D 50, 113002]: Physical models of collective cell motility — covers particle, continuum, vertex, Potts, and phase field approaches.
- **Alert & Trepat (2020)** [Annu. Rev. Condens. Matter Phys. 11, 77]: Physical models of collective cell migration, connecting cell mechanics to tissue flows.
- **Alt, Ganguly & Salbreux (2017)** [Phil. Trans. R. Soc. B 372, 20150520]: Vertex model review — derivation from continuum mechanics, line tension, adhesion.
- **Lenne & Trivedi (2022)** [Nat. Commun. 13, 949]: Perspective on phase transitions in tissues — thermodynamic phase separation, rigidity transitions, percolation.
- **Berthier, Flenner & Szamel (2019)** [J. Chem. Phys. 150, 200901]: Glassy dynamics in dense active particle systems — foundational active glass review.
- **Janssen (2019)** [J. Phys. Condens. Matter 31, 503002]: Active glasses review — mode-coupling theory, effective temperatures, caging.

---

*Last updated: 2026-02-18*
