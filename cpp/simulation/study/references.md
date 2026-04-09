# Research Paper References

This document contains references relevant to the jamming study and vertex model simulations, including detailed summaries and a comprehensive theoretical introduction.

---

## Table of Contents

1. [Core Papers](#core-papers)
2. [Paper Summaries](#paper-summaries)
3. [Theoretical Background](#theoretical-background)
   - [The Vertex Model](#the-vertex-model)
   - [Glass and Jamming Transitions](#glass-and-jamming-transitions)
   - [The Shape Index and Rigidity](#the-shape-index-and-rigidity)
   - [T1 Transitions and Topological Rearrangements](#t1-transitions-and-topological-rearrangements)
   - [Active Matter and Self-Propulsion](#active-matter-and-self-propulsion)
   - [Cell Division and Death](#cell-division-and-death)
   - [Collective Migration and Flocking](#collective-migration-and-flocking)
4. [Key Concepts Reference](#key-concepts-reference)

---

## Core Papers

### 1. Density-Independent Glass Transition
**"A density-independent glass transition in biological tissues"**  
Dapeng Bi, J. H. Lopez, J. M. Schwarz, M. Lisa Manning  
*Nature Physics* 11, 1074-1079 (2015)  
[DOI: 10.1038/nphys3471](https://doi.org/10.1038/nphys3471) | [arXiv: 1409.0593](https://arxiv.org/abs/1409.0593)

---

### 2. Motility-Driven Glass and Jamming Transitions
**"Motility-driven glass and jamming transitions in biological tissues"**  
Dapeng Bi, Xingbo Yang, M. Cristina Marchetti, M. Lisa Manning  
*Physical Review X* 6, 021011 (2016)  
[DOI: 10.1103/PhysRevX.6.021011](https://doi.org/10.1103/PhysRevX.6.021011) | [arXiv: 1509.06578](https://arxiv.org/abs/1509.06578)

---

### 3. Active Vertex Model
**"Active Vertex Model for cell-resolution description of epithelial tissue mechanics"**  
Daniel L. Barton, Silke Henkes, Cornelis J. Weijer, Rastko Sknepnek  
*PLOS Computational Biology* 13(6), e1005569 (2017)  
[DOI: 10.1371/journal.pcbi.1005569](https://doi.org/10.1371/journal.pcbi.1005569)

---

### 4. Glassy Dynamics with Mitosis and Apoptosis
**"Glassy dynamics in models of confluent tissue with mitosis and apoptosis"**  
Matthew Czajkowski, Daniel M. Sussman, M. Cristina Marchetti, M. Lisa Manning  
*Soft Matter* 15, 9133-9149 (2019)  
[DOI: 10.1039/c9sm00916g](https://doi.org/10.1039/c9sm00916g)

---

### 5. Flocking Transitions in Confluent Tissues
**"Flocking transitions in confluent tissues"**  
Fabio Giavazzi, Matteo Paoluzzi, Marta Macchi, Dapeng Bi, Giorgio Scita, M. Lisa Manning, Roberto Cerbino, M. Cristina Marchetti  
*Soft Matter* 14, 3471-3477 (2018)  
[DOI: 10.1039/c8sm00126j](https://doi.org/10.1039/c8sm00126j)

---

### 6. T1 Process in Glass-Forming Liquids
**"T1 process and dynamics in glass-forming hard-sphere liquids"**  
Yuxing Zhou, Scott T. Milner  
*Soft Matter* 11, 2700-2705 (2015)  
[DOI: 10.1039/c4sm02459a](https://doi.org/10.1039/c4sm02459a)

---

### 7. Active T1 Transitions
**"Active T1 transitions in cellular networks"**  
Charlie Duclut, Joris Paijmans, Mandar M. Inamdar, Carl D. Modes, Frank Jülicher  
*The European Physical Journal E* 45, 33 (2022)  
[DOI: 10.1140/epje/s10189-022-00175-5](https://doi.org/10.1140/epje/s10189-022-00175-5)

---

### 8. Cell Mechanics and Epithelial Packing
**"The Influence of Cell Mechanics, Cell-Cell Interactions, and Proliferation on Epithelial Packing"**  
Reza Farhadifar, Jens-Christian Röper, Benoit Aigouy, Suzanne Eaton, Frank Jülicher  
*Current Biology* 17(24), 2095-2104 (2007)  
[DOI: 10.1016/j.cub.2007.11.049](https://doi.org/10.1016/j.cub.2007.11.049)

---

## Paper Summaries

### 1. Bi et al. 2015 — Density-Independent Glass Transition

**Key Finding:** Discovery of a fundamentally new type of rigidity transition in confluent tissues that occurs at *constant density*.

**Background:** In traditional particulate systems (colloids, granular materials, foams), glass and jamming transitions are driven by increasing density—pack particles tighter, and they eventually can't move. However, epithelial tissues are *confluent*: cells completely tile the plane with no gaps, so the "density" is always 100%. This raises a puzzle: how can tissues undergo solid-to-liquid transitions if density can't change?

**Main Results:**
- Introduced the **shape index** $p_0 = P_0/\sqrt{A_0}$ as the control parameter for the transition
- Found a critical value $p_0^* \approx 3.81$ (corresponding to a regular pentagon)
- For $p_0 < 3.81$: tissue is rigid/solid-like (cells cannot exchange neighbors)
- For $p_0 > 3.81$: tissue is fluid-like (cells can freely rearrange)
- The transition is controlled by single-cell properties (adhesion, cortical tension) rather than packing

**Biological Implications:** This provides a mechanism for epithelial-to-mesenchymal transitions (EMT) in development and cancer, where cells can fluidize without changing density by modifying their cortical tension or adhesion molecules.

---

### 2. Bi et al. 2016 — Motility-Driven Transitions (PRX)

**Key Finding:** Cell motility introduces a second axis to the jamming phase diagram, enabling motility-driven unjamming even in tissues that would be solid based on shape alone.

**The Model:** Introduced the **Self-Propelled Voronoi (SPV) model**, where:
- Cells are represented by Voronoi tessellation seeds
- Each cell has a self-propulsion force $f_a$ with persistence time $\tau$
- Cell shapes are computed from the Voronoi tessellation
- Forces derive from the vertex model energy functional

**Main Results:**
- Constructed a **3D phase diagram** with axes: shape index $p_0$, motility $v_0 = f_a/\gamma$, and persistence time $\tau$
- High motility can fluidize tissues even when $p_0 < 3.81$
- Identified an experimentally accessible **structural order parameter** that predicts the glass transition
- Connected tissue behavior to Soft Glassy Rheology (SGR) theory
- Found that in the limit of small persistence times, SGR captures the transition, but fails for large persistence

**Impact:** This paper (1000+ citations) established the quantitative framework for understanding how active cell motion couples to the geometry-controlled glass transition.

---

### 3. Barton et al. 2017 — Active Vertex Model

**Key Finding:** A comprehensive computational framework that combines the vertex model with active matter dynamics, enabling simulation of tens of thousands of cells with cell-resolution detail.

**Technical Innovations:**
- Uses Delaunay-Voronoi duality for efficient computation
- T1 transitions emerge naturally from equiangulation without explicit mesh manipulation
- Handles open boundaries (unlike the periodic SPV model)
- Includes cell growth, division, and death
- Multiple alignment mechanisms (velocity, shape, neighbor)

**The Energy Functional:**
$$E_{VM} = \sum_i \frac{K_i}{2}(A_i - A_i^0)^2 + \sum_i \frac{\Gamma_i}{2}P_i^2 + \sum_{\langle\mu,\nu\rangle} 2\Lambda_{\mu\nu}l_{\mu\nu}$$

**Equations of Motion:**
$$\gamma \frac{d\mathbf{r}_i}{dt} = \mathbf{F}_i + f_a \mathbf{n}_i + \boldsymbol{\nu}_i$$
where $\mathbf{F}_i$ is the vertex model force, $f_a\mathbf{n}_i$ is the active self-propulsion, and $\boldsymbol{\nu}_i$ is stochastic noise.

**Key Results:**
- Reproduced the solid-fluid transition at $p_0 \approx 3.81$
- Demonstrated fingering instabilities at tissue boundaries
- Showed cell sorting based on differential adhesion
- Found collective oscillation modes with velocity alignment

**Software:** Implemented in the open-source SAMoS package.

---

### 4. Czajkowski et al. 2019 — Mitosis and Apoptosis

**Key Finding:** Cell division and death can fluidize a tissue, but a glass-like regime with caging behavior persists when cell cycling rates are sufficiently low.

**The Question:** Previous work suggested any finite rate of cell division/death would fluidize tissue. But experiments show glassy dynamics *despite* continued cell cycling. How can this be reconciled?

**Model Setup:**
- Active vertex model with cell division (when area exceeds threshold) and death (after fixed lifetime)
- Division creates two daughter cells along the polarity direction
- Death removes cells instantaneously with tessellation recomputation

**Main Results:**
- Proposed an **additive ansatz**: total relaxation rate = intrinsic relaxation rate + cell cycling rate
- Glass-like behavior (subdiffusive motion, caging) occurs when cell cycling rate << intrinsic relaxation rate
- The mean squared displacement shows three regimes:
  1. Short times: ballistic motion
  2. Intermediate: subdiffusive caging (in glassy regime)
  3. Long times: diffusive motion driven by cell cycling

**Implication:** Tissues can maintain glassy dynamics if cell cycling is slow compared to structural relaxation timescales.

---

### 5. Giavazzi et al. 2018 — Flocking Transitions

**Key Finding:** Orientational alignment interactions promote both collective motion (flocking) and solidification in confluent tissues.

**Motivation:** Real cells exhibit coordinated motion patterns—how does collective alignment emerge in a confluent tissue where cells are constrained by neighbors?

**The Model:** SPV model + alignment interactions where cell polarity aligns with local velocity:
$$\tau_n \frac{d\mathbf{n}_i}{dt} = -J(\mathbf{n}_i - \hat{\mathbf{v}}_i)$$

**Key Results:**
- Identified three phases: solid, liquid (disordered), and flocking (liquid with collective motion)
- Alignment promotes solidification (counterintuitive!)
- Introduced an **effective temperature** framework for this far-from-equilibrium system
- Predicted structural signatures: hexatic order increases with alignment

**Experimental Relevance:** Accounts for collective migration patterns observed in MDCK epithelial monolayers and during wound healing.

---

### 6. Zhou & Milner 2015 — T1 Process in Hard Spheres

**Key Finding:** Introduced a geometric criterion for identifying "T1-active" particles—those that can undergo neighbor exchanges—and showed their percolation threshold corresponds to the glass transition.

**Conceptual Bridge:** T1 transitions (neighbor exchanges) are central to vertex models of tissues. This paper studies an analogous concept in hard-sphere liquids, providing insights into how local rearrangements control the glass transition.

**Main Results:**
- Defined "T1-active" particles as those that can gain/lose a Voronoi neighbor within their free volume
- Fraction of T1-active particles vanishes at random close packing (φ ≈ 0.64)
- Percolation threshold of T1-*inactive* particles matches the glass transition (φ_g ≈ 0.585)
- Provides a purely geometric/structural predictor of dynamic arrest

**Significance:** Suggests that loss of ability to perform neighbor exchanges (T1s) is fundamental to the glass transition, connecting tissue physics to traditional glass physics.

---

### 7. Duclut et al. 2022 — Active T1 Transitions

**Key Finding:** T1 rearrangements driven by active anisotropic stresses produce different patterns depending on whether the stress acts on cell bonds vs. cell bodies.

**Model:** Anisotropic vertex model with two types of active stress:
1. **Anisotropic bond tension**: stress applied along cell-cell junctions
2. **Anisotropic cell stress**: stress applied to the cell body (like myosin contractility)

**Main Results:**
- Different active stress mechanisms produce distinguishable T1 patterns
- Bond tension: T1s align perpendicular to the stress axis
- Cell stress: T1s align with the stress axis
- Provides experimental predictions for distinguishing stress mechanisms in real tissues

**Biological Context:** During convergent extension in embryonic development, coordinated T1 transitions reshape tissues. Understanding what drives T1s could reveal underlying mechanisms.

---

### 8. Farhadifar et al. 2007 — Foundational Vertex Model

**Key Finding:** Established the standard vertex model formulation for epithelial mechanics and demonstrated it captures key features of cell packing in the *Drosophila* wing.

**The Energy Functional:**
$$E = \sum_\alpha \left[ K_\alpha(A_\alpha - A_\alpha^0)^2 + \Gamma_\alpha P_\alpha^2 \right] + \sum_{\langle ij \rangle} \Lambda_{ij} l_{ij}$$

**Physical Interpretation:**
- $K(A - A^0)^2$: area elasticity (cell volume conservation, 3D effects)
- $\Gamma P^2$: perimeter contractility (actomyosin cortex)
- $\Lambda l$: line tension (adhesion molecules at junctions)

**Key Results:**
- Identified four distinct ground states depending on parameters
- Explained polygon distributions (predominantly hexagons with variance)
- Predicted that cell division drives tissue toward specific packing states
- Showed mechanical properties can be inferred from static images

**Legacy:** This paper launched the modern era of vertex model simulations in tissue mechanics.

---

## Theoretical Background

### The Vertex Model

#### Historical Origins

The vertex model originated in the physics of foams in the 1970s (Weaire, Hutzler) and was first applied to biological tissues by Honda and Eguchi in 1980. The key insight is that confluent epithelia—where cells completely tile a surface with no gaps—can be represented as polygonal tessellations analogous to soap foam.

#### Mathematical Formulation

A tissue is represented as a planar mesh where:
- **Polygons** = cells
- **Edges** = cell-cell junctions  
- **Vertices** = points where three (or more) cells meet

The mechanical state is determined by an energy functional that penalizes deviations from preferred geometric properties:

$$E_{VM} = \sum_{i=1}^{N} \left[ \frac{K_i}{2}(A_i - A_i^0)^2 + \frac{\Gamma_i}{2}(P_i - P_i^0)^2 \right]$$

where:
- $A_i$ = actual area of cell $i$
- $A_i^0$ = preferred (target) area
- $P_i$ = actual perimeter
- $P_i^0$ = preferred perimeter
- $K_i$ = area modulus (stiffness against area changes)
- $\Gamma_i$ = perimeter modulus (contractility)

**Physical Origins of Terms:**

| Term | Biophysical Origin |
|------|-------------------|
| Area elasticity $(A - A^0)^2$ | 3D cell incompressibility projected to 2D; bulk cytoplasmic pressure |
| Perimeter contractility $\Gamma P^2$ | Actomyosin cortex generates cortical tension |
| Line tension $\Lambda l$ | Competition between adhesion (cadherins, negative contribution) and cortical tension at junctions |

#### Alternative Formulations

The perimeter term can also be written with line tension explicitly:
$$E = \sum_i \frac{K_i}{2}(A_i - A_i^0)^2 + \sum_i \frac{\Gamma_i}{2}P_i^2 + \sum_{\langle\mu\nu\rangle} \Lambda_{\mu\nu}l_{\mu\nu}$$

When all junctions have equal tension $\Lambda$, this reduces to the target-perimeter form with $P^0 = -\Lambda/\Gamma$.

#### Dynamics

In the overdamped limit (appropriate for cellular scales where inertia is negligible), vertex positions evolve according to:

$$\gamma \frac{d\mathbf{r}_\mu}{dt} = -\nabla_{\mathbf{r}_\mu} E_{VM}$$

where $\gamma$ is a friction coefficient and $\mathbf{r}_\mu$ is the position of vertex $\mu$.

---

### Glass and Jamming Transitions

#### Traditional Glass Transition

In particulate systems (colloids, granular matter, molecular glasses), the glass transition occurs when:
1. **Density increases** beyond a threshold
2. Particles become "caged" by neighbors
3. Relaxation time diverges (structural relaxation ceases)
4. The system falls out of equilibrium

Key signatures:
- Mean squared displacement shows subdiffusive plateau at intermediate times
- Dynamic heterogeneity: spatially correlated regions of fast/slow motion
- Two-step relaxation in correlation functions
- Diverging viscosity/relaxation time

#### The Confluence Puzzle

In confluent tissues, cells completely cover the available space—there are no gaps. This means:
- Packing fraction $\phi = 1$ always
- Traditional density-driven jamming cannot explain solid-liquid transitions
- Yet experiments clearly show tissues can be solid-like or liquid-like

**Resolution:** The relevant control parameter is not density but **cell shape**.

#### Jamming in Vertex Models

The jamming transition in vertex models is controlled by the **shape index**:
$$p_0 = \frac{P_0}{\sqrt{A_0}}$$

This dimensionless parameter characterizes the preferred aspect ratio of cells. Different shapes have different values:
- Regular hexagon: $p_0 = 3.722$
- Regular pentagon: $p_0 = 3.812$
- Square: $p_0 = 4.0$
- Equilateral triangle: $p_0 = 4.559$

**Critical Behavior:**
- $p_0 < p_0^* \approx 3.81$: Solid/jammed state
  - Energy barriers to T1 transitions
  - Cells cannot exchange neighbors
  - Subdiffusive dynamics
  
- $p_0 > p_0^*$: Fluid/unjammed state
  - No energy barriers to T1s
  - Free neighbor exchange
  - Diffusive dynamics

The remarkable aspect is that the critical shape index corresponds almost exactly to a regular pentagon ($p_0 = 3.812$), suggesting deep geometric constraints on the transition.

---

### The Shape Index and Rigidity

#### Physical Meaning

The shape index $p_0 = P_0/\sqrt{A_0}$ sets the target shape for each cell. Its value reflects the balance between:
- **Line tension/adhesion** (favors shorter perimeters, lower $p_0$)
- **Cortical contractility** (favors larger perimeters, higher $p_0$)

Higher $p_0$ → cells prefer more elongated shapes → easier to rearrange  
Lower $p_0$ → cells prefer more isotropic shapes → harder to rearrange

#### Connection to Biological Regulation

$p_0$ can be modulated by:
- **E-cadherin expression**: more adhesion → lower effective line tension → lower $p_0$
- **Myosin activity**: more contractility → higher cortical tension → higher $p_0$
- **Rac/Rho signaling**: controls cytoskeletal organization and contractility

This provides a direct connection between molecular biology and tissue-scale mechanics.

#### Experimental Measurement

From static images, one can measure the **actual** shape index:
$$p = \frac{P}{\sqrt{A}}$$

In equilibrium, $\langle p \rangle = p_0$. Experiments on asthmatic vs. healthy airway epithelia (Park et al. 2015) showed:
- Healthy tissue: $\langle p \rangle < 3.81$ (solid-like)
- Asthmatic tissue: $\langle p \rangle > 3.81$ (fluid-like, unjammed)

#### The Isoperimetric Inequality

The shape index is bounded below by the isoperimetric inequality. For any shape:
$$p = \frac{P}{\sqrt{A}} \geq 2\sqrt{\pi} \approx 3.545$$

with equality only for a circle. This sets a fundamental lower bound on how compact cells can be.

---

### T1 Transitions and Topological Rearrangements

#### Definition

A **T1 transition** (or T1 process) is a local topological change where:
1. A cell-cell junction shrinks to zero length
2. Four cells meet at a point (a 4-fold vertex)
3. A new junction forms perpendicular to the original
4. The four cells have exchanged neighbors

```
Before:        At T1:        After:
  1--2          1  2          1  2
  |  |           \/           |  |
  |  |           /\           |  |
  3--4          3  4          3--4
  
1-2 neighbors  4-fold vertex  1-3 neighbors
3-4 neighbors               2-4 neighbors
```

#### Energy Barriers

In the vertex model, T1 transitions may or may not require crossing an energy barrier:

**Solid phase ($p_0 < 3.81$):**
- T1s require crossing energy barriers
- Barrier height scales as the distance from the transition: $\Delta E \propto (p_0^* - p_0)$
- T1s occur only through thermal/active fluctuations (if barriers can be overcome)

**Fluid phase ($p_0 > 3.81$):**
- T1s are energetically favorable or barrierless
- Cells freely rearrange
- No long-term memory of neighbors

#### T1 Rate and Dynamics

The rate of T1 transitions is a key observable:
- High T1 rate → fluid-like, rapid rearrangements
- Low T1 rate → solid-like, slow dynamics

In active systems, motility can drive T1s even when energy barriers exist, effectively fluidizing the tissue.

#### Higher-Order Rearrangements

Beyond T1s, tissues can exhibit:
- **Rosettes**: 5+ cells meeting at a single vertex (common during convergent extension)
- **T2 transitions**: cell extrusion (a cell leaves the monolayer)
- **Cell division**: creation of new cells (discussed below)

---

### Active Matter and Self-Propulsion

#### From Passive to Active

Traditional vertex models are **passive**: motion is driven only by minimizing the energy functional. Real cells are **active**: they consume ATP to generate forces and move persistently.

The active vertex model adds self-propulsion:
$$\gamma \frac{d\mathbf{r}_i}{dt} = \mathbf{F}_i^{VM} + f_a \mathbf{n}_i + \boldsymbol{\xi}_i$$

where:
- $\mathbf{F}_i^{VM}$ = force from vertex model energy
- $f_a$ = active force magnitude (self-propulsion strength)
- $\mathbf{n}_i$ = polarity vector (direction of self-propulsion)
- $\boldsymbol{\xi}_i$ = stochastic noise

#### Polarity Dynamics

The polarity vector $\mathbf{n}_i$ evolves according to:
$$\frac{d\theta_i}{dt} = \frac{\tau_i}{\gamma_r} + \eta_i$$

where $\theta_i$ is the angle of $\mathbf{n}_i$, $\tau_i$ is a torque (from alignment interactions), $\gamma_r$ is rotational friction, and $\eta_i$ is rotational noise with diffusion coefficient $D_r$.

The **persistence time** $\tau_p \sim 1/D_r$ characterizes how long a cell maintains its direction.

#### Motility Parameters

The key active parameters are:
- **Motile speed**: $v_0 = f_a/\gamma$ (how fast cells move)
- **Persistence time**: $\tau$ (how long they maintain direction)
- **Péclet number**: $Pe = v_0 \tau / a$ (persistence length / cell size)

#### Motility-Induced Fluidization

Even in tissues where $p_0 < 3.81$ (geometric solid), sufficient motility can:
1. Help cells overcome energy barriers to T1s
2. Drive the system out of local minima
3. Fluidize the tissue

This creates a **phase diagram** with axes $(p_0, v_0)$:
- Low $v_0$, low $p_0$: solid
- High $v_0$: fluid regardless of $p_0$
- High $p_0$: fluid regardless of $v_0$

---

### Cell Division and Death

#### Division in the Vertex Model

Cell division is implemented as:
1. Cell grows (increasing $A_0$) until reaching a threshold size $A_c$
2. Division occurs with probability $\propto (A - A_c)$
3. Two daughter cells are created, typically along the polarity axis
4. Each daughter gets half the parent's area; native areas reset

**Hertwig's rule**: cells tend to divide along their long axis (can be implemented via alignment with shape tensor).

#### Death/Extrusion

Cell death (apoptosis) or extrusion removes cells from the monolayer:
1. Cell is marked for death (based on age, damage signals, crowding)
2. Neighbors expand to fill the gap
3. Voronoi tessellation is recomputed

#### Effects on Glassy Dynamics

Cell division and death provide an alternative fluidization mechanism:
- Each division/death event rearranges local structure
- Even in the solid phase, cell cycling drives long-time diffusion
- But glass-like caging can persist at intermediate times if cycling is slow

Czajkowski et al. (2019) showed:
$$\text{Effective diffusion} = D_{\text{intrinsic}} + D_{\text{cycling}}$$

where both contributions add independently.

---

### Collective Migration and Flocking

#### From Single-Cell to Collective Motion

Individual cell motility + cell-cell interactions can produce emergent collective behaviors:
- **Streaming**: cells move in aligned streams
- **Vortices**: rotational flow patterns
- **Swirling**: disordered collective motion
- **Flocking**: globally aligned directed motion

#### Alignment Mechanisms

Cells can align their polarity with:
1. **Neighbors**: $\tau_i = -J_p \sum_j (\mathbf{n}_i \times \mathbf{n}_j)$ (Vicsek-like)
2. **Velocity**: $\tau_i = -J_v (\mathbf{n}_i \times \hat{\mathbf{v}}_i)$ (self-alignment)
3. **Cell shape**: $\tau_i = -J_s (\mathbf{n}_i \times \mathbf{p}_i)$ (shape tensor)

#### Flocking Transition

Giavazzi et al. (2018) found that alignment promotes:
1. Collective motion (obvious)
2. Solidification (counterintuitive!)

The intuition: alignment reduces effective fluctuations, which would otherwise help cells escape cages. An aligned flock can be collectively "frozen."

#### Experimental Signatures

Flocking in tissues produces:
- Hexatic orientational order
- Giant number fluctuations
- Velocity correlation length >> cell size
- Propagating waves of motion

---

## Key Concepts Reference

| Term | Definition | Typical Values |
|------|------------|----------------|
| **Shape Index ($p_0$)** | $P_0/\sqrt{A_0}$, target perimeter-to-root-area ratio | $p_0^* \approx 3.81$ |
| **Critical Shape Index ($p_0^*$)** | Value at rigidity transition | $\approx 3.812$ (regular pentagon) |
| **Area Modulus ($K$)** | Stiffness against area changes | $K \sim 1$ (normalized) |
| **Perimeter Modulus ($\Gamma$)** | Cortical contractility | $\Gamma \sim 0.04 - 1.0$ |
| **Line Tension ($\Lambda$)** | Energy per junction length | $\Lambda < 0$ (adhesive), $> 0$ (contractile) |
| **Active Force ($f_a$)** | Self-propulsion magnitude | $f_a \sim 0.01 - 1.0$ |
| **Persistence Time ($\tau$)** | Polarity decorrelation time | $\tau \sim 1 - 100$ (cell times) |
| **T1 Transition** | Neighbor exchange event | — |
| **Confluent** | No gaps between cells (100% coverage) | — |
| **Caging** | Subdiffusive regime from neighbor constraints | MSD plateau |
| **Flocking** | Collectively aligned directed motion | Order parameter $\psi \to 1$ |

---

## Connection to Current Simulations

Our GPU-accelerated vertex model simulation implements:
- Standard vertex model energy functional
- Overdamped Langevin dynamics
- Periodic boundary conditions
- Active self-propulsion with rotational diffusion
- T1 transitions via edge length monitoring

**Key parameters in our code:**
- `vA` (motility): corresponds to $v_0 = f_a/\gamma$
- `phi` (packing fraction): controls initial density
- `N` (cell number): 288 cells in production runs
- `L` (box size): determines packing ($L = 1600$ for $\phi = 0.85$, $L = 1562$ for $\phi = 0.89$)

Our jamming study explores the $(v_A, \phi)$ phase space to map the glass transition and measure:
- Mean squared displacement (MSD)
- T1 transition rates
- Shape distributions $p(q)$
- Overlap functions $Q(t)$

---

*Last updated: January 15, 2026*
