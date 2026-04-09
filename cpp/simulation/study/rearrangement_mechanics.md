# Rearrangement Mechanics in Phase-Field Cell Models

**Companion to [review_cell_jamming.md](review_cell_jamming.md) §7, §12.4.I and [glass_physics_tutorial.md](glass_physics_tutorial.md)**

*Prepared February 2026*

---

## Contents

1. [T1 Transitions: The Vertex Model Picture](#1-t1-transitions-the-vertex-model-picture)
2. [Why Discrete T1s Are an Artifact](#2-why-discrete-t1s-are-an-artifact)
3. [The Contact Area Observable](#3-the-contact-area-observable)
4. [Rearrangement Events as Trajectories](#4-rearrangement-events-as-trajectories)
5. [The Rearrangement Timescale Distribution](#5-the-rearrangement-timescale-distribution)
6. [Connection to Criticality and the Glass Transition](#6-connection-to-criticality-and-the-glass-transition)
7. [Energy Dissipation per Rearrangement](#7-energy-dissipation-per-rearrangement)
8. [Non-Confluent Rearrangements: A New Class](#8-non-confluent-rearrangements-a-new-class)
9. [Biological Connections](#9-biological-connections)
10. [Feasibility Analysis](#10-feasibility-analysis)
    - [Concrete Experimental Design](#concrete-experimental-design)
    - [The Smoking Gun Observable](#the-smoking-gun)

---

## 1. T1 Transitions: The Vertex Model Picture

In vertex/Voronoi models of confluent tissue, the elementary rearrangement is the **T1 transition** (neighbour exchange). Four cells meet at (or near) a fourfold vertex. One shared junction shrinks to zero length, the four-cell vertex forms, and a new junction opens perpendicular to the original:

```
Before:          4-vertex:         After:
  A | B            A                 A
  --+--          / | \              ---
  C | D        B---+---C           B   C
                 \ | /              ---
                   D                 D
```

The vertex model implements this as a **discrete topological surgery**: when a junction length $\ell_{ij}$ drops below a threshold $\ell_c$, the code deletes two edges, creates two new edges, and updates the connectivity. The transition is instantaneous by construction — one timestep cells $A$ and $D$ are neighbours, the next they are not.

### Energy Barriers

In the solid phase ($p_0 < p_0^*$), there is a finite barrier $\Delta E_{T1}$ to perform a T1:

$$\Delta E_{T1} \sim (p_0^* - p_0)^\psi$$

and Kramers theory gives the thermally/actively activated rate:

$$R_{T1} \sim \exp\!\left(-\frac{\Delta E_{T1}}{T_{\text{eff}}}\right), \qquad T_{\text{eff}} \sim v_A^2 \tau_p.$$

The T1 rate $R_{T1}$ serves as the microscopic fluidity: $R_{T1} = 0$ in the rigid solid, $R_{T1} > 0$ in the fluid. The macroscopic viscosity is related by $\eta \sim 1/R_{T1}$.

### The Problem

The T1 framework encodes three assumptions that are physically questionable:

1. **Discreteness.** The rearrangement happens in zero time. There is no concept of a "duration" — the event either happens or it doesn't.
2. **Completeness.** Every rearrangement runs to completion. If a junction starts shrinking, it always fully collapses and the new junction always opens. There are no aborted rearrangements, no partial exchanges, no lingering intermediate states.
3. **Four-fold symmetry.** The canonical T1 involves exactly four cells at a common vertex. Higher-order rosettes (five or six cells meeting transiently) are rare or handled by sequential T1 cascades.

All three assumptions are artefacts of the discrete-polygon representation.

---

## 2. Why Discrete T1s Are an Artifact

### Real Tissues

In real epithelial monolayers, rearrangements are observed to:

- Take **finite time** ($\sim$minutes to tens of minutes), during which cells visibly slide past one another.
- **Abort partway through**: two cells begin separating, then re-establish contact without completing the exchange.
- Pass through **multi-cell rosette configurations** where five or more cells converge to a common point (common in convergent extension during gastrulation).
- Show **continuous variation** in bond length, with no sharp threshold separating "neighbour" from "non-neighbour."

The vertex model's discrete T1 is a coarse-graining that discards this internal structure.

### Phase-Field Resolution

In the phase-field model, each cell is a continuous scalar field $\phi_i(\mathbf{r},t)$. Two cells are "in contact" wherever $\phi_i(\mathbf{r}) > 0$ and $\phi_j(\mathbf{r}) > 0$ simultaneously — i.e., their diffuse interfaces overlap. A rearrangement occurs when this overlap region:

- Grows (cells approach and make contact)
- Persists (cells press against each other)
- Shrinks continuously to zero (cells separate)
- And a new overlap emerges with a different neighbour

At no point is there a discrete topological operation. The contact area between any two cells varies **continuously in time**, and the "neighbourhood" is a continuous, graded quantity, not a binary label.

---

## 3. The Contact Area Observable

### Definition

The central observable for rearrangement mechanics is the **contact area** (2D: contact length) between cells $i$ and $j$:

$$C_{ij}(t) = \int \phi_i(\mathbf{r},t) \, \phi_j(\mathbf{r},t) \, d\mathbf{r}.$$

$C_{ij}$ is:

- **Nonnegative by construction** (both fields are nonneg).
- **Zero when cells don't touch** — if the supports of $\phi_i$ and $\phi_j$ are disjoint.
- **Smooth in time** — because the PDE dynamics is smooth.
- **Proportional to physical overlap** — the region where both cells claim territory.

### Physical Meaning

$C_{ij}$ is a direct measure of the mechanical interaction between cells $i$ and $j$. The repulsion energy between them is:

$$E_{ij}^{\text{rep}} = \frac{60\kappa}{\lambda^2} \int \phi_i^2 \phi_j^2 \, d\mathbf{r} \sim \frac{\kappa}{\lambda^2} C_{ij}^2$$

(approximately, since $\phi_i$ and $\phi_j$ are order unity in the overlap region). So $C_{ij}$ controls the interaction force: cells with large $C_{ij}$ repel strongly; those with $C_{ij} \to 0$ are about to lose contact.

### The Contact Graph

At any instant, the set $\{C_{ij}(t) > \epsilon\}$ defines a **weighted contact graph** $\mathcal{G}(t)$ where the weight of edge $(i,j)$ is $C_{ij}$. The topology of $\mathcal{G}(t)$ evolves continuously: edges fade in and out rather than snapping on and off. A "T1 event" in this language is a continuous trajectory in which:

1. $C_{ij}(t)$ decreases from a finite value to $\sim 0$ (cells $i$ and $j$ lose contact),
2. simultaneously, $C_{ik}(t)$ increases from $\sim 0$ to a finite value (cell $i$ gains a new neighbour $k$).

But unlike the vertex model, step (1) need not be followed by step (2). The cell can lose a contact without immediately gaining a new one (especially in the non-confluent regime where gaps exist).

---

## 4. Rearrangement Events as Trajectories

### Defining an "Event"

A **rearrangement event** for a given pair $(i,j)$ is a trajectory of the contact area $C_{ij}(t)$ that:

1. Starts above a threshold: $C_{ij}(t_0) > C_{\text{thresh}}$ (cells are in firm contact).
2. Drops below threshold: $C_{ij}(t_1) < C_{\text{thresh}}$ (cells have separated).
3. Remains below threshold for a persistence time: $C_{ij}(t) < C_{\text{thresh}}$ for $t \in [t_1, t_1 + \Delta t_{\text{persist}}]$ (not a transient fluctuation).

The **rearrangement duration** is $\tau_{\text{rearr}} = t_1 - t_0$, measured from the onset of contact decay to complete separation.

### The Choice of $C_{\text{thresh}}$

$C_{\text{thresh}}$ must separate "genuine contact" from "diffuse-interface tails brushing past each other." A natural choice is based on the interface structure:

$$C_{\text{thresh}} \sim \lambda \cdot h$$

where $\lambda$ is the interface width and $h$ is the grid spacing. This captures the physical contact while filtering numerical overlap.

The results should be checked for robustness: if the physics is real, the distribution $P(\tau_{\text{rearr}})$ should be qualitatively invariant to $C_{\text{thresh}}$ over a reasonable range (say, factor of 2–5).

### The Full Trajectory Carries More

Beyond the scalar duration $\tau_{\text{rearr}}$, each event has a full trajectory $C_{ij}(t)$ with shape information:

- **Monotonic decay**: a clean, irreversible separation. The cells decide to part and do so without hesitation.
- **Non-monotonic / oscillatory**: the cells slide, partially separate, re-establish contact, then finally part. This is an **aborted rearrangement** that eventually succeeds.
- **Aborted**: $C_{ij}$ drops toward zero but rebounds — the cells attempt to separate but fail. This event does not complete. 

The ratio of completed to aborted rearrangements is itself an observable that characterizes the energy landscape.

---

## 5. The Rearrangement Timescale Distribution

### Why This Is the Key Object

The distribution $P(\tau_{\text{rearr}})$ is the central quantity that vertex models **cannot access** and that encodes the physics of the rearrangement mechanism.

**Scenario 1: Narrow distribution** (peaked at a single timescale $\tau_0$).

$$P(\tau_{\text{rearr}}) \sim \delta(\tau_{\text{rearr}} - \tau_0)$$

This would validate the vertex model's implicit assumption: rearrangements are stereotyped events with a single intrinsic timescale. The vertex model's discrete T1 is a faithful coarse-graining — you lose no information by treating rearrangements as instantaneous, because they all take the same time anyway.

**Scenario 2: Broad distribution** (exponential or stretched-exponential tails).

$$P(\tau_{\text{rearr}}) \sim \exp\!\left[-\left(\frac{\tau_{\text{rearr}}}{\tau_0}\right)^\beta\right], \qquad \beta \leq 1$$

A spread of timescales, possibly with fat tails. This implies a range of local environments — some rearrangements are easy (low barrier, fast), others hard (high barrier, slow). The mean $\langle \tau_{\text{rearr}} \rangle$ exists but is poor at characterizing the distribution. The participation ratio of these events would carry information about the energy landscape roughness.

**Scenario 3: Power-law distribution** (scale-free).

$$P(\tau_{\text{rearr}}) \sim \tau_{\text{rearr}}^{-\alpha}$$

No characteristic timescale. Rearrangements of arbitrarily long duration occur with non-negligible probability. This signals **critical behaviour**: the system is at (or near) a point where the energy landscape is fractal. This would be a striking finding — it would directly connect the microscopic rearrangement mechanics to the macroscopic glass/jamming transition.

### Motility Dependence

The shape of $P(\tau_{\text{rearr}})$ should depend on $v_A$:

- **High $v_A$ (deep fluid):** cells have ample energy to overcome barriers. Rearrangements are fast and uniform. $P(\tau_{\text{rearr}})$ is narrow.
- **Near the transition:** barriers become important, some rearrangements are facile while others require cooperative motion. $P(\tau_{\text{rearr}})$ broadens. If the transition is truly critical, $P$ develops a power-law tail at the critical $v_A^*$.
- **Deep in the jammed phase:** rearrangements are rare activated events. When they happen, they tend to be slow and cooperative. $P(\tau_{\text{rearr}})$ shifts to large $\tau$ and may become bimodal (fast local rattles + slow cooperative escapes).

Plotting $P(\tau_{\text{rearr}}; v_A)$ as a function of $v_A$ across the transition gives a spectral view of how the microscopic mechanics evolves through the transition — **rearrangement spectroscopy**.

---

## 6. Connection to Criticality and the Glass Transition

### The Mean Rearrangement Time

The mean rearrangement duration $\langle \tau_{\text{rearr}} \rangle$ should relate to the macroscopic relaxation time $\tau_\alpha$. In a mean-field picture:

$$\tau_\alpha \sim N_{\text{coop}} \cdot \langle \tau_{\text{rearr}} \rangle$$

where $N_{\text{coop}}$ is the number of cells that must rearrange cooperatively for the system to relax. Near the glass transition, both $N_{\text{coop}}$ and $\langle \tau_{\text{rearr}} \rangle$ can diverge, and understanding which contribution dominates is a basic question.

If $\langle \tau_{\text{rearr}} \rangle$ diverges while $N_{\text{coop}}$ remains finite: the slowdown comes from individual events becoming sluggish (the "barrier" picture).

If $N_{\text{coop}}$ diverges while $\langle \tau_{\text{rearr}} \rangle$ remains finite: the slowdown is purely cooperative — each event is fast, but the system needs more of them in concert (the "facilitation" picture).

The phase-field model can distinguish these by independently measuring $\tau_\alpha$ (from MSD or $F_s$), $\langle \tau_{\text{rearr}} \rangle$ (from $C_{ij}$ trajectories), and $N_{\text{coop}}$ (from $\chi_4$ or from spatial clustering of simultaneous rearrangement events).

### Correlation Between Rearrangements and Dynamic Heterogeneity

Define the local rearrangement rate $\rho_{\text{rearr}}(\mathbf{r}, t)$ as the density of rearrangement events in a space-time window. This field should be **spatially correlated**: regions of high rearrangement activity cluster together (cooperative rearranging regions). The spatial correlation length of $\rho_{\text{rearr}}$ should coincide with the dynamic heterogeneity length $\xi_d$ from $\chi_4$ — providing a direct, microscopic connection between the two.

Furthermore: are the fast-moving cells (large displacement in time $t^*$) the same cells executing rearrangements? In the facilitation picture, rearrangements propagate as excitations through a background of caged cells. In the random energy picture, some cells are in soft spots and rearrange easily. These two scenarios make different predictions about the **temporal correlation** of rearrangement activity at a given cell.

---

## 7. Energy Dissipation per Rearrangement

### Extractable from the Model

Each rearrangement event has an associated energy cost that can be computed directly from the phase-field dynamics. The total dissipation during a rearrangement event of pair $(i,j)$ between times $t_0$ and $t_1$ is:

$$W_{ij} = \int_{t_0}^{t_1} \left|\frac{\partial E_{ij}}{\partial t}\right| dt$$

where $E_{ij}^{\text{rep}} = \frac{60\kappa}{\lambda^2} \int \phi_i^2 \phi_j^2 \, d\mathbf{r}$ is the interaction energy. This can also be decomposed into contributions from the Cahn-Hilliard (interface) energy, the volume constraint, and the repulsion.

### The Joint Distribution

The joint distribution $P(\tau_{\text{rearr}}, W)$ — rearrangement duration vs. energy dissipated — is the most informative single object. Its structure reveals the nature of the energy landscape:

- **Single cluster** at $(\tau_0, W_0)$: stereotyped events. Vertex model with a single T1 energy barrier.
- **Positive correlation** ($W \propto \tau_{\text{rearr}}$): longer events dissipate more — consistent with viscous friction dominating.
- **Two populations**: fast/low-energy events (thermal rattle-like) and slow/high-energy events (cooperative). This bimodality would directly evidence two classes of rearrangement.
- **Power-law ridge**: scale-invariant relationship between duration and dissipation. Critical behaviour.

---

## 8. Non-Confluent Rearrangements: A New Class

### The Non-Confluent Regime Opens New Channels

At $\phi = 0.85$, the tissue is non-confluent: there are voids between cells. This allows rearrangement mechanisms that have **no counterpart** in confluent vertex models:

**Gap-mediated exchange.** Two cells need not slide directly past each other. Cell $A$ can separate from cell $B$ by moving into an adjacent void, travel through the interstitial space, and re-establish contact with cell $C$ — completing a rearrangement without ever forming a four-cell vertex. The intermediate state involves a free cell surface exposed to empty space, not a compressed multi-cell junction.

**Gap opening.** A contact $C_{ij}$ can go to zero by the cells pulling apart and opening a physical gap, without any third cell intervening. This is topologically distinct from a T1 (which requires a new contact to replace the old one).

**Void migration.** The voids themselves act as topological defects in the contact network. A void migrating through the tissue shuffles the contact topology — cells gain and lose neighbours as the void passes — without any cell executing a traditional T1. This is analogous to vacancy diffusion in a crystal, and the void diffusion constant $D_{\text{void}}$ may control the macroscopic $D$ at low motility.

### Classification

This suggests defining a **rearrangement taxonomy**:

| Type | Description | Accessible to vertex model? |
|---|---|---|
| **T1-like** | Two cells exchange neighbours at a four-cell junction | Yes (by definition) |
| **Rosette** | Multi-cell vertex resolves with different connectivity | Partially (via sequential T1s) |
| **Gap-mediated** | Cell moves through void to new neighbourhood | No |
| **Gap-opening** | Two cells separate into void; no replacement | No |
| **Void-shuttled** | Void migrates, shuffling contacts along its path | No |

The relative frequency of these event types as a function of $\phi$ and $v_A$ is a measurement unique to non-confluent phase-field models.

---

## 9. Biological Connections

### Convergent Extension

During embryonic development (e.g., *Drosophila* germband extension, zebrafish gastrulation), tissues undergo large-scale shape changes driven by polarized cell rearrangements. The standard description uses T1 transitions with a preferred orientation. But live imaging shows that:

- Rearrangements take variable amounts of time (seconds to minutes).
- Multi-cell rosettes are common, especially in *Drosophila*.
- The "T1 rate" is a coarse-grained quantity; the underlying process is continuous.

The rearrangement timescale distribution $P(\tau_{\text{rearr}})$ is directly relevant: if it is broad, the effective T1 rate poorly characterizes the process, and theories built on a single rate parameter miss the physics.

### Tumour Invasion

At the invasion front, cells transition from a jammed epithelium to individual migration. This requires breaking cell-cell contacts — a rearrangement event. The duration of this contact-breaking process determines how quickly a cell can escape the primary tumour. If some cells can break contacts quickly (short $\tau_{\text{rearr}}$) because they are in a locally "soft" environment, they are the first to invade. The tail of $P(\tau_{\text{rearr}})$ controls the invasion rate.

This connects directly to the Griffiths picture (§15 of [glass_physics_tutorial.md](glass_physics_tutorial.md)): cells in a rare fluid region have short $\tau_{\text{rearr}}$ because the local landscape is smooth.

### Wound Healing

After wounding, cells at the margin must rearrange to close the gap. The healing velocity depends on:
1. How quickly individual rearrangements complete ($\langle \tau_{\text{rearr}} \rangle$).
2. How many rearrangements are needed per unit advance.
3. Whether aborted rearrangements slow the process.

The phase-field model can resolve all three contributions. In particular, the fraction of aborted rearrangements at the wound margin — events where cells begin to migrate but are pulled back by neighbours — is unmeasurable in vertex models.

### Cell Extrusion

In monolayers, dying or overcrowded cells are extruded (expelled from the layer). Extrusion involves the target cell losing all contacts simultaneously: $C_{ij}(t) \to 0$ for all neighbours $j$. The dynamics of this process — whether contacts are lost sequentially or simultaneously, and how the remaining cells close the gap — is a rearrangement mechanics question that has been studied only in vertex-model terms (sequential T1 cascade). The phase-field representation would reveal the actual continuous trajectory.

---

## 10. Feasibility Analysis

### What's Needed from the Simulation

**The central measurement** — $C_{ij}(t)$ for all contacting pairs — requires:

1. **Field data.** The full $\phi_i(\mathbf{r},t)$ at each output frame. The current VTK output stores this; the only question is temporal resolution.
2. **Pair identification.** At each frame, determine which cell pairs overlap. The bounding-box pair list already computed for the repulsion kernel provides this.
3. **Integration.** For each pair, compute $C_{ij} = \sum_{\mathbf{r}} \phi_i(\mathbf{r}) \phi_j(\mathbf{r}) \, (\Delta x)^2$. On a $1600^2$ grid, this is a trivial inner product — $O(N_{\text{grid}})$ per pair, and the number of pairs is $O(N)$ (sparse contact graph).

### Temporal Resolution Requirements

This is the key constraint. The rearrangement timescale $\tau_{\text{rearr}}$ must be resolved. From the simulation parameters:

$$\tau_{\text{interface}} \sim \frac{\lambda^2}{M\gamma} = \frac{49}{0.5 \times 1} \sim 100 \quad \text{(timesteps)}$$

This is the timescale for the interface to relax — the fastest timescale in a rearrangement. To resolve $C_{ij}(t)$ properly:

$$\Delta t_{\text{output}} \lesssim \tau_{\text{interface}} / 5 \sim 20 \quad \text{(timesteps)}$$

The current production output interval is every 100 timesteps — marginally sufficient for the slowest rearrangements near the transition, but too coarse for the fast events in the fluid phase. A dedicated rearrangement study would benefit from more frequent output, or from computing $C_{ij}(t)$ **on-device** during the simulation (avoiding the I/O bottleneck entirely).

### On-Device Computation (Preferred)

Instead of writing $C_{ij}$ to disk via VTK and post-processing:

1. After each (or every $k$-th) timestep, compute $C_{ij}$ for all pairs in the bounding-box pair list.
2. Store the time series $\{C_{ij}(t)\}$ in a compact format (one float per pair per output step).
3. Write only this time series to disk.

The per-pair integration is embarrassingly parallel and fits on the GPU with no additional memory (the fields are already resident). The output data volume is tiny: $N_{\text{pairs}} \times T_{\text{frames}} \sim 1000 \times 10000 \sim 10^7$ floats $\sim 40$ MB — negligible compared to the VTK dumps.

### System Size

$N = 288$ at $\phi = 0.85$ gives:

- Average neighbours per cell: $z \sim 5$–$6$ (matching typical 2D packings).
- Total unique contact pairs: $\sim Nz/2 \sim 800$–$900$ at any instant.
- Total contact-loss events per run: depends on fluidity.

At high $v_A$ (fluid phase), cells rearrange frequently — expect $O(10^3)$–$O(10^4)$ events per run. This gives an ample sample for $P(\tau_{\text{rearr}})$.

At low $v_A$ (near jamming), rearrangements are rare — perhaps $O(10)$–$O(100)$ per run. The 100-replica ensemble provides $O(10^3)$–$O(10^4)$ total events even in this regime. Sufficient for the shape of $P(\tau)$, marginal for the tails.

### Concrete Experimental Design

**Phase 1: Rearrangement spectroscopy across the transition**

| Parameter | Value |
|---|---|
| $v_A$ | 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013 |
| $N$ | 288 |
| $\phi$ | 0.85 |
| $t_{\text{end}}$ | $8.8 \times 10^5$ (current production target) |
| Replicas | 100 (existing production runs) |
| Output | $C_{ij}(t)$ every 10 timesteps (on-device) |

**This requires no additional simulation runs.** If $C_{ij}(t)$ computation is added on-device, the existing production sweep directly provides all data.

Alternatively, **Phase 1a** (post-processing from existing VTK):

| Constraint | Existing capability |
|---|---|
| Fields per frame | All $\phi_i$ available in VTK |
| Frame interval | $\Delta t = 100$ timesteps |
| Frames per run | $8800$ at $t_{\text{end}} = 8.8 \times 10^5$ |
| Temporal resolution | $\Delta t = 100$ may miss fast events at high $v_A$; sufficient near the transition |

**Phase 2: Non-confluent rearrangement taxonomy**

Run at multiple $\phi$ values to vary the void fraction:

| $\phi$ | Expected void fraction | Dominant rearrangement type |
|---|---|---|
| 0.75 | $\sim 25\%$ | Gap-mediated, void-shuttled |
| 0.80 | $\sim 20\%$ | Mixed |
| 0.85 | $\sim 15\%$ | T1-like + gap-mediated |
| 0.90 | $\sim 10\%$ | Mostly T1-like |
| 0.95 | $\sim 5\%$ | Nearly confluent T1-like |

Classify events by type (§8 taxonomy). The crossover packing fraction $\phi^*$ where gap-mediated events first appear is a phase-field-specific observable.

**Phase 3: Joint distributions**

For each $v_A$ and $\phi$:
1. Construct $P(\tau_{\text{rearr}})$.
2. Construct $P(W)$ (energy dissipated per event).
3. Construct the joint $P(\tau_{\text{rearr}}, W)$.
4. Correlate rearrangement locations with the $\chi_4$ mobility map.
5. Compute the spatial autocorrelation of rearrangement events $\langle \rho_{\text{rearr}}(\mathbf{r},t) \, \rho_{\text{rearr}}(\mathbf{r}',t) \rangle$.

### Code Changes Required

| Component | Change | Effort |
|---|---|---|
| **Pair $C_{ij}$ kernel** | New CUDA kernel: for each pair $(i,j)$ in the bounding box pair list, compute $\sum_{\mathbf{r}} \phi_i \phi_j \, (\Delta x)^2$ over the bounding box intersection. | Small — the pair list and field access pattern already exist in the repulsion kernel. |
| **Time series storage** | Device-side buffer for $C_{ij}(t)$. Flush to host periodically. | Small. |
| **Output format** | Binary file: header (pair list), then $N_{\text{pairs}} \times T_{\text{frames}}$ float array. | Trivial. |
| **Post-processing** | Python script: read $C_{ij}(t)$, detect events (threshold crossing), compute $\tau_{\text{rearr}}$, build distributions. | Moderate. |

### The Smoking Gun

The most discriminating observable — the result that would constitute a clear, publishable finding — is the **scaling of $P(\tau_{\text{rearr}})$ as $v_A \to v_A^*$ from the fluid side.**

If the glass/jamming transition is truly critical:

$$P(\tau_{\text{rearr}}) \sim \tau_{\text{rearr}}^{-\alpha} \mathcal{F}\!\left(\frac{\tau_{\text{rearr}}}{\tau^*}\right)$$

where $\tau^*(v_A) \sim |v_A - v_A^*|^{-z\nu}$ is a diverging cutoff and $\mathcal{F}$ is a scaling function. The exponent $\alpha$ and the scaling collapse across different $v_A$ values are quantitative, testable predictions that:

1. Are **inaccessible to vertex models** (which have $P(\tau_{\text{rearr}}) = \delta(\tau - 0)$ by construction).
2. Would establish a **new observable for the glass transition** — one that has never been measured in any simulation.
3. Would connect to experiments where rearrangement durations are measurable via live imaging.

The secondary smoking gun is the **aborted rearrangement fraction** $f_{\text{abort}}(v_A)$:

- $f_{\text{abort}} \to 0$ in the fluid (all rearrangements succeed).
- $f_{\text{abort}} \to 1$ approaching the jammed phase (cells attempt to rearrange but fail).
- The $v_A$ dependence of $f_{\text{abort}}$ is a direct probe of the energy landscape.

No prior study, in any model, has measured $f_{\text{abort}}$. Its existence is a prediction of the phase-field framework.

---

## Why This Is Novel

1. The rearrangement timescale distribution $P(\tau_{\text{rearr}})$ has never been computed in any cell model. All existing studies treat rearrangements as either instantaneous (vertex) or unresolved (coarse-grained continuum).
2. The decomposition of the macroscopic relaxation into $\tau_\alpha \sim N_{\text{coop}} \cdot \langle \tau_{\text{rearr}} \rangle$ — separating the cooperative from the single-event contribution — is a new framework for the glass transition that requires a continuous model.
3. Non-confluent rearrangement channels (gap-mediated, void-shuttled) are **undefined** in confluent models. Their existence and relative weight constitute genuinely new physics.
4. The connection to biological processes (invasion, extrusion, wound healing) through rearrangement mechanics is direct and experimentally testable.

---

## References

See [review_cell_jamming.md](review_cell_jamming.md) for the complete reference list. Additional references for this document:

- Duclut, C. et al. Active T1 transitions in cellular networks. *Eur. Phys. J. E* **45**, 33 (2022).
- Rauzi, M. et al. Nature and anisotropy of cortical forces orienting *Drosophila* tissue morphogenesis. *Nat. Cell Biol.* **10**, 1401 (2008).
- Zhou, Y. & Milner, S. T. T1 process and dynamics in glass-forming hard-sphere liquids. *Soft Matter* **11**, 2700 (2015).
- Blankenship, J. T. et al. Multicellular rosette formation links planar cell polarity to tissue morphogenesis. *Dev. Cell* **11**, 459 (2006).
- Falk, M. L. & Langer, J. S. Dynamics of viscoplastic deformation in amorphous solids. *Phys. Rev. E* **57**, 7192 (1998).
- Manning, M. L. & Liu, A. J. Vibrational modes identify soft spots in a sheared disordered packing. *Phys. Rev. Lett.* **107**, 108302 (2011).
