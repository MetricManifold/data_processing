# Glass Physics & Jamming Theory — A Tutorial

**Companion to [review_cell_jamming.md](review_cell_jamming.md)**

*Prepared February 2026*

---

## Contents

1. [Ergodicity and Its Breaking](#1-ergodicity-and-its-breaking)
2. [Dynamic Heterogeneity](#2-dynamic-heterogeneity)
3. [Packing Fraction](#3-packing-fraction)
4. [The "Ballistic" Regime in Overdamped Systems](#4-the-ballistic-regime-in-overdamped-systems)
5. [The Self-Intermediate Scattering Function](#5-the-self-intermediate-scattering-function)
6. [VFT Divergence and Super-Arrhenius Slowing](#6-vft-divergence-and-super-arrhenius-slowing)
7. [The Kauzmann Temperature and the Entropy Crisis](#7-the-kauzmann-temperature-and-the-entropy-crisis)
8. [Falling Out of Equilibrium vs. Breaking Ergodicity](#8-falling-out-of-equilibrium-vs-breaking-ergodicity)
9. [The Mori-Zwanzig Projection and Mode-Coupling Theory](#9-the-mori-zwanzig-projection-and-mode-coupling-theory)
10. [The Density-Density Correlator](#10-the-density-density-correlator)
11. [The J-Point](#11-the-j-point)
12. [Origin of $p^* \approx 3.81$](#12-origin-of-p-approx-381)
13. [Quenched vs. Annealed Disorder](#13-quenched-vs-annealed-disorder)
14. [Model A, Model B, and the Cell Model](#14-model-a-model-b-and-the-cell-model)
15. [Griffiths Effects in Cell Systems](#15-griffiths-effects-in-cell-systems)
    - [Feasibility Analysis: System Size, Time, and Parameter Choices](#feasibility-analysis-what-it-takes-to-observe-griffiths-effects)
    - [Concrete Experimental Design](#concrete-experimental-design)
    - [The Smoking Gun Observable](#the-smoking-gun)

---

## 1. Ergodicity and Its Breaking

### The Ergodic Hypothesis

The foundational claim of equilibrium statistical mechanics is that **time averages equal ensemble averages**:

$$\lim_{T \to \infty} \frac{1}{T} \int_0^T A\bigl(\mathbf{r}(t)\bigr) \, dt = \langle A \rangle_{\text{ensemble}} = \frac{\int A(\mathbf{r}) \, e^{-\beta H(\mathbf{r})} \, d\mathbf{r}}{\int e^{-\beta H(\mathbf{r})} \, d\mathbf{r}}.$$

This works when the system's trajectory explores the entire accessible phase space uniformly over time. In a liquid, a cell wanders everywhere — it visits all possible positions. Wait long enough, and a snapshot of one trajectory looks the same as a snapshot across many independent copies of the system.

### How Ergodicity Breaks at the Glass Transition

As the system approaches the glass transition (increasing density or decreasing temperature), the phase space develops a complex landscape of basins separated by growing energy barriers. The system's trajectory becomes **trapped** in a subset of phase space — it can only explore configurations near its current arrangement. Cells rattle in their cages but never escape.

The **non-ergodicity parameter** $f(q)$ quantifies this directly:

$$f(q) = \lim_{t \to \infty} F_s(q, t).$$

- In a liquid (ergodic): $f(q) = 0$ — the system forgets its initial configuration at long times.
- In a glass (non-ergodic): $f(q) > 0$ — the system permanently remembers where it started.

### Connection to Cell Caging

In the cellular context, ergodicity breaking means cells are trapped by their neighbours. A cell in a jammed tissue oscillates around a mean position but never exchanges places with its neighbours. The MSD plateau is the direct spatial signature:

$$\Delta^2(t) \to r_{\text{loc}}^2 \quad \text{(constant)}$$

where $r_{\text{loc}}$ is the **localization length** — the size of the cage. A flowing tissue ($f(q) = 0$) eventually decorrelates; a jammed tissue ($f(q) > 0$) does not.

---

## 2. Dynamic Heterogeneity

### The Key Observation

Glass-forming systems are not uniformly slow. At any given moment, there are spatially correlated regions of fast and slow particles. Some cells are stuck in rigid clusters, while nearby cells are executing cooperative rearrangements. These regions exchange roles over time — today's fast cell may be tomorrow's slow cell.

### Quantitative Measures

**Non-Gaussian parameter** $\alpha_2(t)$:

$$\alpha_2(t) = \frac{d}{d+2} \frac{\langle r^4(t) \rangle}{\langle r^2(t) \rangle^2} - 1$$

For a population with a single diffusion coefficient, displacements are Gaussian and $\alpha_2 = 0$. A positive $\alpha_2$ means the displacement distribution has fat tails — some cells have moved much further than expected, others much less. The peak of $\alpha_2(t)$ occurs at $t = t^*$, the time of maximum heterogeneity, typically near the end of the caging plateau when some cells are beginning to escape their cages while others remain trapped.

**Four-point susceptibility** $\chi_4(t)$: this measures the *volume* of dynamically correlated regions. Define the overlap:

$$Q(t) = \frac{1}{N} \sum_i \Theta\bigl(a - |\mathbf{r}_i(t) - \mathbf{r}_i(0)|\bigr)$$

then

$$\chi_4(t) = N \bigl[\langle Q(t)^2 \rangle - \langle Q(t) \rangle^2 \bigr].$$

The peak $\chi_4(t^*)$ grows as the transition is approached, scaling as $\chi_4 \sim \xi_d^d$ where $\xi_d$ is a **dynamic correlation length**. This is one of the most important observables for characterizing the glass transition.

### Measurement in Cell Simulations

Both $\alpha_2(t)$ and $\chi_4(t)$ are computable from trajectory data — centroid positions $\mathbf{r}_i(t)$ as a function of time. The existing 100-replica ensemble at each $v_A$ provides the statistical averaging needed for $\chi_4$. A growing $\chi_4$ peak as $v_A$ decreases would be direct evidence that the transition is glass-like, not merely a kinetic slowdown.

---

## 3. Packing Fraction

The packing fraction $\phi$ is simply the fraction of the domain area (2D) or volume (3D) occupied by cells:

$$\phi = \frac{\text{Total cell area}}{\text{Domain area}} = \frac{N \times \pi R^2}{L^2}$$

For our simulations: $N = 288$ cells, $R = 49$, $L = 1600$ gives

$$\phi = \frac{288 \times \pi \times 49^2}{1600^2} \approx 0.849 \approx 85\%.$$

This means 85% of the simulation domain is covered by cells, with 15% being empty (interstitial) space. The non-confluent regime ($\phi < 1$) is precisely what makes our phase-field simulations novel — vertex models enforce $\phi = 1$ by construction.

---

## 4. The "Ballistic" Regime in Overdamped Systems

### Why "Ballistic" Is Misleading

In molecular systems, the short-time $\Delta^2(t) \sim t^2$ regime is truly **ballistic**: particles coast on their inertia ($\Delta r \sim v_{\text{thermal}} \cdot t$) before encountering a collision. But cells are violently overdamped — the Reynolds number is ${\sim}10^{-8}$. There is no inertia. A cell with no applied force stops instantly.

### What the $t^2$ Regime Actually Is

The early-time $t^2$ scaling in active cell systems comes from the **persistence of self-propulsion**, not inertia:

$$\Delta^2(t) \approx v_A^2 t^2 \quad \text{for } t \ll \min(\tau_p, \tau_{\text{collision}}).$$

A cell with active velocity $v_A$ moves in a straight line (direction $\hat{\mathbf{n}}_i$) until either:

1. **It tumbles** — the polarity reorients (after persistence time $\tau_p$), or
2. **It collides** — it runs into a neighbour (after time $\tau_{\text{collision}} \sim \ell_{\text{gap}}/v_A$).

Whichever happens first terminates the $t^2$ regime. The crossover to diffusive behaviour ($\Delta^2 \sim 2dDt$) at long times gives $D \sim v_A^2 \tau_{\text{eff}}$ where $\tau_{\text{eff}} = \min(\tau_p, \tau_{\text{collision}})$.

The correct nomenclature would be **persistent** rather than ballistic, but the literature commonly uses "ballistic" for any $t^2$ scaling.

---

## 5. The Self-Intermediate Scattering Function

### Definition

$$F_s(q,t) = \frac{1}{N} \left\langle \sum_{i=1}^{N} e^{i\mathbf{q} \cdot [\mathbf{r}_i(t) - \mathbf{r}_i(0)]} \right\rangle.$$

At $t = 0$: $F_s(q,0) = 1$ (perfect self-correlation). At $t \to \infty$: $F_s \to 0$ in a liquid (complete decorrelation), $F_s \to f(q) > 0$ in a glass. The wavevector $q$ is typically chosen at the first peak of the static structure factor $S(q)$, probing correlations on the length scale of the inter-particle spacing.

### Two-Step Relaxation

Near the glass transition, $F_s(q,t)$ develops a characteristic two-step decay:

```
F_s(q,t)
  1 |*
    | *
    |  *                           ← Fast β-relaxation
    |   *                             (in-cage vibrations)
    |    *
    |     *-------*-------*        ← Plateau at f(q)
    |                      *          (non-ergodicity parameter)
    |                       *
    |                        *     ← Slow α-relaxation  
    |                         *       (cage escape, cooperative)
    |                          *
    |                           **
  0 |_____________________________***___
    0              log(t)              ∞
```

**$\beta$-relaxation** (fast, $t \sim \tau_\beta$): cells rattle within their cages, decorrelating the fast vibrational degrees of freedom. The scattering function drops from 1 to the plateau value $f(q)$.

**Plateau at $f(q)$** ($\tau_\beta \ll t \ll \tau_\alpha$): cells are trapped. The remaining correlation $f(q)$ measures the fraction of the initial structure that is "frozen" — the cage survives. The plateau widens as the transition is approached ($\tau_\alpha / \tau_\beta$ grows).

**$\alpha$-relaxation** (slow, $t \sim \tau_\alpha$): cells finally escape their cages through cooperative rearrangements. The decay is well-described by a **stretched exponential** (Kohlrausch-Williams-Watts):

$$F_s(q,t) \approx f(q) \exp\!\left[-\left(\frac{t}{\tau_\alpha}\right)^\beta\right]$$

with stretching exponent $\beta < 1$. The stretching arises from the superposition of different local relaxation times (dynamic heterogeneity). At the glass transition, $\tau_\alpha \to \infty$ and the $\alpha$-decay is never completed — $F_s$ permanently stalls at $f(q)$.

---

## 6. VFT Divergence and Super-Arrhenius Slowing

### The Vogel-Fulcher-Tammann Form

The structural relaxation time near the glass transition is often well-described by:

$$\tau_\alpha = \tau_0 \exp\!\left(\frac{A}{T - T_0}\right)$$

where $T_0$ is a **positive constant** (a finite temperature). The key claim is not that we require $T > T_0$ for some operational reason, but rather that the divergence happens at a **finite temperature** $T_0 > 0$, not at $T = 0$.

**Why this matters:** A simple Arrhenius form $\tau = \tau_0 e^{\Delta E / k_B T}$ only diverges at $T = 0$ — you'd need to cool all the way to absolute zero to arrest the system. VFT says the dynamics diverges at $T_0 > 0$: even at a finite temperature, relaxation would take infinite time. This is qualitatively different — it suggests a hidden thermodynamic transition at $T_0$.

In practice, the system falls out of equilibrium at $T_g > T_0$ (the laboratory glass transition), so the predicted divergence is never directly observed. Whether $T_0$ represents a true singularity or is merely a fitting parameter remains one of the deepest open questions in condensed matter physics.

### For Active Cells

The analogue mapping uses effective temperature $T_{\text{eff}} \sim v_A^2 \tau_p$. The VFT form becomes

$$\tau_\alpha \sim \exp\!\left(\frac{A}{v_A^2 \tau - (v_A^2 \tau)_0}\right)$$

and the question is whether there exists a critical motility $(v_A^2 \tau)_0 > 0$ at which the dynamics diverges — a true glass transition in the phase diagram.

---

## 7. The Kauzmann Temperature and the Entropy Crisis

### The Paradox

As a liquid is cooled toward the glass transition, its entropy decreases. Kauzmann (1948) noticed that if you extrapolate the liquid entropy curve below $T_g$ (where the system actually falls out of equilibrium), it would cross and fall below the crystal entropy at a temperature $T_K$.

This is the **Kauzmann paradox**: a disordered system cannot have less entropy than an ordered crystal — that would mean the disordered state has fewer accessible configurations than the ordered one, which is nonsensical.

### Thermodynamic Resolution: RFOT

Random First-Order Transition (RFOT) theory resolves the paradox by positing that a true thermodynamic transition occurs at $T_K$, where the **configurational entropy** $S_c$ — the entropy counting the number of distinct amorphous packings — vanishes:

$$S_c(T_K) = 0.$$

At $T_K$, only a sub-exponential number of metastable states (glass configurations) remain. The system is locked into essentially one amorphous state.

### The Adam-Gibbs Relation

RFOT connects entropy to dynamics through the Adam-Gibbs relation:

$$\tau_\alpha = \tau_0 \exp\!\left(\frac{C}{T \, S_c(T)}\right).$$

The physical picture: relaxation requires cooperative rearrangement of a region of size $\xi^d$ where $\xi$ is the **cooperative rearranging region (CRR)** length. The minimum CRR size is set by the configurational entropy:

$$\xi^d \sim \frac{1}{S_c}.$$

As $S_c \to 0$ (at $T_K$), the CRR size diverges — the entire system must rearrange cooperatively — and the relaxation time diverges. If $S_c$ vanishes linearly, $S_c \sim (T - T_K)$, the Adam-Gibbs relation reproduces VFT with $T_0 = T_K$.

### The Temperature Hierarchy

$$T_d > T_g > T_K$$

| Temperature | Name | Meaning |
|---|---|---|
| $T_d$ | Dynamical (MCT) crossover | Free energy barriers appear; MCT divergence; landscape trapping begins |
| $T_g$ | Laboratory glass transition | The system falls out of equilibrium on experimental timescales ($\tau_\alpha \sim 10^2$–$10^3$ s) |
| $T_K$ | Kauzmann temperature | $S_c = 0$; true thermodynamic transition (RFOT), possibly unreachable |

Between $T_d$ and $T_g$: the system is still technically equilibrating, but dynamics are already super-Arrhenius and heterogeneous. This is the "landscape-influenced" regime.

Below $T_g$: the system is out of equilibrium — properties depend on cooling rate, aging occurs, memory effects appear.

Whether $T_K$ exists as a true singularity (RFOT view) or is merely an artifact of extrapolation (kinetic view) remains debated. Experimentally, it is inaccessible because $T_g$ intervenes first.

### Mapping to Cell Systems

The effective temperature mapping $T_{\text{eff}} \sim v_A^2 \tau$ means:

| Glass concept | Cell analogue |
|---|---|
| $T_d$ | Motility where MCT-like caging first appears |
| $T_g$ | Motility where $\tau_\alpha$ exceeds the observation time |
| $T_K$ | Critical motility where $S_c = 0$ (if it exists) |

---

## 8. Falling Out of Equilibrium vs. Breaking Ergodicity

These are distinct but related concepts that are often conflated.

### Ergodicity Breaking: The Microscopic Mechanism

Ergodicity breaking means the system's trajectory in phase space becomes **confined to a subset** of the states allowed by the energy. The system cannot explore the full equilibrium distribution — certain configurations become dynamically inaccessible, even though they are thermodynamically available. This is captured by $f(q) > 0$: the self-correlation never fully decays, the system permanently remembers where it started.

### Out of Equilibrium: The Macroscopic Consequence

A system is out of equilibrium when its macroscopic properties **depend on its history** — when it was prepared, how fast it was cooled, how long it has been waiting. The hallmarks are:

- **Aging**: properties drift with waiting time $t_w$
- **Protocol dependence**: different cooling rates produce different glasses
- **Violation of FDT**: the fluctuation-dissipation theorem breaks down

### The Distinction

Ergodicity breaking is the **mechanism**; being out of equilibrium is the **consequence**. A system that cannot explore all states cannot find the true equilibrium — it is stuck in a metastable condition whose properties depend on how it got there.

### $T_g$ Is Kinetic, Not Thermodynamic

Crucially, the laboratory glass transition at $T_g$ is **not a true phase transition**. There is no divergent correlation length, no order parameter, no latent heat. $T_g$ depends on the cooling rate: cool faster → higher $T_g$ (the system falls out of equilibrium sooner because it has less time to explore). This is why $T_g$ is defined operationally by $\tau_\alpha \sim 10^{2}\text{–}10^{3}$ s (or equivalently, viscosity $\eta \sim 10^{12}$ Pa·s).

### Active Cells: Always Out of Equilibrium

Here is a subtlety specific to living systems: active cells are **always** out of equilibrium, even in the fluid phase. ATP hydrolysis continuously drives the system, breaking detailed balance at the molecular level. The cell isn't a thermal system with temperature $T$ — it's an energy-consuming engine.

The relevant question is not "is the cell system in equilibrium?" (it never is) but rather "is its **steady state** ergodic?" That is: does the system, under constant active driving, explore all accessible configurations? If yes, time averages converge and the steady state is well-defined (though non-equilibrium). If no, the system is in a non-ergodic steady state — a driven glass.

---

## 9. The Mori-Zwanzig Projection and Mode-Coupling Theory

### The Starting Point: Liouville Dynamics

In classical mechanics, the full microscopic state evolves under the Liouville operator:

$$\frac{dA}{dt} = i\mathcal{L} A, \qquad (i\mathcal{L})A = \{A, H\} = \sum_i \left(\frac{\partial A}{\partial \mathbf{r}_i} \cdot \frac{\partial H}{\partial \mathbf{p}_i} - \frac{\partial A}{\partial \mathbf{p}_i} \cdot \frac{\partial H}{\partial \mathbf{r}_i}\right)$$

where $\{A, H\}$ is the Poisson bracket. This is exact but useless — it contains all $6N$ degrees of freedom. We want an equation for some **slow variable** $A(t)$ (like the density fluctuation $\rho_{\mathbf{q}}$) alone.

### The Projection

Define a projection operator $\mathcal{P}$ onto the subspace spanned by $A$:

$$\mathcal{P} B = \frac{\langle B A^* \rangle}{\langle A A^* \rangle} A$$

and its complement $\mathcal{Q} = 1 - \mathcal{P}$ (the "orthogonal" or "irrelevant" subspace). Then the Mori-Zwanzig identity gives the **exact** generalized Langevin equation:

$$\dot{A}(t) = i\Omega A(t) - \int_0^t m(t - t') A(t') \, dt' + f(t)$$

where:

| Term | Name | Physical Meaning |
|---|---|---|
| $i\Omega A(t)$ | Streaming/frequency term | Reversible oscillation of $A$ |
| $\int_0^t m(t') A(t-t') \, dt'$ | Memory kernel | History-dependent friction from eliminated degrees of freedom |
| $f(t)$ | Fluctuating force | Random kicks from the fast (projected-out) variables |

The memory kernel is:

$$m(t) = \frac{\langle f(t) f^*(0) \rangle}{\langle A A^* \rangle}$$

This is exact — it's just a rearrangement of the full dynamics. The physics is that the "fast" variables we projected out exert two effects on $A$: a systematic drag (memory kernel) and random fluctuations (noise).

### The MCT Closure

The Mori-Zwanzig equation is exact but unclosed — $m(t)$ is as complicated as the original problem. **Mode-coupling theory** makes a specific approximation to close it.

For the density correlator $\Phi(q,t) = F(q,t)/S(q)$, the exact equation is:

$$\ddot{\Phi}(q,t) + \Omega_q^2 \Phi(q,t) + \int_0^t m(q,t-t') \dot{\Phi}(q,t') \, dt' = 0$$

MCT approximates the memory kernel as a **quadratic functional** of the correlator itself:

$$m(q,t) \approx \sum_{\mathbf{k}+\mathbf{p}=\mathbf{q}} V(\mathbf{q};\mathbf{k},\mathbf{p}) \, \Phi(k,t) \, \Phi(p,t)$$

The vertices $V$ are determined entirely by the **static structure factor** $S(q)$. This makes the equation **self-consistent**: $\Phi$ determines $m$, which determines $\Phi$.

The physical meaning: the dominant mechanism for slow density relaxation is the coupling of the density mode at wavevector $\mathbf{q}$ to pairs of density modes at $\mathbf{k}$ and $\mathbf{p} = \mathbf{q} - \mathbf{k}$. Each pair blocks the relaxation of $\mathbf{q}$ (a cell can't move because its cage — described by modes $\mathbf{k}$ and $\mathbf{p}$ — hasn't relaxed), and the cage can't relax because its constituents are themselves caged. This self-consistent feedback is what produces the dynamical arrest.

---

## 10. The Density-Density Correlator

### Full vs. Self Correlator

The **full** density-density correlator measures collective density fluctuations:

$$F(q,t) = \frac{1}{N} \left\langle \rho_{\mathbf{q}}(t) \rho_{-\mathbf{q}}(0) \right\rangle = \frac{1}{N} \left\langle \sum_{i,j} e^{i\mathbf{q}\cdot[\mathbf{r}_i(t) - \mathbf{r}_j(0)]} \right\rangle$$

where $\rho_{\mathbf{q}} = \sum_i e^{i\mathbf{q}\cdot\mathbf{r}_i}$ is the Fourier transform of the microscopic density. At $t = 0$:

$$F(q,0) = S(q) = 1 + \frac{1}{N} \left\langle \sum_{i \neq j} e^{i\mathbf{q}\cdot[\mathbf{r}_i - \mathbf{r}_j]} \right\rangle$$

which is the **static structure factor** — the Fourier transform of the pair correlation function $g(r)$.

The **self** part (§5) keeps only $i = j$ terms. The normalized correlator is:

$$\Phi(q,t) = \frac{F(q,t)}{S(q)}$$

with $\Phi(q,0) = 1$.

### Why MCT Projects onto Density

MCT chooses the density modes $\{\rho_{\mathbf{q}}\}$ as the slow variables because:

1. At long wavelengths, density fluctuations are conserved (particle number conservation) — they must relax slowly.
2. For glass-forming liquids near arrest, the dominant slow modes are density fluctuations at wavevectors near the first peak of $S(q)$ — the inter-particle spacing.
3. $S(q)$ (the equal-time density correlator) encodes all pair-level structural information. MCT shows that this is sufficient to predict the dynamics.

### For Cells

In phase-field simulations, the "density" is naturally defined as

$$\rho(\mathbf{r},t) = \sum_i \phi_i(\mathbf{r},t)$$

which is the total coverage field. The structure factor $S(q)$ computed from this field would probe spatial ordering of cells at different length scales. The first peak of $S(q)$ occurs at $q \sim 2\pi / d_{\text{cell}}$ where $d_{\text{cell}}$ is the typical cell spacing.

---

## 11. The J-Point

### Athermal Jamming

The jamming transition occurs at $T = 0$ (no thermal fluctuations) for soft repulsive particles. It is the critical point $(T = 0, \phi = \phi_J, \Sigma = 0)$ in the Liu-Nagel jamming phase diagram.

At $\phi < \phi_J$: particles don't touch, zero energy, zero pressure — a collection of disconnected objects.

At $\phi > \phi_J$: particles are forced into contact, finite energy, finite pressure — a rigid solid.

The transition at $\phi_J$ is sharp (in the $N \to \infty$ limit) and has critical scaling.

### Isostaticity and Maxwell Counting

At the J-point, the contact number per particle $z$ equals the **Maxwell isostatic value**:

$$z_c = 2d$$

where $d$ is the spatial dimension ($z_c = 4$ in 2D, $z_c = 6$ in 3D). This comes from Maxwell counting: each particle has $d$ positional degrees of freedom, and each contact provides one constraint (a force equation). Stability requires constraints $\geq$ degrees of freedom:

$$\frac{Nz}{2} \geq Nd \implies z \geq 2d.$$

At exactly $z = 2d$ (isostaticity), the system has just barely enough contacts to be rigid — it is **marginally stable**.

### Scaling Laws Near the J-Point

Above $\phi_J$, the following scaling relations hold:

| Quantity | Scaling | Physical Meaning |
|---|---|---|
| Excess contacts $\delta z$ | $\sim (\phi - \phi_J)^{1/2}$ | New contacts form as square root of compression |
| Pressure $P$ | $\sim (\phi - \phi_J)^{\alpha-1}$ | $\alpha = 2$ for harmonic, $5/2$ for Hertzian |
| Shear modulus $G$ | $\sim (\phi - \phi_J)^{1/2}$ | Vanishes continuously — no discontinuity |
| Bulk modulus $B$ | $\sim (\phi - \phi_J)^{\alpha-2}$ | Much stiffer than $G$ near the transition |

The vibrational density of states $D(\omega)$ develops an excess of low-frequency modes (boson peak) that extends to zero frequency at $\phi_J$ — a plateau $D(\omega) \sim \text{const}$ rather than the Debye prediction $D(\omega) \sim \omega^{d-1}$.

### Connection to Tissues

The tissue analogue of the J-point is the shape-driven rigidity transition at $p_0 = p_0^*$. At this point:

- The system is marginally stable (zero modes from Maxwell counting on the vertex network).
- The shear modulus vanishes: $G \sim (p_0^* - p_0)$ (linear, not square root — different universality class from particle jamming).
- Contact topology is critical: the neighbor exchange network is at a percolation threshold.

---

## 12. Origin of $p^* \approx 3.81$

### The Energy Functional

The vertex model energy per cell is:

$$e = \frac{K}{2}(A - A_0)^2 + \frac{\Gamma}{2}(P - P_0)^2$$

Non-dimensionalizing: $e = 0$ when $A = A_0$ and $P = P_0$, i.e., $p \equiv P/\sqrt{A} = p_0$.

### The Geometric Constraint

Cells in a confluent tissue must **tile the plane** — no gaps, no overlaps. This imposes a geometric constraint: not all combinations of $(A, P)$ are simultaneously achievable for every cell while maintaining a tiling.

The isoperimetric inequality gives a lower bound:

$$p = \frac{P}{\sqrt{A}} \geq 2\sqrt{\pi} \approx 3.545$$

with equality for circles (which can't tile the plane). For regular $n$-gons:

| Shape | $p_n$ |
|---|---|
| Triangle | 4.559 |
| Square | 4.000 |
| Pentagon | 3.812 |
| Hexagon | 3.722 |

A regular hexagonal tiling is the most efficient: $p_6 = 3.722$ is the lowest $p$ achievable in any monohedral tiling.

### Maxwell Counting on the Tiling

The transition arises from a **constraint counting argument** on the vertex network. In a vertex model with $N$ cells:

- **Degrees of freedom**: $2V$ (each vertex has 2D position coordinates), where $V$ is the number of vertices. For a 3-connected network: $V = N$, so $2N$ DOF.
- **Constraints from $A = A_0$**: $N$ constraints (one per cell).
- **Constraints from $P = P_0$**: these are only active if $p_0 < p_{\text{min}}$, i.e., when the target perimeter is geometrically frustrated. When $p_0 > p_{\text{min}}$, cells can achieve their target perimeter while tiling — the perimeter constraints are **slack** (automatically satisfied with room to spare).

When perimeter constraints are slack: $2N$ DOF, $N$ constraints → $N$ zero modes → **fluid** (cells can rearrange at zero energy cost).

When perimeter constraints are active: $2N$ DOF, $2N$ constraints → $0$ zero modes → **rigid** (every deformation costs energy).

The transition happens at $p_0 = p_0^*$ where the constraints switch from active to slack.

### Why Not Analytically Derived

The precise value $p_0^* \approx 3.81$ is **not analytically calculable** from the above argument alone. The Maxwell counting tells you *that* a transition exists, but the critical value depends on the specific geometry of the disordered tiling — which polygons appear, what the distribution of cell shapes is, how the network is connected. It has to be determined **numerically** by simulating the vertex model and finding where the shear modulus vanishes.

The coincidence with the regular pentagon ($p_5 = 3.812$) is suggestive — disordered Voronoi tilings have an average of ~5.16 neighbors, making the pentagon a "typical" cell — but this is not a proof. It's a numerical observation.

---

## 13. Quenched vs. Annealed Disorder

### Definitions

**Quenched disorder**: random variables that are **frozen on the dynamical timescale**. They don't fluctuate — they were set at the time of system preparation and remain fixed. Examples:

- Random impurities in a crystal lattice
- Random field strengths in a spin glass
- In cells: inherent differences in protein expression between cells (fixed cell-to-cell $v_{A,i}$, $R_i$, $\kappa_i$)

**Annealed disorder**: random variables that **fluctuate on the dynamical timescale** and are in thermal (or active) equilibrium. They can be averaged over simultaneously with the thermal degrees of freedom. Examples:

- Thermal fluctuations of particle positions
- In cells: stochastic fluctuations in active force direction

### Why the Distinction Matters

The free energy of a system with quenched disorder is:

$$F = -k_B T \overline{\ln Z}$$

where $\overline{(\cdot)}$ denotes the average over disorder realizations and $Z$ is the partition function in a single realization. You must compute $\ln Z$ for each realization **first**, then average.

For annealed disorder:

$$F = -k_B T \ln \overline{Z}$$

You average $Z$ directly. This is much easier mathematically but gives the wrong answer when disorder is truly quenched.

**Jensen's inequality** guarantees $\overline{\ln Z} \leq \ln \overline{Z}$, so the annealed approximation systematically overestimates the entropy (underestimates the free energy). The **replica trick** ($\overline{\ln Z} = \lim_{n \to 0} \frac{\overline{Z^n} - 1}{n}$) is the standard technique for handling $\overline{\ln Z}$.

### Current Simulation: No Quenched Disorder

In the current simulations, all cells are identical: same $R$, same $\kappa$, same $v_A$. There is no quenched disorder. The positional "disorder" (random initial placement) is effectively annealed in the fluid phase (cells rearrange and forget their initial positions) and quenched in the jammed phase (cells get stuck where they started).

This means the current simulations probe the **clean** limit. Adding quenched cell-to-cell variability would be a distinct and important extension — see [§15](#15-griffiths-effects-in-cell-systems).

---

## 14. Model A, Model B, and the Cell Model

### Hohenberg-Halperin Classification

In the Hohenberg-Halperin classification of dynamic universality classes:

**Model A** (non-conserved order parameter, Allen-Cahn):

$$\frac{\partial \phi}{\partial t} = -\Gamma \frac{\delta F}{\delta \phi} + \eta$$

The order parameter can change locally — think of a magnetic spin flipping. The total $\int \phi \, d\mathbf{r}$ is **not** conserved. Relaxation is local and fast.

**Model B** (conserved order parameter, Cahn-Hilliard):

$$\frac{\partial \phi}{\partial t} = M \nabla^2 \frac{\delta F}{\delta \phi} + \nabla \cdot \boldsymbol{\eta}$$

The order parameter is locally conserved — think of a binary alloy where atoms exchange positions but neither species is created or destroyed. The total $\int \phi \, d\mathbf{r}$ is conserved. Relaxation requires transport and is slow.

### Where the Cell Model Fits

The cell model equation is:

$$\frac{\partial \phi_i}{\partial t} = -\frac{M}{2}\frac{\delta F}{\delta \phi_i} + \mathbf{v}_i \cdot \nabla \phi_i$$

This looks like Model A (non-conserved relaxation) plus advection, but it's more complex:

| Feature | Model A | Model B | Cell Model |
|---|---|---|---|
| Conservation | No | Yes (exactly) | Soft (volume constraint $\mu$) |
| Transport | No | Diffusive | Advective ($\mathbf{v} \cdot \nabla\phi$) |
| Fields | 1 | 1 | $N$ (one per cell) |
| Coupling | Self-interaction | Self-interaction | Inter-field repulsion |
| Noise | Thermal | Thermal | Active (persistent) |

The volume constraint $\mu(V_0 - V_i)$ makes the dynamics **approximately** conserved — cells resist volume change — but not exactly conserved as in Model B. There's no $\nabla^2$ operator enforcing local conservation; instead, a global penalty discourages non-conservation.

The closest classification would be: **"Multi-component Model A with soft volume constraints and active advection."** It doesn't fit neatly into A or B — it contains elements of both, plus genuinely new features (multi-field coupling, active driving) that have no counterpart in the standard classification.

Standard phase field models (whether Model A or B) are **clean field theories** — no intrinsic disorder. Disorder is a physics input that you add.

---

## 15. Griffiths Effects in Cell Systems

> See also: [review_cell_jamming.md §11, Question 3](review_cell_jamming.md#11-open-questions) — "Role of disorder"

### What Griffiths Effects Are

In a system near a phase transition, **quenched (frozen) disorder** can create rare spatial regions that are locally in the "wrong" phase. At the Griffiths temperature $T_G$ (between the clean and disordered critical temperatures), these rare regions produce singular contributions to physical observables that decay as **stretched exponentials** or **power laws** rather than simple exponentials.

In a jammed tissue with cell-to-cell heterogeneity, Griffiths effects would manifest as:

**Rare locally-fluid regions** embedded in a globally jammed tissue. If cell $i$ has a higher-than-average motility $v_{A,i}$ (drawn from some quenched distribution), clusters of such cells form rare "hot spots" that are locally above the fluidization threshold, even though the tissue as a whole is jammed.

These rare fluid inclusions:
- Relax exponentially fast locally (they're fluid)
- Produce power-law tails in $Q(t)$ globally (rare events with exponentially distributed relaxation times → power-law sum)
- Break the simple stretched-exponential $\alpha$-relaxation into a more complex form
- Persist indefinitely if the disorder is truly quenched (fixed cell properties)

### Biological Connections

Griffiths-like physics maps naturally onto several biological phenomena:

**Tumour invasion.** "Leader cells" at the invasion front have enhanced motility (EMT, upregulated Rac1) while the bulk tumour remains jammed. This is precisely a rare-region effect: a small fraction of cells with anomalous properties nucleate local fluidization that propagates into an invasion stream. The prediction: invasion onset may be controlled by the tail of the $v_A$ distribution, not its mean.

**Wound healing.** After wounding, cells near the wound margin become motile while those far away remain quiescent. If there's heterogeneity in the response (some cells are "fast responders"), rare highly-motile cells could drive collective migration. Griffiths physics predicts that even a small fraction of fast responders dramatically changes the dynamics because they create local fluid regions that facilitate neighbour rearrangements.

**Development.** Embryonic tissues show coexisting solid-like and fluid-like domains (e.g., in zebrafish body axis elongation). If cells in the posterior have higher effective motility than anterior cells, the solid-fluid boundary emerges from a Griffiths-like mechanism: the local phase is set by the local $v_A$ relative to the local critical $v_A^c$.

**Asthma.** Park et al. (2015) showed that airway epithelial cells from asthmatic patients are partially unjammed compared to healthy controls. If the unjamming is spatially heterogeneous (patchwork of fluid and solid regions), this is a Griffiths phenomenon where the "disorder" comes from cell-to-cell variability in cytoskeletal properties.

### How to Study This in the Phase-Field Model

**Step 1: Introduce quenched heterogeneity.** Assign each cell a motility $v_{A,i}$ drawn from a distribution with mean $\bar{v}_A$ and width $\sigma$:

$$v_{A,i} \sim \mathcal{N}(\bar{v}_A, \sigma^2) \quad \text{or} \quad v_{A,i} \sim \text{LogNormal}(\mu, \sigma^2)$$

Log-normal might be more physical (motilities are positive, biological variability is often log-normal).

**Step 2: Compare clean vs. disordered.** Run the standard protocol at each $\bar{v}_A$:
- $\sigma = 0$: clean limit (current simulations)
- $\sigma = 0.1\bar{v}_A$: weak disorder
- $\sigma = 0.3\bar{v}_A$: moderate disorder
- $\sigma = \bar{v}_A$: strong disorder

**Step 3: Look for Griffiths signatures.**
- **Broadened transition**: the sharp crossover in $D(v_A)$ should smear out with increasing $\sigma$.
- **Power-law tails in $Q(t)$**: in the clean system, $Q(t)$ decays as stretched exponential. With quenched disorder, rare-region contributions add power-law tails: $Q(t) \sim t^{-\alpha}$ at long times.
- **Persistent spatial heterogeneity**: compute the local MSD in subregions. In a Griffiths regime, some regions should have permanently elevated mobility while others remain arrested. The distribution of local diffusion constants $P(D_{\text{local}})$ should develop a long tail.
- **Non-self-averaging**: different disorder realizations should give qualitatively different results (not just different noise realizations of the same average behavior).

### Feasibility Analysis: What It Takes to Observe Griffiths Effects

#### System Size

The current production system ($N = 288$ cells) is **marginal but usable** for the easier signatures, and too small for the subtler ones.

Griffiths effects come from *rare regions* — connected clusters of cells that happen to be locally in the "wrong" phase. The probability of finding a connected cluster of $n$ anomalous cells scales as $p^n$, where $p$ is the fraction with anomalous parameters. For moderate disorder ($\sigma/\bar{v}_A \sim 0.3$), roughly 15% of cells are in the tail ($> 1\sigma$ above mean). The probability of a connected cluster of 5 such cells is $\sim 0.15^5 \times \binom{z}{4} \sim 10^{-3}$ per cell, so in $N = 288$ you expect $\sim 0.3$ such clusters — barely one. At $N = 1152$, you get $\sim 1$–$2$ clusters, which is enough to see the effect in the tails of $Q(t)$.

However, for the **broadened transition** signature (the easiest to detect), $N = 288$ is sufficient — you're averaging over all cells, and even single anomalous cells shift the mean $D$.

| System size | Detectable signatures |
|---|---|
| $N = 288$ | Broadened $D(\bar{v}_A)$ transition, shifted $\chi_4$ peak, $P(D_{\text{local}})$ broadening |
| $N = 576$ | Power-law tails in $Q(t)$, spatial mobility maps, finite-size scaling check |
| $N = 1152$ | Non-self-averaging, rare-region statistics, definitive confirmation |

**Recommendation:** Start with $N = 288$ (existing infrastructure), plan a finite-size check at $N = 576$ for the key signals. If something interesting appears, go to $N = 1152$ to confirm it's not a finite-size artifact.

#### Simulation Time

This is the hard constraint. Griffiths effects produce power-law tails in $Q(t)$ at times $t \gg \tau_\alpha^{\text{clean}}$. To resolve those tails:

$$t_{\text{run}} \gtrsim 10 \times \tau_\alpha^{\text{clean}}$$

From the current production data at $\phi = 0.85$, estimating $\tau_\alpha \sim R^2/D$:

| $v_A$ | $D$ | $\tau_\alpha \sim R^2/D$ | Required $t_{\text{run}}$ | Timesteps ($\text{dt} = 0.01$) |
|---|---|---|---|---|
| 0.004 | $2.15 \times 10^{-3}$ | $\sim 1.1 \times 10^6$ | $\sim 10^7$ | $10^9$ (impractical) |
| 0.007 | $1.55 \times 10^{-2}$ | $\sim 1.5 \times 10^5$ | $\sim 1.5 \times 10^6$ | $1.5 \times 10^8$ |
| 0.010 | $4.47 \times 10^{-2}$ | $\sim 5.4 \times 10^4$ | $\sim 5 \times 10^5$ | $5 \times 10^7$ |

The slow end ($v_A = 0.004$) requires $\sim 10^9$ timesteps — roughly $10\times$ the current production runs. Impractical.

**The sweet spot is near the transition, not deep in the solid.** At $v_A = 0.006$–$0.010$, $\tau_\alpha$ is moderate, and the system is close to where the clean transition sharpens — this is precisely where disorder smears it out and Griffiths effects are strongest.

**Recommendation:** Focus on $v_A = 0.006$–$0.010$ (5 values), run to $t = 2 \times 10^6$ ($2 \times 10^8$ steps, $\sim 2\times$ current production). Use 30–50 replicas per $(\bar{v}_A, \sigma)$ pair — fewer than 100 is acceptable because the target is $Q(t)$ tails, not precise $D$ values.

#### Which Parameters to Disorder

This is the most consequential design decision. The parameters fall into a natural hierarchy:

**Motility $v_{A,i}$ — primary choice (random-field disorder).**

This is the most physically motivated and theoretically clean option:
- Real cells have variable crawling speeds (heterogeneous Rac1/RhoA expression).
- $v_A$ directly tunes the effective temperature $T_{\text{eff}} \sim v_A^2 \tau$, so disordering $v_A$ is equivalent to **random-temperature disorder** — the classic Griffiths setup.
- It couples directly to the glass transition: high-$v_A$ cells are locally fluid, low-$v_A$ cells are locally jammed.
- Easy to implement: draw $v_{A,i}$ from a distribution once at initialization and hold it fixed.

**Distribution choice:** Log-normal:

$$v_{A,i} \sim \text{LogNormal}(\mu, \sigma^2), \qquad \langle v_{A,i} \rangle = \bar{v}_A$$

Log-normal because: (i) $v_A > 0$ by construction, (ii) biological variability is typically log-normal, (iii) it naturally produces a heavy right tail (rare fast cells), which is what drives Griffiths effects.

**Persistence time $\tau_i$ — secondary choice (complementary to $v_A$).**

Since $T_{\text{eff}} \sim v_A^2 \tau$, disordering $\tau$ is complementary to disordering $v_A$ — both affect the effective temperature, but through different physical mechanisms (speed vs. directional memory). A separate set of runs with fixed $v_A$ and disordered $\tau_i$ would disentangle which aspect of activity matters.

**Radius $R_i$ — interesting but harder (geometric/polydispersity disorder).**

Variable cell size is a form of **polydispersity**, standard in glass physics (used to prevent crystallization in colloidal simulations). However:
- Changing $R_i$ modifies the target area $A_0 = \pi R_i^2$ and the local effective packing fraction.
- This conflates density disorder with Griffiths physics — a large cell in a sea of small ones creates a locally compressed region.
- It requires re-equilibrating from scratch for each disorder realization (or careful variable-size initialization).
- The volume constraint continuously adjusts, making this more annealed-like than truly quenched.

**Verdict:** This is a second paper. Do $v_A$ disorder first.

**Adhesion/repulsion $\kappa_i$ — most subtle (random-bond disorder).**

Variable $\kappa$ means some cell pairs repel/adhere differently. This is analogous to **random bond disorder** in spin glasses (Edwards-Anderson model) — physically distinct from the random-field ($v_A$) case:
- Random bond: the *interactions* are disordered → frustration → spin-glass-like behavior.
- Random field: the local *driving force* is disordered → Griffiths rare regions.

Random bond disorder in tissues is physically real (heterogeneous E-cadherin expression) and could produce genuine spin-glass-like tissue states. But it's theoretically much harder to analyze and experimentally harder to connect to biology.

**Verdict:** Not for the first study. Flag as a future direction — "random bond" tissue disorder is essentially unexplored.

**Summary of parameter hierarchy:**

| Parameter | Disorder type | Physical motivation | Difficulty | Priority |
|---|---|---|---|---|
| $v_{A,i}$ | Random field / random temperature | Variable crawling speed | Low (one array) | **Primary** |
| $\tau_i$ | Random persistence | Variable directional memory | Low | Secondary |
| $R_i$ | Polydispersity / geometric | Variable cell size | Medium (re-equilibrate) | Future work |
| $\kappa_i$ | Random bond | Variable adhesion/E-cadherin | High (pair interactions) | Future work |

#### Concrete Experimental Design

**Phase 1: Motility disorder**

| Parameter | Values |
|---|---|
| $\bar{v}_A$ | 0.006, 0.007, 0.008, 0.009, 0.010 |
| $\sigma/\bar{v}_A$ | 0, 0.1, 0.3, 0.5, 1.0 |
| $N$ | 288 |
| $\phi$ | 0.85 |
| $t_{\text{end}}$ | $2 \times 10^6$ |
| Replicas | 30 per $(\bar{v}_A, \sigma)$ pair |
| **Total runs** | $5 \times 5 \times 30 = 750$ |

**Code change required:** The current simulation uses a single global `v_A` in `SimParams`. The implementation needs a per-cell `v_A_i` array, drawn once at initialization from the log-normal distribution and held fixed for the entire run. The tumbling/reorientation dynamics stays unchanged — only the speed varies per cell.

#### What to Measure

1. **$D(\bar{v}_A, \sigma)$**: diffusion coefficient as a function of mean motility and disorder width. Does the transition broaden with $\sigma$?

2. **$Q(t)$ tails on log-log axes.** Clean system: stretched exponential (drops faster than any power law). Disordered system: power-law tail $Q(t) \sim t^{-\alpha(\sigma)}$ at long times. This is the most direct Griffiths signature.

3. **$P(D_{\text{local}})$**: the distribution of single-cell diffusion constants $D_i = \lim_{t \to \infty} \Delta_i^2(t)/4t$. Clean system: narrow, roughly Gaussian. Disordered system: bimodal or heavy-tailed, reflecting coexisting fast and slow populations.

4. **Spatial mobility maps**: colour each cell by its displacement over time $t^*$ (the timescale of maximum heterogeneity). Look for persistent spatial heterogeneity that is **frozen** rather than exchanging — truly quenched hot/cold spots set by the $v_{A,i}$ assignments.

5. **$\chi_4(t)$ peak vs. $\sigma$**: does the cooperative length grow faster or slower with disorder?

#### The Smoking Gun

The clearest Griffiths signature — distinguishable from ordinary dynamic heterogeneity — is the **spatial correlation between the quenched $v_{A,i}$ map and the long-time mobility map:**

$$C = \text{Corr}(v_{A,i}, \; \Delta_i^2(t^*))$$

In a **clean** system ($\sigma = 0$), all cells have the same $v_A$, and dynamic heterogeneity is **transient** — fast and slow cells exchange roles over time. $C$ is undefined or zero.

In a **Griffiths regime** ($\sigma > 0$, near the transition), the fast regions are **permanently co-located** with the high-$v_{A,i}$ clusters. The correlation $C$ should be significantly positive and grow with $\sigma$.

This observable unambiguously separates quenched Griffiths effects from annealed dynamic heterogeneity: in the latter, the correlation between a cell's intrinsic property and its measured mobility would average to zero over long times.

### Why This Is Novel

1. No prior study has examined Griffiths effects in any cell simulation, let alone a phase-field one.
2. The continuous-field representation naturally allows smooth spatial gradients in mobility, unlike vertex models where each cell is a discrete unit.
3. The non-confluent regime adds another dimension: in a non-confluent tissue, rare fluid regions can physically separate (open up gaps), which has no analogue in confluent models.
4. The biological connections are direct and testable experimentally.

---

## References

See [review_cell_jamming.md](review_cell_jamming.md) for the complete reference list. Additional references for this tutorial:

- Adam, G. & Gibbs, J. H. On the Temperature Dependence of Cooperative Relaxation Properties in Glass-Forming Liquids. *J. Chem. Phys.* **43**, 139 (1965).
- Götze, W. *Complex Dynamics of Glass-Forming Liquids: A Mode-Coupling Theory*. Oxford Univ. Press (2009).
- Griffiths, R. B. Nonanalytic Behavior Above the Critical Point in a Random Ising Ferromagnet. *Phys. Rev. Lett.* **23**, 17 (1969).
- Kauzmann, W. The Nature of the Glassy State and the Behavior of Liquids at Low Temperatures. *Chem. Rev.* **43**, 219 (1948).
- Kirkpatrick, T. R. & Thirumalai, D. Dynamics of the Structural Glass Transition and the p-Spin Interaction Spin-Glass Model. *Phys. Rev. Lett.* **58**, 2091 (1987).
- Mori, H. Transport, Collective Motion, and Brownian Motion. *Prog. Theor. Phys.* **33**, 423 (1965).
- Park, J.-A. et al. Unjamming and cell shape in the asthmatic airway epithelium. *Nat. Mater.* **14**, 1040 (2015).
- Vojta, T. Rare region effects at classical, quantum, and nonequilibrium phase transitions. *J. Phys. A* **39**, R143 (2006).
- Zwanzig, R. Memory Effects in Irreversible Thermodynamics. *Phys. Rev.* **124**, 983 (1961).
