# Rigidity, Glassy Dynamics, and Active Matter in Cellular Systems

**A review for the advanced graduate level — from first principles to open questions**

*Prepared February 2026*

---

## Contents

1. [Statistical Mechanics of Disordered Systems](#1-statistical-mechanics-of-disordered-systems)
2. [From Glasses to Jammed Packings](#2-from-glasses-to-jammed-packings)
3. [Cellular Matter as a Soft Material](#3-cellular-matter-as-a-soft-material)
4. [The Rigidity Transition in Confluent Tissues](#4-the-rigidity-transition-in-confluent-tissues)
5. [Active Driving and Motility-Induced Transitions](#5-active-driving-and-motility-induced-transitions)
6. [Dynamic Heterogeneity and Spatiotemporal Correlations](#6-dynamic-heterogeneity-and-spatiotemporal-correlations)
7. [Topology: Rearrangements, Neighbor Exchanges, and Plasticity](#7-topology-rearrangements-neighbor-exchanges-and-plasticity)
8. [Collective Migration and Symmetry Breaking](#8-collective-migration-and-symmetry-breaking)
9. [Turnover: Division, Death, and Non-Equilibrium Steady States](#9-turnover-division-death-and-non-equilibrium-steady-states)
10. [Three Dimensions and In Vivo Relevance](#10-three-dimensions-and-in-vivo-relevance)
11. [Open Questions](#11-open-questions)
12. [What Phase-Field Modelling Brings to the Table](#12-what-phase-field-modelling-brings-to-the-table)

---

## 1. Statistical Mechanics of Disordered Systems

### 1.1 The Glass Problem

A glass is a liquid that has stopped flowing. More precisely, it is a disordered system whose structural relaxation time $\tau_\alpha$ has exceeded the observation timescale, without any accompanying long-range order. This is the central distinction from crystallization: the static structure factor $S(q)$ of a glass and its parent liquid are nearly indistinguishable, yet their dynamical properties differ by orders of magnitude.

Consider $N$ particles in volume $V$ interacting via a pair potential $u(r)$. The partition function is

$$Z = \frac{1}{N! \Lambda^{dN}} \int e^{-\beta \sum_{i<j} u(r_{ij})} \, d\mathbf{r}^N$$

and the free energy landscape $F(\{\mathbf{r}_i\})$ has an exponentially large number of local minima (inherent structures). At high temperature (or low density), the system explores these minima ergodically. As temperature decreases or density increases, the system becomes trapped in a subset of minima — the ergodicity is broken dynamically, not thermodynamically. There is no latent heat, no symmetry breaking in the Landau sense.

### 1.2 Signatures of the Glass Transition

The canonical dynamical fingerprints are:

**Mean squared displacement.** Define

$$\Delta^2(t) = \frac{1}{N} \sum_{i=1}^{N} \langle |\mathbf{r}_i(t) - \mathbf{r}_i(0)|^2 \rangle.$$

In a normal liquid, $\Delta^2(t) \sim 2dDt$ at long times, where $D$ is the diffusion coefficient. In a glass-forming system approaching the transition, $\Delta^2(t)$ develops a **plateau** at intermediate times:

$$\Delta^2(t) \sim \begin{cases} v_{\rm th}^2 \, t^2 & t \ll \tau_\beta \quad \text{(ballistic)} \\ r_{\rm loc}^2 & \tau_\beta \ll t \ll \tau_\alpha \quad \text{(caging)} \\ 2dDt & t \gg \tau_\alpha \quad \text{(diffusive)} \end{cases}$$

The plateau height $r_{\rm loc}^2$ is the **localization length** — the typical cage size. The two timescales $\tau_\beta$ (rattling within the cage) and $\tau_\alpha$ (structural relaxation, i.e., cage escape) separate by orders of magnitude as the transition is approached.

**Self-intermediate scattering function.** Define

$$F_s(q,t) = \frac{1}{N} \left\langle \sum_i e^{i\mathbf{q} \cdot [\mathbf{r}_i(t) - \mathbf{r}_i(0)]} \right\rangle$$

evaluated at $q \sim 2\pi/\sigma$ (the first peak of $S(q)$). This exhibits two-step relaxation: a fast $\beta$-decay onto a plateau $f(q)$ (the non-ergodicity parameter), followed by slow $\alpha$-relaxation. Near the transition, $\tau_\alpha$ grows dramatically. The empirical Vogel-Fulcher-Tammann (VFT) form

$$\tau_\alpha \propto \exp\!\left(\frac{A}{T - T_0}\right)$$

often fits the data, suggesting a divergence at $T_0 > 0$. Whether this divergence is real or whether the growth is merely super-Arrhenius remains debated. Mode-coupling theory (MCT) predicts a power-law divergence $\tau_\alpha \sim (T - T_c)^{-\gamma}$ at a higher temperature $T_c > T_0$, which is interpreted as a dynamical crossover rather than a true singularity.

**Overlap function.** In cellular contexts, the self-overlap

$$Q(t) = \frac{1}{N} \sum_i \Theta\bigl(a - |\mathbf{r}_i(t) - \mathbf{r}_i(0)|\bigr)$$

with cutoff $a \sim 0.3\sigma$ is a convenient binary indicator: $Q \approx 1$ means the system remembers its initial configuration, $Q \to 0$ means complete decorrelation. Its relaxation time defines $\tau_\alpha$, and its fluctuations quantify dynamic heterogeneity (§6).

### 1.3 Mode-Coupling Theory

MCT provides a first-principles (if approximate) framework. Starting from the Mori-Zwanzig projection of the density-density correlator $\Phi(q,t) = F(q,t)/S(q)$, one derives a memory-function equation:

$$\ddot{\Phi}(q,t) + \Omega_q^2 \Phi(q,t) + \int_0^t m(q,t-t') \dot{\Phi}(q,t') \, dt' = 0$$

where $\Omega_q^2 = q^2 k_B T / m S(q)$. The MCT approximation closes the memory kernel as a quadratic functional of $\Phi$ itself:

$$m(q,t) = \sum_{\mathbf{k}+\mathbf{p}=\mathbf{q}} V(\mathbf{q};\mathbf{k},\mathbf{p}) \, \Phi(k,t) \, \Phi(p,t)$$

with vertices $V$ determined entirely by $S(q)$ (i.e., by the static structure). This self-consistent equation predicts a dynamical transition at a critical density or temperature: above $T_c$, $\Phi(q,t\to\infty) = 0$; below $T_c$, $\Phi(q,\infty) = f(q) > 0$, the non-ergodicity parameter. The power-law divergence $\tau_\alpha \sim |T - T_c|^{-\gamma}$ follows, with exponent $\gamma$ determined by the so-called exponent parameter $\lambda$ via

$$\frac{\Gamma(1-a)^2}{\Gamma(1-2a)} = \frac{\Gamma(1+b)^2}{\Gamma(1+2b)} = \lambda, \qquad \gamma = \frac{1}{2a} + \frac{1}{2b}.$$

MCT's key insight for the cellular context is that **the glass transition is encoded in the static structure**: given $S(q)$ — or equivalently, the pair correlation function $g(r)$ — one can predict whether the system flows or is arrested. This idea reappears when Bi et al. (2016) construct a structural order parameter for tissue arrest.

### 1.4 Jamming

Jamming is the athermal cousin of the glass transition. It was formalized by Liu and Nagel (1998), who proposed a unified **jamming phase diagram** with axes: temperature $T$, inverse density $1/\phi$, and applied stress $\Sigma$. The "J-point" at $(T=0, \phi = \phi_J, \Sigma=0)$ is a critical point for athermal packings.

For frictionless spheres in $d$ dimensions, jamming occurs at $\phi_J \approx 0.64$ (3D random close packing). At the J-point:

- The system is marginally stable: the number of contacts per particle $z$ equals the Maxwell isostatic value $z_c = 2d$.
- The excess contact number scales as $\delta z = z - z_c \sim (\phi - \phi_J)^{1/2}$.
- The shear modulus vanishes as $G \sim (\phi - \phi_J)^{1/2}$ (for harmonic repulsion).
- The vibrational density of states $D(\omega)$ develops an excess of low-frequency modes (boson peak) that extends to zero frequency at $\phi_J$.

The marginally stable nature of jammed packings is central: these systems live at the edge of mechanical stability, where any small perturbation can trigger a rearrangement cascade.

---

## 2. From Glasses to Jammed Packings

### 2.1 Unifying Framework

The connection between glasses and jammed packings is made through the **potential energy landscape (PEL)** picture. Each configuration $\{\mathbf{r}_i\}$ maps to a point on a $(dN)$-dimensional energy surface. At high $T$, the system hops freely between basins. The glass transition corresponds to the system becoming confined to a single metabasin.

For the athermal jamming transition, the relevant landscape is the enthalpy $H = E + P V$. At the J-point, the landscape is fractal-like: basins are shallow, and the barriers between them scale to zero. This marginal stability leads to the anomalous scaling exponents cited above.

### 2.2 Soft Glassy Rheology

Soft Glassy Rheology (SGR), due to Sollich et al. (1997), provides a mean-field description of amorphous solids under drive. The idea is that each mesoscopic region ("element") sits in a local energy well of depth $E$, and can hop to a new random well with a rate

$$k(E) = k_0 \, \exp\!\left(-\frac{E}{x}\right)$$

where $x$ plays the role of an effective (noise) temperature. Elements are also strained by macroscopic deformation; when the local strain energy exceeds the well depth, a yield event occurs.

The key prediction: for $x > 1$, the material flows (liquid); for $x < 1$, it is a glass. The glass transition at $x = 1$ in SGR maps naturally to the tissue context: Bi et al. (2016) showed that at low persistence times, the motility-driven tissue transition is consistent with SGR, with the effective temperature set by the active velocity:

$$x_{\text{eff}} \sim v_0^2.$$

This connection breaks down at high persistence, where the non-equilibrium character of active motion can no longer be captured by a single effective temperature.

### 2.3 Random First-Order Transition (RFOT) Theory

An alternative theoretical framework posits that the glass transition is a thermodynamic phase transition masked by disorder. RFOT theory, building on the $p$-spin mean-field models, predicts:

- A dynamical transition (MCT-like) at $T_d$, where free energy barriers appear.
- A thermodynamic (Kauzmann) transition at $T_K < T_d$, where the configurational entropy $S_c$ vanishes.
- The relaxation time is governed by the Adam-Gibbs relation: $\tau_\alpha \sim \exp(C / T S_c)$.
- Cooperative rearranging regions (CRRs) grow in size as $T_K$ is approached.

The mosaic/CRR picture of RFOT is evocative for tissues: one can ask whether dynamically heterogeneous regions in tissue (§6) correspond to the CRRs of RFOT, with a cooperative length $\xi$ that grows as the tissue approaches rigidity.

---

## 3. Cellular Matter as a Soft Material

### 3.1 Length and Time Scales

A single epithelial cell is $\sim 10\text{–}30 \,\mu\text{m}$ across, with a doubling time of $\sim 12\text{–}24$ hours, a migration speed of $\sim 0.1\text{–}1 \,\mu\text{m/min}$, and a persistence time of $\sim 10\text{–}60$ min. The actomyosin cortex generates contractile stresses of order $\sim 100\text{–}1000$ Pa with turnover times of $\sim 10$ s. Cadherin-based adhesion junctions transmit forces of order $\sim 1\text{–}10$ nN.

At the tissue scale ($\sim 10^2$–$10^4$ cells), the effective viscosity is huge: $\eta_{\text{eff}} \sim 10^5\text{–}10^7$ Pa$\cdot$s, comparable to pitch or glacier ice. The Reynolds number is

$$\text{Re} = \frac{\rho v L}{\eta} \sim 10^{-8}.$$

This violently overdamped regime means inertia is entirely negligible and the dynamics is governed by force balance:

$$\gamma \dot{\mathbf{r}}_i = \mathbf{F}_i$$

where $\gamma$ is an effective friction (cell-substrate or cell-cell). All the interesting dynamics is encoded in $\mathbf{F}_i$.

### 3.2 Forces on a Cell

The net force on cell $i$ decomposes conceptually as:

$$\mathbf{F}_i = \underbrace{-\nabla_i E_{\text{elastic}}}_{\text{shape restoration}} + \underbrace{\mathbf{F}_i^{\text{adhesion}}}_{\text{cell-cell}} + \underbrace{f_a \hat{\mathbf{n}}_i}_{\text{active}} + \underbrace{\boldsymbol{\xi}_i}_{\text{noise}}.$$

**(i) Elastic restoring force.** Each cell has a preferred area $A_0$ (set by osmotic pressure, cytoplasmic incompressibility) and a preferred perimeter $P_0$ (set by cortex contractility and adhesion). Deviations cost energy:

$$E_{\text{cell}} = \frac{K}{2}(A - A_0)^2 + \frac{\Gamma}{2}(P - P_0)^2.$$

The area modulus $K$ arises from the cell's resistance to volume change projected onto 2D. The perimeter modulus $\Gamma$ encodes the competition between cortical contractility ($\Gamma P^2$ favours small perimeter) and adhesion ($-\Lambda P$ favours large contact area, effectively increasing $P_0$). The target shape index $p_0 = P_0/\sqrt{A_0}$ sets the dimensionless target aspect ratio.

**(ii) Adhesion and contact mechanics.** Cells adhere through cadherin junctions. At contact, there is an attractive interaction (lower energy when cells touch) competing with cortical tension (which resists deformation). The net effect is a force that is attractive at moderate separations and repulsive at short range (steric/osmotic exclusion). In the vertex model, this enters through the line tension $\Lambda$. In phase-field models, it emerges from the coupling between phase fields at interfaces.

**(iii) Active self-propulsion.** Cells convert chemical energy (ATP) into directed motion via actin polymerization at the leading edge. The polarity $\hat{\mathbf{n}}_i$ specifies the direction of propulsion. The polarity dynamics is typically modelled as persistent random motion:

$$\dot{\theta}_i = \sqrt{2 D_r} \, \eta_i(t), \qquad \langle \eta_i(t) \eta_j(t') \rangle = \delta_{ij} \delta(t-t')$$

giving a persistence time $\tau_p = 1/D_r$ and a persistence length $\ell_p = v_0 \tau_p$. The Péclet number $\text{Pe} = v_0 \tau_p / a$ (with $a$ a cell diameter) controls whether active driving is a small perturbation ($\text{Pe} \ll 1$, effectively thermal) or a fundamentally non-equilibrium effect ($\text{Pe} \gg 1$).

An important distinction: in run-and-tumble dynamics, the polarity changes discontinuously at Poisson-distributed times with rate $1/\tau$. In active Brownian motion, the polarity diffuses continuously with $D_r = 1/\tau$. These are equivalent in the long-time diffusion limit but produce distinct short-time statistics — different distributions of persistent run lengths.

**(iv) Stochastic forces.** Even in the absence of active propulsion, cells experience fluctuating forces from the stochastic dynamics of the cytoskeleton, motor proteins, and signalling noise. These are typically modelled as Gaussian white noise with an effective temperature that may differ from the thermodynamic temperature.

### 3.3 Why Cells Are Not Particles

The particle analogy is useful but must be handled carefully. Cells differ from colloids or grains in several fundamental ways:

- **Deformability.** Cells change shape dramatically under confinement. The shape degree of freedom is not simply a higher-order correction; it can be the control parameter for the rigidity transition (§4).
- **Internal activity.** The forces are not derivable from a Hamiltonian. Energy is continuously injected at the scale of each cell, driving the system far from equilibrium. Detailed balance is broken.
- **Variable interactions.** Cells modulate their adhesion (cadherin expression), contractility (Rho/ROCK signalling), and motility (Rac1) in response to mechanical and chemical cues. The "pair potential" is not fixed.
- **Turnover.** Cells divide and die. The number $N$ is not conserved. This introduces an additional fluidization mechanism (§9).
- **Information.** Cells sense and respond. Mechanotransduction, chemotaxis, and contact inhibition of locomotion couple internal biochemistry to external mechanics.

---

## 4. The Rigidity Transition in Confluent Tissues

### 4.1 Shape-Driven Rigidity

The central discovery of Bi et al. (2015) is that a confluent tissue — where cells tile the plane with no gaps, so $\phi = 1$ identically — can undergo a rigidity transition controlled purely by cell shape.

The energy per cell is

$$e(A, P) = \frac{K}{2}(A - A_0)^2 + \frac{\Gamma}{2}(P - P_0)^2.$$

Non-dimensionalize: let $\tilde{A} = A/A_0$, $\tilde{P} = P/\sqrt{A_0}$, define $p_0 = P_0/\sqrt{A_0}$. The energy in the ground state ($\tilde{A} = 1$) is controlled by whether it is geometrically possible to achieve $\tilde{P} = p_0$ while tiling the plane.

The isoperimetric inequality bounds the perimeter of any simple shape:

$$p = \frac{P}{\sqrt{A}} \geq 2\sqrt{\pi} \approx 3.545$$

with equality for a circle. For a regular $n$-gon:

$$p_n = 2\sqrt{n \tan(\pi/n)}.$$

The critical observation: if $p_0 > p_0^*$, there exist cell shapes that simultaneously satisfy $A = A_0$ and $P = P_0$ while tiling the plane. Then $e = 0$ is achievable, and the energy landscape has a degenerate ground-state manifold — marginal modes exist, cells can rearrange at zero energy cost: the tissue is **fluid**.

If $p_0 < p_0^*$, geometry forbids $P = P_0$ and $A = A_0$ simultaneously in a tiling. The ground state is frustrated: $e > 0$, and any rearrangement costs additional energy. The tissue is **rigid**.

The critical value $p_0^* \approx 3.81$ corresponds to a regular pentagon ($p_5 = 3.812$). In the energy landscape language, at $p_0 = p_0^*$ the system is **marginally stable**: the number of zero modes matches the number of degrees of freedom (Maxwell counting), analogous to the isostatic condition at the J-point.

### 4.2 Linear Response and Shear Modulus

The rigidity side of the transition can be probed by the shear modulus $G$. In the solid phase,

$$G \sim (p_0^* - p_0)$$

which vanishes linearly at the transition (compared to $G \sim (\phi - \phi_J)^{1/2}$ for jammed spheres). The different exponent reflects the different nature of the constraint: geometry-based (shape index) vs. contact-based (packing fraction).

In finite-size systems, there is a crossover length $\ell^* \sim |p_0 - p_0^*|^{-\nu}$ below which the system cannot distinguish solid from fluid. The finite-size scaling theory is still under development.

### 4.3 Non-Confluent Tissues and the Density Axis

Most of the rigidity transition theory applies to confluent tissues ($\phi = 1$). But real tissues are often non-confluent: cells at lower densities do not tile the plane, and gaps exist between them. This is the domain of our phase-field simulations at $\phi = 0.85$ and $\phi = 0.89$.

In the non-confluent regime, two distinct rigidity-controlling mechanisms compete:

1. **Density-driven jamming.** As in particulate systems: increase $\phi$ until cells are geometrically constrained. There should exist a critical $\phi_J$ analogous to random close packing.
2. **Shape-driven rigidity.** As in the confluent theory: cell shape preferences make the tissue rigid even before RCP.

The interplay between these two mechanisms — and whether there is a single unified transition or two distinct ones — is an open question. The non-confluent regime has been almost entirely unexplored computationally, because vertex models assume confluence by construction.

### 4.4 Beyond Equilibrium: Mechanical Probes

Several rheological quantities characterize the material:

- **Bulk modulus** $B$: resistance to uniform compression. Connects to area elasticity $K$.
- **Shear modulus** $G$: resistance to shear. Vanishes at the fluid transition.
- **Loss modulus** $G''(\omega)$: energy dissipated per cycle at frequency $\omega$. In the glassy phase, $G''(\omega)/\omega \to \eta$ as $\omega \to 0$ gives the zero-shear viscosity.
- **Yield stress** $\sigma_y$: the stress at which the solid begins to flow. Exists only in the jammed/glassy phase.

In active tissues, one must be careful: these quantities depend on the measurement protocol because the system is not in equilibrium. The fluctuation-dissipation theorem does not hold, and the linear response can have both reversible and irreversible components.

---

## 5. Active Driving and Motility-Induced Transitions

### 5.1 The Active Tissue Phase Diagram

Bi et al. (2016) established a three-parameter phase diagram $(p_0, v_0, \tau)$ for the SPV model. The key finding is that motility opens a second route to fluidization: even for $p_0 < p_0^*$, sufficiently high $v_0$ drives the tissue from solid to fluid.

The physical picture: each cell's active force acts as a sustained perturbation that can push the system over energy barriers (which exist in the solid phase). The effective "noise temperature" due to motility is

$$T_{\text{eff}} \sim v_0^2 \tau$$

(dimensionally, energy $\sim$ force $\times$ displacement $\sim f_a \times v_0 \tau$, and $f_a = \gamma v_0$). When $T_{\text{eff}}$ exceeds the barrier heights $\Delta E \sim (p_0^* - p_0)$, cells can escape their cages.

### 5.2 Transition Line

The phase boundary between solid and fluid in the $(v_0, p_0)$ plane has been mapped numerically. For small $v_0$:

$$v_0^c(p_0) \sim (p_0^* - p_0)^a$$

with an exponent $a$ that depends on the persistence time. At high persistence ($\tau \to \infty$), cells push deterministically in one direction and the fluidization mechanism is qualitatively different from thermal activation. The transition line in $(v_0, 1/\tau)$ space separates the solid from the fluid, and the solid region shrinks as persistence increases.

### 5.3 Motility-Induced Phase Separation (MIPS)

In non-confluent active systems, a distinct phenomenon can occur: motility-induced phase separation (MIPS). Self-propelled particles accumulate where they slow down (due to crowding), creating a positive feedback: dense regions are slow → attract more particles → become denser. This can lead to coexistence of a dense, slowly-moving "liquid" phase and a dilute, fast-moving "gas" phase — purely from activity, without attractive interactions.

For cells at intermediate $\phi$, MIPS competes with adhesion-driven aggregation. The interplay between MIPS (kinetic clustering), adhesion (thermodynamic clustering), and the jamming transition within the dense phase creates a rich landscape of possible behaviors.

### 5.4 Effective Temperature and Breakdown of Equilibrium Mapping

The concept of effective temperature is seductive but limited. It works when:
- Persistence is short ($\tau \ll$ structural relaxation time)
- The system is close to equilibrium (activity is a small perturbation)

It fails when:
- Persistence is long (directed forcing, not diffusive)
- Activity produces non-Gaussian fluctuations
- Different observables give different $T_{\text{eff}}$ (violation of FDT)

The departure from effective-temperature descriptions is physically important: it signals genuinely non-equilibrium physics that cannot be reduced to equilibrium with rescaled parameters.

---

## 6. Dynamic Heterogeneity and Spatiotemporal Correlations

### 6.1 The Problem of Heterogeneous Dynamics

Glass-forming systems are not uniformly slow. Instead, they develop spatially heterogeneous dynamics: some regions relax quickly while others remain frozen, and these regions exchange roles over time. This dynamic heterogeneity is widely considered to be the key feature that any theory of the glass transition must capture.

### 6.2 Quantifying Heterogeneity

**Non-Gaussian parameter.** For a system with isotropic displacements, the distribution of displacements at time $t$ is Gaussian if the dynamics is homogeneous. Deviations are captured by

$$\alpha_2(t) = \frac{d}{d+2} \frac{\langle r^4(t) \rangle}{\langle r^2(t) \rangle^2} - 1$$

where $r(t) = |\mathbf{r}(t) - \mathbf{r}(0)|$ and $d$ is dimensionality. For a Gaussian, $\alpha_2 = 0$. The peak of $\alpha_2(t)$ occurs at $t = t^*$, the time of maximum heterogeneity — typically near the end of the caging plateau, when some cells begin to escape their cages while others remain trapped.

**Four-point susceptibility.** Define the fluctuating overlap density:

$$Q(t) = \frac{1}{N} \sum_i w_i(t), \qquad w_i(t) = \Theta\bigl(a - |\mathbf{r}_i(t) - \mathbf{r}_i(0)|\bigr).$$

The four-point susceptibility is the variance:

$$\chi_4(t) = N \left[ \langle Q(t)^2 \rangle - \langle Q(t) \rangle^2 \right].$$

This measures the volume of dynamically correlated regions. Near the glass transition:

$$\chi_4(t^*) \sim \xi_d^d$$

where $\xi_d$ is a dynamic correlation length. The growth of $\xi_d$ signals increasingly cooperative relaxation.

**Spatial correlations.** One can define a four-point correlation function:

$$G_4(\mathbf{r}, t) = \langle w_i(t) w_j(t) \rangle - \langle w_i(t) \rangle \langle w_j(t) \rangle$$

for pairs $i,j$ separated by $\mathbf{r}$. Its Fourier transform at $q \to 0$ gives $\chi_4$, and its decay defines $\xi_d$. In tissues, $\xi_d$ can be compared to the correlation length of velocity fields measured experimentally (e.g., via PIV), providing a direct observable link.

### 6.3 Heterogeneity in Active Tissues

In active tissues, dynamic heterogeneity has a richer structure because:

1. Activity itself can be heterogeneous (different cells have different $v_0$).
2. Active forces inject energy locally, creating spatial gradients in the effective temperature.
3. The persistence of active motion introduces **directional** correlations absent in thermal systems.

An open question is whether the dynamic length $\xi_d$ in active tissues follows the same scaling as in passive glasses, or whether activity modifies the universality class.

---

## 7. Topology: Rearrangements, Neighbor Exchanges, and Plasticity

### 7.1 T1 Transitions as Elementary Plastic Events

In confluent tissues, the elementary rearrangement — analogous to a shear transformation zone (STZ) in metallic glasses — is the T1 transition (neighbour exchange). Four cells sharing two junctions exchange neighbours when one junction shrinks to zero length and is replaced by a perpendicular junction.

The T1 rate $R_{T1}$ is a direct measure of fluidity. In the solid phase at zero activity, $R_{T1} = 0$. In the fluid phase, $R_{T1} > 0$ and increases with $v_0$ and $p_0$. The relationship between $R_{T1}$ and the macroscopic transport properties (viscosity, diffusion coefficient) is an analogue of the Stokes-Einstein relation — and its possible breakdown at the glass transition is interesting.

### 7.2 Energy Barriers to T1s

In the solid phase ($p_0 < p_0^*$), there exists a finite energy barrier $\Delta E_{T1}$ to perform a T1:

$$\Delta E_{T1} \sim (p_0^* - p_0)^\psi$$

with exponent $\psi$ that has been estimated numerically. This barrier must be overcome either by thermal fluctuations or by active driving. The Kramers rate for thermally activated escape gives

$$R_{T1} \sim \exp\!\left(-\frac{\Delta E_{T1}}{k_B T_{\text{eff}}}\right)$$

and the effective temperature picture (§5.4) predicts the transition when $T_{\text{eff}} \sim \Delta E_{T1}$.

### 7.3 Geometry of Rearrangements

Zhou and Milner (2015) introduced the concept of "T1-active" particles in hard-sphere systems: particles whose Voronoi neighbors can change within their available free volume. The fraction of T1-active particles $f_{T1}$ decreases with density and vanishes at random close packing. The percolation threshold of T1-*inactive* particles matches the colloidal glass transition at $\phi_g \approx 0.585$.

This framework provides a geometric predictor of dynamical arrest: when an unbroken network of immobile particles percolates through the system, the material behaves as a solid. The tissue analogue would be a percolating network of cells that cannot exchange neighbours. Whether such a percolation transition exists in non-confluent tissues, and how it relates to the shape index transition, is unknown.

### 7.4 Continuous vs. Discrete Rearrangements

In vertex models, T1s are instantaneous: one moment cells are neighbours, the next they are not. This discreteness is a modelling artifact. In real tissues, the contact area between two cells can decrease continuously to zero during a rearrangement. The process may take a finite time, involve intermediate configurations (rosettes, partial contacts), and even abort partway through.

Whether rearrangements are "effectively discrete" — i.e., whether the intermediate configurations are transient and the final state is always a completed exchange — or whether they form a genuinely continuous family of states is an open question that discrete models cannot address.

---

## 8. Collective Migration and Symmetry Breaking

### 8.1 Emergence of Collective Motion

Individual motile cells can produce coherent, collective migration patterns through purely local interactions. Experimentally, epithelial monolayers display:

- **Velocity correlation lengths** of $\sim 10$–$20$ cell diameters (Angelini et al., PNAS 2011).
- **Swirling flows** and vortices (Rossen et al., Nat. Comm. 2014).
- **Guided streams** in wound healing (Poujade et al., PNAS 2007).
- **Propagating waves** of mechanical stress (Serra-Picamal et al., Nat. Phys. 2012).

### 8.2 Alignment Mechanisms

The theoretical framework borrows from the Vicsek model of active matter. Cell polarity alignment can occur through:

- **Contact guidance**: cells align polarity with their elongation axis. This couples shape to motility.
- **Velocity alignment**: cell polarity relaxes toward its instantaneous velocity direction, a mechanism called "self-alignment" or "weathervane effect."
- **Neighbour alignment**: cells align with their neighbours' polarities, producing Vicsek-type ordering.

Giavazzi et al. (2018) showed that velocity alignment in confluent tissues produces three distinct phases:

1. **Solid**: no flow, no order.
2. **Liquid**: disordered flow, finite diffusion $D$.
3. **Flocking**: ordered directed motion, broken rotational symmetry, $D$ large.

Counterintuitively, alignment promotes solidification: by reducing the effective rotational noise, alignment helps cells maintain persistent cage configurations rather than stochastically escaping them.

### 8.3 Flocking as a Nonequilibrium Phase Transition

The flocking transition is in the universality class of the Toner-Tu equations (the "Navier-Stokes of active matter"):

$$\partial_t \mathbf{v} + \lambda_1 (\mathbf{v} \cdot \nabla)\mathbf{v} = (\alpha - \beta |\mathbf{v}|^2)\mathbf{v} - \nabla P + D_T \nabla^2 \mathbf{v} + \mathbf{f}$$

with $\mathbf{f}$ a stochastic forcing. The broken rotational symmetry $(\langle \mathbf{v} \rangle \neq 0)$ has remarkable consequences:

- Long-range order is possible in 2D (unlike equilibrium, where the Mermin-Wagner theorem forbids it).
- Giant number fluctuations: $\delta N \sim N^{1/2 + \alpha}$ with $\alpha > 0$.
- Anomalous sound modes and algebraically decaying correlations.

Whether these predictions hold in the dense, mechanically constrained setting of a tissue is an active research question.

---

## 9. Turnover: Division, Death, and Non-Equilibrium Steady States

### 9.1 Fluidization by Cell Cycling

Cell division and death are unique to biological systems. Each event locally rearranges cells, providing an athermal mechanism for structural relaxation even deep in the solid phase.

Czajkowski et al. (2019) modelled this in the vertex framework and found an additive relationship:

$$D_{\text{total}} \approx D_{\text{intrinsic}} + D_{\text{cycling}}$$

where $D_{\text{intrinsic}}$ is the diffusion coefficient from active motility alone and $D_{\text{cycling}} \sim k_d a^2$ with $k_d$ the division rate and $a$ the cell size.

The glass-like intermediate regime (caging plateau in MSD) persists when $k_d \ll 1/\tau_\alpha$: cells experience caging for times $\tau_\beta \ll t \ll 1/k_d$ but eventually diffuse due to turnover.

### 9.2 Homeostatic Pressure

A confined tissue with division and death reaches a homeostatic steady state where the mechanical pressure balances the growth pressure. The homeostatic pressure $P_h$ is the pressure at which division rate equals death rate. Tissues with $P > P_h$ are compressed (death dominates), and those with $P < P_h$ expand (division dominates). This sets a natural length scale for tissue mechanics.

### 9.3 Implications for Cancer

Tumour invasion involves the fluidization of a tissue boundary. During the epithelial-to-mesenchymal transition (EMT), cells lose adhesion (E-cadherin downregulation), increase motility, and gain invasive properties. In the language of the phase diagram, EMT corresponds to a trajectory from the solid phase toward the fluid phase — via increasing $p_0$ (reduced adhesion → higher shape index), increasing $v_0$ (enhanced motility), or both.

Understanding the quantitative jamming phase diagram of tissue is therefore directly relevant to predicting when and how tumours invade.

---

## 10. Three Dimensions and In Vivo Relevance

### 10.1 Why 3D Matters

Most theoretical and computational work has been in 2D, but real tissues are three-dimensional. The 3D context introduces:

- **Surface vs. bulk**: cells at the tissue surface experience an asymmetric environment (free boundary on one side, neighbours on the other). Surface tension, curvature, and confinement effects arise.
- **3D shape metrics**: the shape index $p = P/\sqrt{A}$ has no unique 3D generalization. Candidates include $s = S/V^{2/3}$ (isoperimetric ratio), asphericity, acylindricity, and the moment-of-inertia tensor eigenvalue ratios.
- **Contact topology**: in 3D, cells have faces (not edges) of contact. Rearrangements are topologically richer.
- **Mechanical coupling**: the 3D stress tensor has 6 independent components (vs. 3 in 2D). Normal and shear stresses along different axes decouple.

### 10.2 State of 3D Simulations

3D vertex models exist (Bi et al., unpublished; Merkel and Manning, 2018) but are computationally expensive and topologically complex (handling T1 analogues in 3D polyhedra is nontrivial). Most 3D studies have been confined to small systems ($\lesssim 64$ cells) and short times.

3D phase-field models have a significant computational advantage here: the field equations are dimensionally agnostic, and no topological surgery is required. The same code and physics that works in 2D works in 3D, with the main cost being the $O(N N_x N_y N_z)$ scaling of the grid.

### 10.3 3D-Specific Predictions

- **Surface fluidization**: cells at the tissue boundary may be unjammed (low coordination, free surface) while the bulk is jammed, creating a "solid core, liquid shell" structure.
- **Anisotropic rigidity**: the tissue may be rigid along one axis but fluid along another, depending on cell shape anisotropy.
- **3D critical shape**: the isoperimetric bound for a 3D volume is $s = S/V^{2/3} \geq (36\pi)^{1/3} \approx 4.836$ (sphere). The analogue of $p_0^* \approx 3.81$ in 3D is unknown but is expected to be related to tiling 3-space with cells of a critical isoperimetric ratio.

---

## 11. Open Questions

### Fundamental

1. **Nature of the rigidity transition.** Is it a true phase transition (in the thermodynamic sense, $N \to \infty$) or a crossover? What are the critical exponents? Does it have a field-theoretic description? Is there universality across different models (vertex, Voronoi, phase field, cellular Potts)?

2. **Effective temperature.** Under what conditions can the non-equilibrium tissue dynamics be mapped to an equilibrium system at effective temperature $T_{\text{eff}}$? Where does this mapping break down, and what is the correct non-equilibrium framework?

3. **Role of disorder.** Cell-to-cell variability (in size, stiffness, adhesion, motility) is ubiquitous. How does quenched vs. annealed disorder affect the transition? Are there Griffiths-like rare-region effects?

4. **3D physics.** What is the critical shape index in 3D? How do surface effects modify the bulk transition? Does the 2D phase diagram survive in 3D?

### Computational

5. **Non-confluent regime.** The entire region $0.5 < \phi < 1$ is essentially unexplored. What is the phase diagram in $(v_0, \phi)$? Is there a tricritical point where the density-driven and shape-driven transitions merge?

6. **Continuous rearrangements.** Are cell rearrangements fundamentally discrete (topology changes) or continuous? What is the distribution of rearrangement timescales? Can one define a meaningful "T1 rate" in a continuous model?

7. **Interface fluctuations.** Do cell boundary fluctuations diverge at the transition (critical fluctuations)? What is the fluctuation spectrum $S(k)$?

### Biological

8. **Prediction of invasion.** Can the jamming phase diagram predict the onset of tumour invasion? What observables are accessible experimentally?

9. **Mechanotransduction.** How do cells sense and respond to the tissue's mechanical state? Is there feedback between the jamming transition and gene expression?

10. **Wound healing.** How does tissue fluidization at a wound margin couple to collective migration? What sets the healing velocity?

---

## 12. What Phase-Field Modelling Brings to the Table

### 12.1 The Model

We represent each cell $i$ by a scalar field $\phi_i(\mathbf{r},t)$ on a regular grid, with $\phi_i \approx 1$ inside the cell and $\phi_i \approx 0$ outside, connected by a smooth interface of width $\lambda$. The dynamics is

$$\frac{\partial \phi_i}{\partial t} = -\frac{M}{2}\frac{\delta F}{\delta \phi_i} + \mathbf{v}_i \cdot \nabla \phi_i$$

where $M$ is a mobility coefficient, $\mathbf{v}_i = v_A \hat{\mathbf{n}}_i$ is the active velocity, and $F[\{\phi_i\}]$ is a free energy functional containing:

- A **Cahn-Hilliard bulk potential** $f(\phi) = \frac{60}{\lambda^2}\phi^2(1-\phi)^2$ that enforces $\phi \in \{0,1\}$ with a smooth interface.
- A **gradient energy** $\frac{\gamma}{2}|\nabla \phi_i|^2$ penalizing interfacial area.
- A **volume constraint** $\frac{\mu}{V_0}(V_0 - V_i)$ that conserves cell area/volume.
- A **repulsion term** $\frac{60\kappa}{\lambda^2}\phi_i \sum_{j \neq i} \phi_j^2$ preventing cell overlap.

The resulting equation is a nonlinear, coupled reaction-diffusion-advection PDE — one for each cell, coupled through the repulsion term. The GPU implementation solves this on a $1600 \times 1600$ grid (2D) or $N^3$ grid (3D) with $O(N_{\text{cells}} \cdot N_{\text{grid}})$ computational cost per timestep.

### 12.2 Fundamental Advantages

**No topological surgery.** Vertex models require explicit detection and handling of T1 transitions (edge collapse, edge insertion, vertex insertion/deletion). This introduces a discrete, artificial timescale and requires tolerance parameters. In phase fields, all rearrangements emerge naturally from the continuous evolution of the fields. There is no ambiguity about when a rearrangement "happens" — it is a smooth, finite-duration process.

**Non-confluent states.** Vertex and Voronoi models assume that cells tile the plane. The phase-field model naturally handles gaps ($\phi_{\text{total}} < 1$) and overlap ($\phi_{\text{total}} > 1$). This makes the entire non-confluent regime $\phi < 1$ directly accessible — a regime that is almost entirely unexplored theoretically.

**Curved interfaces.** Cell boundaries emerge from the competition between bulk and gradient energies. They are naturally curved, with curvature set by the local force balance. Interface roughness and shape fluctuations — quantities with no counterpart in vertex models — become computable observables.

**Cell individuality in a continuum framework.** Each cell is a continuous field that can deform, extend pseudopods, and wrap around neighbours. The shape is not constrained to be a polygon. This is critical for capturing the mechanics of highly motile, deformable cells.

### 12.3 What We've Measured: Preliminary Results

Our production runs at $\phi = 0.85$ with $N = 288$ cells across 10 motilities ($v_A \in [0.004, 0.013]$), each with 100 independent replicas, yield a diffusion coefficient that varies over a factor of $\sim 40$:

| $v_A$ | $D$ | $D_{\text{err}}$ |
|--------|-----|----------|
| 0.004 | $2.15 \times 10^{-3}$ | $5.6 \times 10^{-5}$ |
| 0.013 | $8.69 \times 10^{-2}$ | $1.2 \times 10^{-3}$ |

All $D > 0$ at $\phi = 0.85$, implying the system is in the fluid phase across the measured $v_A$ range. The power-law fit $D \sim v_A^\alpha$ yields the scaling exponent, which can be compared to the $D \sim v_0^2$ prediction for persistent random walkers ($\alpha = 2$) and to the more complex Bi et al. phase diagram predictions. Deviations from $\alpha = 2$ indicate the presence of caging effects even in the fluid phase.

These results are **already in novel territory**: no prior study has measured $D(v_A)$ in a non-confluent phase-field tissue model with this level of statistical control.

### 12.4 Scientific Opportunities Unique to Phase Fields

**I. Continuous rearrangement spectroscopy.** Because rearrangements are continuous, we can decompose them into a spectrum of timescales. Define the contact area between cells $i$ and $j$ as $C_{ij}(t) = \int \phi_i \phi_j \, d\mathbf{r}$. A "rearrangement event" is a trajectory in the $\{C_{ij}\}$ space. We can measure the duration, the energy dissipated, and the intermediate states of each event, and construct the distribution $P(\tau_{\text{rearrange}})$. If rearrangements are truly discrete, this distribution should be narrowly peaked at a single timescale. If they form a continuum, the distribution will be broad — possibly power-law — signalling critical-like behaviour.

**II. Interface fluctuation criticality.** Near the rigidity transition, the fluctuation spectrum of cell boundaries $S(k) = \langle|\hat{\phi}(k)|^2\rangle$ can be decomposed into interfacial modes. If the transition is continuous, boundary fluctuations should diverge with a characteristic exponent $S(k) \sim k^{-(d+2-\eta)}$, where $\eta$ is a critical exponent. This is unmeasurable in vertex models, which have no interface fluctuations.

**III. Overlap physics.** At high packing ($\phi > 1$ locally), cells in phase-field models overlap: $\phi_i \phi_j > 0$ in the same region. This mimics real cells compressing against each other, and the overlap integral $O = \int \phi_i \phi_j \, d\mathbf{r}$ provides a measure of compression stress. The existence of an **overlap-dominated regime** — where the physics is controlled by cell-cell compression rather than shape — is a phase-field-specific prediction with no vertex model counterpart.

**IV. The non-confluent phase diagram.** Our simulations at $\phi = 0.85$ and $\phi = 0.89$ are the first steps toward mapping the $(v_A, \phi)$ phase diagram outside the confluent limit. Key questions:
- Is there a critical $\phi^*(v_A)$ below which the system is always fluid?
- How does $\phi^*$ depend on adhesion strength?
- Does the non-confluent transition connect smoothly to the confluent $p_0^*$ transition as $\phi \to 1$?
- Is there MIPS-like phase separation at intermediate $\phi$ and high $v_A$?

**V. 3D without topological overhead.** Our GPU code runs the same Cahn-Hilliard dynamics in 3D with no additional topological machinery. 3D equilibration is currently running (Job 8333840, $t_{\text{end}} = 4000$), and 3D production will follow. This positions us to measure:
- Surface vs. bulk MSD (layer-resolved dynamics)
- The 3D shape index transition ($s^* = S/V^{2/3}$)
- 3D rearrangement geometry

without needing to solve the notoriously difficult problem of 3D T1 transitions in polyhedral meshes.

### 12.5 Strategic Research Directions

With the simulation performance improvements now underway, the following program becomes feasible:

1. **Complete the 2D $(v_A, \phi)$ phase diagram.** Two packing fractions ($\phi = 0.85, 0.89$) are in production. Extending to $\phi = 0.75, 0.80, 0.95$ would map the non-confluent transition. Even three or four points in $\phi$ give an interpolatable phase boundary.

2. **Compute $MSD(t)$ curves.** The aggregate $D$ is valuable, but the full $MSD(t)$ carries more information: the presence or absence of a caging plateau, the caging timescale $\tau_\beta$, and the $\beta$-relaxation exponent. These are computable from the existing trajectory data.

3. **Measure the shape index distribution $P(p)$.** Extract cell perimeters and areas from the phase field contours. The distribution $P(p)$ across different $v_A$ and $\phi$ values will reveal whether the $p_0^* \approx 3.81$ boundary survives in the non-confluent, continuous-interface regime.

4. **Rearrangement statistics.** Compute the contact area $C_{ij}(t)$ between neighbouring cell pairs, detect rearrangement events from topology changes in the contact graph, and measure the timescale distribution.

5. **Dynamic heterogeneity.** Compute $\alpha_2(t)$ and $\chi_4(t)$ from trajectories. Determine whether the cooperative length $\xi_d$ grows as the transition is approached, and whether it follows the same scaling as in particulate glasses.

6. **3D production.** After equilibration, launch 3D production runs across the same $v_A$ range. Measure layer-resolved MSD, $s = S/V^{2/3}$ distributions, and compare to 2D results.

This represents a set of measurements that, taken together, would amount to the first characterization of the glass/jamming transition in a non-confluent, continuous-interface model with full 2D and 3D coverage — a combination that no existing study provides.

---

## References

- Angelini, T. E. et al. Glass-like dynamics of collective cell migration. *PNAS* **108**, 4714 (2011).
- Barton, D. L. et al. Active Vertex Model for cell-resolution description of epithelial tissue mechanics. *PLOS Comput. Biol.* **13**, e1005569 (2017).
- Bi, D. et al. A density-independent rigidity transition in biological tissues. *Nat. Phys.* **11**, 1074 (2015).
- Bi, D. et al. Motility-driven glass and jamming transitions in biological tissues. *Phys. Rev. X* **6**, 021011 (2016).
- Czajkowski, M. et al. Glassy dynamics in models of confluent tissue with mitosis and apoptosis. *Soft Matter* **15**, 9133 (2019).
- Duclut, C. et al. Active T1 transitions in cellular networks. *Eur. Phys. J. E* **45**, 33 (2022).
- Farhadifar, R. et al. The influence of cell mechanics, cell-cell interactions, and proliferation on epithelial packing. *Curr. Biol.* **17**, 2095 (2007).
- Giavazzi, F. et al. Flocking transitions in confluent tissues. *Soft Matter* **14**, 3471 (2018).
- Götze, W. *Complex Dynamics of Glass-Forming Liquids: A Mode-Coupling Theory*. Oxford Univ. Press (2009).
- Liu, A. J. & Nagel, S. R. Nonlinear dynamics: Jamming is not just cool any more. *Nature* **396**, 21 (1998).
- Merkel, M. & Manning, M. L. A geometrically controlled rigidity transition in a model for confluent 3D tissues. *New J. Phys.* **20**, 022002 (2018).
- O'Hern, C. S. et al. Jamming at zero temperature and zero applied stress. *Phys. Rev. Lett.* **88**, 075507 (2002).
- Park, J.-A. et al. Unjamming and cell shape in the asthmatic airway epithelium. *Nat. Mater.* **14**, 1040 (2015).
- Serra-Picamal, X. et al. Mechanical waves during tissue expansion. *Nat. Phys.* **8**, 628 (2012).
- Sollich, P. et al. Rheology of soft glassy materials. *Phys. Rev. Lett.* **78**, 2020 (1997).
- Toner, J. & Tu, Y. Long-range order in a two-dimensional dynamical XY model. *Phys. Rev. Lett.* **75**, 4326 (1995).
- Zhou, Y. & Milner, S. T. T1 process and dynamics in glass-forming hard-sphere liquids. *Soft Matter* **11**, 2700 (2015).
