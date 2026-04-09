# Literature Review: Quenched Motility Disorder and the Glass Transition in Active Tissues

## 1. Scope and Motivation

Biological tissues exhibit glass-like dynamics: caging, dynamical heterogeneity, subdiffusive motion, and collective arrest [Angelini 2011, Garcia 2015, Atia 2018]. Unlike colloidal glasses, confluent tissues jam at constant density through a shape-controlled transition [Bi 2015]. Cell motility can unjam these tissues [Bi 2016], and the interplay between motility, shape, and crowding produces a rich nonequilibrium phase diagram.

A conspicuous biological feature absent from nearly all simulations is **cell-to-cell variability in motility**. Real tissues contain cells with heterogeneous crawling speeds arising from differences in cytoskeletal dynamics, signaling state, or metabolic activity. When this heterogeneity is frozen (quenched) rather than fluctuating, it creates persistent fast and slow subpopulations—a motility disorder analogous to quenched bond disorder in spin systems.

This review surveys the theoretical and experimental landscape relevant to studying **quenched motility disorder** in a confluent tissue modeled by the multiphase field method. We address: (i) what is known about active glasses and tissue jamming; (ii) what quenched disorder does in simpler active systems; (iii) whether Griffiths-type rare-region effects apply; (iv) what experiments constrain the phenomenology; and (v) what observable signatures our simulations should produce.

---

## 2. Active Glasses: Foundations

### 2.1 The active glass transition

Self-propelled particles at high density undergo a nonequilibrium glass transition sharing many features with thermal glasses: two-step relaxation of $F_s(q,t)$, caging plateaus in the MSD, non-Gaussian displacement distributions, and growing dynamical heterogeneity quantified by $\chi_4(t)$ [Berthier 2019 review; Ni et al. 2013; Berthier 2014; Flenner et al. 2016].

Activity generally **fluidizes**: self-propulsion shifts the glass transition to higher packing fractions. At low persistence time $\tau_p$, the active system maps approximately onto an effective-temperature passive system [Berthier & Kurchan 2013]. At high $\tau_p$, this mapping breaks down and qualitatively new dynamics emerges [Keta et al. 2022; Mandal & Sollich 2020], including intermittent relaxation between transient mechanical equilibria, multiple aging regimes, and velocity correlations with no equilibrium analog.

**Key references**: Berthier 2019 (JCP 150, 200901); Ni et al. 2013 (Nature Comms 4, 2704); Berthier 2014 (PRL 112, 220602); Flenner et al. 2016 (Soft Matter 12, 7136); Janssen 2019 (J. Phys. CM 31, 503002).

### 2.2 The cage length as unifying control parameter

Debets, de Wit & Janssen [2021, PRL 127, 278002] identified the ratio of persistence length $l_p = v_0 \tau_p$ to the cage length $l_c$ as the single parameter governing the departure from passive-like dynamics:

- $l_p < l_c$: enhanced but qualitatively passive-like relaxation. Stokes-Einstein relation $D\tau_\alpha$ holds. Fragility independent of particle softness.
- $l_p \approx l_c$: **optimal fluidization**. Most efficient scanning of the local cage → fastest relaxation.
- $l_p > l_c$: dynamics **slows down**. Stokes-Einstein breaks down. Fragility depends on particle softness. Qualitatively different from passive.

This nonmonotonic behavior is universal across ABP, AOUP, thermal, and athermal models [Debets et al. 2021]. Softer particles show the same qualitative picture with reduced peak height and shifted optimum [Debets & Janssen 2022, PhysRevRes 4, L042033].

**Implication for our system**: In a tissue with quenched motility disorder $v_{A,i} \sim \mathcal{N}(\bar{v}_A, \sigma^2)$, different cells sit at different points on the $l_p/l_c$ curve simultaneously. Some cells have $l_p < l_c$ (enhanced dynamics), some near the optimum, and some with $l_p \gg l_c$ (effectively arrested). The population-level heterogeneity in cage-scanning efficiency is the microscopic origin of the persistent dynamical heterogeneity we observe.

### 2.3 Mode-coupling theory and RFOT

Theoretical frameworks for active glasses include:
- **MCT for ABP**: Szamel [2019, JCP 150, 124901] derived mode-coupling predictions for steady-state dynamics.
- **MCT for thermal active particles**: Feng & Hou [2017, Soft Matter 13, 4464].
- **RFOT**: Nandi et al. [2018, PNAS 115, 7688] extended random first-order transition theory to active glasses, predicting an activity-dependent glass transition temperature.
- **MCT for Voronoi tissue**: Ruscher et al. [2021, J. Phys. CM 33, 064001] applied MCT to binary Voronoi fluid.

None of these theories address quenched motility disorder. Our simulations can provide benchmark data for extending these frameworks.

---

## 3. Quenched Disorder in Active Systems

### 3.1 Polydisperse size disorder (Keta, Jack & Berthier 2022)

The closest precedent to our study is Keta et al. [PRL 129, 048002], who introduced **size polydispersity** (20% uniform) into dense self-propelled particles. Key findings:

- Polydispersity stabilizes a homogeneous active liquid at arbitrary persistence (suppresses MIPS).
- The liquid undergoes a glass transition at all $\tau_p$ values.
- At low $\tau_p$: near-equilibrium, spatially heterogeneous dynamics.
- At high $\tau_p$: **intermittent** dynamics. Particles sit at minima of an effective potential $U_\text{eff}(\mathbf{r}, \mathbf{p}) = U(\mathbf{r}) - \sum_i \mathbf{p}_i \cdot \mathbf{r}_i$. As propulsion vectors evolve, mechanical equilibria destabilize → fast plastic rearrangements.
- $\alpha_2(t)$ (non-Gaussian parameter) peaks at timescales $\gg \tau_p$.
- $\chi_4(t,a)$ is large at all times, confirming spatially correlated dynamics.

**Distinction from our work**: Their disorder is in particle size (geometric/packing), which is standard in glass physics to prevent crystallization. Ours is in the *active drive* $v_A$—a fundamentally different type of quenched disorder that directly modulates the nonequilibrium character of each particle.

### 3.2 Chirality disorder (Debets, Löwen & Janssen 2023)

Debets et al. [PRL 130, 058201] studied **quenched chirality disorder** in chiral active Brownian particles. Their chiral glass shows:

- Nonmonotonic dynamics (regime I: harmonic-trap-like behavior)
- **Reentrant** dynamics (regime II: "hammering" mechanism — spinning particles systematically collide with the same neighbor, remodeling the cage)
- Collective swirl regime (regime III: velocity alignment for slow spinning)
- Key result: **disorder in an active degree of freedom produces qualitatively different physics from simple activity enhancement**. Random chirality ≠ enhanced temperature.

**Relevance**: Establishes the principle that quenched heterogeneity in active parameters is not simply "more noise"—it creates qualitatively new mechanisms for cage escape and collective dynamics.

### 3.3 What's missing: motility disorder in tissue models

No prior study examines quenched motility disorder ($v_A$ heterogeneity) in a tissue model, whether vertex, Voronoi, or phase field. The closest candidate is Sadhukhan et al. [2024, bioRxiv 2024.03.14.584932], who study how motility drives glassy dynamics in a self-propelled Voronoi model—but they examine uniform motility, not quenched disorder.

Our study fills this gap by:
1. Introducing a log-normal distribution of per-cell motilities $v_{A,i}$, parameterized by a target mean $\bar{v}_A$ and standard deviation $\sigma$ (with the log-normal ensuring $v_{A,i} > 0$), frozen for the duration of each simulation.
2. Using a multiphase field model that captures continuous cell deformation, curved boundaries, and realistic mechanics.
3. Systematically varying $\sigma$ to map out disorder-dependent glass observables.

**Note on the distribution:** The implementation uses a log-normal distribution (not a truncated normal) to ensure strictly positive motilities without artificial truncation. The log-normal parameters $\mu_{\ln}, \sigma_{\ln}$ are computed from the desired mean $\bar{v}_A$ and std $\sigma$ via $\sigma_{\ln} = \sqrt{\ln(1 + (\sigma/\bar{v}_A)^2)}$ and $\mu_{\ln} = \ln(\bar{v}_A) - \sigma_{\ln}^2/2$. For $\sigma/\bar{v}_A \lesssim 0.75$, the log-normal is nearly symmetric and closely approximates the corresponding Gaussian. At $\sigma/\bar{v}_A = 1$, the distribution becomes noticeably right-skewed, with a heavier tail of fast cells than a Gaussian would produce.

---

## 4. Griffiths Physics: When Does It Apply?

### 4.1 Classical formulation

Griffiths [1969, PRL 23, 17] showed that dilute Ising ferromagnets have essential singularities in the paramagnetic phase near $T_c$. Rare spatial regions of pure material can locally order, producing non-analytic contributions to thermodynamic quantities.

Vojta [2006, J. Phys. A 39, R143] classified rare-region effects:
- **Class A** ($d_{RR} < d_c^-$): Conventional Griffiths phase → stretched exponential ($\beta < 1$) relaxation.
- **Class B** ($d_{RR} = d_c^-$): Infinite-randomness fixed point → activated scaling.
- **Class C** ($d_{RR} > d_c^-$): Smeared transition.

Requirements: (i) quenched spatial disorder; (ii) rare regions with dimensionality comparable to $d_c^-$; (iii) static heterogeneity.

### 4.2 Griffiths phases in complex networks

Moretti & Muñoz [2013, Nature Comms 4, 2521] extended Griffiths physics to hierarchical modular networks, showing that structural heterogeneity stretches criticality from a point to a broad region. The disorder is topological/structural, not dynamical.

### 4.3 Assessment for our system

Our quenched motility disorder satisfies requirement (iii) — the heterogeneity is static. However:

- The disorder is in an **active drive parameter** ($v_{A,i}$), not a coupling constant or spatial coordinate.
- There is no spatial correlation built into the disorder assignment; cells with similar $v_A$ are not necessarily neighbors.
- The system is 2D, and rare-region effects require rare large patches — the probability of a large connected region of uniformly slow cells decreases exponentially with patch size.

**Prediction**: Classical Griffiths physics (Class A: stretched exponential with $\beta < 1$) is *unlikely* to be the dominant signature, because:
1. Cells are mobile (even slow cells diffuse), so "rare regions" are not truly static spatial patches.
2. The active nature of the system provides an additional escape mechanism (cage scanning by fast neighbors) absent in equilibrium Griffiths systems.

Instead, the signature should be closer to what Keta et al. and Debets et al. observed: **persistent dynamical heterogeneity** with two-population dynamics (fast vs. slow cells), potentially with enhanced Stokes-Einstein violation.

**Update (Feb 2026):** This prediction was partially confirmed by simulation data. The $\chi_4$ prediction was wrong — disorder *decreases* $\chi_4$ rather than increasing it, because fast cells act as active stirrers that break up cooperative caging (see Sec. 9). The persistent dynamical heterogeneity prediction ($\alpha_2$ increase, two-population dynamics) was correct.

---

## 5. Compressed vs. Stretched Exponential Relaxation

### 5.1 What determines β in $F_s(q^*,t) \sim \exp(-(t/\tau_\alpha)^\beta)$?

| $\beta$ regime | Physical origin | Systems |
|---|---|---|
| $\beta < 1$ (stretched) | Heterogeneous relaxation; superposition of exponentials with different rates | Supercooled liquids, structural glasses, Griffiths phases |
| $\beta = 1$ (simple) | Single relaxation timescale | Dilute systems |
| $\beta > 1$ (compressed) | **Internal driving** or stress; ballistic-like relaxation; strain-coupled rearrangements | Sheared glasses, growing tissues, driven colloidal suspensions |

### 5.2 Compressed exponentials in driven active matter

Tjhung & Berthier [2020, PhysRevRes 2, 043334] showed that **growing** dense active matter (particle division + steric repulsion) displays compressed exponential decay ($\beta > 1$) in $F_s$. The global expansion plays a role analogous to shear rate in driven glasses. This produces:
- Aging (dynamics slows with waiting time $t_w$)
- Crossover from superdiffusive (short time) to subdiffusive (long time)
- Dynamic heterogeneity: coexistence of slow (caged) and fast (plastic event) particles

### 5.3 Interpretation of our N=288 results

Our preliminary N=288 data showed $\beta \approx 1.15$–$1.28$ (compressed). This is consistent with **driven-glass phenomenology**, not Griffiths physics. Possible sources of internal driving:

1. **Finite-size effects**: With N=288 cells in a periodic domain of L=1562, the system has only ~17 cells per linear dimension. Persistent cell motion creates effective internal strain as cells push against their periodic images.
2. **Polarity rotation**: Continuous rotation of self-propulsion direction at rate $1/\tau$ creates effective stirring.
3. **Collective velocity correlations**: Active systems develop long-range velocity correlations absent in equilibrium [Keta et al. 2022], creating strain-like cooperative displacements.

**Prediction for finite-size scaling**: $\beta$ should **decrease toward 1 or below** at larger $N$, because:
- Finite-size driving (periodic boundary strain) weakens as $L$ grows.
- At N=1152, 4608, the system is less constrained → more heterogeneous, less driven → $\beta$ should drop.
- If $\beta$ drops below 1 at large $N$, this would indicate genuine stretched exponential relaxation consistent with heterogeneous dynamics or near-Griffiths phenomenology.

---

## 6. Tissue-Specific Glass Transition

### 6.1 Shape-controlled jamming (vertex model framework)

**Bi et al. 2015** [Nature Physics 11, 1074 — DOI: 10.1038/nphys3471]: Confluent tissues jam at constant density via a shape-controlled transition at $p_0^* \approx 3.81$ (regular pentagon). This is fundamentally different from colloidal jamming—no density change is needed.

**Bi et al. 2016** [PRX 6, 021011 — DOI: 10.1103/PhysRevX.6.021011]: Added self-propulsion via the Self-Propelled Voronoi model. Motility unjams tissues: in the $(p_0, v_0)$ phase diagram, high motility fluidizes even when $p_0 < 3.81$. The structural order parameter (mean cell shape) predicts the transition.

### 6.2 Cell softness and anomalous dynamics

**Li, Das & Bi 2021** [PRE 103, 022607 — DOI: 10.1103/PhysRevE.103.022607]: The mechanical softness of cells (controlled by $p_0$) qualitatively changes the glassy dynamics:
- **Stiff cells** ($p_0 = 3.0$): Conventional glass — MSD plateau, two-step ISF, super-Arrhenius $\tau_\alpha(T)$.
- **Soft cells** ($p_0 = 3.81$): Anomalous glass — extended subdiffusion (MSD $\sim t^\alpha$, $0 < \alpha < 1$), sub-Arrhenius behavior, fractal-like energy landscape. Many cell edges become short → T1 transitions with nearly zero energy barrier → free diffusion along specific phase-space directions.

**Relevance**: Our phase field cells are inherently deformable (soft). We should expect anomalous rather than conventional glassy dynamics as the baseline, with motility disorder adding another layer of heterogeneity on top.

### 6.3 Controlled rearrangements in tissues

**Das, Sastry & Bi 2021** [PRX 11, 041037 — DOI: 10.1103/PhysRevX.11.041037]: T1 transitions (neighbor exchanges) in tissues are rate-limited—they cannot occur instantaneously due to finite time required for adhesion molecule turnover. This rate limitation creates intermittent flow and glassy dynamics distinct from standard particle-based glass formers.

**Relevance**: Phase field models naturally capture continuous (not discrete) rearrangements with finite duration, unlike vertex models where T1s are instantaneous.

### 6.4 Unjamming vs. EMT

**Mitchel et al. 2020** [Nature Comms 11, 5053 — DOI: 10.1038/s41467-020-18841-x]: Unjamming transition is distinct from epithelial-to-mesenchymal transition (EMT). Both increase motility but through different mechanisms. Our disorder-induced fluidization is a mechanical effect, not biological reprogramming.

---

## 7. Experimental Context

### 7.1 Glass-like dynamics in cell monolayers

**Angelini et al. 2011** [PNAS 108, 4714 — DOI: 10.1073/pnas.1010059108]: MDCK wound-healing assay shows caging, dynamical heterogeneities, non-Gaussian displacements. Established experimentally that tissue dynamics are glassy.

**Garcia et al. 2015** [PNAS 112, 15314 — DOI: 10.1073/pnas.1510973112]: Active jamming in cell monolayers. Measured MSD showing crossover from ballistic to caged to diffusive. Velocity correlations growing with density.

**What our simulation can compare to**: MSD shape (subdiffusive exponent), timescale of caging plateau, non-Gaussian parameter $\alpha_2(t)$ magnitude and peak time.

### 7.2 Geometric constraints during jamming

**Atia et al. 2018** [Nature Physics 14, 613 — DOI: 10.1038/s41567-018-0089-9]: Measured cell shape statistics during progressive jamming of HBEC monolayers. Found:
- Cell aspect ratio and its variability are mutually constrained by a purely geometric relationship.
- This relationship is universal across non-asthmatic donors, asthmatic donors, and *Drosophila* ventral furrow.
- Shape variability collapses to a common distributional family governed by the approach to jamming.
- More jammed → less elongated, less variable.

**What our simulation should measure**: Distribution of cell aspect ratios $P(AR)$ as a function of $\sigma$. If motility disorder unjams the tissue, we expect broader $P(AR)$ (more shape variability) with increasing $\sigma$, and the mean $AR$ should increase if cells become more elongated during unjamming.

### 7.3 Cell shape as tissue fluidity indicator

**Grosser et al. 2021** [PRX 11, 011033 — DOI: 10.1103/PhysRevX.11.011033]: Cell and nucleus shape correlate with tissue fluidity in carcinoma. Links shape index measurements to metastatic potential.

**Park et al. 2015** [Nature Materials 14, 1040 — DOI: 10.1038/nmat4357]: Asthmatic airway epithelium shows altered jamming—cells remain more unjammed (fluid-like) with higher shape index, consistent with vertex model predictions.

### 7.4 Motility heterogeneity in real tissues

Real tissues exhibit substantial cell-to-cell motility variation:
- **Leader–follower dynamics** in wound healing: a subset of cells at the wound edge are highly motile, pulling along less motile followers [Vishwakarma et al. 2018, Nature Comms 9, 3246 — DOI: 10.1038/s41467-018-05927-6].
- **Heterogeneous proliferation**: Cells in different cell-cycle phases have different mechanical properties and motility [Bocanegra-Moreno et al. 2023, Nature Physics 19, 1767 — DOI: 10.1038/s41567-023-02213-3].
- **Cancer biology**: Tumor cells exhibit a wide distribution of migration speeds, with fast cells driving invasion [Grosser et al. 2021].

**Our model captures this**: Quenched $v_{A,i}$ heterogeneity is a minimal representation of the persistent cell-to-cell motility variation observed in real epithelia. The question is whether this heterogeneity merely smears the transition or produces qualitatively new collective behavior.

---

## 8. The Phase Field Model: Advantages for This Study

### 8.1 What PFM captures that vertex/Voronoi cannot

| Feature | Vertex/Voronoi | Phase Field |
|---|---|---|
| Cell boundaries | Straight segments | Curved, diffuse interfaces |
| Rearrangements | Discrete, instantaneous T1 | Continuous, finite-duration interface evolution |
| Cell shapes | Polygons | Arbitrary (concave, protrusive, highly irregular) |
| Non-confluent states | Forbidden | Naturally handled via $\phi < 1$ |
| Cell overlap/compression | Forbidden | Allowed, penalized energetically via repulsion $\kappa$ |
| Adhesion | Single line tension per edge | Gradient coupling ($J \int \nabla\phi_i \cdot \nabla\phi_j$) or quartic repulsion |
| Interface mechanics | Absent | Emergent from gradient energy |

### 8.2 Specific advantages for the disorder study

1. **Continuous shape deformation**: As cells with different $v_A$ push against each other, their shapes respond continuously. The phase field captures how a fast cell deforms a slow neighbor's boundary—information lost in vertex models.

2. **Natural coupling between motility and shape**: A fast cell stretches along its polarity direction; a caged slow cell remains compact. This motility-shape coupling emerges self-consistently in PFM.

3. **No artificial T1 threshold**: In vertex models, rearrangements require edge lengths to cross a numerical threshold. In PFM, contact areas evolve continuously, allowing gradual neighbor exchanges without arbitrary cutoffs.

### 8.3 Model parameters and the physical regime

Our simulation uses: N cells, each with a phase field $\phi_i(\mathbf{r},t)$ evolving via:
$$\frac{\partial \phi_i}{\partial t} = -\mathbf{v}_i \cdot \nabla \phi_i - \frac{1}{2}\left(-2\gamma\nabla^2\phi_i + f'(\phi_i) + \mu_V \frac{\delta E_V}{\delta \phi_i} + \kappa_{rep}\frac{\delta E_{rep}}{\delta \phi_i}\right)$$

with $\mathbf{v}_i = v_{A,i} \hat{p}_i$ where $v_{A,i}$ is drawn from a log-normal distribution with target mean $\bar{v}_A$ and target std $\sigma$ (see Sec. 3.3 for parameterization details), drawn once and frozen. Key parameters: $R = 49$, $\lambda = 7$, $\kappa = 10$, $\mu = 1$, $\tau = 10000$, $dt = 0.02$.

---

## 9. Predicted Observables and Expected Signatures

### 9.1 Static structure factor $S(q)$

**What to measure**: Radially averaged structure factor from cell centroid positions at each time frame.

**Expected behavior**: $S(q)$ should show a peak at $q^* \approx 2\pi / d_{nn}$ where $d_{nn}$ is the mean nearest-neighbor distance. As $\sigma$ increases:
- Peak height may decrease (less structural order → more fluid-like).
- Peak position should be insensitive to $\sigma$ (interparticle spacing set by density, not motility).
- Low-$q$ behavior may change: if disorder creates mesoscale density fluctuations, $S(q \to 0)$ could increase.

### 9.2 Self-intermediate scattering function $F_s(q^*,t)$

**What to measure**: $F_s(q^*,t) = \langle e^{i q^* \cdot [\mathbf{r}_j(t) - \mathbf{r}_j(0)]} \rangle$ at the structure factor peak $q^*$.

**Expected behavior at large $N$ and sufficient $t$**:
- $F_s$ should show a two-step decay: fast ($\beta$-relaxation) followed by slow ($\alpha$-relaxation).
- Fit to stretched/compressed exponential: $F_s \sim \exp(-(t/\tau_\alpha)^\beta)$.
- **Key diagnostic**: $\beta < 1$ → heterogeneous relaxation (Griffiths-like or standard glass). $\beta > 1$ → driven glass (finite-size artifact).
- **Prediction**: At N=288, $\beta > 1$ (driven by finite-size effects). At N=1152 and N=4608, $\beta$ should decrease, ideally below 1.
- **$\sigma$-dependence**: Increasing $\sigma$ should increase the spread of relaxation rates → lower $\beta$ (more stretched).

### 9.3 Relaxation time $\tau_\alpha$ and its $\sigma$-dependence

**What to measure**: $\tau_\alpha$ defined by $F_s(q^*, \tau_\alpha) = e^{-1}$.

**Expected behavior**:
- If disorder fluidizes: $\tau_\alpha$ decreases with increasing $\sigma$ (fast cells help slow cells escape cages).
- If disorder arrests: $\tau_\alpha$ increases with increasing $\sigma$ (slow cells create bottlenecks).
- The Debets cage-length picture predicts **nonmonotonic** behavior: moderate $\sigma$ fluidizes (some cells reach optimal $l_p \sim l_c$), but extreme $\sigma$ can arrest (many cells pushed to $l_p \gg l_c$ where dynamics slows).

### 9.4 Mean squared displacement

**What to measure**: $\langle \Delta r^2(t) \rangle = \langle |\mathbf{r}_i(t) - \mathbf{r}_i(0)|^2 \rangle$.

**Expected regimes**:
1. Ballistic: $\Delta r^2 \sim t^2$ (short times, $t \ll \tau$)
2. Subdiffusive plateau (caging): $\Delta r^2 \sim t^\alpha$, $\alpha < 1$
3. Diffusive: $\Delta r^2 \sim t$ (long times)

**Key measure — cage length**: $l_c = \sqrt{\Delta r^2(t_c)}$ where $t_c$ is the inflection point (minimum of log-slope $\Delta(t)$). Compare $l_c$ to the persistence length $l_p = v_A \tau$ to test the Debets framework.

**$\sigma$-dependence**: Increasing $\sigma$ should broaden the caging-to-diffusive crossover (heterogeneous escape times) and increase the slope $\Delta(t)$ at intermediate times.

### 9.5 Diffusion coefficient $D$ and Stokes-Einstein violation $D\tau_\alpha$

**What to measure**: $D = \lim_{t\to\infty} \langle \Delta r^2(t) \rangle / 4t$ (ensemble average). Also per-cell $D_i$ from individual MSDs.

**Expected behavior**:
- $D$ should increase with $\sigma$ (fast cells contribute disproportionately to diffusion).
- Per-cell $D_i$ distribution should broaden with $\sigma$, potentially bimodal (fast vs. slow populations).
- $D\tau_\alpha$ should increase with $\sigma$ → **Stokes-Einstein violation**. This is the hallmark signature of dynamical heterogeneity [Debets & Janssen 2022]. Larger $D\tau_\alpha$ means some cells are very mobile while the collective structure relaxes slowly.

### 9.6 Non-Gaussian parameter $\alpha_2(t)$

**What to measure**: $\alpha_2(t) = \frac{\langle \Delta r^4(t) \rangle}{2 \langle \Delta r^2(t) \rangle^2} - 1$.

**Expected behavior**: $\alpha_2 > 0$ reflects non-Gaussian tails in displacement distribution. Peak of $\alpha_2(t)$ marks the timescale of maximum dynamical heterogeneity.

- As $\sigma$ increases, peak $\alpha_2$ should increase (more heterogeneous dynamics).
- Peak time should shift to **later** times with increasing $\sigma$ (slow cells create a tail of late escapers).

### 9.7 van Hove self-correlation function $G_s(\Delta x, t)$

**What to measure**: Distribution of single-particle displacements at lag times around $\tau_\alpha$.

**Expected behavior**: For homogeneous ($\sigma = 0$) systems, $G_s$ should be close to Gaussian. For $\sigma > 0$:
- Develop **exponential tails** (indicative of a mobile subpopulation).
- At large $\sigma$, potentially **bimodal**: a narrow Gaussian (slow cells) plus a broad distribution (fast cells).
- This is a direct visual indicator of two-population dynamics.

### 9.8 Dynamic susceptibility $\chi_4(t)$

**What to measure**: $\chi_4(t) = N[\langle Q(t)^2 \rangle - \langle Q(t) \rangle^2]$ where $Q(t) = \frac{1}{N}\sum_i \theta(a - |\mathbf{r}_i(t) - \mathbf{r}_i(0)|)$.

**Expected behavior**: $\chi_4$ quantifies the spatial extent of cooperatively rearranging regions.
- Peak of $\chi_4$ should grow with increasing $\sigma$ if disorder enhances cooperative rearrangement.
- Alternatively, $\chi_4$ may flatten if fast cells break spatial correlations.
- The $\sigma$-dependence of $\chi_4$ directly tests whether disorder creates **more** or **less** cooperativity.

### 9.9 Per-cell motility–dynamics correlation

**Unique to our study**: Because $v_{A,i}$ is quenched and known, we can directly correlate each cell's assigned motility with its dynamical behavior.

**What to measure**:
- $D_i$ vs. $v_{A,i}$: Is per-cell diffusivity linearly proportional to $v_{A,i}$, or is there a threshold?
- Cage escape time vs. $v_{A,i}$: Do slow cells remain caged indefinitely or does neighbor-assisted escape occur?
- Spatial correlation of mobility: Do fast cells cluster, or are they homogeneously distributed?

**Expected regimes**:
1. **Weak disorder** ($\sigma \ll \bar{v}_A$): All cells above the cage-scanning threshold → collective behavior similar to uniform case.
2. **Moderate disorder** ($\sigma \sim \bar{v}_A$): Two populations emerge — fast cage-scanners and slow/arrested cells → persistent heterogeneity.
3. **Strong disorder** ($\sigma \geq \bar{v}_A$): Some cells have $v_{A,i} \approx 0$ → permanently arrested inclusions → potential Griffiths-like rare-region effects if arrested patches grow.

---

## 10. Finite-Size Scaling Predictions

### 10.1 System sizes

| N | Domain L | Cells per side | Status |
|---|---|---|---|
| 288 | 1562 | ~17 | Complete (t=880k) |
| 1152 | 3124 | ~34 | In progress (t~185k) |
| 4608 | 6249 | ~68 | In progress (t~200k) |

### 10.2 What should converge with N

- **$S(q)$**: Peak position $q^*$ and peak height should converge by N=1152. Low-$q$ behavior ($q < q^*$) may show $N$-dependent changes if mesoscale fluctuations exist.
- **$\tau_\alpha$**: Should increase with $N$ (larger systems relax slower due to fewer "boundary escape routes"). Convergence rate indicates whether relaxation is intrinsic or boundary-assisted.
- **$\beta$**: The stretching exponent is the most diagnostic. If $\beta > 1$ at N=288 drops to $\beta < 1$ at N=1152/4608, this confirms finite-size driving. If $\beta$ remains $> 1$, the compressed relaxation is intrinsic to the active system.
- **$D$**: Should decrease with $N$ (cells in the bulk of larger systems have fewer boundary effects).
- **$D\tau_\alpha$**: The Stokes-Einstein violation should be $N$-independent once finite-size effects are removed—this is the most robust measure of intrinsic disorder effects.

### 10.3 What should NOT change with N

- **Qualitative trends with $\sigma$**: If $D\tau_\alpha$ increases with $\sigma$ at N=288, it should also increase at N=1152 and N=4608 (possibly with different magnitude).
- **Phase diagram topology**: The ordering of $\sigma$ groups (which value of $\sigma$ gives fastest/slowest dynamics) should be robust.

---

## 11. What to Watch For: Danger Signals and Artifacts

### 11.1 Insufficient equilibration

**Signal**: $F_s(q^*, t)$ does not fully decay within the observation window → fits to $\tau_\alpha$ and $\beta$ are extrapolations.
**Diagnostic**: Check that the maximum lag time is at least $3\tau_\alpha$.
**Status**: N=1152 data at t~185k gives max lag ~100k, but fitted $\tau_\alpha$ ~170k–300k. This means the fits are unreliable. Need t > 500k.

### 11.2 Finite-size artifacts in β

**Signal**: $\beta > 1$ (compressed exponential).
**Diagnostic**: Compare N=288, 1152, 4608. If $\beta$ decreases systematically, the compression is a finite-size artifact.

### 11.3 Crystallization or spurious order

**Signal**: Sharp Bragg-like peaks in $S(q)$, regular cell packing.
**Diagnostic**: Visual inspection of snapshots; $S(q)$ isotropy check.
**Risk**: Low with quenched disorder and run-and-tumble dynamics breaking any configurational memory.

### 11.4 Drift in per-cell v_A assignment

**Signal**: Per-cell $v_{A,i}$ values change between restarts or across chains.
**Diagnostic**: Checkpoint stores per-cell $v_A$ (v4 format). Verify consistency across chains by checking header.

---

## 12. Synthesis: Position of Our Study in the Literature

### 12.1 What is genuinely new

1. **First study of quenched motility disorder in a phase field tissue model.** All prior work on quenched disorder in active glass used ABP or AOUP point-particle models (Keta 2022 for size, Debets 2023 for chirality). Our PFM adds realistic cell deformation, continuous interfaces, and tissue-specific mechanics.

2. **Systematic variation of disorder strength σ.** We explore the full range from $\sigma = 0$ (uniform motility, standard active glass) to $\sigma = \bar{v}_A$ (maximally disordered, some cells nearly arrested).

3. **Finite-size scaling with three system sizes.** N = 288, 1152, 4608 enables separating intrinsic disorder effects from finite-size artifacts—particularly critical for interpreting $\beta$.

4. **Direct motility–dynamics correlation.** Because $v_{A,i}$ is known and quenched, we can directly measure how each cell's assigned motility maps to its dynamical behavior—impossible in experimental systems where motility heterogeneity is neither controlled nor measured.

### 12.2 What we expect to find

Based on the cage-length framework [Debets 2021, 2022] and the Keta/Debets-chirality precedents:

1. **$D\tau_\alpha$ increases with $\sigma$**: Stokes-Einstein violation grows as disorder creates a wider spread of individual relaxation rates.
2. **$\tau_\alpha$ varies nonmonotonically with $\sigma$**: Moderate disorder may fluidize (optimal cage scanning for some cells), extreme disorder arrests (many cells stuck in $l_p \gg l_c$ regime).
3. **van Hove develops tails with $\sigma$**: Two-population dynamics (fast/slow) becomes visible.
4. **$\beta$ approaches or drops below 1 at large $N$**: Compressed exponential from finite-size effects gives way to stretched relaxation from heterogeneous dynamics.
5. **Per-cell $D_i$ vs. $v_{A,i}$ shows threshold behavior**: Below some critical $v_{A,i}$, cells are permanently caged; above it, diffusion increases linearly. This threshold should be related to $l_c$.

### 12.3 How it connects to experiments

- Angelini 2011 / Garcia 2015: Our MSD and $\alpha_2$ predictions can be compared to wound-healing assay measurements.
- Atia 2018: Our cell shape statistics $P(AR)$ vs. $\sigma$ can be compared to the universal shape distributions they found.
- Park 2015 / Mitchel 2020: Disorder-induced unjamming provides a mechanical pathway for tissue fluidization distinct from EMT, consistent with the Park/Mitchel distinction between unjamming and EMT.
- Cancer biology: Our prediction that motility heterogeneity enhances tissue fluidity connects to observations that tumors with heterogeneous cell populations are more invasive.

---

## 13. References

### Active glass — foundational
- Berthier 2019, JCP 150, 200901 (review)
- Ni et al. 2013, Nature Comms 4, 2704
- Berthier 2014, PRL 112, 220602
- Flenner et al. 2016, Soft Matter 12, 7136
- Janssen 2019, J. Phys. CM 31, 503002 (review)
- Berthier & Kurchan 2013, Nature Physics 9, 310

### Cage length and softness
- Debets, de Wit & Janssen 2021, PRL 127, 278002
- Debets & Janssen 2022, PhysRevRes 4, L042033
- Liluashvili et al. 2017, PRE 96, 062608

### Quenched disorder in active systems
- Keta, Jack & Berthier 2022, PRL 129, 048002
- Debets, Löwen & Janssen 2023, PRL 130, 058201

### Tissue glass transition and jamming
- Bi et al. 2015, Nature Physics 11, 1074
- Bi et al. 2016, PRX 6, 021011
- Li, Das & Bi 2021, PRE 103, 022607
- Das, Sastry & Bi 2021, PRX 11, 041037
- Sadhukhan et al. 2024, bioRxiv 2024.03.14.584932

### Compressed exponentials and driven glasses
- Tjhung & Berthier 2020, PhysRevRes 2, 043334

### Griffiths physics
- Griffiths 1969, PRL 23, 17
- Vojta 2006, J. Phys. A 39, R143
- Moretti & Muñoz 2013, Nature Comms 4, 2521

### Experiments
- Angelini et al. 2011, PNAS 108, 4714
- Garcia et al. 2015, PNAS 112, 15314
- Atia et al. 2018, Nature Physics 14, 613
- Park et al. 2015, Nature Materials 14, 1040
- Grosser et al. 2021, PRX 11, 011033
- Mitchel et al. 2020, Nature Comms 11, 5053
- Vishwakarma et al. 2018, Nature Comms 9, 3246
- Bocanegra-Moreno et al. 2023, Nature Physics 19, 1767

### Aging in active glass
- Mandal & Sollich 2020, PRL 125, 218001
- Janzen & Janssen 2022, PhysRevRes 4, L012038
- Mandal et al. 2020, Nature Comms 11, 2581

### Reentrant and polar active glass
- Paoluzzi et al. 2024, Commun. Phys. 7, 57
- Paoluzzi et al. 2022, Commun. Phys. 5, 111

### Phase field models for tissues
- Palmieri et al. 2015, Sci. Rep. 5, 11745
- Najem & Grant 2016, PRE 93, 052405
- Löber et al. 2015, Sci. Rep. 5, 9172
- Wenzel & Voigt 2021, PRE 104, 054410
- Nonomura 2012, PLOS ONE 7, e33501

### Theory (MCT/RFOT)
- Szamel 2019, JCP 150, 124901
- Feng & Hou 2017, Soft Matter 13, 4464
- Nandi et al. 2018, PNAS 115, 7688
- Ruscher et al. 2021, J. Phys. CM 33, 064001

---

*Last updated: February 18, 2026*
