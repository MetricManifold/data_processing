# Agent Review

> **Note (Feb 19, 2026):** This review was generated from a manuscript version whose
> Phase 0 quench table (Table IV) contained bilinear adhesion data ($\Delta r/R = 0.27,
> 3.7, 7.1$) and claimed a "sharp rigidity transition." The manuscript has been updated
> with correct gradient-coupling results showing smooth, sub-cell displacements
> ($\Delta r/R = 0.025$–$0.132$) and no neighbor exchanges. Sections referencing the
> old displacement numbers or the "sharp transition" (Claim 3, parts of Sec. 6.3)
> should be read with this correction in mind. The rest of the review — model
> formulation, stability analysis, missing analyses, literature gaps, parameter
> audit — remains applicable.

**Reviewer Report:** "Adhesion-controlled rigidity transition in a multi-cell phase field model"

## 1. Summary and Significance

The manuscript introduces gradient-coupling adhesion, $F_\text{adh} = J \sum_{i<j} \int \nabla\phi_i \cdot \nabla\phi_j \, dA$, into a GPU-accelerated multi-cell phase field model of confluent tissue ($N = 288$–$1152$ cells, $\phi = 0.89$). The dimensionless parameter $\tilde{J} = J/(2\gamma)$ measures the fraction of surface energy removed at shared interfaces, with an analytically derived stability bound $J < 2\gamma$. Two-cell simulations validate mass conservation (~1.2%), shape preservation, and the predicted critical point. The central result is an "adhesion quench" experiment: starting from a well-equilibrated zero-adhesion state and instantaneously enabling adhesion at $v_A = 0$, the authors observe a sharp rigidity transition between $\tilde{J} = 0.25$ (cells remain caged, displacement $0.27R$) and $\tilde{J} = 0.50$ (full rearrangements, displacement $3.7R$). The authors argue this demonstrates that gradient-coupling adhesion tunes the energy landscape analogously to the shape index $p_0$ in vertex models.

**Novelty assessment.** The quench experiment is genuinely novel. Vertex models at $T = 0$, $v_0 = 0$ are frozen regardless of $p_0$; only phase field models with continuous overdamped dynamics can reveal the relaxation path after a parameter quench. The formulation is clean — a single-parameter variational adhesion with an analytical stability bound and a sharp-interface limit recovering vertex model line tension. However, the manuscript is at a very early stage: Phases 1 and 2 (motility probe and full phase diagram) are entirely TODO, the effective shape index $p_\text{eff}$ is never measured, there are zero figures, and several data points are missing. In its current form, this is a preliminary report of the Phase 0 quench result plus model derivation, not a complete study.

**Intended audience.** Soft matter / tissue mechanics community interested in the vertex-model-to-phase-field connection, with secondary appeal to computational biophysics.

---

## 2. Scientific Rigor — Claim-by-Claim Assessment

### Claim 1: "The stability bound is $J < 2\gamma$"

**Assessment:** Well-supported. The derivation in Sec. II.D is clean and presented two ways: (a) substituting anti-parallel gradients at a shared interface to get $E_\text{shared} = (2\gamma - J) \int |\nabla\phi|^2 dx$, and (b) a positive-definiteness argument on the sum field $S = \phi_1 + \phi_2$. Both yield $J < 2\gamma$. The two-cell validation (Table III) confirms the prediction: the interface begins dissolving at $\tilde{J} = 1.0$ and cells merge at $\tilde{J} = 1.5$. The comparison with Nonomura's regularized model is informative and correctly identifies how the regularization extends the stability range.

One minor point: the manuscript states "the quartic repulsion provides additional stabilization that shifts the effective merger point slightly above $J = 2\gamma$" but does not quantify this shift. From the two-cell data, $J = 2.0$ ($\tilde{J} = 1.0$) already shows "interface dissolving," and $J = 3.0$ ($\tilde{J} = 1.5$) shows full merger. A statement like "the effective merger threshold lies between $\tilde{J} = 1.0$ and $1.5$" would be more precise.

The stability derivation also assumes perfectly anti-parallel gradients ($\nabla\phi_1 \approx -\nabla\phi_2$), which holds at a flat shared boundary but breaks at curved interfaces, triple junctions, and corners. In a many-cell packing, triple junctions are ubiquitous — three interfaces meet at roughly $120°$ angles, and the gradient dot products are no longer simply $-|\mathbf{g}|^2$. The effective stability margin at triple junctions may be smaller than the two-cell prediction suggests. The manuscript should at minimum acknowledge this limitation or provide a triple-junction stability estimate.

### Claim 2: "The sharp-interface limit recovers vertex model adhesion"

**Assessment:** Convincingly demonstrated. The $\tanh$-profile calculation in Sec. II.E yields $F_\text{adh} \sim -(J/3\lambda) \sum_{i<j} \ell_{ij}$, which matches the vertex model form $-\gamma_\text{vm} \sum \ell_{ij}$ with identification $\gamma_\text{vm} = J/(3\lambda)$. The integral $\int \text{sech}^4 u \, du = 4/3$ is correct. This is a strength of the paper — it establishes the formal connection between the phase field and vertex model adhesion at the sharp-interface level.

However, the manuscript does not verify this correspondence numerically. The two-cell data (Table III) could be used to check whether the equilibrium separation $d_\text{eq}$ is consistent with the predicted contact angle from the Young-Dupré relation (derived in the LITERATURE_REVIEW but absent from the manuscript). This would strengthen the claim that the phase field implementation faithfully reproduces the sharp-interface physics.

### Claim 3: "A sharp rigidity transition between $\tilde{J} = 0.25$ and $0.50$"

**Assessment:** The data in Table IV show a 14-fold jump in mean displacement ($0.27R \to 3.7R$) between these two values. This is a clear signal. However, several issues limit confidence:

- **Only 4 data points.** The quench was run at $\tilde{J} \in \{0, 0.25, 0.50, 0.75\}$ — too coarse to characterize the transition. The TODO note promises additional values ($J/\kappa = 0.20$–$0.50$, i.e., $\tilde{J} = 1.0$–$2.5$), but these exceed the stability bound. The physically meaningful intermediate values needed are $\tilde{J} \in \{0.30, 0.35, 0.40, 0.45\}$ — between the two existing points that bracket the transition.
- **Single replicate.** The quench is deterministic at $v_A = 0$, so run-to-run variability doesn't apply for a single starting configuration. But the transition point may depend on the starting configuration. The LOG_BOOK mentions that "three independent checkpoints on different clusters produce qualitatively identical transitions," but this is not presented in the manuscript. The manuscript itself says only one checkpoint was used.
- **No error analysis.** The mean displacement has no uncertainty estimate. While there is no stochastic noise (deterministic dynamics), the displacement is an average over 288 cells, and the cell-to-cell distribution provides meaningful error bars (standard deviation or interquartile range).
- **The control is good.** $\Delta r = 0.82$ grid units ($0.017R$) at $J = 0$ confirms equilibration. This is a critical check and it passes.

### Claim 4: "The adhesion quench has no vertex model analog"

**Assessment:** Correctly argued. In the vertex model at $T = 0$, $v_0 = 0$, the system is frozen in whatever configuration it occupies regardless of $p_0$. Changing $p_0$ relabels the energy without generating forces. In the phase field, overdamped dynamics $\partial_t \phi = -M \delta F/\delta\phi$ means that changing $J$ creates actual driving forces proportional to $\delta F/\delta\phi$. This is a genuine conceptual advantage of the phase field framework, and the manuscript makes this point clearly.

However, the cellular Potts model (CPM) *can* perform a similar quench — changing the adhesion energy $J_{\tau\tau'}$ between Monte Carlo sweeps would produce an analogous transient response. The claim should be narrowed from "no vertex model analog" to "no analog in vertex models at $T = 0$" — or the comparison should be extended to mention that the CPM can perform quench experiments but with Monte Carlo dynamics that lack a physical time scale. Chiang & Marenduzzo (2016, EPL 116, 28009) studied glass transitions in the CPM as a function of adhesion and should be cited in this context; Devanny et al. (2023, bioRxiv) explicitly examine jamming signatures in the CPM as a function of adhesion.

### Claim 5: "Gradient-coupling adhesion tunes the energy landscape analogously to $p_0$"

**Assessment:** Plausible but unverified. The manuscript makes this claim in the abstract, introduction, discussion, and conclusion, but never measures the effective shape index $p_\text{eff}$. Without $p_\text{eff}$ data, the mapping $\tilde{J} \to p_\text{eff}$ is a hypothesis, not a result. The Discussion (Sec. VIII) honestly states "the most important missing piece is the effective shape index," but the claim nevertheless appears in the abstract and conclusion as if established.

The sharp-interface limit proves that the adhesion energy has the correct functional form (proportional to contact length), but this does not prove that the transition occurs at $\langle p_\text{eff} \rangle \approx 3.81$. The diffuse interface ($\lambda/R = 0.14$), the quartic repulsion, and the finite packing fraction could all shift the critical shape index. Until $p_\text{eff}$ is measured, the analogy to $p_0$ is theoretical motivation, not a demonstrated correspondence.

### Claim 6: "Mass is conserved to within 1.2%"

**Assessment:** Supported by the two-cell data (Table III: mass ranges from 15,803 to 15,726 across $\tilde{J} \in [0, 0.75]$). However, this is for an isolated two-cell system over 500 TU. The many-cell ($N = 288$) system running for 20,000 TU may accumulate more drift. The manuscript reports no mass data for the quench runs. The LOG_BOOK shows that the $h(\phi)$ implementation (which preceded gradient coupling) had up to 17% mass inflation at strong adhesion. While the gradient coupling is expected to perform better (as argued in LOG_BOOK), the many-cell mass conservation should be verified and reported.

Moreover, the 1.2% mass drift in the two-cell case is not small — it corresponds to a loss of $\sim 77$ field units out of 15,803, roughly 0.5% per cell. Over 288 cells and 40× longer run time, even a constant drift rate would produce detectable volume changes. Since the dynamics are Allen-Cahn (non-conserved), mass is not guaranteed; only the volume penalty $\mu$ provides approximate conservation. The manuscript would benefit from plotting $\int \phi_i^2 \, dA$ vs. time for representative cells during the quench to verify that the volume penalty adequately constrains mass at all $\tilde{J}$ values.

---

## 3. Missing Controls and Analyses

**Effective shape index $p_\text{eff}$.** This is the single most critical missing analysis. The manuscript is structured around the analogy $\tilde{J} \leftrightarrow p_0$, but $p_\text{eff}$ is never computed. The marching-squares algorithm on $\phi_i = 0.5$ contours is described in Sec. III.F but never applied. Without $p_\text{eff}$, the central claim of the paper — that gradient-coupling adhesion is the phase field analog of the vertex model shape index — is unsupported.

**Finer $\tilde{J}$ resolution near the transition.** The transition lies between $\tilde{J} = 0.25$ and $0.50$. With only these two data points bracketing it, the transition could be sharp (first-order-like) or gradual (crossover). At minimum, $\tilde{J} \in \{0.30, 0.35, 0.40, 0.45\}$ are needed to characterize the transition width and sharpness.

**Multiple starting configurations.** The quench is deterministic for a given initial state, but the transition point (critical $\tilde{J}^*$) may depend on the initial configuration. Ten independent checkpoints from the pool of 100 equilibrated states should be tested at one or two $\tilde{J}$ values near the transition.

**Energy time series.** The Discussion (Sec. VIII.C) argues that "below the threshold, adjacent energy minima are separated by barriers exceeding the adhesion-induced energy gain; above it, the landscape tilts toward new minima." This claim is testable with the energy decomposition already described in Sec. III.F — but the data are listed as TODO in Sec. V.C. This is a critical missing analysis: the distinction between smooth decay (interface relaxation) and stepwise drops (rearrangement events) would directly demonstrate the energy landscape change.

**Neighbor topology analysis.** Sec. V.D is entirely TODO. Counting T1-like neighbor exchanges would definitively establish whether the displacement at $\tilde{J} = 0.50$ corresponds to actual rearrangements versus mere cell deformation. Without this, "displacement $3.7R$" could theoretically be achieved by large cell deformation without neighbor changes (though this seems unlikely at this magnitude).

**MSD during quench.** The analysis script computes MSD vs. time, which would show whether the displacement saturates (system reaches a new minimum) or continues growing (system flows continuously). This is computed but not plotted or discussed in the manuscript.

**Contact angle verification.** The LITERATURE_REVIEW derives the Young-Dupré relation $\cos\alpha = 1 - \tilde{J}$ for the contact angle at the shared interface. This prediction can be tested against the two-cell data but is not verified in the manuscript.

**Reverse quench (hysteresis).** As noted in Sec. 6.3, starting from the post-quench configuration at $\tilde{J} = 0.50$ and setting $J = 0$ would establish whether the rearrangements are reversible. If the system returns to the original packing, the transition is elastic; if it locks into the new topology, the transition involves irreversible neighbor exchanges. This is a single additional run per $\tilde{J}$ value and provides critical information about the nature of the transition.

**van Hove function or displacement distribution.** The analysis script computes mean displacement, but the full distribution $P(\Delta r)$ at each $\tilde{J}$ would reveal whether:
- All cells move uniformly (homogeneous rearrangement)
- A subset of cells move large distances while others remain caged (heterogeneous, Griffiths-like)
- The distribution is bimodal (two populations: caged + rearranging)

This distinction is essential for understanding the mechanism of the transition.

**MSD during quench showing saturation.** The key diagnostic that is computed but not presented: does $\langle \Delta r^2(t) \rangle$ saturate (new energy minimum reached) or grow indefinitely? At $v_A = 0$, an overdamped system must reach a minimum eventually. The time at which saturation occurs (relaxation time) as a function of $\tilde{J}$ would characterize the critical behavior.

**Two-cell contact angle vs. $\tilde{J}$.** The LITERATURE_REVIEW derives $\cos\alpha = 1 - \tilde{J}$. Extracting the angle from the $\phi = 0.5$ contours in the two-cell simulations would provide a direct, independent test of the sharp-interface prediction.

**Sensitivity to $\kappa$ and $\lambda$.** While $\kappa = 10$ and $\lambda = 7$ are inherited from the established Palmieri et al. model, the adhesion term is new and its interaction with these parameters has not been characterized. A two-cell equilibrium separation at a single $\tilde{J}$ value (e.g., $\tilde{J} = 0.50$) with $\kappa = 5, 10, 20$ and $\lambda = 5, 7, 9$ would verify that the transition location is robust to the inherited parameters.

**Phases 1 and 2.** These are entirely missing. Sections VI and VII consist solely of TODO placeholders. This is the most obvious incompleteness. The motility probe and phase diagram are described in Table II as part of the study design, but no data exist.

---

## 4. Presentation and Clarity

**No figures.** The manuscript has zero figures. This is unacceptable for any journal. At minimum:
- (a) Schematic of the gradient-coupling adhesion showing anti-parallel gradients at a shared interface.
- (b) Two-cell equilibrium separation $d_\text{eq}$ vs. $\tilde{J}$ from Table III.
- (c) Centroid displacement $\langle \Delta r \rangle / R$ vs. $\tilde{J}$ from Table IV, showing the sharp transition.
- (d) Representative VTK snapshots before and after quench at $\tilde{J} = 0$ (control), $0.25$ (rigid), and $0.50$ (rearranging).
- (e) Energy time series during quench at representative $\tilde{J}$ values.
- (f) Cell-level displacement distributions (histograms) at each $\tilde{J}$.

**Incomplete sections.** Secs. V.C (energy landscape), V.D (neighbor rearrangements), VI (motility probe), and VII (phase diagram) are entirely TODO. This means roughly half the Results section is empty. The paper is currently a model derivation + two-cell validation + one table of quench data.

**Well-written model sections.** Secs. II.A through II.E are clear and thorough. The free energy formulation, stability analysis, equations of motion, literature context, and sharp-interface limit are all well-presented and logically organized. The comparison with Nonomura (1 page within Sec. II.D) is particularly informative for positioning the work within the existing literature. This is the strongest part of the manuscript.

**Good abstract.** The abstract accurately conveys the content and is self-contained.

**Notation is consistent.** Unlike the Griffiths manuscript, this paper uses $\tilde{J}$, $v_A$, and $p_\text{eff}$ consistently throughout, with macros defined at the top.

**Table quality.** Tables I–IV are well-formatted with appropriate units and labels.

---

## 5. Literature and Context

**Missing citations — significant gaps:**

- **Manning, Foty, Steinberg & Schoetz (2010)**, PNAS 107, 12517: "Coaction of intercellular adhesion and cortical tension specifies tissue surface tension." This is the key theoretical paper connecting adhesion and cortical tension to tissue surface tension in the vertex model framework — directly relevant to the relationship between adhesion and $p_0$. 419 citations. Uncited.

- **Chiang & Marenduzzo (2016)**, EPL 116, 28009: Glass transitions in the cellular Potts model controlled by the ratio of cell-cell adhesion to cell-medium adhesion. The CPM is the natural precursor to phase field adhesion studies and can perform adhesion quenches (unlike the vertex model). This should be cited alongside the claim that the quench "has no analog in discrete-topology models."

- **Devanny, Lee, Kampman & Kaufman (2023)**, bioRxiv: "Signatures of Jamming in the Cellular Potts Model." Explicitly examines jamming as a function of adhesion energy and cell shape in the CPM — the closest prior work to the adhesion quench idea. Should be cited.

- **Jain, Voigt & Angheluta (2023)**, Sci. Rep. 13: "Robust statistical properties of T1 transitions in a multi-phase field model of cell monolayers." Directly characterizes T1-like rearrangements in a multiphase field model with confluent monolayers — relevant to the neighbor topology analysis promised in Sec. V.D.

- **Montel, Guigue & Pontani (2022)**, Front. Phys. 10: "Adhesion regulation and the control of cellular rearrangements: From emulsions to developing tissues." Reviews how adhesion controls rearrangements across scales, directly relevant to the manuscript's central theme.

- **Grosser et al. (2021)**, Phys. Rev. X 11, 011033: Cell and nucleus shape as an indicator of tissue fluidity in carcinoma. Extends the shape index criterion to 3D and cancer tissues. 145 citations. The manuscript cites this in the LITERATURE_REVIEW but not in the manuscript itself.

- **Li, Wei, Paoluzzi & Ciamarra (2021)**, Phys. Rev. E 103, 022607: "Softness, anomalous dynamics, and fractal-like energy landscape in model cell tissues." Directly relevant to the energy landscape discussion in Sec. VIII.C — shows that near the vertex model transition, the energy landscape becomes fractal with many near-zero-barrier pathways. 23 citations.

- **Pinto, Telo da Gama & Araújo (2022)**, Phys. Rev. Research 4, 023187: "Hierarchical structure of the energy landscape in the Voronoi model of dense tissue." Finds that at $T = 0$ the Voronoi model has a disordered solid with no rigidity transition. This is directly relevant to the comparison with Bi et al. (2015, 2016) and should be cited in the context of the claim that the phase field quench reveals what vertex/Voronoi models cannot.

- **Fielding, Cochran, Huang & Bi (2023)**, Phys. Rev. E 108, L042602: Constitutive model for the rheology of biological tissue. Derives continuum rheology from the vertex model, including shear-induced solidification — relevant for connecting the phase field dynamics to tissue rheology.

- **Beatrici, Kirch, Henkes & Graner (2023)**, Soft Matter: "Comparing individual-based models of collective cell motion in a benchmark flow geometry." A systematic comparison of five simulation models including phase field and vertex — directly relevant for contextualizing the claim that phase field models offer advantages over vertex models.

**Accuracy of cited claims:** The citations of Bi et al. (2015, 2016), Nonomura (2012), Palmieri et al. (2015), Najem & Grant (2016), Löber et al. (2015), and Moshe et al. (2018) are all accurate and appropriately invoked. The comparison with Nonomura's parameters is careful and correct.

**The literature context section (Sec. II.B) is excellent.** It clearly distinguishes the three existing adhesion mechanisms (Nonomura, Löber, Najem) and explains why each was designed for its specific purpose. This is one of the best parts of the manuscript.

---

## 6. Technical Concerns

### 6.1 Parameter choices and provenance

**Parameter provenance.** A literature cross-check confirms that the core model parameters ($\gamma = 1$, $\lambda = 7$, $R = 49$, $\kappa = 10$, $\mu = 1$, $M = 1/2$, $\tau = 10{,}000$) are inherited directly from Palmieri, Bresler, Wirtz & Grant (2015, Sci. Rep. 5:11745), Table 1 and Eq. (7). The double-well prefactor $30\gamma/\lambda^2$ is also from Palmieri Eq. (7). The manuscript should cite this provenance explicitly — currently, the parameters are listed in Table I without attribution, giving the impression they are arbitrary.

**Double-well prefactor $30\gamma/\lambda^2$.** **Double-well prefactor $30\gamma/\lambda^2$.** The Palmieri form $V(\phi) = (30\gamma/\lambda^2)\phi^2(1-\phi)^2$ with gradient energy $\gamma|\nabla\phi|^2$ produces a $\tanh$ profile with width parameter $\delta = \lambda\sqrt{2/30} \approx 0.258\lambda$. The 10%-to-90% transition width is $2\delta\ln 9 \approx 1.13\lambda$, so $\lambda$ serves as the practical interface width. This is a well-motivated choice. However, the manuscript does not state this relationship. A brief note explaining that $\lambda$ corresponds to the approximate 10%-90% transition width would help readers from different phase field communities, where conventions differ substantially — Nonomura (2012) uses $D_0, 1/4 u^2(1-u)^2$, Loewe et al. (2020) use $\alpha/4 \phi^2(\phi-\phi_0)^2$ with $\xi = \sqrt{2K/\alpha}$, Wenzel & Voigt (2021) use $\phi \in [-1,1]$ with $W = 1/4(\phi^2-1)^2$, and Graham et al. (2024) use $(g/\lambda)\phi^2(1-\phi)^2$. Stating the convention prevents confusion.

**Interface width $\lambda = 7$ vs. cell radius $R = 49$.** The ratio $\lambda/R \approx 0.14$ means the interface is roughly $14\%$ of the cell diameter. This is not infinitesimally thin — a cell has only $\sim 2R/\lambda \approx 14$ interface widths across its diameter. The sharp-interface limit calculation (Sec. II.E) assumes well-separated, non-overlapping $\tanh$ profiles, which requires the interfaces of adjacent cells to be at least $\sim 4\lambda$ apart in the transition zone. At $\phi = 0.89$, the inter-cell gap is small — roughly $(L/\sqrt{N\pi} - 2R) \approx 10$ grid units, comparable to $\lambda$. The overlapping tails of adjacent interfaces mean the actual adhesion integral differs from the sharp-interface prediction. The manuscript should estimate the finite-interface correction, or at minimum verify numerically that the adhesion energy per unit contact length matches $J/(3\lambda)$ in the two-cell data.

**Repulsion strength $\kappa = 10$ (from Palmieri et al.).** While this value is established in the source model, the energy scale separation between adhesion and repulsion is worth noting. In the overlap zone, $\kappa \phi_i^2 \phi_j^2 \sim \kappa = 10$, while adhesion energy density is $\sim J/\lambda^2 \approx 0.031$ at $J = 1.5$. The repulsion is $\sim 300\times$ stronger at full overlap, meaning the quartic repulsion dominates in the overlap region and adhesion acts only in the interface tail — physically correct for surface adhesion but limiting the accessible adhesion range.

**Volume penalty $\mu = 1$ (from Palmieri et al.).** At $\mu = 1$, a 10% area deviation costs $\sim 5700$ energy units vs. the adhesion energy per cell of $\sim 22$ energy units — the volume constraint is $\sim 260\times$ stiffer. Cells cannot swell or shrink significantly in response to adhesion. If area is approximately conserved, increasing contact area requires perimeter increase — the mechanism by which adhesion should raise $p_\text{eff}$. The manuscript never verifies this chain of reasoning.

**Packing fraction $\phi = 0.89$.** The choice is well-motivated by both the vertex model literature and the Palmieri precedent. Palmieri et al. used both $\rho = 0.85$ and $0.90$, finding that packing fraction significantly affects the dynamics (bursty migration at $\rho = 0.90$ but not at $0.85$). The present $\phi = 0.89$ matches the regime where vertex model comparisons are most relevant. Note that the Griffiths study by the same authors used $\phi \approx 0.85$ ($L = 1600$), so direct parameter-level comparisons between the two studies carry this caveat.

**Tumble time $\tau = 10{,}000$ (from Palmieri et al.).** The persistence length $l_p = v_A \tau$ determines the motility regime. At $v_A = 0.002$ (Phase 1): $l_p = 20$ grid units $\approx 0.4R$, producing very short persistence ($l_p < R/2$). This may make the system behave more like effective thermal fluctuations than genuinely active matter. The Phase 1 probe motility choice should be physically motivated.

**Comparison across phase field models.** For context, the multi-cell phase field literature uses several non-equivalent parameterizations:

| Paper | Field range | Double-well | Repulsion | Adhesion |
|---|---|---|---|---|
| Palmieri et al. (2015) | $[0,1]$ | $(30\gamma/\lambda^2)\phi^2(1-\phi)^2$ | $\kappa\phi_i^2\phi_j^2$ | None |
| Nonomura (2012) | $[0,1]$ | $(D_0/4) u^2(1-u)^2$ | $\beta h(u_i)h(u_j)$ | $-\gamma_N \nabla h_i \cdot \nabla h_j + c|\nabla h_i|^2$ |
| Loewe et al. (2020) | $[0,\phi_0]$ | $(\alpha/4)\phi^2(\phi-\phi_0)^2$ | $\epsilon\phi_i^2\phi_j^2$ | None |
| Wenzel & Voigt (2021) | $[-1,1]$ | $(1/4)(\phi^2-1)^2$ | Signed-distance | None |
| Graham et al. (2024) | $[0,1]$ | $(g/\lambda)\phi^2(1-\phi)^2$ | $(k/\lambda)\phi_i^2\phi_j^2$ | None |
| **This work** | $[0,1]$ | $(30\gamma/\lambda^2)\phi^2(1-\phi)^2$ | $\kappa\phi_i^2\phi_j^2$ | $J\nabla\phi_i\cdot\nabla\phi_j$ |

The model is squarely within the Palmieri lineage, with gradient-coupling adhesion as the single new ingredient. Including such a comparison in the manuscript would help readers from different modeling traditions.

### 6.2 Two-cell validation concerns

**Domain size $400 \times 400$ with $R = 49$.** Two cells of diameter $\sim 100$ in a $400 \times 400$ domain leave $\sim 200$ grid points between the cells and their periodic images. This is comfortably large for non-interacting cells, but at $\tilde{J} = 0.75$ the cells are pulled closer together ($d_\text{eq} = 92.3$), and the periodic copies are at distance $400 - 92 = 308$. The gradient tails of the periodic images overlap slightly with the primary cell pair. A quick test in a $600 \times 600$ domain at $\tilde{J} = 0.75$ would verify that the equilibrium separation is converged.

**Duration 500 TU.** Has equilibrium been reached for all $\tilde{J}$ values? At $\tilde{J} = 0.75$, the two cells must displace by 5 grid units (from $d_0 = 90$ to $d_\text{eq} = 92.3$??). Wait — the initial separation is $d_0 = 90$ and in Table III the control ($J = 0$) gives $d_\text{eq} = 97.5$, which is *larger* than the initial distance. This means the $J = 0$ two-cell system repels to $d = 97.5$ over 500 TU. But with adhesion on, $d_\text{eq}$ decreases below 97.5 — i.e., adhesion pulls cells closer than the repulsion-only equilibrium. This dynamics seems correct. But is 500 TU sufficient for convergence at all $J$? The manuscript should plot $d(t)$ vs. time for at least two $\tilde{J}$ values to verify equilibrium — or report the final velocity $v_\text{final}$ of the approach.

**No aspect ratio data presented.** The manuscript claims "cells maintain their circular shape at all stable $J$" but does not report the aspect ratio numerically. This should be a column in Table III — it is an important check that the gradient coupling does not produce spurious cell elongation at the two-cell level.

### 6.3 Quench protocol concerns

**Quench is instantaneous — physical?** The adhesion is enabled as a step function from $J = 0$ to $J > 0$ at $t = 80{,}000$. In real biology, adhesion turns on gradually as cells express cadherins. An instantaneous quench creates the maximum possible driving force at $t = 0^+$, which may produce overshooting — cells displace further than the eventual equilibrium. A gentler ramp (e.g., $J(t) = J_\text{final}(1 - e^{-t/t_\text{ramp}})$) would test whether the displacement results are robust to the quench protocol, or whether the 14× jump is partly a transient artifact. The manuscript should at minimum report whether the displacement has saturated by $t = 20{,}000$ or is still growing.

**No reverse quench (hysteresis test).** If the system truly undergoes a rigidity transition, the reverse quench — starting from the $\tilde{J} = 0.50$ post-quench state and setting $J = 0$ — should show whether the rearrangements are reversible. If the system returns to the original configuration, the rearrangements are elastic and the transition is continuous. If it stays in the new configuration, the rearrangements involve topological changes and the transition has a hysteretic component. This is a simple, cheap experiment that would add substantial physical insight.

**Quench duration $2\tau = 20{,}000$ TU.** The quench runs for $2\tau$, but the relaxation time at the transition should diverge (critical slowing down). At $\tilde{J} = 0.25$ (rigid side), the system relaxes quickly (interface adjustment only). But right at the critical point, the relaxation time should grow without bound. The current coarse sampling ($\tilde{J}$ spaced by 0.25) cannot detect this divergence. A finer scan with longer runs near $\tilde{J}^*$ would be needed to characterize the critical behavior.

**What does displacement $3.7R$ mean physically?** A mean displacement of $3.7R \approx 181$ grid units means the average cell has moved $\sim 3.7$ cell radii. In a system with $\sim 17$ cells per linear dimension, this is $\sim 22\%$ of the system size. Is this asymptoting to a new equilibrium, or is the system flowing indefinitely? The distinction matters enormously: if the system reaches a new minimum, the quench reveals an energy landscape reorganization; if it flows indefinitely (at $v_A = 0$!), something is fundamentally wrong — a purely overdamped system without driving forces must eventually reach a minimum. The MSD vs. time curve (which the analysis script computes but the manuscript does not show) would immediately answer this.

### 6.4 Model specification concerns

**Forward Euler stability at high $J$.** The equation of motion is integrated with explicit Euler at $\Delta t = 0.02$. The adhesion contributes $-J \nabla^2 \phi_j$ to the variational derivative, which adds a second-derivative term with coefficient $J$ to the already-present $-2\gamma \nabla^2 \phi_i$. The effective Laplacian coefficient becomes $2\gamma + J$ (worst case), which for $J = 1.5$ ($\tilde{J} = 0.75$) is $3.5$ — a 75% increase over the $J = 0$ case. The CFL-like condition for stability of an explicit scheme on $\partial_t u = -D \nabla^2 u$ is $\Delta t < \Delta x^2 / (4D)$, giving $\Delta t < 1/(4 \times 3.5) \approx 0.071$. The current $\Delta t = 0.02$ satisfies this, but the margin narrows. The manuscript should note this and state that no instability was observed.

**Equilibration at $\phi = 0.89$.** The equilibration follows Palmieri et al. (2015) with $8\tau$ at $v_A = 0$, $J = 0$. The control displacement of $0.017R$ confirms this is adequate. However, the manuscript does not discuss how equilibration changes with system size. The LOG_BOOK notes that 1152-cell and 4608-cell equilibrations are in progress. If these are used for system-size comparisons, the equilibration quality should be verified separately.

**Mobility $M = 0.5$.** The equations of motion (Eq. 7) use $M$ as a prefactor on $\delta F/\delta\phi$. With $M = 0.5$ and $\Delta t = 0.02$, the effective update is $\Delta \phi = -0.5 \times 0.02 \times \delta F/\delta\phi = -0.01 \times \delta F/\delta\phi$. This is equivalent to $M = 1$ with $\Delta t = 0.01$. The mobility is stated in Table I but never discussed. In the Cahn-Hilliard literature, the standard form is $\partial_t \phi = M \nabla^2 (\delta F/\delta\phi)$ (conserved dynamics); the equation here is Allen-Cahn-like (non-conserved). This distinction should be made explicit since it affects mass conservation.

**Mass conservation mechanism.** The dynamics are Allen-Cahn (non-conserved order parameter), not Cahn-Hilliard (conserved). The manuscript claims (via the LITERATURE_REVIEW) that $\int \nabla^2 \phi_j \, dA = 0$ by the divergence theorem, so the adhesion cannot cause mass drift at the continuum level. This is correct — but the Allen-Cahn dynamics with the double-well potential can cause mass drift regardless of the adhesion term. The volume constraint $\mu(\int \phi^2 dA - A_0)^2$ is the mechanism that enforces approximate mass conservation. The manuscript should be more precise about what conserves mass (the volume penalty, not the divergence theorem) and acknowledge that mass conservation is approximate, not exact.

**Volume computation uses $\int \phi^2 dA$, not $\int \phi \, dA$.** The volume constraint (Eq. 4) penalizes $(\int \phi_i^2 \, dA - A_0)^2$. For a sharp interface ($\phi = 0$ or $1$), $\int \phi^2 = \int \phi$. But for the diffuse interface, $\phi^2 < \phi$ in the transition zone ($0 < \phi < 1$), so $\int \phi^2 < \int \phi$. This means the "volume" measured by Eq. 4 is smaller than the total field mass, and the target $A_0 = \pi R^2$ should be consistent with this definition. This is standard in phase field models but worth a brief note for readers from the vertex model community.

**Packing fraction definition.** $\phi = N\pi R^2/L^2 = 0.89$ assumes cells are discs of radius $R$. In reality, the diffuse interface means the "effective radius" is slightly larger than $R$ (the $\phi = 0.5$ contour extends beyond the target disc). This could push the effective packing fraction above 0.89. At $\lambda/R = 0.14$, the correction is probably small (~2%), but it should be noted.

**The adhesion variational derivative (Eq. 6) has a sign issue to clarify.** The adhesion is $F_\text{adh} = J \sum_{i<j} \int \nabla\phi_i \cdot \nabla\phi_j \, dA$. Taking the variational derivative $\delta F/\delta\phi_i$ requires integration by parts: $\delta F/\delta\phi_i = -J \sum_{j \neq i} \nabla^2 \phi_j$. But this should carry a factor of 1, not 2, because cell $i$ appears once in each pair $(i,j)$ via the $\sum_{i<j}$ convention — when differentiating, cell $i$ contributes to all pairs where it is the first or second index. The factor is correct as written (each pair counted once, $\sum_{j \neq i}$ not $2\sum_{j \neq i}$), but the manuscript does not explain why the $\sum_{i<j}$ in the energy becomes $\sum_{j \neq i}$ in the variational derivative. This implicit factor-of-2 accounting step should be made explicit for the reader.

**Periodic boundary conditions and the five-point stencil.** The adhesion force is computed as $-J(\nabla^2 S - \nabla^2 \phi_i)$ using a five-point Laplacian stencil. At periodic boundaries, the stencil wraps around, correctly including the gradient contributions from cells on the opposite side of the domain. But for the sum field $S = \sum_k \phi_k$, evaluating $\nabla^2 S$ near a domain edge includes contributions from periodic images of all 288 cells. Since $\phi_k \approx 0$ far from cell $k$'s center, only nearby cells contribute in practice. But the manuscript should state that the stencil correctly handles periodicity — a reader might worry about edge artifacts.

**Cage radius $a = 0.3 d_\text{cell} \approx 28$ grid units.** This is the same value used in the Griffiths manuscript, and the same concerns apply: the choice is ad hoc, not extracted from the MSD plateau. In the adhesion study, the cage structure changes with $\tilde{J}$ — stronger adhesion may tighten or loosen cages. Using the same $a$ for all $\tilde{J}$ values could obscure the transition in $Q(t)$. A sensitivity analysis of $Q(t)$ to the cage radius choice, or extraction of $a$ from the MSD at each $\tilde{J}$, would be appropriate.

**The vertex model connection assumes all cell-cell interfaces are shared.** The sharp-interface result $F_\text{adh} \sim -(J/3\lambda) \sum_{i<j} \ell_{ij}$ sums over all shared interfaces. But at $\phi = 0.89$, cells are confluent but not perfectly space-filling — there are small interstitial gaps where no cell field exceeds 0.5. At these gaps, neither cell contributes gradient, so the adhesion is zero. The effective adhesion per cell is thus not $-(J/3\lambda) \times P_i$ (total perimeter) but $-(J/3\lambda) \times P_\text{shared}$ (shared perimeter only). The ratio $P_\text{shared}/P_\text{total}$ is itself a function of $\phi$ and $\tilde{J}$. This distinction matters for the quantitative mapping to the vertex model, where all cell boundaries are shared by definition.

---

## 7. Structural Issues

**The paper is half-written.** Secs. VI, VII, and parts of V (energy landscape, neighbor topology) are entirely TODO. This means the paper currently contains: model derivation → two-cell validation → one quench data table → discussion → conclusion. It reads as a methods paper with a single preliminary result.

**The structure is appropriate for the eventual paper.** If Phases 1 and 2 are completed, the section structure (Model → Methods → Two-cell → Quench → Motility probe → Phase diagram → Discussion) is logical and well-organized. The Discussion (Sec. VIII) already contains genuine physical insight about the energy landscape and the comparison with vertex models.

**The Discussion section is premature.** Several paragraphs discuss results that don't exist yet (motility probe, phase diagram). The discussion of "what vertex models cannot show" (Sec. VIII.C) is well-argued but repeats the Introduction. The Limitations subsection (Sec. VIII.D) is honest and appropriate.

**Bibliography.** 31 references, all correctly formatted with DOIs and journal information. The bibliography is adequate for the model derivation but needs expansion when the results sections are populated (see Sec. 5 above for missing citations).

**Supplemental material is needed.** The LOG_BOOK documents an extensive history of failed adhesion implementations ($h(\phi) \cdot \phi^2$, bilinear, etc.) before arriving at gradient coupling. A brief Supplemental note explaining why simpler forms fail would be valuable for the community and would justify the choice of $\nabla\phi_i \cdot \nabla\phi_j$ more strongly than the single paragraph in Sec. II.C.

---

## 8. Verdict: Major Revisions

The manuscript presents a well-formulated variational adhesion model with a clean stability analysis and a compelling preliminary result (the adhesion quench). However, it is fundamentally incomplete — roughly half the Results section is TODO, there are zero figures, the central mapping $\tilde{J} \to p_\text{eff}$ is never tested, and the motility probe and phase diagram that would make this a complete study are absent.

**Required changes, prioritized:**

1. **(Critical)** Add figures. At absolute minimum: two-cell equilibrium separation vs. $\tilde{J}$, quench displacement vs. $\tilde{J}$, VTK snapshots before/after quench, energy time series during quench, displacement distributions (histograms).

2. **(Critical)** Measure the effective shape index $p_\text{eff}$. Extract $\phi_i = 0.5$ contours and compute $P_i/\sqrt{A_i}$ at each $\tilde{J}$. Plot $\langle p_\text{eff} \rangle$ vs. $\tilde{J}$ and compare to $p_0^* \approx 3.81$. Without this, the paper's central thesis ($\tilde{J}$ is the phase field analog of $p_0$) is unsupported.

3. **(Critical)** Add intermediate $\tilde{J}$ values in the transition region. At minimum $\tilde{J} \in \{0.30, 0.35, 0.40, 0.45\}$ to characterize the transition width and sharpness.

4. **(Critical)** Complete the energy time series analysis (Sec. V.C). The distinction between smooth decay and stepwise rearrangement events is the key energy landscape signature that the Discussion promises.

5. **(Critical)** Show the MSD time series during the quench. This is already computed by the analysis script but absent from the manuscript. It answers whether the displacement saturates (new minimum) or grows indefinitely (continuous flow), which is fundamental to interpreting the transition.

6. **(Essential)** Complete the neighbor topology analysis (Sec. V.D). Count T1-like neighbor exchanges to confirm that the displacement at $\tilde{J} = 0.50$ corresponds to actual rearrangements, not just deformation.

7. **(Essential)** Test with multiple starting configurations. Run the quench from $\geq 10$ independent equilibrated states at $\tilde{J} = 0.35$ and $0.45$ (near the transition) to establish reproducibility and quantify the dependence on initial configuration.

8. **(Essential)** Report many-cell mass conservation. Plot $\int \phi_i^2 \, dA$ vs. time for representative cells during the quench at several $\tilde{J}$. The two-cell data show 1.2% drift over 500 TU; the many-cell quench runs 40× longer.

9. **(Essential)** Perform the reverse quench (hysteresis test). Take the post-quench configuration at $\tilde{J} = 0.50$ and set $J = 0$. If the system returns to the original packing, the rearrangements are elastic; if not, they are irreversible topology changes.

10. **(Important)** Complete Phase 1 (motility probe). Without at least a few $(\tilde{J}, v_A)$ data points, the paper cannot claim to establish a path toward quantitative comparison with the $(p_0, v_0)$ phase diagram.

11. **(Important)** Explicitly attribute model parameters to Palmieri et al. (2015). Clarify the convention for $\lambda$ (10%-90% interface width) and the distinction between Allen-Cahn (non-conserved) and Cahn-Hilliard (conserved) dynamics. Include a brief comparison table of phase field model conventions across the literature.

12. **(Important)** Verify the sharp-interface adhesion prediction numerically. Compare the adhesion energy per contact length from the two-cell data to $J/(3\lambda)$ to quantify the finite-interface correction at $\lambda/R = 0.14$.

13. **(Important)** Expand the bibliography. Add Manning et al. (2010), Chiang & Marenduzzo (2016), Devanny et al. (2023), Jain et al. (2023), Montel et al. (2022), Grosser et al. (2021), Li et al. (2021), Pinto et al. (2022b), Fielding et al. (2023), and Beatrici et al. (2023). Clarify the comparison with the cellular Potts model.

14. **(Important)** Extract the two-cell contact angle and compare to the Young-Dupré prediction $\cos\alpha = 1 - \tilde{J}$.

15. **(Minor)** Clarify that the dynamics are Allen-Cahn (non-conserved), not Cahn-Hilliard (conserved), and that mass conservation is enforced by the volume penalty, not by the equation structure. Note the forward-Euler stability margin. Explain the factor-of-2 accounting from $\sum_{i<j}$ to $\sum_{j\neq i}$ in the variational derivative.

**Summary.** The model formulation and stability analysis are solid, the sharp-interface recovery of vertex model adhesion is convincing, and the quench experiment is genuinely novel. But the manuscript is in a preliminary state — it needs the effective shape index measurement, energy landscape analysis, finer $\tilde{J}$ resolution, figures, and ideally at least Phase 1 data before it can be considered for publication. With these additions, this could become a strong PRE Regular Article establishing the phase field adhesion framework as a quantitative complement to vertex models.

---

## Appendix: Parameter Audit Summary

For reference, the following table summarizes every stated parameter, its provenance, and what remains to be verified.

| Parameter | Value | Source | Sensitivity tested? | Comment |
|---|---|---|---|---|
| $\gamma$ (gradient energy) | 1 | Palmieri et al. (2015) | N/A (sets energy scale) | Convention |
| $\lambda$ (interface width) | 7 | Palmieri et al. (2015) | No | Ratio $\lambda/R = 0.14$; sharp-interface corrections not quantified |
| $\kappa$ (repulsion) | 10 | Palmieri et al. (2015) | No | $\kappa/\gamma = 10$; $\sim 300\times$ stronger than adhesion in overlap zone |
| $\mu$ (volume penalty) | 1 | Palmieri et al. (2015) | No | Very stiff ($\sim 260\times$ adhesion energy for 10% deviation) |
| $M$ (mobility) | 0.5 | Palmieri et al. (2015), Eq. (1) | No | Degenerate with $\Delta t$; Allen-Cahn dynamics, not Cahn-Hilliard |
| $30/\lambda^2$ (double-well) | 30 | Palmieri et al. (2015), Eq. (7) | No | Makes $\lambda$ the 10%-90% interface width |
| $\Delta t$ (time step) | 0.02 | Not attributed | No | CFL margin $\approx 3.5\times$ at $\tilde{J} = 0.75$ |
| $R$ (cell radius) | 49 | Palmieri et al. (2015) | N/A (sets length scale) | — |
| $N$ (cell count) | 288 | This work | No (1152 in progress) | $\sim 17$ cells per linear dimension |
| $L$ (domain side) | 1562 | From $\phi = 0.89$ | N/A | — |
| $\phi$ (packing fraction) | 0.89 | Motivated by vertex model + Palmieri $\rho = 0.90$ | No | Palmieri found different dynamics at $\rho = 0.85$ vs. $0.90$ |
| $\tau$ (tumble time) | 10,000 | Palmieri et al. (2015) | No | Sets persistence length $l_p = v_A \tau$ |
| $a$ (cage radius) | 28 ($0.3 d_\text{cell}$) | Ad hoc | No | Not extracted from MSD plateau |
| $t_\text{eq}$ (equilibration) | 80,000 ($8\tau$) | Palmieri et al. (2015) protocol | Checked via control | Control $\Delta r = 0.017R$ |
| $t_\text{quench}$ (quench duration) | 20,000 ($2\tau$) | This work | No | Relaxation time diverges at $\tilde{J}^*$ |

Of the 15 parameters, 10 are inherited from the established Palmieri et al. model and 1 is checked by a control. The new adhesion parameter $J$ is the study variable. The main parameters requiring further validation are the interface width in the many-cell regime (sharp-interface correction) and the quench duration (whether relaxation has completed near $\tilde{J}^*$).

---
---

# Second Agent Review — Feb 25, 2026

> **Context:** This review was conducted after reading the full manuscript (manuscript.tex),
> the complete LOG_BOOK.md (~1060 lines), the supplementary_validation.tex, all analysis
> scripts (analyze_phase0_grad.py, analyze_two_cell.py, analyze_adhesion.py,
> plot_fig1_twocell.py), the LITERATURE_REVIEW.md, the supervisor's research plan
> (palmieri_model_2d_extensions.pdf), and the first agent review above. The manuscript
> has been updated since the first review: Phase 0 now shows correct gradient-coupling
> results, Phase 2 data exists for $\tilde{J} \in \{0, 0.125\}$, and the two-cell
> table uses Bresler parameters ($\gamma = 3.75$).

## 1. What's Working Well

### 1.1 The gradient-coupling form is a genuinely good choice

The journey through seven adhesion forms documented in the LOG_BOOK (bilinear → smooth
step → reduced-$\kappa$ → h(φ)·φ² → h(φ)·h(φ) → gradient coupling) is not just
development history — it's evidence of rigorous elimination. The gradient coupling is the
*only* form that simultaneously satisfies variational, surface-localized, self-gating, and
single-parameter criteria. Manuscript Sec. II.B–C tells this story well with the
three-model comparison (Nonomura, Löber, Najem).

### 1.2 The stability bound $J < 2\gamma$ is clean and publishable

The positive-definiteness proof via $S = \phi_1 + \phi_2$ (Sec. II.D) is elegant and
extends to $N$ cells in any spatial dimension. The two-cell validation (Table II)
confirming the critical point at $\tilde{J} = 1.0$ (interface dissolving) and merger at
$\tilde{J} = 1.5$ is exactly the right validation. The comparison with Nonomura's
regularization ($\sim 8.7\times$ extended range from adding $c\langle h'^2\rangle$
stiffness) correctly positions the trade-off: narrower range for analytical transparency.

### 1.3 The quench protocol is genuinely novel

The argument that this has no vertex model analog (at $T=0$, $v_0=0$, changing $p_0$
does nothing) is correct and clearly stated. The result — smooth, sub-cell interface
relaxation with no T1 events at $\phi = 0.89$ — is honest and physically interpretable.
The quench traces a continuous relaxation path that is fundamentally inaccessible to
discrete-topology models.

### 1.4 Phase 2 partial data (Table III) is the strongest result so far

The doubling of $D_\text{eff}$ from $4.4 \times 10^{-3}$ to $9.6 \times 10^{-3}$ at
$v_A = 0.008$ with just 12.5% surface energy reduction ($\tilde{J} = 0.125$) is a
clear, quantitative signal. The $\tau_\alpha$ drop from $>10^5$ to $92{,}649$ is
convincing evidence that adhesion lowers the motility threshold for unjamming. The
stretched exponent $\beta$ transitioning from 2.0 (jammed) through 1.32 to 0.82
(glassy) traces the expected progression.

### 1.5 The sharp-interface limit is the theoretical bridge

The calculation $F_\text{adh} \sim -(J/3\lambda)\sum_{i<j}\ell_{ij}$ recovering vertex
model adhesion $-\gamma_\text{vm}\sum\ell_{ij}$ with $\gamma_\text{vm} = J/(3\lambda)$
provides the formal connection needed. This is a strength of the paper.

### 1.6 The manuscript structure is well-organized

The logical flow — Model → Methods → Two-cell → Quench → Motility → Phase diagram →
Discussion — is clean. The literature context section (Sec. II.B) clearly distinguishes
the three existing adhesion mechanisms and is one of the best parts of the paper.

---

## 2. Critical Gaps

### 2.1 The manuscript is missing its centerpiece figure

The `\TODO` on Fig. 1 (two-cell + quench + energy) is blocking. The `plot_fig1_twocell.py`
script is drafted and two-cell data exists locally. The quench data from rorqual (run_01)
is ready. **This figure should be the next concrete deliverable** — without it, the paper
reads as a theory document rather than a simulation study.

### 2.2 Phase 2 $\tilde{J}$ range stops at 0.125

Table III in the manuscript only has data for $\tilde{J} \in \{0, 0.125\}$. The
instructions mention the McLellan isotropic Laplacian stencil was needed for higher
$\tilde{J}$. **The phase boundary cannot be drawn with only two adhesion values.** At
minimum $\tilde{J} \in \{0, 0.125, 0.25, 0.375, 0.50\}$ are needed to see the shape
of the transition. This is the single biggest data gap.

### 2.3 Shape index $p_\text{eff}$ is defined but never measured

The manuscript promises a vertex model connection via
$\langle p_\text{eff}\rangle \approx 3.81$ at the transition, but every shape index
section is a `\TODO`. This is a referee red flag — the abstract and introduction promise
a connection that the results never deliver. Required:
- Marching-squares extraction of $\phi_i = 0.5$ contours from per-cell VTK fields
- Perimeter $P_i$ and area $A_i$ from those contours
- $p_{\text{eff},i} = P_i / \sqrt{A_i}$ distribution as a function of $\tilde{J}$

This is technically the hardest remaining analysis but also the most impactful. Without
it, the paper is "adhesion modifies cell dynamics" (incremental) rather than "phase field
adhesion maps onto the vertex model shape index" (novel).

### 2.4 Energy decomposition during quench is missing

Sec. V.B (`\TODO{Energy time series...}`) is empty. The quench is the novel experiment —
showing energy traces would prove it's smooth relaxation (interface reshaping) rather than
stepwise rearrangements. This distinguishes "cells adjust contact angles" from "cells
rearrange."

### 2.5 Contact angle measurement not attempted

The instructions and LOG_BOOK discuss the Young-Dupré prediction
$\cos\alpha = 1 - \tilde{J}$. This is a clean, quantitative test of the sharp-interface
limit. A referee will request it (it's listed in the "Analyses a Referee Would Likely
Request" section of the adhesion-study instructions).

---

## 3. Comparison to the Supervisor's Research Plan

The supervisor's research plan (`palmieri_model_2d_extensions.pdf`) is a **much broader**
program — 5 phases, up to 6,000 cells, percolation of fluidisation, polydispersity,
contact inhibition of locomotion, oscillatory shear, etc. The adhesion study is
essentially a **focused subset of Phase 4** (Adhesion) from that plan. Specifically:

| Supervisor's Phase 4 | Adhesion Study | Status |
|---|---|---|
| 4A: Jamming phase diagram with adhesion (all-normal, varying $J_{NN}$) | Phase 2: $(\tilde{J}, v_A)$ diagram | **Partial** — only $\tilde{J} \leq 0.125$ |
| 4B: Differential adhesion ($J_{CN}$ varied, cancer vs. normal) | Not applicable (single cell type) | — |
| 4C: Percolation with adhesion | Not planned | — |
| 4D: Adhesion mismatch alone ($\gamma_c = \gamma_n$) | Not applicable | — |

This is perfectly fine for a focused PRE paper — the full Phase 4 isn't needed. But it
means **this paper needs to be sharp and complete within its scope**: the $(\tilde{J}, v_A)$
phase diagram with the vertex model connection.

Other items from the supervisor's plan relevant to this scope:

- **Shape index (plan Phase 2 analysis (a))** — the supervisor explicitly asks for
  $p_0 = L_n/\sqrt{A_n}$ distributions and comparison to 3.81. This *must* be done.
- **Voronoi tessellation (plan Phase 2 analysis (b))** — not essential for a PRE letter
  but useful supplementary material.
- **Cage-relative displacements (plan Phase 2 analysis (e))** — important for
  distinguishing true rearrangement from collective drift.
- **T1 event detection (plan Phase 2 analysis (f))** — the manuscript mentions counting
  neighbor changes but has no implementation.
- **Tissue surface tension (plan Phase 4 analysis (n))** — the supervisor specifically
  requests this. The gradient coupling gives a clean prediction: effective surface
  tension = $\gamma(1 - \tilde{J})$ at shared interfaces. Validating this with a tissue
  strip simulation would strengthen the paper considerably.

---

## 4. Suggestions — Prioritized

### Priority 1 (must-have for submission)

1. **Fill out Phase 2 to at least $\tilde{J} = 0.50$** — resolve the McLellan stencil
   issue and run the sweep. Without this, the phase boundary cannot be located.

2. **Extract $p_\text{eff}$** from existing VTK data — without this, the paper's central
   claim (connection to vertex model $p_0^*$) is unsubstantiated.

3. **Generate Fig. 1** — two-cell validation + quench displacement curve + representative
   snapshots. The script exists; the data exists.

4. **Energy decomposition** during quench for at least 3 $\tilde{J}$ values — the
   signature that the Discussion section promises but the Results section doesn't deliver.

### Priority 2 (strongly recommended)

5. **Contact angle measurement** — test $\cos\alpha = 1 - \tilde{J}$ from two-cell or
   quench VTK data. Clean quantitative validation of the sharp-interface limit.

6. **One finite-size check** — compare $N = 288$ and $N = 1152$ at a single
   $(\tilde{J}, v_A)$ point. The 1152-cell equilibrations exist on multiple clusters.

7. **Error bars** on Phase 2 from 3 replicates (infrastructure already exists).

### Priority 3 (would elevate the paper)

8. **Relaxation timescale** vs $\tilde{J}$ from the quench — does it diverge? This would
   be a direct measure of proximity to a soft mode.

9. **Energy barrier estimation** — even a crude estimate of the T1 barrier height vs.
   $\tilde{J}$ would explain *why* motility is needed at $\phi = 0.89$.

10. **Tissue surface tension measurement** (small tissue droplet or strip). Since
    $\gamma_\text{tissue} \propto (1 - \tilde{J})$, this is a clean quantitative
    prediction unique to the model.

---

## 5. One Key Concern

The manuscript's abstract states that $\tilde{J}$ "plays the same role as the vertex model
target shape index $p_0$." This is a strong claim that currently rests on the
sharp-interface limit algebra alone. Without the $p_\text{eff}$ measurement showing that
$\langle p_\text{eff}\rangle$ crosses $\approx 3.81$ at the transition, a referee will
(justifiably) object. Two options:

- **(Preferred)** Measure $p_\text{eff}$ and show the crossing — making the claim empirical.
- **(Fallback)** Soften the claim to: "$\tilde{J}$ plays an *analogous* role, and the
  sharp-interface limit predicts a mapping to the vertex model line tension."

---

## 6. Agreement with First Review

The first agent review (Feb 19) identified many of the same issues. I concur with its
critical items, particularly:

- **Missing $p_\text{eff}$** (first review items 2, 5) — still the highest-priority gap
- **Finer $\tilde{J}$ resolution** (first review item 3) — partially addressed by Phase 2
  at $\tilde{J} = 0.125$, but the full range $\{0.25, 0.375, 0.50\}$ remains absent
- **Energy time series** (first review item 4) — still TODO
- **Figures** (first review item 1) — still zero figures
- **Contact angle** (first review item 14) — still unmeasured
- **Multiple starting configurations** (first review item 7) — the infrastructure
  (100 equilibrated checkpoints on nibi) exists but hasn't been used for Phase 2

Items from the first review that are now **less urgent** given the updated manuscript:

- The mass conservation concern (first review item 8) is mitigated by the gradient
  coupling's better behavior vs. h(φ) — the LOG_BOOK reports no mass inflation with
  gradient coupling (unlike the 17% seen with h(φ)). Still worth one verification plot.
- The reverse quench / hysteresis test (first review item 9) remains interesting but is
  lower priority now that Phase 0 shows no T1 events — there's nothing to reverse.
- The CPM comparison (first review Sec. 2, Claim 4) is a valid refinement but doesn't
  change the core argument.

### New items not in the first review

- **Bresler parameters** ($\gamma = 3.75$, $\kappa = 10$, $\mu = 0.5$, $\xi = 1000$)
  have been adopted since the first review, shifting the stability bound from $J < 2$ to
  $J < 7.5$. The manuscript correctly reflects this, but the first review's parameter
  audit table (which lists $\gamma = 1$, $\mu = 1$) is now outdated.
- **The McLellan nine-point isotropic Laplacian stencil** is mentioned in Sec. III.A as
  the production stencil, replacing the five-point stencil discussed in the first review.
  This is important since the isotropic stencil reduces lattice anisotropy artifacts that
  could affect contact angle measurements and cell shape at high $\tilde{J}$.
- **Phase 2 data now exists** (Table III), which the first review could not assess. The
  clean-system transition at $v_A^* \approx 0.009$ and the enhancement at
  $\tilde{J} = 0.125$ are solid new results.
- **Supervisor's plan context** — the adhesion study sits within a broader program. This
  is fine for a focused PRE paper, but it means the paper must be *complete within its
  scope* to stand alone.

---

## 7. Bottom Line

The physics is solid, the gradient-coupling form is well-motivated, the experimental
design is systematic, and the manuscript is well-organized. The main gaps are
**data completeness** (Phase 2 beyond $\tilde{J} = 0.125$) and **the shape index
extraction** (the key observable connecting to the vertex model). Those two items will
make or break the paper. With them, this is a strong PRE Regular Article establishing the
phase field adhesion framework as a quantitative complement to vertex models. Without
them, it's a model derivation paper with a preliminary result.
