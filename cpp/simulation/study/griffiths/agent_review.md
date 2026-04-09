# Agent Review

**Reviewer Report:** "Absence of Griffiths rare-region effects in a phase field model of active tissue"

## 1. Summary and Significance

The manuscript claims that quenched motility disorder — assigning each cell a fixed self-propulsion speed $v_{A,i} \sim \mathcal{N}(\bar{v}_A, \sigma^2)$ — does not produce the rare-region (Griffiths) phenomenology expected from equilibrium statistical mechanics when applied to a confluent tissue near its motility-driven jamming transition. Instead, the authors report that disorder accelerates relaxation ($\tau_\alpha$ drops $\sim 4.5\times$), narrows the relaxation spectrum ($\beta$ rises from 0.57 to 0.90), and suppresses cooperative dynamics ($\chi_4^{\max}$ drops by an order of magnitude). They attribute this to a "stirred glass" mechanism in which fast cells mechanically disrupt their neighbors' cages.

**Novelty assessment.** Testing whether Griffiths physics extends to active biological matter is a genuinely interesting and original question. The Griffiths framework has been applied to quantum magnets, absorbing-state transitions, and network systems (Vojta 2006, Moretti & Muñoz 2013), but never to a tissue-level mechanical model with quenched activity disorder. The idea of mapping the dilute Ising ferromagnet's rare regions onto fast/slow cell patches is conceptually appealing, and demonstrating that the mapping fails is a substantive negative result. However, as I elaborate below, the current manuscript is in a preliminary state that falls well short of publication readiness for PRE.

**Intended audience.** Active matter + glass physics community, with secondary appeal to cell mechanics. The framing is appropriate for this audience, though the paper would benefit from clearer signposting of what is truly novel versus expected from prior work on active fluidization (Mandal et al. 2020, Berthier 2019).

---

## 2. Scientific Rigor — Claim-by-Claim Assessment

## Claim 1: "The clean system is near the fluid phase" ($D = 0.013$, $D_\text{free} = 0.16$)

**Assessment:** Partially supported, but problematic for the study design. The system has $D > 0$ at $\sigma = 0$, meaning it is already on the fluid side of the jamming transition. The Griffiths scenario most naturally predicts anomalous dynamics when the mean control parameter sits near the transition, but the most dramatic signatures (frozen rare regions) arise when the bulk is jammed and rare fluid pockets nucleate. The authors acknowledge this in the Discussion, but the choice to anchor the study at $\bar{v}_A = 0.008$ in the fluid phase weakens the test. The Griffiths prediction is not that disorder in the fluid phase produces frozen rare regions — it is that near the transition, disorder produces a broad Griffiths phase between $v_A^*(\sigma)$ and $v_A^*(0)$. If the clean system is already fluidized, the relevant Griffiths rare regions would be jammed islands — and the narrative should be reframed accordingly. This is mentioned but underexplored.

## Claim 2: "$\beta$ increases from 0.57 to 0.90 — opposite to Griffiths prediction"

**Assessment:** Insufficiently supported. This is the paper's central result, yet critical details are absent:

- **Fit uncertainties are not reported.** Three replicates per condition is marginal for a noisy quantity like $\beta$. The manuscript presents single numbers (e.g., $\beta = 0.57$, $\beta = 0.90$) without confidence intervals or standard errors across runs. From the LOG_BOOK, the $\sigma = 0.008$ condition shows enormous run-to-run variability ($D$ varying from 0.008 to 0.063). With such sensitivity to disorder realization at strong disorder, the reported $\beta$ values for $\sigma \leq 0.006$ likely also have substantial run-to-run variation.
- **Fitting range is unspecified.** The stretched exponential $Q(t) = \exp[-(t/\tau_\alpha)^\beta]$ is sensitive to the time range used. Are short-time transients excluded? What fraction of the decay is captured?
- **$R^2$ or goodness-of-fit metrics are absent from the manuscript.** The analysis script (analyze_griffiths_deep.py) computes $R^2$, but this is never reported in the text or tables. For $\sigma = 0.006$, $\beta = 0.90$ is near simple exponential — could the data equally be fit with a simple exponential ($\beta = 1$)? The reader cannot evaluate this.
- **Table II has TODO entries for $\sigma = 0.008$.** This is a critical data point — the strongest disorder — and it is missing. The manuscript explicitly states: "The data at $\sigma = 0.008$ are currently insufficient for reliable stretched-exponential fits" (Sec. IV.E). A manuscript submitted without its most extreme data point is incomplete.
- The $\beta$ trend relies on only 3 data points ($\sigma = 0, 0.003, 0.006$) — the $\sigma = 0.008$ row is blank. It is difficult to claim a "monotonic" trend with 3 points.

## Claim 3: "$\chi_4^{\max}$ drops by an order of magnitude (38 → 3.3)"

**Assessment:** Concerning statistical robustness. $\chi_4$ is the variance of $Q(t)$ — a variance-of-a-variance quantity. With 20 time origins per run and 3 runs, the effective statistics for $\chi_4$ are inherently noisy. The authors do not report error bars on $\chi_4^{\max}$. A factor-of-10 drop is large and likely real, but the absolute values (from 38 to 3.3) should be accompanied by uncertainty estimates. Furthermore, $\chi_4$ is known to be system-size dependent — at $N = 288$ the maximum cooperative cluster is $\sim$8–10 cells (as acknowledged in Sec. V.E), which already truncates $\chi_4$. Whether the peak of 38 for the clean system already reflects finite-size saturation (i.e., the entire system is one cooperatively rearranging region) is not discussed. If $\chi_4^{\max} \approx 38 \approx N/8$, the system may be in a regime where $\chi_4$ saturates at the system size — in which case the clean value is artificially capped, making the comparison to disordered systems misleading.

## Claim 4: "The stirred glass mechanism"

**Assessment:** Narrative, not a tested hypothesis. The "stirred glass" mechanism is a physically intuitive picture, but the manuscript does not provide any prediction that distinguishes it from the simpler explanation that disorder trivially increases the effective noise. In a system where $v_{A,i}$ is drawn from a distribution, the mean-squared propulsion $\langle v_{A,i}^2 \rangle = \bar{v}_A^2 + \sigma^2$ increases with $\sigma$. Thus the effective active forcing is simply stronger for any $\sigma > 0$, even at fixed $\bar{v}_A$. Could the fluidization be entirely explained by this effective-temperature increase? The authors explicitly invoke Debets et al. (2023) who showed that quenched chirality disorder is "not reducible to enhanced effective temperature," but they do not demonstrate the same for their system. A critical control would be: compare $\sigma > 0$ at $\bar{v}_A = 0.008$ with $\sigma = 0$ at $\bar{v}_A^{\text{eff}} = \sqrt{\bar{v}_A^2 + \sigma^2}$. If the dynamics are indistinguishable, the "stirred glass" mechanism reduces to a trivial effective-activity argument.

## Claim 5: Connection to the Debets cage-length framework

**Assessment:** Purely qualitative and post hoc. The manuscript invokes $l_p / l_c$ as "the microscopic origin" of the fluidization, but $l_c$ is never measured. The MSD is never shown — there is no caging plateau from which to extract $l_c$. Without a measured cage length, the mapping to Debets et al. is speculative. The claim that "cells in the high-$v_A$ tail sit near or above the optimal $l_p \approx l_c$ point" is unverifiable from the data presented.

## Claim 6: $\tau_\alpha$ decreases by $\sim 4.5\times$

**Assessment:** Plausible but confounded. As noted above, the effective forcing increases with $\sigma$ at fixed $\bar{v}_A$. The decrease in $\tau_\alpha$ could reflect this rather than any qualitative change in relaxation mechanism. The motility sweep (Experiment B) helps but is also limited: three $\bar{v}_A$ values at a single $\sigma$ provide very sparse coverage.

---

## 3. Missing Controls and Analyses

**Annealed vs. quenched disorder control.** This is the single most critical missing experiment. The entire Griffiths argument rests on the disorder being quenched. A control where the motility distribution is the same but reassigned periodically (annealed disorder) would isolate the role of quenching. If the annealed case shows the same fluidization, the quenched nature of the disorder is irrelevant and the Griffiths framing is artificial.

**Effective-temperature control.** Compare $(\bar{v}_A = 0.008, \sigma = 0.006)$ against $(\bar{v}_A = \sqrt{0.008^2 + 0.006^2} \approx 0.010, \sigma = 0)$. If the dynamics are similar, disorder is acting through a trivial effective-activity mechanism. This is computationally cheap (one additional clean simulation).

**Mean squared displacement.** Remarkably, the MSD is never shown. This is the most fundamental dynamical observable in glass physics — the caging plateau, the crossover to diffusion, the $l_c$ extraction. Without it, the paper cannot claim to characterize "glassy dynamics" or make quantitative contact with the cage-length framework.

**van Hove self-correlation function.** The displacement distribution $G_s(\Delta x, t)$ at lag times $\sim \tau_\alpha$ would directly visualize the bimodal fast/slow population that the authors invoke. Its absence is a significant gap.

**Sensitivity analysis of cage radius $a$.** The cage radius $a = 0.3\,d_\text{cell} \approx 28$ grid units is stated but not justified. Could the trends in $\beta$ or $\chi_4$ reverse for different $a$ choices? A quick sensitivity check (e.g., $a = 0.2$ and $a = 0.4$) is standard practice.

**Direct $v_{A,i}$–$D_i$ correlation.** The LOG_BOOK notes that this was "BLOCKED" due to a code issue (Bug 5 — simulation writes mean $v_A$ for all cells). The per-cell correlation between quenched motility and measured diffusivity is essential for establishing that the disorder is the mechanistic cause of the dynamical heterogeneity, rather than a secondary effect.

**Spatial correlation function.** The $\xi \approx 3.3$ cell spacings mentioned in the LOG_BOOK (Section 14.6, Check 7) never appears in the manuscript. Showing how the spatial extent of dynamical heterogeneity changes with $\sigma$ would substantially strengthen the "stirred glass" argument.

**Finite-size scaling.** The authors acknowledge $N = 288$ is small and that larger simulations ($N = 1152$, $N = 4608$) are in progress. Without at least one larger system size, the claims about the absence of Griffiths effects cannot be distinguished from the absence of large enough rare regions at this system size. The finite-size cutoff analysis in LOG_BOOK Section 14.5 explicitly estimates that $N = 288$ can only probe rare clusters up to $\ell \approx 8$–10 cells, which the authors themselves describe as "barely sufficient" — yet the manuscript title makes an unqualified claim about the "absence" of Griffiths effects.

**Figures.** The manuscript contains zero figures. For PRE, this is a critical deficiency. The data are presented entirely in two tables, one of which is incomplete. At minimum, the following figures are needed: (a) $Q(t)$ on log-log axes for each $\sigma$; (b) $\chi_4(t)$ curves; (c) MSD; (d) per-cell displacement distributions; (e) $\beta$ vs. $\sigma$ with error bars.

---

## 4. Presentation and Clarity

**No figures.** This is the most serious presentation issue. A manuscript for PRE that reports dynamical observables ($Q(t)$, $\chi_4(t)$, etc.) without any graphical representation of them is fundamentally unsuitable for peer review. Tables II and I are necessary but not sufficient.

**Incomplete data.** Table II has `\TODO{}` entries for the $\sigma = 0.008$ row. This is clearly a draft. The $\sigma = 0.008$ data point is arguably the most interesting (strongest disorder, nearest to the regime where rare Griffiths regions might form), and its absence substantially weakens the paper.

**Inconsistent notation.** The manuscript uses $v_A$ and `\vA` interchangeably (the latter is a custom macro). In the Introduction, the notation switches between $v_0$ (Bi et al.'s convention) and $v_A$ without comment.

The abstract is well-written and accurately summarizes the main findings. It is self-contained and clear.

Section V (Results) is well-organized with clear subsections. The logical flow — clean baseline → single-cell heterogeneity → collective dynamics → motility sweep → strong disorder variability — is effective.

The Discussion provides genuine insight. The three-point argument for why the Griffiths analogy breaks down (Sec. V) is the strongest part of the paper. The distinction between single-cell and collective heterogeneity (Sec. V.D) is well-articulated and physically illuminating.

Minor typographic issues: The text "the $(p_0, v_0)$ plane" in the Introduction should read "$(p_0, v_A)$" or "$(p_0, v_0)$" — there is an inconsistency in whether $v_0$ or $v_A$ denotes the active speed.

---

## 5. Literature and Context

**Missing citations — significant gaps:**

- **Henkes et al. (2020)**, Nature Comms 11, 1405: "Dense active matter model of motion patterns in confluent cell monolayers." This paper studies a dense active matter model directly relevant to confluent tissue dynamics and is uncited despite being highly relevant.

- **Sussman et al. (2018)**: The self-propelled Voronoi model studies by D. M. Sussman (e.g., "No unjamming transition in a Voronoi model of biological tissue," Soft Matter 2018) examine the glass transition in tissue models with motility. This body of work is absent.

- **Flenner and Szamel series**: E. Flenner and G. Szamel published several papers on active glassy dynamics in self-propelled particle systems (Soft Matter 2016, J. Chem. Phys. 2019). These are foundational works on $\chi_4$, dynamical heterogeneity, and relaxation in active glasses and should be cited.

- **Nandi et al. (2018)**, PNAS 115, 7688: Extended RFOT theory to active glasses, predicting activity-dependent glass transition. Directly relevant to the theoretical framework.

- **Szamel (2019)**, J. Chem. Phys. 150, 124901: Mode-coupling theory for active Brownian particles. Relevant for theoretical context on active glass transitions.

- **Pinto, Telo da Gama & Araújo (2022)**, Phys. Rev. Research 4, 023186: "Cell motility in confluent tissues induced by substrate disorder" — studies quenched substrate disorder effects on confluent tissue, the closest topical precedent in tissue models.

- **Paoluzzi, Levis & Pagonabarraga (2022, 2024)**: Studies of MIPS-to-glass crossover and polar active glass phases. Directly relevant to the phase diagram of dense active matter.

- **Griffiths physics in non-equilibrium systems**: The manuscript cites Vojta (2006, 2019) but does not discuss the substantial literature on Griffiths effects in absorbing-state transitions (contact processes with quenched disorder). Vojta (2006) Sec. IV classifies rare-region effects in non-equilibrium; the authors should comment on where their system sits in this classification. Moretti & Muñoz (2013, Nature Comms.) on Griffiths phases in brain networks is also relevant, as is the experimental observation of Griffiths effects in Rydberg gas networks (Wintermantel et al. 2021, Nature Comms.).

**Accuracy of cited claims:** The citation of Debets et al. (2021) for the cage-length framework is accurate. The citation of Vojta (2006) for Griffiths predictions is appropriate. The Berthier & Biroli (2011) review is correctly invoked for $\chi_4$ interpretation.

**Griffiths physics representation:** The manuscript does not carefully distinguish between Griffiths singularities (thermodynamic, static free energy) and Griffiths phases (dynamic, anomalous relaxation). Section I states "Griffiths rare-region effects: anomalous power-law relaxation" — but the original Griffiths (1969) result is about essential singularities in the free energy, not power-law relaxation. The dynamic Griffiths effects (stretched exponential or power-law $C(t)$, as derived in detail in the LOG_BOOK Sec. 14.1) are a separate, later development associated with Fisher (1992) and Vojta's classification. The manuscript should be more careful here.

---

## 6. Technical Concerns

**Phase field model specification.** The model is reasonably well-specified with parameters listed in Sec. III. However, the mobility coefficient $M = 1/2$ is stated but not obviously related to a standard convention. The factor of $1/2$ in Eq. (2) is unusual for Model B / Cahn-Hilliard dynamics (typically $M = 1$) — is this absorbed into the definition? The precise form of the double-well prefactor $W/\lambda^2$ vs. the commonly used $60/\lambda^2$ should be clarified.

**Cage radius $a = 0.3\,d_\text{cell}$.** This is stated without justification. In equilibrium glass physics, the cage radius is typically extracted from the MSD plateau ($l_c = \sqrt{\langle \Delta r^2 \rangle_\text{plateau}}$). Using a fixed fraction of cell spacing is an approximation. The authors should either justify this choice (e.g., by showing the MSD plateau agrees with $a \approx 28$) or perform a sensitivity test.

**Fitting protocol.** The analysis script reveals that the stretched exponential is fit using scipy.optimize.curve_fit with bounds $\beta \in [0.01, 2.0]$ and $\tau \in [10^{-3}, 10^{10}]$. The fitting range excludes $Q > 0.99$ and $Q < 0.05$. None of this information appears in the manuscript. The fitting range should be reported, and the sensitivity to the cutoffs should be checked.

**Equilibration.** $t_\text{eq} = 80{,}000$ is used. The LOG_BOOK (Sec. 16.2) shows RMS velocities decreasing to $\sim 5 \times 10^{-6}$ for the 4608-cell system, with only 8% decrease in the last 20,000 time units. This seems adequate for the $v_A = 0$ equilibration phase, but a quantitative criterion for equilibration (beyond "velocities are small") should be stated.

**Production run length.** $T_\text{prod} = 250{,}000$ at $\tau_\alpha \approx 72{,}000$ gives $T_\text{prod}/\tau_\alpha \approx 3.5$. This is marginal — reliable $Q(t)$ fits typically require $T_\text{prod} > 5\tau_\alpha$, and $\chi_4$ estimates benefit from more time origins. For the $\sigma = 0.006$ case with $\tau_\alpha = 16{,}000$, the ratio is $\sim 16$, which is good. But for $\sigma = 0$, the 3.5× ratio is thin, especially given the 20-origin scheme for $\chi_4$.

**Truncated normal distribution.** At $\sigma = 0.008 = \bar{v}_A$, the truncated normal has a significant pile-up near $v_{A,i} = 0$. The fraction of cells with $v_{A,i} < 0$ before truncation is $\Phi(0) = \Phi(-1) \approx 0.159$ — about 16% of cells would be assigned negative velocities and get clamped to 0. This creates a delta-function spike at $v_{A,i} = 0$ in the effective distribution, which is a qualitatively different kind of disorder from the Gaussian tails. This artifact should be quantified (how many cells have $v_{A,i} \approx 0$ at each $\sigma$?) and its effect on the observables discussed. A log-normal distribution would avoid this issue.

**CUDA implementation validation.** The instructions file mentions that the MPI and CUDA versions have been validated against each other to $\sim 10^{-5}$ precision. This is reassuring, but the manuscript should briefly note that the code has been validated (e.g., single-cell volume conservation, multi-cell repulsion tests at minimum).

---

## 7. Structural Issues

**Length.** The manuscript is currently too short for a PRE Regular Article and too incomplete for a PRE Letter. It has no figures, an incomplete table, and explicitly marks results as TODO. In its current state it reads as a working draft, not a submission.

**Supplemental material.** Once the paper is fleshed out with figures and additional analyses, the model details (full parameter table, equilibration validation, fitting protocol) could move to a Supplemental to keep the main text focused. The stretched-exponential fitting methodology and $\chi_4$ computation protocol should at minimum be in a Methods Supplement.

**Discussion.** The Discussion (Sec. V) is actually quite strong and adds genuine insight. The three-property breakdown of why Griffiths assumptions fail (Sec. V.C) is the paper's intellectual contribution. However, the "stirred glass" mechanism needs to be cast as a testable hypothesis (with predictions for annealed-disorder control, spatial correlations, system-size scaling) rather than a narrative.

**Conclusions.** The final sentence is essentially: "we'll have bigger simulations soon." This is appropriate for a talk but not for a published paper. The Conclusions should either (a) present the $N = 1152$ and $N = 4608$ data if available, or (b) make cleaner, more circumscribed claims about what can and cannot be concluded at $N = 288$.

---

## 8. Verdict: Major Revisions

This manuscript addresses an interesting and original question (does Griffiths physics apply to active tissue?) and arrives at a physically compelling answer (no — active forcing qualitatively changes how disorder couples to dynamics). However, the current draft has fundamental deficiencies that prevent publication:

**Required changes, prioritized:**

1. **(Critical)** Add figures. At absolute minimum: $Q(t)$ log-log plot, $\chi_4(t)$ curves, $\beta$ and $\chi_4^{\max}$ vs. $\sigma$ with error bars, MSD, and displacement distributions. Without figures this cannot be reviewed as a PRE article.

2. **(Critical)** Complete the $\sigma = 0.008$ data or remove the claim that the analysis extends to strong disorder. The TODO entries in Table II must be resolved.

3. **(Critical)** Report error bars and fit quality for all fitted quantities ($\beta$, $\tau_\alpha$, $\chi_4^{\max}$). With only 3 replicates, run-to-run variability should be shown explicitly.

4. **(Essential)** Show the MSD. This is the foundational dynamical observable and is required to make any connection to the cage-length framework. Extract $l_c$ and compute $l_p / l_c$ for each cell subpopulation.

5. **(Essential)** Add the effective-temperature control. Simulate $\sigma = 0$ at $\bar{v}_A^{\text{eff}} = \sqrt{\bar{v}_A^2 + \sigma^2}$ and compare. If the dynamics match $(\bar{v}_A, \sigma > 0)$, the "stirred glass" mechanism is falsified.

6. **(Important)** Add annealed-disorder control. This isolates the role of quenching. Without it, the Griffiths framing is unanchored.

7. **(Important)** Incorporate finite-size scaling data. Include at least $N = 1152$ results. At $N = 288$, the title claim "absence of Griffiths effects" is overreaching — it could equally be "absence of Griffiths effects at this system size."

8. **(Important)** Quantify the truncation artifact. Report the fraction of cells clamped to $v_{A,i} = 0$ at each $\sigma$ and discuss whether the pile-up at zero changes the interpretation.

9. **(Minor)** Expand the bibliography. Add Henkes et al. 2020, Flenner & Szamel (Soft Matter 2016), Nandi et al. 2018, Szamel 2019, Pinto et al. 2022, Paoluzzi et al. 2022, Moretti & Muñoz 2013. Clarify the distinction between thermodynamic Griffiths singularities and dynamic Griffiths phases.

10. **(Minor)** Fix notation inconsistencies ($v_0$ vs. $v_A$), clarify model equation conventions, and add a brief note on code validation.

**Summary.** The core idea is sound and potentially publishable, but the manuscript is in a pre-submission state. The combination of missing figures, incomplete data tables, absent error bars, and no MSD/displacement analysis means that the scientific claims cannot be independently evaluated. With the above revisions — particularly the effective-temperature control and at least one finite-size comparison — this could become a solid PRE Regular Article.

