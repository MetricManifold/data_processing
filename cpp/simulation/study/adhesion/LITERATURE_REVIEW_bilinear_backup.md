# Literature Review: Adhesion in Phase Field Models of Cell Monolayers

> **Purpose:** Comprehensive review of how cell-cell adhesion has been modeled in multi-cell phase field frameworks, with context from the vertex model and tissue jamming literature. This document supports the manuscript introduction.

---

## 1. Tissue Jamming and the Vertex Model

Dense epithelial monolayers exhibit a jamming transition between a solid-like, non-rearranging state and a fluid-like, collectively flowing state. The earliest quantitative observation was by Angelini et al. (2011), who reported glass-like dynamics in MDCK monolayers near a wound edge, with caging and subdiffusive motion at high density. Park et al. (2015) subsequently demonstrated that human bronchial epithelial cells from asthmatic donors are unjammed and adopt elongated shapes ($p > 3.81$), while healthy-donor cells remain jammed and compact. Mongera et al. (2018) showed that a fluid-to-solid jamming transition along the zebrafish tailbud axis is controlled by changes in cell-cell adhesion, providing direct in vivo evidence. Garcia et al. (2015) demonstrated active jamming in expanding MDCK monolayers driven by contact inhibition of locomotion. Malinverno et al. (2017) found that unjamming in MDCK-II monolayers can be triggered by endocytic reawakening of motility. Atia et al. (2018) showed that the shape distributions of individual cells in diverse epithelia collapse onto a universal curve predicted by the vertex model.

The dominant theoretical framework for confluent tissue mechanics is the **vertex model**, in which cells are polygons with energies depending on area and perimeter deviations from target values (Nagai & Honda, 2001; Farhadifar et al., 2007; Staple et al., 2010; Fletcher et al., 2014). Bi et al. (2015) showed that at zero temperature and motility, the vertex model exhibits a density-independent rigidity transition at a critical target shape index $p_0^* \approx 3.81$. Bi et al. (2016) extended the framework to active tissues, mapping the $(p_0, v_0)$ phase diagram in which motility drives unjamming. Further extensions include the active vertex model with explicit polarity (Barton et al., 2017), the role of multicellular rosettes in fluidization (Yan & Bi, 2019), the 3D extension to a surface-area-based shape index (Merkel & Manning, 2018), the effect of cell division and apoptosis (Czajkowski et al., 2019), and GPU-accelerated implementations (Sussman, 2017).

In the vertex model, cell-cell adhesion enters through the line tension $\Lambda_{ij}$ at shared edges. The effective line tension is reduced by adhesion molecules (cadherins), so stronger adhesion raises the target shape index $p_0$, pushing the tissue toward the fluid phase. This is how the differential adhesion hypothesis (Steinberg, verified by Foty & Steinberg, 2005; computationally demonstrated with the Potts model by Graner & Glazier, 1992) connects to the rigidity transition: differential adhesion tunes $p_0$ and hence tissue-scale fluidity.

However, vertex models enforce several constraints: cell boundaries are straight shared edges, topology changes (T1 transitions) are instantaneous operations when edges fall below a threshold length, and cells tile the plane without gaps. These are not biological constraints but artifacts of the polygonal representation. Phase field models remove all of these.

---

## 2. Multi-Cell Phase Field Models

Phase field methods describe each cell $i$ by a smooth scalar field $\phi_i(\mathbf{r}, t)$, where $\phi_i \approx 1$ inside the cell and $\phi_i \approx 0$ outside, with a diffuse interface of width $\lambda$. Early single-cell phase field models were developed by Shao et al. (2010) for cell morphodynamics and Ziebert et al. (2012) for keratocyte motility.

**Nonomura (2012)** introduced the first multi-cell phase field model. The free energy has a cell-shape (Cahn-Hilliard) term, a volume constraint, and interaction terms. The cell-cell interaction energy is:

$$E_\text{int} = \sum_{m \neq m'} \frac{\beta}{2} \int h(\phi_m) h(\phi_{m'}) \, dr + \sum_{m \neq m'} \frac{\gamma}{2} \int \nabla h(\phi_m) \cdot \nabla h(\phi_{m'}) \, dr + \sum_m \frac{c}{2} \int |\nabla h(\phi_m)|^2 \, dr$$

where $h(\phi) = \phi^2(3 - 2\phi)$. The first term is excluded volume (repulsion via smooth-step overlap $h(\phi_m)h(\phi_{m'})$). The second term ($\gamma > 0$) is a **gradient-overlap adhesion** that decreases the energy when cell cortices ($\nabla h(\phi_i)$ terms, localized at the interface) overlap. The third term regularizes to prevent divergence, required when $c > \gamma$.

Key features:
- Adhesion is **interface-localized**: the $\nabla h(\phi_i) \cdot \nabla h(\phi_j)$ coupling is nonzero only in the diffuse interface region.
- The interaction range is set by the interface width.
- Nonomura demonstrated cell sorting (differential adhesion) with this formulation.

**Palmieri, Bresler, Wirtz & Grant (2015)** adopted the Nonomura framework for cell migration in monolayers. They used the quartic repulsion $\kappa \sum_{i<j} \int \phi_i^2 \phi_j^2 \, dA$ and focused on collective motility driven by elastic mismatch between cells, without a separate adhesion term. The volume constraint is enforced by a penalty $\mu(\int \phi_i \, dA - V_0)^2$. This work established the equilibration protocol ($8\tau$ at zero motility before production runs) that we follow.

**Löber, Ziebert & Aranson (2015)** developed a multi-cell model with a single-cell phase field motility model (Ziebert & Aranson) extended to multiple interacting cells. Their interaction terms appear in the equation of motion directly:

$$\partial_t \rho_i = D_\rho \Delta \rho_i - \rho_i(\rho_i - \delta_i)(\rho_i - 1) - \alpha A \mathbf{p} \cdot \nabla \rho_i - \lambda \rho_i \sum_{j \neq i} \rho_j^2 - \kappa \nabla \rho_i \cdot \sum_{j \neq i} \hat{f}(\nabla \rho_j)$$

The key terms are:
- **Steric repulsion**: $-\lambda \rho_i \sum_{j \neq i} \rho_j^2$. This penalizes overlap of the field $\rho_i$ with the squared field of neighbors. Löber et al. use $\lambda = 30$ (strong repulsion). Note this is *not* the bilinear $\rho_i \rho_j$ form — it is $\rho_i \rho_j^2$, cubic in the fields, though it can be derived from a variational derivative of the quartic energy $\frac{\lambda}{2} \int \rho_i^2 \rho_j^2 \, dA$.
- **Adhesion**: $- \kappa \nabla \rho_i \cdot \sum_{j \neq i} \hat{f}(\nabla \rho_j)$. This is an **advection-based adhesion** that drives cell $i$'s interface along the outward normal of cell $j$'s interface. The function $\hat{f}$ regularizes the gradient. This is effectively a gradient-coupling that attracts cell boundaries toward each other.

Key features:
- Repulsion is effectively quartic overlap ($\rho_i^2 \rho_j^2$ energy).
- Adhesion is **not a free energy term** — it is a non-variational advection added directly to the equation of motion.
- The adhesion parameter $\kappa$ controls the collision outcome: elastic (low $\kappa$) vs. inelastic (high $\kappa$, cells stick).

**Najem & Grant (2016)** introduced a different adhesion mechanism. Each cell has two fields: a phase field $\phi_i$ and an auxiliary range field $C_i$ that defines its interaction neighborhood. $C_i$ is wider than $\phi_i$ and obeys its own dynamics. The adhesion energy involves:

$$F_\text{adh} = w \sum_i \int C_i \phi_i^2(1 - \phi_i) \sum_{j \neq i} (1 - \phi_j) \, dA$$

This has the structure: cell $i$'s cortex ($\phi_i^2(1 - \phi_i)$, localized at the interface) interacts with the exterior of cell $j$ ($1 - \phi_j$), but only within the range set by $C_i$. The parameter $w$ controls adhesion strength.

Key features:
- Adhesion has a controlled range set by $C_i$ (independent of the interface width).
- The adhesion term is interface-localized on cell $i$ but extends into the exterior of $j$.
- Najem & Grant showed that tissue surface tension scales linearly with their adhesion parameter: $T_\text{st}/\sigma = 2w/\sigma$.
- They tested $w/\sigma \in [0.1, 0.5]$, confirming a differential adhesion mechanism.

**Loewe, Chiang, Marenduzzo & Marchetti (2020)** used a multi-phase field model for deformable active particles. Their free energy uses the standard quartic repulsion:

$$F = \sum_i \left[ \int \frac{\alpha}{4} \phi_i^2(\phi_i - \phi_0)^2 + \frac{K}{2} |\nabla \phi_i|^2 \, d^2r + \lambda \left(1 - \frac{\int \phi_i^2 \, d^2r}{\pi R^2 \phi_0^2}\right)^2 + \varepsilon \sum_{j>i} \int \phi_i^2 \phi_j^2 \, d^2r \right]$$

There is **no adhesion term**; cells interact only through quartic repulsion ($\varepsilon$). The key control parameter is deformability $d = \varepsilon/\alpha$: at low $d$ cells overlap rather than deform, and at high $d$ they deform to tile space. They find:
- Continuous solid-liquid transition at high deformability (similar to vertex model).
- First-order-like transition at low deformability.
- An intermittent regime near the transition at low $d$.

This work, though it lacks adhesion, is important because it demonstrates that multi-phase field models produce solid-liquid transitions qualitatively similar to vertex models, establishing the framework.

**Wenzel & Voigt (2021)** systematically compared four multiphase field model variants. All four share the same passive free energy — a Cahn-Hilliard cell-shape term and a signed-distance-based short-range repulsive potential:

$$F_\text{INT} = \sum_i \frac{1}{\text{In}} \int B(\phi_i) \sum_{j \neq i} w(d_j) \, dx$$

where $B(\phi_i) = \frac{3}{\sqrt{2}} W(\phi_i)$ approximates a delta function at the interface and $w(d_j) = \exp(-d_j^2/2)$ is a Gaussian repulsion in terms of the signed distance $d_j$ to cell $j$'s boundary. This is repulsion only — no explicit adhesion. As they note, "most previous multiphase field models consider the interaction only effectively using terms proportional to $\phi_i^2 \phi_j^2$ for cell-cell repulsion and $|\nabla \phi_i|^2 |\nabla \phi_j|^2$ for cell-cell attraction."

The four variants differ entirely in how self-propulsion $\mathbf{v}_i$ enters the advection term $v_0(\mathbf{v}_i \cdot \nabla \phi_i)$:

1. **Random orientation** (following Loewe et al. 2020): constant propulsion speed $v_0$, direction $\theta_i$ evolving under rotational noise $d\theta_i = \sqrt{2D_r} \, dW_i$. Active Brownian particles generalized to deformable cells. No coupling between propulsion and cell shape or neighbors.
2. **Elongation-based**: propulsion direction and magnitude are set by the cell's own deformation. A Q-tensor $\mathbf{S}_i$ is computed from the gradient of each phase field, and the active velocity is $\mathbf{v}_i = \int \tilde{\phi}_i \nabla \cdot \mathbf{S} \, dx$. More elongated cells propel faster along their elongation axis. No subcellular dynamics, but neighbors influence propulsion indirectly through mutual deformation.
3. **Polar gel (contraction)**: each cell has a subcellular polarization field $\mathbf{P}_i$ obeying Frank-Oseen dynamics, with a unity constraint inside the cell and $\mathbf{P}_i = 0$ outside. The active stress is contractile: $\mathbf{v}_i \propto -\nabla \cdot (\mathbf{P}_i \otimes \mathbf{P}_i)$. Cell motility arises from spontaneous symmetry breaking of the polarization field.
4. **Polar gel (traction)**: same subcellular $\mathbf{P}_i$ field, but the active driving is $\mathbf{v}_i \propto \mathbf{P}_i |\nabla \phi_i|^2$, concentrating the force at the cell boundary (mimicking actin-driven traction at the leading edge). Based on the Ziebert-Aranson single-cell model extended to many cells.

The key finding is that these microscopic differences in activity produce qualitatively different behavior in solid-to-liquid transitions, nematic ordering, vorticity correlations, and topological defect dynamics, even though the passive energy is identical. Models 3 and 4 (with subcellular polarization) gave the best agreement with experimental MDCK data. This underscores that the choice of activity model matters as much as the interaction potential for collective tissue behavior.

**Graham, Zhang & Yeomans (2024)** studied cell sorting by active forces. Their model uses the free energy of Loewe et al. (quartic repulsion, no adhesion):

$$F = \sum_i F_i^\text{single} + \varepsilon \sum_{i<j} \int \phi_i^2 \phi_j^2 \, d^2r$$

Activity is modeled as extensile or contractile dipolar forcing. They demonstrate that differential activity alone can drive sorting without any thermodynamic adhesion mechanism.

**Saito & Ishihara (2024)** introduced the "Fourier contour cell model" (not a traditional phase field), in which each cell's boundary is expressed as a Fourier expansion of polar coordinates $R(\theta)$. Cells interact only through excluded volume repulsion and self-propulsion. They demonstrated a **fluid-to-fluid transition**: at moderate deformability cells behave like a vertex-model fluid (polygonal, with T1-like rearrangements), while at high deformability cells acquire round, overlapping shapes and the tissue enters a "soft fluid" phase characterized by percolation of topological defects. No adhesion term is included.

---

## 3. Summary: How Adhesion Has Been Modeled

| Reference | Repulsion | Adhesion type | Adhesion form | Range |
|---|---|---|---|---|
| Nonomura (2012) | $h(\phi_i) h(\phi_j)$ smooth step | Gradient overlap | $-\gamma \nabla h(\phi_i) \cdot \nabla h(\phi_j)$ | Interface-localized |
| Palmieri et al. (2015) | $\kappa \phi_i^2 \phi_j^2$ quartic | None | — | — |
| Löber et al. (2015) | $\lambda \rho_i^2 \rho_j^2$ quartic | Advection along normal | $-\kappa \nabla \rho_i \cdot \hat{f}(\nabla \rho_j)$ | Interface-localized (non-variational) |
| Najem & Grant (2016) | $\kappa \phi_i^2 \phi_j^2$ quartic | Range-field cortex | $w C_i \phi_i^2(1-\phi_i)(1-\phi_j)$ | Controlled by $C_i$ |
| Loewe et al. (2020) | $\varepsilon \phi_i^2 \phi_j^2$ quartic | None | — | — |
| Wenzel & Voigt (2021) | $B(\phi_i) w(d_j)$ signed-distance | None (repulsion only) | — | — |
| Graham et al. (2024) | $\varepsilon \phi_i^2 \phi_j^2$ quartic | None | — | — |
| Saito & Ishihara (2024) | Excluded volume (Fourier contour) | None | — | — |
| **This work** | $\kappa \phi_i^2 \phi_j^2$ quartic | **Bilinear overlap** | $-J \phi_i \phi_j$ | **Extended (bulk-like)** |

Key observations:
1. **Most recent models include no adhesion at all** — only quartic repulsion. (Loewe 2020, Wenzel 2021, Graham 2024, Saito 2024, Palmieri 2015.)
2. The models that do include adhesion use **interface-localized** forms: Nonomura's gradient overlap, Löber's advection along the normal, or Najem's range-field cortex coupling.
3. **No prior work uses the simple bilinear form $-J \phi_i \phi_j$ as an adhesion potential.** Löber uses $\rho_i \rho_j^2$ (from quartic energy) for repulsion. The variational derivative of a bilinear energy $\frac{J}{2} \int \phi_i \phi_j \, dA$ is simply $J \phi_j$, which is the simplest possible coupling.
4. The bilinear form has a longer interaction range than interface-localized terms: it is nonzero wherever both $\phi_i$ and $\phi_j$ are nonzero, extending into the cell bulk rather than only at the cortex.
5. **No phase field study has systematically scanned adhesion through a rigidity transition.** Najem & Grant varied adhesion to measure surface tension, but did not study jamming. Loewe et al. studied the solid-liquid transition but varied motility and deformability, not adhesion.

This last point is the central gap that our manuscript addresses.

---

## 4. Justification of the Bilinear Adhesion Term $-J\phi_i\phi_j$

### 4.1. The gap in the literature

As summarized in Section 3, the phase field literature splits into two camps: (a) models that include adhesion through interface-localized or non-variational couplings (Nonomura 2012, Löber 2015, Najem & Grant 2016), and (b) more recent models that include **no adhesion at all**, studying the solid-liquid transition through motility, deformability, or activity alone (Palmieri 2015, Loewe 2020, Wenzel & Voigt 2021, Graham 2024, Saito & Ishihara 2024). No multi-cell phase field study has systematically scanned adhesion strength through a rigidity transition. This gap motivates the introduction of a minimal adhesion term designed specifically for this purpose.

### 4.2. Landau expansion of pairwise coupling

The most general pairwise coupling between two scalar phase fields $\phi_i$ and $\phi_j$ can be expanded in powers of the fields:

$$F_\text{pair} = \int \left[ a_{11}\,\phi_i\phi_j + a_{21}\,\phi_i^2\phi_j + a_{12}\,\phi_i\phi_j^2 + a_{22}\,\phi_i^2\phi_j^2 + \cdots \right] dA$$

The existing literature already uses the **quartic** ($a_{22}$) term for repulsion: $+\kappa \int \phi_i^2\phi_j^2 \, dA$. The cubic terms ($a_{21}$, $a_{12}$) break the $\phi_i \leftrightarrow \phi_j$ exchange symmetry unless $a_{21} = a_{12}$, and even then they break the $\phi \to 1-\phi$ symmetry that distinguishes interior from exterior. The **bilinear** ($a_{11}$) term is the lowest-order coupling that:

1. Preserves the $i \leftrightarrow j$ exchange symmetry of the pair interaction
2. Provides attraction (with $a_{11} = -J < 0$) to complement the quartic repulsion
3. Has the simplest possible variational derivative: $\delta F_\text{adh}/\delta\phi_i = -J\phi_j$

The bilinear term is therefore the natural lowest-order attractive complement to the established quartic repulsion — the same relationship that a harmonic attraction has to a hard-core repulsion in molecular dynamics.

### 4.3. Pair-energy analysis: $J/\kappa$ as a control parameter

Consider the pairwise energy density at a point where two cells overlap with field values $\phi_i$ and $\phi_j$:

$$e(\phi_i, \phi_j) = \kappa\,\phi_i^2\phi_j^2 - J\,\phi_i\phi_j$$

Setting $\phi_i = \phi_j \equiv \phi$ (symmetric overlap) gives a one-dimensional energy:

$$e(\phi) = \kappa\,\phi^4 - J\,\phi^2$$

This is minimized at $\phi^2 = J/(2\kappa)$, with minimum energy $e_\text{min} = -J^2/(4\kappa)$. At $J = 0$, the minimum is at $\phi = 0$ (no overlap; hard repulsion). As $J/\kappa$ increases, cells settle into a progressively deeper overlap well:

| $J/\kappa$ | $\phi_\text{eq}$ | Interpretation |
|---|---|---|
| 0 | 0 | Pure repulsion, no cell-cell contact |
| 0.1 | 0.22 | Weak adhesion, shallow overlap |
| 0.5 | 0.50 | Moderate adhesion, significant overlap |
| 1.0 | 0.71 | Strong adhesion, deep overlap |
| >1 | — | Instability: adhesion overcomes repulsion → cell merger |

The ratio $J/\kappa$ thus controls the equilibrium overlap depth between neighbors, and the stability limit is $J/\kappa \lesssim 1$ (with corrections from the interface profile and the double-well potential).

### 4.4. Sharp-interface scaling: recovery of vertex model adhesion

For two cells in contact, each with a sharp interface of width $\epsilon$ and a $\tanh$-like profile, the overlap integral has the scaling:

$$\int \phi_i \phi_j \, dA \;\sim\; \epsilon \cdot \ell_{ij}$$

where $\ell_{ij}$ is the contact length between cells $i$ and $j$. This is because $\phi_i \phi_j$ is nonzero only in the strip of width $\sim\epsilon$ where both interface tails overlap. The bilinear adhesion energy therefore becomes:

$$F_\text{adh} = -J \sum_{i<j} \int \phi_i \phi_j \, dA \;\sim\; -J\epsilon \sum_{i<j} \ell_{ij}$$

In the vertex model, adhesion enters through a line tension $\Lambda_{ij}$ at shared edges, contributing $\sum_{i<j} \Lambda_{ij}\,\ell_{ij}$ to the energy. With adhesion reducing the effective line tension, this becomes $-\gamma \sum_{i<j} \ell_{ij}$. Therefore, in the sharp-interface limit, the bilinear phase field adhesion reduces to the vertex model adhesion with the identification $\gamma = J\epsilon$.

The gradient-based forms (Nonomura's $\nabla h(\phi_i) \cdot \nabla h(\phi_j)$, Löber's $\nabla\rho_i \cdot \hat{f}(\nabla\rho_j)$) achieve the same sharp-interface scaling more tightly localized at the interface. The bilinear form has a slightly longer effective range (it extends into the bulk wherever both fields are nonzero), which is arguably more physical for cadherin-mediated adhesion that acts across a finite intercellular gap, but both converge to the same contact-length proportionality in the limit $\epsilon/R \to 0$.

### 4.5. Connection to the vertex model shape index

In the vertex model, adhesion enters indirectly. The perimeter energy $K_P(P_i - P_0)^2$ contains a target $P_0$ that encodes line tension. Cell-cell adhesion reduces the effective line tension at shared edges, increasing $P_0$ and hence the target shape index $p_0 = P_0/\sqrt{A_0}$. The rigidity transition occurs at $p_0^* \approx 3.81$.

In our phase field model, the ratio $J/\kappa$ is the control parameter. The adhesion energy $-J \int \phi_i \phi_j \, dA$ rewards overlap between neighboring cells, favoring larger contact area. Larger contact area corresponds to more elongated, higher-perimeter cell shapes — the same geometric effect that raising $p_0$ produces in the vertex model. The sharp-interface analysis above (Section 4.4) confirms that the bilinear adhesion reduces to the vertex model contact-length energy, making this correspondence rigorous rather than merely analogical.

Therefore we argue that $J/\kappa$ is the phase field analog of $p_0$, and the rigidity transition in $J/\kappa$ should correspond to $\langle p_\text{eff} \rangle \approx 3.81$ as measured from phase field contours. The mapping $J/\kappa \to p_\text{eff}$ is a **result** of the simulation, not an assumption — we impose adhesion and measure the emergent shape index.

### 4.6. Why existing models chose the adhesion forms they did

Each adhesion mechanism in the literature was chosen for specific physical or computational reasons tied to the questions the authors were asking. Understanding these choices explains both why those forms are appropriate for their original contexts and why the bilinear term was never tried.

**Nonomura (2012): gradient-overlap as the cortex.**
Nonomura's model is built around the smooth step function $h(\phi) = \phi^2(3-2\phi)$, which maps the diffuse phase field to a sharp interior/exterior indicator (with $h(0)=0$, $h(1)=1$). The gradient $\nabla h(\phi)$ is then localized at the cell boundary — Nonomura explicitly interprets this as the cell **cortex**, writing that "the position of the membrane and/or the cortex of each cell" can be expressed through the phase field without extra variables. This is the paper's central selling point.

With this cortex interpretation established, the gradient coupling $\nabla h(\phi_i) \cdot \nabla h(\phi_j)$ is the natural adhesion term: it measures the overlap of two cortices, acting only where two cell boundaries are adjacent. Nonomura explicitly states (p.4 of the paper) that the interaction terms were chosen because they are "the simplest among the several alternatives, which can be written in terms of **w**" — the auxiliary sum field $w_\ell = \sum_m h(u_m) \delta_{\ell_m,\ell}$ that enables $O(ML)$ rather than $O(M^2)$ computation per grid point.

Why not the bilinear term? Because Nonomura's repulsion is already $h(\phi_i) h(\phi_j)$ — a bulk overlap term similar in character to $\phi_i \phi_j$. Adding a second bulk overlap term with opposite sign would simply reduce the effective repulsion strength, not create a qualitatively distinct cortex-localized adhesion. For Nonomura's purpose — demonstrating cell sorting via differential adhesion — the gradient form correctly implements the biological picture of surface-receptor-mediated adhesion.

**Löber, Ziebert & Aranson (2015): advection from a motility model.**
The Löber model is not derived from a free energy. It is a direct extension of the Ziebert-Aranson single-cell motility model, which carries detailed subcellular fields: actin polarization $\mathbf{p}$, adhesive bonds $A$, and substrate displacement $\mathbf{u}$. The dynamics are equation-of-motion-based, with each term representing a specific biophysical process.

The cell-cell adhesion term $-\kappa \nabla \rho_i \cdot \hat{f}(\nabla \rho_j)$ is described by Löber as: "the field $\rho_i$ is advected along the normal vector to the interface of cell $j$ with rate $\kappa$, which is equivalent to attraction between the cells." This advection form is the natural multi-cell extension of the single-cell model: it drives cell $i$'s boundary toward cell $j$'s boundary, mimicking the biological process of cadherin-mediated membrane–membrane attraction. It is non-variational by design, because the underlying single-cell model already contains non-variational terms (actin-driven motility, substrate friction).

The bilinear term $-J\phi_i\phi_j$ cannot appear in this framework at all — it requires a free energy, and the Löber model doesn't have one. More broadly, the Löber group was not studying adhesion as a control parameter; their interest was the collision outcomes (elastic vs. inelastic) between motile cells and how these produce collective migration.

**Najem & Grant (2016): range-field from continuum mechanics.**
Najem & Grant begin from an explicit continuum mechanics starting point. They write: "In a continuum mechanics formulation, the adhesion potential given by $W(x) = w\,e^{-d^2(x)/\delta^2}$ contributes $-\int_S W\,ds$ to the total energy functional, where $\delta$ determines its interaction range and $d(x)$ measures the distance separating the cells' interfaces."

This is the most physically motivated formulation. The auxiliary field $C_i$ is their translation of the distance-dependent adhesion potential into phase field language: $C_i$ defines the spatial range over which cell $i$ exerts adhesive influence, independent of the interface width $\epsilon$. The adhesion term $w C_i \phi_i^2(1-\phi_i) \sum_{j \neq i}(1-\phi_j)$ has the structure:
- $\phi_i^2(1-\phi_i)$: a bell-shaped function localized at cell $i$'s interface (zero at $\phi=0$ and $\phi=1$)
- $(1-\phi_j)$: nonzero outside cell $j$ — the empty space between cells
- $C_i$: defines the reach of cell $i$'s adhesion molecules

Najem & Grant's goal was to measure tissue surface tension as a function of the adhesion-to-cortical-tension ratio, verifying the differential adhesion hypothesis. For this, independently controllable range was essential. Their finding $T_\text{st}/\sigma = 2w/\sigma$ confirmed the DAH quantitatively.

The bilinear term $-J\phi_i\phi_j$ has no explicit range parameter — its effective range is set by the interface width $\epsilon$, making it unsuitable if one wants to separate adhesion range from interface structure. But this coupling to $\epsilon$ is not a drawback for studying the rigidity transition, where the relevant variable is adhesion *strength*, not range.

**Loewe et al. (2020), Wenzel & Voigt (2021), Graham et al. (2024): no adhesion.**
These groups omitted adhesion entirely because they were asking different questions. Loewe et al. studied the solid-liquid transition as a function of **deformability** ($d = \varepsilon/\alpha$) and motility (Pe), explicitly framing their model as a bridge between overlapping soft particles and space-filling vertex-like models. Wenzel & Voigt compared four **activity** mechanisms. Graham et al. asked whether **differential activity** alone can drive cell sorting without adhesion. In each case, adhesion was not the variable of interest, and including it would have complicated the parameter space.

### 4.7. Why the bilinear term was never tried

The reasons are both cultural and technical:

1. **Interface localization as a design principle.** The multi-cell phase field community inherits a strong tradition from materials science (Cahn-Hilliard, Allen-Cahn models) where the physics happens at interfaces. There is an implicit expectation that interaction terms should be interface-localized. The bilinear $\phi_i \phi_j$ violates this: it is nonzero wherever both fields are nonzero, including the cell bulk. Every group that included adhesion designed it to act at the cortex or membrane, consistent with the biology of cadherin-mediated surface adhesion.

2. **The quartic repulsion establishes the wrong precedent.** The standard quartic repulsion $\kappa \phi_i^2 \phi_j^2$ is *also* a bulk overlap term, and nobody objects to it. But when researchers think about adhesion, they immediately think "surface phenomenon" and reach for interface-localized forms. The asymmetry in treatment — bulk term acceptable for repulsion, required to be interface-localized for adhesion — is a cultural bias, not a physical necessity.

3. **Superficial resemblance to softened repulsion.** At first glance, adding $-J\phi_i \phi_j$ to $+\kappa \phi_i^2 \phi_j^2$ looks like it might simply soften the repulsion. The key distinction — that the bilinear term creates a finite-overlap *minimum* rather than just lowering a barrier — requires the pair-energy analysis of Section 4.3, which is simple but was not performed because nobody was looking at the term.

4. **Each group solved a different problem.** Nonomura wanted cell sorting. Löber wanted collision dynamics. Najem wanted tissue surface tension. Everyone after 2016 was focused on activity, deformability, or topology. Nobody was asking "what is the minimal adhesion term sufficient to produce a rigidity transition in the phase field framework?" — which is the question that leads naturally to the bilinear form.

5. **The obvious is the last to be tried.** The bilinear coupling is the simplest possible interaction between two scalar fields — the field-theoretic equivalent of $-Js_i s_j$. It is so simple that researchers may have assumed either that it had already been studied or that it was too crude to capture anything interesting. In fact, as the Landau expansion (Section 4.2), pair-energy analysis (Section 4.3), and sharp-interface scaling (Section 4.4) demonstrate, the bilinear term encodes exactly the right physics: it rewards contact area between cells, creates a tunable overlap well controlled by a single dimensionless parameter $J/\kappa$, and reduces to the vertex model adhesion energy in the sharp-interface limit.

### 4.8. Why the bilinear term is the right choice for this study

The bilinear form $-J\phi_i\phi_j$ is not intended to supersede the existing adhesion models. It is designed for a specific purpose: to answer, with minimal assumptions, whether an adhesion-controlled rigidity transition exists in the multi-cell phase field framework and whether it maps onto the vertex model.

For this purpose, the bilinear form has several decisive advantages:

1. **Single control parameter.** $J/\kappa$ is the only new parameter. Nonomura requires both $\gamma$ (adhesion coupling) and $c$ (regularization, with $c > \gamma$). Najem requires $w$ plus the dynamics of the auxiliary $C_i$ field.

2. **Variational.** Unlike Löber's advection, the bilinear term derives from a free energy. This means the zero-motility dynamics are purely relaxational (gradient descent), energy is well-defined, and the $J$-quench protocol has a clear thermodynamic interpretation.

3. **Analytically tractable.** The variational derivative is $-J\phi_j$ — linear and local. This makes stability analysis direct (Section 4.3), enables clean decomposition of energy contributions, and allows comparison with mean-field predictions.

4. **Correct sharp-interface limit.** It reduces to $-J\epsilon \sum \ell_{ij}$, the vertex model adhesion energy (Section 4.4).

5. **Same computational infrastructure.** The sum field $S_1 = \sum_k \phi_k$ is the only additional quantity needed, identical in structure to $S_2 = \sum_k \phi_k^2$ already used for repulsion. There is zero additional algorithmic complexity.

More biophysically detailed adhesion models (Najem's range-field, Nonomura's gradient coupling) are the natural follow-up once the basic phenomenology — whether the transition exists, where it occurs, and whether it maps onto the vertex model — is established with the minimal model.

---

## 5. Active Jamming in Particle and Agent-Based Models

Before the vertex model shape-index transition was discovered, the connection between biological tissue dynamics and jamming physics was explored using particle-based models. **Henkes, Fily & Marchetti (2011)** studied self-propelled soft repulsive disks at high density [DOI: 10.1103/PhysRevE.84.040301]. They found an active jammed phase at high packing $\phi$ and low propulsion $v_0$, with glassy dynamics and displacement fields resembling those seen in confluent monolayer experiments (Angelini et al., 2011). The polarity of each disk aligns with its instantaneous velocity via a lag-time coupling, producing run-and-persist-style dynamics. Unlike the vertex model, cells are circular and non-deformable, so the transition is a conventional density-dependent jamming transition shifted by activity. This work was foundational in establishing the conceptual link between active matter physics and tissue jamming.

**Chiang & Marenduzzo (2016)** studied glass transitions in the cellular Potts model (CPM) [DOI: 10.1209/0295-5075/116/28009], the lattice-based predecessor of continuous phase field methods. They found glassy arrest at high density with a crossover controlled by the ratio of cell-cell adhesion to cell-medium adhesion. The CPM captures deformable cell shapes and intrinsically includes differential adhesion, but its Monte Carlo dynamics lack a clear physical time scale.

---

## 6. Topological Defects, Active Nematics, and Tissue Mechanics

Epithelial monolayers exhibit features of active nematic liquid crystals, including topological defects in the cell alignment field. **Saw et al. (2017)** [DOI: 10.1038/nature21718] showed that comet-shaped ($+1/2$) defects in MDCK monolayers are sites of cell extrusion and death, governed by the compressive stress environment at the defect core. **Mueller & Yeomans (2019)** [DOI: 10.1103/PhysRevLett.122.048004] showed using a multi-phase field model (similar to Wenzel & Voigt) that active nematic behaviour emerges from isotropic cells purely through a feedback between cell shape deformation and dipolar active driving. Their model uses quartic repulsion and elongation-dependent activity, with no adhesion term. At high activity, topological defects proliferate and the system enters an active turbulent state. **Lin & Zhang (2021)** [DOI: 10.1038/s42005-021-00530-6] analyzed the energetics of this mesoscale turbulence in multi-phase field monolayers.

These works are relevant because topological defects provide an alternative structural characterization of the solid-liquid transition. In our adhesion study, tracking defect proliferation as a function of $J/\kappa$ could complement the MSD-based jamming classification.

---

## 7. The Geometric Frustration Picture

**Moshe, Bowick & Marchetti (2018)** [DOI: 10.1103/PhysRevLett.120.268105] developed a continuum elasticity limit of the vertex model and showed that the rigidity transition at $p_0^* \approx 3.81$ can be understood as the onset of **geometric frustration**: when the target area and target perimeter become geometrically incompatible (no polygon can simultaneously satisfy both constraints), the tissue acquires a finite shear modulus. Below $p_0^*$, the target shape is compatible and the tissue has a continuous family of zero-energy ground states (zero modes), making it mechanically floppy. This picture provides a geometric explanation for why the transition occurs at the specific value $3.81$ (the shape index of a regular pentagon, the most symmetric polygon that tiles the plane with 6-fold coordination).

This is directly relevant to our work: the bilinear adhesion $-J\phi_i\phi_j$ changes the effective preferred contact area between cells, which in the vertex model language corresponds to modifying the target perimeter. The geometric frustration framework predicts that once $J/\kappa$ pushes the effective shape index past compatibility, the tissue becomes floppy.

---

## 8. Unjamming vs. Epithelial-to-Mesenchymal Transition

**Mitchel et al. (2020)** [DOI: 10.1038/s41467-020-18841-7] demonstrated experimentally that the unjamming transition (UJT) and the partial epithelial-to-mesenchymal transition (pEMT) are **distinct** biological programs in primary human bronchial epithelial cells. After triggering UJT, cell-cell junctions and apico-basal polarity remain intact while cells elongate and migrate cooperatively; mesenchymal markers do not appear. After triggering pEMT, junctions disassemble and mesenchymal markers emerge. A vertex-model-based computational analysis attributes UJT mainly to augmented cellular propulsion (increasing $v_0$) and pEMT mainly to diminished junctional tension (increasing $p_0$).

This distinction is important for our work: our adhesion parameter $J/\kappa$ maps onto the junctional tension axis ($p_0$), meaning it probes the pEMT-like route to fluidity rather than the motility-driven UJT route. In Phase 2 of our study, where we scan both $J/\kappa$ and $v_A$, we will be able to distinguish which axis dominates fluidization.

---

## 9. Experimental Measurements of Tissue Surface Tension

The differential adhesion hypothesis (Steinberg; Foty & Steinberg, 2005) posits that cell-cell adhesion governs tissue-scale surface tension and drives sorting. **Mongera et al. (2018)** [DOI: 10.1038/s41586-018-0479-2] directly measured tissue surface tension along the zebrafish body axis, finding a gradient that correlates with the fluid-to-solid transition. **Malinverno et al. (2017)** [DOI: 10.1038/nmat4848] showed that Rab5-mediated endocytic reawakening fluidizes MDCK-II monolayers without changing cadherin expression, indicating that the endocytic pathway modulates effective adhesion at the cell surface. **Garcia et al. (2015)** [DOI: 10.1073/pnas.1510973112] showed that active jamming in expanding MDCK monolayers is controlled by contact inhibition of locomotion. These experiments establish that adhesion modulation, along multiple biochemical pathways, is a primary control variable for tissue fluidity — the phenomenon our simulations aim to model.

---

## 10. Reviews

- **Camley & Rappel (2017)** [DOI: 10.1088/1361-6463/aa56fe]: Comprehensive review of physical models for collective cell motility, covering particle models, continuum active gel, vertex, Potts, and phase field approaches.
- **Alert & Trepat (2020)** [DOI: 10.1146/annurev-conmatphys-031218-013516]: Review of physical models of collective cell migration, emphasizing the connection between cell-level mechanics and tissue-scale flows.
- **Alt, Ganguly & Salbreux (2017)** [DOI: 10.1098/rstb.2015.0520]: Review of vertex models, covering their derivation from continuum mechanics, the role of line tension and adhesion, and extensions to active tissues.
- **Lenne & Trivedi (2022)** [DOI: 10.1038/s41467-022-28151-9]: Perspective on phase transitions in biological tissues, connecting thermodynamic phase separation, rigidity transitions, and percolation phenomena.

---

*Last updated: 2026-02-15*
