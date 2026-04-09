# Research Agent Instructions: Phase Field Model Studies of Tissue Mechanics

> **When to consult this file:** You are exploring research questions, designing experiments, analyzing results in the context of tissue mechanics literature, or connecting simulation output to physical interpretations. This file has no `applyTo` scope — it applies to any research-oriented task across the codebase. For simulation mechanics or CLI, see [cell-simulation.instructions.md](cell-simulation.instructions.md). For the adhesion-specific study, see [adhesion-study.instructions.md](adhesion-study.instructions.md).

## Mission Statement

You are assisting with computational research on the glass/jamming transition in biological tissues using a **phase field model** (PFM). Your role is to help identify novel results that fill gaps in the existing literature, which has primarily relied on **vertex models**. The phase field approach offers unique advantages that can address questions the vertex model cannot answer well.

---

## Essential Background

### Required Reading

Before proceeding, familiarize yourself with the theoretical background and literature:

📖 **Primary References:**
- `cpp/simulation/AGENT_ONBOARDING.md` — Physics model, equations of motion, parameter definitions
- `cpp/simulation/RUNBOOK.md` — Operational runbook
- Key papers listed in the [Key References](#key-references-quick-links) section below

### The Research Gap

The vertex model literature (Bi et al. 2015, 2016; Barton et al. 2017) has established:
- The shape index $p_0 \approx 3.81$ controls the solid-fluid transition
- Cell motility can drive unjamming
- T1 transitions (neighbor exchanges) are the elementary rearrangement events

**However, vertex models have fundamental limitations:**

| Vertex Model Limitation | Phase Field Model Advantage |
|------------------------|----------------------------|
| Cells have straight edges | Cells have realistic curved boundaries |
| Discrete topology changes (T1s) | Continuous interface evolution |
| Assumes confluent tissue (no gaps) | Naturally handles non-confluent states |
| Cell shape is polygon-based | Cell shape emerges from field dynamics |
| Adhesion is a line tension parameter | Adhesion arises from field interactions |
| Motility is an external force | Motility can couple to internal dynamics |
| Fixed cell number between divisions | Cells can smoothly merge/split |

---

## The Phase Field Model

### Model Overview

Our simulation represents each cell $i$ as a continuous **phase field** $\phi_i(\mathbf{r}, t)$ where:
- $\phi_i \approx 1$ inside cell $i$
- $\phi_i \approx 0$ outside cell $i$
- Smooth interface of width $\epsilon$ between cells

The total phase field $\phi = \sum_i \phi_i$ represents local cell density.

### Energy Functional

The system evolves to minimize a free energy functional:

$$F = \int d\mathbf{r} \left[ \sum_i f_{CH}(\phi_i) + \sum_{i<j} f_{int}(\phi_i, \phi_j) + f_{bulk}(\phi) \right]$$

Where:
- $f_{CH}$: Cahn-Hilliard term (interface energy, cell shape)
- $f_{int}$: Cell-cell interaction — **repulsion** ($\kappa \sum \phi_i^2 \phi_j^2$) and **adhesion** (gradient coupling, enabled via `--adhesion J`)
- $f_{bulk}$: Bulk constraint (area/volume conservation)

**Adhesion term (implemented):** $F_{\text{adh}} = J \sum_{i<j} \int \nabla\phi_i \cdot \nabla\phi_j \, dA$ (gradient coupling). The variational derivative $\delta F/\delta \phi_i = -J(\nabla^2 S - \nabla^2\phi_i)$ where $S = \sum_k\phi_k$, computed via a global sum field and five-point Laplacian stencil. This is surface-localized, variational, and has a stability bound $J < 2\gamma$. See [adhesion-study.instructions.md](adhesion-study.instructions.md) for full physics. When `--adhesion` is not specified (J=0), no extra memory is allocated and zero overhead is incurred.

### Dynamics

Each phase field evolves according to:

$$\frac{\partial \phi_i}{\partial t} = -M \frac{\delta F}{\delta \phi_i} + \mathbf{v}_i \cdot \nabla \phi_i + \eta_i$$

Where:
- $M$: mobility coefficient
- $\mathbf{v}_i$: active velocity (self-propulsion)
- $\eta_i$: noise term

### Key Parameters

| Parameter | Symbol | Physical Meaning |
|-----------|--------|------------------|
| Motility | `vA` | Self-propulsion speed |
| Packing fraction | `phi` | Area fraction occupied by cells |
| Interface width | `epsilon` / `lambda` | Cell boundary sharpness |
| Gradient energy | `gamma` | Interfacial stiffness / surface tension |
| Repulsion | `kappa` | Cell-cell repulsion strength |
| Volume constraint | `mu` | Resistance to area changes |
| Friction | `xi` | Friction coefficient (dissipation rate) |
| Adhesion strength | `J` | Cell-cell attraction (`--adhesion J`; 0=disabled, no overhead) |
| Number of cells | `N` | System size |
| Box size | `L` | Determines effective density |

### Parameter Sets

Two parameter calibrations exist in the codebase. **Each study specifies which set it uses.**

- **Palmieri (2015):** The binary defaults. No parameter overrides needed. Run `cell_sim -h` to see current values.
- **Bresler (2018):** Reparametrised for the sharp-interface limit. Requires overrides: `gamma=3.75`, `mu=0.5`, `xi=1000`. All other parameters use binary defaults.

The Bresler calibration reparametrised $\gamma$, $\mu$, $\xi$ for the sharp-interface limit. The two sets produce **different physics** and are not interchangeable. Check the study-specific instructions for which to use.

---

## Research Questions to Explore

### Priority 1: Novel Results (Vertex Model Gaps)

These questions **cannot be answered** by vertex models and represent the highest-impact research directions:

#### 1.1 Cell Shape Fluctuations at the Transition

**Question:** How do cell boundary fluctuations (membrane undulations) change across the jamming transition?

**Why vertex models can't answer:** Vertex models have straight edges with no fluctuations. The shape index $p$ is computed from polygon geometry, missing dynamic boundary fluctuations.

**What to measure:**
- Interface roughness: $\langle (\nabla \phi)^2 \rangle$
- Fluctuation spectrum: $S(k) = \langle |\hat{\phi}(k)|^2 \rangle$
- Temporal correlations of boundary position

**Hypothesis:** Boundary fluctuations may diverge at the transition (critical fluctuations) or show qualitatively different scaling in solid vs. fluid phases.

#### 1.2 Non-Confluent Jamming

**Question:** How does the jamming transition change when tissues are not fully confluent (gaps exist between cells)?

**Why vertex models can't answer:** Vertex models assume 100% confluence by construction. They cannot represent gaps.

**What to measure:**
- Jamming transition as function of packing fraction $\phi < 1$
- Gap size distribution $P(A_{gap})$
- Percolation of cell-free regions
- Compare $\phi_c$ (critical packing) vs. $p_0^*$ (shape index transition)

**Hypothesis:** There may be a second jamming transition at intermediate $\phi$ distinct from the confluent $p_0^*$ transition.

#### 1.3 Cell Overlap and Compression

**Question:** What happens when cells are forced to overlap (high compression)?

**Why vertex models can't answer:** Vertex models assume cells tile the plane with no overlap—this is geometrically enforced.

**What to measure:**
- Overlap integral: $O = \int \phi_i \phi_j \, d\mathbf{r}$ for neighboring cells
- Stress under compression
- Transition from overlap-dominated to shape-dominated regime

**Hypothesis:** High compression may create a distinct "squeezed" phase not accessible to vertex models.

#### 1.4 Continuous vs. Discrete Rearrangements

**Question:** Are cell rearrangements truly discrete "T1 events" or continuous processes?

**Why vertex models can't answer:** Vertex models enforce discrete topology—edges either exist or don't. T1s are instantaneous by construction.

**What to measure:**
- Time evolution of neighbor contact area during rearrangements
- Distribution of rearrangement timescales
- Intermediate states during "T1-like" events

**Hypothesis:** Rearrangements may be continuous, with the discrete T1 picture being an artifact of the vertex model discretization.

#### 1.5 Interface Mechanics and Adhesion Gradients

**Question:** How does spatially varying adhesion affect tissue mechanics?

**Why vertex models can't answer:** Vertex models have a single line tension $\Lambda$ per edge. They cannot represent adhesion gradients along a single interface.

**Status:** Uniform adhesion $J$ is now implemented (`--adhesion` flag). Spatially varying adhesion would require per-cell $J_i$ values (future extension).

**What to measure:**
- Response to adhesion gradients
- Cell sorting with continuous adhesion variation
- Interface width changes with local adhesion

---

### Priority 2: Validate/Extend Vertex Model Results

These questions test whether vertex model predictions hold in a more realistic model:

#### 2.1 Shape Index Transition

**Question:** Does the $p_0^* \approx 3.81$ transition survive in phase field models?

**What to measure:**
- Compute effective shape index: $p = P/\sqrt{A}$ from phase field contours
- Identify solid-fluid transition point
- Compare to vertex model critical value

**Expected outcome:** May find a shifted or broadened transition due to interface fluctuations.

#### 2.2 Motility-Driven Unjamming

**Question:** Does increasing motility $v_A$ fluidize jammed tissues as predicted?

**What to measure:**
- MSD as function of $v_A$ at fixed packing
- Transition line in $(v_A, \phi)$ or $(v_A, p)$ space
- Compare to Bi et al. 2016 phase diagram

#### 2.3 Dynamic Heterogeneity

**Question:** Do phase field tissues exhibit the same dynamic heterogeneity (fast/slow regions) as vertex models?

**What to measure:**
- Four-point susceptibility $\chi_4(t)$
- Spatial correlation of mobility
- Non-Gaussian parameter of displacements

---

### Priority 3: 3D-Specific Questions

The simulation supports 3D. These questions are uniquely accessible:

#### 3.1 Surface vs. Bulk Dynamics

**Question:** How do cells at tissue surfaces behave differently from bulk cells?

**What to measure:**
- Layer-resolved MSD
- Surface cell shape vs. bulk cell shape
- T1 rate gradient from surface to bulk

#### 3.2 3D Cell Shape and Jamming

**Question:** What is the 3D analog of the shape index transition?

**What to measure:**
- 3D shape metrics: surface area to volume ratio $s = S/V^{2/3}$
- Asphericity, acylindricity
- Identify 3D critical shape

---

## Simulation Infrastructure

### Code Location

```
cpp/simulation/
├── include/           # Header files
│   ├── cell.cuh       # 2D cell operations
│   ├── cell3d.cuh     # 3D cell operations
│   ├── simulation.cuh # Main simulation class
│   └── ...
├── src/               # CUDA source files
│   ├── main.cu        # Entry point
│   ├── kernels_solver.cu   # Production GPU solver kernels
│   ├── kernels_shared.cu   # Shared helper kernels
│   ├── kernels3d.cu        # 3D kernel implementations
│   └── ...
├── cluster/           # HPC job management
│   └── ...
└── postprocessing/    # Visualization & analysis tools
```

### Running Simulations

**On the cluster (nibi.alliancecan.ca):**

All cluster operations go through the **compute-canada MCP tool**:
- `connect` — establish SSH session
- `start_simulation` — submit a fresh simulation
- `resume_simulation` — continue from checkpoint
- `list_jobs` / `check_progress` — monitor status

**Do NOT use `run_command` with `sbatch` or legacy submission scripts.** The MCP tools handle GPU selection, SLURM accounts, job chaining, and validation automatically.

**Key directories:**
- Production: `/scratch/ssilber/jamming_study/production/vA_*/run_*/`
- Equilibration: `/scratch/ssilber/eq_phi*/run_*/`

### Output Format

Simulations output VTK files containing:
- Phase field values $\phi_i$ at each grid point
- Cell centroids
- Local density

Analysis scripts can compute:
- Cell shapes (from contours)
- Mean squared displacement
- Neighbor lists
- Order parameters

---

## Analysis Guidance

### Key Observables

| Observable | Definition | What it reveals |
|------------|------------|-----------------|
| MSD | $\langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle$ | Diffusive vs. caged dynamics |
| Shape index | $p = P/\sqrt{A}$ | Cell geometry |
| Overlap function | $Q(t) = \langle \theta(a - |\mathbf{r}_i(t) - \mathbf{r}_i(0)|) \rangle$ | Structural relaxation |
| T1 rate | Number of neighbor changes per time | Fluidity |
| Interface width | $w = \int |\nabla \phi|^{-1} d\ell$ | Adhesion/mechanics |
| Non-Gaussianity | $\alpha_2 = \langle r^4 \rangle / (d+2)\langle r^2 \rangle^2 - 1$ | Dynamic heterogeneity |

### Phase Diagram Exploration

The primary phase diagram axes are:
1. **Motility** $v_A$: 0.004 to 0.013 (10 values in production)
2. **Packing fraction** $\phi$: 0.85 and 0.89 currently

**Suggested extensions:**
- Lower $\phi$ (0.6-0.8) to study non-confluent regime
- Higher $v_A$ to find full fluidization
- Vary adhesion strength to probe different physics

### Statistical Requirements

For publication-quality results:
- **Replicas:** 100 independent runs per parameter set (current setup)
- **Equilibration:** Verify energy plateau before production
- **Time series:** Long enough to observe diffusive regime (MSD linear in $t$)
- **System size:** N=288 is modest; consider finite-size checks at N=576, N=1152

---

## Workflow for Novel Results

### Step 1: Identify the Question

Choose a research question from Priority 1 (vertex model gaps). Be specific about:
- What observable will you measure?
- What parameter range will you explore?
- What is the null hypothesis (vertex model prediction)?

### Step 2: Design the Simulation

Determine required:
- Parameter values
- System size
- Run duration
- Number of replicas

### Step 3: Run Production

Submit jobs using the `start_simulation` or `resume_simulation` MCP tools. Monitor with `list_jobs` and `check_progress`.

### Step 4: Analyze Data

Write analysis scripts to compute observables. Use:
- Python with numpy, scipy
- VTK file readers (pyvista, vtk)
- Visualization (matplotlib, pyvista)

### Step 5: Compare to Literature

Reference the papers in the [Key References](#key-references-quick-links) section:
- How do your results compare to vertex model predictions?
- What is genuinely new?
- What is the physical interpretation?

### Step 6: Document Findings

Record:
- Parameter values used
- Raw data location
- Analysis scripts
- Key figures
- Physical interpretation

---

## Communication Guidelines

### When Reporting Results

Always specify:
1. **Parameter values:** $v_A$, $\phi$, $N$, $L$, and any others varied
2. **Statistics:** Number of replicas, run duration, equilibration time
3. **Comparison:** How does this relate to vertex model literature?
4. **Novelty:** What can phase field models reveal that vertex models cannot?

### Terminology

Use consistent terminology:
- "Jammed" = solid-like, non-diffusive, caged
- "Unjammed" = fluid-like, diffusive, freely rearranging
- "Confluent" = no gaps between cells ($\phi \approx 1$)
- "Shape index" = $p = P/\sqrt{A}$ (measured) or $p_0$ (target, in vertex models)
- "T1 transition" = neighbor exchange event

### Figures to Generate

Key figures for any study:
1. **Phase diagram** showing jammed/unjammed regions
2. **MSD curves** showing caging plateau (jammed) vs. linear growth (unjammed)
3. **Shape distributions** $P(p)$ in different phases
4. **Snapshots** showing cell configurations

---

## Current State of Production Runs

As of February 2026, three concurrent studies are running:

| Study | Status | Instruction file |
|-------|--------|------------------|
| **Adhesion** | Phases 0-2 in progress (Bresler params, $\rho = 0.89$, N=288) | `adhesion-study.instructions.md` |
| **Griffiths** | Production runs on cluster (Bresler params, $\rho = 0.89$, N=288) | `griffiths-study.instructions.md` |
| **Palmieri extension** | Equilibration campaign starting (Palmieri params, multiple $\rho$, N=100–12800) | `palmieri-extension.instructions.md` |

Check the study-specific LOG_BOOK.md files for current status. Use `list_jobs` and `check_progress` MCP tools for cluster status.

---

## Key References Quick Links

| Paper | Key Result | DOI |
|-------|-----------|-----|
| Bi 2015 | Shape index transition $p_0^* = 3.81$ | [10.1038/nphys3471](https://doi.org/10.1038/nphys3471) |
| Bi 2016 | Motility-driven unjamming | [10.1103/PhysRevX.6.021011](https://doi.org/10.1103/PhysRevX.6.021011) |
| Barton 2017 | Active Vertex Model | [10.1371/journal.pcbi.1005569](https://doi.org/10.1371/journal.pcbi.1005569) |
| Czajkowski 2019 | Cell division/death effects | [10.1039/c9sm00916g](https://doi.org/10.1039/c9sm00916g) |
| Giavazzi 2018 | Flocking transitions | [10.1039/c8sm00126j](https://doi.org/10.1039/c8sm00126j) |

**Full summaries:** See `cpp/simulation/AGENT_ONBOARDING.md` and the papers listed in [Key References](#key-references-quick-links).

---

## Summary: Your Research Mission

1. **Understand the background** by reading `cpp/simulation/AGENT_ONBOARDING.md`
2. **Identify vertex model gaps** from the Priority 1 questions above
3. **Design simulations** that probe these gaps using phase field advantages
4. **Analyze results** with appropriate observables
5. **Compare to literature** to establish novelty
6. **Document findings** for publication

The goal is to produce results that are:
- **Novel:** Only accessible to phase field models
- **Significant:** Address important open questions in tissue mechanics
- **Rigorous:** Well-controlled, statistically sound, physically interpretable

Good luck with the research!

---

*Last updated: January 15, 2026*
