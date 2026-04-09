---
applyTo: "cpp/simulation/study/griffiths/**"
---

# Griffiths Rare-Region Study — Agent Instructions

> **When to consult this file:** You are running, analyzing, designing experiments, or writing for the Griffiths rare-region / quenched motility disorder study. This covers the physics background, experimental protocol, analysis pipeline, cluster execution, interpretation framework, and the manuscript. For general simulation CLI or builds, see [cell-simulation.instructions.md](cell-simulation.instructions.md). For cluster operations, see [cluster-operations.instructions.md](cluster-operations.instructions.md).

---

## Related Files

| File | Purpose |
|------|---------|
| `cpp/simulation/study/griffiths/LOG_BOOK.md` | Primary logbook — all results, decisions, progress |
| `cpp/simulation/study/griffiths/LITERATURE_REVIEW.md` | Comprehensive literature review |
| `cpp/simulation/study/griffiths/manuscript.tex` | PRE manuscript (in progress) |
| `cpp/simulation/study/griffiths/REVIEW_PROMPT.md` | Self-review guidance for the manuscript |
| `cpp/simulation/study/griffiths/analyze_griffiths.py` | Basic trajectory analysis |
| `cpp/simulation/study/griffiths/analyze_griffiths_batch.py` | Batch analysis across parameter sets |
| `cpp/simulation/study/griffiths/analyze_griffiths_deep.py` | Deep analysis: Q(t) fits, χ₄, v_A correlation |

## Related Instruction Files

| Task | Instruction File |
|------|-----------------|
| Building & running simulations | [cell-simulation.instructions.md](cell-simulation.instructions.md) |
| Cluster operations & job submission | [cluster-operations.instructions.md](cluster-operations.instructions.md) |
| Post-processing & visualization | [postprocessing.instructions.md](postprocessing.instructions.md) |
| Adhesion study (companion) | [adhesion-study.instructions.md](adhesion-study.instructions.md) |

---

## Study Overview

**Title:** Absence of Griffiths rare-region effects in a phase field model of active tissue

**Goal:** Test whether quenched cell-to-cell motility disorder near the jamming transition produces Griffiths rare-region physics (anomalous power-law relaxation from frozen rare regions) or qualitatively different active-matter behavior.

**Hypothesis under test:** Griffiths rare-region physics predicts that quenched disorder creates frozen rare regions with anomalous power-law relaxation. The alternative — which preliminary data supports — is a "stirred glass" mechanism where fast cells act as local mechanical stirrers that break up collective caging, producing anti-Griffiths behavior (disorder fluidizes rather than freezing).

**Target journal:** Physical Review E

---

## Physics Background

### The Griffiths Hypothesis

In equilibrium statistical mechanics, quenched disorder near a phase transition produces **Griffiths rare-region effects** (Griffiths 1969, Vojta 2006):

1. **Rare spatial regions** that are locally on the other side of the transition dominate long-time dynamics
2. The self-overlap $Q(t)$ develops **power-law tails** (or stretched exponential with $\beta \to 0$)
3. The four-point susceptibility $\chi_4$ **grows** with disorder (larger cooperatively rearranging regions)
4. The structural relaxation time $\tau_\alpha$ **increases** with disorder

These effects arise from the competition between exponentially rare clusters ($P(\ell) \sim e^{-c\ell^d}$) and exponentially slow relaxation ($\tau(\ell) \sim e^{b\ell^{d-1}}$).

### Why It Might Not Apply to Active Tissue

Active tissues violate the three Griffiths assumptions:

1. **Frozen, passive impurities → active force generators.** In the Ising model, vacant sites are inert. In active tissue, fast cells mechanically push their neighbors out of cages.

2. **Independent rare regions → mechanically coupled.** Cells generate propagating perturbations through the confluent tissue, coupling rare regions and invalidating the independent-relaxation picture.

3. **Dominance of slow modes → fast cells destroy slow modes.** In Griffiths systems, the slowest clusters dominate late-time dynamics. In active tissue, fast cells actively erode neighboring slow regions.

### The Stirred Glass Mechanism

In the clean system ($\sigma = 0$), cage escape requires collective coordination: cell A can't escape unless neighbor B yields, requiring C to move, etc. This produces strong cooperativity (high $\chi_4$), broad relaxation spectrum (low $\beta$), and long $\tau_\alpha$.

With disorder ($\sigma > 0$), fast cells break cages *without cooperative assistance* and mechanically disrupt neighbors. This short-circuits collective cage-breaking, producing reduced cooperativity (low $\chi_4$), narrower spectrum (high $\beta$), and faster relaxation (low $\tau_\alpha$).

**Testable prediction:** The effect should be strongest when the clean system sits near the optimal cage-scanning point $l_p \approx l_c$ (Debets framework). Experiment B tests this — check whether fluidization peaks at intermediate $\bar{v}_A$ and diminishes at both higher and lower values.

### Connection to the Cage-Length Framework

Debets et al. (2021, PRL 127, 278002) showed that the ratio $l_p/l_c$ (persistence length to cage length) controls active glass dynamics:

- $l_p < l_c$: enhanced but passive-like relaxation
- $l_p \approx l_c$: optimal fluidization
- $l_p > l_c$: dynamics slows, Stokes-Einstein breaks down

With quenched disorder, each cell has its own $l_{p,i} = v_{A,i}\tau$ while sharing approximately the same $l_c$. Cells in the high-$v_A$ tail sit near optimal cage scanning and transmit the effect mechanically to neighbors.

---

## Model

### Phase Field Equations

Each cell $i$ has a continuous scalar field $\phi_i(\mathbf{r},t)$. The equation of motion is:

$$\frac{\partial\phi_i}{\partial t} = -M\frac{\delta F}{\delta\phi_i} - \mathbf{v}_i \cdot \nabla\phi_i$$

where $M = 0.5$ is the mobility, $\mathbf{v}_i = v_{A,i}\hat{p}_i$ is the self-propulsion velocity, and $F$ is the free energy:

$$F = \sum_i \int \left[\gamma|\nabla\phi_i|^2 + \frac{30\gamma}{\lambda^2}\phi_i^2(1-\phi_i)^2\right] dA + \mu\sum_i(V_i - A_0)^2 + \kappa\sum_{i<j}\int\phi_i^2\phi_j^2\,dA$$

**Note on advection sign:** The term $-\mathbf{v}\cdot\nabla\phi$ translates the cell in the direction of $\mathbf{v}$. This matches the kernel implementation (`dphi_dt = -0.5*var_deriv - advection` where `advection = v·∇φ`).

### Quenched Motility Disorder

Per-cell motilities are drawn from a **log-normal** distribution:

$$\ln v_{A,i} \sim \mathcal{N}(\mu_{\ln},\, \sigma_{\ln}^2)$$

where $ \sigma_{\ln} = \sqrt{\ln(1 + \sigma^2/\bar{v}_A^2)}$ and $\mu_{\ln} = \ln\bar{v}_A - \sigma_{\ln}^2/2$ ensure mean $\bar{v}_A$ and std $\sigma$. The log-normal guarantees $v_{A,i} > 0$ without truncation.

- For $\sigma/\bar{v}_A \lesssim 0.75$: nearly symmetric, close to Gaussian
- For $\sigma/\bar{v}_A = 1$: noticeably right-skewed, enriched fast-cell tail

Once drawn at $t = 0$, the $\{v_{A,i}\}$ are held fixed for the entire simulation.

**Implementation:** `integrator.cu` lines 750–770. Per-cell values are stored in the checkpoint (v4 format) and restored on resume.

---

## Experimental Design

### Parameter Space

**Experiment A — Disorder sweep** (fixed $\bar{v}_A = 0.008$, near clean transition):

| $\sigma$ | $\sigma/\bar{v}_A$ | Physical regime |
|----------|---------------------|-----------------|
| 0.000 | 0 | **Control** (clean, mandatory) |
| 0.003 | 0.375 | Weak disorder |
| 0.006 | 0.75 | Moderate disorder |
| 0.008 | 1.0 | Strong disorder (extreme run-to-run variability) |

**Experiment B — Motility sweep** (fixed $\sigma = 0.006$):

| $\bar{v}_A$ | Regime | Purpose |
|-------------|--------|---------|
| 0.006 | Below clean transition | More jammed |
| 0.008 | Near clean transition | Optimal cage scanning |
| 0.010 | Above clean transition | More fluid |

**Total:** 6 parameter combinations × 3 replicates = 18 simulations.

### Fixed Parameters (Bresler 2018 calibration)

This study uses **Bresler parameters**. The non-default overrides that must be specified when submitting:

| Parameter | Value | Notes |
|-----------|-------|-------|
| $\gamma$ | **3.75** | Gradient energy — Bresler "hard" cell stiffness |
| $\mu$ | **0.5** | Volume penalty — Bresler value |
| $\xi$ | **1000** | Friction — Bresler value |

All other physics parameters ($\kappa$, $\tau$, $dt$, $R$, $\lambda$, $M$) use **binary defaults** — no overrides needed. Run `cell_sim -h` for current default values.

Geometry: $N = 288$, $L = 1562$ ($\phi \approx 0.89$). Production duration: 250,000 ($25\tau$). Equilibration: 80,000 ($8\tau$).

> **Parameter history:** Earlier runs (before Feb 20, 2026) used Palmieri defaults (γ=1, μ=1, ξ=1500). These produced cells that were too soft to jam at ρ=0.89. The switch to Bresler 2018 parameters (γ=3.75, μ=0.5, ξ=1000) produces stiff cells with a clean-system jamming transition at v*≈0.009 at ρ=0.89, matching Bresler Fig 2. **All old soft-cell data should be considered invalid for this study.**
>
> Cluster data with correct parameters lives under `/scratch/ssilber/griffiths_stiff/` (not the old `eq_phi89/` or `griffiths_study/` directories).

**Packing fraction:** The primary study uses $\phi \approx 0.89$ ($L = 1562$), which places the system near the Bresler jamming transition. A lower-density $\phi \approx 0.85$ ($L = 1600$) comparison is not currently planned — the system is too fluid at that density with the Bresler parameters to show interesting jamming physics.

### Mandatory Controls

**Every experiment must include $\sigma = 0$ with identical protocol.** The control:
- Verifies that the starting state is equilibrated (displacement should be negligible)
- Establishes baseline collective dynamics ($\beta$, $\chi_4$, $\tau_\alpha$)
- Enables computing disorder-*induced* changes, not disorder-*correlated* fluctuations

---

## Cluster Execution

Use the `resume_simulation` MCP tool with `parameter_sweep` for production runs. This iterates over parameter combinations, submitting replicates with chain dependencies. **Do NOT use `run_command` with `sbatch` or legacy scripts — the MCP tools handle GPU selection, SLURM accounts, and chaining automatically.**

Example: to run the full Griffiths sweep, call `resume_simulation` with `parameter_sweep` containing the v_A × sigma grid and replicate count. See `tools/compute_canada_mcp/DESIGN.md` for the `parameter_sweep` schema.

Cluster data lives under `/scratch/ssilber/griffiths_stiff/` on all Alliance clusters. Production directories follow the pattern `prod_phi89/vA_{X.XXX}_sigma_{X.XXX}/`. Equilibration checkpoints are in `equil_phi89/checkpoint.bin`. Use the **compute-canada MCP tool** (`list_jobs`, `check_progress`, `discover`) to check current job status, completion, and available data — do not rely on hardcoded status here.

> **Old data warning:** Directories under `/scratch/ssilber/eq_phi89/`, `/scratch/ssilber/griffiths_study/`, or `/scratch/ssilber/bresler_validation_phi89/` used Palmieri parameters (γ=1, μ=1) and should NOT be used for this study. Only data under `griffiths_stiff/` has the correct Bresler parameters.

### System Size Progression

| $N$ | $L$ | $\phi$ | Purpose | Status |
|-----|-----|--------|---------|--------|
| 288 | 1562 | 89% | **Primary data** (Bresler γ=3.75) | Running on narval/rorqual/fir |
| 1152 | 3124 | 89% | Finite-size check | Not started with Bresler params |
| 4608 | 6249 | 89% | Publication quality | Not started |

> **Note:** The old 85% ($L = 1600$) runs used Palmieri parameters (γ=1) and are superseded. All new runs use Bresler parameters at 89%.

**Why larger systems matter:** With 288 cells, the largest rare cluster is $\ell \approx 8$–10. The Griffiths argument requires that power-law tails emerge from clusters of increasing size. At $N = 4608$ ($\sim$68 cells per linear dimension), clusters of $\ell \approx 16$ become statistically accessible.

---

## Observables

### Primary Observables (All Must Be Computed)

| Observable | Definition | Griffiths prediction | Anti-Griffiths if... |
|------------|-----------|---------------------|---------------------|
| $Q(t)$ | Self-overlap function (Eq. below) | Power-law tail ($\beta \to 0$) | $\beta$ *increases* with $\sigma$ |
| $\chi_4(t)$ | $N[\langle Q^2\rangle - \langle Q\rangle^2]$ | Peak grows with $\sigma$ | Peak *drops* with $\sigma$ |
| $\tau_\alpha$ | $Q(\tau_\alpha) = e^{-1}$ | Increases with $\sigma$ | *Decreases* with $\sigma$ |
| $\alpha_2(t)$ | $\langle\Delta r^4\rangle/(2\langle\Delta r^2\rangle^2) - 1$ | Persistently elevated | Also persistent (ambiguous) |
| CV | std($D_i$)/mean($D_i$) | Broadens | Also broadens (ambiguous) |
| Persistence | Fraction retaining jammed/motile label | Increases | Also increases (ambiguous) |

**The key diagnostic distinction:** Single-cell observables ($\alpha_2$, CV, persistence) are ambiguous — they increase with any broadened motility distribution. Collective observables ($\beta$, $\chi_4$) are definitive — they distinguish correlated (Griffiths) from decorrelated (stirred glass) dynamics.

Current results are in `LOG_BOOK.md` and `manuscript.tex` Tables I–III.

### Self-Overlap Function

$$Q(\Delta t) = \frac{1}{N}\sum_{i=1}^N \Theta(a - |\mathbf{r}_i(t_0 + \Delta t) - \mathbf{r}_i(t_0)|)$$

where $a = 0.3\,d_\text{cell} \approx 28$ grid units. Fit to stretched exponential $Q(t) = \exp[-(t/\tau_\alpha)^\beta]$.

- Fitting: Levenberg-Marquardt nonlinear least squares, range $0.05 < Q < 0.99$, initial transients excluded
- Quality: $R^2 > 0.98$ for all reported fits
- Sensitivity: Results insensitive to $a$ over $0.2$–$0.4\,d_\text{cell}$

### Four-Point Susceptibility

$$\chi_4(\Delta t) = N[\langle Q(\Delta t)^2\rangle - \langle Q(\Delta t)\rangle^2]$$

Computed from 20 equally spaced time origins in the first half of each trajectory. Peak height measures the number of cooperatively rearranging cells.

**Statistical warning:** $\chi_4$ is a variance-of-a-variance quantity. With small replicate counts, statistical power is limited. Report trends and order-of-magnitude changes as robust; precise values should be treated as approximate. Larger $N$ simulations will improve statistics.

### Class-Resolved Analysis

Cells are classified as "jammed" or "motile" based on the $\sigma = 0$ mean mobility threshold. Compute $Q(t)$ separately for each class, fit stretched exponentials independently, and extract:

- $\tau_j/\tau_m$: relaxation time ratio (jammed to motile)
- $\beta_j$, $\beta_m$: class-resolved stretching exponents

---

## Analysis Pipeline

### Step 1: Data Transfer

Subsample cluster trajectories (every 500th line) to reduce transfer size:

```bash
# On cluster
cd /scratch/ssilber/griffiths_study
for dir in vA_*_sigma_*/run_*; do
    awk 'NR==1 || NR%500==0' "$dir/trajectory.txt" > \
        "${dir}_subsampled.txt"
done
```

Transfer subsampled files to local `subsampled_data/` directory.

### Step 2: Batch Analysis

```powershell
cd cpp/simulation/study/griffiths
python analyze_griffiths_batch.py subsampled_data/
```

Produces: MSD curves, displacement distributions, per-cell diffusivities, CV, $\alpha_2$.

### Step 3: Deep Analysis

```powershell
python analyze_griffiths_deep.py subsampled_data/
```

Produces: $Q(t)$ with stretched-exponential fits, $\chi_4(t)$, $v_A$–mobility correlation, class-resolved analysis.

### Step 4: Output

All figures go to `output/` with dated filenames:
- `griffiths_loglog_Qt_YYYYMMDD.png`
- `griffiths_chi4_YYYYMMDD.png`
- `griffiths_vA_correlation_YYYYMMDD.png`

---

## Discovering What Needs Doing

The study's progress is tracked in `LOG_BOOK.md` and the manuscript TODO markers. To determine what work remains:

1. **Read `LOG_BOOK.md`** — the latest entries describe completed runs and next steps
2. **Check `manuscript.tex`** — search for `\TODO` markers to find data gaps
3. **Use the compute-canada MCP tool** — `list_jobs` and `check_progress` show what's running/pending on cluster
4. **Read `REVIEW_PROMPT.md`** — lists what a referee would request before acceptance

### Analyses a Referee Would Likely Request

1. **Annealed disorder control:** Reassign $v_{A,i}$ periodically to show quench matters
2. **Cage length sensitivity:** Direct $l_c$ measurement from MSD plateau
3. **Spatial $\chi_4$ maps:** Show cooperative regions being destroyed
4. **$\phi = 0.89$ comparison:** Verify system is truly near the jammed regime
5. **Finite-size convergence:** $\beta(N)$ and $\chi_4(N)$ across system sizes

---

## Interpretation Framework

### What Each Observable Tests

| Observable | Tests for... | Griffiths if... | Anti-Griffiths if... |
|------------|-------------|-----------------|---------------------|
| $\beta$ (stretching) | Relaxation spectrum breadth | Decreases with $\sigma$ | Increases with $\sigma$ |
| $\chi_4$ peak | Cooperative region size | Grows with $\sigma$ | Drops with $\sigma$ |
| $\tau_\alpha$ | Bulk relaxation speed | Increases with $\sigma$ | Decreases with $\sigma$ |
| $\tau_j/\tau_m$ | Timescale separation | $\sim e^{b\ell}$ (exponential in cluster size) | Modest ratio |
| $\alpha_2$ | Displacement heterogeneity | Persistent (not definitive) | Also persistent |
| CV | Individual cell diversity | Broadens (not definitive) | Also broadens |

**The distinction:** Single-cell metrics ($\alpha_2$, CV) are ambiguous — they increase with any broadened distribution. Collective metrics ($\beta$, $\chi_4$) are definitive — they distinguish correlated (Griffiths) from decorrelated (stirred glass) dynamics.

### Effective Temperature vs. Stirred Glass

A naive "disorder = more thermal noise" model predicts monotonic fluidization at all $\bar{v}_A$. The stirred glass mechanism predicts **non-monotone** $\bar{v}_A$ dependence (fluidization peaks near the cage-scanning optimum $l_p \approx l_c$). Check Experiment B data to distinguish these: if fluidization is strongest at intermediate $\bar{v}_A$ and weaker at both lower and higher values, this rules out the effective temperature picture.

### Finite-Size Caveats

At $N = 288$ (~17 cells per linear dimension):
- Largest statistically expected rare cluster: $\ell \approx 8$–10
- $\beta > 1$ in $F_s(q^*,t)$ (compressed exponential) is a known finite-size artifact (Tjhung & Berthier 2020)
- The self-overlap $Q(t)$ and $F_s(q^*,t)$ have different finite-size sensitivities — always check which observable is being discussed
- Arguments for intrinsic behavior: (i) stirred glass mechanism operates locally (~3–5 cells), (ii) monotonic trends across multiple $\sigma$ values, (iii) larger-$N$ data should show convergence

---

## Key References

| Paper | Key finding | Relevance |
|-------|------------|-----------|
| Griffiths 1969, PRL 23, 17 | Essential singularities from rare regions | Original theory we test |
| Vojta 2006, J. Phys. A 39, R143 | Classification of rare-region effects | Theoretical framework |
| Bi et al. 2015, Nat. Phys. 11, 1074 | Shape index transition $p_0^* \approx 3.81$ | Tissue jamming foundation |
| Bi et al. 2016, PRX 6, 021011 | $(p_0, v_0)$ phase diagram | Motility-driven unjamming |
| Debets et al. 2021, PRL 127, 278002 | Cage-length framework $l_p/l_c$ | Interpretation of disorder effect |
| Debets et al. 2023, PRL 130, 058201 | Chirality disorder → reentrant glass | Precedent: active disorder ≠ noise |
| Keta et al. 2022, PRL 129, 048002 | Size polydispersity + active glass | Closest precedent; focuses on size, not motility |
| Mandal et al. 2020, Soft Matter 16, 3059 | Active fluidization in dense glasses | Supports stirred glass mechanism |
| Czajkowski et al. 2019, Soft Matter 15, 9133 | Cell turnover as biological stirring | Analogous mechanism via different route |
| Berthier & Biroli 2011, Rev. Mod. Phys. 83, 587 | Glass physics review | Defines Q(t), χ₄, α₂ |
| Tjhung & Berthier 2020, PhysRevRes 2, 043334 | Compressed exponentials from finite-size driving | Explains N=288 $F_s$ artifact |

---

## Common Mistakes to Avoid

1. **Confusing $Q(t)$ fits with $F_s(q^*, t)$ fits.** At $N = 288$, $F_s$ shows compressed exponentials ($\beta > 1$) due to finite-size effects. The self-overlap $Q(t)$ gives $\beta < 1$ (stretched). These are different observables with different finite-size sensitivities. The manuscript reports $Q(t)$ results.

2. **Over-interpreting $\sigma = 0.008$.** At $\sigma/\bar{v}_A = 1$, individual runs show 5× variability in $D$. Three replicates are insufficient for reliable stretched-exponential fits. Report the run-to-run variability as a finding, not the mean.

3. **Confusing single-cell and collective heterogeneity.** $\alpha_2$ increasing does NOT confirm Griffiths physics — it trivially follows from broadened $v_A$ distribution. $\chi_4$ decreasing definitively refutes Griffiths physics.

4. **Missing the non-monotone $\bar{v}_A$ dependence.** The stirred glass prediction is that fluidization is strongest near $l_p \approx l_c$. Check Experiment B data (Table III in manuscript): fluidization peaks at $\bar{v}_A = 0.008$.

5. **Forgetting the log-normal distribution.** The code uses log-normal, not truncated normal. The manuscript and literature review must reflect this. For $\sigma/\bar{v}_A = 1$, the distribution is right-skewed.

6. **Not running $\sigma = 0$ controls.** Every batch of simulations must include the clean control. Without it, you cannot compute disorder-*induced* changes.

7. **Insufficient trajectory sampling.** $Q(t)$ requires fine temporal resolution. Use trajectory interval = 100 steps (every 2 TU). The 85% study uses 125,000 snapshots per run.

---

## Logbook Protocol

The logbook (`LOG_BOOK.md`) is the primary record. Update it after every analysis session, batch of completed runs, or methodological decision.

### Entry Template

```markdown
### YYYY-MM-DD — [Brief Title]

**Runs:** [parameter combinations, cluster/local, completion status]
**Analysis:** [scripts run, figures generated]

**Key results:**
- [quantitative finding with numbers]
- [comparison to prediction or previous data]

**Interpretation:** [physical meaning]
**Next:** [what to do based on these results]
```

### Figure Conventions

All figures saved to `output/` with format:
```
griffiths_<analysis>_<params>_YYYYMMDD.png
```

---

*Last updated: February 19, 2026*
