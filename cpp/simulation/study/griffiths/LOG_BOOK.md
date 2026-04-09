# Griffiths Rare-Region Study — LOG BOOK

> **Study launched:** January 2026  
> **Cluster:** Nibi (Alliance Canada), H100 GPUs (MIG 1g.10gb for 288c, full H100 for >500c)  
> **Status:** 85% study: 17/18 DONE. 89% 288c: 180 jobs in queue. 4608c griffiths: submitted (90 jobs). Equilibrations: N=1152 & N=4608 DONE.

---

## Table of Contents

1. [Motivation: Why Griffiths Effects?](#1-motivation-why-griffiths-effects)
2. [Physics Background](#2-physics-background)
3. [Experimental Design](#3-experimental-design)
4. [Simulation Parameters](#4-simulation-parameters)
5. [Cluster Execution](#5-cluster-execution)
6. [Live Progress](#6-live-progress)
7. [Data Quality Validation](#7-data-quality-validation)
8. [Analysis Plan](#8-analysis-plan)
9. [Expected Signatures](#9-expected-signatures)
10. [Open Questions & Notes](#10-open-questions--notes)
11. [References](#11-references)
12. [Preliminary Results (Session 3)](#12-preliminary-results-session-3--feb-11-2026)
13. [Session 4 — Queue Cleanup & 89% Launch](#13-session-4--queue-cleanup--89-launch-feb-11-2026)
14. [Session 5 — 4608c Griffiths Launch & Infrastructure](#15-session-5--4608c-griffiths-launch--infrastructure-overhaul-feb-15-2026)

---

## 1. Motivation: Why Griffiths Effects?

### The clean jamming transition

In a uniform tissue with identical cells (all sharing the same active velocity $v_A$),
there is a sharp motility-driven jamming transition. Below a critical $v_A^*$, cells
are caged by their neighbors and the tissue is solid-like; above $v_A^*$, cells
rearrange freely and the tissue flows. This has been established in vertex models
(Bi et al. 2016) and confirmed in our phase field model (PFM) production runs.

The order parameter for this transition is the long-time diffusion coefficient:

$$D = \lim_{t \to \infty} \frac{\langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle}{4t}$$

In a clean system, $D$ jumps from $\approx 0$ (jammed) to a finite value (fluid) at
$v_A = v_A^*$.

### The problem with clean transitions

Real tissues are **not** uniform. Cells differ in motility, adhesion, stiffness, and
contractility. Even clonal populations exhibit cell-to-cell variability in gene
expression and mechanical properties. A sharp transition at a single critical $v_A$
is therefore a theoretical idealization.

### Griffiths physics: what disorder does

**Griffiths effects** arise when quenched (frozen) disorder is introduced into a
system near a phase transition. The key insight, first described by Griffiths (1969)
for the dilute Ising ferromagnet, is:

> In a disordered system, **rare spatial regions** that are locally on the other
> side of the transition dominate the long-time dynamics.

For our tissue model, this means:

- At a *mean* $v_A$ just below the clean transition, rare clusters of cells with
  locally high $v_A$ form **fluid pockets** embedded in the jammed bulk.
- At a *mean* $v_A$ just above the transition, rare clusters with locally low $v_A$
  form **jammed islands** that slow down relaxation.

These rare regions produce qualitatively different physics from the clean case.

### Why this matters for biology

Cell-to-cell motility variation is ubiquitous in real tissues:
- Tumor heterogeneity produces cells with widely varying migratory capacity
- Wound healing fronts show leader/follower cell distinctions
- Developmental tissues have morphogen-driven motility gradients

If Griffiths effects are present, they would explain:
- Why tissue unjamming appears gradual rather than sharp
- Why spatial heterogeneity in cell mobility persists over long times
- Why isolated "fluid" regions can exist within otherwise solid tissue
- How rare highly-motile cells can nucleate collective migration

---

## 2. Physics Background

### 2.1 The jamming transition in tissues

The jamming/unjamming transition (JUT) in tissues is controlled by a competition
between cell-cell adhesion (which promotes caging) and active self-propulsion
(which drives rearrangements).

In our PFM, each cell $i$ has a phase field $\phi_i(\mathbf{r}, t)$ evolving under:

$$\frac{\partial \phi_i}{\partial t} = -M \frac{\delta F}{\delta \phi_i} + \mathbf{v}_i \cdot \nabla \phi_i$$

where $\mathbf{v}_i = v_{A,i} \hat{p}_i$ is the self-propulsion velocity and
$\hat{p}_i$ is a unit polarity vector that reorients with persistence time $\tau$.

**Key control parameters:**
| Parameter | Symbol | Role |
|-----------|--------|------|
| Active velocity | $v_A$ | Drives unjamming |
| Persistence time | $\tau$ | Sets run length before reorientation |
| Packing fraction | $\phi$ | Geometric confinement |
| Cell radius | $R$ | Sets cell size scale |

### 2.2 Quenched vs. annealed disorder

**Quenched disorder** means the randomness is frozen in time: each cell $i$ is
assigned a motility $v_{A,i}$ drawn from a distribution at $t = 0$, and this value
never changes. This is the relevant case for Griffiths physics.

Our implementation draws:

$$v_{A,i} \sim \mathcal{N}(\bar{v}_A, \sigma^2) \quad \text{truncated at } v_{A,i} \geq 0$$

where $\bar{v}_A$ is the mean motility (the `--v-A` parameter) and $\sigma$ is the
disorder strength (the `--v-A-sigma` parameter).

**Annealed disorder** (where motilities fluctuate in time) would give different
physics — the system self-averages and the transition remains sharp. We use quenched
disorder specifically to access the Griffiths regime.

### 2.3 The Griffiths phase

In a clean system, define the critical motility $v_A^*$ where the JUT occurs.
With quenched disorder of strength $\sigma$, the physics changes qualitatively in
a **Griffiths region** around $v_A^*$:

```
       Jammed          Griffiths Region           Fluid
    ◄──────────┼─────────────────────────┼──────────────►  v_A
               v_A* − σ                 v_A* + σ
```

Within this region:

1. **The transition is smeared.** There is no single critical $v_A$ — instead, $D(\bar{v}_A)$ increases continuously from 0.

2. **Rare regions dominate.** The overlap function $Q(t)$ (fraction of cells that haven't moved beyond a threshold distance $a$) decays as:

$$Q(t) \sim t^{-\alpha(\bar{v}_A)} \quad \text{(power law, not exponential)}$$

The exponent $\alpha$ varies continuously with $\bar{v}_A$ within the Griffiths phase.
This is in contrast to the clean system where $Q(t)$ decays exponentially in the
fluid phase and plateaus in the jammed phase.

3. **Spatial heterogeneity is persistent.** Cells with high $v_{A,i}$ remain mobile
   and cells with low $v_{A,i}$ remain caged, even at long times.

### 2.4 Rare-region mechanism

Consider a mean motility $\bar{v}_A$ slightly below $v_A^*$ (the jammed side).
In a clean system, every cell is jammed. But with disorder:

- Most cells have $v_{A,i} \approx \bar{v}_A < v_A^*$ → jammed
- Rare clusters of cells where all members have $v_{A,i} > v_A^*$ → locally fluid

The probability of finding a fluid cluster of size $\ell$ scales as:

$$P(\ell) \sim e^{-c \ell^d}$$

where $c$ depends on $|\bar{v}_A - v_A^*|/\sigma$ and $d$ is the spatial dimension.
These clusters are exponentially rare but each contributes a relaxation that is
exponentially slow (the cluster must wait for its rare configuration to thermally
activate a rearrangement). The product of exponentially rare × exponentially slow
produces **power-law** tails in the relaxation — the hallmark of Griffiths physics.

---

## 3. Experimental Design

### 3.1 Two-axis exploration

We probe the Griffiths phase along two complementary axes:

**Experiment A — Disorder scan** (fixed mean motility, vary disorder strength):

| Config | $\bar{v}_A$ | $\sigma$ | Purpose |
|--------|-------------|----------|---------|
| A1 | 0.008 | 0.000 | Clean baseline (no disorder) |
| A2 | 0.008 | 0.003 | Weak disorder |
| A3 | 0.008 | 0.006 | Moderate disorder ($\sigma/\bar{v}_A = 0.75$) |
| A4 | 0.008 | 0.008 | Strong disorder ($\sigma/\bar{v}_A = 1.0$) |

This scan reveals how increasing disorder progressively smears the transition
and introduces power-law dynamics. $\bar{v}_A = 0.008$ is chosen because it sits
near the clean transition: our production data shows the jamming transition
occurs around $v_A^* \approx 0.007$–$0.009$.

**Experiment B — Motility scan** (fixed disorder, vary mean motility):

| Config | $\bar{v}_A$ | $\sigma$ | Purpose |
|--------|-------------|----------|---------|
| B1 | 0.006 | 0.006 | Below clean $v_A^*$ with moderate disorder |
| B2 | 0.010 | 0.006 | Above clean $v_A^*$ with moderate disorder |

Combined with A3 ($\bar{v}_A = 0.008$, $\sigma = 0.006$), this gives a three-point
traverse across the transition at fixed disorder, revealing how the Griffiths
region broadens around $v_A^*$.

### 3.2 Statistical replication

- **3 independent runs** per parameter combination (different random seeds)
- Each run uses a different equilibrated starting configuration (run_1, run_2, run_3
  from the ensemble equilibration set)
- The quenched disorder realization ($\{v_{A,i}\}$) is seeded independently per run

Total: 6 configs × 3 runs = **18 independent simulations**.

### 3.3 Why 3 runs (not 100)?

Our production jamming study uses 100 replicas per velocity, but those runs are
clean (no disorder). For the Griffiths study:

- Each run already has internal disorder (288 cells with different $v_{A,i}$),
  providing within-run statistics
- The primary observables (MSD, $Q(t)$, per-cell diffusivity) are self-averaging
  over the 288 cells within a single run
- 3 runs gives run-to-run variability estimates without excessive compute cost
- We can always add more runs later if statistical precision is insufficient

---

## 4. Simulation Parameters

### 4.1 Full parameter table

| Parameter | Value | Notes |
|-----------|-------|-------|
| N (cells) | 288 | Standard production count |
| L (domain) | 1600 × 1600 | Gives $\phi = N \pi R^2 / L^2 = 0.85$ |
| R (radius) | 49 | Cell radius |
| $\phi$ (confluence) | 85% | ⚠️ See note below |
| dt | 0.02 | Time step (2D default) |
| $\tau$ (persistence) | 10,000 | Polarity reorientation time ($= 1/D_r$) |
| T_start | 81,000 | From equilibration checkpoint |
| T_end | 331,000 | 250,000 time units of production |
| TRAJECTORY_INTERVAL | 100 | Every 2 time units (⚠️ very frequent) |
| CHECKPOINT_INTERVAL | 75,000 | Safety checkpoints |
| SAVE_INTERVAL | 0 | VTK output disabled |
| PRINT_INTERVAL | 10,000 | Console progress |

### 4.2 Time scales

| Time scale | Value | Meaning |
|------------|-------|---------|
| dt | 0.02 | Integration step |
| $\tau$ | 10,000 | Persistence time ($= 500,000$ steps) |
| T_production | 250,000 | Run duration ($= 25 \tau$) |
| Trajectory sampling | every 100 steps = 2.0 time units | Data resolution |

The production window of $25\tau$ is sufficient for cells to undergo many
persistence-time reorientations and reach diffusive behavior.

### 4.3 Notes on packing fraction

The current runs use $L = 1600$, giving $\phi = 288 \times \pi \times 49^2 / 1600^2 \approx 0.849$ (85%).
Previous discussions considered using $L = 1562$ for $\phi \approx 0.89$ (89% confluence).
The equilibration checkpoints were prepared at $L = 1600$.

### 4.4 Note on trajectory output frequency

TRAJECTORY_INTERVAL = 100 steps corresponds to recording every 2 time units.
Over 250,000 time units of production, this yields ~125,000 snapshots per run,
each with 288 cells = ~36 million trajectory lines per run.

Current file sizes (mid-run) range from 161 MB to 1.3 GB. Projected final sizes
are ~2–3 GB per run, or ~36–54 GB total for 18 runs. This is much more data than
needed for MSD analysis (where ~2000 time points suffice) but provides excellent
temporal resolution for computing $Q(t)$ with fine lag-time binning, which is
critical for resolving power-law tails.

---

## 5. Cluster Execution

### 5.1 Job structure

Each of the 18 simulations is submitted as a chain of 15 SLURM jobs:

```
Job_chain_1 → Job_chain_2 → ... → Job_chain_15
   (3 hr)       (3 hr)              (3 hr)
```

Each chain job:
- Loads the checkpoint from the previous chain (or the equilibration checkpoint for chain 1)
- Runs for up to 3 hours walltime on an H100 MIG 3g.40gb GPU
- Saves a checkpoint at exit
- The next chain job has a SLURM dependency and starts automatically

Total: 18 runs × 15 chains = **270 SLURM jobs** (18 running, 252 pending at launch).

### 5.2 Performance

Measured throughput on H100 MIG 3g.40gb for 288 cells (2D):
- ~24,900 to ~48,000 time units per hour (varies with MIG slice load)
- Each 3-hour chain completes ~75,000–144,000 time units
- Expected chains needed: ~3–4 per run (of 15 provisioned)

On RTX 4090 Laptop (for reference):
- ~17,484 time units per hour for 288 cells

### 5.3 Submission command

```bash
cd ~/cell_simulation/cluster
./submit_job.sh --griffiths --runs 3 -t 331000
```

This invokes the `griffiths` mode in `submit_job.sh` (lines 639–718), which:
1. Iterates over Experiment A: sigma ∈ {0.000, 0.003, 0.006, 0.008} at $v_A = 0.008$
2. Iterates over Experiment B: $v_A$ ∈ {0.006, 0.010} at sigma = 0.006
3. For each combo: 3 runs × 15 chains = 45 SLURM jobs
4. Each chain picks up the previous chain's checkpoint (or the equilibration checkpoint)

### 5.4 Directory layout

```
/scratch/ssilber/griffiths_study/
├── vA_0.006_sigma_0.006/
│   ├── run_1/
│   │   ├── trajectory.txt     ← main data output
│   │   └── checkpoint.bin     ← latest checkpoint
│   ├── run_2/
│   └── run_3/
├── vA_0.008_sigma_0.000/      ← clean baseline (Experiment A1)
│   ├── run_1/
│   ├── run_2/
│   └── run_3/
├── vA_0.008_sigma_0.003/
├── vA_0.008_sigma_0.006/
├── vA_0.008_sigma_0.008/
└── vA_0.010_sigma_0.006/
```

Logs: `~/cell_sim_logs/griffiths/`

---

## 6. Live Progress

### Session 2 status (Jan 2026)

| Configuration | run_1 | run_2 | run_3 |
|:---|:---:|:---:|:---:|
| **vA=0.006, σ=0.006** | t=113k (329M) | t=108k (288M) | t=101k (218M) |
| **vA=0.008, σ=0.000** | t=134k (550M) | t=152k (838M) | t=153k (863M) |
| **vA=0.008, σ=0.003** | t=181k (1.3G) | t=152k (846M) | t=151k (826M) |
| **vA=0.008, σ=0.006** | t=147k (752M) | t=128k (488M) | t=136k (592M) |
| **vA=0.008, σ=0.008** | t=108k (279M) | t=141k (676M) | t=147k (770M) |
| **vA=0.010, σ=0.006** | t=102k (215M) | t=100k (204M) | t=96k (161M) |

### Session 3 status (Feb 11, 2026, early)

See **Section 12** for preliminary displacement analysis.

### Session 4 status (Feb 11, 2026, late)

**85% Griffiths — 17/18 complete:**

| Configuration | run_1 | run_2 | run_3 |
|:---|:---:|:---:|:---:|
| **vA=0.006, σ=0.006** | ✅ 331k | ✅ 331k | ✅ 331k |
| **vA=0.008, σ=0.000** | t=256k (3.0G) | ✅ 331k | ✅ 331k |
| **vA=0.008, σ=0.003** | ✅ 331k | ✅ 331k | ✅ 331k |
| **vA=0.008, σ=0.006** | ✅ 331k | ✅ 331k | ✅ 331k |
| **vA=0.008, σ=0.008** | ✅ 331k | ✅ 331k | ✅ 331k |
| **vA=0.010, σ=0.006** | ✅ 331k | ✅ 331k | ✅ 331k |

Disk: **54 GB** for 85% study.

**89% Griffiths — submitted, waiting in queue:**
- 18 runs × 15 chains = 271 pending jobs
- Same 6 parameter configurations as 85% study
- Using equilibration checkpoints from `/scratch/ssilber/eq_phi89/`
- Output: `/scratch/ssilber/griffiths_study_phi89/`

**Equilibrations for larger systems — submitted:**
- N=1152 cells (L=3125, φ=89%): 3 runs × 6 chains = 18 jobs pending
- N=4608 cells (L=6249, φ=89%): 3 runs × 20 chains = 58 jobs pending

**Queue:** 1 running, 347 pending. Excess 85% chains cancelled.

### Completion estimate

At ~50,000 time units/hour (average cluster rate), the 250,000 time-unit production
window takes ~5 hours. With 3-hour walltime jobs, this is ~2 chain jobs per run.
Most runs should complete within chains 1–2 (of 15 provisioned).

---

## 7. Data Quality Validation

### 7.1 Trajectory format check

Header format confirmed:
```
# Trajectory data for MSD computation
# Format: time cell_id x y vx vy px py theta
# v_A=0.008 N=288 Lx=1600 Ly=1600
```

Data columns verified: `time cell_id x y vx vy px py theta` (9 columns per line).

### 7.2 Physics sanity checks

| Check | σ=0.000 | σ=0.008 | Expected |
|-------|---------|---------|----------|
| Mean \|v\| | 0.00779 | 0.00804 | ≈ $v_A$ |
| \|p\| (polarity) | 1.000 | 1.000 | Exactly 1 |
| Cells per timestep | 288 | 288 | 288 |
| Positions | within [0, 1600] | within [0, 1600] | Within domain |
| θ range | [0.006, 6.24] | [0.006, 6.24] | [0, 2π] |

**Velocity sanity:** For $\sigma = 0$, all cells have the same $v_A = 0.008$, so
mean $|v|$ should be close to 0.008. The measured 0.00779 reflects that instantaneous
velocity includes relaxation dynamics, not just self-propulsion. For $\sigma = 0.008$,
the slightly higher mean (0.00804) is consistent with the truncated normal
distribution (can't go below 0, so the mean shifts up slightly).

---

## 8. Analysis Plan

### 8.1 Primary observables

#### A. Mean squared displacement (MSD)

$$\text{MSD}(\Delta t) = \left\langle \frac{1}{N} \sum_{i=1}^{N} |\mathbf{r}_i(t + \Delta t) - \mathbf{r}_i(t)|^2 \right\rangle_t$$

Compute for each ($\bar{v}_A$, $\sigma$) combination. Use periodic boundary unwrapping.

**What to look for:**
- $\sigma = 0$: MSD should show clear caging plateau (if jammed) or linear growth (if fluid)
- $\sigma > 0$: MSD may show an intermediate regime — subdiffusive at intermediate times as jammed cells contribute zero displacement while fluid cells contribute linear growth

#### B. Self-overlap function $Q(t)$

$$Q(t) = \frac{1}{N} \sum_{i=1}^{N} \Theta\!\left(a - |\mathbf{r}_i(t) - \mathbf{r}_i(0)|\right)$$

where $\Theta$ is the Heaviside function and $a$ is a threshold distance (typically $a \sim 0.3 R$).

**This is the smoking-gun observable for Griffiths effects:**
- Clean system ($\sigma = 0$): $Q(t) \to$ const (jammed) or $Q(t) \sim e^{-t/\tau_\alpha}$ (fluid)
- Griffiths phase ($\sigma > 0$): $Q(t) \sim t^{-\alpha}$ **power-law decay**
  - The exponent $\alpha$ varies continuously with $\bar{v}_A$ and $\sigma$
  - This power law arises from the superposition of exponential relaxations from rare regions of different sizes

#### C. Per-cell diffusivity distribution $P(D_i)$

For each cell $i$, compute its individual long-time diffusion coefficient:

$$D_i = \lim_{\Delta t \to \infty} \frac{|\mathbf{r}_i(t_0 + \Delta t) - \mathbf{r}_i(t_0)|^2}{4 \Delta t}$$

**What to look for:**
- $\sigma = 0$: $P(D_i)$ should be narrow (all cells have same $D$)
- $\sigma > 0$: $P(D_i)$ should be **broad**, with a peak near $D = 0$ (jammed cells) and a tail extending to high $D$ (fluid cells)
- The shape of $P(D_i)$ reveals the disorder landscape

#### D. Spatial mobility map

Plot the per-cell $D_i$ on the spatial positions of the cells to create a "mobility map."

**Key test:** Compare the mobility map to the quenched $v_{A,i}$ map. If Griffiths
physics is operating, there should be a strong spatial correlation — patches of
high $v_{A,i}$ cells should coincide with patches of high $D_i$.

This is the most direct test: it shows that **quenched disorder in motility
produces persistent spatial heterogeneity in dynamics**.

### 8.2 Secondary observables

#### E. Transition broadening: $D(\bar{v}_A)$ curve

Compare the disorder-averaged diffusion coefficient $D$ vs $\bar{v}_A$ for:
- $\sigma = 0$ (clean): should show a sharp jump at $v_A^*$
- $\sigma = 0.006$: should show a gradual, smeared transition
- The width of the crossover region should scale with $\sigma$

Using Experiment B data ($\bar{v}_A \in \{0.006, 0.008, 0.010\}$ at $\sigma = 0.006$),
we get three points on this smeared curve.

#### F. Non-Gaussian parameter

$$\alpha_2(\Delta t) = \frac{\langle |\Delta r|^4 \rangle}{(1 + 2/d) \langle |\Delta r|^2 \rangle^2} - 1$$

Measures deviation from Gaussian displacement statistics. In the Griffiths phase:
- $\alpha_2$ should be large and persistent (unlike clean systems where it peaks then decays)
- The persistent non-Gaussianity reflects the coexistence of mobile and immobile populations

#### G. Contact area observable $C_{ij}(t)$

(See `docs/rearrangement_mechanics.md` for details.)

The overlap integral between neighboring cells:

$$C_{ij}(t) = \int \phi_i(\mathbf{r}, t) \, \phi_j(\mathbf{r}, t) \, d\mathbf{r}$$

This is unique to the PFM (vertex models can't compute this). Changes in $C_{ij}$
indicate rearrangements without requiring a discrete "T1" definition.

**In the Griffiths context:** Do rearrangement rates correlate spatially with the
quenched $v_{A,i}$ map? This would give a mechanistic picture of how disorder
controls local tissue fluidity.

### 8.3 Analysis script

A local analysis script (`analyze_griffiths.py`, if available) computes MSD and
diffusion coefficients from trajectory data. It uses:

- `load_trajectory()` to parse the trajectory format
- `compute_msd()` with periodic boundary unwrapping ($L = 1600$)
- `compute_diffusion()` from long-time MSD slope
- `compute_per_cell_displacement()` for $P(D_i)$

For cluster-scale analysis of the full dataset, C implementations (`msd_calculator.c`)
provide the necessary performance. See `docs/cluster-postprocessing.instructions.md`
for the development workflow.

---

## 9. Expected Signatures

### 9.1 Predictions by observable

| Observable | Clean ($\sigma=0$) | Griffiths ($\sigma>0$) |
|:---|:---|:---|
| MSD | Plateau (jammed) or linear (fluid) | Subdiffusive intermediate regime |
| $Q(t)$ | Plateau or exponential decay | **Power-law decay** $t^{-\alpha}$ |
| $P(D_i)$ | Narrow, single peak | **Broad, bimodal** (jammed + fluid) |
| Mobility map | Spatially uniform | **Correlated with quenched** $v_{A,i}$ |
| $D(\bar{v}_A)$ | Sharp jump at $v_A^*$ | **Smeared**, gradual crossover |
| $\alpha_2$ | Peaks then decays | **Persistently large** |
| Rearrangement rate | Uniform | Correlated with $v_{A,i}$ map |

### 9.2 What would be novel?

Griffiths effects have been studied in spin systems and some active matter models,
but have **never been demonstrated in a tissue-level phase field model**. The key
novelties of this study:

1. **Continuous interface model:** Unlike vertex models or lattice models, the PFM
   allows cells to deform and squeeze past each other continuously. This could
   reveal whether rare-region dynamics are enhanced or suppressed by interface
   fluctuations.

2. **Contact area observable:** The $C_{ij}(t)$ "rearrangement spectroscopy"
   (see `rearrangement_mechanics.md`) is unique to PFMs. If we can show that
   rearrangement timescales correlate with the local disorder realization, this
   provides a mechanistic link between quenched disorder and dynamics that is
   inaccessible to vertex models.

3. **Non-confluent regime:** Unlike vertex models (which assume 100% confluence),
   our PFM at $\phi = 0.85$ has gaps between cells. The interaction between
   gaps and rare fluid regions could produce qualitatively different Griffiths
   physics.

### 9.3 Null result interpretation

If we observe:
- $Q(t)$ decays exponentially even with $\sigma > 0$ → system self-averages; Griffiths effects may be absent at this system size (288 cells may be too small for rare regions to form)
- $P(D_i)$ is narrow for all $\sigma$ → motility disorder doesn't translate to dynamical heterogeneity (would need to investigate coupling mechanism)
- No spatial correlation between $v_{A,i}$ and $D_i$ → collective effects wash out the disorder (interesting in its own right — would imply tissue-level cooperativity)

---

## 10. Open Questions & Notes

### ⚠️ Issue: Trajectory output frequency

TRAJECTORY_INTERVAL = 100 (every 2 time units) is very aggressive. For
comparison, the production jamming study uses TRAJECTORY_INTERVAL = 18000
(every 360 time units). The current setting produces ~180× more data.

**Pros:**
- Fine temporal resolution for $Q(t)$ power-law fitting (need many decades in time)
- Can always downsample in post-processing

**Cons:**
- ~2–3 GB per run, ~36–54 GB total
- I/O overhead may slow simulations slightly

**Decision:** Accept for this study. The fine resolution is valuable for
distinguishing power-law from stretched-exponential decay in $Q(t)$.

### ✅ Resolved: Packing fraction (85% vs 89%)

The initial 85% runs ($L = 1600$) are nearly complete. A matching 89% study
($L = 1562$) has been submitted using the `eq_phi89` equilibration checkpoints.
Comparing both packing fractions will reveal how confluence level affects:
- The location of $v_A^*$ (tighter packing → higher $v_A^*$)
- The width of the Griffiths region
- Whether 89% is deep enough into jamming to see stronger rare-region effects

### ✅ New: Larger system equilibrations (N=1152, N=4608)

Equilibration runs have been submitted for two larger system sizes at 89% confluence:

| System | N cells | Domain L | φ | Chains | Purpose |
|:---|---:|---:|---:|---:|:---|
| 4× | 1152 | 3125 | 89% | 6 per run | Finite-size check, rare-region size scaling |
| 16× | 4608 | 6249 | 89% | 20 per run | Publication-quality, large rare regions |

Once equilibrated (t=80,000 at v_A=0), these can be used for Griffiths production
runs. The larger systems are critical because:
- Griffiths effects depend on rare clusters — need enough cells for statistics
- 288 cells may be too small for spatially extended rare regions to form
- N=4608 provides 16× more cells, enabling rare-region clusters of ~50-100 cells

### Note: Chain provisioning

15 chains were provisioned per run, but only ~3–4 are expected to be needed.
Excess chains will start, find that T_end has already been reached, and exit
immediately. This wastes some scheduler overhead but is harmless.

### Note: Random seed for disorder

The quenched $v_{A,i}$ distribution is seeded per run by the simulation code.
The exact seed is determined by the run number and the `--v-A-sigma` flag.
To verify that different runs have different disorder realizations, we can
inspect the per-cell velocity distributions in the trajectory data.

---

## 11. References

### Review Papers

| Reference | DOI | Relevance |
|:---|:---|:---|
| Vojta (2006), "Rare region effects at classical, quantum, and nonequilibrium phase transitions," *J. Phys. A* 39, R143–R205 | [10.1088/0305-4470/39/22/R01](https://doi.org/10.1088/0305-4470/39/22/R01) | **The** comprehensive Griffiths / rare-region review. Classifies singularities by symmetry, dimensionality, dynamics. Section 4 on "smeared transitions" directly relevant to our anti-Griffiths observation. |
| Vojta (2019), "Disorder in Quantum Many-Body Systems," *Annu. Rev. Condens. Matter Phys.* 10, 233–252 | [10.1146/annurev-conmatphys-031218-013433](https://doi.org/10.1146/annurev-conmatphys-031218-013433) | Updated review covering infinite-randomness fixed points and activated scaling. Good for the theoretical machinery behind Griffiths singularities. |
| Marchetti et al. (2013), "Hydrodynamics of soft active matter," *Rev. Mod. Phys.* 85, 1143 | [10.1103/RevModPhys.85.1143](https://doi.org/10.1103/RevModPhys.85.1143) | Foundational active matter review: self-propelled particles, flocking, active gels. Essential for situating our motile tissue in the broader field. |
| Berthier & Biroli (2011), "Theoretical perspective on the glass transition and amorphous materials," *Rev. Mod. Phys.* 83, 587 | [10.1103/RevModPhys.83.587](https://doi.org/10.1103/RevModPhys.83.587) | Encyclopedic glass physics review. Defines all our observables ($Q(t)$, $\chi_4$, $\alpha_2$, stretched exponentials). Sections on dynamic heterogeneity and four-point correlations essential for interpreting our $\chi_4$ measurements. |
| Berthier (2019), "Glassy dynamics in dense systems of self-propelled particles," *J. Chem. Phys.* 150, 200901 | [10.1063/1.5093240](https://doi.org/10.1063/1.5093240) | Perspective on how activity modifies the glass transition. Discusses re-entrant glass at high activity — matches our $v_A$ sweep non-monotonicity. Directly relevant to our finding that disorder fluidizes rather than freezing. |

### Key Research Papers

| Reference | DOI | Relevance |
|:---|:---|:---|
| Griffiths (1969), "Nonanalytic Behavior Above the Critical Point in a Random Ising Ferromagnet," *Phys. Rev. Lett.* 23, 17 | [10.1103/PhysRevLett.23.17](https://doi.org/10.1103/PhysRevLett.23.17) | Original 2-page proof. Elegant and short — worth reading in full. |
| Fisher (1992), "Random transverse field Ising spin chains," *Phys. Rev. Lett.* 69, 534 | [10.1103/PhysRevLett.69.534](https://doi.org/10.1103/PhysRevLett.69.534) | Shows 1D Griffiths effects are maximally strong: infinite-randomness fixed point. The 1D case gives power-law $Q(t) \sim t^{-\alpha}$ exactly. |
| Bi et al. (2015), "A density-independent glass transition in biological tissues," *Nat. Phys.* 11, 1074 | [10.1038/nphys3471](https://doi.org/10.1038/nphys3471) | Shape index transition at $p_0^* = 3.81$. Confluent tissues jam without density change. |
| Bi et al. (2016), "Motility-driven glass and jamming transitions in biological tissues," *Phys. Rev. X* 6, 021011 | [10.1103/PhysRevX.6.021011](https://doi.org/10.1103/PhysRevX.6.021011) | **The** paper mapping the $(v_0, p_0)$ phase diagram for tissue jamming. Our entire parameter space is built around this. |
| Barton et al. (2017), "Active Vertex Model for cell-resolution description of epithelial tissue mechanics," *PLOS Comp. Biol.* 13(6), e1005569 | [10.1371/journal.pcbi.1005569](https://doi.org/10.1371/journal.pcbi.1005569) | Comprehensive active vertex model framework with growth, division, death. |
| Czajkowski et al. (2019), "Glassy dynamics in models of confluent tissue with mitosis and apoptosis," *Soft Matter* 15, 9133 | [10.1039/c9sm00916g](https://doi.org/10.1039/c9sm00916g) | Cell cycling as fluidization mechanism; additive relaxation ansatz. |
| Mandal et al. (2020), "Active fluidization in dense glassy systems," *Soft Matter* 16, 3059 | [10.1039/C9SM01873E](https://doi.org/10.1039/C9SM01873E) | Activity fluidizes dense glasses; re-entrant glass at high activity. Supports our observation that activity disorder decorrelates collective caging. |
| Reichhardt & Olson Reichhardt (2017), "Depinning and nonequilibrium dynamic phases of particle assemblies driven over random and ordered substrates," *Rep. Prog. Phys.* 80, 026501 | [10.1088/1361-6633/80/2/026501](https://doi.org/10.1088/1361-6633/80/2/026501) | Driven particles + quenched substrate disorder. Closest precedent to our "active cells with quenched motility disorder." |
| Nonomura (2012), "Study on Multicellular Systems Using a Phase Field Model," *PLoS ONE* 7, e33501 | [10.1371/journal.pone.0033501](https://doi.org/10.1371/journal.pone.0033501) | Early multi-cell phase field model. Establishes the model class we use. |
| Palmieri, Bresler et al. (2015), "Multiple scale model for cell migration in monolayers," *Sci. Rep.* 5, 11745 | [10.1038/srep11745](https://doi.org/10.1038/srep11745) | Phase field tissue model with motility coupling. Closest methodological ancestor to our simulation. |

### Experimental Papers

| Reference | DOI | Relevance |
|:---|:---|:---|
| Angelini et al. (2011), "Glass-like dynamics of collective cell migration," *PNAS* 108, 4714 | [10.1073/pnas.1010059108](https://doi.org/10.1073/pnas.1010059108) | First paper to identify glass physics in cell monolayers: caging, $\alpha_2$, dynamic heterogeneity. Our simulation reproduces the same phenomenology. |
| Park et al. (2015), "Unjamming and cell shape in the asthmatic airway epithelium," *Nat. Mater.* 14, 1040 | [10.1038/nmat4357](https://doi.org/10.1038/nmat4357) | Landmark experiment: asthmatic tissue unjammed ($p > 3.81$), healthy tissue jammed ($p < 3.81$). Motivates why disorder in cell properties matters biologically. |
| Atia et al. (2018), "Geometric constraints during epithelial jamming," *Nat. Phys.* 14, 613 | [10.1038/s41567-018-0089-9](https://doi.org/10.1038/s41567-018-0089-9) | Experimental validation of vertex model jamming predictions. Cell shape predicts jamming in MDCK monolayers. Grounds our simulation work in experiment. |

### Project documentation

| Document | Path |
|:---|:---|
| Glass physics tutorial | `cluster/docs/glass_physics_tutorial.md` |
| Literature review | `cluster/docs/LITERATURE_REVIEW.md` |
| Cell jamming review | `cluster/docs/review_cell_jamming.md` |
| Rearrangement mechanics | `cluster/docs/rearrangement_mechanics.md` |
| Paper summaries | `cluster/docs/references.md` |
| Submission script | `cluster/submit_job.sh` (griffiths mode, lines 639–718) |
| Cluster operations | `.github/instructions/cluster-operations.instructions.md` |
| Simulation reference | `.github/instructions/cell-simulation.instructions.md` |

---

## 12. Preliminary Results (Session 3 — Feb 11, 2026)

### 12.1 Completion status

**8 of 18 runs have reached T_end = 331,000:**

| Configuration | run_1 | run_2 | run_3 |
|:---|:---:|:---:|:---:|
| **vA=0.006, σ=0.006** | t=178k | t=172k | t=167k |
| **vA=0.008, σ=0.000** | t=194k | ✅ 331k | ✅ 331k |
| **vA=0.008, σ=0.003** | ✅ 331k | ✅ 331k | ✅ 331k |
| **vA=0.008, σ=0.006** | t=327k | t=223k | t=291k |
| **vA=0.008, σ=0.008** | t=170k | ✅ 331k | ✅ 331k |
| **vA=0.010, σ=0.006** | t=165k | t=166k | t=162k |

Queue: 11 running, 227 pending. Total disk: **38 GB**.

### 12.2 Per-cell displacement analysis

Computed each cell's total displacement over the full trajectory
(t = 80,000 → 331,000, dt ≈ 251,000 time units) for all 7 completed runs:

| Run | Mean disp | Std disp | CV | Min | Max | D |
|:---|---:|---:|---:|---:|---:|---:|
| σ=0.000 / r2 | 102.6 | 52.3 | 0.51 | 9.7 | 301.9 | 0.0132 |
| σ=0.000 / r3 | 105.9 | 56.3 | 0.53 | 2.6 | 298.3 | 0.0143 |
| σ=0.003 / r1 | 125.8 | 65.1 | 0.52 | 14.3 | 342.0 | 0.0201 |
| σ=0.003 / r2 | 114.1 | 57.7 | 0.51 | 6.4 | 300.8 | 0.0163 |
| σ=0.003 / r3 | 105.8 | 55.7 | 0.53 | 4.4 | 338.5 | 0.0142 |
| **σ=0.008 / r2** | **68.4** | **59.7** | **0.87** | **2.4** | **582.6** | **0.0082** |
| **σ=0.008 / r3** | **209.6** | **138.1** | **0.66** | **1.1** | **795.2** | **0.0628** |

### 12.3 Key observations

**1. v_A = 0.008 is near or in the fluid phase (even without disorder).**
The clean runs (σ = 0) show mean displacements of ~100 units and D ≈ 0.013.
This is non-zero diffusion, indicating the system is not deeply jammed at this
motility. This is consistent with v_A = 0.008 being near the transition.

**2. Weak disorder (σ = 0.003) barely changes the dynamics.**
Mean displacements and D values are similar to the clean case (D ≈ 0.014–0.020).
The coefficient of variation (CV ≈ 0.51–0.53) is unchanged from clean.
This suggests σ = 0.003 is too weak to enter the Griffiths regime.

**3. Strong disorder (σ = 0.008) shows clear Griffiths signatures:**

- **Dramatically broadened displacement distribution:**
  CV jumps to 0.66–0.87 (vs 0.51–0.53 for clean).
  
- **Enormous run-to-run variation:**
  Mean displacement ranges from 68.4 (r2) to 209.6 (r3) — a 3× difference!
  This reflects the sensitivity to the specific disorder realization (quenched randomness).
  
- **Extreme outlier displacements:**
  Maximum single-cell displacement of 795.2 units (half the domain!) in r3,
  versus ~300 for clean runs.
  
- **Coexistence of mobile and immobile cells:**
  Min displacement as low as 1.1 (essentially caged) alongside max 795.
  Some cells are fully jammed while others are highly fluid — within the same run.

**4. The disorder realization matters enormously for σ = 0.008.**
Run 2 (D = 0.008) is *less* mobile than the clean system, while run 3 (D = 0.063)
is 5× *more* mobile. This huge sensitivity to the specific {v_{A,i}} draw is
characteristic of the Griffiths phase where rare spatial regions dominate.

### 12.4 Interpretation

These preliminary results are consistent with the Griffiths rare-region picture:

- At σ = 0.008 (σ/v̄_A = 1.0), the per-cell motility distribution is broad enough
  that some cells draw v_{A,i} well above the clean transition threshold while others
  draw v_{A,i} well below it.
  
- The resulting spatial heterogeneity in motility produces persistent dynamical
  heterogeneity: patches of fluid cells embedded in a jammed matrix (or vice versa).
  
- The huge run-to-run variation reflects that different random realizations of the
  disorder can produce very different numbers of rare fluid/jammed clusters.

### 12.5 Next steps

1. **Compute Q(t) overlap function** for completed runs — look for power-law vs
   exponential decay
2. **Per-cell D_i vs v_{A,i} correlation** — verify that individual cell diffusivity
   correlates with its assigned motility
3. **Spatial mobility maps** — visualize where mobile/immobile cells are located
4. ~~Wait for all 18 runs to complete for full statistical analysis~~
   → **17/18 done** (only σ=0.000/run_1 still running at t≈256k)
5. **Compute MSD curves** with proper time averaging for clean vs disordered comparison
6. **Repeat all analysis at 89%** once those runs complete
7. **Run Griffiths production at N=1152 and N=4608** once equilibrated

---

## 13. Session 4 — Queue Cleanup & 89% Launch (Feb 11, 2026)

### 13.1 Actions taken

1. **85% study nearly complete.** 17 of 18 runs reached T_end = 331,000.
   Only `vA_0.008_sigma_0.000/run_1` still running (t ≈ 256k). Total disk: 54 GB.

2. **Cancelled excess 85% chain jobs.** ~407 pending chains for already-completed
   runs were blocking the queue. Cancelled all except the one running chain.

3. **89% Griffiths study submitted** (previous session).
   - Same 6-config grid: sigma scan (0.000, 0.003, 0.006, 0.008 at v_A=0.008)
     plus v_A scan (0.006, 0.010 at sigma=0.006)
   - 3 runs each, 15 chains = 271 jobs pending
   - Equilibration base: `/scratch/ssilber/eq_phi89/`
   - Output: `/scratch/ssilber/griffiths_study_phi89/`
   - Parameters: N=288, L=1562 (φ=89%), R=49, dt=0.02

4. **Larger-system equilibrations submitted** (previous session).
   - N=1152, L=3125, 3 runs × 6 chains = 18 jobs
   - N=4608, L=6249, 3 runs × 20 chains = 58 jobs
   - Both at v_A=0, φ=89%, dt=0.02, t_end=80,000

### 13.2 Current queue state

| Category | Running | Pending |
|:---|---:|---:|
| 85% Griffiths (last run) | 1 | 0 |
| 89% Griffiths | 0 | 271 |
| Equilibration 1152 | 0 | 18 |
| Equilibration 4608 | 0 | 58 |
| **Total** | **1** | **347** |

### 13.3 Why both packing fractions?

Running at both 85% and 89% serves two purposes:

1. **Locate the transition more precisely.** If 85% is fluidish (D > 0 even at σ=0,
   as preliminary results suggest), then 89% may be firmly jammed. This would
   bracket v_A^* and let us map the Griffiths region from both sides.

2. **Stronger Griffiths signatures at 89%.** If the clean system at 89% is jammed
   (D ≈ 0), then disorder-induced rare fluid pockets will contrast more sharply
   against the jammed background, making Q(t) power-law tails more visible.

---

*Last updated: February 15, 2026 — Session 5 (4608c equilibrations verified; GPU decision matrix; 4608c griffiths submitted)*

---

## 15. Session 5 — 4608c Griffiths Launch & Infrastructure Overhaul (Feb 15, 2026)

### 15.1 4608c Equilibration Status: Complete

All three N=4608 equilibration runs reached t=80,000:

| Run | Final time | Checkpoint size | RMS velocity | Domain |
|:----|:-----------|:---------------|:-------------|:-------|
| run_1 | 80,000 | 447 MB | 4.4×10⁻⁶ | 6249×6249 |
| run_2 | 80,000 | 448 MB | ~10⁻⁶ | 6249×6249 |
| run_3 | 80,000 | 449 MB | ~10⁻⁶ | 6249×6249 |

Equilibrium confirmed: cell velocities are effectively zero (RMS ~10⁻⁶).
No VTK frames were saved during equilibration (only checkpoint + trajectory).
Checkpoint visualizations generated on cluster — each shows the full 4608-cell
system alongside a zoomed 288-cell subdomain.  Cells are tightly packed at
φ=89% with no visible gaps or overlaps.

### 15.2 ⚠️ 3 replicates only

The 4608c system has **only 3 equilibration runs** (vs 100 for 288c).  This
means all large-system Griffiths results will be based on n=3 replicates per
parameter combination.  This is a significant statistical limitation:

- **Confidence intervals** will be wide; individual outlier runs can dominate
- **Run-to-run variation** is expected to be large (as seen in 288c σ=0.008)
- **Cannot compute robust error bars** for MSD or diffusion coefficients
- The 4608c results should be interpreted as **exploratory/qualitative**
  rather than statistically definitive

If initial results show interesting structure, additional equilibrations
(5-10 more runs) should be created.  Each equilibration costs ~5h of H100
time (t=80k at 4.7 t/s ≈ 4.7h).

### 15.3 GPU decision matrix overhaul

The `submit_job.sh` script was updated with a GPU decision matrix:

| Cells | GPU GRES | Partition | Walltime | TIME_PER_JOB |
|:------|:---------|:----------|:---------|:-------------|
| ≤500 | `gpu:1` (generic, fastest scheduling) | b1 | 3h | 90,000 |
| 501–2000 | `gpu:h100:1` (full H100) | b1 | 3h | 160,000 |
| 2001+ | `gpu:h100:1` (full H100) | b2 | 12h | 180,000 |

Key changes:
- **288c uses generic `gpu:1`** for fastest scheduling. This accesses the
  full ~200+ GPU pool (all nodes). Chain count (10) uses worst-case MIG rate
  (10 t/s), so job completes regardless of GPU assignment. Getting a full
  H100 is a bonus (59 t/s → finishes in 1-2 chains).
- **1152c uses 3h (b1) instead of 12h (b2)**: at 17.3 t/s on full H100,
  3h gives ~160k t.u. per job — enough for 5 chains to cover 800k t.u.
  Lower queue cost than 12h jobs.
- **`--eq-base` now requires `-n`**: prevents accidental use of 288c
  resource defaults when running with different cell counts.
- **Exact MIG GRES names discovered** via `sinfo -o '%G'`:
  `gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`, `..._2g.20gb:1`, `..._3g.40gb:1`
  (Available for manual override via `--gpu-type` if needed.)

### 15.4 Queue cleanup

Cancelled 457 excess chain jobs from 288c griffiths (had 35-36 per run from
double-submission; reduced to 10 per run, 180 total).

### 15.5 4608c Griffiths submitted

Submitted 18 runs × 5 chains = 90 jobs:
- Same 6-config Griffiths grid as 288c and 1152c studies
- GPU: full H100, 12h walltime (b2)
- Equilibration base: `/scratch/ssilber/eq_4608_phi89/`
- Output: `/scratch/ssilber/griffiths_v2/`
- Expected wall time: 4.7 t/s × 43200s = ~203k t.u. per job,
  800k total → 4-5 jobs per chain

---

## Session 16 — GPU Decision Matrix Revision & Equilibration Verification (Feb 15 2026)

### 16.1 GPU decision matrix revised

The previous version (Session 15.3) used explicit MIG GRES names
(`gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`) for ≤500 cells. This was reverted
to generic `gpu:1` after recognizing two problems:

1. **Explicit MIG name restricts scheduling to ~8 MIG nodes (g30–g37).**
   Generic `gpu:1` accesses the full ~200+ GPU pool across all nodes,
   leading to much faster scheduling.
2. **Chain counts already handle the worst case.** With TIME_PER_JOB=90,000
   based on the MIG rate (10 t/s), a 288c run chains ~10 jobs to reach
   t=880k. If the scheduler happens to assign a full H100 (59 t/s), the
   run finishes in 1–2 chains instead — a scheduling bonus, not a problem.

Final matrix (now deployed to cluster):

| Cells | `--gres` | Walltime | TIME_PER_JOB | Rationale |
|:------|:---------|:---------|:-------------|:----------|
| ≤500 | `gpu:1` | 3h (b1) | 90,000 | Fastest scheduling; chain count handles MIG worst-case |
| 501–2000 | `gpu:h100:1` | 3h (b1) | 160,000 | MIG impractical for medium domains |
| 2001+ | `gpu:h100:1` | 12h (b2) | 180,000 | Full H100 required; 12h for meaningful progress |

Updated: `submit_job.sh`, `cluster-operations.instructions.md`,
`cell-simulation.instructions.md`, `LOG_BOOK.md` Session 15.3.

### 16.2 Equilibration verification — 4608c velocity convergence

Extracted RMS velocity at all saved timepoints from
`/scratch/ssilber/eq_4608_phi89/run_{1,2,3}/trajectory.txt` (4608 cells,
v_A=0, φ=0.89, dt=0.02, trajectory interval ~800 t.u.).

**RMS velocity vs time (averaged over 4608 cells per snapshot):**

| Time | Run 1 | Run 2 | Run 3 | Avg |
|-----:|------:|------:|------:|----:|
| 800 | 1.58e-4 | 1.70e-4 | 1.85e-4 | 1.71e-4 |
| 5,000 | 3.67e-5 | 4.13e-5 | 3.95e-5 | 3.92e-5 |
| 10,000 | 2.31e-5 | 2.36e-5 | 2.12e-5 | 2.26e-5 |
| 20,000 | 1.27e-5 | 1.26e-5 | 1.16e-5 | 1.23e-5 |
| 30,000 | 8.26e-6 | 9.39e-6 | 8.88e-6 | 8.84e-6 |
| 40,000 | 7.84e-6 | 7.26e-6 | 7.14e-6 | 7.41e-6 |
| 50,000 | 6.01e-6 | 6.36e-6 | 5.46e-6 | 5.94e-6 |
| 60,000 | 5.44e-6 | 5.60e-6 | 5.67e-6 | 5.57e-6 |
| 70,000 | 5.10e-6 | 6.01e-6 | 4.81e-6 | 5.31e-6 |
| 80,000 | 4.44e-6 | 6.08e-6 | 4.80e-6 | 5.11e-6 |

**Convergence analysis:**

- **Transient (t=0→10k):** 7.5× velocity drop (1.71e-4 → 2.26e-5).
  Rapid relaxation from random initial placement.
- **Intermediate (t=10k→40k):** 3× drop (2.26e-5 → 7.41e-6).
  Significant structural rearrangement.
- **Plateau (t=40k→80k):** Only 31% decrease (7.41e-6 → 5.11e-6) over
  40k t.u. Small periodic oscillations (±1e-6, period ~7k) around a
  slowly-decaying baseline — characteristic of residual collective modes.
- **t=60k→80k:** 8% decrease (5.57e-6 → 5.11e-6). Effectively flat.

**Physical justification for velocity as equilibrium criterion:**
With v_A=0 the dynamics are purely relaxational (Model B / overdamped).
Cell velocity v_i = −∇_i E / γ, so |v| → 0 is the exact mechanical
equilibrium condition. In practice, floating-point precision and slow
collective modes prevent exact zero, but RMS|v| ~ 5e-6 means a cell
drifts < R/3 over an entire 800k production run (even if velocity stayed
constant, which it doesn't — it continues to decay).

**Conclusion:** t=80k is sufficient for 4608 cells. The velocity has
decreased by ~34× from its initial value and is plateauing. The residual
drift (~0.3R over 800k t.u.) is negligible compared to motility-driven
displacements in production (v_A ~ 0.004–0.013 → displacements of tens
of R per run).

> **Note:** Run 2 is noisier (ends at 6.08e-6 vs ~4.6e-6 for runs 1 & 3),
> showing larger periodic oscillations that haven't fully damped. All runs
> are still well within the plateau regime. For future studies at even
> larger cell counts, monitoring the velocity time series explicitly
> (rather than relying on a fixed t_eq from paper parameters) is
> recommended.

---

## 14. Griffiths Rare-Region Physics — First-Principles Derivation

> **Purpose:** A self-contained tutorial deriving Griffiths rare-region effects
> from scratch, with enough detail to independently verify every prediction
> against our simulation data. Written February 14, 2026 after preliminary
> results showed **anti-Griffiths** behavior (see `research_logbook.md`).

---

### 14.1 The Original Griffiths Singularity (1969)

#### The setup: dilute Ising ferromagnet

Griffiths' original argument (Phys. Rev. Lett. 23, 17, 1969) considers a
simple system: a $d$-dimensional Ising ferromagnet on a lattice where each
site is occupied with probability $p$ and vacant with probability $1 - p$.
Occupied sites interact ferromagnetically; vacant sites are inert.

The **clean** Ising model ($p = 1$, no vacancies) has a sharp phase transition
at a critical temperature $T_c$:

$$\begin{cases}
T < T_c: & \text{ordered (ferromagnetic), spontaneous magnetization } m > 0 \\
T > T_c: & \text{disordered (paramagnetic), } m = 0
\end{cases}$$

With dilution ($p < 1$), the critical temperature decreases:
$T_c(p) < T_c(1)$, because vacancies weaken the ferromagnetic coupling.
For $p$ below the percolation threshold $p_c$, there is no ordered phase
at any temperature.

#### The Griffiths phase

Griffiths showed that in the temperature range:

$$T_c(p) < T < T_c(1)$$

the **free energy is non-analytic** in the external magnetic field $h$,
even though the system is macroscopically disordered (no long-range order).

This region is called the **Griffiths phase**. It extends from the actual
critical temperature of the diluted system up to the critical temperature
of the clean system:

```
        Ordered          Griffiths Phase            Paramagnetic
    ◄──────────┼──────────────────────────┼──────────────────►  T
              T_c(p)                     T_c(1)
              (diluted)                  (clean)
```

#### The mathematical argument

The proof relies on a remarkably simple observation:

1. **Rare regions exist.** In a randomly diluted lattice, there will be
   arbitrarily large connected regions that happen to have no vacancies.
   A fully occupied region of linear size $L$ (containing $\sim L^d$ sites)
   occurs with probability:

   $$P(L) = p^{L^d} = e^{-c L^d}, \quad c = -\ln p > 0$$

   These regions are **exponentially rare** in their volume, but they exist
   with certainty in an infinite system.

2. **Each rare region is locally ordered.** A fully occupied region of size $L$
   behaves like a small clean Ising model. If $T < T_c(1)$, this region is
   locally in its ordered phase. Its magnetization reversal requires a
   coherent flip of all $\sim L^d$ spins, which has an energy barrier:

   $$\Delta E(L) \propto L^{d-1} \quad \text{(surface energy of a domain wall)}$$

   For $d \geq 2$, this barrier grows with $L$, so the relaxation time is:

   $$\tau(L) \sim \tau_0 \, e^{\Delta E(L) / k_B T} \sim e^{b L^{d-1}}$$

   where $b = \Sigma / k_B T$ and $\Sigma$ is the domain-wall surface tension.

3. **Competition between rarity and slowness produces anomalous relaxation.**

   Every physical response of the system — how it responds to a probe field,
   how it loses memory of its initial state — is a sum over contributions
   from all possible rare regions, weighted by their probability. The key
   insight is that large regions are exponentially rare ($P \sim e^{-cL^d}$)
   but also exponentially slow ($\tau \sim e^{bL^{d-1}}$), so the dominant
   contribution comes from a **specific intermediate size** $L^*$ that
   balances these two exponentials.

   We derive this in two ways: first for the autocorrelation $C(t)$
   (conceptually cleaner), then for the dynamic susceptibility
   $\text{Im}\,\chi(\omega)$.

   ---

   **The autocorrelation function $C(t)$.**

   Define the order-parameter autocorrelation:

   $$C(t) = \frac{1}{N}\sum_{i=1}^{N} \langle s_i(0) \, s_i(t) \rangle$$

   where $s_i$ is the spin (or, in our tissue, a dynamical variable like
   the cage overlap for cell $i$). $C(t)$ measures how much the system
   remembers its configuration after time $t$.

   Each rare region of linear size $L$ is locally ordered and relaxes
   independently on its own timescale $\tau(L) = e^{bL^{d-1}}$, contributing
   a decaying exponential $e^{-t/\tau(L)}$ to $C(t)$. The full autocorrelation
   is a sum over all rare regions weighted by their probability:

   $$C(t) \sim \int_0^\infty dL \; \underbrace{e^{-c L^d}}_{\text{probability } P(L)} \;\; \underbrace{e^{-t \, e^{-b L^{d-1}}}}_{\text{decay factor } e^{-t/\tau(L)}}$$

   At long times, the integrand is sharply peaked around a saddle point
   $L = L^*(t)$ that we find by optimizing the exponent. Write the total
   exponent as:

   $$\Phi(L) = -c L^d - t \, e^{-b L^{d-1}}$$

   Setting $d\Phi/dL = 0$:

   $$c \, d \, L^{d-1} = t \cdot b(d-1) L^{d-2} \, e^{-b L^{d-1}}$$

   For large $t$, the right side must be large, which forces $L$ to be
   large so that $e^{-bL^{d-1}}$ is not too small. Specifically, taking
   logs of the dominant balance $t \cdot e^{-bL^{d-1}} \sim 1$ gives:

   $$b \, L^{*\,d-1} \sim \ln t \quad \implies \quad L^*(t) \sim \left(\frac{\ln t}{b}\right)^{1/(d-1)}$$

   **Physical meaning of $L^*$**: at time $t$, the dominant contribution to
   $C(t)$ comes from rare regions of linear size $L^*$. Smaller regions
   ($L < L^*$) have already fully relaxed ($\tau(L) \ll t$, so
   $e^{-t/\tau} \approx 0$). Larger regions ($L > L^*$) are so rare that
   their probability $P(L) = e^{-cL^d}$ is negligible. The physics lives
   at the crossover size $L^*$ where **just-barely-surviving regions are
   just-barely-probable**.

   Now substitute $L^*$ back into $P(L^*)$:

   $$P(L^*) = e^{-c \, L^{*\,d}} = \exp\!\left(-c \left(\frac{\ln t}{b}\right)^{d/(d-1)}\right) = \exp\!\left(-\frac{c}{b^{d/(d-1)}} (\ln t)^{d/(d-1)}\right)$$

   Defining $\Lambda \equiv c / b^{d/(d-1)}$:

   $$\boxed{C(t) \sim \exp\!\left(-\Lambda \, (\ln t)^{d/(d-1)}\right), \quad \Lambda = \frac{c}{b^{d/(d-1)}}}$$

   This is **slower than any power law** but faster than any stretched
   exponential $e^{-(t/\tau)^\beta}$ with $\beta > 0$. In $d = 2$ (our case):

   $$C(t) \sim \exp\!\left(-\Lambda \, (\ln t)^2\right) \quad \text{(2D)}$$

   ---

   **The dynamic susceptibility $\text{Im}\,\chi(\omega)$.**

   The dynamic susceptibility $\chi(\omega)$ measures the system's linear
   response to an oscillating external field $h(t) = h_0 e^{-i\omega t}$.
   It is complex: $\chi(\omega) = \text{Re}\,\chi + i\,\text{Im}\,\chi$.
   The imaginary part $\text{Im}\,\chi(\omega)$ quantifies the **energy
   absorbed** (dissipation) at frequency $\omega$.

   The connection to $C(t)$ comes from the **fluctuation-dissipation
   theorem** (FDT). In thermal equilibrium, the way a system responds to
   a perturbation is identical to the way it relaxes from a spontaneous
   fluctuation. Specifically, the causal response function $R(t)$
   (the response at time $t$ to a delta-function kick at $t=0$) is:

   $$R(t) = -\frac{1}{k_BT}\frac{dC}{dt} \quad (t > 0)$$

   The susceptibility is the Fourier transform of $R(t)$:
   $\chi(\omega) = \int_0^\infty R(t)\,e^{i\omega t}\,dt$. Substituting
   $R = -\dot{C}/k_BT$ and integrating by parts:

   $$\text{Im}\,\chi(\omega) = \frac{\omega}{k_BT} \int_0^\infty C(t) \cos(\omega t) \, dt$$

   For a single rare region with relaxation time $\tau$, $C(t) = C_0 e^{-t/\tau}$,
   and evaluating the integral gives the **Debye loss function**:

   $$\text{Im}\,\chi_\tau(\omega) = \frac{C_0}{k_BT} \cdot \frac{\omega\tau}{1 + \omega^2\tau^2}$$

   This is peaked at $\omega = 1/\tau$ — a rare region absorbs energy most
   efficiently when the driving frequency matches its natural relaxation
   rate. This is the same principle as resonance: the $\cos(\omega t)$
   oscillation stays in phase with the $e^{-t/\tau}$ decay only when
   $\omega\tau \sim 1$. For $\omega \ll 1/\tau$ the region equilibrates
   too fast to absorb (it tracks the field adiabatically); for
   $\omega \gg 1/\tau$ the field oscillates too fast for the sluggish
   region to respond at all. Maximum absorption occurs at the crossover.

   This is why $\text{Im}\,\chi(\omega)$ is dominated by regions with
   $\tau(L) \sim 1/\omega$.

   Inverting $\tau(L) = e^{bL^{d-1}} = 1/\omega$ gives the resonant size:

   $$L^*(\omega) = \left(\frac{\ln(1/\omega)}{b}\right)^{1/(d-1)}$$

   Each such region contributes $\sim L^{*\,d}/\omega$ to $\text{Im}\,\chi$
   (the factor $L^d$ is the cluster volume, i.e. how many spins respond).
   Weighted by probability $P(L^*) = e^{-cL^{*\,d}}$:

   $$\text{Im}\,\chi(\omega) \sim \frac{L^{*\,d}}{\omega} \, P(L^*) = \frac{1}{\omega}\left(\frac{\ln(1/\omega)}{b}\right)^{d/(d-1)} \exp\!\left(-\Lambda (\ln(1/\omega))^{d/(d-1)}\right)$$

   This has the same stretched-logarithmic form as $C(t)$ (as it must,
   since they are Fourier-transform related). Note: this is NOT a power
   law in $\omega$ for $d \geq 2$. (A power law
   $\text{Im}\,\chi \sim \omega^{\alpha-1}$ arises only in $d = 1$ — see
   below.)

   #### Special case: $d = 1$ (Ising chain)

   In 1D, domain-wall energy barriers are constant ($\Delta E = 2J$,
   independent of $L$). However, if each added spin in the cluster
   multiplicatively increases the relaxation time — as in the quantum
   random transverse-field Ising model, or when each cell in a jammed
   chain adds a sequential barrier — then $\tau(L) \sim e^{bL}$ and the
   autocorrelation becomes a **pure power law**:

   $$C(t) \sim t^{-\alpha}, \quad \alpha = c/b$$

   (Here $c$ and $b$ are still distinct constants: $c$ from the rarity
   $P(L) \sim e^{-cL}$ and $b$ from the barrier $\tau \sim e^{bL}$.)

   While our system is 2D, the effective dimensionality of rare regions
   may be reduced by the geometry of jammed clusters (quasi-1D percolation
   paths), potentially bringing us closer to power-law behavior.

#### Why this matters: non-analyticity of the static free energy

The free energy $f(h)$ as a function of external field $h$ has an
**essential singularity** at $h = 0$ throughout the Griffiths phase:

$$f_{\text{sing}}(h) \sim e^{-A/|h|}$$

> **Note:** The exponent $d/(d-1)$ that appears in the dynamical
> autocorrelation $C(t) \sim e^{-\Lambda(\ln t)^{d/(d-1)}}$ does NOT
> appear here. The **static** free energy singularity has the simpler form
> $e^{-A/|h|}$ in all dimensions $d$. See derivation below.

#### Derivation of $e^{-A/|h|}$ via Lee-Yang zeros

The cleanest route to the functional form uses the **Lee-Yang theorem**:
for a ferromagnet, all zeros of the partition function $Z(h)$ in the complex
magnetic field plane lie on the imaginary axis ($\text{Re}(h) = 0$).

**Step 1: Zeros from a single rare cluster.**
A locally ordered cluster of $n$ spins behaves as a giant two-state magnet
with moment $\mu = m_0 n$ (where $m_0(T)$ is the local spontaneous
magnetization). Its partition function is:

$$Z_n(h) = 2\cosh(\beta \mu h) = 2\cosh(\beta m_0 n h)$$

This has zeros at purely imaginary field values:

$$h_k = \frac{i\pi(2k+1)}{2\beta m_0 n}, \quad k = 0, \pm 1, \pm 2, \ldots$$

The **nearest zero** to $h = 0$ is at distance:

$$|h_{\min}| = \frac{\pi}{2\beta m_0 n}$$

So a cluster of $n$ spins puts a zero at imaginary-$h$ distance $\sim 1/n$
from the origin.

**Step 2: Density of zeros from all rare clusters.**
In the diluted system, the density of clusters of size $n$ per unit volume
is $\rho(n) \sim e^{-cn}$ where $c = -\ln p$. Each such cluster contributes
zeros at $|\text{Im}(h)| \sim 1/n$. Changing variables from $n$ to
$\eta = \pi/(2\beta m_0 n)$ (imaginary-$h$ distance of the closest zero):

$$n = \frac{\pi}{2\beta m_0 \eta}, \quad dn = -\frac{\pi}{2\beta m_0 \eta^2} d\eta$$

The density of zeros on the imaginary-$h$ axis near $\eta = 0$ is:

$$g(\eta) \sim \rho(n(\eta)) \cdot |dn/d\eta| \sim e^{-c\pi/(2\beta m_0 \eta)} \cdot \eta^{-2}$$

$$\boxed{g(\eta) \sim \frac{1}{\eta^2} \exp\!\left(-\frac{A}{\eta}\right), \quad A = \frac{c\pi}{2\beta m_0}}$$

As $\eta \to 0$, the density of zeros goes to zero **faster than any power
of $\eta$** (it vanishes like $e^{-A/\eta}$). This is why the system is
NOT critical (no phase transition at finite $T$ in the Griffiths phase) —
but the zeros DO accumulate toward $h = 0$, which is enough to break
analyticity.

**Step 3: From zeros to free energy.**
The free energy per site is related to the density of Lee-Yang zeros by:

$$f(h) = \int d\eta \; g(\eta) \; \ln\!\left(h^2 + \eta^2\right) + \text{analytic part}$$

For real $h \to 0^+$, the non-analytic contribution comes from the crossover
region $\eta \sim h$ (where $\ln(h^2 + \eta^2)$ transitions between
$\ln h^2$ and $\ln \eta^2$):

$$f_{\text{sing}}(h) \sim g(h) \cdot h \sim \frac{1}{h^2} \cdot e^{-A/h} \cdot h = \frac{e^{-A/|h|}}{|h|}$$

Dropping the algebraic prefactor (which is subleading to the essential
singularity):

$$f_{\text{sing}}(h) \sim e^{-A/|h|}$$

**Step 4: Why the Taylor series has zero radius of convergence.**
Alternatively, one can see this directly from the high-order susceptibilities.
The $n$-th cumulant susceptibility (connected $n$-point function) receives
contributions from all cluster sizes. A cluster of $V$ spins contributes
$\sim (\beta m_0)^n V^n$ to the $n$-th cumulant, weighted by the cluster
probability $e^{-cV}$. Summing (integrating) over cluster sizes:

$$\kappa_n \sim \int_0^\infty V^n \, e^{-cV} \, dV \cdot (\beta m_0)^n = \frac{n!}{c^{n+1}} \cdot (\beta m_0)^n$$

The Taylor coefficient of $h^n$ in the free energy is:

$$a_n = \frac{\kappa_n}{n!} \sim \frac{(\beta m_0)^n}{c^{n+1}}$$

These coefficients do NOT grow factorially — but the full susceptibilities
$\chi_n$ also involve the Bernoulli numbers $B_{2k}$ from derivatives of
$\ln\cosh$. Since $|B_{2k}| \sim 2(2k)!/(2\pi)^{2k}$, the even-order
susceptibilities grow as:

$$\chi_{2k} \sim \frac{(2k)!}{c^{2k+1}} (\beta m_0)^{2k} \cdot \frac{2(2k)!}{(2\pi)^{2k}}$$

The Taylor coefficient $a_{2k} = \chi_{2k}/(2k)!$ retains one power of
$(2k)!$:

$$a_{2k} \sim \frac{(2k)!}{c^{2k+1}} \left(\frac{\beta m_0}{2\pi}\right)^{2k}$$

By Stirling, $|a_{2k+2}/a_{2k}| \sim (2k+2)(\beta m_0/(2\pi c))^2 \to \infty$
as $k \to \infty$. **The radius of convergence is zero.** Every Taylor
coefficient is finite, but the series diverges for any $h \neq 0$.

The Borel sum of this divergent series recovers the $e^{-A/|h|}$ form — this
is the standard connection between factorially divergent perturbation series
and non-perturbative effects.

#### Summary: static vs. dynamic Griffiths singularities

| Quantity | Functional form | Origin of exponent |
|:---|:---|:---|
| Free energy $f(h)$ | $e^{-A/\|h\|}$ | Cluster field energy $\sim V h$; probability $\sim e^{-cV}$; crossover $V^* \sim 1/h$ |
| Autocorrelation $C(t)$ | $e^{-\Lambda(\ln t)^{d/(d-1)}}$ | Barrier $\sim L^{d-1}$; probability $\sim e^{-cL^d}$; crossover $L^* \sim (\ln t)^{1/(d-1)}$; $\Lambda = c/b^{d/(d-1)}$ |
| Autocorrelation ($d=1$) | $t^{-c/b}$ (power law) | Sequential barriers; $\tau \sim e^{bL}$; probability $\sim e^{-cL}$ |

The crucial difference: in the static case, the field $h$ couples to the
**volume** ($V$) directly, giving $V^* \sim 1/h$ and a simple essential
singularity. In the dynamic case, the relaxation barrier couples to the
**surface** ($L^{d-1}$), requiring the logarithmic inversion $L^* \sim (\ln t)^{1/(d-1)}$
which introduces the dimension-dependent exponent $d/(d-1)$.

This is why Griffiths effects are called "non-perturbative" — they arise
from exponentially rare events that are invisible to any finite-order
expansion around the clean system.

---

### 14.2 Mapping to Active Tissue Jamming

#### The correspondence

To apply Griffiths physics to our cell simulation, we need a mapping between
the Ising magnet and the active tissue:

| Ising Magnet | Active Tissue |
|:---|:---|
| Lattice site | Cell $i$ |
| Spin state $s_i = \pm 1$ | Dynamical state: jammed/unjammed |
| Temperature $T$ | Mean motility $\bar{v}_A$ |
| Vacancy (dilution) | Low $v_{A,i}$ (quenched motility disorder) |
| Ordered phase ($T < T_c$) | Jammed tissue ($\bar{v}_A < v_A^*$) |
| Disordered phase ($T > T_c$) | Fluid tissue ($\bar{v}_A > v_A^*$) |
| $T_c(1)$ (clean) | $v_A^*$ (clean jamming transition) |
| $T_c(p)$ (diluted) | $v_A^*(p)$ (shifted transition with disorder) |
| Magnetization $m$ | Cage order parameter $Q(\infty)$ |
| Magnetic field $h$ | External drive / anisotropy |

**Critical difference:** In the Ising model, the "disorder" (vacancies)
reduces the order (weakens ferromagnetism). In our system, the disorder
(motility heterogeneity) can either promote or inhibit jamming depending
on which cells get the disorder. A cell with $v_{A,i} \gg v_A^*$ is locally
unjammed; a cell with $v_{A,i} \ll v_A^*$ is locally jammed.

#### The Griffiths region in $(v_A, \sigma)$ space

With disorder strength $\sigma$ (standard deviation of the $v_{A,i}$
distribution), the Griffiths region is:

$$v_A^*(\sigma) < \bar{v}_A < v_A^*(0)$$

where $v_A^*(\sigma)$ is the (lowered) transition of the disordered system
and $v_A^*(0)$ is the clean transition.

For $\bar{v}_A$ in this region:
- The **bulk** of the system is jammed (most cells have $v_{A,i} < v_A^*$)
- **Rare clusters** of cells where all members have $v_{A,i} > v_A^*$ are
  locally fluid
- These fluid clusters are the "rare regions" that produce Griffiths effects

Conversely, for $\bar{v}_A$ slightly above $v_A^*$ (on the fluid side):
- The bulk is fluid
- Rare clusters where all cells have $v_{A,i} < v_A^*$ form **jammed islands**
- These islands are the rare regions from the other side of the transition

#### Key assumption: locality

The Griffiths argument requires that the dynamical state of a region is
determined **locally** — i.e., a cluster of high-$v_A$ cells behaves like a
small fluid tissue regardless of its surroundings. This is where the analogy
may break down in active matter, because:

1. **Motile cells push their neighbors.** Unlike immobile magnetic vacancies,
   a fast cell exerts force on adjacent slow cells. The "rare region" is not
   isolated from the bulk.

2. **Cells can rearrange.** In the Ising model, sites are fixed on a lattice.
   In our tissue, cells move, so the spatial arrangement of high-$v_A$ and
   low-$v_A$ cells evolves (their identities are quenched, but their
   positions are not).

3. **Excluded volume matters.** At $\phi = 0.89$, cells are tightly packed.
   A locally fluid region may be mechanically prevented from relaxing if
   its boundary is rigid.

These differences are precisely why simulation is needed — the Griffiths
prediction is well-defined, but whether it applies to a motile, deformable,
out-of-equilibrium tissue is an open question.

---

### 14.3 Detailed Derivation of Observables

#### 14.3.1 The self-overlap function $Q(t)$

Define:

$$Q(t) = \frac{1}{N} \sum_{i=1}^{N} \Theta\!\left(a - |\mathbf{r}_i(t) - \mathbf{r}_i(0)|\right)$$

where $a$ is a threshold distance (the "cage size") and $\Theta$ is the
Heaviside step function. $Q(t)$ is the fraction of cells that have not
moved beyond distance $a$ after time $t$.

**Clean system predictions:**

- **Jammed** ($\bar{v}_A < v_A^*$): All cells are caged, so $Q(t) \to Q_\infty > 0$
  as $t \to \infty$ (some fraction remains permanently caged)

- **Fluid** ($\bar{v}_A > v_A^*$): All cells eventually escape their cages, so
  $Q(t) \to 0$. The decay is exponential:
  $$Q(t) \sim e^{-t/\tau_\alpha}$$
  where $\tau_\alpha$ is the structural relaxation time.

- **At the transition** ($\bar{v}_A = v_A^*$): $Q(t)$ decays as a power law:
  $$Q(t) \sim t^{-\beta/\nu z}$$
  where $\beta$, $\nu$, $z$ are critical exponents.

**Griffiths phase predictions:**

In the Griffiths region ($\bar{v}_A$ near but not at $v_A^*$, with $\sigma > 0$):

Each cell $i$ has a local relaxation rate $\Gamma_i$ that depends on its
neighborhood. For a cell embedded in a jammed cluster of size $\ell$, the
relaxation rate is:

$$\Gamma(\ell) \sim e^{-b \ell^\psi}$$

where $\psi$ is a geometry-dependent exponent ($\psi = d - 1 = 1$ for 2D
droplet-like excitations, $\psi = 1$ for 1D-like stringy paths).

The overlap function is a weighted sum over all possible local environments:

$$Q(t) = \sum_\ell P(\ell) \, e^{-\Gamma(\ell) \, t} = \sum_\ell e^{-c \ell^d} \, e^{-e^{-b \ell^\psi} \, t}$$

The long-time behavior is dominated by the saddle point in $\ell$. Setting
$c \ell^d = e^{-b \ell^\psi} t$ and solving:

- For **$d = 2$, $\psi = 1$** (2D droplet excitations):
  $$Q(t) \sim \exp\!\left(-A \, (\ln t)^{d/\psi}\right) = \exp\!\left(-A \, (\ln t)^2\right)$$

- For **effective $\psi = d$** (compact rare regions):
  $$Q(t) \sim t^{-\alpha}, \quad \alpha = c/b$$
  This is a **pure power law** with continuously varying exponent $\alpha$.

The distinction matters: pure power-law $Q(t)$ is the strongest Griffiths
signature (equivalent to 1D-like or random-field physics), while the
$\exp(-(\ln t)^2)$ form is weaker but still qualitatively different from
exponential.

**How to test in our data:**

Plot $\ln Q(t)$ vs $\ln t$:
- Exponential: curves downward (concave)
- Power law: straight line with slope $-\alpha$
- $\exp(-(\ln t)^2)$: straight line on $\ln Q$ vs $(\ln t)^2$ plot

**Stretched exponential** $Q(t) = \exp(-(t/\tau)^\beta)$ is the generic
glass phenomenology. On a $\ln Q$ vs $\ln t$ plot, it curves but may appear
locally linear. To distinguish from power law, need $\geq 2$ decades in time.

#### 14.3.2 The non-Gaussian parameter $\alpha_2(t)$

Define the displacement $\Delta r_i(t) = |\mathbf{r}_i(t) - \mathbf{r}_i(0)|$.
Then:

$$\alpha_2(t) = \frac{\langle \Delta r^4(t) \rangle}{(1 + 2/d) \langle \Delta r^2(t) \rangle^2} - 1$$

In $d = 2$:

$$\alpha_2(t) = \frac{\langle \Delta r^4(t) \rangle}{2 \langle \Delta r^2(t) \rangle^2} - 1$$

For a Gaussian displacement distribution, $\alpha_2 = 0$. Positive $\alpha_2$
indicates heavy tails (some cells move much more or much less than average).

**Clean system:** $\alpha_2(t)$ peaks at $t \sim \tau_\alpha$ (when caging
effects are strongest) and then decays back to zero as the system enters
the diffusive regime and the central limit theorem takes over.

**Griffiths prediction:** $\alpha_2(t)$ remains **persistently elevated**
for all $t$ in the Griffiths phase. This is because the displacement
distribution never becomes Gaussian — the quenched disorder permanently
separates cells into fast and slow populations. Specifically:

$$\alpha_2(t \to \infty) = \frac{\langle D_i^2 \rangle}{2 \langle D_i \rangle^2} - 1 \approx \frac{\text{Var}(D_i)}{2 \langle D_i \rangle^2}$$

where $D_i$ is the per-cell diffusion coefficient. If $\text{Var}(D_i) / \langle D_i \rangle^2 \sim O(1)$, then $\alpha_2 \sim O(1)$ even as $t \to \infty$.

**Verification calculation:** Given our disorder distribution with
$\bar{v}_A = 0.008$ and $\sigma = 0.008$, estimate the expected $\alpha_2$:

If $D_i \propto v_{A,i}^2$ (the diffusion coefficient scales as the square
of the motility in the persistent random walk limit $D = v_A^2 \tau / 2d$),
and $v_{A,i} \sim \mathcal{N}(0.008, 0.008^2)$ truncated at 0:

- $\langle D \rangle \propto \langle v_{A,i}^2 \rangle = \bar{v}_A^2 + \sigma^2 = 2 \times 0.008^2$
- $\langle D^2 \rangle \propto \langle v_{A,i}^4 \rangle = \bar{v}_A^4 + 6\bar{v}_A^2 \sigma^2 + 3\sigma^4 = 0.008^4(1 + 6 + 3) = 10 \times 0.008^4$
- $\alpha_2 = \langle D^2 \rangle / (2 \langle D \rangle^2) - 1 = 10 \times 0.008^4 / (2 \times 4 \times 0.008^4) - 1 = 10/8 - 1 = 0.25$

This is a **lower bound** because it assumes the free-particle relation
$D \propto v_A^2$. In a caged system, the dependence is more nonlinear
(cells below $v_A^*$ have $D \approx 0$), which amplifies $\alpha_2$.

**Our data:** We measured $\alpha_2 = 12.70$ at $\sigma = 0.008$ in the
288-cell study (research_logbook.md), far exceeding the free-particle
estimate. This indicates strong caging effects amplifying the disorder.

#### 14.3.3 The four-point susceptibility $\chi_4(t)$

$$\chi_4(t) = N \left[ \langle Q(t)^2 \rangle - \langle Q(t) \rangle^2 \right]$$

where the average is over different time origins (or different disorder
realizations). $\chi_4(t)$ measures **how much the relaxation varies from
one observation to another**.

**Physical meaning:** $\chi_4$ counts the number of cells in a
**cooperatively rearranging region** (CRR). A high peak means cells
rearrange in large correlated clusters; a low peak means cells relax
independently.

**Griffiths prediction:** This is where the active tissue deviates from
equilibrium Griffiths physics:

- **Equilibrium Griffiths** (Ising model): $\chi_4$ should **grow** with
  disorder, because rare regions create large correlated clusters that
  relax together.

- **Active matter** (our system): $\chi_4$ may **decrease** with disorder,
  because fast cells actively break up collective caging, causing cells to
  relax more independently.

**Our data shows the active-matter scenario:** $\chi_4$ drops from 38 (σ=0)
to 3.3 (σ=0.006). This is the core anti-Griffiths result. See Section 14.7
for the physical explanation.

#### 14.3.4 The stretching exponent $\beta$

Fit $Q(t)$ to a stretched exponential:

$$Q(t) = Q_\infty + (1 - Q_\infty) \exp\!\left(-(t/\tau_\alpha)^\beta\right)$$

The exponent $\beta$ characterizes the breadth of the relaxation spectrum:

| $\beta$ | Meaning | Physical picture |
|:---:|:---|:---|
| 1.0 | Simple exponential | Single relaxation time (single process) |
| 0.5–0.9 | Stretched exponential | Multiple timescales (typical glass) |
| $\to 0$ | Extremely stretched | Approaching power law (Griffiths) |
| > 1.0 | Compressed exponential | Faster-than-exponential (stress-driven) |

**Griffiths prediction:** $\beta$ should **decrease** with increasing $\sigma$,
because the quenched disorder creates a broader spectrum of local relaxation
rates (each cell/cluster has a different $\tau_i$).

**Our data:** $\beta$ **increases** from 0.57 (σ=0) to 0.90 (σ=0.006).
→ Anti-Griffiths. The disorder narrows the relaxation spectrum.

---

### 14.4 Time Scales — What Each One Means

| Time scale | Symbol | Expression | Our value | Physical meaning |
|:---|:---|:---|:---|:---|
| Integration step | $\delta t$ | — | 0.02 | Numerical discretization |
| Persistence time | $\tau$ | $1/D_r$ | 10,000 | Time between polarity reorientations |
| Run length | $\ell/v_A$ | $v_A \tau$ | 80 grid units | Distance a free cell travels before turning |
| Cage escape time | $\tau_{\text{cage}}$ | — | $\sim 10^3$–$10^4$ | Time for a cell to escape its neighbor cage |
| Structural relaxation | $\tau_\alpha$ | $Q(\tau_\alpha) = 1/e$ | $10^4$–$10^5$ | Time for the tissue to lose memory of its configuration |
| Rare-region relaxation | $\tau_{\text{rare}}(L)$ | $\sim e^{b L}$ | $10^4$–$10^6$ | Time for a jammed cluster of size $L$ to rearrange |
| Observation time | $T_{\text{obs}}$ | — | 250,000 | How long we run the simulation |

#### Hierarchy of time scales

For Griffiths effects to be observable, we need:

$$\delta t \ll \tau_{\text{cage}} \ll \tau_\alpha \ll \tau_{\text{rare}}(L_{\max}) \lesssim T_{\text{obs}}$$

1. $\delta t \ll \tau_{\text{cage}}$: We resolve individual cage dynamics.
   ✅ $0.02 \ll 10^3$.

2. $\tau_{\text{cage}} \ll \tau_\alpha$: There's a clear caging plateau in Q(t).
   ✅ We see caging plateaus in the σ=0 data.

3. $\tau_\alpha \ll \tau_{\text{rare}}(L_{\max})$: Rare regions relax more slowly
   than the bulk. This is the Griffiths signature.
   **Depends on $L_{\max}$** — with 288 cells, the largest rare region is
   $L_{\max} \sim 5$–10 cells, giving $\tau_{\text{rare}} \sim e^{5b}$–$e^{10b}$.

4. $\tau_{\text{rare}}(L_{\max}) \lesssim T_{\text{obs}}$: We must observe the simulation
   long enough to see these rare relaxation events.
   **Marginal** at $T_{\text{obs}} = 250,000$ for the 288-cell system.

#### Estimating $\tau_{\text{rare}}$

The rare-region relaxation time depends on the barrier exponent $b$. From
the Ising analogy, $b \propto J/(k_B T)$ — the ratio of coupling strength
to thermal fluctuation. In our system, the analogous ratio is:

$$b \sim \frac{\text{(cage stiffness)}}{\text{(active fluctuation strength)}} \sim \frac{\kappa R^2}{v_A^2 \tau}$$

With $\kappa = 10$, $R = 49$, $v_A = 0.008$, $\tau = 10{,}000$:

$$b \sim \frac{10 \times 49^2}{0.008^2 \times 10000} \sim \frac{240{,}100}{0.64} \sim 37{,}500$$

This is an enormous barrier exponent. If taken literally, $\tau_{\text{rare}}(L=2) \sim e^{75{,}000}$, which is astronomically long — rare regions of even $L = 2$ cells would never relax.

**But this estimate is too conservative** because:
- Active forcing is not thermal: cells push deterministically, not through random kicks
- The cage barrier is not a simple energy landscape but a dynamic constraint
- The effective barrier depends on the local configuration, not just global parameters

The practical question is whether the **measured** relaxation times show signs of
the exponential-in-$L$ scaling. From our Q(t) data:

- $\tau_\alpha(\sigma=0) \approx 72{,}000$ time units (stretched exp fit)
- $\tau_\alpha(\sigma=0.006) \approx 16{,}000$ time units

If these are dominated by rare regions of different sizes, the ratio
$72{,}000 / 16{,}000 \approx 4.5$ constrains the barrier:
$e^{b \Delta L} \approx 4.5 \implies b \Delta L \approx 1.5$. For
$\Delta L \sim 3$ cells, $b \approx 0.5$ — a **much** weaker effective
barrier than the naive estimate.

**Practical guidance for checking:** If you plot the per-cell relaxation
time $\tau_i$ (from individual Q_i(t) fits) versus the local cluster size
around cell $i$, the $\ln \tau_i$ vs $L_{\text{local}}$ relationship should
be linear with slope $b$ in the Griffiths regime.

---

### 14.5 System Size Requirements

#### Why 288 cells may not be enough

The Griffiths argument requires rare regions of size $L$ that are locally
on the other side of the transition. In a system with $N$ cells arranged
in a 2D domain at cell spacing $a = \sqrt{L^2/N}$:

- The maximum linear dimension of a rare region is $L_{\max} \sim \sqrt{N}$
  (the entire system is one rare region)
- The probability of finding a connected cluster of $\ell$ cells all with
  $v_{A,i} > v_A^*$ is $P(\ell) \sim p_+^\ell$, where
  $p_+ = \Pr(v_{A,i} > v_A^*)$

For our parameters ($\bar{v}_A = 0.008$, $\sigma = 0.008$, $v_A^* \approx 0.008$):
$p_+ \approx 0.5$. So:

| Cluster size $\ell$ | Probability $P(\ell)$ | Expected count in $N$ cells |
|---:|---:|---:|
| 1 | 0.50 | 144 |
| 5 | 0.031 | 9 |
| 10 | $9.8 \times 10^{-4}$ | 0.3 |
| 15 | $3.1 \times 10^{-5}$ | 0.009 |
| 20 | $9.5 \times 10^{-7}$ | 0.0003 |

In 288 cells, we can reliably observe clusters up to $\ell \approx 8$–10.
Clusters of $\ell = 15$ would need $\sim 100$ disorder realizations (runs)
to see even one. Clusters of $\ell = 20$ require $N \gtrsim 10^6$.

**This is why Griffiths effects are hard to observe:** The most dramatic rare
regions are the largest ones, but those are exponentially rare and need
exponentially large systems to find.

#### System size scaling

| $N$ cells | $\sqrt{N}$ (linear size) | $L_{\max}$ (cells) | Largest rare region ($p_+=0.5$) |
|---:|---:|---:|:---|
| 288 | 17 | ~8 | Small clusters only |
| 1,152 | 34 | ~12 | Moderate clusters, barely test scaling |
| 4,608 | 68 | ~16 | Large enough for 2 decades of cluster sizes |
| 18,432 | 136 | ~20 | Proper rare-region statistics |

The 4,608-cell system provides $\sim 16$-cell rare regions with $\sim$1 expected
occurrence per realization. This is the **minimum** for testing the Griffiths
$\tau(L)$ scaling prediction.

#### Finite-size cutoff

In a finite system, Griffiths singularities are **rounded**. The power-law
(or stretched) tail of $Q(t)$ is cut off at:

$$\tau_{\text{cutoff}} \sim e^{b L_{\max}^\psi} \sim e^{b N^{\psi/(2d)}}$$

For $t > \tau_{\text{cutoff}}$, $Q(t)$ decays exponentially. The power-law
(Griffiths) regime only exists for $\tau_\alpha < t < \tau_{\text{cutoff}}$.

With $N = 288$ and $\psi = 1$, $d = 2$: $\tau_{\text{cutoff}} \sim e^{b \cdot 288^{1/4}} \sim e^{4b}$.
If $b \approx 0.5$, then $\tau_{\text{cutoff}} \sim e^2 \approx 7$ — the
Griffiths window barely exists! This means:

> **With 288 cells, we are unlikely to see clean Griffiths power laws in
> $Q(t)$. The system is too small for rare regions to create the necessary
> scale separation.**

With $N = 4608$: $\tau_{\text{cutoff}} \sim e^{b \cdot 4608^{1/4}} \sim e^{4.1b}$ —
still modest, but larger. The improvement $\propto N^{1/4}$ is logarithmically
slow, reflecting the fundamental difficulty of Griffiths physics.

---

### 14.6 How to Verify: Checklist for Griffiths Signatures

For each dataset, perform these checks in order. Each has a **pass/fail
criterion** and an **expected value** you can verify against.

#### Check 1: Is the clean system near the transition?

**Test:** Compute $D(\sigma=0)$ from ensemble MSD slope at long lag times.

- If $D > 0$ clearly: system is in the fluid phase; Griffiths jammed
  islands should be the rare regions
- If $D \approx 0$ with caging plateau: system is jammed; Griffiths
  fluid pockets should be the rare regions

**Our 85% data:** $D = 0.013$ → fluid. We are on the fluid side.
**Our 89% data** (pending): expect $D$ smaller, possibly $\approx 0$.

#### Check 2: Does disorder broaden the $D_i$ distribution?

**Test:** Compute per-cell $D_i$ for σ=0 and σ>0. Plot the distributions.

- **Pass:** $\text{CV}(D_i) = \text{std}(D_i)/\text{mean}(D_i)$ increases with σ
- **Expected:** CV ≈ 0.5 at σ=0 (from intrinsic fluctuations), CV > 1 at σ=0.008

**Our data:** CV = 0.51 at σ=0, CV = 0.87 at σ=0.008 → ✅ PASS.

#### Check 3: Does $D_i$ correlate with $v_{A,i}$?

**Test:** Compute Pearson $r(v_{A,i}, D_i)$ for runs with σ > 0.

- **Pass:** $r > 0.3$ (moderate positive correlation)
- **Expected:** $r$ increases with σ; for σ=0, $r = 0$ by construction

**Our data:** Not yet measurable (Bug 5 — simulation writes mean $v_A$ for all
cells instead of per-cell values). **BLOCKED.**

#### Check 4: $Q(t)$ functional form

**Test:** Fit $Q(t)$ to both:
- Stretched exponential: $Q = \exp(-(t/\tau)^\beta)$, extract $\beta$
- Power law: $Q = A \, t^{-\alpha}$, extract $\alpha$

**Griffiths prediction:**
- $\beta$ decreases with increasing σ (broader relaxation spectrum)
- Power-law fit improves relative to stretched-exp fit at higher σ

**Anti-Griffiths (our observation):**
- $\beta$ increases with σ (narrower spectrum) → disorder fluidizes

**Verification:** Compare AIC or BIC of the two fits. If $\Delta\text{AIC} > 10$
in favor of power law over stretched exp, the Griffiths prediction is supported.

#### Check 5: $\alpha_2(t)$ persistence

**Test:** Measure $\alpha_2(t)$ at $t = T_{\text{obs}}/2$.

- **Griffiths:** $\alpha_2(T_{\text{obs}}/2) > 0.5$ and not decaying → persistent
- **Glass without Griffiths:** $\alpha_2$ peaked at $t \sim \tau_\alpha$, then decays to 0
- **Fluid:** $\alpha_2 \approx 0$ everywhere

**Our data:** $\alpha_2 = 12.70$ at σ=0.008 (persistent) → heterogeneity confirmed,
but this could be from single-cell disorder without collective rare regions.

#### Check 6: $\chi_4(t)$ response to disorder

**Test:** Compare $\chi_4$ peak height between σ=0 and σ>0.

- **Griffiths:** Peak height grows with σ (collective rare-region dynamics)
- **Anti-Griffiths:** Peak height decreases with σ (disorder decorrelates cells)

**Our data:** Peak drops from 38 → 3.3 → **Anti-Griffiths.** This is the
most definitive test.

#### Check 7: Spatial correlation of mobility

**Test:** Compute spatial autocorrelation of per-cell $D_i$:
$$C_D(r) = \langle \delta D_i \, \delta D_j \rangle_{|r_i - r_j| = r}$$
Extract correlation length $\xi_D$ from exponential fit.

- **Griffiths:** $\xi_D$ grows with σ → spatially extended rare regions
- **Single-cell effect:** $\xi_D \approx 1$ cell spacing → no spatial correlation,
  disorder is purely local

**Our data:** $\xi = 3.3$ cell spacings at σ=0.008 → modest spatial correlations
exist but are limited by system size.

#### Summary scorecard

| Check | Griffiths prediction | Our result (288 cells) | Verdict |
|:---|:---|:---|:---|
| 1. Near transition | Required | ✅ D>0 but small | On fluid side |
| 2. CV($D_i$) grows | CV >> 1 at high σ | ✅ 0.87 at σ=0.008 | PASS |
| 3. $r(v_A, D)$ > 0 | Strong positive | ⬜ Blocked (Bug 5) | PENDING |
| 4. β decreases | β → 0 | ✗ β increases | **ANTI-GRIFFITHS** |
| 5. α₂ persistent | α₂ > 0.5 | ✅ 12.70 | PASS (but ambiguous) |
| 6. χ₄ grows | Peak grows with σ | ✗ Peak drops 10× | **ANTI-GRIFFITHS** |
| 7. ξ_D grows | ξ >> 1 | ✅ ξ=3.3 cells | Moderate |

**Overall:** Checks 2, 5, 7 pass (single-cell heterogeneity exists and has
some spatial extent). But checks 4 and 6 definitively fail — the **collective**
signatures are anti-Griffiths. Disorder does not create collectively frozen
rare regions; instead it disrupts cooperative dynamics.

---

### 14.7 What We Actually Found: The Anti-Griffiths Mechanism

#### Why active matter breaks the Griffiths analogy

The Griffiths argument relies on three properties of the Ising model that
**do not hold** in active tissue:

**Property 1: Frozen impurities.**
In the dilute Ising model, vacant sites are immobile. They cannot influence
their neighbors beyond their static coupling. A vacancy is a passive hole.

In our tissue, a cell with high $v_{A,i}$ is not a passive "hole in jamming" —
it is an **active stirrer**. It pushes on its neighbors with force $\sim v_{A,i}$,
physically displacing them. A fast cell in a jammed region doesn't just locally
unjam itself; it **mechanically disrupts the cages of its neighbors.**

**Property 2: Independent rare regions.**
In the Ising model, two well-separated rare regions evolve independently
(their interaction decays exponentially with distance). The total relaxation
is a simple sum over independent contributions.

In active tissue, fast cells generate **propagating stress waves** through the
viscoelastic medium. A single fast cell can initiate a cascade of
rearrangements that extends far beyond its immediate neighbors. Rare regions
are not independent — they communicate through mechanical fields.

**Property 3: Dominance of slow modes.**
In Griffiths physics, the rare jammed regions are the slowest part of
the system, and they dominate the long-time dynamics (because the fast
regions relax quickly and contribute nothing at late times).

In active tissue, the fast cells **actively destroy the slow modes**.
A jammed cluster adjacent to a fast cell is progressively eroded as the
fast cell pushes its boundary inward. The slow modes don't just relax on
their own timescale — they are externally forced to relax on the timescale
set by the neighboring fast cells.

#### The "stirred glass" mechanism

In the clean system (σ=0), all 288 cells push with the same force $v_A = 0.008$.
Cages break only through **collective coordination** — cell A can't escape unless
cell B moves, which requires cell C to yield, etc. This produces:
- Strong cooperativity (high $\chi_4$)
- Broad spectrum of cage lifetimes (low $\beta$)
- Long structural relaxation time (high $\tau_\alpha$)

With disorder (σ > 0), some cells push harder than others. The fast cells
($v_{A,i} > v_A^*$) can break their cages **without help from neighbors**.
These independent escapes don't trigger cooperative cascades — the surrounding
cells simply accommodate the local rearrangement. The result:

- **Reduced cooperativity** (low $\chi_4$): each cell relaxes semi-independently
  based on its own $v_{A,i}$ and local environment
- **Narrower relaxation spectrum** (higher $\beta → 1$): with less cooperative
  multi-cell cage-breaking, the remaining relaxation is simpler
- **Faster overall relaxation** (lower $\tau_\alpha$): the bottleneck of
  collective coordination is removed

This is essentially **local stirring** by the fast cells. Each fast cell
acts as a local fluidizer, shaking its neighbors out of their cages faster
than collective dynamics would achieve. The disorder **decorrelates** the
cooperative caging network rather than creating frozen rare regions.

#### The reconciliation: single-cell vs. collective

Our data shows both Griffiths-like and anti-Griffiths signatures. The
resolution is that they probe different physics:

| Observable | Probes | Response to disorder | Assessment |
|:---|:---|:---|:---|
| α₂ (non-Gaussian) | Single-cell heterogeneity | ↑ Increases | Not Griffiths-specific |
| Persistence | Single-cell classification stability | ↑ Increases | Not Griffiths-specific |
| CV(D_i) | Single-cell diversity | ↑ Increases | Not Griffiths-specific |
| ξ (correlation length) | Spatial extent of heterogeneity | ↑ Modest increase | Weak Griffiths hint |
| β (stretching exponent) | **Collective** relaxation spectrum | ↑ Increases (anti-Griffiths) | Definitive test |
| χ₄ (susceptibility) | **Collective** cooperativity | ↓ Decreases (anti-Griffiths) | Definitive test |

The single-cell observables (α₂, persistence, CV) increase because disorder
genuinely makes individual cells more different from each other. But these
metrics don't require rare regions — a completely uncorrelated system with
each cell having its own $D_i$ drawn from a broad distribution would show
the same behavior.

The collective observables (β, χ₄) are the true Griffiths tests because they
measure **correlations between cells**. The Griffiths prediction is that
disorder creates spatially extended, collectively frozen rare regions. Our
data shows the opposite: disorder breaks up collective behavior.

#### Analogy: "turbulence vs. laminar flow"

Think of the clean jammed tissue as a dam: all cells exert similar pressure,
and the structure holds collectively. Adding quenched motility disorder is
like replacing some bricks in the dam with fire hoses — the high-$v_A$ cells
blast through their local region, disrupting the structural integrity of the
entire dam. The result is not a dam with rare cracks (Griffiths), but a dam
that has been **actively demolished** from within (anti-Griffiths).

In the Ising analogy, the vacancies are like missing bricks — they weaken
the structure passively. In active matter, the disorder is like adding
randomly-placed engines — they actively tear the structure apart.

---

### 14.8 What Would Need to Be True for Griffiths Effects

Given our anti-Griffiths results, under what conditions **would** Griffiths
rare-region physics emerge in tissue? The analysis above suggests:

1. **Frozen spatial disorder (not motility disorder).**
   If the disorder pins cells to specific positions rather than changing their
   propulsion force, the rare regions would be truly static. Example: random
   substrate adhesion (some cells stick to the surface, others don't).

2. **Very weak coupling between fast and slow cells.**
   If fast cells couldn't push their slow neighbors (e.g., cells separated by
   compressible gaps), the rare regions would be approximately independent.
   This might occur at $\phi < 0.7$ where cells are not in contact.

3. **Disorder that does not fluidize (e.g., stiffness disorder).**
   Instead of varying $v_A$ (which directly fights caging), vary cell
   stiffness $\kappa$. Soft cells would be easier to deform but not
   actively motile, so they wouldn't disrupt neighbors.

4. **Much longer observation times.**
   Even in active matter, the anti-Griffiths mechanism (fast cells
   disrupting slow regions) operates on the timescale $\tau_{\text{disrupt}} \sim$
   (rare-region size) / $v_A$. At times $t \gg \tau_{\text{disrupt}}$,
   all rare regions have been disrupted and the system shows fluid behavior.
   But at intermediate times $\tau_\alpha < t < \tau_{\text{disrupt}}$,
   there may be a window where undisrupted rare regions contribute
   Griffiths-like power-law tails. Testing this requires:
   $T_{\text{obs}} \gg \tau_{\text{disrupt}} \gg \tau_\alpha$, which may
   need $T_{\text{obs}} > 10^6$.

5. **Non-motile disorder near a geometric transition.**
   Use the shape index $p_0$ (see references.md) as the control parameter
   and introduce disorder in $p_0$ rather than $v_A$. Since $p_0$ controls
   the energy barrier to T1 transitions (cell rearrangements), disorder in
   $p_0$ would create regions with different barrier heights — closer to
   the Ising analogy. This is not currently implemented in our PFM but
   would be accessible in vertex model simulations.

---

### 14.9 Dimensional Analysis: Natural Units of the Problem

For sanity-checking parameter choices, here are the natural scales:

| Quantity | Expression | Value | Units |
|:---|:---|---:|:---|
| Cell area | $\pi R^2$ | 7,543 | grid² |
| Cell spacing | $\sqrt{L^2/N}$ | 94.3 (288 cells) / 94.2 (1152 cells) | grid |
| Cage size | $a \approx \text{spacing} - 2R$ | $\approx -3.7$ (!) | grid |

Wait — the cage size is negative at these packing fractions! This reveals
that cells at $\phi = 0.89$ are **overlapping** (or rather, their interfaces
overlap, which is how the phase field represents tight packing). The
"cage size" for $Q(t)$ must be set empirically.

Empirically, from Q(t) analysis, we used $a = 0.3 \times \text{spacing} \approx 28$
grid units $\approx 0.57R$. This means a cell must move more than half its radius
to be counted as "escaped."

| Quantity | Expression | Value | Units |
|:---|:---|---:|:---|
| Run length | $v_A \tau$ | 80 | grid |
| Run length / spacing | $v_A \tau / a_{\text{cell}}$ | 0.85 | dimensionless |
| $v_A / v_{\text{char}}$ | $v_A / \sqrt{2D/\tau}$ | — | Péclet-like |
| Observation time / $\tau$ | $T_{\text{obs}} / \tau$ | 25 | reorientation times |
| Observation time / $\tau_\alpha$ | $T_{\text{obs}} / \tau_\alpha$ | ~3.5 (σ=0) | relaxation times |
| Cells across domain | $L / a_{\text{cell}}$ | 17 (288) / 34 (1152) / 68 (4608) | cells |

**Key ratio for Griffiths:** The run length ($v_A \tau = 80$) is comparable
to one cell spacing (94). This means a cell traverses roughly one neighbor
per persistence time — a slow, caged regime. For the fluid regime, we'd
want $v_A \tau \gg a_{\text{cell}}$, which would require either higher $v_A$
or longer $\tau$.

---

### 14.10 Literature Context

#### What has been shown before

| System | Type of disorder | Griffiths observed? | Reference |
|:---|:---|:---|:---|
| Dilute Ising model | Site dilution | ✅ Yes (proved) | Griffiths 1969 |
| Random-field Ising | Field disorder | ✅ Yes (proved) | Imry-Ma 1975 |
| Random transverse-field Ising chain | Coupling disorder | ✅ Yes, infinite-randomness | Fisher 1992, 1995 |
| Active Brownian particles | Quenched speed disorder | Partial (subdiffusion) | Reichhardt & Olson 2016 |
| Vertex model | None (studied clean) | N/A | Bi et al. 2015, 2016 |
| Phase field model (ours) | Quenched $v_A$ disorder | ✗ Anti-Griffiths | This work |

**Gap in the literature:** No previous study has systematically tested
Griffiths rare-region predictions in a tissue-level model with quenched
activity disorder. Our finding of **anti-Griffiths** behavior (disorder
enhancing fluidity rather than creating frozen rare regions) appears to be
a novel result that distinguishes active cellular matter from equilibrium
systems.

#### Closest related work

1. **Reichhardt & Olson Reichhardt (2016)** studied active particles with
   quenched disorder in substrate friction. They found subdiffusive dynamics
   at intermediate times, consistent with caging by pinned obstacles, but
   did not analyze $Q(t)$ functional forms or $\chi_4$.

2. **Berthier (2019)** and **Mandal et al. (2020)** studied active Brownian
   particles without quenched disorder, finding a re-entrant glass transition
   at high activity — consistent with our v_A sweep non-monotonicity.

3. **Vojta (2006)** provides the most comprehensive review of rare-region
   effects across statistical physics, including the classification of
   Griffiths singularities by symmetry and dimensionality. Our active-matter
   system falls outside his classification because it is far from equilibrium.

---

### 14.11 Summary: The Key Equations to Check

For reference, here are all the quantitative predictions you can verify
against simulation data:

#### Free-particle diffusion (no caging)

$$D_{\text{free}} = \frac{v_A^2 \tau}{2d} = \frac{0.008^2 \times 10{,}000}{4} = 0.16 \quad \text{grid}^2/\text{time}$$

If $D_{\text{measured}} \ll D_{\text{free}}$, the system is caged. Our data
shows $D \approx 0.013$ at σ=0, which is $\sim 12 \times$ smaller than the
free-particle value → **strong caging** confirmed.

#### Disorder-dependent $\alpha_2$ (lower bound)

$$\alpha_2 \geq \frac{\langle v_{A,i}^4 \rangle}{2 \langle v_{A,i}^2 \rangle^2} - 1$$

For $v_{A,i} \sim \mathcal{N}(\mu, \sigma^2)$:
- σ=0: $\alpha_2 \geq 0$ (Gaussian → 0)
- σ=0.003: $\alpha_2 \geq \frac{(\mu^4 + 6\mu^2 \sigma^2 + 3\sigma^4)}{2(\mu^2 + \sigma^2)^2} - 1 \approx 0.02$
- σ=0.008: $\alpha_2 \geq 0.25$ (see derivation in 14.3.2)

Measured values far exceed these bounds → caging amplifies disorder.

#### Rare-region probability

$$P(\ell) = p_+^\ell, \quad p_+ = \Pr(v_{A,i} > v_A^*)$$

For $p_+ = 0.5$: $P(10) = 10^{-3}$, need $N \gtrsim 1000$ cells for one
such cluster.

#### Griffiths relaxation (if observed)

$$Q(t) \sim t^{-\alpha}, \quad \alpha = \frac{|\ln p_+|}{b}$$

where $b$ is the barrier exponent. Measuring $\alpha$ from a log-log fit
of $Q(t)$ and independently measuring $p_+$ from the $v_{A,i}$ distribution
would provide a consistency check.

#### Stretched exponential (glass baseline)

$$Q(t) = \exp\!\left(-(t/\tau_\alpha)^\beta\right)$$

At the clean glass transition, $\beta \approx 0.5$–$0.7$ is typical. Our
σ=0 value of $\beta = 0.57$ is consistent with a standard glass.

---

*Section added: February 14, 2026. Based on data from 288-cell Griffiths study
at φ=0.85, with preliminary analysis from 1152-cell local runs at φ=0.89.*
