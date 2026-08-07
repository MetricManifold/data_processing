# Is the monolayer caged at rho = 0.90? — measured, 2026-08-06

Short answer: **no.** There is no caging plateau, no alpha_2 peak, and no
cage/escape bimodality in the van Hove function, measured across four decades
of lag. This matters because the percolation-of-mobile-regions analysis
presupposes a caged background with mobile pockets.

## Provenance

Trajectories generated with `cpp/gh200_sim` (`cell_gh200`), resumed from the
corrected-physics equilibrated states at
`/scratch/project_2017848/ssilber/fss_fixed/eq/100c_{ctrl,soft}/seed_a`.
Parameters verified from the trajectory header, not assumed:

    v_A=0.010000  N=100  Lx=916  Ly=916  dim=2  dt=0.010000  tau=10000.0000

Window t = 80,000 -> 279,990 = **20 tau**, sampled every 10 TU (1e-3 tau),
2,000,000 rows per branch. Roihu `gpumedium`, ~0.5 GPU-h total.
gh200_sim carries the CORRECT repulsion coefficient (`kNumerInteraction = 60`,
with a `static_assert` that it is exactly twice the bulk numerator, and
`motility_coeff` defined as `interaction/xi` so the invariant holds
structurally). Its `--self-test` gates passed.

## Method and self-checks

`caging_gr_selfcheck.py` runs two analytic checks before measuring anything;
both must pass or the numbers are meaningless:

| check | expected | measured |
|---|---|---|
| uniform translation of all cells | cage-relative displacement exactly 0 | 6.2e-15 |
| uncorrelated displacements | CR-MSD/MSD = 1 + 1/z | 1.0933 vs 1.0921 (0.11%) |

The chain is then: g(r) -> first minimum = neighbour cutoff -> cage-relative
displacements -> alpha_2(lag) -> would-be T_obs -> van Hove at that lag ->
would-be threshold. `caging_alpha2.py` does the 20-tau sweep.

## Results

**1. g(r) — this part is solid and reusable.**
First peak **95.0 px** (0.97 cell diameters, i.e. essentially at contact for
2R = 98), first minimum **132.8 px** (1.36 diameters). This is the measured
neighbour cutoff for the cage; it does not depend on anything below.

**2. alpha_2 declines monotonically; there is no interior peak.**

| lag/tau | 0.003 | 0.02 | 0.22 | 1.08 | 2.81 | ~8 | 10 |
|---|---|---|---|---|---|---|---|
| ctrl | 0.383 | 0.332 | 0.346 | 0.153 | 0.066 | ~0.08 | 0.113 |
| soft | 0.376 | 0.290 | 0.283 | 0.120 | 0.036 | ~-0.06 | -0.065 |

The only structure is a broad shoulder (alpha_2 ~ 0.33 ctrl / 0.28 soft) over
0.01-0.25 tau, far too flat to locate a crossover. **tau\* does not exist**, so
the Keys et al. prescription (T_obs at the alpha_2 maximum) has no value to
take here.

**3. The van Hove self-correlation is unimodal at every lag.** The nominal
"first minimum" comes out at 0.19 px = 0.002 diameters, i.e. noise. So the
Gao/Dyre prescription (threshold at the first van Hove minimum) also has
nothing to key off.

**4. Two useful by-products.**
 - At the 8 tau lag where D_eff is read, alpha_2 ~ 0.08 (ctrl) and ~ -0.06
   (soft): essentially Gaussian. Independent confirmation that 8 tau is in the
   diffusive regime, consistent with dlogMSD/dlogt ~ 1 measured separately.
   **The D_eff extraction is sound.**
 - CR/absolute MSD runs 0.74-0.90 with a minimum near 1-2 tau. Cage-relative is
   consistently smaller than absolute, so neighbourhood motion is genuinely
   correlated and the subtraction removes real collective displacement. The
   metric does something even in the absence of caging.

## Consequences

Both literature prescriptions written into the methodology draft are
inapplicable at rho = 0.90, and more importantly "percolation of mobile
regions" presupposes a caged background this model does not have at this
density. `f_c*` may not exist here.

Options, in order of preference:
 1. Test higher rho (0.95-1.00) and find where caging appears; the supervisor's
    plan Phase 2 asks for rho = 0.70-1.00 precisely to locate rho_J.
 2. Calibrate the mobility threshold on the f_c = 0 control (e.g. the 95th
    percentile of its CR-displacement distribution), held FIXED across the
    f_c sweep so the mobile fraction still floats.
 3. Report "no caging plateau at confluence" as a result. It bears directly on
    the plan's aim (iii), the vertex-model connection.

## Loose end

The ctrl run printed `class_exhausted 210526` while soft reported "no fatal
flags". The meaning of that counter in gh200_sim has not been chased. Check it
before these trajectories are used for anything load-bearing.

## Consistency note

An earlier attempt at this measurement used a 2 tau window and reached only
0.67 tau in lag, which could not have resolved a crossover near the 8 tau D_eff
lag. Its conclusion happened to agree but was not supported by its data. The
numbers above supersede it.
