# Manuscript Checklist — Palmieri Extension

## Status Legend
- [ ] Not started
- [~] In progress / partial data
- [x] Complete

---

## 1. Validation (Palmieri Reproduction)

- [x] **1.1** Phase field snapshot showing correct cell morphology at ρ=0.90
- [x] **1.2** L_n time series: soft cell L_n ≈ 1.15, normal cell ≈ 1.06
- [x] **1.3** Displacement velocity: soft cell faster than ctrl (7.6% ± 1.6%)
- [x] **1.4** Velocity distribution G(v_i): non-Gaussian tails with σ_G ≈ 0.003 (Palmieri: 0.0029)
- [x] **1.5** MSD/4t plateau at 8τ confirming diffusive regime
- [x] **1.6** D_eff extraction method validated: lag = 8τ (Palmieri convention)
- [ ] **1.7** Fig 2-style panel: L_n deformation snapshot (soft cell highlighted in monolayer)
- [ ] **1.8** Fig 4-style panel: G(v_i) rescaled velocity distribution with Eq. 5 fit

## 2. Finite-Size Scaling (Phase 1)

- [x] **2.1** N=100, ρ=0.90: D_eff ratio = 1.34 ± 0.13 (10 replicates, 200τ)
- [~] **2.2** N=400, ρ=0.90: D_eff ratio = 1.25 ± 0.17 (10 replicates, 67τ preliminary)
- [~] **2.3** N=200, ρ=0.90: Submitted, ~24h to complete
- [ ] **2.4** N=800, ρ=0.90: Pending on cluster
- [ ] **2.5** N=1600, ρ=0.90: Pending on cluster (chain jobs, ~5 days)
- [ ] **2.6** D_eff ratio vs N plot with error bars (need ≥3 data points)
- [ ] **2.7** L_n gap vs N plot with error bars
- [ ] **2.8** Mean speed ratio vs N plot
- [ ] **2.9** Extrapolation to N→∞ (1/√N fit)
- [ ] **2.10** ρ=0.85 replicates (not yet submitted)

## 3. Percolation (Phase 3)

### 3A — Pairwise Cooperativity (N=2000, ρ=0.90)
- [~] **3A.1** d=2R: 3 seeds running (~4h elapsed of 24h)
- [~] **3A.2** d=4R: 3 seeds running
- [~] **3A.3** d=8R: 3 seeds running
- [ ] **3A.4** d=6R: pending (chain dependency)
- [ ] **3A.5** d=12R: pending
- [~] **3A.6** d=20R: 3 seeds running
- [ ] **3A.7** Analysis: D_eff enhancement vs separation distance
- [ ] **3A.8** Plot: cooperative effect decay length

### 3B — Fraction Scan (N=4000, ρ=0.90)
- [~] **3B.1** f_c=0.05 (5%): 2 runs actively running (2d 6h of 3d), 3 more chained
- [ ] **3B.2** f_c=0.10 (10%): pending
- [ ] **3B.3** f_c=0.15 (15%): pending
- [ ] **3B.4** f_c=0.20 (20%): pending
- [ ] **3B.5** f_c=0.30 (30%): pending
- [ ] **3B.6** f_c=0.50 (50%): pending
- [ ] **3B.7** All-normal control: 5 seeds pending
- [ ] **3B.8** Analysis: population D_eff vs f_c
- [ ] **3B.9** Analysis: mobile fraction and percolation threshold f_c*
- [ ] **3B.10** Plot: D_eff vs f_c with error bars

### 3C — Clustered vs Dispersed (N=4000, ρ=0.90)
- [ ] **3C.1** f_c=0.10 clustered: 5 seeds pending
- [ ] **3C.2** f_c=0.20 clustered: 5 seeds pending
- [ ] **3C.3** Analysis: compare D_eff(clustered) vs D_eff(dispersed) at same f_c
- [ ] **3C.4** Plot: interior vs edge cancer cell motility

## 4. Figures for Manuscript

- [ ] **4.1** Fig 1: Model schematic + phase field snapshot
- [ ] **4.2** Fig 2: Validation panel (L_n, velocity distribution, MSD)
- [ ] **4.3** Fig 3: FSS — D_eff ratio vs N with error bars
- [ ] **4.4** Fig 4: FSS — L_n gap and speed ratio vs N
- [ ] **4.5** Fig 5: Pairwise cooperativity — enhancement vs separation
- [ ] **4.6** Fig 6: Percolation — D_eff vs cancer fraction
- [ ] **4.7** Fig 7: Clustered vs dispersed comparison

## 5. Writing

- [ ] **5.1** Introduction: motivation, Palmieri context, what's new
- [ ] **5.2** Model: phase field equations, parameters table
- [ ] **5.3** Methods: equilibration protocol, D_eff extraction at 8τ, ensemble averaging
- [ ] **5.4** Results §1: Validation
- [ ] **5.5** Results §2: Finite-size scaling
- [ ] **5.6** Results §3: Percolation
- [ ] **5.7** Discussion: thermodynamic limit, cooperativity range, percolation universality
- [ ] **5.8** Conclusion

## 6. Infrastructure

- [x] **6.1** Rust study pipeline with TOML configs
- [x] **6.2** D_eff extraction at 8τ (updated in Rust)
- [x] **6.3** Paired soft/ctrl comparison with proper error propagation
- [x] **6.4** SVG plot generation from Rust
- [ ] **6.5** Cross-compile Rust binary for cluster (run analysis remotely)
- [x] **6.6** Study TOML configs: fss.toml, percolation.toml
- [ ] **6.7** Study TOML config: pairwise.toml (Phase 3A)
