# Validation — Palmieri Extension Study

Evidence that the simulation reproduces Palmieri et al. (2015) and the analysis pipeline is correct.

## Files

### Phase Field Snapshot
- **phase_field_snapshot.png** — VTK phase field Σφ_i from v2_90s checkpoint at t=1,287,000 TU. 100 cells at ρ=0.90, domain 916×916. Cell coverage = 90.01% (expected 89.90%). Max φ = 1.04 (no strong overlaps).

### Cell 0 Diagnostic Panels (5-row format)
Each of these shows: (1) L_n histogram at first snapshot, (2) L_n time series, (3) displacement speed, (4) wrapped trajectory, (5) MSD/4t.

- **ln_velocity_comparison.png** — Local 100τ soft vs FSS v3 200τ soft vs FSS v3 ctrl. Establishes that L_n dynamics are identical between local and cluster runs (mean L_n ≈ 1.155 for soft, 1.06 for ctrl).
- **ln_velocity_v2.png** — v2 200τ soft vs ctrl (clean data, corrected subsample). Shows D_eff(8τ) extraction.
- **ln_velocity_fss_rep.png** — FSS replicate seed=5 soft vs ctrl. Independent verification from the 10-replicate campaign.

### MSD Method Investigation
- **msd_method_comparison.png** — Comparison of Palmieri-style (single-origin |r(t)-r(0)|²) vs time-averaged (sliding window) MSD. Population MSD agrees between methods; single-cell MSD is noisy in both. Conclusion: D_eff must be extracted at 8τ plateau, not averaged over a wide lag range.
- **rust_msd_comparison.png** — Rust cell_analyze output: population MSD/4t for all 3 systems + per-cell D_eff distribution. Shows cell 0 at 97th percentile in local run vs 16th in FSS.

### Trajectory Comparison (Suspicious Cases)
- **fss_matched_comparison.png** — Matched-domain trajectory comparison for N=200 ρ=0.90 and N=800 ρ=0.85. Old v3 single-replicate data. Identified issues: chain overlap corruption in 800c_85s, non-plateauing MSD/4t for individual cells.

### Ensemble Results
- **fss_reps_combined.json** — Rust study pipeline output for N=100 (10 seeds) and N=400 (10 seeds, 67τ preliminary). Key result: D_eff ratio = 1.34 ± 0.13 (N=100), 1.25 ± 0.17 (N=400).

## Key Validation Findings

1. **γ correctly applied**: Soft cell L_n = 1.155 vs normal = 1.06 (gap = 0.09), consistent across all runs and seeds.
2. **Displacement speed enhanced**: Soft cell 7.6% faster (N=100, p < 0.001 over 10 seeds).
3. **D_eff ratio**: 1.34 ± 0.13 at N=100 (Palmieri: ~1.5 at N=72). Enhancement is real and statistically significant.
4. **MSD method**: D_eff must be evaluated at lag = 8τ (the Palmieri plateau), not averaged over 5–50τ.
5. **Chain overlap**: Handled correctly by Rust io.rs; Python scripts must also handle it.
6. **Subsampling**: awk-based subsampling can interleave chain segments — use cell-0-aware methods.
