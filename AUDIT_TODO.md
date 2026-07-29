# Audit follow-up list

From the 2026-07-28/29 audit of the CUDA cell simulation, the CPU references,
and the `cell_analyze` pipeline. Ordered by priority. Work one at a time.

Status key: `[ ]` open · `[~]` in progress · `[x]` done

---

## Done

- [x] **Repulsion coefficient was half of Palmieri Eq. (S15).**
  `interaction_coeff` returned `30κ/λ²`, should be `60κ/λ²` (the factor 2 is
  the ordered pair sum in Eq. 10, not the mobility). Fixed in CUDA, Python and
  Rust references. Commit `1637b7c`. Verified black-box on GH200: measured
  coefficient 6.12245 → 12.24490 over ~2×10⁴ overlap pixels.
- [x] Non-circular regression test (`tests/python/test_variational.py`):
  numerical δF/δφ check in a two-cell overlap, plus the convention-free
  invariant `interaction_coeff / motility_coeff == ξ`.
- [x] Manuscript `F_vol` (missing `1/A₀`) and `F_rep` (missing `30/λ²`,
  ordered sum) corrected; explicit PDE and `v_I` added; `Δt` 0.02 → 0.01;
  pre-existing `\TODO`/`\textbf` compile blocker fixed.
- [x] Full derivation + measurement written up:
  `cpp/simulation/tests/validation_notes/palmieri_coefficient_audit.pdf`.

---

## P0 — affects data we have or are about to publish

- [ ] **1. Trajectory centroid is off by the rebind shift.**
  `sim.cu:1666-1674` computes `Cx/Cy` pre-rebind; `launch_rebind` then
  advances `origin`; the run loop (`sim.cu:1911`) reads both afterwards, so
  the recorded centroid is `true + s`, `s ∈ {-1,0,+1}` px. Fires only when a
  rebind actually shifts (~0.1% of rebinds), but every trajectory sample is a
  rebind step. MSD impact negligible; a 1 px jump between samples 180 TU apart
  is a spurious speed ~0.006 vs `v_A = 0.01`, so it matters for burst
  detection. Fix: read `Cx/Cy` before the rebind, or add `s` back out.

- [ ] **2. Percolation adjacency is not Voronoi and connects through
  immobile cells.** `percolation_cluster.rs:114-145`: candidates are sorted by
  polar angle and then *all* returned — no Delaunay filtering, the sort is
  dead code. It is a distance-cutoff graph at 4R (~12-18 neighbours when
  confluent) while reporting `adjacency_used = "voronoi"`, and
  `adj_cutoff_factor` is ignored on that branch. Compounding it, the graph is
  built from mobile-cell coordinates only (line 240 / 221-224), so mobile
  cells connect *through* the immobile cells physically separating them. Both
  errors over-connect in the same direction: S_max and P_∞ inflated, apparent
  threshold pushed to an artificially low mobile fraction. **Blocks any
  percolation rerun** — fix before spending GPU time.

---

## P1 — silent-failure hardening (chained SLURM runs)

- [ ] **3. All output-file failures are silent and the run exits 0.**
  `MKDIR` return ignored (`main.cu:435`); `save_checkpoint` is `void` and
  returns on `fopen` failure (`sim.cu:2354`); short writes from a full quota
  delete the `.tmp` and return (`sim.cu:2408/2416/2444/2451/2478`);
  `trajectory.txt` and VTK are gated on a possibly-null `FILE*`
  (`sim.cu:1881`, `2285`). Leg N writes nothing, leg N+1 resumes from an older
  checkpoint, nothing reports it. Worst failure mode on purge-prone scratch.

- [ ] **4. `--v-A-sigma` on resume destroys disorder instead of being
  ignored.** `sim.cu:1354` clears the `VA_A` sidecar, but
  `apply_v_A_disorder()` is only called from `init()`. Re-passing the flag on
  each chained leg — the natural thing — converts a disordered population to
  monodisperse at the first resume. **Threatens the Griffiths study**; check
  whether any existing chain re-passed it.

- [ ] **5. `--trajectory-interval` is computed from the default `t_end`/`dt`
  on resume.** `main.cu:426-433` uses `p.t_end/p.dt` (CLI defaults 100/0.01)
  unless `-t` and `--dt` are also passed; `init_from_checkpoint` then
  recomputes cadence from the checkpoint's values. Silent ~50× sampling error.
  Note the production I/O guidance recommends exactly this flag.

- [ ] **6. `checkpoint_interval` is not persisted.** Written as
  `prefix.reserved = 0` and discarded on read (`sim.cu:1151`, `1174`). Every
  leg must re-pass it or rolling checkpoints silently turn off.
  `--save-interval` and `--trajectory-samples` *are* persisted.

- [ ] **7. `--v-A` on resume: sidecar beats CLI, contrary to the docs.**
  `sim.cu:1359-1364` clears the sidecar only when it sums to ~0. Changing
  `v_A` mid-chain on a motile checkpoint is silently ignored, while the new
  value is written into the checkpoint header and `trajectory.txt`. Also
  `params.v_A` gates the polarity update (`kernels.cu:836`), so a leg run with
  `--v-A 0` freezes polarity for every later leg. Current FSS/percolation
  protocols are safe (eq is exactly `v_A=0`), but fix the code/doc mismatch.

---

## P2 — performance

- [ ] **8. Grid is sized from `TILE_AREA`, not the active rect.**
  `REDUCE_CHUNKS_PER_CELL = TILE_AREA/CHUNK_PIXELS = 25`, but the rect at
  R=49 needs 7 — ~72% of launched blocks do no work. Nothing is saturated
  (SM 46-54%, DRAM 25-36%). Measured headroom via a `TILE_T=192` A/B:
  −20% at 72 cells, −13% at 288, −11% at 1152. Fix: size `grid.x` from the
  current max rect at each rebind — **bit-exact**, since dropped chunks
  contribute `+0.0` to the fixed-order reduction. Needs the CUDA-graph cache
  keyed on chunk count (grid dims are baked into a captured graph). Keep
  `TILE_T=320`; the headroom is a real requirement for deformed cells.
  Secondary: move the `p_start < rect_total` guard in `k_reduce_mb_fast`
  (`kernels.cu:248`) outside, so empty blocks skip `block_sum` + 3 stores.

---

## P3 — latent / robustness

- [ ] 9. `--gamma "a;b"` in one flag sets global `gamma` to 0
  (`main.cu:300-307`; `bare_gamma` never assigned for multi-segment specs).
  `gamma_ref = 0` makes soft cells take the stiff-cell rect margin → possible
  silent mass loss. Current scripts are safe (their specs contain `:`).
- [ ] 10. Legacy `sp_sz == 72` offset table contradicts `conftest.py`
  (`sim.cu:950-965`): the 4-byte shift is applied only for `sp_sz == 92`. If
  conftest is right, v3/v4 resumes read `tau` from the `xi` slot. Needs a real
  archived legacy file to settle. Only legacy test asserts `num_cells == 4`.
- [ ] 11. Float sidecar length never checked against `num_cells`
  (`sim.cu:1086-1094`): a short sidecar silently restores the first k cells and
  falls back to uniform params for the rest. `RNGS` does check; `VA_A`/`GAMA`/
  `RADI`/`POLR` do not.
- [ ] 12. Changing `--dt` on resume ends the run at the wrong physical time
  (`sim.cu:1864`): `target_step = t_end/dt` assumes dt constant since step 0;
  nothing checks `cur_time ≈ step_count·dt`.
- [ ] 13. `SimParams` is `fwrite`n verbatim with no `static_assert` on size or
  field offsets — a reorder or repurposed field is a silent misread, which is
  exactly what happened with `subdomain_padding`.
- [ ] 14. Legacy resumes silently disable trajectory + archival checkpoints
  (`sim.cu:1165-1174`, `968`); `abp` is never decoded on the `sp_sz == 92`
  branch, so an ABP run reverts to RTP unless `--abp` is re-passed.
- [ ] 15. Stale `Cx/Cy` in the final/SIGTERM checkpoint (up to 7 steps old).
  Metadata only; physics on resume is unaffected.
- [ ] 16. `S` scatter uses `atomicAdd` → runs are not bitwise reproducible.
  Acknowledged in `kernels.cu:207-208`. Everything downstream is deterministic.

---

## Wishlist — multi-GPU (not currently used)

- [ ] 17. Migration leaves a stale scratch half (`sim.cu:2729-2792`):
  `k_compact_phi` copies full tiles into the destination, but nothing scrubs
  the old half, which becomes `phi_in` two steps later with a foreign ring
  outside the new rect. The comment at `kernels.cu:1344` describes a memset
  that does not exist.
- [ ] 18. Halo bands alias when `slab_h < 2·HALO_H`; the guard
  (`sim.cu:294`) only checks `slab_h + 2·HALO_H ≤ Ny`, which is strictly
  weaker. Correct assertion: `slab_h >= 2*HALO_H`.
- [ ] 19. `k_classify_migrants` stay-fallback writes out of bounds
  (`kernels.cu:1238-1248`): `slab_local_y` is not clamped to `ext_height`, so
  `k_scatter_S` `atomicAdd`s past the end of `S`. `slab_in_window` exists but
  has **zero call sites**.
- [ ] 20. NCCL failures `std::exit(1)` from a worker thread while others are
  parked at a barrier, bypassing the final checkpoint; no
  `ncclCommGetAsyncError` polling, so a failed collective burns the full wall
  clock. `cudaSetDevice` at `sim.cu:3024` is unchecked.
- [ ] 21. Migration ordering is `atomicAdd`-nondeterministic, so multi-GPU
  runs are not reproducible run-to-run.
- [ ] 22. Single-GPU resume discards the checkpoint's global cell ids
  (`sim.cu:2204`, `2434` guard on `gpus > 1`), renumbering cells across the
  G>1 → merge → G=1 boundary.
- [ ] 23. `Nx == Ny` is assumed by every kernel but only enforced in
  `main.cu`; a checkpoint with `Nx != Ny` would wrap wrongly.

---

## Campaign / manuscript

- [~] **FSS N=100 rerun with fixed physics** — nibi, seeds a-d, ctrl+soft.
  eq array `18694296`, prod `18694307` (chained `afterok`).
  Output `/scratch/ssilber/fss_fixed/`. Binary SHA in
  `~/canonical_binary_fixed.env`.
- [~] **Parity fixture regeneration** — nibi CPU array `18699075`. The stored
  `ref_trajectory.txt` / `ref_final_phi.npz` encode the old physics, so
  `test_cutover_parity` fails until these land.
- [ ] **Decide seed count.** Baseline N=100 ratio is `1.107 ± 0.077` (n=10).
  The scatter is *within-run* sampling noise at lag 8τ (~25 independent
  windows per 200τ run), confirmed by sd growing monotonically with lag
  (0.144 → 0.244 from 0.5τ to 8τ). So re-analysis cannot tighten it; **more
  seeds is the right lever**. n=4 gives sem ±0.17 and cannot resolve anything
  smaller than a ±0.5 shift. Recommend extending to all 10 seeds (purely
  additive to what is queued).
- [ ] **Percolation rerun** — staged on Roihu (`~/roihu_perc_*.sbatch`, seeds
  verified against `eq_v8`), not submitted. Blocked on item 2. Consider nibi
  (free RRG) instead of Roihu BU.
- [ ] **Manuscript: production duration.** Text says `t = 100,000 (10τ)`;
  every canonical sbatch runs to `t = 2,080,000` (**200τ**) and the run
  metadata confirms it. Factor of 20.
- [ ] **Manuscript: bounding-box claim.** §Implementation says side
  `≈ 2.5 × 2R`; the code computes `hw = 2σ + R/2` → 160 px at R=49, i.e.
  `≈ 1.6 × 2R` (full tile is 320, `≈ 3.3 × 2R`). Neither reading gives 2.5.
- [ ] **Re-equilibration is mandatory for any corrected rerun.** All existing
  eq checkpoints (nibi `fss_canon_nibi/eq`, Mahti `eq_v8`) were produced by
  the κ_eff=5 binary and are invalid initial conditions.

---

## Retracted during the audit

- ~~"The analysis discards 96% of each trajectory."~~ Wrong. `msd_palmieri`
  uses `n_origins = n - lag` origins across the full 200τ; 8τ is the maximum
  *lag*, not the analysis window. The design is correct as intended.
- ~~"Re-analysis is cheaper than more seeds."~~ Backwards — see the seed-count
  item above.
- ~~"The bbox margin truncates the field at small R."~~ Tested at
  R ∈ {49,36,20,10} with `--subdomain-padding` raised 3× and 6×: volumes
  identical to five significant figures. The `TILE_BBOX_MIN = 32` floor
  already exceeds the interface decay length.
