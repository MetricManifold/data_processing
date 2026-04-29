# cpu_ref_2tau parity fixtures

Reference data for the GPU sim_v3 ↔ f64 Rust `cpu_ref` parity regression test
(see `test_cutover_parity.py`).

## Files

| file | size | content |
|------|------|---------|
| `ic_checkpoint.bin` | 2.3 MB | initial-condition v7 checkpoint, 16 cells, R=49, L=376, dt=0.01, t=0.01, polarity_seed=12345; carries POLR sidecar |
| `events.txt` | 1.5 KB | 31 deterministic tumble events (`# t cid old_theta new_theta`) replayed via `--scripted-events` |
| `ref_trajectory.txt` | 296 KB | f64 Rust `cpu_ref` ground-truth trajectory, 400 frames over 2τ |

## Generation protocol

1. **Source:** `rust/cpu_ref` binary built on Compute Canada Nibi, single-threaded
   (`--threads 1`-equivalent path; with rayon, all reductions are deterministic
   when N=1 or with the parallel reduction in this code).
2. **Fix:** Job ran AFTER commit `abfd183` ("rust cpu_ref: read POLR sidecar")
   so the IC's POLR sidecar is honored — initial polarities match the GPU
   exactly. (Earlier Rust runs silently regenerated θ from a Xoshiro PRNG;
   reference data from before that commit is invalid.)
3. **Command:**
   ```
   cpu_ref --ic ic_checkpoint.bin \
           --v-A 0.01 --tau 10000 --t-end 20000 --dt 0.01 \
           --save-every 5000 --polarity-seed 12345 \
           --events events.txt --trajectory ref_trajectory.txt
   ```
4. **Origin:** `/scratch/ssilber/cpu_ref_validate/run_2tau_v2/` on Nibi
   (slurm job `12968279`, completed 2026-04-29 14:29 EDT, 3h13m walltime).

## Why this is the bit-truth reference

- `rust/cpu_ref` runs in `f64` end-to-end; GPU sim_v3 uses `f32` for hot
  arrays. The drift between the two is dominated by f32↔f64 roundoff and
  algorithmic-order differences (atomicAdd ordering, fused-multiply-add
  scheduling). It is **not** dominated by physics differences.
- Empirical drift over 2τ with this fixture: rms |Δr| ≈ 0.10 sim units,
  max |Δr| ≈ 0.27 sim units (about 0.6% of cell radius R=49). Spikes to
  ~1 sim unit at tumble-event boundaries are PBC-seam wrap differences,
  not physical divergence.

## When to regenerate

If sim_v3 source is changed in any way that affects the integration
order of operations (kernel launches, reduction strategies, atomic
patterns, FP precision policy), the absolute thresholds in
`test_cutover_parity.py` may need re-tuning. The reference itself does
not need regenerating unless the f64 Rust path or the IC layout changes.
