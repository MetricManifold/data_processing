# cpu_ref_2tau_soft parity fixtures

Reference data for the sim_v3 ↔ cpu_ref parity test with one soft cell
(γ=0.35, Palmieri-extension parameters) and v_A = 0.014.
Used by `test_cutover_parity.py::TestCutoverParitySoft::test_2tau_soft_scripted_events`.

## Files

| file | size | content |
|------|------|---------|
| `ic_checkpoint.bin` | 6.4 MB | initial-condition v6 checkpoint, 16 cells, R=49, L=376, dt=0.01, t=0.01, seed=12345; cell 0 has γ=0.35, all others γ=1.0; carries POLR + GAMA sidecars |
| `events.txt` | 1.5 KB | deterministic tumble events (`# t cid old_theta new_theta`) replayed via `--scripted-events` |
| `ref_trajectory.txt` | 296 KB | f64 Rust `cpu_ref` ground-truth trajectory, 400 frames over 2τ |
| `ref_final_phi.npz` | 2.6 MB | final-frame φ field for whole-array comparison (extracted from the much larger per-frame `traj.npz`) |

## Generation protocol

1. **Source:** `rust/cpu_ref` binary built on Compute Canada Nibi at commit
   `d9447a7` ("per-cell gamma in cpu_ref") — adds per-cell γ via the GAMA
   sidecar.
2. **IC:** generated with
   ```
   cell_sim -n 16 -N 376 -r 49 -t 0.01 \
            --gamma 0.35:cell0 \
            --save-final-checkpoint --seed 12345
   ```
3. **Reference run** (slurm `run_2tau_soft.sbatch`,
   `/scratch/ssilber/cpu_ref_validate/run_2tau_soft/` on Nibi, completed
   2026-05-01 14:48):
   ```
   cpu_ref --ic ic_checkpoint.bin \
           --v-A 0.014 --tau 10000 --t-end 20000 --dt 0.01 \
           --save-every 5000 --polarity-seed 12345 \
           --out traj.npz --trajectory trajectory.txt \
           --events events.txt
   ```
4. **Final-frame extraction** (compresses 3.6 GB → 2.6 MB):
   ```python
   d = np.load('traj.npz', mmap_mode='r')
   np.savez_compressed('ref_final_phi.npz',
       t=d['t'][-1], phi=d['phi'][-1],
       vx=d['vx'][-1], vy=d['vy'][-1], vol=d['vol'][-1],
       px=d['px'][-1], py=d['py'][-1])
   ```

## Why this is the bit-truth reference

- Same f64 single-threaded Rust path as `cpu_ref_2tau/` — the only
  difference is that cell 0 carries a per-cell γ=0.35 read from the
  checkpoint's GAMA sidecar. Drift envelopes and tolerances therefore
  match the hard fixture: `rms|Δr|<0.5`, `max|Δr|p95<0.5`,
  `final max|Δr|<1.0`, `phi_rms<5e-2`, `phi_max<0.7`.

## When to regenerate

- If the soft-cell coupling in `rust/cpu_ref` changes (per-cell γ read
  path, scaling of γ in the bulk/grad terms).
- If the IC layout or the GAMA sidecar format changes.
- Re-tune the absolute thresholds in `test_cutover_parity.py` only when
  the f64 Rust path or the GPU integration order changes.

## Regenerated 2026-07-30 (repulsion-coefficient fix)

Superseded by the fix in commit `1637b7c`: the repulsion coefficient in
`dphi/dt` was `30k/l^2` and is now `60k/l^2` (Palmieri Eq. S15). The previous
reference data encoded the old physics, so `test_cutover_parity` would have
failed against it for the wrong reason.

Regenerated with the corrected `rust/cpu_ref` built from commit `1ac93f2`
(cpp/simulation and rust/cpu_ref identical at `1637b7c`), single-threaded,
nibi slurm job `18732646`, ~14h20m wall. Same IC, same `events.txt`, same
command as the original protocol above.

NOTE: the cpu_ref npz writer now emits every saved snapshot
(`phi` shape `(nsnap,16,H,W)`) plus 7 extra metadata keys, where the original
fixture holds only the final frame (`phi` shape `(16,H,W)`) with 7 keys.
The raw output was therefore trimmed to the final frame and recompressed
(3.6 GB -> 2.3 MB) via `~/trim_fixture.py` on nibi. Verified against the old
fixture: identical key set, shapes, dtypes and final `t` (19950.01).
