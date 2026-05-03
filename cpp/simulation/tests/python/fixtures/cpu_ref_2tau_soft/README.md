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
