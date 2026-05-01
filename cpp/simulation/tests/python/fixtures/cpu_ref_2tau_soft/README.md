# cpu_ref_2tau_soft parity fixtures

Reference data for a sim_v3 ↔ cpu_ref parity test with one soft cell
(γ=0.35, Palmieri-extension parameters) and v_A = 0.014.

## Files

| file | content |
|------|---------|
| `ic_checkpoint.bin` | initial-condition checkpoint, 16 cells, R=49, L=376, dt=0.01, t=0.01, seed=12345; cell 0 has γ=0.35, all others γ=1.0; carries POLR + GAMA sidecars |
| `events.txt` | deterministic tumble events (`# t cid old_theta new_theta`) replayed via `--scripted-events` |
| `ref_trajectory.txt` | f64 Rust `cpu_ref` ground-truth trajectory, 400 frames over 2τ |
| `ref_final_phi.npz` | final-frame φ field for whole-array comparison |

## Generation protocol

1. **Source:** `rust/cpu_ref` binary built on Compute Canada Nibi at commit
   `d9447a7` ("per-cell gamma in cpu_ref") which adds per-cell γ via the GAMA
   sidecar.
2. **IC:** generated with
   ```
   cell_sim -n 16 -N 376 -r 49 -t 0.01 \
            --gamma 0.35:cell0 \
            --save-final-checkpoint --seed 12345
   ```
3. **Reference run:**
   ```
   cpu_ref --ic ic_checkpoint.bin \
           --v-a 0.014 --tau 10000 --t-end 20000 --dt 0.01 \
           --save-every 5000 --polarity-seed 12345 \
           --out traj.npz --trajectory ref_trajectory.txt \
           --events events.txt
   ```
