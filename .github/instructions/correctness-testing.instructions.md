---
applyTo: "cpp/simulation/tests/**,rust/cpu_ref/**"
---

# Correctness Testing Instructions

> **When to consult this file:** You are running, modifying, debugging, or extending the correctness/validation tests for the cell simulator. This covers the pytest harness in [cpp/simulation/tests/python/](../../cpp/simulation/tests/python), the f64 Rust reference simulator in [rust/cpu_ref/](../../rust/cpu_ref), the parity-test fixtures, and the HTML test report. For running the simulator itself see [cell-simulation.instructions.md](cell-simulation.instructions.md). For cluster jobs see [cluster-operations.instructions.md](cluster-operations.instructions.md).

## TL;DR

- One pytest suite at [cpp/simulation/tests/python/](../../cpp/simulation/tests/python) drives **all** automated correctness checks. It auto-discovers the binary, drives it with subprocess, parses checkpoints/trajectories, asserts on metrics, and writes an HTML report.
- The **bit-truth reference** is `rust/cpu_ref` — a single-file, f64, rayon-parallel reimplementation of the PDE that consumes the same v7 checkpoints as the GPU sim and replays the same scripted tumble events. Cluster-generated cpu_ref output lives in `cpp/simulation/tests/python/fixtures/<scenario>/` and is committed to git.
- Run the fast suite with `pytest cpp/simulation/tests/python/`. Add `--run-slow` for the long parity/MSD tests. Reports land in `cpp/simulation/tests/python/test_report/report.html`.

## Layout

```
cpp/simulation/tests/python/
├─ conftest.py                  # binary discovery, run_sim/run_baseline, fixtures (sim, baseline_sim, v2_sim)
├─ report.py                    # HTML report plugin: record_metric/_timeseries/_trajectory/_comparison_panel
├─ cpu_reference.py             # NumPy f64 in-process reference (used by short Phase-H tests)
├─ test_correctness.py          # Tier-1 plumbing: smoke, trajectory integrity, checkpoint format
├─ test_physics.py              # Tier-2 physics: steady state, interface width, MSD, two-cell, energy
├─ test_features.py             # CLI-flag matrix (~118 tests across 11 sections)
├─ test_cutover_parity.py       # GPU sim vs Rust cpu_ref bit-parity over 2τ  ← the headline test
├─ fixtures/
│  ├─ cpu_ref_2tau/             # uniform γ, v_A=0.01 (commit abfd183, Apr 29 2026)
│  ├─ cpu_ref_2tau_soft/        # one soft cell γ=0.35, v_A=0.014 (Palmieri params; pending nibi job 13099234)
│  └─ golden_v6_twocell.bin     # committed v6 checkpoint, sha256 d7356f… (format-drift guard)
├─ test_report/                 # generated: report.html + PNGs (gitignored)
└─ AUDIT.md                     # rationale for every test / what it gates / known caveats
```

## Methodology — what each layer is for

The suite is layered so a regression at any level fails at the most informative layer first.

### Tier 1 — Plumbing (`test_correctness.py`)
Smoke + I/O round-trips. Single/multi-cell runs complete without NaNs; trajectory headers carry `N, Lx, Ly, dim, tau, v_A`; timestamps are strictly monotonic; checkpoint stores `cur_time` as f64 (regression test for the v5 fix where f32 capped time at 2¹⁸). Every test ≤ 1 s.

### Tier 2 — Physics with analytical baselines (`test_physics.py`)
Each test class has either a closed-form prediction or a documented metastable-state caveat. Highlights:
- `TestSingleCellSteadyState` — V → πR² to 1 % (gradient flow of single-cell Lyapunov F).
- `TestInterfaceWidth` — 10–90 width = 0.80λ analytical (sim integrates `−½ δF/δφ`, coefficient `30γ/λ²`).
- `TestEnergyMonotonicity` — F[φ] non-increasing at v_A=0.
- `TestAnalyticalSteadyState` — PDE residual = 0 on measured equilibrium φ (independent of cpu_ref).
- `TestMSDCrossover` — ballistic exponent in [3.5, 4.5] over short window; diffusive at long time. v_A=10⁻², τ=10⁴ ⇒ D_eff = v_A²·τ/2 = 0.5 px²/TU.
- `TestTwoCellRepulsion` — d_eq = 2R + 2λ analytical.
- `TestPalmieriSoftVsCtrl` (@slow) — 5-seed ensemble, asserts D_soft/D_ctrl > 1.2 (paper says 1.5).

### Tier 3 — CLI / format (`test_features.py`)
Every CLI flag gets a "accepted, has effect, persists in checkpoint" trio.

### Tier 4 — Bit-parity vs the f64 reference (`test_cutover_parity.py`)
The headline regression test. Drives the GPU sim and the cpu_ref output side-by-side and checks two views:
1. **Per-frame trajectory drift |Δr|(t)** with PBC wrap, asserting envelopes on `rms_max`, `max_p95`, `max_final`, `max_any`.
2. **Final phase-field RMS** — paint each cell's TILE_T tile onto the periodic (Ny, Nx) domain, compare against the cpu_ref final Σφᵢ snapshot, assert on `phi_rms` and `phi_max`.

Empirical envelopes for the `cpu_ref_2tau` fixture (GPU sm_75, Linux x86_64 Rust f64): rms |Δr| ≈ 0.10, max |Δr| ≈ 0.27, phi_rms ≈ 1.1e-2. Test thresholds are ~5× observed, set to catch real regressions without flapping. Drift is dominated by f32↔f64 round-off and atomic-add ordering — **not** physics differences. PBC-seam wraps at tumble boundaries can spike to ~1 px and are excluded from `max_final`.

## The Rust f64 reference simulator (`rust/cpu_ref/`)

Single-binary reimplementation of the same PDE the GPU runs, but in f64 end-to-end with deterministic-when-N=1 rayon reductions. Reads the same v7 checkpoints (`magic = 0x43454C4C`, version 7) and the same scripted tumble events.

**Key contract:**
- Reads `POLR` sidecar (per-cell initial polarities) — without this, Rust silently regenerates θ from a Xoshiro PRNG and the parity test is meaningless. Reference data generated before commit `abfd183` is invalid.
- Reads `GAMA` sidecar (per-cell γ) — added in commit `d9447a7`. Without this, soft-cell scenarios silently use uniform γ from `params.gamma`.
- Outputs `traj.npz` (full per-frame phi field), optional `trajectory.txt` (Σφ²-weighted periodic-aware centroids in the same format as the GPU's `trajectory.txt`), and optional `events.txt` (replay log).
- `--scripted-events <file>` bypasses the per-step PRNG tumble path entirely; this is what makes the parity test deterministic.

**Build (local Windows):**
```powershell
cd rust/cpu_ref
cargo build --release --bin cpu_ref
```

**Build (nibi):**
```bash
module load StdEnv/2023 rust/1.91.0
cd /scratch/ssilber/cpu_ref_validate/src_repo/rust/cpu_ref
cargo build --release --bin cpu_ref
# Binary at target/release/cpu_ref
```

**Reference command (matches `run_2tau_v2.sbatch`):**
```bash
cpu_ref --ic ic_checkpoint.bin \
        --v-a 0.01 --tau 10000 --t-end 20000 --dt 0.01 \
        --save-every 5000 --polarity-seed 12345 \
        --out traj.npz --trajectory trajectory.txt --events events.txt \
        --threads $SLURM_CPUS_PER_TASK
```

## Fixtures and how reference data is generated

Each parity scenario has a self-contained directory under `fixtures/`:

| file | purpose |
|---|---|
| `ic_checkpoint.bin` | v7 checkpoint, t=0.01 (one-step burn-in to materialize sidecars) |
| `events.txt` | scripted tumble events, format `# t cid old_theta new_theta` |
| `ref_trajectory.txt` | f64 cpu_ref ground-truth centroids, 400 frames over 2τ |
| `ref_final_phi.npz` | final-frame full-domain Σφᵢ for whole-array RMS comparison |
| `README.md` | provenance, exact command, slurm job ID, drift envelopes |

**Generation protocol** (mirror the existing fixture's README exactly when adding a new one):
1. Generate IC locally with `cell_sim` using `--save-final-checkpoint` and the relevant `--gamma <val>:cellN` flag(s). Always use `--seed 12345` to match the existing fixtures unless deliberately diverging.
2. Commit the IC + a stub README to `cpp/simulation/tests/python/fixtures/<name>/`.
3. On nibi, run cpu_ref via slurm with the IC, capture `trajectory.txt` and `events.txt`. Use 8h walltime, 24 cores, account `rrg-mkarttu-ab-phase_cpu`. Template: `/scratch/ssilber/cpu_ref_validate/run_2tau_v2.sbatch`.
4. Extract `ref_final_phi.npz` from `traj.npz` (`module load StdEnv/2023 gcc/12.3 python/3.11 scipy-stack/2025a`, sum phi over the cells axis).
5. Download the three artifacts, drop into the fixture dir, update the README with the slurm job ID and observed envelopes, commit.

**When to regenerate**: only if the f64 Rust path or IC layout changes. GPU-side changes (kernel launch order, FMA scheduling, atomics) re-tune the *thresholds* in `test_cutover_parity.py` but don't invalidate the reference itself.

## Running the tests

### Local (Windows)
```powershell
# Fast suite (skips @slow)
cd c:\Users\stevensilber\source\repos\data_processing
pytest cpp/simulation/tests/python/ -v

# Full suite including @slow (cpu_ref parity, Palmieri ensemble, MSD scaling)
pytest cpp/simulation/tests/python/ -v --run-slow

# Single test file or class
pytest cpp/simulation/tests/python/test_cutover_parity.py -v --run-slow

# Persist parity artifacts (drift PNG, phi-error map, parity_stats.npz) outside tmp_path
pytest cpp/simulation/tests/python/test_cutover_parity.py -v --run-slow \
       --parity-artifacts cpp/simulation/tests/python/test_report/parity_artifacts
```

### Binary discovery
- Default: `SIM_NAME=cell_sim`, searched under `cpp/simulation/build/bin/[Release/]`.
- Override: `$env:SIM_BINARY = "C:\path\to\cell_sim.exe"`.
- Baseline (for migration tests): `$env:BASELINE_BINARY = ...`. Auto-detected baseline is suppressed when it points at the same file as `SIM_BINARY` (post-cutover guard).
- The harness probes `--help` once and `requires_flag(...)` skips tests for flags the binary lacks.

### TILE_T and the `build_t192/` build tree
- Default CMake `CELL_SIM_TILE_T=320`. Fast tests use `-N 200` style domains
  that are **smaller than 320**, so they fatal against the default build
  ("[FATAL] domain (200 x 200) smaller than TILE_T=320").
- A pre-configured `cpp/simulation/build_t192/` tree exists with
  `CELL_SIM_TILE_T=192`. Rebuild it (`cmake --build cpp/simulation/build_t192
  --config Release`) and point pytest at its binary:
  `$env:SIM_BINARY = (Resolve-Path cpp/simulation/build_t192/bin/cell_sim.exe).Path`.
- May 23, 2026 baseline on this binary: **126 passed, 40 skipped, 1 failed**
  (the failure is the pre-existing v4-legacy-checkpoint test, unrelated).
- Long-term fix tracked as a TEST-DEBT item in `cpp/simulation/FIX_REPORT.md`
  — tests should derive valid `-N` from the binary's TILE_T, not hard-code.

### Markers
- `@pytest.mark.slow` — deselected by default. Enable with `--run-slow`.
- Tests that need the baseline use `@requires_baseline()` and skip if it's missing.

## Output: artifacts and the HTML report

Every test that calls `record_*` from [report.py](../../cpp/simulation/tests/python/report.py) contributes to the report at `cpp/simulation/tests/python/test_report/report.html`.

| record API | what it produces | when to use |
|---|---|---|
| `record_description(node, text)` | one-line italic header on the test card | always, first call wins |
| `record_metric(node, key, value, expected, tolerance, unit)` | a row in the test's metrics table; tolerance can be `"5%"` or absolute float | the assert should be redundant with this |
| `record_snapshot(node, phi_2d, title)` | single-panel PNG (inferno cmap) | final-state snapshots |
| `record_phi_from_checkpoint(node, chk, title)` | composites all per-cell tiles onto domain Ny×Nx, then `record_snapshot` | most physics tests |
| `record_timeseries(node, x, {label: y, …}, ylog=)` | line plot, log-y optional | drift envelopes, MSD, energy traces |
| `record_trajectory(node, xs, ys)` or `(node, {"cell 0": (xs, ys), …})` | xy path with start/end markers, equal aspect | centroid trajectories |
| `record_comparison_panel(node, sim_grid, ref_grid, title)` | 3-panel sim | ref | log-scale `|Δ|` | parity tests vs cpu_reference |
| `record_composite_frame(node, phi, caption)` | extra single panel attached to the same card | second snapshot when no ref to compare |

The plugin runs on `pytest_sessionfinish`; the report regenerates every session. Report dir is gitignored.

**Report structure:**
- Top: counts (PASS / FAIL / INFO / SKIP / NOT RUN, snapshot+chart counts).
- Summary table: one row per test with the most-interesting metric and PASS/FAIL/INFO status.
- Detailed cards: visuals row (snapshot, extras, trajectory, timeseries) + metrics table.
- Skipped section: reason for each skipped test.

For `test_cutover_parity.py` specifically, the test also writes (in `tmp_path/parity_artifacts/`, optionally mirrored via `--parity-artifacts`):
- `parity_stats.npz` — `ts, rms_dr, max_dr, dr_per_cell, phi_err_2d, summary scalars`.
- `summary.txt` — newline-separated `key: value`.
- `drift.png` — per-cell drift on log-y with rms/max envelopes.
- `phi_err.png` — final phase-field error map (RdBu_r, symmetric vmin/vmax).

## Interpreting results

**A green `test_2tau_scripted_events`** means: the GPU's integrated trajectories agree with f64 cpu_ref to within ~0.5 px rms and ~1 px max over 2τ, and the painted final phase field agrees to RMS < 5e-2 / max < 0.7. This is the strongest correctness signal we have. It is **not** a physics test — it's an "are we doing the same arithmetic?" test.

**A red `test_2tau_scripted_events`** means one of:
1. **f32-vs-f64 envelope grew** — accept new envelope only after re-validating against an analytical baseline (e.g. `TestEnergyMonotonicity` and `TestAnalyticalSteadyState` still pass at v_A=0).
2. **PBC seam regression** — `max_any` spikes mid-run while `max_final` and `phi_rms` stay nominal. Look at `dr_per_cell` in `parity_stats.npz`; a single cell jumping by L/2 = 188 means a cell crossed a boundary and one side is wrapping differently from the other.
3. **Sidecar regression** — if `phi_rms` is large from frame 0, an IC sidecar isn't being honored. Check for the `[init] using POLR sidecar` and (when applicable) `[init] using GAMA sidecar` log lines from cpu_ref.
4. **Real physics regression** — physics tier (`test_physics.py`) goes red simultaneously. Fix that first; cutover_parity will follow.

For the soft-parity scenario (`cpu_ref_2tau_soft`, γ=0.35 on cell 0), the same envelopes apply but expect the soft cell's drift to be the `max_any` driver — its larger excursions are physical, not numerical, and that's the whole point of the test.

## Common workflows

### Adding a new parity scenario
1. Decide what you're varying (γ, v_A, IC layout). Keep everything else identical to `cpu_ref_2tau` to make the comparison meaningful.
2. Generate IC locally → commit IC + stub README → submit cpu_ref slurm job.
3. Once slurm completes, drop `ref_trajectory.txt`, `events.txt`, `ref_final_phi.npz` next to the IC, fill in README, commit.
4. Add a `TestCutoverParity<Name>` class to `test_cutover_parity.py` parameterised on `FIXTURE_DIR`, reusing all the helpers (`_per_frame_drift`, `_final_phi_full`, `_save_artifacts`).
5. Mark it `@pytest.mark.slow`.

### Diagnosing a flaky parity assertion
1. Re-run with `--parity-artifacts <dir>`; compare `drift.png` and `phi_err.png` to the previous good run.
2. `np.load(parity_stats.npz)` → inspect `dr_per_cell` to find which cell(s) drift first.
3. Bisect against git: `test_cutover_parity` tolerates ~5× the empirical envelope, so a real regression usually shows a >2× growth in `rms_max` over a single commit.
4. If only `max_any` spikes (not `rms_max`), it's almost certainly a PBC-seam timing artifact, not a physics drift.

### Adding a new test that needs a CPU baseline
- For short-time stability checks (≤ 100 steps, ≤ 200×200 domain): use [cpu_reference.py](../../cpp/simulation/tests/python/cpu_reference.py) — it's a pure-NumPy in-process implementation, no subprocess, no fixtures.
- For long-time checks (need f64 over thousands of steps): add a fixture under `fixtures/` and a test in `test_cutover_parity.py`. Don't try to run cpu_ref in-process — it's a slurm-class workload.

## Pitfalls

- **Don't skip the IC burn-in step.** Run `cell_sim ... -t 0.01 --save-final-checkpoint` to step once before saving — this is what materialises POLR/GAMA sidecars in the right places.
- **Don't pass `--polarities` and a POLR-bearing IC together.** The CLI flag wins and silently overrides the sidecar; you'll get apparent parity failures with no obvious cause.
- **Don't rebuild cpu_ref with a non-default Rust toolchain.** rayon's parallel reductions are deterministic only at the assumption of a stable thread count and reduction tree. Pin to `rust/1.91.0` on nibi.
- **Don't commit `target/`, `Cargo.lock` is fine.** `.gitignore` already excludes the build dir.
- **`record_metric` with `tolerance` controls PASS/FAIL on the report card, not the test exit code.** The `assert` is what fails the test. Keep both in sync.
- **The HTML report regenerates on every session.** If you only ran one test, only that test's card is in the report — previous snapshots are wiped. Use `--parity-artifacts` for persistence.
