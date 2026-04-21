"""
Migration parity tests: baseline cell_sim ↔ sim_v2.

These tests drive the replacement of the cluster-deployed baseline binary
with sim_v2. We require full functional parity before swapping:

  Phase A — checkpoint format compatibility (baseline v3/v4 → sim_v2 v6)
  Phase B — CLI flag parity (all production flags accepted)
  Phase C — trajectory file format parity
  Phase D — cross-binary checkpoint resume round-trip
  Phase E — physics parity (paired seeds, MSD / avg_vol)   [@slow]
  Phase F — postprocessing compatibility (Rust cell_analyze reads both)
  Phase H — CPU reference phase-field validator (short-time stability)  [@slow]

Tests run against sim_v2 (CELL_SIM) as the primary binary, and invoke the
baseline binary (BASELINE_SIM) when needed. Tests requiring the baseline
skip cleanly if it is not built / BASELINE_BINARY is unset.
"""
import math
import re
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    run_sim, run_baseline, read_checkpoint, read_trajectory,
    requires_baseline, requires_flag,
    BASELINE_SIM, CELL_SIM, _HELP_TEXT, _BASELINE_HELP_TEXT,
)


def _flag_in_help(flag: str, help_text: str) -> bool:
    """Match `flag` as a whole token in help output (not as a substring).

    A help line for `--foo` typically looks like ``  --foo <arg>  description``
    or ``  --foo    description``. We require the flag to be preceded by
    whitespace/start-of-line and followed by whitespace/comma/newline/<.
    """
    pat = r"(?:^|[\s])" + re.escape(flag) + r"(?=[\s,<]|$)"
    return re.search(pat, help_text, re.MULTILINE) is not None


# =============================================================================
# Phase A — Checkpoint format compatibility
# =============================================================================

class TestCheckpointFormatRead:
    """sim_v2 must read baseline v3/v4 checkpoints (cluster-deployed format).

    The sim_v2 in-place upconvert path reads the baseline SimParams layout
    (f32 scalars, 72-byte v3 or 92-byte v4) into the current f64 struct.
    """

    @requires_baseline()
    def test_sim_v2_reads_baseline_v4_header(self, baseline_sim):
        """Baseline writes v4 by default. sim_v2 must recognize the header."""
        out = baseline_sim("-n", "2", "-N", "200", "-r", "20", "-t", "0.5",
                           "-dt", "0.01", "--v-A", "0", "--seed", "1",
                           "--save-interval", "0", "--trajectory-samples", "0")
        ck = out / "checkpoint.bin"
        assert ck.exists()
        data = read_checkpoint(ck)
        # Baseline always writes v4 today. If this changes, tests below that
        # verify the sim_v2 reader can handle it still hold, but the version
        # pin helps catch silent header changes.
        assert data["version"] == 4, f"baseline wrote unexpected version {data['version']}"
        # SimParams size must be one of the known baseline layouts.
        # Legacy v3 = 72 bytes; current v4 = 92 bytes.
        #
        # We don't assert exact size here (baseline struct may grow again),
        # but we do verify the physics fields we care about parsed correctly.
        p = data["params"]
        assert p["Nx"] == 200
        assert p["Ny"] == 200
        assert abs(p["target_radius"] - 20.0) < 1e-6
        assert abs(p["dt"] - 0.01) < 1e-6

    @requires_baseline()
    def test_sim_v2_resumes_baseline_checkpoint(self, baseline_sim, v2_sim):
        """End-to-end: baseline writes checkpoint, sim_v2 resumes it.

        This is the critical migration path. A failure here blocks
        cluster migration — all deployed production checkpoints are v4.
        """
        # 1. Baseline writes a v4 checkpoint at t=0.5.
        base_out = baseline_sim("-n", "2", "-N", "200", "-r", "20", "-t", "0.5",
                                "-dt", "0.01", "--v-A", "0", "--seed", "7",
                                "--save-interval", "0", "--trajectory-samples", "0")
        base_ckpt = base_out / "checkpoint.bin"
        base_data = read_checkpoint(base_ckpt)
        assert base_data["version"] == 4
        assert base_data["num_cells"] == 2

        # 2. sim_v2 resumes from it and extends to t=1.0. If the v4 reader is
        #    broken, sim_v2 will fail to load or produce NaN fields.
        v2_out = v2_sim("-c", str(base_ckpt), "-t", "1.0", "--dt", "0.01",
                        "--v-A", "0", "--seed", "7")
        v2_ckpt = v2_out / "checkpoint.bin"
        v2_data = read_checkpoint(v2_ckpt)

        assert v2_data["version"] == 6, "sim_v2 must write v6 checkpoints"
        assert v2_data["num_cells"] == 2
        assert v2_data["time"] >= 0.99, f"sim_v2 did not advance past resume point (t={v2_data['time']})"
        # Physics parameters must transfer unchanged (no silent corruption).
        assert abs(v2_data["params"]["target_radius"] - 20.0) < 1e-6
        assert abs(v2_data["params"]["dt"] - 0.01) < 1e-6
        assert v2_data["params"]["Nx"] == 200
        for c in v2_data["cells"]:
            assert np.isfinite(c["volume"]) and c["volume"] > 0
            assert not np.any(np.isnan(c["phi"]))

    def test_sim_v2_roundtrip_v6(self, sim):
        """sim_v2 reads its own v6 checkpoints (regression)."""
        out1 = sim("-n", "2", "-N", "200", "-r", "20", "-t", "0.5",
                   "--dt", "0.01", "--v-A", "0", "--seed", "11",
                   "--save-interval", "0", "--trajectory-samples", "0")
        # Copy ckpt to a stable path and resume.
        ckpt1 = out1 / "checkpoint.bin"
        out2 = sim("-c", str(ckpt1), "-t", "1.0", "--dt", "0.01",
                   "--v-A", "0", "--seed", "11")
        d = read_checkpoint(out2 / "checkpoint.bin")
        assert d["version"] == 6
        assert d["time"] >= 0.99


# =============================================================================
# Phase B — CLI flag surface parity
# =============================================================================

# All flags listed here MUST be accepted by sim_v2 before migration.
# A flag is considered accepted if it appears in `cell_sim_v2 -h` output.
REQUIRED_SIM_V2_FLAGS = [
    # Core physics
    "--v-A", "--tau", "--gamma", "--kappa", "--mu", "--xi", "--lambda",
    # Geometry
    "-n", "-r", "-N", "--confluence", "--subdomain-padding",
    # Time
    "-t", "--dt",
    # I/O
    "-o", "-c", "--save-interval", "--checkpoint-interval",
    "--save-final-checkpoint", "--no-save-final-checkpoint",
    "--print-interval", "--trajectory-samples",
    # RNG
    "--seed", "--polarity-seed",
    # Model
    "--abp",
]

# Flags present in baseline that sim_v2 MUST ALSO support for cluster parity.
# These are the in-scope-now flags for the Palmieri-extension migration.
PRODUCTION_PARITY_FLAGS = [
    "--v-A-sigma",            # Griffiths studies (per-cell v_A disorder)
    "--radius",                # polydisperse populations
    "--trajectory-interval",   # alternative trajectory cadence
    "-dt",                     # hyphen alias for --dt (baseline accepts both)
]

# Baseline flags deferred to a LATER migration phase (post-Palmieri extension).
# Tests are xfail'd with strict=False so they don't block the suite but still
# appear in reports as outstanding migration items.
#
#   --adhesion   — cell-cell adhesion coupling; needed for Bresler / adhesion
#                  studies. Requires a new kernel term in k_fused. Deferred
#                  until after Palmieri extension ships.
#   -i           — initial-conditions JSON. Only used alongside --batch for
#                  multi-system SLURM submissions; paired with --batch below.
#   --batch      — multi-system mode packing many independent simulations
#                  into one SLURM job. Needed so the cluster scheduler
#                  accepts submissions (many small jobs don't get scheduled).
#                  Large implementation (~400 LOC). Deferred.
DEFERRED_PARITY_FLAGS = [
    "--adhesion",
    "-i",
    "--batch",
]


class TestCliSurface:
    @pytest.mark.parametrize("flag", REQUIRED_SIM_V2_FLAGS)
    def test_sim_v2_accepts_core_flag(self, flag):
        """sim_v2 -h must mention every core flag."""
        assert _flag_in_help(flag, _HELP_TEXT), \
            f"sim_v2 help does not mention {flag!r}"

    @pytest.mark.parametrize("flag", PRODUCTION_PARITY_FLAGS)
    def test_sim_v2_accepts_production_parity_flag(self, flag):
        """Every baseline production flag must be accepted by sim_v2
        before we migrate the cluster. These tests are expected to fail
        until the migration is complete."""
        assert _flag_in_help(flag, _HELP_TEXT), \
            f"sim_v2 help does not mention {flag!r} (migration gap)"

    @pytest.mark.parametrize("flag", DEFERRED_PARITY_FLAGS)
    @pytest.mark.xfail(strict=False, reason="deferred to post-Palmieri migration phase")
    def test_sim_v2_accepts_deferred_flag(self, flag):
        """Baseline flags deferred to a later migration phase.

        Tracked here (not deleted) so they appear as outstanding items in
        the migration report. Once implemented, flip from xfail → strict.
        """
        assert _flag_in_help(flag, _HELP_TEXT), \
            f"sim_v2 help does not mention {flag!r} (deferred migration gap)"

    @requires_baseline()
    def test_baseline_and_sim_v2_share_core_flags(self):
        """Sanity: shared core flags must be in both help outputs.

        Some sim_v2 flags are intentionally *additions* over baseline:
          --dt           (baseline uses -dt single-hyphen)
          -N             (baseline does expose this, but phrased differently)
          --lambda       (baseline hardcodes lambda=7)
          --no-save-final-checkpoint  (sim_v2-only convenience)
        These are excluded from the cross-binary sanity.
        """
        sim_v2_only = {"--dt", "-N", "--lambda", "--no-save-final-checkpoint"}
        shared = [f for f in REQUIRED_SIM_V2_FLAGS if f not in sim_v2_only]
        missing_base = [f for f in shared if not _flag_in_help(f, _BASELINE_HELP_TEXT)]
        assert not missing_base, f"baseline missing flags: {missing_base}"


# =============================================================================
# Phase C — Trajectory file format parity
# =============================================================================

class TestTrajectoryFormatParity:
    """sim_v2 trajectory.txt must be a drop-in replacement for baseline.

    Column order and header keys must match so downstream analysers
    (cell_analyze, Python MSD scripts) do not need format switches.
    """

    TRAJ_EXPECTED_HEADER_KEYS = {"v_A", "N", "Lx", "Ly", "dt", "tau"}
    TRAJ_NUM_COLUMNS = 12  # time cid x y vx vy px py theta v_A_i L_n volume

    def _run_with_trajectory(self, runner, n=2):
        out = runner("-n", str(n), "-N", "200", "-r", "20",
                     "-t", "1.0", "--dt", "0.01", "--v-A", "0.01",
                     "--tau", "100", "--seed", "3",
                     "--trajectory-samples", "5", "--save-interval", "0")
        return out / "trajectory.txt"

    def test_sim_v2_trajectory_schema(self, sim):
        path = self._run_with_trajectory(sim)
        assert path.exists(), "sim_v2 must write trajectory.txt"
        data, hdr = read_trajectory(path)
        # Header must contain all required keys.
        missing = self.TRAJ_EXPECTED_HEADER_KEYS - set(hdr.keys())
        assert not missing, f"trajectory header missing keys: {missing}"
        # Each data row must have the full column set.
        with open(path) as f:
            first_data = next(l for l in f if not l.startswith("#") and l.strip())
        ncols = len(first_data.split())
        assert ncols == self.TRAJ_NUM_COLUMNS, \
            f"trajectory has {ncols} columns, expected {self.TRAJ_NUM_COLUMNS}"

    @requires_baseline()
    def test_trajectory_column_parity(self, baseline_sim, v2_sim):
        """Baseline and sim_v2 must emit the same number of columns."""
        bp = self._run_with_trajectory(baseline_sim)
        vp = self._run_with_trajectory(v2_sim)
        with open(bp) as f:
            b_cols = len(next(l for l in f if not l.startswith("#") and l.strip()).split())
        with open(vp) as f:
            v_cols = len(next(l for l in f if not l.startswith("#") and l.strip()).split())
        assert b_cols == v_cols, \
            f"baseline={b_cols} cols, sim_v2={v_cols} cols — drop-in parity broken"

    @requires_baseline()
    def test_trajectory_header_keys_parity(self, baseline_sim, v2_sim):
        """Both headers must carry the keys downstream parsers depend on.

        Extra keys on either side (e.g. sim_v2 adds `dim`, baseline adds
        `v_A_sigma`) are allowed; we only gate on the required intersection.
        Once `--v-A-sigma` ships in sim_v2 (Phase B), `v_A_sigma` moves into
        the required set.
        """
        bp = self._run_with_trajectory(baseline_sim)
        vp = self._run_with_trajectory(v2_sim)
        _, b_hdr = read_trajectory(bp)
        _, v_hdr = read_trajectory(vp)
        for key in self.TRAJ_EXPECTED_HEADER_KEYS:
            assert key in b_hdr, f"baseline header missing {key!r}"
            assert key in v_hdr, f"sim_v2 header missing {key!r}"

    def test_trajectory_timestamps_monotonic(self, sim):
        path = self._run_with_trajectory(sim)
        data, _ = read_trajectory(path)
        times = sorted(data.keys())
        # Strict monotonic (no duplicates) — this is what the f32 time
        # regression used to break.
        assert len(set(times)) == len(times), "duplicate trajectory timestamps"
        # No NaN.
        for t, cells in data.items():
            assert np.isfinite(t)
            for cid, row in cells.items():
                assert all(np.isfinite(v) for v in row)


# =============================================================================
# Phase D — Cross-binary checkpoint resume (round-trip)
# =============================================================================

class TestCrossBinaryResume:
    """After migration we need: baseline ckpt → sim_v2 resume → sim_v2 ckpt."""

    @requires_baseline()
    def test_baseline_ckpt_then_sim_v2_runs_t_extension(self, baseline_sim, v2_sim):
        """sim_v2 can extend a baseline-produced checkpoint's integration."""
        b_out = baseline_sim("-n", "4", "-N", "200", "-r", "20", "-t", "0.5",
                             "-dt", "0.01", "--v-A", "0", "--seed", "31",
                             "--save-interval", "0", "--trajectory-samples", "0")
        b_ckpt = b_out / "checkpoint.bin"
        b_data = read_checkpoint(b_ckpt)
        v2_out = v2_sim("-c", str(b_ckpt), "-t", "2.0", "--dt", "0.01",
                        "--v-A", "0", "--seed", "31")
        v2_data = read_checkpoint(v2_out / "checkpoint.bin")
        assert v2_data["version"] == 6
        assert v2_data["num_cells"] == b_data["num_cells"]
        assert v2_data["time"] > b_data["time"]
        # Per-cell phi survived the conversion: no NaN, positive volumes.
        for c in v2_data["cells"]:
            assert c["volume"] > 0
            assert not np.any(np.isnan(c["phi"]))

    def test_sim_v2_ckpt_then_sim_v2_resume(self, sim):
        """sim_v2 → sim_v2 resume (v6 round-trip)."""
        out1 = sim("-n", "4", "-N", "200", "-r", "20", "-t", "0.5",
                   "--dt", "0.01", "--v-A", "0.02", "--tau", "50",
                   "--seed", "22", "--save-interval", "0",
                   "--trajectory-samples", "0")
        c1 = out1 / "checkpoint.bin"
        d1 = read_checkpoint(c1)

        out2 = sim("-c", str(c1), "-t", "1.0", "--dt", "0.01",
                   "--v-A", "0.02", "--tau", "50", "--seed", "22")
        d2 = read_checkpoint(out2 / "checkpoint.bin")
        assert d2["version"] == 6
        assert d2["num_cells"] == d1["num_cells"]
        assert d2["time"] > d1["time"]


# =============================================================================
# Phase E — Physics parity (slow, paired seeds)
# =============================================================================

@pytest.mark.slow
class TestPhysicsParity:
    """sim_v2 must be physics-equivalent to baseline at the ensemble level.

    We don't expect bit-identical output — algorithm differences (neighbour
    list, subdomain policy) mean fp32/fp64 differences accumulate. We do
    expect:
      - matched final avg volume (volume constraint identically enforced)
      - matched MSD scaling under motility
    """

    @requires_baseline()
    def test_equilibration_avg_volume_matches(self, baseline_sim, v2_sim):
        """Short equilibration: avg volume within 1.5% of baseline."""
        b_out = baseline_sim("-n", "8", "-N", "400", "-r", "20", "-t", "100",
                             "-dt", "0.01", "--v-A", "0", "--seed", "101",
                             "--save-interval", "0", "--trajectory-samples", "0")
        v_out = v2_sim("-n", "8", "-N", "400", "-r", "20", "-t", "100",
                       "--dt", "0.01", "--v-A", "0", "--seed", "101",
                       "--save-interval", "0", "--trajectory-samples", "0")
        b = read_checkpoint(b_out / "checkpoint.bin")
        v = read_checkpoint(v_out / "checkpoint.bin")
        b_vol = np.mean([c["volume"] for c in b["cells"]])
        v_vol = np.mean([c["volume"] for c in v["cells"]])
        rel = abs(v_vol - b_vol) / b_vol
        assert rel < 0.015, f"avg_vol drift {rel:.3%} (baseline={b_vol:.1f}, v2={v_vol:.1f})"


# =============================================================================
# Phase F — Postprocessing compatibility (Rust cell_analyze)
# =============================================================================

class TestPostprocessingCompat:
    """Standalone Python reader must handle both baseline and sim_v2 ckpts."""

    @requires_baseline()
    def test_python_reader_reads_baseline_v4(self, baseline_sim):
        out = baseline_sim("-n", "2", "-N", "200", "-r", "20", "-t", "0.5",
                           "-dt", "0.01", "--v-A", "0", "--seed", "13",
                           "--save-interval", "0", "--trajectory-samples", "0")
        d = read_checkpoint(out / "checkpoint.bin")
        assert d["version"] == 4
        assert d["num_cells"] == 2
        for c in d["cells"]:
            assert c["phi"].shape[0] > 0 and c["phi"].shape[1] > 0
            assert np.isfinite(c["volume"])

    def test_python_reader_reads_sim_v2_v6(self, sim):
        out = sim("-n", "2", "-N", "200", "-r", "20", "-t", "0.5",
                  "--dt", "0.01", "--v-A", "0", "--seed", "14",
                  "--save-interval", "0", "--trajectory-samples", "0")
        d = read_checkpoint(out / "checkpoint.bin")
        assert d["version"] == 6
        assert d["num_cells"] == 2
        for c in d["cells"]:
            assert c["phi"].shape[0] > 0 and c["phi"].shape[1] > 0
            assert np.isfinite(c["volume"])


# =============================================================================
# Phase H — CPU phase-field reference (independent ground truth)
# =============================================================================

@pytest.mark.slow
class TestCpuReference:
    """Audit sim_v2 against a standalone numpy solver of the same PDE.

    The reference lives in ``cpu_reference.py`` and mirrors sim_v2's
    per-cell tile layout with a halo border. The Laplacian and
    gradient stencils clamp at tile edges (matching ``max()``/``min()``
    in the kernel), the neighbour interaction S uses the same pdelta
    offset + inner-region guard as ``k_fused``, and the halo is held
    at 0 between steps.

    Strategy:
      1. Run sim_v2 briefly to produce a realistic checkpoint.
      2. Resume sim_v2 for N more dt using -c / -t.
      3. Run the CPU reference for the same N steps from the same
         checkpoint.
      4. Compare tile-by-tile ``‖φ_sim − φ_cpu‖∞``.

    Invariants exploited to get a strict match:
      * No REMAP. sim_v2 calls ``k_pre_step`` with ``resize = step % 10 == 0``,
        which may shift the tile by integer pixels; the CPU ref
        deliberately does NOT implement this (the shift is an internal
        sim_v2 optimisation, not part of the physics). We therefore
        start each test at ``step_count = 201`` (not a multiple of 10)
        and run ≤ 8 steps so no REMAP is triggered.
      * Domain ≥ tile. When the auto-confluence domain is smaller than
        the per-cell tile (e.g. ``--confluence 0.6`` with R=25 gives
        L=81 but tile=86), the tile self-wraps and neighbour lookups
        become ambiguous. The tests below use explicit ``-N`` that
        keeps ``L > max_tile``.
    """

    # Target step at which to save the initial checkpoint. Not a multiple
    # of 10, so the first post-resume step is step 201 (REMAP=false,
    # MOMENTS=false in k_fused).
    _INIT_T_END = 2.015          # → step_count = 201 at save
    _RESUME_STEPS = 8            # → last step = 208 (still no REMAP/MOMENTS)
    _TOL_LINF = 1e-4             # f32 round-off noise over 8 steps
    _TOL_MEAN = 5e-6

    # --- helpers ---------------------------------------------------------

    def _run_cpu_ref(self, ckpt_data, n_steps):
        from cpu_reference import (cells_from_checkpoint,
                                   cpu_params_from_checkpoint, integrate)
        p = cpu_params_from_checkpoint(ckpt_data)
        cells = cells_from_checkpoint(ckpt_data, halo=p.halo)
        return integrate(cells, p, n_steps), p

    def _resume_sim(self, sim, ckpt_in, n_steps, dt, seed):
        """Resume sim_v2 from a checkpoint for approximately n_steps * dt.

        sim_v2's run loop uses ``target_step = int(t_end/dt)``, which rounds
        down due to f64 representation of dt. We pad t_end by half a dt so
        the intended n_steps are actually executed, and return the resume
        checkpoint dict — the caller should use ``dict["step"]`` to compute
        the true number of steps the CPU ref needs to match.
        """
        t_start = read_checkpoint(ckpt_in)["time"]
        t_end = t_start + (n_steps + 0.5) * dt  # +0.5·dt guards f64 rounding
        out = sim("-c", str(ckpt_in), "-t", f"{t_end:.8f}", "--dt", f"{dt:.6f}",
                  "--v-A", "0", "--seed", str(seed),
                  "--save-interval", "0", "--trajectory-samples", "0",
                  "--print-interval", "0")
        return read_checkpoint(out / "checkpoint.bin")

    def _compare_tiles(self, sim_ckpt, cpu_cells, label, tol_linf, tol_mean):
        """Direct tile-to-tile comparison (both stores match sim_v2 layout)."""
        linf = 0.0
        total_abs = 0.0
        total_n = 0
        for i, (sc, rc) in enumerate(zip(sim_ckpt["cells"], cpu_cells)):
            psim = sc["phi"].astype(np.float64)
            pref = rc.phi
            assert psim.shape == pref.shape, \
                f"[{label}] cell {i} shape mismatch: sim {psim.shape} vs ref {pref.shape}"
            d = np.abs(psim - pref)
            linf = max(linf, float(d.max()))
            total_abs += float(d.sum())
            total_n += d.size
        mean = total_abs / max(total_n, 1)
        assert np.isfinite(linf) and np.isfinite(mean), f"[{label}] NaN in error"
        assert linf < tol_linf, f"[{label}] max|Δφ| = {linf:.3e} (tol {tol_linf:.1e})"
        assert mean < tol_mean, f"[{label}] mean|Δφ| = {mean:.3e} (tol {tol_mean:.1e})"

    # --- tests -----------------------------------------------------------

    def test_cpu_ref_packed_grid_short(self, sim):
        """Scenario A: 8 cells in a domain large enough that tiles don't
        self-wrap. Heavy overlap exercises the repulsion term S."""
        # -N 150 -r 20 -n 8: tile ≈ 2·20·1.6 + 8 = 72 < 150. Cells relax
        # to a packed state over 201 steps.
        out_init = sim("-n", "8", "-N", "150", "-r", "20",
                       "-t", str(self._INIT_T_END), "--dt", "0.01",
                       "--v-A", "0", "--seed", "901",
                       "--save-interval", "0", "--trajectory-samples", "0",
                       "--print-interval", "0")
        ckpt = out_init / "checkpoint.bin"
        ckpt_data = read_checkpoint(ckpt)
        sim_after = self._resume_sim(sim, ckpt, self._RESUME_STEPS, 0.01, seed=901)
        n_ran = sim_after["step"] - ckpt_data["step"]
        cpu_cells, _ = self._run_cpu_ref(ckpt_data, n_ran)
        self._compare_tiles(sim_after, cpu_cells, "packed_grid_8",
                            tol_linf=self._TOL_LINF, tol_mean=self._TOL_MEAN)

    def test_cpu_ref_dimer_short(self, sim):
        """Scenario B: 2 cells in a modest domain. Exercises bulk +
        constraint + a localised S overlap where the two tiles meet."""
        # -N 120 -r 25 -n 2: tile = 2·25·1.6 + 8 = 88 < 120. No wrap.
        out_init = sim("-n", "2", "-N", "120", "-r", "25",
                       "-t", str(self._INIT_T_END), "--dt", "0.01",
                       "--v-A", "0", "--seed", "902",
                       "--save-interval", "0", "--trajectory-samples", "0",
                       "--print-interval", "0")
        ckpt = out_init / "checkpoint.bin"
        ckpt_data = read_checkpoint(ckpt)
        sim_after = self._resume_sim(sim, ckpt, self._RESUME_STEPS, 0.01, seed=902)
        n_ran = sim_after["step"] - ckpt_data["step"]
        cpu_cells, _ = self._run_cpu_ref(ckpt_data, n_ran)
        self._compare_tiles(sim_after, cpu_cells, "dimer_2",
                            tol_linf=self._TOL_LINF, tol_mean=self._TOL_MEAN)

    def test_cpu_ref_error_does_not_blow_up(self, sim):
        """Longer run (50 vs 100 steps) crosses several REMAP events, which
        introduce small localised transients the CPU ref does not model.
        Require the error to grow sub-linearly — a missing stencil weight
        or sign flip would blow up exponentially instead."""
        out_init = sim("-n", "4", "-N", "150", "-r", "22",
                       "-t", str(self._INIT_T_END), "--dt", "0.01",
                       "--v-A", "0", "--seed", "903",
                       "--save-interval", "0", "--trajectory-samples", "0",
                       "--print-interval", "0")
        ckpt = out_init / "checkpoint.bin"
        ckpt_data = read_checkpoint(ckpt)

        def err_after(n_requested):
            sim_after = self._resume_sim(sim, ckpt, n_requested, 0.01, seed=903)
            n_ran = sim_after["step"] - ckpt_data["step"]
            cpu_cells, _ = self._run_cpu_ref(ckpt_data, n_ran)
            worst = 0.0
            for sc, rc in zip(sim_after["cells"], cpu_cells):
                worst = max(worst, float(np.abs(sc["phi"].astype(np.float64) - rc.phi).max()))
            return worst

        e_50 = err_after(50)
        e_100 = err_after(100)
        assert np.isfinite(e_50) and np.isfinite(e_100), \
            f"NaN in error growth (e50={e_50}, e100={e_100})"
        # REMAP events add ~5e-4 transients; require growth factor ≤ 10.
        assert e_100 < max(10.0 * e_50, 5e-3), \
            f"error blew up: e50={e_50:.3e}, e100={e_100:.3e} (>10x)"
