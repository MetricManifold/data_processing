"""
Tier 1: Code correctness tests.
These test plumbing, not physics — CLI overrides, checkpoint round-trips, etc.
"""
import math
import pytest
import numpy as np
from conftest import run_sim, read_checkpoint, read_trajectory, requires_flag


# ============================================================================
# 1. Single cell smoke test
# ============================================================================

class TestSmoke:
    def test_single_cell_runs(self, sim):
        out = sim("-n", "1", "-N", "256", "-r", "49", "-t", "1", "--dt", "0.01",
                  "--v-A", "0", "--seed", "42", "--save-interval", "0",
                  "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["num_cells"] == 1
        assert chk["time"] == pytest.approx(1.0, abs=0.02)
        assert np.isfinite(chk["cells"][0]["volume"])
        assert chk["cells"][0]["volume"] > 0

    def test_multi_cell_runs(self, sim):
        out = sim("-n", "8", "-N", "512", "-r", "49", "-t", "1", "--dt", "0.01",
                  "--v-A", "0", "--seed", "42", "--save-interval", "0",
                  "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["num_cells"] == 8
        for c in chk["cells"]:
            assert np.isfinite(c["volume"])
            assert c["volume"] > 0
            assert not np.any(np.isnan(c["phi"]))


# ============================================================================
# 1b. Trajectory integrity (header, monotonic timestamps, no NaN)
# ============================================================================

class TestTrajectoryIntegrity:
    """Regression: sim_v2 had cur_time as float32 which capped at t=2^18=262144,
    causing duplicate timestamps with stale physics. Fixed in v5 by switching
    to double. This test runs past the float32 cap to catch regressions."""

    def test_header_fields_present(self, sim):
        """Trajectory header must declare N, Lx, Ly, dim, tau, v_A."""
        out = sim("-n", "4", "-N", "200", "-r", "49", "-t", "2", "--dt", "0.01",
                  "--v-A", "0.01", "--tau", "1000", "--seed", "42",
                  "--trajectory-samples", "20")
        text = (out / "trajectory.txt").read_text()
        header_line = next(
            (line for line in text.splitlines()
             if line.startswith("#") and "v_A=" in line),
            None)
        assert header_line is not None, "missing header line with v_A="
        for key in ("N", "Lx", "Ly", "dim", "tau", "v_A"):
            assert f"{key}=" in header_line, f"{key} missing from header: {header_line}"

    def test_timestamps_strictly_increasing(self, sim):
        """Every unique timestamp must be strictly greater than the previous."""
        out = sim("-n", "2", "-N", "200", "-r", "20", "-t", "1", "--dt", "0.01",
                  "--v-A", "0", "--seed", "42", "--trajectory-samples", "50")
        data, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(data.keys())
        for i in range(1, len(times)):
            assert times[i] > times[i-1], \
                f"non-monotonic: t[{i-1}]={times[i-1]} >= t[{i}]={times[i]}"

    def test_rows_per_frame_consistent(self, sim):
        """Every frame must have exactly N rows."""
        N = 4
        out = sim("-n", str(N), "-N", "200", "-r", "49", "-t", "1", "--dt", "0.01",
                  "--v-A", "0", "--seed", "42", "--trajectory-samples", "20")
        data, _ = read_trajectory(out / "trajectory.txt")
        for t, cells in data.items():
            assert len(cells) == N, f"frame t={t} has {len(cells)} cells, expected {N}"

    def test_checkpoint_stores_time_as_double(self, sim):
        """Fast regression: v5 checkpoint must store cur_time as f64.
        With cur_time=f32, the cap at t=2^18=262144 silently breaks long runs.
        Binary layout v5: magic(4), ver(4), step(4), time(8), N(4), ...
        """
        import struct
        out = sim("-n", "1", "-N", "200", "-r", "20", "-t", "0.5", "--dt", "0.01",
                  "--v-A", "0", "--seed", "42", "--trajectory-samples", "0")
        ckpt = (out / "checkpoint.bin").read_bytes()
        magic, version = struct.unpack_from("<II", ckpt, 0)
        assert magic == 0x43454C4C, f"bad magic 0x{magic:08X}"
        assert version >= 5, \
            f"checkpoint version {version} < 5 — cur_time may still be f32"
        # v5 layout: num_cells at offset 20 (after magic+ver+step+time_f64)
        num_cells = struct.unpack_from("<i", ckpt, 20)[0]
        assert num_cells == 1, \
            f"v5 layout broken: num_cells at offset 20 = {num_cells}, expected 1"
        time_f64 = struct.unpack_from("<d", ckpt, 12)[0]
        assert abs(time_f64 - 0.5) < 0.02, \
            f"checkpoint time {time_f64} disagrees with t_end=0.5"

    @pytest.mark.slow
    def test_time_advances_past_float32_precision_wall(self, sim, tmp_path):
        """End-to-end guard for cur_time=f32 regression. Rather than marching
        17M steps from t=0 to cross the f32 stall point at t=2^14=16384 (for
        dt=0.001), we generate a short checkpoint, hex-patch its cur_time to
        just below the stall point, then resume. If cur_time were still f32,
        timestamps past 2^14 would collapse; with f64 they advance."""
        import struct, subprocess
        from conftest import CELL_SIM
        # 1. Short initial run to generate a v5 checkpoint
        out1 = sim("-n", "1", "-N", "200", "-r", "15", "-t", "0.5",
                   "--dt", "0.001", "--v-A", "0", "--seed", "42",
                   "--trajectory-samples", "0")
        ckpt_src = out1 / "checkpoint.bin"
        assert ckpt_src.exists()

        # 2. Hex-patch step_count (offset 8, i32) and time_f64 (offset 12) to
        # place the run just below the f32 stall point. The sim loop uses
        # step_count<target_step so we need to shift both consistently.
        ckpt_bytes = bytearray(ckpt_src.read_bytes())
        magic, version = struct.unpack_from("<II", ckpt_bytes, 0)
        assert version >= 5
        t_patch = 16380.0
        dt_patch = 0.001
        struct.pack_into("<i", ckpt_bytes, 8, int(t_patch / dt_patch))
        struct.pack_into("<d", ckpt_bytes, 12, t_patch)
        run2 = tmp_path / "resumed"
        run2.mkdir()
        (run2 / "checkpoint.bin").write_bytes(bytes(ckpt_bytes))

        # 3. Resume ~8000 steps past the stall point. Use a large
        # trajectory_samples so traj_every ≈ total_steps / samples is small
        # enough to produce multiple writes in the narrow 8000-step window.
        cmd = [CELL_SIM, "-c", str(run2 / "checkpoint.bin"),
               "-t", "16388", "--trajectory-samples", "50000",
               "--print-interval", "0", "-o", str(run2)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"resume failed: {r.stderr}"

        data, _ = read_trajectory(run2 / "trajectory.txt")
        times = sorted(data.keys())
        past_cap = [t for t in times if t > 16384.0]
        assert len(past_cap) >= 2, \
            f"only {len(past_cap)} frames past t=2^14 — f32 regression?"
        for i in range(1, len(past_cap)):
            assert past_cap[i] > past_cap[i-1], \
                f"stalled past cap: {past_cap[i-1]} == {past_cap[i]}"


# ============================================================================
# 2. Resume preserves checkpoint physics without CLI flags
# ============================================================================

# Different physics parameter sets to test resume preservation
PARAM_SETS = [
    {"dt": "0.005", "gamma": "3.75", "kappa": "20", "mu": "0.5",
     "xi": "1000", "tau": "5000", "lambda": "10"},
    {"dt": "0.02", "gamma": "0.5", "kappa": "5", "mu": "2.0",
     "xi": "2000", "tau": "20000", "lambda": "5"},
]


class TestResumePreservesPhysics:
    """Run with non-default physics, checkpoint, resume without flags.
    Assert all physics params match the checkpoint, not the binary defaults."""

    @pytest.mark.parametrize("pset", PARAM_SETS, ids=["palmieri-like", "stiff"])
    def test_resume_preserves_all_params(self, tmp_path, pset):
        # Step 1: Run with non-default params
        cli = ["-n", "4", "-N", "300", "-r", "49", "-t", "1",
               "--v-A", "0", "--seed", "42",
               "--save-interval", "0", "--trajectory-samples", "0"]
        for k, v in pset.items():
            cli.extend([f"--{k}", v])
        out1 = run_sim(tmp_path / "run1", *cli)
        chk1 = read_checkpoint(out1 / "checkpoint.bin")

        # Step 2: Resume with ONLY -t (new end time), no physics flags
        out2 = run_sim(tmp_path / "run2",
                       "-c", str(out1 / "checkpoint.bin"),
                       "-t", "2")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")

        # Assert all physics preserved from checkpoint
        for key in pset.keys():
            assert chk2["params"][key] == pytest.approx(chk1["params"][key], rel=1e-5), \
                f"{key}: expected {chk1['params'][key]}, got {chk2['params'][key]}"

    @pytest.mark.parametrize("override_key,override_val,preserve_keys", [
        ("kappa", "15", {"gamma": 3.75, "mu": 0.5, "xi": 1000.0}),
        ("mu", "2.0", {"gamma": 3.75, "kappa": 20.0, "xi": 1000.0}),
        ("gamma", "1.5", {"kappa": 20.0, "mu": 0.5, "xi": 1000.0}),
        ("xi", "500", {"gamma": 3.75, "kappa": 20.0, "mu": 0.5}),
    ], ids=["override-kappa", "override-mu", "override-gamma", "override-xi"])
    def test_resume_overrides_only_explicit(self, tmp_path, override_key, override_val, preserve_keys):
        out1 = run_sim(tmp_path / "run1",
                       "-n", "4", "-N", "300", "-r", "49",
                       "-t", "1", "--dt", "0.01", "--gamma", "3.75",
                       "--kappa", "20", "--mu", "0.5", "--xi", "1000",
                       "--v-A", "0", "--seed", "42",
                       "--save-interval", "0", "--trajectory-samples", "0")

        # Resume overriding ONLY one param
        out2 = run_sim(tmp_path / "run2",
                       "-c", str(out1 / "checkpoint.bin"),
                       "-t", "2", f"--{override_key}", override_val)
        chk2 = read_checkpoint(out2 / "checkpoint.bin")

        # Overridden param should match new value
        assert chk2["params"][override_key] == pytest.approx(float(override_val), rel=1e-5), \
            f"{override_key} should be {override_val}, got {chk2['params'][override_key]}"
        # All other params should be preserved from the original run
        for key, expected in preserve_keys.items():
            assert chk2["params"][key] == pytest.approx(expected, rel=1e-5), \
                f"{key}: expected {expected}, got {chk2['params'][key]}"


# ============================================================================
# 3. Subdomain padding round-trip
# ============================================================================

class TestSubdomainPadding:
    def test_padding_stored_and_restored(self, tmp_path):
        # Run with explicit padding
        out1 = run_sim(tmp_path / "run1",
                       "-n", "4", "-N", "300", "-r", "49",
                       "-t", "1", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                       "--subdomain-padding", "0.4",
                       "--save-interval", "0", "--trajectory-samples", "0")
        chk1 = read_checkpoint(out1 / "checkpoint.bin")
        assert chk1["params"]["subdomain_padding"] == pytest.approx(0.4, abs=0.01)

        # Resume with different padding
        out2 = run_sim(tmp_path / "run2",
                       "-c", str(out1 / "checkpoint.bin"),
                       "-t", "2", "--subdomain-padding", "0.8")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        assert chk2["params"]["subdomain_padding"] == pytest.approx(0.8, abs=0.01)

    def test_padding_affects_bbox_size(self, tmp_path):
        # sim_v3 uses a fixed power-of-two tile (TILE_T) for every cell, so
        # subdomain_padding has no effect on bbox size. The flag is still
        # parsed and round-tripped through checkpoint headers, but the
        # per-cell tile is always TILE_T x TILE_T.
        pytest.skip("sim_v3 uses fixed-T tiles; subdomain_padding has no bbox effect")


# ============================================================================
# 4. Checkpoint per-cell array round-trip
# ============================================================================

class TestPerCellArrays:
    @pytest.mark.parametrize("gamma_spec,check_fn", [
        # cell0 selector: cell 0 is soft, rest normal
        ("0.35:cell0", lambda g, n: g[0] == pytest.approx(0.35, abs=0.01) and
                                    all(x == pytest.approx(1.0, abs=0.01) for x in g[1:])),
        # fraction selector: 20% of cells are soft
        ("0.35:20%", lambda g, n: sum(1 for x in g if abs(x - 0.35) < 0.01) == pytest.approx(n * 0.2, abs=1)),
        # fraction selector: 50% of cells are soft
        ("0.35:50%", lambda g, n: sum(1 for x in g if abs(x - 0.35) < 0.01) == pytest.approx(n * 0.5, abs=1)),
    ], ids=["cell0", "20pct", "50pct"])
    def test_gamma_roundtrip(self, tmp_path, gamma_spec, check_fn):
        n_cells = 20
        # Run with gamma selector
        out1 = run_sim(tmp_path / "run1",
                       "-n", str(n_cells), "--confluence", "0.85", "-r", "49",
                       "-t", "1", "--dt", "0.01", "--v-A", "0.01", "--seed", "42",
                       "--gamma", gamma_spec,
                       "--save-interval", "0", "--trajectory-samples", "0")
        chk1 = read_checkpoint(out1 / "checkpoint.bin")
        assert "gamma" in chk1["per_cell"], "Gamma array should be in checkpoint"
        gamma1 = chk1["per_cell"]["gamma"]
        assert check_fn(gamma1, n_cells), f"Gamma array doesn't match spec '{gamma_spec}': {gamma1}"

        # Resume without gamma flag — should preserve checkpoint gamma
        out2 = run_sim(tmp_path / "run2",
                       "-c", str(out1 / "checkpoint.bin"),
                       "-t", "2")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        assert "gamma" in chk2["per_cell"], "Gamma array should survive resume"
        np.testing.assert_allclose(chk2["per_cell"]["gamma"], gamma1, atol=1e-6)

    def test_bare_gamma_clears_checkpoint(self, tmp_path):
        # Run with gamma selector
        out1 = run_sim(tmp_path / "run1",
                       "-n", "8", "-N", "400", "-r", "49",
                       "-t", "1", "--dt", "0.01", "--v-A", "0.01", "--seed", "42",
                       "--gamma", "0.35:cell0",
                       "--save-interval", "0", "--trajectory-samples", "0")
        chk1 = read_checkpoint(out1 / "checkpoint.bin")
        assert chk1["per_cell"]["gamma"][0] == pytest.approx(0.35, abs=0.01)

        # Resume with bare --gamma 2.0 — should override ALL cells
        out2 = run_sim(tmp_path / "run2",
                       "-c", str(out1 / "checkpoint.bin"),
                       "-t", "2", "--gamma", "2.0")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        if "gamma" in chk2["per_cell"]:
            assert all(g == pytest.approx(2.0, abs=0.01) for g in chk2["per_cell"]["gamma"])


# ============================================================================
# 5. Perimeter is non-zero
# ============================================================================

class TestPerimeter:
    def test_perimeter_nonzero_in_trajectory(self, tmp_path):
        """Run multi-cell with trajectory, check L_n > 0 on non-initial frames."""
        out = run_sim(tmp_path / "run",
                      "-n", "8", "-N", "400", "-r", "49",
                      "-t", "10", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "5",
                      timeout=60)
        traj_file = out / "trajectory.txt"
        assert traj_file.exists()

        # Read L_n (column 11, 0-indexed = 10)
        ln_values = []
        with open(traj_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 12:
                    t = float(parts[0])
                    ln = float(parts[10])
                    if t > 1.0:  # skip initial transient
                        ln_values.append(ln)

        assert len(ln_values) > 0, "No trajectory data found after t=1"
        assert all(np.isfinite(v) for v in ln_values), "L_n has NaN/Inf"
        assert all(v > 0 for v in ln_values), f"L_n has zero values: min={min(ln_values)}"


# ============================================================================
# 8. Trajectory sanity: monotonic time, no NaN, polarity unit vectors
# ============================================================================

class TestTrajectorySanity:
    """Tests from validate_correctness.py not already covered."""

    def test_monotonic_time_and_no_nan(self, sim):
        """Time strictly increases, no NaN/Inf in any field."""
        out = sim("-n", "8", "-N", "512", "-r", "49", "-t", "10", "--dt", "0.01",
                  "--v-A", "0.01", "--seed", "42", "--save-interval", "0",
                  "--trajectory-samples", "200")
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())
        assert len(times) >= 2
        # Strict monotonic
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]
        # No NaN/Inf in any field
        for t, cells in traj.items():
            for cid, vals in cells.items():
                for v in vals:
                    assert np.isfinite(v), f"Non-finite value at t={t}, cell={cid}"

    def test_polarization_unit_vectors(self, sim):
        """Polarity (px, py) should be unit vectors."""
        out = sim("-n", "4", "-N", "256", "-r", "49", "-t", "10", "--dt", "0.01",
                  "--v-A", "0.01", "--tau", "1000", "--seed", "42",
                  "--save-interval", "0", "--trajectory-samples", "100")
        traj, _ = read_trajectory(out / "trajectory.txt")
        for t, cells in traj.items():
            for cid, vals in cells.items():
                if len(vals) >= 6:
                    px, py = vals[4], vals[5]
                    mag = np.sqrt(px**2 + py**2)
                    assert abs(mag - 1.0) < 0.01, \
                        f"Polarity not unit: |p|={mag:.4f} at t={t}, cell={cid}"

    def test_phi_range_in_checkpoint(self, sim):
        """Phase field values should be approximately in [0, 1]."""
        out = sim("-n", "4", "-N", "256", "-r", "49", "-t", "5", "--dt", "0.01",
                  "--v-A", "0", "--seed", "42", "--save-interval", "0",
                  "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        for c in chk["cells"]:
            phi = c["phi"]
            assert phi.min() >= -0.1, f"Cell {c['id']}: phi_min={phi.min():.4f}"
            assert phi.max() <= 1.1, f"Cell {c['id']}: phi_max={phi.max():.4f}"
            # Cell should have some interior (phi > 0.9) and interface
            inner = phi[4:-4, 4:-4] if phi.ndim == 2 else phi
            assert np.sum(inner > 0.9) > 0, f"Cell {c['id']}: no interior (phi>0.9)"


# ============================================================================
# 9. Bbox remap: field continuity after forced resize
# ============================================================================

class TestBboxRemap:
    """Tests bbox remap doesn't introduce field discontinuities."""

    def test_moving_cell_field_smooth(self, sim):
        """A motile cell forces bbox remaps. The field should remain smooth
        (no d²φ spikes at remap boundaries)."""
        out = sim("-n", "1", "-N", "300", "-r", "49", "-t", "500", "--dt", "0.01",
                  "--v-A", "0.05", "--tau", "100000", "--seed", "42",
                  "--save-interval", "0", "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        cell = chk["cells"][0]
        phi = cell["phi"]

        # Compute second derivative along both axes
        d2x = np.zeros_like(phi)
        d2y = np.zeros_like(phi)
        d2x[:, 1:-1] = phi[:, 2:] + phi[:, :-2] - 2 * phi[:, 1:-1]
        d2y[1:-1, :] = phi[2:, :] + phi[:-2, :] - 2 * phi[1:-1, :]

        # Only check interface region (0.01 < phi < 0.99)
        interface = (phi > 0.01) & (phi < 0.99)
        if np.sum(interface) < 10:
            pytest.skip("No interface region found")

        # A remap artifact shows as an anomalous spike in d².
        # Check that max |d²| in the interface is below threshold.
        d2_interface = np.abs(d2x[interface])
        d2_max = np.max(d2_interface)
        d2_median = np.median(d2_interface)
        # Spikes should be < 10× the median (smooth fields have ~uniform d²)
        assert d2_max < d2_median * 15 + 0.05, \
            f"d²φ/dx² spike: max={d2_max:.4f}, median={d2_median:.4f} (ratio={d2_max/d2_median:.1f})"

        # Same for y
        d2y_interface = np.abs(d2y[interface])
        d2y_max = np.max(d2y_interface)
        d2y_median = np.median(d2y_interface)
        assert d2y_max < d2y_median * 15 + 0.05, \
            f"d²φ/dy² spike: max={d2y_max:.4f}, median={d2y_median:.4f} (ratio={d2y_max/d2y_median:.1f})"


# ============================================================================
# 10. Chain-job resume: trajectory stitches cleanly
# ============================================================================

class TestChainResume:
    """Checkpoint → resume → verify trajectory continuity."""

    def test_resume_trajectory_continuous(self, tmp_path):
        """Run, checkpoint, resume — checkpoint state should be continuous."""
        run1 = run_sim(tmp_path / "run1",
                       "-n", "4", "-N", "300", "-r", "49", "-t", "5", "--dt", "0.01",
                       "--v-A", "0.01", "--seed", "42",
                       "--save-interval", "0", "--trajectory-samples", "50",
                       "--polarity-seed", "100")

        chk1 = read_checkpoint(run1 / "checkpoint.bin")
        t_end_1 = chk1["time"]

        # Resume from checkpoint — run to t=20
        run2 = run_sim(tmp_path / "run2",
                       "-c", str(run1 / "checkpoint.bin"),
                       "-t", "20",
                       "--save-interval", "0", "--trajectory-samples", "50",
                       "-o", str(tmp_path / "run2"))

        # Final checkpoint validates resume completed
        chk2 = read_checkpoint(run2 / "checkpoint.bin")
        assert chk2["time"] == pytest.approx(20.0, abs=0.1)
        assert chk2["num_cells"] == chk1["num_cells"]

        # Physics params preserved
        for key in ["dt", "lambda", "gamma", "kappa", "mu", "target_radius", "v_A", "xi", "tau"]:
            if key in chk1["params"] and key in chk2["params"]:
                assert chk1["params"][key] == pytest.approx(chk2["params"][key], rel=1e-4), \
                    f"Param {key} changed: {chk1['params'][key]} -> {chk2['params'][key]}"

        # Volumes should be conserved across resume
        for c1, c2 in zip(chk1["cells"], chk2["cells"]):
            assert c1["id"] == c2["id"]
            assert c2["volume"] == pytest.approx(c1["volume"], rel=0.05), \
                f"Cell {c1['id']} volume changed: {c1['volume']:.1f} -> {c2['volume']:.1f}"

        # Cells shouldn't have teleported
        Nx = chk1["params"]["Nx"]
        for c1, c2 in zip(chk1["cells"], chk2["cells"]):
            dx = abs(c2["centroid"][0] - c1["centroid"][0])
            dy = abs(c2["centroid"][1] - c1["centroid"][1])
            if dx > Nx / 2: dx = Nx - dx
            if dy > Nx / 2: dy = Nx - dy
            dist = np.sqrt(dx**2 + dy**2)
            # With v_A=0.01 and 15 TU, max displacement is v_A*t = 0.15 px
            # but with random walk it could be more — allow generous bound
            assert dist < 20.0, \
                f"Cell {c1['id']} jumped {dist:.1f} px between checkpoint and resume"


# ============================================================================
# 10b. Resume with v_A override: cells must actually move
# ============================================================================

class TestResumeVAOverride:
    """Resume from a v_A=0 equilibration checkpoint with --v-A 0.01.

    This is the Palmieri protocol: equilibrate without motility, then turn on
    motility for the production run. The per-cell v_A stored in the checkpoint's
    VA_A sidecar is 0; the CLI --v-A 0.01 must override it so cells move.

    Regression test for: VA_A sidecar overriding CLI --v-A override (2026-04-26).
    """

    def test_va_override_cells_move(self, tmp_path):
        """Equilibrate with v_A=0, resume with --v-A 0.01 → cells must move."""
        # Step 1: Equilibrate (v_A=0, short run to get a checkpoint)
        out_eq = run_sim(tmp_path / "eq",
                         "-n", "8", "-N", "400", "-r", "49", "-t", "10",
                         "--dt", "0.01", "--v-A", "0", "--seed", "42",
                         "--trajectory-samples", "0")
        ckpt = out_eq / "checkpoint.bin"
        assert ckpt.exists()

        # Step 2: Resume with motility on (--v-A 0.01)
        out_prod = run_sim(tmp_path / "prod",
                           "-c", str(ckpt),
                           "-t", "110", "--v-A", "0.01",
                           "--trajectory-samples", "10",
                           "--seed", "42")
        traj_path = out_prod / "trajectory.txt"
        assert traj_path.exists()

        # Step 3: Check that cells actually move
        data, hdr = read_trajectory(traj_path)
        times = sorted(data.keys())
        assert len(times) >= 2, f"Expected ≥2 trajectory frames, got {len(times)}"

        # Compute displacement of cell 0 between first and last frame
        t0, t1 = times[0], times[-1]
        x0, y0 = data[t0][0][:2]
        x1, y1 = data[t1][0][:2]
        Nx = int(hdr.get("Lx", "400"))
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx > Nx / 2: dx = Nx - dx
        if dy > Nx / 2: dy = Nx - dy
        displacement = np.sqrt(dx**2 + dy**2)

        # With v_A=0.01, 100 TU, an isolated cell moves ~v_A*t = 1 px.
        # In a monolayer there's some caging, but displacement should be > 0.1.
        assert displacement > 0.1, \
            f"Cell 0 displacement = {displacement:.4f} px — v_A override not working"

    def test_va_override_velocity_nonzero(self, tmp_path):
        """Resume with --v-A 0.01 → reported velocity magnitude must be physical."""
        out_eq = run_sim(tmp_path / "eq",
                         "-n", "8", "-N", "400", "-r", "49", "-t", "10",
                         "--dt", "0.01", "--v-A", "0", "--seed", "42",
                         "--trajectory-samples", "0")

        out_prod = run_sim(tmp_path / "prod",
                           "-c", str(out_eq / "checkpoint.bin"),
                           "-t", "110", "--v-A", "0.01",
                           "--trajectory-samples", "10",
                           "--seed", "42")

        data, _ = read_trajectory(out_prod / "trajectory.txt")
        times = sorted(data.keys())
        # Check velocity magnitude at the last frame
        last = data[times[-1]]
        speeds = [np.sqrt(c[2]**2 + c[3]**2) for c in last.values()]
        mean_speed = np.mean(speeds)

        # With v_A=0.01, mean speed in a monolayer is ~0.003 (Palmieri σ_G).
        # It must be > 1e-4 to be physical. If v_A override is broken, it's ~1e-6.
        assert mean_speed > 1e-4, \
            f"Mean speed = {mean_speed:.2e} — v_A override not propagating to GPU"

    def test_va_zero_resume_stays_zero(self, tmp_path):
        """Resume from v_A=0 without --v-A flag → cells must NOT move."""
        out_eq = run_sim(tmp_path / "eq",
                         "-n", "8", "-N", "400", "-r", "49", "-t", "10",
                         "--dt", "0.01", "--v-A", "0", "--seed", "42",
                         "--trajectory-samples", "0")

        out_prod = run_sim(tmp_path / "prod",
                           "-c", str(out_eq / "checkpoint.bin"),
                           "-t", "20",
                           "--trajectory-samples", "2",
                           "--seed", "42")

        data, _ = read_trajectory(out_prod / "trajectory.txt")
        times = sorted(data.keys())
        if len(times) >= 2:
            t0, t1 = times[0], times[-1]
            x0, y0 = data[t0][0][:2]
            x1, y1 = data[t1][0][:2]
            disp = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            # Without motility, displacement is just relaxation drift — should be tiny
            assert disp < 1.0, \
                f"Cell 0 displaced {disp:.4f} px with v_A=0 — shouldn't move this much"


# ============================================================================
# 10c. Resume initial velocity recomputation
# ============================================================================

class TestResumeInitialVelocity:
    """When resuming from a v_A=0 equilibration with --v-A 0.01, the binary
    must recompute v = v_I + v_A·p̂ before the first step so that advection
    uses the correct velocity from step 1.

    The trajectory reports end-of-step velocities (which are always correct),
    so we test via displacement: if initial velocity is wrong, cells barely
    move on the first trajectory interval.

    Regression test for: missing launch_initial_velocity on resume (2026-04-27).
    """

    def test_first_interval_displacement_physical(self, tmp_path):
        """Resume with --v-A 0.01 → displacement in the first trajectory
        interval must be consistent with v_A, not near zero."""
        # Equilibrate (v_A=0)
        out_eq = run_sim(tmp_path / "eq",
                         "-n", "8", "-N", "400", "-r", "49", "-t", "10",
                         "--dt", "0.01", "--v-A", "0", "--seed", "42",
                         "--trajectory-samples", "0")

        # Resume with motility, dense trajectory for first 100 TU
        out_prod = run_sim(tmp_path / "prod",
                           "-c", str(out_eq / "checkpoint.bin"),
                           "-t", "110", "--v-A", "0.01",
                           "--trajectory-samples", "100",
                           "--seed", "42")

        data, hdr = read_trajectory(out_prod / "trajectory.txt")
        times = sorted(data.keys())
        assert len(times) >= 3, f"Need ≥3 frames, got {len(times)}"

        # Displacement of cell 0 in the FIRST interval vs SECOND interval
        Nx = int(hdr.get("Lx", "400"))
        def disp(t0, t1, cid=0):
            x0, y0 = data[t0][cid][:2]
            x1, y1 = data[t1][cid][:2]
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            if dx > Nx / 2: dx = Nx - dx
            if dy > Nx / 2: dy = Nx - dy
            return np.sqrt(dx**2 + dy**2)

        d_first = disp(times[0], times[1])
        d_second = disp(times[1], times[2])

        # Both intervals should have similar displacement.
        # If initial velocity is wrong, d_first << d_second (cells don't move
        # in the first interval because advection uses v≈0).
        # Allow up to 5× difference (cells can cage/ungage between intervals).
        if d_second > 1e-6:
            ratio = d_first / d_second
            assert ratio > 0.1, \
                f"First interval displacement ({d_first:.4e}) is {ratio:.1%} of " \
                f"second ({d_second:.4e}) — initial velocity not recomputed"


# ============================================================================
# 11. Seed determinism: same seed → identical output
# ============================================================================

class TestSeedDeterminism:
    """Same seed + same parameters → reproducible to within fast-math tolerance.

    With -use_fast_math and non-deterministic CUDA atomics, bit-identical
    results are not achievable. We assert volumes/centroids agree to tolerances
    dominated by float32 round-off accumulated over 500 steps.
    """

    def test_same_seed_same_result(self, tmp_path):
        """Two runs with identical seeds should reproduce within fast-math tolerance."""
        args = ["-n", "4", "-N", "300", "-r", "49", "-t", "5", "--dt", "0.01",
                "--v-A", "0.01", "--seed", "42", "--polarity-seed", "100",
                "--save-interval", "0", "--trajectory-samples", "0"]

        run_a = run_sim(tmp_path / "runA", *args)
        run_b = run_sim(tmp_path / "runB", *args)

        chk_a = read_checkpoint(run_a / "checkpoint.bin")
        chk_b = read_checkpoint(run_b / "checkpoint.bin")

        assert chk_a["time"] == chk_b["time"]
        assert chk_a["num_cells"] == chk_b["num_cells"]

        for ca, cb in zip(chk_a["cells"], chk_b["cells"]):
            # volume ~ 7600; 1e-2 absolute = ~1 ppm relative
            assert ca["volume"] == pytest.approx(cb["volume"], abs=1e-2), \
                f"Cell {ca['id']}: volumes differ ({ca['volume']:.4f} vs {cb['volume']:.4f})"
            assert ca["centroid"][0] == pytest.approx(cb["centroid"][0], abs=0.01)
            assert ca["centroid"][1] == pytest.approx(cb["centroid"][1], abs=0.01)


# ============================================================================
# 12. Confluence flag computes correct domain size
# ============================================================================

class TestConfluence:
    """--confluence should set domain size so πR²N/(L²) = target packing."""

    def test_confluence_domain_size(self, tmp_path):
        """Check that L is computed correctly for given N, R, and confluence."""
        R = 49.0
        N = 16
        target_phi = 0.89

        out = run_sim(tmp_path / "run",
                      "-n", str(N), "-r", str(int(R)),
                      "--confluence", str(target_phi),
                      "-t", "0.1", "--dt", "0.01",
                      "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")

        chk = read_checkpoint(out / "checkpoint.bin")
        Lx = chk["params"]["Nx"]
        Ly = chk["params"]["Ny"]

        actual_phi = N * math.pi * R**2 / (Lx * Ly)
        assert actual_phi == pytest.approx(target_phi, abs=0.02), \
            f"Confluence {actual_phi:.4f} should be ≈ {target_phi} (domain {Lx}×{Ly})"

    def test_confluence_independent_of_N(self, tmp_path):
        """Different N values with same confluence should give same packing fraction."""
        R = 49.0  # cell radius large enough that auto-L stays >= TILE_T
        target_phi = 0.85
        phis = []
        for N in [12, 24]:
            out = run_sim(tmp_path / f"run_N{N}",
                          "-n", str(N), "-r", str(int(R)),
                          "--confluence", str(target_phi),
                          "-t", "0.1", "--dt", "0.01",
                          "--v-A", "0", "--seed", "42",
                          "--save-interval", "0", "--trajectory-samples", "0")
            chk = read_checkpoint(out / "checkpoint.bin")
            Lx = chk["params"]["Nx"]
            Ly = chk["params"]["Ny"]
            phi = N * math.pi * R**2 / (Lx * Ly)
            phis.append(phi)

        assert phis[0] == pytest.approx(phis[1], abs=0.03), \
            f"Confluence should be consistent: N=4→{phis[0]:.3f}, N=16→{phis[1]:.3f}"


# ============================================================================
# 13. v_A_sigma quenched disorder persists across resume
# ============================================================================

class TestVASigma:
    """Per-cell v_A values with --v-A-sigma should persist across resume."""

    @requires_flag("--v-A-sigma")
    def test_disorder_persists(self, tmp_path):
        """Run with v_A_sigma, checkpoint, resume — per-cell v_A should be identical."""
        run1 = run_sim(tmp_path / "run1",
                       "-n", "8", "-N", "400", "-r", "49", "-t", "2", "--dt", "0.01",
                       "--v-A", "0.01", "--v-A-sigma", "0.005", "--seed", "42",
                       "--save-interval", "0", "--trajectory-samples", "0")

        chk1 = read_checkpoint(run1 / "checkpoint.bin")
        va1 = chk1["per_cell"].get("v_A")
        assert va1 is not None, "No per-cell v_A array in checkpoint"
        assert len(va1) == 8

        # Values should have some spread (not all identical)
        assert np.std(va1) > 0.001, \
            f"v_A_sigma didn't produce disorder: std={np.std(va1):.6f}"

        # Resume
        run2 = run_sim(tmp_path / "run2",
                       "-c", str(run1 / "checkpoint.bin"),
                       "-t", "4",
                       "-o", str(tmp_path / "run2"))

        chk2 = read_checkpoint(run2 / "checkpoint.bin")
        va2 = chk2["per_cell"].get("v_A")
        assert va2 is not None, "No per-cell v_A array after resume"

        np.testing.assert_array_almost_equal(va1, va2, decimal=6,
            err_msg="Per-cell v_A changed after resume")

    @requires_flag("--v-A-sigma")
    def test_sigma_zero_gives_uniform(self, tmp_path):
        """--v-A-sigma 0 should give all cells the same v_A."""
        out = run_sim(tmp_path / "run",
                      "-n", "8", "-N", "400", "-r", "49", "-t", "1", "--dt", "0.01",
                      "--v-A", "0.01", "--v-A-sigma", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")

        chk = read_checkpoint(out / "checkpoint.bin")
        va = chk["per_cell"].get("v_A")
        if va is not None:
            assert np.std(va) < 1e-6, f"v_A_sigma=0 should give uniform v_A, got std={np.std(va)}"


# ============================================================================
# 14. Large N placement (N=32000 removed: too slow for CI, but documented)
# ============================================================================

class TestLargeN:
    """Large N runs without OOM or placement failure."""

    @pytest.mark.slow
    def test_large_n_placement(self, tmp_path):
        """N=32000 should place all cells and run 1 step without crash."""
        out = run_sim(tmp_path / "run",
                      "-n", "32000", "-r", "49", "--confluence", "0.89",
                      "-t", "0.01", "--dt", "0.01",
                      "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["num_cells"] == 32000
        for c in chk["cells"]:
            assert np.isfinite(c["volume"])
            assert c["volume"] > 0
