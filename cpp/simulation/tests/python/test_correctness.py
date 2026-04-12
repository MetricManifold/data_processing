"""
Tier 1: Code correctness tests.
These test plumbing, not physics — CLI overrides, checkpoint round-trips, etc.
"""
import pytest
import numpy as np
from conftest import run_sim, read_checkpoint


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
        # Small padding
        out_small = run_sim(tmp_path / "small",
                            "-n", "4", "-N", "300", "-r", "49",
                            "-t", "10", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                            "--subdomain-padding", "0.3",
                            "--save-interval", "0", "--trajectory-samples", "0")
        # Large padding
        out_large = run_sim(tmp_path / "large",
                            "-n", "4", "-N", "300", "-r", "49",
                            "-t", "10", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                            "--subdomain-padding", "1.5",
                            "--save-interval", "0", "--trajectory-samples", "0")
        chk_s = read_checkpoint(out_small / "checkpoint.bin")
        chk_l = read_checkpoint(out_large / "checkpoint.bin")

        avg_w_small = np.mean([c["bbox_w"] for c in chk_s["cells"]])
        avg_w_large = np.mean([c["bbox_w"] for c in chk_l["cells"]])
        assert avg_w_large > avg_w_small * 1.2, \
            f"Large padding bbox ({avg_w_large:.0f}) should be >20% bigger than small ({avg_w_small:.0f})"


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
