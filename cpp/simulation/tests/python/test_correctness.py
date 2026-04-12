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

class TestResumePreservesPhysics:
    """Run with non-default physics, checkpoint, resume without flags.
    Assert all physics params match the checkpoint, not the binary defaults."""

    def test_resume_preserves_all_params(self, tmp_path):
        # Step 1: Run with non-default params
        out1 = run_sim(tmp_path / "run1",
                       "-n", "4", "-N", "300", "-r", "49",
                       "-t", "1", "--dt", "0.005",
                       "--gamma", "3.75", "--kappa", "20", "--mu", "0.5",
                       "--xi", "1000", "--tau", "5000", "--lambda", "10",
                       "--v-A", "0", "--seed", "42",
                       "--save-interval", "0", "--trajectory-samples", "0")
        chk1 = read_checkpoint(out1 / "checkpoint.bin")

        # Step 2: Resume with ONLY -t (new end time), no physics flags
        out2 = run_sim(tmp_path / "run2",
                       "-c", str(out1 / "checkpoint.bin"),
                       "-t", "2")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")

        # Assert all physics preserved from checkpoint
        for key in ["dt", "gamma", "kappa", "mu", "xi", "tau", "lambda"]:
            assert chk2["params"][key] == pytest.approx(chk1["params"][key], rel=1e-6), \
                f"{key}: expected {chk1['params'][key]}, got {chk2['params'][key]}"

    def test_resume_overrides_only_explicit(self, tmp_path):
        # Run with non-default gamma
        out1 = run_sim(tmp_path / "run1",
                       "-n", "4", "-N", "300", "-r", "49",
                       "-t", "1", "--dt", "0.01", "--gamma", "3.75",
                       "--kappa", "20", "--mu", "0.5",
                       "--v-A", "0", "--seed", "42",
                       "--save-interval", "0", "--trajectory-samples", "0")
        chk1 = read_checkpoint(out1 / "checkpoint.bin")

        # Resume overriding ONLY kappa
        out2 = run_sim(tmp_path / "run2",
                       "-c", str(out1 / "checkpoint.bin"),
                       "-t", "2", "--kappa", "15")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")

        # kappa should change, gamma should stay
        assert chk2["params"]["kappa"] == pytest.approx(15.0, rel=1e-6)
        assert chk2["params"]["gamma"] == pytest.approx(3.75, rel=1e-6)
        assert chk2["params"]["mu"] == pytest.approx(0.5, rel=1e-6)


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
    def test_gamma_roundtrip(self, tmp_path):
        # Run with gamma selector
        out1 = run_sim(tmp_path / "run1",
                       "-n", "8", "-N", "400", "-r", "49",
                       "-t", "1", "--dt", "0.01", "--v-A", "0.01", "--seed", "42",
                       "--gamma", "0.35:cell0",
                       "--save-interval", "0", "--trajectory-samples", "0")
        chk1 = read_checkpoint(out1 / "checkpoint.bin")
        assert "gamma" in chk1["per_cell"]
        gamma1 = chk1["per_cell"]["gamma"]
        assert gamma1[0] == pytest.approx(0.35, abs=0.01)
        assert all(g == pytest.approx(1.0, abs=0.01) for g in gamma1[1:])

        # Resume without gamma flag — should preserve checkpoint gamma
        out2 = run_sim(tmp_path / "run2",
                       "-c", str(out1 / "checkpoint.bin"),
                       "-t", "2")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        assert "gamma" in chk2["per_cell"]
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
