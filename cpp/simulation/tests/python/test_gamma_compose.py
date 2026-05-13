"""Tests for --gamma compose, gamma sidecar resume, and cell_analyze find-pair.

These were added when Phase 3A required two specific cells at controlled
separation to be soft. The mechanism is:

1. Equilibrate normally (one cell baseline).
2. `cell_analyze find-pair <ckpt> --distance D --format gamma-flags` returns
   `--gamma 0.35:nearest(x1,y1) --gamma 0.35:nearest(x2,y2)`.
3. Resume with those flags pasted in. --gamma now composes (was last-wins).
4. Once cells are soft, they stay soft via the GAMA sidecar — no need to
   keep re-passing --gamma on subsequent resumes.
"""
import math
import struct
import subprocess
import pytest
from conftest import CELL_SIM, read_checkpoint, run_sim


# Path to cell_analyze binary. Test skips if not built.
def _cell_analyze_binary():
    from pathlib import Path
    candidates = [
        Path(__file__).parents[4] / "rust" / "cell_analyze" / "target" / "release" / "cell_analyze.exe",
        Path(__file__).parents[4] / "rust" / "cell_analyze" / "target" / "release" / "cell_analyze",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    pytest.skip("cell_analyze binary not built")


class TestGammaCompose:
    """--gamma flags must accumulate (formerly last-wins)."""

    def test_two_gamma_flags_both_apply(self, tmp_path):
        """Two --gamma flags targeting different cells should both take effect."""
        out = run_sim(tmp_path, "-n", "4", "-N", "640", "-r", "49",
                      "-t", "0.5", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0",
                      "--gamma", "0.35:cell0",
                      "--gamma", "0.50:cell2")
        chk = read_checkpoint(out / "checkpoint.bin")
        gammas = chk["per_cell"].get("gamma")
        assert gammas is not None, "GAMA sidecar missing"
        assert len(gammas) == 4
        # cell 0 should be 0.35, cell 2 should be 0.50, others default 1.0.
        assert gammas[0] == pytest.approx(0.35, abs=1e-5)
        assert gammas[2] == pytest.approx(0.50, abs=1e-5)
        assert gammas[1] == pytest.approx(1.0, abs=1e-5)
        assert gammas[3] == pytest.approx(1.0, abs=1e-5)

    def test_scalar_then_subset_composes(self, tmp_path):
        """--gamma 0.5 --gamma 0.35:cell0 sets baseline 0.5 with cell 0 = 0.35."""
        out = run_sim(tmp_path, "-n", "4", "-N", "640", "-r", "49",
                      "-t", "0.5", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0",
                      "--gamma", "0.5",
                      "--gamma", "0.35:cell0")
        chk = read_checkpoint(out / "checkpoint.bin")
        gammas = chk["per_cell"]["gamma"]
        assert gammas[0] == pytest.approx(0.35, abs=1e-5)
        for i in range(1, 4):
            assert gammas[i] == pytest.approx(0.5, abs=1e-5)


class TestGammaSidecarResume:
    """GAMA sidecar must preserve per-cell γ across resume when --gamma is not passed."""

    def test_soft_cell_stays_soft_on_resume(self, tmp_path):
        """Equilibrate with cell 0 soft, resume without --gamma, cell 0 still soft."""
        sub1 = tmp_path / "sub1"; sub1.mkdir()
        out1 = run_sim(sub1, "-n", "4", "-N", "640", "-r", "49",
                       "-t", "1.0", "--seed", "42",
                       "--save-interval", "0", "--trajectory-samples", "0",
                       "--gamma", "0.35:cell0")
        chk1 = read_checkpoint(out1 / "checkpoint.bin")
        assert chk1["per_cell"]["gamma"][0] == pytest.approx(0.35, abs=1e-5)

        sub2 = tmp_path / "sub2"; sub2.mkdir()
        ckpt = out1 / "checkpoint.bin"
        # Resume WITHOUT --gamma — sidecar should preserve the per-cell value.
        out2 = run_sim(sub2, "-c", str(ckpt), "-t", "2.0", "--seed", "42",
                       "--save-interval", "0", "--trajectory-samples", "0")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        gammas = chk2["per_cell"]["gamma"]
        assert gammas[0] == pytest.approx(0.35, abs=1e-5), \
            f"cell 0 lost soft γ on resume: got {gammas[0]}"
        for i in range(1, 4):
            assert gammas[i] == pytest.approx(1.0, abs=1e-5)


class TestFindPair:
    """`cell_analyze find-pair` finds the cell pair closest to a target distance."""

    def test_find_pair_deterministic(self, tmp_path):
        """For a small system, find-pair returns the same pair every call."""
        binary = _cell_analyze_binary()
        # Generate a checkpoint with N=4 cells in a known layout.
        out = run_sim(tmp_path, "-n", "4", "-N", "640", "-r", "49",
                      "-t", "0.5", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        ckpt = out / "checkpoint.bin"
        # Find pair near distance 300 (within domain).
        r1 = subprocess.run([binary, "find-pair", str(ckpt), "--distance", "300"],
                            capture_output=True, text=True, timeout=30)
        r2 = subprocess.run([binary, "find-pair", str(ckpt), "--distance", "300"],
                            capture_output=True, text=True, timeout=30)
        assert r1.returncode == 0, f"find-pair failed: {r1.stderr}"
        assert r1.stdout == r2.stdout, "find-pair not deterministic"
        # Output should reference two distinct cell ids.
        assert "cell" in r1.stdout

    def test_find_pair_gamma_flags_format(self, tmp_path):
        """--format gamma-flags emits ready-to-paste --gamma arguments."""
        binary = _cell_analyze_binary()
        out = run_sim(tmp_path, "-n", "4", "-N", "640", "-r", "49",
                      "-t", "0.5", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        ckpt = out / "checkpoint.bin"
        r = subprocess.run([binary, "find-pair", str(ckpt),
                            "--distance", "300", "--format", "gamma-flags",
                            "--soft-gamma", "0.4"],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, f"find-pair failed: {r.stderr}"
        line = r.stdout.strip()
        assert line.startswith("--gamma 0.4:nearest("), f"bad output: {line!r}"
        # Should contain two --gamma flags.
        assert line.count("--gamma") == 2
        assert line.count(":nearest(") == 2
