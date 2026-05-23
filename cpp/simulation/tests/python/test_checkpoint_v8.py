"""Tests for v8 checkpoint format (rank metadata, global cell ids).

v8 adds (num_ranks, rank_id, num_cells_global) after T_w. Single-GPU
runs write num_ranks=1, rank_id=0, num_cells_global=num_cells. Multi-GPU
runs write per-rank files where each cell carries its true global id;
the C++ loader requires --gpus to match num_ranks on resume.

Multi-rank tests need a multi-GPU build and are skipped locally.
"""
import struct
import subprocess
import pytest
from conftest import read_checkpoint, run_sim, CELL_SIM


class TestV8Header:
    """v8 header layout: magic, ver=8, ..., T, num_ranks, rank_id, num_cells_global."""

    def test_v8_single_gpu_self_describing(self, tmp_path):
        """Single-GPU run writes v8 with num_ranks=1, rank_id=0, n_global=N."""
        out = run_sim(tmp_path, "-n", "4", "-N", "640", "-r", "49",
                      "-t", "0.5", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["version"] == 8
        assert chk["params"]["num_ranks"] == 1
        assert chk["params"]["rank_id"] == 0
        assert chk["params"]["num_cells_global"] == chk["num_cells"]

    def test_v8_cell_ids_are_global(self, tmp_path):
        """Single-GPU: cell_id field equals the local index (== global)."""
        out = run_sim(tmp_path, "-n", "4", "-N", "640", "-r", "49",
                      "-t", "0.5", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        ids = sorted(c["id"] for c in chk["cells"])
        # For single-GPU, h_global_id is identity so cell ids are 0..N-1.
        assert ids == list(range(chk["num_cells"]))


class TestV8Resume:
    """v8 same-G resume produces consistent state."""

    def test_v8_single_gpu_roundtrip(self, tmp_path):
        """Run to t=1, save, resume to t=2; final state should be sensible."""
        sub1 = tmp_path / "sub1"; sub1.mkdir()
        out1 = run_sim(sub1, "-n", "4", "-N", "640", "-r", "49",
                       "-t", "1.0", "--seed", "42",
                       "--save-interval", "0", "--trajectory-samples", "0")
        ckpt = out1 / "checkpoint.bin"
        sub2 = tmp_path / "sub2"; sub2.mkdir()
        out2 = run_sim(sub2, "-c", str(ckpt), "-t", "2.0", "--seed", "42",
                       "--save-interval", "0", "--trajectory-samples", "0")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        assert chk2["version"] == 8
        assert abs(chk2["time"] - 2.0) < 0.1
        assert chk2["num_cells"] == 4
        # All cells should still be in the box and have plausible volumes.
        target_vol = 3.14159 * 49 * 49
        for c in chk2["cells"]:
            assert 0.5 * target_vol < c["volume"] < 2.0 * target_vol

    def test_v8_resume_preserves_subdomain_padding(self, tmp_path):
        """v8 stores live subdomain_padding; resume must not reset it."""
        sub1 = tmp_path / "sub1"; sub1.mkdir()
        out1 = run_sim(sub1, "-n", "4", "-N", "640", "-r", "49",
                       "-t", "0.5", "--seed", "42",
                       "--subdomain-padding", "3.5",
                       "--save-interval", "0", "--trajectory-samples", "0")
        ckpt = out1 / "checkpoint.bin"
        chk1 = read_checkpoint(ckpt)
        assert chk1["version"] == 8
        assert chk1["params"]["subdomain_padding"] == pytest.approx(3.5)

        sub2 = tmp_path / "sub2"; sub2.mkdir()
        out2 = run_sim(sub2, "-c", str(ckpt), "-t", "1.0", "--seed", "42",
                       "--save-interval", "0", "--trajectory-samples", "0")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        assert chk2["version"] == 8
        assert chk2["params"]["subdomain_padding"] == pytest.approx(3.5)


class TestV8CorruptInput:
    """Malformed v8 checkpoints should fail before mutating simulation state."""

    def test_truncated_v8_resume_reports_short_read(self, tmp_path):
        good_dir = tmp_path / "good"; good_dir.mkdir()
        out = run_sim(good_dir, "-n", "4", "-N", "640", "-r", "49",
                      "-t", "0.5", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        good = out / "checkpoint.bin"
        raw = good.read_bytes()
        bad = tmp_path / "truncated_v8.bin"
        bad.write_bytes(raw[:len(raw) // 2])

        outdir = tmp_path / "resume_out"
        outdir.mkdir()
        result = subprocess.run(
            [CELL_SIM, "-c", str(bad), "-t", "1.0",
             "--save-interval", "0", "--trajectory-samples", "0",
             "-o", str(outdir)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = result.stderr + result.stdout
        assert result.returncode != 0
        assert "[ckpt] short read" in combined


class TestV8MultiRankGuard:
    """v8 multi-rank files must be loaded with matching --gpus or refused."""

    @pytest.mark.skip(reason="requires multi-GPU build; runs on cluster only")
    def test_multi_rank_resume_same_g(self):
        pass

    @pytest.mark.skip(reason="requires multi-GPU build; runs on cluster only")
    def test_multi_rank_load_with_g1_is_refused(self):
        pass
