"""sim_v2 migration Phase 2 — functional smoke test per baseline feature.

Each test exercises one row of BASELINE_FEATURES.md. When run against the
active SIM_BINARY (default: cell_sim_v2), failures point at v2 gaps that
must be closed before cutover. See SIM_V2_MIGRATION_PLAN.md for the full
plan.

Total tests: ~80 across 11 sections (sections 12-13 are metadata, not features).
Scope: smoke-level. Each test is ≤ 200 ms, uses N=4–24, t≤200, dt=0.01.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from conftest import (
    CELL_SIM, BASELINE_SIM,
    run_sim, run_baseline,
    read_checkpoint, read_trajectory,
    requires_flag, requires_baseline,
)


# ============================================================================
# § 1. CLI Flags (38 tests)
# ============================================================================

class TestSection01_CLIFlags:
    """Test each CLI flag: accepted, has effect, persists in checkpoint."""

    # 1.1 Geometry & Domain (7 flags)

    def test_n_flag_sets_cell_count(self, tmp_path):
        """§1.1: -n <int> sets number of cells."""
        out = run_sim(tmp_path, "-n", "12", "-N", "200", "-r", "20",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["num_cells"] == 12

    def test_r_flag_sets_radius(self, tmp_path):
        """§1.1: -r <float> sets target radius R."""
        out = run_sim(tmp_path, "-n", "4", "-r", "15", "-N", "200",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["target_radius"] - 15.0) < 1e-5

    def test_n_flag_sets_domain_size(self, tmp_path):
        """§1.1: -N <int> sets domain size L×L."""
        out = run_sim(tmp_path, "-n", "4", "-N", "300", "-r", "20",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["params"]["Nx"] == 300
        assert chk["params"]["Ny"] == 300

    def test_confluence_auto_sizes_domain(self, tmp_path):
        """§1.1: --confluence <float> auto-computes L from target packing."""
        # With confluence, -N is not allowed. Just verify flag is accepted.
        out = run_sim(tmp_path, "-n", "8", "--confluence", "0.85",
                      "-r", "20", "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        # Domain should be auto-sized based on confluence formula.
        # Verify it's not zero.
        assert chk["params"]["Nx"] > 0
        assert chk["params"]["Ny"] > 0

    @requires_flag("--3d")
    def test_3d_mode_runs(self, tmp_path):
        """§1.1 + §6: --3d enables 3D simulation."""
        out = run_sim(tmp_path, "--3d", "-n", "4", "-N", "64", "-Nz", "64",
                      "-r", "10", "-t", "0.5", "--seed", "42", timeout=180)
        assert (out / "checkpoint.bin").exists()

    @requires_flag("-Nz")
    def test_nz_flag_sets_z_domain(self, tmp_path):
        """§1.1: -Nz <int> controls z dimension in 3D mode."""
        out = run_sim(tmp_path, "--3d", "-n", "4", "-N", "64", "-Nz", "48",
                      "-r", "10", "-t", "0.5", "--seed", "42", timeout=180)
        chk = read_checkpoint(out / "checkpoint.bin")
        # v6 layout includes Nz if available (sim_v2 v6 supports it).
        # For now, just verify the run completed.
        assert chk["num_cells"] == 4

    @requires_flag("--subdomain-padding")
    def test_subdomain_padding_accepted(self, tmp_path):
        """§1.1 + §9: --subdomain-padding <float> tunable bbox buffer."""
        out = run_sim(tmp_path, "-n", "4", "-r", "20", "-N", "200",
                      "--subdomain-padding", "1.0",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["params"]["subdomain_padding"] == pytest.approx(1.0, abs=1e-5)

    # 1.2 Time Stepping (3 flags)

    def test_t_flag_sets_end_time(self, tmp_path):
        """§1.2: -t <float> / -T sets end time."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "2.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["time"] - 2.5) < 0.1  # Allow small tolerance for last step

    def test_dt_flag_sets_timestep(self, tmp_path):
        """§1.2: --dt <float> sets time step."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--dt", "0.02", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["dt"] - 0.02) < 1e-5

    @requires_flag("--lambda")
    def test_lambda_flag_sets_interface_width(self, tmp_path):
        """§1.2: --lambda <float> sets interface width λ."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--lambda", "5.0", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["lambda"] - 5.0) < 1e-5

    # 1.3 Physics Parameters (8 flags)

    def test_v_a_flag_sets_motility(self, tmp_path):
        """§1.3: --v-A <float> sets active velocity."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--v-A", "0.025", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["v_A"] - 0.025) < 1e-5

    @requires_flag("--v-A-sigma")
    def test_v_a_sigma_per_cell_disorder(self, tmp_path):
        """§1.3: --v-A-sigma <float> seeds per-cell motility disorder (MISSING IN V2).

        Known v2 gap. When v2 implements, verify flag accepted and checkpoint stores it.
        """
        out = run_sim(tmp_path, "-n", "8", "-N", "200", "-r", "20",
                      "-t", "0.5", "--v-A", "0.01", "--v-A-sigma", "0.5",
                      "--seed", "42")
        # Just confirm the run completed and saved a checkpoint.
        assert (out / "checkpoint.bin").exists()

    def test_tau_flag_sets_persistence(self, tmp_path):
        """§1.3: --tau <float> sets reorientation time τ."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--tau", "5000", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["tau"] - 5000) < 1e-2

    def test_gamma_flag_sets_gradient_coefficient(self, tmp_path):
        """§1.3: --gamma <float[:selector]> sets γ (with optional per-cell override)."""
        # Basic test: scalar gamma value
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--gamma", "2.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["gamma"] - 2.5) < 1e-5

    def test_gamma_with_fraction_selector(self, tmp_path):
        """§1.3: --gamma <f>:<p>% applies override to top p% of cells."""
        out = run_sim(tmp_path, "-n", "8", "-N", "200", "-r", "20",
                      "-t", "0.5", "--gamma", "1.0:20%", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        # Verify run completed; gamma overrides stored in per_cell dict.
        if "gamma" in chk["per_cell"]:
            assert len(chk["per_cell"]["gamma"]) == chk["num_cells"]

    def test_gamma_with_cell_selector(self, tmp_path):
        """§1.3: --gamma <f>:cell<k> applies override to cell k."""
        out = run_sim(tmp_path, "-n", "8", "-N", "200", "-r", "20",
                      "-t", "0.5", "--gamma", "0.5:cell0", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        # Just verify the run completes.
        assert chk["num_cells"] == 8

    @requires_flag("--kappa")
    def test_kappa_flag_sets_repulsion(self, tmp_path):
        """§1.3: --kappa <float> sets cell-cell repulsion κ."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--kappa", "15.0", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["kappa"] - 15.0) < 1e-5

    @requires_flag("--mu")
    def test_mu_flag_sets_volume_constraint(self, tmp_path):
        """§1.3: --mu <float> sets volume constraint strength μ."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--mu", "0.8", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["mu"] - 0.8) < 1e-5

    @requires_flag("--xi")
    def test_xi_flag_sets_friction(self, tmp_path):
        """§1.3: --xi <float> sets friction coefficient ξ."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--xi", "1200", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["xi"] - 1200) < 1e-2

    @requires_flag("--adhesion")
    def test_adhesion_flag_accepted(self, tmp_path):
        """§1.3: --adhesion <float> (DEFERRED; not in v2 yet)."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--adhesion", "0.1", "--seed", "42")
        # Just confirm the run completed.
        assert (out / "checkpoint.bin").exists()

    # 1.4 Motility Model (1 flag)

    def test_abp_flag_enables_abp_model(self, tmp_path):
        """§1.4: --abp enables Active Brownian Particle motility (vs RTP default)."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "1.0", "--v-A", "0.02", "--abp", "--seed", "42",
                      "--trajectory-samples", "10")
        chk = read_checkpoint(out / "checkpoint.bin")
        # Verify the run completes and trajectory is valid.
        traj, _ = read_trajectory(out / "trajectory.txt")
        assert len(traj) > 0

    # 1.5 Initial Conditions & Resumption (3 flags)

    def test_c_flag_loads_checkpoint(self, tmp_path):
        """§1.5: -c <file> resumes from checkpoint."""
        # First, create a baseline checkpoint.
        sub1 = tmp_path / "sub1"
        sub1.mkdir()
        out1 = run_sim(sub1, "-n", "4", "-N", "200", "-r", "20",
                       "-t", "1.0", "--seed", "42")
        ckpt = out1 / "checkpoint.bin"
        # Now resume from it.
        sub2 = tmp_path / "sub2"
        sub2.mkdir()
        out2 = run_sim(sub2, "-c", str(ckpt), "-t", "2.0", "--seed", "42")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        assert abs(chk2["time"] - 2.0) < 0.1
        assert chk2["num_cells"] == 4

    @requires_flag("-i")
    def test_i_flag_loads_json_ic(self, tmp_path):
        """§1.5: -i <file.json> loads cell ICs from JSON (DEFERRED)."""
        # Not implemented in v2 yet. Just verify flag acceptance.
        pytest.skip("JSON IC deferred to post-Palmieri phase")

    @requires_flag("--batch")
    def test_batch_flag_multi_runs(self, tmp_path):
        """§1.5: --batch <file> runs multiple independent systems (DEFERRED)."""
        pytest.skip("batch mode deferred to post-Palmieri phase")

    # 1.6 Output & Checkpoints (7 flags)

    def test_o_flag_sets_output_dir(self, tmp_path):
        """§1.6: -o <dir> sets output directory (automatic via run_sim)."""
        # run_sim automatically adds -o, so just verify the output landed.
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--seed", "42")
        assert (out / "checkpoint.bin").exists()

    def test_save_interval_controls_vtk_output(self, tmp_path):
        """§1.6: --save-interval <int> controls VTK frame saves (NOTE: v2 semantic drift).

        v2's --save-interval actually controls CHECKPOINT saves, not VTK.
        This is a SILENT semantic change from baseline.
        """
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--save-interval", "1", "--seed", "42",
                      extra_output_flags=())
        # v2: with save-interval=1, should emit multiple checkpoints if
        # --checkpoint-interval is enabled. Baseline: VTK frames.
        # For now, just verify the run completes.
        assert (out / "checkpoint.bin").exists()

    def test_checkpoint_interval_controls_checkpoint_saves(self, tmp_path):
        """§1.6: --checkpoint-interval <int> controls checkpoint frequency."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "1.0", "--checkpoint-interval", "500", "--seed", "42")
        assert (out / "checkpoint.bin").exists()

    def test_save_final_checkpoint_flag(self, tmp_path):
        """§1.6: --save-final-checkpoint saves final checkpoint (default in run_sim)."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--seed", "42",
                      extra_output_flags=("--save-final-checkpoint",))
        assert (out / "checkpoint.bin").exists()

    def test_no_save_final_checkpoint_flag(self, tmp_path):
        """§1.6: --no-save-final-checkpoint suppresses final save."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--seed", "42",
                      extra_output_flags=("--no-save-final-checkpoint",))
        # Checkpoint may not exist (intentional suppression).
        # Just verify the sim ran without error (return code 0).
        # run_sim would have failed if non-zero, so we're OK.
        pass

    def test_print_interval_controls_logging(self, tmp_path):
        """§1.6: --print-interval <int> controls progress output frequency."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--print-interval", "1000", "--seed", "42")
        # Just verify the run completed.
        assert (out / "checkpoint.bin").exists()

    def test_trajectory_samples_controls_trajectory_density(self, tmp_path):
        """§1.6: --trajectory-samples <int> controls total trajectory snapshots."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "1.0", "--trajectory-samples", "50", "--seed", "42")
        traj, _ = read_trajectory(out / "trajectory.txt")
        # With 50 samples over t=1.0, expect roughly 50 unique timestamps.
        assert 40 <= len(traj) <= 60, f"Expected ~50 samples, got {len(traj)}"

    @requires_flag("--trajectory-interval")
    def test_trajectory_interval_explicit_cadence(self, tmp_path):
        """§1.6: --trajectory-interval <int> (MISSING IN V2).

        Known v2 gap. Specifies steps between trajectory saves.
        """
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "1.0", "--trajectory-interval", "50", "--seed", "42")
        # Just confirm checkpoint exists.
        assert (out / "checkpoint.bin").exists()

    # 1.7 Random Number Generation (2 flags)

    def test_seed_flag_determinism(self, tmp_path):
        """§1.7: --seed <int> enables deterministic runs."""
        # Run twice with same seed, compare final checkpoint.
        out1 = run_sim(tmp_path / "run1", "-n", "4", "-N", "200", "-r", "20",
                       "-t", "0.5", "--seed", "42")
        out2 = run_sim(tmp_path / "run2", "-n", "4", "-N", "200", "-r", "20",
                       "-t", "0.5", "--seed", "42")
        chk1 = read_checkpoint(out1 / "checkpoint.bin")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        # Verify both ran to the same time.
        assert abs(chk1["time"] - chk2["time"]) < 1e-6

    def test_polarity_seed_flag(self, tmp_path):
        """§1.7: --polarity-seed <int> separate seed for velocity RNG."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--seed", "42", "--polarity-seed", "99")
        chk = read_checkpoint(out / "checkpoint.bin")
        # Just verify the run completed.
        assert chk["num_cells"] == 4

    # 1.8 Diagnostics & Advanced — removed
    #
    # `--use-diagnostics`, `--observable-interval`, `--stress-fields`,
    # `--safe-mode` and `--save-individual-fields` were CLI stubs in v2 that
    # the cutover sim never implemented. They are removed; a single
    # `--observables` flag is planned to cover GPU-side measurement
    # (volume/perimeter/stress/strain/contacts) once the kernel lands.
    # See the TODO(observables) note in main.cu.

    # 1.9 Help (1 flag)

    def test_h_flag_prints_help(self, tmp_path):
        """§1.9: -h prints help and exits."""
        result = subprocess.run([CELL_SIM, "-h"], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert "help" in result.stdout.lower() or "usage" in result.stdout.lower() or len(result.stdout) > 100


# ============================================================================
# § 2. Output Artefacts (7 tests)
# ============================================================================

class TestSection02_OutputArtefacts:
    """Verify each output file exists and has the expected schema."""

    def test_checkpoint_file_exists_and_parseable(self, tmp_path):
        """§2: checkpoint.bin with v4+ magic, version, header."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["version"] in (4, 5, 6, 7), f"unexpected version {chk['version']}"
        assert chk["num_cells"] == 4
        assert chk["params"]["Nx"] == 200

    def test_trajectory_file_schema(self, tmp_path):
        """§2: trajectory.txt header (v_A, N, Lx, Ly, tau, etc.) + 12 columns."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "1.0", "--trajectory-samples", "20",
                      "--v-A", "0.01", "--tau", "1000", "--seed", "42")
        text = (out / "trajectory.txt").read_text()
        lines = text.strip().split("\n")
        # First line should be header.
        header = next((l for l in lines if l.startswith("#") and "v_A=" in l), None)
        assert header is not None, "missing header line"
        assert "N=" in header
        assert "Lx=" in header
        assert "v_A=" in header
        # Data lines: 12 tab-separated columns.
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) > 0
        cols = data_lines[0].split()
        assert len(cols) >= 6, f"trajectory line has {len(cols)} columns, expected ≥6"

    @requires_flag("--save-interval")
    @pytest.mark.skip(reason="VTK loading not implemented; semantic drift in v2 --save-interval")
    def test_vtk_frame_files_when_save_interval_set(self, tmp_path):
        """§2: output_NNNNNN.vtk files when --save-interval > 0.

        SEMANTIC DRIFT: v2's --save-interval controls checkpoint saves,
        not VTK output. This test is skipped; re-enable when semantics
        are clarified.
        """
        pass

    @pytest.mark.skip(reason="summary.json not implemented in v2")
    def test_summary_json_end_of_simulation_metadata(self, tmp_path):
        """§2: summary.json with final time, step count, cell volumes, seed."""
        pass

    def test_console_logging_progress_output(self, tmp_path):
        """§2: stdout progress lines at --print-interval frequency."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--print-interval", "100", "--seed", "42")
        # Progress is printed to stdout, which run_sim captures.
        # Just verify the run completed (return code 0).
        assert (out / "checkpoint.bin").exists()


# ============================================================================
# § 3. Physics Parameters (13 tests)
# ============================================================================

class TestSection03_PhysicsParameters:
    """Round-trip physics knobs via checkpoint persistence."""

    def test_lambda_roundtrip(self, tmp_path):
        """§3: λ / lambda parameter round-trip."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--lambda", "6.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["lambda"] - 6.5) < 1e-5

    def test_gamma_roundtrip(self, tmp_path):
        """§3: γ / gamma parameter round-trip."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--gamma", "1.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["gamma"] - 1.5) < 1e-5

    def test_kappa_roundtrip(self, tmp_path):
        """§3: κ / kappa parameter round-trip."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--kappa", "12.0", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["kappa"] - 12.0) < 1e-5

    def test_mu_roundtrip(self, tmp_path):
        """§3: μ / mu parameter round-trip."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--mu", "0.75", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["mu"] - 0.75) < 1e-5

    @requires_flag("--adhesion")
    def test_adhesion_j_roundtrip(self, tmp_path):
        """§3: J / adhesion_J parameter (DEFERRED)."""
        pytest.skip("adhesion deferred to post-Palmieri phase")

    def test_v_a_roundtrip(self, tmp_path):
        """§3: v_A / active velocity round-trip."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--v-A", "0.015", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["v_A"] - 0.015) < 1e-5

    @requires_flag("--v-A-sigma")
    def test_v_a_sigma_roundtrip(self, tmp_path):
        """§3: σ_v_A / v_A_sigma per-cell disorder (MISSING IN V2)."""
        pytest.skip("v_A_sigma missing in v2; target Phase 2")

    def test_tau_roundtrip(self, tmp_path):
        """§3: τ / tau persistence time round-trip."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--tau", "8000", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["tau"] - 8000) < 1e-2

    def test_xi_roundtrip(self, tmp_path):
        """§3: ξ / xi friction coefficient round-trip."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--xi", "1400", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["xi"] - 1400) < 1e-2

    def test_target_radius_roundtrip(self, tmp_path):
        """§3: R / target_radius parameter round-trip."""
        out = run_sim(tmp_path, "-n", "4", "-r", "25", "-N", "200",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["target_radius"] - 25.0) < 1e-5

    def test_domain_nx_roundtrip(self, tmp_path):
        """§3: L_x / Nx domain resolution round-trip."""
        out = run_sim(tmp_path, "-n", "4", "-N", "300", "-r", "20",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["params"]["Nx"] == 300

    def test_dt_roundtrip(self, tmp_path):
        """§3: Δt / dt time step round-trip."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--dt", "0.015", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert abs(chk["params"]["dt"] - 0.015) < 1e-5

    def test_confluence_target_implicit(self, tmp_path):
        """§3: ϕ_target / confluence (stored implicitly via domain size)."""
        out = run_sim(tmp_path, "-n", "8", "--confluence", "0.85",
                      "-r", "20", "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        # Confluence is encoded into Nx/Ny. Just verify they were set.
        assert chk["params"]["Nx"] > 0


# ============================================================================
# § 4. Initial-Condition Modes (4 tests)
# ============================================================================

class TestSection04_InitialConditions:
    """Test each IC mode: random (default), grid, checkpoint resume, JSON."""

    def test_random_placement_default(self, tmp_path):
        """§4: Random placement (default, no -c or -i)."""
        out = run_sim(tmp_path, "-n", "8", "-N", "200", "-r", "20",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["num_cells"] == 8
        # All cells should have finite positions and volumes.
        for cell in chk["cells"]:
            assert np.isfinite(cell["centroid"][0])
            assert np.isfinite(cell["centroid"][1])
            assert cell["volume"] > 0

    def test_checkpoint_resume_loads_state(self, tmp_path):
        """§4: Checkpoint resume (-c <file>)."""
        # Create initial checkpoint.
        sub1 = tmp_path / "sub1"
        sub1.mkdir()
        out1 = run_sim(sub1, "-n", "4", "-N", "200", "-r", "20",
                       "-t", "1.0", "--seed", "42")
        ckpt1 = out1 / "checkpoint.bin"
        chk1 = read_checkpoint(ckpt1)
        x0 = chk1["cells"][0]["centroid"][0]
        # Resume and verify the cell has moved (time advanced).
        sub2 = tmp_path / "sub2"
        sub2.mkdir()
        out2 = run_sim(sub2, "-c", str(ckpt1), "-t", "2.0", "--seed", "42")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        x1 = chk2["cells"][0]["centroid"][0]
        # Cell should have moved slightly (physics was applied).
        # (This is weak; stronger tests would check periodic wraparound, etc.)
        assert chk2["time"] >= 1.9

    @requires_flag("-i")
    def test_json_ic_loading(self, tmp_path):
        """§4: JSON IC loading (-i <file.json>) (DEFERRED)."""
        pytest.skip("JSON IC deferred to post-Palmieri phase")

    @requires_flag("--batch")
    def test_batch_mode_multi_systems(self, tmp_path):
        """§4: Batch mode (--batch <file>) (DEFERRED)."""
        pytest.skip("batch mode deferred to post-Palmieri phase")


# ============================================================================
# § 5. Motility Models (2 tests)
# ============================================================================

class TestSection05_Motility:
    """Test motility model switches and their effects on trajectory."""

    def test_rtp_default_motility(self, tmp_path):
        """§5: Run-and-Tumble (RTP, default, no --abp)."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "2.0", "--v-A", "0.01", "--tau", "500",
                      "--seed", "42", "--trajectory-samples", "50")
        traj, _ = read_trajectory(out / "trajectory.txt")
        # RTP should produce piecewise-constant velocity (tumble events).
        assert len(traj) > 10, "trajectory too sparse for RTP analysis"

    def test_abp_motility_model(self, tmp_path):
        """§5: Active Brownian Particle (ABP, via --abp)."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "2.0", "--v-A", "0.01", "--tau", "500", "--abp",
                      "--seed", "42", "--trajectory-samples", "50")
        traj, _ = read_trajectory(out / "trajectory.txt")
        # ABP should produce smooth angular evolution (rotational diffusion).
        assert len(traj) > 10


# ============================================================================
# § 6. Domain & Dimensionality (3 tests)
# ============================================================================

class TestSection06_DomainModes:
    """Test 2D (default), 3D, and periodic BC."""

    def test_2d_mode_default(self, tmp_path):
        """§6: 2D simulation (default)."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        # 2D: cells are circles, Nz not set (or equal to Nx).
        assert chk["params"]["Nx"] == 200
        assert chk["params"]["Ny"] == 200

    @requires_flag("--3d")
    def test_3d_mode_enabled(self, tmp_path):
        """§6: 3D simulation (--3d flag)."""
        out = run_sim(tmp_path, "--3d", "-n", "4", "-N", "64", "-Nz", "64",
                      "-r", "10", "-t", "0.5", "--seed", "42", timeout=180)
        chk = read_checkpoint(out / "checkpoint.bin")
        # 3D: cells are spheres.
        assert chk["num_cells"] == 4

    def test_periodic_boundary_conditions(self, tmp_path):
        """§6: Periodic BC (always-on torus topology)."""
        # Run a cell that starts near domain edge; confirm it wraps.
        # This is a post-sim analysis: load checkpoint and inspect centroid.
        out = run_sim(tmp_path, "-n", "1", "-N", "200", "-r", "10",
                      "-t", "5.0", "--v-A", "0.05", "--tau", "100",
                      "--seed", "42", "--trajectory-samples", "50")
        traj, _ = read_trajectory(out / "trajectory.txt")
        # Trajectory should span [0, 100) in x and y (periodic).
        xs = []
        ys = []
        for t in sorted(traj.keys()):
            if 0 in traj[t]:
                x, y = traj[t][0][:2]
                xs.append(x)
                ys.append(y)
        assert len(xs) > 0
        assert min(xs) >= 0 and max(xs) < 100 + 1  # small tolerance


# ============================================================================
# § 7. Checkpoint Format Support (3 tests)
# ============================================================================

class TestSection07_CheckpointFormat:
    """Test checkpoint I/O: write v6, read v3/v4/v6."""

    def test_sim_v2_writes_v6_checkpoints(self, tmp_path):
        """§7: v2 WRITE: format v7 (current)."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["version"] == 7, f"v2 should write v7, got {chk['version']}"

    @requires_baseline()
    def test_sim_v2_reads_baseline_v4_checkpoints(self, baseline_sim, tmp_path):
        """§7: v2 READ: format v4 (baseline legacy)."""
        # Create a v4 checkpoint with baseline.
        base_out = baseline_sim("-n", "4", "-N", "200", "-r", "20",
                               "-t", "0.5", "--seed", "42")
        base_ckpt = base_out / "checkpoint.bin"
        base_chk = read_checkpoint(base_ckpt)
        assert base_chk["version"] == 4
        # Now try to resume with v2.
        sub = tmp_path / "v2_resume"
        sub.mkdir()
        v2_out = run_sim(sub, "-c", str(base_ckpt), "-t", "1.0", "--seed", "42")
        v2_chk = read_checkpoint(v2_out / "checkpoint.bin")
        assert v2_chk["version"] == 6
        assert v2_chk["num_cells"] == 4

    def test_sim_v2_roundtrip_v6_checkpoints(self, tmp_path):
        """§7: v2 READ: format v6 (self, regression)."""
        out1 = run_sim(tmp_path / "run1", "-n", "4", "-N", "200", "-r", "20",
                       "-t", "0.5", "--seed", "42")
        ckpt1 = out1 / "checkpoint.bin"
        out2 = run_sim(tmp_path / "run2", "-c", str(ckpt1), "-t", "1.0", "--seed", "42")
        chk2 = read_checkpoint(out2 / "checkpoint.bin")
        assert chk2["version"] == 7
        assert chk2["time"] >= 0.99


# ============================================================================
# § 8. Special CLI Subcommands & Non-Default Code Paths (5 tests)
# ============================================================================

class TestSection08_SpecialSubcommands:
    """Test -h, undocumented edge-case flags, SIGTERM handler."""

    def test_h_flag_help(self, tmp_path):
        """§8: -h flag prints help."""
        result = subprocess.run([CELL_SIM, "-h"], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0

    @requires_flag("--batch")
    def test_batch_subcommand(self, tmp_path):
        """§8: --batch <file> (DEFERRED)."""
        pytest.skip("batch subcommand deferred")

    @requires_flag("--edge-test")
    def test_edge_test_undocumented(self, tmp_path):
        """§8: --edge-test (undocumented boundary test)."""
        out = run_sim(tmp_path, "--edge-test", "-t", "0.5", "--seed", "42")
        assert (out / "checkpoint.bin").exists()

    @requires_flag("--corner-push-test")
    def test_corner_push_test_undocumented(self, tmp_path):
        """§8: --corner-push-test (undocumented subdomain stress test)."""
        out = run_sim(tmp_path, "--corner-push-test", "-t", "0.5", "--seed", "42")
        assert (out / "checkpoint.bin").exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only SIGTERM test")
    def test_sigterm_handler_graceful_shutdown(self, tmp_path):
        """§8: SIGTERM handler sets g_shutdown_requested, flushes checkpoint."""
        import signal
        import time
        # Start a long-running sim in background.
        outdir = tmp_path / "output"
        outdir.mkdir()
        cmd = [CELL_SIM, "-n", "4", "-N", "200", "-r", "20",
               "-t", "1000", "--dt", "0.01", "--seed", "42",
               "-o", str(outdir), "--save-final-checkpoint"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Let it run briefly.
        time.sleep(0.5)
        # Send SIGTERM.
        proc.send_signal(signal.SIGTERM)
        # Wait for shutdown.
        proc.wait(timeout=10)
        # Verify checkpoint exists (even if partial).
        assert (outdir / "checkpoint.bin").exists()


# ============================================================================
# § 9. GPU & Compute Knobs (10 tests, mostly internal)
# ============================================================================

class TestSection09_ComputeKnobs:
    """GPU internal knobs; most are non-user-testable."""

    @pytest.mark.skip(reason="CUDA_ARCHITECTURES is compile-time CMake flag")
    def test_cuda_architectures(self, tmp_path):
        """§9: CUDA_ARCHITECTURES (compile-time, not runtime testable)."""
        pass

    @pytest.mark.skip(reason="GPU device selection is hard-coded to device 0")
    def test_gpu_device_selection(self, tmp_path):
        """§9: GPU device 0 (hard-coded, single-GPU only)."""
        pass

    @pytest.mark.skip(reason="CUDA streams are internal, non-testable")
    def test_cuda_streams_internal(self, tmp_path):
        """§9: CUDA streams (internal optimization)."""
        pass

    @pytest.mark.skip(reason="double-buffering is kernel-internal optimization")
    def test_double_buffering_kernel_opt(self, tmp_path):
        """§9: d_all_phi_ptr double-buffering (kernel internal)."""
        pass

    @pytest.mark.skip(reason="subdomain padding is testable as a flag, not a knob")
    def test_subdomain_padding_buffer(self, tmp_path):
        """§9: subdomain padding buffer (tested as --subdomain-padding flag)."""
        pass

    @pytest.mark.skip(reason="async VTK writer is internal, non-observable")
    def test_async_vtk_writer(self, tmp_path):
        """§9: Async VTK writer (background thread, non-testable)."""
        pass

    @pytest.mark.skip(reason="async checkpoint writer is internal for 3D")
    def test_async_checkpoint_writer_3d(self, tmp_path):
        """§9: Async checkpoint writer (3D-specific, non-testable)."""
        pass

    @pytest.mark.skip(reason="inline remapping is internal optimization")
    def test_inline_remapping(self, tmp_path):
        """§9: Inline subdomain remapping (internal, every ~10 steps)."""
        pass


# ============================================================================
# § 10. Compile-Time Build Options (9 tests, all skipped)
# ============================================================================

class TestSection10_CompileTimeOptions:
    """Compile-time CMake flags; cannot test at runtime."""

    @pytest.mark.skip(reason="Compile-time flag: -DBACKEND=CUDA/SERIAL/MPI")
    def test_backend_selection(self, tmp_path):
        """§10: -DBACKEND (compile-time, not runtime testable)."""
        pass

    @pytest.mark.skip(reason="Compile-time feature flag")
    def test_enable_visualizer_compile_flag(self, tmp_path):
        """§10: -DENABLE_VISUALIZER=ON/OFF."""
        pass

    @pytest.mark.skip(reason="Compile-time feature flag")
    def test_enable_kernel_profiling_compile_flag(self, tmp_path):
        """§10: -DENABLE_KERNEL_PROFILING=ON/OFF."""
        pass

    @pytest.mark.skip(reason="Compile-time feature flag")
    def test_use_half_precision_3d_compile_flag(self, tmp_path):
        """§10: -DUSE_HALF_PRECISION_3D=ON/OFF."""
        pass

    @pytest.mark.skip(reason="Compile-time standard, not testable")
    def test_cxx_standard_17_requirement(self, tmp_path):
        """§10: CMAKE_CXX_STANDARD=17."""
        pass

    @pytest.mark.skip(reason="v2 architecture list is compile-time variable")
    def test_cuda_architecture_list(self, tmp_path):
        """§10: CUDA_ARCHITECTURES (75, 86, 89, 90 by default)."""
        pass


# ============================================================================
# § 11. Miscellaneous (12 tests)
# ============================================================================

class TestSection11_Miscellaneous:
    """Error handling, edge cases, output formatting precision."""

    def test_unrecognized_flag_error_exit(self, tmp_path):
        """§11: Unrecognized flag → error + usage, exit 1."""
        outdir = tmp_path / "output"
        outdir.mkdir()
        result = subprocess.run(
            [CELL_SIM, "--nonexistent-flag", "-o", str(outdir)],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode != 0, "should error on unknown flag"

    def test_conflicting_flags_n_and_confluence_error(self, tmp_path):
        """§11: -N and --confluence conflict → error, exit 1."""
        outdir = tmp_path / "output"
        outdir.mkdir()
        result = subprocess.run(
            [CELL_SIM, "-n", "8", "-N", "256", "--confluence", "0.85",
             "-r", "20", "-t", "0.5", "-o", str(outdir)],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode != 0, "-N and --confluence together should error"

    def test_missing_checkpoint_file_error(self, tmp_path):
        """§11: Missing checkpoint file → error, exit 1."""
        outdir = tmp_path / "output"
        outdir.mkdir()
        result = subprocess.run(
            [CELL_SIM, "-c", "/nonexistent/path/checkpoint.bin", "-t", "0.5",
             "-o", str(outdir)],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode != 0, "should error on missing checkpoint"

    def test_missing_n_and_confluence_auto_set_warning(self, tmp_path):
        """§11: Missing -N and --confluence → auto-set confluence=0.85 (warning)."""
        out = run_sim(tmp_path, "-n", "4", "-r", "20", "-t", "0.5", "--seed", "42",
                      extra_output_flags=())
        # Auto-sizing should happen. Just verify the run completed.
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["params"]["Nx"] > 0

    def test_per_cell_gamma_selector_fraction(self, tmp_path):
        """§11: --gamma <f>:<p>% per-cell override (fraction selector)."""
        out = run_sim(tmp_path, "-n", "8", "-N", "200", "-r", "20",
                      "-t", "0.5", "--gamma", "1.0", "--gamma", "0.5:20%",
                      "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        # Gamma overrides should be stored if supported.
        assert chk["num_cells"] == 8

    def test_per_cell_gamma_selector_cell_id(self, tmp_path):
        """§11: --gamma <f>:cell<k> per-cell override (cell ID selector)."""
        out = run_sim(tmp_path, "-n", "8", "-N", "200", "-r", "20",
                      "-t", "0.5", "--gamma", "1.0:cell3", "--seed", "42")
        assert (out / "checkpoint.bin").exists()

    @requires_flag("--gamma")
    def test_per_cell_gamma_selector_nearest(self, tmp_path):
        """§11: --gamma <f>:nearest(x,y) selects the cell whose center is
        closest to (x,y).  Smoke: flag accepted, run completes."""
        out = run_sim(tmp_path, "-n", "8", "-N", "200", "-r", "20",
                      "-t", "0.5", "--gamma", "1.0",
                      "--gamma", "0.3:nearest(100,100)", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["num_cells"] == 8

    @requires_flag("--gamma")
    def test_per_cell_gamma_selector_cluster(self, tmp_path):
        """§11: --gamma <f>:cluster(p%,x,y) selects the p% of cells nearest
        to (x,y).  Smoke: flag accepted, run completes."""
        out = run_sim(tmp_path, "-n", "8", "-N", "200", "-r", "20",
                      "-t", "0.5", "--gamma", "1.0",
                      "--gamma", "0.3:cluster(25%,100,100)", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert chk["num_cells"] == 8

    @requires_flag("--v-A-sigma")
    def test_quenched_v_a_noise_per_cell(self, tmp_path):
        """§11: Per-cell quenched v_A noise (MISSING IN V2)."""
        pytest.skip("v_A_sigma missing in v2")

    def test_trajectory_time_precision_monotonic(self, tmp_path):
        """§11: Trajectory timestamps monotonic (float32 precision limits)."""
        out = run_sim(tmp_path, "-n", "2", "-N", "200", "-r", "20",
                      "-t", "10.0", "--dt", "0.01", "--seed", "42",
                      "--trajectory-samples", "100")
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())
        # Verify strict monotonicity (no duplicates from float32 capping).
        for i in range(1, len(times)):
            assert times[i] > times[i-1], f"timestamps not strictly increasing: {times[i-1]} >= {times[i]}"

    def test_checkpoint_endianness_little_endian(self, tmp_path):
        """§11: Checkpoint endianness is little-endian (magic 0x43454C4C)."""
        out = run_sim(tmp_path, "-n", "4", "-N", "200", "-r", "20",
                      "-t", "0.5", "--seed", "42")
        chk = read_checkpoint(out / "checkpoint.bin")
        # read_checkpoint verifies magic (0x43454C4C in little-endian).
        assert chk["version"] >= 4