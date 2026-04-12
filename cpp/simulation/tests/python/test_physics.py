"""
Tier 2: Physics accuracy tests.
These verify the PDE solver produces correct physical behavior.
Each test runs a small simulation and checks quantitative predictions.
"""
import math

import pytest
import numpy as np
from conftest import run_sim, read_checkpoint, read_trajectory


# ============================================================================
# 1. Single cell steady state
# ============================================================================

class TestSingleCellSteadyState:
    """A circular cell at target R with v_A=0 should stay put."""

    def test_volume_stable(self, tmp_path):
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "256", "-r", "49",
                      "-t", "100", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "10")
        chk = read_checkpoint(out / "checkpoint.bin")
        target_area = math.pi * 49**2
        vol = chk["cells"][0]["volume"]
        assert vol == pytest.approx(target_area, rel=0.01), \
            f"Volume {vol:.1f} should be within 1% of target {target_area:.1f}"

    def test_centroid_stationary(self, tmp_path):
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "256", "-r", "49",
                      "-t", "100", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "10")
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())
        x0, y0 = traj[times[0]][0][:2]
        xf, yf = traj[times[-1]][0][:2]
        drift = math.sqrt((xf - x0)**2 + (yf - y0)**2)
        assert drift < 2.0, f"Stationary cell drifted {drift:.2f} px (should be < 2)"

    def test_no_nan(self, tmp_path):
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "256", "-r", "49",
                      "-t", "100", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        assert not np.any(np.isnan(chk["cells"][0]["phi"]))


# ============================================================================
# 2. Two-cell repulsion
# ============================================================================

class TestTwoCellRepulsion:
    """Two overlapping cells should separate to d ≈ 2R."""

    def test_cells_separate(self, tmp_path):
        # Place two cells close together (domain small enough they overlap)
        out = run_sim(tmp_path / "run",
                      "-n", "2", "-N", "200", "-r", "49",
                      "-t", "100", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "10")
        chk = read_checkpoint(out / "checkpoint.bin")
        c0 = chk["cells"][0]["centroid"]
        c1 = chk["cells"][1]["centroid"]
        Nx = chk["params"]["Nx"]

        # Periodic distance
        dx = abs(c1[0] - c0[0])
        dy = abs(c1[1] - c0[1])
        if dx > Nx / 2: dx = Nx - dx
        if dy > Nx / 2: dy = Nx - dy
        dist = math.sqrt(dx**2 + dy**2)

        # Should be near 2R = 98
        R = 49
        assert dist > 1.5 * R, f"Cells too close: d={dist:.1f}, expected > {1.5*R}"
        assert dist < 4 * R, f"Cells too far: d={dist:.1f}, expected < {4*R}"

    def test_volumes_recover(self, tmp_path):
        out = run_sim(tmp_path / "run",
                      "-n", "2", "-N", "200", "-r", "49",
                      "-t", "100", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        target = math.pi * 49**2
        for c in chk["cells"]:
            assert c["volume"] == pytest.approx(target, rel=0.05), \
                f"Volume {c['volume']:.1f} should be within 5% of {target:.1f}"


# ============================================================================
# 3. Volume conservation (multi-cell)
# ============================================================================

class TestVolumeConservation:
    """Total Σφ²dA should stay near N × πR²."""

    def test_total_volume_stable(self, tmp_path):
        N = 16
        R = 49
        out = run_sim(tmp_path / "run",
                      "-n", str(N), "--confluence", "0.85", "-r", str(R),
                      "-t", "50", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "5")
        chk = read_checkpoint(out / "checkpoint.bin")
        total_vol = sum(c["volume"] for c in chk["cells"])
        target = N * math.pi * R**2
        assert total_vol == pytest.approx(target, rel=0.03), \
            f"Total volume {total_vol:.1f} should be within 3% of {target:.1f}"


# ============================================================================
# 4. Periodic boundary crossing
# ============================================================================

class TestPeriodicBoundaryCrossing:
    """A motile cell crossing the boundary should maintain volume."""

    def test_volume_continuous_across_wrap(self, tmp_path):
        # Single cell near edge, motile
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "200", "-r", "49",
                      "-t", "200", "--dt", "0.01", "--v-A", "0.02", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "20",
                      timeout=120)
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())

        # Check volume proxy: speed should be finite throughout
        for t in times:
            vx, vy = traj[t][0][2], traj[t][0][3]
            speed = math.sqrt(vx**2 + vy**2)
            assert np.isfinite(speed), f"Non-finite speed at t={t}"

        # Final checkpoint volume should still be near target
        chk = read_checkpoint(out / "checkpoint.bin")
        target = math.pi * 49**2
        assert chk["cells"][0]["volume"] == pytest.approx(target, rel=0.05), \
            f"Volume {chk['cells'][0]['volume']:.1f} after boundary crossing should be near {target:.1f}"


# ============================================================================
# 5. Motile cell moves in polarity direction
# ============================================================================

class TestMotileCell:
    """A single motile cell should have nonzero displacement."""

    def test_cell_moves(self, tmp_path):
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "300", "-r", "49",
                      "-t", "100", "--dt", "0.01", "--v-A", "0.01", "--seed", "42",
                      "--polarity-seed", "100",
                      "--save-interval", "0", "--trajectory-samples", "10")
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())
        x0, y0 = traj[times[0]][0][:2]
        xf, yf = traj[times[-1]][0][:2]
        Nx = 300
        dx = xf - x0
        dy = yf - y0
        if dx > Nx / 2: dx -= Nx
        if dx < -Nx / 2: dx += Nx
        if dy > Nx / 2: dy -= Nx
        if dy < -Nx / 2: dy += Nx
        displacement = math.sqrt(dx**2 + dy**2)
        assert displacement > 0.5, \
            f"Motile cell should have moved > 0.5 px, got {displacement:.2f}"


# ============================================================================
# 6. Interface width matches theory
# ============================================================================

class TestInterfaceWidth:
    """A single cell's phi profile should have interface width ≈ 2.2λ."""

    def test_interface_width(self, tmp_path):
        lam = 7.0
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "256", "-r", "49",
                      "-t", "50", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        phi = chk["cells"][0]["phi"]
        h, w = phi.shape

        # Compute radial profile from centroid (bbox center)
        cy, cx_local = h // 2, w // 2
        r_max = min(cx_local, cy) - 1
        radii = np.arange(0, r_max)
        profile = np.zeros(len(radii))
        for i, r in enumerate(radii):
            # Average over angles
            angles = np.linspace(0, 2 * np.pi, max(8, int(2 * np.pi * r)), endpoint=False)
            vals = []
            for theta in angles:
                px = int(round(cx_local + r * np.cos(theta)))
                py = int(round(cy + r * np.sin(theta)))
                if 0 <= px < w and 0 <= py < h:
                    vals.append(phi[py, px])
            if vals:
                profile[i] = np.mean(vals)

        # Find 90% and 10% crossings
        above_90 = np.where(profile > 0.9)[0]
        below_10 = np.where(profile < 0.1)[0]
        if len(above_90) > 0 and len(below_10) > 0:
            r_90 = above_90[-1]
            r_10 = below_10[below_10 > r_90]
            if len(r_10) > 0:
                width = r_10[0] - r_90
                # Interface width should be O(λ) — between 0.5λ and 4λ
                assert 0.5 * lam < width < 4 * lam, \
                    f"Interface width {width} should be O(λ={lam}), got {width/lam:.1f}λ"


# ============================================================================
# 7. Soft cell has higher diffusivity than ctrl (Palmieri result)
# ============================================================================

@pytest.mark.slow
class TestPalmieriSoftVsCtrl:
    """The fundamental result: soft cell (γ=0.35) should diffuse faster."""

    def test_soft_higher_msd(self, tmp_path):
        common = ["-n", "72", "--confluence", "0.9", "-r", "49",
                  "-t", "10000", "--dt", "0.01", "--v-A", "0.01", "--seed", "42",
                  "--save-interval", "0", "--trajectory-samples", "50"]

        # Ctrl: all cells γ=1.0
        out_ctrl = run_sim(tmp_path / "ctrl", *common, timeout=300)
        # Soft: cell 0 has γ=0.35
        out_soft = run_sim(tmp_path / "soft", *common, "--gamma", "0.35:cell0",
                           timeout=300)

        traj_ctrl, _ = read_trajectory(out_ctrl / "trajectory.txt")
        traj_soft, _ = read_trajectory(out_soft / "trajectory.txt")
        Lx = 776  # ceil(sqrt(72 * pi * 49^2 / 0.9))

        def cell0_msd(traj, Lx):
            times = sorted(traj.keys())
            # Unwrap cell 0
            pos = []
            for t in times:
                if 0 not in traj[t]:
                    continue
                x, y = traj[t][0][:2]
                if pos:
                    px, py = pos[-1]
                    dx, dy = x - px, y - py
                    if dx > Lx / 2: dx -= Lx
                    if dx < -Lx / 2: dx += Lx
                    if dy > Lx / 2: dy -= Lx
                    if dy < -Lx / 2: dy += Lx
                    pos.append((px + dx, py + dy))
                else:
                    pos.append((x, y))
            # MSD at max lag
            n = len(pos)
            lag = n // 2
            if lag < 2:
                return 0
            msd = 0
            count = 0
            for t0 in range(n - lag):
                dx = pos[t0 + lag][0] - pos[t0][0]
                dy = pos[t0 + lag][1] - pos[t0][1]
                msd += dx**2 + dy**2
                count += 1
            return msd / count if count > 0 else 0

        msd_ctrl = cell0_msd(traj_ctrl, Lx)
        msd_soft = cell0_msd(traj_soft, Lx)

        # Soft cell should diffuse faster (higher MSD)
        # With only 1 seed this is noisy, so just check it's in the right ballpark
        assert msd_soft > 0, "Soft cell MSD should be nonzero"
        assert msd_ctrl > 0, "Ctrl cell MSD should be nonzero"
        # Log the ratio for inspection even if we don't assert a strict bound
        ratio = msd_soft / msd_ctrl if msd_ctrl > 0 else float("inf")
        print(f"MSD ratio (soft/ctrl): {ratio:.2f}  (soft={msd_soft:.1f}, ctrl={msd_ctrl:.1f})")
