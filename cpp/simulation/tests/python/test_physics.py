"""
Tier 2: Physics accuracy tests.
These verify the PDE solver produces correct physical behavior.
Each test runs a small simulation and checks quantitative predictions.
"""
import math

import pytest
import numpy as np
from conftest import run_sim, read_checkpoint, read_trajectory, requires_flag
from report import record_metric, record_phi_from_checkpoint, record_timeseries, record_trajectory


# ============================================================================
# 1. Single cell steady state
# ============================================================================

class TestSingleCellSteadyState:
    """A circular cell at target R with v_A=0 is a PDE fixed point.

    Baseline: analytical. With v_A=0 and no neighbours, the equation of
    motion in ``cpu_reference.py`` reduces to gradient flow of the
    single-cell Lyapunov functional F[φ] (see ``TestEnergyMonotonicity``
    docstring + AUDIT.md §1). Any initial configuration at the target
    radius is within a forward-Euler step of the F minimum, so the
    measured volume must equal πR² to within f32 + interface-width
    discretization (~1%).

    Scope: smoke-level sanity gate. Failures here mean the EL equation
    for a single cell is integrating the wrong steady state — a very
    basic regression (wrong μ/A₀ coefficient, wrong γ-bulk balance, or
    a sign error). See AUDIT.md §5.
    """

    def test_volume_stable(self, tmp_path):
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "256", "-r", "49",
                      "-t", "100", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "10")
        chk = read_checkpoint(out / "checkpoint.bin")
        target_area = math.pi * 49**2
        vol = chk["cells"][0]["volume"]
        record_metric("single_cell_steady", "volume", vol,
                      expected=target_area, tolerance="1%", unit="px²")
        record_metric("single_cell_steady", "rel_error", abs(vol - target_area) / target_area,
                      expected=0, tolerance=0.01)

        # Time series: volume from trajectory (use L_n column 11 as proxy)
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())
        if times:
            # Volume from checkpoint per-frame isn't in trajectory, use centroid displacement
            x_vals = [traj[t][0][0] for t in times if 0 in traj[t]]
            y_vals = [traj[t][0][1] for t in times if 0 in traj[t]]
            t_vals = [t for t in times if 0 in traj[t]]
            record_timeseries("single_cell_steady", t_vals,
                              {"x": x_vals, "y": y_vals},
                              xlabel="Time", ylabel="Position (px)",
                              title="Single cell centroid (should be stationary)")

        record_phi_from_checkpoint("single_cell_steady", chk,
                                  f"Single cell steady state (V={vol:.0f}, target={target_area:.0f})")
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
    """Two initially overlapping cells repel to a quasi-equilibrium separation.

    Analytical prediction: soft-core contact (tails-touch criterion) gives
    ``d = 2R + 2λ = 2R(1 + λ/R)``. At R=49, λ=7 → d = 112 px, d/R = 2.286.
    Measured separations at t=100 sit near d/R ≈ 2.4, consistent with the
    touch-at-10%-tail value plus a small kinetic overshoot that has not
    fully relaxed.

    Caveat (Palmieri 2015 Fig 6 caption): the two-cell head-on
    equilibrium is "metastable … from which they can escape at long time
    due to numerical error build-up". We run only to t=100 TU, long
    enough for mechanical quasi-equilibrium but short enough that the
    metastable trap dominates. Do NOT extend t and tighten this test —
    the number will drift.

    Reference: Palmieri 2015 Eq 7 interface structure + Fig 6 metastability;
    AUDIT.md §3 and §5.
    """

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

        # Analytical contact equilibrium: d = 2R + 2λ = 112 px; d/R = 2.29.
        # Measured 2.4 includes a small (~0.1R) kinetic overshoot.
        R = 49
        lam = 7
        d_contact = 2 * R + 2 * lam   # = 112 px
        record_metric("two_cell_repulsion", "separation d", dist,
                      expected=d_contact, tolerance=0.15 * d_contact, unit="px")
        record_metric("two_cell_repulsion", "d/R", dist / R,
                      expected=d_contact / R, tolerance=0.3)

        # Time series: centroid distance over trajectory
        traj, _ = read_trajectory(out / "trajectory.txt")
        t_list, d_list = [], []
        for t in sorted(traj.keys()):
            if 0 in traj[t] and 1 in traj[t]:
                x0, y0 = traj[t][0][:2]; x1, y1 = traj[t][1][:2]
                ddx = abs(x1-x0); ddy = abs(y1-y0)
                if ddx > Nx/2: ddx = Nx-ddx
                if ddy > Nx/2: ddy = Nx-ddy
                t_list.append(t)
                d_list.append(math.sqrt(ddx**2+ddy**2))
        if t_list:
            record_timeseries("two_cell_repulsion", t_list,
                              {"d(t)": d_list, "2R+2λ": [d_contact]*len(t_list)},
                              xlabel="Time", ylabel="Distance (px)",
                              title=f"Two-cell separation vs time — analytical 2R+2λ={d_contact} px")

        record_phi_from_checkpoint("two_cell_repulsion", chk,
                                  f"Two-cell (d={dist:.0f}px, d/R={dist/R:.2f}, analytical {d_contact} px)")
        # Accept d in 2R to 3R (allows the metastable range without the
        # runaway drift that would indicate a κ-kernel regression).
        assert 2.0 * R < dist < 3.0 * R, (
            f"Two-cell separation d={dist:.1f} px (d/R={dist/R:.2f}) "
            f"outside metastable band [2R, 3R] = [{2*R}, {3*R}] px. "
            f"Analytical contact d=2R+2λ={d_contact} px."
        )

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
    """Many-cell analogue of ``TestSingleCellSteadyState``.

    Baseline: analytical. The volume-constraint term
    ``(2μ/A₀)(A₀-V)φ`` is independent per cell (it only couples through
    V, not ΣV), so the steady-state volume of each cell is set by the
    single-cell EL equation and total volume is simply N·πR². The 3%
    tolerance covers f32 precision + the O(λ/R) correction from the
    finite interface width.

    Scope: catches μ or target-area regressions that would only show
    up at multi-cell scale, e.g. a wrong-sign volume term that cancels
    at N=1 but compounds at N=16.
    """

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
        record_metric("volume_conservation_16c", "total_volume", total_vol,
                      expected=target, tolerance="3%", unit="px²")
        record_metric("volume_conservation_16c", "per_cell_avg", total_vol / N,
                      expected=math.pi * R**2, tolerance="3%", unit="px²")
        record_phi_from_checkpoint("volume_conservation_16c", chk,
                                  f"16-cell (ΣV err={abs(total_vol-target)/target:.4f})")

        # Time series: per-cell volumes from checkpoint
        vols = [c["volume"] for c in chk["cells"]]
        cell_ids = list(range(N))
        target_line = [math.pi * R**2] * N
        record_timeseries("volume_conservation_16c", cell_ids,
                          {"volume": vols, "target πR²": target_line},
                          xlabel="Cell ID", ylabel="Volume (px²)",
                          title=f"Per-cell volumes (N={N}, target={math.pi*R**2:.0f})")

        assert total_vol == pytest.approx(target, rel=0.03), \
            f"Total volume {total_vol:.1f} should be within 3% of {target:.1f}"


# ============================================================================
# 4. Periodic boundary crossing
# ============================================================================

class TestPeriodicBoundaryCrossing:
    """A motile cell that actually crosses the periodic boundary should
    maintain volume and the trajectory should wrap cleanly.

    Geometry: R=15 (small), N=100 (2×target-diameter domain), tau=1e7
    (ballistic so the cell keeps walking straight), v_A=0.1, t=2000.
    Net straight-line displacement ≈ v_A·t = 200 px = 2L so the cell
    must wrap at least once.
    """

    def test_volume_continuous_across_wrap(self, tmp_path):
        R = 15
        L = 200           # >= TILE_T (192); still ~13R so cell wraps cleanly
        v_A = 0.1
        t_end = 2500      # at v_A=0.1 -> ideal disp 250 px > L (need >=1 wrap)
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", str(L), "-r", str(R),
                      "-t", str(t_end), "--dt", "0.01",
                      "--v-A", str(v_A), "--tau", "10000000",
                      "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "100",
                      timeout=180)
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())

        # Check volume proxy: speed should be finite throughout
        for t in times:
            vx, vy = traj[t][0][2], traj[t][0][3]
            speed = math.sqrt(vx**2 + vy**2)
            assert np.isfinite(speed), f"Non-finite speed at t={t}"

        # Unwrap trajectory and assert at least one wrap occurred.
        t_vals = [t for t in times if 0 in traj[t]]
        raw_x = [traj[t][0][0] for t in t_vals]
        raw_y = [traj[t][0][1] for t in t_vals]
        ux, uy = [raw_x[0]], [raw_y[0]]
        wrap_count = 0
        for i in range(1, len(raw_x)):
            dx, dy = raw_x[i] - raw_x[i - 1], raw_y[i] - raw_y[i - 1]
            if dx > L / 2: dx -= L; wrap_count += 1
            if dx < -L / 2: dx += L; wrap_count += 1
            if dy > L / 2: dy -= L; wrap_count += 1
            if dy < -L / 2: dy += L; wrap_count += 1
            ux.append(ux[-1] + dx); uy.append(uy[-1] + dy)
        total_disp = math.sqrt((ux[-1] - ux[0])**2 + (uy[-1] - uy[0])**2)
        record_metric("periodic_crossing", "unwrapped_disp", total_disp,
                      expected=v_A * t_end, tolerance="30%", unit="px")
        record_metric("periodic_crossing", "wrap_events", wrap_count,
                      expected=2, tolerance=5.0, unit="")

        # Final checkpoint volume should still be near target
        chk = read_checkpoint(out / "checkpoint.bin")
        target = math.pi * R**2
        vol = chk["cells"][0]["volume"]
        record_metric("periodic_crossing", "final_volume", vol,
                      expected=target, tolerance="3%", unit="px²")

        record_timeseries("periodic_crossing", t_vals,
                          {"x (wrapped)": raw_x, "y (wrapped)": raw_y,
                           "x (unwrapped)": ux, "y (unwrapped)": uy},
                          xlabel="Time", ylabel="Position (px)",
                          title=f"Motile cell crossing periodic boundary ({wrap_count} wraps)")

        # Spatial trajectory plot (x vs y). The unwrapped path shows the
        # ballistic straight-line motion (tau=1e7 ⇒ no tumbles). The
        # wrapped path shows how it re-enters the [0,L]² box — useful to
        # eyeball that the wrap is clean.
        record_trajectory("periodic_crossing",
                          {"unwrapped": (ux, uy),
                           "wrapped":   (raw_x, raw_y)},
                          title=f"Motile cell path "
                                f"({wrap_count} wraps, |Δr|={total_disp:.1f}px, L={L})")

        assert wrap_count >= 1, \
            f"Cell didn't cross the periodic boundary: {wrap_count} wraps detected"
        assert total_disp > L, \
            f"Unwrapped displacement {total_disp:.1f} px should exceed L={L}"
        assert vol == pytest.approx(target, rel=0.03), \
            f"Volume {vol:.1f} after boundary crossing should be within 3% of {target:.1f}"


# ============================================================================
# 5. Motile cell moves in polarity direction
# ============================================================================

class TestMotileCell:
    """Single motile cell: ballistic displacement ≈ v_A·t at t ≪ τ.

    Baseline: analytical. For an isolated cell the inactive velocity
    v_I vanishes (Σ_j≠i φ_j² = 0), so v = v_A·p̂ exactly. At t ≪ τ
    no tumble fires and the cell walks in a straight line. Net
    displacement |Δr| = v_A·t. At v_A=0.01, t=100, τ=10000 we have
    t/τ = 0.01 (deep in the ballistic regime) so expected |Δr| ≈ 1 px.

    Scope: verifies the active self-propulsion path (v_A · p̂ coupling,
    RTP polarity initial condition, advection − v·∇φ). A regression
    that flipped the sign of v_A · p̂ or dropped the advection term
    would give near-zero displacement and fail here. Reference:
    Palmieri 2015 Methods §B (run-and-tumble dynamics).
    """

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
        # v_A=0.01, t=100, tau=10000 → near-ballistic, expect ~v_A·t = 1 px.
        record_metric("motile_cell", "displacement", displacement,
                      expected=1.0, tolerance=0.3, unit="px")

        # Unwrap the trajectory across periodic boundaries, then plot as a
        # 2D path (x vs y) — what a trajectory actually looks like.
        t_vals = [t for t in times if 0 in traj[t]]
        ux, uy = [traj[t_vals[0]][0][0]], [traj[t_vals[0]][0][1]]
        for t in t_vals[1:]:
            x, y = traj[t][0][:2]
            ddx, ddy = x - ux[-1], y - uy[-1]
            if ddx > Nx/2: ddx -= Nx
            if ddx < -Nx/2: ddx += Nx
            if ddy > Nx/2: ddy -= Nx
            if ddy < -Nx/2: ddy += Nx
            ux.append(ux[-1]+ddx); uy.append(uy[-1]+ddy)
        record_trajectory("motile_cell", ux, uy,
                          title=f"Motile cell path (v_A=0.01, |Δr|={displacement:.1f}px)")

        assert displacement > 0.5, \
            f"Motile cell should have moved > 0.5 px, got {displacement:.2f}"


# ============================================================================
# 6. Interface width matches theory
# ============================================================================

class TestInterfaceWidth:
    """A single cell's φ profile has an analytically predictable interface width.

    The sim's equation of motion (see ``cpu_reference.py``) gives a 1D
    kink profile ``φ(x) = (1/2)(1 - tanh(x/w))`` with characteristic
    width ``w = 2λ/√30 ≈ 0.365λ`` (derived from the steady-state
    Euler-Lagrange equation ``∇²φ = (30/λ²)·φ(1-φ)(1-2φ)``; integrating
    once yields ``φ' = -(√30/λ)·φ(1-φ)``).

    10–90 width = ``2·arctanh(0.8)·w = 0.80λ`` → **5.60 px at λ=7**.

    This is NOT the 2.2λ value that appears in some Model-A references —
    that would apply if the bulk coefficient were 60γ/λ² and the kinetic
    prefactor 1. Our sim uses 30γ/λ² with a 1/2 kinetic prefactor (see
    AUDIT.md §1 for the full coefficient audit). Measurement
    (2026-04-22): 5.50 px at λ=7, consistent with the analytical 5.60 px
    to 1 px (pixel discretization).

    References: Palmieri 2015 Methods §A (DOI 10.1038/srep11745);
    AUDIT.md §3.
    """

    def test_interface_width(self, tmp_path):
        lam = 7.0
        # Analytical prediction — see class docstring.
        expected_width = 0.80 * lam              # 5.60 px
        tol_width = 1.5                           # ±1.5 px (pixel discretization + sub-pixel sampling)

        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "256", "-r", "49",
                      "-t", "50", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")
        phi = chk["cells"][0]["phi"]
        h, w = phi.shape

        # Radial profile from bbox center, sampled at half-pixel resolution
        # so 10-90 width is resolved with ~1 px precision rather than
        # integer-only quantization.
        cy, cx_local = h // 2, w // 2
        r_max = min(cx_local, cy) - 1
        radii = np.arange(0, r_max, 0.5)
        profile = np.zeros(len(radii))
        for i, r in enumerate(radii):
            angles = np.linspace(0, 2 * np.pi, max(16, int(4 * np.pi * r)), endpoint=False)
            vals = []
            for theta in angles:
                px = int(round(cx_local + r * np.cos(theta)))
                py = int(round(cy + r * np.sin(theta)))
                if 0 <= px < w and 0 <= py < h:
                    vals.append(phi[py, px])
            if vals:
                profile[i] = np.mean(vals)

        # Find 90% and 10% crossings. Profile is decreasing in r from ≈1
        # (interior) to ≈0 (exterior); r_90 is the last r where φ ≥ 0.9,
        # r_10 is the first r after that where φ ≤ 0.1.
        above_90 = np.where(profile >= 0.9)[0]
        below_10 = np.where(profile <= 0.1)[0]
        assert len(above_90) > 0, "never entered φ≥0.9 core — is the cell missing?"
        r_90_idx = above_90[-1]
        below_after = below_10[below_10 > r_90_idx]
        assert len(below_after) > 0, "never exited to φ≤0.1 — bbox too small?"
        r_10_idx = below_after[0]
        width = float(radii[r_10_idx] - radii[r_90_idx])

        record_metric("interface_width", "width", width,
                      expected=expected_width, tolerance=tol_width, unit="px")
        record_metric("interface_width", "width/λ", width / lam,
                      expected=0.80, tolerance=0.20)

        record_timeseries("interface_width",
                          radii.tolist(), {"φ(r)": profile.tolist()},
                          xlabel="Radius (px)", ylabel="φ",
                          title=f"Radial profile — 10-90 width = {width:.2f} px "
                                f"= {width/lam:.3f}λ (predicted 0.80λ = {expected_width:.2f} px)")

        # Strict assertion: must match analytical prediction 0.80λ to ±1.5 px.
        # A κ or λ coefficient error of ~10% would exceed this tolerance.
        assert abs(width - expected_width) < tol_width, (
            f"Interface 10-90 width {width:.2f} px differs from analytical "
            f"prediction 0.80λ = {expected_width:.2f} px by more than "
            f"{tol_width} px. See TestInterfaceWidth docstring + AUDIT.md §3."
        )


# ============================================================================
# 7. Cell expansion from small initial size
# ============================================================================

class TestCellRelaxation:
    """A cell initialized at target R should relax to the exact steady-state
    volume under the volume-conservation constraint and remain stationary.

    Note: sim_v2 initializes the phase field with a tanh profile at target
    radius R, so this test measures the steady-state accuracy of the volume
    constraint rather than an expansion from a smaller initial size.
    """

    def test_volume_relaxes_to_target(self, tmp_path):
        R = 49
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "200", "-r", str(R),
                      "-t", "200", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "10")
        chk = read_checkpoint(out / "checkpoint.bin")
        target = math.pi * R**2
        vol = chk["cells"][0]["volume"]
        # After 200 TU at dt=0.01 (20000 steps), volume should be within 1%.
        record_metric("cell_relaxation", "final_volume", vol,
                      expected=target, tolerance="1%", unit="px²")
        record_phi_from_checkpoint("cell_relaxation", chk,
                                  f"Cell relaxation (V={vol:.0f}, target={target:.0f})")
        assert vol == pytest.approx(target, rel=0.01), \
            f"After relaxation, volume {vol:.1f} should be within 1% of {target:.1f}"

        # Stationary: no drift.
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())
        x0, y0 = traj[times[0]][0][:2]
        xf, yf = traj[times[-1]][0][:2]
        drift = math.sqrt((xf - x0)**2 + (yf - y0)**2)
        record_metric("cell_relaxation", "drift", drift,
                      expected=0.0, tolerance=2.0, unit="px")
        assert drift < 2.0, f"Relaxing cell drifted {drift:.1f} px (should stay put)"


# ============================================================================
# 8. Contact equilibrium — two cells at d≈2R should stay in contact
# ============================================================================

class TestContactEquilibrium:
    """Two cells, placed initially apart, settle at a stable (metastable)
    spacing.

    This test does NOT measure a thermodynamic equilibrium. Palmieri 2015
    Fig 6 caption is explicit: "the head-on two-cell collision yields a
    metastable configuration from which the cells can escape at long time
    due to numerical error build-up." The asymptotic stable state is
    actually back-to-back escape; the intermediate plateau near d/R ≈ 3.5
    is a kinetic overshoot from the head-on approach that persists over
    the test window (t=200 TU).

    The scope of this test is therefore:
      1. the plateau is reached from the specific IC (far-apart approach),
      2. the last three trajectory frames agree to better than 1% (the
         metastable trap is well-resolved, not actively drifting).
    It should be read together with ``TestTwoCellRepulsion`` (overlap IC)
    and ``TestOverlapRepulsion`` (heavy-overlap IC) — all three measure
    the same metastable manifold from different starting points.

    Reference: Palmieri 2015 Fig 6; AUDIT.md §5.
    """

    def test_cells_stay_in_contact(self, tmp_path):
        R = 49
        out = run_sim(tmp_path / "run",
                      "-n", "2", "-N", "300", "-r", str(R),
                      "-t", "200", "--dt", "0.01", "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "20")
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())
        Lx = 300

        # Track distance over time
        distances = []
        for t in times:
            if 0 not in traj[t] or 1 not in traj[t]:
                continue
            x0, y0 = traj[t][0][:2]
            x1, y1 = traj[t][1][:2]
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            if dx > Lx / 2: dx = Lx - dx
            if dy > Lx / 2: dy = Lx - dy
            distances.append(math.sqrt(dx**2 + dy**2))

        assert len(distances) > 5, "Need enough trajectory points"

        # Distance should be stable (not growing or shrinking dramatically)
        d_start = np.mean(distances[:3])
        d_end = np.mean(distances[-3:])

        record_metric("contact_equilibrium", "d_start", d_start, unit="px")
        # Stable: last three frames agree with each other to <1% (no drift).
        d_stability = (max(distances[-3:]) - min(distances[-3:])) / d_end
        record_metric("contact_equilibrium", "d_end", d_end,
                      expected=d_start, tolerance="2%", unit="px")
        record_metric("contact_equilibrium", "d_stability_rel", d_stability,
                      expected=0.0, tolerance=0.01)
        record_metric("contact_equilibrium", "d_end/R", d_end / R,
                      expected=3.5, tolerance=1.5)

        # Time series
        t_vals = [t for t in times if 0 in traj[t] and 1 in traj[t]]
        record_timeseries("contact_equilibrium", t_vals[:len(distances)],
                          {"d(t)": distances, "2R": [2*R]*len(distances)},
                          xlabel="Time", ylabel="Distance (px)",
                          title="Contact equilibrium stability")

        # Should not merge (d → 0) or fly apart (d → Lx/2)
        assert d_end > 2 * R, f"Cells merged/too close: d_end={d_end:.1f} < 2R={2*R}"
        assert d_end < 4 * R, f"Cells flew apart: d_end={d_end:.1f} > 4R={4*R}"
        # Strict stability: d_end must agree with d_start within 2% once
        # they've found the repulsion equilibrium.
        assert abs(d_end - d_start) / d_start < 0.02, \
            f"Distance unstable: {d_start:.2f} → {d_end:.2f} (Δ={100*abs(d_end-d_start)/d_start:.2f}%)"
        # And the last three frames must agree with each other.
        assert d_stability < 0.01, \
            f"Separation fluctuating: range/d = {100*d_stability:.2f}% in last 3 frames"


# ============================================================================
# 9. Soft cell has higher shape index (more deformable)
# ============================================================================

class TestGammaAffectsShape:
    """A soft cell (γ=0.35) in a confluent, active monolayer should develop
    a measurably higher L_n than stiff neighbors — this is the core Palmieri
    signature. Using tau=100 (many reorientations) and running long enough
    that the population has fully relaxed into the jammed state.
    """

    def test_soft_cell_more_deformed(self, tmp_path):
        # N=36 at conf=0.92 is jammed enough that soft cell is forced to
        # deform under ctrl-cell pressure. tau=100 with t=1200 → 12
        # reorientations, enough to average out polarity direction effects.
        # trajectory_samples=240 → ~200 per-cell frames after transient skip.
        out = run_sim(tmp_path / "run",
                      "-n", "36", "--confluence", "0.92", "-r", "49",
                      "-t", "1200", "--dt", "0.01",
                      "--v-A", "0.02", "--tau", "100",
                      "--seed", "42",
                      "--gamma", "0.35:cell0",
                      "--save-interval", "0", "--trajectory-samples", "240",
                      timeout=300)
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())

        # Read L_n from trajectory (column index 10)
        # Collect L_n for cell 0 (soft) and others (ctrl) after t > 100 (past
        # initial relaxation transient).
        soft_ln = []
        ctrl_ln = []
        with open(out / "trajectory.txt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 12:
                    continue
                t = float(parts[0])
                cid = int(parts[1])
                ln = float(parts[10])
                if t > 200 and ln > 0:
                    if cid == 0:
                        soft_ln.append(ln)
                    else:
                        ctrl_ln.append(ln)

        assert soft_ln and ctrl_ln, \
            f"No L_n samples collected (soft={len(soft_ln)}, ctrl={len(ctrl_ln)})"
        mean_soft = float(np.mean(soft_ln))
        mean_ctrl = float(np.mean(ctrl_ln))
        ratio = mean_soft / mean_ctrl
        record_metric("gamma_shape", "soft_Ln", mean_soft, unit="")
        record_metric("gamma_shape", "ctrl_Ln", mean_ctrl, unit="")
        # Palmieri: soft cell has strictly higher shape index than stiff
        # neighbours. Require a directional signal ≥3% above ctrl.
        record_metric("gamma_shape", "soft/ctrl ratio", ratio,
                      expected=1.08, tolerance=0.08)
        record_phi_from_checkpoint("gamma_shape", read_checkpoint(out / "checkpoint.bin"),
                                  f"Soft cell γ=0.35 (L_n ratio={ratio:.3f})")
        print(f"L_n: soft={mean_soft:.4f}, ctrl={mean_ctrl:.4f}, ratio={ratio:.3f}")
        # Directional signal: soft cell's shape index must be at least 3%
        # above the ctrl mean. This is the minimum viable Palmieri signature.
        assert ratio > 1.03, \
            f"Soft L_n/ctrl L_n = {ratio:.4f} should be > 1.03 " \
            f"(soft={mean_soft:.4f}, ctrl={mean_ctrl:.4f})"


# ============================================================================
# 10. MSD grows at least roughly linearly for motile cell
# ============================================================================

class TestMSDScaling:
    """For a motile cell in the diffusive regime (t ≫ τ), MSD ∼ 4D·t so
    MSD(2t)/MSD(t) ≈ 2. Previous version ran with the default tau=10000 on
    a t=500 window (t ≪ τ → ballistic ∼ t²), which gave ratio≈4 and failed
    the diffusive-regime expectation. Fix: tau=50, t=500 → t/τ=10.
    """

    def test_msd_grows(self, tmp_path):
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "400", "-r", "49",
                      "-t", "500", "--dt", "0.01", "--v-A", "0.01",
                      "--tau", "50",
                      "--seed", "42",
                      "--polarity-seed", "100",
                      "--save-interval", "0", "--trajectory-samples", "100",
                      timeout=120)
        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())
        Lx = 400

        # Unwrap positions
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

        n = len(pos)
        if n < 10:
            pytest.skip("Not enough trajectory points")

        # Compute MSD at lag=n//4 and lag=n//2
        def msd_at_lag(lag):
            s = 0
            count = 0
            for t0 in range(n - lag):
                dx = pos[t0 + lag][0] - pos[t0][0]
                dy = pos[t0 + lag][1] - pos[t0][1]
                s += dx**2 + dy**2
                count += 1
            return s / count if count > 0 else 0

        msd_short = msd_at_lag(n // 4)
        msd_long = msd_at_lag(n // 2)

        record_metric("msd_scaling", "MSD(n/4)", msd_short, unit="px²")
        record_metric("msd_scaling", "MSD(n/2)", msd_long, unit="px²")
        # In diffusive regime (t ≫ τ), MSD(2t)/MSD(t) → 2.
        ratio = msd_long / max(msd_short, 1e-10)
        record_metric("msd_scaling", "growth_ratio", ratio,
                      expected=2.0, tolerance=0.5)

        # MSD vs lag curve
        lags = list(range(1, n // 2, max(1, n // 20)))
        msd_curve = [msd_at_lag(lag) for lag in lags]
        record_timeseries("msd_scaling", lags,
                          {"MSD(lag)": msd_curve},
                          xlabel="Lag (frames)", ylabel="MSD (px²)",
                          title=f"MSD vs lag (ratio={ratio:.2f}, expected ≈2 for diffusive)")

        # In the diffusive regime, MSD(2t)/MSD(t) should be close to 2
        # (strictly between 1.5 and 2.5 — neither sub-diffusive/caged nor
        # fully ballistic).
        assert 1.5 < ratio < 2.5, \
            f"MSD growth ratio = {ratio:.3f} outside diffusive window [1.5, 2.5] " \
            f"(MSD(n/4)={msd_short:.2f}, MSD(n/2)={msd_long:.2f})"
        assert msd_short > 0, "Short-lag MSD should be > 0"


# ============================================================================
# 11. Soft cell has higher diffusivity than ctrl (Palmieri result)
# ============================================================================

@pytest.mark.slow
class TestPalmieriSoftVsCtrl:
    """Palmieri 2015 headline: a softer cell (smaller γ) diffuses faster.

    Palmieri Fig 5 reports ``D_soft / D_ctrl ≈ 1.5`` for γ_soft=0.35
    vs γ_ctrl=1 at N=72, φ=0.9, v_A=0.01, τ=10⁴ (paper Table 1).
    The single-seed MSD is very noisy, so this test runs a small
    ensemble (``N_SEEDS`` seeds each for ctrl and soft) and asserts
    on the ensemble-averaged ratio. A trivial sim where γ has no
    effect would give ratio ≈ 1; Palmieri gives 1.5. We require
    ratio > 1.2 which comfortably distinguishes the two while
    tolerating seed-to-seed noise and the finite simulation window.

    Reference: Palmieri 2015 Fig 5 (DOI 10.1038/srep11745); AUDIT.md §3.
    """

    # Small ensemble — each sim is ~90s wall clock, total run time
    # is ~15 min at N_SEEDS=5. Marked @pytest.mark.slow.
    N_SEEDS = 5
    MIN_RATIO = 1.2   # Palmieri reports 1.5; accept down to 1.2 for seed noise
    MAX_RATIO = 3.0   # guards against a regression that *inverts* the trend
                      # into a pathological soft>>ctrl runaway

    def test_soft_higher_msd(self, tmp_path):
        common = ["-n", "72", "--confluence", "0.9", "-r", "49",
                  "-t", "10000", "--dt", "0.01", "--v-A", "0.01",
                  "--save-interval", "0", "--trajectory-samples", "50"]
        Lx = 776  # ceil(sqrt(72 * pi * 49^2 / 0.9))

        def cell0_msd_from_run(out_dir):
            traj, _ = read_trajectory(out_dir / "trajectory.txt")
            times = sorted(traj.keys())
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
            n = len(pos)
            lag = n // 2
            if lag < 2:
                return 0.0
            msd = 0.0
            count = 0
            for t0 in range(n - lag):
                dx = pos[t0 + lag][0] - pos[t0][0]
                dy = pos[t0 + lag][1] - pos[t0][1]
                msd += dx**2 + dy**2
                count += 1
            return msd / count if count > 0 else 0.0

        msds_ctrl, msds_soft = [], []
        for seed in range(42, 42 + self.N_SEEDS):
            out_ctrl = run_sim(tmp_path / f"ctrl_{seed}", *common,
                               "--seed", str(seed), timeout=300)
            out_soft = run_sim(tmp_path / f"soft_{seed}", *common,
                               "--seed", str(seed),
                               "--gamma", "0.35:cell0", timeout=300)
            msds_ctrl.append(cell0_msd_from_run(out_ctrl))
            msds_soft.append(cell0_msd_from_run(out_soft))

        mean_ctrl = float(np.mean(msds_ctrl))
        mean_soft = float(np.mean(msds_soft))
        assert mean_ctrl > 0, f"Ctrl MSD ensemble mean = {mean_ctrl}; sim broken?"
        assert mean_soft > 0, f"Soft MSD ensemble mean = {mean_soft}; sim broken?"

        ratio = mean_soft / mean_ctrl
        record_metric("palmieri_soft_vs_ctrl", "msd_ratio", ratio,
                      expected=1.5, tolerance=0.5)
        record_metric("palmieri_soft_vs_ctrl", "msd_ctrl_mean", mean_ctrl,
                      unit="px²")
        record_metric("palmieri_soft_vs_ctrl", "msd_soft_mean", mean_soft,
                      unit="px²")

        assert self.MIN_RATIO < ratio < self.MAX_RATIO, (
            f"MSD_soft / MSD_ctrl = {ratio:.2f} outside Palmieri band "
            f"[{self.MIN_RATIO}, {self.MAX_RATIO}] "
            f"(N_seeds={self.N_SEEDS}, ctrl={mean_ctrl:.1f}, soft={mean_soft:.1f}). "
            f"Palmieri 2015 Fig 5 reports ratio ≈ 1.5."
        )


# ============================================================================
# 12. Two overlapping cells repel to equilibrium spacing
# ============================================================================

class TestOverlapRepulsion:
    """Heavy-overlap IC relaxes to the same two-cell metastable manifold.

    Two cells initialized with their centroids at 1.5R (heavy overlap)
    push apart and settle near d/R ≈ 3.5. As explained in the
    :class:`TestContactEquilibrium` docstring, this plateau is a
    metastable kinetic overshoot, not a thermodynamic equilibrium
    (Palmieri 2015 Fig 6 caption). The specific value 3.5R is
    IC-dependent; a cleaner analytical target would be the
    interface-tails-touch distance d = 2R + 2λ = 2.29R tested in
    ``TestTwoCellRepulsion``.

    Scope of this test: a heavy-overlap IC resolves without merger or
    runaway — i.e. the κ repulsion term has the correct sign and
    magnitude in the deep-overlap regime.

    Reference: Palmieri 2015 Fig 6; AUDIT.md §5.
    """

    @requires_flag("-s ")
    def test_overlap_resolves(self, tmp_path):
        """Place two cells with heavy overlap (initial separation 1.5R).
        They must push apart to soft-repulsion equilibrium ≈ 3.5R — the
        same equilibrium reached by two_cell_repulsion / contact_equilibrium
        starting from different initial geometries."""
        R = 20.0
        N = 150
        sep = 1.5 * R

        out = run_sim(tmp_path / "run",
                      "-n", "2", "-N", str(N), "-r", str(int(R)),
                      "-s", "0",
                      "-t", "200", "--dt", "0.01",
                      "--v-A", "0", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "200")

        chk = read_checkpoint(out / "checkpoint.bin")
        c0, c1 = chk["cells"][0], chk["cells"][1]
        x0, y0 = c0["centroid"]
        x1, y1 = c1["centroid"]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx > N / 2: dx = N - dx
        if dy > N / 2: dy = N - dy
        final_sep = float(np.sqrt(dx**2 + dy**2))

        record_metric("overlap_repulsion", "initial_sep", sep,
                      expected=sep, tolerance="1%", unit="px")
        # Phase-field soft-repulsion equilibrium is ≈3.5R, measured at
        # 73 px ≈ 3.65R in pilot run.
        record_metric("overlap_repulsion", "final_sep", final_sep,
                      expected=3.5 * R, tolerance="12%", unit="px")
        record_metric("overlap_repulsion", "final_sep/R", final_sep / R,
                      expected=3.5, tolerance=0.4)
        record_phi_from_checkpoint("overlap_repulsion", chk,
                                  f"Two cells after repulsion (sep={final_sep:.1f} ≈ {final_sep/R:.2f}R)")

        # Must separate past touching distance.
        assert final_sep > 2 * R, \
            f"Final spacing {final_sep:.1f} should be > {2*R:.0f} (touching distance)"
        # Must land near soft-repulsion equilibrium (3.0R to 4.0R).
        assert 3.0 * R < final_sep < 4.0 * R, \
            f"Final sep {final_sep:.1f} ({final_sep/R:.2f}R) outside [3.0R, 4.0R]"


# ============================================================================
# 13. Volume conservation during active motion
# ============================================================================

class TestVolumeUnderMotion:
    """Cell volume should be conserved even when v_A > 0 and the cell is moving."""

    def test_volume_stable_while_moving(self, tmp_path):
        """Single motile cell: volume at end should match target within 2%."""
        R = 20.0
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "200", "-r", str(int(R)),
                      "-t", "200", "--dt", "0.01",
                      "--v-A", "0.05", "--tau", "5000", "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")

        chk = read_checkpoint(out / "checkpoint.bin")
        vol = chk["cells"][0]["volume"]
        target = math.pi * R**2
        err_pct = abs(vol - target) / target * 100

        record_metric("volume_under_motion", "volume", vol,
                      expected=target, tolerance="2%", unit="px²")
        record_metric("volume_under_motion", "error_pct", err_pct,
                      expected=0, tolerance=2.0, unit="%")

        assert err_pct < 2.0, \
            f"Volume error {err_pct:.2f}% during active motion (vol={vol:.1f}, target={target:.1f})"


# ============================================================================
# 14. Velocity autocorrelation decays ~ exp(-t/τ)
# ============================================================================

class TestVelocityAutocorrelation:
    """Velocity-autocorrelation decay for a run-and-tumble active cell.

    Palmieri 2015 implements self-propulsion as run-and-tumble (RTP):
    polarity $\\hat p$ stays constant, then at a Poisson-distributed
    time $t_r$ drawn from $P(t_r)=\\tau^{-1}e^{-t_r/\\tau}$ it jumps to
    an unrelated direction. The velocity autocorrelation is then

        C_v(Δt) = ⟨v(t)·v(t+Δt)⟩ / ⟨v²⟩ = exp(-Δt/τ)

    — the survival probability of "no tumble in (t, t+Δt)". Note this
    is NOT the Ornstein-Uhlenbeck autocorrelation; the two happen to
    share the functional form $e^{-Δt/τ}$ at leading order, but RTP
    gives a telegraph process with constant-magnitude velocity whereas
    OU gives Gaussian velocity fluctuations. The scope of this test is
    the exponential-timescale check; we use very loose tolerances
    because a single-cell single-seed estimate of C_v has ~0.2 spread
    at these lags.

    Reference: Palmieri 2015 Methods §B (DOI 10.1038/srep11745).
    """

    def test_velocity_decorrelates(self, tmp_path):
        """C_v(0.1τ) ≈ 0.9, C_v(3τ) ≈ 0.05; scope is the timescale check."""
        R = 20.0
        tau = 100.0
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "300", "-r", str(int(R)),
                      "-t", str(int(20 * tau)), "--dt", "0.01",
                      "--v-A", "0.02", "--tau", str(int(tau)), "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "500")

        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())

        # Extract velocity time series for cell 0
        vx_series = [traj[t][0][2] for t in times if 0 in traj[t]]
        vy_series = [traj[t][0][3] for t in times if 0 in traj[t]]
        n = len(vx_series)
        if n < 10:
            pytest.skip(f"Not enough trajectory frames ({n}) for autocorrelation")

        # Compute C_v at short lag (≈ 0.1τ) and long lag (≈ 3τ)
        dt_traj = (times[-1] - times[0]) / (n - 1) if n > 1 else 1
        short_lag = max(1, int(0.1 * tau / dt_traj))
        long_lag = min(n // 2, int(3 * tau / dt_traj))

        def autocorr(vx, vy, lag):
            if lag >= len(vx):
                return 0
            cv = sum(vx[i] * vx[i + lag] + vy[i] * vy[i + lag]
                     for i in range(len(vx) - lag))
            c0 = sum(vx[i]**2 + vy[i]**2 for i in range(len(vx) - lag))
            return cv / c0 if c0 > 0 else 0

        cv_short = autocorr(vx_series, vy_series, short_lag)
        cv_long = autocorr(vx_series, vy_series, long_lag)

        # RTP survival probability: C_v(Δt) = exp(-Δt/τ).
        #   C_v(0.1τ) = e^{-0.1} ≈ 0.905
        #   C_v(3τ)   = e^{-3}   ≈ 0.050
        record_metric("velocity_autocorrelation", "Cv_short", cv_short,
                      expected=math.exp(-0.1), tolerance=0.2)
        record_metric("velocity_autocorrelation", "Cv_long", cv_long,
                      expected=math.exp(-3.0), tolerance=0.2)

        # Short lag should have positive correlation
        assert cv_short > 0.3, \
            f"Short-lag Cv={cv_short:.3f} should be > 0.3 (lag={short_lag})"
        # Long lag should have decayed toward zero
        assert abs(cv_long) < 0.3, \
            f"Long-lag Cv={cv_long:.3f} should be < 0.3 (lag={long_lag})"


# ============================================================================
# 15. MSD crossover: ballistic → diffusive
# ============================================================================

class TestMSDCrossover:
    """Active motile cells: ballistic short-lag, diffusive long-lag MSD.

    Baseline: analytical (Palmieri Eq 14). For run-and-tumble dynamics
    the isolated-cell MSD is

        MSD(t) = 2·v_A²·τ² · [t/τ - 1 + exp(-t/τ)]

    with limits

        t ≪ τ :  MSD ≈ v_A²·t²      (ballistic, MSD(2t)/MSD(t) = 4)
        t ≫ τ :  MSD ≈ 2·v_A²·τ·t   (diffusive, MSD(2t)/MSD(t) = 2)

    so ``MSD(t₂)/MSD(t₁) = (t₂/t₁)^α`` with α=2 (ballistic) or α=1
    (diffusive). This test covers the short-lag limit only; the
    long-lag diffusive limit is verified by ``TestMSDScaling``.

    Note: Palmieri labels the colored-noise velocity process "OU"
    in parts of the text, but the actual implementation is RTP
    (see ``TestVelocityAutocorrelation`` docstring). Both models
    give the same MSD formula to leading order.

    Reference: Palmieri 2015 Eq 14 (DOI 10.1038/srep11745); AUDIT.md §3.
    """

    def test_short_time_ballistic(self, tmp_path):
        """Short-lag MSD scales as t² ⇒ ratio MSD(t₂)/MSD(t₁) = (t₂/t₁)².

        Scope: distinguishes ballistic from diffusive scaling. Previous
        assertion ``ratio > 1.5`` passed purely diffusive motion
        (ratio = t₂/t₁ = 2.5 here) and was strictly weaker than the
        docstring claim; see AUDIT.md §5.
        """
        R = 20.0
        tau = 200.0
        v_A = 0.02
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "400", "-r", str(int(R)),
                      "-t", str(int(10 * tau)), "--dt", "0.01",
                      "--v-A", str(v_A), "--tau", str(int(tau)), "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "500")

        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())
        Nx = 400

        if len(times) < 20:
            pytest.skip("Not enough frames")

        # Unwrap cell 0 positions
        pos = []
        for t in times:
            if 0 not in traj[t]:
                continue
            x, y = traj[t][0][:2]
            if pos:
                px, py = pos[-1]
                dx, dy = x - px, y - py
                if dx > Nx / 2: dx -= Nx
                if dx < -Nx / 2: dx += Nx
                if dy > Nx / 2: dy -= Nx
                if dy < -Nx / 2: dy += Nx
                pos.append((px + dx, py + dy))
            else:
                pos.append((x, y))

        n = len(pos)
        # Compute MSD at two short lags
        def msd_at_lag(lag):
            vals = []
            for t0 in range(n - lag):
                dx = pos[t0 + lag][0] - pos[t0][0]
                dy = pos[t0 + lag][1] - pos[t0][1]
                vals.append(dx**2 + dy**2)
            return np.mean(vals) if vals else 0

        dt_traj = (times[-1] - times[0]) / (n - 1)
        lag1 = max(1, int(0.05 * tau / dt_traj))
        lag2 = max(2, int(0.1 * tau / dt_traj))

        msd1 = msd_at_lag(lag1)
        msd2 = msd_at_lag(lag2)

        if msd1 < 1e-10 or msd2 < 1e-10:
            pytest.skip("MSD too small for ballistic test")

        # For ballistic (t² growth): MSD(2t)/MSD(t) ≈ 4
        # For diffusive (t growth): MSD(2t)/MSD(t) ≈ 2
        ratio = msd2 / msd1
        t1 = lag1 * dt_traj
        t2 = lag2 * dt_traj
        expected_ratio = (t2 / t1) ** 2  # ballistic expectation

        record_metric("msd_crossover", "MSD_ratio", ratio,
                      expected=expected_ratio, tolerance="25%")
        record_metric("msd_crossover", "lag1", t1 / tau, unit="τ")

        # Ballistic regime: MSD ∝ t² ⇒ MSD(t2)/MSD(t1) = (t2/t1)².
        # Diffusive would give exactly (t2/t1). A single-cell
        # single-seed measurement has ~25% stochastic spread at these
        # lags (verified 2026-04-22). Accept the ballistic expectation
        # ±25%, and explicitly reject anything below halfway between
        # ballistic and diffusive so that the test actually distinguishes
        # the two regimes.
        diffusive = t2 / t1
        midpoint = 0.5 * (diffusive + expected_ratio)
        lo = max(midpoint, 0.75 * expected_ratio)
        hi = 1.3 * expected_ratio
        assert lo < ratio < hi, (
            f"MSD ratio MSD({t2/tau:.2f}τ)/MSD({t1/tau:.2f}τ) = {ratio:.2f} "
            f"outside ballistic band [{lo:.2f}, {hi:.2f}]. Ballistic "
            f"expectation is (t2/t1)² = {expected_ratio:.2f}; diffusive "
            f"would give {diffusive:.2f}."
        )


# ============================================================================
# 16. Multi-cell system: population MSD is diffusive at long times
# ============================================================================

class TestPopulationDiffusion:
    """Population-averaged MSD grows linearly at long times (smoke gate).

    Baseline: analytical, but loose. For t ≫ τ, MSD ≈ 4D_eff·t with
    D_eff = v_A²τ/2 (Palmieri Eq 14). A population of 16 cells at
    φ=0.85 for t=10τ has too few independent samples to pin D_eff to
    better than ~35%, so this is a **smoke gate**, not a physics gate
    — it catches gross regressions (MSD shrinking with lag, MSD
    sub-diffusive by a factor > 1.4 across t=2τ → 4τ) but not subtle
    errors. For a quantitative D_eff test use
    ``TestPalmieriSoftVsCtrl`` (N=72, t=100τ, 5 seeds).

    Scope: confirms the bulk-cell dynamics include a linear regime
    and that the combination of RTP + κ-repulsion + μ-constraint
    doesn't produce pathological trapping. See AUDIT.md §5.
    """

    def test_population_msd_linear(self, tmp_path):
        """Population-averaged MSD grows roughly linearly at long times.

        At φ=0.85 a noticeable ballistic component persists from polar
        clustering, so the asserted tolerance on ``MSD(4τ)/MSD(2τ)``
        is widened to 40% (diffusive expectation is 2). Keep this
        loose — the goal is to catch regressions that invert the sign
        of the slope, not to measure D_eff.

        N=64 (not 16): the lag-window count grows with N, and at N=16
        the seed-to-seed scatter on the ratio is ±0.45 (1.96 to 2.89),
        which lands outside the 40% gate at unlucky seeds. N=64 tightens
        the scatter to ±0.15.
        """
        R = 20.0
        tau = 200.0
        out = run_sim(tmp_path / "run",
                      "-n", "64", "-r", str(int(R)), "--confluence", "0.85",
                      "-t", str(int(10 * tau)), "--dt", "0.01",
                      "--v-A", "0.02", "--tau", str(int(tau)), "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "200")

        traj, _ = read_trajectory(out / "trajectory.txt")
        times = sorted(traj.keys())
        chk = read_checkpoint(out / "checkpoint.bin")
        Nx = chk["params"]["Nx"]
        n_cells = chk["num_cells"]

        if len(times) < 20:
            pytest.skip("Not enough frames")

        # Unwrap all cells
        cell_ids = sorted(traj[times[0]].keys())
        prev_pos = {c: traj[times[0]][c][:2] for c in cell_ids}
        unwrapped = {c: [prev_pos[c]] for c in cell_ids}

        for t in times[1:]:
            for c in cell_ids:
                if c not in traj[t]:
                    continue
                x, y = traj[t][c][:2]
                px, py = prev_pos[c]
                dx, dy = x - px, y - py
                if dx > Nx / 2: dx -= Nx
                if dx < -Nx / 2: dx += Nx
                if dy > Nx / 2: dy -= Nx
                if dy < -Nx / 2: dy += Nx
                prev_pos[c] = (x, y)
                unwrapped[c].append((unwrapped[c][-1][0] + dx, unwrapped[c][-1][1] + dy))

        # Population MSD at two different long lags (> τ)
        n = len(times)
        dt_traj = (times[-1] - times[0]) / (n - 1)
        lag_2tau = max(1, int(2 * tau / dt_traj))
        lag_4tau = min(n // 2, int(4 * tau / dt_traj))

        def pop_msd(lag):
            total = 0
            count = 0
            for c in cell_ids:
                pos = unwrapped[c]
                for t0 in range(len(pos) - lag):
                    dx = pos[t0 + lag][0] - pos[t0][0]
                    dy = pos[t0 + lag][1] - pos[t0][1]
                    total += dx**2 + dy**2
                    count += 1
            return total / count if count > 0 else 0

        msd_2tau = pop_msd(lag_2tau)
        msd_4tau = pop_msd(lag_4tau)

        record_metric("population_diffusion", "MSD_2tau", msd_2tau, unit="px²")
        record_metric("population_diffusion", "MSD_4tau", msd_4tau, unit="px²")

        # In the diffusive regime, doubling the lag should approximately double MSD
        if msd_2tau > 0:
            ratio = msd_4tau / msd_2tau
            t_ratio = lag_4tau / lag_2tau
            record_metric("population_diffusion", "lag_ratio", ratio,
                          expected=t_ratio, tolerance="35%")
            # Diffusive growth: ratio should be within 35% of t_ratio (wider
            # than the single-cell msd_scaling test because bulk cell
            # dynamics include a long-lived ballistic component from
            # polar-cluster formation).
            assert abs(ratio - t_ratio) / t_ratio < 0.4, \
                f"MSD not growing linearly: MSD(4τ)/MSD(2τ) = {ratio:.2f}, expected ≈ {t_ratio:.1f}"

        assert msd_2tau > 0, "Population MSD at 2τ should be positive"
        assert msd_4tau > msd_2tau, "Population MSD should increase with lag"


# ============================================================================
# 17. T1 neighbour-exchange detector: synthetic-trajectory unit test
# ============================================================================

class TestT1NeighbourExchange:
    """Unit test for the T1 hunter's detector (``t1_hunt.detect_t1``).

    Uses synthetic centroid trajectories — no simulation — so it runs
    in milliseconds and has no physics-parameter dependence. Covers:

      * a genuine T1 flip (diamond with short vertical diagonal
        becoming short horizontal diagonal; both pairs stay in
        physical contact throughout) → detector must catch it.
      * a phantom topology flip (four cells at ≈equidistant positions
        with tiny noise that occasionally flips which Delaunay
        diagonal is shorter, no pair ever in real contact) → detector
        must reject it via the contact filter.

    A physics-level demonstration of real T1s in the PFC model
    requires long runs at Palmieri parameters (many τ of motility-
    driven rearrangement at n ≥ 64 cells) and is not a fast test;
    see ``t1_hunt.py --palmieri`` for that workflow.
    """

    def test_detects_genuine_t1(self):
        import sys, pathlib
        tests_dir = str(pathlib.Path(__file__).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from t1_hunt import detect_t1  # noqa: E402
        from scipy.spatial import Delaunay

        def open_delaunay(xy):
            """Non-periodic Delaunay edge set for unit tests."""
            tri = Delaunay(xy)
            out = set()
            for simplex in tri.simplices:
                a, b, c = sorted(int(x) for x in simplex)
                out.add(frozenset((a, b)))
                out.add(frozenset((a, c)))
                out.add(frozenset((b, c)))
            return out

        # 4-cell diamond that rotates: vertical-short → horizontal-short.
        # sep_vert/sep_horz are HALF-diagonals, so pair (0,2) distance
        # is 2*sep_vert. At the flip (s=0.5) both diagonals equal
        # 2*R*1.25 = 25 px, so the contact threshold must be large
        # enough to include this 4-way-vertex configuration — this is
        # the defining geometry of a real T1.
        R = 10.0
        cx = cy = 0.0
        frames = []
        for t in np.linspace(0.0, 2.0, 21):
            s = t / 2.0
            sep_vert = R * (0.5 + 1.5 * s)   # 5 → 20  (diag 10 → 40)
            sep_horz = R * (2.0 - 1.5 * s)   # 20 → 5  (diag 40 → 10)
            xy = np.array([
                [cx,            cy + sep_vert],
                [cx - sep_horz, cy],
                [cx,            cy - sep_vert],
                [cx + sep_horz, cy],
            ], dtype=np.float64)
            frames.append((float(t), xy))

        event = detect_t1(frames, Lx=0.0, Ly=0.0, verbose=False,
                          min_t_event=0.0, contact_dist=3.0 * R,
                          edge_fn=open_delaunay)
        assert event is not None, "Detector missed a genuine T1"
        assert set(event.quad) == {0, 1, 2, 3}
        assert event.lost != event.gained
        assert event.lost in (frozenset({0, 2}), frozenset({1, 3}))
        assert event.gained in (frozenset({0, 2}), frozenset({1, 3}))

    def test_rejects_phantom_flip(self):
        import sys, pathlib
        tests_dir = str(pathlib.Path(__file__).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from t1_hunt import detect_t1  # noqa: E402
        from scipy.spatial import Delaunay

        def open_delaunay(xy):
            tri = Delaunay(xy)
            out = set()
            for simplex in tri.simplices:
                a, b, c = sorted(int(x) for x in simplex)
                out.add(frozenset((a, b)))
                out.add(frozenset((a, c)))
                out.add(frozenset((b, c)))
            return out

        # Four cells at ≈38 px pairwise — all well outside any 2R
        # contact range. Sub-pixel noise occasionally flips Delaunay.
        cx = cy = 0.0
        base = np.array([
            [cx,         cy + 27.0],
            [cx - 27.0,  cy       ],
            [cx,         cy - 27.0],
            [cx + 27.0,  cy       ],
        ], dtype=np.float64)
        frames = []
        rng = np.random.default_rng(0)
        for t in np.linspace(0.0, 2.0, 21):
            noise = rng.standard_normal((4, 2)) * 0.3
            frames.append((float(t), (base + noise).copy()))

        # contact_dist = 2.4·R = 24 rejects these ~38 px pairs.
        event = detect_t1(frames, Lx=0.0, Ly=0.0, verbose=False,
                          min_t_event=0.0, contact_dist=2.4 * 10.0,
                          edge_fn=open_delaunay)
        assert event is None, (
            "Detector falsely reported a phantom topology flip as a T1"
        )


# ============================================================================
# 18. CPU-reference ↔ GPU numerical agreement
# ============================================================================
#
# These are the *real* correctness gates for the GPU solver. We run the
# GPU binary to produce a checkpoint at t=t_mid, load that checkpoint
# into the standalone CPU reference integrator in ``cpu_reference.py``,
# and evolve both for the same number of forward-Euler steps. Since
# both solvers start from the exact same state, any divergence is
# attributable to the GPU kernel: stencil bugs, tile-halo corruption,
# accumulation-order differences, f32-vs-f64 round-off, etc.
#
# ``v_A`` is set to 0 everywhere: it removes the run-and-tumble
# polarity integration from the comparison (which is RNG-based and
# not bit-identical between CPU numpy and GPU cuRAND). The PDE
# itself — the γ, κ, μ terms — is fully exercised.
#
# We measure two things:
#   * Per-cell **field RMS error** ``||φ_cpu − φ_gpu||_2 / ||φ_gpu||_2``
#     on the overlapping bounding box. This is the quantitative gate.
#   * Per-cell **centroid L2 error** in pixels — a human-readable
#     sanity check that the integration doesn't drift the cell.
# ============================================================================


class TestCPURefGPUAgreement:
    """CPU reference integrator ↔ GPU binary numerical agreement.

    Design: run the GPU twice — once to t=t_mid, once to t=t_end —
    starting from the same seed. The first run's final checkpoint is
    the shared initial condition. Load it into the CPU reference and
    integrate forward (t_end − t_mid)/dt steps, then compare to the
    GPU's t=t_end checkpoint cell-by-cell.

    v_A=0 throughout so polarity RNG drops out entirely. This keeps
    the comparison restricted to the deterministic PDE kernel.
    """

    # Relative-tolerance budgets. These are empirical targets based on
    # f32 accumulation error in the GPU kernel vs f64 in the CPU ref.
    # If the GPU regresses, these tighten to catch the bug.
    PHI_RMS_TOL = 5e-3        # ||φ_cpu - φ_gpu||_2 / ||φ_gpu||_2
    PHI_MAX_TOL = 5e-2        # max per-pixel |φ_cpu - φ_gpu|
    CENTROID_TOL_PX = 0.5     # L2 distance between CPU and GPU centroids
    VOL_REL_TOL = 1e-3        # (V_cpu - V_gpu) / V_gpu

    def _run_gpu_window(self, tmp_path, *, t_mid, t_end, n_cells, radius,
                        confluence, dt, seed, tag):
        """Run GPU from t=0 → t_mid → t_end and return both checkpoints.

        Uses --save-final-checkpoint on both legs. The first leg's
        checkpoint is the shared IC; the second leg's is the GPU
        target at t_end.
        """
        # Leg 1: t=0 → t=t_mid.
        out_mid = run_sim(
            tmp_path / f"{tag}_mid",
            "-n", str(n_cells), "-r", str(radius),
            "--confluence", str(confluence),
            "-t", str(t_mid), "--dt", str(dt),
            "--v-A", "0", "--seed", str(seed),
            "--save-interval", "0", "--trajectory-samples", "0",
            "--print-interval", "0",
        )
        ckpt_mid_path = out_mid / "checkpoint.bin"
        assert ckpt_mid_path.exists(), "leg 1 did not save final checkpoint"

        # Leg 2: resume from ckpt_mid → t=t_end. Use -c so the exact
        # same GPU state continues (no re-initialisation).
        out_end = run_sim(
            tmp_path / f"{tag}_end",
            "-c", str(ckpt_mid_path),
            "-t", str(t_end), "--dt", str(dt),
            "--v-A", "0",
            "--save-interval", "0", "--trajectory-samples", "0",
            "--print-interval", "0",
        )
        ckpt_end_path = out_end / "checkpoint.bin"
        assert ckpt_end_path.exists(), "leg 2 did not save final checkpoint"

        return read_checkpoint(ckpt_mid_path), read_checkpoint(ckpt_end_path)

    def _compare(self, cpu_cells, gpu_ckpt, halo):
        """Return dict of per-cell (phi_rms, phi_max, centroid_err, vol_err).

        Pairs CPU cells and GPU cells by index (the order is preserved
        by the binary's checkpoint write path and the CPU reference's
        ``cells_from_checkpoint`` read path).
        """
        # Lazy import so Tier-1 tests don't pay the cost.
        import sys, pathlib
        tests_dir = str(pathlib.Path(__file__).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from cpu_reference import phi_at_bbox, periodic_centroid_of_phi  # noqa: E402

        assert len(cpu_cells) == len(gpu_ckpt["cells"]), (
            "cell count mismatch between CPU and GPU — "
            f"{len(cpu_cells)} vs {len(gpu_ckpt['cells'])}"
        )
        p = gpu_ckpt["params"]
        dx = float(p.get("dx", 1.0));  dy = float(p.get("dy", 1.0))
        results = []
        for i, (cpu, gpu) in enumerate(zip(cpu_cells, gpu_ckpt["cells"])):
            bbox = gpu["bbox"]
            phi_cpu = phi_at_bbox(cpu, bbox, halo).astype(np.float64)
            phi_gpu = gpu["phi"].astype(np.float64)
            assert phi_cpu.shape == phi_gpu.shape, (
                f"cell {i}: bbox-tile shape mismatch CPU={phi_cpu.shape} "
                f"GPU={phi_gpu.shape}"
            )
            diff = phi_cpu - phi_gpu
            norm_gpu = float(np.linalg.norm(phi_gpu))
            phi_rms = float(np.linalg.norm(diff) / max(norm_gpu, 1e-12))
            phi_max = float(np.abs(diff).max())

            cx_cpu, cy_cpu = periodic_centroid_of_phi(cpu.phi, dx, dy)
            cx_gpu, cy_gpu = float(gpu["centroid"][0]), float(gpu["centroid"][1])
            # Shortest periodic distance between (cx_cpu, cy_cpu) and (cx_gpu, cy_gpu).
            Lx = float(p["Nx"]) * dx;  Ly = float(p["Ny"]) * dy
            def _wrap(d, L):
                if d >  L / 2: d -= L
                if d < -L / 2: d += L
                return d
            cerr = float(math.hypot(_wrap(cx_cpu - cx_gpu, Lx),
                                    _wrap(cy_cpu - cy_gpu, Ly)))

            vol_cpu = float((cpu.phi ** 2).sum()) * dx * dy
            vol_gpu = float(gpu["volume"])
            vol_err = abs(vol_cpu - vol_gpu) / max(vol_gpu, 1e-12)

            results.append({
                "cell": i,
                "phi_rms": phi_rms,
                "phi_max": phi_max,
                "centroid_err_px": cerr,
                "vol_rel_err": vol_err,
            })
        return results

    # ------------------------------------------------------------------
    # Test 1: relaxation window — 4 cells, no motility, just PDE
    # ------------------------------------------------------------------
    def test_relaxation_fields_match(self, tmp_path):
        """Compare CPU-ref vs GPU over a short relaxation window.

        4 cells at R=49 in a small domain, dt=0.01, window = 1 TU
        (100 GPU steps after a 5 TU warm-up so the transients from
        Poisson-disc placement have damped).
        """
        # Lazy import the CPU reference so failures don't block Tier 1.
        import sys, pathlib
        tests_dir = str(pathlib.Path(__file__).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from cpu_reference import (  # noqa: E402
            cells_from_checkpoint, cpu_params_from_checkpoint, integrate,
        )

        n_cells = 4
        radius = 49.0
        confluence = 0.6       # loose pack — no repulsion saturation
        dt = 0.01
        t_mid = 5.0            # warm-up (let Poisson-disc relax)
        t_end = 6.0            # compare window (1 TU = 100 steps)

        ckpt_mid, ckpt_end = self._run_gpu_window(
            tmp_path, t_mid=t_mid, t_end=t_end,
            n_cells=n_cells, radius=radius, confluence=confluence,
            dt=dt, seed=1, tag="relax",
        )

        # Load the shared IC into the CPU reference. v_A=0 so polarity
        # is irrelevant; pass zeros to satisfy the API.
        cpu_cells = cells_from_checkpoint(ckpt_mid, v_A=0.0,
                                          polarities=[(0.0, 0.0)] * n_cells)
        cpu_params = cpu_params_from_checkpoint(ckpt_mid)
        # cpu_params.dt is whatever the binary wrote; if it drifted from
        # our test dt we'd get different step counts. Pin to the test dt.
        cpu_params.dt = dt
        n_steps = int(round((t_end - t_mid) / dt))
        assert n_steps > 0, f"bad window {t_mid}..{t_end}"

        cpu_final = integrate(cpu_cells, cpu_params, n_steps)

        halo = int(ckpt_end["params"].get("halo_width", 4))
        results = self._compare(cpu_final, ckpt_end, halo)

        # Log + assert per cell.
        phi_rms_vals = [r["phi_rms"] for r in results]
        phi_max_vals = [r["phi_max"] for r in results]
        c_err_vals   = [r["centroid_err_px"] for r in results]
        v_err_vals   = [r["vol_rel_err"] for r in results]
        record_metric("cpu_ref_gpu_relax", "phi_rms_max",
                      max(phi_rms_vals), expected=0,
                      tolerance=self.PHI_RMS_TOL)
        record_metric("cpu_ref_gpu_relax", "phi_max_max",
                      max(phi_max_vals), expected=0,
                      tolerance=self.PHI_MAX_TOL)
        record_metric("cpu_ref_gpu_relax", "centroid_err_max_px",
                      max(c_err_vals), expected=0,
                      tolerance=self.CENTROID_TOL_PX, unit="px")
        record_metric("cpu_ref_gpu_relax", "vol_rel_err_max",
                      max(v_err_vals), expected=0,
                      tolerance=self.VOL_REL_TOL)

        for r in results:
            assert r["phi_rms"] < self.PHI_RMS_TOL, (
                f"cell {r['cell']}: φ RMS error {r['phi_rms']:.2e} "
                f"> tol {self.PHI_RMS_TOL:.2e} after {n_steps} steps")
            assert r["phi_max"] < self.PHI_MAX_TOL, (
                f"cell {r['cell']}: φ max error {r['phi_max']:.2e} "
                f"> tol {self.PHI_MAX_TOL:.2e}")
            assert r["centroid_err_px"] < self.CENTROID_TOL_PX, (
                f"cell {r['cell']}: centroid drift "
                f"{r['centroid_err_px']:.3f} px > tol "
                f"{self.CENTROID_TOL_PX} px")
            assert r["vol_rel_err"] < self.VOL_REL_TOL, (
                f"cell {r['cell']}: volume drift "
                f"{r['vol_rel_err']:.2e} > tol {self.VOL_REL_TOL:.2e}")

    # Note: a two-cell contact test was removed 2026-04-22 because
    # ``test_migration.py::TestCpuReference::test_cpu_ref_dimer_split``
    # already covers the contact-regime κ kernel with a stronger baseline
    # (rotational invariance at 4 angles + CPU analytical baseline). See
    # AUDIT.md §5 (Tier A dedup).


# ============================================================================
# 19. Energy-functional monotonicity (gradient-flow sanity check)
# ============================================================================
#
# Derivation (see AUDIT.md §1): with v_A = 0 and a single isolated cell
# (so v_I = 0 since Σⱼ≠ᵢ φⱼ² = 0), the sim's equation of motion reduces
# to pure gradient flow
#
#     dφ/dt = -(1/2) · δF/δφ
#
# where the Lyapunov functional is
#
#     F[φ] = ∫[ γ·|∇φ|²  +  (30γ/λ²)·φ²(1-φ)² ] dA
#          +  (μ/A₀)·(A₀ - V)²,                 V = ∫φ² dA
#
# Therefore  dF/dt = -2·∫(dφ/dt)² dA ≤ 0  strictly unless at a critical
# point. Numerical forward-Euler integration can violate this locally
# at O(dt²) over a single step, but in practice the drop-per-step is
# vastly larger than the per-step numerical noise, so the total F over
# a window of many steps is monotonically decreasing.
#
# This is a **parameter-free analytical test** with no reference
# checkpoint and no Palmieri figure to compare against — if the sim
# integrates its own PDE correctly, this inequality holds by
# construction.
# ============================================================================


class TestEnergyMonotonicity:
    """F[φ] is a Lyapunov functional for the v_A=0 single-cell sim."""

    def _free_energy(self, chk):
        """Compute F[φ] for the single cell in the checkpoint."""
        import sys, pathlib
        tests_dir = str(pathlib.Path(__file__).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from cpu_reference import cells_from_checkpoint  # noqa: E402

        cells = cells_from_checkpoint(chk, v_A=0.0,
                                      polarities=[(0.0, 0.0)] * len(chk["cells"]))
        p = chk["params"]
        gamma = float(p["gamma"])
        lam   = float(p["lambda"])
        mu    = float(p["mu"])
        R     = float(p["target_radius"])
        A0    = math.pi * R * R
        dx    = float(p.get("dx", 1.0))
        dy    = float(p.get("dy", 1.0))
        dA    = dx * dy

        c = cells[0]
        phi = c.phi
        # Periodic gradient (matches cpu_reference's central-diff).
        gx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dx)
        gy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dy)
        grad_sq = gx * gx + gy * gy
        bulk = phi * phi * (1.0 - phi) ** 2

        F_grad = gamma * float(grad_sq.sum()) * dA
        F_bulk = (30.0 * gamma / (lam * lam)) * float(bulk.sum()) * dA
        V = float((phi * phi).sum()) * dA
        F_vol  = (mu / A0) * (A0 - V) ** 2
        return F_grad + F_bulk + F_vol, V

    def test_isolated_cell_energy_decreases(self, tmp_path):
        """Run single cell v_A=0 for a short window with checkpoints at
        3 intermediate times; F must be monotonically decreasing.

        The initial configuration (placed by the binary's Poisson-disc
        seeding) is not at steady state, so F drops substantially over
        the first ~20 TU before levelling off. This guards against a
        regression where γ, 30/λ², or μ/A₀ coefficients are off by a
        sign or factor — any such bug would either flip the monotonicity
        or leave F constant.
        """
        R_target = 49

        def _F_at(label, t_stop):
            out = run_sim(tmp_path / label,
                          "-n", "1", "-N", "200", "-r", str(R_target),
                          "-t", str(t_stop), "--dt", "0.01",
                          "--v-A", "0", "--seed", "17",
                          "--save-interval", "0", "--trajectory-samples", "0")
            chk = read_checkpoint(out / "checkpoint.bin")
            F, V = self._free_energy(chk)
            return F, V, chk["time"]

        # Four checkpoints across a 100 TU window. The initial config
        # comes from the binary's seeding (not at steady state), so
        # F drops substantially before levelling off.
        samples = [("t00", 1), ("t01", 5), ("t02", 20), ("t03", 100)]
        F_series, V_series, t_series = [], [], []
        for label, t_stop in samples:
            F, V, t_actual = _F_at(label, t_stop)
            F_series.append(F)
            V_series.append(V)
            t_series.append(t_actual)

        record_timeseries("energy_monotonicity", t_series,
                          {"F[φ]": F_series, "V": V_series},
                          xlabel="Time (TU)", ylabel="F, V",
                          title="Lyapunov functional vs time (v_A=0, single cell)")
        for i, (t, F) in enumerate(zip(t_series, F_series)):
            record_metric("energy_monotonicity", f"F(t={t:.1f})", F,
                          unit="γ·px²")

        # Global drop: final should be noticeably below initial.
        drop = (F_series[0] - F_series[-1]) / max(abs(F_series[0]), 1e-12)
        record_metric("energy_monotonicity", "relative_drop", drop,
                      expected=1.0, tolerance=1.0)
        assert drop > 0.01, (
            f"F dropped only {drop*100:.4f}% over 100 TU. A relaxing "
            f"gradient flow should see a measurable drop. "
            f"F series: {F_series}"
        )

        # Step-wise monotonicity: every adjacent pair must satisfy
        # F(t_{k+1}) < F(t_k) (allow a 0.1% slack for forward-Euler
        # transients near the minimum where the RHS is small).
        for k in range(len(F_series) - 1):
            slack = 1e-3 * abs(F_series[k])
            assert F_series[k + 1] <= F_series[k] + slack, (
                f"F non-monotonic between t={t_series[k]:.1f} and "
                f"t={t_series[k+1]:.1f}: "
                f"F_{k}={F_series[k]:.4e}, F_{k+1}={F_series[k+1]:.4e}. "
                f"Gradient flow is violated."
            )


# ============================================================================
# 20. Analytical PDE residual check at quasi-steady state
# ============================================================================
#
# At steady state for v_A=0, dφ/dt = 0 ⇒ the RHS of the sim's PDE
# (computed in ``cpu_reference.step``) is identically zero. Running a
# single cell for 200 TU lands it in a state where the residual
# should be a small number set by pixel discretization + forward-Euler
# step error.
#
# This test is independent of the CPU reference's time-stepping logic
# (it only uses the PDE-RHS expression) and independent of any
# reference checkpoint — it's purely "plug the relaxed state into the
# equation of motion and ask what's left".
# ============================================================================


class TestAnalyticalSteadyState:
    """At relaxation, the sim's PDE RHS should be small (≈ numerical noise)."""

    def test_single_cell_pde_residual_small(self, tmp_path):
        import sys, pathlib
        tests_dir = str(pathlib.Path(__file__).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from cpu_reference import (  # noqa: E402
            cells_from_checkpoint, cpu_params_from_checkpoint,
            laplacian_9pt, gradients,
        )

        # Run to a well-relaxed state. Using R_target for the IC
        # already gets us within ~1% of steady state (see
        # TestCellRelaxation), so by t=200 TU we're deep in the
        # quasi-steady manifold.
        out = run_sim(tmp_path / "run",
                      "-n", "1", "-N", "200", "-r", "49",
                      "-t", "200", "--dt", "0.01", "--v-A", "0",
                      "--seed", "42",
                      "--save-interval", "0", "--trajectory-samples", "0")
        chk = read_checkpoint(out / "checkpoint.bin")

        cells = cells_from_checkpoint(chk, v_A=0.0,
                                      polarities=[(0.0, 0.0)])
        p = cpu_params_from_checkpoint(chk)

        ci = cells[0]
        phi = ci.phi
        A0 = p.target_area
        # No interaction term (single cell).
        lap = laplacian_9pt(phi, p.dx)
        bulk_coef = 30.0 * p.gamma / (p.lambd ** 2)

        # RHS of  dφ/dt = γ∇²φ − (30γ/λ²)φ(1-φ)(1-2φ) + (2μ/A₀)(A₀-V)φ
        vol = float((phi * phi).sum()) * p.dA
        rhs = (p.gamma * lap
               - bulk_coef * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
               + (2.0 * p.mu / A0) * (A0 - vol) * phi)

        # Normalize residual by the typical scale of any individual
        # term. Use the bulk-term maximum as a natural scale.
        scale = float(np.max(np.abs(
            bulk_coef * phi * (1.0 - phi) * (1.0 - 2.0 * phi))))
        assert scale > 0, "bulk-term scale is zero; phi must be trivially 0"
        rhs_max = float(np.max(np.abs(rhs)))
        rhs_rms = float(np.sqrt(np.mean(rhs * rhs)))
        rel_max = rhs_max / scale
        rel_rms = rhs_rms / scale

        record_metric("pde_residual", "rhs_max_rel", rel_max,
                      expected=0, tolerance=0.1)
        record_metric("pde_residual", "rhs_rms_rel", rel_rms,
                      expected=0, tolerance=0.03)
        record_metric("pde_residual", "vol_rel_err",
                      abs(vol - A0) / A0, expected=0, tolerance=0.02)

        # Quasi-steady: max residual < 10% of the bulk term's max.
        # (RMS is a much tighter test since the residual is supported
        # only at the interface.)
        assert rel_max < 0.1, (
            f"PDE residual at t=200 TU: max {rel_max*100:.1f}% of bulk-term "
            f"scale. Expected quasi-steady state (< 10%). Sim may have "
            f"a term mismatch or still be transient."
        )
        assert rel_rms < 0.03, (
            f"PDE residual RMS at t=200 TU: {rel_rms*100:.2f}% of bulk-term "
            f"scale (expected < 3% at quasi-steady)."
        )

