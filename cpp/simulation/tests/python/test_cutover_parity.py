"""GPU sim_v3 ↔ f64 Rust `cpu_ref` parity regression test.

Validates that the GPU simulator reproduces the f64 single-threaded Rust
reference to sub-pixel accuracy over 2τ when running with identical IC
and identical scripted tumble events.

See `fixtures/cpu_ref_2tau/README.md` for the reference data provenance.
"""
from pathlib import Path

import numpy as np
import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cpu_ref_2tau"
DOMAIN_L = 376.0   # IC has L = 376
TAU = 10000.0
RADIUS = 49.0


def _wrap(d, L):
    return d - L * np.round(d / L)


def _frame_pos(frame, cids):
    """Frame dict {cid: (x, y, ...)} → (N, 2) position array."""
    return np.array([[frame[c][0], frame[c][1]] for c in cids])


def _read_positions(path):
    """Minimal trajectory-position parser.

    Accepts either the cell_sim format (`t cid x y vx vy px py ...`) or
    the Rust cpu_ref format (`# t cid cx cy` with 4 columns). Returns
    {time: {cid: (x, y)}}.
    """
    data = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 4:
                continue
            t = float(p[0])
            cid = int(p[1])
            x, y = float(p[2]), float(p[3])
            data.setdefault(t, {})[cid] = (x, y)
    return data


@pytest.mark.slow
class TestCutoverParity:
    """Bit-parity regression: GPU sim_v3 ≈ f64 Rust cpu_ref over 2τ."""

    def test_2tau_scripted_events(self, sim):
        ic = FIXTURE_DIR / "ic_checkpoint.bin"
        events = FIXTURE_DIR / "events.txt"
        ref_traj_path = FIXTURE_DIR / "ref_trajectory.txt"
        for p in (ic, events, ref_traj_path):
            assert p.exists(), f"missing fixture: {p}"

        # Match the reference's 400-sample grid (every 50 sim units).
        out = sim(
            "-c", str(ic),
            "--scripted-events", str(events),
            "--v-A", "0.01",
            "--tau", "10000",
            "-t", "20000",
            "--trajectory-samples", "400",
            "--checkpoint-interval", "0",
            "--save-interval", "0",
            "--print-interval", "0",
            timeout=900,  # ~5 min on Turing GPU; allow 15 min cushion
        )

        gpu_data = _read_positions(out / "trajectory.txt")
        cpu_data = _read_positions(ref_traj_path)

        cpu_times = sorted(cpu_data.keys())
        gpu_times = sorted(gpu_data.keys())
        assert len(cpu_times) >= 100, f"reference has only {len(cpu_times)} frames"
        assert len(gpu_times) >= 100, f"GPU produced only {len(gpu_times)} frames"

        rms_per_frame, max_per_frame, ts = [], [], []
        for tg in gpu_times:
            j = min(range(len(cpu_times)), key=lambda i: abs(cpu_times[i] - tg))
            tc = cpu_times[j]
            if abs(tc - tg) > 0.6:
                continue
            gframe, cframe = gpu_data[tg], cpu_data[tc]
            cids = sorted(set(gframe.keys()) & set(cframe.keys()))
            if len(cids) != 16:
                continue
            d = _wrap(_frame_pos(gframe, cids) - _frame_pos(cframe, cids), DOMAIN_L)
            dr = np.linalg.norm(d, axis=1)
            rms_per_frame.append(float(np.sqrt((dr ** 2).mean())))
            max_per_frame.append(float(dr.max()))
            ts.append(tg)

        ts = np.asarray(ts)
        rms_per_frame = np.asarray(rms_per_frame)
        max_per_frame = np.asarray(max_per_frame)
        assert len(ts) >= 100, f"only {len(ts)} aligned frames"
        assert ts[-1] >= 19000, f"GPU run ended at t={ts[-1]:.0f} (expected ≥19000)"

        # Empirical baseline (sm_75 + Linux x86_64 Rust f64): rms ≤ 0.12,
        # max ≤ 0.27 (steady) with rare spikes to ~1.1 at tumble PBC seams.
        # Thresholds chosen ~5× the observed values.
        rms_max = rms_per_frame.max()
        max_p95 = float(np.percentile(max_per_frame, 95))
        max_final = float(max_per_frame[-1])

        assert rms_max < 0.5, (
            f"rms|Δr| envelope = {rms_max:.3f} sim units exceeds 0.5 "
            f"(GPU has drifted from f64 Rust reference)"
        )
        assert max_p95 < 0.5, (
            f"95th-percentile max|Δr| = {max_p95:.3f} sim units exceeds 0.5"
        )
        # Final-frame steady-state max (no event at exactly t=20000).
        assert max_final < 1.0, (
            f"final-frame max|Δr| = {max_final:.3f} sim units exceeds 1.0"
        )
        # Sanity: drift must be far below cell radius.
        assert max_per_frame.max() < RADIUS / 5, (
            f"any-frame max|Δr| = {max_per_frame.max():.3f} exceeds R/5 = {RADIUS/5:.1f}"
        )
