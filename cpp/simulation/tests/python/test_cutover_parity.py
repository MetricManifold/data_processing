"""GPU sim_v3 ↔ f64 Rust `cpu_ref` parity regression test.

Validates that the GPU simulator reproduces the f64 single-threaded Rust
reference to sub-pixel accuracy over 2τ when running with identical IC
and identical scripted tumble events.

Two views are checked:
  1. Per-frame trajectory drift |Δr|(t) -- assertions on rms/max envelopes.
  2. Final phase-field RMS over the full domain (paint each cell's
     TILE_T tile into Nx*Ny then compare to the cpu_ref final snapshot).

Artifacts (per-frame stats CSV, |Δr|(t) plot, final-phi error map) are
always written to tmp_path; pass `--parity-artifacts DIR` to also mirror
them to a persistent directory.

See `fixtures/cpu_ref_2tau/README.md` for the reference data provenance.
"""
from pathlib import Path
import shutil
import struct

import numpy as np
import pytest

from report import (record_metric, record_timeseries,
                    record_comparison_panel, record_description)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cpu_ref_2tau"
DOMAIN_L = 376.0   # IC has L = 376
TAU = 10000.0
RADIUS = 49.0


def _read_v7_final_phi(path):
    """Read a v7 (sim_v3) checkpoint and paint cells onto a periodic
    (Ny, Nx) grid. Returns the (Ny, Nx) sum-of-phi field. Minimal v7-only
    parser; mirrors `cpu_truth_pkg/read_ckpt.py`."""
    with open(path, "rb") as f:
        magic, version, step = struct.unpack("<III", f.read(12))
        assert magic == 0x43454C4C, f"bad magic {hex(magic)}"
        assert version == 7, f"expected v7, got v{version}"
        time_val = struct.unpack("<d", f.read(8))[0]  # noqa: F841
        n = struct.unpack("<i", f.read(4))[0]
        f.read(16)  # 4 i32 runtime opts
        sp_size = struct.unpack("<I", f.read(4))[0]
        sp_buf = f.read(sp_size)
        Nx = struct.unpack_from("<i", sp_buf, 0)[0]
        Ny = struct.unpack_from("<i", sp_buf, 4)[0]
        T = struct.unpack("<i", f.read(4))[0]
        phi_full = np.zeros((Ny, Nx), dtype=np.float64)
        for _ in range(n):
            f.read(4)                              # cid
            ox, oy = struct.unpack("<2i", f.read(8))
            f.read(20)                             # cx,cy,vx,vy,vol (5 f32)
            phi = np.frombuffer(f.read(T * T * 4), dtype=np.float32).copy()
            phi = phi.reshape(T, T).astype(np.float64)
            ys = (oy + np.arange(T)) % Ny
            xs = (ox + np.arange(T)) % Nx
            phi_full[np.ix_(ys, xs)] += phi
    return phi_full


def _wrap(d, L):
    return d - L * np.round(d / L)


def _frame_pos(frame, cids):
    return np.array([[frame[c][0], frame[c][1]] for c in cids])


def _read_positions(path):
    """Parse `t cid x y ...` (cell_sim) or `# t cid cx cy` (cpu_ref)."""
    data = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 4:
                continue
            t = float(p[0]); cid = int(p[1])
            x, y = float(p[2]), float(p[3])
            data.setdefault(t, {})[cid] = (x, y)
    return data


def _per_frame_drift(gpu_data, cpu_data, n_cells=16):
    """Returns (ts, rms, max_dr, dr_per_cell) over aligned frames."""
    cpu_times = sorted(cpu_data.keys())
    gpu_times = sorted(gpu_data.keys())
    rows = []
    for tg in gpu_times:
        j = min(range(len(cpu_times)), key=lambda i: abs(cpu_times[i] - tg))
        tc = cpu_times[j]
        if abs(tc - tg) > 0.6:
            continue
        cids = sorted(set(gpu_data[tg].keys()) & set(cpu_data[tc].keys()))
        if len(cids) != n_cells:
            continue
        d = _wrap(_frame_pos(gpu_data[tg], cids) - _frame_pos(cpu_data[tc], cids),
                  DOMAIN_L)
        dr = np.linalg.norm(d, axis=1)
        rows.append((tg, dr))
    ts = np.array([r[0] for r in rows])
    dr_pc = np.stack([r[1] for r in rows]) if rows else np.zeros((0, n_cells))
    rms = np.sqrt((dr_pc ** 2).mean(axis=1)) if rows else np.zeros(0)
    mx = dr_pc.max(axis=1) if rows else np.zeros(0)
    return ts, rms, mx, dr_pc


def _final_phi_full(gpu_ckpt_path):
    """Read GPU final v7 checkpoint, return Σφ_i over cells on (Ny, Nx)."""
    return _read_v7_final_phi(gpu_ckpt_path)


def _save_artifacts(out_dir, ts, rms, mx, dr_pc, phi_err_2d, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "parity_stats.npz",
        ts=ts, rms_dr=rms, max_dr=mx, dr_per_cell=dr_pc,
        phi_err_2d=phi_err_2d.astype(np.float32),
        **{k: v for k, v in summary.items() if isinstance(v, (int, float))},
    )
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n"
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    # Plot 1: |Δr|(t)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in range(dr_pc.shape[1]):
        ax.plot(ts / TAU, dr_pc[:, n], color="C3", alpha=0.3, lw=0.7)
    ax.plot(ts / TAU, mx, "k--", lw=1.5, label="max")
    ax.plot(ts / TAU, rms, "k-", lw=2, label="rms")
    ax.set_yscale("log")
    ax.set_xlabel("t / τ"); ax.set_ylabel("|Δr| (sim units)")
    ax.set_title("GPU sim_v3 vs Rust cpu_ref (f64): per-cell drift")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "drift.png", dpi=130); plt.close(fig)
    # Plot 2: final-phi error map
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(phi_err_2d, cmap="RdBu_r",
                   vmin=-np.abs(phi_err_2d).max(),
                   vmax=np.abs(phi_err_2d).max(), origin="lower")
    ax.set_title(f"Final-phi error (GPU - cpu_ref) at t≈20τ\n"
                 f"rms={summary['phi_rms']:.2e}  max|err|={summary['phi_max']:.2e}")
    plt.colorbar(im, ax=ax, label="Δφ")
    fig.tight_layout(); fig.savefig(out_dir / "phi_err.png", dpi=130); plt.close(fig)


@pytest.mark.slow
class TestCutoverParity:
    """Bit-parity regression: GPU sim_v3 ≈ f64 Rust cpu_ref over 2τ.

    Stats reported (also saved as artifacts):
      - rms_max   : max over time of frame-rms |Δr|
      - max_p95   : 95th percentile of frame-max |Δr|
      - max_final : last-frame max |Δr|
      - phi_rms   : full-domain RMS of (φ_GPU - φ_cpu_ref) at t≈20τ
      - phi_max   : full-domain max |φ_GPU - φ_cpu_ref|
    """

    def test_2tau_scripted_events(self, sim, request, tmp_path):
        ic = FIXTURE_DIR / "ic_checkpoint.bin"
        events = FIXTURE_DIR / "events.txt"
        ref_traj_path = FIXTURE_DIR / "ref_trajectory.txt"
        ref_phi_path = FIXTURE_DIR / "ref_final_phi.npz"
        for p in (ic, events, ref_traj_path):
            assert p.exists(), f"missing fixture: {p}"

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
            "--save-final-checkpoint",
            timeout=900,
        )

        # --- Trajectory drift ---
        gpu_data = _read_positions(out / "trajectory.txt")
        cpu_data = _read_positions(ref_traj_path)
        assert len(cpu_data) >= 100 and len(gpu_data) >= 100, \
            f"frame counts: cpu={len(cpu_data)} gpu={len(gpu_data)}"
        ts, rms, mx, dr_pc = _per_frame_drift(gpu_data, cpu_data)
        assert len(ts) >= 100, f"only {len(ts)} aligned frames"
        assert ts[-1] >= 19000, f"GPU run ended at t={ts[-1]:.0f}"

        rms_max = float(rms.max())
        max_p95 = float(np.percentile(mx, 95))
        max_final = float(mx[-1])
        max_any = float(mx.max())

        # --- Final-phi RMS ---
        gpu_ckpt = out / "checkpoint.bin"
        assert gpu_ckpt.exists(), f"no GPU final checkpoint at {gpu_ckpt}"
        phi_gpu = _final_phi_full(gpu_ckpt)              # Σφ_i over cells, (Ny, Nx)
        if ref_phi_path.exists():
            ref = np.load(ref_phi_path)
            phi_cpu = np.asarray(ref["phi"]).sum(axis=0).astype(np.float64)
            assert phi_cpu.shape == phi_gpu.shape, \
                f"phi shape mismatch gpu={phi_gpu.shape} cpu={phi_cpu.shape}"
            phi_err_2d = phi_gpu - phi_cpu
            phi_rms = float(np.sqrt((phi_err_2d ** 2).mean()))
            phi_max = float(np.abs(phi_err_2d).max())
        else:
            phi_err_2d = np.zeros_like(phi_gpu)
            phi_rms = float("nan"); phi_max = float("nan")

        summary = dict(
            rms_max=rms_max, max_p95=max_p95, max_final=max_final, max_any=max_any,
            phi_rms=phi_rms, phi_max=phi_max,
            n_frames=int(len(ts)), t_final=float(ts[-1]),
        )

        # --- Artifacts ---
        art = tmp_path / "parity_artifacts"
        _save_artifacts(art, ts, rms, mx, dr_pc, phi_err_2d, summary)
        persist = request.config.getoption("--parity-artifacts")
        if persist:
            dst = Path(persist)
            dst.mkdir(parents=True, exist_ok=True)
            for f in art.iterdir():
                shutil.copy(f, dst / f.name)
            print(f"\n[parity] artifacts copied to {dst}")
        print("\n[parity] " + "  ".join(f"{k}={v:.4g}"
              if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))

        # --- HTML report integration ---
        nodeid = request.node.nodeid
        record_description(nodeid,
            "GPU sim_v3 vs f64 Rust cpu_ref over 2τ. Trajectory drift |Δr|(t) "
            "plus final phase-field error (Σφᵢ on full domain).")
        record_metric(nodeid, "rms|Δr| envelope", rms_max, expected=0.0,
                      tolerance=0.5, unit="px")
        record_metric(nodeid, "max|Δr| p95", max_p95, expected=0.0,
                      tolerance=0.5, unit="px")
        record_metric(nodeid, "max|Δr| final", max_final, expected=0.0,
                      tolerance=1.0, unit="px")
        record_metric(nodeid, "max|Δr| any-frame", max_any, expected=0.0,
                      tolerance=RADIUS / 5, unit="px")
        if not np.isnan(phi_rms):
            record_metric(nodeid, "phi RMS (final)", phi_rms, expected=0.0,
                          tolerance=5e-2, unit="")
            record_metric(nodeid, "phi max|err| (final)", phi_max,
                          expected=0.0, tolerance=0.7, unit="")
        record_timeseries(nodeid, ts, {"rms|Δr|": rms, "max|Δr|": mx},
                          xlabel="t", ylabel="|Δr| (px)",
                          title="GPU vs cpu_ref drift", ylog=True)
        if not np.isnan(phi_rms):
            record_comparison_panel(nodeid, phi_gpu, phi_gpu - phi_err_2d,
                                    title="Final Σφᵢ: GPU | cpu_ref | |err|")

        # --- Assertions (envelopes from sm_75 + Linux x86_64 Rust f64) ---
        assert rms_max < 0.5, f"rms|Δr| envelope = {rms_max:.3f} > 0.5"
        assert max_p95 < 0.5, f"max|Δr| p95 = {max_p95:.3f} > 0.5"
        assert max_final < 1.0, f"final max|Δr| = {max_final:.3f} > 1.0"
        assert max_any < RADIUS / 5, \
            f"any-frame max|Δr| = {max_any:.3f} > R/5={RADIUS/5:.1f}"
        if not np.isnan(phi_rms):
            # Empirical at TILE_T=320, RTX 4090: phi_rms ~ 1.1e-2 with
            # max|err| ~ 0.14 concentrated at cell-cell interfaces (steep
            # gradients amplify sub-pixel position drift). Thresholds
            # ~5x observed to catch regressions without flapping.
            assert phi_rms < 5e-2, f"final phi RMS = {phi_rms:.3e} > 5e-2"
            assert phi_max < 0.7, f"final phi max|err| = {phi_max:.3f} > 0.7"
