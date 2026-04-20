"""
Standalone test harness for cell_sim.
No imports from the simulation source — everything is self-contained.
"""
import struct
import subprocess
import os
import math
from pathlib import Path

# Register the report plugin
from report import pytest_sessionfinish, record_metric, record_snapshot, record_phi_from_checkpoint, record_timeseries, record_trajectory, record_skip  # noqa: F401

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Binary location
# ---------------------------------------------------------------------------

def _find_binary():
    """Find the simulation binary to test.

    Precedence:
      1. $SIM_BINARY env var — absolute path to a binary
      2. $SIM_NAME (default: cell_sim_v2) — searches known build dirs
    """
    env = os.environ.get("SIM_BINARY")
    if env:
        if not Path(env).exists():
            pytest.skip(f"SIM_BINARY={env} does not exist")
        return env

    name = os.environ.get("SIM_NAME", "cell_sim_v2")
    # conftest.py lives at <repo>/cpp/simulation/tests/python/conftest.py
    repo = Path(__file__).parents[4]  # data_processing/
    candidates = [
        # sim_v2 builds (preferred on sim-v2 branch)
        repo.parent / "data_processing_v2" / "cpp" / "sim_v2" / "build" / "bin" / "Release" / f"{name}.exe",
        repo.parent / "data_processing_v2" / "cpp" / "sim_v2" / "build" / "bin" / "Release" / name,
        repo / "cpp" / "sim_v2" / "build" / "bin" / "Release" / f"{name}.exe",
        repo / "cpp" / "sim_v2" / "build" / "bin" / "Release" / name,
        # Baseline build
        repo / "cpp" / "simulation" / "build" / "bin" / f"{name}.exe",
        repo / "cpp" / "simulation" / "build" / "bin" / name,
        repo / "cpp" / "simulation" / "build" / "bin" / "Release" / f"{name}.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    pytest.skip(f"Simulation binary ({name}) not found — build first")


# ---------------------------------------------------------------------------
# Markers: 'slow' is deselected by default; opt-in with `--run-slow`.
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run tests marked @pytest.mark.slow")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (skipped unless --run-slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test; use --run-slow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture skip reasons for the HTML report."""
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "setup" and rep.skipped:
        # rep.longrepr for skips is (filename, lineno, reason) or a str
        reason = ""
        lr = rep.longrepr
        if isinstance(lr, tuple) and len(lr) == 3:
            reason = str(lr[2])
        elif lr is not None:
            reason = str(lr)
        # strip leading "Skipped: "
        if reason.startswith("Skipped: "):
            reason = reason[len("Skipped: "):]
        record_skip(item.nodeid, reason or "skipped")


CELL_SIM = _find_binary()


# Detect which CLI flags the binary supports (from --help output).
try:
    _help_result = subprocess.run([CELL_SIM, "-h"], capture_output=True, text=True, timeout=10)
    _HELP_TEXT = (_help_result.stdout or "") + (_help_result.stderr or "")
except Exception:
    _HELP_TEXT = ""


def requires_flag(flag):
    """Pytest decorator: skip test if binary doesn't support `flag`."""
    return pytest.mark.skipif(
        flag not in _HELP_TEXT,
        reason=f"binary does not support {flag}")


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_sim(tmp_path, *args, timeout=120):
    """Run cell_sim with given args, return output directory Path."""
    outdir = tmp_path / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [CELL_SIM] + list(args) + ["-o", str(outdir), "--save-final-checkpoint"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        pytest.fail(f"cell_sim failed (rc={result.returncode}):\n{result.stderr}\n{result.stdout}")
    return outdir


# ---------------------------------------------------------------------------
# Checkpoint reader (standalone, no simulation imports)
# ---------------------------------------------------------------------------

def read_checkpoint(path):
    """Read a checkpoint file and return a dict with header, params, and cells."""
    path = Path(path)
    with open(path, "rb") as f:
        # Header
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x43454C4C, f"Bad magic: {hex(magic)}"
        version = struct.unpack("<I", f.read(4))[0]
        step = struct.unpack("<i", f.read(4))[0]
        # v5+: cur_time is f64 (8 bytes). Earlier versions: f32 (4 bytes).
        if version >= 5:
            time_val = struct.unpack("<d", f.read(8))[0]
        else:
            time_val = struct.unpack("<f", f.read(4))[0]
        num_cells = struct.unpack("<i", f.read(4))[0]

        # Runtime options
        save_interval = struct.unpack("<i", f.read(4))[0]
        checkpoint_interval = struct.unpack("<i", f.read(4))[0]
        trajectory_samples = struct.unpack("<i", f.read(4))[0]
        flags = f.read(4)  # 4 bools

        # SimParams
        if version >= 4:
            sp_size = struct.unpack("<I", f.read(4))[0]
        else:
            sp_size = 72
        sp_buf = f.read(sp_size)

        params = {}
        # Three layouts coexist:
        #   baseline (sp_size=72 or 92): lambda@28, gamma@32, ..., subdomain_padding@68 (f32)
        #   sim_v2 v5 (sp_size=88):      lambda@24, gamma@28, ..., subdomain_padding@56 (f32)
        #   sim_v2 v6 (sp_size=144):     scalars as f64, int fields at end
        if sp_size == 144:
            # sim_v2 v6 layout: 2 ints + 13 doubles + 6 ints + 1 bool + pad
            fields_i = [("Nx", 0), ("Ny", 4)]
            fields_d = [
                ("dx", 8), ("dy", 16),
                ("dt", 24), ("t_end", 32),
                ("lambda", 40), ("gamma", 48),
                ("kappa", 56), ("target_radius", 64),
                ("mu", 72), ("v_A", 80),
                ("xi", 88), ("tau", 96),
                ("subdomain_padding", 104),
            ]
            fields_i2 = [
                ("halo_width", 112),
                ("save_interval", 116),
                ("print_interval", 120),
                ("trajectory_samples", 124),
                ("seed", 128),
                ("polarity_seed", 132),
            ]
            for name, off in fields_i + fields_i2:
                params[name] = struct.unpack_from("<i", sp_buf, off)[0]
            for name, off in fields_d:
                params[name] = struct.unpack_from("<d", sp_buf, off)[0]
            params["abp"] = bool(sp_buf[136])
        elif sp_size == 88:
            # sim_v2 layout
            fields = [
                ("Nx", "i", 0), ("Ny", "i", 4),
                ("dx", "f", 8), ("dy", "f", 12),
                ("dt", "f", 16), ("t_end", "f", 20),
                ("lambda", "f", 24), ("gamma", "f", 28),
                ("kappa", "f", 32), ("target_radius", "f", 36),
                ("mu", "f", 40), ("v_A", "f", 44),
                ("xi", "f", 48), ("tau", "f", 52),
                ("subdomain_padding", "f", 56), ("halo_width", "i", 60),
            ]
            for name, fmt, off in fields:
                params[name] = struct.unpack_from(f"<{fmt}", sp_buf, off)[0]
        elif len(sp_buf) >= 72:
            # baseline layout
            params["Nx"] = struct.unpack_from("<i", sp_buf, 0)[0]
            params["Ny"] = struct.unpack_from("<i", sp_buf, 4)[0]
            params["dx"] = struct.unpack_from("<f", sp_buf, 8)[0]
            params["dy"] = struct.unpack_from("<f", sp_buf, 12)[0]
            params["dt"] = struct.unpack_from("<f", sp_buf, 16)[0]
            params["t_end"] = struct.unpack_from("<f", sp_buf, 20)[0]
            params["lambda"] = struct.unpack_from("<f", sp_buf, 28)[0]
            params["gamma"] = struct.unpack_from("<f", sp_buf, 32)[0]
            params["kappa"] = struct.unpack_from("<f", sp_buf, 36)[0]
            params["target_radius"] = struct.unpack_from("<f", sp_buf, 40)[0]
            params["mu"] = struct.unpack_from("<f", sp_buf, 44)[0]
            params["v_A"] = struct.unpack_from("<f", sp_buf, 48)[0]
            params["xi"] = struct.unpack_from("<f", sp_buf, 52)[0]
            params["tau"] = struct.unpack_from("<f", sp_buf, 56)[0]
            params["halo_width"] = struct.unpack_from("<i", sp_buf, 60)[0]
            params["subdomain_padding"] = struct.unpack_from("<f", sp_buf, 68)[0]
            if len(sp_buf) >= 92:
                params["adhesion_J"] = struct.unpack_from("<f", sp_buf, 88)[0]

        # Cells
        halo = params.get("halo_width", 4)
        cells = []
        for _ in range(num_cells):
            cid = struct.unpack("<i", f.read(4))[0]
            x0, y0, x1, y1 = struct.unpack("<4i", f.read(16))
            cx, cy = struct.unpack("<2f", f.read(8))
            vx, vy = struct.unpack("<2f", f.read(8))
            volume = struct.unpack("<f", f.read(4))[0]
            w = (x1 - x0) + 2 * halo
            h = (y1 - y0) + 2 * halo
            phi = np.frombuffer(f.read(w * h * 4), dtype=np.float32).copy()
            cells.append({
                "id": cid,
                "bbox": (x0, y0, x1, y1),
                "bbox_w": w, "bbox_h": h,
                "centroid": (cx, cy),
                "velocity": (vx, vy),
                "volume": volume,
                "phi": phi.reshape(h, w),
            })

        # Optional per-cell arrays
        per_cell = {}
        for name, magic_val in [("v_A", 0x56415F41), ("gamma", 0x47414D41), ("radius", 0x52414449)]:
            pos = f.tell()
            raw = f.read(4)
            if len(raw) < 4:
                break
            m = struct.unpack("<I", raw)[0]
            if m == magic_val:
                count = struct.unpack("<i", f.read(4))[0]
                data = np.frombuffer(f.read(count * 4), dtype=np.float32).copy()
                per_cell[name] = data
            else:
                f.seek(pos)  # seek back, try next

        return {
            "version": version,
            "step": step,
            "time": time_val,
            "num_cells": num_cells,
            "params": params,
            "cells": cells,
            "per_cell": per_cell,
        }


# ---------------------------------------------------------------------------
# Trajectory reader
# ---------------------------------------------------------------------------

def read_trajectory(path):
    """Read trajectory.txt, return dict of {time: {cell_id: (x, y, vx, vy, ...)}}."""
    path = Path(path)
    data = {}
    header_params = {}
    with open(path) as f:
        for line in f:
            if line.startswith("# v_A=") or line.startswith("# v_A ="):
                for kv in line.strip("# \n").split():
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        header_params[k] = v
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            t = float(parts[0])
            cid = int(parts[1])
            x, y = float(parts[2]), float(parts[3])
            vx, vy = float(parts[4]), float(parts[5])
            px = float(parts[6]) if len(parts) > 6 else 0.0
            py = float(parts[7]) if len(parts) > 7 else 0.0
            if t not in data:
                data[t] = {}
            data[t][cid] = (x, y, vx, vy, px, py)
    return data, header_params


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim(tmp_path):
    """Fixture that returns a run_sim helper bound to tmp_path."""
    def _run(*args, **kwargs):
        return run_sim(tmp_path, *args, **kwargs)
    return _run
