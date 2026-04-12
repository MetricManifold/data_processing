"""
Standalone test harness for cell_sim.
No imports from the simulation source — everything is self-contained.
"""
import struct
import subprocess
import os
import math
from pathlib import Path

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Binary location
# ---------------------------------------------------------------------------

def _find_binary():
    """Find cell_sim executable."""
    candidates = [
        Path(__file__).parents[2] / "build" / "bin" / "cell_sim.exe",
        Path(__file__).parents[2] / "build" / "bin" / "cell_sim",
        Path(__file__).parents[2] / "build" / "bin" / "Release" / "cell_sim.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    pytest.skip("cell_sim binary not found — build first")


CELL_SIM = _find_binary()


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
        if len(sp_buf) >= 72:
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
            if t not in data:
                data[t] = {}
            data[t][cid] = (x, y, vx, vy)
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
