import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import CELL_SIM, read_checkpoint


def _read_vtk_grid(path):
    raw = Path(path).read_bytes()
    marker = b"LOOKUP_TABLE default\n"
    split = raw.find(marker)
    assert split >= 0, "VTK missing LOOKUP_TABLE marker"
    header = raw[:split].decode("ascii", errors="replace")
    dims = None
    for line in header.splitlines():
        if line.startswith("DIMENSIONS"):
            _, nx, ny, _nz = line.split()
            dims = (int(nx), int(ny))
            break
    assert dims is not None, "VTK missing DIMENSIONS"
    nx, ny = dims
    payload = raw[split + len(marker):]
    grid = np.frombuffer(payload, dtype=">f4", count=nx * ny).astype(np.float32)
    assert grid.size == nx * ny
    return grid.reshape(ny, nx)


def _stamp_cells(cells, nx, ny):
    grid = np.zeros((ny, nx), dtype=np.float32)
    for cell in cells:
        ox, oy = cell["origin"]
        phi = cell["phi"]
        for ly in range(phi.shape[0]):
            gy = (oy + ly) % ny
            for lx in range(phi.shape[1]):
                gx = (ox + lx) % nx
                value = phi[ly, lx]
                if value > grid[gy, gx]:
                    grid[gy, gx] = value
    return grid


def test_multi_gpu_vtk_contains_rank1_cells(tmp_path):
    """2-rank VTK output must include cells owned by rank 1."""
    outdir = tmp_path / "mg_vtk"
    cmd = [
        CELL_SIM,
        "--gpus", "2",
        "-n", "32",
        "-N", "640",
        "-r", "20",
        "-t", "0.01",
        "--dt", "0.01",
        "--seed", "12345",
        "--polarity-seed", "67890",
        "--vtk-interval", "1",
        "--trajectory-samples", "0",
        "--save-interval", "0",
        "--print-interval", "0",
        "-o", str(outdir),
    ]
    env = os.environ.copy()
    env.setdefault("CELL_SIM_LOOPBACK_DEVICE", "0")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        unsupported_needles = [
            "without ENABLE_MULTI_GPU",
            "NCCL not found",
            "requested 2 GPUs but only",
            "no CUDA-capable device",
            "CUDA driver",
        ]
        loopback_nccl_unsupported = "LOOPBACK MODE" in output and "invalid usage" in output
        if loopback_nccl_unsupported or any(needle in output for needle in unsupported_needles):
            pytest.skip("active binary/environment does not support multi-GPU loopback")
        pytest.fail(f"multi-GPU VTK smoke failed rc={result.returncode}\n{output}")

    vtk_path = outdir / "output_000001.vtk"
    assert vtk_path.exists(), output
    rank0 = read_checkpoint(outdir / "checkpoint.bin")
    rank1 = read_checkpoint(outdir / "rank1" / "checkpoint.bin")
    grid = _read_vtk_grid(vtk_path)
    ny, nx = grid.shape
    assert nx == rank1["params"]["Nx"]
    assert ny == rank1["params"]["Ny"]

    rank1_slab_cells = [c for c in rank1["cells"] if (c["origin"][1] % ny) >= ny // 2]
    assert rank1_slab_cells, "test seed produced no rank-1 cells with origin.y in rank-1 slab"

    rank0_grid = _stamp_cells(rank0["cells"], nx, ny)
    rank1_grid = _stamp_cells(rank1_slab_cells, nx, ny)
    rank1_dominates = rank1_grid > (rank0_grid + 0.1)
    assert np.any(rank1_dominates), "rank-1 slab cells never dominate rank-0-only composite"

    expected_global = np.maximum(rank0_grid, rank1_grid)
    actual_rank1 = grid[rank1_dominates]
    expected_rank1 = expected_global[rank1_dominates]
    assert np.max(actual_rank1) > 0.5
    np.testing.assert_allclose(actual_rank1, expected_rank1, rtol=1e-5, atol=1e-6)