"""
CPU reference implementation of the sim_v2 phase-field PDE.

This module exists so Phase-H tests have an auditable, independent
ground truth for the core equation sim_v2 solves. It is deliberately
single-file and uses the same data layout as sim_v2 (per-cell tile
buffers with a halo border) so the math can be reviewed line-by-line
against ``cpp/sim_v2/src/kernels.cu`` :: ``k_fused``.

Equation solved per interior pixel of cell i's tile:

    dφᵢ/dt = γ ∇²φᵢ
             − (30 γ / λ²) · φᵢ · (1 − φᵢ) · (1 − 2φᵢ)         (bulk double-well)
             + (2 μ / A₀) · (A₀ − Vᵢ) · φᵢ                     (volume constraint)
             − (30 κ / λ²) · φᵢ · Σⱼ∈ℕᵢ φⱼ²                    (soft repulsion)
             − (vₓ · ∂φᵢ/∂x + v_y · ∂φᵢ/∂y)                    (advection)

which sim_v2 writes as ``np = pv + dt·(−0.5·var_deriv − advection)`` with
``var_deriv = −2γ·lap + 60γ/λ²·bulk + −4·(μ/A₀)·(A₀−V)·pv + 60κ/λ²·pv·S``.

Tile conventions (match sim_v2):
  * Each cell owns an ``(h, w)`` float tile ``phi``, with a ``halo`` pixel
    border held at 0. Only the inner region
    ``(halo ≤ lx < w−halo, halo ≤ ly < h−halo)`` is updated.
  * Cell i's tile is anchored at global ``(OX[i], OY[i])``; tile-local
    ``(lx, ly)`` ↔ global ``(OX[i]+lx, OY[i]+ly)``.
  * Laplacian/gradient stencils CLAMP at tile edges
    (``max(lx-1,0)``/``min(lx+1,w-1)``) — matches sim_v2 kernel.
  * Neighbor sum S at tile-local ``(lx, ly)`` of cell i: for each
    neighbor j, the corresponding tile-local position in j's tile is
    ``(nlx, nly) = (lx - (OX[j]-OX[i]), ly - (OY[j]-OY[i]))`` with the
    integer delta reduced into ``[−Nx/2, Nx/2)``. Contribution included
    only when ``(nlx, nly)`` is in j's INNER region.

Provenance anchor (sim_v2 ``k_fused`` lines ~380-410):
    bulk       = tgb · pv · (1 − pv) · (1 − 2 pv)           (tgb = 60γ/λ²)
    constraint = −4 · vc · vd · pv                          (vc = μ/A₀, vd = A₀−V)
    repulsion  = two_keff · pv · S                          (two_keff = 60κ/λ²)
    var_deriv  = −tg · lap + bulk + constraint + repulsion  (tg = 2γ)
    np         = pv + dt · (−0.5 · var_deriv − (vx·gx + vy·gy))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass
class CPUParams:
    """Subset of sim_v2 ``SimParams`` needed by the reference."""
    Nx: int
    Ny: int
    dx: float = 1.0
    dy: float = 1.0
    dt: float = 0.01
    lambd: float = 7.0
    gamma: float = 1.0
    kappa: float = 10.0
    mu: float = 1.0
    target_radius: float = 20.0
    halo: int = 4

    @property
    def target_area(self) -> float:
        return float(np.pi * self.target_radius ** 2)

    @property
    def dA(self) -> float:
        return self.dx * self.dy


# ---------------------------------------------------------------------------
# Per-cell state (mirrors sim_v2 ``CellArrays``)
# ---------------------------------------------------------------------------

@dataclass
class CPUCell:
    phi: np.ndarray      # (h, w), float64; halo pixels held at 0
    ox: int
    oy: int
    vx: float = 0.0
    vy: float = 0.0
    vol: float = 0.0

    @property
    def h(self) -> int:
        return self.phi.shape[0]

    @property
    def w(self) -> int:
        return self.phi.shape[1]


# ---------------------------------------------------------------------------
# Clamp-stencil helpers (operate on a single tile)
# ---------------------------------------------------------------------------

def _clamped_neighbors(phi: np.ndarray):
    """Return (E, W, N, S, NE, NW, SE, SW) with clamp-at-edge BCs.

    Mirrors sim_v2 ``max(srx-1, 0)`` / ``min(srx+1, old_w-1)`` exactly.
    """
    E  = np.empty_like(phi); E[:,  :-1] = phi[:, 1:];   E[:, -1]  = phi[:, -1]
    W  = np.empty_like(phi); W[:,  1:]  = phi[:, :-1];  W[:,  0]  = phi[:,  0]
    N  = np.empty_like(phi); N[:-1, :]  = phi[1:, :];   N[-1, :]  = phi[-1, :]
    S  = np.empty_like(phi); S[1:, :]   = phi[:-1, :];  S[0,  :]  = phi[0,  :]
    NE = np.empty_like(phi); NE[:-1, :-1] = phi[1:, 1:]
    NE[-1, :] = N[-1, :]; NE[:, -1] = E[:, -1]
    NW = np.empty_like(phi); NW[:-1, 1:]  = phi[1:, :-1]
    NW[-1, :] = N[-1, :]; NW[:,  0] = W[:,  0]
    SE = np.empty_like(phi); SE[1:,  :-1] = phi[:-1, 1:]
    SE[0,  :] = S[0,  :]; SE[:, -1] = E[:, -1]
    SW = np.empty_like(phi); SW[1:,  1:]  = phi[:-1, :-1]
    SW[0,  :] = S[0,  :]; SW[:,  0] = W[:,  0]
    return E, W, N, S, NE, NW, SE, SW


def laplacian_9pt_clamped(phi: np.ndarray, h: float) -> np.ndarray:
    """9-point isotropic Laplacian with clamp-at-edge BCs (tile-local)."""
    E, W, N, S, NE, NW, SE, SW = _clamped_neighbors(phi)
    return (4.0 * (E + W + N + S) + (NE + NW + SE + SW) - 20.0 * phi) / (6.0 * h * h)


def gradients_clamped(phi: np.ndarray, dx: float, dy: float):
    """Central-difference gradients with clamp-at-edge BCs (tile-local)."""
    E, W, N, S, *_ = _clamped_neighbors(phi)
    gx = (E - W) / (2.0 * dx)
    gy = (N - S) / (2.0 * dy)
    return gx, gy


def _pdelta_int(d: int, L: int) -> int:
    """Reduce signed integer delta into [−L/2, L/2). Matches sim_v2."""
    if d >  L // 2: d -= L
    if d < -L // 2: d += L
    return d


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def step(cells: List[CPUCell], p: CPUParams) -> List[CPUCell]:
    """Advance all cells by one ``dt``. Returns a new list of CPUCell."""
    halo = p.halo
    tg = 2.0 * p.gamma
    tgb = 60.0 * p.gamma / (p.lambd ** 2)
    two_keff = 60.0 * p.kappa / (p.lambd ** 2)
    vc = p.mu / p.target_area

    out: List[CPUCell] = []
    for i, ci in enumerate(cells):
        w, h = ci.w, ci.h
        phi = ci.phi
        vd = p.target_area - ci.vol

        lap = laplacian_9pt_clamped(phi, p.dx)
        gx, gy = gradients_clamped(phi, p.dx, p.dy)

        # Inner-region mask (sim_v2: halo ≤ lx < w−halo && halo ≤ ly < h−halo).
        inner = np.zeros((h, w), dtype=bool)
        inner[halo:h - halo, halo:w - halo] = True

        # ---- Neighbor interaction S(lx, ly) ----
        # For each neighbor j, every tile-local (lx, ly) inside i's inner
        # region maps to (nlx, nly) = (lx − dx_int, ly − dy_int) in j's
        # tile. Contribution φⱼ²(nlx, nly) included only when that
        # position is in j's INNER region.
        S = np.zeros((h, w), dtype=np.float64)
        for j, cj in enumerate(cells):
            if j == i:
                continue
            dx_int = _pdelta_int(cj.ox - ci.ox, p.Nx)
            dy_int = _pdelta_int(cj.oy - ci.oy, p.Ny)
            nw, nh = cj.w, cj.h
            # Valid i-tile lx range: halo ≤ lx < w−halo AND halo ≤ lx−dx < nw−halo
            lx_lo = max(halo,       dx_int + halo)
            lx_hi = min(w - halo,   dx_int + nw - halo)
            ly_lo = max(halo,       dy_int + halo)
            ly_hi = min(h - halo,   dy_int + nh - halo)
            if lx_lo >= lx_hi or ly_lo >= ly_hi:
                continue
            nly_lo = ly_lo - dy_int; nly_hi = ly_hi - dy_int
            nlx_lo = lx_lo - dx_int; nlx_hi = lx_hi - dx_int
            phi_j_block = cj.phi[nly_lo:nly_hi, nlx_lo:nlx_hi]
            S[ly_lo:ly_hi, lx_lo:lx_hi] += phi_j_block * phi_j_block

        # ---- PDE update ----
        bulk       = tgb * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
        constraint = -4.0 * vc * vd * phi
        repulsion  = two_keff * phi * S
        var_deriv  = -tg * lap + bulk + constraint + repulsion
        advection  = ci.vx * gx + ci.vy * gy
        phi_new    = phi + p.dt * (-0.5 * var_deriv - advection)

        # Halo pixels zeroed (sim_v2: ``if (!inner) np = 0.0f``).
        phi_new = np.where(inner, phi_new, 0.0)

        vol_new = float((phi_new * phi_new).sum() * p.dA)

        out.append(CPUCell(
            phi=phi_new,
            ox=ci.ox, oy=ci.oy,
            vx=ci.vx, vy=ci.vy,
            vol=vol_new,
        ))

    return out


def integrate(cells: List[CPUCell], p: CPUParams, n_steps: int) -> List[CPUCell]:
    """Advance ``n_steps`` forward-Euler steps."""
    state = cells
    for _ in range(n_steps):
        state = step(state, p)
    return state


# ---------------------------------------------------------------------------
# Checkpoint → CPUCell list
# ---------------------------------------------------------------------------

def cells_from_checkpoint(ckpt: dict, halo: int = 4) -> List[CPUCell]:
    """Build a ``CPUCell`` list from ``conftest.read_checkpoint`` output.

    The checkpoint stores the full tile (halo-included) at its raw tile
    offset ``(ox, oy) = (x0 − halo, y0 − halo)``. Velocities/volumes
    come from the checkpoint so advection and constraint terms match the
    state sim_v2 would see on resume.
    """
    cells: List[CPUCell] = []
    for c in ckpt["cells"]:
        phi_tile = c["phi"].astype(np.float64)
        x0 = int(c["bbox"][0]); y0 = int(c["bbox"][1])
        vx, vy = c.get("velocity", (0.0, 0.0))
        cells.append(CPUCell(
            phi=phi_tile,
            ox=x0 - halo,
            oy=y0 - halo,
            vx=float(vx), vy=float(vy),
            vol=float(c.get("volume", 0.0)),
        ))
    return cells


def cpu_params_from_checkpoint(ckpt: dict) -> CPUParams:
    p = ckpt["params"]
    return CPUParams(
        Nx=int(p["Nx"]), Ny=int(p["Ny"]),
        dx=float(p.get("dx", 1.0)), dy=float(p.get("dy", 1.0)),
        dt=float(p["dt"]),
        lambd=float(p["lambda"]),
        gamma=float(p["gamma"]),
        kappa=float(p["kappa"]),
        mu=float(p["mu"]),
        target_radius=float(p["target_radius"]),
        halo=int(p.get("halo_width", 4)),
    )
