"""
CPU reference implementation of the cell-simulation phase-field PDE.

This module is the "naive, correct" ground truth against which the
production simulator is validated. Design priorities, in order:

  1. **Readability.** Each cell owns a full ``(Ny, Nx)`` field on the
     periodic domain. No tiles, no bboxes, no halos, no REMAP/RESIZE
     bookkeeping. The only thing on the page is the physics.
  2. **Traceability.** The lines below the ``var_deriv = ...`` comment
     map one-to-one with the production kernel, so a reviewer can
     diff them side-by-side.
  3. **Correctness over speed.** A 1000-step run of a small 200×200
     case takes a couple of seconds — fine for Phase-H tests, we
     never scale this.

Equation solved per cell ``i`` on the full domain ``(x, y)``:

    dφᵢ/dt = γ ∇²φᵢ
             − (30 γ / λ²) · φᵢ · (1 − φᵢ) · (1 − 2φᵢ)    (bulk double-well)
             + (2 μ / A₀) · (A₀ − Vᵢ) · φᵢ                (volume constraint)
             − (30 κ / λ²) · φᵢ · Σⱼ≠ᵢ φⱼ²                (soft repulsion)
             − (vₓ · ∂φᵢ/∂x + v_y · ∂φᵢ/∂y)                (advection)

written in the production kernel as
``np = pv + dt·(−0.5·var_deriv − advection)`` with

    var_deriv = −2γ·lap + 60γ/λ²·bulk + −4·(μ/A₀)·(A₀−V)·pv + 60κ/λ²·pv·S

The velocity ``(vₓ, v_y)`` is updated at end of step to

    vₓ = mc · Σ_domain(pv · gₓ · S) · dA + v_A · pₓ        (mc = 60κ/(ξ λ²))

(and similarly for y), so the next step's advection uses the repulsion-
driven momentum plus the active self-propulsion ``v_A·p̂``. The run-
and-tumble polarity ``p̂`` is kept fixed here — for parity tests we
run with ``tau`` large enough that no tumble fires during the window.

Boundary conditions: periodic on the full domain. The production
simulator uses clamp-at-tile-edge stencils, but the cell's support
is far from the tile edge in every test, so clamp and periodic are
equivalent to f32 round-off.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass
class CPUParams:
    """Subset of the production ``SimParams`` needed by the reference."""
    Nx: int
    Ny: int
    dx: float = 1.0
    dy: float = 1.0
    dt: float = 0.01
    lambd: float = 7.0
    gamma: float = 1.0
    kappa: float = 10.0
    mu: float = 1.0
    xi: float = 1500.0
    target_radius: float = 20.0

    @property
    def target_area(self) -> float:
        return float(np.pi * self.target_radius ** 2)

    @property
    def dA(self) -> float:
        return self.dx * self.dy

    @property
    def motility_coeff(self) -> float:
        """``mc = 60·κ / (ξ·λ²)`` — matches ``SimParams::motility_coeff``."""
        return 60.0 * self.kappa / (self.xi * self.lambd ** 2)


# ---------------------------------------------------------------------------
# Per-cell state — one full-domain field per cell
# ---------------------------------------------------------------------------

@dataclass
class CPUCell:
    phi: np.ndarray      # (Ny, Nx) float64, periodic domain, no halo
    vx: float = 0.0      # advection velocity in x (includes v_A·pₓ carry-over)
    vy: float = 0.0
    vol: float = 0.0     # Σφ² · dA
    v_A: float = 0.0     # active motility magnitude (per cell)
    px: float = 0.0      # run-and-tumble polarity, held fixed during integrate()
    py: float = 0.0


# ---------------------------------------------------------------------------
# Periodic stencils
# ---------------------------------------------------------------------------

def laplacian_9pt(phi: np.ndarray, h: float) -> np.ndarray:
    """9-point isotropic Laplacian, periodic BCs."""
    E  = np.roll(phi, -1, axis=1)
    W  = np.roll(phi,  1, axis=1)
    N  = np.roll(phi, -1, axis=0)
    S  = np.roll(phi,  1, axis=0)
    NE = np.roll(E,   -1, axis=0)
    SE = np.roll(E,    1, axis=0)
    NW = np.roll(W,   -1, axis=0)
    SW = np.roll(W,    1, axis=0)
    return (4.0 * (E + W + N + S) + (NE + NW + SE + SW) - 20.0 * phi) / (6.0 * h * h)


def gradients(phi: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference gradients, periodic BCs."""
    gx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dx)
    gy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dy)
    return gx, gy


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def step(cells: List[CPUCell], p: CPUParams) -> List[CPUCell]:
    """Advance all cells by one ``dt``. Returns a new list of CPUCell."""
    tg  = 2.0 * p.gamma
    tgb = 60.0 * p.gamma / (p.lambd ** 2)
    two_keff = 60.0 * p.kappa / (p.lambd ** 2)
    vc = p.mu / p.target_area
    mc = p.motility_coeff

    # Σⱼ φⱼ² on the full domain, computed once per step.
    psq = [c.phi * c.phi for c in cells]
    S_total = np.sum(psq, axis=0) if psq else np.zeros_like(cells[0].phi)

    # --- Pass 1: compute velocity v^n from φ^n for ALL cells ---
    # v must be computed before the phi update so advection uses v(φ^n).
    velocities = []
    for i, ci in enumerate(cells):
        phi = ci.phi
        S = S_total - psq[i]
        gx, gy = gradients(phi, p.dx, p.dy)
        vx_int = mc * float((phi * gx * S).sum()) * p.dA
        vy_int = mc * float((phi * gy * S).sum()) * p.dA
        vx_n = vx_int + ci.v_A * ci.px
        vy_n = vy_int + ci.v_A * ci.py
        velocities.append((vx_n, vy_n))

    # --- Pass 2: PDE update using the freshly computed v^n ---
    out: List[CPUCell] = []
    for i, ci in enumerate(cells):
        phi = ci.phi
        S = S_total - psq[i]
        vd = p.target_area - ci.vol

        lap = laplacian_9pt(phi, p.dx)
        gx, gy = gradients(phi, p.dx, p.dy)

        bulk       = tgb * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
        constraint = -4.0 * vc * vd * phi
        repulsion  = two_keff * phi * S
        var_deriv  = -tg * lap + bulk + constraint + repulsion

        vx_n, vy_n = velocities[i]
        advection  = vx_n * gx + vy_n * gy
        phi_new    = phi + p.dt * (-0.5 * var_deriv - advection)

        vol_new = float((phi_new * phi_new).sum()) * p.dA

        out.append(CPUCell(
            phi=phi_new,
            vx=vx_n, vy=vy_n,
            vol=vol_new,
            v_A=ci.v_A, px=ci.px, py=ci.py,
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

def cells_from_checkpoint(
    ckpt: dict,
    *,
    v_A: Optional[float] = None,
    polarities: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[CPUCell]:
    """Build CPUCells from a production checkpoint.

    The checkpoint stores each cell's φ as a halo-padded tile anchored
    at ``bbox``. We paint it into a zero-initialized full ``(Ny, Nx)``
    array — the halo pixels are 0 so they leave the background
    unchanged. All other per-cell fields (velocity, volume) are read
    through so the first step after resume sees the same state as the
    production simulator.

    ``v_A`` (scalar, applied to all cells) and ``polarities`` (list of
    ``(pₓ, p_y)``, per cell) are optional because the checkpoint does
    not persist polarity. For v_A parity tests the caller should
    extract the polarity from the first trajectory snapshot.
    """
    Nx = int(ckpt["params"]["Nx"])
    Ny = int(ckpt["params"]["Ny"])
    halo = int(ckpt["params"].get("halo_width", 4))
    n = len(ckpt["cells"])

    if polarities is None:
        polarities = [(0.0, 0.0)] * n
    assert len(polarities) == n, \
        f"polarities length {len(polarities)} != num_cells {n}"

    cells: List[CPUCell] = []
    for idx, c in enumerate(ckpt["cells"]):
        phi_tile = c["phi"].astype(np.float64)        # (h_t, w_t) w/ halo
        x0 = int(c["bbox"][0]);  y0 = int(c["bbox"][1])
        ox, oy = x0 - halo, y0 - halo                  # tile origin in global px
        h_t, w_t = phi_tile.shape

        full = np.zeros((Ny, Nx), dtype=np.float64)
        ys = (oy + np.arange(h_t)) % Ny
        xs = (ox + np.arange(w_t)) % Nx
        full[np.ix_(ys, xs)] = phi_tile

        vx, vy = c.get("velocity", (0.0, 0.0))
        px, py = polarities[idx]
        cells.append(CPUCell(
            phi=full,
            vx=float(vx), vy=float(vy),
            vol=float(c.get("volume", 0.0)),
            v_A=float(v_A) if v_A is not None else 0.0,
            px=float(px), py=float(py),
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
        xi=float(p.get("xi", 1500.0)),
        target_radius=float(p["target_radius"]),
    )


# ---------------------------------------------------------------------------
# Comparison helpers (full-domain aware)
# ---------------------------------------------------------------------------

def centroid_of_phi(phi: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> Tuple[float, float]:
    """Σ(x·φ²) / Σ(φ²) with non-periodic indexing — good enough for
    tests where support is compact and far from the periodic edges.
    """
    psq = phi * phi
    total = float(psq.sum())
    if total <= 0:
        return (float("nan"), float("nan"))
    h, w = psq.shape
    ys = np.arange(h, dtype=np.float64) * dy
    xs = np.arange(w, dtype=np.float64) * dx
    cx = float((xs[None, :] * psq).sum() / total)
    cy = float((ys[:, None] * psq).sum() / total)
    return cx, cy


def periodic_centroid_of_phi(phi: np.ndarray,
                             dx: float = 1.0, dy: float = 1.0) -> Tuple[float, float]:
    """Periodic-aware centroid using the circular-mean trick.

    For a φ² mass distribution on a periodic grid of shape (h, w), a
    plain Σxφ²/Σφ² is wrong when the cell's support wraps the edge —
    it lands near the box centre, not on the cell. The fix is to map
    each grid index to an angle on the unit circle, take the φ²-
    weighted mean of (cos θ, sin θ), and map the angle back to a
    linear coordinate. Works for any periodic 1-D distribution and
    extends factor-wise to 2-D.

    Returns (cx, cy) in physical coordinates (not indices). Assumes
    the grid spans [0, w·dx) × [0, h·dy).
    """
    psq = phi * phi
    total = float(psq.sum())
    if total <= 0:
        return (float("nan"), float("nan"))
    h, w = psq.shape
    tx = 2.0 * np.pi * np.arange(w, dtype=np.float64) / w
    ty = 2.0 * np.pi * np.arange(h, dtype=np.float64) / h
    # Weighted means of (cos, sin).
    ux = float((np.cos(tx)[None, :] * psq).sum()) / total
    vx = float((np.sin(tx)[None, :] * psq).sum()) / total
    uy = float((np.cos(ty)[:, None] * psq).sum()) / total
    vy = float((np.sin(ty)[:, None] * psq).sum()) / total
    ang_x = (np.arctan2(vx, ux) + 2.0 * np.pi) % (2.0 * np.pi)
    ang_y = (np.arctan2(vy, uy) + 2.0 * np.pi) % (2.0 * np.pi)
    cx = ang_x * (w * dx) / (2.0 * np.pi)
    cy = ang_y * (h * dy) / (2.0 * np.pi)
    return float(cx), float(cy)


def composite_phi_sq(cells: List[CPUCell]) -> np.ndarray:
    """Σᵢ φᵢ²(x, y) on the full domain."""
    if not cells:
        raise ValueError("no cells to composite")
    g = np.zeros_like(cells[0].phi)
    for c in cells:
        g += c.phi * c.phi
    return g


def phi_at_bbox(cell: CPUCell, bbox: Tuple[int, int, int, int],
                halo: int) -> np.ndarray:
    """Extract a halo-padded tile from a CPU cell's full-domain φ,
    aligned with the production simulator's ``bbox = (x0, y0, x1, y1)``.

    Used for tile-level comparisons: the production simulator stores φ
    in a tile of dimensions ``(y1-y0+2·halo, x1-x0+2·halo)`` anchored at
    ``(x0-halo, y0-halo)``. This function slices the CPU's full field
    at the same global coordinates so the two arrays can be subtracted
    pixel-by-pixel.
    """
    x0, y0, x1, y1 = bbox
    Ny, Nx = cell.phi.shape
    ox, oy = x0 - halo, y0 - halo
    w_t = (x1 - x0) + 2 * halo
    h_t = (y1 - y0) + 2 * halo
    ys = (oy + np.arange(h_t)) % Ny
    xs = (ox + np.arange(w_t)) % Nx
    return cell.phi[np.ix_(ys, xs)]
