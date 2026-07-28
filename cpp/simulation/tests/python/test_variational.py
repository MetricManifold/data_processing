"""Non-circular checks on the PDE coefficients.

Every other physics test compares the simulator against cpu_reference.py or
the Rust reference, which encode the same derived constants — circular for
coefficient errors. And the two oracle tests
(test_single_cell_pde_residual_small, test_isolated_cell_energy_decreases)
are single-cell, so the repulsion coefficient — the only one needing >=2
cells — was never exercised. That gap let a factor-of-2 error in the
repulsion term survive.

These tests assume no coefficient from the implementation under test.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cpu_reference import (  # noqa: E402
    CPUCell, CPUParams, laplacian_9pt, step,
)


def free_energy(phis, p):
    """Palmieri Eq. (7) + Eq. (10), discretised. No derived constants.

    The gradient term is written as -gamma * sum phi.(L9 phi), the discrete
    integration by parts of gamma*int|grad phi|^2. L9 is symmetric, so its
    derivative is exactly -2 gamma L9 phi — no stencil mismatch.
    """
    dA, A0 = p.dA, p.target_area
    F = 0.0
    for phi in phis:
        F += -p.gamma * float((phi * laplacian_9pt(phi, p.dx)).sum()) * dA
        F += (30.0 * p.gamma / p.lambd**2) * float(
            (phi**2 * (1.0 - phi) ** 2).sum()) * dA
        V = float((phi * phi).sum()) * dA
        F += (p.mu / A0) * (V - A0) ** 2
    ck = 30.0 * p.kappa / p.lambd**2
    for n in range(len(phis)):
        for m in range(len(phis)):
            if m != n:                      # ordered sum: pairs count twice
                F += ck * float((phis[n] ** 2 * phis[m] ** 2).sum()) * dA
    return F


def _numeric_dF(phis, p, cell, pixels, eps=1e-6):
    out = []
    for (iy, ix) in pixels:
        saved = phis[cell][iy, ix]
        phis[cell][iy, ix] = saved + eps
        Fp = free_energy(phis, p)
        phis[cell][iy, ix] = saved - eps
        Fm = free_energy(phis, p)
        phis[cell][iy, ix] = saved
        out.append((Fp - Fm) / (2.0 * eps) / p.dA)
    return np.array(out)


def _two_overlapping(p, sep):
    yy, xx = np.mgrid[0:p.Ny, 0:p.Nx].astype(float)
    cy = p.Ny / 2.0
    w = 0.5164 * p.lambd
    return [0.5 * (1.0 - np.tanh(
        (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - p.target_radius) / w))
        for cx in (p.Nx / 2.0 - sep / 2.0, p.Nx / 2.0 + sep / 2.0)]


def test_rhs_is_half_the_variational_derivative():
    """dphi/dt must equal -1/2 dF/dphi where F is written from the paper.

    xi is huge so the interaction velocity — and hence advection — vanishes
    and the step is pure gradient flow.
    """
    p = CPUParams(Nx=96, Ny=96, dt=1e-7, lambd=4.0, gamma=1.0, kappa=10.0,
                  mu=1.0, xi=1e14, target_radius=12.0)
    phis = _two_overlapping(p, sep=20.0)
    cells = [CPUCell(phi=f.copy(), vol=float((f * f).sum()) * p.dA)
             for f in phis]

    rhs = (step(cells, p)[0].phi - cells[0].phi) / p.dt

    live = np.argwhere((phis[0] > 0.05) & (phis[0] < 0.95)
                       & (phis[1] ** 2 > 1e-3))
    assert len(live) > 20, "no overlapping interface pixels; adjust sep"
    pix = [tuple(live[i]) for i in
           np.random.default_rng(0).choice(len(live), 24, replace=False)]

    expected = -0.5 * _numeric_dF([f.copy() for f in phis], p, 0, pix)
    got = np.array([rhs[iy, ix] for (iy, ix) in pix])
    rel = np.max(np.abs(got - expected)) / np.max(np.abs(expected))
    assert rel < 1e-3, (
        f"RHS is not -1/2 dF/dphi in the overlap region: max relative error "
        f"{rel:.2%}, median got/expected = {np.median(got / expected):.4f}")


@pytest.mark.parametrize("kappa,lambd,xi",
                         [(10.0, 7.0, 1500.0), (3.0, 4.0, 250.0)])
def test_repulsion_over_motility_equals_xi(kappa, lambd, xi):
    """Convention-free: both coefficients are M dF_int/dphi, one over xi.

    Holds for any normalisation of F_int, so this needs no reference to the
    paper. Mirrors the invariant asserted in include/types.cuh.
    """
    p = CPUParams(Nx=8, Ny=8, lambd=lambd, kappa=kappa, xi=xi)
    rhs_rep = 0.5 * (120.0 * p.kappa / p.lambd**2)   # rhs = -0.5*var_deriv
    ratio = rhs_rep / p.motility_coeff
    assert abs(ratio - xi) / xi < 1e-12, (
        f"repulsion/motility ratio is {ratio:.1f} but must be xi={xi}; "
        f"off by a factor of {ratio / xi:.3f}")
