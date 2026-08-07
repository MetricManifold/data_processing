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
import re
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


def _measure_repulsion_coeff(p):
    """Recover C in ``dphi/dt = ... - C * phi_0 * S_other`` from step() itself.

    Two steps that differ ONLY in the neighbour amplitude are differenced, so
    every term of cell 0 that does not involve S -- Laplacian, double well,
    volume constraint -- cancels identically. Nothing is assumed about the
    coefficient, and no free energy is written down, so this cannot inherit an
    ordered-vs-unordered mistake the way test_rhs_is_half_the_variational_
    derivative can. p.xi must be huge so advection vanishes.
    """
    yy, xx = np.mgrid[0:p.Ny, 0:p.Nx].astype(float)
    cy, w = p.Ny / 2.0, 0.5164 * p.lambd

    def disc(cx, amp):
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        return amp * 0.5 * (1.0 - np.tanh((r - p.target_radius) / w))

    phi0 = disc(p.Nx / 2.0 - 10.0, 1.0)

    def rhs(neigh):
        cells = [CPUCell(phi=f.copy(), vol=float((f * f).sum()) * p.dA)
                 for f in (phi0, neigh)]
        return (step(cells, p)[0].phi - phi0) / p.dt

    n_a, n_b = disc(p.Nx / 2.0 + 10.0, 1.0), disc(p.Nx / 2.0 + 10.0, 0.5)
    dS = n_a ** 2 - n_b ** 2
    num, den = rhs(n_a) - rhs(n_b), phi0 * dS      # num = -C * phi0 * dS
    live = np.abs(den) > 1e-4 * np.abs(den).max()
    assert live.sum() > 50, "no overlap pixels; adjust geometry"
    return float(np.median(-num[live] / den[live]))


@pytest.mark.parametrize("kappa,lambd,xi",
                         [(10.0, 7.0, 1500.0), (3.0, 4.0, 250.0)])
def test_repulsion_over_motility_equals_xi(kappa, lambd, xi):
    """Convention-free: both coefficients are M dF_int/dphi, one over xi.

    Holds for any normalisation of F_int, so this needs no reference to the
    paper. Mirrors the invariant asserted in include/types.cuh.

    The repulsion coefficient is *measured* from step() rather than written
    down here; an earlier version of this test hardcoded it, which made the
    assertion an identity between two literals that could never fail.
    """
    probe = CPUParams(Nx=96, Ny=96, dt=1e-7, lambd=lambd, gamma=1.0,
                      kappa=kappa, mu=1.0, xi=1e14, target_radius=12.0)
    C = _measure_repulsion_coeff(probe)             # independent of xi
    p = CPUParams(Nx=8, Ny=8, lambd=lambd, kappa=kappa, xi=xi)
    ratio = C / p.motility_coeff
    assert abs(ratio - xi) / xi < 1e-3, (
        f"measured repulsion coeff {C:.6f} over motility coeff is {ratio:.1f} "
        f"but must be xi={xi}; off by a factor of {ratio / xi:.3f}")


def test_cuda_coefficients_satisfy_the_same_invariant():
    """Guard the CUDA constants directly -- no GPU, no compiler needed.

    test_relaxation_fields_match covers types.cuh only transitively, by
    comparing the built binary against cpu_reference.py, and it needs a GPU.
    This reads the literals so a lone edit to types.cuh fails on any machine.
    """
    src = (pathlib.Path(__file__).parents[2]
           / "include" / "types.cuh").read_text()

    def literal(fn, denom):
        m = re.search(
            r"T\s+" + fn + r"\s*\([^)]*\)\s*\{\s*return\s+T\((\d+(?:\.\d+)?)\)"
            r"\s*\*\s*kappa\s*/\s*\(\s*" + denom + r"\s*\)", src)
        assert m, f"could not read {fn} out of types.cuh"
        return float(m.group(1))

    rep = literal("interaction_coeff", r"lambda \* lambda")
    mot = literal("motility_coeff", r"xi \* lambda \* lambda")
    assert rep == mot, (
        f"interaction_coeff has {rep} and motility_coeff has {mot}; both are "
        f"M dF_int/dphi so the numerators must match and differ only by xi. "
        f"A mismatch means one of them is off by {mot / rep:.2f}x.")
