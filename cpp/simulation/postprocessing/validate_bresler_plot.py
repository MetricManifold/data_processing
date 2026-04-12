#!/usr/bin/env python3
"""
Plot validation results against Bresler, Palmieri & Grant (2018).

Loads the .npz output from validate_bresler_cluster.py and produces
multi-panel comparison figures.

Usage:
  python validate_bresler_plot.py validation_results.npz [--output-dir figures/]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# Color map for motility values (low=red/jammed, high=green/fluid)
def vA_color(vA, vA_min, vA_max):
    """Map vA to color: red (jammed) → green (fluid)."""
    frac = (vA - vA_min) / (vA_max - vA_min) if vA_max > vA_min else 0.5
    return plt.cm.RdYlGn(0.15 + 0.7 * frac)


# ---------------------------------------------------------------------------
# Panel 1: MSD vs time (cf. Bresler Fig 1b)
# ---------------------------------------------------------------------------

def plot_msd(ax, data, vA_values):
    vA_min, vA_max = min(vA_values), max(vA_values)
    tau = float(data.get("param_tau", 10000))

    for i, vA in enumerate(vA_values):
        tag = f"vA{vA:.3f}".replace(".", "p")
        lag_key = f"msd_lag_{tag}"
        val_key = f"msd_val_{tag}"
        if lag_key not in data or val_key not in data:
            continue
        lags = data[lag_key]
        msd = data[val_key]
        if len(lags) < 2:
            continue

        color = vA_color(vA, vA_min, vA_max)
        ax.loglog(lags, msd, "-", color=color, linewidth=1.2,
                  label=f"$v_A = {vA:.3f}$", alpha=0.85)

    # Reference lines
    if len(vA_values) > 0:
        # Get a reference MSD for positioning
        tag = f"vA{vA_values[-1]:.3f}".replace(".", "p")
        ref_lags = data.get(f"msd_lag_{tag}", np.array([]))
        if len(ref_lags) > 2:
            t_ref = np.logspace(np.log10(ref_lags[1]), np.log10(ref_lags[-1]), 50)
            # Ballistic reference
            vA_max_val = max(vA_values)
            msd_ballistic = (vA_max_val * t_ref) ** 2
            ax.loglog(t_ref, msd_ballistic, "k--", alpha=0.3, linewidth=1,
                      label="$\\sim t^2$ (ballistic)")
            # Diffusive reference
            D_iso = vA_max_val ** 2 * tau / 2
            msd_diffusive = 4 * D_iso * t_ref
            ax.loglog(t_ref, msd_diffusive, "k:", alpha=0.3, linewidth=1,
                      label="$\\sim t$ (diffusive)")

    # Mark τ
    ax.axvline(tau, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.text(tau * 1.1, ax.get_ylim()[0] * 2, "$\\tau$", color="gray", fontsize=8)

    ax.set_xlabel("Time lag $\\Delta t$")
    ax.set_ylabel("MSD $\\langle \\Delta r^2 \\rangle$")
    ax.set_title("(a) MSD vs time  [cf. Bresler Fig 1b]")
    ax.legend(loc="lower right", fontsize=7, ncol=2)


# ---------------------------------------------------------------------------
# Panel 2: D_eff vs vA (cf. Bresler Fig 1d)
# ---------------------------------------------------------------------------

def plot_deff_vs_vA(ax, data, vA_values):
    deff = data["deff_values"]
    derr = data["deff_errors"]
    tau = float(data.get("param_tau", 10000))

    ax.errorbar(vA_values, deff, yerr=derr, fmt="ko-", markersize=5,
                capsize=3, linewidth=1.2, label="Simulation")

    # Analytical D_iso = vA²τ/2 (isolated cell)
    vA_range = np.linspace(0, max(vA_values) * 1.1, 50)
    D_iso = vA_range ** 2 * tau / 2
    ax.plot(vA_range, D_iso, "b--", alpha=0.4, linewidth=1,
            label="$D_{iso} = v_A^2 \\tau/2$")

    # Linear fit to estimate v*
    # Use points where D_eff > 20% of max (above noise floor)
    valid = deff > 0.2 * np.max(deff)
    if np.sum(valid) >= 3:
        coeffs = np.polyfit(vA_values[valid], deff[valid], 1)
        v_star = -coeffs[1] / coeffs[0] if coeffs[0] > 0 else 0
        fit_vA = np.linspace(max(0, v_star - 0.001), max(vA_values) * 1.05, 50)
        fit_D = np.polyval(coeffs, fit_vA)
        ax.plot(fit_vA, np.maximum(fit_D, 0), "g-", alpha=0.6, linewidth=1.5,
                label=f"Linear fit → $v^* \\approx {v_star:.4f}$")
        ax.axvline(v_star, color="red", linestyle=":", alpha=0.5, linewidth=1)
        ax.annotate(f"$v^*$", xy=(v_star, 0), fontsize=9, color="red",
                    ha="center", va="bottom")

    # Noise floor reference
    noise = 2e-3  # Bresler's noise floor
    ax.axhline(noise, color="gray", linestyle=":", alpha=0.3)
    ax.text(vA_values[0], noise * 1.3, "Bresler noise floor", fontsize=7,
            color="gray")

    ax.set_xlabel("Active velocity $v_A$")
    ax.set_ylabel("$D_{eff}$")
    ax.set_title("(b) Effective diffusion  [cf. Bresler Fig 1d]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Panel 3: Voronoi shape index (cf. Bresler Fig 2e)
# ---------------------------------------------------------------------------

def plot_voronoi_q(ax, data, vA_values):
    q_mean = data["voronoi_q_mean"]
    q_std = data["voronoi_q_std"]
    deff = data["deff_values"]

    # Estimate v*
    valid = deff > 0.2 * np.max(deff)
    v_star = 0
    if np.sum(valid) >= 3:
        coeffs = np.polyfit(vA_values[valid], deff[valid], 1)
        v_star = -coeffs[1] / coeffs[0] if coeffs[0] > 0 else 0

    # Plot vs vA - v*
    delta_v = vA_values - v_star

    ax.errorbar(delta_v * 1000, q_mean, yerr=q_std / np.sqrt(10),
                fmt="ko-", markersize=5, capsize=3, linewidth=1.2)

    # Reference lines
    ax.axhline(3.81, color="blue", linestyle="--", alpha=0.4, linewidth=1,
               label="$q^*_{SPV} = 3.81$ (Bi 2016)")
    ax.axhline(3.765, color="red", linestyle="--", alpha=0.4, linewidth=1,
               label="$q^*_V = 3.765$ (Bresler 2018)")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("$(v_A - v^*) \\times 10^3$")
    ax.set_ylabel("Voronoi shape index $\\langle q_V \\rangle$")
    ax.set_title("(c) Shape index  [cf. Bresler Fig 2e]")
    ax.legend(loc="lower right", fontsize=8)


# ---------------------------------------------------------------------------
# Panel 4: g₆(r) bond orientational correlation (cf. Bresler Fig 1c)
# ---------------------------------------------------------------------------

def plot_g6(ax, data, vA_values):
    vA_min, vA_max = min(vA_values), max(vA_values)
    R = float(data.get("param_R", 49))

    for i, vA in enumerate(vA_values):
        tag = f"vA{vA:.3f}".replace(".", "p")
        r_key = f"g6_r_{tag}"
        val_key = f"g6_val_{tag}"
        if r_key not in data or val_key not in data:
            continue
        r = data[r_key]
        g6 = data[val_key]
        if len(r) < 2:
            continue

        color = vA_color(vA, vA_min, vA_max)
        # Normalize r by R
        ax.loglog(r / R, np.abs(g6) + 1e-6, "o-", color=color,
                  markersize=3, linewidth=1, alpha=0.8,
                  label=f"$v_A = {vA:.3f}$")

    # KT prediction: η = 1/4 power law
    r_ref = np.logspace(0, np.log10(15), 50)
    ax.loglog(r_ref, 0.5 * r_ref ** (-0.25), "k--", alpha=0.4, linewidth=1.5,
              label="$r^{-1/4}$ (KT)")

    ax.set_xlabel("$r / R$")
    ax.set_ylabel("$|g_6(r)|$")
    ax.set_title("(d) Bond-orientational order  [cf. Bresler Fig 1c]")
    ax.legend(loc="lower left", fontsize=7, ncol=2)


# ---------------------------------------------------------------------------
# Panel 5: D_eff ratio to isolated cell
# ---------------------------------------------------------------------------

def plot_deff_ratio(ax, data, vA_values):
    deff = data["deff_values"]
    tau = float(data.get("param_tau", 10000))
    rho = float(data.get("param_rho", 0.85))

    D_iso = vA_values ** 2 * tau / 2
    ratio = np.where(D_iso > 0, deff / D_iso, 0)

    ax.plot(vA_values, ratio, "ko-", markersize=5, linewidth=1.2)

    # Bresler prediction: D/D_iso ≈ 1 - ρ/ρ_J (Eq. 2)
    rho_J = 0.843  # Bresler estimate
    pred_ratio = max(0, 1 - rho / rho_J)
    ax.axhline(pred_ratio, color="red", linestyle="--", alpha=0.5,
               label=f"Bresler Eq.2: $1-\\rho/\\rho_J = {pred_ratio:.3f}$"
                     f"\n($\\rho={rho:.3f}$, $\\rho_J=0.843$)")

    ax.set_xlabel("Active velocity $v_A$")
    ax.set_ylabel("$D_{eff} / D_{iso}$")
    ax.set_title("(e) Diffusion suppression by crowding")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)


# ---------------------------------------------------------------------------
# Panel 6: MSD scaling check (MSD/t² vs t)
# ---------------------------------------------------------------------------

def plot_msd_rescaled(ax, data, vA_values):
    """Plot MSD/t vs t to identify diffusive regime (plateau = 4D)."""
    vA_min, vA_max = min(vA_values), max(vA_values)
    tau = float(data.get("param_tau", 10000))

    for i, vA in enumerate(vA_values):
        tag = f"vA{vA:.3f}".replace(".", "p")
        lags = data.get(f"msd_lag_{tag}", np.array([]))
        msd = data.get(f"msd_val_{tag}", np.array([]))
        if len(lags) < 2:
            continue

        color = vA_color(vA, vA_min, vA_max)
        # MSD/t → should plateau at 4*D_eff in diffusive regime
        with np.errstate(divide="ignore", invalid="ignore"):
            msd_over_t = np.where(lags > 0, msd / lags, 0)
        ax.semilogx(lags, msd_over_t, "-", color=color, linewidth=1,
                     alpha=0.8, label=f"$v_A = {vA:.3f}$")

    ax.axvline(tau, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Time lag $\\Delta t$")
    ax.set_ylabel("MSD / $\\Delta t$")
    ax.set_title("(f) MSD/$t$ (plateau = $4D_{eff}$)")
    ax.legend(loc="best", fontsize=7, ncol=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Bresler validation")
    parser.add_argument("npz_file", help="Input .npz file from cluster analysis")
    parser.add_argument("--output-dir", "-o", default=".",
                        help="Directory for output figures")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = dict(np.load(args.npz_file, allow_pickle=True))
    vA_values = data["vA_values"]
    print(f"Loaded results for {len(vA_values)} motility values: {vA_values}")
    print(f"D_eff values: {data['deff_values']}")
    print(f"Voronoi <q>: {data['voronoi_q_mean']}")

    rho = float(data.get("param_rho", 0.85))
    N = int(data.get("param_N", 288))
    print(f"Parameters: ρ={rho:.4f}, N={N}")

    # ---- Figure 1: Main validation (2x3 panels) ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    plot_msd(axes[0, 0], data, vA_values)
    plot_deff_vs_vA(axes[0, 1], data, vA_values)
    plot_voronoi_q(axes[0, 2], data, vA_values)
    plot_g6(axes[1, 0], data, vA_values)
    plot_deff_ratio(axes[1, 1], data, vA_values)
    plot_msd_rescaled(axes[1, 2], data, vA_values)

    fig.suptitle(
        f"Phase Field Model Validation against Bresler et al. (2018)\n"
        f"$N = {N}$, $\\rho = {rho:.3f}$, $\\gamma = 1.0$ "
        f"(Bresler: sharp-interface, $\\gamma = 1.35$–$3.75$)",
        fontsize=13, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    outfile = out_dir / "bresler_validation.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {outfile}")

    if not args.no_show:
        plt.show()
    plt.close()

    # ---- Print quantitative comparison ----
    print("\n" + "=" * 70)
    print("QUANTITATIVE COMPARISON WITH BRESLER et al. (2018)")
    print("=" * 70)

    tau = 10000
    deff = data["deff_values"]

    # Jamming velocity estimate
    valid = deff > 0.2 * np.max(deff)
    if np.sum(valid) >= 3:
        coeffs = np.polyfit(vA_values[valid], deff[valid], 1)
        v_star = -coeffs[1] / coeffs[0] if coeffs[0] > 0 else 0
        slope = coeffs[0]
    else:
        v_star = 0
        slope = 0

    print(f"\n1. Jamming velocity estimate:")
    print(f"   v* = {v_star:.5f}  (this work, ρ={rho:.3f}, γ=1.0)")
    print(f"   Bresler prediction (Eq.3, soft cells, ρ=0.85): v* ≈ 0.003-0.005")
    print(f"   Bresler ρ_J ≈ 0.843 → D(ρ) = D₀(1 - ρ/ρ_J)")
    if rho < 0.843:
        print(f"   At ρ={rho:.3f}: D/D₀ ≈ {1 - rho/0.843:.3f} (predicted)")
    else:
        print(f"   At ρ={rho:.3f}: JAMMED (ρ > ρ_J)")

    print(f"\n2. D_eff linear slope:")
    print(f"   dD/dvA = {slope:.2f}  (this work)")
    print(f"   Bresler: dD/dvA ≈ 8.264 (universal above transition)")
    print(f"   Bresler Turnbull: ½ × 2R = {2*49/2:.1f} = {49:.1f}")

    print(f"\n3. Voronoi shape index at transition:")
    # Find q closest to v*
    idx_star = np.argmin(np.abs(vA_values - v_star))
    q_at_star = data["voronoi_q_mean"][idx_star]
    print(f"   <q_V> at v* = {q_at_star:.4f}  (this work)")
    print(f"   Bresler: q*_V ≈ 3.765")
    print(f"   Bi (SPV): q*_SPV = 3.81")

    print(f"\n4. Diffusion suppression (highest vA = {vA_values[-1]:.3f}):")
    D_sim = deff[-1]
    D_iso = vA_values[-1] ** 2 * tau / 2
    print(f"   D_eff(sim)  = {D_sim:.6f}")
    print(f"   D_iso       = {D_iso:.6f}")
    print(f"   D/D_iso     = {D_sim/D_iso:.4f}  ({(1-D_sim/D_iso)*100:.1f}% suppression)")
    print(f"   Bresler: ~10× suppression from isolated value (at ρ=0.90)")

    print(f"\n5. MSD regimes:")
    print(f"   Expected: ballistic (t²) for t << τ={tau}")
    print(f"   Expected: diffusive (t) for t >> τ")
    print(f"   Check the MSD plot visually for crossover at t ≈ τ")


if __name__ == "__main__":
    main()
