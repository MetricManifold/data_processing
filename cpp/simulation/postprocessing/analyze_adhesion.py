#!/usr/bin/env python3
"""
Adhesion Study — Batch Trajectory Analysis

Designed to run as a SLURM job on cluster data. Processes trajectory files
from Phase 1/2 adhesion runs and computes:
  - MSD(t) with periodic boundary unwrapping
  - Self-overlap Q(t) and stretched-exponential fit
  - Non-Gaussian parameter alpha_2(t)
  - Four-point susceptibility chi_4(t)
  - Effective diffusion coefficient D

Input:  base directory containing Jk_*/run_*/trajectory.txt
Output: JSON summary + per-run CSV files

Usage:
    # Single directory
    python analyze_adhesion.py /scratch/ssilber/adhesion_study/phase1_motility \
        --output /scratch/ssilber/adhesion_study/analysis/phase1_results.json

    # Specify analysis type
    python analyze_adhesion.py /path/to/data --analysis msd q_overlap diffusion

    # Phase 0 mode (static observables only)
    python analyze_adhesion.py /path/to/phase0 --analysis displacement energy_decomp
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

L_DEFAULT = 1562.0
R_CELL = 49.0
TAU = 10000.0
Q_THRESHOLD = 0.3 * R_CELL  # overlap threshold: 0.3R


# ---------------------------------------------------------------------------
# Trajectory I/O
# ---------------------------------------------------------------------------

def parse_header(filepath):
    """Extract v_A, N, Lx, Ly from trajectory header."""
    with open(filepath, "r") as f:
        for line in f:
            if "v_A=" in line:
                params = {}
                for token in line.strip().lstrip("# ").split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        try:
                            params[k] = float(v)
                        except ValueError:
                            params[k] = v
                return params
            if not line.startswith("#"):
                break
    return {}


def load_trajectory(filepath, max_rows=None):
    """Load trajectory file into structured arrays.

    Returns:
        times: (T,) unique time points
        positions: (T, N, 2) cell centroids
        header: dict with v_A, N, Lx, Ly
    """
    header = parse_header(filepath)
    N = int(header.get("N", 288))
    Lx = header.get("Lx", L_DEFAULT)
    Ly = header.get("Ly", L_DEFAULT)

    data = np.loadtxt(filepath, max_rows=max_rows)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    times_raw = data[:, 0]
    unique_times = np.unique(times_raw)
    n_times = len(unique_times)

    positions = np.zeros((n_times, N, 2))
    for t_idx, t in enumerate(unique_times):
        mask = times_raw == t
        frame = data[mask]
        for row in frame:
            cid = int(row[1])
            if 0 <= cid < N:
                positions[t_idx, cid, :] = row[2:4]

    return unique_times, positions, header


# ---------------------------------------------------------------------------
# Periodic boundary unwrapping
# ---------------------------------------------------------------------------

def unwrap_positions(positions, Lx, Ly):
    """Unwrap periodic boundary crossings."""
    unwrapped = positions.copy()
    T, N, _ = positions.shape
    for t in range(1, T):
        dx = unwrapped[t, :, 0] - unwrapped[t - 1, :, 0]
        dy = unwrapped[t, :, 1] - unwrapped[t - 1, :, 1]
        unwrapped[t, :, 0] -= np.round(dx / Lx) * Lx
        unwrapped[t, :, 1] -= np.round(dy / Ly) * Ly
    return unwrapped


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def compute_msd(unwrapped, times):
    """Compute ensemble-averaged MSD vs lag time."""
    T, N, _ = unwrapped.shape
    max_lag = T // 2
    lags = np.arange(1, max_lag + 1)
    dt = times[1] - times[0] if T > 1 else 1.0

    msd = np.zeros(max_lag)
    for lag in lags:
        displacements = unwrapped[lag:] - unwrapped[:-lag]
        sq_disp = np.sum(displacements ** 2, axis=-1)  # (T-lag, N)
        msd[lag - 1] = np.mean(sq_disp)

    lag_times = lags * dt
    return lag_times, msd


def compute_q_overlap(unwrapped, times, threshold=Q_THRESHOLD):
    """Compute self-overlap function Q(t).

    Q(t) = (1/N) sum_i Theta(a - |r_i(t0+t) - r_i(t0)|)
    averaged over t0.
    """
    T, N, _ = unwrapped.shape
    max_lag = T // 2
    dt = times[1] - times[0] if T > 1 else 1.0
    lags = np.arange(1, max_lag + 1)

    q_values = np.zeros(max_lag)
    for lag in lags:
        displacements = unwrapped[lag:] - unwrapped[:-lag]
        distances = np.sqrt(np.sum(displacements ** 2, axis=-1))  # (T-lag, N)
        overlap = (distances < threshold).astype(float)
        q_values[lag - 1] = np.mean(overlap)

    lag_times = lags * dt
    return lag_times, q_values


def compute_alpha2(unwrapped, times):
    """Compute non-Gaussian parameter alpha_2(t).

    alpha_2 = (d/(d+2)) * <r^4> / <r^2>^2 - 1   (d=2)
    """
    T, N, _ = unwrapped.shape
    max_lag = T // 2
    dt = times[1] - times[0] if T > 1 else 1.0
    lags = np.arange(1, max_lag + 1)

    alpha2 = np.zeros(max_lag)
    for lag in lags:
        displacements = unwrapped[lag:] - unwrapped[:-lag]
        r2 = np.sum(displacements ** 2, axis=-1)  # (T-lag, N)
        r4 = r2 ** 2
        mean_r2 = np.mean(r2)
        mean_r4 = np.mean(r4)
        if mean_r2 > 0:
            alpha2[lag - 1] = 0.5 * mean_r4 / (mean_r2 ** 2) - 1.0
        else:
            alpha2[lag - 1] = 0.0

    lag_times = lags * dt
    return lag_times, alpha2


def compute_chi4(unwrapped, times, threshold=Q_THRESHOLD):
    """Compute four-point susceptibility chi_4(t).

    chi_4(t) = N * [<Q(t)^2> - <Q(t)>^2]
    """
    T, N, _ = unwrapped.shape
    max_lag = T // 2
    dt = times[1] - times[0] if T > 1 else 1.0
    lags = np.arange(1, max_lag + 1)

    chi4 = np.zeros(max_lag)
    for lag in lags:
        displacements = unwrapped[lag:] - unwrapped[:-lag]
        distances = np.sqrt(np.sum(displacements ** 2, axis=-1))  # (T-lag, N)
        overlap = (distances < threshold).astype(float)
        Q_per_origin = np.mean(overlap, axis=1)  # (T-lag,)
        chi4[lag - 1] = N * np.var(Q_per_origin)

    lag_times = lags * dt
    return lag_times, chi4


def compute_diffusion(lag_times, msd, fit_fraction=0.5):
    """Extract diffusion coefficient from long-time MSD slope.

    D = MSD / (4 * t) in 2D, fitted over the last fit_fraction of data.
    """
    n = len(lag_times)
    start = int(n * (1 - fit_fraction))
    if start >= n - 1:
        start = max(0, n - 2)

    t_fit = lag_times[start:]
    msd_fit = msd[start:]

    if len(t_fit) < 2 or np.all(t_fit == 0):
        return 0.0, 0.0

    # Linear fit: MSD = 4D*t + c
    coeffs = np.polyfit(t_fit, msd_fit, 1)
    D = coeffs[0] / 4.0
    return max(D, 0.0), coeffs[0]  # D, slope


def compute_displacement(unwrapped, times):
    """Compute final displacement relative to cell radius."""
    r_final = unwrapped[-1] - unwrapped[0]
    distances = np.sqrt(np.sum(r_final ** 2, axis=-1))
    mean_dr = np.mean(distances) / R_CELL
    rms_dr = np.sqrt(np.mean(distances ** 2))
    return mean_dr, rms_dr


# ---------------------------------------------------------------------------
# Stretched exponential fit for Q(t)
# ---------------------------------------------------------------------------

def fit_stretched_exp(lag_times, q_values):
    """Fit Q(t) = exp(-(t/tau_alpha)^beta).

    Returns tau_alpha, beta or (nan, nan) on failure.
    """
    try:
        from scipy.optimize import curve_fit

        def model(t, tau, beta):
            return np.exp(-((t / tau) ** beta))

        # Only fit where Q > 0.01
        mask = q_values > 0.01
        if np.sum(mask) < 3:
            return np.nan, np.nan

        t_fit = lag_times[mask]
        q_fit = q_values[mask]

        popt, _ = curve_fit(
            model, t_fit, q_fit,
            p0=[lag_times[len(lag_times) // 2], 0.7],
            bounds=([0, 0.1], [lag_times[-1] * 10, 2.0]),
            maxfev=5000,
        )
        return popt[0], popt[1]
    except Exception:
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def discover_runs(base_dir):
    """Find all trajectory files organized by parameter combo.

    Expected structure: base_dir/COMBO_LABEL/run_NN/trajectory.txt

    Returns:
        dict: {combo_label: [trajectory_paths]}
    """
    runs = {}
    base = Path(base_dir)
    for combo_dir in sorted(base.iterdir()):
        if not combo_dir.is_dir():
            continue
        label = combo_dir.name
        trajs = []
        for run_dir in sorted(combo_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            traj = run_dir / "trajectory.txt"
            if traj.exists():
                trajs.append(str(traj))
        if trajs:
            runs[label] = trajs
    return runs


def parse_label(label):
    """Parse combo label like 'Jk_0.05_vA0.002' into physics parameters."""
    params = {}
    jk_match = re.search(r"Jk_([\d.]+)", label)
    va_match = re.search(r"vA([\d.]+)", label)
    if jk_match:
        jk = float(jk_match.group(1))
        params["J_over_kappa"] = jk
        params["J_tilde"] = jk * 5.0  # J̃ = 5 * J/κ for κ=10, γ=1
    if va_match:
        params["v_A"] = float(va_match.group(1))
    return params


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

ANALYSIS_MAP = {
    "msd": compute_msd,
    "q_overlap": compute_q_overlap,
    "alpha2": compute_alpha2,
    "chi4": compute_chi4,
    "displacement": compute_displacement,
}


def analyze_single_run(trajectory_path, analyses):
    """Run all requested analyses on a single trajectory.

    Returns dict of results.
    """
    times, positions, header = load_trajectory(trajectory_path)
    Lx = header.get("Lx", L_DEFAULT)
    Ly = header.get("Ly", L_DEFAULT)
    unwrapped = unwrap_positions(positions, Lx, Ly)

    results = {"trajectory": trajectory_path, "header": header}

    if "displacement" in analyses:
        mean_dr, rms = compute_displacement(unwrapped, times)
        results["displacement"] = {"mean_dr_over_R": mean_dr, "rms": rms}

    if "msd" in analyses:
        lag_times, msd = compute_msd(unwrapped, times)
        results["msd"] = {"lag_times": lag_times.tolist(), "values": msd.tolist()}

    if "diffusion" in analyses or "msd" in analyses:
        if "msd" not in results:
            lag_times, msd = compute_msd(unwrapped, times)
        else:
            lag_times = np.array(results["msd"]["lag_times"])
            msd = np.array(results["msd"]["values"])
        D, slope = compute_diffusion(lag_times, msd)
        results["diffusion"] = {"D": D, "slope": slope}

    if "q_overlap" in analyses:
        lag_times_q, q_vals = compute_q_overlap(unwrapped, times)
        tau_alpha, beta = fit_stretched_exp(lag_times_q, q_vals)
        results["q_overlap"] = {
            "lag_times": lag_times_q.tolist(),
            "values": q_vals.tolist(),
            "tau_alpha": float(tau_alpha) if not np.isnan(tau_alpha) else None,
            "beta": float(beta) if not np.isnan(beta) else None,
        }

    if "alpha2" in analyses:
        lag_times_a, a2 = compute_alpha2(unwrapped, times)
        peak_idx = np.argmax(a2)
        results["alpha2"] = {
            "lag_times": lag_times_a.tolist(),
            "values": a2.tolist(),
            "peak_time": float(lag_times_a[peak_idx]),
            "peak_value": float(a2[peak_idx]),
        }

    if "chi4" in analyses:
        lag_times_c, c4 = compute_chi4(unwrapped, times)
        peak_idx = np.argmax(c4)
        results["chi4"] = {
            "lag_times": lag_times_c.tolist(),
            "values": c4.tolist(),
            "peak_time": float(lag_times_c[peak_idx]),
            "peak_value": float(c4[peak_idx]),
        }

    return results


def analyze_batch(base_dir, analyses, output_path):
    """Analyze all discovered runs and produce summary."""
    runs = discover_runs(base_dir)

    if not runs:
        print(f"No trajectory files found in {base_dir}")
        sys.exit(1)

    print(f"Discovered {sum(len(v) for v in runs.values())} trajectory files "
          f"across {len(runs)} parameter combos")

    summary = {
        "metadata": {
            "base_dir": str(base_dir),
            "analyses": analyses,
            "timestamp": datetime.now().isoformat(),
            "n_combos": len(runs),
            "n_total_runs": sum(len(v) for v in runs.values()),
        },
        "combos": {},
    }

    for label, traj_paths in runs.items():
        params = parse_label(label)
        print(f"\n--- {label} ({len(traj_paths)} replicates) ---")

        combo_results = {
            "params": params,
            "n_replicates": len(traj_paths),
            "replicates": [],
        }

        D_values = []
        displacement_values = []
        tau_alpha_values = []
        alpha2_peak_values = []
        chi4_peak_values = []

        for traj_path in traj_paths:
            run_name = Path(traj_path).parent.name
            print(f"  Processing {run_name}...", end=" ", flush=True)

            try:
                result = analyze_single_run(traj_path, analyses)
                combo_results["replicates"].append({
                    "run": run_name,
                    "status": "ok",
                    "diffusion": result.get("diffusion"),
                    "displacement": result.get("displacement"),
                    "q_overlap_fit": {
                        "tau_alpha": result.get("q_overlap", {}).get("tau_alpha"),
                        "beta": result.get("q_overlap", {}).get("beta"),
                    } if "q_overlap" in result else None,
                    "alpha2_peak": result.get("alpha2", {}).get("peak_value"),
                    "chi4_peak": result.get("chi4", {}).get("peak_value"),
                })

                if "diffusion" in result:
                    D_values.append(result["diffusion"]["D"])
                if "displacement" in result:
                    displacement_values.append(result["displacement"]["mean_dr_over_R"])
                if "q_overlap" in result and result["q_overlap"]["tau_alpha"] is not None:
                    tau_alpha_values.append(result["q_overlap"]["tau_alpha"])
                if "alpha2" in result:
                    alpha2_peak_values.append(result["alpha2"]["peak_value"])
                if "chi4" in result:
                    chi4_peak_values.append(result["chi4"]["peak_value"])

                print("ok")
            except Exception as e:
                combo_results["replicates"].append({
                    "run": run_name,
                    "status": "error",
                    "error": str(e),
                })
                print(f"ERROR: {e}")

        # Aggregate statistics
        stats = {}
        if D_values:
            stats["D_mean"] = float(np.mean(D_values))
            stats["D_stderr"] = float(np.std(D_values) / np.sqrt(len(D_values)))
        if displacement_values:
            stats["displacement_mean"] = float(np.mean(displacement_values))
            stats["displacement_stderr"] = float(
                np.std(displacement_values) / np.sqrt(len(displacement_values))
            )
        if tau_alpha_values:
            stats["tau_alpha_mean"] = float(np.mean(tau_alpha_values))
            stats["tau_alpha_stderr"] = float(
                np.std(tau_alpha_values) / np.sqrt(len(tau_alpha_values))
            )
        if alpha2_peak_values:
            stats["alpha2_peak_mean"] = float(np.mean(alpha2_peak_values))
        if chi4_peak_values:
            stats["chi4_peak_mean"] = float(np.mean(chi4_peak_values))

        combo_results["stats"] = stats
        summary["combos"][label] = combo_results

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results written to {output_path}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'Label':<30} {'D':>12} {'<Δr>/R':>10} {'τ_α':>12}")
    print("-" * 70)
    for label, combo in sorted(summary["combos"].items()):
        s = combo.get("stats", {})
        D_str = f"{s['D_mean']:.2e} ± {s['D_stderr']:.1e}" if "D_mean" in s else "—"
        dr_str = f"{s['displacement_mean']:.4f}" if "displacement_mean" in s else "—"
        ta_str = f"{s['tau_alpha_mean']:.0f}" if "tau_alpha_mean" in s else "—"
        print(f"{label:<30} {D_str:>12} {dr_str:>10} {ta_str:>12}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Adhesion study trajectory analysis (batch mode for SLURM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("data_dir", help="Base directory with Jk_*/run_*/trajectory.txt")
    parser.add_argument(
        "--output", "-o",
        default="analysis_results.json",
        help="Output JSON path (default: analysis_results.json)",
    )
    parser.add_argument(
        "--analysis", "-a",
        nargs="+",
        default=["msd", "diffusion", "q_overlap", "alpha2", "chi4", "displacement"],
        choices=["msd", "diffusion", "q_overlap", "alpha2", "chi4", "displacement"],
        help="Analysis types to run",
    )
    args = parser.parse_args()

    print(f"Adhesion study analysis")
    print(f"  Data:     {args.data_dir}")
    print(f"  Output:   {args.output}")
    print(f"  Analyses: {', '.join(args.analysis)}")
    print()

    analyze_batch(args.data_dir, args.analysis, args.output)


if __name__ == "__main__":
    main()
