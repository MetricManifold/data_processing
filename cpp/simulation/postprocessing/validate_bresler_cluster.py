#!/usr/bin/env python3
"""
Cluster-side analysis script for validating phase field simulation
against Bresler, Palmieri & Grant (2018) arXiv:1807.10318.

Computes from production trajectory data:
  1. MSD(t) for each motility vA  (cf. Bresler Fig 1b)
  2. D_eff vs vA from long-time MSD  (cf. Bresler Fig 1d)
  3. Voronoi shape index <q_V>  (cf. Bresler Fig 2e)
  4. Sixfold bond-orientational order g6(r)  (cf. Bresler Fig 1c)

Usage:
  module load python/3.11 scipy-stack
  python3 validate_bresler_cluster.py /scratch/ssilber/jamming_study/production_288 \
      --output /scratch/ssilber/jamming_study/validation_results.npz \
      --max-runs 20

Outputs a single .npz file with all computed observables.
"""

import argparse
import glob
import os
import sys
import time as clock

import numpy as np


# ---------------------------------------------------------------------------
# Trajectory loader
# ---------------------------------------------------------------------------

def load_trajectory(filepath):
    """Load trajectory file. Returns (data, params) where params is a dict."""
    params = {}
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                # Parse parameter header
                if "v_A=" in line:
                    for tok in line.split():
                        if "=" in tok and not tok.startswith("#"):
                            k, v = tok.split("=", 1)
                            try:
                                params[k] = float(v)
                            except ValueError:
                                params[k] = v
                continue
            parts = line.split()
            if len(parts) >= 9:
                rows.append([float(x) for x in parts[:9]])
    data = np.array(rows, dtype=np.float64)
    return data, params


# ---------------------------------------------------------------------------
# MSD computation (unwrapped positions, multi-origin)
# ---------------------------------------------------------------------------

def compute_msd_single_run(filepath, max_lag_frac=0.5):
    """Compute MSD for a single run from its trajectory file.

    Returns (lag_times, msd_values, n_cells) or None on failure.
    """
    data, params = load_trajectory(filepath)
    if len(data) == 0:
        return None

    N = int(params.get("N", len(np.unique(data[:, 1]))))
    Lx = float(params.get("Lx", 1600))
    Ly = float(params.get("Ly", 1600))

    times = np.unique(data[:, 0])
    n_times = len(times)
    if n_times < 10:
        return None

    # Build position arrays: shape (n_times, N, 2)
    # Unwrap periodic boundaries
    pos = np.full((n_times, N, 2), np.nan)
    time_to_idx = {t: i for i, t in enumerate(times)}

    for row in data:
        t_idx = time_to_idx[row[0]]
        cid = int(row[1])
        if cid < N:
            pos[t_idx, cid, 0] = row[2]
            pos[t_idx, cid, 1] = row[3]

    # Unwrap periodic BCs per cell
    for c in range(N):
        for t in range(1, n_times):
            if np.isnan(pos[t, c, 0]) or np.isnan(pos[t - 1, c, 0]):
                continue
            dx = pos[t, c, 0] - pos[t - 1, c, 0]
            dy = pos[t, c, 1] - pos[t - 1, c, 1]
            if dx > Lx / 2:
                pos[t:, c, 0] -= Lx
            elif dx < -Lx / 2:
                pos[t:, c, 0] += Lx
            if dy > Ly / 2:
                pos[t:, c, 1] -= Ly
            elif dy < -Ly / 2:
                pos[t:, c, 1] += Ly

    # MSD via multi-origin averaging
    max_lag = max(2, int(n_times * max_lag_frac))
    lag_times = np.zeros(max_lag)
    msd_vals = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    for lag in range(1, max_lag):
        dt = times[lag] - times[0]  # assume uniform spacing
        displacements_sq = []
        for t0 in range(n_times - lag):
            dr = pos[t0 + lag] - pos[t0]  # shape (N, 2)
            valid = ~np.isnan(dr[:, 0])
            if valid.any():
                dsq = np.sum(dr[valid] ** 2, axis=1)
                displacements_sq.extend(dsq.tolist())

        if displacements_sq:
            lag_times[lag] = dt
            msd_vals[lag] = np.mean(displacements_sq)
            counts[lag] = len(displacements_sq)

    valid = counts > 0
    return lag_times[valid], msd_vals[valid], counts[valid], N


# ---------------------------------------------------------------------------
# D_eff from MSD (long-time linear fit)
# ---------------------------------------------------------------------------

def compute_deff_from_msd(lag_times, msd_vals, fit_frac=0.3):
    """Compute effective diffusion coefficient from long-time MSD.

    D_eff = MSD / (4*t) in 2D.
    Uses linear fit to last `fit_frac` of the data.
    """
    if len(lag_times) < 5:
        return 0.0, 0.0

    n = len(lag_times)
    start = max(1, int(n * (1 - fit_frac)))
    t_fit = lag_times[start:]
    msd_fit = msd_vals[start:]

    if len(t_fit) < 3:
        return 0.0, 0.0

    # Linear fit: MSD = 4*D*t + b
    coeffs = np.polyfit(t_fit, msd_fit, 1)
    slope = coeffs[0]
    D_eff = slope / 4.0  # 2D
    # Uncertainty from residuals
    residuals = msd_fit - np.polyval(coeffs, t_fit)
    slope_err = np.sqrt(np.sum(residuals ** 2) / (len(t_fit) - 2)) / np.sqrt(
        np.sum((t_fit - np.mean(t_fit)) ** 2)
    )
    D_err = slope_err / 4.0

    return max(D_eff, 0.0), D_err


# ---------------------------------------------------------------------------
# Voronoi shape index
# ---------------------------------------------------------------------------

def compute_voronoi_shape_index(centroids, Lx, Ly):
    """Compute Voronoi shape index q = P/sqrt(A) for each cell.

    Uses periodic images (3x3 replication) to handle PBCs.
    Returns array of shape indices for cells in the central box.
    """
    from scipy.spatial import Voronoi

    N = len(centroids)
    if N < 4:
        return np.array([])

    # Create 3x3 periodic replicas
    offsets = []
    for dx in [-Lx, 0, Lx]:
        for dy in [-Ly, 0, Ly]:
            offsets.append([dx, dy])
    offsets = np.array(offsets)

    replicated = []
    for off in offsets:
        replicated.append(centroids + off)
    all_points = np.vstack(replicated)

    try:
        vor = Voronoi(all_points)
    except Exception:
        return np.array([])

    # Only analyze the central copy (indices 4*N to 5*N, offset [0,0] is index 4)
    central_idx = 4  # [0,0] is the 5th offset (index 4 in 3x3 grid)
    shape_indices = []

    for i in range(central_idx * N, (central_idx + 1) * N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        vertices = vor.vertices[region]
        # Polygon area (shoelace)
        x = vertices[:, 0]
        y = vertices[:, 1]
        A = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        # Polygon perimeter
        P = np.sum(np.sqrt(np.diff(np.append(x, x[0])) ** 2 +
                           np.diff(np.append(y, y[0])) ** 2))
        if A > 0:
            shape_indices.append(P / np.sqrt(A))

    return np.array(shape_indices)


# ---------------------------------------------------------------------------
# Bond-orientational order ψ₆ and g₆(r)
# ---------------------------------------------------------------------------

def compute_psi6(centroids, Lx, Ly, r_cut):
    """Compute ψ₆ for each cell.

    ψ₆(j) = (1/n_j) Σ_k exp(6i θ_jk)
    where k are neighbors within r_cut.
    """
    N = len(centroids)
    psi6 = np.zeros(N, dtype=complex)

    for j in range(N):
        n_neighbors = 0
        psi_sum = 0.0j
        for k in range(N):
            if k == j:
                continue
            dx = centroids[k, 0] - centroids[j, 0]
            dy = centroids[k, 1] - centroids[j, 1]
            # Periodic BCs
            if dx > Lx / 2:
                dx -= Lx
            elif dx < -Lx / 2:
                dx += Lx
            if dy > Ly / 2:
                dy -= Ly
            elif dy < -Ly / 2:
                dy += Ly
            r = np.sqrt(dx ** 2 + dy ** 2)
            if r < r_cut and r > 1e-10:
                theta = np.arctan2(dy, dx)
                psi_sum += np.exp(6j * theta)
                n_neighbors += 1
        if n_neighbors > 0:
            psi6[j] = psi_sum / n_neighbors

    return psi6


def compute_g6(centroids, psi6, Lx, Ly, n_bins=50, r_max=None):
    """Compute g₆(r) = ⟨ψ₆(r)ψ₆*(0)⟩ as function of separation r."""
    N = len(centroids)
    if r_max is None:
        r_max = min(Lx, Ly) / 2

    dr = r_max / n_bins
    g6_hist = np.zeros(n_bins)
    g6_counts = np.zeros(n_bins)

    for j in range(N):
        for k in range(j + 1, N):
            dx = centroids[k, 0] - centroids[j, 0]
            dy = centroids[k, 1] - centroids[j, 1]
            if dx > Lx / 2:
                dx -= Lx
            elif dx < -Lx / 2:
                dx += Lx
            if dy > Ly / 2:
                dy -= Ly
            elif dy < -Ly / 2:
                dy += Ly
            r = np.sqrt(dx ** 2 + dy ** 2)
            b = int(r / dr)
            if 0 <= b < n_bins:
                g6_val = np.real(psi6[j] * np.conj(psi6[k]))
                g6_hist[b] += g6_val
                g6_counts[b] += 1

    valid = g6_counts > 0
    r_vals = (np.arange(n_bins) + 0.5) * dr
    g6_vals = np.where(valid, g6_hist / g6_counts, 0.0)

    return r_vals[valid], g6_vals[valid]


# ---------------------------------------------------------------------------
# Extract centroids at a specific time from trajectory data
# ---------------------------------------------------------------------------

def extract_centroids(data, target_time, N):
    """Get cell centroids at the time closest to target_time."""
    times = np.unique(data[:, 0])
    closest_t = times[np.argmin(np.abs(times - target_time))]
    mask = data[:, 0] == closest_t
    snapshot = data[mask]

    centroids = np.full((N, 2), np.nan)
    for row in snapshot:
        cid = int(row[1])
        if cid < N:
            centroids[cid] = [row[2], row[3]]

    valid = ~np.isnan(centroids[:, 0])
    return centroids[valid], closest_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bresler validation analysis")
    parser.add_argument("base_dir", help="Production data base directory")
    parser.add_argument("--output", "-o", default="validation_results.npz",
                        help="Output .npz file")
    parser.add_argument("--max-runs", type=int, default=20,
                        help="Max runs per motility to process")
    parser.add_argument("--vA-list", type=str, default=None,
                        help="Comma-separated list of vA values to process")
    args = parser.parse_args()

    base = args.base_dir

    # Discover motility directories
    va_dirs = sorted(glob.glob(os.path.join(base, "vA_*")))
    if not va_dirs:
        print(f"No vA_* directories found in {base}", file=sys.stderr)
        sys.exit(1)

    # Filter vA values if specified
    if args.vA_list:
        target_vAs = set(args.vA_list.split(","))
        va_dirs = [d for d in va_dirs
                   if os.path.basename(d).replace("vA_", "") in target_vAs]

    print(f"Found {len(va_dirs)} motility directories")

    # Parameters
    R = 49
    r_cut = 2.5 * 2 * R  # Neighbor cutoff for ψ₆ (Bresler uses 2.5*R₀ but with R₀=2 in their units)
    # In our model, R=49 grid units, so 2.5 * (2*R) ~ 2.5 * cell diameter
    # But Bresler's 2.5*R₀ where R₀ is cell radius → r_cut = 2.5 * 49 = 122.5
    r_cut = 2.5 * R  # 122.5 grid units

    # Results storage
    results = {
        "vA_values": [],
        "msd_lags": [],      # list of arrays
        "msd_values": [],    # list of arrays averaged over runs
        "msd_stds": [],      # list of arrays
        "deff_values": [],
        "deff_errors": [],
        "voronoi_q_mean": [],
        "voronoi_q_std": [],
        "g6_r": [],
        "g6_values": [],
        "n_runs_processed": [],
        "params": {},
    }

    for va_dir in va_dirs:
        va_name = os.path.basename(va_dir)
        vA = float(va_name.replace("vA_", ""))
        print(f"\n{'='*60}")
        print(f"Processing {va_name} (vA = {vA})")
        print(f"{'='*60}")

        # Find run directories
        run_dirs = sorted(glob.glob(os.path.join(va_dir, "run_*")))
        if len(run_dirs) > args.max_runs:
            run_dirs = run_dirs[: args.max_runs]

        print(f"  Processing {len(run_dirs)} runs")

        # ---- MSD computation ----
        all_msds = []  # list of (lag, msd) per run
        ref_lags = None
        n_cells = 288
        Lx = Ly = 1600
        deff_per_run = []

        for i, rd in enumerate(run_dirs):
            traj_file = os.path.join(rd, "trajectory.txt")
            if not os.path.isfile(traj_file):
                continue

            t0 = clock.time()
            result = compute_msd_single_run(traj_file)
            dt = clock.time() - t0

            if result is None:
                continue

            lags, msd, counts, N = result
            n_cells = N
            print(f"    run {i+1}/{len(run_dirs)}: {len(lags)} lag points, "
                  f"t_max={lags[-1]:.0f}, took {dt:.1f}s")

            if ref_lags is None:
                ref_lags = lags
            # Interpolate to common lag grid
            if len(lags) > 0:
                msd_interp = np.interp(ref_lags, lags, msd, left=np.nan, right=np.nan)
                all_msds.append(msd_interp)

                # D_eff per run
                d, _ = compute_deff_from_msd(lags, msd)
                deff_per_run.append(d)

            # Read params from first file
            if i == 0:
                _, params = load_trajectory(traj_file)
                Lx = float(params.get("Lx", 1600))
                Ly = float(params.get("Ly", 1600))
                results["params"] = {
                    "N": int(params.get("N", 288)),
                    "Lx": Lx,
                    "Ly": Ly,
                    "R": R,
                    "rho": float(params.get("N", 288)) * np.pi * R ** 2 / (Lx * Ly),
                    "tau": 10000,
                }

        # Average MSD across runs
        if all_msds:
            msd_stack = np.array(all_msds)
            msd_mean = np.nanmean(msd_stack, axis=0)
            msd_std = np.nanstd(msd_stack, axis=0) / np.sqrt(np.sum(~np.isnan(msd_stack), axis=0))
        else:
            msd_mean = np.array([])
            msd_std = np.array([])
            ref_lags = np.array([])

        # Ensemble D_eff
        if deff_per_run:
            deff_mean = np.mean(deff_per_run)
            deff_err = np.std(deff_per_run) / np.sqrt(len(deff_per_run))
        else:
            deff_mean = 0.0
            deff_err = 0.0

        print(f"  D_eff = {deff_mean:.6f} ± {deff_err:.6f}")

        # ---- Voronoi shape index (from late-time snapshots) ----
        all_q = []
        # Use last few snapshots from a subset of runs
        for rd in run_dirs[:min(10, len(run_dirs))]:
            traj_file = os.path.join(rd, "trajectory.txt")
            if not os.path.isfile(traj_file):
                continue
            data, params = load_trajectory(traj_file)
            if len(data) == 0:
                continue

            N = int(params.get("N", 288))
            times = np.unique(data[:, 0])
            # Use last 5 snapshots
            for t in times[-5:]:
                centroids, _ = extract_centroids(data, t, N)
                if len(centroids) < N * 0.9:
                    continue
                q = compute_voronoi_shape_index(centroids, Lx, Ly)
                if len(q) > 0:
                    all_q.extend(q.tolist())

        if all_q:
            q_mean = np.mean(all_q)
            q_std = np.std(all_q)
        else:
            q_mean = q_std = 0.0
        print(f"  Voronoi <q> = {q_mean:.4f} ± {q_std:.4f}")

        # ---- Sixfold bond order g₆(r) (from late-time snapshots) ----
        g6_accum = None
        g6_count = 0
        for rd in run_dirs[:min(5, len(run_dirs))]:
            traj_file = os.path.join(rd, "trajectory.txt")
            if not os.path.isfile(traj_file):
                continue
            data, params = load_trajectory(traj_file)
            if len(data) == 0:
                continue
            N = int(params.get("N", 288))
            times = np.unique(data[:, 0])
            # Last snapshot
            centroids, _ = extract_centroids(data, times[-1], N)
            if len(centroids) < N * 0.9:
                continue

            psi6 = compute_psi6(centroids, Lx, Ly, r_cut)
            r_g6, g6_vals = compute_g6(centroids, psi6, Lx, Ly, n_bins=40)

            if g6_accum is None:
                g6_accum = g6_vals.copy()
                g6_r = r_g6.copy()
                g6_count = 1
            elif len(g6_vals) == len(g6_accum):
                g6_accum += g6_vals
                g6_count += 1

        if g6_accum is not None and g6_count > 0:
            g6_mean = g6_accum / g6_count
        else:
            g6_r = np.array([])
            g6_mean = np.array([])

        # Store results
        results["vA_values"].append(vA)
        results["msd_lags"].append(ref_lags if ref_lags is not None else np.array([]))
        results["msd_values"].append(msd_mean)
        results["msd_stds"].append(msd_std)
        results["deff_values"].append(deff_mean)
        results["deff_errors"].append(deff_err)
        results["voronoi_q_mean"].append(q_mean)
        results["voronoi_q_std"].append(q_std)
        results["g6_r"].append(g6_r)
        results["g6_values"].append(g6_mean)
        results["n_runs_processed"].append(len(all_msds))

    # ---- Save results ----
    save_dict = {
        "vA_values": np.array(results["vA_values"]),
        "deff_values": np.array(results["deff_values"]),
        "deff_errors": np.array(results["deff_errors"]),
        "voronoi_q_mean": np.array(results["voronoi_q_mean"]),
        "voronoi_q_std": np.array(results["voronoi_q_std"]),
        "n_runs_processed": np.array(results["n_runs_processed"]),
    }
    # Store parameters
    for k, v in results["params"].items():
        save_dict[f"param_{k}"] = v

    # Store MSD and g6 as variable-length arrays
    for i, vA in enumerate(results["vA_values"]):
        tag = f"vA{vA:.3f}".replace(".", "p")
        save_dict[f"msd_lag_{tag}"] = results["msd_lags"][i]
        save_dict[f"msd_val_{tag}"] = results["msd_values"][i]
        save_dict[f"msd_std_{tag}"] = results["msd_stds"][i]
        save_dict[f"g6_r_{tag}"] = results["g6_r"][i]
        save_dict[f"g6_val_{tag}"] = results["g6_values"][i]

    np.savez_compressed(args.output, **save_dict)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'vA':>8s}  {'D_eff':>10s}  {'D_err':>8s}  {'<q_V>':>8s}  {'n_runs':>6s}")
    for i, vA in enumerate(results["vA_values"]):
        print(f"{vA:8.4f}  {results['deff_values'][i]:10.6f}  "
              f"{results['deff_errors'][i]:8.6f}  "
              f"{results['voronoi_q_mean'][i]:8.4f}  "
              f"{results['n_runs_processed'][i]:6d}")

    # Analytical reference
    tau = 10000
    print(f"\nAnalytical D_eff(isolated) = vA²τ/2:")
    for vA in results["vA_values"]:
        print(f"  vA={vA:.4f}: D_iso = {vA**2 * tau / 2:.6f}")

    # Packing fraction
    rho = results["params"].get("rho", 0)
    print(f"\nPacking fraction ρ = {rho:.4f}")
    print(f"Bresler jamming concentration ρ_J ≈ 0.843")

    # Jamming velocity estimate
    deff = np.array(results["deff_values"])
    vAs = np.array(results["vA_values"])
    # Linear fit to upper half of D_eff
    above_noise = deff > 0.5 * np.max(deff)
    if np.sum(above_noise) >= 2:
        coeffs = np.polyfit(vAs[above_noise], deff[above_noise], 1)
        v_star = -coeffs[1] / coeffs[0] if coeffs[0] > 0 else 0
        print(f"Estimated jamming velocity v* ≈ {v_star:.5f} (from linear fit)")


if __name__ == "__main__":
    main()
