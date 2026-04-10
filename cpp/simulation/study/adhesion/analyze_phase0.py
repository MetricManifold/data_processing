#!/usr/bin/env python3
"""
Phase 0 Adhesion Quench Analysis

Analyzes centroid displacement and MSD across J/kappa values from the
adhesion quench experiment (v_A=0, varying J). Produces summary table,
displacement-vs-J/kappa plot, and MSD time series.

Usage:
    python analyze_phase0.py [data_dir]

    data_dir defaults to the local phase0_quench/ directory.
    Pass a cluster download path to analyze cluster data.

Figures saved to: postprocessing/output/adhesion_phase0_*.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import date

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "cpp", "simulation", "postprocessing", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "phase0_quench")

# Correct domain size for phi=0.89, N=288, R=49
L = 1562.0
R = 49.0
TAU = 10000.0  # tumble time

# All J/kappa values including control
JK_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

TODAY = date.today().strftime("%Y%m%d")


def parse_header(filepath):
    """Extract v_A, N, Lx, Ly from trajectory header line."""
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("# v_A="):
                parts = {}
                for token in line[2:].strip().split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        parts[k] = float(v)
                return parts
            if not line.startswith("#"):
                break
    return {}


def load_trajectory(filepath):
    """Load trajectory, return (times, positions_dict, velocities_dict, header).

    positions_dict[cell_id] = array of (x, y) at each unique time.
    """
    header = parse_header(filepath)
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    times = np.unique(data[:, 0])
    n_cells_per_frame = int(np.sum(data[:, 0] == times[0]))

    # Sort by (time, cell_id) for consistent ordering
    sort_idx = np.lexsort((data[:, 1], data[:, 0]))
    data = data[sort_idx]

    # Reshape: (n_times, n_cells, n_cols)
    n_times = len(times)
    n_cols = data.shape[1]
    # Handle possible incomplete last frame
    expected_rows = n_times * n_cells_per_frame
    if len(data) > expected_rows:
        data = data[:expected_rows]
    elif len(data) < expected_rows:
        n_times = len(data) // n_cells_per_frame
        data = data[: n_times * n_cells_per_frame]
        times = times[:n_times]

    frames = data.reshape(n_times, n_cells_per_frame, n_cols)

    return times, frames, n_cells_per_frame, header


def unwrap_displacement(x, x0, domain):
    """Minimum image displacement."""
    dx = x - x0
    return dx - domain * np.round(dx / domain)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

print("=" * 80)
print("Phase 0 Adhesion Quench Analysis")
print(f"Data directory: {BASE}")
print(f"Domain L = {L}, R = {R}, tau = {TAU}")
print("=" * 80)
print()

results = []
msd_data = {}  # jk -> (sample_times, msd_values)

for jk in JK_VALUES:
    dirname = f"Jk_{jk:.2f}"
    traj_path = os.path.join(BASE, dirname, "trajectory.txt")

    if not os.path.exists(traj_path):
        print(f"  {dirname}: MISSING")
        continue

    times, frames, n_cells, header = load_trajectory(traj_path)
    n_times = len(times)

    # Use header domain size if available, otherwise use default
    Lx = header.get("Lx", L)
    Ly = header.get("Ly", L)

    # Initial and final positions
    x0 = frames[0, :, 2]
    y0 = frames[0, :, 3]
    xf = frames[-1, :, 2]
    yf = frames[-1, :, 3]

    dx = unwrap_displacement(xf, x0, Lx)
    dy = unwrap_displacement(yf, y0, Ly)
    disp = np.sqrt(dx ** 2 + dy ** 2)

    # MSD time series (sample ~200 points for plotting)
    stride = max(1, n_times // 200)
    sample_idx = list(range(0, n_times, stride))
    if sample_idx[-1] != n_times - 1:
        sample_idx.append(n_times - 1)
    msd_times = []
    msd_vals = []
    for i in sample_idx:
        dxt = unwrap_displacement(frames[i, :, 2], x0, Lx)
        dyt = unwrap_displacement(frames[i, :, 3], y0, Ly)
        msd_vals.append(np.mean(dxt ** 2 + dyt ** 2))
        msd_times.append(times[i] - times[0])

    msd_data[jk] = (np.array(msd_times), np.array(msd_vals))

    # RMS velocity at final time
    vx_f = frames[-1, :, 4]
    vy_f = frames[-1, :, 5]
    v_rms = np.sqrt(np.mean(vx_f ** 2 + vy_f ** 2))

    # Mid-run displacement
    mid_idx = n_times // 2
    dx_mid = unwrap_displacement(frames[mid_idx, :, 2], x0, Lx)
    dy_mid = unwrap_displacement(frames[mid_idx, :, 3], y0, Ly)
    mean_disp_mid = np.mean(np.sqrt(dx_mid ** 2 + dy_mid ** 2))

    results.append({
        "jk": jk,
        "J": jk * 10,
        "mean_disp": np.mean(disp),
        "max_disp": np.max(disp),
        "total_disp": np.sum(disp),
        "v_rms_final": v_rms,
        "mean_disp_mid": mean_disp_mid,
        "n_snapshots": n_times,
        "n_cells": n_cells,
        "msd_final": msd_vals[-1],
        "t_start": times[0],
        "t_end": times[-1],
    })

    still_moving = "YES" if v_rms > 1e-4 else "no"
    print(f"J/κ = {jk:.2f}  (J = {jk * 10:.1f})")
    print(f"  Cells: {n_cells}, Snapshots: {n_times}")
    print(f"  Time range: {times[0]:.0f} → {times[-1]:.0f} ({times[-1]-times[0]:.0f} TU = {(times[-1]-times[0])/TAU:.1f}τ)")
    print(f"  Mean displacement: {np.mean(disp):.4f}  (at t_mid: {mean_disp_mid:.4f})")
    print(f"  Max displacement:  {np.max(disp):.4f}  ({np.max(disp)/R:.2f}R)")
    print(f"  Final MSD: {msd_vals[-1]:.6f}")
    print(f"  Final v_rms: {v_rms:.6f}  (still moving: {still_moving})")
    print()

if not results:
    print("No data found. Exiting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'J/κ':>6} {'J':>6} {'<|Δr|>':>10} {'max|Δr|':>10} {'max/R':>7} {'MSD_final':>12} {'v_rms':>10}")
print("-" * 70)
for r in results:
    print(
        f"{r['jk']:>6.2f} {r['J']:>6.1f} {r['mean_disp']:>10.4f} "
        f"{r['max_disp']:>10.4f} {r['max_disp']/R:>7.2f} "
        f"{r['msd_final']:>12.6f} {r['v_rms_final']:>10.6f}"
    )

# Check control
control = [r for r in results if r["jk"] == 0.00]
if control:
    ctrl_disp = control[0]["mean_disp"]
    print(f"\nControl (J=0) mean displacement: {ctrl_disp:.6f}")
    if ctrl_disp > 0.1:
        print("  ⚠️  Control displacement > 0.1 — starting state may not be fully equilibrated!")
    else:
        print("  ✓  Control looks clean (negligible displacement)")
else:
    print("\n⚠️  No J=0 control found! Cannot verify equilibration quality.")

# ---------------------------------------------------------------------------
# Transition detection
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("DIAGNOSTIC: Transition Detection")
print("=" * 80)
jks = [r["jk"] for r in results]
disps = [r["mean_disp"] for r in results]

if len(disps) > 1:
    ratios = [disps[i + 1] / max(disps[i], 1e-10) for i in range(len(disps) - 1)]
    max_ratio_idx = int(np.argmax(ratios))
    print(f"Largest displacement jump: J/κ = {jks[max_ratio_idx]:.2f} → {jks[max_ratio_idx+1]:.2f}")
    print(f"  Ratio: {ratios[max_ratio_idx]:.2f}×")

    significant = [r for r in results if r["max_disp"] > R]
    if significant:
        print(f"\nSignificant rearrangements (max_disp > R={R}):")
        for s in significant:
            print(f"  J/κ = {s['jk']:.2f}: max_disp = {s['max_disp']:.2f} ({s['max_disp']/R:.1f}R)")
    else:
        print(f"\nNo cell moved more than R={R}.")

# ---------------------------------------------------------------------------
# Figure 1: Displacement vs J/kappa
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(jks, disps, "o-", color="C0", markersize=8, linewidth=2)
if control:
    ax.axhline(ctrl_disp, color="gray", linestyle="--", alpha=0.5, label=f"Control (J=0): {ctrl_disp:.4f}")
    ax.legend()
ax.set_xlabel("$J/\\kappa$", fontsize=13)
ax.set_ylabel("Mean displacement $\\langle|\\Delta\\mathbf{r}|\\rangle$", fontsize=13)
ax.set_title("Phase 0: Displacement vs Adhesion Strength")
ax.set_xlim(-0.02, 0.52)

ax = axes[1]
max_disps = [r["max_disp"] for r in results]
ax.plot(jks, max_disps, "s-", color="C1", markersize=8, linewidth=2)
ax.axhline(R, color="red", linestyle=":", alpha=0.5, label=f"Cell radius R={R}")
ax.set_xlabel("$J/\\kappa$", fontsize=13)
ax.set_ylabel("Max displacement $\\max|\\Delta\\mathbf{r}|$", fontsize=13)
ax.set_title("Phase 0: Maximum Cell Displacement")
ax.set_xlim(-0.02, 0.52)
ax.legend()

plt.tight_layout()
disp_path = os.path.join(OUTPUT_DIR, f"adhesion_phase0_displacement_vs_Jk_{TODAY}.png")
plt.savefig(disp_path, dpi=150)
print(f"\nSaved: {disp_path}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: MSD time series for all J/kappa
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for jk in sorted(msd_data.keys()):
    t, msd = msd_data[jk]
    label = f"$J/\\kappa={jk:.2f}$"
    style = "--" if jk == 0.0 else "-"
    ax.plot(t / TAU, msd, style, label=label, linewidth=1.5)
ax.set_xlabel("Time ($\\tau$)", fontsize=13)
ax.set_ylabel("MSD", fontsize=13)
ax.set_title("Phase 0: MSD Time Series")
ax.legend(fontsize=8, ncol=2)

ax = axes[1]
for jk in sorted(msd_data.keys()):
    t, msd = msd_data[jk]
    mask = (t > 0) & (msd > 0)
    if np.any(mask):
        style = "--" if jk == 0.0 else "-"
        ax.loglog(t[mask] / TAU, msd[mask], style, label=f"$J/\\kappa={jk:.2f}$", linewidth=1.5)
ax.set_xlabel("Time ($\\tau$)", fontsize=13)
ax.set_ylabel("MSD", fontsize=13)
ax.set_title("Phase 0: MSD (log-log)")
ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
msd_path = os.path.join(OUTPUT_DIR, f"adhesion_phase0_msd_timeseries_{TODAY}.png")
plt.savefig(msd_path, dpi=150)
print(f"Saved: {msd_path}")
plt.close()

print("\nDone.")
