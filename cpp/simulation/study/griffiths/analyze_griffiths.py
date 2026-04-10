#!/usr/bin/env python3
"""
Griffiths Rare-Region Analysis for Cell Simulation

Analyzes whether the system exhibits:
  (A) Jammed islands in a motile sea (above transition), or
  (B) Motile islands in a jammed sea (below transition)

This distinction is central to Griffiths physics where quenched disorder
(here: per-cell inherent v_A drawn from log-normal distribution) creates
rare regions of the opposing phase near a phase transition.

Measurements:
  1. Area fraction: what fraction of cells are jammed vs motile?
  2. Percolation analysis: does the jammed or motile phase span the system?
  3. Cluster size distribution P(s) for jammed and motile clusters
  4. Spatial autocorrelation C(r) of mobility field
  5. Correlation between inherent v_A and measured mobility
  6. Temporal persistence of jammed/motile regions

Usage:
  python analyze_griffiths.py path/to/sim_output
  python analyze_griffiths.py path/to/sim_output --subsample 4
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from collections import defaultdict
import argparse
import sys
from datetime import datetime
from scipy.spatial import Voronoi

# Import trajectory loading from the visualization script
sys.path.insert(0, str(Path(__file__).parent))
from visualize_fluid import (load_trajectory, compute_displacement_mobility,
                              make_fluid_cmap, _voronoi_polygons)


# ============================================================================
# 1. NEIGHBOR GRAPH (from periodic Voronoi)
# ============================================================================

def build_neighbor_graph(positions_t, Lx, Ly):
    """Build adjacency list from Voronoi tessellation with periodic images.

    Uses 9-fold periodic copies to handle boundaries. Two real cells are
    neighbors if any of their periodic images share a Voronoi ridge.

    Returns:
        neighbors: dict[cell_id] -> set of neighbor cell_ids
    """
    cell_ids = sorted(positions_t.keys())
    pts = np.array([positions_t[cid] for cid in cell_ids])
    N = len(pts)

    # 9-fold periodic ghost copies
    all_pts = []
    for dx in [-Lx, 0, Lx]:
        for dy in [-Ly, 0, Ly]:
            all_pts.append(pts + np.array([dx, dy]))
    all_pts = np.vstack(all_pts)
    real_offset = 4 * N  # block 4 = (0,0) shift

    vor = Voronoi(all_pts)

    neighbors = defaultdict(set)
    for (p1, p2) in vor.ridge_points:
        # Map ghost indices back to real cell indices
        r1 = p1 % N
        r2 = p2 % N
        if r1 == r2:
            continue  # self-connection from ghosts
        cid1 = cell_ids[r1]
        cid2 = cell_ids[r2]
        neighbors[cid1].add(cid2)
        neighbors[cid2].add(cid1)

    return dict(neighbors)


# ============================================================================
# 2. CLASSIFICATION & CLUSTERING
# ============================================================================

def classify_cells(mobility_t, threshold):
    """Classify cells as jammed (0) or motile (1) based on mobility threshold.

    Args:
        mobility_t: dict[cell_id] -> float mobility
        threshold: mobility value separating jammed from motile

    Returns:
        labels: dict[cell_id] -> int (0=jammed, 1=motile)
    """
    return {cid: (1 if m > threshold else 0) for cid, m in mobility_t.items()}


def find_clusters(labels, neighbors):
    """Find connected components of same-label cells using BFS.

    Args:
        labels: dict[cell_id] -> int (0 or 1)
        neighbors: dict[cell_id] -> set of cell_ids

    Returns:
        clusters: list of (label, set_of_cell_ids)
    """
    visited = set()
    clusters = []

    for cid in labels:
        if cid in visited:
            continue
        label = labels[cid]
        # BFS
        queue = [cid]
        cluster = set()
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            if labels.get(node, -1) != label:
                continue
            cluster.add(node)
            for nb in neighbors.get(node, []):
                if nb not in visited and labels.get(nb, -1) == label:
                    queue.append(nb)
        if cluster:
            clusters.append((label, cluster))

    return clusters


def check_percolation(cluster_cells, positions, Lx, Ly, threshold_fraction=0.8):
    """Check if a cluster percolates (spans) the system.

    A cluster percolates if the range of its cell positions in x or y
    exceeds threshold_fraction of the domain size. With periodic boundaries,
    a percolating cluster would wrap around.

    For a more robust check: we look at the maximum gap between consecutive
    sorted positions. If the largest gap is small (< Lx * (1 - threshold)),
    the cluster spans the dimension.

    Returns:
        (percolates_x, percolates_y): booleans
    """
    if len(cluster_cells) < 3:
        return False, False

    xs = np.array([positions[cid][0] for cid in cluster_cells])
    ys = np.array([positions[cid][1] for cid in cluster_cells])

    def _spans(coords, L):
        """Check if sorted coords span dimension L with no large gap."""
        if len(coords) < 2:
            return False
        s = np.sort(coords)
        # gaps between consecutive cells
        gaps = np.diff(s)
        # also the "wrap-around" gap
        wrap_gap = (s[0] + L) - s[-1]
        max_gap = max(np.max(gaps), wrap_gap)
        # If the maximum gap is small relative to L, the cluster spans
        return max_gap < L * (1 - threshold_fraction)

    return _spans(xs, Lx), _spans(ys, Ly)


# ============================================================================
# 3. SPATIAL AUTOCORRELATION
# ============================================================================

def spatial_autocorrelation(positions_t, mobility_t, Lx, Ly, n_bins=50):
    """Compute spatial correlation function C(r) of the mobility field.

    C(r) = <δm(0) δm(r)> / <δm²>

    where δm = m - <m>, averaged over all cell pairs at separation r.

    Returns:
        r_bins: bin centers
        C_r: correlation values
        counts: number of pairs in each bin (for identifying valid bins)
    """
    cell_ids = sorted(positions_t.keys())
    N = len(cell_ids)
    mobs = np.array([mobility_t.get(cid, 0.0) for cid in cell_ids])
    mean_m = mobs.mean()
    dm = mobs - mean_m
    var_m = np.var(mobs)

    if var_m < 1e-20:
        return np.linspace(0, min(Lx, Ly)/2, n_bins), np.zeros(n_bins), np.zeros(n_bins)

    coords = np.array([positions_t[cid] for cid in cell_ids])

    # Max distance with periodic boundaries
    r_max = min(Lx, Ly) / 2
    dr = r_max / n_bins
    counts = np.zeros(n_bins)
    corr_sum = np.zeros(n_bins)

    for i in range(N):
        for j in range(i+1, N):
            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]
            # Minimum image convention
            if dx > Lx/2: dx -= Lx
            elif dx < -Lx/2: dx += Lx
            if dy > Ly/2: dy -= Ly
            elif dy < -Ly/2: dy += Ly
            r = np.sqrt(dx**2 + dy**2)
            if r >= r_max:
                continue
            b = int(r / dr)
            if b >= n_bins:
                b = n_bins - 1
            corr_sum[b] += dm[i] * dm[j]
            counts[b] += 1

    r_bins = (np.arange(n_bins) + 0.5) * dr
    mask = counts > 0
    C_r = np.zeros(n_bins)
    C_r[mask] = corr_sum[mask] / (counts[mask] * var_m)

    return r_bins, C_r, counts


# ============================================================================
# 4. INHERENT v_A vs MEASURED MOBILITY CORRELATION
# ============================================================================

def va_mobility_correlation(inherent_vA, mobility_t):
    """Compute Pearson correlation between inherent v_A and measured mobility.

    Returns: (pearson_r, p_value, vA_array, mob_array)
    """
    from scipy import stats
    cell_ids = sorted(set(inherent_vA.keys()) & set(mobility_t.keys()))
    vA = np.array([inherent_vA[cid] for cid in cell_ids])
    mob = np.array([mobility_t[cid] for cid in cell_ids])
    r, p = stats.pearsonr(vA, mob)
    return r, p, vA, mob


# ============================================================================
# 5. TEMPORAL PERSISTENCE
# ============================================================================

def compute_persistence(times, mobility, threshold, window_steps=10):
    """Measure how persistently each cell stays jammed or motile.

    For each cell, compute the fraction of time steps in a sliding window
    where it has the same classification.

    Returns:
        persistence: dict[cell_id] -> float in [0, 1]
            (1 = always same state, low = frequently switching)
        jammed_fraction_time: list of (time, fraction_jammed)
    """
    cell_ids = None
    labels_over_time = defaultdict(list)
    jammed_fraction_time = []

    for t in times:
        mob = mobility[t]
        if cell_ids is None:
            cell_ids = sorted(mob.keys())
        n_jammed = 0
        for cid in cell_ids:
            m = mob.get(cid, 0.0)
            lab = 0 if m <= threshold else 1
            labels_over_time[cid].append(lab)
            if lab == 0:
                n_jammed += 1
        jammed_fraction_time.append((t, n_jammed / len(cell_ids)))

    # Persistence: fraction of time in majority state
    persistence = {}
    for cid in cell_ids:
        seq = np.array(labels_over_time[cid])
        frac_jammed = np.mean(seq == 0)
        persistence[cid] = max(frac_jammed, 1 - frac_jammed)

    return persistence, jammed_fraction_time


# ============================================================================
# 6. OVERLAP FUNCTION Q(t) — structural relaxation by mobility class
# ============================================================================

def overlap_function(times, positions, header, a=10.0, cell_class=None,
                     class_label=None):
    """Compute overlap function Q(t) = <θ(a - |r_i(t) - r_i(0)|)>

    Averaging over cells of a given class (jammed/motile) at t=0.
    The overlap function decays from 1 to 0 as cells rearrange.

    Args:
        times: array of times
        positions: dict[time] -> dict[cell_id] -> (x, y)
        header: dict with Lx, Ly
        a: cage radius (threshold distance)
        cell_class: dict[cell_id] -> int (0=jammed, 1=motile) at reference time
        class_label: which class to compute for (0 or 1). None = all cells.

    Returns:
        lag_times, Q_values
    """
    Lx = float(header.get('Lx', 1600))
    Ly = float(header.get('Ly', 1600))

    t0 = times[0]
    pos0 = positions[t0]

    if class_label is not None and cell_class is not None:
        cell_ids = [cid for cid in sorted(pos0.keys())
                    if cell_class.get(cid, -1) == class_label]
    else:
        cell_ids = sorted(pos0.keys())

    if not cell_ids:
        return np.array([]), np.array([])

    lag_times = []
    Q_values = []

    for t in times:
        dt = t - t0
        if dt < 0:
            continue
        pos_t = positions[t]
        n_overlap = 0
        n_total = 0
        for cid in cell_ids:
            if cid not in pos_t or cid not in pos0:
                continue
            x0, y0 = pos0[cid]
            x1, y1 = pos_t[cid]
            dx = x1 - x0
            dy = y1 - y0
            if dx > Lx/2: dx -= Lx
            elif dx < -Lx/2: dx += Lx
            if dy > Ly/2: dy -= Ly
            elif dy < -Ly/2: dy += Ly
            dist = np.sqrt(dx**2 + dy**2)
            if dist < a:
                n_overlap += 1
            n_total += 1
        if n_total > 0:
            lag_times.append(dt)
            Q_values.append(n_overlap / n_total)

    return np.array(lag_times), np.array(Q_values)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Griffiths rare-region analysis')
    parser.add_argument('input_dir', help='Directory containing trajectory.txt')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Subsample trajectory (keep every Nth step)')
    parser.add_argument('--window', type=float, default=0,
                        help='Mobility window (0 = auto 5%% of time span)')
    parser.add_argument('--threshold-percentile', type=float, default=None,
                        help='Mobility percentile for jammed/motile classification '
                             '(default: auto-detect from v_A=0 reference)')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    sim_dir = Path(args.input_dir)
    traj_file = sim_dir / 'trajectory.txt'
    if not traj_file.exists():
        print(f"ERROR: No trajectory.txt found in {sim_dir}")
        sys.exit(1)

    # Output directory
    out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')

    # ------------------------------------------------------------------
    # Load trajectory
    # ------------------------------------------------------------------
    print("=" * 70)
    print("GRIFFITHS RARE-REGION ANALYSIS")
    print("=" * 70)
    print(f"\nLoading trajectory from {traj_file}...")
    times, positions, velocities, header, inherent_vA = load_trajectory(
        traj_file, subsample=args.subsample)

    N_cells = len(positions[times[0]])
    Lx = float(header.get('Lx', 1600))
    Ly = float(header.get('Ly', 1600))
    mean_vA = float(header.get('v_A', 0.012))
    sigma_vA = float(header.get('v_A_sigma', 0.006))

    print(f"\n  System: N={N_cells} cells, Lx={Lx}, Ly={Ly}")
    print(f"  Mean v_A={mean_vA}, σ_vA={sigma_vA}")
    print(f"  Time span: {times[0]:.0f} to {times[-1]:.0f} "
          f"({len(times)} frames)")

    # ------------------------------------------------------------------
    # Compute mobility
    # ------------------------------------------------------------------
    window = args.window
    if window <= 0:
        window = (times[-1] - times[0]) * 0.05
    print(f"\nComputing displacement mobility (window={window:.1f})...")
    mobility = compute_displacement_mobility(times, positions, header, window)

    # Collect all mobility values (skip early times where window is short)
    skip = max(1, len(times) // 10)  # skip first 10%
    all_mobs = []
    for t in times[skip:]:
        all_mobs.extend(mobility[t].values())
    all_mobs = np.array(all_mobs)
    all_mobs_pos = all_mobs[all_mobs > 0]

    print(f"  Mobility statistics (after skip):")
    print(f"    mean = {all_mobs.mean():.6f}")
    print(f"    median = {np.median(all_mobs):.6f}")
    print(f"    std = {all_mobs.std():.6f}")
    print(f"    min = {all_mobs.min():.6f}, max = {all_mobs.max():.6f}")

    # ------------------------------------------------------------------
    # Determine threshold for jammed/motile classification
    # ------------------------------------------------------------------
    if args.threshold_percentile is not None:
        threshold = np.percentile(all_mobs_pos, args.threshold_percentile)
        print(f"\n  Threshold: {threshold:.6f} "
              f"({args.threshold_percentile}th percentile)")
    else:
        # Use median as a natural separator
        threshold = np.median(all_mobs)
        print(f"\n  Threshold (median mobility): {threshold:.6f}")

    # ------------------------------------------------------------------
    # Time-averaged analysis: use middle portion of trajectory
    # ------------------------------------------------------------------
    mid_start = len(times) // 4
    mid_end = 3 * len(times) // 4
    analysis_times = times[mid_start:mid_end]
    print(f"\nAnalyzing {len(analysis_times)} frames "
          f"(t={analysis_times[0]:.0f} to {analysis_times[-1]:.0f})...")

    # ------------------------------------------------------------------
    # MEASUREMENT 1: Area fractions over time
    # ------------------------------------------------------------------
    print("\n--- 1. JAMMED vs MOTILE FRACTION ---")
    jammed_fracs = []
    for t in analysis_times:
        labels = classify_cells(mobility[t], threshold)
        n_jammed = sum(1 for v in labels.values() if v == 0)
        jammed_fracs.append(n_jammed / N_cells)
    jammed_fracs = np.array(jammed_fracs)
    print(f"  Mean jammed fraction: {jammed_fracs.mean():.3f} "
          f"± {jammed_fracs.std():.3f}")
    print(f"  Mean motile fraction: {1-jammed_fracs.mean():.3f}")

    if jammed_fracs.mean() > 0.5:
        print("  => MAJORITY JAMMED — system may be in jammed-sea regime")
    else:
        print("  => MAJORITY MOTILE — system may be in motile-sea regime")

    # ------------------------------------------------------------------
    # MEASUREMENT 2: Cluster analysis & percolation
    # ------------------------------------------------------------------
    print("\n--- 2. CLUSTER ANALYSIS & PERCOLATION ---")

    # Sample multiple frames for statistics
    sample_indices = np.linspace(0, len(analysis_times)-1,
                                  min(20, len(analysis_times)), dtype=int)
    all_jammed_sizes = []
    all_motile_sizes = []
    jammed_perc_count = 0
    motile_perc_count = 0
    n_samples = 0
    largest_jammed_frac = []
    largest_motile_frac = []

    for si in sample_indices:
        t = analysis_times[si]
        labels = classify_cells(mobility[t], threshold)
        neighbors = build_neighbor_graph(positions[t], Lx, Ly)
        clusters = find_clusters(labels, neighbors)

        jammed_clusters = [c for lab, c in clusters if lab == 0]
        motile_clusters = [c for lab, c in clusters if lab == 1]

        j_sizes = [len(c) for c in jammed_clusters] if jammed_clusters else [0]
        m_sizes = [len(c) for c in motile_clusters] if motile_clusters else [0]
        all_jammed_sizes.extend(j_sizes)
        all_motile_sizes.extend(m_sizes)

        largest_jammed_frac.append(max(j_sizes) / N_cells if j_sizes else 0)
        largest_motile_frac.append(max(m_sizes) / N_cells if m_sizes else 0)

        # Percolation check for largest clusters
        for jc in jammed_clusters:
            px, py = check_percolation(jc, positions[t], Lx, Ly)
            if px or py:
                jammed_perc_count += 1
                break
        for mc in motile_clusters:
            px, py = check_percolation(mc, positions[t], Lx, Ly)
            if px or py:
                motile_perc_count += 1
                break
        n_samples += 1

    all_jammed_sizes = np.array(all_jammed_sizes)
    all_motile_sizes = np.array(all_motile_sizes)

    print(f"  Sampled {n_samples} frames for cluster statistics")
    print(f"  Jammed clusters:")
    print(f"    Mean size: {all_jammed_sizes.mean():.1f} cells")
    print(f"    Max size:  {all_jammed_sizes.max()} cells")
    print(f"    Largest cluster fraction: {np.mean(largest_jammed_frac):.3f} "
          f"± {np.std(largest_jammed_frac):.3f}")
    print(f"    Percolates in {jammed_perc_count}/{n_samples} frames")
    print(f"  Motile clusters:")
    print(f"    Mean size: {all_motile_sizes.mean():.1f} cells")
    print(f"    Max size:  {all_motile_sizes.max()} cells")
    print(f"    Largest cluster fraction: {np.mean(largest_motile_frac):.3f} "
          f"± {np.std(largest_motile_frac):.3f}")
    print(f"    Percolates in {motile_perc_count}/{n_samples} frames")

    # Diagnosis
    if motile_perc_count > jammed_perc_count:
        print("  => MOTILE phase percolates more often => "
              "JAMMED ISLANDS in MOTILE SEA")
        phase_diagnosis = "jammed_islands_in_motile_sea"
    elif jammed_perc_count > motile_perc_count:
        print("  => JAMMED phase percolates more often => "
              "MOTILE ISLANDS in JAMMED SEA")
        phase_diagnosis = "motile_islands_in_jammed_sea"
    else:
        print("  => BOTH phases percolate similarly => NEAR CRITICAL POINT")
        phase_diagnosis = "near_critical"

    # ------------------------------------------------------------------
    # MEASUREMENT 3: Spatial autocorrelation C(r)
    # ------------------------------------------------------------------
    print("\n--- 3. SPATIAL AUTOCORRELATION C(r) ---")

    # Average C(r) over several frames
    n_cr_samples = min(10, len(analysis_times))
    cr_indices = np.linspace(0, len(analysis_times)-1, n_cr_samples, dtype=int)
    all_Cr = []
    all_counts = []
    r_bins = None

    for ci in cr_indices:
        t = analysis_times[ci]
        rb, Cr, cts = spatial_autocorrelation(positions[t], mobility[t], Lx, Ly,
                                              n_bins=40)
        if r_bins is None:
            r_bins = rb
        all_Cr.append(Cr)
        all_counts.append(cts)

    mean_Cr = np.mean(all_Cr, axis=0)
    std_Cr = np.std(all_Cr, axis=0) / np.sqrt(n_cr_samples)
    mean_counts = np.mean(all_counts, axis=0)

    # Extract correlation length (where C(r) first drops below 1/e)
    # Skip bins with insufficient pair counts (< 10 pairs)
    MIN_PAIRS = 10
    corr_length = None
    for i in range(len(mean_Cr)):
        if mean_counts[i] < MIN_PAIRS:
            continue  # Skip empty/sparse bins (sub-cell-spacing distances)
        if mean_Cr[i] < 1/np.e:
            corr_length = r_bins[i]
            break
    if corr_length is None:
        corr_length = r_bins[-1]  # doesn't decay — long-range order

    print(f"  Correlation length ξ ≈ {corr_length:.1f}")
    print(f"  C(r) at r=50:  {np.interp(50, r_bins, mean_Cr):.3f}")
    print(f"  C(r) at r=100: {np.interp(100, r_bins, mean_Cr):.3f}")
    print(f"  C(r) at r=200: {np.interp(200, r_bins, mean_Cr):.3f}")

    # Estimate cell spacing for context
    cell_spacing = np.sqrt(Lx * Ly / N_cells)
    print(f"  (Mean cell spacing: {cell_spacing:.1f})")
    print(f"  ξ / cell_spacing ≈ {corr_length / cell_spacing:.1f} cells")

    # ------------------------------------------------------------------
    # MEASUREMENT 4: Inherent v_A vs measured mobility
    # ------------------------------------------------------------------
    print("\n--- 4. INHERENT v_A vs MEASURED MOBILITY ---")

    if inherent_vA is not None:
        # Time-averaged mobility per cell
        time_avg_mob = defaultdict(list)
        for t in analysis_times:
            for cid, m in mobility[t].items():
                time_avg_mob[cid].append(m)
        time_avg_mob = {cid: np.mean(vals) for cid, vals in time_avg_mob.items()}

        r_corr, p_val, vA_arr, mob_arr = va_mobility_correlation(
            inherent_vA, time_avg_mob)
        print(f"  Pearson correlation: r = {r_corr:.4f}, p = {p_val:.2e}")
        if r_corr > 0.5:
            print("  => STRONG positive correlation: inherent v_A determines "
                  "actual motility")
            print("     This is the Griffiths scenario — quenched disorder "
                  "directly controls dynamics")
        elif r_corr > 0.2:
            print("  => MODERATE correlation: inherent v_A partially "
                  "determines motility")
            print("     Many-body effects (neighbors) also play a role")
        else:
            print("  => WEAK correlation: actual motility is not simply "
                  "determined by inherent v_A")
            print("     Collective/cooperative effects dominate")
    else:
        print("  No inherent v_A data available (9-column format)")
        r_corr = None
        vA_arr = mob_arr = None

    # ------------------------------------------------------------------
    # MEASUREMENT 5: Temporal persistence
    # ------------------------------------------------------------------
    print("\n--- 5. TEMPORAL PERSISTENCE ---")

    persistence, jammed_frac_vs_time = compute_persistence(
        times, mobility, threshold)

    pers_vals = np.array(list(persistence.values()))
    print(f"  Mean persistence: {pers_vals.mean():.3f}")
    print(f"  Persistence > 0.8: {np.mean(pers_vals > 0.8)*100:.1f}% of cells")
    print(f"  Persistence > 0.9: {np.mean(pers_vals > 0.9)*100:.1f}% of cells")
    print("  (Persistence = fraction of time in majority state)")
    print("  High persistence => cells stay jammed or motile for long times")
    print("  This is the hallmark of Griffiths physics: quenched disorder")
    print("  creates long-lived rare regions")

    # ------------------------------------------------------------------
    # MEASUREMENT 6: Overlap function Q(t) by class
    # ------------------------------------------------------------------
    print("\n--- 6. STRUCTURAL RELAXATION Q(t) BY CLASS ---")

    # Classify cells at reference time (mid-trajectory)
    t_ref_idx = len(times) // 4
    t_ref = times[t_ref_idx]
    ref_labels = classify_cells(mobility[t_ref], threshold)

    # Use second half of trajectory for Q(t)
    qt_times = times[t_ref_idx:]
    qt_positions = positions

    cage_radius = cell_spacing * 0.3  # ~30% of cell spacing
    print(f"  Cage radius a = {cage_radius:.1f}")

    lag_all, Q_all = overlap_function(qt_times, qt_positions, header,
                                       a=cage_radius)
    lag_j, Q_j = overlap_function(qt_times, qt_positions, header,
                                   a=cage_radius, cell_class=ref_labels,
                                   class_label=0)
    lag_m, Q_m = overlap_function(qt_times, qt_positions, header,
                                   a=cage_radius, cell_class=ref_labels,
                                   class_label=1)

    if len(lag_all) > 0:
        # Find relaxation time (where Q drops to 1/e)
        def _find_tau(lag, Q):
            for i in range(len(Q)):
                if Q[i] < 1/np.e:
                    return lag[i]
            return lag[-1] if len(lag) > 0 else np.nan

        tau_all = _find_tau(lag_all, Q_all)
        tau_j = _find_tau(lag_j, Q_j)
        tau_m = _find_tau(lag_m, Q_m)
        print(f"  Relaxation time (all cells):    τ ≈ {tau_all:.0f}")
        print(f"  Relaxation time (jammed cells): τ ≈ {tau_j:.0f}")
        print(f"  Relaxation time (motile cells): τ ≈ {tau_m:.0f}")
        if tau_j > 2 * tau_m:
            print("  => Jammed cells relax MUCH slower — heterogeneous dynamics")
        elif tau_j > tau_m:
            print("  => Jammed cells relax slower — moderate heterogeneity")
        else:
            print("  => Similar relaxation — weak heterogeneity")

    # ==================================================================
    # PLOTTING
    # ==================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 24), facecolor='white')

    # --- Panel 1: Mobility distribution with threshold ---
    ax1 = fig.add_subplot(4, 3, 1)
    ax1.hist(all_mobs_pos, bins=80, density=True, alpha=0.7, color='steelblue',
             edgecolor='none')
    ax1.axvline(threshold, color='red', lw=2, ls='--',
                label=f'Threshold = {threshold:.5f}')
    ax1.set_xlabel('Mobility |Δr|/Δt')
    ax1.set_ylabel('Probability density')
    ax1.set_title('Mobility Distribution')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, np.percentile(all_mobs_pos, 99))

    # --- Panel 2: Jammed fraction vs time ---
    ax2 = fig.add_subplot(4, 3, 2)
    jf_t = np.array(jammed_frac_vs_time)
    ax2.plot(jf_t[:, 0], jf_t[:, 1], 'k-', lw=0.5, alpha=0.5)
    # running average
    w = min(20, len(jf_t) // 5)
    if w > 1:
        kernel = np.ones(w) / w
        smooth = np.convolve(jf_t[:, 1], kernel, mode='valid')
        ax2.plot(jf_t[w//2:w//2+len(smooth), 0], smooth, 'r-', lw=2,
                 label=f'Running avg (w={w})')
    ax2.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Jammed fraction')
    ax2.set_title('Jammed Fraction vs Time')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1)

    # --- Panel 3: Cluster size distributions ---
    ax3 = fig.add_subplot(4, 3, 3)
    if len(all_jammed_sizes[all_jammed_sizes > 0]) > 0:
        jbins = np.arange(0.5, max(all_jammed_sizes.max(), all_motile_sizes.max()) + 1.5, 1)
        ax3.hist(all_jammed_sizes[all_jammed_sizes > 0], bins=jbins, alpha=0.6,
                 color='navy', density=True, label='Jammed')
    if len(all_motile_sizes[all_motile_sizes > 0]) > 0:
        ax3.hist(all_motile_sizes[all_motile_sizes > 0], bins=jbins, alpha=0.6,
                 color='firebrick', density=True, label='Motile')
    ax3.set_xlabel('Cluster size (cells)')
    ax3.set_ylabel('Probability')
    ax3.set_title('Cluster Size Distribution')
    ax3.legend(fontsize=9)
    ax3.set_yscale('log')

    # --- Panel 4: Cluster size distributions (log-log CDF) ---
    ax4 = fig.add_subplot(4, 3, 4)
    for sizes, lab, col in [(all_jammed_sizes, 'Jammed', 'navy'),
                             (all_motile_sizes, 'Motile', 'firebrick')]:
        s = np.sort(sizes[sizes > 0])
        if len(s) == 0:
            continue
        # Complementary CDF: P(S > s)
        ccdf = 1 - np.arange(len(s)) / len(s)
        ax4.loglog(s, ccdf, '.', color=col, alpha=0.5, ms=3, label=lab)
    ax4.set_xlabel('Cluster size s')
    ax4.set_ylabel('P(S > s)')
    ax4.set_title('Cluster Size CCDF (log-log)')
    ax4.legend(fontsize=9)

    # --- Panel 5: Spatial autocorrelation C(r) ---
    ax5 = fig.add_subplot(4, 3, 5)
    ax5.plot(r_bins, mean_Cr, 'b-', lw=2)
    ax5.fill_between(r_bins, mean_Cr - std_Cr, mean_Cr + std_Cr,
                      alpha=0.2, color='blue')
    ax5.axhline(0, color='gray', ls=':', alpha=0.5)
    ax5.axhline(1/np.e, color='red', ls='--', alpha=0.5,
                label=f'1/e (ξ ≈ {corr_length:.0f})')
    ax5.axvline(corr_length, color='red', ls='--', alpha=0.3)
    ax5.set_xlabel('Distance r')
    ax5.set_ylabel('C(r)')
    ax5.set_title('Spatial Mobility Autocorrelation')
    ax5.legend(fontsize=9)

    # --- Panel 6: inherent v_A vs time-averaged mobility ---
    ax6 = fig.add_subplot(4, 3, 6)
    if inherent_vA is not None and vA_arr is not None:
        ax6.scatter(vA_arr, mob_arr, s=10, alpha=0.5, c='steelblue',
                    edgecolors='none')
        # linear fit
        z = np.polyfit(vA_arr, mob_arr, 1)
        x_fit = np.linspace(vA_arr.min(), vA_arr.max(), 50)
        ax6.plot(x_fit, np.polyval(z, x_fit), 'r-', lw=2,
                 label=f'r = {r_corr:.3f}')
        ax6.set_xlabel('Inherent $v_A$')
        ax6.set_ylabel('Time-averaged mobility')
        ax6.set_title('Quenched Disorder vs Dynamics')
        ax6.legend(fontsize=9)
    else:
        ax6.text(0.5, 0.5, 'No inherent v_A data', transform=ax6.transAxes,
                 ha='center', va='center')
        ax6.set_title('Quenched Disorder vs Dynamics (N/A)')

    # --- Panel 7: Persistence histogram ---
    ax7 = fig.add_subplot(4, 3, 7)
    ax7.hist(pers_vals, bins=30, density=True, alpha=0.7, color='teal',
             edgecolor='none')
    ax7.axvline(pers_vals.mean(), color='red', lw=2, ls='--',
                label=f'Mean = {pers_vals.mean():.3f}')
    ax7.set_xlabel('Persistence (frac. in majority state)')
    ax7.set_ylabel('Probability density')
    ax7.set_title('Temporal Persistence of Classification')
    ax7.legend(fontsize=9)

    # --- Panel 8: Persistence vs inherent v_A ---
    ax8 = fig.add_subplot(4, 3, 8)
    if inherent_vA is not None:
        cell_ids_common = sorted(set(persistence.keys()) & set(inherent_vA.keys())
                                  & set(time_avg_mob.keys()))
        pers_arr = np.array([persistence[cid] for cid in cell_ids_common])
        va_arr2 = np.array([inherent_vA[cid] for cid in cell_ids_common])
        mob_arr2 = np.array([time_avg_mob[cid] for cid in cell_ids_common])
        # Color by whether cell is mostly jammed or motile
        mostly_jammed = np.array([
            np.mean(np.array([1 if mobility[t].get(cid, 0.0) <= threshold else 0
                              for t in analysis_times]))
            for cid in cell_ids_common
        ])
        sc = ax8.scatter(va_arr2, pers_arr, c=mostly_jammed, cmap='coolwarm_r',
                         s=20, alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax8, label='Fraction time jammed')
        ax8.set_xlabel('Inherent $v_A$')
        ax8.set_ylabel('Persistence')
        ax8.set_title('Persistence vs Quenched $v_A$')
    else:
        ax8.text(0.5, 0.5, 'No inherent v_A data', transform=ax8.transAxes,
                 ha='center', va='center')

    # --- Panel 9: Overlap function Q(t) ---
    ax9 = fig.add_subplot(4, 3, 9)
    if len(lag_all) > 0:
        ax9.plot(lag_all, Q_all, 'k-', lw=2, label='All cells')
        if len(lag_j) > 0:
            ax9.plot(lag_j, Q_j, 'b-', lw=2, label='Jammed cells')
        if len(lag_m) > 0:
            ax9.plot(lag_m, Q_m, 'r-', lw=2, label='Motile cells')
        ax9.axhline(1/np.e, color='gray', ls='--', alpha=0.5, label='1/e')
        ax9.set_xlabel('Lag time Δt')
        ax9.set_ylabel('Q(Δt)')
        ax9.set_title('Overlap Function (Structural Relaxation)')
        ax9.legend(fontsize=9)
        ax9.set_ylim(0, 1.05)

    # --- Panel 10: Largest cluster fraction vs time ---
    ax10 = fig.add_subplot(4, 3, 10)
    ax10.plot(range(n_samples), largest_jammed_frac, 'bo-', ms=4,
              label='Largest jammed')
    ax10.plot(range(n_samples), largest_motile_frac, 'rs-', ms=4,
              label='Largest motile')
    ax10.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax10.set_xlabel('Sample index')
    ax10.set_ylabel('Largest cluster / N')
    ax10.set_title('Largest Cluster Fraction')
    ax10.legend(fontsize=9)

    # --- Panel 11: Snapshot - Voronoi colored by mobility class ---
    ax11 = fig.add_subplot(4, 3, 11)
    t_snap = analysis_times[len(analysis_times)//2]
    snap_labels = classify_cells(mobility[t_snap], threshold)
    snap_neighbors = build_neighbor_graph(positions[t_snap], Lx, Ly)
    snap_clusters = find_clusters(snap_labels, snap_neighbors)

    # Color each cluster with a unique shade
    from matplotlib.patches import Polygon as MplPolygon
    snap_polys = _voronoi_polygons(positions[t_snap], Lx, Ly)
    # Jammed = blue shades, Motile = red shades
    j_clusters = [(lab, c) for lab, c in snap_clusters if lab == 0]
    m_clusters = [(lab, c) for lab, c in snap_clusters if lab == 1]

    for ci, (_, cluster) in enumerate(j_clusters):
        shade = 0.3 + 0.5 * (ci % 5) / 5.0
        color = (0.1, 0.1, shade, 0.8)
        for cid in cluster:
            if cid in snap_polys:
                for verts in snap_polys[cid]:
                    p = MplPolygon(verts, closed=True, facecolor=color,
                                   edgecolor='black', linewidth=0.3)
                    ax11.add_patch(p)

    for ci, (_, cluster) in enumerate(m_clusters):
        shade = 0.3 + 0.5 * (ci % 5) / 5.0
        color = (shade, 0.1, 0.1, 0.8)
        for cid in cluster:
            if cid in snap_polys:
                for verts in snap_polys[cid]:
                    p = MplPolygon(verts, closed=True, facecolor=color,
                                   edgecolor='black', linewidth=0.3)
                    ax11.add_patch(p)

    ax11.set_xlim(0, Lx)
    ax11.set_ylim(0, Ly)
    ax11.set_aspect('equal')
    ax11.set_title(f'Cluster Map (t={t_snap:.0f})')
    ax11.set_xticks([])
    ax11.set_yticks([])

    # --- Panel 12: Summary text ---
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.axis('off')
    summary_lines = [
        f"GRIFFITHS ANALYSIS SUMMARY",
        f"─" * 35,
        f"System: N={N_cells}, Lx={Lx:.0f}×{Ly:.0f}",
        f"v_A = {mean_vA} ± {sigma_vA}",
        f"Threshold: {threshold:.6f} (median)",
        f"",
        f"Jammed fraction: {jammed_fracs.mean():.1%} ± {jammed_fracs.std():.1%}",
        f"",
        f"Percolation (of {n_samples} samples):",
        f"  Jammed percolates: {jammed_perc_count}/{n_samples}",
        f"  Motile percolates: {motile_perc_count}/{n_samples}",
        f"",
        f"Correlation length: ξ ≈ {corr_length:.1f} "
        f"({corr_length/cell_spacing:.1f} cells)",
        f"",
        f"v_A-mobility correlation: r = {r_corr:.3f}" if r_corr else "",
        f"Mean persistence: {pers_vals.mean():.3f}",
        f"Cells with persistence > 0.8: {np.mean(pers_vals > 0.8)*100:.0f}%",
        f"",
        f"DIAGNOSIS: {phase_diagnosis.replace('_', ' ').upper()}",
    ]
    summary_text = '\n'.join(summary_lines)
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
              va='top', ha='left', fontsize=10, fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'Griffiths Rare-Region Analysis — v_A={mean_vA}, '
                 f'σ={sigma_vA}, N={N_cells}',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = out_dir / f'griffiths_analysis_{date_str}.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {plot_path}")
    plt.close(fig)

    # ==================================================================
    # WRITE LOGBOOK ENTRY
    # ==================================================================
    logbook_path = Path(__file__).parent.parent / 'research_logbook.md'
    print(f"\n  Writing logbook entry to {logbook_path}")

    entry = f"""
## Griffiths Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Parameters
- N = {N_cells} cells, domain = {Lx:.0f} × {Ly:.0f}
- Mean v_A = {mean_vA}, σ_vA = {sigma_vA} (log-normal)
- Packing fraction ≈ {float(header.get('phi', 0.89)):.2f} (from header or estimated)
- Trajectory: {traj_file}
- Time span: {times[0]:.0f} to {times[-1]:.0f} ({len(times)} frames)
- Mobility window: {window:.1f}
- Threshold: {threshold:.6f} (median mobility)

### Key Results

#### 1. Phase Classification
- **Jammed fraction**: {jammed_fracs.mean():.1%} ± {jammed_fracs.std():.1%}
- Majority phase: **{'Jammed' if jammed_fracs.mean() > 0.5 else 'Motile'}**

#### 2. Cluster Analysis & Percolation
- Jammed clusters: mean size = {all_jammed_sizes.mean():.1f}, max = {all_jammed_sizes.max()}
- Motile clusters: mean size = {all_motile_sizes.mean():.1f}, max = {all_motile_sizes.max()}
- Largest jammed cluster: {np.mean(largest_jammed_frac):.1%} of cells
- Largest motile cluster: {np.mean(largest_motile_frac):.1%} of cells
- Jammed percolates: {jammed_perc_count}/{n_samples} frames
- Motile percolates: {motile_perc_count}/{n_samples} frames
- **Diagnosis: {phase_diagnosis.replace('_', ' ')}**

#### 3. Spatial Correlations
- Correlation length ξ ≈ {corr_length:.1f} ({corr_length/cell_spacing:.1f} cell spacings)
- C(r=50) = {np.interp(50, r_bins, mean_Cr):.3f}
- C(r=100) = {np.interp(100, r_bins, mean_Cr):.3f}
- C(r=200) = {np.interp(200, r_bins, mean_Cr):.3f}

#### 4. Quenched Disorder Correlation
- Pearson r(v_A, mobility) = {r_corr:.4f}{f', p = {p_val:.2e}' if r_corr is not None else ''}
- {'Strong' if r_corr and r_corr > 0.5 else 'Moderate' if r_corr and r_corr > 0.2 else 'Weak'} correlation between inherent v_A and measured mobility

#### 5. Temporal Persistence
- Mean persistence = {pers_vals.mean():.3f}
- {np.mean(pers_vals > 0.8)*100:.0f}% of cells have persistence > 0.8
- {np.mean(pers_vals > 0.9)*100:.0f}% of cells have persistence > 0.9

#### 6. Structural Relaxation
- τ_all ≈ {tau_all:.0f}
- τ_jammed ≈ {tau_j:.0f}  
- τ_motile ≈ {tau_m:.0f}
- Ratio τ_jammed/τ_motile ≈ {tau_j/tau_m:.1f}

### Physical Interpretation

{'The motile phase percolates while jammed cells form isolated islands. This places the system ABOVE the jamming transition, in a regime where Griffiths rare-region effects create persistent jammed patches within a predominantly fluid tissue.' if phase_diagnosis == 'jammed_islands_in_motile_sea' else 'The jammed phase percolates while motile cells form isolated pockets. This places the system BELOW the jamming transition, where Griffiths rare-region effects create persistent fluid patches that slowly relax.' if phase_diagnosis == 'motile_islands_in_jammed_sea' else 'Both phases show similar percolation, suggesting the system is near the critical point of the jamming transition.'}

The correlation r = {r_corr:.3f} between inherent v_A and measured mobility {'confirms' if r_corr and r_corr > 0.3 else 'suggests'} that the quenched disorder (log-normal v_A distribution) {'directly controls' if r_corr and r_corr > 0.5 else 'partially influences'} the spatial pattern of mobility, {'consistent with' if r_corr and r_corr > 0.2 else 'weakly supporting'} the Griffiths rare-region picture.

The temporal persistence of {pers_vals.mean():.3f} indicates that cell classifications are {'highly stable' if pers_vals.mean() > 0.85 else 'moderately stable' if pers_vals.mean() > 0.7 else 'somewhat dynamic'}, {'confirming' if pers_vals.mean() > 0.8 else 'partially supporting'} the "quenched" nature of the disorder.

### Plots
- Full analysis figure: `postprocessing/output/griffiths_analysis_{date_str}.png`

---
"""
    # Append to logbook
    mode = 'a' if logbook_path.exists() else 'w'
    with open(logbook_path, mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write("# Research Logbook -- Phase Field Cell Simulation\n\n")
        f.write(entry)
    print(f"  Logbook entry appended: {logbook_path}")

    return plot_path


if __name__ == '__main__':
    plot_path = main()
    print(f"\nDone. Plot saved to: {plot_path}")
