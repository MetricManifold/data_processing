#!/usr/bin/env python3
"""
Batch Griffiths Rare-Region Analysis — Comparative Study

Processes all subsampled trajectory files from the cluster Griffiths study
and produces comparative plots across parameter space (v_A, σ_vA).

The key question: Does quenched disorder (σ_vA > 0) create persistent
spatial heterogeneity (Griffiths rare regions) compared to the homogeneous
control (σ_vA = 0)?

Parameter space (6 combos × 3 replicates = 18 runs):
  Fixed σ=0.006, varying v_A: {0.006, 0.008, 0.010}
  Fixed v_A=0.008, varying σ: {0.000, 0.003, 0.006, 0.008}

The σ=0 case serves as the CONTROL — with all cells identical (v_A=0.008),
any dynamic heterogeneity is purely stochastic, NOT quenched disorder.

Usage:
  python analyze_griffiths_batch.py path/to/griffiths_subsampled/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
from collections import defaultdict
import argparse
import sys
import re
from datetime import datetime
from scipy.spatial import Voronoi
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from visualize_fluid import load_trajectory, compute_displacement_mobility, _voronoi_polygons
from analyze_griffiths import (
    build_neighbor_graph, classify_cells, find_clusters,
    check_percolation, spatial_autocorrelation, compute_persistence,
    overlap_function
)


def parse_filename(fname):
    """Extract v_A and sigma from filename like vA_0.008_sigma_0.006_run_1.txt"""
    m = re.match(r'vA_([\d.]+)_sigma_([\d.]+)_run_(\d+)\.txt', fname)
    if m:
        return float(m.group(1)), float(m.group(2)), int(m.group(3))
    return None, None, None


def analyze_single_run(traj_file, threshold=None, verbose=False):
    """Run core Griffiths analysis on a single trajectory file.
    
    Args:
        traj_file: Path to trajectory file
        threshold: mobility threshold for jammed/motile (None=auto)
        verbose: print progress
    
    Returns:
        dict of analysis results
    """
    if verbose:
        print(f"  Processing {traj_file.name}...")
    
    times, positions, velocities, header, inherent_vA = load_trajectory(
        str(traj_file), subsample=1)
    
    N_cells = len(positions[times[0]])
    Lx = float(header.get('Lx', 1600))
    Ly = float(header.get('Ly', 1600))
    cell_spacing = np.sqrt(Lx * Ly / N_cells)
    
    # Compute mobility
    window = (times[-1] - times[0]) * 0.05
    mobility = compute_displacement_mobility(times, positions, header, window)
    
    # Collect mobility values (skip first 10%)
    skip = max(1, len(times) // 10)
    all_mobs = []
    for t in times[skip:]:
        all_mobs.extend(mobility[t].values())
    all_mobs = np.array(all_mobs)
    
    # Use provided threshold or median
    if threshold is None:
        threshold = np.median(all_mobs)
    
    # Analysis window: middle 50%
    mid_start = len(times) // 4
    mid_end = 3 * len(times) // 4
    analysis_times = times[mid_start:mid_end]
    
    results = {
        'N_cells': N_cells,
        'Lx': Lx, 'Ly': Ly,
        'cell_spacing': cell_spacing,
        't_start': times[0], 't_end': times[-1],
        'n_frames': len(times),
        'threshold': threshold,
        'mean_mobility': all_mobs.mean(),
        'std_mobility': all_mobs.std(),
        'median_mobility': np.median(all_mobs),
    }
    
    # === 1. Jammed fraction ===
    jammed_fracs = []
    for t in analysis_times:
        labels = classify_cells(mobility[t], threshold)
        n_jammed = sum(1 for v in labels.values() if v == 0)
        jammed_fracs.append(n_jammed / N_cells)
    results['jammed_frac_mean'] = np.mean(jammed_fracs)
    results['jammed_frac_std'] = np.std(jammed_fracs)
    
    # === 2. Cluster analysis & percolation ===
    sample_indices = np.linspace(0, len(analysis_times)-1,
                                  min(20, len(analysis_times)), dtype=int)
    all_jammed_sizes = []
    all_motile_sizes = []
    jammed_perc = 0
    motile_perc = 0
    n_samples = 0
    largest_j = []
    largest_m = []
    n_jammed_clusters_list = []
    n_motile_clusters_list = []
    
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
        
        largest_j.append(max(j_sizes) / N_cells if j_sizes else 0)
        largest_m.append(max(m_sizes) / N_cells if m_sizes else 0)
        n_jammed_clusters_list.append(len(jammed_clusters))
        n_motile_clusters_list.append(len(motile_clusters))
        
        for jc in jammed_clusters:
            px, py = check_percolation(jc, positions[t], Lx, Ly)
            if px or py:
                jammed_perc += 1
                break
        for mc in motile_clusters:
            px, py = check_percolation(mc, positions[t], Lx, Ly)
            if px or py:
                motile_perc += 1
                break
        n_samples += 1
    
    all_jammed_sizes = np.array(all_jammed_sizes)
    all_motile_sizes = np.array(all_motile_sizes)
    
    results['largest_jammed_frac'] = np.mean(largest_j)
    results['largest_motile_frac'] = np.mean(largest_m)
    results['jammed_perc_frac'] = jammed_perc / n_samples
    results['motile_perc_frac'] = motile_perc / n_samples
    results['mean_jammed_cluster_size'] = all_jammed_sizes.mean()
    results['mean_motile_cluster_size'] = all_motile_sizes.mean()
    results['n_jammed_clusters'] = np.mean(n_jammed_clusters_list)
    results['n_motile_clusters'] = np.mean(n_motile_clusters_list)
    
    # === 3. Spatial autocorrelation ===
    n_cr = min(8, len(analysis_times))
    cr_idx = np.linspace(0, len(analysis_times)-1, n_cr, dtype=int)
    all_Cr = []
    all_counts = []
    r_bins = None
    for ci in cr_idx:
        t = analysis_times[ci]
        rb, Cr, cts = spatial_autocorrelation(positions[t], mobility[t], Lx, Ly, n_bins=40)
        if r_bins is None:
            r_bins = rb
        all_Cr.append(Cr)
        all_counts.append(cts)
    mean_Cr = np.mean(all_Cr, axis=0)
    mean_counts = np.mean(all_counts, axis=0)
    
    # Correlation length — skip bins with < 10 pairs
    # (sub-cell-spacing bins have no pairs, producing C(r)=0 artifact)
    MIN_PAIRS = 10
    corr_length = r_bins[-1]
    for i in range(len(mean_Cr)):
        if mean_counts[i] < MIN_PAIRS:
            continue  # Skip empty bins
        if mean_Cr[i] < 1/np.e:
            corr_length = r_bins[i]
            break
    
    results['corr_length'] = corr_length
    results['corr_length_cells'] = corr_length / cell_spacing
    results['r_bins'] = r_bins
    results['C_r'] = mean_Cr
    
    # === 4. Temporal persistence ===
    persistence, jammed_frac_time = compute_persistence(times, mobility, threshold)
    pers_vals = np.array(list(persistence.values()))
    results['mean_persistence'] = pers_vals.mean()
    results['persistence_gt08'] = np.mean(pers_vals > 0.8)
    results['persistence_gt09'] = np.mean(pers_vals > 0.9)
    results['persistence_distribution'] = pers_vals
    
    # === 5. Overlap function Q(t) ===
    t_ref_idx = len(times) // 4
    t_ref = times[t_ref_idx]
    ref_labels = classify_cells(mobility[t_ref], threshold)
    qt_times = times[t_ref_idx:]
    cage_radius = cell_spacing * 0.3
    
    lag_all, Q_all = overlap_function(qt_times, positions, header, a=cage_radius)
    lag_j, Q_j = overlap_function(qt_times, positions, header, a=cage_radius,
                                   cell_class=ref_labels, class_label=0)
    lag_m, Q_m = overlap_function(qt_times, positions, header, a=cage_radius,
                                   cell_class=ref_labels, class_label=1)
    
    def _find_tau(lag, Q):
        for i in range(len(Q)):
            if Q[i] < 1/np.e:
                return lag[i]
        return lag[-1] if len(lag) > 0 else np.nan
    
    results['tau_all'] = _find_tau(lag_all, Q_all) if len(lag_all) > 0 else np.nan
    results['tau_jammed'] = _find_tau(lag_j, Q_j) if len(lag_j) > 0 else np.nan
    results['tau_motile'] = _find_tau(lag_m, Q_m) if len(lag_m) > 0 else np.nan
    results['Q_lag'] = lag_all
    results['Q_all'] = Q_all
    results['Q_jammed_lag'] = lag_j
    results['Q_jammed'] = Q_j
    results['Q_motile_lag'] = lag_m
    results['Q_motile'] = Q_m
    
    # === 6. Mobility distribution ===
    results['mobility_values'] = all_mobs
    
    # === 7. Non-Gaussian parameter ===
    # α₂ = <Δr⁴>/(2<Δr²>²) - 1  (2D)
    # High α₂ indicates dynamic heterogeneity
    displacements = []
    dt_target = (times[-1] - times[0]) * 0.1  # 10% of full time
    for ti in range(len(times) // 2):
        t0 = times[ti]
        # find time closest to t0 + dt_target
        tj = np.searchsorted(times, t0 + dt_target)
        if tj >= len(times):
            break
        t1 = times[tj]
        for cid in positions[t0]:
            if cid not in positions[t1]:
                continue
            x0, y0 = positions[t0][cid]
            x1, y1 = positions[t1][cid]
            dx = x1 - x0
            dy = y1 - y0
            if dx > Lx/2: dx -= Lx
            elif dx < -Lx/2: dx += Lx
            if dy > Ly/2: dy -= Ly
            elif dy < -Ly/2: dy += Ly
            displacements.append(dx**2 + dy**2)
    
    displacements = np.array(displacements)
    if len(displacements) > 0:
        r2 = displacements
        r4 = displacements**2
        alpha2 = r4.mean() / (2 * r2.mean()**2) - 1  # 2D
        results['alpha2'] = alpha2
        results['MSD'] = r2.mean()
    else:
        results['alpha2'] = np.nan
        results['MSD'] = np.nan
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch Griffiths analysis across parameter sets')
    parser.add_argument('data_dir',
                        help='Directory containing subsampled trajectory files')
    parser.add_argument('--threshold-from-control', action='store_true',
                        default=True,
                        help='Use σ=0 mobility distribution to set threshold')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Discover all files
    files = sorted(data_dir.glob('vA_*_sigma_*_run_*.txt'))
    if not files:
        print(f"ERROR: No trajectory files found in {data_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("BATCH GRIFFITHS RARE-REGION ANALYSIS")
    print("=" * 70)
    print(f"Found {len(files)} trajectory files in {data_dir}")
    
    # Group by parameter set
    param_groups = defaultdict(list)
    for f in files:
        vA, sigma, run = parse_filename(f.name)
        if vA is not None:
            param_groups[(vA, sigma)].append((run, f))
    
    print(f"\nParameter sets ({len(param_groups)}):")
    for (vA, sigma), runs in sorted(param_groups.items()):
        print(f"  v_A={vA:.3f}, σ={sigma:.3f}: {len(runs)} replicates")
    
    # ================================================================
    # STEP 1: Establish threshold from σ=0 control (if available)
    # ================================================================
    threshold = None
    control_key = None
    for key in param_groups:
        if key[1] == 0.0:  # σ=0
            control_key = key
            break
    
    if control_key is not None and args.threshold_from_control:
        print(f"\n--- Establishing threshold from CONTROL (σ=0, v_A={control_key[0]}) ---")
        control_mobs = []
        for run, fpath in param_groups[control_key]:
            times, positions, _, header, _ = load_trajectory(str(fpath), subsample=1)
            window = (times[-1] - times[0]) * 0.05
            mob = compute_displacement_mobility(times, positions, header, window)
            skip = max(1, len(times) // 10)
            for t in times[skip:]:
                control_mobs.extend(mob[t].values())
        control_mobs = np.array(control_mobs)
        
        # Strategy: The control (σ=0) has ALL cells with identical v_A.
        # Many cells are truly caged (mobility=0) at this packing fraction.
        # The control MEAN represents the typical mobility of a cell with
        # v_A=0.008 in this system.  Using a fraction of the mean as
        # threshold separates genuinely slow cells from the normally
        # fluctuating population.
        #
        # We use 0.5 × mean: cells below this are "slow relative to average"
        # which in the disordered case picks out cells whose inherent v_A
        # makes them systematically slower.
        control_mean = control_mobs.mean()
        control_median = np.median(control_mobs)
        # If median is zero (many caged cells), use a percentile of nonzero values
        nonzero_mobs = control_mobs[control_mobs > 0]
        if len(nonzero_mobs) > 0:
            q25_nz = np.percentile(nonzero_mobs, 25)
        else:
            q25_nz = 0.0
        
        # Use mean of all control mobilities as threshold
        threshold = control_mean
        
        q25, q75 = np.percentile(control_mobs, [25, 75])
        frac_zero = np.mean(control_mobs == 0)
        print(f"  Control mobility: mean={control_mean:.6f}, "
              f"median={control_median:.6f}, "
              f"std={control_mobs.std():.6f}")
        print(f"  Fraction exactly zero: {frac_zero:.1%}")
        print(f"  Q25 of nonzero: {q25_nz:.6f}")
        print(f"  IQR: [{q25:.6f}, {q75:.6f}]")
        print(f"  Using threshold = {threshold:.6f} (control MEAN)")
        print(f"  Rationale: cells below the control mean are systematically")
        print(f"  slower than average — in disordered cases, this identifies")
        print(f"  cells whose low inherent v_A keeps them jammed.")
    else:
        print("\n  No σ=0 control available, will use per-run median threshold")
    
    # ================================================================
    # STEP 2: Process all parameter sets
    # ================================================================
    all_results = {}  # (vA, sigma) -> list of result dicts (one per replicate)
    
    for (vA, sigma), runs in sorted(param_groups.items()):
        print(f"\n{'='*50}")
        print(f"Processing v_A={vA:.3f}, σ={sigma:.3f} ({len(runs)} replicates)")
        print(f"{'='*50}")
        
        run_results = []
        for run, fpath in sorted(runs):
            res = analyze_single_run(fpath, threshold=threshold, verbose=True)
            res['vA'] = vA
            res['sigma'] = sigma
            res['run'] = run
            run_results.append(res)
        
        all_results[(vA, sigma)] = run_results
    
    # ================================================================
    # STEP 3: Aggregate statistics per parameter set
    # ================================================================
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)
    
    summary = {}
    for key in sorted(all_results.keys()):
        vA, sigma = key
        runs = all_results[key]
        n = len(runs)
        
        s = {}
        for metric in ['jammed_frac_mean', 'largest_jammed_frac',
                        'largest_motile_frac', 'jammed_perc_frac',
                        'motile_perc_frac', 'corr_length',
                        'corr_length_cells', 'mean_persistence',
                        'persistence_gt08', 'persistence_gt09',
                        'tau_all', 'tau_jammed', 'tau_motile',
                        'alpha2', 'MSD', 'mean_mobility',
                        'n_jammed_clusters', 'n_motile_clusters',
                        'mean_jammed_cluster_size', 'mean_motile_cluster_size']:
            vals = [r[metric] for r in runs if not np.isnan(r.get(metric, np.nan))]
            if vals:
                s[metric] = (np.mean(vals), np.std(vals) / np.sqrt(len(vals)))
            else:
                s[metric] = (np.nan, np.nan)
        
        summary[key] = s
        
        print(f"\nv_A={vA:.3f}, σ={sigma:.3f} ({n} replicates):")
        print(f"  Jammed fraction:     {s['jammed_frac_mean'][0]:.3f} ± {s['jammed_frac_mean'][1]:.3f}")
        print(f"  Largest jammed:      {s['largest_jammed_frac'][0]:.3f} ± {s['largest_jammed_frac'][1]:.3f}")
        print(f"  Largest motile:      {s['largest_motile_frac'][0]:.3f} ± {s['largest_motile_frac'][1]:.3f}")
        print(f"  Jammed percolation:  {s['jammed_perc_frac'][0]:.2f}")
        print(f"  Motile percolation:  {s['motile_perc_frac'][0]:.2f}")
        print(f"  Corr. length ξ:      {s['corr_length'][0]:.1f} ({s['corr_length_cells'][0]:.1f} cells)")
        print(f"  Persistence:         {s['mean_persistence'][0]:.3f}")
        print(f"  Persist > 0.8:       {s['persistence_gt08'][0]*100:.1f}%")
        print(f"  τ_all:               {s['tau_all'][0]:.0f}")
        print(f"  τ_jammed:            {s['tau_jammed'][0]:.0f}")
        print(f"  τ_motile:            {s['tau_motile'][0]:.0f}")
        tau_ratio = s['tau_jammed'][0] / s['tau_motile'][0] if s['tau_motile'][0] > 0 else np.nan
        print(f"  τ_j/τ_m ratio:       {tau_ratio:.2f}")
        print(f"  α₂ (non-Gauss.):     {s['alpha2'][0]:.3f}")
        print(f"  MSD:                 {s['MSD'][0]:.4f}")
    
    # ================================================================
    # STEP 4: Generate comparative plots
    # ================================================================
    print("\n" + "=" * 70)
    print("GENERATING COMPARATIVE PLOTS")
    print("=" * 70)
    
    # Sort parameter sets into two sweeps:
    # (A) Fixed σ=0.006, varying v_A
    # (B) Fixed v_A=0.008, varying σ
    sigma_sweep = sorted([(k, v) for k, v in summary.items() if k[0] == 0.008],
                          key=lambda x: x[0][1])
    vA_sweep = sorted([(k, v) for k, v in summary.items() if k[1] == 0.006],
                       key=lambda x: x[0][0])
    
    # Colors
    sigma_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sigma_sweep)))
    vA_colors = plt.cm.magma(np.linspace(0.2, 0.9, len(vA_sweep)))
    
    # ================================================================
    # FIGURE 1: Disorder strength comparison (fixed v_A=0.008)
    # ================================================================
    fig1, axes1 = plt.subplots(3, 4, figsize=(24, 18), facecolor='white')
    fig1.suptitle('Griffiths Analysis: Effect of Disorder Strength σ\n'
                  f'(Fixed v_A=0.008, N=288, threshold from σ=0 control)',
                  fontsize=16, fontweight='bold')
    
    # Panel 1a: Mobility distributions
    ax = axes1[0, 0]
    for i, ((vA, sigma), s) in enumerate(sigma_sweep):
        runs = all_results[(vA, sigma)]
        all_mob_vals = np.concatenate([r['mobility_values'] for r in runs])
        ax.hist(all_mob_vals, bins=80, density=True, alpha=0.4,
                color=sigma_colors[i], label=f'σ={sigma:.3f}',
                histtype='stepfilled')
    if threshold is not None:
        ax.axvline(threshold, color='red', lw=2, ls='--', label='Threshold')
    ax.set_xlabel('Mobility |Δr|/Δt')
    ax.set_ylabel('Probability density')
    ax.set_title('Mobility Distribution by σ')
    ax.legend(fontsize=8)
    ax.set_xlim(0, np.percentile(all_mob_vals, 99.5))
    
    # Panel 1b: Jammed fraction vs σ
    ax = axes1[0, 1]
    sigmas = [k[1] for k, _ in sigma_sweep]
    jf_means = [s['jammed_frac_mean'][0] for _, s in sigma_sweep]
    jf_errs = [s['jammed_frac_mean'][1] for _, s in sigma_sweep]
    ax.errorbar(sigmas, jf_means, yerr=jf_errs, fmt='o-', color='navy',
                capsize=4, lw=2, ms=8)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Disorder strength σ')
    ax.set_ylabel('Jammed fraction')
    ax.set_title('Jammed Fraction vs σ')
    ax.set_ylim(0, 1)
    
    # Panel 1c: Percolation fractions vs σ
    ax = axes1[0, 2]
    jp = [s['jammed_perc_frac'][0] for _, s in sigma_sweep]
    mp = [s['motile_perc_frac'][0] for _, s in sigma_sweep]
    ax.plot(sigmas, jp, 'bo-', ms=8, lw=2, label='Jammed')
    ax.plot(sigmas, mp, 'rs-', ms=8, lw=2, label='Motile')
    ax.set_xlabel('σ')
    ax.set_ylabel('Percolation fraction')
    ax.set_title('Percolation vs σ')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    # Panel 1d: Correlation length vs σ 
    ax = axes1[0, 3]
    xi = [s['corr_length_cells'][0] for _, s in sigma_sweep]
    xi_err = [s['corr_length_cells'][1] for _, s in sigma_sweep]
    ax.errorbar(sigmas, xi, yerr=xi_err, fmt='o-', color='darkgreen',
                capsize=4, lw=2, ms=8)
    ax.set_xlabel('σ')
    ax.set_ylabel('ξ / cell spacing')
    ax.set_title('Correlation Length vs σ')
    
    # Panel 2a: Persistence vs σ
    ax = axes1[1, 0]
    p_mean = [s['mean_persistence'][0] for _, s in sigma_sweep]
    p_err = [s['mean_persistence'][1] for _, s in sigma_sweep]
    ax.errorbar(sigmas, p_mean, yerr=p_err, fmt='o-', color='teal',
                capsize=4, lw=2, ms=8)
    ax.set_xlabel('σ')
    ax.set_ylabel('Mean persistence')
    ax.set_title('Temporal Persistence vs σ')
    
    # Panel 2b: Persistence distributions
    ax = axes1[1, 1]
    for i, ((vA, sigma), _) in enumerate(sigma_sweep):
        runs = all_results[(vA, sigma)]
        all_pers = np.concatenate([r['persistence_distribution'] for r in runs])
        ax.hist(all_pers, bins=30, density=True, alpha=0.4,
                color=sigma_colors[i], label=f'σ={sigma:.3f}',
                histtype='stepfilled')
    ax.set_xlabel('Persistence')
    ax.set_ylabel('Probability density')
    ax.set_title('Persistence Distribution by σ')
    ax.legend(fontsize=8)
    
    # Panel 2c: Relaxation times τ vs σ
    ax = axes1[1, 2]
    tau_j = [s['tau_jammed'][0] for _, s in sigma_sweep]
    tau_m = [s['tau_motile'][0] for _, s in sigma_sweep]
    tau_a = [s['tau_all'][0] for _, s in sigma_sweep]
    ax.plot(sigmas, tau_j, 'bo-', ms=8, lw=2, label='τ_jammed')
    ax.plot(sigmas, tau_m, 'rs-', ms=8, lw=2, label='τ_motile')
    ax.plot(sigmas, tau_a, 'k^-', ms=8, lw=2, label='τ_all')
    ax.set_xlabel('σ')
    ax.set_ylabel('Relaxation time τ')
    ax.set_title('Structural Relaxation vs σ')
    ax.legend(fontsize=9)
    
    # Panel 2d: τ_jammed / τ_motile ratio vs σ
    ax = axes1[1, 3]
    tau_ratio = [s['tau_jammed'][0] / s['tau_motile'][0] if s['tau_motile'][0] > 0 else np.nan
                 for _, s in sigma_sweep]
    ax.plot(sigmas, tau_ratio, 'ko-', ms=8, lw=2)
    ax.axhline(1, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('σ')
    ax.set_ylabel('τ_jammed / τ_motile')
    ax.set_title('Relaxation Time Ratio vs σ')
    
    # Panel 3a: Non-Gaussian parameter α₂ vs σ
    ax = axes1[2, 0]
    a2 = [s['alpha2'][0] for _, s in sigma_sweep]
    a2_err = [s['alpha2'][1] for _, s in sigma_sweep]
    ax.errorbar(sigmas, a2, yerr=a2_err, fmt='o-', color='purple',
                capsize=4, lw=2, ms=8)
    ax.set_xlabel('σ')
    ax.set_ylabel('α₂')
    ax.set_title('Non-Gaussian Parameter vs σ')
    
    # Panel 3b: MSD vs σ
    ax = axes1[2, 1]
    msd = [s['MSD'][0] for _, s in sigma_sweep]
    msd_err = [s['MSD'][1] for _, s in sigma_sweep]
    ax.errorbar(sigmas, msd, yerr=msd_err, fmt='o-', color='brown',
                capsize=4, lw=2, ms=8)
    ax.set_xlabel('σ')
    ax.set_ylabel('MSD')
    ax.set_title('Mean Squared Displacement vs σ')
    
    # Panel 3c: Spatial autocorrelation C(r) for each σ
    ax = axes1[2, 2]
    for i, ((vA, sigma), _) in enumerate(sigma_sweep):
        runs = all_results[(vA, sigma)]
        # Average C(r) over replicates
        all_cr = [r['C_r'] for r in runs]
        mean_cr = np.mean(all_cr, axis=0)
        r_bins_plot = runs[0]['r_bins']
        ax.plot(r_bins_plot, mean_cr, color=sigma_colors[i], lw=2,
                label=f'σ={sigma:.3f}')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axhline(1/np.e, color='red', ls='--', alpha=0.3)
    ax.set_xlabel('Distance r')
    ax.set_ylabel('C(r)')
    ax.set_title('Spatial Autocorrelation by σ')
    ax.legend(fontsize=8)
    
    # Panel 3d: Cluster sizes vs σ
    ax = axes1[2, 3]
    nc_j = [s['n_jammed_clusters'][0] for _, s in sigma_sweep]
    nc_m = [s['n_motile_clusters'][0] for _, s in sigma_sweep]
    ms_j = [s['mean_jammed_cluster_size'][0] for _, s in sigma_sweep]
    ms_m = [s['mean_motile_cluster_size'][0] for _, s in sigma_sweep]
    x = np.arange(len(sigmas))
    w = 0.35
    ax.bar(x - w/2, ms_j, w, label='Jammed', color='navy', alpha=0.7)
    ax.bar(x + w/2, ms_m, w, label='Motile', color='firebrick', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s:.3f}' for s in sigmas])
    ax.set_xlabel('σ')
    ax.set_ylabel('Mean cluster size')
    ax.set_title('Mean Cluster Size vs σ')
    ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = out_dir / f'griffiths_sigma_sweep_{date_str}.png'
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path1}")
    plt.close(fig1)
    
    # ================================================================
    # FIGURE 2: v_A sweep (fixed σ=0.006)
    # ================================================================
    if len(vA_sweep) > 1:
        fig2, axes2 = plt.subplots(3, 4, figsize=(24, 18), facecolor='white')
        fig2.suptitle('Griffiths Analysis: Effect of Mean Motility v_A\n'
                      f'(Fixed σ=0.006, N=288)',
                      fontsize=16, fontweight='bold')
        
        vAs = [k[0] for k, _ in vA_sweep]
        
        # Mobility distributions
        ax = axes2[0, 0]
        for i, ((vA, sigma), s) in enumerate(vA_sweep):
            runs = all_results[(vA, sigma)]
            mob_vals = np.concatenate([r['mobility_values'] for r in runs])
            ax.hist(mob_vals, bins=80, density=True, alpha=0.4,
                    color=vA_colors[i], label=f'v_A={vA:.3f}',
                    histtype='stepfilled')
        if threshold is not None:
            ax.axvline(threshold, color='red', lw=2, ls='--', label='Threshold')
        ax.set_xlabel('Mobility')
        ax.set_ylabel('PDF')
        ax.set_title('Mobility by v_A')
        ax.legend(fontsize=8)
        
        # Jammed fraction vs v_A
        ax = axes2[0, 1]
        jf = [summary[k]['jammed_frac_mean'][0] for k, _ in vA_sweep]
        jf_e = [summary[k]['jammed_frac_mean'][1] for k, _ in vA_sweep]
        ax.errorbar(vAs, jf, yerr=jf_e, fmt='o-', color='navy', capsize=4, lw=2, ms=8)
        ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('v_A')
        ax.set_ylabel('Jammed fraction')
        ax.set_title('Jammed Fraction vs v_A')
        ax.set_ylim(0, 1)
        
        # Percolation
        ax = axes2[0, 2]
        jp = [summary[k]['jammed_perc_frac'][0] for k, _ in vA_sweep]
        mp = [summary[k]['motile_perc_frac'][0] for k, _ in vA_sweep]
        ax.plot(vAs, jp, 'bo-', ms=8, lw=2, label='Jammed')
        ax.plot(vAs, mp, 'rs-', ms=8, lw=2, label='Motile')
        ax.set_xlabel('v_A')
        ax.set_ylabel('Percolation fraction')
        ax.set_title('Percolation vs v_A')
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        
        # Correlation length
        ax = axes2[0, 3]
        xi = [summary[k]['corr_length_cells'][0] for k, _ in vA_sweep]
        xi_e = [summary[k]['corr_length_cells'][1] for k, _ in vA_sweep]
        ax.errorbar(vAs, xi, yerr=xi_e, fmt='o-', color='darkgreen', capsize=4, lw=2, ms=8)
        ax.set_xlabel('v_A')
        ax.set_ylabel('ξ / cell spacing')
        ax.set_title('Correlation Length vs v_A')
        
        # Persistence
        ax = axes2[1, 0]
        pm = [summary[k]['mean_persistence'][0] for k, _ in vA_sweep]
        pe = [summary[k]['mean_persistence'][1] for k, _ in vA_sweep]
        ax.errorbar(vAs, pm, yerr=pe, fmt='o-', color='teal', capsize=4, lw=2, ms=8)
        ax.set_xlabel('v_A')
        ax.set_ylabel('Mean persistence')
        ax.set_title('Persistence vs v_A')
        
        # Persistence distribution
        ax = axes2[1, 1]
        for i, ((vA, sigma), _) in enumerate(vA_sweep):
            runs = all_results[(vA, sigma)]
            all_pers = np.concatenate([r['persistence_distribution'] for r in runs])
            ax.hist(all_pers, bins=30, density=True, alpha=0.4,
                    color=vA_colors[i], label=f'v_A={vA:.3f}',
                    histtype='stepfilled')
        ax.set_xlabel('Persistence')
        ax.set_ylabel('PDF')
        ax.set_title('Persistence Distribution by v_A')
        ax.legend(fontsize=8)
        
        # Relaxation times
        ax = axes2[1, 2]
        tj = [summary[k]['tau_jammed'][0] for k, _ in vA_sweep]
        tm = [summary[k]['tau_motile'][0] for k, _ in vA_sweep]
        ta = [summary[k]['tau_all'][0] for k, _ in vA_sweep]
        ax.plot(vAs, tj, 'bo-', ms=8, lw=2, label='τ_jammed')
        ax.plot(vAs, tm, 'rs-', ms=8, lw=2, label='τ_motile')
        ax.plot(vAs, ta, 'k^-', ms=8, lw=2, label='τ_all')
        ax.set_xlabel('v_A')
        ax.set_ylabel('τ')
        ax.set_title('Relaxation Times vs v_A')
        ax.legend(fontsize=9)
        
        # τ ratio
        ax = axes2[1, 3]
        tr = [summary[k]['tau_jammed'][0] / summary[k]['tau_motile'][0]
              if summary[k]['tau_motile'][0] > 0 else np.nan for k, _ in vA_sweep]
        ax.plot(vAs, tr, 'ko-', ms=8, lw=2)
        ax.axhline(1, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('v_A')
        ax.set_ylabel('τ_jammed / τ_motile')
        ax.set_title('Relaxation Ratio vs v_A')
        
        # α₂
        ax = axes2[2, 0]
        a2 = [summary[k]['alpha2'][0] for k, _ in vA_sweep]
        a2e = [summary[k]['alpha2'][1] for k, _ in vA_sweep]
        ax.errorbar(vAs, a2, yerr=a2e, fmt='o-', color='purple', capsize=4, lw=2, ms=8)
        ax.set_xlabel('v_A')
        ax.set_ylabel('α₂')
        ax.set_title('Non-Gaussianity vs v_A')
        
        # MSD
        ax = axes2[2, 1]
        msd = [summary[k]['MSD'][0] for k, _ in vA_sweep]
        msde = [summary[k]['MSD'][1] for k, _ in vA_sweep]
        ax.errorbar(vAs, msd, yerr=msde, fmt='o-', color='brown', capsize=4, lw=2, ms=8)
        ax.set_xlabel('v_A')
        ax.set_ylabel('MSD')
        ax.set_title('MSD vs v_A')
        
        # C(r)
        ax = axes2[2, 2]
        for i, ((vA, sigma), _) in enumerate(vA_sweep):
            runs = all_results[(vA, sigma)]
            mean_cr = np.mean([r['C_r'] for r in runs], axis=0)
            ax.plot(runs[0]['r_bins'], mean_cr, color=vA_colors[i], lw=2,
                    label=f'v_A={vA:.3f}')
        ax.axhline(0, color='gray', ls=':', alpha=0.5)
        ax.axhline(1/np.e, color='red', ls='--', alpha=0.3)
        ax.set_xlabel('r')
        ax.set_ylabel('C(r)')
        ax.set_title('Spatial Autocorrelation by v_A')
        ax.legend(fontsize=8)
        
        # Cluster sizes
        ax = axes2[2, 3]
        ms_j = [summary[k]['mean_jammed_cluster_size'][0] for k, _ in vA_sweep]
        ms_m = [summary[k]['mean_motile_cluster_size'][0] for k, _ in vA_sweep]
        x = np.arange(len(vAs))
        ax.bar(x - w/2, ms_j, w, label='Jammed', color='navy', alpha=0.7)
        ax.bar(x + w/2, ms_m, w, label='Motile', color='firebrick', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.3f}' for v in vAs])
        ax.set_xlabel('v_A')
        ax.set_ylabel('Mean cluster size')
        ax.set_title('Cluster Size vs v_A')
        ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path2 = out_dir / f'griffiths_vA_sweep_{date_str}.png'
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path2}")
        plt.close(fig2)
    
    # ================================================================
    # FIGURE 3: Q(t) comparison — overlap functions
    # ================================================================
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
    fig3.suptitle('Structural Relaxation Q(t) — Griffiths Analysis',
                  fontsize=16, fontweight='bold')
    
    # σ sweep Q(t) — all cells
    ax = axes3[0, 0]
    for i, ((vA, sigma), _) in enumerate(sigma_sweep):
        runs = all_results[(vA, sigma)]
        for r in runs:
            if len(r['Q_lag']) > 0:
                ax.plot(r['Q_lag'], r['Q_all'], color=sigma_colors[i],
                        alpha=0.3, lw=0.5)
        # Average
        if len(runs[0]['Q_lag']) > 0:
            min_len = min(len(r['Q_all']) for r in runs if len(r['Q_all']) > 0)
            avg_Q = np.mean([r['Q_all'][:min_len] for r in runs
                            if len(r['Q_all']) >= min_len], axis=0)
            ax.plot(runs[0]['Q_lag'][:min_len], avg_Q, color=sigma_colors[i],
                    lw=2.5, label=f'σ={sigma:.3f}')
    ax.axhline(1/np.e, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Lag time Δt')
    ax.set_ylabel('Q(Δt)')
    ax.set_title('Q(t) All Cells by σ')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    
    # σ sweep Q(t) — jammed cells only
    ax = axes3[0, 1]
    for i, ((vA, sigma), _) in enumerate(sigma_sweep):
        runs = all_results[(vA, sigma)]
        for r in runs:
            if len(r['Q_jammed_lag']) > 0:
                ax.plot(r['Q_jammed_lag'], r['Q_jammed'], color=sigma_colors[i],
                        alpha=0.3, lw=0.5)
    ax.axhline(1/np.e, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Lag time Δt')
    ax.set_ylabel('Q(Δt)')
    ax.set_title('Q(t) Jammed Cells by σ')
    ax.set_ylim(0, 1.05)
    
    # σ sweep Q(t) — motile cells only
    ax = axes3[0, 2]
    for i, ((vA, sigma), _) in enumerate(sigma_sweep):
        runs = all_results[(vA, sigma)]
        for r in runs:
            if len(r['Q_motile_lag']) > 0:
                ax.plot(r['Q_motile_lag'], r['Q_motile'], color=sigma_colors[i],
                        alpha=0.3, lw=0.5)
    ax.axhline(1/np.e, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Lag time Δt')
    ax.set_ylabel('Q(Δt)')
    ax.set_title('Q(t) Motile Cells by σ')
    ax.set_ylim(0, 1.05)
    
    # v_A sweep Q(t) — all, jammed, motile
    if len(vA_sweep) > 1:
        for col, (qt_key, qt_label) in enumerate([
            ('Q_all', 'All Cells'),
            ('Q_jammed', 'Jammed Cells'),
            ('Q_motile', 'Motile Cells')
        ]):
            ax = axes3[1, col]
            lag_key = qt_key.replace('Q_', 'Q_') + '_lag'
            if qt_key == 'Q_all':
                lag_key = 'Q_lag'
            elif qt_key == 'Q_jammed':
                lag_key = 'Q_jammed_lag'
            else:
                lag_key = 'Q_motile_lag'
            
            for i, ((vA, sigma), _) in enumerate(vA_sweep):
                runs = all_results[(vA, sigma)]
                for r in runs:
                    if len(r[lag_key]) > 0:
                        ax.plot(r[lag_key], r[qt_key], color=vA_colors[i],
                                alpha=0.3, lw=0.5)
                if len(runs[0][lag_key]) > 0 and len(runs) > 0:
                    ax.plot([], [], color=vA_colors[i], lw=2.5,
                            label=f'v_A={vA:.3f}')
            ax.axhline(1/np.e, color='gray', ls='--', alpha=0.5)
            ax.set_xlabel('Lag time Δt')
            ax.set_ylabel('Q(Δt)')
            ax.set_title(f'Q(t) {qt_label} by v_A')
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.05)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path3 = out_dir / f'griffiths_Qt_comparison_{date_str}.png'
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path3}")
    plt.close(fig3)
    
    # ================================================================
    # FIGURE 4: Key Griffiths signatures summary
    # ================================================================
    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
    fig4.suptitle('Key Griffiths Signatures — Does Disorder Create Rare Regions?',
                  fontsize=16, fontweight='bold')
    
    # Panel A: σ=0 control vs σ=0.008 persistence distribution
    ax = axes4[0, 0]
    for key_label, color, ls in [(0.000, 'black', '-'), (0.003, 'blue', '-'),
                                   (0.006, 'green', '-'), (0.008, 'red', '-')]:
        k = (0.008, key_label)
        if k in all_results:
            runs = all_results[k]
            all_pers = np.concatenate([r['persistence_distribution'] for r in runs])
            ax.hist(all_pers, bins=30, density=True, alpha=0.35,
                    color=color, label=f'σ={key_label:.3f}',
                    histtype='stepfilled', edgecolor=color)
    ax.set_xlabel('Persistence (frac. time in majority state)')
    ax.set_ylabel('PDF')
    ax.set_title('KEY TEST: Does Disorder\nIncrease Persistence?')
    ax.legend(fontsize=9)
    ax.annotate('Higher persistence = stronger\nGriffiths rare-region effect',
                xy=(0.95, 0.95), xycoords='axes fraction', fontsize=8,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel B: τ_j/τ_m ratio vs σ  (key Griffiths signature)
    ax = axes4[0, 1]
    sigmas_plot = [k[1] for k, _ in sigma_sweep]
    ratios = [s['tau_jammed'][0] / s['tau_motile'][0]
              if s['tau_motile'][0] > 0 else np.nan for _, s in sigma_sweep]
    colors_bar = ['black' if s == 0 else 'steelblue' for s in sigmas_plot]
    ax.bar(range(len(sigmas_plot)), ratios, color=colors_bar, alpha=0.8,
           edgecolor='black')
    ax.set_xticks(range(len(sigmas_plot)))
    ax.set_xticklabels([f'σ={s:.3f}' for s in sigmas_plot], fontsize=9)
    ax.axhline(1, color='gray', ls=':', alpha=0.5)
    ax.set_ylabel('τ_jammed / τ_motile')
    ax.set_title('KEY TEST: Does Disorder\nSlow Jammed Cells Differentially?')
    ax.annotate('Ratio > 1 = jammed cells\nrelax slower (Griffiths)',
                xy=(0.95, 0.95), xycoords='axes fraction', fontsize=8,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel C: α₂ vs σ  (dynamic heterogeneity)
    ax = axes4[0, 2]
    a2_vals = [summary[(0.008, s)]['alpha2'][0] for s in [0.000, 0.003, 0.006, 0.008]
               if (0.008, s) in summary]
    a2_sigmas = [s for s in [0.000, 0.003, 0.006, 0.008] if (0.008, s) in summary]
    ax.bar(range(len(a2_sigmas)), a2_vals,
           color=['black' if s == 0 else 'coral' for s in a2_sigmas],
           alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(a2_sigmas)))
    ax.set_xticklabels([f'σ={s:.3f}' for s in a2_sigmas], fontsize=9)
    ax.set_ylabel('Non-Gaussian parameter α₂')
    ax.set_title('KEY TEST: Does Disorder Increase\nDynamic Heterogeneity?')
    ax.annotate('Higher α₂ = more\nheterogeneous dynamics',
                xy=(0.95, 0.95), xycoords='axes fraction', fontsize=8,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel D: Summary table
    ax = axes4[1, 0]
    ax.axis('off')
    table_data = []
    table_cols = ['σ', 'Jammed %', 'Persist.', 'τ_j/τ_m', 'ξ/a', 'α₂',
                  'J_perc', 'M_perc']
    for (vA, sigma), s in sigma_sweep:
        tr = s['tau_jammed'][0] / s['tau_motile'][0] if s['tau_motile'][0] > 0 else np.nan
        table_data.append([
            f'{sigma:.3f}',
            f'{s["jammed_frac_mean"][0]*100:.1f}%',
            f'{s["mean_persistence"][0]:.3f}',
            f'{tr:.2f}',
            f'{s["corr_length_cells"][0]:.1f}',
            f'{s["alpha2"][0]:.2f}',
            f'{s["jammed_perc_frac"][0]:.2f}',
            f'{s["motile_perc_frac"][0]:.2f}',
        ])
    table = ax.table(cellText=table_data, colLabels=table_cols,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    # Highlight control row
    for j in range(len(table_cols)):
        table[1, j].set_facecolor('#ffeeee')
    ax.set_title('Summary (v_A=0.008, varying σ)', fontweight='bold', pad=20)
    
    # Panel E: v_A table
    ax = axes4[1, 1]
    ax.axis('off')
    if len(vA_sweep) > 1:
        table_data2 = []
        for (vA, sigma), s in vA_sweep:
            tr = s['tau_jammed'][0] / s['tau_motile'][0] if s['tau_motile'][0] > 0 else np.nan
            table_data2.append([
                f'{vA:.3f}',
                f'{s["jammed_frac_mean"][0]*100:.1f}%',
                f'{s["mean_persistence"][0]:.3f}',
                f'{tr:.2f}',
                f'{s["corr_length_cells"][0]:.1f}',
                f'{s["alpha2"][0]:.2f}',
                f'{s["jammed_perc_frac"][0]:.2f}',
                f'{s["motile_perc_frac"][0]:.2f}',
            ])
        table_cols2 = ['v_A', 'Jammed %', 'Persist.', 'τ_j/τ_m', 'ξ/a', 'α₂',
                        'J_perc', 'M_perc']
        table2 = ax.table(cellText=table_data2, colLabels=table_cols2,
                          loc='center', cellLoc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1.2, 1.5)
        ax.set_title('Summary (σ=0.006, varying v_A)', fontweight='bold', pad=20)
    
    # Panel F: Physical interpretation
    ax = axes4[1, 2]
    ax.axis('off')
    
    # Determine physical conclusions
    control_pers = summary.get((0.008, 0.000), {}).get('mean_persistence', (np.nan, 0))[0]
    max_disorder_pers = summary.get((0.008, 0.008), {}).get('mean_persistence', (np.nan, 0))[0]
    
    control_a2 = summary.get((0.008, 0.000), {}).get('alpha2', (np.nan, 0))[0]
    max_disorder_a2 = summary.get((0.008, 0.008), {}).get('alpha2', (np.nan, 0))[0]
    
    interp_lines = [
        "PHYSICAL INTERPRETATION",
        "─" * 30,
        "",
    ]
    
    if not np.isnan(control_pers) and not np.isnan(max_disorder_pers):
        if max_disorder_pers > control_pers + 0.02:
            interp_lines.append("✓ Disorder INCREASES persistence")
            interp_lines.append("  → Griffiths rare regions confirmed")
        elif max_disorder_pers < control_pers - 0.02:
            interp_lines.append("✗ Disorder DECREASES persistence")
            interp_lines.append("  → Unexpected — need investigation")
        else:
            interp_lines.append("~ Persistence unchanged by disorder")
            interp_lines.append("  → Weak Griffiths effect")
    
    if not np.isnan(control_a2) and not np.isnan(max_disorder_a2):
        interp_lines.append("")
        if max_disorder_a2 > control_a2 * 1.2:
            interp_lines.append("✓ Disorder INCREASES α₂")
            interp_lines.append("  → Enhanced dynamic heterogeneity")
        else:
            interp_lines.append("~ α₂ similar with/without disorder")
    
    # Percolation interpretation
    ctrl_jp = summary.get((0.008, 0.000), {}).get('jammed_perc_frac', (np.nan, 0))[0]
    ctrl_mp = summary.get((0.008, 0.000), {}).get('motile_perc_frac', (np.nan, 0))[0]
    interp_lines.append("")
    if not np.isnan(ctrl_jp):
        if ctrl_jp > 0.5 and ctrl_mp > 0.5:
            interp_lines.append(f"Control: Both percolate → near critical")
        elif ctrl_jp > ctrl_mp:
            interp_lines.append(f"Control: Jammed percolates → below transition")
        else:
            interp_lines.append(f"Control: Motile percolates → above transition")
    
    interp_lines.append("")
    interp_lines.append("NOTE: 288 cells may be too small for")
    interp_lines.append("accurate percolation. Continuation runs")
    interp_lines.append("(→ t=800k) will provide better statistics.")
    interp_lines.append("Large system (18,432 cells) in progress.")
    
    ax.text(0.05, 0.95, '\n'.join(interp_lines), transform=ax.transAxes,
            va='top', ha='left', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path4 = out_dir / f'griffiths_key_signatures_{date_str}.png'
    fig4.savefig(path4, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path4}")
    plt.close(fig4)
    
    # ================================================================
    # WRITE LOGBOOK ENTRY
    # ================================================================
    logbook_path = Path(__file__).parent.parent / 'research_logbook.md'
    print(f"\n  Writing logbook entry to {logbook_path}")
    
    entry = f"""
## Batch Griffiths Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Overview
Comparative analysis of Griffiths rare-region effects across disorder strengths
and mean motility values. Data from nibi cluster, subsampled trajectories
(every 100th timestep, ~1278 frames per run).

**18 runs total**: 6 parameter combos × 3 replicates

### Parameter Space
| v_A | σ | Purpose |
|-----|---|---------|
| 0.008 | **0.000** | **CONTROL** — no disorder |
| 0.008 | 0.003 | Weak disorder |
| 0.008 | 0.006 | Moderate disorder |
| 0.008 | 0.008 | Strong disorder (σ ~ v_A) |
| 0.006 | 0.006 | Lower motility |
| 0.010 | 0.006 | Higher motility |

### Threshold Selection
{'Used σ=0 control MEAN mobility = ' + f'{threshold:.6f}' if threshold else 'Per-run median (not recommended)'}

This is a key methodological choice: by using the control's mean mobility,
we apply the SAME absolute threshold to all parameter sets. Cells with
mobility below this threshold are classified as "jammed" — they are
systematically slower than the average cell in the disorder-free control.
In disordered cases, this identifies cells whose low inherent v_A keeps
them jammed relative to the homogeneous baseline.

### Results Summary — σ Sweep (fixed v_A=0.008)

| σ | Jammed % | Persistence | τ_j/τ_m | ξ/a | α₂ | J_perc | M_perc |
|---|----------|-------------|---------|-----|-----|--------|--------|
"""
    for (vA, sigma), s in sigma_sweep:
        tr = s['tau_jammed'][0] / s['tau_motile'][0] if s['tau_motile'][0] > 0 else np.nan
        entry += (f"| {sigma:.3f} | {s['jammed_frac_mean'][0]*100:.1f}% | "
                  f"{s['mean_persistence'][0]:.3f} | {tr:.2f} | "
                  f"{s['corr_length_cells'][0]:.1f} | {s['alpha2'][0]:.2f} | "
                  f"{s['jammed_perc_frac'][0]:.2f} | {s['motile_perc_frac'][0]:.2f} |\n")
    
    if len(vA_sweep) > 1:
        entry += f"""
### Results Summary — v_A Sweep (fixed σ=0.006)

| v_A | Jammed % | Persistence | τ_j/τ_m | ξ/a | α₂ | J_perc | M_perc |
|-----|----------|-------------|---------|-----|-----|--------|--------|
"""
        for (vA, sigma), s in vA_sweep:
            tr = s['tau_jammed'][0] / s['tau_motile'][0] if s['tau_motile'][0] > 0 else np.nan
            entry += (f"| {vA:.3f} | {s['jammed_frac_mean'][0]*100:.1f}% | "
                      f"{s['mean_persistence'][0]:.3f} | {tr:.2f} | "
                      f"{s['corr_length_cells'][0]:.1f} | {s['alpha2'][0]:.2f} | "
                      f"{s['jammed_perc_frac'][0]:.2f} | {s['motile_perc_frac'][0]:.2f} |\n")
    
    entry += f"""
### Physical Interpretation

**Key question**: Does quenched disorder (σ > 0) create persistent Griffiths
rare regions compared to the homogeneous control (σ = 0)?

#### What to look for:
1. **Persistence increasing with σ**: If cells with high/low inherent v_A
   remain jammed/motile for longer than the dynamic fluctuation timescale
   in the control, this is the hallmark of Griffiths rare regions.

2. **τ_jammed/τ_motile ratio increasing with σ**: In Griffiths physics,
   rare jammed regions embedded in a motile sea have anomalously slow
   relaxation (power-law tails instead of exponential).

3. **Non-Gaussian parameter α₂ increasing with σ**: Dynamic heterogeneity
   should increase as quenched disorder creates a wider distribution of
   local relaxation rates.

4. **Correlation length ξ increasing with σ**: Spatial correlations should
   grow as inherent v_A clusters create correlated mobility patterns.

#### Limitations at current time/size:
- **288 cells** may be too small for reliable percolation analysis
- **t ≈ 330,000** may not be long enough — continuation to t=800,000 in progress
- Threshold = control median is better than per-run median, but still crude
- Need to verify that the σ=0 control truly shows NO persistent spatial
  heterogeneity (its persistence should be ~0.5 for random fluctuations)

### Plots
- Disorder sweep: `postprocessing/output/griffiths_sigma_sweep_{date_str}.png`
- Motility sweep: `postprocessing/output/griffiths_vA_sweep_{date_str}.png`
- Q(t) comparison: `postprocessing/output/griffiths_Qt_comparison_{date_str}.png`
- Key signatures: `postprocessing/output/griffiths_key_signatures_{date_str}.png`

---
"""
    
    mode = 'a' if logbook_path.exists() else 'w'
    with open(logbook_path, mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write("# Research Logbook — Phase Field Cell Simulation\n\n")
        f.write(entry)
    print(f"  Logbook entry written.")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    return all_results, summary


if __name__ == '__main__':
    all_results, summary = main()
