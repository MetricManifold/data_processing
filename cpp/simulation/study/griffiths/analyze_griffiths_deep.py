#!/usr/bin/env python3
"""
Deep Griffiths Analysis — Power-law Q(t), χ₄(t), and v_A-Mobility Correlation

Extends the batch Griffiths analysis with three critical measurements:

1. **Log-log Q(t) with stretched-exponential fits**
   Q(t) = exp(-(t/τ)^β)
   β = 1 → simple exponential (no Griffiths)
   β < 1 → stretched exponential (broad distribution of relaxation times)
   Griffiths rare regions predict β → 0 with power-law tails.

2. **Four-point susceptibility χ₄(t)**
   χ₄(t) = N × [<Q(t)²> - <Q(t)>²]
   Measures the variance of the overlap function across starting times.
   Peak in χ₄(t) indicates cooperative relaxation; peak height grows with
   Griffiths rare-region effects.

3. **v_A–Mobility Pearson Correlation**
   For disordered runs (σ > 0), cells have inherent v_A_i drawn from
   N(v_A, σ²). If quenched disorder matters, inherent v_A should predict
   time-averaged mobility. The Pearson r should increase with σ.
   For the control (σ=0), all v_A_i are identical so r is undefined.

Usage:
  python analyze_griffiths_deep.py path/to/griffiths_subsampled/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import sys
import re
from datetime import datetime
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent))
from visualize_fluid import load_trajectory, compute_displacement_mobility


def parse_filename(fname):
    """Extract v_A and sigma from filename like vA_0.008_sigma_0.006_run_1.txt"""
    m = re.match(r'vA_([\d.]+)_sigma_([\d.]+)_run_(\d+)\.txt', fname)
    if m:
        return float(m.group(1)), float(m.group(2)), int(m.group(3))
    return None, None, None


# ============================================================================
# 1. Multi-origin Q(t) with per-cell resolution (needed for χ₄)
# ============================================================================

def overlap_per_cell(times, positions, header, t0_idx, a=10.0):
    """Compute per-cell overlap w_i(t) = θ(a - |r_i(t) - r_i(t0)|).
    
    Returns:
        lag_times: array of lag times (t - t0)
        W:         2D array (n_lags, n_cells), each entry is 0 or 1
        cell_ids:  list of cell IDs in column order of W
    """
    Lx = float(header.get('Lx', 1600))
    Ly = float(header.get('Ly', 1600))
    
    t0 = times[t0_idx]
    pos0 = positions[t0]
    cell_ids = sorted(pos0.keys())
    N = len(cell_ids)
    
    lag_times = []
    W_rows = []
    
    for ti in range(t0_idx, len(times)):
        t = times[ti]
        dt = t - t0
        pos_t = positions[t]
        
        w = np.zeros(N)
        for j, cid in enumerate(cell_ids):
            if cid not in pos_t:
                w[j] = 0
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
            w[j] = 1.0 if dist < a else 0.0
        
        lag_times.append(dt)
        W_rows.append(w)
    
    return np.array(lag_times), np.array(W_rows), cell_ids


def compute_chi4(times, positions, header, a=10.0, n_origins=20):
    """Compute four-point susceptibility χ₄(t).
    
    χ₄(t) = N × [<Q(t)²> - <Q(t)>²]
    
    where the average is over multiple starting times (origins).
    Q(t) for each origin is the fraction of cells that haven't moved > a.
    
    Args:
        times: array of times
        positions: dict[time] -> dict[cell_id] -> (x,y)
        header: dict with Lx, Ly
        a: cage radius
        n_origins: number of starting times to average over
    
    Returns:
        lag_times: common lag grid
        chi4: χ₄(t) values
        Q_mean: average Q(t)
        Q_std: std of Q(t) over origins
    """
    # Choose starting times evenly spaced in first half of trajectory
    n_total = len(times)
    max_origin = n_total // 2  # use first half as origins
    origin_indices = np.linspace(0, max_origin - 1, n_origins, dtype=int)
    
    # Compute Q(t) for each origin 
    all_Q = []  # list of (lag_array, Q_array) tuples
    N = len(positions[times[0]])
    
    for oi in origin_indices:
        lags, W, _ = overlap_per_cell(times, positions, header, oi, a)
        # Q(t) = mean over cells
        Q = W.mean(axis=1)
        all_Q.append((lags, Q))
    
    # Build common lag grid (use the longest one, which is from earliest origin)
    max_lag = min(q[0][-1] for q in all_Q)
    # Use the lag grid from the first origin, truncated
    ref_lags = all_Q[0][0]
    mask = ref_lags <= max_lag
    common_lags = ref_lags[mask]
    n_lags = len(common_lags)
    
    # Interpolate all Q(t) onto common lag grid
    Q_matrix = np.zeros((n_origins, n_lags))
    for k, (lags, Q) in enumerate(all_Q):
        Q_matrix[k, :] = np.interp(common_lags, lags, Q)
    
    # Q statistics
    Q_mean = Q_matrix.mean(axis=0)
    Q_var = Q_matrix.var(axis=0)
    Q_std = Q_matrix.std(axis=0)
    
    # χ₄(t) = N × Var[Q(t)]
    chi4 = N * Q_var
    
    return common_lags, chi4, Q_mean, Q_std


# ============================================================================
# 2. Stretched exponential fitting
# ============================================================================

def stretched_exp(t, tau, beta):
    """Stretched exponential Q(t) = exp(-(t/tau)^beta)"""
    return np.exp(-(t / tau) ** beta)


def fit_stretched_exp(lag, Q, min_Q=0.05):
    """Fit Q(t) to stretched exponential. Returns (tau, beta, r_squared)."""
    # Only fit where Q > min_Q (avoid noise at small Q)
    mask = (lag > 0) & (Q > min_Q) & (Q < 0.99)
    if mask.sum() < 5:
        return np.nan, np.nan, 0.0
    
    t_fit = lag[mask]
    Q_fit = Q[mask]
    
    try:
        # Initial guesses
        # tau ~ time where Q ≈ 1/e
        tau0 = t_fit[np.argmin(np.abs(Q_fit - 1/np.e))] if np.any(Q_fit < 1/np.e) else t_fit[-1]
        if tau0 <= 0:
            tau0 = t_fit[-1] / 2
        
        popt, _ = curve_fit(stretched_exp, t_fit, Q_fit,
                           p0=[tau0, 0.8],
                           bounds=([1e-3, 0.01], [1e10, 2.0]),
                           maxfev=5000)
        tau, beta = popt
        
        # Compute R²
        Q_pred = stretched_exp(t_fit, tau, beta)
        ss_res = np.sum((Q_fit - Q_pred) ** 2)
        ss_tot = np.sum((Q_fit - Q_fit.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return tau, beta, r2
    except (RuntimeError, ValueError):
        return np.nan, np.nan, 0.0


# ============================================================================
# 3. v_A–Mobility correlation
# ============================================================================

def compute_vA_mobility_correlation(times, positions, mobility, inherent_vA, header):
    """Compute Pearson correlation between inherent v_A and time-averaged mobility.
    
    Args:
        times: array of times
        positions: dict[time] -> dict[cell_id] -> (x,y)
        mobility: dict[time] -> dict[cell_id] -> float
        inherent_vA: dict[cell_id] -> float (from 10th column of trajectory)
        header: dict with Lx, Ly
    
    Returns:
        pearson_r: Pearson correlation coefficient
        p_value: statistical significance
        cell_vA: array of inherent v_A per cell
        cell_mob: array of time-averaged mobility per cell
    """
    if inherent_vA is None:
        return np.nan, np.nan, np.array([]), np.array([])
    
    # Compute time-averaged mobility per cell
    cell_ids = sorted(inherent_vA.keys())
    skip = max(1, len(times) // 10)  # skip first 10%
    
    mob_sums = {cid: 0.0 for cid in cell_ids}
    mob_counts = {cid: 0 for cid in cell_ids}
    
    for t in times[skip:]:
        if t not in mobility:
            continue
        for cid in cell_ids:
            if cid in mobility[t]:
                mob_sums[cid] += mobility[t][cid]
                mob_counts[cid] += 1
    
    cell_vA_list = []
    cell_mob_list = []
    for cid in cell_ids:
        if mob_counts[cid] > 0:
            cell_vA_list.append(inherent_vA[cid])
            cell_mob_list.append(mob_sums[cid] / mob_counts[cid])
    
    cell_vA_arr = np.array(cell_vA_list)
    cell_mob_arr = np.array(cell_mob_list)
    
    if len(cell_vA_arr) < 3 or cell_vA_arr.std() == 0:
        return np.nan, np.nan, cell_vA_arr, cell_mob_arr
    
    from scipy import stats
    r, p = stats.pearsonr(cell_vA_arr, cell_mob_arr)
    return r, p, cell_vA_arr, cell_mob_arr


# ============================================================================
# 4. Time-resolved MSD(t) and D_eff
# ============================================================================

def compute_msd_curve(times, positions, header, n_origins=20, max_lag_frac=0.5):
    """Compute time-resolved MSD(Δt) averaged over multiple origins.
    
    Returns:
        lag_times: array of lag times
        msd: mean squared displacement at each lag
        msd_per_cell: dict[cell_id] -> array of per-cell MSD (for D_eff)
    """
    Lx = float(header.get('Lx', 1600))
    Ly = float(header.get('Ly', 1600))
    n_total = len(times)
    max_origin = n_total // 2
    origin_indices = np.linspace(0, max_origin - 1, n_origins, dtype=int)
    
    cell_ids = sorted(positions[times[0]].keys())
    N = len(cell_ids)
    cid_to_idx = {c: i for i, c in enumerate(cell_ids)}
    
    # Use log-spaced lags for efficiency
    max_lag_idx = int(n_total * max_lag_frac)
    lag_indices = np.unique(np.geomspace(1, max_lag_idx, 80).astype(int))
    lag_indices = lag_indices[lag_indices < n_total]
    
    # Accumulate MSD
    msd_sum = np.zeros(len(lag_indices))
    msd_count = np.zeros(len(lag_indices))
    per_cell_dr2_sum = np.zeros((N, len(lag_indices)))
    per_cell_dr2_count = np.zeros((N, len(lag_indices)))
    
    for oi in origin_indices:
        pos0 = positions[times[oi]]
        for li, lag_idx in enumerate(lag_indices):
            ti = oi + lag_idx
            if ti >= n_total:
                break
            pos_t = positions[times[ti]]
            for j, cid in enumerate(cell_ids):
                if cid not in pos0 or cid not in pos_t:
                    continue
                x0, y0 = pos0[cid]
                x1, y1 = pos_t[cid]
                dx = x1 - x0
                dy = y1 - y0
                if dx > Lx/2: dx -= Lx
                elif dx < -Lx/2: dx += Lx
                if dy > Ly/2: dy -= Ly
                elif dy < -Ly/2: dy += Ly
                dr2 = dx**2 + dy**2
                msd_sum[li] += dr2
                msd_count[li] += 1
                per_cell_dr2_sum[j, li] += dr2
                per_cell_dr2_count[j, li] += 1
    
    valid = msd_count > 0
    lag_times = np.array([(times[oi + lag_idx] - times[oi])
                          for oi, lag_idx in zip([origin_indices[0]], lag_indices)])[valid[:len(lag_indices)]]
    # Recompute lag_times properly
    lag_times = np.array([times[min(origin_indices[0] + li, n_total-1)] - times[origin_indices[0]]
                          for li in lag_indices])
    msd = np.where(msd_count > 0, msd_sum / msd_count, np.nan)
    
    # Per-cell MSD for D_eff
    per_cell_msd = {}
    for j, cid in enumerate(cell_ids):
        valid_c = per_cell_dr2_count[j] > 0
        if valid_c.sum() > 5:
            per_cell_msd[cid] = np.where(valid_c, per_cell_dr2_sum[j] / per_cell_dr2_count[j], np.nan)
    
    return lag_times, msd, per_cell_msd, cell_ids


def compute_deff(lag_times, msd, tau, fit_range=(0.5, 1.0)):
    """Extract effective diffusion coefficient from long-time MSD slope.
    
    D_eff = lim_{t→∞} MSD(t) / (4t)  (2D)
    
    Fits a line to MSD vs t in the range [fit_range[0]*tau, fit_range[1]*max_lag].
    """
    t_min = fit_range[0] * tau
    t_max = lag_times[np.isfinite(msd)][-1] * fit_range[1] if np.any(np.isfinite(msd)) else np.inf
    mask = (lag_times >= t_min) & (lag_times <= t_max) & np.isfinite(msd)
    if mask.sum() < 3:
        # Fall back to last 30% of data
        n_pts = max(3, int(0.3 * np.isfinite(msd).sum()))
        valid_idx = np.where(np.isfinite(msd))[0]
        if len(valid_idx) < 3:
            return np.nan
        mask = np.zeros_like(msd, dtype=bool)
        mask[valid_idx[-n_pts:]] = True
    
    t_fit = lag_times[mask]
    msd_fit = msd[mask]
    
    if len(t_fit) < 2 or t_fit[-1] == t_fit[0]:
        return np.nan
    
    slope, intercept = np.polyfit(t_fit, msd_fit, 1)
    return slope / 4.0  # D = slope/4 in 2D


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Griffiths analysis')
    parser.add_argument('data_dir', help='Directory containing subsampled trajectory files')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Discover files
    files = sorted(data_dir.glob('vA_*_sigma_*_run_*.txt'))
    if not files:
        print(f"ERROR: No files in {data_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("DEEP GRIFFITHS ANALYSIS")
    print("  Log-log Q(t) | χ₄(t) | v_A–Mobility Correlation")
    print("=" * 70)
    print(f"Found {len(files)} files")
    
    # Group by parameter set
    param_groups = defaultdict(list)
    for f in files:
        vA, sigma, run = parse_filename(f.name)
        if vA is not None:
            param_groups[(vA, sigma)].append((run, f))
    
    # ================================================================
    # Establish threshold from control
    # ================================================================
    control_key = None
    for key in param_groups:
        if key[1] == 0.0:
            control_key = key
            break
    
    threshold = None
    if control_key:
        print(f"\nEstablishing threshold from control (σ=0, v_A={control_key[0]})...")
        control_mobs = []
        for run, fpath in param_groups[control_key]:
            times, positions, _, header, _ = load_trajectory(str(fpath), subsample=1)
            window = (times[-1] - times[0]) * 0.05
            mob = compute_displacement_mobility(times, positions, header, window)
            skip = max(1, len(times) // 10)
            for t in times[skip:]:
                control_mobs.extend(mob[t].values())
        control_mobs = np.array(control_mobs)
        threshold = control_mobs.mean()
        print(f"  Threshold = {threshold:.6f} (control mean)")
    
    # ================================================================
    # Process all runs
    # ================================================================
    all_data = {}
    
    for (vA, sigma), runs in sorted(param_groups.items()):
        print(f"\n{'='*50}")
        print(f"v_A={vA:.3f}, σ={sigma:.3f} ({len(runs)} replicates)")
        print(f"{'='*50}")
        
        run_data = []
        for run_id, fpath in sorted(runs):
            print(f"  Run {run_id}: {fpath.name}")
            
            times, positions, velocities, header, inherent_vA = load_trajectory(
                str(fpath), subsample=1)
            
            N_cells = len(positions[times[0]])
            Lx = float(header.get('Lx', 1600))
            Ly = float(header.get('Ly', 1600))
            cell_spacing = np.sqrt(Lx * Ly / N_cells)
            cage_radius = cell_spacing * 0.3
            
            # Compute mobility
            window = (times[-1] - times[0]) * 0.05
            mobility = compute_displacement_mobility(times, positions, header, window)
            
            # --- Multi-origin Q(t) + χ₄ ---
            print(f"    Computing χ₄(t) with 20 origins...")
            chi4_lags, chi4, Q_mean, Q_std = compute_chi4(
                times, positions, header, a=cage_radius, n_origins=20)
            
            # --- Stretched exponential fit ---
            tau_se, beta_se, r2_se = fit_stretched_exp(chi4_lags, Q_mean)
            print(f"    Stretched exp fit: τ={tau_se:.0f}, β={beta_se:.3f}, R²={r2_se:.3f}")
            
            # --- χ₄ peak ---
            if len(chi4) > 0 and np.any(np.isfinite(chi4)):
                chi4_peak = np.nanmax(chi4)
                chi4_peak_t = chi4_lags[np.nanargmax(chi4)]
            else:
                chi4_peak = np.nan
                chi4_peak_t = np.nan
            print(f"    χ₄ peak = {chi4_peak:.2f} at t = {chi4_peak_t:.0f}")
            
            # --- v_A–mobility correlation ---
            r_corr, p_corr, cell_vA_arr, cell_mob_arr = compute_vA_mobility_correlation(
                times, positions, mobility, inherent_vA, header)
            if not np.isnan(r_corr):
                print(f"    Pearson r(v_A, mob) = {r_corr:.3f} (p={p_corr:.2e})")
            else:
                print(f"    Pearson r: N/A (σ=0 or insufficient data)")
            
            # --- Class-resolved Q(t) ---
            from analyze_griffiths import classify_cells, overlap_function
            t_ref_idx = len(times) // 4
            t_ref = times[t_ref_idx]
            ref_labels = classify_cells(mobility[t_ref], threshold or 0)
            qt_times = times[t_ref_idx:]
            
            lag_all, Q_all = overlap_function(qt_times, positions, header, a=cage_radius)
            lag_j, Q_j = overlap_function(qt_times, positions, header, a=cage_radius,
                                          cell_class=ref_labels, class_label=0)
            lag_m, Q_m = overlap_function(qt_times, positions, header, a=cage_radius,
                                          cell_class=ref_labels, class_label=1)
            
            # Fit each class
            tau_j, beta_j, r2_j = fit_stretched_exp(lag_j, Q_j)
            tau_m, beta_m, r2_m = fit_stretched_exp(lag_m, Q_m)
            print(f"    Jammed:  τ={tau_j:.0f}, β={beta_j:.3f}")
            print(f"    Motile:  τ={tau_m:.0f}, β={beta_m:.3f}")
            
            # --- Time-resolved MSD(t) and D_eff ---
            print(f"    Computing MSD(t) and D_eff...")
            tau_phys = float(header.get('tau', 10000))
            msd_lags, msd_vals, per_cell_msd, msd_cell_ids = compute_msd_curve(
                times, positions, header, n_origins=20)
            deff = compute_deff(msd_lags, msd_vals, tau_phys)
            print(f"    D_eff = {deff:.6f}")
            
            # Per-cell D_eff for CV computation
            cell_deffs = {}
            for cid in per_cell_msd:
                cell_deffs[cid] = compute_deff(msd_lags, per_cell_msd[cid], tau_phys)
            deff_arr = np.array([v for v in cell_deffs.values() if np.isfinite(v)])
            deff_cv = deff_arr.std() / deff_arr.mean() if len(deff_arr) > 0 and deff_arr.mean() > 0 else np.nan
            print(f"    D_eff CV = {deff_cv:.3f} ({len(deff_arr)} cells)")
            
            run_data.append({
                'run': run_id,
                'N': N_cells,
                'cell_spacing': cell_spacing,
                # χ₄
                'chi4_lags': chi4_lags,
                'chi4': chi4,
                'Q_mean': Q_mean,
                'Q_std': Q_std,
                'chi4_peak': chi4_peak,
                'chi4_peak_t': chi4_peak_t,
                # Stretched exp (multi-origin average)
                'tau_se': tau_se, 'beta_se': beta_se, 'r2_se': r2_se,
                # Class-resolved
                'lag_all': lag_all, 'Q_all_curve': Q_all,
                'lag_j': lag_j, 'Q_j': Q_j,
                'lag_m': lag_m, 'Q_m': Q_m,
                'tau_j': tau_j, 'beta_j': beta_j,
                'tau_m': tau_m, 'beta_m': beta_m,
                # v_A correlation
                'pearson_r': r_corr, 'pearson_p': p_corr,
                'cell_vA': cell_vA_arr, 'cell_mob': cell_mob_arr,
                # MSD and D_eff
                'msd_lags': msd_lags, 'msd': msd_vals,
                'deff': deff, 'deff_cv': deff_cv,
                'per_cell_deffs': deff_arr,
            })
        
        all_data[(vA, sigma)] = run_data
    
    # ================================================================
    # FIGURE 1: Log-log Q(t) with stretched exponential fits
    # ================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    sigma_sweep = sorted([(k, v) for k, v in all_data.items() if k[0] == 0.008],
                          key=lambda x: x[0][1])
    vA_sweep = sorted([(k, v) for k, v in all_data.items() if k[1] == 0.006],
                       key=lambda x: x[0][0])
    
    sigma_colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(sigma_sweep)))
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(20, 13), facecolor='white')
    fig1.suptitle('Power-Law vs Exponential Relaxation — Griffiths Test\n'
                  'Q(t) = exp(-(t/τ)^β): β=1 exponential, β<1 stretched (Griffiths)',
                  fontsize=14, fontweight='bold')
    
    # Panel A: Log-log Q(t) all cells, σ sweep
    ax = axes1[0, 0]
    for i, ((vA, sigma), runs) in enumerate(sigma_sweep):
        for r in runs:
            mask = (r['lag_all'] > 0) & (r['Q_all_curve'] > 0.01)
            if mask.any():
                ax.loglog(r['lag_all'][mask], r['Q_all_curve'][mask],
                         color=sigma_colors[i], alpha=0.3, lw=0.8)
        # Plot fit line
        r0 = runs[0]
        if not np.isnan(r0['tau_se']) and r0['tau_se'] > 0:
            t_fit_plot = np.logspace(0, np.log10(max(r0['chi4_lags'][-1], 100)), 100)
            Q_fit_plot = stretched_exp(t_fit_plot, r0['tau_se'], r0['beta_se'])
            ax.loglog(t_fit_plot, Q_fit_plot, '--', color=sigma_colors[i], 
                     lw=2, label=f'σ={sigma:.3f} β={r0["beta_se"]:.2f}')
    ax.axhline(1/np.e, color='gray', ls=':', alpha=0.5, lw=1)
    ax.set_xlabel('Lag time Δt', fontsize=11)
    ax.set_ylabel('Q(Δt)', fontsize=11)
    ax.set_title('Log-Log Q(t) — All Cells (σ sweep)')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0.01, 1.2)
    
    # Panel B: Log-log Q(t) jammed cells
    ax = axes1[0, 1]
    for i, ((vA, sigma), runs) in enumerate(sigma_sweep):
        for r in runs:
            mask = (r['lag_j'] > 0) & (r['Q_j'] > 0.01)
            if mask.any():
                ax.loglog(r['lag_j'][mask], r['Q_j'][mask],
                         color=sigma_colors[i], alpha=0.3, lw=0.8)
        r0 = runs[0]
        if not np.isnan(r0['tau_j']):
            ax.plot([], [], color=sigma_colors[i], lw=2,
                   label=f'σ={sigma:.3f} β_j={r0["beta_j"]:.2f}')
    ax.axhline(1/np.e, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Lag time Δt', fontsize=11)
    ax.set_ylabel('Q(Δt)', fontsize=11)
    ax.set_title('Log-Log Q(t) — Jammed Cells')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0.01, 1.2)
    
    # Panel C: Log-log Q(t) motile cells
    ax = axes1[0, 2]
    for i, ((vA, sigma), runs) in enumerate(sigma_sweep):
        for r in runs:
            mask = (r['lag_m'] > 0) & (r['Q_m'] > 0.01)
            if mask.any():
                ax.loglog(r['lag_m'][mask], r['Q_m'][mask],
                         color=sigma_colors[i], alpha=0.3, lw=0.8)
        r0 = runs[0]
        if not np.isnan(r0['tau_m']):
            ax.plot([], [], color=sigma_colors[i], lw=2,
                   label=f'σ={sigma:.3f} β_m={r0["beta_m"]:.2f}')
    ax.axhline(1/np.e, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Lag time Δt', fontsize=11)
    ax.set_ylabel('Q(Δt)', fontsize=11)
    ax.set_title('Log-Log Q(t) — Motile Cells')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0.01, 1.2)
    
    # Panel D: β exponent vs σ
    ax = axes1[1, 0]
    sigmas_plot = []
    beta_all_vals = []
    beta_j_vals = []
    beta_m_vals = []
    for (vA, sigma), runs in sigma_sweep:
        sigmas_plot.append(sigma)
        betas_all = [r['beta_se'] for r in runs if not np.isnan(r['beta_se'])]
        betas_j = [r['beta_j'] for r in runs if not np.isnan(r['beta_j'])]
        betas_m = [r['beta_m'] for r in runs if not np.isnan(r['beta_m'])]
        beta_all_vals.append((np.mean(betas_all), np.std(betas_all)/np.sqrt(len(betas_all))) if betas_all else (np.nan, 0))
        beta_j_vals.append((np.mean(betas_j), np.std(betas_j)/np.sqrt(len(betas_j))) if betas_j else (np.nan, 0))
        beta_m_vals.append((np.mean(betas_m), np.std(betas_m)/np.sqrt(len(betas_m))) if betas_m else (np.nan, 0))
    
    ax.errorbar(sigmas_plot, [v[0] for v in beta_all_vals], yerr=[v[1] for v in beta_all_vals],
               fmt='ko-', ms=8, lw=2, capsize=4, label='All cells')
    ax.errorbar(sigmas_plot, [v[0] for v in beta_j_vals], yerr=[v[1] for v in beta_j_vals],
               fmt='bs-', ms=7, lw=1.5, capsize=4, label='Jammed')
    ax.errorbar(sigmas_plot, [v[0] for v in beta_m_vals], yerr=[v[1] for v in beta_m_vals],
               fmt='r^-', ms=7, lw=1.5, capsize=4, label='Motile')
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Pure exponential')
    ax.set_xlabel('Disorder strength σ', fontsize=11)
    ax.set_ylabel('Stretching exponent β', fontsize=11)
    ax.set_title('KEY TEST: β < 1 → Griffiths\n(Broad relaxation spectrum)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.5)
    ax.annotate('β = 1: simple exponential\nβ < 1: stretched (Griffiths)\nβ → 0: power-law tails',
                xy=(0.95, 0.95), xycoords='axes fraction', fontsize=8,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel E: Relaxation timescale ratio τ_j/τ_m vs σ
    ax = axes1[1, 1]
    tau_ratios_se = []
    for (vA, sigma), runs in sigma_sweep:
        ratios = [r['tau_j'] / r['tau_m'] for r in runs
                  if not np.isnan(r['tau_j']) and not np.isnan(r['tau_m']) and r['tau_m'] > 0]
        if ratios:
            tau_ratios_se.append((np.mean(ratios), np.std(ratios) / np.sqrt(len(ratios))))
        else:
            tau_ratios_se.append((np.nan, 0))
    
    ax.errorbar(sigmas_plot, [v[0] for v in tau_ratios_se], [v[1] for v in tau_ratios_se],
               fmt='ko-', ms=8, lw=2, capsize=4)
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('σ', fontsize=11)
    ax.set_ylabel('τ_jammed / τ_motile (from fit)', fontsize=11)
    ax.set_title('Relaxation Timescale Separation')
    
    # Panel F: Summary table of fits
    ax = axes1[1, 2]
    ax.axis('off')
    table_data = []
    for idx, ((vA, sigma), runs) in enumerate(sigma_sweep):
        # Average over replicates
        b_all = np.nanmean([r['beta_se'] for r in runs])
        b_j = np.nanmean([r['beta_j'] for r in runs])
        b_m = np.nanmean([r['beta_m'] for r in runs])
        t_all = np.nanmean([r['tau_se'] for r in runs])
        t_j = np.nanmean([r['tau_j'] for r in runs])
        t_m = np.nanmean([r['tau_m'] for r in runs])
        r2 = np.nanmean([r['r2_se'] for r in runs])
        table_data.append([
            f'{sigma:.3f}',
            f'{b_all:.3f}',
            f'{b_j:.3f}',
            f'{b_m:.3f}',
            f'{t_all:.0f}',
            f'{t_j:.0f}',
            f'{t_m:.0f}',
            f'{r2:.3f}',
        ])
    
    cols = ['σ', 'β_all', 'β_jam', 'β_mot', 'τ_all', 'τ_jam', 'τ_mot', 'R²']
    table = ax.table(cellText=table_data, colLabels=cols,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    # Color code: highlight control row
    for j in range(len(cols)):
        table[1, j].set_facecolor('#ffeeee')
    ax.set_title('Stretched Exponential Fit Parameters', fontweight='bold', pad=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path1 = out_dir / f'griffiths_loglog_Qt_{date_str}.png'
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path1}")
    plt.close(fig1)
    
    # ================================================================
    # FIGURE 2: χ₄(t) four-point susceptibility
    # ================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6), facecolor='white')
    fig2.suptitle('Four-Point Susceptibility χ₄(t) — Dynamic Correlation Length\n'
                  'χ₄ = N × Var[Q(t)]: Peak indicates cooperative rearrangements',
                  fontsize=14, fontweight='bold')
    
    # Panel A: χ₄(t) by σ
    ax = axes2[0]
    for i, ((vA, sigma), runs) in enumerate(sigma_sweep):
        for r in runs:
            mask = r['chi4_lags'] > 0
            ax.plot(r['chi4_lags'][mask], r['chi4'][mask],
                   color=sigma_colors[i], alpha=0.3, lw=0.8)
        # Average
        min_len = min(len(r['chi4']) for r in runs)
        avg_chi4 = np.mean([r['chi4'][:min_len] for r in runs], axis=0)
        avg_lags = runs[0]['chi4_lags'][:min_len]
        ax.plot(avg_lags[avg_lags > 0], avg_chi4[avg_lags > 0],
               color=sigma_colors[i], lw=2.5, label=f'σ={sigma:.3f}')
    ax.set_xlabel('Lag time Δt', fontsize=11)
    ax.set_ylabel('χ₄(Δt)', fontsize=11)
    ax.set_title('χ₄(t) by Disorder Strength σ')
    ax.legend(fontsize=9)
    
    # Panel B: χ₄ peak height vs σ
    ax = axes2[1]
    chi4_peaks = []
    chi4_peak_ts = []
    for (vA, sigma), runs in sigma_sweep:
        peaks = [r['chi4_peak'] for r in runs if not np.isnan(r['chi4_peak'])]
        peak_ts = [r['chi4_peak_t'] for r in runs if not np.isnan(r['chi4_peak_t'])]
        chi4_peaks.append((np.mean(peaks), np.std(peaks)/np.sqrt(len(peaks))) if peaks else (np.nan, 0))
        chi4_peak_ts.append((np.mean(peak_ts), np.std(peak_ts)/np.sqrt(len(peak_ts))) if peak_ts else (np.nan, 0))
    
    ax.errorbar(sigmas_plot, [v[0] for v in chi4_peaks], [v[1] for v in chi4_peaks],
               fmt='ko-', ms=8, lw=2, capsize=4)
    ax.set_xlabel('σ', fontsize=11)
    ax.set_ylabel('χ₄ peak height', fontsize=11)
    ax.set_title('KEY: χ₄ Peak vs σ\n(Higher = more cooperative)')
    ax.annotate('Growing χ₄ peak with σ\n→ increasing dynamic\ncorrelation length',
                xy=(0.95, 0.95), xycoords='axes fraction', fontsize=8,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel C: χ₄ peak time vs σ
    ax = axes2[2]
    ax.errorbar(sigmas_plot, [v[0] for v in chi4_peak_ts], [v[1] for v in chi4_peak_ts],
               fmt='ro-', ms=8, lw=2, capsize=4)
    ax.set_xlabel('σ', fontsize=11)
    ax.set_ylabel('Time of χ₄ peak', fontsize=11)
    ax.set_title('χ₄ Peak Position vs σ')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path2 = out_dir / f'griffiths_chi4_{date_str}.png'
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path2}")
    plt.close(fig2)
    
    # ================================================================
    # FIGURE 3: v_A–Mobility Correlation
    # ================================================================
    # Only for σ > 0 runs that have per-cell v_A data (10-column format)
    disordered_sweep = [(k, v) for k, v in sigma_sweep if k[1] > 0]
    # Check if any run actually has v_A data
    has_vA_data = any(
        len(r['cell_vA']) > 0
        for _, runs in disordered_sweep
        for r in runs
    )
    
    if disordered_sweep and has_vA_data:
        n_combos = len(disordered_sweep)
        n_cols = min(3, n_combos)
        n_rows = (n_combos + n_cols - 1) // n_cols + 1  # +1 for summary
        
        fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows),
                                    facecolor='white')
        fig3.suptitle('Quenched Disorder: Inherent v_A vs Time-Averaged Mobility\n'
                      'Pearson r → 1 means inherent motility predicts dynamic mobility',
                      fontsize=14, fontweight='bold')
        if n_rows == 1:
            axes3 = np.array([axes3])
        if n_cols == 1:
            axes3 = axes3.reshape(-1, 1)
        
        # Scatter plots for each disordered parameter set
        for idx, ((vA, sigma), runs) in enumerate(disordered_sweep):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes3[row, col]
            
            for r in runs:
                if len(r['cell_vA']) > 0:
                    ax.scatter(r['cell_vA'], r['cell_mob'], alpha=0.3, s=10,
                             color=sigma_colors[1 + idx])
            
            # Combined stats
            all_vA = np.concatenate([r['cell_vA'] for r in runs if len(r['cell_vA']) > 0])
            all_mob = np.concatenate([r['cell_mob'] for r in runs if len(r['cell_mob']) > 0])
            pearson_avg = np.nanmean([r['pearson_r'] for r in runs])
            
            if len(all_vA) > 2 and all_vA.std() > 0:
                # Trend line
                from scipy import stats
                slope, intercept, _, _, _ = stats.linregress(all_vA, all_mob)
                x_line = np.linspace(all_vA.min(), all_vA.max(), 50)
                ax.plot(x_line, slope * x_line + intercept, 'r-', lw=2)
            
            ax.set_xlabel('Inherent v_A_i', fontsize=10)
            ax.set_ylabel('Time-avg mobility', fontsize=10)
            ax.set_title(f'σ={sigma:.3f}: r = {pearson_avg:.3f}', fontsize=11)
        
        # Hide unused axes in scatter rows
        for idx in range(n_combos, n_cols * (n_rows - 1)):
            row = idx // n_cols
            col = idx % n_cols
            axes3[row, col].set_visible(False)
        
        # Summary panel: Pearson r vs σ
        ax = axes3[-1, 0]
        disorder_sigmas = [k[1] for k, _ in disordered_sweep]
        r_means = []
        r_errs = []
        for (vA, sigma), runs in disordered_sweep:
            rs = [r['pearson_r'] for r in runs if not np.isnan(r['pearson_r'])]
            r_means.append(np.mean(rs) if rs else np.nan)
            r_errs.append(np.std(rs) / np.sqrt(len(rs)) if len(rs) > 1 else 0)
        ax.errorbar(disorder_sigmas, r_means, yerr=r_errs,
                   fmt='ko-', ms=8, lw=2, capsize=4)
        ax.axhline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Disorder strength σ', fontsize=11)
        ax.set_ylabel('Pearson r(v_A, mobility)', fontsize=11)
        ax.set_title('Quenched Disorder Predictability')
        ax.set_ylim(-0.2, 1.0)
        ax.annotate('r → 1: inherent v_A fully\ndetermines mobility\n'
                   '(frozen Griffiths regions)',
                   xy=(0.95, 0.95), xycoords='axes fraction', fontsize=8,
                   ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide remaining axes
        for col in range(1, n_cols):
            if axes3[-1, col] is not None:
                axes3[-1, col].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        path3 = out_dir / f'griffiths_vA_correlation_{date_str}.png'
        fig3.savefig(path3, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path3}")
        plt.close(fig3)
    else:
        print("  Skipping v_A correlation plot (no per-cell v_A data in trajectories)")
        print("  To enable: re-extract subsampled data with all 10 columns from cluster")
    
    # ================================================================
    # PRINT SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("DEEP GRIFFITHS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'σ':>6} {'β_all':>7} {'β_jam':>7} {'β_mot':>7} {'χ₄_pk':>8} "
          f"{'t_pk':>7} {'r(vA)':>7} {'D_eff':>9} {'CV':>6}")
    print("-" * 70)
    
    for (vA, sigma), runs in sigma_sweep:
        beta_avg = np.nanmean([r['beta_se'] for r in runs])
        beta_j_avg = np.nanmean([r['beta_j'] for r in runs])
        beta_m_avg = np.nanmean([r['beta_m'] for r in runs])
        chi4_pk_avg = np.nanmean([r['chi4_peak'] for r in runs])
        chi4_t_avg = np.nanmean([r['chi4_peak_t'] for r in runs])
        r_avg = np.nanmean([r['pearson_r'] for r in runs])
        deff_avg = np.nanmean([r['deff'] for r in runs])
        cv_avg = np.nanmean([r['deff_cv'] for r in runs])
        
        print(f"{sigma:6.3f} {beta_avg:7.3f} {beta_j_avg:7.3f} {beta_m_avg:7.3f} "
              f"{chi4_pk_avg:8.2f} {chi4_t_avg:7.0f} {r_avg:7.3f} "
              f"{deff_avg:9.6f} {cv_avg:6.3f}")
    
    # ================================================================
    # LOGBOOK ENTRY
    # ================================================================
    logbook_path = Path(__file__).parent.parent / 'research_logbook.md'
    
    entry = f"""
## Deep Griffiths Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M')}

### New Measurements

#### 1. Stretched Exponential Fits: Q(t) = exp(-(t/τ)^β)

The stretching exponent β is a key Griffiths indicator:
- β = 1: simple exponential → single relaxation time
- β < 1: stretched exponential → broad distribution of relaxation times
- β → 0: approaches power-law → hallmark of Griffiths rare regions

| σ | β_all | β_jammed | β_motile | τ_all | τ_jammed | τ_motile | R² |
|---|-------|----------|----------|-------|----------|----------|-----|
"""
    for (vA, sigma), runs in sigma_sweep:
        b_a = np.nanmean([r['beta_se'] for r in runs])
        b_j = np.nanmean([r['beta_j'] for r in runs])
        b_m = np.nanmean([r['beta_m'] for r in runs])
        t_a = np.nanmean([r['tau_se'] for r in runs])
        t_j = np.nanmean([r['tau_j'] for r in runs])
        t_m = np.nanmean([r['tau_m'] for r in runs])
        r2 = np.nanmean([r['r2_se'] for r in runs])
        entry += f"| {sigma:.3f} | {b_a:.3f} | {b_j:.3f} | {b_m:.3f} | {t_a:.0f} | {t_j:.0f} | {t_m:.0f} | {r2:.3f} |\n"
    
    entry += f"""
#### 2. Four-Point Susceptibility χ₄(t)

χ₄(t) = N × Var[Q(t)] measures the spatial extent of cooperative
rearrangements. A growing peak height with σ indicates increasing
dynamic correlation length due to Griffiths rare regions.

| σ | χ₄ peak | t at peak |
|---|---------|-----------|
"""
    for (vA, sigma), runs in sigma_sweep:
        pk = np.nanmean([r['chi4_peak'] for r in runs])
        tp = np.nanmean([r['chi4_peak_t'] for r in runs])
        entry += f"| {sigma:.3f} | {pk:.2f} | {tp:.0f} |\n"
    
    entry += f"""
#### 3. Inherent v_A – Mobility Correlation

For σ > 0, each cell has a fixed inherent v_A drawn from N(v_A, σ²).
The Pearson r measures how well inherent motility predicts dynamic mobility.
r → 1 means quenched disorder completely determines dynamics (frozen Griffiths regions).

| σ | Pearson r | p-value |
|---|-----------|---------|
"""
    for (vA, sigma), runs in sigma_sweep:
        r_avg = np.nanmean([r['pearson_r'] for r in runs])
        p_avg = np.nanmean([r['pearson_p'] for r in runs])
        if sigma == 0:
            entry += f"| {sigma:.3f} | N/A (control) | N/A |\n"
        else:
            entry += f"| {sigma:.3f} | {r_avg:.3f} | {p_avg:.2e} |\n"
    
    entry += f"""
### Plots
- Log-log Q(t) + β fits: `postprocessing/output/griffiths_loglog_Qt_{date_str}.png`
- χ₄(t) susceptibility: `postprocessing/output/griffiths_chi4_{date_str}.png`
- v_A correlation: `postprocessing/output/griffiths_vA_correlation_{date_str}.png`

---
"""
    
    with open(logbook_path, 'a', encoding='utf-8') as f:
        f.write(entry)
    print(f"\n  Logbook entry appended to {logbook_path}")
    
    print("\n" + "=" * 70)
    print("DONE — Deep Griffiths Analysis Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
