#!/usr/bin/env python3
"""
Analyze trajectory data to reproduce plots from jamming transition paper.
Computes:
- Mean Squared Displacement (MSD)
- Velocity autocorrelation function
- Polarization autocorrelation function
- Effective diffusion coefficient
"""

import numpy as np
import matplotlib
import argparse
from pathlib import Path

# Parse args early to set backend before importing pyplot
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--no-show', action='store_true')
_args, _ = _parser.parse_known_args()
if _args.no_show:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def load_trajectory(filepath):
    """Load trajectory data from file."""
    data = []
    v_A = None
    N = None
    Lx, Ly = None, None  # Domain size
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                # Parse header for parameters
                if 'v_A=' in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith('v_A='):
                            v_A = float(p.split('=')[1])
                        elif p.startswith('N='):
                            N = int(p.split('=')[1])
                        elif p.startswith('Lx='):
                            Lx = float(p.split('=')[1])
                        elif p.startswith('Ly='):
                            Ly = float(p.split('=')[1])
                continue
            
            parts = line.strip().split()
            if len(parts) >= 9:
                t, cell_id, x, y, vx, vy, px, py, theta = map(float, parts[:9])
                data.append([t, int(cell_id), x, y, vx, vy, px, py, theta])
    
    data = np.array(data)
    
    # Fallback: infer N from unique cell IDs if not in header
    if N is None and len(data) > 0:
        N = len(np.unique(data[:, 1]))
    
    # Fallback: estimate domain size from position range if not in header
    # Cells should stay within [0, L), so max position is close to L
    if (Lx is None or Ly is None) and len(data) > 0:
        if Lx is None:
            # Round up to nearest reasonable domain size
            max_x = data[:, 2].max()
            Lx = np.ceil(max_x / 10) * 10  # Round up to nearest 10
        if Ly is None:
            max_y = data[:, 3].max()
            Ly = np.ceil(max_y / 10) * 10
        print(f"Warning: Domain size not in header, estimated Lx={Lx:.0f}, Ly={Ly:.0f}")
    
    return data, v_A, N, Lx, Ly


def compute_msd(data, N, Lx=None, Ly=None):
    """Compute mean squared displacement as a function of time lag.
    
    Uses unwrapped positions to correctly handle periodic boundaries.
    """
    # Get unique times
    times = np.unique(data[:, 0])
    n_times = len(times)
    
    # Reorganize data by cell and time, with unwrapped positions
    positions = {}  # positions[cell_id][time_idx] = (x, y) unwrapped
    
    for cell_id in np.unique(data[:, 1]).astype(int):
        mask = data[:, 1] == cell_id
        x = data[mask, 2].copy()
        y = data[mask, 3].copy()
        cell_times = data[mask, 0]
        
        # Unwrap periodic boundaries
        if Lx is not None and Ly is not None:
            for j in range(1, len(x)):
                dx = x[j] - x[j-1]
                dy = y[j] - y[j-1]
                
                while dx > Lx / 2:
                    x[j:] -= Lx
                    dx = x[j] - x[j-1]
                while dx < -Lx / 2:
                    x[j:] += Lx
                    dx = x[j] - x[j-1]
                    
                while dy > Ly / 2:
                    y[j:] -= Ly
                    dy = y[j] - y[j-1]
                while dy < -Ly / 2:
                    y[j:] += Ly
                    dy = y[j] - y[j-1]
        
        positions[cell_id] = {}
        for j, t in enumerate(cell_times):
            time_idx = np.where(times == t)[0][0]
            positions[cell_id][time_idx] = np.array([x[j], y[j]])
    
    # Compute MSD for various time lags
    max_lag = n_times // 2  # Use at most half the trajectory
    lags = []
    msds = []
    msd_stds = []
    
    for lag in range(1, max_lag):
        displacements_sq = []
        
        for cell_id in positions:
            for t_idx in range(n_times - lag):
                if t_idx in positions[cell_id] and (t_idx + lag) in positions[cell_id]:
                    r0 = positions[cell_id][t_idx]
                    r1 = positions[cell_id][t_idx + lag]
                    dr = r1 - r0
                    displacements_sq.append(np.sum(dr**2))
        
        if displacements_sq:
            lags.append((times[lag] - times[0]))
            msds.append(np.mean(displacements_sq))
            msd_stds.append(np.std(displacements_sq) / np.sqrt(len(displacements_sq)))
    
    return np.array(lags), np.array(msds), np.array(msd_stds)


def compute_velocity_autocorrelation(data, N):
    """Compute velocity autocorrelation function."""
    times = np.unique(data[:, 0])
    n_times = len(times)
    
    # Reorganize velocity data
    velocities = {}
    for row in data:
        t, cell_id, vx, vy = row[0], int(row[1]), row[4], row[5]
        if cell_id not in velocities:
            velocities[cell_id] = {}
        time_idx = np.where(times == t)[0][0]
        velocities[cell_id][time_idx] = np.array([vx, vy])
    
    # Compute autocorrelation
    max_lag = n_times // 2
    lags = []
    autocorrs = []
    autocorr_stds = []
    
    for lag in range(0, max_lag):
        correlations = []
        
        for cell_id in velocities:
            for t_idx in range(n_times - lag):
                if t_idx in velocities[cell_id] and (t_idx + lag) in velocities[cell_id]:
                    v0 = velocities[cell_id][t_idx]
                    v1 = velocities[cell_id][t_idx + lag]
                    # Normalize by |v0|^2
                    v0_sq = np.sum(v0**2)
                    if v0_sq > 1e-12:
                        correlations.append(np.dot(v0, v1) / v0_sq)
        
        if correlations:
            lags.append(times[lag] - times[0])
            autocorrs.append(np.mean(correlations))
            autocorr_stds.append(np.std(correlations) / np.sqrt(len(correlations)))
    
    return np.array(lags), np.array(autocorrs), np.array(autocorr_stds)


def compute_polarization_autocorrelation(data, N):
    """Compute polarization direction autocorrelation function."""
    times = np.unique(data[:, 0])
    n_times = len(times)
    
    # Reorganize polarization data
    polarizations = {}
    for row in data:
        t, cell_id, px, py = row[0], int(row[1]), row[6], row[7]
        if cell_id not in polarizations:
            polarizations[cell_id] = {}
        time_idx = np.where(times == t)[0][0]
        polarizations[cell_id][time_idx] = np.array([px, py])
    
    # Compute autocorrelation
    max_lag = n_times // 2
    lags = []
    autocorrs = []
    autocorr_stds = []
    
    for lag in range(0, max_lag):
        correlations = []
        
        for cell_id in polarizations:
            for t_idx in range(n_times - lag):
                if t_idx in polarizations[cell_id] and (t_idx + lag) in polarizations[cell_id]:
                    p0 = polarizations[cell_id][t_idx]
                    p1 = polarizations[cell_id][t_idx + lag]
                    # p is unit vector, so p0·p1 = cos(delta_theta)
                    correlations.append(np.dot(p0, p1))
        
        if correlations:
            lags.append(times[lag] - times[0])
            autocorrs.append(np.mean(correlations))
            autocorr_stds.append(np.std(correlations) / np.sqrt(len(correlations)))
    
    return np.array(lags), np.array(autocorrs), np.array(autocorr_stds)


def compute_effective_diffusion(lags, msd):
    """Compute effective diffusion coefficient from MSD slope at long times."""
    # D_eff = MSD / (4 * t) in 2D
    # Use linear fit to long-time behavior
    if len(lags) > 10:
        # Use last half of data for long-time behavior
        n_half = len(lags) // 2
        slope, intercept = np.polyfit(lags[n_half:], msd[n_half:], 1)
        D_eff = slope / 4  # MSD = 4*D*t in 2D
        return D_eff
    return None


def plot_msd(output_dir, lags_msd, msd, msd_std, v_A, N):
    """Plot Mean Squared Displacement analysis."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. MSD vs time (log-log) - shows power-law scaling behavior
    ax = axes[0]
    ax.errorbar(lags_msd, msd, yerr=msd_std, fmt='o-', markersize=4, capsize=2, 
                color='C0', label='Data')
    
    # Add reference slope lines (not fitted, just for visual comparison)
    if len(lags_msd) > 5:
        t_min, t_max = lags_msd[1], lags_msd[-1]
        t_ref = np.logspace(np.log10(t_min), np.log10(t_max), 50)
        
        # Position reference lines in middle of plot (geometric mean of MSD range)
        msd_mid = np.sqrt(msd[1] * msd[-1])
        t_mid = np.sqrt(t_min * t_max)
        
        # Ballistic: MSD ~ t^2
        ax.plot(t_ref, msd_mid * (t_ref/t_mid)**2, 'k--', alpha=0.4, linewidth=1.5, label='~t² (ballistic)')
        # Diffusive: MSD ~ t
        ax.plot(t_ref, msd_mid * (t_ref/t_mid)**1, 'k:', alpha=0.4, linewidth=1.5, label='~t (diffusive)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time lag Δt')
    ax.set_ylabel('MSD ⟨Δr²⟩')
    ax.set_title('Log-log scale (power-law behavior)')
    ax.legend(loc='lower right', fontsize=9)
    
    # 2. MSD vs time (linear) - for reading absolute values and D_eff
    ax = axes[1]
    ax.errorbar(lags_msd, msd, yerr=msd_std, fmt='o-', markersize=4, capsize=2, color='C0')
    ax.set_xlabel('Time lag Δt')
    ax.set_ylabel('MSD ⟨Δr²⟩')
    ax.set_title('Linear scale (diffusion coefficient)')
    ax.grid(True, alpha=0.3)
    
    # Compute and show diffusion coefficient
    D_eff = compute_effective_diffusion(lags_msd, msd)
    if D_eff is not None:
        # Draw the linear fit line (only in the fitted region)
        n_half = len(lags_msd) // 2
        slope = 4 * D_eff
        intercept = np.mean(msd[n_half:]) - slope * np.mean(lags_msd[n_half:])
        t_fit = np.array([lags_msd[n_half], lags_msd[-1]])
        ax.plot(t_fit, slope * t_fit + intercept, 'r-', alpha=0.7, linewidth=2, 
                label=f'Fit: D_eff = {D_eff:.4f}')
        ax.legend(loc='lower right', fontsize=9)
    
    plt.suptitle(f'Mean Squared Displacement (N={N}, v_A={v_A})', fontsize=14)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'msd.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if not _args.no_show:
        plt.show()
    plt.close()
    
    return D_eff


def plot_autocorrelations(output_dir, lags_v, v_autocorr, v_std,
                          lags_p, p_autocorr, p_std, v_A, N):
    """Plot velocity and polarization autocorrelation functions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Velocity autocorrelation
    ax = axes[0]
    ax.errorbar(lags_v, v_autocorr, yerr=v_std, fmt='o-', markersize=4, capsize=2, color='C0')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.2)
    ax.set_xlabel('Time lag Δt')
    ax.set_ylabel('C_v(Δt) = ⟨v(t)·v(t+Δt)⟩ / ⟨v²⟩')
    ax.set_title('Velocity Autocorrelation')
    ax.grid(True, alpha=0.3)
    
    # 2. Polarization autocorrelation
    ax = axes[1]
    ax.errorbar(lags_p, p_autocorr, yerr=p_std, fmt='o-', markersize=4, capsize=2, color='C0')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    tau_fit = None
    # Fit exponential decay to get persistence time
    if len(lags_p) > 3:
        # C_p(t) = exp(-t/tau)
        valid = p_autocorr > 0.1  # Only fit positive values above noise
        if np.sum(valid) > 3:
            try:
                slope, _ = np.polyfit(lags_p[valid], np.log(p_autocorr[valid]), 1)
                tau_fit = -1 / slope
                # Plot fit
                t_fit = np.linspace(lags_p[0], lags_p[-1], 100)
                ax.plot(t_fit, np.exp(-t_fit/tau_fit), 'r-', alpha=0.7, linewidth=2,
                        label=f'Fit: τ_p = {tau_fit:.0f}')
                ax.legend(loc='upper right', fontsize=9)
            except:
                pass
    
    ax.set_xlabel('Time lag Δt')
    ax.set_ylabel('C_p(Δt) = ⟨p(t)·p(t+Δt)⟩')
    ax.set_title('Polarization Autocorrelation (directional persistence)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    plt.suptitle(f'Autocorrelation Functions (N={N}, v_A={v_A})', fontsize=14)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'autocorrelations.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if not _args.no_show:
        plt.show()
    plt.close()
    
    return tau_fit


def plot_trajectories(data, output_dir, N, v_A, Lx=None, Ly=None):
    """Plot cell trajectories in 2D, splitting at periodic boundary crossings."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    cell_ids = np.unique(data[:, 1]).astype(int)
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(cell_ids), 20)))
    
    for i, cell_id in enumerate(cell_ids):
        mask = data[:, 1] == cell_id
        x = data[mask, 2]
        y = data[mask, 3]
        
        color = colors[i % len(colors)]
        
        if Lx is not None and Ly is not None:
            # Split trajectory at periodic boundary crossings
            # Draw each continuous segment separately
            segment_start = 0
            for j in range(1, len(x)):
                dx = abs(x[j] - x[j-1])
                dy = abs(y[j] - y[j-1])
                
                # If jump is more than half domain, it's a periodic wrap
                if dx > Lx / 2 or dy > Ly / 2:
                    # Draw segment up to the wrap
                    if j > segment_start:
                        ax.plot(x[segment_start:j], y[segment_start:j], '-', 
                               color=color, alpha=0.7, linewidth=1.5)
                    segment_start = j
            
            # Draw final segment
            if segment_start < len(x):
                ax.plot(x[segment_start:], y[segment_start:], '-', 
                       color=color, alpha=0.7, linewidth=1.5)
        else:
            # No domain info, just plot raw trajectory
            ax.plot(x, y, '-', color=color, alpha=0.7, linewidth=1.5)
        
        # Mark start and end positions
        ax.plot(x[0], y[0], 'o', color=color, markersize=8)
        ax.plot(x[-1], y[-1], 's', color=color, markersize=8)
    
    # Draw domain boundary if known
    if Lx is not None and Ly is not None:
        # Draw domain box
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), Lx, Ly, fill=False, edgecolor='gray', 
                         linestyle='--', linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        # Set limits to exactly match domain with small padding
        pad = min(Lx, Ly) * 0.02
        ax.set_xlim(-pad, Lx + pad)
        ax.set_ylim(-pad, Ly + pad)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Cell Trajectories (N={N}, v_A={v_A})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'trajectories.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if not _args.no_show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze trajectory data for MSD and correlations')
    parser.add_argument('output_dir', help='Directory containing trajectory.txt')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots (save only)')
    args = parser.parse_args()
    
    trajectory_file = Path(args.output_dir) / 'trajectory.txt'
    if not trajectory_file.exists():
        print(f"Error: {trajectory_file} not found")
        return
    
    print(f"Loading trajectory from: {trajectory_file}")
    data, v_A, N, Lx, Ly = load_trajectory(trajectory_file)
    print(f"Loaded {len(data)} data points for {N} cells, v_A={v_A}")
    if Lx is not None and Ly is not None:
        print(f"Domain size: {Lx} x {Ly}")
    
    times = np.unique(data[:, 0])
    print(f"Time range: {times[0]:.1f} to {times[-1]:.1f} ({len(times)} samples)")
    
    print("\nComputing MSD...")
    lags_msd, msd, msd_std = compute_msd(data, N, Lx, Ly)
    
    print("Computing velocity autocorrelation...")
    lags_v, v_autocorr, v_std = compute_velocity_autocorrelation(data, N)
    
    print("Computing polarization autocorrelation...")
    lags_p, p_autocorr, p_std = compute_polarization_autocorrelation(data, N)
    
    print("\nPlotting MSD...")
    D_eff = plot_msd(args.output_dir, lags_msd, msd, msd_std, v_A, N)
    
    print("Plotting autocorrelations...")
    tau_p = plot_autocorrelations(args.output_dir, lags_v, v_autocorr, v_std,
                                   lags_p, p_autocorr, p_std, v_A, N)
    
    print("Plotting trajectories...")
    plot_trajectories(data, args.output_dir, N, v_A, Lx, Ly)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)
    if D_eff is not None:
        print(f"Effective diffusion coefficient: D_eff = {D_eff:.6f}")
    
    if tau_p is not None:
        print(f"Polarization persistence time: τ_p = {tau_p:.1f}")
    
    # Expected D_eff for active Brownian particle: D_eff = v_A^2 * tau / 2
    if v_A and v_A > 0:
        tau_expected = 1e4  # From simulation parameters
        D_expected = v_A**2 * tau_expected / 2
        print(f"Expected D_eff (free ABP): v_A²τ/2 = {D_expected:.6f}")
        if tau_p is not None:
            print(f"Ratio τ_p / τ = {tau_p / tau_expected:.3f} (1.0 = free ABP)")


if __name__ == '__main__':
    main()
