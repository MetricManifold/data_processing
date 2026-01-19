"""Analyze MSD from 3D checkpoint files to verify motility is working."""
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'postprocessing')
from visualize_3d import read_checkpoint_3d

def load_trajectories(output_dir):
    """Load cell centroids from all checkpoints."""
    checkpoint_files = sorted(glob.glob(os.path.join(output_dir, 'checkpoint_3d_*.bin')))
    
    times = []
    all_centroids = []  # [time_idx][cell_idx] -> (x, y, z)
    
    for f in checkpoint_files:
        params, cells = read_checkpoint_3d(f)
        times.append(params['time'])
        
        # Sort cells by ID for consistent ordering
        cells.sort(key=lambda c: c['id'])
        centroids = np.array([c['centroid'] for c in cells])
        all_centroids.append(centroids)
    
    return np.array(times), np.array(all_centroids)  # shape: (n_times, n_cells, 3)

def compute_msd(times, centroids, Nx, Ny, Nz):
    """Compute MSD with periodic boundary unwrapping."""
    n_times, n_cells, _ = centroids.shape
    
    # Unwrap periodic boundaries
    unwrapped = np.zeros_like(centroids)
    unwrapped[0] = centroids[0]
    
    for t in range(1, n_times):
        delta = centroids[t] - centroids[t-1]
        # Unwrap: if jump > half domain, it crossed boundary
        delta[:, 0] = np.where(delta[:, 0] > Nx/2, delta[:, 0] - Nx, delta[:, 0])
        delta[:, 0] = np.where(delta[:, 0] < -Nx/2, delta[:, 0] + Nx, delta[:, 0])
        delta[:, 1] = np.where(delta[:, 1] > Ny/2, delta[:, 1] - Ny, delta[:, 1])
        delta[:, 1] = np.where(delta[:, 1] < -Ny/2, delta[:, 1] + Ny, delta[:, 1])
        delta[:, 2] = np.where(delta[:, 2] > Nz/2, delta[:, 2] - Nz, delta[:, 2])
        delta[:, 2] = np.where(delta[:, 2] < -Nz/2, delta[:, 2] + Nz, delta[:, 2])
        unwrapped[t] = unwrapped[t-1] + delta
    
    # Compute MSD for different lag times
    max_lag = n_times // 2
    lag_times = []
    msd_values = []
    
    for lag in range(1, max_lag):
        displacements = unwrapped[lag:] - unwrapped[:-lag]  # (n_times - lag, n_cells, 3)
        squared_disp = np.sum(displacements**2, axis=2)  # (n_times - lag, n_cells)
        msd = np.mean(squared_disp)  # Average over cells and time origins
        
        dt = times[lag] - times[0]
        lag_times.append(dt)
        msd_values.append(msd)
    
    return np.array(lag_times), np.array(msd_values)

def main():
    output_dir = 'agent_test_runs/motility_test_3d'
    
    print("Loading trajectories...")
    times, centroids = load_trajectories(output_dir)
    print(f"Loaded {len(times)} time points, {centroids.shape[1]} cells")
    print(f"Time range: {times[0]:.2f} to {times[-1]:.2f}")
    
    # Get domain size
    params, _ = read_checkpoint_3d(glob.glob(os.path.join(output_dir, 'checkpoint_3d_*.bin'))[0])
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    
    print("\nComputing MSD...")
    lag_times, msd = compute_msd(times, centroids, Nx, Ny, Nz)
    
    # Theoretical prediction for Run-and-Tumble in 3D:
    # Short time (t << tau): MSD = v_A^2 * t^2 (ballistic)
    # Long time (t >> tau): MSD = 6 * D_eff * t, where D_eff = v_A^2 * tau / 3
    v_A = 0.5
    tau = 20.0
    D_eff_theory = v_A**2 * tau / 3
    
    print(f"\n=== MSD Analysis ===")
    print(f"Parameters: v_A = {v_A}, tau = {tau}")
    print(f"Theoretical D_eff = v_A² τ / 3 = {D_eff_theory:.4f}")
    
    # Fit diffusion coefficient from long-time behavior (t > tau)
    long_time_mask = lag_times > 2 * tau
    if np.sum(long_time_mask) > 10:
        # MSD = 6 * D * t for 3D
        slope, intercept = np.polyfit(lag_times[long_time_mask], msd[long_time_mask], 1)
        D_eff_measured = slope / 6
        print(f"Measured D_eff (from slope/6): {D_eff_measured:.4f}")
        print(f"Ratio measured/theory: {D_eff_measured / D_eff_theory:.2f}")
    else:
        D_eff_measured = None
        print("Not enough long-time data for diffusion fit")
    
    # Check ballistic regime (t << tau)
    short_time_mask = lag_times < tau / 2
    if np.sum(short_time_mask) > 5:
        # MSD ~ v_A^2 * t^2
        t_short = lag_times[short_time_mask]
        msd_short = msd[short_time_mask]
        # Fit log(MSD) vs log(t) to get exponent
        log_t = np.log(t_short[t_short > 0])
        log_msd = np.log(msd_short[t_short > 0])
        slope_log, _ = np.polyfit(log_t, log_msd, 1)
        print(f"\nShort-time exponent (MSD ~ t^n): n = {slope_log:.2f}")
        print(f"  (Expected: n=2 for ballistic, n=1 for diffusive)")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear plot
    ax1.plot(lag_times, msd, 'b-', label='Measured MSD')
    ax1.plot(lag_times, 6 * D_eff_theory * lag_times, 'r--', 
             label=f'Theory: 6Dt (D={D_eff_theory:.3f})')
    if D_eff_measured:
        ax1.plot(lag_times, 6 * D_eff_measured * lag_times, 'g:', 
                 label=f'Fit: 6Dt (D={D_eff_measured:.3f})')
    ax1.axvline(tau, color='gray', linestyle=':', alpha=0.5, label=f'τ = {tau}')
    ax1.set_xlabel('Lag time t')
    ax1.set_ylabel('MSD')
    ax1.set_title('Mean Squared Displacement (3D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-log plot
    ax2.loglog(lag_times, msd, 'b-', label='Measured MSD')
    ax2.loglog(lag_times, v_A**2 * lag_times**2, 'r--', alpha=0.5,
               label=f'Ballistic: v²t² (v={v_A})')
    ax2.loglog(lag_times, 6 * D_eff_theory * lag_times, 'g--', alpha=0.5,
               label=f'Diffusive: 6Dt (D={D_eff_theory:.3f})')
    ax2.axvline(tau, color='gray', linestyle=':', alpha=0.5, label=f'τ = {tau}')
    ax2.set_xlabel('Lag time t')
    ax2.set_ylabel('MSD')
    ax2.set_title('MSD (log-log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'msd_analysis.png'), dpi=150)
    print(f"\nPlot saved to: {output_dir}/msd_analysis.png")
    
    # Summary
    print("\n=== SUMMARY ===")
    if D_eff_measured and D_eff_measured > 0.01:
        print("✓ MOTILITY IS WORKING: Cells show significant diffusion")
        print(f"  D_measured = {D_eff_measured:.4f}")
        print(f"  D_theory   = {D_eff_theory:.4f}")
    else:
        print("✗ WARNING: Cells may not be moving (D ≈ 0)")

if __name__ == '__main__':
    main()
