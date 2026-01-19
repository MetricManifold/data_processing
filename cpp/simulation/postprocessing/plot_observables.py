#!/usr/bin/env python3
"""Plot diagnostic observables from simulation output."""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_observables(csv_path: str, output_dir: str = None):
    """Plot all observable quantities from the CSV file."""
    df = pd.read_csv(csv_path, comment='#')
    
    # Parse header to get column names (handle comment line)
    with open(csv_path) as f:
        header = f.readline().strip()
        if header.startswith('#'):
            header = header[1:].strip()
        cols = header.split(',')
    df.columns = cols
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # 1. Energy components
    ax = axes[0, 0]
    ax.plot(df['time'], df['E_grad'], label='E_gradient', alpha=0.8)
    ax.plot(df['time'], df['E_bulk'], label='E_bulk', alpha=0.8)
    ax.plot(df['time'], df['E_int'], label='E_interaction', alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Components')
    ax.legend(fontsize=8)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # 2. Total energy
    ax = axes[0, 1]
    ax.plot(df['time'], df['E_total'], 'k-', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Total Energy')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # 3. Stress components
    ax = axes[0, 2]
    ax.plot(df['time'], df['sigma_xx'], label='σ_xx', alpha=0.8)
    ax.plot(df['time'], df['sigma_yy'], label='σ_yy', alpha=0.8)
    ax.plot(df['time'], df['sigma_xy'], label='σ_xy', alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Stress')
    ax.set_title('Stress Tensor Components')
    ax.legend(fontsize=8)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # 4. Pressure
    ax = axes[1, 0]
    ax.plot(df['time'], df['pressure'], 'b-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Pressure')
    ax.set_title('Pressure (negative = compression)')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # 5. Mean coordination number
    ax = axes[1, 1]
    ax.plot(df['time'], df['z_mean'], 'g-', linewidth=1.5)
    ax.fill_between(df['time'], 
                    df['z_mean'] - df['z_std'], 
                    df['z_mean'] + df['z_std'],
                    alpha=0.3, color='g', label='±1 std')
    ax.set_xlabel('Time')
    ax.set_ylabel('Coordination z')
    ax.set_title('Mean Coordination Number')
    ax.legend(fontsize=8)
    
    # 6. Coordination distribution
    ax = axes[1, 2]
    ax.plot(df['time'], df['z_min'], 'r-', label='z_min', alpha=0.7)
    ax.plot(df['time'], df['z_max'], 'b-', label='z_max', alpha=0.7)
    ax.plot(df['time'], df['z_mean'], 'k-', label='z_mean', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Coordination z')
    ax.set_title('Coordination Range')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        out_path = Path(output_dir) / 'observables_plot.png'
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n=== Observable Summary ===")
    print(f"Time range: {df['time'].iloc[0]:.2f} - {df['time'].iloc[-1]:.2f}")
    print(f"Data points: {len(df)}")
    print(f"\nEnergy:")
    print(f"  E_total: {df['E_total'].mean():.2f} ± {df['E_total'].std():.2f}")
    print(f"  E_grad:  {df['E_grad'].mean():.2f} ± {df['E_grad'].std():.2f}")
    print(f"  E_bulk:  {df['E_bulk'].mean():.2f} ± {df['E_bulk'].std():.2f}")
    print(f"  E_int:   {df['E_int'].mean():.2f} ± {df['E_int'].std():.2f}")
    print(f"\nStress:")
    print(f"  σ_xx: {df['sigma_xx'].mean():.2f} ± {df['sigma_xx'].std():.2f}")
    print(f"  σ_yy: {df['sigma_yy'].mean():.2f} ± {df['sigma_yy'].std():.2f}")
    print(f"  σ_xy: {df['sigma_xy'].mean():.2f} ± {df['sigma_xy'].std():.2f}")
    print(f"\nPressure: {df['pressure'].mean():.2f} ± {df['pressure'].std():.2f}")
    print(f"\nCoordination:")
    print(f"  z_mean: {df['z_mean'].mean():.3f} ± {df['z_mean'].std():.4f}")
    print(f"  z_range: [{df['z_min'].min()}, {df['z_max'].max()}]")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default to local observables.csv
        csv_path = Path(__file__).parent / 'observables.csv'
    else:
        csv_path = sys.argv[1]
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_observables(csv_path, output_dir)
