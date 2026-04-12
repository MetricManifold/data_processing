#!/usr/bin/env python3
"""
Plot energy vs time from equilibration analysis.

Usage:
    python plot_energy.py energy_results.txt
    python plot_energy.py energy_phi85.txt energy_phi89.txt  # Compare
    python plot_energy.py energy_results.txt --output energy.png
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path


def load_energy_data(filename, t_max=None):
    """Load energy data from analyzer output."""
    times = []
    energies = []
    errors = []
    counts = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                t = float(parts[0])
                if t_max is not None and t > t_max:
                    continue
                times.append(t)
                energies.append(float(parts[1]))
                errors.append(float(parts[2]))
                if len(parts) >= 4:
                    counts.append(int(parts[3]))
                else:
                    counts.append(1)
    
    return np.array(times), np.array(energies), np.array(errors), np.array(counts)


def plot_energy(files, output=None, no_show=False, t_max=None):
    """Create energy vs time plot with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(files)))
    
    for i, filename in enumerate(files):
        path = Path(filename)
        label = path.stem
        
        # Make label more readable
        if 'phi89' in label:
            label = 'φ = 89% (L=1562)'
        elif 'phi85' in label:
            label = 'φ = 85% (L=1600)'
        
        times, energies, errors, counts = load_energy_data(filename, t_max)
        
        print(f"\n{filename}:")
        print(f"  Time range: {times[0]:.1f} to {times[-1]:.1f}")
        print(f"  Initial KE: {energies[0]:.6e}")
        print(f"  Final KE: {energies[-1]:.6e}")
        print(f"  Decay ratio: {energies[-1]/energies[0]:.4e}")
        print(f"  Data points: {len(times)}")
        print(f"  Replicas: {counts[0]}")
        
        # Linear plot (left)
        ax1 = axes[0]
        ax1.errorbar(times, energies, yerr=errors, 
                     fmt='-', color=colors[i], label=label,
                     alpha=0.8, errorevery=max(1, len(times)//50),
                     capsize=2, linewidth=1.5)
        
        # Log plot (right) - for seeing decay
        ax2 = axes[1]
        # Filter out zeros/negatives for log plot
        mask = energies > 0
        ax2.errorbar(times[mask], energies[mask], yerr=errors[mask],
                     fmt='-', color=colors[i], label=label,
                     alpha=0.8, errorevery=max(1, len(times)//50),
                     capsize=2, linewidth=1.5)
    
    # Format linear plot
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Kinetic Energy', fontsize=12)
    axes[0].set_title('Equilibration: Energy vs Time', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
    
    # Format log plot  
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Kinetic Energy (log scale)', fontsize=12)
    axes[1].set_title('Equilibration: Energy Decay', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {output}")
    
    if not no_show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot equilibration energy vs time')
    parser.add_argument('files', nargs='+', 
                        help='Energy data file(s) from energy_analyzer')
    parser.add_argument('-o', '--output', 
                        help='Output image file (e.g., energy.png)')
    parser.add_argument('--no-show', action='store_true',
                        help='Save plot without showing')
    parser.add_argument('--t-max', type=float, default=None,
                        help='Maximum time to plot (filters out later times)')
    
    args = parser.parse_args()
    
    # Validate files exist
    for f in args.files:
        if not Path(f).exists():
            print(f"ERROR: File not found: {f}")
            sys.exit(1)
    
    plot_energy(args.files, args.output, args.no_show, args.t_max)


if __name__ == '__main__':
    main()
