#!/usr/bin/env python3
"""
Plot jamming transition: Diffusion coefficient D vs active velocity v_A.

Reads output from msd_calculator batch processing and creates publication-ready plots.

Usage:
    python plot_jamming_transition.py diffusion_results.txt
    python plot_jamming_transition.py diffusion_results.txt --output jamming.png
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_diffusion_results(filepath):
    """Load diffusion results from msd_calculator output.
    
    Expected format:
        # Diffusion coefficients from jamming study
        # v_A D D_stderr n_replicates
        0.004000 1.234567e-02 5.678901e-04 100
        ...
    
    Returns: v_A, D, D_err, n_replicates as numpy arrays
    """
    data = np.loadtxt(filepath, comments='#')
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    v_A = data[:, 0]
    D = data[:, 1]
    D_err = data[:, 2] if data.shape[1] > 2 else np.zeros_like(D)
    n_rep = data[:, 3].astype(int) if data.shape[1] > 3 else np.ones_like(D)
    
    return v_A, D, D_err, n_rep


def plot_jamming_transition(v_A, D, D_err, n_rep, output_path=None, title=None):
    """Create jamming transition plot: D vs v_A."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Plot 1: D vs v_A (linear) ---
    ax1 = axes[0]
    ax1.errorbar(v_A, D, yerr=D_err, fmt='o-', capsize=4, 
                 markersize=8, linewidth=2, color='#2E86AB')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel(r'Active velocity $v_A$', fontsize=12)
    ax1.set_ylabel(r'Diffusion coefficient $D$', fontsize=12)
    ax1.set_title('Jamming Transition', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Shade jammed region (D ≈ 0)
    D_threshold = max(D) * 0.05  # 5% of max as threshold
    jammed_mask = D < D_threshold
    if any(jammed_mask):
        v_A_crit_idx = np.where(~jammed_mask)[0]
        if len(v_A_crit_idx) > 0:
            v_A_crit = v_A[v_A_crit_idx[0]]
            ax1.axvspan(v_A.min(), v_A_crit, alpha=0.1, color='blue', label='Jammed')
            ax1.axvline(x=v_A_crit, color='red', linestyle=':', alpha=0.7, 
                       label=f'$v_A^c \\approx {v_A_crit:.4f}$')
            ax1.legend(fontsize=10)
    
    # --- Plot 2: D vs v_A (log-log) ---
    ax2 = axes[1]
    # Only plot positive D values on log scale
    pos_mask = D > 0
    if any(pos_mask):
        ax2.errorbar(v_A[pos_mask], D[pos_mask], yerr=D_err[pos_mask], 
                    fmt='o-', capsize=4, markersize=8, linewidth=2, color='#2E86AB')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'Active velocity $v_A$', fontsize=12)
        ax2.set_ylabel(r'Diffusion coefficient $D$', fontsize=12)
        ax2.set_title('Jamming Transition (log-log)', fontsize=14)
        ax2.grid(True, alpha=0.3, which='both')
        
        # Fit power law in unjammed region: D ~ (v_A - v_A_c)^alpha
        # For now, just show reference lines
        v_ref = np.linspace(v_A[pos_mask].min(), v_A[pos_mask].max(), 100)
        # D ~ v_A^2 reference line
        D_ref = D[pos_mask].max() * (v_ref / v_A[pos_mask].max())**2
        ax2.plot(v_ref, D_ref, 'k--', alpha=0.4, label=r'$D \propto v_A^2$')
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'All D ≤ 0\n(fully jammed)', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=14)
    
    # Add replicate counts as annotation
    info_text = f"Replicates per point: {n_rep[0]}"
    if not all(n_rep == n_rep[0]):
        info_text = f"Replicates: {n_rep.min()}-{n_rep.max()}"
    fig.text(0.99, 0.01, info_text, ha='right', va='bottom', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def print_summary(v_A, D, D_err, n_rep):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("JAMMING TRANSITION SUMMARY")
    print("="*60)
    print(f"{'v_A':>10} {'D':>14} {'D_err':>12} {'n_rep':>8}")
    print("-"*60)
    for i in range(len(v_A)):
        status = "JAMMED" if D[i] <= 0 else ""
        print(f"{v_A[i]:>10.5f} {D[i]:>14.6e} {D_err[i]:>12.6e} {n_rep[i]:>8d}  {status}")
    print("-"*60)
    
    # Find critical velocity
    D_threshold = max(max(D), 1e-10) * 0.05
    jammed = D < D_threshold
    if any(jammed) and any(~jammed):
        # Transition point is between last jammed and first unjammed
        last_jammed = np.where(jammed)[0][-1]
        first_unjammed = np.where(~jammed)[0][0]
        v_A_crit = (v_A[last_jammed] + v_A[first_unjammed]) / 2
        print(f"\nEstimated critical velocity: v_A^c ≈ {v_A_crit:.5f}")
        print(f"  (between v_A={v_A[last_jammed]:.5f} and v_A={v_A[first_unjammed]:.5f})")
    elif all(jammed):
        print(f"\nAll velocities show D ≈ 0 (fully jammed)")
        print(f"Try higher v_A values")
    else:
        print(f"\nNo jammed phase detected (all D > 0)")
        print(f"Try lower v_A values to find transition")


def main():
    parser = argparse.ArgumentParser(
        description='Plot jamming transition from MSD calculator results')
    parser.add_argument('input_file', help='diffusion_results.txt from msd_calculator')
    parser.add_argument('-o', '--output', help='Output image path (default: jamming_transition.png)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1
    
    print(f"Loading: {input_path}")
    v_A, D, D_err, n_rep = load_diffusion_results(input_path)
    print(f"Loaded {len(v_A)} velocity points")
    
    print_summary(v_A, D, D_err, n_rep)
    
    output_path = args.output or input_path.parent / 'jamming_transition.png'
    plot_jamming_transition(v_A, D, D_err, n_rep, output_path)
    
    if not args.no_show:
        plt.show()
    
    return 0


if __name__ == '__main__':
    exit(main())
