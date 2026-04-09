"""Run two-cell adhesion sweep and analyze results.

Places 2 cells in a 400x400 box starting at d₀ ≈ 1.9R,
runs to t=3000 with Palmieri parameters (γ=1, dt=0.01), and
extracts equilibrium centroid distance for each J̃ value.
"""
import os, sys, subprocess, json, math, struct
import numpy as np

EXE = os.path.join(os.path.dirname(__file__), '..', 'build', 'bin', 'cell_sim.exe')
EXE = os.path.normpath(EXE)
BASE = os.path.join(os.environ['TEMP'], 'tc2_palmieri')
R = 49.0
L = 400
D0 = 93  # initial separation ≈ 1.898R

CASES = [
    ('J0',      0.0,   0.000),
    ('Jt0.125', 0.25,  0.125),
    ('Jt0.250', 0.50,  0.250),
    ('Jt0.375', 0.75,  0.375),
    ('Jt0.500', 1.00,  0.500),
    ('Jt0.625', 1.25,  0.625),
    ('Jt0.750', 1.50,  0.750),
]

def make_init_json(path):
    """Create init JSON with 2 cells centered, d = D0 px (≈1.9R)."""
    init = {
        "Nx": L, "Ny": L, "target_radius": R,
        "cells": [
            {"cx": L/2 - D0/2, "cy": L/2},
            {"cx": L/2 + D0/2, "cy": L/2},
        ]
    }
    with open(path, 'w') as f:
        json.dump(init, f)

def run_sim(label, J):
    outdir = os.path.join(BASE, label)
    if os.path.isdir(outdir):
        import shutil
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    init_json = os.path.join(BASE, 'init.json')
    make_init_json(init_json)

    cmd = [
        EXE,
        '-i', init_json,
        '-t', '3000',
        '--dt', '0.01',
        '--v-A', '0',
        '--save-interval', '300000',
        '--print-interval', '100000',
        '--trajectory-samples', '200',
        '-o', outdir,
    ]
    if J > 0:
        cmd += ['--adhesion', str(J)]

    print(f'  Running {label} (J={J:.4f}, J̃={J/(2*1.0):.3f})...', end=' ', flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f'FAILED: {result.stderr[-200:]}')
        return False
    print('done.')
    return True

def analyze(label):
    """Extract equilibrium d/R from trajectory."""
    outdir = os.path.join(BASE, label)
    traj = np.loadtxt(os.path.join(outdir, 'trajectory.txt'), comments='#')

    # Get last few time points
    times = sorted(set(traj[:, 0]))
    # Use last 10% of trajectory for equilibrium average
    n_eq = max(1, len(times) // 10)
    eq_times = times[-n_eq:]

    d_vals = []
    for t in eq_times:
        rows = traj[traj[:, 0] == t]
        if len(rows) < 2:
            continue
        x0, y0 = rows[0, 2], rows[0, 3]
        x1, y1 = rows[1, 2], rows[1, 3]
        dx = abs(x1 - x0); dx = min(dx, L - dx)
        dy = abs(y1 - y0); dy = min(dy, L - dy)
        d_vals.append(math.sqrt(dx**2 + dy**2))

    d_eq = np.mean(d_vals)
    d_std = np.std(d_vals) if len(d_vals) > 1 else 0.0
    return d_eq, d_std

def main():
    os.makedirs(BASE, exist_ok=True)

    print(f'Two-cell adhesion sweep (Palmieri γ=1, dt=0.01, t=3000)')
    print(f'  Box: {L}x{L}, R={R}, initial d/R ≈ {D0/R:.2f}')
    print()

    # Run all simulations
    for label, J, Jt in CASES:
        if not run_sim(label, J):
            sys.exit(1)

    # Analyze
    print()
    print(f'{"J̃":>6s} {"d_eq/R":>8s} {"±":>6s}')
    print('-' * 24)

    results = []
    for label, J, Jt in CASES:
        d_eq, d_std = analyze(label)
        print(f'{Jt:6.3f} {d_eq/R:8.4f} {d_std/R:6.4f}')
        results.append((Jt, d_eq/R, d_std/R))

    # Save results
    out_file = os.path.join(BASE, 'results.txt')
    with open(out_file, 'w') as f:
        f.write('# Jt  d_eq_over_R  d_std_over_R\n')
        for Jt, d, s in results:
            f.write(f'{Jt:.3f} {d:.6f} {s:.6f}\n')
    print(f'\nResults saved to {out_file}')

if __name__ == '__main__':
    main()
