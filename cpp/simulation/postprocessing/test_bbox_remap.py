"""Unit tests for bounding box remap correctness.

Tests the simulation's bbox resize/remap by:
1. Running with conditions that force bbox resizes
2. Checking field continuity (no discontinuities)
3. Comparing with a no-resize reference

Each test starts two identical simulations:
  (a) Normal run (bbox resizes enabled, the default)
  (b) Reference: would need bbox resizing disabled, but since we can't
      do that, we check for discontinuities in the normal run.

The discontinuity metric: for each row through the interface, compute
the second derivative d²φ/dx². A smooth field has d²φ/dx² varying
smoothly. A remap artifact shows as a spike in d²φ/dx².
"""
import os, sys, subprocess, struct, math, json
import numpy as np
from skimage.measure import find_contours

EXE = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                     '..', 'build', 'bin', 'cell_sim.exe'))
TMPDIR = os.path.join(os.environ['TEMP'], 'bbox_tests')
R = 49.0

def load_vtk_bin(path):
    with open(path, 'rb') as f:
        content = f.read()
    he = content.find(b'LOOKUP_TABLE default\n')
    header = content[:he].decode('ascii', errors='replace')
    for line in header.split('\n'):
        if 'DIMENSIONS' in line:
            nx, ny = int(line.split()[1]), int(line.split()[2])
    ds = he + len(b'LOOKUP_TABLE default\n')
    return np.array(struct.unpack('>' + 'f' * (nx * ny),
                    content[ds:ds + 4 * nx * ny])).reshape(ny, nx), nx, ny

def load_vtk_ascii(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    nx = ny = 0; ds = 0
    for i, line in enumerate(lines):
        if line.startswith('DIMENSIONS'):
            p = line.split(); nx, ny = int(p[1]), int(p[2])
        if line.startswith('LOOKUP_TABLE'):
            ds = i + 1; break
    vals = [float(lines[i].strip()) for i in range(ds, len(lines)) if lines[i].strip()]
    return np.array(vals).reshape(ny, nx), nx, ny

def run_sim(name, init_json_path, extra_args, t_end, save_interval):
    """Run a simulation and return the output directory."""
    outdir = os.path.join(TMPDIR, name)
    if os.path.isdir(outdir):
        import shutil
        shutil.rmtree(outdir)
    cmd = [
        EXE,
        '-i', init_json_path,
        '-t', str(t_end),
        '--dt', '0.01',
        '--v-A', '0',
        '--save-interval', str(save_interval),
        '--trajectory-samples', '200',
        '--save-individual-fields',
        '-o', outdir,
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f'  FAILED: {result.stderr[-300:]}')
        return None
    return outdir

def check_field_continuity(phi, label, threshold=0.15):
    """Check for discontinuities in phi by looking at the second derivative.
    
    Returns list of (row, col, severity) for each detected discontinuity.
    Severity = |d²φ/dx²| / expected_smooth_value.
    """
    anomalies = []
    
    # Check column-wise (horizontal) continuity
    d2_dx2 = np.zeros_like(phi)
    d2_dx2[:, 1:-1] = phi[:, 2:] + phi[:, :-2] - 2 * phi[:, 1:-1]
    
    # Check row-wise (vertical) continuity
    d2_dy2 = np.zeros_like(phi)
    d2_dy2[1:-1, :] = phi[2:, :] + phi[:-2, :] - 2 * phi[1:-1, :]
    
    # Only check in the interface region (0.01 < phi < 0.99)
    interface = (phi > 0.01) & (phi < 0.99)
    
    if not interface.any():
        return anomalies
    
    # A discontinuity shows as an anomalously high d² compared to neighbors
    for d2, direction in [(d2_dx2, 'x'), (d2_dy2, 'y')]:
        # Compare each point's d² to its neighbors' d²
        # Use a rolling median to define "expected"
        for row in range(2, phi.shape[0] - 2):
            iface_cols = np.where(interface[row, :])[0]
            if len(iface_cols) < 10:
                continue
            
            vals = d2[row, iface_cols]
            
            # Check for spikes: values that differ from moving median by > threshold * max
            window = 5
            for i in range(window, len(vals) - window):
                local_median = np.median(vals[i-window:i+window+1])
                local_range = np.max(np.abs(vals[i-window:i+window+1]))
                if local_range < 1e-6:
                    continue
                deviation = abs(vals[i] - local_median) / max(local_range, 1e-8)
                if deviation > threshold:
                    col = iface_cols[i]
                    anomalies.append((row, col, deviation, direction))
    
    return anomalies

def make_init_json(path, nx, ny, cells):
    init = {"Nx": nx, "Ny": ny, "target_radius": R, "cells": cells}
    with open(path, 'w') as f:
        json.dump(init, f)

# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Single cell, no adhesion — bbox should shrink as cell relaxes
# ═══════════════════════════════════════════════════════════════════════════
def test_single_cell_no_adhesion():
    """A single cell in a large box. Bbox should shrink.
    No discontinuities should appear."""
    print('\n=== Test 1: Single cell, no adhesion ===')
    init_path = os.path.join(TMPDIR, 'test1_init.json')
    os.makedirs(TMPDIR, exist_ok=True)
    make_init_json(init_path, 300, 300, [{"cx": 150, "cy": 150}])
    
    outdir = run_sim('test1_single', init_path, [], t_end=1000, save_interval=100000)
    if outdir is None:
        print('  SKIPPED (sim failed)')
        return False
    
    # Load final per-cell field
    fd = os.path.join(outdir, 'fields')
    frames = sorted(set(f.split('_')[1] for f in os.listdir(fd) if '_cell_000' in f))
    last = frames[-1]
    phi, nx, ny = load_vtk_ascii(os.path.join(fd, f'frame_{last}_cell_000.vtk'))
    
    anomalies = check_field_continuity(phi, 'single cell')
    if anomalies:
        print(f'  FAIL: {len(anomalies)} anomalies found')
        for r, c, sev, d in anomalies[:5]:
            print(f'    row={r}, col={c}, severity={sev:.2f}, dir={d}')
        return False
    else:
        print('  PASS: No discontinuities')
        return True

# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Two cells with adhesion — triggers bbox growth on contact side
# ═══════════════════════════════════════════════════════════════════════════
def test_two_cell_adhesion():
    """Two cells with J̃=0.75 adhesion, same as the artifact case.
    Check for discontinuities at multiple time points."""
    print('\n=== Test 2: Two cells with adhesion (J̃=0.75) ===')
    init_path = os.path.join(TMPDIR, 'test2_init.json')
    os.makedirs(TMPDIR, exist_ok=True)
    make_init_json(init_path, 400, 400,
                   [{"cx": 153.5, "cy": 200}, {"cx": 246.5, "cy": 200}])
    
    # Run with frequent saves to catch the artifact appearing
    outdir = run_sim('test2_adhesion', init_path,
                     ['--adhesion', '1.5'],
                     t_end=2000, save_interval=20000)
    if outdir is None:
        print('  SKIPPED (sim failed)')
        return False
    
    fd = os.path.join(outdir, 'fields')
    frames = sorted(set(f.split('_')[1] for f in os.listdir(fd) if '_cell_000' in f))
    
    all_pass = True
    for frame in frames:
        phi0, nx, ny = load_vtk_ascii(os.path.join(fd, f'frame_{frame}_cell_000.vtk'))
        step = int(frame)
        t_val = step * 0.01
        
        anomalies = check_field_continuity(phi0, f't={t_val:.0f}')
        
        # Also check specific col 145 ratio
        if phi0[249, 145] > 0.01 and phi0[249, 146] > 0.01:
            ratio = phi0[249, 146] / phi0[249, 145]
            is_artifact = ratio < 0.95
        else:
            ratio = 1.0
            is_artifact = False
        
        nzc = np.where(phi0.max(axis=0) > 1e-10)[0]
        bbox_str = f'[{nzc[0]},{nzc[-1]}]'
        
        status = 'FAIL' if (anomalies or is_artifact) else 'pass'
        print(f'  t={t_val:6.0f}: col145_ratio={ratio:.4f}, anomalies={len(anomalies):3d}, '
              f'bbox={bbox_str} [{status}]')
        
        if anomalies or is_artifact:
            all_pass = False
    
    if all_pass:
        print('  PASS: No discontinuities at any saved frame')
    else:
        print('  FAIL: Discontinuities detected')
    return all_pass

# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Single cell forced to move — maximum bbox resizing
# ═══════════════════════════════════════════════════════════════════════════
def test_single_cell_moving():
    """A single cell with motility that forces centroid drift → bbox remap.
    Check field continuity after the cell has moved significantly."""
    print('\n=== Test 3: Single cell with motility (forces bbox remaps) ===')
    init_path = os.path.join(TMPDIR, 'test3_init.json')
    os.makedirs(TMPDIR, exist_ok=True)
    make_init_json(init_path, 300, 300, [{"cx": 150, "cy": 150}])
    
    outdir = run_sim('test3_moving', init_path,
                     ['--v-A', '0.05', '--tau', '100000'],  # constant direction, fast
                     t_end=500, save_interval=50000)
    if outdir is None:
        print('  SKIPPED (sim failed)')
        return False
    
    fd = os.path.join(outdir, 'fields')
    frames = sorted(set(f.split('_')[1] for f in os.listdir(fd) if '_cell_000' in f))
    
    all_pass = True
    for frame in frames:
        phi, nx, ny = load_vtk_ascii(os.path.join(fd, f'frame_{frame}_cell_000.vtk'))
        step = int(frame)
        t_val = step * 0.01
        
        anomalies = check_field_continuity(phi, f't={t_val:.0f}')
        nzc = np.where(phi.max(axis=0) > 1e-10)[0]
        nzr = np.where(phi.max(axis=1) > 1e-10)[0]
        bbox_str = f'c[{nzc[0]},{nzc[-1]}] r[{nzr[0]},{nzr[-1]}]'
        
        status = 'FAIL' if anomalies else 'pass'
        print(f'  t={t_val:6.0f}: anomalies={len(anomalies):3d}, bbox={bbox_str} [{status}]')
        if anomalies:
            all_pass = False
            for r, c, sev, d in anomalies[:3]:
                print(f'    row={r}, col={c}, severity={sev:.2f}, dir={d}')
    
    if all_pass:
        print('  PASS')
    else:
        print('  FAIL')
    return all_pass

# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Two cells NO adhesion — control for Test 2
# ═══════════════════════════════════════════════════════════════════════════
def test_two_cell_no_adhesion():
    """Same geometry as Test 2 but J=0. Checks if the artifact is adhesion-specific."""
    print('\n=== Test 4: Two cells WITHOUT adhesion (control) ===')
    init_path = os.path.join(TMPDIR, 'test4_init.json')
    os.makedirs(TMPDIR, exist_ok=True)
    make_init_json(init_path, 400, 400,
                   [{"cx": 153.5, "cy": 200}, {"cx": 246.5, "cy": 200}])
    
    outdir = run_sim('test4_no_adhesion', init_path, [],
                     t_end=2000, save_interval=20000)
    if outdir is None:
        print('  SKIPPED (sim failed)')
        return False
    
    fd = os.path.join(outdir, 'fields')
    frames = sorted(set(f.split('_')[1] for f in os.listdir(fd) if '_cell_000' in f))
    
    all_pass = True
    for frame in frames:
        phi0, nx, ny = load_vtk_ascii(os.path.join(fd, f'frame_{frame}_cell_000.vtk'))
        step = int(frame)
        t_val = step * 0.01
        
        if phi0[249, 145] > 0.01 and phi0[249, 146] > 0.01:
            ratio = phi0[249, 146] / phi0[249, 145]
        else:
            ratio = 1.0
        
        anomalies = check_field_continuity(phi0, f't={t_val:.0f}')
        nzc = np.where(phi0.max(axis=0) > 1e-10)[0]
        
        status = 'FAIL' if (anomalies or ratio < 0.95) else 'pass'
        print(f'  t={t_val:6.0f}: col145_ratio={ratio:.4f}, anomalies={len(anomalies):3d}, '
              f'bbox=[{nzc[0]},{nzc[-1]}] [{status}]')
        if anomalies or ratio < 0.95:
            all_pass = False
    
    if all_pass:
        print('  PASS')
    else:
        print('  FAIL')
    return all_pass

# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Remap-specific — check that remapped phi matches original at global coords
# ═══════════════════════════════════════════════════════════════════════════
def test_remap_conservation():
    """Run a short sim, save phi before and after a bbox resize step,
    and verify the global-coordinate values match exactly."""
    print('\n=== Test 5: Remap data conservation (before/after resize) ===')
    init_path = os.path.join(TMPDIR, 'test5_init.json')
    os.makedirs(TMPDIR, exist_ok=True)
    make_init_json(init_path, 400, 400,
                   [{"cx": 153.5, "cy": 200}, {"cx": 246.5, "cy": 200}])
    
    # Run with very frequent saves to catch the exact remap frame
    outdir = run_sim('test5_remap', init_path,
                     ['--adhesion', '1.5'],
                     t_end=100, save_interval=1000)
    if outdir is None:
        print('  SKIPPED (sim failed)')
        return False
    
    fd = os.path.join(outdir, 'fields')
    frames = sorted(set(f.split('_')[1] for f in os.listdir(fd) if '_cell_000' in f))
    
    prev_phi = None
    prev_bbox = None
    all_pass = True
    
    for frame in frames:
        phi, nx, ny = load_vtk_ascii(os.path.join(fd, f'frame_{frame}_cell_000.vtk'))
        nzc = np.where(phi.max(axis=0) > 1e-10)[0]
        nzr = np.where(phi.max(axis=1) > 1e-10)[0]
        cur_bbox = (nzr[0], nzr[-1], nzc[0], nzc[-1])
        
        step = int(frame)
        t_val = step * 0.01
        
        if prev_bbox is not None and cur_bbox != prev_bbox:
            print(f'  t={t_val:.1f}: BBOX CHANGED {prev_bbox} → {cur_bbox}')
            # After a bbox change, the phi values at global coords within the
            # OVERLAP of old and new bbox should be identical (remap is lossless)
            # But they WON'T be, because ~100 timesteps elapsed between saves.
            # What we CAN check: is the field smooth?
            anomalies = check_field_continuity(phi, f'post-remap t={t_val:.0f}')
            if anomalies:
                print(f'    FAIL: {len(anomalies)} anomalies after bbox change')
                all_pass = False
            else:
                print(f'    pass: field smooth after bbox change')
        
        prev_phi = phi.copy()
        prev_bbox = cur_bbox
    
    if all_pass:
        print('  PASS')
    else:
        print('  FAIL')
    return all_pass


def main():
    os.makedirs(TMPDIR, exist_ok=True)
    print(f'Bbox remap unit tests')
    print(f'Executable: {EXE}')
    print(f'Output dir: {TMPDIR}')
    
    results = {}
    results['test1_single'] = test_single_cell_no_adhesion()
    results['test2_adhesion'] = test_two_cell_adhesion()
    results['test3_moving'] = test_single_cell_moving()
    results['test4_control'] = test_two_cell_no_adhesion()
    results['test5_remap'] = test_remap_conservation()
    
    print('\n' + '=' * 50)
    print('SUMMARY:')
    for name, passed in results.items():
        print(f'  {name}: {"PASS" if passed else "FAIL"}')
    
    n_pass = sum(results.values())
    n_total = len(results)
    print(f'\n{n_pass}/{n_total} passed')
    return 0 if n_pass == n_total else 1

if __name__ == '__main__':
    sys.exit(main())
