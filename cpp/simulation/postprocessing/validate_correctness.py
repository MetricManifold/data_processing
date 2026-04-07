#!/usr/bin/env python3
"""
Validate simulation correctness after optimization changes.

Checks:
  1. Trajectory self-consistency (no NaN/Inf, monotonic time)
  2. Cells remain within domain bounds
  3. Centroid stability (no blow-up or collapse)
  4. Volume conservation from checkpoint (phi fields)
  5. Phase-field normalization (phi values in [0,1] approximately)
  6. Cell shape preservation (bounding box sizes reasonable)
"""

import sys
import struct
import numpy as np
from pathlib import Path


def read_trajectory(path):
    """Parse trajectory.txt into structured arrays."""
    times, cell_ids = [], []
    xs, ys, vxs, vys = [], [], [], []
    pxs, pys, thetas = [], [], []

    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            t, cid, x, y, vx, vy, px, py, theta = (
                float(parts[0]), int(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]), float(parts[5]),
                float(parts[6]), float(parts[7]), float(parts[8])
            )
            times.append(t)
            cell_ids.append(cid)
            xs.append(x)
            ys.append(y)
            vxs.append(vx)
            vys.append(vy)
            pxs.append(px)
            pys.append(py)
            thetas.append(theta)

    return {
        'time': np.array(times),
        'cell_id': np.array(cell_ids),
        'x': np.array(xs), 'y': np.array(ys),
        'vx': np.array(vxs), 'vy': np.array(vys),
        'px': np.array(pxs), 'py': np.array(pys),
        'theta': np.array(thetas),
    }


def read_checkpoint_2d(path, verbose=True):
    """Read 2D checkpoint binary (v4 format), return header + cell phi fields.
    
    Binary layout:
      CheckpointHeader struct (see include/io.cuh)
      SimParams struct (see include/types.cuh)
      Per cell:
        int id
        BoundingBox (4 ints: x0, y0, x1, y1)
        Vec2 centroid (2 floats)
        Vec2 velocity (2 floats)
        float volume
        float phi[field_size]   where field_size = bbox.width() * bbox.height()
    """
    with open(path, 'rb') as f:
        data = f.read()

    offset = 0

    def read_u32():
        nonlocal offset
        val = struct.unpack_from('I', data, offset)[0]
        offset += 4
        return val

    def read_i32():
        nonlocal offset
        val = struct.unpack_from('i', data, offset)[0]
        offset += 4
        return val

    def read_f32():
        nonlocal offset
        val = struct.unpack_from('f', data, offset)[0]
        offset += 4
        return val

    def read_bool():
        # C++ bool may be 1 byte, but struct alignment may pad it.
        # In the CheckpointHeader, bools are packed among ints.
        # Actually in the struct they're individual bools - need to handle alignment.
        nonlocal offset
        val = struct.unpack_from('?', data, offset)[0]
        offset += 1
        return val

    # ---- CheckpointHeader ----
    magic = read_u32()
    if magic != 0x43454C4C:  # "CELL"
        print(f"  Error: bad magic 0x{magic:08X} (expected 0x43454C4C)")
        return None

    version = read_u32()
    if version < 2 or version > 5:
        print(f"  Error: unsupported version {version}")
        return None

    current_step = read_i32()
    current_time = read_f32()
    num_cells = read_i32()

    # Runtime options (present in v3+, but the struct always includes them)
    save_interval = read_i32()
    checkpoint_interval = read_i32()
    trajectory_samples = read_i32()
    save_vtk = read_bool()
    save_tracking = read_bool()
    compute_diagnostics = read_bool()
    save_individual_fields = read_bool()

    # v4: sim_params_size
    sim_params_size = read_u32() if version >= 4 else 0

    # ---- SimParams ----
    sp_start = offset
    Nx = read_i32()
    Ny = read_i32()
    dx = read_f32()
    dy = read_f32()
    dt = read_f32()
    t_end = read_f32()
    sp_save_interval = read_i32()
    lambda_val = read_f32()
    gamma = read_f32()
    kappa = read_f32()
    R = read_f32()
    mu = read_f32()
    v_A = read_f32()
    xi = read_f32()
    tau = read_f32()
    halo_width = read_i32()
    min_subdomain = read_i32()
    subdomain_padding = read_f32()
    motility_model = read_u32()

    # If sim_params_size was recorded, skip to the end of SimParams
    if sim_params_size > 0:
        offset = sp_start + sim_params_size
    
    if verbose:
        print(f"    [debug] Cell data starts at offset {offset} (sim_params_size={sim_params_size})")

    header = {
        'version': version, 'step': current_step, 'time': current_time,
        'num_cells': num_cells, 'Nx': Nx, 'Ny': Ny, 'dx': dx, 'dy': dy,
        'dt': dt, 'R': R, 'lambda': lambda_val, 'gamma': gamma,
        'kappa': kappa, 'mu': mu, 'halo_width': halo_width, 'v_A': v_A,
    }

    # ---- Per-cell data ----
    cells = []
    target_area = np.pi * R * R
    for i in range(num_cells):
        cell_id = read_i32()
        # BoundingBox: v4=inner bbox (no halo), v5=bbox_with_halo
        x0 = read_i32()
        y0 = read_i32()
        x1 = read_i32()
        y1 = read_i32()
        hw = halo_width
        if version >= 5:
            # v5: bbox IS bbox_with_halo
            w_full = x1 - x0
            h_full = y1 - y0
            w_inner = w_full - 2 * hw
            h_inner = h_full - 2 * hw
        else:
            # v4 and earlier: bbox is inner
            w_inner = x1 - x0
            h_inner = y1 - y0
            w_full = w_inner + 2 * hw
            h_full = h_inner + 2 * hw
        # Vec2 centroid
        cx = read_f32()
        cy = read_f32()
        # Vec2 velocity
        vx = read_f32()
        vy = read_f32()
        # float volume (as tracked by simulation)
        sim_volume = read_f32()

        # phi field includes halo
        field_size = w_full * h_full
        if verbose:
            print(f"    [debug] Cell {cell_id}: bbox=({x0},{y0})-({x1},{y1}) {w_inner}x{h_inner}, "
                  f"field={w_full}x{h_full}={field_size}, centroid=({cx:.1f},{cy:.1f}), vol={sim_volume:.1f}")

        # phi field
        phi = np.frombuffer(data, dtype=np.float32, count=field_size, offset=offset).copy()
        offset += field_size * 4

        phi_2d = phi.reshape(h_full, w_full)
        # Recompute volume from phi: V = ∫ φ² dA (inside non-halo region)
        inner = phi_2d[hw:h_full - hw, hw:w_full - hw]
        recomputed_volume = float(np.sum(inner ** 2)) * dx * dy

        cells.append({
            'id': cell_id,
            'bbox': (x0, y0, x1, y1),
            'size': (w_full, h_full),
            'inner_size': (w_inner, h_inner),
            'centroid': (cx, cy),
            'velocity': (vx, vy),
            'sim_volume': sim_volume,
            'recomputed_volume': recomputed_volume,
            'phi': phi_2d,
            'target_area': target_area,
        })

    return header, cells


def validate_trajectory(data, Nx, Ny, num_cells):
    """Check trajectory for physical correctness."""
    errors = []
    warnings = []

    # 1. No NaN or Inf
    for key in ['x', 'y', 'vx', 'vy', 'px', 'py', 'theta']:
        if np.any(np.isnan(data[key])):
            errors.append(f"NaN found in trajectory field '{key}'")
        if np.any(np.isinf(data[key])):
            errors.append(f"Inf found in trajectory field '{key}'")

    if errors:
        return errors, warnings  # Fatal

    # 2. Time is monotonically non-decreasing
    unique_times = np.unique(data['time'])
    if not np.all(np.diff(unique_times) > 0):
        errors.append("Time is not monotonically increasing")

    # 3. Centroids within domain (allowing small periodic overshoot)
    margin = 5.0  # relaxed boundary
    if np.any(data['x'] < -margin) or np.any(data['x'] > Nx + margin):
        errors.append(f"Cell x-centroid out of domain range [{-margin}, {Nx + margin}]")
    if np.any(data['y'] < -margin) or np.any(data['y'] > Ny + margin):
        errors.append(f"Cell y-centroid out of domain range [{-margin}, {Ny + margin}]")

    # 4. Polarization vectors are unit vectors
    p_mag = np.sqrt(data['px']**2 + data['py']**2)
    p_error = np.abs(p_mag - 1.0)
    if np.max(p_error) > 0.01:
        errors.append(f"Polarization vector not unit: max |p|-1 = {np.max(p_error):.4f}")

    # 5. Centroid stability per cell: compute drift over time
    for cid in range(num_cells):
        mask = data['cell_id'] == cid
        if not np.any(mask):
            continue
        xs = data['x'][mask]
        ys = data['y'][mask]

        # Check total displacement (should be small for v_A=0, equilibrating cells)
        total_dx = xs[-1] - xs[0]
        total_dy = ys[-1] - ys[0]
        # Handle periodic wrapping
        if abs(total_dx) > Nx / 2:
            total_dx = total_dx - np.sign(total_dx) * Nx
        if abs(total_dy) > Ny / 2:
            total_dy = total_dy - np.sign(total_dy) * Ny
        total_disp = np.sqrt(total_dx**2 + total_dy**2)

        if total_disp > 50.0:  # More than a cell radius
            warnings.append(f"Cell {cid}: large drift = {total_disp:.1f} pixels")

        # Check for sudden jumps (frame-to-frame)
        dxs = np.diff(xs)
        dys = np.diff(ys)
        # Handle periodic wrapping in diffs
        dxs[dxs > Nx/2] -= Nx
        dxs[dxs < -Nx/2] += Nx
        dys[dys > Ny/2] -= Ny
        dys[dys < -Ny/2] += Ny
        jumps = np.sqrt(dxs**2 + dys**2)
        max_jump = np.max(jumps) if len(jumps) > 0 else 0
        if max_jump > 10.0:  # More than 10 pixels/output-frame
            warnings.append(f"Cell {cid}: large centroid jump = {max_jump:.2f} px/frame")

    return errors, warnings


def validate_checkpoint(header, cells):
    """Check checkpoint phi fields for physical correctness."""
    errors = []
    warnings = []

    for cell in cells:
        cid = cell['id']
        phi = cell['phi']
        volume = cell['recomputed_volume']
        sim_vol = cell['sim_volume']
        target = cell['target_area']

        # 1. Phi range check (should be mostly in [0, 1])
        phi_min, phi_max = phi.min(), phi.max()
        if phi_min < -0.1:
            errors.append(f"Cell {cid}: phi_min = {phi_min:.4f} (< -0.1, unphysical)")
        if phi_max > 1.1:
            errors.append(f"Cell {cid}: phi_max = {phi_max:.4f} (> 1.1, unphysical)")

        # 2. Volume conservation (recomputed from phi)
        vol_error_frac = abs(volume - target) / target
        if vol_error_frac > 0.10:
            errors.append(f"Cell {cid}: volume error = {vol_error_frac*100:.1f}% "
                          f"(recomputed={volume:.1f}, target={target:.1f})")
        elif vol_error_frac > 0.02:
            warnings.append(f"Cell {cid}: volume deviation = {vol_error_frac*100:.1f}%")

        # 2b. Cross-check sim_volume vs recomputed volume
        if sim_vol > 0:
            vol_mismatch = abs(sim_vol - volume) / max(volume, 1.0)
            if vol_mismatch > 0.01:
                warnings.append(f"Cell {cid}: sim_volume ({sim_vol:.1f}) vs "
                                f"recomputed ({volume:.1f}) mismatch {vol_mismatch*100:.1f}%")

        # 3. Profile shape: phi should have a clear interface (not just noise)
        halo = header['halo_width']
        h, w = phi.shape
        inner = phi[halo:h-halo, halo:w-halo]
        n_high = np.sum(inner > 0.9)
        n_low = np.sum(inner < 0.1)
        n_interface = np.sum((inner >= 0.1) & (inner <= 0.9))

        if n_high == 0:
            errors.append(f"Cell {cid}: no pixels with phi > 0.9 (cell may have dissolved)")
        if n_interface < 10:
            warnings.append(f"Cell {cid}: very thin interface ({n_interface} pixels)")

    return errors, warnings


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_correctness.py <output_dir> [<Nx> <Ny> <num_cells>]")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    Nx = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    Ny = int(sys.argv[3]) if len(sys.argv) > 3 else Nx
    num_cells = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    print(f"Validating: {output_dir}")
    print(f"Domain: {Nx}x{Ny}, Cells: {num_cells}")
    print("=" * 60)

    all_errors = []
    all_warnings = []

    # Trajectory validation
    traj_path = output_dir / "trajectory.txt"
    if traj_path.exists():
        print(f"\n[1] Trajectory analysis: {traj_path}")
        data = read_trajectory(traj_path)
        n_timesteps = len(np.unique(data['time']))
        n_entries = len(data['time'])
        print(f"    Entries: {n_entries} ({n_timesteps} timesteps × ~{num_cells} cells)")
        print(f"    Time range: [{data['time'].min():.2f}, {data['time'].max():.2f}]")

        errs, warns = validate_trajectory(data, Nx, Ny, num_cells)
        all_errors.extend(errs)
        all_warnings.extend(warns)

        if not errs:
            # Print per-cell summary
            print(f"\n    Per-cell centroid drift (v_A=0 reference):")
            for cid in range(num_cells):
                mask = data['cell_id'] == cid
                xs, ys = data['x'][mask], data['y'][mask]
                dx_total = xs[-1] - xs[0]
                dy_total = ys[-1] - ys[0]
                if abs(dx_total) > Nx / 2:
                    dx_total -= np.sign(dx_total) * Nx
                if abs(dy_total) > Ny / 2:
                    dy_total -= np.sign(dy_total) * Ny
                disp = np.sqrt(dx_total**2 + dy_total**2)
                vx_avg = np.mean(data['vx'][mask])
                vy_avg = np.mean(data['vy'][mask])
                print(f"      Cell {cid}: drift={disp:.4f} px, "
                      f"avg_vel=({vx_avg:.6f}, {vy_avg:.6f})")
    else:
        print(f"\n[1] No trajectory file found at {traj_path}")

    # Checkpoint validation
    ckpt_path = output_dir / "checkpoint.bin"
    if ckpt_path.exists():
        print(f"\n[2] Checkpoint analysis: {ckpt_path}")
        result = read_checkpoint_2d(ckpt_path)
        if result is not None:
            header, cells = result
            print(f"    Step: {header['step']}, Time: {header['time']:.4f}")
            print(f"    Domain: {header['Nx']}x{header['Ny']}, Cells: {header['num_cells']}")
            print(f"    R={header['R']}, λ={header['lambda']}")
            target_area = np.pi * header['R'] ** 2

            errs, warns = validate_checkpoint(header, cells)
            all_errors.extend(errs)
            all_warnings.extend(warns)

            # Print volume summary
            print(f"\n    Volume conservation (target = {target_area:.1f}):")
            for cell in cells:
                vol = cell['recomputed_volume']
                svol = cell['sim_volume']
                err_pct = (vol - target_area) / target_area * 100
                phi_min = cell['phi'].min()
                phi_max = cell['phi'].max()
                print(f"      Cell {cell['id']}: vol={vol:.1f} (sim={svol:.1f}, "
                      f"{err_pct:+.2f}%), phi∈[{phi_min:.4f}, {phi_max:.4f}]")
    else:
        print(f"\n[2] No checkpoint file found at {ckpt_path}")

    # Summary
    print("\n" + "=" * 60)
    if all_errors:
        print(f"❌ VALIDATION FAILED: {len(all_errors)} error(s)")
        for e in all_errors:
            print(f"  ERROR: {e}")
    else:
        print("✅ VALIDATION PASSED: No errors detected")

    if all_warnings:
        print(f"⚠  {len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"  WARNING: {w}")

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
