#!/usr/bin/env python3
"""
Validation script for 3D cell simulation checkpoints.
Checks volume conservation, cell shapes, and physics consistency.
"""

import struct
import numpy as np
import sys
import os
from pathlib import Path


def read_checkpoint_3d(filepath):
    """Read a 3D checkpoint binary file and extract cell data."""
    cells = []
    
    with open(filepath, 'rb') as f:
        # Read header
        # Format from io3d.cu:
        #   Magic (4 bytes): "CS3D"
        #   Version (4 bytes): int
        #   Step (4 bytes): int  
        #   Time (4 bytes): float
        #   Num cells (4 bytes): int
        #   SimParams3D struct (variable size)
        
        magic = f.read(4)
        if magic != b'CS3D':  # "CS3D" magic
            raise ValueError(f"Invalid magic: {magic}, expected b'CS3D'")
        
        version = struct.unpack('i', f.read(4))[0]
        step = struct.unpack('i', f.read(4))[0]
        time = struct.unpack('f', f.read(4))[0]
        num_cells = struct.unpack('i', f.read(4))[0]
        
        # Read SimParams3D struct
        # Layout from types3d.cuh:
        #   int Nx, Ny, Nz (12 bytes)
        #   float dx, dy, dz (12 bytes) 
        #   float dt, t_end (8 bytes)
        #   int save_interval (4 bytes)
        #   float lambda, gamma (8 bytes)
        #   float kappa, target_radius, mu (12 bytes)
        #   float v_A, xi, tau (12 bytes)
        #   int halo_width, min_subdomain_size (8 bytes)
        #   float subdomain_padding (4 bytes)
        #   int motility_model (4 bytes)
        # Total: 84 bytes
        
        Nx, Ny, Nz = struct.unpack('3i', f.read(12))
        dx, dy, dz = struct.unpack('3f', f.read(12))
        dt, t_end = struct.unpack('2f', f.read(8))
        save_interval = struct.unpack('i', f.read(4))[0]
        lambda_, gamma = struct.unpack('2f', f.read(8))
        kappa, target_radius, mu = struct.unpack('3f', f.read(12))
        v_A, xi, tau = struct.unpack('3f', f.read(12))
        halo_width, min_subdomain_size = struct.unpack('2i', f.read(8))
        subdomain_padding = struct.unpack('f', f.read(4))[0]
        motility_model = struct.unpack('i', f.read(4))[0]
        
        params = {
            'version': version,
            'step': step,
            'time': time,
            'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
            'dx': dx, 'dy': dy, 'dz': dz,
            'dt': dt,
            't_end': t_end,
            'lambda': lambda_,
            'gamma': gamma,
            'kappa': kappa,
            'target_radius': target_radius,
            'mu': mu,
            'v_A': v_A,
            'xi': xi,
            'tau': tau,
            'subdomain_padding': subdomain_padding,
            'halo_width': halo_width,
            'num_cells': num_cells
        }
        
        target_volume = (4.0/3.0) * np.pi * target_radius**3
        params['target_volume'] = target_volume
        
        # Read each cell
        # Per-cell format from io3d.cu:
        #   int id (4 bytes)
        #   BoundingBox3D bbox (6 ints = 24 bytes)
        #   BoundingBox3D bbox_with_halo (6 ints = 24 bytes)
        #   Vec3 centroid (3 floats = 12 bytes)
        #   float volume (4 bytes)
        #   float theta (4 bytes)
        #   float phi_pol (4 bytes)
        #   Vec3 polarization (3 floats = 12 bytes)
        #   Vec3 velocity (3 floats = 12 bytes)
        #   phi data: width * height * depth * float
        
        for i in range(num_cells):
            cell_id = struct.unpack('i', f.read(4))[0]
            
            # Read bounding boxes (each is 6 ints)
            bbox = struct.unpack('6i', f.read(24))  # x0,y0,z0,x1,y1,z1
            bbox_halo = struct.unpack('6i', f.read(24))
            
            # Vec3 centroid
            centroid = struct.unpack('3f', f.read(12))
            
            # float volume, theta, phi_pol
            volume = struct.unpack('f', f.read(4))[0]
            theta = struct.unpack('f', f.read(4))[0]
            phi_pol = struct.unpack('f', f.read(4))[0]
            
            # Vec3 polarization, velocity
            polarization = struct.unpack('3f', f.read(12))
            velocity = struct.unpack('3f', f.read(12))
            
            # Calculate field dimensions from bbox_with_halo
            width = bbox_halo[3] - bbox_halo[0]
            height = bbox_halo[4] - bbox_halo[1]
            depth = bbox_halo[5] - bbox_halo[2]
            field_size = width * height * depth
            
            # Read field data
            phi_data = np.frombuffer(f.read(field_size * 4), dtype=np.float32)
            
            # Compute actual volume from field
            width = bbox_halo[3] - bbox_halo[0]
            height = bbox_halo[4] - bbox_halo[1]
            depth = bbox_halo[5] - bbox_halo[2]
            
            phi_3d = phi_data.reshape((depth, height, width))
            actual_volume = np.sum(phi_3d**2) * dx * dy * dz
            
            # Exclude halo from volume calculation
            halo = halo_width
            if halo > 0 and depth > 2*halo and height > 2*halo and width > 2*halo:
                phi_interior = phi_3d[halo:-halo, halo:-halo, halo:-halo]
                actual_volume_interior = np.sum(phi_interior**2) * dx * dy * dz
            else:
                actual_volume_interior = actual_volume
            
            cells.append({
                'id': cell_id,
                'bbox': bbox,
                'bbox_halo': bbox_halo,
                'stored_volume': volume,
                'computed_volume': actual_volume_interior,
                'centroid': centroid,
                'velocity': velocity,
                'polarization': polarization,
                'theta': theta,
                'phi_pol': phi_pol,
                'field_size': field_size,
                'subdomain_dims': (width, height, depth),
                'phi_max': np.max(phi_data),
                'phi_min': np.min(phi_data),
                'phi_mean': np.mean(phi_data[phi_data > 0.1]) if np.any(phi_data > 0.1) else 0
            })
    
    return params, cells


def validate_checkpoint(filepath):
    """Validate a 3D checkpoint file."""
    print(f"\n{'='*60}")
    print(f"Validating: {filepath}")
    print(f"{'='*60}")
    
    try:
        params, cells = read_checkpoint_3d(filepath)
    except Exception as e:
        print(f"ERROR: Failed to read checkpoint: {e}")
        return False
    
    print(f"\nSimulation Parameters:")
    print(f"  Step: {params['step']}, Time: {params['time']:.4f}")
    print(f"  Domain: {params['Nx']} x {params['Ny']} x {params['Nz']}")
    print(f"  Target radius: R = {params['target_radius']:.1f}")
    print(f"  Target volume: V₀ = {params['target_volume']:.1f}")
    print(f"  Number of cells: {params['num_cells']}")
    print(f"  λ = {params['lambda']:.1f}, κ = {params['kappa']:.1f}, μ = {params['mu']:.3f}")
    
    # Validation checks
    all_valid = True
    
    # Check 1: Volume conservation
    print(f"\n--- Volume Conservation ---")
    volumes = [c['computed_volume'] for c in cells]
    target_vol = params['target_volume']
    
    vol_mean = np.mean(volumes)
    vol_std = np.std(volumes)
    vol_min = np.min(volumes)
    vol_max = np.max(volumes)
    
    vol_error_pct = 100 * (vol_mean - target_vol) / target_vol
    vol_spread_pct = 100 * vol_std / target_vol
    
    print(f"  Target volume: {target_vol:.1f}")
    print(f"  Mean volume:   {vol_mean:.1f} ({vol_error_pct:+.2f}% from target)")
    print(f"  Std dev:       {vol_std:.1f} ({vol_spread_pct:.2f}% of target)")
    print(f"  Range:         [{vol_min:.1f}, {vol_max:.1f}]")
    
    if abs(vol_error_pct) > 10:
        print(f"  ⚠️  WARNING: Mean volume deviates >10% from target")
        all_valid = False
    elif abs(vol_error_pct) > 5:
        print(f"  ⚠️  CAUTION: Mean volume deviates >5% from target")
    else:
        print(f"  ✓ Volume conservation looks good")
    
    # Check 2: Field values (φ should be in [0,1])
    print(f"\n--- Field Value Ranges ---")
    phi_maxes = [c['phi_max'] for c in cells]
    phi_mins = [c['phi_min'] for c in cells]
    
    global_max = max(phi_maxes)
    global_min = min(phi_mins)
    
    print(f"  φ_max across all cells: {global_max:.4f}")
    print(f"  φ_min across all cells: {global_min:.4f}")
    
    if global_max > 1.1:
        print(f"  ⚠️  WARNING: φ exceeds 1.0 significantly")
        all_valid = False
    elif global_max > 1.01:
        print(f"  ⚠️  CAUTION: φ slightly exceeds 1.0")
    else:
        print(f"  ✓ φ values in expected range")
    
    if global_min < -0.1:
        print(f"  ⚠️  WARNING: φ has negative values")
        all_valid = False
    
    # Check 3: Cell centroids within domain
    print(f"\n--- Centroid Positions ---")
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    out_of_bounds = 0
    
    for c in cells:
        cx, cy, cz = c['centroid']
        if cx < 0 or cx >= Nx or cy < 0 or cy >= Ny or cz < 0 or cz >= Nz:
            out_of_bounds += 1
    
    if out_of_bounds > 0:
        print(f"  ⚠️  WARNING: {out_of_bounds} cells have centroids outside domain")
        all_valid = False
    else:
        print(f"  ✓ All centroids within domain bounds")
    
    # Check 4: Check for NaN/Inf
    print(f"\n--- Numerical Stability ---")
    has_nan = any(np.isnan(c['phi_max']) or np.isnan(c['computed_volume']) for c in cells)
    has_inf = any(np.isinf(c['phi_max']) or np.isinf(c['computed_volume']) for c in cells)
    
    if has_nan:
        print(f"  ❌ ERROR: NaN values detected!")
        all_valid = False
    elif has_inf:
        print(f"  ❌ ERROR: Inf values detected!")
        all_valid = False
    else:
        print(f"  ✓ No NaN or Inf values")
    
    # Check 5: Subdomain sizes
    print(f"\n--- Memory Usage ---")
    total_voxels = sum(c['field_size'] for c in cells)
    total_mb = total_voxels * 4 / (1024 * 1024)
    avg_subdomain = np.mean([c['subdomain_dims'][0] for c in cells])
    
    print(f"  Total voxels: {total_voxels:,}")
    print(f"  Total phi storage: {total_mb:.1f} MB")
    print(f"  Average subdomain size: {avg_subdomain:.0f}³")
    print(f"  Voxels per cell: {total_voxels / len(cells):,.0f}")
    
    # Summary
    print(f"\n{'='*60}")
    if all_valid:
        print(f"✓ VALIDATION PASSED")
    else:
        print(f"❌ VALIDATION FAILED - see warnings above")
    print(f"{'='*60}")
    
    return all_valid


def compare_checkpoints(filepath1, filepath2):
    """Compare two checkpoints to verify evolution is physical."""
    print(f"\n{'='*60}")
    print(f"Comparing checkpoints:")
    print(f"  Early: {filepath1}")
    print(f"  Late:  {filepath2}")
    print(f"{'='*60}")
    
    params1, cells1 = read_checkpoint_3d(filepath1)
    params2, cells2 = read_checkpoint_3d(filepath2)
    
    dt_sim = params2['time'] - params1['time']
    print(f"\nTime evolution: t={params1['time']:.2f} → t={params2['time']:.2f} (Δt={dt_sim:.2f})")
    
    # Volume change
    vols1 = np.array([c['computed_volume'] for c in cells1])
    vols2 = np.array([c['computed_volume'] for c in cells2])
    
    vol_change = np.mean(vols2) - np.mean(vols1)
    vol_change_pct = 100 * vol_change / params1['target_volume']
    
    print(f"\nVolume evolution:")
    print(f"  Mean volume: {np.mean(vols1):.1f} → {np.mean(vols2):.1f} ({vol_change_pct:+.2f}%)")
    
    # Centroid movement
    if len(cells1) == len(cells2):
        displacements = []
        for c1, c2 in zip(cells1, cells2):
            dx = c2['centroid'][0] - c1['centroid'][0]
            dy = c2['centroid'][1] - c1['centroid'][1]
            dz = c2['centroid'][2] - c1['centroid'][2]
            # Handle periodic boundaries
            Nx, Ny, Nz = params1['Nx'], params1['Ny'], params1['Nz']
            if dx > Nx/2: dx -= Nx
            if dx < -Nx/2: dx += Nx
            if dy > Ny/2: dy -= Ny
            if dy < -Ny/2: dy += Ny
            if dz > Nz/2: dz -= Nz
            if dz < -Nz/2: dz += Nz
            displacements.append(np.sqrt(dx**2 + dy**2 + dz**2))
        
        mean_disp = np.mean(displacements)
        max_disp = np.max(displacements)
        
        print(f"\nCell movement:")
        print(f"  Mean displacement: {mean_disp:.2f}")
        print(f"  Max displacement:  {max_disp:.2f}")
        
        if mean_disp > 0.1 and params1['v_A'] == 0:
            print(f"  Note: Cells moving despite v_A=0 (equilibration in progress)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_3d.py <checkpoint_dir_or_file> [checkpoint2]")
        print("\nExamples:")
        print("  python validate_3d.py agent_test_runs/baseline_3d_32")
        print("  python validate_3d.py checkpoint_3d_001000.bin")
        print("  python validate_3d.py early.bin late.bin  # compare two")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_dir():
        # Find all checkpoints in directory
        checkpoints = sorted(path.glob("checkpoint_3d_*.bin"))
        if not checkpoints:
            print(f"No checkpoint_3d_*.bin files found in {path}")
            sys.exit(1)
        
        print(f"Found {len(checkpoints)} checkpoints")
        
        # Validate first and last
        validate_checkpoint(str(checkpoints[0]))
        
        if len(checkpoints) > 1:
            validate_checkpoint(str(checkpoints[-1]))
            compare_checkpoints(str(checkpoints[0]), str(checkpoints[-1]))
    
    elif len(sys.argv) >= 3:
        # Compare two specific files
        validate_checkpoint(sys.argv[1])
        validate_checkpoint(sys.argv[2])
        compare_checkpoints(sys.argv[1], sys.argv[2])
    
    else:
        # Single file
        validate_checkpoint(str(path))


if __name__ == "__main__":
    main()
