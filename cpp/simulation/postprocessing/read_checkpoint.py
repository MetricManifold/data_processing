#!/usr/bin/env python3
"""
Read and display 2D checkpoint file parameters.
Usage: python read_checkpoint.py <checkpoint.bin>
"""

import struct
import sys
from pathlib import Path


def read_checkpoint_2d(filename):
    """Read a 2D checkpoint file and return all parameters."""
    with open(filename, 'rb') as f:
        # CheckpointHeader (40 bytes in v4)
        magic = struct.unpack('I', f.read(4))[0]
        if magic != 0x43454C4C:  # "CELL"
            raise ValueError(f"Invalid magic number: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        current_step = struct.unpack('i', f.read(4))[0]
        current_time = struct.unpack('f', f.read(4))[0]
        num_cells = struct.unpack('i', f.read(4))[0]
        
        # Runtime options (v3+)
        save_interval = struct.unpack('i', f.read(4))[0]
        checkpoint_interval = struct.unpack('i', f.read(4))[0]
        trajectory_samples = struct.unpack('i', f.read(4))[0]
        save_vtk = struct.unpack('?', f.read(1))[0]
        save_tracking = struct.unpack('?', f.read(1))[0]
        compute_diagnostics = struct.unpack('?', f.read(1))[0]
        save_individual_fields = struct.unpack('?', f.read(1))[0]
        
        # v4 has sim_params_size
        if version >= 4:
            sim_params_size = struct.unpack('I', f.read(4))[0]
        else:
            sim_params_size = 0
        
        header = {
            'magic': hex(magic),
            'version': version,
            'current_step': current_step,
            'current_time': current_time,
            'num_cells': num_cells,
            'save_interval': save_interval,
            'checkpoint_interval': checkpoint_interval,
            'trajectory_samples': trajectory_samples,
            'save_vtk': save_vtk,
            'save_tracking': save_tracking,
            'compute_diagnostics': compute_diagnostics,
            'save_individual_fields': save_individual_fields,
            'sim_params_size': sim_params_size,
        }
        
        # SimParams (76 bytes based on struct definition)
        Nx = struct.unpack('i', f.read(4))[0]
        Ny = struct.unpack('i', f.read(4))[0]
        dx = struct.unpack('f', f.read(4))[0]
        dy = struct.unpack('f', f.read(4))[0]
        
        dt = struct.unpack('f', f.read(4))[0]
        t_end = struct.unpack('f', f.read(4))[0]
        save_interval_params = struct.unpack('i', f.read(4))[0]
        
        lambda_ = struct.unpack('f', f.read(4))[0]
        gamma = struct.unpack('f', f.read(4))[0]
        
        kappa = struct.unpack('f', f.read(4))[0]
        
        target_radius = struct.unpack('f', f.read(4))[0]
        mu = struct.unpack('f', f.read(4))[0]
        
        v_A = struct.unpack('f', f.read(4))[0]
        xi = struct.unpack('f', f.read(4))[0]
        tau = struct.unpack('f', f.read(4))[0]
        
        halo_width = struct.unpack('i', f.read(4))[0]
        min_subdomain_size = struct.unpack('i', f.read(4))[0]
        subdomain_padding = struct.unpack('f', f.read(4))[0]
        
        # Motility model (v4+)
        if version >= 4 and sim_params_size >= 76:
            motility_model = struct.unpack('i', f.read(4))[0]
            motility_model_str = "RunAndTumble" if motility_model == 0 else "ABP"
        else:
            motility_model_str = "RunAndTumble (default)"
        
        params = {
            'Nx': Nx,
            'Ny': Ny,
            'dx': dx,
            'dy': dy,
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
            'halo_width': halo_width,
            'min_subdomain_size': min_subdomain_size,
            'subdomain_padding': subdomain_padding,
            'motility_model': motility_model_str,
        }
        
        return header, params


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_checkpoint.py <checkpoint.bin>")
        sys.exit(1)
    
    filename = sys.argv[1]
    if not Path(filename).exists():
        print(f"Error: File not found: {filename}")
        sys.exit(1)
    
    header, params = read_checkpoint_2d(filename)
    
    print("=" * 60)
    print(f"Checkpoint: {filename}")
    print("=" * 60)
    
    print("\n--- Header ---")
    print(f"  Version:        {header['version']}")
    print(f"  Current step:   {header['current_step']:,}")
    print(f"  Current time:   {header['current_time']:,.2f}")
    print(f"  Num cells:      {header['num_cells']}")
    print(f"  Save interval:  {header['save_interval']}")
    print(f"  Checkpoint int: {header['checkpoint_interval']}")
    
    print("\n--- Domain ---")
    print(f"  Nx × Ny:        {params['Nx']} × {params['Ny']}")
    print(f"  dx, dy:         {params['dx']}, {params['dy']}")
    
    print("\n--- Time Stepping ---")
    print(f"  dt:             {params['dt']}")
    print(f"  t_end:          {params['t_end']:,.0f}")
    
    print("\n--- Physics ---")
    print(f"  λ (interface):  {params['lambda']}")
    print(f"  γ (gradient):   {params['gamma']}")
    print(f"  κ (interaction):{params['kappa']}")
    print(f"  R (radius):     {params['target_radius']}")
    print(f"  μ (volume):     {params['mu']}")
    
    print("\n--- Motility ---")
    print(f"  v_A:            {params['v_A']}")
    print(f"  ξ (friction):   {params['xi']:.2e}")
    print(f"  τ (reorient):   {params['tau']:.2e}")
    print(f"  Model:          {params['motility_model']}")
    
    # Derived quantities
    target_area = 3.14159 * params['target_radius']**2
    print("\n--- Derived ---")
    print(f"  Target area:    {target_area:,.0f}")
    print(f"  Bulk coeff:     {30.0 / params['lambda']**2:.4f}")
    print(f"  Volume coeff:   {params['mu'] / target_area:.6f}")
    
    # Progress info
    if params['t_end'] > 0:
        progress = header['current_time'] / params['t_end'] * 100
        print(f"\n--- Progress ---")
        print(f"  {header['current_time']:,.0f} / {params['t_end']:,.0f} ({progress:.1f}%)")


if __name__ == "__main__":
    main()
