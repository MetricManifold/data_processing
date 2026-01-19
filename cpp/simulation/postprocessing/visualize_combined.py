#!/usr/bin/env python3
"""
Combined visualization: simulation movie with synchronized observable traces.

Creates a video with:
- Top: simulation frame (cell positions)
- Bottom: observable data traced out in sync with the movie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import argparse
import re
from vtk import vtkStructuredPointsReader
from vtk.util.numpy_support import vtk_to_numpy
import imageio.v2 as imageio
from io import BytesIO


def load_vtk(filepath):
    """Load VTK structured points file and return phi field."""
    reader = vtkStructuredPointsReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    
    data = reader.GetOutput()
    dims = data.GetDimensions()
    
    # Get phi field
    phi = vtk_to_numpy(data.GetPointData().GetScalars('phi'))
    phi = phi.reshape(dims[1], dims[0])  # VTK uses column-major
    
    return phi, dims


def get_frame_number(filename):
    """Extract frame number from VTK filename."""
    match = re.search(r'frame_(\d+)\.vtk', str(filename))
    return int(match.group(1)) if match else 0


def create_combined_video(data_dir, output_path=None, fps=30):
    """Create combined video with simulation and observables."""
    data_dir = Path(data_dir)
    
    # Load observables
    obs_file = data_dir / 'observables.csv'
    if not obs_file.exists():
        print(f"Error: {obs_file} not found")
        return
    
    # Read CSV with comment header
    with open(obs_file) as f:
        header = f.readline().strip()
        if header.startswith('#'):
            header = header[1:].strip()
    
    df = pd.read_csv(obs_file, comment='#', header=None)
    df.columns = header.split(',')
    
    # Find VTK files
    vtk_files = sorted(data_dir.glob('frame_*.vtk'), key=get_frame_number)
    if not vtk_files:
        print(f"Error: No VTK files found in {data_dir}")
        return
    
    print(f"Found {len(vtk_files)} VTK frames and {len(df)} observable records")
    
    # Get frame numbers and match to observable steps
    frame_numbers = [get_frame_number(f) for f in vtk_files]
    obs_steps = df['step'].values
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Top: simulation view (larger)
    ax_sim = fig.add_axes([0.05, 0.45, 0.6, 0.5])
    
    # Right side: energy and stress  
    ax_energy = fig.add_axes([0.7, 0.55, 0.27, 0.35])
    ax_stress = fig.add_axes([0.7, 0.12, 0.27, 0.35])
    
    # Bottom: pressure and coordination
    ax_press = fig.add_axes([0.05, 0.08, 0.28, 0.3])
    ax_coord = fig.add_axes([0.38, 0.08, 0.28, 0.3])
    
    # Load first frame to get dimensions
    phi0, dims = load_vtk(vtk_files[0])
    
    # Initialize simulation plot
    im = ax_sim.imshow(phi0, origin='lower', cmap='viridis', vmin=0, vmax=1)
    ax_sim.set_title('Cell Simulation', fontsize=12)
    ax_sim.set_xlabel('x')
    ax_sim.set_ylabel('y')
    time_text = ax_sim.text(0.02, 0.98, '', transform=ax_sim.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Initialize observable traces
    time = df['time'].values
    
    # Energy plot
    line_eg, = ax_energy.plot([], [], 'b-', label='E_grad', linewidth=1)
    line_eb, = ax_energy.plot([], [], 'orange', label='E_bulk', linewidth=1)
    line_ei, = ax_energy.plot([], [], 'g-', label='E_int', linewidth=1)
    ax_energy.set_xlim(time[0], time[-1])
    ax_energy.set_ylim(0, df['E_grad'].max() * 1.1)
    ax_energy.set_xlabel('Time')
    ax_energy.set_ylabel('Energy')
    ax_energy.set_title('Energy Components')
    ax_energy.legend(fontsize=8, loc='upper right')
    ax_energy.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # Stress plot
    line_sxx, = ax_stress.plot([], [], 'b-', label='σ_xx', linewidth=1)
    line_syy, = ax_stress.plot([], [], 'orange', label='σ_yy', linewidth=1)
    line_sxy, = ax_stress.plot([], [], 'g-', label='σ_xy', linewidth=1)
    ax_stress.set_xlim(time[0], time[-1])
    stress_max = max(df['sigma_xx'].max(), df['sigma_yy'].max()) * 1.1
    stress_min = min(df['sigma_xx'].min(), df['sigma_yy'].min(), df['sigma_xy'].min()) * 1.1
    ax_stress.set_ylim(stress_min, stress_max)
    ax_stress.set_xlabel('Time')
    ax_stress.set_ylabel('Stress')
    ax_stress.set_title('Stress Tensor')
    ax_stress.legend(fontsize=8, loc='upper right')
    ax_stress.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # Pressure plot
    line_p, = ax_press.plot([], [], 'b-', linewidth=1.5)
    ax_press.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_press.set_xlim(time[0], time[-1])
    ax_press.set_ylim(df['pressure'].min() * 1.1, max(0, df['pressure'].max()) * 1.1)
    ax_press.set_xlabel('Time')
    ax_press.set_ylabel('Pressure')
    ax_press.set_title('Pressure')
    ax_press.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # Coordination plot
    line_z, = ax_coord.plot([], [], 'g-', linewidth=1.5)
    ax_coord.fill_between([], [], [], alpha=0.3, color='g')
    ax_coord.set_xlim(time[0], time[-1])
    z_min = (df['z_mean'] - df['z_std']).min()
    z_max = (df['z_mean'] + df['z_std']).max()
    ax_coord.set_ylim(z_min - 0.5, z_max + 0.5)
    ax_coord.set_xlabel('Time')
    ax_coord.set_ylabel('Coordination z')
    ax_coord.set_title('Mean Coordination')
    
    # Marker for current position on each plot
    marker_e, = ax_energy.plot([], [], 'ko', markersize=6)
    marker_s, = ax_stress.plot([], [], 'ko', markersize=6)
    marker_p, = ax_press.plot([], [], 'ko', markersize=6)
    marker_z, = ax_coord.plot([], [], 'ko', markersize=6)
    
    def init():
        line_eg.set_data([], [])
        line_eb.set_data([], [])
        line_ei.set_data([], [])
        line_sxx.set_data([], [])
        line_syy.set_data([], [])
        line_sxy.set_data([], [])
        line_p.set_data([], [])
        line_z.set_data([], [])
        marker_e.set_data([], [])
        marker_s.set_data([], [])
        marker_p.set_data([], [])
        marker_z.set_data([], [])
        time_text.set_text('')
        return (im, line_eg, line_eb, line_ei, line_sxx, line_syy, line_sxy,
                line_p, line_z, marker_e, marker_s, marker_p, marker_z, time_text)
    
    def update(frame_idx):
        # Load VTK frame
        phi, _ = load_vtk(vtk_files[frame_idx])
        im.set_array(phi)
        
        # Find corresponding observable index
        frame_step = frame_numbers[frame_idx]
        obs_idx = np.searchsorted(obs_steps, frame_step)
        obs_idx = min(obs_idx, len(df) - 1)
        
        # Update time text
        t = df['time'].iloc[obs_idx]
        step = df['step'].iloc[obs_idx]
        time_text.set_text(f't = {t:.1f}\nstep = {step}')
        
        # Update traces (show data up to current point)
        idx = obs_idx + 1
        t_slice = time[:idx]
        
        line_eg.set_data(t_slice, df['E_grad'].values[:idx])
        line_eb.set_data(t_slice, df['E_bulk'].values[:idx])
        line_ei.set_data(t_slice, df['E_int'].values[:idx])
        
        line_sxx.set_data(t_slice, df['sigma_xx'].values[:idx])
        line_syy.set_data(t_slice, df['sigma_yy'].values[:idx])
        line_sxy.set_data(t_slice, df['sigma_xy'].values[:idx])
        
        line_p.set_data(t_slice, df['pressure'].values[:idx])
        line_z.set_data(t_slice, df['z_mean'].values[:idx])
        
        # Update fill for coordination std
        for coll in list(ax_coord.collections):
            coll.remove()
        ax_coord.fill_between(t_slice,
                             df['z_mean'].values[:idx] - df['z_std'].values[:idx],
                             df['z_mean'].values[:idx] + df['z_std'].values[:idx],
                             alpha=0.3, color='g')
        
        # Update markers at current position
        marker_e.set_data([t], [df['E_grad'].iloc[obs_idx]])
        marker_s.set_data([t], [df['sigma_xx'].iloc[obs_idx]])
        marker_p.set_data([t], [df['pressure'].iloc[obs_idx]])
        marker_z.set_data([t], [df['z_mean'].iloc[obs_idx]])
        
        return (im, line_eg, line_eb, line_ei, line_sxx, line_syy, line_sxy,
                line_p, line_z, marker_e, marker_s, marker_p, marker_z, time_text)
    
    # Create animation by generating each frame as image
    print(f"Creating animation with {len(vtk_files)} frames at {fps} fps...")
    
    if output_path is None:
        output_path = data_dir / 'combined_movie.mp4'
    
    # Generate frames and save using imageio
    frames = []
    for frame_idx in range(len(vtk_files)):
        if frame_idx % 10 == 0:
            print(f"  Processing frame {frame_idx+1}/{len(vtk_files)}...")
        
        # Load VTK frame
        phi, _ = load_vtk(vtk_files[frame_idx])
        im.set_array(phi)
        
        # Find corresponding observable index
        frame_step = frame_numbers[frame_idx]
        obs_idx = np.searchsorted(obs_steps, frame_step)
        obs_idx = min(obs_idx, len(df) - 1)
        
        # Update time text
        t = df['time'].iloc[obs_idx]
        step = df['step'].iloc[obs_idx]
        time_text.set_text(f't = {t:.1f}\nstep = {step}')
        
        # Update traces (show data up to current point)
        idx = obs_idx + 1
        t_slice = time[:idx]
        
        line_eg.set_data(t_slice, df['E_grad'].values[:idx])
        line_eb.set_data(t_slice, df['E_bulk'].values[:idx])
        line_ei.set_data(t_slice, df['E_int'].values[:idx])
        
        line_sxx.set_data(t_slice, df['sigma_xx'].values[:idx])
        line_syy.set_data(t_slice, df['sigma_yy'].values[:idx])
        line_sxy.set_data(t_slice, df['sigma_xy'].values[:idx])
        
        line_p.set_data(t_slice, df['pressure'].values[:idx])
        line_z.set_data(t_slice, df['z_mean'].values[:idx])
        
        # Update fill for coordination std
        for coll in list(ax_coord.collections):
            coll.remove()
        ax_coord.fill_between(t_slice,
                             df['z_mean'].values[:idx] - df['z_std'].values[:idx],
                             df['z_mean'].values[:idx] + df['z_std'].values[:idx],
                             alpha=0.3, color='g')
        
        # Update markers at current position
        marker_e.set_data([t], [df['E_grad'].iloc[obs_idx]])
        marker_s.set_data([t], [df['sigma_xx'].iloc[obs_idx]])
        marker_p.set_data([t], [df['pressure'].iloc[obs_idx]])
        marker_z.set_data([t], [df['z_mean'].iloc[obs_idx]])
        
        # Render to buffer
        fig.canvas.draw()
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
    
    # Save as video
    print(f"Writing video to {output_path}...")
    imageio.mimwrite(str(output_path), frames, fps=fps)
    print(f"Saved: {output_path}")
    
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create combined simulation + observables video')
    parser.add_argument('data_dir', help='Directory with VTK files and observables.csv')
    parser.add_argument('-o', '--output', help='Output video path')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    args = parser.parse_args()
    create_combined_video(args.data_dir, args.output, args.fps)
