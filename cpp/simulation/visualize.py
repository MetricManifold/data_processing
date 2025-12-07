"""
Simple VTK visualization for cell simulation output.
Reads legacy VTK structured points and plots with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys


def load_trajectory_data(output_dir):
    """Load trajectory data with polarization information.
    
    Returns dict mapping time -> list of (cell_id, x, y, px, py)
    """
    trajectory_file = Path(output_dir) / 'trajectory.txt'
    if not trajectory_file.exists():
        return None
    
    data_by_time = {}
    with open(trajectory_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 9:
                t = float(parts[0])
                cell_id = int(parts[1])
                x, y = float(parts[2]), float(parts[3])
                px, py = float(parts[6]), float(parts[7])
                
                if t not in data_by_time:
                    data_by_time[t] = []
                data_by_time[t].append((cell_id, x, y, px, py))
    
    return data_by_time


def get_polarization_for_frame(trajectory_data, frame_time):
    """Get interpolated polarization data for a given frame time.
    
    Returns dict mapping cell_id -> (x, y, px, py)
    """
    if trajectory_data is None:
        return None
    
    times = sorted(trajectory_data.keys())
    if not times:
        return None
    
    # Find the closest time in trajectory data
    closest_time = min(times, key=lambda t: abs(t - frame_time))
    
    # If trajectory time is too far from frame time, skip
    # (trajectory samples might be sparse)
    if abs(closest_time - frame_time) > 500:  # Allow up to 500 time units difference
        return None
    
    result = {}
    for cell_id, x, y, px, py in trajectory_data[closest_time]:
        result[cell_id] = (x, y, px, py)
    
    return result


def read_vtk_structured_points(filename):
    """Read a legacy VTK structured points file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    dims = None
    origin = None
    spacing = None
    n_points = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('DIMENSIONS'):
            parts = line.split()
            dims = (int(parts[1]), int(parts[2]), int(parts[3]))
        elif line.startswith('ORIGIN'):
            parts = line.split()
            origin = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif line.startswith('SPACING'):
            parts = line.split()
            spacing = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif line.startswith('POINT_DATA'):
            n_points = int(line.split()[1])
        elif line.startswith('SCALARS'):
            # Skip LOOKUP_TABLE line
            i += 2  # Skip SCALARS and LOOKUP_TABLE lines
            break
        i += 1
    
    if dims is None:
        raise ValueError("Could not parse VTK dimensions")
    
    # Read scalar data
    values = []
    while len(values) < n_points and i < len(lines):
        data_line = lines[i].strip()
        if data_line:
            values.extend([float(x) for x in data_line.split()])
        i += 1
    
    # Reshape to 2D (assuming z=1)
    data = np.array(values).reshape((dims[1], dims[0]))
    
    return dims, origin, spacing, {'phi': data}


def plot_frame(vtk_file, output_file=None, show=True, polarization_data=None, arrow_scale=15):
    """Plot a single VTK frame with optional polarization arrows.
    
    Args:
        vtk_file: Path to VTK file
        output_file: Path to save PNG (optional)
        show: Whether to display the plot
        polarization_data: Dict mapping cell_id -> (x, y, px, py)
        arrow_scale: Length scale for polarization arrows
    """
    dims, origin, spacing, cell_data = read_vtk_structured_points(vtk_file)
    
    # Create figure with fixed size
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot phase field
    data = cell_data['phi']
    im = ax.imshow(data, origin='lower', cmap='viridis', vmin=0, vmax=1, 
                   extent=[0, dims[0], 0, dims[1]])
    
    # Set fixed axis limits to prevent resizing due to arrows
    ax.set_xlim(0, dims[0])
    ax.set_ylim(0, dims[1])
    
    # Draw polarization arrows if available
    if polarization_data:
        for cell_id, (x, y, px, py) in polarization_data.items():
            # Draw arrow from centroid in polarization direction
            # Use FancyArrow with clip_on=True to clip arrows at plot boundaries
            ax.arrow(x, y, px * arrow_scale, py * arrow_scale,
                    head_width=3, head_length=2, fc='red', ec='darkred',
                    linewidth=1.5, alpha=0.8, zorder=10, clip_on=True)
    
    # Extract frame number from filename
    frame_num = int(Path(vtk_file).stem.split('_')[1])
    ax.set_title(f'Phase Field (frame {frame_num})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.046, label='φ')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_frames(output_dir, save_images=True, show_last=False, reverse=False, 
                    frame=None, frame_range=None, show_polarization=True, arrow_scale=15):
    """Plot VTK frames in a directory.
    
    Args:
        output_dir: Directory containing VTK files
        save_images: Whether to save PNG images
        show_last: Whether to display the last frame
        reverse: Process frames in reverse order (last to first)
        frame: Specific frame number to process (e.g., 500)
        frame_range: Tuple of (start, end) frame numbers to process
        show_polarization: Whether to show polarization arrows
        arrow_scale: Length scale for polarization arrows
    """
    output_dir = Path(output_dir)
    vtk_files = sorted(output_dir.glob('frame_*.vtk'))
    
    if not vtk_files:
        print(f"No VTK files found in {output_dir}")
        return
    
    print(f"Found {len(vtk_files)} VTK frames")
    
    # Load trajectory data for polarization arrows
    trajectory_data = None
    if show_polarization:
        trajectory_data = load_trajectory_data(output_dir)
        if trajectory_data:
            print(f"Loaded trajectory data with {len(trajectory_data)} time samples")
        else:
            print("No trajectory data found - polarization arrows will not be shown")
    
    # Try to get dt from params file or infer from trajectory times
    dt = 0.01  # Default dt
    params_file = output_dir / 'params.txt'
    if params_file.exists():
        with open(params_file, 'r') as f:
            for line in f:
                if 'dt=' in line or 'dt =' in line:
                    try:
                        dt = float(line.split('=')[1].strip())
                    except:
                        pass
    
    # Filter to specific frame if requested
    if frame is not None:
        frame_file = output_dir / f'frame_{frame:06d}.vtk'
        if frame_file.exists():
            vtk_files = [frame_file]
            print(f"Processing single frame: {frame}")
        else:
            print(f"Frame {frame} not found: {frame_file}")
            return
    
    # Filter to frame range if requested
    if frame_range is not None:
        start_frame, end_frame = frame_range
        vtk_files = [f for f in vtk_files 
                     if start_frame <= int(f.stem.split('_')[1]) <= end_frame]
        print(f"Processing frames {start_frame} to {end_frame} ({len(vtk_files)} frames)")
    
    # Reverse order if requested
    if reverse:
        vtk_files = vtk_files[::-1]
        print("Processing in reverse order (last to first)")
    
    # Create images directory
    if save_images:
        img_dir = output_dir / 'images'
        img_dir.mkdir(exist_ok=True)
    
    for i, vtk_file in enumerate(vtk_files):
        print(f"Processing {vtk_file.name}...")
        
        img_file = img_dir / f"{vtk_file.stem}.png" if save_images else None
        show = False  # Never show windows
        
        # Get polarization data for this frame
        polarization = None
        if trajectory_data:
            frame_num = int(vtk_file.stem.split('_')[1])
            frame_time = frame_num * dt  # Convert step to time
            polarization = get_polarization_for_frame(trajectory_data, frame_time)
        
        try:
            plot_frame(vtk_file, output_file=img_file, show=show, 
                      polarization_data=polarization, arrow_scale=arrow_scale)
        except Exception as e:
            print(f"  Error: {e}")


def plot_tracking(output_dir):
    """Plot cell tracking data."""
    tracking_file = Path(output_dir) / 'tracking.txt'
    
    if not tracking_file.exists():
        print(f"No tracking file found: {tracking_file}")
        return
    
    # Read tracking data: time x y cell_id
    data = []
    with open(tracking_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                t, x, y, cell_id = float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
                data.append((t, x, y, cell_id))
    
    if not data:
        print("No tracking data found")
        return
    
    data = np.array(data)
    
    # Get unique cells
    cell_ids = np.unique(data[:, 3].astype(int))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot trajectories
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(cell_ids)))
    for i, cell_id in enumerate(cell_ids):
        mask = data[:, 3] == cell_id
        ax.plot(data[mask, 1], data[mask, 2], '-o', markersize=3, 
                color=colors[i], label=f'Cell {int(cell_id)}', alpha=0.7)
        # Mark start with square
        ax.plot(data[mask, 1][0], data[mask, 2][0], 's', markersize=10, color=colors[i])
        # Mark end with star
        ax.plot(data[mask, 1][-1], data[mask, 2][-1], '*', markersize=12, color=colors[i])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Cell Trajectories (□=start, ★=end)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot x,y positions over time
    ax = axes[1]
    for i, cell_id in enumerate(cell_ids):
        mask = data[:, 3] == cell_id
        ax.plot(data[mask, 0], data[mask, 1], '-', color=colors[i], label=f'Cell {int(cell_id)} x')
        ax.plot(data[mask, 0], data[mask, 2], '--', color=colors[i], alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_title('Cell Positions vs Time (solid=x, dashed=y)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'tracking_plot.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {Path(output_dir) / 'tracking_plot.png'}")
    plt.show()


#=============================================================================
# ENERGY VISUALIZATION
#=============================================================================

def load_energy_metrics(output_dir):
    """Load energy metrics computed during simulation (if available).
    
    The simulation can save accurate energy metrics including true interaction energy
    when run with --save-individual-fields.
    
    Returns list of dicts with keys: frame, time, total, gradient, bulk, interaction
    Returns None if file doesn't exist.
    """
    energy_file = Path(output_dir) / 'energy_metrics.txt'
    if not energy_file.exists():
        return None
    
    energy_data = []
    with open(energy_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    energy_data.append({
                        'frame': int(parts[0]),
                        'time': float(parts[1]),
                        'total': float(parts[2]),
                        'gradient': float(parts[3]),
                        'bulk': float(parts[4]),
                        'interaction': float(parts[5])
                    })
                except ValueError:
                    continue
    
    return energy_data if energy_data else None


def load_individual_cell_fields(output_dir, frame_num):
    """Load individual cell fields for a given frame.
    
    Returns:
        cell_fields: list of 2D arrays, one per cell
        sum_field: 2D array with sum of all cell fields
        dims: (Nx, Ny)
    
    Returns None if individual fields are not available.
    """
    fields_dir = Path(output_dir) / 'fields'
    if not fields_dir.exists():
        return None, None, None
    
    # Find sum field
    sum_file = fields_dir / f'frame_{frame_num:06d}_sum.vtk'
    if not sum_file.exists():
        return None, None, None
    
    dims, origin, spacing, sum_data = read_vtk_structured_points(sum_file)
    sum_field = sum_data.get('phi_sum', sum_data.get('phi', None))
    
    # Find cell fields
    cell_files = sorted(fields_dir.glob(f'frame_{frame_num:06d}_cell_*.vtk'))
    cell_fields = []
    for cf in cell_files:
        _, _, _, cell_data = read_vtk_structured_points(cf)
        cell_fields.append(cell_data['phi'])
    
    return cell_fields, sum_field, dims


def compute_interaction_energy_from_fields(cell_fields, dx=1.0, dy=1.0, kappa=10.0, lambda_=7.0):
    """Compute true interaction energy from individual cell fields.
    
    E_int = (30κ/λ²) ∫ Σ_{n≠m} φ_n² φ_m² dx
    """
    if not cell_fields or len(cell_fields) < 2:
        return 0.0, None
    
    interaction_coeff = 30.0 * kappa / (lambda_ * lambda_)
    dA = dx * dy
    
    # Compute pairwise interaction energy density
    interaction_density = np.zeros_like(cell_fields[0])
    
    for i in range(len(cell_fields)):
        for j in range(i + 1, len(cell_fields)):
            phi_i = cell_fields[i]
            phi_j = cell_fields[j]
            # φ_i² φ_j²
            interaction_density += phi_i**2 * phi_j**2
    
    interaction_density *= interaction_coeff
    total_interaction = np.sum(interaction_density) * dA
    
    return total_interaction, interaction_density


def compute_energy_from_phi(phi, dx=1.0, dy=1.0, lambda_=7.0, gamma=1.0, kappa=10.0, 
                            mu=1.0, target_radius=49.0):
    """Compute energy components from a combined phase field.
    
    For a multi-cell field where cells are summed together, we compute:
    - Gradient energy: γ ∫ |∇φ|² dx
    - Bulk potential: (30/λ²) ∫ φ²(1-φ)² dx  
    - Interaction energy: excess bulk energy where φ > 1 (cells overlapping)
    
    Also computes strain field for visualization:
    - Laplacian of φ shows compression (positive) vs tension (negative)
    - This creates a field that extends into cell interiors
    
    Returns dict with energy components and spatial energy density.
    """
    bulk_coeff = 30.0 / (lambda_ * lambda_)
    interaction_coeff = 30.0 * kappa / (lambda_ * lambda_)
    
    # Compute gradient using central differences
    grad_y, grad_x = np.gradient(phi, dy, dx)
    grad_sq = grad_x**2 + grad_y**2
    
    # Gradient energy density
    gradient_energy_density = gamma * grad_sq
    
    # Bulk potential density: f(φ) = (30/λ²) φ²(1-φ)²
    phi_clipped = np.clip(phi, 0, 2)  # Allow some overshoot for overlap
    bulk_energy_density = bulk_coeff * (phi_clipped**2) * ((1 - phi_clipped)**2)
    
    # Interaction/overlap energy density
    # This measures energy due to cell-cell overlap (φ > 1 regions)
    # Use the bulk potential evaluated at φ > 1 which penalizes overlap
    overlap = np.maximum(0, phi - 1)
    # Also count regions where bulk potential is elevated due to overlap
    interaction_energy_density = interaction_coeff * overlap**2
    
    # For visualization: compute Laplacian (measures local curvature/strain)
    # ∇²φ > 0: concave region (compression)
    # ∇²φ < 0: convex region (tension/bulging)
    laplacian = np.zeros_like(phi)
    laplacian[1:-1, 1:-1] = (phi[2:, 1:-1] + phi[:-2, 1:-1] + 
                              phi[1:-1, 2:] + phi[1:-1, :-2] - 
                              4 * phi[1:-1, 1:-1]) / (dx * dy)
    
    # Compute second derivatives for strain tensor
    grad_xx = np.gradient(grad_x, dx, axis=1)
    grad_yy = np.gradient(grad_y, dy, axis=0)
    grad_xy = np.gradient(grad_x, dy, axis=0)
    
    # Principal strain magnitude (simplified)
    # This shows deformation intensity
    strain_magnitude = np.sqrt(grad_xx**2 + grad_yy**2 + 2*grad_xy**2)
    
    # Total energy density (for heatmap)
    total_energy_density = gradient_energy_density + bulk_energy_density + interaction_energy_density
    
    # Integrated energies
    dA = dx * dy
    gradient_energy = np.sum(gradient_energy_density) * dA
    bulk_energy = np.sum(bulk_energy_density) * dA
    interaction_energy = np.sum(interaction_energy_density) * dA
    total_energy = gradient_energy + bulk_energy + interaction_energy
    
    return {
        'gradient': gradient_energy,
        'bulk': bulk_energy,
        'interaction': interaction_energy,
        'total': total_energy,
        'gradient_density': gradient_energy_density,
        'bulk_density': bulk_energy_density,
        'interaction_density': interaction_energy_density,
        'total_density': total_energy_density,
        'overlap': overlap,
        'laplacian': laplacian,
        'strain_magnitude': strain_magnitude
    }


def plot_energy_frame(vtk_file, output_file=None, show=True, 
                      energy_history=None, current_idx=0,
                      polarization_data=None, arrow_scale=15,
                      lambda_=7.0, gamma=1.0, kappa=10.0):
    """Plot a single frame with energy visualization.
    
    Layout:
    - Left: Cell contours with energy heatmap overlay (large square)
    - Right: 4 stacked energy time series plots (Total, Gradient, Bulk, Interaction)
    """
    from matplotlib.ticker import ScalarFormatter
    
    dims, origin, spacing, cell_data = read_vtk_structured_points(vtk_file)
    phi = cell_data['phi']
    dx, dy = spacing[0], spacing[1]
    
    # Compute energy
    energy = compute_energy_from_phi(phi, dx, dy, lambda_, gamma, kappa)
    
    # Create figure with side-by-side layout
    fig = plt.figure(figsize=(16, 10))
    
    # Main axes for cell visualization (left side, square)
    ax_main = fig.add_axes([0.05, 0.08, 0.50, 0.85])  # [left, bottom, width, height]
    ax_cbar = fig.add_axes([0.56, 0.08, 0.015, 0.85])  # Colorbar (moved left)
    
    # 4 stacked energy time series axes (right side - more space from colorbar)
    plot_height = 0.18
    plot_gap = 0.04
    plot_left = 0.68  # More space from colorbar
    plot_width = 0.30
    
    ax_total = fig.add_axes([plot_left, 0.08 + 3*(plot_height + plot_gap), plot_width, plot_height])
    ax_gradient = fig.add_axes([plot_left, 0.08 + 2*(plot_height + plot_gap), plot_width, plot_height])
    ax_bulk = fig.add_axes([plot_left, 0.08 + 1*(plot_height + plot_gap), plot_width, plot_height])
    ax_interaction = fig.add_axes([plot_left, 0.08, plot_width, plot_height])
    
    # === Main visualization: strain field that extends into cells ===
    
    # Strategy: Use a combination of:
    # 1. Gradient energy at interfaces (bright edges)
    # 2. Strain magnitude that propagates into cells (shows compression/tension)
    # 3. Laplacian to show where cells are being pushed
    
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import plasma, magma, inferno
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    
    gradient_density = energy['gradient_density']
    strain_mag = energy['strain_magnitude']
    laplacian = energy['laplacian']
    
    # Create a "stress field" that extends into cells:
    # 1. Start with gradient energy at interfaces
    # 2. Diffuse it inward using distance transform weighted by strain
    
    # Mask for cell regions (where φ > 0.3)
    cell_mask = phi > 0.3
    cell_interior = phi > 0.8
    
    # Method: Create stress field by combining:
    # a) Interface energy (gradient)
    # b) Strain magnitude smoothed into cells
    # c) Distance from interface weighted by local curvature
    
    # Smooth the strain magnitude to show it extending into cells
    strain_smoothed = gaussian_filter(strain_mag, sigma=3)
    
    # Create interface stress that decays into cell
    # Distance from interface (φ=0.5)
    interface_region = (phi > 0.3) & (phi < 0.7)
    
    # Combine: interface energy + smoothed strain in interior
    interface_energy = np.sqrt(gradient_density)  # Compress range
    
    # Weight strain by distance from interface (decays into cell)
    # Use phi itself as a proxy - stress is highest at interface, decays toward center
    # stress ~ gradient at interface, strain deeper in
    phi_weight = np.clip(phi, 0, 1)
    edge_weight = 4 * phi_weight * (1 - phi_weight)  # Peaks at φ=0.5 (interface)
    interior_weight = phi_weight * (1 - edge_weight)  # Peaks inside cell
    
    # Combined stress field:
    # - At interface: dominated by gradient energy
    # - Inside cell: shows strain that propagates from compressed regions
    stress_field = (interface_energy * (0.3 + 0.7 * edge_weight) + 
                    strain_smoothed * interior_weight * 2)
    
    # Also add Laplacian contribution to show compression vs tension
    # Positive Laplacian = concave = being pushed in
    laplacian_smoothed = gaussian_filter(laplacian, sigma=2)
    compression = np.maximum(0, laplacian_smoothed) * phi_weight
    
    # Final stress visualization: normalize and apply colormap
    stress_for_vis = stress_field + compression * 0.5
    
    # Normalize
    vmax = np.percentile(stress_for_vis[cell_mask], 99) if np.any(cell_mask) else 1
    vmax = max(vmax, 0.01)  # Avoid division by zero
    stress_normalized = np.clip(stress_for_vis / vmax, 0, 1)
    
    # Create custom colormap: dark purple (low) -> orange -> yellow (high)
    # This shows stress intensity with good visibility
    cmap = inferno  # Dark background, bright stress
    stress_rgba = cmap(stress_normalized)
    
    # Make background (outside cells) dark
    outside_cells = phi < 0.2
    for i in range(3):
        stress_rgba[:,:,i] = np.where(outside_cells, 0.05, stress_rgba[:,:,i])
    
    im = ax_main.imshow(stress_rgba, origin='lower', 
                        extent=[0, dims[0], 0, dims[1]])
    
    # Add a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    # Colorbar
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Stress Intensity', fontsize=10)
    
    # Show overlap regions (φ > 1) if any
    X, Y = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))
    if np.max(phi) > 1.0:
        overlap_contours = ax_main.contour(X, Y, phi, levels=[1.0, 1.5], 
                                            colors=['yellow', 'red'], 
                                            linewidths=[1.5, 2.0], 
                                            linestyles=['--', '-'],
                                            alpha=0.9)
    
    # Polarization arrows (only if data provided)
    if polarization_data:
        for cell_id, (x, y, px, py) in polarization_data.items():
            ax_main.arrow(x, y, px * arrow_scale, py * arrow_scale,
                         head_width=3, head_length=2, fc='white', ec='black',
                         linewidth=1.0, alpha=0.9, zorder=10, clip_on=True)
    
    ax_main.set_xlim(0, dims[0])
    ax_main.set_ylim(0, dims[1])
    ax_main.set_aspect('equal')
    
    frame_num = int(Path(vtk_file).stem.split('_')[1])
    ax_main.set_title(f'Interface Energy (frame {frame_num})\n'
                      f'Total: {energy["total"]:.1f} | '
                      f'Interaction: {energy["interaction"]:.2f}', fontsize=11)
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    
    # === Energy time series (4 separate plots) ===
    if energy_history is not None and len(energy_history) > 0:
        # Only plot up to current frame (progressive drawing)
        times = [e['time'] for e in energy_history[:current_idx+1]]
        totals = [e['total'] for e in energy_history[:current_idx+1]]
        gradients = [e['gradient'] for e in energy_history[:current_idx+1]]
        bulks = [e['bulk'] for e in energy_history[:current_idx+1]]
        interactions = [e['interaction'] for e in energy_history[:current_idx+1]]
        
        # Get full time range for consistent x-axis
        all_times = [e['time'] for e in energy_history]
        t_min, t_max = min(all_times), max(all_times)
        
        # Get y-ranges for each component (from all data for consistency)
        # Skip first point to avoid initialization artifacts
        skip_n = min(1, len(energy_history) - 1)  # Don't skip more than we have
        all_totals = [e['total'] for e in energy_history]
        all_gradients = [e['gradient'] for e in energy_history]
        all_bulks = [e['bulk'] for e in energy_history]
        all_interactions = [e['interaction'] for e in energy_history]
        
        # For axis limits, use data after skipping first few frames
        totals_for_limits = all_totals[skip_n:] if len(all_totals) > skip_n else all_totals
        gradients_for_limits = all_gradients[skip_n:] if len(all_gradients) > skip_n else all_gradients
        bulks_for_limits = all_bulks[skip_n:] if len(all_bulks) > skip_n else all_bulks
        interactions_for_limits = all_interactions[skip_n:] if len(all_interactions) > skip_n else all_interactions
        
        # Plot each component with consistent formatting
        def setup_axis(ax, data, times_so_far, all_data, data_for_limits, color, label):
            ax.plot(times_so_far, data, '-', color=color, linewidth=1.5)
            if times_so_far:
                ax.scatter([times_so_far[-1]], [data[-1]], color=color, s=30, zorder=5)
                ax.axvline(times_so_far[-1], color='gray', linestyle=':', alpha=0.3)
            ax.set_xlim(t_min, t_max)
            # Use limits computed from data after skipping initial frames
            y_min, y_max = min(data_for_limits), max(data_for_limits)
            margin = (y_max - y_min) * 0.1 if y_max > y_min else 1
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_ylabel(label, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Force plain number formatting (no scientific notation offset)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='y')
        
        setup_axis(ax_total, totals, times, all_totals, totals_for_limits, 'black', 'Total')
        setup_axis(ax_gradient, gradients, times, all_gradients, gradients_for_limits, 'blue', 'Gradient')
        setup_axis(ax_bulk, bulks, times, all_bulks, bulks_for_limits, 'green', 'Bulk')
        setup_axis(ax_interaction, interactions, times, all_interactions, interactions_for_limits, 'red', 'Interaction')
        
        # Only show x-label on bottom plot
        ax_total.set_xticklabels([])
        ax_gradient.set_xticklabels([])
        ax_bulk.set_xticklabels([])
        ax_interaction.set_xlabel('Time', fontsize=9)
        
        # Title for the energy plots
        ax_total.set_title('Energy Components', fontsize=10)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return energy


def generate_energy_movie(output_dir, fps=10, lambda_=7.0, gamma=1.0, kappa=10.0,
                          show_polarization=True, arrow_scale=15, last_n=None):
    """Generate energy visualization movie.
    
    Args:
        last_n: If set, only process the last N frames
    
    If simulation was run with --save-individual-fields, uses pre-computed
    energy metrics (including accurate interaction energy).
    Otherwise, computes energy from combined VTK fields.
    
    First pass: compute/load energy for all frames (or last N)
    Second pass: generate images with progressive time series
    """
    output_dir = Path(output_dir)
    vtk_files = sorted(output_dir.glob('frame_*.vtk'), 
                       key=lambda x: int(x.stem.split('_')[1]))
    
    if not vtk_files:
        print(f"No VTK files found in {output_dir}")
        return
    
    # Filter to last N frames if requested
    if last_n is not None and last_n > 0:
        vtk_files = vtk_files[-last_n:]
        print(f"Processing last {len(vtk_files)} frames")
    
    print(f"Found {len(vtk_files)} VTK frames for energy visualization")
    
    # Load trajectory data for polarization
    trajectory_data = None
    if show_polarization:
        trajectory_data = load_trajectory_data(output_dir)
        if trajectory_data:
            print(f"Loaded trajectory data with {len(trajectory_data)} time samples")
    
    # Get dt
    dt = 0.01
    params_file = output_dir / 'params.txt'
    if params_file.exists():
        with open(params_file, 'r') as f:
            for line in f:
                if 'dt=' in line or 'dt =' in line:
                    try:
                        dt = float(line.split('=')[1].strip())
                    except:
                        pass
    
    # Check if simulation-computed energy metrics are available
    sim_energy = load_energy_metrics(output_dir)
    use_sim_energy = False
    
    if sim_energy:
        print(f"Found simulation-computed energy metrics ({len(sim_energy)} entries)")
        print("  Using accurate interaction energy from individual cell fields")
        use_sim_energy = True
        # Create lookup by frame number
        sim_energy_by_frame = {e['frame']: e for e in sim_energy}
    
    # First pass: compute/load energy for all frames
    print("Computing energy for all frames...")
    energy_history = []
    for vtk_file in vtk_files:
        dims, origin, spacing, cell_data = read_vtk_structured_points(vtk_file)
        phi = cell_data['phi']
        dx, dy = spacing[0], spacing[1]
        
        # Always compute from combined field for visualization (gradient, strain, etc.)
        energy = compute_energy_from_phi(phi, dx, dy, lambda_, gamma, kappa)
        frame_num = int(vtk_file.stem.split('_')[1])
        frame_time = frame_num * dt
        
        # If we have simulation-computed energy, use those values for time series
        if use_sim_energy and frame_num in sim_energy_by_frame:
            sim_e = sim_energy_by_frame[frame_num]
            energy_entry = {
                'time': sim_e['time'],
                'frame': frame_num,
                'total': sim_e['total'],
                'gradient': sim_e['gradient'],
                'bulk': sim_e['bulk'],
                'interaction': sim_e['interaction']
            }
        else:
            energy_entry = {
                'time': frame_time,
                'frame': frame_num,
                'total': energy['total'],
                'gradient': energy['gradient'],
                'bulk': energy['bulk'],
                'interaction': energy['interaction']
            }
        
        energy_history.append(energy_entry)
    
    print(f"Energy range: {min(e['total'] for e in energy_history):.1f} - "
          f"{max(e['total'] for e in energy_history):.1f}")
    
    # Save energy data to file
    energy_file = output_dir / 'energy.txt'
    with open(energy_file, 'w') as f:
        f.write("# Energy data\n")
        f.write("# time frame total gradient bulk interaction\n")
        for e in energy_history:
            f.write(f"{e['time']:.4f} {e['frame']} {e['total']:.6f} "
                   f"{e['gradient']:.6f} {e['bulk']:.6f} {e['interaction']:.6f}\n")
    print(f"Saved energy data: {energy_file}")
    
    # Create energy images directory (clear old images first)
    img_dir = output_dir / 'energy_images'
    if img_dir.exists():
        # Remove old images
        for old_img in img_dir.glob('*.png'):
            old_img.unlink()
    img_dir.mkdir(exist_ok=True)
    
    # Track generated image files
    generated_images = []
    
    # Second pass: generate images with progressive time series
    print("Generating energy visualization frames...")
    for i, vtk_file in enumerate(vtk_files):
        print(f"  Processing {vtk_file.name}...")
        
        img_file = img_dir / f"{vtk_file.stem}.png"
        
        # Get polarization data
        polarization = None
        if trajectory_data:
            frame_num = int(vtk_file.stem.split('_')[1])
            frame_time = frame_num * dt
            polarization = get_polarization_for_frame(trajectory_data, frame_time)
        
        try:
            plot_energy_frame(vtk_file, output_file=img_file, show=False,
                             energy_history=energy_history, current_idx=i,
                             polarization_data=polarization, arrow_scale=arrow_scale,
                             lambda_=lambda_, gamma=gamma, kappa=kappa)
            generated_images.append(img_file)
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Create movie from generated images only
    movie_path = output_dir / 'energy_movie.mp4'
    img_files = sorted(generated_images, key=lambda x: int(x.stem.split('_')[1]))
    
    if not img_files:
        print("No energy images generated")
        return
    
    print(f"Creating energy movie from {len(img_files)} images...")
    
    try:
        import imageio.v2 as imageio
        from PIL import Image
        
        first_img = Image.open(img_files[0])
        target_size = first_img.size
        first_img.close()
        
        # Make dimensions divisible by 16
        target_size = ((target_size[0] // 16) * 16, (target_size[1] // 16) * 16)
        
        with imageio.get_writer(str(movie_path), fps=fps, macro_block_size=1) as writer:
            for img_file in img_files:
                img = Image.open(img_file)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                writer.append_data(np.array(img))
        
        print(f"Saved energy movie: {movie_path}")
        
    except ImportError:
        print("imageio not available - install with: pip install imageio imageio-ffmpeg")
    except Exception as e:
        print(f"Error creating movie: {e}")


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    parser = argparse.ArgumentParser(description='Visualize cell simulation VTK output')
    parser.add_argument('output_dir', nargs='?', default='test_output',
                        help='Directory containing VTK files')
    parser.add_argument('-r', '--reverse', action='store_true',
                        help='Process frames in reverse order (last to first)')
    parser.add_argument('-f', '--frame', type=int, default=None,
                        help='Process only a specific frame number')
    parser.add_argument('--start', type=int, default=None,
                        help='Start frame number for range')
    parser.add_argument('--end', type=int, default=None,
                        help='End frame number for range')
    parser.add_argument('--last', type=int, default=None,
                        help='Process only the last N frames')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save PNG images')
    parser.add_argument('--movie', action='store_true',
                        help='Create a movie from the generated images')
    parser.add_argument('--movie-only', action='store_true',
                        help='Create a movie from existing images (skip image generation)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for movie (default: 10)')
    parser.add_argument('--use-arrows', action='store_true',
                        help='Show polarization arrows on cells')
    parser.add_argument('--arrow-scale', type=float, default=15,
                        help='Scale factor for polarization arrows (default: 15)')
    parser.add_argument('--energy', action='store_true',
                        help='Generate energy visualization movie (contours + heatmap + time series)')
    parser.add_argument('--energy-only', action='store_true',
                        help='Generate energy movie from existing data (skip recomputation)')
    
    args = parser.parse_args()
    
    print(f"Visualizing output from: {args.output_dir}")
    
    # Determine frame range
    frame_range = None
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else 999999
        frame_range = (start, end)
    
    # Handle --last option
    if args.last is not None:
        output_dir = Path(args.output_dir)
        vtk_files = sorted(output_dir.glob('frame_*.vtk'))
        if vtk_files:
            last_frames = vtk_files[-args.last:]
            start_frame = int(last_frames[0].stem.split('_')[1])
            end_frame = int(last_frames[-1].stem.split('_')[1])
            frame_range = (start_frame, end_frame)
            print(f"Processing last {args.last} frames")
    
    # Energy visualization mode
    if args.energy or args.energy_only:
        print("Generating energy visualization...")
        generate_energy_movie(args.output_dir, fps=args.fps,
                             show_polarization=args.use_arrows,
                             arrow_scale=args.arrow_scale,
                             last_n=args.last)
        # Exit after energy visualization (don't do normal viz)
        sys.exit(0)
    
    # Plot frames (skip if --movie-only)
    if not args.movie_only:
        plot_all_frames(args.output_dir, 
                        save_images=not args.no_save, 
                        reverse=args.reverse,
                        frame=args.frame,
                        frame_range=frame_range,
                        show_polarization=args.use_arrows,
                        arrow_scale=args.arrow_scale)
    
    # Create movie if requested
    if args.movie or args.movie_only:
        output_dir = Path(args.output_dir)
        img_dir = output_dir / 'images'
        movie_path = output_dir / 'simulation.mp4'
        
        # Get sorted list of image files (format: frame_N.png)
        img_files = sorted(img_dir.glob('frame_*.png'), key=lambda x: int(x.stem.split('_')[1]))
        
        if not img_files:
            print("No images found to create movie")
        else:
            print(f"Creating movie from {len(img_files)} images...")
            
            try:
                import imageio.v2 as imageio
                from PIL import Image
                
                # Get the size of the first image for consistent sizing
                first_img = Image.open(img_files[0])
                target_size = first_img.size
                first_img.close()
                
                # Make dimensions divisible by 16 for codec compatibility
                target_size = (
                    (target_size[0] // 16) * 16,
                    (target_size[1] // 16) * 16
                )
                
                # Use imageio to create movie
                with imageio.get_writer(str(movie_path), fps=args.fps, macro_block_size=1) as writer:
                    for img_file in img_files:
                        img = Image.open(img_file)
                        # Resize to consistent dimensions
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        image = np.array(img)
                        writer.append_data(image)
                print(f"Saved movie: {movie_path}")
                
            except ImportError:
                # Fallback to ffmpeg via subprocess
                import subprocess
                try:
                    # Use ffmpeg to create movie from numbered images
                    cmd = [
                        'ffmpeg', '-y',
                        '-framerate', str(args.fps),
                        '-pattern_type', 'glob',
                        '-i', str(img_dir / '*.png'),
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        str(movie_path)
                    ]
                    subprocess.run(cmd, check=True)
                    print(f"Saved movie: {movie_path}")
                except FileNotFoundError:
                    print("Error: Neither imageio nor ffmpeg available for movie creation")
                    print("Install imageio: pip install imageio imageio-ffmpeg")
                except subprocess.CalledProcessError as e:
                    print(f"Error creating movie with ffmpeg: {e}")
