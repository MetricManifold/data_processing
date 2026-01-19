"""
Stress field visualization for cell simulation output.
Reads VTK files containing stress tensor components and visualizes them.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from scipy.ndimage import gaussian_filter
from pathlib import Path
import sys


class PowerLawNorm(Normalize):
    """Non-linear normalization using power law: emphasizes low values when gamma < 1."""
    def __init__(self, vmin=None, vmax=None, gamma=0.5, clip=False):
        self.gamma = gamma
        super().__init__(vmin, vmax, clip)
    
    def __call__(self, value, clip=None):
        # Normalize to [0, 1] first
        result = super().__call__(value, clip)
        # Apply power law transformation
        return np.power(result, self.gamma)
    
    def inverse(self, value):
        return super().inverse(np.power(value, 1.0/self.gamma))


class SymmetricPowerNorm(Normalize):
    """Symmetric power-law norm for diverging colormaps (like pressure)."""
    def __init__(self, vmin=None, vmax=None, gamma=0.6, clip=False):
        self.gamma = gamma
        super().__init__(vmin, vmax, clip)
    
    def __call__(self, value, clip=None):
        # Map to [-1, 1] centered at 0
        result = np.ma.masked_array(value)
        vmin, vmax = self.vmin, self.vmax
        
        # Scale to [-1, 1]
        result = 2.0 * (result - vmin) / (vmax - vmin) - 1.0
        
        # Apply signed power law (preserves sign, compresses middle)
        sign = np.sign(result)
        result = sign * np.power(np.abs(result), self.gamma)
        
        # Map back to [0, 1]
        return np.clip((result + 1.0) / 2.0, 0, 1)


# Custom colormaps for stress visualization
def create_stress_colormaps():
    """Create beautiful multi-color colormaps for stress fields."""
    
    # Von Mises: Black -> Deep Purple -> Blue -> Cyan -> Green -> Yellow -> Orange -> Red -> White
    # More color stops for smoother gradients
    vm_colors = [
        (0.0, '#050510'),   # Near black with hint of blue
        (0.05, '#1a0a30'),  # Very dark purple
        (0.12, '#2d1b4e'), # Dark purple
        (0.2, '#1e3a5f'),   # Dark blue
        (0.3, '#0066cc'),   # Blue
        (0.4, '#00b4d8'),   # Cyan
        (0.5, '#00cc66'),   # Green
        (0.62, '#99e600'),  # Yellow-green
        (0.72, '#ffcc00'),  # Yellow
        (0.82, '#ff6600'),  # Orange
        (0.92, '#ff0000'),  # Red
        (1.0, '#ffffff'),   # White (extreme)
    ]
    
    # Shear stress: DRAMATICALLY improved colormap
    # Black -> Deep violet -> Electric blue -> Cyan -> Lime -> Yellow -> Orange -> Hot pink -> White
    # This gives much better color differentiation across the range
    shear_colors = [
        (0.0, '#000008'),   # Near black
        (0.08, '#0f0030'),  # Very dark violet
        (0.16, '#1a0060'),  # Dark violet
        (0.24, '#2000a0'),  # Violet
        (0.32, '#0040ff'),  # Electric blue
        (0.40, '#00a0ff'),  # Sky blue  
        (0.48, '#00e0e0'),  # Cyan
        (0.56, '#00ff80'),  # Cyan-green
        (0.64, '#80ff00'),  # Lime
        (0.72, '#ffff00'),  # Yellow
        (0.80, '#ffa000'),  # Orange
        (0.88, '#ff4080'),  # Hot pink
        (0.94, '#ff80c0'),  # Light pink
        (1.0, '#ffffff'),   # White (extreme)
    ]
    
    # Pressure: Blue (compression) -> White (neutral) -> Red (tension)
    # More colors for better visualization
    pressure_colors = [
        (0.0, '#00008b'),   # Dark blue (high compression)
        (0.15, '#0066cc'),  # Blue
        (0.3, '#00b4d8'),   # Cyan
        (0.45, '#90e0ef'),  # Light cyan
        (0.5, '#f8f8f8'),   # Near white (neutral)
        (0.55, '#ffc8c8'),  # Light red
        (0.7, '#ff6b6b'),   # Salmon
        (0.85, '#cc0000'),  # Red
        (1.0, '#8b0000'),   # Dark red (high tension)
    ]
    
    # Normal stress components: similar diverging but with green-yellow center
    normal_colors = [
        (0.0, '#0a2463'),   # Dark blue
        (0.2, '#3e92cc'),   # Blue
        (0.35, '#88d498'),  # Light green
        (0.5, '#f8f8f8'),   # Near white
        (0.65, '#f9dc5c'),  # Yellow
        (0.8, '#ed7d3a'),   # Orange
        (1.0, '#8b0000'),   # Dark red
    ]
    
    def make_cmap(colors, name):
        positions = [c[0] for c in colors]
        hex_colors = [c[1] for c in colors]
        rgb_colors = [tuple(int(h.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) for h in hex_colors]
        return LinearSegmentedColormap.from_list(name, list(zip(positions, rgb_colors)))
    
    return {
        'von_mises': make_cmap(vm_colors, 'von_mises'),
        'shear': make_cmap(shear_colors, 'shear'),
        'pressure': make_cmap(pressure_colors, 'pressure'),
        'normal': make_cmap(normal_colors, 'normal'),
    }

STRESS_CMAPS = create_stress_colormaps()


def add_glow_effect(data, mask, cmap=None, norm=None, threshold_percentile=70, glow_sigma=4, glow_intensity=1.5,
                    multi_layer=True):
    """Add a dramatic glow effect to high-stress regions.
    
    Uses multiple blur layers with different sigma values to create
    a more realistic bloom/glow effect that scales with stress intensity.
    
    Args:
        data: 2D stress field array
        mask: boolean mask of valid (non-zero) regions
        cmap: optional colormap to use for glow colors (defaults to white/yellow)
        norm: normalization function if cmap is provided
        threshold_percentile: percentile above which to apply glow (lower = more glow)
        glow_sigma: base sigma for gaussian blur
        glow_intensity: overall glow brightness multiplier
        multi_layer: whether to use multi-layer glow for bloom effect
    
    Returns an RGBA array with glow overlay.
    """
    # Find high stress regions
    valid_data = data[mask]
    if len(valid_data) == 0:
        return np.zeros((*data.shape, 4))
    
    threshold = np.nanpercentile(valid_data, threshold_percentile)
    max_val = np.nanmax(valid_data)
    
    # Create intensity-scaled glow mask (higher values = stronger glow)
    # Normalized to [0, 1] based on how far above threshold
    intensity = np.clip((data - threshold) / (max_val - threshold + 1e-10), 0, 1)
    intensity[~mask] = 0
    
    # Boost the intensity curve to make high-stress regions pop more
    intensity = np.power(intensity, 0.5)  # gamma < 1 expands bright regions
    
    if multi_layer:
        # Multi-layer glow creates more natural bloom effect
        # Layer 1: Tight, bright core glow
        glow1 = gaussian_filter(intensity, sigma=glow_sigma * 0.5)
        # Layer 2: Medium spread 
        glow2 = gaussian_filter(intensity, sigma=glow_sigma * 1.0)
        # Layer 3: Wide, soft halo
        glow3 = gaussian_filter(intensity, sigma=glow_sigma * 2.5)
        
        # Combine with decreasing weights for outer layers
        glow = glow1 * 1.0 + glow2 * 0.5 + glow3 * 0.25
    else:
        glow = gaussian_filter(intensity, sigma=glow_sigma)
    
    # Normalize and apply intensity
    if glow.max() > 0:
        glow = glow / glow.max()
    glow = np.clip(glow * glow_intensity, 0, 1)
    
    # Create RGBA glow layer
    glow_rgba = np.zeros((*data.shape, 4))
    
    if cmap is not None and norm is not None:
        # Use colormap-based glow (glow in stress colors)
        data_norm = norm(data)
        if hasattr(data_norm, 'filled'):
            data_norm = data_norm.filled(0)
        glow_colors = cmap(data_norm)
        
        # Brighten the colors for glow effect
        glow_rgba[:, :, 0] = np.clip(glow_colors[:, :, 0] * 1.4, 0, 1)
        glow_rgba[:, :, 1] = np.clip(glow_colors[:, :, 1] * 1.2, 0, 1) 
        glow_rgba[:, :, 2] = np.clip(glow_colors[:, :, 2] * 1.4, 0, 1)
    else:
        # Default white/yellow glow
        glow_rgba[:, :, 0] = 1.0  # R
        glow_rgba[:, :, 1] = 0.95  # G (slight yellow tint)
        glow_rgba[:, :, 2] = 0.7  # B
    
    glow_rgba[:, :, 3] = glow  # Alpha from glow intensity
    
    return glow_rgba


def read_vtk_all_scalars(filename):
    """Read all scalar fields from a legacy VTK structured points file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
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
            break
        i += 1
    
    if dims is None:
        raise ValueError("Could not parse VTK dimensions")
    
    # Now read all scalar fields
    scalars = {}
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('SCALARS'):
            parts = line.split()
            field_name = parts[1]
            
            # Skip LOOKUP_TABLE line
            i += 1
            if i < len(lines) and lines[i].strip().startswith('LOOKUP_TABLE'):
                i += 1
            
            # Read data
            values = []
            while len(values) < n_points and i < len(lines):
                data_line = lines[i].strip()
                if data_line and not data_line.startswith(('SCALARS', 'VECTORS', 'LOOKUP_TABLE')):
                    values.extend([float(x) for x in data_line.split()])
                    i += 1
                elif data_line.startswith('SCALARS') or data_line.startswith('VECTORS'):
                    break
                else:
                    i += 1
            
            # Reshape to 2D
            data = np.array(values[:n_points]).reshape((dims[1], dims[0]))
            scalars[field_name] = data
        else:
            i += 1
    
    return dims, origin, spacing, scalars


def compute_principal_stresses(sigma_xx, sigma_yy, sigma_xy):
    """Compute principal stresses and maximum shear stress from stress tensor."""
    # Mean stress (hydrostatic)
    sigma_mean = 0.5 * (sigma_xx + sigma_yy)
    
    # Deviatoric components
    diff = sigma_xx - sigma_yy
    
    # Principal stresses: eigenvalues of 2D stress tensor
    # σ₁,₂ = (σxx + σyy)/2 ± √[(σxx - σyy)²/4 + σxy²]
    discriminant = np.sqrt(0.25 * diff**2 + sigma_xy**2)
    
    sigma_1 = sigma_mean + discriminant  # Major principal stress
    sigma_2 = sigma_mean - discriminant  # Minor principal stress
    
    # Maximum shear stress
    tau_max = discriminant  # = (σ₁ - σ₂) / 2
    
    # von Mises stress (2D plane stress)
    # σ_vm = √(σ₁² - σ₁σ₂ + σ₂²)
    sigma_vm = np.sqrt(sigma_1**2 - sigma_1*sigma_2 + sigma_2**2)
    
    return sigma_1, sigma_2, tau_max, sigma_vm


def plot_stress_fields(vtk_file, output_file=None, show=True, add_glow=True):
    """Plot stress tensor components and derived quantities with beautiful colormaps and glow."""
    dims, origin, spacing, scalars = read_vtk_all_scalars(vtk_file)
    
    # Check if stress fields exist
    if 'sigma_xx' not in scalars:
        print("No stress fields found in VTK file")
        return None
    
    phi = scalars.get('phi', np.ones((dims[1], dims[0])))
    sigma_xx = scalars['sigma_xx']
    sigma_yy = scalars['sigma_yy']
    sigma_xy = scalars['sigma_xy']
    pressure = scalars.get('pressure', -0.5*(sigma_xx + sigma_yy))
    
    # Compute derived quantities
    sigma_1, sigma_2, tau_max, sigma_vm = compute_principal_stresses(sigma_xx, sigma_yy, sigma_xy)
    
    # Create mask for cell regions (where phi > 0.1)
    cell_mask = phi > 0.1
    
    # Create figure with dark background for better glow visibility
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#1a1a2e')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a2e')
    
    extent = [0, dims[0], 0, dims[1]]
    
    # Mask for outside cells (set to NaN for transparent)
    def masked_field(field):
        masked = field.copy()
        masked[~cell_mask] = np.nan
        return masked
    
    def plot_with_glow(ax, data, cmap, title, is_diverging=False, vmin=None, vmax=None,
                        use_power_norm=False, gamma=0.5):
        """Plot a field with optional glow effect.
        
        Args:
            ax: matplotlib axis
            data: 2D data array
            cmap: colormap to use
            title: plot title
            is_diverging: whether this is a diverging colormap (centered at 0)
            vmin, vmax: color limits (auto-computed if None)
            use_power_norm: use power-law normalization to expand low values
            gamma: gamma value for power norm (< 1 expands low values)
        """
        masked = masked_field(data)
        
        # Determine color limits
        if vmin is None or vmax is None:
            if is_diverging:
                vmax_calc = np.nanpercentile(np.abs(masked), 99)
                vmin, vmax = -vmax_calc, vmax_calc
            else:
                vmin = 0
                vmax = np.nanpercentile(masked, 99)
        
        # Choose normalization
        if use_power_norm and not is_diverging:
            norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
        elif is_diverging:
            norm = SymmetricPowerNorm(vmin=vmin, vmax=vmax, gamma=0.7)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Plot the base stress field
        im = ax.imshow(masked, origin='lower', extent=extent, cmap=cmap, 
                       norm=norm, interpolation='bilinear')
        
        # Add glow effect for high stress regions
        if add_glow and not is_diverging:
            glow = add_glow_effect(data, cell_mask, cmap=cmap, norm=norm,
                                   threshold_percentile=70, 
                                   glow_sigma=5, glow_intensity=1.2, multi_layer=True)
            ax.imshow(glow, origin='lower', extent=extent, interpolation='bilinear')
        elif add_glow and is_diverging:
            # For diverging, glow on both extremes
            glow_pos = add_glow_effect(data, cell_mask & (data > 0), 
                                       threshold_percentile=85, glow_sigma=3, glow_intensity=0.3)
            glow_neg = add_glow_effect(-data, cell_mask & (data < 0), 
                                       threshold_percentile=85, glow_sigma=3, glow_intensity=0.3)
            # Tint negative glow blue
            glow_neg[:, :, 0] = 0.5  # Less red
            glow_neg[:, :, 2] = 1.0  # More blue
            ax.imshow(glow_pos, origin='lower', extent=extent, interpolation='bilinear')
            ax.imshow(glow_neg, origin='lower', extent=extent, interpolation='bilinear')
        
        # Cell boundaries (bright cyan for visibility on dark background)
        ax.contour(phi, levels=[0.5], colors='#00ffff', linewidths=0.8, 
                   extent=extent, alpha=0.7)
        
        ax.set_title(title, fontsize=11, color='white', fontweight='bold')
        ax.set_xlabel('x', color='white')
        ax.set_ylabel('y', color='white')
        ax.tick_params(colors='white')
        ax.set_xlim(0, dims[0])
        ax.set_ylim(0, dims[1])
        ax.set_aspect('equal')
        
        # Colorbar with white text
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        return im
    
    # Plot 1: von Mises stress (power norm to show low values better)
    plot_with_glow(axes[0, 0], sigma_vm, STRESS_CMAPS['von_mises'], 
                   'von Mises Stress σ_vm', is_diverging=False,
                   use_power_norm=True, gamma=0.5)
    
    # Plot 2: Maximum shear stress (power norm critical for cell interiors being ~0)
    plot_with_glow(axes[0, 1], tau_max, STRESS_CMAPS['shear'], 
                   'Max Shear Stress τ_max', is_diverging=False,
                   use_power_norm=True, gamma=0.4)  # Lower gamma = more color for low values
    
    # Plot 3: Pressure
    plot_with_glow(axes[0, 2], pressure, STRESS_CMAPS['pressure'], 
                   'Pressure p', is_diverging=True)
    
    # Plot 4: σ_xx
    plot_with_glow(axes[1, 0], sigma_xx, STRESS_CMAPS['normal'], 
                   'Normal Stress σ_xx', is_diverging=True)
    
    # Plot 5: σ_yy
    plot_with_glow(axes[1, 1], sigma_yy, STRESS_CMAPS['normal'], 
                   'Normal Stress σ_yy', is_diverging=True)
    
    # Plot 6: σ_xy
    plot_with_glow(axes[1, 2], sigma_xy, STRESS_CMAPS['normal'], 
                   'Shear Stress σ_xy', is_diverging=True)
    
    # Extract frame number and time
    frame_num = int(Path(vtk_file).stem.split('_')[1])
    fig.suptitle(f'Stress Field Analysis (Frame {frame_num})', 
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Saved: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return scalars


def plot_stress_single(vtk_file, field='von_mises', output_file=None, show=True):
    """Plot a single stress field (large format suitable for presentations)."""
    dims, origin, spacing, scalars = read_vtk_all_scalars(vtk_file)
    
    if 'sigma_xx' not in scalars:
        print("No stress fields found")
        return
    
    phi = scalars.get('phi', np.ones((dims[1], dims[0])))
    sigma_xx = scalars['sigma_xx']
    sigma_yy = scalars['sigma_yy']
    sigma_xy = scalars['sigma_xy']
    pressure = scalars.get('pressure', -0.5*(sigma_xx + sigma_yy))
    
    sigma_1, sigma_2, tau_max, sigma_vm = compute_principal_stresses(sigma_xx, sigma_yy, sigma_xy)
    
    cell_mask = phi > 0.1
    
    # Select field to plot
    field_map = {
        'von_mises': (sigma_vm, 'hot', 'von Mises Stress σ_vm'),
        'tau_max': (tau_max, 'plasma', 'Maximum Shear Stress τ_max'),
        'pressure': (pressure, 'RdBu_r', 'Pressure p'),
        'sigma_xx': (sigma_xx, 'RdBu_r', 'Normal Stress σ_xx'),
        'sigma_yy': (sigma_yy, 'RdBu_r', 'Normal Stress σ_yy'),
        'sigma_xy': (sigma_xy, 'RdBu_r', 'Shear Stress σ_xy'),
        'sigma_1': (sigma_1, 'RdBu_r', 'Major Principal Stress σ₁'),
        'sigma_2': (sigma_2, 'RdBu_r', 'Minor Principal Stress σ₂'),
    }
    
    if field not in field_map:
        print(f"Unknown field: {field}")
        print(f"Available: {list(field_map.keys())}")
        return
    
    data, cmap, title = field_map[field]
    
    # Mask outside cells
    masked_data = data.copy()
    masked_data[~cell_mask] = np.nan
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    extent = [0, dims[0], 0, dims[1]]
    
    # Determine color limits
    if cmap == 'hot' or cmap == 'plasma':
        vmin = 0
        vmax = np.nanpercentile(masked_data, 99)
    else:
        vmax = np.nanpercentile(np.abs(masked_data), 99)
        vmin = -vmax
    
    im = ax.imshow(masked_data, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Cell boundaries
    ax.contour(phi, levels=[0.5], colors='cyan' if cmap in ['hot', 'plasma'] else 'black', 
               linewidths=1, extent=extent)
    
    frame_num = int(Path(vtk_file).stem.split('_')[1])
    ax.set_title(f'{title} (Frame {frame_num})', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(title)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_stress_movie(output_dir, field='von_mises', fps=10, last_n=None, movie_only=False):
    """Generate a movie showing stress field evolution.
    
    Args:
        output_dir: Directory containing VTK files
        field: Which stress field to visualize
        fps: Frames per second
        last_n: Only process last N frames
        movie_only: If True, skip image generation and just create movie from existing images
    """
    output_dir = Path(output_dir)
    vtk_files = sorted(output_dir.glob('frame_*.vtk'), 
                       key=lambda x: int(x.stem.split('_')[1]))
    
    if not vtk_files:
        print(f"No VTK files found in {output_dir}")
        return
    
    # Check if stress fields exist
    dims, _, _, scalars = read_vtk_all_scalars(vtk_files[0])
    if 'sigma_xx' not in scalars:
        print("No stress fields in VTK files - run simulation with --stress-fields")
        return
    
    if last_n is not None:
        vtk_files = vtk_files[-last_n:]
    
    print(f"Found {len(vtk_files)} VTK frames")
    
    # Create stress images directory
    img_dir = output_dir / f'stress_images_{field}'
    img_dir.mkdir(exist_ok=True)
    
    # Generate images (unless movie_only and images exist)
    generated = []
    if movie_only:
        # Just look for existing images
        generated = sorted(img_dir.glob('*.png'))
        if generated:
            print(f"Movie-only mode: using {len(generated)} existing images")
        else:
            print("No existing images found, generating them...")
            movie_only = False  # Fall through to generation
    
    if not movie_only:
        # Clear old images
        for old_img in img_dir.glob('*.png'):
            old_img.unlink()
        
        print(f"Generating {field} stress visualization...")
        for vtk_file in vtk_files:
            print(f"  Processing {vtk_file.name}...")
            img_file = img_dir / f"{vtk_file.stem}.png"
            try:
                plot_stress_single(vtk_file, field=field, output_file=img_file, show=False)
                generated.append(img_file)
            except Exception as e:
                print(f"    Error: {e}")
    
    # Create movie
    movie_path = output_dir / f'stress_{field}_movie.mp4'
    print(f"Creating movie from {len(generated)} images...")
    
    try:
        import imageio.v2 as imageio
        from PIL import Image
        
        first_img = Image.open(generated[0])
        target_size = first_img.size
        target_size = ((target_size[0] // 16) * 16, (target_size[1] // 16) * 16)
        first_img.close()
        
        with imageio.get_writer(str(movie_path), fps=fps, macro_block_size=1) as writer:
            for img_file in generated:
                img = Image.open(img_file)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                writer.append_data(np.array(img))
        
        print(f"Saved movie: {movie_path}")
    except ImportError:
        print("imageio not available - install with: pip install imageio imageio-ffmpeg")


def generate_combined_stress_movie(output_dir, fps=10, last_n=None, add_glow=True, movie_only=False):
    """Generate a movie showing ALL stress fields in a 2x3 grid with glow effects.
    
    Args:
        output_dir: Directory containing VTK files
        fps: Frames per second
        last_n: Only process last N frames
        add_glow: Add glow effects to high-stress regions
        movie_only: If True, skip image generation and just create movie from existing images
    """
    output_dir = Path(output_dir)
    vtk_files = sorted(output_dir.glob('frame_*.vtk'), 
                       key=lambda x: int(x.stem.split('_')[1]))
    
    if not vtk_files:
        print(f"No VTK files found in {output_dir}")
        return
    
    # Check if stress fields exist (only if we need to generate images)
    if not movie_only:
        dims, _, _, scalars = read_vtk_all_scalars(vtk_files[0])
        if 'sigma_xx' not in scalars:
            print("No stress fields in VTK files - run simulation with --stress-fields")
            return
    
    if last_n is not None:
        vtk_files = vtk_files[-last_n:]
    
    print(f"Found {len(vtk_files)} VTK frames")
    
    # Create combined stress images directory
    img_dir = output_dir / 'stress_images_combined'
    img_dir.mkdir(exist_ok=True)
    
    # Generate images (unless movie_only and images exist)
    generated = []
    if movie_only:
        # Just look for existing images
        generated = sorted(img_dir.glob('*.png'))
        if generated:
            print(f"Movie-only mode: using {len(generated)} existing images")
        else:
            print("No existing images found, generating them...")
            movie_only = False  # Fall through to generation
    
    if not movie_only:
        # Clear old images
        for old_img in img_dir.glob('*.png'):
            old_img.unlink()
        
        print("Generating combined stress visualization with glow effects...")
        for vtk_file in vtk_files:
            print(f"  Processing {vtk_file.name}...")
            img_file = img_dir / f"{vtk_file.stem}.png"
            try:
                plot_stress_fields(vtk_file, output_file=img_file, show=False, add_glow=add_glow)
                generated.append(img_file)
            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Create movie
    movie_path = output_dir / 'stress_combined_movie.mp4'
    print(f"Creating combined movie from {len(generated)} images...")
    
    try:
        import imageio.v2 as imageio
        from PIL import Image
        
        first_img = Image.open(generated[0])
        target_size = first_img.size
        target_size = ((target_size[0] // 16) * 16, (target_size[1] // 16) * 16)
        first_img.close()
        
        with imageio.get_writer(str(movie_path), fps=fps, macro_block_size=1) as writer:
            for img_file in generated:
                img = Image.open(img_file)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                writer.append_data(np.array(img))
        
        print(f"Saved combined movie: {movie_path}")
    except ImportError:
        print("imageio not available - install with: pip install imageio imageio-ffmpeg")


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    
    parser = argparse.ArgumentParser(description='Visualize stress fields from cell simulation')
    parser.add_argument('output_dir', nargs='?', default='output',
                        help='Directory containing VTK files')
    parser.add_argument('-f', '--frame', type=int, default=None,
                        help='Specific frame number to visualize')
    parser.add_argument('--field', default='von_mises',
                        choices=['von_mises', 'tau_max', 'pressure', 'sigma_xx', 
                                 'sigma_yy', 'sigma_xy', 'sigma_1', 'sigma_2'],
                        help='Stress field to visualize (default: von_mises)')
    parser.add_argument('--all-fields', action='store_true',
                        help='Show all stress fields in a single figure')
    parser.add_argument('--movie', action='store_true',
                        help='Generate a movie of stress evolution')
    parser.add_argument('--combined', action='store_true',
                        help='Generate combined 6-panel movie with all stress fields')
    parser.add_argument('--movie-only', action='store_true',
                        help='Skip image generation if images exist, just create movie')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for movie')
    parser.add_argument('--last', type=int, default=None,
                        help='Only process the last N frames')
    parser.add_argument('--no-glow', action='store_true',
                        help='Disable glow effects on high-stress regions')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.combined or (args.movie_only and not args.movie):
        # Combined 6-panel movie (default for --movie-only without --movie)
        generate_combined_stress_movie(output_dir, fps=args.fps, last_n=args.last, 
                                       add_glow=not args.no_glow, 
                                       movie_only=args.movie_only)
    elif args.movie:
        # Single field movie
        generate_stress_movie(output_dir, field=args.field, fps=args.fps, last_n=args.last,
                              movie_only=args.movie_only)
    elif args.frame is not None:
        vtk_file = output_dir / f'frame_{args.frame:06d}.vtk'
        if not vtk_file.exists():
            print(f"Frame not found: {vtk_file}")
            sys.exit(1)
        
        if args.all_fields:
            plot_stress_fields(vtk_file, show=False, add_glow=not args.no_glow,
                              output_file=output_dir / f'stress_all_{args.frame:06d}.png')
        else:
            plot_stress_single(vtk_file, field=args.field, show=False,
                              output_file=output_dir / f'stress_{args.field}_{args.frame:06d}.png')
    else:
        # Default: show last frame with all fields
        vtk_files = sorted(output_dir.glob('frame_*.vtk'))
        if vtk_files:
            last_vtk = vtk_files[-1]
            frame_num = int(last_vtk.stem.split('_')[1])
            print(f"Visualizing frame {frame_num}")
            
            if args.all_fields:
                plot_stress_fields(last_vtk, show=False, add_glow=not args.no_glow,
                                  output_file=output_dir / f'stress_all_{frame_num:06d}.png')
            else:
                plot_stress_single(last_vtk, field=args.field, show=False,
                                  output_file=output_dir / f'stress_{args.field}_{frame_num:06d}.png')
        else:
            print(f"No VTK files found in {output_dir}")
