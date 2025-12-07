#!/usr/bin/env python3
"""
3D Cell Simulation Visualization using PyVista
Reads binary checkpoint files and renders 3D isosurfaces of cells.
"""

import numpy as np
import struct
import os
import glob
import argparse

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. Install with: pip install pyvista")

# Checkpoint file format constants
MAGIC_3D = b"CS3D"
VERSION_3D = 1


def read_checkpoint_3d(filename):
    """Read a 3D checkpoint file and return simulation data."""
    with open(filename, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != MAGIC_3D:
            raise ValueError(f"Invalid magic: {magic}")
        
        version = struct.unpack('i', f.read(4))[0]
        if version != VERSION_3D:
            raise ValueError(f"Unsupported version: {version}")
        
        step = struct.unpack('i', f.read(4))[0]
        time = struct.unpack('f', f.read(4))[0]
        num_cells = struct.unpack('i', f.read(4))[0]
        
        # Read SimParams3D (must match C++ struct exactly)
        # SimParams3D layout (21 fields, 84 bytes total):
        #   int Nx, Ny, Nz                          (3 ints = 12 bytes)
        #   float dx, dy, dz                        (3 floats = 12 bytes)
        #   float dt, t_end                         (2 floats = 8 bytes)
        #   int save_interval                       (1 int = 4 bytes)
        #   float lambda, gamma                     (2 floats = 8 bytes)
        #   float kappa                             (1 float = 4 bytes)
        #   float target_radius, mu                 (2 floats = 8 bytes)
        #   float v_A, xi, tau                      (3 floats = 12 bytes)
        #   int halo_width, min_subdomain_size      (2 ints = 8 bytes)
        #   float subdomain_padding                 (1 float = 4 bytes)
        #   int motility_model                      (1 int = 4 bytes)
        # Total: 84 bytes
        
        Nx = struct.unpack('i', f.read(4))[0]
        Ny = struct.unpack('i', f.read(4))[0]
        Nz = struct.unpack('i', f.read(4))[0]
        dx = struct.unpack('f', f.read(4))[0]
        dy = struct.unpack('f', f.read(4))[0]
        dz = struct.unpack('f', f.read(4))[0]
        # Skip remaining params: 15 more fields (11 floats + 4 ints = 60 bytes)
        f.read(4 * 15)
        
        params = {
            'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
            'dx': dx, 'dy': dy, 'dz': dz,
            'step': step, 'time': time
        }
        
        cells = []
        for i in range(num_cells):
            # Read cell metadata
            cell_id = struct.unpack('i', f.read(4))[0]
            
            # BoundingBox3D: x0, y0, z0, x1, y1, z1 (physical cell bounds)
            bbox = struct.unpack('6i', f.read(24))
            x0, y0, z0, x1, y1, z1 = bbox
            
            # BoundingBox3D: bbox_with_halo (includes ghost cells, actual field dimensions)
            bbox_halo = struct.unpack('6i', f.read(24))
            hx0, hy0, hz0, hx1, hy1, hz1 = bbox_halo
            
            # Vec3 centroid
            centroid = struct.unpack('3f', f.read(12))
            
            # volume, theta, phi_pol
            volume = struct.unpack('f', f.read(4))[0]
            theta = struct.unpack('f', f.read(4))[0]
            phi_pol = struct.unpack('f', f.read(4))[0]
            
            # Vec3 polarization
            polarization = struct.unpack('3f', f.read(12))
            
            # Vec3 velocity
            velocity = struct.unpack('3f', f.read(12))
            
            # Read field data - use bbox_with_halo dimensions!
            width = hx1 - hx0
            height = hy1 - hy0
            depth = hz1 - hz0
            size = width * height * depth
            
            phi_data = np.frombuffer(f.read(size * 4), dtype=np.float32)
            phi = phi_data.reshape((depth, height, width))  # z, y, x order
            
            cells.append({
                'id': cell_id,
                'bbox': bbox,
                'bbox_halo': bbox_halo,
                'centroid': centroid,
                'volume': volume,
                'theta': theta,
                'phi_pol': phi_pol,
                'polarization': polarization,
                'velocity': velocity,
                'phi': phi,
                'width': width,
                'height': height,
                'depth': depth
            })
        
        return params, cells

def combine_cells_to_grid(params, cells):
    """Combine all cell phase fields into a single domain grid.
    
    Optimized version using numpy vectorized operations.
    """
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    
    # Create combined phi array
    phi_combined = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    cell_ids = np.full((Nz, Ny, Nx), -1, dtype=np.int32)
    
    for cell in cells:
        # Use bbox_halo for iteration since that's what the field dimensions match
        hx0, hy0, hz0, hx1, hy1, hz1 = cell['bbox_halo']
        phi = cell['phi']  # Shape: (depth, height, width) = (z, y, x)
        
        # Create coordinate arrays for the subdomain
        lx = np.arange(cell['width'])
        ly = np.arange(cell['height'])
        lz = np.arange(cell['depth'])
        
        # Compute global coordinates with periodic wrapping
        gx = (hx0 + lx) % Nx
        gy = (hy0 + ly) % Ny
        gz = (hz0 + lz) % Nz
        
        # Use meshgrid to create index arrays
        gx_3d, gy_3d, gz_3d = np.meshgrid(gx, gy, gz, indexing='ij')
        
        # Transpose phi to match (x, y, z) indexing for meshgrid
        phi_xyz = phi.transpose(2, 1, 0)  # (width, height, depth) = (x, y, z)
        
        # Create mask where this cell's phi is greater than current combined
        current_vals = phi_combined[gz_3d, gy_3d, gx_3d]
        update_mask = phi_xyz > current_vals
        
        # Update combined grid where this cell has higher values
        phi_combined[gz_3d[update_mask], gy_3d[update_mask], gx_3d[update_mask]] = phi_xyz[update_mask]
        
        # Update cell IDs where phi > 0.5
        id_mask = update_mask & (phi_xyz > 0.5)
        cell_ids[gz_3d[id_mask], gy_3d[id_mask], gx_3d[id_mask]] = cell['id']
    
    return phi_combined, cell_ids


def visualize_checkpoint(filename, show_grid=False, isosurface_value=0.5, save_screenshot=None, 
                         volume_mode=False, cmap='viridis', cell_ids_to_show=None, boundary_mode=True):
    """Visualize a single 3D checkpoint.
    
    Args:
        filename: Path to checkpoint file
        show_grid: Show grid lines
        isosurface_value: Value for isosurface extraction (default 0.5)
        save_screenshot: Path to save screenshot (None for interactive)
        volume_mode: Use volume rendering instead of isosurfaces (3D heatmap style)
        cmap: Colormap for volume rendering (default 'viridis')
        cell_ids_to_show: List of cell IDs to visualize (None = all cells)
        boundary_mode: If True, show only cell boundaries (transparent centers)
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista is required for 3D visualization")
        return
    
    print(f"Loading {filename}...")
    params, cells = read_checkpoint_3d(filename)
    
    print(f"  Step: {params['step']}, Time: {params['time']:.4f}")
    print(f"  Domain: {params['Nx']} x {params['Ny']} x {params['Nz']}")
    print(f"  Cells: {len(cells)}")
    
    # Filter cells if specific IDs requested
    if cell_ids_to_show is not None:
        cells = [c for c in cells if c['id'] in cell_ids_to_show]
        print(f"  Showing cells: {[c['id'] for c in cells]}")
    
    dx, dy, dz = params['dx'], params['dy'], params['dz']
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    
    # Create plotter
    off_screen = save_screenshot is not None
    pl = pv.Plotter(off_screen=off_screen)
    pl.set_background('white')
    
    if volume_mode:
        # Volume rendering mode
        phi_combined, _ = combine_cells_to_grid(params, cells)
        phi_xyz = phi_combined.transpose(2, 1, 0)
        
        grid = pv.ImageData(
            dimensions=(Nx + 1, Ny + 1, Nz + 1),
            spacing=(dx, dy, dz),
            origin=(0, 0, 0)
        )
        grid.cell_data['phi'] = phi_xyz.flatten(order='F')
        
        if boundary_mode:
            # Soap bubble effect: nearly invisible inside, visible only at thin interface
            # phi=0: outside (invisible), phi~0.4-0.6: interface (visible), phi=1: inside (invisible)
            # Sharp peak at interface, transparent everywhere else
            opacity = [0, 0, 0, 0.6, 0.8, 0.4, 0.02, 0]  # Sharp peak at interface only
            print(f"  Boundary mode: soap bubble effect (membranes only)")
        else:
            # Standard semi-transparent cells
            opacity = [0, 0, 0.05, 0.15, 0.3, 0.5]
        
        pl.add_volume(grid, scalars='phi', cmap=cmap, opacity=opacity,
                      shade=True, show_scalar_bar=True)
        print(f"  Volume rendering with cmap='{cmap}'")
    else:
        # Isosurface mode - extract surfaces for each cell
        pl.enable_anti_aliasing('ssaa')
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta',
                  'lime', 'pink', 'teal', 'brown', 'coral', 'gold', 'indigo', 'olive']
        
        for i, cell in enumerate(cells):
            hx0, hy0, hz0, hx1, hy1, hz1 = cell['bbox_halo']
            phi = cell['phi']
            phi_xyz = phi.transpose(2, 1, 0)
            
            print(f"  Cell {cell['id']}: subdomain {cell['width']}x{cell['height']}x{cell['depth']}, "
                  f"phi range [{phi.min():.3f}, {phi.max():.3f}]")
            
            cell_grid = pv.ImageData(
                dimensions=(cell['width'] + 1, cell['height'] + 1, cell['depth'] + 1),
                spacing=(dx, dy, dz),
                origin=(hx0 * dx, hy0 * dy, hz0 * dz)
            )
            cell_grid.cell_data['phi'] = phi_xyz.flatten(order='F')
            cell_grid = cell_grid.cell_data_to_point_data()
            
            try:
                contour = cell_grid.contour([isosurface_value], scalars='phi')
                if contour.n_points > 0:
                    contour = contour.compute_normals(auto_orient_normals=True)
                    print(f"    Contour: {contour.n_points} points")
                    
                    color = colors[i % len(colors)]
                    # Semi-transparent cells so we can see through
                    pl.add_mesh(contour, color=color, opacity=0.6,
                               smooth_shading=True, specular=0.5, specular_power=15,
                               diffuse=0.7, ambient=0.2)
            except Exception as e:
                print(f"  Warning: Could not extract isosurface for cell {cell['id']}: {e}")
    
    # Add domain bounding box - more opaque (90% = 0.9 opacity)
    Lx = params['Nx'] * dx
    Ly = params['Ny'] * dy
    Lz = params['Nz'] * dz
    bounds = pv.Box(bounds=(0, Lx, 0, Ly, 0, Lz))
    pl.add_mesh(bounds, style='wireframe', color='black', line_width=2, opacity=0.9)
    
    if show_grid:
        pl.show_grid()
    
    pl.add_title(f"Step {params['step']}, t = {params['time']:.2f}", font_size=12)
    
    # Set camera position and reset to fit all objects
    pl.camera_position = 'iso'
    pl.reset_camera()
    
    if save_screenshot:
        pl.screenshot(save_screenshot)
        print(f"  Saved screenshot to {save_screenshot}")
    
    return pl, params, cells


def create_movie_from_images(directory, fps=10):
    """Create a movie from PNG images in the images/ subdirectory."""
    img_dir = os.path.join(directory, 'images')
    movie_path = os.path.join(directory, 'simulation_3d.mp4')
    
    # Get sorted list of image files
    img_files = sorted(glob.glob(os.path.join(img_dir, 'checkpoint_3d_*.png')))
    
    if not img_files:
        print(f"No images found in {img_dir}")
        return
    
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
        with imageio.get_writer(movie_path, fps=fps, macro_block_size=1) as writer:
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
                '-framerate', str(fps),
                '-pattern_type', 'glob',
                '-i', os.path.join(img_dir, '*.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                movie_path
            ]
            subprocess.run(cmd, check=True)
            print(f"Saved movie: {movie_path}")
        except FileNotFoundError:
            print("Error: Neither imageio nor ffmpeg available for movie creation")
            print("Install imageio: pip install imageio imageio-ffmpeg")
        except subprocess.CalledProcessError as e:
            print(f"Error creating movie with ffmpeg: {e}")


def visualize_directory(directory, isosurface_value=0.5, movie=False, movie_only=False, fps=10,
                        volume_mode=False, cmap='viridis', cell_ids_to_show=None, 
                        boundary_mode=True, frame=None, frame_range=None):
    """Visualize all checkpoints in a directory.
    
    Args:
        directory: Directory containing checkpoint files
        isosurface_value: Value for isosurface extraction
        movie: Create movie after generating images
        movie_only: Create movie from existing images (skip rendering)
        fps: Frames per second for movie
        volume_mode: Use volume rendering
        cmap: Colormap for volume rendering
        cell_ids_to_show: List of cell IDs to visualize (None = all)
        boundary_mode: Show cell boundaries (transparent interiors)
        frame: Specific frame number to process
        frame_range: Tuple of (start, end) frame numbers
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista is required for 3D visualization")
        return
    
    # Find all 3D checkpoint files
    pattern = os.path.join(directory, "checkpoint_3d_*.bin")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No 3D checkpoint files found in {directory}")
        return
    
    print(f"Found {len(files)} checkpoint files")
    
    # Filter to specific frame if requested
    if frame is not None:
        frame_file = os.path.join(directory, f'checkpoint_3d_{frame:06d}.bin')
        if os.path.exists(frame_file):
            files = [frame_file]
            print(f"Processing single frame: {frame}")
        else:
            print(f"Frame {frame} not found: {frame_file}")
            return
    
    # Filter to frame range if requested
    if frame_range is not None:
        start_frame, end_frame = frame_range
        filtered = []
        for f in files:
            try:
                frame_num = int(os.path.basename(f).replace('checkpoint_3d_', '').replace('.bin', ''))
                if start_frame <= frame_num <= end_frame:
                    filtered.append(f)
            except ValueError:
                continue
        files = filtered
        print(f"Processing frames {start_frame} to {end_frame} ({len(files)} frames)")
    
    # Create images directory
    img_dir = os.path.join(directory, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    # Generate images (skip if movie_only)
    if not movie_only:
        for i, filename in enumerate(files):
            basename = os.path.basename(filename)
            frame_name = basename.replace('.bin', '.png')
            img_path = os.path.join(img_dir, frame_name)
            
            print(f"Rendering {i+1}/{len(files)}: {basename}...")
            
            try:
                visualize_checkpoint(filename, 
                                    isosurface_value=isosurface_value,
                                    save_screenshot=img_path,
                                    volume_mode=volume_mode,
                                    cmap=cmap,
                                    cell_ids_to_show=cell_ids_to_show,
                                    boundary_mode=boundary_mode)
            except Exception as e:
                print(f"  Error: {e}")
    
    # Create movie if requested
    if movie or movie_only:
        create_movie_from_images(directory, fps=fps)


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D cell simulation checkpoints')
    parser.add_argument('input', help='Checkpoint file or directory containing checkpoints')
    parser.add_argument('--iso', type=float, default=0.5, help='Isosurface value (default: 0.5)')
    parser.add_argument('--movie', action='store_true', help='Create movie after generating images')
    parser.add_argument('--movie-only', action='store_true', help='Create movie from existing images (skip rendering)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for movie')
    parser.add_argument('--grid', action='store_true', help='Show grid lines')
    parser.add_argument('--screenshot', type=str, help='Save screenshot to file (PNG)')
    parser.add_argument('--volume', action='store_true', help='Use volume rendering (3D heatmap style)')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap for volume rendering')
    parser.add_argument('--cells', type=str, help='Comma-separated list of cell IDs to show (e.g., "0,1,5")')
    parser.add_argument('--no-boundary', action='store_true', help='Disable boundary mode (show solid cells)')
    parser.add_argument('-f', '--frame', type=int, default=None, help='Process only a specific frame number')
    parser.add_argument('--start', type=int, default=None, help='Start frame number for range')
    parser.add_argument('--end', type=int, default=None, help='End frame number for range')
    
    args = parser.parse_args()
    
    if not PYVISTA_AVAILABLE:
        print("Error: PyVista is required. Install with: pip install pyvista")
        return
    
    # Parse cell IDs if provided
    cell_ids = None
    if args.cells:
        cell_ids = [int(x.strip()) for x in args.cells.split(',')]
        print(f"Showing only cells: {cell_ids}")
    
    boundary_mode = not args.no_boundary
    
    # Determine frame range
    frame_range = None
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else 999999
        frame_range = (start, end)
    
    if os.path.isdir(args.input):
        visualize_directory(args.input, 
                          isosurface_value=args.iso, 
                          movie=args.movie,
                          movie_only=args.movie_only,
                          fps=args.fps,
                          volume_mode=args.volume, 
                          cmap=args.cmap,
                          cell_ids_to_show=cell_ids,
                          boundary_mode=boundary_mode,
                          frame=args.frame,
                          frame_range=frame_range)
    elif os.path.isfile(args.input):
        pl, params, cells = visualize_checkpoint(args.input, 
                                                  show_grid=args.grid,
                                                  isosurface_value=args.iso,
                                                  save_screenshot=args.screenshot,
                                                  volume_mode=args.volume,
                                                  cmap=args.cmap,
                                                  cell_ids_to_show=cell_ids,
                                                  boundary_mode=boundary_mode)
        if args.screenshot is None:
            pl.show()
    else:
        print(f"Error: {args.input} not found")


if __name__ == '__main__':
    main()
