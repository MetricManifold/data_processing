"""Fig 1: Two-cell adhesion validation — Palmieri parameters (gamma=1).

Generates individual panel PDFs for LaTeX \\includegraphics assembly:
  figures/fig1_snap_Jt{X.XXX}.pdf     (4 snapshot panels for 2x2 grid)
  figures/fig1_contact.pdf             (contact angle + contact length vs Jt)

Contact angle alpha is measured from the phi=0.5 contour geometry at the
cell-cell junction. The Young-Dupre prediction cos(alpha) = 1 - Jt is overlaid.

Contact length ell is the vertical extent of the shared interface region.
"""
import os, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.0,
})
import matplotlib.pyplot as plt
from skimage import measure

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.environ['USERPROFILE'], 'AppData', 'Local', 'Temp',
                    'tc2_palmieri')
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', 'study', 'adhesion', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

R = 49.0
GAMMA = 1.0

ALL_CASES = [
    ('J0',      0.0,   0.000),
    ('Jt0.125', 0.25,  0.125),
    ('Jt0.250', 0.50,  0.250),
    ('Jt0.375', 0.75,  0.375),
    ('Jt0.500', 1.00,  0.500),
    ('Jt0.625', 1.25,  0.625),
    ('Jt0.750', 1.50,  0.750),
]

SNAP_INDICES = [0, 2, 4, 6]  # Jt = 0, 0.25, 0.5, 0.75 for 2x2 snapshots


# ─── VTK loader ──────────────────────────────────────────────────────────────
def load_vtk(path):
    with open(path, 'rb') as f:
        content = f.read()
    he = content.find(b'LOOKUP_TABLE default\n')
    header = content[:he].decode('ascii', errors='replace')
    for line in header.split('\n'):
        if 'DIMENSIONS' in line:
            parts = line.split()
            nx, ny = int(parts[1]), int(parts[2])
    ds = he + len(b'LOOKUP_TABLE default\n')
    data = np.array(struct.unpack('>' + 'f' * (nx * ny),
                                  content[ds:ds + 4 * nx * ny]))
    return data.reshape(ny, nx), nx, ny


def get_final_vtk(case_dir):
    vtks = sorted([f for f in os.listdir(case_dir) if f.endswith('.vtk')])
    return os.path.join(case_dir, vtks[-1]) if vtks else None


# ─── Contact geometry ─────────────────────────────────────────────────────────
def measure_contact_geometry(phi, nx, ny, cx0=150, cx1=250):
    """Measure contact length and contact angle from the phi field.
    
    Returns (contact_length, contact_angle_deg).
    """
    midx = (cx0 + cx1) / 2.0
    midy = ny / 2.0
    
    contours = measure.find_contours(phi, 0.5)
    if not contours:
        return 0.0, 180.0
    
    # Two separate contours = cells not in contact
    if len(contours) >= 2:
        # Check if the two largest contours are separate blobs (not artifacts)
        sorted_c = sorted(contours, key=len, reverse=True)
        c1_x = sorted_c[0][:, 1].mean()
        c2_x = sorted_c[1][:, 1].mean()
        if abs(c1_x - c2_x) > R:  # clearly separate cells
            return 0.0, 180.0
    
    # Single contour (peanut shape) = cells in contact
    c = max(contours, key=len)
    cx_arr = c[:, 1]
    cy_arr = c[:, 0]
    
    # Find junction points: where contour crosses x = midx
    near_mid = np.abs(cx_arr - midx) < 2.0
    if np.sum(near_mid) < 2:
        return 0.0, 180.0
    
    y_at_mid = cy_arr[near_mid]
    y_min, y_max = y_at_mid.min(), y_at_mid.max()
    contact_length = y_max - y_min
    
    # Contact angle measurement:
    # At each junction point, measure the angle of the phi=0.5 contour.
    # The contact angle alpha is the angle between the two cell surfaces
    # at the triple point.
    angles = []
    for y_junc in [y_min, y_max]:
        # Find contour index closest to junction
        dists = np.sqrt((cx_arr - midx)**2 + (cy_arr - y_junc)**2)
        idx = np.argmin(dists)
        n = len(cx_arr)
        
        # Get segment around junction
        window = 25
        indices = np.arange(idx - window, idx + window + 1) % n
        seg_x = cx_arr[indices]
        seg_y = cy_arr[indices]
        
        # Split into left branch (cell 1 side) and right branch (cell 2 side)
        left_mask = seg_x < midx - 3
        right_mask = seg_x > midx + 3
        
        if np.sum(left_mask) < 3 or np.sum(right_mask) < 3:
            continue
        
        # Fit tangent on each side (use points closest to junction)
        def fit_tangent(sx, sy, midx_ref, n_pts=6):
            dist = np.abs(sx - midx_ref)
            order = np.argsort(dist)[:n_pts]
            if len(order) < 3:
                return None
            p = np.polyfit(sx[order], sy[order], 1)
            return np.arctan(p[0])
        
        theta_l = fit_tangent(seg_x[left_mask], seg_y[left_mask], midx)
        theta_r = fit_tangent(seg_x[right_mask], seg_y[right_mask], midx)
        
        if theta_l is not None and theta_r is not None:
            # Contact angle at this junction
            opening = abs(theta_l - theta_r)
            alpha = np.pi - opening
            angles.append(alpha)
    
    alpha_deg = np.degrees(np.mean(angles)) if angles else 180.0
    return contact_length, alpha_deg


# ─── d_eq from trajectory ────────────────────────────────────────────────────
def get_deq(case_dir):
    traj = os.path.join(case_dir, 'trajectory.txt')
    data = np.loadtxt(traj, comments='#')
    times = np.unique(data[:, 0])
    final = data[data[:, 0] == times[-1]]
    return np.sqrt((final[0, 2] - final[1, 2])**2 +
                   (final[0, 3] - final[1, 3])**2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=== Two-cell adhesion validation (Palmieri gamma=1) ===\n')
    
    # ─── Collect measurements ─────────────────────────────────────────────
    results = []
    for label, J, Jt in ALL_CASES:
        case_dir = os.path.join(BASE, label)
        d_eq = get_deq(case_dir)
        
        vtk_path = get_final_vtk(case_dir)
        if vtk_path:
            phi, nx, ny = load_vtk(vtk_path)
            cl, alpha = measure_contact_geometry(phi, nx, ny)
        else:
            cl, alpha = 0.0, 180.0
        
        cos_a = np.cos(np.radians(alpha))
        results.append({
            'label': label, 'J': J, 'Jt': Jt,
            'd_eq_R': d_eq / R,
            'cl_R': cl / R,
            'alpha': alpha,
            'cos_alpha': cos_a,
        })
        
        theory = 1 - Jt
        print(f'  Jt={Jt:.3f}: d/R={d_eq/R:.4f}  '
              f'l/R={cl/R:.3f}  alpha={alpha:.1f} deg  '
              f'cos(alpha)={cos_a:.3f}  (theory: {theory:.3f})')
    
    # ─── Snapshot panels (1.6" square each) ───────────────────────────────
    print('\nSnapshot panels:')
    for idx in SNAP_INDICES:
        label, J, Jt = ALL_CASES[idx]
        case_dir = os.path.join(BASE, label)
        vtk_path = get_final_vtk(case_dir)
        if not vtk_path:
            print(f'  {label}: NO VTK')
            continue
        
        phi, nx, ny = load_vtk(vtk_path)
        
        # Crop to ROI around the cell pair
        margin = 25
        x0 = int(150 - R - margin)
        x1 = int(250 + R + margin)
        y0 = int(200 - R - margin)
        y1 = int(200 + R + margin)
        crop = phi[y0:y1, x0:x1]
        
        fig, ax = plt.subplots(figsize=(1.6, 1.2))
        ax.imshow(crop, origin='lower', cmap='inferno', vmin=0, vmax=1.0,
                  extent=[x0, x1, y0, y1], aspect='equal',
                  interpolation='bilinear')
        
        # phi=0.5 contour
        contours = measure.find_contours(phi, 0.5)
        for cont in contours:
            mask = ((cont[:, 1] >= x0) & (cont[:, 1] <= x1) &
                    (cont[:, 0] >= y0) & (cont[:, 0] <= y1))
            if np.any(mask):
                ax.plot(cont[mask, 1], cont[mask, 0], 'w-', lw=0.5)
        
        Jt_str = str(Jt) if Jt > 0 else '0'
        ax.set_title(r'$\tilde{J}=' + Jt_str + '$', fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
        
        fname = f'fig1_snap_Jt{Jt:.3f}'
        fig.savefig(os.path.join(OUTDIR, fname + '.pdf'),
                    bbox_inches='tight', facecolor='white', pad_inches=0.02)
        fig.savefig(os.path.join(OUTDIR, fname + '.png'),
                    dpi=600, bbox_inches='tight', facecolor='white',
                    pad_inches=0.02)
        plt.close(fig)
        print(f'  {fname}')
    
    # ─── Contact geometry plot (single-column width) ──────────────────────
    print('\nContact plots:')
    Jt_arr = np.array([r['Jt'] for r in results])
    cos_arr = np.array([r['cos_alpha'] for r in results])
    cl_arr = np.array([r['cl_R'] for r in results])
    deq_arr = np.array([r['d_eq_R'] for r in results])
    
    fig, axes = plt.subplots(1, 2, figsize=(3.375, 1.8))
    
    # Left: cos(alpha) vs Jt
    ax = axes[0]
    ax.plot(Jt_arr, cos_arr, 's-', color='#2166ac', ms=3.5, lw=0.8,
            mfc='#2166ac', mec='black', mew=0.3, zorder=3, label='Simulation')
    Jt_th = np.linspace(0, 0.8, 50)
    ax.plot(Jt_th, 1 - Jt_th, '--', color='#b2182b', lw=0.8,
            zorder=2, label=r'$1 - \tilde{J}$')
    ax.set_xlabel(r'$\tilde{J}$', fontsize=9)
    ax.set_ylabel(r'$\cos\alpha$', fontsize=9)
    ax.set_xlim(-0.03, 0.80)
    ax.set_ylim(0.0, 1.15)
    ax.legend(fontsize=6, frameon=False, loc='upper right')
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.95, '(e)', transform=ax.transAxes, fontsize=8,
            va='top', fontweight='bold')
    
    # Right: contact length vs Jt
    ax = axes[1]
    ax.plot(Jt_arr, cl_arr, 's-', color='#2166ac', ms=3.5, lw=0.8,
            mfc='#2166ac', mec='black', mew=0.3, zorder=3)
    ax.set_xlabel(r'$\tilde{J}$', fontsize=9)
    ax.set_ylabel(r'$\ell\,/\,R$', fontsize=9)
    ax.set_xlim(-0.03, 0.80)
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.95, '(f)', transform=ax.transAxes, fontsize=8,
            va='top', fontweight='bold')
    
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(OUTDIR, 'fig1_contact.pdf'),
                bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUTDIR, 'fig1_contact.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  fig1_contact')
    
    # ─── Data file for PGFplots ───────────────────────────────────────────
    dat = os.path.join(OUTDIR, 'fig1_twocell_data.dat')
    with open(dat, 'w') as f:
        f.write('# Two-cell validation: Palmieri gamma=1, R=49, dt=0.01, t=1000\n')
        f.write('# Jt  J  d_eq_R  contact_length_R  alpha_deg  cos_alpha\n')
        for r in results:
            f.write(f'{r["Jt"]:.3f}  {r["J"]:.2f}  {r["d_eq_R"]:.4f}  '
                    f'{r["cl_R"]:.4f}  {r["alpha"]:.2f}  '
                    f'{r["cos_alpha"]:.4f}\n')
    print(f'\nData: {dat}')
    print('Done.')
