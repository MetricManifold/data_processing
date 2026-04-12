"""Analyze two-cell adhesion sweep: contact angle measurement.

For two cells in contact, the contact angle θ is the angle each cell's
free surface makes with the flat shared interface at the triple point.

Force balance (Neumann condition) predicts:
    cos(θ) = 1 - J̃

where J̃ = J/(2γ) is the dimensionless adhesion strength.

At J̃ = 0: θ = 0° (no contact, tangent touching)
At J̃ = 0.5: θ = 60°
At J̃ = 1: θ = 90° (stability limit)
"""
import os, struct, math, numpy as np
from skimage.measure import find_contours

def load_vtk(path):
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


def measure_contact_angle(S, cx0, cy0, cx1, cy1):
    """Measure the contact angle θ from the φ=0.5 contour geometry.
    
    Strategy: The cells are aligned along the x-axis (by construction).
    At the triple point (top and bottom of the contact zone), the contour
    bends from the free-surface arc into the shared-interface region.
    
    We measure θ by fitting the contour tangent direction at the triple
    points. The contact angle is between the free surface tangent and
    the perpendicular to the cell-cell axis (i.e., the contact plane).
    
    Returns: (theta_degrees, contact_length) or (None, 0) if no contact.
    """
    contours = find_contours(S, 0.5)
    contours.sort(key=lambda c: len(c), reverse=True)
    
    # Cell-cell axis
    axis = np.array([cx1 - cx0, cy1 - cy0])
    axis_len = np.linalg.norm(axis)
    if axis_len < 1:
        return None, 0.0
    axis_hat = axis / axis_len
    perp_hat = np.array([-axis_hat[1], axis_hat[0]])
    mid = np.array([(cx0 + cx1) / 2, (cy0 + cy1) / 2])
    
    merged = len(contours) == 1 and len(contours[0]) > 200
    
    if merged:
        # Single peanut contour — find the two waist points (triple points)
        c = contours[0]
        pts = c[:, ::-1]  # (x, y)
        rel = pts - mid
        proj_ax = rel @ axis_hat
        proj_pp = rel @ perp_hat
        
        # Triple points are near the midplane (proj_ax ≈ 0) at max/min perp
        near_mid = np.abs(proj_ax) < 8
        if near_mid.sum() < 4:
            return None, 0.0
        
        # Contact length = perpendicular extent at the waist
        perp_near = proj_pp[near_mid]
        contact_len = perp_near.max() - perp_near.min()
        
        # Find the two triple points (top and bottom of waist)
        idx_top = np.where(near_mid)[0][np.argmax(proj_pp[near_mid])]
        idx_bot = np.where(near_mid)[0][np.argmin(proj_pp[near_mid])]
        
        angles = []
        for idx in [idx_top, idx_bot]:
            win = 15
            n = len(c)
            idx_fwd = (idx + win) % n
            idx_bwd = (idx - win) % n
            
            t_fwd = c[idx_fwd, ::-1] - c[idx, ::-1]
            t_bwd = c[idx_bwd, ::-1] - c[idx, ::-1]
            t_fwd = t_fwd / (np.linalg.norm(t_fwd) + 1e-10)
            t_bwd = t_bwd / (np.linalg.norm(t_bwd) + 1e-10)
            
            for t in [t_fwd, t_bwd]:
                cos_angle = abs(np.dot(t, axis_hat))
                theta = math.asin(min(1.0, cos_angle))
                angles.append(theta)
        
        if angles:
            return math.degrees(np.mean(angles)), contact_len
        return None, contact_len
    
    elif len(contours) >= 2:
        c1, c2 = contours[0], contours[1]
        
        from scipy.spatial import cKDTree
        tree2 = cKDTree(c2)
        dists, _ = tree2.query(c1)
        
        contact_mask = dists < 14
        if contact_mask.sum() < 2:
            return 0.0, 0.0  # No contact → θ=0
        
        contact_pts = c1[contact_mask]
        d = np.diff(contact_pts, axis=0)
        contact_len = np.sum(np.sqrt(d[:, 0]**2 + d[:, 1]**2))
        
        # Find transition points where contact starts/ends
        transitions = np.diff(contact_mask.astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        
        if len(starts) == 0 and len(ends) == 0:
            return None, contact_len
        
        angles = []
        for idx_list, direction in [(starts, -1), (ends, 1)]:
            for idx in idx_list:
                if idx < 15 or idx >= len(c1) - 15:
                    continue
                win = 15
                t_free = c1[idx + direction * win, ::-1] - c1[idx, ::-1]
                t_free = t_free / (np.linalg.norm(t_free) + 1e-10)
                cos_angle = abs(np.dot(t_free, axis_hat))
                theta = math.asin(min(1.0, cos_angle))
                angles.append(theta)
        
        if angles:
            return math.degrees(np.mean(angles)), contact_len
        return None, contact_len
    
    return None, 0.0


BASE = os.path.join(os.environ['TEMP'], 'tc2_palmieri')
R = 49.0
L = 400

cases = [
    ('J0',      0.0,   0.000),
    ('Jt0.125', 0.25,  0.125),
    ('Jt0.250', 0.50,  0.250),
    ('Jt0.375', 0.75,  0.375),
    ('Jt0.500', 1.00,  0.500),
    ('Jt0.625', 1.25,  0.625),
    ('Jt0.750', 1.50,  0.750),
]

print(f"{'Jt':>6s} {'theta_meas':>11s} {'theta_pred':>11s} {'l_c/R':>7s} {'d/R':>7s} {'contours':>9s}")
print('-' * 60)

results = []
for label, J, Jt in cases:
    d = os.path.join(BASE, label)
    vtks = sorted([f for f in os.listdir(d) if f.endswith('.vtk')])
    S, nx, ny = load_vtk(os.path.join(d, vtks[-1]))

    traj = np.loadtxt(os.path.join(d, 'trajectory.txt'), comments='#')
    fl = traj[-2:]
    cx0, cy0 = fl[0, 2], fl[0, 3]
    cx1, cy1 = fl[1, 2], fl[1, 3]
    dx = abs(cx1 - cx0); dx = min(dx, L - dx)
    dy = abs(cy1 - cy0); dy = min(dy, L - dy)
    d_eq = math.sqrt(dx**2 + dy**2)

    contours = find_contours(S, 0.5)
    n_contours = len([c for c in contours if len(c) > 50])
    
    theta_meas, l_c = measure_contact_angle(S, cx0, cy0, cx1, cy1)
    theta_pred = math.degrees(math.acos(1 - Jt)) if Jt < 1 else 90.0

    theta_str = f"{theta_meas:8.1f}" if theta_meas is not None else "     N/A"
    pred_str = f"{theta_pred:8.1f}"
    
    print(f'{Jt:6.3f} {theta_str:>11s} {pred_str:>11s} {l_c/R:7.3f} {d_eq/R:7.3f} {n_contours:>9d}')
    results.append((Jt, theta_meas, theta_pred, l_c / R, d_eq / R))

# Save for plotting
out_file = os.path.join(BASE, 'contact_angle.txt')
with open(out_file, 'w') as f:
    f.write('# Jt  theta_measured  theta_predicted  l_c_over_R  d_over_R\n')
    for Jt, tm, tp, lc, dR in results:
        tm_val = tm if tm is not None else float('nan')
        f.write(f'{Jt:.3f} {tm_val:.4f} {tp:.4f} {lc:.6f} {dR:.6f}\n')
print(f'\nSaved to {out_file}')
