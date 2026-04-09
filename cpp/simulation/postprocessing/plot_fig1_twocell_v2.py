"""Fig 1: Two-cell adhesion validation.

Generates:
  figures/fig1_snap_Jt{X.XX}.pdf    — 4 snapshot panels (J̃ = 0, 0.25, 0.50, 0.75)
  figures/fig1_contact_length.pgf   — ℓ_c/R vs J̃ (native LaTeX via PGF)
  figures/fig1_contact_length.pdf   — same plot as standalone PDF

Run from: cpp/simulation/
Requires: tc2_palmieri/ sweep data in %TEMP%
"""
import os, sys, struct, math, numpy as np

# --- PGF backend for native LaTeX rendering ---
import matplotlib
matplotlib.use('pgf')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 9,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'pdflatex',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
})
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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


BASE = os.path.join(os.environ['TEMP'], 'tc2_palmieri')
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', 'study', 'adhesion', 'figures')
FIGDIR = os.path.normpath(FIGDIR)
os.makedirs(FIGDIR, exist_ok=True)
R = 49.0

# ── Snapshot panels ──────────────────────────────────────────────────────────

snap_cases = [
    ('J0',      0.000),
    ('Jt0.250', 0.250),
    ('Jt0.500', 0.500),
    ('Jt0.750', 0.750),
]

# Crop region: focus on the two cells (skip the empty surrounding box)
# Cells are near center of 400×400 box
CROP = 80  # pixels of padding around cell pair

for label, Jt in snap_cases:
    d = os.path.join(BASE, label)
    vtks = sorted([f for f in os.listdir(d) if f.endswith('.vtk')])
    S, nx, ny = load_vtk(os.path.join(d, vtks[-1]))

    # Get centroid positions for cropping
    traj = np.loadtxt(os.path.join(d, 'trajectory.txt'), comments='#')
    fl = traj[-2:]
    cx = (fl[0, 2] + fl[1, 2]) / 2
    cy = (fl[0, 3] + fl[1, 3]) / 2

    # Crop around the pair
    x0 = max(0, int(cx) - CROP - 49)
    x1 = min(nx, int(cx) + CROP + 49)
    y0 = max(0, int(cy) - CROP - 49)
    y1 = min(ny, int(cy) + CROP + 49)
    S_crop = S[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    ax.imshow(S_crop, origin='lower', cmap='inferno', vmin=0, vmax=1.0,
              aspect='equal', interpolation='bilinear')

    # φ=0.5 contour in white
    contours = find_contours(S_crop, 0.5)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], 'w-', lw=0.5)

    Jt_str = f'{Jt:.2f}' if Jt > 0 else '0'
    ax.set_title(r'$\tilde{{J}} = {}$'.format(Jt_str), fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    fname = f'fig1_snap_Jt{Jt:.2f}'
    fig.savefig(os.path.join(FIGDIR, fname + '.pdf'),
                bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f'  {fname}.pdf')

# ── Contact length plot ──────────────────────────────────────────────────────

# Load metrics
metrics = np.loadtxt(os.path.join(BASE, 'metrics.txt'), comments='#')
Jt_arr = metrics[:, 0]
lc_arr = metrics[:, 1]

fig, ax = plt.subplots(figsize=(3.375, 2.2))
ax.plot(Jt_arr, lc_arr, 's-', color='#2166ac', ms=4, lw=1.0,
        mfc='#2166ac', mec='black', mew=0.3, zorder=3)
ax.set_xlabel(r'$\tilde{J} = J\,/\,2\gamma$')
ax.set_ylabel(r'$\ell_c\,/\,R$')
ax.set_xlim(-0.03, 0.80)
ax.set_ylim(1.2, 1.8)
ax.tick_params(labelsize=7)

# Save as PGF (native LaTeX) and PDF
fig.savefig(os.path.join(FIGDIR, 'fig1_contact_length.pgf'),
            bbox_inches='tight')
fig.savefig(os.path.join(FIGDIR, 'fig1_contact_length.pdf'),
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  fig1_contact_length.pgf')
print('  fig1_contact_length.pdf')

print(f'\nAll files in {FIGDIR}')
