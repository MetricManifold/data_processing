"""Fig 1: Two-cell adhesion validation — individual files.

Generates:
  - fig1_twocell_Jt{X.XXX}.pdf   (one per J_tilde value, snapshot only)
  - fig1_twocell_deq.pdf          (d_eq/R vs J_tilde chart)
"""
import os, struct, numpy as np, matplotlib
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
    'lines.linewidth': 1.2,
})
import matplotlib.pyplot as plt

def load_vtk(path):
    with open(path, 'rb') as f:
        content = f.read()
    he = content.find(b'LOOKUP_TABLE default\n')
    header = content[:he].decode('ascii', errors='replace')
    for line in header.split('\n'):
        if 'DIMENSIONS' in line:
            nx, ny = int(line.split()[1]), int(line.split()[2])
    ds = he + len(b'LOOKUP_TABLE default\n')
    return np.array(struct.unpack('>' + 'f' * (nx*ny),
                    content[ds:ds+4*nx*ny])).reshape(ny, nx), nx, ny

base = os.environ['TEMP']
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(outdir, exist_ok=True)
R = 49.0

all_cases = [
    ('J0',      0,      0),
    ('Jt0.125', 0.9375, 0.125),
    ('Jt0.250', 1.875,  0.250),
    ('Jt0.375', 2.8125, 0.375),
    ('Jt0.500', 3.75,   0.500),
    ('Jt0.625', 4.6875, 0.625),
    ('Jt0.750', 5.625,  0.750),
]

# --- Individual snapshot frames ---
for label, J, Jt in all_cases:
    d = os.path.join(base, 'tc2_' + label)
    vtks = sorted([f for f in os.listdir(d) if f.endswith('.vtk')])
    phi, nx, ny = load_vtk(os.path.join(d, vtks[-1]))

    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    ax.imshow(phi, origin='lower', cmap='inferno', vmin=0, vmax=1.0,
              extent=[0, nx, 0, ny], aspect='equal', interpolation='bilinear')
    ax.contour(phi, levels=[0.5], colors='white', linewidths=0.4,
               extent=[0, nx, 0, ny])
    Jt_str = str(Jt) if Jt > 0 else '0'
    ax.set_title(r'$\tilde{J}=' + Jt_str + '$', fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    fname = 'fig1_twocell_Jt' + '{:.3f}'.format(Jt)
    fig.savefig(os.path.join(outdir, fname + '.pdf'),
                bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(outdir, fname + '.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ' + fname)

# --- d_eq chart ---
Jt_arr = []
deq_arr = []
for label, J, Jt in all_cases:
    traj = os.path.join(base, 'tc2_' + label, 'trajectory.txt')
    data = np.loadtxt(traj, comments='#')
    fl = data[-2:]
    d_eq = np.sqrt((fl[0, 2] - fl[1, 2])**2 + (fl[0, 3] - fl[1, 3])**2)
    Jt_arr.append(Jt)
    deq_arr.append(d_eq)
    print(f'  Jt={Jt:.3f}: d_eq/R = {d_eq/R:.3f}')

Jt_arr = np.array(Jt_arr)
deq_arr = np.array(deq_arr)

fig, ax = plt.subplots(figsize=(3.375, 2.2))
ax.plot(Jt_arr, deq_arr / R, 's-', color='#2166ac', ms=4, lw=1.0,
        mfc='#2166ac', mec='black', mew=0.3, zorder=3)
ax.axhline(2.0, color='#999999', ls='--', lw=0.5, zorder=1)
ax.text(0.72, 2.005, r'$2R$', fontsize=7, color='#666666', va='bottom')
ax.set_xlabel(r'$\tilde{J} = J\,/\,2\gamma$', fontsize=9)
ax.set_ylabel(r'$d_{\mathrm{eq}}\,/\,R$', fontsize=9)
ax.set_xlim(-0.03, 0.80)
ax.set_ylim(1.78, 2.08)
ax.tick_params(labelsize=7)

fig.savefig(os.path.join(outdir, 'fig1_twocell_deq.pdf'),
            bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(outdir, 'fig1_twocell_deq.png'),
            dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  fig1_twocell_deq')
print('Done.')
