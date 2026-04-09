"""Debug vertical artifact at row 252, col 145."""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_vtk_ascii(path):
    with open(path, 'r') as f: lines = f.readlines()
    nx=ny=0; ds=0
    for i,line in enumerate(lines):
        if line.startswith('DIMENSIONS'): p=line.split(); nx,ny=int(p[1]),int(p[2])
        if line.startswith('LOOKUP_TABLE'): ds=i+1; break
    vals=[float(lines[i].strip()) for i in range(ds,len(lines)) if lines[i].strip()]
    return np.array(vals).reshape(ny,nx),nx,ny

BASE=os.path.join(os.environ['TEMP'],'tc2_Jt0750_8tau','fields')
FIGDIR=r'c:\Users\stevensilber\source\repos\data_processing\cpp\simulation\study\adhesion\figures'

phi0,nx,ny = load_vtk_ascii(os.path.join(BASE,'frame_8000001_cell_000.vtk'))
phi1,_,_ = load_vtk_ascii(os.path.join(BASE,'frame_8000001_cell_001.vtk'))
S = phi0 + phi1

# Print raw values
print('=== phi0 around (row=252, col=145) ===')
for r in range(249, 256):
    line = '  row %d:' % r
    for c in range(141, 150):
        line += ' %10.6f' % phi0[r, c]
    print(line)

print('\n=== phi1 around (row=252, col=145) ===')
for r in range(249, 256):
    line = '  row %d:' % r
    for c in range(141, 150):
        line += ' %10.6f' % phi1[r, c]
    print(line)

print('\n=== S = phi0+phi1 around (row=252, col=145) ===')
for r in range(249, 256):
    line = '  row %d:' % r
    for c in range(141, 150):
        line += ' %10.6f' % S[r, c]
    print(line)

# Bounding boxes
nz0r = np.where(phi0.max(axis=1) > 0.001)[0]
nz0c = np.where(phi0.max(axis=0) > 0.001)[0]
nz1r = np.where(phi1.max(axis=1) > 0.001)[0]
nz1c = np.where(phi1.max(axis=0) > 0.001)[0]
print('\nphi0 nonzero: rows [%d, %d], cols [%d, %d]' % (nz0r[0], nz0r[-1], nz0c[0], nz0c[-1]))
print('phi1 nonzero: rows [%d, %d], cols [%d, %d]' % (nz1r[0], nz1r[-1], nz1c[0], nz1c[-1]))

# Transition in phi1 along row 252 near col 145
print('\nphi1 row 252, transition region:')
for c in range(138, 158):
    v = phi1[252, c]
    if v > 1e-12 or (c > 138 and phi1[252, c-1] > 1e-12):
        print('  col %d: phi1=%.10e  phi0=%.10e  S=%.10e' % (c, phi1[252,c], phi0[252,c], S[252,c]))

# Generate zoomed figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: S zoomed into (252, 145) area
ax = axes[0]
r0, r1 = 246, 258
c0c, c1c = 138, 158
ext = [c0c-0.5, c1c-0.5, r0-0.5, r1-0.5]
ax.imshow(S[r0:r1, c0c:c1c], origin='lower', cmap='inferno', vmin=0, vmax=0.1,
          extent=ext, interpolation='nearest')
ax.set_title('S (vmax=0.1)', fontsize=10)
# Mark the point
ax.plot(145, 252, 'rx', ms=10, mew=2)
# Draw bbox boundaries
ax.axhline(nz0r[-1]+0.5, color='cyan', ls='--', lw=0.8, label='phi0 bbox top')
ax.axvline(nz1c[0]-0.5, color='lime', ls='--', lw=0.8, label='phi1 bbox left')
ax.legend(fontsize=7)

# Panel 2: phi0 same area
ax = axes[1]
ax.imshow(phi0[r0:r1, c0c:c1c], origin='lower', cmap='inferno', vmin=0, vmax=0.1,
          extent=ext, interpolation='nearest')
ax.set_title('phi0 only (vmax=0.1)', fontsize=10)
ax.axhline(nz0r[-1]+0.5, color='cyan', ls='--', lw=0.8)

# Panel 3: phi1 same area
ax = axes[2]
ax.imshow(phi1[r0:r1, c0c:c1c], origin='lower', cmap='inferno', vmin=0, vmax=0.01,
          extent=ext, interpolation='nearest')
ax.set_title('phi1 only (vmax=0.01)', fontsize=10)
ax.axvline(nz1c[0]-0.5, color='lime', ls='--', lw=0.8)

fig.tight_layout()
fname = os.path.join(FIGDIR, 'debug_artifact_col145.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved %s' % fname)
