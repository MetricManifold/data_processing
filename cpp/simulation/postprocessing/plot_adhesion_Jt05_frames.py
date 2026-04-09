"""Plot VTK frames from adhesion instability test J/kappa=0.5, dt=0.01, 72 cells."""
import sys
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(r"c:\Users\stevensilber\source\repos\data_processing\cpp\simulation\agent_test_runs\adhesion_Jt0.5_dt0.01_72c")
OUT_PATH = Path(r"c:\Users\stevensilber\source\repos\data_processing\cpp\simulation\postprocessing\output\adhesion_Jt0.5_dt001_72c_frames_20260224.png")
LOG_PATH = OUT_PATH.parent / "plot_adhesion_log.txt"

# Redirect all output to a log file to avoid terminal noise
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
log_file = open(LOG_PATH, 'w')
sys.stdout = log_file
sys.stderr = log_file

# ── 1. Inspect trajectory ──────────────────────────────────────────────────
traj_path = DATA_DIR / "trajectory.txt"
print("=== Trajectory inspection ===")
with open(traj_path) as f:
    lines = f.readlines()

for h in lines[:3]:
    print(h.strip())

data_lines = lines[3:]
times = []
for l in data_lines:
    t = float(l.split()[0])
    if not times or t != times[-1]:
        times.append(t)

print(f"First time: {times[0]}")
print(f"Last time:  {times[-1]}")
print(f"Unique timestamps: {len(times)}")
n_cells = len(data_lines) // len(times)
print(f"N cells: {n_cells}")
print()

# ── 2. Parse binary VTK ────────────────────────────────────────────────────
def read_vtk_binary(path):
    """Read legacy binary VTK structured points: ASCII header then big-endian float32."""
    with open(path, 'rb') as f:
        raw = f.read()

    # Find header lines (ASCII portion)
    nx, ny, nz = None, None, None
    pos = 0
    while True:
        end = raw.index(b'\n', pos)
        line = raw[pos:end].decode('ascii', errors='replace')
        if line.startswith('DIMENSIONS'):
            parts = line.split()
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
        if line.startswith('LOOKUP_TABLE'):
            pos = end + 1  # data starts right after this newline
            break
        pos = end + 1

    n_points = nx * ny * nz
    data = np.frombuffer(raw[pos:pos + n_points * 4], dtype='>f4')  # big-endian float32
    field = data.reshape((ny, nx)) if nz == 1 else data.reshape((nz, ny, nx))
    return field, nx, ny

# ── 3. Select frames: first, two middle, last ──────────────────────────────
vtk_files = sorted(DATA_DIR.glob("frame_*.vtk"))
print(f"=== VTK frames ({len(vtk_files)} total) ===")
for v in vtk_files:
    print(f"  {v.name}")

# Pick 4 frames: first, 1/3, 2/3, last
indices = [0, len(vtk_files)//3, 2*len(vtk_files)//3, len(vtk_files)-1]
# Deduplicate while preserving order
seen = set()
selected = []
for i in indices:
    if i not in seen:
        seen.add(i)
        selected.append(i)

print(f"\nSelected frame indices: {selected}")
frames = []
for i in selected:
    vf = vtk_files[i]
    field, nx, ny = read_vtk_binary(vf)
    frames.append((vf.name, field, nx, ny))
    print(f"  {vf.name}: {nx}x{ny}, range [{field.min():.3f}, {field.max():.3f}]")

# ── 4. Plot ────────────────────────────────────────────────────────────────
n = len(frames)
fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
if n == 1:
    axes = [axes]

for ax, (name, field, nx, ny) in zip(axes, frames):
    im = ax.imshow(field, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1,
                   extent=[0, nx, 0, ny], aspect='equal')
    # Extract frame number for approximate time
    frame_num = int(name.replace('frame_', '').replace('.vtk', ''))
    ax.set_title(f"frame {frame_num:,}", fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

fig.suptitle(r'Adhesion instability: $J/\kappa = 0.5$, dt=0.01, 72 cells', fontsize=13, y=1.02)
fig.colorbar(im, ax=axes, shrink=0.8, label=r'$\phi$')
plt.tight_layout()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_PATH}")
log_file.close()
