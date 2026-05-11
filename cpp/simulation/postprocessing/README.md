# Postprocessing Scripts

This directory holds the **3D-specific** Python helpers that have not yet
been ported to the Rust analysis binary. For 2D, use Rust.

| What | Where |
|---|---|
| 2D phi snapshots, movies, study pipeline | `rust/vtk_viewer/` (`cell_analyze`) |
| Per-paper plotting and analysis | `cpp/simulation/study/<study>/` |
| Regression / correctness tests | `cpp/simulation/tests/python/` (pytest) |

## Contents

| File | Purpose |
|------|---------|
| `visualize_3d.py` | 3D isosurface rendering via PyVista. Reads 3D checkpoints. |
| `validate_3d.py` | 3D checkpoint integrity / cross-checkpoint comparison. |
| `analyze_msd_3d.py` | 3D mean-squared-displacement analysis. |

These are kept here only because `cell_analyze` is 2D-only today. When
3D support lands in Rust, this directory should be emptied and removed.

## Usage

```bash
# Render
python visualize_3d.py <checkpoint.bin>
python visualize_3d.py <output_dir> --animate

# Validate
python validate_3d.py <checkpoint.bin>
python validate_3d.py <ckpt_a.bin> --compare <ckpt_b.bin>

# 3D MSD
python analyze_msd_3d.py <output_dir>
```

## Dependencies

```bash
pip install -r requirements.txt
```
