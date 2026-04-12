"""
pytest plugin: collect test metrics and generate a final HTML report
with phase field snapshots and quantitative results.
"""
import json
import time
from pathlib import Path

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Shared state: metrics collected during tests
# ---------------------------------------------------------------------------

_metrics = {}  # test_name -> dict of key-value pairs
_snapshots = {}  # test_name -> path to snapshot PNG
_report_dir = None


def get_report_dir():
    global _report_dir
    if _report_dir is None:
        _report_dir = Path(__file__).parent / "test_report"
        _report_dir.mkdir(exist_ok=True)
    return _report_dir


def record_metric(test_name, key, value):
    """Record a metric for the final report."""
    _metrics.setdefault(test_name, {})[key] = value


def record_snapshot(test_name, phi_2d, title="", vmin=0, vmax=1.1):
    """Save a phase field snapshot as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    d = get_report_dir()
    safe_name = test_name.replace("::", "__").replace("[", "_").replace("]", "")
    out = d / f"{safe_name}.png"

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(phi_2d, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(title or test_name, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    _snapshots[test_name] = str(out.name)


def record_phi_from_checkpoint(test_name, chk, title=""):
    """Composite all cell phi fields onto the domain grid and save snapshot."""
    Nx = chk["params"]["Nx"]
    Ny = chk["params"]["Ny"]
    halo = chk["params"].get("halo_width", 4)
    grid = np.zeros((Ny, Nx), dtype=np.float64)

    for cell in chk["cells"]:
        x0, y0, x1, y1 = cell["bbox"]
        phi = cell["phi"]
        h, w = phi.shape
        bw, bh = x1 - x0, y1 - y0
        inner = phi[halo:halo+bh, halo:halo+bw]
        for ly in range(bh):
            gy = (y0 + ly) % Ny
            for lx in range(bw):
                gx = (x0 + lx) % Nx
                grid[gy, gx] += inner[ly, lx] ** 2

    record_snapshot(test_name, grid, title=title)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report():
    """Generate HTML report with metrics and snapshots."""
    d = get_report_dir()

    # Save metrics JSON
    with open(d / "metrics.json", "w") as f:
        json.dump(_metrics, f, indent=2, default=str)

    # Build HTML
    html = []
    html.append("<!DOCTYPE html><html><head>")
    html.append("<title>Cell Sim Test Report</title>")
    html.append("<style>")
    html.append("body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }")
    html.append("h1 { color: #333; } h2 { color: #555; margin-top: 30px; }")
    html.append("table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
    html.append("th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }")
    html.append("th { background: #f5f5f5; }")
    html.append(".pass { color: #2a2; font-weight: bold; } .fail { color: #c22; font-weight: bold; }")
    html.append(".snapshot { max-width: 300px; margin: 5px; border: 1px solid #ccc; }")
    html.append(".grid { display: flex; flex-wrap: wrap; }")
    html.append("</style></head><body>")
    html.append(f"<h1>Cell Simulation Test Report</h1>")
    html.append(f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html.append(f"<p>Tests with metrics: {len(_metrics)} | Snapshots: {len(_snapshots)}</p>")

    # Metrics table
    if _metrics:
        html.append("<h2>Test Metrics</h2>")
        html.append("<table><tr><th>Test</th><th>Metric</th><th>Value</th></tr>")
        for test_name, metrics in sorted(_metrics.items()):
            first = True
            for k, v in sorted(metrics.items()):
                tname = test_name.split("::")[-1] if first else ""
                if isinstance(v, float):
                    vstr = f"{v:.6g}"
                else:
                    vstr = str(v)
                html.append(f"<tr><td>{tname}</td><td>{k}</td><td>{vstr}</td></tr>")
                first = False
        html.append("</table>")

    # Snapshots gallery
    if _snapshots:
        html.append("<h2>Phase Field Snapshots</h2>")
        html.append('<div class="grid">')
        for test_name, fname in sorted(_snapshots.items()):
            label = test_name.split("::")[-1]
            html.append(f'<div><img src="{fname}" class="snapshot"><br><small>{label}</small></div>')
        html.append("</div>")

    html.append("</body></html>")

    report_path = d / "report.html"
    with open(report_path, "w") as f:
        f.write("\n".join(html))
    print(f"\n{'='*60}")
    print(f"Test report: {report_path}")
    print(f"Metrics: {d / 'metrics.json'}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------

def pytest_sessionfinish(session, exitstatus):
    """Generate report after all tests complete."""
    if _metrics or _snapshots:
        generate_report()
