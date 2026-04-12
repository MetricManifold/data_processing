"""
pytest plugin: collect test metrics and generate a final HTML report
with phase field snapshots, time series plots, and quantitative results.
"""
import json
import time
from pathlib import Path

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_metrics = {}       # test_name -> list of {key, value, expected, tolerance, status}
_snapshots = {}     # test_name -> filename
_timeseries = {}    # test_name -> filename
_report_dir = None


def get_report_dir():
    global _report_dir
    if _report_dir is None:
        _report_dir = Path(__file__).parent / "test_report"
        _report_dir.mkdir(exist_ok=True)
    return _report_dir


def _safe_name(test_name):
    return test_name.replace("::", "__").replace("[", "_").replace("]", "").replace(" ", "_")


# ---------------------------------------------------------------------------
# Recording functions
# ---------------------------------------------------------------------------

def record_metric(test_name, key, value, expected=None, tolerance=None, unit=""):
    """Record a metric with optional expected value and tolerance."""
    entry = {"key": key, "value": value, "unit": unit}
    if expected is not None:
        entry["expected"] = expected
        if tolerance is not None:
            entry["tolerance"] = tolerance
            if isinstance(tolerance, str) and tolerance.endswith("%"):
                pct = float(tolerance[:-1]) / 100
                entry["status"] = "PASS" if abs(value - expected) <= abs(expected * pct) else "FAIL"
            else:
                entry["status"] = "PASS" if abs(value - expected) <= float(tolerance) else "FAIL"
        else:
            entry["status"] = "INFO"
    else:
        entry["status"] = "INFO"
    _metrics.setdefault(test_name, []).append(entry)


def record_snapshot(test_name, phi_2d, title="", vmin=0, vmax=1.1):
    """Save a phase field snapshot as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    d = get_report_dir()
    fname = f"{_safe_name(test_name)}.png"
    out = d / fname

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(phi_2d, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(title or test_name, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    _snapshots[test_name] = fname


def record_timeseries(test_name, x, y_dict, xlabel="Time", ylabel="Value", title=""):
    """Save a time series plot. y_dict: {label: y_array}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    d = get_report_dir()
    fname = f"{_safe_name(test_name)}_ts.png"
    out = d / fname

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    for label, y in y_dict.items():
        ax.plot(x[:len(y)], y, label=label, linewidth=1.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title or test_name, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    _timeseries[test_name] = fname


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
    d = get_report_dir()

    # Save metrics JSON
    with open(d / "metrics.json", "w") as f:
        json.dump(_metrics, f, indent=2, default=str)

    html = []
    html.append("<!DOCTYPE html><html><head>")
    html.append("<title>Cell Sim Test Report</title>")
    html.append("<style>")
    html.append("""
body { font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }
h1 { color: #222; border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { color: #444; margin-top: 35px; }
h3 { color: #555; margin-top: 20px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; background: white; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 13px; }
th { background: #f0f0f0; font-weight: 600; }
.pass { color: #1a7; font-weight: bold; }
.fail { color: #c22; font-weight: bold; }
.info { color: #666; }
.metric-val { font-family: monospace; }
.snapshot { max-width: 320px; margin: 8px; border: 1px solid #ccc; border-radius: 4px; }
.ts-plot { max-width: 600px; margin: 8px; border: 1px solid #ccc; border-radius: 4px; }
.gallery { display: flex; flex-wrap: wrap; gap: 15px; }
.card { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 10px; }
.card small { display: block; margin-top: 4px; color: #666; }
.summary { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 15px; margin: 15px 0; }
    """)
    html.append("</style></head><body>")
    html.append(f"<h1>Cell Simulation Test Report</h1>")
    html.append(f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")

    # Summary
    n_pass = sum(1 for m_list in _metrics.values() for m in m_list if m.get("status") == "PASS")
    n_fail = sum(1 for m_list in _metrics.values() for m in m_list if m.get("status") == "FAIL")
    n_info = sum(1 for m_list in _metrics.values() for m in m_list if m.get("status") == "INFO")
    html.append(f'<div class="summary">')
    html.append(f'<b>Metrics:</b> <span class="pass">{n_pass} PASS</span>')
    if n_fail:
        html.append(f' | <span class="fail">{n_fail} FAIL</span>')
    html.append(f' | <span class="info">{n_info} info</span>')
    html.append(f' &nbsp;|&nbsp; <b>Snapshots:</b> {len(_snapshots)} &nbsp;|&nbsp; <b>Time series:</b> {len(_timeseries)}')
    html.append(f'</div>')

    # Metrics table — grouped by test
    if _metrics:
        html.append("<h2>Quantitative Results</h2>")
        for test_name, entries in sorted(_metrics.items()):
            html.append(f"<h3>{test_name}</h3>")

            # Show time series if available
            if test_name in _timeseries:
                html.append(f'<img src="{_timeseries[test_name]}" class="ts-plot">')

            # Show snapshot if available
            if test_name in _snapshots:
                html.append(f'<img src="{_snapshots[test_name]}" class="snapshot" style="float:right; margin-left:15px;">')

            html.append("<table><tr><th>Metric</th><th>Measured</th><th>Expected</th><th>Tolerance</th><th>Status</th></tr>")
            for e in entries:
                val = f"{e['value']:.6g}" if isinstance(e["value"], float) else str(e["value"])
                exp = f"{e['expected']:.6g}" if "expected" in e and isinstance(e["expected"], float) else str(e.get("expected", "—"))
                tol = str(e.get("tolerance", "—"))
                unit = f" {e['unit']}" if e.get("unit") else ""
                status = e.get("status", "")
                scls = status.lower()
                html.append(f'<tr><td>{e["key"]}{unit}</td><td class="metric-val">{val}</td>'
                           f'<td class="metric-val">{exp}</td><td>{tol}</td>'
                           f'<td class="{scls}">{status}</td></tr>')
            html.append("</table>")
            html.append('<div style="clear:both;"></div>')

    # Remaining snapshots (tests without metrics)
    remaining_snaps = {k: v for k, v in _snapshots.items() if k not in _metrics}
    if remaining_snaps:
        html.append("<h2>Additional Snapshots</h2>")
        html.append('<div class="gallery">')
        for test_name, fname in sorted(remaining_snaps.items()):
            html.append(f'<div class="card"><img src="{fname}" class="snapshot"><small>{test_name}</small></div>')
        html.append("</div>")

    # Remaining time series
    remaining_ts = {k: v for k, v in _timeseries.items() if k not in _metrics}
    if remaining_ts:
        html.append("<h2>Additional Time Series</h2>")
        for test_name, fname in sorted(remaining_ts.items()):
            html.append(f'<div class="card"><img src="{fname}" class="ts-plot"><small>{test_name}</small></div>')

    html.append("</body></html>")

    report_path = d / "report.html"
    with open(report_path, "w") as f:
        f.write("\n".join(html))
    print(f"\n{'='*60}")
    print(f"Test report: {report_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------

def pytest_sessionfinish(session, exitstatus):
    if _metrics or _snapshots or _timeseries:
        generate_report()
