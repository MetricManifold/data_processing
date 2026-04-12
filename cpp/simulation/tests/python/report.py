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
    html.append('<meta charset="utf-8">')
    html.append("<title>Cell Sim Test Report</title>")
    html.append("<style>")
    html.append("""
body { font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 15px; background: #fafafa; font-size: 13px; }
h1 { color: #222; border-bottom: 2px solid #333; padding-bottom: 6px; font-size: 20px; margin-bottom: 10px; }
h2 { color: #444; margin-top: 20px; font-size: 16px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
table { border-collapse: collapse; background: white; font-size: 12px; }
th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
th { background: #f0f0f0; font-weight: 600; }
.pass { color: #1a7; font-weight: bold; }
.fail { color: #c22; font-weight: bold; }
.info { color: #888; }
.metric-val { font-family: monospace; font-size: 12px; }
img.snap { width: 190px; border: 1px solid #ccc; border-radius: 3px; }
img.chart { width: 320px; border: 1px solid #ccc; border-radius: 3px; }
.test-card { display: flex; gap: 12px; align-items: flex-start; background: white;
             border: 1px solid #ddd; border-radius: 6px; padding: 10px; margin: 8px 0; }
.test-card .visuals { display: flex; flex-direction: row; gap: 8px; flex-shrink: 0; }
.test-card .data { flex: 1; min-width: 280px; }
.test-card h3 { margin: 0 0 6px 0; font-size: 14px; color: #333; }
.summary-bar { display: flex; gap: 20px; background: white; border: 1px solid #ddd;
               border-radius: 6px; padding: 10px 15px; margin: 10px 0; align-items: center; }
.summary-bar .stat { font-size: 18px; font-weight: 700; }
.summary-bar .label { font-size: 11px; color: #666; }
.summary-table { width: 100%; font-size: 12px; margin-top: 8px; }
.summary-table td { padding: 3px 6px; }
.summary-table .test-name { font-weight: 600; }
    """)
    html.append("</style></head><body>")
    html.append(f"<h1>Cell Simulation Test Report</h1>")
    html.append(f"<p style='color:#666;margin:0;'>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")

    # Summary bar
    n_pass = sum(1 for m_list in _metrics.values() for m in m_list if m.get("status") == "PASS")
    n_fail = sum(1 for m_list in _metrics.values() for m in m_list if m.get("status") == "FAIL")
    n_info = sum(1 for m_list in _metrics.values() for m in m_list if m.get("status") == "INFO")
    n_tests = len(_metrics)
    html.append(f'<div class="summary-bar">')
    html.append(f'<div><div class="stat pass">{n_pass}</div><div class="label">PASS</div></div>')
    if n_fail:
        html.append(f'<div><div class="stat fail">{n_fail}</div><div class="label">FAIL</div></div>')
    html.append(f'<div><div class="stat info">{n_info}</div><div class="label">INFO</div></div>')
    html.append(f'<div><div class="stat">{len(_snapshots)}</div><div class="label">Snapshots</div></div>')
    html.append(f'<div><div class="stat">{len(_timeseries)}</div><div class="label">Charts</div></div>')
    html.append(f'<div><div class="stat">{n_tests}</div><div class="label">Tests</div></div>')
    html.append(f'</div>')

    # Summary table — one row per test, key metric only
    html.append('<h2>Summary</h2>')
    html.append('<table class="summary-table"><tr><th>Test</th><th>Key Metric</th>'
                '<th>Measured</th><th>Expected</th><th>Status</th></tr>')
    for test_name, entries in sorted(_metrics.items()):
        # Pick the most interesting metric (first one with expected value, or first)
        key_entry = next((e for e in entries if "expected" in e and e.get("status") != "INFO"), entries[0])
        val = f"{key_entry['value']:.4g}" if isinstance(key_entry["value"], float) else str(key_entry["value"])
        exp = f"{key_entry['expected']:.4g}" if "expected" in key_entry and isinstance(key_entry["expected"], float) else "—"
        status = key_entry.get("status", "")
        scls = status.lower()
        html.append(f'<tr><td class="test-name">{test_name}</td><td>{key_entry["key"]}</td>'
                    f'<td class="metric-val">{val}</td><td class="metric-val">{exp}</td>'
                    f'<td class="{scls}">{status}</td></tr>')
    html.append('</table>')

    # Test cards — compact layout
    html.append("<h2>Detailed Results</h2>")
    for test_name, entries in sorted(_metrics.items()):
        html.append(f'<div class="test-card">')

        # Visuals column (snapshot + chart stacked)
        has_visuals = test_name in _snapshots or test_name in _timeseries
        if has_visuals:
            html.append(f'<div class="visuals">')
            if test_name in _snapshots:
                html.append(f'<img src="{_snapshots[test_name]}" class="snap">')
            if test_name in _timeseries:
                html.append(f'<img src="{_timeseries[test_name]}" class="chart">')
            html.append(f'</div>')

        # Data column
        html.append(f'<div class="data">')
        html.append(f'<h3>{test_name}</h3>')
        html.append("<table><tr><th>Metric</th><th>Measured</th><th>Expected</th><th>Tol</th><th></th></tr>")
        for e in entries:
            val = f"{e['value']:.6g}" if isinstance(e["value"], float) else str(e["value"])
            exp = f"{e['expected']:.6g}" if "expected" in e and isinstance(e["expected"], float) else "—"
            tol = str(e.get("tolerance", "—"))
            unit = f" {e['unit']}" if e.get("unit") else ""
            status = e.get("status", "")
            scls = status.lower()
            html.append(f'<tr><td>{e["key"]}{unit}</td><td class="metric-val">{val}</td>'
                       f'<td class="metric-val">{exp}</td><td>{tol}</td>'
                       f'<td class="{scls}">{status}</td></tr>')
        html.append("</table>")
        html.append(f'</div></div>')

    # Remaining snapshots (tests without metrics)
    remaining_snaps = {k: v for k, v in _snapshots.items() if k not in _metrics}
    remaining_ts = {k: v for k, v in _timeseries.items() if k not in _metrics}
    if remaining_snaps or remaining_ts:
        html.append("<h2>Additional</h2>")
        html.append('<div style="display:flex;flex-wrap:wrap;gap:10px;">')
        for test_name, fname in sorted({**remaining_snaps, **remaining_ts}.items()):
            html.append(f'<div style="background:white;border:1px solid #ddd;border-radius:4px;padding:8px;">')
            html.append(f'<img src="{fname}" class="snap"><br><small>{test_name}</small></div>')
        html.append("</div>")

    html.append("</body></html>")

    report_path = d / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
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
