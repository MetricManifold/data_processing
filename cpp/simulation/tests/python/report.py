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
_extras = {}        # test_name -> list of (filename, caption) extra frames
_extras = {}        # test_name -> list of (filename, caption) extra frames
_timeseries = {}    # test_name -> filename
_trajectories = {}  # test_name -> filename (xy path chart)
_skipped = {}       # test_name -> reason
_descriptions = {}  # test_name -> human-readable description
_collected = set()  # nodeids seen during collection (full inventory)
_report_dir = None


def record_description(test_name, text):
    """Attach a one-line description to a test card.

    The first non-empty call wins so the recording site (``record_metric``
    callers) can't accidentally clobber a richer description set earlier.
    """
    if not text:
        return
    trimmed = str(text).strip()
    if not trimmed:
        return
    _descriptions.setdefault(test_name, trimmed)


def record_skip(test_name, reason):
    _skipped[test_name] = reason


def get_report_dir():
    global _report_dir
    if _report_dir is None:
        _report_dir = Path(__file__).parent / "test_report"
        _report_dir.mkdir(exist_ok=True)
    return _report_dir


def _safe_name(test_name):
    # pytest nodeids include path separators ("cpp/simulation/tests/python/...").
    # Without stripping them, savefig() lands artifacts inside a nested tree
    # whose parents don't exist and the test fails on write. Strip both
    # slash variants so every artifact is a flat file in the report dir.
    return (test_name
            .replace("::", "__")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("[", "_")
            .replace("]", "")
            .replace(" ", "_"))


def _short_name(test_name):
    """Human-readable display name.

    ``path/to/test_file.py::TestClass::test_method``  \u2192  ``TestClass :: test_method``
    ``test_file.py::test_method``                     \u2192  ``test_method``
    """
    parts = test_name.split("::")
    # Drop the file path (first part) unconditionally.
    if len(parts) >= 3:
        return " :: ".join(parts[1:])
    if len(parts) == 2:
        return parts[1]
    return test_name


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


def record_timeseries(test_name, x, y_dict, xlabel="Time", ylabel="Value", title="", ylog=False):
    """Save a time series plot. y_dict: {label: y_array}.

    Set ``ylog=True`` for a log-scaled y axis (useful when series span
    several orders of magnitude, e.g. error growth across REMAP events).
    Non-positive and NaN entries are dropped when ``ylog``.
    """
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
        y_arr = np.asarray(y, dtype=float)
        x_arr = np.asarray(x[:len(y_arr)], dtype=float)
        mask = np.isfinite(y_arr)
        if ylog:
            mask &= y_arr > 0
        ax.plot(x_arr[mask], y_arr[mask], "-o", label=label, linewidth=1.5, markersize=4)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title or test_name, fontsize=10)
    if ylog:
        ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    _timeseries[test_name] = fname


def record_trajectory(test_name, xs, ys=None, title="",
                      xlabel="x (px)", ylabel="y (px)",
                      styles=None):
    """Save an (x, y) path plot with start/end markers.

    Two calling conventions:
      * Single path: ``record_trajectory(name, xs, ys)`` — xs, ys are
        1-D arrays of the same length.
      * Multiple paths: ``record_trajectory(name, {"cell 0": (xs0, ys0),
        "cell 1": (xs1, ys1)})`` — plots every path in one figure with
        its own colour, shared axes, and one legend.

    ``styles`` (dict-mode only): optional ``{label: {color, linestyle,
    linewidth, ...}}`` mapping. Any matplotlib Line2D kwargs supplied
    here override the auto-assigned colour. Useful to group paired
    paths (e.g. two cells of the same doublet) under a shared colour
    while differentiating them by linestyle.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    d = get_report_dir()
    fname = f"{_safe_name(test_name)}_traj.png"
    out = d / fname

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    if isinstance(xs, dict):
        paths = xs
        styles = styles or {}
        colors = plt.get_cmap("tab10").colors
        for k, (label, (px, py)) in enumerate(paths.items()):
            kw = {"color": colors[k % len(colors)], "linestyle": "-",
                  "linewidth": 1.5, "alpha": 0.85}
            kw.update(styles.get(label, {}))
            ax.plot(px, py, label=label, **kw)
            marker_color = kw["color"]
            ax.plot(px[0], py[0], "o", color=marker_color, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.2)
            ax.plot(px[-1], py[-1], "s", color=marker_color, markersize=7,
                    markeredgecolor="black", markeredgewidth=1.0)
    else:
        ax.plot(xs, ys, "-", color="#2266cc", linewidth=1.5, alpha=0.85)
        ax.plot(xs[0], ys[0], "o", color="#1a7a3a", markersize=8, label="start")
        ax.plot(xs[-1], ys[-1], "s", color="#c22222", markersize=8, label="end")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title or test_name, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    # Legend outside the axes to the right, so it cannot overlap the paths.
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    _trajectories[test_name] = fname


def record_composite_frame(test_name, phi_composite, caption,
                           vmin=0, vmax=None, slug=None):
    """Save a single-panel composite φ² (or other scalar) frame.

    Use this when you want an extra frame attached to the same test
    card as an existing comparison panel (e.g. the final state of a
    long simulation where no CPU reference is available to compare
    against). The frame is rendered in the visuals row after the
    primary snapshot, with the caption below it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    import numpy as _np
    arr = _np.asarray(phi_composite, dtype=_np.float64)
    if vmax is None:
        vmax = max(1.0, float(arr.max()) if arr.size else 1.0)

    d = get_report_dir()
    slug = slug or caption.replace(" ", "_").replace("/", "_")
    fname = f"{_safe_name(test_name)}__frame_{_safe_name(slug)}.png"
    out = d / fname

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 4.6))
    im = ax.imshow(arr, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(caption, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    _extras.setdefault(test_name, []).append((fname, caption))


def record_phi_from_checkpoint(test_name, chk, title=""):
    """Save a single-panel composite φ² (or other scalar) frame.

    Use this when you want an extra frame attached to the same test
    card as an existing comparison panel (e.g. the final state of a
    long simulation where no CPU reference is available to compare
    against). The frame is rendered in the visuals row after the
    primary snapshot, with the caption below it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    import numpy as _np
    arr = _np.asarray(phi_composite, dtype=_np.float64)
    if vmax is None:
        vmax = max(1.0, float(arr.max()) if arr.size else 1.0)

    d = get_report_dir()
    slug = slug or caption.replace(" ", "_").replace("/", "_")
    fname = f"{_safe_name(test_name)}__frame_{_safe_name(slug)}.png"
    out = d / fname

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 4.6))
    im = ax.imshow(arr, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(caption, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    _extras.setdefault(test_name, []).append((fname, caption))


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


def record_comparison_panel(test_name, sim_grid, ref_grid, title=""):
    """Save a 3-panel comparison: sim | CPU-ref | |error|.

    Used by Phase H tests to visualise sim vs cpu_reference parity.
    All three panels share the same spatial layout; the error panel
    uses a log-scale colormap so both f32 round-off (~1e-7) and real
    drift (~1e-3) are visible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        return

    d = get_report_dir()
    fname = f"{_safe_name(test_name)}.png"
    out = d / fname

    err = np.abs(sim_grid.astype(np.float64) - ref_grid.astype(np.float64))
    err_floor = max(1e-10, float(err[err > 0].min()) if np.any(err > 0) else 1e-10)
    err_top = max(err_floor * 10, float(err.max()) if err.max() > 0 else 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    vmax = max(1.0, float(max(sim_grid.max(), ref_grid.max())))
    axes[0].imshow(sim_grid, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title("sim ϕ²  (composite)", fontsize=10)
    axes[1].imshow(ref_grid, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("CPU reference ϕ²  (composite)", fontsize=10)
    im = axes[2].imshow(err, origin="lower", cmap="viridis",
                        norm=LogNorm(vmin=err_floor, vmax=err_top))
    axes[2].set_title(f"|Δϕ²|  (max={err.max():.2e})", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=axes[2], shrink=0.75)
    if title:
        fig.suptitle(title, fontsize=11, y=0.99)
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    _snapshots[test_name] = fname


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
img.snap { height: 220px; width: auto; border: 1px solid #ccc; border-radius: 3px; }
img.chart { height: 220px; width: auto; border: 1px solid #ccc; border-radius: 3px; }
img.panel { height: 220px; width: auto; border: 1px solid #ccc; border-radius: 3px; }
figure.extra-frame { margin: 0; display: flex; flex-direction: column;
                     align-items: center; gap: 2px; }
figure.extra-frame figcaption { font-size: 11px; color: #555; text-align: center;
                                 max-width: 220px; }
.test-card { display: flex; flex-direction: column; gap: 10px; background: white;
             border: 1px solid #ddd; border-radius: 6px; padding: 10px; margin: 8px 0; }
.test-card .header { }
.test-card .body { display: flex; flex-direction: row; gap: 12px;
                   align-items: flex-start; flex-wrap: wrap; }
.test-card .visuals { display: flex; flex-direction: row; gap: 8px; flex-shrink: 0; }
.test-card .data { flex: 1; min-width: 280px; }
.test-card h3 { margin: 0 0 4px 0; font-size: 14px; color: #333; }
.test-card .desc { margin: 0; font-size: 12px; color: #555; line-height: 1.4;
                   font-style: italic; max-width: 960px; }
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
    n_skip = len(_skipped)
    n_untested = len(_collected - set(_metrics) - set(_skipped))
    html.append(f'<div class="summary-bar">')
    html.append(f'<div><div class="stat pass">{n_pass}</div><div class="label">PASS</div></div>')
    if n_fail:
        html.append(f'<div><div class="stat fail">{n_fail}</div><div class="label">FAIL</div></div>')
    html.append(f'<div><div class="stat info">{n_info}</div><div class="label">INFO</div></div>')
    if n_skip:
        html.append(f'<div><div class="stat" style="color:#b58900">{n_skip}</div><div class="label">SKIPPED</div></div>')
    if n_untested:
        html.append(f'<div><div class="stat" style="color:#888">{n_untested}</div><div class="label">NOT RUN</div></div>')
    html.append(f'<div><div class="stat">{len(_snapshots)}</div><div class="label">Snapshots</div></div>')
    html.append(f'<div><div class="stat">{len(_timeseries) + len(_trajectories)}</div><div class="label">Charts</div></div>')
    html.append(f'<div><div class="stat">{n_tests}</div><div class="label">Tests w/ metrics</div></div>')
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
        html.append(f'<tr><td class="test-name">{_short_name(test_name)}</td><td>{key_entry["key"]}</td>'
                    f'<td class="metric-val">{val}</td><td class="metric-val">{exp}</td>'
                    f'<td class="{scls}">{status}</td></tr>')
    # Skipped tests: one row each with the reason in the "Key Metric" column
    for test_name, reason in sorted(_skipped.items()):
        html.append(f'<tr><td class="test-name">{_short_name(test_name)}</td>'
                    f'<td colspan="3" style="color:#666;font-style:italic;">{reason}</td>'
                    f'<td style="color:#b58900;font-weight:700;">SKIP</td></tr>')
    # Tests collected this session that produced neither metrics nor a skip
    # record (— typically: filtered out by node-id selection or collected
    # but errored during setup before record_skip could fire).
    untested = sorted(_collected - set(_metrics) - set(_skipped))
    for test_name in untested:
        html.append(f'<tr><td class="test-name">{_short_name(test_name)}</td>'
                    f'<td colspan="3" style="color:#888;font-style:italic;">not run this session</td>'
                    f'<td style="color:#888;font-weight:700;">—</td></tr>')
    html.append('</table>')

    # Test cards — title + description on top, visuals + metrics below
    html.append("<h2>Detailed Results</h2>")
    for test_name, entries in sorted(_metrics.items()):
        html.append(f'<div class="test-card">')

        # Header: title + description always span full card width
        html.append(f'<div class="header">')
        html.append(f'<h3>{_short_name(test_name)}</h3>')
        if test_name in _descriptions:
            html.append(f'<p class="desc">{_descriptions[test_name]}</p>')
        html.append(f'</div>')

        # Body: visuals (left) + metrics table (right)
        html.append(f'<div class="body">')

        has_visuals = (test_name in _snapshots or test_name in _timeseries
                       or test_name in _trajectories
                       or test_name in _extras)
        if has_visuals:
            html.append(f'<div class="visuals">')
            if test_name in _snapshots:
                html.append(f'<img src="{_snapshots[test_name]}" class="snap">')
            for fname, caption in _extras.get(test_name, []):
                html.append(f'<figure class="extra-frame">'
                            f'<img src="{fname}" class="snap">'
                            f'<figcaption>{caption}</figcaption></figure>')
            if test_name in _trajectories:
                html.append(f'<img src="{_trajectories[test_name]}" class="chart">')
            if test_name in _timeseries:
                html.append(f'<img src="{_timeseries[test_name]}" class="chart">')
            html.append(f'</div>')

        html.append(f'<div class="data">')
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
        html.append(f'</div>')    # .data

        html.append(f'</div>')    # .body
        html.append(f'</div>')    # .test-card

    # Skipped tests (e.g. @slow without --run-slow)
    if _skipped:
        html.append('<h2>Skipped</h2>')
        html.append('<table><tr><th>Test</th><th>Reason</th></tr>')
        for tn, reason in sorted(_skipped.items()):
            html.append(f'<tr><td class="test-name">{_short_name(tn)}</td><td>{reason}</td></tr>')
        html.append('</table>')

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

def pytest_collection_modifyitems(config, items):
    """Record every collected test so the report can list the full inventory.

    Note: pytest only collects items matching the CLI selection (``-k``,
    file/node paths). Tests excluded by node-id selection are not seen
    here — this captures the *current session's* inventory.
    """
    for item in items:
        _collected.add(item.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if _metrics or _snapshots or _timeseries:
        generate_report()
