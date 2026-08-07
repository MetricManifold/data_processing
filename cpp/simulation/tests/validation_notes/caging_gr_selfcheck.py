"""Measure the cage-relative construction from real trajectories.

Chain (each step feeds the next, nothing imported from the literature):
  1. g(r)          -> first minimum r_min  = neighbour cutoff
  2. cage-relative displacements using that cutoff
  3. alpha_2(tau)  -> peak tau*            = observation window T_obs
  4. van Hove self at tau* -> first minimum = mobility threshold a_c

Two self-checks with exact analytic answers are run first; if either fails the
measurement below is meaningless.
"""
import sys, glob, os
import numpy as np

R = 49.0

def load(path):
    t, cid, x, y = [], [], [], []
    for line in open(path):
        if line[0] == '#':
            continue
        p = line.split()
        t.append(float(p[0])); cid.append(int(p[1]))
        x.append(float(p[2])); y.append(float(p[3]))
    t = np.array(t); cid = np.array(cid); x = np.array(x); y = np.array(y)
    ids = np.unique(cid); times = np.unique(t)
    nt, nc = len(times), len(ids)
    X = np.full((nt, nc), np.nan); Y = np.full((nt, nc), np.nan)
    ti = {v: i for i, v in enumerate(times)}; ci = {v: i for i, v in enumerate(ids)}
    for k in range(len(t)):
        X[ti[t[k]], ci[cid[k]]] = x[k]; Y[ti[t[k]], ci[cid[k]]] = y[k]
    keep = ~np.isnan(X).any(axis=1)
    return times[keep], X[keep], Y[keep]

def unwrap(A, L):
    """Per-column continuous unwrap against the previous unwrapped frame."""
    out = np.empty_like(A); out[0] = A[0]
    for i in range(1, len(A)):
        d = A[i] - A[i-1]
        d -= L * np.round(d / L)
        out[i] = out[i-1] + d
    return out

def pair_gr(X, Y, L, nbins=200, rmax=None):
    """Radial distribution function, minimum-image, averaged over frames."""
    rmax = rmax or L / 2
    edges = np.linspace(0, rmax, nbins + 1)
    h = np.zeros(nbins)
    nt, nc = X.shape
    frames = np.linspace(0, nt - 1, min(nt, 200)).astype(int)
    for f in frames:
        dx = X[f][:, None] - X[f][None, :]
        dy = Y[f][:, None] - Y[f][None, :]
        dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
        r = np.sqrt(dx*dx + dy*dy)
        r = r[~np.eye(nc, dtype=bool)]
        h += np.histogram(r, bins=edges)[0]
    h /= len(frames)
    rc = 0.5 * (edges[1:] + edges[:-1])
    shell = np.pi * (edges[1:]**2 - edges[:-1]**2)
    rho = nc / L**2
    g = h / (shell * rho * nc)
    return rc, g

def first_min_after_peak(x, y):
    """First local minimum after the global maximum, smoothed."""
    k = np.ones(5) / 5
    ys = np.convolve(y, k, mode='same')
    pk = int(np.argmax(ys))
    for i in range(pk + 2, len(ys) - 2):
        if ys[i] <= ys[i-1] and ys[i] <= ys[i+1]:
            return x[i], pk
    return x[int(np.argmin(ys[pk:])) + pk], pk

def neighbours(X, Y, f, L, cutoff):
    dx = X[f][:, None] - X[f][None, :]
    dy = Y[f][:, None] - Y[f][None, :]
    dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
    d2 = dx*dx + dy*dy
    np.fill_diagonal(d2, np.inf)
    return d2 < cutoff**2

def cage_disp(XU, YU, f0, lag, nb):
    """Cage-relative displacement vectors for origin f0 over `lag` frames."""
    dx = XU[f0+lag] - XU[f0]; dy = YU[f0+lag] - YU[f0]
    z = nb.sum(axis=1)
    mx = np.where(z > 0, (nb * dx[None, :]).sum(axis=1) / np.maximum(z, 1), 0.0)
    my = np.where(z > 0, (nb * dy[None, :]).sum(axis=1) / np.maximum(z, 1), 0.0)
    return dx - mx, dy - my, z

# ---------------------------------------------------------------- self-checks
def selfchecks():
    rng = np.random.default_rng(0)
    L, nc = 100.0, 60
    X0 = rng.uniform(0, L, nc); Y0 = rng.uniform(0, L, nc)
    nb = neighbours(np.array([X0]), np.array([Y0]), 0, L, 25.0)
    z = nb.sum(axis=1)

    # (1) uniform translation -> cage-relative displacement must be EXACTLY 0
    d = np.array([3.7, -1.9])
    XU = np.array([X0, X0 + d[0]]); YU = np.array([Y0, Y0 + d[1]])
    cx, cy, _ = cage_disp(XU, YU, 0, 1, nb)
    m0 = z > 0
    e1 = np.max(np.abs(np.concatenate([cx[m0], cy[m0]])))

    # (2) uncorrelated displacements -> CR-MSD/MSD = 1 + 1/z  exactly
    trials, num, den, zs = 4000, 0.0, 0.0, []
    for _ in range(trials):
        ux = rng.normal(0, 1, nc); uy = rng.normal(0, 1, nc)
        XU = np.array([X0, X0 + ux]); YU = np.array([Y0, Y0 + uy])
        cx, cy, _ = cage_disp(XU, YU, 0, 1, nb)
        m = z > 0
        num += np.sum(cx[m]**2 + cy[m]**2); den += np.sum(ux[m]**2 + uy[m]**2)
        zs.append(z[m])
    ratio = num / den
    zh = 1.0 / np.mean(1.0 / np.concatenate(zs))   # harmonic mean of z
    expect = 1.0 + 1.0 / zh
    print(f"  [check 1] uniform translation -> max|dr_cage| = {e1:.3e}  (must be ~0)")
    print(f"  [check 2] uncorrelated CR/MSD = {ratio:.4f}, expected 1+1/z = {expect:.4f}"
          f"  (dev {100*abs(ratio-expect)/expect:.2f}%)")
    return e1 < 1e-10 and abs(ratio - expect) / expect < 0.02

# ---------------------------------------------------------------- measurement
print("=== SELF-CHECKS (analytic) ===")
if not selfchecks():
    print("SELF-CHECK FAILED - measurement below is not trustworthy"); sys.exit(1)

paths = sorted(glob.glob(sys.argv[1]))
print(f"\n=== MEASUREMENT ({len(paths)} runs) ===")
L = float(sys.argv[2]); TAU = 10000.0

grs, rmins = [], []
for p in paths:
    t, X, Y = load(p)
    rc, g = pair_gr(X, Y, L)
    rmin, _ = first_min_after_peak(rc, g)
    grs.append((rc, g)); rmins.append(rmin)
rmin = float(np.median(rmins))
rc, g = grs[0]
pk = rc[int(np.argmax(np.convolve(g, np.ones(5)/5, mode='same')))]
print(f"1. g(r): first peak {pk:.1f} px ({pk/(2*R):.2f} diam), "
      f"first minimum {rmin:.1f} px ({rmin/(2*R):.2f} diam)  [median over runs]")

# alpha_2 and van Hove of the CAGE-RELATIVE displacement
lags = np.unique(np.geomspace(1, 400, 26).astype(int))
a2 = np.zeros(len(lags)); crmsd = np.zeros(len(lags)); absmsd = np.zeros(len(lags))
store = {}
for p in paths:
    t, X, Y = load(p)
    dt = t[1] - t[0]
    XU, YU = unwrap(X, L), unwrap(Y, L)
    nt = len(t)
    origins = np.linspace(0, nt - max(lags) - 1, 40).astype(int)
    for li, lag in enumerate(lags):
        r2 = []; r4 = []; a2s = []; ab = []
        for f0 in origins:
            nb = neighbours(X, Y, f0, L, rmin)
            cx, cy, z = cage_disp(XU, YU, f0, lag, nb)
            m = z > 0
            s2 = cx[m]**2 + cy[m]**2
            r2.append(s2); r4.append(s2**2)
            ab.append((XU[f0+lag]-XU[f0])[m]**2 + (YU[f0+lag]-YU[f0])[m]**2)
            a2s.append(s2)
        r2 = np.concatenate(r2); r4 = np.concatenate(r4); ab = np.concatenate(ab)
        a2[li] += np.mean(r4) / (2 * np.mean(r2)**2) - 1
        crmsd[li] += np.mean(r2); absmsd[li] += np.mean(ab)
        store.setdefault(lag, []).append(np.sqrt(np.concatenate(a2s)))
a2 /= len(paths); crmsd /= len(paths); absmsd /= len(paths)
lag_tau = lags * dt / TAU

istar = int(np.argmax(a2))
print(f"2. alpha_2 peak at lag = {lag_tau[istar]:.3f} tau  (alpha_2* = {a2[istar]:.3f})"
      f"   -> T_obs")
print(f"3. CR-MSD / absolute MSD at that lag = {crmsd[istar]/absmsd[istar]:.4f}")

d = np.concatenate(store[lags[istar]])
hist, edges = np.histogram(d, bins=120, range=(0, np.percentile(d, 99.5)))
rcv = 0.5*(edges[1:]+edges[:-1])
Gs = hist / np.maximum(rcv, 1e-9)          # 2D van Hove: divide by r
ac, _ = first_min_after_peak(rcv, Gs)
print(f"4. van Hove self (cage-relative) first minimum a_c = {ac:.2f} px "
      f"= {ac/(2*R):.3f} diameters = {ac/R:.3f} R")
print()
print(f"   for comparison: manuscript currently uses a = 0.3*2R = {0.3*2*R:.1f} px;"
      f" supervisor's plan says a = R/2 = {R/2:.1f} px")
print(f"   alpha_2 curve (lag/tau, alpha_2):")
for i in range(0, len(lags), 3):
    print(f"     {lag_tau[i]:7.3f}  {a2[i]:7.4f}")
