"""Digitise the Motor-CAD graph photos (perspective-corrected with 4 axis points each).

Each graph: 4 pixel -> data calibration points read off the photo (full-resolution 4000x3000 pixels),
curve pixels = pixels clearly darker / more coloured than the local background inside the plot frame.
Writes mcad_digitised.json and check_*.png overlays.   Run: python digitize.py
"""
import json, os
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
UP = r'C:\Users\ADMIN\.claude\uploads\1c6b414f-50ee-42f3-b4ac-fb5e82ea91bc'


def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    h = np.linalg.svd(np.array(A))[2][-1]
    return h.reshape(3, 3) / h[-1]


def apply(Hm, x, y):
    p = Hm @ np.vstack([x, y, np.ones_like(x)])
    return p[0] / p[2], p[1] / p[2]


def load(name):
    return np.asarray(Image.open(os.path.join(UP, name)).convert('RGB')).astype(float)


def curve_mask(im, box, dark=25, sat=None):
    """Pixels darker than the local background by `dark` (grey levels) inside box (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    sub = im[y0:y1, x0:x1]
    g = sub.mean(2)
    bg = ndimage.median_filter(g, size=31)
    m = (bg - g) > dark
    if sat is not None:
        m &= sat(sub)
    m = ndimage.binary_opening(m, np.ones((2, 2)))
    # keep long thin strokes (curves), drop moire speckle and text: component bbox >= 60 px and sparse fill
    lab, n = ndimage.label(ndimage.binary_dilation(m, np.ones((3, 3))))
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        fill = (lab[sl] == i).sum() / (h * w)
        keep[i] = max(h, w) >= 60 and fill < 0.35
    m &= keep[lab]
    ys, xs = np.nonzero(m)
    return xs + x0, ys + y0, sub[ys, xs]


def clusters(theta, val, g, half, gap=0.3, minpts=3):
    v = np.sort(val[np.abs(theta - g) < half])
    cl = []
    for x in v:
        if cl and x - cl[-1][-1] < gap: cl[-1].append(x)
        else: cl.append([x])
    return [float(np.mean(c)) for c in cl if len(c) >= minpts]


def track(theta, val, grid, init, tol=0.8):
    """Multi-track following: per bin, assign cluster centres to tracks by linear prediction (best permutation)."""
    import itertools
    tracks = [[v] for v in init]
    half = (grid[1] - grid[0]) / 2
    for k in range(1, len(grid)):
        cen = clusters(theta, val, grid[k], half)
        preds = [t[-1] + (np.clip(t[-1] - t[-2], -0.6, 0.6) if len(t) > 1 else 0) for t in tracks]
        best, bc = None, 1e9
        opts = cen + [None] * len(tracks)
        for perm in itertools.permutations(range(len(opts)), len(tracks)):
            cost = 0
            for p, j in zip(preds, perm):
                c = opts[j]
                cost += tol ** 2 if c is None else min((c - p) ** 2, 1e6 if abs(c - p) > tol else (c - p) ** 2)
            # two tracks may share one centre only at a crossing; permutations already forbid it, allow via None
            if cost < bc: bc, best = cost, perm
        for t, p, j in zip(tracks, preds, best):
            c = opts[j]
            t.append(c if c is not None and abs(c - p) <= tol else (cen and min(cen, key=lambda q: abs(q - p)) if cen and min(abs(q - p) for q in cen) < 0.25 else p))
    return [np.array(t, float) for t in tracks]


out = {}

# ---------------- Back EMF (17e55a0f): x 0..340 E-deg, y -12..12 V ----------------
im = load('17e55a0f-image.jpg')
Hm = homography([(422, 2448), (3928, 2502), (412, 659), (3918, 751)], [(0, -12), (340, -12), (0, 12), (340, 12)])
xs, ys, _ = curve_mask(im, (365, 680, 4000, 2440), dark=28)
th, v = apply(Hm, xs.astype(float), ys.astype(float))
ok = (v > -11.95) & (v < 11.9)
th, v = th[ok], v[ok]
# follow the 3 curves column by column (nearest cluster to a slope prediction; crossings share a cluster)
grid = np.arange(-3, 347, 0.5)
tr = [[c] for c in sorted(clusters(th, v, grid[0], 0.25, minpts=2), reverse=True)[:3]]
miss = [[] for _ in tr]
for gk in grid[1:]:
    cen = clusters(th, v, gk, 0.25, minpts=2)
    for i, t in enumerate(tr):
        sl = (t[-1] - t[-5]) / 4 if len(t) >= 5 else 0
        p = t[-1] + sl
        c = min(cen, key=lambda q: abs(q - p)) if cen else None
        hit = c is not None and abs(c - p) < 0.5
        t.append(c if hit else p); miss[i].append(not hit)
tr = [np.array(t) for t in tr]
# a track is valid until its first run of > 8 predicted (no-pixel) steps
valid = []
for m in miss:
    m = np.array([False] + m); run = 0; end = len(m)
    for k, x in enumerate(m):
        run = run + 1 if x else 0
        if run > 8: end = k - run; break
    valid.append(np.arange(len(m)) < end)
# phase 1 = track starting near +11 V; others are phase 1 shifted by 120 or 240 deg; half-wave antisymmetry
# E(theta + 180) = -E(theta) fills what the photo does not show
g360 = np.arange(0, 360, 1.0)
def fold(lags):
    T = np.concatenate([(grid[vm] - s) % 360 for s, vm in zip(lags, valid)]); V = np.concatenate([t[vm] for t, vm in zip(tr, valid)])
    return np.concatenate([T, (T + 180) % 360]), np.concatenate([V, -V])
def spread(lags):
    T, V = fold(lags); b = np.round(T).astype(int) % 360
    return sum(np.var(V[b == i]) for i in range(360) if np.sum(b == i) > 1)
import itertools
lags = min(([0, a, b] for a, b in itertools.product((120, 240), repeat=2)), key=spread)
pts_t, pts_v = fold(lags)
ph1 = np.array([np.median(pts_v[np.abs(((pts_t - g + 180) % 360) - 180) <= 0.5]) for g in g360])
out['backemf'] = dict(theta_e=g360.tolist(), phase1=ph1.tolist(), lags=lags,
                      phases_raw=dict(theta_e=grid.tolist(), curves=[t.tolist() for t in tr]))
print('backEMF: rms %.3f  peak %.3f  fund peak %.3f  lags %s' % (
    np.sqrt(np.mean(ph1 ** 2)), np.max(np.abs(ph1)), 2 * abs(np.mean(ph1 * np.exp(-1j * np.radians(g360)))), lags))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(th, v, ',', color='0.6'); [ax.plot(grid, t, lw=1) for t in tr]; ax.plot(g360, ph1, 'k--', lw=1.5)
fig.savefig(os.path.join(HERE, 'check_backemf.png'), dpi=90); plt.close(fig)
def colour_curve(name, pts, data, box, sel, xgrid, dark=6):
    im = load(name)
    Hm = homography(pts, data)
    x0, y0, x1, y1 = box
    sub = im[y0:y1, x0:x1]
    g = sub.mean(2); bg = ndimage.median_filter(g, size=31)
    m = sel(sub) & ((bg - g) > dark)
    m = ndimage.binary_opening(m, np.ones((2, 2)))
    lab, n = ndimage.label(ndimage.binary_dilation(m, np.ones((5, 5))))
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m &= np.isin(lab, 1 + np.nonzero(sizes >= 40)[0])
    ys, xs = np.nonzero(m)
    X, Y = apply(Hm, xs + float(x0), ys + float(y0))
    h = xgrid[1] - xgrid[0]
    val = np.array([np.median(Y[np.abs(X - q) <= h / 2]) if np.sum(np.abs(X - q) <= h / 2) >= 3 else np.nan for q in xgrid])
    return X, Y, val

# ---------------- Cogging (b623fe78): x 0..6 mech deg, y -0.34..0.34 N.m, red VW curve ----------------
red = lambda s: (s[:, :, 0] - s[:, :, 1] > 25) & (s[:, :, 0] - s[:, :, 2] > 15)
xc = np.arange(0, 6.0001, 0.02)
X, Y, cg = colour_curve('b623fe78-image.jpg', [(255, 455), (315, 2208), (3910, 480), (3865, 2142)],
                        [(0, 0.34), (0, -0.34), (6, 0.34), (6, -0.34)], (330, 470, 3880, 2190), red, xc)
# the photo loses the curve near both ends (glare); bridge the gap using the 6-deg periodicity
flat = np.r_[False, np.abs(np.diff(cg)) < 1e-9]
ok = ~np.isnan(cg) & ~flat & (xc > 0.32) & (xc < 5.08)
cg = np.interp(xc, xc[ok], cg[ok], period=6.0)
out['cogging'] = dict(theta_m=xc.tolist(), vw=cg.tolist())
print('cogging: min %.3f max %.3f pk-pk %.3f  (Motor-CAD 0.663)' % (cg.min(), cg.max(), np.ptp(cg)))
fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(X, Y, ',', color='0.6'); ax.plot(xc, cg, 'k'); ax.grid()
fig.savefig(os.path.join(HERE, 'check_cogging.png'), dpi=90); plt.close(fig)

# ---------------- Air-gap B (1f469e6b): x 0..1800 E-deg, y -0.85..0.85 T; Br green, Bt magenta ----------------
green = lambda s: (s[:, :, 1] - s[:, :, 0] > 8) & (s[:, :, 1] - s[:, :, 2] > 3)
mag = lambda s: (s[:, :, 0] - s[:, :, 1] > 20) & (s[:, :, 2] - s[:, :, 1] > 20)
xa = np.arange(0, 1800.01, 2.0)
AP = [(140, 698), (242, 2388), (3830, 770), (3678, 2392)]; AD = [(0, 0.85), (0, -0.85), (1800, 0.85), (1800, -0.85)]
Xg, Yg, br = colour_curve('1f469e6b-image.jpg', AP, AD, (250, 710, 3800, 2380), green, xa, dark=4)
Xm, Ym, bt = colour_curve('1f469e6b-image.jpg', AP, AD, (250, 710, 3800, 2380), mag, xa, dark=4)
bad = (xa > 520) & (xa < 1180)          # screen moire in the photo: unreadable
br[bad] = np.nan; bt[bad] = np.nan
out['airgap'] = dict(theta_e=xa.tolist(), Br=[None if np.isnan(q) else float(q) for q in br], Bt=[None if np.isnan(q) else float(q) for q in bt])
print('airgap: Br max %.3f min %.3f, coverage %d/%d' % (np.nanmax(br), np.nanmin(br), np.sum(~np.isnan(br)), len(xa)))
fig, ax = plt.subplots(figsize=(14, 4)); ax.plot(Xg, Yg, ',', color='0.6'); ax.plot(xa, br, 'g', lw=.8); ax.plot(xa, bt, 'm', lw=.8); ax.grid()
fig.savefig(os.path.join(HERE, 'check_airgap.png'), dpi=90); plt.close(fig)
json.dump(out, open(os.path.join(HERE, 'mcad_digitised.json'), 'w'), indent=1)




