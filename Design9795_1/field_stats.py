"""No-load field results from check.ans (rotor 0 deg, magnetic shaft): flux map PNGs + field_stats.json."""
import json, math, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.tri as mtri
import backemf as M

HERE = M.HERE
fn = os.path.join(HERE, 'check.ans')
s = open(fn).read().split('\n')
l0 = next(i for i, l in enumerate(s) if l.startswith('[NumBlockLabels]'))
labs = [l.split() for l in s[l0 + 1:l0 + 1 + int(s[l0].split('=')[1])]]
k = s.index('[Solution]'); n = int(s[k + 1])
nodes = np.array([[float(v) for v in l.split()[:3]] for l in s[k + 2:k + 2 + n]])
ne = int(s[k + 2 + n])
el = np.array([[int(v) for v in l.split()[:4]] for l in s[k + 3 + n:k + 3 + n + ne]])

x, y, A = nodes[:, 0] / 1e3, nodes[:, 1] / 1e3, nodes[:, 2]
i, j, kk = el[:, 0], el[:, 1], el[:, 2]
b = np.stack([y[j] - y[kk], y[kk] - y[i], y[i] - y[j]], 1); c = np.stack([x[kk] - x[j], x[i] - x[kk], x[j] - x[i]], 1)
a2 = x[i] * b[:, 0] + x[j] * b[:, 1] + x[kk] * b[:, 2]
Ae = np.stack([A[i], A[j], A[kk]], 1)
Bx = (c * Ae).sum(1) / a2; By = -(b * Ae).sum(1) / a2
Bmag = np.hypot(Bx, By)
area = np.abs(a2) / 2 * 1e6                                   # mm2
mat = np.array([int(labs[l][2]) for l in el[:, 3]])          # 1 Air 2 Cu 3 M235 4 SCM420 5 Magnet
mdir = np.radians(np.array([float(labs[l][5]) for l in el[:, 3]]))
Bm = Bx * np.cos(mdir) + By * np.sin(mdir)
Hm = (Bm - M.BR) / (M.MU0 * M.MU_MAG)
cx, cy = nodes[el[:, :3], 0].mean(1), nodes[el[:, :3], 1].mean(1); r = np.hypot(cx, cy)
th = np.degrees(np.arctan2(cy, cx)) % 360

sg = M.slot_geometry(M.STATOR)
wavg = lambda m: float(np.sum(Bmag[m] * area[m]) / np.sum(area[m]))
st = {}
# tooth body: stator steel between tip and slot bottom, within half tooth width of a tooth axis (axes at 105 + 30k deg)
d_ax = np.array([min(abs(((t - 105) % 30) - 0), abs(((t - 105) % 30) - 30)) for t in th])
tooth = (mat == 3) & (r > sg['C'][1] + 1) & (r < sg['rb'] - 1) & (r * np.sin(np.radians(d_ax)) < M.STATOR['tooth_w'] / 2)
yoke = (mat == 3) & (r > sg['rb'] + 0.5)
st['tooth_B_mean'] = wavg(tooth); st['tooth_B_max'] = float(np.percentile(Bmag[tooth], 99))
st['yoke_B_mean'] = wavg(yoke); st['yoke_B_max'] = float(np.percentile(Bmag[yoke], 99))
rot_steel = (mat == 3) & (r < M.R_ROT)
st['bridge_B_max'] = float(np.percentile(Bmag[rot_steel & (r < 24)], 99))
mag = mat == 5
st['magnet_B_along_M_mean'] = float(np.sum(Bm[mag] * area[mag]) / np.sum(area[mag]))
st['magnet_B_along_M_min'] = float(np.min(Bm[mag]))
st['magnet_H_min_kA_m'] = float(np.min(Hm[mag]) / 1e3)
st['magnet_area_mm2_per_pole'] = float(np.sum(area[mag]) / 10)
# air-gap radial B at mid gap, binned 0.5 deg
band = (mat == 1) & (r > M.R_ROT + 0.02) & (r < M.R_BORE - 0.02)
brad = (Bx[band] * cx[band] + By[band] * cy[band]) / r[band]
kb = np.minimum((th[band] / 0.5).astype(int), 719)
gap = np.bincount(kb, brad, 720) / np.maximum(np.bincount(kb, minlength=720), 1)
h = np.abs(np.fft.rfft(gap)) / 720 * 2
st['gap_Br'] = gap.tolist(); st['gap_Br_fund_peak'] = float(h[5]); st['gap_Br_peak'] = float(np.max(np.abs(gap)))
json.dump(st, open(os.path.join(HERE, 'field_stats.json'), 'w'), indent=1)
print({k: (round(v, 3) if isinstance(v, float) else '...') for k, v in st.items()})

tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], el[:, :3])
for name, z, fs in (('fluxmap_full.png', None, 7), ('fluxmap_zoom.png', ((-32, 32), (12, 76)), 7)):
    fig, ax = plt.subplots(figsize=(fs, fs * (0.92 if z is None else 1.0)))
    pc = ax.tripcolor(tri, facecolors=Bmag, cmap='turbo', vmin=0, vmax=2.1, rasterized=True)
    ax.tricontour(tri, nodes[:, 2], levels=36, colors='k', linewidths=.35)
    ax.set_aspect('equal'); ax.axis('off')
    if z: ax.set_xlim(*z[0]); ax.set_ylim(*z[1])
    cb = fig.colorbar(pc, ax=ax, shrink=.75, pad=.02); cb.set_label('|B| (T)')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, name), dpi=110, transparent=False, facecolor='white'); plt.close(fig)
