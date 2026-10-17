"""|B| map + flux lines from a FEMM .ans file:  python plot_field.py check.ans"""
import sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.tri as mtri


def read_ans(fn):
    s = open(fn).read().split('\n')
    k = s.index('[Solution]'); n = int(s[k + 1])
    nodes = np.array([[float(v) for v in l.split()[:3]] for l in s[k + 2:k + 2 + n]])
    ne = int(s[k + 2 + n])
    el = np.array([[int(v) for v in l.split()[:4]] for l in s[k + 3 + n:k + 3 + n + ne]])
    return nodes, el


fn = sys.argv[1]
nodes, el = read_ans(fn)
x, y, A = nodes[:, 0] / 1e3, nodes[:, 1] / 1e3, nodes[:, 2]
i, j, k = el[:, 0], el[:, 1], el[:, 2]
b = np.stack([y[j] - y[k], y[k] - y[i], y[i] - y[j]], 1); c = np.stack([x[k] - x[j], x[i] - x[k], x[j] - x[i]], 1)
a2 = x[i] * b[:, 0] + x[j] * b[:, 1] + x[k] * b[:, 2]
Ae = np.stack([A[i], A[j], A[k]], 1)
B = np.hypot((c * Ae).sum(1) / a2, (b * Ae).sum(1) / a2)
tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], el[:, :3])
fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
for a, z in zip(ax, (None, ((-30, 30), (15, 75)))):
    pc = a.tripcolor(tri, facecolors=B, cmap='viridis', vmin=0, vmax=2.0)
    a.tricontour(tri, A, levels=40, colors='w', linewidths=.4)
    a.set_aspect('equal')
    if z: a.set_xlim(*z[0]); a.set_ylim(*z[1])
fig.colorbar(pc, ax=ax, shrink=.8, label='|B| (T)')
fig.savefig(fn.replace('.ans', '_B.png'), dpi=110)
