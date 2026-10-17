"""Render the flux maps and demagnetisation maps of alu_maps.py solutions -> report_full/fig/flux_<d>.jpg, demag_<d>.jpg
python alu_maps_render.py [designs...]"""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
import backemf as M

H = M.HERE
SRC = os.path.join(H, 'work', 'maps')
OUT = os.path.join(H, 'report_full', 'fig')
BR_COLD = M.BR20 * (1 + M.BR_TC * (-20.0 - 20))


def read(fn):
    s = open(fn).read().split('\n')
    l0 = next(i for i, l in enumerate(s) if l.startswith('[NumBlockLabels]'))
    labs = [l.split() for l in s[l0 + 1:l0 + 1 + int(s[l0].split('=')[1])]]
    i0 = next(i for i, l in enumerate(s) if l.startswith('[BlockProps]'))
    names = [l.split('=')[1].strip().strip('"') for l in s[i0:l0] if l.strip().startswith('<BlockName>')]
    k = s.index('[Solution]'); n = int(s[k + 1])
    nodes = np.array([[float(x) for x in l.split()[:3]] for l in s[k + 2:k + 2 + n]])
    ne = int(s[k + 2 + n])
    el = np.array([[int(x) for x in l.split()[:4]] for l in s[k + 3 + n:k + 3 + n + ne]])
    x, y, A = nodes[:, 0] / 1e3, nodes[:, 1] / 1e3, nodes[:, 2]
    i, j, kk = el[:, 0], el[:, 1], el[:, 2]
    b = np.stack([y[j] - y[kk], y[kk] - y[i], y[i] - y[j]], 1); c = np.stack([x[kk] - x[j], x[i] - x[kk], x[j] - x[i]], 1)
    a2 = x[i] * b[:, 0] + x[j] * b[:, 1] + x[kk] * b[:, 2]
    Ae = np.stack([A[i], A[j], A[kk]], 1)
    Bx, By = (c * Ae).sum(1) / a2, -(b * Ae).sum(1) / a2
    lab = [labs[l] for l in el[:, 3]]
    mat = np.array([names[int(q[2]) - 1] for q in lab])
    mdir = np.radians(np.array([float(q[5]) if names[int(q[2]) - 1] == 'Magnet' else 0.0 for q in lab]))
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], el[:, :3])
    cen = nodes[el[:, :3], :2].mean(1)
    return dict(tri=tri, A=A, B=np.hypot(Bx, By), Bp=Bx * np.cos(mdir) + By * np.sin(mdir), mat=mat, cen=cen)


def flux_fig(d):
    fig, axs = plt.subplots(1, 4, figsize=(15.5, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 0.9, 0.04]))
    for ax, case, title in zip(axs[:2], ('noload_0', 'peak_0'), ('No load, magnets 100 °C', 'Peak point 360 A rms, γ 20°, magnets 100 °C')):
        r = read(os.path.join(SRC, '%s_%s.ans' % (d, case)))
        pc = ax.tripcolor(r['tri'], facecolors=r['B'], cmap='turbo', vmin=0, vmax=2.1, rasterized=True)
        ax.tricontour(r['tri'], r['A'], levels=36, colors='k', linewidths=0.35)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=10, loc='left')
        if case == 'peak_0':
            z = axs[2]
            z.tripcolor(r['tri'], facecolors=r['B'], cmap='turbo', vmin=0, vmax=2.1, rasterized=True)
            z.tricontour(r['tri'], r['A'], levels=60, colors='k', linewidths=0.35)
            z.set_xlim(-26, 26); z.set_ylim(28, 84); z.set_aspect('equal'); z.axis('off'); z.set_title('Peak point, zoom on one pole and tooth', fontsize=10, loc='left')
    fig.colorbar(pc, cax=axs[3], label='|B| (T)')
    fig.savefig(os.path.join(OUT, 'flux_%s.jpg' % d), dpi=85, bbox_inches='tight', pil_kwargs=dict(quality=82)); plt.close(fig)


def demag_fig(d, stats):
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 0.04]))
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.1, vmax=0.5)
    for ax, th in zip(axs[:2], ('0', '3')):
        r = read(os.path.join(SRC, '%s_demag_d_%s.ans' % (d, th)))
        mag = r['mat'] == 'Magnet'
        steel = np.isin(r['mat'], ['M235-35A', 'SCM420'])
        ax.tripcolor(mtri.Triangulation(r['tri'].x, r['tri'].y, r['tri'].triangles[steel]), facecolors=np.ones(steel.sum()), cmap='Greys', vmin=0, vmax=5, rasterized=True)
        pc = ax.tripcolor(mtri.Triangulation(r['tri'].x, r['tri'].y, r['tri'].triangles[mag]), facecolors=r['Bp'][mag], cmap='RdBu', norm=norm, rasterized=True)
        ax.tricontour(r['tri'], r['A'], levels=50, colors='k', linewidths=0.3, alpha=0.6)
        ang = np.degrees(np.arctan2(r['cen'][mag, 1], r['cen'][mag, 0]))
        st = stats.get('demag_d', {}).get(th, {})
        ax.set_xlim(-24, 24); ax.set_ylim(14, 56); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title('Rotor at %s°\n1 %% of magnet area below %.3f T · %.2f %% below 0.1 T' % (th, st.get('B_p1', np.nan), 100 * st.get('share_below_0p1', np.nan)), fontsize=9, loc='left')
    fig.colorbar(pc, cax=axs[2], label='B along magnetisation (T), red = toward demagnetisation')
    fig.suptitle('%s: magnets −20 °C, 360 A rms on the −d axis' % d, x=0.05, ha='left', fontsize=10.5)
    fig.savefig(os.path.join(OUT, 'demag_%s.jpg' % d), dpi=85, bbox_inches='tight', pil_kwargs=dict(quality=82)); plt.close(fig)


if __name__ == '__main__':
    stats = {}
    for fn in glob.glob(os.path.join(SRC, 'stats_*.json')):
        stats.update(json.load(open(fn)))
    json.dump(stats, open(os.path.join(H, 'report_full', 'maps_stats.json'), 'w'), indent=1)
    ds = sys.argv[1:] or sorted(stats)
    for d in ds:
        if not os.path.exists(os.path.join(SRC, '%s_demag_d_3.ans' % d)):
            print(d, 'not solved yet'); continue
        flux_fig(d); demag_fig(d, stats.get(d, {}))
        print(d, 'rendered', flush=True)
