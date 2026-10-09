"""Flux-density maps (|B| + flux lines) and the magnet demag field, drawn straight from FEMM solution files.

Solves the calibrated model at rotor 0 deg for no load and rated current (MTPA angle), keeps the .ans files
(field_noload.ans / field_rated.ans, open them in FEMM) and writes fig_B_noload.png, fig_B_rated.png, fig_demag.png.
Run: python field_maps.py   (~10 min)
"""
import json, math, os, re, shutil
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import femm
import femm_motor as M

HERE = M.HERE


def read_ans(fn):
    s = open(fn).read().split('\n')
    lab0 = next(i for i, l in enumerate(s) if l.startswith('[NumBlockLabels]'))
    labs = [l.split() for l in s[lab0 + 1:lab0 + 1 + int(s[lab0].split('=')[1])]]
    k = s.index('[Solution]'); n = int(s[k + 1])
    nodes = np.array([[float(v) for v in l.split()[:3]] for l in s[k + 2:k + 2 + n]])
    ne = int(s[k + 2 + n])
    el = np.array([[int(v) for v in l.split()[:4]] for l in s[k + 3 + n:k + 3 + n + ne]])
    return nodes, el, labs


def element_fields(nodes, el, labs, full=False):
    """B (T) per element from the nodal vector potential (linear triangles), plus magnet H along magnetisation."""
    x, y, A = nodes[:, 0] / 1e3, nodes[:, 1] / 1e3, nodes[:, 2]
    i, j, k = el[:, 0], el[:, 1], el[:, 2]
    b = np.stack([y[j] - y[k], y[k] - y[i], y[i] - y[j]], 1); c = np.stack([x[k] - x[j], x[i] - x[k], x[j] - x[i]], 1)
    area2 = x[i] * b[:, 0] + x[j] * b[:, 1] + x[k] * b[:, 2]
    Ae = np.stack([A[i], A[j], A[k]], 1)
    Bx = (c * Ae).sum(1) / area2; By = -(b * Ae).sum(1) / area2
    mat = np.array([int(labs[l][2]) if l >= 0 else 0 for l in el[:, 3]])          # 4 = Ferrite (build order)
    mdir = np.radians(np.array([float(labs[l][5]) if l >= 0 else 0 for l in el[:, 3]]))
    u = np.stack([np.cos(mdir), np.sin(mdir)], 1)
    Bm = Bx * u[:, 0] + By * u[:, 1]
    Hm = np.where(mat == 4, (Bm - M.BR) / (M.MU0 * M.MU_MAG), np.nan)               # linear recoil line
    return (np.hypot(Bx, By), Hm, mat, Bx, By, Bm) if full else (np.hypot(Bx, By), Hm, mat)


def draw(fn, title, out, zoom=None, demag=False):
    nodes, el, labs = read_ans(fn)
    Bmag, Hm, mat = element_fields(nodes, el, labs)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], el[:, :3])
    fig, ax = plt.subplots(figsize=(8, 8))
    if demag:
        mask = mat != 4
        tri.set_mask(mask)
        pc = ax.tripcolor(tri, facecolors=np.nan_to_num(Hm / 1e3), cmap='RdBu', vmin=-600, vmax=600)
        tri.set_mask(None)
        ax.tripcolor(tri, facecolors=np.where(mask, Bmag, np.nan), cmap='Greys', vmin=0, vmax=3, alpha=.35)
        ax.tricontour(tri, nodes[:, 2], levels=40, colors='k', linewidths=.35)
        cb = fig.colorbar(pc, ax=ax, shrink=.8); cb.set_label('H along magnetisation (kA/m); HcJ = -%.0f' % (M.HCJ / 1e3))
    else:
        pc = ax.tripcolor(tri, facecolors=Bmag, cmap='viridis', vmin=0, vmax=2.0)
        ax.tricontour(tri, nodes[:, 2], levels=40, colors='w', linewidths=.4)
        cb = fig.colorbar(pc, ax=ax, shrink=.8); cb.set_label('|B| (T)')
    ax.set_aspect('equal'); ax.set_title(title, fontsize=9); ax.set_xlabel('mm'); ax.set_ylabel('mm')
    if zoom: ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return Bmag, Hm, mat


def stator_stats(fn):
    """|B| statistics over stator steel only (rotor uses the same steel, so filter by radius)."""
    nodes, el, labs = read_ans(fn)
    Bm, _, mat = element_fields(nodes, el, labs)
    c = nodes[el[:, :3], :2].mean(1)
    st = (mat == 3) & (np.hypot(c[:, 0], c[:, 1]) > M.R_BORE)
    return dict(B_stator_p99=float(np.percentile(Bm[st], 99)), B_stator_mean=float(Bm[st].mean()))


def main():
    r = json.load(open(os.path.join(HERE, 'results.json')))
    phi, g = r['noload']['phi'], r['gamma_sweep']['gamma_opt']
    M.worker_init()                                   # builds the calibrated model in work/
    stats = {}
    for case, cur in (('noload', M.ZERO), ('rated', M.phase_currents(0, phi, g, M.I_LINE_RMS))):
        M.solve(0, cur)
        ans = os.path.join(HERE, 'work', 'w%d.ans' % os.getpid())
        shutil.copy(ans, os.path.join(HERE, 'field_%s.ans' % case))
        shutil.copy(ans[:-4] + '.fem', os.path.join(HERE, 'field_%s.fem' % case))
    femm.closefemm()
    lbl = 'calibrated: magnets %.0f C, Br %.3f T' % (M.CFG['magnet_temp_C'], M.BR)
    for case, t in (('noload', 'No load'), ('rated', 'Rated %.0f A rms, gamma %.1f deg' % (M.I_LINE_RMS, g))):
        Bm, Hm, mat = draw(os.path.join(HERE, 'field_%s.ans' % case), '|B| and flux lines - %s (%s)' % (t, lbl),
                           os.path.join(HERE, 'fig_B_%s.png' % case))
        draw(os.path.join(HERE, 'field_%s.ans' % case), '|B| one pole pair - %s' % t,
             os.path.join(HERE, 'fig_B_%s_zoom.png' % case), zoom=((-30, 30), (25, 82)))
        stats[case] = stator_stats(os.path.join(HERE, 'field_%s.ans' % case))
    nodes, el, labs = read_ans(os.path.join(HERE, 'field_rated.ans'))
    Bm, Hm, mat = element_fields(nodes, el, labs)
    x, y = nodes[el[:, :3], 0].mean(1), nodes[el[:, :3], 1].mean(1)
    area = 0.5 * abs((nodes[el[:, 1], 0] - nodes[el[:, 0], 0]) * (nodes[el[:, 2], 1] - nodes[el[:, 0], 1])
                     - (nodes[el[:, 2], 0] - nodes[el[:, 0], 0]) * (nodes[el[:, 1], 1] - nodes[el[:, 0], 1]))
    m = mat == 4
    stats['demag_rated'] = dict(H_min_kA_m=float(np.nanmin(Hm[m]) / 1e3),
                                area_beyond_HcJ_pct=float(100 * area[m & (Hm < -M.HCJ)].sum() / area[m].sum()),
                                area_beyond_80pct_HcJ_pct=float(100 * area[m & (Hm < -0.8 * M.HCJ)].sum() / area[m].sum()))
    draw(os.path.join(HERE, 'field_rated.ans'), 'Magnet field along magnetisation at rated current (red = demagnetising)',
         os.path.join(HERE, 'fig_demag.png'), zoom=((-22, 22), (20, 56)), demag=True)
    json.dump(stats, open(os.path.join(HERE, 'field_stats.json'), 'w'), indent=1)
    print(stats)


if __name__ == '__main__':
    main()
