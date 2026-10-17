"""Multi-page aluminium-winding report for Design9795_1: report_alu/index.html (one file, figures embedded).

Pages: summary, method + copper baseline, one page per stator OD (160, 163, 165, 167), comparison across ODs, drive cycle and range,
cost, recommended design changes, FEMM stator variants, assumptions and files.
Inputs: alu9795_search.json, alu9795_picks.json, alu9795_femm.json, alu9795_verify.json, alu9795_ops.json (OP1-OP3, recomputed
here for every key design), lab_compare_9795.json (Motor-CAD envelope).

python build_alu9795_full.py
"""
import base64, io, json, math, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
import alu9795_study as S
import alu9795_search as SE
import alu9795_analyse as A
import alu9795_ops as O
import backemf as M

H = S.H
OUT = os.path.join(H, 'report_alu')
R = json.load(open(os.path.join(H, 'alu9795_search.json')))
PK = json.load(open(os.path.join(H, 'alu9795_picks.json')))
FM = json.load(open(os.path.join(H, 'alu9795_femm.json')))
VF = json.load(open(os.path.join(H, 'alu9795_verify.json')))
MC = json.load(open(os.path.join(H, 'lab_compare_9795.json')))['mcad_envelope']
CU, DS = R['copper'], R['designs']
BY = {d['label']: d for d in DS}
CUL = 'Copper (Design9795_1)'
BY[CUL] = CU
TEL, WM = 'Telematics', 'WMTC Class 1'
ODS = (160, 163, 165, 167)
REC, ALT, MINR, DROP = 'Al F2 L66 N10 4x1.70', 'Al F2 L72 N9 5x1.60', 'Al F1 L78 N8 6x1.50', 'Al drop-in (SWG 11 size, 64 mm)'
SVS = {160: ['D0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6'], 163: ['G1', 'G2', 'G3'], 165: ['B1', 'B2', 'B3'], 167: ['F1', 'F2', 'F3', 'F4']}
CCU = '#b45309'
PAL = ['#2563eb', '#059669', '#7c3aed', '#0891b2', '#dc2626']
OCOL = {160: '#64748b', 163: '#0891b2', 165: '#7c3aed', 167: '#2563eb'}
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 110, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 2, 'figure.constrained_layout.use': True})


# ---------------- key designs per OD ----------------
def keys_for(od):
    pf, pr = PK['picks']['full|%d' % od], PK['picks']['relaxed|%d' % od]
    if od == 167:
        ks = [('Least range loss', MINR), ('Cooler alternative', ALT), ('Recommended', REC)]
    else:
        ks = [('Least range loss', pf['best_range']['label']), ('Largest saving, full torque', pf['cheapest']['label'])]
        rl = (pr.get('cheapest_within_1.5') or pr.get('cheapest_within_2.0') or pr['cheapest'])['label']
        if rl not in [k for _, k in ks]:
            ks.append(('Largest saving, ≤ 3 % high-speed torque', rl))
    return ks


KEYS = {od: keys_for(od) for od in ODS}
ALLKEYS = [CUL] + sorted({l for od in ODS for _, l in KEYS[od]} | {DROP}, key=lambda l: (BY[l]['geo']['od'], l))


# ---------------- helpers ----------------
def png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', bbox_inches='tight', facecolor='white'); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def fig_html(src, cap):
    return '<figure><img src="%s" alt="%s" loading="lazy"><figcaption>%s</figcaption></figure>' % (src, cap[:90].replace('"', ''), cap)


def f(x, n=2, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '–'
    return (('%+.' + str(n) + 'f') % x if sign else ('%.' + str(n) + 'f') % x).replace('-', '−')


def table(head, rows, cls='num', note=None, hl=()):
    h = ''.join('<th>%s</th>' % c for c in head)
    b = ''.join('<tr%s>%s</tr>' % (' class="hl"' if k in hl else '', ''.join('<td>%s</td>' % c for c in r)) for k, r in enumerate(rows))
    n = '<p class="tnote">%s</p>' % note if note else ''
    return '<div class="tw"><table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>%s' % (cls, h, b, n)


def design(lab):
    if lab == CUL:
        return S.copper()
    d = BY[lab]
    return S.Design(d['metal'], d['sv'], d['Nc'], d['ns'], d['d'], label=lab, Lmm=d['L'])


def short(lab):
    return lab.replace('Al ', '').replace(' (SWG 11 size, 64 mm)', ' (SWG 11, 64 mm)')


def saving(d, p=None):
    return 0.0 if d is CU else -A.dcost(d, p or A.PRICE['Al'])


def rng(d, c=TEL):
    return 0.0 if d is CU else d['range_pct'][c]


def pwdg(d):
    return d['peak']['pcu'] + d['peak']['pac']


# ---------------- computed data: maps, operating points, cycle per step ----------------
EFF, OPR, STEP = {}, {}, {}


def per_step(d):
    """Loss per cycle step (W) on both cycles, same method as alu9795_search.cycle."""
    env, mp = d.run(SE.SP_CY, SE.TQ_CY, npts=241, keys=('loss',))
    tmax = np.array([e['Tsh'] if e else 0.0 for e in env])
    Z = np.array(mp['loss'], float)
    for j in range(Z.shape[1]):
        c = Z[:, j]; ok = ~np.isnan(c)
        if ok.any():
            c[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), c[ok])
    Z[:, 0] = Z[:, 1]
    F = RegularGridInterpolator((SE.TQ_CY, SE.SP_CY), Z, bounds_error=False, fill_value=None)
    out = {}
    for name, (T, n, km) in SE.CYC.items():
        tl = np.interp(n, SE.SP_CY, tmax)
        Tc = np.clip(T, -tl, tl)
        q = np.c_[np.clip(np.abs(Tc), SE.TQ_CY[0], SE.TQ_CY[-1]), np.clip(n, SE.SP_CY[1], SE.SP_CY[-1])]
        out[name] = (np.where(n >= 50, F(q), 0.0), Tc, n, km)
    return out


for lab in ALLKEYS:
    d = design(lab)
    _, mp = d.run(SE.SP_MAP, SE.TQ_MAP, npts=241, keys=('eff',))
    EFF[lab] = mp['eff']
    OPR[lab] = {k: O.op(d, o) for k, o in O.OPS.items()}
    STEP[lab] = per_step(d)
    print('computed', lab, flush=True)
XM, YM = np.meshgrid(SE.SP_MAP, SE.TQ_MAP)


# ---------------- figure builders ----------------
def f_slots(svs):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for k, sv in enumerate(['D0'] + [s for s in svs if s != 'D0']):
        s = dict(M.STATOR, **FM['designs'][sv]['geo']['stator'])
        g = M.slot_geometry(s)
        a0, a1 = math.atan2(g['E'][1], g['E'][0]), math.atan2(g['E'][1], -g['E'][0])
        arc = [(g['rb'] * math.cos(t), g['rb'] * math.sin(t)) for t in np.linspace(a0, a1, 40)]
        p = np.array([g['A'], g['B'], g['C'], g['E']] + arc[1:-1] + [(-x, y) for x, y in (g['E'], g['C'], g['B'], g['A'])])
        col = CCU if sv == 'D0' else plt.cm.tab10((k - 1) % 10)
        ax.plot(p[:, 0], p[:, 1], color=col, lw=2.2 if sv == 'D0' else 1.4, label=sv + (' (copper)' if sv == 'D0' else ''))
        t = np.linspace(math.radians(62), math.radians(118), 60)
        ax.plot(s['od'] / 2 * np.cos(t), s['od'] / 2 * np.sin(t), color=col, lw=1, ls='--')
    t = np.linspace(math.radians(62), math.radians(118), 60)
    ax.plot(M.R_BORE * np.cos(t), M.R_BORE * np.sin(t), color='k', lw=1)
    ax.set_aspect('equal'); ax.set_ylim(50, 86); ax.set_xlim(-26, 26); ax.legend(fontsize=8, loc='lower center', ncol=4); ax.set_xlabel('mm'); ax.grid(alpha=.15)
    ax.set_title('Slot outlines; dashed = stator OD, black = bore 107 mm', loc='left', fontsize=10)
    return png(fig)


def f_scatter(od, keys):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ds = [d for d in DS if round(d['geo']['od']) == od]
    for c, col, lab in (('none', '#cbd5e1', 'meets neither'), ('relaxed', '#93c5fd', 'envelope ≥ 97 %'), ('full', OCOL[od], 'envelope ≥ 99 % (full)')):
        sel = [d for d in ds if (c == 'none' and not A.ok(d, 'relaxed')) or (c == 'relaxed' and A.ok(d, 'relaxed') and not A.ok(d, 'full')) or (c == 'full' and A.ok(d, 'full'))]
        ax.scatter([rng(d) for d in sel], [saving(d) for d in sel], s=10, color=col, alpha=.7, lw=0, label='%s (%d)' % (lab, len(sel)))
    fr = PK['fronts']['full|%d|370' % od]
    if fr:
        ax.plot([d['range'] for d in fr], [-d['dcost'] for d in fr], '-', color=OCOL[od], lw=1.4)
    for k, (role, lab) in enumerate(keys):
        d = BY[lab]
        ax.scatter(rng(d), saving(d), s=140, facecolor='none', edgecolor=PAL[k], lw=2.2, zorder=5)
        ax.annotate(role, (rng(d), saving(d)), xytext=(8, 6), textcoords='offset points', fontsize=8.5, color=PAL[k])
    ax.set_xlabel('Range change, telematics cycle (%)'); ax.set_ylabel('Cost saving per motor (INR)'); ax.legend(fontsize=8, loc='lower left')
    ax.set_xlim(max(-6, min(rng(d) for d in ds) - .2), 0.2)
    return png(fig)


def f_stack(od):
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for Nc, ls in ((8, ':'), (9, '--'), (10, '-')):
        rows = []
        for Lm in SE.LS:
            ds = [d for d in DS if round(d['geo']['od']) == od and d['Nc'] == Nc and d['L'] == Lm and d['sv'] in SVS[od] and A.ok(d, 'relaxed')]
            if ds:
                rows.append(max(ds, key=rng))
        if not rows:
            continue
        Ls = [d['L'] for d in rows]
        ax[0].plot(Ls, [rng(d) for d in rows], ls, color=OCOL[od], label='%d turns/coil' % Nc)
        ax[1].plot(Ls, [saving(d) for d in rows], ls, color=OCOL[od])
        ax[2].plot(Ls, [100 * d['env_ratio_min'] for d in rows], ls, color=OCOL[od])
        for a, fn in zip(ax, (rng, saving, lambda d: 100 * d['env_ratio_min'])):
            a.scatter([d['L'] for d in rows if A.ok(d, 'full')], [fn(d) for d in rows if A.ok(d, 'full')], s=18, color=OCOL[od], zorder=4)
    ax[0].set_ylabel('Range change (%)'); ax[1].set_ylabel('Cost saving (INR)'); ax[2].set_ylabel('Worst envelope ratio vs copper (%)')
    ax[2].axhline(99, color='k', lw=.7); ax[2].axhline(97, color='k', lw=.7, ls=':')
    for a in ax:
        a.set_xlabel('Stack length (mm)')
    ax[0].legend(fontsize=8)
    return png(fig)


def f_env(labs):
    spd = SE.SP_MAP
    ref = np.array([np.nan if v is None else v for v in CU['env_T']])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.8))
    ax[0].plot(spd, ref, color=CCU, label='Copper')
    for k, (role, lab) in enumerate(labs):
        e = np.array([np.nan if v is None else v for v in BY[lab]['env_T']])
        ax[0].plot(spd, e, color=PAL[k], lw=1.5, label='%s: %s' % (role, short(lab)))
        ax[1].plot(spd, 100 * e / ref, color=PAL[k], lw=1.5)
    ax[1].axhline(100, color=CCU, lw=1); ax[1].axhline(99, color='k', lw=.6, ls=':'); ax[1].axhline(97, color='k', lw=.6, ls=':')
    ax[1].set_ylim(94, 104)
    ax[0].set_ylabel('Shaft torque (N·m)'); ax[1].set_ylabel('Torque vs copper (%)')
    for a in ax:
        a.set_xlabel('Speed (rpm)')
    ax[0].legend(fontsize=7.5)
    return png(fig)


def f_dmaps(labs):
    n = len(labs)
    fig, ax = plt.subplots(1, n, figsize=(4.4 * n, 3.9), squeeze=False)
    ax = ax[0]
    for a, (role, lab) in zip(ax, labs):
        dm = EFF[lab] - EFF[CUL]
        cs = a.contourf(XM, YM, dm, levels=np.arange(-4, 2.01, .25), cmap='RdBu', norm=TwoSlopeNorm(0, -4, 2), extend='both')
        c2 = a.contour(XM, YM, dm, levels=[-2, -1, 0, 0.5], colors='k', linewidths=.5); a.clabel(c2, fontsize=7, fmt='%g')
        for k, o in O.OPS.items():
            r = OPR[lab][k]
            a.plot(r['speed'], r['T_shaft'], 'k^', ms=6); a.annotate(k, (r['speed'], r['T_shaft']), xytext=(-18, 5), textcoords='offset points', fontsize=7.5)
        a.set_title('%s\n%s' % (role, short(lab)), loc='left', fontsize=9); a.set_xlabel('Speed (rpm)'); a.grid(False)
    ax[0].set_ylabel('Shaft torque (N·m)')
    fig.colorbar(cs, ax=list(ax), label='efficiency, Al − Cu (points)', shrink=.9)
    return png(fig)


def f_map(lab, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    lv = [80, 85, 88, 90, 91, 92, 93, 94, 95, 95.5]
    ax.contourf(XM, YM, EFF[lab], levels=[60] + lv + [97], cmap='viridis')
    c2 = ax.contour(XM, YM, EFF[lab], levels=lv, colors='w', linewidths=.6); ax.clabel(c2, fontsize=7, fmt='%g')
    for k in O.OPS:
        r = OPR[lab][k]
        ax.plot(r['speed'], r['T_shaft'], 'r^', ms=7); ax.annotate(k, (r['speed'], r['T_shaft']), xytext=(-18, 5), textcoords='offset points', fontsize=8, color='w')
    ax.set_title(title, loc='left', fontsize=10); ax.set_xlabel('Speed (rpm)'); ax.set_ylabel('Shaft torque (N·m)'); ax.grid(False)
    return png(fig)


LK = [('cu_dc', 'DC winding', '#b45309'), ('cu_ac', 'AC winding', '#f59e0b'), ('st_hyst', 'Stator hysteresis', '#475569'),
      ('st_eddy', 'Stator eddy', '#94a3b8'), ('rot', 'Rotor iron', '#7c3aed'), ('mech', 'Bearings + windage', '#0f766e')]


def f_oploss(labs):
    names = [CUL] + [l for _, l in labs]
    tags = ['Copper'] + [r for r, _ in labs]
    fig, ax = plt.subplots(1, 3, figsize=(13, 0.9 + 0.62 * len(names)))
    for a, opk in zip(ax, ('OP1', 'OP2', 'OP3')):
        left = np.zeros(len(names))
        for key, nm, col in LK:
            v = np.array([(OPR[n][opk]['loss']['rot_hyst'] + OPR[n][opk]['loss']['rot_eddy']) if key == 'rot' else OPR[n][opk]['loss'][key] for n in names])
            a.barh(tags, v, left=left, color=col, height=.62, label=nm, edgecolor='white', lw=1.5); left += v
        for k, n in enumerate(names):
            a.text(left[k] + 12, k, '%.0f W · %.2f %%' % (OPR[n][opk]['loss']['total'], OPR[n][opk]['eff']), va='center', fontsize=8)
        a.set_xlim(0, left.max() * 1.45); a.invert_yaxis(); a.grid(axis='y', alpha=0); a.set_xlabel('Loss (W)')
        a.set_title('%s: %s' % (opk, O.OPS[opk]['label']), loc='left', fontsize=10)
        if opk != 'OP1':
            a.set_yticklabels([])
    h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='outside lower center', ncol=6, fontsize=8)
    return png(fig)


def f_cycle_where(labs, cyc=TEL):
    """Extra loss energy against copper over the cycle, by motor speed band, motoring and regen."""
    bands = np.arange(0, 7001, 500)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.6))
    l0, T0, n0, km = STEP[CUL][cyc]
    w = .8 / max(1, len(labs))
    for k, (role, lab) in enumerate(labs):
        l1, T1, n1, _ = STEP[lab][cyc]
        dl = (l1 - l0) / 3600 / km                          # Wh/km per step
        for j, (m, nm) in enumerate(((T0 >= 0, 'motoring'), (T0 < 0, 'regenerating'))):
            v = [dl[m & (n0 >= a) & (n0 < a + 500)].sum() for a in bands[:-1]]
            ax[j].bar(bands[:-1] + 250 + (k - (len(labs) - 1) / 2) * w * 500, v, width=w * 500, color=PAL[k], label='%s (%s)' % (role, short(lab)))
            ax[j].set_title('Extra loss vs copper while %s (Wh/km), %s' % (nm, cyc), loc='left', fontsize=9.5)
    for a in ax:
        a.axhline(0, color='k', lw=.7); a.set_xlabel('Motor speed band (rpm)')
    ax[0].set_ylabel('Wh/km'); ax[0].legend(fontsize=7.5)
    return png(fig)


def f_cycle_losses(labs, cyc=TEL):
    names = [CUL] + [l for _, l in labs]
    tags = ['Copper'] + [r for r, _ in labs]
    fig, ax = plt.subplots(figsize=(9.5, 0.9 + 0.55 * len(names)))
    left = np.zeros(len(names))
    for key, nm, col in (('pcu', 'DC winding', '#b45309'), ('pac', 'AC winding', '#f59e0b'), ('pfe', 'Iron', '#64748b'), ('pme', 'Bearings + windage', '#0f766e')):
        v = np.array([BY[n]['cycle'][cyc]['loss'][key] / BY[n]['cycle'][cyc]['km'] for n in names])
        ax.barh(tags, v, left=left, color=col, height=.6, label=nm, edgecolor='white', lw=1.5); left += v
    for k, n in enumerate(names):
        ax.text(left[k] + .1, k, '%.2f Wh/km' % left[k], va='center', fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0, left.max() * 1.2); ax.grid(axis='y', alpha=0); ax.set_xlabel('Motor loss over the %s cycle (Wh/km)' % cyc)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    return png(fig)


# ---------------- tables ----------------
def spec_table(labs, extra_first=True):
    cols = ([('Copper (today)', CU)] if extra_first else []) + [(r, BY[l]) for r, l in labs]
    g = lambda d: d['geo']
    rows = [
        ('Design', lambda d: 'Cu SWG 11' if d is CU else short(d['label'])),
        ('Stator OD (mm)', lambda d: f(g(d)['od'], 0)), ('Stator variant', lambda d: d.get('sv', 'D0')),
        ('Stack length, stator = rotor = magnets (mm)', lambda d: f(d['L'], 0)),
        ('Slot depth (mm)', lambda d: f(g(d)['slot_depth'])), ('Tooth width (mm)', lambda d: f(g(d)['tooth_w'])),
        ('Yoke (mm)', lambda d: f(g(d)['yoke'])), ('Slot area (mm²)', lambda d: '%s (×%.3f)' % (f(g(d)['slot_area'], 1), g(d)['slot_area'] / CU['geo']['slot_area'])),
        ('Turns per coil', lambda d: '%d' % d['Nc']), ('Strands × bare diameter (mm)', lambda d: '%d × %.2f' % (d['ns'], d['d'])),
        ('Conductor area (mm²)', lambda d: f(d['masses']['conductor_area_mm2'])), ('Insulated fill', lambda d: f(d['fill'], 3)),
        ('Mean turn length (mm)', lambda d: f(d['mlt'] * 1e3 if d['mlt'] < 1 else d['mlt'], 1)),
        ('Resistance per path, 100 °C (mΩ)', lambda d: f(d['R_path'] * 1e3)),
        ('Peak torque, 1000 rpm (N·m)', lambda d: f(d['peak']['T'])),
        ('Worst envelope ratio vs copper', lambda d: '–' if d is CU else f(100 * d['env_ratio_min'], 1) + ' %'),
        ('Winding loss at peak torque (W)', lambda d: f(pwdg(d), 0)), ('Current density at peak (A/mm² rms)', lambda d: f(d['peak']['J'], 1)),
        ('Peak efficiency (%)', lambda d: f(d['peak_eff'])), ('Map efficiency, mean change (points)', lambda d: '–' if d is CU else f(d['map_dEta_mean'], 2, True)),
        ('Telematics: Wh/km at motor terminals', lambda d: f(d['cycle'][TEL]['Wh_km'], 3)),
        ('Range change, telematics', lambda d: '–' if d is CU else f(rng(d), 2, True) + ' %'),
        ('Range change, WMTC Class 1', lambda d: '–' if d is CU else f(rng(d, WM), 2, True) + ' %'),
        ('Winding mass (kg)', lambda d: f(d['masses']['winding'], 3)), ('Stator laminations (kg)', lambda d: f(d['masses']['stator_lam'], 3)),
        ('Rotor laminations + magnets (kg)', lambda d: f(d['masses']['rotor_lam'] + d['masses']['magnets'], 3)),
        ('Active mass (kg)', lambda d: f(d['masses']['active'], 3)), ('Material cost (INR)', lambda d: f(d['cost']['total'], 0)),
        ('Saving vs copper (INR)', lambda d: '–' if d is CU else '%s (%s %%)' % (f(saving(d), 0), f(100 * saving(d) / CU['cost']['total'], 1))),
    ]
    hl = {'Stator OD (mm)', 'Stack length, stator = rotor = magnets (mm)', 'Turns per coil', 'Strands × bare diameter (mm)', 'Range change, telematics', 'Saving vs copper (INR)'}
    body = [[name] + [fn(d) for _, d in cols] for name, fn in rows]
    return table([''] + [n for n, _ in cols], body, hl=[k for k, (name, _) in enumerate(rows) if name in hl])


def front_table(od, c):
    fr = PK['fronts']['%s|%d|370' % (c, od)]
    if not fr:
        return '<p>No design at this OD meets this criterion.</p>'
    rows = []
    for d in sorted(fr, key=lambda d: -d['range']):
        x = BY[d['label']]
        rows.append([short(d['label']), x['sv'], f(x['geo']['yoke']), f(d['L'], 0), '%d × (%d × %.2f)' % (d['Nc'], d['ns'], d['d']), f(100 * d['env'], 1) + ' %',
                     f(d['range'], 2, True) + ' %', f(d['range_wmtc'], 2, True) + ' %', f(-d['dcost'], 0), f(d['pwdg'], 0), f(d['dmass'], 2, True)])
    return table(['Design', 'Stator', 'Yoke (mm)', 'Stack (mm)', 'Turns × (strands × d)', 'Worst envelope', 'Range, telematics', 'Range, WMTC', 'Saving (INR)',
                  'Winding loss at peak (W)', 'Mass change (kg)'], rows)


def op_table(opk, labs, mcad=True):
    cols = ([('Motor-CAD, copper', O.mcad().get(opk))] if mcad and opk in O.mcad() else []) + [('Copper (FEMM)', OPR[CUL][opk])] + [(r, OPR[l][opk]) for r, l in labs]
    rot = lambda r: r['loss']['rot'] if 'rot' in r['loss'] else r['loss']['rot_hyst'] + r['loss']['rot_eddy']
    stl = lambda r: r['loss']['st'] if 'st' in r['loss'] else r['loss']['st_hyst'] + r['loss']['st_eddy']
    spec = [('Shaft torque (N·m)', lambda r: f(r['T_shaft'])), ('Electromagnetic torque (N·m)', lambda r: f(r['T_em'])),
            ('Shaft power (W)', lambda r: f(r['P_shaft'], 0)), ('Input power (W)', lambda r: f(r['P_in'], 0)), ('Efficiency (%)', lambda r: f(r['eff'])),
            ('Line current (A rms)', lambda r: f(r['I_line_rms'], 1)), ('Phase current (A rms, delta)', lambda r: f(r['I_phase_rms'], 1)),
            ('Id, phase (A peak)', lambda r: f(r['Id_phase_pk'], 1)), ('Iq, phase (A peak)', lambda r: f(r['Iq_phase_pk'], 1)),
            ('Phase advance (° el)', lambda r: f(r['phase_advance'], 1)), ('Phase voltage (V peak, limit 49.4)', lambda r: f(r['V_phase_pk'])),
            ('Power factor', lambda r: f(r['pf'], 3)), ('ψd (mV·s)', lambda r: f(r['psid_mVs'])), ('ψq (mV·s)', lambda r: f(r['psiq_mVs'])),
            ('DC bus current (A)', lambda r: f(r['Idc'], 1)), ('Current density (A/mm² rms)', lambda r: f(r['J'], 1)), ('Active limit', lambda r: r['limit']),
            ('DC winding loss (W)', lambda r: f(r['loss']['cu_dc'], 1)), ('AC winding loss (W)', lambda r: f(r['loss']['cu_ac'], 1)),
            ('Stator hysteresis (W)', lambda r: f(r['loss']['st_hyst'], 1)), ('Stator eddy (W)', lambda r: f(r['loss']['st_eddy'], 1)),
            ('Stator iron total (W)', lambda r: f(stl(r), 1)), ('Rotor iron (W)', lambda r: f(rot(r), 1)), ('Iron total (W)', lambda r: f(r['loss']['iron'], 1)),
            ('Bearings + windage (W)', lambda r: f(r['loss']['mech'], 1)), ('Total loss (W)', lambda r: f(r['loss']['total'], 1))]
    body = []
    for name, fn in spec:
        row = [name]
        for _, r in cols:
            try:
                row.append(fn(r))
            except KeyError:
                row.append('–')
        body.append(row)
    hl = [k for k, (n, _) in enumerate(spec) if n in ('Efficiency (%)', 'Phase advance (° el)', 'Total loss (W)', 'DC winding loss (W)', 'AC winding loss (W)')]
    return table(['%s · %s' % (opk, O.OPS[opk]['label'])] + [n for n, _ in cols], body, hl=hl)


def op_delta_table(labs):
    rows = [['%s: %s' % (k, O.OPS[k]['label'])] + [f(OPR[l][k]['eff'] - OPR[CUL][k]['eff'], 2, True) for _, l in labs] for k in O.OPS]
    return table(['Efficiency change vs copper (points)'] + [r for r, _ in labs], rows)


def cycle_table(labs):
    rows = []
    for nm, lab in [('Copper', CUL)] + list(labs):
        d = BY[lab]
        for c in (TEL, WM):
            r = d['cycle'][c]; l = r['loss']; km = r['km']
            rows.append([nm if c == TEL else '', c, f(r['Wh_km'], 3), f(r['E_shaft_mot'] / km, 2), f(r['E_in_mot'] / km, 2), f(r['eff_mot']), f(r['E_shaft_gen'] / km, 2),
                         f(r['E_out_gen'] / km, 2), f(r['eff_gen']), f(l['pcu'] / km), f(l['pac'] / km), f(l['pfe'] / km), f(l['pme'] / km), f(l['loss'] / km),
                         '–' if d is CU else f(rng(d, c), 2, True) + ' %'])
    return table(['Design', 'Cycle', 'Net Wh/km', 'Shaft out, motoring', 'Electrical in, motoring', 'η motoring (%)', 'Shaft in, regen', 'Electrical out, regen',
                  'η regen (%)', 'DC winding loss', 'AC winding loss', 'Iron loss', 'Mech. loss', 'Total loss', 'Range change'], rows,
                 note='Energies and losses in Wh/km at the motor (motor losses only; inverter, battery and vehicle are the same for every design).')


def femm_table(svs):
    rows = []
    for sv in svs:
        v = FM['designs'][sv]; g = v['geo']
        rows.append([sv, f(g['stator']['od'], 0), f(g['stator']['slot_depth']), f(g['stator']['tooth_w']), f(g['yoke_mm']), f(g['slot_area'], 1),
                     '×%.3f' % (g['slot_area'] / FM['designs']['D0']['geo']['slot_area']), f(np.max(v['T'])),
                     f(100 * (np.max(v['T']) / np.max(FM['designs']['D0']['T']) - 1), 1, True) + ' %', f(np.max(v['tooth_B99'])), f(np.max(v['yoke_B99'])),
                     f(g['st_mass'], 3)])
    return table(['Stator', 'OD', 'Slot depth', 'Tooth', 'Yoke', 'Slot area (mm²)', 'vs copper', 'Max torque (N·m)*', 'vs copper', 'Tooth B 99 % (T)',
                  'Yoke B 99 % (T)', 'Stator steel (kg)*'], rows,
                 note='* FEMM 2D, 64 mm stack, 10 turns per coil, peak over the 4 × 4 current grid at 360 A (no skew). Dimensions in mm.')


def cost_table(labs):
    rows = []
    for nm, lab in [('Copper', CUL)] + list(labs):
        d = BY[lab]; c = d['cost']; m = d['masses']
        rows.append([nm, f(m['winding'], 3), f(c['wire'], 0), f(m['stator_lam'] + m['rotor_lam'], 3), f(c['steel'], 0), f(m['magnets'], 3), f(c['magnet'], 0),
                     f(c['total'], 0), '–' if d is CU else f(saving(d), 0)] + ['–' if d is CU else f(saving(d, p), 0) for p in (230.0, 450.0)])
    return table(['Design', 'Winding (kg)', 'Wire (INR)', 'Laminations (kg)', 'Laminations (INR)', 'Magnets (kg)', 'Magnets (INR)', 'Total (INR)',
                  'Saving, Al ₹370', 'Saving, Al ₹230', 'Saving, Al ₹450'], rows)


# ---------------- pages ----------------
P = {}
rec, alt, mn, dr = BY[REC], BY[ALT], BY[MINR], BY[DROP]
pct = lambda d: 100 * saving(d) / CU['cost']['total']
K3 = [('Least range loss', MINR), ('Cooler alternative', ALT), ('Recommended', REC)]

# summary
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
for k, c in enumerate(('full', 'relaxed')):
    a = ax[k]
    for o in ODS:
        ds = [d for d in DS if A.ok(d, c) and round(d['geo']['od']) == o]
        a.scatter([rng(d) for d in ds], [saving(d) for d in ds], s=9, color=OCOL[o], alpha=.25, lw=0)
        fr = PK['fronts']['%s|%d|370' % (c, o)]
        if fr:
            a.plot([d['range'] for d in fr], [-d['dcost'] for d in fr], 'o-', ms=5, color=OCOL[o], label='OD %d' % o)
    for kk, (role, lab) in enumerate(K3):
        d = BY[lab]
        a.scatter(rng(d), saving(d), s=130, facecolor='none', edgecolor=PAL[kk], lw=2.2, zorder=5)
        a.annotate(role, (rng(d), saving(d)), xytext=(8, -12), textcoords='offset points', fontsize=8.5, color=PAL[kk])
    a.scatter(rng(dr), saving(dr), s=60, marker='x', color='#dc2626', zorder=5)
    a.annotate('drop-in (3 % torque short\nat 3000 rpm)', (rng(dr), saving(dr)), xytext=(8, -4), textcoords='offset points', fontsize=8, color='#dc2626')
    a.set_xlabel('Range change, telematics cycle (%)'); a.set_ylabel('Cost saving per motor (INR)'); a.legend(fontsize=8, loc='lower left')
    a.set_title(('Full performance: envelope ≥ 99 % at every speed' if c == 'full' else 'Relaxed: envelope ≥ 97 % at every speed'), fontsize=9.5, loc='left')
F_PARETO = png(fig)
od_sum = []
for o in ODS:
    pf = PK['picks']['full|%d' % o]
    br, ch = pf['best_range'], pf['cheapest']
    od_sum.append(['OD %d' % o, short(br['label']), f(br['range'], 2, True) + ' %', f(-br['dcost'], 0), short(ch['label']), f(ch['range'], 2, True) + ' %', f(-ch['dcost'], 0)])
P['summary'] = ('Summary', f'''
<p class="lede">Aluminium can replace the copper winding of Design9795_1 with stator-only changes (rotor cross-section, magnets, air gap and M235-35A kept), but not at equal efficiency within the 167 mm OD limit. The best balance saves ₹{saving(rec):.0f} per motor ({pct(rec):.0f} % of active-material cost) for {-rng(rec):.2f} % range.</p>
<div class="cards">
 <div class="card hi"><h4>Recommended · OD 167 · 66 mm</h4><p class="big">−₹{saving(rec):.0f}</p><p>Range {rng(rec):+.2f} % (telematics), {rng(rec, WM):+.2f} % (WMTC). Stator F2 (yoke 7.85 mm, slot 22.15 mm deep), 10 turns × 4 × 1.70 mm Al. Torque within 1 % of copper at every speed, {rec['dmass']:+.2f} kg. Winding loss at peak torque +{100 * (pwdg(rec) / pwdg(CU) - 1):.0f} %.</p></div>
 <div class="card"><h4>Cooler alternative · OD 167 · 72 mm</h4><p class="big">−₹{saving(alt):.0f}</p><p>Range {rng(alt):+.2f} % / {rng(alt, WM):+.2f} %. 9 turns × 5 × 1.60 mm. Winding loss at peak +{100 * (pwdg(alt) / pwdg(CU) - 1):.0f} %, {alt['dmass']:+.2f} kg.</p></div>
 <div class="card"><h4>Least range loss · OD 167 · 78 mm</h4><p class="big">−₹{saving(mn):.0f}</p><p>Range {rng(mn):+.2f} % / {rng(mn, WM):+.2f} %. Stator F1 (copper yoke), 8 turns × 6 × 1.50 mm. Winding loss at peak {100 * (pwdg(mn) / pwdg(CU) - 1):+.0f} %, {mn['dmass']:+.2f} kg.</p></div>
</div>
<h3>Cost against range, all designs</h3>
{fig_html(F_PARETO, 'Every aluminium design evaluated (%d), Al wire at ₹370/kg. Lines: best designs at each OD (none both cheaper and better on range). Left: torque within 1 %% of copper at every speed. Right: up to 3 %% short in the high-speed, DC-power-limited region.' % len(DS))}
<h3>Best designs at each OD (full torque)</h3>
{table(['OD', 'Least range loss', 'Range', 'Saving (INR)', 'Largest saving', 'Range', 'Saving (INR)'], od_sum)}
<p>Range parity is not reached: aluminium needs 1.64 × copper's conductor area for the same resistance; the largest slot that keeps the teeth and yoke from saturating gives 1.37–1.52 ×. Each OD page has the details: stator variants, all best designs, efficiency maps, OP1–OP3, drive cycle and cost.</p>
<h3>Key designs</h3>
{spec_table(K3 + [('Drop-in (reference)', DROP)])}
<h3>Operating points OP1–OP3</h3>
{op_delta_table(K3 + [('Drop-in', DROP)])}
<p>Aluminium is better than copper at 9000 rpm (OP1) and at the 9339 rpm cruise point (OP2): copper's single 2.95 mm wire has {OPR[CUL]['OP1']['loss']['cu_ac']:.0f} W AC loss at OP1, aluminium's 1.50–1.70 mm strands {OPR[MINR]['OP1']['loss']['cu_ac']:.0f}–{OPR[REC]['OP1']['loss']['cu_ac']:.0f} W. At peak torque (OP3) the higher DC resistance costs efficiency. Full tables on the OD 167 page.</p>
''')

# method + copper
P['method'] = ('Method and copper baseline', f'''
<p class="lede">Every number comes from the Motor-CAD-equivalent FEMM model of Design9795_1, extended with FEMM solutions of each stator variant, then run through the vehicle drive cycles.</p>
{table(['Stage', 'Script', 'What it computes', 'Count'], [
 ['1. Copper model', 'lab_model.py (v3), make_lab_v2.py', 'Motor-CAD-equivalent Lab model: skew slices, Motor-CAD torque, minor-loop hysteresis + harmonic eddy iron loss', 'validated on 840 Motor-CAD map points'],
 ['2. Stator variants', 'alu9795_femm.py (FEMM)', 'Torque, d-q flux, stator iron-loss sums, slot field, tooth and yoke flux density on a 4 × 4 current grid × 2 rotor positions', '17 stators × 32 = 544 solves'],
 ['3. Design search', 'alu9795_study.py, alu9795_search.py', 'Each stator × stack 64–80 mm × 8/9/10 turns per coil × 1–6 strands: envelope, efficiency map, peak point, drive cycles, cost', '%d designs' % len(DS)],
 ['4. Best designs', 'alu9795_analyse.py', 'Cost against range fronts per OD and criterion; Al price sensitivity', '4 ODs × 2 criteria × 3 prices'],
 ['5. FEMM check', 'alu9795_verify.py', 'Direct solves between grid nodes, 4 rotor positions, against the interpolation', '60 solves'],
 ['6. Operating points', 'alu9795_ops.py', 'OP1–OP3: currents, phase advance, voltage, flux, power factor, loss breakdown', '%d designs' % len(ALLKEYS)],
], 'plain')}
<h3>Copper baseline (Design9795_1)</h3>
<p>Stator OD 160 mm, bore 107 mm, 12 slots, slot depth 17.65 mm, tooth 15.54 mm, yoke 8.85 mm, slot 281.2 mm²; 10-pole spoke ferrite rotor; 64 mm stack; delta, 2 parallel paths, 10 turns per coil of SWG 11 (2.946 mm) copper, fill 0.513; M235-35A. Drive: 52 V, modulation 0.95 (49.4 V phase peak), 360 A rms line, 280 A DC, MTPA. Winding and magnets at 100 °C.</p>
<p>The FEMM model matches Motor-CAD Lab within 0.28 efficiency points (mean over 840 map points), 1.6 % torque and 2.7 % total loss; envelope torque within 0.7 % from 1000 to 9000 rpm (see the Lab validation report).</p>
<div class="pair">{fig_html(f_map(CUL, 'Copper efficiency map (%), with OP1–OP3'), 'Copper efficiency map from the Motor-CAD-equivalent FEMM model, with the three operating points.')}</div>
{op_table('OP1', [])}{op_table('OP2', [])}{op_table('OP3', [])}
<p class="tnote">Motor-CAD values from its envelope export (OP1, OP3). Motor-CAD splits iron loss into stator/rotor and hysteresis/eddy for the machine; its OP1 hysteresis {O.mcad()['OP1']['loss']['hyst']:.0f} W, eddy {O.mcad()['OP1']['loss']['eddy']:.0f} W. FEMM's OP1 iron loss is lower (the known hysteresis gap), so copper's FEMM efficiency is {OPR[CUL]['OP1']['eff'] - O.mcad()['OP1']['eff']:+.2f} points against Motor-CAD at OP1 and {OPR[CUL]['OP3']['eff'] - O.mcad()['OP3']['eff']:+.2f} at OP3. Aluminium designs are compared with FEMM copper, so this offset cancels.</p>
<h3>How an aluminium design is evaluated</h3>
<ul>
<li>Turns: currents scaled by turns (same ampere-turns give the same field), flux linkage × turns, end-winding inductance × turns².</li>
<li>Stack length: flux, torque, iron loss and windage scale with length (2D); end-turn length unchanged.</li>
<li>Stator variant: FEMM differences against the copper stator at the same ampere-turns, added to the copper model (flux), or as ratios (stator iron loss, slot field). Checked against direct FEMM solves: ψq within {max(abs(r['dpsiq_femm4'] - r['dpsiq_interp']) / abs(r['psiq_D0']) for rows in VF.values() for r in rows) * 100:.2f} %, iron-loss ratio within {max(abs(r['rst_Sa_femm4'] / r['rst_Sa_interp'] - 1) for rows in VF.values() for r in rows) * 100:.1f} %.</li>
<li>Winding: aluminium EC-1350 (2.826 × 10⁻⁸ Ω·m at 20 °C, 0.00403/K, 2703 kg/m³) at 100 °C; insulated fill at or below copper's 0.513; IEC and SWG wire sizes from 1.0 mm, 1–6 strands; mean turn = 2L + end turn corrected for tooth width and coil thickness; AC loss by Motor-CAD's hybrid method (factor 2) with FEMM's slot field.</li>
<li>Acceptance: <strong>full</strong> = peak torque ≥ 99.5 % of copper and envelope ≥ 99 % at every speed 0–9500 rpm; <strong>relaxed</strong> = envelope ≥ 97 % (shortfall only in the DC-power-limited region above about 3000 rpm).</li>
</ul>
''')

# one page per OD
OD_TXT = {
    160: 'Same OD and housing as copper. The slot can only grow inward (thinner yoke) or sideways (thinner teeth), and both cost torque per ampere-turn: −1.3 % per mm of yoke, −1.9 % per mm of tooth. Aluminium therefore needs a long stack (72–78 mm).',
    163: 'Three extra millimetres of OD deepen the slot by 1.5 mm with the copper yoke and tooth (slot ×1.118), or by 2.5 mm with a 1 mm thinner yoke.',
    165: 'Five extra millimetres deepen the slot by 2.5 mm with the copper yoke and tooth (slot ×1.198), up to ×1.346 with a 1 mm thinner yoke and tooth.',
    167: 'The housing limit. With the copper yoke and tooth the slot is 3.5 mm deeper (×1.281); with a 1 mm thinner yoke 4.5 mm deeper (×1.365). This OD gives the best designs at every cost level.',
}
for od in ODS:
    ks = KEYS[od]
    svs = SVS[od] if od != 160 else SVS[160]
    n_full = sum(1 for d in DS if round(d['geo']['od']) == od and A.ok(d, 'full'))
    n_rel = sum(1 for d in DS if round(d['geo']['od']) == od and A.ok(d, 'relaxed'))
    n_all = sum(1 for d in DS if round(d['geo']['od']) == od)
    l64 = [d for d in DS if round(d['geo']['od']) == od and d['L'] == 64 and A.ok(d, 'relaxed')]
    l64_txt = ('At the copper stack of 64 mm, no design at this OD keeps full torque at every speed; with up to 3 %% shortfall at high speed the best is %s: range %+.2f %%, saving ₹%.0f.' %
               (short(max(l64, key=rng)['label']), rng(max(l64, key=rng)), saving(max(l64, key=rng))) if l64 else
               'At the copper stack of 64 mm, no design at this OD meets either criterion: aluminium at 64 mm loses torque either at low speed (fewer turns, 360 A limit) or above 2500 rpm (10 turns, higher resistance).')
    main_lab = ks[-1][1] if od != 167 else REC
    P['od%d' % od] = ('OD %d mm' % od, f'''
<p class="lede">{OD_TXT[od]} {n_all} designs evaluated at this OD; {n_full} keep full torque at every speed, {n_rel} stay within 3 %.</p>
<h3>Stator variants at OD {od}</h3>
{femm_table((['D0'] if od != 160 else []) + svs)}
{fig_html(f_slots(svs), 'Slot outlines of the OD %d stators against the copper stator (D0). Tooth tip and slot opening are unchanged in every variant.' % od)}
<h3>Cost against range</h3>
{fig_html(f_scatter(od, ks), 'All designs at OD %d. Dark: full torque at every speed; light blue: within 3 %%; grey: neither. Line: best designs (full). Circles: designs detailed below.' % od)}
<h4>Best designs, full torque at every speed</h4>
{front_table(od, 'full')}
<h4>Best designs, up to 3 % torque shortfall at high speed</h4>
{front_table(od, 'relaxed')}
<p>{l64_txt}</p>
<h3>Effect of stack length and turns</h3>
{fig_html(f_stack(od), 'Best design (range) at each stack length and turn count at OD %d, among those within 3 %% of copper torque; filled dots also meet the full criterion. Fewer turns need a longer stack for the same torque at 360 A.' % od)}
<h3>Designs in detail</h3>
{spec_table(ks)}
<h3>Torque-speed envelope</h3>
{fig_html(f_env(ks), 'Maximum shaft torque at each speed against copper, same drive limits. Right: ratio to copper; dotted lines 99 % and 97 %.')}
<h3>Efficiency maps against copper</h3>
{fig_html(f_dmaps(ks), 'Efficiency change against copper at the same speed and shaft torque (blue = aluminium better). Triangles: OP1–OP3.')}
{fig_html(f_map(main_lab, '%s: efficiency (%%)' % short(main_lab)), 'Efficiency map of %s.' % short(main_lab))}
<h3>Operating points OP1, OP2, OP3</h3>
{op_delta_table(ks)}
{fig_html(f_oploss(ks), 'Loss breakdown at the operating points; label = total loss and efficiency.')}
{op_table('OP1', ks)}{op_table('OP2', ks)}{op_table('OP3', ks)}
<h3>Drive cycle and range</h3>
{cycle_table(ks)}
{fig_html(f_cycle_losses(ks), 'Motor losses over the telematics cycle by type.')}
{fig_html(f_cycle_where(ks), 'Where the range is lost: extra loss against copper over the telematics cycle, by motor speed band, motoring (left) and regenerating (right).')}
<h3>Cost</h3>
{cost_table(ks)}
''')

# comparison across ODs
cmp_labs = [('OD %d: %s' % (od, r), l) for od in ODS for r, l in KEYS[od]]
best_od = []
for od in ODS:
    for c in ('full', 'relaxed'):
        pk = PK['picks']['%s|%d' % (c, od)]
        for nm, key in (('least range loss', 'best_range'), ('within 1 % range', 'cheapest_within_1.0'), ('within 1.5 %', 'cheapest_within_1.5'), ('within 2 %', 'cheapest_within_2.0'), ('largest saving', 'cheapest')):
            if key in pk:
                d = pk[key]
                best_od.append(['%d' % od, c, nm, short(d['label']), f(d['range'], 2, True) + ' %', f(d['range_wmtc'], 2, True) + ' %', f(-d['dcost'], 0), f(d['pwdg'], 0), f(d['dmass'], 2, True)])
fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.8))
for c, ls in (('full', '-'), ('relaxed', '--')):
    ax[0].plot(ODS, [PK['picks']['%s|%d' % (c, o)]['best_range']['range'] for o in ODS], 'o' + ls, color='#2563eb', label=c)
    ax[1].plot(ODS, [-PK['picks']['%s|%d' % (c, o)]['cheapest']['dcost'] for o in ODS], 'o' + ls, color='#059669', label=c)
    for lim, mk in ((1.0, 's'), (1.5, 'D')):
        v = [(-PK['picks']['%s|%d' % (c, o)]['cheapest_within_%.1f' % lim]['dcost']) if 'cheapest_within_%.1f' % lim in PK['picks']['%s|%d' % (c, o)] else np.nan for o in ODS]
        ax[1].plot(ODS, v, mk + ls, color='#7c3aed', ms=5, lw=1, label='%s, within %.1f %% range' % (c, lim))
ax[0].set_ylabel('Least range loss (%)'); ax[1].set_ylabel('Largest saving (INR)')
for a in ax:
    a.set_xlabel('Stator OD (mm)'); a.set_xticks(ODS); a.legend(fontsize=7.5)
F_ODS = png(fig)
P['compare'] = ('Comparison across ODs', f'''
<p class="lede">A larger OD moves the whole cost-range front up and to the right: each mm of OD adds 0.5 mm of slot depth (about 11 mm² per slot) for about 0.06 kg (₹18) of stator steel. That is cheaper than adding stack length, which also lengthens the rotor and magnets.</p>
{fig_html(F_ODS, 'Left: least range loss at each OD. Right: largest saving at each OD, overall and within 1 % / 1.5 % range loss.')}
{table(['OD', 'Criterion', 'Choice', 'Design', 'Range, telematics', 'Range, WMTC', 'Saving (INR)', 'Winding loss at peak (W)', 'Mass change (kg)'], best_od)}
<h3>Efficiency maps of the key designs at each OD</h3>
{''.join(fig_html(f_dmaps(KEYS[od]), 'OD %d: efficiency change against copper.' % od) for od in ODS)}
<h3>What each lever does</h3>
{table(['Change', 'Gain', 'Cost'], [
 ['OD +1 mm (yoke and tooth kept)', 'Slot +0.5 mm deep, +11 mm²; torque per amp −0.1 %', '+0.06 kg stator steel (₹18); new die and housing bore'],
 ['Yoke −1 mm', 'Slot +1 mm deep, +22 mm²', 'Peak torque −1.3 % (yoke B 1.70 → 1.78 T)'],
 ['Tooth −1 mm', 'Slot +15 mm²', 'Peak torque −1.9 %; tooth saturates earlier'],
 ['Stack +2 mm', 'Torque and flux +3 %; resistance per N·m falls', '+₹83 to ₹86 stator steel, rotor laminations and magnets; iron loss and windage +3 %'],
 ['10 → 9 → 8 turns per coil', 'Lower resistance per path (−19 % / −36 %), wider wire', 'Needs about 11 % / 25 % longer stack for the same torque at 360 A'],
 ['More strands of thinner wire', 'AC loss at high speed falls steeply (∝ d²)', 'Wire cost about the same; more strands to handle in winding'],
], 'plain')}
''')

# drive cycle page
T_t, n_t, km_t = SE.CYC[TEL]; T_w, n_w, km_w = SE.CYC[WM]
fig, ax = plt.subplots(2, 2, figsize=(12.5, 6.6))
ax[0, 0].plot(np.arange(len(n_t)), n_t, lw=.6, color='#2563eb'); ax[0, 0].set_title('Telematics: motor speed (rpm), %.1f km' % km_t, loc='left', fontsize=9.5)
ax[0, 1].plot(np.arange(len(n_w)), n_w, lw=.8, color='#059669'); ax[0, 1].set_title('WMTC Class 1 (2 × part 1, reduced): motor speed (rpm), %.1f km' % km_w, loc='left', fontsize=9.5)
for a in ax[0]:
    a.set_xlabel('time step (s)')
for a, (T, n, nm, col) in zip(ax[1], ((T_t, n_t, TEL, '#2563eb'), (T_w, n_w, WM, '#059669'))):
    a.contourf(XM, YM, EFF[CUL], levels=[60, 80, 85, 88, 90, 92, 93, 94, 95, 97], cmap='Greys', alpha=.5)
    on = n > 50
    a.hist2d(n[on], np.abs(T[on]), bins=[np.arange(0, 7001, 250), np.arange(0, 50.1, 2)], cmin=1, cmap='viridis', alpha=.85)
    a.set_title('%s: time at each operating point (|torque|), on the copper map' % nm, loc='left', fontsize=9.5)
    a.set_xlabel('Speed (rpm)'); a.set_ylabel('|Shaft torque| (N·m)'); a.set_xlim(0, 7000); a.set_ylim(0, 50); a.grid(False)
F_CYC = png(fig)
mot = (T_t > 0) & (n_t > 50)
P['cycle'] = ('Drive cycle and range', f'''
<p class="lede">Range is computed from energy per km at the motor terminals over two speed traces: the telematics log supplied earlier and WMTC Class 1. Range change = copper Wh/km ÷ aluminium Wh/km − 1, counting motor losses only.</p>
{fig_html(F_CYC, 'Top: motor speed over the two cycles (the telematics log includes long stops). Bottom: time spent at each operating point, over the copper efficiency map (grey).')}
<h3>Vehicle and method</h3>
<ul>
<li>Vehicle (given): GVW 757 kg, CdA 1.117 m², wheel radius 0.30 m, gear ratio 13.9. Assumed: rolling resistance 0.015, gearbox efficiency 0.95, rotating-mass factor 1.05, flat road.</li>
<li>Motor torque from road load and acceleration at each 1 s step; regenerative braking down to −26.4 N·m (largest regen torque in the log), friction brakes beyond.</li>
<li>Telematics: {km_t:.2f} km, {len(n_t)} steps (logging gaps skipped), motor up to {n_t.max():.0f} rpm, {100 * mot.mean():.0f} % of steps motoring. WMTC Class 1: {km_w:.2f} km, up to {n_w.max():.0f} rpm.</li>
<li>The logged motor torque is negative even at cruise, so energy is computed from the speed trace through the vehicle model, not from the logged torque.</li>
<li>Losses are read from each design's own loss map (MTPA) at the cycle's speed and torque; points above a design's torque limit are held at the limit.</li>
<li>Inverter, battery and vehicle are the same for every design, so the battery-level range change is about the motor-level change.</li>
</ul>
<h3>All key designs</h3>
{cycle_table([(short(l), l) for l in ALLKEYS[1:]])}
<h3>Where the range goes</h3>
{fig_html(f_cycle_losses([(short(l), l) for l in ALLKEYS[1:]]), 'Motor losses over the telematics cycle by type.')}
{fig_html(f_cycle_where(K3 + [('Drop-in', DROP)]), 'Extra loss against copper by speed band, motoring and regenerating, telematics cycle.')}
{fig_html(f_cycle_where(K3 + [('Drop-in', DROP)], WM), 'Same for WMTC Class 1.')}
<ul>
<li>The cycles run mostly at light and medium torque below 6000 rpm. There, aluminium's higher DC resistance and the longer stacks' extra iron loss outweigh its lower AC loss.</li>
<li>Copper's AC winding loss over the telematics cycle is {CU['cycle'][TEL]['loss']['pac'] / km_t:.2f} Wh/km; aluminium's thin strands cut it to {rec['cycle'][TEL]['loss']['pac'] / km_t:.2f} Wh/km (recommended), but its DC loss rises from {CU['cycle'][TEL]['loss']['pcu'] / km_t:.2f} to {rec['cycle'][TEL]['loss']['pcu'] / km_t:.2f} Wh/km and iron loss from {CU['cycle'][TEL]['loss']['pfe'] / km_t:.2f} to {rec['cycle'][TEL]['loss']['pfe'] / km_t:.2f} Wh/km.</li>
<li>Each 1 % of range is about {CU['cycle'][TEL]['Wh_km'] / 100:.2f} Wh/km at the motor. Battery capacity was not given, so the battery cost of making up the range is not included.</li>
</ul>
''')

# cost
P['cost'] = ('Cost', f'''
<p class="lede">Active-material cost per motor, at the prices given: copper wire ₹1100/kg, ferrite magnets ₹800/kg, M235-35A laminations ₹300/kg; aluminium wire ₹370/kg (copper ÷ 3.0, from Indian enamelled-wire prices).</p>
{cost_table([(short(l), l) for l in ALLKEYS[1:]])}
<ul>
<li>Aluminium wire is about 8 % of the motor's material cost, so its price moves the saving by only about ±₹100 between ₹230 and ₹450/kg. The ranking of the designs does not change.</li>
<li>Longer stacks add stator steel, rotor laminations and magnets: about ₹41–43 per mm of stack. A larger OD adds only stator steel, about ₹18 per mm of OD.</li>
<li>Lamination cost is on net mass (stator + rotor). The larger blank at OD 167 (gross {CU['steel_gross']:.2f} → {rec['steel_gross']:.2f} kg for the recommended design) is not included, nor are the die, housing, terminations and winding process.</li>
<li>Copper's active-material cost is ₹{CU['cost']['total']:.0f}: wire ₹{CU['cost']['wire']:.0f}, laminations ₹{CU['cost']['steel']:.0f}, magnets ₹{CU['cost']['magnet']:.0f}.</li>
</ul>
''')

# changes for the recommended design
P['changes'] = ('Recommended design: changes', f'''
<p class="lede">OD 167 mm, stator F2, 66 mm stack, 10 turns of 4 × 1.70 mm aluminium: −₹{saving(rec):.0f} per motor, range {rng(rec):+.2f} % (telematics) / {rng(rec, WM):+.2f} % (WMTC), torque within 1 % of copper at every speed.</p>
{spec_table([('Recommended', REC)])}
<dl>
<dt>Stator lamination</dt><dd>OD 160 → <strong>167 mm</strong>. Slot depth 17.65 → <strong>22.15 mm</strong> (slot bottom radius 71.15 → 75.65 mm). Yoke 8.85 → <strong>7.85 mm</strong>. Unchanged: bore 107 mm, 12 slots, tooth 15.54 mm, tooth tip, slot opening, slot-bottom corner radius, M235-35A 0.35 mm. Slot area 281.2 → {rec['geo']['slot_area']:.1f} mm². New lamination die and housing bore.</dd>
<dt>Stack length</dt><dd>64 → <strong>66 mm</strong> for stator, rotor laminations and magnets (same cross-sections). Shaft, bearing span and housing +2 mm.</dd>
<dt>Winding</dt><dd><strong>Aluminium EC-1350, grade-2 enamel, 4 strands in hand × 1.70 mm</strong> (about 1.77 mm over enamel), 10 turns per coil, delta, 2 parallel paths, one coil per tooth. Fill {rec['fill']:.3f} (copper 0.513). Mean turn {rec['mlt'] * 1e3 if rec['mlt'] < 1 else rec['mlt']:.1f} mm; {rec['R_path'] * 1e3:.2f} mΩ per path at 100 °C (copper {CU['R_path'] * 1e3:.2f}).</dd>
<dt>Terminations</dt><dd>Welded or crimped joints with aluminium-rated lugs or Al-Cu transitions (no soldering), sealed against galvanic corrosion.</dd>
<dt>Drive</dt><dd>Same limits. New flux and resistance tables in the controller (flux and torque per amp +3 % with the 66 mm stack); aluminium temperature coefficient 0.00403/K.</dd>
<dt>Thermal (to check)</dt><dd>Winding loss at peak torque {pwdg(CU):.0f} → {pwdg(rec):.0f} W (+{100 * (pwdg(rec) / pwdg(CU) - 1):.0f} %), current density {CU['peak']['J']:.1f} → {rec['peak']['J']:.1f} A/mm². Check the peak duration in Motor-CAD Thermal. Fallback: the cooler alternative (72 mm, 9 turns × 5 × 1.60 mm, +{100 * (pwdg(alt) / pwdg(CU) - 1):.0f} %), ₹{saving(rec) - saving(alt):.0f} less saving, range {rng(alt) - rng(rec):+.2f} %.</dd>
</dl>
''')

# FEMM variants
vrows = [[sv, f(r['id'], 0), f(r['iq'], 0), f(1e3 * r['psiq_D0']), f(1e3 * r['dpsiq_femm4'], 3, True), f(1e3 * r['dpsiq_interp'], 3, True), f(1e3 * r['dpsid_femm4'], 3, True),
          f(1e3 * r['dpsid_interp'], 3, True), f(r['rst_Sa_femm4'], 3), f(r['rst_Sa_interp'], 3)] for sv, rows in VF.items() for r in rows]
P['femm'] = ('FEMM stator variants', f'''
<p>17 stators, each solved at 4 × 4 d-q currents over the 360 A circle and 2 rotor positions (half model, magnets 100 °C, FEA fill 1.0, as the copper model). Variants change only the slot depth, tooth width and OD; the tooth tip, slot opening, bore, rotor and air gap stay as Design9795_1.</p>
{femm_table(list(FM['designs']))}
<ul>
<li>Thinning the yoke by 1 mm costs about 1.3 % of peak torque (yoke B 1.70 → 1.78 T); by 2 mm, 4.4 %.</li>
<li>Thinning the teeth by 1 mm costs about 1.9 %; tooth flux density is already 2.2–2.3 T at the 99th percentile at full current in every variant.</li>
<li>A larger OD with the same yoke and teeth costs at most 0.6 % (deeper slot, slightly more slot leakage).</li>
</ul>
<h3>Check of the interpolation against direct FEMM solves</h3>
{table(['Stator', 'id (A)', 'iq (A)', 'ψq, copper stator (mV·s)', 'Δψq FEMM', 'Δψq model', 'Δψd FEMM', 'Δψd model', 'Iron-loss ratio FEMM', 'Iron-loss ratio model'], vrows,
       note='Path currents between the grid nodes (10-turn equivalent), mean of 4 rotor positions (0, 3, 6, 9°). Δ = variant − copper stator at the same currents.')}
''')

P['files'] = ('Assumptions and files', f'''
<h3>Limits of the study</h3>
<ul>
<li>2D FEA; end effects only through the end-turn resistance and end-winding inductance.</li>
<li>Thermal behaviour not simulated; the higher winding loss at peak torque needs a Motor-CAD thermal check.</li>
<li>Aluminium wire fill held at copper's 0.513; a real aluminium winding with several strands may reach a different fill.</li>
<li>AC loss by the low-frequency hybrid method; for single wires near the skin depth (about 2.6 mm hot copper, 3.3 mm aluminium at 750 Hz) it is approximate.</li>
<li>Flux maps, demagnetisation, torque ripple and THD were not recomputed for these stator-only variants: the rotor is unchanged, and the slot opening and tooth tip are identical to copper's.</li>
<li>Costs are active materials only.</li>
</ul>
<h3>Files (folder Design9795_1)</h3>
{table(['File', 'Content'], [['alu9795_femm.py / .json', 'FEMM stator variants (17 stators × 32 solves)'], ['alu9795_study.py', 'Aluminium design model on the Motor-CAD-equivalent copper model'],
 ['alu9795_search.py / .json', 'All %d designs: envelope, map, peak, drive cycles, cost' % len(DS)], ['alu9795_analyse.py, alu9795_picks.json', 'Best designs per OD, criterion and Al price'],
 ['alu9795_verify.py / .json', 'FEMM check of the interpolation'], ['alu9795_ops.py / .json', 'OP1–OP3 details'], ['build_alu9795_full.py', 'This report']], 'plain')}
''')

order = ['summary', 'method', 'od160', 'od163', 'od165', 'od167', 'compare', 'cycle', 'cost', 'changes', 'femm', 'files']
nav = ''.join('<a href="#%s" data-p="%s">%s</a>' % (k, k, P[k][0]) for k in order)
pages = ''.join('<section class="page" id="%s" aria-label="%s"><header class="ph"><p class="eyebrow">Design9795 · aluminium winding</p><h2>%s</h2></header>%s</section>' % (k, P[k][0], P[k][0], P[k][1]) for k in order)
tpl = open(os.path.join(OUT, 'template.html'), encoding='utf-8').read()
tpl = (tpl.replace('<title>Spoke V2 Aluminium Winding Study</title>', '<title>Design9795 Aluminium Winding</title>')
          .replace('<h1 class="title">Spoke V2 <span>aluminium</span> winding study</h1>', '<h1 class="title">Design9795 <span>aluminium</span> winding</h1>')
          .replace('FEMM 4.2 · tuned motor model<br>Vehicle: 757 kg, ratio 13.9<br>16 Oct 2026', 'FEMM 4.2 · Motor-CAD-equivalent model<br>OD 160–167 · M235-35A<br>17 Oct 2026')
          .replace('.tw:has(table.sheet)', 'tr.hl td { background: var(--alu-tint) } .tw:has(table.sheet)'))
html = tpl.replace('{{NAV}}', nav).replace('{{PAGES}}', pages)
open(os.path.join(OUT, 'index.html'), 'w', encoding='utf-8').write(html)
print('written', len(html) // 1024, 'kB')
