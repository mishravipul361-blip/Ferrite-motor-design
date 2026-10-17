"""Aluminium winding for Design9795_1: report_alu/alu_winding.html

Inputs: alu9795_search.json (alu9795_search.py), alu9795_picks.json (alu9795_analyse.py), alu9795_femm.json (alu9795_femm.py),
alu9795_verify.json (alu9795_verify.py, optional)."""
import base64, io, json, math, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import alu9795_study as S
import alu9795_search as SE
import alu9795_analyse as A
import backemf as M

H = S.H
R = json.load(open(os.path.join(H, 'alu9795_search.json')))
PK = json.load(open(os.path.join(H, 'alu9795_picks.json')))
FM = json.load(open(os.path.join(H, 'alu9795_femm.json')))
VF = json.load(open(os.path.join(H, 'alu9795_verify.json'))) if os.path.exists(os.path.join(H, 'alu9795_verify.json')) else {}
CU, DS = R['copper'], R['designs']
BY = {d['label']: d for d in DS}
CY = 'Telematics'

REC, ALT, MINR = 'Al F2 L66 N10 4x1.70', 'Al F2 L72 N9 5x1.60', 'Al F1 L78 N8 6x1.50'
DROP = 'Al drop-in (SWG 11 size, 64 mm)'
NAMES = {REC: 'Recommended', ALT: 'Cooler alternative', MINR: 'Least range loss', DROP: 'Drop-in (same wire size)'}
ODS = (160, 163, 165, 167)
OCOL = {160: '#64748b', 163: '#0891b2', 165: '#7c3aed', 167: '#2563eb'}
CCU, CREC, CALT, CMIN, CDROP = '#b45309', '#2563eb', '#059669', '#7c3aed', '#dc2626'
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 2, 'figure.constrained_layout.use': True})


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', bbox_inches='tight', facecolor='white'); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def od(d):
    return int(round(d['geo']['od']))


def design(lab):
    d = BY[lab]
    return S.Design(d['metal'], d['sv'], d['Nc'], d['ns'], d['d'], label=lab, Lmm=d['L'])


# ---------------- figure 1: cost vs range, all designs meeting full performance ----------------
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
for k, c in enumerate(('full', 'relaxed')):
    a = ax[k]
    for o in ODS:
        ds = [d for d in DS if A.ok(d, c) and od(d) == o]
        a.scatter([d['range_pct'][CY] for d in ds], [-A.dcost(d) for d in ds], s=9, color=OCOL[o], alpha=.25, lw=0)
        f = PK['fronts']['%s|%d|370' % (c, o)]
        if f:
            a.plot([d['range'] for d in f], [-d['dcost'] for d in f], 'o-', ms=5, color=OCOL[o], label='OD %d' % o)
    for lab, col in ((REC, CREC), (ALT, CALT), (MINR, CMIN)):
        d = BY[lab]
        a.scatter(d['range_pct'][CY], -A.dcost(d), s=130, facecolor='none', edgecolor=col, lw=2.2, zorder=5)
        a.annotate(NAMES[lab], (d['range_pct'][CY], -A.dcost(d)), xytext=(8, -12), textcoords='offset points', fontsize=8.5, color=col)
    d = BY[DROP]
    a.scatter(d['range_pct'][CY], -A.dcost(d), s=60, marker='x', color=CDROP, zorder=5)
    a.annotate('drop-in (fails on torque\nabove 2500 rpm)', (d['range_pct'][CY], -A.dcost(d)), xytext=(8, -4), textcoords='offset points', fontsize=8, color=CDROP)
    a.set_xlabel('Range change, telematics cycle (%)'); a.set_ylabel('Cost saving per motor (INR)')
    a.set_title(('Full performance: peak torque ≥ 99.5 %, envelope ≥ 99 % at every speed' if c == 'full' else
                 'Relaxed: peak torque ≥ 99.5 %, envelope ≥ 97 % at every speed'), fontsize=9.5, loc='left')
    a.legend(fontsize=8, loc='lower left')
F_PARETO = png(fig)

# ---------------- figure 2: effect of stack length, OD 167 ----------------
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
for Nc, ls in ((8, ':'), (9, '--'), (10, '-')):
    for sv, col in (('F1', '#0891b2'), ('F2', CREC)):
        rows = []
        for Lm in SE.LS:
            ds = [d for d in DS if d['sv'] == sv and d['Nc'] == Nc and d['L'] == Lm]
            if ds:
                rows.append(max(ds, key=lambda d: d['range_pct'][CY]))
        Ls = [d['L'] for d in rows]
        ax[0].plot(Ls, [d['range_pct'][CY] for d in rows], ls, color=col, label='%s, %d turns' % (sv, Nc))
        ax[1].plot(Ls, [-A.dcost(d) for d in rows], ls, color=col)
        ax[2].plot(Ls, [100 * d['env_ratio_min'] for d in rows], ls, color=col)
ax[0].set_ylabel('Range change (%)'); ax[1].set_ylabel('Cost saving (INR)'); ax[2].set_ylabel('Lowest envelope ratio vs copper (%)')
ax[2].axhline(99, color='k', lw=.8); ax[2].set_ylim(80, 102)
for a in ax:
    a.set_xlabel('Stack length (mm)')
ax[0].legend(fontsize=7.5, ncol=2)
ax[0].set_title('Range', loc='left', fontsize=10); ax[1].set_title('Cost saving', loc='left', fontsize=10)
ax[2].set_title('Torque envelope (worst speed)', loc='left', fontsize=10)
F_STACK = png(fig)

# ---------------- figure 3: envelopes ----------------
spd = SE.SP_MAP
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.0))
ref = np.array([np.nan if v is None else v for v in CU['env_T']])
ax[0].plot(spd, ref, color=CCU, label='Copper (Design9795_1)')
for lab, col in ((REC, CREC), (ALT, CALT), (MINR, CMIN), (DROP, CDROP)):
    e = np.array([np.nan if v is None else v for v in BY[lab]['env_T']])
    ax[0].plot(spd, e, '--' if lab == DROP else '-', color=col, lw=1.6, label='%s: %s' % (NAMES[lab], lab.replace('Al ', '')))
    ax[1].plot(spd, 100 * e / ref, '--' if lab == DROP else '-', color=col, lw=1.6)
ax[0].set_ylabel('Shaft torque (N·m)'); ax[1].set_ylabel('Torque vs copper (%)'); ax[1].set_ylim(95, 103); ax[1].axhline(100, color=CCU, lw=1)
ax[1].axhline(99, color='k', lw=.6, ls=':')
for a in ax:
    a.set_xlabel('Speed (rpm)')
ax[0].legend(fontsize=7.5)
F_ENV = png(fig)

# ---------------- figure 4: efficiency maps ----------------
cu_d, rec_d = S.copper(), design(REC)
_, mc = cu_d.run(SE.SP_MAP, SE.TQ_MAP, npts=241, keys=('eff',))
_, mr = rec_d.run(SE.SP_MAP, SE.TQ_MAP, npts=241, keys=('eff',))
X, Y = np.meshgrid(SE.SP_MAP, SE.TQ_MAP)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
lv = [80, 85, 88, 90, 91, 92, 93, 94, 95, 95.5]
for a, m_, t in ((ax[0], mc['eff'], 'Copper'), (ax[1], mr['eff'], 'Recommended aluminium')):
    cs = a.contourf(X, Y, m_, levels=[70] + lv + [97], cmap='viridis')
    c2 = a.contour(X, Y, m_, levels=lv, colors='w', linewidths=.6); a.clabel(c2, fontsize=7, fmt='%g')
    a.set_title(t + ' (%)', loc='left', fontsize=10)
dm = mr['eff'] - mc['eff']
cs = ax[2].contourf(X, Y, dm, levels=np.arange(-3, 1.01, .25), cmap='RdBu', norm=TwoSlopeNorm(0, -3, 1), extend='both')
fig.colorbar(cs, ax=ax[2], label='points')
ax[2].set_title('Aluminium − copper (efficiency points)', loc='left', fontsize=10)
for a in ax:
    a.set_xlabel('Speed (rpm)'); a.set_ylabel('Shaft torque (N·m)'); a.grid(False)
F_MAP = png(fig)
both = ~np.isnan(dm)
DM = dict(mean=float(np.mean(dm[both])), p10=float(np.percentile(dm[both], 10)),
          below2000=float(np.nanmean(np.where((X < 2500) & (Y > 30), dm, np.nan))), light=float(np.nanmean(np.where((X > 2000) & (Y < 15), dm, np.nan))))

# ---------------- figure 5: slot drawing ----------------
def outline(sv):
    s = dict(M.STATOR, **FM['designs'][sv]['geo']['stator'])
    g = M.slot_geometry(s)
    a0, a1 = math.atan2(g['E'][1], g['E'][0]), math.atan2(g['E'][1], -g['E'][0])
    arc = [(g['rb'] * math.cos(t), g['rb'] * math.sin(t)) for t in np.linspace(a0, a1, 40)]
    pts = [g['A'], g['B'], g['C'], g['E']] + arc[1:-1] + [(-x, y) for x, y in (g['E'], g['C'], g['B'], g['A'])]
    return np.array(pts), s


fig, ax = plt.subplots(figsize=(6.2, 5.0))
for sv, col, lab in (('D0', CCU, 'Copper stator, OD 160'), ('F2', CREC, 'Aluminium stator F2, OD 167')):
    p, s = outline(sv)
    ax.fill(p[:, 0], p[:, 1], color=col, alpha=.12); ax.plot(p[:, 0], p[:, 1], color=col, lw=1.6, label=lab)
    t = np.linspace(math.radians(60), math.radians(120), 80)
    ax.plot(s['od'] / 2 * np.cos(t), s['od'] / 2 * np.sin(t), color=col, lw=1.2, ls='--')
t = np.linspace(math.radians(60), math.radians(120), 80)
ax.plot(M.R_BORE * np.cos(t), M.R_BORE * np.sin(t), color='k', lw=1)
ax.set_aspect('equal'); ax.set_ylim(50, 86); ax.set_xlim(-24, 24); ax.legend(fontsize=8, loc='lower center'); ax.set_xlabel('mm'); ax.grid(alpha=.15)
ax.set_title('One slot (dashed: stator OD, black: bore 107)', loc='left', fontsize=10)
F_SLOT = png(fig)

# ---------------- figure 6: cost breakdown ----------------
fig, ax = plt.subplots(figsize=(8.5, 3.4))
labs = ['Copper', 'Drop-in', 'Least range loss', 'Cooler alternative', 'Recommended']
recs = [CU, BY[DROP], BY[MINR], BY[ALT], BY[REC]]
left = np.zeros(len(recs))
for k, (key, col, nm) in enumerate((('wire', '#b45309', 'Winding wire'), ('steel', '#64748b', 'Laminations (stator + rotor)'), ('magnet', '#334155', 'Ferrite magnets'))):
    v = np.array([r['cost'][key] for r in recs])
    ax.barh(labs, v, left=left, color=col, height=.6, label=nm, edgecolor='white', lw=2)
    left += v
for k, r in enumerate(recs):
    ax.text(left[k] + 40, k, '₹%.0f' % r['cost']['total'], va='center', fontsize=9)
ax.set_xlabel('Active-material cost per motor (INR)'); ax.legend(fontsize=8, loc='lower right'); ax.set_xlim(0, 5200); ax.grid(axis='y', alpha=0)
F_COST = png(fig)


# ---------------- tables ----------------
def f(x, fmt='%.2f'):
    return fmt % x


def spec_rows():
    cols = [('Copper (today)', CU), (NAMES[DROP], BY[DROP]), (NAMES[MINR], BY[MINR]), (NAMES[ALT], BY[ALT]), (NAMES[REC], BY[REC])]
    def row(name, fn, hl=False):
        return '<tr%s><td>%s</td>%s</tr>' % (' class="hl"' if hl else '', name, ''.join('<td>%s</td>' % fn(d) for _, d in cols))
    g = lambda d: d['geo']
    out = '<thead><tr><th></th>%s</tr></thead><tbody>' % ''.join('<th>%s</th>' % n for n, _ in cols)
    out += row('Winding metal', lambda d: 'Copper' if d is CU else 'Aluminium EC-1350')
    out += row('Stator OD (mm)', lambda d: '%.0f' % g(d)['od'], True)
    out += row('Stack length, stator = rotor = magnets (mm)', lambda d: '%.0f' % d['L'], True)
    out += row('Slot depth (mm)', lambda d: '%.2f' % g(d)['slot_depth'], True)
    out += row('Tooth width (mm)', lambda d: '%.2f' % g(d)['tooth_w'])
    out += row('Back-iron (yoke) (mm)', lambda d: '%.2f' % g(d)['yoke'], True)
    out += row('Slot area (mm²)', lambda d: '%.1f (×%.3f)' % (g(d)['slot_area'], g(d)['slot_area'] / CU['geo']['slot_area']))
    out += row('Turns per coil', lambda d: '%d' % d['Nc'], True)
    out += row('Strands in hand × bare diameter (mm)', lambda d: '%d × %.2f' % (d['ns'], d['d']), True)
    out += row('Conductor area per turn (mm²)', lambda d: '%.2f' % d['masses']['conductor_area_mm2'])
    out += row('Wire fill (insulated)', lambda d: '%.3f' % d['fill'])
    out += row('Mean turn length (mm)', lambda d: '%.1f' % d['mlt'])
    out += row('Resistance per path, 100 °C (mΩ)', lambda d: '%.2f' % (d['R_path'] * 1e3))
    out += row('Peak torque, 1000 rpm (N·m)', lambda d: '%.2f' % d['peak']['T'])
    out += row('Lowest envelope ratio vs copper', lambda d: '—' if d is CU else '%.1f %%' % (100 * d['env_ratio_min']))
    out += row('Winding loss at peak torque (W)', lambda d: '%.0f' % (d['peak']['pcu'] + d['peak']['pac']))
    out += row('Current density at peak (A/mm² rms)', lambda d: '%.1f' % d['peak']['J'])
    out += row('Peak efficiency (%)', lambda d: '%.2f' % d['peak_eff'])
    out += row('Map efficiency, mean change (points)', lambda d: '—' if d is CU else '%+.2f' % d['map_dEta_mean'])
    out += row('Telematics energy (Wh/km)', lambda d: '%.3f' % d['cycle'][CY]['Wh_km'])
    out += row('Range change, telematics', lambda d: '—' if d is CU else '%+.2f %%' % d['range_pct'][CY], True)
    out += row('Range change, WMTC Class 1', lambda d: '—' if d is CU else '%+.2f %%' % d['range_pct']['WMTC Class 1'])
    out += row('Winding mass (kg)', lambda d: '%.3f' % d['masses']['winding'])
    out += row('Stator laminations (kg, net)', lambda d: '%.3f' % d['masses']['stator_lam'])
    out += row('Rotor laminations + magnets (kg)', lambda d: '%.3f' % (d['masses']['rotor_lam'] + d['masses']['magnets']))
    out += row('Active mass (kg)', lambda d: '%.3f' % d['masses']['active'])
    out += row('Material cost (INR)', lambda d: '%.0f' % d['cost']['total'])
    out += row('Saving vs copper (INR)', lambda d: '—' if d is CU else '%.0f (%.1f %%)' % (-A.dcost(d), -100 * A.dcost(d) / CU['cost']['total']), True)
    return out + '</tbody>'


def od_rows(c):
    out = ''
    for o in ODS:
        fr = PK['fronts']['%s|%d|370' % (c, o)]
        if not fr:
            out += '<tr><td>%d</td><td colspan="9">no design meets this criterion</td></tr>' % o
            continue
        for d in sorted(fr, key=lambda d: -d['range']):
            x = BY[d['label']]
            out += '<tr%s><td>%d</td><td>%s</td><td>%.0f</td><td>%d × (%d × %.2f)</td><td>%.2f</td><td>%.1f %%</td><td>%+.2f %%</td><td>%+.2f %%</td><td>%.0f (%.1f %%)</td><td>%.0f W</td></tr>' % (
                ' class="hl"' if d['label'] == REC else '', o, d['sv'], d['L'], d['Nc'], d['ns'], d['d'], x['geo']['yoke'], 100 * d['env'],
                d['range'], d['range_wmtc'], -d['dcost'], -100 * d['dcost'] / CU['cost']['total'], d['pwdg'])
    return out


def price_rows():
    out = ''
    for lab in (MINR, ALT, REC, DROP):
        d = BY[lab]
        out += '<tr><td>%s</td>%s</tr>' % (NAMES[lab], ''.join('<td>%.0f</td>' % -A.dcost(d, p) for p in A.AL_PRICES))
    return out


def loss_rows():
    out = ''
    for nm, d in (('Copper', CU), (NAMES[MINR], BY[MINR]), (NAMES[ALT], BY[ALT]), (NAMES[REC], BY[REC]), (NAMES[DROP], BY[DROP])):
        l = d['cycle'][CY]['loss']; km = d['cycle'][CY]['km']
        out += '<tr><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.3f</td></tr>' % (
            nm, l['pcu'] / km, l['pac'] / km, l['pfe'] / km, l['pme'] / km, l['loss'] / km, d['cycle'][CY]['eff_mot'], d['cycle'][CY]['Wh_km'])
    return out


def femm_rows():
    out = ''
    for sv in ('D0', 'A1', 'A4', 'G1', 'B1', 'F1', 'F2', 'F3', 'F4'):
        v = FM['designs'][sv]; g = v['geo']
        out += '<tr%s><td>%s</td><td>%.0f</td><td>%.2f</td><td>%.2f</td><td>%.1f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.3f</td></tr>' % (
            ' class="hl"' if sv == 'F2' else '', sv, g['stator']['od'], g['yoke_mm'], g['stator']['tooth_w'], g['slot_area'], np.max(v['T']),
            np.max(v['tooth_B99']), np.max(v['yoke_B99']), g['st_mass'])
    return out


def verify_rows():
    out = ''
    for sv, rows in VF.items():
        for r in rows:
            out += '<tr><td>%s</td><td>%.0f</td><td>%.0f</td><td>%.2f</td><td>%+.2f</td><td>%+.2f</td><td>%.2f</td><td>%+.2f</td><td>%+.2f</td><td>%.3f</td><td>%.3f</td></tr>' % (
                sv, r['id'], r['iq'], 1e3 * r['psiq_D0'], 1e3 * r['dpsiq_femm4'], 1e3 * r['dpsiq_interp'], 1e3 * r['psid_D0'], 1e3 * r['dpsid_femm4'],
                1e3 * r['dpsid_interp'], r['rst_Sa_femm4'], r['rst_Sa_interp'])
    return out


def vmax():
    e = [abs(r['dpsiq_femm4'] - r['dpsiq_interp']) / abs(r['psiq_D0']) for rows in VF.values() for r in rows if abs(r['psiq_D0']) > 1e-3]
    s = [abs(r['rst_Sa_femm4'] / r['rst_Sa_interp'] - 1) for rows in VF.values() for r in rows]
    return (100 * max(e), 100 * max(s)) if e else (float('nan'), float('nan'))


rec, alt, mn, dr = BY[REC], BY[ALT], BY[MINR], BY[DROP]
sv_ = lambda d: -A.dcost(d)
pct = lambda d: -100 * A.dcost(d) / CU['cost']['total']
VE = vmax()

# ---------------- operating points OP1-OP3 ----------------
OPJ = json.load(open(os.path.join(H, 'alu9795_ops.json')))
ODN = list(OPJ['designs'])
LK = [('cu_dc', 'DC winding', '#b45309'), ('cu_ac', 'AC winding', '#f59e0b'), ('st_hyst', 'Stator hysteresis', '#475569'),
      ('st_eddy', 'Stator eddy', '#94a3b8'), ('rot', 'Rotor iron', '#7c3aed'), ('mech', 'Bearings + windage', '#0f766e')]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, opk in zip(ax, ('OP1', 'OP2', 'OP3')):
    left = np.zeros(len(ODN))
    for key, nm, col in LK:
        v = np.array([(OPJ['designs'][n]['ops'][opk]['loss']['rot_hyst'] + OPJ['designs'][n]['ops'][opk]['loss']['rot_eddy']) if key == 'rot'
                      else OPJ['designs'][n]['ops'][opk]['loss'][key] for n in ODN])
        a.barh(ODN, v, left=left, color=col, height=.62, label=nm, edgecolor='white', lw=1.5); left += v
    for k, n in enumerate(ODN):
        r = OPJ['designs'][n]['ops'][opk]
        a.text(left[k] + 15, k, '%.0f W · %.2f %%' % (r['loss']['total'], r['eff']), va='center', fontsize=8)
    a.set_xlim(0, left.max() * 1.42); a.invert_yaxis(); a.grid(axis='y', alpha=0); a.set_xlabel('Loss (W)')
    a.set_title('%s: %s' % (opk, OPJ['ops'][opk]), loc='left', fontsize=10)
    if opk != 'OP1':
        a.set_yticklabels([])
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='outside lower center', ncol=6, fontsize=8)
F_OPS = png(fig)


def op_table(opk):
    cols = ([('Motor-CAD, copper', OPJ['mcad_copper'][opk])] if opk in OPJ['mcad_copper'] else []) + [(n, OPJ['designs'][n]['ops'][opk]) for n in ODN]

    def row(name, fn, hl=False):
        cells = []
        for _, r in cols:
            try:
                cells.append(fn(r))
            except KeyError:
                cells.append('—')
        return '<tr%s><td>%s</td>%s</tr>' % (' class="hl"' if hl else '', name, ''.join('<td>%s</td>' % c for c in cells))
    L_ = lambda r: r['loss']
    rot = lambda r: L_(r)['rot'] if 'rot' in L_(r) else L_(r)['rot_hyst'] + L_(r)['rot_eddy']
    stl = lambda r: L_(r)['st'] if 'st' in L_(r) else L_(r)['st_hyst'] + L_(r)['st_eddy']
    out = '<thead><tr><th>%s</th>%s</tr></thead><tbody>' % (opk, ''.join('<th>%s</th>' % n for n, _ in cols))
    out += row('Speed (rpm)', lambda r: '%.0f' % r['speed'])
    out += row('Shaft torque (N·m)', lambda r: '%.2f' % r['T_shaft'], True)
    out += row('Electromagnetic torque (N·m)', lambda r: '%.2f' % r['T_em'])
    out += row('Shaft power (W)', lambda r: '%.0f' % r['P_shaft'])
    out += row('Input power (W)', lambda r: '%.0f' % r['P_in'])
    out += row('Efficiency (%)', lambda r: '%.2f' % r['eff'], True)
    out += row('Line current (A rms)', lambda r: '%.1f' % r['I_line_rms'])
    out += row('Phase current (A rms, delta)', lambda r: '%.1f' % r['I_phase_rms'])
    out += row('Id, phase (A peak)', lambda r: '%.1f' % r['Id_phase_pk'])
    out += row('Iq, phase (A peak)', lambda r: '%.1f' % r['Iq_phase_pk'])
    out += row('Phase advance (° el)', lambda r: '%.1f' % r['phase_advance'], True)
    out += row('Phase voltage (V peak; limit %.1f)' % OPJ['designs'][ODN[0]]['ops'][opk]['V_limit'], lambda r: '%.2f' % r['V_phase_pk'])
    out += row('Power factor', lambda r: '%.3f' % r['pf'])
    out += row('ψd (mV·s)', lambda r: '%.2f' % r['psid_mVs'])
    out += row('ψq (mV·s)', lambda r: '%.2f' % r['psiq_mVs'])
    out += row('DC bus current (A)', lambda r: '%.1f' % r['Idc'])
    out += row('Current density (A/mm² rms)', lambda r: '%.1f' % r['J'])
    out += row('Active limit', lambda r: r['limit'])
    out += row('DC winding loss (W)', lambda r: '%.1f' % L_(r)['cu_dc'], True)
    out += row('AC winding loss (W)', lambda r: '%.1f' % L_(r)['cu_ac'], True)
    out += row('Stator iron, hysteresis (W)', lambda r: '%.1f' % L_(r)['st_hyst'])
    out += row('Stator iron, eddy (W)', lambda r: '%.1f' % L_(r)['st_eddy'])
    out += row('Stator iron, total (W)', lambda r: '%.1f' % stl(r), True)
    out += row('Rotor iron (W)', lambda r: '%.1f' % rot(r), True)
    out += row('Iron, total (W)', lambda r: '%.1f' % L_(r)['iron'])
    out += row('Bearings + windage (W)', lambda r: '%.1f' % L_(r)['mech'], True)
    out += row('Total loss (W)', lambda r: '%.1f' % L_(r)['total'], True)
    return out + '</tbody>'


def op_eff(opk, n):
    return OPJ['designs'][n]['ops'][opk]['eff'] - OPJ['designs'][ODN[0]]['ops'][opk]['eff']


OPD = OPJ['designs']
CSS = open(os.path.join(H, 'report_emf', 'backemf_validation.html'), encoding='utf-8').read().split('<style>')[1].split('</style>')[0]
CSS += """
.wrap { max-width:1180px } tr.hl td { background:var(--ok-bg) } h3 { font:600 18px/1.25 var(--f-d); margin:20px 0 6px }
.two { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr)); gap:16px } td, th { white-space:nowrap }
"""
html = f'''<title>Design9795 Aluminium Winding</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap"><div class="sheet">
<p class="eyebrow">Design9795_1 · copper to aluminium winding · stator only · M235-35A unchanged · OD 160–167 mm (167 hard limit) · 52 V · 360 A rms · Idc 280 A</p>
<h1>Aluminium winding for Design9795: cost against range</h1>
<div class="verdict"><strong>Recommended: OD 167 mm, 66 mm stack, 1 mm thinner back-iron with a 4.5 mm deeper slot, 10 turns of 4 × 1.70 mm aluminium.</strong> It saves ₹{sv_(rec):.0f} per motor ({pct(rec):.0f} % of the active-material cost), loses {-rec['range_pct'][CY]:.2f} % of range on the telematics cycle ({-rec['range_pct']['WMTC Class 1']:.2f} % on WMTC Class 1), keeps the copper motor's torque within 1 % at every speed, and is {-rec['dmass']:.2f} kg lighter.
Aluminium cannot reach copper's efficiency within OD 167: the smallest range loss found is {-mn['range_pct'][CY]:.2f} % (OD 167, 78 mm stack, 8 turns), and it saves ₹{sv_(mn):.0f}. Going from that design to the recommended one buys a further ₹{sv_(rec) - sv_(mn):.0f} for {mn['range_pct'][CY] - rec['range_pct'][CY]:.2f} % of range.</div>

<div class="tw"><table class="kpi"><thead><tr><th>Headline (telematics cycle, Al wire ₹370/kg)</th><th>Copper</th><th>Recommended Al</th><th>Change</th></tr></thead><tbody>
<tr><td>Active-material cost</td><td>₹{CU['cost']['total']:.0f}</td><td>₹{rec['cost']['total']:.0f}</td><td>−₹{sv_(rec):.0f} ({-pct(rec):+.1f} %)</td></tr>
<tr><td>Energy at the motor terminals</td><td>{CU['cycle'][CY]['Wh_km']:.2f} Wh/km</td><td>{rec['cycle'][CY]['Wh_km']:.2f} Wh/km</td><td>range {rec['range_pct'][CY]:+.2f} %</td></tr>
<tr><td>Cycle motor efficiency (motoring)</td><td>{CU['cycle'][CY]['eff_mot']:.2f} %</td><td>{rec['cycle'][CY]['eff_mot']:.2f} %</td><td>{rec['cycle'][CY]['eff_mot'] - CU['cycle'][CY]['eff_mot']:+.2f} points</td></tr>
<tr><td>Peak efficiency</td><td>{CU['peak_eff']:.2f} %</td><td>{rec['peak_eff']:.2f} %</td><td>{rec['peak_eff'] - CU['peak_eff']:+.2f} points</td></tr>
<tr><td>Peak torque, 1000 rpm</td><td>{CU['peak']['T']:.2f} N·m</td><td>{rec['peak']['T']:.2f} N·m</td><td>{100 * (rec['peak']['T'] / CU['peak']['T'] - 1):+.1f} %</td></tr>
<tr><td>Active mass</td><td>{CU['masses']['active']:.2f} kg</td><td>{rec['masses']['active']:.2f} kg</td><td>{rec['dmass']:+.2f} kg</td></tr>
<tr><td>Winding loss at peak torque (thermal)</td><td>{CU['peak']['pcu'] + CU['peak']['pac']:.0f} W</td><td>{rec['peak']['pcu'] + rec['peak']['pac']:.0f} W</td><td>{100 * ((rec['peak']['pcu'] + rec['peak']['pac']) / (CU['peak']['pcu'] + CU['peak']['pac']) - 1):+.0f} %</td></tr>
</tbody></table></div>

<h2>1. The trade-off</h2>
<figure><img src="{F_PARETO}" alt="Cost saving against range change"><figcaption>Every aluminium design evaluated ({len(DS)} designs: 17 stators × stack 64–80 mm × 8/9/10 turns per coil × 1–6 strands in hand), Al wire at ₹370/kg. Lines: best designs at each OD (no design is both cheaper and better on range). Left: designs that keep the copper motor's torque within 1 % at every speed. Right: designs allowed to lose up to 3 % in the high-speed, DC-power-limited region.</figcaption></figure>
<p>Points to the right have less range loss, points higher up save more. Every curve slopes down: moving to aluminium always costs some range, and saving more money costs more range. A larger OD moves the whole curve up and to the right, because the deeper slot holds more aluminium, so OD 167 gives the best designs. Range parity with copper is not reached anywhere within OD 167: aluminium needs 1.64× the copper's conductor area for the same resistance, while the largest slot that keeps the teeth and yoke from saturating gives at most 1.37–1.52×.</p>
<h3>Three aluminium designs worth considering</h3>
<div class="tw"><table>{spec_rows()}</table></div>
<p class="note">Rows highlighted: what changes against copper. All designs keep the rotor cross-section, magnets, air gap, bore 107 mm, tooth tip and slot opening, delta connection, 2 parallel paths and M235-35A. The drop-in (copper's SWG 11 size in aluminium, nothing else changed) is shown for reference: it saves the most but loses 3 % torque at 3000 rpm (1–2 % above) and {-dr['range_pct'][CY]:.1f} % range, and its winding loss at peak torque is {100 * ((dr['peak']['pcu'] + dr['peak']['pac']) / (CU['peak']['pcu'] + CU['peak']['pac']) - 1):.0f} % higher.</p>
<figure><img src="{F_COST}" alt="Cost breakdown"><figcaption>Active-material cost: wire (with 5 % waste) at ₹1100/kg copper or ₹370/kg aluminium, net laminations ₹300/kg, ferrite ₹800/kg. A longer stack adds rotor laminations and magnets as well as stator steel, which is why the least-range-loss design saves less.</figcaption></figure>

<h2>2. Results for each OD from 160 to 167 mm</h2>
<p>Best designs at each OD (none of them both cheaper and better on range than another one at the same OD), keeping the copper motor's torque within 1 % at every speed. Highlighted: the recommended design.</p>
<div class="tw"><table><thead><tr><th>OD</th><th>Stator</th><th>Stack (mm)</th><th>Turns × (strands × d)</th><th>Yoke (mm)</th><th>Envelope (worst)</th><th>Range, telematics</th><th>Range, WMTC</th><th>Saving (INR)</th><th>Winding loss at peak</th></tr></thead><tbody>{od_rows('full')}</tbody></table></div>
<h3>Relaxed: up to 3 % torque shortfall at high speed</h3>
<div class="tw"><table><thead><tr><th>OD</th><th>Stator</th><th>Stack (mm)</th><th>Turns × (strands × d)</th><th>Yoke (mm)</th><th>Envelope (worst)</th><th>Range, telematics</th><th>Range, WMTC</th><th>Saving (INR)</th><th>Winding loss at peak</th></tr></thead><tbody>{od_rows('relaxed')}</tbody></table></div>
<p>What the OD buys, at full performance:</p>
<ul>
<li><strong>OD 160</strong> (same housing and lamination blank): the slot can only grow by thinning the yoke and teeth, which costs torque, so a longer stack (74–78 mm) is needed. Best range {PK['picks']['full|160']['best_range']['range']:+.2f} %, best saving ₹{-PK['picks']['full|160']['cheapest']['dcost']:.0f} at {PK['picks']['full|160']['cheapest']['range']:+.2f} %.</li>
<li><strong>OD 163</strong>: best range {PK['picks']['full|163']['best_range']['range']:+.2f} %, best saving ₹{-PK['picks']['full|163']['cheapest']['dcost']:.0f} at {PK['picks']['full|163']['cheapest']['range']:+.2f} %.</li>
<li><strong>OD 165</strong>: best range {PK['picks']['full|165']['best_range']['range']:+.2f} %, best saving ₹{-PK['picks']['full|165']['cheapest']['dcost']:.0f} at {PK['picks']['full|165']['cheapest']['range']:+.2f} %.</li>
<li><strong>OD 167</strong>: best range {PK['picks']['full|167']['best_range']['range']:+.2f} %, best saving ₹{-PK['picks']['full|167']['cheapest']['dcost']:.0f} at {PK['picks']['full|167']['cheapest']['range']:+.2f} %. Each mm of OD adds 0.5 mm of slot depth and about 11 mm² of slot area per slot, for about 0.06 kg of stator steel (₹18).</li>
</ul>
<p>At the original 64 mm stack, no aluminium design keeps full torque at every speed at any OD up to 167. With 10 turns, the higher resistance costs voltage and power above 2500 rpm; with fewer turns, the 360 A current limit costs low-speed torque. A slightly longer stack (66 mm at OD 167) restores the torque, so the stack length had to change. The rotor lamination and magnet cross-sections stay exactly as they are; only their length changes.</p>

<h2>3. Effect of stack length and turns (OD 167)</h2>
<figure><img src="{F_STACK}" alt="Stack length sweep"><figcaption>Best wire option for each stack length and turn count, stators F1 (OD 167, yoke 8.85 mm as copper) and F2 (OD 167, yoke 7.85 mm). Fewer turns need a longer stack to make the same torque at the 360 A limit; the longer stack then lowers the copper loss per N·m but adds iron loss, windage and material cost.</figcaption></figure>

<h2>4. Performance of the recommended design</h2>
<figure><img src="{F_ENV}" alt="Torque envelopes"><figcaption>Maximum shaft torque at each speed, same drive limits as copper (52 V, modulation 0.95, 360 A rms, 280 A DC, MTPA).</figcaption></figure>
<figure><img src="{F_MAP}" alt="Efficiency maps"><figcaption>Efficiency maps (winding and magnets at 100 °C). Right: change in efficiency points at the same speed and torque. Mean over the map {DM['mean']:+.2f} points; high torque below 2500 rpm {DM['below2000']:+.2f} points; light load above 2000 rpm {DM['light']:+.2f} points. Aluminium is better at high speed and light load, where copper's single 2.95 mm wire has more AC (eddy) loss than 1.70 mm strands. The small steps in the torque ratio come from the current grid of the model (±0.3 %).</figcaption></figure>
<h3>Where the extra energy goes, telematics cycle (Wh/km)</h3>
<div class="tw"><table><thead><tr><th>Design</th><th>DC winding</th><th>AC winding</th><th>Iron</th><th>Bearings and windage</th><th>Total loss</th><th>Motoring efficiency (%)</th><th>Energy at terminals</th></tr></thead><tbody>{loss_rows()}</tbody></table></div>
<p>The recommended design has more DC winding loss (higher resistance) and more iron loss (longer stack and a slightly more saturated yoke). It has less AC winding loss, because its strands are thinner than the single 2.95 mm copper wire. On this cycle the motor works mostly at light and medium load, where iron loss counts as much as winding loss. That is why making the stack longer for less resistance stops paying back beyond about 78 mm.</p>

<h2>5. Operating points OP1, OP2 and OP3</h2>
<p>The three operating points of the earlier aluminium study: <strong>OP1</strong> 9000 rpm at maximum power, <strong>OP2</strong> 9339 rpm at 3.24 N·m (light-load high-speed cruise), <strong>OP3</strong> 2000 rpm at maximum torque (360 A rms line). In the earlier study OP3 was set at 44.34 N·m, the peak torque of that motor. Design9795 copper makes 44.11 N·m at 2000 rpm (Motor-CAD), so here OP3 is each design's own maximum at 360 A. Each point uses the least current that meets the torque within the 52 V, 360 A and 280 A DC limits (MTPA, with field weakening where needed). Winding and magnets are at 100 °C. Where Motor-CAD's envelope export has the point, copper is also shown as Motor-CAD computes it.</p>
<figure><img src="{F_OPS}" alt="Loss breakdown at OP1-OP3"><figcaption>Loss breakdown at the three operating points; the label gives total loss and efficiency.</figcaption></figure>
<div class="tw"><table><thead><tr><th>Efficiency change vs copper (points)</th>{''.join('<th>%s</th>' % n for n in ODN[1:])}</tr></thead><tbody>{''.join('<tr><td>%s: %s</td>%s</tr>' % (k, OPJ['ops'][k], ''.join('<td>%+.2f</td>' % op_eff(k, n) for n in ODN[1:])) for k in ('OP1', 'OP2', 'OP3'))}</tbody></table></div>
<h3>OP1: 9000 rpm, maximum power</h3>
<div class="tw"><table>{op_table('OP1')}</table></div>
<h3>OP2: 9339 rpm, 3.24 N·m</h3>
<div class="tw"><table>{op_table('OP2')}</table></div>
<h3>OP3: 2000 rpm, maximum torque at 360 A</h3>
<div class="tw"><table>{op_table('OP3')}</table></div>
<p class="note">Torque at OP2 within +0.05 N·m of 3.24 (current grid step). Motor-CAD splits its iron loss into stator and rotor, and into hysteresis and eddy over the whole machine (OP1: hysteresis {OPJ['mcad_copper']['OP1']['loss']['hyst']:.0f} W, eddy {OPJ['mcad_copper']['OP1']['loss']['eddy']:.0f} W; OP3: {OPJ['mcad_copper']['OP3']['loss']['hyst']:.0f} W and {OPJ['mcad_copper']['OP3']['loss']['eddy']:.0f} W), so its stator rows show the total only. At OP1, FEMM's copper iron loss is lower than Motor-CAD's (the residual hysteresis gap noted in the Lab validation) and its AC loss {100 * (1 - OPD[ODN[0]]['ops']['OP1']['loss']['cu_ac'] / OPJ['mcad_copper']['OP1']['loss']['cu_ac']):.0f} % lower. Copper's efficiency is {OPD[ODN[0]]['ops']['OP1']['eff'] - OPJ['mcad_copper']['OP1']['eff']:+.2f} points against Motor-CAD at OP1 and {OPD[ODN[0]]['ops']['OP3']['eff'] - OPJ['mcad_copper']['OP3']['eff']:+.2f} at OP3.</p>
<ul>
<li><strong>OP1 (9000 rpm):</strong> all three aluminium designs are <em>more</em> efficient than copper ({op_eff('OP1', 'Recommended'):+.2f}, {op_eff('OP1', 'Cooler alternative'):+.2f} and {op_eff('OP1', 'Least range loss'):+.2f} points). Copper's single 2.95 mm wire has {OPD[ODN[0]]['ops']['OP1']['loss']['cu_ac']:.0f} W of AC (eddy) winding loss at 750 Hz. Aluminium's thinner strands (1.50–1.70 mm) and higher resistivity cut this to {OPD['Least range loss']['ops']['OP1']['loss']['cu_ac']:.0f}–{OPD['Recommended']['ops']['OP1']['loss']['cu_ac']:.0f} W, more than the extra DC loss. The 280 A DC limit applies, so the lower loss leaves more for the shaft.</li>
<li><strong>OP2 (cruise):</strong> AC loss drops in the same way, and the recommended design is {op_eff('OP2', 'Recommended'):+.2f} points against copper. The longer stacks add iron loss and windage, so the least-range-loss design gains less here ({op_eff('OP2', 'Least range loss'):+.2f}).</li>
<li><strong>OP3 (peak torque):</strong> this is where aluminium costs efficiency. DC winding loss is {OPD['Recommended']['ops']['OP3']['loss']['cu_dc']:.0f} W against copper's {OPD[ODN[0]]['ops']['OP3']['loss']['cu_dc']:.0f} W in the recommended design ({op_eff('OP3', 'Recommended'):+.2f} points), {OPD['Cooler alternative']['ops']['OP3']['loss']['cu_dc']:.0f} W in the cooler alternative ({op_eff('OP3', 'Cooler alternative'):+.2f}) and {OPD['Least range loss']['ops']['OP3']['loss']['cu_dc']:.0f} W in the least-range-loss design ({op_eff('OP3', 'Least range loss'):+.2f}, at {100 * (OPD['Least range loss']['ops']['OP3']['T_shaft'] / OPD[ODN[0]]['ops']['OP3']['T_shaft'] - 1):+.1f} % torque). The drop-in loses {-op_eff('OP3', 'Drop-in'):.1f} points.</li>
</ul>
<p>Over the telematics cycle (section 4 table), the AC winding loss saved by aluminium is smaller than the extra DC winding loss and iron loss, so the range change stays negative even though aluminium wins at OP1 and OP2.</p>

<h2>6. Aluminium price sensitivity</h2>
<div class="tw"><table><thead><tr><th>Design</th><th>Saving, Al ₹230/kg</th><th>Saving, Al ₹370/kg (base)</th><th>Saving, Al ₹450/kg</th></tr></thead><tbody>{price_rows()}</tbody></table></div>
<p class="note">Base ₹370/kg = copper ₹1100 ÷ 3.0 (Indian enamelled aluminium wire is quoted at 2.6–3× below copper; on the metal alone, LME, the ratio is about 4.8×). Aluminium wire is only about 8 % of the motor's material cost, so its price moves the saving by about ±₹100. The ranking of the designs does not change.</p>

<h2>7. Complete list of changes (recommended design)</h2>
<dl>
<dt>Stator lamination</dt><dd>OD 160 → <strong>167 mm</strong>. Slot depth {CU['geo']['slot_depth']:.2f} → <strong>{rec['geo']['slot_depth']:.2f} mm</strong> (slot bottom radius 71.15 → 75.65 mm). Back-iron {CU['geo']['yoke']:.2f} → <strong>{rec['geo']['yoke']:.2f} mm</strong>. Unchanged: bore 107 mm, 12 slots, tooth width 15.54 mm, tooth tip, slot opening, slot-bottom corner radius, M235-35A 0.35 mm. Slot area 281.2 → {rec['geo']['slot_area']:.1f} mm² (×{rec['geo']['slot_area'] / CU['geo']['slot_area']:.3f}). New lamination die and a new housing bore (167 mm).</dd>
<dt>Stack length</dt><dd>64 → <strong>{rec['L']:.0f} mm</strong> for the stator, rotor laminations and magnets (same cross-sections; magnets +{rec['L'] - 64:.0f} mm long). Shaft, bearing span and housing length to suit (+{rec['L'] - 64:.0f} mm, plus the longer end windings).</dd>
<dt>Winding</dt><dd>Copper SWG 11 (2.946 mm), 1 strand → <strong>aluminium EC-1350 grade 2 enamel, 4 strands in hand of 1.70 mm</strong> (IEC size, about 1.77 mm over enamel). 10 turns per coil, delta, 2 parallel paths, concentrated (one coil per tooth), same as copper. Insulated wire fill {rec['fill']:.3f} (copper 0.513). Mean turn length {rec['mlt']:.1f} mm; resistance {rec['R_path'] * 1e3:.2f} mΩ per path at 100 °C (copper {CU['R_path'] * 1e3:.2f}).</dd>
<dt>Terminations</dt><dd>Aluminium needs welded or crimped joints with aluminium-compatible lugs or Al-Cu bimetal transitions (no soldering), with sealing against galvanic corrosion at the Al-Cu interface. 4 strands make the coils easier to wind and to form round the tooth than one 2.95 mm wire.</dd>
<dt>Drive and calibration</dt><dd>Same 52 V / 360 A rms / 280 A DC limits. New flux-linkage and resistance tables in the controller (ψ and torque per amp change by about +3 % with the 66 mm stack). Resistance temperature coefficient 0.00403/K for aluminium.</dd>
<dt>Thermal</dt><dd>Winding loss at peak torque rises from {CU['peak']['pcu'] + CU['peak']['pac']:.0f} W to {rec['peak']['pcu'] + rec['peak']['pac']:.0f} W ({100 * ((rec['peak']['pcu'] + rec['peak']['pac']) / (CU['peak']['pcu'] + CU['peak']['pac']) - 1):+.0f} %). The current density falls from {CU['peak']['J']:.1f} to {rec['peak']['J']:.1f} A/mm², the stator surface is larger and the winding is spread over a longer stack. Check the peak-torque duration in Motor-CAD Thermal before release. If it does not pass, the cooler alternative (72 mm stack, 9 turns of 5 × 1.60 mm) adds only {100 * ((alt['peak']['pcu'] + alt['peak']['pac']) / (CU['peak']['pcu'] + CU['peak']['pac']) - 1):+.0f} % winding loss, for ₹{sv_(rec) - sv_(alt):.0f} less saving and {rec['range_pct'][CY] - alt['range_pct'][CY]:+.2f} % range.</dd>
<dt>Mass</dt><dd>Winding {CU['masses']['winding']:.2f} → {rec['masses']['winding']:.2f} kg; stator steel {CU['masses']['stator_lam']:.2f} → {rec['masses']['stator_lam']:.2f} kg; rotor laminations + magnets {CU['masses']['rotor_lam'] + CU['masses']['magnets']:.2f} → {rec['masses']['rotor_lam'] + rec['masses']['magnets']:.2f} kg; active total {CU['masses']['active']:.2f} → {rec['masses']['active']:.2f} kg.</dd>
</dl>
<figure><img src="{F_SLOT}" alt="Slot drawing"><figcaption>One slot of the copper stator and of the recommended stator. The tooth, tip and opening are unchanged; the slot is deeper and the OD larger.</figcaption></figure>

<h2>8. Stator variants solved in FEMM</h2>
<p>17 stators, each solved at a 4 × 4 grid of d-q currents over the 360 A circle at 2 rotor positions (half model, magnets 100 °C), on top of the Motor-CAD-equivalent copper model (skew, Motor-CAD torque, v3 iron loss). The differences against the copper stator (flux linkage, stator iron-loss sums, slot field) are applied to the copper Lab model.</p>
<div class="tw"><table><thead><tr><th>Stator</th><th>OD</th><th>Yoke (mm)</th><th>Tooth (mm)</th><th>Slot (mm²)</th><th>Max torque, 64 mm, 10 turns (N·m, 2D)</th><th>Tooth B 99th pct (T)</th><th>Yoke B 99th pct (T)</th><th>Stator steel (kg, 64 mm)</th></tr></thead><tbody>{femm_rows()}</tbody></table></div>
<p class="note">Thinning the yoke by 1 mm costs about 1.3 % of peak torque (yoke flux density 1.70 → 1.78 T); thinning the teeth by 1 mm costs about 1.9 %. A larger OD with the same yoke and teeth costs only 0.6 % at most (the slot gets deeper, so slot leakage rises slightly). That is why the best designs keep the teeth and thin the yoke by at most 1 mm.</p>
{'<h3>Check of the interpolation against direct FEMM solves</h3><div class="tw"><table><thead><tr><th>Stator</th><th>id (A)</th><th>iq (A)</th><th>ψq copper stator (mV·s)</th><th>Δψq FEMM</th><th>Δψq model</th><th>ψd copper stator</th><th>Δψd FEMM</th><th>Δψd model</th><th>Iron-loss ratio FEMM</th><th>Iron-loss ratio model</th></tr></thead><tbody>' + verify_rows() + '</tbody></table></div><p class="note">Currents between the grid nodes, 4 rotor positions (0, 3, 6, 9°). Largest error of the model in ψq: %.2f %% of ψq; in the stator iron-loss ratio: %.1f %%.</p>' % VE if VF else ''}

<h2>9. Method and assumptions</h2>
<dl>
<dt>Electromagnetics</dt><dd>Motor-CAD-equivalent FEMM Lab model of Design9795_1 (efficiency within 0.28 points of Motor-CAD over 840 map points), with the stator-variant corrections from FEMM. Turns: currents scaled by turns (same ampere-turns, same field), flux linkage × turns, end-winding inductance × turns². Stack length: flux, torque, iron loss and windage scale with length (2D); end-winding length is unchanged.</dd>
<dt>Winding</dt><dd>Aluminium EC-1350, 2.826 × 10⁻⁸ Ω·m at 20 °C, 0.00403/K, 2703 kg/m³; winding at 100 °C. Insulated fill held at or below copper's 0.513 (grade-2 enamel build 0.042 + 0.014 d mm). Wire sizes: IEC 60317 and SWG from 1.0 mm, 1–6 strands in hand. Mean turn length from the stack length, tooth width and coil thickness. AC loss by the hybrid method (factor 2), slot field from FEMM.</dd>
<dt>Range</dt><dd>Telematics speed trace (Spoke V2, {R['cycles'][CY]['km']:.1f} km) and WMTC Class 1 (2 × part 1 reduced, {R['cycles']['WMTC Class 1']['km']:.1f} km). Vehicle GVW 757 kg, CdA 1.117 m², wheel radius 0.30 m, ratio 13.9; assumed Crr 0.015, gearbox 0.95, rotating mass factor 1.05, flat road, regenerative braking down to −26.4 N·m. Range change = energy per km at the motor terminals, copper ÷ aluminium − 1; inverter and battery taken as equal for all designs, so the battery-level change is about the same.</dd>
<dt>Cost</dt><dd>Active materials only, INR/kg: copper wire 1100, aluminium wire 370 (sensitivity 230–450), ferrite 800, M235-35A finished laminations 300 (net mass, stator + rotor), wire waste 5 %. Not included: lamination die, housing, terminations, winding process, scrap from the larger blank (the gross blank goes from {CU['steel_gross']:.2f} to {rec['steel_gross']:.2f} kg).</dd>
<dt>Performance criteria</dt><dd>Full: peak torque at 1000 rpm ≥ 99.5 % of copper's and envelope ≥ 99 % of copper's at every speed from 0 to 9500 rpm. Relaxed: envelope ≥ 97 %.</dd>
<dt>Files</dt><dd>alu9795_femm.py (FEMM stator variants), alu9795_study.py (aluminium design model), alu9795_search.py (designs, cycles, cost), alu9795_analyse.py (best designs), alu9795_verify.py (FEMM check), alu9795_ops.py (OP1–OP3), build_alu9795_report.py (this report).</dd>
</dl>
</div></div>
'''
os.makedirs(os.path.join(H, 'report_alu'), exist_ok=True)
open(os.path.join(H, 'report_alu', 'alu_winding.html'), 'w', encoding='utf-8').write(html)
print(len(html) // 1024, 'kB', DM)
