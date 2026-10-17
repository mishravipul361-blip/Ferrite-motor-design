"""Builds alu_report.html (aluminium winding study) from alu_sweep.json, alu_femm.json and the scaled Lab model."""
import base64, io, json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import alu_study as A
import backemf as M

H = A.H
R = json.load(open(os.path.join(H, 'alu_sweep.json')))
FE = json.load(open(os.path.join(H, 'alu_femm.json')))
for r in R:                                                   # operating point not reachable -> large penalty
    r['dEta_ops'] = {o: (v if v is not None and v == v else -99.0) for o, v in r['dEta_ops'].items()}
BY = {r['label']: r for r in R}
CU = R[0]
lamof = lambda r: r.get('lam') or next((l for l in ('NO27', 'NO20') if r['label'].endswith(l)), 'M235-35A')
worst = lambda r: min(r['dEta_ops'].values())
OK = lambda r, tol=-0.1: r['metal'] == 'Al' and worst(r) >= tol and r['map_dEta_mean'] >= -0.15 and r['env_ratio_min'] >= 0.985
ALS = [r for r in R if r['metal'] == 'Al']

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', bbox_inches='tight', facecolor='white'); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def design_of(r):
    lab = r['label']
    if lab.startswith('Al drop-in'):
        return A.Design('Al', 'S0', 63.0, 20, 1, 2.9464, label=lab)
    if lab.startswith('Copper baseline'):
        return A.baseline()
    return A.Design(r['metal'], r['sv'], r['L'], r['Nph'], r['ns'], r['d'], lam=lamof(r), label=lab)


# ---------------- design picks ----------------
DROP = BY['Al drop-in SWG 11 S0']
okall = [r for r in ALS if OK(r)]
FINAL = min(okall, key=lambda r: r['masses']['total'])                                     # lightest equal-efficiency design
ok_m235 = [r for r in ALS if OK(r) and lamof(r) == 'M235-35A']
SAME_STEEL = min(ok_m235, key=lambda r: r['masses']['total']) if ok_m235 else max(
    [r for r in ALS if lamof(r) == 'M235-35A' and r['env_ratio_min'] >= 0.985], key=lambda r: worst(r))
ok_s0 = [r for r in ALS if OK(r) and r['sv'] == 'S0']
SAME_LAM = min(ok_s0, key=lambda r: r['masses']['total']) if ok_s0 else None                # same lamination tool, only steel grade changes
light = [r for r in ALS if r['env_ratio_min'] >= 0.985 and r['masses']['total'] <= CU['masses']['total'] + 0.05]
LIGHT = max(light, key=lambda r: (worst(r) + r['map_dEta_mean'])) if light else None       # no heavier than copper, best efficiency
PICKS = [('Copper baseline', CU), ('Aluminium drop-in', DROP), ('Same steel grade, best', SAME_STEEL)]
if SAME_LAM and SAME_LAM is not FINAL:
    PICKS.append(('Current lamination shape, thin steel', SAME_LAM))
if LIGHT:
    PICKS.append(('No heavier than copper', LIGHT))
PICKS.append(('Recommended', FINAL))
print('\n'.join('%-36s %s  worst %+.2f map %+.2f m %.2f' % (n, r['label'], worst(r), r['map_dEta_mean'], r['masses']['total']) for n, r in PICKS))

# ---------------- full maps for the picks ----------------
SP = np.arange(0, 12001, 100); TQ = np.arange(0.5, 60.01, 0.5)
FULL = {}
for name, r in PICKS:
    d = design_of(r)
    env, mp = d.run(SP, TQ, npts=241, keys=('eff', 'loss', 'pcu', 'pac', 'pfe', 'pme', 'I_line'))
    FULL[name] = dict(d=d, env=env, mp=mp, ops={o: d.op(o) for o in A.OPS})
S, T = np.meshgrid(SP, TQ)
img = {}


def diffmap(name):
    a, b = FULL[name]['mp']['eff'], FULL['Copper baseline']['mp']['eff']
    Z = np.where(np.isnan(b), np.nan, a - b)
    lost = ~np.isnan(b) & np.isnan(a)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    lv = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, -0.2, 0.2, 0.5, 1, 1.5, 2, 3])
    cs = ax.contourf(S, T, np.clip(Z, -6, 3), levels=lv, cmap='RdBu', norm=TwoSlopeNorm(0, -6, 3), extend='both')
    if lost.any():
        ax.contourf(S, T, lost.astype(float), levels=[0.5, 1.5], colors='none', hatches=['xx'])
    ax.plot(SP[1:], [e['Tsh'] if e else np.nan for e in FULL['Copper baseline']['env'][1:]], color='#52514e', lw=1.2, ls='--', label='copper envelope')
    fig.colorbar(cs, ax=ax, label='efficiency difference, points (Al − Cu)')
    ax.set_xlabel('speed [rpm]'); ax.set_ylabel('shaft torque [N·m]'); ax.set_xlim(0, 12000); ax.set_ylim(0, 50)
    for o, c in A.OPS.items():
        p = FULL['Copper baseline']['ops'][o]
        ax.plot(c['n'], p['Tsh'], 'o', mfc='white', mec='black', ms=6); ax.annotate(o, (c['n'], p['Tsh']), xytext=(6, 4), textcoords='offset points', fontsize=9)
    ax.legend(loc='upper right', fontsize=8); ax.grid(False)
    return png(fig)


for name, _ in PICKS[1:]:
    img['diff_' + name] = diffmap(name)

# efficiency maps side by side: copper vs recommended
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
lv = [70, 75, 80, 84, 86, 88, 90, 91, 92, 93, 94, 95, 96]
for a_, name in zip(ax, ('Copper baseline', 'Recommended')):
    cs = a_.contourf(S, T, FULL[name]['mp']['eff'], levels=lv, cmap='RdYlGn', extend='both')
    cl = a_.contour(S, T, FULL[name]['mp']['eff'], levels=[86, 90, 92, 94, 95], colors='k', linewidths=.5); a_.clabel(cl, fmt='%.0f', fontsize=8)
    a_.set_title(name + (' (Cu, 63 mm)' if name.startswith('Copper') else ' (Al)'), fontsize=10); a_.set_xlabel('speed [rpm]'); a_.grid(False)
    a_.set_xlim(0, 12000); a_.set_ylim(0, 50)
ax[0].set_ylabel('shaft torque [N·m]'); fig.colorbar(cs, ax=ax, label='efficiency [%]')
img['effmaps'] = png(fig)

# envelope
fig, ax = plt.subplots(1, 2, figsize=(13, 4.0))
for k, (name, _) in enumerate(PICKS):
    e = FULL[name]['env']
    sty = dict(color='#0b0b0b', lw=2.4) if k == 0 else dict(color=C[(k - 1) % 8], lw=1.6, ls='-' if name == 'Recommended' else '--')
    ax[0].plot(SP, [x['Tsh'] if x else np.nan for x in e], label=name, **sty)
    ax[1].plot(SP[1:], [x['psh'] / 1e3 if x else np.nan for x in e[1:]], label=name, **sty)
ax[0].set_ylabel('max shaft torque [N·m]'); ax[1].set_ylabel('max shaft power [kW]')
for a_ in ax:
    a_.set_xlabel('speed [rpm]'); a_.set_xlim(0, 12000)
ax[0].legend(fontsize=8)
img['env'] = png(fig)

# losses at the operating points
fig, ax = plt.subplots(1, 3, figsize=(13, 4.0), sharey=False)
comp = [('pcu', 'Copper/Al DC'), ('pac', 'AC winding'), ('pfe', 'Iron'), ('pme', 'Mechanical')]
for a_, o in zip(ax, A.OPS):
    names = [n for n, _ in PICKS]
    bottom = np.zeros(len(names))
    for j, (k, lab) in enumerate(comp):
        v = np.array([FULL[n]['ops'][o][k] for n in names])
        a_.bar(range(len(names)), v, bottom=bottom, color=C[j], edgecolor='white', linewidth=1.5, label=lab, width=.7)
        bottom += v
    for i, n in enumerate(names):
        a_.text(i, bottom[i] + 12, '%.1f %%' % FULL[n]['ops'][o]['eff'], ha='center', fontsize=8)
    a_.set_xticks(range(len(names))); a_.set_xticklabels([n.replace(', ', ',\n').replace(' ', '\n', 1) for n in names], fontsize=7)
    a_.set_title('%s · %s' % (o, A.OPS[o]['label']), fontsize=10); a_.set_ylabel('loss [W]'); a_.grid(axis='x', visible=False)
ax[0].legend(fontsize=8, loc='upper left')
img['ops'] = png(fig)

# trade-off: mass vs worst-OP efficiency difference, Pareto fronts per family
fam = {}
for r in ALS:
    if r['env_ratio_min'] < 0.985 or r['label'].startswith('Al drop-in') or worst(r) < -50:
        continue
    key = ('current slot' if r['sv'] == 'S0' else 'enlarged slot') + ', ' + {'M235-35A': 'M235-35A (as now)', 'NO27': '0.27 mm steel', 'NO20': '0.20 mm steel'}[lamof(r)]
    fam.setdefault(key, []).append(r)
order = ['current slot, M235-35A (as now)', 'enlarged slot, M235-35A (as now)', 'current slot, 0.27 mm steel', 'enlarged slot, 0.27 mm steel',
         'current slot, 0.20 mm steel', 'enlarged slot, 0.20 mm steel']
FRONTS = {}
for key in order:
    pts = sorted(fam.get(key, []), key=lambda r: r['masses']['total'])
    front, best = [], -1e9
    for r in pts:
        if worst(r) > best + 1e-6:
            front.append(r); best = worst(r)
    FRONTS[key] = front
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for k, key in enumerate(order):
    f = FRONTS[key]
    if not f:
        continue
    ls = '-' if 'enlarged' in key else '--'
    ax[0].plot([r['masses']['total'] for r in f], [worst(r) for r in f], ls, marker='o', ms=4, color=C[k // 2 * 2 + (k % 2) if False else [0, 0, 1, 1, 2, 2][k]], alpha=1 if 'enlarged' in key else .75, label=key)
    ax[1].plot([r['masses']['total'] for r in f], [r['map_dEta_mean'] for r in f], ls, marker='o', ms=4, color=C[[0, 0, 1, 1, 2, 2][k]], alpha=1 if 'enlarged' in key else .75, label=key)
for a_ in ax:
    a_.axhline(0, color='#52514e', lw=1); a_.axvline(CU['masses']['total'], color='#52514e', lw=1, ls=':')
    a_.plot(CU['masses']['total'], 0, 's', color='black', ms=7); a_.annotate('copper', (CU['masses']['total'], 0), xytext=(5, 6), textcoords='offset points', fontsize=9)
    a_.plot(DROP['masses']['total'], worst(DROP) if a_ is ax[0] else DROP['map_dEta_mean'], 'x', color='#e34948', ms=8, mew=2)
    a_.annotate('Al drop-in', (DROP['masses']['total'], worst(DROP) if a_ is ax[0] else DROP['map_dEta_mean']), xytext=(6, -4), textcoords='offset points', fontsize=9)
    a_.plot(FINAL['masses']['total'], worst(FINAL) if a_ is ax[0] else FINAL['map_dEta_mean'], '*', color='#008300', ms=14)
    a_.set_xlabel('active mass: winding + stator + rotor laminations + magnets [kg]')
ax[0].set_ylabel('worst of OP1/OP2/OP3: efficiency Al − Cu [points]'); ax[1].set_ylabel('mean over the map: efficiency Al − Cu [points]')
ax[0].set_ylim(-5.5, 2.5); ax[1].set_ylim(-2.2, 1.0)
ax[0].legend(fontsize=7.5, loc='lower right')
img['trade'] = png(fig)

# stack-length sweep for the recommended family
famr = sorted([r for r in ALS if r['sv'] == FINAL['sv'] and lamof(r) == lamof(FINAL) and r['Nph'] == FINAL['Nph']], key=lambda r: r['L'])
bestL = {}
for r in famr:
    k = round(r['L'], 2)
    if k not in bestL or worst(r) > worst(bestL[k]):
        bestL[k] = r
famr = [bestL[k] for k in sorted(bestL)]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.9))
for j, o in enumerate(A.OPS):
    ax[0].plot([r['L'] for r in famr], [r['dEta_ops'][o] for r in famr], marker='o', ms=4, color=C[j], label=o + ' · ' + A.OPS[o]['label'])
ax[0].plot([r['L'] for r in famr], [r['map_dEta_mean'] for r in famr], marker='s', ms=4, color=C[6], label='map mean')
ax[0].axhline(0, color='#52514e', lw=1); ax[0].axvline(FINAL['L'], color='#008300', lw=1, ls=':')
ax[0].set_xlabel('stack length [mm]'); ax[0].set_ylabel('efficiency Al − Cu [points]'); ax[0].legend(fontsize=8)
parts = [('winding', 'winding'), ('stator_lam', 'stator laminations'), ('rotor_lam', 'rotor laminations'), ('magnets', 'magnets')]
b = np.zeros(len(famr))
for j, (k, lab) in enumerate(parts):
    v = np.array([r['masses'][k] for r in famr]); ax[1].bar([r['L'] for r in famr], v, bottom=b, width=2.2, color=C[j], edgecolor='white', linewidth=1.2, label=lab); b += v
ax[1].axhline(CU['masses']['total'], color='black', lw=1.2, ls='--'); ax[1].annotate('copper design %.2f kg' % CU['masses']['total'], (famr[0]['L'] - 1, CU['masses']['total']), xytext=(0, 5), textcoords='offset points', fontsize=8)
ax[1].set_xlabel('stack length [mm]'); ax[1].set_ylabel('active mass [kg]'); ax[1].legend(fontsize=8, loc='upper left'); ax[1].grid(axis='x', visible=False)
img['stack'] = png(fig)


# slot drawing: baseline vs recommended slot, one slot pitch
def slot_poly(st):
    s = dict(M.STATOR, **st)
    rb = 53.5 + s['slot_depth']
    old = dict(M.STATOR); M.STATOR.update(s)
    g = M.slot_geometry(s); M.STATOR.clear(); M.STATOR.update(old)
    A_, B_, C_, E_ = g['A'], g['B'], g['C'], g['E']
    a0, a1 = math.atan2(E_[1], E_[0]), math.atan2(E_[1], -E_[0])
    arc = [(rb * math.cos(t), rb * math.sin(t)) for t in np.linspace(a0, a1, 50)]
    mir = lambda p: (-p[0], p[1])
    return [A_, B_, C_, E_] + arc[1:-1] + [mir(E_), mir(C_), mir(B_), mir(A_)]


fig, ax = plt.subplots(figsize=(6.2, 5.2))
th = np.linspace(math.radians(75), math.radians(105), 80)
for r_, c_ in ((53.5, '#898781'), (80, '#898781')):
    ax.plot(r_ * np.cos(th), r_ * np.sin(th), color=c_, lw=1)
for st, lab, col, ls in ((FE['designs']['S0']['geo'], 'copper design slot (S0, %.1f mm²)' % A.slot_area('S0'), '#52514e', '--'),
                         (FE['designs'][FINAL['sv']]['geo'], 'recommended slot (%s, %.1f mm²)' % (FINAL['sv'], A.slot_area(FINAL['sv'])), '#2a78d6', '-')):
    p = np.array(slot_poly(st)); p = np.vstack([p, p[:1]])
    ax.plot(p[:, 0], p[:, 1], ls, color=col, lw=2, label=lab)
    if ls == '-':
        ax.fill(p[:, 0], p[:, 1], color=col, alpha=.12)
ax.set_aspect('equal'); ax.legend(fontsize=8, loc='lower center'); ax.set_xlabel('mm'); ax.set_ylabel('mm')
ax.set_xlim(-22, 22); ax.set_ylim(48, 82)
img['slot'] = png(fig)

# ---------------- tables ----------------
f2 = lambda v: '%.2f' % v
sg = lambda v: '%+.2f' % v
pill = lambda v, good=-0.1, mid=-0.5: '<span class="pill %s">%+.2f</span>' % ('ok' if v >= good else 'mid' if v >= mid else 'bad', v)

mat_rows = ''.join('<tr><td class="txt">%s</td><td>%s</td><td>%s</td><td class="txt">%s</td></tr>' % x for x in [
    ('Resistivity at 20 °C', '1.724 µΩ·cm', '2.826 µΩ·cm', '×1.64 (61 % IACS)'),
    ('Temperature coefficient', '0.00386 /K', '0.00403 /K', 'at 100 °C the ratio is ×1.66'),
    ('Resistivity at 100 °C', '%.3f µΩ·cm' % (A.MAT['Cu']['rho20'] * (1 + A.MAT['Cu']['alpha'] * 80) * 1e8), '%.3f µΩ·cm' % (A.MAT['Al']['rho20'] * (1 + A.MAT['Al']['alpha'] * 80) * 1e8), '×%.3f' % ((A.MAT['Al']['rho20'] * (1 + A.MAT['Al']['alpha'] * 80)) / (A.MAT['Cu']['rho20'] * (1 + A.MAT['Cu']['alpha'] * 80)))),
    ('Density', '8933 kg/m³', '2703 kg/m³', '×0.30: the same resistance in Al weighs about half'),
    ('Thermal conductivity', '≈ 390 W/m·K', '≈ 230 W/m·K', 'small effect: slot heat flow is limited by enamel and resin'),
    ('Thermal expansion', '17 µm/m·K', '23 µm/m·K', 'more movement against enamel, resin and terminals'),
    ('Tensile strength (annealed)', '≈ 220 MPa', '≈ 70–90 MPa', 'lower winding tension, softer bends; creep at bolted joints'),
    ('Surface', 'solderable', 'hard Al₂O₃ oxide', 'needs crimp/ultrasonic/TIG terminations or Al–Cu transition pieces')])

ft = FE['designs']
s0 = ft['S0']
slot_rows = ''
for k, x in ft.items():
    g = x['geo']; As = A.slot_area(k)
    dT = (x['T'][-2][-1] / s0['T'][-2][-1] - 1) * 100            # point id -49 / iq 147 (near peak-torque MTPA)
    dFe = (np.mean(x['st_Sa']) / np.mean(s0['st_Sa']) - 1) * 100
    slot_rows += '<tr><td>%s</td><td>%.1f</td><td>%.1f</td><td>%.1f</td><td>%.1f (%+.1f %%)</td><td>%+.2f %%</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%+.1f %%</td></tr>' % (
        k, g['slot_depth'], g['tooth_w'], 80 - 53.5 - g['slot_depth'], As, (As / A.slot_area('S0') - 1) * 100, dT, x['tooth_Bmean'][-2][-1], x['yoke_Bmean'][-2][-1],
        x['yoke_B99'][-2][-1], dFe)

wire_rows = ''
for w in A.wire_options(FINAL['sv'], FINAL['Nph'] // 2, 'Al'):
    d = A.Design('Al', FINAL['sv'], FINAL['L'], FINAL['Nph'], w['ns'], w['d'], lam=lamof(FINAL))
    r = BY.get(d.label)
    if not r:
        continue
    wire_rows += '<tr%s><td>%d × %.3f mm</td><td>%.2f</td><td>%.3f</td><td>%.2f</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
        ' style="font-weight:600"' if r is FINAL else '', w['ns'], w['d'], w['A'], w['fill'], d.R_path * 1e3 / 2, sg(r['dEta_ops']['OP1']), sg(r['dEta_ops']['OP2']),
        sg(r['dEta_ops']['OP3']), sg(r['map_dEta_mean']))


def spec(r, d):
    m = r['masses']
    return [('Conductor', A.MAT[d.metal]['name']), ('Stator slot', '%s · depth %.1f mm, tooth %.1f mm, %.1f mm²' % (r['sv'], A.SLOTS.geo(r['sv'])['slot_depth'], A.SLOTS.geo(r['sv'])['tooth_w'], d.As)),
            ('Lamination', A.LAMS[lamof(r)]['name']), ('Stack length', '%.1f mm' % d.L),
            ('Turns', '%d turns/coil · %d series turns/phase · delta, 2 parallel paths' % (d.Nc, d.Nph)),
            ('Conductor', '%d strand%s × %.3f mm bare (%.2f mm² per turn)' % (d.ns, 's' if d.ns > 1 else '', d.d, d.A * 1e6)),
            ('Wire slot fill', '%.3f (limit %.4f, as the copper design)' % (d.fill, A.FILL_MAX)),
            ('Mean turn length', '%.1f mm' % (d.mlt * 1e3)), ('Phase resistance at 100 °C', '%.2f mΩ' % (d.R_path * 1e3 / 2)),
            ('Winding mass', '%.2f kg' % m['winding']), ('Active mass', '%.2f kg' % m['total'])]


def spec_table(names):
    rows = {n: dict(spec(dict(PICKS)[n], FULL[n]['d'])) for n in names}
    keys = list(rows[names[0]].keys())
    keys = list(dict.fromkeys(k for k in keys))
    out = '<tr><th></th>' + ''.join('<th>%s</th>' % n for n in names) + '</tr>'
    for k in keys:
        out += '<tr><td class="txt">%s</td>' % k + ''.join('<td>%s</td>' % rows[n].get(k, '') for n in names) + '</tr>'
    return out


import alu_cost as AC
COST = lambda r, P=None: AC.cost(r['label'], lamof(r), P or AC.PRICE)[2]
pick_rows = ''
for n, r in PICKS:
    m = r['masses']; o = r['dEta_ops']
    pick_rows += '<tr><td class="txt"><b>%s</b><br><span class="src">%s</span></td><td>%.1f</td><td>%d</td><td>%s</td><td>%s</td><td>%.2f</td><td>%+.2f</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%.1f %%</td><td>%.2f</td><td>%s</td></tr>' % (
        n, r['label'], r['L'], r['Nph'], r['sv'], lamof(r), m['total'], m['total'] - CU['masses']['total'],
        pill(o['OP1']) if n != 'Copper baseline' else '—', pill(o['OP2']) if n != 'Copper baseline' else '—', pill(o['OP3']) if n != 'Copper baseline' else '—',
        pill(r['map_dEta_mean'], -0.15, -0.6) if n != 'Copper baseline' else '—', 100 * r['env_ratio_min'],
        COST(r)['total'], '—' if n == 'Copper baseline' else '%+.2f' % (COST(r)['total'] - COST(CU)['total']))

front_rows = ''
for key in order:
    for r in FRONTS[key]:
        if worst(r) < -3:
            continue
        front_rows += '<tr><td class="txt">%s</td><td>%s</td><td>%.1f</td><td>%d</td><td>%d × %.3f</td><td>%.2f (%+.2f)</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
            key, r['sv'], r['L'], r['Nph'], r['ns'], r['d'], r['masses']['total'], r['dmass'], sg(r['dEta_ops']['OP1']), sg(r['dEta_ops']['OP2']), sg(r['dEta_ops']['OP3']), sg(r['map_dEta_mean']))

op_rows = ''
for o in A.OPS:
    for n, _ in PICKS:
        p = FULL[n]['ops'][o]
        op_rows += '<tr><td>%s</td><td class="txt">%s</td><td>%.2f</td><td>%.0f</td><td>%.1f</td><td>%.0f</td><td>%.0f</td><td>%.0f</td><td>%.0f</td><td>%.0f</td><td>%.2f</td></tr>' % (
            o, n, p['Tsh'], p['I_line'], p['v'], p['pcu'], p['pac'], p['pfe'], p['pme'], p['loss'], p['eff'])

mass_rows = ''
for n, r in PICKS:
    m = r['masses']
    mass_rows += '<tr><td class="txt">%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td><b>%.2f</b></td><td>%+.2f</td></tr>' % (
        n, m['winding'], m['stator_lam'], m['rotor_lam'], m['magnets'], m['total'], m['total'] - CU['masses']['total'])

fd = FULL['Recommended']['d']
V = V0 = dict(
    FINAL_LABEL=FINAL['label'], FINAL_L='%.1f' % FINAL['L'], FINAL_N=str(FINAL['Nph']), FINAL_NC=str(FINAL['Nph'] // 2), FINAL_SV=FINAL['sv'],
    FINAL_LAM=A.LAMS[lamof(FINAL)]['name'], FINAL_WIRE='%d × %.3f mm' % (FINAL['ns'], FINAL['d']), FINAL_M='%.2f' % FINAL['masses']['total'],
    FINAL_DM='%+.2f' % FINAL['dmass'], FINAL_WDG='%.2f' % FINAL['masses']['winding'], CU_WDG='%.2f' % CU['masses']['winding'],
    CU_M='%.2f' % CU['masses']['total'], FINAL_AS='%.1f' % A.slot_area(FINAL['sv']), AS0='%.1f' % A.slot_area('S0'),
    FINAL_DEPTH='%.1f' % A.SLOTS.geo(FINAL['sv'])['slot_depth'], FINAL_TOOTH='%.1f' % A.SLOTS.geo(FINAL['sv'])['tooth_w'],
    FINAL_YOKE='%.1f' % (80 - 53.5 - A.SLOTS.geo(FINAL['sv'])['slot_depth']),
    FINAL_OP1=sg(FINAL['dEta_ops']['OP1']), FINAL_OP2=sg(FINAL['dEta_ops']['OP2']), FINAL_OP3=sg(FINAL['dEta_ops']['OP3']), FINAL_MAP=sg(FINAL['map_dEta_mean']),
    FINAL_PEAK='%.2f' % np.nanmax(FULL['Recommended']['mp']['eff']), CU_PEAK='%.2f' % np.nanmax(FULL['Copper baseline']['mp']['eff']),
    DROP_OP1=sg(DROP['dEta_ops']['OP1']), DROP_OP2=sg(DROP['dEta_ops']['OP2']), DROP_OP3=sg(DROP['dEta_ops']['OP3']), DROP_MAP=sg(DROP['map_dEta_mean']),
    DROP_M='%.2f' % DROP['masses']['total'], DROP_DM='%+.2f' % DROP['dmass'], DROP_WDG='%.2f' % DROP['masses']['winding'],
    DROP_PEAK='%.2f' % np.nanmax(FULL['Aluminium drop-in']['mp']['eff']), DROP_ENV='%.1f' % (100 * DROP['env_ratio_min']),
    SS_LABEL=SAME_STEEL['label'], SS_L='%.1f' % SAME_STEEL['L'], SS_N=str(SAME_STEEL['Nph']), SS_SV=SAME_STEEL['sv'], SS_WORST=sg(worst(SAME_STEEL)),
    SS_M='%.2f' % SAME_STEEL['masses']['total'], SS_DM='%+.2f' % SAME_STEEL['dmass'], SS_MAP=sg(SAME_STEEL['map_dEta_mean']),
    SS_OP1=sg(SAME_STEEL['dEta_ops']['OP1']), SS_OP2=sg(SAME_STEEL['dEta_ops']['OP2']), SS_OP3=sg(SAME_STEEL['dEta_ops']['OP3']),
    RPH_CU='%.2f' % (A.baseline().R_path * 1e3 / 2), RPH_DROP='%.2f' % (FULL['Aluminium drop-in']['d'].R_path * 1e3 / 2), RPH_FIN='%.2f' % (fd.R_path * 1e3 / 2),
    MLT_FIN='%.1f' % (fd.mlt * 1e3), N_DESIGNS=str(len(ALS)),
    MAT_ROWS=mat_rows, SLOT_ROWS=slot_rows, WIRE_ROWS=wire_rows, PICK_ROWS=pick_rows, FRONT_ROWS=front_rows, OP_ROWS=op_rows, MASS_ROWS=mass_rows,
    SPEC_ROWS=spec_table(['Copper baseline', 'Aluminium drop-in', 'Recommended']),
    **{'IMG_' + k.upper().replace(' ', '_').replace(',', '').replace('-', '_'): v for k, v in img.items()})

# ---------------- narrative text (data-driven) ----------------
CU20 = BY.get('Cu SWG 11 S0 NO20')
fop = FULL['Recommended']['ops']; cop = FULL['Copper baseline']['ops']
ANSWER = (
    '<div class="note"><b>Short answer</b><div>Changing only the conductor (same slot, same wire, same turns) keeps the torque but not the efficiency: '
    'peak efficiency drops from %s %% to %s %%, and OP3 (2000 rpm, 44 N·m) loses %.2f points. It saves %s kg. '
    'Getting the same torque <em>and</em> the same efficiency with aluminium needs four changes together: '
    '<b>(1)</b> a bigger slot (%s: depth %.1f mm, tooth %.1f mm, %.0f mm² instead of %.0f mm²), '
    '<b>(2)</b> a longer stack with fewer turns (%.0f mm with %d turns/coil instead of 63 mm with 10), '
    '<b>(3)</b> each turn split into %d parallel strands of %.2f mm, and '
    '<b>(4)</b> thinner 0.20 mm lamination steel to pay for the extra iron loss of the longer stack. '
    'The result matches copper within 0.1 points at all three operating points and over the whole map (%s, %s, %s; map %s), '
    'but it weighs <b>%s kg more</b> than copper (%s vs %s kg active). '
    'Aluminium saves %s kg of winding (%s → %s kg), but the longer stack adds more iron and magnet than that.</div></div>'
    ) % (V0['CU_PEAK'], V0['DROP_PEAK'], -DROP['dEta_ops']['OP3'], '%.2f' % -DROP['dmass'],
         FINAL['sv'], A.SLOTS.geo(FINAL['sv'])['slot_depth'], A.SLOTS.geo(FINAL['sv'])['tooth_w'], A.slot_area(FINAL['sv']), A.slot_area('S0'),
         FINAL['L'], FINAL['Nph'] // 2, FINAL['ns'], FINAL['d'],
         sg(FINAL['dEta_ops']['OP1']), sg(FINAL['dEta_ops']['OP2']), sg(FINAL['dEta_ops']['OP3']), sg(FINAL['map_dEta_mean']),
         '%.2f' % FINAL['dmass'], '%.2f' % FINAL['masses']['total'], '%.2f' % CU['masses']['total'],
         '%.2f' % (CU['masses']['winding'] - FINAL['masses']['winding']), '%.2f' % CU['masses']['winding'], '%.2f' % FINAL['masses']['winding'])
if CU20:
    ANSWER += ('<p class="muted">Fair comparison: the 0.20 mm steel alone would also raise the <em>copper</em> design by %s points on the map mean. '
               'If copper gets the thin steel too, aluminium at equal efficiency needs a longer stack again. Treat the steel change as part of the aluminium package, not a free gain.</p>') % sg(CU20['map_dEta_mean'])
ANSWER += '<p>The table compares the copper design, the straight swap, the best options under other constraints, and the recommended design.</p>'

nm235 = [r for r in ALS if lamof(r) == 'M235-35A' and r['env_ratio_min'] >= 0.985]
b235 = max(nm235, key=lambda r: worst(r)) if nm235 else None
TRADE = '<ul>'
TRADE += '<li><b>With the present M235-35A steel, aluminium cannot match copper at every operating point.</b> Any stack long enough to recover OP3 adds iron loss that pulls OP2 (9339 rpm, light load) down. The best balance found is %s: worst operating point %s points, map %s, %+.2f kg.</li>' % (
    b235['label'], sg(worst(b235)), sg(b235['map_dEta_mean']), b235['dmass']) if b235 else ''
if LIGHT:
    TRADE += '<li><b>No heavier than copper:</b> the best design at or below the copper mass is %s, at %.2f kg. Worst operating point %s points, map %s. It is a little lighter but clearly less efficient.</li>' % (
        LIGHT['label'], LIGHT['masses']['total'], sg(worst(LIGHT)), sg(LIGHT['map_dEta_mean']))
TRADE += '<li><b>Equal efficiency:</b> the lightest design that matches copper everywhere is %.2f kg, %+.2f kg. ' % (FINAL['masses']['total'], FINAL['dmass'])
TRADE += 'Each extra 3 mm of stack costs about 0.3 kg. Going longer than the recommended stack buys OP3 efficiency and loses OP2 (see the stack sweep in section 7).</li>'
TRADE += '<li><b>Summary:</b> aluminium makes the <em>winding</em> lighter (%.2f → %.2f kg). At equal active mass it costs efficiency, and at equal efficiency it costs mass. Its gains are lower conductor cost and lower AC loss at high speed, not weight.</li></ul>' % (CU['masses']['winding'], FINAL['masses']['winding'])

DESIGN = ('<p>%s: aluminium EC-1350 winding, slot %s (depth %.1f mm, tooth %.1f mm, yoke %.1f mm), %s, stack %.0f mm, %d turns/coil '
          '(%d series turns/phase), delta with 2 parallel paths, %d × %.3f mm strands in hand. Phase resistance at 100 °C is %s mΩ, against %s mΩ for copper. '
          'Losses at OP3: winding DC %.0f W against %.0f W for copper, iron %.0f W against %.0f W. At OP1 the strands cut AC loss from %.0f W to %.0f W. '
          'Peak efficiency %s %% (copper %s %%).</p>') % (
    'Recommended design', FINAL['sv'], A.SLOTS.geo(FINAL['sv'])['slot_depth'], A.SLOTS.geo(FINAL['sv'])['tooth_w'], 80 - 53.5 - A.SLOTS.geo(FINAL['sv'])['slot_depth'],
    A.LAMS[lamof(FINAL)]['name'], FINAL['L'], FINAL['Nph'] // 2, FINAL['Nph'], FINAL['ns'], FINAL['d'], V0['RPH_FIN'], V0['RPH_CU'],
    fop['OP3']['pcu'], cop['OP3']['pcu'], fop['OP3']['pfe'], cop['OP3']['pfe'], cop['OP1']['pac'], fop['OP1']['pac'], V0['FINAL_PEAK'], V0['CU_PEAK'])
V.update(ANSWER=ANSWER, TRADE_TEXT=TRADE, DESIGN_TEXT=DESIGN)

# ---------------- cost section ----------------
cost_rows = ''
c0 = COST(CU)['total']
for n, lab, st in AC.DESIGNS:
    r = BY[lab]; c = COST(r); o = r['dEta_ops']; base = n.startswith('Cu, M235')
    cells = ('—',) * 4 if base else (pill(o['OP1']), pill(o['OP2']), pill(o['OP3']), pill(r['map_dEta_mean'], -0.15, -0.6))
    cost_rows += '<tr%s><td class="txt">%s</td><td>%.1f</td><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td><b>%.2f</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % ((
        ' style="font-weight:600"' if lab == FINAL['label'] else '', n, r['L'], 'M235-35A' if st == 'M235-35A' else '0.20 mm', r['masses']['total'], c['wire'], c['steel'], c['magnet'], c['total'],
        '—' if base else '%+.2f' % (c['total'] - c0)) + cells)
P = AC.PRICE; RF, RM = BY['Al S3 L81.0 N16 3x2.05 NO20'], BY['Al S3 L81.0 N16 3x2.05']
dd = lambda r, Q: COST(r, Q)['total'] - COST(CU, Q)['total']
sens_rows = ''
for desc, Q in [('As assumed', P), ('Copper wire 12 $/kg', dict(P, Cu=12.0)), ('Copper wire 20 $/kg', dict(P, Cu=20.0)), ('0.20 mm steel 3.0 $/kg', dict(P, NO20=3.0)),
                ('0.20 mm steel 5.0 $/kg', dict(P, NO20=5.0)), ('Aluminium wire 7.0 $/kg', dict(P, Al=7.0))]:
    sens_rows += '<tr><td class="txt">%s</td><td>%+.2f</td><td>%s</td></tr>' % (desc, dd(RF, Q), '—' if 'steel' in desc else '%+.2f' % dd(RM, Q))
gross = AC.cost(RF['label'], 'NO20')[1]
be = P['NO20'] - dd(RF, P) / gross
cu20 = BY['Cu SWG 11 S0 NO20']
COST_TEXT = ('<div class="note"><b>Cost answer</b><div>With the present <b>M235-35A</b> steel, aluminium saves about <b>%.0f $ per motor</b> (%.0f %% of active material cost) '
             'but costs about %.1f points of map-mean efficiency and up to %.1f points at an operating point. With <b>0.20 mm steel</b>, aluminium reaches copper efficiency but costs '
             '<b>%+.0f $ per motor</b>: the winding saves %.0f $, and the longer stack of thin steel adds %.0f $. If the copper motor also gets the thin steel, it gains %+.2f points and '
             'aluminium stays behind it. Equal efficiency is therefore cheaper with copper unless 0.20 mm steel costs no more than about %.1f $/kg.</div></div>') % (
    -dd(RM, P), -100 * dd(RM, P) / c0, -RM['map_dEta_mean'], -min(RM['dEta_ops'].values()), dd(RF, P),
    COST(CU)['wire'] - COST(RF)['wire'], COST(RF)['steel'] - COST(CU)['steel'], cu20['map_dEta_mean'], be)
BREAK = "<p>Break-even: the aluminium design with 0.20 mm steel costs the same as today's copper motor if 0.20 mm steel costs about <b>%.1f $/kg</b> (%.1f × M235-35A). The longer stack needs %.1f kg of sheet per motor against %.1f kg today.</p>" % (
    be, be / P['M235-35A'], gross, AC.cost(CU['label'], 'M235-35A')[1])
V.update(COST_TEXT=COST_TEXT, COST_ROWS=cost_rows, SENS_ROWS=sens_rows, BREAKEVEN=BREAK)
tpl = open(os.path.join(H, 'alu_template.html'), encoding='utf8').read()
for k, v in V.items():
    tpl = tpl.replace('__%s__' % k, v)
left = [l for l in tpl.split('\n') if '__' in l and 'data:image' not in l]
assert not left, left[:3]
open(os.path.join(H, 'alu_report.html'), 'w', encoding='utf8').write(tpl)
json.dump(dict(picks={n: r['label'] for n, r in PICKS}, img_keys=list(img)), open(os.path.join(H, 'alu_report_picks.json'), 'w'), indent=1)
print('alu_report.html', len(tpl) // 1024, 'kB', list(img))
