"""Sensitivity study report -> sens_report.html (from sens_results.json + design IDs)."""
import base64, io, json, math, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sens as S
from sens_models import design_ids

H = S.H
R = json.load(open(os.path.join(H, 'sens_results.json')))
D, C, SK = R['designs'], R.get('candidates', {}), R['skew']
IDS = design_ids()
base = D['baseline']
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .3})
OUT = [('T360', 'Torque at 360 A, 2000 rpm', 'N·m', +1), ('loss', 'Loss at 9339 rpm, 3.24 N·m', 'W', -1),
       ('thd', 'Back-EMF THD', '%', -1), ('rip', 'Ripple at 9000 rpm (worst full-load point)', '%', -1)]


def val(r, k):
    return {'T360': r['T360'], 'loss': (r['op2'] or {}).get('loss', float('nan')), 'thd': r['thd'], 'rip': r['rip_op1_pct']}[k]


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=110, bbox_inches='tight', facecolor='white'); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


params = []
for v, d in IDS.items():
    if d['param'] and d['value'] is not None and d['param'] not in params:
        params.append(d['param'])
LBL = {'slot_open': 'slot opening', 'tooth_w': 'tooth width', 'tip_depth': 'tooth tip depth', 'magnet_opening': 'magnet opening (mouth)',
       'magnet_Y': 'Magnet_Y (bulge half-width)', 'magnet_Z': 'Magnet_Z (top half-width)', 'magnet_inset': 'magnet inset', 'magnet_depth': 'magnet depth',
       'Bridge_thickness': 'bridge thickness', 'bridge_offset': 'bridge offset'}
eff = {}   # param -> output -> (low_val, low_delta, high_val, high_delta)
for p in params:
    pts = sorted([(d['value'], v) for v, d in IDS.items() if d['param'] == p])
    eff[p] = {}
    for k, *_ in OUT:
        row = []
        for x, v in pts:
            r = D.get(v)
            row.append((x, val(r, k) - val(base, k) if r else float('nan')))
        eff[p][k] = row

img = {}
# tornado per output
fig, axs = plt.subplots(1, 4, figsize=(18, 5.6), sharey=True)
order = sorted(params, key=lambda p: -np.nanmax([abs(e[1]) / abs(val(base, k)) for k, *_ in OUT for e in eff[p][k]]))
y = np.arange(len(order))
for ax, (k, t, u, sgn) in zip(axs, OUT):
    lo = [eff[p][k][0][1] for p in order]; hi = [eff[p][k][-1][1] for p in order]
    ax.barh(y + .2, lo, .38, color='#74c0fc', label='low value'); ax.barh(y - .2, hi, .38, color='#1c7ed6', label='high value')
    ax.axvline(0, color='k', lw=.8)
    ax.set(title='%s\n(Δ vs D00, %s; %s is better)' % (t, u, 'right' if sgn > 0 else 'left'), xlabel=u)
axs[0].set_yticks(y, ['%s\n%g / %g mm' % (LBL[p], eff[p]['T360'][0][0], eff[p]['T360'][-1][0]) for p in order], fontsize=8)
axs[0].invert_yaxis(); axs[0].legend(fontsize=8, loc='lower right')
img['tornado'] = png(fig)

# influence matrix: change per +1 mm (central/one-sided slope), as % of baseline (THD, ripple: points)
mat = np.zeros((len(order), 4))
for i, p in enumerate(order):
    for j, (k, *_ ) in enumerate(OUT):
        (x0, d0), (x1, d1) = eff[p][k][0], eff[p][k][-1]
        s = (d1 - d0) / (x1 - x0)
        mat[i, j] = 100 * s / val(base, k) if k in ('T360', 'loss') else s
fig, ax = plt.subplots(figsize=(8.5, 6.2))
good = mat * np.array([1, -1, -1, -1])
lim = np.nanmax(np.abs(good))
im = ax.imshow(good, cmap='RdYlGn', vmin=-lim, vmax=lim, aspect='auto')
for i in range(len(order)):
    for j in range(4):
        ax.text(j, i, ('%+.1f %%' if j < 2 else '%+.2f pt') % mat[i, j], ha='center', va='center', fontsize=8)
ax.set_xticks(range(4), ['torque\n(% per mm)', 'OP2 loss\n(% per mm)', 'THD\n(pt per mm)', 'ripple OP1\n(pt per mm)'])
ax.set_yticks(range(len(order)), [LBL[p] for p in order], fontsize=9)
ax.set_title('Influence of +1 mm on each output (green = beneficial)')
ax.grid(False)
img['matrix'] = png(fig)

# trade-off scatter
fig, ax = plt.subplots(figsize=(9, 6))
for v, r in D.items():
    if not r['op2']:
        continue
    ok = (r['thd'] < 5) + (r['rip_op1_pct'] < 10)
    c = ['#e03131', '#f59f00', '#2f9e44'][ok]
    mk = 's' if v == 'baseline' else ('D' if v == 'MotorCAD_original' else 'o')
    ax.scatter(r['op2']['loss'], r['T360'], s=70 if v in ('baseline', 'MotorCAD_original') else 40, c=c, marker=mk, edgecolors='k', lw=.5, zorder=3)
    ax.annotate(IDS[v]['id'], (r['op2']['loss'], r['T360']), xytext=(4, 3), textcoords='offset points', fontsize=7)
for v, r in C.items():
    if r['op2']:
        ok = (r['thd'] < 5) + (r['rip_op1_pct'] < 10)
        ax.scatter(r['op2']['loss'], r['T360'], s=160, c=['#e03131', '#f59f00', '#2f9e44'][ok], marker='*', edgecolors='k', lw=.6, zorder=4)
        ax.annotate(v, (r['op2']['loss'], r['T360']), xytext=(6, -10), textcoords='offset points', fontsize=9, weight='bold')
ax.set(xlabel='loss at 9339 rpm, 3.24 N·m (W)  ← better', ylabel='torque at 360 A, 2000 rpm (N·m)  better →',
       title='Trade-off between the two objectives (colour = requirements met: red 0, amber 1, green 2)')
img['scatter'] = png(fig)

# skew curves
fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
sk = sorted((float(s), r) for s, r in SK.items())
xs = [s for s, _ in sk]
axs[0].plot(xs, [r['thd_cont'] for _, r in sk], 'k-', label='THD, D00 (continuous skew)')
for nm, cl in (('OP1', '#1c7ed6'), ('OP2', '#e8590c'), ('OP3', '#2f9e44')):
    axs[1].plot(xs, [r['%s_cont_pct' % nm] for _, r in sk], '-', color=cl, label='%s continuous' % nm)
    axs[1].plot(xs, [r['%s_3step_pct' % nm] for _, r in sk], '--', color=cl, label='%s 3-step' % nm)
for v, r in C.items():
    s2 = sorted((float(s), q) for s, q in r['skew'].items())
    axs[0].plot([s for s, _ in s2], [q['thd_cont'] for _, q in s2], ':', lw=2, label='THD, %s' % v)
axs[0].axhline(5, color='r', lw=1); axs[1].axhline(10, color='r', lw=1)
axs[0].set(xlabel='rotor skew (mech deg)', ylabel='%', title='Back-EMF THD vs skew (limit 5 %)'); axs[0].legend(fontsize=8)
axs[1].set(xlabel='rotor skew (mech deg)', ylabel='% of mean torque', title='Torque ripple vs skew, D00 waveforms (limit 10 %)', ylim=(0, 100)); axs[1].legend(fontsize=7, ncol=2)
img['skew'] = png(fig)
# candidate skew ripple
if C:
    fig, axs = plt.subplots(1, len(C), figsize=(7.5 * len(C), 4.2), squeeze=False)
    for ax, (v, r) in zip(axs[0], C.items()):
        s2 = sorted((float(s), q) for s, q in r['skew'].items())
        for nm, cl in (('OP1', '#1c7ed6'), ('OP2', '#e8590c'), ('OP3', '#2f9e44')):
            ax.plot([s for s, _ in s2], [q['%s_cont_pct' % nm] for _, q in s2], '-', color=cl, label=nm + ' continuous')
            ax.plot([s for s, _ in s2], [q['%s_3step_pct' % nm] for _, q in s2], '--', color=cl, label=nm + ' 3-step')
        ax.axhline(10, color='r', lw=1); ax.set(title='%s: ripple vs skew' % v, xlabel='skew (mech deg)', ylabel='%', ylim=(0, 100)); ax.legend(fontsize=7, ncol=2)
    img['candskew'] = png(fig)


def fmt(x, n=2):
    return '–' if x is None or (isinstance(x, float) and math.isnan(x)) else ('%.*f' % (n, x))


def chk(x, lim):
    return '<span class="pill %s">%s</span>' % ('ok' if x < lim else 'bad', fmt(x))


rows = ''
for v, d in IDS.items():
    r = D.get(v)
    if not r:
        rows += '<tr><td>%s</td><td class="txt">%s</td><td colspan="8">pending</td></tr>' % (d['id'], v); continue
    o2 = r['op2'] or {}
    rows += '<tr%s><td>%s</td><td class="txt">%s</td><td>%s</td><td>%+.1f %%</td><td>%s</td><td>%+.1f %%</td><td>%s</td><td>%s</td><td>%s</td><td class="src">%s</td></tr>' % (
        ' class="base"' if v == 'baseline' else '', d['id'], 'baseline (shaft 30, bridge 1.0)' if v == 'baseline' else ('reference: Motor-CAD original (shaft 25)' if v == 'MotorCAD_original' else '%s = %g mm' % (LBL[d['param']], d['value'])),
        fmt(r['T360']), 100 * (r['T360'] / base['T360'] - 1), fmt(o2.get('loss'), 1), 100 * (o2.get('loss', float('nan')) / base['op2']['loss'] - 1),
        chk(r['thd'], 5), chk(r['rip_op1_pct'], 10), fmt(r['E1_rms'], 3), d['file'])
crow = ''
for v, r in C.items():
    o2 = r['op2'] or {}
    s6 = r['skew'].get('6.0') or r['skew'].get(6.0)
    chg = ', '.join('%s %g' % (LBL[k], x) for k, x in list(r['stator'].items()) + list(r['rotor'].items()))
    crow += '<tr><td><b>%s</b></td><td class="txt">%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s / %s / %s</td></tr>' % (
        v, chg, fmt(r['T360']), fmt(o2.get('loss'), 1), chk(r['thd'], 5), chk(r['rip_op1_pct'], 10), chk(r['rip_op2_pct'], 10), chk(r['rip_op3_pct'], 10),
        chk(s6['OP1_3step_pct'], 10), chk(s6['OP2_3step_pct'], 10), chk(s6['OP3_3step_pct'], 10))
tpl = open(os.path.join(H, 'sens_template.html'), encoding='utf8').read()
for k, v in img.items():
    tpl = tpl.replace('{{IMG_%s}}' % k.upper(), v)
tpl = tpl.replace('{{IMG_CANDSKEW}}', '').replace('{{ROWS}}', rows).replace('{{CROWS}}', crow or '<tr><td colspan="9">candidates pending</td></tr>')
r0 = D.get('MotorCAD_original')
tpl = tpl.replace('{{R00}}', ('R00 (shaft 25 mm): torque %s N·m, loss %s W, THD %s %%, ripple %s %%' % (fmt(r0['T360']), fmt(r0['op2']['loss'], 1), fmt(r0['thd']), fmt(r0['rip_op1_pct'], 1))) if r0 else 'R00 pending')
assert '{{' not in tpl, [l for l in tpl.split('\n') if '{{' in l][:3]
open(os.path.join(H, 'sens_report.html'), 'w', encoding='utf8').write(tpl)
print('sens_report.html %d kB' % (len(tpl) // 1024))
