"""Motor-CAD vs FEMM Lab consistency report (100 C) -> consistency_report.html"""
import base64, io, json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, 'mcad_lab'))
from load_mcad import M as MC, E as ME
import lab_model as LM

v1 = json.load(open(os.path.join(H, 'lab_compare_v1.json')))
v2 = json.load(open(os.path.join(H, 'lab_compare_v2.json')))
fit = json.load(open(os.path.join(H, 'iron_variants.json')))
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .3})
CM, C1, C2 = '#d9480f', '#868e96', '#1c7ed6'


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=110, bbox_inches='tight', facecolor='white'); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


A = lambda d, q: np.array([np.nan if x is None else x for x in d['envelope'][q]], float)
n_e = ME['Speed']
img = {}
# ---- envelope overlays ----
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for a, q, yl in ((ax[0, 0], 'Shaft_Torque', 'shaft torque (N·m)'), (ax[0, 1], 'Efficiency', 'efficiency (%)'),
                 (ax[1, 0], 'Iron_Loss', 'iron loss (W)'), (ax[1, 1], 'Total_Loss', 'total loss (W)')):
    a.plot(n_e, ME[q], 'o-', color=CM, lw=2.2, ms=4, label='Motor-CAD')
    a.plot(n_e, A(v1, q), '--', color=C1, lw=1.6, label='FEMM before (v1)')
    a.plot(n_e, A(v2, q), '-', color=C2, lw=2, label='FEMM now (v2)')
    a.set(xlabel='speed (rpm)', ylabel=yl, xlim=(0, 9000)); a.legend(fontsize=8)
ax[0, 1].set_ylim(80, 96)
fig.suptitle('Envelope at 100 °C: 52 V, Mi 0.95, 360 A rms, 280 A DC, MTPA', y=1.0)
img['env'] = png(fig)
# loss breakdown along the envelope
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharex=True)
for a, q, t in zip(ax, ('Stator_Copper_Loss_DC', 'Stator_Copper_Loss_AC', 'Iron_Loss_Stator', 'Iron_Loss_Rotor'),
                   ('copper DC', 'copper AC', 'iron, stator', 'iron, rotor')):
    a.plot(n_e, ME[q], 'o-', color=CM, ms=3, label='Motor-CAD'); a.plot(n_e, A(v1, q), '--', color=C1, label='FEMM v1'); a.plot(n_e, A(v2, q), color=C2, label='FEMM v2')
    a.set(title=t, xlabel='rpm', ylabel='W', xlim=(0, 9000))
ax[0].legend(fontsize=8)
img['envloss'] = png(fig)

# ---- efficiency maps on Motor-CAD's own points ----
S, T = MC['Speed'], MC['Shaft_Torque']
mask = (S > 0) & (T > 0.5)
e_m = MC['Efficiency']; e_f = np.array([[np.nan if x is None else x for x in r] for r in v2['A']['Efficiency']], float)
e_1 = np.array([[np.nan if x is None else x for x in r] for r in v1['A']['Efficiency']], float)
ok = mask & ~np.isnan(e_f)
tri = mtri.Triangulation(S[ok], T[ok])
lev = [80, 84, 86, 88, 90, 91, 92, 93, 94, 95, 96]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for a, z, t in ((ax[0], e_m[ok], 'Motor-CAD'), (ax[1], e_f[ok], 'FEMM now (v2)'), (ax[2], e_1[ok], 'FEMM before (v1)')):
    cs = a.tricontourf(tri, z, levels=lev, cmap='RdYlGn', extend='both')
    a.clabel(a.tricontour(tri, z, levels=[88, 90, 92, 93, 94, 95], colors='k', linewidths=.6), fmt='%d', fontsize=8)
    a.plot(n_e, ME['Shaft_Torque'], 'k-', lw=1.5)
    a.set(title='Efficiency, ' + t, xlabel='speed (rpm)', ylabel='shaft torque (N·m)', xlim=(0, 9000), ylim=(0, 46))
fig.colorbar(cs, ax=ax, label='%', shrink=.9)
img['effmaps'] = png(fig)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for a, z, t in ((ax[0], e_1[ok] - e_m[ok], 'before (v1)'), (ax[1], e_f[ok] - e_m[ok], 'now (v2)')):
    cs = a.tricontourf(tri, z, levels=np.arange(-3, 3.01, 0.5), cmap='RdBu_r', extend='both')
    a.plot(n_e, ME['Shaft_Torque'], 'k-', lw=1.2)
    a.set(title='Efficiency FEMM − Motor-CAD, ' + t, xlabel='speed (rpm)', ylabel='shaft torque (N·m)', xlim=(0, 9000), ylim=(0, 46))
    fig.colorbar(cs, ax=a, label='percentage points')
img['effdiff'] = png(fig)

# ---- parity plots, same currents (B) ----
Bv = lambda d, q: np.array([[np.nan if x is None else x for x in r] for r in d['B'][q]], float)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, q, t in zip(ax, ('Iron_Loss_Stator', 'Iron_Loss_Rotor', 'Stator_Copper_Loss_AC', 'Electromagnetic_Torque'),
                   ('iron, stator (W)', 'iron, rotor (W)', 'copper AC (W)', 'EM torque (N·m)')):
    x = MC[q][mask]; a.plot(x, Bv(v1, q)[mask], '.', color=C1, ms=3, label='FEMM v1'); a.plot(x, Bv(v2, q)[mask], '.', color=C2, ms=3, label='FEMM v2')
    lim = [0, np.nanmax(x) * 1.15]; a.plot(lim, lim, 'k-', lw=.8)
    a.set(title=t, xlabel='Motor-CAD', ylabel='FEMM', xlim=lim, ylim=lim)
ax[0].legend(fontsize=8, markerscale=3)
img['parity'] = png(fig)

# ---- iron-loss formulation test ----
names = {'all': 'every harmonic (stator + rotor)', 'fund': 'fundamental only', 'peak': 'peak-B (no harmonics)'}
form_rows = ''.join('<tr><td class="txt">%s</td>%s</tr>' % (names[v], ''.join('<td>%+.0f %%</td>' % fit['errors'][v][k]['mean_pct']
                    for k in ('stator', 'rotor', 'hysteresis', 'eddy', 'total'))) for v in ('all', 'fund', 'peak'))

# ---- error table ----
QN = [('Shaft_Torque', 'Shaft torque'), ('Electromagnetic_Torque', 'EM torque'), ('Efficiency', 'Efficiency (abs. points)'),
      ('Stator_Current_Line_RMS', 'Line current'), ('Voltage_Phase_Peak', 'Phase voltage'), ('Power_Factor', 'Power factor'),
      ('Flux_Linkage_D', 'Flux linkage d'), ('Flux_Linkage_Q', 'Flux linkage q'), ('Stator_Copper_Loss_DC', 'Copper loss DC'),
      ('Stator_Copper_Loss_AC', 'Copper loss AC'), ('Iron_Loss_Stator', 'Iron loss stator'), ('Iron_Loss_Rotor', 'Iron loss rotor'),
      ('Iron_Loss_Hysteresis', 'Iron loss hysteresis'), ('Iron_Loss_Eddy', 'Iron loss eddy'), ('Mechanical_Loss', 'Mechanical loss'),
      ('Total_Loss', 'Total loss'), ('Idc', 'DC current')]


def cell(s, q):
    if not s:
        return '<td>–</td>'
    if q == 'Efficiency':
        v = s['mean']; cls = 'ok' if abs(v) <= 0.75 else ('mid' if abs(v) <= 1.5 else 'bad')
        return '<td><span class="pill %s">%+.2f pt</span></td>' % (cls, v)
    v = s['mean_abs_rel_pct']; cls = 'ok' if v <= 5 else ('mid' if v <= 15 else 'bad')
    return '<td><span class="pill %s">%.1f %%</span> <span class="src">(bias %+.1f %%)</span></td>' % (cls, v, s['mean_rel_pct'])


err_rows = ''.join('<tr><td class="txt">%s</td>%s%s%s</tr>' % (lab, cell(v1['summary']['A_same_point'][q], q), cell(v2['summary']['A_same_point'][q], q),
                   cell(v2['summary']['B_same_current'][q], q)) for q, lab in QN)


def env_tbl():
    rows = ''
    for k, n in enumerate(n_e):
        if n % 1000:
            continue
        g = lambda d, q: d['envelope'][q][k]
        rows += '<tr><td>%d</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.0f</td><td>%.0f</td><td>%.0f</td></tr>' % (
            n, ME['Shaft_Torque'][k], g(v2, 'Shaft_Torque'), g(v1, 'Efficiency') or 0, ME['Efficiency'][k], g(v2, 'Efficiency') or 0,
            g(v2, 'Efficiency') - ME['Efficiency'][k] if n else 0, ME['Iron_Loss'][k], g(v1, 'Iron_Loss'), g(v2, 'Iron_Loss'))
    return rows


pk_m = np.nanmax(np.where(mask, e_m, np.nan)); pk_f = np.nanmax(np.where(ok, e_f, np.nan)); pk_1 = np.nanmax(np.where(ok, e_1, np.nan))
sa1, sa2 = v1['summary']['A_same_point'], v2['summary']['A_same_point']
rep = dict(ENV_ROWS=env_tbl(), ERR_ROWS=err_rows, FORM_ROWS=form_rows, PK_M='%.2f' % pk_m, PK_F='%.2f' % pk_f, PK_1='%.2f' % pk_1,
           EFF1='%+.2f' % sa1['Efficiency']['mean'], EFF2='%+.2f' % sa2['Efficiency']['mean'],
           FE1='%+.0f' % sa1['Iron_Loss']['mean_rel_pct'], FE2='%+.0f' % sa2['Iron_Loss']['mean_rel_pct'],
           TL1='%+.0f' % sa1['Total_Loss']['mean_rel_pct'], TL2='%+.0f' % sa2['Total_Loss']['mean_rel_pct'], NPTS='%d' % sa2['Efficiency']['n'])
tpl = open(os.path.join(H, 'consistency_template.html'), encoding='utf8').read()
for k, v in img.items():
    tpl = tpl.replace('{{IMG_%s}}' % k.upper(), v)
for k, v in rep.items():
    tpl = tpl.replace('{{%s}}' % k, v)
assert '{{' not in tpl, [l for l in tpl.split('\n') if '{{' in l][:3]
open(os.path.join(H, 'consistency_report.html'), 'w', encoding='utf8').write(tpl)
print(rep['PK_M'], rep['PK_F'], rep['PK_1'], rep['EFF1'], rep['EFF2'], rep['FE1'], rep['FE2'], rep['TL1'], rep['TL2'], rep['NPTS'], '%d kB' % (len(tpl) // 1024))
