"""Detailed FEMM vs Motor-CAD Lab report for Design9795_1 (100 C): report_lab/lab_validation.html

Inputs: lab_compare_9795.json (compare9795.py), lab_100C.json (make_lab_v2.py), dq_100C.json, sens_q.json, oc_validate.json,
work/sf097/lab_compare_9795.json (first build, FEA stacking 0.97, for the refinement history), mcad_lab/mcad_lab.json."""
import base64, io, json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, 'mcad_lab'))
from load_mcad import E as MCE, M as MCM
import lab_model as LM

C = json.load(open(os.path.join(H, 'lab_compare_9795.json')))
LAB = json.load(open(os.path.join(H, 'lab_100C.json')))
SENS = json.load(open(os.path.join(H, 'sens_q.json')))
OC = json.load(open(os.path.join(H, 'oc_validate.json')))
OLD = json.load(open(os.path.join(H, 'work', 'sf097', 'lab_compare_9795.json')))
V2L = json.load(open(os.path.join(H, 'work', 'lab_compare_v2loss.json')))   # refined fields, v2 iron-loss formulation
IV = json.load(open(os.path.join(H, 'iron_variants.json')))
EQ, D2 = C['mcad_eq'], C['2d']

COL = dict(mc='#1f2937', eq='#2563eb', d2='#d97706')     # Motor-CAD ink, FEMM Motor-CAD-equivalent blue, FEMM 2D amber
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 2, 'figure.constrained_layout.use': True})


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', bbox_inches='tight', facecolor='white'); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


# ---------------- reference point: Motor-CAD full current at 1000 rpm ----------------
ref_i = (-84.98, 281.39)
REF = {}
for name, kw in (('2D, air-gap force torque', {}), ('Motor-CAD equivalent', dict(slices=(-2, 0, 2), torque='flux'))):
    r = LM.Model('v3', **kw).evaluate(1000, np.array([ref_i[0] / 2]), np.array([ref_i[1] / 2]))
    REF[name] = {k: float(v[0]) for k, v in r.items()}
mk = np.flatnonzero((MCM['Speed'] == 1000).ravel())
mi = mk[np.argmax(np.hypot(MCM['Id_Peak'].ravel()[mk], MCM['Iq_Peak'].ravel()[mk]))]
REF_MC = {q: float(MCM[q].ravel()[mi]) for q in ('Flux_Linkage_D', 'Flux_Linkage_Q', 'Electromagnetic_Torque', 'Iron_Loss', 'Shaft_Torque')}

# ---------------- figures ----------------
spd = np.array(C['mcad_envelope']['Speed'])
fig, ax = plt.subplots(2, 3, figsize=(12, 6.6))
for a, (q, lab, sc) in zip(ax.ravel(), (('Shaft_Torque', 'Shaft torque (N·m)', 1), ('Shaft_Power', 'Shaft power (kW)', 1e-3),
                                        ('Efficiency', 'Efficiency (%)', 1), ('Stator_Current_Line_RMS', 'Line current (A rms)', 1),
                                        ('Voltage_Phase_Peak', 'Phase voltage (V peak)', 1), ('Phase_Advance', 'Phase advance (deg)', 1))):
    mc = np.array(C['mcad_envelope'][q], float) * sc
    ok = spd > 0 if q == 'Efficiency' else spd >= 0
    a.plot(spd[ok], mc[ok], '-', color=COL['mc'], label='Motor-CAD')
    a.plot(spd[ok], arr(EQ['envelope'][q])[ok] * sc, '--', color=COL['eq'], label='FEMM, Motor-CAD equivalent')
    a.plot(spd[ok], arr(D2['envelope'][q])[ok] * sc, ':', color=COL['d2'], label='FEMM 2D')
    a.set_title(lab, fontsize=10, loc='left'); a.set_xlabel('Speed (rpm)')
    if q == 'Efficiency':
        a.set_ylim(80, 96)
ax[0, 0].legend(fontsize=8.5, loc='lower left')
F_ENV = png(fig)

# efficiency maps: Motor-CAD (its map points) and FEMM Motor-CAD equivalent (lab_100C.json), same levels, 0-9500 rpm
S, T = np.array(LAB['speeds']), np.array(LAB['torques']); EFF = np.array(LAB['maps']['eff'], float)
lv = [80, 84, 86, 88, 90, 91, 92, 93, 94, 94.5, 95, 95.5]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
x, y, z = MCM['Speed'].ravel(), MCM['Shaft_Torque'].ravel(), MCM['Efficiency'].ravel()
ok = (x > 0) & (y > 0.5) & np.isfinite(z)
cs = ax[0].tricontourf(x[ok], y[ok], z[ok], levels=lv, cmap='viridis', extend='both')
ax[0].clabel(ax[0].tricontour(x[ok], y[ok], z[ok], levels=lv, colors='k', linewidths=.5), fontsize=7, fmt='%g')
ax[0].set_title('Motor-CAD Lab efficiency (%)', loc='left', fontsize=10)
sm = S <= 9500
ax[1].contourf(S[sm], T, EFF[:, sm], levels=lv, cmap='viridis', extend='both')
ax[1].clabel(ax[1].contour(S[sm], T, EFF[:, sm], levels=lv, colors='k', linewidths=.5), fontsize=7, fmt='%g')
ax[1].set_title('FEMM Lab efficiency (%), Motor-CAD equivalent', loc='left', fontsize=10)
for a in ax[:2]:
    a.plot(spd, MCE['Shaft_Torque'], '-', color=COL['mc'], lw=1.6); a.plot(spd, arr(EQ['envelope']['Shaft_Torque']), '--', color='white', lw=1.4)
    a.set_xlabel('Speed (rpm)'); a.set_ylabel('Shaft torque (N·m)'); a.set_ylim(0, 47); a.grid(False)
P = EQ['map']['points']; de = arr(EQ['map']['A']['Efficiency']) - np.array(P['Efficiency'])
sc_ = ax[2].scatter(P['Speed'], P['Shaft_Torque'], c=de, s=9, cmap='RdBu_r', norm=TwoSlopeNorm(0, -1, 1))
fig.colorbar(sc_, ax=ax[2], label='FEMM − Motor-CAD (points)')
ax[2].set_title('Efficiency difference at Motor-CAD map points', loc='left', fontsize=10); ax[2].set_xlabel('Speed (rpm)'); ax[2].set_ylim(0, 47)
fig.colorbar(cs, ax=ax[:2], shrink=.9, label='Efficiency (%)')
F_MAP = png(fig)

# parity plots, same (speed, torque), map points
fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))
for a, (q, lab) in zip(ax.ravel(), (('Stator_Copper_Loss_DC', 'DC copper loss (W)'), ('Stator_Copper_Loss_AC', 'AC copper loss (W)'),
                                    ('Iron_Loss_Stator', 'Stator iron loss (W)'), ('Iron_Loss_Rotor', 'Rotor iron loss (W)'),
                                    ('Total_Loss', 'Total loss (W)'), ('Efficiency', 'Efficiency (%)'), ('Stator_Current_Line_RMS', 'Line current (A rms)'),
                                    ('Voltage_Phase_Peak', 'Phase voltage (V peak)'))):
    m = np.array(P[q]); f = arr(EQ['map']['A'][q]); ok = np.isfinite(f)
    if q == 'Efficiency':
        ok &= m > 70
    a.scatter(m[ok], f[ok], s=6, c=np.array(P['Speed'])[ok], cmap='cividis', alpha=.8)
    lo, hi = np.nanmin(np.r_[m[ok], f[ok]]), np.nanmax(np.r_[m[ok], f[ok]])
    a.plot([lo, hi], [lo, hi], '-', color='#9ca3af', lw=1)
    a.set_title(lab, loc='left', fontsize=10); a.set_xlabel('Motor-CAD'); a.set_ylabel('FEMM')
F_PAR = png(fig)

# d-q flux linkage at Motor-CAD's own currents (B), both FEMM variants
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
PB = EQ['map']['points']; Ipk = np.hypot(PB['Id_Peak'], PB['Iq_Peak'])
for a, q, lab in ((ax[0], 'Flux_Linkage_D', 'ψd'), (ax[1], 'Flux_Linkage_Q', 'ψq'),
                  (ax[2], 'Electromagnetic_Torque', 'Electromagnetic torque')):
    m = np.array(PB[q])
    for v, c, lb in ((D2, COL['d2'], 'FEMM 2D'), (EQ, COL['eq'], 'FEMM, Motor-CAD equivalent')):
        f = arr(v['map']['B'][q])
        a.scatter(Ipk, 100 * (f / m - 1), s=6, color=c, alpha=.6, label=lb)
    a.axhline(0, color=COL['mc'], lw=1); a.set_title(lab + ', FEMM vs MC (%)', loc='left', fontsize=9.5)
    a.set_xlabel('Phase current (A peak)'); a.set_ylim(-10, 6)
ax[0].legend(fontsize=8.5, markerscale=2)
F_DQ = png(fig)

# grid: efficiency difference table as heat map (torque x speed)
G = EQ['grid']['points']; dG = arr(EQ['grid']['A']['Efficiency']) - np.array(G['Efficiency'])
fig, ax = plt.subplots(figsize=(12, 4.6))
sc_ = ax.scatter(G['Speed'], G['Shaft_Torque'], c=dG, s=110, marker='s', cmap='RdBu_r', norm=TwoSlopeNorm(0, -1, 1))
for xx, yy, dd in zip(G['Speed'], G['Shaft_Torque'], dG):
    if np.isfinite(dd):
        ax.text(xx, yy, '%+.1f' % dd, ha='center', va='center', fontsize=5.5)
fig.colorbar(sc_, ax=ax, label='FEMM − Motor-CAD efficiency (points)')
ax.set_xlabel('Speed (rpm)'); ax.set_ylabel('Shaft torque (N·m)'); ax.grid(False)
ax.set_title('Grid sheet: efficiency difference at each Motor-CAD grid point (Motor-CAD-equivalent FEMM)', loc='left', fontsize=10)
F_GRID = png(fig)

# ---------------- tables ----------------
NAMES = [('Shaft_Torque', 'Shaft torque'), ('Electromagnetic_Torque', 'Electromagnetic torque'), ('Efficiency', 'Efficiency (points)'),
         ('Stator_Current_Line_RMS', 'Line current'), ('Id_Peak', 'Id'), ('Iq_Peak', 'Iq'), ('Voltage_Phase_Peak', 'Phase voltage'),
         ('Power_Factor', 'Power factor'), ('Flux_Linkage_D', 'd-axis flux'), ('Flux_Linkage_Q', 'q-axis flux'),
         ('Stator_Copper_Loss_DC', 'DC copper loss'), ('Stator_Copper_Loss_AC', 'AC copper loss'), ('Iron_Loss_Stator', 'Stator iron loss'),
         ('Iron_Loss_Rotor', 'Rotor iron loss'), ('Iron_Loss_Hysteresis', 'Hysteresis loss'), ('Iron_Loss_Eddy', 'Eddy-current loss'),
         ('Iron_Loss', 'Iron loss'), ('Mechanical_Loss', 'Mechanical loss'), ('Total_Loss', 'Total loss'), ('Idc', 'DC current')]


def cell(s, q):
    if not s:
        return '<td>–</td>'
    if q == 'Efficiency':
        v = s['mean_abs']; return '<td class="%s">%.2f</td>' % ('ok' if v <= 0.5 else 'warn', v)
    v = s['mean_abs_rel_pct']
    return '<td class="%s">%.1f %%</td>' % ('ok' if v is not None and v <= 3 else 'warn', v) if v is not None else '<td>–</td>'


def stat_table(v):
    rows = ''
    for q, n in NAMES:
        rows += '<tr><td>%s</td>%s%s%s%s</tr>' % (n, cell(v['map']['summary']['A'][q], q), cell(v['map']['summary']['B'][q], q),
                                                 cell(v['grid']['summary']['A'][q], q), cell(v['grid']['summary']['B'][q], q))
    return rows


def env_rows():
    out = ''
    for k, n in enumerate(spd):
        if n % 1000 or n == 0 and False:
            continue
        g = lambda q: (C['mcad_envelope'][q][k], EQ['envelope'][q][k])
        t, p, e, i, v = g('Shaft_Torque'), g('Shaft_Power'), g('Efficiency'), g('Stator_Current_Line_RMS'), g('Voltage_Phase_Peak')
        if t[1] is None:
            continue
        out += '<tr><td>%d</td><td>%.2f</td><td>%.2f</td><td>%+.1f %%</td><td>%.2f</td><td>%.2f</td><td>%s</td><td>%s</td><td>%.0f</td><td>%.0f</td><td>%.1f</td><td>%.1f</td></tr>' % (
            n, t[0], t[1], 100 * (t[1] / t[0] - 1), p[0] / 1e3, p[1] / 1e3, '%.2f' % e[0] if n else '–', '%.2f' % e[1] if n else '–', i[0], i[1], v[0], v[1])
    return out


def sens_rows():
    base = {(r['id'], r['iq']): r for r in SENS if r['variant'] == 'base'}
    out = ''
    for name in dict.fromkeys(r['variant'] for r in SENS):
        if name == 'base':
            continue
        rr = [r for r in SENS if r['variant'] == name]
        dq_ = ['%+.1f %%' % (100 * (r['psiq'] / base[(r['id'], r['iq'])]['psiq'] - 1)) for r in rr]
        dd_ = ['%+.1f %%' % (100 * (r['psid'] / base[(r['id'], r['iq'])]['psid'] - 1)) for r in rr]
        out += '<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (' class="hl"' if 'stacking' in name else '', name, dq_[0], dq_[1], dd_[0], dd_[1])
    return out


def hist_rows():
    o2 = OLD['2d']['map']['summary']; e2 = EQ['map']['summary']
    rows = [('First build: 2D, FEA stacking 0.97, air-gap force torque, iron grid 4 × 3', o2),
            ('Refined fields: FEA stacking 1.0, skew slices −2/0/+2°, d-q flux torque, iron grid 4 × 5 (iron loss as first build)', V2L['mcad_eq']['map']['summary']),
            ('Final: as above + Motor-CAD minor-loop hysteresis, harmonic eddy loss', e2)]
    f = lambda s, q, k='mean_abs_rel_pct': ('%.2f' % s['A'][q]['mean_abs']) if q == 'Efficiency' else '%.1f %%' % s['A'][q][k]
    return ''.join('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
        n, f(s, 'Shaft_Torque'), f(s, 'Efficiency'), f(s, 'Iron_Loss'), f(s, 'Total_Loss'), f(s, 'Stator_Current_Line_RMS')) for n, s in rows)


def iron_rows():
    names = dict(all='Every harmonic, hysteresis and eddy', fund='Fundamental only', peak='Peak B per component (first build, stator)',
                 ml='Peak B + minor loops (Kc 0.65) for hysteresis, harmonics for eddy (final)')
    out = ''
    for v in ('fund', 'peak', 'all', 'ml'):
        e = IV['errors'][v]
        out += '<tr%s><td>%s</td>%s</tr>' % (' class="hl"' if v == 'ml' else '', names[v], ''.join('<td>%+.0f %%</td>' % e[k]['mean_pct'] for k in ('stator', 'rotor', 'hysteresis', 'eddy', 'total')))
    return out


def psiq_gap():
    P0 = OLD['2d']['map']['points']; I = np.hypot(P0['Id_Peak'], P0['Iq_Peak']); r = 100 * (arr(OLD['2d']['map']['B']['Flux_Linkage_Q']) / np.array(P0['Flux_Linkage_Q']) - 1)
    return np.nanmedian(r[I < 100]), np.nanmedian(r[I > 250])


def ref_rows():
    out = '<tr><td>Motor-CAD Lab</td><td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>' % (REF_MC['Flux_Linkage_D'], REF_MC['Flux_Linkage_Q'], REF_MC['Electromagnetic_Torque'])
    for n, r in REF.items():
        out += '<tr><td>FEMM %s</td><td>%.2f (%+.1f %%)</td><td>%.2f (%+.1f %%)</td><td>%.2f (%+.1f %%)</td></tr>' % (
            n, r['Flux_Linkage_D'], 100 * (r['Flux_Linkage_D'] / REF_MC['Flux_Linkage_D'] - 1), r['Flux_Linkage_Q'], 100 * (r['Flux_Linkage_Q'] / REF_MC['Flux_Linkage_Q'] - 1),
            r['Electromagnetic_Torque'], 100 * (r['Electromagnetic_Torque'] / REF_MC['Electromagnetic_Torque'] - 1))
    return out


sA, sG = EQ['map']['summary']['A'], EQ['grid']['summary']['A']
eff_pk_fe = float(np.nanmax(EFF[:, S <= 9500])); eff_pk_mc = float(np.nanmax(MCM['Efficiency']))
k0 = 0
KPI = [('Peak efficiency', '%.2f %%' % eff_pk_mc, '%.2f %%' % eff_pk_fe),
       ('Torque at 0 rpm, 360 A', '%.2f N·m' % MCE['Shaft_Torque'][0], '%.2f N·m' % EQ['envelope']['Shaft_Torque'][0]),
       ('Efficiency, map points', 'mean |difference|', '%.2f points' % sA['Efficiency']['mean_abs']),
       ('Shaft torque, map points', 'mean |difference|', '%.1f %%' % sA['Shaft_Torque']['mean_abs_rel_pct']),
       ('Total loss, map points', 'mean |difference|', '%.1f %%' % sA['Total_Loss']['mean_abs_rel_pct']),
       ('Iron loss, map points', 'mean difference', '%+.1f %%' % sA['Iron_Loss']['mean_rel_pct']),
       ('Efficiency, grid points', 'mean |difference|', '%.2f points' % sG['Efficiency']['mean_abs'])]
kpi = ''.join('<tr><td>%s</td><td>%s</td><td>%s</td></tr>' % k for k in KPI)

CSS = open(os.path.join(H, 'report_emf', 'backemf_validation.html'), encoding='utf-8').read().split('<style>')[1].split('</style>')[0]
CSS += """
.wrap { max-width:1180px } tr.hl td { background:var(--ok-bg) } .kpi td:first-child { min-width:220px }
h3 { font:600 18px/1.25 var(--f-d); margin:20px 0 6px } .two { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr)); gap:16px }
"""
html = f'''<title>Design9795 Lab Validation</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap"><div class="sheet">
<p class="eyebrow">Design9795_1 · Lab model · winding and magnets 100 °C · 52 V · 360 A rms · Idc 280 A · MTPA</p>
<h1>FEMM Lab model against Motor-CAD Lab</h1>
<div class="verdict"><strong>Equivalent within engineering accuracy.</strong> Over the {sA['Efficiency']['n']} points of the Motor-CAD efficiency map, the refined FEMM model is within {sA['Efficiency']['mean_abs']:.2f} efficiency points on average, shaft torque within {sA['Shaft_Torque']['mean_abs_rel_pct']:.1f} %, total loss within {sA['Total_Loss']['mean_abs_rel_pct']:.1f} %. Over the {sG['Efficiency']['n']} grid points: {sG['Efficiency']['mean_abs']:.2f} points and {sG['Shaft_Torque']['mean_abs_rel_pct']:.1f} %. The largest remaining difference is the hysteresis part of the iron loss ({sA['Iron_Loss_Hysteresis']['mean_rel_pct']:+.0f} %); eddy loss agrees within {sA['Iron_Loss_Eddy']['mean_abs_rel_pct']:.0f} %, and iron loss overall is {sA['Iron_Loss']['mean_rel_pct']:+.0f} %, a small share of the total.</div>
<div class="tw"><table class="kpi"><thead><tr><th>Headline</th><th>Motor-CAD</th><th>FEMM</th></tr></thead><tbody>{kpi}</tbody></table></div>

<h2>1. How the FEMM model was refined</h2>
<p>The first FEMM Lab build (2D, full machine) matched the back-EMF but its torque at full current sat about 1 % under Motor-CAD, and adding the Motor-CAD rotor skew made that worse. The cause was traced to the q-axis flux, then fixed step by step:</p>
<dl>
<dt>Half-machine model</dt><dd>12 slots / 10 poles repeat every 180° with the sign flipped: the model is one half with anti-periodic boundaries and FEMM's air-gap element (fixed meshes, no remeshing per rotor position). Same results as the full machine within 0.1 % (flux linkage) and 0.2 % (torque), 2.2–2.8 × faster.</dd>
<dt>Stacking factor in the field solution</dt><dd>With the steel BH curve reduced by the 0.97 stacking factor, FEMM's q-axis flux was 3–8 % under Motor-CAD while the d axis matched. A one-at-a-time sensitivity study (table below) shows that only the stacking factor has that signature. Motor-CAD's flux map is reproduced with the BH curve applied unreduced (fill 1.0); 0.97 stays in the masses and the iron loss, which match Motor-CAD's masses exactly.</dd>
<dt>Rotor skew</dt><dd>Motor-CAD skews the rotor in 3 slices at −2 / 0 / +2° mechanical, and its Lab includes it (its d-axis flux at zero current, 19.835 mV·s, is the skewed open-circuit value). FEMM evaluates each slice with the stator current turned by ∓10° electrical in that slice's rotor frame, turns the flux back and averages: exact for the fundamental, and correct under saturation.</dd>
<dt>Torque</dt><dd>Motor-CAD Lab torque is 1.5 p (ψd iq − ψq id) (its 44.76 N·m at 1000 rpm reproduces exactly). The refined FEMM model uses the same expression; the FEMM air-gap force torque is shown as the 2D variant.</dd>
<dt>Iron-loss grid</dt><dd>Fixed-mesh, full-cycle field solutions (30 points per cycle, as Motor-CAD) at 4 d-axis × 5 q-axis currents (was 4 × 3), Motor-CAD's Steinmetz coefficients with the 2π² eddy convention and build factor 1.6. The finer grid showed that the first build's 3-level fit had over-predicted the loss at light load by 20–30 %, which happened to offset an under-prediction; on the true curve, the first-build formulation was 18 % low.</dd>
<dt>Iron-loss formulation</dt><dd>Motor-CAD's loss settings include a minor-loop factor Kc = 0.65 that the first build ignored. Hysteresis now uses the peak flux density per element and component with the Lavers minor-loop factor 1 + Kc·ΣΔB<sub>minor</sub>/B<sub>pk</sub>; eddy loss uses the sum over harmonics (the dB/dt content). Both come from the same saved field solutions, with no new FEA and no fitted factors.</dd>
</dl>
<h3>What moves the q-axis flux (FEMM, rotor at 0°)</h3>
<div class="tw"><table><thead><tr><th>Change from the first build</th><th>ψq, iq 60 A</th><th>ψq, full current</th><th>ψd, iq 60 A</th><th>ψd, full current</th></tr></thead><tbody>{sens_rows()}</tbody></table></div>
<p class="note">Low current = id 0, iq 60 A phase peak; full current = Motor-CAD's 1000 rpm envelope point (id −85.0 A, iq 281.4 A phase peak). Single rotor position, so absolute values differ slightly from the position-averaged map. On the map points, the first FEMM build's ψq was {psiq_gap()[0]:+.1f} % (phase current under 100 A) and {psiq_gap()[1]:+.1f} % (over 250 A) against Motor-CAD.</p>
<h3>At Motor-CAD's full-current point, 1000 rpm</h3>
<div class="tw"><table><thead><tr><th>Model</th><th>ψd (mV·s)</th><th>ψq (mV·s)</th><th>Torque (N·m)</th></tr></thead><tbody>{ref_rows()}</tbody></table></div>
<h3>Iron-loss formulations against Motor-CAD (same currents and speed, all map points)</h3>
<div class="tw"><table><thead><tr><th>Formulation (same FEMM fields)</th><th>Stator</th><th>Rotor</th><th>Hysteresis</th><th>Eddy</th><th>Total</th></tr></thead><tbody>{iron_rows()}</tbody></table></div>
<p class="note">Mean signed difference from Motor-CAD.</p>
<h3>Effect of the refinement over all map points (same speed and torque)</h3>
<div class="tw"><table><thead><tr><th>FEMM build</th><th>Shaft torque</th><th>Efficiency (points)</th><th>Iron loss</th><th>Total loss</th><th>Line current</th></tr></thead><tbody>{hist_rows()}</tbody></table></div>
<p class="note">Mean absolute difference from Motor-CAD (relative, except efficiency in points).</p>

<h2>2. Torque-speed envelope</h2>
<figure><img src="{F_ENV}" alt="Envelope comparison"><figcaption>Motor-CAD envelope (envelop_1.xlsx) against the FEMM Lab model with the same limits: 52 V DC, modulation 0.95, 360 A rms, 280 A DC, MTPA with field weakening.</figcaption></figure>
<div class="tw"><table><thead><tr><th>rpm</th><th>T Motor-CAD</th><th>T FEMM</th><th>ΔT</th><th>P MC (kW)</th><th>P FEMM</th><th>η MC (%)</th><th>η FEMM</th><th>I MC (A)</th><th>I FEMM</th><th>V MC</th><th>V FEMM</th></tr></thead><tbody>{env_rows()}</tbody></table></div>

<h2>3. Efficiency map</h2>
<figure><img src="{F_MAP}" alt="Efficiency maps and difference"><figcaption>Left: Motor-CAD map (Map_1.xlsx). Middle: FEMM. Right: FEMM − Motor-CAD at each Motor-CAD map point, FEMM run at the same speed and shaft torque (MTPA).</figcaption></figure>
<figure><img src="{F_GRID}" alt="Grid comparison"><figcaption>Grid.xlsx: each Motor-CAD grid point (torque × speed), FEMM at the same point.</figcaption></figure>

<h2>4. Losses, currents and voltage</h2>
<figure><img src="{F_PAR}" alt="Parity plots"><figcaption>Every Motor-CAD map point, FEMM at the same speed and shaft torque; colour = speed. Diagonal = perfect agreement.</figcaption></figure>
<h3>Mean absolute difference, Motor-CAD-equivalent FEMM</h3>
<div class="tw"><table><thead><tr><th>Quantity</th><th>Map: same (n, T)</th><th>Map: same currents</th><th>Grid: same (n, T)</th><th>Grid: same currents</th></tr></thead><tbody>{stat_table(EQ)}</tbody></table></div>
<p class="note">"Same (n, T)": FEMM finds its own MTPA currents for Motor-CAD's speed and shaft torque (what a user of either Lab sees). "Same currents": FEMM evaluated at Motor-CAD's own id and iq, which isolates the flux and loss models. Efficiency in points; the rest relative. Green ≤ 3 % (≤ 0.5 points).</p>
<h3>FEMM 2D (no skew, air-gap force torque, for reference)</h3>
<div class="tw"><table><thead><tr><th>Quantity</th><th>Map: same (n, T)</th><th>Map: same currents</th><th>Grid: same (n, T)</th><th>Grid: same currents</th></tr></thead><tbody>{stat_table(D2)}</tbody></table></div>

<h2>5. d-q flux linkage and torque at the same currents</h2>
<figure><img src="{F_DQ}" alt="d-q flux comparison"><figcaption>FEMM against Motor-CAD at Motor-CAD's own id and iq (all map points), against phase current. These set Ld, Lq, the torque per amp and the field-weakening behaviour.</figcaption></figure>

<h2>6. Open circuit (from the back-EMF validation)</h2>
<p>Back-EMF fundamental {OC['skewed (MC slices)']['E_fund_pk']:.2f} V against Motor-CAD {OC['MC']['E_fund_pk']:.2f} V at 1000 rpm, Ke {OC['skewed (MC slices)']['Ke']:.4f} against {OC['MC']['Ke']:.4f} V·s/rad, magnet flux {OC['skewed (MC slices)']['psi_m_mVs']:.2f} against {1e3 * OC['MC']['psi_d']:.2f} mV·s, THD {OC['skewed (MC slices)']['THD']:.2f} % against {OC['MC']['THD']:.2f} %.</p>

<h2>7. What still differs, and why</h2>
<ul>
<li><strong>Hysteresis loss</strong> ({sA['Iron_Loss_Hysteresis']['mean_rel_pct']:+.0f} %, larger at high speed in field weakening). Eddy loss now agrees, so the field waveforms and frequencies are right; the remaining gap is in how the hysteresis loop is counted. Motor-CAD's exact minor-loop algorithm is not documented in the exported data. FEMM counts loops from 30 samples per cycle per flux component. A further candidate is the lamination fill: if Motor-CAD evaluates the loss at the steel flux density B/0.97, that alone adds about 6 % (0.97<sup>−2.07</sup>). It was not applied because it cannot be confirmed from the exports. Iron loss is 1–3 % of the input power, so efficiency stays within about 0.1 point.</li>
<li><strong>Rotor iron loss</strong> ({sA['Iron_Loss_Rotor']['mean_rel_pct']:+.0f} %). In a spoke ferrite rotor it comes only from slot and MMF harmonics, the most method-dependent term; a few watts.</li>
<li><strong>AC copper loss</strong> is a few watts; both use the hybrid method with adjustment factor 2, FEMM from its own slot field.</li>
<li><strong>Current and phase advance at high speed</strong>: where the voltage limit is active, small flux differences move the MTPA point along the voltage ellipse; the shaft torque and efficiency stay close because the operating point is defined by them.</li>
</ul>

<h2>8. Method and files</h2>
<dl>
<dt>FEMM model</dt><dd>backemf.py (geometry from the Design9795_1 screenshots; SYM=1 half model; STACK_FEA), lab_map.py (d-q flux map 6 × 6 currents × 4 rotor positions), iron_loss_age.py (iron loss), iron_fit.py (iron-loss formulations from the saved fields), lab_model.py (Lab: losses, limits, skew slices, torque; loss model v3), make_lab_v2.py (maps), compare9795.py (this comparison).</dd>
<dt>Motor-CAD</dt><dd>Map_1.xlsx, envelop_1.xlsx, Grid.xlsx (Lab, 100 °C), M235-35A.xlsx; same BH curve and loss coefficients in FEMM.</dd>
<dt>Settings</dt><dd>DC 52 V, modulation index 0.95, 360 A rms line, Idc 280 A, delta, 10 turns/coil, 2 parallel paths, SWG 11, MLT 203.6 mm, end-winding inductance 1.954 µH, bearing and windage loss as Motor-CAD (148.9 W at 9000 rpm).</dd>
</dl>
</div></div>
'''
os.makedirs(os.path.join(H, 'report_lab'), exist_ok=True)
open(os.path.join(H, 'report_lab', 'lab_validation.html'), 'w', encoding='utf-8').write(html)
print(len(html) // 1024, 'kB')
