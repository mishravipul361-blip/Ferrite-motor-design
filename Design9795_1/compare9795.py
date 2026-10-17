"""FEMM Lab model (half model, 100 C) vs the three Motor-CAD Lab exports of Design9795_1: envelope, map, grid.

For the map and the grid, every Motor-CAD point (speed > 0, shaft torque > 0.5 N.m) is compared two ways:
  A. same operating point: Motor-CAD (speed, shaft torque) -> FEMM MTPA point -> all quantities
  B. same currents: FEMM evaluated at Motor-CAD's own id / iq and speed -> isolates the flux and loss models
Two FEMM variants: 2D as solved (air-gap force torque), and Motor-CAD equivalent (rotor skew slices -2/0/+2 deg, d-q flux torque).
python compare9795.py -> lab_compare_9795.json"""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, 'mcad_lab'))
from load_mcad import E, M
import lab_model as LM

KS = 0.990
_d = json.load(open(os.path.join(H, 'mcad_lab', 'mcad_lab.json')))
G = {k: np.array([[np.nan if v in (None, 'None') else float(v) for v in r] for r in rows], float)
     for k, rows in _d['grid'].items() if k not in ('Units', 'Sheet1') and not k.startswith('Drive')}
Q = ['Shaft_Torque', 'Electromagnetic_Torque', 'Shaft_Power', 'Efficiency', 'Stator_Current_Line_RMS', 'Id_Peak', 'Iq_Peak', 'Voltage_Phase_Peak',
     'Stator_Copper_Loss_DC', 'Stator_Copper_Loss_AC', 'Iron_Loss_Stator', 'Iron_Loss_Rotor', 'Iron_Loss', 'Iron_Loss_Hysteresis', 'Iron_Loss_Eddy',
     'Mechanical_Loss', 'Total_Loss', 'Idc', 'Power_Factor', 'Flux_Linkage_D', 'Flux_Linkage_Q', 'Phase_Advance']


def points(D):
    m = (D['Speed'] > 0) & (D['Shaft_Torque'] > 0.5)
    return {q: D[q][m] for q in Q + ['Speed']}


def compare(mod, P, Gd):
    n_pts = len(P['Speed'])
    A = {q: np.full(n_pts, np.nan) for q in Q}; B = {q: np.full(n_pts, np.nan) for q in Q}
    for n in np.unique(P['Speed']):
        idx = np.flatnonzero(P['Speed'] == n)
        for k, r in zip(idx, mod.operate(n, P['Shaft_Torque'][idx], Gd)):
            if r:
                for q in Q:
                    A[q][k] = r[q]
        r = mod.evaluate(n, P['Id_Peak'][idx] / 2, P['Iq_Peak'][idx] / 2)
        for q in Q:
            B[q][idx] = r[q]
    return A, B


def stats(F, ref):
    d = F - ref; m = ~np.isnan(d) & ~np.isnan(ref)
    if not m.any():
        return None
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = 100 * d[m] / np.abs(ref[m])
    rel = rel[np.isfinite(rel) & (np.abs(ref[m]) > 1e-6)]
    return dict(n=int(m.sum()), mean=float(np.mean(d[m])), mean_abs=float(np.mean(np.abs(d[m]))), max_abs=float(np.max(np.abs(d[m]))),
                mean_rel_pct=float(np.mean(rel)) if len(rel) else None, mean_abs_rel_pct=float(np.mean(np.abs(rel))) if len(rel) else None)


out = dict(skew_factor=KS, mcad_envelope={q: E[q].tolist() for q in Q + ['Speed']})
sets = dict(map=points(M), grid=points(G))
VARIANTS = (('2d', dict()), ('mcad_eq', dict(slices=(-2, 0, 2), torque='flux')))
for name, kw in VARIANTS:
    mod = LM.Model('v3', **kw); Gd = mod.grid(201)
    env = {q: [] for q in Q}
    for n in E['Speed']:
        r = mod.envelope_point(n if n > 0 else 1.0, Gd)
        for q in Q:
            env[q].append(r[q] if r else None)
    res = dict(envelope=env)
    for sname, P in sets.items():
        A, B = compare(mod, P, Gd)
        res[sname] = dict(points={q: P[q].tolist() for q in Q + ['Speed']},
                          A={q: np.where(np.isnan(v), None, v).tolist() for q, v in A.items()},
                          B={q: np.where(np.isnan(v), None, v).tolist() for q, v in B.items()},
                          summary=dict(A={q: stats(A[q], P[q]) for q in Q}, B={q: stats(B[q], P[q]) for q in Q}))
    out[name] = res
    print('==== FEMM %s' % name)
    for sname in sets:
        s = res[sname]['summary']
        print('-- %s (%d points)   A same (n,T): mean diff / mean|rel|    B same currents' % (sname, len(sets[sname]['Speed'])))
        for q in Q:
            a, b = s['A'][q], s['B'][q]
            f = lambda x: '%9.3f / %5.1f%%' % (x['mean'], x['mean_abs_rel_pct'] or 0) if x else '-'
            print('   %-26s %22s   %22s' % (q, f(a), f(b)))
    print('envelope rpm  T_mc  T_fe   P_mc  P_fe   eff_mc eff_fe  Iline_mc Iline_fe  V_mc V_fe')
    for k, n in enumerate(E['Speed']):
        if env['Shaft_Torque'][k] is not None and n % 1000 == 0:
            print('%6d  %6.2f %6.2f  %6.0f %6.0f  %6.2f %6.2f  %6.0f %6.0f  %5.1f %5.1f' % (
                n, E['Shaft_Torque'][k], env['Shaft_Torque'][k], E['Shaft_Power'][k], env['Shaft_Power'][k], E['Efficiency'][k],
                env['Efficiency'][k] or 0, E['Stator_Current_Line_RMS'][k], env['Stator_Current_Line_RMS'][k], E['Voltage_Phase_Peak'][k], env['Voltage_Phase_Peak'][k]))
json.dump(out, open(os.path.join(H, 'lab_compare_9795.json'), 'w'))
