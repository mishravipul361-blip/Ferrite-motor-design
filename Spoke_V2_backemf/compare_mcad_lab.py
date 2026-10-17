"""FEMM Lab model vs Motor-CAD Lab exports (100 C): envelope and every map point.

  A. same operating point: Motor-CAD (speed, shaft torque) -> FEMM MTPA point -> all quantities
  B. same currents: FEMM evaluated at Motor-CAD's own id/iq and speed -> isolates the loss / flux models
Writes lab_compare_<loss>.json.   Run: python compare_mcad_lab.py [v1|v2]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mcad_lab'))
from load_mcad import E, M
import lab_model as LM

loss = sys.argv[1] if len(sys.argv) > 1 else 'v2'
mod = LM.Model(loss)
G = mod.grid(201)
Q = ['Shaft_Torque', 'Electromagnetic_Torque', 'Efficiency', 'Stator_Current_Line_RMS', 'Id_Peak', 'Iq_Peak', 'Voltage_Phase_Peak',
     'Stator_Copper_Loss_DC', 'Stator_Copper_Loss_AC', 'Iron_Loss_Stator', 'Iron_Loss_Rotor', 'Iron_Loss', 'Iron_Loss_Hysteresis', 'Iron_Loss_Eddy',
     'Mechanical_Loss', 'Total_Loss', 'Idc', 'Power_Factor', 'Flux_Linkage_D', 'Flux_Linkage_Q', 'Phase_Advance']

# ---- envelope ----
env = {q: [] for q in Q + ['Speed', 'Shaft_Power']}
for n in E['Speed']:
    r = mod.envelope_point(n if n > 0 else 1.0, G)
    env['Speed'].append(float(n))
    for q in Q + ['Shaft_Power']:
        env[q].append(r[q] if r else None)

# ---- map, A: same (speed, torque) ----
S, Tm = M['Speed'], M['Shaft_Torque']
A = {q: np.full(S.shape, np.nan) for q in Q}
for i in range(S.shape[0]):
    n = S[i, 0]
    if n <= 0:
        continue
    res = mod.operate(n, Tm[i], G)
    for j, r in enumerate(res):
        if r and Tm[i, j] > 0.05:
            for q in Q:
                A[q][i, j] = r[q]
# ---- map, B: same currents ----
B = {q: np.full(S.shape, np.nan) for q in Q}
for i in range(S.shape[0]):
    n = S[i, 0]
    if n <= 0:
        continue
    r = mod.evaluate(n, M['Id_Peak'][i] / 2, M['Iq_Peak'][i] / 2)
    for q in Q:
        B[q][i] = r[q]


def stats(F, ref, mask):
    d = F - ref
    m = mask & ~np.isnan(d)
    if not m.any():
        return None
    rel = 100 * d[m] / np.where(np.abs(ref[m]) > 1e-9, np.abs(ref[m]), np.nan)
    return dict(n=int(m.sum()), mean_abs=float(np.nanmean(np.abs(d[m]))), mean=float(np.nanmean(d[m])),
                max_abs=float(np.nanmax(np.abs(d[m]))), mean_rel_pct=float(np.nanmean(rel)), mean_abs_rel_pct=float(np.nanmean(np.abs(rel))))


valid = (S > 0) & (Tm > 0.5)
summary = {'A_same_point': {}, 'B_same_current': {}}
for q in Q:
    summary['A_same_point'][q] = stats(A[q], M[q], valid)
    summary['B_same_current'][q] = stats(B[q], M[q], valid)
out = dict(loss_model=loss, envelope=env, mcad_envelope={q: E[q].tolist() for q in Q + ['Speed', 'Shaft_Power']},
           A={q: np.where(np.isnan(v), None, v).tolist() for q, v in A.items()},
           B={q: np.where(np.isnan(v), None, v).tolist() for q, v in B.items()},
           summary=summary)
json.dump(out, open(os.path.join(LM.H, 'lab_compare_%s.json' % loss), 'w'))
print('loss model', loss)
print('%-24s %28s   %28s' % ('quantity', 'A: same (n,T) mean / mean|rel| %', 'B: same (n,id,iq) mean / mean|rel| %'))
for q in Q:
    a, b = summary['A_same_point'][q], summary['B_same_current'][q]
    fa = '%9.3f / %6.1f%%' % (a['mean'], a['mean_abs_rel_pct']) if a else '-'
    fb = '%9.3f / %6.1f%%' % (b['mean'], b['mean_abs_rel_pct']) if b else '-'
    print('%-24s %28s   %28s' % (q, fa, fb))
print('envelope speed  T_mcad  T_femm   eff_mcad eff_femm   Pfe_mcad Pfe_femm')
for k, n in enumerate(E['Speed']):
    if n % 1000 == 0 and env['Shaft_Torque'][k] is not None:
        print('%6d  %6.2f  %6.2f   %6.2f  %6.2f   %6.1f  %6.1f' % (n, E['Shaft_Torque'][k], env['Shaft_Torque'][k], E['Efficiency'][k],
              env['Efficiency'][k] or 0, E['Iron_Loss'][k], env['Iron_Loss'][k]))
