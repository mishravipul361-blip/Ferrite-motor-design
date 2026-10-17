"""Rebuild lab_100C.json (maps + envelope, 0-12 000 rpm) with the corrected v2 loss model (lab_model.py)."""
import json, math, os
import numpy as np
import lab_map as L
import lab_model as LM

mod = LM.Model('v2')
i_d, i_q = mod.grid(241)
speeds = np.arange(0, 12001, 100)
tq = np.arange(0.5, 60.01, 0.5)
keys = ('eff', 'loss', 'pcu', 'pac', 'pfe_st', 'pfe_rot', 'pmech', 'I_line', 'V_ph', 'pf', 'I_dc', 'P_in', 'id', 'iq', 'gamma')
mp = {k: np.full((len(tq), len(speeds)), np.nan) for k in keys}
env = []
for a, n in enumerate(speeds):
    R = mod.evaluate(max(n, 1.0), i_d, i_q)
    feas = (R['Voltage_Phase_Peak'] <= L.V_MAX) & (R['Terminal_Power'] <= L.P_DC_MAX + 1e-6) & (R['Shaft_Torque'] > 0)
    if not feas.any():
        break
    k = np.flatnonzero(feas)[np.argmax(R['Shaft_Torque'][feas])]
    lim = ('voltage' if R['Voltage_Phase_Peak'][k] > L.V_MAX - 0.3 else '') + (' dc-current' if R['Terminal_Power'][k] > L.P_DC_MAX - 50 else '') + \
          (' current' if R['Ipk'][k] > L.I_PATH_PK - 1 else '')
    env.append(dict(rpm=int(n), T=float(R['Shaft_Torque'][k]), P_kW=float(R['Shaft_Power'][k] / 1e3) if n else 0.0,
                    I_line=float(R['Stator_Current_Line_RMS'][k]), V=float(R['Voltage_Phase_Peak'][k]), I_dc=float(R['Idc'][k]),
                    id=float(i_d[k]), iq=float(i_q[k]), eff=float(R['Efficiency'][k]) if n else None,
                    pcu=float(R['Stator_Copper_Loss_DC'][k]), pac=float(R['Stator_Copper_Loss_AC'][k]) if n else 0.0,
                    pfe=float(R['Iron_Loss'][k]) if n else 0.0, pmech=float(R['Mechanical_Loss'][k]) if n else 0.0, limit=lim))
    if n == 0:
        continue
    for b, T in enumerate(tq):
        c = np.flatnonzero(feas & (np.abs(R['Shaft_Torque'] - T) <= 0.25))
        if not len(c):
            continue
        k = c[np.argmin(R['Ipk'][c])]
        vals = dict(eff=R['Efficiency'][k], loss=R['Total_Loss'][k], pcu=R['Stator_Copper_Loss_DC'][k], pac=R['Stator_Copper_Loss_AC'][k],
                    pfe_st=R['Iron_Loss_Stator'][k], pfe_rot=R['Iron_Loss_Rotor'][k], pmech=R['Mechanical_Loss'][k],
                    I_line=R['Stator_Current_Line_RMS'][k], V_ph=R['Voltage_Phase_Peak'][k], pf=R['Power_Factor'][k], I_dc=R['Idc'][k],
                    P_in=R['Terminal_Power'][k], id=i_d[k], iq=i_q[k], gamma=R['Phase_Advance'][k])
        for kk, vv in vals.items():
            mp[kk][b, a] = vv
old = json.load(open(os.path.join(L.H, 'lab_100C.json')))
res = dict(settings=dict(old['settings'], loss_model='v2: fixed-mesh full-cycle FEA; stator peak-B Steinmetz, rotor per-harmonic Steinmetz, 2pi^2 eddy'),
           info=old['info'], speeds=speeds.tolist(), torques=tq.tolist(), envelope=env, dq=old['dq'],
           maps={k: np.where(np.isnan(v), None, np.round(v, 4)).tolist() for k, v in mp.items()})
json.dump(res, open(os.path.join(L.H, 'lab_100C.json'), 'w'))
e = mp['eff']; k = np.unravel_index(np.nanargmax(e), e.shape)
print('peak efficiency %.2f %% at %d rpm, %.1f N.m' % (e[k], speeds[k[1]], tq[k[0]]))
for rw in env:
    if rw['rpm'] % 1000 == 0:
        print('%6d  T %6.2f  P %6.2f kW  eff %s  Fe %.0f W  %s' % (rw['rpm'], rw['T'], rw['P_kW'], '%.2f' % rw['eff'] if rw['eff'] else '-', rw['pfe'], rw['limit']))
