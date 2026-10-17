"""One-to-one comparison of three Motor-CAD Lab operating points (screenshots 11 Oct 2026, 100 C) with the FEMM v2 model.
  same_point:   FEMM MTPA / field-weakening point at Motor-CAD's speed and shaft torque (or max current for OP1)
  same_current: FEMM evaluated at Motor-CAD's phase current and phase advance
Writes op_compare.json."""
import json, math, os
import numpy as np
import lab_map as L
import lab_model as LM

MC = {
    'OP1 9000 rpm, max current': dict(mode='maxcur', Speed=9000, Shaft_Torque=14.02, Shaft_Power=13210, Efficiency=90.74, Ripple_pp=2.561, Ripple_pct=17.65,
        I_ph_rms=139.7, I_line_rms=241.9, Idc=280, V_ph_rms=34.93, Phase_Advance=66.38, Psi_d=2.951, Psi_q=9.404, Psi_m=20.5, Ld=0.09696, Lq=0.1188,
        Total_Loss=1348, Cu=882.4, Cu_DC=390.5, Cu_AC=492, Iron=317.2, Magnet_Loss=0, Mech=148.9, Windage=44.66, Friction=104.2,
        EM_Power=13680, EM_Torque=14.51, Magnet_Torque=12.16, Reluctance_Torque=2.35, Terminal_Power=14560, PF=0.9948),
    'OP2 9339 rpm, 3.24 N.m': dict(mode='torque', Speed=9339, Shaft_Torque=3.24, Shaft_Power=3169, Efficiency=82.95, Ripple_pp=3.039, Ripple_pct=83.22,
        I_ph_rms=73.7, I_line_rms=127.7, Idc=73.46, V_ph_rms=34.93, Phase_Advance=78.5, Psi_d=9.535, Psi_q=2.828, Psi_m=19.95, Ld=0.102, Lq=0.1361,
        Total_Loss=651.1, Cu=251.2, Cu_DC=108.7, Cu_AC=142.5, Iron=243.7, Magnet_Loss=0, Mech=156.2, Windage=46.86, Friction=109.3,
        EM_Power=3571, EM_Torque=3.651, Magnet_Torque=3.108, Reluctance_Torque=0.5434, Terminal_Power=3820, PF=0.4946),
    'OP3 2000 rpm, 45 N.m': dict(mode='torque', Speed=2000, Shaft_Torque=44.9, Shaft_Power=9405, Efficiency=89.86, Ripple_pp=1.906, Ripple_pct=4.188,
        I_ph_rms=208.8, I_line_rms=361.7, Idc=201.3, V_ph_rms=23.38, Phase_Advance=20.78, Psi_d=11.38, Psi_q=27.94, Psi_m=19.11, Ld=0.07376, Lq=0.1012,
        Total_Loss=1061, Cu=932.1, Cu_DC=872.8, Cu_AC=59.35, Iron=107.6, Magnet_Loss=0, Mech=21.07, Windage=6.321, Friction=14.75,
        EM_Power=9533, EM_Torque=45.52, Magnet_Torque=39.57, Reluctance_Torque=5.95, Terminal_Power=10470, PF=0.7146),
}
mod = LM.Model('v2')
G = mod.grid(321)
P = L.P


def femm_quantities(n, i_d, i_q):
    """All quantities at path-peak currents i_d, i_q (scalars), Motor-CAD units (phase values, rms where Motor-CAD shows rms)."""
    R = {k: float(v[0]) for k, v in mod.evaluate(n, np.array([i_d]), np.array([i_q])).items()}
    psi_m = float(mod.F['psid'](np.array([[0.0, i_q]]))[0])            # d-axis flux with id = 0, same iq (cross-saturated)
    Ld = (R['Flux_Linkage_D'] / 1e3 - L.L_EW_PATH * i_d - psi_m) / i_d if abs(i_d) > 1e-6 else float('nan')
    Lq = (R['Flux_Linkage_Q'] / 1e3) / i_q if abs(i_q) > 1e-6 else float('nan')
    iq_ph = 2 * i_q
    Tmag = 1.5 * P * psi_m * iq_ph
    wm = n * 2 * math.pi / 60
    mech = L.mech_loss(n); b = L.LOSS['bearing']; w = L.LOSS['windage']
    return dict(Speed=n, Shaft_Torque=R['Shaft_Torque'], Shaft_Power=R['Shaft_Power'], Efficiency=R['Efficiency'],
                I_ph_rms=R['Ipk'] * 2 / math.sqrt(2), I_line_rms=R['Stator_Current_Line_RMS'], Idc=R['Idc'], V_ph_rms=R['Voltage_Phase_Peak'] / math.sqrt(2),
                Phase_Advance=R['Phase_Advance'], Psi_d=R['Flux_Linkage_D'], Psi_q=R['Flux_Linkage_Q'], Psi_m=psi_m * 1e3,
                Ld=(Ld + L.L_EW_PATH) * 1e3 / 2 if Ld == Ld else float('nan'), Lq=(Lq + L.L_EW_PATH) * 1e3 / 2 if Lq == Lq else float('nan'),   # phase = path / 2 (2 paths)
                Total_Loss=R['Total_Loss'], Cu=R['Stator_Copper_Loss_DC'] + R['Stator_Copper_Loss_AC'], Cu_DC=R['Stator_Copper_Loss_DC'],
                Cu_AC=R['Stator_Copper_Loss_AC'], Iron=R['Iron_Loss'], Iron_Stator=R['Iron_Loss_Stator'], Iron_Rotor=R['Iron_Loss_Rotor'],
                Iron_Hyst=R['Iron_Loss_Hysteresis'], Iron_Eddy=R['Iron_Loss_Eddy'], Magnet_Loss=0.0, Mech=mech,
                Windage=w['W'] * (n / w['ref_rpm']) ** w['exponent'], Friction=(b['front_W'] + b['rear_W']) * (n / b['ref_rpm']) ** b['exponent'],
                EM_Power=R['Electromagnetic_Torque'] * wm, EM_Torque=R['Electromagnetic_Torque'], Magnet_Torque=Tmag,
                Reluctance_Torque=R['Electromagnetic_Torque'] - Tmag, Terminal_Power=R['Terminal_Power'], PF=R['Power_Factor'],
                V_limit_pct=100 * R['Voltage_Phase_Peak'] / L.V_MAX, id_path=i_d, iq_path=i_q)


out = {}
for name, m in MC.items():
    n = m['Speed']
    Ipk_ph = m['I_ph_rms'] * math.sqrt(2); g = math.radians(m['Phase_Advance'])
    same_cur = femm_quantities(n, -Ipk_ph * math.sin(g) / 2, Ipk_ph * math.cos(g) / 2)
    if m['mode'] == 'maxcur':
        r = mod.envelope_point(n, G)
    else:
        r = mod.operate(n, [m['Shaft_Torque']], G)[0] or mod.envelope_point(n, G)   # OP3: 44.9 N.m is above FEMM's 360 A limit -> max torque
    same_pt = femm_quantities(n, r['Id_Peak'] / 2, r['Iq_Peak'] / 2) if r else None
    out[name] = dict(mcad=m, same_point=same_pt, same_current=same_cur)
    print('\n==', name)
    print('%-18s %10s %10s %7s %10s %7s' % ('quantity', 'Motor-CAD', 'FEMM pt', 'diff', 'FEMM cur', 'diff'))
    for k, v in m.items():
        if k == 'mode':
            continue
        a = same_pt.get(k) if same_pt else None; b_ = same_cur.get(k)
        d = lambda x: ('%+6.1f%%' % (100 * (x - v) / v)) if (x is not None and v) else '   -  '
        print('%-18s %10.4g %10s %7s %10s %7s' % (k, v, '%.4g' % a if a is not None else '-', d(a), '%.4g' % b_ if b_ is not None else '-', d(b_)))
json.dump(out, open(os.path.join(L.H, 'op_compare.json'), 'w'), indent=1)
