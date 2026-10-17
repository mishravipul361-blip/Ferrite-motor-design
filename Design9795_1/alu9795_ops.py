"""Operating points OP1-OP3 (as in the Spoke V2 aluminium study) for copper Design9795_1 and the aluminium designs:
currents, phase advance, voltage, flux linkage, power factor and the full loss breakdown.

OP1  9000 rpm, maximum shaft power (drive limits)
OP2  9339 rpm, 3.24 N.m shaft (light-load high-speed cruise)
OP3  2000 rpm, maximum shaft torque (360 A rms line). In the Spoke V2 study OP3 was 44.34 N.m, that motor's peak; Design9795_1
     copper reaches 44.11 N.m at 2000 rpm, so OP3 here is each design's own maximum.
Motor-CAD Lab values for copper at OP1 and OP3 (envelope export) for comparison.

python alu9795_ops.py   -> alu9795_ops.json
"""
import json, math, os
import numpy as np
import lab_map as L
import alu9795_study as S

H = S.H
OPS = {'OP1': dict(n=9000, T=None, label='9000 rpm, maximum power'), 'OP2': dict(n=9339, T=3.24, label='9339 rpm, 3.24 N·m'),
       'OP3': dict(n=2000, T=None, label='2000 rpm, maximum torque')}
DESIGNS = [('Copper (Design9795_1)', None), ('Recommended', 'Al F2 L66 N10 4x1.70'), ('Cooler alternative', 'Al F2 L72 N9 5x1.60'),
           ('Least range loss', 'Al F1 L78 N8 6x1.50'), ('Drop-in', 'Al drop-in (SWG 11 size, 64 mm)')]
NI, NG = 481, 481
I_ = np.linspace(0, L.I_PATH_PK, NI)[1:]; G_ = np.radians(np.linspace(0, 90, NG))
II, GG = np.meshgrid(I_, G_, indexing='ij')
GRID = (np.ascontiguousarray(-II.ravel() * np.sin(GG.ravel())), np.ascontiguousarray(II.ravel() * np.cos(GG.ravel())))


def make(lab):
    if lab is None:
        return S.copper()
    d = {r['label']: r for r in json.load(open(os.path.join(H, 'alu9795_search.json')))['designs']}[lab]
    return S.Design(d['metal'], d['sv'], d['Nc'], d['ns'], d['d'], label=lab, Lmm=d['L'])


def op(d, o):
    R = d.evaluate(o['n'], G=GRID)
    feas = d.feasible(R)
    if o['T'] is None:
        k = np.flatnonzero(feas)[np.argmax(R['psh'][feas])] if o['n'] > 5000 else np.flatnonzero(feas)[np.argmax(R['Tsh'][feas])]
    else:
        idx = np.flatnonzero(feas & (R['Tsh'] >= o['T']))
        k = idx[np.argmin(R['Ipk'][idx] + 1e-3 * (R['Tsh'][idx] - o['T']))]
    g = lambda q: float(R[q][k])
    Ipk = g('Ipk')
    out = dict(speed=o['n'], T_shaft=g('Tsh'), T_em=g('Tem'), P_shaft=g('psh'), P_in=g('pin'), eff=g('eff'), Idc=g('Idc'),
               I_line_rms=g('I_line'), I_phase_rms=2 * Ipk / math.sqrt(2), I_strand_path_pk=Ipk, Id_phase_pk=2 * g('id'), Iq_phase_pk=2 * g('iq'),
               phase_advance=math.degrees(math.atan2(-g('id'), g('iq'))), V_phase_pk=g('v'), V_limit=L.V_MAX,
               psid_mVs=1e3 * g('psid'), psiq_mVs=1e3 * g('psiq'),
               pf=(g('vd') * g('id') + g('vq') * g('iq')) / (g('v') * Ipk),
               J=Ipk / math.sqrt(2) / (d.A * 1e6), f_el=o['n'] / 60 * L.P,
               loss=dict(cu_dc=g('pcu'), cu_ac=g('pac'), st_hyst=g('pfe_st_h'), st_eddy=g('pfe_st_e'), rot_hyst=g('pfe_rot_h'), rot_eddy=g('pfe_rot_e'),
                         iron=g('pfe'), mech=g('pme'), total=g('loss')))
    out['limit'] = ', '.join(n for n, c in (('voltage', g('v') > L.V_MAX - 0.3), ('DC current', g('pin') > L.P_DC_MAX - 30),
                                            ('360 A', Ipk > L.I_PATH_PK - 0.5)) if c) or '—'
    return out


def mcad():
    e = json.load(open(os.path.join(H, 'lab_compare_9795.json')))['mcad_envelope']
    out = {}
    for name, n in (('OP3', 2000), ('OP1', 9000)):
        k = e['Speed'].index(n)
        x = {q: e[q][k] for q in e}
        out[name] = dict(speed=n, T_shaft=x['Shaft_Torque'], T_em=x['Electromagnetic_Torque'], P_shaft=x['Shaft_Power'],
                         P_in=x['Shaft_Power'] + x['Total_Loss'], eff=x['Efficiency'], Idc=x['Idc'], I_line_rms=x['Stator_Current_Line_RMS'],
                         I_phase_rms=x['Stator_Current_Line_RMS'] / math.sqrt(3), Id_phase_pk=x['Id_Peak'], Iq_phase_pk=x['Iq_Peak'],
                         phase_advance=x['Phase_Advance'], V_phase_pk=x['Voltage_Phase_Peak'], psid_mVs=x['Flux_Linkage_D'], psiq_mVs=x['Flux_Linkage_Q'],
                         pf=x['Power_Factor'], loss=dict(cu_dc=x['Stator_Copper_Loss_DC'], cu_ac=x['Stator_Copper_Loss_AC'], st=x['Iron_Loss_Stator'],
                                                         rot=x['Iron_Loss_Rotor'], hyst=x['Iron_Loss_Hysteresis'], eddy=x['Iron_Loss_Eddy'],
                                                         iron=x['Iron_Loss'], mech=x['Mechanical_Loss'], total=x['Total_Loss']))
    return out


if __name__ == '__main__':
    res = dict(ops={k: v['label'] for k, v in OPS.items()}, designs={}, mcad_copper=mcad())
    for name, lab in DESIGNS:
        d = make(lab)
        res['designs'][name] = dict(label=d.label, ops={k: op(d, o) for k, o in OPS.items()})
        for k, r in res['designs'][name]['ops'].items():
            l = r['loss']
            print('%-22s %s T %6.2f P %6.0f eff %5.2f Iline %5.1f adv %5.1f V %5.2f pf %.3f | dc %6.1f ac %6.1f st %5.1f/%5.1f rot %5.1f mech %5.1f tot %6.1f  %s' % (
                name, k, r['T_shaft'], r['P_shaft'], r['eff'], r['I_line_rms'], r['phase_advance'], r['V_phase_pk'], r['pf'], l['cu_dc'], l['cu_ac'],
                l['st_hyst'], l['st_eddy'], l['rot_hyst'] + l['rot_eddy'], l['mech'], l['total'], r['limit']), flush=True)
    for k, r in res['mcad_copper'].items():
        l = r['loss']
        print('Motor-CAD copper       %s T %6.2f P %6.0f eff %5.2f Iline %5.1f adv %5.1f V %5.2f pf %.3f | dc %6.1f ac %6.1f st %5.1f rot %5.1f mech %5.1f tot %6.1f' % (
            k, r['T_shaft'], r['P_shaft'], r['eff'], r['I_line_rms'], r['phase_advance'], r['V_phase_pk'], r['pf'], l['cu_dc'], l['cu_ac'], l['st'], l['rot'], l['mech'], l['total']))
    json.dump(res, open(os.path.join(H, 'alu9795_ops.json'), 'w'), indent=1)
