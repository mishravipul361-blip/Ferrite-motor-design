"""Design IDs for the sensitivity study and export of one FEMM model per design to FEMM_models/sensitivity/.
D00 = baseline, D01..D18 = one-at-a-time variations in the order of sens.VARIANTS.
Each .fem: full geometry, materials (magnets at 100 C), 12 coil circuits at 0 A, rotor at 0 deg. Run: python sens_models.py"""
import os, re
import sens as S
import backemf as M

H = S.H
OUT = os.path.join(H, 'FEMM_models', 'sensitivity')


def winding_check(stator):
    """SWG 11 fixed (2.9464 mm bare / 3.0302 mm insulated), 20 conductors per slot, 2 parallel paths, 360 A rms line (delta).
    Slot area as Motor-CAD (validated: baseline 278.7 mm2, wire fill 0.5175)."""
    import math, numpy as np
    g = M.slot_geometry(stator)
    B, C, E, rb = g['B'], g['C'], g['E'], g['rb']
    a0, a1 = math.atan2(E[1], E[0]), math.atan2(E[1], -E[0])
    arc = [(rb * math.cos(t), rb * math.sin(t)) for t in np.linspace(a0, a1, 60)]
    poly = [B, C, E] + arc[1:-1] + [(-x, y) for x, y in reversed([B, C, E])]
    x, y = np.array(poly).T
    area = (0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) - 2 * (1 - math.pi / 4) * stator['corner_r'] ** 2) * 0.998
    a_cu = math.pi / 4 * 2.9464 ** 2; i_cond = 360 / math.sqrt(3) / 2
    return dict(slot_area=area, wire_fill=20 * math.pi / 4 * 3.0302 ** 2 / area, copper_fill=20 * a_cu / area,
                J_rms=i_cond / a_cu, J_peak=i_cond * math.sqrt(2) / a_cu)


def design_ids():
    ids = {}
    for k, (v, rot, sta) in enumerate(S.VARIANTS):
        if v == 'baseline':
            fn = 'D00_baseline_shaft30_bridge1.0'
        elif v == 'MotorCAD_original':
            ids[v] = dict(id='R00', file='R00_MotorCAD_original_shaft25_bridge1.0.fem', param='(reference: Motor-CAD original)', value=None,
                          rotor=dict(S.BASE_R, **rot), stator=dict(S.BASE_S, **sta))
            continue
        else:
            par, val = v.split('=')
            fn = 'D%02d_%s_%smm' % (k, par, ('%.2f' % float(val)).rstrip('0').rstrip('.') if '.' in ('%.2f' % float(val)) else val)
        ids[v] = dict(id='D%02d' % k, file=fn + '.fem', param=None if v == 'baseline' else v.split('=')[0],
                      value=None if v == 'baseline' else float(v.split('=')[1]), rotor=dict(S.BASE_R, **rot), stator=dict(S.BASE_S, **sta))
    for v, rot, sta in S.CANDIDATES:                           # combined candidates
        chg = '_'.join('%s%g' % (k, x) for k, x in list(sta.items()) + list(rot.items()))
        ids[v] = dict(id=v, file='%s_combined_%s.fem' % (v, chg), param='(combined candidate)', value=None,
                      rotor=dict(S.BASE_R, **rot), stator=dict(S.BASE_S, **sta))
    return ids


if __name__ == '__main__':
    import femm
    os.makedirs(OUT, exist_ok=True)
    femm.openfemm(1)
    for v, d in design_ids().items():
        S.W['variant'] = None; S.W['pid'] = os.getpid()
        M.ROTOR.clear(); M.ROTOR.update(d['rotor']); M.STATOR.clear(); M.STATOR.update(d['stator'])
        M.build(os.path.join(OUT, d['file']), True)
        print(d['id'], d['file'], flush=True)
    femm.closefemm()
