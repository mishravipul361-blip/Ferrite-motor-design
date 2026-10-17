"""Energy per km and relative range from a speed trace on a flat road (vehicle model), motoring + regen, per motor design.
Speed traces: telematics wheel rpm (1 s per logged sample, steps across logging gaps skipped) and WMTC Class 1.
Vehicle: GVW 757 kg, CdA 1.117, r 0.30 m, ratio 13.9 (user); Crr 0.015, gear eff 0.95, rotating mass 1.05 (assumed).
Regen: braking force recovered by the motor down to -26.4 N.m (largest regen torque in the telematics file), rest friction.
python tele_range.py -> tele_range.json"""
import json, math, os
import numpy as np
import alu_study as A
import tele_cycle as TC
import wmtc_cycle as W

V = W.VEH
TREG = -26.4


def motor_points(v, valid):
    vm, a = (v[1:] + v[:-1]) / 2, np.diff(v)
    vm, a = vm[valid], a[valid]
    F = V['kr'] * V['m'] * a + V['m'] * 9.81 * V['crr'] * (vm > 0.05) + 0.5 * V['rho'] * V['cda'] * vm ** 2
    n = vm / V['r'] * 60 / (2 * math.pi) * V['i']
    T = np.where(F > 0, F * V['r'] / (V['i'] * V['eta_g']), np.maximum(F * V['r'] * V['eta_g'] / V['i'], TREG))
    return T, n, float(np.sum(vm)) / 1000


xs = TC.X
v_tel = xs[:, 2].clip(0) * 2 * math.pi * V['r'] / 60
CYC = {'Telematics speed trace': motor_points(v_tel, np.diff(xs[:, 0]) == 1)}
v1 = W.trace(W.CLASSES['Class 1'])
CYC['WMTC Class 1'] = motor_points(v1, np.ones(len(v1) - 1, bool))

if __name__ == '__main__':
    out = {}
    for name, mk in TC.MAIN:
        mp = TC.Map(mk())
        out[name] = {}
        for c, (T, n, km) in CYC.items():
            r = mp.cycle(T, n)
            r['km'] = km; r['Wh_km'] = r['E_net'] / km
            out[name][c] = r
        print('%-30s ' % name + '  '.join('%s: %.2f Wh/km (motoring %.2f %%, regen %.2f %%, cycle %.2f %%)' % (c, r['Wh_km'], r['eff_mot'], r['eff_gen'], r['eff_cycle'])
                                         for c, r in out[name].items()), flush=True)
    json.dump(out, open(os.path.join(A.H, 'tele_range.json'), 'w'), indent=1)
