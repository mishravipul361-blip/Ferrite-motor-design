"""FEMM verification of the aluminium candidates (sens.py ALU runs): peak torque at 360 A, back-EMF THD, torque ripple at
OP1/OP2/OP3 (same path currents as copper: Nph 20, 63 mm), and the effect of rotor step skew.  python alu_verify.py ALU_S0 ALU_Q1 ..."""
import json, os, sys
import numpy as np
import sens_analyse as SA

if __name__ == '__main__':
    out = {}
    for v in sys.argv[1:]:
        d = SA.design(v)
        R = [r for r in SA.cache if r['v'] == v]
        if d is None or len([r for r in R if r['kind'] in ('rip2', 'rip3')]) < 24:
            print(v, 'incomplete (%d solves)' % len(R)); continue
        sg = 1 if np.mean([r['T'] for r in R if r['kind'] == 'tq']) > 0 else -1
        wav = {k: [sg * r['T'] for r in sorted([r for r in R if r['kind'] == k], key=lambda r: r['th'])] for k in ('rip', 'rip2', 'rip3')}
        row = dict(T360=d['T360'], gamma=d['gamma_opt'], skew={})
        for s in (0.0, 2.0, 3.0, 4.0):
            for mode, st in (('cont', None), ('3step', 3)):
                if s == 0 and mode == '3step':
                    continue
                k = '%.0f deg %s' % (s, mode) if s else 'none'
                row['skew'][k] = dict(thd=SA.emf_metrics(d['psi'], s, st)['thd'], k1=SA.skew_factor(1, s, st),
                                      **{op: SA.ripple_metrics(wav[w], 1.0, s, st)['pct'] for w, op in (('rip', 'OP1'), ('rip2', 'OP2'), ('rip3', 'OP3'))})
        out[v] = row
        print('%-8s T360 %.2f N.m (gamma %.1f)' % (v, d['T360'], d['gamma_opt']))
        for k, x in row['skew'].items():
            print('    skew %-12s THD %5.2f %%  ripple OP1 %5.1f  OP2 %5.1f  OP3 %5.1f %%  torque x%.3f' % (k, x['thd'], x['OP1'], x['OP2'], x['OP3'], x['k1']))
    fn = os.path.join(SA.H, 'alu_verify.json')
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update(out); json.dump(old, open(fn, 'w'), indent=1)
