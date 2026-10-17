"""Aluminium winding for Design9795_1: Pareto fronts (cost saving vs range) per stator OD and performance criterion.

Criteria against the copper motor (same drive limits):
  full     peak torque (1000 rpm) >= 99.5 % and envelope >= 99 % at every speed
  relaxed  peak torque >= 99.5 % and envelope >= 97 % at every speed (shortfall only in the DC-power-limited region)
Thermal check (reported, not a filter): winding loss at peak torque, 1000 rpm, against copper's.
Price sensitivity: aluminium wire 230 / 370 / 450 INR/kg.

python alu9795_analyse.py   -> alu9795_picks.json
"""
import json, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(H, os.environ.get('IN', 'alu9795_search.json'))))
CU = R['copper']; DS = R['designs']; PRICE = R['price']
CY = 'Telematics'
CRIT = {'full': (0.995, 0.99), 'relaxed': (0.995, 0.97)}
ODS = (160, 163, 165, 167, 170)
AL_PRICES = (230.0, 370.0, 450.0)


def ok(d, c):
    pk, env = CRIT[c]
    return d['peak']['T'] >= pk * CU['peak']['T'] and d['env_ratio_min'] >= env


def dcost(d, p_al=PRICE['Al']):
    """Cost change against copper with aluminium wire at p_al INR/kg."""
    c = d['cost']
    wire = c['wire'] * p_al / PRICE['Al']
    return wire + c['steel'] + c['magnet'] - CU['cost']['total']


def pareto(ds, p_al=PRICE['Al']):
    """Non-dominated set: lower cost change and higher range change."""
    s = sorted(ds, key=lambda d: (dcost(d, p_al), -d['range_pct'][CY]))
    out, best = [], -1e9
    for d in s:
        if d['range_pct'][CY] > best + 1e-6:
            out.append(d); best = d['range_pct'][CY]
    return out


def brief(d, p_al=PRICE['Al']):
    return dict(label=d['label'], sv=d['sv'], od=d['geo']['od'], L=d['L'], Nc=d['Nc'], ns=d['ns'], d=d['d'], dcost=dcost(d, p_al),
                range=d['range_pct'][CY], range_wmtc=d['range_pct']['WMTC Class 1'], env=d['env_ratio_min'], Tpk=d['peak']['T'],
                pwdg=d['peak']['pcu'] + d['peak']['pac'], dmass=d['dmass'], peak_eff=d['peak_eff'], dEta=d['map_dEta_mean'])


def main():
    out = dict(copper=CU, fronts={}, picks={})
    for c in CRIT:
        for od in ODS + ('all',):
            ds = [d for d in DS if ok(d, c) and (od == 'all' or abs(d['geo']['od'] - od) < 0.5)]
            for p in AL_PRICES:
                out['fronts']['%s|%s|%.0f' % (c, od, p)] = [brief(d, p) for d in pareto(ds, p)]
    # picks per OD (relaxed and full, Al 370): best range; cheapest within 0.5 / 1 / 2 % range loss; cheapest overall
    for c in CRIT:
        for od in ODS + ('all',):
            f = out['fronts']['%s|%s|370' % (c, od)]
            if not f:
                continue
            pk = {'best_range': max(f, key=lambda d: d['range'])}
            for lim in (0.5, 1.0, 1.5, 2.0):
                g = [d for d in f if d['range'] >= -lim]
                if g:
                    pk['cheapest_within_%.1f' % lim] = min(g, key=lambda d: d['dcost'])
            pk['cheapest'] = min(f, key=lambda d: d['dcost'])
            out['picks']['%s|%s' % (c, od)] = pk
    json.dump(out, open(os.path.join(H, os.environ.get('PICKS', 'alu9795_picks.json')), 'w'), indent=1, default=float)
    print('copper: T %.2f N.m, winding loss %.0f W, cost %.0f INR, %.3f Wh/km' % (CU['peak']['T'], CU['peak']['pcu'] + CU['peak']['pac'],
          CU['cost']['total'], CU['cycle'][CY]['Wh_km']))
    for k, pk in out['picks'].items():
        print('==', k)
        for n, d in pk.items():
            print('  %-22s %-24s dcost %+6.0f range %+5.2f %% (WMTC %+5.2f) env %.3f T %.2f Pw %4.0f W dm %+5.2f kg dEta %+.2f' % (
                n, d['label'], d['dcost'], d['range'], d['range_wmtc'], d['env'], d['Tpk'], d['pwdg'], d['dmass'], d['dEta']))


if __name__ == '__main__':
    main()
