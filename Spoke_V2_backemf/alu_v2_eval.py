"""OD-limited aluminium designs (P/Q stators, alu_femm.py): merge the sweep workers, then drive-cycle energy (telematics speed
trace and WMTC Class 1, tele_range.py method), cost and current density of the short-stack candidates.
python alu_v2_eval.py  -> alu_sweep_v2.json, alu_v2_eval.json"""
import glob, json, math, os
from multiprocessing import Pool
import numpy as np
import alu_study as A

H = A.H


def merge():
    R, have = [], set()
    for fn in sorted(glob.glob(os.path.join(H, 'work', 'sw_[PQRS]*.json'))):
        for r in json.load(open(fn)):
            if r['label'] not in have:
                R.append(r); have.add(r['label'])
    json.dump(R, open(os.path.join(H, 'alu_sweep_v2.json'), 'w'))
    return R


def design(r):
    if r['label'].startswith('Copper baseline'):
        return A.baseline()
    return A.Design(r['metal'], r['sv'], r['L'], r['Nph'], r['ns'], r['d'], lam=r.get('lam') or 'M235-35A')


def cost(r, P=dict(Cu=16.5, Al=5.5, **{'M235-35A': 1.8, 'NO20': 4.0}, magnet=4.0, scrap=0.3)):
    m = r['masses']
    blank = (A.SLOTS.geo(r['sv']).get('od', 160.0) + 4.0) * 1e-3
    gross = blank ** 2 * r['L'] * 1e-3 * 7650 * 0.97
    c = dict(wire=m['winding'] * 1.05 * P[r['metal']], steel=gross * P[r.get('lam') or 'M235-35A'] - (gross - m['stator_lam'] - m['rotor_lam']) * P['scrap'],
             magnet=m['magnets'] * P['magnet'])
    c['total'] = sum(c.values())
    return c


def job(r):
    import tele_cycle as TC, tele_range as TR
    mp = TC.Map(design(r), npts=121)
    out = {}
    for c, (T, n, km) in TR.CYC.items():
        x = mp.cycle(T, n); out[c] = x['E_net'] / km
    d = design(r)
    J = 360 / math.sqrt(3) / 2 / (d.A * 1e6)                      # A/mm2 at 360 A rms line, delta, 2 paths
    return r['label'], dict(Wh_km=out, J360=J, cost=cost(r))


if __name__ == '__main__':
    R = merge()
    base = json.load(open(os.path.join(H, 'alu_sweep.json')))
    B = next(r for r in base if r['label'].startswith('Copper baseline'))
    sel = [B] + [r for r in R if r['metal'] == 'Al' and r['L'] <= float(os.environ.get('LMAX', 72)) and r['ops']['OP3'] is not None
           and r['sv'][0] in os.environ.get('SV', 'PQ')]
    if os.environ.get('PREFILTER'):                    # hard constraints first: peak torque, envelope, OP1/OP2 losses not above copper
        sel = [sel[0]] + [r for r in sel[1:] if r['env_ratio_at'].get('2000', 0) >= 0.999 and r['env_ratio_min'] >= 0.985
                          and r['dEta_ops']['OP1'] is not None and r['dEta_ops']['OP1'] >= 0 and r['dEta_ops']['OP2'] >= 0]
    out = {}
    with Pool(int(os.environ.get('NWORK', 4))) as p:
        for lab, x in p.imap_unordered(job, sel):
            out[lab] = x
    b = out[B['label']]
    for r in sel:
        x = out[r['label']]
        x['dRange'] = {c: 100 * (b['Wh_km'][c] / x['Wh_km'][c] - 1) for c in x['Wh_km']}
        x['rec'] = {k: r.get(k) for k in ('label', 'sv', 'L', 'Nph', 'ns', 'd', 'lam', 'fill', 'env_ratio_min', 'env_ratio_at', 'map_dEta_mean', 'dEta_ops', 'masses', 'dmass')}
        x['rec']['op_loss'] = {o: (r['ops'][o] or {}).get('loss') for o in r['ops']}
    json.dump(out, open(os.path.join(H, os.environ.get('EVAL_OUT', 'alu_v2_eval.json')), 'w'), indent=1)
    print('%-32s %6s %6s %6s %6s %6s %6s %6s %5s %6s %6s %7s' % ('design', 'T2000', 'envmin', 'OP1', 'OP2', 'OP3', 'tele%', 'wmtc%', 'J', 'dm kg', 'cost', 'OP2 W'))
    for r in sorted(sel, key=lambda r: -out[r['label']]['dRange']['WMTC Class 1']):
        x = out[r['label']]; o = r['dEta_ops']; f = lambda v: np.nan if v is None else v
        print('%-32s %6.3f %6.3f %+6.2f %+6.2f %+6.2f %+6.2f %+6.2f %5.1f %+6.2f %6.1f %7.1f' % (
            r['label'], r['env_ratio_at'].get('2000', np.nan), r['env_ratio_min'], f(o['OP1']), f(o['OP2']), f(o['OP3']),
            x['dRange']['Telematics speed trace'], x['dRange']['WMTC Class 1'], x['J360'], r['dmass'], x['cost']['total'], f((r['ops']['OP2'] or {}).get('loss'))))
