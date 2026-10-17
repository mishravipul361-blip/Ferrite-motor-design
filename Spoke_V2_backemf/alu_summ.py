import json, sys
R = json.load(open('alu_sweep.json'))
for r in R:
    r['dEta_ops'] = {o: (v if v is not None and v == v else -99.0) for o, v in r['dEta_ops'].items()}
lamof = lambda r: r.get('lam') or next((l for l in ('NO27', 'NO20') if r['label'].endswith(l)), 'M235-35A')
best = {}
for r in R[1:]:
    if r['metal'] != 'Al':
        print('REF', r['label'], 'map %+.2f' % r['map_dEta_mean'], {o: round(v, 2) for o, v in r['dEta_ops'].items()}); continue
    k = (r['sv'], lamof(r), r['Nph'], round(r['L'], 2))
    sc = min(v for v in r['dEta_ops'].values())
    if k not in best or sc > min(best[k]['dEta_ops'].values()): best[k] = r
for k in sorted(best):
    if sys.argv[1:] and k[0] not in sys.argv[1:]: continue
    r = best[k]; o = r['dEta_ops']
    ok = '*' if min(o.values()) >= -0.1 and r['env_ratio_min'] >= 0.985 else ' '
    print('%s %s %-8s N%d L%6.2f %dx%.3f env %.3f map %+.2f  OP1 %+.2f OP2 %+.2f OP3 %+.2f  m %.2f (%+.2f)' % (ok, k[0], k[1], k[2], k[3], r['ns'], r['d'], r['env_ratio_min'], r['map_dEta_mean'], o['OP1'], o['OP2'], o['OP3'], r['masses']['total'], r['dmass']))
