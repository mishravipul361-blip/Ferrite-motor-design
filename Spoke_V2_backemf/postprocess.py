"""Re-derive the EMF summaries from the coil flux linkages saved in results.json (no FEMM solve)."""
import json, math, os
import numpy as np
import backemf as M

fn = os.path.join(M.HERE, 'results.json')
res = json.load(open(fn))
for name in ('shaft_magnetic', 'shaft_nonmagnetic'):
    r = res[name]
    th = np.array(r['theta_m'])
    path, emf, _ = M.analyse(th, r['coil_psi'])
    r['path_psi'] = {k: v.tolist() for k, v in path.items()}
    r['emf_phase_1000rpm'] = {k: v.tolist() for k, v in emf.items()}
    e1, e2, e3 = emf['Ph1'], emf['Ph2'], emf['Ph3']
    ll = e1 - e2
    h, hl = M.harmonics(e1), M.harmonics(ll)
    s = r['summary']
    s.update(E_phase_rms=float(np.sqrt(np.mean(e1 ** 2))), E_phase_peak=float(np.max(np.abs(e1))),
             E_phase_fund_peak=float(h[1]), E_phase_fund_rms=float(h[1] / math.sqrt(2)),
             E_line_rms=float(np.sqrt(np.mean(ll ** 2))), E_line_peak=float(np.max(np.abs(ll))),
             E_line_fund_rms=float(hl[1] / math.sqrt(2)),
             phase_rms_all=[float(np.sqrt(np.mean(e ** 2))) for e in (e1, e2, e3)],
             THD_phase=float(np.sqrt(np.sum(h[2:12] ** 2)) / h[1]),
             THD_line=float(np.sqrt(np.sum(hl[2:12] ** 2)) / hl[1]),
             harmonics_phase_pct={int(k): float(100 * h[k] / h[1]) for k in (3, 5, 7, 9, 11)},
             harmonics_line_pct={int(k): float(100 * hl[k] / hl[1]) for k in (5, 7, 11)})
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, indent=1))
json.dump(res, open(fn, 'w'), indent=1)
M.plot(res)
