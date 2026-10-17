"""Half-machine (SYM=1) model against the full model: no-load (hires_100C.json) and loaded points (work/dq100_partial.json).
python sym_check.py -> sym_check.json, work/sym_model.png"""
import json, os, time
os.environ['SYM'] = '1'
os.environ['MAG_T'] = '100'
import numpy as np
import backemf as M
import lab_map as L
import femm

H = M.HERE

if __name__ == '__main__':
    femm.openfemm(1)
    import runguard as RG; RG.register_femm()
    fn = os.path.join(H, 'work', 'sym_check.fem'); M.W['file'] = fn
    t0 = time.time(); M.build(fn, True); print('built %.1f s, cut segments %s' % (time.time() - t0, [tuple(round(v, 3) for v in c) for c in M.HALF_CUT]), flush=True)
    femm.mi_zoomnatural(); femm.mi_savebitmap(os.path.join(H, 'work', 'sym_model.png'))
    out = dict(noload=[], loaded=[])
    hr = json.load(open(os.path.join(H, 'hires_100C.json')))
    for th in (0.0, 3.0, 7.0, 11.0):
        t0 = time.time(); _, psi, T = M.job(th); dt = time.time() - t0
        i = hr['theta_m'].index(th); ref = np.array(hr['coil_psi'][i])
        e = float(np.max(np.abs(np.array(psi) - ref)) / np.max(np.abs(ref)) * 100)
        out['noload'].append(dict(th=th, T=T, T_full=hr['emf_torque'][i], psi_err_pct=e, s=dt))
        print('no-load th %4.1f  T %.3f (full %.3f)  coil psi max err %.3f %% of peak  %.1f s' % (th, T, hr['emf_torque'][i], e, dt), flush=True)
    full = json.load(open(os.path.join(H, 'work', 'dq100_partial.json')))
    for r in [full[k] for k in (21, 23, 30, 45, 46)]:
        t0 = time.time(); s = L.job((r['th'], r['id'], r['iq'])); dt = time.time() - t0
        d = dict(th=r['th'], id=r['id'], iq=r['iq'], T=s['T'], T_full=r['T'], s=dt,
                 **{k: (s[k], r[k]) for k in ('st_S2', 'st_Sa', 'cu_B2', 'tooth_B')},
                 psi=(s['psi_ph'], r['psi_ph']))
        out['loaded'].append(d)
        print('id %5.0f iq %5.0f th %4.1f  T %.3f (full %.3f, %+.2f %%)  psi Ph1 %.5f (%.5f)  st_Sa %.2f (%.2f)  cuB2 %.4f (%.4f)  %.1f s' % (
            r['id'], r['iq'], r['th'], s['T'], r['T'], 100 * (s['T'] / r['T'] - 1) if abs(r['T']) > 1e-3 else 0, s['psi_ph']['Ph1'], r['psi_ph']['Ph1'],
            s['st_Sa'], r['st_Sa'], s['cu_B2'], r['cu_B2'], dt), flush=True)
    json.dump(out, open(os.path.join(H, 'sym_check.json'), 'w'), indent=1)
    femm.closefemm()
