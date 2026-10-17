"""q-axis flux sensitivity (half model, 100 C, rotor at 0 deg): which model input moves psi_q by the 3-8 % that
separates FEMM from Motor-CAD Lab.   python sens_q.py -> sens_q.json"""
import json, os, time
os.environ['SYM'] = '1'
os.environ['MAG_T'] = '100'
import backemf as M
import lab_map as L
import femm

PTS = [(0.0, 30.0), (-42.49, 140.70)]          # path peak A: low-current q axis; Motor-CAD full-current point at 1000 rpm
BASE = dict(GM=M.GM, MU_MAG=M.MU_MAG, STACKING=M.STACKING)
VARIANTS = [('base', {}, True), ('shaft non-magnetic', {}, False), ('magnet mu_r 1.15', dict(MU_MAG=1.15), True),
            ('magnet mu_r 1.00', dict(MU_MAG=1.00), True), ('stacking 1.00', dict(STACKING=1.00), True), ('air-gap mesh x0.5', dict(GM=0.5), True)]

if __name__ == '__main__':
    femm.openfemm(1)
    import runguard as RG; RG.register_femm()
    fn = os.path.join(M.HERE, 'work', 'sens_q.fem'); M.W['file'] = fn
    out = []
    for name, ch, shaft in VARIANTS:
        for k, v in dict(BASE, **ch).items():
            setattr(M, k, v)
        M.build(fn, shaft)
        for i_d, i_q in PTS:
            t0 = time.time(); r = L.job((0.0, i_d, i_q))
            d, q = L.park(0.0, r['psi_ph'])
            out.append(dict(variant=name, id=i_d, iq=i_q, psid=d, psiq=q, T=r['T']))
            print('%-20s id %6.1f iq %6.1f  psid %.3f  psiq %.3f mVs  T %.2f  (%.0f s)' % (name, i_d, i_q, 1e3 * d, 1e3 * q, r['T'], time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(M.HERE, 'sens_q.json'), 'w'), indent=1)
    femm.closefemm()
