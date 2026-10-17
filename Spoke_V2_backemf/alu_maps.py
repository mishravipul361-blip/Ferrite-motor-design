"""Flux maps and demagnetisation of every aluminium-study geometry (alu_femm.DESIGNS, copper geometry = S0).
Per design (rotor at 0 deg; demag cases also at 3 deg):
  noload    magnets 100 C, no current                       -> flux map
  peak      magnets 100 C, 360 A rms at gamma 20 deg el      -> flux map (2000 rpm peak point)
  demag_d   magnets -20 C, 360 A rms all on -d axis          -> demag statistics + map (worst case)
  demag_pk  magnets -20 C, 360 A rms at gamma 23 deg         -> demag statistics
Currents are the copper path currents (Nph 20): the same ampere-turns as any 10 turns/coil design at 360 A;
designs with 9 / 8 turns per coil see 90 % / 80 % of this field at 360 A.
Solutions are kept in work/maps/<design>_<case>_<th>.ans.   ALU=S0,Q1 MAPS_OUT=work/maps/stats_x.json python alu_maps.py"""
import json, math, os, shutil
import numpy as np
import runguard as RG
import sens as S
import backemf as M
import lab_map as L
import demag_check as D
import femm

OUT = os.path.join(S.H, 'work', 'maps')
BR100 = S.BR100


def _reset():
    try:
        femm.closefemm()
    except Exception:
        pass
    S.W.pop('pid', None); S.W['variant'] = None


@RG.femm_retry(_reset)
def solve(v, br, i_d, i_q, th, tag):
    S.ensure(v); S.set_br(br); M.set_angle(th)
    ip = L.phase_currents(th, i_d, i_q)
    for ph, teeth in M.PHASE_TEETH.items():
        for t in teeth:
            femm.mi_modifycircprop('T%d' % t, 1, L.SIGN[ph][t] * ip[ph])
    femm.mi_analyze(1)
    dst = os.path.join(OUT, '%s_%s.ans' % (v[4:], tag))
    shutil.copyfile(S.W['file'].replace('.fem', '.ans'), dst)
    return dst


def demag_stats(fn, br):
    Bp, ar = D.magnet_field(fn)
    H = (Bp - br) / (M.MU0 * M.MU_MAG)
    o = np.argsort(Bp); cum = np.cumsum(ar[o]) / ar.sum()
    return dict(Bmin=float(Bp.min()), B_p1=float(Bp[o][np.searchsorted(cum, 0.01)]), share_below_0p1=float(ar[Bp < 0.10].sum() / ar.sum()),
                share_below_0=float(ar[Bp < 0.0].sum() / ar.sum()), Hmin_kA_m=float(H.min() / 1e3), Bmean=float(np.sum(Bp * ar) / ar.sum()))


if __name__ == '__main__':
    RG.keep_awake()
    os.makedirs(OUT, exist_ok=True)
    fo = os.path.join(S.H, os.environ.get('MAPS_OUT', 'work/maps/stats.json'))
    res = json.load(open(fo)) if os.path.exists(fo) else {}
    I = S.I360
    for v, *_ in S.VARIANTS:
        if v[4:] in res and res[v[4:]].get('complete'):
            continue
        r = {}
        g = math.radians(20.0)
        solve(v, BR100, 0.0, 0.0, 0.0, 'noload_0')
        solve(v, BR100, -I * math.sin(g), I * math.cos(g), 0.0, 'peak_0')
        for case, gam in (('demag_d', 90.0), ('demag_pk', 23.0)):
            g = math.radians(gam)
            st = {}
            for th in (0.0, 3.0):
                fn = solve(v, D.BR_COLD, -I * math.sin(g), I * math.cos(g), th, '%s_%d' % (case, th))
                st['%d' % th] = demag_stats(fn, D.BR_COLD)
            r[case] = st
        r['complete'] = True
        res[v[4:]] = r
        RG.dump_atomic(res, fo, indent=1)
        print(v, 'done  demag_d 1%%-B %.3f / %.3f T  area<0.1T %.2f / %.2f %%' % (
            r['demag_d']['0']['B_p1'], r['demag_d']['3']['B_p1'], 100 * r['demag_d']['0']['share_below_0p1'], 100 * r['demag_d']['3']['share_below_0p1']), flush=True)
    try:
        femm.closefemm()
    except Exception:
        pass
