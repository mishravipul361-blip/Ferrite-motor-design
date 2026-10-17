"""Comparative demagnetisation check of the near-air-gap rotor change: cold magnets (-20 C), full 360 A rms on the -d axis
(worst case, all current demagnetising) and the 2000 rpm peak point (gamma 23 deg).  Field in each magnet element
resolved along its magnetisation: B_par; internal H = (B_par - Br) / (mu0 mu_r).  Reported: min B_par, 1st percentile,
magnet area share below 0.10 T, most negative H.  python demag_check.py   (ALU variants from alu_femm.DESIGNS)"""
import json, math, os
os.environ.setdefault('ALU', 'S0,Q1,Q2')
import numpy as np
import runguard as RG
import sens as S
import backemf as M
import lab_map as L
import femm

T_COLD = -20.0
BR_COLD = M.BR20 * (1 + M.BR_TC * (T_COLD - 20))


def magnet_field(fn):
    s = open(fn).read().split('\n')
    l0 = next(i for i, l in enumerate(s) if l.startswith('[NumBlockLabels]'))
    labs = [l.split() for l in s[l0 + 1:l0 + 1 + int(s[l0].split('=')[1])]]
    i0 = next(i for i, l in enumerate(s) if l.startswith('[BlockProps]'))
    names = [l.split('=')[1].strip().strip('"') for l in s[i0:l0] if l.strip().startswith('<BlockName>')]
    k = s.index('[Solution]'); n = int(s[k + 1])
    nodes = np.array([[float(x) for x in l.split()[:3]] for l in s[k + 2:k + 2 + n]])
    ne = int(s[k + 2 + n])
    el = np.array([[int(x) for x in l.split()[:4]] for l in s[k + 3 + n:k + 3 + n + ne]])
    x, y, A = nodes[:, 0] / 1e3, nodes[:, 1] / 1e3, nodes[:, 2]
    i, j, kk = el[:, 0], el[:, 1], el[:, 2]
    b = np.stack([y[j] - y[kk], y[kk] - y[i], y[i] - y[j]], 1); c = np.stack([x[kk] - x[j], x[i] - x[kk], x[j] - x[i]], 1)
    a2 = x[i] * b[:, 0] + x[j] * b[:, 1] + x[kk] * b[:, 2]
    Ae = np.stack([A[i], A[j], A[kk]], 1)
    Bx, By = (c * Ae).sum(1) / a2, -(b * Ae).sum(1) / a2
    area = np.abs(a2) / 2 * 1e6
    lab = [labs[l] for l in el[:, 3]]
    mag = np.array([names[int(q[2]) - 1] == 'Magnet' for q in lab])
    mdir = np.radians(np.array([float(q[5]) if names[int(q[2]) - 1] == 'Magnet' else 0.0 for q in lab]))
    Bp = Bx * np.cos(mdir) + By * np.sin(mdir)
    return Bp[mag], area[mag]


def _reset():
    try:
        femm.closefemm()
    except Exception:
        pass
    S.W.pop('pid', None); S.W['variant'] = None


@RG.femm_retry(_reset)
def run(v, i_d, i_q, th):
    S.ensure(v); S.set_br(BR_COLD); M.set_angle(th)
    ip = L.phase_currents(th, i_d, i_q)
    for ph, teeth in M.PHASE_TEETH.items():
        for t in teeth:
            femm.mi_modifycircprop('T%d' % t, 1, L.SIGN[ph][t] * ip[ph])
    femm.mi_analyze(1)
    Bp, ar = magnet_field(S.W['file'].replace('.fem', '.ans'))
    H = (Bp - BR_COLD) / (M.MU0 * M.MU_MAG)
    o = np.argsort(Bp); cum = np.cumsum(ar[o]) / ar.sum()
    return dict(Bmin=float(Bp.min()), B_p1=float(Bp[o][np.searchsorted(cum, 0.01)]), share_below_0p1=float(ar[Bp < 0.10].sum() / ar.sum()),
                Hmin_kA_m=float(H.min() / 1e3), Bmean=float(np.sum(Bp * ar) / ar.sum()))


if __name__ == '__main__':
    RG.keep_awake()
    out = {}
    I = S.I360; g = math.radians(23.0)
    cases = {'360 A on -d axis': (-I, 0.0), 'peak point 360 A, gamma 23 deg': (-I * math.sin(g), I * math.cos(g)), 'no load': (0.0, 0.0)}
    for v, *_ in S.VARIANTS:
        for cn, (i_d, i_q) in cases.items():
            for th in (0.0, 3.0):
                r = run(v, i_d, i_q, th)
                key = '%s | %s' % (v, cn)
                if key not in out or r['Bmin'] < out[key]['Bmin']:
                    out[key] = r
                print('%-10s %-32s th %.0f  Bmin %+.3f T  B(1%%) %+.3f T  area<0.1T %.2f %%  Hmin %.0f kA/m  Bmean %.3f' % (
                    v, cn, th, r['Bmin'], r['B_p1'], 100 * r['share_below_0p1'], r['Hmin_kA_m'], r['Bmean']), flush=True)
    fn = os.path.join(S.H, 'demag_check.json')                 # merge with earlier variants instead of overwriting them
    if os.path.exists(fn):
        out = dict(json.load(open(fn))['results'], **out)
    RG.dump_atomic(dict(T_magnet=T_COLD, Br=BR_COLD, results=out), os.path.join(S.H, 'demag_check.json'), indent=1)
