"""Open-circuit validation against Motor-CAD (Design9795_1, magnets 100 C, 1000 rpm, I = 0, 30 points per cycle,
rotor skew 3 slices at -2 / 0 / +2 mech deg, equal lengths).

FEMM: hires_100C.json (1 deg mech steps = 72 points per electrical cycle, unskewed 2D).
Skew: each slice sees the same field shifted by its angle, so the phase flux linkage is the slice average
psi_skew(th) = [psi(th - 2) + psi(th) + psi(th + 2)] / 3, done exactly on the Fourier series
(electrical harmonic k is multiplied by (1 + 2 cos(k * 5 * 2 deg)) / 3).
EMF: Motor-CAD style, 30 samples per electrical cycle and a forward difference, harmonics 2..15 for THD;
the exact spectral EMF is given as a check.
python oc_validate.py -> oc_validate.json"""
import json, math, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
P = 5
MC = dict(E_ph_pk=10.26, E_ll_pk=10.26, E_fund_pk=10.36, E_ph_rms=7.329, E_ll_rms=7.329, THD=1.247, Ke=0.0980174, Ke_fund=0.0989643,
          psi_d=19.829e-3, R_ph=6.74e-3, MLT=203.6, L_ew=1.96e-6, gap_avg=0.4956, gap_max=0.7731, tooth=1.044, tip=1.469, yoke=0.9289, rotor=1.518)
SLICES = (-2.0, 0.0, 2.0)
WM = 2 * math.pi * 1000 / 60


def metrics(c, n=30):
    """c: complex Fourier coefficients of the phase flux linkage per electrical harmonic (0..K)."""
    th_e = np.arange(n) * 2 * math.pi / n
    psi = np.real(sum(c[k] * np.exp(1j * k * th_e) for k in range(1, len(c))))
    dt = 2 * math.pi / n / (P * WM)
    e = -(np.roll(psi, -1) - psi) / dt                                   # forward difference, Motor-CAD style
    h = np.abs(np.fft.rfft(e)) / n * 2
    kmax = min(15, len(h) - 1)
    thd = 100 * math.sqrt(np.sum(h[2:kmax + 1] ** 2)) / h[1]
    th_f = np.linspace(0, 2 * math.pi, 3600, endpoint=False)              # exact EMF from the series
    e_ex = np.real(sum(-1j * k * P * WM * c[k] * np.exp(1j * k * th_f) for k in range(1, len(c))))
    return dict(E_pk=float(np.max(np.abs(e))), E_rms=float(np.sqrt(np.mean(e ** 2))), E_fund_pk=float(h[1]), THD=float(thd),
                h_pct={int(k): float(100 * h[k] / h[1]) for k in (3, 5, 7, 9, 11, 13)},
                E_pk_exact=float(np.max(np.abs(e_ex))), E_rms_exact=float(np.sqrt(np.mean(e_ex ** 2))),
                psi_fund=float(abs(c[1])), wave=e.tolist())


if __name__ == '__main__':
    d = json.load(open(os.path.join(H, 'hires_100C.json')))
    th = np.array(d['theta_m']); psi = np.array(d['path_psi']['Ph1'])
    N = len(th)
    F = np.fft.fft(psi) / N
    c0 = np.array([2 * F[k] if k else F[0] for k in range(N // 2)])     # one-sided, electrical harmonic k (72 deg = 1 cycle)
    k = np.arange(len(c0))
    kf = np.array([np.mean([np.exp(1j * kk * P * math.radians(a)) for a in SLICES]) for kk in k])
    out = {'unskewed': metrics(c0), 'skewed (MC slices)': metrics(c0 * kf)}
    for v in out.values():
        v['Ke'] = v['E_pk'] / WM; v['Ke_fund'] = v['E_fund_pk'] / WM; v['psi_m_mVs'] = 1e3 * v['E_fund_pk'] / (P * WM)
    f = json.load(open(os.path.join(H, 'oc_fields.json'))) if os.path.exists(os.path.join(H, 'oc_fields.json')) else {}
    out['fields'] = {k: v for k, v in f.items() if k != 'raw'}
    rho = 1.724e-8 * (1 + 0.003862 * 80); mlt = 2 * 0.064 + 0.075634; A = math.pi / 4 * 2.9464e-3 ** 2
    out['R_ph'] = rho * 10 * mlt / A * 2 / 2
    out['MC'] = MC
    json.dump(out, open(os.path.join(H, 'oc_validate.json'), 'w'), indent=1)
    s, u = out['skewed (MC slices)'], out['unskewed']
    rows = [('Back-EMF phase peak (V)', s['E_pk'], u['E_pk'], MC['E_ph_pk']), ('Back-EMF fundamental peak (V)', s['E_fund_pk'], u['E_fund_pk'], MC['E_fund_pk']),
            ('Back-EMF phase rms (V)', s['E_rms'], u['E_rms'], MC['E_ph_rms']), ('THD (%)', s['THD'], u['THD'], MC['THD']),
            ('Ke (V s/rad)', s['Ke'], u['Ke'], MC['Ke']), ('Ke fundamental (V s/rad)', s['Ke_fund'], u['Ke_fund'], MC['Ke_fund']),
            ('Magnet flux linkage (mV s)', s['psi_m_mVs'], u['psi_m_mVs'], 1e3 * MC['psi_d'])]
    print('%-32s %10s %10s %10s %8s' % ('', 'FEMM skew', 'FEMM 2D', 'Motor-CAD', 'diff'))
    for n, a, b, m in rows:
        print('%-32s %10.4f %10.4f %10.4f %+7.2f%%' % (n, a, b, m, 100 * (a / m - 1)))
    print('harmonics skewed %', {k: round(v, 3) for k, v in s['h_pct'].items()})
    print('harmonics 2D %', {k: round(v, 3) for k, v in u['h_pct'].items()})
    print('R phase %.5f ohm (MC %.5f)' % (out['R_ph'], MC['R_ph']))
    if out['fields']:
        for k, m in (('airgap_avg', 'gap_avg'), ('airgap_max', 'gap_max'), ('tooth_max', 'tooth'), ('tip_max', 'tip'), ('yoke_max', 'yoke'), ('rotor_max', 'rotor')):
            print('%-12s FEMM %.4f  MC %.4f  %+.1f%%' % (k, out['fields'][k], MC[m], 100 * (out['fields'][k] / MC[m] - 1)))
