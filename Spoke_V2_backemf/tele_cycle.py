"""Telematics drive cycle (telematics_cycle.txt) on the motor maps: copper vs aluminium, motoring + regeneration, range.

File columns: time s | motor torque N.m (shaft; negative = regen) | wheel speed rpm | flag | flag.
Each logged sample = 1 s (30 logging gaps are skipped, not filled). Motor speed = wheel rpm x 13.9.
Motoring: battery-side motor input = T*w + loss(T, n). Regen: electrical output = |T|*w - loss(|T|, n)
(losses at the same |T| as motoring, by symmetry of the PM machine). Below 0.5 N.m the 0.5 N.m loss is used.
Motor below 50 rpm: no power, no loss. Inverter and battery losses are not included.

python tele_cycle.py            -> main designs        python tele_cycle.py search  -> every design in alu_sweep.json
"""
import json, math, os, sys, time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import alu_study as A

H = A.H
RATIO, RW = 13.9, 0.30
X = np.loadtxt(os.path.join(H, 'telematics_cycle.txt'))
T_CY, NW = X[:, 1], np.maximum(X[:, 2], 0.0)
N_CY = NW * RATIO
DIST_KM = float(np.sum(NW * 2 * math.pi * RW / 60) / 1000)          # 1 s per sample
SP = np.arange(0, 7601, 100.0); TQ = np.arange(0.5, 30.01, 0.5)


class Map:
    def __init__(self, d, npts=241):
        env, mp = d.run(SP, TQ, npts=npts, keys=('loss', 'pcu', 'pac', 'pfe', 'pme'))
        self.tmax = np.array([e['Tsh'] if e else 0.0 for e in env])
        self.f = {}
        for k in ('loss', 'pcu', 'pac', 'pfe', 'pme'):
            Z = np.array(mp[k], float)
            for j in range(Z.shape[1]):
                c = Z[:, j]; ok = ~np.isnan(c)
                if ok.any():
                    c[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), c[ok])
            Z[:, 0] = Z[:, 1]
            self.f[k] = RegularGridInterpolator((TQ, SP), Z, bounds_error=False, fill_value=None)

    def cycle(self, T=T_CY, n=N_CY):
        on = n >= 50
        w = n * math.pi / 30
        tl = np.interp(n, SP, self.tmax)
        Tc = np.clip(T, -tl, tl)
        q = np.c_[np.clip(np.abs(Tc), TQ[0], TQ[-1]), np.clip(n, SP[1], SP[-1])]
        L = {k: np.where(on, f(q), 0.0) for k, f in self.f.items()}
        Pm = np.where(on, Tc * w, 0.0)
        Pe = Pm + L['loss']                              # battery-side (motor terminals) power, both directions
        mot, gen = Pm > 0, Pm < 0
        Wh = lambda a: float(np.sum(a)) / 3600
        r = dict(E_shaft_mot=Wh(Pm[mot]), E_in_mot=Wh(Pe[mot]), E_shaft_gen=Wh(-Pm[gen]), E_out_gen=Wh(-Pe[gen]),
                 E_net=Wh(Pe), loss={k: Wh(v) for k, v in L.items()}, n_short=int(np.sum(np.abs(T) > tl + 1e-9)))
        r['eff_mot'] = 100 * r['E_shaft_mot'] / r['E_in_mot']
        r['eff_gen'] = 100 * r['E_out_gen'] / r['E_shaft_gen']
        r['eff_cycle'] = 100 * (r['E_shaft_mot'] - r['E_shaft_gen']) / r['E_net']     # net mechanical / net electrical
        r['Wh_km'] = r['E_net'] / DIST_KM
        return r


MAIN = [('Copper, M235 (today)', lambda: A.baseline()),
        ('Al recommended 81 mm, 0.20 mm', lambda: A.Design('Al', 'S3', 81.0, 16, 3, 2.053, lam='NO20')),
        ('Al recommended 81 mm, M235', lambda: A.Design('Al', 'S3', 81.0, 16, 3, 2.053, lam='M235-35A')),
        ('Al drop-in 63 mm, M235', lambda: A.Design('Al', 'S0', 63.0, 20, 1, 2.9464)),
        ('Copper, 0.20 mm steel', lambda: A.Design('Cu', 'S0', 63.0, 20, 1, 2.9464, lam='NO20'))]


def design(r):
    if r['label'].startswith('Al drop-in'):
        return A.Design('Al', 'S0', 63.0, 20, 1, 2.9464)
    if r['label'].startswith('Copper baseline'):
        return A.baseline()
    return A.Design(r['metal'], r['sv'], r['L'], r['Nph'], r['ns'], r['d'], lam=r.get('lam') or 'M235-35A')


def job(r):
    return r['label'], Map(design(r), npts=121).cycle()


if __name__ == '__main__':
    mot = T_CY > 0
    print('samples %d, distance %.2f km, motor speed max %.0f rpm, T max %.1f / min %.1f N.m, motoring %d s, regen %d s' % (
        len(T_CY), DIST_KM, N_CY.max(), T_CY.max(), T_CY.min(), np.sum(mot & (N_CY >= 50)), np.sum((T_CY < 0) & (N_CY >= 50))))
    if len(sys.argv) > 1 and sys.argv[1] == 'search':
        from multiprocessing import Pool
        t0 = time.time()
        R = json.load(open(os.path.join(H, 'alu_sweep.json')))
        out = {}
        with Pool(7) as p:
            for k, (lab, res) in enumerate(p.imap_unordered(job, R, chunksize=4), 1):
                out[lab] = res
                if k % 200 == 0:
                    print(k, '%.0f s' % (time.time() - t0), flush=True)
        json.dump(out, open(os.path.join(H, 'tele_search.json'), 'w'))
        print('done %.0f s' % (time.time() - t0))
    else:
        res = {}
        for name, mk in MAIN:
            r = res[name] = Map(mk()).cycle()
            print('%-30s cycle %.2f %%  motoring %.2f %%  regen %.2f %%  net %.1f Wh  %.2f Wh/km  losses %s  short %d' % (
                name, r['eff_cycle'], r['eff_mot'], r['eff_gen'], r['E_net'], r['Wh_km'], {k: round(v, 1) for k, v in r['loss'].items()}, r['n_short']), flush=True)
        json.dump(dict(distance_km=DIST_KM, results=res), open(os.path.join(H, 'tele_result.json'), 'w'), indent=1)
