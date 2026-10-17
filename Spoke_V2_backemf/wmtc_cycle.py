"""WMTC drive-cycle efficiency of the copper design and aluminium designs, from the scaled Lab model maps (alu_study.py).

Vehicle (user): GVW 757 kg, CdA 1.117 m2, wheel radius 0.30 m, gear ratio 13.9.
Assumed: Crr 0.015, gearbox efficiency 0.95, rotating-mass factor 1.05, air 1.2 kg/m3, flat road.
Traction only: braking is not regenerated (map is motoring only); motor efficiency = shaft energy / electrical energy
over all seconds with positive wheel force. Losses below 0.5 N.m use the 0.5 N.m loss (conservative).
Cycle: UN GTR No.2 Amend.1 Annex 5 tables (wmtc_cycles.json).

python wmtc_cycle.py            -> prints results, writes wmtc_result.json
"""
import json, math, os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import alu_study as A

H = A.H
VEH = dict(m=757.0, cda=1.117, r=0.30, i=13.9, crr=0.015, eta_g=0.95, kr=1.05, rho=1.2)
CY = json.load(open(os.path.join(H, 'wmtc_cycles.json')))['parts']
CLASSES = {'Class 1': ['P1R', 'P1R'], 'Class 2-1': ['P1R', 'P2R'], 'Class 2-2': ['P1', 'P2']}
SP = np.arange(0, 12001, 100.0); TQ = np.arange(0.5, 60.01, 0.5)


def trace(parts):
    v = []
    for p in parts:
        x = np.array([np.nan if s is None else s for s in CY[p]], float)
        k = np.arange(len(x)); bad = np.isnan(x)
        x[bad] = np.interp(k[bad], k[~bad], x[~bad])
        v += [0.0] + list(x)                     # t = 0 s at standstill, then 1..600 s
    return np.array(v) / 3.6


def demand(v, veh=VEH):
    """Per second: motor speed rpm, motor torque N.m (traction only), using mean speed and the speed change in each second."""
    vm, a = (v[1:] + v[:-1]) / 2, np.diff(v)
    F = veh['kr'] * veh['m'] * a + veh['m'] * 9.81 * veh['crr'] * (vm > 0.05) + 0.5 * veh['rho'] * veh['cda'] * vm ** 2
    n = vm / veh['r'] * 60 / (2 * math.pi) * veh['i']
    T = np.where(F > 0, F * veh['r'] / (veh['i'] * veh['eta_g']), 0.0)
    return n, T


class MapEval:
    def __init__(self, d, sp=SP, tq=TQ, npts=241):
        env, mp = d.run(sp, tq, npts=npts, keys=('eff', 'loss'))
        self.loss = np.array(mp['loss'], float)                 # (tq, sp)
        self.tmax = np.array([e['Tsh'] if e else 0.0 for e in env])
        L = self.loss.copy()
        for j in range(L.shape[1]):                            # fill outside-envelope cells with the last valid loss (used only when clipped)
            col = L[:, j]; ok = ~np.isnan(col)
            if ok.any():
                col[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), col[ok]); L[:, j] = col
        L[:, np.isnan(L).all(0)] = np.nan
        if np.isnan(L[:, 0]).all():
            L[:, 0] = L[:, 1]
        self.f = RegularGridInterpolator((tq, sp), L, bounds_error=False, fill_value=None)
        self.sp, self.tq = sp, tq

    def cycle(self, n, T):
        tr = T > 0
        n_, T_ = np.clip(n[tr], self.sp[1], self.sp[-1]), T[tr]
        tlim = np.interp(n_, self.sp, self.tmax)
        short = T_ > tlim + 1e-9
        Tc = np.minimum(T_, tlim)
        Pm = Tc * n_ * math.pi / 30
        loss = self.f(np.c_[np.clip(Tc, self.tq[0], self.tq[-1]), n_])
        return dict(E_shaft=float(Pm.sum()) / 3600, E_in=float((Pm + loss).sum()) / 3600, E_loss=float(loss.sum()) / 3600,
                    eff=float(100 * Pm.sum() / (Pm + loss).sum()), n_short=int(short.sum()),
                    E_short=float(((T_ - Tc) * n_ * math.pi / 30)[short].sum()) / 3600, t_trac=int(tr.sum()))


DESIGNS = [('Copper, M235 (today)', lambda: A.baseline()),
           ('Al recommended, 0.20 mm', lambda: A.Design('Al', 'S3', 81.0, 16, 3, 2.053, lam='NO20')),
           ('Al recommended shape, M235', lambda: A.Design('Al', 'S3', 81.0, 16, 3, 2.053, lam='M235-35A')),
           ('Copper, 0.20 mm steel', lambda: A.Design('Cu', 'S0', 63.0, 20, 1, 2.9464, lam='NO20')),
           ('Al drop-in, M235', lambda: A.Design('Al', 'S0', 63.0, 20, 1, 2.9464))]

if __name__ == '__main__':
    res = {}
    dem = {c: demand(trace(p)) for c, p in CLASSES.items()}
    for c, (n, T) in dem.items():
        tr = T > 0
        print('%s: %d s, traction %d s, motor n max %.0f rpm, T max %.1f N.m, T mean (traction) %.1f N.m, n mean %.0f rpm' % (
            c, len(n), tr.sum(), n.max(), T.max(), T[tr].mean(), n[tr].mean()))
    for name, mk in DESIGNS:
        me = MapEval(mk())
        res[name] = {c: me.cycle(*dem[c]) for c in CLASSES}
        print('%-28s' % name, '  '.join('%s %.2f %% (loss %.1f Wh, short %d s)' % (c, r['eff'], r['E_loss'], r['n_short']) for c, r in res[name].items()), flush=True)
    json.dump(dict(vehicle=VEH, results=res), open(os.path.join(H, 'wmtc_result.json'), 'w'), indent=1)
