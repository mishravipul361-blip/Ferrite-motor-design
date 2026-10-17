"""Class 1 cycle efficiency with minimum-total-loss current control instead of minimum current (MTPA)."""
import numpy as np, alu_study as A, wmtc_cycle as W


def run_minloss(self, speeds, torques, npts=161, keys=('eff', 'loss')):
    G = self.grid(npts)
    mp = {k: np.full((len(torques), len(speeds)), np.nan) for k in keys}
    env = []
    for a, n in enumerate(speeds):
        R = self.evaluate(max(n, 1.0), *G, key=('grid', round(self.s, 6), npts))
        feas = self.feasible(R)
        if not feas.any():
            env.append(None); continue
        k = np.flatnonzero(feas)[np.argmax(R['Tsh'][feas])]
        env.append({kk: float(R[kk][k]) for kk in ('Tsh', 'psh', 'eff', 'loss', 'pcu', 'pac', 'pfe', 'pme', 'I_line', 'v', 'Idc')})
        for b, T in enumerate(torques):
            c = np.flatnonzero(feas & (np.abs(R['Tsh'] - T) <= 0.25))
            if len(c):
                j = c[np.argmin(R['loss'][c] - (R['Tsh'][c] - T) * max(n, 1) * np.pi / 30)]   # least input power for the torque
                for kk in keys:
                    mp[kk][b, a] = R[kk][j]
    return env, mp


if __name__ == '__main__':
    dem = {c: W.demand(W.trace(W.CLASSES[c])) for c in ('Class 1',)}
    SP = np.arange(0, 12001, 100.0); TQ = np.arange(0.5, 60.01, 0.5)
    for name, mk in W.DESIGNS[:3]:
        d = mk()
        base = W.MapEval(d, SP, TQ, npts=241).cycle(*dem['Class 1'])
        A.Design.run = run_minloss
        ml = W.MapEval(d, SP, TQ, npts=241).cycle(*dem['Class 1'])
        A.Design.run = W.A.Design.run if False else A.Design.run
        import importlib; importlib.reload(A)
        print('%-28s MTPA %.2f %%   min-loss %.2f %%' % (name, base['eff'], ml['eff']), flush=True)
