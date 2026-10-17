"""Direct FEMM torque waveform at the three Motor-CAD operating-point currents (100 C): rotor 0-11.5 deg mech in 0.5 deg
(one 12 deg on-load ripple period), moving rotor, weighted stress tensor.  Writes op_ripple.json."""
import json, math, os, time
os.environ.setdefault('MAG_T', '100')
os.environ['AGE'] = '0'
import numpy as np
import runguard as RG
import backemf as M
import lab_map as L
import femm

H = M.HERE
OPS = {'OP1': (139.7, 66.38), 'OP2': (73.7, 78.5), 'OP3': (208.8, 20.78)}      # phase rms A, phase advance el deg
POS = list(np.arange(0, 12, 0.5))


def job(j):
    name, th = j
    irms, adv = OPS[name]
    ipk = irms * math.sqrt(2) / 2                                      # path peak
    i_d, i_q = -ipk * math.sin(math.radians(adv)), ipk * math.cos(math.radians(adv))
    M.set_angle(th)
    ip = L.phase_currents(th, i_d, i_q)
    for ph, teeth in M.PHASE_TEETH.items():
        for t in teeth:
            femm.mi_modifycircprop('T%d' % t, 1, L.SIGN[ph][t] * ip[ph])
    femm.mi_analyze(1); femm.mi_loadsolution()
    psi_c = [femm.mo_getcircuitproperties('T%d' % k)[2] for k in range(1, M.NS + 1)]
    femm.mo_groupselectblock(1); T = femm.mo_blockintegral(22); femm.mo_clearblock()
    femm.mo_close()
    psi = {ph: sum(L.SIGN[ph][t] * psi_c[t - 1] for t in teeth) / M.PATHS for ph, teeth in M.PHASE_TEETH.items()}
    d, q = L.park(th, psi)
    return dict(op=name, th=float(th), T=T, psid=d, psiq=q)


def main():
    from multiprocessing import Pool
    import psutil
    t0 = time.time()
    jobs = [(n, th) for n in OPS for th in POS]
    with Pool(M.NWORK, initializer=M.worker_init, initargs=(True,)) as pool:
        out = []
        for r in pool.imap_unordered(job, jobs):
            out.append(r)
            print(time.strftime('%H:%M:%S'), '%d/%d' % (len(out), len(jobs)), r['op'], 'th %.1f T %.3f' % (r['th'], r['T']), flush=True)
    RG.kill_registered()                                   # only this run's FEMM windows
    res = {}
    for n in OPS:
        rs = sorted([r for r in out if r['op'] == n], key=lambda r: r['th'])
        T = np.array([r['T'] for r in rs]); sgn = 1 if T.mean() > 0 else -1; T = sgn * T
        res[n] = dict(theta=[r['th'] for r in rs], T=T.tolist(), T_mean=float(T.mean()), ripple_pp=float(np.ptp(T)),
                      ripple_pct=float(100 * np.ptp(T) / T.mean()), psid=float(np.mean([r['psid'] for r in rs])), psiq=float(np.mean([r['psiq'] for r in rs])))
        print(n, {k: round(v, 3) for k, v in res[n].items() if isinstance(v, float)})
    json.dump(res, open(os.path.join(H, 'op_ripple.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
