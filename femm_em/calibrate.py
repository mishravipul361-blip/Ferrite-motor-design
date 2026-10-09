"""Sensitivity of peak torque (360 A rms, best current angle) to the calibration knobs -> calib.json.
Used to tune motor_config.json so FEMM matches the Motor-CAD peak torque (45 N.m).
"""
import json, os
import numpy as np
import femm_motor as M

CASES = {'baseline': {}, 'stacking 0.95': {'stack': 0.95}, 'magnets 60 C': {'temp': 60}, 'magnet mu_r 1.10': {'mu': 1.10}}
GAMMAS, POS = [20, 30], [0, 4, 8]


def main():
    from multiprocessing import Pool
    r = json.load(open(os.path.join(M.HERE, 'results.json')))
    phi, sgn = r['noload']['phi'], r['gamma_sweep']['sign']
    jobs, keys = [], []
    for name, c in CASES.items():
        br = M.BR20 * (1 + M.BR_TC * (c.get('temp', 20) - 20))
        o = {'mu': c.get('mu', 1.05), 'stack': c.get('stack', 0.97)}
        for g in GAMMAS:
            for t in POS:
                jobs.append((t, M.phase_currents(M.P * t, phi, g, M.I_LINE_RMS), br, None, o)); keys.append((name, g))
    with Pool(M.NWORK, initializer=M.worker_init) as pool:
        out = pool.map(M.job, jobs, chunksize=1)
    res = {}
    for (name, g), (T, _, _) in zip(keys, out):
        res.setdefault(name, {}).setdefault(g, []).append(sgn * T)
    res = {n: max(float(np.mean(v)) for v in d.values()) for n, d in res.items()}
    json.dump(res, open(os.path.join(M.HERE, 'calib.json'), 'w'), indent=1)
    for n, T in res.items():
        print('%-18s %6.2f N.m  (%+.1f %%)' % (n, T, 100 * (T / res['baseline'] - 1)))


if __name__ == '__main__':
    main()
