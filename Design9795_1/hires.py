"""Fine no-load run for overlay with Motor-CAD: EMF at 1 deg mech over half an electrical period (half-wave
antisymmetry gives the other half) + cogging over one 6 deg period at 0.2 deg.  MAG_T env sets magnet temperature.
Writes hires_<T>C.json.   Run:  set MAG_T=30 & python hires.py
"""
import json, os, time
import numpy as np
import runguard as RG
import psutil
import backemf as M


def main():
    from multiprocessing import Pool
    t0 = time.time()
    th_e = np.arange(0, 36, float(os.environ.get('STEP_E', 1.0)))
    th_c = np.arange(0, 6, float(os.environ.get('STEP_C', 0.2)))
    with Pool(M.NWORK, initializer=M.worker_init, initargs=(True,)) as pool:
        out = pool.map(M.job, list(th_e) + list(th_c), chunksize=1)
    emf_out, cog_out = out[:len(th_e)], out[len(th_e):]
    psi = [o[1] for o in emf_out]
    psi = psi + [[-v for v in p] for p in psi]                    # psi(theta + 36 deg) = -psi(theta)
    th = np.concatenate([th_e, th_e + 36])
    path, emf, _ = M.analyse(th, psi)
    res = dict(magnet_T=M.MAG_T, Br=M.BR, theta_m=th.tolist(), coil_psi=psi,
               path_psi={k: v.tolist() for k, v in path.items()}, emf_phase_1000rpm={k: v.tolist() for k, v in emf.items()},
               emf_torque=[o[2] for o in emf_out], cog_theta_m=th_c.tolist(), cogging=[o[2] for o in cog_out],
               minutes=(time.time() - t0) / 60)
    fn = os.path.join(M.HERE, 'hires_%dC%s.json' % (round(M.MAG_T), os.environ.get('TAG', '')))
    json.dump(res, open(fn, 'w'), indent=1)
    e = emf['Ph1']
    print('%s  Erms %.3f  Epk %.3f  cog pk-pk %.3f  (%.1f min)' % (fn, np.sqrt(np.mean(e ** 2)), np.max(np.abs(e)),
                                                              np.ptp(res['cogging']), res['minutes']), flush=True)
    RG.kill_registered()                                   # only this run's FEMM windows


if __name__ == '__main__':
    main()
