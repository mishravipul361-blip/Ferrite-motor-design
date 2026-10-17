"""WMTC Class 1 (and 2-1) cycle efficiency of every design in alu_sweep.json -> wmtc_search.json.  python wmtc_search.py"""
import json, os, time
import numpy as np
from multiprocessing import Pool
import alu_study as A
import wmtc_cycle as W

SP = np.arange(0, 12001, 125.0); TQ = np.arange(0.5, 60.01, 0.5)
DEM = {c: W.demand(W.trace(W.CLASSES[c])) for c in ('Class 1', 'Class 2-1')}
lamof = lambda r: r.get('lam') or 'M235-35A'


def design(r):
    if r['label'].startswith('Al drop-in'):
        return A.Design('Al', 'S0', 63.0, 20, 1, 2.9464)
    if r['label'].startswith('Copper baseline'):
        return A.baseline()
    return A.Design(r['metal'], r['sv'], r['L'], r['Nph'], r['ns'], r['d'], lam=lamof(r))


def job(r):
    me = W.MapEval(design(r), SP, TQ, npts=121)
    return r['label'], {c: me.cycle(*DEM[c]) for c in DEM}


if __name__ == '__main__':
    t0 = time.time()
    R = json.load(open(os.path.join(A.H, 'alu_sweep.json')))
    fn = os.path.join(A.H, 'wmtc_search.json')
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    todo = [r for r in R if r['label'] not in out]
    print('designs', len(todo), flush=True)
    with Pool(7) as p:
        for k, (lab, res) in enumerate(p.imap_unordered(job, todo, chunksize=4), 1):
            out[lab] = res
            if k % 50 == 0:
                json.dump(out, open(fn, 'w')); print(k, '%.0f s' % (time.time() - t0), flush=True)
    json.dump(out, open(fn, 'w'))
    print('done %.0f s' % (time.time() - t0))
