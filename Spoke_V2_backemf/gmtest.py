import os, sys, json
from multiprocessing import Pool
os.environ['MAG_T'] = '20'
import backemf as M

def run(gm):
    os.environ['GAPMESH'] = str(gm)
    import importlib; importlib.reload(M)
    import femm
    femm.openfemm(1)
    M.W['file'] = os.path.join(M.HERE, 'work', 'gm_%s.fem' % gm)
    M.build(M.W['file'], True)
    out = [M.job(t)[2] for t in (1.6, 4.4)]
    femm.closefemm()
    return gm, out

if __name__ == '__main__':
    with Pool(3) as p:
        for gm, out in p.imap(run, [1.0, 0.5, 0.25]):
            print('gap mesh x%.2f  T(1.6)=%.4f  T(4.4)=%.4f  pk-pk~%.3f' % (gm, out[0], out[1], out[0] - out[1]), flush=True)
