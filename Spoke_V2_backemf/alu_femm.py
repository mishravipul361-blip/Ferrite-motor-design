"""Aluminium-winding study, FEMM part: enlarged-slot stator variants (deeper slot / narrower tooth) against the baseline.

Same rotor as the Lab model (Motor-CAD original), magnets 100 C, 10 turns/coil, 2 paths (the winding is rescaled later in
alu_study.py through ampere-turns, so the FEMM coil turns do not matter). Per design: 4 x 4 (id, iq) grid over the 360 A
circle, rotor at 0 and 6 deg (cancels the 12 deg ripple) -> torque, d-q flux, stator iron Steinmetz sums, tooth/yoke B.
Results are used as ratios variant/baseline on top of the tuned Lab model (lab_model.py).

Run: python alu_femm.py       -> alu_femm.json   (cache: work/alu_solves_<pid>.jsonl)
"""
import glob, json, math, os, time
os.environ['AGE'] = '0'
os.environ.setdefault('MAG_T', '100')
import numpy as np
import runguard as RG
import backemf as M
import lab_map as L
import femm

H = M.HERE
BASE_S, BASE_R = dict(M.STATOR), dict(M.ROTOR)
DESIGNS = {'S0': dict(slot_depth=17.5, tooth_w=15.5), 'S1': dict(slot_depth=18.5, tooth_w=15.5),
           'S2': dict(slot_depth=19.5, tooth_w=15.5), 'S3': dict(slot_depth=19.5, tooth_w=14.5),
           'S4': dict(slot_depth=20.5, tooth_w=14.5),
           # larger stator OD within the 167 mm housing limit, rotor unchanged (structural design untouched)
           'P1': dict(od=165.0, slot_depth=20.0, tooth_w=15.5), 'P2': dict(od=167.0, slot_depth=21.0, tooth_w=15.5),
           'P3': dict(od=167.0, slot_depth=22.0, tooth_w=15.5), 'P4': dict(od=167.0, slot_depth=22.0, tooth_w=14.5),
           # combined: near-air-gap rotor change (inset 1.5, opening 8.0; shaft-side webs untouched) + slot tip changes
           'Q1': dict(od=167.0, slot_depth=21.0, tooth_w=15.5, magnet_inset=1.5, magnet_opening=8.0, slot_open=4.0),
           'Q2': dict(od=167.0, slot_depth=21.0, tooth_w=15.5, magnet_inset=1.5, magnet_opening=8.0, slot_open=4.0, tip_depth=1.2),
           'Q3': dict(od=167.0, slot_depth=21.0, tooth_w=15.5, magnet_inset=1.5, magnet_opening=8.0),
           'Q4': dict(od=165.0, slot_depth=20.0, tooth_w=15.5, magnet_inset=1.5, magnet_opening=8.0, slot_open=4.0),
           # same OD 160 as copper: deeper/narrower slots + the same near-air-gap rotor change and 4 mm slot opening
           'R1': dict(slot_depth=18.5, tooth_w=15.5, magnet_inset=1.5, magnet_opening=8.0, slot_open=4.0),
           'R2': dict(slot_depth=19.5, tooth_w=15.5, magnet_inset=1.5, magnet_opening=8.0, slot_open=4.0),
           'R3': dict(slot_depth=19.5, tooth_w=14.5, magnet_inset=1.5, magnet_opening=8.0, slot_open=4.0),
           'R0': dict(slot_depth=17.5, tooth_w=15.5, magnet_inset=1.5, magnet_opening=8.0, slot_open=4.0)}
IDS = np.linspace(-L.I_PATH_PK, 0, 4); IQS = np.linspace(0, L.I_PATH_PK, 4)
POS = (0.0, 6.0)
W = {'design': None}


def ensure(d):
    if W['design'] == d:
        return
    if 'open' not in W:
        femm.openfemm(1); W['open'] = 1
    g = DESIGNS[d]
    M.ROTOR.clear(); M.ROTOR.update(BASE_R, **{k: v for k, v in g.items() if k in BASE_R})        # near-air-gap rotor changes
    M.STATOR.clear(); M.STATOR.update(BASE_S, **{k: v for k, v in g.items() if k not in BASE_R})
    W['file'] = os.path.join(H, 'work', 'alu_%d.fem' % os.getpid()); M.W['file'] = W['file']
    M.build(W['file'], True)
    W['design'] = d


def fields(fn):
    s = open(fn).read().split('\n')
    l0 = next(i for i, l in enumerate(s) if l.startswith('[NumBlockLabels]'))
    labs = [l.split() for l in s[l0 + 1:l0 + 1 + int(s[l0].split('=')[1])]]
    k = s.index('[Solution]'); n = int(s[k + 1])
    nodes = np.array([[float(x) for x in l.split()[:3]] for l in s[k + 2:k + 2 + n]])
    ne = int(s[k + 2 + n])
    el = np.array([[int(x) for x in l.split()[:4]] for l in s[k + 3 + n:k + 3 + n + ne]])
    x, y, A = nodes[:, 0] / 1e3, nodes[:, 1] / 1e3, nodes[:, 2]
    i, j, kk = el[:, 0], el[:, 1], el[:, 2]
    b = np.stack([y[j] - y[kk], y[kk] - y[i], y[i] - y[j]], 1); c = np.stack([x[kk] - x[j], x[i] - x[kk], x[j] - x[i]], 1)
    a2 = x[i] * b[:, 0] + x[j] * b[:, 1] + x[kk] * b[:, 2]
    Ae = np.stack([A[i], A[j], A[kk]], 1)
    B = np.hypot((c * Ae).sum(1) / a2, (b * Ae).sum(1) / a2); area = np.abs(a2) / 2
    mat = np.array([int(labs[l][2]) for l in el[:, 3]])
    r = np.hypot(nodes[el[:, :3], 0].mean(1), nodes[el[:, :3], 1].mean(1)) * 1e3 / 1e3
    mass = area * M.STACK * 1e-3 * M.STACKING * L.DENS
    rb = M.slot_geometry(M.STATOR)['rb']
    st = (mat == 3) & (r > M.R_BORE - 0.05); cu = mat == 2
    tooth, yoke = st & (r < rb - 0.3), st & (r > rb + 0.3)
    Bh = math.sqrt(2) * B
    return dict(st_Sa=float(np.sum(mass[st] * Bh[st] ** L.ALPHA)), st_S2=float(np.sum(mass[st] * Bh[st] ** 2)),
                st_mass=float(mass[st].sum()), cu_B2=float(np.sum(area[cu] * B[cu] ** 2) / np.sum(area[cu])),
                tooth_B99=float(np.percentile(B[tooth], 99)), tooth_Bmean=float(np.average(B[tooth], weights=area[tooth])),
                yoke_B99=float(np.percentile(B[yoke], 99)), yoke_Bmean=float(np.average(B[yoke], weights=area[yoke])))


def _reset():
    try:
        femm.closefemm()
    except Exception:
        pass
    W.pop('open', None); W['design'] = None


@RG.femm_retry(_reset)
def job(t):
    d, i_d, i_q, th = t
    ensure(d)
    M.set_angle(th)
    ip = L.phase_currents(th, i_d, i_q)
    for ph, teeth in M.PHASE_TEETH.items():
        for tt in teeth:
            femm.mi_modifycircprop('T%d' % tt, 1, L.SIGN[ph][tt] * ip[ph])
    femm.mi_analyze(1); femm.mi_loadsolution()
    psi_c = [femm.mo_getcircuitproperties('T%d' % k)[2] for k in range(1, M.NS + 1)]
    femm.mo_groupselectblock(1); T = femm.mo_blockintegral(22); femm.mo_clearblock()
    femm.mo_close()
    psi = {ph: sum(L.SIGN[ph][tt] * psi_c[tt - 1] for tt in teeth) / M.PATHS for ph, teeth in M.PHASE_TEETH.items()}
    pd, pq = L.park(th, psi)
    out = dict(key='|'.join(map(str, t)), d=d, id=i_d, iq=i_q, th=th, T=T, psid=pd, psiq=pq, **fields(W['file'].replace('.fem', '.ans')))
    with open(os.path.join(H, 'work', 'alu_solves_%d.jsonl' % os.getpid()), 'a') as fh:
        fh.write(json.dumps(out) + '\n')
    return out


def load_done():
    done = {}
    for fn in glob.glob(os.path.join(H, 'work', 'alu_solves_*.jsonl')):
        for line in open(fn):
            try:
                r = json.loads(line); done[r['key']] = r
            except ValueError:
                pass
    return done


def reduce(done):
    out = dict(ids=IDS.tolist(), iqs=IQS.tolist(), positions=POS, designs={})
    for d, geo in DESIGNS.items():
        g = {}
        for a, i_d in enumerate(IDS):
            for b, i_q in enumerate(IQS):
                rs = [done['|'.join(map(str, (d, float(i_d), float(i_q), th)))] for th in POS]
                for k in rs[0]:
                    if isinstance(rs[0][k], float) and k not in ('id', 'iq', 'th'):
                        g.setdefault(k, np.zeros((len(IDS), len(IQS))))[a, b] = np.mean([r[k] for r in rs])
        out['designs'][d] = dict(geo=geo, **{k: v.tolist() for k, v in g.items()})
    sgn = 1 if out['designs']['S0']['T'][-1][-1] > 0 else -1
    for d in out['designs'].values():
        d['T'] = (sgn * np.array(d['T'])).tolist()
    RG.dump_atomic(out, os.path.join(H, 'alu_femm.json'), indent=1)


def main():
    from multiprocessing import Pool
    import psutil
    RG.keep_awake()
    os.makedirs(os.path.join(H, 'work'), exist_ok=True)
    t0 = time.time()
    done = load_done()
    todo = [(d, float(i_d), float(i_q), th) for d in DESIGNS for i_d in IDS for i_q in IQS for th in POS]
    todo = [t for t in todo if '|'.join(map(str, t)) not in done]
    print('jobs to run', len(todo), flush=True)
    with Pool(4) as pool:
        try:
            for k, r in enumerate(pool.imap(job, todo, chunksize=8), 1):
                done[r['key']] = r
                if k % 8 == 0:
                    print(time.strftime('%H:%M:%S'), '%d/%d' % (k, len(todo)), r['d'], 'T %.2f' % r['T'], flush=True)
        finally:
            RG.kill_femm_children()
    reduce(done)
    print('ALL DONE in %.0f min' % ((time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
