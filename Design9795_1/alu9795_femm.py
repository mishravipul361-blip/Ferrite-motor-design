"""Aluminium winding for Design9795_1, FEMM part: stator-only variants against the Design9795_1 stator.

Rotor, magnets, air gap, bore (107 mm), stack (64 mm) and steel grade (M235-35A) are never changed. Variants change
the slot depth, tooth width and stator OD (160-167 mm, 167 hard limit) only. Half model (SYM=1), magnets 100 C, FEA fill 1.0 as the
refined Lab model. Per design: 4 x 4 (id, iq) grid over the 360 A circle at rotor 0 and 6 deg (cancels the 12 deg
torque ripple) -> torque, d-q flux, stator Steinmetz sums, AC field in the slot, tooth / yoke flux density.
Used in alu9795_study.py as differences / ratios against the copper stator (D0) on top of the Lab model.

python alu9795_femm.py        -> alu9795_femm.json   (cache: work/alu9795_<pid>.jsonl)
"""
import glob, json, math, os, time
os.environ['SYM'] = '1'
os.environ.setdefault('MAG_T', '100')
import numpy as np
import runguard as RG
import backemf as M
import lab_map as L
import femm

H = M.HERE
BASE_S = dict(M.STATOR)
D0, T0, YOKE0 = BASE_S['slot_depth'], BASE_S['tooth_w'], BASE_S['od'] / 2 - BASE_S['bore'] / 2 - BASE_S['slot_depth']   # yoke 8.85 mm


def at(od=160.0, yoke=YOKE0, dt=0.0):
    """Stator with outer diameter od, back-iron thickness yoke and tooth width change dt (mm)."""
    return dict(od=od, slot_depth=od / 2 - BASE_S['bore'] / 2 - yoke, tooth_w=T0 + dt)


DESIGNS = {'D0': at(),                                            # copper stator, Design9795_1
           # same OD 160 (same housing and lamination blank)
           'A1': at(yoke=7.85), 'A2': at(yoke=6.85), 'A3': at(dt=-1.0), 'A4': at(yoke=7.85, dt=-1.0), 'A5': at(yoke=6.85, dt=-1.0),
           'A6': at(yoke=7.85, dt=-2.0),
           # larger stator OD, hard limit 167 mm (housing), rotor unchanged
           'G1': at(od=163.0), 'G2': at(od=163.0, yoke=7.85), 'G3': at(od=163.0, yoke=7.85, dt=-1.0),
           'B1': at(od=165.0), 'B2': at(od=165.0, yoke=7.85), 'B3': at(od=165.0, yoke=7.85, dt=-1.0),
           'F1': at(od=167.0), 'F2': at(od=167.0, yoke=7.85), 'F3': at(od=167.0, yoke=7.85, dt=-1.0), 'F4': at(od=167.0, yoke=6.85, dt=-1.0)}
IDS = np.linspace(-L.I_PATH_PK, 0, 4); IQS = np.linspace(0, L.I_PATH_PK, 4)
POS = (0.0, 6.0)
W = {'design': None}


def slot_area(stator):
    """Slot area as Motor-CAD (validated on Design9795_1: 281.2 mm2)."""
    g = M.slot_geometry(stator)
    B, C, E, rb = g['B'], g['C'], g['E'], g['rb']
    a0, a1 = math.atan2(E[1], E[0]), math.atan2(E[1], -E[0])
    arc = [(rb * math.cos(t), rb * math.sin(t)) for t in np.linspace(a0, a1, 60)]
    x, y = np.array([B, C, E] + arc[1:-1] + [(-x, y) for x, y in reversed([B, C, E])]).T
    return float((0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) - 2 * (1 - math.pi / 4) * stator['corner_r'] ** 2) * 0.998)


def geometry(d):
    s = dict(BASE_S, **DESIGNS[d])
    g = M.slot_geometry(s)
    A_tooth = M.tooth_area(g, s)
    stack = M.STACK * 1e-3 * 7650 * M.STACKING
    teeth = A_tooth * 1e-6 * stack * M.NS
    yoke = math.pi * ((s['od'] / 2) ** 2 - g['rb'] ** 2) * 1e-6 * stack
    return dict(stator=s, slot_area=slot_area(s), yoke_mm=s['od'] / 2 - g['rb'], st_mass=teeth + yoke, teeth_mass=teeth, yoke_mass=yoke)


def ensure(d):
    if W['design'] == d:
        return
    if 'open' not in W:
        femm.openfemm(1); W['open'] = 1
    M.STATOR.clear(); M.STATOR.update(BASE_S, **DESIGNS[d])
    W['file'] = os.path.join(H, 'work', 'alu9795_%d.fem' % os.getpid()); M.W['file'] = W['file']
    M.build(W['file'], True)
    W['design'] = d


def fields(fn):
    s = open(fn).read().split('\n')
    l0 = next(i for i, l in enumerate(s) if l.startswith('[NumBlockLabels]'))
    labs = [l.split() for l in s[l0 + 1:l0 + 1 + int(s[l0].split('=')[1])]]
    k = s.index('[Solution]'); n = int(s[k + 1])
    nodes = np.array([[float(v) for v in l.split()[:3]] for l in s[k + 2:k + 2 + n]])
    ne = int(s[k + 2 + n])
    el = np.array([[int(v) for v in l.split()[:4]] for l in s[k + 3 + n:k + 3 + n + ne]])
    x, y, A = nodes[:, 0] / 1e3, nodes[:, 1] / 1e3, nodes[:, 2]
    i, j, kk = el[:, 0], el[:, 1], el[:, 2]
    b = np.stack([y[j] - y[kk], y[kk] - y[i], y[i] - y[j]], 1); c = np.stack([x[kk] - x[j], x[i] - x[kk], x[j] - x[i]], 1)
    a2 = x[i] * b[:, 0] + x[j] * b[:, 1] + x[kk] * b[:, 2]
    Ae = np.stack([A[i], A[j], A[kk]], 1)
    B = np.hypot((c * Ae).sum(1) / a2, (b * Ae).sum(1) / a2)
    area = np.abs(a2) / 2 * M.SYM_K
    mat = np.array([int(labs[l][2]) for l in el[:, 3]])        # 1 Air 2 Cu 3 M235 4 SCM420 5 Magnet
    r = np.hypot(nodes[el[:, :3], 0].mean(1), nodes[el[:, :3], 1].mean(1))
    mass = area * M.STACK * 1e-3 * M.STACKING * L.DENS
    rb = M.slot_geometry(M.STATOR)['rb']
    st = (mat == 3) & (r > M.R_BORE - 0.05); cu = mat == 2
    tooth, yoke = st & (r < rb - 0.3), st & (r > rb + 0.3)
    Bh = math.sqrt(2) * B
    return dict(st_Sa=float(np.sum(mass[st] * Bh[st] ** L.ALPHA)), st_S2=float(np.sum(mass[st] * Bh[st] ** 2)), st_mass_fea=float(mass[st].sum()),
                cu_B2=float(np.sum(area[cu] * B[cu] ** 2) / np.sum(area[cu])), cu_area=float(area[cu].sum() * 1e6 / 24),
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
    psi_c = M.coil_psi(); T = M.torque()
    femm.mo_close()
    psi = {ph: sum(L.SIGN[ph][tt] * psi_c[tt - 1] for tt in teeth) / M.PATHS for ph, teeth in M.PHASE_TEETH.items()}
    pd, pq = L.park(th, psi)
    out = dict(key='|'.join(map(str, t)), d=d, id=i_d, iq=i_q, th=th, T=T, psid=pd, psiq=pq, **fields(W['file'].replace('.fem', '.ans')))
    with open(os.path.join(H, 'work', 'alu9795_%d.jsonl' % os.getpid()), 'a') as fh:
        fh.write(json.dumps(out) + '\n')
    return out


def load_done():
    done = {}
    for fn in glob.glob(os.path.join(H, 'work', 'alu9795_*.jsonl')):
        for line in open(fn):
            try:
                r = json.loads(line); done[r['key']] = r
            except ValueError:
                pass
    return done


def reduce(done):
    out = dict(ids=IDS.tolist(), iqs=IQS.tolist(), positions=POS, designs={})
    for d in DESIGNS:
        g = {}
        for a, i_d in enumerate(IDS):
            for b, i_q in enumerate(IQS):
                rs = [done['|'.join(map(str, (d, float(i_d), float(i_q), th)))] for th in POS]
                for k in rs[0]:
                    if isinstance(rs[0][k], float) and k not in ('id', 'iq', 'th'):
                        g.setdefault(k, np.zeros((len(IDS), len(IQS))))[a, b] = np.mean([r[k] for r in rs])
        out['designs'][d] = dict(geo=geometry(d), **{k: v.tolist() for k, v in g.items()})
    RG.dump_atomic(out, os.path.join(H, 'alu9795_femm.json'), indent=1)


def main():
    from multiprocessing import Pool
    RG.keep_awake()
    t0 = time.time()
    for d in DESIGNS:
        g = geometry(d)
        print('%s od %.0f depth %.2f tooth %.2f yoke %.2f slot %.1f mm2 (x%.3f) stator %.3f kg' % (
            d, g['stator']['od'], g['stator']['slot_depth'], g['stator']['tooth_w'], g['yoke_mm'], g['slot_area'],
            g['slot_area'] / geometry('D0')['slot_area'], g['st_mass']), flush=True)
    done = load_done()
    todo = [(d, float(i_d), float(i_q), th) for d in DESIGNS for i_d in IDS for i_q in IQS for th in POS]
    todo = [t for t in todo if '|'.join(map(str, t)) not in done]
    print('jobs to run', len(todo), flush=True)
    with Pool(int(os.environ.get('NWORK', 6))) as pool:
        try:
            for k, r in enumerate(pool.imap(job, todo, chunksize=8), 1):
                done[r['key']] = r
                if k % 8 == 0:
                    print(time.strftime('%H:%M:%S'), '%d/%d' % (k, len(todo)), r['d'], 'T %.2f tooth %.2f yoke %.2f T' % (r['T'], r['tooth_B99'], r['yoke_B99']), flush=True)
        finally:
            RG.kill_femm_children()
    reduce(done)
    print('ALL DONE in %.0f min' % ((time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
