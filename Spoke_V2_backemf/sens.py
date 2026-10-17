"""One-at-a-time geometric sensitivity study (FEMM 2D, moving rotor).

Per design:
  'tq'   360 A rms line, gamma 17/23/29 deg, rotor 0 and 6 deg (cancels the 12-deg ripple) -> MTPA torque at 2000 rpm  [100 C]
  'rip'  OP1 currents (139.7 A rms ph, 66.38 deg), rotor 0..11 deg step 1 -> ripple at the worst full-load point      [100 C]
  'nl'   no load, rotor 0..33.6 deg step 2.4 (half cycle, 30 pts/cycle) -> back-EMF THD (Motor-CAD definition)      [20 C]
  'op2'  2 currents near the baseline OP2 point, rotor 0 and 6 deg -> local d-q model -> loss at 9339 rpm, 3.24 N.m [100 C]
Results per job are cached in work/sens_cache.json (resumable). Run: python sens.py
"""
import json, math, os, sys, time
os.environ['AGE'] = '0'
os.environ.setdefault('MAG_T', '100')
import numpy as np
import runguard as RG
import backemf as M
import lab_map as L
import femm

H = M.HERE
# hard constraints (11 Oct 2026): shaft OD 30 mm, Bridge_thickness >= 1 mm, bridge_offset > 3.5 mm
BASE_R = dict(M.ROTOR, shaft_diameter=30.0, Bridge_thickness=1.0, bridge_offset=4.0)
BASE_S = dict(M.STATOR)
VARIANTS = [('baseline', {}, {})]
for k, lo, hi in (('slot_open', 4.0, 6.0), ('tooth_w', 14.5, 16.5), ('tip_depth', 0.6, 1.2)):
    VARIANTS += [('%s=%g' % (k, lo), {}, {k: lo}), ('%s=%g' % (k, hi), {}, {k: hi})]
for k, lo, hi in (('magnet_opening', 7.5, 9.5), ('magnet_Y', 9.5, 11.5), ('magnet_Z', 5.5, 6.5), ('magnet_inset', 1.5, 2.5),
                  ('magnet_depth', 26.0, 30.0), ('Bridge_thickness', 1.25, 1.5), ('bridge_offset', 3.6, 5.0)):
    VARIANTS += [('%s=%g' % (k, lo), {k: lo}, {}), ('%s=%g' % (k, hi), {k: hi}, {})]
VARIANTS += [('MotorCAD_original', {'shaft_diameter': 25.0, 'Bridge_thickness': 1.0, 'bridge_offset': 4.0}, {})]   # reference only (violates constraints)
# combined candidates from the one-at-a-time sensitivities (run with CAND=1: python sens.py)
CANDIDATES = [('C1', {'magnet_inset': 1.5}, {'slot_open': 4.0, 'tip_depth': 1.2, 'tooth_w': 16.5}),
              ('C2', {'magnet_inset': 1.5}, {'slot_open': 4.0, 'tooth_w': 16.5}),
              ('C3', {'magnet_inset': 1.5}, {'slot_open': 4.0, 'tip_depth': 1.2})]          # C1 without the wider tooth: SWG 11 must still fit
if os.environ.get('CAND') == '1':
    VARIANTS = CANDIDATES
# aluminium study: Motor-CAD original rotor (shaft 25, structurally unchanged) with stator variants from alu_femm.py, e.g. ALU=S0,O2
ALU_ROT = {'shaft_diameter': 25.0, 'Bridge_thickness': 1.0, 'bridge_offset': 4.0}
if os.environ.get('ALU'):
    import alu_femm as _AF
    VARIANTS = [('ALU_' + sv, dict(ALU_ROT, **{k: x for k, x in _AF.DESIGNS[sv].items() if k in _AF.BASE_R}),
                 {k: x for k, x in _AF.DESIGNS[sv].items() if k not in _AF.BASE_R}) for sv in os.environ['ALU'].split(',')]
    CANDIDATES = CANDIDATES + VARIANTS
VD = {v[0]: v for v in VARIANTS + CANDIDATES}
BR20 = M.BR20; BR100 = M.BR20 * (1 + M.BR_TC * 80)
OPC = json.load(open(os.path.join(H, 'op_compare.json')))
OP2_ID, OP2_IQ = [OPC[k]['same_point'] for k in OPC if k.startswith('OP2')][0]['id_path'], [OPC[k]['same_point'] for k in OPC if k.startswith('OP2')][0]['iq_path']
I360 = L.I_PATH_PK
OP1 = (139.7 * math.sqrt(2) / 2, 66.38)


def tasks(v):
    t = [(v, 'tq', g, th) for g in (17.0, 23.0, 29.0) for th in (0.0, 6.0)]
    t += [(v, 'rip', 0.0, float(th)) for th in range(12)]
    t += [(v, 'nl', 0.0, round(k * 2.4, 4)) for k in range(15)]
    t += [(v, 'op2', k, th) for k in (0, 1) for th in (0.0, 6.0)]
    if v in [c[0] for c in CANDIDATES]:                       # candidates: ripple waveforms at OP2 and OP3 too
        t += [(v, 'rip2', 0.0, float(th)) for th in range(12)] + [(v, 'rip3', 0.0, float(th)) for th in range(12)]
    return t


W = {'variant': None}


def set_br(br):
    femm.mi_modifymaterial('Magnet', 3, br / (M.MU0 * M.MU_MAG))


def ensure(v):
    if W['variant'] == v:
        return
    _, rot, sta = VD[v]
    M.ROTOR.clear(); M.ROTOR.update(BASE_R, **rot)
    M.STATOR.clear(); M.STATOR.update(BASE_S, **sta)
    if 'pid' not in W:
        femm.openfemm(1); W['pid'] = os.getpid()
    os.makedirs(os.path.join(H, 'work'), exist_ok=True)
    W['file'] = os.path.join(H, 'work', 'sens_%d.fem' % os.getpid())
    M.W['file'] = W['file']
    M.build(W['file'], True)
    W['variant'] = v


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
    r = np.hypot(nodes[el[:, :3], 0].mean(1), nodes[el[:, :3], 1].mean(1))
    mass = area * M.STACK * 1e-3 * M.STACKING * L.DENS
    st = (mat == 3) & (r > M.R_BORE - 0.05); cu = mat == 2
    Bh = math.sqrt(2) * B
    return dict(st_Sa=float(np.sum(mass[st] * Bh[st] ** L.ALPHA)), st_S2=float(np.sum(mass[st] * Bh[st] ** 2)),
                cu_B2=float(np.sum(area[cu] * B[cu] ** 2) / np.sum(area[cu])))


def _reset():
    try:
        femm.closefemm()
    except Exception:
        pass
    W.pop('pid', None); W['variant'] = None


@RG.femm_retry(_reset)
def job(j):
    v, kind, a, th = j
    ensure(v)
    if kind == 'nl':
        set_br(BR20); i_d = i_q = 0.0
    else:
        set_br(BR100)
        if kind == 'tq':
            i_d, i_q = -I360 * math.sin(math.radians(a)), I360 * math.cos(math.radians(a))
        elif kind in ('rip', 'rip2', 'rip3'):
            irms, adv = {'rip': OP1, 'rip2': (73.7 * math.sqrt(2) / 2, 78.5), 'rip3': (208.8 * math.sqrt(2) / 2, 20.78)}[kind]
            i_d, i_q = -irms * math.sin(math.radians(adv)), irms * math.cos(math.radians(adv))
        else:
            i_d, i_q = OP2_ID - 10.0 * a, OP2_IQ
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
    out = dict(key='|'.join(map(str, j)), v=v, kind=kind, a=a, th=th, T=T, psid=d, psiq=q, psi1=psi['Ph1'], id=i_d, iq=i_q)
    if kind == 'op2':
        out.update(fields(W['file'].replace('.fem', '.ans')))
    with open(os.path.join(H, 'work', 'sens_solves_%d.jsonl' % os.getpid()), 'a') as fh:   # saved per solve: nothing lost if stopped
        fh.write(json.dumps(out) + '\n')
    return out


def main():
    from multiprocessing import Pool
    import psutil
    RG.keep_awake()
    t0 = time.time()
    ck = os.path.join(H, 'work', 'sens_cache%s.json' % ('_alu' if os.environ.get('ALU') else '_cand' if os.environ.get('CAND') == '1' else ''))
    done = json.load(open(ck)) if os.path.exists(ck) else []
    import glob
    keys = {r['key'] for r in done}
    for fn in glob.glob(os.path.join(H, 'work', 'sens_solves_*.jsonl')):      # per-solve saves from earlier (stopped) runs
        for line in open(fn):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r['key'] not in keys:
                done.append(r); keys.add(r['key'])
    RG.dump_atomic(done, ck)
    have = {r['key'] for r in done}
    todo = [t for v in VARIANTS for t in tasks(v[0]) if '|'.join(map(str, t)) not in have]
    print('jobs to run', len(todo), flush=True)
    with Pool(int(os.environ.get('NWORK', M.NWORK))) as pool:
        try:
            for r in pool.imap(job, todo, chunksize=1 if (os.environ.get('CAND') == '1' or os.environ.get('ALU')) else 37):   # one worker = one whole design (one geometry build)
                done.append(r); RG.dump_atomic(done, ck)
                nv = sum(1 for v in VARIANTS if all(('|'.join(map(str, t)) in {x['key'] for x in done}) for t in tasks(v[0])))
                if (r['kind'] == 'op2' and r['a'] == 1 and r['th'] == 6.0) or (r['kind'] == 'rip3' and r['th'] == 11.0):
                    print(time.strftime('%H:%M:%S'), 'design complete: %s  (%d/%d designs)' % (r['v'], nv, len(VARIANTS)), flush=True)
        finally:
            RG.kill_femm_children()                               # close only the FEMM windows opened by this run's workers
    print('ALL DONE in %.0f min' % ((time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()

