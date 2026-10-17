"""Motor-CAD-style iron loss map (100 C) on a FIXED mesh (FEMM air-gap element) - per-element Fourier, Steinmetz per harmonic.

Why: with mi_moverotate FEMM remeshes at every rotor position, so element-wise B jumps between positions (first-order
elements) and appears as broadband 'harmonics' that the eddy term (n^2) amplifies. With the air-gap element the stator
and rotor meshes never change; the rotor angle is the AGE boundary's inner angle. Every element then has a clean B(t).

For each d-q point: 15 rotor positions over half an electrical cycle (30 points per cycle, as Motor-CAD's model build).
  stator element:  B(x, th+36) = -B(x, th)
  rotor element:   B(p, th+36) = -B(p rotated +36 deg, th)   (nearest rotor-steel element)
  P_hyst = sum_{r,t} sum_n Kh (n f) |B_n|^alpha ;  P_eddy = sum_{r,t} sum_n 2 pi^2 Keddy (n f)^2 |B_n|^2 ; x mass ; x 1.6 later
Run: python iron_loss_age.py   -> iron_100C.json (+ work/age_raw_*.npz)
"""
import json, math, os, time
os.environ['AGE'] = '1'
os.environ.setdefault('MAG_T', '100')
import numpy as np
import runguard as RG
import backemf as M
import lab_map as L
import femm
from scipy.spatial import cKDTree

H = M.HERE
IDS = np.array([-L.I_PATH_PK, -2 * L.I_PATH_PK / 3, -L.I_PATH_PK / 3, 0.0])
IQS = np.array([0.0, L.I_PATH_PK / 2, L.I_PATH_PK])
NPC = 30
POS = [k * 72.0 / NPC for k in range(NPC // 2)]
NH = NPC // 2 - 1
KH, KE, ALPHA = L.KH, L.KE, L.ALPHA


def parse(fn):
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
    Bx = (c * Ae).sum(1) / a2; By = -(b * Ae).sum(1) / a2
    cx, cy = nodes[el[:, :3], 0].mean(1), nodes[el[:, :3], 1].mean(1)
    mat = np.array([int(labs[l][2]) for l in el[:, 3]])
    return dict(Bx=Bx, By=By, cx=cx, cy=cy, mat=mat, area=np.abs(a2) / 2)


def job(j):
    i_d, i_q = j
    t0 = time.time()
    Bs, T, psi, geo = [], [], [], None
    for th in POS:
        M.set_angle(th)
        ip = L.phase_currents(th, i_d, i_q)
        for ph, teeth in M.PHASE_TEETH.items():
            for t in teeth:
                femm.mi_modifycircprop('T%d' % t, 1, L.SIGN[ph][t] * ip[ph])
        femm.mi_analyze(1); femm.mi_loadsolution()
        psi_c = [femm.mo_getcircuitproperties('T%d' % k)[2] for k in range(1, M.NS + 1)]
        femm.mo_groupselectblock(1); T.append(femm.mo_blockintegral(22)); femm.mo_clearblock()
        femm.mo_close()
        psi.append({ph: sum(L.SIGN[ph][t] * psi_c[t - 1] for t in teeth) / M.PATHS for ph, teeth in M.PHASE_TEETH.items()})
        f = parse(M.W['file'].replace('.fem', '.ans'))
        if geo is None:
            geo = f
        assert len(f['Bx']) == len(geo['Bx']), 'mesh changed - air-gap element not active?'
        ang = np.arctan2(f['cy'], f['cx'])
        Bs.append(np.stack([f['Bx'] * np.cos(ang) + f['By'] * np.sin(ang), -f['Bx'] * np.sin(ang) + f['By'] * np.cos(ang)]).astype(np.float32))
    Bs = np.stack(Bs)                                         # (15, 2, n_elements)
    r = np.hypot(geo['cx'], geo['cy'])
    mass = geo['area'] * M.STACK * 1e-3 * M.STACKING * L.DENS
    st = (geo['mat'] == 3) & (r > M.R_BORE - 0.05)
    rt = (geo['mat'] == 3) & (r < M.R_MID)
    tooth = st & (r < M.slot_geometry(M.STATOR)['rb'])
    ridx = np.flatnonzero(rt)
    a = math.radians(36.0)
    px, py = geo['cx'][ridx] * math.cos(a) - geo['cy'][ridx] * math.sin(a), geo['cx'][ridx] * math.sin(a) + geo['cy'][ridx] * math.cos(a)
    _, nn = cKDTree(np.column_stack([geo['cx'][ridx], geo['cy'][ridx]])).query(np.column_stack([px, py]))
    np.savez_compressed(os.path.join(H, 'work', 'age_raw_%d_%d.npz' % (round(i_d), round(i_q))), Bs=Bs[:, :, st | rt], st=st[st | rt],
                        mass=mass[st | rt], tooth=tooth[st | rt], rmap=nn)
    out = dict(id=i_d, iq=i_q, T=float(np.mean(T)), T_wave=T, minutes=(time.time() - t0) / 60)
    d_, q_ = zip(*[L.park(th, p) for th, p in zip(POS, psi)])
    out['psid'], out['psiq'] = float(np.mean(d_)), float(np.mean(q_))
    out.update(losses(Bs[:, :, st], Bs[:, :, ridx], mass[st], mass[ridx], tooth[st], nn))
    return out


def losses(Bst, Brt, ms, mr, tooth, rmap):
    s = np.concatenate([Bst, -Bst], 0)
    r = np.concatenate([Brt, -Brt[:, :, rmap]], 0)
    out = {}
    for name, B, m in (('stator', s, ms), ('tooth', s[:, :, tooth], ms[tooth]), ('yoke', s[:, :, ~tooth], ms[~tooth]), ('rotor', r, mr)):
        Bn = np.abs(np.fft.rfft(B.astype(float), axis=0)) / NPC * 2
        n = np.arange(1, NH + 1)[:, None, None]
        hn = np.sum(m * KH * n * Bn[1:NH + 1] ** ALPHA, axis=(1, 2))
        en = np.sum(m * 2 * math.pi ** 2 * KE * n ** 2 * Bn[1:NH + 1] ** 2, axis=(1, 2))
        out[name + '_Hn'], out[name + '_En'] = hn.tolist(), en.tolist()
        out[name + '_H1'], out[name + '_E1'] = float(hn.sum()), float(en.sum())
    return out


def main():
    from multiprocessing import Pool
    import psutil
    t0 = time.time()
    ck = os.path.join(H, 'work', 'age100_partial.json')
    done = json.load(open(ck)) if os.path.exists(ck) else []
    todo = [(float(a), float(b)) for a in IDS for b in IQS if not any(abs(r['id'] - a) < 1e-6 and abs(r['iq'] - b) < 1e-6 for r in done)]
    todo.sort(key=lambda p: abs(p[0]) + abs(p[1]))
    with Pool(M.NWORK, initializer=M.worker_init, initargs=(True,)) as pool:
        for r in pool.imap_unordered(job, todo):
            done.append(r); json.dump(done, open(ck, 'w'))
            f = 1000 / 60 * 5
            print(time.strftime('%H:%M:%S'), '%d/%d' % (len(done), len(IDS) * len(IQS)),
                  'id %.0f iq %.0f T %.2f | @1000rpm x1.6: stator %.1f W (h %.1f e %.1f) rotor %.1f W (%.1f min)' % (
                      r['id'], r['iq'], r['T'], 1.6 * (r['stator_H1'] * f + r['stator_E1'] * f * f), 1.6 * r['stator_H1'] * f,
                      1.6 * r['stator_E1'] * f * f, 1.6 * (r['rotor_H1'] * f + r['rotor_E1'] * f * f), r['minutes']), flush=True)
    RG.kill_registered()                                   # only this run's FEMM windows
    g = {}
    for k in ('stator_H1', 'stator_E1', 'tooth_H1', 'tooth_E1', 'yoke_H1', 'yoke_E1', 'rotor_H1', 'rotor_E1', 'T', 'psid', 'psiq'):
        a = np.zeros((len(IDS), len(IQS)))
        for r in done:
            a[int(np.argmin(abs(IDS - r['id']))), int(np.argmin(abs(IQS - r['iq'])))] = r[k]
        g[k] = a.tolist()
    json.dump(dict(id=IDS.tolist(), iq=IQS.tolist(), npc=NPC, build_factor=L.BF, method='AGE fixed mesh, per-element Fourier', **g),
              open(os.path.join(H, 'iron_100C.json'), 'w'), indent=1)
    print('done in %.0f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
