"""Motor-CAD-style iron loss map (100 C): full-cycle FEA per d-q point, Fourier per element, Steinmetz per harmonic.

For each d-q current point the rotor is turned through half an electrical cycle in 15 steps (30 points per cycle,
as Motor-CAD's model build); the second half follows from half-wave antisymmetry B(theta + 36 deg) = -B(theta).
Flux density is sampled at fixed stator points and at rotor-fixed points (rotor elements, in the rotor frame),
split into radial and tangential components, Fourier analysed, and

   P_hyst = sum_comp sum_n  Kh * (n f) * |B_n|^alpha                (W/kg)
   P_eddy = sum_comp sum_n  2 pi^2 * Keddy * (n f)^2 * |B_n|^2        (Motor-CAD Steinmetz eddy term)

summed with element mass and the 1.6 build factor. Stored per Hz (hysteresis) and per Hz^2 (eddy), so
P(f) = H1 * f + E1 * f^2, exactly as Motor-CAD scales its FEA loss map with speed.

Run: python iron_loss_map.py   (~1.7 h, resumable; writes iron_100C.json)
"""
import json, math, os, time
os.environ.setdefault('MAG_T', '100')
import numpy as np
import runguard as RG
import matplotlib.tri as mtri
import backemf as M
import lab_map as L
import femm

H = M.HERE
IDS = np.array([-L.I_PATH_PK, -2 * L.I_PATH_PK / 3, -L.I_PATH_PK / 3, 0.0])
IQS = np.array([0.0, L.I_PATH_PK / 2, L.I_PATH_PK])
NPC = 30                                  # points per electrical cycle
STEP = 72.0 / NPC                         # mech deg
POS = [k * STEP for k in range(NPC // 2)]
KH, KE, ALPHA = L.KH, L.KE, L.ALPHA
CELL_R, CELL_A = 0.5, 0.5                # mm, deg
NA = int(360 / CELL_A)
NH = NPC // 2 - 1                         # harmonics 1..14 (odd ones carry the loss)


def parse(fn):
    s = open(fn).read().split('\n')
    l0 = next(i for i, l in enumerate(s) if l.startswith('[NumBlockLabels]'))
    labs = [l.split() for l in s[l0 + 1:l0 + 1 + int(s[l0].split('=')[1])]]
    k = s.index('[Solution]'); n = int(s[k + 1])
    nodes = np.array([[float(v) for v in l.split()[:3]] for l in s[k + 2:k + 2 + n]])
    ne = int(s[k + 2 + n])
    el = np.array([[int(v) for v in l.split()[:4]] for l in s[k + 3 + n:k + 3 + n + ne]])
    x, y, A = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    i, j, kk = el[:, 0], el[:, 1], el[:, 2]
    xm, ym = x / 1e3, y / 1e3
    b = np.stack([ym[j] - ym[kk], ym[kk] - ym[i], ym[i] - ym[j]], 1); c = np.stack([xm[kk] - xm[j], xm[i] - xm[kk], xm[j] - xm[i]], 1)
    a2 = xm[i] * b[:, 0] + xm[j] * b[:, 1] + xm[kk] * b[:, 2]
    Ae = np.stack([A[i], A[j], A[kk]], 1)
    Bx = (c * Ae).sum(1) / a2; By = -(b * Ae).sum(1) / a2
    mat = np.array([int(labs[l][2]) for l in el[:, 3]])
    tri = mtri.Triangulation(x, y, el[:, :3])
    return dict(tri=tri, Bx=Bx, By=By, mat=mat, area=np.abs(a2) / 2, cx=x[el[:, :3]].mean(1), cy=y[el[:, :3]].mean(1))


def job(j):
    i_d, i_q = j
    t0 = time.time()
    Bst, Brt, T, psi = [], [], [], []
    pts = None
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
        # area-weighted averages of B (radial, tangential) over fixed polar cells (0.5 mm x 0.5 deg), steel only:
        # stator cells fixed in space, rotor cells fixed to the rotor -> no remeshing noise between rotor positions
        r = np.hypot(f['cx'], f['cy']); ang = np.arctan2(f['cy'], f['cx'])
        br = f['Bx'] * np.cos(ang) + f['By'] * np.sin(ang); bt = -f['Bx'] * np.sin(ang) + f['By'] * np.cos(ang)
        steel = f['mat'] == 3
        res = {}
        for part, sel, a0 in (('s', steel & (r > M.R_BORE - 0.05), 0.0), ('r', steel & (r < M.R_MID), th)):
            r0 = M.R_BORE if part == 's' else M.ROTOR['shaft_diameter'] / 2
            nr = int(((M.STATOR['od'] / 2 if part == 's' else M.R_MID) - r0) / CELL_R) + 1
            ir = np.clip(((r[sel] - r0) / CELL_R).astype(int), 0, nr - 1)
            ia = (((np.degrees(ang[sel]) - a0) % 360) / CELL_A).astype(int) % NA
            idx = ir * NA + ia; w = f['area'][sel]
            A_ = np.bincount(idx, w, nr * NA)
            res[part] = (A_, np.bincount(idx, w * br[sel], nr * NA), np.bincount(idx, w * bt[sel], nr * NA), nr)
        if pts is None:                     # cell masses and tooth mask from the first solve
            rb = M.slot_geometry(M.STATOR)['rb']
            ns = res['s'][3]
            rr = M.R_BORE + (np.arange(ns) + 0.5) * CELL_R
            pts = dict(sm=res['s'][0] * M.STACK * 1e-3 * M.STACKING * L.DENS, s_tooth=np.repeat(rr < rb, NA),
                       rm=res['r'][0] * M.STACK * 1e-3 * M.STACKING * L.DENS)
        for part, lst in (('s', Bst), ('r', Brt)):
            A_, sr, stt, _ = res[part]
            with np.errstate(invalid='ignore', divide='ignore'):
                lst.append(np.nan_to_num(np.stack([sr / A_, stt / A_])))
    out = dict(id=i_d, iq=i_q, T=float(np.mean(T)), T_wave=T, minutes=(time.time() - t0) / 60)
    d_, q_ = zip(*[L.park(th, p) for th, p in zip(POS, psi)])
    out['psid'], out['psiq'] = float(np.mean(d_)), float(np.mean(q_))
    Bst, Brt = np.stack(Bst).astype(np.float32), np.stack(Brt).astype(np.float32)
    np.savez_compressed(os.path.join(H, 'work', 'iron_raw_%d_%d.npz' % (round(i_d), round(i_q))), Bst=Bst, Brt=Brt,
                        sm=pts['sm'], rm=pts['rm'], s_tooth=pts['s_tooth'])
    out.update(harmonic_losses(Bst, Brt, pts['sm'], pts['rm'], pts['s_tooth']))
    return out


def full_cycle(Bst, Brt):
    """30 samples per electrical cycle from the 15 solved half-cycle positions.
    Stator frame: B(x, theta + 36) = -B(x, theta)  (rotor shifted one pole pitch = magnets reversed, currents reversed).
    Rotor frame:  B(p, theta + 36) = -B(p + 36 deg, theta)  (the rotor point then sits where the next pole's point was)."""
    s = np.concatenate([Bst, -Bst], 0)
    nr = Brt.shape[-1] // NA
    shifted = np.roll(Brt.reshape(Brt.shape[0], 2, nr, NA), -int(round(36 / CELL_A)), axis=-1).reshape(Brt.shape)
    r = np.concatenate([Brt, -shifted], 0)
    return s, r


def harmonic_losses(Bst, Brt, sm, rm, s_tooth):
    """Steinmetz per harmonic and component (radial, tangential), per Hz / per Hz^2, before build factor."""
    s, r = full_cycle(Bst, Brt)
    out = {}
    for name, B, m in (('stator', s, sm), ('tooth', s[:, :, s_tooth], sm[s_tooth]), ('yoke', s[:, :, ~s_tooth], sm[~s_tooth]), ('rotor', r, rm)):
        Bn = np.abs(np.fft.rfft(B, axis=0)) / NPC * 2                 # (16, 2, cells)
        n = np.arange(1, NH + 1)[:, None, None]
        hn = np.sum(m * KH * n * Bn[1:NH + 1] ** ALPHA, axis=(1, 2))   # per harmonic
        en = np.sum(m * 2 * math.pi ** 2 * KE * n ** 2 * Bn[1:NH + 1] ** 2, axis=(1, 2))
        out[name + '_Hn'], out[name + '_En'] = hn.tolist(), en.tolist()
        out[name + '_H1'], out[name + '_E1'] = float(hn.sum()), float(en.sum())
    return out

def main():
    from multiprocessing import Pool
    import psutil
    t0 = time.time()
    ck = os.path.join(H, 'work', 'iron100_partial.json')
    done = json.load(open(ck)) if os.path.exists(ck) else []
    todo = [(float(a), float(b)) for a in IDS for b in IQS if not any(abs(r['id'] - a) < 1e-6 and abs(r['iq'] - b) < 1e-6 for r in done)]
    todo.sort(key=lambda p: (abs(p[0]) + abs(p[1])))          # no-load first: early check against Motor-CAD
    with Pool(M.NWORK, initializer=M.worker_init, initargs=(True,)) as pool:
        for r in pool.imap_unordered(job, todo):
            done.append(r); json.dump(done, open(ck, 'w'))
            print(time.strftime('%H:%M:%S'), '%d/%d' % (len(done), len(IDS) * len(IQS)),
                  'id %.0f iq %.0f  T %.2f  stator H1 %.4f E1 %.3e  rotor H1 %.4f E1 %.3e  (%.1f min)' % (
                      r['id'], r['iq'], r['T'], r['stator_H1'], r['stator_E1'], r['rotor_H1'], r['rotor_E1'], r['minutes']), flush=True)
    RG.kill_registered()                                   # only this run's FEMM windows
    g = {}
    for k in ('stator_H1', 'stator_E1', 'tooth_H1', 'tooth_E1', 'rotor_H1', 'rotor_E1', 'T', 'psid', 'psiq'):
        a = np.zeros((len(IDS), len(IQS)))
        for r in done:
            a[int(np.argmin(abs(IDS - r['id']))), int(np.argmin(abs(IQS - r['iq'])))] = r[k]
        g[k] = a.tolist()
    json.dump(dict(id=IDS.tolist(), iq=IQS.tolist(), npc=NPC, build_factor=L.BF, **g), open(os.path.join(H, 'iron_100C.json'), 'w'), indent=1)
    print('done in %.0f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
