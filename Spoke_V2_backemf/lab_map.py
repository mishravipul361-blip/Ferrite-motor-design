"""FEMM d-q flux map at 100 C + Motor-CAD-Lab-style torque-speed and efficiency maps for Spoke V2.

Lab settings (Motor-CAD Calculation tab, 10 Oct 2026): DC 52 V, modulation index 0.95, motor mode, MTPA,
DC current limit 280 A, max stator current 360 A rms (509.1 A peak), max speed 12 000 rpm,
winding 100 C, magnets 100 C (flux coefficient -0.1797 %/K), delta, 10 turns/coil, 63 mm.
Losses (Motor-CAD Settings > Losses): Steinmetz Kh 0.02533, Keddy 2.199e-6, alpha 2.07, build factor 1.6 (stator,
rotor); bearings 11.85 + 8.89 W and windage 8.89 W at 2600 rpm, exponent 1.3; AC winding hybrid, adjustment 2;
magnet loss (ferrite, 100 ohm.m) ~0; converter loss 0.

Step 1 (FEMM, ~1.5 h, cached in dq_100C.json):  python lab_map.py solve
Step 2 (seconds):                                python lab_map.py     -> lab_100C.json, lab_100C.png
"""
import json, math, os, sys, time
os.environ.setdefault('MAG_T', '100')
import numpy as np
import runguard as RG
import backemf as M
import femm

H = M.HERE
P = 5
VDC, MI = 52.0, 0.95
V_MAX = MI * VDC                                   # peak phase (= line, delta) voltage
I_LINE_RMS = 360.0
I_PATH_PK = I_LINE_RMS * math.sqrt(2) / math.sqrt(3) / 2   # 146.97 A
IDC_MAX = 280.0
P_DC_MAX = VDC * IDC_MAX                           # converter loss 0 -> 14.56 kW
T_WDG = 100.0
RHO20, ALPHA_CU = 1.724e-8, 0.003862
D_CU = 2.9464e-3; A_CU = math.pi / 4 * D_CU ** 2
MLT = 2 * 0.063 + 0.075615
R_COIL = RHO20 * (1 + ALPHA_CU * (T_WDG - 20)) * M.TURNS * MLT / A_CU
R_PATH = 2 * R_COIL
L_EW_PATH = 2 * 1.954e-6                           # Motor-CAD end-winding inductance 1.954 uH per phase, phase = 2 paths
LOSS = json.load(open(os.path.join(H, 'motorcad_losses.json')))
KH, KE, ALPHA = LOSS['iron_loss']['Kh'], LOSS['iron_loss']['Keddy'], LOSS['iron_loss']['alpha']
BF = LOSS['iron_loss']['build_factor_stator']
AC_ADJ = LOSS['ac_winding']['adjustment_factor']
DENS = 7650.0
N_COND = 12 * 20
IDS = np.linspace(-I_PATH_PK, 0, 6); IQS = np.linspace(0, I_PATH_PK, 6)
POS = [0.0, 7.5, 15.0, 22.5]                      # mech deg: half a slot pitch (rotor ripple) and the 12 deg torque period


def mech_loss(n):
    b, w = LOSS['bearing'], LOSS['windage']
    return (b['front_W'] + b['rear_W']) * (n / b['ref_rpm']) ** b['exponent'] + w['W'] * (n / w['ref_rpm']) ** w['exponent']


def phase_setup():
    """Coil signs and phase angles from the no-load coil flux linkages (results.json, same geometry)."""
    r = json.load(open(os.path.join(H, 'results.json')))['shaft_magnetic']
    th = 5 * np.radians(np.array(r['theta_m'])); psi = np.array(r['coil_psi'])
    X = 2 * np.mean(psi * np.exp(-1j * th)[:, None], axis=0)
    sign, phi = {}, {}
    Y = {}
    for ph, teeth in M.PHASE_TEETH.items():
        ref = X[teeth[0] - 1]
        s = {t: (1 if np.real(X[t - 1] * np.conj(ref)) > 0 else -1) for t in teeth}
        Y[ph] = sum(s[t] * X[t - 1] for t in teeth) / M.PATHS
        sign[ph] = s
    for ph in ('Ph2', 'Ph3'):
        if np.real(Y[ph] * np.conj(Y['Ph1'])) > 0:
            Y[ph] = -Y[ph]; sign[ph] = {t: -v for t, v in sign[ph].items()}
    for ph in Y:
        phi[ph] = float(np.angle(Y[ph]))
    return sign, phi


SIGN, PHI = phase_setup()


def phase_currents(th_m, i_d, i_q):
    a = {ph: math.radians(P * th_m) + PHI[ph] for ph in PHI}
    return {ph: i_d * math.cos(a[ph]) - i_q * math.sin(a[ph]) for ph in PHI}


def park(th_m, psi_ph):
    a = {ph: math.radians(P * th_m) + PHI[ph] for ph in PHI}
    return (2 / 3 * sum(psi_ph[ph] * math.cos(a[ph]) for ph in PHI), -2 / 3 * sum(psi_ph[ph] * math.sin(a[ph]) for ph in PHI))


# ---------------- FEMM worker ----------------
RB = dict(dr=0.5, da=0.5, r0=M.ROTOR['shaft_diameter'] / 2, r1=M.R_ROT)
NR = int((RB['r1'] - RB['r0']) / RB['dr']) + 1; NA = int(360 / RB['da'])


def read_fields(fn, th_m):
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
    area = np.abs(a2) / 2                                      # m^2
    mat = np.array([int(labs[l][2]) for l in el[:, 3]])        # 1 Air 2 Cu 3 M235 4 SCM420 5 Magnet
    cx, cy = nodes[el[:, :3], 0].mean(1), nodes[el[:, :3], 1].mean(1); r = np.hypot(cx, cy)
    mass = area * M.STACK * 1e-3 * M.STACKING * DENS
    st = (mat == 3) & (r > M.R_BORE - 0.05)
    rt = (mat == 3) & (r < M.R_MID)
    cu = mat == 2
    Bh = math.sqrt(2) * B
    out = dict(st_S2=float(np.sum(mass[st] * Bh[st] ** 2)), st_Sa=float(np.sum(mass[st] * Bh[st] ** ALPHA)),
               st_mass=float(mass[st].sum()), cu_B2=float(np.sum(area[cu] * B[cu] ** 2) / np.sum(area[cu])),
               tooth_B=float(np.percentile(B[st & (r < M.slot_geometry(M.STATOR)['rb'])], 99)))
    # rotor field on a rotor-fixed polar grid (to get the slot-ripple swing between rotor positions)
    ang = (np.degrees(np.arctan2(cy, cx)) - th_m) % 360
    ir = np.clip(((r[rt] - RB['r0']) / RB['dr']).astype(int), 0, NR - 1); ia = np.minimum((ang[rt] / RB['da']).astype(int), NA - 1)
    idx = ir * NA + ia
    sB = np.bincount(idx, B[rt] * area[rt], NR * NA); sA = np.bincount(idx, area[rt], NR * NA)
    out['rot_B'] = np.where(sA > 0, sB / np.maximum(sA, 1e-30), np.nan).astype(np.float32).tolist()
    out['rot_m'] = (sA * M.STACK * 1e-3 * M.STACKING * DENS).astype(np.float32).tolist()
    return out


def job(j):
    th, i_d, i_q = j
    M.set_angle(th)
    ip = phase_currents(th, i_d, i_q)
    for ph, teeth in M.PHASE_TEETH.items():
        for t in teeth:
            femm.mi_modifycircprop('T%d' % t, 1, SIGN[ph][t] * ip[ph])
    femm.mi_analyze(1); femm.mi_loadsolution()
    psi_c = [femm.mo_getcircuitproperties('T%d' % k)[2] for k in range(1, M.NS + 1)]
    femm.mo_groupselectblock(1); T = femm.mo_blockintegral(22); femm.mo_clearblock()
    femm.mo_close()
    psi_ph = {ph: sum(SIGN[ph][t] * psi_c[t - 1] for t in teeth) / M.PATHS for ph, teeth in M.PHASE_TEETH.items()}
    f = read_fields(M.W['file'].replace('.fem', '.ans'), th)
    return dict(th=th, id=i_d, iq=i_q, T=T, psi_ph=psi_ph, **f)


def solve():
    from multiprocessing import Pool
    import psutil
    t0 = time.time()
    pts = [(t, float(a), float(b)) for a in IDS for b in IQS for t in POS]
    ck = os.path.join(H, 'work', 'dq100_partial.json')
    done = json.load(open(ck)) if os.path.exists(ck) else []
    with Pool(M.NWORK, initializer=M.worker_init, initargs=(True,)) as pool:
        for r in pool.imap(job, pts[len(done):], chunksize=1):
            done.append(r); json.dump(done, open(ck, 'w'))
            print(time.strftime('%H:%M:%S'), '%d/%d' % (len(done), len(pts)), 'id %.0f iq %.0f th %.1f T %.2f' % (r['id'], r['iq'], r['th'], r['T']), flush=True)
    RG.kill_registered()                                   # only this run's FEMM windows
    # reduce: average over rotor positions
    na, nb = len(IDS), len(IQS)
    g = {k: np.zeros((na, nb)) for k in ('psid', 'psiq', 'T', 'st_S2', 'st_Sa', 'cu_B2', 'tooth_B', 'rot_Sa', 'rot_S2')}
    rot = {}
    for r in done:
        a, b = int(np.argmin(abs(IDS - r['id']))), int(np.argmin(abs(IQS - r['iq'])))
        d, q = park(r['th'], r['psi_ph'])
        g['psid'][a, b] += d / len(POS); g['psiq'][a, b] += q / len(POS); g['T'][a, b] += r['T'] / len(POS)
        for k in ('st_S2', 'st_Sa', 'cu_B2', 'tooth_B'):
            g[k][a, b] += r[k] / len(POS)
        rot.setdefault((a, b), []).append((np.array(r['rot_B']), np.array(r['rot_m'])))
    for (a, b), lst in rot.items():
        Bs = np.stack([x[0] for x in lst]); m = np.nanmean(np.stack([x[1] for x in lst]), 0)
        dB = (np.nanmax(Bs, 0) - np.nanmin(Bs, 0)) / 2          # slot-ripple amplitude in the rotor
        ok = ~np.isnan(dB)
        g['rot_Sa'][a, b] = float(np.sum(m[ok] * dB[ok] ** ALPHA)); g['rot_S2'][a, b] = float(np.sum(m[ok] * dB[ok] ** 2))
    sgn = 1 if g['T'][-1, -1] > 0 else -1                        # id 0, iq max must motor
    g['T'] *= sgn
    out = dict(id=IDS.tolist(), iq=IQS.tolist(), Br=M.BR, magnet_T=M.MAG_T, positions=POS, torque_sign=sgn,
               st_mass=done[0]['st_mass'], **{k: v.tolist() for k, v in g.items()})
    json.dump(out, open(os.path.join(H, 'dq_100C.json'), 'w'), indent=1)
    print('solved in %.0f min' % ((time.time() - t0) / 60))


# ---------------- Lab post-processing ----------------
def lab():
    from scipy.interpolate import RegularGridInterpolator
    m = json.load(open(os.path.join(H, 'dq_100C.json')))
    grid = (np.array(m['id']), np.array(m['iq']))
    F = {k: RegularGridInterpolator(grid, np.array(m[k]), method='cubic') for k in ('psid', 'psiq', 'T', 'st_S2', 'st_Sa', 'cu_B2', 'rot_Sa', 'rot_S2')}
    i_d, i_q = np.meshgrid(np.linspace(-I_PATH_PK, 0, 241), np.linspace(0, I_PATH_PK, 241), indexing='ij')
    ok = np.hypot(i_d, i_q) <= I_PATH_PK + 1e-9
    Pt = np.column_stack([i_d[ok], i_q[ok]])
    V = {k: f(Pt) for k, f in F.items()}
    psd, psq, Tem = V['psid'] + L_EW_PATH * Pt[:, 0], V['psiq'] + L_EW_PATH * Pt[:, 1], V['T']
    Ipk = np.hypot(Pt[:, 0], Pt[:, 1])
    I_line = Ipk * 2 * math.sqrt(3) / math.sqrt(2)
    pcu = 3 * R_PATH * Ipk ** 2                                  # 6 paths x R x Ipk^2/2
    speeds = np.arange(0, 12001, 100)
    tq_grid = np.arange(0.5, 60.01, 0.5)
    keys = ('eff', 'loss', 'pcu', 'pac', 'pfe_st', 'pfe_rot', 'pmech', 'I_line', 'V_ph', 'pf', 'I_dc', 'P_in', 'id', 'iq', 'gamma')
    mp = {k: np.full((len(tq_grid), len(speeds)), np.nan) for k in keys}
    env = []
    for a, n in enumerate(speeds):
        wm = n * 2 * math.pi / 60; we = P * wm; f = we / (2 * math.pi)
        vd = R_PATH * Pt[:, 0] - we * psq; vq = R_PATH * Pt[:, 1] + we * psd
        v = np.hypot(vd, vq)
        pfe_st = BF * (KH * f * V['st_Sa'] + KE * f * f * V['st_S2'])
        fs = 12 * n / 60
        pfe_rot = BF * (KH * fs * V['rot_Sa'] + KE * fs * fs * V['rot_S2'])
        pac = AC_ADJ * N_COND * math.pi * 0.063 * D_CU ** 4 * (2 * math.pi * f) ** 2 * (2 * np.maximum(V['cu_B2'], 0)) / (128 * RHO20 * (1 + ALPHA_CU * (T_WDG - 20)))
        pme = mech_loss(n)
        pshaft = Tem * wm - pfe_st - pfe_rot - pme
        loss = pcu + pac + pfe_st + pfe_rot + pme
        pin = pshaft + loss                                        # = Tem*wm + pcu + pac
        tsh = pshaft / wm if n > 0 else Tem.copy()
        feas = (v <= V_MAX) & (pin <= P_DC_MAX + 1e-6) & (tsh > 0)
        if not feas.any():
            break
        # MTPA / field weakening: for a torque, the least-current feasible point (Motor-CAD Lab control strategy MTPA)
        kmax = np.flatnonzero(feas)[np.argmax(tsh[feas])]
        env.append(dict(rpm=int(n), T=float(tsh[kmax]), P_kW=float(pshaft[kmax] / 1e3), I_line=float(I_line[kmax]), V=float(v[kmax]),
                        I_dc=float(pin[kmax] / VDC), id=float(Pt[kmax, 0]), iq=float(Pt[kmax, 1]),
                        eff=float(100 * pshaft[kmax] / pin[kmax]) if n > 0 else None,
                        pcu=float(pcu[kmax]), pac=float(pac[kmax]), pfe=float(pfe_st[kmax] + pfe_rot[kmax]), pmech=float(pme),
                        limit=('voltage' if v[kmax] > V_MAX - 0.3 else '') + (' dc-current' if pin[kmax] > P_DC_MAX - 50 else '') + (' current' if Ipk[kmax] > I_PATH_PK - 1 else '')))
        if n == 0:
            continue
        for b, T in enumerate(tq_grid):
            c = np.flatnonzero(feas & (np.abs(tsh - T) <= 0.25))
            if not len(c):
                continue
            k = c[np.argmin(Ipk[c])]
            vals = dict(eff=100 * pshaft[k] / pin[k], loss=loss[k], pcu=pcu[k], pac=pac[k], pfe_st=pfe_st[k], pfe_rot=pfe_rot[k],
                        pmech=pme, I_line=I_line[k], V_ph=v[k], pf=(vd[k] * Pt[k, 0] + vq[k] * Pt[k, 1]) / (v[k] * Ipk[k]) if Ipk[k] > 0 else np.nan,
                        I_dc=pin[k] / VDC, P_in=pin[k], id=Pt[k, 0], iq=Pt[k, 1], gamma=math.degrees(math.atan2(-Pt[k, 0], Pt[k, 1])))
            for kk, vv in vals.items():
                mp[kk][b, a] = vv
    ids, iqs = np.array(m['id']), np.array(m['iq']); pd_, pq_ = np.array(m['psid']), np.array(m['psiq'])
    psim = pd_[-1, 0]
    info = dict(psi_m=psim, Ld_unsat_phase_uH=float((pd_[-2, 0] - psim) / ids[-2] * 1e6 / 2),
                Lq_unsat_phase_uH=float(pq_[-1, 1] / iqs[1] * 1e6 / 2), R_phase_100C_mOhm=R_COIL * 1e3, R_path=R_PATH,
                V_max=V_MAX, I_path_pk=I_PATH_PK, P_dc_max=P_DC_MAX, st_mass=m['st_mass'], Br=m['Br'])
    res = dict(settings=dict(VDC=VDC, MI=MI, I_line_rms_max=I_LINE_RMS, I_dc_max=IDC_MAX, T_wdg=T_WDG, T_mag=m['magnet_T'],
                             control='MTPA (least current per torque, field weakening when voltage-limited)', loss=LOSS),
               info=info, speeds=speeds.tolist(), torques=tq_grid.tolist(), envelope=env, dq=m,
               maps={k: np.where(np.isnan(v), None, np.round(v, 4)).tolist() for k, v in mp.items()})
    json.dump(res, open(os.path.join(H, 'lab_100C.json'), 'w'))
    e = mp['eff']; k = np.unravel_index(np.nanargmax(e), e.shape)
    print(json.dumps(info, indent=1))
    print('peak efficiency %.2f %% at %d rpm, %.1f N.m' % (e[k], speeds[k[1]], tq_grid[k[0]]))
    for rw in env:
        if rw['rpm'] % 1000 == 0:
            print('%6d rpm  T %6.2f  P %6.2f kW  I %5.0f A  Idc %5.0f A  V %5.1f  eff %s  %s' % (rw['rpm'], rw['T'], rw['P_kW'], rw['I_line'], rw['I_dc'], rw['V'],
                  '%.1f' % rw['eff'] if rw['eff'] else '-', rw['limit']))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'solve':
        solve()
    else:
        lab()
