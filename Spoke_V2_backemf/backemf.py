"""No-load back-EMF of Motor-CAD model E:\\Spoke_V2\\ADAPTIVE_MODEL_COMPLETE_4.mot, solved in FEMM 4.2 (2D magnetostatic).

Stator: Motor-CAD 'Parallel Tooth', 12 slots.  Rotor: adaptive spoke template (same parameter names as
Ferrite_Motor_design/rotor/spoke_rotor_explorer.html).  Each tooth coil is its own FEMM circuit at 0 A, so the
coil flux linkage is read directly; phase = 2 adjacent coils in series per path, 2 paths in parallel.

Run:  python backemf.py      writes results.json + backemf.png
"""
import json, math, os, sys, time
import numpy as np
import femm

HERE = os.path.dirname(os.path.abspath(__file__))
MU0 = 4e-7 * math.pi

# ---------------- Motor-CAD inputs (screenshots 2026-10-10) ----------------
STACK = 63.0                       # stator lam length = rotor lam length = magnet length
STATOR = dict(slots=12, od=160.0, bore=107.0, tooth_w=15.5, slot_depth=17.5, corner_r=1.95,
              tip_depth=0.8, slot_open=5.0, tip_angle=20.0)
ROTOR = dict(pole_number=10, rotor_diameter=106.0, shaft_diameter=25.0, magnet_inset=2.0, magnet_depth=28.0,
             magnet_opening=8.5, magnet_X=18.0, magnet_Y=10.5, magnet_Z=6.0, magnet_glue_gap=0.1,
             Bridge_thickness=1.0, bridge_offset=4.0)
TURNS = 10                         # per coil; 20 conductors/slot, 2 layers
PATHS = 2
BR20, MU_MAG, BR_TC = 0.46, 1.08, -0.001797    # DMEGC_DM4654; Motor-CAD magnet flux coefficient -0.1797 %/K
MAG_T = float(os.environ.get('MAG_T', 20.0))
BR = BR20 * (1 + BR_TC * (MAG_T - 20))
STACKING = 0.97                    # Motor-CAD default; consistent with the lamination masses shown
# M235-35A from the Motor-CAD material database (Steel BH tab), H [A/m] -> B [T]; above 1.8 T extended with
# slope mu0 (air-like), as Motor-CAD extrapolates beyond the last point
BH_LAM_MCAD = [(0, 0), (24.7, 0.1), (32.6, 0.2), (38.1, 0.3), (43.1, 0.4), (48.2, 0.5), (53.9, 0.6), (60.7, 0.7),
               (68.8, 0.8), (79.3, 0.9), (93.7, 1.0), (115, 1.1), (156, 1.2), (260, 1.3), (690, 1.4), (1950, 1.5),
               (4410, 1.6), (7630, 1.7), (12000, 1.8)]
BH_LAM_MCAD += [(h, 1.8 + 4e-7 * math.pi * (h - 12000)) for h in (30000, 60000, 120000, 250000, 500000)]
# M235-35A typical DC curve (first runs; kept for reference)
BH_LAM_TYPICAL = [(0, 0), (50, 0.75), (100, 1.00), (150, 1.12), (200, 1.20), (300, 1.29), (500, 1.38), (1000, 1.45),
          (2500, 1.53), (5000, 1.63), (10000, 1.75), (20000, 1.87), (50000, 2.02), (100000, 2.13), (200000, 2.30)]
BH_LAM = BH_LAM_MCAD if os.environ.get('BH', 'mcad') == 'mcad' else BH_LAM_TYPICAL
# SCM420 (Cr-Mo case-hardening steel) typical curve
BH_SHAFT = [(0, 0), (200, 0.45), (400, 0.85), (700, 1.10), (1000, 1.25), (2000, 1.45), (4000, 1.58),
            (8000, 1.70), (16000, 1.82), (50000, 2.00), (100000, 2.10), (200000, 2.25)]
FILLETS = dict(lam=0.3, pocket=1.0, slot=1.95) if os.environ.get('FILLETS', '1') == '1' else None
GM = float(os.environ.get('GAPMESH', '1.0'))   # air-gap mesh scale (1 = 0.2 mm elements)
STEP = 2.0                         # deg mech; 72 deg = one electrical period
NWORK = 4

R_BORE = STATOR['bore'] / 2
R_ROT = ROTOR['rotor_diameter'] / 2
R_MID = (R_BORE + R_ROT) / 2
AGE = os.environ.get('AGE', '0') == '1'
R_AGE_IN, R_AGE_OUT = R_ROT + (R_BORE - R_ROT) / 3, R_BORE - (R_BORE - R_ROT) / 3
NS = STATOR['slots']
# tooth coil k sits between slot k and slot k+1 (Motor-CAD slot table: slot 1 = 20 x phase 1, ...)
PHASE_TEETH = {'Ph1': (12, 1, 6, 7), 'Ph2': (2, 3, 8, 9), 'Ph3': (4, 5, 10, 11)}


def rot(p, a_deg):
    a = math.radians(a_deg); c, s = math.cos(a), math.sin(a)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def line(a, b):
    femm.mi_drawline(a[0], a[1], b[0], b[1])


def polyline(pts):
    for a, b in zip(pts, pts[1:]):
        line(a, b)


def arc_ccw(r, a0, a1, maxseg, bdry=''):
    p, q = rot((r, 0), a0), rot((r, 0), a1)
    femm.mi_drawarc(p[0], p[1], q[0], q[1], a1 - a0, maxseg)
    if bdry:
        m = rot((r, 0), (a0 + a1) / 2)
        femm.mi_selectarcsegment(m[0], m[1]); femm.mi_setarcsegmentprop(maxseg, bdry, 0, 0); femm.mi_clearselected()


def label(p, mat, mesh, circ='', magdir=0, turns=0, grp=0):
    femm.mi_addblocklabel(p[0], p[1]); femm.mi_selectlabel(p[0], p[1])
    femm.mi_setblockprop(mat, 0, mesh, circ, magdir, grp, turns); femm.mi_clearselected()


def ang(p):
    return math.degrees(math.atan2(p[1], p[0]))


def slot_geometry(s):
    """Right half of one slot, slot centreline on +y (x > 0 = clockwise side). Returns key points."""
    hs, ws = 180 / s['slots'], s['tooth_w'] / 2
    so = s['slot_open'] / 2
    A = (so, math.sqrt(R_BORE ** 2 - so ** 2))                      # tip corner on the bore
    B = (so, A[1] + s['tip_depth'])
    n = (math.cos(math.radians(hs)), -math.sin(math.radians(hs)))   # tooth normal; tooth side: p.n = -ws
    u = (math.sin(math.radians(hs)), math.cos(math.radians(hs)))    # tooth axis
    d = (math.cos(math.radians(s['tip_angle'])), math.sin(math.radians(s['tip_angle'])))
    nB = B[0] * n[0] + B[1] * n[1]
    t = (-ws - nB) / (d[0] * n[0] + d[1] * n[1])
    C = (B[0] + t * d[0], B[1] + t * d[1])
    rb = R_BORE + s['slot_depth']                                    # slot bottom radius
    sE = math.sqrt(rb ** 2 - ws ** 2)
    E = (-ws * n[0] + sE * u[0], -ws * n[1] + sE * u[1])
    return dict(A=A, B=B, C=C, E=E, rb=rb)


def tooth_area(sg, s):
    """Lamination area of one tooth (between slot-bottom arc and bore) for the mass check."""
    from matplotlib.path import Path
    mir = lambda p: (-p[0], p[1])
    hs = 180 / s['slots']
    right = [sg['A'], sg['B'], sg['C'], sg['E']]                     # wall of slot centred at +y
    left_next = [rot(mir(p), -2 * hs) for p in right]                # same tooth, wall of the next slot clockwise
    poly = right + [rot((sg['rb'], 0), a) for a in np.linspace(ang(sg['E']), ang(left_next[3]), 40)] + \
        left_next[::-1] + [rot((R_BORE, 0), a) for a in np.linspace(ang(left_next[0]), ang(sg['A']), 40)]
    x, y = np.array(poly).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def rotor_geometry(p):
    """Same as Ferrite_Motor_design/femm_em/femm_motor.py (port of spoke_rotor_explorer.html), pole on +y."""
    t, rs, rr = p['Bridge_thickness'] / 2, p['shaft_diameter'] / 2, p['rotor_diameter'] / 2
    rom = rr - p['magnet_inset']; rim = rom - p['magnet_depth']; ho = p['magnet_opening'] / 2
    a = math.radians(180 / p['pole_number']); T, Cs = math.tan(a), math.cos(a)
    P = (-(rim * T - t / Cs), rim); Q = (-p['magnet_Y'], rim + p['magnet_X']); R = (-p['magnet_Z'], rom)
    D = (-ho, rom); C = (-ho, math.sqrt(rr * rr - ho * ho))
    yg = rs + p['bridge_offset']; G = (-(yg * T - t / Cs), yg)
    gap, ctr = p['magnet_glue_gap'], (0, (rim + rom) / 2)

    def off(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy); nx, ny = -dy / L, dx / L
        if (ctr[0] - (a[0] + b[0]) / 2) * nx + (ctr[1] - (a[1] + b[1]) / 2) * ny < 0: nx, ny = -nx, -ny
        return (a[0] + nx * gap, a[1] + ny * gap), (b[0] + nx * gap, b[1] + ny * gap)

    def X(l1, l2):
        (a, b), (c, d) = l1, l2; r = (b[0] - a[0], b[1] - a[1]); q = (d[0] - c[0], d[1] - c[1])
        k = ((c[0] - a[0]) * q[1] - (c[1] - a[1]) * q[0]) / (r[0] * q[1] - r[1] * q[0])
        return (a[0] + r[0] * k, a[1] + r[1] * k)

    PQ, QR = off(P, Q), off(Q, R)
    Pm, Qm, Rm = X(PQ, ((0, rim), (1, rim))), X(PQ, QR), X(QR, ((0, rom), (1, rom)))
    return dict(P=P, Q=Q, R=R, D=D, C=C, G=G, Pm=Pm, Qm=Qm, Rm=Rm, rs=rs, rr=rr, rim=rim, rom=rom, yg=yg)


MAGLAB, ANGLE = [], [0.0]
FILLET_OK, FILLET_FAIL = set(), set()


def build(path, shaft_magnetic):
    s = STATOR
    femm.newdocument(0)
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, STACK, 30, 0)
    femm.mi_smartmesh(0)
    femm.mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    femm.mi_addmaterial('Copper', 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0, 0, 0)
    femm.mi_addmaterial('M235-35A', 5000, 5000, 0, 0, 0, 0.35, 0, STACKING, 0, 0, 0, 0, 0)
    for h, b in BH_LAM:
        femm.mi_addbhpoint('M235-35A', b, h)
    femm.mi_addmaterial('SCM420', 1000, 1000, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    for h, b in BH_SHAFT:
        femm.mi_addbhpoint('SCM420', b, h)
    femm.mi_addmaterial('Magnet', MU_MAG, MU_MAG, BR / (MU0 * MU_MAG), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    femm.mi_addboundprop('A0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for k in range(1, NS + 1):
        femm.mi_addcircprop('T%d' % k, 0, 1)

    # ---- stator ----
    sg = slot_geometry(s)
    mir = lambda p: (-p[0], p[1])
    pitch = 360 / NS
    for k in range(1, NS + 1):
        a = pitch * (k - 1)                         # slot k centreline at 90 + a deg
        L = lambda p: rot(p, a)
        A, B, C, E = (sg[q] for q in 'ABCE')
        polyline([L(p) for p in (A, B, C, E)])
        polyline([L(mir(p)) for p in (A, B, C, E)])
        arc_ccw(sg['rb'], 90 + a - (90 - ang(E)), 90 + a + (90 - ang(E)), 1)       # slot bottom
        arc_ccw(R_BORE, 90 + a - (90 - ang(A)), 90 + a + (90 - ang(A)), 0.5 * GM)       # slot mouth
        arc_ccw(R_BORE, 90 + a + (90 - ang(A)), 90 + a + pitch - (90 - ang(A)), 0.5 * GM)  # tooth face
        line(L(C), L(mir(C)))                                               # wedge / coil split
        line(L((0, C[1])), L((0, sg['rb'])))                                # layer split (side by side)
        label(L((0, (A[1] + C[1]) / 2)), 'Air', 0.5)
        ym = (C[1] + sg['rb']) / 2
        # left half (CCW side) -> coil on tooth k (between slot k and k+1); right half -> tooth k-1
        label(L((-3.5, ym)), 'Copper', 1.2, 'T%d' % k, 0, TURNS)
        label(L((3.5, ym)), 'Copper', 1.2, 'T%d' % ((k - 2) % NS + 1), 0, -TURNS)
    arc_ccw(s['od'] / 2, 0, 180, 2, 'A0'); arc_ccw(s['od'] / 2, 180, 360, 2, 'A0')
    label((0, -(sg['rb'] + s['od'] / 2) / 2), 'M235-35A', 1.5)
    if AGE:   # FEMM air-gap element: meshes stay fixed, rotor angle set through the boundary (no remeshing)
        femm.mi_addboundprop('AGE', 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0)
        for rr_ in (R_AGE_IN, R_AGE_OUT):
            arc_ccw(rr_, 0, 180, 0.4 * GM, 'AGE'); arc_ccw(rr_, 180, 360, 0.4 * GM, 'AGE')
        label(rot(((R_AGE_OUT + R_BORE) / 2, 0), 90 + pitch / 2), 'Air', 0.2 * GM)
        p_ = rot(((R_AGE_IN + R_AGE_OUT) / 2, 0), 90 + pitch / 2)
        femm.mi_addblocklabel(*p_); femm.mi_selectlabel(*p_); femm.mi_setblockprop('<No Mesh>', 1, 0, '', 0, 0, 0); femm.mi_clearselected()
    else:
        arc_ccw(R_MID, 0, 180, 0.4 * GM); arc_ccw(R_MID, 180, 360, 0.4 * GM)
        label(rot(((R_MID + R_BORE) / 2, 0), 90 + pitch / 2), 'Air', 0.2 * GM)

    # ---- rotor (group 1) ----
    g = rotor_geometry(ROTOR)
    n = ROTOR['pole_number']; pp = 360 / n
    dC = math.degrees(math.atan2(-g['C'][0], g['C'][1]))
    rG = math.hypot(*g['G']); dG = math.degrees(math.atan2(-g['G'][0], g['G'][1]))
    MAGLAB.clear()
    for k in range(n):
        a = pp * k - 90
        L = lambda p: rot(p, a)
        cav = [g['G'], g['P'], g['Q'], g['R'], g['Rm'], g['Qm'], g['Pm']]
        polyline([L(p) for p in cav] + [L(mir(p)) for p in reversed(cav)])
        polyline([L(p) for p in (g['Rm'], g['D'], mir(g['D']), mir(g['Rm']))])
        line(L(g['D']), L(g['C'])); line(L(mir(g['D'])), L(mir(g['C'])))
        arc_ccw(rG, pp * k - dG, pp * k + dG, 1)
        arc_ccw(g['rr'], pp * k + dC, pp * (k + 1) - dC, 0.4 * GM)
        mdir = a + (0 if k % 2 == 0 else 180)
        MAGLAB.append((L((0, (g['rim'] + g['rom']) / 2)), mdir))
        label(MAGLAB[-1][0], 'Magnet', 1.0, magdir=mdir, grp=1)
        label(L((0, (rG + g['rim']) / 2)), 'Air', 0.8, grp=1)
    if FILLETS:   # Motor-CAD template fillets: lamination 0.3 (mouth at OD), pocket 1.0; slot corner 1.95
        for k in range(n):
            L = lambda p: rot(p, pp * k - 90)
            for p, rad in ((g['C'], FILLETS['lam']), (g['P'], FILLETS['pocket']), (g['G'], FILLETS['pocket'])):
                for q in (p, mir(p)):
                    try:
                        femm.mi_createradius(*L(q), rad); FILLET_OK.add(str(p))
                    except Exception:
                        FILLET_FAIL.add(str(p))
        for k in range(1, NS + 1):
            for q in (sg['E'], mir(sg['E'])):
                try:
                    femm.mi_createradius(*rot(q, pitch * (k - 1)), FILLETS['slot']); FILLET_OK.add('E')
                except Exception:
                    FILLET_FAIL.add('E')
    arc_ccw(g['rs'], 0, 180, 2); arc_ccw(g['rs'], 180, 360, 2)
    label((0, 0), 'SCM420' if shaft_magnetic else 'Air', 3, grp=1)
    label(rot(((g['rs'] + g['yg']) / 2 + 1, 0), pp / 2 - 90 + pp / 2), 'M235-35A', 1.0, grp=1)
    label(rot(((g['rr'] + (R_AGE_IN if AGE else R_MID)) / 2, 0), pp / 2), 'Air', 0.2 * GM, grp=1)
    femm.mi_selectcircle(0, 0, R_MID - 0.01, 4); femm.mi_setgroup(1); femm.mi_clearselected()
    ANGLE[0] = 0.0
    femm.mi_zoomnatural()
    femm.mi_saveas(path)
    return sg, g


def set_angle(theta_m):
    if AGE:
        femm.mi_modifyboundprop('AGE', 10, theta_m); ANGLE[0] = theta_m
        return
    d = theta_m - ANGLE[0]
    if abs(d) > 1e-9:
        femm.mi_selectgroup(1); femm.mi_moverotate(0, 0, d); femm.mi_clearselected()
        ANGLE[0] = theta_m
    for p, mdir in MAGLAB:
        q = rot(p, theta_m)
        femm.mi_selectlabel(q[0], q[1]); femm.mi_setblockprop('Magnet', 0, 1.0, '', mdir + theta_m, 1, 0)
        femm.mi_clearselected()


W = {}


def worker_init(shaft_magnetic):
    femm.openfemm(1)
    import runguard as _RG; _RG.register_femm()
    os.makedirs(os.path.join(HERE, 'work'), exist_ok=True)
    W['file'] = os.path.join(HERE, 'work', 'w%d_%d.fem' % (os.getpid(), shaft_magnetic))
    build(W['file'], shaft_magnetic)


def job(theta_m):
    set_angle(theta_m)
    femm.mi_analyze(1); femm.mi_loadsolution()
    psi = [femm.mo_getcircuitproperties('T%d' % k)[2] for k in range(1, NS + 1)]
    femm.mo_groupselectblock(1); T = femm.mo_blockintegral(22); femm.mo_clearblock()
    femm.mo_close()
    return theta_m, psi, T


def analyse(th, psi, p=5):
    """psi[i][k]: coil k flux linkage. Returns per-phase path flux linkage and EMF waveforms per 1000 rpm."""
    psi = np.array(psi)
    the = np.radians(p * th)
    X = 2 * np.mean(psi * np.exp(-1j * the)[:, None], axis=0)   # coil fundamental phasors
    out = {}
    for ph, teeth in PHASE_TEETH.items():
        ref = X[teeth[0] - 1]
        sg = [1 if np.real(X[t - 1] * np.conj(ref)) > 0 else -1 for t in teeth]
        coils = [sg[i] * psi[:, t - 1] for i, t in enumerate(teeth)]
        path = (coils[0] + coils[1] + coils[2] + coils[3]) / PATHS          # 2 coils in series per path
        out[ph] = path
    # phase polarity: Ph2/Ph3 must sit +-120 deg el from Ph1 (not 60), as in a balanced 3-phase winding
    f = {ph: np.mean(v * np.exp(-1j * the)) for ph, v in out.items()}
    for ph in ('Ph2', 'Ph3'):
        if np.real(f[ph] * np.conj(f['Ph1'])) > 0:
            out[ph] = -out[ph]
    w =2 * math.pi * 1000 / 60                                          # rad/s mech at 1000 rpm
    dth = np.radians(th[1] - th[0])
    emf = {ph: -np.gradient(np.concatenate([v, v[:1]]), dth)[:-1] * w for ph, v in out.items()}
    # periodic central difference
    for ph, v in out.items():
        emf[ph] = -(np.roll(v, -1) - np.roll(v, 1)) / (2 * dth) * w
    return out, emf, X


def harmonics(y):
    c = np.fft.rfft(y) / len(y) * 2
    return np.abs(c)


def main():
    from multiprocessing import Pool
    log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
    th = np.arange(0, 72, STEP)
    res = dict(inputs=dict(stator=STATOR, rotor=ROTOR, stack=STACK, Br20=BR20, mu_r=MU_MAG, magnet_T=MAG_T, Br=BR,
                           stacking=STACKING, turns_per_coil=TURNS, paths=PATHS))
    cases = [('shaft_magnetic', True), ('shaft_nonmagnetic', False)]
    for name, shaft in cases:
        t0 = time.time()
        with Pool(NWORK, initializer=worker_init, initargs=(shaft,)) as pool:
            out = pool.map(job, th.tolist(), chunksize=1)
        psi = [o[1] for o in out]; tq = [o[2] for o in out]
        path, emf, X = analyse(th, psi)
        r = dict(theta_m=th.tolist(), coil_psi=psi, cogging=tq,
                 path_psi={k: v.tolist() for k, v in path.items()}, emf_phase_1000rpm={k: v.tolist() for k, v in emf.items()})
        e1 = emf['Ph1']; e2 = emf['Ph2']; e3 = emf['Ph3']
        ll = e1 - e2
        h = harmonics(e1)                       # index = electrical harmonic (one el. period sampled)
        r['summary'] = dict(
            psi_path_fund_peak=float(harmonics(path['Ph1'])[1]),
            E_phase_rms=float(np.sqrt(np.mean(e1 ** 2))), E_phase_peak=float(np.max(np.abs(e1))),
            E_phase_fund_peak=float(h[1]), E_phase_fund_rms=float(h[1] / math.sqrt(2)),
            E_line_rms=float(np.sqrt(np.mean(ll ** 2))), E_line_peak=float(np.max(np.abs(ll))),
            E_line_fund_rms=float(harmonics(ll)[1] / math.sqrt(2)),
            phase_rms_all=[float(np.sqrt(np.mean(e ** 2))) for e in (e1, e2, e3)],
            THD_phase=float(np.sqrt(np.sum(h[2:12] ** 2)) / h[1]),
            harmonics_phase_pct={int(k): float(100 * h[k] / h[1]) for k in (3, 5, 7, 9, 11)},
            cogging_pkpk=float(np.ptp(tq)), solve_minutes=(time.time() - t0) / 60)
        res[name] = r
        log(name, json.dumps(r['summary']))
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    plot(res)


def plot(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, ls in (('shaft_magnetic', '-'), ('shaft_nonmagnetic', '--')):
        r = res[name]; the = 5 * np.array(r['theta_m'])
        for ph, c in zip(('Ph1', 'Ph2', 'Ph3'), ('r', 'g', 'b')):
            ax[0].plot(the, r['emf_phase_1000rpm'][ph], c + ls, lw=1.2, label='%s %s' % (ph, name) if ph == 'Ph1' or name == 'shaft_magnetic' else None)
        e = np.array(r['emf_phase_1000rpm']['Ph1']) - np.array(r['emf_phase_1000rpm']['Ph2'])
        ax[1].plot(the, e, 'k' + ls, label='Line 1-2 (star) %s' % name)
    ax[0].set(title='Phase back-EMF @ 1000 rpm, magnets %.0f C' % res['inputs']['magnet_T'], xlabel='electrical deg', ylabel='V')
    ax[1].set(title='Line-line back-EMF @ 1000 rpm (star)', xlabel='electrical deg', ylabel='V')
    for a in ax: a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'backemf.png'), dpi=130)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        sg = slot_geometry(STATOR)
        A = tooth_area(sg, STATOR)
        mass = A * 1e-6 * STACK * 1e-3 * 7650 * STACKING * NS
        yoke = math.pi * ((STATOR['od'] / 2) ** 2 - sg['rb'] ** 2) * 1e-6 * STACK * 1e-3 * 7650 * STACKING
        print('slot pts', {k: (round(v[0], 3), round(v[1], 3)) if isinstance(v, tuple) else v for k, v in sg.items()})
        print('tooth area %.1f mm2 -> teeth %.3f kg (Motor-CAD 1.615), yoke %.3f kg (Motor-CAD 1.996)' % (A, mass, yoke))
        g = rotor_geometry(ROTOR)
        print('rotor', {k: (round(v[0], 3), round(v[1], 3)) if isinstance(v, tuple) else round(v, 3) for k, v in g.items()})
        femm.openfemm(1); build(os.path.join(HERE, 'check.fem'), True)
        t0 = time.time(); femm.mi_analyze(1); femm.mi_loadsolution()
        print('solve %.1f s' % (time.time() - t0))
        print('psi', [round(femm.mo_getcircuitproperties('T%d' % k)[2], 5) for k in range(1, NS + 1)])
        femm.mo_showdensityplot(1, 0, 2.0, 0, 'mag'); femm.mo_zoom(-82, -82, 82, 82)
        femm.mo_savebitmap(os.path.join(HERE, 'check_B.bmp'))
        femm.mo_zoom(-30, 20, 30, 80); femm.mo_savebitmap(os.path.join(HERE, 'check_B_zoom.bmp'))
        femm.closefemm()
    else:
        main()
