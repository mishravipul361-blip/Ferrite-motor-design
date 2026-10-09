"""2D magnetostatic FEMM model of the v7 motor: 12-slot segmented stator, 10-pole spoke ferrite rotor.

Stator: exact tooth profile + conductor positions from cad/fieldlab_v7.html (same CAD data).
Rotor : rotor/spoke_rotor_explorer.html defaults (Adaptive_temp_24_09.py parameters).
Rotation: the rotor group (everything inside the mid-gap circle) is turned with mi_moverotate;
torque = weighted Maxwell stress tensor on the rotor. (FEMM's air-gap element solved ~50x slower here.)

Run:  python femm_motor.py    (~1 h on a 4-core laptop; writes results.json), then python plot_results.py
"""
import hashlib, json, math, os, re, sys, threading, time
import numpy as np
import femm
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------- inputs ----------------
STACK = 65.0                 # mm, axial
I_LINE_RMS = 360.0           # A rms, line (delta)
VDC = 52.0                   # V, battery
R_COIL = {20: 5.122e-3, 100: 6.732e-3}   # ohm per coil = per phase (RESISTANCE_v7.txt, 2.90 mm wire)
WIRE_D = 2.90                # mm bare
MU0 = 4e-7 * math.pi
BR20, HCJ = 0.45, 340e3      # high-grade sintered ferrite (Y40 class) at 20 C
BR_TC = -0.0020              # /K, ferrite Br temperature coefficient
# calibration knobs live in motor_config.json (MOTOR_CFG env var overrides, used by calibrate.py)
CFG = dict(magnet_temp_C=20.0, stacking=0.97, mu_mag=1.05)
if os.path.exists(os.path.join(HERE, 'motor_config.json')):
    CFG.update(json.load(open(os.path.join(HERE, 'motor_config.json'))))
CFG.update(json.loads(os.environ.get('MOTOR_CFG', '{}')))
BR = BR20 * (1 + BR_TC * (CFG['magnet_temp_C'] - 20))
MU_MAG = CFG['mu_mag']
ROTOR = dict(pole_number=10, rotor_diameter=106, shaft_diameter=30, magnet_inset=2, magnet_depth=28,
             magnet_opening=10, magnet_X=18, magnet_Y=8.5, magnet_Z=6, magnet_glue_gap=0.1,
             Bridge_thickness=2, bridge_offset=3.5)
# M235-35A (0.35 mm, 2.35 W/kg @1.5T 50Hz) typical DC magnetisation curve, H [A/m] -> B [T]
# ponytail: catalogue-typical curve; drop in your supplier's sheet data here if you have it
# slope dB/dH strictly falling (initial mu_r ~12000): FEMM's nonlinear solver converges ~20x faster than with an S-shaped knee
BH = [(0, 0), (50, 0.75), (100, 1.00), (150, 1.12), (200, 1.20), (300, 1.29), (500, 1.38), (1000, 1.45),
      (2500, 1.53), (5000, 1.63), (10000, 1.75), (20000, 1.87), (50000, 2.02), (100000, 2.13), (200000, 2.30)]
STACKING = CFG['stacking']
NWORK = 4                    # parallel FEMM instances (one per physical core)
SOLVE_TIMEOUT = 900          # s; a solve that has not converged by then is dropped (FEMM instance restarted)

PHASE = {t: ph for ph, ts in (('R', (1, 6, 7, 12)), ('Y', (4, 5, 10, 11)), ('B', (2, 3, 8, 9))) for t in ts}
R_BORE, R_MID = 53.5, 53.25    # stator bore; mid-gap split circle (rotor OD 53.0)


def load_stator():
    s = open(os.path.join(ROOT, 'cad', 'fieldlab_v7.html'), encoding='utf8').read()
    get = lambda i: json.loads(re.search(r'id="%s"[^>]*>(.*?)</script>' % i, s, re.S).group(1))
    w, d = get('wpData'), get('data')
    return w['profile'], w['cond'], w['turns'], d['sign']


def rot(p, a_deg):
    a = math.radians(a_deg); c, s = math.cos(a), math.sin(a)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def line(a, b):
    femm.mi_drawline(a[0], a[1], b[0], b[1])


def polyline(pts):
    for a, b in zip(pts, pts[1:]):
        line(a, b)


def arc_ccw(r, a0, a1, maxseg, bdry='', grp=0):
    """CCW arc of radius r from angle a0 to a1 (deg), optionally tagged with a boundary."""
    p, q = rot((r, 0), a0), rot((r, 0), a1)
    femm.mi_drawarc(p[0], p[1], q[0], q[1], a1 - a0, maxseg)
    m = rot((r, 0), (a0 + a1) / 2)
    femm.mi_selectarcsegment(m[0], m[1])
    femm.mi_setarcsegmentprop(maxseg, bdry, 0, grp)
    femm.mi_clearselected()


def label(p, mat, mesh, circ='', magdir=0, turns=0, grp=0):
    femm.mi_addblocklabel(p[0], p[1])
    femm.mi_selectlabel(p[0], p[1])
    femm.mi_setblockprop(mat, 0 if mesh else 1, mesh, circ, magdir, grp, turns)
    femm.mi_clearselected()


def convex_hull(pts):
    pts = sorted(set(pts))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0: hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def rotor_geometry(p):
    """Python port of build() in rotor/spoke_rotor_explorer.html, pole on +y."""
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


def build(path):
    profile, cond, _, sign = load_stator()
    femm.newdocument(0)
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, STACK, 30, 0)
    femm.mi_smartmesh(0)          # use the label mesh sizes below, not FEMM's auto-refinement
    femm.mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    femm.mi_addmaterial('Copper', 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0, 0, 0)
    femm.mi_addmaterial('M235-35A', 5000, 5000, 0, 0, 0, 0.35, 0, STACKING, 0, 0, 0, 0, 0)
    for h, b in BH:
        femm.mi_addbhpoint('M235-35A', b, h)
    femm.mi_addmaterial('Ferrite', MU_MAG, MU_MAG, BR / (MU0 * MU_MAG), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    femm.mi_addboundprop('A0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for ph in 'RYB':
        femm.mi_addcircprop(ph, 0, 1)

    # ---- stator: merge the 12 segments into one outline (ideal butt joints) ----
    P = [tuple(p) for p in profile]
    idx = lambda x, y: min(range(len(P)), key=lambda i: (P[i][0] - x) ** 2 + (P[i][1] - y) ** 2)
    i1, i2, i3, i4 = idx(18.506, 69.064), idx(20.706, 77.274), idx(-20.706, 77.274), idx(-18.506, 69.064)
    inner = P[i4:] + P[:i1 + 1]          # -x joint -> tip -> +x joint
    outer = P[i2:i3 + 1]                 # +x joint -> yoke OD -> -x joint
    # inner runs -x -> +x (tooth t's +x end == tooth t-1's -x end); outer runs +x -> -x
    tooth = lambda t, pts: [rot(p, 30 * (t - 1) - 90) for p in pts]
    for chain, grp, order in ((inner, 2, [1] + list(range(12, 1, -1))), (outer, 9, range(1, 13))):
        loop = [q for t in order for q in tooth(t, chain)[:-1]]
        loop.append(loop[0])
        for a, b in zip(loop, loop[1:]):
            line(a, b)
            femm.mi_selectsegment((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            femm.mi_setsegmentprop('A0' if grp == 9 else '', 0, 1, 0, grp)
            femm.mi_clearselected()
    tip = math.degrees(math.atan2(10.927, 52.372))
    for t in range(1, 13):               # close each slot mouth at the bore radius
        arc_ccw(math.hypot(10.927, 52.372), 30 * (t - 2) + tip, 30 * (t - 1) - tip, 0.5)
    label(rot((0, 65), 0 - 90), 'M235-35A', 1.5)
    side_hull = convex_hull([(c[0] + WIRE_D / 2 * math.cos(a), c[1] + WIRE_D / 2 * math.sin(a))
                             for c in cond for a in np.radians(np.arange(0, 360, 45))])
    for t in range(1, 13):
        label(rot((0, 56), 30 * (t - 1) + 15 - 90), 'Air', 1.5)      # slot, on the slot centreline
        for side in (1, -1):   # ponytail: coil side = hull around its 10 wires (copper and air both mu_r 1)
            hull = [rot((side * x, y), 30 * (t - 1) - 90) for x, y in side_hull]
            polyline(hull + hull[:1])
            cx, cy = np.mean([rot((side * c[0], c[1]), 30 * (t - 1) - 90) for c in cond], axis=0)
            label((cx, cy), 'Copper', 1.0, PHASE[t], 0, sign[t - 1] * side * len(cond))

    # ---- air gap: rotor band | stator band, split at mid-gap ----
    arc_ccw(R_MID, 0, 180, 0.5); arc_ccw(R_MID, 180, 360, 0.5)
    label(rot((0, (R_MID + R_BORE) / 2), -90), 'Air', 0.25)

    # ---- rotor ----
    g = rotor_geometry(ROTOR)
    n = ROTOR['pole_number']; pitch = 360 / n
    mir = lambda p: (-p[0], p[1])
    dC = math.degrees(math.atan2(-g['C'][0], g['C'][1]))     # half mouth angle at the OD
    rG = math.hypot(*g['G']); dG = math.degrees(math.atan2(-g['G'][0], g['G'][1]))
    mags = []
    MAGLAB.clear()
    for k in range(n):
        a = pitch * k - 90                                   # magnet k on axis pitch*k
        L = lambda p: rot(p, a)
        cav = [g['G'], g['P'], g['Q'], g['R'], g['Rm'], g['Qm'], g['Pm']]
        polyline([L(p) for p in cav] + [L(mir(p)) for p in reversed(cav)])
        polyline([L(p) for p in (g['Rm'], g['D'], mir(g['D']), mir(g['Rm']))])
        line(L(g['D']), L(g['C'])); line(L(mir(g['D'])), L(mir(g['C'])))
        arc_ccw(rG, pitch * k - dG, pitch * k + dG, 1)                   # pocket floor
        arc_ccw(g['rr'], pitch * k + dC, pitch * (k + 1) - dC, 0.5)      # rotor OD between mouths
        mag = [g['Pm'], g['Qm'], g['Rm'], mir(g['Rm']), mir(g['Qm']), mir(g['Pm'])]
        mags.append([L(p) for p in mag])
        mdir = a + (0 if k % 2 == 0 else 180)                # tangential, alternating
        MAGLAB.append((L((0, (g['rim'] + g['rom']) / 2)), mdir))
        label(MAGLAB[-1][0], 'Ferrite', 1.0, magdir=mdir, grp=1)
        label(L((0, (rG + g['rim']) / 2)), 'Air', 0.8, grp=1)           # pocket + glue
    arc_ccw(g['rs'], 0, 180, 2); arc_ccw(g['rs'], 180, 360, 2)
    label((0, 0), 'Air', 3, grp=1)                           # ponytail: shaft modelled non-magnetic
    label(rot((40, 0), pitch / 2), 'M235-35A', 1.0, grp=1)
    label(rot(((g['rr'] + R_MID) / 2, 0), pitch / 2), 'Air', 0.25, grp=1)
    femm.mi_selectcircle(0, 0, R_MID + 0.01, 4)              # everything inside mid-gap -> rotor group
    femm.mi_setgroup(1)
    femm.mi_clearselected()
    ANGLE[0] = 0.0
    femm.mi_zoomnatural()
    femm.mi_saveas(path)
    return mags, g


def phase_currents(theta_e, phi, gamma, i_line_rms):
    """Coil (= parallel-path) currents, in phase with each phase EMF when gamma = 0."""
    ip = math.sqrt(2) * i_line_rms / math.sqrt(3)
    return {ph: ip / 2 * math.cos(math.radians(theta_e + phi[ph] + 90 + gamma)) for ph in 'RYB'}


MAGLAB, ANGLE = [], [0.0]


def set_angle(theta_m):
    d = theta_m - ANGLE[0]
    if abs(d) > 1e-9:
        femm.mi_selectgroup(1); femm.mi_moverotate(0, 0, d); femm.mi_clearselected()
        ANGLE[0] = theta_m
    for p, mdir in MAGLAB:                                    # magnetisation turns with the rotor
        q = rot(p, theta_m)
        femm.mi_selectlabel(q[0], q[1]); femm.mi_setblockprop('Ferrite', 0, 1.0, '', mdir + theta_m, 1, 0)
        femm.mi_clearselected()


def solve(theta_m, cur, keep=False):
    set_angle(theta_m)
    for ph in 'RYB':
        femm.mi_modifycircprop(ph, 1, cur[ph])
    femm.mi_analyze(1)
    femm.mi_loadsolution()
    femm.mo_groupselectblock(1); T = femm.mo_blockintegral(22); femm.mo_clearblock()
    psi = {ph: femm.mo_getcircuitproperties(ph)[2] / 2 for ph in 'RYB'}   # /2: two coils per path
    if not keep:
        femm.mo_close()
    return T, psi


ZERO = dict(R=0, Y=0, B=0)
P = 5  # pole pairs
W = {}


def worker_init():
    import psutil
    before = {p.pid for p in psutil.process_iter(['name']) if p.info['name'] == 'femm.exe'}
    femm.openfemm(1)
    new = [p.pid for p in psutil.process_iter(['name']) if p.info['name'] == 'femm.exe' and p.pid not in before]
    W['femm_pid'] = new[0] if new else None
    os.makedirs(os.path.join(HERE, 'work'), exist_ok=True)
    W['file'] = os.path.join(HERE, 'work', 'w%d' % os.getpid())
    W['mags'], _ = build(W['file'] + '.fem')
    W['mat'] = (BR, MU_MAG, STACKING); W['dead'] = False


def _kill_solver():
    """Watchdog: kill this worker's stuck solver and its FEMM instance (FEMM never recovers from a killed solver)."""
    import psutil
    for p in psutil.process_iter(['name', 'cmdline']):
        try:
            if p.info['name'] == 'fkn.exe' and W['file'] in ' '.join(p.info['cmdline'] or []):
                p.kill()
        except psutil.Error:
            pass
    try:
        psutil.Process(W['femm_pid']).kill()
    except Exception:
        pass


def fields(case):
    """Tooth/yoke/gap B and magnet operating points, read per element from the worker's solution file (seconds)."""
    from field_maps import read_ans, element_fields
    nodes, el, labs = read_ans(W['file'] + '.ans')
    Bmag, Hm, mat, Bx, By, Bm = element_fields(nodes, el, labs, full=True)
    c = nodes[el[:, :3], :2].mean(1); x, y = c[:, 0], c[:, 1]; r = np.hypot(x, y)
    near = lambda q: (np.hypot(x - q[0], y - q[1]) < 1.5) & (mat == 3)
    f = dict(tooth_B=[float(Bmag[near(rot((0, 63), 30 * t - 90))].mean()) for t in range(12)],
             yoke_B=[float(Bmag[near(rot((0, 75.7), 30 * t + 15 - 90))].mean()) for t in range(12)])
    band = (r > R_MID) & (r < R_BORE) & (mat == 1)
    ang = np.degrees(np.arctan2(y[band], x[band])) % 360
    k = np.minimum((ang / 0.5).astype(int), 719)
    brad = (Bx[band] * x[band] + By[band] * y[band]) / r[band]
    f['gap_Br'] = (np.bincount(k, brad, 720) / np.maximum(np.bincount(k, minlength=720), 1)).tolist()
    m = mat == 4
    i = int(np.nanargmin(np.where(m, Hm, np.inf)))
    f.update(H_along_M_min=float(Hm[i]), B_along_M_min=float(np.min(Bm[m])), worst_point=[float(x[i]), float(y[i])])
    return f


def job(j):
    """j = (theta_m, currents, Br, fields_case or None[, {'mu': mu_mag, 'stack': stacking}]) -> (T, psi, fields)"""
    if W.get('dead'):
        worker_init()                                       # previous solve timed out: fresh FEMM instance
    timer = threading.Timer(SOLVE_TIMEOUT, _kill_solver); timer.start()
    try:
        return _job(j)
    except Exception:
        W['dead'] = True
        return None
    finally:
        timer.cancel()


def _job(j):
    th, cur, br, case = j[:4]
    o = j[4] if len(j) > 4 else {}
    mat = (br, o.get('mu', MU_MAG), o.get('stack', STACKING))
    if mat != W['mat']:
        br, mu, st = mat
        femm.mi_modifymaterial('Ferrite', 1, mu); femm.mi_modifymaterial('Ferrite', 2, mu)
        femm.mi_modifymaterial('Ferrite', 3, br / (MU0 * mu))
        femm.mi_modifymaterial('M235-35A', 8, st)
        W['mat'] = mat
    T, psi = solve(th, cur)
    return T, psi, (fields(case) if case else None)


def main():
    from multiprocessing import Pool
    res = dict(inputs=dict(stack_mm=STACK, I_line_rms=I_LINE_RMS, Vdc=VDC, Br=BR, magnet_temp_C=CFG['magnet_temp_C'], mu_r=MU_MAG, HcJ=HCJ,
                           steel='M235-35A 0.35 mm', stacking=STACKING, rotor=ROTOR))
    log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
    ck = os.path.join(HERE, 'results_partial.json')          # checkpoint after each stage; rerun resumes
    if os.path.exists(ck):
        old = json.load(open(ck))
        if old.get('inputs') == json.loads(json.dumps(res['inputs'])):
            res = old; log('resuming from checkpoint:', [k for k in res if k != 'inputs'])
    save = lambda: json.dump(res, open(ck, 'w'), indent=1)
    with Pool(NWORK, initializer=worker_init) as pool:
        run = lambda jobs: pool.map(job, jobs, chunksize=1)
        # 1. no load (one electrical period) + cogging (one 6 deg cogging period)
        th = np.arange(0, 72, 3.0); thc = np.arange(0, 6, 0.25)
        if 'noload' not in res:
            stage1(run, res, th, thc, log); save()
        phi = res['noload']['phi']
        pos = [0, 4, 8]
        cur = lambda t, g, i=I_LINE_RMS: phase_currents(P * t, phi, g, i)
        if 'gamma_sweep' not in res:
            stage2(run, res, cur, pos, log); save()
        sgn, gopt = res['gamma_sweep']['sign'], res['gamma_sweep']['gamma_opt']
        stage3(lambda jobs: run_cached(pool, jobs, log), res, cur, pos, sgn, gopt, log)
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    os.remove(ck)


def run_cached(pool, jobs, log):
    """pool.imap with every result saved as it arrives; a rerun skips the saved ones."""
    key = hashlib.md5(json.dumps(jobs, default=str).encode()).hexdigest()[:10]
    ck = os.path.join(HERE, 'work', 'cache_%s.json' % key)
    out = json.load(open(ck)) if os.path.exists(ck) else []
    if out:
        log('resuming %d/%d cached solves' % (len(out), len(jobs)))
    for r in pool.imap(job, jobs[len(out):], chunksize=1):
        out.append(r); json.dump(out, open(ck, 'w'))
        if r is None:
            log('solve %d did not converge in %d s - dropped' % (len(out) - 1, SOLVE_TIMEOUT))
    return out


def stage1(run, res, th, thc, log):
    out = run([(t, ZERO, BR, None) for t in th] + [(t, ZERO, BR, None) for t in thc])
    nl, cg = out[:len(th)], out[len(th):]
    psi = {ph: np.array([x[1][ph] for x in nl]) for ph in 'RYB'}
    X = {ph: 2 * np.mean(psi[ph] * np.exp(-1j * np.radians(P * th))) for ph in 'RYB'}
    phi = {ph: math.degrees(np.angle(X[ph])) for ph in 'RYB'}
    res['noload'] = dict(theta_m=th.tolist(), psi={k: v.tolist() for k, v in psi.items()},
                         psi1_peak={ph: abs(X[ph]) for ph in 'RYB'}, phi=phi, torque=[x[0] for x in nl])
    res['cogging'] = dict(theta_m=thc.tolist(), torque=[x[0] for x in cg])
    log('no-load psi1', {k: round(abs(v), 5) for k, v in X.items()}, 'phi', {k: round(v, 1) for k, v in phi.items()})


def stage2(run, res, cur, pos, log):
    # 2. current-angle sweep, averaged over 3 positions across the 12 deg ripple period
    gams = np.arange(-10, 71, 10)
    out = run([(t, cur(t, g), BR, None) for g in gams for t in pos])
    Tg = [float(np.mean([x[0] for x in out[i * 3:i * 3 + 3]])) for i in range(len(gams))]
    sgn = 1 if max(Tg) > -min(Tg) else -1            # FEMM CCW sign vs motoring direction
    Tg = [sgn * x for x in Tg]
    k = int(np.clip(np.argmax(Tg), 1, len(gams) - 2)); c = np.polyfit(gams[k - 1:k + 2], Tg[k - 1:k + 2], 2)
    gopt = float(-c[1] / (2 * c[0])) if c[0] < 0 else float(gams[int(np.argmax(Tg))])
    res['gamma_sweep'] = dict(gamma=gams.tolist(), torque=Tg, gamma_opt=gopt, sign=sgn)
    log('gamma sweep', list(zip(gams.tolist(), np.round(Tg, 2))), 'gamma_opt %.1f' % gopt)


def stage3(run, res, cur, pos, sgn, gopt, log):
    # 3. rated load waveform, torque vs current, hot magnets, field cases
    thl = np.arange(0, 12, 0.5); fr = [0.25, 0.5, 0.75, 1.25]; br_hot = BR20 * (1 + BR_TC * 60)   # one 12 deg ripple period
    jobs = [(t, cur(t, gopt), BR, None) for t in thl]
    jobs += [(t, cur(t, gopt, f * I_LINE_RMS), BR, None) for f in fr for t in pos]
    jobs += [(t, cur(t, gopt), br_hot, None) for t in pos]
    jobs += [(0, cur(0, gopt), BR, 'rated'), (0, cur(0, 90), BR, 'demag_gamma90'), (0, ZERO, BR, 'noload')]
    out = run(jobs)
    T_ = lambda x: sgn * x[0] if x else float('nan')        # None = solve dropped
    ld, out = out[:len(thl)], out[len(thl):]
    ok = [k for k, x in enumerate(ld) if x]
    res['load'] = dict(theta_m=[float(thl[k]) for k in ok], torque=[T_(ld[k]) for k in ok])
    Tl = float(np.mean(res['load']['torque']))
    avg = lambda xs: float(np.nanmean([T_(x) for x in xs])) if any(xs) else float('nan')
    cs = [avg(out[i * 3:i * 3 + 3]) for i in range(len(fr))]
    res['current_sweep'] = dict(I_line_rms=[f * I_LINE_RMS for f in (0.25, 0.5, 0.75, 1.0, 1.25)],
                                torque=cs[:3] + [Tl, cs[3]])
    out = out[3 * len(fr):]
    res['hot80'] = dict(Br=br_hot, torque=avg(out[:3]))
    res['fields'] = {c: (x[2] if x else None) for c, x in zip(('rated', 'demag_gamma90', 'noload'), out[3:])}
    log('rated torque %.2f N.m' % Tl)


if __name__ == '__main__':
    main()
