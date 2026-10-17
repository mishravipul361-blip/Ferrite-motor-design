"""Aluminium winding for Design9795_1: every stator variant x winding against the copper motor.
Metrics: torque-speed envelope, efficiency map (Motor-CAD Lab grid), drive-cycle energy and range (telematics speed trace
and WMTC Class 1 through the vehicle model of the Spoke V2 study), peak-torque winding loss (thermal), material cost.

Vehicle (user): GVW 757 kg, CdA 1.117 m2, wheel radius 0.30 m, ratio 13.9. Assumed: Crr 0.015, gear efficiency 0.95,
rotating-mass factor 1.05, flat road; regen down to -26.4 N.m (largest regen torque in the telematics file), rest friction.
Range change = (Wh/km copper) / (Wh/km design) - 1, motor losses only (inverter and battery taken as equal for all designs).
Prices (INR/kg): copper winding wire 1100, ferrite magnets 800, M235-35A laminations 300 (finished laminations, stator +
rotor; user, 2026-10-17); aluminium winding wire 370 = copper / 3.0 (Indian enamelled-wire listings 2.6-3x; LME metal
alone 4.8x on 12 Oct 2026); wire waste 5 %.

python alu9795_search.py   -> alu9795_search.json
"""
import json, math, os, time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import lab_map as L
import alu9795_study as S

H = S.H
SPOKE = os.path.join(os.path.dirname(H), 'Spoke_V2_backemf')
VEH = dict(m=757.0, cda=1.117, r=0.30, i=13.9, crr=0.015, eta_g=0.95, kr=1.05, rho=1.2)
TREG = -26.4
PRICE = dict(Cu=1100.0, Al=370.0, steel=300.0, magnet=800.0)
WIRE_WASTE, BLANK_ADD = 1.05, 4.0
SP_MAP = np.arange(0, 9501, 100.0); TQ_MAP = np.arange(0.5, 60.01, 0.5)


# ---------------- drive cycles ----------------
def motor_points(v, valid):
    vm, a = (v[1:] + v[:-1]) / 2, np.diff(v)
    vm, a = vm[valid], a[valid]
    F = VEH['kr'] * VEH['m'] * a + VEH['m'] * 9.81 * VEH['crr'] * (vm > 0.05) + 0.5 * VEH['rho'] * VEH['cda'] * vm ** 2
    n = vm / VEH['r'] * 60 / (2 * math.pi) * VEH['i']
    T = np.where(F > 0, F * VEH['r'] / (VEH['i'] * VEH['eta_g']), np.maximum(F * VEH['r'] * VEH['eta_g'] / VEH['i'], TREG))
    return T, n, float(np.sum(vm)) / 1000


def wmtc_trace(parts):
    CY = json.load(open(os.path.join(SPOKE, 'wmtc_cycles.json')))['parts']
    v = []
    for p in parts:
        x = np.array([np.nan if s is None else s for s in CY[p]], float)
        k = np.arange(len(x)); bad = np.isnan(x)
        x[bad] = np.interp(k[bad], k[~bad], x[~bad])
        v += [0.0] + list(x)
    return np.array(v) / 3.6


X = np.loadtxt(os.path.join(SPOKE, 'telematics_cycle.txt'))
v_tel = X[:, 2].clip(0) * 2 * math.pi * VEH['r'] / 60
CYC = {'Telematics': motor_points(v_tel, np.diff(X[:, 0]) == 1)}
v1 = wmtc_trace(['P1R', 'P1R'])
CYC['WMTC Class 1'] = motor_points(v1, np.ones(len(v1) - 1, bool))
N_MAX = max(float(c[1].max()) for c in CYC.values()) + 200
SP_CY = np.arange(0, N_MAX + 1, 100.0); TQ_CY = np.arange(0.5, 50.01, 0.5)


def cycle(d):
    env, mp = d.run(SP_CY, TQ_CY, npts=241, keys=('loss', 'pcu', 'pac', 'pfe', 'pme'))
    tmax = np.array([e['Tsh'] if e else 0.0 for e in env])
    F = {}
    for k in ('loss', 'pcu', 'pac', 'pfe', 'pme'):
        Z = np.array(mp[k], float)
        for j in range(Z.shape[1]):
            c = Z[:, j]; ok = ~np.isnan(c)
            if ok.any():
                c[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), c[ok])
        Z[:, 0] = Z[:, 1]
        F[k] = RegularGridInterpolator((TQ_CY, SP_CY), Z, bounds_error=False, fill_value=None)
    out = {}
    for name, (T, n, km) in CYC.items():
        on = n >= 50
        w = n * math.pi / 30
        tl = np.interp(n, SP_CY, tmax)
        Tc = np.clip(T, -tl, tl)
        q = np.c_[np.clip(np.abs(Tc), TQ_CY[0], TQ_CY[-1]), np.clip(n, SP_CY[1], SP_CY[-1])]
        Ls = {k: np.where(on, f(q), 0.0) for k, f in F.items()}
        Pm = np.where(on, Tc * w, 0.0)
        Pe = Pm + Ls['loss']
        mot, gen = Pm > 0, Pm < 0
        Wh = lambda a: float(np.sum(a)) / 3600
        r = dict(km=km, E_net=Wh(Pe), E_shaft_mot=Wh(Pm[mot]), E_in_mot=Wh(Pe[mot]), E_shaft_gen=Wh(-Pm[gen]), E_out_gen=Wh(-Pe[gen]),
                 loss={k: Wh(v) for k, v in Ls.items()}, n_short=int(np.sum(np.abs(T) > tl + 1e-9)))
        r['eff_mot'] = 100 * r['E_shaft_mot'] / r['E_in_mot']; r['eff_gen'] = 100 * r['E_out_gen'] / max(r['E_shaft_gen'], 1e-9)
        r['Wh_km'] = r['E_net'] / km
        out[name] = r
    return out


# ---------------- cost ----------------
def cost(d, P=PRICE):
    m = d.masses()
    od = d.geo['stator']['od']
    blank = ((od + BLANK_ADD) * 1e-3) ** 2 * d.L * 1e-3 * 7650 * 0.97              # gross steel per motor (rotor punched from the same blank)
    c = dict(wire=m['winding'] * WIRE_WASTE * P[d.metal], steel=(m['stator_lam'] + m['rotor_lam']) * P['steel'], magnet=m['magnets'] * P['magnet'])
    c['total'] = c['wire'] + c['steel'] + c['magnet']
    return c, blank


# ---------------- one design ----------------
REF = None


def metrics(d):
    env, mp = d.run(SP_MAP, TQ_MAP, npts=241)
    eff = mp['eff']
    out = dict(label=d.label, metal=d.metal, sv=d.sv, L=d.L, Nc=d.Nc, ns=d.ns, d=d.d, fill=d.fill, R_path=d.R_path, mlt=d.mlt,
               geo=dict(od=d.geo['stator']['od'], slot_depth=d.geo['stator']['slot_depth'], tooth_w=d.geo['stator']['tooth_w'],
                        yoke=d.geo['yoke_mm'], slot_area=d.geo['slot_area']),
               masses=d.masses(), env_T=[e['Tsh'] if e else None for e in env], peak_eff=float(np.nanmax(eff)))
    R0 = d.evaluate(1000, 241); k = np.argmax(np.where(d.feasible(R0), R0['Tsh'], -1))
    out['peak'] = dict(T=float(R0['Tsh'][k]), pcu=float(R0['pcu'][k]), pac=float(R0['pac'][k]), J=float(R0['Ipk'][k] / math.sqrt(2) / (d.A * 1e6)))
    out['cycle'] = cycle(d)
    c, blank = cost(d); out['cost'] = c; out['steel_gross'] = blank
    if REF is not None:
        both = ~np.isnan(eff) & ~np.isnan(REF['eff'])
        de = (eff - REF['eff'])[both]
        out['map_dEta_mean'] = float(de.mean()); out['map_dEta_min'] = float(de.min()); out['map_dEta_p10'] = float(np.percentile(de, 10))
        out['map_missing'] = int((~np.isnan(REF['eff']) & np.isnan(eff)).sum())
        tr = [a / b for a, b in zip(out['env_T'], REF['env_T']) if a and b]
        out['env_ratio_min'] = float(min(tr))
        out['range_pct'] = {c_: 100 * (REF['cycle'][c_]['Wh_km'] / out['cycle'][c_]['Wh_km'] - 1) for c_ in CYC}
        out['dcost'] = c['total'] - REF['cost']['total']
        out['dmass'] = out['masses']['active'] - REF['masses']['active']
    return out, eff


LS = tuple(float(x) for x in os.environ['LS'].split(',')) if os.environ.get('LS') else (64, 66, 68, 70, 72, 74, 76, 78, 80)


NCS = tuple(int(x) for x in os.environ['NCS'].split(',')) if os.environ.get('NCS') else (8, 9, 10)
NS_MAX = int(os.environ.get('NS_MAX', 6))


def wires(sv, Nc):
    ws = S.wire_options(sv, Nc, 'Al', ns_max=NS_MAX)
    if not S.BUNDLE or not ws:
        return ws
    # bundle effect: the strand count changes neither DC nor AC loss at equal area -> the largest area, and the fewest strands within 3 % of it
    big = max(ws, key=lambda w: w['A'])
    few = min((w for w in ws if w['A'] >= 0.97 * big['A']), key=lambda w: w['ns'])
    return [big] if few is big else [few, big]


def designs(svs=None, Ls=LS, Ncs=NCS):
    out = []
    svs = svs or (list(S.FEM['designs']) if S.FEM else ['D0'])
    for sv in svs:
        for Nc in Ncs:
            for w in wires(sv, Nc):
                for Lmm in Ls:
                    out.append(dict(metal='Al', sv=sv, Nc=Nc, ns=w['ns'], d=w['d'], Lmm=float(Lmm)))
    out.append(dict(metal='Al', sv='D0', Nc=10, ns=1, d=S.D_CU0, Lmm=64.0, label='Al drop-in (SWG 11 size, 64 mm)'))
    return out


def _init():
    global REF
    ref, eff = metrics(S.copper())
    REF = dict(ref, eff=eff)


def _job(kw):
    return metrics(S.Design(**kw))[0]


if __name__ == '__main__':
    import sys
    from multiprocessing import Pool
    t0 = time.time()
    for c_, (T, n, km) in CYC.items():
        print('%s: %.2f km, n max %.0f rpm, T %.1f..%.1f N.m' % (c_, km, n.max(), T.min(), T.max()))
    _init()
    ref = {k: v for k, v in REF.items() if k != 'eff'}
    print('copper: peak eff %.2f, T1000 %.2f, pcu %.0f W, cost %s, cycle %s' % (ref['peak_eff'], ref['peak']['T'], ref['peak']['pcu'],
          {k: round(v, 2) for k, v in ref['cost'].items()}, {c_: round(r['Wh_km'], 3) for c_, r in ref['cycle'].items()}), flush=True)
    svs = [a for a in sys.argv[1:] if not a.startswith('-')] or None
    todo = designs(svs)
    print(len(todo), 'designs', flush=True)
    res = []
    with Pool(int(os.environ.get('NWORK', 6)), initializer=_init) as pool:
        for k, r in enumerate(pool.imap(_job, todo, chunksize=4), 1):
            res.append(r)
            if k % 50 == 0 or k == len(todo):
                print(time.strftime('%H:%M:%S'), k, '/', len(todo), r['label'], 'env %.3f range %+.2f %%' % (r['env_ratio_min'], r['range_pct']['Telematics']), flush=True)
    fn = os.environ.get('OUT', 'alu9795_search.json')
    json.dump(dict(copper=ref, designs=res, price=PRICE, vehicle=VEH, cycles={c_: dict(km=v[2]) for c_, v in CYC.items()}),
              open(os.path.join(H, fn), 'w'), default=float)
    print('%.0f s' % (time.time() - t0))
