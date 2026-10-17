"""Aluminium winding in place of copper - design study on top of the tuned Lab model (lab_model.py, v2 losses).

A design = conductor material, stator slot variant (S0..S4 from alu_femm.py), stack length, series turns per phase
(Nph = 2 x turns/coil, delta, 2 parallel paths kept: adjacent 12s/10p coils are 30 deg el apart, so a = 2 is the most
parallel paths allowed), strands in hand and strand diameter.

Scaling (2D FEA, exact apart from end effects), with kL = L/63 and s = Nph/20:
  the field depends only on coil ampere-turns, so the design is evaluated at the baseline-equivalent path current
  i0 = s * i (i = actual path current)    ->   T = kL * T0(i0),   psi = s * kL * psi0(i0),   P_iron = kL * P_iron0(i0)
  slot variants: FEMM differences / ratios against S0 at 63 mm (alu_femm.json) are added on top.
Winding: R_path = rho(T) * 2 coils * Nc * MLT / A_cond,  MLT = 2 L + end part (75.615 mm at tooth 15.5, -2 mm per mm of
tooth width); AC loss = Motor-CAD hybrid formula of lab_model with n_s strands of d_s, active length L and rho of the metal.

Run: python alu_study.py   -> alu_results.json
"""
import json, math, os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import lab_map as L
import lab_model as LM

H = L.H
MAT = {'Cu': dict(rho20=1.724e-8, alpha=0.003862, dens=8933.0, name='Copper (100 % IACS)'),
       'Al': dict(rho20=2.826e-8, alpha=0.00403, dens=2703.0, name='Aluminium EC-1350 (61 % IACS)')}
FILL_MAX = 0.5175                         # Motor-CAD wire slot fill of the copper baseline (insulated wire area / slot area)
D_INS_ADD = 3.0302 - 2.9464               # grade-2 enamel build, as SWG 11 in Motor-CAD
IEC_SIZES = [1.0, 1.06, 1.12, 1.18, 1.25, 1.32, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.12, 2.24, 2.36, 2.5, 2.65, 2.8, 3.0,
             3.15, 3.35, 3.55, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
SWG_AWG = [3.251, 2.9464, 2.642, 2.337, 2.032, 1.829, 1.626, 1.422, 2.906, 2.588, 2.305, 2.053, 1.628, 1.450, 3.264, 3.665]
WIRE_SIZES = sorted(set(IEC_SIZES + SWG_AWG))
MLT_END0 = 75.615
MAG_AREA = 473.19088491688336             # mm2 per pole (field_stats.json)
DENS_FERRITE = 4800.0
T_W = L.T_WDG
BASE = json.load(open(os.path.join(H, 'lab_100C.json')))
# lamination options: classical eddy ~ t^2 (same Steinmetz fit and 1.6 build factor), thinner grades a little higher hysteresis
LAMS = {'M235-35A': dict(kh=1.0, ke=1.0, t=0.35, name='M235-35A, 0.35 mm (baseline)'),
        'NO27': dict(kh=1.0, ke=(0.27 / 0.35) ** 2, t=0.27, name='0.27 mm non-oriented (e.g. NO27 grade)'),
        'NO20': dict(kh=1.1, ke=(0.20 / 0.35) ** 2, t=0.20, name='0.20 mm non-oriented (e.g. NO20 grade)')}
OPS = {'OP1': dict(n=9000, mode='max', label='9000 rpm, max power'), 'OP2': dict(n=9339, T=3.24, label='9339 rpm, 3.24 N.m'),
       'OP3': dict(n=2000, T=44.34, label='2000 rpm, 44.3 N.m')}


class Slots:
    """FEMM corrections of the slot variants against S0 (same 63 mm, same ampere-turns)."""
    def __init__(self):
        fn = os.path.join(H, 'alu_femm.json')
        self.d = json.load(open(fn)) if os.path.exists(fn) else None
        self.f = {}

    def geo(self, sv):
        return dict(slot_depth=17.5, tooth_w=15.5) if self.d is None else self.d['designs'][sv]['geo']

    def corr(self, sv, Pt):
        if sv == 'S0' or self.d is None:
            one, zero = np.ones(len(Pt)), np.zeros(len(Pt))
            return dict(dT=zero, dpsid=zero, dpsiq=zero, rSa=one, rS2=one, rcu=one)
        if sv not in self.f:
            g = (np.array(self.d['ids']), np.array(self.d['iqs']))
            a, b = self.d['designs'][sv], self.d['designs']['S0']
            I = lambda z: RegularGridInterpolator(g, np.array(z), method='linear', bounds_error=False, fill_value=None)
            self.f[sv] = dict(dT=I(np.array(a['T']) - b['T']), dpsid=I(np.array(a['psid']) - b['psid']), dpsiq=I(np.array(a['psiq']) - b['psiq']),
                              rSa=I(np.array(a['st_Sa']) / b['st_Sa']), rS2=I(np.array(a['st_S2']) / b['st_S2']),
                              rcu=I(np.array(a['cu_B2']) / b['cu_B2']))
        return {k: f(Pt) for k, f in self.f[sv].items()}

    def st_mass(self, sv):
        return 3.6094498551359173 if self.d is None else float(np.mean(self.d['designs'][sv]['st_mass']))

    def tooth_yoke(self, sv):
        if self.d is None:
            return None
        x = self.d['designs'][sv]
        return dict(tooth_Bmean=x['tooth_Bmean'][-1][-1], yoke_Bmean=x['yoke_Bmean'][-1][-1], yoke_B99=x['yoke_B99'][-1][-1],
                    tooth_B99=x['tooth_B99'][-1][-1])


SLOTS = Slots()
_CACHE = {}
BASEM = LM.Model('v2')


def slot_area(sv):
    import sens_models as SM, backemf as M
    return float(SM.winding_check(dict(M.STATOR, **SLOTS.geo(sv)))['slot_area'])


def rotor_lam_mass(Lmm):
    return ROTOR_LAM_63 * Lmm / 63.0


def _rotor_lam_63():
    """Rotor lamination mass at 63 mm from the solved no-load model (M235 elements inside the air gap)."""
    fn = os.path.join(H, 'FEMM_models', 'Spoke_V2_12s10p_NoLoad_Magnets100C.ans')
    s = open(fn).read().split('\n')
    l0 = next(i for i, l in enumerate(s) if l.startswith('[NumBlockLabels]'))
    labs = [l.split() for l in s[l0 + 1:l0 + 1 + int(s[l0].split('=')[1])]]
    k = s.index('[Solution]'); n = int(s[k + 1])
    nodes = np.array([[float(x) for x in l.split()[:2]] for l in s[k + 2:k + 2 + n]])
    ne = int(s[k + 2 + n])
    el = np.array([[int(x) for x in l.split()[:4]] for l in s[k + 3 + n:k + 3 + n + ne]])
    p = nodes[el[:, :3]]
    area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    r = np.hypot(p[:, :, 0].mean(1), p[:, :, 1].mean(1))
    mats = {}
    i0 = next(i for i, l in enumerate(s) if l.startswith('[BlockProps]'))
    names = [l.split('=')[1].strip().strip('"') for l in s[i0:l0] if l.strip().startswith('<BlockName>')]
    mat = np.array([int(labs[l][2]) for l in el[:, 3]])
    lam = mat == names.index('M235-35A') + 1
    sh = mat == names.index('SCM420') + 1
    return float(np.sum(area[lam & (r < 53.25)]) * 1e-6 * 0.063 * 0.97 * 7650), float(np.sum(area[sh]) * 1e-6 * 0.063 * 7850)


ROTOR_LAM_63, SHAFT_63 = _rotor_lam_63()


def wire_options(sv, Nc, metal):
    """Strand options that fit the slot at the baseline wire fill: n strands in hand, largest IEC size that fits."""
    As = slot_area(sv)
    a_ins = FILL_MAX * As / (2 * Nc)
    out = []
    for ns in (1, 2, 3, 4):
        dmax = math.sqrt(4 * a_ins / (math.pi * ns)) - D_INS_ADD
        fits = [d for d in WIRE_SIZES if d <= dmax + 1e-9]
        if fits:
            d = fits[-1]
            out.append(dict(ns=ns, d=d, A=ns * math.pi / 4 * d * d, fill=2 * Nc * ns * math.pi / 4 * (d + D_INS_ADD) ** 2 / As))
    return out


class Design:
    def __init__(self, metal='Cu', sv='S0', Lmm=63.0, Nph=20, ns=1, d=2.9464, label=None, lam='M235-35A'):
        self.metal, self.sv, self.L, self.Nph, self.ns, self.d = metal, sv, Lmm, Nph, ns, d
        self.Nc = Nph // 2
        self.lam, self.kh, self.ke = lam, LAMS[lam]['kh'], LAMS[lam]['ke']
        self.kL, self.s = Lmm / 63.0, Nph / 20.0
        m = MAT[metal]
        self.rho = m['rho20'] * (1 + m['alpha'] * (T_W - 20))
        self.A = ns * math.pi / 4 * (d * 1e-3) ** 2
        geo = SLOTS.geo(sv)
        self.As = slot_area(sv)
        # end turn: tooth width change, plus pi x change of mean coil thickness (half the slot area over the coil height)
        tc = self.As / 2 / (geo['slot_depth'] - 1.3); tc0 = slot_area('S0') / 2 / (17.5 - 1.3)
        self.mlt = (2 * Lmm + MLT_END0 + 2 * (geo['tooth_w'] - 15.5) + math.pi * (tc - tc0)) * 1e-3
        self.R_path = self.rho * 2 * self.Nc * self.mlt / self.A
        self.As = slot_area(sv)
        self.fill = 2 * self.Nc * ns * math.pi / 4 * (d + D_INS_ADD) ** 2 / self.As
        self.label = label or '%s %s L%.1f N%d %dx%.2f' % (metal, sv, Lmm, Nph, ns, d) + ('' if lam == 'M235-35A' else ' ' + lam)

    def masses(self):
        wdg = 12 * self.Nc * self.mlt * self.A * MAT[self.metal]['dens']
        st = SLOTS.st_mass(self.sv) * self.kL
        rt = rotor_lam_mass(self.L)
        mg = MAG_AREA * 10 * self.L * 1e-9 * DENS_FERRITE
        return dict(winding=wdg, stator_lam=st, rotor_lam=rt, magnets=mg, total=wdg + st + rt + mg)

    def evaluate(self, n, i0d, i0q, key=None):
        """n rpm; i0d/i0q = baseline-equivalent path peak currents (ampere-turns / 10)."""
        Pt = np.column_stack([np.ravel(i0d), np.ravel(i0q)])
        ck = (self.sv, key)
        if key is None or ck not in _CACHE:            # field quantities depend only on ampere-turns and slot variant
            st = ({k: f(Pt) for k, f in BASEM.F.items()}, SLOTS.corr(self.sv, Pt), {k: np.maximum(fn(Pt), 0) for k, fn in BASEM.I.items()})
            if key is not None:
                if len(_CACHE) > 40:
                    _CACHE.clear()
                _CACHE[ck] = st
        V, C, Wi = _CACHE[ck] if key is not None else st
        kL, s = self.kL, self.s
        Tem = kL * (V['T'] + C['dT'])
        psd = s * kL * (V['psid'] + C['dpsid']) + L.L_EW_PATH * s * Pt[:, 0]
        psq = s * kL * (V['psiq'] + C['dpsiq']) + L.L_EW_PATH * s * Pt[:, 1]
        i_d, i_q = Pt[:, 0] / s, Pt[:, 1] / s                       # actual path currents
        Ipk = np.hypot(i_d, i_q)
        wm = n * 2 * math.pi / 60; we = L.P * wm; f = we / (2 * math.pi)
        vd = self.R_path * i_d - we * psq; vq = self.R_path * i_q + we * psd
        v = np.hypot(vd, vq)
        pcu = 3 * self.R_path * Ipk ** 2
        pac = L.AC_ADJ * (12 * 2 * self.Nc) * math.pi * (self.L * 1e-3) * self.ns * (self.d * 1e-3) ** 4 * (2 * math.pi * f) ** 2 * \
            (2 * np.maximum(V['cu_B2'] * C['rcu'], 0)) / (128 * self.rho)
        st_h = L.BF * Wi['stator_H1'] * f * kL * C['rSa'] * self.kh; st_e = L.BF * Wi['stator_E1'] * f * f * kL * C['rS2'] * self.ke
        rt_h = L.BF * Wi['rotor_H1'] * f * kL * self.kh; rt_e = L.BF * Wi['rotor_E1'] * f * f * kL * self.ke
        pfe_st, pfe_rot = st_h + st_e, rt_h + rt_e
        b, w = L.LOSS['bearing'], L.LOSS['windage']
        pme = (b['front_W'] + b['rear_W']) * (n / b['ref_rpm']) ** b['exponent'] + kL * w['W'] * (n / w['ref_rpm']) ** w['exponent']
        pfe = pfe_st + pfe_rot
        Tsh = Tem - (pfe + pme) / wm if n > 0 else Tem.copy()
        psh = Tsh * wm
        loss = pcu + pac + pfe + pme
        pin = psh + loss
        with np.errstate(invalid='ignore', divide='ignore'):
            eff = np.where(psh > 0, 100 * psh / pin, np.nan)
        return dict(Tsh=Tsh, Tem=Tem, psh=psh, eff=eff, v=v, Ipk=Ipk, I_line=Ipk * 2 * math.sqrt(3) / math.sqrt(2), pcu=pcu, pac=pac,
                    pfe_st=pfe_st, pfe_rot=pfe_rot, pfe=pfe, pfe_h=st_h + rt_h, pfe_e=st_e + rt_e, pme=np.full_like(Tem, pme), loss=loss, pin=pin, Idc=pin / L.VDC,
                    i_d=i_d, i_q=i_q)

    def grid(self, npts=161):
        r = self.s * L.I_PATH_PK
        a, b = np.meshgrid(np.linspace(-r, 0, npts), np.linspace(0, r, npts), indexing='ij')
        ok = np.hypot(a, b) <= r + 1e-9
        return a[ok], b[ok]

    def feasible(self, R):
        return (R['v'] <= L.V_MAX) & (R['pin'] <= L.P_DC_MAX + 1e-6) & (R['Tsh'] > 0)

    def run(self, speeds, torques, npts=161, keys=('eff', 'loss', 'pcu', 'pac', 'pfe', 'pme', 'I_line')):
        G = self.grid(npts)
        mp = {k: np.full((len(torques), len(speeds)), np.nan) for k in keys}
        env = []
        for a, n in enumerate(speeds):
            R = self.evaluate(max(n, 1.0), *G, key=('grid', round(self.s, 6), npts))
            feas = self.feasible(R)
            if not feas.any():
                env.append(None); continue
            k = np.flatnonzero(feas)[np.argmax(R['Tsh'][feas])]
            env.append({kk: float(R[kk][k]) for kk in ('Tsh', 'psh', 'eff', 'loss', 'pcu', 'pac', 'pfe', 'pme', 'I_line', 'v', 'Idc')})
            for b, T in enumerate(torques):
                c = np.flatnonzero(feas & (np.abs(R['Tsh'] - T) <= 0.25))
                if len(c):
                    j = c[np.argmin(R['Ipk'][c])]
                    for kk in keys:
                        mp[kk][b, a] = R[kk][j]
        return env, mp

    def op(self, name, nI=400, ng=721):
        """Operating point on a polar (current, advance angle) grid: max power, or the least current giving the torque."""
        o = OPS[name]
        r = self.s * L.I_PATH_PK
        I, g = np.meshgrid(np.linspace(r / nI, r, nI), np.radians(np.linspace(0, 90, ng)), indexing='ij')
        R = self.evaluate(o['n'], -I.ravel() * np.sin(g.ravel()), I.ravel() * np.cos(g.ravel()), key=('polar', round(self.s, 6), nI, ng))
        feas = self.feasible(R)
        if o.get('mode') == 'max':
            k = np.flatnonzero(feas)[np.argmax(R['Tsh'][feas])]
        else:
            Tm = np.where(feas, R['Tsh'], -np.inf).reshape(nI, ng)
            rows = np.flatnonzero(Tm.max(1) >= o['T'])
            if not len(rows):
                return None
            a = rows[0]
            cols = np.flatnonzero(Tm[a] >= o['T'])
            k = a * ng + cols[np.argmin(Tm[a, cols])]
        return {kk: float(R[kk][k]) for kk in ('Tsh', 'psh', 'eff', 'loss', 'pcu', 'pac', 'pfe', 'pfe_h', 'pfe_e', 'pme', 'I_line', 'v', 'Idc', 'i_d', 'i_q')}


def baseline():
    return Design('Cu', 'S0', 63.0, 20, 1, 2.9464, label='Copper baseline (SWG 11, 63 mm)')


if __name__ == '__main__':
    import time
    t0 = time.time()
    d = baseline()
    print('baseline masses', {k: round(v, 3) for k, v in d.masses().items()}, 'R_path', d.R_path, 'fill', d.fill, 'shaft63', SHAFT_63)
    for o in OPS:
        print(o, {k: round(v, 2) for k, v in d.op(o).items()})
    sp = np.arange(0, 12001, 100); tq = np.arange(0.5, 60.01, 0.5)
    env, mp = d.run(sp, tq, npts=241)
    e = np.array(BASE['maps']['eff'], float)
    print('map max |diff| vs lab_100C.json', np.nanmax(np.abs(mp['eff'] - e)), 'peak', np.nanmax(mp['eff']), np.nanmax(e))
    print('env diff', max(abs(a['Tsh'] - b['T']) for a, b in zip(env, BASE['envelope']) if a))
    print('%.0f s' % (time.time() - t0))
