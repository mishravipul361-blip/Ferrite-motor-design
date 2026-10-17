"""Aluminium winding for Design9795_1: design model on top of the refined Motor-CAD-equivalent Lab model
(lab_model.Model('v3', slices=(-2, 0, 2), torque='flux')), stator-only changes, same steel (M235-35A).

A design = conductor metal, stator variant (alu9795_femm.py: slot depth, tooth width, OD 160-167 mm; rotor cross-section,
bore and steel untouched), stack length L (stator and rotor together; 64 mm = Design9795_1), turns per coil Nc (delta,
2 parallel paths), strands in hand ns and strand diameter d.

Scaling (2D, exact apart from end effects), s = Nc / 10, kL = L / 64:
  the field depends only on the coil ampere-turns -> evaluate the copper map at i0 = s * i (i = actual path current):
  psi = s kL psi0(i0) (+ stator-variant FEMM difference), torque = 1.5 p (psi_d i_q - psi_q i_d) as Motor-CAD Lab,
  stator iron = kL x v3 copper-stator iron x FEMM Steinmetz-sum ratio variant / D0, rotor iron x kL, windage x kL,
  AC loss field x FEMM ratio; end-winding inductance x s^2.
Winding: R_path = rho(100 C) * 2 coils * Nc * MLT / A,  MLT = 2 L + 75.634 mm (Motor-CAD) + 2 x tooth-width change
+ pi x change of mean coil thickness; wire fill (insulated) <= copper's 0.5129; AC loss = Motor-CAD hybrid formula.
Drive limits as Motor-CAD Lab: 52 V, MI 0.95, 360 A rms line, Idc 280 A, MTPA.

python alu9795_study.py     -> checks the copper reference against lab_100C.json
"""
import json, math, os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import lab_map as L
import lab_model as LM

H = L.H
P = L.P
MAT = {'Cu': dict(rho20=1.724e-8, alpha=0.003862, dens=8933.0, name='Copper (100 % IACS)'),
       'Al': dict(rho20=2.826e-8, alpha=0.00403, dens=2703.0, name='Aluminium EC-1350 (61 % IACS)')}
D_CU0 = 2.9464
INS = lambda d: 0.042 + 0.014 * d                    # grade-2 enamel build (SWG 11: 3.0302 - 2.9464 = 0.084 mm)
IEC = [1.0, 1.06, 1.12, 1.18, 1.25, 1.32, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.12, 2.24, 2.36, 2.5, 2.65, 2.8, 3.0,
       3.15, 3.35, 3.55, 3.75, 4.0, 4.25, 4.5]
SWG = [3.251, 2.9464, 2.642, 2.337, 2.032, 1.829, 1.626, 1.422]
WIRE_SIZES = sorted(set(IEC + SWG))
MLT_END0 = 75.634                                   # Motor-CAD EWdg MLT of Design9795_1 at its armature EWdg MLT multiplier 0.7
EW_MULT = float(os.environ.get('EW_MULT_AL', 0.7))   # armature end-winding MLT multiplier for ALUMINIUM windings (copper reference stays at 0.7)
KEW = EW_MULT / 0.7
STACK = L.M.STACK
ROTOR_LAM_AREA, MAGNET_AREA = 3056.8, 4492.7     # mm2, full machine (FEMM, Design9795_1 rotor)
DENS_FERRITE = 4800.0
FEM = json.load(open(os.path.join(H, 'alu9795_femm.json'))) if os.path.exists(os.path.join(H, 'alu9795_femm.json')) else None
D0GEO = None


def geo(sv):
    import alu9795_femm as F
    return FEM['designs'][sv]['geo'] if FEM else F.geometry(sv)


FILL_MAX = 20 * math.pi / 4 * (D_CU0 + INS(D_CU0)) ** 2 / geo('D0')['slot_area']      # 0.5129 (Motor-CAD wire fill)
BASE = LM.Model('v3', slices=(-2, 0, 2), torque='flux')


class Corr:
    """FEMM differences / ratios of a stator variant against the copper stator D0 (same ampere-turns)."""
    def __init__(self):
        self.f = {}

    def __call__(self, sv, Pt):
        if sv == 'D0' or FEM is None:
            one, zero = np.ones(len(Pt)), np.zeros(len(Pt))
            return dict(dpsid=zero, dpsiq=zero, rSa=one, rS2=one, rcu=one)
        if sv not in self.f:
            g = (np.array(FEM['ids']), np.array(FEM['iqs']))
            a, b = FEM['designs'][sv], FEM['designs']['D0']
            I = lambda z: RegularGridInterpolator(g, np.array(z), method='linear', bounds_error=False, fill_value=None)
            self.f[sv] = dict(dpsid=I(np.array(a['psid']) - b['psid']), dpsiq=I(np.array(a['psiq']) - b['psiq']),
                              rSa=I(np.array(a['st_Sa']) / b['st_Sa']), rS2=I(np.array(a['st_S2']) / b['st_S2']),
                              rcu=I(np.array(a['cu_B2']) / b['cu_B2']))
        return {k: f(Pt) for k, f in self.f[sv].items()}


CORR = Corr()
_FCACHE = {}


def fields(sv, s, npts, G=None):
    """Field quantities on the actual-current grid (path peak A, radius I_PATH_PK), cached per variant and turns."""
    key = (sv, round(s, 6), npts) if G is None else (sv, round(s, 6), 'G', G[0].tobytes(), G[1].tobytes())
    if key not in _FCACHE:
        i_d, i_q = BASE.grid(npts) if G is None else G
        Pt0 = np.column_stack([s * i_d, s * i_q])
        V = BASE._fields(Pt0)
        Wi = {k: np.maximum(BASE._slice_avg(fn, Pt0), 0) for k, fn in BASE.I.items()}
        C = CORR(sv, Pt0)
        _FCACHE[key] = (i_d, i_q, Pt0, V, Wi, C)
    return _FCACHE[key]


def wire_options(sv, Nc, metal, ns_max=6, d_min=1.0):
    """Per strand count: the largest wire that keeps the insulated wire fill at or below the copper design's."""
    As = geo(sv)['slot_area']
    out = []
    for ns in range(1, ns_max + 1):
        fits = [d for d in WIRE_SIZES if d >= d_min and 2 * Nc * ns * math.pi / 4 * (d + INS(d)) ** 2 <= FILL_MAX * As + 1e-9]
        if fits:
            d = fits[-1]
            out.append(dict(ns=ns, d=d, A=ns * math.pi / 4 * d * d, fill=2 * Nc * ns * math.pi / 4 * (d + INS(d)) ** 2 / As))
    return out


class Design:
    def __init__(self, metal='Cu', sv='D0', Nc=10, ns=1, d=D_CU0, label=None, Lmm=STACK):
        self.metal, self.sv, self.Nc, self.ns, self.d, self.L = metal, sv, Nc, ns, d, Lmm
        self.s, self.kL = Nc / 10.0, Lmm / STACK
        m = MAT[metal]
        self.rho = m['rho20'] * (1 + m['alpha'] * (L.T_WDG - 20))
        self.A = ns * math.pi / 4 * (d * 1e-3) ** 2
        g, g0 = geo(sv), geo('D0')
        self.geo = g
        tc = g['slot_area'] / 2 / (g['stator']['slot_depth'] - 1.3); tc0 = g0['slot_area'] / 2 / (g0['stator']['slot_depth'] - 1.3)
        self.mlt = (2 * Lmm + (KEW if metal == 'Al' else 1.0) * (MLT_END0 + 2 * (g['stator']['tooth_w'] - g0['stator']['tooth_w']) + math.pi * (tc - tc0))) * 1e-3
        self.R_path = self.rho * 2 * Nc * self.mlt / self.A
        self.fill = 2 * Nc * ns * math.pi / 4 * (d + INS(d)) ** 2 / g['slot_area']
        self.label = label or '%s %s L%.0f N%d %dx%.2f' % (metal, sv, Lmm, Nc, ns, d)

    def masses(self):
        wdg = 12 * self.Nc * self.mlt * self.A * MAT[self.metal]['dens']
        rl = ROTOR_LAM_AREA * 1e-6 * self.L * 1e-3 * 7650 * 0.97
        mg = MAGNET_AREA * 1e-6 * self.L * 1e-3 * DENS_FERRITE
        st = self.geo['st_mass'] * self.kL
        return dict(winding=wdg, stator_lam=st, rotor_lam=rl, magnets=mg, active=wdg + st + rl + mg, conductor_area_mm2=self.A * 1e6, mlt_mm=self.mlt * 1e3)

    def evaluate(self, n, npts=121, G=None):
        i_d, i_q, Pt0, V, Wi, C = fields(self.sv, self.s, npts, G)
        s, kL = self.s, self.kL
        psd0 = kL * (V['psid'] + C['dpsid']); psq0 = kL * (V['psiq'] + C['dpsiq'])
        Tem = 1.5 * P * (psd0 * 2 * Pt0[:, 1] - psq0 * 2 * Pt0[:, 0])            # Motor-CAD Lab torque, ampere-turn invariant
        psd = s * psd0 + L.L_EW_PATH * s * s * i_d; psq = s * psq0 + L.L_EW_PATH * s * s * i_q
        Ipk = np.hypot(i_d, i_q)
        wm = max(n, 1.0) * 2 * math.pi / 60; we = P * wm; f = we / (2 * math.pi)
        vd = self.R_path * i_d - we * psq; vq = self.R_path * i_q + we * psd
        v = np.hypot(vd, vq)
        pcu = 3 * self.R_path * Ipk ** 2
        pac = L.AC_ADJ * (24 * self.Nc * self.ns) * math.pi * self.L * 1e-3 * (self.d * 1e-3) ** 4 * (2 * math.pi * f) ** 2 * \
            (2 * np.maximum(V['cu_B2'] * C['rcu'], 0)) / (128 * self.rho)
        st_h = kL * L.BF * Wi['stator_H1'] * f * C['rSa']; st_e = kL * L.BF * Wi['stator_E1'] * f * f * C['rS2']
        rt_h = kL * L.BF * Wi['rotor_H1'] * f; rt_e = kL * L.BF * Wi['rotor_E1'] * f * f
        pfe = st_h + st_e + rt_h + rt_e
        b_, w_ = L.LOSS['bearing'], L.LOSS['windage']
        pme = (b_['front_W'] + b_['rear_W']) * (n / b_['ref_rpm']) ** b_['exponent'] + kL * w_['W'] * (n / w_['ref_rpm']) ** w_['exponent']
        Tsh = Tem - (pfe + pme) / wm if n > 0 else Tem.copy()
        psh = Tsh * wm
        loss = pcu + pac + pfe + pme
        pin = psh + loss
        with np.errstate(invalid='ignore', divide='ignore'):
            eff = np.where(psh > 0, 100 * psh / pin, np.nan)
        return dict(Tsh=Tsh, Tem=Tem, psh=psh, eff=eff, v=v, Ipk=Ipk, I_line=Ipk * 2 * math.sqrt(3) / math.sqrt(2), pcu=pcu, pac=pac,
                    pfe_st=st_h + st_e, pfe_rot=rt_h + rt_e, pfe_st_h=st_h, pfe_st_e=st_e, pfe_rot_h=rt_h, pfe_rot_e=rt_e,
                    psid=psd, psiq=psq, vd=vd, vq=vq, id=i_d, iq=i_q, pfe=pfe, pme=np.full_like(Tem, pme), loss=loss, pin=pin, Idc=pin / L.VDC)

    @staticmethod
    def feasible(R):
        return (R['v'] <= L.V_MAX) & (R['pin'] <= L.P_DC_MAX + 1e-6) & (R['Tsh'] > 0)

    def run(self, speeds, torques, npts=121, keys=('eff', 'loss', 'pcu', 'pac', 'pfe', 'pme', 'I_line'), tol=0.25):
        """Envelope (max shaft torque per speed) and MTPA maps on (torque, speed): least current giving the torque."""
        mp = {k: np.full((len(torques), len(speeds)), np.nan) for k in keys}
        env = []
        dT = torques[1] - torques[0]
        for a, n in enumerate(speeds):
            R = self.evaluate(max(n, 1.0), npts)
            feas = self.feasible(R)
            if not feas.any():
                env.append(None); continue
            k = np.flatnonzero(feas)[np.argmax(R['Tsh'][feas])]
            env.append({kk: float(R[kk][k]) for kk in ('Tsh', 'psh', 'eff', 'loss', 'pcu', 'pac', 'pfe', 'pme', 'I_line', 'v', 'Idc')})
            idx = np.flatnonzero(feas)
            b = np.rint((R['Tsh'][idx] - torques[0]) / dT).astype(int)
            near = (b >= 0) & (b < len(torques)) & (np.abs(R['Tsh'][idx] - (torques[0] + b * dT)) <= tol)
            idx, b = idx[near], b[near]
            o = np.lexsort((R['Ipk'][idx], b))                     # per torque bin, least current first
            idx, b = idx[o], b[o]
            first = np.r_[True, b[1:] != b[:-1]]
            for kk in keys:
                mp[kk][b[first], a] = R[kk][idx[first]]
        return env, mp


def copper():
    return Design('Cu', 'D0', 10, 1, D_CU0, label='Copper, Design9795_1 (SWG 11)')


if __name__ == '__main__':
    import time
    t0 = time.time()
    d = copper()
    print('R_path %.3f mohm (lab_map %.3f), fill %.4f, MLT %.2f mm, masses %s' % (d.R_path * 1e3, L.R_PATH * 1e3, d.fill, d.mlt * 1e3,
                                                                                {k: round(v, 3) for k, v in d.masses().items()}))
    lab = json.load(open(os.path.join(H, 'lab_100C.json')))
    sp, tq = np.array(lab['speeds'], float), np.array(lab['torques'], float)
    env, mp = d.run(sp, tq, npts=241)
    e = np.array(lab['maps']['eff'], float)
    both = ~np.isnan(e) & ~np.isnan(mp['eff'])
    print('map |diff| mean %.3f max %.3f pts on %d points (lab %d, here %d); peak %.2f vs %.2f' % (
        np.mean(np.abs(mp['eff'] - e)[both]), np.max(np.abs(mp['eff'] - e)[both]), both.sum(), (~np.isnan(e)).sum(), (~np.isnan(mp['eff'])).sum(),
        np.nanmax(mp['eff']), np.nanmax(e)))
    print('envelope max |dT| %.3f N.m' % max(abs(a['Tsh'] - b['T']) for a, b in zip(env, lab['envelope']) if a and b))
    for npts in (121, 161):
        env2, mp2 = d.run(sp, tq, npts=npts)
        b2 = ~np.isnan(e) & ~np.isnan(mp2['eff'])
        print('npts %d: map |diff| mean %.3f max %.3f, envelope max |dT| %.3f' % (npts, np.mean(np.abs(mp2['eff'] - e)[b2]), np.max(np.abs(mp2['eff'] - e)[b2]),
                                                                              max(abs(a['Tsh'] - b['T']) for a, b in zip(env2, lab['envelope']) if a and b)))
    print('%.0f s' % (time.time() - t0))
