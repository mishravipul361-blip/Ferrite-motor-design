"""Lab model v2 (100 C) - FEMM d-q map + Motor-CAD-consistent loss models.

Loss models
  v1 (first report): stator iron from one snapshot, fundamental only, eddy Keddy*f^2*B^2, rotor from slot ripple
  v2 (this):        stator + rotor iron from full-cycle FEA Fourier per element (iron_100C.json),
                     eddy 2*pi^2*Keddy*(n f)^2*B_n^2, hysteresis Kh*(n f)*B_n^alpha, build factor 1.6
  v3 (refined):      same fields; hysteresis from the peak B per element and component with Motor-CAD's minor-loop
                     factor (Kc 0.65), eddy from the harmonic sum; stator and rotor alike (iron_fit.py 'ml')
Everything else (d-q flux map, DC/AC copper, mechanical, limits) is shared with lab_map.py.
"""
import json, math, os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import lab_map as L

H = L.H
P = L.P


def grid_quad_cubic(ids, iqs, Z):
    """Interpolant for a 4 (id) x 3 (iq) grid: exact quadratic in iq per id row, cubic spline across id.
    With 4 or more iq levels (refined grid): bicubic."""
    if len(iqs) > 3:
        f_ = RegularGridInterpolator((ids, iqs), Z, method='cubic', bounds_error=False, fill_value=None)
        return lambda Pt: f_(Pt)
    from scipy.interpolate import CubicSpline
    C = np.array([np.polyfit(iqs, Z[a], len(iqs) - 1) for a in range(len(ids))])     # (n_id, 3)
    S = CubicSpline(ids, C, axis=0)

    def f(Pt):
        c = S(Pt[:, 0])
        return sum(c[:, k] * Pt[:, 1] ** (c.shape[1] - 1 - k) for k in range(c.shape[1]))
    return f


class Model:
    def __init__(self, loss='v2', skew=1.0, slices=None, torque='stress'):
        """skew: flat factor on flux and torque (0.990 = fundamental factor of the Motor-CAD skew).
        slices: rotor skew slices in mech deg, e.g. (-2, 0, 2): each slice sees the stator current turned by -P*delta
        in its own rotor frame; its d-q flux is turned back and the slices are averaged (torque and losses too).
        torque: 'stress' = FEMM air-gap force (averaged over rotor positions); 'flux' = 1.5 p (psi_d i_q - psi_q i_d),
        as Motor-CAD Lab computes it."""
        self.skew, self.slices, self.torque = skew, slices, torque
        m = json.load(open(os.path.join(H, 'dq_100C.json')))
        g = (np.array(m['id']), np.array(m['iq']))
        self.F = {k: RegularGridInterpolator(g, np.array(m[k]), method='cubic', bounds_error=False, fill_value=None)
                  for k in ('psid', 'psiq', 'T', 'st_S2', 'st_Sa', 'cu_B2', 'rot_Sa', 'rot_S2')}
        self.loss = loss
        if loss in ('v2', 'v3'):
            # fixed-mesh (air-gap element) full-cycle fields: stator = peak-B Steinmetz per element and component,
            # rotor = Steinmetz per harmonic (rotor iron only sees slot / MMF harmonics); see iron_fit.py
            fe = json.load(open(os.path.join(H, 'iron_variants.json')))
            src = dict(stator_H1='stator_peak_H', stator_E1='stator_peak_E', rotor_H1='rotor_all_H', rotor_E1='rotor_all_E')
            if loss == 'v3':
                src = dict(stator_H1='stator_ml_H', stator_E1='stator_ml_E', rotor_H1='rotor_ml_H', rotor_E1='rotor_ml_E')
            self.I = {k: grid_quad_cubic(np.array(fe['ids'], float), np.array(fe['iqs'], float), np.array(fe['grid'][v]))
                      for k, v in src.items()}

    def _slice_pts(self):
        return [0.0] if not self.slices else [np.radians(P * d) for d in self.slices]

    def _slice_avg(self, fn, Pt):
        out = 0
        for de in self._slice_pts():
            c, s_ = np.cos(de), np.sin(de)
            out = out + fn(np.column_stack([Pt[:, 0] * c + Pt[:, 1] * s_, -Pt[:, 0] * s_ + Pt[:, 1] * c]))
        return out / len(self._slice_pts())

    def _fields(self, Pt):
        """d-q map quantities, averaged over the skew slices (flux turned back into the reference frame)."""
        acc = None
        for de in self._slice_pts():
            c, s_ = np.cos(de), np.sin(de)
            Ps = np.column_stack([Pt[:, 0] * c + Pt[:, 1] * s_, -Pt[:, 0] * s_ + Pt[:, 1] * c])   # current in the slice rotor frame
            V = {k: f(Ps) for k, f in self.F.items()}
            d, q = V['psid'], V['psiq']
            V['psid'], V['psiq'] = d * c - q * s_, d * s_ + q * c                                 # flux back to the reference frame
            acc = V if acc is None else {k: acc[k] + V[k] for k in acc}
        return {k: v / len(self._slice_pts()) for k, v in acc.items()}

    def evaluate(self, n, i_d, i_q):
        """n rpm (scalar), i_d/i_q path peak A (arrays). Returns dict of arrays (Motor-CAD names where possible)."""
        Pt = np.column_stack([np.ravel(i_d), np.ravel(i_q)])
        V = self._fields(Pt)
        for k in ('psid', 'psiq', 'T'):
            V[k] = V[k] * self.skew
        if self.torque == 'flux':                                  # path quantities: phase current = 2 x path current
            V['T'] = 1.5 * P * (V['psid'] * 2 * Pt[:, 1] - V['psiq'] * 2 * Pt[:, 0])
        psd = V['psid'] + L.L_EW_PATH * Pt[:, 0]; psq = V['psiq'] + L.L_EW_PATH * Pt[:, 1]
        Tem = V['T']
        Ipk = np.hypot(Pt[:, 0], Pt[:, 1])
        wm = n * 2 * math.pi / 60; we = P * wm; f = we / (2 * math.pi)
        vd = L.R_PATH * Pt[:, 0] - we * psq; vq = L.R_PATH * Pt[:, 1] + we * psd
        v = np.hypot(vd, vq)
        pcu = 3 * L.R_PATH * Ipk ** 2
        pac = L.AC_ADJ * L.N_COND * math.pi * L.M.STACK * 1e-3 * L.D_CU ** 4 * (2 * math.pi * f) ** 2 * (2 * np.maximum(V['cu_B2'], 0)) / (
            128 * L.RHO20 * (1 + L.ALPHA_CU * (L.T_WDG - 20)))
        if self.loss in ('v2', 'v3'):
            W = {k: np.maximum(self._slice_avg(fn, Pt), 0) for k, fn in self.I.items()}
            st_h, st_e = L.BF * W['stator_H1'] * f, L.BF * W['stator_E1'] * f * f
            rt_h, rt_e = L.BF * W['rotor_H1'] * f, L.BF * W['rotor_E1'] * f * f
            pfe_st, pfe_rot = st_h + st_e, rt_h + rt_e
            hyst, eddy = st_h + rt_h, st_e + rt_e
        else:
            pfe_st = L.BF * (L.KH * f * V['st_Sa'] + L.KE * f * f * V['st_S2'])
            fs = 12 * n / 60
            pfe_rot = L.BF * (L.KH * fs * V['rot_Sa'] + L.KE * fs * fs * V['rot_S2'])
            hyst = L.BF * (L.KH * f * V['st_Sa'] + L.KH * fs * V['rot_Sa']); eddy = pfe_st + pfe_rot - hyst
        pme = L.mech_loss(n)
        pfe = pfe_st + pfe_rot
        Tsh = Tem - (pfe + pme) / wm if n > 0 else Tem.copy()
        psh = Tsh * wm
        loss = pcu + pac + pfe + pme
        pin = psh + loss
        with np.errstate(invalid='ignore', divide='ignore'):
            eff = np.where(psh > 0, 100 * psh / pin, np.nan)
            pf = (vd * Pt[:, 0] + vq * Pt[:, 1]) / (v * Ipk)
        return dict(Shaft_Torque=Tsh, Electromagnetic_Torque=Tem, Shaft_Power=psh, Efficiency=eff,
                    Stator_Current_Line_RMS=Ipk * 2 * math.sqrt(3) / math.sqrt(2), Id_Peak=2 * Pt[:, 0], Iq_Peak=2 * Pt[:, 1],
                    Voltage_Phase_Peak=v, Stator_Copper_Loss_DC=pcu, Stator_Copper_Loss_AC=pac, Iron_Loss_Stator=pfe_st,
                    Iron_Loss_Rotor=pfe_rot, Iron_Loss=pfe, Iron_Loss_Hysteresis=hyst, Iron_Loss_Eddy=eddy, Mechanical_Loss=np.full_like(Tem, pme),
                    Total_Loss=loss, Idc=pin / L.VDC, Terminal_Power=pin, Power_Factor=pf, Flux_Linkage_D=psd * 1e3, Flux_Linkage_Q=psq * 1e3,
                    Phase_Advance=np.degrees(np.arctan2(-Pt[:, 0], Pt[:, 1])), Ipk=Ipk)

    def grid(self, npts=241):
        i_d, i_q = np.meshgrid(np.linspace(-L.I_PATH_PK, 0, npts), np.linspace(0, L.I_PATH_PK, npts), indexing='ij')
        ok = np.hypot(i_d, i_q) <= L.I_PATH_PK + 1e-9
        return i_d[ok], i_q[ok]

    def operate(self, n, T_list, G=None):
        """MTPA / field-weakening operating point for each shaft torque at speed n (least current that meets the
        torque within the voltage, current and DC-current limits). Returns list of dicts (None if not reachable)."""
        i_d, i_q = G if G is not None else self.grid()
        R = self.evaluate(n, i_d, i_q)
        feas = (R['Voltage_Phase_Peak'] <= L.V_MAX) & (R['Terminal_Power'] <= L.P_DC_MAX + 1e-6)
        out = []
        for T in T_list:
            c = np.flatnonzero(feas & (np.abs(R['Shaft_Torque'] - T) <= 0.15))
            if not len(c):
                out.append(None); continue
            k = c[np.argmin(R['Ipk'][c])]
            out.append({key: float(v[k]) for key, v in R.items()})
        return out

    def envelope_point(self, n, G=None):
        i_d, i_q = G if G is not None else self.grid()
        R = self.evaluate(n, i_d, i_q)
        feas = (R['Voltage_Phase_Peak'] <= L.V_MAX) & (R['Terminal_Power'] <= L.P_DC_MAX + 1e-6)
        if not feas.any():
            return None
        k = np.flatnonzero(feas)[np.argmax(R['Shaft_Torque'][feas])]
        return {key: float(v[k]) for key, v in R.items()}
