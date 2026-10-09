"""Efficiency / loss / current / power-factor maps over the operating envelope (52 V, 360 A rms, 11 kW DC).

For each (speed, torque) the d-q current with the lowest total loss is chosen from the FEMM d-q map (dq.json).
Losses:
  copper   DC, 100 C (AC/proximity loss not included)
  iron     stator: sum over FEMM elements of (KH f + KE f^2) Bpk^2 m, Bpk from the no-load solution
           (field_noload.ans) scaled by |psi_dq|/psi_m, x BUILD; rotor = ROTOR_FE x stator
  mech     bearings + windage, MECH_A n + MECH_B n^2
  inverter constant ETA_INV (DC power = (shaft + motor losses) / ETA_INV)
Run after dq_study.py and field_maps.py:  python eff_map.py  (seconds) -> eff.json, fig_eff.png
"""
import json, math, os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import femm_motor as M
import dq_study as D
from field_maps import read_ans, element_fields

HERE = M.HERE
KH, KE = 0.0177, 6.2e-5      # W/kg/(Hz T^2), W/kg/(Hz^2 T^2): M235-35A typ. 2.35 W/kg @1.5T 50Hz, ~17 W/kg @1T 400Hz
BUILD, ROTOR_FE = 1.5, 0.10  # punching/rotational/harmonic build factor; rotor iron share of stator iron loss
MECH_A, MECH_B = 4e-3, 2.2e-7  # W/rpm, W/rpm^2 -> ~14 W @3000, ~80 W @12000 rpm
DENS = 7650                  # kg/m^3 lamination


def stator_B2m():
    """sum(Bpk^2 * mass) over stator iron at no load [T^2 kg]; Bpk^2 ~ 2 x instantaneous B^2 averaged over the stator."""
    nodes, el, labs = read_ans(os.path.join(HERE, 'field_noload.ans'))
    Bm, _, mat = element_fields(nodes, el, labs)
    p = nodes[el[:, :3], :2]
    area = 0.5 * abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    r = np.hypot(p[:, :, 0].mean(1), p[:, :, 1].mean(1))
    st = (mat == 3) & (r > M.R_BORE - 0.1)
    mass = area * 1e-6 * M.STACK * 1e-3 * M.STACKING * DENS
    return float(2 * (Bm[st] ** 2 * mass[st]).sum()), float(mass[st].sum())


def main():
    m = json.load(open(os.path.join(HERE, 'dq.json')))
    B2m, m_st = stator_B2m()
    g = (np.array(m['id']), np.array(m['iq']))
    fd, fq, fT = (RegularGridInterpolator(g, np.array(m[k]), method='cubic') for k in ('psid', 'psiq', 'T'))
    psim = float(fd([[0, 0]])[0])
    ilim = D.I_PEAK
    i_d, i_q = np.meshgrid(np.linspace(-ilim, 0, 181), np.linspace(0, ilim, 181), indexing='ij')
    ok = np.hypot(i_d, i_q) <= ilim + 1e-9
    P = np.column_stack([i_d[ok], i_q[ok]])
    psd, psq, Tq = fd(P), fq(P), fT(P)
    pcu = 3 * D.R_PATH * (P[:, 0] ** 2 + P[:, 1] ** 2)
    flux_ratio2 = (psd ** 2 + psq ** 2) / psim ** 2
    speeds = np.arange(250, 12001, 250); torques = np.arange(1, 46.01, 1.0)
    keys = ('eff', 'eff_sys', 'loss', 'pcu', 'pfe', 'pmech', 'I_rms', 'pf', 'P_dc_kW', 'id', 'iq')
    out = {k: np.full((len(torques), len(speeds)), np.nan) for k in keys}
    env = []
    for a, n in enumerate(speeds):
        w = n * 2 * math.pi / 60 * M.P; f = w / (2 * math.pi)
        vd, vq = D.R_PATH * P[:, 0] - w * psq, D.R_PATH * P[:, 1] + w * psd
        v = np.hypot(vd, vq)
        pfe = (KH * f + KE * f * f) * B2m * flux_ratio2 * BUILD * (1 + ROTOR_FE)
        pme = MECH_A * n + MECH_B * n * n
        pout = Tq * n * 2 * math.pi / 60 - pme              # shaft torque = EM torque - mech drag
        loss = pcu + pfe + pme
        pdc = (pout + loss) / D.ETA_INV
        feas = (v <= D.V_MAX) & (pdc <= D.P_DC_MAX) & (pout > 0)
        if not feas.any():
            break
        tsh = pout / (n * 2 * math.pi / 60)                  # shaft torque
        env.append(dict(rpm=float(n), T=float(tsh[feas].max()), P_kW=float(pout[feas].max() / 1e3)))
        for b, T in enumerate(torques):
            c = np.flatnonzero(feas & (abs(tsh - T) <= 0.5))
            if not len(c):
                continue
            k = c[np.argmin(loss[c])]
            ip = math.hypot(*P[k])
            vals = dict(eff=pout[k] / (pout[k] + loss[k]) * 100,
                        eff_sys=pout[k] / pdc[k] * 100, loss=loss[k], pcu=pcu[k], pfe=pfe[k], pmech=pme,
                        I_rms=ip * 2 * math.sqrt(3) / math.sqrt(2),
                        pf=(vd[k] * P[k, 0] + vq[k] * P[k, 1]) / (v[k] * ip), P_dc_kW=pdc[k] / 1e3, id=P[k, 0], iq=P[k, 1])
            for kk, vv in vals.items():
                out[kk][b, a] = vv
    res = dict(speeds=speeds.tolist(), torques=torques.tolist(), envelope=env,
               model=dict(KH=KH, KE=KE, BUILD=BUILD, ROTOR_FE=ROTOR_FE, MECH_A=MECH_A, MECH_B=MECH_B,
                          stator_mass_kg=m_st, stator_B2m_noload=B2m, ETA_INV=D.ETA_INV, P_DC_MAX=D.P_DC_MAX, V_MAX=D.V_MAX),
               **{k: np.where(np.isnan(v), None, np.round(v, 4)).tolist() for k, v in out.items()})
    json.dump(res, open(os.path.join(HERE, 'eff.json'), 'w'))
    e = out['eff']
    k = np.unravel_index(np.nanargmax(e), e.shape)
    print('stator iron mass %.2f kg; peak motor efficiency %.1f %% at %d rpm, %.0f N.m; peak system %.1f %%'
          % (m_st, e[k], speeds[k[1]], torques[k[0]], np.nanmax(out['eff_sys'])))
    for n0 in (1000, 3000, 6000, 9000, 12000):
        a = int(np.argmin(abs(speeds - n0)))
        print('  %5d rpm: max shaft T %.1f N.m, motor eff at 10 N.m %.1f %%, iron loss @10 N.m %.0f W'
              % (speeds[a], env[a]['T'] if a < len(env) else float('nan'), out['eff'][9, a], out['pfe'][9, a]))
    fig, ax = plt.subplots(figsize=(10, 6))
    cs = ax.contourf(speeds, torques, e, levels=[70, 75, 80, 84, 86, 88, 90, 91, 92, 93, 94, 95], cmap='viridis')
    ax.clabel(ax.contour(speeds, torques, e, levels=[80, 85, 88, 90, 92, 94], colors='k', linewidths=.5), fmt='%d')
    ax.plot([r['rpm'] for r in env], [r['T'] for r in env], 'k-', lw=2)
    fig.colorbar(cs, label='motor efficiency %'); ax.set(xlabel='rpm', ylabel='shaft torque N.m', title='Efficiency map')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_eff.png'), dpi=110)


if __name__ == '__main__':
    main()
