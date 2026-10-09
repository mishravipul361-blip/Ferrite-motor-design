"""d-q flux-linkage map (saturation included) -> Ld, Lq and torque-speed envelope on the 52 V battery.

All d-q quantities are per parallel path (coil current, path flux linkage), peak values.
Phase (delta) current = 2 x path current; phase voltage = path voltage = line-line voltage.
Run: python dq_study.py   (~1 h solve, cached in dq.json; delete it to re-solve)
"""
import json, math, os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import femm_motor as M

HERE = M.HERE
R_PATH = 2 * M.R_COIL[100]                  # 2 coils in series per path, 100 C
V_MAX = 0.95 * M.VDC                        # SVPWM: line-line peak = Vdc; 5 % for drops/dead-time
P_DC_MAX = 11e3                             # W, battery-side (DC) power limit
ETA_INV = 0.97                              # inverter efficiency (DC -> AC)
I_PEAK = math.sqrt(2) * M.I_LINE_RMS / math.sqrt(3) / 2   # path peak at 360 A rms line (147 A)
IDS = np.linspace(-150, 0, 5); IQS = np.linspace(0, 150, 5); POS = [0, 4, 8]   # spans the 12 deg torque-ripple period


def currents(th_m, phi, i_d, i_q):
    return {ph: i_d * math.cos(math.radians(M.P * th_m + phi[ph])) - i_q * math.sin(math.radians(M.P * th_m + phi[ph]))
            for ph in 'RYB'}


def park(th_m, phi, psi):
    a = {ph: math.radians(M.P * th_m + phi[ph]) for ph in 'RYB'}
    return (2 / 3 * sum(psi[ph] * math.cos(a[ph]) for ph in 'RYB'),
            -2 / 3 * sum(psi[ph] * math.sin(a[ph]) for ph in 'RYB'))


def solve_map():
    from multiprocessing import Pool
    r = json.load(open(os.path.join(HERE, 'results.json')))
    phi, sgn = r['noload']['phi'], r['gamma_sweep']['sign']
    pts = [(i_d, i_q, t) for i_d in IDS for i_q in IQS for t in POS]
    with Pool(M.NWORK, initializer=M.worker_init) as pool:
        out = pool.map(M.job, [(t, currents(t, phi, i_d, i_q), M.BR, None) for i_d, i_q, t in pts], chunksize=1)
    pd, pq, T = (np.zeros((len(IDS), len(IQS))) for _ in range(3))
    for (i_d, i_q, t), (tq, psi, _) in zip(pts, out):
        a, b = list(IDS).index(i_d), list(IQS).index(i_q)
        d, q = park(t, phi, psi)
        pd[a, b] += d / len(POS); pq[a, b] += q / len(POS); T[a, b] += sgn * tq / len(POS)
    m = dict(id=IDS.tolist(), iq=IQS.tolist(), psid=pd.tolist(), psiq=pq.tolist(), T=T.tolist())
    json.dump(m, open(os.path.join(HERE, 'dq.json'), 'w'), indent=1)
    return m


def envelope(m, i_lim, speeds, p_dc=None):
    """Max torque per speed under |i| <= i_lim (path peak), |v| <= V_MAX and, if given, DC input power <= p_dc.
    DC power = (shaft power + copper loss) / ETA_INV; iron loss not included."""
    g = (np.array(m['id']), np.array(m['iq']))
    fd, fq, fT = (RegularGridInterpolator(g, np.array(m[k]), method='cubic') for k in ('psid', 'psiq', 'T'))
    i_d, i_q = np.meshgrid(np.linspace(-i_lim, 0, 121), np.linspace(0, i_lim, 121), indexing='ij')
    ok = np.hypot(i_d, i_q) <= i_lim + 1e-9
    P = np.column_stack([i_d[ok], i_q[ok]])
    psd, psq, T = fd(P), fq(P), fT(P)
    pcu = 3 * R_PATH * (P[:, 0] ** 2 + P[:, 1] ** 2)       # 6 paths x R x (i_peak^2 / 2)
    rows = []
    for n in speeds:
        w = n * 2 * math.pi / 60 * M.P
        v = np.hypot(R_PATH * P[:, 0] - w * psq, R_PATH * P[:, 1] + w * psd)
        pdc = (T * n * 2 * math.pi / 60 + pcu) / ETA_INV
        f = (v <= V_MAX) & ((pdc <= p_dc) if p_dc else True)
        if not f.any() or T[f].max() <= 0:
            break
        c = np.flatnonzero(f & (T >= T[f].max() - 0.05))      # near-max torque -> least current (min copper loss)
        k = c[np.argmin(pcu[c])]
        rows.append(dict(rpm=n, T=float(T[k]), P_kW=float(T[k] * n * 2 * math.pi / 60 / 1e3), id=float(P[k, 0]),
                         iq=float(P[k, 1]), V=float(v[k]), P_dc_kW=float(pdc[k] / 1e3), Pcu_W=float(pcu[k]), I_line_rms=float(np.hypot(*P[k]) * 2 * math.sqrt(3) / math.sqrt(2))))
    return rows


def main():
    f = os.path.join(HERE, 'dq.json')
    m = json.load(open(f)) if os.path.exists(f) else solve_map()
    ids, iqs = np.array(m['id']), np.array(m['iq'])
    pd, pq, T = (np.array(m[k]) for k in ('psid', 'psiq', 'T'))
    psim = pd[-1, 0]                                     # id = iq = 0
    # apparent inductances per path; delta phase = 2 paths in parallel -> /2
    Ld = (pd[:-1, :] - psim) / ids[:-1, None]            # id != 0 rows
    Lq = pq[:, 1:] / iqs[None, 1:]                        # iq != 0 cols
    # check the d-q torque formula against FEMM's stress-tensor torque
    Tdq = 2 * 1.5 * M.P * (pd * iqs[None, :] - pq * ids[:, None])
    speeds = np.arange(0, 12001, 50)
    il = lambda lim: lim * 2 * math.sqrt(3) / math.sqrt(2)      # path peak -> line rms
    env = {'%.0f A rms + %.0f kW DC' % (il(I_PEAK), P_DC_MAX / 1e3): (envelope(m, I_PEAK, speeds, P_DC_MAX), '-'),
           '%.0f A rms + %.0f kW DC' % (il(I_PEAK / 2), P_DC_MAX / 1e3): (envelope(m, I_PEAK / 2, speeds, P_DC_MAX), '--'),
           '%.0f A rms, no DC limit' % il(I_PEAK): (envelope(m, I_PEAK, speeds), ':')}

    L = []
    L.append('D-Q MAP  (path values, peak; phase current = 2 x path current)')
    L.append('  magnet flux linkage psi_m            %.4f Wb-turn' % psim)
    L.append('  Ld per path, unsat (id=-37.5 A)      %.1f uH    -> per delta phase %.1f uH' % (Ld[-1, 0] * 1e6, Ld[-1, 0] * 5e5))
    L.append('  Lq per path, unsat (iq=37.5 A)       %.1f uH    -> per delta phase %.1f uH' % (Lq[-1, 0] * 1e6, Lq[-1, 0] * 5e5))
    k, j = 2, 3                                           # grid point id -75, iq 112.5 ~ rated MTPA (|i| 135 A)
    ldr, lqr = (pd[k, j] - psim) / ids[k], pq[k, j] / iqs[j]
    L.append('  at load (id %.0f, iq %.1f, ~rated):    Ld %.1f uH, Lq %.1f uH per path (delta phase %.1f / %.1f uH), Lq/Ld %.2f'
             % (ids[k], iqs[j], ldr * 1e6, lqr * 1e6, ldr * 5e5, lqr * 5e5, lqr / ldr))
    L.append('  characteristic current psi_m/Ld      %.0f A path peak = %.0f A rms line  (current limit %.0f A rms)'
             % (psim / Ld[-1, 0], psim / Ld[-1, 0] * 2 * math.sqrt(3) / math.sqrt(2), M.I_LINE_RMS))
    L.append('  d-q torque formula vs FEMM torque    max diff %.1f N.m' % np.max(abs(Tdq - T)))
    for name, (rows, _) in env.items():
        L.append('')
        L.append('TORQUE-SPEED ENVELOPE  %s, %.0f V (phase peak %.1f V), R at 100 C, inverter eff %.2f' % (name, M.VDC, V_MAX, ETA_INV))
        L.append('     rpm    T N.m  shaft kW  DC kW  Cu loss W  I A rms    id    iq   V peak')
        for rw in rows:
            if rw['rpm'] % 1000 == 0 or rw is rows[-1]:
                L.append('  %6d  %7.2f  %7.2f  %6.2f  %8.0f  %7.0f  %5.0f %5.0f  %6.1f' % (rw['rpm'], rw['T'], rw['P_kW'], rw['P_dc_kW'],
                         rw['Pcu_W'], rw['I_line_rms'], rw['id'], rw['iq'], rw['V']))
        T0 = rows[0]['T']; base = next((rw['rpm'] for rw in rows if rw['T'] < 0.98 * T0), rows[-1]['rpm'])
        L.append('  base speed (torque < 98 %% of zero-speed) ~%d rpm, peak power %.2f kW, max speed %d rpm'
                 % (base, max(rw['P_kW'] for rw in rows), rows[-1]['rpm']))
    L += ['', 'Calibration (motor_config.json): %s' % {k: v for k, v in M.CFG.items() if not k.startswith('_')},
          'Not included: end-winding leakage inductance (adds to Ld and Lq -> lower base speed),',
          'iron and mechanical losses (shaft power is a little lower, DC power a little higher), 3D magnet end leakage.']
    txt = '\n'.join(L); print(txt)
    open(os.path.join(HERE, 'dq_summary.txt'), 'w').write(txt + '\n')

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for b, c in zip(range(len(iqs)), plt.cm.viridis(np.linspace(0, .9, len(iqs)))):
        ax[0, 0].plot(ids, pd[:, b] * 1e3, 'o-', c=c, label='iq %.0f A' % iqs[b])
    ax[0, 0].set(title='psi_d vs i_d (path)', xlabel='i_d path peak (A)', ylabel='mWb-turn'); ax[0, 0].legend(fontsize=8)
    for a, c in zip(range(len(ids)), plt.cm.plasma(np.linspace(0, .9, len(ids)))):
        ax[0, 1].plot(iqs, pq[a, :] * 1e3, 'o-', c=c, label='id %.0f A' % ids[a])
    ax[0, 1].set(title='psi_q vs i_q (path)', xlabel='i_q path peak (A)', ylabel='mWb-turn'); ax[0, 1].legend(fontsize=8)
    ax2 = ax[1, 0].twinx()
    for name, (rows, ls) in env.items():
        n = [rw['rpm'] for rw in rows]
        ax[1, 0].plot(n, [rw['T'] for rw in rows], 'b' + ls, label='torque, ' + name)
        ax2.plot(n, [rw['P_kW'] for rw in rows], 'r' + ls, label='shaft power, ' + name)
        if ls != ':':
            ax[1, 1].plot(n, [rw['id'] for rw in rows], 'g' + ls, label='i_d, ' + name)
            ax[1, 1].plot(n, [rw['iq'] for rw in rows], 'm' + ls, label='i_q, ' + name)
    ax[1, 0].set(title='Torque-speed envelope, %.0f V DC' % M.VDC, xlabel='rpm', ylabel='N.m'); ax2.set_ylabel('kW', color='r')
    ax[1, 0].legend(loc='lower left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
    ax[1, 1].set(title='Optimal d-q currents (path peak)', xlabel='rpm', ylabel='A'); ax[1, 1].legend(fontsize=8)
    for a_ in ax.flat: a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'dq_results.png'), dpi=110)


if __name__ == '__main__':
    main()
