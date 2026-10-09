"""Turn results.json from femm_motor.py into results.png, summary.txt and PNG field plots."""
import json, math, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, 'results.json')))
inp = r['inputs']; p = 5
R_PATH = 2 * 6.732e-3            # 2 coils in series per parallel path, 100 C
V_MAX = 0.95 * inp['Vdc']        # SVPWM line-line peak = Vdc; delta -> phase = line; 5 % for drops/dead-time
Ip_line = math.sqrt(2) * inp['I_line_rms']; Ip_ph = Ip_line / math.sqrt(3); Ip_path = Ip_ph / 2


def phasor(th_m, x):
    th = np.radians(p * np.asarray(th_m))
    return 2 * np.mean(np.asarray(x) * np.exp(-1j * th))


# ---- no load
nl = r['noload']; th = np.array(nl['theta_m'])
psi1 = np.mean(list(nl['psi1_peak'].values()))
w1k = 2 * math.pi * 1000 / 60 * p
E_ll_rms_1k = w1k * psi1 / math.sqrt(2)                   # delta: line-line = path voltage
# ---- load
ld = r['load']; T = np.array(ld['torque']); thl = np.array(ld['theta_m'])
Tavg = T.mean(); ripple = (T.max() - T.min()) / Tavg * 100
g = r['gamma_sweep']['gamma_opt']; phi = nl['phi']
Pcu = 3 * (inp['I_line_rms'] / math.sqrt(3)) ** 2 * 6.732e-3
cog = np.array(r['cogging']['torque'])
f = {k: v for k, v in r['fields'].items() if v}      # a dropped (non-converged) case is None

lines = [
    'v7 12s/10p spoke ferrite motor - FEMM 2D magnetostatic, stack %.0f mm' % inp['stack_mm'],
    'Ferrite Br %.2f T (mu_r %.2f, HcJ %.0f kA/m)   steel %s, stacking %.2f' % (inp['Br'], inp['mu_r'], inp['HcJ'] / 1e3, inp['steel'], inp['stacking']),
    'Current %.0f A rms line (delta) = %.1f A rms phase = %.1f A rms per coil; peak line %.0f A' % (inp['I_line_rms'], inp['I_line_rms'] / math.sqrt(3), inp['I_line_rms'] / math.sqrt(3) / 2, Ip_line),
    '',
    'NO LOAD',
    '  flux linkage per path (fundamental, peak)   %.4f Wb-turn' % psi1,
    '  back-EMF line-line @1000 rpm                 %.2f V rms  (%.2f V peak)' % (E_ll_rms_1k, E_ll_rms_1k * math.sqrt(2)),
    '  cogging torque                               %.3f N.m pk-pk' % (cog.max() - cog.min()),
    '  air-gap B (radial) peak                      %s' % ('%.3f T' % max(abs(x) for x in f['noload']['gap_Br']) if 'noload' in f else 'n/a'),
    '',
    'RATED %.0f A rms' % inp['I_line_rms'],
    '  best current angle gamma (MTPA)              %.1f deg (advance from q-axis)' % g,
    '  average torque                               %.2f N.m' % Tavg,
    '  torque ripple                                %.1f %% pk-pk  (%.2f..%.2f N.m)' % (ripple, T.min(), T.max()),
    '  torque constant Kt                           %.4f N.m/A rms line' % (Tavg / inp['I_line_rms']),
    '  torque with magnets at 80 C (Br %.3f)        %.2f N.m' % (r['hot80']['Br'], r['hot80']['torque']),
    '  tooth B max / yoke B max                     %s' % ('%.2f / %.2f T' % (max(f['rated']['tooth_B']), max(f['rated']['yoke_B'])) if 'rated' in f else 'n/a'),
    '  copper loss (100 C, DC)                      %.0f W' % Pcu,
    '  base speed / torque-speed                    see dq_summary.txt',
    '',
    'MAGNET DEMAGNETISATION (min H along magnetisation, min B along magnetisation)',
]
for k in [k for k in ('noload', 'rated', 'demag_gamma90') if k in f]:
    h = f[k]['H_along_M_min']; lines.append('  %-14s H %7.0f kA/m (margin to -HcJ %5.0f kA/m)   B %.3f T' % (k, h / 1e3, (h + inp['HcJ']) / 1e3, f[k]['B_along_M_min']))
lines += ['', 'TORQUE vs CURRENT (gamma fixed at rated MTPA)']
lines += ['  %5.0f A rms  %6.2f N.m' % (i, t) for i, t in zip(r['current_sweep']['I_line_rms'], r['current_sweep']['torque'])]
lines += ['', 'Not in a 2D model: end-winding leakage (raises voltage, lowers base speed), axial end effects of the',
          'ferrite spoke rotor (3D torque typically 5-15 % lower), iron/magnet/AC copper losses.']
txt = '\n'.join(lines); print(txt)
open(os.path.join(HERE, 'summary.txt'), 'w').write(txt + '\n')

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
e = {ph: np.gradient(np.array(nl['psi'][ph]), np.radians(th)) * 2 * math.pi * 1000 / 60 for ph in 'RYB'}
for ph, c in zip('RYB', ('#c0392b', '#d4a017', '#2d5c9f')):
    ax[0, 0].plot(th * p, e[ph], c, label=ph)
ax[0, 0].set(title='Back-EMF per phase (= line-line, delta) @1000 rpm', xlabel='electrical deg', ylabel='V'); ax[0, 0].legend()
ax[0, 1].plot(r['cogging']['theta_m'], cog, 'k.-'); ax[0, 1].set(title='Cogging torque', xlabel='rotor deg (mech)', ylabel='N.m')
ax[0, 2].plot(r['gamma_sweep']['gamma'], r['gamma_sweep']['torque'], 'o-'); ax[0, 2].axvline(g, ls=':')
ax[0, 2].set(title='Torque vs current angle @ %.0f A rms' % inp['I_line_rms'], xlabel='gamma (deg, + = field weakening)', ylabel='N.m')
ax[1, 0].plot(thl, T, 'b-'); ax[1, 0].axhline(Tavg, ls=':')
ax[1, 0].set(title='Rated torque: avg %.1f N.m, ripple %.0f %%' % (Tavg, ripple), xlabel='rotor deg (mech)', ylabel='N.m')
ax[1, 1].plot(r['current_sweep']['I_line_rms'], r['current_sweep']['torque'], 'o-'); ax[1, 1].set(title='Torque vs line current', xlabel='A rms line', ylabel='N.m')
ga = np.arange(0, 360, 0.5)
for k in [k for k in ('noload', 'rated') if k in f]:
    ax[1, 2].plot(ga, f[k]['gap_Br'], lw=.8, label=k.replace('noload', 'no load'))
ax[1, 2].set(title='Air-gap radial B', xlabel='deg (mech)', ylabel='T'); ax[1, 2].legend()
for a_ in ax.flat: a_.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'results.png'), dpi=110)


