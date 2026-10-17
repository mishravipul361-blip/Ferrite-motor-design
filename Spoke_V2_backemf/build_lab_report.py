"""Lab report (100 C): torque-speed, efficiency and loss maps + back-EMF validation -> lab_report.html"""
import base64, io, json, math, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(H, 'lab_100C.json')))
fine = json.load(open(os.path.join(H, 'hires_20C_fine_mcadBH.json')))
cogf = json.load(open(os.path.join(H, 'hires_20C_cog_fillet.json')))
mc = json.load(open(os.path.join(H, 'mcad', 'mcad_digitised.json')))
S, T = np.array(r['speeds']), np.array(r['torques'])
Mp = {k: np.array(v, dtype=float) for k, v in r['maps'].items()}
env = r['envelope']; info = r['info']; st = r['settings']
en, eT, eP = [e['rpm'] for e in env], [e['T'] for e in env], [e['P_kW'] for e in env]
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 110})


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', bbox_inches='tight', facecolor='white'); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def cmap(key, title, unit, levels=None, cm='viridis', fmt='%.0f', clabels=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    Z = Mp[key]
    cs = ax.contourf(S, T, Z, levels=levels if levels is not None else 14, cmap=cm, extend='both')
    if clabels is not None:
        ax.clabel(ax.contour(S, T, Z, levels=clabels, colors='k', linewidths=.6), fmt=fmt, fontsize=8)
    ax.plot(en, eT, 'k-', lw=2)
    fig.colorbar(cs, ax=ax, label=unit)
    ax.set(xlabel='speed (rpm)', ylabel='shaft torque (N·m)', title=title, xlim=(0, 12000), ylim=(0, 48))
    return png(fig)


img = {}
# torque / power vs speed
fig, ax = plt.subplots(figsize=(9, 4.6)); ax2 = ax.twinx()
ax.plot(en, eT, 'C0', lw=2.2, label='shaft torque'); ax2.plot(en, eP, 'C3', lw=2.2, label='shaft power')
n_dc = next(e['rpm'] for e in env if 'dc' in e['limit'])
ax.axvline(n_dc, color='0.6', ls=':', lw=1)
ax.text(n_dc / 2, 47, 'current limit (360 A rms)', ha='center', fontsize=8, color='0.3')
ax.text((n_dc + 12000) / 2, 47, 'DC current limit (280 A), with voltage limit from ~6000 rpm', ha='center', fontsize=8, color='0.3')
ax.set(xlabel='speed (rpm)', ylabel='torque (N·m)', ylim=(0, 50), xlim=(0, 12000)); ax2.set(ylabel='power (kW)', ylim=(0, 16))
ax2.grid(False); fig.legend(loc='upper right', bbox_to_anchor=(.88, .88))
ax.set_title('Torque and power envelope, 52 V, 360 A rms, 280 A DC, 100 °C')
img['env'] = png(fig)
# envelope currents / voltage / losses
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].plot(en, [e['I_line'] for e in env], label='line current A rms'); ax[0].plot(en, [e['I_dc'] for e in env], label='DC current A')
ax[0].plot(en, [e['V'] * 10 for e in env], label='phase voltage peak ×10 V'); ax[0].axhline(494, color='0.5', ls=':', lw=1)
ax[0].set(xlabel='rpm', title='Envelope: currents and voltage', xlim=(0, 12000)); ax[0].legend(fontsize=8)
for k, lab in (('pcu', 'copper DC'), ('pac', 'copper AC'), ('pfe', 'iron'), ('pmech', 'mechanical')):
    ax[1].plot(en, [e[k] for e in env], label=lab)
ax[1].set(xlabel='rpm', ylabel='W', title='Losses along the envelope', xlim=(0, 12000)); ax[1].legend(fontsize=8)
img['env2'] = png(fig)
img['eff'] = cmap('eff', 'Motor efficiency (shaft / electrical input)', '%', levels=[70, 75, 80, 84, 86, 88, 90, 91, 92, 93, 94, 95, 96, 97], cm='RdYlGn',
                  clabels=[80, 85, 88, 90, 92, 94, 95, 96])
img['loss'] = cmap('loss', 'Total loss', 'W', levels=np.arange(0, 1601, 100), cm='magma_r', clabels=[100, 200, 400, 600, 800, 1000])
img['pcu'] = cmap('pcu', 'Copper loss, DC (100 °C)', 'W', levels=np.arange(0, 901, 50), cm='YlOrRd', clabels=[50, 100, 200, 400, 600, 800])
img['pac'] = cmap('pac', 'Copper loss, AC (hybrid estimate ×2)', 'W', levels=np.arange(0, 1001, 50), cm='YlOrRd', clabels=[25, 50, 100, 200, 400, 800])
Mp['pfe'] = Mp['pfe_st'] + Mp['pfe_rot']
img['pfe'] = cmap('pfe', 'Iron loss, stator + rotor (build factor 1.6)', 'W', levels=np.arange(0, 501, 25), cm='YlOrRd', clabels=[25, 50, 100, 200, 300, 400])
img['I'] = cmap('I_line', 'Line current', 'A rms', levels=np.arange(0, 361, 20), cm='viridis', clabels=[50, 100, 150, 200, 250, 300])
img['Idc'] = cmap('I_dc', 'DC current', 'A', levels=np.arange(0, 281, 20), cm='viridis', clabels=[50, 100, 150, 200, 250])
img['V'] = cmap('V_ph', 'Phase voltage (peak)', 'V', levels=np.arange(0, 50, 2.5), cm='viridis', clabels=[10, 20, 30, 40, 49])
img['pf'] = cmap('pf', 'Power factor', '', levels=np.arange(0.5, 1.001, 0.025), cm='viridis', fmt='%.2f', clabels=[0.7, 0.8, 0.9, 0.95])
img['gamma'] = cmap('gamma', 'Current advance angle γ (from q-axis)', 'el. deg', levels=np.arange(0, 91, 5), cm='viridis', clabels=[10, 20, 30, 45, 60, 75])
# d-q map
dq = r['dq']; ids, iqs = np.array(dq['id']), np.array(dq['iq'])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for b, c in zip(range(len(iqs)), plt.cm.viridis(np.linspace(0, .9, len(iqs)))):
    ax[0].plot(ids, np.array(dq['psid'])[:, b] * 1e3, 'o-', c=c, label='iq %.0f' % iqs[b])
for a, c in zip(range(len(ids)), plt.cm.plasma(np.linspace(0, .9, len(ids)))):
    ax[1].plot(iqs, np.array(dq['psiq'])[a, :] * 1e3, 'o-', c=c, label='id %.0f' % ids[a])
cs = ax[2].contourf(ids, iqs, np.array(dq['T']).T, levels=14, cmap='viridis'); fig.colorbar(cs, ax=ax[2], label='N·m')
th = np.linspace(math.pi / 2, math.pi, 50); ax[2].plot(info['I_path_pk'] * np.cos(th), info['I_path_pk'] * np.sin(th), 'w--', lw=1)
ax[0].set(title='ψd (per path)', xlabel='id path peak (A)', ylabel='mWb'); ax[0].legend(fontsize=7)
ax[1].set(title='ψq (per path)', xlabel='iq path peak (A)', ylabel='mWb'); ax[1].legend(fontsize=7)
ax[2].set(title='Torque map (dashed: 360 A rms limit)', xlabel='id path peak (A)', ylabel='iq path peak (A)')
img['dq'] = png(fig)
# back-EMF validation
psi = np.array(fine['path_psi']['Ph1']); N = len(psi); c = np.fft.rfft(psi) / N; we = 5 * 2 * math.pi * 1000 / 60
E = -c * 1j * np.arange(len(c)) * we; e = np.fft.irfft(E * N, N); amp = 2 * np.abs(E); amp[0] = 0
th_f = 5 * np.array(fine['theta_m']); th_m = np.array(mc['backemf']['theta_e']); e_m = np.array(mc['backemf']['phase1'])
fund = lambda t, y: 2 * np.mean(y * np.exp(-1j * np.radians(t)))
d = math.degrees(np.angle(fund(th_f, e)) - np.angle(fund(th_m, e_m)))
def psi_at(t): return sum((2 * c[n] * np.exp(1j * n * t)).real for n in range(1, N // 2))
t30 = np.arange(30) * 2 * math.pi / 30; p30 = psi_at(t30); e30 = -(np.roll(p30, -1) - p30) / (2 * math.pi / 30 / we)
h30 = np.abs(np.fft.rfft(e30)) / 30 * 2
thd = lambda h, n=13: 100 * math.sqrt(sum(h[2:n + 1] ** 2)) / h[1]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
ax[0].plot(th_m, e_m, 'C1', lw=2, label='Motor-CAD (digitised)'); ax[0].plot((th_f + d) % 360, e, 'C0.', ms=3, label='FEMM, 2° el, Motor-CAD B-H')
ax[0].set(xlabel='electrical deg', ylabel='V @ 1000 rpm', title='Back-EMF, 20 °C', xlim=(0, 360)); ax[0].legend(fontsize=8)
xm, cm_ = np.array(mc['cogging']['theta_m']), np.array(mc['cogging']['vw'])
xc, cf = np.array(cogf['cog_theta_m']), -np.array(cogf['cogging'])
ax[1].plot(xm, cm_, 'C1', lw=2, label='Motor-CAD VW (digitised)'); ax[1].plot(xc, cf, 'C0o-', ms=3, lw=1, label='FEMM 0.2° steps, fillets')
ax[1].set(xlabel='mech deg', ylabel='N·m', title='Cogging torque', xlim=(0, 6)); ax[1].legend(fontsize=8)
img['val'] = png(fig)
val = dict(rms=float(np.sqrt(np.mean(e ** 2))), pk=float(np.abs(e).max()), f1=float(amp[1]), thd=thd(amp, 45), h11=100 * amp[11] / amp[1],
           thd30=thd(h30, 13), rms30=float(np.sqrt(np.mean(e30 ** 2))), cog=float(np.ptp(cogf['cogging'])))

ek = np.unravel_index(np.nanargmax(Mp['eff']), Mp['eff'].shape)
base = next((e['rpm'] for e in env if e['T'] < 0.98 * env[1]['T']), None)
pk_P = max(eP)
def at(n):
    return next(e for e in env if e['rpm'] == n)
env_rows = ''.join('<tr><td>%d</td><td>%.2f</td><td>%.2f</td><td>%.0f</td><td>%.0f</td><td>%.1f</td><td>%s</td><td>%.0f / %.0f / %.0f / %.0f</td><td>%s</td></tr>' % (
    e['rpm'], e['T'], e['P_kW'], e['I_line'], e['I_dc'], e['V'], '%.1f' % e['eff'] if e['eff'] else '–', e['pcu'], e['pac'], e['pfe'], e['pmech'], e['limit'].strip() or '–')
    for e in env if e['rpm'] % 1000 == 0)
rep = dict(
    PEAK_T='%.1f' % env[0]['T'], PEAK_P='%.1f' % pk_P, BASE='%d' % base, EFF='%.1f' % Mp['eff'][ek], EFF_N='%d' % S[ek[1]], EFF_T='%.1f' % T[ek[0]],
    T12='%.1f' % at(12000)['T'], P12='%.1f' % at(12000)['P_kW'], PSIM='%.2f' % (info['psi_m'] * 1e3), LD='%.0f' % info['Ld_unsat_phase_uH'], LQ='%.0f' % info['Lq_unsat_phase_uH'],
    RPH='%.3f' % info['R_phase_100C_mOhm'], BR='%.3f' % info['Br'], STM='%.2f' % info['st_mass'], ENV_ROWS=env_rows,
    V_RMS='%.3f' % val['rms'], V_PK='%.3f' % val['pk'], V_F1='%.3f' % val['f1'], V_THD='%.2f' % val['thd'], V_H11='%.2f' % val['h11'],
    V_THD30='%.2f' % val['thd30'], V_RMS30='%.3f' % val['rms30'], V_COG='%.3f' % val['cog'])
tpl = open(os.path.join(H, 'lab_template.html'), encoding='utf8').read()
for k, v in img.items():
    tpl = tpl.replace('{{IMG_%s}}' % k.upper(), v)
for k, v in rep.items():
    tpl = tpl.replace('{{%s}}' % k, v)
assert '{{' not in tpl, [l for l in tpl.split('\n') if '{{' in l][:3]
open(os.path.join(H, 'lab_report.html'), 'w', encoding='utf8').write(tpl)
print(rep['PEAK_T'], rep['PEAK_P'], rep['BASE'], rep['EFF'], val, '%d kB' % (len(tpl) // 1024))
