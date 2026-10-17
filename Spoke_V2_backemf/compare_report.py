"""Motor-CAD (digitised from the screen photos) vs FEMM, no load, 1000 rpm, magnets 20 C -> compare_report.html"""
import base64, json, math, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
fe = json.load(open(os.path.join(H, 'results.json')))['shaft_magnetic']
fs = json.load(open(os.path.join(H, 'field_stats.json')))
mc = json.load(open(os.path.join(H, 'mcad', 'mcad_digitised.json')))

# ---------- back-EMF ----------
th_f = 5 * np.array(fe['theta_m'])                    # electrical deg, 10 deg steps
# FEMM EMF = central difference of psi over +-10 deg el = the exact MEAN EMF over that 20 deg window.
# For a like-for-like comparison Motor-CAD's curve is averaged over the same window (e_m_avg).
e_f = np.array(fe['emf_phase_1000rpm']['Ph1'])
th_m = np.array(mc['backemf']['theta_e']); e_m = np.array(mc['backemf']['phase1'])
rms = lambda y: float(np.sqrt(np.mean(np.asarray(y) ** 2)))
fund = lambda th, y, n=1: 2 * np.mean(y * np.exp(-1j * n * np.radians(th)))
# align the angle origin: shift FEMM so its fundamental phase equals Motor-CAD's
d_el = math.degrees(np.angle(fund(th_f, e_f)) - np.angle(fund(th_m, e_m)))
d_el = (d_el + 180) % 360 - 180
th_fa = (th_f + d_el) % 360
o = np.argsort(th_fa); th_fa, e_fa = th_fa[o], e_f[o]
e_m_at_f = np.interp(th_fa, th_m, e_m, period=360)
diff = e_fa - e_m_at_f
harm = lambda th, y: [abs(fund(th, y, n)) for n in range(1, 18)]
hm, hf, hms = harm(th_m, e_m), harm(th_fa, e_fa), harm(th_fa, e_m_at_f)
box = np.ones(21) / 21                                     # +-10 deg el window, 1 deg grid
e_m_avg = np.convolve(np.r_[e_m[-10:], e_m, e_m[:10]], box, 'valid')
e_m_avg_at_f = np.interp(th_fa, th_m, e_m_avg, period=360)
diff_avg = e_fa - e_m_avg_at_f
hma = harm(th_fa, e_m_avg_at_f)
thd = lambda h: 100 * math.sqrt(sum(x ** 2 for x in h[1:])) / h[0]

# ---------- cogging: Motor-CAD curve vs FEMM samples folded into the 6 deg period ----------
xc = np.array(mc['cogging']['theta_m']); cm = np.array(mc['cogging']['vw'])
tf = np.array(fe['theta_m']); cf = np.array(fe['cogging'])
best = None
for sgn in (1, -1):
    for sh in np.arange(0, 6, 0.02):
        t = (tf + sh) % 6
        err = np.mean((sgn * cf - np.interp(t, xc, cm, period=6)) ** 2)
        if best is None or err < best[0]: best = (err, sgn, sh)
_, c_sgn, c_sh = best
cog_f = dict(theta=((tf + c_sh) % 6).tolist(), T=(c_sgn * cf).tolist())

# ---------- air-gap Br: FEMM (rotor 0, 720 bins over 360 mech = 1800 el) vs Motor-CAD (clean parts) ----------
gb = np.array(fs['gap_Br']); ga = np.arange(720) * 2.5 + 1.25             # electrical deg
xa = np.array(mc['airgap']['theta_e']); bm = np.array([np.nan if v is None else v for v in mc['airgap']['Br']])
bm[(xa > 1170) & (xa < 1345)] = np.nan   # residual moire
okm = ~np.isnan(bm)
bestg = None
for sgn in (1, -1):
    for sh in np.arange(0, 1800, 2.5):
        y = sgn * np.interp((xa[okm] - sh) % 1800, ga, gb, period=1800)
        err = np.mean((y - bm[okm]) ** 2)
        if bestg is None or err < bestg[0]: bestg = (err, sgn, sh)
_, g_sgn, g_sh = bestg
gb_al = g_sgn * np.interp((xa - g_sh) % 1800, ga, gb, period=1800)
gap_rms_err = float(np.sqrt(bestg[0]))

# ---------- scalar table (Motor-CAD values typed from the screens) ----------
w = 2 * math.pi * 1000 / 60
S = fe['summary']
rows = [  # name, Motor-CAD, FEMM, unit, note
    ('Back-EMF phase rms (= line, delta)', 8.213, S['E_phase_rms'], 'V', 'Output Data â€º Drive'),
    ('Back-EMF phase peak', 11.51, S['E_phase_peak'], 'V', 'FEMM peak limited by 10Â° el sampling'),
    ('Back-EMF fundamental peak', 11.60, S['E_phase_fund_peak'], 'V', ''),
    ('Ke (peak / Ï‰ mech)', 0.1099, S['E_phase_peak'] / w, 'VÂ·s/rad', ''),
    ('Ke fundamental', 0.1107, S['E_phase_fund_peak'] / w, 'VÂ·s/rad', ''),
    ('Flux linkage, d-axis (open circuit)', 22.0493, 1000 * S['psi_path_fund_peak'], 'mWb', 'FEMM: fundamental, per phase'),
    ('Back-EMF THD', 5.419, S['THD_phase'] * 100, '%', 'FEMM coarse sampling, see Â§2'),
    ('Air-gap B maximum', 0.8338, fs['gap_Br_peak'], 'T', 'FEMM: |Br| at mid-gap, rotor 0Â°'),
    ('Stator tooth B maximum', 1.161, fs['tooth_B_max'], 'T', 'FEMM: 99th percentile of tooth body'),
    ('Stator back-iron B maximum', 1.003, fs['yoke_B_max'], 'T', 'FEMM: 99th percentile of yoke'),
    ('Cogging torque pk-pk (VW)', 0.66303, S['cogging_pkpk'], 'NÂ·m', 'FEMM: 2Â° steps, aliased'),
    ('Phase resistance, 20 Â°C', 5.098, 5.0978, 'mÎ©', ''),
    ('Line-line resistance, 20 Â°C (delta)', 3.399, 3.3986, 'mÎ©', ''),
    ('Armature MLT', 201.6, 201.615, 'mm', ''),
    ('Turns per phase', 20, 20, '', ''),
    ('Fundamental winding factor', 0.933, 0.933, '', ''),
    ('Conductor CSA', 6.818, math.pi / 4 * 2.9464 ** 2, 'mmÂ²', ''),
    ('Stacking factor (stator / rotor)', 0.97, 0.97, '', ''),
    ('Electrical frequency @ 1000 rpm', 83.33, 83.33, 'Hz', ''),
    ('Cogging period', 6, 6, 'mech deg', ''),
]
row_html = ''
for n, a, b, u, note in rows:
    d = 100 * (b - a) / a
    cls = 'ok' if abs(d) <= 2 else ('mid' if abs(d) <= 6 else 'bad')
    fmt = (lambda v: '%.4f' % v) if a < 1 else (lambda v: '%.3f' % v if a < 100 else '%.1f' % v)
    row_html += '<tr><td class="txt">%s</td><td>%s</td><td>%s</td><td>%s</td><td><span class="pill %s">%+.1f %%</span></td><td class="txt src">%s</td></tr>' % (
        n, fmt(a), fmt(b), u, cls, d, note)

dig = dict(rms=rms(e_m), pk=float(np.max(np.abs(e_m))), f1=abs(fund(th_m, e_m)), thd=thd(hm))
data = dict(th_m=th_m.tolist(), e_m=e_m.tolist(), e_m_avg=e_m_avg.tolist(), th_f=th_fa.tolist(), e_f=e_fa.tolist(),
            diff=diff_avg.tolist(), hm=[100 * x / hm[0] for x in hm], hma=[100 * x / hma[0] for x in hma],
            hf=[100 * x / hf[0] for x in hf],
            cog_x=xc.tolist(), cog_m=cm.tolist(), cog_f=cog_f,
            gap_x=xa.tolist(), gap_m=[None if np.isnan(v) else float(v) for v in bm], gap_f=gb_al.tolist())
rep = {
    '__DATA__': json.dumps(data), '__ROWS__': row_html,
    '__D_EL__': '%.1f' % d_el, '__RMS_M__': '%.3f' % dig['rms'], '__PK_M__': '%.3f' % dig['pk'], '__F1_M__': '%.3f' % dig['f1'],
    '__THD_M__': '%.2f' % dig['thd'], '__THD_MA__': '%.2f' % thd(hma), '__THD_F__': '%.2f' % thd(hf),
    '__DIFF_RMS__': '%.3f' % rms(diff_avg), '__DIFF_MAX__': '%.3f' % float(np.max(np.abs(diff_avg))),
    '__RMS_F__': '%.3f' % S['E_phase_rms'], '__F1_F__': '%.3f' % S['E_phase_fund_peak'],
    '__C_SH__': '%.2f' % c_sh, '__G_SH__': '%.1f' % g_sh, '__G_ERR__': '%.3f' % gap_rms_err,
    '__COG_F__': '%.3f' % S['cogging_pkpk'],
}
tpl = open(os.path.join(H, 'compare_template.html'), encoding='utf8').read()
for k, v in rep.items():
    tpl = tpl.replace(k, v)
assert '__' not in tpl, [l for l in tpl.split('\n') if '__' in l][:3]
open(os.path.join(H, 'compare_report.html'), 'w', encoding='utf8').write(tpl)
print('shift el %.1f | MCAD dig rms %.3f pk %.3f f1 %.3f thd %.2f | windowed: MCAD thd %.2f FEMM thd %.2f | diff rms %.3f max %.3f'
      % (d_el, dig['rms'], dig['pk'], dig['f1'], dig['thd'], thd(hma), thd(hf), rms(diff_avg), np.max(np.abs(diff_avg))))
print('cog shift %.2f sign %d | gap shift %.1f sign %d rms err %.3f' % (c_sh, c_sgn, g_sh, g_sgn, gap_rms_err))
print('harm MA', np.round([100 * x / hma[0] for x in hma[:13]], 2)); print('harm F ', np.round([100 * x / hf[0] for x in hf[:13]], 2))