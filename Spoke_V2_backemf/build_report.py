"""Builds report.html from results.json, field_stats.json and the flux-map PNGs."""
import base64, json, math, os
import numpy as np
import backemf as M

H = M.HERE
res = json.load(open(os.path.join(H, 'results.json')))
fs = json.load(open(os.path.join(H, 'field_stats.json')))
r = res['shaft_magnetic']; rn = res['shaft_nonmagnetic']
S, SN = r['summary'], rn['summary']
img = lambda f: 'data:image/png;base64,' + base64.b64encode(open(os.path.join(H, f), 'rb').read()).decode()

# ---- winding resistance (Motor-CAD wire SWG 11, EWdg MLT) ----
D_BARE, D_INS = 2.9464, 3.0302
A_W = math.pi / 4 * D_BARE ** 2                       # mm2
MLT = 2 * M.STACK + 75.615                            # mm per turn: 2 x active + end-winding MLT (front+rear)
L_COIL = M.TURNS * MLT                                # mm
RHO, ALPHA = 1.724e-8, 0.003862
r_coil = lambda T: RHO * (1 + ALPHA * (T - 20)) * L_COIL * 1e-3 / (A_W * 1e-6)
R = [dict(T=T, coil=r_coil(T) * 1e3, phase=r_coil(T) * 2 / M.PATHS * 1e3, ll_star=2 * r_coil(T) * 2 / M.PATHS * 1e3,
          ll_delta=2 / 3 * r_coil(T) * 2 / M.PATHS * 1e3) for T in (20, 75, 100, 120, 150)]
cu_mass = 12 * M.TURNS * MLT * 1e-3 * A_W * 1e-6 * 8933

g = M.rotor_geometry(M.ROTOR); sg = M.slot_geometry(M.STATOR)
pt = lambda p: '(%.2f, %.2f)' % p
data = dict(
    theta_e=[5 * t for t in r['theta_m']],
    emf={k: v for k, v in r['emf_phase_1000rpm'].items()},
    emf_nm=rn['emf_phase_1000rpm']['Ph1'],
    psi=r['path_psi'], cog=r['cogging'], theta_m=r['theta_m'],
    gap=fs['gap_Br'],
    coil_psi=r['coil_psi'])
w1000 = 2 * math.pi * 1000 / 60
ke = S['E_phase_rms'] / w1000
vals = dict(
    E_ph_rms=S['E_phase_rms'], E_ph_pk=S['E_phase_peak'], E_ph_f_pk=S['E_phase_fund_peak'], E_ph_f_rms=S['E_phase_fund_rms'],
    E_ll_rms=S['E_line_rms'], E_ll_pk=S['E_line_peak'], E_ll_f_rms=S['E_line_fund_rms'],
    thd_ph=100 * S['THD_phase'], thd_ll=100 * S['THD_line'], psi=S['psi_path_fund_peak'], cog=S['cogging_pkpk'],
    ke=ke, ke_ll=S['E_line_rms'] / w1000, kt=3 * ke, E_nm=SN['E_phase_rms'],
    dE=100 * (SN['E_phase_rms'] / S['E_phase_rms'] - 1), cog_nm=SN['cogging_pkpk'])
f2 = lambda v: '%.2f' % v
f3 = lambda v: '%.3f' % v
f4 = lambda v: '%.4f' % v

harm_rows = ''.join('<tr><td>%d</td><td>%.2f</td><td>%.3f</td></tr>' % (n, S['harmonics_phase_pct'][str(n)],
                    S['harmonics_phase_pct'][str(n)] / 100 * S['E_phase_fund_peak']) for n in (3, 5, 7, 9, 11))
res_rows = ''.join('<tr><td>%d</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.3f</td></tr>' %
                   (x['T'], x['coil'], x['phase'], x['ll_star'], x['ll_delta']) for x in R)
bh_rows = ''.join('<tr><td>%g</td><td>%.2f</td></tr>' % (h, b) for h, b in M.BH_LAM)
bh_sh = ''.join('<tr><td>%g</td><td>%.2f</td></tr>' % (h, b) for h, b in M.BH_SHAFT)
mag_rows = ''.join('<tr><td>%d</td><td>%.4f</td><td>%.1f</td><td>%.2f</td></tr>' %
                   (T, M.BR20 * (1 + M.BR_TC * (T - 20)), M.BR20 * (1 + M.BR_TC * (T - 20)) / (M.MU0 * M.MU_MAG) / 1e3,
                    S['E_phase_rms'] * (1 + M.BR_TC * (T - 20))) for T in (20, 60, 80, 100, 120))

tpl = open(os.path.join(H, 'report_template.html'), encoding='utf8').read()
rep = {
    '__DATA__': json.dumps(data),
    '__IMG_FULL__': img('fluxmap_full.png'), '__IMG_ZOOM__': img('fluxmap_zoom.png'),
    '__HARM_ROWS__': harm_rows, '__RES_ROWS__': res_rows, '__BH_ROWS__': bh_rows, '__BH_SHAFT__': bh_sh, '__MAG_ROWS__': mag_rows,
    '__A_W__': f3(A_W), '__D_BARE__': '%.4f' % D_BARE, '__MLT__': f2(MLT), '__L_COIL__': '%.1f' % L_COIL, '__CU_MASS__': f3(cu_mass),
    '__P_P__': pt(g['P']), '__P_Q__': pt(g['Q']), '__P_R__': pt(g['R']), '__P_D__': pt(g['D']), '__P_C__': pt(g['C']), '__P_G__': pt(g['G']),
    '__P_PM__': pt(g['Pm']), '__P_QM__': pt(g['Qm']), '__P_RM__': pt(g['Rm']),
    '__S_A__': pt(sg['A']), '__S_B__': pt(sg['B']), '__S_C__': pt(sg['C']), '__S_E__': pt(sg['E']),
    '__MAG_AREA__': '%.1f' % fs['magnet_area_mm2_per_pole'],
    '__MAG_MASS__': f3(fs['magnet_area_mm2_per_pole'] * 10 * M.STACK * 1e-9 * 4800),
    '__TOOTH_B__': f2(fs['tooth_B_mean']), '__TOOTH_BMAX__': f2(fs['tooth_B_max']),
    '__YOKE_B__': f2(fs['yoke_B_mean']), '__YOKE_BMAX__': f2(fs['yoke_B_max']),
    '__BRIDGE_B__': f2(fs['bridge_B_max']), '__MAG_BM__': f3(fs['magnet_B_along_M_mean']), '__MAG_BMIN__': f3(fs['magnet_B_along_M_min']),
    '__MAG_HMIN__': '%.0f' % fs['magnet_H_min_kA_m'], '__GAP_B1__': f3(fs['gap_Br_fund_peak']), '__GAP_BPK__': f3(fs['gap_Br_peak']),
}
for k, v in vals.items():
    rep['__V_%s__' % k.upper()] = f4(v) if k in ('psi', 'ke', 'ke_ll', 'kt') else f2(v)
for k, v in rep.items():
    tpl = tpl.replace(k, v)
assert '__' not in tpl.replace('__proto__', ''), [l for l in tpl.split('\n') if '__' in l][:3]
open(os.path.join(H, 'report.html'), 'w', encoding='utf8').write(tpl)
print('report.html', len(tpl) // 1024, 'kB'); print({k: round(v, 4) for k, v in vals.items()}); print(R)

