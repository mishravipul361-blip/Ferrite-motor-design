"""Build report/v7_motor_report.html from whatever calibrated results exist; missing analyses show as pending.
Sources: results.json (or results_partial.json mid-run), dq.json, eff.json, field_stats.json, calib.json, fig_*.png.
Run any time:  python build_report.py
"""
import base64, datetime, json, math, os, re
import numpy as np
import femm_motor as M
import dq_study as D

HERE = M.HERE
P = M.P
f1 = lambda v: '%.1f' % v
f2 = lambda v: '%.2f' % v
NA = '—'


def J(f):
    p = os.path.join(HERE, f)
    return json.load(open(p)) if os.path.exists(p) else None


def calibrated(r):
    return r is not None and r.get('inputs', {}).get('magnet_temp_C') == M.CFG['magnet_temp_C']


def main():
    r = J('results.json')
    if not calibrated(r):
        r = J('results_partial.json')
    r = r if calibrated(r) else {}
    dq, ef, fs, cal = J('dq.json'), J('eff.json'), J('field_stats.json'), J('calib.json')
    have = dict(noload='noload' in r, gamma='gamma_sweep' in r, load='load' in r,
                fields='fields' in r and bool(r['fields'].get('rated')) and bool(r['fields'].get('noload')),
                maps=ef is not None, pics=os.path.exists(os.path.join(HERE, 'fig_B_rated.png')), fstats=fs is not None)
    v, data = {}, {}

    # ---- no load
    if have['noload']:
        nl = r['noload']; th = np.array(nl['theta_m'])
        psi1 = float(np.mean(list(nl['psi1_peak'].values())))
        E1k = 2 * math.pi * 1000 / 60 * P * psi1 / math.sqrt(2)
        emf = {ph: (np.gradient(np.array(nl['psi'][ph]), np.radians(th)) * 2 * math.pi * 1000 / 60).round(3).tolist() for ph in 'RYB'}
        emf['theta_e'] = (th * P).tolist()
        cog = np.array(r['cogging']['torque'])
        data.update(emf=emf, cog=dict(theta=r['cogging']['theta_m'], T=cog.round(4).tolist()))
        v.update(E1k=f2(E1k), cog='%.3f' % (cog.max() - cog.min()))
    if have['gamma']:
        g = r['gamma_sweep']
        data['gamma'] = dict(g=g['gamma'], T=np.round(g['torque'], 2).tolist())
        v['gopt'] = f1(g['gamma_opt'])
    # ---- load
    if have['load']:
        T = np.array(r['load']['torque']); Tm = float(T.mean())
        cs = r['current_sweep']
        data['ripple'] = dict(theta=r['load']['theta_m'], T=T.round(3).tolist(), mean=round(Tm, 3))
        data['ti'] = dict(I=cs['I_line_rms'], T=np.round(cs['torque'], 2).tolist(), I_rated=M.I_LINE_RMS, T_hot=round(r['hot80']['torque'], 2))
        v.update(T_load=f1(Tm), ripple=f1((T.max() - T.min()) / Tm * 100), kt='%.3f' % (Tm / M.I_LINE_RMS),
                 T_hot=f1(r['hot80']['torque']), cog_pct=f2((cog.max() - cog.min()) / Tm * 100) if have['noload'] else NA)
    if have['fields']:
        F = r['fields']
        data['gap'] = dict(ang=np.arange(0, 360, 0.5).tolist(), nl=np.round(F['noload']['gap_Br'], 3).tolist(),
                           rated=np.round(F['rated']['gap_Br'], 3).tolist())
        hcj = M.HCJ / 1e3
        H = lambda c: F[c]['H_along_M_min'] / 1e3
        st = lambda h: ('bad', 'past HcJ') if h < -hcj else (('warn', '< 20 % margin') if h < -0.8 * hcj else ('ok', 'OK'))
        v.update(Bgap=f2(max(abs(x) for x in F['noload']['gap_Br'])), B_tooth=f2(max(F['rated']['tooth_B'])),
                 B_yoke=f2(max(F['rated']['yoke_B'])), H_min='%.0f' % H('rated'), H_nl='%.0f' % H('noload'),
                 demag_cls={'bad': 'bad', 'warn': 'warn', 'ok': ''}[st(H('rated'))[0]],
                 demag_pill=st(H('rated'))[0], demag_txt=st(H('rated'))[1])
        if F.get('demag_gamma90'):
            v.update(H_90='%.0f' % H('demag_gamma90'), demag90_pill=st(H('demag_gamma90'))[0], demag90_txt=st(H('demag_gamma90'))[1])
        else:   # FEMM did not converge for full -d current
            v.update(H_90=NA, demag90_pill='warn', demag90_txt='solver did not converge')
    if have['fstats']:
        v.update(demag_area=f1(fs['demag_rated']['area_beyond_HcJ_pct']), demag_area80=f1(fs['demag_rated']['area_beyond_80pct_HcJ_pct']),
                 B_st_rated=f2(fs['rated']['B_stator_p99']))

    # ---- d-q map, inductances, torque-speed
    ids, iqs = np.array(dq['id']), np.array(dq['iq']); pd, pq = np.array(dq['psid']), np.array(dq['psiq'])
    psim = pd[-1, 0]
    Ld0 = (pd[-2, 0] - psim) / ids[-2]; Lq0 = pq[-1, 1] / iqs[1]
    Ldr = (pd[2, 3] - psim) / ids[2]; Lqr = pq[2, 3] / iqs[3]
    ich_path = psim / Ld0
    sp = np.arange(0, 12001, 100)
    env = {'dc360': D.envelope(dq, D.I_PEAK, sp, D.P_DC_MAX), 'dc180': D.envelope(dq, D.I_PEAK / 2, sp, D.P_DC_MAX),
           'nodc': D.envelope(dq, D.I_PEAK, sp)}
    ts = {k: dict(rpm=[x['rpm'] for x in e], T=[round(x['T'], 2) for x in e], P=[round(x['P_kW'], 3) for x in e],
                  I=[round(x['I_line_rms'], 1) for x in e]) for k, e in env.items()}
    base = lambda rows: next((x['rpm'] for x in rows if x['T'] < 0.98 * rows[0]['T']), rows[-1]['rpm'])
    ts['table'] = [[int(x['rpm']), f1(x['T']), f2(x['P_kW']), f2(x['P_dc_kW']), '%.0f' % x['I_line_rms'], '%.0f' % x['Pcu_W']]
                   for x in env['dc360'] if x['rpm'] % 1000 == 0]
    data.update(dq=dict(id=dq['id'], iq=dq['iq'], psid=dq['psid'], psiq=dq['psiq']), ts=ts)
    v.update(T_peak=f1(env['nodc'][0]['T']), n_base_dc='%d' % base(env['dc360']), n_base_nodc='%d' % base(env['nodc']),
             P_max_dc=f1(max(x['P_kW'] for x in env['dc360'])), psim='%.4f' % psim,
             Ld0='%.0f' % (Ld0 * 1e6), Lq0='%.0f' % (Lq0 * 1e6), Ldr='%.0f' % (Ldr * 1e6), Lqr='%.0f' % (Lqr * 1e6),
             Ld0p='%.0f' % (Ld0 * 5e5), Lq0p='%.0f' % (Lq0 * 5e5), Ldrp='%.0f' % (Ldr * 5e5), Lqrp='%.0f' % (Lqr * 5e5),
             sal=f2(Lqr / Ldr), ich_path='%.0f' % ich_path, ich='%.0f' % (ich_path * 2 * math.sqrt(3) / math.sqrt(2)))
    if not have['noload']:
        v['E1k'] = f2(2 * math.pi * 1000 / 60 * P * psim / math.sqrt(2))

    # ---- efficiency map
    if have['maps']:
        e = np.array(ef['eff'], dtype=float); k = np.unravel_index(np.nanargmax(e), e.shape)
        data['maps'] = {key: ef[key] for key in ('eff', 'eff_sys', 'loss', 'pcu', 'pfe', 'I_rms', 'pf', 'P_dc_kW')}
        data['maps'].update(speeds=ef['speeds'], torques=ef['torques'], env_rpm=[x['rpm'] for x in ef['envelope']],
                            env_T=[round(x['T'], 2) for x in ef['envelope']])
        v.update(eff_peak=f1(e[k]), eff_peak_at='%d rpm, %.0f N·m' % (ef['speeds'][k[1]], ef['torques'][k[0]]))

    # ---- fixed inputs
    v.update(date=datetime.date.today().isoformat(), stack='%.0f' % M.STACK, I_rms='%.0f' % M.I_LINE_RMS,
             I_ph=f1(M.I_LINE_RMS / math.sqrt(3)), I_coil=f1(M.I_LINE_RMS / math.sqrt(3) / 2), Vdc='%.0f' % M.VDC,
             Pdc='%.0f' % (D.P_DC_MAX / 1e3), Vmax=f1(D.V_MAX), eta_inv='%.0f' % (D.ETA_INV * 100), HcJ='%.0f' % (M.HCJ / 1e3),
             Pcu='%.0f' % (3 * (M.I_LINE_RMS / math.sqrt(3)) ** 2 * M.R_COIL[100]),
             rotor_od='%.0f' % M.ROTOR['rotor_diameter'], shaft='%.0f' % M.ROTOR['shaft_diameter'],
             mag_depth='%.0f' % M.ROTOR['magnet_depth'], bridge='%.0f' % M.ROTOR['Bridge_thickness'],
             Br='%.3f' % M.BR, mag_temp='%.0f' % M.CFG['magnet_temp_C'], stacking='%.2f' % M.STACKING, mu='%.2f' % M.MU_MAG,
             s_stack=f1(100 * (cal['stacking 0.95'] / cal['baseline'] - 1)), s_mu=f1(100 * (cal['magnet mu_r 1.10'] / cal['baseline'] - 1)))

    # ---- verification tables
    import eff_map as E
    row = lambda *c: '<tr>' + ''.join('<td class="n">%s</td>' % x if i else '<td>%s</td>' % x for i, x in enumerate(c)) + '</tr>'
    rows = lambda pairs: ''.join(row(k, val) for k, val in pairs)
    HcB = M.BR / (M.MU0 * M.MU_MAG)
    v['tbl_magnet'] = rows([
        ('Remanence Br at 20 °C', '%.3f T' % M.BR20), ('Br temperature coefficient', '%.2f %%/K' % (M.BR_TC * 100)),
        ('Magnet temperature (calibrated)', '%.0f °C' % M.CFG['magnet_temp_C']), ('Br used', '%.4f T' % M.BR),
        ('Recoil permeability μr', '%.2f' % M.MU_MAG), ('Coercivity HcB = Br/(μ0·μr)', '%.0f kA/m' % (HcB / 1e3)),
        ('Intrinsic coercivity HcJ (demag check)', '%.0f kA/m' % (M.HCJ / 1e3)), ('Br at 80 °C (hot case)', '%.3f T' % (M.BR20 * (1 + M.BR_TC * 60))),
        ('Magnetisation', 'tangential, alternating'), ('Electrical conductivity', '0 (no eddy loss)'),
        ('Density (mass only)', '4.9 g/cm³'), ('Model', 'linear recoil line')])
    v['tbl_steel'] = rows([
        ('Thickness', '0.35 mm'), ('Stacking factor', '%.2f' % M.STACKING), ('Density', '%d kg/m³' % E.DENS),
        ('Initial relative permeability', '≈ 12 000'), ('B at 2500 / 5000 A/m', '1.53 / 1.63 T'),
        ('Loss, catalogue', '2.35 W/kg @ 1.5 T 50 Hz; ≈17 W/kg @ 1 T 400 Hz'),
        ('Hysteresis coefficient kh', '%.4f W/(kg·Hz·T²)' % E.KH), ('Eddy coefficient ke', '%.2e W/(kg·Hz²·T²)' % E.KE),
        ('Iron-loss build factor', '%.1f' % E.BUILD), ('Rotor iron loss', '+%.0f %% of stator' % (E.ROTOR_FE * 100)),
        ('Stator lamination mass', '%.2f kg' % ef['model']['stator_mass_kg'] if ef else NA), ('Same steel in rotor and stator', 'yes')])
    v['tbl_bh'] = ''.join(row('%g' % h, '%.2f' % bb, '%.0f' % (bb / (M.MU0 * h)) if h else '—') for h, bb in M.BH)
    data['bh'] = dict(H=[h for h, _ in M.BH if h], B=[bb for h, bb in M.BH if h])
    rho, alpha, Lc, Acu = 1.724e-8, 0.00393, 1.9624, math.pi * (M.WIRE_D / 2) ** 2
    Rc = lambda T: rho * (1 + alpha * (T - 20)) * Lc / (Acu * 1e-6)
    Ts = (20, 100, 150, 180)
    v['tbl_res'] = (row('Per coil = per phase (2 series × 2 parallel)', *['%.3f mΩ' % (Rc(T) * 1e3) for T in Ts])
                    + row('Line-line (delta: R·2R/3R)', *['%.3f mΩ' % (Rc(T) * 2 / 3 * 1e3) for T in Ts])
                    + row('Per parallel path (2 coils)', *['%.3f mΩ' % (2 * Rc(T) * 1e3) for T in Ts])
                    + row('Copper loss at %.0f A rms' % M.I_LINE_RMS, *['%.0f W' % (3 * (M.I_LINE_RMS / math.sqrt(3)) ** 2 * Rc(T)) for T in Ts]))
    v['tbl_copper'] = rows([
        ('Wire', 'Ø%.2f mm bare, %.3f mm² copper' % (M.WIRE_D, Acu)), ('Resistivity at 20 °C', '1.724e-8 Ω·m'),
        ('Temperature coefficient', '0.00393 /K'), ('Coil length, lead tip to lead tip', '%.1f mm (CAD)' % (Lc * 1e3)),
        ('Turns per coil / conductors per slot', '%d / %d' % (10, 20)), ('Current density at %.0f A rms' % M.I_LINE_RMS, '%.2f A/mm² rms' % (M.I_LINE_RMS / math.sqrt(3) / 2 / Acu)),
        ('Copper mass (12 coils)', '≈1.39 kg'), ('Temperature used for losses and voltage', '100 °C'),
        ('Busbar loss (not in maps)', '≈33 W at 100 °C (RESISTANCE_v7.txt)'), ('AC / proximity loss', 'not included')])
    knobs = [
        ('Magnet temperature', '%.0f °C' % M.CFG['magnet_temp_C'], '−0.18 % torque per °C', 'motor_config.json: magnet_temp_C'),
        ('Stacking factor', '%.2f' % M.STACKING, '0.97→0.95: −0.7 % torque', 'motor_config.json: stacking'),
        ('Magnet μr', '%.2f' % M.MU_MAG, '1.05→1.10: −1.3 % torque', 'motor_config.json: mu_mag'),
        ('Br at 20 °C, temp. coeff., HcJ', '%.2f T, %.2f %%/K, %.0f kA/m' % (M.BR20, M.BR_TC * 100, M.HCJ / 1e3), 'torque ∝ ~Br; demag margin', 'femm_motor.py: BR20, BR_TC, HCJ'),
        ('Steel B-H curve', 'M235-35A typical', 'saturation, peak torque', 'femm_motor.py: BH'),
        ('Rotor geometry', 'see Inputs', 'everything', 'femm_motor.py: ROTOR'),
        ('Stack length', '%.0f mm' % M.STACK, 'torque, EMF ∝ length', 'femm_motor.py: STACK'),
        ('Rated current', '%.0f A rms line' % M.I_LINE_RMS, 'torque, demag', 'femm_motor.py: I_LINE_RMS'),
        ('Coil resistance', '%.3f mΩ @100 °C' % (M.R_COIL[100] * 1e3), 'copper loss, voltage', 'femm_motor.py: R_COIL'),
        ('DC bus voltage', '%.0f V' % M.VDC, 'base speed', 'femm_motor.py: VDC'),
        ('Voltage utilisation', '95 %', 'base speed', 'dq_study.py: V_MAX'),
        ('Battery DC power limit', '%.0f kW' % (D.P_DC_MAX / 1e3), 'power above base speed', 'dq_study.py: P_DC_MAX'),
        ('Inverter efficiency', '%.0f %%' % (D.ETA_INV * 100), 'DC power, system efficiency', 'dq_study.py: ETA_INV'),
        ('Iron-loss coefficients / build factor', 'kh %.4f, ke %.1e, ×%.1f' % (E.KH, E.KE, E.BUILD), 'efficiency map', 'eff_map.py: KH, KE, BUILD'),
        ('Mechanical loss', '%.0e·n + %.1e·n² W' % (E.MECH_A, E.MECH_B), 'efficiency at speed', 'eff_map.py: MECH_A, MECH_B'),
        ('Solver mesh sizes', 'gap 0.25, rotor 1.0, stator 1.5 mm', 'accuracy vs time', 'femm_motor.py: build()'),
    ]
    v['tbl_knobs'] = ''.join('<tr><td>%s</td><td class="n">%s</td><td>%s</td><td><code>%s</code></td></tr>' % k for k in knobs)
    v['B_st_mean'] = f2(fs['rated']['B_stator_mean']) if fs and 'B_stator_mean' in fs.get('rated', {}) else NA

    # ---- status banner
    parts = [('No-load back-EMF and cogging', have['noload']), ('Current-angle sweep', have['gamma']),
             ('d-q map, inductances, torque–speed', True), ('Rated waveform, torque vs current, hot magnets', have['load']),
             ('Demagnetisation check', have['fields'] and have['fstats']), ('Flux-density maps', have['pics']),
             ('Efficiency and loss maps', have['maps'])]
    todo = [n for n, ok in parts if not ok]
    v['status_banner'] = '' if not todo else (
        '<div class="banner"><b>Interim report, %s.</b><span>Complete: %s.</span><span>Still running, added when done: %s.</span></div>'
        % (datetime.datetime.now().strftime('%d %b %H:%M'), ', '.join(n for n, ok in parts if ok), ', '.join(todo)))

    s = open(os.path.join(HERE, 'report_template.html'), encoding='utf8').read()
    s = s.replace("'Magnets 35 °C'", "'Magnets %s °C'" % v['mag_temp'])
    pend = '<div class="pending">Pending: this analysis is still running and will be added automatically.</div>'
    for key in ('B_noload', 'B_rated', 'B_noload_zoom', 'B_rated_zoom', 'demag'):
        fn = os.path.join(HERE, 'fig_%s.png' % key)
        if os.path.exists(fn):
            v['img_' + key] = 'data:image/png;base64,' + base64.b64encode(open(fn, 'rb').read()).decode()
        else:
            s = re.sub(r'<img src="\{\{img_%s\}\}"[^>]*>' % key, pend, s)
    s = re.sub(r'\{\{(\w+)\}\}', lambda m_: v.get(m_.group(1), NA), s)
    s = s.replace('/*__DATA__*/', json.dumps(data, separators=(',', ':'), default=float).replace('NaN', 'null'))
    os.makedirs(os.path.join(HERE, 'report'), exist_ok=True)
    out = os.path.join(HERE, 'report', 'v7_motor_report.html')
    open(out, 'w', encoding='utf8').write(s)
    print(out, '%.0f kB' % (len(s) / 1e3), '| pending:', todo or 'none')


if __name__ == '__main__':
    main()
