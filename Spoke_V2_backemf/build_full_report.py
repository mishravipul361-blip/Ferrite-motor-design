"""Full aluminium-winding report: report_full/index.html (+ fig/).  python build_full_report.py"""
import json, math, os
import alu_femm as AF

H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, 'report_full')
D = json.load(open(os.path.join(R, 'report_data.json'), encoding='utf-8'))
DC = json.load(open(os.path.join(H, 'demag_check.json'), encoding='utf-8'))['results']
MS = json.load(open(os.path.join(R, 'maps_stats.json'), encoding='utf-8')) if os.path.exists(os.path.join(R, 'maps_stats.json')) else {}
ROWS = {r['design']: r for r in D['femm_grid']['rows']}
FEAS = D['feasible']
KR, OPL, COST, CYC = D['key_records'], D['op_losses'], D['cost'], D['cycle']
RIP = D['ripple']
CU = 'Copper today'


def f(x, n=2, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '–'
    s = ('%+.' + str(n) + 'f') % x if sign else ('%.' + str(n) + 'f') % x
    return s.replace('-', '−')


def table(head, rows, cls='', note=None):
    h = ''.join('<th>%s</th>' % c for c in head)
    b = ''.join('<tr>%s</tr>' % ''.join('<td>%s</td>' % c for c in r) for r in rows)
    n = '<p class="tnote">%s</p>' % note if note else ''
    return '<div class="tw"><table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>%s' % (cls, h, b, n)


def fig(src, cap, wide=True):
    return '<figure class="%s"><img src="fig/%s" alt="%s" loading="lazy"><figcaption>%s</figcaption></figure>' % ('wide' if wide else '', src, cap.replace('"', ''), cap)


def geo(d):
    g = dict(AF.DESIGNS[d])
    return dict(od=g.get('od', 160.0), depth=g['slot_depth'], tooth=g['tooth_w'], open=g.get('slot_open', 5.0), tip=g.get('tip_depth', 0.8),
                inset=g.get('magnet_inset', 2.0), mopen=g.get('magnet_opening', 8.5))


def geo_rows(ds):
    out = []
    for d in ds:
        g, r = geo(d), ROWS[d]
        t0 = ROWS['S0']['T_peak']
        out.append([d, f(g['od'], 0), f(g['depth'], 1), f(g['tooth'], 1), f(g['open'], 1), f(g['tip'], 1), f(g['inset'], 1), f(g['mopen'], 1),
                    f(r['slot_area'], 0), f(r['stator_mass_63'], 2), f(r['T_peak'], 2), f(100 * (r['T_peak'] / t0 - 1), 1, True), f(r['tooth_B99'], 2), f(r['yoke_B99'], 2),
                    f(r['st_Sa_nl'] / ROWS['S0']['st_Sa_nl'], 2)])
    return table(['Stator', 'OD', 'Slot depth', 'Tooth', 'Slot opening', 'Tip depth', 'Magnet inset', 'Magnet opening', 'Slot area mm²', 'Stator steel kg (63 mm)',
                  'EM torque N·m', 'vs copper geometry', 'Tooth B 99 %', 'Yoke B 99 %', 'No-load iron index'], out, 'num',
                 'FEMM, 63 mm, same ampere-turns as the copper winding at 360 A rms (path peak id −49 A, iq 147 A). Lengths in mm. '
                 'Iron index = stator Σ mass·B^α at no load relative to the copper stator (hysteresis-weighted), a proxy for the no-load iron loss.')


def feas_table(ods, n=14):
    pts = sorted([p for p in FEAS if p['od'] in ods], key=lambda p: -p['wm'])[:n]
    rows = []
    for p in pts:
        rows.append([p['label'].replace('Al ', ''), f(p['L'], 1), str(p['Nph'] // 2), p['lam'] or 'M235-35A', f(p['m'], 2), f(p['c'], 1), f(p['t2000'], 3), f(p['env'], 3),
                     f(p['op1'], 2, True), f(p['op2d'], 2, True), f(p['op3'], 2, True), f(p['op2'], 0), f(p['J'], 1), f(p['te'], 2, True), f(p['wm'], 2, True)])
    return table(['Design (stator, stack, turns/phase, strands)', 'Stack mm', 'Turns/coil', 'Steel', 'Mass kg', 'Cost $', 'T 2000 rpm ratio', 'Envelope min',
                  'ΔEff OP1', 'ΔEff OP2', 'ΔEff OP3', 'OP2 loss W', 'J A/mm²', 'Range tele %', 'Range WMTC %'], rows, 'num',
                 'Designs that meet the hard limits (torque at 2000 rpm ≥ copper, torque-speed limit ≥ 98.5 % of copper, OP1 and OP2 efficiency ≥ copper), '
                 'sorted by WMTC Class 1 range. Copper today: 7.76 kg, $51.6, OP2 loss 682 W, J 15.2 A/mm².')


def key_table(names):
    rows = []
    for nm in names:
        k, o, c, y = KR[nm], OPL[nm], COST[nm], CYC[nm]
        ns, d = k['ns'] or 1, k['d'] or 2.9464
        rows.append([nm, f(k['L'], 1), str((k['Nph'] or 20) // 2), '%d × %.3f' % (ns, d), f(k['fill'], 3), f(1e3 * k['R_path'], 2), f(c['masses']['total'], 2), f(c['total'], 1),
                     f(k['env_ratio_at'].get('2000', 1), 3), f(k['env_ratio_min'], 3), f(o['OP1']['eff'], 2), f(o['OP2']['eff'], 2), f(o['OP3']['eff'], 2),
                     f(o['OP2']['loss'], 0), f(o['OP3']['loss'], 0), f(y['Wh_km']['Telematics speed trace'], 2), f(y['Wh_km']['WMTC Class 1'], 2)])
    return table(['Design', 'Stack mm', 'Turns/coil', 'Strands × mm', 'Wire fill', 'R per path mΩ (hot)', 'Mass kg', 'Cost $', 'T 2000 rpm ratio', 'Envelope min',
                  'Eff OP1 %', 'Eff OP2 %', 'Eff OP3 %', 'OP2 loss W', 'OP3 loss W', 'Wh/km tele', 'Wh/km WMTC'], rows, 'num',
                 'Model results (tuned FEMM-based model, winding at its hot temperature, magnets 100 °C). Efficiencies are this model\'s values; '
                 'Motor-CAD gives copper 90.71 / 82.39 / 90.03 % at OP1 / OP2 / OP3.')


def mc_sheet(nm):
    k = KR[nm]; sv = k['sv']; g = geo(sv)
    ns, d = k['ns'] or 1, k['d'] or 2.9464
    metal = 'Copper' if nm == CU else 'Aluminium EC-1350 (61 % IACS)'
    rows = [['Stator outer diameter', f(g['od'], 1) + ' mm'], ['Stator bore', '107.0 mm'], ['Slot depth', f(g['depth'], 1) + ' mm'], ['Tooth width', f(g['tooth'], 1) + ' mm'],
            ['Slot opening', f(g['open'], 1) + ' mm'], ['Tooth tip depth', f(g['tip'], 1) + ' mm'], ['Tooth tip angle', '20°'], ['Slot corner radius', '1.95 mm'],
            ['Rotor diameter', '106.0 mm'], ['Shaft diameter', '25.0 mm (unchanged)'], ['Magnet depth', '28.0 mm (unchanged)'], ['Bridge thickness / offset', '1.0 / 4.0 mm (unchanged)'],
            ['Magnet inset', f(g['inset'], 1) + ' mm'], ['Magnet opening', f(g['mopen'], 1) + ' mm'], ['Magnet X / Y / Z', '18.0 / 10.5 / 6.0 mm (unchanged)'],
            ['Stack length (stator and rotor)', f(k['L'], 1) + ' mm'], ['Steel', k.get('lam') or 'M235-35A'], ['Winding', 'Delta, 2 parallel paths, 12 coils (unchanged layout)'],
            ['Turns per coil', str((k['Nph'] or 20) // 2)], ['Conductor', '%s, %d strands in hand × %.3f mm bare' % (metal, ns, d)],
            ['Wire slot fill (insulated)', f(k['fill'], 3)], ['Mean turn length', f(1e3 * k['mlt'], 1) + ' mm']]
    return table(['Motor-CAD input: ' + nm, 'Value'], rows, 'sheet')


def demag_rows(ds):
    out = []
    for dn in ds:
        s = MS.get(dn)
        if not s and ('ALU_%s | 360 A on -d axis' % dn) in DC:
            x, y = DC['ALU_%s | 360 A on -d axis' % dn], DC['ALU_%s | peak point 360 A, gamma 23 deg' % dn]
            out.append([dn + ' *', f(x['B_p1'], 3), f(100 * x['share_below_0p1'], 2), '–', f(x['Hmin_kA_m'], 0), f(y['B_p1'], 3), f(100 * y['share_below_0p1'], 2)]); continue
        if not s:
            out.append([dn] + ['solving'] * 6); continue
        w = lambda c, k: min(s[c]['0'][k], s[c]['3'][k])
        mx = lambda c, k: max(s[c]['0'][k], s[c]['3'][k])
        out.append([dn, f(w('demag_d', 'B_p1'), 3), f(100 * mx('demag_d', 'share_below_0p1'), 2), f(100 * mx('demag_d', 'share_below_0'), 2), f(w('demag_d', 'Hmin_kA_m'), 0),
                    f(w('demag_pk', 'B_p1'), 3), f(100 * mx('demag_pk', 'share_below_0p1'), 2)])
    return table(['Geometry', '−d axis: B at 1 % of magnet area (T)', '−d axis: area below 0.1 T (%)', '−d axis: area below 0 T (%)', '−d axis: most negative H (kA/m)',
                  'Peak point: B at 1 % area (T)', 'Peak point: area below 0.1 T (%)'], out, 'num',
                 'Magnets at −20 °C (Br %.3f T), 360 A rms with the copper ampere-turns (10 turns/coil). Worst of rotor positions 0° and 3°. '
                 'Most negative H sits in single corner elements and is mesh-dependent; compare designs on the area measures.' %(0.46 * (1 - 0.001797 * 40)))


def flux_block(ds):
    return ''.join(fig('flux_%s.jpg' % dn, '%s: flux density and flux lines at no load and at the 360 A peak point (stator %s mm OD, slot %s × %s mm, slot opening %s mm, magnet inset %s mm, opening %s mm).' % (
        dn, f(geo(dn)['od'], 0), f(geo(dn)['depth'], 1), f(geo(dn)['tooth'], 1), f(geo(dn)['open'], 1), f(geo(dn)['inset'], 1), f(geo(dn)['mopen'], 1))) for dn in ds
        if os.path.exists(os.path.join(R, 'fig', 'flux_%s.jpg' % dn)))


def ripple_table(vs):
    rows = []
    for v in vs:
        x = RIP.get(v)
        if not x:
            continue
        s0, s3 = x['verify']['skew']['none'], x['verify']['skew']['3 deg 3step']
        rows.append([x['name'], f(x['T360'], 2), f(x['gamma'], 1), f(x['thd'], 2), f(s0['OP1'], 1), f(s0['OP2'], 1), f(s0['OP3'], 1), f(s3['thd'], 2), f(s3['OP1'], 1), f(s3['OP2'], 1), f(s3['OP3'], 1), f(s3['k1'], 3)])
    return table(['Geometry', 'T at 360 A N·m', 'Best γ °el', 'THD %', 'Ripple OP1 %', 'Ripple OP2 %', 'Ripple OP3 %', 'THD 3° skew', 'OP1 3° skew', 'OP2 3° skew', 'OP3 3° skew', 'Torque factor'],
                 rows, 'num', 'FEMM at 63 mm with the copper path currents (10 turns/coil). Ripple is peak-to-peak / mean over one 12° slot pitch. '
                 '3° skew = three axial rotor segments stepped 0 / 1.5 / 3° mechanical. Limits: THD < 5 %, ripple < 10 %.')


def ops_table(names):
    rows = []
    for nm in names:
        for op in ('OP1', 'OP2', 'OP3'):
            o = OPL[nm][op]
            rows.append([nm if op == 'OP1' else '', op, f(o['Tsh'], 2), f(o['I_line'], 1), f(o['pcu'], 0), f(o['pac'], 0), f(o['pfe'], 0), f(o['pme'], 0), f(o['loss'], 0), f(o['eff'], 2)])
    return table(['Design', 'Point', 'Shaft torque N·m', 'Line current A rms', 'Winding DC W', 'Winding AC W', 'Iron W', 'Mechanical W', 'Total W', 'Efficiency %'], rows, 'num')


S = {}
best = lambda o: sorted([p for p in FEAS if p['od'] == o], key=lambda p: -p['wm'])
n160, n165, n167 = (len([p for p in FEAS if p['od'] == o]) for o in (160.0, 165.0, 167.0))

S['summary'] = ('Summary', f'''
<p class="lede">Aluminium can replace the copper winding of the Spoke V2 motor without touching the rotor structure. The best result needs the stator OD at 167 mm. At the copper OD of 160 mm it costs either range or peak-point efficiency.</p>
<div class="cards">
 <div class="card hi"><h4>OD 167 mm · Q1 · 63 mm stack</h4><p class="big">7.24 kg</p><p>0.52 kg lighter than copper, $61.0 material (+$9.4). Torque at 2000 rpm, 360 A equals copper (FEMM 45.87 vs 45.73 N·m). Loss at 9339 rpm, 3.24 N·m: 483 W vs 682 W. Range +0.5 % (telematics) / +0.3 % (WMTC Class 1). Efficiency at 2000 rpm, 44 N·m: −2.4 points.</p></div>
 <div class="card"><h4>OD 165 mm · Q4 · 70 mm stack</h4><p class="big">7.87 kg</p><p>+0.11 kg, $66.2. Range +0.2 % / +0.1 %. OP3 −1.2 points. At 63 mm the OD 165 stator falls just short of the torque-speed limit (0.984 against 0.985).</p></div>
 <div class="card"><h4>OD 160 mm · R1 · 70 mm stack</h4><p class="big">7.36 kg</p><p>0.40 kg lighter, $62.5. Range −0.1 % / −0.3 %. OP3 −2.2 points. Matching copper at OP3 needs 81 mm (R3, 8.27 kg, $72.5) and costs 0.5–0.7 % range.</p></div>
</div>
<h3>What every option shares</h3>
<ul>
<li>Rotor near the shaft unchanged: shaft 25 mm, webs 1.0 mm at 4.0 mm offset, magnet depth 28 mm. The near-air-gap change (magnet inset 2.0 → 1.5 mm, opening 8.5 → 8.0 mm) lowers the centrifugal load on the webs by 0.12 %.</li>
<li>0.20 mm electrical steel. With M235-35A, aluminium designs still meet the torque and OP1/OP2 loss limits and cost about $16–20 less than copper, but lose 1.3–2.2 % range.</li>
<li>Ripple: the rotor change cuts OP2 ripple from 87 % to about 20 %. Getting every operating point below 10 % needs a 3° three-step rotor skew (copper needs it as well). THD is 4.5–4.6 % without skew.</li>
<li>Demagnetisation: with magnets at −20 °C and full 360 A on the −d axis, 1.1–1.3 % of the magnet area falls below 0.1 T, against 0.9–1.0 % for copper. Designs with 8–9 turns per coil see 10–20 % less field at 360 A.</li>
<li>Current density at 360 A: 10.5–13.6 A/mm² (copper 15.2, limit 16).</li>
</ul>
{key_table(list(KR))}
{fig('tradeoff_scatter.png', 'Every design that meets the hard limits, by OD: mass against WMTC range, mass against OP3 efficiency, cost against telematics range. Star = copper today.')}
''')

S['method'] = ('What was simulated', f'''
<p>The study chains FEMM 4.2 finite-element solutions into a tuned motor model, then into vehicle drive cycles. Every number in this report comes from one of the stages below.</p>
{table(['Stage', 'Tool / script', 'What it computes', 'Count'], [
 ['1. Geometry grids', 'alu_femm.py (FEMM)', 'Torque, d/q flux, stator iron-loss indices, winding-zone field, tooth and yoke flux density on a 4 × 4 id/iq grid at two rotor angles, per stator/rotor geometry', '17 geometries × 32 = 544 solves'],
 ['2. Winding design sweep', 'alu_study.py + alu_sweep.py', 'Each geometry × stack 63–99 mm × 10/9/8/7 turns per coil × 1–4 strands (largest wire at copper\'s fill) × steel grade: efficiency map, torque-speed limit, OP1/OP2/OP3, masses', '2,592 designs (phase 2) + 1,756 (phase 1)'],
 ['3. Drive cycle and cost', 'alu_v2_eval.py (tele_range method)', 'Wh/km on the telematics speed trace and WMTC Class 1, current density at 360 A, material cost', '%d designs scored' % (217 + 189)],
 ['4. FEMM verification', 'sens.py ALU mode + alu_verify.py', 'Peak torque vs current angle, back-EMF THD (15-point no-load waveform), torque ripple at OP1/OP2/OP3 (12 positions each), OP2 field snapshot, rotor skew', '6 geometries × 61 solves'],
 ['5. Flux and demagnetisation maps', 'alu_maps.py + alu_maps_render.py', 'No-load and peak-point flux maps; cold demagnetisation on the −d axis and at the peak point, two rotor angles', '17 geometries × 7 = 119 solves'],
 ['6. Structural screen', 'rotor_struct.py', 'Centrifugal load carried by the shaft-side webs (Σ ρ·A·r of pole steel and magnet outside the web root)', '5 rotor variants'],
], 'plain')}
<h3>Model chain and scaling</h3>
<p>The tuned model (lab_model.py, v2 losses) reproduces the Motor-CAD copper design: torque 45.73 N·m at 360 A in FEMM, THD 5.31 % (Motor-CAD 5.42 %). An aluminium design is evaluated at the baseline-equivalent current i₀ = s·i, with s = Nph / 20 and kL = L / 63: torque = kL·T₀(i₀), flux = s·kL·ψ₀(i₀), iron loss = kL·P₀(i₀). Each geometry adds its FEMM difference to the copper geometry at the same ampere-turns. Winding resistance uses ρ at the hot winding temperature, the turn length 2L + end turn (end turn corrected for tooth width and for the mean coil thickness, π·Δt), and the Motor-CAD hybrid AC-loss formula with the strand diameter.</p>
<h3>Hard limits used to accept a design</h3>
{table(['Limit', 'Value', 'Source'], [
 ['Stator OD', '≤ 167 mm (165 preferred)', 'User'], ['Rotor near the shaft', 'Unchanged: shaft 25, web 1.0 mm, offset 4.0 mm, magnet depth 28 mm', 'User (structural)'],
 ['Torque at 2000 rpm, 360 A', '≥ copper', 'User objective'], ['Torque-speed limit', '≥ 98.5 % of copper at every speed', 'Study rule'],
 ['Losses OP1 (9000 rpm, max power) and OP2 (9339 rpm, 3.24 N·m)', '≤ copper; OP2 minimised', 'User objective'],
 ['Current density at 360 A', '< 16 A/mm²', 'User'], ['THD', '< 5 %', 'User'], ['Torque ripple', '< 10 % at every point', 'User'],
 ['Drive cycle', 'Wh/km close to copper', 'User'],
], 'plain')}
<h3>Geometries</h3>
{geo_rows(list(ROWS))}
{fig('femm_grid.png', 'FEMM comparison of all 17 geometries: torque near the peak point, slot area, and stator flux density. Blue = OD 160, orange = OD 165, green = OD 167.')}
''')

S['copper'] = ('Copper baseline', f'''
<p>The reference is the Motor-CAD design: stator OD 160 mm, bore 107 mm, slot 17.5 mm deep, tooth 15.5 mm, slot opening 5.0 mm, 63 mm stack, delta winding with 2 parallel paths and 10 turns per coil of SWG 11 copper, M235-35A steel, 52 V, 360 A rms.</p>
{mc_sheet(CU)}
{ops_table([CU])}
{fig('effmap_copper.png', 'Copper today: efficiency map from the tuned model, with the three operating points.', False)}
{flux_block(['S0'])}
<p>Motor-CAD values for reference: OP1 90.71 %, OP2 82.39 %, OP3 90.03 %, peak efficiency 95.31 %, active mass 7.76 kg, material cost $51.57 at the assumed prices.</p>
''')

S['od160'] = ('OD 160 mm', f'''
<p class="lede">Same OD as copper. The slot can only grow inward toward a thinner yoke, so aluminium needs a longer stack to keep peak torque and efficiency. {n160} designs meet the hard limits.</p>
<h3>Stator and rotor geometries at OD 160</h3>
{geo_rows(['S0', 'S1', 'S2', 'S3', 'S4', 'R0', 'R1', 'R2', 'R3'])}
<p>S1–S4 keep the copper rotor and slot tip. R0–R3 add the near-air-gap rotor change (inset 1.5, opening 8.0) and a 4.0 mm slot opening. Deeper slots cost torque per ampere-turn because the thinner yoke saturates: S2–S4 lose 5–13 % and R2–R3 lose 4–5 % against the copper geometry (yoke B 99 % rises from 1.59 T to 1.76–1.80 T). The rotor change adds 0.7–0.9 % on each stator.</p>
<h3>Best designs at OD 160</h3>
{feas_table([160.0])}
{key_table([CU, 'OD160 R1 70 mm', 'OD160 R3 81 mm'])}
{ops_table(['OD160 R1 70 mm', 'OD160 R3 81 mm'])}
<div class="pair">{fig('effdelta_1.png', 'R1, 70 mm: efficiency change against copper.', False)}{fig('effdelta_2.png', 'R3, 81 mm: efficiency change against copper.', False)}</div>
{mc_sheet('OD160 R1 70 mm')}
{mc_sheet('OD160 R3 81 mm')}
<h3>Ripple, THD and demagnetisation at OD 160</h3>
{ripple_table(['ALU_S0', 'ALU_R0', 'ALU_R1'])}
{demag_rows(['S0', 'S1', 'S2', 'S3', 'S4', 'R0', 'R1', 'R2', 'R3'])}
<h3>Flux maps, OD 160</h3>
{flux_block(['S1', 'S2', 'S3', 'S4', 'R0', 'R1', 'R2', 'R3'])}
''')

S['od165'] = ('OD 165 mm', f'''
<p class="lede">A 165 mm OD adds 2.5 mm of radial room, used for a 20 mm slot with the copper tooth and a 9 mm yoke. {n165} designs meet the hard limits; all need a 66–72 mm stack.</p>
{geo_rows(['P1', 'Q4'])}
<p>P1 keeps the copper rotor and tip; Q4 adds the rotor change and 4.0 mm slot opening. At 63 mm and 10 turns per coil, Q4 reaches 98.4 % of copper's torque-speed limit around 3000 rpm, just under the 98.5 % rule, so the 63 mm option is excluded. Its other results at 63 mm (2 × 2.24 mm strands, 0.20 mm steel): 7.10 kg, $59.7, OP2 +3.7 points, OP3 −2.9 points, range +0.2 % / +0.1 %.</p>
<h3>Best designs at OD 165</h3>
{feas_table([165.0])}
{key_table([CU, 'OD165 Q4 70 mm'])}
{ops_table(['OD165 Q4 70 mm'])}
{fig('effdelta_3.png', 'Q4, 70 mm: efficiency change against copper.', False)}
{mc_sheet('OD165 Q4 70 mm')}
<h3>Demagnetisation and flux maps, OD 165</h3>
{demag_rows(['S0', 'P1', 'Q4'])}
<p>No separate THD and ripple FEMM run was made for P1 and Q4. Q4 has the same rotor and slot tip as Q1 and R1, which gave THD 4.50–4.55 % and ripple 14.4–14.5 / 20.1–20.3 / 5.6 % without skew.</p>
{flux_block(['P1', 'Q4'])}
''')

S['od167'] = ('OD 167 mm', f'''
<p class="lede">The 167 mm OD gives a 21 mm slot (357 mm², 28 % more than copper) with the copper tooth width and a 9 mm yoke. That is enough for aluminium at the copper stack length of 63 mm. {n167} designs meet the hard limits.</p>
{geo_rows(['P2', 'P3', 'P4', 'Q1', 'Q2', 'Q3'])}
<p>P3 and P4 push the slot to 22 mm with an 8 mm yoke; the yoke saturates and no design on them meets the limits. Q1 adds the rotor change and 4.0 mm slot opening to P2, Q2 also deepens the tooth tip to 1.2 mm, and Q3 has the rotor change only.</p>
<h3>Best designs at OD 167</h3>
{feas_table([167.0])}
{key_table([CU, 'OD167 Q1 63 mm', 'OD167 Q2 70 mm'])}
{ops_table(['OD167 Q1 63 mm', 'OD167 Q2 70 mm'])}
<div class="pair">{fig('effdelta_4.png', 'Q1, 63 mm: efficiency change against copper.', False)}{fig('effdelta_5.png', 'Q2, 70 mm: efficiency change against copper.', False)}</div>
{mc_sheet('OD167 Q1 63 mm')}
{mc_sheet('OD167 Q2 70 mm')}
<h3>Ripple, THD and demagnetisation at OD 167</h3>
{ripple_table(['ALU_S0', 'ALU_Q1', 'ALU_Q2'])}
{demag_rows(['S0', 'P2', 'P3', 'P4', 'Q1', 'Q2', 'Q3'])}
<h3>Flux maps, OD 167</h3>
{flux_block(['P2', 'P3', 'P4', 'Q1', 'Q2', 'Q3'])}
''')

S['tradeoffs'] = ('Trade-offs', f'''
<p class="lede">Three things pull against each other: peak-point efficiency wants a longer stack, drive-cycle range wants a short stack (less iron at light load), and mass and cost want both small. A larger OD relieves all three at once.</p>
{fig('tradeoff_scatter.png', 'All designs meeting the hard limits. Left: mass against WMTC range. Middle: mass against OP3 efficiency. Right: cost against telematics range (the cheap cluster near $33 uses M235-35A steel).')}
{fig('stack_effect.png', 'Best design at each stack length, per OD. Range falls and OP3 efficiency rises as the stack grows.')}
{fig('envelopes.png', 'Torque-speed limit of the key designs and their ratio to copper.')}
{fig('effmap_delta.png', 'Efficiency change against copper over the whole operating range. Blue = better than copper, red = worse. The dashed line is the aluminium design\'s torque limit.')}
{table(['Choice', 'Gain', 'Cost'], [
 ['OD 160 → 167 mm', '+28 % slot area at the same tooth and stack; 63 mm stack becomes possible', '+0.4 kg stator steel at the same length, larger housing'],
 ['Stack +3 mm', 'OP3 efficiency +0.3 to +0.4 points', '+0.3 kg, +$2.5, range −0.1 to −0.2 %'],
 ['10 → 9 turns per coil', 'Lower demagnetising field at 360 A, more current headroom', 'Needs about 11 % longer stack for the same torque'],
 ['M235-35A → 0.20 mm steel', 'Eddy-current iron loss falls to about one third; range +1.4 to +1.5 %', '+$32 to +$41 per motor at the assumed $4.0/kg'],
 ['Rotor inset 1.5 / opening 8.0 + slot opening 4.0', 'Torque +0.5 to +0.9 %, OP2 ripple 87 → 20 %, THD 5.3 → 4.5 %', 'Demag margin slightly lower (1.2 vs 0.9 % area below 0.1 T)'],
 ['3° three-step rotor skew', 'Ripple below 10 % everywhere, THD 3.6 %', 'Torque −0.3 %, segmented rotor assembly'],
], 'plain')}
''')

flux_all = flux_block(list(ROWS))
solved = [d for d in ROWS if os.path.exists(os.path.join(R, 'fig', 'flux_%s.jpg' % d))]
pending = [d for d in ROWS if d not in solved]
done_note = ('Maps solved so far: %s. Still solving: %s. This page is updated when they finish.' % (', '.join(solved), ', '.join(pending))) if pending else 'All 17 geometries are solved.'
S['flux'] = ('Flux maps', f'''
<p>{done_note}</p>
<p>Every geometry is solved at no load and at the 2000 rpm peak point (360 A rms at 20° current angle, magnets 100 °C), with the rotor at 0°. Colour = |B|, black lines = flux lines. The right panel zooms on one pole and tooth at the peak point.</p>
{flux_all if flux_all else '<p>Flux maps are being rendered.</p>'}
''')

demag_figs = ''.join(fig('demag_%s.jpg' % dn, '%s: field along the magnetisation in the magnets, −20 °C, 360 A rms on the −d axis.' % dn) for dn in ROWS
                     if os.path.exists(os.path.join(R, 'fig', 'demag_%s.jpg' % dn)))
S['demag'] = ('Demagnetisation', f'''
<p class="lede">Ferrite loses coercivity in the cold, so the worst case is a cold start with full current on the −d axis (field weakening or a fault). Every geometry was checked at −20 °C, 360 A rms, two rotor positions.</p>
{demag_rows(list(ROWS))}
<ul>
<li>The copper geometry already has 0.9–1.0 % of the magnet area below 0.1 T in this case, at the magnet corners next to the air gap.</li>
<li>Moving the magnet 0.5 mm closer to the gap (inset 1.5) raises this to 1.1–1.3 %. The stator OD and slot shape barely change it.</li>
<li>The values use 10 turns per coil. Designs with 9 or 8 turns per coil carry 90 % or 80 % of this field at 360 A, so their margin is better than copper's.</li>
<li>Confirm in Motor-CAD with the DM4654 cold Hcj curve before release. If the margin is not enough, keep the copper inset of 2.0 mm: this gives back about 0.3 % torque.</li>
</ul>
{demag_figs if demag_figs else '<p>Demagnetisation maps are being rendered.</p>'}
''')

S['ripple'] = ('Ripple and THD', f'''
{ripple_table(['ALU_S0', 'ALU_Q1', 'ALU_Q2', 'ALU_R0', 'ALU_R1'])}
{fig('ripple_waves.png', 'Torque over one slot pitch at the three operating points (FEMM, 1° steps).')}
{fig('emf_skew.png', 'Left: no-load back-EMF harmonics. Right: worst ripple of OP1/OP2/OP3 against three-step rotor skew.')}
<p>All aluminium geometries with the rotor change have THD below 5 % without skew. Ripple below 10 % at every point needs a 3° three-step skew; copper needs the same and stays at 54 % at OP2 even with it, because its OP2 ripple starts from 87 %.</p>
''')

ci = D['cycle_info']
S['cycle'] = ('Drive cycle', f'''
<p>Vehicle: gross mass 757 kg, CdA 1.117 m², wheel radius 0.30 m, gear ratio 13.9 (given). Assumed: rolling resistance 0.015, gear efficiency 0.95, rotating-mass factor 1.05, regen down to −26.4 N·m (largest regen torque in the log). Energy is counted at the motor terminals, motoring and regen, on a flat road.</p>
{fig('cycles.png', 'WMTC Class 1, the telematics speed trace, and the motor operating points they produce.')}
{table(['Design', 'Telematics Wh/km', 'WMTC Class 1 Wh/km', 'Range telematics', 'Range WMTC'], [[nm, f(CYC[nm]['Wh_km']['Telematics speed trace'], 2), f(CYC[nm]['Wh_km']['WMTC Class 1'], 2), f(CYC[nm]['dRange']['Telematics speed trace'], 2, True) + ' %', f(CYC[nm]['dRange']['WMTC Class 1'], 2, True) + ' %'] for nm in KR], 'num')}
<p>Telematics trace: {f(ci['Telematics speed trace']['km'], 2)} km over {ci['Telematics speed trace']['n_pts']} one-second steps (30 logging gaps skipped). WMTC Class 1: {f(ci['WMTC Class 1']['km'], 2)} km. The logged motor torque is negative even at cruise, so energy is computed from the speed trace through the vehicle model. Speed steps in the log create short acceleration spikes; those points are held at the motor torque limit.</p>
<p>Both cycles spend most of their time below 15 N·m, where iron loss dominates. That is why thin steel and a short stack decide the range, and why higher OP3 copper loss hardly shows.</p>
''')

S['cost'] = ('Cost-benefit', f'''
{fig('cost_bars.png', 'Material cost per motor: winding wire, electrical steel net of scrap credit, ferrite magnets.')}
{table(['Design', 'Wire $', 'Steel $', 'Magnets $', 'Total $', 'vs copper $', 'Mass kg', 'vs copper kg', 'Range WMTC'], [[nm, f(COST[nm]['wire'], 2), f(COST[nm]['steel'], 2), f(COST[nm]['magnet'], 2), f(COST[nm]['total'], 2), f(COST[nm]['total'] - COST[CU]['total'], 2, True), f(COST[nm]['masses']['total'], 2), f(COST[nm]['masses']['total'] - COST[CU]['masses']['total'], 2, True), f(CYC[nm]['dRange']['WMTC Class 1'], 2, True) + ' %'] for nm in KR], 'num')}
<h3>Price assumptions</h3>
{table(['Item', 'USD/kg', 'Note'], [['Copper magnet wire', '16.5', 'Includes 5 % wire waste'], ['Aluminium magnet wire', '5.5', 'Includes 5 % wire waste'], ['M235-35A', '1.8', ''], ['0.20 mm steel (NO20 class)', '4.0', 'Estimate; the largest uncertainty'], ['Ferrite magnet', '4.0', ''], ['Steel scrap credit', '0.3', 'Square blank = stator OD + 4 mm']], 'plain')}
<ul>
<li>Aluminium saves $22–23 of wire per motor. The 0.20 mm steel adds $32–41 (more steel for the larger OD or longer stack). Net: +$9 to +$21.</li>
<li>With M235-35A, aluminium designs that meet the torque and OP1/OP2 loss limits cost $32–35, about $16–20 less than copper, but lose 1.3–2.2 % range (P2 66 mm: 7.56 kg, $32.4, range −1.4 %).</li>
<li>Each 1 % of range is worth about 0.6 Wh/km. Whether $20 per motor beats 1–2 % range depends on the battery cost per kWh, which was not given.</li>
</ul>
''')

S['structure'] = ('Structure', f'''
<p>The spoke rotor has no bridge at the air gap: each pole piece and its magnet hang on the web between the magnet pockets near the shaft. The web stress scales with the centrifugal load of everything outside the web root, so the screen compares Σ ρ·A·r of the pole steel and magnets (per unit length) against the copper rotor.</p>
{table(['Rotor change', 'Web load vs copper', 'Magnet area mm²', 'Pole steel area mm²'], [
 ['Copper rotor (as built)', '0.00 %', '4731.9', '2294.4'], ['Magnet inset 1.5', '−0.28 %', '4761.2', '2237.7'], ['Magnet opening 7.5', '+0.45 %', '4731.9', '2313.3'],
 ['Inset 1.5 + opening 7.5', '+0.04 %', '4761.2', '2251.1'], ['Inset 1.5 + opening 8.0 (used)', '−0.12 %', '4761.2', '2244.3']], 'num')}
<ul>
<li>Unchanged: shaft 25 mm, web thickness 1.0 mm, web offset 4.0 mm, magnet depth 28 mm, magnet cross-section 18 × 10.5 mm block geometry.</li>
<li>A longer stack does not change the stress: load and web section both scale with length. It does lengthen the shaft span, so check the first bending mode for the 70–81 mm options.</li>
<li>Longer stator teeth (21 mm slot at OD 167): check tooth bending modes for NVH and the tooth-to-housing heat path.</li>
<li>Aluminium conductors: plan crimp or weld terminations rated for aluminium, and keep the enamel grade and impregnation of the copper design.</li>
</ul>
''')

S['runlog'] = ('Run log and limits', f'''
<h3>Model limits</h3>
<ul>
<li>2D FEA: end effects are only in the end-turn resistance and the end-winding inductance of the model.</li>
<li>Aluminium wire fill is held at the copper design\'s 0.5175 insulated fill. A real aluminium winding may reach a different fill.</li>
<li>Efficiency values are from the tuned model; differences between designs are more reliable than the absolute values.</li>
<li>Thermal behaviour was not simulated. The OP3 loss rise of the 63–70 mm designs (+25 % at 44 N·m) needs a Motor-CAD thermal check of the peak duration.</li>
<li>Ripple, THD and demagnetisation use the copper ampere-turns (10 turns/coil, 63 mm). Designs with fewer turns and a longer stack are expected to behave alike per unit length.</li>
</ul>
<h3>Run interruptions</h3>
<p>On 16 Oct 2026 at 07:15 and 07:20, Windows Update restarted the computer for planned updates (active hours were 11:00–05:00). All FEMM runs stopped. Every solve is saved as it finishes, so no results were lost and the runs resumed. The scripts were then hardened: they keep the computer awake while running, write result files atomically, restart FEMM and retry a solve after a FEMM crash, and close their own FEMM windows when done.</p>
<h3>Files</h3>
{table(['File', 'Content'], [['alu_femm.json', 'FEMM grids of the 17 geometries'], ['alu_sweep_v2.json', 'Phase-2 winding sweep (2,592 designs)'], ['alu_v2_eval.json, alu_od160_eval.json', 'Drive cycle, cost and current density of the candidates'],
 ['alu_verify.json', 'FEMM peak torque, THD, ripple and skew'], ['report_full/maps_stats.json', 'Demagnetisation statistics of all geometries'], ['alu_tradeoff_summary.csv', 'Key designs in one table'], ['work/maps/*.ans', 'FEMM solutions behind the flux and demagnetisation maps']], 'plain')}
''')

order = ['summary', 'method', 'copper', 'od160', 'od165', 'od167', 'tradeoffs', 'flux', 'demag', 'ripple', 'cycle', 'cost', 'structure', 'runlog']
nav = ''.join('<a href="#%s" data-p="%s">%s</a>' % (k, k, S[k][0]) for k in order)
pages = ''.join('<section class="page" id="%s" aria-label="%s"><header class="ph"><p class="eyebrow">Spoke V2 · aluminium winding</p><h2>%s</h2></header>%s</section>' % (k, S[k][0], S[k][0], S[k][1]) for k in order)
tpl = open(os.path.join(H, 'report_full_template.html'), encoding='utf-8').read()
html = tpl.replace('{{NAV}}', nav).replace('{{PAGES}}', pages)
open(os.path.join(R, 'index.html'), 'w', encoding='utf-8').write(html)
print('written', len(html) // 1024, 'kB')
