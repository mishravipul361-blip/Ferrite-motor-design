"""Sensitivity study -> Excel: Spoke_V2_sensitivity.xlsx (README, Designs, Sensitivity, Skew, Raw_solves)."""
import json, math, os
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import sens as S
from sens_models import design_ids, winding_check

H = S.H
R = json.load(open(os.path.join(H, 'sens_results.json')))
from sens_analyse import cache
IDS = design_ids()
THD_MAX, RIP_MAX = 5.0, 10.0
HDR = Font(bold=True, color='FFFFFF'); HFILL = PatternFill('solid', fgColor='0F6E8C')
OK = PatternFill('solid', fgColor='E1F3E6'); BAD = PatternFill('solid', fgColor='FDE4E1'); BASE = PatternFill('solid', fgColor='FFF4D6')


def sheet(wb, title, header, rows, widths=None, freeze='B2'):
    ws = wb.create_sheet(title)
    ws.append(header)
    for c in ws[1]:
        c.font = HDR; c.fill = HFILL; c.alignment = Alignment(wrap_text=True, vertical='center')
    for r in rows:
        ws.append(r)
    ws.freeze_panes = freeze
    for i, h in enumerate(header, 1):
        ws.column_dimensions[get_column_letter(i)].width = (widths or {}).get(h, max(10, min(28, len(str(h)) + 2)))
    ws.row_dimensions[1].height = 45
    return ws


wb = Workbook(); wb.remove(wb.active)
# ---------------- README ----------------
readme = [
    ['Spoke V2 - geometric sensitivity study (FEMM 2D, tuned to Motor-CAD)', ''],
    ['Generated', '11 Oct 2026'],
    ['', ''],
    ['NOTATION', ''],
    ['D00', 'Baseline: Motor-CAD ADAPTIVE_MODEL_COMPLETE geometry with the hard constraints applied - shaft OD 30 mm, Bridge_thickness 1.0 mm (minimum), bridge_offset 4.0 mm (> 3.5)'],
    ['D01..D20', 'One-at-a-time variations; one input changed, all others at D00 (see sheet Designs). All satisfy the hard constraints.'],
    ['R00', 'Reference only: original Motor-CAD geometry (shaft 25 mm, bridge 1.0 mm) - violates the hard constraints'],
    ['C1, C2', 'Combined candidates chosen from the sensitivities and verified by full FEMM solves (C1 recommended, with 6 deg 3-step skew). Ripple at OP2/OP3 and skew results in sheet Candidates'],
    ['HARD CONSTRAINTS', 'Shaft OD = 30 mm; Bridge_thickness >= 1.0 mm (never below); bridge_offset > 3.5 mm; wire SWG 11 only (2.9464 mm bare, 20 conductors/slot, 10 turns/coil); current density < 16 A/mm2 at 360 A (rms per conductor = 15.24 A/mm2 for all designs)'],
    ['Slot fill', 'Wire slot fill = 20 x insulated-wire area / slot area (Motor-CAD definition, baseline 0.5175). Larger fill = harder to wind; C1/C2 (tooth 16.5 mm) exceed the baseline by 6-9 %'],
    ['Model file', 'FEMM_models\\sensitivity\\<ID>_<parameter>_<value>mm.fem  (geometry, materials, circuits T1..T12 at 0 A, rotor 0 deg)'],
    ['Sxxxx', 'Solve number in sheet Raw_solves (one FEMM magnetostatic solution)'],
    ['', ''],
    ['INPUT NAMES', ''],
    ['S_*', 'Stator: slots, od, bore, tooth_w, slot_depth, corner_r, tip_depth, slot_open, tip_angle (mm / deg)'],
    ['R_*', 'Rotor spoke template: rotor_diameter, shaft_diameter, magnet_inset, magnet_depth, magnet_opening, magnet_X/Y/Z, magnet_glue_gap, Bridge_thickness, bridge_offset (mm)'],
    ['', ''],
    ['OUTPUTS', ''],
    ['O1_T360', 'OBJECTIVE 1 (maximise): MTPA electromagnetic torque at 360 A rms line, 2000 rpm, magnets/winding 100 C [N.m]'],
    ['O1_gamma', 'Current advance angle at MTPA [el deg]'],
    ['O2_loss', 'OBJECTIVE 2 (minimise): total loss at 9339 rpm, 3.24 N.m shaft, 52 V / Mi 0.95 (field weakening) [W]'],
    ['O2_Cu_DC / O2_Cu_AC / O2_Fe / O2_mech', 'Loss split at that point [W]'],
    ['C1_THD', 'REQUIREMENT THD < 5 %: no-load back-EMF THD, 20 C, 30 points/cycle, harmonics 2..13 (Motor-CAD definition; baseline Motor-CAD 5.42 %)'],
    ['C2_ripple_OP1', 'REQUIREMENT ripple < 10 %: torque ripple pk-pk / mean at the worst full-load point, 9000 rpm max current (139.7 A rms ph, 66.38 deg) [%]'],
    ['dX_vs_D00', 'Change against baseline'],
    ['', ''],
    ['CONDITIONS', '52 V DC, Mi 0.95 (49.4 V peak phase), 360 A rms line max, 280 A DC max, delta, 10 turns/coil, 2 parallel paths, stack 63 mm, M235-35A Motor-CAD B-H, DMEGC_DM4654 Br 0.46 T @20 C'],
    ['Method', 'O1: 3 current angles x 2 rotor positions, parabola. O2: local d-q model (4 solves) at the voltage limit; Cu DC exact, Cu AC hybrid x2, iron scaled from the tuned model. C1: 15 half-cycle no-load solves. C2: 12 rotor positions over the 12 deg ripple period.'],
    ['Skew', 'Sheet Skew: rotor skew evaluated analytically from the waveform harmonics of D00 (continuous and 3-step); fine OP1/OP2/OP3 ripple waveforms from op_ripple.json'],
]
ws = wb.create_sheet('README')
for r in readme:
    ws.append(r)
ws.column_dimensions['A'].width = 34; ws.column_dimensions['B'].width = 140
for c in ('A1', 'A4', 'A10', 'A14'):
    ws[c].font = Font(bold=True, size=12 if c == 'A1' else 11)

# ---------------- Designs ----------------
skeys = ['slots', 'od', 'bore', 'tooth_w', 'slot_depth', 'corner_r', 'tip_depth', 'slot_open', 'tip_angle']
rkeys = ['rotor_diameter', 'shaft_diameter', 'magnet_inset', 'magnet_depth', 'magnet_opening', 'magnet_X', 'magnet_Y', 'magnet_Z', 'magnet_glue_gap',
         'Bridge_thickness', 'bridge_offset']
out_hdr = ['O1_T360 [N.m]', 'O1_gamma [deg]', 'O2_loss [W]', 'O2_Cu_DC [W]', 'O2_Cu_AC [W]', 'O2_Fe [W]', 'O2_mech [W]', 'O2_I_line [A rms]', 'O2_eff [%]',
           'C1_THD [%]', 'h3 [%]', 'h5 [%]', 'h7 [%]', 'h9 [%]', 'h11 [%]', 'h13 [%]', 'E1 @1000rpm 20C [V rms]',
           'C2_ripple_OP1 [%]', 'C2_ripple_OP1 [N.m pk-pk]', 'T_OP1 mean [N.m]', 'THD<5%', 'ripple<10%',
           'dT360_vs_D00 [%]', 'dLoss_vs_D00 [%]', 'dTHD_vs_D00 [pt]', 'dRipple_vs_D00 [pt]',
           'Wire', 'Slot area [mm2]', 'Wire slot fill (slot area)', 'Copper slot fill', 'J rms @360A [A/mm2]', 'J peak @360A [A/mm2]', 'J<16 A/mm2 (rms)']
hdr = ['ID', 'Model file', 'Varied parameter', 'Value [mm]', 'Baseline value [mm]'] + ['S_' + k for k in skeys] + ['R_' + k for k in rkeys] + out_hdr
base = R['designs'].get('baseline')
rows = []
for v, d in IDS.items():
    r = R['designs'].get(v) or R.get('candidates', {}).get(v)
    ins = [d['id'], 'FEMM_models\\sensitivity\\' + d['file'], d['param'] or '(baseline)', d['value'],
           (S.BASE_R.get(d['param'], S.BASE_S.get(d['param'])) if d['param'] else None)]
    ins += [d['stator'][k] for k in skeys] + [d['rotor'][k] for k in rkeys]
    if r is None:
        rows.append(ins + ['pending'] + [None] * (len(out_hdr) - 1)); continue
    o2 = r['op2'] or {}
    outs = [r['T360'], r['gamma_opt'], o2.get('loss'), o2.get('pcu'), o2.get('pac'), o2.get('pfe'), S.L.mech_loss(9339.0) if o2 else None,
            o2.get('I_line'), o2.get('eff'), r['thd']] + r['emf_h'] + [r['E1_rms'], r['rip_op1_pct'], r.get('rip_op1_pp'), r.get('T_op1'),
            'PASS' if r['thd'] < THD_MAX else 'FAIL', 'PASS' if r['rip_op1_pct'] < RIP_MAX else 'FAIL']
    if base:
        outs += [100 * (r['T360'] / base['T360'] - 1), 100 * (o2['loss'] / base['op2']['loss'] - 1) if o2 and base['op2'] else None,
                 r['thd'] - base['thd'], r['rip_op1_pct'] - base['rip_op1_pct']]
    w = winding_check(d['stator'])
    outs += [None] * (len(out_hdr) - 7 - len(outs)) + ['SWG 11 (2.9464 mm), 20 cond/slot', w['slot_area'], w['wire_fill'], w['copper_fill'], w['J_rms'], w['J_peak'],
             'PASS' if w['J_rms'] < 16 else 'FAIL']
    rows.append(ins + outs)
ws = sheet(wb, 'Designs', hdr, rows, widths={'Model file': 44, 'Varied parameter': 18, 'ID': 6}, freeze='C2')
for row in ws.iter_rows(min_row=2):
    if row[0].value == 'D00':
        for c in row:
            c.fill = BASE
    for c in row:
        if c.value == 'PASS':
            c.fill = OK
        elif c.value == 'FAIL':
            c.fill = BAD
        if isinstance(c.value, float):
            c.number_format = '0.000'

# ---------------- Sensitivity ----------------
srows = []
params = []
for v, d in IDS.items():
    if d['param'] and d['value'] is not None and d['param'] not in params:
        params.append(d['param'])
for p in params:
    pts = sorted([(d['value'], R['designs'].get(v)) for v, d in IDS.items() if d['param'] == p], key=lambda x: x[0])
    b0 = S.BASE_R.get(p, S.BASE_S.get(p))
    if any(x[1] is None for x in pts) or not base:
        srows.append([p, b0] + ['pending'] * 8); continue
    (lo, rl), (hi, rh) = pts
    dv = hi - lo
    g = lambda f: (f(rh) - f(rl)) / dv
    loss = lambda r: r['op2']['loss'] if r['op2'] else float('nan')
    srows.append([p, b0, lo, hi, g(lambda r: r['T360']), g(loss), g(lambda r: r['thd']), g(lambda r: r['rip_op1_pct']),
                  100 * g(lambda r: r['T360']) / base['T360'], 100 * g(loss) / base['op2']['loss']])
sheet(wb, 'Sensitivity', ['Parameter', 'Baseline [mm]', 'Low [mm]', 'High [mm]', 'dT360/dx [N.m/mm]', 'dLoss_OP2/dx [W/mm]', 'dTHD/dx [pt/mm]',
                          'dRipple_OP1/dx [pt/mm]', 'dT360/dx [%/mm]', 'dLoss/dx [%/mm]'], srows, widths={'Parameter': 20})

# ---------------- Skew ----------------
sk = R['skew']
krows = [[float(s), r['thd_cont'], r['thd_3step'], r['OP1_cont_pct'], r['OP1_3step_pct'], r['OP2_cont_pct'], r['OP2_3step_pct'],
          r['OP3_cont_pct'], r['OP3_3step_pct'], 100 * (r['E1_cont'] / sk['0.0']['E1_cont'] - 1), 100 * (r['E1_3step'] / sk['0.0']['E1_3step'] - 1)]
         for s, r in sk.items()]
sheet(wb, 'Skew', ['Skew [mech deg]', 'THD cont [%]', 'THD 3-step [%]', 'Ripple OP1 cont [%]', 'Ripple OP1 3-step [%]', 'Ripple OP2 cont [%]',
                   'Ripple OP2 3-step [%]', 'Ripple OP3 cont [%]', 'Ripple OP3 3-step [%]', 'EMF/torque change cont [%]', 'EMF/torque change 3-step [%]'], krows)

# ---------------- Raw solves ----------------
order = {v: i for i, (v, _, _) in enumerate(S.VARIANTS + S.CANDIDATES)}
kinds = {'tq': 'torque 360 A', 'rip': 'ripple OP1', 'nl': 'no load 20C', 'op2': 'OP2 d-q', 'rip2': 'ripple OP2', 'rip3': 'ripple OP3'}
raw = sorted([r for r in cache if r['v'] in order], key=lambda r: (order[r['v']], r['kind'], r['a'], r['th']))
rrows = [['S%04d' % (i + 1), IDS[r['v']]['id'], kinds[r['kind']], r['a'], r['th'], r['id'], r['iq'], r['T'], r['psid'] * 1e3, r['psiq'] * 1e3,
          r['psi1'] * 1e3, r.get('st_Sa'), r.get('st_S2'), r.get('cu_B2')] for i, r in enumerate(raw)]
sheet(wb, 'Raw_solves', ['Solve', 'Design ID', 'Solve type', 'gamma [deg] / op2 step', 'Rotor angle [mech deg]', 'id path pk [A]', 'iq path pk [A]',
                         'Torque [N.m] (FEMM sign)', 'psi_d [mWb]', 'psi_q [mWb]', 'psi phase1 path [mWb]', 'stator sum m*Bpk^a', 'stator sum m*Bpk^2',
                         'slot mean B^2 [T^2]'], rrows)
crow = []
for v, c in R.get('candidates', {}).items():
    for s, q in sorted(c['skew'].items(), key=lambda t: float(t[0])):
        crow.append([v, IDS[v]['file'], float(s), c['T360'] * q['k1_3step'], c['op2']['loss'] if c['op2'] else None, q['thd_cont'], q['thd_3step'],
                     q['OP1_cont_pct'], q['OP1_3step_pct'], q['OP2_cont_pct'], q['OP2_3step_pct'], q['OP3_cont_pct'], q['OP3_3step_pct']])
sheet(wb, 'Candidates', ['Candidate', 'Model file', 'Skew [mech deg]', 'Torque 360A (3-step skew) [N.m]', 'OP2 loss (no skew) [W]', 'THD cont [%]', 'THD 3-step [%]',
                         'Ripple OP1 cont [%]', 'Ripple OP1 3-step [%]', 'Ripple OP2 cont [%]', 'Ripple OP2 3-step [%]', 'Ripple OP3 cont [%]', 'Ripple OP3 3-step [%]'],
      crow, widths={'Model file': 60})
fn = os.path.join(H, 'Spoke_V2_sensitivity.xlsx')
wb.save(fn)
print(fn, len(rows), 'designs', len(rrows), 'solves')
