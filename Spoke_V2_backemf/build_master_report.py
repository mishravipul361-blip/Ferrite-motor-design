"""Combined report: earlier vs tuned FEMM, root causes, final Lab maps, scope and data requests -> master_report.html
Reuses the figures already rendered in consistency_report.html and lab_report.html (no new solves)."""
import json, os, re
H = os.path.dirname(os.path.abspath(__file__))


def imgs(fn, names):
    s = open(os.path.join(H, fn), encoding='utf8').read()
    found = re.findall(r'src="(data:image/png;base64,[A-Za-z0-9+/=]+)"', s)
    assert len(found) == len(names), (fn, len(found), len(names))
    return dict(zip(names, found))


I = {}
I.update(imgs('consistency_report.html', ['effmaps', 'effdiff', 'env', 'envloss', 'parity']))
I.update(imgs('lab_report.html', ['val', 'lab_env', 'lab_env2', 'eff', 'loss', 'pcu', 'pac', 'pfe', 'I', 'Idc', 'V', 'pf', 'gamma', 'dq']))
v1 = json.load(open(os.path.join(H, 'lab_compare_v1.json')))['summary']['A_same_point']
v2 = json.load(open(os.path.join(H, 'lab_compare_v2.json')))['summary']['A_same_point']
QN = [('Efficiency', 'Efficiency'), ('Shaft_Torque', 'Shaft torque'), ('Stator_Current_Line_RMS', 'Line current'), ('Voltage_Phase_Peak', 'Phase voltage'),
      ('Power_Factor', 'Power factor'), ('Flux_Linkage_D', 'Flux linkage d'), ('Stator_Copper_Loss_DC', 'Copper loss DC'), ('Stator_Copper_Loss_AC', 'Copper loss AC'),
      ('Iron_Loss_Stator', 'Iron loss stator'), ('Iron_Loss_Rotor', 'Iron loss rotor'), ('Iron_Loss', 'Iron loss total'), ('Mechanical_Loss', 'Mechanical loss'),
      ('Total_Loss', 'Total loss'), ('Idc', 'DC current')]


def c(s, q):
    if q == 'Efficiency':
        v = s['mean']; cls = 'ok' if abs(v) <= .75 else ('mid' if abs(v) <= 1.5 else 'bad')
        return '<td><span class="pill %s">%+.2f pt</span></td>' % (cls, v)
    v = s['mean_abs_rel_pct']; cls = 'ok' if v <= 5 else ('mid' if v <= 15 else 'bad')
    return '<td><span class="pill %s">%.1f %%</span></td><td class="src">%+.1f %%</td>' % (cls, v, s['mean_rel_pct'])


rows = ''.join('<tr><td class="txt">%s</td>%s%s</tr>' % (lab, c(v1[q], q) + ('<td></td>' if q == 'Efficiency' else ''), c(v2[q], q) + ('<td></td>' if q == 'Efficiency' else ''))
               for q, lab in QN)
tpl = open(os.path.join(H, 'master_template.html'), encoding='utf8').read()
for k, v in I.items():
    tpl = tpl.replace('{{IMG_%s}}' % k.upper(), v)
tpl = tpl.replace('{{ERR_ROWS}}', rows)
assert '{{' not in tpl, [l for l in tpl.split('\n') if '{{' in l][:3]
open(os.path.join(H, 'master_report.html'), 'w', encoding='utf8').write(tpl)
print('master_report.html %d kB' % (len(tpl) // 1024))
