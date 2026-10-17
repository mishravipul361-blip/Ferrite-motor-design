"""Operating-point report: Motor-CAD vs FEMM at three points -> op_report.html"""
import base64, io, json, math, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(H, 'op_compare.json')))
RP = json.load(open(os.path.join(H, 'op_ripple.json'))) if os.path.exists(os.path.join(H, 'op_ripple.json')) else {}
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .3})


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=110, bbox_inches='tight', facecolor='white'); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


ROWS = [  # key, label, unit, group
    ('Shaft_Torque', 'Shaft torque', 'N·m', 'Output'), ('EM_Torque', 'Electromagnetic torque', 'N·m', 'Output'),
    ('Magnet_Torque', 'Magnet torque', 'N·m', 'Output'), ('Reluctance_Torque', 'Reluctance torque', 'N·m', 'Output'),
    ('Shaft_Power', 'Shaft power', 'W', 'Output'), ('EM_Power', 'Electromagnetic power', 'W', 'Output'),
    ('Terminal_Power', 'Terminal (input) power', 'W', 'Output'), ('Efficiency', 'Efficiency', '%', 'Output'),
    ('Ripple_pp', 'Torque ripple pk-pk', 'N·m', 'Output'), ('Ripple_pct', 'Torque ripple', '%', 'Output'),
    ('I_ph_rms', 'Phase current (rms)', 'A', 'Electrical'), ('I_line_rms', 'Line current (rms)', 'A', 'Electrical'),
    ('Idc', 'DC current', 'A', 'Electrical'), ('V_ph_rms', 'Phase = line voltage (rms)', 'V', 'Electrical'),
    ('Phase_Advance', 'Phase advance', '° el', 'Electrical'), ('PF', 'Power factor', '', 'Electrical'),
    ('Psi_d', 'Flux linkage d', 'mVs', 'Machine'), ('Psi_q', 'Flux linkage q', 'mVs', 'Machine'), ('Psi_m', 'Magnet flux linkage', 'mVs', 'Machine'),
    ('Ld', 'Ld (phase)', 'mH', 'Machine'), ('Lq', 'Lq (phase)', 'mH', 'Machine'),
    ('Total_Loss', 'Total loss', 'W', 'Losses'), ('Cu', 'Copper loss', 'W', 'Losses'), ('Cu_DC', 'Copper DC', 'W', 'Losses'),
    ('Cu_AC', 'Copper AC', 'W', 'Losses'), ('Iron', 'Iron loss', 'W', 'Losses'), ('Magnet_Loss', 'Magnet loss', 'W', 'Losses'),
    ('Mech', 'Mechanical loss', 'W', 'Losses'), ('Windage', 'Windage', 'W', 'Losses'), ('Friction', 'Friction (bearings)', 'W', 'Losses')]


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '–'
    a = abs(v)
    return '%.0f' % v if a >= 1000 else ('%.1f' % v if a >= 100 else ('%.2f' % v if a >= 1 else '%.4f' % v))


def pill(v, ref, key):
    if v is None or ref in (None, 0) or (isinstance(v, float) and math.isnan(v)):
        return '<td>–</td>'
    if key == 'Efficiency':
        d = v - ref; cls = 'ok' if abs(d) <= .5 else ('mid' if abs(d) <= 1.5 else 'bad')
        return '<td><span class="pill %s">%+.2f pt</span></td>' % (cls, d)
    d = 100 * (v - ref) / abs(ref); cls = 'ok' if abs(d) <= 3 else ('mid' if abs(d) <= 10 else 'bad')
    return '<td><span class="pill %s">%+.1f %%</span></td>' % (cls, d)


blocks, figs = [], {}
for k, (name, d) in enumerate(C.items()):
    m, a, b = d['mcad'], d['same_point'], d['same_current']
    op = name.split()[0]
    if op in RP:
        b = dict(b, Ripple_pp=RP[op]['ripple_pp'], Ripple_pct=RP[op]['ripple_pct'])
    rows, grp = '', None
    for key, lab, unit, g in ROWS:
        if g != grp:
            rows += '<tr class="grp"><td colspan="6">%s</td></tr>' % g; grp = g
        rows += '<tr><td class="txt">%s</td><td class="u">%s</td><td>%s</td><td>%s</td>%s<td>%s</td>%s</tr>' % (
            lab, unit, fmt(m.get(key)), fmt(a.get(key)) if a else '–', pill(a.get(key), m.get(key), key) if a else '<td>–</td>',
            fmt(b.get(key)), pill(b.get(key), m.get(key), key))
    blocks.append((name, rows))
    # loss bar chart
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    keys = [('Cu_DC', 'Cu DC'), ('Cu_AC', 'Cu AC'), ('Iron', 'Iron'), ('Mech', 'Mech')]
    x = np.arange(len(keys)); w = .27
    ax.bar(x - w, [m[q] for q, _ in keys], w, color='#d9480f', label='Motor-CAD')
    if a:
        ax.bar(x, [a[q] for q, _ in keys], w, color='#74c0fc', label='FEMM, same speed/torque')
    ax.bar(x + w, [b[q] for q, _ in keys], w, color='#1c7ed6', label='FEMM, same current')
    ax.set_xticks(x, [l for _, l in keys]); ax.set_ylabel('W'); ax.set_title(name, fontsize=10); ax.legend(fontsize=7)
    figs[op] = png(fig)
if RP:
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for a_, (op, r) in zip(ax, RP.items()):
        mc = [d['mcad'] for n, d in C.items() if n.startswith(op)][0]
        a_.plot(r['theta'], r['T'], 'o-', color='#1c7ed6', ms=3, label='FEMM EM torque')
        a_.axhline(mc['EM_Torque'], color='#d9480f', ls='--', lw=1.2, label='Motor-CAD EM torque (mean)')
        a_.axhspan(mc['EM_Torque'] - mc['Ripple_pp'] / 2, mc['EM_Torque'] + mc['Ripple_pp'] / 2, color='#d9480f', alpha=.12, label='Motor-CAD ripple band')
        a_.set(title='%s: FEMM ripple %.2f N·m (%.1f %%)' % (op, r['ripple_pp'], r['ripple_pct']), xlabel='rotor position (mech deg)', ylabel='N·m')
    ax[0].legend(fontsize=7)
    figs['ripple'] = png(fig)

tpl = open(os.path.join(H, 'op_template.html'), encoding='utf8').read()
for i, (name, rows) in enumerate(blocks):
    tpl = tpl.replace('{{ROWS%d}}' % (i + 1), rows).replace('{{NAME%d}}' % (i + 1), name)
for k, v in figs.items():
    tpl = tpl.replace('{{IMG_%s}}' % k.upper(), v)
for op in ('OP1', 'OP2', 'OP3'):
    r = RP.get(op, {})
    tpl = tpl.replace('{{RIP_%s}}' % op, '%.2f N·m (%.1f %%)' % (r['ripple_pp'], r['ripple_pct']) if r else 'pending')
tpl = tpl.replace('{{RIPPLE_FIG}}', '<figure><img src="%s" alt="Torque ripple FEMM vs Motor-CAD"><figcaption>FEMM torque over one 12° ripple period at Motor-CAD\'s exact currents. The shaded band is Motor-CAD\'s mean ± half its pk-pk ripple.</figcaption></figure>' % figs['ripple'] if RP else '<p class="muted">Torque-ripple FEMM run in progress.</p>')
assert '{{' not in tpl, [l for l in tpl.split('\n') if '{{' in l][:3]
open(os.path.join(H, 'op_report.html'), 'w', encoding='utf8').write(tpl)
print('op_report.html %d kB' % (len(tpl) // 1024))
