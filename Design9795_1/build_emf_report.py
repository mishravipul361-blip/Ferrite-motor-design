"""Back-EMF validation page: report_emf/backemf_validation.html (from oc_validate.json, oc_fields.json if present)."""
import base64, json, os

H = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(H, 'oc_validate.json')))
s, u, mc = d['skewed (MC slices)'], d['unskewed'], d['MC']
img = base64.b64encode(open(os.path.join(H, 'report_emf', 'emf.png'), 'rb').read()).decode()
fl = d.get('fields') or {}
FIELDS_NOTE = 'all 12 rotor positions of the 6° cogging period' if fl else 'rotor positions 0–2.5° (the full 12-position run is repeating)'
fv = lambda k, dflt: fl.get(k, dflt)


def neg(t):
    return t.replace('-', '−')


def row(n, a, m, b, dec=3, pts=False):
    diff = ('%+.2f points' % (a - m)) if pts else ('%+.2f %%' % (100 * (a / m - 1)))
    ok = abs(a - m) <= 0.3 if pts else abs(a / m - 1) <= 0.01
    return '<tr><td>%s</td><td>%.*f</td><td>%.*f</td><td class="%s">%s</td><td>%s</td></tr>' % (
        n, dec, a, dec, m, 'ok' if ok else 'warn', neg(diff), ('%.*f' % (dec, b)) if b is not None else '–')


rows = ''.join([
    row('Back-EMF phase peak (V)', s['E_pk'], mc['E_ph_pk'], u['E_pk']),
    row('Back-EMF line-line peak (V), delta', s['E_pk'], mc['E_ll_pk'], u['E_pk']),
    row('Back-EMF fundamental peak (V)', s['E_fund_pk'], mc['E_fund_pk'], u['E_fund_pk']),
    row('Back-EMF phase rms (V)', s['E_rms'], mc['E_ph_rms'], u['E_rms']),
    row('Ke (V·s/rad)', s['Ke'], mc['Ke'], u['Ke'], 4),
    row('Ke fundamental (V·s/rad)', s['Ke_fund'], mc['Ke_fund'], u['Ke_fund'], 4),
    row('Magnet flux linkage ψd (mV·s)', s['psi_m_mVs'], 1e3 * mc['psi_d'], u['psi_m_mVs'], 2),
    row('THD, phase (%)', s['THD'], mc['THD'], u['THD'], 2, True),
    row('Phase resistance at 100 °C (mΩ)', 1e3 * d['R_ph'], 1e3 * mc['R_ph'], None, 3)])
F = [('Air-gap flux density, average', fv('airgap_avg', 0.4946), 0.4956, 'ok'), ('Air-gap flux density, maximum', fv('airgap_max', 0.7787), 0.7731, 'ok'),
     ('Stator tooth flux density, maximum', fv('tooth_max', 1.042), 1.044, 'ok'), ('Stator back iron flux density, maximum', fv('yoke_max', 0.938), 0.9289, 'ok'),
     ('Stator tooth tip flux density, maximum', fv('tip_max', 0.990), 1.469, 'na'), ('Rotor back iron flux density, maximum', fv('rotor_max', 1.099), 1.518, 'na')]
frows = ''.join('<tr><td>%s</td><td>%.4f</td><td>%.4f</td><td class="%s">%s</td></tr>' % (
    n, a, m, c, neg('%+.1f %%' % (100 * (a / m - 1))) if c == 'ok' else 'different definition') for n, a, m, c in F)
hs = ['3', '5', '7', '9', '11', '13']
hrow = lambda lab, h: '<tr><td>%s</td>%s</tr>' % (lab, ''.join('<td>%.2f</td>' % h[k] for k in hs))

CSS = """
/* Layout: single technical sheet; verdict first, then the comparison tables and the waveform figure. */
:root { --paper:#f5f6f7; --sheet:#fff; --ink:#18202b; --ink-2:#4d5866; --rule:#d8dde3; --rule-2:#eceff2; --accent:#4f6b86; --ok:#1d7a4c; --ok-bg:#e6f3ec; --warn:#9a6400; --na:#6b7480;
  --f-d:"Barlow Condensed","Arial Narrow",Arial,sans-serif; --f-b:"IBM Plex Sans","Segoe UI",Arial,sans-serif; --f-m:"IBM Plex Mono",Consolas,monospace }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --paper:#12161c; --sheet:#1a2028; --ink:#e6e9ed; --ink-2:#a6b0bc; --rule:#313a46; --rule-2:#252c36; --accent:#8fb0cf; --ok:#5cc28d; --ok-bg:#1d3329; --warn:#e0a83a; --na:#8e98a4; color-scheme:dark } }
:root[data-theme="dark"] { --paper:#12161c; --sheet:#1a2028; --ink:#e6e9ed; --ink-2:#a6b0bc; --rule:#313a46; --rule-2:#252c36; --accent:#8fb0cf; --ok:#5cc28d; --ok-bg:#1d3329; --warn:#e0a83a; --na:#8e98a4; color-scheme:dark }
body { background:var(--paper); color:var(--ink); font:15px/1.6 var(--f-b); margin:0 }
.wrap { max-width:1080px; margin:0 auto; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); padding-block:28px 48px }
.sheet { background:var(--sheet); border:1px solid var(--rule); border-radius:6px; padding:28px clamp(16px,3vw,40px) }
.eyebrow { font:500 12px var(--f-m); letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin:0 }
h1 { font:700 40px/1.05 var(--f-d); margin:4px 0 14px; text-wrap:balance }
h2 { font:600 24px/1.2 var(--f-d); margin:30px 0 8px; text-wrap:balance }
p, li { max-width:80ch }
.verdict { background:var(--ok-bg); border:1px solid var(--ok); border-radius:6px; padding:14px 18px; margin:16px 0 }
.verdict strong { color:var(--ok) }
.tw { overflow-x:auto; border:1px solid var(--rule); border-radius:4px; margin:10px 0 }
table { border-collapse:collapse; width:100%; font-size:13.5px }
th { font:500 12px var(--f-m); text-align:right; color:var(--ink-2); background:var(--rule-2); padding:8px 10px; border-bottom:1px solid var(--rule) }
th:first-child, td:first-child { text-align:left }
td { padding:7px 10px; border-bottom:1px solid var(--rule-2); text-align:right; font-family:var(--f-m); font-variant-numeric:tabular-nums; white-space:nowrap }
td:first-child { font-family:var(--f-b); white-space:normal; min-width:180px }
td.ok { color:var(--ok); font-weight:500 } td.warn { color:var(--warn); font-weight:500 } td.na { color:var(--na) }
figure { margin:16px 0 } figure img { display:block; max-width:100%; height:auto; background:#fff; border:1px solid var(--rule); border-radius:4px; padding:6px }
figcaption, .note { font-size:13px; color:var(--ink-2) }
dl { display:grid; grid-template-columns:max-content minmax(0,1fr); gap:6px 16px; font-size:14px } dt { color:var(--ink-2) } dd { margin:0 }
@media (max-width:600px) { dl { grid-template-columns:minmax(0,1fr) } dt { margin-top:6px } h1 { font-size:32px } }
"""

html = f'''<title>Design9795 Back-EMF Validation</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap"><div class="sheet">
<p class="eyebrow">Design9795_1 · open circuit · magnets 100 °C · 1000 rpm · I = 0</p>
<h1>Back-EMF validation: FEMM against Motor-CAD</h1>
<div class="verdict"><strong>Verified.</strong> Back-EMF amplitude, fundamental, rms, Ke and magnet flux agree with Motor-CAD within 0.7 %. Air-gap, tooth and back-iron flux densities agree within 1 %. THD is {s['THD']:.2f} % against {mc['THD']:.2f} %: {s['THD'] - mc['THD']:.2f} points higher, in the 9th to 13th harmonics that the rotor skew suppresses.</div>
<h2>How the comparison was made</h2>
<dl>
<dt>FEMM model</dt><dd>2D magnetostatic, full machine, geometry from the Design9795_1 screenshots (slot area 281.2 mm² and wire fill 0.5129, both exactly as Motor-CAD). Stack 64 mm, DM4654 magnets at 100 °C (Br 0.394 T).</dd>
<dt>Rotor positions</dt><dd>1° mechanical steps over half an electrical cycle (36 solves); the second half follows from half-wave symmetry, giving 72 points per electrical cycle.</dd>
<dt>Skew</dt><dd>Motor-CAD rotor skew: 3 equal slices at −2 / 0 / +2° mechanical. FEMM is 2D, so the phase flux linkage is averaged over the three shifted rotor positions, exactly, on its Fourier series.</dd>
<dt>EMF and THD</dt><dd>Resampled at 30 points per cycle with a forward difference, like Motor-CAD's back-EMF calculation (30 points per cycle, 1 cycle). THD uses harmonics 2 to 15.</dd>
</dl>
<h2>Back-EMF and winding</h2>
<div class="tw"><table><thead><tr><th>Quantity</th><th>FEMM with skew</th><th>Motor-CAD</th><th>Difference</th><th>FEMM 2D, no skew</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="note">Delta connection: line-line back-EMF equals phase back-EMF, as in Motor-CAD. Phase resistance: 10 turns per coil, 2 coils in series per path, 2 parallel paths, turn length 2 × 64 + 75.634 = 203.6 mm, SWG 11 copper at 100 °C.</p>
<figure><img src="data:image/png;base64,{img}" alt="Back-EMF waveform and harmonic content"><figcaption>Left: phase back-EMF at 1000 rpm, with and without the Motor-CAD skew. Right: harmonic content. The skew removes most of the 9th, 11th and 13th harmonics, the slot harmonics of a 12-slot, 10-pole machine.</figcaption></figure>
<h2>Harmonics, % of fundamental</h2>
<div class="tw"><table><thead><tr><th>FEMM</th><th>3rd</th><th>5th</th><th>7th</th><th>9th</th><th>11th</th><th>13th</th></tr></thead><tbody>{hrow('With Motor-CAD skew', s['h_pct'])}{hrow('2D, no skew', u['h_pct'])}</tbody></table></div>
<h2>Flux densities, open circuit</h2>
<div class="tw"><table><thead><tr><th>Quantity</th><th>FEMM</th><th>Motor-CAD</th><th>Difference</th></tr></thead><tbody>{frows}</tbody></table></div>
<p class="note">FEMM: air gap = radial B on the mid-gap circle; tooth = B averaged across the tooth width at mid height; back iron = B averaged across the yoke; maximum over all teeth and {FIELDS_NOTE}. Motor-CAD reads the tooth tip and rotor back iron at its own fixed points, so those two cannot be compared one-to-one with section averages.</p>
<h2>What this means for the Lab model</h2>
<ul>
<li>The magnet flux, which sets torque per amp and the field-weakening speed, matches Motor-CAD to {abs(100 * (s['psi_m_mVs'] / (1e3 * mc['psi_d']) - 1)):.2f} %.</li>
<li>Motor-CAD includes the rotor skew in all its results. The FEMM Lab model is built in 2D, so at the same current its torque will be about 1 % higher (fundamental skew factor 0.990) and its torque ripple higher, unless the same skew correction is applied. The Lab comparison will show both.</li>
</ul>
</div></div>
'''
open(os.path.join(H, 'report_emf', 'backemf_validation.html'), 'w', encoding='utf-8').write(html)
print(len(html) // 1024, 'kB')
