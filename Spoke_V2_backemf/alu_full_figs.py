"""Figures and data for the full aluminium report (report_full/).  python alu_full_figs.py
Model charts: efficiency maps, envelopes, losses, trade-off scatters, cost, drive cycle.
FEMM charts: stator grid torque/flux density, ripple waveforms, back-EMF harmonics, skew."""
import json, math, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import alu_study as A
import alu_femm as AF

H = A.H
OUT = os.path.join(H, 'report_full', 'fig')
os.makedirs(OUT, exist_ok=True)
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', violet='#6250d6', red='#e34948',
         ink='#1d2430', ink2='#5a6372', grid='#d9dde3', cu='#9a5b2e')
OD_COL = {160.0: C['blue'], 165.0: C['orange'], 167.0: C['aqua']}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': C['ink2'], 'axes.labelcolor': C['ink'], 'xtick.color': C['ink2'],
                     'ytick.color': C['ink2'], 'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'legend.frameon': False})
KEY = [('Copper today', 'Copper baseline (SWG 11, 63 mm)', C['cu']),
       ('OD160 R1 70 mm', 'Al R1 L70.0 N18 2x2.24 NO20', C['blue']),
       ('OD160 R3 81 mm', 'Al R3 L81.0 N16 3x2.05 NO20', C['violet']),
       ('OD165 Q4 70 mm', 'Al Q4 L70.0 N18 2x2.36 NO20', C['orange']),
       ('OD167 Q1 63 mm', 'Al Q1 L63.0 N20 4x1.63 NO20', C['aqua']),
       ('OD167 Q2 70 mm', 'Al Q2 L70.0 N18 4x1.70 NO20', C['green'])]
OPS = [(9000, None, 'OP1'), (9339, 3.24, 'OP2'), (2000, 44.34, 'OP3')]


def design_of(label):
    if label.startswith('Copper'):
        return A.baseline()
    p = label.split()
    ns, d = p[4].split('x')
    return A.Design('Al', p[1], float(p[2][1:]), int(p[3][1:]), int(ns), float(d), lam=p[5] if len(p) > 5 else 'M235-35A')


def od(sv):
    return AF.DESIGNS[sv].get('od', 160.0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches='tight'); plt.close(fig)


def eff_maps(data):
    SP = np.arange(250, 12001, 125.0); TQ = np.arange(0.5, 56.01, 0.5)
    res = {}
    for name, lab, col in KEY:
        env, mp = design_of(lab).run(SP, TQ, npts=141)
        res[name] = (np.array([e['Tsh'] if e else np.nan for e in env]), np.array(mp['eff']))
    tb, eb = res['Copper today']
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cs = ax.contourf(SP, TQ, eb, levels=[60, 70, 80, 85, 88, 90, 92, 93, 94, 95, 96], cmap='viridis', extend='both')
    ax.contour(SP, TQ, eb, levels=[85, 90, 93, 95], colors='w', linewidths=0.5)
    ax.plot(SP, tb, color=C['ink'], lw=1.6)
    for n, T, l in OPS:
        T = T if T else np.interp(n, SP, tb)
        ax.plot(n, T, 'o', ms=6, mfc='white', mec=C['ink']); ax.annotate(l, (n, T), xytext=(6, 4), textcoords='offset points', fontsize=8)
    fig.colorbar(cs, ax=ax, label='efficiency (%)'); ax.set_xlabel('speed (rpm)'); ax.set_ylabel('shaft torque (NÂ·m)')
    ax.set_title('Copper today: efficiency map (model)', loc='left', fontsize=10)
    save(fig, 'effmap_copper.png')
    names = [k for k, *_ in KEY[1:]]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, nm in zip(axs.ravel(), names):
        t, e = res[nm]
        de = e - eb
        cs = ax.contourf(SP, TQ, de, levels=np.arange(-4, 4.01, 0.5), cmap='RdBu', extend='both')
        ax.contour(SP, TQ, de, levels=[0], colors=C['ink'], linewidths=0.8)
        ax.plot(SP, tb, color=C['cu'], lw=1.4, label='copper limit'); ax.plot(SP, t, color=C['ink'], lw=1.4, ls='--', label='Al limit')
        for n, T, l in OPS:
            T = T if T else np.interp(n, SP, tb)
            ax.plot(n, T, 'o', ms=5, mfc='white', mec=C['ink'])
        ax.set_title(nm, loc='left', fontsize=9.5)
    axs[0, 0].legend(loc='upper right', fontsize=7.5)
    axs[1, 2].axis('off')
    for ax in axs.ravel()[:5]:
        ax.set_xticks([0, 3000, 6000, 9000, 12000])
    for ax in axs[1]:
        ax.set_xlabel('speed (rpm)')
    for ax in axs[:, 0]:
        ax.set_ylabel('torque (NÂ·m)')
    fig.colorbar(cs, ax=axs, shrink=0.85, label='efficiency change vs copper (points)')
    save(fig, 'effmap_delta.png')
    for i, nm in enumerate(names):                      # one map per design for its own report page
        t, e = res[nm]; de = e - eb
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        cs = ax.contourf(SP, TQ, de, levels=np.arange(-4, 4.01, 0.5), cmap='RdBu', extend='both')
        ax.contour(SP, TQ, de, levels=[0], colors=C['ink'], linewidths=0.8)
        ax.plot(SP, tb, color=C['cu'], lw=1.4, label='copper limit'); ax.plot(SP, t, color=C['ink'], lw=1.4, ls='--', label='aluminium limit')
        for n, T, l in OPS:
            T = T if T else np.interp(n, SP, tb)
            ax.plot(n, T, 'o', ms=5, mfc='white', mec=C['ink']); ax.annotate(l, (n, T), xytext=(5, 4), textcoords='offset points', fontsize=7.5)
        ax.set_xlabel('speed (rpm)'); ax.set_ylabel('torque (N·m)'); ax.legend(fontsize=7.5, loc='upper right')
        fig.colorbar(cs, ax=ax, label='efficiency change vs copper (points)')
        ax.set_title(nm + ': efficiency change vs copper', loc='left', fontsize=9.5)
        save(fig, 'effdelta_%d.png' % (i + 1))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for name, lab, col in KEY:
        t, e = res[name]
        axs[0].plot(SP, t, color=col, lw=1.8 if name.startswith('Copper') else 1.3, label=name)
        axs[1].plot(SP, t / tb, color=col, lw=1.3, label=name)
    axs[0].set_xlabel('speed (rpm)'); axs[0].set_ylabel('max shaft torque (NÂ·m)'); axs[0].legend(fontsize=7.5)
    axs[1].axhline(0.985, color=C['red'], lw=0.8, ls=':'); axs[1].text(400, 0.9865, 'limit 0.985', color=C['red'], fontsize=7.5)
    axs[1].set_xlabel('speed (rpm)'); axs[1].set_ylabel('torque ratio vs copper'); axs[1].set_ylim(0.95, 1.12)
    axs[0].set_title('Torque-speed limit (52 V, 360 A)', loc='left', fontsize=10); axs[1].set_title('Ratio to copper', loc='left', fontsize=10)
    save(fig, 'envelopes.png')
    data['env'] = {nm: dict(speed=SP.tolist(), T=[None if np.isnan(v) else round(float(v), 3) for v in res[nm][0]]) for nm in res}


def loss_bars(R, data):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    parts = [('pcu', 'winding DC', C['orange']), ('pac', 'winding AC', C['yellow']), ('pfe', 'iron', C['blue']), ('pme', 'bearing + windage', C['ink2'])]
    data['op_losses'] = {}
    for ax, op in zip(axs, ('OP1', 'OP2', 'OP3')):
        y = np.arange(len(KEY))
        left = np.zeros(len(KEY))
        for k, lab, col in parts:
            v = np.array([(R[l]['ops'][op] or {}).get(k, 0) for _, l, _ in KEY])
            ax.barh(y, v, left=left, color=col, height=0.62, label=lab, edgecolor='white', linewidth=1)
            left += v
        for i, (nm, l, _) in enumerate(KEY):
            o = R[l]['ops'][op]
            ax.text(left[i] + 8, i, '%.0f W Â· %.2f %%' % (o['loss'], o['eff']), va='center', fontsize=7.5, color=C['ink'])
            data['op_losses'].setdefault(nm, {})[op] = {k: round(o[k], 2) for k in ('pcu', 'pac', 'pfe', 'pme', 'loss', 'eff', 'Tsh', 'I_line')}
        ax.set_yticks(y); ax.set_yticklabels([k for k, *_ in KEY] if op == 'OP1' else [], fontsize=8); ax.invert_yaxis()
        ax.set_xlim(0, left.max() * 1.45); ax.set_xlabel('loss (W)')
        ax.set_title({'OP1': 'OP1 9000 rpm, max power', 'OP2': 'OP2 9339 rpm, 3.24 NÂ·m', 'OP3': 'OP3 2000 rpm, 44.3 NÂ·m'}[op], loc='left', fontsize=9.5)
    axs[0].legend(fontsize=7.5, loc='lower right')
    save(fig, 'op_losses.png')


def evals():
    return {**json.load(open(os.path.join(H, 'alu_v2_eval.json'))), **json.load(open(os.path.join(H, 'alu_od160_eval.json')))}


def feasible(r):
    return (r['env_ratio_at'].get('2000', 0) >= 0.999 and r['env_ratio_min'] >= 0.985 and r['dEta_ops']['OP1'] is not None
            and r['dEta_ops']['OP1'] >= 0 and r['dEta_ops']['OP2'] >= 0)


def tradeoff(data):
    E = evals()
    pts = []
    for k, x in E.items():
        r = x['rec']
        if k.startswith('Copper') or not feasible(r):
            continue
        pts.append(dict(label=k, od=od(r['sv']), sv=r['sv'], L=r['L'], Nph=r['Nph'], m=r['masses']['total'], c=x['cost']['total'], op1=r['dEta_ops']['OP1'],
                        op2d=r['dEta_ops']['OP2'], op3=r['dEta_ops']['OP3'], op2=r['op_loss']['OP2'], op3w=r['op_loss']['OP3'], wm=x['dRange']['WMTC Class 1'],
                        te=x['dRange']['Telematics speed trace'], J=x['J360'], env=r['env_ratio_min'], t2000=r['env_ratio_at'].get('2000'), lam=r.get('lam')))
    data['feasible'] = pts
    cu = E['Copper baseline (SWG 11, 63 mm)']
    mcu, ccu = cu['rec']['masses']['total'], cu['cost']['total']
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    spec = [('m', 'wm', 'active mass (kg)', 'range change, WMTC Class 1 (%)'), ('m', 'op3', 'active mass (kg)', 'efficiency change at OP3 (points)'),
            ('c', 'te', 'material cost (USD)', 'range change, telematics trace (%)')]
    keyl = {l: n for n, l, _ in KEY}
    for ax, (xk, yk, xl, yl) in zip(axs, spec):
        for o, col in OD_COL.items():
            s = [p for p in pts if p['od'] == o]
            ax.scatter([p[xk] for p in s], [p[yk] for p in s], s=16, color=col, alpha=0.75, edgecolors='white', linewidths=0.4, label='OD %.0f mm (%d designs)' % (o, len(s)))
        ax.scatter([mcu if xk == 'm' else ccu], [0], s=90, marker='*', color=C['cu'], edgecolors=C['ink'], linewidths=0.5, label='copper today', zorder=5)
        for p in pts:
            if p['label'] in keyl:
                ax.annotate(keyl[p['label']].split(' ', 1)[1], (p[xk], p[yk]), xytext=(5, 4), textcoords='offset points', fontsize=7, color=C['ink'])
                ax.scatter([p[xk]], [p[yk]], s=40, facecolors='none', edgecolors=C['ink'], linewidths=1, zorder=6)
        ax.axhline(0, color=C['ink2'], lw=0.8); ax.set_xlabel(xl); ax.set_ylabel(yl)
    axs[0].legend(fontsize=7.5, loc='lower left')
    save(fig, 'tradeoff_scatter.png')
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
    for o, col in OD_COL.items():
        s = [p for p in pts if p['od'] == o]
        Ls = sorted(set(p['L'] for p in s))
        axs[0].plot(Ls, [max(p['wm'] for p in s if p['L'] == L) for L in Ls], 'o-', color=col, ms=4, lw=1.4, label='OD %.0f' % o)
        axs[1].plot(Ls, [max(p['op3'] for p in s if p['L'] == L) for L in Ls], 'o-', color=col, ms=4, lw=1.4, label='OD %.0f' % o)
    for ax in axs:
        ax.axhline(0, color=C['ink2'], lw=0.8); ax.set_xlabel('stack length (mm)')
    axs[0].set_ylabel('best WMTC range change (%)'); axs[0].legend(fontsize=8)
    axs[1].set_ylabel('best OP3 efficiency change (points)')
    axs[0].set_title('Range: best design at each stack length', loc='left', fontsize=9.5)
    axs[1].set_title('OP3 efficiency: best design at each stack length', loc='left', fontsize=9.5)
    save(fig, 'stack_effect.png')


def cost_bars(data):
    E = evals()
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(KEY)); left = np.zeros(len(KEY))
    data['cost'] = {}
    for k, lab, col in (('wire', 'winding wire', C['orange']), ('steel', 'electrical steel (net of scrap)', C['blue']), ('magnet', 'ferrite magnets', C['ink2'])):
        v = np.array([E[l]['cost'][k] for _, l, _ in KEY])
        ax.barh(y, v, left=left, color=col, height=0.6, label=lab, edgecolor='white'); left += v
    for i, (nm, l, _) in enumerate(KEY):
        x = E[l]
        data['cost'][nm] = dict(**{k: round(v, 2) for k, v in x['cost'].items()}, masses={k: round(v, 3) for k, v in x['rec']['masses'].items()})
        ax.text(left[i] + 0.8, i, '$%.1f Â· %.2f kg' % (x['cost']['total'], x['rec']['masses']['total']), va='center', fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([k for k, *_ in KEY]); ax.invert_yaxis(); ax.set_xlim(0, left.max() * 1.3); ax.set_xlabel('material cost per motor (USD)')
    ax.legend(fontsize=7.5, loc='lower right')
    save(fig, 'cost_bars.png')


def femm_grid(data):
    F = json.load(open(os.path.join(H, 'alu_femm.json')))
    D = F['designs']
    rows = []
    for d in D:
        g = D[d]['geo']
        rows.append(dict(design=d, od=g.get('od', 160.0), slot_depth=g['slot_depth'], tooth=g['tooth_w'], slot_open=g.get('slot_open', 5.0), tip_depth=g.get('tip_depth', 0.8),
                         inset=g.get('magnet_inset', 2.0), opening=g.get('magnet_opening', 8.5), slot_area=A.slot_area(d),
                         T_peak=D[d]['T'][2][3], T_q=D[d]['T'][3][3], T_mid=D[d]['T'][2][2], stator_mass_63=float(np.mean(D[d]['st_mass'])),
                         tooth_B99=D[d]['tooth_B99'][2][3], yoke_B99=D[d]['yoke_B99'][2][3], tooth_Bmean=D[d]['tooth_Bmean'][2][3], yoke_Bmean=D[d]['yoke_Bmean'][2][3],
                         psid0=D[d]['psid'][3][0], st_Sa_nl=D[d]['st_Sa'][3][0], st_Sa_pk=D[d]['st_Sa'][2][3], T_grid=D[d]['T']))
    data['femm_grid'] = dict(ids=F['ids'], iqs=F['iqs'], rows=rows)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(rows)); cols = [OD_COL[r['od']] for r in rows]
    t0 = rows[0]['T_peak']
    axs[0].bar(x, [100 * (r['T_peak'] / t0 - 1) for r in rows], color=cols, width=0.7)
    axs[0].set_ylabel('torque change vs copper geometry (%)'); axs[0].set_title('EM torque at id âˆ’49 A, iq 147 A (path peak), 63 mm', loc='left', fontsize=9.5)
    axs[1].bar(x, [r['slot_area'] for r in rows], color=cols, width=0.7); axs[1].axhline(rows[0]['slot_area'], color=C['cu'], lw=1)
    axs[1].set_ylabel('slot area (mmÂ²)'); axs[1].set_title('Slot area (copper line in brown)', loc='left', fontsize=9.5)
    axs[2].bar(x - 0.18, [r['tooth_B99'] for r in rows], width=0.36, color=C['ink2'], label='tooth, 99th percentile')
    axs[2].bar(x + 0.18, [r['yoke_B99'] for r in rows], width=0.36, color=C['yellow'], label='yoke, 99th percentile')
    axs[2].set_ylabel('flux density (T)'); axs[2].legend(fontsize=7.5); axs[2].set_title('Stator flux density near the peak point', loc='left', fontsize=9.5)
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels([r['design'] for r in rows], fontsize=7.5)
    save(fig, 'femm_grid.png')


def ripple_emf(data):
    import sens_analyse as SA
    V = json.load(open(os.path.join(H, 'alu_verify.json')))
    vs = [('ALU_S0', 'Copper geometry (S0)', C['cu']), ('ALU_Q1', 'Q1 OD167', C['aqua']), ('ALU_Q2', 'Q2 OD167', C['green']),
          ('ALU_R1', 'R1 OD160', C['blue']), ('ALU_R0', 'R0 OD160', C['violet'])]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    data['ripple'] = {}
    for v, nm, col in vs:
        R = [r for r in SA.cache if r['v'] == v]
        sg = 1 if np.mean([r['T'] for r in R if r['kind'] == 'tq']) > 0 else -1
        wav = {k: [sg * r['T'] for r in sorted([r for r in R if r['kind'] == k], key=lambda r: r['th'])] for k in ('rip', 'rip2', 'rip3')}
        d = SA.design(v); em = SA.emf_metrics(d['psi'])
        data['ripple'][v] = dict(name=nm, waves=wav, thd=em['thd'], h=em['h'], E1=em['E1_rms'], T360=d['T360'], gamma=d['gamma_opt'], T_gamma=d['T_gamma'], verify=V.get(v))
        for ax, k in zip(axs, ('rip', 'rip2', 'rip3')):
            w = np.array(wav[k] + wav[k][:1])
            ax.plot(np.arange(13), w, '-o', ms=3, color=col, lw=1.3, label=nm)
    for ax, t in zip(axs, ('OP1 9000 rpm, 139.7 A', 'OP2 9339 rpm, 73.7 A', 'OP3 2000 rpm, 208.8 A')):
        ax.set_xlabel('rotor position (mech deg, one 12-deg period)'); ax.set_ylabel('EM torque (NÂ·m)'); ax.set_title(t + ' line rms', loc='left', fontsize=9.5)
    axs[0].legend(fontsize=7.5)
    save(fig, 'ripple_waves.png')
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    hs = (3, 5, 7, 9, 11, 13); w = 0.16
    for i, (v, nm, col) in enumerate(vs):
        x = data['ripple'][v]
        axs[0].bar(np.arange(len(hs)) + (i - 2) * w, x['h'], width=w, color=col, label='%s (THD %.2f %%)' % (nm, x['thd']))
    axs[0].set_xticks(range(len(hs))); axs[0].set_xticklabels(['%d' % h for h in hs]); axs[0].set_xlabel('harmonic order'); axs[0].set_ylabel('% of fundamental')
    axs[0].legend(fontsize=7); axs[0].set_title('No-load back-EMF harmonics (20 Â°C)', loc='left', fontsize=9.5)
    sk = ['none', '2 deg 3step', '3 deg 3step', '4 deg 3step']
    for v, nm, col in vs:
        x = V.get(v)
        if not x:
            continue
        axs[1].plot([0, 2, 3, 4], [max(x['skew'][s]['OP1'], x['skew'][s]['OP2'], x['skew'][s]['OP3']) for s in sk], '-o', color=col, ms=4, label=nm)
    axs[1].axhline(10, color=C['red'], lw=0.8, ls=':'); axs[1].set_yscale('log'); axs[1].set_yticks([3, 5, 10, 20, 50, 100]); axs[1].set_yticklabels(['3', '5', '10', '20', '50', '100'])
    axs[1].set_xlabel('3-step rotor skew (mech deg)'); axs[1].set_ylabel('worst ripple of OP1/OP2/OP3 (%)'); axs[1].legend(fontsize=7)
    axs[1].set_title('Ripple vs step skew (10 % limit dotted)', loc='left', fontsize=9.5)
    save(fig, 'emf_skew.png')


def cycle(data):
    import tele_range as TR, wmtc_cycle as W, tele_cycle as TC
    E = evals()
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    v1 = W.trace(W.CLASSES['Class 1']); axs[0].plot(np.arange(len(v1)), v1 * 3.6, color=C['blue'], lw=0.9)
    axs[0].set_xlabel('time (s)'); axs[0].set_ylabel('vehicle speed (km/h)'); axs[0].set_title('WMTC Class 1', loc='left', fontsize=9.5)
    xs = TC.X; vt = xs[:, 2].clip(0) * 2 * math.pi * TR.V['r'] / 60 * 3.6
    axs[1].plot(np.arange(len(vt)), vt, color=C['orange'], lw=0.6); axs[1].set_xlabel('logged sample (1 s)'); axs[1].set_ylabel('vehicle speed (km/h)')
    axs[1].set_title('Telematics speed trace', loc='left', fontsize=9.5)
    for (c, (T, n, km)), col in zip(TR.CYC.items(), (C['orange'], C['blue'])):
        axs[2].scatter(n, T, s=3, alpha=0.25, color=col, label=c)
    axs[2].set_xlabel('motor speed (rpm)'); axs[2].set_ylabel('motor torque (NÂ·m)'); axs[2].legend(fontsize=7.5, markerscale=4)
    axs[2].set_title('Motor operating points (vehicle model)', loc='left', fontsize=9.5)
    save(fig, 'cycles.png')
    data['cycle'] = {nm: dict(Wh_km=E[l]['Wh_km'], dRange=E[l]['dRange']) for nm, l, _ in KEY}
    data['cycle_info'] = {c: dict(km=km, n_pts=int(len(T)), T_max=float(np.max(T)), T_min=float(np.min(T)), n_max=float(np.max(n))) for c, (T, n, km) in TR.CYC.items()}


if __name__ == '__main__':
    R = {r['label']: r for r in json.load(open(os.path.join(H, 'alu_sweep_v2.json')))}
    R.update({r['label']: r for r in json.load(open(os.path.join(H, 'alu_sweep.json'))) if r['label'].startswith('Copper baseline')})
    data = {}
    femm_grid(data); print('femm grid', flush=True)
    ripple_emf(data); print('ripple', flush=True)
    tradeoff(data); cost_bars(data); print('tradeoff/cost', flush=True)
    loss_bars(R, data); print('losses', flush=True)
    cycle(data); print('cycle', flush=True)
    eff_maps(data); print('maps', flush=True)
    data['key'] = [dict(name=n, label=l) for n, l, _ in KEY]
    data['key_records'] = {}
    for n, l, _ in KEY:
        data['key_records'][n] = {k: R[l].get(k) for k in ('label', 'sv', 'L', 'Nph', 'ns', 'd', 'fill', 'R_path', 'mlt', 'env_ratio_min', 'env_ratio_at', 'map_dEta_mean',
                                                         'map_dEta_min', 'map_dEta_p10', 'peak_eff', 'dEta_ops', 'masses', 'dmass', 'lam')}
    json.dump(data, open(os.path.join(H, 'report_full', 'report_data.json'), 'w'), indent=1, default=float)
    print('done')
