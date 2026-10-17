"""Evaluate the sensitivity study (work/sens_cache.json) -> sens_results.json
  T360     MTPA EM torque at 360 A rms (2000 rpm objective), parabola over gamma 17/23/29
  loss_op2 total loss at 9339 rpm, 3.24 N.m shaft (local d-q model at the 49.4 V limit; Cu DC + AC + iron + mech)
  thd      no-load back-EMF THD (20 C, 30 points/cycle, harmonics 2..13, Motor-CAD definition)
  rip_op1  torque ripple pk-pk / mean at 9000 rpm max current (worst full-load point)
Skew factors are applied analytically from the waveform harmonics."""
import json, math, os
import numpy as np
import lab_map as L
import sens as S

H = L.H
P = 5
cache = json.load(open(os.path.join(H, 'work', 'sens_cache.json')))
_keys = {r['key'] for r in cache}
import glob as _g
for _fn in _g.glob(os.path.join(H, 'work', 'sens_solves_*.jsonl')):     # per-solve saves of the running study
    for _l in open(_fn):
        try:
            _r = json.loads(_l)
        except ValueError:
            continue
        if _r['key'] not in _keys:
            cache.append(_r); _keys.add(_r['key'])
OPC = json.load(open(os.path.join(H, 'op_compare.json')))
BASE_OP2 = [OPC[k]['same_point'] for k in OPC if k.startswith('OP2')][0]
N2, T2 = 9339.0, 3.24
W2 = N2 * 2 * math.pi / 60; F2 = N2 * P / 60


def skew_factor(order_el, skew_mech_deg, steps=None):
    """Skew factor for an electrical harmonic of order n (n*p cycles per mech rev): continuous or step skew."""
    x = math.radians(order_el * P * skew_mech_deg)
    if skew_mech_deg == 0 or order_el == 0:
        return 1.0
    if steps:
        offs = (np.arange(steps) - (steps - 1) / 2) * skew_mech_deg / steps   # slice centres
        return float(np.mean(np.cos(np.radians(order_el * P * offs))))
    return math.sin(x / 2) / (x / 2)


def emf_metrics(psi, skew=0.0, steps=None):
    """psi: 15 half-cycle samples of phase-1 path flux (2.4 deg mech) -> EMF at 1000 rpm, Motor-CAD style forward difference."""
    p = np.concatenate([psi, -np.array(psi)])
    c = np.fft.rfft(p) / 30
    n = np.arange(len(c))
    c = c * np.array([skew_factor(k, skew, steps) for k in n])
    p = np.fft.irfft(c * 30, 30)
    we = P * 1000 * 2 * math.pi / 60; dt = 2 * math.pi / 30 / we
    e = -(np.roll(p, -1) - p) / dt
    h = np.abs(np.fft.rfft(e)) / 30 * 2
    return dict(E1_rms=float(h[1] / math.sqrt(2)), thd=float(100 * math.sqrt(np.sum(h[2:14] ** 2)) / h[1]),
                h=[float(100 * h[k] / h[1]) for k in (3, 5, 7, 9, 11, 13)])


def ripple_metrics(T, step_deg, skew=0.0, steps=None):
    """Torque samples over one 12-deg mech period. Torque harmonic m (period 12/m deg) = electrical order 6m."""
    T = np.asarray(T, float); N = len(T)
    c = np.fft.rfft(T)
    c = c * np.array([skew_factor(6 * m, skew, steps) if m > 0 else 1.0 for m in range(len(c))])
    Tw = np.fft.irfft(c, N)
    mean = float(Tw.mean()) * skew_factor(1, skew, steps)      # skewed slices see shifted current angles: mean ~ x k1
    return dict(mean=mean, pp=float(np.ptp(Tw)), pct=float(100 * np.ptp(Tw) / abs(mean)))


def design(v):
    R = [r for r in cache if r['v'] == v]
    if len(R) < 37:
        return None
    sg = 1 if np.mean([r['T'] for r in R if r['kind'] == 'tq']) > 0 else -1
    # torque vs gamma
    gs = (17.0, 23.0, 29.0)
    Tg = [np.mean([sg * r['T'] for r in R if r['kind'] == 'tq' and r['a'] == g]) for g in gs]
    a, b, c = np.polyfit(gs, Tg, 2)
    gopt = float(np.clip(-b / (2 * a), 10, 40)) if a < 0 else float(gs[int(np.argmax(Tg))])
    T360 = float(np.polyval([a, b, c], gopt))
    # ripple OP1
    rip = sorted([r for r in R if r['kind'] == 'rip'], key=lambda r: r['th'])
    Trip = [sg * r['T'] for r in rip]
    # THD
    nl = sorted([r for r in R if r['kind'] == 'nl'], key=lambda r: r['th'])
    psi = [r['psi1'] for r in nl]
    # OP2 local model
    o = {k: [r for r in R if r['kind'] == 'op2' and r['a'] == k] for k in (0, 1)}
    av = lambda k, q: float(np.mean([r[q] for r in o[k]]))
    id0, iq0 = o[0][0]['id'], o[0][0]['iq']; id1 = o[1][0]['id']
    Ld = (av(1, 'psid') - av(0, 'psid')) / (id1 - id0); dpq = (av(1, 'psiq') - av(0, 'psiq')) / (id1 - id0)
    lin = lambda k0, k1, q, x: av(0, q) + (av(1, q) - av(0, q)) * (x - id0) / (id1 - id0)
    T0 = sg * av(0, 'T'); Tf0 = 3 * P * (av(0, 'psid') * iq0 - av(0, 'psiq') * id0); kT = T0 / Tf0
    ids = np.linspace(id0 - 40, id0 + 30, 281); iqs = np.linspace(1, 40, 157)
    I_d, I_q = np.meshgrid(ids, iqs, indexing='ij')
    psd = av(0, 'psid') + Ld * (I_d - id0); psq = (av(0, 'psiq') + dpq * (I_d - id0)) * I_q / iq0
    Tem = kT * 3 * P * (psd * I_q - psq * I_d)
    vd = L.R_PATH * I_d - P * W2 * (psq + L.L_EW_PATH * I_q); vq = L.R_PATH * I_q + P * W2 * (psd + L.L_EW_PATH * I_d)
    V = np.hypot(vd, vq)
    # losses
    sa = lin(0, 1, 'st_Sa', I_d); s2 = lin(0, 1, 'st_S2', I_d); cub = np.maximum(lin(0, 1, 'cu_B2', I_d), 0)
    snap = L.BF * (L.KH * F2 * sa + 2 * math.pi ** 2 * L.KE * F2 ** 2 * s2)
    pcu = 3 * L.R_PATH * (I_d ** 2 + I_q ** 2)
    pac = L.AC_ADJ * L.N_COND * math.pi * 0.063 * L.D_CU ** 4 * (2 * math.pi * F2) ** 2 * 2 * cub / (128 * L.RHO20 * (1 + L.ALPHA_CU * 80))
    return dict(v=v, T360=T360, gamma_opt=gopt, T_gamma=Tg, Trip=Trip, psi=psi, snap=snap, pcu=pcu, pac=pac, Tem=Tem, V=V, I_d=I_d, I_q=I_q)


def solve_op2(d, iron_cal):
    """min-current point with V <= 49.4 V and shaft torque = 3.24 N.m; iron = snapshot x calibration (baseline -> FEMM v2)."""
    pfe = d['snap'] * iron_cal
    tsh = d['Tem'] - (pfe + L.mech_loss(N2)) / W2
    ok = (d['V'] <= L.V_MAX) & (np.abs(tsh - T2) <= 0.03)
    if not ok.any():
        return None
    I = np.hypot(d['I_d'], d['I_q']); k = np.flatnonzero(ok.ravel())[np.argmin(I.ravel()[ok.ravel()])]
    g = lambda a: float(np.ravel(a)[k])
    loss = g(d['pcu']) + g(d['pac']) + g(pfe) + L.mech_loss(N2)
    return dict(loss=loss, pcu=g(d['pcu']), pac=g(d['pac']), pfe=g(pfe), id=g(d['I_d']), iq=g(d['I_q']),
                I_line=math.hypot(g(d['I_d']), g(d['I_q'])) * 2 * math.sqrt(3) / math.sqrt(2), eff=100 * T2 * W2 / (T2 * W2 + loss))


def main():
    res = {}
    base = design('baseline')
    # calibrate snapshot iron loss so the baseline OP2 iron equals the tuned FEMM v2 model (259.9 W)
    k0 = np.unravel_index(np.argmin(np.hypot(base['I_d'] - BASE_OP2['id_path'], base['I_q'] - BASE_OP2['iq_path'])), base['I_d'].shape)
    iron_cal = BASE_OP2['Iron'] / float(base['snap'][k0])
    for v, _, _ in [x for x in S.VARIANTS if x[0] not in [c[0] for c in S.CANDIDATES]]:
        d = design(v)
        if d is None:
            continue
        op2 = solve_op2(d, iron_cal)
        em = emf_metrics(d['psi']); rp = ripple_metrics(d['Trip'], 1.0)
        res[v] = dict(T360=d['T360'], gamma_opt=d['gamma_opt'], T_gamma=d['T_gamma'], op2=op2, thd=em['thd'], E1_rms=em['E1_rms'],
                      emf_h=em['h'], rip_op1_pct=rp['pct'], rip_op1_pp=rp['pp'], T_op1=rp['mean'], Trip=d['Trip'], psi=d['psi'])
        print('%-22s T360 %6.2f (g %4.1f)  loss_op2 %6.1f  THD %5.2f%%  ripOP1 %5.1f%%  E1 %.3f V' % (
            v, d['T360'], d['gamma_opt'], op2['loss'] if op2 else float('nan'), em['thd'], rp['pct'], em['E1_rms']))
    # skew study on baseline waveforms (fine 0.5 deg data at OP1/2/3 and the 30-pt EMF)
    rp = json.load(open(os.path.join(H, 'op_ripple.json')))
    sk = {}
    for s in np.arange(0, 6.01, 0.5):
        row = {'skew': float(s)}
        for op in ('OP1', 'OP2', 'OP3'):
            for mode, st in (('cont', None), ('3step', 3)):
                m = ripple_metrics(rp[op]['T'], 0.5, s, st)
                row['%s_%s_pct' % (op, mode)] = m['pct']; row['%s_%s_mean' % (op, mode)] = m['mean']
        for mode, st in (('cont', None), ('3step', 3)):
            e = emf_metrics(res['baseline']['psi'], s, st)
            row['thd_%s' % mode] = e['thd']; row['E1_%s' % mode] = e['E1_rms']
        sk[float(s)] = row
    # combined candidates: same metrics + ripple at OP2 / OP3 and skew on their own waveforms
    cand = {}
    for v, rot, sta in S.CANDIDATES:
        d = design(v)
        Rv = [r for r in cache if r['v'] == v]
        if d is None or len([r for r in Rv if r['kind'] in ('rip2', 'rip3')]) < 24:
            continue
        sg = 1 if np.mean([r['T'] for r in Rv if r['kind'] == 'tq']) > 0 else -1
        wav = {k: [sg * r['T'] for r in sorted([r for r in Rv if r['kind'] == k], key=lambda r: r['th'])] for k in ('rip', 'rip2', 'rip3')}
        op2 = solve_op2(d, iron_cal); em = emf_metrics(d['psi'])
        skc = {}
        for s in np.arange(0, 6.01, 0.5):
            row = {}
            for mode, st in (('cont', None), ('3step', 3)):
                for k, nm in (('rip', 'OP1'), ('rip2', 'OP2'), ('rip3', 'OP3')):
                    row['%s_%s_pct' % (nm, mode)] = ripple_metrics(wav[k], 1.0, s, st)['pct']
                e = emf_metrics(d['psi'], s, st); row['thd_%s' % mode] = e['thd']; row['k1_%s' % mode] = skew_factor(1, s, st)
            skc[float(s)] = row
        cand[v] = dict(rotor=rot, stator=sta, T360=d['T360'], gamma_opt=d['gamma_opt'], op2=op2, thd=em['thd'], E1_rms=em['E1_rms'], emf_h=em['h'],
                       rip_op1_pct=ripple_metrics(wav['rip'], 1.0)['pct'], rip_op2_pct=ripple_metrics(wav['rip2'], 1.0)['pct'],
                       rip_op3_pct=ripple_metrics(wav['rip3'], 1.0)['pct'], waves=wav, skew=skc)
        print('%-4s T360 %.2f  loss %.1f  THD %.2f  ripple OP1/OP2/OP3 %.1f / %.1f / %.1f %%' % (v, d['T360'], op2['loss'] if op2 else float('nan'), em['thd'],
              cand[v]['rip_op1_pct'], cand[v]['rip_op2_pct'], cand[v]['rip_op3_pct']))
    json.dump(dict(designs=res, skew=sk, iron_cal=iron_cal, candidates=cand), open(os.path.join(H, 'sens_results.json'), 'w'), indent=1)
    print('skew (baseline): deg  THD  ripOP1  ripOP2  ripOP3  [continuous | 3-step]')
    for s, r in sk.items():
        if s in (0, 2, 3, 4, 5, 6):
            print('  %.1f  %5.2f|%5.2f  %5.1f|%5.1f  %5.1f|%5.1f  %5.1f|%5.1f  torque x%.3f' % (s, r['thd_cont'], r['thd_3step'], r['OP1_cont_pct'], r['OP1_3step_pct'],
                  r['OP2_cont_pct'], r['OP2_3step_pct'], r['OP3_cont_pct'], r['OP3_3step_pct'], r['E1_cont'] / sk[0.0]['E1_cont']))


if __name__ == '__main__':
    main()
