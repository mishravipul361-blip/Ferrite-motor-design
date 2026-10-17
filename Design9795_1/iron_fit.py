"""Offline test of iron-loss formulations on the saved fixed-mesh field waveforms (work/age_raw_*.npz)
against Motor-CAD's map (same speed and same id/iq).  No FEMM solves.

Formulations (all Steinmetz with Motor-CAD coefficients, 2*pi^2 on Keddy, build factor 1.6):
  all   per element, radial + tangential, every harmonic n = 1..14: Kh(nf)B_n^a + 2pi^2 Ke (nf)^2 B_n^2
  fund  same, fundamental only
  peak  per element and component, peak value over the cycle (no harmonic split): Kh f Bpk^a + 2pi^2 Ke f^2 Bpk^2
  ml    hysteresis = peak with Motor-CAD's minor-loop (Lavers) factor 1 + Kc/Bpk * sum(dB_minor), Kc = 0.65;
        eddy = 'all' (harmonic sum, i.e. the dB/dt content).  This is the v3 Lab model.
Writes iron_variants.json and prints the error of each formulation vs Motor-CAD.
"""
import glob, json, math, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mcad_lab'))
from load_mcad import M as MC
import lab_map as L
import lab_model as LM

H = L.H
KH, KE, A, BF, NPC = L.KH, L.KE, L.ALPHA, L.BF, 30
KC = L.LOSS['iron_loss']['Kc_minor_loop']


def variants(fn):
    d = np.load(fn)
    Bs, st, m, tooth, rmap = d['Bs'].astype(float), d['st'], d['mass'], d['tooth'], d['rmap']
    rsg = d['rsg'] if 'rsg' in d.files else 1.0               # half model: image below the cut has the sign flipped
    s = np.concatenate([Bs[:, :, st], -Bs[:, :, st]], 0)
    Br = Bs[:, :, ~st]
    r = np.concatenate([Br, -Br[:, :, rmap] * rsg], 0)
    out = {}
    for part, B, mm in (('stator', s, m[st]), ('rotor', r, m[~st])):
        Bn = np.abs(np.fft.rfft(B, axis=0)) / NPC * 2
        n = np.arange(1, 15)[:, None, None]
        out[part + '_all_H'] = float(np.sum(mm * KH * n * Bn[1:15] ** A)); out[part + '_all_E'] = float(np.sum(mm * 2 * math.pi ** 2 * KE * n ** 2 * Bn[1:15] ** 2))
        out[part + '_fund_H'] = float(np.sum(mm * KH * Bn[1] ** A)); out[part + '_fund_E'] = float(np.sum(mm * 2 * math.pi ** 2 * KE * Bn[1] ** 2))
        Bpk = np.abs(B - B.mean(0)).max(0)
        out[part + '_peak_H'] = float(np.sum(mm * KH * Bpk ** A)); out[part + '_peak_E'] = float(np.sum(mm * 2 * math.pi ** 2 * KE * Bpk ** 2))
        tv = np.abs(np.diff(np.concatenate([B, B[:1]], 0), axis=0)).sum(0)        # total variation over the cycle
        minor = np.maximum(tv - 4 * Bpk, 0) / 2                                    # sum of the minor-loop swings
        k = 1 + KC * minor / np.maximum(Bpk, 1e-6)
        out[part + '_ml_H'] = float(np.sum(mm * KH * Bpk ** A * k)); out[part + '_ml_E'] = out[part + '_all_E']
    return out


pts = {}
for fn in glob.glob(os.path.join(H, 'work', 'age_raw_*.npz')):
    a, b = map(int, re.findall(r'age_raw_(-?\d+)_(-?\d+)', fn)[0])
    pts[(a, b)] = variants(fn)
ids = sorted({k[0] for k in pts}); iqs = sorted({k[1] for k in pts})
print('points', len(pts), 'id', ids, 'iq', iqs)
if len(pts) < 12:
    print('waiting for the full grid'); sys.exit()
keys = list(next(iter(pts.values())).keys())
grid = {k: np.array([[pts[(a, b)][k] for b in iqs] for a in ids]) for k in keys}
F = {k: LM.grid_quad_cubic(np.array(ids, float), np.array(iqs, float), grid[k]) for k in keys}

S = MC['Speed']; mask = (S > 0) & (MC['Shaft_Torque'] > 0.5)
Pt = np.column_stack([MC['Id_Peak'][mask] / 2, MC['Iq_Peak'][mask] / 2])
f = S[mask] * L.P / 60                         # electrical frequency, Hz
res = {}
for v in ('all', 'fund', 'peak', 'ml'):
    st = BF * (np.maximum(F['stator_%s_H' % v](Pt), 0) * f + np.maximum(F['stator_%s_E' % v](Pt), 0) * f * f)
    rt = BF * (np.maximum(F['rotor_%s_H' % v](Pt), 0) * f + np.maximum(F['rotor_%s_E' % v](Pt), 0) * f * f)
    hy = BF * (np.maximum(F['stator_%s_H' % v](Pt), 0) + np.maximum(F['rotor_%s_H' % v](Pt), 0)) * f
    ed = st + rt - hy
    row = {}
    for name, x, ref in (('stator', st, MC['Iron_Loss_Stator'][mask]), ('rotor', rt, MC['Iron_Loss_Rotor'][mask]),
                         ('hysteresis', hy, MC['Iron_Loss_Hysteresis'][mask]), ('eddy', ed, MC['Iron_Loss_Eddy'][mask]),
                         ('total', st + rt, MC['Iron_Loss'][mask])):
        row[name] = dict(mean_pct=float(100 * np.mean((x - ref) / ref)), mean_abs_pct=float(100 * np.mean(np.abs(x - ref) / ref)),
                         ratio_median=float(np.median(x / ref)))
    res[v] = row
    print('%-5s ' % v + '  '.join('%s %+.0f%% (|%.0f%%|)' % (k, r_['mean_pct'], r_['mean_abs_pct']) for k, r_ in row.items()))
json.dump(dict(ids=ids, iqs=iqs, grid={k: v.tolist() for k, v in grid.items()}, errors=res), open(os.path.join(H, 'iron_variants.json'), 'w'), indent=1)
