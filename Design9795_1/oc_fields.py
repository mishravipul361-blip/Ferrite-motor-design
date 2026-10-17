"""Open-circuit flux densities at 100 C for comparison with Motor-CAD Output Data > Flux Densities.
Rotor stepped 0-5.5 deg mech (0.5 deg): with 12 teeth this covers every tooth-to-pole position (6 deg period).
  airgap: radial B on the mid-gap circle (720 points): mean |Br| and max |Br| (max over positions)
  tooth:  |B| averaged across the tooth width at mid tooth height, max over teeth and positions
  tip:    |B| averaged along the tooth tip at half tip depth, max over tips and positions
  yoke:   |B| averaged across the back iron on each tooth and slot centreline, max
  rotor:  |B| averaged across each pole piece at mid magnet depth, max
python oc_fields.py -> oc_fields.json"""
import json, math, os
os.environ['MAG_T'] = '100'
os.environ['SYM'] = '0'                     # probes go round the full machine: full model
import numpy as np
import backemf as M
import femm


def avg_line(p, q, n=25):
    return float(np.mean([math.hypot(*femm.mo_getb(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)) for t in np.linspace(0.04, 0.96, n)]))


if __name__ == '__main__':
    femm.openfemm(1)
    fn = os.path.join(M.HERE, 'work', 'oc_fields.fem')
    M.W['file'] = fn
    sg, g = M.build(fn, True)
    s = M.STATOR; ws = s['tooth_w'] / 2; rb = sg['rb']; ro = s['od'] / 2; hs = 180 / s['slots']
    res = dict(gap_avg=[], gap_max=[], tooth=[], tip=[], yoke=[], rotor=[])
    for th in np.arange(0, 6, 0.5):
        M.set_angle(th); femm.mi_analyze(1); femm.mi_loadsolution()
        a = np.radians(np.arange(0, 360, 0.5))
        br = [abs(np.dot(femm.mo_getb(M.R_MID * math.cos(t), M.R_MID * math.sin(t)), (math.cos(t), math.sin(t)))) for t in a]
        res['gap_avg'].append(float(np.mean(br))); res['gap_max'].append(float(np.max(br)))
        tooth, tip, yoke, rot = [], [], [], []
        for k in range(s['slots']):
            c = math.radians(90 + 360 / s['slots'] * k + hs)            # tooth centreline between slot k and k+1
            u, nrm = (math.cos(c), math.sin(c)), (-math.sin(c), math.cos(c))
            rm = (M.R_BORE + s['tip_depth'] + rb) / 2
            tooth.append(avg_line((rm * u[0] - ws * nrm[0], rm * u[1] - ws * nrm[1]), (rm * u[0] + ws * nrm[0], rm * u[1] + ws * nrm[1])))
            rt = M.R_BORE + s['tip_depth'] / 2
            half = rt * math.radians(360 / s['slots']) / 2 - s['slot_open'] / 2
            for sgn in (-1, 1):
                tip.append(avg_line((rt * u[0] + sgn * ws * nrm[0], rt * u[1] + sgn * ws * nrm[1]), (rt * u[0] + sgn * half * nrm[0], rt * u[1] + sgn * half * nrm[1])))
            for cc in (c, c + math.radians(hs)):
                uu = (math.cos(cc), math.sin(cc))
                yoke.append(avg_line((rb * uu[0], rb * uu[1]), (ro * uu[0], ro * uu[1])))
        rmid = (g['rim'] + g['rom']) / 2
        for k in range(M.ROTOR['pole_number']):
            c = math.radians(360 / M.ROTOR['pole_number'] * k + th + 180 / M.ROTOR['pole_number'])   # pole piece between magnets (magnet centrelines at 36k + th)
            w = rmid * math.radians(180 / M.ROTOR['pole_number']) - 9.2
            u, nrm = (math.cos(c), math.sin(c)), (-math.sin(c), math.cos(c))
            rot.append(avg_line((rmid * u[0] - w * nrm[0], rmid * u[1] - w * nrm[1]), (rmid * u[0] + w * nrm[0], rmid * u[1] + w * nrm[1])))
        res['tooth'].append(max(tooth)); res['tip'].append(max(tip)); res['yoke'].append(max(yoke)); res['rotor'].append(max(rot))
        femm.mo_close()
        print('th %.1f  gap avg %.4f max %.4f  tooth %.3f  tip %.3f  yoke %.3f  rotor %.3f' % (
            th, res['gap_avg'][-1], res['gap_max'][-1], res['tooth'][-1], res['tip'][-1], res['yoke'][-1], res['rotor'][-1]), flush=True)
    out = dict(airgap_avg=float(np.mean(res['gap_avg'])), airgap_max=float(np.max(res['gap_max'])), tooth_max=float(np.max(res['tooth'])),
               tip_max=float(np.max(res['tip'])), yoke_max=float(np.max(res['yoke'])), rotor_max=float(np.max(res['rotor'])), raw=res)
    json.dump(out, open(os.path.join(M.HERE, 'oc_fields.json'), 'w'), indent=1)
    print({k: round(v, 4) for k, v in out.items() if k != 'raw'})
    femm.closefemm()
