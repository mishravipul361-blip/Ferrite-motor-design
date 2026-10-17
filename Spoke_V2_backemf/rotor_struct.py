"""Structural screen for near-air-gap rotor changes: centrifugal load that the shaft-side webs carry.
The rotor is open at the air gap (no outer bridge), so each pole piece + magnet hangs on the web between the pockets.
Load ~ sum(rho * area * r) over everything outside the web root radius (shaft radius + bridge_offset), per unit length.
Web thickness (Bridge_thickness) and position (bridge_offset) are fixed, so web stress scales with this load.
Mesh-based: one no-load FEMM solve per variant.  python rotor_struct.py"""
import json, math, os, sys
os.environ.setdefault('MAG_T', '100')
import numpy as np
import backemf as M
import femm

H = M.HERE
BASE_R = dict(M.ROTOR)                                # Motor-CAD original rotor (shaft 25, bridge 1.0, offset 4.0)
VARIANTS = {'copper rotor (as built)': {}, 'inset 1.5': dict(magnet_inset=1.5), 'opening 7.5': dict(magnet_opening=7.5),
            'inset 1.5 + opening 7.5': dict(magnet_inset=1.5, magnet_opening=7.5), 'inset 1.5 + opening 8.0': dict(magnet_inset=1.5, magnet_opening=8.0),
            'inset 1.5 + Z 5.5': dict(magnet_inset=1.5, magnet_Z=5.5), 'inset 1.5 + opening 7.5 + Z 5.5': dict(magnet_inset=1.5, magnet_opening=7.5, magnet_Z=5.5)}


def load(fn):
    s = open(fn).read().split('\n')
    l0 = next(i for i, l in enumerate(s) if l.startswith('[NumBlockLabels]'))
    labs = [l.split() for l in s[l0 + 1:l0 + 1 + int(s[l0].split('=')[1])]]
    i0 = next(i for i, l in enumerate(s) if l.startswith('[BlockProps]'))
    names = [l.split('=')[1].strip().strip('"') for l in s[i0:l0] if l.strip().startswith('<BlockName>')]
    k = s.index('[Solution]'); n = int(s[k + 1])
    nodes = np.array([[float(x) for x in l.split()[:2]] for l in s[k + 2:k + 2 + n]])
    ne = int(s[k + 2 + n])
    el = np.array([[int(x) for x in l.split()[:4]] for l in s[k + 3 + n:k + 3 + n + ne]])
    p = nodes[el[:, :3]]
    area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    r = np.hypot(p[:, :, 0].mean(1), p[:, :, 1].mean(1))
    mat = np.array([names[int(labs[l][2]) - 1] for l in el[:, 3]])
    return area, r, mat


if __name__ == '__main__':
    femm.openfemm(1)
    out = {}
    fn = os.path.join(H, 'work', 'rotor_struct.fem')
    for name, ch in VARIANTS.items():
        M.ROTOR.clear(); M.ROTOR.update(BASE_R, **ch)
        M.W['file'] = fn
        M.build(fn, True)
        femm.mi_analyze(1); femm.mi_close() if False else None
        area, r, mat = load(fn.replace('.fem', '.ans'))
        r0 = M.ROTOR['shaft_diameter'] / 2 + M.ROTOR['bridge_offset']
        rr = M.ROTOR['rotor_diameter'] / 2
        lam = (mat == 'M235-35A') & (r < rr + 0.05) & (r > r0)
        mag = mat == 'Magnet'
        F = float(np.sum(area[lam] * r[lam]) * 7650 * 0.97 + np.sum(area[mag] * r[mag]) * 4900)      # kg/m3 * mm^3/mm -> relative
        out[name] = dict(change=ch, F=F, mag_area=float(area[mag].sum()), pole_iron_area=float(area[lam].sum()))
        print('%-34s load %.4e  magnet area %.1f mm2  pole iron area %.1f mm2' % (name, F, out[name]['mag_area'], out[name]['pole_iron_area']), flush=True)
    b = out['copper rotor (as built)']['F']
    for k, v in out.items():
        v['F_ratio'] = v['F'] / b
        print('%-34s web load vs copper rotor: %+.2f %%' % (k, 100 * (v['F_ratio'] - 1)))
    json.dump(out, open(os.path.join(H, 'rotor_struct.json'), 'w'), indent=1)
    femm.closefemm()
