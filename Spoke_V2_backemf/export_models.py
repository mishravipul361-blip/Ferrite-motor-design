"""Writes clean, solved FEMM models with readable names to FEMM_models/ (Motor-CAD B-H, fillets, rotor at 0 deg)."""
import json, math, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'FEMM_models')
CASES = [  # file name, magnet temperature, operating point
    ('Spoke_V2_12s10p_NoLoad_Magnets20C', 20, None),
    ('Spoke_V2_12s10p_NoLoad_Magnets100C', 100, None),
    ('Spoke_V2_12s10p_PeakTorque_360Arms_Magnets100C', 100, 'peak'),
]

if len(sys.argv) == 1:
    os.makedirs(OUT, exist_ok=True)
    for name, t, op in CASES:     # one process per case: magnet temperature is read at import
        env = dict(os.environ, MAG_T=str(t))
        subprocess.run([sys.executable, __file__, name, str(t), op or ''], env=env, check=True)
    sys.exit()

name, t, op = sys.argv[1], float(sys.argv[2]), sys.argv[3]
import femm
import backemf as M
femm.openfemm(1)
fem = os.path.join(OUT, name + '.fem')
M.build(fem, True)
M.set_angle(0.0)
if op == 'peak':
    import lab_map as L
    e = json.load(open(os.path.join(HERE, 'lab_100C.json')))['envelope'][0]
    ip = L.phase_currents(0.0, e['id'], e['iq'])
    for ph, teeth in M.PHASE_TEETH.items():
        for tt in teeth:
            femm.mi_modifycircprop('T%d' % tt, 1, L.SIGN[ph][tt] * ip[ph])
    print(name, 'id %.1f iq %.1f A path peak' % (e['id'], e['iq']))
femm.mi_saveas(fem)
femm.mi_analyze(1); femm.mi_loadsolution()
femm.mo_groupselectblock(1); T = femm.mo_blockintegral(22); femm.mo_clearblock()
cur = [femm.mo_getcircuitproperties('T%d' % k)[0] for k in range(1, 13)]
psi = [femm.mo_getcircuitproperties('T%d' % k)[2] for k in range(1, 13)]
print(name, 'torque %.2f N.m' % T, 'coil currents', [round(c, 1) for c in cur], 'coil psi', [round(p, 5) for p in psi])
femm.closefemm()
