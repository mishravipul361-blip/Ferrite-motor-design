"""Build the rated-load case (rotor 0 deg, 360 A rms at MTPA angle) as v7_rated.fem; view it with view_rated.py."""
import json, os
import femm, femm_motor as M

HERE = M.HERE
r = json.load(open(os.path.join(HERE, 'results.json')))
fem = os.path.join(HERE, 'v7_rated.fem').replace(os.sep, '/')
femm.openfemm(1)
M.build(fem)
cur = M.phase_currents(0, r['noload']['phi'], r['gamma_sweep']['gamma_opt'], M.I_LINE_RMS)
for ph in 'RYB':
    femm.mi_modifycircprop(ph, 1, cur[ph])
femm.mi_saveas(fem)
femm.closefemm()
