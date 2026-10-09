"""Open the rated-load case solved in a visible FEMM window with the |B| map; stays open until you close FEMM."""
import os, time
import femm

fem = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v7_rated.fem')
femm.openfemm(0)
femm.main_maximize()
femm.opendocument(fem)
femm.mi_smartmesh(0)
femm.mi_analyze(0)
femm.mi_loadsolution()
femm.mo_showdensityplot(1, 0, 2.0, 0, 'bmag')
femm.mo_zoomnatural()
try:
    while True:            # keep the COM link alive; exits with an error once FEMM is closed
        femm.mo_getprobleminfo()
        time.sleep(5)
except Exception:
    pass
