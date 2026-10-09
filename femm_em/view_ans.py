"""Open a solved FEMM model (.ans) in a visible FEMM window with the |B| map; stays open until you close FEMM.
Usage: python view_ans.py calibrated_rated.ans"""
import os, sys, time
import femm

fn = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'calibrated_rated.ans')
femm.openfemm(0)
femm.main_maximize()
femm.opendocument(fn)
femm.mo_showdensityplot(1, 0, 2.0, 0, 'bmag')
femm.mo_showcontourplot(-1, 0, 0, 'real')   # -1 = FEMM default flux lines
femm.mo_zoomnatural()
try:
    while True:                 # keep the link alive; ends when FEMM is closed
        femm.mo_getprobleminfo(); time.sleep(5)
except Exception:
    pass
