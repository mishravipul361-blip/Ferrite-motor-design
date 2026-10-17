"""Motor-CAD Lab exports (100 C) as flat arrays: envelope E[name] (19 speeds) and map M[name] (speed x torque points)."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_d = json.load(open(os.path.join(HERE, 'mcad_lab.json')))
SKIP = ('Units', 'Sheet1')


def _arr(rows):
    return np.array([[np.nan if v in (None, 'None') else float(v) for v in r] for r in rows], dtype=float)


E = {k: _arr(v)[:, 0] for k, v in _d['env'].items() if k not in SKIP and not k.startswith('Drive')}
M = {k: _arr(v) for k, v in _d['map'].items() if k not in SKIP and not k.startswith('Drive')}
UNITS = dict(_d['env']['Units'])
