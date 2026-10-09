"""3D busbar current-flow animation, v7 (delta, 2 parallel x 2 series per phase).
Dimensions from CIRCUIT_v7.txt / BUSBAR_v7.stp. Writes videos/BUSBAR_current_flow_v7_3D.mp4 (run from busbar/).
  python busbar_animation_3d.py            -> full video
  python busbar_animation_3d.py 20 45 80   -> preview stills at those seconds
"""
import sys, subprocess
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

# ------------------------------------------------------------------ design data (v7)
I_PH = 208.0
I_PK = I_PH * np.sqrt(2)
R_IN, R_OUT, R_MID = 59.0, 69.0, 64.0          # ring band 10 x 3 mm
R_POST = 71.75                                  # tab posts / coil leads
Z0 = {"LINK": 54.0, "L2": 58.0, "L3": 62.0, "L1": 66.0}       # bottom of each 3 mm level
IDX = {"LINK": 0, "L2": 1, "L3": 2, "L1": 3}
T_BAR = 3.0
RING_ARC = {"L1": (186.8, 210.0), "L2": (306.8, 210.0), "L3": (66.8, 210.0)}
TERM = {"L1": 15.0, "L2": 45.0, "L3": 75.0}
# phase: (from ring, to ring, angle, [(tooth on 'from' ring, tooth on 'to' ring)])
PHASES = {"R": ("L1", "L2", 15.0, [(12, 1), (7, 6)]),
          "Y": ("L2", "L3", -105.0, [(4, 5), (11, 10)]),
          "B": ("L3", "L1", 135.0, [(3, 2), (8, 9)])}
EXPLODE = 12.0                                   # extra mm per level when exploded

tooth = lambda k: (k - 1) * 30.0
F = lambda k: (tooth(k) + 6.8) % 360             # finish lead -> ring
S = lambda k: (tooth(k) - 5.6) % 360             # start lead  -> series link
W = lambda e: 73.0 + 3 * e                       # weld plane
TTOP = lambda e: 92.0 + 3 * e                    # terminal top
zb = lambda lv, e: Z0[lv] + IDX[lv] * e
ztop = lambda lv, e: zb(lv, e) + T_BAR

hexc = lambda h: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
LV_COL = {"L1": "#ff9f43", "L2": "#22c3d6", "L3": "#c77dff", "LINK": "#c8ccd4"}
PH_COL = {"R": "#ff4d5e", "Y": "#ffd23f", "B": "#4d8dff"}
LV_NAME = {"LINK": "LINK level", "L2": "ring L2", "L3": "ring L3", "L1": "ring L1"}
LV_Z = {"LINK": 55.5, "L2": 59.5, "L3": 63.5, "L1": 67.5}

# ------------------------------------------------------------------ mesh helpers
def cyl(r, a, z):
    a = np.radians(a)
    return np.array([r * np.cos(a), r * np.sin(a), z])


def polydata(pts, polys):
    p = vtk.vtkPoints()
    for q in pts: p.InsertNextPoint(*q)
    c = vtk.vtkCellArray()
    for f in polys:
        c.InsertNextCell(len(f))
        for i in f: c.InsertCellPoint(i)
    d = vtk.vtkPolyData(); d.SetPoints(p); d.SetPolys(c)
    return d


def sector(r0, r1, a0, a1, z0, z1):
    """annular sector box; a1 may be < a0 (handled as ccw a0 -> a1)."""
    span = (a1 - a0) % 360 or 360
    n = max(2, int(span / 2) + 2)
    angs = a0 + np.linspace(0, span, n)
    pts, polys = [], []
    for a in angs:
        pts += [cyl(r0, a, z0), cyl(r1, a, z0), cyl(r1, a, z1), cyl(r0, a, z1)]
    for i in range(n - 1):
        b, c = 4 * i, 4 * (i + 1)
        for k in range(4):
            polys.append([b + k, b + (k + 1) % 4, c + (k + 1) % 4, c + k])
    polys += [[3, 2, 1, 0], [4 * (n - 1), 4 * (n - 1) + 1, 4 * (n - 1) + 2, 4 * (n - 1) + 3]]
    return polydata(pts, polys)


def post(a, z0, z1, r0=71.0, r1=72.5, w=6.0):
    da = np.degrees(w / 2 / R_POST)
    return sector(r0, r1, a - da, a + da, z0, z1)


def stub(a, z0, r0=R_OUT, r1=72.5, w=6.0):
    da = np.degrees(w / 2 / R_POST)
    return sector(r0, r1, a - da, a + da, z0, z0 + T_BAR)


def tube(pts, r=1.25):
    p = vtk.vtkPoints()
    for q in pts: p.InsertNextPoint(*q)
    line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(len(pts))
    for i in range(len(pts)): line.GetPointIds().SetId(i, i)
    c = vtk.vtkCellArray(); c.InsertNextCell(line)
    d = vtk.vtkPolyData(); d.SetPoints(p); d.SetLines(c)
    t = vtk.vtkTubeFilter(); t.SetInputData(d); t.SetRadius(r); t.SetNumberOfSides(10); t.CappingOn(); t.Update()
    return t.GetOutput()


def merge(*ds):
    ap = vtk.vtkAppendPolyData()
    for d in ds: ap.AddInputData(d)
    nm = vtk.vtkPolyDataNormals(); nm.SetInputConnection(ap.GetOutputPort())
    nm.SetFeatureAngle(35); nm.SplittingOn(); nm.Update()
    return nm.GetOutput()


# ------------------------------------------------------------------ conductor paths
def ring_walk(lv, a_from, a_to, e, n=None):
    a0 = RING_ARC[lv][0]
    u0, u1 = (a_from - a0) % 360, (a_to - a0) % 360
    us = np.linspace(u0, u1, n or max(3, int(abs(u1 - u0)) + 2))
    return np.array([cyl(R_MID, a0 + u, ztop(lv, e) + 0.9) for u in us])


def rounded_rect(hv=10.0, hz=38.0, rc=5.0, v_start=8.5):
    """(v, z) loop around the tooth, starting on the top edge at v_start heading +v."""
    seg = []
    corners = [(hv - rc, hz - rc, 90, 0), (hv - rc, -hz + rc, 0, -90),
               (-hv + rc, -hz + rc, -90, -180), (-hv + rc, hz - rc, 180, 90)]
    for cv, cz, t0, t1 in corners:
        for t in np.linspace(t0, t1, 10):
            seg.append((cv + rc * np.cos(np.radians(t)), cz + rc * np.sin(np.radians(t))))
    seg = np.array(seg)
    k = np.argmin(np.where(seg[:, 1] > hz - 0.01, np.abs(seg[:, 0] - v_start), 1e9))
    loop = np.vstack([[v_start, hz], seg[k:], seg[:k]])
    loop = np.vstack([loop, [v_start, hz]])
    # resample densely so the turns look round
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))]
    s = np.linspace(0, d[-1], 90)
    return np.c_[np.interp(s, d, loop[:, 0]), np.interp(s, d, loop[:, 1])]


LOOP = rounded_rect()


def coil_path(k, e):
    """F lead (weld plane) -> down -> 10 turns spiralling inward -> crossover -> up S lead."""
    th = np.radians(tooth(k))
    uF, vF = R_POST * np.cos(np.radians(6.8)), R_POST * np.sin(np.radians(6.8))
    uS, vS = R_POST * np.cos(np.radians(-5.6)), R_POST * np.sin(np.radians(-5.6))
    loc = [(uF, vF, W(e)), (uF, vF, 40.5), (70.0, vF, 38.0)]
    N = len(LOOP) - 1
    for turn in range(10):
        for j in range(N):
            s = (turn + j / N) / 10
            loc.append((70.0 - 13.0 * s, LOOP[j, 0], LOOP[j, 1]))
    loc += [(57.0, vF, 38.0), (57.0, vF, 42.0), (uS, vS, 42.0), (uS, vS, W(e))]
    loc = np.array(loc)
    x = loc[:, 0] * np.cos(th) - loc[:, 1] * np.sin(th)
    y = loc[:, 0] * np.sin(th) + loc[:, 1] * np.cos(th)
    return np.c_[x, y, loc[:, 2]]


# ------------------------------------------------------------------ elements
elements = []   # dict(kind, tag, level, spec(e)->(pts, mesh), fn(state)->A, col)


def add(kind, tag, level, spec, fn, col):
    elements.append(dict(kind=kind, tag=tag, level=level, spec=spec, fn=fn, col=col, off=0.0))
    return elements[-1]


def taps_of(lv):
    out = []
    for ph, (a, b, _, paths) in PHASES.items():
        for ta, tb in paths:
            if a == lv: out.append((F(ta), lambda s, ph=ph: s[ph] / 2))
            if b == lv: out.append((F(tb), lambda s, ph=ph: -s[ph] / 2))
    return out


def line_current(s, lv):
    return sum(fn(s) for _, fn in taps_of(lv))


for ph, (ra, rb, _, paths) in PHASES.items():
    for ta, tb in paths:
        f = lambda s, ph=ph: s[ph] / 2
        g = lambda s, ph=ph: -s[ph] / 2
        for t_, lv, fn in ((ta, ra, f), (tb, rb, g)):
            a = F(t_)
            add("tap", ph, lv, lambda e, a=a, lv=lv: (
                np.array([cyl(R_MID, a, ztop(lv, e) + 0.9), cyl(R_POST, a, ztop(lv, e) + 0.9), cyl(R_POST, a, W(e))]),
                merge(stub(a, zb(lv, e)), post(a, zb(lv, e), W(e)))), fn, LV_COL[lv])
        add("coil", ph, None, lambda e, k=ta: (coil_path(k, e), None), f, PH_COL[ph]).update(tooth=ta)
        add("coil", ph, None, lambda e, k=tb: (coil_path(k, e), None), g, PH_COL[ph]).update(tooth=tb)
        sa, sb = S(ta), S(tb)
        ccw = (sb - sa) % 360 < 180
        def link_spec(e, sa=sa, sb=sb, ccw=ccw):
            z = ztop("LINK", e) + 0.9
            span = (sb - sa) % 360 if ccw else -((sa - sb) % 360)
            arcp = [cyl(R_MID, sa + u, z) for u in np.linspace(0, span, 32)]
            pts = np.array([cyl(R_POST, sa, W(e)), cyl(R_POST, sa, z)] + arcp + [cyl(R_POST, sb, z), cyl(R_POST, sb, W(e))])
            lo, hi = (sa, sb) if ccw else (sb, sa)
            mesh = merge(sector(R_IN, R_OUT, lo - 2.4, hi + 2.4, zb("LINK", e), ztop("LINK", e)),
                         stub(sa, zb("LINK", e)), stub(sb, zb("LINK", e)),
                         post(sa, zb("LINK", e), W(e)), post(sb, zb("LINK", e), W(e)))
            return pts, mesh
        add("link", ph, "LINK", link_spec, f, LV_COL["LINK"]).update(pair=(ta, tb))

bar_segments = {}
for lv, (a0, span) in RING_ARC.items():
    taps = taps_of(lv)
    us = sorted({(a - a0) % 360 for a, _ in taps} | {(TERM[lv] - a0) % 360})
    tu = (TERM[lv] - a0) % 360
    for i in range(len(us) - 1):
        u0, u1 = us[i], us[i + 1]

        def flow(s, u0=u0, lv=lv, taps=taps, tu=tu, a0=a0):   # ccw flow after point u0
            q = line_current(s, lv) if tu <= u0 + 1e-9 else 0.0
            return q - sum(fn(s) for a, fn in taps if (a - a0) % 360 <= u0 + 1e-9)
        el = add("bar", lv, lv, lambda e, lv=lv, a0=a0, u0=u0, u1=u1: (
            ring_walk(lv, a0 + u0, a0 + u1, e),
            sector(R_IN, R_OUT, a0 + u0 - (2.4 if u0 == 0 else 0), a0 + u1 + (2.4 if u1 == span else 0),
                   zb(lv, e), ztop(lv, e))), flow, LV_COL[lv])
        el.update(u=(a0 + u0, a0 + u1))
        bar_segments.setdefault(lv, []).append(el)
    a = TERM[lv]
    add("term", lv, lv, lambda e, a=a, lv=lv: (
        np.array([cyl(72.75, a, TTOP(e)), cyl(72.75, a, ztop(lv, e) + 0.9), cyl(R_MID, a, ztop(lv, e) + 0.9)]),
        merge(stub(a, zb(lv, e), w=10), post(a, zb(lv, e), TTOP(e), 71.0, 74.5, 10))),
        lambda s, lv=lv: line_current(s, lv), LV_COL[lv])


def state_ac(theta):
    return {p: I_PK * np.sin(theta + np.radians(v[2])) for p, v in PHASES.items()}


def state_dc(ph="R", i=I_PH):
    return {p: (i if p == ph else 0.0) for p in PHASES}


# ------------------------------------------------------------------ self-checks vs CIRCUIT_v7.txt
_th = np.linspace(0, 2 * np.pi, 361)[:-1]
_rms = lambda fn: float(np.sqrt(np.mean([fn(state_ac(t)) ** 2 for t in _th])))
EXPECT = {"L1": [104.0, 180.1, 275.2, 104.0], "L2": [104.0, 180.1, 180.1, 104.0], "L3": [104.0, 275.2, 180.1, 104.0]}
for lv, segs in bar_segments.items():
    got = [round(_rms(el["fn"]), 1) for el in segs]
    assert np.allclose(got, EXPECT[lv], atol=0.3), (lv, got)
for lv in RING_ARC:
    assert abs(_rms(lambda s, lv=lv: line_current(s, lv)) - 360.3) < 0.3
assert all(abs(sum(line_current(state_ac(t), lv) for lv in RING_ARC)) < 1e-9 for t in _th)
for el in elements:
    el["rms"] = _rms(el["fn"])
print("checks OK: ring rms matches CIRCUIT_v7.txt, line 360.3 A, line currents sum to zero")

# ------------------------------------------------------------------ VTK scene
W3D, H = 1340, 1080
FULLW = 1920
ren = vtk.vtkRenderer()
ren.GradientBackgroundOn(); ren.SetBackground(0.03, 0.035, 0.05); ren.SetBackground2(0.11, 0.12, 0.16)
ren.SetUseDepthPeeling(1); ren.SetMaximumNumberOfPeels(8); ren.UseFXAAOn()
win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.AddRenderer(ren); win.SetSize(W3D, H)
win.SetAlphaBitPlanes(1); win.SetMultiSamples(0)
lk = vtk.vtkLightKit(); lk.AddLightsToRenderer(ren)


def actor_for(mesh, col, spec=0.4):
    m = vtk.vtkPolyDataMapper(); m.SetInputData(mesh)
    a = vtk.vtkActor(); a.SetMapper(m)
    p = a.GetProperty(); p.SetColor(*hexc(col)); p.SetSpecular(spec); p.SetSpecularPower(30)
    ren.AddActor(a)
    return a


for k in range(1, 13):                           # teeth + yoke: static context
    t = tooth(k)
    a_ = actor_for(merge(sector(54, 71.5, t - 5.1, t + 5.1, -31.5, 31.5), sector(52, 55, t - 12, t + 12, -31.5, 31.5)),
                   "#4a515e", 0.15)   # ponytail: yoke left out on purpose, it hides the coils
    a_.GetProperty().SetOpacity(0.55)

E_CUR = [None]


def set_explode(e):
    if E_CUR[0] is not None and abs(E_CUR[0] - e) < 1e-6:
        return
    E_CUR[0] = e
    for el in elements:
        pts, mesh = el["spec"](e)
        if mesh is None:
            mesh = tube(pts)
        if "actor" not in el:
            el["actor"] = actor_for(mesh, el["col"])
        else:
            el["actor"].GetMapper().SetInputData(mesh)
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        el.update(pts=pts, cum=d, len=d[-1])
    welds = [cyl(R_POST, F(k), W(e) + 0.4) for k in range(1, 13)] + [cyl(R_POST, S(k), W(e) + 0.4) for k in range(1, 13)]
    set_points(weld_src, np.array(welds))


def glyph_actor(radius, col, res=8):
    pd = vtk.vtkPolyData(); pd.SetPoints(vtk.vtkPoints())
    sph = vtk.vtkSphereSource(); sph.SetRadius(radius); sph.SetThetaResolution(res); sph.SetPhiResolution(res - 2)
    g = vtk.vtkGlyph3D(); g.SetInputData(pd); g.SetSourceConnection(sph.GetOutputPort()); g.ScalingOff()
    m = vtk.vtkPolyDataMapper(); m.SetInputConnection(g.GetOutputPort())
    a = vtk.vtkActor(); a.SetMapper(m); a.GetProperty().SetColor(*hexc(col))
    a.GetProperty().SetAmbient(0.6); a.GetProperty().SetDiffuse(0.6)
    ren.AddActor(a)
    return pd, a


def set_points(pd, xyz):
    p = vtk.vtkPoints()
    for q in xyz: p.InsertNextPoint(*q)
    pd.SetPoints(p); pd.Modified()


dot_src, _ = glyph_actor(1.05, "#fff6cf")
trc_src, trc_act = glyph_actor(2.8, "#ffe066", 12)
weld_src, _ = glyph_actor(1.4, "#f5f5f5")
set_explode(0.0)

cam = ren.GetActiveCamera(); cam.SetViewAngle(30)


def set_cam(focal, az, el, dist):
    a, b = np.radians(az), np.radians(el)
    cam.SetFocalPoint(*focal)
    cam.SetPosition(focal[0] + dist * np.cos(b) * np.cos(a), focal[1] + dist * np.cos(b) * np.sin(a), focal[2] + dist * np.sin(b))
    cam.SetViewUp(0, 0, 1)
    ren.ResetCameraClippingRange()


def project(p):
    ren.SetWorldPoint(p[0], p[1], p[2], 1.0); ren.WorldToDisplay()
    x, y, _ = ren.GetDisplayPoint()
    return x, H - y


# ------------------------------------------------------------------ flow
SPACING = 7.0
vis = lambda i: 75.0 * np.tanh(i / 208.0)        # mm/s, monotonic, capped to avoid strobing


def flow_dots(s, active, dt):
    xyz = []
    for el, ok in zip(elements, active):
        i = el["fn"](s)
        el["off"] = (el["off"] + vis(i) * dt) % SPACING
        if ok and abs(i) > 1.0:
            d = np.arange(el["off"], el["len"], SPACING)
            xyz.append(np.c_[[np.interp(d, el["cum"], el["pts"][:, j]) for j in range(3)]].T)
    set_points(dot_src, np.vstack(xyz) if xyz else np.empty((0, 3)))


def style(s, active, iref):
    for el, ok in zip(elements, active):
        p = el["actor"].GetProperty()
        if ok:
            k = 0.45 + 0.55 * min(1.0, abs(el["fn"](s)) / iref)
            p.SetColor(*[c * k for c in hexc(el["col"])]); p.SetOpacity(1.0)
        else:
            p.SetColor(*hexc(el["col"])); p.SetOpacity(0.10)


def junction(p, s):
    ins, outs = [], []
    for el in elements:
        for end, sgn in ((0, -1), (-1, 1)):
            if np.linalg.norm(el["pts"][end] - p) < 1e-6:
                i = el["fn"](s) * sgn
                if abs(i) > 0.5: (ins if i > 0 else outs).append(abs(i))
    f = lambda L: " + ".join(f"{v:.0f}" for v in L) or "0"
    return f"in  {f(ins)} A", f"out {f(outs)} A"


def find(kind, **kw):
    return next(el for el in elements if el["kind"] == kind and all(el.get(k) == v for k, v in kw.items()))


def poly_concat(*parts):
    return np.vstack(parts)


# ------------------------------------------------------------------ 2D overlay
FONT = "C:/Windows/Fonts/"
fnt = lambda sz, b=False: ImageFont.truetype(FONT + ("segoeuib.ttf" if b else "segoeui.ttf"), sz)
mono = lambda sz: ImageFont.truetype(FONT + "consola.ttf", sz)
F_T, F_C, F_L, F_S, F_M = fnt(30, True), fnt(22), fnt(19, True), fnt(16), mono(17)
rgb = lambda h, a=255: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) + (a,)


def box_text(d, xy, lines, col="#ffe066", font=F_M, pad=9):
    x, y = xy
    w = max(d.textbbox((0, 0), ln, font=font)[2] for ln in lines)
    lh = font.size + 5
    d.rounded_rectangle([x, y, x + w + 2 * pad, y + lh * len(lines) + 2 * pad - 4], 8,
                        fill=(20, 23, 31, 235), outline=rgb(col), width=2)
    for i, ln in enumerate(lines):
        d.text((x + pad, y + pad + i * lh), ln, font=font, fill=rgb("#eeeeee") if i else rgb(col))
    return x + w + 2 * pad, y + lh * len(lines) + 2 * pad


def callout(d, p3, lines, dxy, col="#ffe066"):
    px, py = project(p3)
    bx, by = px + dxy[0], py + dxy[1]
    d.ellipse([px - 16, py - 16, px + 16, py + 16], outline=rgb(col), width=3)
    x1, y1 = box_text(d, (bx, by), lines, col)
    d.line([px, py, min(max(px, bx), x1), min(max(py, by), y1)], fill=rgb(col, 200), width=2)


def label(d, p3, text, col, at):
    px, py = project(p3)
    d.line([px, py, at[0] + (0 if at[0] > px else d.textbbox((0, 0), text, font=F_L)[2] + 14), at[1] + 13],
           fill=rgb(col, 170), width=2)
    d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=rgb(col))
    d.rounded_rectangle([at[0] - 6, at[1] - 2, at[0] + d.textbbox((0, 0), text, font=F_L)[2] + 8, at[1] + 28], 6,
                        fill=(15, 17, 23, 210))
    d.text(at, text, font=F_L, fill=rgb(col))


STACK = [("TERMINALS (delta points)", "z 92", "#e6e6e6", "term"), ("WELD PLANE (all 24 leads)", "z 73", "#f5f5f5", "weld"),
         ("L1  ring", "z 67.5", LV_COL["L1"], "L1"), ("L3  ring", "z 63.5", LV_COL["L3"], "L3"),
         ("L2  ring", "z 59.5", LV_COL["L2"], "L2"), ("LINK  6 series links", "z 55.5", LV_COL["LINK"], "LINK"),
         ("COILS  end turns", "z <= 44", "#9aa0aa", "coil")]


def draw_stack(d, x0, y0, hot=(), info=None):
    d.text((x0, y0), "Z-STACK  (side view, bottom -> top)", font=F_L, fill=rgb("#ffffff"))
    y = y0 + 40
    for name, z, col, key in STACK:
        on = key in hot
        h = 30
        if key in ("L1", "L2", "L3", "LINK"):
            d.rectangle([x0, y + 8, x0 + 150, y + 22], fill=rgb(col, 255 if on or not hot else 70))
        elif key == "weld":
            for i in range(8): d.ellipse([x0 + 8 + i * 18, y + 10, x0 + 18 + i * 18, y + 20], fill=rgb(col, 255 if on or not hot else 70))
        elif key == "term":
            for i in range(3): d.rectangle([x0 + 20 + i * 50, y + 2, x0 + 30 + i * 50, y + 28], fill=rgb(col, 255 if on or not hot else 70))
        else:
            for i in range(5): d.arc([x0 + 5 + i * 28, y + 2, x0 + 33 + i * 28, y + 28], 0, 360, fill=rgb(col, 255 if on or not hot else 70), width=3)
        if on:
            d.polygon([(x0 - 22, y + 7), (x0 - 8, y + 15), (x0 - 22, y + 23)], fill=rgb("#ffe066"))
        d.text((x0 + 170, y + 3), name, font=F_S, fill=rgb(col if on or not hot else "#7a808a"))
        d.text((x0 + 420, y + 3), z, font=F_S, fill=rgb("#cfcfcf" if on or not hot else "#6a707a"))
        if info and key in info:
            d.text((x0 + 170, y + 21), info[key], font=mono(13), fill=rgb("#ffe066"))
        y += h + 22
    return y


def draw_ladder(d, x0, y0, step, s):
    d.text((x0, y0), "PHASE R CIRCUIT (what the 3D view shows)", font=F_L, fill=rgb("#ffffff"))
    y0 += 40
    c = lambda k: rgb("#ffe066") if k == step else rgb("#8a909a")
    d.text((x0, y0 + 70), "L1\nterm", font=F_S, fill=c(0))
    d.line([x0 + 45, y0 + 90, x0 + 75, y0 + 90], fill=c(1), width=3)
    for row, (ta, tb) in enumerate(PHASES["R"][3]):
        yy = y0 + 30 + row * 120
        d.line([x0 + 75, y0 + 90, x0 + 75, yy + 20], fill=c(1), width=3)
        d.line([x0 + 75, yy + 20, x0 + 100, yy + 20], fill=c(1), width=3)
        d.rounded_rectangle([x0 + 100, yy, x0 + 185, yy + 40], 8, outline=c(2), width=3)
        d.text((x0 + 112, yy + 8), f"T{ta}", font=F_L, fill=c(2))
        d.line([x0 + 185, yy + 20, x0 + 245, yy + 20], fill=c(3), width=3)
        d.text((x0 + 192, yy + 25), "link", font=F_S, fill=c(3))
        d.rounded_rectangle([x0 + 245, yy, x0 + 330, yy + 40], 8, outline=c(4), width=3)
        d.text((x0 + 262, yy + 8), f"T{tb}", font=F_L, fill=c(4))
        d.line([x0 + 330, yy + 20, x0 + 365, yy + 20], fill=c(5), width=3)
        d.line([x0 + 365, yy + 20, x0 + 365, y0 + 90], fill=c(5), width=3)
        d.text((x0 + 112, yy + 44), f"{abs(s['R']) / 2:.0f} A each", font=mono(14), fill=rgb("#cfcfcf"))
    d.line([x0 + 365, y0 + 90, x0 + 395, y0 + 90], fill=c(6), width=3)
    d.text((x0 + 400, y0 + 70), "L2\nterm", font=F_S, fill=c(6))
    d.text((x0, y0 + 270), "L1 z67.5 > coils > LINK z55.5 > coils > L2 z59.5",
           font=mono(14), fill=rgb("#aab0ba"))


def draw_delta(d, x0, y0, s, offs):
    d.text((x0, y0), "DELTA: terminals are the corners", font=F_L, fill=rgb("#ffffff"))
    C = {"L1": np.array([x0 + 250, y0 + 110]), "L2": np.array([x0 + 80, y0 + 360]), "L3": np.array([x0 + 420, y0 + 360])}
    G = sum(C.values()) / 3
    for ph, (a, b, _, _) in PHASES.items():
        pa, pb = C[a], C[b]
        n = np.array([-(pb - pa)[1], (pb - pa)[0]]); n = n / np.linalg.norm(n)
        for o in (-7, 7):
            d.line([*(pa + n * o), *(pb + n * o)], fill=rgb(PH_COL[ph], 200), width=4)
            L = np.linalg.norm(pb - pa); i = s[ph] / 2
            for t in np.arange(offs[ph] % 26, L, 26):
                q = pa + n * o + (pb - pa) * t / L
                if abs(i) > 2: d.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=rgb("#fff6cf"))
        m = (pa + pb) / 2; m = m + (m - G) / np.linalg.norm(m - G) * 42
        d.text((m[0] - 34, m[1] - 10), f"{ph} {s[ph]:+.0f}", font=mono(16), fill=rgb(PH_COL[ph]))
    for lv, p in C.items():
        d.ellipse([p[0] - 14, p[1] - 14, p[0] + 14, p[1] + 14], fill=rgb("#14171f"), outline=rgb(LV_COL[lv]), width=4)
        il = line_current(s, lv)
        dy = -44 if lv == "L1" else 24
        d.text((p[0] - 60, p[1] + dy), f"{lv} line {il:+.0f}", font=mono(16), fill=rgb(LV_COL[lv]))


def draw_wave(d, x0, y0, w, h, theta):
    d.text((x0, y0 - 26), "phase currents (A, peak 294)", font=F_S, fill=rgb("#aab0ba"))
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=rgb("#3a3f4b"))
    d.line([x0, y0 + h / 2, x0 + w, y0 + h / 2], fill=rgb("#3a3f4b"))
    xs = np.linspace(0, 2 * np.pi, 120)
    for ph, v in PHASES.items():
        pts = [(x0 + w * t / (2 * np.pi), y0 + h / 2 - (h / 2 - 6) * np.sin(t + np.radians(v[2]))) for t in xs]
        d.line(pts, fill=rgb(PH_COL[ph]), width=3)
    cx = x0 + w * ((theta % (2 * np.pi)) / (2 * np.pi))
    d.line([cx, y0, cx, y0 + h], fill=rgb("#ffe066"), width=2)


# ------------------------------------------------------------------ timeline
FPS = 30
smooth = lambda x: (lambda c: c * c * (3 - 2 * c))(min(1.0, max(0.0, x)))
GLOBAL = ((0, 0, 40), 52, 32, 390)
GLOBAL_X = ((0, 0, 58), 52, 30, 420)
COILV = ((64, 0, 40), 20, 22, 240)
SCENES = [("intro", 6), ("explode", 9), ("coil", 16), ("single", 34), ("three_intro", 7), ("three", 24), ("collapse", 8), ("end", 6)]
T_TOTAL = sum(d for _, d in SCENES)
AC_PERIOD = 8.0

# single-phase steps: (t0, t1, caption, z-stack hot, ladder step)
STEPS = [(0, 4, "208 A enters at terminal L1 (a delta point, top z 92) and drops down onto ring L1 (z 67.5)", ("term", "L1"), 0),
         (4, 9, "JUNCTION on L1: at the T12 tab 104 A drops off, the other 104 A keeps going along L1 to the T7 tab", ("L1",), 1),
         (9, 15, "Each 104 A goes up its tab to the weld plane, then DOWN the coil's F-lead, 10 turns round the tooth, UP the S-lead", ("weld", "coil"), 2),
         (15, 19, "The S-leads land on the LINK level (z 55.5, lowest). Each series link carries 104 A to the next coil", ("LINK",), 3),
         (19, 25, "Second coil of each path: down its S-lead, 10 turns, up its F-lead to the weld plane -> tab of ring L2", ("coil", "weld"), 4),
         (25, 30, "JUNCTION on L2 (z 59.5): the two 104 A paths arrive from T1 and T6 and meet at the L2 terminal", ("L2",), 5),
         (30, 34, "104 + 104 = 208 A leaves at terminal L2 (delta point). Phase R = 2 parallel paths x 2 coils in series", ("term", "L2"), 6)]


def lerp_cam(a, b, f):
    return (tuple(np.array(a[0]) * (1 - f) + np.array(b[0]) * f), *[x * (1 - f) + y * f for x, y in zip(a[1:], b[1:])])


def scene_at(t):
    t0 = 0
    for name, d in SCENES:
        if t < t0 + d: return name, t - t0, d
        t0 += d
    return SCENES[-1][0], SCENES[-1][1], SCENES[-1][1]


def trace_parts(path_no, e):
    ta, tb = PHASES["R"][3][path_no]
    term1, term2 = find("term", tag="L1")["pts"], find("term", tag="L2")["pts"]
    tap1 = next(el for el in elements if el["kind"] == "tap" and el["tag"] == "R" and abs(el["pts"][0][1] - cyl(R_MID, F(ta), 0)[1]) < 1e-6 and el["level"] == "L1")["pts"]
    tap2 = next(el for el in elements if el["kind"] == "tap" and el["tag"] == "R" and abs(el["pts"][0][1] - cyl(R_MID, F(tb), 0)[1]) < 1e-6 and el["level"] == "L2")["pts"]
    c1, c2 = find("coil", tooth=ta)["pts"], find("coil", tooth=tb)["pts"]
    lk = find("link", pair=(ta, tb))["pts"]
    return [term1, ring_walk("L1", TERM["L1"], F(ta), e), poly_concat(tap1, c1), lk, poly_concat(c2[::-1], tap2[::-1]),
            ring_walk("L2", F(tb), TERM["L2"], e), term2[::-1]]


def along(pts, f, trail=6, gap=2.2):
    """tracer head plus a short trail behind it"""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    ss = np.clip(f * d[-1] - gap * np.arange(trail), 0, None)
    return np.array([[np.interp(x, d, pts[:, j]) for j in range(3)] for x in ss])


def render(frame, dt):
    t = frame / FPS
    name, tl, dur = scene_at(t)
    img = Image.new("RGBA", (FULLW, H), (0, 0, 0, 255))
    ov = Image.new("RGBA", (FULLW, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    trc = []
    ALL = [True] * len(elements)
    title, caption = "", ""
    callouts, labels_on, panel = [], False, None
    s = state_dc("R", 0.0); active = ALL; iref = 208; e = 0.0; camv = GLOBAL

    if name == "intro":
        e = 0.0; camv = (GLOBAL[0], 32 + 40 * smooth(tl / dur), 32, 390)
        title = "Stator v7: 12 tooth coils (grey teeth) with the busbar stack on top, REAL scale"
        caption = "The copper above the coils is 4 separate levels stacked in Z. At real scale they are only 1 mm apart, so we pull them apart next."
        panel = ("stack", ())
    elif name == "explode":
        e = EXPLODE * smooth(tl / 5.0)
        camv = lerp_cam((GLOBAL[0], 72, 32, 390), GLOBAL_X, smooth(tl / 5.0))
        title = "Exploded in Z: each copper level separated so you can see it"
        caption = "Bottom to top: LINK z55.5 (6 series links) · ring L2 z59.5 · ring L3 z63.5 · ring L1 z67.5 · all coil leads weld at the weld plane · terminals on top"
        labels_on = tl > 3; panel = ("stack", ())
    elif name == "coil":
        e = EXPLODE
        f = smooth(tl / 2.5) * (1 - smooth((tl - 14) / 2))
        camv = lerp_cam(GLOBAL_X, COILV, f)
        c1 = find("coil", tooth=1)
        active = [el is c1 or (el["kind"] == "tap" and el["tag"] == "R" and el["level"] == "L2" and np.allclose(el["pts"][-1], c1["pts"][0]))
                  or (el["kind"] == "link" and el.get("pair") == (12, 1)) for el in elements]
        s = {"R": -I_PH, "Y": 0.0, "B": 0.0}        # only used for T1's direction: F -> S
        title = "One coil (tooth T1): IN by one lead, OUT by a different lead"
        caption = "Current comes DOWN the F-lead, winds 10 turns round the tooth, and goes UP the S-lead. Two separate leads, two separate busbar parts."
        fr = min(1.0, max(0.0, (tl - 2.5) / 10.5))
        trc = list(along(c1["pts"], fr))
        callouts = [(c1["pts"][1] + [0, 0, 12], ["IN : F-lead", "from ring L2 tab"], (40, -150), "#7CFC9A"),
                    (c1["pts"][-2] + [0, 0, 12], ["OUT: S-lead", "to series link (LINK level)"], (-330, -160), "#ff8f6b"),
                    (c1["pts"][len(c1["pts"]) // 2], ["10 turns around", "tooth T1"], (60, 60), "#ffe066")]
        panel = ("stack", ("coil", "weld"))
    elif name == "single":
        e = EXPLODE; camv = (GLOBAL_X[0], 52 - 6 * tl / dur, 30, 420)
        ramp = smooth(tl / 1.0)
        s = state_dc("R", I_PH * ramp)
        active = [el["tag"] in ("R", "L1", "L2") for el in elements]
        st = next(x for x in STEPS if x[0] <= tl < x[1]) if tl < STEPS[-1][1] else STEPS[-1]
        k = STEPS.index(st)
        title = "Step 1 · Single phase R only (DC 208 A): follow the current level by level"
        caption = st[2]
        frac = (tl - st[0]) / (st[1] - st[0])
        for pn in (0, 1):
            parts = trace_parts(pn, e)
            trc += list(along(parts[k], min(1.0, frac)))
        if k == 1:
            p = find("tap", tag="R", level="L1")["pts"][0]
            callouts = [(p, ["JUNCTION  ring L1  z67.5", *junction(p, s)], (60, -170), "#ffe066")]
        if k == 5 or k == 6:
            p = find("term", tag="L2")["pts"][-1]
            callouts = [(p, ["JUNCTION  terminal L2", *junction(p, s)], (70, -40), "#ffe066")]
        if k == 0:
            p = find("term", tag="L1")["pts"][0]
            callouts = [(p, ["TERMINAL L1 (delta point)", f"in  {line_current(s, 'L1'):.0f} A"], (40, -100), "#ffe066")]
        if k == 3:
            lk = find("link", pair=(12, 1))["pts"]
            callouts = [(lk[len(lk) // 2], ["SERIES LINK  z55.5", f"{abs(s['R']) / 2:.0f} A  T12 -> T1"], (40, 60), "#ffe066")]
        panel = ("stack+ladder", st[3], k)
    elif name in ("three_intro", "three", "collapse", "end"):
        if name == "three_intro":
            theta = np.radians(75.0)
        else:
            T0 = sum(dd for nn, dd in SCENES[:5])
            theta = 2 * np.pi * (t - T0) / AC_PERIOD + np.radians(75.0)
        s = state_ac(theta)
        e = EXPLODE * (1 - smooth((tl - 1) / 5)) if name == "collapse" else (0.0 if name == "end" else EXPLODE)
        camv = {"collapse": lerp_cam(GLOBAL_X, GLOBAL, smooth((tl - 1) / 5)), "end": GLOBAL}.get(name, GLOBAL_X)
        iref = I_PK
        if name == "three_intro":
            title = "Step 2 · All three phases: R (L1->L2), Y (L2->L3), B (L3->L1)"
            caption = "Each terminal is a delta point joining two phases: line 360 A rms = two phase windings of 208 A, each split into two 104 A paths"
        elif name == "three":
            title = "Step 3 · Three-phase AC (slowed 1 cycle / 8 s): every terminal is a live junction"
            caption = "Watch the dots reverse as each phase swings. At every junction: current in = current out (KCL)."
            k = int(((t - sum(dd for nn, dd in SCENES[:5])) // 8)) % 3
            lv = ["L1", "L2", "L3"][k]
            p = find("term", tag=lv)["pts"][-1]
            callouts = [(p, [f"TERMINAL {lv} on ring {lv}", *junction(p, s)], (70, -60) if lv != "L3" else (60, -150), LV_COL[lv])]
        elif name == "collapse":
            title = "Back to real spacing"
            caption = "The 4 copper levels are 3 mm thick with 1 mm insulation gaps; tabs lift every ring to the same weld plane (z 73)."
        else:
            title = "Summary"
            caption = "Ring rms 104 / 180 / 275 A · 360 A line at each terminal · every coil 104 A · ring J <= 9.2 A/mm2"
        panel = ("delta", theta)

    set_explode(e)
    set_cam(*camv)
    style(s, active, iref)
    flow_dots(s, active, dt)
    set_points(trc_src, np.array(trc) if trc else np.empty((0, 3)))
    win.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.ReadFrontBufferOff(); w2i.Update()
    arr = vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(H, W3D, -1)[::-1, :, :3]
    img.paste(Image.fromarray(np.ascontiguousarray(arr)), (0, 0))

    # ---- annotations in the 3D view
    if labels_on or name in ("coil", "single", "three_intro", "three", "collapse"):
        anch = {lv: cyl(R_OUT, a, ztop(lv, e)) for lv, a in (("L1", 318), ("L3", 268), ("L2", 296))}   # front-left side
        anch["LINK"] = cyl(R_OUT, 330, ztop("LINK", e))
        weldp = min([cyl(R_POST, F(k), W(e)) for k in range(1, 13)], key=lambda q: project(q)[0])
        ys = sorted([(project(anch[k])[1], k) for k in anch])
        if name != "coil":
            for i, (yy, k) in enumerate(ys):
                label(d, anch[k], f"{LV_NAME[k]}  z {LV_Z[k]}", LV_COL[k], (24, 300 + i * 70 if e > 4 else 380 + i * 44))
            label(d, weldp, "weld plane z 73", "#f5f5f5", (24, 230))
            for lv in TERM:
                px, py = project(find("term", tag=lv)["pts"][0])
                d.text((px - 18, py - 34), lv, font=fnt(22, True), fill=rgb(LV_COL[lv]))
    for p3, lines, dxy, col in callouts:
        callout(d, p3, lines, dxy, col)

    # ---- title / caption / panel
    d.rectangle([0, 0, FULLW, 62], fill=(8, 9, 13, 230))
    d.text((26, 12), title, font=F_T, fill=rgb("#ffffff"))
    d.rectangle([0, H - 70, FULLW, H], fill=(8, 9, 13, 235))
    d.text((26, H - 54), caption, font=F_C, fill=rgb("#f0f0f0"))
    d.rectangle([W3D, 62, FULLW, H - 70], fill=(14, 16, 22, 255))
    d.line([W3D, 62, W3D, H - 70], fill=rgb("#2a2f3a"), width=2)
    px0 = W3D + 40
    if panel:
        if panel[0] == "stack":
            yb = draw_stack(d, px0, 95, panel[1])
            d.text((px0, yb + 20), "How to read the 3D view", font=F_L, fill=rgb("#ffffff"))
            for i, ln in enumerate(["• coloured flat bars = busbar copper levels",
                                    "• red / yellow / blue tubes = coils of phase R / Y / B",
                                    "• white dots on top = weld points (lead meets tab)",
                                    "• small moving dots = current direction & flow",
                                    "• the 3 tall blades on top = terminals (delta points)"]):
                d.text((px0, yb + 55 + i * 28), ln, font=F_S, fill=rgb("#c8ccd4"))
        elif panel[0] == "stack+ladder":
            yb = draw_stack(d, px0, 95, panel[1])
            draw_ladder(d, px0, yb + 10, panel[2], s)
        else:
            theta = panel[1]
            for ph in PHASES:                     # dots on the delta sides move with the phase current
                _delta_off[ph] = _delta_off.get(ph, 0.0) + np.tanh(s[ph] / 208) * 1.2
            draw_delta(d, px0, 90, s, _delta_off)
            draw_wave(d, px0, 560, 500, 170, theta)
            d.text((px0, 770), "rms per element", font=F_L, fill=rgb("#ffffff"))
            for i, ln in enumerate(["coil / link / path      104 A", "phase winding           208 A",
                                    "ring segments   104 · 180 · 275 A", "line (each terminal)    360 A",
                                    "line = phase x √3  (delta)"]):
                d.text((px0, 805 + i * 26), ln, font=mono(16), fill=rgb("#c8ccd4"))
    img.alpha_composite(ov)
    return np.asarray(img.convert("RGB"))


_delta_off = {}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for sec in map(float, sys.argv[1:]):
            Image.fromarray(render(int(sec * FPS), 1 / FPS)).save(f"preview3d_{sec:g}s.png")
            print("preview", sec)
    else:
        out = "videos/BUSBAR_current_flow_v7_3D.mp4"
        ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-s", f"{FULLW}x{H}", "-r", str(FPS), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", out],
                              stdin=subprocess.PIPE)
        n = int(T_TOTAL * FPS)
        for fr in range(n):
            ff.stdin.write(render(fr, 1 / FPS).tobytes())
            if fr % 150 == 0: print(f"{fr}/{n}", flush=True)
        ff.stdin.close(); ff.wait()
        print("wrote", out, n, "frames")
