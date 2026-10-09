"""Follow-cam version of the v7 busbar animation: the camera rides with the current.
Reuses the scene from busbar_animation_3d.py (unchanged). Writes videos/BUSBAR_current_follow_v7_3D.mp4 (run from busbar/).
  python busbar_animation_3d_follow.py           -> full video
  python busbar_animation_3d_follow.py 20 60     -> preview stills at those seconds
Story: intro -> phase R (L1->L2) -> phase Y (L2->L3) -> phase B (L3->L1) -> all three phases AC.
Each phase: terminal in -> ring -> split junction -> tab -> coil -> series link -> coil -> tab -> ring
-> merge junction -> terminal out.  Phases chain around the delta, so each starts where the last ended.
"""
import sys, subprocess
import numpy as np
from PIL import Image, ImageDraw
import busbar_animation_3d as B
from busbar_animation_3d import (PHASES, TERM, RING_ARC, LV_COL, PH_COL, LV_Z, EXPLODE, I_PH, I_PK, R_MID, F, S,
                                 cyl, ztop, ring_walk, find, elements, set_explode, set_cam, style, flow_dots,
                                 set_points, project, junction, state_dc, state_ac, line_current, smooth, along,
                                 callout, box_text, fnt, mono, rgb, draw_stack, draw_delta, draw_wave, F_C, F_T)
import imageio_ffmpeg

W, H, FPS = 1920, 1080, 30
B.win.SetSize(W, H)
_tube = B.tube
B.tube = lambda pts, r=0.85: _tube(pts, r)   # thinner coil wire so the flow dots show on it
uarc = lambda lv, a: (a - RING_ARC[lv][0]) % 360

# per-segment camera: (distance, elevation, azimuth offset from the focal point's own angle)
CAM = {"term": (165, 24, 32), "ring": (140, 36, 26), "pause": (215, 46, 18), "tap": (130, 20, 28),
       "coil": (195, 12, 18), "link": (140, 34, 26), "hold": (230, 40, 25)}
SPEED_RING = 45.0          # mm/s on the busbar


def tap_el(lv, t):
    p = cyl(R_MID, F(t), 0)
    return next(el for el in elements if el["kind"] == "tap" and el["level"] == lv
                and np.allclose(el["pts"][0][:2], p[:2], atol=1e-6))


def build_journey(ph):
    ra, rb, _, paths = PHASES[ph]
    uT = uarc(ra, TERM[ra])
    p1, p2 = sorted(paths, key=lambda p: abs(uarc(ra, F(p[0])) - uT))     # follow the path nearest the terminal
    (ta1, tb1), (ta2, tb2) = p1, p2
    e = EXPLODE
    J = {}
    # ---- start ring: where do the two paths split?
    u1, u2 = uarc(ra, F(ta1)) - uT, uarc(ra, F(ta2)) - uT
    if np.sign(u1) == np.sign(u2):
        split_a, ring1a, ring1b = F(ta1), ring_walk(ra, TERM[ra], F(ta1), e), None
        J["sec_split"] = ring_walk(ra, F(ta1), F(ta2), e)
    else:
        split_a, ring1a, ring1b = TERM[ra], None, ring_walk(ra, TERM[ra], F(ta1), e)
        J["sec_split"] = ring_walk(ra, TERM[ra], F(ta2), e)
    # ---- end ring: where do they meet again?
    vT = uarc(rb, TERM[rb])
    v1, v2 = uarc(rb, F(tb1)) - vT, uarc(rb, F(tb2)) - vT
    if np.sign(v1) == np.sign(v2):
        assert abs(v1) < abs(v2), "followed path must reach the merge tap first"
        merge_a, ring2a, ring2b = F(tb1), None, ring_walk(rb, F(tb1), TERM[rb], e)
        J["sec_merge"] = ring_walk(rb, F(tb2), F(tb1), e)
    else:
        merge_a, ring2a, ring2b = TERM[rb], ring_walk(rb, F(tb1), TERM[rb], e), None
        J["sec_merge"] = ring_walk(rb, F(tb2), TERM[rb], e)
    J["split_p"] = cyl(R_MID, split_a, ztop(ra, e) + 0.9)
    J["merge_p"] = cyl(R_MID, merge_a, ztop(rb, e) + 0.9)
    lz = lambda lv: f"z {LV_Z[lv]}"
    term_in, term_out = find("term", tag=ra)["pts"], find("term", tag=rb)["pts"][::-1]
    rdur = lambda p: max(1.6, np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) / SPEED_RING)
    segs = [
        ("term", term_in, 4.0, 208, f"Terminal {ra} (a delta point): 208 A comes in and runs down the blade onto ring {ra} ({lz(ra)})", ("term", ra), "T " + ra),
        ("ring", ring1a, None, 208, f"208 A travels along ring {ra} ({lz(ra)}) to the first junction", (ra,), "ring " + ra),
        ("pause", J["split_p"], 4.0, 208, f"SPLIT: 104 A goes to coil T{ta1}, the other 104 A carries on to coil T{ta2} (the parallel path)", (ra,), "split"),
        ("ring", ring1b, None, 104, f"104 A along ring {ra} to the T{ta1} tab", (ra,), "ring " + ra),
        ("tap", tap_el(ra, ta1)["pts"], 2.2, 104, f"Up the T{ta1} tab to the weld plane (z 73), welded to the coil's F-lead", ("weld",), "tab"),
        ("coil", find("coil", tooth=ta1)["pts"], 6.5, 104, f"Coil T{ta1}: DOWN the F-lead, 10 turns round the tooth, UP the S-lead", ("coil",), f"T{ta1}"),
        ("link", find("link", pair=(ta1, tb1))["pts"], 4.0, 104, f"S-lead drops to the LINK level (z 55.5): series link T{ta1} -> T{tb1}, still 104 A", ("LINK",), "link"),
        ("coil", find("coil", tooth=tb1)["pts"][::-1], 6.5, 104, f"Coil T{tb1} (in series): DOWN the S-lead, 10 turns, UP the F-lead", ("coil",), f"T{tb1}"),
        ("tap", tap_el(rb, tb1)["pts"][::-1], 2.2, 104, f"Weld plane -> down the T{tb1} tab onto ring {rb} ({lz(rb)})", ("weld", rb), "tab"),
        ("ring", ring2a, None, 104, f"104 A along ring {rb} toward the meeting point", (rb,), "ring " + rb),
        ("pause", J["merge_p"], 4.0, 208, f"MERGE: 104 A from T{tb1} + 104 A from T{tb2} = 208 A", (rb,), "merge"),
        ("ring", ring2b, None, 208, f"208 A along ring {rb} to terminal {rb}", (rb,), "ring " + rb),
        ("term", term_out, 4.0, 208, f"Up terminal {rb}: phase {ph} TERMINATES at delta point {rb} (208 A out)", ("term", rb), "T " + rb),
        ("hold", term_out[-1], 2.5, 208, f"Phase {ph} done: 2 parallel paths x 2 coils in series between terminals {ra} and {rb}", ("term", rb), "T " + rb),
    ]
    out = []
    for kind, pts, dur, carry, cap, hot, chip in segs:
        if pts is None:
            continue
        out.append(dict(kind=kind, pts=pts, dur=dur if dur else rdur(pts), carry=carry, cap=cap, hot=hot, chip=chip))
    J.update(ph=ph, ra=ra, rb=rb, segs=out, path=(p1, p2))
    J["chips"] = list(dict.fromkeys(s["chip"] for s in out))
    return J


B.E_CUR[0] = None
set_explode(EXPLODE)
JOURNEYS = [build_journey(ph) for ph in ("R", "Y", "B")]
for J in JOURNEYS:
    s = state_dc(J["ph"])
    for p in (J["split_p"], J["merge_p"]):                    # KCL at both junctions: 208 = 104 + 104
        ins, outs = junction(p, s)
        assert ins.replace(" ", "") in ("in208A", "in104+104A") and outs.replace(" ", "") in ("out208A", "out104+104A"), (ins, outs)
print("journeys OK:", [(J["ph"], J["path"]) for J in JOURNEYS])

# ------------------------------------------------------------------ timeline (per frame)
INTRO, FINALE = 10.0, 18.0
timeline = []          # per frame: dict(block, ...)
for f in range(int(INTRO * FPS)):
    timeline.append(dict(block="intro", tl=f / FPS))
for J in JOURNEYS:
    for si, sg in enumerate(J["segs"]):
        n = int(round(sg["dur"] * FPS))
        for f in range(n):
            timeline.append(dict(block="journey", J=J, si=si, frac=(f + 1) / n))
for f in range(int(FINALE * FPS)):
    timeline.append(dict(block="finale", tl=f / FPS))
N = len(timeline)

# tracer head + raw camera per frame
for fr in timeline:
    if fr["block"] != "journey":
        continue
    sg = fr["J"]["segs"][fr["si"]]
    if sg["kind"] in ("pause", "hold"):
        fr["head"] = np.asarray(sg["pts"], float); fr["trail"] = [fr["head"]]
    else:
        tr = along(sg["pts"], fr["frac"])
        fr["head"], fr["trail"] = tr[0], list(tr)
    fr["cam"] = CAM[sg["kind"]]

GLOBAL_X = ((0, 0, 58), 52, 30, 420)
foc = np.zeros((N, 3)); az = np.zeros(N); el = np.zeros(N); dist = np.zeros(N)
first = next(fr for fr in timeline if fr["block"] == "journey")
for i, fr in enumerate(timeline):
    if fr["block"] == "intro":
        k = smooth((fr["tl"] - 6.5) / 3.5)                 # fly from overview down to terminal L1
        hp = first["head"]; d0, e0, a0 = CAM["term"]
        foc[i] = np.array(GLOBAL_X[0]) * (1 - k) + hp * k
        az[i] = (30 + 22 * smooth(fr["tl"] / 6.5)) * (1 - k) + (np.degrees(np.arctan2(hp[1], hp[0])) + a0) * k
        el[i] = 30 * (1 - k) + e0 * k; dist[i] = 420 * (1 - k) + d0 * k
    elif fr["block"] == "journey":
        d0, e0, a0 = fr["cam"]
        foc[i] = fr["head"]; dist[i], el[i] = d0, e0
        az[i] = np.degrees(np.arctan2(fr["head"][1], fr["head"][0])) + a0
    else:
        k = smooth(fr["tl"] / 3.0)
        foc[i] = foc[i - 1] * (1 - k) + np.array(GLOBAL_X[0]) * k if fr["tl"] < 3 else np.array(GLOBAL_X[0])
        az[i] = 0; el[i] = 30; dist[i] = 420

# smooth the camera: positions with a Gaussian, azimuth unwrapped first
def gsmooth(x, sigma):
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, pad)


# finale azimuth: continue from the last journey angle and orbit slowly
jl = max(i for i, fr in enumerate(timeline) if fr["block"] == "journey")
az_u = np.degrees(np.unwrap(np.radians(az)))
for i in range(jl + 1, N):
    az_u[i] = az_u[jl] + 7.0 * (i - jl) / FPS
foc_s = gsmooth(foc, 16)
az_s, el_s, dist_s = gsmooth(az_u, 22), gsmooth(el, 22), gsmooth(dist, 22)

# ------------------------------------------------------------------ overlay helpers
def chips(d, J, cur, y):
    x = 26
    for c in J["chips"]:
        w = d.textbbox((0, 0), c, font=fnt(17, True))[2] + 22
        on = c == cur
        d.rounded_rectangle([x, y, x + w, y + 30], 8, fill=rgb("#ffe066", 235) if on else (25, 28, 36, 220),
                            outline=rgb("#ffe066" if on else "#4a505c"), width=2)
        d.text((x + 11, y + 4), c, font=fnt(17, True), fill=rgb("#14171f") if on else rgb("#aab0ba"))
        x += w + 8
        if c != J["chips"][-1]:
            d.text((x - 2, y + 2), "›", font=fnt(20, True), fill=rgb("#6a707a")); x += 14


def tag(d, p3, text, col="#ffe066"):
    px, py = project(p3)
    tw = d.textbbox((0, 0), text, font=fnt(20, True))[2]
    d.rounded_rectangle([px + 18, py - 46, px + 34 + tw, py - 14], 7, fill=(14, 16, 22, 225), outline=rgb(col), width=2)
    d.text((px + 26, py - 44), text, font=fnt(20, True), fill=rgb(col))
    d.line([px + 6, py - 6, px + 18, py - 16], fill=rgb(col), width=2)


def panel_bg(d, box):
    d.rounded_rectangle(box, 14, fill=(10, 12, 17, 238), outline=rgb("#2a2f3a"), width=2)


_doff = {}


def render(i, dt):
    fr = timeline[i]
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    trc, calls, e = [], [], EXPLODE
    if fr["block"] == "intro":
        e = EXPLODE * smooth((fr["tl"] - 2.0) / 4.0)
        s = state_dc("R", 0.0); active = [True] * len(elements); iref = 208
        title = "Ride along with the current · v7 busbar (exploded in Z)"
        cap = ("Real scale first; the 4 copper levels are only 1 mm apart..." if fr["tl"] < 2 else
               "...so we pull them apart: LINK z55.5 · L2 z59.5 · L3 z63.5 · L1 z67.5 · weld plane z73 · terminals on top")
        hot = ()
    elif fr["block"] == "journey":
        J, sg = fr["J"], fr["J"]["segs"][fr["si"]]
        ph = J["ph"]
        s = state_dc(ph, I_PH * (1.0 if fr["si"] or fr["frac"] > 0.15 else fr["frac"] / 0.15))
        active = [el["tag"] in (ph, J["ra"], J["rb"]) for el in elements]; iref = 208
        title = f"Phase {ph}  ·  terminal {J['ra']} -> terminal {J['rb']}  ·  following path T{J['path'][0][0]}-T{J['path'][0][1]}"
        cap, hot = sg["cap"], sg["hot"]
        trc = fr["trail"]
        if sg["kind"] == "pause":
            p = sg["pts"]
            calls.append((p, [sg["chip"].upper() + f"  ring {J['ra'] if sg['chip'] == 'split' else J['rb']}", *junction(p, s)], (70, -190), "#ffe066"))
            sec = J["sec_split"] if sg["chip"] == "split" else J["sec_merge"]
            trc = trc + list(along(sec, smooth(fr["frac"]) if sg["chip"] == "split" else fr["frac"]))
        if sg["kind"] not in ("hold",):
            calls_tag = (fr["head"], f"{sg['carry']} A")
        else:
            calls_tag = None
    else:
        tl = fr["tl"]
        theta = 2 * np.pi * tl / 8.0
        s = state_ac(theta); active = [True] * len(elements); iref = I_PK
        e = EXPLODE * (1 - smooth((tl - 12) / 4))
        title = "All three phases together (AC, slowed): every terminal is a live junction"
        cap = ("Each terminal joins two phases: line current = difference of the two phase currents (360 A rms)" if tl < 12
               else "Closing back to real spacing: 3 mm copper levels with 1 mm gaps, all welded at one plane")
        hot = ()
        lv = ["L1", "L2", "L3"][int(tl // 4) % 3]
        if tl < 12:
            p = find("term", tag=lv)["pts"][-1]
            calls.append((p, [f"TERMINAL {lv}", *junction(p, s)], (70, -120), LV_COL[lv]))

    set_explode(e)
    set_cam(tuple(foc_s[i]), az_s[i], el_s[i], dist_s[i])
    style(s, active, iref)
    flow_dots(s, active, dt)
    set_points(B.trc_src, np.array(trc) if trc else np.empty((0, 3)))
    B.win.Render()
    w2i = B.vtk.vtkWindowToImageFilter(); w2i.SetInput(B.win); w2i.ReadFrontBufferOff(); w2i.Update()
    arr = B.vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(H, W, -1)[::-1, :, :3]
    img = Image.fromarray(np.ascontiguousarray(arr)).convert("RGBA")

    for p3, lines, dxy, col in calls:
        callout(d, p3, lines, dxy, col)
    if fr["block"] == "journey":
        if calls_tag: tag(d, *calls_tag)
        chips(d, fr["J"], sg["chip"], H - 118)
    # compact z-stack, top right
    panel_bg(d, [W - 560, 78, W - 20, 470])
    draw_stack(d, W - 530, 96, hot)
    if fr["block"] == "finale" and fr["tl"] < 12:
        panel_bg(d, [W - 560, 490, W - 20, 1000])
        for ph in PHASES:
            _doff[ph] = _doff.get(ph, 0.0) + np.tanh(s[ph] / 208) * 1.2
        draw_delta(d, W - 540, 500, s, _doff)
    d.rectangle([0, 0, W, 62], fill=(8, 9, 13, 225))
    d.text((26, 12), title, font=F_T, fill=rgb("#ffffff"))
    d.rectangle([0, H - 72, W, H], fill=(8, 9, 13, 235))
    d.text((26, H - 56), cap, font=fnt(24), fill=rgb("#f2f2f2"))
    img.alpha_composite(ov)
    return np.asarray(img.convert("RGB"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for sec in map(float, sys.argv[1:]):
            Image.fromarray(render(min(N - 1, int(sec * FPS)), 1 / FPS)).save(f"follow_{sec:g}s.png")
            print("preview", sec)
    else:
        out = "videos/BUSBAR_current_follow_v7_3D.mp4"
        ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", out],
                              stdin=subprocess.PIPE)
        for i in range(N):
            ff.stdin.write(render(i, 1 / FPS).tobytes())
            if i % 300 == 0: print(f"{i}/{N}", flush=True)
        ff.stdin.close(); ff.wait()
        print("wrote", out, N, "frames", f"{N / FPS:.0f} s")
