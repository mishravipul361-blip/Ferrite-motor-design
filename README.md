# Ferrite_Motor_design – v7 EV two-wheeler motor

12-slot / 10-pole spoke-ferrite PMSM for a 52 V two-wheeler: CAD, busbar design, rotor geometry and the FEMM
electromagnetic study (calibrated to Motor-CAD).

| | |
|---|---|
| Topology | 12 slots, 10 poles, concentrated winding, segmented stator, spoke ferrite rotor |
| Size | stator Ø160 / bore Ø107 mm, rotor Ø106 mm, air gap 0.5 mm, stack 65 mm |
| Winding | delta, 2 parallel × 2 series per phase, 10 turns/coil, Ø2.90 mm wire |
| Drive | 52 V battery, 360 A rms line, 11 kW battery-side limit |
| Result | 45 N·m peak, 45 N·m to ~2150 rpm then ~10.4 kW to 12 000 rpm |

## Folders

| Folder | Contents | Start with |
|---|---|---|
| [`cad/`](cad) | v7 stator CAD: tooth segment, bobbin halves, coil, segment and ring assemblies (butt / dovetail / T-tab joints), full stator, busbar assembly and parts (`cad/busbar/`). SolidWorks `.SLDPRT/.SLDASM` + STEP. Design checks: `CHECKS_v7.txt` (clearances, connections), `CIRCUIT_v7.txt` (winding circuit), `RESISTANCE_v7.txt` | `STATOR_v7.step`, `fieldlab_v7.html` (stator field lab), `VERIFICATION_v7.html` (busbar verification) |
| [`rotor/`](rotor) | Spoke rotor parametric geometry explorer (same parameters as `Adaptive_temp_24_09.py`) | `spoke_rotor_explorer.html` |
| [`busbar/`](busbar) | Busbar design study pages, current-flow animation scripts and videos, optimisation report | `busbar_study_analysis_v7.html`, `optimisation_report/report.html` |
| [`femm_em/`](femm_em) | FEMM 2D electromagnetic study: scripts, results, flux maps, efficiency maps, HTML report | `report/v7_motor_report.html`, `VERIFICATION.md`, `README.md` |

The SolidWorks assemblies find their parts by relative path: keep each assembly in the same folder as its parts
(`cad/` and `cad/busbar/`) when moving files.

## Open and rerun
- HTML pages open directly in a browser (charts in the FEMM report load Plotly from the internet).
- FEMM study: install FEMM 4.2 and Python 3.10 with `pyfemm numpy scipy matplotlib psutil pywin32`, then run `femm_em/run_all.bat`.
- Busbar videos: `cd busbar && python busbar_animation_3d.py` (needs `numpy pillow imageio-ffmpeg`).

Large binaries (CAD, FEMM solutions) are stored with Git LFS (`.gitattributes`). The busbar current-flow videos
(`busbar/videos/`) and superseded design iterations (`archive/`) are kept locally and not pushed yet; the videos can be
regenerated with the busbar animation scripts.
