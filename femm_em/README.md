# v7 motor – FEMM 2D electromagnetic study

12-slot / 10-pole spoke-ferrite motor, 52 V battery, 360 A rms line (delta, 2 parallel × 2 series), 65 mm stack.
FEMM 4.2 (22 Oct 2023 x64 build, `c:\femm42`) driven from Python (pyfemm). Calibrated to the Motor-CAD peak torque (45 N·m).

**Start here:** `report/v7_motor_report.html` (open in a browser) and `VERIFICATION.md` (every input, material and setting).

## Model inputs
| Item | Value | Where |
|---|---|---|
| Stator | exact v7 tooth profile + conductor positions from `../cad/fieldlab_v7.html` | `femm_motor.py` |
| Rotor | `../rotor/spoke_rotor_explorer.html` defaults: OD 106, shaft 30, magnet 28 deep, bridge 2 mm, air gap 0.5 mm | `ROTOR` in `femm_motor.py` |
| Magnet | ferrite Br 0.45 T @ 20 °C, −0.20 %/K, HcJ 340 kA/m | `femm_motor.py` |
| Steel | M235-35A 0.35 mm, typical B-H curve | `BH` in `femm_motor.py` |
| Winding R | 6.732 mΩ per coil at 100 °C (`../cad/RESISTANCE_v7.txt`) | `R_COIL` |
| **Calibration** | magnet temp 35 °C, stacking 0.95, magnet μr 1.10 | **`motor_config.json`** |
| Battery | 52 V, DC power limit 11 kW, inverter efficiency 0.97 | `VDC` in `femm_motor.py`, `P_DC_MAX`/`ETA_INV` in `dq_study.py` |

## Run
Double-click **`run_all.bat`** (or run it in a terminal). It runs every step below in order, resumes if interrupted
(per-solve cache in `work/`), and rebuilds the report. ~1–2 h on a 4-core laptop; keep other heavy apps closed.

| Step | Script | Does | Writes |
|---|---|---|---|
| 1 | `femm_motor.py` | no-load EMF, cogging, current-angle sweep, rated torque ripple, torque vs current, hot magnets, demag | `results.json` |
| 2 | `plot_results.py` | plots + text summary of step 1 | `results.png`, `summary.txt` |
| 3 | `field_maps.py` | flux-density and demag pictures from the FEMM solutions | `field_*.fem/.ans`, `fig_*.png`, `field_stats.json` |
| 4 | `dq_study.py` | d-q flux map → Ld, Lq, torque–speed with voltage, current and DC limits (map cached in `dq.json`) | `dq.json`, `dq_summary.txt`, `dq_results.png` |
| 5 | `eff_map.py` | efficiency, loss, current, power-factor maps | `eff.json`, `fig_eff.png` |
| 6 | `build_report.py` | the HTML report from all of the above | `report/v7_motor_report.html` |
| – | `calibrate.py` | torque sensitivity to stacking / magnet temperature / μr (run on its own) | `calib.json` |
| – | `view_ans.py` | open a solved model in FEMM with the flux map: `python view_ans.py field_rated.ans` | – |

Change a setting (see the table in `VERIFICATION.md`) → delete `dq.json` and `results.json` → run `run_all.bat`.
`archive_uncalibrated/` holds the first results (20 °C magnets, stacking 0.97, μr 1.05: 47.1 N·m).

## Headline results (calibrated)
- Peak torque 45.0 N·m at 360 A rms, best current angle 23.2°, ripple 5.0 %, cogging 0.075 N·m
- Back-EMF 7.80 V rms line-line per 1000 rpm; Ld/Lq per delta phase 110/154 µH (light load), 94/114 µH (rated)
- With 11 kW DC limit: 45 N·m to ~2150 rpm, then ~10.4 kW shaft power up to 12 000 rpm (~16 kW without the limit)
- Demagnetisation: magnet tip corners pass HcJ at rated current (0.18 % of magnet area) – needs a design action

## Not modelled (2D)
End-winding leakage inductance, axial magnet end leakage, AC copper and magnet losses (iron loss is estimated), mechanical stress.
Solver note: FEMM's air-gap element and its "smart mesh" both made solves 5–50× slower on this laptop; the
model uses a meshed air gap with rotor rotation (`mi_moverotate`) and `mi_smartmesh(0)` instead.
Requirements: FEMM 4.2, Python 3.10 with `pyfemm numpy scipy matplotlib psutil pywin32`.
