# v7 motor – FEMM analyses and inputs, for checking against Motor-CAD

Everything here is read from the model files (`femm_motor.py`, `motor_config.json`, `../cad/fieldlab_v7.html`).
All analyses complete (2026-10-09 17:05). Model: FEMM 4.2, 2D magnetostatic, full 360° motor, rotor turned by `mi_moverotate`,
torque by weighted Maxwell stress tensor on the rotor.

## 1. Rotor inputs (rotor/spoke_rotor_explorer.html / Adaptive_temp_24_09.py names)

| Parameter (script name) | Value | Meaning |
|---|---|---|
| `pole_number` | 10 | pole pitch 36° |
| `rotor_diameter` | 106.0 mm | OD, r = 53.0 (stator file lists 106.1 – check which Motor-CAD uses) |
| `shaft_diameter` | 30.0 mm | modelled **non-magnetic** |
| `magnet_inset` | 2.0 mm | magnet top below OD → r_om = 51.0 |
| `magnet_depth` | 28.0 mm | radial magnet length → r_im = 23.0 |
| `magnet_opening` | 10.0 mm | pocket mouth at the OD (10.83°), open to the air gap |
| `magnet_X` | 18.0 mm | height of the widest point above r_im |
| `magnet_Y` | 8.5 mm | half-width at the widest point (pocket) |
| `magnet_Z` | 6.0 mm | half-width at the top (pocket) |
| `magnet_glue_gap` | 0.1 mm | on the two side faces only |
| `Bridge_thickness` | 2.0 mm | web between pockets, on the pole-piece axis |
| `bridge_offset` | 3.5 mm | pocket floor at y = 15 + 3.5 = 18.5 mm |

Derived (what FEMM actually meshed):

| Quantity | Value |
|---|---|
| Magnet shape | **6-sided tapered/bulged spoke, not rectangular** |
| Magnet width bottom / widest / top (after glue gap) | 12.64 / 16.80 / 11.79 mm |
| Magnet area per pole | 407.9 mm² (pocket 413.6 mm²) |
| Magnet mass, 10 × 65 mm at 4.9 g/cm³ | ≈ 1.30 kg |
| Pocket key points, left half, pole on +y (mm) | P(−6.422, 23.0) Q(−8.5, 41.0) R(−6.0, 51.0) D(−5.0, 51.0) C(−5.0, 52.764) G(−4.96, 18.5) |
| Magnet corners (after glue gap) | Pm(−6.321, 23.0) Qm(−8.399, 40.993) Rm(−5.897, 51.0) |
| Lip under the mouth | 1.0 mm wide (x = 5.0 → 6.0), magnet top touches it |
| Air pocket below magnet | r 23.0 → floor arc r = 19.153 mm |
| Pole arc ratio at OD | 0.699 |
| Magnetisation | tangential, alternating |

## 2. Stator, winding, materials

| Item | Value |
|---|---|
| Slots / type | 12, segmented teeth, concentrated winding, exact v7 lamination profile |
| Bore / OD / air gap | Ø107.0 / Ø160.0 / 0.50 mm |
| Tooth width / tip span / slot opening | 14.5 mm / ±11.785° / 6.43° (≈ 6.0 mm) |
| Slot bottom r / yoke | ≈ 71.5 mm / 8.5 mm; Ø5 × 4 mm back notch on each tooth axis |
| Segment joints | ideal (no gap) |
| Stack | 65 mm, rotor = stator |
| Winding | delta, 2 parallel × 2 series per phase, 10 turns/coil, 20 conductors/slot, Ø2.90 mm wire |
| Coil polarity | R: T1+ T6+ T7− T12− · Y: T4− T5+ T10+ T11− · B: T2+ T3− T8− T9+ |
| Parallel paths | R (T12,T1)∥(T7,T6) · Y (T4,T5)∥(T11,T10) · B (T3,T2)∥(T8,T9) |
| Resistance per coil = per phase | 5.122 mΩ @20 °C, 6.732 mΩ @100 °C |
| Magnet | ferrite Br 0.45 T @20 °C, −0.20 %/K, **35 °C → 0.4365 T**, **μr 1.10**, HcB 316 kA/m, HcJ 340 kA/m |
| Steel | M235-35A 0.35 mm, **stacking 0.95**, B-H (A/m : T) 0:0, 50:0.75, 100:1.00, 150:1.12, 200:1.20, 300:1.29, 500:1.38, 1000:1.45, 2500:1.53, 5000:1.63, 10000:1.75, 20000:1.87, 50000:2.02, 100000:2.13, 200000:2.30 |
| Current | 360 A rms line = 207.8 A rms phase = 103.9 A rms per path (146.97 A peak per path) |
| Current angle γ | measured from the q-axis, + = advance (field weakening) — same sense as Motor-CAD phase advance |
| Drive | 52 V DC, SVPWM, phase-voltage limit 49.4 V peak, 11 kW DC limit, inverter η 97 % |

## 3. Analyses run

| # | Analysis | FEMM settings | Result (calibrated) | File | Motor-CAD equivalent |
|---|---|---|---|---|---|
| 1 | Calibration sensitivity | 360 A rms, γ 20° & 30°, rotor 0/4/8° | baseline 47.06 N·m; stacking 0.95 −0.7 %; magnets 60 °C −7.2 %; μr 1.10 −1.3 % | `calib.json` | rerun peak torque with each setting |
| 2 | No-load back-EMF | I = 0, rotor 0–69° step 3° (one electrical period) | ψ1 0.0211 Wb-turn per path; **7.8 V rms line-line per 1000 rpm** | `results_partial.json` | Open-circuit back-EMF, delta line-line |
| 3 | Cogging torque | I = 0, rotor 0–5.75° step 0.25° (6° period) | < 0.1 N·m pk-pk | `results_partial.json` | Cogging torque |
| 4 | Torque vs current angle | 360 A rms, γ −10…70° step 10°, avg of rotor 0/4/8° | max **44.86 N·m at γ 23.2°** | `results_partial.json` | On-load torque, phase-advance sweep |
| 5 | d-q flux-linkage map | id −150…0, iq 0…150 A peak per path (5 × 5), rotor 0/4/8° | ψm 0.0211; Ld/Lq per delta phase 110/154 µH (light), 94/114 µH (rated) | `dq.json`, `dq_summary.txt` | Lab: Ld/Lq / flux-linkage map |
| 6 | Torque–speed envelope | from #5: max torque per speed with ≤ 360 A, ≤ 49.4 V, ≤ 11 kW DC | **45.0 N·m to ~2150 rpm, then ~10.4 kW to 12 000 rpm**; 16 kW / base 3300 rpm without DC limit | `dq_summary.txt` | Lab: torque–speed curve (set DC 52 V, 360 A rms, power limit) |
| 7 | Rated torque ripple | 360 A rms, γ 23.2°, rotor 0–11.5° step 0.5° (one 12° ripple period) | **44.96 N·m mean, ripple 5.0 % (43.94–46.17)** | `results.json` | On-load torque waveform |
| 8 | Torque vs current | 90/180/270/450 A rms at γ 23.2°, avg rotor 0/4/8° | 11.7 / 23.9 / 35.2 / 45.0 / 52.9 N·m at 90 / 180 / 270 / 360 / 450 A rms | `results.json` | Torque vs current |
| 9 | Hot magnets | 360 A rms, magnets 80 °C (Br 0.396 T) | 41.2 N·m | `results.json` | same with magnet temperature 80 °C |
| 10 | Field + demag cases | rotor 0°: rated, pure −d (γ 90°), no load | min H along magnetisation: no load −179, rated −664, γ 90° −495 kA/m (HcJ −340); 0.18 % of magnet area past HcJ at rated | `results.json`, `field_stats.json` | Demagnetisation analysis |
| 11 | Flux-density maps | rated + no load, rotor 0° | tooth body 1.61 T, yoke 1.50 T, stator mean 0.98 T at rated | `field_*.ans`, `fig_*.png` | Field plots |
| 12 | Efficiency / loss maps | from #5 + estimated iron + mech loss | peak motor efficiency ≈97 % at 5250 rpm, 16 N·m (estimate, optimistic) | `eff.json` | Lab: efficiency map |
| – | First (uncalibrated) run | 20 °C, stacking 0.97, μr 1.05 | 47.1 N·m, ripple 4.3 %, Kt 0.131 | `archive_uncalibrated/` | – |

Open any solved model in FEMM: `python view_ans.py calibrated_rated.ans` (rated, rotor 0°, γ 23.2°).

## 4. Material and loss data (as used)

### Magnet: sintered ferrite
| Property | Value |
|---|---|
| Br at 20 °C / temp. coefficient | 0.450 T / −0.20 %/K |
| Magnet temperature (calibrated) → Br used | 35 °C → 0.4365 T |
| Recoil permeability μr | 1.10 |
| HcB = Br/(μ0·μr) | 316 kA/m |
| HcJ (demag check) | 340 kA/m |
| Br at 80 °C (hot case) | 0.396 T |
| Conductivity / density | 0 (no eddy loss) / 4.9 g/cm³ |

### Steel: M235-35A
| Property | Value |
|---|---|
| Thickness / stacking factor / density | 0.35 mm / 0.95 / 7650 kg/m³ |
| B-H (H A/m : B T) | 0:0, 50:0.75, 100:1.00, 150:1.12, 200:1.20, 300:1.29, 500:1.38, 1000:1.45, 2500:1.53, 5000:1.63, 10000:1.75, 20000:1.87, 50000:2.02, 100000:2.13, 200000:2.30 |
| Catalogue loss | 2.35 W/kg @ 1.5 T 50 Hz; ≈17 W/kg @ 1 T 400 Hz |
| Loss model | P = (kh·f + ke·f²)·B²·mass, kh 0.0177 W/(kg·Hz·T²), ke 6.2e-5 W/(kg·Hz²·T²), ×1.5 build factor, rotor +10 % |
| Stator lamination mass | 3.37 kg |

### Copper and resistance (DC)
| | 20 °C | 100 °C | 150 °C | 180 °C |
|---|---|---|---|---|
| Per coil = per phase | 5.122 mΩ | 6.732 mΩ | 7.739 mΩ | 8.343 mΩ |
| Line-line (delta) | 3.415 mΩ | 4.488 mΩ | 5.159 mΩ | 5.562 mΩ |
| Per parallel path | 10.244 mΩ | 13.464 mΩ | 15.478 mΩ | 16.686 mΩ |
| Copper loss at 360 A rms | 665 W | 872 W | 1003 W | 1081 W |

Wire Ø2.90 mm bare (6.605 mm²), ρ20 1.724e-8 Ω·m, α 0.00393 /K, coil length 1962.4 mm, 10 turns,
J 15.75 A/mm² rms at 360 A. Busbar ≈33 W at 100 °C (not in the maps). AC/proximity loss not included.
Losses and voltage limits use 100 °C.

### Mechanical / drive
| Item | Value |
|---|---|
| Mechanical loss | 4e-3·n + 2.2e-7·n² W (≈14 W @ 3000, ≈80 W @ 12 000 rpm) |
| DC bus / voltage use | 52 V, 95 % → 49.4 V peak phase |
| Battery limit / inverter η | 11 kW DC / 97 % |

## 5. Settings you can change

| Setting | Value | Where |
|---|---|---|
| Magnet temperature, stacking, μr | 35 °C, 0.95, 1.10 | `motor_config.json` |
| Br20, BR_TC, HcJ | 0.45 T, −0.20 %/K, 340 kA/m | `femm_motor.py` |
| B-H curve | above | `femm_motor.py: BH` |
| Rotor geometry | section 1 | `femm_motor.py: ROTOR` |
| Stack, current, coil R, DC bus | 65 mm, 360 A rms, 6.732 mΩ, 52 V | `femm_motor.py: STACK, I_LINE_RMS, R_COIL, VDC` |
| DC limit, inverter η, voltage use | 11 kW, 97 %, 95 % | `dq_study.py: P_DC_MAX, ETA_INV, V_MAX` |
| Iron/mech loss model | kh, ke, build 1.5, mech | `eff_map.py` |

After a change: delete `dq.json` and `results.json`, then run `run_all.bat` (it reruns every step and rebuilds the report).
