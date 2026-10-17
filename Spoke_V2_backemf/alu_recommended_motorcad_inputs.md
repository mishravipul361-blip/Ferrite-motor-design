# Spoke V2 aluminium: recommended design (Al S3 L81 N16 3x2.05 NO20), Motor-CAD inputs

Start from ADAPTIVE_MODEL_COMPLETE_4.mot (copper baseline) and change only the rows marked CHANGED.

## Stator (Parallel Tooth)
| Parameter | Copper baseline | Aluminium design |
|---|---|---|
| Slot number | 12 | 12 |
| Stator lam diameter | 160.0 mm | 160.0 mm |
| Stator bore | 107.0 mm | 107.0 mm |
| Tooth width | 15.5 mm | **14.5 mm** CHANGED |
| Slot depth | 17.5 mm | **19.5 mm** CHANGED |
| Slot corner radius | 1.95 mm | 1.95 mm |
| Tooth tip depth | 0.8 mm | 0.8 mm |
| Slot opening | 5.0 mm | 5.0 mm |
| Tooth tip angle | 20 deg | 20 deg |
| Resulting yoke thickness (check) | 9.0 mm | 7.0 mm |
| Resulting slot area (check) | 278.7 mm2 | 339.6 mm2 |

## Rotor (adaptive spoke template), all unchanged
Pole number 10, rotor diameter 106.0 mm (air gap 0.5 mm), shaft diameter 25.0 mm, magnet inset 2.0 mm, magnet depth 28.0 mm,
magnet opening 8.5 mm, magnet_X 18.0 mm, magnet_Y 10.5 mm, magnet_Z 6.0 mm, magnet glue gap 0.1 mm, bridge thickness 1.0 mm,
bridge offset 4.0 mm. Magnet DMEGC DM4654 (unchanged).

## Axial
| Parameter | Copper | Aluminium |
|---|---|---|
| Stator lamination length | 63 mm | **81 mm** CHANGED |
| Rotor lamination length | 63 mm | **81 mm** CHANGED |
| Magnet length | 63 mm | **81 mm** CHANGED |
| Stacking factor | 0.97 | 0.97 |

## Winding
| Parameter | Copper | Aluminium |
|---|---|---|
| Type | concentrated, 2 layers | same |
| Connection | delta | delta |
| Parallel paths | 2 | 2 |
| Turns per coil | 10 | **8** CHANGED |
| Conductors per slot | 20 | **16** CHANGED |
| Series turns per phase | 20 | 16 |
| Wire | SWG 11, 2.9464 bare / 3.0302 ins. | **AWG 12, 2.053 mm bare / ~2.137 mm insulated (grade 2)** CHANGED |
| Strands in hand | 1 | **3** CHANGED |
| Conductor material | Copper | **Aluminium EC-1350: 2.826e-8 ohm·m at 20 C, alpha 0.00403 /K, 2703 kg/m3** CHANGED |
| Liner, separators, impregnation | baseline | unchanged |
| Expected wire slot fill | 0.5175 | about 0.507 |
| Expected MLT | 201.6 mm | about 235.6 mm (let Motor-CAD calculate) |
| Expected phase R at 100 C | 6.67 mohm | about 7.09 mohm |

## Lamination material
Run twice: (a) **0.20 mm grade** from the Motor-CAD database (e.g. a 20JNEH / NO20 type), which is the recommended case;
(b) M235-35A unchanged, to separate the steel effect. The study modelled 0.20 mm steel as eddy x (0.20/0.35)^2 and hysteresis x1.1 against M235-35A.

## Drive and conditions (unchanged)
52 V DC, modulation index 0.95, 360 A rms line, 280 A DC current limit, MTPA, winding and magnets at 100 C, AC loss hybrid adjustment 2.

## Expected results to compare (FEMM-scaled model)
| | Copper | Al design (0.20 mm steel) |
|---|---|---|
| OP1 9000 rpm, max power | 90.71 % | 93.23 % |
| OP2 9339 rpm, 3.24 N·m | 82.39 % | 83.71 % |
| OP3 2000 rpm, 44.34 N·m | 90.03 % | 89.98 % |
| Peak efficiency | 95.31 % | 95.48 % |
| Active mass (winding / stator lam / rotor lam / magnets) | 1.47 / 3.61 / 1.24 / 1.43 = 7.76 kg | 0.61 / 4.20 / 1.60 / 1.84 = 8.25 kg |
| Peak torque at 360 A | about the same; envelope >= 98.5 % of copper at all speeds | |
