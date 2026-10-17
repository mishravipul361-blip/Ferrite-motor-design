Spoke V2 - 12 slot / 10 pole ferrite spoke motor - FEMM 4.2 models
Source: Motor-CAD E:\Spoke_V2\ADAPTIVE_MODEL_COMPLETE_4.mot (geometry, winding, materials as of 11 Oct 2026)

FILES (open the .fem in FEMM; the .ans next to it is the solved result - File > Open the .ans to view fields)
  Spoke_V2_12s10p_NoLoad_Magnets20C.fem               open circuit, magnets 20 C (Br 0.460 T)  - back-EMF check
  Spoke_V2_12s10p_NoLoad_Magnets100C.fem              open circuit, magnets 100 C (Br 0.394 T)
  Spoke_V2_12s10p_PeakTorque_360Arms_Magnets100C.fem  360 A rms line, MTPA (id -55.7 / iq 135.9 A peak per path),
                                                      magnets 100 C -> 45.2 N.m at this rotor position (44.9 N.m averaged)

MODEL
  2D planar magnetostatic, mm, depth 63 mm, full 360 deg, A = 0 on the stator OD
  Stator: Motor-CAD Parallel Tooth - OD 160, bore 107, tooth 15.5, slot depth 17.5, slot opening 5.0,
          tip depth 0.8, tip angle 20 deg, slot corner radius 1.95
  Rotor:  spoke adaptive template - OD 106, shaft 25 (SCM420, magnetic), magnet depth 28, inset 2, opening 8.5,
          Magnet_X/Y/Z 18/10.5/6, bridge 1.0, bridge offset 4, glue gap 0.1, fillets: lamination 0.3, pocket 1.0
  Air gap 0.5 mm, split at r = 53.25 mm. Everything inside that circle is GROUP 1 (rotor).
  Rotor at 0 deg: magnet k sits on the axis at (36*k - 90) deg; magnetisation tangential, alternating.

MATERIALS
  M235-35A   Motor-CAD B-H table (0.1 T @ 24.7 A/m ... 1.8 T @ 12000 A/m, then slope mu0), stacking 0.97
  Magnet     DMEGC_DM4654 linear: mu_r 1.08, Hc = Br/(mu0*mu_r); Br 0.46 T @ 20 C, -0.1797 %/K
  SCM420     shaft, typical carbon-steel B-H
  Copper     mu_r 1 (coil sides)

WINDING (circuits)
  One circuit per tooth coil: T1 ... T12, 10 turns each (+10 in one slot half, -10 in the next).
  Coil Tk sits on the tooth between slot k and slot k+1 (Motor-CAD slot numbering; slot 1 centred at +90 deg).
  Phase 1 = T12, T1, T6, T7    Phase 2 = T2, T3, T8, T9    Phase 3 = T4, T5, T10, T11
  Coil polarity in its phase (sign of coil current = sign x phase path current):
    Ph1: T12 +, T1 -, T6 -, T7 +     Ph2: T2 -, T3 +, T8 +, T9 -     Ph3: T4 +, T5 -, T10 -, T11 +
  Delta, 2 parallel paths per phase (2 adjacent coils in series per path).
  Phase flux linkage = (sum of the 4 signed coil flux linkages) / 2.
  Circuit current = coil (= path) current; path current = phase current / 2 = line current / (2*sqrt(3)).

TO ROTATE THE ROTOR
  Select group 1, Move > Rotate by the angle about (0,0), and add the same angle to the
  magnetisation direction of every magnet block (or use backemf.py set_angle()).
  Torque: Post-processor, select group-1 blocks, Block Integral "Torque via Weighted Stress Tensor".

Scripts that build these models: ..\backemf.py (build), ..\lab_map.py (currents/d-q), ..\export_models.py
