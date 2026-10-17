"""Material cost of copper vs aluminium designs (per motor, active parts). Prices are assumptions, see PRICE."""
import json, sys
BY = {r['label']: r for r in json.load(open('alu_sweep.json'))}
PRICE = dict(Cu=16.5, Al=5.5, **{'M235-35A': 1.8, 'NO20': 4.0}, magnet=4.0, scrap=0.3)   # USD/kg
WIRE_WASTE, BLANK = 1.05, 0.164                                    # wire leads/scrap; 164 mm square blank per lamination
DESIGNS = [('Cu, M235 (today)', 'Copper baseline (SWG 11, 63 mm)', 'M235-35A'),
           ('Cu, 0.20 mm steel', 'Cu SWG 11 S0 NO20', 'NO20'),
           ('Al drop-in, M235', 'Al drop-in SWG 11 S0', 'M235-35A'),
           ('Al best, M235', 'Al S1 L78.8 N16 3x1.90', 'M235-35A'),
           ('Al recommended geometry, M235', 'Al S3 L81.0 N16 3x2.05', 'M235-35A'),
           ('Al recommended, 0.20 mm', 'Al S3 L81.0 N16 3x2.05 NO20', 'NO20'),
           ('Al same mass, 0.20 mm', 'Al S2 L75.0 N18 3x1.90 NO20', 'NO20')]


def cost(lab, steel, P=PRICE):
    r = BY[lab]; m = r['masses']
    import alu_study as A
    blank = (A.SLOTS.geo(r['sv']).get('od', 160.0) + 4.0) * 1e-3          # square blank = stator OD + 4 mm
    gross = blank ** 2 * r['L'] * 1e-3 * 7650 * 0.97
    net = m['stator_lam'] + m['rotor_lam']
    c = dict(wire=m['winding'] * WIRE_WASTE * P[r['metal']], steel=gross * P[steel] - (gross - net) * P['scrap'], magnet=m['magnets'] * P['magnet'])
    c['total'] = sum(c.values())
    return r, gross, c


if __name__ == '__main__':
    base = cost(DESIGNS[0][1], 'M235-35A')[2]['total']
    print('%-30s %5s %6s %6s %6s %6s %6s %7s  %s' % ('design', 'L', 'kg', 'wire$', 'steel$', 'mag$', 'tot$', 'd$', 'OP1/OP2/OP3, map'))
    for n, lab, st in DESIGNS:
        r, g, c = cost(lab, st)
        o = r['dEta_ops']
        print('%-30s %5.1f %6.2f %6.2f %6.2f %6.2f %6.2f %+7.2f  %+.2f/%+.2f/%+.2f, %+.2f  gross steel %.1f kg' % (
            n, r['L'], r['masses']['total'], c['wire'], c['steel'], c['magnet'], c['total'], c['total'] - base, o['OP1'], o['OP2'], o['OP3'], r['map_dEta_mean'], g))
    # sensitivity of the recommended design's delta
    print('\nsensitivity, Al recommended 0.20 mm minus Cu M235:')
    for desc, P in [('base', PRICE), ('Cu 12/kg', dict(PRICE, Cu=12.0)), ('Cu 20/kg', dict(PRICE, Cu=20.0)), ('NO20 3.0', dict(PRICE, NO20=3.0)),
                    ('NO20 5.0', dict(PRICE, NO20=5.0)), ('Al wire 7.0', dict(PRICE, Al=7.0))]:
        d = cost('Al S3 L81.0 N16 3x2.05 NO20', 'NO20', P)[2]['total'] - cost(DESIGNS[0][1], 'M235-35A', P)[2]['total']
        d2 = cost('Al S3 L81.0 N16 3x2.05', 'M235-35A', P)[2]['total'] - cost(DESIGNS[0][1], 'M235-35A', P)[2]['total']
        print('  %-12s Al+0.20: %+6.2f   Al+M235: %+6.2f' % (desc, d, d2))
