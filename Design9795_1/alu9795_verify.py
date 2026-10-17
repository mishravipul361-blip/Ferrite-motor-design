"""Check of the stator-variant interpolation used in alu9795_study.py: direct FEMM solves for a variant and the copper
stator D0 at currents between the 4 x 4 grid nodes and at 4 rotor positions (0, 3, 6, 9 deg), against the grid
interpolation (Corr) of the flux-linkage difference and the stator Steinmetz-sum ratio.

python alu9795_verify.py F2 [A1 ...]   -> alu9795_verify.json
"""
import json, os, sys
import numpy as np
import alu9795_femm as F
import lab_map as L

PTS = [(-0.15, 0.90), (-0.30, 0.60), (-0.10, 0.30), (-0.50, 0.40), (-0.70, 0.70)]     # x path peak current (10-turn equivalent)
POS = (0.0, 3.0, 6.0, 9.0)


def main():
    from multiprocessing import Pool
    import alu9795_study as S
    svs = sys.argv[1:]
    todo = [(d, float(a * L.I_PATH_PK), float(b * L.I_PATH_PK), th) for d in ['D0'] + svs for a, b in PTS for th in POS]
    done = F.load_done()
    run = [t for t in todo if '|'.join(map(str, t)) not in done]
    print(len(run), 'solves', flush=True)
    with Pool(min(6, max(1, len(run)))) as pool:
        try:
            for r in pool.imap(F.job, run, chunksize=4):
                done[r['key']] = r
        finally:
            F.RG.kill_femm_children()
    fn = os.path.join(F.H, 'alu9795_verify.json')
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for sv in svs:
        rows = []
        for a, b in PTS:
            i_d, i_q = float(a * L.I_PATH_PK), float(b * L.I_PATH_PK)
            g = lambda d, k, pos=POS: float(np.mean([done['|'.join(map(str, (d, i_d, i_q, th)))][k] for th in pos]))
            C = S.CORR(sv, np.array([[i_d, i_q]]))
            row = dict(id=i_d, iq=i_q)
            for k, ck in (('psid', 'dpsid'), ('psiq', 'dpsiq')):
                row[k + '_D0'] = g('D0', k)
                row['d' + k + '_femm4'] = g(sv, k) - g('D0', k)
                row['d' + k + '_femm2'] = g(sv, k, (0.0, 6.0)) - g('D0', k, (0.0, 6.0))
                row['d' + k + '_interp'] = float(C[ck][0])
            for k, ck in (('st_Sa', 'rSa'), ('st_S2', 'rS2'), ('cu_B2', 'rcu')):
                row['r' + k + '_femm4'] = g(sv, k) / g('D0', k)
                row['r' + k + '_interp'] = float(C[ck][0])
            row['T_D0'] = g('D0', 'T'); row['T_sv'] = g(sv, 'T')
            rows.append(row)
            print(sv, '%6.1f %6.1f' % (i_d, i_q), 'dpsiq femm %+.5f interp %+.5f (of %.4f) | rSa femm %.3f interp %.3f | T %.2f -> %.2f' % (
                row['dpsiq_femm4'], row['dpsiq_interp'], row['psiq_D0'], row['rst_Sa_femm4'], row['rst_Sa_interp'], row['T_D0'], row['T_sv']), flush=True)
        out[sv] = rows
    json.dump(out, open(fn, 'w'), indent=1)


if __name__ == '__main__':
    main()
