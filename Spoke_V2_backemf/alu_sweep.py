"""Sweeps for the aluminium study: metrics of one design against the copper baseline.
python alu_sweep.py [slot variants ...]   -> alu_sweep.json (appended per run)"""
import json, os, sys, time
import numpy as np
import runguard as RG
import alu_study as A

SP = np.arange(250, 12001, 250); TQ = np.arange(1.0, 60.01, 1.0)
B = A.baseline()
B_ENV, B_MP = B.run(SP, TQ, npts=121)
B_OPS = {o: B.op(o) for o in A.OPS}


def metrics(d):
    env, mp = d.run(SP, TQ, npts=121)
    both = ~np.isnan(mp['eff']) & ~np.isnan(B_MP['eff'])
    miss = ~np.isnan(B_MP['eff']) & np.isnan(mp['eff'])
    de = (mp['eff'] - B_MP['eff'])[both]
    tr = [e['Tsh'] / b['Tsh'] for e, b in zip(env, B_ENV) if e and b]
    ops = {o: d.op(o) for o in A.OPS}
    m = d.masses()
    return dict(label=d.label, metal=d.metal, lam=d.lam, sv=d.sv, L=d.L, Nph=d.Nph, ns=d.ns, d=d.d, fill=d.fill, R_path=d.R_path, mlt=d.mlt,
                env_ratio_min=min(tr), env_ratio_at={int(s): e['Tsh'] / b['Tsh'] for s, e, b in zip(SP, env, B_ENV) if e and b and s % 1000 == 0},
                map_dEta_mean=float(de.mean()), map_dEta_min=float(de.min()), map_dEta_p10=float(np.percentile(de, 10)),
                map_share_ge=float(np.mean(de >= -0.05)), map_missing=int(miss.sum()), map_points=int(both.sum()),
                peak_eff=float(np.nanmax(mp['eff'])), ops=ops, dEta_ops={o: (ops[o]['eff'] - B_OPS[o]['eff']) if ops[o] else None for o in ops},
                masses=m, dmass=m['total'] - B.masses()['total'])


def wire_designs(metal, sv, Lmm, Nph, lam='M235-35A'):
    return [A.Design(metal, sv, Lmm, Nph, w['ns'], w['d'], lam=lam) for w in A.wire_options(sv, Nph // 2, metal)]


if __name__ == '__main__':
    RG.keep_awake()
    t0 = time.time()
    args = sys.argv[1:] or ['S0']
    svs = [a for a in args if a[0] in 'SPQR' and a[1:].isdigit()]
    lams = [a for a in args if a in A.LAMS] or ['M235-35A']
    fn = os.path.join(A.H, 'alu_sweep.json')
    res = json.load(open(fn)) if os.path.exists(fn) else []
    have = {r['label'] for r in res}
    if os.environ.get('ALU_OUT'):                     # parallel worker: write only new records to its own file
        fn = os.path.join(A.H, os.environ['ALU_OUT']); res = []
        if os.environ.get('ALU_FRESH'):                   # recompute everything (model changed), do not skip labels of alu_sweep.json
            have = set()
    if 'Copper baseline (SWG 11, 63 mm)' not in have:
        res.append(metrics(B))
    jobs = []
    for sv in svs:
        for lam in lams:
            jobs.append(A.Design('Al', sv, 63.0, 20, 1, 2.9464, label='Al drop-in SWG 11 %s' % sv, lam=lam) if sv == 'S0' and lam == 'M235-35A' else None)
            if lam != 'M235-35A':
                jobs.append(A.Design('Cu', sv, 63.0, 20, 1, 2.9464, lam=lam, label='Cu SWG 11 %s %s' % (sv, lam)))
            for Nph in (20, 18, 16, 14):
                Ls = sorted(set([float(x) for x in np.arange(63.0, 99.01, 3.0)] + [20 * 63.0 / Nph]))
                for Lmm in Ls:
                    if Nph * Lmm < 20 * 63 - 1e-6 or (Nph == 20 and Lmm > 75):
                        continue
                    jobs += wire_designs('Al', sv, Lmm, Nph, lam)
    for d in jobs:
        if d is None or d.label in have:
            continue
        res.append(metrics(d)); have.add(d.label)
        r = res[-1]
        print('%-34s fill %.3f  env %.3f  dEta map %+.2f  OP1 %+.2f OP2 %+.2f OP3 %+.2f  dm %+.2f kg' % (
            r['label'], r['fill'], r['env_ratio_min'], r['map_dEta_mean'], *(r['dEta_ops'][o] or np.nan for o in ('OP1', 'OP2', 'OP3')), r['dmass']), flush=True)
        RG.dump_atomic(res, fn)
    print('%.0f s' % (time.time() - t0))
