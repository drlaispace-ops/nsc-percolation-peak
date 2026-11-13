"""П1.3c. Перевірка точної квадратури (cactusexact) незалежним Монте-Карло на нескінченному дереві клік (cactus.sample):
BFS від кореня по ліниво породжених кліках, NodeSC за тим самим означенням, що й sampled._sources_sc,
обрізання при s_k > S_CAP з геометричним хвостом. Порівнюються NSC_∞ і частка обрізаних коренів з точним P.

    NUMBA_NUM_THREADS=8 .venv/bin/python code/P1/p1_3c_validate.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import cactus as ca  # noqa: E402
import cactusexact as ce  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

SEED, NROOTS, S_CAP = 202611132, 60000, 10000
CASES = {"bethe12": (2,) * 12, "K3x6": (3,) * 6, "K5x3": (5,) * 3, "K7x2": (7, 7), "gleeson_c8": (8,) + (2,) * 5}
FACTORS = (0.8, 1.3, 1.7)


def main():
    out = {}
    zs = []
    for ic, (name, sizes) in enumerate(CASES.items()):
        pc = ca.pc(sizes)
        ps = np.array([f * pc for f in FACTORS])
        t = time.time()
        sc, tail, cap, _ = ca.sample(sizes, ps, NROOTS, seed=SEED + ic, scap=S_CAP)
        tot = sc + np.nan_to_num(tail)
        pts = []
        for j, p in enumerate(ps):
            ex = ce.nsc(sizes, p)
            mc, se = tot[:, j].mean(), tot[:, j].std(ddof=1) / np.sqrt(NROOTS)
            P = ca.branching(sizes, p)[1]
            cf = cap[:, j].mean()
            z = (mc - ex) / se
            zP = (cf - P) / np.sqrt(max(P * (1 - P), 1e-12) / NROOTS)
            zs.append(z)
            pts.append(dict(p=float(p), exact=ex, mc=float(mc), se=float(se), z=float(z),
                            tail_mean=float(np.nan_to_num(tail[:, j]).mean()), tail_nan=int(np.isnan(tail[:, j]).sum()),
                            P_exact=P, capfrac=float(cf), zP=float(zP)))
            print(f"{name} p={p:.5f}: точно {ex:.5f}, МК {mc:.5f} ± {se:.5f} (z = {z:+.2f}); "
                  f"P {P:.4f} проти {cf:.4f} (z = {zP:+.2f}); хвіст {pts[-1]['tail_mean']:.1e}", flush=True)
        out[name] = dict(sizes=list(sizes), pc=pc, points=pts, sec=time.time() - t)
    zs = np.array(zs)
    summary = dict(n=len(zs), mean_z=float(zs.mean()), sd_z=float(zs.std(ddof=1)), max_abs_z=float(abs(zs).max()))
    print("підсумок", summary)
    meta = runmeta(__file__, params=dict(NROOTS=NROOTS, S_CAP=S_CAP, FACTORS=FACTORS,
                                         cases={k: list(v) for k, v in CASES.items()}), seeds=dict(root=SEED))
    save(ROOT / "results" / "P1" / "p1_3c_validate.json", meta, cases=out, summary=summary)


if __name__ == "__main__":
    main()
