"""П1.3c, доповнення після рецензії (referee_cactus_report.md, M4): усі дерева клік з z = 12 і N3 = 18…30.

Однорідна сім'я і сім'я Gleeson (p1_3c_run.py) потрапляють до ГЦК (N3 = 24) лише інтерполяцією; справжня
структура з z = 12 і N3 = 24 одна — K8 + K4 + 2K2. Тут — пік NSC_∞ для всіх розбиттів z = 12 = Σ(m_i − 1)
з N3 = Σ C(m_i − 1, 2) у [18, 30]: чи визначає N3 зсув від Бете і де стоїть справжня точка N3 = 24.
Метод — той самий, що в p1_3c_run.analyse (точні pc і P, квадратура cactusexact, нуль центральної похідної,
крок 1e-4), без кривої для рисунка; N4 = 3 Σ C(m_i − 1, 3) (усі 4-цикли — з хордою, усередині клік).

    .venv/bin/python code/P1/p1_3c_extra.py     (~30 хв)
"""
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import cactus as ca  # noqa: E402
import cactusexact as ce  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

Z, N3_LO, N3_HI, STEP = 12, 18, 30, 1e-4


def partitions(n, m=None):
    m = m or n
    if n == 0:
        yield ()
        return
    for a in range(min(n, m), 0, -1):
        for r in partitions(n - a, a):
            yield (a,) + r


def structures():
    out = []
    for parts in partitions(Z):
        n3 = sum(comb(a, 2) for a in parts)
        if N3_LO <= n3 <= N3_HI:
            out.append(tuple(a + 1 for a in parts))
    return out


def label(sizes):
    big = [m for m in sizes if m > 2]
    s = "+".join(f"K{m}" for m in big)
    j = sizes.count(2)
    return s + (f"+{j}K2" if j else "")


def analyse(sizes, pb):
    pc = ca.pc(sizes)

    def f(x, h=0.01):
        return (ce.nsc(sizes, x + STEP, h) - ce.nsc(sizes, x - STEP, h)) / (2 * STEP)

    ps = np.linspace(1.05 * pc, 1.9 * pc, 10)
    d = [f(x) for x in ps]
    i = next(i for i in range(len(d) - 1) if d[i] > 0 >= d[i + 1])
    pstar = brentq(f, ps[i], ps[i + 1], xtol=1e-8)
    return dict(label=label(sizes), **ca.structure(sizes),
                N4=3 * sum(comb(m - 1, 3) for m in sizes), pc=pc, p_star=pstar,
                P_star=ca.branching(sizes, pstar)[1], s=pstar / pc - 1, delta=pstar / pb - 1,
                p_star_h005=brentq(lambda x: f(x, 0.005), ps[i], ps[i + 1], xtol=1e-8))


def main():
    t0 = time.time()
    main_res = json.loads((ROOT / "results/P1/p1_3c_cactus.json").read_text())
    pb = main_res["bethe"]["p_star"]
    rows = []
    for sizes in structures():
        r = analyse(sizes, pb)
        rows.append(r)
        print(f"{r['label']:>16} N3={r['t']:2d} N4={r['N4']:3d} pc={r['pc']:.6f} p*={r['p_star']:.6f} "
              f"δ={100 * r['delta']:+.3f}% s={100 * r['s']:.2f}% P*={r['P_star']:.4f} "
              f"(h/2: {r['p_star_h005'] - r['p_star']:+.1e}) [{time.time() - t0:.0f} с]", flush=True)
    meta = runmeta(__file__, params=dict(Z=Z, N3_range=[N3_LO, N3_HI], STEP=STEP, h=0.01,
                                         pstar_bethe_source="results/P1/p1_3c_cactus.json:bethe.p_star"))
    save(ROOT / "results" / "P1" / "p1_3c_extra.json", meta, pstar_bethe=pb, structures=rows)
    print("час", round(time.time() - t0), "с")


if __name__ == "__main__":
    main()
