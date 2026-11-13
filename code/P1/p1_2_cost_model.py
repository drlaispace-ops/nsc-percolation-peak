"""П1.2. Остаточна модель вартості за замірами p1_2_cost.json (нових вимірів не робить).

Історія: (1) t = a·n + b·W з однією b, підгонка на n ≤ 10⁶ — недооцінка до −71% при 3·10⁷
(p1_2_cost_linear.json); (2) b(N_c) з одним зламом (кеш), підгонка на n ≤ 10⁶ — −46% для 3D-гіганта
(p1_2_cost.json, поле check). Екстраполяцію з n ≤ 10⁶ за межу оперативної пам'яті модель не дає.
Тут: b(N) = b0 + b1·(ln N − ln N1)₊ + b2·(ln N − ln N2)₊ (кеш і пам'ять), підгонка на всіх розмірах
до 3·10⁷ — розміри П1.3 лежать у цьому діапазоні; перевірка — перехресна «без одного L». Також
повторюваність часу: обидва прогони (linear і поточний) мали ті самі сіди, тобто ті самі графи.
Запуск: .venv/bin/python code/P1/p1_2_cost_model.py  →  results/P1/p1_2_cost_model.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from runmeta import runmeta, save  # noqa: E402

RES = ROOT / "results" / "P1"
rows = json.load(open(RES / "p1_2_cost.json"))["rows"]
rows0 = json.load(open(RES / "p1_2_cost_linear.json"))["rows"]
X0 = [5e-7, 1, 1, .3, np.log(1e5), .5, np.log(3e6)]
LB = [0, 0, 0, 0, np.log(1e3), 0, np.log(1e5)]
UB = [1e-5, 20, 20, 20, np.log(1e9), 20, np.log(1e9)]
TOL = 0.30


def per_work_ns(th, N):
    N = np.asarray(N, float)
    return th[2] + th[3] * np.maximum(0, np.log(N) - th[4]) + th[5] * np.maximum(0, np.log(N) - th[6])


def t_est(th, n, W_ex, big_N, big_W):
    return th[0] * n + 1e-9 * (th[1] * W_ex + np.sum(per_work_ns(th, big_N) * np.asarray(big_W, float)))


def fit(rs):
    return least_squares(lambda t: [t_est(t, r["n"], r["work_exact"], r["big_N"], r["big_W"]) / r["t_est"] - 1 for r in rs],
                         X0, bounds=(LB, UB)).x


model, loo, repeat = {}, [], []
for d in (2, 3):
    rs = [r for r in rows if r["d"] == d]
    th = fit(rs)
    Ls = sorted({r["L"] for r in rs})
    mem = np.polyfit([r["n"] for r in rs], [r["rss_gb"] for r in rs], 1)
    model[d] = dict(c_gen=float(np.sum([r["t_gen"] for r in rs]) / np.sum([r["n"] for r in rs])), theta=th.tolist(),
                    theta_names=["a", "b_exact_ns", "b0_ns", "b1_ns", "lnN1", "b2_ns", "lnN2"],
                    N1=float(np.exp(th[4])), N2=float(np.exp(th[6])), rss_gb_per_n=float(mem[0]), rss_gb_0=float(mem[1]),
                    resid=[float(t_est(th, r["n"], r["work_exact"], r["big_N"], r["big_W"]) / r["t_est"] - 1) for r in rs])
    for Lh in Ls:
        th_h = fit([r for r in rs if r["L"] != Lh])
        for r in rs:
            if r["L"] == Lh:
                loo.append(dict(d=d, L=Lh, p=r["p"], n=r["n"], edge=Lh in (Ls[0], Ls[-1]), t=r["t_est"],
                                rel_err=float(t_est(th_h, r["n"], r["work_exact"], r["big_N"], r["big_W"]) / r["t_est"] - 1)))
    for r, r0 in zip(rs, [r for r in rows0 if r["d"] == d]):
        assert (r["L"], r["p"], r["work"]) == (r0["L"], r0["p"], r0["work"])   # ті самі графи
        repeat.append(dict(d=d, L=r["L"], p=r["p"], t1=r0["t_est"], t2=r["t_est"], rel=float(r["t_est"] / r0["t_est"] - 1)))

inner = [x["rel_err"] for x in loo if not x["edge"]]
crit = bool(max(abs(e) for e in inner) <= TOL)
for x in loo:
    print(f"d={x['d']} L={x['L']:5d} p={x['p']:.4f}  t={x['t']:8.2f} s  без цього L: {x['rel_err']:+.0%}{'  (край)' if x['edge'] else ''}")
big = [x["rel"] for x in repeat if x["t1"] > 1]
print("повторюваність (t > 1 с): медіана |Δ| = %.0f%%, макс %.0f%%" % (100 * np.median(np.abs(big)), 100 * np.max(np.abs(big))))
print("внутрішні L: макс |похибка| = %.0f%%, критерій %d%%: %s" % (100 * max(abs(e) for e in inner), 100 * TOL, crit))
meta = runmeta(__file__, params=dict(tol=TOL, inputs=["p1_2_cost.json", "p1_2_cost_linear.json"]))
save(RES / "p1_2_cost_model.json", meta, model=model, loo=loo, repeat=repeat, criterion_inner_30pct=crit,
     max_abs_inner=float(max(abs(e) for e in inner)),
     repeat_median_abs=float(np.median(np.abs(big))), repeat_max_abs=float(np.max(np.abs(big))))
