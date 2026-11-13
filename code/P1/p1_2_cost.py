"""П1.2. Вартість вибіркової NSC: час і пам'ять від n = 10⁴ до 3·10⁷, модель вартості.

Кожен розмір рахується в окремому процесі (чесний пік пам'яті, компіляція numba поза виміром).
Модель: t_gen = c_d·n (спряжені зв'язки і граф при p), t_est = a·n + b_ex·W_ex + Σ_c b(N_c)·W_c, де W —
обсяг роботи BFS (переглянуті вузли й ребра) точної частини і кожної великої компоненти c з N_c вузлами;
b(N) = b0 + b1·max(0, ln N − ln N_cache) — ціна росте, коли компонента не вміщається в кеш (перша версія з
однією сталою b недооцінювала n = 3·10⁷ до 71%). Параметри підганяються на n ≤ 10⁶ і перевіряються на
n > 10⁶ (критерій плану: похибка прогнозу ≤ 30%).
Запуск: .venv/bin/python code/P1/p1_2_cost.py  →  results/P1/p1_2_cost.json
"""
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))

M, S0 = 500, 1000
CASES = [(2, L, p) for L in (100, 178, 316, 562, 1000, 1778, 3162, 5477) for p in (0.5, 0.52)] + \
        [(3, L, p) for L in (22, 34, 46, 68, 100, 147, 215, 310) for p in (0.2488, 0.266)]
FIT_MAX = 10 ** 6


def per_work_ns(th, N):
    """Ціна одиниці роботи BFS у компоненті з N вузлами: стала, доки компонента в кеші, далі росте як ln N."""
    return th[2] + th[3] * np.maximum(0.0, np.log(N) - th[4])


def t_est(th, r):
    """Модель часу оцінювача: a·n + b_ex·W_ex + Σ_c b(N_c)·W_c (c — великі компоненти з вибіркою)."""
    return th[0] * r["n"] + 1e-9 * (th[1] * r["work_exact"] + np.sum(per_work_ns(th, np.array(r["big_N"])) * np.array(r["big_W"])))


def one(d, L, p, seed):
    import sampled as sm
    w = sm.nsc_sampled(sm.at_p(*sm.coupled_bonds(8, d, np.random.default_rng(0))[:4], p)[0], M=5, s0=10)  # компіляція
    del w
    rng = np.random.default_rng(seed)
    t = time.perf_counter()
    n, u, v, wb, _ = sm.coupled_bonds(L, d, rng)
    A, _ = sm.at_p(n, u, v, wb, p)
    t_gen = time.perf_counter() - t
    del u, v, wb
    t = time.perf_counter()
    r = sm.nsc_sampled(A, M=M, s0=S0, rng=rng)
    t_est = time.perf_counter() - t
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # байти на macOS
    keys = ("nsc", "se", "P", "n_big", "n_src_exact", "n_src_sampled", "work", "work_exact", "work_sampled")
    return dict(d=d, L=L, p=p, n=n, t_gen=t_gen, t_est=t_est, rss_gb=rss / 2 ** 30, big_N=r["big_N"], big_W=r["big_W"],
                **{k: float(r[k]) for k in keys})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        d, L, p, seed = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
        print(json.dumps(one(d, L, p, seed)))
        sys.exit()
    from runmeta import runmeta, save
    t0 = time.time()
    seeds = np.random.SeedSequence(20261110).generate_state(len(CASES))
    rows = []
    for (d, L, p), s in zip(CASES, seeds):
        run = subprocess.run([sys.executable, __file__, "--one", str(d), str(L), str(p), str(int(s))],
                             capture_output=True, text=True, check=True)
        r = json.loads(run.stdout.strip().splitlines()[-1])
        rows.append(r)
        print(f"d={d} L={L:5d} p={p:.4f} n={r['n']:.2e}  gen {r['t_gen']:7.2f} s  est {r['t_est']:7.2f} s  "
              f"W={r['work']:.2e}  RSS {r['rss_gb']:.2f} GB  P={r['P']:.3f}", flush=True)
    model, check = {}, []
    for d in (2, 3):
        R = [r for r in rows if r["d"] == d]
        fit = [r for r in R if r["n"] <= FIT_MAX]
        c = np.sum([r["t_gen"] for r in fit]) / np.sum([r["n"] for r in fit])
        th = least_squares(lambda t: [t_est(t, r) / r["t_est"] - 1 for r in fit], [5e-7, 1.0, 1.0, 0.3, np.log(1e5)],
                           bounds=([0, 0, 0, 0, np.log(1e3)], [1e-5, 20, 20, 20, np.log(1e9)])).x
        mem = np.polyfit([r["n"] for r in R], [r["rss_gb"] for r in R], 1)
        model[d] = dict(c_gen=float(c), a=float(th[0]), b_exact_ns=float(th[1]), b0_ns=float(th[2]), b1_ns=float(th[3]),
                        N_cache=float(np.exp(th[4])), rss_gb_per_n=float(mem[0]), rss_gb_0=float(mem[1]))
        for r in R:
            pred = c * r["n"] + t_est(th, r)
            obs = r["t_gen"] + r["t_est"]
            check.append(dict(d=d, L=r["L"], p=r["p"], n=r["n"], held_out=r["n"] > FIT_MAX, obs=obs, pred=float(pred),
                              rel_err=float(pred / obs - 1)))
    held = [c["rel_err"] for c in check if c["held_out"]]
    crit = bool(max(abs(e) for e in held) <= 0.30)
    for c in check:
        print(f"d={c['d']} L={c['L']:5d} p={c['p']:.4f}  факт {c['obs']:8.2f} s  прогноз {c['pred']:8.2f} s  "
              f"{c['rel_err']:+.1%}{'  (перевірка)' if c['held_out'] else ''}")
    print("модель:", model, "критерій 30%:", crit)
    meta = runmeta(__file__, params=dict(M=M, s0=S0, cases=CASES, fit_max_n=FIT_MAX),
                   seeds=dict(root=20261110, scheme="SeedSequence(root).generate_state(len(CASES))"))
    save(ROOT / "results" / "P1" / "p1_2_cost.json", meta, rows=rows, model=model, check=check,
         criterion_30pct=crit, max_abs_rel_err_held_out=float(max(abs(e) for e in held)), seconds=time.time() - t0)
