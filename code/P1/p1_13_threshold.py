"""П1.13, частина середнього поля. Поведінка NSC біля порога для ER і RRG z = 3, 4; решітки — окремо, на даних П1.2–П1.3.

Перевіряємо виведені в notes/P1_gw_identities.tex передбачення (ε = (z−1)p − 1 або c − 1, σ² — дисперсія нащадка, A = z/(z−2), для ER A = 1):
  g_∞ = a∞ − q + o(q);  NSC′(0⁺) = NSC′(0⁻) = A·(a∞ − NSC(0));  NSC″(0⁺) − NSC″(0⁻) = −8A/σ².
Запуск: .venv/bin/python code/P1/p1_13_threshold.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import gwexact as gx  # noqa: E402
from gwtree import POISSON  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

MODELS = {"ER": dict(child=(POISSON, 1.0, 0), A=1.0, s2=1.0), "z3": dict(child=gx.rrg_laws(3)[1], A=3.0, s2=0.5), "z4": dict(child=gx.rrg_laws(4)[1], A=2.0, s2=2 / 3)}
EPS_Q = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.001, 3e-4, 1e-4]
EPS_S = [0.04, 0.02, 0.01, 0.005, 0.002, 0.001]


def X(m, e):
    return 1 + e if m == "ER" else (1 + e) / (int(m[1:]) - 1)


def q_of(m, x):
    return gx.giant(x) if m == "ER" else gx._q_rrg(int(m[1:]), x)


def s1_er(a, ainf):
    """S₁ = Σ n·B(n; 1)(a_n − a∞) з хвостом за (α ln n + β)/n; потрібна лише для другої похідної зліва."""
    N = len(a) - 1
    n = np.arange(3, N + 1, dtype=float)
    b = gx.borel(N, 1.0)
    head = float(np.sum(n * b[3:] * (a[3:] - ainf))) + 1 * b[1] * (0 - ainf) + 2 * b[2] * (0 - ainf)
    w = n >= 2000
    al, be = np.linalg.lstsq(np.c_[np.log(n[w]), np.ones(w.sum())], (a[3:][w] - ainf) * n[w], rcond=None)[0]
    tail = (al * (2 * np.log(N) + 4) + 2 * be) / np.sqrt(2 * np.pi * N)
    return head + tail, float(tail)


if __name__ == "__main__":
    res = {}
    for m, p in MODELS.items():
        ainf = gx.ainf_quad(p["child"])
        N0 = gx.pair_quad(m, X(m, 0.0))
        L = p["A"] * (ainf - N0)
        rows = []
        for e in EPS_Q:
            x = X(m, e)
            pg, P, q = gx.pair_quad(m, x, survive=True), gx.giant(x, m), q_of(m, x)
            gi = pg / P
            rows.append(dict(eps=e, part_inf=pg, P=P, q=q, g_inf=gi, gap=gi - ainf, gap_over_q=(gi - ainf) / q))
        qq = np.array([r["q"] for r in rows if r["eps"] <= 0.005])
        rr = np.array([r["gap_over_q"] for r in rows if r["eps"] <= 0.005])
        co = np.linalg.lstsq(np.c_[np.ones_like(qq), qq, qq * np.log(qq)], rr, rcond=None)[0]
        sl = []
        for e in EPS_S:
            up, dn = gx.pair_quad(m, X(m, e)), gx.pair_quad(m, X(m, -e))
            sl.append(dict(eps=e, nsc_up=up, nsc_dn=dn, slope_up=(up - N0) / e, slope_dn=(N0 - dn) / e, E=(up - dn - 2 * L * e) / e ** 2,
                           C=(up + dn - 2 * N0) / e ** 2))
        ee = np.array([r["eps"] for r in sl if r["eps"] <= 0.01])
        EE = np.array([r["E"] for r in sl if r["eps"] <= 0.01])
        cE = np.linalg.lstsq(np.c_[np.ones_like(ee), ee, ee * np.log(ee)], EE, rcond=None)[0]
        res[m] = dict(ainf=ainf, nsc_c=N0, slope_pred=L, A=p["A"], s2=p["s2"], gap=rows, gap_ratio_extrap=float(co[0]), slopes=sl,
                      jump_half_pred=-4 * p["A"] / p["s2"], jump_half_extrap=float(cE[0]), jump_pred=-8 * p["A"] / p["s2"], jump_extrap=2 * float(cE[0]))
        print(m, {k: res[m][k] for k in ("ainf", "nsc_c", "slope_pred", "gap_ratio_extrap", "jump_extrap", "jump_pred")}, flush=True)
    a = np.load(ROOT / "results" / "P1" / "p1_11_a_table.npy")
    S1, tail = s1_er(a, res["ER"]["ainf"])
    D = res["ER"]["nsc_c"] - res["ER"]["ainf"]
    res["ER_second"] = dict(S1=S1, S1_tail=tail, D=D, left_pred=2 * D - S1, right_pred=2 * D - S1 - 8, sum_half_pred=2 * D - S1 - 4, sum_half_numeric=[r["C"] for r in res["ER"]["slopes"]])
    print(res["ER_second"], flush=True)
    meta = runmeta(__file__, dict(EPS_Q=EPS_Q, EPS_S=EPS_S))
    print(save(ROOT / "results" / "P1" / "p1_13_threshold.json", meta, **res))
