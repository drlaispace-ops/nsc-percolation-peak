"""П1.3c. Кактус-Бете: точний пік NSC_∞ на нескінченних деревах клік з z = 12 (дві сім'ї) і порівняння з ГЦК.

Сім'ї (z = 12, t — трикутників на вузол):
  homog — кожен вузол у k клік K_n, k(n − 1) = 12: n = 2 (Бете), 3, 4, 5, 7 (t = 0, 6, 12, 18, 30);
  gleeson — одна кліка K_c і 13 − c окремих ребер (Gleeson2009): c = 3…10 (t = C(c − 1, 2) = 1…36).
Для кожної структури: pc і P(p) — точно (cactus.branching), NSC_∞(p) і пік — точна квадратура (cactusexact);
P*, s = p*/pc − 1, δ = p*/p*_Бете − 1, крива NSC_∞(p) на сітці для рисунка.
Інтерполяція δ(t) при t = 24 (ГЦК) усередині кожної сім'ї — кубічний сплайн за t.
Порівняння з ГЦК — results/P1/p1_3b_bethe_lattices.json (lattices.fcc).

    .venv/bin/python code/P1/p1_3c_run.py     (~5–10 хв)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import cactus as ca  # noqa: E402
import cactusexact as ce  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

T_FCC = 24
FAMILIES = {
    "homog": {n: (n,) * (12 // (n - 1)) for n in (2, 3, 4, 5, 7)},
    "gleeson": {c: (c,) + (2,) * (13 - c) for c in range(3, 11)},   # c ≤ 10 (t ≤ 36) покриває t = 24
}
STEP = 1e-4


def analyse(sizes, pstar_bethe):
    pc = ca.pc(sizes)
    f = lambda x: (ce.nsc(sizes, x + STEP) - ce.nsc(sizes, x - STEP)) / (2 * STEP)  # noqa: E731
    lo, hi = 1.05 * pc, 1.9 * pc
    ps = np.linspace(lo, hi, 10)
    d = [f(x) for x in ps]
    i = next(i for i in range(len(d) - 1) if d[i] > 0 >= d[i + 1])
    pstar = brentq(f, ps[i], ps[i + 1], xtol=1e-8)
    curve = np.linspace(0.6 * pc, 2.0 * pc, 30)   # 30 вузлів: точно pc не потрапляє (на pc збіжність за k повільна)
    out = dict(**ca.structure(sizes), pc=pc, p_star=pstar, P_star=ca.branching(sizes, pstar)[1],
               nsc_star=ce.nsc(sizes, pstar), s=pstar / pc - 1,
               curve=dict(p=curve.tolist(), nsc=[ce.nsc(sizes, x) for x in curve],
                          P=[ca.branching(sizes, x)[1] for x in curve]))
    if pstar_bethe is not None:
        out["delta"] = pstar / pstar_bethe - 1
    # чутливість до кроку похідної і сітки квадратури
    out["p_star_h005"] = brentq(lambda x: (ce.nsc(sizes, x + STEP, h=0.005) - ce.nsc(sizes, x - STEP, h=0.005))
                                / (2 * STEP), ps[i], ps[i + 1], xtol=1e-8)
    return out


def main():
    t0 = time.time()
    bethe = analyse((2,) * 12, None)
    pb = bethe["p_star"]
    bethe["delta"] = 0.0
    res = {"bethe": bethe}
    print(f"Бете z=12: pc={bethe['pc']:.6f} p*={pb:.6f} P*={bethe['P_star']:.4f}", flush=True)
    for fam, items in FAMILIES.items():
        res[fam] = {}
        for key, sizes in items.items():
            r = bethe if sizes == (2,) * 12 else analyse(sizes, pb)
            res[fam][str(key)] = r
            print(f"{fam} {key}: t={r['t']} pc={r['pc']:.6f} p*={r['p_star']:.6f} δ={100 * r['delta']:+.3f}% "
                  f"s={100 * r['s']:.2f}% P*={r['P_star']:.4f} (h/2: {r['p_star_h005'] - r['p_star']:+.1e}) "
                  f"[{time.time() - t0:.0f} с]", flush=True)
    # інтерполяція при t = 24
    interp = {}
    for fam in FAMILIES:
        rows = sorted(res[fam].values(), key=lambda r: r["t"])
        t = np.array([r["t"] for r in rows], float)
        out = {}
        for q in ("delta", "s", "P_star", "p_star", "pc"):
            y = np.array([r[q] for r in rows])
            out[q] = float(CubicSpline(t, y)(T_FCC))
            out[q + "_linear"] = float(np.interp(T_FCC, t, y))
        interp[fam] = out
        print(f"{fam} при t=24: δ={100 * out['delta']:+.3f}% (лін. {100 * out['delta_linear']:+.3f}%), "
              f"s={100 * out['s']:.2f}%, P*={out['P_star']:.4f}, pc={out['pc']:.5f}", flush=True)
    fcc = json.loads((ROOT / "results/P1/p1_3b_bethe_lattices.json").read_text())["lattices"]["fcc"]
    meta = runmeta(__file__, params=dict(STEP=STEP, T_FCC=T_FCC, h=0.01,
                                         families={f: {str(k): list(v) for k, v in it.items()}
                                                   for f, it in FAMILIES.items()}))
    save(ROOT / "results" / "P1" / "p1_3c_cactus.json", meta, bethe=bethe, families={f: res[f] for f in FAMILIES},
         interp_t24=interp, fcc_ref=dict(source="results/P1/p1_3b_bethe_lattices.json:lattices.fcc",
                                         vs_bethe=fcc.get("vs_bethe")))
    print("час", round(time.time() - t0), "с")


if __name__ == "__main__":
    main()
