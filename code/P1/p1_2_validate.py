"""П1.2. Перевірка вибіркового оцінювача: (вибіркова − точна)/SE на незалежних графах.

Критерій плану: на ≥ 200 графах |середнє z| < 0.1 і SD z у 0.9–1.1. Перевіряємо окремо для кожної
моделі (240 графів на модель) і разом; також залежність похибки від M (джерела спільні для всіх M:
перші M у порядку однієї перестановки).
Запуск: .venv/bin/python code/P1/p1_2_validate.py  →  results/P1/p1_2_validate.json
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import perc  # noqa: E402
import sampled as sm  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 240
MS = [50, 100, 200, 500]
M0 = 200
S0 = 1000
MODELS = {
    "2D bond L=128 p=0.5": lambda g: (perc.hypercubic_bond(128, 0.5, g, 2), None),
    "2D bond L=128 p=0.53": lambda g: (perc.hypercubic_bond(128, 0.53, g, 2), None),
    "2D site L=128 p=0.5927": lambda g: perc.hypercubic_site(128, 0.592746, g, 2),
    "3D bond L=24 p=0.2488": lambda g: (perc.hypercubic_bond(24, 0.2488, g, 3), None),
    "3D bond L=24 p=0.27": lambda g: (perc.hypercubic_bond(24, 0.27, g, 3), None),
    "ER n=20000 c=1.40": lambda g: (perc.er_graph(20000, 1.40 / 20000, g), None),
}
QTY = [("nsc", "se"), ("g_giant", "se_giant"), ("g_fin", "se_fin")]

ss = np.random.SeedSequence(20261109)
children = ss.spawn(len(MODELS))
t0 = time.time()
out = {}
ZS = []
for (name, make), child in zip(MODELS.items(), children):
    rows = []
    for gs in child.spawn(R):
        g = np.random.default_rng(gs)
        A, occ = make(g)
        ex = sm.nsc_sampled(A, M=1, s0=10 ** 12, rng=g, occ=occ)       # точно: великих компонент немає
        perm = g.permutation(A.shape[0])
        row = dict(exact={q: ex[q] for q, _ in QTY}, P=ex["P"], n_big=None, est={})
        for M in MS:
            r = sm.nsc_sampled(A, M=M, s0=S0, perm=perm, occ=occ)
            row["est"][M] = {q: [r[q], r[s]] for q, s in QTY}
            row["n_big"] = r["n_big"]
        rows.append(row)
    res = {}
    for M in MS:
        d = {}
        for q, _ in QTY:
            e = np.array([[r["est"][M][q][0] - r["exact"][q], r["est"][M][q][1]] for r in rows])
            ok = e[:, 1] > 0
            z = e[ok, 0] / e[ok, 1]
            d[q] = dict(n=int(ok.sum()), mean_z=float(z.mean()) if ok.any() else None,
                        sd_z=float(z.std(ddof=1)) if ok.sum() > 1 else None,
                        rms_err=float(np.sqrt(np.mean(e[:, 0] ** 2))), mean_se=float(e[ok, 1].mean()) if ok.any() else None)
        res[M] = d
    out[name] = dict(R=R, mean_P=float(np.mean([r["P"] for r in rows])),
                     mean_n_big=float(np.mean([r["n_big"] for r in rows])),
                     mean_exact={q: float(np.mean([r["exact"][q] for r in rows])) for q, _ in QTY}, by_M=res)
    ZS.append(np.array([(r["est"][M0]["nsc"][0] - r["exact"]["nsc"]) / r["est"][M0]["nsc"][1] for r in rows
                        if r["est"][M0]["nsc"][1] > 0]))
    z = res[M0]["nsc"]
    print(f"{name:26s} P={out[name]['mean_P']:.3f} big={out[name]['mean_n_big']:.1f}  "
          f"M={M0}: z̄={z['mean_z']:+.3f} sd={z['sd_z']:.3f} rms={z['rms_err']:.2e}  [{time.time()-t0:.0f}s]", flush=True)

# разом за моделями (M = M0, NSC)
zall = np.concatenate(ZS)
pooled = dict(n=len(zall), mean_z=float(zall.mean()), sd_z=float(zall.std(ddof=1)))
print("разом:", pooled)
crit = dict(pooled=bool(abs(pooled["mean_z"]) < 0.1 and 0.9 <= pooled["sd_z"] <= 1.1))
for name in MODELS:
    z = out[name]["by_M"][M0]["nsc"]
    crit[name] = bool(abs(z["mean_z"]) < 0.1 and 0.9 <= z["sd_z"] <= 1.1)
meta = runmeta(__file__, params=dict(R=R, MS=MS, M0=M0, s0=S0, models=list(MODELS)),
               seeds=dict(root=20261109, scheme="SeedSequence(root).spawn(models)[i].spawn(R)"))
save(ROOT / "results" / "P1" / "p1_2_validate.json", meta, models=out, pooled_M0=pooled, criterion_M0=crit,
     seconds=time.time() - t0)
print("критерій:", crit)
