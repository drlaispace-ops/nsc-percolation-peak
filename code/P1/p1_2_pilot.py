"""П1.2. Пілот для бюджету П1.3: шум положення піку і частки гіганта в піку на малих L.

Для кожної решітки і кількох малих L — R_PILOT спряжених реалізацій на сітці p (одне число на
зв'язок/вузол, ті самі джерела для всієї сітки). З них:
  * SE p* і SE P(p*) бутстрепом за реалізаціями (stats.peak_bootstrap; для P — значення середньої
    кривої P(p) у p* кожної бутстреп-вибірки);
  * розклад дисперсії NSC у піку на міжграфову σ_g² і вибіркову σ_s²/M (для вибору M);
  * обсяг роботи на вузол у кожній точці сітки (для моделі вартості).
Обидві нормалізації для site: на всі вузли і на зайняті.
Запуск: .venv/bin/python code/P1/p1_2_pilot.py  →  results/P1/p1_2_pilot.json
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import sampled as sm  # noqa: E402
import stats  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

R_PILOT = int(sys.argv[1]) if len(sys.argv) > 1 else 40
M, S0, NP, B = 500, 1000, 31, 1000
# pc — з data/P1/constants.csv; сітка p — з запасом навколо очікуваного піку (CLAUDE.md: зсув 2D ≈ 0,
# 3D +7%, 4D +19%, 5D +31%); для 6D очікування не відоме, сітка ширша.
LAT = {
    "2D bond sq": dict(d=2, kind="bond", pc=0.5, Ls=[64, 128, 256], grid=(0.44, 0.60)),
    "2D site sq": dict(d=2, kind="site", pc=0.59274621, Ls=[64, 128, 256], grid=(0.55, 0.72)),
    "2D bond tri": dict(d=2, kind="tri", pc=0.3472963553, Ls=[64, 128, 256], grid=(0.30, 0.44)),
    "3D bond": dict(d=3, kind="bond", pc=0.24881182, Ls=[16, 24, 32], grid=(0.22, 0.32)),
    "4D bond": dict(d=4, kind="bond", pc=0.16013122, Ls=[8, 12, 16], grid=(0.14, 0.24)),
    "5D bond": dict(d=5, kind="bond", pc=0.11817145, Ls=[6, 8, 10], grid=(0.10, 0.20)),
    "6D bond": dict(d=6, kind="bond", pc=0.09420165, Ls=[6, 8], grid=(0.08, 0.18)),
}


def realization(cfg, L, ps, rng):
    if cfg["kind"] == "tri":
        n, u, v, w, ws = sm.coupled_triangular(L, rng)
    else:
        n, u, v, w, ws = sm.coupled_bonds(L, cfg["d"], rng, kind=cfg["kind"])
    perm = rng.permutation(n)
    out = {k: np.empty(len(ps)) for k in ("nsc", "se", "nsc_occ", "se_occ", "P", "P_occ", "work_ex", "work_s")}
    for i, p in enumerate(ps):
        A, occ = sm.at_p(n, u, v, w, p, ws)
        r = sm.nsc_sampled(A, M=M, s0=S0, perm=perm, occ=occ)
        for k in ("nsc", "se", "nsc_occ", "se_occ", "P", "P_occ"):
            out[k][i] = r[k]
        out["work_ex"][i] = r["work_exact"] / n
        out["work_s"][i] = r["work_sampled"] / n
    return out


def analyse(ps, Y, Pc, SE, rng):
    """Пік середньої кривої, SE p* і SE P(p*) бутстрепом; σ_g² і σ_s² у піку."""
    pb = stats.peak_bootstrap(ps, Y, B=B, rng=rng)
    R = len(Y)
    Pstar = []
    for _ in range(B):
        idx = rng.integers(0, R, R)
        x, _ = stats.cubic_peak(ps, Y[idx].mean(0))
        if np.isfinite(x):
            Pstar.append(np.interp(x, ps, Pc[idx].mean(0)))
    i = int(np.clip(np.searchsorted(ps, pb["x"]), 0, len(ps) - 1))
    tot = Y[:, i].var(ddof=1)
    samp = np.mean(SE[:, i] ** 2)                 # σ_s²/M у цій точці
    return dict(p_star=pb["x"], se_p=pb["se_x"], nsc_star=pb["y"], fail=pb["fail"],
                P_star=float(np.interp(pb["x"], ps, Pc.mean(0))), se_P=float(np.std(Pstar, ddof=1)),
                var_total=float(tot), var_sampling=float(samp), var_graph=float(max(tot - samp, 0.0)))


t0 = time.time()
res = {}
ss = np.random.SeedSequence(20261111)
for (name, cfg), child in zip(LAT.items(), ss.spawn(len(LAT))):
    ps = np.linspace(*cfg["grid"], NP)
    res[name] = dict(cfg=cfg, ps=ps.tolist(), L={})
    for L, lc in zip(cfg["Ls"], child.spawn(len(cfg["Ls"]))):
        t = time.time()
        rows = [realization(cfg, L, ps, np.random.default_rng(g)) for g in lc.spawn(R_PILOT)]
        sec = (time.time() - t) / R_PILOT
        S = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        rng = np.random.default_rng(lc.generate_state(1)[0])
        an = {"all": analyse(ps, S["nsc"], S["P"], S["se"], rng)}
        if cfg["kind"] == "site":
            an["occ"] = analyse(ps, S["nsc_occ"], S["P_occ"], S["se_occ"], rng)
        res[name]["L"][L] = dict(n=L ** cfg["d"], R=R_PILOT, sec_per_realization=sec, analysis=an,
                                 work_exact_per_node=S["work_ex"].mean(0).tolist(),
                                 work_sampled_per_node=S["work_s"].mean(0).tolist(),
                                 mean_nsc=S["nsc"].mean(0).tolist(), mean_P=S["P"].mean(0).tolist(),
                                 mean_nsc_occ=S["nsc_occ"].mean(0).tolist())
        a = an["all"]
        print(f"{name:12s} L={L:4d}  p*={a['p_star']:.4f}±{a['se_p']:.4f}  P*={a['P_star']:.3f}±{a['se_P']:.3f}  "
              f"σ_s²/σ²={a['var_sampling'] / a['var_total']:.2f}  fail={a['fail']:.2f}  {sec:.2f} s/реал.  [{time.time() - t0:.0f}s]",
              flush=True)

meta = runmeta(__file__, params=dict(R_pilot=R_PILOT, M=M, s0=S0, NP=NP, B=B, lattices={k: v for k, v in LAT.items()}),
               seeds=dict(root=20261111, scheme="SeedSequence(root).spawn(lattices)[i].spawn(Ls)[j].spawn(R)"))
save(ROOT / "results" / "P1" / "p1_2_pilot.json", meta, lattices=res, seconds=time.time() - t0)
