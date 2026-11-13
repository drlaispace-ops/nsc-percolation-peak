"""Незалежна перевірка завдання B (П1.15). Не імпортує p1_15_*.py і p1_3_run.py.
Читає сирі шматки П1.3 (data/P1/p1_3/...), сітку p з results/P1/p1_3/*.json і покластерні шматки data/P1/p1_15/.
Запуск: NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=1 .venv/bin/python notes/orchestra/verify_B.py
"""
import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import stats as sst

ROOT = Path(__file__).resolve().parents[2]
J = json.load(open(ROOT / "results/P1/p1_15_classes.json"))
rng = np.random.default_rng(777)
OUT = {}
PC = {"2D_bond_sq": 0.5, "2D_bond_tri": 2 * np.sin(np.pi / 18), "2D_site_sq": 0.59274621, "3D_bond": 0.24881182}


def load_journal(slug, L):
    ps = np.array(json.load(open(ROOT / f"results/P1/p1_3/{slug}_L{L}.json"))["ps"])
    fs = sorted(glob.glob(str(ROOT / f"data/P1/p1_3/{slug}/L{L}/chunk_*.npz")))
    c, n, sc = [], [], []
    for f in fs:
        z = np.load(f)
        c.append(z["sbin_c"]); n.append(z["sbin_n"]); sc.append(z["sbin_sc"])
    return ps, np.concatenate(c).astype(float), np.concatenate(n).astype(float), np.concatenate(sc)


def part_a(slug, L, b, B=2000):
    """g_b(p) у вікні |ε| ≤ 0.1 БЕЗ інтерпольованої точки pc; кінцеві точки, нахил МНК, бутстреп за реалізаціями
    (індекси, а не мультиноміальні ваги); також GLS-χ² сталості з повною бутстреп-коваріацією на проріджених точках."""
    pc = PC[slug]
    ps, c, n, sc = load_journal(slug, L)
    R = len(c)
    eps = (ps - pc) / pc
    tot_c = c.sum(0)
    nr = (c > 0).sum(0)
    sel = (np.abs(eps) <= 0.1 + 1e-9) & (tot_c[:, b] >= 30) & (nr[:, b] >= 10)
    x = ps[sel]
    g = sc[:, sel, b].sum(0) / n[:, sel, b].sum(0)
    sbar = n[:, sel, b].sum(0) / c[:, sel, b].sum(0)
    gb = np.empty((B, sel.sum()))
    for i in range(B):
        j = rng.integers(0, R, R)
        gb[i] = sc[j][:, sel, b].sum(0) / n[j][:, sel, b].sum(0)
    sl = np.polyfit(x, g, 1)[0]
    slb = np.array([np.polyfit(x, y, 1)[0] for y in gb])
    d_ends = g[0] - g[-1]
    sd_ends = np.std(gb[:, 0] - gb[:, -1], ddof=1)
    # GLS сталість (кожна 3-тя точка, щоб коваріація була оборотною й стабільною)
    k = np.arange(0, len(x), 3)
    C = np.cov(gb[:, k].T)
    Ci = np.linalg.inv(C)
    one = np.ones(len(k))
    mu = one @ Ci @ g[k] / (one @ Ci @ one)
    r = g[k] - mu
    chi2 = float(r @ Ci @ r)
    # у pc: інтерполяція між сусідами
    kk = np.searchsorted(x, pc)
    w = (pc - x[kk - 1]) / (x[kk] - x[kk - 1])
    gpc = (1 - w) * g[kk - 1] + w * g[kk]
    return dict(L=L, bin=b, R=R, p=x.tolist(), g=g.tolist(), g_pc_interp=float(gpc), sbar=sbar.tolist(),
                p_ends=[float(x[0]), float(x[-1])], g_ends=[float(g[0]), float(g[-1])],
                delta_ends=float(d_ends), se_delta=float(sd_ends), slope=float(sl), se_slope=float(np.std(slb, ddof=1)),
                z_slope=float(sl / np.std(slb, ddof=1)), gls_chi2=chi2, gls_dof=len(k) - 1,
                gls_p=float(sst.chi2.sf(chi2, len(k) - 1)), sbar_rel_shift=float(sbar.max() / sbar.min() - 1))


def jrow(name, L, b):
    T = next(t for t in J["journal"][name] if t["L"] == L)
    r = next(r for r in T["rows"] if r["bin"] == b)
    return r


print("=== Частина A: журнал П1.3 ===")
A = {}
for slug, name, L, b in [("2D_bond_sq", "2D bond sq", 1024, 8), ("3D_bond", "3D bond", 256, 8),
                         ("2D_bond_sq", "2D bond sq", 512, 8), ("3D_bond", "3D bond", 128, 8),
                         ("2D_site_sq", "2D site sq", 512, 13), ("3D_bond", "3D bond", 256, 14),
                         ("2D_bond_sq", "2D bond sq", 1024, 14)]:
    v = part_a(slug, L, b)
    r = jrow(name, L, b)
    v["executor"] = dict(g_first=r["g"][0], g_pc=r["g_pc"], g_last=r["g"][-1], delta_ends=r["delta_ends"], se_delta=r["se_delta"],
                         slope=r["slope"], se_slope=r["se_slope"], z_slope=r["z_slope"], sbar_shift_rel=r["sbar_shift_rel"],
                         corr_max=r["corr_max"], z_slope_corr=r["z_slope_corr"])
    A[f"{name} L{L} b{b}"] = v
    print(f"{name} L={L} b={b}: g {v['g_ends'][0]:.4f} / pc {v['g_pc_interp']:.4f} / {v['g_ends'][1]:.4f}; "
          f"Δ = {v['delta_ends']:.4f} ± {v['se_delta']:.4f}; нахил {v['slope']:.4f} ± {v['se_slope']:.4f} (z {v['z_slope']:.1f}); "
          f"GLS χ²/dof = {v['gls_chi2']:.1f}/{v['gls_dof']} (p = {v['gls_p']:.2g}); s̄ зсув {v['sbar_rel_shift']:.3f}")
    print("   виконавець:", {k: round(x, 4) for k, x in v["executor"].items()})
OUT["part_A"] = A

# Порівняння L для нахилів (власні)
for nm, a, bb in [("2D bond sq L512/1024 b8", "2D bond sq L512 b8", "2D bond sq L1024 b8"),
                  ("3D L128/256 b8", "3D bond L128 b8", "3D bond L256 b8")]:
    x, y = A[a], A[bb]
    z = (x["slope"] - y["slope"]) / np.hypot(x["se_slope"], y["se_slope"])
    print(f"контроль L {nm}: нахили {x['slope']:.4f}±{x['se_slope']:.4f} і {y['slope']:.4f}±{y['se_slope']:.4f}, z = {z:.2f}")
    OUT.setdefault("L_slopes", {})[nm] = float(z)

# Твердження з таблиці L_compare
lc = {}
for name, prs in J["L_compare"].items():
    for pr in prs:
        lc[f"{name} {pr['L1']}/{pr['L2']}"] = max(abs(b["z_slope_diff"]) for b in pr["bins"])
print("max |z_slope_diff| за парами L (з JSON):", {k: round(v, 2) for k, v in lc.items()})
OUT["L_compare_maxz_from_json"] = lc

# Мінімальне |z| для найбільших L (з JSON)
minz = {}
for name, T in J["journal"].items():
    t = T[-1]
    minz[f"{name} L{t['L']}"] = min(abs(r["z_slope"]) for r in t["rows"])
print("min |z_slope| для найбільших L (з JSON):", {k: round(v, 1) for k, v in minz.items()})
OUT["min_z_from_json"] = minz

# ---------------------------------------------------------------- частина B
print("\n=== Частина B: покластерні дані ===")


def load_cl(slug, keys=("s", "c", "E", "g", "rid", "lmean", "diam", "deg")):
    fs = sorted(glob.glob(str(ROOT / f"data/P1/p1_15/{slug}/chunk_*.npz")))
    out = {f: {k: [] for k in keys} for f in ("0.9", "1.0", "1.1")}
    for p in fs:
        z = np.load(p)
        for f in out:
            for k in keys:
                out[f][k].append(z[f"{f}_{k}"])
    return {f: {k: np.concatenate(v) for k, v in o.items()} for f, o in out.items()}


def exact_s_shift(D, lo, hi, fa, fb="1.0", B=300):
    """Стратифікація за ТОЧНИМ s: Σ_s w_s [ḡ_a(s) − ḡ_b(s)] / Σ w_s, w_s = 1/(1/n_a + 1/n_b); бутстреп за реалізаціями."""
    def est(ia, ib):
        sa, ga = D[fa]["s"][ia], D[fa]["g"][ia]
        sb_, gb_ = D[fb]["s"][ib], D[fb]["g"][ib]
        na = np.bincount(sa - lo, minlength=hi - lo).astype(float)
        nb_ = np.bincount(sb_ - lo, minlength=hi - lo).astype(float)
        ma = np.bincount(sa - lo, ga, hi - lo) / np.maximum(na, 1)
        mb = np.bincount(sb_ - lo, gb_, hi - lo) / np.maximum(nb_, 1)
        ok = (na >= 2) & (nb_ >= 2)
        w = 1 / (1 / na[ok] + 1 / nb_[ok])
        return float(np.sum(w * (ma[ok] - mb[ok])) / np.sum(w))
    ia = np.flatnonzero((D[fa]["s"] >= lo) & (D[fa]["s"] < hi))
    ib = np.flatnonzero((D[fb]["s"] >= lo) & (D[fb]["s"] < hi))
    e = est(ia, ib)
    ra, rb = D[fa]["rid"][ia], D[fb]["rid"][ib]
    R = int(max(ra.max(), rb.max())) + 1
    ga_ = [ia[ra == r] for r in range(R)]
    gb_l = [ib[rb == r] for r in range(R)]
    bs = []
    for _ in range(B):
        pick = rng.integers(0, R, R)   # та сама реалізація для обох p (спряжені)
        bs.append(est(np.concatenate([ga_[r] for r in pick]), np.concatenate([gb_l[r] for r in pick])))
    return float(e), float(np.std(bs, ddof=1))


def eta(y, X0, X1):
    r0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    r1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    return 1 - r1.var() / r0.var()


B_OUT = {}
for slug, name, d in [("2D_bond_sq_L256", "2D bond sq", 2), ("3D_bond_L48", "3D bond", 3)]:
    D = load_cl(slug)
    res = {}
    # тотожність: c/s визначається гістограмою степенів
    o = D["1.0"]
    k = np.arange(1, 2 * d + 1)
    cs_deg = 0.5 * (o["deg"] * k).sum(1) / o["s"] - 1 + 1 / o["s"]
    res["max_abs_cs_minus_degree_identity"] = float(np.max(np.abs(cs_deg - o["c"] / o["s"])))
    res["E_check"] = int(np.max(np.abs(o["E"] - (o["c"] + o["s"] - 1))))
    # η при pc, біни 6–8
    for b in (6, 7, 8):
        i = np.flatnonzero((o["s"] >= 2 ** b) & (o["s"] < 2 ** (b + 1)))
        s = o["s"][i].astype(float); g = o["g"][i]; cs = o["c"][i] / s; ll = np.log(o["lmean"][i])
        X0 = np.c_[np.ones_like(s), np.log(s)]
        res[f"eta_pc_b{b}"] = dict(cs=float(eta(g, X0, np.c_[X0, cs, cs ** 2])), ell=float(eta(g, X0, np.c_[X0, ll, ll ** 2])),
                                   ell_lin=float(eta(g, X0, np.c_[X0, ll])), frac_cyclic=float(np.mean(o["c"][i] > 0)), n=int(len(i)))
    # Зсув між p за точним s (бін 6 і вузький діапазон)
    for fa in ("0.9", "1.1"):
        for lo, hi in ((64, 128), (128, 256)):
            res[f"shift_exact_s_{fa}-1.0_[{lo},{hi})"] = exact_s_shift(D, lo, hi, fa)
    # Вузлове середнє в біні 8 — порівняння з журналом П1.3 того самого L (= 256 для 2D) — лише 2D
    if d == 2:
        for f in ("0.9", "1.0", "1.1"):
            i = np.flatnonzero((D[f]["s"] >= 256) & (D[f]["s"] < 512))
            res[f"g_nodeweighted_b8_{f}"] = float((D[f]["g"][i] * D[f]["s"][i]).sum() / D[f]["s"][i].sum())
            res[f"g_clusterweighted_b8_{f}"] = float(D[f]["g"][i].mean())
    # AUROC 0.9/1.1 бін 6, власний поділ (не той самий піднабір — очікуємо близькі числа)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    ia = np.flatnonzero((D["0.9"]["s"] >= 64) & (D["0.9"]["s"] < 128))
    ib = np.flatnonzero((D["1.1"]["s"] >= 64) & (D["1.1"]["s"] < 128))
    m = min(len(ia), len(ib), 4000)
    ia, ib = rng.choice(ia, m, replace=False), rng.choice(ib, m, replace=False)

    def F(f, i):
        s = D[f]["s"][i].astype(float)
        return dict(lns=np.log(s), cs=D[f]["c"][i] / s, g=D[f]["g"][i]), D[f]["rid"][i]
    Fa, ra = F("0.9", ia); Fb, rb = F("1.1", ib)
    y = np.r_[np.ones(m), np.zeros(m)]
    rid = np.r_[ra, rb]
    tr = rid % 2 == 1          # навпаки до виконавця: непарні — навчання
    au = {}
    for nm, keys in {"ln s": ["lns"], "ln s + NodeSC": ["lns", "g"], "ln s + c/s": ["lns", "cs"], "ln s + c/s + NodeSC": ["lns", "cs", "g"]}.items():
        X = np.c_[tuple(np.r_[Fa[k_], Fb[k_]] for k_ in keys)]
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[tr], y[tr])
        au[nm] = float(roc_auc_score(y[~tr], mdl.decision_function(X[~tr])))
    res["auroc_0.9_vs_1.1_b6"] = au
    # η при 0.9 і 1.1 для c/s, бін 6 (перевірка «8–15%»)
    for f in ("0.9", "1.1"):
        o2 = D[f]
        i = np.flatnonzero((o2["s"] >= 64) & (o2["s"] < 128))
        s = o2["s"][i].astype(float); g = o2["g"][i]; cs = o2["c"][i] / s
        X0 = np.c_[np.ones_like(s), np.log(s)]
        res[f"eta_cs_b6_{f}"] = float(eta(g, X0, np.c_[X0, cs, cs ** 2]))
    B_OUT[name] = res
    print(name, json.dumps(res, ensure_ascii=False, indent=0))

    # виконавець
    cyc = J["clusters"][name]["cycles"]
    for r in cyc:
        if r["bin"] in (6, 7, 8):
            print(f"  виконавець бін {r['bin']}: η(c/s)@pc = {r['by_p']['1.0']['eta']['c/s']['est']:.3f}, "
                  f"η(⟨ℓ⟩)@pc = {r['by_p']['1.0']['eta']['⟨ℓ⟩']['est']:.3f}, "
                  f"shift ln s 0.9−1.0 = {r['shift']['ln s']['0.9-1.0']['est']:.4f} ± {r['shift']['ln s']['0.9-1.0']['se']:.4f}, "
                  f"1.1−1.0 = {r['shift']['ln s']['1.1-1.0']['est']:.4f} ± {r['shift']['ln s']['1.1-1.0']['se']:.4f}")
OUT["part_B"] = B_OUT

# Журнал П1.3 2D L=256 при найближчих p (порівняння з вузловим середнім частини B)
ps, c, n, sc = load_journal("2D_bond_sq", 256)
g = sc[:, :, 8].sum(0) / n[:, :, 8].sum(0)
OUT["journal_2D_L256_b8"] = {f"{p:.5f}": float(v) for p, v in zip(ps, g) if 0.44 <= p <= 0.56}
print("журнал П1.3 2D L256 бін 8:", {k: round(v, 4) for k, v in OUT["journal_2D_L256_b8"].items()})

# Біноміальна ймовірність ≥ 5 із 36 при 5%
OUT["binom_ge5_of_36"] = float(sst.binom.sf(4, 36, 0.05))
print("P(X ≥ 5 | 36, 0.05) =", OUT["binom_ge5_of_36"])

# Позитивний контроль і dip (з JSON)
cl = {nm: [(r["bin"], {k: (v["groups_dip"], v["groups_dbic"]) for k, v in r.get("positive_control_0.9_vs_1.1", {}).items()})
           for r in J["clusters"][nm]["classes"]] for nm in J["clusters"]}
print("позитивний контроль (dip, ΔBIC):", cl)
nd = sum(v["groups_dip"] for nm in J["clusters"] for r in J["clusters"][nm]["classes"] for v in r["feats"].values())
nb_ = sum(v["groups_dbic"] for nm in J["clusters"] for r in J["clusters"][nm]["classes"] for v in r["feats"].values())
ntot = sum(len(r["feats"]) for nm in J["clusters"] for r in J["clusters"][nm]["classes"])
OUT["classes_counts"] = dict(dip=int(nd), dbic=int(nb_), total=int(ntot))
print("dip / ΔBIC перевищень:", nd, nb_, "із", ntot)

json.dump(OUT, open(ROOT / "notes/orchestra/verify_B.json", "w"), ensure_ascii=False, indent=1)
print("записано notes/orchestra/verify_B.json")
