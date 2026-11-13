"""П1.14. Форма дерева за заданого розміру: точний P(C | n), нульова модель, позитивні контролі, приріст NSC понад базовий рівень.

Закони: Пуассон, RRG z = 3, 4 (унімодулярні), плоский геометричний (типовий вузол).
Статистики груп порівнюються з власним розподілом на нульовій моделі, а не з табличними порогами.
Контрольні точки: data/P1/p1_14/. Запуск: .venv/bin/python code/P1/p1_14_shapes.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import multiprocessing as mp
import sys
import time
from pathlib import Path

import diptest
import numba as nb
import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import gwexact as gx  # noqa: E402
import gwtree as gt  # noqa: E402
import treeenum as te  # noqa: E402
import treefeat as tf  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

SEED = 202611071
POI, GEO = (gt.POISSON, 1.0, 0), (gt.GEOM, 0.5, 0)
LAWS = {"poisson": (POI, POI), "z3": gx.rrg_laws(3), "z4": gx.rrg_laws(4), "geom": (GEO, GEO)}
EXACT_N = list(range(6, 25)) + [26]
POOL_N = [24, 30, 100, 300, 1000, 3000]
POOL_LAWS = ["poisson", "z4", "geom"]
M_POOL = 200_000
K_POOL = 32
M_DATA = 1000  # дерев в одному наборі даних
B = 1000  # наборів на сценарій
TASKS = {"T1 Пуассон/геом": ("poisson", "geom"), "T2 Пуассон/z=4": ("poisson", "z4"), "T3 рекурсивне/геом": ("rrt", "geom")}
TASK_N = [15, 30, 100, 300]
R_CLASS = 3000
DATA = ROOT / "data" / "P1" / "p1_14"
OUT = ROOT / "results" / "P1"


# --- Точний P(C | n) ---
def exact_one(n):
    path = DATA / f"exact_{n}.npz"
    if not path.exists():
        pm = [(gx.law_pmf(r, n), gx.law_pmf(c, n)) for r, c in LAWS.values()]
        rows, count = te.shape_stream(n, pm)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, count=count, **{k: np.array([r[k] for r in rows]) for k in ("mass", "mean", "sd", "skew", "cmin", "cmax", "hist")})
        os.replace(tmp, path)
    return n


# --- Вибірки ---
@nb.njit(parallel=True, cache=True)
def _pool(n, seeds, cdf, mode, b, g, K):
    M = seeds.size
    C = np.zeros(M)
    D = np.zeros((M, K))
    for i in nb.prange(M):
        acc = np.zeros(K)
        sc = gt.tree_sc_reroot(gt.cond_law_parent(n, seeds[i], cdf, mode, b), n, g, K, acc)
        C[i] = sc.mean()
        D[i] = acc / n
    return C, D


def pool(name, n, seq):
    path = DATA / f"pool_{name}_{n}.npz"
    if not path.exists():
        root, child = LAWS[name]
        C, D = _pool(n, gt.seeds_from(seq, M_POOL), gx.root_given_n(n, root, child), child[0], int(child[2]), gt.gtable(n), K_POOL)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, C=C, D=D.astype(np.float32))
        os.replace(tmp, path)
    z = np.load(path)
    return z["C"], z["D"].astype(float)


def dbic(x):
    """ΔBIC 1 проти 2 гаусіан на одновимірній вибірці; > 0 на користь двох."""
    from sklearn.mixture import GaussianMixture
    x = x.reshape(-1, 1)
    m = len(x)
    ll1 = -0.5 * m * (np.log(2 * np.pi * x.var()) + 1)
    bic1 = -2 * ll1 + 2 * np.log(m)
    g = GaussianMixture(2, n_init=1, reg_covar=1e-8, random_state=0, max_iter=200).fit(x)
    return float(bic1 - g.bic(x))


def silhouette2(x):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    x = x.reshape(-1, 1)
    return float(silhouette_score(x, KMeans(2, n_init=3, random_state=0).fit_predict(x)))


def stats_of(c, p1, p2, naive):
    d, pv = diptest.diptest(c)
    row = dict(dip=float(d), dip_p=float(pv), dbic=dbic(c), dip_pc1=float(diptest.dipstat(p1)), dip_pc2=float(diptest.dipstat(p2)))
    if naive:
        row["sil"] = silhouette2(c)
    return row


def run_cell(task):
    """Для одного n: пороги на половині A кожного закону, сценарії на половині B."""
    n, seed = task
    rng = np.random.default_rng(seed)
    data = {}
    for name in POOL_LAWS:
        z = np.load(DATA / f"pool_{name}_{n}.npz")
        C, D = z["C"], z["D"].astype(float)
        half = M_POOL // 2
        data[name] = dict(Ca=C[:half], Da=D[:half], Cb=C[half:], Db=D[half:])
    pca = {}
    for name, x in data.items():
        mu = x["Da"][:20000].mean(0)
        _, _, vt = np.linalg.svd(x["Da"][:20000] - mu, full_matrices=False)
        pca[name] = (mu, vt[:2])

    def draw(name, half, m):
        x = data[name]
        idx = rng.choice(len(x["C" + half]), m, replace=False)
        return x["C" + half][idx], x["D" + half][idx]

    def make(ref, comps, naive=False):
        """comps: [(закон, частка)]; проєкції — на PCA еталонного закону."""
        cs, ds = [], []
        left = M_DATA
        for i, (name, w) in enumerate(comps):
            m = left if i == len(comps) - 1 else int(round(w * M_DATA))
            left -= m
            c, d = draw(name, comps_half, m)
            cs.append(c)
            ds.append(d)
        c, d = np.concatenate(cs), np.concatenate(ds)
        mu, vt = pca[ref]
        pr = (d - mu) @ vt.T
        return stats_of(c, pr[:, 0], pr[:, 1], naive)

    res = dict(n=n, ref={}, scen={})
    comps_half = "a"
    for ref in POOL_LAWS:
        rows = [make(ref, [(ref, 1.0)]) for _ in range(B)]
        res["ref"][ref] = {k: float(np.quantile([r[k] for r in rows], 0.95)) for k in ("dip", "dbic", "dip_pc1", "dip_pc2")}
    comps_half = "b"
    scen = {"нуль: Пуассон": ("poisson", [("poisson", 1.0)]), "нуль: z=4": ("z4", [("z4", 1.0)]), "нуль: геом": ("geom", [("geom", 1.0)]),
            "чужий еталон: геом проти Пуассона": ("poisson", [("geom", 1.0)]), "чужий еталон: z=4 проти Пуассона": ("poisson", [("z4", 1.0)]),
            "суміш Пуассон/геом 50/50": ("poisson", [("poisson", 0.5), ("geom", 0.5)]), "суміш Пуассон/геом 90/10": ("poisson", [("poisson", 0.9), ("geom", 0.1)]),
            "суміш Пуассон/z=4 50/50": ("poisson", [("poisson", 0.5), ("z4", 0.5)])}
    for name, (ref, comps) in scen.items():
        rows = [make(ref, comps, naive=(i < 200)) for i in range(B)]
        q = res["ref"][ref]
        res["scen"][name] = dict(cal_dip=float(np.mean([r["dip"] > q["dip"] for r in rows])), cal_dbic=float(np.mean([r["dbic"] > q["dbic"] for r in rows])),
                                 cal_pc1=float(np.mean([r["dip_pc1"] > q["dip_pc1"] for r in rows])), cal_pc2=float(np.mean([r["dip_pc2"] > q["dip_pc2"] for r in rows])),
                                 table_dip=float(np.mean([r["dip_p"] < 0.05 for r in rows])), naive_dbic10=float(np.mean([r["dbic"] > 10 for r in rows])),
                                 naive_sil05=float(np.mean([r["sil"] > 0.5 for r in rows if "sil" in r])), mean_sil=float(np.mean([r["sil"] for r in rows if "sil" in r])))
    return res


# --- Ознаки: AUROC за однакового n ---
def rrt_parent(n, rng):
    return np.r_[-1, [rng.integers(0, i) for i in range(1, n)]].astype(np.int64)


def sample_task(args):
    name, n, seq = args
    path = DATA / f"feat_{name}_{n}.npz"
    if path.exists():
        return str(path)
    seeds = gt.seeds_from(seq, R_CLASS)
    cdf = None
    if name != "rrt":
        root, child = LAWS[name]
        cdf = gx.root_given_n(n, root, child)
    rows = []
    for s in seeds:
        if name == "rrt":
            par = rrt_parent(n, np.random.default_rng(int(s)))
        else:
            par = gt.cond_law_parent(n, int(s), cdf, child[0], int(child[2]))
        rows.append(tf.tree_features(par))
    out = {k: np.array([r[k] for r in rows]) for k in ("C", "diam", "deg", "nsc", "portrait", "netlsd")}
    degs = np.array([r["degs"] for r in rows])  # кількість вузлів кожного степеня: для оракула
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, degs=degs.astype(np.int32), **out)
    os.replace(tmp, path)
    return str(path)


SETS = {"C": ["C"], "діаметр": ["diam"], "степені": ["deg"], "база": ["diam", "deg"], "NSC": ["nsc"], "портрет": ["portrait"], "NetLSD": ["netlsd"],
        "база+C": ["diam", "deg", "C"], "база+NSC": ["diam", "deg", "nsc"], "база+портрет": ["diam", "deg", "portrait"], "база+NetLSD": ["diam", "deg", "netlsd"]}
PAIRS = [("база+C", "база"), ("база+NSC", "база"), ("база+портрет", "база"), ("база+NetLSD", "база"), ("NSC", "діаметр"), ("NSC", "степені"), ("NSC", "портрет"), ("NSC", "NetLSD")]


def oracle(a, b, degs):
    """Логарифм відношення правдоподібності законів ГВ залежить лише від кількостей вузлів за степенем (notes, П1.14); для T3 його немає."""
    d = np.arange(degs.shape[1])
    def lw(name):
        if name == "poisson":
            return np.zeros_like(d, float)  # μ(c)·c! стала
        if name == "geom":
            return np.where(d >= 1, gammaln(np.maximum(d, 1)), -np.inf) + d * np.log(0.5)
        z = int(name[1:])
        return np.where((d >= 1) & (d <= z), -gammaln(np.maximum(z - d + 1, 1)) + d * np.log(1 / (z - 2)), -np.inf)
    la, lb = lw(a), lw(b)
    pos = degs > 0
    with np.errstate(invalid="ignore"):
        A = np.where(pos, degs * la, 0.0).sum(1)
        B_ = np.where(pos, degs * lb, 0.0).sum(1)
    return A - B_


def eval_task(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    title, a, b, n, seed = args
    za, zb = np.load(DATA / f"feat_{a}_{n}.npz"), np.load(DATA / f"feat_{b}_{n}.npz")
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(R_CLASS), np.zeros(R_CLASS)].astype(int)
    perm = rng.permutation(2 * R_CLASS)
    tr, te_ = perm[: R_CLASS], perm[R_CLASS:]
    scores, res = {}, dict(task=title, n=n, auc={}, auc_hgb={}, ci={}, diff={}, diff_hgb={}, roc={})
    for name, groups in SETS.items():
        X = np.c_[tuple(np.r_[za[g], zb[g]] for g in groups)]
        s1 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], y[tr]).decision_function(X[te_])
        s2 = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr]).decision_function(X[te_])
        scores[name] = (s1, s2)
        res["auc"][name] = float(roc_auc_score(y[te_], s1))
        res["auc_hgb"][name] = float(roc_auc_score(y[te_], s2))
    if b != "rrt" and a != "rrt":
        degs = np.r_[za["degs"], zb["degs"]]
        s = oracle(a, b, degs)[te_]
        s = np.nan_to_num(s, posinf=1e6, neginf=-1e6)
        res["auc_oracle"] = float(roc_auc_score(y[te_], s))
    idx = [rng.integers(0, len(te_), len(te_)) for _ in range(2000)]
    yt = y[te_]
    ok = [i for i in idx if yt[i].min() != yt[i].max()]
    for name, (s1, _) in scores.items():
        v = [roc_auc_score(yt[i], s1[i]) for i in ok[:1000]]
        res["ci"][name] = [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    for A, B_ in PAIRS:
        for which, key in ((0, "diff"), (1, "diff_hgb")):
            sa, sb = scores[A][which], scores[B_][which]
            v = [roc_auc_score(yt[i], sa[i]) - roc_auc_score(yt[i], sb[i]) for i in ok]
            res[key][f"{A} − {B_}"] = dict(est=float(roc_auc_score(yt, sa) - roc_auc_score(yt, sb)), lo=float(np.quantile(v, 0.025)), hi=float(np.quantile(v, 0.975)))
    for name in ("C", "діаметр", "степені", "база", "NSC", "портрет", "NetLSD", "база+NSC"):
        fpr, tpr, _ = roc_curve(yt, scores[name][0])
        keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
        res["roc"][name] = [fpr[keep].tolist(), tpr[keep].tolist()]
    return res


if __name__ == "__main__":
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    res = {}
    ss = np.random.SeedSequence(SEED)
    s_pool, s_cell, s_feat, s_eval = ss.spawn(4)

    # --- Точний закон (один процес, щоб не заважати паралельним numba далі) ---
    with mp.get_context("spawn").Pool(2) as ex:
        ex.map(exact_one, EXACT_N[:-1], chunksize=1)
    print("exact ≤24", time.time() - t0, flush=True)
    exact_one(EXACT_N[-1])
    print("exact 26", time.time() - t0, flush=True)
    a = {"poisson": np.load(OUT / "p1_11_a_table.npy"), "z3": np.load(OUT / "p1_12_a_tables.npz")["z3"], "z4": np.load(OUT / "p1_12_a_tables.npz")["z4"]}
    names = list(LAWS)
    res["exact"] = {}
    for n in EXACT_N:
        z = np.load(DATA / f"exact_{n}.npz")
        res["exact"][str(n)] = dict(count=int(z["count"]), **{nm: dict(mass=float(z["mass"][i]), mean=float(z["mean"][i]), sd=float(z["sd"][i]), skew=float(z["skew"][i]),
                                                                       cmin=float(z["cmin"][i]), cmax=float(z["cmax"][i]),
                                                                       mean_vs_table=float(z["mean"][i] - a[nm][n]) if nm in a else None) for i, nm in enumerate(names)})

    # --- Вибірки ---
    res["pool"] = {}
    seqs = s_pool.spawn(len(POOL_N) * len(POOL_LAWS))
    k = 0
    for n in POOL_N:
        res["pool"][str(n)] = {}
        for nm in POOL_LAWS:
            C, _ = pool(nm, n, seqs[k])
            k += 1
            q = np.quantile(C, [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
            sd = C.std()
            res["pool"][str(n)][nm] = dict(mean=float(C.mean()), se=float(sd / np.sqrt(M_POOL)), sd=float(sd), skew=float(((C - C.mean()) ** 3).mean() / sd ** 3),
                                           quantiles=q.tolist(), z_table=float((C.mean() - a[nm][n]) / (sd / np.sqrt(M_POOL))) if nm in a else None)
        print("pool", n, time.time() - t0, flush=True)
    # КС проти точного закону при n = 24
    z = np.load(DATA / "exact_24.npz")
    edges = np.linspace(0, 4, z["hist"].shape[1] + 1)
    res["ks_24"] = {}
    for nm in POOL_LAWS:
        C = np.load(DATA / f"pool_{nm}_24.npz")["C"]
        i = names.index(nm)
        emp = np.searchsorted(np.sort(C), edges[1:], side="right") / len(C)
        ks = float(np.abs(emp - np.cumsum(z["hist"][i])).max())
        res["ks_24"][nm] = dict(ks_on_bin_edges=ks, ks_times_sqrtM=ks * np.sqrt(len(C)), z_mean=float((C.mean() - z["mean"][i]) / (z["sd"][i] / np.sqrt(len(C)))),
                                sd_rel=float(C.std() / z["sd"][i] - 1))

    # --- Нульова модель і контролі ---
    with mp.get_context("spawn").Pool(6) as ex:
        cells = ex.map(run_cell, [(n, int(s.generate_state(1)[0])) for n, s in zip(POOL_N[1:], s_cell.spawn(len(POOL_N) - 1))], chunksize=1)
    res["cells"] = {str(c["n"]): c for c in cells}
    print("cells", time.time() - t0, flush=True)

    # --- Ознаки ---
    jobs, seqs = [], s_feat.spawn(100)
    i = 0
    for n in TASK_N:
        for nm in ("poisson", "geom", "z4", "rrt"):
            jobs.append((nm, n, seqs[i]))
            i += 1
    with mp.get_context("spawn").Pool(8) as ex:
        ex.map(sample_task, jobs, chunksize=1)
    print("features", time.time() - t0, flush=True)
    jobs = [(t, a_, b_, n, int(s.generate_state(1)[0])) for (t, (a_, b_)), ss_ in zip(TASKS.items(), s_eval.spawn(len(TASKS))) for n, s in zip(TASK_N, ss_.spawn(len(TASK_N)))]
    with mp.get_context("spawn").Pool(8) as ex:
        res["auroc"] = ex.map(eval_task, jobs, chunksize=1)
    print("auroc", time.time() - t0, flush=True)

    res["seconds"] = time.time() - t0
    meta = runmeta(__file__, dict(EXACT_N=EXACT_N, POOL_N=POOL_N, M_POOL=M_POOL, K_POOL=K_POOL, M_DATA=M_DATA, B=B, TASK_N=TASK_N, R_CLASS=R_CLASS), dict(entropy=SEED))
    print(save(OUT / "p1_14_shapes.json", meta, **res))
