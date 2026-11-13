"""П1.11. Чотири розклади NSC для ER, усюди за вузлами.

(а) зв'язні компоненти: B(n; c)·a_n і накопичений внесок; (б) (1−P)·g_fin і P·g_∞;
(в) масштаби: E[δ_k] окремо для скінченних кластерів і гіганта;
(г) координати ознак: у режимі delta Var(C) = Σ_k Cov(δ̄_k, C) точно; для p і sb це частка дисперсії R².
Журнал ER — перевірка точних формул. Запуск: .venv/bin/python code/P1/p1_11_er.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import gwexact as gx  # noqa: E402
import gwtree as gt  # noqa: E402
import ledger as lg  # noqa: E402
from nscfeat import ball_density, bfs_profile  # noqa: E402
from runmeta import runmeta, save  # noqa: E402
from stats import bootstrap  # noqa: E402

SEED = 202611061
N_TABLE = 6000
CSTAR = 1.40381
CS_LEDGER = [0.8, 1.0, 1.2, CSTAR, 2.0]
LEDGER = dict(n=200_000, R=60, K=32, nmax=2000)
OUT = ROOT / "results" / "P1"
CHUNKS = ROOT / "data" / "P1" / "p1_11_ledger"


def er_worker(i, seq, args):
    """Один граф ER, згорнутий за розміром компоненти (рядки на компоненту зайняли б гігабайти)."""
    L = lg.er_ledger(args["n"], args["c"], np.random.default_rng(seq), K=args["K"])
    nn = np.minimum(L["n"], args["nmax"])
    size = args["nmax"] + 1
    tree = L["m"] == L["n"] - 1
    return dict(comps=np.bincount(nn, minlength=size), nodes=np.bincount(nn, weights=L["n"], minlength=size),
                sc=np.bincount(nn, weights=L["sum_sc"], minlength=size), delta=L["delta"].sum(0),
                nodes_cyc=L["n"][~tree].sum(), sc_cyc=L["sum_sc"][~tree].sum(),
                c_mean_comp=(L["sum_sc"] / L["n"]).mean(), ncomp=np.int64(len(L["n"])), giant=L["giant_size"], N=L["N"])


def first_at(cum, q):
    """Найменше n з часткою ≥ q; None, якщо поріг лежить за межами таблиці."""
    return int(np.argmax(cum >= q)) if cum[-1] >= q else None


def fit_ainf(a):
    """a∞ у кількох формах поправки і на кількох вікнах; розкид показує систематику."""
    out = {}
    forms = {"ln/n+1/n": lambda n: [np.log(n) / n, 1 / n],
             "+1/n^1.5": lambda n: [np.log(n) / n, 1 / n, n ** -1.5],
             "+1/n^2": lambda n: [np.log(n) / n, 1 / n, n ** -2.0],
             "+ln/n^2": lambda n: [np.log(n) / n, 1 / n, np.log(n) / n ** 2, n ** -2.0]}
    for name, f in forms.items():
        for lo in (500, 1000, 2000, 3000):
            n = np.arange(lo, len(a), dtype=float)
            X = np.column_stack([np.ones_like(n), *f(n)])
            co = np.linalg.lstsq(X, a[lo:], rcond=None)[0]
            out[f"{name}, n≥{lo}"] = float(co[0])
    return out


BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 10 ** 9)]


def coordinates(sizes, R, K, seed, sampler=gt.prufer_parent):
    """(г): за фіксованого n частка Var(C) за масштабами і R² окремих координат; за замовчуванням дерева рівномірні (= Poisson ГВ | n)."""
    res = {}
    for n, seq in zip(sizes, np.random.SeedSequence(seed).spawn(len(sizes))):
        s_tree, s_boot, s_split = seq.spawn(3)
        seeds = gt.seeds_from(s_tree, R)
        d, p, ls, c, diam = np.zeros((R, K)), np.zeros((R, K)), np.zeros((R, K)), np.zeros(R), np.zeros(R)
        for r in range(R):
            ip, ix = gt.csr_of_parent(sampler(int(n), int(seeds[r])))
            S, B, D = bfs_profile(ip, ix, int(n), K=K, full=True, nthreads=1)
            d[r], p[r], ls[r] = D.mean(0), ball_density(S, B).mean(0), np.log(S).mean(0)
            c[r] = D.sum(1).mean()
            diam[r] = (np.diff(S, axis=1) > 0).sum(1).max() + 1
        cc = c - c.mean()
        var = c.var()
        cov = (d - d.mean(0)).T @ cc / R / var  # Σ_k дає точно 1
        rng = np.random.default_rng(s_boot)
        idx = [[k for k in range(K) if lo <= k + 1 <= hi] for lo, hi in BINS]
        share = [float(cov[i].sum()) for i in idx]
        boot = []
        for _ in range(500):
            b = rng.integers(0, R, R)
            cb = c[b] - c[b].mean()
            co = (d[b] - d[b].mean(0)).T @ cb / R / c[b].var()
            boot.append([co[i].sum() for i in idx])

        def r2(X):
            out = np.zeros(X.shape[1])
            for k in range(X.shape[1]):
                if X[:, k].std() > 0:
                    out[k] = np.corrcoef(X[:, k], c)[0, 1] ** 2
            return out

        split = np.random.default_rng(s_split).permutation(R)
        tr, te = split[: R // 2], split[R // 2:]

        def joint(X, m=16):
            """R² на відкладеній половині: лінійна модель на перших m координатах."""
            A = np.column_stack([np.ones(R), X[:, :m]])
            co = np.linalg.lstsq(A[tr], c[tr], rcond=None)[0]
            return float(1 - np.sum((c[te] - A[te] @ co) ** 2) / np.sum((c[te] - c[te].mean()) ** 2))

        rd, rp, rs = r2(d[:, :K - 1]), r2(p[:, :K - 1]), r2(ls[:, :K - 1])
        res[str(n)] = dict(mean_C=float(c.mean()), sd_C=float(c.std()), bins=[list(x) for x in BINS], delta_share=share,
                           delta_share_se=np.std(boot, axis=0).tolist(), sum_share=float(cov.sum()),
                           best_k=dict(delta=int(rd.argmax()) + 1, p=int(rp.argmax()) + 1, logs=int(rs.argmax()) + 1),
                           best_r2=dict(delta=float(rd.max()), p=float(rp.max()), logs=float(rs.max())),
                           r2_diam=float(np.corrcoef(diam, c)[0, 1] ** 2),
                           joint_r2_16=dict(delta=joint(d), p=joint(p), logs=joint(ls)),
                           diam_mean=float(diam.mean()), diam_max=float(diam.max()), K=K, R=R)
    return res


if __name__ == "__main__":
    t0 = time.time()
    res = {}

    # --- Таблиця a_n і (а) ---
    a = gx.a_table(N_TABLE)
    np.save(OUT / "p1_11_a_table.npy", a)
    fits = fit_ainf(a)
    ainf = fits["+1/n^2, n≥1000"]
    vals = np.array(list(fits.values()))
    res["a_table"] = dict(N=N_TABLE, a_values={str(n): float(a[n]) for n in (3, 5, 10, 100, 1000, 6000)}, ainf=ainf,
                          fits=fits, fit_spread=float(vals.max() - vals.min()),
                          table_vs_quad={str(c): dict(table=gx.nsc_from_table(a, c)[0], tail=gx.nsc_from_table(a, c)[1],
                                                      quad=gx.nsc_quad("ER", c)) for c in (0.3, 0.5, 0.7, 0.8, 0.9)})
    print("a_table", res["a_table"]["fit_spread"], ainf, time.time() - t0, flush=True)
    res["size_contrib"] = {}
    for c in (0.8, 1.0, 1.2, CSTAR, 2.0):
        b = gx.borel(N_TABLE, c)
        part = b * a
        tot = float(part.sum())
        tail = max(0.0, 1 - gx.giant(c) - float(b.sum()))
        cum = np.cumsum(part) / (tot + tail * ainf)  # знаменник з хвостом n > N: при c = 1 це 3.4%
        share = {str(m): float(cum[m]) for m in (2, 5, 10, 20, 50, 100, 500, 1000, 6000)}
        res["size_contrib"][str(c)] = dict(finite_part=tot + tail * ainf, tail_mass=tail, node_mass_n_le_2=float(b[:3].sum()),
                                           cum_share=share, n50=first_at(cum, 0.5), n90=first_at(cum, 0.9), n99=first_at(cum, 0.99))

    # --- (б) ---
    grid = np.r_[np.arange(0.5, 0.96, 0.05), np.arange(1.02, 1.31, 0.02), np.arange(1.35, 4.01, 0.05)]
    res["parts"] = [gx.decompose(float(c)) for c in grid]
    res["crit"] = dict(g_fin_1=gx.nsc_from_table(a, 1.0, ainf)[0], tail_1=gx.nsc_from_table(a, 1.0, ainf)[1])
    res["g_inf_near_threshold"] = [gx.decompose(1 + e) for e in (0.04, 0.02, 0.01, 0.005, 0.0025)]
    res["decomp_peak"] = gx.decompose(CSTAR)
    print("parts", time.time() - t0, flush=True)

    # --- (в) ---
    K = 64
    cs = np.r_[np.arange(0.6, 0.96, 0.05), np.arange(1.05, 3.01, 0.05)]
    pf, pg = np.zeros((len(cs), K)), np.full((len(cs), K), np.nan)
    tot = np.zeros((len(cs), K))
    for i, c in enumerate(cs):
        tot[i] = gx.quad_profile("ER", float(c), K=K)
        if c > 1:
            P = gx.giant(float(c))
            pf[i] = (1 - P) * gx.quad_profile("ER", gx.dual(float(c)), K=K)
            pg[i] = tot[i] - pf[i]  # внесок гіганта до NSC за вузлами
        else:
            pf[i] = tot[i]
    res["scales"] = dict(c=cs.tolist(), K=K, part_fin=pf.tolist(), part_inf=np.nan_to_num(pg).tolist(),
                         max_sum_err=float(np.abs(tot.sum(1) - np.array([gx.nsc_quad("ER", float(c)) for c in cs])).max()))
    print("scales", time.time() - t0, flush=True)

    # --- (г) ---
    res["coords"] = coordinates([20, 100, 500], 2000, 128, SEED + 1)
    print("coords", time.time() - t0, flush=True)

    # --- Журнал ER ---
    res["ledger"] = {}
    for j, c in enumerate(CS_LEDGER):
        t1 = time.time()
        args = dict(c=c, **{k: LEDGER[k] for k in ("n", "K", "nmax")})
        paths = lg.run_chunks(er_worker, args, SEED + 100 + j, LEDGER["R"], CHUNKS / f"c{c}", nproc=8)
        ch = lg.load_chunks(paths)
        nodes = np.array([x["nodes"].sum() for x in ch], float)
        sc = np.array([x["sc"].sum() for x in ch])
        n = LEDGER["n"]
        Nf = nodes.sum()
        est, se, ci = bootstrap(sc / nodes, rng=SEED)
        b = dict(est=float(est), se=float(se), ci=[float(ci[0]), float(ci[1])])
        pf_all = sc / n
        exact_g = res["crit"]["g_fin_1"] if c == 1.0 else gx.decompose(c)["g_fin"]
        exact_part = exact_g * (1 - gx.giant(c))
        # a_n за малими розмірами
        by_real = np.array([x["sc"][3:11] / np.maximum(x["nodes"][3:11], 1) for x in ch])
        z = (by_real.mean(0) - a[3:11]) / (by_real.std(0) / np.sqrt(len(ch)))
        res["ledger"][str(c)] = dict(
            g_fin=float((sc / nodes).mean()), g_fin_se=float((sc / nodes).std() / np.sqrt(len(ch))), bootstrap=b,
            part_fin=float(pf_all.mean()), part_fin_se=float(pf_all.std() / np.sqrt(len(ch))),
            exact_g_fin=exact_g, exact_part_fin=exact_part,
            z_part=float((pf_all.mean() - exact_part) / (pf_all.std() / np.sqrt(len(ch)))),
            z_a_small_n=z.tolist(), dev_a_small_n=(by_real.mean(0) - a[3:11]).tolist(), cyc_node_fraction=float(sum(x["nodes_cyc"] for x in ch) / Nf),
            cyc_sc_fraction=float(sum(x["sc_cyc"] for x in ch) / sc.sum()),
            comp_mean=float(np.mean([x["c_mean_comp"] for x in ch])), giant_mean=float(np.mean([x["giant"] for x in ch])),
            R=len(ch), seconds=time.time() - t1)
        print(c, {k: res["ledger"][str(c)][k] for k in ("g_fin", "g_fin_se", "exact_g_fin", "z_part", "cyc_sc_fraction", "seconds")}, flush=True)

    res["seconds"] = time.time() - t0
    meta = runmeta(__file__, dict(N_TABLE=N_TABLE, LEDGER=LEDGER, CS_LEDGER=CS_LEDGER, CSTAR=CSTAR), dict(entropy=SEED))
    print(save(OUT / "p1_11_er.json", meta, **res))
