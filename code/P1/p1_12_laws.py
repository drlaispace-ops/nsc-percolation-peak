"""П1.12. Розклади NSC для RRG z = 3, 4 і закони нащадків: a_n, межа a∞, корінь проти типового вузла.

Двоїстість RRG: Bin(m, p) за умови виродження — Bin(m, p′), тому скінченні кластери при p — це RRG при p′.
Точне: перебір до n = 16, ДП (S, Z) для a_n, формула Двасса, формула гратки Бете, квадратура хребта для a∞.
Вибіркове: корінь і типовий вузол умовних дерев; журнал справжніх RRG n = 2·10⁵.
Запуск: .venv/bin/python code/P1/p1_12_laws.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "lib"), str(ROOT / "code" / "P1")]
import gwexact as gx  # noqa: E402
import gwtree as gt  # noqa: E402
import ledger as lg  # noqa: E402
import treeenum as te  # noqa: E402
from nsccore import to_csr  # noqa: E402
from p1_11_er import coordinates, first_at, fit_ainf  # noqa: E402
from perc import rrg_bond  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

SEED = 202611062
N_TABLE = 3000
POI, GEO, BIN2 = (gt.POISSON, 1.0, 0), (gt.GEOM, 0.5, 0), (gt.BINOM, 0.5, 2)
LAWS = {"poisson": (POI, POI), "z3": gx.rrg_laws(3), "z4": gx.rrg_laws(4), "geom": (GEO, GEO), "bin2": (BIN2, BIN2)}
UNIMODULAR = ("poisson", "z3", "z4")
PSTAR = {3: 0.56782, 4: 0.40182}  # results/P1/p1_9_cstar.json
LEDGER_P = {3: [0.4, 0.5, PSTAR[3], 0.6, 0.8], 4: [0.27, 1 / 3, PSTAR[4], 0.45, 0.6]}
LEDGER = dict(n=200_000, R=60, K=32, nmax=2000)
OUT = ROOT / "results" / "P1"
CHUNKS = ROOT / "data" / "P1" / "p1_12_ledger"


def rrg_worker(i, seq, args):
    """Один RRG, згорнутий за розміром компоненти; вище порога найбільша компонента прибирається до BFS."""
    ip, ix, _ = to_csr(rrg_bond(args["n"], args["z"], args["p"], np.random.default_rng(seq)))
    L = lg.graph_ledger(ip, ix, args["n"], K=args["K"], exclude_largest=(args["z"] - 1) * args["p"] > 1)
    nn = np.minimum(L["n"], args["nmax"])
    size = args["nmax"] + 1
    tree = L["m"] == L["n"] - 1
    return dict(nodes=np.bincount(nn, weights=L["n"], minlength=size), sc=np.bincount(nn, weights=L["sum_sc"], minlength=size),
                nodes_cyc=L["n"][~tree].sum(), sc_cyc=L["sum_sc"][~tree].sum(), giant=L["giant_size"], N=L["N"])


def root_vs_typical(name, cdfs, seq, R):
    """Вибірка умовних дерев: NodeSC кореня, середнє за вузлами, їхня парна різниця."""
    mode, _, b = LAWS[name][1]
    out = {}
    for n, s in zip(cdfs, seq.spawn(len(cdfs))):
        seeds = gt.seeds_from(s, R)
        rt, ty = np.zeros(R), np.zeros(R)
        for r in range(R):
            sc = gt.node_sc_tree(gt.cond_law_parent(n, int(seeds[r]), cdfs[n], mode, int(b)), K=8)[0]
            rt[r], ty[r] = sc[0], sc.mean()
        d = rt - ty
        out[str(n)] = dict(root=float(rt.mean()), root_se=float(rt.std() / np.sqrt(R)), typ=float(ty.mean()), typ_se=float(ty.std() / np.sqrt(R)),
                           diff=float(d.mean()), diff_se=float(d.std() / np.sqrt(R)), z_diff=float(d.mean() / (d.std() / np.sqrt(R))))
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {}
    OUT.mkdir(exist_ok=True)

    # --- Перебір проти ДП, формули розміру ---
    res["enum"] = {}
    for name, (root, child) in {**LAWS, "z3_p0.4": gx.rrg_laws(3, 0.4)}.items():
        a, z = gx.a_table_law(16, root, child), gx.size_law(16, root, child)
        rp, cp = gx.law_pmf(root, 16), gx.law_pmf(child, 16)
        rows = {n: te.law_moments(n, rp, cp) for n in range(3, 17)}
        res["enum"][name] = dict(max_dp_err=max(abs(rows[n][1] - a[n]) for n in rows), max_size_rel_err=max(abs(rows[n][0] / z[n] - 1) for n in rows),
                                 a_root={str(n): rows[n][1] for n in rows}, a_typ={str(n): rows[n][2] for n in rows},
                                 max_root_typ=max(abs(rows[n][1] - rows[n][2]) for n in rows))
    res["bethe_formula"] = {}
    for z, p in [(3, 0.4), (3, 0.5), (3, 0.7), (4, 0.27), (4, 1 / 3), (4, 0.6)]:
        f, d = gx.rrg_cluster(2000, z, p), gx.size_law(2000, *gx.rrg_laws(z, p))
        ok = d[1:] > 1e-280
        res["bethe_formula"][f"z{z}_p{p:.4f}"] = float(np.abs(f[1:][ok] / d[1:][ok] - 1).max())
    print("enum", time.time() - t0, flush=True)

    # --- Таблиці a_n і межа a∞ ---
    tables = {name: gx.a_table_law(N_TABLE, *law) for name, law in LAWS.items()}
    np.savez(OUT / "p1_12_a_tables.npz", **tables)
    child = {k: v[1] for k, v in LAWS.items()}
    res["ainf"] = {}
    for name, a in tables.items():
        typ, plain = gx.ainf_quad(child[name]), gx.ainf_quad(child[name], plain=True)
        typ_c = gx.ainf_quad(child[name], ulo=-22.0)
        fits = fit_ainf(a)
        vals = np.array([v for k, v in fits.items() if "n≥3000" not in k])  # при N = 3000 це вікно має одну точку
        rootlim = typ if name in UNIMODULAR else plain
        res["ainf"][name] = dict(typ=typ, plain=plain, cutoff_shift=abs(typ - typ_c), fit_root=fits["+1/n^2, n≥1000"], fit_spread=float(vals.max() - vals.min()),
                                 fit_minus_quad=fits["+1/n^2, n≥1000"] - rootlim, a_values={str(n): float(a[n]) for n in (3, 5, 10, 100, 1000, 3000)})
        print(name, res["ainf"][name]["typ"], plain, res["ainf"][name]["fit_minus_quad"], time.time() - t0, flush=True)

    # --- Корінь проти типового вузла ---
    sizes = [10, 30, 100, 300, 1000, 3000]
    seqs = np.random.SeedSequence(SEED + 1).spawn(len(LAWS))
    res["root_typ"] = {}
    for (name, law), seq in zip(LAWS.items(), seqs):
        cdfs = {n: gx.root_given_n(n, *law) for n in sizes}
        res["root_typ"][name] = root_vs_typical(name, cdfs, seq, 2000)
        for n in sizes:
            row = res["root_typ"][name][str(n)]
            row["table"] = float(tables[name][n])
            row["z_root_table"] = (row["root"] - row["table"]) / row["root_se"]
        print(name, "root_typ", time.time() - t0, flush=True)

    # --- Повторення з новим сідом: точки, що у першому заході вийшли за 2σ (вибрані після перегляду даних) ---
    res["replicate"] = {}
    for (name, n), seq in zip([("poisson", 100), ("z3", 30), ("z3", 100)], np.random.SeedSequence(SEED + 2).spawn(3)):
        row = root_vs_typical(name, {n: gx.root_given_n(n, *LAWS[name])}, seq, 20000)[str(n)]
        row["table"] = float(tables[name][n])
        row["z_root_table"] = (row["root"] - row["table"]) / row["root_se"]
        res["replicate"][f"{name}_{n}"] = row
    print("replicate", time.time() - t0, flush=True)

    # --- Розрив g_∞ − a∞ біля порога проти частки живих гілок q ---
    res["gap"] = {}
    for model, name in (("ER", "poisson"), ("z3", "z3"), ("z4", "z4")):
        rows = []
        for e in (0.04, 0.02, 0.01, 0.005, 0.0025):
            x = 1 + e if model == "ER" else (1 + e) / (int(model[1:]) - 1)
            dec = gx.decompose(x, model)
            q = dec["P"] if model == "ER" else gx._q_rrg(int(model[1:]), x)
            gap = dec["g_inf"] - res["ainf"][name]["typ"]
            rows.append(dict(eps=e, g_inf=dec["g_inf"], gap=gap, q=q, gap_over_q=gap / q, gap_over_eps=gap / e))
        res["gap"][model] = rows
    print("gap", time.time() - t0, flush=True)

    # --- Чотири розклади RRG ---
    res["rrg"] = {}
    for z in (3, 4):
        model = f"z{z}"
        pc = 1 / (z - 1)
        pmax = 0.95
        eps = np.r_[np.arange(-0.4, -0.019, 0.04), np.arange(0.02, 0.301, 0.02), np.arange(0.35, (z - 1) * pmax - 1 + 1e-9, 0.05)]
        ps = (1 + eps) / (z - 1)
        a = tables[model]
        ainf = res["ainf"][model]["typ"]
        block = dict(pc=pc, pstar=PSTAR[z], parts=[gx.decompose(float(p), model) for p in ps])
        block["crit"] = gx.decompose(pc, model)
        block["decomp_peak"] = gx.decompose(PSTAR[z], model)
        block["g_inf_near_threshold"] = [gx.decompose((1 + e) / (z - 1), model) for e in (0.04, 0.02, 0.01, 0.005, 0.0025)]
        print(model, "parts", time.time() - t0, flush=True)
        block["size_contrib"] = {}
        for p in LEDGER_P[z]:
            rho = gx.rrg_cluster(N_TABLE, z, p)
            part = rho * a
            tail = max(0.0, 1 - gx.giant(p, model) - float(rho.sum()))
            tot = float(part.sum()) + tail * ainf
            cum = np.cumsum(part) / tot
            block["size_contrib"][f"{p:.5f}"] = dict(finite_part=tot, tail_mass=tail, node_mass_n_le_2=float(rho[:3].sum()), n50=first_at(cum, 0.5),
                                                     n90=first_at(cum, 0.9), n99=first_at(cum, 0.99), share_tail=tail * ainf / tot)
        K = 64
        grid = (1 + np.r_[np.arange(-0.4, -0.019, 0.05), np.arange(0.05, 1.01, 0.05)]) / (z - 1)
        grid = grid[grid <= pmax + 1e-9]
        tot, pf = np.zeros((len(grid), K)), np.zeros((len(grid), K))
        for i, p in enumerate(grid):
            tot[i] = gx.quad_profile(model, float(p), K=K)
            P = gx.giant(float(p), model)
            pf[i] = (1 - P) * gx.quad_profile(model, gx.dual(float(p), model), K=K) if P > 0 else tot[i]
        block["scales"] = dict(p=grid.tolist(), K=K, part_fin=pf.tolist(), part_inf=(tot - pf).tolist())
        print(model, "scales", time.time() - t0, flush=True)
        law = gx.rrg_laws(z)
        cdfs = {n: gx.root_given_n(n, *law) for n in (20, 100, 500)}
        block["coords"] = coordinates([20, 100, 500], 2000, 128, SEED + 10 + z,
                                      sampler=lambda n, s, c=cdfs, b=z - 1: gt.cond_law_parent(n, s, c[n], gt.BINOM, b))
        print(model, "coords", time.time() - t0, flush=True)

        block["ledger"] = {}
        for j, p in enumerate(LEDGER_P[z]):
            t1 = time.time()
            args = dict(z=z, p=float(p), **{k: LEDGER[k] for k in ("n", "K", "nmax")})
            paths = lg.run_chunks(rrg_worker, args, SEED + 100 * z + j, LEDGER["R"], CHUNKS / f"z{z}_p{p:.5f}", nproc=8)
            ch = lg.load_chunks(paths)
            sc = np.array([x["sc"].sum() for x in ch])
            nodes = np.array([x["nodes"].sum() for x in ch], float)
            exact = gx.decompose(float(p), model)
            gf = sc / nodes
            pfin = sc / LEDGER["n"]
            R = len(ch)
            pool_n = sum(x["nodes"][3:11] for x in ch)
            pool_a = sum(x["sc"][3:11] for x in ch) / np.maximum(pool_n, 1)  # спільно за реалізаціями: окремі комірки бувають порожні
            block["ledger"][f"{p:.5f}"] = dict(g_fin=float(gf.mean()), g_fin_se=float(gf.std() / np.sqrt(R)), exact_g_fin=exact["g_fin"],
                                               z_g=float((gf.mean() - exact["g_fin"]) / (gf.std() / np.sqrt(R))), part_fin=float(pfin.mean()),
                                               part_fin_se=float(pfin.std() / np.sqrt(R)), exact_part_fin=exact["part_fin"],
                                               z_part=float((pfin.mean() - exact["part_fin"]) / (pfin.std() / np.sqrt(R))),
                                               dev_a_small_n=[float(pool_a[i] - a[3 + i]) if pool_n[i] >= 1000 else None for i in range(8)], cyc_node_fraction=float(sum(x["nodes_cyc"] for x in ch) / nodes.sum()),
                                               cyc_sc_fraction=float(sum(x["sc_cyc"] for x in ch) / sc.sum()),
                                               giant_mean=float(np.mean([x["giant"] for x in ch])) / LEDGER["n"], exact_P=exact["P"], R=R, seconds=time.time() - t1)
            print(model, f"{p:.5f}", {k: block["ledger"][f"{p:.5f}"][k] for k in ("g_fin", "g_fin_se", "exact_g_fin", "z_part", "cyc_sc_fraction", "seconds")}, flush=True)
        res["rrg"][model] = block

    res["seconds"] = time.time() - t0
    meta = runmeta(__file__, dict(N_TABLE=N_TABLE, LEDGER=LEDGER, LEDGER_P={str(k): v for k, v in LEDGER_P.items()}, PSTAR={str(k): v for k, v in PSTAR.items()}), dict(entropy=SEED))
    print(save(OUT / "p1_12_laws.json", meta, **res))
