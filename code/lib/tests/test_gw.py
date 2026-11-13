"""Тести середовища ГВ (П1.10): .venv/bin/python -m pytest code/lib/tests/test_gw.py -q"""
import sys
from collections import Counter
from math import factorial
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy import stats as st

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / "lib"), str(CODE / "lib" / "tests")]

import gwexact as gx  # noqa: E402
import gwtree as gt  # noqa: E402
import ledger as lg  # noqa: E402
import treeenum as te  # noqa: E402
from _gw_workers import toy_worker  # noqa: E402
from gw import nsc_inf  # noqa: E402
from nsccore import _node_sc_all, csr_from_edges, er_graph, node_sc, to_csr  # noqa: E402

A055 = {3: 1, 4: 2, 5: 3, 6: 6, 7: 11, 8: 23, 9: 47, 10: 106, 11: 235, 12: 551}


def nx_of_parent(parent):
    G = nx.Graph()
    G.add_nodes_from(range(len(parent)))
    G.add_edges_from((int(parent[i]), i) for i in range(1, len(parent)))
    return G


def test_reroot_equals_bfs_random_trees():
    for n, seed in [(3, 1), (50, 2), (500, 3), (3000, 4)]:
        p = gt.prufer_parent(n, seed)
        ip, ix = gt.csr_of_parent(p)
        ref = _node_sc_all(ip, ix, n, 2)
        out, acc = gt.node_sc_tree(p, K=8)
        assert np.abs(out - ref).max() < 1e-12
        assert abs(acc.sum() - ref.sum()) < 1e-9


def test_reroot_small_trees_and_guard():
    assert gt.node_sc_tree(np.array([-1]))[0].tolist() == [0.0]
    assert gt.node_sc_tree(np.array([-1, 0]))[0].tolist() == [0.0, 0.0]
    # шлях із завеликим стеком іде через BFS і дає те саме, що переукорінення на малому шляху
    n = 400
    path = np.arange(-1, n - 1)
    a = gt.node_sc_tree(path, K=8)[0]
    old = gt.MAX_STACK_BYTES
    gt.MAX_STACK_BYTES = 0
    try:
        b = gt.node_sc_tree(path, K=8)[0]
    finally:
        gt.MAX_STACK_BYTES = old
    assert np.abs(a - b).max() < 1e-12


def test_enumeration_counts_and_cayley():
    for n, cnt in A055.items():
        assert te.count_trees(n) == cnt
        C, w = te.trees(n)
        assert len(C) == cnt
        assert abs(w.sum() - 1) < 1e-10  # Σ n!/|Aut| = n^{n−2}


def test_aut_against_networkx():
    from networkx.algorithms.isomorphism import GraphMatcher

    for n in (5, 7):
        total = sum(factorial(n) / sum(1 for _ in GraphMatcher(G, G).isomorphisms_iter())
                    for G in nx.nonisomorphic_trees(n))
        assert total == pytest.approx(n ** (n - 2))


def test_a_table_equals_enumeration():
    a = gx.a_table(14)
    for n in range(3, 15):
        C, w = te.trees(n)
        assert abs(float(np.sum(C * w)) - a[n]) < 1e-11
    assert a[1] == 0 and a[2] == 0


def test_enumeration_against_nsc():
    C, _ = te.trees(8)
    ref = sorted(float(node_sc(G).mean()) for G in nx.nonisomorphic_trees(8))
    assert np.allclose(sorted(C), ref, atol=1e-12)


@pytest.mark.parametrize("c", [0.5, 0.8])
def test_table_equals_quadrature(c):
    a = gx.a_table(800)
    v, tail = gx.nsc_from_table(a, c)
    assert tail < 1e-10
    assert abs(v - gx.nsc_quad("ER", c)) < 1e-9


def test_quad_profile_sums_to_nsc_and_k1():
    c = 1.2
    prof = gx.quad_profile("ER", c, K=64)
    assert abs(prof.sum() - gx.nsc_quad("ER", c)) < 1e-12
    z = np.arange(1, 300.0)
    logp = z * np.log(c) - c - np.cumsum(np.log(z))
    assert abs(prof[0] - np.sum(np.exp(logp) * 2 * (z - 1) / (1 + z) ** 2)) < 1e-11


def test_duality_identities():
    for c in (1.2, 1.4, 3.0):
        cd, P = gx.dual(c), gx.giant(c)
        assert cd < 1
        assert abs(cd * np.exp(-cd) - c * np.exp(-c)) < 1e-12
        assert abs(1 - P - cd / c) < 1e-12
    d = gx.decompose(1.4)
    assert abs(d["part_fin"] + d["part_inf"] - d["nsc"]) < 1e-14


def test_nsc_inf_matches_quadrature():
    """nsc_inf (Монте-Карло) і квадратура збігаються; рознесення за сідом фіксоване."""
    for c, seed in ((0.8, 11), (1.4, 12)):
        rng = np.random.default_rng(seed)
        r = nsc_inf(lambda m: rng.poisson(c, m), lambda Z: rng.poisson(c * Z), 400_000)
        assert abs(r["nsc"] - gx.nsc_quad("ER", c)) < 4 * r["se"]


def test_root_equals_uniform_only_when_unimodular():
    """Bin(3,p)/Bin(2,p) унімодулярний: E[NodeSC кореня] = E[середня за вузлами NodeSC дерева].
    Плоский Bin(2,p) не унімодулярний: ці дві величини різні. Числа нотатки 04: 0.36619 проти 0.35581."""
    K, R, cap = 8, 40000, 200000
    g = gt.gtable(cap)
    seeds = gt.seeds_from(np.random.SeedSequence(5), 8)
    _, root, mean, _, over = gt.gw_stats(seeds, R, gt.BINOM, 0.4, 3, gt.BINOM, 0.4, 2, cap, g, K)
    assert over.sum() == 0
    d = (root - mean).ravel()
    assert abs(d.mean()) < 4 * d.std() / np.sqrt(d.size)
    _, root, mean, _, over = gt.gw_stats(seeds, R, gt.BINOM, 0.4, 2, gt.BINOM, 0.4, 2, cap, g, K)
    d = (root - mean).ravel()
    assert over.sum() == 0 and d.mean() > 5 * d.std() / np.sqrt(d.size)


def _chi2_ok(obs, exp, alpha=1e-4):
    keys = list(exp)
    o = np.array([obs.get(k, 0) for k in keys], float)
    e = np.array([exp[k] for k in keys], float) * o.sum()
    return st.chisquare(o, e).pvalue > alpha


@pytest.mark.parametrize("gen", ["prufer", "cond_poisson"])
def test_poisson_generators_chi2(gen):
    n, R = 6, 60000
    C, w = te.trees(n)
    fun = gt.prufer_parent if gen == "prufer" else gt.cond_poisson_parent
    counts = Counter()
    for i in range(R):
        counts[nx.wiener_index(nx_of_parent(fun(n, 1000 + i)))] += 1
    exp = Counter()
    for G, x in zip(nx.nonisomorphic_trees(n), _shape_weights(n)):
        exp[nx.wiener_index(G)] += x
    assert abs(sum(exp.values()) - 1) < 1e-10
    assert _chi2_ok(counts, exp)


def _shape_weights(n):
    from networkx.algorithms.isomorphism import GraphMatcher

    return [factorial(n) / sum(1 for _ in GraphMatcher(G, G).isomorphisms_iter()) / n ** (n - 2)
            for G in nx.nonisomorphic_trees(n)]


def test_geometric_generator_uniform_plane_trees():
    """Геометричні нащадки за умови розміру: рівномірно серед C_{n−1} плоских дерев, шлях Лукасевича — ідентифікатор."""
    n, R = 6, 60000
    counts = Counter()
    for i in range(R):
        p = gt.cond_geom_parent(n, 5000 + i)
        counts[tuple(np.bincount(p[1:], minlength=n).tolist())] += 1
    assert len(counts) == 42  # число Каталана C_5
    assert _chi2_ok(counts, {k: 1 / 42 for k in counts})


def test_ledger_small_forest():
    # ізольований вузол, ребро, шлях 3, трикутник
    u = np.array([1, 3, 4, 6, 7, 6])
    v = np.array([2, 4, 5, 7, 8, 8])
    ip, ix, _ = to_csr(csr_from_edges(9, u, v))
    L = lg.graph_ledger(ip, ix, 9, K=4)
    order = np.argsort(L["n"] * 10 + L["m"])
    assert L["n"][order].tolist() == [1, 2, 3, 3]
    assert L["m"][order].tolist() == [0, 1, 2, 3]
    assert L["sum_sc"][order][0] == 0 and L["sum_sc"][order][1] == 0
    assert L["sum_sc"][order][3] == 0  # трикутник повний: p = 1, δ = 0
    C, w = te.trees(3)
    assert L["sum_sc"][order][2] / 3 == pytest.approx(C[0])


def test_ledger_matches_node_sc_on_er():
    rng = np.random.default_rng(3)
    n = 3000
    ip, ix, _ = to_csr(er_graph(n, 0.7 / n, rng))
    L = lg.graph_ledger(ip, ix, n, K=16)
    assert L["n"].sum() == n
    assert L["sum_sc"].sum() / n == pytest.approx(_node_sc_all(ip, ix, n, 2).mean(), abs=1e-12)


def test_run_chunks_independent_of_nproc_and_resume(tmp_path):
    args = dict(m=5)
    p1 = lg.run_chunks(toy_worker, args, 7, 4, tmp_path / "a", nproc=1)
    p2 = lg.run_chunks(toy_worker, args, 7, 4, tmp_path / "b", nproc=2)
    a, b = lg.load_chunks(p1), lg.load_chunks(p2)
    for x, y in zip(a, b):
        assert (x["x"] == y["x"]).all() and x["i"] == y["i"]
    # перерваний прогін: один шматок зник, відновлення дає той самий підсумок і не чіпає решту
    t0 = p1[0].stat().st_mtime_ns
    p1[2].unlink()
    p3 = lg.run_chunks(toy_worker, args, 7, 4, tmp_path / "a", nproc=1)
    assert p1[0].stat().st_mtime_ns == t0
    c = lg.load_chunks(p3)
    assert (c[2]["x"] == a[2]["x"]).all()


# --- П1.12: окремий закон кореня, RRG, межа a∞ ---
POI = (gt.POISSON, 1.0, 0)
GEO = (gt.GEOM, 0.5, 0)
BIN2 = (gt.BINOM, 0.5, 2)
LAWS_12 = {"poisson": (POI, POI), "z3": gx.rrg_laws(3), "z4": gx.rrg_laws(4), "geom": (GEO, GEO), "bin2": (BIN2, BIN2)}


@pytest.mark.parametrize("name", list(LAWS_12))
def test_law_dp_equals_enumeration(name):
    root, child = LAWS_12[name]
    N = 12
    a, z = gx.a_table_law(N, root, child), gx.size_law(N, root, child)
    rp, cp = gx.law_pmf(root, N), gx.law_pmf(child, N)
    for n in range(3, N + 1):
        Z, ar, at = te.law_moments(n, rp, cp)
        assert abs(Z / z[n] - 1) < 1e-12
        assert abs(ar - a[n]) < 1e-12
        if name in ("poisson", "z3", "z4"):  # унімодулярні: вага форми не залежить від кореня
            assert abs(ar - at) < 1e-12


def test_law_enumeration_at_n16_below_1e12():
    """Критерій П1.12: ДП проти перебору до 10⁻¹² при n = 16, зокрема для закону з нескінченним двійковим записом."""
    root, child = gx.rrg_laws(4)
    a, z = gx.a_table_law(16, root, child), gx.size_law(16, root, child)
    Z, ar, _ = te.law_moments(16, gx.law_pmf(root, 16), gx.law_pmf(child, 16))
    assert abs(ar - a[16]) < 1e-12 and abs(Z / z[16] - 1) < 1e-12


def test_law_generic_matches_poisson_table():
    assert np.abs(gx.a_table(150) - gx.a_table_law(150, POI, POI)).max() < 1e-11


def test_a_table_independent_of_parameter():
    a = gx.a_table_law(80, *gx.rrg_laws(3))
    for p in (0.3, 0.7):
        assert np.abs(a - gx.a_table_law(80, *gx.rrg_laws(3, p))).max() < 1e-11


@pytest.mark.parametrize("z,p", [(3, 0.4), (3, 0.5), (3, 0.7), (4, 0.3), (4, 0.5)])
def test_rrg_cluster_formula_and_duality(z, p):
    N = 200
    f = gx.rrg_cluster(N, z, p)
    d = gx.size_law(N, *gx.rrg_laws(z, p))
    assert np.abs(f[1:] / d[1:] - 1).max() < 1e-10
    model = f"z{z}"
    dual = gx.dual(p, model)
    assert (z - 1) * dual <= 1 + 1e-12
    assert np.abs(f[1:] / ((1 - gx.giant(p, model)) * gx.rrg_cluster(N, z, dual)[1:]) - 1).max() < 1e-10


def test_rrg_decompose_sums_to_nsc():
    d = gx.decompose(0.6, "z3")
    assert abs(gx.dual(0.6, "z3") - 0.4) < 1e-12  # z = 3: p′ = 1 − p
    with pytest.raises(ValueError):
        gx.dual(1.0, "z3")
    assert abs(d["part_fin"] + d["part_inf"] - d["nsc"]) < 1e-14
    assert abs(d["P"] * d["g_inf"] + (1 - d["P"]) * d["g_fin"] - d["nsc"]) < 1e-14


def test_ainf_quad_consistency():
    poi = gx.ainf_quad(POI)
    assert abs(poi - gx.ainf_quad(POI, plain=True)) < 1e-12  # для Пуассона корінь не зміщений
    assert abs(poi - gx.ainf_quad(POI, ulo=-22.0)) < 1e-7  # стійкість до обрізання
    assert abs(poi - gx.ainf_quad(POI, h=0.04)) < 1e-9
    a = gx.a_table(2000)
    n = np.arange(500, 2001, dtype=float)
    X = np.column_stack([np.ones_like(n), np.log(n) / n, 1 / n, n ** -2.0])
    assert abs(np.linalg.lstsq(X, a[500:], rcond=None)[0][0] - poi) < 1e-4


def _rooted_key(parent):
    kids = [[] for _ in parent]
    for v in range(len(parent) - 1, 0, -1):
        kids[parent[v]].append(v)

    def key(v):
        return "(" + "".join(sorted(key(w) for w in kids[v])) + ")"
    return key(0)


@pytest.mark.parametrize("name,mode,b,draw", [("z3", gt.BINOM, 2, (gt.BINOM, 0.5, 3, gt.BINOM, 0.5, 2)),
                                              ("geom", gt.GEOM, 0, (gt.GEOM, 0.5, 0, gt.GEOM, 0.5, 0)),
                                              ("poisson", gt.POISSON, 0, (gt.POISSON, 1.0, 0, gt.POISSON, 1.0, 0))])
def test_cond_law_parent_matches_rejection(name, mode, b, draw):
    """Умовний генератор проти незалежного відбору: χ² за кореневими формами n = 6."""
    n, R = 6, 20000
    cdf = gx.root_given_n(n, *LAWS_12[name])
    seq = np.random.SeedSequence(12)
    s1, s2 = seq.spawn(2)
    A = Counter(_rooted_key(gt.cond_law_parent(n, int(s), cdf, mode, b)) for s in gt.seeds_from(s1, R))
    B = Counter()
    rng = np.random.default_rng(s2)
    mr, ar, br, mc, ac, bc = draw
    while sum(B.values()) < R:
        # відбір: python-генератор, щоб не залежати від numba-коду цього ж модуля
        kids = [rng.poisson(ar) if mr == gt.POISSON else (rng.geometric(1 - ar) - 1 if mr == gt.GEOM else rng.binomial(br, ar))]
        par = [-1]
        i, ok = 0, True
        while i < len(par):
            z = kids[i] if i == 0 else (rng.poisson(ac) if mc == gt.POISSON else (rng.geometric(1 - ac) - 1 if mc == gt.GEOM else rng.binomial(bc, ac)))
            if len(par) + z > n:
                ok = False
                break
            par += [i] * int(z)
            i += 1
        if ok and len(par) == n:
            B[_rooted_key(np.array(par))] += 1
    keys = sorted(set(A) | set(B))
    table = np.array([[A[k] for k in keys], [B[k] for k in keys]])
    assert st.chi2_contingency(table)[1] > 1e-3


def test_cond_law_parent_is_valid_tree():
    cdf = gx.root_given_n(300, *gx.rrg_laws(4))
    for s in range(20):
        par = gt.cond_law_parent(300, s, cdf, gt.BINOM, 3)
        assert (par[1:] < np.arange(1, 300)).all() and par[0] == -1
        d = np.bincount(par[1:], minlength=300)
        assert d[0] <= 4 and d[1:].max() <= 3


def test_shape_stream_matches_dp_and_size_law():
    laws = [gx.rrg_laws(4), (POI, POI), (GEO, GEO), gx.rrg_laws(3)]
    n = 13
    pm = [(gx.law_pmf(r, n), gx.law_pmf(c, n)) for r, c in laws]
    out, count = te.shape_stream(n, pm)
    assert count == 1301  # вільних дерев із 13 вузлами
    for row, (r, c), (rp, cp) in zip(out, laws, pm):
        Z, _, at = te.law_moments(n, rp, cp)
        assert abs(row["mass"] / Z - 1) < 1e-13
        assert abs(row["mean"] - at) < 1e-12
        assert abs(row["hist"].sum() - 1) < 1e-12
    # старий перебір на тих самих вагах
    C, w = te.trees(n)
    assert abs((w * C).sum() - out[1]["mean"]) < 1e-11
    assert abs(np.sqrt((w * C ** 2).sum() - (w * C).sum() ** 2) - out[1]["sd"]) < 1e-10


@pytest.mark.parametrize("model,x", [("ER", 1.2), ("z3", 0.6), ("z4", 0.45)])
def test_pair_quad_matches_dual_decomposition(model, x):
    d = gx.decompose(x, model)
    assert abs(gx.pair_quad(model, x) - d["nsc"]) < 1e-10
    assert abs(gx.pair_quad(model, x, survive=True) - d["part_inf"]) < 1e-10


def test_treefeat_against_networkx():
    import treefeat as tf
    par = gt.prufer_parent(40, 3)
    f = tf.tree_features(par)
    G = nx_of_parent(par)
    assert f["diam"][0] == nx.diameter(G)
    lam = np.linalg.eigvalsh(nx.normalized_laplacian_matrix(G, nodelist=range(40)).toarray())
    assert np.abs(np.exp(-np.outer(tf.NETLSD_T, lam)).sum(1) / 40 - f["netlsd"]).max() < 1e-12
    assert abs(f["deg"].sum() - 1) < 1e-12 and abs(f["C"][0] - gt.node_sc_tree(par)[0].mean()) < 1e-12
