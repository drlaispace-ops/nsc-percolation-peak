"""Тести ядра NSC (колишній code/validate.py), генераторів перколяції і статистики П1.
    .venv/bin/python -m pytest code/lib/tests -q
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / "lib")]

import perc  # noqa: E402
import stats  # noqa: E402
from nsccore import node_sc, node_sc_profile, nsc  # noqa: E402


# ------------------------------------------------ аналітика PRE 2024 (code/validate.py)
def test_complete_graph_zero():
    assert nsc(nx.complete_graph(10)) == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize("q,exp,tol", [(3, 104 / 225, 1e-12), (4, 0.74, 5e-4), (5, 0.7018, 5e-5), (6, 0.865, 5e-4)])
def test_prisms(q, exp, tol):
    # Y5: у статті 0.687 — помилка; правильне значення 0.70181
    assert nsc(nx.circular_ladder_graph(q)) == pytest.approx(exp, abs=tol)


@pytest.mark.parametrize("q", [5, 20])
def test_two_cliques_sharing_node(q):  # (B3)
    G = nx.contracted_nodes(nx.disjoint_union(nx.complete_graph(q), nx.complete_graph(q)), 0, q, self_loops=False)
    assert nsc(G) == pytest.approx(q * (q - 1) / (2 * q - 1) ** 2, rel=1e-12)


@pytest.mark.parametrize("q", [6, 30])
def test_clique_minus_edge(q):  # (B5)
    G = nx.complete_graph(q)
    G.remove_edge(0, 1)
    assert nsc(G) == pytest.approx(2 * (q * (q - 1) - 2) / (q ** 2 * (q - 1) ** 2), rel=1e-12)


def _series(s, b):
    p = b / (s * (s - 1) / 2)
    return (p * (1 - p)).sum()


def test_square_lattice_sums():
    k = np.arange(1, 10 ** 6, dtype=float)
    # (14), об'єм: s_k = 1 + 2k(k+1), b_k = 4k²; у статті 0.931
    assert _series(1 + 2 * k * (k + 1), 4 * k ** 2) == pytest.approx(0.9333, abs=1e-4)
    # (C1), середина краю відкритої решітки: s_k = (k+1)², b_k = k(2k+1); у статті 1.313
    assert _series((k + 1) ** 2, k * (2 * k + 1)) == pytest.approx(1.3165, abs=1e-4)


def test_open_grid_edge_and_torus():
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(201, 201), ordering="sorted")
    v = node_sc(G)
    assert 1.29 < v[100] < 1.3165  # середина краю; скінченний розмір занижує
    rng = np.random.default_rng(0)
    t = [nsc(perc.hypercubic_bond(L, 1.0, rng)) for L in (51, 101, 201)]
    assert t[0] < t[1] < t[2] < 0.9333


def test_tree_identity():
    # для дерев δ_k = 2(s_k − 2)/s_k²: перевірка на шляху (s_k = 1 + 2k від центру)
    P = node_sc_profile(nx.path_graph(41), K=40)[20]
    s = 1 + 2 * np.arange(1, 21)
    assert np.allclose(P[:20], 2 * (s - 2) / s ** 2, atol=1e-12)


# ------------------------------------------------ генератори
def test_er_graph_unbiased():
    rng = np.random.default_rng(1)
    n, c = 20000, 3.0
    A = perc.er_graph(n, c / n, rng)
    deg = np.asarray((A + A.T).astype(bool).sum(1)).ravel()
    assert deg.mean() == pytest.approx(c, rel=0.03)
    # до 03.10.2026 вузли з великими індексами недоотримували ребра
    assert abs(deg[: n // 2].mean() - deg[n // 2:].mean()) < 0.1


@pytest.mark.parametrize("d", [2, 3, 4])
def test_hypercubic_full(d):
    A = perc.hypercubic_bond(5, 1.0, np.random.default_rng(0), d)
    deg = np.asarray((A + A.T).astype(bool).sum(1)).ravel()
    assert (deg == 2 * d).all()
    A, occ = perc.hypercubic_site(5, 1.0, np.random.default_rng(0), d)
    assert occ.all() and (np.asarray((A + A.T).astype(bool).sum(1)).ravel() == 2 * d).all()


def test_triangular_and_rrg():
    A = perc.triangular_bond(8, 1.0, np.random.default_rng(0))
    assert (np.asarray((A + A.T).astype(bool).sum(1)).ravel() == 6).all()
    A = perc.rrg_bond(100, 3, 1.0, np.random.default_rng(0))
    assert (np.asarray((A + A.T).astype(bool).sum(1)).ravel() == 3).all()


def test_product_rule():
    U, V = perc.product_rule(1000, 1500, 7)
    assert len(U) == 1500 and (U != V).all()
    U2, _ = perc.product_rule(1000, 1500, 7)
    assert (U == U2).all()  # відтворюваність за сідом


def test_measure_site_normalisation():
    rng = np.random.default_rng(3)
    A, occ = perc.hypercubic_site(64, 0.6, rng, 2)
    m = perc.measure(A, occ)
    assert m["nsc"] == pytest.approx(m["n_occ"] / m["n"] * m["nsc_occ"], rel=1e-12)
    assert m["nsc_occ"] == pytest.approx((1 - m["P_occ"]) * m["g_fin"] + m["P_occ"] * m["g_giant"], rel=1e-12)
    assert m["P"] == pytest.approx(m["P_occ"] * m["n_occ"] / m["n"])


def test_measure_bond_decomposition():
    m = perc.measure(perc.hypercubic_bond(64, 0.5, np.random.default_rng(4), 2))
    assert m["nsc"] == pytest.approx((1 - m["P"]) * m["g_fin"] + m["P"] * m["g_giant"], rel=1e-12)
    assert m["nsc"] == m["nsc_occ"]


# ------------------------------------------------ статистика
def test_cubic_peak_exact():
    xs = np.linspace(0, 1, 41)
    y = -(xs - 0.4137) ** 2 + 0.3 * (xs - 0.4137) ** 3
    x, _ = stats.cubic_peak(xs, y, window=6)
    assert x == pytest.approx(0.4137, abs=1e-10)


def test_cubic_peak_halfwidth_nonuniform():
    """Вікно в одиницях x на нерівномірній сітці: та сама точність; вузьке вікно зменшує зсув від асиметрії."""
    xs = np.unique(np.r_[np.linspace(0, 1, 13), np.linspace(0.38, 0.45, 15)])
    y = -(xs - 0.4137) ** 2 + 0.3 * (xs - 0.4137) ** 3
    x, _ = stats.cubic_peak(xs, y, halfwidth=0.03)
    assert x == pytest.approx(0.4137, abs=1e-10)
    y2 = y - 2 * (xs - 0.4137) ** 4                     # не кубіка: зсув росте з шириною вікна
    e = [abs(stats.cubic_peak(xs, y2, halfwidth=h)[0] - 0.4137) for h in (0.03, 0.3)]
    assert e[0] < e[1]


def test_peak_bootstrap_coverage():
    rng = np.random.default_rng(5)
    xs = np.linspace(0, 1, 41)
    Y = -(xs - 0.52) ** 2 + 0.05 * rng.standard_normal((200, len(xs)))
    r = stats.peak_bootstrap(xs, Y, window=8, B=400, rng=6)
    assert abs(r["x"] - 0.52) < 4 * r["se_x"] and r["se_x"] < 0.02 and r["fail"] < 0.01


def test_tau_int_ar1():
    rng = np.random.default_rng(7)
    rho, N = 0.8, 200000
    e = rng.standard_normal(N)
    x = np.empty(N)
    x[0] = e[0]
    for t in range(1, N):
        x[t] = rho * x[t - 1] + e[t]
    tau, _ = stats.tau_int(x)
    assert tau == pytest.approx((1 + rho) / (2 * (1 - rho)), rel=0.1)  # 4.5
    se, _ = stats.binning(x)
    assert se == pytest.approx(x.std() * np.sqrt(2 * tau / N), rel=0.2)


def test_bootstrap_mean():
    x = np.random.default_rng(8).standard_normal(400)
    est, se, ci = stats.bootstrap(x, B=2000, rng=9)
    assert se == pytest.approx(1 / 20, rel=0.15) and ci[0] < est < ci[1]
