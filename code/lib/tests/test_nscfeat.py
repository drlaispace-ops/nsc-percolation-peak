"""Тести пакета nscfeat:  .venv/bin/python -m pytest code/lib/tests -q"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / "lib"), str(CODE)]

import nsc as ref  # еталон: code/nsc.py (без побічних ефектів при імпорті)
from nscfeat import NSCFeaturizer, ball_density, bfs_profile, csr_from_graph, encode, nsc_encoding
from nscfeat.testgraphs import PAIRS, rook4, shrikhande


def prof(G, K=8, full=True, nthreads=None):
    indptr, indices, n = csr_from_graph(G)
    return bfs_profile(indptr, indices, n, K, full, nthreads)


def multiset(X, nd=9):
    return sorted(map(tuple, np.round(X, nd)))


# ---------------------------------------------------------------- аналітика
def test_path3():
    S, B, D = prof(nx.path_graph(3), K=3)
    # центр: k=1 — уся куля (3 вузли, 2 ребра), p = 2/3
    assert np.allclose(D[1], [2 / 9, 0, 0])
    # кінець: k=1 — ребро (p=1, δ=0), k=2 — увесь граф
    assert np.allclose(D[0], [0, 2 / 9, 0])
    assert S[0].tolist() == [2, 3, 3] and B[0].tolist() == [1, 2, 2]


def test_complete_graph_zero():
    S, B, D = prof(nx.complete_graph(7), K=4)
    assert np.all(D == 0) and np.allclose(ball_density(S, B), 1)


def test_isolated_nodes():
    G = nx.empty_graph(4)
    G.add_edge(0, 1)
    S, B, D = prof(G, K=3)
    assert np.all(S[2:] == 1) and np.all(B[2:] == 0) and np.all(D[2:] == 0)
    assert np.all(ball_density(S, B)[2:] == 0)


def test_tree_identity():
    """Для дерев δ_k = 2(s_k − 2)/s_k² (поки куля росте)."""
    G = nx.random_labeled_tree(200, seed=1)
    S, B, D = prof(G, K=6, full=False)
    assert np.all(B == S - 1)
    grow = np.diff(np.hstack([np.ones((len(S), 1), int), S]), axis=1) > 0
    assert np.allclose(D[grow], (2 * (S - 2) / S**2)[grow])


# ---------------------------------------------------------------- еталон
@pytest.mark.parametrize("make", [
    lambda: nx.gnp_random_graph(300, 3 / 300, seed=2),          # кілька компонент
    lambda: nx.barabasi_albert_graph(250, 2, seed=3),
    lambda: nx.grid_2d_graph(12, 12, periodic=True),
    lambda: nx.watts_strogatz_graph(200, 6, 0.1, seed=4),
])
def test_rowsum_equals_node_sc(make):
    G = make()
    _, _, D = prof(G, K=5, full=True)
    assert np.allclose(D.sum(1), ref.node_sc(G), atol=1e-12)
    assert np.allclose(D, ref.node_sc_profile(G, K=5), atol=1e-12)


def test_parallel_equals_serial():
    G = nx.gnp_random_graph(500, 4 / 500, seed=5)
    a = prof(G, K=6)
    b = prof(G, K=6, nthreads=4)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_truncated_matches_full():
    G = nx.gnp_random_graph(400, 3 / 400, seed=6)
    Sf, Bf, Df = prof(G, K=4, full=True)
    St, Bt, Dt = prof(G, K=4, full=False)
    assert np.array_equal(Sf, St) and np.array_equal(Bf, Bt)
    assert np.allclose(Df[:, :3], Dt[:, :3])
    assert np.all(Df[:, 3] >= Dt[:, 3] - 1e-15)


def test_permutation_equivariance():
    G = nx.gnp_random_graph(120, 5 / 120, seed=7)
    perm = np.random.default_rng(0).permutation(120)
    H = nx.Graph()
    H.add_nodes_from(range(120))  # рядки йдуть у порядку G.nodes()
    H.add_edges_from((int(perm[u]), int(perm[v])) for u, v in G.edges())
    for mode in ("delta", "p", "sb"):
        a = encode(*prof(G), mode)
        b = encode(*prof(H), mode)
        assert np.allclose(a, b[perm])


def test_edge_index_input_dedup():
    ei = np.array([[0, 1, 1, 2, 2, 2], [1, 0, 2, 1, 2, 0]])  # дублікати, петля 2–2
    X = nsc_encoding(ei, 3, K=2, mode="sb")
    assert X.shape == (3, 4)
    assert np.allclose(nsc_encoding(ei, 3, K=2), 0)  # трикутник: p = 1


# ---------------------------------------------------------------- обгортки
def test_featurizer_matches_exp5():
    Gs = [nx.gnp_random_graph(100, 0.05, seed=s) for s in range(3)]
    F = NSCFeaturizer(K=16).fit_transform(Gs)
    assert F.shape == (3, 16 + 5 + 2)
    P = ref.node_sc_profile(Gs[0], K=16)
    ns = P.sum(1)
    exp5 = np.concatenate([P.mean(0), np.quantile(ns, [0.1, 0.25, 0.5, 0.75, 0.9]), [ns.mean(), ns.std()]])
    assert np.allclose(F[0], exp5)
    assert len(NSCFeaturizer(K=16).get_feature_names_out()) == F.shape[1]


def test_pyg_transform():
    torch = pytest.importorskip("torch")
    from torch_geometric.utils import from_networkx
    from nscfeat.pyg import AddNSCProfile
    G = nx.barabasi_albert_graph(60, 2, seed=8)
    data = from_networkx(G)
    out = AddNSCProfile(K=6, mode="delta")(data.clone())
    assert out.nsc_pe.shape == (60, 6) and out.nsc_pe.dtype == torch.float32
    assert np.allclose(out.nsc_pe.sum(1).numpy(), ref.node_sc(G), atol=1e-5)
    data.x = torch.ones(60, 3)
    out = AddNSCProfile(K=4, mode="sb", attr_name=None)(data)
    assert out.x.shape == (60, 3 + 8)


# ---------------------------------------------------------------- прогноз П4.B2
def test_srg_parameters():
    for G in (shrikhande(), rook4()):
        assert nx.is_strongly_regular(G) and nx.intersection_array(G) == ([6, 3], [1, 2])
    assert not nx.is_isomorphic(shrikhande(), rook4())


@pytest.mark.parametrize("name,expected", [
    ("C6 / 2×C3", True), ("декалін / біциклопентил", True), ("Шрікханде / тура 4×4", False)])
def test_b2_prediction(name, expected):
    G1, G2 = (f() for f in PAIRS[name])
    assert nx.weisfeiler_lehman_graph_hash(G1, iterations=10) == nx.weisfeiler_lehman_graph_hash(G2, iterations=10)
    for mode in ("delta", "p", "sb"):
        diff = multiset(encode(*prof(G1, K=6), mode)) != multiset(encode(*prof(G2, K=6), mode))
        assert diff == expected, mode
