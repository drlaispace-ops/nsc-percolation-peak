"""
Ядро NSC-ознак: профіль BFS-куль кожного вузла.

Для вузла i і радіуса k = 1..K береться куля B_k(i) — індукований підграф
вузлів на відстані ≤ k (Snarskii, PRE 110, 054309, 2024):
  s_k — кількість вузлів кулі, b_k — кількість ребер,
  p_k = b_k / (s_k (s_k − 1) / 2),  δ_k = p_k (1 − p_k),  NodeSC_i = Σ_k δ_k.

Ядро повертає три матриці n × K: S (s_k), B (b_k) і D (δ_k).
  * Після того як BFS вичерпав компоненту, s_k і b_k лишаються сталими
    (куля = компонента), а δ_k = 0 — так само, як в оригінальному алгоритмі.
  * full=True: BFS іде до кінця, і внески k > K додаються в останній
    стовпчик D, тож рядкова сума D дорівнює NodeSC (як nsc.node_sc_profile).
    Вартість O(n·E) на зв'язному графі.
  * full=False: BFS зупиняється на радіусі K, вартість O(Σ_i |B_K(i)|);
    останній стовпчик D містить лише δ_K.
Ізольований вузол: s_k = 1, b_k = 0, δ_k = 0, p_k = 0.
"""
import numpy as np
import numba as nb
import scipy.sparse as sp

MODES = ("delta", "p", "sb")
SERIAL_MAX_N = 4096  # менші графи рахуються в одному потоці (без накладних витрат пулу)


@nb.njit(cache=True)
def _bfs_range(lo, hi, indptr, indices, K, full, S, B, D, dist, frontier, nxt, visited):
    for i in range(lo, hi):
        dist[i] = 0
        frontier[0] = i
        nf = 1
        nvis = 1
        visited[0] = i
        s = 1
        b2 = 0  # подвоєна кількість ребер кулі
        k = 0
        kdone = 0
        while True:
            if not full and k >= K:
                break
            k += 1
            nn = 0
            for a in range(nf):
                u = frontier[a]
                for e in range(indptr[u], indptr[u + 1]):
                    v = indices[e]
                    if dist[v] == -1:
                        dist[v] = k
                        nxt[nn] = v
                        nn += 1
                        visited[nvis] = v
                        nvis += 1
            if nn == 0:
                break
            for a in range(nn):
                v = nxt[a]
                for e in range(indptr[v], indptr[v + 1]):
                    d = dist[indices[e]]
                    if d == k - 1:
                        b2 += 2
                    elif d == k:
                        b2 += 1  # ребро всередині шару зустрінеться двічі
            s += nn
            kdone = k
            p = (b2 / 2.0) / (s * (s - 1) / 2.0)
            D[i, min(k, K) - 1] += p * (1.0 - p)
            if k <= K:
                S[i, k - 1] = s
                B[i, k - 1] = b2 // 2
            for a in range(nn):
                frontier[a] = nxt[a]
            nf = nn
        for j in range(min(kdone, K), K):  # куля насичена: s, b сталі
            S[i, j] = s
            B[i, j] = b2 // 2
        for a in range(nvis):
            dist[visited[a]] = -1


def _scratch(n):
    return (-np.ones(n, dtype=np.int64), np.empty(n, dtype=np.int64),
            np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64))


@nb.njit(cache=True)
def _profile_serial(indptr, indices, n, K, full):
    S = np.ones((n, K), dtype=np.int64)
    B = np.zeros((n, K), dtype=np.int64)
    D = np.zeros((n, K))
    dist = -np.ones(n, dtype=np.int64)
    frontier = np.empty(n, dtype=np.int64)
    nxt = np.empty(n, dtype=np.int64)
    visited = np.empty(n, dtype=np.int64)
    _bfs_range(0, n, indptr, indices, K, full, S, B, D, dist, frontier, nxt, visited)
    return S, B, D


@nb.njit(parallel=True, cache=True)
def _profile_parallel(indptr, indices, n, K, full, nchunks):
    S = np.ones((n, K), dtype=np.int64)
    B = np.zeros((n, K), dtype=np.int64)
    D = np.zeros((n, K))
    chunk = (n + nchunks - 1) // nchunks
    for t in nb.prange(nchunks):
        lo = t * chunk
        hi = min(n, lo + chunk)
        if lo >= hi:
            continue
        dist = -np.ones(n, dtype=np.int64)
        frontier = np.empty(n, dtype=np.int64)
        nxt = np.empty(n, dtype=np.int64)
        visited = np.empty(n, dtype=np.int64)
        _bfs_range(lo, hi, indptr, indices, K, full, S, B, D, dist, frontier, nxt, visited)
    return S, B, D


# ------------------------------------------------------------------ входи
def csr_from_edge_index(edge_index, num_nodes):
    """edge_index (2, E) → (indptr, indices): симетризація, без петель і кратних ребер."""
    ei = np.asarray(edge_index, dtype=np.int64)
    u, v = ei[0], ei[1]
    m = u != v
    u, v = u[m], v[m]
    A = sp.coo_array((np.ones(2 * len(u), dtype=np.int8),
                      (np.concatenate([u, v]), np.concatenate([v, u]))),
                     shape=(num_nodes, num_nodes)).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    return A.indptr.astype(np.int64), A.indices.astype(np.int64)


def csr_from_graph(G):
    """networkx.Graph або scipy sparse → (indptr, indices, n).
    Для networkx рядки йдуть у порядку G.nodes() (як nx.to_scipy_sparse_array), а не за мітками."""
    if hasattr(G, "edges") and hasattr(G, "nodes"):  # networkx без жорсткої залежності
        import networkx as nx
        G = nx.convert_node_labels_to_integers(G, ordering="default")
        n = G.number_of_nodes()
        ei = np.array(list(G.edges()), dtype=np.int64).reshape(-1, 2).T
    else:
        A = sp.coo_array(G)
        n = A.shape[0]
        ei = np.vstack([A.row, A.col]).astype(np.int64)
    indptr, indices = csr_from_edge_index(ei, n)
    return indptr, indices, n


# ------------------------------------------------------------------ API
def bfs_profile(indptr, indices, n, K=8, full=True, nthreads=None):
    """(S, B, D) — матриці n × K: розмір кулі, кількість ребер кулі, δ_k."""
    if K < 1:
        raise ValueError("K ≥ 1")
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    if n <= SERIAL_MAX_N and nthreads is None:
        return _profile_serial(indptr, indices, n, K, full)
    nchunks = (nthreads or nb.get_num_threads()) * 4
    return _profile_parallel(indptr, indices, n, K, full, nchunks)


def ball_density(S, B):
    """p_k = b_k / C(s_k, 2); для s_k = 1 — 0."""
    pairs = S * (S - 1) / 2.0
    return np.divide(B, pairs, out=np.zeros(S.shape), where=pairs > 0)


def encode(S, B, D, mode="delta"):
    """Вузлове кодування з профілю.
    delta — δ_k (K стовпчиків; величина зі статті, рядкова сума = NodeSC при full=True);
            δ_k = 0 позначає насичену кулю (ексцентриситет вузла < k);
    p     — p_k (K стовпчиків); δ = p(1−p) неін'єктивна, але p після насичення
            стала й ексцентриситет втрачає — жоден з режимів delta/p не містить іншого
            (див. results/P4/b2_atlas_expressivity.json, rrg4);
    sb    — [log s_k, b_k/s_k] (2K стовпчиків; сирий профіль куль: з нього виводяться
            і p_k, і δ_k при k ≤ K)."""
    if mode == "delta":
        return D
    if mode == "p":
        return ball_density(S, B)
    if mode == "sb":
        return np.hstack([np.log(S), B / S])
    raise ValueError(f"mode ∈ {MODES}")


def nsc_encoding(edge_index, num_nodes, K=8, mode="delta", full=True, nthreads=None):
    """Вузлове NSC-кодування (num_nodes × K або × 2K для mode='sb')."""
    indptr, indices = csr_from_edge_index(edge_index, num_nodes)
    S, B, D = bfs_profile(indptr, indices, num_nodes, K, full, nthreads)
    return encode(S, B, D, mode)
