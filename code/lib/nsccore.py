"""
Network Structural Complexity (NSC) — реалізація алгоритму з
A. A. Snarskii, Phys. Rev. E 110, 054309 (2024).

Для кожного вузла i виконується BFS; на k-му кроці береться індукований
підграф усіх досягнутих вузлів (s_k вузлів, b_k ребер),
p_k = b_k / (s_k (s_k - 1) / 2),  delta_k = p_k (1 - p_k),
NodeSC_i = sum_k delta_k,  NSC = mean_i NodeSC_i.

Реалізація на numba: O(Q * E) у гіршому випадку, паралельно по вузлах.
Кількість ребер оновлюється інкрементно: на кроці k додаються лише ребра,
що інцидентні новому шару BFS (до шару k-1 та всередині шару k).
"""
import numpy as np
import numba as nb
import networkx as nx
import scipy.sparse as sp


@nb.njit(cache=True)
def _node_sc_single(src, indptr, indices, dist, frontier, nxt, visited_list):
    """NodeSC одного вузла; повертає (C_i, кількість відвіданих вузлів)."""
    dist[src] = 0
    frontier[0] = src
    nf = 1
    nvis = 1
    visited_list[0] = src
    s = 1
    b2 = 0  # подвоєна кількість ребер (щоб рахувати ребра всередині шару цілими)
    c = 0.0
    k = 0
    while True:
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
                    visited_list[nvis] = v
                    nvis += 1
        if nn == 0:
            break
        for a in range(nn):
            v = nxt[a]
            for e in range(indptr[v], indptr[v + 1]):
                w = indices[e]
                d = dist[w]
                if d == k - 1:
                    b2 += 2
                elif d == k:
                    b2 += 1  # ребро всередині шару зустрінеться двічі
        s += nn
        m = s * (s - 1) / 2.0
        p = (b2 / 2.0) / m
        c += p * (1.0 - p)
        for a in range(nn):
            frontier[a] = nxt[a]
        nf = nn
    return c, nvis


@nb.njit(parallel=True, cache=True)
def _node_sc_all(indptr, indices, n, nthreads):
    out = np.zeros(n)
    chunk = (n + nthreads - 1) // nthreads
    for t in nb.prange(nthreads):
        lo = t * chunk
        hi = min(n, lo + chunk)
        if lo >= hi:
            continue
        dist = -np.ones(n, dtype=np.int64)
        frontier = np.empty(n, dtype=np.int64)
        nxt = np.empty(n, dtype=np.int64)
        visited = np.empty(n, dtype=np.int64)
        for i in range(lo, hi):
            c, nvis = _node_sc_single(i, indptr, indices, dist, frontier, nxt, visited)
            out[i] = c
            for a in range(nvis):
                dist[visited[a]] = -1
    return out


def to_csr(G):
    """networkx.Graph або scipy sparse -> (indptr, indices, n) без петель."""
    if isinstance(G, nx.Graph):
        A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.int8)
    else:
        A = sp.csr_array(G)
    A = sp.csr_array(A)
    A.setdiag(0)
    A.eliminate_zeros()
    A = ((A + A.T) > 0).astype(np.int8).tocsr()
    return A.indptr.astype(np.int64), A.indices.astype(np.int64), A.shape[0]


def node_sc(G, nthreads=None):
    """Вектор NodeSC для всіх вузлів."""
    indptr, indices, n = to_csr(G)
    if nthreads is None:
        nthreads = nb.get_num_threads() * 4
    return _node_sc_all(indptr, indices, n, nthreads)


def nsc(G):
    """NSC графа = середнє NodeSC."""
    return float(node_sc(G).mean())


def csr_from_edges(n, u, v):
    """Швидко збирає CSR з масивів кінців ребер (без networkx)."""
    data = np.ones(len(u), dtype=np.int8)
    A = sp.coo_array((data, (u, v)), shape=(n, n)).tocsr()
    return A


# ---------------------------------------------------------------- генератори
def er_graph(n, p, rng):
    """G(n,p) через вибір m ~ Binomial ребер (швидко для розріджених графів)."""
    M = n * (n - 1) // 2
    m = rng.binomial(M, p)
    u = rng.integers(0, n, size=int(m * 1.1) + 10)
    v = rng.integers(0, n, size=len(u))
    mask = u != v
    a, b = np.minimum(u[mask], v[mask]), np.maximum(u[mask], v[mask])
    # перемішування обов'язкове: np.unique сортує ключі, і зріз [:m] без нього
    # відкидав би ребра вузлів з великими індексами (виправлено 2026-10-03)
    key = rng.permutation(np.unique(a * n + b))[:m]
    return csr_from_edges(n, key // n, key % n)


def lattice_bond_perc(L, p, rng, dim=2):
    """Bond-перколяція на гіперкубічній решітці L^dim з періодичними межами."""
    n = L ** dim
    idx = np.arange(n).reshape((L,) * dim)
    us, vs = [], []
    for ax in range(dim):
        nb_ = np.roll(idx, -1, axis=ax)
        keep = rng.random(n) < p
        us.append(idx.ravel()[keep])
        vs.append(nb_.ravel()[keep])
    return csr_from_edges(n, np.concatenate(us), np.concatenate(vs))


# ------------------------------------------------ багатомасштабний профіль
@nb.njit(parallel=True, cache=True)
def _profile_all(indptr, indices, n, K, nthreads):
    """delta_k для k=1..K кожного вузла (хвіст k>K додається в останній стовпчик)."""
    out = np.zeros((n, K))
    chunk = (n + nthreads - 1) // nthreads
    for t in nb.prange(nthreads):
        lo = t * chunk
        hi = min(n, lo + chunk)
        if lo >= hi:
            continue
        dist = -np.ones(n, dtype=np.int64)
        frontier = np.empty(n, dtype=np.int64)
        nxt = np.empty(n, dtype=np.int64)
        visited = np.empty(n, dtype=np.int64)
        for i in range(lo, hi):
            dist[i] = 0
            frontier[0] = i
            nf = 1
            nvis = 1
            visited[0] = i
            s = 1
            b2 = 0
            k = 0
            while True:
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
                            b2 += 1
                s += nn
                p = (b2 / 2.0) / (s * (s - 1) / 2.0)
                out[i, min(k, K) - 1] += p * (1.0 - p)
                for a in range(nn):
                    frontier[a] = nxt[a]
                nf = nn
            for a in range(nvis):
                dist[visited[a]] = -1
    return out


def node_sc_profile(G, K=16, nthreads=None):
    """Матриця n x K: внесок кожного масштабу (кроку BFS) у NodeSC.
    Рядкова сума = NodeSC. Це «спектр» структурної складності вузла."""
    indptr, indices, n = to_csr(G)
    if nthreads is None:
        nthreads = nb.get_num_threads() * 4
    return _profile_all(indptr, indices, n, K, nthreads)
