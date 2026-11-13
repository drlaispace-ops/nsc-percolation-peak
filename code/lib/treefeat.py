"""Ознаки дерева для порівняння в П1.14: базовий рівень, NSC-профіль, портрет, NetLSD.

Портрет (Bagrow 2008): B_{l,k} — скільки вузлів мають k вузлів на відстані l. Матриця різного розміру для різних дерев,
тому векторизація тут наш вибір: частки за k = 0…8+ при l ≤ 8, далі середнє і SD оболонки в log-бінах l.
NetLSD (Tsitsulin 2018): слід теплового ядра нормованого лапласіана в 32 точках t ∈ [10⁻², 10²], поділений на n.
"""
import numba as nb
import numpy as np

from gwtree import csr_of_parent
from nscfeat import bfs_profile

NSC_K = 32
LBINS = [(9, 16), (17, 32), (33, 64), (65, 128)]
NETLSD_T = np.logspace(-2, 2, 32)


@nb.njit(cache=True)
def _far(ip, ix, n, start):
    dist = np.full(n, -1, np.int64)
    q = np.empty(n, np.int64)
    q[0] = start
    dist[start] = 0
    h, t = 0, 1
    while h < t:
        u = q[h]
        h += 1
        for e in range(ip[u], ip[u + 1]):
            w = ix[e]
            if dist[w] < 0:
                dist[w] = dist[u] + 1
                q[t] = w
                t += 1
    return q[t - 1], dist[q[t - 1]]


def diameter(ip, ix, n):
    """Діаметр дерева подвійним обходом."""
    a, _ = _far(ip, ix, n, 0)
    return int(_far(ip, ix, n, a)[1])


def degree_hist(ip, n, top=12):
    """Частки вузлів степеня 1, …, top−1 і ≥ top."""
    return np.bincount(np.minimum(np.diff(ip), top), minlength=top + 1)[1:] / n


def portrait_vec(S):
    """Векторизований портрет із матриці куль S (n × K)."""
    n, K = S.shape
    shell = np.diff(np.c_[np.ones(n), S], axis=1)
    out = []
    for l in range(min(8, K)):
        out.append(np.bincount(np.minimum(shell[:, l].astype(np.int64), 8), minlength=9) / n)
    res = np.concatenate(out)
    tail = []
    for lo, hi in LBINS:
        b = shell[:, lo - 1 : min(hi, K)]
        tail += [b.mean(), b.std()] if b.size else [0.0, 0.0]
    return np.r_[res, tail]


def netlsd_vec(ip, ix, n):
    """Слід e^{−tL}/n для нормованого лапласіана."""
    deg = np.diff(ip).astype(float)
    A = np.zeros((n, n))
    for u in range(n):
        A[u, ix[ip[u] : ip[u + 1]]] = 1.0
    d = 1.0 / np.sqrt(deg)
    L = np.eye(n) - d[:, None] * A * d[None, :]
    lam = np.linalg.eigvalsh(L)
    return np.exp(-np.outer(NETLSD_T, lam)).sum(1) / n


def tree_features(parent, K=128):
    """Словник груп ознак дерева; parent — будь-який порядок із parent[0] = −1."""
    n = len(parent)
    ip, ix = csr_of_parent(parent)
    S, _, D = bfs_profile(ip, ix, n, K=K, full=True, nthreads=None)
    dm = D.mean(0)
    deg = np.diff(ip)
    return dict(C=np.array([D.sum(1).mean()]), diam=np.array([diameter(ip, ix, n)], float), deg=degree_hist(ip, n),
                nsc=np.r_[dm[: NSC_K - 1], dm[NSC_K - 1 :].sum()], portrait=portrait_vec(S), netlsd=netlsd_vec(ip, ix, n),
                degs=np.bincount(deg, minlength=n))
