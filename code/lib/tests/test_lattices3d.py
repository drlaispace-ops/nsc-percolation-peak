"""Тести решіток П1.3b (bcc, fcc, diamond; sc — контроль).
    .venv/bin/python -m pytest code/lib/tests/test_lattices3d.py -q

Незалежна перевірка: (1) кожне ребро графа має зміщення (за модулем бокса) з геометричного набору векторів
сусідів; (2) кількість коротких циклів через вузол на графі = перебору простих шляхів на нескінченній решітці
(з координатами, без графа); (3) loops_per_node (сліди A³, A⁴) = перебору.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

import lattices3d as lt  # noqa: E402
import sampled as sm  # noqa: E402

NAMES = ("sc", "bcc", "fcc", "diamond")
DIA = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]


def nbr_vectors(name, x):
    """Вектори сусідів точки x нескінченної решітки (цілі координати: sc — a, bcc і fcc — a/2, diamond — a/4)."""
    if name == "sc":
        return [s * np.eye(3, dtype=int)[i] for i in range(3) for s in (1, -1)]
    if name == "bcc":
        return [np.array(t) for t in itertools.product((1, -1), repeat=3)]
    if name == "fcc":
        return [np.array(t) for t in set(itertools.permutations((1, 1, 0))) | set(itertools.permutations((1, -1, 0)))
                | set(itertools.permutations((-1, -1, 0)))]
    sgn = 1 if x[0] % 2 == 0 else -1                # A — парні координати, B — непарні
    return [sgn * np.array(t) for t in DIA]


def coords(name, L):
    """Координати вузлів за індексами генератора і розмір бокса в тих самих одиницях."""
    n = lt.EDGES[name](L)[0]
    i = np.arange(n)
    if name == "sc":
        return np.stack(np.unravel_index(i, (L,) * 3), 1), L
    if name == "bcc":
        r = np.stack(np.unravel_index(i % L ** 3, (L,) * 3), 1)
        return 2 * r + (i >= L ** 3)[:, None], 2 * L

    def fcc(j):
        x, y, zh = j // (2 * L * L), (j // L) % (2 * L), j % L
        return np.stack([x, y, 2 * zh + (x + y) % 2], 1)
    if name == "fcc":
        return fcc(i), 2 * L
    nf = 4 * L ** 3
    return 2 * fcc(i % nf) + (i >= nf)[:, None], 4 * L


def cycles_geometric(name, k):
    """Прості цикли довжини k через початок координат нескінченної решітки (перебір шляхів / 2)."""
    o = np.zeros(3, int)
    cnt = 0

    def dfs(x, depth, seen):
        nonlocal cnt
        for t in nbr_vectors(name, x):
            y = x + t
            if depth == k - 1:
                cnt += not y.any()
            elif tuple(y) not in seen:
                seen.add(tuple(y))
                dfs(y, depth + 1, seen)
                seen.discard(tuple(y))
    dfs(o, 0, {tuple(o)})
    return cnt // 2


def cycles_graph(A, s, k):
    ip, ix = A.indptr, A.indices
    cnt = 0

    def dfs(x, depth, seen):
        nonlocal cnt
        for y in ix[ip[x]:ip[x + 1]]:
            if depth == k - 1:
                cnt += y == s
            elif y not in seen:
                seen.add(y)
                dfs(y, depth + 1, seen)
                seen.discard(y)
    dfs(s, 0, {s})
    return cnt // 2


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("L", [3, 4])
def test_structure(name, L):
    n, u, v = lt.EDGES[name](L)
    z = lt.Z[name]
    assert len(u) == n * z // 2
    assert not np.any(u == v)                                     # петель немає
    key = np.minimum(u, v) * n + np.maximum(u, v)
    assert len(np.unique(key)) == len(key)                        # кратних ребер немає
    deg = np.bincount(np.r_[u, v], minlength=n)
    assert np.all(deg == z)
    X, M = coords(name, L)
    assert len(np.unique(X % M, axis=0)) == n                     # різні вузли — різні точки бокса
    dx = np.mod(X[v] - X[u] + M // 2, M) - M // 2                 # мінімальний образ (періодичність)
    for e in range(len(u)):
        assert any(np.array_equal(dx[e], t) for t in nbr_vectors(name, X[u[e]]))


# очікувані значення на вузол (нескінченна решітка): трикутники, 4-цикли, 6-цикли (лише diamond), |S₂|
KNOWN = {"sc": (0, 12, None, 18), "bcc": (0, 48, None, 26), "fcc": (24, 132, None, 42), "diamond": (0, 0, 12, 12)}


@pytest.mark.parametrize("name", NAMES)
def test_short_cycles(name):
    tri, sq, hexa, s2 = KNOWN[name]
    g3, g4 = cycles_geometric(name, 3), cycles_geometric(name, 4)
    assert g3 == tri
    if sq is not None:
        assert g4 == sq
    L = 5 if name == "sc" else 3                                  # sc: петля навколо тора — L кроків
    n, u, v = lt.EDGES[name](L)
    A = lt.adjacency(n, u, v)
    for s in (0, n - 1, n // 2 + 1):                              # обидві підгратки
        assert cycles_graph(A, s, 3) == g3
        assert cycles_graph(A, s, 4) == g4
    if hexa is not None:
        assert cycles_geometric(name, 6) == hexa
        assert cycles_graph(A, 0, 6) == hexa and cycles_graph(A, n - 1, 6) == hexa
    m = lt.loops_per_node(n, u, v)
    assert m["z"] == lt.Z[name] and m["triangles"] == g3 and m["squares"] == g4 and m["sphere2"] == s2


@pytest.mark.parametrize("name", ("bcc", "fcc", "diamond"))
def test_coupled(name):
    rng = np.random.default_rng(5)
    n, u, v, w, ws = getattr(lt, f"coupled_{name}")(4, rng)
    assert ws is None and w.shape == u.shape and (0 <= w).all() and (w < 1).all()
    A, occ = sm.at_p(n, u, v, w, 1.0)
    assert occ is None and A.nnz == len(u)
    A, _ = sm.at_p(n, u, v, w, 0.3)
    assert abs(A.nnz / len(u) - 0.3) < 0.05
