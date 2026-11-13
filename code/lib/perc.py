"""Генератори перколяції і вимірювання NSC з розкладом за кластерами (П1).

Перенесено з code/exp2c_duality.py (site_sq, bond_tri) і code/exp8_dimension_clusters.py
(rrg_bond, measure, grow_pr) без глобального стану: кожна функція отримує свій rng.
Межі — періодичні (конвенція FSS). Решітки:
  hypercubic_bond(L, p, rng, d)  — bond, Z_L^d;
  hypercubic_site(L, p, rng, d)  — site, Z_L^d; повертає (A, occ);
  triangular_bond(L, p, rng)     — bond, трикутна решітка L×L;
  rrg_bond(n, z, p, rng)         — bond на випадковому z-регулярному графі;
  er_graph(n, p, rng)            — G(n, p) (виправлений 03.10.2026);
  product_rule(n, M, seed)       — вибухова перколяція, правило добутку Achlioptas.
measure(A, occ) повертає NSC у двох нормалізаціях (на всі вузли і на зайняті), частку гіганта P
і середні NodeSC гіганта та скінченних кластерів: NSC = (1 − P)·g_fin + P·g_giant.
"""
import networkx as nx
import numba as nb
import numpy as np
import scipy.sparse.csgraph as cg

from nsccore import _node_sc_all, csr_from_edges, er_graph, lattice_bond_perc, to_csr  # noqa: F401


def hypercubic_bond(L, p, rng, d=2):
    """Bond-перколяція на Z_L^d з періодичними межами (L ≥ 3, інакше сусіди ±1 збігаються)."""
    return lattice_bond_perc(L, p, rng, d)


def hypercubic_site(L, p, rng, d=2):
    """Site-перколяція на Z_L^d з періодичними межами: (матриця суміжності, маска зайнятих)."""
    n = L ** d
    idx = np.arange(n).reshape((L,) * d)
    occ = rng.random(n) < p
    us, vs = [], []
    for ax in range(d):
        a, b = idx.ravel(), np.roll(idx, -1, axis=ax).ravel()
        k = occ[a] & occ[b]
        us.append(a[k])
        vs.append(b[k])
    return csr_from_edges(n, np.concatenate(us), np.concatenate(vs)), occ


def triangular_bond(L, p, rng):
    """Bond-перколяція на трикутній решітці L×L (квадратна + одна діагональ), періодичні межі."""
    n = L * L
    idx = np.arange(n).reshape(L, L)
    us, vs = [], []
    for sh in ((-1, 0), (0, -1), (-1, -1)):
        b = np.roll(idx, sh, axis=(0, 1)).ravel()
        k = rng.random(n) < p
        us.append(idx.ravel()[k])
        vs.append(b[k])
    return csr_from_edges(n, np.concatenate(us), np.concatenate(vs))


def rrg_bond(n, z, p, rng):
    """Bond-перколяція на випадковому z-регулярному графі з n вузлами."""
    e = np.array(nx.random_regular_graph(z, n, seed=int(rng.integers(2 ** 31))).edges())
    keep = rng.random(len(e)) < p
    return csr_from_edges(n, e[keep, 0], e[keep, 1])


@nb.njit(cache=True)
def _find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


@nb.njit(cache=True)
def product_rule(n, M, seed):
    """Правило добутку Achlioptas: з двох випадкових ребер додаємо те, що з'єднує кластери
    з меншим добутком розмірів. Повертає кінці M ребер у порядку додавання (t = M/n)."""
    np.random.seed(seed)
    par = np.arange(n)
    size = np.ones(n, np.int64)
    U = np.empty(M, np.int64)
    V = np.empty(M, np.int64)
    for t in range(M):
        a = np.random.randint(n)
        b = np.random.randint(n)
        while b == a:
            b = np.random.randint(n)
        c = np.random.randint(n)
        d = np.random.randint(n)
        while d == c:
            d = np.random.randint(n)
        if size[_find(par, c)] * size[_find(par, d)] < size[_find(par, a)] * size[_find(par, b)]:
            a, b = c, d
        U[t] = a
        V[t] = b
        ra, rb = _find(par, a), _find(par, b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            par[rb] = ra
            size[ra] += size[rb]
    return U, V


def measure(A, occ=None, nthreads=None):
    """NSC і розклад за кластерами для графа A (scipy sparse).

    occ — маска зайнятих вузлів (site-перколяція); None — усі вузли «зайняті».
    Повертає dict:
      nsc      — середнє NodeSC за всіма n вузлами (як у PRE 2024);
      nsc_occ  — за зайнятими (для site: nsc = (n_occ/n)·nsc_occ);
      P, P_occ — частка найбільшого кластера серед усіх / зайнятих вузлів;
      g_giant, g_fin — середнє NodeSC у найбільшому кластері / у решті зайнятих вузлів;
      n, n_occ.
    """
    ip, ix, n = to_csr(A)
    if nthreads is None:
        nthreads = nb.get_num_threads() * 4
    ns = _node_sc_all(ip, ix, n, nthreads)
    occ = np.ones(n, bool) if occ is None else np.asarray(occ, bool)
    _, lab = cg.connected_components(A, directed=False)
    sizes = np.bincount(lab[occ], minlength=lab.max() + 1)
    g = (lab == sizes.argmax()) & occ
    fin = occ & ~g
    n_occ = int(occ.sum())
    return dict(nsc=float(ns.mean()), nsc_occ=float(ns[occ].mean()) if n_occ else 0.0,
                P=float(g.sum() / n), P_occ=float(g.sum() / n_occ) if n_occ else 0.0,
                g_giant=float(ns[g].mean()) if g.any() else 0.0,
                g_fin=float(ns[fin].mean()) if fin.any() else 0.0, n=n, n_occ=n_occ)
