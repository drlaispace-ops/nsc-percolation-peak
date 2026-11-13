"""П1.3b. Тривимірні решітки з однаковим z і різними короткими петлями: ОЦК (bcc, z = 8), ГЦК (fcc, z = 12),
алмазна (diamond, z = 4); bond-перколяція, періодичні межі, спряжені реалізації.

Сигнатура генераторів — як sampled.coupled_bonds: coupled_<решітка>(L, rng) → (n, u, v, w, None);
граф при p — зв'язки з w < p (sampled.at_p). L — кількість кубічних комірок уздовж ребра бокса (L ≥ 3 для bcc, fcc, diamond; sc — L ≥ 5),
щоб петлі навколо тора (2L кроків у bcc, fcc; 4L у diamond; L у sc) були довшими за найкоротші локальні петлі
(4; 6 для diamond) і не змінювали їхньої кількості.

Побудова (координати цілі, періодичні):
  * bcc: n = 2L³. Підгратка A — кути комірок r ∈ Z_L³ (індекс r), B — центри (індекс L³ + r, центр r + ½).
    Зв'язки A_r — B_{r+δ}, δ ∈ {0, −1}³ (8 на вузол A). Двочастковий граф: трикутників немає.
  * fcc: n = 4L³. Вузли — точки x ∈ Z_{2L}³ з парною сумою координат (одиниця a/2); сусіди — 12 векторів
    перестановок (±1, ±1, 0); зв'язки — 6 «додатних» векторів від кожного вузла.
  * diamond: n = 8L³. Дві ГЦК-підгратки: A_y (y — вузол fcc) і B_y (атом у y + (¼,¼,¼)a); зв'язки
    A_y — B_{y+t}, t ∈ {0, (0,−1,−1), (−1,0,−1), (−1,−1,0)} (одиниця a/2). Найкоротші петлі — шестикутники.

Короткі петлі на вузол (нескінченна решітка; перевіряється в тестах незалежним перебором шляхів):
  трикутники через вузол: sc 0, bcc 0, fcc 24 (24 ребра кубооктаедра сусідів), diamond 0;
  4-цикли через вузол: sc 12, bcc 48, fcc 132 (з перебором хорд), diamond 0; 6-цикли diamond — 12 (кожен атом у 12 кільцях).
`loops_per_node` рахує ці числа з графа через сліди A³, A⁴ (для 4-циклів — з поправкою на неповоротні блукання)
і розмір другої сфери |S₂| (порівняння з деревом: z(z − 1)).
"""
import numpy as np
import scipy.sparse as sp


def _idx3(L):
    return np.arange(L ** 3, dtype=np.int64).reshape(L, L, L)


def bcc_edges(L):
    n = 2 * L ** 3
    r = _idx3(L)
    u, v = [], []
    for dx in (0, -1):
        for dy in (0, -1):
            for dz in (0, -1):
                u.append(r.ravel())
                v.append(L ** 3 + np.roll(r, (-dx, -dy, -dz), axis=(0, 1, 2)).ravel())
    return n, np.concatenate(u), np.concatenate(v)


FCC_HALF = ((1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1))


def _fcc_points(L):
    """Точки ГЦК у Z_{2L}³ з парною сумою координат і їхні індекси (x·2L + y)·L + z//2."""
    M = 2 * L
    g = np.stack(np.meshgrid(np.arange(M), np.arange(M), np.arange(M), indexing="ij"), -1).reshape(-1, 3)
    g = g[g.sum(1) % 2 == 0]
    return g, _fcc_index(g, L)


def _fcc_index(x, L):
    M = 2 * L
    x = np.mod(x, M)
    return (x[:, 0] * M + x[:, 1]) * L + x[:, 2] // 2


def fcc_edges(L):
    g, i0 = _fcc_points(L)
    n = 4 * L ** 3
    assert len(g) == n and np.array_equal(np.sort(i0), np.arange(n))
    u = np.concatenate([i0] * 6)
    v = np.concatenate([_fcc_index(g + np.array(t), L) for t in FCC_HALF])
    return n, u, v


DIAMOND_T = ((0, 0, 0), (0, -1, -1), (-1, 0, -1), (-1, -1, 0))


def diamond_edges(L):
    g, i0 = _fcc_points(L)
    nf = 4 * L ** 3
    u = np.concatenate([i0] * 4)
    v = np.concatenate([nf + _fcc_index(g + np.array(t), L) for t in DIAMOND_T])
    return 2 * nf, u, v


def sc_edges(L):
    n = L ** 3
    r = _idx3(L)
    return n, np.concatenate([r.ravel()] * 3), np.concatenate([np.roll(r, -1, axis=a).ravel() for a in range(3)])


EDGES = {"bcc": bcc_edges, "fcc": fcc_edges, "diamond": diamond_edges, "sc": sc_edges}
Z = {"bcc": 8, "fcc": 12, "diamond": 4, "sc": 6}


def coupled(name, L, rng):
    """Спряжена bond-реалізація: (n, u, v, w, None), як sampled.coupled_bonds."""
    n, u, v = EDGES[name](L)
    return n, u, v, rng.random(len(u)), None


def coupled_bcc(L, rng):
    return coupled("bcc", L, rng)


def coupled_fcc(L, rng):
    return coupled("fcc", L, rng)


def coupled_diamond(L, rng):
    return coupled("diamond", L, rng)


def adjacency(n, u, v):
    A = sp.coo_array((np.ones(len(u), np.int64), (u, v)), shape=(n, n)).tocsr()
    return (A + A.T).tocsr()


def loops_per_node(n, u, v):
    """Середні на вузол: трикутники через вузол, 4-цикли через вузол, |S₂| (друга сфера), z.
    4-цикли через i: [(A⁴)_ii − d_i² − Σ_{j~i}(d_j − 1)]/2 (віднято повертальні замкнені блукання)."""
    A = adjacency(n, u, v).astype(np.int64)
    d = np.asarray(A.sum(1)).ravel()
    A2 = A @ A
    tri = np.asarray((A2.multiply(A)).sum(1)).ravel() / 2
    a4 = np.asarray(A2.multiply(A2).sum(1)).ravel()          # (A⁴)_ii = Σ_j (A²)_ij²
    sq = (a4 - d ** 2 - A @ (d - 1)) / 2
    B = ((A2 + A) > 0).astype(np.int8).tolil()
    B.setdiag(0)
    s2 = np.asarray(B.tocsr().sum(1)).ravel() - d             # вузли на відстані 2
    return dict(z=float(d.mean()), triangles=float(tri.mean()), squares=float(sq.mean()), sphere2=float(s2.mean()))
