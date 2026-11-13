"""Дерева Гальтона–Ватсона як графи: генератори, умовні за розміром вибірки, NodeSC через переукорінення.

Дерево задається масивом parent у порядку BFS: parent[0] = -1, parent[i] < i.
Закони нащадків: POISSON (a = середнє), GEOM (P(j) = (1−a)a^j), BINOM (Bin(b, a)).
Перенесено з прототипу code/proto_gw/gwlib.py; перевірки — code/lib/tests/test_gw.py.
"""
import numba as nb
import numpy as np

POISSON, GEOM, BINOM = 0, 1, 2
# Переукорінення тримає стек (H+1)·(2H+3) цілих; на вироджених деревах (шлях) це сотні ГБ.
MAX_STACK_BYTES = 4 * 10 ** 9


@nb.njit(cache=True)
def _draw(mode, a, b):
    if mode == 0:
        return np.random.poisson(a)          # Poisson(a)
    if mode == 1:
        return np.random.geometric(1.0 - a) - 1   # P(j) = (1-a) a^j
    if mode == 2:
        return np.random.binomial(int(b), a)   # Bin(b, a)
    return 0


@nb.njit(cache=True)
def gw_parent(mr, ar, br, mc, ac, bc, cap, parent):
    """Дерево ГВ у порядку BFS: parent[i] < i, parent[0] = -1. Повертає n або -1 при перевищенні cap."""
    parent[0] = -1
    n = 1
    i = 0
    while i < n:
        z = _draw(mr, ar, br) if i == 0 else _draw(mc, ac, bc)
        if n + z > cap:
            return -1
        for j in range(z):
            parent[n + j] = i
        n += z
        i += 1
    return n


@nb.njit(cache=True)
def tree_sc_reroot(parent, n, g, Kacc, acc):
    """Точне NodeSC усіх вузлів дерева через переукорінення: час O(Σ_v ecc(v) + Σ_v height(v)).
    parent[i] < i (порядок BFS). g[s] = 2(s-2)/s^2. acc[k-1] += сума δ_k за вузлами (k<=Kacc, хвіст — в останній)."""
    out = np.zeros(n)
    if n < 3:
        return out
    h = np.zeros(n, np.int64)
    for v in range(n - 1, 0, -1):
        p = parent[v]
        if h[v] + 1 > h[p]:
            h[p] = h[v] + 1
    off = np.zeros(n + 1, np.int64)
    for v in range(n):
        off[v + 1] = off[v] + h[v] + 1
    down = np.zeros(off[n], np.int64)
    for v in range(n - 1, -1, -1):
        down[off[v]] += 1
        if v > 0:
            p = parent[v]
            lv = h[v] + 1
            for j in range(lv):
                down[off[p] + 1 + j] += down[off[v] + j]
    # діти неперервні в порядку BFS
    first = np.full(n + 1, -1, np.int64)
    cnt = np.zeros(n, np.int64)
    for v in range(n - 1, 0, -1):
        first[parent[v]] = v
        cnt[parent[v]] += 1
    depth = np.zeros(n, np.int64)
    H = 0
    for v in range(1, n):
        depth[v] = depth[parent[v]] + 1
        if depth[v] > H:
            H = depth[v]
    W = 2 * H + 3  # довжина L_v не більша за діаметр + 2
    stack_L = np.zeros((H + 1, W), np.int64)
    stack_len = np.zeros(H + 1, np.int64)
    # обхід у глибину явним стеком
    st = np.empty(n, np.int64)
    sp = 0
    st[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        v = st[sp]
        d = depth[v]
        L = stack_L[d]
        if v == 0:
            ln = h[0] + 1
            for j in range(ln):
                L[j] = down[off[0] + j]
        else:
            P = stack_L[d - 1]
            lp = stack_len[d - 1]
            hv = h[v]
            ov = off[v]
            ln = max(hv + 1, lp + 1)
            L[0] = 1
            for k in range(1, ln):
                x = 0
                if k <= hv:
                    x += down[ov + k]
                if k - 1 < lp:
                    x += P[k - 1]
                if 0 <= k - 2 <= hv:
                    x -= down[ov + k - 2]
                L[k] = x
            while ln > 1 and L[ln - 1] == 0:
                ln -= 1
        stack_len[d] = ln
        s = 1
        c = 0.0
        for k in range(1, ln):
            s += L[k]
            t = g[s]
            c += t
            acc[min(k, Kacc) - 1] += t
        out[v] = c
        for j in range(cnt[v]):
            st[sp] = first[v] + j
            sp += 1
    return out


def gtable(nmax):
    g = np.zeros(nmax + 2)
    s = np.arange(2, nmax + 2, dtype=np.float64)
    g[2:] = 2 * (s - 2) / s ** 2
    return g


@nb.njit(cache=True)
def _gw_batch(seed, R, mr, ar, br, mc, ac, bc, cap, g, K):
    np.random.seed(seed)
    parent = np.empty(cap, np.int64)
    ns = np.zeros(R, np.int64)
    root = np.zeros(R)
    mean = np.zeros(R)
    prof = np.zeros(K)   # Σ_T (1/|T|) Σ_v δ_k
    over = 0
    acc = np.zeros(K)
    for r in range(R):
        n = gw_parent(mr, ar, br, mc, ac, bc, cap, parent)
        if n < 0:
            over += 1
            continue
        acc[:] = 0.0
        x = tree_sc_reroot(parent, n, g, K, acc)
        ns[r] = n
        root[r] = x[0]
        mean[r] = x.mean()
        prof += acc / n
    return ns, root, mean, prof, over


@nb.njit(parallel=True, cache=True)
def gw_stats(seeds, Rchunk, mr, ar, br, mc, ac, bc, cap, g, K):
    C = len(seeds)
    NS = np.zeros((C, Rchunk), np.int64)
    RT = np.zeros((C, Rchunk))
    MN = np.zeros((C, Rchunk))
    PF = np.zeros((C, K))
    OV = np.zeros(C, np.int64)
    for t in nb.prange(C):
        a, b, c_, d, e = _gw_batch(seeds[t], Rchunk, mr, ar, br, mc, ac, bc, cap, g, K)
        NS[t] = a
        RT[t] = b
        MN[t] = c_
        PF[t] = d
        OV[t] = e
    return NS, RT, MN, PF, OV


# ------------------------------------------------- Умовні за розміром генератори
@nb.njit(cache=True)
def prufer_parent(n, seed):
    """Рівномірне мічене дерево (= Poisson-ГВ за умови |T|=n): декодування коду Прюфера O(n) + BFS від вузла 0."""
    np.random.seed(seed)
    if n == 1:
        return np.array([-1], np.int64)
    code = np.random.randint(0, n, n - 2) if n > 2 else np.zeros(0, np.int64)
    deg = np.ones(n, np.int64)
    for x in code:
        deg[x] += 1
    U = np.empty(n - 1, np.int64)
    V = np.empty(n - 1, np.int64)
    ptr = 0
    while deg[ptr] != 1:
        ptr += 1
    leaf = ptr
    for i in range(n - 2):
        x = code[i]
        U[i] = leaf
        V[i] = x
        deg[x] -= 1
        if deg[x] == 1 and x < ptr:
            leaf = x
        else:
            ptr += 1
            while deg[ptr] != 1:
                ptr += 1
            leaf = ptr
    # останнє ребро: leaf — (n-1)
    U[n - 2] = leaf
    V[n - 2] = n - 1
    return _bfs_relabel(n, U, V, 0)


@nb.njit(cache=True)
def _bfs_relabel(n, U, V, r):
    ip = np.zeros(n + 1, np.int64)
    for i in range(n - 1):
        ip[U[i] + 1] += 1
        ip[V[i] + 1] += 1
    ip = np.cumsum(ip)
    pos = ip[:-1].copy()
    ix = np.empty(2 * (n - 1), np.int64)
    for i in range(n - 1):
        ix[pos[U[i]]] = V[i]
        pos[U[i]] += 1
        ix[pos[V[i]]] = U[i]
        pos[V[i]] += 1
    new = np.full(n, -1, np.int64)
    parent = np.empty(n, np.int64)
    order = np.empty(n, np.int64)
    order[0] = r
    new[r] = 0
    parent[0] = -1
    m = 1
    i = 0
    while i < m:
        u = order[i]
        for e in range(ip[u], ip[u + 1]):
            w = ix[e]
            if new[w] < 0:
                new[w] = m
                parent[m] = i
                order[m] = w
                m += 1
        i += 1
    return parent


@nb.njit(cache=True)
def cycle_lemma_parent(xi):
    """xi — кількості нащадків (Σ = n-1) у довільному порядку. Циклічний зсув (лема про цикл) робить
    шлях Лукасевича додатним; повертає parent у порядку BFS (черга)."""
    n = len(xi)
    # мінімум префіксної суми Σ(xi-1): старт одразу після останнього мінімуму
    s = 0
    mn = 0
    start = 0
    for i in range(n):
        s += xi[i] - 1
        if s < mn:
            mn = s
            start = i + 1
    start %= n
    parent = np.empty(n, np.int64)
    parent[0] = -1
    m = 1
    for i in range(n):
        z = xi[(start + i) % n]
        for j in range(z):
            parent[m] = i
            m += 1
    return parent


@nb.njit(cache=True)
def cond_poisson_parent(n, seed):
    """Спосіб Devroye для Пуассона: (ξ_1..ξ_n) | Σ=n-1 — Multinomial(n-1; 1/n,...,1/n), далі лема про цикл."""
    np.random.seed(seed)
    xi = np.zeros(n, np.int64)
    for _ in range(n - 1):
        xi[np.random.randint(0, n)] += 1
    return cycle_lemma_parent(xi)


@nb.njit(cache=True)
def cond_geom_parent(n, seed):
    """Геометричні нащадки (рівномірне плоске дерево, Catalan): рівномірна композиція n-1 у n частин, лема про цикл."""
    np.random.seed(seed)
    perm = np.random.permutation(2 * n - 2)  # n−1 кульок і n−1 перегородок: n частин
    mark = np.zeros(2 * n - 2, np.int64)
    mark[perm[: n - 1]] = 1
    xi = np.zeros(n, np.int64)
    j = 0
    for i in range(2 * n - 2):
        if mark[i] == 1:
            xi[j] += 1
        else:
            j += 1
    return cycle_lemma_parent(xi)


@nb.njit(cache=True)
def cond_law_parent(n, seed, root_cdf, mode, b):
    """Дерево розміру n з окремим законом кореня (Devroye). root_cdf — gwexact.root_given_n.
    За умови Σξ = m нащадки: Bin — рівномірна підмножина (n−1)·b слотів, Geom — рівномірна композиція, Poisson — мультиноміальний.
    Лема про цикл для лісу: j добрих зсувів — після перших проходжень рівнів min, …, min+j−1; беремо рівномірний."""
    np.random.seed(seed)
    j = int(np.searchsorted(root_cdf, np.random.random(), side="right"))
    t = n - 1
    m = t - j
    xi = np.zeros(t, np.int64)
    if mode == 2:
        perm = np.random.permutation(t * b)
        for i in range(m):
            xi[perm[i] // b] += 1
    elif mode == 1:
        size = m + t - 1
        mark = np.zeros(size, np.int64)
        perm = np.random.permutation(size)
        for i in range(m):
            mark[perm[i]] = 1
        h = 0
        for i in range(size):
            if mark[i] == 1:
                xi[h] += 1
            else:
                h += 1
    else:
        for _ in range(m):
            xi[np.random.randint(0, t)] += 1
    w = 0
    mn = 0
    for i in range(t):
        w += xi[i] - 1
        if w < mn:
            mn = w
    level = mn + np.random.randint(0, j)
    w = 0
    start = 0
    for i in range(t):
        w += xi[i] - 1
        if w == level:
            start = (i + 1) % t
            break
    parent = np.empty(n, np.int64)
    parent[0] = -1
    for i in range(j):
        parent[1 + i] = 0
    h = 1 + j
    for i in range(t):
        for _ in range(xi[(start + i) % t]):
            parent[h] = i + 1
            h += 1
    return parent


def csr_of_parent(parent):
    import scipy.sparse as sp
    n = len(parent)
    v = np.arange(1, n)
    u = parent[1:]
    A = sp.coo_array((np.ones(2 * (n - 1), np.int8), (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    return A.indptr.astype(np.int64), A.indices.astype(np.int64)


@nb.njit(cache=True)
def tree_height(parent, n):
    """Висота дерева від кореня."""
    depth = np.zeros(n, np.int64)
    H = 0
    for v in range(1, n):
        depth[v] = depth[parent[v]] + 1
        H = max(H, depth[v])
    return H


def node_sc_tree(parent, K=32):
    """(NodeSC усіх вузлів, суми δ_k за вузлами для k = 1…K; хвіст k > K — в останньому елементі).

    Для глибоких дерев, де стек переукорінення завеликий, переходить на звичайний BFS O(n²)."""
    parent = np.ascontiguousarray(parent, dtype=np.int64)
    n = len(parent)
    H = int(tree_height(parent, n)) if n > 0 else 0
    if (H + 1) * (2 * H + 3) * 8 > MAX_STACK_BYTES:
        return _node_sc_bfs(parent, K)
    acc = np.zeros(K)
    out = tree_sc_reroot(parent, n, gtable(n), K, acc)
    return out, acc


def _node_sc_bfs(parent, K):
    from nscfeat import bfs_profile  # окремо, щоб модуль не тягнув зайвого при звичайному шляху

    ip, ix = csr_of_parent(parent)
    _, _, D = bfs_profile(ip, ix, len(parent), K=K, full=True)
    return D.sum(1), D.sum(0)


def seeds_from(seq, k):
    """k цілих сідів для numba-коду з numpy.random.SeedSequence (стабільно щодо кількості процесів)."""
    return np.array([int(s.generate_state(1, np.uint32)[0]) for s in seq.spawn(k)], np.int64)
