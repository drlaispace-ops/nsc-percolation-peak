"""П1.3c. Bond-перколяція на нескінченних деревах клік (кактус-Бете) і NSC у границі n → ∞.

Структура: кожен вузол належить фіксованому набору клік `sizes` (розміри m ≥ 2; m = 2 — звичайне ребро);
граф клік — нескінченне дерево (кожна кліка і кожен вузол поза деревом — нові). Степінь z = Σ (m − 1),
трикутників на вузол t = Σ C(m − 1, 2). sizes = (2,)*z — гратка Бете; (3,)*6 — кактус Хусімі з z = 12;
(c,) + (2,)*(13 − c) — модель Gleeson2009 з однією клікою на вузол.

Точно (`branching`): π_m(j) — розподіл розміру компоненти вершини в G(m, p); h_m — імовірність, що кліка
розміру m, до якої вузол увійшов, не веде до нескінченності: h_m = Σ_j π_m(j)·H_m^{j−1}, H_m = Π_{m'∈sizes∖{m}} h_m';
P = 1 − Π_{m∈sizes} h_m. pc — спектральний радіус матриці середніх = 1.

Монте-Карло (`sample`): BFS від кореня на дереві клік, кліки (вузли і рівномірні ваги ребер) породжуються ліниво,
спільно для всієї сітки p (спряжені реалізації). Для кожного p — NodeSC = Σ_k δ_k, як у sampled._sources_sc
(b_k рахується за шарами; кінець — порожній шар). Якщо s_k > S_cap — зупинка і геометричний хвіст
δ_k·r/(1 − r), r = (s_{k−2}/s_k)^{1/2} (δ_k ∝ b_k/s_k² ∝ 1/s_k при експоненційному рості). Генератор
випадкових чисел — splitmix64 від (seed, номер кореня): результат не залежить від розкладу потоків.
"""
import math

import numba as nb
import numpy as np
from scipy.optimize import brentq
from scipy.special import comb


# --- Точні величини ---
def comp_pmf(m, p):
    """π_m(j), j = 1..m: імовірність, що компонента фіксованої вершини в G(m, p) має j вершин."""
    q = 1.0 - p
    conn = np.zeros(m + 1)          # conn[j] — імовірність, що G(j, p) зв'язний
    conn[1] = 1.0
    for j in range(2, m + 1):
        conn[j] = 1.0 - sum(comb(j - 1, i - 1) * conn[i] * q ** (i * (j - i)) for i in range(1, j))
    return np.array([comb(m - 1, j - 1) * conn[j] * q ** (j * (m - j)) for j in range(1, m + 1)])


def _others(sizes, m):
    s = list(sizes)
    s.remove(m)
    return s


def mean_matrix(sizes, p):
    types = sorted(set(sizes))
    M = np.zeros((len(types), len(types)))
    for a, m in enumerate(types):
        ej = float(np.dot(np.arange(m), comp_pmf(m, p)))          # E[j − 1]
        oth = _others(sizes, m)
        for b, m2 in enumerate(types):
            M[a, b] = ej * oth.count(m2)
    return M


def pc(sizes):
    return brentq(lambda p: max(abs(np.linalg.eigvals(mean_matrix(sizes, p)))) - 1.0, 1e-6, 1 - 1e-9, xtol=1e-14)


def branching(sizes, p, tol=1e-15, itmax=200000):
    """h_m для кожного типу і P (частка вузлів у нескінченному кластері)."""
    types = sorted(set(sizes))
    pm = {m: comp_pmf(m, p) for m in types}
    h = {m: 0.0 for m in types}
    for _ in range(itmax):
        new = {}
        for m in types:
            H = np.prod([h[x] for x in _others(sizes, m)]) if len(sizes) > 1 else 1.0
            new[m] = float(np.dot(pm[m], H ** np.arange(m)))
        if max(abs(new[m] - h[m]) for m in types) < tol:
            h = new
            break
        h = new
    P = 1.0 - float(np.prod([h[m] for m in sizes]))
    return h, P


def structure(sizes):
    return dict(z=int(sum(m - 1 for m in sizes)), t=int(sum(comb(m - 1, 2, exact=True) for m in sizes)),
                sizes=[int(m) for m in sizes])


# --- Монте-Карло на нескінченному дереві клік ---
@nb.njit(cache=True)
def _mix(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31)), x


@nb.njit(cache=True)
def _grow(a, need):
    if need <= len(a):
        return a
    b = np.empty(max(need, 2 * len(a)), a.dtype)
    b[:len(a)] = a
    return b


@nb.njit(cache=True)
def _one_root(sizes, ps, scap, state, MM):
    """NodeSC (сума), хвіст, ознака обрізання, s у кінці — для кожного p (ps за спаданням)."""
    L = len(sizes)
    NP = len(ps)
    cap0 = 1024
    # вузли
    nclq = np.full(cap0 * L, -1, np.int64)     # кліка в слоті
    npos = np.zeros(cap0 * L, np.int64)        # позиція вузла в кліці слоту
    pslot = np.full(cap0, -1, np.int64)        # слот батьківської кліки
    expd = np.zeros(cap0, np.bool_)
    stamp = np.full(cap0, -1, np.int64)
    dist = np.zeros(cap0, np.int64)
    nn = 1
    # кліки
    cmem = np.empty(cap0 * MM, np.int64)
    cw = np.empty(cap0 * MM * MM)
    csz = np.empty(cap0, np.int64)
    nc = 0
    q = np.empty(cap0, np.int64)
    out_sc = np.zeros(NP)
    out_tail = np.zeros(NP)
    out_cap = np.zeros(NP, np.bool_)
    out_s = np.zeros(NP)
    for ip in range(NP):
        p = ps[ip]
        run = ip
        stamp[0] = run
        dist[0] = 0
        q[0] = 0
        lo, hi = 0, 1
        s = 1
        b2 = 0
        c = 0.0
        k = 0
        s_hist1 = 1.0
        s_hist2 = 1.0
        dl = 0.0
        capped = False
        while True:
            k += 1
            top = hi
            for a in range(lo, hi):
                u = q[a]
                # розгорнути u (породити кліки), якщо треба
                if not expd[u]:
                    for sl in range(L):
                        if sl == pslot[u]:
                            continue
                        m = sizes[sl]
                        if nc + 1 > len(csz):
                            csz = _grow(csz, nc + 1)
                            cmem = _grow(cmem, (nc + 1) * MM)
                            cw = _grow(cw, (nc + 1) * MM * MM)
                        cid = nc
                        nc += 1
                        csz[cid] = m
                        need = nn + m - 1
                        if need > len(pslot):
                            newcap = max(need, 2 * len(pslot))
                            nclq = _grow(nclq, newcap * L)
                            npos = _grow(npos, newcap * L)
                            old = len(pslot)
                            pslot = _grow(pslot, newcap)
                            expd = _grow(expd, newcap)
                            stamp = _grow(stamp, newcap)
                            dist = _grow(dist, newcap)
                            q = _grow(q, newcap)
                            for x in range(old, len(pslot)):
                                expd[x] = False
                                stamp[x] = -1
                        cmem[cid * MM] = u
                        nclq[u * L + sl] = cid
                        npos[u * L + sl] = 0
                        for b in range(1, m):
                            v = nn
                            nn += 1
                            cmem[cid * MM + b] = v
                            # слот батьківської кліки у v — перший слот розміру m
                            for sl2 in range(L):
                                if sizes[sl2] == m:
                                    pslot[v] = sl2
                                    nclq[v * L + sl2] = cid
                                    npos[v * L + sl2] = b
                                    break
                            expd[v] = False
                            stamp[v] = -1
                        for a1 in range(m):
                            for a2 in range(a1 + 1, m):
                                r, state = _mix(state)
                                cw[cid * MM * MM + a1 * MM + a2] = (r >> np.uint64(11)) * (1.0 / 9007199254740992.0)
                    expd[u] = True
                for sl in range(L):
                    cid = nclq[u * L + sl]
                    m = csz[cid]
                    au = npos[u * L + sl]
                    for b in range(m):
                        if b == au:
                            continue
                        i1, i2 = (au, b) if au < b else (b, au)
                        if cw[cid * MM * MM + i1 * MM + i2] < p:
                            v = cmem[cid * MM + b]
                            if stamp[v] != run:
                                stamp[v] = run
                                dist[v] = k
                                if top >= len(q):
                                    q = _grow(q, top + 1)
                                q[top] = v
                                top += 1
            if top == hi:
                break
            # ребра нового шару (вузли шару ще треба розгорнути, щоб бачити їхні кліки)
            for a in range(hi, top):
                v = q[a]
                if not expd[v]:
                    # розгортаємо так само, як вище (дублювання коду — заради numba без замикань)
                    for sl in range(L):
                        if sl == pslot[v]:
                            continue
                        m = sizes[sl]
                        if nc + 1 > len(csz):
                            csz = _grow(csz, nc + 1)
                            cmem = _grow(cmem, (nc + 1) * MM)
                            cw = _grow(cw, (nc + 1) * MM * MM)
                        cid = nc
                        nc += 1
                        csz[cid] = m
                        need = nn + m - 1
                        if need > len(pslot):
                            newcap = max(need, 2 * len(pslot))
                            nclq = _grow(nclq, newcap * L)
                            npos = _grow(npos, newcap * L)
                            old = len(pslot)
                            pslot = _grow(pslot, newcap)
                            expd = _grow(expd, newcap)
                            stamp = _grow(stamp, newcap)
                            dist = _grow(dist, newcap)
                            q = _grow(q, newcap)
                            for x in range(old, len(pslot)):
                                expd[x] = False
                                stamp[x] = -1
                        cmem[cid * MM] = v
                        nclq[v * L + sl] = cid
                        npos[v * L + sl] = 0
                        for b in range(1, m):
                            x = nn
                            nn += 1
                            cmem[cid * MM + b] = x
                            for sl2 in range(L):
                                if sizes[sl2] == m:
                                    pslot[x] = sl2
                                    nclq[x * L + sl2] = cid
                                    npos[x * L + sl2] = b
                                    break
                            expd[x] = False
                            stamp[x] = -1
                        for a1 in range(m):
                            for a2 in range(a1 + 1, m):
                                r, state = _mix(state)
                                cw[cid * MM * MM + a1 * MM + a2] = (r >> np.uint64(11)) * (1.0 / 9007199254740992.0)
                    expd[v] = True
                for sl in range(L):
                    cid = nclq[v * L + sl]
                    m = csz[cid]
                    av = npos[v * L + sl]
                    for b in range(m):
                        if b == av:
                            continue
                        i1, i2 = (av, b) if av < b else (b, av)
                        if cw[cid * MM * MM + i1 * MM + i2] < p:
                            x = cmem[cid * MM + b]
                            if stamp[x] == run:
                                d = dist[x]
                                if d == k - 1:
                                    b2 += 2
                                elif d == k:
                                    b2 += 1
            s_hist2 = s_hist1
            s_hist1 = s
            s += top - hi
            pk = (b2 / 2.0) / (s * (s - 1) / 2.0)
            dl = pk * (1.0 - pk)
            c += dl
            lo, hi = hi, top
            if s > scap:
                capped = True
                break
        out_sc[ip] = c
        out_s[ip] = s
        out_cap[ip] = capped
        if capped:
            r = math.sqrt(s_hist2 / s)
            if r < 0.999:
                out_tail[ip] = dl * r / (1.0 - r)
            else:
                out_tail[ip] = np.nan
    return out_sc, out_tail, out_cap, out_s


@nb.njit(parallel=True, cache=True)
def _sample(sizes, ps, scap, seed, start, nroots, MM):
    NP = len(ps)
    sc = np.zeros((nroots, NP))
    tail = np.zeros((nroots, NP))
    cap = np.zeros((nroots, NP), np.bool_)
    ss = np.zeros((nroots, NP))
    for i in nb.prange(nroots):
        st = np.uint64(seed) * np.uint64(0x100000001B3) ^ np.uint64(start + i)
        _, st = _mix(st)
        a, b, c_, d = _one_root(sizes, ps, scap, st, MM)
        sc[i] = a
        tail[i] = b
        cap[i] = c_
        ss[i] = d
    return sc, tail, cap, ss


def sample(sizes, ps, nroots, seed, start=0, scap=10000):
    """Для коренів start…start+nroots−1: NodeSC без хвоста, хвіст, обрізано?, s у кінці; масиви [nroots, len(ps)].
    ps — будь-який порядок (усередині сортується за спаданням: найбільший кластер породжується першим)."""
    ps = np.asarray(ps, float)
    order = np.argsort(-ps)
    sizes = np.asarray(sorted(sizes, reverse=True), np.int64)
    sc, tail, cap, ss = _sample(sizes, ps[order], int(scap), int(seed), int(start), int(nroots), int(sizes.max()))
    inv = np.argsort(order)
    return sc[:, inv], tail[:, inv], cap[:, inv], ss[:, inv]
