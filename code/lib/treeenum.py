"""Перебір вільних дерев (Wright, Richmond, Odlyzko, McKay 1986; порт на numba) і закон форми.

Для пуассонівського дерева ГВ π(T; c) = n·e^{−cn}·c^{n−1}/|Aut T|; за умови розміру n частка форми
дорівнює (n!/|Aut T|)/n^{n−2} і від c не залежить. Перенесено з code/proto_gw/a4_enum.py.
"""
import numba as nb
import numpy as np

from gwtree import gtable


@nb.njit(cache=True)
def next_rooted(L, Lnew, n, p):
    if p < 0:
        p = n - 1
        while L[p] == 1:
            p -= 1
    if p == 0:
        return False
    q = p - 1
    while L[q] != L[p] - 1:
        q -= 1
    Lnew[:] = L
    for i in range(p, n):
        Lnew[i] = Lnew[i - p + q]
    return True


@nb.njit(cache=True)
def split_info(L, n):
    """Повертає m: початок другого піддерева кореня."""
    m = n
    found = False
    for i in range(n):
        if L[i] == 1:
            if found:
                m = i
                break
            found = True
    return m


@nb.njit(cache=True)
def lex_gt(L, m):
    """left > rest лексикографічно: left=[L[i]-1, i=1..m-1], rest=[0]+L[m:]; лише при рівних довжинах."""
    a = m - 1
    for i in range(a):
        x = L[1 + i] - 1
        y = 0 if i == 0 else L[m + i - 1]
        if x != y:
            return x > y
    return False


@nb.njit(cache=True)
def next_tree(L, Lnew, n):
    """Один крок WROM: повертає (ok), Lnew — канонічний кандидат."""
    m = split_info(L, n)
    lh = 0
    for i in range(1, m):
        if L[i] - 1 > lh:
            lh = L[i] - 1
    rh = 0
    for i in range(m, n):
        if L[i] > rh:
            rh = L[i]
    valid = rh >= lh
    if valid and rh == lh:
        ll = m - 1
        rl = 1 + n - m
        if ll > rl:
            valid = False
        elif ll == rl and lex_gt(L, m):
            valid = False
    if valid:
        Lnew[:] = L
        return True
    p = m - 1
    if not next_rooted(L, Lnew, n, p):
        return False
    if L[p] > 2:
        m2 = split_info(Lnew, n)
        nh = 0
        for i in range(1, m2):
            if Lnew[i] - 1 > nh:
                nh = Lnew[i] - 1
        ln = nh + 1
        for j in range(ln):
            Lnew[n - ln + j] = j + 1
    return True


@nb.njit(cache=True)
def mix(x):
    x = np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@nb.njit(cache=True)
def tree_eval(L, n, g, lf):
    """Для рівневої послідовності: (C = середнє NodeSC, log|Aut|)."""
    sc, _, la = tree_full(L, n, g, lf)
    return sc.sum() / n, la


@nb.njit(cache=True)
def tree_full(L, n, g, lf):
    """(NodeSC вузлів, степені, log|Aut|). Це O(n²) BFS від кожного вузла."""
    par = np.empty(n, np.int64)
    stack = np.empty(n + 1, np.int64)
    sp = 0
    for i in range(n):
        while sp > 0 and L[stack[sp - 1]] >= L[i]:
            sp -= 1
        par[i] = stack[sp - 1] if sp > 0 else -1
        stack[sp] = i
        sp += 1
    deg = np.zeros(n, np.int64)
    ip = np.zeros(n + 1, np.int64)
    for i in range(1, n):
        deg[i] += 1
        deg[par[i]] += 1
    for i in range(n):
        ip[i + 1] = ip[i] + deg[i]
    pos = ip[:-1].copy()
    ix = np.empty(2 * n, np.int64)
    for i in range(1, n):
        ix[pos[i]] = par[i]; pos[i] += 1
        ix[pos[par[i]]] = i; pos[par[i]] += 1
    # NSC
    sc = np.zeros(n)
    fr = np.empty(n, np.int64)
    nx_ = np.empty(n, np.int64)
    fp = np.empty(n, np.int64)
    nxp = np.empty(n, np.int64)
    for r in range(n):
        fr[0] = r; fp[0] = -1; nf = 1; s = 1
        while True:
            nn = 0
            for a in range(nf):
                u = fr[a]
                for e in range(ip[u], ip[u + 1]):
                    w = ix[e]
                    if w != fp[a]:
                        nx_[nn] = w; nxp[nn] = u; nn += 1
            if nn == 0:
                break
            s += nn
            sc[r] += g[s]
            for a in range(nn):
                fr[a] = nx_[a]; fp[a] = nxp[a]
            nf = nn
    # центр: обдирання листків (масиви замість списків: перебір викликає це десятки мільйонів разів)
    d2 = deg.copy()
    rem = n
    cur = np.empty(n, np.int64)
    nxt = np.empty(n, np.int64)
    nc = 0
    for i in range(n):
        if d2[i] <= 1:
            cur[nc] = i
            nc += 1
    while rem > 2:
        rem -= nc
        nn = 0
        for a in range(nc):
            u = cur[a]
            d2[u] = 0
            for e in range(ip[u], ip[u + 1]):
                w = ix[e]
                if d2[w] > 0:
                    d2[w] -= 1
                    if d2[w] == 1:
                        nxt[nn] = w
                        nn += 1
        cur, nxt = nxt, cur
        nc = nn
    c1 = cur[0]
    c2 = cur[1] if nc == 2 else -1
    # AHU-хеш + кратності: порядок BFS від центру (біцентр — віртуальний корінь над ребром)
    order = np.empty(n, np.int64)
    pa = np.full(n, -2, np.int64)
    head = 0; tail = 0
    order[tail] = c1; tail += 1; pa[c1] = -1
    if c2 >= 0:
        order[tail] = c2; tail += 1; pa[c2] = -1
    while head < tail:
        u = order[head]; head += 1
        for e in range(ip[u], ip[u + 1]):
            w = ix[e]
            if pa[w] == -2:
                pa[w] = u
                order[tail] = w; tail += 1
    H = np.zeros(n, np.uint64)
    logaut = 0.0
    buf = np.empty(n, np.uint64)
    for a in range(n - 1, -1, -1):
        u = order[a]
        k = 0
        for e in range(ip[u], ip[u + 1]):
            w = ix[e]
            if w != pa[u] and pa[w] == u:
                buf[k] = H[w]; k += 1
        ss = np.uint64(0)
        if k > 1:
            b = np.sort(buf[:k])
            run = 1
            for j in range(1, k):
                if b[j] == b[j - 1]:
                    run += 1
                else:
                    logaut += lf[run]
                    run = 1
            logaut += lf[run]
        for j in range(k):
            ss += mix(buf[j])
        H[u] = mix(ss + np.uint64(k))
    if c2 >= 0 and H[c1] == H[c2]:
        logaut += np.log(2.0)
    return sc, deg, logaut


@nb.njit(cache=True)
def enumerate_all(n, g, lf, want, store):
    """Усі вільні дерева порядку n≥3. want: 0 — лише генерація, 1 — обчислення. store: зберігати (C, logAut)."""
    h = n // 2
    L = np.zeros(n, np.int64)
    for i in range(h + 1):
        L[i] = i
    j = h + 1
    for i in range(1, (n + 1) // 2):
        L[j] = i; j += 1
    Lnew = L.copy()
    Ltmp = L.copy()
    count = 0
    sumw = 0.0
    suma = 0.0
    C_ = []
    A_ = []
    logn = lf[n]
    while True:
        if not next_tree(L, Lnew, n):
            break
        count += 1
        if want == 1:
            c, la = tree_eval(Lnew, n, g, lf)
            w = np.exp(logn - la - (n - 2) * np.log(n))  # частка міток n!/|Aut| / n^{n-2}
            sumw += w
            suma += w * c
            if store:
                C_.append(c); A_.append(la)
        if not next_rooted(Lnew, Ltmp, n, -1):
            break
        L[:] = Ltmp
    return count, sumw, suma, C_, A_


@nb.njit(cache=True)
def _neumaier(total, comp, x):
    """Компенсоване додавання: сотні тисяч доданків з округленням в один бік дають при наївній сумі ~10⁻¹² уже при n = 16."""
    t = total + x
    if abs(total) >= abs(x):
        comp += (total - t) + x
    else:
        comp += (x - t) + total
    return t, comp


@nb.njit(cache=True)
def _enumerate_law(n, g, lf, rw, cw):
    """Z = P(|T| = n) і ті самі суми, зважені NodeSC кореня та середнім за вузлами.
    P(форма T, корінь v) = rw[d_v]·Π_{u≠v} cw[d_u − 1]/|Aut T|, де rw[d] = P_кор(d)·d!, cw[c] = P_нащ(c)·c!."""
    h = n // 2
    L = np.zeros(n, np.int64)
    for i in range(h + 1):
        L[i] = i
    j = h + 1
    for i in range(1, (n + 1) // 2):
        L[j] = i
        j += 1
    Lnew = L.copy()
    Ltmp = L.copy()
    Z = aroot = atyp = cz = cr = ct = 0.0
    while True:
        if not next_tree(L, Lnew, n):
            break
        sc, deg, la = tree_full(Lnew, n, g, lf)
        C = sc.sum() / n
        aut = np.exp(-la)
        for v in range(n):
            x = rw[deg[v]] * aut
            for u in range(n):
                if u != v:
                    x *= cw[deg[u] - 1]
            Z, cz = _neumaier(Z, cz, x)
            aroot, cr = _neumaier(aroot, cr, x * sc[v])
            atyp, ct = _neumaier(atyp, ct, x * C)
        if not next_rooted(Lnew, Ltmp, n, -1):
            break
        L[:] = Ltmp
    return Z + cz, aroot + cr, atyp + ct


@nb.njit(cache=True)
def _stream(n, g, lf, rw, cw, nbins, cmax):
    """Потоковий закон C за формами для кількох законів (рядок rw, cw — закон): маса, моменти, гістограма; без збереження дерев.
    Вага форми T: Σ_v rw[d_v]·Π_{u≠v} cw[d_u−1]/|Aut T| = base·Σ_v rw[d_v]/cw[d_v−1]."""
    L_ = rw.shape[0]
    h = n // 2
    L = np.zeros(n, np.int64)
    for i in range(h + 1):
        L[i] = i
    j = h + 1
    for i in range(1, (n + 1) // 2):
        L[j] = i
        j += 1
    Lnew = L.copy()
    Ltmp = L.copy()
    hist = np.zeros((L_, nbins))
    mom = np.zeros((L_, 4))
    comp = np.zeros((L_, 4))
    cmin = np.full(L_, np.inf)
    cmaxv = np.full(L_, -np.inf)
    count = 0
    while True:
        if not next_tree(L, Lnew, n):
            break
        sc, deg, la = tree_full(Lnew, n, g, lf)
        C = sc.sum() / n
        count += 1
        aut = np.exp(-la)
        b = min(int(C / cmax * nbins), nbins - 1)
        for l in range(L_):
            base = aut
            for u in range(n):
                base *= cw[l, deg[u] - 1]
            if base == 0.0:
                continue
            x = 0.0
            for v in range(n):
                x += rw[l, deg[v]] / cw[l, deg[v] - 1]
            x *= base
            hist[l, b] += x
            cmin[l] = min(cmin[l], C)
            cmaxv[l] = max(cmaxv[l], C)
            y = x
            for m in range(4):
                mom[l, m], comp[l, m] = _neumaier(mom[l, m], comp[l, m], y)
                y *= C
        if not next_rooted(Lnew, Ltmp, n, -1):
            break
        L[:] = Ltmp
    return mom + comp, hist, cmin, cmaxv, count


def shape_stream(n, laws, nbins=4000, cmax=4.0):
    """Точний P(C | n) для списку законів [(pmf кореня, pmf нащадка)]. Повертає для кожного закону масу, C̄, σ, асиметрію, min, max, гістограму."""
    g = gtable(n + 2)
    lf = np.r_[0.0, np.cumsum(np.log(np.arange(1, n + 3, dtype=float)))]
    fac = np.exp(lf[: n + 1])
    rw = np.array([np.asarray(r[: n + 1]) * fac for r, _ in laws])
    cw = np.array([np.asarray(c[: n + 1]) * fac for _, c in laws])
    mom, hist, lo, hi, count = _stream(n, g, lf, rw, cw, nbins, cmax)
    out = []
    for l in range(len(laws)):
        z, m1, m2, m3 = mom[l, 0], mom[l, 1] / mom[l, 0], mom[l, 2] / mom[l, 0], mom[l, 3] / mom[l, 0]
        var = m2 - m1 ** 2
        out.append(dict(mass=float(z), mean=float(m1), sd=float(np.sqrt(var)), skew=float((m3 - 3 * m1 * var - m1 ** 3) / var ** 1.5),
                        cmin=float(lo[l]), cmax=float(hi[l]), hist=hist[l] / z))
    return out, int(count)


def law_moments(n, root_pmf, child_pmf):
    """Перебір для довільного закону: (P(|T| = n), a_n кореня, a_n рівномірного вузла). pmf задано на 0…n."""
    g = gtable(n + 2)
    lf = np.r_[0.0, np.cumsum(np.log(np.arange(1, n + 3, dtype=float)))]
    fac = np.exp(lf[: n + 1])
    Z, ar, at = _enumerate_law(n, g, lf, np.asarray(root_pmf[: n + 1]) * fac, np.asarray(child_pmf[: n + 1]) * fac)
    return float(Z), float(ar / Z), float(at / Z)


def trees(n):
    """Усі вільні дерева порядку n ≥ 3: (C — середня NodeSC, частка форми за умови розміру n)."""
    g = gtable(n + 2)
    lf = np.r_[0.0, np.cumsum(np.log(np.arange(1, n + 3, dtype=float)))]
    cnt, _, _, C, A = enumerate_all(n, g, lf, 1, True)
    C, A = np.array(C), np.array(A)
    w = np.exp(lf[n] - A - (n - 2) * np.log(n))
    assert len(C) == cnt
    return C, w


def count_trees(n):
    g = gtable(n + 2)
    lf = np.r_[0.0, np.cumsum(np.log(np.arange(1, n + 3, dtype=float)))]
    return int(enumerate_all(n, g, lf, 0, False)[0])
