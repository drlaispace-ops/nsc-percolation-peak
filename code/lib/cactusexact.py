"""П1.3c. Точна NSC_∞ (n → ∞) для bond-перколяції на нескінченному дереві клік — узагальнення gwexact на кактуси.
Вивід — notes/P1_cactus.tex. Як і gwexact, спирається на перестановку суми за k і середнього (лема про хвіст, П1.16).

Позначення: вузол належить клікам розмірів `sizes`; s_k, b_k — вузли і ребра кулі радіуса k у кластері кореня;
δ_k = p_k(1 − p_k), p_k = 2b_k/(s_k(s_k − 1)); NodeSC = Σ_k δ_k·1[шар k непорожній] (як у sampled._sources_sc).

1. Кліка розміру m, у яку увійшли через вузол 0: BFS-шари членів від 0 у G(m, p) (найкоротший шлях між членами
   кліки лежить у кліці: граф клік — дерево) і число e_r відкритих ребер між членами з d ≤ r — комбінаторно
   (`_layers`); для m ≤ 7 збігається з повним перебором графів (`_rtables_enum`). Ребро кліки входить у кулю,
   якщо обидва кінці на відстані ≤ r = k − (відстань входу).
2. Твірні функції за залишковим радіусом r (x — ребра, y — вузли, z — вузли на межі r = 0):
     Ψ^{(m)}_0 = y·z,   Ψ^{(m)}_r = y·Π_{c ∈ sizes∖{m}} Φ^{(c)}_r,
     Φ^{(m)}_r = Σ_графи P(граф)·x^{e_r}·Π_{v: d_v ≤ r} Ψ^{(m)}_{r − d_v},   корінь R_k = y·Π_{c ∈ sizes} Φ^{(c)}_k.
   z = 1 — без умови; z = 0 — шар k порожній. Різниця дає E[·; шар k непорожній].
3. Похідні за x при x = 1 (струмені (f, f', f'')): E[b·y^s], E[b(b − 1)·y^s]. Лапласові подання (s ≥ 2; при s = 1 b = 0):
     2/(s(s−1)) = 2∫(e^t − 1)e^{−ts}dt,   1/(s²(s−1)²) = ∫[t e^t − 2(e^t − 1) + t] e^{−ts}dt,
   тож E[δ_k; шар k непорожній] = ∫dt {2(e^t − 1)ΔB − 4K₂(t)(ΔC + ΔB)}, y = e^{−t}, сітка за u = ln t.
"""
import math

import numba as nb
import numpy as np
from scipy.optimize import brentq
from scipy.special import comb

_TABLES = {}


@nb.njit(cache=True)
def _enum(m):
    """Для кожного графа на m вершинах: число ребер, n_d (d = 1..m−1) і e_r (r = 1..m−1)."""
    E = m * (m - 1) // 2
    ea = np.empty(E, np.int64)
    eb = np.empty(E, np.int64)
    c = 0
    for a in range(m):
        for b in range(a + 1, m):
            ea[c] = a
            eb[c] = b
            c += 1
    G = 1 << E
    etot = np.empty(G, np.int64)
    nd = np.zeros((G, m), np.int64)
    er = np.zeros((G, m), np.int64)
    adj = np.zeros((m, m), np.bool_)
    dist = np.empty(m, np.int64)
    q = np.empty(m, np.int64)
    for g in range(G):
        adj[:, :] = False
        ne = 0
        for i in range(E):
            if (g >> i) & 1:
                adj[ea[i], eb[i]] = True
                adj[eb[i], ea[i]] = True
                ne += 1
        etot[g] = ne
        dist[:] = -1
        dist[0] = 0
        q[0] = 0
        lo, hi = 0, 1
        while lo < hi:
            u = q[lo]
            lo += 1
            for v in range(m):
                if adj[u, v] and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q[hi] = v
                    hi += 1
        for v in range(1, m):
            if dist[v] > 0:
                nd[g, dist[v]] += 1
        for r in range(1, m):
            cnt = 0
            for i in range(E):
                if (g >> i) & 1 and 0 <= dist[ea[i]] <= r and 0 <= dist[eb[i]] <= r:
                    cnt += 1
            er[g, r] = cnt
    return etot, nd, er


def _table(m):
    """Унікальні рядки (n_1..n_{m−1}, e_1..e_{m−1}) і кількість графів з кожним числом ребер."""
    if m not in _TABLES:
        etot, nd, er = _enum(m)
        key = np.concatenate([nd[:, 1:], er[:, 1:]], axis=1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        E = m * (m - 1) // 2
        cnt = np.zeros((len(uniq), E + 1))
        np.add.at(cnt, (inv.ravel(), etot), 1.0)
        _TABLES[m] = (uniq[:, :m - 1].astype(np.int64), uniq[:, m - 1:].astype(np.int64), cnt)
    return _TABLES[m]


def _compositions(total):
    """Усі послідовності додатних цілих із сумою ≤ total (включно з порожньою)."""
    out = [()]
    for first in range(1, total + 1):
        out += [(first,) + rest for rest in _compositions(total - first)]
    return out


def _layers(m, p):
    """Шари BFS від вершини 0 у G(m, p) комбінаторно: для кожного набору розмірів шарів (n_1..n_D)
    імовірність і для r = 1..m−1 перші два факторіальні моменти e_r (ребра між вершинами з d ≤ r).
    P = (m−1)!/(Π n_i!·u!)·Π_i (1 − q^{n_i})^{n_{i+1}}·q^{n_{i+1}·Σ_{j<i} n_j}·q^{u·(1 + Σ n)};
    ребра всередині шару ~ Bin(C(n_i, 2), p), між шарами i, i+1 — сума n_{i+1} усічених Bin(n_i, p | ≥ 1)."""
    q = 1.0 - p
    rows = []
    for comp in _compositions(m - 1):
        n = (1,) + comp
        D = len(comp)
        u = m - 1 - sum(comp)
        lw = math.lgamma(m) - sum(math.lgamma(x + 1) for x in comp) - math.lgamma(u + 1)
        pr = math.exp(lw)
        cum = 1
        for i in range(D):
            pr *= (1 - q ** n[i]) ** n[i + 1] * q ** (n[i + 1] * (cum - n[i]))
            cum += n[i + 1]
        pr *= q ** (u * cum)
        if pr == 0.0:
            continue
        mean = np.zeros(m)
        var = np.zeros(m)
        for r in range(1, m):
            Lr = min(r, D)
            mu = v = 0.0
            for i in range(1, Lr + 1):                 # усередині шарів 1..Lr
                c = n[i] * (n[i] - 1) / 2
                mu += c * p
                v += c * p * q
            for i in range(0, Lr):                     # між шарами i, i+1
                ni, cnt = n[i], n[i + 1]
                z = 1 - q ** ni
                ey = ni * p / z
                ey2 = (ni * p * q + (ni * p) ** 2) / z
                mu += cnt * ey
                v += cnt * (ey2 - ey * ey)
            mean[r] = mu
            var[r] = v
        rows.append((comp, pr, mean, var))
    return rows


def _rtables(m, p, mm):
    """Для r = 1..m−1: рядки (n_1..n_r, 0…) і струмені Σ P·(1, E e_r, E e_r(e_r − 1)), згруповані за n_{≤r}.
    Масиви вирівняно до mm − 1 стовпчиків (mm — найбільший розмір кліки)."""
    lay = _layers(m, p)
    rows, jets, start = [], [], [0]
    for r in range(1, m):
        acc = {}
        for comp, pr, mean, var in lay:
            key = comp[:r]
            e1 = mean[r]
            e2 = var[r] + e1 * e1 - e1
            J = acc.setdefault(key, np.zeros(3))
            J += (pr, pr * e1, pr * e2)
        for key, J in acc.items():
            pad = np.zeros(mm - 1, np.int64)
            pad[:len(key)] = key
            rows.append(pad)
            jets.append(J)
        start.append(start[-1] + len(acc))
    return np.array(rows, np.int64), np.array(jets), np.array(start, np.int64)


def _rtables_enum(m, p, mm):
    """Те саме повним перебором 2^{C(m,2)} графів (перевірка _rtables для m ≤ 7)."""
    nd, er, cnt = _table(m)
    E = m * (m - 1) // 2
    e = np.arange(E + 1)
    w = cnt @ (p ** e * (1 - p) ** (E - e))
    rows, jets, start = [], [], [0]
    for r in range(1, m):
        trunc = nd[:, :r]
        u, inv = np.unique(trunc, axis=0, return_inverse=True)
        inv = inv.ravel()
        J = np.zeros((len(u), 3))
        np.add.at(J, inv, np.stack([w, w * er[:, r - 1], w * er[:, r - 1] * (er[:, r - 1] - 1)], axis=1))
        pad = np.zeros((len(u), mm - 1), np.int64)
        pad[:, :r] = u
        rows.append(pad)
        jets.append(J)
        start.append(start[-1] + len(u))
    return np.concatenate(rows), np.concatenate(jets), np.array(start, np.int64)


@nb.njit(cache=True)
def _jmul(a0, a1, a2, b0, b1, b2):
    return a0 * b0, a1 * b0 + a0 * b1, a2 * b0 + 2 * a1 * b1 + a0 * b2


@nb.njit(cache=True)
def _jpow(f0, f1, f2, n):
    if n == 0:
        return 1.0, 0.0, 0.0
    pn1 = f0 ** (n - 1)
    pn2 = f0 ** (n - 2) if n >= 2 else 0.0
    return pn1 * f0, n * pn1 * f1, n * (n - 1) * pn2 * f1 * f1 + n * pn1 * f2


@nb.njit(parallel=True, cache=True)
def _profile(types, mult, rows_all, jets_all, starts, roff, t, wB, wC, K, kmax, tol):
    """Профіль Σ_t ваги·Δструменів за k. types — розміри типів, mult[i] — кратність типу i в sizes;
    rows_all/jets_all/starts — таблиці всіх типів; roff[i] — зсув блоку starts типу i.
    Вузли t незалежні (паралельно); для кожного k іде до власної збіжності.
    Значення твірних функцій зберігаються як g = 1 − f (log1p/expm1), бо при малих t f = 1 − O(t·s)
    і e^{−t} у подвійній точності дорівнює 1 (як у gwexact). Похідні за x — звичайні числа."""
    T = len(t)
    nt = len(types)
    mm = 0
    for i in range(nt):
        if types[i] > mm:
            mm = types[i]
    H = mm                                 # глибина історії Ψ: потрібні r − 1 … r − (m − 1)
    profT = np.zeros((T, K))
    last = np.zeros(T, np.int64)
    for j in nb.prange(T):
        gy = -np.expm1(-t[j])
        y0 = 1.0 - gy
        # історія Ψ^{(тип)}_{r'} для z = 1 і z = 0: [z, тип, r' mod H, (g, f', f'')]
        psi = np.zeros((2, nt, H, 3))
        phi = np.zeros((2, nt, 3))
        for i in range(nt):
            psi[0, i, 0, 0] = gy            # Ψ_0 = y·z: z = 1 → g = 1 − y; z = 0 → g = 1
            psi[1, i, 0, 0] = 1.0
        k = 0
        while k < kmax:
            k += 1
            for zz in range(2):
                for i in range(nt):
                    m = types[i]
                    rr = k if k < m - 1 else m - 1
                    b0 = starts[roff[i] + rr - 1]
                    b1 = starts[roff[i] + rr]
                    s0 = 0.0
                    s1 = 0.0
                    s2 = 0.0
                    for row in range(b0, b1):
                        W0 = jets_all[row, 0]
                        if W0 == 0.0 and jets_all[row, 1] == 0.0 and jets_all[row, 2] == 0.0:
                            continue
                        lg = 0.0
                        c0, c1, c2 = 1.0, 0.0, 0.0
                        for d in range(1, rr + 1):
                            n = rows_all[row, d - 1]
                            if n == 0:
                                continue
                            src = (k - d) % H
                            g = psi[zz, i, src, 0]
                            lg += n * np.log1p(-g)
                            e0, e1, e2 = _jpow(1.0 - g, psi[zz, i, src, 1], psi[zz, i, src, 2], n)
                            c0, c1, c2 = _jmul(c0, c1, c2, e0, e1, e2)
                        q0, q1, q2 = _jmul(W0, jets_all[row, 1], jets_all[row, 2], c0, c1, c2)
                        s0 += W0 * (-np.expm1(lg))      # Σ W·(1 − ΠΨ^n); Σ W = 1 точно
                        s1 += q1
                        s2 += q2
                    phi[zz, i, 0] = s0
                    phi[zz, i, 1] = s1
                    phi[zz, i, 2] = s2
            # корінь і нові Ψ_k
            dB = 0.0
            dC = 0.0
            for zz in range(2):
                r0, r1, r2 = y0, 0.0, 0.0
                for i in range(nt):
                    for _ in range(mult[i]):
                        r0, r1, r2 = _jmul(r0, r1, r2, 1.0 - phi[zz, i, 0], phi[zz, i, 1], phi[zz, i, 2])
                sg = 1.0 if zz == 0 else -1.0
                dB += sg * r1
                dC += sg * r2
                for i in range(nt):
                    lg = -t[j]
                    p0, p1, p2 = y0, 0.0, 0.0
                    for i2 in range(nt):
                        cnt = mult[i2] - (1 if i2 == i else 0)
                        for _ in range(cnt):
                            lg += np.log1p(-phi[zz, i2, 0])
                            p0, p1, p2 = _jmul(p0, p1, p2, 1.0 - phi[zz, i2, 0], phi[zz, i2, 1], phi[zz, i2, 2])
                    dst = k % H
                    psi[zz, i, dst, 0] = -np.expm1(lg)
                    psi[zz, i, dst, 1] = p1
                    psi[zz, i, dst, 2] = p2
            contrib = wB[j] * dB - wC[j] * (dC + dB)
            profT[j, min(k, K) - 1] += contrib
            # зупинка, коли внесок < tol, а dB, dC — на рівні округлення різниці z = 1 і z = 0
            # (dB < 1e-8: у надкритичному режимі на малих t dB ~ P·k не мале, і точка лишається)
            if k > mm and abs(contrib) < tol and abs(dB) < 1e-8 and abs(dC) < 1e-6:
                break
        last[j] = k
    prof = np.zeros(K)
    for j in range(T):
        for kk in range(K):
            prof[kk] += profT[j, kk]
    kmaxed = 0
    for j in range(T):
        if last[j] >= kmax:
            kmaxed = -1
    return prof, kmaxed


def _grid(h=0.01, ulo=-40.0, uhi=np.log(60.0)):
    u = np.arange(ulo, uhi + h, h)
    t = np.exp(u)
    k2 = np.where(t < 1e-3, t ** 3 / 6 + t ** 4 / 12 + t ** 5 / 40, t * np.exp(t) + t - 2 * np.expm1(t))
    return t, h * t * 2 * np.expm1(t), h * t * 4 * k2


def profile(sizes, p, K=64, h=0.01, kmax=200000, tol=1e-16):
    """E[δ_k; шар k непорожній], k = 1..K (хвіст k > K — в останньому елементі); сума — NSC_∞."""
    types = sorted(set(sizes))
    mm = max(types)
    rows, jets, starts, off = [], [], [], 0
    for m in types:
        r_, j_, s_ = _rtables(m, p, mm)
        rows.append(r_)
        jets.append(j_)
        starts.append(s_ + off)
        off += len(r_)
    roff = np.cumsum([0] + [len(s) for s in starts[:-1]]).astype(np.int64)
    t, wB, wC = _grid(h)
    prof, k = _profile(np.array(types, np.int64), np.array([list(sizes).count(m) for m in types], np.int64),
                       np.concatenate(rows), np.concatenate(jets), np.concatenate(starts), roff, t, wB, wC,
                       K, kmax, tol)
    if k < 0:   # хоч один вузол t не збігся за kmax кроків
        raise RuntimeError(f"немає збіжності за k: sizes={sizes}, p={p}")
    return prof


def nsc(sizes, p, h=0.01):
    return float(profile(sizes, p, K=1, h=h).sum())


def peak(sizes, lo, hi, step=1e-4, h=0.01):
    """Нуль центральної різниці NSC_∞(p) на [lo, hi]."""
    return brentq(lambda x: (nsc(sizes, x + step, h) - nsc(sizes, x - step, h)) / (2 * step), lo, hi, xtol=1e-7)
