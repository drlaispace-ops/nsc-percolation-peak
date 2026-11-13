"""Точні величини для NSC на деревах ГВ у границі n → ∞. Вивід — notes/P1_gw_identities.tex.

Формула NSC_∞ спирається на недоведену лему про хвіст за k (П1.16); кореневий закон — унімодулярний.
Позначення: B(n; c) — закон Бореля; c′ — двоїстий параметр; a_n — середня NodeSC за розміру n.
"""
import numba as nb
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, lambertw

from gwtree import BINOM, GEOM, POISSON


# --- Двоїстість ER ---
def _q_rrg(z, p):
    """q = 1 − w, де w — імовірність виродження гілки RRG: w = (1 − p + p·w)^{z−1}; 0 не вище порога."""
    if (z - 1) * p <= 1:
        return 0.0
    return brentq(lambda q: 1 + np.expm1((z - 1) * np.log1p(-p * q)) / q, 1e-12, 1.0, xtol=1e-15)


def giant(x, model="ER"):
    """Частка гіганта: ER з середнім степенем x або RRG ("z3", "z4") з bond-імовірністю x."""
    if model != "ER":
        z = int(model[1:])
        return float(-np.expm1(z * np.log1p(-x * _q_rrg(z, x))))
    if x <= 1:
        return 0.0
    return float(1 + lambertw(-x * np.exp(-x), 0).real / x)


def dual(x, model="ER"):
    """Двоїстий докритичний параметр: закон за умови виродження — той самий закон при x′.
    ER: Poisson(c) → Poisson(c′). RRG: Bin(m, p) → Bin(m, p′), p′ = p·w/(1 − p + p·w), для кореня й нащадків однаково."""
    if model != "ER":
        if x >= 1:
            raise ValueError("p = 1: гілка не вироджується, двоїстого закону немає")
        z = int(model[1:])
        q = _q_rrg(z, x)
        return float(x) if q == 0 else float(x * (1 - q) / (1 - x * q))
    return float(x) if x <= 1 else float(-lambertw(-x * np.exp(-x), 0).real)


def borel(N, c):
    """B(n; c) для n = 0…N; B(0) = 0."""
    n = np.arange(1, N + 1, dtype=float)
    return np.r_[0.0, np.exp(-c * n + (n - 1) * np.log(c * n) - gammaln(n + 1))]


# --- Квадратура ---
def _logg(model, x):
    """(ln g_root, ln g_child) як функції q = 1 − s."""
    if model == "ER":
        f = lambda q: -x * q
        return f, f
    z = int(model[1:])  # "z3": RRG, bond-перколяція з імовірністю x
    return (lambda q: z * np.log1p(-x * q)), (lambda q: (z - 1) * np.log1p(-x * q))


def mean_child(model, x):
    return x if model == "ER" else (int(model[1:]) - 1) * x


def quad_profile(model, x, K=64, h=0.01):
    """E[δ_k; Z_k>0] для k = 1…K, хвіст k > K — в останньому елементі; сума дає NSC_∞."""
    critical = abs(mean_child(model, x) - 1) < 1e-3
    u = np.arange(-18.0 if critical else -40.0, np.log(60.0) + h, h)
    t = np.exp(u)
    kmax = 30000 if critical else 2_000_000
    lroot, lchild = _logg(model, x)
    prof = np.zeros(K)
    ti, weight = t, h * (2 - 4 * t) * t
    qa, qb = -np.expm1(-ti), np.ones_like(ti)
    for k in range(kmax):
        d = np.expm1(-ti + lroot(qa)) - np.expm1(-ti + lroot(qb))
        prof[min(k, K - 1)] += float(np.sum(weight * d))
        keep = np.abs(d) >= 1e-13  # A_k спадає за k; нижче 1e-13 лишається округлення
        if not keep.any():
            return prof
        ti, weight, qa, qb = ti[keep], weight[keep], qa[keep], qb[keep]
        qa, qb = -np.expm1(-ti + lchild(qa)), -np.expm1(-ti + lchild(qb))
    if critical:
        return prof  # рівно на порозі: похибка до ~1e-6
    raise RuntimeError(f"немає збіжності за k: {model}, x={x}")


def nsc_quad(model, x, h=0.01):
    return float(quad_profile(model, x, K=1, h=h).sum())


def peak(model, lo, hi, step=1e-4):
    """Нуль центральної різниці NSC_∞(x) на відрізку [lo, hi]."""
    return brentq(lambda x: (nsc_quad(model, x + step) - nsc_quad(model, x - step)) / (2 * step), lo, hi, xtol=1e-7)


# --- Квадратура пари початкових умов: точний внесок гіганта без різниці двох NSC ---
@nb.njit(cache=True)
def _pair_sum(t, z, x, lw, kmax):
    """Σ_{k≥1} [L_k(t,1) − L_k(t,w)], ln w = lw. z = 0: ER з середнім x; інакше RRG із bond-імовірністю x.
    w = 0 (lw = −inf) дає A_k, сума — NSC; w за виродженням дає внесок гіганта: 1 − w^{Z_k} — імовірність вижити.
    Стан (q_a, δ), q_b = q_a + δ: q_a мала при малих t, δ мала наприкінці, і обидві мають лишатися відносно точними.
    Якщо рахувати q_b і δ, то q_a ~ 10⁻¹⁸ губиться, стає нулем, а нуль вище порога — нестійка нерухома точка: цикл не закінчується."""
    qa = -np.expm1(-t)
    qb0 = 1.0 if lw == -np.inf else -np.expm1(-t + lw)
    dq = qb0 - qa
    tot = 0.0
    for _ in range(kmax):
        if z == 0:
            ca = -x * qa
            dc = x * dq
            ra, dr = ca, dc
        else:
            la = np.log1p(-x * qa)
            ld = np.log1p(-x * dq / (1 - x * qa))  # ln[(1 − x·q_b)/(1 − x·q_a)] < 0
            ca, dc = (z - 1) * la, -(z - 1) * ld
            ra, dr = z * la, -z * ld
        # L(1) − L(w) = e^{−t+r_a} − e^{−t+r_b} = e^{−t+r_a}(1 − e^{−dr})
        d = -np.exp(-t + ra) * np.expm1(-dr)
        tot += d
        if d < 1e-17:
            break
        dq = -np.exp(-t + ca) * np.expm1(-dc)
        qa = -np.expm1(-t + ca)
    return tot


@nb.njit(parallel=True, cache=True)
def _pair_grid(t, z, x, lw, kmax):
    out = np.zeros(t.size)
    for i in nb.prange(t.size):
        out[i] = _pair_sum(t[i], z, x, lw, kmax)
    return out


def pair_quad(model, x, survive=False, ulo=-40.0, h=0.01):
    """∫(2−4t)Σ_k[...]dt у лог-сітці. survive=False — NSC(x); True — внесок гіганта P·g_∞, без віднімання.
    Нижче t₀ частина береться за асимптотикою Σ ~ C·t^{−1/2}; вона помітна лише рівно на порозі."""
    z = 0 if model == "ER" else int(model[1:])
    if survive:
        q = giant(x) if model == "ER" else _q_rrg(z, x)
        lw = np.log1p(-q)
    else:
        lw = -np.inf
    if abs(mean_child(model, x) - 1) < 1e-12:
        ulo = max(ulo, -24.0)
    t = np.exp(np.arange(ulo, np.log(60.0) + h, h))
    s = _pair_grid(t, z, float(x), float(lw), 2_000_000_000)
    f = (2 - 4 * t) * s * t
    return float(h * (f.sum() - 0.5 * f[0] - 0.5 * f[-1]) + 4 * s[0] * t[0])


# --- Розклад ER ---
def decompose(x, model="ER"):
    """NSC, P, (1−P)·g_fin, P·g_∞, g_fin, g_∞ без Монте-Карло: g_fin(x) = NSC(x′) вище порога."""
    P = giant(x, model)
    nsc = nsc_quad(model, x)
    gf = nsc_quad(model, dual(x, model)) if P > 0 else nsc
    fin = (1 - P) * gf
    return dict(c=float(x), nsc=nsc, P=P, part_fin=fin, part_inf=nsc - fin, g_fin=gf,
                g_inf=(nsc - fin) / P if P > 0 else float("nan"))


# --- Таблиця a_n ---
@nb.njit(cache=True)
def _forward(N, eps):
    """F(s, z) = Σ_k P(S_k = s, Z_k = z) для Poisson(1)."""
    F = np.zeros((N + 1, N + 1))
    lf = np.zeros(N + 2)
    lg = np.zeros(N + 2)
    for i in range(1, N + 2):
        lg[i] = np.log(i)
        lf[i] = lf[i - 1] + lg[i]
    F[1, 1] = 1.0
    for s2 in range(2, N + 1):
        for z2 in range(1, s2):
            s1 = s2 - z2
            x = 0.0
            for z in range(1, s1 + 1):
                f = F[s1, z]
                if f > eps:
                    x += f * np.exp(z2 * lg[z] - z - lf[z2])
            F[s2, z2] = x
    return F, lf, lg


@nb.njit(parallel=True, cache=True)
def _a_num(F, lf, lg, N, eps):
    """B(n; 1)·a_n: формула Оттера–Двасса для лісу з z дерев дає m = n − s вузлів нащадків."""
    out = np.zeros(N + 1)
    for n in nb.prange(3, N + 1):
        x = 0.0
        for s in range(3, n + 1):
            gs = 2.0 * (s - 2) / (s * s)
            for z in range(1, s):
                f = F[s, z]
                if f > eps:
                    m = n - s
                    t = m + z
                    x += f * gs * (z / t) * np.exp(m * lg[t] - t - lf[m])
        out[n] = x
    return out


def a_table(N, eps=1e-30):
    """a_n, n = 0…N: E[NodeSC кореня | |T| = n] для пуассонівського ГВ; від c не залежить."""
    F, lf, lg = _forward(N, eps)
    num = _a_num(F, lf, lg, N, eps)
    b = borel(N, 1.0)
    a = np.zeros(N + 1)
    a[1:] = num[1:] / b[1:]
    return a


def nsc_from_table(a, c, ainf=None):
    """NSC(c) = Σ B(n; c)·a_n при c ≤ 1 (c′ при c > 1 дає g_fin); хвіст n > N оцінюється через ainf."""
    N = len(a) - 1
    b = borel(N, dual(c))
    head = float(np.sum(b * a))
    tail = max(0.0, 1.0 - float(b.sum()))
    return head + tail * (a[-1] if ainf is None else ainf), tail


# --- Довільні закони: закон — кортеж (mode, a, b) у позначеннях gwtree ---
def rrg_laws(z, p=None):
    """(закон кореня, закон нащадка) для bond-перколяції на RRG; за замовчуванням критична p."""
    p = 1 / (z - 1) if p is None else p
    return (BINOM, p, z), (BINOM, p, z - 1)


def law_pmf(law, N):
    """P(ξ = j), j = 0…N."""
    mode, a, b = law
    j = np.arange(N + 1, dtype=float)
    if mode == POISSON:
        return np.exp(j * np.log(a) - a - gammaln(j + 1))
    if mode == GEOM:
        return (1 - a) * a ** j
    out = np.zeros(N + 1)
    m = min(N, int(b))
    jj = j[: m + 1]
    out[: m + 1] = np.exp(gammaln(b + 1) - gammaln(jj + 1) - gammaln(b - jj + 1) + jj * np.log(a) + (b - jj) * np.log1p(-a))
    return out


@nb.njit(cache=True)
def _lsum(mode, a, b, t, m, lf, lg, la, l1a):
    """ln P(ξ_1 + … + ξ_t = m), t ≥ 1."""
    if mode == 0:
        return m * (lg[t] + la) - t * a - lf[m]
    if mode == 1:
        return lf[t + m - 1] - lf[m] - lf[t - 1] + t * l1a + m * la
    T = t * b
    if m > T:
        return -np.inf
    return lf[T] - lf[m] - lf[T - m] + m * la + (T - m) * l1a


def _tables(N, child):
    mode, a, b = child
    size = 2 * N + 3 if mode == GEOM else (N * int(b) + 3 if mode == BINOM else N + 3)
    lg = np.zeros(size)
    lg[1:] = np.log(np.arange(1, size, dtype=float))
    return np.cumsum(lg), lg, float(np.log(a)), float(np.log1p(-a)) if mode != POISSON else 0.0


@nb.njit(parallel=True, cache=True)
def _forward_law(N, rootp, mode, a, b, lf, lg, la, l1a, eps):
    """F(s, z) = Σ_{k≥1} P(S_k = s, Z_k = z); перше покоління — за законом кореня."""
    F = np.zeros((N + 1, N + 1))
    for j in range(1, N):
        F[1 + j, j] = rootp[j]
    for s2 in range(3, N + 1):
        for z2 in nb.prange(1, s2 - 1):
            s1 = s2 - z2
            x = 0.0
            for z in range(1, s1):
                f = F[s1, z]
                if f > eps:
                    x += f * np.exp(_lsum(mode, a, b, z, z2, lf, lg, la, l1a))
            F[s2, z2] += x
    return F


@nb.njit(parallel=True, cache=True)
def _num_law(F, N, mode, a, b, lf, lg, la, l1a, eps):
    """P(|T| = n)·a_n: ліс із z дерев (Двасс) дає решту m = n − s вузлів."""
    out = np.zeros(N + 1)
    for n in nb.prange(3, N + 1):
        x = 0.0
        for s in range(3, n + 1):
            gs = 2.0 * (s - 2) / (s * s)
            for z in range(1, s):
                f = F[s, z]
                if f > eps:
                    m = n - s
                    t = m + z
                    x += f * gs * (z / t) * np.exp(_lsum(mode, a, b, t, m, lf, lg, la, l1a))
        out[n] = x
    return out


def size_law(N, root, child):
    """P(|T| = n), n = 0…N, за формулою Двасса: корінь має j дітей, ліс із j дерев має n − 1 вузлів."""
    mode, a, b = child
    lf, lg, la, l1a = _tables(N, child)
    rp = law_pmf(root, N)
    out = np.zeros(N + 1)
    out[1] = rp[0]
    for n in range(2, N + 1):
        t = n - 1
        x = 0.0
        for j in range(1, n):
            if rp[j] > 0:
                x += rp[j] * (j / t) * np.exp(_lsum(mode, a, int(b), t, t - j, lf, lg, la, l1a))
        out[n] = x
    return out


def root_given_n(n, root, child):
    """CDF кількості дітей кореня за умови |T| = n; потрібна умовному генератору."""
    mode, a, b = child
    lf, lg, la, l1a = _tables(n, child)
    rp = law_pmf(root, n)
    t = n - 1
    w = np.zeros(n)
    for j in range(1, n):
        if rp[j] > 0:
            w[j] = rp[j] * (j / t) * np.exp(_lsum(mode, a, int(b), t, t - j, lf, lg, la, l1a))
    return np.cumsum(w) / w.sum()


def a_table_law(N, root, child, eps=1e-30):
    """a_n для кореня й нащадків із різними законами. Сім'ї Poisson, Bin, Geom експоненційні, тому a_n не залежить від параметра."""
    mode, a, b = child
    lf, lg, la, l1a = _tables(N, child)
    F = _forward_law(N, law_pmf(root, N), mode, a, int(b), lf, lg, la, l1a, eps)
    num = _num_law(F, N, mode, a, int(b), lf, lg, la, l1a, eps)
    z = size_law(N, root, child)
    out = np.zeros(N + 1)
    out[3:] = num[3:] / z[3:]
    return out


def rrg_cluster(N, z, p):
    """P(|C| = s) для bond-перколяції на гратці Бете: s·g_s·p^{s−1}(1−p)^{s(z−2)+2}, g_s = z((z−1)s)!/(s!(s(z−2)+2)!)."""
    s = np.arange(1, N + 1, dtype=float)
    ln = (np.log(s) + np.log(z) + gammaln((z - 1) * s + 1) - gammaln(s + 1) - gammaln(s * (z - 2) + 3)
          + (s - 1) * np.log(p) + (s * (z - 2) + 2) * np.log1p(-p))
    return np.r_[0.0, np.exp(ln)]


# --- Межа a∞: хребет дерева Кестена, без підгонки ---
@nb.njit(cache=True)
def _spine_sum(t, mode, a, b, plain, kmax):
    """Σ_{k≥1} E[e^{−t S_k}] для нескінченного дерева: q_k = 1 − E e^{−t S_k} звичайного критичного дерева,
    K_k — той самий перетин уздовж хребта. plain: корінь має зміщені за розміром ξ; інакше 1 + ξ (рівномірний вузол)."""
    q = -np.expm1(-t)
    lk = -t
    tot = 0.0
    for _ in range(kmax):
        if mode == 0:
            lgq = -a * q
            lgp = -a * q
        elif mode == 1:
            lgq = -np.log1p(q)
            lgp = -2 * np.log1p(q)
        else:
            x = np.log1p(-a * q)
            lgq = b * x
            lgp = (b - 1) * x
        term = np.exp(-t + (lgp if plain else lgq) + lk)
        tot += term
        if term < 1e-17:
            break
        lk += -t + lgp
        q = -np.expm1(-t + lgq)
    return tot


@nb.njit(parallel=True, cache=True)
def _spine_grid(t, mode, a, b, plain, kmax):
    out = np.zeros(t.size)
    for i in nb.prange(t.size):
        out[i] = _spine_sum(t[i], mode, a, b, plain, kmax)
    return out


def ainf_quad(child, plain=False, ulo=-24.0, h=0.02):
    """lim a_n = ∫(2−4t)Σ_k E[e^{−tS_k}]dt на локальній границі умовного критичного дерева; закон нащадка критичний.
    plain=True — корінь плоского дерева, False — рівномірний вузол (для Пуассона й RRG це той самий корінь).
    Частина t < t₀ береться за асимптотикою Σ_k ~ C·t^{−1/2}."""
    mode, a, b = child
    u = np.arange(ulo, np.log(60.0) + h, h)
    t = np.exp(u)
    s = _spine_grid(t, mode, float(a), float(b), plain, 2_000_000_000)
    f = (2 - 4 * t) * s * t
    return float(h * (f.sum() - 0.5 * f[0] - 0.5 * f[-1]) + 4 * s[0] * t[0])
