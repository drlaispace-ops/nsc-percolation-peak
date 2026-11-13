"""П1.9. Положення піку NSC_∞ для ER і RRG(z=3,4) двома різними способами.

A. Квадратура. На дереві δ = 2/S − 4/S² = ∫₀^∞ (2−4t) e^{−tS} dt, тому
   NSC = ∫₀^∞ (2−4t) Σ_{k≥1} A_k(t) dt, A_k(t) = E[e^{−tS_k}; Z_k>0] = L_k(t,1) − L_k(t,0),
   L_k(t,x) = E[e^{−tS_k} x^{Z_k}]. Рекурсія за першим поколінням:
   L'_0 = x e^{−t}, L'_k = e^{−t} g_child(L'_{k−1}), L_k = e^{−t} g_root(L'_{k−1}).
   Індикатор Z_k>0 потрібен: BFS зупиняється, коли куля перестала рости (див. nsc_inf).
B. Ланцюг (S_k, Z_k) для ER із S ≤ N; маса, що вийшла за N, продовжується детерміновано
   (Z→cZ, S→S+Z). Спосіб не використовує перетворення Лапласа; похибка обрізання видна за N.

Запуск: .venv/bin/python code/P1/p1_9_cstar.py [N_chain ...]
"""
import sys
import time
from pathlib import Path

import numba as nb
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from gw import nsc_inf  # noqa: E402
from runmeta import runmeta, save  # noqa: E402
from stats import cubic_peak  # noqa: E402


# --- A. Квадратура ---
def logg(model, x):
    """(ln g_root, ln g_child) як функції q = 1 − s; так зберігається точність при малих t (s → 1)."""
    if model == "ER":
        f = lambda q: -x * q
        return f, f
    z = int(model[1:])  # "z3" → RRG, bond-перколяція з імовірністю p = x
    return (lambda q: z * np.log1p(-x * q)), (lambda q: (z - 1) * np.log1p(-x * q))


def sum_A(model, x, t, kmax=2_000_000, soft=False):
    """Σ_{k≥1} A_k(t) на масиві t, у змінних q = 1 − L'.

    A_k спадає за k (S_k росте, подія Z_k>0 звужується), тому вузол t заморожуємо, щойно A_k < 1e-13;
    далі різниця обмежена округленням. Повільні лише малі t, де вага t·dt мала."""
    lroot, lchild = logg(model, x)
    tot = np.zeros_like(t)
    idx = np.arange(t.size)
    ti = t
    qa = -np.expm1(-ti)  # L'_0 при x = 1
    qb = np.ones_like(ti)  # L'_0 при x = 0
    for _ in range(kmax):
        d = np.expm1(-ti + lroot(qa)) - np.expm1(-ti + lroot(qb))
        tot[idx] += d
        keep = np.abs(d) >= 1e-13
        if not keep.any():
            return tot
        idx, ti, qa, qb = idx[keep], ti[keep], qa[keep], qb[keep]
        qa, qb = -np.expm1(-ti + lchild(qa)), -np.expm1(-ti + lchild(qb))
    if soft:
        return tot
    raise RuntimeError(f"немає збіжності за k: {model}, x={x}")


def mean_child(model, x):
    return x if model == "ER" else (int(model[1:]) - 1) * x


def nsc_quad(model, x, h=0.01, uhi=np.log(60.0)):
    """Трапеції за u = ln t. Обрізання знизу: e^{ulo}·K, 1e-15 далеко від порога.
    Рівно на порозі k ~ 1/√t і шум округлення не падає нижче 1e-13, тому там ulo = −18, 30000 поколінь
    і похибка до ~1e-6. Така точка трапляється лише на краю вікна exp10 і в пік не входить."""
    critical = abs(mean_child(model, x) - 1) < 1e-3
    u = np.arange(-18.0 if critical else -40.0, uhi + h, h)
    t = np.exp(u)
    sa = sum_A(model, x, t, kmax=30000, soft=True) if critical else sum_A(model, x, t)
    return float(h * np.sum((2 - 4 * t) * sa * t))


def peak(model, lo, hi, step=1e-4):
    dF = lambda x: (nsc_quad(model, x + step) - nsc_quad(model, x - step)) / (2 * step)
    return brentq(dF, lo, hi, xtol=1e-7)


def giant(model, x):
    if model == "ER":
        return float(1 + lambertw(-x * np.exp(-x), 0).real / x)
    z = int(model[1:])
    w = brentq(lambda w: 1 - x + x * w ** (z - 1) - w, 0.0, 1 - 1e-9)
    return float(1 - w ** z)  # w — імовірність, що ребро не веде до нескінченності


# --- B. Ланцюг (S, Z) для ER ---
@nb.njit(cache=True)
def _phi(s):
    return 2.0 * (s - 2.0) / (s * s)


@nb.njit(cache=True)
def _tail(s, z, c):
    """Детерміноване продовження після виходу за N: φ у наступних поколіннях."""
    tot = 0.0
    for _ in range(400):
        z = c * z
        s = s + z
        tot += _phi(s)
        if s > 1e15:
            break
    return tot + 2.0 / s * c / (c - 1.0) * 1.0 / c


@nb.njit(cache=True)
def chain(c, N, eps=1e-16, kmax=400):
    P = np.zeros((N + 1, N + 1))
    P[1, 1] = 1.0  # S_0 = 1, Z_0 = 1
    tot, lost = 0.0, 0.0
    for _ in range(kmax):
        Q = np.zeros((N + 1, N + 1))
        mass = 0.0
        for s in range(1, N + 1):
            for z in range(1, s + 1):
                m = P[s, z]
                if m < eps:
                    lost += m
                    continue
                mu = c * z
                # Poisson(mu) від моди в обидва боки, щоб не мати underflow e^{-mu}
                z0 = int(mu)
                p0 = np.exp(z0 * np.log(mu) - mu - np.sum(np.log(np.arange(1, z0 + 1))))
                for sign in (1, -1):
                    p = p0
                    zn = z0 if sign == 1 else z0 - 1
                    if sign == -1:
                        p = p0 * z0 / mu if z0 > 0 else 0.0
                    while zn >= 0 and p > 1e-19:
                        if zn > 0:
                            sn = s + zn
                            w = m * p
                            tot += w * _phi(sn)
                            if sn <= N:
                                Q[sn, zn] += w
                                mass += w
                            else:
                                tot += w * _tail(float(sn), float(zn), c)
                        if sign == 1:
                            zn += 1
                            p = p * mu / zn
                        else:
                            p = p * zn / mu
                            zn -= 1
        P = Q
        if mass < 1e-15:
            break
    return tot, lost


if __name__ == "__main__":
    Ns = [int(a) for a in sys.argv[1:]] or [500, 1000]
    t0 = time.time()
    res = {}

    # Перевірка рекурсії: k = 1 задається прямою сумою за законом Пуассона
    c = 1.4
    z = np.arange(1, 400.0)
    logp = z * np.log(c) - c - np.cumsum(np.log(z))
    direct = float(np.sum(np.exp(logp) * (2 * (1 + z) - 4) / (1 + z) ** 2))
    u = np.arange(-40, np.log(60.0), 0.005)
    t = np.exp(u)
    et = np.exp(-t)
    a1 = et * (np.exp(c * (et - 1)) - np.exp(-c))
    res["check_k1"] = dict(direct=direct, quad=float(0.005 * np.sum((2 - 4 * t) * a1 * t)))

    # Збіжність за кроком і значення для c < 1, відомі з таблиці a_n
    res["check_h"] = {str(h): nsc_quad("ER", 1.4, h=h) for h in (0.04, 0.02, 0.01, 0.005)}
    res["check_c_lt_1"] = {str(c): nsc_quad("ER", c) for c in (0.5, 0.8)}

    # Пік
    res["peak"] = {}
    exp10 = __import__("json").load(open(ROOT / "results" / "exp10_gw_fine.json"))
    for model, key, lo, hi in [("ER", "ER", 1.3, 1.5), ("z3", "RRG z=3", 0.52, 0.62), ("z4", "RRG z=4", 0.37, 0.44)]:
        x = peak(model, lo, hi)
        e = exp10[key]
        xs, ys, ses = map(np.array, (e["x"], e["nsc"], e["se"]))
        exact = np.array([nsc_quad(model, v) for v in xs])
        zs = (ys - exact) / ses
        par = np.polyfit(xs, exact, 2)
        res["peak"][key] = dict(
            xstar=x, nsc=nsc_quad(model, x), P=giant(model, x), P_at_recorded=giant(model, e["xstar"]),
            recorded=e["xstar"], recorded_err=e["xstar_err"],
            parabola_on_exact_same_window=float(-par[1] / (2 * par[0])),
            cubic_on_exact_same_window=float(cubic_peak(xs, exact, window=len(xs))[0]),
            exp10_z_min=float(zs.min()), exp10_z_max=float(zs.max()), exp10_points=len(xs))
        print(key, res["peak"][key], flush=True)

    # B. Ланцюг для ER: сітка навколо піку, точніша за очікуваний зсув
    cs = np.array([1.37, 1.385, 1.40, 1.415, 1.43])
    res["chain"] = {}
    qv = np.array([nsc_quad("ER", c) for c in cs])
    for N in Ns:
        t1 = time.time()
        v = np.array([chain(float(c), N)[0] for c in cs])
        xc = cs.mean()
        co = np.polyfit(cs - xc, v, 3)
        r = np.roots(np.polyder(co))
        r = r[np.isreal(r)].real
        xstar = float(r[np.argmin(np.abs(r))] + xc)
        res["chain"][str(N)] = dict(c=cs.tolist(), nsc=v.tolist(), quad=qv.tolist(), max_abs_diff=float(np.max(np.abs(v - qv))),
                                    cubic_xstar=xstar, seconds=time.time() - t1)
        print("chain", N, res["chain"][str(N)], flush=True)
    co = np.polyfit(cs - cs.mean(), qv, 3)
    r = np.roots(np.polyder(co))
    r = r[np.isreal(r)].real
    res["chain_grid_quad_cubic_xstar"] = float(r[np.argmin(np.abs(r))] + cs.mean())

    # Незалежна вибірка: nsc_inf у точці запису і в новій точці, спільного ГВЧ немає
    ss = np.random.SeedSequence(20261106)
    s1, s2 = ss.spawn(2)
    M = 4_000_000
    res["mc"] = {}
    for c, s in ((res["peak"]["ER"]["xstar"], s1), (1.4147, s2)):
        rng = np.random.default_rng(s)
        r = nsc_inf(lambda m, c=c: rng.poisson(c, m), lambda Z, c=c: rng.poisson(c * Z), M)
        res["mc"][f"{c:.5f}"] = dict(mc=r["nsc"], se=r["se"], quad=nsc_quad("ER", c))
    print(res["mc"], flush=True)
    res["seconds"] = time.time() - t0
    meta = runmeta(__file__, params=dict(Ns=Ns, h=0.01, ulo="-40 (−18 рівно на порозі)", M=M), seeds=dict(entropy=20261106))
    print(save(ROOT / "results" / "P1" / "p1_9_cstar.json", meta, **res))
    print(res["check_k1"], res["check_h"], res["check_c_lt_1"], res["chain_grid_quad_cubic_xstar"])
