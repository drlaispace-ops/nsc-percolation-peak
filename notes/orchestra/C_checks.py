"""Завдання C (П1.13, решітки): контрольні обчислення до notes/P1_threshold_lattice.tex. Нових симуляцій немає.

Читає results/P1/p1_13_lattice.json, p1_13_threshold.json, p1_11_a_table.npy, p1_3_analysis.json, p1_3/*.json
і сирі дані data/P1/p1_3/ (через p1_3_run.load). Пише notes/orchestra/C_checks.json (з runmeta).
  1. psi_check   — ψ = β + x з JSON (лише підгонки з χ²/dof < 5), 1/σ = ν d_f, ψ_F = β + θ/σ, перерахунок лінійного фону в ε ln ε;
  2. giant_C0    — підгонки D = g∞ − g_G(p) зі сталою C0 (перевірка D(0⁺) = 0, тобто спільної межі);
  3. gG_pc       — g_G(pc, L), g_fin(pc, L) з p1_3/*.json (лінійна інтерполяція між вузлами сітки);
  4. mf_theta    — точна таблиця a_n (ER): n(a_n − a∞) = α ln n + β0 і калібрування трипараметричної підгонки θ (як у p1_13_lattice.py);
  5. fd_fss      — NSC′(pc, L) зі скінченних різниць: стала проти a + b ln L; L, з якого крок сітки ширший за вікно L^(−1/ν);
  6. one_sided   — нахили між сусідніми вузлами сітки навколо pc (бутстреп);
  7. psi_direct  — пряма підгонка NSC(p) = R0 + R1 ε + R2 ε² − c·ε₊^ψ (бутстреп) і положення її максимуму.
Запуск: NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python notes/orchestra/C_checks.py  (~2–3 хв)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "lib"), str(ROOT / "code" / "P1")]
import p1_3_run as run  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20261112
R1 = ROOT / "results" / "P1"
J = json.load(open(R1 / "p1_13_lattice.json"))["lattices"]
T = json.load(open(R1 / "p1_13_threshold.json"))
AN = json.load(open(R1 / "p1_3_analysis.json"))["lattices"]
NU = {n: J[n]["exponents"]["nu"] for n in J}


def curve(name, L):
    ps = np.array(json.load(open(run.RES / f"{run.SLUG[name]}_L{L}.json"))["ps"])
    return ps, run.load(run.DATA / run.SLUG[name] / f"L{L}", ["nsc"])["nsc"]


def psi_check():
    out = {}
    for n, R in J.items():
        ex, fits = R["exponents"], R["giant"]["fits"]
        xs = {m: f["x"] for m, f in fits.items() if f and f["chi2dof"] < 5}
        inv_s = ex["nu"] * ex["d_f"]
        th = {b: f["theta"] for b, f in R["journal"][-1]["fits"].items()}
        A, x, B = fits["lin_bg"]["par"]
        out[n] = dict(x_used=xs, psi=[ex["beta"] + min(xs.values()), ex["beta"] + max(xs.values())], psi_json=R["psi_range"],
                      inv_sigma=inv_s, psi_F={b: ex["beta"] + t * inv_s for b, t in th.items()},
                      psi_F_hyp=ex["beta"] + R["hypothesis"]["theta"] * inv_s, x_hyp=R["hypothesis"]["x"],
                      linbg_eps_coef=A + B, linbg_epslog_coef=A * (1 - x))
    return out


def giant_C0():
    models = {"C0+A e^x(1+c e)": (lambda e, C, A, x, c: C + A * e ** x * (1 + c * e), [0, 2, 0.8, -1]),
              "C0+a e+b e ln e": (lambda e, C, a, b: C + a * e + b * e * np.log(e), [0, 1, -1])}
    out = {}
    for n, R in J.items():
        G = R["giant"]
        e, D, s = map(np.array, (G["eps"], G["D"], G["se"]))
        out[n] = {}
        for m, (f, p0) in models.items():
            c, cov = curve_fit(f, e, D, p0=p0, sigma=s, maxfev=50000)
            r = (D - f(e, *c)) / s
            out[n][m] = dict(C0=float(c[0]), se_C0=float(np.sqrt(cov[0, 0])), par=c[1:].tolist(),
                             se=np.sqrt(np.diag(cov))[1:].tolist(), chi2dof=float(np.sum(r ** 2) / (len(e) - len(c))))
    return out


def gG_pc():
    out = {}
    for n in J:
        pc = AN[n]["pc"]
        out[n] = {}
        for row in AN[n]["rows"]:
            L = row["L"]
            d = json.load(open(run.RES / f"{run.SLUG[n]}_L{L}.json"))
            ps, m = np.array(d["ps"]), d["mean"]
            k = int(np.searchsorted(ps, pc)) - 1
            w = (pc - ps[k]) / (ps[k + 1] - ps[k])
            at = {key: float((1 - w) * m[key][k] + w * m[key][k + 1]) for key in ("g_giant", "g_fin", "P", "nsc")}
            out[n][str(L)] = at
    return out


def mf_theta():
    a = np.load(R1 / "p1_11_a_table.npy")
    ainf = T["ER"]["ainf"]
    out = dict(ainf=ainf, linlog={}, local_slope={}, calib={})
    for lo, hi in ((100, 1000), (1000, 3000), (2000, len(a) - 1)):
        n = np.arange(lo, hi + 1)
        y = n * (a[n] - ainf)
        al, be = np.polyfit(np.log(n), y, 1)
        out["linlog"][f"{lo}-{hi}"] = dict(alpha=float(al), beta0=float(be), rms=float(np.std(y - al * np.log(n) - be)))
    dd = np.abs(a - ainf)
    for N in (30, 100, 300, 1000, 2500):
        out["local_slope"][str(N)] = float(np.log(dd[2 * N] / dd[N]) / np.log(2))
    n = np.arange(len(a))
    b = np.floor(np.log2(np.maximum(n, 1))).astype(int)
    w = np.where(n >= 1, np.maximum(n, 1) ** -1.5, 0.0)  # вага вузла s·n_s(pc) ~ s^(−3/2)
    B, S, G = [], [], []
    for k in range(2, int(b.max()) + 1):
        m = (b == k) & (n >= 3)
        if m.any() and n[m].max() == 2 ** (k + 1) - 1:
            B.append(k)
            S.append(np.sum(w[m] * n[m]) / np.sum(w[m]))
            G.append(np.sum(w[m] * a[m]) / np.sum(w[m]))
    B, S, G = map(np.array, (B, S, G))
    for bmin in (4, 6, 8):
        sel = B >= bmin
        c = curve_fit(lambda s, g, A, t: g - A * s ** -t, S[sel], G[sel], p0=[G[sel][-1], 1, 0.5], maxfev=20000)[0]
        out["calib"][str(bmin)] = dict(g_inf=float(c[0]), dg=float(c[0] - ainf), theta=float(c[2]), bins=B[sel].tolist())
    return out


def fd_fss():
    out = {}
    for n, R in J.items():
        d = R["derivs"]
        L = np.array([q["L"] for q in d], float)
        y, s = np.array([q["d1"] for q in d]), np.array([q["se_d1"] for q in d])
        wm = float(np.sum(y / s ** 2) / np.sum(1 / s ** 2))
        c, cov = np.polyfit(np.log(L), y, 1, w=1 / s, cov="unscaled")
        out[n] = dict(const=wm, chi2dof_const=float(np.sum(((y - wm) / s) ** 2) / (len(y) - 1)),
                      b_lnL=float(c[0]), se_b=float(np.sqrt(cov[0, 0])),
                      chi2dof_lnL=float(np.sum(((y - np.polyval(c, np.log(L))) / s) ** 2) / (len(y) - 2)),
                      L_step_exceeds_window=float((d[0]["h"] / R["pc"]) ** (-NU[n])), eps_node=(d[0]["p"] - R["pc"]) / R["pc"])
        m = y < -0.3  # 2D: від'ємний нахил, чиста степенева −b·L^a без фону; ψ = 1 − a·ν
        if m.sum() >= 3:
            c, cov = curve_fit(lambda L, b, a: -b * L ** a, L[m], y[m], p0=[0.5, 0.3], sigma=s[m])
            r = (y[m] + c[0] * L[m] ** c[1]) / s[m]
            out[n]["pure_power"] = dict(L=L[m].tolist(), a=float(c[1]), se_a=float(np.sqrt(cov[1, 1])),
                                        chi2dof=float(np.sum(r ** 2) / (m.sum() - 2)), psi=float(1 - c[1] * NU[n]),
                                        se_psi=float(np.sqrt(cov[1, 1]) * NU[n]))
    return out


def one_sided(rng):
    out = {}
    for n in ("2D bond sq", "2D bond tri", "2D site sq", "3D bond"):
        pc = AN[n]["pc"]
        out[n] = {}
        for row in AN[n]["rows"]:
            ps, Y = curve(n, row["L"])
            k = int(np.searchsorted(ps, pc))
            idx = range(k - 3, k + 2)

            def sl(Y):
                y = Y.mean(0)
                return np.array([(y[i + 1] - y[i]) / (ps[i + 1] - ps[i]) for i in idx])
            bs = np.array([sl(Y[rng.integers(0, len(Y), len(Y))]) for _ in range(300)])
            out[n][str(row["L"])] = dict(eps_mid=[float(((ps[i] + ps[i + 1]) / 2 - pc) / pc) for i in idx],
                                         eps_lo=[float((ps[i] - pc) / pc) for i in idx], eps_hi=[float((ps[i + 1] - pc) / pc) for i in idx],
                                         slope=sl(Y).tolist(), se=bs.std(0, ddof=1).tolist())
    return out


def psi_direct(rng):
    f = lambda e, R0, R1, R2, c, psi: R0 + R1 * e + R2 * e ** 2 - c * np.clip(e, 0, None) ** psi  # noqa: E731
    out = {}
    for n, L, win in (("2D bond sq", 1024, (-0.03, 0.06)), ("2D bond sq", 1024, (-0.05, 0.1)), ("2D bond tri", 512, (-0.03, 0.06)),
                      ("2D site sq", 512, (-0.03, 0.06)), ("3D bond", 256, (-0.06, 0.12)), ("3D bond", 256, (-0.1, 0.2)),
                      ("3D bond", 192, (-0.06, 0.12)), ("4D bond", 32, (-0.1, 0.2)), ("4D bond", 32, (-0.15, 0.3))):
        pc = AN[n]["pc"]
        ps, Y = curve(n, L)
        e = (ps - pc) / pc
        for excl in (0.0, 1.0):  # 1.0: відкинути |ε| < L^(−1/ν) (зона заокруглення)
            m = (e >= win[0]) & (e <= win[1]) & (np.abs(e) >= excl * L ** (-1 / NU[n]))

            def fit(Y):
                y, s = Y[:, m].mean(0), Y[:, m].std(0, ddof=1) / np.sqrt(len(Y))
                try:
                    c = curve_fit(f, e[m], y, p0=[y.mean(), 1, 0, 1, 1.3], sigma=s, maxfev=20000,
                                  bounds=([-10, -100, -1e4, 0, 0.3], [10, 100, 1e4, 1e3, 3]))[0]
                    return c, float(np.sum(((y - f(e[m], *c)) / s) ** 2) / (m.sum() - 5))
                except (RuntimeError, ValueError):
                    return np.full(5, np.nan), np.nan
            c, chi = fit(Y)
            bs = np.array([fit(Y[rng.integers(0, len(Y), len(Y))])[0] for _ in range(150)])
            ee = np.linspace(win[0], win[1], 200001)
            k = int(np.argmax(f(ee, *c)))
            out[f"{n} | L={L} | eps in {list(win)} | excl={excl}"] = dict(
                npts=int(m.sum()), par=dict(zip(("R0", "R1", "R2", "c", "psi"), c.tolist())), se_psi=float(np.nanstd(bs[:, 4], ddof=1)),
                chi2dof_naive=chi, p_peak_fit=float(pc * (1 + ee[k])), eps_peak_fit=float(ee[k]))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    res = dict(psi_check=psi_check(), giant_C0=giant_C0(), gG_pc=gG_pc(), mf_theta=mf_theta(), fd_fss=fd_fss())
    res["one_sided"] = one_sided(rng)
    res["psi_direct"] = psi_direct(rng)
    for key, v in res.items():
        print(key, json.dumps(v, ensure_ascii=False, default=float)[:1500], "\n", flush=True)
    print(save(Path(__file__).with_name("C_checks.json"), runmeta(__file__, params=dict(B_slope=300, B_psi=150), seeds=[SEED]), **res))
