"""Завдання C, виправлення за вердиктом C_verdict.md: додаткові контрольні обчислення. Нових симуляцій немає.

Використовує функції й дані з notes/orchestra/C_checks.py (імпорт без запуску) і notes/orchestra/C_checks.json.
Пише notes/orchestra/C_checks2.json (з runmeta):
  1. psi_cubic   — пряма підгонка NSC = R0 + R1 ε + R2 ε² + R3 ε³ − c·ε₊^ψ (кубічний фон), ті самі вікна, що в C_checks.psi_direct;
  2. C0_total    — C0 з похибкою σ(g∞) (зсув g∞ зсуває C0 на ту саму величину) і з g∞, екстрапольованим при θ = θ_hyp;
  3. theta_drift — θ(s_min) з похибками з p1_13_lattice.json і значущість спаду 2⁴ → 2⁸;
  4. er_second   — екстраполяція чисельної напівсуми других похідних (ER) до ε → 0;
  5. fd_z        — скейлінгова змінна z = ε_node·L^(1/ν) вузла сітки, на якому береться NSC′(pc, L).
Запуск: NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python notes/orchestra/C_checks2.py  (~1–2 хв)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import C_checks as C  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20261113
CC = json.load(open(HERE / "C_checks.json"))
J, T, AN, NU = C.J, C.T, C.AN, C.NU
CONFIGS = C_CONFIGS = (("2D bond sq", 1024, (-0.03, 0.06)), ("2D bond sq", 1024, (-0.05, 0.1)), ("2D bond tri", 512, (-0.03, 0.06)),
                       ("2D site sq", 512, (-0.03, 0.06)), ("3D bond", 256, (-0.06, 0.12)), ("3D bond", 256, (-0.1, 0.2)),
                       ("3D bond", 192, (-0.06, 0.12)), ("4D bond", 32, (-0.1, 0.2)), ("4D bond", 32, (-0.15, 0.3)))


def psi_cubic(rng, B=100):
    f = lambda e, R0, R1, R2, R3, c, psi: R0 + R1 * e + R2 * e ** 2 + R3 * e ** 3 - c * np.clip(e, 0, None) ** psi  # noqa: E731
    out = {}
    for n, L, win in CONFIGS:
        pc = AN[n]["pc"]
        ps, Y = C.curve(n, L)
        e = (ps - pc) / pc
        for excl in (0.0, 1.0):
            m = (e >= win[0]) & (e <= win[1]) & (np.abs(e) >= excl * L ** (-1 / NU[n]))
            if excl and m.sum() == ((e >= win[0]) & (e <= win[1])).sum():
                continue  # виключення нічого не змінює (3D, 4D)

            def fit(Y):
                y, s = Y[:, m].mean(0), Y[:, m].std(0, ddof=1) / np.sqrt(len(Y))
                try:
                    c = curve_fit(f, e[m], y, p0=[y.mean(), 1, 0, 0, 1, 1.3], sigma=s, maxfev=40000,
                                  bounds=([-10, -100, -1e4, -1e6, 0, 0.3], [10, 100, 1e4, 1e6, 1e3, 3]))[0]
                    return c, float(np.sum(((y - f(e[m], *c)) / s) ** 2) / (m.sum() - 6))
                except (RuntimeError, ValueError):
                    return np.full(6, np.nan), np.nan
            c, chi = fit(Y)
            bs = np.array([fit(Y[rng.integers(0, len(Y), len(Y))])[0] for _ in range(B)])
            ee = np.linspace(win[0], win[1], 200001)
            k = int(np.argmax(f(ee, *c)))
            pk = np.array([pc * (1 + ee[int(np.argmax(f(ee, *b)))]) for b in bs if np.all(np.isfinite(b))])
            quad = CC["psi_direct"][f"{n} | L={L} | eps in {list(win)} | excl={excl}"]
            out[f"{n} | L={L} | eps in {list(win)} | excl={excl}"] = dict(
                npts=int(m.sum()), par=dict(zip(("R0", "R1", "R2", "R3", "c", "psi"), c.tolist())),
                se_psi=float(np.nanstd(bs[:, 5], ddof=1)), chi2dof_naive=chi, p_peak_fit=float(pc * (1 + ee[k])),
                se_p_peak=float(np.std(pk, ddof=1)), psi_quadratic=quad["par"]["psi"], chi2dof_quadratic=quad["chi2dof_naive"],
                p_peak_quadratic=quad["p_peak_fit"])
    return out


def C0_total():
    out = {}
    for n, R in J.items():
        jr = R["journal"][-1]
        fit6 = jr["fits"]["6"]
        th_h = R["hypothesis"]["theta"]
        b, s, g = map(np.array, (jr["bins"], jr["s_mean"], jr["g"]))
        sel = b >= 6
        cf = curve_fit(lambda s, gi, a: gi - a * s ** -th_h, s[sel], g[sel], p0=[g[sel][-1], 1.0])[0]
        shift = float(cf[0] - fit6["g_inf"])
        gG, gF = CC["gG_pc"][n][str(jr["L"])]["g_giant"], CC["gG_pc"][n][str(jr["L"])]["g_fin"]
        out[n] = dict(g_inf=fit6["g_inf"], se_g_inf=fit6["se_g_inf"], g_inf_thetahyp=float(cf[0]), shift_thetahyp=shift,
                      naive_amp=gG - gF, models={})
        for mname, q in CC["giant_C0"][n].items():
            tot = float(np.hypot(q["se_C0"], fit6["se_g_inf"]))
            out[n]["models"][mname] = dict(C0=q["C0"], se_stat=q["se_C0"], se_total=tot, z=q["C0"] / tot,
                                           rel=abs(q["C0"]) / (gG - gF), C0_thetahyp=q["C0"] + shift,
                                           rel_thetahyp=abs(q["C0"] + shift) / (gG - gF))
    return out


def theta_drift():
    out = {}
    for n, R in J.items():
        F = R["journal"][-1]["fits"]
        th = {k: (F[k]["theta"], F[k]["se_theta"]) for k in F}
        d = th["4"][0] - th["8"][0]
        out[n] = dict(theta=th, drop_4_8=d, z_drop_4_8=d / float(np.hypot(th["4"][1], th["8"][1])))
    return out


def er_second():
    eps = np.array([0.04, 0.02, 0.01, 0.005, 0.002, 0.001])  # EPS_S з code/P1/p1_13_threshold.py
    y = np.array(T["ER_second"]["sum_half_numeric"])
    out = dict(pred=T["ER_second"]["sum_half_pred"], last=float(y[-1]), fits={})
    for emax in (0.01, 0.005):
        m = eps <= emax
        X = np.c_[np.ones(m.sum()), eps[m], eps[m] * np.log(eps[m])]
        if m.sum() >= 3:
            out["fits"][f"1,e,e ln e; e<={emax}"] = float(np.linalg.lstsq(X, y[m], rcond=None)[0][0])
    m = eps <= 0.005
    out["fits"]["1,e; e<=0.005"] = float(np.polyfit(eps[m], y[m], 1)[1])
    return out


def fd_z():
    out = {}
    for n, R in J.items():
        out[n] = {str(q["L"]): dict(eps_node=(q["p"] - R["pc"]) / R["pc"], z=(q["p"] - R["pc"]) / R["pc"] * q["L"] ** (1 / NU[n]))
                  for q in R["derivs"]}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    res = dict(C0_total=C0_total(), theta_drift=theta_drift(), er_second=er_second(), fd_z=fd_z())
    res["psi_cubic"] = psi_cubic(rng)
    for key, v in res.items():
        print(key, json.dumps(v, ensure_ascii=False, default=float, indent=0)[:4000], "\n", flush=True)
    print(save(HERE / "C_checks2.json", runmeta(__file__, params=dict(B_psi=100, base="C_checks.json"), seeds=[SEED]), **res))
