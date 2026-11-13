"""Перевірки для відповіді рецензенту (стаття 1, П1): лише аналіз наявних даних, без нових симуляцій.

Запуск:  .venv/bin/python paper/P1/referee_checks.py   ->  paper/P1/referee_checks.json
Джерела: results/P1/p1_3_analysis.json, results/P1/p1_3b_bethe_lattices.json, results/P1/p1_3/*.json (сітки ps),
сирі криві за реалізаціями data/P1/p1_3/<решітка>/L<L>/chunk_*.npz (через p1_3_run.load; імпорт без запуску).

Розділи JSON:
  M1  — пік NSC проти піку χ (p_chi) і точки P = 1/2 (p_half) на найбільшому L; FSS-екстраполяція p_chi.
  M4  — 2D: (а) залежність кубіки від вікна на реальних кривих; (б) оцінювач «зміна знака скінченних різниць» (FD)
        і кубіка з вікном h(L) = h0·(L/Lmax)^(−1/ν) — з бутстрепом; (в) нуль-модель «злам точно в pc» (дві прямі +
        спільна парабола, підігнані до кривої найбільшого L) і різкий злам +1/−5 — зміщення кожного оцінювача на тій
        самій сітці; (г) порівняння FSS-моделей p*(L) = p∞ + a·L^(−1/ν) і p*(L) = pc + a·L^(−ω).
  M3  — сталість P*: χ²/dof сталої, нахил b у P* = a + b·ln L, ефективний показник κ у P* ∝ L^(−κ) проти β/ν,
        для L ≥ Lmin; решітки П1.3 і П1.3b (алмазна, ОЦК, ГЦК, кубічна з П1.3b).
  occ — перколяція вузлів, нормування на зайняті вузли.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "lib"), str(ROOT / "code" / "P1")]
import p1_3_run as run  # noqa: E402
from p1_3_analyse import poly_peak  # noqa: E402
from runmeta import runmeta  # noqa: E402

RES = ROOT / "results" / "P1"
AN = json.load(open(RES / "p1_3_analysis.json"))
LAT = AN["lattices"]
BN = AN["meta"]["params"]["beta_nu"]
A3 = json.load(open(RES / "p1_3b_bethe_lattices.json"))["lattices"]
B = 300
SEED = 20261112
OUT = {}


def rel(x, pc):
    return float(x / pc - 1)


# ============================================================ M1
m1 = {}
for name, v in LAT.items():
    pc, a = v["pc"], v["rows"][-1]["all"]
    d = v["d"]
    nu = BN[str(d)]["nu"]
    Ls = np.array([r["L"] for r in v["rows"]], float)
    pch = np.array([r["all"]["p_chi"] for r in v["rows"]])
    sch = np.array([r["all"]["se_p_chi"] for r in v["rows"]])
    ok = np.isfinite(pch) & np.isfinite(sch) & (sch > 0)
    fit = None
    if ok.sum() >= 3:
        (pinf, aa), cov = curve_fit(lambda L, p0, a1: p0 + a1 * L ** (-1 / nu), Ls[ok], pch[ok], sigma=sch[ok],
                                    absolute_sigma=True, p0=[pch[ok][-1], 0.0])
        mod = pinf + aa * Ls[ok] ** (-1 / nu)
        fit = dict(p_inf=float(pinf), se_p_inf=float(np.sqrt(cov[0, 0])), rel=rel(pinf, pc),
                   chi2dof=float(np.sum(((pch[ok] - mod) / sch[ok]) ** 2) / max(ok.sum() - 2, 1)), n=int(ok.sum()))
    m1[name] = dict(L=v["rows"][-1]["L"], pc=pc, rel_p_star=rel(a["p_star"], pc), rel_p_chi=rel(a["p_chi"], pc),
                    se_rel_p_chi=a["se_p_chi"] / pc, rel_p_half=rel(a["p_half"], pc), se_rel_p_half=a["se_p_half"] / pc,
                    rel_p_inf=rel(v["hypothesis_B"]["all"]["p_inf"], pc), P_star=a["P_star"],
                    p_star_minus_p_half_rel=float((a["p_star"] - a["p_half"]) / pc), p_chi_fss=fit)
for nm in ("diamond", "bcc", "fcc"):
    v = A3[nm]
    m1["A3:" + nm] = dict(L=v["rows"][-1]["L"], pc=v["pc"], rel_p_star=rel(v["rows"][-1]["p_star"], v["pc"]),
                          rel_p_half=rel(v["p_half_largest"], v["pc"]), P_star=v["P_star_largest"])
OUT["M1"] = m1


# ============================================================ M4
def fd_peak(ps, y):
    """Нуль похідної за скінченними різницями: перша зміна знака нахилу (від + до −) біля argmax,
    лінійна інтерполяція нахилів між серединами інтервалів."""
    s = np.diff(y) / np.diff(ps)
    mid = 0.5 * (ps[1:] + ps[:-1])
    i = int(np.argmax(y))
    # шукаємо інтервал k (нахил s[k] > 0, s[k+1] <= 0) найближчий до argmax
    ks = [k for k in range(len(s) - 1) if s[k] > 0 >= s[k + 1]]
    if not ks:
        return np.nan
    k = min(ks, key=lambda k: abs(mid[k] - ps[i]))
    return float(mid[k] + s[k] * (mid[k + 1] - mid[k]) / (s[k] - s[k + 1]))


def scaled_h(L, Lmax, nu, h0=0.004):
    return h0 * (L / Lmax) ** (-1 / nu)


def boot(fn, Y, rng):
    R = len(Y)
    vals = np.array([fn(Y[rng.integers(0, R, R)].mean(0)) for _ in range(B)])
    ok = np.isfinite(vals)
    return float(np.std(vals[ok], ddof=1)), float(1 - ok.mean())


HWS = (0.004, 0.006, 0.008, 0.012, 0.018)
m4 = {}
for name in ("2D bond sq", "2D bond tri", "2D site sq"):
    v = LAT[name]
    pc, nu = v["pc"], BN["2"]["nu"]
    Lmax = v["rows"][-1]["L"]
    rows = []
    rng = np.random.default_rng(SEED)
    for r in v["rows"]:
        L = r["L"]
        ps = np.array(json.load(open(run.RES / f"{run.SLUG[name]}_L{L}.json"))["ps"])
        Y = run.load(run.DATA / run.SLUG[name] / f"L{L}", ["nsc"])["nsc"]
        y = Y.mean(0)
        cub = {str(h): poly_peak(ps, y, h)[0] for h in HWS}
        hL = scaled_h(L, Lmax, nu)
        p_sc = poly_peak(ps, y, hL)[0]
        se_sc, f_sc = boot(lambda yy: poly_peak(ps, yy, hL)[0], Y, rng)
        p_fd = fd_peak(ps, y)
        se_fd, f_fd = boot(lambda yy: fd_peak(ps, yy), Y, rng)
        rows.append(dict(L=L, R=len(Y), cubic_by_h=cub, rel_cubic_by_h={k: rel(x, pc) for k, x in cub.items()},
                         h_scaled=hL, p_scaled=p_sc, se_scaled=se_sc, fail_scaled=f_sc, rel_scaled=rel(p_sc, pc),
                         p_fd=p_fd, se_fd=se_fd, fail_fd=f_fd, rel_fd=rel(p_fd, pc),
                         p_cubic_main=r["all"]["p_star"],
                         err_cubic_main=float(np.hypot(r["all"]["se_p_star"], r["all"]["syst_window"])),
                         rounding_scale_p=float(pc * L ** (-1 / nu))))
    # ---------- нуль-модель «злам точно в pc»: підгонка до кривої Lmax, |p − pc| ≤ 0.02
    sm = json.load(open(run.RES / f"{run.SLUG[name]}_L{Lmax}.json"))
    ps = np.array(sm["ps"])
    y, se = np.array(sm["mean"]["nsc"]), np.array(sm["mean"]["se"])
    w = np.abs(ps - pc) <= 0.02

    def kink(p, a0, sl, sr, q, pk):
        x = p - pk
        return a0 + np.where(x < 0, sl * x, sr * x) + q * x ** 2

    (a0, sl, sr, q), _ = curve_fit(lambda p, a0, sl, sr, q: kink(p, a0, sl, sr, q, pc), ps[w], y[w], sigma=se[w],
                                   p0=[y[w].max(), 1, -5, 0])
    chi_pc = float(np.sum(((y[w] - kink(ps[w], a0, sl, sr, q, pc)) / se[w]) ** 2))
    parf, covf = curve_fit(kink, ps[w], y[w], sigma=se[w], absolute_sigma=True, p0=[a0, sl, sr, q, pc - 0.002])
    chi_free = float(np.sum(((y[w] - kink(ps[w], *parf)) / se[w]) ** 2))
    null = {}
    for lab, curve in (("fitted_kink_at_pc", kink(ps, a0, sl, sr, q, pc)),
                       ("sharp_+1_-5", np.where(ps < pc, 1.0, -5.0) * (ps - pc))):
        null[lab] = dict(rel_cubic_by_h={str(h): rel(poly_peak(ps, curve, h)[0], pc) for h in HWS},
                         rel_fd=rel(fd_peak(ps, curve), pc),
                         rel_scaled_Lmax=rel(poly_peak(ps, curve, scaled_h(Lmax, Lmax, nu))[0], pc))
    sharp = {}
    for slv in (0.5, 1.0, 2.0):
        for srv in (-3.0, -5.0, -8.0):
            c = np.where(ps < pc, slv, srv) * (ps - pc)
            sharp[f"{slv}/{srv}"] = rel(poly_peak(ps, c, 0.012)[0], pc)
    # ---------- FSS-моделі
    Ls = np.array([r["L"] for r in rows], float)
    fss = {}
    for est, pk, er in (("cubic_main", "p_cubic_main", "err_cubic_main"), ("fd", "p_fd", "se_fd"),
                        ("scaled", "p_scaled", "se_scaled")):
        p = np.array([r[pk] for r in rows])
        s = np.array([r[er] for r in rows])
        o = {}
        try:
            (pinf, aa), cov = curve_fit(lambda L, p0, a1: p0 + a1 * L ** (-1 / nu), Ls, p, sigma=s,
                                        absolute_sigma=True, p0=[p[-1], 0.0])
            o["A_pinf_free"] = dict(p_inf=float(pinf), se=float(np.sqrt(cov[0, 0])), rel=rel(pinf, pc),
                                    chi2=float(np.sum(((p - pinf - aa * Ls ** (-1 / nu)) / s) ** 2)), dof=len(p) - 2)
        except RuntimeError:
            pass
        try:
            (aa, om), cov = curve_fit(lambda L, a1, om: pc + a1 * L ** (-om), Ls, p, sigma=s, absolute_sigma=True,
                                      p0=[-0.01, 0.3], maxfev=20000)
            o["B_to_pc"] = dict(omega=float(om), se_omega=float(np.sqrt(cov[1, 1])),
                                chi2=float(np.sum(((p - pc - aa * Ls ** (-om)) / s) ** 2)), dof=len(p) - 2)
        except RuntimeError:
            pass
        fss[est] = o
    m4[name] = dict(pc=pc, Lmax=Lmax, rows=rows,
                    null_fit=dict(sl=float(sl), sr=float(sr), q=float(q), chi2_kink_at_pc=chi_pc,
                                  chi2_kink_free=chi_free, kink_free_rel=rel(parf[4], pc),
                                  se_kink_free_rel=float(np.sqrt(covf[4, 4]) / pc), npts=int(w.sum())),
                    null_bias=null, sharp_kink_cubic012=sharp, fss_models=fss)
    print(name, "FD", [round(r["rel_fd"] * 100, 2) for r in rows], "null", null["fitted_kink_at_pc"])
OUT["M4"] = m4


# ============================================================ M3
def ptests(Ls, P, sP, bn, Lmin):
    m = Ls >= Lmin
    Ls, P, sP = Ls[m], P[m], sP[m]
    if len(Ls) < 3:
        return None
    w = 1 / sP ** 2
    Pb = np.sum(w * P) / np.sum(w)
    chi_c = float(np.sum(w * (P - Pb) ** 2) / (len(P) - 1))
    (a, b), cov = curve_fit(lambda x, a, b: a + b * np.log(x), Ls, P, sigma=sP, absolute_sigma=True)
    (A, k), covk = curve_fit(lambda x, A, k: A * x ** (-k), Ls, P, sigma=sP, absolute_sigma=True, p0=[P[0], 0.1])
    Ap = np.sum(w * P * Ls ** -bn) / np.sum(w * Ls ** (-2 * bn))
    return dict(Lmin=int(Lmin), Ls=[int(x) for x in Ls], L_ratio=float(Ls[-1] / Ls[0]), P_mean=float(Pb),
                chi2dof_const=chi_c, b_lnL=float(b), se_b=float(np.sqrt(cov[1, 1])), z_b=float(b / np.sqrt(cov[1, 1])),
                kappa=float(k), se_kappa=float(np.sqrt(covk[1, 1])), beta_over_nu=float(bn),
                chi2dof_power=float(np.sum(w * (P - Ap * Ls ** -bn) ** 2) / (len(P) - 1)))


m3 = {}
for name, v in LAT.items():
    d = v["d"]
    bn = BN[str(d)]["beta"] / BN[str(d)]["nu"]
    Ls = np.array([r["L"] for r in v["rows"]], float)
    P = np.array([r["all"]["P_star"] for r in v["rows"]])
    sP = np.array([r["all"]["se_P_star"] for r in v["rows"]])
    m3[name] = {f"L>={int(Lm)}": ptests(Ls, P, sP, bn, Lm) for Lm in sorted(set(Ls))[:-2]}
for nm in ("diamond", "sc", "bcc", "fcc"):
    v = A3[nm]
    bn = BN["3"]["beta"] / BN["3"]["nu"]
    Ls = np.array([r["L"] for r in v["rows"]], float)
    P = np.array([r["P_star"] for r in v["rows"]])
    sP = np.array([r["se_P_star"] for r in v["rows"]])
    m3["A3:" + nm] = {f"L>={int(Lm)}": ptests(Ls, P, sP, bn, Lm) for Lm in sorted(set(Ls))[:-2]}
OUT["M3"] = m3

# ============================================================ occ (перколяція вузлів, нормування на зайняті)
s = LAT["2D site sq"]
OUT["occ"] = dict(rows=[dict(L=r["L"], p_star=r["occ"]["p_star"], se_p_star=r["occ"]["se_p_star"],
                             P_star=r["occ"]["P_star"], se_P_star=r["occ"]["se_P_star"]) for r in s["rows"]],
                  hypothesis_B=s["hypothesis_B"]["occ"], rel_p_inf=rel(s["hypothesis_B"]["occ"]["p_inf"], s["pc"]),
                  all_rel_p_inf=rel(s["hypothesis_B"]["all"]["p_inf"], s["pc"]))

meta = runmeta(__file__, params=dict(B=B, seed=SEED, HWS=HWS, h0_scaled=0.004))
with open(Path(__file__).resolve().parent / "referee_checks.json", "w") as f:
    json.dump(dict(meta=meta, **OUT), f, ensure_ascii=False, indent=1, default=float)
print("готово")
