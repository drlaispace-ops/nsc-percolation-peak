"""П1.3. Аналіз прогону p1_3_run.py: таблиця за L і перевірка гіпотези B (числа для ⛳, рішення — не тут).

Для кожної завершеної задачі (results/P1/p1_3/<решітка>_L<L>.json) з сирих шматків:
  * p*(L), NSC(p*), P(p*) — кубіка у вікні HW, бутстреп за реалізаціями (B_BOOT);
  * систематика вікна: кубіка при 0.6·HW і 1.5·HW, поліном 4-го степеня при 1.5·HW — максимум |Δp*| від основної;
  * кривина піку −NSC″(p*) (з тієї ж кубіки), p_{1/2}(L): P = 1/2, p_χ(L): пік χ (кубіка, вікно HW);
  * для site — обидві нормалізації.
Гіпотеза B (план, ⛳ кінець грудня): у d = 3, 4 P(p*(L), L) стала в межах 2σ, а p*(L) збігається (χ²/dof < 2);
відкидаємо, якщо P* спадає як L^(−β/ν). Тут — лише статистики: χ²/dof сталої і закону A·L^(−β/ν) (β, ν — constants.csv),
максимальне |P*(L) − P̄|/σ, χ²/dof для p*(L) = p∞ + a·L^(−1/ν). L з невиконаною ціллю SE позначаються.
Запуск: .venv/bin/python code/P1/p1_3_analyse.py  →  results/P1/p1_3_analysis.json
"""
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
sys.path.insert(0, str(ROOT / "code" / "P1"))
import p1_3_run as run  # noqa: E402  (імпорт без запуску: розрахунок лише під __main__)
from runmeta import runmeta, save  # noqa: E402

B_BOOT = 500
warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanstd на порожніх (p_χ без піку у вікні — NaN)
CONST = {}
for r in csv.DictReader(open(ROOT / "data" / "P1" / "constants.csv")):
    if r["system"].startswith("perc_") and r["quantity"] in ("beta", "nu"):
        CONST.setdefault(int(r["d"]), {})[r["quantity"]] = float(r["value"])


def poly_peak(ps, y, hw, deg=3):
    """Пік полінома степеня deg у вікні |p − p_argmax| ≤ hw: (p*, y*, −y″(p*))."""
    i = int(np.argmax(y))
    s = np.abs(ps - ps[i]) <= hw * (1 + 1e-9)
    if s.sum() < deg + 2:
        return np.nan, np.nan, np.nan
    x = ps[s]
    xc = x.mean()
    c = np.polyfit(x - xc, y[s], deg)
    r = np.roots(np.polyder(c))
    r = r[np.isreal(r)].real
    r = r[(r >= x.min() - xc) & (r <= x.max() - xc) & (np.polyval(np.polyder(c, 2), r) < 0)]
    if not len(r):
        return np.nan, np.nan, np.nan
    r = r[np.argmax(np.polyval(c, r))]
    return float(r + xc), float(np.polyval(c, r)), float(-np.polyval(np.polyder(c, 2), r))


def crossing(ps, y, level):
    k = np.flatnonzero((y[:-1] < level) & (y[1:] >= level))
    if not len(k):
        return np.nan
    k = k[0]
    return float(ps[k] + (level - y[k]) * (ps[k + 1] - ps[k]) / (y[k + 1] - y[k]))


def stats_one(ps, Y, Pc, chi, hw):
    def est(idx):
        y, P = Y[idx].mean(0), Pc[idx].mean(0)
        x, h, curv = poly_peak(ps, y, hw)
        return [x, h, curv, np.interp(x, ps, P) if np.isfinite(x) else np.nan, crossing(ps, P, 0.5),
                poly_peak(ps, chi[idx].mean(0), hw)[0]]
    R = len(Y)
    full = est(np.arange(R))
    rng = np.random.default_rng(20261113)
    bs = np.array([est(rng.integers(0, R, R)) for _ in range(B_BOOT)])
    se = np.nanstd(bs, axis=0, ddof=1)
    y = Y.mean(0)
    alt = [poly_peak(ps, y, 0.6 * hw)[0], poly_peak(ps, y, 1.5 * hw)[0], poly_peak(ps, y, 1.5 * hw, 4)[0]]
    names = ["p_star", "nsc_star", "curv", "P_star", "p_half", "p_chi"]
    out = {k: float(v) for k, v in zip(names, full)}
    out.update({"se_" + k: float(v) for k, v in zip(names, se)})
    out.update(p_star_alt=dict(cubic_0_6hw=alt[0], cubic_1_5hw=alt[1], quartic_1_5hw=alt[2]),
               syst_window=float(np.nanmax(np.abs(np.array(alt) - full[0]))), fail=float(np.mean(~np.isfinite(bs[:, 0]))))
    return out


def chi2dof(y, s, model):
    return float(np.sum(((y - model) / s) ** 2) / max(len(y) - 1, 1))


def main():
    res = {}
    for name in run.NAMES:
        d, kind = run.CFG[name]["d"], run.CFG[name]["kind"]
        rows = []
        for f in sorted(run.RES.glob(f"{run.SLUG[name]}_L*.json"), key=lambda f: int(f.stem.split("_L")[-1])):
            sm = json.load(open(f))
            L = sm["meta"]["params"]["L"]
            ps = np.array(sm["ps"])
            S = run.load(run.DATA / run.SLUG[name] / f"L{L}", ["nsc", "nsc_occ", "P", "P_occ", "chi", "chi_occ"])
            row = dict(L=L, R=len(S["nsc"]), hours=sm["hours"], met_target=sm["met_target"], exact_check=sm["exact_check"],
                       all=stats_one(ps, S["nsc"], S["P"], S["chi"], run.HW[d]))
            if kind == "site":
                row["occ"] = stats_one(ps, S["nsc_occ"], S["P_occ"], S["chi_occ"], run.HW[d])
            rows.append(row)
            a = row["all"]
            print(f"{name:12s} L={L:5d} R={row['R']:5d}  p*={a['p_star']:.4f}±{a['se_p_star']:.4f} (вікно ±{a['syst_window']:.4f})"
                  f"  P*={a['P_star']:.3f}±{a['se_P_star']:.3f}  p½={a['p_half']:.4f}  p_χ={a['p_chi']:.4f}"
                  f"{'' if row['met_target'] else '  [ціль SE не досягнуто]'}", flush=True)
        hyp = {}
        if len(rows) >= 3 and d in CONST:
            bn, nu = CONST[d]["beta"] / CONST[d]["nu"], CONST[d]["nu"]
            for norm in ("all", "occ") if kind == "site" else ("all",):
                Ls = np.array([r["L"] for r in rows], float)
                P = np.array([r[norm]["P_star"] for r in rows])
                sP = np.array([r[norm]["se_P_star"] for r in rows])
                p = np.array([r[norm]["p_star"] for r in rows])
                sp = np.hypot([r[norm]["se_p_star"] for r in rows], [r[norm]["syst_window"] for r in rows])
                w = 1 / sP ** 2
                Pbar = float(np.sum(w * P) / np.sum(w))
                A = float(np.sum(w * P * Ls ** -bn) / np.sum(w * Ls ** (-2 * bn)))
                h = dict(beta_over_nu=bn, P_const=Pbar, chi2dof_const=chi2dof(P, sP, Pbar),
                         max_dev_sigma=float(np.max(np.abs(P - Pbar) / sP)),
                         chi2dof_power=chi2dof(P, sP, A * Ls ** -bn), A_power=A)
                try:
                    (pinf, aa), cov = curve_fit(lambda L, pinf, aa: pinf + aa * L ** (-1 / nu), Ls, p, sigma=sp,
                                                absolute_sigma=True, p0=[p[-1], 0.0])
                    m = pinf + aa * Ls ** (-1 / nu)
                    h.update(p_inf=float(pinf), se_p_inf=float(np.sqrt(cov[0, 0])),
                             chi2dof_p=float(np.sum(((p - m) / sp) ** 2) / max(len(p) - 2, 1)))
                except RuntimeError:
                    pass
                hyp[norm] = h
                print(f"   B ({norm}): P̄={Pbar:.3f}, χ²/dof стала {h['chi2dof_const']:.2f}, L^(−β/ν) {h['chi2dof_power']:.2f}; "
                      f"p∞={h.get('p_inf', np.nan):.4f}±{h.get('se_p_inf', np.nan):.4f} (χ²/dof {h.get('chi2dof_p', np.nan):.2f})",
                      flush=True)
        res[name] = dict(d=d, kind=kind, pc=run.CFG[name]["pc"], rows=rows, hypothesis_B=hyp)
    meta = runmeta(__file__, params=dict(B_boot=B_BOOT, halfwidth=run.HW, beta_nu=CONST))
    save(run.RES.parent / "p1_3_analysis.json", meta, lattices=res)


if __name__ == "__main__":
    main()
