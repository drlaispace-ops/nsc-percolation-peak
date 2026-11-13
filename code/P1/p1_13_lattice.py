"""П1.13, решітки. Поведінка NSC біля порога на сирих даних П1.3 (data/P1/p1_3/), нових розрахунків немає.

Розклад NSC = Σ_s s·n_s·g(s) + P·g_G; сингулярна частина виникає, лише якщо маса, що переходить у гігант,
має NodeSC, відмінну від гіганта. Перевіряємо:
  (1) g(s; pc) з журналу скінченних кластерів за ⌊log₂ s⌋: g(s) = g∞ − a·s^(−θ); межа g∞ проти g_G(pc);
      гіпотеза плану θ = 1 − d_min/d_f (обрив суми δ_k на хімічній відстані ~ s^(d_min/d_f));
  (2) відхід гіганта вище порога: g∞ − g_G(p) = A·ε^x (+ фон), ε = (p − pc)/pc, лише точки, де дві найбільші L
      збігаються (режим L ≫ ξ); гіпотеза x = ν(d_f − d_min); сингулярний член NSC ~ ε^ψ, ψ = β + x;
  (3) похідні NSC у pc скінченними різницями на рідній густій сітці (без вікна апроксимації) від L:
      NSC′(pc, L) ~ L^((1−ψ)/ν) при ψ < 1 (розбіжний нахил), збігається при ψ > 1.
Запуск: .venv/bin/python code/P1/p1_13_lattice.py → results/P1/p1_13_lattice.json, fig_p1_13_lattice.png
"""
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import OptimizeWarning, curve_fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "lib"), str(ROOT / "code" / "P1")]
import p1_3_run as run  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED = 20261113
B_FD, B_TH = 400, 150
BMIN = (4, 6, 8)          # нижня межа бінів журналу, s ≥ 2^BMIN
NAMES = ("2D bond sq", "2D bond tri", "2D site sq", "3D bond", "4D bond")
an = json.load(open(ROOT / "results" / "P1" / "p1_3_analysis.json"))["lattices"]
EXP = {}
for r in csv.DictReader(open(ROOT / "data" / "P1" / "constants.csv")):
    if r["system"].startswith("perc_") and r["quantity"] in ("beta", "nu", "d_f", "d_min"):
        EXP.setdefault(int(r["d"]), {})[r["quantity"]] = float(r["value"])


def chunks(name, L, keys):
    ps = np.array(json.load(open(run.RES / f"{run.SLUG[name]}_L{L}.json"))["ps"])
    return ps, run.load(run.DATA / run.SLUG[name] / f"L{L}", keys)


def derivs(name, L, pc, rng):
    """NSC′ і NSC″ у вузлі сітки, найближчому до pc (нерівномірна тристочкова схема), бутстреп."""
    ps, S = chunks(name, L, ["nsc"])
    Y = S["nsc"]
    k = int(np.argmin(np.abs(ps - pc)))
    h1, h2 = ps[k] - ps[k - 1], ps[k + 1] - ps[k]

    def fd(Y):
        y = Y.mean(0)
        return ((y[k + 1] - y[k - 1]) / (h1 + h2), 2 * ((y[k + 1] - y[k]) / h2 - (y[k] - y[k - 1]) / h1) / (h1 + h2))
    e = fd(Y)
    bs = np.array([fd(Y[rng.integers(0, len(Y), len(Y))]) for _ in range(B_FD)])
    return dict(L=L, R=len(Y), p=float(ps[k]), h=float(h2), d1=float(e[0]), se_d1=float(bs[:, 0].std(ddof=1)),
                d2=float(e[1]), se_d2=float(bs[:, 1].std(ddof=1)))


def journal(name, L, pc, rng):
    """g(s; pc) за бінами (лінійна інтерполяція між вузлами сітки навколо pc) і підгонка g∞ − a·s^(−θ)."""
    ps, S = chunks(name, L, ["sbin_n", "sbin_sc", "sbin_c", "g_giant"])
    k = np.searchsorted(ps, pc) - 1
    w = (pc - ps[k]) / (ps[k + 1] - ps[k])
    at = {key: (1 - w) * S[key][:, k] + w * S[key][:, k + 1] for key in S}
    f = lambda s, g, a, t: g - a * s ** -t  # noqa: E731

    def bins(idx):
        n, sc, c = at["sbin_n"][idx].sum(0), at["sbin_sc"][idx].sum(0), at["sbin_c"][idx].sum(0)
        ok = c >= 30
        return np.flatnonzero(ok), n[ok] / c[ok], sc[ok] / n[ok]

    def fit(idx, bmin):
        b, s, g = bins(idx)
        sel = b >= bmin
        try:
            return curve_fit(f, s[sel], g[sel], p0=[g[sel][-1], 1.0, 0.5], maxfev=20000)[0]
        except (RuntimeError, ValueError, TypeError):
            return np.full(3, np.nan)
    allidx = np.arange(len(at["sbin_n"]))
    b, s, g = bins(allidx)
    out = dict(L=L, bins=b.tolist(), s_mean=s.tolist(), g=g.tolist(), g_giant_pc=float(at["g_giant"].mean()), fits={})
    for bmin in BMIN:
        e = fit(allidx, bmin)
        bs = np.array([fit(rng.integers(0, len(allidx), len(allidx)), bmin) for _ in range(B_TH)])
        out["fits"][str(bmin)] = dict(g_inf=float(e[0]), se_g_inf=float(np.nanstd(bs[:, 0], ddof=1)), a=float(e[1]),
                                      theta=float(e[2]), se_theta=float(np.nanstd(bs[:, 2], ddof=1)),
                                      fail=float(np.mean(~np.isfinite(bs[:, 2]))))
    return out


MODELS = {"pure": (lambda e, A, x: A * e ** x, [1.0, 0.7]),
          "lin_bg": (lambda e, A, x, B: A * e ** x + B * e, [1.0, 0.7, 0.0]),
          "mult": (lambda e, A, x, c: A * e ** x * (1 + c * e), [1.0, 0.7, 0.0])}


def giant_branch(name, Ls, pc, ginf):
    """g∞ − g_G(p) вище порога; точки, де дві найбільші L збігаються в межах 3σ + 0.002."""
    (ps, S1), (_, S2) = chunks(name, Ls[0], ["g_giant"]), chunks(name, Ls[1], ["g_giant"])
    g1, g2 = S1["g_giant"].mean(0), S2["g_giant"].mean(0)
    s1 = S1["g_giant"].std(0, ddof=1) / np.sqrt(len(S1["g_giant"]))
    s2 = S2["g_giant"].std(0, ddof=1) / np.sqrt(len(S2["g_giant"]))
    eps = (ps - pc) / pc
    ok = (eps > 0) & (np.abs(g2 - g1) < 3 * np.hypot(s1, s2) + 0.002)
    e, D, sD = eps[ok], ginf - g2[ok], s2[ok] + 1e-4
    out = dict(L=list(Ls), eps=e.tolist(), D=D.tolist(), se=sD.tolist(), fits={})
    for m, (f, p0) in MODELS.items():
        try:
            c, cov = curve_fit(f, e, D, p0=p0, sigma=sD, maxfev=50000)
            r = (D - f(e, *c)) / sD
            out["fits"][m] = dict(par=c.tolist(), x=float(c[1]), se_x=float(np.sqrt(cov[1, 1])),
                                  chi2dof=float(np.sum(r ** 2) / (len(e) - len(c))))
        except RuntimeError:
            out["fits"][m] = None
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    for name in NAMES:
        v = an[name]
        pc, d = v["pc"], v["d"]
        Ls = [r["L"] for r in v["rows"]]
        ex = EXP[d]
        hyp = dict(theta=1 - ex["d_min"] / ex["d_f"], x=ex["nu"] * (ex["d_f"] - ex["d_min"])) if "d_min" in ex else {}
        R = dict(d=d, pc=pc, exponents=ex, hypothesis=hyp)
        R["derivs"] = [derivs(name, L, pc, rng) for L in Ls]
        R["journal"] = [journal(name, L, pc, rng) for L in Ls[-3:]]
        ginf = R["journal"][-1]["fits"]["6"]["g_inf"]
        R["giant"] = giant_branch(name, Ls[-2:], pc, ginf) if len(Ls) >= 2 else None
        xs = [f["x"] for f in R["giant"]["fits"].values() if f and f["chi2dof"] < 5]
        R["psi_range"] = [ex["beta"] + min(xs), ex["beta"] + max(xs)] if xs else None
        res[name] = R
        j = R["journal"][-1]["fits"]
        print(f"{name}: θ = " + ", ".join(f"{j[b]['theta']:.3f}±{j[b]['se_theta']:.3f} (s≥2^{b})" for b in j)
              + f"; g∞ = {ginf:.4f}, g_G(pc) = {R['journal'][-1]['g_giant_pc']:.4f}; гіпотеза θ = {hyp.get('theta', np.nan):.3f}")
        print("   x: " + ", ".join(f"{m} {f['x']:.3f}±{f['se_x']:.3f} (χ²/dof {f['chi2dof']:.2f})"
                                    for m, f in R["giant"]["fits"].items() if f)
              + f"; гіпотеза x = {hyp.get('x', np.nan):.3f}; ψ ∈ {R['psi_range']}")
        print("   NSC′(pc, L): " + ", ".join(f"{q['L']}: {q['d1']:.2f}±{q['se_d1']:.2f}" for q in R["derivs"]))
        print("   NSC″(pc, L): " + ", ".join(f"{q['L']}: {q['d2']:.0f}±{q['se_d2']:.0f}" for q in R["derivs"]), flush=True)
    save(ROOT / "results" / "P1" / "p1_13_lattice.json",
         runmeta(__file__, params=dict(B_fd=B_FD, B_theta=B_TH, bmin=BMIN, models=list(MODELS)), seeds=[SEED]), lattices=res)
    figure(res)


def figure(res):
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
                         "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})
    COL = {"2D bond sq": "#2a78d6", "2D bond tri": "#1baf7a", "2D site sq": "#eb6834", "3D bond": "#7b5cd6", "4D bond": "#d64545"}
    LAB = {"2D bond sq": "2D квадратна (зв.)", "2D bond tri": "2D трикутна", "2D site sq": "2D квадратна (вузли)",
           "3D bond": "3D", "4D bond": "4D"}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    for name, R in res.items():
        J = R["journal"][-1]
        ginf = J["fits"]["6"]["g_inf"]
        s, g = np.array(J["s_mean"]), np.array(J["g"])
        m = s >= 4
        a.plot(s[m], ginf - g[m], "o-", color=COL[name], ms=4, lw=1.3, label=f"{LAB[name]}, L = {J['L']}")
        G = R["giant"]
        if G and name != "2D site sq":
            b.errorbar(G["eps"], G["D"], G["se"], fmt="o", color=COL[name], ms=4, capsize=0, label=LAB[name])
            e = np.geomspace(min(G["eps"]), max(G["eps"]), 50)
            for mod, ls in (("lin_bg", "-"), ("mult", ":")):
                f = G["fits"].get(mod)
                if f:
                    b.plot(e, MODELS[mod][0](e, *f["par"]), ls, color=COL[name], lw=1)
        L = [q["L"] for q in R["derivs"]]
        c.errorbar(L, [q["d1"] for q in R["derivs"]], [q["se_d1"] for q in R["derivs"]], fmt="o-", color=COL[name],
                   ms=4, lw=1.3, capsize=2, label=LAB[name])
    s0 = np.array([30, 3e4])
    for th, ls in ((0.4035, "--"), (0.65, ":")):
        a.plot(s0, 0.5 * (s0 / 30) ** -th, ls, color=MUTED, lw=1)
        a.text(s0[1], 0.5 * (s0[1] / 30) ** -th, f" s^−{th:.2f}", color=MUTED, fontsize=8, va="center")
    a.set(xscale="log", yscale="log", xlabel="розмір кластера s", ylabel="g∞ − g(s; pc)",
          title="Скінченні кластери при pc: g(s) → g∞\n(штрих: гіпотеза θ = 1 − d_min/d_f для 2D)")
    a.legend(fontsize=7.5, frameon=False, loc="lower left")
    b.set(xscale="log", yscale="log", xlabel="ε = (p − pc)/pc", ylabel="g∞ − g_G(p)",
          title="Гігант вище порога: g∞ − g_G = A·ε^x + фон\n(лінія — лінійний фон, пунктир — множний)")
    b.legend(fontsize=8, frameon=False, loc="upper left")
    c.axhline(0, color=MUTED, lw=1)
    c.set(xscale="log", xlabel="розмір решітки L", ylabel="NSC′ у вузлі сітки біля pc",
          title="Нахил NSC у порозі (скінченні різниці,\nкрок 0.0017 у 2D, 0.0028 у 3D)")
    c.legend(fontsize=8, frameon=False, loc="center right")
    fig.savefig(ROOT / "results" / "P1" / "fig_p1_13_lattice.png", dpi=150)


if __name__ == "__main__":
    main()
