"""П1.14 і П1.13 (середнє поле). Рисунки: P(C | n), ширина, нульова модель і контролі, ROC, поведінка біля порога."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "P1"
DATA = ROOT / "data" / "P1" / "p1_14"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})
d = json.load(open(OUT / "p1_14_shapes.json"))


def title(ax, t):
    ax.set_title(t, loc="left", color=INK, fontsize=11)


# --- P(C | n) ---
fig, (x1, x2, x3) = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
for i, n in enumerate((12, 18, 26)):
    z = np.load(DATA / f"exact_{n}.npz")
    h = z["hist"][0]
    w = 80 if n > 12 else 80
    edges = np.linspace(0, 4, len(h) + 1)
    k = len(h) // w
    hh = h[: k * w].reshape(-1, k).sum(1)
    cc = edges[: k * w + 1][::k]
    x1.stairs(hh / np.diff(cc), cc, color=COL[i], lw=2, label=f"n = {n}, точно")
for i, n in enumerate((100, 1000)):
    C = np.load(DATA / f"pool_poisson_{n}.npz")["C"]
    x1.hist(C, bins=60, density=True, histtype="step", color=COL[3 + i], lw=2, label=f"n = {n}, вибірка")
x1.set(xlabel="C, середня NodeSC дерева", ylabel="густина", xlim=(0.3, 2.1))
x1.legend(frameon=False, fontsize=9, loc="upper left")
title(x1, "Пуассон: P(C | n); при малих n нерівний через мало форм")

grid = np.linspace(-4, 5, 90)
for i, n in enumerate((30, 100, 300, 1000, 3000)):
    C = np.load(DATA / f"pool_poisson_{n}.npz")["C"]
    zz = (C - C.mean()) / C.std()
    x2.hist(zz, bins=grid, density=True, histtype="step", color=COL[i], lw=1.8, label=f"n = {n}")
x2.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color=MUTED, lw=1, ls="--", label="гаусіана")
x2.set(xlabel="z = (C − C̄ₙ)/σₙ", ylabel="густина")
x2.legend(frameon=False, fontsize=9, loc="upper right")
title(x2, "Нормована форма; асиметрія спадає повільно")

LAB = {"poisson": "Пуассон", "z4": "RRG z = 4", "geom": "геометричний"}
for i, nm in enumerate(LAB):
    ns = [int(n) for n in d["exact"]]
    x3.loglog(ns, [d["exact"][str(n)][nm]["sd"] for n in ns], color=COL[i], lw=2, label=LAB[nm])
    pn = [int(n) for n in d["pool"]]
    x3.loglog(pn, [d["pool"][str(n)][nm]["sd"] for n in pn], color=COL[i], lw=0, marker="o", ms=6, mec=SURF, mew=1.2)
x3.set(xlabel="n", ylabel="σₙ, стандартне відхилення C")
x3.legend(frameon=False, fontsize=9, loc="lower left")
title(x3, "σₙ: лінія — точно, точки — вибірка")
fig.savefig(OUT / "fig_p1_14_shape.png", dpi=170)

# --- Нульова модель і контролі ---
ns = list(d["cells"])
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
x = np.arange(len(ns))
null = ["нуль: Пуассон", "нуль: z=4", "нуль: геом"]
ax = axes[0]
meth = [("cal_dip", "dip, поріг із нуль-моделі"), ("cal_dbic", "ΔBIC, поріг із нуль-моделі"), ("table_dip", "dip, таблиця"), ("naive_dbic10", "ΔBIC > 10, правило")]
for i, (k, lab) in enumerate(meth):
    v = [np.mean([d["cells"][n]["scen"][s][k] for s in null]) for n in ns]
    ax.plot(x, v, color=COL[i], lw=2, marker="o", ms=6, mec=SURF, mew=1.2, label=lab)
ax.axhline(0.05, color=MUTED, lw=1, ls="--")
ax.set(xticks=x, xticklabels=ns, xlabel="n", ylabel="частка хибних виявлень", ylim=(-0.01, 0.25))
ax.legend(frameon=False, fontsize=9, loc="upper right")
title(ax, "Чистий закон: груп немає (пунктир 5%)")
for ax, s, t in zip(axes[1:], ("суміш Пуассон/геом 50/50", "суміш Пуассон/z=4 50/50"), ("Суміш Пуассон/геометричний 50/50", "Суміш Пуассон/z = 4, 50/50")):
    for i, (k, lab) in enumerate([("cal_dip", "dip за C"), ("cal_dbic", "ΔBIC за C"), ("cal_pc1", "dip, PC1 профілю"), ("cal_pc2", "dip, PC2 профілю")]):
        ax.plot(x, [d["cells"][n]["scen"][s][k] for n in ns], color=COL[i], lw=2, marker="o", ms=6, mec=SURF, mew=1.2, label=lab)
    ax.set(xticks=x, xticklabels=ns, xlabel="n", ylabel="потужність, 1000 дерев", ylim=(-0.03, 1.05))
    ax.legend(frameon=False, fontsize=9, loc="center right")
    title(ax, t)
fig.savefig(OUT / "fig_p1_14_null.png", dpi=170)

# --- ROC ---
rows = {(r["task"], r["n"]): r for r in d["auroc"]}
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True, sharey=True)
for ax, (task, n) in zip(axes, [("T1 Пуассон/геом", 30), ("T2 Пуассон/z=4", 30), ("T3 рекурсивне/геом", 30)]):
    r = rows[(task, n)]
    for i, nm in enumerate(("C", "діаметр", "степені", "NSC", "база+NSC")):
        fpr, tpr = r["roc"][nm]
        ax.plot(fpr, tpr, color=COL[i], lw=2, label=f"{nm}: {r['auc'][nm]:.3f}")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.set(xlabel="хибнопозитивні", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(frameon=False, fontsize=9, loc="lower right", title="AUROC")
    title(ax, f"{task}, n = {n}")
axes[0].set_ylabel("істиннопозитивні")
fig.savefig(OUT / "fig_p1_14_roc.png", dpi=170)

# --- Поріг ---
t = json.load(open(OUT / "p1_13_threshold.json"))
fig, (x1, x2, x3) = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
LABM = {"ER": "ER", "z3": "RRG z = 3", "z4": "RRG z = 4"}
for i, m in enumerate(LABM):
    r = t[m]
    x1.semilogx([g["eps"] for g in r["gap"]], [g["gap_over_q"] for g in r["gap"]], color=COL[i], lw=2, marker="o", ms=6, mec=SURF, mew=1.2, label=LABM[m])
    s = r["slopes"]
    e = [v["eps"] for v in s]
    x2.semilogx(e, [v["slope_up"] / r["slope_pred"] for v in s], color=COL[i], lw=2, marker="o", ms=6, mec=SURF, mew=1.2, label=f"{LABM[m]}, вище")
    x2.semilogx(e, [v["slope_dn"] / r["slope_pred"] for v in s], color=COL[i], lw=2, ls="--", marker="s", ms=5, mec=SURF, mew=1.2, label=f"{LABM[m]}, нижче")
    x3.semilogx(e, [v["E"] / r["jump_half_pred"] for v in s], color=COL[i], lw=2, marker="o", ms=6, mec=SURF, mew=1.2, label=f"{LABM[m]}: стрибок {r['jump_pred']:.0f}")
x1.axhline(-1, color=MUTED, lw=1, ls="--")
x1.set(xlabel="ε, відстань від порога", ylabel="(g_∞ − a∞)/q")
x1.legend(frameon=False, fontsize=9, loc="lower left")
title(x1, "g_∞ = a∞ − q біля порога")
x2.axhline(1, color=MUTED, lw=1, ls="--")
x2.set(xlabel="ε", ylabel="нахил / A(a∞ − NSC(0))")
x2.legend(frameon=False, fontsize=8, loc="lower left", ncols=2)
title(x2, "Нахили з обох боків сходяться: зламу немає")
x3.axhline(1, color=MUTED, lw=1, ls="--")
x3.set(xlabel="ε", ylabel="оцінка / прогноз −4A/σ²")
x3.legend(frameon=False, fontsize=9, loc="lower left")
title(x3, "Стрибок NSC″ = −8A/σ²")
fig.savefig(OUT / "fig_p1_13_threshold.png", dpi=170)
