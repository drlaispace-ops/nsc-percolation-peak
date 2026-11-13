"""П1.2. Рисунки: похибка від M (p1_2_validate.json); час від n і модель (p1_2_cost*.json); бюджет П1.3 (p1_2_budget.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "P1"
BLUE, ORANGE, AQUA, VIOLET, RED, GOLD, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#7b5cd6", "#d64545", "#c9a227", "#0b0b0b", "#52514e", "#e6e5e1"
COLS = [BLUE, ORANGE, AQUA, VIOLET, RED, GOLD, MUTED]
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
val = json.load(open(RES / "p1_2_validate.json"))
cost = json.load(open(RES / "p1_2_cost.json"))["rows"]
cm = json.load(open(RES / "p1_2_cost_model.json"))
bud = json.load(open(RES / "p1_2_budget.json"))


def per_work_ns(th, N):
    N = np.asarray(N, float)
    return th[2] + th[3] * np.maximum(0, np.log(N) - th[4]) + th[5] * np.maximum(0, np.log(N) - th[6])


fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
# (а) похибка від M
for (name, v), col in zip(val["models"].items(), COLS):
    Ms = np.array([int(m) for m in v["by_M"]])
    rms = np.array([v["by_M"][str(m)]["nsc"]["rms_err"] for m in Ms])
    se = np.array([v["by_M"][str(m)]["nsc"]["mean_se"] for m in Ms])
    a.plot(Ms, rms, "o-", color=col, lw=2, ms=6, mec="#fcfcfb", label=name)
    a.plot(Ms, se, ":", color=col, lw=1.5)
m = np.array([50, 500])
a.plot(m, 0.03 * np.sqrt(50 / m), color=INK, lw=1, ls="--", label=r"$\propto M^{-1/2}$")
a.set(xscale="log", yscale="log", xlabel="джерел на велику компоненту, M", ylabel="похибка NSC",
      title="Похибка від M: факт (лінія з точками)\nі оцінена SE (пунктир), 240 графів на модель")
a.set_xticks([50, 100, 200, 500], ["50", "100", "200", "500"])
a.minorticks_off()
a.legend(fontsize=7.5, frameon=False, loc="lower left")
# (б) час від n
for d, col in ((2, BLUE), (3, ORANGE)):
    th = cm["model"][str(d)]["theta"]
    cg = cm["model"][str(d)]["c_gen"]
    for p, mk in sorted({(r["p"], "o" if r["p"] in (0.5, 0.2488) else "s") for r in cost if r["d"] == d}):
        rs = sorted([r for r in cost if r["d"] == d and r["p"] == p], key=lambda r: r["n"])
        n = np.array([r["n"] for r in rs])
        t = np.array([r["t_gen"] + r["t_est"] for r in rs])
        pred = np.array([cg * r["n"] + th[0] * r["n"] + 1e-9 * (th[1] * r["work_exact"] + np.sum(per_work_ns(th, r["big_N"]) * np.array(r["big_W"])))
                         for r in rs])
        b.plot(n, t, mk, color=col, ms=6, mec="#fcfcfb", label=f"{d}D, p = {p}")
        b.plot(n, pred, "-" if mk == "o" else "--", color=col, lw=1.5)
b.set(xscale="log", yscale="log", xlabel="вузлів n", ylabel="секунд на граф (M = 500)",
      title="Час від n: заміри (точки) і модель (лінії)\nціна BFS росте, коли компонента виходить за кеш")
b.legend(fontsize=8, frameon=False, loc="upper left")
# (в) бюджет
names = list(bud["lattices"])
left = np.zeros(len(names))
Lsets = []
for i, nm in enumerate(names):
    for r in bud["lattices"][nm]["plan"]:
        c.barh(i, r["hours"], left=left[i], color=ORANGE if r["extrapolated"] else BLUE, edgecolor="#fcfcfb", lw=1)
        left[i] += r["hours"]
    c.text(left[i] + 1, i, f"{left[i]:.1f} год", va="center", fontsize=8.5, color=INK)
c.set_yticks(range(len(names)), names)
c.invert_yaxis()
c.axvline(300, color=RED, lw=1)
c.text(298, len(names) - 0.6, "поріг рішення 300 год", ha="right", color=RED, fontsize=8.5)
c.bar(0, 0, color=BLUE, label="L у межах пілоту")
c.bar(0, 0, color=ORANGE, label="L екстрапольовано")
c.legend(fontsize=8, frameon=False, loc="center right")
c.set(xlabel="годин (Mac mini M4)", title=f"Прогноз бюджету П1.3: разом {bud['total_hours']:.0f} год\n(сегменти — розміри L)")
c.grid(axis="y", visible=False)
fig.savefig(RES / "fig_p1_2_estimator.png", dpi=150)
print("ok")
