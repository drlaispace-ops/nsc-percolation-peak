"""П1.9. Рисунок: NSC(c) для ER із точками exp10 і c*(ширина вікна) для параболи й кубіки."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_9_cstar import ROOT, nsc_quad  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
d = json.load(open(ROOT / "results" / "P1" / "p1_9_cstar.json"))["peak"]["ER"]
e = json.load(open(ROOT / "results" / "exp10_gw_fine.json"))["ER"]
xs, ys, se = map(np.array, (e["x"], e["nsc"], e["se"]))
fine = np.linspace(1.26, 1.54, 141)
exact = np.array([nsc_quad("ER", c) for c in fine])
par = np.polyfit(xs, np.array([nsc_quad("ER", c) for c in xs]), 2)

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
a.plot(fine, np.polyval(par, fine), color=ORANGE, lw=2, label="Парабола на точних значеннях, вікно exp10")
a.plot(fine, exact, color=BLUE, lw=2, label="Точна крива (квадратура)")
a.errorbar(xs, ys, yerr=se, fmt="o", ms=5, color=AQUA, mec="#fcfcfb", mew=1, ecolor=AQUA, elinewidth=1, label="Монте-Карло exp10, ±1 SE")
a.axvline(d["xstar"], color=BLUE, lw=1)
a.axvline(d["recorded"], color=ORANGE, lw=1)
a.annotate(f"пік кривої\nc* = {d['xstar']:.4f}", (d["xstar"], 0.5812), xytext=(-6, 0), textcoords="offset points", ha="right", color=INK)
a.annotate(f"записано\nc* = {d['recorded']:.4f}", (d["recorded"], 0.5812), xytext=(6, 0), textcoords="offset points", ha="left", color=INK)
a.set(xlabel="c", ylabel="NSC")
a.set_title("Крива асиметрична, тому парабола зсуває пік", loc="left", color=INK, fontsize=11)
a.legend(frameon=False, loc="lower center", fontsize=9, bbox_to_anchor=(0.54, 0.0))

hw = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.14])
res = {"par": [], "cub": []}
for w in hw:
    g = np.linspace(1.40 - w, 1.40 + w, 15)
    v = np.array([nsc_quad("ER", c) for c in g])
    gc = g - 1.40
    p2, p3 = np.polyfit(gc, v, 2), np.polyfit(gc, v, 3)
    res["par"].append(1.40 - p2[1] / (2 * p2[0]))
    r = np.roots(np.polyder(p3))
    r = r[np.isreal(r)].real
    res["cub"].append(1.40 + r[np.argmin(np.abs(r))])
b.axhline(d["xstar"], color=BLUE, lw=2, label="Нуль похідної точної кривої")
b.plot(hw, res["par"], "-o", color=ORANGE, lw=2, ms=6, mec="#fcfcfb", label="Парабола")
b.plot(hw, res["cub"], "-o", color=AQUA, lw=2, ms=6, mec="#fcfcfb", label="Кубіка")
b.set(xlabel="півширина вікна навколо c = 1.40", ylabel="c*")
b.set_title("Зміщення параболи росте з шириною вікна", loc="left", color=INK, fontsize=11)
b.legend(frameon=False, fontsize=9, loc="upper left")
out = ROOT / "results" / "P1" / "fig_p1_9_cstar.png"
fig.savefig(out, dpi=170)
print(out, [round(x, 5) for x in res["par"]], [round(x, 5) for x in res["cub"]])
