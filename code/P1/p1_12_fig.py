"""П1.12. Рисунки: a_n за законами, корінь проти типового вузла; частини NSC для RRG поруч з ER."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import gwexact as gx  # noqa: E402

OUT = ROOT / "results" / "P1"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})
d = json.load(open(OUT / "p1_12_laws.json"))
T = np.load(OUT / "p1_12_a_tables.npz")
LAB = {"poisson": "Пуассон (ER)", "z3": "RRG z = 3", "z4": "RRG z = 4", "geom": "плоский геометричний, корінь", "bin2": "плоский Bin(2, ½), корінь"}


def title(ax, t):
    ax.set_title(t, loc="left", color=INK, fontsize=11)


fig, (x1, x2) = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
for i, (name, lab) in enumerate(LAB.items()):
    a = T[name]
    n = np.arange(3, len(a))
    x1.semilogx(n, a[3:], color=COL[i], lw=2, label=lab)
    lim = d["ainf"][name]["typ" if name in ("poisson", "z3", "z4") else "plain"]
    x1.plot([len(a) * 1.05], [lim], "o", color=COL[i], ms=7, mec=SURF, mew=1.5)
x1.set(xlabel="розмір дерева n", ylabel="aₙ, середня NodeSC кореня")
x1.legend(frameon=False, fontsize=9, loc="lower right")
title(x1, "aₙ за законами; точки праворуч — межа a∞ із хребта")

sizes = [10, 30, 100, 300, 1000, 3000]
for i, name in enumerate(LAB):
    rows = [d["root_typ"][name][str(n)] for n in sizes]
    x2.errorbar(sizes, [r["diff"] for r in rows], yerr=[2 * r["diff_se"] for r in rows], color=COL[i], lw=2, marker="o", ms=5, capsize=2,
                label=LAB[name].replace(", корінь", ""))
x2.set_xscale("log")
x2.axhline(0, color=MUTED, lw=1)
x2.set(xlabel="розмір дерева n", ylabel="NodeSC кореня − середнє за вузлами")
x2.legend(frameon=False, fontsize=9, loc="upper left", ncols=2)
x2.set_ylim(top=0.17)
title(x2, "Корінь проти типового вузла (смуги ±2σ)")
fig.savefig(OUT / "fig_p1_12_laws.png", dpi=170)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True, sharey=True)
for ax, model in zip(axes, ("z3", "z4")):
    b = d["rrg"][model]
    p = sorted(b["parts"] + [b["crit"]], key=lambda r: r["c"])
    x = np.array([r["c"] for r in p])
    ax.plot(x, [r["nsc"] for r in p], color=COL[0], lw=2, label="NSC")
    ax.plot(x, [r["part_fin"] for r in p], color=COL[1], lw=2, label="(1−P)·g_fin, скінченні")
    ax.plot(x, [r["part_inf"] for r in p], color=COL[2], lw=2, label="P·g_∞, гігант")
    ax.axvline(b["pc"], color=GRID, lw=1)
    ax.axvline(b["pstar"], color=MUTED, lw=1)
    ax.annotate("p*", (b["pstar"], 0.03), xytext=(4, 0), textcoords="offset points", color=INK)
    dp = b["decomp_peak"]
    ax.set(xlabel="p, імовірність відкритого ребра")
    title(ax, f"RRG z = {model[1:]}: у піку гігант дає {dp['part_inf'] / dp['nsc']:.0%} NSC")
axes[0].set_ylabel("внесок у NSC")
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.savefig(OUT / "fig_p1_12_rrg_parts.png", dpi=170)
