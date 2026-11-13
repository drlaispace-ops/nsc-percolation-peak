"""П1.11. Рисунки: a_n, внесок за розміром, частини NSC; мапи δ_k і координати."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
OUT = ROOT / "results" / "P1"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})
d = json.load(open(OUT / "p1_11_er.json"))
a = np.load(OUT / "p1_11_a_table.npy")
ainf = d["a_table"]["ainf"]


def title(ax, t):
    ax.set_title(t, loc="left", color=INK, fontsize=11)


fig, (x1, x2, x3) = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
n = np.arange(3, len(a))
x1.semilogx(n, a[3:], color=COL[0], lw=2)
x1.axhline(ainf, color=MUTED, lw=1)
x1.annotate(f"a∞ = {ainf:.5f}", (3, ainf), xytext=(0, -14), textcoords="offset points", color=INK)
x1.set(xlabel="розмір кластера n", ylabel="aₙ, середня NodeSC")
title(x1, "Великі кластери виходять на сталу")

from gwexact import borel, dual, giant  # noqa: E402

for i, (c, lab) in enumerate([(0.8, "c = 0.8"), (1.0, "c = 1"), (1.2, "c = 1.2"), (1.40381, "c* = 1.404"), (2.0, "c = 2")]):
    b = borel(len(a) - 1, c) * a
    fp = d["size_contrib"][str(c)]["finite_part"]
    x2.semilogx(np.arange(len(a))[1:], np.cumsum(b)[1:] / fp, color=COL[i], lw=2, label=lab)
x2.set(xlabel="n, кластери не більші за n", ylabel="частка скінченної частини NSC", ylim=(0, 1.02))
x2.legend(frameon=False, fontsize=9, loc="lower right")
title(x2, "Внесок скінченних кластерів за розміром")

# сітка розрахунку обходить поріг; густі точки навколо c = 1 щоб не зрізати максимум скінченної частини
from gwexact import decompose  # noqa: E402

extra = [decompose(float(v)) for v in (0.96, 0.97, 0.98, 0.99, 0.995, 1.005, 1.01)]
at1 = d["crit"]["g_fin_1"]
extra.append(dict(c=1.0, nsc=at1, part_fin=at1, part_inf=0.0))
p = sorted(d["parts"] + extra, key=lambda x: x["c"])
c = np.array([x["c"] for x in p])
x3.plot(c, [x["nsc"] for x in p], color=COL[0], lw=2, label="NSC")
x3.plot(c, [x["part_fin"] for x in p], color=COL[1], lw=2, label="(1−P)·g_fin, скінченні")
x3.plot(c, [x["part_inf"] for x in p], color=COL[2], lw=2, label="P·g_∞, гігант")
x3.axvline(1, color=GRID, lw=1)
x3.axvline(1.40381, color=MUTED, lw=1)
x3.annotate("c*", (1.40381, 0.03), xytext=(4, 0), textcoords="offset points", color=INK)
x3.set(xlabel="c", ylabel="внесок у NSC")
x3.legend(frameon=False, fontsize=9, loc="upper right")
title(x3, "У піку гігант дає 84% NSC")
fig.savefig(OUT / "fig_p1_11_decomposition.png", dpi=170)

blue = LinearSegmentedColormap.from_list("b", ["#f2f6fc", "#86b6ef", "#2a78d6", "#123a6b"])
s = d["scales"]
cs, pf, pi = np.array(s["c"]), np.array(s["part_fin"]), np.array(s["part_inf"])
K = s["K"]
fig, (m1, m2, x4) = plt.subplots(1, 3, figsize=(15.5, 4.3), constrained_layout=True)
vmax = max(pf[:, :K - 1].max(), pi[:, :K - 1].max())
for ax, M, t in ((m1, pf, "Скінченні кластери: внесок масштабу k"), (m2, pi, "Гігант: внесок масштабу k")):
    q = ax.pcolormesh(cs, np.arange(1, K), M[:, :K - 1].T, cmap=blue, vmin=0, vmax=vmax, shading="nearest")
    ax.set_yscale("log")
    ax.set(xlabel="c", ylabel="k")
    ax.grid(False)
    ax.axvline(1.40381, color=COL[1], lw=1)
    title(ax, t)
fig.colorbar(q, ax=[m1, m2], label="E[δₖ], внесок у NSC за вузлами", shrink=0.9)
co = d["coords"]
lab = ["1", "2", "3–4", "5–8", "9–16", "17–32", "33–64", ">64"]
for i, nn in enumerate(("20", "100", "500")):
    v = co[nn]
    x4.errorbar(range(8), v["delta_share"], yerr=v["delta_share_se"], color=COL[i], lw=2, marker="o", ms=5, mec=SURF, label=f"n = {nn}")
x4.axhline(0, color=MUTED, lw=1)
x4.set_xticks(range(8), lab)
x4.set(xlabel="масштаб k", ylabel="частка Var(C) за фіксованого n")
x4.legend(frameon=False, fontsize=9)
title(x4, "Розкид C задають k порядку діаметра")
fig.savefig(OUT / "fig_p1_11_scales.png", dpi=170)
