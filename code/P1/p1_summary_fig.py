"""Підсумковий рисунок для звіту: внески гіганта і скінченних кластерів у NSC для трьох моделей."""
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
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})
j11 = json.load(open(OUT / "p1_11_er.json"))
j12 = json.load(open(OUT / "p1_12_laws.json"))
MODELS = [("ER", "Випадковий граф (ER)", "c, середній степінь", 1.0, 1.40381, 0.5, 3.0),
          ("z4", "Регулярний граф, z = 4", "p, частка відкритих ребер", 1 / 3, j12["rrg"]["z4"]["pstar"], 0.2, 0.75),
          ("z3", "Регулярний граф, z = 3", "p, частка відкритих ребер", 0.5, j12["rrg"]["z3"]["pstar"], 0.3, 0.95)]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
for ax, (m, name, xlab, xc, xs, lo, hi) in zip(axes, MODELS):
    below = np.linspace(lo, xc, 40, endpoint=False)
    above = xc + (hi - xc) * np.linspace(0.003, 1, 110) ** 1.6
    x, tot, fin = [], [], []
    for v in below:
        n = gx.pair_quad(m, float(v))
        x.append(v); tot.append(n); fin.append(n)
    n0 = gx.pair_quad(m, float(xc))
    x.append(xc); tot.append(n0); fin.append(n0)
    for v in above:
        n = gx.pair_quad(m, float(v))
        x.append(v); tot.append(n); fin.append(n - gx.pair_quad(m, float(v), survive=True))
    x, tot, fin = map(np.array, (x, tot, fin))
    gi = tot - fin
    ax.fill_between(x, 0, gi, color=GREEN, alpha=0.16, lw=0)
    ax.fill_between(x, 0, fin, color=ORANGE, alpha=0.16, lw=0)
    ax.plot(x, tot, color=BLUE, lw=2.4, label="NSC, уся складність")
    ax.plot(x, gi, color=GREEN, lw=2, label="гігантський кластер")
    ax.plot(x, fin, color=ORANGE, lw=2, label="малі кластери")
    nsc_s, gi_s = gx.pair_quad(m, float(xs)), gx.pair_quad(m, float(xs), survive=True)
    ax.plot([xs, xs], [0, nsc_s], color=MUTED, lw=1, ls="--")
    ax.plot([xs], [nsc_s], "o", color=BLUE, ms=8, mec=SURF, mew=1.6)
    ax.annotate(f"пік: гігант дає {gi_s / nsc_s:.0%}", (xs, nsc_s), xytext=(10, 8), textcoords="offset points", color=INK, fontsize=11)
    ax.set(xlabel=xlab, xlim=(lo, hi), ylim=(0, tot.max() * 1.17))
    ax.set_title(name, loc="left", color=INK, fontsize=12)
axes[0].set_ylabel("внесок у складність")
axes[0].legend(frameon=False, fontsize=10, loc="center right")
fig.savefig(OUT / "fig_gw_summary.png", dpi=180)
