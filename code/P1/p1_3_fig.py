"""П1.3. Рисунок для звіту (проміжний, з results/P1/p1_3_analysis.json і підсумків задач).

(а) P(p) для трьох решіток на найбільшому готовому L, точка — пік NSC: де на кривій гіганта стоїть пік;
(б) відносний зсув піку (p* − pc)/pc від L; (в) частка гіганта в піку P* від L.
Запуск: .venv/bin/python code/P1/p1_3_fig.py  →  results/P1/fig_p1_3_report.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "P1"
an = json.load(open(RES / "p1_3_analysis.json"))["lattices"]
COL = {"2D bond sq": "#2a78d6", "2D site sq": "#eb6834", "2D bond tri": "#1baf7a", "3D bond": "#7b5cd6",
       "4D bond": "#d64545", "5D bond": "#c9a227", "6D bond": "#52514e"}
LAB = {"2D bond sq": "2D квадратна (зв.)", "2D site sq": "2D квадратна (вузли)", "2D bond tri": "2D трикутна",
       "3D bond": "3D", "4D bond": "4D", "5D bond": "5D", "6D bond": "6D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for name in ("2D bond sq", "3D bond", "4D bond"):
    v = an[name]
    r = v["rows"][-1]
    sm = json.load(open(RES / "p1_3" / f"{name.replace(' ', '_')}_L{r['L']}.json"))
    ps, P = np.array(sm["ps"]), np.array(sm["mean"]["P"])
    x = (ps - v["pc"]) / v["pc"]
    a.plot(x, P, "-", color=COL[name], lw=2, label=f"{LAB[name]}, L = {r['L']}")
    xs = (r["all"]["p_star"] - v["pc"]) / v["pc"]
    a.plot(xs, r["all"]["P_star"], "o", color=COL[name], ms=9, mec="white", mew=1.5, zorder=5)
    a.annotate(f"P* = {r['all']['P_star']:.2f}", (xs, r["all"]["P_star"]), textcoords="offset points",
               xytext=(8, -14), fontsize=9, color=COL[name])
a.axhline(0.5, color=MUTED, lw=1, ls="--")
a.axvline(0, color=MUTED, lw=1)
a.set(xlim=(-0.15, 0.35), ylim=(0, 0.85), xlabel="(p − pc)/pc", ylabel="частка вузлів у гіганті P",
      title="Де на кривій гіганта стоїть пік NSC (точки)")
a.legend(fontsize=8.5, frameon=False, loc="lower right")
for name, v in an.items():
    if not v["rows"]:
        continue
    L = np.array([r["L"] for r in v["rows"]])
    ps = np.array([r["all"]["p_star"] for r in v["rows"]])
    s = np.hypot([r["all"]["se_p_star"] for r in v["rows"]], [r["all"]["syst_window"] for r in v["rows"]])
    P = np.array([r["all"]["P_star"] for r in v["rows"]])
    sP = np.array([r["all"]["se_P_star"] for r in v["rows"]])
    kw = dict(fmt="o-", color=COL[name], ms=5, lw=1.5, capsize=2, label=LAB[name])
    b.errorbar(L, (ps - v["pc"]) / v["pc"], s / v["pc"], **kw)
    c.errorbar(L, P, sP, **kw)
b.axhline(0, color=MUTED, lw=1)
b.set(xscale="log", xlabel="розмір решітки L", ylabel="(p* − pc)/pc",
      title="Відносний зсув піку від порогу\n(похибка: бутстреп ⊕ залежність від вікна)")
b.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
c.axhline(0.5, color=MUTED, lw=1, ls="--")
c.set(xscale="log", xlabel="розмір решітки L", ylabel="P*", title="Частка гіганта в піку NSC")
fig.savefig(RES / "fig_p1_3_report.png", dpi=150)
print("ok")
sys.exit(0)
