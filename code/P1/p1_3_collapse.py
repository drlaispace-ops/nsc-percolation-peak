"""П1.3. Колапс даних у скейлінговій змінній x = (p − pc)·L^(1/ν) для 2D bond-квадратної і 3D bond.

Контроль: P·L^(β/ν) від x має колапсувати за стандартним FSS (перевірка даних і констант).
Питання: чи колапсує NSC(p, L) у тій самій змінній без перемасштабування (тоді пік NSC — порогова величина),
чи криві за різних L лише зсуваються (пік прив'язаний не до порога).
Запуск: .venv/bin/python code/P1/p1_3_collapse.py → results/P1/fig_p1_3_collapse.png
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
an = json.load(open(RES / "p1_3_analysis.json"))
BN = an["meta"]["params"]["beta_nu"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})
CASES = [("2D bond sq", "2D квадратна", 2), ("3D bond", "3D", 3)]


def main():
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    for col, (name, lab, d) in enumerate(CASES):
        v = an["lattices"][name]
        beta, nu = BN[str(d)]["beta"], BN[str(d)]["nu"]
        Ls = [r["L"] for r in v["rows"]]
        cmap = plt.get_cmap("Blues")
        for i, L in enumerate(Ls):
            s = json.load(open(RES / "p1_3" / f"{name.replace(' ', '_')}_L{L}.json"))
            ps = np.array(s["ps"])
            x = (ps - v["pc"]) * L ** (1 / nu)
            c = cmap(0.35 + 0.65 * i / max(len(Ls) - 1, 1))
            axs[0, col].plot(x, np.array(s["mean"]["P"]) * L ** (beta / nu), "o-", color=c, ms=3, lw=1.2,
                             label=f"L = {L}")
            axs[1, col].plot(x, s["mean"]["nsc"], "o-", color=c, ms=3, lw=1.2, label=f"L = {L}")
            r = next(r for r in v["rows"] if r["L"] == L)["all"]
            axs[1, col].plot((r["p_star"] - v["pc"]) * L ** (1 / nu), r["nsc_star"], "D", color="#eb6834", ms=5,
                             zorder=5)
        lim = (-4, 6) if d == 2 else (-4, 12)
        for row in (0, 1):
            axs[row, col].set_xlim(lim)
            axs[row, col].axvline(0, color=MUTED, lw=1)
        axs[0, col].set(xlabel="(p − pc)·L^(1/ν)", ylabel="P·L^(β/ν)",
                        title=f"{lab}: контроль — частка гіганта (має колапсувати)")
        axs[1, col].set(xlabel="(p − pc)·L^(1/ν)", ylabel="NSC (на всі вузли)",
                        title=f"{lab}: NSC у тій самій змінній (ромби — p*(L))")
        axs[0, col].legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(RES / "fig_p1_3_collapse.png", dpi=150)
    print("ok")


if __name__ == "__main__":
    main()
