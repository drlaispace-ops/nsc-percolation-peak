"""П1.3c. Рисунок: кактус-Бете з z = 12 проти ГЦК.
(а) δ = p*/p*_Бете − 1 проти t (трикутників на вузол) для двох сімей дерев клік; ГЦК (t = 24) — діапазон оцінювачів
    і вузьковіконна оцінка з похибкою (results/P1/p1_3b_bethe_lattices.json, notes/orchestra/A_bethe_findings.md);
(б) s = p*/pc − 1 проти t: у кактусах пік стоїть на ~27–34% над власним pc, у ГЦК — на 5.4%;
(в) NSC_∞(p) для Бете, K5×3, K7×2 і точка піку ГЦК.

    .venv/bin/python code/P1/p1_3c_fig.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / "results/P1/p1_3c_cactus.json").read_text())
A = json.loads((ROOT / "results/P1/p1_3b_bethe_lattices.json").read_text())["lattices"]["fcc"]
V = json.loads((ROOT / "notes/orchestra/verify_A.json").read_text())["peaks"]["fcc"]
# діапазон δ ГЦК за оцінювачами: вузьковіконна кубіка (J) і сплайн / парабола ±0.006 / квартика ±0.02 (verify_A.json)
_ests = [A["vs_bethe"]["narrow"]["rel_p_inf"]] + [V[k]["rel_p_inf"] for k in ("spline", "quad006", "quartic02")]
FCC_RANGE = (min(_ests), max(_ests))
COL = {"homog": "#1f5fa8", "gleeson": "#c8582a"}
LAB = {"homog": "k клік K$_n$ на вузол", "gleeson": "одна кліка K$_c$ + ребра (Gleeson)"}


def fam_rows(f):
    return sorted(R["families"][f].values(), key=lambda r: r["t"])


fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))
vs = A["vs_bethe"]["narrow"]
d_fcc, e_fcc = vs["rel_p_inf"], vs["err_rel_p_inf"]
s_fcc, es_fcc = A["vs_bethe"]["decomposition"]["s"], A["vs_bethe"]["decomposition"]["err_s"]
for f in ("homog", "gleeson"):
    rows = fam_rows(f)
    t = [r["t"] for r in rows]
    ax[0].plot(t, [100 * r["delta"] for r in rows], "o-", color=COL[f], ms=5, lw=1.6, label=LAB[f])
    ax[1].plot(t, [100 * r["s"] for r in rows], "o-", color=COL[f], ms=5, lw=1.6, label=LAB[f])
    i24 = R["interp_t24"][f]
    ax[0].plot([24], [100 * i24["delta"]], "D", mfc="white", mec=COL[f], ms=7, zorder=5)
ax[0].axhspan(100 * FCC_RANGE[0], 100 * FCC_RANGE[1], xmin=0, xmax=1, color="0.85", zorder=0)
ax[0].errorbar([25], [100 * d_fcc], yerr=[100 * e_fcc], fmt="s", color="k", ms=7, capsize=3, zorder=6,
               label="ГЦК (t = 24; зміщено вправо), вузьковіконна оцінка")
ax[0].text(0.5, 100 * FCC_RANGE[1] + 0.2, "діапазон ГЦК за оцінювачами", fontsize=8, color="0.35")
ax[0].set_xlabel("t — трикутників на вузол")
ax[0].set_ylabel("δ = p*/p*$_{Бете}$ − 1, %")
ax[0].set_title("(а) зсув піку від Бете з z = 12")
ax[0].legend(fontsize=8, loc="upper left")
ax[0].set_xlim(-1, 40)
ax[1].errorbar([24], [100 * s_fcc], yerr=[100 * es_fcc], fmt="s", color="k", ms=7, capsize=3, label="ГЦК")
ax[1].set_xlabel("t — трикутників на вузол")
ax[1].set_ylabel("s = p*/pc − 1, %")
ax[1].set_title("(б) відстань піку від власного порогу")
ax[1].set_ylim(0, 40)
ax[1].legend(fontsize=8, loc="center left")
ax[1].set_xlim(-1, 40)
cur = [("Бете z = 12", R["bethe"], "0.2", "-")]
for key, ls in (("5", "--"), ("7", ":")):
    cur.append((f"K$_{key}$ × {12 // (int(key) - 1)}", R["families"]["homog"][key], COL["homog"], ls))
for lab, r, c, ls in cur:
    ax[2].plot(r["curve"]["p"], r["curve"]["nsc"], ls, color=c, lw=1.6, label=lab)
    ax[2].plot([r["p_star"]], [r["nsc_star"]], "o", color=c, ms=5)
ax[2].axvline(A["narrow"]["p_inf"], color="k", lw=1, ls="-.", label="p∞ ГЦК")
ax[2].axvline(A["pc"], color="0.5", lw=1, ls=(0, (1, 2)), label="pc ГЦК")
ax[2].set_xlabel("p")
ax[2].set_ylabel("NSC$_\\infty$")
ax[2].set_title("(в) NSC$_\\infty$(p) дерев клік")
ax[2].legend(fontsize=8, loc="lower right")
fig.tight_layout()
out = ROOT / "results/P1/fig_p1_3c_cactus.png"
fig.savefig(out, dpi=160)
print(out)
