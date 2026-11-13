"""Стаття 1 (П1): усі рисунки і похідні числа з results/P1/*.json.

Запуск (з кореня проєкту або будь-звідки):
    .venv/bin/python paper/P1/make_figures.py
Вихід:
    paper/P1/figs/fig1_curves.pdf … fig8_classes.pdf  — рисунки статті (вектор);
    paper/P1/numbers.json                              — похідні числа, на які посилається текст
                                                         (кожне з полем "src": звідки і як пораховано).
Нових симуляцій немає: лише читання JSON (і точний підрахунок кількох детермінованих сум
для додатка C через code/lib/nsccore.py). Стиль і кольори — як у code/P1/p1_3_fig.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "P1"
ORC = ROOT / "notes" / "orchestra"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figs"
FIG.mkdir(exist_ok=True)


def load(p):
    with open(p) as f:
        return json.load(f)


AN = load(RES / "p1_3_analysis.json")["lattices"]
BETHE = {r["d"]: r for r in load(RES / "p1_3_bethe.json")["rows"]}
CSTAR = load(RES / "p1_9_cstar.json")
ER = load(RES / "p1_11_er.json")
LAWS = load(RES / "p1_12_laws.json")
THR = load(RES / "p1_13_threshold.json")
LAT13 = load(RES / "p1_13_lattice.json")["lattices"]
A3B = load(RES / "p1_3b_bethe_lattices.json")
VAL = load(RES / "p1_2_validate.json")
CL15 = load(RES / "p1_15_classes.json")
CC = load(ORC / "C_checks.json")
CC2 = load(ORC / "C_checks2.json")
CA = load(RES / "p1_3c_cactus.json")          # П1.3c: дерева клік з z = 12
CV = load(RES / "p1_3c_validate.json")
SNAPJ = load(RES / "p1_3_snapshots_paper.json")  # знімки NodeSC, 2D, L = 256 (масиви — p1_3_snapshots_paper.npz)
SNAPZ = np.load(RES / "p1_3_snapshots_paper.npz")
CX = load(RES / "p1_3c_extra.json")          # П1.3c після рецензії: усі дерева клік з z = 12, N3 = 18…30
VA_ = load(ORC / "verify_A.json")

# ------------------------------------------------------------------ стиль (code/P1/p1_3_fig.py)
COL = {"2D bond sq": "#2a78d6", "2D site sq": "#56b4e9", "2D bond tri": "#1baf7a", "3D bond": "#7b5cd6",
       "4D bond": "#d64545", "5D bond": "#c9a227", "6D bond": "#52514e"}
LAB = {"2D bond sq": "2D квадратна (зв.)", "2D site sq": "2D квадратна (вузли)", "2D bond tri": "2D трикутна",
       "3D bond": "3D кубічна", "4D bond": "4D", "5D bond": "5D", "6D bond": "6D"}
MK = {"2D bond sq": "o", "2D site sq": "s", "2D bond tri": "^", "3D bond": "D", "4D bond": "v", "5D bond": "P",
      "6D bond": "X"}
# решітки П1.3b (A) і відповідники П1.3
A_COL = {"diamond": "#0f8b8d", "bcc": "#a0522d", "fcc": "#e0559a", "sc": COL["3D bond"], "hc4": COL["4D bond"],
         "hc5": COL["5D bond"], "hc6": COL["6D bond"], "sq": COL["2D bond sq"], "tri": COL["2D bond tri"]}
A_LAB = {"diamond": "алмазна", "bcc": "ОЦК", "fcc": "ГЦК", "sc": "3D кубічна", "hc4": "4D", "hc5": "5D",
         "hc6": "6D", "sq": "2D квадратна", "tri": "2D трикутна"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
W1, W2 = 3.3, 6.9          # ширина колонки і сторінки (дюйми)


def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("рисунок", name)


def task(name, L):
    return load(RES / "p1_3" / f"{name.replace(' ', '_')}_L{L}.json")


NUM = {}


def put(key, value, src):
    NUM[key] = {"value": value, "src": src}


# ================================================================== похідні числа
# --- П1.3: загальні обсяги
tot_R, tot_h, n_tasks, nmax = 0, 0.0, 0, 0
for name, v in AN.items():
    for r in v["rows"]:
        tot_R += r["R"]
        tot_h += r["hours"]
        n_tasks += 1
        nmax = max(nmax, r["L"] ** v["d"])
put("p1_3.n_tasks", n_tasks, "p1_3_analysis.json: кількість lattices.*.rows")
put("p1_3.total_R", tot_R, "p1_3_analysis.json: Σ lattices.*.rows[].R")
put("p1_3.total_hours", round(tot_h, 1), "p1_3_analysis.json: Σ lattices.*.rows[].hours")
put("p1_3.max_nodes", nmax, "p1_3_analysis.json: max L^d")
put("p1_3.n_met_target", sum(r["met_target"] for v in AN.values() for r in v["rows"]),
    "p1_3_analysis.json: Σ rows[].met_target")

# --- 3D: стала P* для L ≥ 32 (зважене середнє, χ²/dof)
rows3 = [r for r in AN["3D bond"]["rows"] if r["L"] >= 32]
Ps = np.array([r["all"]["P_star"] for r in rows3])
sPs = np.array([r["all"]["se_P_star"] for r in rows3])
w = 1 / sPs ** 2
Pm = float((w * Ps).sum() / w.sum())
sPm = float(1 / np.sqrt(w.sum()))
chi = float(((Ps - Pm) ** 2 * w).sum() / (len(Ps) - 1))
put("p1_3.3D.Pstar_Lge32", {"mean": Pm, "se": sPm, "chi2dof": chi, "n": len(Ps)},
    "p1_3_analysis.json: lattices['3D bond'].rows[L>=32].all.P_star, se_P_star; зважене середнє")

# --- П1.13: діапазони ψ з прямих підгонок (квадратичний і кубічний фон)
psi = {}
for src, dct in (("q", CC["psi_direct"]), ("c", CC2["psi_cubic"])):
    for k, v in dct.items():
        lat = k.split(" | ")[0]
        psi.setdefault(lat, []).append((v["par"]["psi"], v["chi2dof_naive"], src, k))
psi_rng = {}
for lat, lst in psi.items():
    ok = [x for x in lst if (x[1] < 2 or not lat.startswith("2D"))]
    psi_rng[lat] = {"min": min(x[0] for x in ok), "max": max(x[0] for x in ok), "n_fits": len(ok),
                    "n_all": len(lst), "rule": "2D: лише χ²/dof < 2; d ≥ 3: усі підгонки"}
put("p1_13.psi_direct_range", psi_rng,
    "notes/orchestra/C_checks.json: psi_direct.*.par.psi + C_checks2.json: psi_cubic.*.par.psi")

# --- ER: стрибок g∞ − g_fin у порозі (наївна амплітуда в середньому полі)
put("mf.ER.naive_jump", THR["ER"]["ainf"] - THR["ER"]["nsc_c"],
    "p1_13_threshold.json: ER.ainf − ER.nsc_c (= g∞(1⁺) − g_fin(1))")

# --- Додаток C: точні значення величин з Snarskii2024b (детерміновані суми)
sys.path[:0] = [str(ROOT / "code" / "lib")]
try:
    import networkx as nx
    from nsccore import nsc as _nsc
    yq = {q: float(_nsc(nx.circular_ladder_graph(q))) for q in (3, 4, 5, 6)}
except Exception as e:  # noqa: BLE001
    yq = {"error": repr(e)}
k = np.arange(1, 2 * 10 ** 6, dtype=float)


def _series(s, b):
    p = b / (s * (s - 1) / 2)
    return float((p * (1 - p)).sum())


put("pre.prisms", yq, "code/lib/nsccore.nsc(nx.circular_ladder_graph(q)); Y_q = GP(q,1)")
put("pre.eq14_square_bulk", _series(1 + 2 * k * (k + 1), 4 * k ** 2),
    "Σ_k p_k(1−p_k), s_k = 1 + 2k(k+1), b_k = 4k², k ≤ 2·10⁶ (PRE (13)–(14))")
put("pre.C1_square_edge", _series((k + 1) ** 2, k * (2 * k + 1)),
    "Σ_k p_k(1−p_k), s_k = (k+1)², b_k = k(2k+1), k ≤ 2·10⁶ (PRE (C1)–(C4) з s_k = (k+1)²)")

# --- середнє поле: відносний зсув піку від порога
put("mf.rel_shift", {"ER": CSTAR["peak"]["ER"]["xstar"] / 1.0 - 1,
                     "RRG z=3": CSTAR["peak"]["RRG z=3"]["xstar"] / LAWS["rrg"]["z3"]["pc"] - 1,
                     "RRG z=4": CSTAR["peak"]["RRG z=4"]["xstar"] / LAWS["rrg"]["z4"]["pc"] - 1},
    "p1_9_cstar.json: peak.*.xstar; поріг ER c=1, RRG — p1_12_laws.json: rrg.z*.pc")

# --- 2D: спад P* з L проти L^(−β/ν)
r2 = AN["2D bond sq"]["rows"]
bn = AN["2D bond sq"]["hypothesis_B"]["all"]["beta_over_nu"]
put("p1_3.2D.Pstar_drop", {"P_first": r2[0]["all"]["P_star"], "L_first": r2[0]["L"], "P_last": r2[-1]["all"]["P_star"],
                           "L_last": r2[-1]["L"], "ratio": r2[-1]["all"]["P_star"] / r2[0]["all"]["P_star"],
                           "ratio_power_law": (r2[-1]["L"] / r2[0]["L"]) ** (-bn), "beta_over_nu": bn},
    "p1_3_analysis.json: lattices['2D bond sq'].rows[0,-1].all.P_star; hypothesis_B.all.beta_over_nu")

# --- Додаток D: мінімальне |z| нахилу g_b(p) для найбільших L (П1.15)
zmin = {}
for name, lst in CL15["journal"].items():
    j = lst[-1]
    zmin[name] = {"L": j["L"], "min_abs_z": min(abs(r["z_slope"]) for r in j["rows"] if r["bin"] >= 6)}
put("p1_15.min_abs_z_slope", zmin, "p1_15_classes.json: journal.<решітка>[-1].rows[bin>=6].z_slope, min |z|")

# --- Додаток A: валідація оцінювача
put("valid.pooled_M0", VAL["pooled_M0"], "p1_2_validate.json: pooled_M0")

# --- Дерева клік (П1.3c): інтерполяція при N3 = 24, сфера радіуса 2, діапазон оцінювачів ГЦК, перевірка МК
fccA = A3B["lattices"]["fcc"]
fcc_ests = {"narrow": fccA["vs_bethe"]["narrow"]["rel_p_inf"],
            **{k: VA_["peaks"]["fcc"][k]["rel_p_inf"] for k in ("spline", "quad006", "quartic02")}}
put("cactus.fcc_range", {"min": min(fcc_ests.values()), "max": max(fcc_ests.values()), "by_estimator": fcc_ests},
    "p1_3b_bethe_lattices.json: lattices.fcc.vs_bethe.narrow.rel_p_inf; verify_A.json: peaks.fcc.{spline,quad006,quartic02}.rel_p_inf")
put("cactus.t24", {f: {**CA["interp_t24"][f], "pc_below_fcc": 1 - CA["interp_t24"][f]["pc"] / fccA["pc"]}
                   for f in ("homog", "gleeson")},
    "p1_3c_cactus.json: interp_t24.<сім'я>; pc_below_fcc = 1 − pc/pc(ГЦК), pc(ГЦК) — p1_3b_bethe_lattices.json: lattices.fcc.pc")

# стійкість вердиктів критерію П1.3c (referee_cactus_report.md, M3): спосіб інтерполяції, початковий набір Gleeson,
# смуга з п'ятьох оцінювачів ГЦК (з кубікою ±0.02)
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator  # noqa: E402


def _interp24(rows):
    t = np.array([r["t"] for r in rows], float)
    y = np.array([r["delta"] for r in rows])
    return {"spline": float(CubicSpline(t, y)(24)), "spline_natural": float(CubicSpline(t, y, bc_type="natural")(24)),
            "pchip": float(PchipInterpolator(t, y)(24)), "akima": float(Akima1DInterpolator(t, y)(24)),
            "linear": float(np.interp(24, t, y))}


def _verdict(d, lo, hi):
    return "узгоджено" if lo <= d <= hi else ("не підтверджено" if not 0.02 <= d <= 0.08 else "не розрізнено")


fcc5 = {**fcc_ests, "conventional": fccA["vs_bethe"]["conventional"]["rel_p_inf"]}
rob = {}
for f in ("homog", "gleeson"):
    rows = sorted(CA["families"][f].values(), key=lambda r: r["t"])
    ii = _interp24(rows)
    rob[f] = {"interp": ii, "min": min(ii.values()), "max": max(ii.values()),
              "verdicts4": sorted({_verdict(v, min(fcc_ests.values()), max(fcc_ests.values())) for v in ii.values()}),
              "verdicts5": sorted({_verdict(v, min(fcc5.values()), max(fcc5.values())) for v in ii.values()})}
planned = sorted([CA["families"]["gleeson"][str(c)] for c in (5, 7, 8, 9)], key=lambda r: r["t"])
rob["gleeson_planned"] = _interp24(planned)
rob["fcc5"] = {"min": min(fcc5.values()), "max": max(fcc5.values())}
put("cactus.robust", rob, "p1_3c_cactus.json: families.*.{t,delta} — сплайн (not-a-knot, natural), PCHIP, Akima, "
    "лінійна при N3 = 24; gleeson_planned — c ∈ {5,7,8,9} з плану; fcc5 — 4 оцінювачі + lattices.fcc.vs_bethe."
    "conventional; вердикти — критерій П1.3c")
# модель «пік на s̄ над власним pc» для ГЦК і розклад зсуву на поріг і ε
ms3 = A3B["tests"]["models_3D"]
pB12 = CA["bethe"]["p_star"]
put("cactus.near_pc_fcc", {"s_bar": ms3["s_bar"], "p_pred": fccA["pc"] * (1 + ms3["s_bar"]),
                           "delta_pred": fccA["pc"] * (1 + ms3["s_bar"]) / pB12 - 1},
    "p1_3b_bethe_lattices.json: tests.models_3D.s_bar, lattices.fcc.pc; p1_3c_cactus.json: bethe.p_star")
put("cactus.decomp", {**{f: {"pc_ratio": CA["interp_t24"][f]["pc"] / CA["bethe"]["pc"] - 1,
                             "eps_ratio": (1 + CA["interp_t24"][f]["s"]) / (1 + CA["bethe"]["s"]) - 1}
                         for f in ("homog", "gleeson")},
                      "fcc": {"pc_ratio": fccA["pc"] / CA["bethe"]["pc"] - 1,
                              "eps_ratio": (1 + fccA["vs_bethe"]["decomposition"]["s"]) / (1 + CA["bethe"]["s"]) - 1}},
    "1 + Δ_Б = (pc/pc_Б)·(1 + ε*)/(1 + ε*_Б); p1_3c_cactus.json: interp_t24.*.{pc,s}, bethe.{pc,s}; "
    "p1_3b_bethe_lattices.json: lattices.fcc.{pc, vs_bethe.decomposition.s}")
# усі дерева клік з z = 12 і N3 = 18…30 (p1_3c_extra.json)
cx = {r["label"]: {k: r[k] for k in ("t", "N4", "pc", "p_star", "delta", "s", "P_star")} for r in CX["structures"]}
near = [r for r in CX["structures"] if 21 <= r["t"] <= 27]
put("cactus.extra", {"rows": cx, "K8K4": cx["K8+K4+2K2"],
                     "near24": {"N3": [21, 27], "n": len(near), "min": min(r["delta"] for r in near),
                                "max": max(r["delta"] for r in near)},
                     "h005_max": max(abs(r["p_star_h005"] - r["p_star"]) for r in CX["structures"]),
                     "P_star_range": [min(r["P_star"] for r in CX["structures"]),
                                      max(r["P_star"] for r in CX["structures"])],
                     "s_range": [min(r["s"] for r in CX["structures"]), max(r["s"] for r in CX["structures"])]},
    "p1_3c_extra.json: structures.*; near24 — Δ_Б для 21 ≤ N3 ≤ 27")


def sphere2(sizes):
    z = sum(m - 1 for m in sizes)
    return sum((m - 1) * (z - (m - 1)) for m in sizes)


put("cactus.S2", {f"{f}:{k}": {"N3": r["t"], "S2": sphere2(r["sizes"])} for f in ("homog", "gleeson")
                  for k, r in CA["families"][f].items()},
    "p1_3c_cactus.json: families.*.sizes; |S_2| = Σ(m−1)(z−m+1) — кількість вузлів на відстані 2 у дереві клік")
put("cactus.P_star_range", [min(r["P_star"] for f in ("homog", "gleeson") for r in CA["families"][f].values()),
                            max(r["P_star"] for f in ("homog", "gleeson") for r in CA["families"][f].values())],
    "p1_3c_cactus.json: families.*.P_star")
put("cactus.h005_max", max(abs(r["p_star_h005"] - r["p_star"]) for f in ("homog", "gleeson")
                           for r in CA["families"][f].values()),
    "p1_3c_cactus.json: families.*.{p_star_h005,p_star}")
put("cactus.validate", {**CV["summary"], "max_abs_zP": max(abs(q["zP"]) for c in CV["cases"].values()
                                                             for q in c["points"]),
                        "nroots": CV["meta"]["params"]["NROOTS"], "scap": CV["meta"]["params"]["S_CAP"]},
    "p1_3c_validate.json: summary; cases.*.points[].zP; meta.params")

with open(OUT / "numbers.json", "w") as f:
    json.dump(NUM, f, ensure_ascii=False, indent=1)
print("numbers.json:", len(NUM), "записів")

# ================================================================== Рис. 1: NSC(ε) і P(ε)
fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.45), constrained_layout=True)
for name in ("2D bond sq", "3D bond", "4D bond", "5D bond", "6D bond"):
    v = AN[name]
    r = v["rows"][-1]
    t = task(name, r["L"])
    ps = np.array(t["ps"])
    x = (ps - v["pc"]) / v["pc"]
    y, se = np.array(t["mean"]["nsc"]), np.array(t["mean"]["se"])
    a.plot(x, y, "-", color=COL[name], label=f"{LAB[name]}, $L={r['L']}$")
    a.fill_between(x, y - 2 * se, y + 2 * se, color=COL[name], alpha=0.2, lw=0)
    xs = (r["all"]["p_star"] - v["pc"]) / v["pc"]
    a.plot(xs, r["all"]["nsc_star"], "o", color=COL[name], ms=5, mec="white", mew=1, zorder=5)
    P = np.array(t["mean"]["P"])
    b.plot(x, P, "-", color=COL[name], label=f"{LAB[name]}, $L={r['L']}$")
    b.plot(xs, r["all"]["P_star"], "o", color=COL[name], ms=5, mec="white", mew=1, zorder=5)
for ax in (a, b):
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-0.15, 0.4)
    ax.set_xlabel(r"$\varepsilon=(p-p_c)/p_c$")
a.set_ylabel("NSC (нормування на всі вузли)")
b.set_ylabel(r"частка вузлів у гіганті $P$")
b.axhline(0.5, color=MUTED, lw=0.8, ls="--")
a.legend(frameon=False, loc="upper right")
panel(a, "a")
panel(b, "b")
save(fig, "fig1_curves.pdf")

# ================================================================== Рис. 2: FSS піку
fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.5), constrained_layout=True)
ins = a.inset_axes([0.55, 0.50, 0.42, 0.40])
for name, v in AN.items():
    L = np.array([r["L"] for r in v["rows"]])
    ps = np.array([r["all"]["p_star"] for r in v["rows"]])
    s = np.hypot([r["all"]["se_p_star"] for r in v["rows"]], [r["all"]["syst_window"] for r in v["rows"]])
    P = np.array([r["all"]["P_star"] for r in v["rows"]])
    sP = np.array([r["all"]["se_P_star"] for r in v["rows"]])
    kw = dict(fmt=MK[name] + "-", color=COL[name], ms=3, lw=1.1, capsize=1.5, elinewidth=0.8, label=LAB[name],
              mfc="white" if v["d"] == 2 else COL[name])
    a.errorbar(L, (ps - v["pc"]) / v["pc"], s / v["pc"], **kw)
    b.errorbar(L, P, sP, **kw)
    if v["d"] == 2:
        ins.errorbar(L, 100 * (ps - v["pc"]) / v["pc"], 100 * s / v["pc"], **{**kw, "label": None, "ms": 2.5})
    d = v["d"]
    if d >= 3 and name.endswith("bond"):
        rb = BETHE[d]
        a.plot([L.max() * 1.15, L.max() * 1.6], [(rb["p_star"] - v["pc"]) / v["pc"]] * 2, "-",
               color=COL[name], lw=2.2, alpha=0.6)
        b.plot([L.max() * 1.15, L.max() * 1.6], [rb["P_star"]] * 2, "-", color=COL[name], lw=2.2, alpha=0.6)
a.axhline(0, color=MUTED, lw=0.8)
a.set(xscale="log", xlabel="розмір решітки $L$", ylabel=r"$(p^*-p_c)/p_c$")
b.axhline(0.5, color=MUTED, lw=0.8, ls="--")
b.set(xscale="log", xlabel="розмір решітки $L$", ylabel=r"частка гіганта в піку $P^*$")
ins.axhline(0, color=MUTED, lw=0.6)
ins.set_xscale("log")
ins.set_ylim(-2.2, 0.4)
ins.tick_params(labelsize=5.5)
ins.set_title("2D, %", fontsize=6, pad=1)
ins.grid(True, lw=0.4)
b.legend(frameon=False, ncol=2, fontsize=6.0, loc="lower left")
panel(a, "a")
panel(b, "b")
save(fig, "fig2_fss.pdf")

# ================================================================== Рис. 3: точне середнє поле (ER, RRG)
parts = ER["parts"]
c = np.array([q["c"] for q in parts])
nscv = np.array([q["nsc"] for q in parts])
Pv = np.array([q["P"] for q in parts])
pf = np.array([q["part_fin"] for q in parts])
pi = np.array([q["part_inf"] for q in parts])
gf = np.array([q["g_fin"] for q in parts])
gi = np.array([np.nan if q["g_inf"] is None else q["g_inf"] for q in parts], dtype=float)
fig, (a, b, cc) = plt.subplots(1, 3, figsize=(W2, 2.3), constrained_layout=True)
a.plot(c, nscv, "-", color=INK, label="NSC")
a.plot(c, pf, "--", color=COL["2D bond sq"], label=r"$(1-P)\,g_{\rm fin}$")
a.plot(c, pi, "-.", color=COL["4D bond"], label=r"$P\,g_\infty$")
cs = CSTAR["peak"]["ER"]["xstar"]
a.axvline(1, color=MUTED, lw=0.8)
a.axvline(cs, color=MUTED, lw=0.8, ls=":")
a.text(cs + 0.04, 0.45, r"$c^*$", color=MUTED, transform=a.get_xaxis_transform())
a.set(xlabel="середній степінь $c$", ylabel="NSC і внески", xlim=(0.5, c.max()))
a.legend(frameon=False, loc="upper right")
b.plot(c, gf, "-", color=COL["2D bond sq"], label=r"$g_{\rm fin}$")
b.plot(c, gi, "-", color=COL["4D bond"], label=r"$g_\infty$")
b.plot(c, Pv, "-", color=MUTED, lw=1, label="$P$")
b.plot([1], [THR["ER"]["ainf"]], "o", color=COL["4D bond"], ms=4, mec="white")
b.plot([1], [THR["ER"]["nsc_c"]], "o", color=COL["2D bond sq"], ms=4, mec="white")
b.axvline(1, color=MUTED, lw=0.8)
b.set(xlabel="середній степінь $c$", ylabel="середня NodeSC; $P$", xlim=(0.5, c.max()))
b.legend(frameon=False, loc="center right")
for z, col in (("z3", COL["2D bond tri"]), ("z4", COL["3D bond"])):
    rr = LAWS["rrg"][z]
    pp = np.array([q["c"] for q in rr["parts"]])
    yy = np.array([q["nsc"] for q in rr["parts"]])
    cc.plot(pp / rr["pc"] - 1, yy, "-", color=col, label=f"RRG $z={z[1]}$")
    pk = CSTAR["peak"]["RRG z=3" if z == "z3" else "RRG z=4"]
    cc.plot(pk["xstar"] / rr["pc"] - 1, pk["nsc"], "o", color=col, ms=4, mec="white")
er_x = c - 1
cc.plot(er_x, nscv, "-", color=INK, label="ER")
cc.plot(cs - 1, CSTAR["peak"]["ER"]["nsc"], "o", color=INK, ms=4, mec="white")
cc.axvline(0, color=MUTED, lw=0.8)
cc.set(xlabel=r"$\varepsilon$ (від порога)", ylabel="NSC", xlim=(-0.5, 1.5))
cc.legend(frameon=False, loc="lower right")
panel(a, "a")
panel(b, "b")
panel(cc, "c")
save(fig, "fig3_meanfield.pdf")

# ================================================================== Рис. 4: решітки проти гратки Бете
L3 = A3B["lattices"]
fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.5), constrained_layout=True)
for nm in ("diamond", "sc", "bcc", "fcc", "hc4", "hc5", "hc6"):
    v = L3[nm]
    d = v["vs_bethe"]["narrow"]
    x = v["loops"]["Lambda4"]
    a.errorbar(x, 100 * d["rel_p_inf"], 100 * d["err_rel_p_inf"], fmt="o", color=A_COL[nm], ms=4, capsize=1.5,
               elinewidth=0.8)
    a.annotate(f"{A_LAB[nm]} ($z={v['z']}$)", (x, 100 * d["rel_p_inf"]), textcoords="offset points",
               xytext={"diamond": (6, -9), "sc": (5, 3), "hc4": (5, -2), "hc5": (6, -7), "bcc": (-8, -14), "hc6": (-22, 7),
                       "fcc": (-34, 5)}.get(nm, (5, 2)),
               fontsize=6.3, color=A_COL[nm])
    dec = v["vs_bethe"]["decomposition"]
    xb = dec["pc_over_bethe"] - 1
    b.errorbar(100 * xb, 100 * dec["delta"], 100 * d["err_rel_p_inf"], fmt="o", color=A_COL[nm], ms=4, capsize=1.5,
               elinewidth=0.8, label=A_LAB[nm])
a.axhline(0, color=MUTED, lw=0.8)
a.set(xlabel=r"$\Lambda_4=\mathcal{N}_3p^3+\mathcal{N}_4p^4$ при $p=p^*_{\rm Б}$",
      ylabel=r"$\Delta_{\rm Б}=(p_\infty-p^*_{\rm Б})/p^*_{\rm Б}$, %")
m3 = A3B["tests"]["models_3D"]
xx = np.linspace(-0.25, 0.0, 50)
b.axhline(0, color=MUTED, lw=1, label="модель «пік = Бете»")
b.plot(100 * xx, 100 * ((1 + xx) * (1 + m3["s_bar"]) - 1), "--", color=MUTED, lw=1,
       label=rf"модель «пік на $\bar\varepsilon_\infty={100 * m3['s_bar']:.1f}\%$ над $p_c$»")
b.set(xlabel=r"$p_c/p^*_{\rm Б}-1$, %", ylabel=r"$\Delta_{\rm Б}$, %", ylim=(-5.5, 6.5))
b.legend(frameon=False, fontsize=5.8, loc="lower left", ncol=2)
panel(a, "a")
panel(b, "b")
save(fig, "fig4_bethe.pdf")

# ================================================================== Рис. 5: поріг — спільна межа і відхід гіганта
SEL = ("2D bond sq", "2D bond tri", "2D site sq", "3D bond", "4D bond")
fig, (a, b, cpan) = plt.subplots(1, 3, figsize=(W2, 2.6), constrained_layout=True)
for name in SEL:
    v = LAT13[name]
    j = v["journal"][-1]
    f6 = j["fits"]["6"]
    s = np.array(j["s_mean"])
    g = np.array(j["g"])
    m = s >= 8
    a.loglog(s[m], f6["g_inf"] - g[m], MK[name], color=COL[name], ms=2.6, label=f"{LAB[name]}, $L={j['L']}$")
    ss = np.logspace(np.log10(64), np.log10(s.max()), 30)
    a.loglog(ss, f6["a"] * ss ** (-f6["theta"]), "-", color=COL[name], lw=0.9)
    gi_ = v["giant"]
    e, D, se = np.array(gi_["eps"]), np.array(gi_["D"]), np.array(gi_["se"])
    ok = D > 0
    b.errorbar(e[ok], D[ok], se[ok], fmt=MK[name] + "-", color=COL[name], ms=2.3, lw=0.9, capsize=0, elinewidth=0.7)
a.set(xlabel="розмір кластера $s$", ylabel=r"$g_\infty-g(s;p_c)$")
fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=6, loc="outside lower center", ncol=5)
b.set(xscale="log", yscale="log", xlabel=r"$\varepsilon$", ylabel=r"$D_G=g_\infty-g_G(p)$")
names5 = ("2D bond sq", "2D bond tri", "2D site sq", "3D bond", "4D bond")
xs = np.arange(len(names5))
for off, mod, col, lab in ((-0.18, "C0+A e^x(1+c e)", "#6f6e69", r"множна поправка"),
                           (0.18, "C0+a e+b e ln e", "#bdbcb6", r"фон $a\varepsilon+b\varepsilon\ln\varepsilon$")):
    na = np.array([CC2["C0_total"][n]["naive_amp"] for n in names5])
    c0 = np.array([abs(CC2["C0_total"][n]["models"][mod]["C0"]) for n in names5])
    e0 = np.array([CC2["C0_total"][n]["models"][mod]["se_total"] for n in names5])
    th = np.array([CC2["C0_total"][n]["models"][mod]["rel_thetahyp"] for n in names5])
    cpan.bar(xs + off, 100 * c0 / na, 0.36, yerr=100 * e0 / na, color=col, capsize=1, error_kw={"elinewidth": 0.7},
             label=lab)
    cpan.plot(xs + off, 100 * th, "_", color=INK, ms=7, mew=1.2)
cpan.plot([], [], "_", color=INK, ms=7, mew=1.2, label=r"$g_\infty$ з $\theta=\theta_{\rm hyp}$")
cpan.set_xticks(xs, ["2D кв.", "2D тр.", "2D вузли", "3D", "4D"], rotation=30, fontsize=6.5)
cpan.set(ylabel=r"$|C_0|/(g_G-g_F)$, %", ylim=(0, 12))
cpan.legend(frameon=False, fontsize=5.8, loc="upper left")
b.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
panel(a, "a")
panel(b, "b")
panel(cpan, "c")
save(fig, "fig5_threshold.pdf")

# ================================================================== Рис. 6: односторонні нахили
fig, a = plt.subplots(figsize=(W1, 2.3), constrained_layout=True)
for name, L in (("3D bond", "256"), ("2D bond sq", "1024")):
    o = CC["one_sided"][name][L]
    em, lo, hi = np.array(o["eps_mid"]), np.array(o["eps_lo"]), np.array(o["eps_hi"])
    sl, se = np.array(o["slope"]), np.array(o["se"])
    a.errorbar(em, sl, se, xerr=[em - lo, hi - em], fmt="o", color=COL[name], ms=3.5, capsize=1.5,
               elinewidth=0.8, label=f"{LAB[name]}, $L={L}$")
a.axvline(0, color=MUTED, lw=0.8)
a.axhline(0, color=MUTED, lw=0.6)
a.set(xlabel=r"$\varepsilon=(p-p_c)/p_c$", ylabel=r"$d\,\mathrm{NSC}/dp$ на інтервалі")
a.legend(frameon=False, loc="lower left", fontsize=6.5)
save(fig, "fig6_slopes.pdf")

# ================================================================== Рис. 7 (дод. B): залежність піку від вікна
fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1]})
for nm in ("diamond", "bcc", "fcc"):
    v = L3[nm]
    r = v["rows"][-1]
    hw = np.array([float(h) for h in r["p_star_by_hw"]])
    pv = np.array(list(r["p_star_by_hw"].values()))
    pb = v["bethe"]["p_star"]
    a.plot(hw, 100 * (pv / pb - 1), "o-", color=A_COL[nm], ms=3, label=f"{A_LAB[nm]}, $L={r['L']}$")
    a.plot([0.02], [100 * (r["p_star"] / pb - 1)], "s", color=A_COL[nm], ms=3.5, mfc="white")
a.set(xlabel="півширина вікна $h$", ylabel=r"$p^*(h)/p^*_{\rm Б}-1$, %")
a.legend(frameon=False, fontsize=6)
labs = ["ER", "RRG 3", "RRG 4"]
keys = ["ER", "RRG z=3", "RRG z=4"]
ex = [CSTAR["peak"][k]["xstar"] for k in keys]
par = [CSTAR["peak"][k]["parabola_on_exact_same_window"] for k in keys]
cub = [CSTAR["peak"][k]["cubic_on_exact_same_window"] for k in keys]
xx = np.arange(3)
b.bar(xx - 0.2, [100 * (p / e - 1) for p, e in zip(par, ex)], 0.4, color=COL["4D bond"], label="парабола")
b.bar(xx + 0.2, [100 * (q / e - 1) for q, e in zip(cub, ex)], 0.4, color=COL["3D bond"], label="кубіка")
b.axhline(0, color=MUTED, lw=0.8)
b.set_xticks(xx, labs, fontsize=6.5)
b.set(ylabel="зсув від точного піку, %", ylim=(-0.3, 2.4))
b.legend(frameon=False, fontsize=6, loc="upper left")
panel(a, "a")
panel(b, "b")
save(fig, "fig7_window.pdf")

# ================================================================== Рис. 8 (дод. D): g(s; p) за фіксованого s
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), constrained_layout=True)
cmap = plt.get_cmap("viridis")
for ax, name, let in ((axs[0], "2D bond sq", "a"), (axs[1], "3D bond", "b")):
    j = CL15["journal"][name][-1]
    pc = j["pc"]
    bins = [r for r in j["rows"] if r["bin"] in (6, 8, 10, 12)]
    for i, r in enumerate(bins):
        p = np.array(r["p"])
        g, se = np.array(r["g"]), np.array(r["se"])
        col = cmap(i / max(1, len(bins) - 1) * 0.85)
        ax.errorbar((p - pc) / pc, g, 2 * se, fmt="o-", ms=2, lw=0.9, color=col, capsize=0, elinewidth=0.6,
                    label=f"$s\\in[{r['s_range'][0]},{r['s_range'][1] + 1})$")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set(xlabel=r"$\varepsilon=(p-p_c)/p_c$", ylabel=r"$g(s;p)$, середня NodeSC")
    ax.set_title(f"{LAB[name]}, $L={j['L']}$", fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    panel(ax, let)
save(fig, "fig8_classes.pdf")
# ================================================================== Рис. 9: дерева клік (П1.3c) проти ГЦК
CF_COL = {"homog": "#2a78d6", "gleeson": "#d6772a"}
CF_LAB = {"homog": r"$k$ клік $K_n$, $k(n-1)=12$", "gleeson": r"кліка $K_c$ і $13-c$ ребер"}
fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.45), constrained_layout=True)
fr = NUM["cactus.fcc_range"]["value"]
a.axhspan(100 * fr["min"], 100 * fr["max"], color=GRID, zorder=0, lw=0)
a.text(0.5, 100 * fr["max"] + 0.25, "ГЦК: діапазон оцінювачів піку", fontsize=6.3, color=MUTED)
fam_lab = {tuple(r["sizes"]) for f in ("homog", "gleeson") for r in CA["families"][f].values()}
oth = [r for r in CX["structures"] if tuple(r["sizes"]) not in fam_lab and r["label"] != "K8+K4+2K2"]
for ax_, q in ((a, "delta"), (b, "s")):
    ax_.plot([r["t"] for r in oth], [100 * r[q] for r in oth], "o", color=MUTED, ms=2.6, alpha=0.8, zorder=3,
             label="інші дерева клік, $z=12$" if ax_ is a else None)
for f in ("homog", "gleeson"):
    rows = sorted(CA["families"][f].values(), key=lambda r: r["t"])
    t = np.array([r["t"] for r in rows], float)
    tt = np.linspace(t[0], t[-1], 200)
    for ax_, q in ((a, "delta"), (b, "s")):
        ax_.plot(tt, 100 * CubicSpline(t, [r[q] for r in rows])(tt), "-", color=CF_COL[f], lw=1.2, zorder=4)
        ax_.plot(t, [100 * r[q] for r in rows], "o", color=CF_COL[f], ms=3.5, zorder=4,
                 label=CF_LAB[f] if ax_ is a else None)
    a.plot([24], [100 * CA["interp_t24"][f]["delta"]], "D", mfc="white", mec=CF_COL[f], ms=4.5, mew=1.1, zorder=5,
           label="сплайн при $\\mathcal{N}_3=24$" if f == "gleeson" else None)
k8 = cx["K8+K4+2K2"]
for ax_, q in ((a, "delta"), (b, "s")):
    ax_.plot([24], [100 * k8[q]], "*", color="#7b5cd6", ms=7.5, zorder=7, label=r"$K_8+K_4+2K_2$" if ax_ is a else None)
nv = fccA["vs_bethe"]["narrow"]
a.errorbar([24.9], [100 * nv["rel_p_inf"]], [100 * nv["err_rel_p_inf"]], fmt="s", color=INK, ms=4, capsize=1.5,
           elinewidth=0.8, zorder=6, label=r"ГЦК ($\mathcal{N}_3=24$, зсунуто), $p_\infty$")
a.axhline(0, color=MUTED, lw=0.8)
a.set(xlabel=r"$\mathcal{N}_3$ — трикутників через вузол", ylabel=r"$\Delta_{\rm Б}=p^*/p^*_{\rm Б}-1$, %",
      xlim=(-1.5, 38))
a.legend(frameon=False, fontsize=6, loc="upper left")
dec = fccA["vs_bethe"]["decomposition"]
b.errorbar([24.9], [100 * dec["s"]], [100 * dec["err_s"]], fmt="s", color=INK, ms=4, capsize=1.5, elinewidth=0.8,
           label=r"ГЦК, $\varepsilon_\infty$")
b.set(xlabel=r"$\mathcal{N}_3$ — трикутників через вузол", ylabel=r"$\varepsilon^*=p^*/p_c-1$, %",
      xlim=(-1.5, 38), ylim=(0, 38))
b.legend(frameon=False, fontsize=6, loc="lower left")
panel(a, "a")
panel(b, "b")
save(fig, "fig9_cactus.pdf")
# ================================================================== Рис. 10: знімки NodeSC (2D, L = 256)
from matplotlib.colors import Normalize  # noqa: E402
SN_NORM, SN_CMAP, SN_FADE = Normalize(0, 2.0), plt.get_cmap("Blues"), 0.3
put("snap", [{k: q[k] for k in ("p", "P", "nsc", "g_giant", "g_rest", "g_rest_nonisolated")} for q in SNAPJ["panels"]],
    "p1_3_snapshots_paper.json: panels[] (одна реалізація, 2D bond-квадратна, L = 256, точна NodeSC)")
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.55), constrained_layout=True)
what = ("нижче порога", r"пік $p^*(256)$", "вище порога")
for i, ax in enumerate(axs):
    sc, giant, q = SNAPZ[f"sc{i}"], SNAPZ[f"giant{i}"], SNAPJ["panels"][i]
    rgb = SN_CMAP(SN_NORM(sc))[..., :3]
    rgb[~giant] = 1 - SN_FADE * (1 - rgb[~giant])
    ax.imshow(rgb, interpolation="nearest", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GRID)
    ax.set_title(f"$p={q['p']}$ ({what[i]}), $P={q['P']:.2f}$\n"
                 f"$g_{{\\rm max}}={q['g_giant']:.2f}$, решта ${q['g_rest']:.2f}$", fontsize=7.5)
    panel(ax, "abc"[i])
cb = fig.colorbar(plt.cm.ScalarMappable(SN_NORM, SN_CMAP), ax=axs, shrink=0.85, extend="max", pad=0.01)
cb.set_label("NodeSC", color=MUTED)
cb.outline.set_visible(False)
save(fig, "fig10_snapshots.pdf")
# ================================================================== Рис. 11: колапс у скейлінговій змінній
BN_ = load(RES / "p1_3_analysis.json")["meta"]["params"]["beta_nu"]
fig, axs = plt.subplots(2, 2, figsize=(W2, 4.3), constrained_layout=True)
xstar = {}
for col, (name, d, lim) in enumerate((("2D bond sq", 2, (-4, 6)), ("3D bond", 3, (-4, 12)))):
    v = AN[name]
    beta, nu = BN_[str(d)]["beta"], BN_[str(d)]["nu"]
    Ls = [r["L"] for r in v["rows"]]
    cmap = plt.get_cmap("Blues")
    xstar[name] = {}
    for i, L in enumerate(Ls):
        tk = task(name, L)
        x = (np.array(tk["ps"]) - v["pc"]) * L ** (1 / nu)
        c = cmap(0.35 + 0.65 * i / max(len(Ls) - 1, 1))
        axs[0, col].plot(x, np.array(tk["mean"]["P"]) * L ** (beta / nu), "o-", color=c, ms=2, lw=0.9,
                         label=f"$L={L}$")
        axs[1, col].plot(x, tk["mean"]["nsc"], "o-", color=c, ms=2, lw=0.9)
        r = next(r for r in v["rows"] if r["L"] == L)["all"]
        xs = (r["p_star"] - v["pc"]) * L ** (1 / nu)
        xstar[name][str(L)] = xs
        axs[1, col].plot(xs, r["nsc_star"], "D", color="#eb6834", ms=3.5, zorder=5,
                         label=r"пік $p^*(L)$" if i == 0 else None)
    for row in (0, 1):
        axs[row, col].set_xlim(lim)
        axs[row, col].axvline(0, color=MUTED, lw=0.8)
        axs[row, col].set_xlabel(r"$(p-p_c)\,L^{1/\nu}$")
    axs[0, col].set_ylabel(r"$P\,L^{\beta/\nu}$")
    axs[1, col].set_ylabel("NSC (на всі вузли)")
    axs[0, col].set_title(LAB[name], fontsize=8.5)
    axs[0, col].legend(frameon=False, fontsize=5.8, loc="upper left", ncol=1 if d == 2 else 2)
    axs[1, col].legend(frameon=False, fontsize=6, loc="lower right" if d == 2 else "lower left")
    panel(axs[0, col], "ab"[col])
    panel(axs[1, col], "cd"[col])
put("collapse.xstar", xstar, "p1_3_analysis.json: lattices.<решітка>.{pc,rows[].all.p_star}, meta.params.beta_nu; "
    "x* = (p* − pc)·L^(1/ν)")
save(fig, "fig11_collapse.pdf")
# числа рис. 10–11 додано після першого запису numbers.json — перезаписати
with open(OUT / "numbers.json", "w") as f:
    json.dump(NUM, f, ensure_ascii=False, indent=1)
print("numbers.json:", len(NUM), "записів")
print("готово")
