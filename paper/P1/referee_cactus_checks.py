"""Незалежна перевірка рецензента: підрозділ «Дерева клік» (розд. 5), табл. tab:cactus, рис. fig:cactus,
дод. app:cactus і речення про дерева клік в анотації, вступі, методиці й висновках `paper/P1/main_uk.tex`.

Запуск:  .venv/bin/python paper/P1/referee_cactus_checks.py   ->  paper/P1/referee_cactus_checks.json   (~3–5 хв)

Розділи JSON:
  numbers    — кожне число з обсягу: рядок .tex, значення в тексті, перераховане значення, статус (ok / FALSE / approx).
  indep_pc   — пороги й P(p) власною реалізацією (рекурсія Гілберта + багатотипний процес розгалуження),
               незалежно від code/lib/cactus.py; для сім'ї Gleeson — ще й рівняння (7) з Gleeson2009;
               K3×6 — корінь кубічного рівняння 10(p + p² − p³) = 1.
  layers     — формула ймовірності набору BFS-шарів кліки (дод. Д): нормування для m = 2…10, перебір для m = 7
               (тест покриває лише m ≤ 6), МК для m = 10.
  s2         — |S_2| = z(z − 1) − 2N3 перевірено BFS на явно побудованій околиці дерева клік; ГЦК: N3, N4, |S_2|,
               трикутників на ребро — BFS/перебір на періодичній ГЦК (networkx).
  mc         — власне Монте-Карло NSC на нескінченному дереві клік (numba, інший код і сід, ніж cactus.py) при p*
               для K2×12, K3×6, K9+4K2 (остання структура в МК-перевірці статті відсутня) і для (8,4,2,2).
  interp     — стійкість Δ_Б(N3 = 24) до способу інтерполяції і до набору точок (сім'я Gleeson з плану: c ∈ {5,7,8,9}).
  extra      — справжня (не інтерпольована) структура з z = 12 і N3 = 24: K8 + K4 + 2K2, точна квадратура.
  criterion  — застосування заздалегідь записаного критерію з різними діапазонами ГЦК; z-оцінки відносно похибок ГЦК;
               розклад Δ_Б дерева клік на зсув порога і зміну ε*.
"""
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import networkx as nx
import numba as nb
import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import cactusexact as ce  # noqa: E402  (лише для квадратури в «extra» і для перебору шарів)

T0 = time.time()
J = lambda p: json.loads((ROOT / p).read_text())  # noqa: E731
CA = J("results/P1/p1_3c_cactus.json")
CV = J("results/P1/p1_3c_validate.json")
NU = J("paper/P1/numbers.json")
A3 = J("results/P1/p1_3b_bethe_lattices.json")["lattices"]["fcc"]
VA = J("notes/orchestra/verify_A.json")["peaks"]["fcc"]
OUT = {}
NUM = []


def chk(line, what, text, value, digits=None, scale=1.0, tol=None, note=""):
    """text — рядок з тексту; value — перераховане; порівняння після округлення до digits знаків (у шкалі scale)."""
    v = value * scale
    if tol is not None:
        ok = abs(float(text) - v) <= tol
        st = "ok" if ok else "FALSE"
    else:
        r = round(v, digits)
        ok = abs(float(text) - r) < 0.5 * 10 ** (-digits) + 1e-12
        st = "ok" if ok else "FALSE"
    NUM.append(dict(line=line, what=what, text=text, recomputed=v, status=st, note=note))


fam = CA["families"]
pb = CA["bethe"]["p_star"]
I24 = CA["interp_t24"]
fr = NU["cactus.fcc_range"]["value"]
nv = A3["vs_bethe"]["narrow"]
cv = A3["vs_bethe"]["conventional"]
dec = A3["vs_bethe"]["decomposition"]

# ------------------------------------------------------------------ 1. числа з тексту
# анотація, р. 56
chk(56, "анотація: Δ однорідна (сплайн), %", "4.6", I24["homog"]["delta"], 1, 100)
chk(56, "анотація: Δ Gleeson (сплайн), %", "6.4", I24["gleeson"]["delta"], 1, 100)
chk(56, "анотація: ГЦК min оцінювачів, %", "4.0", fr["min"], 1, 100)
chk(56, "анотація: ГЦК max оцінювачів, %", "5.8", fr["max"], 1, 100)
chk(56, "анотація: z = 12", "12", sum(m - 1 for m in CA["bethe"]["sizes"]), 0)
chk(56, "анотація: 24 трикутники ГЦК", "24", A3["loops"]["c3"], 0)
# р. 268
for n, t in zip((2, 3, 4, 5, 7), (0, 6, 12, 18, 30)):
    chk(268, f"N3 для K{n}×{12 // (n - 1)}", str(t), fam["homog"][str(n)]["t"], 0)
    chk(268, f"N3 формула Σ C(m−1,2), K{n}", str(t), sum(math.comb(m - 1, 2) for m in fam["homog"][str(n)]["sizes"]), 0)
chk(268, "N3 Gleeson c=3", "1", fam["gleeson"]["3"]["t"], 0)
chk(268, "N3 Gleeson c=10", "36", fam["gleeson"]["10"]["t"], 0)
chk(268, "крок похідної 1e-4", "0.0001", CA["meta"]["params"]["STEP"], 6)
chk(268, "критерій: ГЦК min, %", "4.0", fr["min"], 1, 100)
chk(268, "критерій: ГЦК max, %", "5.8", fr["max"], 1, 100)
# р. 271
chk(271, "Δ однорідна сплайн, %", "4.6", I24["homog"]["delta"], 1, 100)
chk(271, "Δ Gleeson сплайн, %", "6.4", I24["gleeson"]["delta"], 1, 100)
chk(271, "Δ однорідна лінійно, %", "4.8", I24["homog"]["delta_linear"], 1, 100)
chk(271, "Δ Gleeson лінійно, %", "6.5", I24["gleeson"]["delta_linear"], 1, 100)
chk(271, "ГЦК Δ narrow, %", "4.0", nv["rel_p_inf"], 1, 100)
chk(271, "ГЦК Δ narrow похибка, %", "0.8", nv["err_rel_p_inf"], 1, 100)
chk(271, "p* однорідна при 24", "0.1274", I24["homog"]["p_star"], 4)
chk(271, "p* Gleeson при 24", "0.1296", I24["gleeson"]["p_star"], 4)
chk(271, "p∞ ГЦК", "0.1267", A3["narrow"]["p_inf"], 4)
chk(271, "p∞ ГЦК похибка (×1e4)", "10", A3["narrow"]["err_p_inf"], 0, 1e4)
chk(271, "pc однорідна при 24", "0.0988", I24["homog"]["pc"], 4)
chk(271, "pc Gleeson при 24", "0.1002", I24["gleeson"]["pc"], 4)
chk(271, "pc нижче ГЦК, однорідна, %", "18", 1 - I24["homog"]["pc"] / A3["pc"], 0, 100)
chk(271, "pc нижче ГЦК, Gleeson, %", "17", 1 - I24["gleeson"]["pc"] / A3["pc"], 0, 100)
# таблиця tab:cactus, р. 283–300
rows = [("K2×12", fam["homog"]["2"], 283), ("K3×6", fam["homog"]["3"], 284), ("K4×4", fam["homog"]["4"], 285),
        ("K5×3", fam["homog"]["5"], 286), ("K7×2", fam["homog"]["7"], 287)] + \
       [(f"K{c}+{13 - c}K2", fam["gleeson"][str(c)], 286 + c) for c in range(3, 11)]
TAB = {283: (0, "0.09091", "0.12179", "0", "34.0", "0.527"), 284: (6, "0.09227", "0.12279", "0.82", "33.1", "0.527"),
       285: (12, "0.09395", "0.12400", "1.82", "32.0", "0.527"), 286: (18, "0.09607", "0.12549", "3.04", "30.6", "0.527"),
       287: (30, "0.10239", "0.12989", "6.65", "26.9", "0.527"), 289: (1, "0.09113", "0.12195", "0.14", "33.8", "0.527"),
       290: (3, "0.09162", "0.12233", "0.45", "33.5", "0.527"), 291: (6, "0.09245", "0.12299", "0.99", "33.0", "0.528"),
       292: (10, "0.09375", "0.12403", "1.84", "32.3", "0.528"), 293: (15, "0.09567", "0.12563", "3.16", "31.3", "0.530"),
       294: (21, "0.09852", "0.12807", "5.16", "30.0", "0.532"), 295: (28, "0.10286", "0.13191", "8.31", "28.2", "0.535"),
       296: (36, "0.10984", "0.13832", "13.58", "25.9", "0.540")}
for name, r, ln in rows:   # Gleeson c = 3…10 — рядки 289…296
    t, pc_, ps_, d_, e_, P_ = TAB[ln]
    chk(ln, f"табл.: {name} N3", str(t), r["t"], 0)
    chk(ln, f"табл.: {name} pc", pc_, r["pc"], 5)
    chk(ln, f"табл.: {name} p*", ps_, r["p_star"], 5)
    chk(ln, f"табл.: {name} Δ, %", d_, r.get("delta", 0.0), 2, 100)
    chk(ln, f"табл.: {name} ε*, %", e_, r["s"], 1, 100)
    chk(ln, f"табл.: {name} P*", P_, r["P_star"], 3)
for fm, ln in (("homog", 298), ("gleeson", 299)):
    i = I24[fm]
    chk(ln, f"табл. інтерп. {fm}: pc", ("0.0988" if fm == "homog" else "0.1002"), i["pc"], 4)
    chk(ln, f"табл. інтерп. {fm}: p*", ("0.1274" if fm == "homog" else "0.1296"), i["p_star"], 4)
    chk(ln, f"табл. інтерп. {fm}: Δ, %", ("4.61" if fm == "homog" else "6.38"), i["delta"], 2, 100)
    chk(ln, f"табл. інтерп. {fm}: ε*, %", ("28.9" if fm == "homog" else "29.3"), i["s"], 1, 100)
    chk(ln, f"табл. інтерп. {fm}: P*", ("0.527" if fm == "homog" else "0.533"), i["P_star"], 3)
    # узгодженість інтерпольованих рядків між собою: p*/pc − 1 проти сплайна s
    NUM.append(dict(line=ln, what=f"узгодж. інтерп. {fm}: p*(сплайн)/pc(сплайн) − 1 проти s(сплайн), %",
                    text=f"{100 * i['s']:.2f}", recomputed=100 * (i["p_star"] / i["pc"] - 1), status="info"))
chk(300, "табл. ГЦК pc", "0.1202", A3["pc"], 4)
chk(300, "табл. ГЦК Δ, %", "4.0", nv["rel_p_inf"], 1, 100)
chk(300, "табл. ГЦК ε∞, %", "5.4", dec["s"], 1, 100)
chk(300, "табл. ГЦК ε∞ похибка, %", "0.8", dec["err_s"], 1, 100)
chk(300, "табл. ГЦК P*", "0.474", A3["P_star_largest"], 3)
# монотонність Δ
for fm in ("homog", "gleeson"):
    rr = sorted(fam[fm].values(), key=lambda r: r["t"])
    mono = all(b.get("delta", 0) > a.get("delta", 0) for a, b in zip(rr, rr[1:]))
    NUM.append(dict(line=271, what=f"Δ монотонно зростає з N3 ({fm})", text="так", recomputed=mono,
                    status="ok" if mono else "FALSE"))
# підпис рис., р. 307
allS = [r["s"] for fm in ("homog", "gleeson") for r in fam[fm].values()]
chk(307, "підпис: ε* min, %", "26", min(allS), 0, 100)
chk(307, "підпис: ε* max, %", "34", max(allS), 0, 100)
chk(307, "підпис: ГЦК ε∞, %", "5.4", dec["s"], 1, 100)
chk(307, "підпис: кількість оцінювачів у смузі", "4", len(fr["by_estimator"]), 0)
# р. 311
chk(311, "(i) ε* ≈ 29% (однорідна)", "29", I24["homog"]["s"], 0, 100)
chk(311, "(i) ε* ≈ 29% (Gleeson)", "29", I24["gleeson"]["s"], 0, 100)
chk(311, "(i) P* ≈ 0.53 (однорідна)", "0.53", I24["homog"]["P_star"], 2)
chk(311, "(i) P* ≈ 0.53 (Gleeson)", "0.53", I24["gleeson"]["P_star"], 2)
chk(311, "(ii) розбіжність сімей, в. п.", "1.8", I24["gleeson"]["delta"] - I24["homog"]["delta"], 1, 100)
chk(311, "(ii) |S2| при N3 = 24", "84", 12 * 11 - 2 * 24, 0)
chk(311, "(ii) 2N3/z", "4", 2 * 24 / 12, 0)
chk(311, "(iii) N4 ГЦК", "132", A3["loops"]["c4"], 0)
chk(311, "(iii) |S2| ГЦК", "42", A3["loops"]["sphere2"], 0)
# висновок, р. 458
chk(458, "висновок: Δ дерев клік min, %", "4.6", I24["homog"]["delta"], 1, 100)
chk(458, "висновок: Δ дерев клік max, %", "6.4", I24["gleeson"]["delta"], 1, 100)
chk(458, "висновок: ГЦК min, %", "4.0", fr["min"], 1, 100)
chk(458, "висновок: ГЦК max, %", "5.8", fr["max"], 1, 100)
# дод. Д, р. 621–622
s = CV["summary"]
chk(621, "МК: коренів", "60000", CV["meta"]["params"]["NROOTS"], 0)
chk(621, "МК: обрізання s_k > 1e4", "10000", CV["meta"]["params"]["S_CAP"], 0)
chk(621, "МК: точок", "15", s["n"], 0)
chk(621, "МК: середнє z", "-0.32", s["mean_z"], 2)
chk(621, "МК: SD z", "1.08", s["sd_z"], 2)
chk(621, "МК: max|z|", "2.39", s["max_abs_z"], 2)
chk(621, "МК: max|z_P|", "1.92", max(abs(q["zP"]) for c in CV["cases"].values() for q in c["points"]), 2)
chk(621, "h/2: max |Δp*| ≤ 4e-12", "4e-12", max(abs(r["p_star_h005"] - r["p_star"]) for fm in ("homog", "gleeson")
                                                  for r in fam[fm].values()), tol=4e-12)
# незалежний перерахунок МК-зведення з точок
zs = np.array([q["z"] for c in CV["cases"].values() for q in c["points"]])
zz = np.array([(q["mc"] - q["exact"]) / q["se"] for c in CV["cases"].values() for q in c["points"]])
NUM.append(dict(line=621, what="МК: z перераховано з (mc − exact)/se, max розбіжність з полем z",
                text="—", recomputed=float(np.max(np.abs(zs - zz))), status="ok" if np.max(np.abs(zs - zz)) < 1e-9 else "FALSE"))
NUM.append(dict(line=621, what="МК: SD z (ddof=1)", text="1.08", recomputed=float(zs.std(ddof=1)), status="info"))
# розбіжність діапазону оцінювачів ГЦК з р. 262 і табл. tab:bethe
chk(262, "р. 262: ГЦК верхня межа оцінювачів, %", "6.1", cv["rel_p_inf"], 1, 100,
    note="у р. 56, 268, 271, 307, 458 верхня межа 5.8 — без звичайної кубіки ±0.02 (основного оцінювача, табл. tab:fss)")
OUT["numbers"] = NUM
print(f"[{time.time() - T0:.0f} с] чисел перевірено: {len(NUM)}; FALSE: {sum(x['status'] == 'FALSE' for x in NUM)}",
      flush=True)


# ------------------------------------------------------------------ 2. власні pc і P
def conn_prob(jmax, p):
    q = 1 - p
    C = np.zeros(jmax + 1)
    C[1] = 1.0
    for j in range(2, jmax + 1):
        C[j] = 1 - sum(math.comb(j - 1, i - 1) * C[i] * q ** (i * (j - i)) for i in range(1, j))
    return C


def pi_m(m, p):
    C = conn_prob(m, p)
    q = 1 - p
    return np.array([math.comb(m - 1, j - 1) * C[j] * q ** (j * (m - j)) for j in range(1, m + 1)])


def mean_new(m, p):
    pm = pi_m(m, p)
    return float(sum((j - 1) * pm[j - 1] for j in range(1, m + 1)))


def my_pc(sizes):
    types = sorted(set(sizes))

    def rho(p):
        M = np.zeros((len(types), len(types)))
        for a, m in enumerate(types):
            for b, m2 in enumerate(types):
                M[a, b] = mean_new(m, p) * (list(sizes).count(m2) - (1 if m2 == m else 0))
        return max(abs(np.linalg.eigvals(M))) - 1
    return brentq(rho, 1e-6, 0.99, xtol=1e-15)


def my_P(sizes, p, it=100000):
    types = sorted(set(sizes))
    h = {m: 0.0 for m in types}
    for _ in range(it):
        prod = np.prod([h[m] for m in sizes])
        new = {}
        for m in types:
            H = np.prod([h[m2] for k, m2 in enumerate(sizes) if k != list(sizes).index(m)])
            new[m] = float(sum(pi_m(m, p)[j - 1] * H ** (j - 1) for j in range(1, m + 1)))
        if max(abs(new[m] - h[m]) for m in types) < 1e-15:
            h = new
            break
        h = new
    return 1 - np.prod([h[m] for m in sizes])


ind = {}
for fm in ("homog", "gleeson"):
    for key, r in fam[fm].items():
        sz = tuple(r["sizes"])
        pc_me = my_pc(sz)
        P_me = my_P(sz, r["p_star"])
        d = dict(sizes=list(sz), pc_json=r["pc"], pc_mine=pc_me, d_pc=pc_me - r["pc"],
                 P_star_json=r["P_star"], P_star_mine=P_me, d_P=P_me - r["P_star"])
        ind[f"{fm}:{key}"] = d
# Gleeson ур. (7): D_c(p) = p·Σ(m − 1)P(m|c) у позначеннях статті Gleeson = p·E_c[j − 1]; поріг: p(12 − c) + (13 − c)·p·E_c[j−1] = 1
for c in range(3, 11):
    g = brentq(lambda p: p * (12 - c) + (13 - c) * p * mean_new(c, p) - 1, 1e-6, 0.99, xtol=1e-15)
    ind[f"gleeson:{c}"]["pc_gleeson_eq7"] = g
    ind[f"gleeson:{c}"]["d_pc_eq7"] = g - fam["gleeson"][str(c)]["pc"]
roots = [x.real for x in np.roots([-1, 1, 1, -0.1]) if abs(x.imag) < 1e-12 and 0 < x.real < 1]
ind["K3x6_cubic"] = dict(eq="10(p + p² − p³) = 1", roots_in_01=roots, table="0.09227", json=fam["homog"]["3"]["pc"])
ind["max_abs_d_pc"] = max(abs(v["d_pc"]) for k, v in ind.items() if isinstance(v, dict) and "d_pc" in v)
ind["max_abs_d_P"] = max(abs(v["d_P"]) for k, v in ind.items() if isinstance(v, dict) and "d_P" in v)
ind["max_abs_d_pc_eq7"] = max(abs(v["d_pc_eq7"]) for k, v in ind.items() if isinstance(v, dict) and "d_pc_eq7" in v)
OUT["indep_pc"] = ind
print(f"[{time.time() - T0:.0f} с] pc: max|Δ| = {ind['max_abs_d_pc']:.1e}, P*: {ind['max_abs_d_P']:.1e}, "
      f"Gleeson (7): {ind['max_abs_d_pc_eq7']:.1e}; K3×6 корінь {roots}", flush=True)


# ------------------------------------------------------------------ 3. формула шарів
def layer_pr(m, p):
    """Власна реалізація формули ймовірності набору шарів з дод. Д (без коду cactusexact)."""
    q = 1 - p
    out = {}

    def comps(total):
        yield ()
        for f in range(1, total + 1):
            for rest in comps(total - f):
                yield (f,) + rest
    for comp in comps(m - 1):
        n = (1,) + comp
        u = m - 1 - sum(comp)
        w = math.factorial(m - 1) / (math.prod(math.factorial(x) for x in comp) * math.factorial(u))
        w *= q ** (u * (1 + sum(comp)))
        for i in range(len(comp)):
            w *= (1 - q ** n[i]) ** n[i + 1] * q ** (n[i + 1] * sum(n[:i]))
        out[comp] = w
    return out


lay = {}
for m in range(2, 11):
    lay[f"norm_m{m}"] = max(abs(sum(layer_pr(m, p).values()) - 1) for p in (0.05, 0.13, 0.5, 0.9))
# перебір для m = 7 (2^21 графів): розподіл n_1..n_6 і E[e_r | шари] через _rtables_enum проти _rtables
for p in (0.13, 0.45):
    r1, j1, s1 = ce._rtables(7, p, 7)
    r2, j2, s2 = ce._rtables_enum(7, p, 7)
    worst = 0.0
    for r in range(6):
        d1 = {tuple(x): y for x, y in zip(r1[s1[r]:s1[r + 1]], j1[s1[r]:s1[r + 1]])}
        d2 = {tuple(x): y for x, y in zip(r2[s2[r]:s2[r + 1]], j2[s2[r]:s2[r + 1]])}
        assert d1.keys() == d2.keys()
        worst = max(worst, max(float(np.max(np.abs(d1[k] - d2[k]))) for k in d1))
    lay[f"m7_enum_vs_comb_p{p}"] = worst
    # власна формула проти перебору (лише ймовірності шарів, r = 6 — повні набори)
    mine = layer_pr(7, p)
    d2 = {tuple(int(v) for v in x if v): y[0] for x, y in zip(r2[s2[5]:s2[6]], j2[s2[5]:s2[6]])}
    lay[f"m7_myformula_vs_enum_p{p}"] = max(abs(mine.get(k, 0.0) - d2.get(k, 0.0)) for k in set(mine) | set(d2))


@nb.njit(cache=True)
def _mc_layers(m, p, N, seed):
    np.random.seed(seed)
    acc = np.zeros(m)          # E[e_r], r = 1..m−1
    acc2 = np.zeros(m)         # E[n_r]
    adj = np.zeros((m, m), np.bool_)
    dist = np.empty(m, np.int64)
    qq = np.empty(m, np.int64)
    for _ in range(N):
        for a in range(m):
            for b in range(a + 1, m):
                o = np.random.random() < p
                adj[a, b] = o
                adj[b, a] = o
        dist[:] = -1
        dist[0] = 0
        qq[0] = 0
        lo, hi = 0, 1
        while lo < hi:
            u = qq[lo]
            lo += 1
            for v in range(m):
                if adj[u, v] and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    qq[hi] = v
                    hi += 1
        for v in range(1, m):
            if dist[v] > 0:
                acc2[dist[v]] += 1
        for r in range(1, m):
            c = 0
            for a in range(m):
                for b in range(a + 1, m):
                    if adj[a, b] and 0 <= dist[a] <= r and 0 <= dist[b] <= r:
                        c += 1
            acc[r] += c
    return acc / N, acc2 / N


for m in (10,):
    p = 0.2
    N = 400000
    mc_e, mc_n = _mc_layers(m, p, N, 7331)
    r1, j1, s1 = ce._rtables(m, p, m)
    ex_e = [float(j1[s1[r]:s1[r + 1], 1].sum()) for r in range(m - 1)]
    tab = layer_pr(m, p)
    ex_n = [sum(w * (comp[r - 1] if len(comp) >= r else 0) for comp, w in tab.items()) for r in range(1, m)]
    lay[f"mc_m{m}_p{p}"] = dict(N=N, E_e_r_exact=ex_e, E_e_r_mc=mc_e[1:].tolist(), E_n_r_myformula=ex_n,
                                E_n_r_mc=mc_n[1:].tolist(),
                                max_rel_dev_e=float(max(abs(a - b) / max(b, 1e-9) for a, b in zip(mc_e[1:], ex_e) if b > 1e-3)),
                                max_abs_dev_n=float(max(abs(a - b) for a, b in zip(mc_n[1:], ex_n))))
OUT["layers"] = lay
print(f"[{time.time() - T0:.0f} с] шари: {json.dumps({k: v for k, v in lay.items() if not isinstance(v, dict)})}",
      flush=True)


# ------------------------------------------------------------------ 4. |S2|, ГЦК
def cactus_ball(sizes, R=2):
    """Явна околиця радіуса R дерева клік (без перколяції): вузли на відстані 1 і 2 від кореня."""
    G = nx.Graph()
    G.add_node(0)
    nid = [1]
    frontier = [(0, None)]          # (вузол, індекс кліки, через яку увійшли)
    for _ in range(R):
        nxt = []
        for v, ent in frontier:
            for ci, m in enumerate(sizes):
                if ci == ent:
                    continue
                mem = [v] + list(range(nid[0], nid[0] + m - 1))
                nid[0] += m - 1
                for a, b in combinations(mem, 2):
                    G.add_edge(a, b)
                nxt += [(w, ci) for w in mem[1:]]
        frontier = nxt
    d = nx.single_source_shortest_path_length(G, 0, cutoff=2)
    return sum(1 for x in d.values() if x == 1), sum(1 for x in d.values() if x == 2), \
        sum(1 for c in nx.cycle_basis(G.subgraph([k for k, x in d.items() if x <= 1])) if len(c) == 4)


s2 = {}
for fm in ("homog", "gleeson"):
    for key, r in fam[fm].items():
        z1, n2, _ = cactus_ball(tuple(r["sizes"]))
        s2[f"{fm}:{key}"] = dict(N3=r["t"], S1=z1, S2_bfs=n2, S2_formula=12 * 11 - 2 * r["t"],
                                 ok=(n2 == 12 * 11 - 2 * r["t"]))
for sz in ((8, 4, 2, 2), (6, 4, 3, 2)):
    z1, n2, _ = cactus_ball(sz)
    t = sum(math.comb(m - 1, 2) for m in sz)
    zz_ = sum(m - 1 for m in sz)
    s2[str(sz)] = dict(N3=t, z=zz_, S2_bfs=n2, S2_formula=zz_ * (zz_ - 1) - 2 * t, ok=(n2 == zz_ * (zz_ - 1) - 2 * t))
# нерегулярне дерево клік (корінь з (3,3,2,...) , сусіди іншого складу) — формула для «будь-якого дерева» не загальна
s2["note"] = ("формула |S2| = z(z−1) − 2N3 виконується, якщо всі вузли мають той самий степінь z (як у всіх структурах "
              "статті); для дерева клік з різними складами вузлів вона не загальна")
# ГЦК: періодична, L = 4 кубічні комірки (бокс 8), вузли x+y+z парне
Lb = 8
F = nx.Graph()
pts = [(x, y, z) for x in range(Lb) for y in range(Lb) for z in range(Lb) if (x + y + z) % 2 == 0]
for (x, y, z) in pts:
    for dx, dy, dz in [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]:
        F.add_edge((x, y, z), ((x + dx) % Lb, (y + dy) % Lb, (z + dz) % Lb))
v0 = (0, 0, 0)
Nb = list(F[v0])
tri = sum(1 for a, b in combinations(Nb, 2) if F.has_edge(a, b))
c4 = sum(len((set(F[a]) & set(F[c])) - {v0}) for a, c in combinations(Nb, 2))
dd = nx.single_source_shortest_path_length(F, v0, cutoff=2)
per_edge = [len(set(F[v0]) & set(F[a])) for a in Nb]
s2["fcc"] = dict(z=F.degree(v0), N3=tri, N4=c4, S2=sum(1 for x in dd.values() if x == 2),
                 triangles_per_edge=sorted(set(per_edge)))


def c4_counts(G, v):
    """Прості 4-цикли через v (як у lattices3d.loops_per_node) і з них — без хорд (v–x і a–c не суміжні)."""
    Nb = list(G[v])
    tot = chordless = 0
    for a, c in combinations(Nb, 2):
        for x in (set(G[a]) & set(G[c])) - {v}:
            tot += 1
            if not G.has_edge(a, c) and not G.has_edge(v, x):
                chordless += 1
    return tot, chordless


def cactus_graph(sizes, R=2):
    G = nx.Graph()
    G.add_node(0)
    nid = 1
    frontier = [(0, None)]
    for _ in range(R):
        nxt = []
        for v, ent in frontier:
            for ci, m in enumerate(sizes):
                if ci == ent:
                    continue
                mem = [v] + list(range(nid, nid + m - 1))
                nid += m - 1
                G.add_edges_from(combinations(mem, 2))
                nxt += [(w, ci) for w in mem[1:]]
        frontier = nxt
    return G


n4 = {}
for fm in ("homog", "gleeson"):
    for key, r in fam[fm].items():
        tot, cl = c4_counts(cactus_graph(tuple(r["sizes"])), 0)
        n4[f"{fm}:{key}"] = dict(N3=r["t"], N4=tot, N4_formula=sum(3 * math.comb(m - 1, 3) for m in r["sizes"]),
                                 N4_chordless=cl)
for sz in ((8, 4, 2, 2),):
    tot, cl = c4_counts(cactus_graph(sz), 0)
    n4[str(sz)] = dict(N3=24, N4=tot, N4_chordless=cl)
tot, cl = c4_counts(F, v0)
n4["fcc"] = dict(N3=tri, N4=tot, N4_chordless=cl)
for fm in ("homog", "gleeson"):
    rr = sorted((v for k, v in n4.items() if k.startswith(fm)), key=lambda d: d["N3"])
    n4[f"{fm}_interp_N4_at_24"] = float(np.interp(24, [d["N3"] for d in rr], [d["N4"] for d in rr]))
s2["n4"] = n4
# твердження тексту про 4-цикли (визначення N4 — як у табл. tab:bethe: усі прості 4-цикли через вузол)
mx = max(v["N4"] for k, v in n4.items() if isinstance(v, dict) and k != "fcc")
NUM.append(dict(line=268, what="«а 4-циклів немає» у дереві клік", text="0", recomputed=mx, status="FALSE",
                note="4-цикли є всередині клік K_m, m ≥ 4: N4 = Σ 3·C(m−1, 3) (K4×4: 12, K7×2: 120, K10+3K2: 252); "
                     "немає лише 4-циклів без хорд (поза кліками)"))
NUM.append(dict(line=311, what="(iii) «не відтворює 4-циклів ГЦК (N4 = 132)»", text="132 (не відтворено)",
                recomputed=n4["gleeson_interp_N4_at_24"], status="FALSE",
                note=f"сім'я Gleeson при N3 = 24 (лінійно між c = 8 і 9) має N4 ≈ {n4['gleeson_interp_N4_at_24']:.0f}, "
                     f"однорідна ≈ {n4['homog_interp_N4_at_24']:.0f}, K8+K4+2K2 — {n4['(8, 4, 2, 2)']['N4']}; "
                     f"не відтворено лише 4-цикли без хорд: у ГЦК {n4['fcc']['N4_chordless']} з {n4['fcc']['N4']}, у деревах клік 0"))
OUT["s2"] = s2
print(f"[{time.time() - T0:.0f} с] N4: " + "; ".join(f"{k}: {v['N4']}/{v['N4_chordless']}" for k, v in n4.items()
                                                   if isinstance(v, dict)), flush=True)
print(f"[{time.time() - T0:.0f} с] |S2|: усі ok = {all(v['ok'] for k, v in s2.items() if isinstance(v, dict) and 'ok' in v)}; "
      f"ГЦК {s2['fcc']}", flush=True)


# ------------------------------------------------------------------ 5. власне МК на дереві клік
@nb.njit(cache=True)
def _clique(m, p, adj, dist, qq):
    for a in range(m):
        for b in range(a + 1, m):
            o = np.random.random() < p
            adj[a, b] = o
            adj[b, a] = o
    dist[:m] = -1
    dist[0] = 0
    qq[0] = 0
    lo, hi = 0, 1
    while lo < hi:
        u = qq[lo]
        lo += 1
        for v in range(m):
            if adj[u, v] and dist[v] < 0:
                dist[v] = dist[u] + 1
                qq[hi] = v
                hi += 1


@nb.njit(parallel=True, cache=True)
def _mc_root(sizes, p, nroots, seed, cap, KMAX, NCAP):
    nc = len(sizes)
    mm = 0
    for i in range(nc):
        if sizes[i] > mm:
            mm = sizes[i]
    res = np.zeros(nroots)
    tail = np.zeros(nroots)
    capped = np.zeros(nroots, np.bool_)
    overflow = np.zeros(nroots, np.bool_)
    for ir in nb.prange(nroots):
        np.random.seed(seed + 7919 * ir)
        head = -np.ones(KMAX + mm + 2, np.int64)
        nxt = np.empty(NCAP, np.int64)
        ent = np.empty(NCAP, np.int64)        # індекс кліки, через яку увійшли (−1 — корінь)
        S = np.zeros(KMAX + mm + 2, np.int64)
        B = np.zeros(KMAX + mm + 2, np.int64)
        adj = np.zeros((mm, mm), np.bool_)
        dist = np.empty(mm, np.int64)
        qq = np.empty(mm, np.int64)
        memid = np.empty(mm, np.int64)
        nv = 1
        ent[0] = -1
        nxt[0] = -1
        head[0] = 0
        S[0] = 1
        s_cum = 0
        b_cum = 0
        tot = 0.0
        dprev = 0.0
        dcur = 0.0
        maxlev = 0
        L = 0
        while True:
            s_cum += S[L]
            b_cum += B[L]
            if L > 0:
                if S[L] == 0:
                    break
                pk = 2.0 * b_cum / (s_cum * (s_cum - 1.0))
                dprev = dcur
                dcur = pk * (1 - pk)
                tot += dcur
            if s_cum > cap or L >= KMAX:
                capped[ir] = True
                r = dcur / dprev if dprev > 0 else 0.0
                if 0 < r < 1:
                    tail[ir] = dcur * r / (1 - r)
                break
            v = head[L]
            while v >= 0:
                for ci in range(nc):
                    if ci == ent[v]:
                        continue
                    m = sizes[ci]
                    _clique(m, p, adj, dist, qq)
                    for j in range(1, m):
                        if dist[j] > 0:
                            if nv >= NCAP:
                                overflow[ir] = True
                                memid[j] = -1
                                continue
                            lv = L + dist[j]
                            ent[nv] = ci
                            nxt[nv] = head[lv]
                            head[lv] = nv
                            S[lv] += 1
                            if lv > maxlev:
                                maxlev = lv
                            nv += 1
                    for a in range(m):
                        for b in range(a + 1, m):
                            if adj[a, b] and dist[a] >= 0 and dist[b] >= 0:
                                da = dist[a] if dist[a] > dist[b] else dist[b]
                                B[L + da] += 1
                v = nxt[v]
            L += 1
        res[ir] = tot
    return res, tail, capped, overflow


def my_mc(sizes, p, nroots=40000, seed=20261113, cap=20000):
    sz = np.array(sizes, np.int64)
    res, tail, capped, ovf = _mc_root(sz, p, nroots, seed, cap, 400, 60 * cap)
    tot = res + tail
    return dict(p=p, nroots=nroots, cap=cap, seed=seed, mc=float(tot.mean()), se=float(tot.std(ddof=1) / np.sqrt(nroots)),
                tail_mean=float(tail.mean()), capped_frac=float(capped.mean()), overflow=int(ovf.sum()))


mc = {}
_mc_root(np.array((3, 3), np.int64), 0.3, 10, 1, 100, 400, 6000)   # компіляція
for name, sz, pp in (("K2x12", (2,) * 12, fam["homog"]["2"]["p_star"]),
                     ("K3x6", (3,) * 6, fam["homog"]["3"]["p_star"]),
                     ("K3x6_0.8pc", (3,) * 6, 0.8 * fam["homog"]["3"]["pc"]),
                     ("K9+4K2", (9, 2, 2, 2, 2), fam["gleeson"]["9"]["p_star"])):
    t1 = time.time()
    d = my_mc(sz, pp)
    d["exact_quad"] = ce.nsc(sz, pp)
    d["z"] = (d["mc"] - d["exact_quad"]) / d["se"]
    d["P_exact_mine"] = my_P(sz, pp)
    d["zP"] = (d["capped_frac"] - d["P_exact_mine"]) / math.sqrt(max(d["P_exact_mine"] * (1 - d["P_exact_mine"]), 1e-12)
                                                               / d["nroots"])
    d["sec"] = time.time() - t1
    mc[name] = d
    print(f"[{time.time() - T0:.0f} с] МК {name} p={pp:.5f}: {d['mc']:.5f} ± {d['se']:.5f} проти {d['exact_quad']:.5f} "
          f"(z = {d['z']:+.2f}); обрізано {d['capped_frac']:.4f} проти P = {d['P_exact_mine']:.4f} (z = {d['zP']:+.2f}); "
          f"хвіст {d['tail_mean']:.1e}; переповнень {d['overflow']}", flush=True)
OUT["mc"] = mc

# ------------------------------------------------------------------ 6. інтерполяція
interp = {}
for fm in ("homog", "gleeson"):
    rr = sorted(fam[fm].values(), key=lambda r: r["t"])
    t = np.array([r["t"] for r in rr], float)
    d = np.array([r.get("delta", 0.0) for r in rr])
    o = {"cubic_spline_not_a_knot": float(CubicSpline(t, d)(24)), "cubic_spline_natural":
         float(CubicSpline(t, d, bc_type="natural")(24)), "pchip": float(PchipInterpolator(t, d)(24)),
         "akima": float(Akima1DInterpolator(t, d)(24)), "linear": float(np.interp(24, t, d))}
    i = np.argsort(np.abs(t - 24))[:3]
    o["quadratic_3_nearest"] = float(np.polyval(np.polyfit(t[i], d[i], 2), 24))
    i = np.argsort(np.abs(t - 24))[:4]
    o["cubic_4_nearest"] = float(np.polyval(np.polyfit(t[i], d[i], 3), 24))
    o["range"] = [min(o.values()), max(o.values())]
    interp[fm] = o
# сім'я Gleeson з плану (c ∈ {5, 7, 8, 9}; t = 6, 15, 21, 28), як до розширення
rr = [fam["gleeson"][str(c)] for c in (5, 7, 8, 9)]
t = np.array([r["t"] for r in rr], float)
d = np.array([r["delta"] for r in rr])
interp["gleeson_planned_set"] = dict(cubic_spline=float(CubicSpline(t, d)(24)), linear=float(np.interp(24, t, d)),
                                     pchip=float(PchipInterpolator(t, d)(24)))
# у змінній Λ3 = N3·p*³ (очікувані трикутники, що вижили) — інша природна вісь
OUT["interp"] = interp
print(f"[{time.time() - T0:.0f} с] інтерполяція: {json.dumps({k: v.get('range') for k, v in interp.items()})}; "
      f"план Gleeson: {interp['gleeson_planned_set']}", flush=True)

# ------------------------------------------------------------------ 7. справжня структура з N3 = 24
extra = {}
for sz in ((8, 4, 2, 2), (7, 6, 2), (8, 3, 3, 2)):
    t1 = time.time()
    pc_ = my_pc(sz)
    lo, hi = 1.25 * pc_, 1.42 * pc_
    ps_ = ce.peak(sz, lo, hi)
    extra[str(sz)] = dict(z=sum(m - 1 for m in sz), N3=sum(math.comb(m - 1, 2) for m in sz), pc=pc_, p_star=ps_,
                          delta=ps_ / pb - 1, eps=ps_ / pc_ - 1, P_star=my_P(sz, ps_), sec=time.time() - t1)
    print(f"[{time.time() - T0:.0f} с] {sz}: N3={extra[str(sz)]['N3']} pc={pc_:.5f} p*={ps_:.5f} "
          f"Δ={100 * extra[str(sz)]['delta']:+.2f}% ε*={100 * extra[str(sz)]['eps']:.1f}%", flush=True)
OUT["extra"] = extra

# ------------------------------------------------------------------ 8. критерій і розклад
crit = {}
ests = dict(fr["by_estimator"], conventional=cv["rel_p_inf"])
crit["fcc_estimators"] = ests


def verdict(x, lo, hi):
    if lo <= x <= hi:
        return "узгоджено"
    return "не підтверджено" if (x < 0.02 or x > 0.08) else "не розрізнено"


for fm in ("homog", "gleeson"):
    for how in ("delta", "delta_linear"):
        x = I24[fm][how]
        crit[f"{fm}:{how}"] = dict(value=x, four_estimators=verdict(x, fr["min"], fr["max"]),
                                   five_estimators=verdict(x, min(ests.values()), max(ests.values())),
                                   z_vs_narrow=(x - nv["rel_p_inf"]) / nv["err_rel_p_inf"],
                                   z_vs_conventional=(x - cv["rel_p_inf"]) / cv["err_rel_p_inf"])
for k, v in extra.items():
    if v["N3"] == 24:
        crit[f"extra{k}"] = dict(value=v["delta"], four_estimators=verdict(v["delta"], fr["min"], fr["max"]),
                                 z_vs_narrow=(v["delta"] - nv["rel_p_inf"]) / nv["err_rel_p_inf"])
# розклад 1 + Δ = (pc/pc_Б)·(1 + ε*)/(1 + ε*_Б)
B = CA["bethe"]
for fm in ("homog", "gleeson"):
    i = I24[fm]
    crit[f"decomp_{fm}"] = dict(pc_ratio_minus1=i["pc"] / B["pc"] - 1, eps_ratio_minus1=(1 + i["s"]) / (1 + B["s"]) - 1)
crit["decomp_fcc"] = dict(pc_ratio_minus1=A3["pc"] / B["pc"] - 1, eps_ratio_minus1=(1 + dec["s"]) / (1 + B["s"]) - 1)
# що дала б модель «пік на ε̄∞ = 4.49% над власним pc» для ГЦК (розд. 5): p = 1.0449·pc(ГЦК)
crit["near_pc_model_fcc"] = dict(p_pred=1.04489 * A3["pc"], delta_pred=1.04489 * A3["pc"] / pb - 1)
# 4-цикли: для локальності важливий і внесок N4 (кубічна: N4 = 12, Δ = +2.1%; ОЦК: N4 = 48, Δ = +1.1%)
crit["loops_other_3D"] = {k: dict(N3=A["loops"]["c3"], N4=A["loops"]["c4"], S2=A["loops"]["sphere2"],
                                  delta_narrow=A["vs_bethe"]["narrow"]["rel_p_inf"],
                                  err=A["vs_bethe"]["narrow"]["err_rel_p_inf"],
                                  eps_inf=A["vs_bethe"]["decomposition"]["s"])
                          for k, A in J("results/P1/p1_3b_bethe_lattices.json")["lattices"].items()
                          if k in ("diamond", "sc", "bcc", "fcc")}
# «еластичність» піку щодо порога всередині сімей: Δln p* / Δln pc від Бете до найбільшого N3
for fm in ("homog", "gleeson"):
    top = max(fam[fm].values(), key=lambda r: r["t"])
    crit[f"elasticity_{fm}"] = dict(N3=top["t"], dln_pstar=math.log(top["p_star"] / B["p_star"]),
                                    dln_pc=math.log(top["pc"] / B["pc"]),
                                    ratio=math.log(top["p_star"] / B["p_star"]) / math.log(top["pc"] / B["pc"]))
crit["elasticity_fcc_vs_bethe"] = dict(ratio=math.log(A3["narrow"]["p_inf"] / B["p_star"]) / math.log(A3["pc"] / B["pc"]))
# де виникає різниця сімей: профілі E[δ_k] двох структур з однаковими z = 12, N3 = 6, |S2| = 120 при p = 0.123
pa = ce.profile((3,) * 6, 0.123, K=12)
pg = ce.profile((5,) + (2,) * 8, 0.123, K=12)
crit["profile_pair_N3_6_p0.123"] = dict(K3x6=pa[:6].tolist(), K5_8K2=pg[:6].tolist(), diff_k1=float(pg[0] - pa[0]),
                                        diff_k2=float(pg[1] - pa[1]), diff_k_ge3=float(pg[2:].sum() - pa[2:].sum()),
                                        diff_total=float(pg.sum() - pa.sum()))
OUT["criterion"] = crit

OUT["meta"] = dict(script="paper/P1/referee_cactus_checks.py", time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   sec=time.time() - T0, numpy=np.__version__, numba=nb.__version__)
(ROOT / "paper/P1/referee_cactus_checks.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float))
print(f"готово, {time.time() - T0:.0f} с")
