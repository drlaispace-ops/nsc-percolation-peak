"""П1.3b. Аналіз: пік NSC на решітках з різними короткими петлями проти гратки Бете з тим самим z.

Для кожної задачі (решітки П1.3b — results/P1/p1_3b/<решітка>_L<L>.json; решітки П1.3 — results/P1/p1_3/*.json) із сирих
шматків (data/P1/p1_3b, data/P1/p1_3 — лише читання):
  * конвенційна оцінка П1.3 — p1_3_analyse.stats_one: кубіка у вікні ±HW (0.02; 2D 0.012), бутстреп, систематика
    вікна (кубіка 0.6·HW і 1.5·HW, квартика 1.5·HW), P*, p½, p_χ (для П1.3 — взято з results/P1/p1_3_analysis.json);
  * вузьковіконна оцінка: середнє p* кубік у вікнах 0.008, 0.010, 0.012, 0.015 (2D: 0.006–0.012). Причина: на ОЦК і ГЦК
    пік асиметричний (крутий підйом, повільний спад) і кубіка у вікні ±0.02 зсуває p* у бік спаду на 0.0013–0.0026
    (ГЦК L = 32: 0.1274 при ±0.012 і 0.1301 при ±0.02); при hw ≤ 0.015 p*(hw) ще слабко дрейфує (ГЦК L = 40:
    0.1270…0.1282), тож залежність від оцінювача — систематика (діапазон оцінювачів — notes/orchestra/verify_A.json).
    Гратка Бете дає точний argmax, тому для порівняння з нею основна — вузьковіконна оцінка; конвенційна теж звітується.
p∞: підгонка p*(L) = p∞ + a·L^(−1/ν) (ν з data/P1/constants.csv для своєї d) і плато (зважене середнє двох найбільших L):
пік лежить на 3–6% (3D) над pc, де ξ скінченна, тож степеневий закон поправок не гарантований; похибка p∞ = SE_fit ⊕ |p∞ − плато|,
похибка плато = SE·max(1, √χ²) ⊕ |плато − p∞| (після перевірки A_verdict п. 2.2). Тести моделей «Бете» / «біля pc» і Спірмен — discrimination().
Бете (RRG, n → ∞, bond) з тим самим z — точна квадратура gwexact (як code/P1/p1_3_bethe.py; NSC_∞ спирається на
недоведену лему про хвіст, П1.16).
Решітки П1.3 для порівняння: 2D квадратна і трикутна, 3D sc, 4D–6D (bond).
Міри коротких петель на вузол (нескінченна решітка): c_ℓ — прості цикли довжини ℓ = 3…6 через вузол (перебір шляхів
у періодичному графі з петлею навколо тора > 6); |S₂| і дефіцит другої сфери D₂ = 1 − |S₂|/(z(z − 1));
коефіцієнт кластеризації C = c₃/(z(z − 1)/2); очікувана кількість петель через вузол при p = p*_Бете:
Λ₄ = Σ_{ℓ=3,4} c_ℓ p^ℓ, Λ₆ = Σ_{ℓ=3..6} c_ℓ p^ℓ (петля довжини ℓ виживає з імовірністю p^ℓ).
Запуск: .venv/bin/python code/P1/p1_3b_analyse.py → results/P1/p1_3b_bethe_lattices.json, results/P1/fig_p1_3b_bethe.png
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
sys.path.insert(0, str(ROOT / "code" / "P1"))
import gwexact as gx  # noqa: E402
import lattices3d as lt  # noqa: E402
import p1_3b_run as run  # noqa: E402  (імпорт без запуску)
import sampled as sm  # noqa: E402
from p1_3_analyse import CONST as CONST13, poly_peak, stats_one  # noqa: E402
import p1_3_run as run13  # noqa: E402
from p1_3_run import load  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

OUT = ROOT / "results" / "P1" / "p1_3b_bethe_lattices.json"
FIG = ROOT / "results" / "P1" / "fig_p1_3b_bethe.png"
CONST = {(r["system"], r["quantity"]): float(r["value"]) for r in csv.DictReader(open(ROOT / "data" / "P1" / "constants.csv"))}
NU3 = CONST[("perc_3D", "nu")]
A13 = json.load(open(ROOT / "results" / "P1" / "p1_3_analysis.json"))["lattices"]
SEED_A = 20261124


# ------------------------------------------------------------------ Бете
def bethe(z):
    m = f"z{z}"
    pc = 1 / (z - 1)
    p_half = brentq(lambda p: gx.giant(p, m) - 0.5, pc * (1 + 1e-6), 1 - 1e-9, xtol=1e-12)
    grid = np.linspace(pc * 1.01, min(3 * p_half - 2 * pc, 0.99), 41)
    y = np.array([gx.nsc_quad(m, p) for p in grid])
    i = int(np.argmax(y))
    ps = gx.peak(m, grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)])
    return dict(z=z, pc=pc, p_star=ps, P_star=gx.giant(ps, m), nsc_star=gx.nsc_quad(m, ps), p_half=p_half)


# ------------------------------------------------------------------ короткі петлі
def cycles_through(A, s, kmax=6):
    """Кількість простих циклів довжини 3..kmax через вузол s (перебір шляхів / 2)."""
    ip, ix = A.indptr, A.indices
    cnt = np.zeros(kmax + 1, np.int64)
    path = [s]
    on = {s}

    def dfs(x, depth):
        for y in ix[ip[x]:ip[x + 1]]:
            if y == s and depth >= 2:
                cnt[depth + 1] += 1
            elif y not in on and depth + 1 < kmax:
                on.add(y)
                dfs(y, depth + 1)
                on.discard(y)
    dfs(s, 0)
    return {l: int(cnt[l] // 2) for l in range(3, kmax + 1)}


def lattice_graph(key):
    """Граф решітки з петлею навколо тора > 6 кроків (для підрахунку циклів довжини ≤ 6)."""
    rng = np.random.default_rng(0)
    if key in lt.EDGES:
        n, u, v = lt.EDGES[key]({"sc": 7, "bcc": 4, "fcc": 4, "diamond": 2}[key])
    elif key == "tri":
        n, u, v, _, _ = sm.coupled_triangular(8, rng)
    else:                                                  # "sq" або "hc<d>"
        d = 2 if key == "sq" else int(key[2:])
        n, u, v, _, _ = sm.coupled_bonds(7, d, rng)
    return n, u, v


def loops(key):
    n, u, v = lattice_graph(key)
    A = lt.adjacency(n, u, v)
    lp = lt.loops_per_node(n, u, v)
    z = int(round(lp["z"]))
    srcs = (0, n - 1) if key in ("bcc", "diamond") else (0,)        # обидві підгратки
    cs = [cycles_through(A, s) for s in srcs]
    assert all(c == cs[0] for c in cs)
    c = cs[0]
    assert c[3] == lp["triangles"] and c[4] == lp["squares"]
    return dict(z=z, c3=c[3], c4=c[4], c5=c[5], c6=c[6], sphere2=lp["sphere2"],
                D2=1 - lp["sphere2"] / (z * (z - 1)), clustering=c[3] / (z * (z - 1) / 2))


# ------------------------------------------------------------------ p*(L) і p∞
HW_NARROW = {2: (0.006, 0.008, 0.010, 0.012), 3: (0.008, 0.010, 0.012, 0.015)}    # d ≥ 3 — як 3
B_NARROW = 300
P13 = {"sq": "2D bond sq", "tri": "2D bond tri", "sc": "3D bond", "hc4": "4D bond", "hc5": "5D bond", "hc6": "6D bond"}


def narrow_peak(ps, Y, d, seed):
    """Вузьковіконна оцінка: середнє p* кубік у вікнах HW_NARROW (усі в області, де p*(hw) вже не залежить від hw),
    SE — бутстреп цього середнього за реалізаціями, syst — піврозмах за вікнами."""
    hws = HW_NARROW[min(d, 3)]
    est = lambda Ym: np.array([poly_peak(ps, Ym, h)[0] for h in hws])
    full = est(Y.mean(0))
    rng = np.random.default_rng(seed)
    R = len(Y)
    bs = np.array([np.nanmean(est(Y[rng.integers(0, R, R)].mean(0))) for _ in range(B_NARROW)])
    return dict(p_star_narrow=float(np.nanmean(full)), se_p_star_narrow=float(np.nanstd(bs, ddof=1)),
                syst_narrow=float((np.nanmax(full) - np.nanmin(full)) / 2), p_star_by_hw=dict(zip(map(str, hws), full.tolist())))


def fit_pinf(Ls, p, sp, nu):
    f = lambda L, pinf, a: pinf + a * L ** (-1 / nu)
    (pinf, a), cov = curve_fit(f, Ls, p, sigma=sp, absolute_sigma=True, p0=[p[-1], 0.0])
    chi2 = float(np.sum(((p - f(Ls, pinf, a)) / sp) ** 2) / max(len(p) - 2, 1))
    return float(pinf), float(np.sqrt(cov[0, 0])), float(a), chi2


def extrapolate(rows, key, se_key, syst_key, nu):
    """p∞: степенева підгонка p∞ + a·L^(−1/ν) і плато (зважене середнє двох найбільших L); err = SE_fit ⊕ |різниця|."""
    Ls = np.array([r["L"] for r in rows], float)
    p = np.array([r[key] for r in rows])
    sp = np.hypot([r[se_key] for r in rows], [r[syst_key] for r in rows])
    pinf, se, a, chi2 = fit_pinf(Ls, p, sp, nu)
    w = 1 / sp[-2:] ** 2
    plat, se_plat = float(np.sum(w * p[-2:]) / w.sum()), float(1 / np.sqrt(w.sum()))
    chi2_plat = float(np.sum(w * (p[-2:] - plat) ** 2))               # 1 ступінь свободи
    birge = float(max(1.0, np.sqrt(chi2_plat)))
    # Виправлення після перевірки (A_verdict п. 2.2): похибка плато = SE·масштаб Бірге ⊕ дрейф |плато − p∞|
    # (раніше — лише статистична SE, що завищувало σ порівнянь, особливо для 6D, де p*(L) ще росте).
    err_plat = float(np.hypot(se_plat * birge, abs(plat - pinf)))
    return dict(p_inf=pinf, se_p_inf=se, a=a, chi2dof=chi2, nu=nu, plateau=plat, se_plateau=se_plat,
                chi2_plateau=chi2_plat, birge_plateau=birge, err_plateau=err_plat,
                last_minus_prev=float(p[-1] - p[-2]),
                syst_model=abs(plat - pinf), err_p_inf=float(np.hypot(se, abs(plat - pinf))))


def main():
    lat = {}
    for key in ("diamond", "bcc", "fcc") + tuple(P13):
        ours = key in run.NAMES
        if ours:
            files = sorted(run.RES.glob(f"{key}_L*.json"), key=lambda f: int(f.stem.split("_L")[-1]))
            d, nu, pc, z = 3, NU3, run.PC[key], lt.Z[key]
            datadir = lambda L: run.DATA / key / f"L{L}"
            conv = {}
        else:
            nm = P13[key]
            files = sorted(run13.RES.glob(f"{run13.SLUG[nm]}_L*.json"), key=lambda f: int(f.stem.split("_L")[-1]))
            d, pc = run13.CFG[nm]["d"], run13.CFG[nm]["pc"]
            nu, z = CONST13[d]["nu"], 6 if key == "tri" else 2 * d
            datadir = lambda L, nm=nm: run13.DATA / run13.SLUG[nm] / f"L{L}"
            conv = {r["L"]: r["all"] for r in A13[nm]["rows"]}
        rows = []
        for f in files:
            sm_ = json.load(open(f))
            L = sm_["meta"]["params"]["L"]
            ps = np.array(sm_["ps"])
            S = load(datadir(L), ["nsc", "P", "chi"])
            st = stats_one(ps, S["nsc"], S["P"], S["chi"], run.HW) if ours else conv[L]
            row = dict(L=L, R=len(S["nsc"]), met_target=sm_["met_target"], exact_check=sm_["exact_check"],
                       **{k: st[k] for k in ("p_star", "se_p_star", "syst_window", "p_star_alt", "P_star", "se_P_star",
                                             "p_half", "se_p_half", "curv")},
                       **narrow_peak(ps, S["nsc"], d, [SEED_A, L, len(lat)]))
            if ours:
                row.update(n=int(lt.EDGES[key](L)[0]), hours=sm_["hours"])
            rows.append(row)
            print(f"{key:8s} L={L:4d} R={row['R']:4d}  p*(±hw)={row['p_star']:.4f}±{row['se_p_star']:.4f} (вікно "
                  f"±{row['syst_window']:.4f})  p*(вузьк.)={row['p_star_narrow']:.4f}±{row['se_p_star_narrow']:.4f}"
                  f" (±{row['syst_narrow']:.4f})  P*={row['P_star']:.3f}±{row['se_P_star']:.3f}  p½={row['p_half']:.4f}",
                  flush=True)
        e = dict(z=z, d=d, pc=pc, rows=rows,
                 source=("results/P1/p1_3b/*.json, data/P1/p1_3b" if ours else
                         f"results/P1/p1_3/{run13.SLUG[P13[key]]}_L*.json, data/P1/p1_3, results/P1/p1_3_analysis.json"),
                 pc_source="Xu2014 (data/P1/constants.csv)" if ours else "code/P1/p1_3_config.json")
        e["conventional"] = extrapolate(rows, "p_star", "se_p_star", "syst_window", nu)
        e["narrow"] = extrapolate(rows, "p_star_narrow", "se_p_star_narrow", "syst_narrow", nu)
        e.update(P_star_largest=rows[-1]["P_star"], se_P_star_largest=rows[-1]["se_P_star"],
                 p_half_largest=rows[-1]["p_half"], se_p_half_largest=rows[-1]["se_p_half"])
        lat[key] = e
    # Бете
    zs = sorted({v["z"] for v in lat.values()})
    B = {z: bethe(z) for z in zs}
    old = {r["z"]: r["p_star"] for r in json.load(open(ROOT / "results" / "P1" / "p1_3_bethe.json"))["rows"]}
    for z in zs:
        if z in old:
            assert abs(B[z]["p_star"] - old[z]) < 1e-6, (z, B[z]["p_star"], old[z])
        print(f"Бете z={z:2d}: pc={B[z]['pc']:.5f} p*={B[z]['p_star']:.5f} P*={B[z]['P_star']:.4f} p½={B[z]['p_half']:.5f}")
    # петлі і порівняння
    for key, e in lat.items():
        lo = loops(key)
        assert lo["z"] == e["z"]
        b = B[e["z"]]
        x = b["p_star"]
        lo.update(Lambda4=lo["c3"] * x ** 3 + lo["c4"] * x ** 4, Lambda6=sum(lo[f"c{l}"] * x ** l for l in range(3, 7)))
        e["loops"], e["bethe"] = lo, b
        cmp = {}
        for est in ("conventional", "narrow"):
            q = e[est]
            cmp[est] = dict(rel_p_inf=(q["p_inf"] - b["p_star"]) / b["p_star"], err_rel_p_inf=q["err_p_inf"] / b["p_star"],
                            rel_plateau=(q["plateau"] - b["p_star"]) / b["p_star"],
                            err_rel_plateau=q["err_plateau"] / b["p_star"],
                            err_rel_plateau_stat=q["se_plateau"] / b["p_star"])
        # Розклад δ = (pc/p*_Б)·(1 + s) − 1, s = p∞/pc − 1 (A_verdict п. 2.1): альтернатива «пік на s над власним pc».
        s_ = (e["narrow"]["p_inf"] - e["pc"]) / e["pc"]
        cmp["decomposition"] = dict(pc_over_bethe=e["pc"] / b["p_star"], s=s_, err_s=e["narrow"]["err_p_inf"] / e["pc"],
                                    delta=(e["pc"] / b["p_star"]) * (1 + s_) - 1,
                                    xi_proxy=float(s_ ** (-e["narrow"]["nu"])) if s_ > 0 else None,
                                    xi_proxy_note="s^(−ν) у кроках решітки з амплітудою 1 (амплітуда ξ невідома)")
        cmp.update(rel_pc=(e["pc"] - b["pc"]) / b["pc"], rel_shift_peak_vs_pc=(e["narrow"]["p_inf"] - e["pc"]) / e["pc"],
                   dP_star=e["P_star_largest"] - b["P_star"],
                   rel_p_half=(e["p_half_largest"] - b["p_half"]) / b["p_half"])
        e["vs_bethe"] = cmp
        c, n_ = cmp["conventional"], cmp["narrow"]
        print(f"{key:8s} z={e['z']:2d} c3..c6={lo['c3']},{lo['c4']},{lo['c5']},{lo['c6']} D2={lo['D2']:.3f} "
              f"Λ4={lo['Lambda4']:.4f} Λ6={lo['Lambda6']:.4f} | Бете {b['p_star']:.4f}; "
              f"±hw: p∞={e['conventional']['p_inf']:.4f} ({100 * c['rel_p_inf']:+.2f}±{100 * c['err_rel_p_inf']:.2f}%), "
              f"плато {100 * c['rel_plateau']:+.2f}% | вузьк.: p∞={e['narrow']['p_inf']:.4f} ({100 * n_['rel_p_inf']:+.2f}±"
              f"{100 * n_['err_rel_p_inf']:.2f}%), плато {100 * n_['rel_plateau']:+.2f}±{100 * n_['err_rel_plateau']:.2f}% | "
              f"pc {100 * cmp['rel_pc']:+.1f}%  P*={e['P_star_largest']:.3f} (Бете {b['P_star']:.3f})  "
              f"p½ {100 * cmp['rel_p_half']:+.2f}%", flush=True)
    tests = discrimination(lat)
    meta = runmeta(__file__, params=dict(halfwidth=run.HW, hw_narrow=HW_NARROW, B_narrow=B_NARROW, nu_3D=NU3,
                                         B_boot_stats_one=500, bethe="gwexact.nsc_quad, h=0.01; peak via gwexact.peak"),
                   seeds=dict(narrow="default_rng([SEED_A, L, lattice_order])", SEED_A=SEED_A))
    save(OUT, meta, lattices=lat, tests=tests)
    figure(lat, tests)


D3 = ("diamond", "sc", "bcc", "fcc")


def discrimination(lat):
    """Тести, додані після перевірки (A_verdict п. 2.1–2.3).
    * pairs: та сама z через розмірність (ОЦК − 4D, ГЦК − 6D) — різниця δ і σ для підгонки і плато (з новою похибкою плато).
    * models_3D: на чотирьох 3D-решітках дві моделі для p∞:
        «Бете»: p∞ = p*_Б(z) (0 параметрів);
        «біля pc»: p∞ = pc·(1 + s̄) (1 параметр s̄ — зважене середнє s за 3D-решітками).
      Залишки у в. п. від p*_Б і χ² з похибкою err_rel_p_inf (вузьковіконна оцінка, підгонка).
    * spearman: ранговий коефіцієнт δ (підгонка) проти Λ₄ для d ≥ 3 і окремо для 3D проти Λ₄, z, s."""
    from scipy.stats import spearmanr
    out = {}
    pairs = {}
    for a, b_ in (("bcc", "hc4"), ("fcc", "hc6")):
        r = {}
        for k, ek in (("fit", "rel_p_inf"), ("plateau", "rel_plateau")):
            va, vb = lat[a]["vs_bethe"]["narrow"], lat[b_]["vs_bethe"]["narrow"]
            dlt = va[ek] - vb[ek]
            err = float(np.hypot(va["err_" + ek], vb["err_" + ek]))
            r[k] = dict(diff=dlt, err=err, n_sigma=abs(dlt) / err)
        pairs[f"{a}-{b_}"] = r
    out["pairs_same_z"] = pairs
    dl = np.array([lat[k]["vs_bethe"]["narrow"]["rel_p_inf"] for k in D3])
    er = np.array([lat[k]["vs_bethe"]["narrow"]["err_rel_p_inf"] for k in D3])
    s = np.array([lat[k]["vs_bethe"]["decomposition"]["s"] for k in D3])
    es = np.array([lat[k]["vs_bethe"]["decomposition"]["err_s"] for k in D3])
    q = np.array([lat[k]["vs_bethe"]["decomposition"]["pc_over_bethe"] for k in D3])
    w = 1 / es ** 2
    sbar = float(np.sum(w * s) / w.sum())
    pred_near = q * (1 + sbar) - 1
    res_b, res_n = dl, dl - pred_near
    out["models_3D"] = dict(
        lattices=list(D3), delta=dl.tolist(), err_delta=er.tolist(), s=s.tolist(), err_s=es.tolist(),
        pc_over_bethe=q.tolist(), s_bar=sbar, se_s_bar=float(1 / np.sqrt(w.sum())),
        bethe=dict(n_params=0, residuals=res_b.tolist(), max_abs_residual=float(np.max(np.abs(res_b))),
                   chi2=float(np.sum((res_b / er) ** 2)), dof=4),
        near_pc=dict(n_params=1, predicted_delta=pred_near.tolist(), residuals=res_n.tolist(),
                     max_abs_residual=float(np.max(np.abs(res_n))), chi2=float(np.sum((res_n / er) ** 2)), dof=3),
        spread_s=float(s.max() - s.min()), spread_delta=float(dl.max() - dl.min()),
        note="похибки δ і s корельовані (обидві з p∞); χ² — орієнтир, не строгий тест")
    keys = [k for k, e in lat.items() if e["d"] >= 3]
    L4 = [lat[k]["loops"]["Lambda4"] for k in keys]
    dd = [lat[k]["vs_bethe"]["narrow"]["rel_p_inf"] for k in keys]
    rho, pv = spearmanr(L4, dd)
    sp3 = {}
    for nm, xs in (("Lambda4", [lat[k]["loops"]["Lambda4"] for k in D3]), ("z", [lat[k]["z"] for k in D3]), ("s", s.tolist())):
        r3, p3 = spearmanr(xs, dl)
        sp3[nm] = dict(rho=float(r3), p=float(p3))
    out["spearman"] = dict(d_ge_3=dict(lattices=keys, rho=float(rho), p=float(pv)), D3_delta_vs=sp3,
                           D3_order_Lambda4=[D3[i] for i in np.argsort([lat[k]["loops"]["Lambda4"] for k in D3])],
                           D3_order_s=[D3[i] for i in np.argsort(s)])
    print("пари з тим самим z:", json.dumps(pairs, ensure_ascii=False))
    print(f"3D: s̄ = {sbar:.4f}; Бете max|залишок| = {100 * np.max(np.abs(res_b)):.2f} в. п., χ² = "
          f"{out['models_3D']['bethe']['chi2']:.1f}/4; біля pc max|залишок| = {100 * np.max(np.abs(res_n)):.2f} в. п., χ² = "
          f"{out['models_3D']['near_pc']['chi2']:.1f}/3")
    print(f"Спірмен δ–Λ₄ (d ≥ 3, {len(keys)} точок): ρ = {rho:.2f}, p = {pv:.2f}; 3D: {sp3}")
    return out


def figure(lat, tests):
    """Лінійна шкала (після перевірки: symlog розтягував малі δ). Ліворуч — δ проти Λ₄ для d ≥ 3, 2D — у вставці.
    Праворуч — дискримінація двох моделей: δ проти pc/p*_Б − 1; «Бете» — y = 0, «біля pc» — y = (1 + x)(1 + s̄) − 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = {"diamond": "алмаз z=4", "bcc": "ОЦК z=8", "fcc": "ГЦК z=12", "sq": "2D квадр. z=4", "tri": "2D трик. z=6",
           "sc": "3D sc z=6", "hc4": "4D z=8", "hc5": "5D z=10", "hc6": "6D z=12"}
    off = {"hc4": (8, -12), "hc5": (6, 8), "hc6": (-38, 10), "sc": (-62, 6), "bcc": (8, -12), "fcc": (-55, 4),
           "diamond": (8, -14), "sq": (6, -10), "tri": (6, 4)}
    col = {4: "#1b9e77", 6: "#d95f02", 8: "#7570b3", 10: "#666666", 12: "#e7298a"}
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4))

    def pts(ax, k, e, x, dx, plateau=True, label=True, fs=8):
        v = e["vs_bethe"]["narrow"]
        c = col[e["z"]]
        new = k in ("diamond", "bcc", "fcc")
        ax.errorbar(x, 100 * v["rel_p_inf"], yerr=100 * v["err_rel_p_inf"], fmt="o", mfc=c if new else "white",
                    mec=c, ecolor=c, ms=7, capsize=3)
        if plateau:
            ax.errorbar(x + dx, 100 * v["rel_plateau"], yerr=100 * v["err_rel_plateau"], fmt="x", color=c, ms=6,
                        capsize=2, alpha=0.7)
        if label:
            ax.annotate(lab[k], (x, 100 * v["rel_p_inf"]), textcoords="offset points", xytext=off.get(k, (8, 4)),
                        fontsize=fs)

    ax = axs[0]
    for k, e in lat.items():
        if e["d"] >= 3:
            pts(ax, k, e, e["loops"]["Lambda4"], 0.0015)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\Lambda_4=c_3p^3+c_4p^4$ при $p=p^*_{\rm Бете}$")
    ax.set_ylabel(r"$\delta=(p_\infty-p^*_{\rm Бете})/p^*_{\rm Бете}$, %")
    ax.set_xlim(-0.005, 0.09)
    ax.set_ylim(-2, 7.5)
    ax.grid(alpha=0.3)
    ax.set_title("а) d ≥ 3 (лінійна шкала); 2D — у вставці", fontsize=9)
    ins = ax.inset_axes([0.07, 0.62, 0.32, 0.32])
    for k, e in lat.items():
        if e["d"] == 2:
            pts(ins, k, e, e["loops"]["Lambda4"], 0.0, plateau=False, fs=7)
    ins.set_xlim(0.09, 0.17)
    ins.set_ylim(18, 40)
    ins.tick_params(labelsize=7)
    ins.set_title("2D, δ у %", fontsize=7)
    ins.grid(alpha=0.3)

    ax = axs[1]
    m = tests["models_3D"]
    sbar = m["s_bar"]
    for k, e in lat.items():
        if e["d"] >= 3:
            x = 100 * (e["vs_bethe"]["decomposition"]["pc_over_bethe"] - 1)
            pts(ax, k, e, x, 0.0, plateau=False)
    xs = np.linspace(-25, 1, 200)
    ax.axhline(0, color="k", lw=1.0, label="модель «Бете»: $p_\\infty=p^*_{\\rm Бете}(z)$")
    ax.plot(xs, 100 * ((1 + xs / 100) * (1 + sbar) - 1), "--", color="0.4",
            label=f"модель «біля pc» (3D): $p_\\infty=p_c(1+\\bar s)$, $\\bar s$={100 * sbar:.1f}%")
    ax.set_xlim(-25, 1)
    ax.set_ylim(-2, 7.5)
    ax.set_xlabel(r"$p_c/p^*_{\rm Бете}-1$, %")
    ax.set_ylabel(r"$\delta$, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("б) δ проти відстані власного pc від p*_Бете: моделі «Бете» і «біля pc»", fontsize=9)
    fig.suptitle("Пік NSC решітки проти гратки Бете з тим самим z. ● — П1.3b, ○ — П1.3; ○/● — p∞ з підгонки L^(−1/ν) "
                 "(похибка SE ⊕ |p∞ − плато|), × — плато двох найбільших L (SE·Бірге ⊕ |плато − p∞|).\n"
                 "p*(L) — вузьковіконна кубіка (вікна 0.008–0.015, 2D 0.006–0.012). Λ₄ — очікувана кількість 3- і 4-циклів "
                 "через вузол при p*_Бете. Лінійні шкали.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, dpi=150)


if __name__ == "__main__":
    main()
