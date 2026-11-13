"""П1.2. Прогноз бюджету П1.3 і конфігурація (M, R, сітки p) за пілотом і моделлю вартості.

Шум положення піку в пілоті (R_PILOT реалізацій, M = 500) розкладено на міжграфову і вибіркову частини
за часткою σ_s²/σ² у піку, обмеженою [0.05, 0.95] (сама частка оцінена з шумом; у 5D при L = 10 вийшла > 1)
(наближення: та сама частка в нахилі, що й у значенні):
    SE_p²(L, M, R) = (R_PILOT/R)·[G(L) + S(L)·500/M],
На пілотних L беруться виміряні SE; для більших L G(L) і S(L) екстраполюються степеневими законами. SE частки гіганта в піку
екстраполюється одним степеневим законом і від M не залежить (мітки кластерів точні).
R = max(R для SE_p ≤ ціль, R для SE_P ≤ 0.01, 50). M вибирається з сітки так, щоб мінімізувати години.
Вартість реалізації — сума за сіткою p моделі p1_2_cost_model (робота на вузол і частка гіганта
з найбільшого пілотного L; для d ≥ 4 — параметри 3D, у d ≥ 4 модель не перевірялась).
Ціль SE p*: 0.001 у 2D, 0.002 у 3D (план); для 4D–6D план цілі не задає — беремо 0.002 (припущення).
Запуск: .venv/bin/python code/P1/p1_2_budget.py  →  results/P1/p1_2_budget.json, code/P1/p1_3_config.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
sys.path.insert(0, str(ROOT / "code" / "P1"))
from runmeta import runmeta, save  # noqa: E402

RES = ROOT / "results" / "P1"
pilot = json.load(open(RES / "p1_2_pilot.json"))
cm = json.load(open(RES / "p1_2_cost_model.json"))["model"]
M_PILOT = pilot["meta"]["params"]["M"]
MS = [200, 500, 1000, 2000, 5000]
R_MIN, TARGET_P = 50, 0.01
TARGET = {2: 0.001, 3: 0.002, 4: 0.002, 5: 0.002, 6: 0.002}
SIZES = {"2D bond sq": [64, 128, 256, 512, 1024], "2D site sq": [64, 128, 256, 512], "2D bond tri": [64, 128, 256, 512],
         "3D bond": [16, 24, 32, 48, 64, 96, 128, 192, 256], "4D bond": [8, 12, 16, 24, 32], "5D bond": [6, 8, 10, 12, 16],
         "6D bond": [6, 8, 10]}
BUDGET_H = 300
F_LO, F_HI = 0.05, 0.95


def per_work_ns(th, N):
    N = np.asarray(N, float)
    return th[2] + th[3] * np.maximum(0, np.log(N) - th[4]) + th[5] * np.maximum(0, np.log(N) - th[6])


def powerlaw(Ls, y):
    """y ≈ A·L^κ через log-МНК; нульові значення замінюються малим числом."""
    Ls, y = np.asarray(Ls, float), np.maximum(np.asarray(y, float), 1e-30)
    k, lnA = np.polyfit(np.log(Ls), np.log(y), 1)
    return float(np.exp(lnA)), float(k)


def cost_per_realization(lat, L, M):
    d = lat["cfg"]["d"]
    m = cm[str(min(d, 3))]
    th = m["theta"]
    Lmax = max(lat["L"], key=int)
    ref = lat["L"][Lmax]
    n = L ** d
    t = 0.0
    for wex, ws, P in zip(ref["work_exact_per_node"], ref["work_sampled_per_node"], ref["mean_P"]):
        Ng = max(P * n, 2.0)
        Ms = min(M, Ng)
        t += m["c_gen"] * n + th[0] * n + 1e-9 * (th[1] * wex * n + per_work_ns(th, Ng) * ws * n * Ms / M_PILOT)
    return t


out, config = {}, {}
for name, lat in pilot["lattices"].items():
    d = lat["cfg"]["d"]
    Ls = sorted(int(L) for L in lat["L"])
    G, S, SP, fs = [], [], [], []
    for L in Ls:
        r = lat["L"][str(L)]
        R0 = r["R"]
        ans = list(r["analysis"].values())               # для site — обидві нормалізації, беремо гіршу
        se_p = max(a["se_p"] for a in ans)
        se_P = max(a["se_P"] for a in ans)
        a = r["analysis"]["all"]
        f = float(np.clip(a["var_sampling"] / a["var_total"], F_LO, F_HI))   # частка — сама оцінка з шумом
        fs.append(f)
        G.append(se_p ** 2 * (1 - f))
        S.append(se_p ** 2 * f)
        SP.append(se_P ** 2)
    fitG, fitS, fitP = powerlaw(Ls, G), powerlaw(Ls, S), powerlaw(Ls, SP)
    rows = []
    for L in SIZES[name]:
        if L in Ls:                                       # пілотний L — виміряні значення, без підгонки
            i = Ls.index(L)
            g, s, sp = G[i], S[i], SP[i]
        else:
            g, s, sp = (A * L ** k for A, k in (fitG, fitS, fitP))
        best = None
        for M in MS:
            Rp = R0 * (g + s * M_PILOT / M) / TARGET[d] ** 2
            RP = R0 * sp / TARGET_P ** 2
            R = int(np.ceil(max(Rp, RP, R_MIN)))
            c = cost_per_realization(lat, L, M)
            h = R * c / 3600
            if best is None or h < best["hours"]:
                best = dict(L=L, n=L ** d, M=M, R=R, R_for_p=float(Rp), R_for_P=float(RP), sec_per_realization=c, hours=h,
                            extrapolated=L > Ls[-1], mem_gb=cm[str(min(d, 3))]["rss_gb_per_n"] * L ** d + cm[str(min(d, 3))]["rss_gb_0"])
        rows.append(best)
    # перевірка моделі вартості на самому пілоті (M = 500, ті самі L)
    chk = {L: dict(model=cost_per_realization(lat, L, M_PILOT), pilot=lat["L"][str(L)]["sec_per_realization"]) for L in Ls}
    out[name] = dict(d=d, target_se_p=TARGET[d], pilot_Ls=Ls, frac_sampling=fs, fit_graph=fitG, fit_sampling=fitS, fit_P=fitP,
                     plan=rows, hours=float(sum(r["hours"] for r in rows)), cost_check_pilot=chk)
    config[name] = dict(d=d, kind=lat["cfg"]["kind"], pc=lat["cfg"]["pc"], p_grid=lat["ps"], s0=pilot["meta"]["params"]["s0"],
                        sizes={str(r["L"]): dict(M=r["M"], R=r["R"]) for r in rows})

total = sum(v["hours"] for v in out.values())
for name, v in out.items():
    print(f"\n{name}: {v['hours']:.1f} год; частка вибіркового шуму в пілоті {', '.join(f'{x:.2f}' for x in v['frac_sampling'])}; "
          f"G∝L^{v['fit_graph'][1]:.2f}, S∝L^{v['fit_sampling'][1]:.2f}, SE_P²∝L^{v['fit_P'][1]:.2f}")
    for r in v["plan"]:
        print(f"   L={r['L']:5d} n={r['n']:.1e} M={r['M']:5d} R={r['R']:6d} (p: {r['R_for_p']:8.0f}, P: {r['R_for_P']:8.0f})  "
              f"{r['sec_per_realization']:8.1f} с/реал.  {r['hours']:7.1f} год  {r['mem_gb']:.1f} ГБ{'  екстрап.' if r['extrapolated'] else ''}")
    print("   модель проти пілоту, с/реал.:", {L: f"{c['model']:.2f}/{c['pilot']:.2f}" for L, c in v["cost_check_pilot"].items()})
print(f"\nРАЗОМ: {total:.0f} год (поріг ⛳ {BUDGET_H} год)")
meta = runmeta(__file__, params=dict(MS=MS, R_min=R_MIN, target_P=TARGET_P, target_p=TARGET, sizes=SIZES, budget_h=BUDGET_H,
                                     inputs=["p1_2_pilot.json", "p1_2_cost_model.json"]))
save(RES / "p1_2_budget.json", meta, lattices=out, total_hours=total, over_budget=bool(total > BUDGET_H))
(ROOT / "code" / "P1" / "p1_3_config.json").write_text(json.dumps(dict(
    note="Конфігурація П1.3 з прогнозу П1.2 (p1_2_budget.json). R для L понад пілотні — екстраполяція; "
         "уточнювати після перших 50 реалізацій кожного L.", lattices=config), ensure_ascii=False, indent=1), encoding="utf-8")
