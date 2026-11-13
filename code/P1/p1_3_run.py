"""П1.3. Скінченно-розмірний скейлінг NSC, d = 2…6: прогін з контрольними точками.

Розміри, M і початкові R — з code/P1/p1_3_config.json (прогноз П1.2). Сітка p нерівномірна, 31 точка, як у пілоті
(ціна та сама): 19 густих точок у [c − h, c + h] навколо піку пілоту і 12 рідких на решті діапазону пілоту.
Причина: на пілоті p* залежить від ширини вікна кубіки (3D, L = 32: 0.2595 при ±0.013 і 0.2627 при стандартному
вікні ±0.023), тобто широке вікно дає зсув, більший за ціль SE 0.002; густа сітка дає вузьке вікно з достатньою
кількістю точок. Вікно кубіки — HW (у одиницях p); систематику вікна звітує аналіз (p1_3_analyse.py).

Порядок: задачі (решітка, L) від дешевших до дорожчих (прогноз П1.2). Для кожної: перші 50 реалізацій →
перерахунок R = max(50, 50·max((SE_p*/ціль)², (SE_P*/0.01)²)) за бутстрепом (план: «R уточнюється після перших
50 реалізацій») → решта. Перед продовженням — прогноз загального часу; якщо > 300 год, прогін зупиняється (⛳ бюджет).
Шматок = b реалізацій (~3 хв), файл data/P1/p1_3/<решітка>/L<L>/chunk_*.npz; готові шматки пропускаються.
Сід реалізації j шматка i: SeedSequence([ROOT, номер решітки, L], spawn_key=(i,)).spawn(b)[j] — не залежить від R
і переривань. 3D, L ≤ 48: для перших 3 реалізацій ще й точна NSC (перевірка, план).
Зберігається на реалізацію і p: усі скалярні поля nsc_sampled, χ, профіль δ_k (K = 64, float32), журнал скінченних
кластерів за ⌊log₂ s⌋, NodeSC джерел гіганта (float32, NaN-доповнення до M) — сирі дані для П1.13–П1.15.
Підсумок задачі — results/P1/p1_3/<решітка>_L<L>.json (runmeta: коміт має бути чистим).
Запуск (довгий, під caffeinate):
    caffeinate -i .venv/bin/python code/P1/p1_3_run.py [--only "3D bond"] [--max-L 64] >> results/P1/p1_3_run.log
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import sampled as sm  # noqa: E402
import stats  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

SEED_ROOT = 20261112
CFG = json.load(open(ROOT / "code" / "P1" / "p1_3_config.json"))["lattices"]
BUD = json.load(open(ROOT / "results" / "P1" / "p1_2_budget.json"))["lattices"]
NAMES = ["2D bond sq", "2D site sq", "2D bond tri", "3D bond", "4D bond", "5D bond", "6D bond"]   # порядок = номер у сіді
SLUG = {n: n.replace(" ", "_") for n in NAMES}
# (lo, hi) — діапазон пілоту; c — пік пілоту на найбільшому L (вікно ±0.013; для site — між піками двох
# нормалізацій 0.590 і 0.584); h — півширина густої частини (покриває дрейф піку з L і pc, крім 4D–6D, де pc далі).
GRID = {"2D bond sq": (0.44, 0.60, 0.494, 0.015), "2D site sq": (0.55, 0.72, 0.588, 0.015),
        "2D bond tri": (0.30, 0.44, 0.344, 0.015), "3D bond": (0.22, 0.32, 0.258, 0.025),
        "4D bond": (0.14, 0.24, 0.189, 0.02), "5D bond": (0.10, 0.20, 0.149, 0.02), "6D bond": (0.08, 0.18, 0.122, 0.02)}
ND, NC = 19, 12
# вікно кубіки (у p): пробний прогін 3D L = 16, R = 400 — p* при ±0.008/0.013/0.02 = 0.2659/0.2657/0.2662
# (поліном 4-го степеня — 0.2662/0.2658/0.2661), при ±0.03 — 0.2670, ±0.045 — 0.2701: зсув з'являється за ±0.02
HW = {2: 0.012, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02}
TARGET = {2: 0.001, 3: 0.002, 4: 0.002, 5: 0.002, 6: 0.002}      # SE p* (4D–6D — припущення П1.2)
TARGET_P, R_CAL, BUDGET_H, K, B_BOOT, CHUNK_SEC = 0.01, 50, 300.0, 64, 1000, 180.0
SCAL = ("nsc", "se", "nsc_occ", "se_occ", "P", "P_occ", "g_giant", "se_giant", "g_fin", "se_fin", "chi", "chi_occ")
OUT = os.environ.get("P1_3_OUT")                                   # лише для пробного прогону: інша тека виводу
DATA = Path(OUT) / "data" if OUT else ROOT / "data" / "P1" / "p1_3"
RES = Path(OUT) / "res" if OUT else ROOT / "results" / "P1" / "p1_3"


def grid(lo, hi, c, h):
    dense = np.linspace(c - h, c + h, ND)
    a, b = (c - h) - lo, hi - (c + h)
    nl = int(round(NC * a / (a + b)))
    left = np.linspace(lo, c - h, nl + 1)[:-1]
    right = np.linspace(c + h, hi, NC - nl + 1)[1:]
    return np.r_[left, dense, right]


def realization(name, L, M, ps, rng, exact):
    cfg = CFG[name]
    if cfg["kind"] == "tri":
        n, u, v, w, ws = sm.coupled_triangular(L, rng)
    else:
        n, u, v, w, ws = sm.coupled_bonds(L, cfg["d"], rng, kind=cfg["kind"])
    perm = rng.permutation(n)
    NP = len(ps)
    o = {k: np.empty(NP) for k in SCAL}
    o.update(n_big=np.empty(NP, np.int64), giant_sampled=np.empty(NP, bool), delta=np.empty((NP, K), np.float32),
             raw_giant=np.full((NP, M), np.nan, np.float32), nsc_exact=np.full(NP, np.nan),
             **{k: np.empty((NP, sm.NBIN), np.int64 if k in ("sbin_c", "sbin_n") else float)
                for k in ("sbin_c", "sbin_n", "sbin_sc", "sbin_var")})
    for i, p in enumerate(ps):
        A, occ = sm.at_p(n, u, v, w, p, ws)
        r = sm.nsc_sampled(A, M=M, s0=cfg["s0"], perm=perm, occ=occ, K=K, keep_raw=True)
        for k in SCAL + ("n_big", "giant_sampled", "delta", "sbin_c", "sbin_n", "sbin_sc", "sbin_var"):
            o[k][i] = r[k]
        if r["giant_sampled"]:
            o["raw_giant"][i, :len(r["raw_giant_sc"])] = r["raw_giant_sc"]
        if exact:
            o["nsc_exact"][i] = sm.node_sc_exact(A, K=K)[0].sum() / n
    return o


def run_chunk(name, L, M, ps, i, b):
    seq = np.random.SeedSequence([SEED_ROOT, NAMES.index(name), L], spawn_key=(i,))
    rows, sec = [], []
    for j, child in enumerate(seq.spawn(b)):
        t = time.perf_counter()
        g = i * b + j
        rows.append(realization(name, L, M, ps, np.random.default_rng(child),
                                exact=CFG[name]["d"] == 3 and L <= 48 and g < 3))
        sec.append(time.perf_counter() - t)
    out = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    out.update(sec=np.array(sec), ridx=np.arange(i * b, (i + 1) * b))
    return out


def save_atomic(path, data):
    tmp = path.with_name("tmp_" + path.name)                    # не збігається з шаблоном chunk_*.npz
    np.savez(tmp, **data)
    os.replace(tmp, path)


def load(outdir, keys=None):
    ps = sorted(outdir.glob("chunk_*.npz"))
    if not ps:
        return {}
    ch = [np.load(p) for p in ps]
    keys = keys or ch[0].files
    return {k: np.concatenate([c[k] for c in ch]) for k in keys}


def analyse(ps, S, d, kind, rng):
    """p* і SE (бутстреп, вікно HW), P(p*) і SE; для site — обидві нормалізації."""
    out = {}
    for norm, y, P in (("all", "nsc", "P"), ("occ", "nsc_occ", "P_occ")):
        if norm == "occ" and kind != "site":
            continue
        Y, Pc = S[y], S[P]
        pb = stats.peak_bootstrap(ps, Y, B=B_BOOT, rng=rng, halfwidth=HW[d])
        R = len(Y)
        Pstar = []
        for _ in range(B_BOOT):
            idx = rng.integers(0, R, R)
            x, _ = stats.cubic_peak(ps, Y[idx].mean(0), halfwidth=HW[d])
            if np.isfinite(x):
                Pstar.append(np.interp(x, ps, Pc[idx].mean(0)))
        out[norm] = dict(p_star=pb["x"], se_p=pb["se_x"], nsc_star=pb["y"], fail=pb["fail"],
                         P_star=float(np.interp(pb["x"], ps, Pc.mean(0))), se_P=float(np.std(Pstar, ddof=1)))
    return out


def elapsed_h():
    return sum(float(np.load(p)["sec"].sum()) for p in DATA.glob("*/L*/chunk_*.npz")) / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-L", type=int, default=None)
    a = ap.parse_args()
    jobs = []
    for name in NAMES:
        if a.only and name not in a.only:
            continue
        for row in BUD[name]["plan"]:
            L = row["L"]
            if a.max_L and L > a.max_L:
                continue
            jobs.append(dict(name=name, L=L, M=CFG[name]["sizes"][str(L)]["M"], R0=CFG[name]["sizes"][str(L)]["R"],
                             sec_pred=row["sec_per_realization"]))
    jobs.sort(key=lambda j: j["R0"] * j["sec_pred"])
    meta0 = runmeta(__file__)
    if meta0["git_dirty"]:
        print("УВАГА: робоча копія має незакомічені зміни — результати будуть позначені git_dirty", flush=True)
    for jn, job in enumerate(jobs):
        name, L, M = job["name"], job["L"], job["M"]
        d, kind = CFG[name]["d"], CFG[name]["kind"]
        ps = grid(*GRID[name])
        b = int(np.clip(round(CHUNK_SEC / job["sec_pred"]), 1, 50))
        outdir = DATA / SLUG[name] / f"L{L}"
        outdir.mkdir(parents=True, exist_ok=True)
        cal_path = outdir / "calib.json"
        summary = RES / f"{SLUG[name]}_L{L}.json"
        if summary.exists():
            continue

        def fill(nch, tag):
            for i in range(nch):
                path = outdir / f"chunk_{i:05d}.npz"
                if path.exists():
                    continue
                t = time.time()
                save_atomic(path, run_chunk(name, L, M, ps, i, b))
                print(f"[{time.strftime('%d.%m %H:%M')}] {name} L={L} {tag} шматок {i + 1}/{nch} (b={b}) "
                      f"{time.time() - t:.0f} с", flush=True)

        n_cal = math.ceil(R_CAL / b)
        fill(n_cal, "калібрування")
        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            assert cal["b"] == b and np.allclose(cal["ps"], ps)
        else:
            S = load(outdir)
            S = {k: v[:n_cal * b] for k, v in S.items()}
            an = analyse(ps, S, d, kind, np.random.default_rng([SEED_ROOT, 1, NAMES.index(name), L]))
            se_p = max(x["se_p"] for x in an.values())
            se_P = max(x["se_P"] for x in an.values())
            Rn = n_cal * b
            R_need = max(R_CAL, math.ceil(Rn * max((se_p / TARGET[d]) ** 2, (se_P / TARGET_P) ** 2)))
            nch = math.ceil(R_need / b)
            plain = ~np.isfinite(S["nsc_exact"]).any(1)        # без реалізацій з точною перевіркою (вони довші)
            sec = float(S["sec"][plain].mean())
            cal = dict(b=b, ps=ps.tolist(), R_cal=Rn, analysis=an, R_need=R_need, n_chunks=nch, R_final=nch * b,
                       R_config=job["R0"], sec_per_realization=sec, sec_pred=job["sec_pred"])
            cal_path.write_text(json.dumps(cal, ensure_ascii=False, indent=1))
            print(f"   калібрування {name} L={L}: p*={an['all']['p_star']:.4f}±{se_p:.4f}, P*={an['all']['P_star']:.3f}"
                  f"±{se_P:.3f} → R={nch * b} (прогноз П1.2 {job['R0']}); {sec:.1f} с/реал. (прогноз {job['sec_pred']:.1f})",
                  flush=True)
        # бюджет: витрачено + решта цієї задачі + прогноз П1.2 для ще не каліброваних
        rest = (cal["n_chunks"] - len(list(outdir.glob("chunk_*.npz")))) * b * cal["sec_per_realization"] / 3600
        later = 0.0
        for j2 in jobs[jn + 1:]:
            c2 = DATA / SLUG[j2["name"]] / f"L{j2['L']}" / "calib.json"
            if c2.exists():
                cc = json.loads(c2.read_text())
                later += (cc.get("n_chunks2", cc["n_chunks"]) * cc["b"] - cc["R_cal"]) * cc["sec_per_realization"] / 3600
            elif not (RES / f"{SLUG[j2['name']]}_L{j2['L']}.json").exists():
                later += j2["R0"] * j2["sec_pred"] / 3600
        proj = elapsed_h() + rest + later
        print(f"   прогноз загального часу: {proj:.1f} год", flush=True)
        if proj > BUDGET_H:
            print(f"⛳ БЮДЖЕТ: прогноз {proj:.0f} год > {BUDGET_H:.0f} год — зупинка перед {name} L={L}", flush=True)
            sys.exit(3)
        fill(cal["n_chunks"], "основний")
        S = load(outdir)
        an = analyse(ps, S, d, kind, np.random.default_rng([SEED_ROOT, 2, NAMES.index(name), L]))
        # друге (останнє) уточнення: калібрування за 50 реалізаціями шумне; якщо ціль не досягнуто — дорахувати,
        # але не більше ніж удвічі (без повторних циклів)
        ratio = max(max(x["se_p"] for x in an.values()) / TARGET[d], max(x["se_P"] for x in an.values()) / TARGET_P)
        if ratio > 1 and "n_chunks2" not in cal:
            R = len(S["nsc"])
            cal["n_chunks2"] = min(math.ceil(R * ratio ** 2 / b), 2 * cal["n_chunks"])
            cal_path.write_text(json.dumps(cal, ensure_ascii=False, indent=1))
            print(f"   ціль SE не досягнуто (×{ratio:.2f}) → R={cal['n_chunks2'] * b}", flush=True)
        if "n_chunks2" in cal:
            fill(cal["n_chunks2"], "дорахунок")
            S = load(outdir)
            an = analyse(ps, S, d, kind, np.random.default_rng([SEED_ROOT, 3, NAMES.index(name), L]))
        ex = np.isfinite(S["nsc_exact"])
        check = None
        if ex.any():
            z = (S["nsc"][ex] - S["nsc_exact"][ex]) / np.where(S["se"][ex] > 0, S["se"][ex], np.nan)
            check = dict(points=int(ex.sum()), max_abs_err=float(np.nanmax(np.abs(S["nsc"][ex] - S["nsc_exact"][ex]))),
                         z_mean=float(np.nanmean(z)), z_sd=float(np.nanstd(z, ddof=1)), z_n=int(np.isfinite(z).sum()))
        meta = runmeta(__file__, params=dict(name=name, L=L, d=d, kind=kind, pc=CFG[name]["pc"], M=M, s0=CFG[name]["s0"],
                                             K=K, grid=GRID[name], ND=ND, NC=NC, halfwidth=HW[d], target_se_p=TARGET[d],
                                             target_se_P=TARGET_P, chunk=b, data_dir=os.path.relpath(outdir, ROOT)),
                       seeds=dict(root=SEED_ROOT, scheme="SeedSequence([root, lattice_index, L], spawn_key=(chunk,))"
                                                         ".spawn(b)[j]", lattice_index=NAMES.index(name)))
        # стан коду — на момент старту прогону (коміти під час багатоденного прогону не мають змінювати метадані)
        meta.update({k: meta0[k] for k in ("git_commit", "git_dirty", "script_sha256")}, started=meta0["time"])
        R = len(S["nsc"])
        save(summary, meta, R=R, ps=ps.tolist(), calibration=cal, analysis=an, exact_check=check,
             hours=float(S["sec"].sum() / 3600),
             mean={k: S[k].mean(0).tolist() for k in SCAL}, sd={k: S[k].std(0, ddof=1).tolist() for k in SCAL},
             met_target=bool(max(x["se_p"] for x in an.values()) <= TARGET[d] and
                             max(x["se_P"] for x in an.values()) <= TARGET_P))
        A = an["all"]
        print(f"ГОТОВО {name} L={L}: R={R}, p*={A['p_star']:.4f}±{A['se_p']:.4f}, P*={A['P_star']:.3f}±{A['se_P']:.3f}, "
              f"{S['sec'].sum() / 3600:.2f} год" + (f"; точна перевірка z̄={check['z_mean']:+.2f} SD={check['z_sd']:.2f}"
                                                   f" (n={check['z_n']})" if check else ""), flush=True)
    print("усі задачі виконано", flush=True)


if __name__ == "__main__":
    main()
