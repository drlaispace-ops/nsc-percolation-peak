"""П1.3b. Пік NSC на 3D-решітках з різними короткими петлями (алмазна z = 4, ОЦК z = 8, ГЦК z = 12), bond,
періодичні межі: прогін з контрольними точками (за зразком p1_3_run.py).

Мета: перевірити «локальність» піку — чи p∞ решітки збігається з піком гратки Бете з тим самим z (П1.3: 4D–6D так,
3D sc — +2.4%). Решітки — code/lib/lattices3d.py; оцінювач — sampled.nsc_sampled (M = 200, s0 = 1000).
Сітка p нерівномірна, 31 точка (як у П1.3): 19 густих у [c − h, c + h] навколо піку пілоту і 12 рідких на [lo, hi].
Пілот (scratch, 10 реалізацій, 25 точок): diamond L = 8 — пік ≈ 0.408; bcc L = 12 — ≈ 0.197; fcc L = 8 — ≈ 0.13–0.136.
Вікно кубіки HW = 0.02 (як у П1.3 для d ≥ 3); систематику вікна звітує p1_3b_analyse.py.

Порядок: задачі (решітка, L) від дешевших. Для кожної: R_CAL = 50 реалізацій → R = clip(R_CAL·max((SE_p*/0.002)²,
(SE_P*/0.01)²), R_MIN, R_MAX) → решта. Перед продовженням — прогноз загального часу; якщо > BUDGET_H, зупинка (⛳ бюджет).
Шматок = b реалізацій (~60 с), файл data/P1/p1_3b/<решітка>/L<L>/chunk_*.npz; готові шматки пропускаються.
Сід реалізації j шматка i: SeedSequence([SEED_ROOT, номер решітки, L], spawn_key=(i,)).spawn(b)[j].
Для n ≤ 5000 (найменші L) перші 2 реалізації — ще й точна NSC (перевірка оцінювача).
Зберігається на реалізацію і p: скалярні поля nsc_sampled, χ, профіль δ_k (K = 64, float32), журнал скінченних кластерів.
Підсумок задачі — results/P1/p1_3b/<решітка>_L<L>.json (runmeta).
Запуск (кожен виклик працює не довше за --max-min хвилин і виходить; повторювати до «усі задачі виконано»):
    NUMBA_NUM_THREADS=6 OMP_NUM_THREADS=1 .venv/bin/python code/P1/p1_3b_run.py --max-min 8 >> results/P1/p1_3b_run.log
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
sys.path.insert(0, str(ROOT / "code" / "P1"))
import lattices3d as lt  # noqa: E402
import sampled as sm  # noqa: E402
from p1_3_run import analyse, grid, load, save_atomic  # noqa: E402  (імпорт без запуску)
from runmeta import runmeta, save  # noqa: E402

SEED_ROOT = 202611123
NAMES = ["diamond", "bcc", "fcc"]                     # порядок = номер у сіді
PC = {"diamond": 0.3895892, "bcc": 0.18028762, "fcc": 0.12016377}   # Xu2014, табл. V (data/P1/constants.csv)
SIZES = {"diamond": [8, 12, 16, 24, 32, 40], "bcc": [12, 16, 24, 32, 40, 48], "fcc": [8, 12, 16, 24, 32, 40]}
# найбільші L (bcc 48, fcc 40) додано після перших задач: прогноз бюджету 1.2 год дозволив (ціль ≤ 3 год);
# diamond L = 40 (n = 512 000): спершу відкладено (грубий прогноз 3.3 год), додано після решти задач, коли фактичні
# витрати 1.44 год + оцінка 0.6 год вкладалися в 3 год
GRID = {"diamond": (0.37, 0.52, 0.41, 0.03), "bcc": (0.165, 0.26, 0.197, 0.025), "fcc": (0.105, 0.20, 0.133, 0.025)}
HW, TARGET, TARGET_P = 0.02, 0.002, 0.01
M, S0, K, R_CAL, R_MIN, R_MAX, B_BOOT, CHUNK_SEC, BUDGET_H = 200, 1000, 64, 50, 100, 600, 500, 60.0, 3.0
R_PROJ = 130                   # R для прогнозу ще не каліброваних задач (на перших 14 задачах R = 100–126, одна 200)
SEC_PER_UNIT = 1.0e-5          # грубий прогноз с/реалізацію на (n + E) для 31 точки; уточнюється калібруванням
SCAL = ("nsc", "se", "P", "g_giant", "se_giant", "g_fin", "se_fin", "chi")
DATA = ROOT / "data" / "P1" / "p1_3b"
RES = ROOT / "results" / "P1" / "p1_3b"


def realization(name, L, ps, rng, exact):
    n, u, v, w, _ = lt.coupled(name, L, rng)
    perm = rng.permutation(n)
    NP = len(ps)
    o = {k: np.empty(NP) for k in SCAL}
    o.update(n_big=np.empty(NP, np.int64), delta=np.empty((NP, K), np.float32), nsc_exact=np.full(NP, np.nan),
             **{k: np.empty((NP, sm.NBIN), np.int64 if k in ("sbin_c", "sbin_n") else float)
                for k in ("sbin_c", "sbin_n", "sbin_sc", "sbin_var")})
    for i, p in enumerate(ps):
        A, _ = sm.at_p(n, u, v, w, p)
        r = sm.nsc_sampled(A, M=M, s0=S0, perm=perm, K=K)
        for k in SCAL + ("n_big", "delta", "sbin_c", "sbin_n", "sbin_sc", "sbin_var"):
            o[k][i] = r[k]
        if exact:
            o["nsc_exact"][i] = sm.node_sc_exact(A, K=K)[0].sum() / n
    return o


def run_chunk(name, L, ps, i, b):
    seq = np.random.SeedSequence([SEED_ROOT, NAMES.index(name), L], spawn_key=(i,))
    rows, sec = [], []
    for j, child in enumerate(seq.spawn(b)):
        t = time.perf_counter()
        rows.append(realization(name, L, ps, np.random.default_rng(child), exact=lt.EDGES[name](L)[0] <= 5000 and i * b + j < 2))
        sec.append(time.perf_counter() - t)
    out = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    out.update(sec=np.array(sec), ridx=np.arange(i * b, (i + 1) * b))
    return out


def cost_units(name, L):
    n, u, _ = lt.EDGES[name](L)
    return n + len(u)


def elapsed_h():
    return sum(float(np.load(p)["sec"].sum()) for p in DATA.glob("*/L*/chunk_*.npz")) / 3600


def an(ps, S, seed):
    return analyse(ps, S, 3, "bond", np.random.default_rng(seed))["all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-L", type=int, default=None)
    ap.add_argument("--max-min", type=float, default=1e9, help="вийти після стількох хвилин (між шматками)")
    a = ap.parse_args()
    t_start = time.time()
    deadline = t_start + 60 * a.max_min
    jobs = [dict(name=nm, L=L, units=cost_units(nm, L)) for nm in NAMES if not a.only or nm in a.only
            for L in SIZES[nm] if not a.max_L or L <= a.max_L]
    jobs.sort(key=lambda j: j["units"])
    meta0 = runmeta(__file__)
    for jn, job in enumerate(jobs):
        name, L = job["name"], job["L"]
        ps = grid(*GRID[name])
        sec_pred = SEC_PER_UNIT * job["units"] + 0.05
        b = int(np.clip(round(CHUNK_SEC / sec_pred), 1, 50))
        outdir = DATA / name / f"L{L}"
        outdir.mkdir(parents=True, exist_ok=True)
        cal_path = outdir / "calib.json"
        summary = RES / f"{name}_L{L}.json"
        if summary.exists():
            continue

        def fill(nch, tag):
            for i in range(nch):
                path = outdir / f"chunk_{i:05d}.npz"
                if path.exists():
                    continue
                if time.time() > deadline:
                    print(f"[{time.strftime('%d.%m %H:%M')}] ліміт часу виклику — вихід (продовжити тим самим запуском)",
                          flush=True)
                    sys.exit(0)
                t = time.time()
                save_atomic(path, run_chunk(name, L, ps, i, b))
                print(f"[{time.strftime('%d.%m %H:%M')}] {name} L={L} {tag} шматок {i + 1}/{nch} (b={b}) "
                      f"{time.time() - t:.0f} с", flush=True)

        n_cal = math.ceil(R_CAL / b)
        fill(n_cal, "калібрування")
        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            assert cal["b"] == b and np.allclose(cal["ps"], ps)
        else:
            S = {k: v[:n_cal * b] for k, v in load(outdir).items()}
            A0 = an(ps, S, [SEED_ROOT, 1, NAMES.index(name), L])
            Rn = n_cal * b
            R_need = int(np.clip(math.ceil(Rn * max((A0["se_p"] / TARGET) ** 2, (A0["se_P"] / TARGET_P) ** 2)),
                                 R_MIN, R_MAX))
            nch = math.ceil(R_need / b)
            plain = ~np.isfinite(S["nsc_exact"]).any(1)
            sec = float(S["sec"][plain].mean())
            cal = dict(b=b, ps=ps.tolist(), R_cal=Rn, analysis=A0, R_need=R_need, n_chunks=nch, R_final=nch * b,
                       sec_per_realization=sec, sec_pred=sec_pred)
            cal_path.write_text(json.dumps(cal, ensure_ascii=False, indent=1))
            print(f"   калібрування {name} L={L}: p*={A0['p_star']:.4f}±{A0['se_p']:.4f}, P*={A0['P_star']:.3f}"
                  f"±{A0['se_P']:.3f} → R={nch * b}; {sec:.2f} с/реал. (прогноз {sec_pred:.2f})", flush=True)
        rest = (cal["n_chunks"] - len(list(outdir.glob("chunk_*.npz")))) * b * cal["sec_per_realization"] / 3600
        later = 0.0
        ratio = cal["sec_per_realization"] / cal["sec_pred"]           # поправка грубої моделі вартості
        for j2 in jobs[jn + 1:]:
            c2 = DATA / j2["name"] / f"L{j2['L']}" / "calib.json"
            if c2.exists():
                cc = json.loads(c2.read_text())
                later += (cc["n_chunks"] * cc["b"] - cc["R_cal"]) * cc["sec_per_realization"] / 3600
            elif not (RES / f"{j2['name']}_L{j2['L']}.json").exists():
                later += R_PROJ * ratio * (SEC_PER_UNIT * j2["units"] + 0.05) / 3600
        proj = elapsed_h() + rest + later
        print(f"   прогноз загального часу: {proj:.2f} год", flush=True)
        if proj > BUDGET_H:
            print(f"⛳ БЮДЖЕТ: прогноз {proj:.1f} год > {BUDGET_H:.1f} год — зупинка перед {name} L={L}", flush=True)
            sys.exit(3)
        fill(cal["n_chunks"], "основний")
        S = load(outdir)
        A1 = an(ps, S, [SEED_ROOT, 2, NAMES.index(name), L])
        ex = np.isfinite(S["nsc_exact"])
        check = None
        if ex.any():
            z = (S["nsc"][ex] - S["nsc_exact"][ex]) / np.where(S["se"][ex] > 0, S["se"][ex], np.nan)
            check = dict(points=int(ex.sum()), max_abs_err=float(np.nanmax(np.abs(S["nsc"][ex] - S["nsc_exact"][ex]))),
                         z_mean=float(np.nanmean(z)), z_sd=float(np.nanstd(z, ddof=1)), z_n=int(np.isfinite(z).sum()))
        meta = runmeta(__file__, params=dict(name=name, L=L, z=lt.Z[name], pc=PC[name], M=M, s0=S0, K=K, grid=GRID[name],
                                             halfwidth=HW, target_se_p=TARGET, target_se_P=TARGET_P, chunk=b,
                                             data_dir=os.path.relpath(outdir, ROOT)),
                       seeds=dict(root=SEED_ROOT, scheme="SeedSequence([root, lattice_index, L], spawn_key=(chunk,))"
                                                         ".spawn(b)[j]", lattice_index=NAMES.index(name)))
        meta.update({k: meta0[k] for k in ("git_commit", "git_dirty", "script_sha256")}, started=meta0["time"])
        R = len(S["nsc"])
        save(summary, meta, R=R, ps=ps.tolist(), calibration=cal, analysis=A1, exact_check=check,
             hours=float(S["sec"].sum() / 3600), mean={k: S[k].mean(0).tolist() for k in SCAL},
             sd={k: S[k].std(0, ddof=1).tolist() for k in SCAL},
             met_target=bool(A1["se_p"] <= TARGET and A1["se_P"] <= TARGET_P))
        print(f"ГОТОВО {name} L={L}: R={R}, p*={A1['p_star']:.4f}±{A1['se_p']:.4f}, P*={A1['P_star']:.3f}"
              f"±{A1['se_P']:.3f}, {S['sec'].sum() / 3600:.2f} год" +
              (f"; точна перевірка z̄={check['z_mean']:+.2f} SD={check['z_sd']:.2f} (n={check['z_n']})" if check else ""),
              flush=True)
    print("усі задачі виконано", flush=True)


if __name__ == "__main__":
    main()
