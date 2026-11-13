"""П1.15. Пошук класів на решітках: аналіз (секунди–хвилини), без важких розрахунків.

Частина A — журнал скінченних кластерів П1.3 (data/P1/p1_3/<решітка>/L<L>/chunk_*.npz), нових розрахунків немає.
  g_b(p) = Σ_r sbin_sc / Σ_r sbin_n у біні b = ⌊log₂ s⌋ ≥ 6 (s ≥ 64); s̄_b(p) = Σ sbin_n / Σ sbin_c (середній s
  кластера в біні). Бутстреп за реалізаціями (вагами мультиноміального розподілу; спряжені p — разом).
  Тести в інтервалі |ε| ≤ 0.1, ε = (p − pc)/pc:
   (1) нахил лінійної підгонки g_b(p) і різниця крайніх точок, z за бутстрепом; χ²_diag сталості (довідково);
   (2) локальний нахил у pc (|p − pc| ≤ HW_LOC);
   (3) контроль зсуву s̄ у біні: поправка g_b(p) − [G(ln s̄_b(p)) − G(ln s̄_b(pc))], де G — журнал при pc того
       самого L, лінійна інтерполяція за ln s̄ (наближено: g — середнє за вузлами, s̄ — за кластерами);
       точний контроль за фіксованого s — частина B;
   (4) контроль L: g_b(p; L1) − g_b(p; L2) на спільних точках;
   (5) скейлінг: локальний нахил у pc ∝ s̄^y; порівняння y з σ = 1/(ν·d_f).
Частина B — покластерні дані code/P1/p1_15_run.py (data/P1/p1_15/): 2D bond-квадратна L = 256, 3D L = 48,
  p ∈ {0.9, 1, 1.1}·pc (найбільший кластер вилучено, як у журналі):
   (a) частка розкиду NodeSC за фіксованого s (після ln s), пояснена щільністю циклів c/s;
   (b) залежність від p за фіксованого s: зсуви за p після ln s і після ln s + c/s (+ (c/s)²), бутстреп за реалізаціями;
   (c) класи: dip (таблиця Хартиганів) і ΔBIC на залишку NodeSC, c/s і PC1–PC2 профілю δ_k при pc; нульова модель —
       унімодальний сурогат (закон Джонсона S_U, підігнаний до даних), пороги 95%; позитивний контроль —
       суміш кластерів 0.9 pc і 1.1 pc того самого біна;
   (d) інформація понад s: AUROC розрізнення p за однакового біна для наборів ознак (ln s; + c/s; + степені;
       + NodeSC; + профіль δ_k), логістична регресія, поділ за реалізаціями, бутстреп тестової вибірки;
  Доповнення після перевірки (notes/orchestra/B_verdict.md; окремий генератор [SEED, 2], числа вище не змінюються):
   GLS-χ² сталості з бутстреп-коваріацією (частина A); тотожність c/s = ½Σ k f_k − 1 + 1/s; потужність dip-тесту на
   синтетичних сумішах зі зсувом d·sd; мінливість AUROC за 10 випадковими поділами реалізацій; біноміальна ймовірність ΔBIC.
Запуск: OMP_NUM_THREADS=1 .venv/bin/python code/P1/p1_15_analyse.py → results/P1/p1_15_classes.json,
        results/P1/fig_p1_15_journal.png, fig_p1_15_cycles.png
"""
import csv
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sst  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "lib"), str(ROOT / "code" / "P1")]
import p1_3_run as run  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED = 20261116
B_BOOT = 1000
BINS = range(6, 15)
EPS_MAX = 0.1
MIN_C, MIN_R = 30, 10          # мінімум кластерів у (біні, p) за всіма реалізаціями і реалізацій з ними
GLS_STEP = 3                  # проріджування точок для GLS-χ²
HW_LOC = {2: 0.0055, 3: 0.0095}   # півширина вікна локального нахилу в pc (2D: 7 точок густої сітки, 3D: 7)
JOBS = {"2D bond sq": [256, 512, 1024], "2D bond tri": [256, 512], "2D site sq": [256, 512], "3D bond": [64, 128, 256]}
an = json.load(open(ROOT / "results" / "P1" / "p1_3_analysis.json"))["lattices"]
EXP = {}
for r in csv.DictReader(open(ROOT / "data" / "P1" / "constants.csv")):
    if r["system"].startswith("perc_") and r["quantity"] in ("nu", "d_f"):
        EXP.setdefault(int(r["d"]), {})[r["quantity"]] = float(r["value"])
SIGMA = {d: 1 / (v["nu"] * v["d_f"]) for d, v in EXP.items() if "nu" in v and "d_f" in v}


# ------------------------------------------------------------------ частина A
def load_journal(name, L):
    ps = np.array(json.load(open(run.RES / f"{run.SLUG[name]}_L{L}.json"))["ps"])
    S = run.load(run.DATA / run.SLUG[name] / f"L{L}", ["sbin_c", "sbin_n", "sbin_sc"])
    return ps, S


def with_pc(ps, S, pc):
    """Додає точку pc (лінійна інтерполяція сум між сусідніми вузлами сітки), якщо її немає на сітці."""
    if np.isclose(ps, pc).any():
        return ps, S, int(np.argmin(np.abs(ps - pc)))
    k = np.searchsorted(ps, pc) - 1
    w = (pc - ps[k]) / (ps[k + 1] - ps[k])
    out = {key: np.insert(v.astype(float), k + 1, (1 - w) * v[:, k] + w * v[:, k + 1], axis=1) for key, v in S.items()}
    return np.insert(ps, k + 1, pc), out, k + 1


def boot_sums(S, W):
    """Суми за реалізаціями з вагами W (B × R): масиви B × P × NBIN."""
    return {k: np.tensordot(W, v.astype(float), axes=(1, 0)) for k, v in S.items()}


def linfit(x, y, w=None):
    """Нахил лінійної підгонки y(x) уздовж останньої осі; y може мати провідну вісь бутстрепу."""
    w = np.ones_like(x) if w is None else w
    xm = (w * x).sum() / w.sum()
    return ((w * (x - xm)) * (y - (y * w).sum(-1, keepdims=True) / w.sum())).sum(-1) / (w * (x - xm) ** 2).sum()


def gls_const(y, yb, step=GLS_STEP):
    """χ² сталості g_b(p) з повною бутстреп-коваріацією (GLS) на проріджених точках (кожна step-та, щоб
    коваріація була оборотною й стабільною: точки за p корельовані через спряжені реалізації)."""
    k = np.arange(0, len(y), step)
    if len(k) < 2:
        return dict(gls_chi2=np.nan, gls_dof=0, gls_p=np.nan, gls_step=step)
    Yb = yb[:, k]
    Yb = Yb[np.isfinite(Yb).all(1)]
    Ci = np.linalg.inv(np.cov(Yb.T))
    one = np.ones(len(k))
    mu = one @ Ci @ y[k] / (one @ Ci @ one)
    r = y[k] - mu
    chi2 = float(r @ Ci @ r)
    return dict(gls_chi2=chi2, gls_dof=int(len(k) - 1), gls_p=float(sst.chi2.sf(chi2, len(k) - 1)),
                gls_logp10=float(sst.chi2.logsf(chi2, len(k) - 1) / np.log(10)), gls_step=step)


def journal_task(name, L, rng):
    pc, d = an[name]["pc"], an[name]["d"]
    ps0, S0 = load_journal(name, L)
    ps, S, kpc = with_pc(ps0, S0, pc)
    R = len(S["sbin_c"])
    W = rng.multinomial(R, np.full(R, 1 / R), size=B_BOOT).astype(float)
    tot = {k: v.sum(0) for k, v in S.items()}
    bt = boot_sums(S, W)
    g, sbar = tot["sbin_sc"] / tot["sbin_n"], tot["sbin_n"] / tot["sbin_c"]
    gb, sbb = bt["sbin_sc"] / bt["sbin_n"], bt["sbin_n"] / bt["sbin_c"]
    nr = (S["sbin_c"] > 0).sum(0)
    ok = (tot["sbin_c"] >= MIN_C) & (nr >= MIN_R)
    eps = (ps - pc) / pc
    # G(ln s) при pc: журнал усіх бінів із достатньою статистикою (для поправки на s̄)
    okpc = ok[kpc]

    def G(lns, gpc, lspc):
        return np.interp(lns, lspc, gpc)
    rows = []
    for b in BINS:
        sel = ok[:, b] & (np.abs(eps) <= EPS_MAX + 1e-9)
        if sel.sum() < 4 or not ok[kpc, b]:
            continue
        x = ps[sel]
        y, yb = g[sel, b], gb[:, sel, b]
        se = np.nanstd(yb, 0, ddof=1)
        # (1) нахил і крайні точки
        sl, slb = linfit(x, y), linfit(x, yb)
        dlt, dltb = y[0] - y[-1], yb[:, 0] - yb[:, -1]
        wm = np.sum(y / se ** 2) / np.sum(1 / se ** 2)
        chi2 = float(np.sum(((y - wm) / se) ** 2))
        gls = gls_const(y, yb)
        # (2) локальний нахил у pc
        loc = sel & (np.abs(ps - pc) <= HW_LOC[d] + 1e-9)
        if loc.sum() >= 3:
            lsl, lslb = linfit(ps[loc], g[loc, b]), linfit(ps[loc], gb[:, loc, b])
        else:
            lsl, lslb = np.nan, np.full(B_BOOT, np.nan)
        # (3) поправка на зсув s̄ у біні
        lspc_idx = np.flatnonzero(okpc)
        corr = G(np.log(sbar[sel, b]), g[kpc, lspc_idx], np.log(sbar[kpc, lspc_idx])) - \
            G(np.log(sbar[kpc, b]), g[kpc, lspc_idx], np.log(sbar[kpc, lspc_idx]))
        corrb = np.array([G(np.log(sbb[i, sel, b]), gb[i, kpc, lspc_idx], np.log(sbb[i, kpc, lspc_idx])) -
                          G(np.log(sbb[i, kpc, b]), gb[i, kpc, lspc_idx], np.log(sbb[i, kpc, lspc_idx]))
                          for i in range(B_BOOT)])
        yc, ycb = y - corr, yb - corrb
        slc, slcb = linfit(x, yc), linfit(x, ycb)
        if loc.sum() >= 3:
            lm = loc[sel]
            lslc, lslcb = linfit(x[lm], yc[lm]), linfit(x[lm], ycb[:, lm])
        else:
            lslc, lslcb = np.nan, np.full(B_BOOT, np.nan)
        sd = lambda v: float(np.nanstd(v, ddof=1))  # noqa: E731
        rows.append(dict(bin=b, s_range=[2 ** b, 2 ** (b + 1) - 1], p=x.tolist(), g=y.tolist(), se=se.tolist(),
                         sbar=sbar[sel, b].tolist(), n_clusters=tot["sbin_c"][sel, b].tolist(),
                         g_pc=float(g[kpc, b]), se_g_pc=sd(gb[:, kpc, b]), sbar_pc=float(sbar[kpc, b]),
                         slope=float(sl), se_slope=sd(slb), z_slope=float(sl / sd(slb)),
                         delta_ends=float(dlt), se_delta=sd(dltb), z_delta=float(dlt / sd(dltb)), p_ends=[float(x[0]), float(x[-1])],
                         chi2_diag=chi2, dof=int(sel.sum() - 1), **gls,
                         slope_loc=float(lsl), se_slope_loc=sd(lslb), z_slope_loc=float(lsl / sd(lslb)),
                         sbar_shift_rel=float(sbar[sel, b].max() / sbar[sel, b].min() - 1),
                         corr_max=float(np.max(np.abs(corr))), slope_corr=float(slc), se_slope_corr=sd(slcb),
                         z_slope_corr=float(slc / sd(slcb)), slope_loc_corr=float(lslc), se_slope_loc_corr=sd(lslcb),
                         z_slope_loc_corr=float(lslc / sd(lslcb))))
    return dict(L=L, R=R, pc=pc, d=d, rows=rows, _ps=ps, _g=g, _gb=gb, _sbar=sbar, _ok=ok, _kpc=kpc)


def compare_L(A, B_):
    """g_b(p; L1) − g_b(p; L2) на спільних точках сітки в |ε| ≤ 0.1: χ² (діагональ, незалежні вибірки) і max |z|."""
    out = []
    for ra in A["rows"]:
        rb = next((r for r in B_["rows"] if r["bin"] == ra["bin"]), None)
        if rb is None:
            continue
        pa, pb = np.round(ra["p"], 6), np.round(rb["p"], 6)
        com = np.intersect1d(pa, pb)
        if len(com) < 2:
            continue
        ia, ib = [np.flatnonzero(np.isin(x, com)) for x in (pa, pb)]
        z = (np.array(ra["g"])[ia] - np.array(rb["g"])[ib]) / np.hypot(np.array(ra["se"])[ia], np.array(rb["se"])[ib])
        out.append(dict(bin=ra["bin"], n=len(com), chi2=float(np.sum(z ** 2)), p_value=float(sst.chi2.sf(np.sum(z ** 2), len(com))),
                        max_abs_z=float(np.max(np.abs(z))), slope_L1=ra["slope"], se_L1=ra["se_slope"],
                        slope_L2=rb["slope"], se_L2=rb["se_slope"],
                        z_slope_diff=float((ra["slope"] - rb["slope"]) / np.hypot(ra["se_slope"], rb["se_slope"]))))
    return out


def scaling(T, bmax=99):
    """Локальний нахил у pc проти s̄: |∂g/∂p| ∝ s̄^y (зважена підгонка в log-log, біни з z > 3 і b ≤ bmax)."""
    r = [x for x in T["rows"] if np.isfinite(x["slope_loc"]) and abs(x["z_slope_loc"]) > 3 and x["bin"] <= bmax]
    if len(r) < 3:
        return None
    s = np.log([x["sbar_pc"] for x in r])
    y = np.log(np.abs([x["slope_loc"] for x in r]))
    sy = np.array([x["se_slope_loc"] / abs(x["slope_loc"]) for x in r])
    A = np.c_[np.ones_like(s), s] / sy[:, None]
    coef, *_ = np.linalg.lstsq(A, y / sy, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    res = (y - (coef[0] + coef[1] * s)) / sy
    return dict(bins=[x["bin"] for x in r], y=float(coef[1]), se_y=float(np.sqrt(cov[1, 1])),
                chi2dof=float(np.sum(res ** 2) / max(len(r) - 2, 1)), sigma=SIGMA.get(T["d"]))


# ------------------------------------------------------------------ частина B
CL_TASKS = {"2D bond sq": "2D_bond_sq_L256", "3D bond": "3D_bond_L48"}
FACT = ("0.9", "1.0", "1.1")


def load_clusters(slug):
    d = ROOT / "data" / "P1" / "p1_15" / slug
    ch = sorted(d.glob("chunk_*.npz"))
    if not ch:
        return None
    zs = [np.load(p) for p in ch]
    out = {}
    for f in FACT:
        parts = {k: [] for k in ("s", "E", "c", "g", "deg", "diam", "ecc_mean", "lmean", "rid", "prof")}
        prof_row = []
        for z in zs:
            for k in ("s", "E", "c", "g", "deg", "diam", "ecc_mean", "lmean", "rid"):
                parts[k].append(z[f"{f}_{k}"])
            pi = z[f"{f}_prof_idx"]
            m = len(z[f"{f}_s"])
            row = np.full(m, -1, np.int64)
            row[pi] = np.arange(len(pi)) + sum(len(x) for x in parts["prof"])
            prof_row.append(row)
            parts["prof"].append(z[f"{f}_prof"])
        o = {k: np.concatenate(v) for k, v in parts.items()}
        o["prof_row"] = np.concatenate(prof_row)
        o["s_big"] = np.concatenate([z[f"{f}_s_big"] for z in zs])
        out[f] = o
    out["R"] = sum(len(z["ridx"]) for z in zs)
    out["meta"] = json.loads((d / "meta.json").read_text())
    return out


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def cluster_boot(rid, B, rng):
    """Індекси бутстрепу за реалізаціями (кластери однієї реалізації — разом)."""
    u, inv = np.unique(rid, return_inverse=True)
    groups = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        yield np.concatenate([groups[i] for i in pick])


def bin_rows(D, f, b, smin=None):
    o = D[f]
    lo, hi = (2 ** b, 2 ** (b + 1)) if smin is None else smin
    return np.flatnonzero((o["s"] >= lo) & (o["s"] < hi))


def covars(D, f, i):
    """Коваріати кластерів: ln s; c/s і (c/s)²; частки степенів 1..2d−1 (остання — лінійно залежна);
    ln D і (ln D)² (D — хімічний діаметр); ln⟨ℓ⟩ і (ln⟨ℓ⟩)² (⟨ℓ⟩ — середня хімічна відстань пар вузлів)."""
    o = D[f]
    s = o["s"][i].astype(float)
    cs = o["c"][i] / s
    deg = o["deg"][i][:, :-1] / s[:, None]
    ld, ll = np.log(o["diam"][i].astype(float)), np.log(o["lmean"][i])
    return dict(lns=np.log(s)[:, None], cyc=np.c_[cs, cs ** 2], deg=deg, diam=np.c_[ld, ld ** 2], ell=np.c_[ll, ll ** 2],
                g=o["g"][i], rid=o["rid"][i], cs=cs)


ETA_SETS = {"c/s": ["cyc"], "степені": ["deg"], "D": ["diam"], "⟨ℓ⟩": ["ell"], "c/s + ⟨ℓ⟩": ["cyc", "ell"],
            "степені + D + ⟨ℓ⟩": ["deg", "diam", "ell"]}
SHIFT_SETS = {"ln s": [], "ln s + c/s": ["cyc"], "ln s + степені": ["deg"], "ln s + D": ["diam"], "ln s + ⟨ℓ⟩": ["ell"],
              "ln s + c/s + ⟨ℓ⟩": ["cyc", "ell"], "ln s + степені + D + ⟨ℓ⟩": ["deg", "diam", "ell"]}


def design(C, keys):
    return np.c_[(np.ones(len(C["g"])), C["lns"][:, 0]) + tuple(C[k] for k in keys)]


def cycles_part(D, rng, B=200):
    """(a) частка розкиду NodeSC за фіксованого біна (після ln s), пояснена c/s, степенями, діаметром;
    (b) зсуви g за p за фіксованого s (фіктивні змінні p, еталон pc) до і після цих коваріат."""
    res = []
    for b in range(6, 11):
        idx = {f: bin_rows(D, f, b) for f in FACT}
        if min(len(v) for v in idx.values()) < 15:
            continue
        row = dict(bin=b, n={f: int(len(v)) for f, v in idx.items()}, by_p={})
        for f in FACT:
            if len(idx[f]) < 50:
                continue
            C = covars(D, f, idx[f])
            X0 = design(C, [])
            r0 = ols(X0, C["g"])[1]
            boots = list(cluster_boot(C["rid"], B, rng))
            e = dict(sd_resid_lns=float(r0.std()), frac_cyclic=float(np.mean(C["cs"] > 0)),
                     mean_c_per_s=float(C["cs"].mean()), corr_resid_cs=float(np.corrcoef(r0, C["cs"])[0, 1]),
                     corr_resid_ell=float(np.corrcoef(r0, C["ell"][:, 0])[0, 1]), eta={})
            for nm, keys in ETA_SETS.items():
                X1 = design(C, keys)
                r1 = ols(X1, C["g"])[1]
                eb = [1 - ols(X1[j], C["g"][j])[1].var() / ols(X0[j], C["g"][j])[1].var() for j in boots]
                e["eta"][nm] = dict(est=float(1 - r1.var() / r0.var()), se=float(np.std(eb, ddof=1)))
            row["by_p"][f] = e
        Cs = [covars(D, f, idx[f]) for f in FACT]
        C = {k: np.concatenate([c[k] for c in Cs]) for k in Cs[0]}
        lab = np.concatenate([np.full(len(idx[f]), k) for k, f in enumerate(FACT)])
        dum = np.c_[lab == 0, lab == 2].astype(float)
        boots = list(cluster_boot(C["rid"], B, rng))
        row["shift"] = {}
        for nm, keys in SHIFT_SETS.items():
            X = np.c_[design(C, keys), dum]
            beta = ols(X, C["g"])[0]
            bb = np.array([ols(X[j], C["g"][j])[0][-2:] for j in boots])
            se = bb.std(0, ddof=1)
            row["shift"][nm] = {"0.9-1.0": dict(est=float(beta[-2]), se=float(se[0]), z=float(beta[-2] / se[0])),
                                "1.1-1.0": dict(est=float(beta[-1]), se=float(se[1]), z=float(beta[-1] / se[1]))}
        res.append(row)
    return res


def dbic(x):
    from sklearn.mixture import GaussianMixture
    x = x.reshape(-1, 1)
    m = len(x)
    bic1 = m * (np.log(2 * np.pi * x.var()) + 1) + 2 * np.log(m)
    gm = GaussianMixture(2, n_init=2, reg_covar=1e-8, random_state=0, max_iter=300).fit(x)
    return float(bic1 - gm.bic(x))


def null_thresholds(x, m, rng, B=100):
    """Нульова модель: закон Джонсона S_U (унімодальний, чотири параметри — асиметрія і важкі хвости), підігнаний
    до x методом максимальної правдоподібності; пороги 95% для dip і ΔBIC при вибірці m."""
    import diptest
    par = sst.johnsonsu.fit(x)
    dips, bics = [], []
    for _ in range(B):
        y = sst.johnsonsu.rvs(*par, size=m, random_state=rng)
        dips.append(diptest.dipstat(y))
        bics.append(dbic(y))
    return float(np.quantile(dips, 0.95)), float(np.quantile(bics, 0.95)), dips, bics


def classes_part(D, rng):
    """(c) класи при pc у біні: dip/ΔBIC проти нульової моделі; позитивний контроль — суміш 0.9 pc і 1.1 pc."""
    import diptest
    out = []
    for b in (6, 7, 8):
        i = bin_rows(D, "1.0", b)
        if len(i) < 200:
            continue
        o = D["1.0"]
        s, c, g = o["s"][i].astype(float), o["c"][i].astype(float), o["g"][i]
        X0 = np.c_[np.ones_like(s), np.log(s)]
        _, rg = ols(X0, g)
        P = o["prof"][o["prof_row"][i]]
        mu = P.mean(0)
        _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
        pcs = (P - mu) @ vt[:2].T
        _, rl = ols(X0, np.log(o["lmean"][i]))
        _, rgl = ols(design(covars(D, "1.0", i), ["cyc", "ell"]), g)
        cj = (c + rng.uniform(-0.5, 0.5, len(c))) / s     # c ціле: розмиття на ±½, інакше dip реагує на дискретність
        feats = {"NodeSC (залишок після ln s)": rg, "NodeSC (залишок після ln s, c/s, ⟨ℓ⟩)": rgl, "c/s (розмито)": cj,
                 "ln⟨ℓ⟩ (залишок після ln s)": rl, "PC1 профілю": pcs[:, 0], "PC2 профілю": pcs[:, 1]}
        m = min(len(i), 2000)
        row = dict(bin=b, m=m, feats={})
        for nm, x in feats.items():
            xs = rng.choice(x, m, replace=False)
            qd, qb, _, _ = null_thresholds(xs, m, rng)
            d, pv = diptest.diptest(xs)
            row["feats"][nm] = dict(dip=float(d), dip_p_table=float(pv), dbic=dbic(xs), null_q95_dip=qd, null_q95_dbic=qb,
                                    groups_dip=bool(d > qd), groups_dbic=bool(dbic(xs) > qb))
        # позитивний контроль: 0.9 pc і 1.1 pc, 50/50, той самий бін; залишок після спільної регресії на ln s
        j9, j11 = bin_rows(D, "0.9", b), bin_rows(D, "1.1", b)
        mm = min(len(j9), len(j11), m // 2)
        if mm >= 100:
            pick9, pick11 = rng.choice(j9, mm, replace=False), rng.choice(j11, mm, replace=False)
            s2 = np.r_[D["0.9"]["s"][pick9], D["1.1"]["s"][pick11]].astype(float)
            g2 = np.r_[D["0.9"]["g"][pick9], D["1.1"]["g"][pick11]]
            c2 = np.r_[D["0.9"]["c"][pick9], D["1.1"]["c"][pick11]].astype(float)
            _, r2 = ols(np.c_[np.ones_like(s2), np.log(s2)], g2)
            pc_row = {}
            c2j = (c2 + rng.uniform(-0.5, 0.5, len(c2))) / s2
            for nm, x in (("NodeSC (залишок після ln s)", r2), ("c/s (розмито)", c2j)):
                qd, qb, _, _ = null_thresholds(x, len(x), rng)
                d, pv = diptest.diptest(x)
                eff = abs(x[:mm].mean() - x[mm:].mean()) / np.sqrt(0.5 * (x[:mm].var() + x[mm:].var()))
                pc_row[nm] = dict(m=int(len(x)), dip=float(d), dip_p_table=float(pv), dbic=dbic(x), null_q95_dip=qd,
                                  null_q95_dbic=qb, groups_dip=bool(d > qd), groups_dbic=bool(dbic(x) > qb),
                                  cohen_d=float(eff))
            row["positive_control_0.9_vs_1.1"] = pc_row
        out.append(row)
    return out


SETS = {"ln s": ["lns"], "ln s + c/s": ["lns", "cs"], "ln s + NodeSC": ["lns", "g"], "ln s + c/s + NodeSC": ["lns", "cs", "g"],
        "база": ["lns", "deg", "diam"], "база + NodeSC": ["lns", "deg", "diam", "g"], "база + профіль": ["lns", "deg", "diam", "prof"],
        "база + ⟨ℓ⟩": ["lns", "deg", "diam", "ell"], "база + ⟨ℓ⟩ + NodeSC": ["lns", "deg", "diam", "ell", "g"],
        "база + ⟨ℓ⟩ + профіль": ["lns", "deg", "diam", "ell", "prof"]}
PAIRS = [("ln s + NodeSC", "ln s"), ("ln s + c/s", "ln s"), ("ln s + c/s + NodeSC", "ln s + c/s"), ("ln s + NodeSC", "ln s + c/s"),
         ("база + NodeSC", "база"), ("база + профіль", "база"), ("база + ⟨ℓ⟩", "база"), ("база + ⟨ℓ⟩ + NodeSC", "база + ⟨ℓ⟩"),
         ("база + ⟨ℓ⟩ + профіль", "база + ⟨ℓ⟩")]


def auroc_part(D, rng, B=300):
    """(d) AUROC розрізнення двох p за однакового біна; поділ навпіл за реалізаціями."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = []
    for fa, fb in (("0.9", "1.1"), ("0.9", "1.0"), ("1.0", "1.1")):
        for b in (6, 7, 8):
            ia, ib = bin_rows(D, fa, b), bin_rows(D, fb, b)
            if min(len(ia), len(ib)) < 150:
                continue
            m = min(len(ia), len(ib), 4000)
            ia, ib = np.sort(rng.choice(ia, m, replace=False)), np.sort(rng.choice(ib, m, replace=False))

            def F(f, i):
                C = covars(D, f, i)
                return dict(lns=C["lns"], cs=C["cyc"][:, :1], g=C["g"][:, None], deg=C["deg"], diam=C["diam"][:, :1],
                            ell=C["ell"][:, :1], prof=D[f]["prof"][D[f]["prof_row"][i]]), C["rid"]
            Fa, ra = F(fa, ia)
            Fb, rb = F(fb, ib)
            y = np.r_[np.ones(m), np.zeros(m)].astype(int)
            rid = np.r_[ra, rb]
            tr = (rid % 2) == 0             # поділ за реалізаціями: парні — навчання, непарні — тест
            te = ~tr
            scores, auc = {}, {}
            for nm, keys in SETS.items():
                X = np.c_[tuple(np.r_[Fa[k], Fb[k]] for k in keys)]
                mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], y[tr])
                scores[nm] = mdl.decision_function(X[te])
                auc[nm] = float(roc_auc_score(y[te], scores[nm]))
            yt = y[te]
            idx = [rng.integers(0, len(yt), len(yt)) for _ in range(B)]
            ci = {nm: [float(q) for q in np.quantile([roc_auc_score(yt[j], scores[nm][j]) for j in idx], [0.025, 0.975])]
                  for nm in SETS}
            diff = {}
            for A, Bn in PAIRS:
                v = [roc_auc_score(yt[j], scores[A][j]) - roc_auc_score(yt[j], scores[Bn][j]) for j in idx]
                diff[f"{A} − {Bn}"] = dict(est=auc[A] - auc[Bn], lo=float(np.quantile(v, 0.025)), hi=float(np.quantile(v, 0.975)))
            out.append(dict(pair=f"{fa} pc / {fb} pc", bin=b, m=int(m), n_test=int(te.sum()), auc=auc, ci=ci, diff=diff))
    return out


def cs_identity(D):
    """Тотожність c/s = ½Σ_k k f_k − 1 + 1/s (f_k — частка вузлів зі степенем k): c/s — функція гістограми степенів."""
    err = 0.0
    for f in FACT:
        o = D[f]
        s = o["s"].astype(float)
        k = np.arange(1, o["deg"].shape[1] + 1)
        rhs = 0.5 * (o["deg"] * k).sum(1) / s - 1 + 1 / s
        err = max(err, float(np.max(np.abs(o["c"] / s - rhs))))
    return err


def dip_power(D, rng, ds=(0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0), reps=40, m=2000):
    """Потужність dip-тесту (таблиця Хартиганів, p < 0.05) на синтетичній суміші 50/50: залишок NodeSC після ln s
    (або розмита c/s) при pc і його копія, зсунута на d стандартних відхилень. Показує, яке розділення класів
    тест узагалі здатен виявити на реальній формі розподілу при m = 2000."""
    import diptest
    out = []
    for b in (6, 7, 8):
        i = bin_rows(D, "1.0", b)
        if len(i) < 200:
            continue
        o = D["1.0"]
        s, c, g = o["s"][i].astype(float), o["c"][i].astype(float), o["g"][i]
        _, rg = ols(np.c_[np.ones_like(s), np.log(s)], g)
        cj = (c + rng.uniform(-0.5, 0.5, len(c))) / s
        row = dict(bin=b, m=m, reps=reps, d=list(ds), power={})
        for nm, x in (("NodeSC (залишок після ln s)", rg), ("c/s (розмито)", cj)):
            sd = x.std()
            pw = []
            for dd in ds:
                hit = 0
                for _ in range(reps):
                    a = rng.choice(x, m // 2, replace=len(x) < m)
                    bb = rng.choice(x, m // 2, replace=len(x) < m) + dd * sd
                    hit += diptest.diptest(np.r_[a, bb])[1] < 0.05
                pw.append(hit / reps)
            row["power"][nm] = pw
        out.append(row)
    return out


def auroc_splits(D, rng, n_split=10):
    """Мінливість AUROC за поділом: n_split випадкових поділів реалізацій навпіл (і нова підвибірка кластерів)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = []
    for fa, fb in (("0.9", "1.1"), ("0.9", "1.0"), ("1.0", "1.1")):
        for b in (6, 7, 8):
            ia0, ib0 = bin_rows(D, fa, b), bin_rows(D, fb, b)
            if min(len(ia0), len(ib0)) < 150:
                continue
            m = min(len(ia0), len(ib0), 4000)
            A = {nm: [] for nm in SETS}
            for _ in range(n_split):
                ia, ib = np.sort(rng.choice(ia0, m, replace=False)), np.sort(rng.choice(ib0, m, replace=False))
                feats = []
                for f, i in ((fa, ia), (fb, ib)):
                    C = covars(D, f, i)
                    feats.append(dict(lns=C["lns"], cs=C["cyc"][:, :1], g=C["g"][:, None], deg=C["deg"], diam=C["diam"][:, :1],
                                      ell=C["ell"][:, :1], prof=D[f]["prof"][D[f]["prof_row"][i]], rid=C["rid"]))
                y = np.r_[np.ones(m), np.zeros(m)].astype(int)
                rid = np.r_[feats[0]["rid"], feats[1]["rid"]]
                u = np.unique(rid)
                tr = np.isin(rid, rng.choice(u, len(u) // 2, replace=False))
                te = ~tr
                for nm, keys in SETS.items():
                    X = np.c_[tuple(np.r_[feats[0][k], feats[1][k]] for k in keys)]
                    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], y[tr])
                    A[nm].append(float(roc_auc_score(y[te], mdl.decision_function(X[te]))))
            summ = {nm: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), min=float(np.min(v)), max=float(np.max(v)))
                    for nm, v in A.items()}
            dif = {}
            for a_, b_ in PAIRS:
                v = np.array(A[a_]) - np.array(A[b_])
                dif[f"{a_} − {b_}"] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()))
            out.append(dict(pair=f"{fa} pc / {fb} pc", bin=b, m=int(m), n_split=n_split, auc=summ, diff=dif))
    return out


# ------------------------------------------------------------------ рисунки
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e6e5e1"
PCOL = {"0.9": "#2a78d6", "1.0": "#0b0b0b", "1.1": "#eb6834"}


def style():
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": GRIDC, "axes.grid": True, "grid.color": GRIDC,
                         "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})


def fig_journal(J, path):
    style()
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for k, (name, L, a) in enumerate((("2D bond sq", 1024, ax[0]), ("3D bond", 256, ax[1]))):
        T = next(t for t in J[name] if t["L"] == L)
        for r in T["rows"]:
            col = cmap((r["bin"] - 6) / 8)
            a.errorbar(r["p"], r["g"], r["se"], fmt="o-", ms=3, lw=1.1, color=col, capsize=0,
                       label=f"s ∈ [{r['s_range'][0]}, {r['s_range'][1] + 1})")
        a.axvline(T["pc"], color=MUTED, lw=1, ls="--")
        a.set(xlabel="p", ylabel="g(s; p) — середня NodeSC у біні s",
              title=f"{'2D квадратна (зв.)' if k == 0 else '3D кубічна'}, L = {L}: g за фіксованого біна s\n(штрих — pc)")
        a.legend(fontsize=7, frameon=False, ncol=2)
    c = ax[2]
    for name, col in (("2D bond sq", "#2a78d6"), ("2D bond tri", "#1baf7a"), ("2D site sq", "#eb6834"), ("3D bond", "#7b5cd6")):
        T = J[name][-1]
        r = [x for x in T["rows"] if np.isfinite(x["slope_loc"])]
        c.errorbar([x["sbar_pc"] for x in r], [-x["slope_loc"] for x in r], [x["se_slope_loc"] for x in r], fmt="o-",
                   ms=4, lw=1.2, color=col, capsize=2, label=f"{name}, L = {T['L']}")
    s0 = np.array([80, 2e4])
    for d, ls in ((2, "--"), (3, ":")):
        c.plot(s0, 1.2 * (s0 / 80) ** SIGMA[d], ls, color=MUTED, lw=1)
        c.text(s0[1], 1.2 * (s0[1] / 80) ** SIGMA[d], f" s^σ, d={d}", fontsize=8, color=MUTED, va="center")
    c.set(xscale="log", yscale="log", xlabel="s̄ у біні (при pc)", ylabel="−∂g/∂p у pc",
          title="Локальний нахил g(s; p) у pc\n(лінії — нахил σ = 1/(ν d_f), нормування довільне)")
    c.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cycles(D2, D3, path):
    style()
    fig, ax = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
    b = 7
    for k, (D, ttl) in enumerate(((D2, "2D квадратна (зв.), L = 256"), (D3, "3D кубічна, L = 48"))):
        if D is None:
            continue
        for xkey, a in (("cs", ax[0, k]), ("ell", ax[1, k])):
            for f in FACT:
                i = bin_rows(D, f, b)
                o = D[f]
                s = o["s"][i].astype(float)
                x = (o["c"][i] + np.random.default_rng(0).uniform(-0.5, 0.5, len(i))) / s if xkey == "cs" else o["lmean"][i] / s ** 0.5
                a.scatter(x, o["g"][i], s=3, alpha=0.2, color=PCOL[f], lw=0, label=f"p = {f} pc (N = {len(i)})")
            a.set(xlabel="щільність циклів c/s (c розмито на ±½)" if xkey == "cs" else "⟨ℓ⟩ / s^½ (середня хімічна відстань, масштабована)",
                  ylabel="NodeSC кластера (середня за вузлами)", title=f"{ttl}: кластери s ∈ [{2 ** b}, {2 ** (b + 1)})")
            if xkey == "ell":
                a.set_xscale("log")
            lg = a.legend(fontsize=8, frameon=False, markerscale=4)
            for h in lg.legend_handles:
                h.set_alpha(1)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def strip(T):
    return {k: v for k, v in T.items() if not k.startswith("_")}


def main():
    rng = np.random.default_rng(SEED)
    J = {}
    for name, Ls in JOBS.items():
        J[name] = [journal_task(name, L, rng) for L in Ls]
        for T in J[name]:
            zs = [(r["bin"], round(r["z_slope"], 1), round(r["z_slope_corr"], 1), round(r["z_slope_loc"], 1)) for r in T["rows"]]
            print(name, T["L"], "R", T["R"], "(бін, z нахилу, z з поправкою s̄, z локальний):", zs, flush=True)
    Lcmp = {name: [dict(L1=a["L"], L2=b["L"], bins=compare_L(a, b)) for a, b in zip(T[:-1], T[1:])] for name, T in J.items()}
    sc = {name: [dict(L=t["L"], fit=scaling(t), fit_b_le_10=scaling(t, 10)) for t in T] for name, T in J.items()}
    for name, v in sc.items():
        print("скейлінг", name, [(x["L"], x["fit"] and (round(x["fit"]["y"], 3), round(x["fit"]["se_y"], 3)),
                                  x["fit_b_le_10"] and (round(x["fit_b_le_10"]["y"], 3), round(x["fit_b_le_10"]["se_y"], 3))) for x in v],
              "σ =", SIGMA.get(an[name]["d"]), flush=True)
    fig_journal(J, ROOT / "results" / "P1" / "fig_p1_15_journal.png")
    partB = {}
    Ds = {}
    for name, slug in CL_TASKS.items():
        D = load_clusters(slug)
        Ds[name] = D
        if D is None:
            continue
        cyc = cycles_part(D, rng)
        for r in cyc:
            print(name, "бін", r["bin"], r["n"], {f: {k: round(x["est"], 3) for k, x in v["eta"].items()} for f, v in r["by_p"].items()},
                  {k: (round(v["0.9-1.0"]["est"], 4), round(v["0.9-1.0"]["z"], 1), round(v["1.1-1.0"]["est"], 4),
                       round(v["1.1-1.0"]["z"], 1)) for k, v in r["shift"].items()}, flush=True)
        cls = classes_part(D, rng)
        au = auroc_part(D, rng)
        for r in au:
            print(name, r["pair"], r["bin"], {k: round(v, 3) for k, v in r["auc"].items()}, flush=True)
        partB[name] = dict(R=D["R"], data_meta=D["meta"], cycles=cyc, classes=cls, auroc=au,
                           s_big_mean={f: float(D[f]["s_big"].mean()) for f in FACT})
    # доповнення після перевірки (окремий генератор, щоб числа вище не змінилися)
    rng2 = np.random.default_rng([SEED, 2])
    for name, D in Ds.items():
        if D is None:
            continue
        partB[name]["cs_identity_max_err"] = cs_identity(D)
        partB[name]["dip_power"] = dip_power(D, rng2)
        for r in partB[name]["dip_power"]:
            print(name, "потужність dip, бін", r["bin"], r["power"], flush=True)
        partB[name]["auroc_splits"] = auroc_splits(D, rng2)
        for r in partB[name]["auroc_splits"]:
            print(name, r["pair"], r["bin"], {k: (round(v["mean"], 3), round(v["min"], 3), round(v["max"], 3))
                                             for k, v in r["auc"].items() if k in ("ln s", "ln s + NodeSC", "ln s + c/s")}, flush=True)
    n_hit = sum(f["groups_dbic"] for c in partB.values() for r in c["classes"] for f in r["feats"].values())
    n_tot = sum(len(r["feats"]) for c in partB.values() for r in c["classes"])
    n_dip = sum(f["groups_dip"] for c in partB.values() for r in c["classes"] for f in r["feats"].values())
    n_pc = sum(len(r.get("positive_control_0.9_vs_1.1", {})) for c in partB.values() for r in c["classes"])
    n_pc_hit = sum(v["groups_dip"] or v["groups_dbic"] for c in partB.values() for r in c["classes"]
                   for v in r.get("positive_control_0.9_vs_1.1", {}).values())
    summary = dict(n_cases=n_tot, n_dip=int(n_dip), n_dbic=int(n_hit), p_binom_dbic=float(sst.binom.sf(n_hit - 1, n_tot, 0.05)),
                   n_dbic_negative=int(sum(f["groups_dbic"] and f["dbic"] < 0 for c in partB.values() for r in c["classes"]
                                           for f in r["feats"].values())),
                   n_positive_control=n_pc, n_positive_control_detected=int(n_pc_hit))
    print("підсумок класів:", summary, flush=True)
    fig_cycles(Ds.get("2D bond sq"), Ds.get("3D bond"), ROOT / "results" / "P1" / "fig_p1_15_cycles.png")
    meta = runmeta(__file__, params=dict(B_boot=B_BOOT, bins=list(BINS), eps_max=EPS_MAX, min_c=MIN_C, min_r=MIN_R,
                                         hw_loc=HW_LOC, jobs=JOBS, sigma=SIGMA), seeds=[SEED])
    save(ROOT / "results" / "P1" / "p1_15_classes.json", meta,
         journal={name: [strip(t) for t in T] for name, T in J.items()}, L_compare=Lcmp, scaling=sc, clusters=partB,
         classes_summary=summary)


if __name__ == "__main__":
    main()
