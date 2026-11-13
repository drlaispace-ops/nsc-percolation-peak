"""Незалежна перевірка завдання C (П1.13, решітки). Не імпортує аналітичних скриптів виконавця
(p1_13_lattice.py, C_checks.py, p1_13_threshold.py). Читає сирі шматки data/P1/p1_3/<slug>/L<L>/chunk_*.npz
(власний завантажувач), results/P1/p1_3/*.json (лише сітку ps), data/P1/constants.csv, results/P1/p1_11_a_table.npy,
results/P1/p1_13_threshold.json (лише для перерахунку екстраполяції). Бібліотеки: numpy, scipy, code/lib/nsccore.
Запуск: NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=1 .venv/bin/python notes/orchestra/verify_C.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
rng = np.random.default_rng(777)

PC = {"2D_bond_sq": 0.5, "2D_bond_tri": 2 * np.sin(np.pi / 18), "2D_site_sq": 0.59274621,
      "3D_bond": 0.24881182, "4D_bond": 0.16013122}
EX = {}
for r in csv.DictReader(open(ROOT / "data" / "P1" / "constants.csv")):
    if r["system"].startswith("perc_") and r["quantity"] in ("beta", "nu", "d_f", "d_min"):
        EX.setdefault(int(r["d"]), {})[r["quantity"]] = float(r["value"])


def raw(slug, L, keys):
    fs = sorted((ROOT / "data" / "P1" / "p1_3" / slug / f"L{L}").glob("chunk_*.npz"))
    ch = [np.load(f) for f in fs]
    ps = np.array(json.load(open(ROOT / "results" / "P1" / "p1_3" / f"{slug}_L{L}.json"))["ps"])
    return ps, {k: np.concatenate([c[k] for c in ch]) for k in keys}


def interp_at(ps, Y, x):
    k = np.searchsorted(ps, x) - 1
    w = (x - ps[k]) / (ps[k + 1] - ps[k])
    return (1 - w) * Y[:, k] + w * Y[:, k + 1]


out = {}
# ---------- 1. g_G, g_F, NSC при pc (найбільше L) ----------
print("1. g_G(pc), g_F(pc) із сирих шматків")
out["gpc"] = {}
for slug, L in (("2D_bond_sq", 1024), ("2D_bond_tri", 512), ("2D_site_sq", 512), ("3D_bond", 256), ("4D_bond", 32)):
    ps, S = raw(slug, L, ["g_giant", "g_fin", "nsc", "P"])
    v = {k: float(interp_at(ps, S[k], PC[slug]).mean()) for k in S}
    v["gG_minus_gF"] = v["g_giant"] - v["g_fin"]
    v["R"] = int(len(S["nsc"]))
    out["gpc"][f"{slug}_L{L}"] = v
    print(f"  {slug} L={L}: g_G={v['g_giant']:.4f} g_F={v['g_fin']:.4f} diff={v['gG_minus_gF']:.4f} R={v['R']}")

# ---------- 2. журнал g(s; pc): θ, g∞ (власна зважена підгонка) і чутливість g∞ до θ ----------
print("2. θ і g∞ з журналу скінченних кластерів")
out["journal"] = {}


def journal_bins(slug, L):
    ps, S = raw(slug, L, ["sbin_n", "sbin_sc", "sbin_c"])
    at = {k: interp_at(ps, S[k], PC[slug]) for k in S}  # (R, NBIN)
    return at


def bin_stats(at, idx):
    n, sc, c = at["sbin_n"][idx].sum(0), at["sbin_sc"][idx].sum(0), at["sbin_c"][idx].sum(0)
    ok = c >= 30
    return np.flatnonzero(ok), n[ok] / c[ok], sc[ok] / n[ok]


for slug, L, dim in (("2D_bond_sq", 1024, 2), ("3D_bond", 256, 3), ("4D_bond", 32, 4)):
    at = journal_bins(slug, L)
    R = len(at["sbin_n"])
    b, s, g = bin_stats(at, np.arange(R))
    # похибка біну — бутстреп за реалізаціями
    bs = []
    for _ in range(200):
        bb, ss, gg = bin_stats(at, rng.integers(0, R, R))
        row = np.full(len(b), np.nan)
        for i, k in enumerate(b):
            j = np.flatnonzero(bb == k)
            if len(j):
                row[i] = gg[j[0]]
        bs.append(row)
    se = np.nanstd(np.array(bs), 0, ddof=1)
    th_hyp = 1 - EX[dim]["d_min"] / EX[dim]["d_f"]
    res = {}
    for bmin in (4, 6, 8):
        m = (b >= bmin) & np.isfinite(se) & (se > 0)
        f3 = lambda x, gi, a, t: gi - a * x ** -t  # noqa: E731
        c3, cov3 = curve_fit(f3, s[m], g[m], p0=[g[m][-1], 1, 0.6], sigma=se[m], absolute_sigma=True, maxfev=20000)
        chi3 = float(np.sum(((g[m] - f3(s[m], *c3)) / se[m]) ** 2) / (m.sum() - 3))
        f2 = lambda x, gi, a: gi - a * x ** -th_hyp  # noqa: E731
        c2, cov2 = curve_fit(f2, s[m], g[m], p0=[g[m][-1], 1], sigma=se[m], absolute_sigma=True)
        chi2 = float(np.sum(((g[m] - f2(s[m], *c2)) / se[m]) ** 2) / (m.sum() - 2))
        res[str(bmin)] = dict(nbins=int(m.sum()), theta=float(c3[2]), se_theta=float(np.sqrt(cov3[2, 2])), g_inf=float(c3[0]),
                              se_g_inf=float(np.sqrt(cov3[0, 0])), chi2dof=chi3,
                              g_inf_thetahyp=float(c2[0]), chi2dof_thetahyp=chi2)
        print(f"  {slug} L={L} s≥2^{bmin}: θ={c3[2]:.3f}±{np.sqrt(cov3[2,2]):.3f} g∞={c3[0]:.4f}±{np.sqrt(cov3[0,0]):.4f} χ²/dof={chi3:.2f}"
              f" | θ=θ_hyp={th_hyp:.3f}: g∞={c2[0]:.4f} χ²/dof={chi2:.2f}")
    out["journal"][f"{slug}_L{L}"] = dict(theta_hyp=th_hyp, fits=res, s_max=float(s.max()))

# ---------- 3. Відхід гіганта: D = g∞ − g_G(p), C0 ----------
print("3. C0 з підгонки g_G(p) вище порога (L — найбільше), g∞ = журнал s≥2^6")
out["C0"] = {}
for slug, L, Lm in (("2D_bond_sq", 1024, 512), ("3D_bond", 256, 192), ("4D_bond", 32, 24)):
    ps, S = raw(slug, L, ["g_giant"])
    _, S2 = raw(slug, Lm, ["g_giant"])
    g1, g2 = S["g_giant"].mean(0), S2["g_giant"].mean(0)
    s1 = S["g_giant"].std(0, ddof=1) / np.sqrt(len(S["g_giant"]))
    s2 = S2["g_giant"].std(0, ddof=1) / np.sqrt(len(S2["g_giant"]))
    e = (ps - PC[slug]) / PC[slug]
    ok = (e > 0) & (np.abs(g1 - g2) < 3 * np.hypot(s1, s2) + 0.002)
    gi = out["journal"][f"{slug}_L{L}"]["fits"]["6"]
    r = {}
    for lab, ginf in (("g_inf_fit", gi["g_inf"]), ("g_inf_thetahyp", gi["g_inf_thetahyp"])):
        D = ginf - g1[ok]
        f = lambda x, C, a, bb: C + a * x + bb * x * np.log(x)  # noqa: E731
        c, cov = curve_fit(f, e[ok], D, sigma=s1[ok] + 1e-4, absolute_sigma=True)
        # повна похибка C0 з урахуванням похибки g∞ (зсув D як цілого)
        r[lab] = dict(C0=float(c[0]), se_C0_stat=float(np.sqrt(cov[0, 0])),
                      se_C0_with_ginf=float(np.hypot(np.sqrt(cov[0, 0]), gi["se_g_inf"])), eps_min=float(e[ok].min()), npts=int(ok.sum()))
        print(f"  {slug}: {lab}: C0={c[0]:+.4f} ±{np.sqrt(cov[0,0]):.4f} (стат) ⊕ σ(g∞)={gi['se_g_inf']:.4f}; ε_min={e[ok].min():.4f}, n={ok.sum()}")
    out["C0"][slug] = r

# ---------- 4. Пряма підгонка ψ ----------
print("4. Пряма підгонка NSC = R0 + R1 ε + R2 ε² [+R3 ε³] − c ε₊^ψ")
out["psi_direct"] = {}


def fitpsi(e, y, s, cubic):
    if cubic:
        f = lambda x, R0, R1, R2, R3, c, p: R0 + R1 * x + R2 * x ** 2 + R3 * x ** 3 - c * np.clip(x, 0, None) ** p  # noqa: E731
        p0, lo, hi = [y.mean(), 1, 0, 0, 1, 1.2], [-10, -100, -1e4, -1e6, 0, 0.2], [10, 100, 1e4, 1e6, 1e3, 3]
    else:
        f = lambda x, R0, R1, R2, c, p: R0 + R1 * x + R2 * x ** 2 - c * np.clip(x, 0, None) ** p  # noqa: E731
        p0, lo, hi = [y.mean(), 1, 0, 1, 1.2], [-10, -100, -1e4, 0, 0.2], [10, 100, 1e4, 1e3, 3]
    c, cov = curve_fit(f, e, y, p0=p0, sigma=s, absolute_sigma=True, bounds=(lo, hi), maxfev=50000)
    chi = float(np.sum(((y - f(e, *c)) / s) ** 2) / (len(e) - len(c)))
    xx = np.linspace(e.min(), e.max(), 200001)
    return c, float(np.sqrt(cov[-1, -1])), chi, float(xx[np.argmax(f(xx, *c))])


for slug, L, win in (("3D_bond", 256, (-0.06, 0.12)), ("3D_bond", 256, (-0.1, 0.2)), ("4D_bond", 32, (-0.1, 0.2)),
                     ("2D_bond_sq", 1024, (-0.03, 0.06))):
    ps, S = raw(slug, L, ["nsc"])
    Y = S["nsc"]
    e = (ps - PC[slug]) / PC[slug]
    m = (e >= win[0]) & (e <= win[1])
    y, s = Y[:, m].mean(0), Y[:, m].std(0, ddof=1) / np.sqrt(len(Y))
    for cubic in (False, True):
        c, sp, chi, epk = fitpsi(e[m], y, s, cubic)
        key = f"{slug} L={L} win={win} cubic={cubic}"
        out["psi_direct"][key] = dict(psi=float(c[-1]), se_psi=sp, c=float(c[-2]), R1=float(c[1]), chi2dof=chi,
                                      p_peak=float(PC[slug] * (1 + epk)), npts=int(m.sum()))
        print(f"  {key}: ψ={c[-1]:.3f}±{sp:.3f} c={c[-2]:.3f} R1={c[1]:.3f} χ²/dof={chi:.2f} p_peak={PC[slug]*(1+epk):.4f} n={m.sum()}")

# ---------- 5. Односторонні нахили 3D, L = 256 ----------
print("5. Нахили між сусідніми вузлами сітки, 3D L=256")
ps, S = raw("3D_bond", 256, ["nsc"])
y = S["nsc"].mean(0)
k = int(np.searchsorted(ps, PC["3D_bond"]))
sl = [((ps[i] + ps[i + 1]) / 2 / PC["3D_bond"] - 1, (y[i + 1] - y[i]) / (ps[i + 1] - ps[i])) for i in range(k - 3, k + 2)]
out["one_sided_3D_256"] = sl
print("  " + ", ".join(f"ε={a:+.4f}: {b:.2f}" for a, b in sl))

# ---------- 6. Середнє поле: θ з точної таблиці a_n ----------
print("6. Середнє поле: a_n − a∞")
a = np.load(ROOT / "results" / "P1" / "p1_11_a_table.npy")
N = len(a) - 1
n = np.arange(200, N + 1)
mods = {"a∞ + (α ln n + β)/n": (lambda n, ai, al, be: ai + (al * np.log(n) + be) / n, [1.4877, -0.7, -4.7]),
        "a∞ + A n^-θ": (lambda n, ai, A, t: ai - A * n ** -t, [1.4877, 1, 0.9]),
        "a∞ + A n^-1/2 + B/n": (lambda n, ai, A, B: ai + A * n ** -0.5 + B / n, [1.4877, -1, -1])}
out["mf"] = {}
for lab, (f, p0) in mods.items():
    c = curve_fit(f, n.astype(float), a[n], p0=p0, maxfev=50000)[0]
    rms = float(np.std(a[n] - f(n.astype(float), *c)))
    out["mf"][lab] = dict(par=c.tolist(), rms=rms)
    print(f"  {lab}: {np.round(c, 6)} rms={rms:.2e}")
# локальний ефективний показник із вільним a∞ (з першої моделі)
ai = out["mf"]["a∞ + (α ln n + β)/n"]["par"][0]
out["mf"]["ainf_free"] = ai
for NN in (100, 1000, 3000):
    print(f"  локальний нахил ln|a_n−a∞| при n={NN}: {np.log(abs(a[2*NN]-ai)/abs(a[NN]-ai))/np.log(2):.3f}")

# Незалежна перевірка самої таблиці: a_n — середня NodeSC рівномірного дерева Келі з n вузлів (Прюфер)
import nsccore as nc  # noqa: E402


def prufer_tree(nn, rng):
    seq = rng.integers(0, nn, nn - 2)
    deg = np.ones(nn, int)
    np.add.at(deg, seq, 1)
    import heapq
    leaves = [i for i in range(nn) if deg[i] == 1]
    heapq.heapify(leaves)
    u, v = [], []
    for x in seq:
        lf = heapq.heappop(leaves)
        u.append(lf); v.append(x)
        deg[x] -= 1
        if deg[x] == 1:
            heapq.heappush(leaves, x)
    u.append(heapq.heappop(leaves)); v.append(heapq.heappop(leaves))
    return np.array(u), np.array(v)


out["mf"]["mc_check"] = {}
for nn, reps in ((10, 4000), (50, 1500), (300, 300)):
    vals = []
    for _ in range(reps):
        u, v = prufer_tree(nn, rng)
        vals.append(np.mean(nc.node_sc(nc.csr_from_edges(nn, u, v), nthreads=2)))
    vals = np.array(vals)
    mc, se = float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(reps))
    out["mf"]["mc_check"][str(nn)] = dict(mc=mc, se=se, table=float(a[nn]), z=(mc - float(a[nn])) / se)
    print(f"  МК Прюфер n={nn}: {mc:.5f}±{se:.5f}, таблиця {a[nn]:.5f}, z={(mc-a[nn])/se:+.2f}")

# ---------- 7. Формули і підстановки ----------
print("7. Формули")
form = {}
for dim in (2, 3, 4):
    x = EX[dim]
    inv_s = x["nu"] * x["d_f"]
    tau = 2 + x["beta"] / inv_s
    form[dim] = dict(inv_sigma=inv_s, tau=tau, theta_hyp=1 - x["d_min"] / x["d_f"], x_hyp=x["nu"] * (x["d_f"] - x["d_min"]),
                     psiF_hyp=x["beta"] + x["nu"] * (x["d_f"] - x["d_min"]),
                     psiF_from_tau=(tau - 2 + (1 - x["d_min"] / x["d_f"])) * inv_s)
    print(f"  d={dim}: 1/σ={inv_s:.4f} τ={tau:.4f} θ_hyp={form[dim]['theta_hyp']:.4f} x_hyp={form[dim]['x_hyp']:.4f}"
          f" ψ_F(hyp)={form[dim]['psiF_hyp']:.4f} (через τ: {form[dim]['psiF_from_tau']:.4f})")
# межа скінченних різниць L* = (h/pc)^(−ν); крок h — з сітки p1_3 біля pc
for slug, L, dim in (("2D_bond_sq", 1024, 2), ("3D_bond", 256, 3), ("4D_bond", 32, 4)):
    ps = np.array(json.load(open(ROOT / "results" / "P1" / "p1_3" / f"{slug}_L{L}.json"))["ps"])
    k = int(np.argmin(np.abs(ps - PC[slug])))
    h = ps[k + 1] - ps[k]
    form[f"Lstar_{slug}"] = float((h / PC[slug]) ** (-EX[dim]["nu"]))
    print(f"  {slug}: h={h:.5f} L*={(h/PC[slug])**(-EX[dim]['nu']):.0f}")
# середнє поле: c_G = P1 q1, P1 = 2A, q1 = 2/σ², стрибок −2c_G = −8A/σ²
for m, A, s2 in (("ER", 1, 1), ("z3", 3, 0.5), ("z4", 2, 2 / 3)):
    print(f"  {m}: −2·(2A)(2/σ²) = {-2 * 2 * A * 2 / s2:.4f}")
# екстраполяція напівсуми других похідних ER (p1_13_threshold.json: ER_second, slopes)
T = json.load(open(ROOT / "results" / "P1" / "p1_13_threshold.json"))
ee = np.array([r["eps"] for r in T["ER"]["slopes"]])
CC = np.array([r["C"] for r in T["ER"]["slopes"]])
for sel in (ee <= 0.01, ee <= 0.005):
    co = np.linalg.lstsq(np.c_[np.ones(sel.sum()), ee[sel], ee[sel] * np.log(ee[sel])], CC[sel], rcond=None)[0]
    print(f"  ER напівсума: екстраполяція (1, ε, ε ln ε) по ε≤{ee[sel].max()}: {co[0]:.4f}; передбачення {T['ER_second']['sum_half_pred']:.4f}")
    form[f"ER_half_extrap_{ee[sel].max()}"] = float(co[0])
co = np.polyfit(ee[-3:], CC[-3:], 1)
print(f"  ER напівсума: лінійна по трьох найменших ε: {co[1]:.4f}")
form["ER_half_lin3"] = float(co[1])
out["formulas"] = {str(k): v for k, v in form.items()}

json.dump(out, open(Path(__file__).with_name("verify_C.json"), "w"), ensure_ascii=False, indent=1, default=float)
print("збережено notes/orchestra/verify_C.json")
