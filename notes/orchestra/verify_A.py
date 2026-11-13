"""Незалежна перевірка завдання A (П1.3b): решітки, пороги, піки, p∞, Бете.
Не імпортує p1_3b_run / p1_3b_analyse / p1_3_analyse; читає сирі шматки data/P1/p1_3b, data/P1/p1_3 і сітки ps із
results/P1/p1_3b/*.json, results/P1/p1_3/*.json. Бібліотеки: lattices3d (лише як об'єкт перевірки), gwexact.
Запуск: NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=1 .venv/bin/python notes/orchestra/verify_A.py → notes/orchestra/verify_A.json
"""
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit, minimize_scalar
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import gwexact as gx  # noqa: E402
import lattices3d as lt  # noqa: E402

OUT = Path(__file__).with_name("verify_A.json")
rng = np.random.default_rng(20261112)
res = {}

# ------------------------------------------------------------ 1. решітки: незалежна побудова з координат
BASIS = {  # координати в одиницях a (ребро кубічної комірки)
    "sc": [(0, 0, 0)],
    "bcc": [(0, 0, 0), (.5, .5, .5)],
    "fcc": [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)],
    "diamond": [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5), (.25, .25, .25), (.75, .75, .25), (.75, .25, .75),
                (.25, .75, .75)],
}
NN = {"sc": 1.0, "bcc": np.sqrt(3) / 2, "fcc": np.sqrt(2) / 2, "diamond": np.sqrt(3) / 4}


def coords_graph(name, L):
    pts = np.array([np.array(c) + b for c in product(range(L), repeat=3) for b in BASIS[name]], float)
    t = cKDTree(pts, boxsize=L + 1e-9)
    pairs = t.query_pairs(NN[name] * 1.001, output_type="ndarray")
    return pts, pairs


def lt_coords(name, L):
    """Координати вузлів lattices3d за описом у його docstring (для звірки множин ребер)."""
    if name == "bcc":
        r = np.array(list(product(range(L), repeat=3)), float)       # індекс r = (x·L + y)·L + z
        return np.vstack([r, r + .5])
    M = 2 * L
    g = np.array([x for x in product(range(M), repeat=3) if sum(x) % 2 == 0])
    idx = (g[:, 0] * M + g[:, 1]) * L + g[:, 2] // 2
    c = np.empty((4 * L ** 3, 3))
    c[idx] = g / 2.0
    if name == "fcc":
        return c
    return np.vstack([c, c + .25])


def canon(pts, pairs, L):
    key = lambda x: tuple(np.round(np.mod(x, L) * 4).astype(int))
    return {frozenset((key(pts[a]), key(pts[b]))) for a, b in pairs}


def c3_c4(n, pairs):
    import scipy.sparse as sp
    A = sp.coo_array((np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                     shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    C = (A @ A).tocsr()                                          # спільні сусіди m_ik
    C.setdiag(0)
    C.eliminate_zeros()
    tri = np.asarray(C.multiply(A).sum(1)).ravel() / 2
    m = C.copy()
    m.data = m.data * (m.data - 1) / 2                           # Σ_k C(m_ik, 2): 4-цикли з протилежною вершиною k
    sq = np.asarray(m.sum(1)).ravel()
    s2 = np.asarray(((C > 0).astype(int) - (C.multiply(A) > 0).astype(int)).sum(1)).ravel()
    return deg, tri, sq, s2


latt = {}
for name, L in (("bcc", 4), ("fcc", 4), ("diamond", 3), ("sc", 7)):
    pts, pairs = coords_graph(name, L)
    n_lt, u, v = lt.EDGES[name](L)
    same = None
    if name != "sc":
        same = canon(pts, pairs, L) == canon(lt_coords(name, L), np.c_[u, v], L)
    multi = len({frozenset(e) for e in zip(u.tolist(), v.tolist())}) != len(u)
    selfl = bool(np.any(u == v))
    deg, tri, sq, s2 = c3_c4(len(pts), pairs)
    latt[name] = dict(L=L, n=len(pts), n_lt=int(n_lt), edges=len(pairs), edges_lt=len(u), edge_sets_equal=same,
                      multi_edges_lt=multi, self_loops_lt=selfl, z=sorted(set(deg.astype(int).tolist())),
                      c3=sorted(set(tri.tolist())), c4=sorted(set(sq.tolist())), sphere2=sorted(set(s2.tolist())))
    print("решітка", name, latt[name], flush=True)
res["lattices_check"] = latt

# ------------------------------------------------------------ 2. пороги: проти тексту PDF (звірено вручну, Xu2014 табл. V)
CONST = {r["system"]: r for r in csv.DictReader(open(ROOT / "data" / "P1" / "constants.csv"))}
XU_TABLE_V = {"diamond": 0.3895892, "bcc": 0.18028762, "fcc": 0.12016377}   # з pdftotext library/P1/Xu2014.pdf
res["pc_check"] = {k: dict(csv=float(CONST[f"perc_bond_{k}"]["value"]), xu=v,
                           ok=abs(float(CONST[f"perc_bond_{k}"]["value"]) - v) < 1e-9) for k, v in XU_TABLE_V.items()}
print("пороги", res["pc_check"])

# ------------------------------------------------------------ 3. Бете: власний argmax
bethe = {}
for z in (4, 6, 8, 10, 12):
    m = f"z{z}"
    pc = 1 / (z - 1)
    r = {}
    for h in (0.01, 0.005):
        f = lambda p: -gx.nsc_quad(m, p, h=h)
        g = np.linspace(pc * 1.02, pc * 1.8, 60)
        i = int(np.argmin([f(x) for x in g]))
        o = minimize_scalar(f, bracket=(g[i - 1], g[i], g[i + 1]), tol=1e-9)
        r[f"h{h}"] = float(o.x)
    bethe[z] = dict(p_star=r["h0.01"], p_star_h005=r["h0.005"], P_star=gx.giant(r["h0.01"], m))
    print("Бете", z, bethe[z], flush=True)
res["bethe"] = bethe

# ------------------------------------------------------------ 4. піки з сирих даних
SRC = {  # ключ → (тека даних, тека/шаблон підсумків, z, ν)
    "diamond": ("data/P1/p1_3b/diamond", "results/P1/p1_3b/diamond_L{L}.json", 4, 0.876),
    "bcc": ("data/P1/p1_3b/bcc", "results/P1/p1_3b/bcc_L{L}.json", 8, 0.876),
    "fcc": ("data/P1/p1_3b/fcc", "results/P1/p1_3b/fcc_L{L}.json", 12, 0.876),
    "sc": ("data/P1/p1_3/3D_bond", "results/P1/p1_3/3D_bond_L{L}.json", 6, 0.876),
    "hc4": ("data/P1/p1_3/4D_bond", "results/P1/p1_3/4D_bond_L{L}.json", 8, 0.6845),
    "hc6": ("data/P1/p1_3/6D_bond", "results/P1/p1_3/6D_bond_L{L}.json", 12, 0.5),
}


def spline_peak(ps, y):
    """argmax інтерполяційного кубічного сплайна середньої кривої (без вибору вікна)."""
    cs = CubicSpline(ps, y)
    i = int(np.argmax(y))
    lo, hi = ps[max(i - 2, 0)], ps[min(i + 2, len(ps) - 1)]
    x = np.linspace(lo, hi, 2001)
    return float(x[np.argmax(cs(x))])


def quad_peak(ps, y, hw, c):
    """вершина параболи у вікні ±hw навколо c (симетрична модель — для оцінки зміщення від асиметрії)."""
    m = np.abs(ps - c) <= hw + 1e-12
    a, b, _ = np.polyfit(ps[m] - c, y[m], 2)
    return float(c - b / (2 * a))


def quartic_peak(ps, y, hw, c):
    m = np.abs(ps - c) <= hw + 1e-12
    co = np.polyfit(ps[m] - c, y[m], 4)
    x = np.linspace(-hw, hw, 4001)
    return float(c + x[np.argmax(np.polyval(co, x))])


B = 300
peaks = {}
for key, (dd, rp, z, nu) in SRC.items():
    rows = []
    for d in sorted((ROOT / dd).glob("L*"), key=lambda p: int(p.name[1:])):
        L = int(d.name[1:])
        rf = ROOT / rp.format(L=L)
        if not rf.exists():
            continue
        ps = np.array(json.load(open(rf))["ps"])
        Y = np.concatenate([np.load(f)["nsc"] for f in sorted(d.glob("chunk_*.npz"))])
        R = len(Y)
        est = lambda Ym: (spline_peak(ps, Ym), quad_peak(ps, Ym, 0.006, spline_peak(ps, Ym)),
                          quartic_peak(ps, Ym, 0.02, spline_peak(ps, Ym)))
        e0 = est(Y.mean(0))
        bs = np.array([est(Y[rng.integers(0, R, R)].mean(0)) for _ in range(B)])
        se = bs.std(0, ddof=1)
        rows.append(dict(L=L, R=R, spline=e0[0], se_spline=float(se[0]), quad006=e0[1], se_quad006=float(se[1]),
                         quartic02=e0[2], se_quartic02=float(se[2])))
        print(f"{key:8s} L={L:4d} R={R:4d} spline={e0[0]:.4f}±{se[0]:.4f} quad±.006={e0[1]:.4f}±{se[1]:.4f} "
              f"quartic±.02={e0[2]:.4f}±{se[2]:.4f}", flush=True)
    out = dict(z=z, nu=nu, rows=rows)
    Ls = np.array([r["L"] for r in rows], float)
    for est in ("spline", "quad006", "quartic02"):
        p = np.array([r[est] for r in rows])
        s = np.array([r["se_" + est] for r in rows])
        f = lambda L, a0, a1: a0 + a1 * L ** (-1 / nu)
        (pinf, a1), cov = curve_fit(f, Ls, p, sigma=s, absolute_sigma=True, p0=[p[-1], 0])
        chi2 = float(np.sum(((p - f(Ls, pinf, a1)) / s) ** 2))
        dof = len(p) - 2
        se_fit = float(np.sqrt(cov[0, 0]) * max(1, np.sqrt(chi2 / max(dof, 1))))   # масштаб Бірге
        w = 1 / s[-2:] ** 2
        plat = float(np.sum(w * p[-2:]) / w.sum())
        chi2p = float(np.sum(w * (p[-2:] - plat) ** 2))
        se_plat = float(np.sqrt(1 / w.sum()) * max(1, np.sqrt(chi2p)))             # масштаб Бірге, 1 ст. св.
        bp = bethe[z]["p_star"]
        out[est] = dict(p_inf=float(pinf), se_p_inf_birge=se_fit, chi2=chi2, dof=dof, plateau=plat,
                        se_plateau_birge=se_plat, chi2_plateau=chi2p, last_minus_prev=float(p[-1] - p[-2]),
                        rel_p_inf=(pinf - bp) / bp, err_rel_p_inf=float(np.hypot(se_fit, abs(plat - pinf)) / bp),
                        rel_plateau=(plat - bp) / bp, err_rel_plateau=se_plat / bp)
        print(f"   {est:9s} p∞={pinf:.4f}±{se_fit:.4f} (χ²={chi2:.1f}/{dof}) плато={plat:.4f}±{se_plat:.4f} "
              f"(χ²={chi2p:.2f}) → δ={100 * out[est]['rel_p_inf']:+.2f}±{100 * out[est]['err_rel_p_inf']:.2f}%, "
              f"плато {100 * out[est]['rel_plateau']:+.2f}±{100 * out[est]['err_rel_plateau']:.2f}%", flush=True)
    peaks[key] = out
res["peaks"] = peaks

# ------------------------------------------------------------ 5. розклад δ = (pc/p*_Б)(1 + s) − 1
PCS = {"diamond": 0.3895892, "bcc": 0.18028762, "fcc": 0.12016377, "sc": 0.24881182, "hc4": 0.16013122,
       "hc6": 0.09420165}
dec = {}
for key, e in peaks.items():
    bp = bethe[e["z"]]["p_star"]
    pinf = e["spline"]["p_inf"]
    dec[key] = dict(pc_over_bethe=PCS[key] / bp, shift_above_pc=pinf / PCS[key] - 1, delta=pinf / bp - 1)
    print("розклад", key, {k: round(v, 4) for k, v in dec[key].items()})
res["decomposition"] = dec

# ------------------------------------------------------------ 6. Λ4 з власних c3, c4
res["Lambda4"] = {k: latt[k]["c3"][0] * bethe[z]["p_star"] ** 3 + latt[k]["c4"][0] * bethe[z]["p_star"] ** 4
                  for k, z in (("diamond", 4), ("bcc", 8), ("fcc", 12), ("sc", 6))}
print("Λ4", res["Lambda4"])

json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print("→", OUT)
