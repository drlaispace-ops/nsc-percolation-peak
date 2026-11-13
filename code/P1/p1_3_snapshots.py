"""П1.3. Знімки NodeSC: 2D bond-квадратна (L = 200) і центральний зріз 3D bond (L = 64), періодичні межі.

Одна спряжена реалізація на решітку: три значення p (нижче порога, пік p*(L) з П1.3, вище порога).
NodeSC — точна (усі джерела у 2D; у 3D — лише вузли зрізу як джерела, BFS по всій решітці).
Вузли поза найбільшим кластером показано блідо. Запуск: .venv/bin/python code/P1/p1_3_snapshots.py → results/P1/fig_p1_3_snapshots.png

Режим для статті 1 (рис. 1): .venv/bin/python code/P1/p1_3_snapshots.py paper → results/P1/p1_3_snapshots_paper.{npz,json}
  2D bond-квадратна, L = 256 (розмір із FSS П1.3), одна спряжена реалізація при p = 0.46, p*(256) з
  results/P1/p1_3_analysis.json і 0.54; зберігаються масиви NodeSC і маска найбільшого кластера (рисунок — paper/P1/make_figures.py).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import sampled as sm  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

SEED = 20261112
CASES = [  # (назва, d, L, pc, значення p)
    ("2D квадратна, L = 200", 2, 200, 0.5, (0.46, 0.496, 0.54)),
    ("3D, L = 64, зріз z = 32", 3, 64, 0.248812, (0.23, 0.2625, 0.30)),
]
INK, MUTED = "#0b0b0b", "#52514e"
CMAP, NORM, FADE = plt.get_cmap("Blues"), matplotlib.colors.Normalize(0, 2.0), 0.3


def snap(d, L, p, u, v, w, n):
    A, _ = sm.at_p(n, u, v, w, p)
    ip, ix, A = sm.csr(A)
    nc, lab = connected_components(A, directed=False)
    big = np.bincount(lab).argmax()
    if d == 2:
        src = np.arange(n, dtype=np.int64)
    else:
        src = (np.arange(L * L, dtype=np.int64) + (L // 2) * L * L)  # індекс = z·L² + y·L + x, зріз z = L/2
    sc, _, _ = sm._sources_sc(ip, ix, src, 64, sm.nb.get_num_threads())
    shape = (L, L)
    return sc.reshape(shape), (lab[src] == big).reshape(shape), float(np.mean(lab == big))


def main():
    rng = np.random.default_rng(SEED)
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 9.2), constrained_layout=True)
    out = []
    for row, (title, d, L, pc, ps) in enumerate(CASES):
        n, u, v, w, _ = sm.coupled_bonds(L, d, rng)
        for col, p in enumerate(ps):
            sc, giant, P = snap(d, L, p, u, v, w, n)
            ax = axs[row, col]
            rgb = CMAP(NORM(sc))[..., :3]
            rgb[~giant] = 1 - FADE * (1 - rgb[~giant])          # поза найбільшим кластером — блідо
            ax.imshow(rgb, interpolation="nearest", origin="lower")
            lab = ("нижче порога", "пік NSC", "вище порога")[col]
            what = "NSC зрізу" if d == 3 else "NSC"
            ax.set_title(f"{title}\np = {p} ({lab}), P = {P:.2f}, {what} = {sc.mean():.3f}", fontsize=9.5,
                         color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            out.append(dict(d=d, L=L, p=p, P=P, mean_sc_shown=float(sc.mean()),
                            mean_sc_giant=float(sc[giant].mean()) if giant.any() else None,
                            mean_sc_rest=float(sc[~giant].mean())))
            print(out[-1], flush=True)
    im = matplotlib.cm.ScalarMappable(NORM, CMAP)
    cb = fig.colorbar(im, ax=axs, shrink=0.6, label="NodeSC (0 — ізольований вузол)")
    cb.outline.set_visible(False)
    fig.suptitle("NodeSC на решітці: насичені кольори — найбільший кластер, бліді — решта вузлів", color=INK)
    fig.savefig(ROOT / "results" / "P1" / "fig_p1_3_snapshots.png", dpi=150)
    save(ROOT / "results" / "P1" / "p1_3_snapshots.json", runmeta(__file__, params=dict(cases=CASES), seeds=[SEED]),
         panels=out)


def main_paper():
    import json
    an = json.loads((ROOT / "results" / "P1" / "p1_3_analysis.json").read_text())["lattices"]["2D bond sq"]
    L = 256
    pstar = next(r for r in an["rows"] if r["L"] == L)["all"]["p_star"]
    ps = (0.46, round(pstar, 4), 0.54)
    rng = np.random.default_rng(SEED)
    n, u, v, w, _ = sm.coupled_bonds(L, 2, rng)
    arrs, panels = {}, []
    for i, p in enumerate(ps):
        sc, giant, P = snap(2, L, p, u, v, w, n)
        arrs[f"sc{i}"], arrs[f"giant{i}"] = sc.astype(np.float32), giant
        panels.append(dict(p=p, P=P, nsc=float(sc.mean()), g_giant=float(sc[giant].mean()),
                           g_rest=float(sc[~giant].mean()), g_rest_nonisolated=float(sc[(~giant) & (sc > 0)].mean()),
                           max_sc=float(sc.max())))
        print(panels[-1], flush=True)
    np.savez_compressed(ROOT / "results" / "P1" / "p1_3_snapshots_paper.npz", ps=np.array(ps), **arrs)
    save(ROOT / "results" / "P1" / "p1_3_snapshots_paper.json",
         runmeta(__file__, params=dict(d=2, L=L, ps=list(ps), pstar_source="p1_3_analysis.json: lattices['2D bond sq']"
                                       ".rows[L=256].all.p_star", npz="results/P1/p1_3_snapshots_paper.npz"),
                 seeds=[SEED]), panels=panels)


if __name__ == "__main__":
    main_paper() if sys.argv[1:] == ["paper"] else main()
