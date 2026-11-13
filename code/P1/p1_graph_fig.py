"""Ілюстрація для звіту: граф ER у чотирьох станах і розподіл розмірів кластерів (вибірка проти точного закону Бореля)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import gwexact as gx  # noqa: E402
from nsccore import er_graph, to_csr  # noqa: E402

OUT = ROOT / "results" / "P1"
SEED = 202611072
ORANGE, GREEN, GREY = "#eb6834", "#1baf7a", "#b9b7b1"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
STATES = [(0.8, "c = 0.8: лише малі кластери"), (1.0, "c = 1: поріг"), (1.40381, "c = 1.40: пік складності"), (2.0, "c = 2: гігант домінує")]
N_DRAW, N_DIST = 1500, 200_000


def graph(n, c, rng):
    ip, ix, _ = to_csr(er_graph(n, c / n, rng))
    A = sp.csr_array((np.ones(len(ix), np.int8), ix, ip), shape=(n, n))
    return A, *connected_components(A, directed=False)


def layout(A, ncomp, lab, seed):
    """Кожна компонента окремо, далі щільне пакування по спіралі: найбільша в центрі."""
    n = A.shape[0]
    size = np.bincount(lab, minlength=ncomp)
    pos = np.zeros((n, 2))
    rng = np.random.default_rng(seed)
    centres, radii, t = np.zeros((0, 2)), np.zeros(0), 0.0
    for c in np.argsort(-size):
        idx = np.nonzero(lab == c)[0]
        s = len(idx)
        if s == 1:
            xy, r = np.zeros((1, 2)), 0.5
        else:
            G = nx.from_scipy_sparse_array(A[idx][:, idx])
            if s > 60:
                p = nx.spring_layout(G, seed=int(rng.integers(2 ** 31)), iterations=300, k=1.3 / np.sqrt(s))
            else:
                p = nx.kamada_kawai_layout(G)
            xy = np.array([p[i] for i in range(s)])
            xy -= xy.mean(0)
            r = 0.95 * np.sqrt(s) + 0.5
            xy *= r / max(np.linalg.norm(xy, axis=1).max(), 1e-9)
        while True:  # компоненти йдуть за спаданням розміру, тому спіраль не треба починати спочатку
            q = 0.84 * np.sqrt(t + 1e-9) * np.array([np.cos(t), np.sin(t)])
            k = len(centres)
            if k == 0 or (np.hypot(*(centres[:k] - q).T) > radii[:k] + r + 0.25).all():
                break
            t += 0.12
        centres = np.vstack([centres, q]) if len(centres) else q[None, :]
        radii = np.r_[radii, r]
        pos[idx] = xy + q
    return pos


def log_bins(N):
    return np.unique(np.round(np.logspace(0, np.log10(N), 26)).astype(int))


fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), gridspec_kw=dict(height_ratios=[1.45, 1]), constrained_layout=True)
ss = np.random.SeedSequence(SEED).spawn(2 * len(STATES))
for j, (c, ttl) in enumerate(STATES):
    A, nc, lab = graph(N_DRAW, c, np.random.default_rng(ss[2 * j]))
    size = np.bincount(lab, minlength=nc)
    big = int(np.argmax(size))
    has_giant = c > 1
    pos = layout(A, nc, lab, int(ss[2 * j].generate_state(1)[0]))
    node_col = np.array([GREY if size[l] <= 2 else ORANGE for l in lab], dtype=object)
    if has_giant:
        node_col[lab == big] = GREEN
    coo = sp.triu(A).tocoo()
    seg = np.stack([pos[coo.row], pos[coo.col]], axis=1)
    ax = axes[0, j]
    ax.add_collection(LineCollection(seg, colors=[GREY], linewidths=0.45, zorder=1))
    d = np.diff(A.indptr)
    ax.scatter(pos[:, 0], pos[:, 1], s=5 + 2.0 * d, c=list(node_col), linewidths=0, zorder=2)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    ax.set_title(ttl, loc="left", color=INK, fontsize=12.5)
    frac = size[big] / N_DRAW
    ax.text(0.0, -0.01, (f"гігант: {frac:.0%} вузлів" if has_giant else f"найбільший кластер: {size[big]} із {N_DRAW}"), transform=ax.transAxes, color=MUTED, fontsize=10.5, va="top")

    # --- Розподіл розмірів ---
    A2, nc2, lab2 = graph(N_DIST, c, np.random.default_rng(ss[2 * j + 1]))
    sz = np.bincount(lab2, minlength=nc2)
    b = log_bins(N_DIST)
    mass, _ = np.histogram(sz, bins=np.r_[b, N_DIST + 1], weights=sz / N_DIST)
    if has_giant:
        g2 = int(np.argmax(sz))
        rest = np.delete(sz, g2)
        mass, _ = np.histogram(rest, bins=np.r_[b, N_DIST + 1], weights=rest / N_DIST)
    bor = gx.borel(N_DIST, c if c <= 1 else c)
    cum = np.cumsum(bor)
    theo = np.diff(np.r_[cum[b - 1], cum[-1]])
    edges = np.r_[b, N_DIST + 1]
    centres = np.sqrt(b * edges[1:])
    width = np.diff(edges)  # на один розмір: нерівні цілі комірки інакше дають зубець при малих s
    mass, theo = mass / width, theo / width
    ax = axes[1, j]
    ax.grid(True, color=GRID, lw=1)
    keep = theo > 1e-10
    ax.loglog(centres[keep], theo[keep], color=INK, lw=1.5, label="точна формула")
    ok = mass > 0
    ax.plot(centres[ok], mass[ok], "o", color=ORANGE, ms=6, mec=SURF, mew=1.0, label="малі кластери, вибірка")
    if has_giant:
        P = sz[g2] / N_DIST
        ax.plot([sz[g2]], [P], "D", color=GREEN, ms=11, mec=SURF, mew=1.2, label="гігант")
        ax.annotate(f"гігант: {P:.0%} вузлів", (sz[g2], P), xytext=(-10, -4), textcoords="offset points", ha="right", color=INK, fontsize=10.5)
    ax.set(xlim=(0.8, 4e5), ylim=(1e-9, 2), xlabel="розмір кластера s")
    if j == 0:
        ax.set_ylabel("частка вузлів\nу кластерах розміру s")
        ax.legend(frameon=False, fontsize=10, loc="upper right")
fig.savefig(OUT / "fig_gw_graphs.png", dpi=170)
