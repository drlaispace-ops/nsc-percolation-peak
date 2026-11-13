"""Журнал за компонентами: один рядок на зв'язну компоненту, контрольні точки шматками.

Рядок компоненти: n, m (ребра), sum_sc = Σ NodeSC за вузлами, delta[k] = Σ_v δ_k(v) (хвіст k > K — в останній
колонці), дерево чи ні за m = n − 1. Міра задається окремо: внесок у NSC за вузлами — sum_sc / N,
усереднення за компонентами — C = sum_sc / n порівну. Поле giant_size означене лише явно: при c ≤ 1 гіганта немає.
"""
import multiprocessing as mp
import os
from pathlib import Path

import numba as nb
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from nsccore import er_graph, to_csr
from nscfeat import bfs_profile


def graph_ledger(indptr, indices, n, K=32, exclude_largest=False):
    """Рядки компонент графа. exclude_largest прибирає найбільшу компоненту до BFS (гігант — дорогий)."""
    A = sp.csr_array((np.ones(len(indices), np.int8), indices, indptr), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab, minlength=ncomp)
    keep = np.ones(n, bool)
    giant_size = 0
    if exclude_largest:
        big = int(np.argmax(sizes))
        giant_size = int(sizes[big])
        keep[lab == big] = False
    idx = np.nonzero(keep)[0]
    S = A[idx][:, idx].tocsr()
    nf = len(idx)
    _, _, D = bfs_profile(S.indptr.astype(np.int64), S.indices.astype(np.int64), nf, K=K, full=True)
    labf = lab[idx]
    # нова нумерація компонент у межах тих, що лишились
    uniq, inv = np.unique(labf, return_inverse=True)
    deg = np.diff(S.indptr)
    n_c = np.bincount(inv, minlength=len(uniq)).astype(np.int64)
    m_c = (np.bincount(inv, weights=deg, minlength=len(uniq)) / 2).astype(np.int64)
    delta = np.zeros((len(uniq), K))
    for k in range(K):
        delta[:, k] = np.bincount(inv, weights=D[:, k], minlength=len(uniq))
    return dict(n=n_c, m=m_c, sum_sc=delta.sum(1), delta=delta, N=np.int64(n), giant_size=np.int64(giant_size))


def er_ledger(n, c, rng, K=32, exclude_largest=None):
    """Журнал ER(n, c/n). За замовчуванням гігант прибирається лише при c > 1."""
    ip, ix, _ = to_csr(er_graph(n, c / n, rng))
    return graph_ledger(ip, ix, n, K=K, exclude_largest=(c > 1) if exclude_largest is None else exclude_largest)


# --- Контрольні точки ---
def _save_atomic(path, data):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, path)


def _run_one(task):
    worker, i, seq, args, path = task
    nb.set_num_threads(1)  # паралелізм лише за процесами
    _save_atomic(Path(path), worker(i, seq, args))
    return i


def run_chunks(worker, args, seed, nchunks, outdir, nproc=1):
    """Запускає worker(i, seedseq, args) → dict масивів для кожного шматка; готові шматки пропускає.

    Сід шматка залежить лише від (seed, i), тому результат не залежить від nproc і від переривань.
    worker має бути функцією верхнього рівня модуля, а виклик із nproc > 1 — під if __name__ == "__main__"."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seqs = np.random.SeedSequence(seed).spawn(nchunks)
    paths = [outdir / f"chunk_{i:05d}.npz" for i in range(nchunks)]
    tasks = [(worker, i, seqs[i], args, str(paths[i])) for i in range(nchunks) if not paths[i].exists()]
    if tasks and nproc > 1:
        with mp.get_context("spawn").Pool(nproc) as pool:
            for _ in pool.imap_unordered(_run_one, tasks):
                pass
    else:
        for task in tasks:
            _run_one(task)
    return paths


def load_chunks(paths):
    return [dict(np.load(p, allow_pickle=False)) for p in paths]
