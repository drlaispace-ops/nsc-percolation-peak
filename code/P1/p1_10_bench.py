"""П1.10. Час NodeSC на критичних деревах: BFS `_node_sc_all` проти переукорінення."""
import sys
import time
from pathlib import Path

import numba as nb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import gwtree as gt  # noqa: E402
from nsccore import _node_sc_all  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

SEED = 20261106
rows = []
threads = 4
nb.set_num_threads(threads)
seeds = gt.seeds_from(np.random.SeedSequence(SEED), 10)
gt.node_sc_tree(gt.prufer_parent(50, 1))
_node_sc_all(*gt.csr_of_parent(gt.prufer_parent(50, 1)), 50, threads)
for i, n in enumerate([1000, 3000, 10000, 30000, 100000, 300000, 1000000]):
    p = gt.prufer_parent(n, int(seeds[i]))
    H = int(gt.tree_height(p, n))
    t0 = time.perf_counter()
    r = gt.node_sc_tree(p, K=8)[0]
    t1 = time.perf_counter()
    row = dict(n=n, height=H, t_reroot=t1 - t0, t_bfs=None, max_abs_diff=None)
    if n <= 100000:
        ip, ix = gt.csr_of_parent(p)
        t0 = time.perf_counter()
        b = _node_sc_all(ip, ix, n, threads)
        row["t_bfs"] = time.perf_counter() - t0
        row["max_abs_diff"] = float(np.abs(b - r).max())
    rows.append(row)
    print(row, flush=True)
save(ROOT / "results" / "P1" / "p1_10_bench.json", runmeta(__file__, dict(threads=threads, tree="рівномірне мічене"), dict(entropy=SEED)), rows=rows)
