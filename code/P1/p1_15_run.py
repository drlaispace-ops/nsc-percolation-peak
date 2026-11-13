"""П1.15, крок 2. Покластерні ознаки скінченних кластерів на решітках (новий невеликий розрахунок).

Для 2D bond-квадратної (L = 256) і 3D bond-кубічної (L = 48), періодичні межі, p ∈ {0.9 pc, pc, 1.1 pc}
(спряжена реалізація: одне рівномірне число на зв'язок, як у П1.3), для кожного кластера з s ≥ 3:
  s, E (ребра), цикломатичне число c = E − s + 1, точна середня NodeSC кластера (BFS з кожного вузла),
  гістограма степенів (1..2d); хімічний діаметр, середній ексцентриситет і середня хімічна відстань ⟨ℓ⟩; для s ≥ S_PROF — середній профіль δ_k (K = 64,
  хвіст k > K — в останньому стовпчику, як у sampled._sources_sc).
Найбільший кластер не рахується при жодному p (лише його розмір s_big) — як у журналі П1.3, де він вилучений
як «гігант»; при pc він же й найдорожчий (~90% часу).
Кластери з s ≤ 2 мають NodeSC = 0 і не зберігаються (лише їхня кількість).
Контрольні точки: data/P1/p1_15/<решітка>_L<L>/chunk_*.npz (b реалізацій на шматок; готові пропускаються).
Сід реалізації j шматка i: SeedSequence([SEED_ROOT, номер задачі, L], spawn_key=(i,)).spawn(b)[j].
Запуск: NUMBA_NUM_THREADS=3 OMP_NUM_THREADS=1 .venv/bin/python code/P1/p1_15_run.py [--max-min 8]
(--max-min — зупинитися після стількох хвилин; повторний виклик продовжує з контрольних точок).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numba as nb  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse.csgraph as cg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import sampled as sm  # noqa: E402
from runmeta import runmeta  # noqa: E402

SEED_ROOT = 20261115
TASKS = [dict(name="2D bond sq", d=2, L=256, pc=0.5), dict(name="3D bond", d=3, L=48, pc=0.24881182)]
FACT = (0.9, 1.0, 1.1)
R, B, K, S_PROF = 1000, 50, 64, 32
DATA = ROOT / "data" / "P1" / "p1_15"


def slug(t):
    return f"{t['name'].replace(' ', '_')}_L{t['L']}"


@nb.njit(parallel=True, cache=True)
def _ecc_dist(indptr, indices, sources, nthreads):
    """Ексцентриситет і сума хімічних відстаней до всіх вузлів компоненти для кожного джерела (BFS)."""
    m = len(sources)
    n = len(indptr) - 1
    ecc = np.zeros(m, np.int32)
    dsum = np.zeros(m, np.int64)
    for t in nb.prange(nthreads):
        dist = np.full(n, -1, dtype=np.int32)
        q = np.empty(n, dtype=np.int32)
        for j in range(t, m, nthreads):
            src = sources[j]
            dist[src] = 0
            q[0] = src
            head, tail = 0, 1
            tot = 0
            while head < tail:
                a = q[head]
                head += 1
                da = dist[a]
                tot += da
                for e in range(indptr[a], indptr[a + 1]):
                    v = indices[e]
                    if dist[v] == -1:
                        dist[v] = da + 1
                        q[tail] = v
                        tail += 1
            ecc[j] = dist[q[tail - 1]]
            dsum[j] = tot
            for a in range(tail):
                dist[q[a]] = -1
    return ecc, dsum


def clusters_at(n, u, v, w, p, d):
    """Покластерні ознаки при p (див. докстрінг модуля)."""
    A, _ = sm.at_p(n, u, v, w, p)
    ip, ix, A = sm.csr(A)
    deg = np.diff(ip)
    ncomp, lab = cg.connected_components(A, directed=False)
    size = np.bincount(lab, minlength=ncomp)
    E = np.bincount(lab, deg, minlength=ncomp) / 2
    big = int(np.argmax(size))
    keep = size >= 3
    keep[big] = False
    nodes = np.flatnonzero(keep[lab]).astype(np.int64)
    sc, prof, _ = sm._sources_sc(ip, ix, nodes, K, nb.get_num_threads())
    ids = np.flatnonzero(keep)
    remap = np.full(ncomp, -1, np.int64)
    remap[ids] = np.arange(len(ids))
    li = remap[lab[nodes]]
    s = size[ids]
    g = np.bincount(li, sc, len(ids)) / s
    ecc, dsum = _ecc_dist(ip, ix, nodes, nb.get_num_threads())
    diam = np.zeros(len(ids), np.int64)
    np.maximum.at(diam, li, ecc)
    ecc_mean = np.bincount(li, ecc, len(ids)) / s
    lmean = np.bincount(li, dsum.astype(float), len(ids)) / (s * (s - 1.0))
    dh = np.zeros((len(ids), 2 * d), np.int32)
    np.add.at(dh, (li, np.minimum(deg[nodes], 2 * d) - 1), 1)
    pr_ids = np.flatnonzero(s >= S_PROF)
    sel = np.isin(li, pr_ids)
    pm = np.full(len(ids), -1, np.int64)
    pm[pr_ids] = np.arange(len(pr_ids))
    P = np.zeros((len(pr_ids), K))
    np.add.at(P, pm[li[sel]], prof[sel])
    P /= s[pr_ids][:, None]
    return dict(s=s.astype(np.int64), E=E[ids].astype(np.int64), c=(E[ids] - s + 1).astype(np.int64), g=g, deg=dh, diam=diam, ecc_mean=ecc_mean, lmean=lmean, prof_idx=pr_ids.astype(np.int64), prof=P.astype(np.float32),
                n_small=int((size <= 2).sum()), s_big=int(size[big]))


def run_chunk(ti, t, i):
    seq = np.random.SeedSequence([SEED_ROOT, ti, t["L"]], spawn_key=(i,))
    out = {}
    sec = []
    for j, child in enumerate(seq.spawn(B)):
        t0 = time.perf_counter()
        rng = np.random.default_rng(child)
        n, u, v, w, _ = sm.coupled_bonds(t["L"], t["d"], rng)
        for f in FACT:
            r = clusters_at(n, u, v, w, f * t["pc"], t["d"])
            for k, x in r.items():
                out.setdefault(f"{f}_{k}", []).append(x)
        sec.append(time.perf_counter() - t0)
    data = {}
    for k, xs in out.items():
        if np.ndim(xs[0]) == 0:
            data[k] = np.array(xs)
        else:   # конкатенація з індексом реалізації
            data[k] = np.concatenate(xs)
            if k.endswith("_s"):
                data[k[:-2] + "_rid"] = np.concatenate([np.full(len(x), i * B + jj) for jj, x in enumerate(xs)])
            if k.endswith("_prof_idx"):   # зсув індексу профілю на загальну нумерацію кластерів шматка
                lens = np.cumsum([0] + [len(x) for x in out[k[:-9] + "_s"]])[:-1]
                data[k] = np.concatenate([x + o for x, o in zip(xs, lens)])
    data["sec"] = np.array(sec)
    data["ridx"] = np.arange(i * B, (i + 1) * B)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-min", type=float, default=8.0)
    a = ap.parse_args()
    t_start = time.time()
    for ti, t in enumerate(TASKS):
        od = DATA / slug(t)
        od.mkdir(parents=True, exist_ok=True)
        mp = od / "meta.json"
        if not mp.exists():
            mp.write_text(json.dumps(runmeta(__file__, params=dict(task=t, fact=FACT, R=R, B=B, K=K, S_PROF=S_PROF),
                                             seeds=dict(root=SEED_ROOT, scheme="SeedSequence([root, task, L], "
                                                        "spawn_key=(chunk,)).spawn(B)[j]", task=ti)),
                                     ensure_ascii=False, indent=1, default=str))
        for i in range(R // B):
            path = od / f"chunk_{i:04d}.npz"
            if path.exists():
                continue
            if (time.time() - t_start) / 60 > a.max_min:
                print("ліміт часу: продовжити повторним запуском", flush=True)
                return
            t0 = time.time()
            data = run_chunk(ti, t, i)
            tmp = od / f"tmp_{i:04d}.npz"
            np.savez(tmp, **data)
            os.replace(tmp, path)
            print(f"[{time.strftime('%H:%M')}] {t['name']} L={t['L']} шматок {i + 1}/{R // B}: {time.time() - t0:.0f} с",
                  flush=True)
    print("усе пораховано", flush=True)


if __name__ == "__main__":
    main()
