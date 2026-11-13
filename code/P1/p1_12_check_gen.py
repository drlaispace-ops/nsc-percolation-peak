"""П1.12. Чи має умовний генератор зсув: велика вибірка проти точної таблиці a_n і незалежний відбір (вирощування без умови).

Привід: z = 3, n = 30 — корінь нижчий за таблицю у двох заходах (z = −2.40 при R = 2000, −2.11 при R = 20000).
Запуск: .venv/bin/python code/P1/p1_12_check_gen.py
"""
import sys
from pathlib import Path

import numba as nb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import gwexact as gx  # noqa: E402
import gwtree as gt  # noqa: E402
from runmeta import runmeta, save  # noqa: E402

SEED = 202611063
POINTS = [("z3", 30), ("z3", 100), ("poisson", 100), ("z4", 30)]
LAWS = {"poisson": ((gt.POISSON, 1.0, 0), (gt.POISSON, 1.0, 0)), "z3": gx.rrg_laws(3), "z4": gx.rrg_laws(4)}
R_COND = 400_000
R_REJ = 100_000


@nb.njit(cache=True)
def _cond_run(n, seeds, cdf, mode, b, g):
    acc = np.zeros(1)
    rt, ty = np.zeros(seeds.size), np.zeros(seeds.size)
    for i in range(seeds.size):
        sc = gt.tree_sc_reroot(gt.cond_law_parent(n, seeds[i], cdf, mode, b), n, g, 1, acc)
        rt[i], ty[i] = sc[0], sc.mean()
    return rt, ty


@nb.njit(cache=True)
def _rej_run(n, seed, R, mr, ar, br, mc, ac, bc, g):
    np.random.seed(seed)
    parent = np.zeros(n + 1, np.int64)
    acc = np.zeros(1)
    rt, ty = np.zeros(R), np.zeros(R)
    i = tries = 0
    while i < R:
        tries += 1
        if gt.gw_parent(mr, ar, br, mc, ac, bc, n, parent) == n:
            sc = gt.tree_sc_reroot(parent, n, g, 1, acc)
            rt[i], ty[i] = sc[0], sc.mean()
            i += 1
    return rt, ty, tries


def row(rt, ty, a):
    R = rt.size
    se_r, se_t = rt.std() / np.sqrt(R), ty.std() / np.sqrt(R)
    return dict(R=R, root=float(rt.mean()), root_se=float(se_r), z_root=float((rt.mean() - a) / se_r),
                typ=float(ty.mean()), typ_se=float(se_t), z_typ=float((ty.mean() - a) / se_t))


if __name__ == "__main__":
    res = {}
    seqs = np.random.SeedSequence(SEED).spawn(len(POINTS))
    for (name, n), seq in zip(POINTS, seqs):
        root, child = LAWS[name]
        a = float(gx.a_table_law(n, root, child)[n])
        g = gt.gtable(n)
        s1, s2 = seq.spawn(2)
        rc, tc = _cond_run(n, gt.seeds_from(s1, R_COND), gx.root_given_n(n, root, child), child[0], int(child[2]), g)
        rr, tr, tries = _rej_run(n, int(s2.generate_state(1)[0]), R_REJ, root[0], float(root[1]), int(root[2]), child[0], float(child[1]), int(child[2]), g)
        res[f"{name}_{n}"] = dict(table=a, cond=row(rc, tc, a), rejection=row(rr, tr, a), tries=int(tries))
        print(name, n, a, res[f"{name}_{n}"]["cond"], res[f"{name}_{n}"]["rejection"], flush=True)
    meta = runmeta(__file__, dict(POINTS=POINTS, R_COND=R_COND, R_REJ=R_REJ), dict(entropy=SEED))
    print(save(ROOT / "results" / "P1" / "p1_12_check_gen.json", meta, **res))
