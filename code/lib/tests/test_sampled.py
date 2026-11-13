"""Тести вибіркового оцінювача NSC (П1.2).
    .venv/bin/python -m pytest code/lib/tests/test_sampled.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

import perc  # noqa: E402
import sampled as sm  # noqa: E402
from nsccore import _node_sc_all, to_csr  # noqa: E402


def _graphs():
    rng = np.random.default_rng(1)
    yield perc.hypercubic_bond(40, 0.5, rng, 2), None
    yield perc.hypercubic_bond(12, 0.26, rng, 3), None
    yield perc.er_graph(3000, 1.4 / 3000, rng), None
    A, occ = perc.hypercubic_site(40, 0.6, rng, 2)
    yield A, occ


@pytest.mark.parametrize("i", range(4))
def test_kernel_matches_core(i):
    A, _ = list(_graphs())[i]
    ip, ix, n = to_csr(A)
    ref = _node_sc_all(ip, ix, n, 8)
    sc, prof, work = sm.node_sc_exact(A, K=8)
    assert np.max(np.abs(sc - ref)) < 1e-12
    assert np.max(np.abs(prof.sum(1) - sc)) < 1e-12
    assert (work > 0).all()


@pytest.mark.parametrize("i", range(4))
def test_exact_limit(i):
    """Якщо великих компонент немає (s0 > n), оцінювач точний і SE = 0; збігається з measure."""
    A, occ = list(_graphs())[i]
    r = sm.nsc_sampled(A, M=50, s0=10 ** 9, rng=np.random.default_rng(0), occ=occ)
    ref = perc.measure(A, occ)
    for a, b in [("nsc", "nsc"), ("nsc_occ", "nsc_occ"), ("P", "P"), ("g_giant", "g_giant"), ("g_fin", "g_fin")]:
        assert r[a] == pytest.approx(ref[b], abs=1e-12)
    assert r["se"] == 0
    assert r["delta"].sum() == pytest.approx(r["nsc"], abs=1e-12)


def test_sampled_unbiased_small():
    """Середнє за перестановками збігається з точним у межах 4 SE середнього."""
    A, _ = list(_graphs())[0]
    ref = perc.measure(A)["nsc"]
    est = [sm.nsc_sampled(A, M=40, s0=50, rng=np.random.default_rng(s))["nsc"] for s in range(200)]
    assert abs(np.mean(est) - ref) < 4 * np.std(est) / np.sqrt(len(est))


def test_coupled_monotone():
    rng = np.random.default_rng(3)
    n, u, v, w, ws = sm.coupled_bonds(10, 2, rng)
    A1, _ = sm.at_p(n, u, v, w, 0.4)
    A2, _ = sm.at_p(n, u, v, w, 0.6)
    assert ((A1 > 0).astype(int) - (A2 > 0).astype(int)).max() <= 0  # A1 ⊂ A2
    n, u, v, w, ws = sm.coupled_bonds(10, 2, rng, kind="site")
    A, occ = sm.at_p(n, u, v, w, 0.5, ws)
    r, c = A.nonzero()
    assert occ[r].all() and occ[c].all()


def test_triangular_degree():
    n, u, v, w, _ = sm.coupled_triangular(6, np.random.default_rng(0))
    A, _ = sm.at_p(n, u, v, w, 1.01)
    ip, ix, _ = sm.csr(A)
    assert (np.diff(ip) == 6).all()


@pytest.mark.parametrize("i", range(4))
def test_chi_and_size_journal(i):
    """χ і журнал скінченних кластерів за розміром збігаються з прямим підрахунком; журнал + гігант = NSC."""
    import scipy.sparse.csgraph as cg
    A, occ = list(_graphs())[i]
    r = sm.nsc_sampled(A, M=50, s0=200, rng=np.random.default_rng(3), occ=occ)
    _, _, Ac = sm.csr(A)
    _, lab = cg.connected_components(Ac, directed=False)
    size = np.bincount(lab)
    o = np.ones(len(lab), bool) if occ is None else occ
    has = np.bincount(lab[o], minlength=len(size)) > 0
    gi = int(np.argmax(np.where(has, size, -1)))
    fin = has & (np.arange(len(size)) != gi)
    assert np.isclose(r["chi"], np.sum(size[fin].astype(float) ** 2) / len(lab))
    assert r["sbin_c"].sum() == fin.sum() and r["sbin_n"].sum() == size[fin].sum()
    b = np.floor(np.log2(size[fin])).astype(int)
    assert np.array_equal(r["sbin_c"][:b.max() + 1], np.bincount(b))
    Ng = r["P"] * r["n"]
    assert np.isclose(r["sbin_sc"].sum() + r["g_giant"] * Ng, r["nsc"] * r["n"])
    assert np.isclose(r["sbin_sc"].sum() / max(r["sbin_n"].sum(), 1), r["g_fin"]) or r["sbin_n"].sum() == 0
