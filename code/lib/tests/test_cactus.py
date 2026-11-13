"""Тести дерев клік (П1.3c): cactus (точні pc, P, МК) і cactusexact (квадратура NSC_∞)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cactus as ca  # noqa: E402
import cactusexact as ce  # noqa: E402
import gwexact as gx  # noqa: E402


@pytest.mark.parametrize("m", [2, 3, 5, 8, 12])
def test_comp_pmf_normalised(m):
    for p in (0.05, 0.3, 0.9):
        assert abs(ca.comp_pmf(m, p).sum() - 1) < 1e-12


def test_comp_pmf_triangle():
    p, q = 0.3, 0.7
    pm = ca.comp_pmf(3, p)
    assert np.allclose(pm, [q * q, 2 * p * q * q, 1 - q * q - 2 * p * q * q])


@pytest.mark.parametrize("z", [3, 4, 12])
def test_pc_bethe(z):
    assert abs(ca.pc((2,) * z) - 1 / (z - 1)) < 1e-12


def test_pc_husimi_triangles():
    # (k − 1)·E[j − 1] = 1, E[j − 1] = 2(p + p² − p³) для трикутника
    want = brentq(lambda p: 5 * 2 * (p + p * p - p ** 3) - 1, 0, 1)
    assert abs(ca.pc((3,) * 6) - want) < 1e-12


@pytest.mark.parametrize("p", [0.1, 0.12, 0.2])
def test_branching_matches_rrg(p):
    assert abs(ca.branching((2,) * 12, p)[1] - gx.giant(p, "z12")) < 1e-10


def test_structure():
    assert ca.structure((3,) * 6) == dict(z=12, t=6, sizes=[3] * 6)
    assert ca.structure((8,) + (2,) * 5)["t"] == 21


@pytest.mark.parametrize("z,p", [(3, 0.4), (3, 0.7), (4, 0.3), (12, 0.08), (12, 0.12)])
def test_exact_reduces_to_gwexact(z, p):
    assert abs(ce.nsc((2,) * z, p) - gx.nsc_quad(f"z{z}", p)) < 1e-10


@pytest.mark.parametrize("m", [2, 3, 4, 5, 6])
def test_layers_match_enumeration(m):
    for p in (0.1, 0.45):
        r1, j1, s1 = ce._rtables(m, p, 7)
        r2, j2, s2 = ce._rtables_enum(m, p, 7)
        for r in range(m - 1):
            d1 = {tuple(x): y for x, y in zip(r1[s1[r]:s1[r + 1]], j1[s1[r]:s1[r + 1]])}
            d2 = {tuple(x): y for x, y in zip(r2[s2[r]:s2[r + 1]], j2[s2[r]:s2[r + 1]])}
            assert d1.keys() == d2.keys()
            for k in d1:
                assert np.allclose(d1[k], d2[k], atol=1e-13)


def test_mc_matches_exact_small():
    """МК на дереві трикутників (z = 4) проти квадратури: нижче і вище порога."""
    sizes = (3, 3)
    pc = ca.pc(sizes)
    ps = np.array([0.7 * pc, 1.4 * pc])
    sc, tail, cap, _ = ca.sample(sizes, ps, 20000, seed=5, scap=3000)
    tot = sc + np.nan_to_num(tail)
    for j, p in enumerate(ps):
        se = tot[:, j].std() / np.sqrt(len(tot))
        assert abs(tot[:, j].mean() - ce.nsc(sizes, p)) < 4 * se
        P = ca.branching(sizes, p)[1]
        assert abs(cap[:, j].mean() - P) < 4 * np.sqrt(P * (1 - P) / len(tot)) + 1e-3


def test_mc_deterministic():
    a = ca.sample((3, 3), [0.5, 0.6], 200, seed=9)[0]
    b = ca.sample((3, 3), [0.5, 0.6], 200, seed=9)[0]
    assert np.array_equal(a, b)
