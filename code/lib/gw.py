"""Бібліотечна версія розрахунку NSC у границі n -> inf для дерев Гальтона–Ватсона."""
import numpy as np
SMAX = 1e9
def nsc_inf(root, child, M, rng=None):
    Z = root(M).astype(np.float64); S = 1.0 + Z
    F = np.zeros(M)
    idx = np.nonzero(Z > 0)[0]; Zc, Sc = Z[idx], S[idx]
    while len(idx):
        F[idx] += 2 * (Sc - 2) / Sc ** 2
        Zn = child(Zc).astype(np.float64); Sc = Sc + Zn
        keep = (Zn > 0) & (Sc < SMAX)
        idx, Zc, Sc = idx[keep], Zn[keep], Sc[keep]
    return dict(nsc=float(F.mean()), se=float(F.std() / np.sqrt(M)))
