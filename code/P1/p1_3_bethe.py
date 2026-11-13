"""П1.3. Еталон середнього поля для решіток Z^d: гратка Бете (RRG, n → ∞) з координацією z = 2d, bond-перколяція.

Точна квадратура gwexact (NSC_∞ спирається на лему про хвіст, П1.16): пік p*, частка гіганта P* = P(p*),
точка P = 1/2, поріг Бете 1/(z − 1). Порівняння з решітками — results/P1/p1_3_analysis.json.
Запуск: .venv/bin/python code/P1/p1_3_bethe.py  →  results/P1/p1_3_bethe.json
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
import gwexact as gx  # noqa: E402
from runmeta import runmeta, save  # noqa: E402


def main():
    rows = []
    for d in range(2, 7):
        z = 2 * d
        m = f"z{z}"
        pc = 1 / (z - 1)
        # пік: шукаємо між порогом і точкою P = 0.9 (NSC_∞ гладка вище порога)
        p_half = brentq(lambda p: gx.giant(p, m) - 0.5, pc * (1 + 1e-6), 1 - 1e-9, xtol=1e-12)
        grid = np.linspace(pc * 1.01, min(3 * p_half - 2 * pc, 0.99), 41)
        y = np.array([gx.nsc_quad(m, p) for p in grid])
        i = int(np.argmax(y))
        ps = gx.peak(m, grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)])
        # контроль кроку квадратури: h = 0.005
        ps_h = brentq(lambda x: (gx.nsc_quad(m, x + 1e-4, 0.005) - gx.nsc_quad(m, x - 1e-4, 0.005)) / 2e-4,
                      grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)], xtol=1e-7)
        r = dict(d=d, z=z, pc=pc, p_star=ps, p_star_h005=ps_h, rel_shift=(ps - pc) / pc, P_star=gx.giant(ps, m),
                 nsc_star=gx.nsc_quad(m, ps), p_half=p_half, rel_half=(p_half - pc) / pc)
        rows.append(r)
        print(f"d={d} z={z:2d}  pc={pc:.5f}  p*={ps:.5f} (h=0.005: {ps_h:.5f})  (p*−pc)/pc={r['rel_shift']:+.3f}"
              f"  P*={r['P_star']:.4f}  p½={p_half:.5f} ({r['rel_half']:+.3f})  NSC*={r['nsc_star']:.4f}", flush=True)
    save(ROOT / "results" / "P1" / "p1_3_bethe.json", runmeta(__file__, params=dict(method="gwexact.nsc_quad, h=0.01")),
         rows=rows)


if __name__ == "__main__":
    main()
