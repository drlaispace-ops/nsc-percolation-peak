"""Статистика для П1: бутстреп, положення піку, τ_int, бінінг.

Конвенції (CLAUDE.md):
  * положення піку — нуль похідної локальної кубічної апроксимації, похибка — бутстреп
    за реалізаціями (не argmax на сітці);
  * похибки — бутстреп за реалізаціями; для Монте-Карло — бінінг і τ_int.
"""
import numpy as np


def bootstrap(x, stat=np.mean, B=2000, rng=None, axis=0):
    """Бутстреп статистики за першою віссю x. Повертає (оцінка, SE, (2.5%, 97.5%))."""
    rng = np.random.default_rng(rng)
    x = np.asarray(x)
    R = x.shape[axis]
    bs = np.array([stat(np.take(x, rng.integers(0, R, R), axis=axis)) for _ in range(B)])
    return stat(x), bs.std(ddof=1, axis=0), tuple(np.percentile(bs, [2.5, 97.5], axis=0))


def cubic_peak(xs, y, window=None, halfwidth=None):
    """Максимум локальної кубічної апроксимації y(x) навколо argmax.

    window — кількість точок сітки з кожного боку від argmax (None — 1/4 сітки, мінімум 3);
    halfwidth — замість window: усі точки з |x − x_argmax| ≤ halfwidth (для нерівномірних сіток; xs зростає).
    Повертає (x*, y*) або (nan, nan), якщо у вікні немає максимуму кубіки.
    """
    xs, y = np.asarray(xs, float), np.asarray(y, float)
    i = int(np.argmax(y))
    if halfwidth is not None:
        lo = int(np.searchsorted(xs, xs[i] - halfwidth * (1 + 1e-9)))
        hi = int(np.searchsorted(xs, xs[i] + halfwidth * (1 + 1e-9), side="right"))
    else:
        w = max(3, len(xs) // 4) if window is None else window
        lo, hi = max(0, i - w), min(len(xs), i + w + 1)
    if hi - lo < 5:
        return np.nan, np.nan
    xc = xs[lo:hi].mean()  # центрування для обумовленості
    c = np.polyfit(xs[lo:hi] - xc, y[lo:hi], 3)
    roots = np.roots(np.polyder(c))
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= xs[lo] - xc) & (roots <= xs[hi - 1] - xc) & (np.polyval(np.polyder(c, 2), roots) < 0)]
    if not len(roots):
        return np.nan, np.nan
    r = roots[np.argmax(np.polyval(c, roots))]
    return float(r + xc), float(np.polyval(c, r))


def peak_bootstrap(xs, Y, window=None, B=1000, rng=None, halfwidth=None):
    """Положення піку середньої кривої з бутстрепом за реалізаціями.

    Y — масив (R, len(xs)): криві окремих реалізацій (спряжені реалізації — один рядок на всю сітку).
    Повертає dict(x, y, se_x, se_y, ci_x, fail): fail — частка бутстреп-вибірок без максимуму.
    """
    rng = np.random.default_rng(rng)
    Y = np.asarray(Y, float)
    R = len(Y)
    x0, y0 = cubic_peak(xs, Y.mean(0), window, halfwidth)
    bs = np.array([cubic_peak(xs, Y[rng.integers(0, R, R)].mean(0), window, halfwidth) for _ in range(B)])
    ok = np.isfinite(bs[:, 0])
    return dict(x=x0, y=y0, se_x=float(bs[ok, 0].std(ddof=1)), se_y=float(bs[ok, 1].std(ddof=1)),
                ci_x=tuple(np.percentile(bs[ok, 0], [2.5, 97.5])), fail=float(1 - ok.mean()))


def autocorr(x):
    """Нормована автокореляційна функція ρ(t), t = 0..N−1 (через FFT)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    return a / a[0]


def tau_int(x, c=6.0):
    """Інтегрований час автокореляції τ_int = 1/2 + Σ_{t≥1} ρ(t) з автоматичним вікном Сокала
    (найменше W ≥ c·τ_int(W)). Повертає (τ_int, W). Похибка середнього: σ·sqrt(2τ_int/N)."""
    rho = autocorr(x)
    tau = 0.5
    for W in range(1, len(rho)):
        tau += rho[W]
        if W >= c * tau:
            return float(tau), W
    return float(tau), len(rho) - 1


def binning(x, min_blocks=32):
    """Похибка середнього методом блоків (Flyvbjerg–Petersen): блоки 1, 2, 4, …
    Повертає (SE на плато = максимум за рівнями з ≥ min_blocks блоків, [SE за рівнями])."""
    x = np.asarray(x, float)
    out = []
    while len(x) >= min_blocks:
        out.append(x.std(ddof=1) / np.sqrt(len(x)))
        x = 0.5 * (x[: len(x) // 2 * 2:2] + x[1: len(x) // 2 * 2:2])
    return (max(out) if out else np.nan), out
