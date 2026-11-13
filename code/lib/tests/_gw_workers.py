"""Функції верхнього рівня для перевірки run_chunks у процесах (spawn не вміє локальні функції)."""
import numpy as np


def toy_worker(i, seq, args):
    rng = np.random.default_rng(seq)
    return dict(x=rng.normal(size=args["m"]), i=np.int64(i))
