"""Графові (graph-level) NSC-ознаки для scikit-learn.

Вектор ознак як у пілоті code/exp5_ml.py (nsc_feats):
  середній профіль δ_k за вузлами (K) + квантилі NodeSC (за замовчуванням
  0.1, 0.25, 0.5, 0.75, 0.9) + середнє і SD NodeSC.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .core import bfs_profile, csr_from_graph

DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def graph_features(D, quantiles=DEFAULT_QUANTILES):
    """D — матриця δ_k (n × K) з full=True."""
    ns = D.sum(1)
    if len(ns) == 0:
        return np.zeros(D.shape[1] + len(quantiles) + 2)
    return np.concatenate([D.mean(0), np.quantile(ns, quantiles), [ns.mean(), ns.std()]])


class NSCFeaturizer(TransformerMixin, BaseEstimator):
    """Граф (networkx або scipy sparse) → вектор NSC-ознак. Без навчуваних параметрів."""

    def __init__(self, K=16, quantiles=DEFAULT_QUANTILES):
        self.K = K
        self.quantiles = quantiles

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for G in X:
            indptr, indices, n = csr_from_graph(G)
            _, _, D = bfs_profile(indptr, indices, n, self.K, full=True)
            out.append(graph_features(D, self.quantiles))
        return np.vstack(out)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"delta_{k}" for k in range(1, self.K + 1)]
                        + [f"nodesc_q{q:g}" for q in self.quantiles] + ["nodesc_mean", "nodesc_std"])
