"""nscfeat — NSC-ознаки графів (Snarskii, PRE 110, 054309, 2024).

core      — numba-ядро профілю BFS-куль (s_k, b_k, δ_k), режими кодування;
features  — графові ознаки для scikit-learn (NSCFeaturizer);
pyg       — трансформ PyTorch Geometric AddNSCProfile (імпортується окремо,
            щоб ядро не залежало від torch).
"""
from .core import MODES, ball_density, bfs_profile, csr_from_edge_index, csr_from_graph, encode, nsc_encoding
from .features import NSCFeaturizer, graph_features

__all__ = ["MODES", "ball_density", "bfs_profile", "csr_from_edge_index", "csr_from_graph", "encode",
           "nsc_encoding", "NSCFeaturizer", "graph_features"]
