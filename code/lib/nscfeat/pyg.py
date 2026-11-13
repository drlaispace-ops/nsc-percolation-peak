"""Трансформ PyTorch Geometric: NSC-профіль як структурне кодування вузлів.

Аналог torch_geometric.transforms.AddRandomWalkPE (RWSE, Dwivedi2022 / Rampasek2022):
    T.Compose([AddNSCProfile(K=8, mode="delta"), ...])
Рахується на CPU (numba); результат — float32 на пристрої edge_index.
Найкраще використовувати як pre_transform: кодування не залежить від ознак вузлів.
"""
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms.add_positional_encoding import add_node_attr

from .core import MODES, bfs_profile, csr_from_edge_index, encode


class AddNSCProfile(BaseTransform):
    r"""Додає NSC-профіль BFS-куль кожного вузла (Snarskii, PRE 110, 054309, 2024).

    Args:
        K (int): кількість радіусів BFS (стовпчиків профілю).
        mode (str): ``"delta"`` — δ_k (K ознак), ``"p"`` — щільність кулі p_k (K),
            ``"sb"`` — [log s_k, b_k/s_k] (2K; найповніший, містить і p, і δ).
        full (bool): ``True`` — хвіст k > K додається в останній стовпчик δ
            (рядкова сума = NodeSC); ``False`` — BFS обрізано на радіусі K (швидше
            на великих графах).
        attr_name (str, optional): ім'я атрибута; ``None`` — дописати до ``data.x``.
    """

    def __init__(self, K: int = 8, mode: str = "delta", full: bool = True,
                 attr_name: str | None = "nsc_pe") -> None:
        if mode not in MODES:
            raise ValueError(f"mode ∈ {MODES}")
        self.K, self.mode, self.full, self.attr_name = K, mode, full, attr_name

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        N = data.num_nodes
        indptr, indices = csr_from_edge_index(data.edge_index.cpu().numpy(), N)
        S, B, D = bfs_profile(indptr, indices, N, self.K, self.full)
        pe = torch.from_numpy(encode(S, B, D, self.mode)).to(torch.float32)
        return add_node_attr(data, pe.to(data.edge_index.device), attr_name=self.attr_name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K}, mode={self.mode!r}, full={self.full})"
