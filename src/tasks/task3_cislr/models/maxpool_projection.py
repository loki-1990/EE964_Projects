from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.task3_cislr.utils.embedding_utils import pool_features


class MaxPoolProjectionV1(nn.Module):
    """
    Version-1:
        features -> max pool -> dropout -> linear projection -> L2 normalize
    """

    def __init__(
        self,
        input_dim: int = 1024,
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dropout_p = dropout

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_dim, out_dim)

    def forward(self, features: torch.Tensor, feat_len: torch.Tensor) -> torch.Tensor:
        pooled = pool_features(
            features=features,
            feat_lens=feat_len,
            variant="max",
            normalize=False,
        )  # (B, input_dim)

        pooled = self.dropout(pooled)
        emb = self.proj(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class MaxPoolProjectionV2(nn.Module):
    """
    Version-2:
        features -> max pool -> Linear -> ReLU -> Dropout -> Linear -> L2 normalize
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_p = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, features: torch.Tensor, feat_len: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, Tmax, input_dim)
            feat_len: (B,)

        Returns:
            emb: (B, out_dim), L2-normalized
        """
        pooled = pool_features(
            features=features,
            feat_lens=feat_len,
            variant="max",
            normalize=False,
        )  # (B, input_dim)

        x = self.fc1(pooled)     # (B, hidden_dim)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)          # (B, out_dim)
        emb = F.normalize(x, p=2, dim=1)

        return emb