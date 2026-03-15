from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.tasks.task3_cislr.utils.make_mask import make_mask
from src.tasks.task3_cislr.utils.masked_pool import masked_pool

class BiGRUPoolEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 1,
        pool: str = "max",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert pool in {"max", "mean"}

        self.pool = pool
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.proj_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim * 2, out_dim)


    def forward(self, features: torch.Tensor, feat_lens: torch.Tensor) -> torch.Tensor:
        """
        features: (B, T, 1024)
        feat_lens: (B,)
        returns: (B, out_dim) normalized embeddings
        """
        # GRU
        encoded, _ = self.gru(features)  # (B, T, 2H)

        # Temporal pooling
        pooled = masked_pool(encoded, feat_lens, variant=self.pool)  # (B, 2H)

        # Projection
        emb = self.proj_dropout(pooled)
        emb = self.proj(emb)

        # Normalize for cosine retrieval
        emb = F.normalize(emb, dim=-1)

        return emb


class BiGRUAttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.proj_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim * 2, out_dim)

    def attention_pool(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        lengths: (B,)
        """
        B, T, D = x.shape
        mask = make_mask(lengths, T, x.device)  # (B, T)

        scores = self.attn(x).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, T)

        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return pooled

    def forward(self, features: torch.Tensor, feat_lens: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(features)  # (B, T, 2H)
        pooled = self.attention_pool(encoded, feat_lens)
        emb = self.proj_dropout(pooled)
        emb = self.proj(emb)
        emb = F.normalize(emb, dim=-1)
        return emb
