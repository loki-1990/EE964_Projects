from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.embedding_utils import pool_features


class StatPoolEncoder(nn.Module):
    """
    Frame projection -> temporal statistical pooling -> output projection -> L2 normalize
    """

    def __init__(
        self,
        input_dim: int = 1728,
        hidden_dim: int = 256,
        output_dim: int = 256,
        pool: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pool = pool
        self.dropout_p = dropout

        self.frame_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        pooled_dim = hidden_dim
        if pool in {"meanmax", "meanstd"}:
            pooled_dim = hidden_dim * 2

        self.out_proj = nn.Sequential(
            nn.Linear(pooled_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        lengths: (B,)
        """
        x = self.frame_proj(x)  # (B, T, hidden_dim)
        x = pool_features(x, lengths, variant=self.pool, normalize=False)
        x = self.out_proj(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class StatPoolBaseline(nn.Module):
    """
    Shared encoder for word and sentence.
    Cosine similarity -> trainable affine logit.
    """

    def __init__(
        self,
        input_dim: int = 1728,
        hidden_dim: int = 256,
        output_dim: int = 256,
        pool: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pool = pool
        self.dropout_p = dropout

        self.encoder = StatPoolEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pool=pool,
            dropout=dropout,
        )

        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        word_x: torch.Tensor,
        word_len: torch.Tensor,
        sent_x: torch.Tensor,
        sent_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        word_emb = self.encoder(word_x, word_len)
        sent_emb = self.encoder(sent_x, sent_len)

        cosine = F.cosine_similarity(word_emb, sent_emb, dim=1)  # (B,)
        logits = self.alpha * cosine + self.beta
        return logits, cosine