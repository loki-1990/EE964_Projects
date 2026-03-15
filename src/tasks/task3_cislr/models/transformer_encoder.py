from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.tasks.task3_cislr.utils.make_mask import make_mask
from src.tasks.task3_cislr.utils.masked_pool import masked_pool


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        model_dim: int = 256,
        out_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        ff_mult: int = 512,
        dropout: float = 0.1,
        pool: str = "mean",
        max_len: int = 2000,
    ) -> None:
        super().__init__()
        assert pool in {"mean", "max"}

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_mult = ff_mult
        self.dropout_p = dropout
        self.pool = pool

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(d_model=model_dim, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_mult * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.proj_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(model_dim, out_dim)


    def forward(self, features: torch.Tensor, feat_lens: torch.Tensor) -> torch.Tensor:
        """
        features: (B, T, 1024)
        feat_lens: (B,)
        returns: (B, out_dim)
        """
        # (B, T, model_dim)
        x = self.input_proj(features)
        x = self.pos_encoder(x)

        # src_key_padding_mask expects True for padded positions
        pad_mask = ~make_mask(feat_lens, x.size(1), x.device)  # (B, T)

        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, T, model_dim)

        pooled = masked_pool(x, feat_lens, variant=self.pool)  # (B, model_dim)

        emb = self.proj_dropout(pooled)
        emb = self.output_proj(emb)
        emb = F.normalize(emb, dim=-1)

        return emb