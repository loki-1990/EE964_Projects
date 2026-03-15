import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.task4_word_presence.models.stat_pool_baseline import pool_features


class GRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1728,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        pool: str = "mean",
    ):
        super().__init__()

        if pool not in {"mean", "max", "meanmax", "meanstd"}:
            raise ValueError(f"Unsupported pool: {pool}")

        self.pool = pool
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=False,
        )

        pooled_dim = hidden_dim if pool in {"mean", "max"} else hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(pooled_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
        )  # (B, T_valid, H)

        pooled = pool_features(
            features=out,
            feat_lens=lengths,
            variant=self.pool,
            normalize=False,
        )

        pooled = self.dropout(pooled)
        emb = self.output_proj(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class BiGRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1728,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        pool: str = "mean",
    ):
        super().__init__()

        if pool not in {"mean", "max", "meanmax", "meanstd"}:
            raise ValueError(f"Unsupported pool: {pool}")

        self.pool = pool
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=True,
        )

        temporal_dim = hidden_dim * 2
        pooled_dim = temporal_dim if pool in {"mean", "max"} else temporal_dim * 2

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(pooled_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
        )  # (B, T_valid, 2H)

        pooled = pool_features(
            features=out,
            feat_lens=lengths,
            variant=self.pool,
            normalize=False,
        )

        pooled = self.dropout(pooled)
        emb = self.output_proj(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class WordPresenceGRUModel(nn.Module):
    """
    Shared temporal encoder for word and sentence.
    Cosine similarity -> trainable affine logit.
    Matches StatPoolBaseline forward contract exactly.
    """
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
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