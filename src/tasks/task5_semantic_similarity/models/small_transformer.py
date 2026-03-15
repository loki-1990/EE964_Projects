import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallTransformerEncoder(nn.Module):
    """
    Small transformer baseline for Task-5 retrieval.

    Pipeline:
      input sequence -> linear projection -> positional embedding
      -> 1 transformer encoder layer -> masked mean pooling
      -> output projection -> L2 normalize
    """

    def __init__(
        self,
        input_dim: int = 225,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        ff_mult: int = 2,
        output_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 400,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, input_dim)
        mask: (B, T) with 1 for valid frames, 0 for padding
        """
        bsz, seq_len, _ = x.shape

        h = self.input_proj(x)                          # (B, T, d_model)
        h = h + self.pos_embed[:, :seq_len, :]

        key_padding_mask = (mask == 0)                  # True where padding
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)

        mask_exp = mask.float().unsqueeze(-1)           # (B, T, 1)
        h = h * mask_exp
        valid_counts = mask_exp.sum(dim=1).clamp_min(1.0)
        pooled = h.sum(dim=1) / valid_counts            # (B, d_model)

        z = self.output_proj(pooled)                    # (B, output_dim)
        z = F.normalize(z, p=2, dim=-1)
        return z