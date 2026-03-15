import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolTemporalEncoder(nn.Module):
    """
    Simple temporal encoder:
      - frame-wise MLP
      - masked mean pooling over time
      - output projection
      - L2 normalization
    """

    def __init__(
        self,
        input_dim: int = 225,
        frame_hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, frame_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(frame_hidden_dim, frame_hidden_dim),
            nn.ReLU(),
        )

        self.output_proj = nn.Linear(frame_hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, D)
        mask: (B, T) with 1 for valid frames, 0 for padding

        returns:
            (B, output_dim), L2-normalized
        """
        h = self.frame_encoder(x)  # (B, T, H)

        mask = mask.float().unsqueeze(-1)           # (B, T, 1)
        h = h * mask

        valid_counts = mask.sum(dim=1).clamp_min(1.0)   # (B, 1)
        pooled = h.sum(dim=1) / valid_counts            # (B, H)

        z = self.output_proj(pooled)                    # (B, output_dim)
        z = F.normalize(z, p=2, dim=-1)
        return z