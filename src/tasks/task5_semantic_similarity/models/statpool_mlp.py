import torch
import torch.nn as nn
import torch.nn.functional as F


class StatPoolMLP(nn.Module):
    """
    Simple MLP over stat-pooled pose features.

    Input feature is expected to be:
        [mean, std, mean_abs_velocity]
    concatenated over pose dimensions.

    If pose feature dim = 225, then stat-pooled input dim = 225 * 3 = 675.
    """

    def __init__(
        self,
        input_dim: int = 675,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)
        returns: (B, output_dim), L2-normalized
        """
        z = self.net(x)
        z = F.normalize(z, p=2, dim=-1)
        return z