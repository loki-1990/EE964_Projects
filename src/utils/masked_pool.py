import torch
from src.tasks.task3_cislr.utils.make_mask import make_mask


def masked_pool(x: torch.Tensor, lengths: torch.Tensor, variant: str = "max") -> torch.Tensor:
    """
    x: (B, T, D)
    lengths: (B,)
    variant: "max" or "mean"
    returns: (B, D)
    """
    assert variant in {"max", "mean"}

    B, T, D = x.shape
    mask = make_mask(lengths, T, x.device)  # (B, T)

    if variant == "max":
        x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = x.max(dim=1).values
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        return pooled

    x = x * mask.unsqueeze(-1)
    denom = lengths.clamp_min(1).unsqueeze(1).to(x.dtype)
    return x.sum(dim=1) / denom