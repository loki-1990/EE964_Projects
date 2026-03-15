import torch
import torch.nn.functional as F

def _make_mask(feat_lens: torch.Tensor, tmax: int, device: torch.device) -> torch.Tensor:
    """
    Create a boolean mask for variable-length sequences.

    Args:
        feat_lens: (B,) lengths of each sequence
        tmax: maximum sequence length
        device: torch device

    Returns:
        mask: (B, Tmax) boolean mask
    """
    time_idx = torch.arange(tmax, device=device).unsqueeze(0)          # (1, Tmax)
    mask = time_idx < feat_lens.unsqueeze(1)                           # (B, Tmax)
    return mask

def pool_features(
    features: torch.Tensor,
    feat_lens: torch.Tensor,
    variant: str = "mean",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Pool padded temporal features into fixed-size embeddings.

    Args:
        features: (B, Tmax, D) padded features
        feat_lens: (B,) lengths of each sequence
        variant: one of {"mean", "max", "meanmax", "meanstd"}
        normalize: whether to L2-normalize the pooled output

    Returns:
        emb: (B, E)
    """
    _, tmax, _ = features.shape
    device = features.device

    mask = _make_mask(feat_lens, tmax, device)  # (B, Tmax)
    mask_f = mask.unsqueeze(-1).float()  # (B, Tmax, 1)

    summed = (features * mask_f).sum(dim=1)  # (B, D)
    denom = feat_lens.clamp(min=1).unsqueeze(1).float()  # (B, 1)
    mean_pooled = summed / denom  # (B, D)

    if variant == "mean":
        emb = mean_pooled

    elif variant == "max":
        masked_features = features.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        emb = masked_features.max(dim=1).values
        emb[torch.isinf(emb)] = 0.0

    elif variant == "meanmax":
        masked_features = features.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_pooled = masked_features.max(dim=1).values
        max_pooled[torch.isinf(max_pooled)] = 0.0
        emb = torch.cat([mean_pooled, max_pooled], dim=1)

    elif variant == "meanstd":
        centered = (features - mean_pooled.unsqueeze(1)) * mask_f
        var_pooled = (centered ** 2).sum(dim=1) / denom
        std_pooled = torch.sqrt(var_pooled + 1e-8)
        emb = torch.cat([mean_pooled, std_pooled], dim=1)

    else:
        raise ValueError(
            f"Unsupported pooling variant: {variant}. "
            f"Choose from ['mean', 'max', 'meanmax', 'meanstd']."
        )

    if normalize:
        emb = F.normalize(emb, p=2, dim=1)

    return emb