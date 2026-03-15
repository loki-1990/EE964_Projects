import torch
import torch.nn.functional as F


def stat_pool_torch(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Convert padded pose sequences into fixed-size stat-pooled features.

    Args:
        data: (B, T, D)
        mask: (B, T) with 1 for valid frames, 0 for padding

    Returns:
        feats: (B, 3D) = [mean, std, mean_abs_velocity]
    """
    mask = mask.float()
    mask_exp = mask.unsqueeze(-1)  # (B, T, 1)

    valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)

    # mean
    mean_feat = (data * mask_exp).sum(dim=1) / valid_counts  # (B, D)

    # std
    centered = data - mean_feat.unsqueeze(1)                  # (B, T, D)
    var_feat = ((centered ** 2) * mask_exp).sum(dim=1) / valid_counts
    std_feat = torch.sqrt(var_feat + 1e-8)                    # (B, D)

    # mean absolute velocity
    if data.size(1) >= 2:
        diff = torch.abs(data[:, 1:, :] - data[:, :-1, :])    # (B, T-1, D)
        diff_mask = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1).float()
        diff_counts = diff_mask.sum(dim=1).clamp_min(1.0)     # (B, 1)
        vel_feat = (diff * diff_mask).sum(dim=1) / diff_counts
    else:
        vel_feat = torch.zeros_like(mean_feat)

    feats = torch.cat([mean_feat, std_feat, vel_feat], dim=-1)  # (B, 3D)
    return feats


def symmetric_inbatch_contrastive_loss(
    word_emb: torch.Tensor,
    desc_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standard symmetric in-batch contrastive loss.

    Args:
        word_emb: (B, H), already normalized
        desc_emb: (B, H), already normalized
        temperature: scalar

    Returns:
        scalar loss
    """
    logits = (word_emb @ desc_emb.T) / temperature            # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_w2d = F.cross_entropy(logits, targets)
    loss_d2w = F.cross_entropy(logits.T, targets)

    return 0.5 * (loss_w2d + loss_d2w)


def train_one_epoch_statpool_mlp(
    model,
    loader,
    optimizer,
    device,
    temperature: float = 0.07,
    grad_clip: float | None = 1.0,
):
    """
    Train for one epoch using only positive pairs and in-batch negatives.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        word_data = batch["word_data"].to(device)   # (B, T, D)
        word_mask = batch["word_mask"].to(device)   # (B, T)
        desc_data = batch["desc_data"].to(device)   # (B, T, D)
        desc_mask = batch["desc_mask"].to(device)   # (B, T)

        word_feat = stat_pool_torch(word_data, word_mask)     # (B, 675)
        desc_feat = stat_pool_torch(desc_data, desc_mask)     # (B, 675)

        word_emb = model(word_feat)                           # (B, H)
        desc_emb = model(desc_feat)                           # (B, H)

        loss = symmetric_inbatch_contrastive_loss(
            word_emb=word_emb,
            desc_emb=desc_emb,
            temperature=temperature,
        )

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += float(loss.detach().cpu())
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def encode_statpool_pairs(model, loader, device):
    """
    Utility function to inspect embedding generation on a loader.
    Not the final retrieval eval; just a helper.

    Returns:
        all_word_emb: (N, H)
        all_desc_emb: (N, H)
    """
    model.eval()

    word_embs = []
    desc_embs = []

    for batch in loader:
        word_data = batch["word_data"].to(device)
        word_mask = batch["word_mask"].to(device)
        desc_data = batch["desc_data"].to(device)
        desc_mask = batch["desc_mask"].to(device)

        word_feat = stat_pool_torch(word_data, word_mask)
        desc_feat = stat_pool_torch(desc_data, desc_mask)

        word_emb = model(word_feat)
        desc_emb = model(desc_feat)

        word_embs.append(word_emb.cpu())
        desc_embs.append(desc_emb.cpu())

    all_word_emb = torch.cat(word_embs, dim=0)
    all_desc_emb = torch.cat(desc_embs, dim=0)

    return all_word_emb, all_desc_emb