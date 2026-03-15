import torch
import torch.nn.functional as F


def symmetric_inbatch_contrastive_loss(
    word_emb: torch.Tensor,
    desc_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric in-batch contrastive loss.
    """
    logits = (word_emb @ desc_emb.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_w2d = F.cross_entropy(logits, targets)
    loss_d2w = F.cross_entropy(logits.T, targets)

    return 0.5 * (loss_w2d + loss_d2w)


def train_one_epoch_meanpool_temporal(
    model,
    loader,
    optimizer,
    device,
    temperature: float = 0.07,
    grad_clip: float | None = 1.0,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        word_data = batch["word_data"].to(device)
        word_mask = batch["word_mask"].to(device)
        desc_data = batch["desc_data"].to(device)
        desc_mask = batch["desc_mask"].to(device)

        word_emb = model(word_data, word_mask)
        desc_emb = model(desc_data, desc_mask)

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