
import torch
from torch.utils.data import DataLoader
from typing import Any
from src.tasks.task3_cislr.utils.embedding_utils import pool_features

@torch.no_grad()
def build_embedding_bank(
    loader: DataLoader,
    device: torch.device,
    pooling_variant: str = "mean",
    model: torch.nn.Module | None = None,  # Optional model for more complex embedding extraction (not used in simple pooling)
) -> dict[str, Any]:
    """
    Build normalized embeddings for all samples in a loader.

    Args:
        loader: DataLoader returning batch dict with keys:
            - features
            - feat_len
            - uid
            - gloss
        device: torch device
        pooling_variant: one of ["mean", "max", "meanmax", "meanstd"]

    Returns:
        dict with:
            - embeddings: (N, E)
            - uids: list[str]
            - glosses: list[str]
    """
    all_embeddings = []
    all_uids = []
    all_glosses = []

    for batch in loader:
        features = batch["features"].to(device)         # (B, Tmax, D)
        feat_lens = batch["feat_len"].to(device)        # (B,)

        if model is None:
            emb = pool_features(features, feat_lens, variant=pooling_variant)   # (B, E)
        else:
            # Placeholder for more complex embedding extraction using a model
            emb = model(features, feat_lens)  # (B, E)
        all_embeddings.append(emb.cpu())
        all_uids.extend(batch["uid"])
        all_glosses.extend(batch["gloss"])

    all_embeddings = torch.cat(all_embeddings, dim=0)

    return {
        "embeddings": all_embeddings,
        "uids": all_uids,
        "glosses": all_glosses,
    }

def compute_similarity(query_emb: torch.Tensor, bank_emb: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity for normalized embeddings using dot product.

    Args:
        query_emb: (Nq, D)
        bank_emb: (Nb, D)

    Returns:
        sim: (Nq, Nb)
    """
    return query_emb @ bank_emb.T


def evaluate_topk(
    sim: torch.Tensor,
    query_glosses: list[str],
    bank_glosses: list[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """
    Evaluate Top-k accuracy based on gloss match.
    """
    max_k = max(ks)
    topk_idx = torch.topk(sim, k=max_k, dim=1, largest=True, sorted=True).indices  # (Nq, max_k)

    results = {}
    num_queries = len(query_glosses)

    for k in ks:
        correct = 0
        for i in range(num_queries):
            retrieved_glosses = [bank_glosses[j] for j in topk_idx[i, :k].tolist()]
            if query_glosses[i] in retrieved_glosses:
                correct += 1
        results[f"top{k}"] = 100.0 * correct / num_queries

    return results