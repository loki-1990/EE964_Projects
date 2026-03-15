from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def _load_feature_tensor(feature_source, vid: str) -> torch.Tensor:
    if isinstance(feature_source, dict):
        x = feature_source[vid]
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(x, dtype=torch.float32)

    feature_dir = Path(feature_source)
    x = np.load(feature_dir / f"{vid}.npy")
    return torch.tensor(x, dtype=torch.float32)

@torch.no_grad()
def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    acc = (preds == labels).float().mean().item() * 100.0
    return {"accuracy": acc}

@torch.no_grad()
def build_word_embedding_bank(
    model: torch.nn.Module,
    word_ids: list[str],
    feature_source,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Build one embedding per unique word video id.
    """
    model.eval()

    embeddings = {}

    batch_tensors = []
    batch_lengths = []
    batch_ids = []

    def flush_batch():
        nonlocal batch_tensors, batch_lengths, batch_ids, embeddings
        if len(batch_tensors) == 0:
            return

        max_len = max(x.shape[0] for x in batch_tensors)
        feat_dim = batch_tensors[0].shape[1]
        padded = torch.zeros(len(batch_tensors), max_len, feat_dim, dtype=torch.float32)
        lengths = torch.tensor(batch_lengths, dtype=torch.long)

        for i, x in enumerate(batch_tensors):
            padded[i, : x.shape[0]] = x

        padded = padded.to(device)
        lengths = lengths.to(device)

        emb = model.encoder(padded, lengths)
        emb = F.normalize(emb, p=2, dim=1).cpu()

        for vid, e in zip(batch_ids, emb):
            embeddings[vid] = e

        batch_tensors = []
        batch_lengths = []
        batch_ids = []

    for vid in word_ids:
        x = _load_feature_tensor(feature_source, vid)

        batch_tensors.append(x)
        batch_lengths.append(x.shape[0])
        batch_ids.append(vid)

        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()
    return embeddings

@torch.no_grad()
def build_sentence_embedding_bank(
    model: torch.nn.Module,
    sentence_ids: list[str],
    feature_source,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Build one embedding per unique sentence video id.
    """
    model.eval()

    embeddings = {}

    batch_tensors = []
    batch_lengths = []
    batch_ids = []

    def flush_batch():
        nonlocal batch_tensors, batch_lengths, batch_ids, embeddings
        if len(batch_tensors) == 0:
            return

        max_len = max(x.shape[0] for x in batch_tensors)
        feat_dim = batch_tensors[0].shape[1]
        padded = torch.zeros(len(batch_tensors), max_len, feat_dim, dtype=torch.float32)
        lengths = torch.tensor(batch_lengths, dtype=torch.long)

        for i, x in enumerate(batch_tensors):
            padded[i, : x.shape[0]] = x

        padded = padded.to(device)
        lengths = lengths.to(device)

        emb = model.encoder(padded, lengths)
        emb = F.normalize(emb, p=2, dim=1).cpu()

        for vid, e in zip(batch_ids, emb):
            embeddings[vid] = e

        batch_tensors = []
        batch_lengths = []
        batch_ids = []

    for vid in sentence_ids:
        x = _load_feature_tensor(feature_source, vid)

        batch_tensors.append(x)
        batch_lengths.append(x.shape[0])
        batch_ids.append(vid)

        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()
    return embeddings

@torch.no_grad()
def evaluate_top_rank_avg(
    model: torch.nn.Module,
    split_csv: str | Path,
    feature_source,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    df = pd.read_csv(split_csv)

    positive_map: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        if int(row["label"]) == 1:
            positive_map[str(row["word_id"])].add(str(row["sentence_id"]))

    query_word_ids = sorted(positive_map.keys())
    all_sentence_ids = sorted(df["sentence_id"].dropna().astype(str).unique().tolist())

    word_bank = build_word_embedding_bank(
        model=model,
        word_ids=query_word_ids,
        feature_source=feature_source,
        batch_size=batch_size,
        device=device,
    )
    sent_bank = build_sentence_embedding_bank(
        model=model,
        sentence_ids=all_sentence_ids,
        feature_source=feature_source,
        batch_size=batch_size,
        device=device,
    )

    sent_ids = all_sentence_ids
    sent_id_to_idx = {sid: i for i, sid in enumerate(sent_ids)}
    sent_matrix = torch.stack([sent_bank[sid] for sid in sent_ids], dim=0)  # (N, D)

    per_query_best_ranks = []
    all_positive_ranks = []

    for wid in query_word_ids:
        w_emb = word_bank[wid].unsqueeze(0)  # (1, D)
        sim = torch.matmul(w_emb, sent_matrix.T).squeeze(0)  # (N,)

        ranked_idx = torch.argsort(sim, descending=True)

        rank_positions = torch.empty_like(ranked_idx)
        rank_positions[ranked_idx] = torch.arange(
            1, len(ranked_idx) + 1, device=ranked_idx.device
        )

        pos_set = positive_map[wid]
        pos_ranks = []

        for pos_sid in pos_set:
            pos_col = sent_id_to_idx[pos_sid]
            rank_1_based = int(rank_positions[pos_col].item())
            pos_ranks.append(rank_1_based)
            all_positive_ranks.append(rank_1_based)

        per_query_best_ranks.append(min(pos_ranks))

    avg_rank = float(np.mean(per_query_best_ranks)) if per_query_best_ranks else float("nan")
    avg_positive_rank = float(np.mean(all_positive_ranks)) if all_positive_ranks else float("nan")

    return {
        "avg_rank": avg_rank,
        "avg_positive_rank": avg_positive_rank,
        "num_queries": len(query_word_ids),
        "num_candidates": len(all_sentence_ids),
    }