import numpy as np
import torch

from src.tasks.task5_semantic_similarity.training.train_statpool_mlp import stat_pool_torch


@torch.no_grad()
def eval_statpool_mlp_retrieval(
    model,
    val_loader,
    device,
):
    """
    Evaluate StatPoolMLP on Task-5 retrieval.

    Assumptions:
    - val_loader yields batches from Task5PairDataset
    - each sample is a positive pair (word_id, sentence_id)
    - candidate pool = unique sentence_id in the validation split
    - query set = all word_id in the validation split

    Returns:
        dict with Top5%, MeanRank, MedianRank, MRR, etc.
    """
    model.eval()

    query_word_ids = []
    query_sentence_ids = []
    query_word_embs = []

    candidate_sentence_ids = []
    candidate_desc_embs = []
    seen_sentence_ids = set()

    for batch in val_loader:
        word_ids = batch["word_id"]
        sentence_ids = batch["sentence_id"]

        word_data = batch["word_data"].to(device)
        word_mask = batch["word_mask"].to(device)
        desc_data = batch["desc_data"].to(device)
        desc_mask = batch["desc_mask"].to(device)

        word_feat = stat_pool_torch(word_data, word_mask)   # (B, 675)
        desc_feat = stat_pool_torch(desc_data, desc_mask)   # (B, 675)

        word_emb = model(word_feat)                         # (B, H)
        desc_emb = model(desc_feat)                         # (B, H)

        # store all query embeddings
        for i in range(len(word_ids)):
            query_word_ids.append(word_ids[i])
            query_sentence_ids.append(sentence_ids[i])
            query_word_embs.append(word_emb[i].detach().cpu().numpy())

        # build unique candidate pool from desc embeddings
        for i in range(len(sentence_ids)):
            sid = sentence_ids[i]
            if sid not in seen_sentence_ids:
                seen_sentence_ids.add(sid)
                candidate_sentence_ids.append(sid)
                candidate_desc_embs.append(desc_emb[i].detach().cpu().numpy())

    query_word_embs = np.stack(query_word_embs, axis=0)         # (Nq, H)
    candidate_desc_embs = np.stack(candidate_desc_embs, axis=0) # (Nc, H)

    sentence_id_to_index = {
        sid: idx for idx, sid in enumerate(candidate_sentence_ids)
    }

    n_candidates = len(candidate_sentence_ids)
    k_top5pct = max(1, int(np.ceil(0.05 * n_candidates)))

    ranks = []

    for i in range(len(query_word_ids)):
        true_sid = query_sentence_ids[i]
        true_idx = sentence_id_to_index[true_sid]

        scores = candidate_desc_embs @ query_word_embs[i]   # cosine, since outputs normalized
        ranked = np.argsort(-scores)

        rank = int(np.where(ranked == true_idx)[0][0]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)

    metrics = {
        "num_queries": int(len(query_word_ids)),
        "num_candidates": int(n_candidates),
        "top5pct_k": int(k_top5pct),
        "Top5%": float(np.mean(ranks <= k_top5pct)),
        "MeanRank": float(np.mean(ranks)),
        "MedianRank": float(np.median(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
    }

    return metrics