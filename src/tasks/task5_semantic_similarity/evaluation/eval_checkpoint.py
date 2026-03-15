import torch


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    return model, ckpt


@torch.no_grad()
def evaluate_model_retrieval(model, loader, device):
    """
    Generic retrieval evaluation.
    Works for both MeanPoolTemporalEncoder and SmallTransformerEncoder.
    """

    import numpy as np

    query_word_ids = []
    query_sentence_ids = []
    query_word_embs = []

    candidate_sentence_ids = []
    candidate_desc_embs = []
    seen_sentence_ids = set()

    for batch in loader:
        word_ids = batch["word_id"]
        sentence_ids = batch["sentence_id"]

        word_data = batch["word_data"].to(device)
        word_mask = batch["word_mask"].to(device)
        desc_data = batch["desc_data"].to(device)
        desc_mask = batch["desc_mask"].to(device)

        word_emb = model(word_data, word_mask)
        desc_emb = model(desc_data, desc_mask)

        # queries
        for i in range(len(word_ids)):
            query_word_ids.append(word_ids[i])
            query_sentence_ids.append(sentence_ids[i])
            query_word_embs.append(word_emb[i].cpu().numpy())

        # candidate pool
        for i in range(len(sentence_ids)):
            sid = sentence_ids[i]
            if sid not in seen_sentence_ids:
                seen_sentence_ids.add(sid)
                candidate_sentence_ids.append(sid)
                candidate_desc_embs.append(desc_emb[i].cpu().numpy())

    query_word_embs = np.stack(query_word_embs)
    candidate_desc_embs = np.stack(candidate_desc_embs)

    sid_to_index = {sid: i for i, sid in enumerate(candidate_sentence_ids)}

    n_candidates = len(candidate_sentence_ids)
    k_top5pct = max(1, int(np.ceil(0.05 * n_candidates)))

    ranks = []

    for i in range(len(query_word_ids)):
        true_sid = query_sentence_ids[i]
        true_idx = sid_to_index[true_sid]

        scores = candidate_desc_embs @ query_word_embs[i]
        ranked = np.argsort(-scores)

        rank = int(np.where(ranked == true_idx)[0][0]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)

    return {
        "num_queries": int(len(query_word_ids)),
        "num_candidates": int(n_candidates),
        "top5pct_k": int(k_top5pct),
        "Top5%": float((ranks <= k_top5pct).mean()),
        "MeanRank": float(ranks.mean()),
        "MedianRank": float(np.median(ranks)),
        "MRR": float((1.0 / ranks).mean()),
    }