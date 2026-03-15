import numpy as np


def l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x)
    return x / (norm + eps)


def stat_pool_features(data, mask):
    """
    data: (T, D)
    mask: (T,)
    Uses only active frames.
    Returns a fixed vector of size 3D:
      [mean, std, mean_abs_velocity]
    """
    valid = data[mask > 0]

    if len(valid) == 0:
        d = data.shape[1]
        return np.zeros(d * 3, dtype=np.float32)

    mean_feat = valid.mean(axis=0)
    std_feat = valid.std(axis=0)

    if len(valid) >= 2:
        vel = np.abs(valid[1:] - valid[:-1]).mean(axis=0)
    else:
        vel = np.zeros(valid.shape[1], dtype=np.float32)

    feat = np.concatenate([mean_feat, std_feat, vel], axis=0).astype(np.float32)
    feat = l2_normalize(feat)
    return feat


def cosine_similarity_matrix(query_vec, cand_matrix):
    return cand_matrix @ query_vec


def statpool_retrieval_baseline(val_df, pose_dir, load_pose_fn,
                                max_word_frames=200, max_desc_frames=400):
    """
    Retrieval baseline on a split of positive pairs.
    Queries: all rows in val_df
    Candidates: unique sentence_id in val_df
    """
    candidate_ids = val_df["sentence_id"].drop_duplicates().tolist()
    n_candidates = len(candidate_ids)
    k_top5pct = max(1, int(np.ceil(0.05 * n_candidates)))

    # build candidate feature matrix once
    cand_features = []
    cand_id_to_index = {}

    for idx, sent_id in enumerate(candidate_ids):
        desc_path = f"{pose_dir}/{sent_id}.pose"
        desc_data, desc_mask = load_pose_fn(desc_path, max_desc_frames)
        feat = stat_pool_features(desc_data, desc_mask)
        cand_features.append(feat)
        cand_id_to_index[sent_id] = idx

    cand_features = np.stack(cand_features, axis=0)   # (N, F)

    ranks = []

    for _, row in val_df.iterrows():
        word_path = f"{pose_dir}/{row['word_id']}.pose"
        word_data, word_mask = load_pose_fn(word_path, max_word_frames)
        word_feat = stat_pool_features(word_data, word_mask)

        scores = cosine_similarity_matrix(word_feat, cand_features)
        ranked = np.argsort(-scores)

        true_idx = cand_id_to_index[row["sentence_id"]]
        rank = int(np.where(ranked == true_idx)[0][0]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)

    metrics = {
        "num_queries": int(len(val_df)),
        "num_candidates": int(n_candidates),
        "top5pct_k": int(k_top5pct),
        "Top5%": float(np.mean(ranks <= k_top5pct)),
        "MeanRank": float(np.mean(ranks)),
        "MedianRank": float(np.median(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
    }

    return metrics