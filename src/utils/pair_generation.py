import numpy as np
import pandas as pd
def build_labeled_pairs(pos_df: pd.DataFrame, neg_per_pos: int = 3, seed: int = 1990) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pos_df = pos_df.copy()
    pos_df["label"] = 1

    all_sent_ids = sorted(pos_df["sentence_id"].unique())
    sent_text_map = (
        pos_df[["sentence_id", "sentence"]]
        .drop_duplicates()
        .set_index("sentence_id")["sentence"]
        .to_dict()
    )

    positive_map = (
        pos_df.groupby("word_id")["sentence_id"]
        .apply(set)
        .to_dict()
    )

    word_text_map = (
        pos_df[["word_id", "word"]]
        .drop_duplicates()
        .set_index("word_id")["word"]
        .to_dict()
    )

    neg_rows = []

    for row in pos_df.itertuples(index=False):
        word_id = row.word_id
        word = row.word

        positive_sent_ids = positive_map[word_id]
        candidate_neg_ids = [sid for sid in all_sent_ids if sid not in positive_sent_ids]

        if len(candidate_neg_ids) == 0:
            continue

        k = min(neg_per_pos, len(candidate_neg_ids))
        sampled_neg_ids = rng.choice(candidate_neg_ids, size=k, replace=False)

        for neg_sid in sampled_neg_ids:
            neg_rows.append({
                "word_id": word_id,
                "word": word,
                "sentence_id": neg_sid,
                "sentence": sent_text_map[neg_sid],
                "label": 0,
            })

    neg_df = pd.DataFrame(neg_rows)

    labeled_df = pd.concat([pos_df, neg_df], ignore_index=True)
    labeled_df = labeled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return labeled_df