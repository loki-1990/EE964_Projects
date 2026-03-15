import json
import os
import numpy as np
import pandas as pd


def create_splits(df_pos: pd.DataFrame, seed: int = 1990,
                  train_frac: float = 0.8, val_frac: float = 0.1):
    """
    Create description-disjoint train/val/test splits.

    Expected columns in df_pos:
        - word_id
        - sentence_id

    Returns:
        train_pos, val_pos, test_pos, split_info
    """
    required_cols = {"word_id", "sentence_id"}
    if not required_cols.issubset(df_pos.columns):
        raise ValueError(f"df_pos must contain columns: {required_cols}")

    rng = np.random.default_rng(seed)

    # Split by unique description videos
    desc_ids = df_pos["sentence_id"].dropna().unique().tolist()
    rng.shuffle(desc_ids)

    n = len(desc_ids)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_desc = set(desc_ids[:n_train])
    val_desc = set(desc_ids[n_train:n_train + n_val])
    test_desc = set(desc_ids[n_train + n_val:])

    train_pos = df_pos[df_pos["sentence_id"].isin(train_desc)].reset_index(drop=True)
    val_pos = df_pos[df_pos["sentence_id"].isin(val_desc)].reset_index(drop=True)
    test_pos = df_pos[df_pos["sentence_id"].isin(test_desc)].reset_index(drop=True)

    # Hard leakage checks
    train_desc_ids = set(train_pos["sentence_id"].unique())
    val_desc_ids = set(val_pos["sentence_id"].unique())
    test_desc_ids = set(test_pos["sentence_id"].unique())

    assert len(train_desc_ids & val_desc_ids) == 0
    assert len(train_desc_ids & test_desc_ids) == 0
    assert len(val_desc_ids & test_desc_ids) == 0

    split_info = {
        "seed": seed,
        "train_pairs": int(len(train_pos)),
        "val_pairs": int(len(val_pos)),
        "test_pairs": int(len(test_pos)),
        "train_unique_desc": int(train_pos["sentence_id"].nunique()),
        "val_unique_desc": int(val_pos["sentence_id"].nunique()),
        "test_unique_desc": int(test_pos["sentence_id"].nunique()),
        "train_unique_words": int(train_pos["word_id"].nunique()),
        "val_unique_words": int(val_pos["word_id"].nunique()),
        "test_unique_words": int(test_pos["word_id"].nunique()),
        "desc_overlap_train_val": int(len(train_desc_ids & val_desc_ids)),
        "desc_overlap_train_test": int(len(train_desc_ids & test_desc_ids)),
        "desc_overlap_val_test": int(len(val_desc_ids & test_desc_ids)),
    }

    return train_pos, val_pos, test_pos, split_info


def save_splits(train_pos: pd.DataFrame,
                val_pos: pd.DataFrame,
                test_pos: pd.DataFrame,
                split_info: dict,
                out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    train_pos.to_csv(os.path.join(out_dir, "train_pos.csv"), index=False)
    val_pos.to_csv(os.path.join(out_dir, "val_pos.csv"), index=False)
    test_pos.to_csv(os.path.join(out_dir, "test_pos.csv"), index=False)

    with open(os.path.join(out_dir, "split_desc_disjoint.json"), "w") as f:
        json.dump(split_info, f, indent=2)