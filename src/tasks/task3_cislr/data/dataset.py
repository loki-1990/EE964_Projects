from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def reshape_i3d_feature(raw_feature: Any) -> np.ndarray:
    """
    Convert raw I3D feature from shape (1, 1024, T, 1, 1) to (T, 1024).

    Expected raw storage pattern:
        (N, C, T, H, W) with N=1, H=1, W=1

    Returns:
        np.ndarray of shape (T, 1024)
    """

    x = np.asarray(raw_feature, dtype=np.float32)
    x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got shape {x.shape}")

    if x.shape[0] == 1024:
        x = x.T              # (1024, T) -> (T, 1024)
    elif x.shape[1] == 1024:
        pass                 # already (T, 1024)
    else:
        raise ValueError(f"Expected one dimension to be 1024, got shape {x.shape}")

    if x.shape[1] != 1024:
        raise ValueError(f"After processing, expected shape (T, 1024), got {x.shape}")

    return x

class Task3CISLRDataset(Dataset):
    """
    Dataset for Task-2 CISLR using precomputed I3D features.
    
    Expected split csv to contain at least:
        - uid
        - gloss

    Expected features pickle to contain at least:
        - uid or id
        - i3d_feature (stored as a string representation of a numpy array)
    """
    def __init__(self, split_csv: str | Path,
                 features_pkl: str | Path,
                 uid_col: str = "uid",
                 gloss_col: str = "gloss",
                 feature_col: str = "I3D_features") -> None:
        self.split_csv = Path(split_csv)
        self.features_pkl = Path(features_pkl)
        self.uid_col = uid_col
        self.gloss_col = gloss_col
        self.feature_col = feature_col

        self.df = pd.read_csv(self.split_csv)
        self.features_df = pd.read_pickle(self.features_pkl)
        required_cols = {self.uid_col, self.gloss_col}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"Split CSV must contain columns: {required_cols}")

        features_df = pd.read_pickle(self.features_pkl)

        if "id" in features_df.columns and "uid" not in features_df.columns:
            features_df = features_df.rename(columns={"id": self.uid_col})
        
        required_feature_cols = {self.uid_col, self.feature_col}
        missing_feature_cols = required_feature_cols - set(features_df.columns)
        if missing_feature_cols:
            raise ValueError(f"Features pickle must contain columns: {required_feature_cols}, missing: {missing_feature_cols}")
        
        self.feature_map = dict(zip(features_df[self.uid_col], features_df[self.feature_col]))

        missing_uids = sorted(set(self.df[self.uid_col]) - set(self.feature_map.keys()))
        if missing_uids:
            preview = missing_uids[:10]
            raise ValueError(f"UIDs in split CSV not found in features: {len(missing_uids)} missing, preview: {preview}")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        uid = row[self.uid_col]
        gloss = row[self.gloss_col]

        raw_feature = self.feature_map[uid]
        feature_np = reshape_i3d_feature(raw_feature)
        feature_tensor = torch.from_numpy(feature_np).float()

        return {
            "uid": uid,
            "gloss": gloss,
            "features": feature_tensor,
            "feat_len": feature_tensor.shape[0],
        }

        
def task3_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Pad variable-length feature sequences to (B, Tmax, 1024).
    """
    uids = [item["uid"] for item in batch]
    glosses = [item["gloss"] for item in batch]
    features = [item["features"] for item in batch]
    feat_lens = torch.tensor([item["feat_len"] for item in batch], dtype=torch.long)

    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    return {
        "uid": uids,
        "gloss": glosses,
        "features": padded_features,   # (B, Tmax, 1024)
        "feat_len": feat_lens,
    }