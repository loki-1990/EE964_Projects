from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Task4WordPresenceDataset(Dataset):
    """
    CSV columns expected:
        word_id, word, sentence_id, sentence, group_id, label
    """

    def __init__(
        self,
        split_csv: str | Path,
        feature_source,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.df = pd.read_csv(self.split_csv)

        self.feature_dir = None
        self.features = None

        if isinstance(feature_source, dict):
            self.features = feature_source
        else:
            self.feature_dir = Path(feature_source)

    def __len__(self) -> int:
        return len(self.df)

    def _load_feature_tensor(self, vid: str) -> torch.Tensor:
        if self.features is not None:
            x = self.features[vid]
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.tensor(x, dtype=torch.float32)

        x = np.load(self.feature_dir / f"{vid}.npy")
        return torch.tensor(x, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        word_id = row["word_id"]
        sentence_id = row["sentence_id"]
        label = float(row["label"])

        word_x = self._load_feature_tensor(word_id)
        sent_x = self._load_feature_tensor(sentence_id)

        return {
            "word_id": word_id,
            "sentence_id": sentence_id,
            "word_x": word_x,
            "sent_x": sent_x,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def _pad_sequence_list(seq_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([x.shape[0] for x in seq_list], dtype=torch.long)
    batch_size = len(seq_list)
    max_len = int(lengths.max().item())
    feat_dim = seq_list[0].shape[1]

    padded = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    for i, x in enumerate(seq_list):
        padded[i, : x.shape[0]] = x

    return padded, lengths


def task4_collate_fn(batch: list[dict]) -> dict:
    word_x_list = [item["word_x"] for item in batch]
    sent_x_list = [item["sent_x"] for item in batch]

    word_x, word_len = _pad_sequence_list(word_x_list)
    sent_x, sent_len = _pad_sequence_list(sent_x_list)

    labels = torch.stack([item["label"] for item in batch], dim=0)

    return {
        "word_id": [item["word_id"] for item in batch],
        "sentence_id": [item["sentence_id"] for item in batch],
        "word_x": word_x,
        "word_len": word_len,
        "sent_x": sent_x,
        "sent_len": sent_len,
        "label": labels,
    }