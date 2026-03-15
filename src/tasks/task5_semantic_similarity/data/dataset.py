import numpy as np
import torch
from torch.utils.data import Dataset

from pose_format import Pose

MAX_WORD_FRAMES = 200
MAX_DESC_FRAMES = 400

SELECTED_KEYPOINTS = list(range(75))   # body + hands
NUM_KEYPOINTS = len(SELECTED_KEYPOINTS)
NUM_DIMS = 3
FEATURE_DIM = NUM_KEYPOINTS * NUM_DIMS   # 225


def load_pose(filepath, max_frames):
    with open(filepath, "rb") as f:
        pose = Pose.read(f.read())

    data = np.array(pose.body.data)[:, 0, :, :]

    if data.shape[1] >= 75:
        data = data[:, SELECTED_KEYPOINTS, :]

    frames = data.shape[0]
    data = data.reshape(frames, -1)
    data = np.nan_to_num(data, 0.0)

    # simple per-sample normalization
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - mean) / std

    # pad or truncate
    if frames >= max_frames:
        data = data[:max_frames]
        mask = np.ones(max_frames, dtype=np.float32)
    else:
        pad = np.zeros((max_frames - frames, data.shape[1]), dtype=np.float32)
        data = np.vstack([data, pad])
        mask = np.zeros(max_frames, dtype=np.float32)
        mask[:frames] = 1.0

    return data.astype(np.float32), mask


class Task5PairDataset(Dataset):
    def __init__(self, pairs_df, pose_dir,
                 max_word_frames=MAX_WORD_FRAMES,
                 max_desc_frames=MAX_DESC_FRAMES):
        self.pairs = pairs_df.reset_index(drop=True)
        self.pose_dir = pose_dir
        self.max_word_frames = max_word_frames
        self.max_desc_frames = max_desc_frames

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        word_path = f"{self.pose_dir}/{row['word_id']}.pose"
        desc_path = f"{self.pose_dir}/{row['sentence_id']}.pose"

        word_data, word_mask = load_pose(word_path, self.max_word_frames)
        desc_data, desc_mask = load_pose(desc_path, self.max_desc_frames)

        return {
            "word_id": row["word_id"],
            "sentence_id": row["sentence_id"],
            "word_data": torch.FloatTensor(word_data),
            "word_mask": torch.FloatTensor(word_mask),
            "desc_data": torch.FloatTensor(desc_data),
            "desc_mask": torch.FloatTensor(desc_mask),
        }