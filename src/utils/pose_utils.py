from __future__ import annotations

from pathlib import Path
import numpy as np
from pose_format import Pose


def load_pose_file(pose_path: str | Path) -> Pose:
    """
    Load a .pose file and return a Pose object.
    """
    pose_path = Path(pose_path)
    with open(pose_path, "rb") as f:
        pose = Pose.read(f)
    return pose


def pose_to_frame_array(pose: Pose) -> np.ndarray:
    """
    Convert a Pose object to a 2D framewise feature array.

    Input pose body shape:
        (T, 1, J, C)

    Returns:
        x: np.ndarray of shape (T, J*C)
    """
    x = pose.body.data.squeeze(1)   # (T, J, C)
    t = x.shape[0]
    x = x.reshape(t, -1).astype(np.float32)
    return x


def load_pose_as_array(pose_path: str | Path) -> np.ndarray:
    """
    Load a .pose file and return a flattened framewise array.

    Returns:
        x: np.ndarray of shape (T, D)
    """
    pose = load_pose_file(pose_path)
    return pose_to_frame_array(pose)


def normalize_pose_per_video(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-video z-normalization over the temporal axis.

    Args:
        x: np.ndarray of shape (T, D)
        eps: small constant to avoid divide-by-zero

    Returns:
        normalized x of shape (T, D)
    """
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / (std + eps)
    return x.astype(np.float32)


def load_and_normalize_pose(pose_path: str | Path, eps: float = 1e-6) -> np.ndarray:
    """
    Convenience function:
    load .pose -> flatten -> per-video normalize

    Returns:
        x: np.ndarray of shape (T, D)
    """
    x = load_pose_as_array(pose_path)
    x = normalize_pose_per_video(x, eps=eps)
    return x