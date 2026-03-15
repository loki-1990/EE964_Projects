from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.pose_utils import load_and_normalize_pose


def collect_unique_ids(csv_paths: list[str | Path]) -> list[str]:
    ids = set()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        if "word_id" in df.columns:
            ids.update(df["word_id"].dropna().astype(str).tolist())

        if "sentence_id" in df.columns:
            ids.update(df["sentence_id"].dropna().astype(str).tolist())

    return sorted(ids)


def precompute_pose_npy(
    csv_paths: list[str | Path],
    pose_dir: str | Path,
    out_dir: str | Path,
    overwrite: bool = False,
) -> dict:
    pose_dir = Path(pose_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ids = collect_unique_ids(csv_paths)

    num_done = 0
    num_skipped = 0
    num_failed = 0
    failures = []

    for vid in tqdm(all_ids, desc="Precomputing pose npy"):
        src_path = pose_dir / f"{vid}.pose"
        dst_path = out_dir / f"{vid}.npy"

        if dst_path.exists() and not overwrite:
            num_skipped += 1
            continue

        try:
            x = load_and_normalize_pose(src_path)   # (T, 1728)
            x = np.asarray(x, dtype=np.float32)
            np.save(dst_path, x)
            num_done += 1
        except Exception as e:
            num_failed += 1
            failures.append({"id": vid, "error": str(e)})

    summary = {
        "num_total_ids": len(all_ids),
        "num_done": num_done,
        "num_skipped": num_skipped,
        "num_failed": num_failed,
        "failures": failures,
    }

    with open(out_dir / "precompute_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary