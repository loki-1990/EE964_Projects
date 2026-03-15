from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.tasks.task3_cislr.data.dataset import Task3CISLRDataset, task3_collate_fn
from src.tasks.task3_cislr.utils.retrieval_utils import build_embedding_bank, compute_similarity, evaluate_topk


def make_loader(
    split_csv: str | Path,
    feature_pkl: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = Task3CISLRDataset(
        split_csv=split_csv,
        features_pkl=feature_pkl,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=task3_collate_fn,
    )


def run_baseline(
    train_csv: str | Path,
    query_csv: str | Path,
    feature_pkl: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(
        split_csv=train_csv,
        feature_pkl=feature_pkl,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    query_loader = make_loader(
        split_csv=query_csv,
        feature_pkl=feature_pkl,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    bank = build_embedding_bank(train_loader, device)
    query = build_embedding_bank(query_loader, device)

    sim = compute_similarity(query["embeddings"], bank["embeddings"])
    metrics = evaluate_topk(
        sim=sim,
        query_glosses=query["glosses"],
        bank_glosses=bank["glosses"],
        ks=(1, 5, 10),
    )

    results = {
        "train_csv": str(train_csv),
        "query_csv": str(query_csv),
        "feature_pkl": str(feature_pkl),
        "num_bank_samples": len(bank["uids"]),
        "num_query_samples": len(query["uids"]),
        "embedding_dim": int(bank["embeddings"].shape[1]),
        "metrics": metrics,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def main() -> None:
    root = Path(".")

    feature_pkl = root / "dataset/task3_cislr/features/I3D_features.pkl"
    train_csv = root / "dataset/task3_cislr/splits/train.csv"
    val_csv = root / "dataset/task3_cislr/splits/val.csv"
    test_csv = root / "dataset/task3_cislr/splits/test.csv"

    val_save = root / "results/task3_cislr/baselines/i3d_nn_val.json"
    test_save = root / "results/task3_cislr/baselines/i3d_nn_test.json"

    print("\nRunning baseline: train bank -> val queries")
    val_results = run_baseline(
        train_csv=train_csv,
        query_csv=val_csv,
        feature_pkl=feature_pkl,
        batch_size=64,
        num_workers=0,
        save_path=val_save,
    )
    print(json.dumps(val_results, indent=2))

    print("\nRunning baseline: train bank -> test queries")
    test_results = run_baseline(
        train_csv=train_csv,
        query_csv=test_csv,
        feature_pkl=feature_pkl,
        batch_size=64,
        num_workers=0,
        save_path=test_save,
    )
    print(json.dumps(test_results, indent=2))


if __name__ == "__main__":
    main()