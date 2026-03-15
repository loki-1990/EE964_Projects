from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.tasks.task3_cislr.data.dataset import Task3CISLRDataset, task3_collate_fn
from src.tasks.task3_cislr.utils.retrieval_utils import (
    build_embedding_bank,
    compute_similarity,
    evaluate_topk,
)


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


def run_multi_variant_baseline(
    train_csv: str | Path,
    query_csv: str | Path,
    feature_pkl: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    save_path: str | Path | None = None,
    pooling_variants: str | list[str] | tuple[str, ...] = ("mean", "max", "meanmax", "meanstd"),
) -> dict[str, Any]:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

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

    if isinstance(pooling_variants, str):
        pooling_variants = [pooling_variants]
    elif isinstance(pooling_variants, (list, tuple)):
        pooling_variants = list(pooling_variants)
    else:
        raise ValueError("pooling_variants must be a string, list, or tuple of strings.")

    results = {}

    for variant in pooling_variants:
        print(f"Running baseline with {variant} pooling...")

        train_bank = build_embedding_bank(
            train_loader,
            device=device,
            pooling_variant=variant,
        )
        query_bank = build_embedding_bank(
            query_loader,
            device=device,
            pooling_variant=variant,
        )

        sim = compute_similarity(query_bank["embeddings"], train_bank["embeddings"])
        metrics = evaluate_topk(
            sim=sim,
            query_glosses=query_bank["glosses"],
            bank_glosses=train_bank["glosses"],
            ks=(1, 5, 10),
        )

        results[variant] = {
            "num_bank_samples": len(train_bank["uids"]),
            "num_query_samples": len(query_bank["uids"]),
            "embedding_dim": train_bank["embeddings"].shape[1],
            "metrics": metrics,
        }

    output = {
        "train_csv": str(train_csv),
        "query_csv": str(query_csv),
        "feature_pkl": str(feature_pkl),
        "device": str(device),
        "results": results,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {save_path}")

    return output