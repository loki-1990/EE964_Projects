from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.scripts.get_device import get_device
from src.tasks.task4_word_presence.data.dataset import (
    Task4WordPresenceDataset,
    task4_collate_fn,
)
from src.tasks.task4_word_presence.utils.metric_calculation import (
    evaluate_top_rank_avg,
)


def make_loader(
    split_csv: str | Path,
    feature_source,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = Task4WordPresenceDataset(
        split_csv=split_csv,
        feature_source=feature_source,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=task4_collate_fn,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        word_x = batch["word_x"].to(device, non_blocking=True)
        word_len = batch["word_len"].to(device, non_blocking=True)
        sent_x = batch["sent_x"].to(device, non_blocking=True)
        sent_len = batch["sent_len"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        logits, _ = model(word_x, word_len, sent_x, sent_len)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).float()

        total_loss += loss.item() * labels.size(0)
        total += labels.numel()
        correct += (preds == labels).sum().item()

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }


@torch.no_grad()
def evaluate_classification(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        word_x = batch["word_x"].to(device, non_blocking=True)
        word_len = batch["word_len"].to(device, non_blocking=True)
        sent_x = batch["sent_x"].to(device, non_blocking=True)
        sent_len = batch["sent_len"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits, _ = model(word_x, word_len, sent_x, sent_len)
        loss = criterion(logits, labels)

        preds = (torch.sigmoid(logits) >= 0.5).float()

        total_loss += loss.item() * labels.size(0)
        total += labels.numel()
        correct += (preds == labels).sum().item()

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }


def run_training(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    feature_source,
    out_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    early_stopping_patience: int | None = 5,
    resume_ckpt: str | Path | None = None,
    model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    device = get_device()

    if model is None:
        raise ValueError("run_training() requires a model instance, but got None.")

    model = model.to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        split_csv=train_csv,
        feature_source=feature_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = make_loader(
        split_csv=val_csv,
        feature_source=feature_source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    test_loader = make_loader(
        split_csv=test_csv,
        feature_source=feature_source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    best_val_accuracy = -1.0
    best_val_metrics = None
    best_val_rank_metrics = None
    epochs_without_improvement = 0
    start_epoch = 1

    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        history = ckpt.get("history", [])
        best_val_rank_metrics = ckpt.get("best_val_rank_metrics", None)
        best_val_accuracy = ckpt.get("best_val_accuracy", -1.0)
        best_val_metrics = ckpt.get("best_val_metrics", None)

    model_name = model.__class__.__name__
    hidden = getattr(model.encoder, "hidden_dim", getattr(model, "hidden_dim", "NA")) if hasattr(model, "encoder") else getattr(model, "hidden_dim", "NA")
    out = getattr(model.encoder, "output_dim", getattr(model, "output_dim", "NA")) if hasattr(model, "encoder") else getattr(model, "output_dim", "NA")
    pool = getattr(model.encoder, "pool", getattr(model, "pool", "NA")) if hasattr(model, "encoder") else getattr(model, "pool", "NA")

    ckpt_name = f"{model_name}_pool{pool}_h{hidden}_o{out}_lr{lr}_best.pt"
    best_ckpt = out_dir / ckpt_name

    best_val_rank_metrics = None

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate_classification(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        improved = val_metrics["accuracy"] > best_val_accuracy

        if improved:
            val_rank_metrics = evaluate_top_rank_avg(
                model=model,
                split_csv=val_csv,
                feature_source=feature_source,
                batch_size=batch_size,
                device=device,
            )
            current_val_avg_rank = val_rank_metrics["avg_rank"]
        else:
            val_rank_metrics = None
            current_val_avg_rank = None

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_avg_rank": current_val_avg_rank,
        }
        history.append(record)

        if current_val_avg_rank is None:
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['accuracy']:.2f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.2f}"
            )
        else:
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['accuracy']:.2f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.2f} | "
                f"val_avg_rank={current_val_avg_rank:.2f}"
            )

        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_metrics = val_metrics.copy()
            best_val_rank_metrics = val_rank_metrics
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "history": history,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_metrics": best_val_metrics,
                    "best_val_rank_metrics": best_val_rank_metrics,
                    "config": {
                        "input_dim": getattr(model.encoder, "input_dim", getattr(model, "input_dim", None)) if hasattr(model, "encoder") else getattr(model, "input_dim", None),
                        "hidden_dim": hidden,
                        "output_dim": out,
                        "pool": pool,
                        "dropout": getattr(model.encoder, "dropout_p", getattr(model, "dropout_p", None)) if hasattr(model, "encoder") else getattr(model, "dropout_p", None),
                    },
                },
                best_ckpt,
            )
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_classification(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    test_rank_metrics = evaluate_top_rank_avg(
        model=model,
        split_csv=test_csv,
        feature_source=feature_source,
        batch_size=batch_size,
        device=device,
    )

    results = {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "feature_source": (
            str(feature_source)
            if not isinstance(feature_source, dict)
            else "in_memory_feature_dict"
        ),
        "device": str(device),
        "config": {
            "model_name": model.__class__.__name__,
            "model_repr": str(model),
            "batch_size": batch_size,
            "input_dim": getattr(model.encoder, "input_dim", getattr(model, "input_dim", None)) if hasattr(model, "encoder") else getattr(model, "input_dim", None),
            "hidden_dim": hidden,
            "output_dim": out,
            "pool": pool,
            "dropout": getattr(model.encoder, "dropout_p", getattr(model, "dropout_p", None)) if hasattr(model, "encoder") else getattr(model, "dropout_p", None),
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "best_val_accuracy": best_val_accuracy,
        "best_val_metrics": best_val_metrics,
        "val_rank_metrics": best_val_rank_metrics,
        "test_metrics": test_metrics,
        "test_rank_metrics": test_rank_metrics,
        "history": history,
    }

    with open(
        out_dir / f"results_{model_name}_pool{pool}_h{hidden}_o{out}_lr{lr}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)

    print("\nBest validation accuracy:", best_val_accuracy)
    print("Validation rank metrics:", best_val_rank_metrics)
    print("Test metrics:", test_metrics)
    print("Test rank metrics:", test_rank_metrics)

    return results