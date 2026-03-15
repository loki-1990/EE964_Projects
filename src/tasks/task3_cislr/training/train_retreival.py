from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.scripts.get_device import get_device

from src.tasks.task3_cislr.data.dataset import Task3CISLRDataset, task3_collate_fn
from src.tasks.task3_cislr.utils.retrieval_utils import (
    build_embedding_bank,
    compute_similarity,
    evaluate_topk,
)
from src.tasks.task3_cislr.data.samplers import build_grouped_batch_sampler


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


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    glosses: list[str],
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss using same-gloss samples as positives.

    Args:
        embeddings: (B, D), assumed L2-normalized
        glosses: list of gloss strings of length B
        temperature: temperature scaling for similarity

    Returns:
        scalar loss
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    # similarity matrix
    sim = embeddings @ embeddings.T                      # (B, B)
    sim = sim / temperature

    # remove self-comparisons later
    logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

    # positive-pair mask: same gloss but not self
    labels = glosses
    pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j and labels[i] == labels[j]:
                pos_mask[i, j] = True

    # if a sample has no positive inside the batch, it should not contribute
    valid_anchor = pos_mask.sum(dim=1) > 0
    if valid_anchor.sum() == 0:
        # no positive pairs in the batch
        return torch.tensor(0.0, device=device, requires_grad=True)

    # numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - sim_max.detach()

    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # average log-prob over positives for each anchor
    pos_mask_f = pos_mask.float()
    mean_log_prob_pos = (pos_mask_f * log_prob).sum(dim=1) / (pos_mask_f.sum(dim=1) + 1e-12)

    loss = -mean_log_prob_pos[valid_anchor].mean()
    return loss


@torch.no_grad()
def evaluate_retrieval(
    model: torch.nn.Module,
    bank_loader: DataLoader,
    query_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    bank = build_embedding_bank(bank_loader, device=device, model=model)
    query = build_embedding_bank(query_loader, device=device, model=model)

    sim = compute_similarity(query["embeddings"], bank["embeddings"])
    metrics = evaluate_topk(
        sim=sim,
        query_glosses=query["glosses"],
        bank_glosses=bank["glosses"],
        ks=(1, 5, 10),
    )
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0

    for batch in loader:
        features = batch["features"].to(device)
        feat_lens = batch["feat_len"].to(device)
        glosses = batch["gloss"]

        optimizer.zero_grad()

        embeddings = model(features, feat_lens)    # (B, D)
        loss = supervised_contrastive_loss(
            embeddings=embeddings,
            glosses=glosses,
            temperature=temperature,
        )

        # skip optimizer step only if batch had no positive pairs
        if torch.isfinite(loss) and loss.item() > 0:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


def run_training(
    train_csv: str | Path,
    val_csv: str | Path,
    prototype_csv: str | Path,
    test_csv: str | Path,
    feature_pkl: str | Path,
    out_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    pooling: str = "max",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
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

    # train loader
    
    train_dataset = Task3CISLRDataset(
        split_csv=train_csv,
        features_pkl=feature_pkl,
    )

    train_sampler = build_grouped_batch_sampler(
        split_csv=train_csv,
        n_glosses_per_batch=8,
        n_samples_per_gloss=2,
        seed=1990,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=task3_collate_fn,
    )

    train_eval_loader = make_loader(
        split_csv=train_csv,
        feature_pkl=feature_pkl,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # retrieval eval loaders
    prototype_loader = make_loader(
        split_csv=prototype_csv,
        feature_pkl=feature_pkl,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loader = make_loader(
        split_csv=val_csv,
        feature_pkl=feature_pkl,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = make_loader(
        split_csv=test_csv,
        feature_pkl=feature_pkl,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    best_val_top1 = -1.0
    best_val_metrics = None
    epochs_without_improvement = 0
    start_epoch = 1

    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        history = ckpt.get("history", [])
        best_val_top1 = ckpt.get("best_val_top1", -1.0)
        best_val_metrics = ckpt.get("best_val_metrics", None)

    model_name = model.__class__.__name__
    hidden = getattr(model, "hidden_dim", "NA")
    out = getattr(model, "out_dim", "NA")
    ckpt_name = f"{model_name}_h{hidden}_o{out}_lr{lr}_temp{temperature}_best.pt"
    best_ckpt = out_dir / ckpt_name

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=temperature,
        )

        val_metrics = evaluate_retrieval(
            model=model,
            bank_loader=prototype_loader,
            query_loader=val_loader,
            device=device,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "val_top10": val_metrics["top10"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val_top1={val_metrics['top1']:.2f} | "
            f"val_top5={val_metrics['top5']:.2f} | "
            f"val_top10={val_metrics['top10']:.2f}"
        )

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_val_metrics = val_metrics.copy()
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "history": history,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_top1": best_val_top1,
                    "best_val_metrics": best_val_metrics,
                    "config": {
                        "hidden_dim": getattr(model, "hidden_dim", None),
                        "out_dim": getattr(model, "out_dim", None),
                        "pooling": pooling,
                        "num_layers": getattr(model, "num_layers", None),
                        "dropout": getattr(model, "dropout_p", None),
                    },
                },
                best_ckpt,
            )
        else:
            epochs_without_improvement += 1
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # load best model for final test evaluation
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_retrieval(
        model=model,
        bank_loader=prototype_loader,
        query_loader=test_loader,
        device=device,
    )

    results = {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "prototype_csv": str(prototype_csv),
        "test_csv": str(test_csv),
        "feature_pkl": str(feature_pkl),
        "device": str(device),
        "config": {
            "model_name": model.__class__.__name__,
            "model_repr": str(model),
            "batch_size": batch_size,
            "hidden_dim": getattr(model, "hidden_dim", None),
            "input_dim": getattr(model, "input_dim", None),
            "out_dim": getattr(model, "out_dim", None),
            "pooling": pooling,
            "num_layers": getattr(model, "num_layers", None),
            "dropout": getattr(model, "dropout_p", None),
            "lr": lr,
            "weight_decay": weight_decay,
            "temperature": temperature,
            "epochs": epochs,
            "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        "best_val_top1": best_val_top1,
        "test_metrics": test_metrics,
        "best_val_metrics": best_val_metrics,
        "history": history,
    }

    with open(out_dir / f"results_{model_name}_h{hidden}_o{out}_lr{lr}_temp{temperature}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nBest validation top1:", best_val_top1)
    print("Test metrics:", test_metrics)

    return results

