# Save as:
# src/tasks/task5_semantic_similarity/training/grid_search.py

from pathlib import Path
import json
import copy
import torch
from torch.utils.data import DataLoader
from src.utils.expand_grid import expand_grid
from src.tasks.task5_semantic_similarity.data.dataset import Task5PairDataset
from src.tasks.task5_semantic_similarity.models.meanpool_temporal import MeanPoolTemporalEncoder
from src.tasks.task5_semantic_similarity.models.small_transformer import SmallTransformerEncoder
from src.tasks.task5_semantic_similarity.training.train_meanpool_with_checkpoint import (
    train_meanpool_with_checkpoint,
)
from src.tasks.task5_semantic_similarity.training.train_small_transformer_with_checkpoint import (
    train_small_transformer_with_checkpoint,
)
from src.scripts.get_device import get_device





def build_dataloaders(
    train_pos,
    val_pos,
    pose_dir,
    batch_size: int,
    num_workers: int = 2,
):
    train_dataset = Task5PairDataset(train_pos, pose_dir)
    val_dataset = Task5PairDataset(val_pos, pose_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader




def build_meanpool_model(cfg: dict, device: str):
    model = MeanPoolTemporalEncoder(
        input_dim=cfg.get("input_dim", 225),
        frame_hidden_dim=cfg["frame_hidden_dim"],
        output_dim=cfg["output_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    return model


def build_transformer_model(cfg: dict, device: str):
    model = SmallTransformerEncoder(
        input_dim=cfg.get("input_dim", 225),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        ff_mult=cfg["ff_mult"],
        output_dim=cfg["output_dim"],
        dropout=cfg["dropout"],
        max_len=cfg.get("max_len", 400),
    ).to(device)
    return model


def train_one_config(
    model_type: str,
    cfg: dict,
    train_pos,
    val_pos,
    pose_dir: str,
    results_root: str,
    num_epochs: int = 20,
    patience: int = 5,
    num_workers: int = 2,
):
    device = get_device()
    batch_size = cfg["batch_size"]

    train_loader, val_loader = build_dataloaders(
        train_pos=train_pos,
        val_pos=val_pos,
        pose_dir=pose_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if model_type == "meanpool":
        model = build_meanpool_model(cfg, device)
    elif model_type == "transformer":
        model = build_transformer_model(cfg, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    run_name = make_run_name(model_type, cfg)
    save_dir = Path(results_root) / model_type / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    if model_type == "meanpool":
        result = train_meanpool_with_checkpoint(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            save_dir=save_dir,
            num_epochs=num_epochs,
            temperature=cfg["temperature"],
            grad_clip=cfg["grad_clip"],
            patience=patience,
        )
    else:
        result = train_small_transformer_with_checkpoint(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            save_dir=save_dir,
            num_epochs=num_epochs,
            temperature=cfg["temperature"],
            grad_clip=cfg["grad_clip"],
            patience=patience,
        )

    summary = {
        "model_type": model_type,
        "run_name": run_name,
        "config": copy.deepcopy(cfg),
        "best_epoch": result["best_epoch"],
        "best_metrics": result["best_metrics"],
        "save_dir": str(save_dir),
        "device": device,
    }

    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def make_run_name(model_type: str, cfg: dict) -> str:
    if model_type == "meanpool":
        return (
            f"fh{cfg['frame_hidden_dim']}_"
            f"out{cfg['output_dim']}_"
            f"do{str(cfg['dropout']).replace('.', 'p')}_"
            f"lr{str(cfg['lr']).replace('.', 'p')}_"
            f"wd{str(cfg['weight_decay']).replace('.', 'p')}_"
            f"bs{cfg['batch_size']}"
        )

    return (
        f"dm{cfg['d_model']}_"
        f"nh{cfg['nhead']}_"
        f"nl{cfg['num_layers']}_"
        f"ff{cfg['ff_mult']}_"
        f"out{cfg['output_dim']}_"
        f"do{str(cfg['dropout']).replace('.', 'p')}_"
        f"lr{str(cfg['lr']).replace('.', 'p')}_"
        f"wd{str(cfg['weight_decay']).replace('.', 'p')}_"
        f"bs{cfg['batch_size']}"
    )


def sort_summaries(summaries: list[dict]) -> list[dict]:
    return sorted(
        summaries,
        key=lambda x: (
            -x["best_metrics"]["Top5%"],
            x["best_metrics"]["MeanRank"],
            -x["best_metrics"]["MRR"],
        ),
    )


def run_grid_search(
    model_type: str,
    param_grid: dict,
    train_pos,
    val_pos,
    pose_dir: str,
    results_root: str,
    num_epochs: int = 20,
    patience: int = 5,
    num_workers: int = 2,
):
    configs = expand_grid(param_grid)
    summaries = []

    print(f"Running {len(configs)} configs for model_type={model_type}")

    for i, cfg in enumerate(configs, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(configs)}] model_type={model_type}")
        print(cfg)

        summary = train_one_config(
            model_type=model_type,
            cfg=cfg,
            train_pos=train_pos,
            val_pos=val_pos,
            pose_dir=pose_dir,
            results_root=results_root,
            num_epochs=num_epochs,
            patience=patience,
            num_workers=num_workers,
        )
        summaries.append(summary)

        bm = summary["best_metrics"]
        print(
            f"BEST | epoch={summary['best_epoch']} | "
            f"Top5%={bm['Top5%']:.4f} | "
            f"MeanRank={bm['MeanRank']:.4f} | "
            f"MRR={bm['MRR']:.4f}"
        )

    ranked = sort_summaries(summaries)

    out_dir = Path(results_root) / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "grid_search_results.json", "w") as f:
        json.dump(ranked, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL RANKING")
    for j, item in enumerate(ranked, start=1):
        bm = item["best_metrics"]
        print(
            f"{j:02d}. {item['run_name']} | "
            f"Top5%={bm['Top5%']:.4f} | "
            f"MeanRank={bm['MeanRank']:.4f} | "
            f"MRR={bm['MRR']:.4f}"
        )

    return ranked