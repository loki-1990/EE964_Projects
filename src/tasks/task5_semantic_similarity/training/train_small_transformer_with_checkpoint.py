from pathlib import Path
import json
import torch

from src.tasks.task5_semantic_similarity.training.train_small_transformer import (
    train_one_epoch_small_transformer,
)
from src.tasks.task5_semantic_similarity.evaluation.eval_small_transformer import (
    eval_small_transformer_retrieval,
)


def _is_better(curr_metrics, best_metrics):
    """
    Decide whether current checkpoint is better than best checkpoint.

    Priority:
    1. Higher Top-5%
    2. If tied, lower MeanRank
    """
    if best_metrics is None:
        return True

    if curr_metrics["Top5%"] > best_metrics["Top5%"]:
        return True

    if curr_metrics["Top5%"] == best_metrics["Top5%"]:
        if curr_metrics["MeanRank"] < best_metrics["MeanRank"]:
            return True

    return False


def train_small_transformer_with_checkpoint(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    save_dir,
    num_epochs: int = 20,
    temperature: float = 0.07,
    grad_clip: float | None = 1.0,
    patience: int = 5,
):
    """
    Train small transformer baseline with:
      - checkpoint saving on best validation Top-5%
      - tie-break using lower MeanRank
      - early stopping

    Saves:
      - best_model.pt
      - best_metrics.json
      - history.json
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_metrics = None
    best_epoch = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch_small_transformer(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=temperature,
            grad_clip=grad_clip,
        )

        val_metrics = eval_small_transformer_retrieval(
            model=model,
            val_loader=val_loader,
            device=device,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            **val_metrics,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"Top5%={val_metrics['Top5%']:.4f} | "
            f"MeanRank={val_metrics['MeanRank']:.4f} | "
            f"MRR={val_metrics['MRR']:.4f}"
        )

        if _is_better(val_metrics, best_metrics):
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, save_dir / "best_model.pt")

            with open(save_dir / "best_metrics.json", "w") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        **best_metrics,
                    },
                    f,
                    indent=2,
                )

            print("  -> saved new best checkpoint")

        else:
            no_improve += 1
            print(f"  -> no improvement ({no_improve}/{patience})")

        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
    }