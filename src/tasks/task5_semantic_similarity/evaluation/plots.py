from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_test_metric_bars(results_df, save_dir):
    """
    results_df columns expected:
    ['Model', 'Top5%', 'MeanRank', 'MRR']
    """
    ensure_dir(save_dir)

    models = results_df["Model"].tolist()

    # 1) Top-5%
    plt.figure(figsize=(8, 5))
    plt.bar(models, results_df["Top5%"])
    plt.ylabel("Top-5%")
    plt.title("Task-5 Test: Top-5% Retrieval Accuracy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "test_top5_bar.png", dpi=300)
    plt.close()

    # 2) Mean Rank
    plt.figure(figsize=(8, 5))
    plt.bar(models, results_df["MeanRank"])
    plt.ylabel("Mean Rank (lower is better)")
    plt.title("Task-5 Test: Mean Rank")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "test_meanrank_bar.png", dpi=300)
    plt.close()

    # 3) MRR
    plt.figure(figsize=(8, 5))
    plt.bar(models, results_df["MRR"])
    plt.ylabel("MRR")
    plt.title("Task-5 Test: Mean Reciprocal Rank")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "test_mrr_bar.png", dpi=300)
    plt.close()


def plot_val_vs_test_top5(val_test_df, save_dir):
    """
    val_test_df columns expected:
    ['Model', 'ValTop5', 'TestTop5']
    """
    ensure_dir(save_dir)

    models = val_test_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, val_test_df["ValTop5"], width, label="Validation")
    plt.bar(x + width / 2, val_test_df["TestTop5"], width, label="Test")

    plt.xticks(x, models, rotation=15)
    plt.ylabel("Top-5%")
    plt.title("Validation vs Test Top-5%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "val_vs_test_top5.png", dpi=300)
    plt.close()


def plot_history_curves(history_json_path, save_dir, prefix="meanpool_best"):
    """
    history_json_path: path to history.json from a checkpointed run
    """
    ensure_dir(save_dir)

    with open(history_json_path, "r") as f:
        history = json.load(f)

    df = pd.DataFrame(history)

    # Train loss
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{prefix}_train_loss.png", dpi=300)
    plt.close()

    # Validation Top-5%
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["Top5%"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Top-5%")
    plt.title("Validation Top-5% Curve")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{prefix}_val_top5.png", dpi=300)
    plt.close()

    # Validation MeanRank
    if "MeanRank" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["MeanRank"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Mean Rank")
        plt.title("Validation Mean Rank Curve")
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"{prefix}_val_meanrank.png", dpi=300)
        plt.close()


def make_meanpool_heatmap_from_ranked(meanpool_ranked, save_dir):
    """
    meanpool_ranked: list of dicts from grid search results json
    Produces one heatmap per frame_hidden_dim using:
      rows = lr
      cols = dropout
      values = Top5%
    """
    ensure_dir(save_dir)

    records = []
    for item in meanpool_ranked:
        cfg = item["config"]
        metrics = item["best_metrics"]
        records.append({
            "frame_hidden_dim": cfg["frame_hidden_dim"],
            "lr": cfg["lr"],
            "dropout": cfg["dropout"],
            "Top5%": metrics["Top5%"],
        })

    df = pd.DataFrame(records)

    for fh in sorted(df["frame_hidden_dim"].unique()):
        sub = df[df["frame_hidden_dim"] == fh].copy()

        pivot = sub.pivot_table(
            index="lr",
            columns="dropout",
            values="Top5%",
            aggfunc="max"
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x) for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(x) for x in pivot.index])

        ax.set_xlabel("Dropout")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"MeanPool Grid Search Heatmap (frame_hidden_dim={fh})")

        # annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center")

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"meanpool_heatmap_fh{fh}.png", dpi=300)
        plt.close()


def make_transformer_heatmap_from_ranked(transformer_ranked, save_dir):
    """
    transformer_ranked: list of dicts from grid search results json

    Produces one heatmap per (d_model, nhead, num_layers, ff_mult)
    using:
      rows = lr
      cols = dropout
      values = Top5%
    """
    ensure_dir(save_dir)

    records = []
    for item in transformer_ranked:
        cfg = item["config"]
        metrics = item["best_metrics"]
        records.append({
            "d_model": cfg["d_model"],
            "nhead": cfg["nhead"],
            "num_layers": cfg["num_layers"],
            "ff_mult": cfg["ff_mult"],
            "lr": cfg["lr"],
            "dropout": cfg["dropout"],
            "Top5%": metrics["Top5%"],
        })

    df = pd.DataFrame(records)

    group_cols = ["d_model", "nhead", "num_layers", "ff_mult"]

    for keys, sub in df.groupby(group_cols):
        d_model, nhead, num_layers, ff_mult = keys

        pivot = sub.pivot_table(
            index="lr",
            columns="dropout",
            values="Top5%",
            aggfunc="max"
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x) for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(x) for x in pivot.index])

        ax.set_xlabel("Dropout")
        ax.set_ylabel("Learning Rate")
        ax.set_title(
            f"Transformer Heatmap\n"
            f"d_model={d_model}, nhead={nhead}, layers={num_layers}, ff_mult={ff_mult}"
        )

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center")

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        fname = f"transformer_heatmap_dm{d_model}_nh{nhead}_nl{num_layers}_ff{ff_mult}.png"
        plt.savefig(Path(save_dir) / fname, dpi=300)
        plt.close()

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_similarity_heatmap(model, loader, device, save_path, max_queries=30):
    """
    Plot similarity matrix between query embeddings and candidate embeddings.

    Args:
        model : trained model
        loader : dataloader (val or test)
        device : cpu / cuda / mps
        save_path : where to save figure
        max_queries : limit heatmap size for readability
    """

    model.eval()

    word_embeddings = []
    desc_embeddings = []

    with torch.no_grad():
        for batch in loader:

            word_data = batch["word_data"].to(device)
            word_mask = batch["word_mask"].to(device)

            desc_data = batch["desc_data"].to(device)
            desc_mask = batch["desc_mask"].to(device)

            word_emb = model(word_data, word_mask)
            desc_emb = model(desc_data, desc_mask)

            word_embeddings.append(word_emb.cpu())
            desc_embeddings.append(desc_emb.cpu())

    word_embeddings = torch.cat(word_embeddings)
    desc_embeddings = torch.cat(desc_embeddings)

    # cosine similarity
    sim_matrix = torch.matmul(word_embeddings, desc_embeddings.T).numpy()

    # reduce size for readability
    sim_matrix = sim_matrix[:max_queries, :max_queries]

    plt.figure(figsize=(8, 7))
    plt.imshow(sim_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Cosine Similarity")

    plt.xlabel("Candidate Descriptions")
    plt.ylabel("Query Words")
    plt.title("Retrieval Similarity Heatmap")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def plot_similarity_heatmap_generic(model, loader, device, save_path, max_queries=30, title="Retrieval Similarity Heatmap"):
    """
    Generic similarity heatmap for any model with signature:
        model(x, mask) -> embedding

    loader must yield:
        word_data, word_mask, desc_data, desc_mask
    """
    model.eval()

    word_embeddings = []
    desc_embeddings = []

    for batch in loader:
        word_data = batch["word_data"].to(device)
        word_mask = batch["word_mask"].to(device)
        desc_data = batch["desc_data"].to(device)
        desc_mask = batch["desc_mask"].to(device)

        word_emb = model(word_data, word_mask)
        desc_emb = model(desc_data, desc_mask)

        word_embeddings.append(word_emb.cpu())
        desc_embeddings.append(desc_emb.cpu())

    word_embeddings = torch.cat(word_embeddings, dim=0)
    desc_embeddings = torch.cat(desc_embeddings, dim=0)

    sim_matrix = (word_embeddings @ desc_embeddings.T).numpy()
    sim_matrix = sim_matrix[:max_queries, :max_queries]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.imshow(sim_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Cosine Similarity")
    plt.xlabel("Candidate Descriptions")
    plt.ylabel("Query Words")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


@torch.no_grad()
def collect_retrieval_case_study(model, loader, device, top_k=5):
    """
    Build retrieval outputs for all queries in a split.

    Returns a list of dicts, one per query:
      {
        "word_id": ...,
        "true_sentence_id": ...,
        "true_rank": ...,
        "topk_ids": [...],
        "topk_scores": [...],
      }
    """
    model.eval()

    query_word_ids = []
    query_true_sentence_ids = []
    query_word_embs = []

    candidate_sentence_ids = []
    candidate_desc_embs = []
    seen_sentence_ids = set()

    for batch in loader:
        word_ids = batch["word_id"]
        sentence_ids = batch["sentence_id"]

        word_data = batch["word_data"].to(device)
        word_mask = batch["word_mask"].to(device)
        desc_data = batch["desc_data"].to(device)
        desc_mask = batch["desc_mask"].to(device)

        word_emb = model(word_data, word_mask)
        desc_emb = model(desc_data, desc_mask)

        for i in range(len(word_ids)):
            query_word_ids.append(word_ids[i])
            query_true_sentence_ids.append(sentence_ids[i])
            query_word_embs.append(word_emb[i].detach().cpu().numpy())

        for i in range(len(sentence_ids)):
            sid = sentence_ids[i]
            if sid not in seen_sentence_ids:
                seen_sentence_ids.add(sid)
                candidate_sentence_ids.append(sid)
                candidate_desc_embs.append(desc_emb[i].detach().cpu().numpy())

    query_word_embs = np.stack(query_word_embs, axis=0)
    candidate_desc_embs = np.stack(candidate_desc_embs, axis=0)

    sentence_id_to_index = {sid: idx for idx, sid in enumerate(candidate_sentence_ids)}

    cases = []

    for i in range(len(query_word_ids)):
        true_sid = query_true_sentence_ids[i]
        true_idx = sentence_id_to_index[true_sid]

        scores = candidate_desc_embs @ query_word_embs[i]
        ranked = np.argsort(-scores)

        true_rank = int(np.where(ranked == true_idx)[0][0]) + 1

        topk_idx = ranked[:top_k]
        topk_ids = [candidate_sentence_ids[j] for j in topk_idx]
        topk_scores = [float(scores[j]) for j in topk_idx]

        cases.append({
            "word_id": query_word_ids[i],
            "true_sentence_id": true_sid,
            "true_rank": true_rank,
            "topk_ids": topk_ids,
            "topk_scores": topk_scores,
        })

    return cases


def plot_retrieval_case_study(case, save_path, title_prefix="Retrieval Case Study"):
    """
    Make a report-friendly plot for one query:
      - left panel: query/ground-truth text
      - right panel: top-k retrieved candidates + scores
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    topk_ids = case["topk_ids"]
    topk_scores = case["topk_scores"]
    true_sid = case["true_sentence_id"]

    colors = ["tab:green" if sid == true_sid else "tab:blue" for sid in topk_ids]
    labels = []
    for rank, sid in enumerate(topk_ids, start=1):
        marker = "  ← TRUE" if sid == true_sid else ""
        labels.append(f"{rank}. {sid}{marker}")

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 2.0])

    ax_text = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    info_text = (
        f"Query word_id:\n{case['word_id']}\n\n"
        f"True sentence_id:\n{true_sid}\n\n"
        f"True rank: {case['true_rank']}"
    )

    ax_text.text(
        0.02, 0.95, info_text,
        va="top", ha="left", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="whitesmoke", edgecolor="gray")
    )
    ax_text.axis("off")

    y = np.arange(len(topk_ids))
    ax_bar.barh(y, topk_scores, color=colors)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Cosine Similarity")
    ax_bar.set_title(f"{title_prefix}: Top-{len(topk_ids)} Retrieved Candidates")

    for i, score in enumerate(topk_scores):
        ax_bar.text(score, i, f"  {score:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def pick_best_and_worst_cases(cases):
    """
    Returns:
      best_case  : true rank = 1, if available, otherwise best rank
      worst_case : largest true rank
    """
    sorted_cases = sorted(cases, key=lambda x: x["true_rank"])
    best_case = sorted_cases[0]
    worst_case = sorted(cases, key=lambda x: x["true_rank"], reverse=True)[0]
    return best_case, worst_case