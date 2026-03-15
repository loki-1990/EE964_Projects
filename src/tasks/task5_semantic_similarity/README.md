# Task-5: Semantic Similarity Retrieval using Pose Sequences

## Overview

This project investigates whether **pose sequences extracted from sign language videos encode sufficient semantic information** to support gesture retrieval.

The task is formulated as a **retrieval problem**:

> Given a pose sequence representing an **isolated word gesture**, the model must retrieve the **most semantically similar sentence gesture segment** from a set of candidates.

The system operates entirely on **pose features (`.pose` files)** without using **RGB frames or text inputs**.

---
## Repository Structure

```text
src/
└── tasks/
    └── task5_semantic_similarity/
        ├── data/
        │   └── dataset.py
        ├── models/
        │   ├── meanpool_temporal.py
        │   ├── small_transformer.py
        │   └── statpool_mlp.py
        ├── training/
        │   ├── train_meanpool_temporal.py
        │   ├── train_small_transformer_with_checkpoint.py
        │   └── grid_search.py
        ├── evaluation/
        │   ├── baselines.py
        │   ├── eval_meanpool_temporal.py
        │   ├── eval_small_transformer.py
        │   ├── eval_checkpoint.py
        │   └── plots.py
        └── utils/
            └── figure_utils.py
```


# Experiment outputs are stored in:
```text
results/task5_semantic_similarity
│
├── meanpool
├── transformer
└── figures
```

---

# Dataset

Each sample consists of **two pose sequences**:

- **Word Pose** – isolated word gesture  
- **Description Pose** – corresponding sentence gesture segment  

Each pose frame contains:

- **225-dimensional pose feature vector**

### Dataset Statistics

| Split | Queries |
|------|-------|
| Train | ~1037 |
| Validation | 58 |
| Test | 59 |

All sequences from the **same video are kept in the same split** to avoid **data leakage**.

---

# Models Implemented

## 1️⃣ Statistical Pooling Baseline

A simple baseline that **ignores temporal order**.

### Steps
```text
pose sequence
↓
mean pooling
std pooling
↓
concatenate features
↓
cosine similarity retrieval
```
---

## 2️⃣ MeanPool Temporal Encoder

A lightweight temporal model.

### Architecture
```text
Pose Frames (225D)
↓
Frame Projection (MLP)
↓
ReLU + Dropout
↓
Temporal Mean Pooling
↓
L2 Normalized Embedding
```
Training uses **contrastive InfoNCE loss**.

### Best Configuration
```text
frame_hidden_dim = 256
embedding_dim = 128
dropout = 0.1
learning_rate = 5e-4
```
---

## 3️⃣ Small Transformer Encoder

A transformer-based temporal model.

### Architecture
```text
Pose Frames
↓
Linear Projection
↓
Positional Encoding
↓
Transformer Encoder
↓
Mean Pooling
↓
Embedding
```
### Best Configuration
```text
d_model = 128
nhead = 4
num_layers = 1
dropout = 0.2
```
---

# Evaluation Metrics

Retrieval performance is measured using:

### 1️⃣ Top-5% Retrieval Accuracy

Percentage of queries whose correct match appears in the **top 5% of ranked candidates**.

---

### 2️⃣ Mean Rank

Average position of the correct match.

---

### 3️⃣ Median Rank

Median rank of the correct match.

---

### 4️⃣ Mean Reciprocal Rank (MRR)

![MRR formula](https://latex.codecogs.com/png.image?\dpi{120}MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i})

---

# Test Results

| Model | Top-5% | Mean Rank | Median Rank | MRR |
|------|------|------|------|------|
| StatPool | 0.0508 | 29.86 | 30 | 0.080 |
| MeanPool Temporal | **0.4915** | **7.58** | 4 | **0.442** |
| Small Transformer | 0.4407 | 7.73 | 4 | 0.398 |

---

# Key Findings

- **Temporal modeling is essential** for semantic gesture retrieval.
- Learned encoders significantly outperform **statistical pooling baselines**.
- The **MeanPool temporal encoder slightly outperforms the transformer model** on this dataset.

This likely occurs because the **dataset is relatively small**, making **simpler architectures easier to train**.

---

# Visualization

The repository includes visualization tools for:

- Retrieval similarity heatmaps
- Grid search heatmaps
- Training curves
- Retrieval case studies

Example figures are stored in:

results/task5_semantic_similarity/figures

---

# Running the Experiments

### Train MeanPool Model
train_meanpool_temporal.py

### Train Transformer Model
train_small_transformer_with_checkpoint.py

### Hyperparameter Search
grid_search.py

### Evaluate Model
eval_checkpoint.py

### Generate Plots
plots.py

---

# Example Retrieval Visualization

Example case study output:
```text
Query word gesture
↓
Top-5 retrieved sentence gestures
↓
Similarity scores
```
These visualizations help analyze **both successful retrievals and failure cases**.

---

# Conclusion

This project demonstrates that **pose sequences contain sufficient semantic information to support gesture retrieval tasks**.

A **lightweight MeanPool temporal encoder** provides the best performance on the current dataset, outperforming both **statistical baselines and transformer models**.