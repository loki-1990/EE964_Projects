---

# Pose-Based Sign Language Understanding

### EE964 Deep Learning Project — IIT Kanpur eMasters

This repository contains the implementation of a **pose-based sign language understanding system** developed for the **EE964 course project at IIT Kanpur (eMasters Program)**.

The project investigates how **deep learning models can learn meaningful representations from pose sequences extracted from sign language videos**, enabling tasks such as retrieval, word detection, and semantic similarity.

---

# Project Motivation

Sign language recognition is a challenging problem due to the complex **spatiotemporal structure of gestures**. Many approaches rely on raw video frames, which are computationally heavy and sensitive to visual noise.

This project explores an alternative approach: **pose-based representation learning**, where human joint coordinates extracted from videos are used as the primary input.

Advantages of pose-based modeling include:

* compact representation of motion
* reduced sensitivity to background variation
* interpretable joint trajectories
* lower computational cost compared to raw video processing

---

# Tasks Investigated

The project focuses on three sign language understanding tasks.

### Continuous Sign Language Retrieval (CISLR)

Learn embeddings for sign language videos such that **semantically similar signs are close in embedding space**.

Evaluation metrics include:

* Top-1 Accuracy
* Top-5 Accuracy
* Top-10 Accuracy
* Mean Reciprocal Rank (MRR)

---

### Word Presence Detection

Predict whether a given **word appears in a sign language video**.

This is formulated as a **multi-label classification task** over pose sequences.

---

### Semantic Similarity Prediction

Learn embeddings that capture **semantic similarity between sign language videos and textual descriptions**.

The objective is to map pose sequences into a **shared semantic embedding space**.

---

# Dataset and Pose Representation

Instead of raw RGB video, the dataset uses **pose representations extracted from sign language videos**.

Each pose file contains frame-wise keypoint coordinates describing:

* body joints
* hand landmarks
* facial keypoints

A pose sequence can be represented as a tensor:

```
(T, 1, 576, 3)
```

Where:

* **T** = number of frames
* **576** = number of detected keypoints
* **3** = spatial coordinates

These pose sequences are converted into **feature vectors** and passed into temporal models.

---

# Model Architectures

Several neural architectures were explored to model temporal pose sequences.

## Pooling-Based Baselines

Simple aggregation methods were used as initial baselines:

* Mean pooling
* Max pooling
* Statistical pooling + MLP

These approaches summarize the pose sequence into a fixed representation.

---

## GRU Encoder

A **Bidirectional GRU encoder** was implemented to model temporal dynamics.

Key advantages:

* captures sequential dependencies
* efficient training
* strong performance on medium-length sequences

---

## Transformer Encoder

A lightweight **Transformer-based temporal encoder** was also implemented.

Advantages include:

* attention-based temporal modeling
* ability to capture long-range dependencies
* scalable architecture

---

# Training Setup

All models were implemented using **PyTorch**.

Key components of the training pipeline:

**Optimizer**

* Adam

**Regularization**

* dropout
* weight decay

**Training strategy**

* validation-based model selection
* consistent training framework across tasks

Training was performed on **Apple Silicon using Metal Performance Shaders (MPS)**.

---

# Experimental Findings

Experiments compared multiple architectures across the three tasks.

Key observations include:

* **Temporal models significantly outperform simple pooling baselines**
* **GRU encoders provide strong performance with moderate complexity**
* **Transformer models capture long-range dependencies effectively**
* **Pose-based representations are sufficient to learn meaningful gesture embeddings**

These findings support the use of **pose-based sequence modeling for sign language understanding tasks**.

---

# Repository Structure

```
src
│
├── scripts
│   ├── authenticate_GCP.py
│   ├── download_from_GCP.py
│   ├── download_from_hugging_face.py
│   └── preprocessing.ipynb
│
├── tasks
│
│   ├── task3_cislr
│   │   ├── data
│   │   ├── models
│   │   ├── training
│   │   └── evaluation
│   │
│   ├── task4_word_presence
│   │   ├── data
│   │   ├── models
│   │   └── training
│   │
│   └── task5_semantic_similarity
│       ├── data
│       ├── models
│       ├── training
│       └── evaluation
│
└── utils
```

The repository is structured to support **modular experimentation across tasks and model architectures**.

---

# Setup

Clone the repository:

```bash
git clone https://github.com/loki-1990/EE964_Projects.git
cd EE964_Projects
```

Install dependencies:

```bash
pip install torch numpy pandas matplotlib tqdm
```

---

# Notes

Large artifacts such as:

* datasets
* pose files
* trained model checkpoints

are **not included in this repository** due to GitHub size limitations.

Scripts are provided to download and preprocess required data.

---

# Skills Demonstrated

This project demonstrates practical experience in:

* Deep Learning
* PyTorch
* Transformer Architectures
* Sequence Modeling
* Representation Learning
* Retrieval Metrics (MRR, Top-K)
* Experiment Design
* Data Processing Pipelines

---

# Authors

**Lokesh Kumar**
Roll No: 241562482
[lokeshk24@iitk.ac.in](mailto:lokeshk24@iitk.ac.in)

**Rakesh BR**
Roll No: 241562524
[rakeshbr24@iitk.ac.in](mailto:rakeshbr24@iitk.ac.in)

Indian Institute of Technology Kanpur (IIT Kanpur)

---

# License

This repository is provided for **academic and research purposes**.

---
