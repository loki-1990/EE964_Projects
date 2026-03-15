from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
from torch.utils.data import Sampler


class GroupedBatchSampler(Sampler[list[int]]):
    """
    Batch sampler that builds batches with multiple samples per gloss.

    Example:
        n_glosses_per_batch = 8
        n_samples_per_gloss = 2

    => batch size = 16, with structure like:
       g1_a, g1_b, g2_a, g2_b, ..., g8_a, g8_b

    This is useful for supervised contrastive learning because every gloss
    in the batch has at least one positive pair.
    """

    def __init__(
        self,
        split_csv: str | Path,
        gloss_col: str = "gloss",
        n_glosses_per_batch: int = 8,
        n_samples_per_gloss: int = 2,
        steps_per_epoch: int | None = None,
        seed: int = 1990,
        drop_last: bool = True,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.gloss_col = gloss_col
        self.n_glosses_per_batch = n_glosses_per_batch
        self.n_samples_per_gloss = n_samples_per_gloss
        self.seed = seed
        self.drop_last = drop_last

        self.df = pd.read_csv(self.split_csv)

        # Build mapping: gloss -> dataset row indices
        gloss_to_indices = defaultdict(list)
        for idx, gloss in enumerate(self.df[self.gloss_col].tolist()):
            gloss_to_indices[gloss].append(idx)

        # Keep only glosses with enough samples
        self.gloss_to_indices = {
            gloss: indices
            for gloss, indices in gloss_to_indices.items()
            if len(indices) >= self.n_samples_per_gloss
        }

        self.eligible_glosses = sorted(self.gloss_to_indices.keys())

        if len(self.eligible_glosses) < self.n_glosses_per_batch:
            raise ValueError(
                f"Not enough eligible glosses ({len(self.eligible_glosses)}) "
                f"for n_glosses_per_batch={self.n_glosses_per_batch}."
            )

        self.batch_size = self.n_glosses_per_batch * self.n_samples_per_gloss

        # If not provided, estimate a reasonable number of steps per epoch
        # based on how many total eligible samples exist.
        if steps_per_epoch is None:
            total_eligible_samples = sum(len(v) for v in self.gloss_to_indices.values())
            self.steps_per_epoch = total_eligible_samples // self.batch_size
            if self.steps_per_epoch == 0:
                self.steps_per_epoch = 1
        else:
            self.steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)

        # Per-gloss shuffled pools and pointers
        gloss_pools = {}
        gloss_ptrs = {}

        for gloss, indices in self.gloss_to_indices.items():
            pool = indices.copy()
            rng.shuffle(pool)
            gloss_pools[gloss] = pool
            gloss_ptrs[gloss] = 0

        for _ in range(self.steps_per_epoch):
            selected_glosses = rng.sample(self.eligible_glosses, self.n_glosses_per_batch)

            batch_indices: list[int] = []

            for gloss in selected_glosses:
                pool = gloss_pools[gloss]
                ptr = gloss_ptrs[gloss]

                # If not enough samples remain, reshuffle and restart this gloss pool
                if ptr + self.n_samples_per_gloss > len(pool):
                    pool = self.gloss_to_indices[gloss].copy()
                    rng.shuffle(pool)
                    gloss_pools[gloss] = pool
                    ptr = 0

                chosen = pool[ptr: ptr + self.n_samples_per_gloss]
                batch_indices.extend(chosen)

                gloss_ptrs[gloss] = ptr + self.n_samples_per_gloss

            if len(batch_indices) == self.batch_size:
                yield batch_indices
            elif not self.drop_last and len(batch_indices) > 0:
                yield batch_indices


def build_grouped_batch_sampler(
    split_csv: str | Path,
    gloss_col: str = "gloss",
    n_glosses_per_batch: int = 8,
    n_samples_per_gloss: int = 2,
    steps_per_epoch: int | None = None,
    seed: int = 1990,
) -> GroupedBatchSampler:
    return GroupedBatchSampler(
        split_csv=split_csv,
        gloss_col=gloss_col,
        n_glosses_per_batch=n_glosses_per_batch,
        n_samples_per_gloss=n_samples_per_gloss,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )