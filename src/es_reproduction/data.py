from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

FINAL_ANSWER_PATTERN = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def dataset_manifest(data_dir: str | Path) -> dict[str, dict[str, int]]:
    root = Path(data_dir)
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        path = root / "main" / f"{split}-00000-of-00001.parquet"
        frame = pd.read_parquet(path, columns=["question", "answer"])
        result[split] = {"rows": len(frame)}
    return result


def extract_gold_answer(answer: str) -> str:
    match = FINAL_ANSWER_PATTERN.search(str(answer))
    if match is None:
        raise ValueError("GSM8K answer does not contain a final #### value")
    return match.group(1).replace(",", "")


def load_split(data_dir: str | Path, split: str) -> tuple[list[str], list[str]]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    path = Path(data_dir) / "main" / f"{split}-00000-of-00001.parquet"
    frame = pd.read_parquet(path, columns=["question", "answer"])
    questions = [str(value) for value in frame["question"].tolist()]
    answers = [extract_gold_answer(value) for value in frame["answer"].tolist()]
    return questions, answers


def build_training_schedule(
    *,
    n_train: int,
    batch_size: int,
    epochs: int,
    seed: int,
) -> list[list[int]]:
    if n_train <= 0 or batch_size <= 0 or epochs < 0:
        raise ValueError("n_train and batch_size must be positive; epochs must be non-negative")
    rng = np.random.default_rng(seed)
    schedule: list[list[int]] = []
    for _ in range(epochs):
        indices = np.arange(n_train)
        rng.shuffle(indices)
        for start in range(0, n_train, batch_size):
            schedule.append([int(index) for index in indices[start : start + batch_size]])
    return schedule
