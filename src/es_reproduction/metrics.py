from __future__ import annotations

from typing import Any


def compute_mean_at_k(
    rows: list[dict[str, Any]],
    *,
    k: int,
) -> float:
    if not rows:
        raise ValueError("mean@k requires at least one question")
    sample_count = len(rows[0].get("samples", []))
    if sample_count == 0:
        raise ValueError("mean@k requires retained samples")
    if any(len(row.get("samples", [])) != sample_count for row in rows):
        raise ValueError("every question must retain the same number of samples")
    if k <= 0 or k > sample_count:
        raise ValueError("requested k is outside the retained sample range")
    return sum(
        int(bool(sample.get("correct")))
        for row in rows
        for sample in row["samples"][:k]
    ) / (len(rows) * k)
