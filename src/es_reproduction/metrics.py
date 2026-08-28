from __future__ import annotations

import math
from typing import Any


def _validated_sample_count(rows: list[dict[str, Any]], metric_name: str) -> int:
    if not rows:
        raise ValueError(f"{metric_name} requires at least one question")
    sample_count = len(rows[0].get("samples", []))
    if sample_count == 0:
        raise ValueError(f"{metric_name} requires retained samples")
    if any(len(row.get("samples", [])) != sample_count for row in rows):
        raise ValueError("every question must retain the same number of samples")
    return sample_count


def compute_pass_at_1(rows: list[dict[str, Any]]) -> float:
    sample_count = _validated_sample_count(rows, "Pass@1")
    return sum(
        int(bool(sample.get("correct")))
        for row in rows
        for sample in row["samples"]
    ) / (len(rows) * sample_count)


def compute_pass_at_k(rows: list[dict[str, Any]], *, k: int) -> float:
    sample_count = _validated_sample_count(rows, "Pass@K")
    if k <= 0 or k > sample_count:
        raise ValueError("requested K is outside the retained sample range")

    def pass_for_row(row: dict[str, Any]) -> float:
        correct = sum(bool(sample.get("correct")) for sample in row["samples"])
        if sample_count - correct < k:
            return 1.0
        return 1.0 - math.prod(
            (sample_count - correct - index) / (sample_count - index)
            for index in range(k)
        )

    return sum(pass_for_row(row) for row in rows) / len(rows)


def compute_sample_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "pass_at_1": compute_pass_at_1(rows),
        "pass_at_16": compute_pass_at_k(rows, k=16),
        "pass_at_32": compute_pass_at_k(rows, k=32),
    }
