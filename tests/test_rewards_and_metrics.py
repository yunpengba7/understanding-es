from __future__ import annotations

import pytest

from es_reproduction.metrics import (
    compute_pass_at_1,
    compute_pass_at_k,
    compute_sample_metrics,
)
from es_reproduction.rewards import (
    extract_gsm8k_prediction,
    score_training_response,
)


def test_training_reward_preserves_correct_format_and_missing_format_cases() -> None:
    assert score_training_response(r"Reasoning. \boxed{18}", "18", format_reward=0.1) == 1.0
    assert score_training_response(r"Reasoning. \boxed{19}", "18", format_reward=0.1) == 0.1
    assert score_training_response("The answer is 18.", "18", format_reward=0.1) == 0.0


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"Work. \boxed{1,024}", "1024"),
        ("Work.\n#### -3.5", "-3.5"),
        ("Therefore, the answer is 42.", "42"),
        ("No numeric result", None),
    ],
)
def test_evaluation_prediction_extraction(response: str, expected: str | None) -> None:
    assert extract_gsm8k_prediction(response) == expected


def test_pass_at_1_uses_all_retained_samples() -> None:
    rows = [
        {
            "gold": "2",
            "samples": [
                {"response": r"\boxed{2}", "correct": True},
                {"response": r"\boxed{3}", "correct": False},
                {"response": r"\boxed{2.0}", "correct": True},
                {"response": "invalid", "correct": False},
            ],
        },
        {
            "gold": "5",
            "samples": [
                {"response": r"\boxed{4}", "correct": False},
                {"response": r"\boxed{5}", "correct": True},
                {"response": r"\boxed{4}", "correct": False},
                {"response": r"\boxed{5}", "correct": True},
            ],
        },
    ]

    assert compute_pass_at_1(rows) == 0.5


def test_pass_at_k_uses_the_unbiased_subset_estimator() -> None:
    rows = [
        {
            "samples": [
                {"correct": True},
                {"correct": True},
                {"correct": False},
                {"correct": False},
            ]
        },
        {
            "samples": [
                {"correct": True},
                {"correct": False},
                {"correct": False},
                {"correct": False},
            ]
        },
    ]

    assert compute_pass_at_k(rows, k=2) == pytest.approx(2 / 3)
    assert compute_pass_at_k(rows, k=4) == 1.0


def test_sample_metrics_use_pass_at_1_16_32_output_names() -> None:
    rows = [
        {"samples": [{"correct": False} for _ in range(32)]},
        {
            "samples": [
                {"correct": True},
                *({"correct": False} for _ in range(31)),
            ]
        },
    ]

    assert compute_sample_metrics(rows) == {
        "pass_at_1": 1 / 64,
        "pass_at_16": 0.25,
        "pass_at_32": 0.5,
    }
