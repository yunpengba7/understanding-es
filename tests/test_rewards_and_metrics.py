from __future__ import annotations

import pytest

from es_reproduction.metrics import compute_mean_at_k
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


def test_mean_at_k_uses_the_first_retained_samples() -> None:
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

    assert compute_mean_at_k(rows, k=2) == 0.5
    assert compute_mean_at_k(rows, k=4) == 0.5
