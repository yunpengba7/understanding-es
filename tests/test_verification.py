from __future__ import annotations

import json
from pathlib import Path

import pytest

from es_reproduction.verify import verify_evaluation

ROOT = Path(__file__).resolve().parents[1]


def _expected_results() -> dict:
    return json.loads((ROOT / "reference" / "expected_results.json").read_text())


def _exact_base_evaluation(expected: dict) -> dict:
    reference = expected["evaluation"]["base"]
    return {
        "greedy": {
            "correct": reference["greedy_correct"],
            "total": reference["greedy_total"],
            "accuracy": reference["greedy"],
        },
        "provenance": {
            "dataset": expected["dataset"],
            "weight_sha256": {
                "model.safetensors": expected["model"]["base_weight_sha256"]
            },
        },
        "sampling": {"mean_at_32": reference["mean_at_32"]},
    }


def test_evaluation_contract_rejects_a_different_greedy_correct_count() -> None:
    expected = _expected_results()
    actual = _exact_base_evaluation(expected)
    actual["greedy"] = {"correct": 981, "total": 1319, "accuracy": 981 / 1319}

    result = verify_evaluation(
        actual,
        expected,
        model_key="base",
        mean_tolerance=0.005,
    )

    assert not result.passed
    assert result.failures == (
        "greedy correct: actual=981 expected=988",
    )


def test_evaluation_contract_accepts_exact_greedy_count_and_mean_at_32() -> None:
    expected = _expected_results()
    actual = _exact_base_evaluation(expected)

    result = verify_evaluation(
        actual,
        expected,
        model_key="base",
        mean_tolerance=0.005,
    )

    assert result.passed


def test_evaluation_contract_rejects_mean_at_32_outside_tolerance() -> None:
    expected = _expected_results()
    actual = _exact_base_evaluation(expected)
    actual["sampling"]["mean_at_32"] += 0.006

    result = verify_evaluation(
        actual,
        expected,
        model_key="base",
        mean_tolerance=0.005,
    )

    assert not result.passed
    assert result.failures[0].startswith("mean_at_32:")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correct", 988.5),
        ("correct", "988"),
        ("correct", True),
        ("total", 1319.5),
        ("total", "1319"),
        ("total", True),
    ],
)
def test_evaluation_contract_requires_integer_greedy_counts(
    field: str,
    value: object,
) -> None:
    expected = _expected_results()
    actual = _exact_base_evaluation(expected)
    actual["greedy"][field] = value

    result = verify_evaluation(
        actual,
        expected,
        model_key="base",
        mean_tolerance=0.005,
    )

    assert f"greedy {field} must be an integer" in result.failures


def test_evaluation_contract_rejects_inconsistent_greedy_accuracy() -> None:
    expected = _expected_results()
    actual = _exact_base_evaluation(expected)
    actual["greedy"]["accuracy"] = 0.0

    result = verify_evaluation(
        actual,
        expected,
        model_key="base",
        mean_tolerance=0.005,
    )

    assert "greedy accuracy is inconsistent with correct / total" in result.failures
