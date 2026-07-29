from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    checked: int
    failures: tuple[str, ...]


def verify_evaluation(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    model_key: str,
    mean_tolerance: float,
) -> VerificationResult:
    reference = expected["evaluation"][model_key]
    failures = []
    actual_benchmark = actual.get("benchmark")
    if actual_benchmark not in {None, "gsm8k"}:
        failures.append(f"benchmark: actual={actual_benchmark!r} expected='gsm8k'")
    greedy = actual["greedy"]
    greedy_correct = greedy.get("correct")
    greedy_total = greedy.get("total")
    greedy_accuracy = greedy.get("accuracy")
    correct_is_integer = type(greedy_correct) is int
    total_is_integer = type(greedy_total) is int
    if not correct_is_integer:
        failures.append("greedy correct must be an integer")
    elif greedy_correct != reference["greedy_correct"]:
        failures.append(
            f"greedy correct: actual={greedy_correct} "
            f"expected={reference['greedy_correct']}"
        )
    if not total_is_integer:
        failures.append("greedy total must be an integer")
    elif greedy_total != reference["greedy_total"]:
        failures.append(
            f"greedy total: actual={greedy_total} expected={reference['greedy_total']}"
        )
    if type(greedy_accuracy) not in {int, float}:
        failures.append("greedy accuracy must be a number")
    elif (
        correct_is_integer
        and total_is_integer
        and greedy_total > 0
        and float(greedy_accuracy) != greedy_correct / greedy_total
    ):
        failures.append("greedy accuracy is inconsistent with correct / total")
    provenance = actual.get("provenance", {})
    if provenance.get("dataset") != expected["dataset"]:
        failures.append("dataset identity does not match the fixed gsm8k snapshot")
    expected_weight = expected["model"][f"{model_key}_weight_sha256"]
    actual_weights = provenance.get("weight_sha256", {})
    if expected_weight not in actual_weights.values():
        failures.append(f"{model_key} model weight hash does not match the reference")
    sampling = actual["sampling"]
    metric = "mean_at_32"
    error = abs(float(sampling[metric]) - float(reference[metric]))
    if error > mean_tolerance:
        failures.append(
            f"{metric}: actual={float(sampling[metric]):.8f} "
            f"expected={float(reference[metric]):.8f} error={error:.8f}"
        )
    return VerificationResult(not failures, 7, tuple(failures))


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify artifact evaluation results")
    parser.add_argument("--reference", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    evaluation = subparsers.add_parser("evaluation")
    evaluation.add_argument("--result", type=Path, required=True)
    evaluation.add_argument("--model-key", choices=("base", "es_step_234"), required=True)
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = _load_json(args.reference)
    result = verify_evaluation(
        _load_json(args.result),
        reference,
        model_key=args.model_key,
        mean_tolerance=float(reference["evaluation"]["mean_absolute_tolerance"]),
    )
    print(json.dumps({"passed": result.passed, "failures": result.failures}, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(cli())
