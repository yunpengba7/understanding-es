from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_uses_github_math_blocks() -> None:
    content = README.read_text(encoding="utf-8")

    for unsupported in (r"\(", r"\)", r"\[", r"\]", r"\operatorname", "$$"):
        assert unsupported not in content
    assert content.count("```math\n") == 2


def test_readme_uses_full_model_names() -> None:
    content = README.read_text(encoding="utf-8")

    abbreviated_names = (
        r"Qwen2\.5-0\.5B(?!-Instruct)",
        r"Qwen2\.5-1\.5B(?!-Instruct)",
        r"Qwen2\.5-3B(?!-Instruct)",
        r"Qwen2\.5-7B(?!-Instruct)",
        r"Qwen/" + r"Lla" + r"ma",
    )
    for pattern in abbreviated_names:
        assert re.search(pattern, content) is None


def test_readme_links_machine_readable_reference_results() -> None:
    content = README.read_text(encoding="utf-8")

    assert "reference/results.json" in content


def test_reference_results_have_the_documented_endpoint_schema() -> None:
    results = json.loads(
        (ROOT / "reference" / "results.json").read_text(encoding="utf-8")
    )

    assert set(results) == {"model", "dataset", "evaluation"}
    assert set(results["model"]) == {"base_id", "base_revision"}
    assert set(results["dataset"]) == {"train", "test"}
    assert set(results["evaluation"]) == {"base", "es_step_234"}
    metric_fields = {
        "greedy_correct",
        "greedy_total",
        "greedy",
        "pass_at_1",
        "pass_at_16",
        "pass_at_32",
    }
    assert set(results["evaluation"]["base"]) == metric_fields
    assert set(results["evaluation"]["es_step_234"]) == metric_fields
