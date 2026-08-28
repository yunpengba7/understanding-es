from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["README.md", "README_zh-CN.md"])
def test_readme_uses_github_math_delimiters(filename: str) -> None:
    content = (ROOT / filename).read_text(encoding="utf-8")

    for unsupported in (r"\(", r"\)", r"\[", r"\]"):
        assert unsupported not in content
    assert content.count("$$") > 0
    assert content.count("$$") % 2 == 0


@pytest.mark.parametrize("filename", ["README.md", "README_zh-CN.md"])
def test_readme_uses_full_model_names(filename: str) -> None:
    content = (ROOT / filename).read_text(encoding="utf-8")

    abbreviated_names = (
        r"Qwen2\.5-0\.5B(?!-Instruct)",
        r"Qwen2\.5-1\.5B(?!-Instruct)",
        r"Qwen2\.5-3B(?!-Instruct)",
        r"Qwen2\.5-7B(?!-Instruct)",
        r"Qwen/" + r"Lla" + r"ma",
    )
    for pattern in abbreviated_names:
        assert re.search(pattern, content) is None


@pytest.mark.parametrize("filename", ["README.md", "README_zh-CN.md"])
def test_readme_links_machine_readable_reference_results(filename: str) -> None:
    content = (ROOT / filename).read_text(encoding="utf-8")

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
