from __future__ import annotations

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
