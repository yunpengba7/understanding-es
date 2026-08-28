from __future__ import annotations

from es_reproduction.audit import audit_repository


def test_release_audit_allows_public_author_metadata(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "Zhi Zheng <zhi.zheng@u.nus.edu>\n",
        encoding="utf-8",
    )

    assert audit_repository(tmp_path) == []


def test_release_audit_rejects_machine_local_paths(tmp_path) -> None:
    (tmp_path / "config.md").write_text(
        "MODEL=/" + "home/example/checkpoints/model\n",
        encoding="utf-8",
    )

    assert audit_repository(tmp_path) == ["absolute local path in file: config.md"]
