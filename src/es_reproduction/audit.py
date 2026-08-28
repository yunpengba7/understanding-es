from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "outputs",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
LOCAL_PATH_PREFIXES = ("/" + "home" + "/", "/" + "mnt" + "/")
GENERATED_NAMES = {
    "result.json",
    "greedy_samples.jsonl",
    "sampled_samples.jsonl",
}
GENERATED_SUFFIXES = (".log", ".safetensors")
GENERATED_PARTS = {"outputs", "models", "ray_results"}


def _is_generated_artifact(path: str) -> bool:
    candidate = Path(path)
    return bool(
        GENERATED_PARTS.intersection(candidate.parts)
        or candidate.name in GENERATED_NAMES
        or path.endswith(GENERATED_SUFFIXES)
        or (candidate.name.startswith("pytorch_model") and candidate.suffix == ".bin")
    )


def audit_repository(root: str | Path) -> list[str]:
    """Check a public checkout for machine-local paths and generated artifacts."""
    repository = Path(root).resolve()
    failures: list[str] = []

    for path in sorted(repository.rglob("*")):
        relative = path.relative_to(repository)
        if any(part in IGNORED_PARTS for part in relative.parts) or not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(prefix in content for prefix in LOCAL_PATH_PREFIXES):
            failures.append(f"absolute local path in file: {relative}")

    git_dir = repository / ".git"
    if git_dir.exists():
        tracked_paths = subprocess.run(
            ["git", "ls-files"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        prohibited_tracked = [path for path in tracked_paths if _is_generated_artifact(path)]
        if prohibited_tracked:
            failures.append(
                "generated results, logs, or model files are tracked: "
                + ", ".join(prohibited_tracked)
            )
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the public release for hygiene")
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures = audit_repository(args.root)
    if failures:
        print("\n".join(failures))
        return 1
    print("Release hygiene audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
