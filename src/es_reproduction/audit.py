from __future__ import annotations

import argparse
import re
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


def _forbidden_method_name() -> str:
    return "ES" + "SAM"


def _excluded_benchmark_name() -> str:
    return "GP" + "QA"


def _excluded_scope_rules() -> dict[str, tuple[str, ...]]:
    return {
        "partial-training-replay": (
            "first" + "-10",
            "first" + "_10",
            "first" + "-ten",
            "first" + " ten",
        ),
        "additional-evaluation-metrics": (
            "pass" + "_at_",
            "maj" + "_at_",
            "pass" + "@",
            "maj" + "@",
            "plural" + "ity",
        ),
    }


def _excluded_scope_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple(
        (rule, re.compile(re.escape(term), re.IGNORECASE))
        for rule, terms in _excluded_scope_rules().items()
        for term in terms
    )


def _matching_scope_rule(
    value: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> str | None:
    return next(
        (rule for rule, pattern in patterns if pattern.search(value)),
        None,
    )


def audit_repository(root: str | Path) -> list[str]:
    repository = Path(root).resolve()
    failures: list[str] = []
    forbidden = re.compile(re.escape(_forbidden_method_name()), re.IGNORECASE)
    excluded_benchmark = re.compile(
        re.escape(_excluded_benchmark_name()),
        re.IGNORECASE,
    )
    excluded_scope = _excluded_scope_patterns()
    local_paths = (
        "/" + "home" + "/",
        "/" + "mnt" + "/",
    )
    email_pattern = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

    for path in sorted(repository.rglob("*")):
        relative = path.relative_to(repository)
        if any(part in IGNORED_PARTS for part in relative.parts) or not path.is_file():
            continue
        if forbidden.search(relative.as_posix()):
            failures.append(f"forbidden method name in path: {relative}")
        if excluded_benchmark.search(relative.as_posix()):
            failures.append(f"excluded benchmark in path: {relative}")
        path_scope_rule = _matching_scope_rule(relative.as_posix(), excluded_scope)
        if path_scope_rule is not None:
            failures.append(
                f"excluded publication scope ({path_scope_rule}) in path: {relative}"
            )
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if forbidden.search(text):
            failures.append(f"forbidden method name in file: {relative}")
        if excluded_benchmark.search(text):
            failures.append(f"excluded benchmark in file: {relative}")
        content_scope_rule = _matching_scope_rule(text, excluded_scope)
        if content_scope_rule is not None:
            failures.append(
                f"excluded publication scope ({content_scope_rule}) in file: {relative}"
            )
        if any(prefix in text for prefix in local_paths):
            failures.append(f"absolute local path in file: {relative}")
        emails = {
            match.group(0).lower()
            for match in email_pattern.finditer(text)
            if match.group(0).lower() != "anonymous@example.com"
        }
        if emails:
            failures.append(f"non-anonymous email in file: {relative}")

    git_dir = repository / ".git"
    if git_dir.exists():
        revisions = subprocess.run(
            ["git", "rev-list", "--all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if revisions:
            historical_contents = subprocess.run(
                [
                    "git",
                    "grep",
                    "-I",
                    "-i",
                    "-n",
                    "-e",
                    _excluded_benchmark_name(),
                    *[
                        argument
                        for terms in _excluded_scope_rules().values()
                        for term in terms
                        for argument in ("-e", term)
                    ],
                    *revisions,
                ],
                cwd=repository,
                check=False,
                capture_output=True,
                text=True,
            )
            if historical_contents.returncode not in {0, 1}:
                raise RuntimeError(historical_contents.stderr.strip())
            if historical_contents.stdout:
                failures.append("excluded content remains in reachable Git history")
            historical_paths = subprocess.run(
                ["git", "log", "--all", "--name-only", "--pretty=format:"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            if any(excluded_benchmark.search(path) for path in historical_paths):
                failures.append(
                    "excluded benchmark path remains in reachable Git history"
                )
            if any(
                _matching_scope_rule(path, excluded_scope) is not None
                for path in historical_paths
            ):
                failures.append(
                    "excluded publication path remains in reachable Git history"
                )
        tracked_paths = subprocess.run(
            ["git", "ls-files"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        generated_names = {
            "result.json",
            "greedy_samples.jsonl",
            "sampled_samples.jsonl",
        }
        generated_suffixes = (".log", ".safetensors")
        generated_parts = {"outputs", "models", "ray_results"}
        prohibited_tracked = [
            path
            for path in tracked_paths
            if (
                generated_parts.intersection(Path(path).parts)
                or Path(path).name in generated_names
                or path.endswith(generated_suffixes)
                or (
                    Path(path).name.startswith("pytorch_model")
                    and Path(path).suffix == ".bin"
                )
            )
        ]
        if prohibited_tracked:
            failures.append(
                "generated results, logs, or model files are tracked: "
                + ", ".join(prohibited_tracked)
            )
        remotes = subprocess.run(
            ["git", "remote"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if remotes:
            failures.append("Git remotes must be empty")
        identities = subprocess.run(
            ["git", "log", "--format=%an <%ae>%n%cn <%ce>"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if any(
            identity != "Anonymous Authors <anonymous@example.com>"
            for identity in identities
        ):
            failures.append("Git history contains a non-anonymous author or committer identity")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the public artifact for anonymity")
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures = audit_repository(args.root)
    if failures:
        print("\n".join(failures))
        return 1
    print("Anonymity audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
