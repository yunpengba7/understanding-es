# Contributing

This reproduction package intentionally has a narrow executable scope: the
canonical Qwen2.5-1.5B-Instruct ES easy-setting training run plus separate
Qwen2.5-1.5B-Instruct base-checkpoint and ES-checkpoint evaluation jobs for
GSM8K. Changes
should preserve the fixed configuration and current public CLI surface:
two-epoch training, the one-step smoke test, greedy/mean@32 evaluation, and
evaluation-result verification. Do not add partial-training replay claims or
additional evaluation metrics.

Before proposing a change, run:

```bash
uv run pytest -q
uv run ruff check .
uv run es-audit .
```

Do not commit model weights, generated responses, local filesystem paths,
machine identifiers, run results, or additional optimization methods. Public
author and repository metadata should remain synchronized with the paper.
