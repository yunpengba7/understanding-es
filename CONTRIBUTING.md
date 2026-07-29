# Contributing

This artifact intentionally has a narrow scope: the canonical ES easy-setting
training run plus separate Base and ES evaluation jobs for GSM8K. Changes
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
machine identifiers, user identities, run results, or additional optimization
methods.
