# ES for LLM Reasoning

This anonymous artifact accompanies the AAAI-27 submission:

> **Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO**

The repository contains only the Evolution Strategies (ES) implementation needed
for the paper's easy-setting reproduction with Qwen2.5-1.5B-Instruct. It has no
dependency on the private research repository and contains no model weights.

## Reproduction claims

The artifact supports three distinct runs:

1. An eight-GPU, two-epoch ES training run.
2. A single-GPU GSM8K evaluation of the base model.
3. A separate single-GPU GSM8K evaluation of the merged two-epoch ES model.

The Base and ES evaluation jobs are independent.
`scripts/run_two_evaluations.py` assigns each job to any currently idle GPU; it
runs them concurrently when two GPUs are available and sequentially when only
one is available. Each job uses exactly one visible GPU.

The expected GSM8K results are:

| Model | Greedy | mean@32 |
|---|---:|---:|
| Base | 988 / 1319 | 0.702710 |
| ES, two epochs | 966 / 1319 | 0.731994 |

Greedy evaluation must reproduce the exact correct-question count shown above.
`mean@32` uses an absolute tolerance of `0.005` to cover small cross-run
numerical differences in GPU generation. Dataset and model-weight identities
remain exact.

## Environment

Python 3.13 and all runtime packages are pinned in `pyproject.toml` and
`uv.lock`.

The evaluation entrypoint also fixes `VLLM_ENABLE_V1_MULTIPROCESSING=0`
before importing vLLM. The reference evaluations used the in-process V1
EngineCore, and changing this execution path can change BF16 greedy
generations even when the model, prompts, decoding parameters, and GPU are
otherwise identical.

```bash
uv sync --extra dev --locked
uv run pytest -q
uv run ruff check .
uv run es-audit .
```

The base model is fixed to:

- model ID: `Qwen/Qwen2.5-1.5B-Instruct`
- revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

Passing the model ID downloads that exact revision. A local directory containing
the same snapshot may be passed instead.

## ES update convention

For perturbations \(\theta+\sigma\epsilon_i\), the implementation applies

\[
\theta \leftarrow \theta
  + \frac{\texttt{learning\_rate}}{N}
    \sum_{i=1}^{N}\hat r_i\epsilon_i ,
\]

where \(\hat r_i\) denotes the normalized reward. The conventional
\(1/\sigma\) factor is already absorbed into `training.learning_rate`; do not
divide by \(\sigma\) again when interpreting or reproducing the configured
update.

## Two-epoch ES training

Expose exactly eight GPUs. Each GPU hosts one tensor-parallel-size-one vLLM
engine.

Run the fixed two-epoch protocol:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/train_two_epochs.sh \
  Qwen/Qwen2.5-1.5B-Instruct \
  outputs/two-epochs
```

Only merged Hugging Face checkpoints `step_117/` and `step_234/`, compact
metrics, and TensorBoard events are produced. Resume snapshots and raw
optimizer states are intentionally outside the artifact contract.
Engine synchronization follows `training.engine_sync_every` independently of
checkpoint export, and every checkpoint is synchronized before it is written.

For an inexpensive resource check without a numerical reproduction claim:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/smoke_test.sh \
  Qwen/Qwen2.5-1.5B-Instruct \
  outputs/smoke
```

## Separate single-GPU evaluation tasks

Run one task directly:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/evaluate_gsm8k.sh \
  /path/to/model \
  base \
  outputs/eval-base
```

Or schedule both independent tasks on any idle GPUs:

```bash
uv run python scripts/run_two_evaluations.py \
  --base-model /path/to/Qwen2.5-1.5B-Instruct \
  --es-model /path/to/step_234 \
  --output-root outputs/evaluations
```

Verify each summary:

```bash
uv run es-verify --reference reference/expected_results.json \
  evaluation --model-key base \
  --result outputs/evaluations/base/result.json

uv run es-verify --reference reference/expected_results.json \
  evaluation --model-key es_step_234 \
  --result outputs/evaluations/es_step_234/result.json
```

New evaluation generations are written under the requested output directory so
greedy and mean@32 can be audited locally. The entire `outputs/` tree is
ignored by Git: no generated responses, run logs, model checkpoints, or actual
evaluation result files are included in the anonymous repository. Only the
compact greedy and mean@32 acceptance reference is committed.

## Repository map

- `src/es_reproduction/`: ES training, GSM8K evaluation, verification, and audit
- `configs/easy_qwen25_1p5b.yaml`: immutable paper protocol
- `data/gsm8k/`: exact MIT-licensed train and test snapshots
- `reference/expected_results.json`: compact numerical acceptance reference
- `scripts/`: reproducible commands and free-GPU evaluation scheduler
- `docs/`: public scope, specification, and design decisions

## License

Code is released under the MIT License. The included GSM8K snapshots retain
their MIT dataset license; see `data/gsm8k/README.md`.
