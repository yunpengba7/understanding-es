from __future__ import annotations

import argparse
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .config import load_canonical_config
from .data import dataset_manifest, load_split
from .metrics import compute_mean_at_k
from .prompts import render_gsm8k_prompt
from .rewards import is_gsm8k_correct
from .train import repository_root

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
VLLM_V1_MULTIPROCESSING = "0"


def configure_vllm_runtime() -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = VLLM_V1_MULTIPROCESSING


# The reference evaluations used the in-process V1 EngineCore. This must be set
# before importing vLLM so greedy generation follows the same execution path.
configure_vllm_runtime()


def greedy_sampling_kwargs(max_tokens: int) -> dict[str, int | float]:
    return {"temperature": 0.0, "top_p": 1.0, "max_tokens": int(max_tokens)}


def sampled_sampling_kwargs(
    *,
    temperature: float,
    top_p: float,
    num_samples: int,
    max_tokens: int,
) -> dict[str, int | float]:
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": int(num_samples),
        "max_tokens": int(max_tokens),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generated_texts(outputs: list[Any], expected_candidates: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for request_output in outputs:
        candidates = getattr(request_output, "outputs", None)
        if candidates is None or len(candidates) != expected_candidates:
            raise RuntimeError(
                f"Expected {expected_candidates} candidate(s), "
                f"received {0 if candidates is None else len(candidates)}"
            )
        rows.append([str(candidate.text) for candidate in candidates])
    return rows


def resolve_model(model: str, label: str) -> tuple[str, str | None]:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path.resolve()), None
    if label == "base":
        if model != BASE_MODEL_ID:
            raise ValueError(
                f"The remote Qwen2.5-1.5B-Instruct base checkpoint must be {BASE_MODEL_ID}; "
                "pass a local directory to evaluate another snapshot"
            )
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(repo_id=model, revision=BASE_MODEL_REVISION)
        return str(Path(snapshot).resolve()), BASE_MODEL_REVISION
    return model, None


def run_evaluation(
    *,
    model: str,
    output_dir: Path,
    data_dir: Path,
    label: str,
) -> dict[str, Any]:
    configure_vllm_runtime()
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Each evaluation task requires exactly one visible GPU; "
            f"found {torch.cuda.device_count()}"
        )
    resolved_model, revision = resolve_model(model, label)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    config = load_canonical_config(repository_root() / "configs" / "easy_qwen25_1p5b.yaml")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
    questions, golds = load_split(data_dir, "test")
    prompts = [render_gsm8k_prompt(question, tokenizer) for question in questions]
    engine = LLM(
        model=resolved_model,
        tensor_parallel_size=1,
        dtype=config.evaluation.dtype,
        gpu_memory_utilization=config.evaluation.gpu_memory_utilization,
        enable_prefix_caching=False,
        disable_log_stats=True,
        max_model_len=config.evaluation.max_model_length,
        trust_remote_code=True,
        seed=config.evaluation.seed,
    )

    greedy_outputs = engine.generate(
        prompts,
        SamplingParams(**greedy_sampling_kwargs(config.evaluation.max_new_tokens)),
    )
    greedy_texts = [row[0] for row in _generated_texts(greedy_outputs, 1)]
    greedy_rows = [
        {
            "index": index,
            "gold": gold,
            "response": response,
            "correct": is_gsm8k_correct(response, gold),
        }
        for index, (response, gold) in enumerate(zip(greedy_texts, golds, strict=True))
    ]
    greedy_correct = sum(bool(row["correct"]) for row in greedy_rows)

    sampled_outputs = engine.generate(
        prompts,
        SamplingParams(
            **sampled_sampling_kwargs(
                temperature=config.evaluation.temperature,
                top_p=config.evaluation.top_p,
                num_samples=config.evaluation.num_samples,
                max_tokens=config.evaluation.max_new_tokens,
            )
        ),
    )
    sampled_texts = _generated_texts(sampled_outputs, config.evaluation.num_samples)
    sampled_rows = []
    for index, (responses, gold) in enumerate(zip(sampled_texts, golds, strict=True)):
        sampled_rows.append(
            {
                "index": index,
                "gold": gold,
                "samples": [
                    {
                        "response": response,
                        "correct": is_gsm8k_correct(response, gold),
                    }
                    for response in responses
                ],
            }
        )
    mean_at_32 = compute_mean_at_k(sampled_rows, k=32)
    result = {
        "benchmark": "gsm8k",
        "label": label,
        "greedy": {
            "correct": int(greedy_correct),
            "total": len(greedy_rows),
            "accuracy": greedy_correct / len(greedy_rows),
        },
        "sampling": {"mean_at_32": mean_at_32},
        "provenance": {
            "dataset": dataset_manifest(data_dir),
            "model_name": Path(resolved_model).name,
            "model_revision": revision,
            "python": platform.python_version(),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "vllm": version("vllm"),
            "vllm_enable_v1_multiprocessing": os.environ[
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ],
            "visible_gpu": torch.cuda.get_device_name(0),
        },
    }
    _write_json(output_dir / "result.json", result)
    with (output_dir / "greedy_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in greedy_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "sampled_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in sampled_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one model on GSM8K using one GPU"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--label", choices=("base", "es_step_234"), required=True)
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_evaluation(
        model=args.model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        label=args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
