from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

from es_reproduction import evaluation
from es_reproduction.evaluation import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    VLLM_V1_MULTIPROCESSING,
    configure_vllm_runtime,
    resolve_model,
    sampled_sampling_kwargs,
)
from es_reproduction.evaluation import (
    build_parser as build_evaluation_parser,
)
from es_reproduction.train import build_parser as build_training_parser


def test_evaluation_uses_reference_vllm_engine_process_mode() -> None:
    assert VLLM_V1_MULTIPROCESSING == "0"
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_evaluation_restores_reference_vllm_engine_process_mode(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    configure_vllm_runtime()

    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_canonical_training_cli_only_accepts_artifact_locations() -> None:
    parser = build_training_parser()
    destinations = {action.dest for action in parser._actions}
    mode_action = next(action for action in parser._actions if action.dest == "mode")

    assert {"model", "output_dir", "data_dir", "mode"} <= destinations
    assert set(mode_action.choices) == {"two-epochs", "smoke"}
    assert "sigma" not in destinations
    assert "population_size" not in destinations
    assert "batch_size" not in destinations


def test_evaluation_cli_is_one_model_one_task_one_gpu() -> None:
    parser = build_evaluation_parser()
    destinations = {action.dest for action in parser._actions}

    assert {"model", "output_dir", "data_dir", "label"} <= destinations
    assert "benchmark" not in destinations
    assert "tensor_parallel_size" not in destinations


def test_sampled_evaluation_uses_engine_seed_stream_not_one_shared_request_seed() -> None:
    kwargs = sampled_sampling_kwargs(
        temperature=0.6,
        top_p=1.0,
        num_samples=32,
        max_tokens=2048,
    )
    assert "seed" not in kwargs


def test_evaluation_writes_pass_metrics_and_sample_hits(
    monkeypatch,
    tmp_path,
) -> None:
    generation_calls = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            assert kwargs["seed"] == 0

        def generate(self, prompts, sampling):
            generation_calls.append(sampling.kwargs)
            assert prompts == ["prompt: first", "prompt: second"]
            if sampling.kwargs.get("n") == 32:
                return [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(text="right" if index < 16 else "wrong")
                            for index in range(32)
                        ]
                    ),
                    SimpleNamespace(
                        outputs=[SimpleNamespace(text="wrong") for _ in range(32)]
                    ),
                ]
            return [
                SimpleNamespace(outputs=[SimpleNamespace(text="right")]),
                SimpleNamespace(outputs=[SimpleNamespace(text="wrong")]),
            ]

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            get_device_name=lambda index: "Test GPU",
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    monkeypatch.setattr(
        evaluation,
        "load_split",
        lambda data_dir, split: (["first", "second"], ["1", "2"]),
    )
    monkeypatch.setattr(
        evaluation,
        "dataset_manifest",
        lambda data_dir: {"test": {"rows": 2}},
    )
    monkeypatch.setattr(
        evaluation,
        "render_gsm8k_prompt",
        lambda question, tokenizer: f"prompt: {question}",
    )
    monkeypatch.setattr(
        evaluation,
        "is_gsm8k_correct",
        lambda response, gold: response == "right",
    )
    model_dir = tmp_path / "Qwen2.5-1.5B-Instruct"
    model_dir.mkdir()
    output_dir = tmp_path / "evaluation"

    result = evaluation.run_evaluation(
        model=str(model_dir),
        output_dir=output_dir,
        data_dir=tmp_path,
        label="base",
    )

    assert generation_calls[0]["n"] == 32
    assert "seed" not in generation_calls[0]
    assert generation_calls[1]["temperature"] == 0.0
    assert result["greedy"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert result["sampling"] == {
        "pass_at_1": 0.25,
        "pass_at_16": (1 - 1 / math.comb(32, 16)) / 2,
        "pass_at_32": 0.5,
    }
    assert json.loads((output_dir / "result.json").read_text()) == result
    sampled_rows = [
        json.loads(line)
        for line in (output_dir / "sampled_samples.jsonl").read_text().splitlines()
    ]
    assert [row["hits"] for row in sampled_rows] == [16, 0]
    assert all(len(row["samples"]) == 32 for row in sampled_rows)
    assert all(
        set(sample) == {"response", "correct"}
        for row in sampled_rows
        for sample in row["samples"]
    )
    assert [sample["correct"] for sample in sampled_rows[0]["samples"]] == [
        *([True] * 16),
        *([False] * 16),
    ]


def test_remote_base_evaluation_pins_canonical_revision(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_snapshot_download(*, repo_id, revision):
        captured.update(repo_id=repo_id, revision=revision)
        return str(tmp_path)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )
    model, revision = resolve_model(BASE_MODEL_ID, "base")

    assert captured == {"repo_id": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION}
    assert model == str(tmp_path.resolve())
    assert revision == BASE_MODEL_REVISION


def test_local_base_evaluation_uses_local_snapshot_without_remote_revision(
    tmp_path,
) -> None:
    model, revision = resolve_model(str(tmp_path), "base")

    assert model == str(tmp_path.resolve())
    assert revision is None
