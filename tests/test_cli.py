from __future__ import annotations

import os

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
