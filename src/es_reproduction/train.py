from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import socket
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import ArtifactConfig, load_canonical_config
from .data import build_training_schedule, dataset_manifest, load_split
from .prompts import render_gsm8k_prompt
from .rewards import score_training_response


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_population_rewards(
    rewards: np.ndarray,
    *,
    ddof: int,
    epsilon: float,
) -> np.ndarray:
    mean = float(np.mean(rewards))
    standard_deviation = float(np.std(rewards, ddof=int(ddof)))
    return (rewards - mean) / (standard_deviation + float(epsilon))


def shard_seeds(
    seeds: list[int],
    num_engines: int,
) -> list[tuple[list[int], list[int]]]:
    if num_engines < 1:
        raise ValueError("num_engines must be positive")
    shard_size = math.ceil(len(seeds) / num_engines)
    shards = []
    for index in range(num_engines):
        start = index * shard_size
        end = min(start + shard_size, len(seeds))
        if start < len(seeds):
            shards.append((list(range(start, end)), seeds[start:end]))
    return shards


def synchronization_due(
    *,
    completed_step: int,
    engine_sync_every: int,
    checkpoint_due: bool,
) -> bool:
    periodic_sync = (
        int(engine_sync_every) > 0
        and int(completed_step) % int(engine_sync_every) == 0
    )
    return periodic_sync or bool(checkpoint_due)


def _extract_texts(outputs: list[Any]) -> list[str]:
    texts = []
    for request_output in outputs:
        candidates = getattr(request_output, "outputs", None)
        if not candidates:
            raise RuntimeError("vLLM returned a request without a generated candidate")
        texts.append(str(candidates[0].text))
    return texts


class EngineEvaluator:
    def __init__(self, engine: Any, format_reward: float):
        self.engine = engine
        self.format_reward = float(format_reward)

    def evaluate_perturbations(
        self,
        seeds: list[int],
        sigma: float,
        prompts: list[str],
        answers: list[str],
        max_tokens: int,
    ) -> list[float]:
        import ray
        from vllm import SamplingParams

        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_tokens))
        rewards = []
        for seed in seeds:
            ray.get(
                self.engine.collective_rpc.remote(
                    "apply_perturbation",
                    args=[int(seed), float(sigma)],
                )
            )
            try:
                outputs = ray.get(self.engine.generate.remote(prompts, sampling))
            finally:
                ray.get(
                    self.engine.collective_rpc.remote(
                        "revert_perturbation",
                        args=[int(seed), float(sigma)],
                    )
                )
            response_rewards = [
                score_training_response(
                    response,
                    gold,
                    format_reward=self.format_reward,
                )
                for response, gold in zip(_extract_texts(outputs), answers, strict=True)
            ]
            rewards.append(float(np.mean(response_rewards)) if response_rewards else 0.0)
        return rewards


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _resolve_model(model: str) -> str:
    candidate = Path(model).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    if model != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError(
            "The canonical artifact accepts a local model directory or "
            "Qwen/Qwen2.5-1.5B-Instruct"
        )
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model,
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )


def _launch_engines(
    *,
    model: str,
    config: ArtifactConfig,
    num_engines: int,
) -> tuple[list[Any], list[Any], list[Any]]:
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from vllm import LLM

    class RayLLM(LLM):
        def __init__(self, *args: Any, **kwargs: Any):
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            super().__init__(*args, **kwargs)

    ray.init(ignore_reinit_error=True)
    placement_groups = [
        placement_group([{"GPU": 1, "CPU": 1}])
        for _ in range(num_engines)
    ]
    ray.get([group.ready() for group in placement_groups])
    engine_actor = ray.remote(num_cpus=1, num_gpus=1)(RayLLM)
    evaluator_actor = ray.remote(EngineEvaluator)

    engines = []
    evaluators = []
    for group in placement_groups:
        engine = engine_actor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=group,
            )
        ).remote(
            model=model,
            tensor_parallel_size=1,
            distributed_executor_backend="mp",
            worker_extension_cls="es_reproduction.worker.WorkerExtension",
            dtype=config.generation.dtype,
            gpu_memory_utilization=config.generation.gpu_memory_utilization,
            enable_prefix_caching=False,
            disable_log_stats=True,
            max_model_len=config.generation.max_model_length,
            trust_remote_code=True,
        )
        result = ray.get(engine.collective_rpc.remote("init_es"))
        payload = result[0] if isinstance(result, list) else result
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise RuntimeError(f"ES worker initialization failed: {result!r}")
        engines.append(engine)
        evaluators.append(evaluator_actor.remote(engine, config.format_reward))
    return engines, evaluators, placement_groups


def _initialize_sync(engines: list[Any]) -> None:
    import ray

    metadata = []
    for engine in engines:
        result = ray.get(engine.collective_rpc.remote("sync_preflight"))
        metadata.append(result[0] if isinstance(result, list) else result)
    hostnames = {str(item["hostname"]) for item in metadata}
    tensor_counts = {int(item["n_tensors"]) for item in metadata}
    if len(hostnames) != 1 or len(tensor_counts) != 1:
        raise RuntimeError("All ES engines must be homogeneous and located on one host")
    port = _open_port()
    futures = [
        engine.collective_rpc.remote(
            "init_sync",
            args=["127.0.0.1", port, rank, len(engines)],
        )
        for rank, engine in enumerate(engines)
    ]
    ray.get(futures)


def _synchronize_engines(engines: list[Any]) -> None:
    import ray

    ray.get(
        [
            engine.collective_rpc.remote("broadcast_parameters", args=[0])
            for engine in engines
        ]
    )


def _export_checkpoint(
    *,
    engine: Any,
    tokenizer: Any,
    output_dir: Path,
    completed_step: int,
) -> Path:
    import ray

    checkpoint_dir = output_dir / f"step_{completed_step}"
    result = ray.get(
        engine.collective_rpc.remote(
            "export_model",
            args=[str(checkpoint_dir.resolve())],
        )
    )
    payload = result[0] if isinstance(result, list) else result
    if not isinstance(payload, dict) or not payload.get("ok"):
        raise RuntimeError(f"Checkpoint export failed: {result!r}")
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_training(
    *,
    model: str,
    output_dir: Path,
    data_dir: Path,
    mode: str,
) -> dict[str, Any]:
    import ray
    import torch
    from ray.util.placement_group import remove_placement_group
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from transformers import AutoTokenizer

    config = load_canonical_config(repository_root() / "configs" / "easy_qwen25_1p5b.yaml")
    if mode not in {"two-epochs", "smoke"}:
        raise ValueError(f"Unknown training mode: {mode}")

    if mode == "smoke":
        required_gpus = 1
        population_size = 2
        max_steps = 1
        max_tokens = min(config.generation.max_new_tokens, 128)
    else:
        required_gpus = config.training.num_engines
        population_size = config.training.population_size
        max_steps = config.training.max_steps
        max_tokens = config.generation.max_new_tokens
    if torch.cuda.device_count() != required_gpus:
        raise RuntimeError(
            f"{mode} requires exactly {required_gpus} visible GPU(s); "
            f"found {torch.cuda.device_count()}"
        )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = _resolve_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    questions, answers = load_split(data_dir, "train")
    schedule = build_training_schedule(
        n_train=len(questions),
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        seed=config.training.seed,
    )[:max_steps]
    if mode == "smoke":
        schedule = [schedule[0][:2]]

    manifest = {
        "artifact_config": asdict(config),
        "dataset": dataset_manifest(data_dir),
        "mode": mode,
        "model": {
            "name": Path(model_path).name,
            "base_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        },
        "visible_gpus": torch.cuda.device_count(),
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    engines: list[Any] = []
    evaluators: list[Any] = []
    placement_groups: list[Any] = []
    metrics_path = output_dir / "metrics.jsonl"
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir()
    event_writer = EventFileWriter(str(tensorboard_dir))
    rng = np.random.default_rng(config.training.seed)
    history = []
    try:
        engines, evaluators, placement_groups = _launch_engines(
            model=model_path,
            config=config,
            num_engines=required_gpus,
        )
        if required_gpus > 1:
            _initialize_sync(engines)

        for zero_based_step, batch_indices in enumerate(schedule):
            started = time.monotonic()
            prompts = [
                render_gsm8k_prompt(questions[index], tokenizer)
                for index in batch_indices
            ]
            golds = [answers[index] for index in batch_indices]
            seeds = rng.integers(0, 2**32, size=population_size).tolist()
            shards = shard_seeds(seeds, required_gpus)
            futures = [
                (
                    indices,
                    evaluators[engine_index].evaluate_perturbations.remote(
                        sub_seeds,
                        config.training.sigma,
                        prompts,
                        golds,
                        max_tokens,
                    ),
                )
                for engine_index, (indices, sub_seeds) in enumerate(shards)
            ]
            rewards = np.zeros(population_size, dtype=np.float64)
            for indices, future in futures:
                partial = ray.get(future)
                for index, reward in zip(indices, partial, strict=True):
                    rewards[index] = float(reward)
            normalized = normalize_population_rewards(
                rewards,
                ddof=config.training.reward_normalization_ddof,
                epsilon=config.training.reward_normalization_epsilon,
            )
            ray.get(
                [
                    engine.collective_rpc.remote(
                        "apply_population_update",
                        args=[
                            [int(seed) for seed in seeds],
                            normalized.tolist(),
                            config.training.learning_rate,
                        ],
                    )
                    for engine in engines
                ]
            )

            completed_step = zero_based_step + 1
            row = {
                "step": completed_step,
                "mean_reward": float(np.mean(rewards)),
                "max_reward": float(np.max(rewards)),
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            summary = Summary(
                value=[
                    Summary.Value(tag="train/mean_reward", simple_value=row["mean_reward"]),
                    Summary.Value(tag="train/max_reward", simple_value=row["max_reward"]),
                ]
            )
            event_writer.add_event(
                Event(wall_time=time.time(), step=completed_step, summary=summary)
            )
            event_writer.flush()
            print(
                f"step={completed_step}/{max_steps} "
                f"mean_reward={row['mean_reward']:.17g} "
                f"max_reward={row['max_reward']:.17g}",
                flush=True,
            )

            checkpoint_due = (
                mode == "two-epochs"
                and completed_step in config.training.checkpoint_steps
            )
            sync_due = (
                required_gpus > 1
                and synchronization_due(
                    completed_step=completed_step,
                    engine_sync_every=config.training.engine_sync_every,
                    checkpoint_due=checkpoint_due,
                )
            )
            if sync_due:
                _synchronize_engines(engines)
            if checkpoint_due:
                _export_checkpoint(
                    engine=engines[0],
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    completed_step=completed_step,
                )

        return {"history": history, "output_dir": str(output_dir)}
    finally:
        event_writer.close()
        for group in placement_groups:
            with contextlib.suppress(Exception):
                remove_placement_group(group)
        if ray.is_initialized():
            ray.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical ES training protocol")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument(
        "--mode",
        choices=("two-epochs", "smoke"),
        default="two-epochs",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_training(
        model=args.model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
