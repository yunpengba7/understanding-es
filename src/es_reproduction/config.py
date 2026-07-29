from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    population_size: int
    batch_size: int
    sigma: float
    learning_rate: float
    epochs: int
    max_steps: int
    seed: int
    num_engines: int
    reward_normalization_ddof: int
    reward_normalization_epsilon: float
    engine_sync_every: int
    checkpoint_steps: tuple[int, ...]


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    max_model_length: int
    gpu_memory_utilization: float
    dtype: str


@dataclass(frozen=True)
class EvaluationConfig:
    temperature: float
    top_p: float
    num_samples: int
    max_new_tokens: int
    max_model_length: int
    gpu_memory_utilization: float
    dtype: str
    seed: int


@dataclass(frozen=True)
class ArtifactConfig:
    training: TrainingConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    format_reward: float
    prompt_mode: str


def _require_mapping(payload: Any, field: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be a mapping")
    return payload


def load_canonical_config(path: str | Path) -> ArtifactConfig:
    config_path = Path(path)
    payload = _require_mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    training = _require_mapping(payload.get("training"), "training")
    generation = _require_mapping(payload.get("generation"), "generation")
    evaluation = _require_mapping(payload.get("evaluation"), "evaluation")

    result = ArtifactConfig(
        training=TrainingConfig(
            population_size=int(training["population_size"]),
            batch_size=int(training["batch_size"]),
            sigma=float(training["sigma"]),
            learning_rate=float(training["learning_rate"]),
            epochs=int(training["epochs"]),
            max_steps=int(training["max_steps"]),
            seed=int(training["seed"]),
            num_engines=int(training["num_engines"]),
            reward_normalization_ddof=int(training["reward_normalization_ddof"]),
            reward_normalization_epsilon=float(training["reward_normalization_epsilon"]),
            engine_sync_every=int(training["engine_sync_every"]),
            checkpoint_steps=tuple(int(step) for step in training["checkpoint_steps"]),
        ),
        generation=GenerationConfig(
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            max_new_tokens=int(generation["max_new_tokens"]),
            max_model_length=int(generation["max_model_length"]),
            gpu_memory_utilization=float(generation["gpu_memory_utilization"]),
            dtype=str(generation["dtype"]),
        ),
        evaluation=EvaluationConfig(
            temperature=float(evaluation["temperature"]),
            top_p=float(evaluation["top_p"]),
            num_samples=int(evaluation["num_samples"]),
            max_new_tokens=int(evaluation["max_new_tokens"]),
            max_model_length=int(evaluation["max_model_length"]),
            gpu_memory_utilization=float(evaluation["gpu_memory_utilization"]),
            dtype=str(evaluation["dtype"]),
            seed=int(evaluation["seed"]),
        ),
        format_reward=float(payload["format_reward"]),
        prompt_mode=str(payload["prompt_mode"]),
    )
    validate_canonical_config(result)
    return result


def validate_canonical_config(config: ArtifactConfig) -> None:
    if config.training.num_engines != 8:
        raise ValueError("The paper reproduction run requires exactly eight engines")
    if config.training.population_size != 32:
        raise ValueError("The paper reproduction run requires population_size=32")
    if config.training.max_steps != 234:
        raise ValueError("The canonical two-epoch run requires max_steps=234")
    if config.training.checkpoint_steps != (117, 234):
        raise ValueError("The canonical checkpoint steps must be 117 and 234")
    if config.generation.temperature != 0.0 or config.generation.top_p != 1.0:
        raise ValueError("Canonical training generation must be greedy")
    if config.evaluation.num_samples != 32:
        raise ValueError("Canonical sampled evaluation requires 32 retained samples")
