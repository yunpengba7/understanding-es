from __future__ import annotations

from pathlib import Path

from es_reproduction.config import load_canonical_config
from es_reproduction.data import (
    build_training_schedule,
    dataset_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_config_exposes_the_paper_protocol() -> None:
    config = load_canonical_config(ROOT / "configs" / "easy_qwen25_1p5b.yaml")

    assert config.training.population_size == 32
    assert config.training.batch_size == 64
    assert config.training.sigma == 0.0015
    assert config.training.learning_rate == 0.00025
    assert config.training.epochs == 2
    assert config.training.max_steps == 234
    assert config.training.num_engines == 8
    assert config.generation.temperature == 0.0
    assert config.generation.max_new_tokens == 2048


def test_bundled_snapshot_row_counts_and_schedule_are_stable() -> None:
    manifest = dataset_manifest(ROOT / "data" / "gsm8k")
    assert manifest == {
        "train": {"rows": 7473},
        "test": {"rows": 1319},
    }

    schedule = build_training_schedule(n_train=7473, batch_size=64, epochs=2, seed=42)
    assert len(schedule) == 234
    assert [len(schedule[index]) for index in (0, 116, 117, 233)] == [64, 49, 64, 49]
    assert schedule[0][:12] == [
        3082,
        2184,
        5897,
        2437,
        6330,
        819,
        3812,
        5878,
        5302,
        5876,
        5308,
        1534,
    ]
