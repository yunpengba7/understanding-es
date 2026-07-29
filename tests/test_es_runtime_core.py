from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from es_reproduction.train import (
    normalize_population_rewards,
    shard_seeds,
    synchronization_due,
)
from es_reproduction.worker import (
    _apply_population_update,
    _apply_seeded_noise,
    _stable_tensor_id,
    to_hf_state_dict,
)


def test_seed_shards_are_contiguous_and_cover_the_population() -> None:
    shards = shard_seeds(list(range(10)), 3)
    assert shards == [
        ([0, 1, 2, 3], [0, 1, 2, 3]),
        ([4, 5, 6, 7], [4, 5, 6, 7]),
        ([8, 9], [8, 9]),
    ]


def test_population_normalization_matches_the_canonical_formula() -> None:
    rewards = np.asarray([0.1, 0.4, 0.9], dtype=np.float64)
    actual = normalize_population_rewards(rewards, ddof=0, epsilon=1e-8)
    expected = (rewards - rewards.mean()) / (rewards.std(ddof=0) + 1e-8)
    np.testing.assert_array_equal(actual, expected)


def test_engine_synchronization_follows_periodic_and_checkpoint_schedules() -> None:
    checkpoint_steps = (117, 234)

    assert synchronization_due(
        completed_step=10,
        engine_sync_every=10,
        checkpoint_due=False,
    )
    assert synchronization_due(
        completed_step=117,
        engine_sync_every=100,
        checkpoint_due=117 in checkpoint_steps,
    )
    assert not synchronization_due(
        completed_step=11,
        engine_sync_every=10,
        checkpoint_due=False,
    )


def test_seeded_noise_replays_and_population_update_uses_float32_accumulation() -> None:
    parameter = torch.zeros((2, 3), dtype=torch.bfloat16)
    info = [("layer.weight", _stable_tensor_id("layer.weight"), parameter)]

    _apply_seeded_noise(info, seed=7, sigma=0.0015)
    perturbed = parameter.clone()
    _apply_seeded_noise(info, seed=7, sigma=-0.0015)
    assert not torch.equal(perturbed, parameter)

    before = parameter.clone()
    _apply_population_update(info, seeds=[7, 9], weights=[1.25, -0.5], learning_rate=0.2)
    assert not torch.equal(before, parameter)


def test_packed_qwen_weights_are_split_into_huggingface_names() -> None:
    config = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=6,
    )
    state = {
        "model.layers.0.self_attn.qkv_proj.weight": torch.arange(32).reshape(8, 4),
        "model.layers.0.mlp.gate_up_proj.weight": torch.arange(48).reshape(12, 4),
    }

    converted = to_hf_state_dict(state, config)

    assert converted["model.layers.0.self_attn.q_proj.weight"].shape == (4, 4)
    assert converted["model.layers.0.self_attn.k_proj.weight"].shape == (2, 4)
    assert converted["model.layers.0.self_attn.v_proj.weight"].shape == (2, 4)
    assert converted["model.layers.0.mlp.gate_proj.weight"].shape == (6, 4)
    assert converted["model.layers.0.mlp.up_proj.weight"].shape == (6, 4)
