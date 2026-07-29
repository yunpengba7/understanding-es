from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from typing import Any

import torch

PACKED_MODULES = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _find_model(worker: Any) -> Any:
    if hasattr(worker, "gpu_model_runner") and hasattr(worker.gpu_model_runner, "model"):
        return worker.gpu_model_runner.model
    if hasattr(worker, "model_runner") and hasattr(worker.model_runner, "model"):
        return worker.model_runner.model
    if hasattr(worker, "model"):
        return worker.model
    raise RuntimeError("Cannot locate the model inside the vLLM worker")


def _stable_tensor_id(name: str) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _mixed_seed(base_seed: int, tensor_id: int) -> int:
    return (int(base_seed) ^ int(tensor_id)) & 0xFFFFFFFFFFFFFFFF


def _named_parameters(model: Any) -> list[tuple[str, int, torch.Tensor]]:
    seen: set[int] = set()
    result = []
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result.append((str(name), _stable_tensor_id(str(name)), parameter))
    return result


@torch.no_grad()
def _apply_seeded_noise(
    parameter_infos: list[tuple[str, int, torch.Tensor]],
    *,
    seed: int,
    sigma: float,
) -> None:
    for _, tensor_id, parameter in parameter_infos:
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed(_mixed_seed(seed, tensor_id))
        delta = torch.randn(
            parameter.shape,
            generator=generator,
            dtype=torch.float32,
            device=parameter.device,
        )
        delta.mul_(float(sigma))
        parameter.add_(delta.to(dtype=parameter.dtype))


@torch.no_grad()
def _apply_population_update(
    parameter_infos: list[tuple[str, int, torch.Tensor]],
    *,
    seeds: list[int],
    weights: list[float],
    learning_rate: float,
) -> None:
    if len(seeds) != len(weights) or not seeds:
        raise ValueError("seeds and weights must be non-empty and have equal lengths")
    scale = float(learning_rate) / len(seeds)
    for _, tensor_id, parameter in parameter_infos:
        total_delta = torch.zeros(
            parameter.shape,
            dtype=torch.float32,
            device=parameter.device,
        )
        for seed, weight in zip(seeds, weights, strict=True):
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(_mixed_seed(seed, tensor_id))
            noise = torch.randn(
                parameter.shape,
                generator=generator,
                dtype=torch.float32,
                device=parameter.device,
            )
            total_delta.add_(noise, alpha=float(scale * weight))
        parameter.add_(total_delta.to(dtype=parameter.dtype))


def _packed_sizes(name: str, tensor: torch.Tensor, config: Any) -> tuple[int, ...]:
    if name == "qkv_proj":
        hidden_size = int(config.hidden_size)
        attention_heads = int(config.num_attention_heads)
        key_value_heads = int(getattr(config, "num_key_value_heads", attention_heads))
        head_dim = int(getattr(config, "head_dim", hidden_size // attention_heads))
        sizes = (
            attention_heads * head_dim,
            key_value_heads * head_dim,
            key_value_heads * head_dim,
        )
    elif name == "gate_up_proj":
        intermediate_size = int(getattr(config, "intermediate_size", tensor.shape[0] // 2))
        sizes = (intermediate_size, intermediate_size)
    else:
        raise ValueError(f"Unsupported packed module: {name}")
    if sum(sizes) != int(tensor.shape[0]):
        raise ValueError(f"Packed sizes {sizes} do not cover tensor shape {tuple(tensor.shape)}")
    return sizes


def to_hf_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if ".attn._" in name:
            continue
        for packed_name, unpacked_names in PACKED_MODULES.items():
            marker = f".{packed_name}."
            if marker not in name:
                continue
            shards = torch.split(tensor, _packed_sizes(packed_name, tensor, config), dim=0)
            for unpacked_name, shard in zip(unpacked_names, shards, strict=True):
                converted[name.replace(marker, f".{unpacked_name}.", 1)] = (
                    shard.detach().cpu().contiguous().clone()
                )
            break
        else:
            converted[name] = tensor.detach().cpu().contiguous().clone()
    return converted


def _engine_sync_communicator(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[Any, Any]:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    group = StatelessProcessGroup.create(
        host=str(host),
        port=int(port),
        rank=int(rank),
        world_size=int(world_size),
    )
    communicator = PyNcclCommunicator(group, device=device)
    if not bool(getattr(communicator, "available", False)) or bool(
        getattr(communicator, "disabled", True)
    ):
        raise RuntimeError("The NCCL communicator is unavailable")
    return group, communicator


class WorkerExtension:
    @torch.no_grad()
    def init_es(self) -> dict[str, Any]:
        model = _find_model(self)
        self._es_parameter_infos = _named_parameters(model)
        self._es_is_perturbed = False
        return {"ok": True, "n_tensors": len(self._es_parameter_infos)}

    def _parameters(self) -> list[tuple[str, int, torch.Tensor]]:
        parameters = getattr(self, "_es_parameter_infos", None)
        if parameters is None:
            raise RuntimeError("The ES worker has not been initialized")
        return parameters

    @torch.no_grad()
    def apply_perturbation(self, seed: int, sigma: float) -> dict[str, bool]:
        if bool(getattr(self, "_es_is_perturbed", False)):
            raise RuntimeError("The worker is already perturbed")
        _apply_seeded_noise(self._parameters(), seed=int(seed), sigma=float(sigma))
        self._es_is_perturbed = True
        return {"ok": True}

    @torch.no_grad()
    def revert_perturbation(self, seed: int, sigma: float) -> dict[str, bool]:
        if not bool(getattr(self, "_es_is_perturbed", False)):
            raise RuntimeError("The worker is not perturbed")
        _apply_seeded_noise(self._parameters(), seed=int(seed), sigma=-float(sigma))
        self._es_is_perturbed = False
        return {"ok": True}

    @torch.no_grad()
    def apply_population_update(
        self,
        seeds: list[int],
        weights: list[float],
        learning_rate: float,
    ) -> dict[str, bool]:
        if bool(getattr(self, "_es_is_perturbed", False)):
            raise RuntimeError("Population updates require an unperturbed model")
        _apply_population_update(
            self._parameters(),
            seeds=[int(seed) for seed in seeds],
            weights=[float(weight) for weight in weights],
            learning_rate=float(learning_rate),
        )
        return {"ok": True}

    @torch.no_grad()
    def sync_preflight(self) -> dict[str, Any]:
        parameters = sorted(self._parameters(), key=lambda item: item[0])
        if bool(getattr(self, "_es_is_perturbed", False)):
            raise RuntimeError("Synchronization requires an unperturbed model")
        devices = {parameter.device for _, _, parameter in parameters}
        if len(devices) != 1:
            raise RuntimeError("Each engine must keep all parameters on one device")
        device = next(iter(devices))
        if device.type != "cuda":
            raise RuntimeError("Synchronization requires CUDA parameters")
        if any(not parameter.is_contiguous() for _, _, parameter in parameters):
            raise RuntimeError("Synchronization requires contiguous parameters")
        return {
            "ok": True,
            "hostname": socket.gethostname(),
            "device": str(device),
            "n_tensors": len(parameters),
        }

    @torch.no_grad()
    def init_sync(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
    ) -> dict[str, Any]:
        metadata = self.sync_preflight()
        device = next(iter({parameter.device for _, _, parameter in self._parameters()}))
        group, communicator = _engine_sync_communicator(
            host,
            int(port),
            int(rank),
            int(world_size),
            device,
        )
        self._es_sync_group = group
        self._es_sync_communicator = communicator
        return {
            "ok": True,
            "rank": int(rank),
            "world_size": int(world_size),
            "n_tensors": metadata["n_tensors"],
        }

    @torch.no_grad()
    def broadcast_parameters(self, source_rank: int = 0) -> dict[str, Any]:
        communicator = getattr(self, "_es_sync_communicator", None)
        if communicator is None:
            raise RuntimeError("The synchronization communicator is not initialized")
        parameters = sorted(self._parameters(), key=lambda item: item[0])
        device = parameters[0][2].device
        stream = torch.cuda.current_stream(device=device)
        num_bytes = 0
        for _, _, parameter in parameters:
            communicator.broadcast(parameter, src=int(source_rank), stream=stream)
            num_bytes += parameter.numel() * parameter.element_size()
        torch.cuda.synchronize(device=device)
        self._es_is_perturbed = False
        return {
            "ok": True,
            "rank": int(communicator.rank),
            "n_tensors": len(parameters),
            "num_bytes": int(num_bytes),
        }

    @torch.no_grad()
    def export_model(self, export_dir: str) -> dict[str, Any]:
        from safetensors.torch import save_file

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        model = _find_model(self)
        state = {
            name: tensor
            for name, tensor in model.state_dict().items()
            if torch.is_tensor(tensor)
        }
        converted = to_hf_state_dict(state, model.config)
        save_file(converted, str(export_path / "model.safetensors"))
        model.config.save_pretrained(export_path)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(export_path)
        return {"ok": True, "path": str(export_path), "n_tensors": len(converted)}
