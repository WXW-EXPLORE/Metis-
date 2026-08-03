"""Checkpoint helpers for Metis delta checkpoints.

The backbone is frozen during Metis training, so a normal HF checkpoint
mostly duplicates immutable backbone weights.  Delta checkpoints save only the
trainable Metis parameters plus the regular Trainer state files.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .configuration_metis import MetisConfig
from .modeling_metis import MetisForCausalLM
from .weight_utils import (
    _gather_parameters_for_update,
    _is_zero3_parameter,
    is_deepspeed_zero3_enabled,
    load_metis_from_backbone,
)

logger = logging.getLogger(__name__)

DELTA_WEIGHTS_NAME = "metis_delta.safetensors"
DELTA_MANIFEST_NAME = "metis_delta_manifest.json"
DELTA_FORMAT_VERSION = "metis_delta_v1"

FULL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def is_delta_checkpoint(checkpoint_path: str | os.PathLike[str]) -> bool:
    path = Path(checkpoint_path)
    return (path / DELTA_MANIFEST_NAME).is_file() and (path / DELTA_WEIGHTS_NAME).is_file()


def is_full_checkpoint(checkpoint_path: str | os.PathLike[str]) -> bool:
    path = Path(checkpoint_path)
    return any((path / name).is_file() for name in FULL_WEIGHT_FILES)


def is_metis_checkpoint(checkpoint_path: str | os.PathLike[str]) -> bool:
    return is_delta_checkpoint(checkpoint_path) or is_full_checkpoint(checkpoint_path)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _safe_dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _trainable_state_dict(model, state_dict: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    unwrapped = _unwrap_model(model)
    trainable_parameters = [
        (name, param)
        for name, param in unwrapped.named_parameters()
        if param.requires_grad
    ]

    # A normal state_dict from a ZeRO-3 model contains only local
    # placeholders. Gather one trainable tensor at a time instead of asking
    # Accelerate/DeepSpeed to consolidate the complete frozen 27B model.
    if state_dict is None and any(
        _is_zero3_parameter(param) for _, param in trainable_parameters
    ):
        delta: dict[str, torch.Tensor] = {}
        for name, parameter in trainable_parameters:
            with _gather_parameters_for_update([parameter]) as should_collect:
                if should_collect:
                    delta[name] = parameter.detach().cpu().clone()
        return delta

    source = state_dict if state_dict is not None else unwrapped.state_dict()
    delta: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for name, _ in trainable_parameters:
        tensor = source.get(name)
        if tensor is None:
            tensor = source.get(f"module.{name}")
        if tensor is None:
            missing.append(name)
            continue
        delta[name] = tensor.detach().cpu()

    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(
            f"Could not find {len(missing)} trainable tensors in checkpoint state_dict: {preview}"
        )
    if not delta:
        raise RuntimeError("No trainable tensors found; refusing to write an empty Metis delta checkpoint.")
    return delta


def save_metis_delta_checkpoint(
    model,
    output_dir: str | os.PathLike[str],
    *,
    tokenizer=None,
    base_model_path: str = "",
    state_dict: dict[str, torch.Tensor] | None = None,
    is_main_process: bool = True,
) -> None:
    """Save a compact checkpoint containing only trainable Metis weights."""
    output = Path(output_dir)
    unwrapped = _unwrap_model(model)
    uses_zero3_collectives = any(
        _is_zero3_parameter(parameter)
        for parameter in unwrapped.parameters()
    )

    if base_model_path:
        meta = dict(getattr(unwrapped.config, "backbone_meta", {}) or {})
        meta["backbone_path"] = base_model_path
        unwrapped.config.backbone_meta = meta

    # All ZeRO-3 ranks must participate in the per-parameter gathers above.
    # Only rank 0 receives tensors and writes files.
    delta = _trainable_state_dict(unwrapped, state_dict=state_dict)
    if not is_main_process:
        if uses_zero3_collectives:
            torch.distributed.barrier()
        return

    output.mkdir(parents=True, exist_ok=True)
    unwrapped.config.save_pretrained(output)
    generation_config = getattr(unwrapped, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output)
    if tokenizer is not None:
        tokenizer.save_pretrained(output)

    save_file(delta, str(output / DELTA_WEIGHTS_NAME))

    manifest = {
        "format": DELTA_FORMAT_VERSION,
        "base_model_path": base_model_path or getattr(unwrapped.config, "backbone_meta", {}).get("backbone_path", ""),
        "backbone_type": getattr(unwrapped.config, "backbone_meta", {}).get("backbone_type", ""),
        "num_tensors": len(delta),
        "num_parameters": int(sum(t.numel() for t in delta.values())),
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": _safe_dtype_name(tensor),
            }
            for name, tensor in sorted(delta.items())
        },
    }
    with open(output / DELTA_MANIFEST_NAME, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Saved Metis delta checkpoint -> %s (%d tensors, %.2fM parameters)",
        output,
        len(delta),
        manifest["num_parameters"] / 1_000_000,
    )
    if uses_zero3_collectives:
        torch.distributed.barrier()


def load_delta_manifest(checkpoint_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(checkpoint_path) / DELTA_MANIFEST_NAME
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("format") != DELTA_FORMAT_VERSION:
        raise ValueError(f"Unsupported Metis delta checkpoint format in {path}: {manifest.get('format')!r}")
    return manifest


def resolve_backbone_path(
    checkpoint_path: str | os.PathLike[str],
    *,
    model_path: str | None = None,
) -> str:
    """Resolve the immutable backbone path for a delta checkpoint.

    Explicit ``model_path`` wins.  ``METIS_BASE_MODEL_PATH`` is a convenient
    machine-local override when moving a delta checkpoint between machines.
    """
    if model_path:
        return model_path
    env_path = os.environ.get("METIS_BASE_MODEL_PATH") or os.environ.get("MODEL_PATH")
    if env_path:
        return env_path

    manifest = load_delta_manifest(checkpoint_path)
    if manifest.get("base_model_path"):
        return manifest["base_model_path"]

    config_path = Path(checkpoint_path) / "config.json"
    if config_path.is_file():
        with open(config_path) as f:
            config = json.load(f)
        backbone_path = (config.get("backbone_meta") or {}).get("backbone_path")
        if backbone_path:
            return backbone_path

    raise ValueError(
        f"Could not resolve backbone path for delta checkpoint {checkpoint_path}. "
        "Pass --model_path or set METIS_BASE_MODEL_PATH."
    )


def _memory_arg(memory: dict[str, Any], key: str, default: Any) -> Any:
    return memory.get(key, default)


def _build_metis_from_delta_config(
    checkpoint_path: str | os.PathLike[str],
    *,
    model_path: str | None,
    backbone_type: str | None,
    device: str | torch.device,
    dtype: torch.dtype,
) -> MetisForCausalLM:
    config = MetisConfig.from_pretrained(checkpoint_path)
    memory = dict(getattr(config, "memory_configs", {}) or {})
    backbone_meta = dict(getattr(config, "backbone_meta", {}) or {})
    base_model_path = resolve_backbone_path(checkpoint_path, model_path=model_path)
    resolved_backbone_type = backbone_type or backbone_meta.get("backbone_type") or "qwen3_5"

    model, _ = load_metis_from_backbone(
        base_model_path,
        backbone_type=resolved_backbone_type,
        device=device,
        dtype=dtype,
        metis_block_type=_memory_arg(memory, "metis_block_type", "NormedReweightLearnedQueryMetisBlock"),
        metis_hyper_memory_type=_memory_arg(memory, "metis_hyper_memory_type", "StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory"),
        metis_local_memory_type=_memory_arg(memory, "metis_local_memory_type", "NormalizedDeltaNetMetisLocalMemory"),
        update_ratio=float(_memory_arg(memory, "update_ratio", 0.9)),
        commit_hidden_offset=int(_memory_arg(memory, "commit_hidden_offset", 0)),
        mem_norm_init=float(_memory_arg(memory, "mem_norm_init", 1.0)),
        uniform_num_selected=int(_memory_arg(memory, "uniform_num_selected", 16)),
        stride_interval=int(_memory_arg(memory, "stride_interval", 8)),
        pool_temperature=float(_memory_arg(memory, "pool_temperature", 1.0)),
        gumbel_topk_noise=bool(_memory_arg(memory, "gumbel_topk_noise", True)),
        alpha_top_p=float(_memory_arg(memory, "alpha_top_p", 0.9)),
        alpha_min_tokens=int(_memory_arg(memory, "alpha_min_tokens", 1)),
        alpha_max_tokens=int(_memory_arg(memory, "alpha_max_tokens", 0)),
        alpha_max_fraction=float(_memory_arg(memory, "alpha_max_fraction", 0.0)),
        gated_delta_alpha_init=float(_memory_arg(memory, "gated_delta_alpha_init", 1.0)),
        gated_delta_beta_init=float(_memory_arg(memory, "gated_delta_beta_init", 1.0)),
        qk_kernel_type=_memory_arg(memory, "qk_kernel_type", "elu_plus_one"),
        metis_reweight_gamma=float(_memory_arg(memory, "metis_reweight_gamma", 0.9)),
    )
    model.config.backbone_meta = {
        **backbone_meta,
        "backbone_type": resolved_backbone_type,
        "backbone_path": base_model_path,
    }
    return model


def load_metis_delta_into_model(
    model,
    checkpoint_path: str | os.PathLike[str],
) -> None:
    """Load trainable Metis weights from a delta checkpoint into an existing model."""
    checkpoint = Path(checkpoint_path)
    if not is_delta_checkpoint(checkpoint):
        raise ValueError(f"Not a Metis delta checkpoint: {checkpoint}")

    unwrapped = _unwrap_model(model)
    if is_deepspeed_zero3_enabled():
        manifest = load_delta_manifest(checkpoint)
        manifest_tensors = manifest.get("tensors") or {}
        parameter_map = dict(unwrapped.named_parameters())
        file_path = checkpoint / DELTA_WEIGHTS_NAME

        with safe_open(str(file_path), framework="pt", device="cpu") as tensors:
            file_names = set(tensors.keys())
            manifest_names = set(manifest_tensors)
            if file_names != manifest_names:
                missing = sorted(manifest_names - file_names)
                unexpected = sorted(file_names - manifest_names)
                raise RuntimeError(
                    "Delta manifest and safetensors file do not match: "
                    f"missing={missing[:10]}, unexpected={unexpected[:10]}"
                )

            unknown = sorted(file_names - parameter_map.keys())
            if unknown:
                raise RuntimeError(
                    f"Delta checkpoint has {len(unknown)} unexpected keys: "
                    + ", ".join(unknown[:10])
                )

            # Validate every logical shape before entering a collective. This
            # prevents one rank from raising mid-gather while its peers wait.
            for name in sorted(file_names):
                parameter = parameter_map[name]
                logical_shape = tuple(
                    getattr(parameter, "ds_shape", parameter.shape)
                )
                checkpoint_shape = tuple(manifest_tensors[name]["shape"])
                if logical_shape != checkpoint_shape:
                    raise RuntimeError(
                        f"Delta tensor shape mismatch for {name}: "
                        f"model={logical_shape}, checkpoint={checkpoint_shape}"
                    )

            # Materialize one trainable tensor at a time. Rank 0 reads it,
            # then GatheredParameters broadcasts and re-partitions the update.
            for name in sorted(file_names):
                parameter = parameter_map[name]
                with _gather_parameters_for_update([parameter]) as should_update:
                    if should_update:
                        value = tensors.get_tensor(name)
                        parameter.copy_(
                            value.to(
                                device=parameter.device,
                                dtype=parameter.dtype,
                            )
                        )

        logger.info(
            "Loaded Metis delta weights into ZeRO-3 shards from %s (%d tensors)",
            checkpoint,
            len(file_names),
        )
        return

    delta = load_file(str(checkpoint / DELTA_WEIGHTS_NAME), device="cpu")
    incompatible = unwrapped.load_state_dict(delta, strict=False)
    if incompatible.unexpected_keys:
        preview = ", ".join(incompatible.unexpected_keys[:10])
        raise RuntimeError(
            f"Delta checkpoint has {len(incompatible.unexpected_keys)} unexpected keys: {preview}"
        )
    logger.info("Loaded Metis delta weights from %s (%d tensors)", checkpoint, len(delta))


def load_metis_model_from_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    model_path: str | None = None,
    backbone_type: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MetisForCausalLM:
    """Load either a compact delta checkpoint or a legacy full checkpoint."""
    checkpoint = Path(checkpoint_path)
    if is_delta_checkpoint(checkpoint):
        model = _build_metis_from_delta_config(
            checkpoint,
            model_path=model_path,
            backbone_type=backbone_type,
            device=device,
            dtype=dtype,
        )
        load_metis_delta_into_model(model, checkpoint)
        if is_deepspeed_zero3_enabled():
            return model
        return model.to(device=device, dtype=dtype)

    if is_full_checkpoint(checkpoint):
        model = MetisForCausalLM.from_pretrained(checkpoint, dtype=dtype)
        if is_deepspeed_zero3_enabled():
            return model
        return model.to(device=device, dtype=dtype)

    raise ValueError(
        f"{checkpoint} is neither a Metis delta checkpoint nor a legacy full checkpoint."
    )
