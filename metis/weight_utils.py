"""Utilities for building MetisForCausalLM from a pretrained backbone."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .configuration_metis import MetisConfig
from .modeling_metis import MetisForCausalLM

# Map CLI-style lowercase names to the CamelCase names used in backbone_meta.
_BACKBONE_TYPE_MAP: dict[str, str] = {
    "qwen3_5": "Qwen3_5",
    "qwen3":   "Qwen3",
    "llama":   "Llama",
    "gemma4":  "Gemma4",
}


def is_deepspeed_zero3_enabled() -> bool:
    """Return whether Transformers currently has an active ZeRO-3 config.

    ``TrainingArguments`` must be constructed before calling the model loader
    so Transformers can register the DeepSpeed config used by
    ``PreTrainedModel._from_config`` and ``from_pretrained``.
    """
    try:
        from transformers.integrations.deepspeed import (
            is_deepspeed_zero3_enabled as transformers_zero3_enabled,
        )
    except ImportError:
        return False
    return bool(transformers_zero3_enabled())


def _is_zero3_parameter(parameter: nn.Parameter) -> bool:
    return (
        hasattr(parameter, "ds_id")
        and getattr(parameter, "ds_tensor", None) is not None
        and hasattr(parameter, "ds_numel")
    )


@contextmanager
def _gather_parameters_for_update(parameters):
    """Gather only the listed ZeRO-3 parameters and repartition on exit.

    All ranks enter the context, but only rank 0 mutates the materialized
    tensors. ``modifier_rank=0`` then broadcasts the updated values before
    DeepSpeed re-partitions them.
    """
    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if parameter is None or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        unique.append(parameter)

    zero_parameters = [parameter for parameter in unique if _is_zero3_parameter(parameter)]
    if not zero_parameters:
        yield True
        return

    import deepspeed

    with deepspeed.zero.GatheredParameters(zero_parameters, modifier_rank=0):
        yield torch.distributed.get_rank() == 0


def _copy_materialized_module_state(
    target: nn.Module,
    source: nn.Module,
    *,
    should_update: bool,
) -> bool:
    """Copy a small, already-materialized module without creating state dicts."""
    target_parameters = dict(target.named_parameters())
    source_parameters = dict(source.named_parameters())
    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    if target_parameters.keys() != source_parameters.keys():
        return False
    if target_buffers.keys() != source_buffers.keys():
        return False

    for name, target_parameter in target_parameters.items():
        source_parameter = source_parameters[name]
        if target_parameter.shape != source_parameter.shape:
            return False
        if should_update:
            target_parameter.copy_(
                source_parameter.to(
                    device=target_parameter.device,
                    dtype=target_parameter.dtype,
                )
            )

    for name, target_buffer in target_buffers.items():
        source_buffer = source_buffers[name]
        if target_buffer.shape != source_buffer.shape:
            return False
        if should_update:
            target_buffer.copy_(
                source_buffer.to(device=target_buffer.device, dtype=target_buffer.dtype)
            )
    return True


@torch.no_grad()
def _copy_module_state(target: nn.Module, source: nn.Module) -> None:
    """Copy module state, directly shard-to-shard when both models use ZeRO-3."""
    target_parameters = dict(target.named_parameters())
    source_parameters = dict(source.named_parameters())
    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())

    missing = sorted(source_parameters.keys() - target_parameters.keys())
    unexpected = sorted(target_parameters.keys() - source_parameters.keys())
    missing_buffers = sorted(source_buffers.keys() - target_buffers.keys())
    unexpected_buffers = sorted(target_buffers.keys() - source_buffers.keys())
    if missing or unexpected or missing_buffers or unexpected_buffers:
        raise RuntimeError(
            "Backbone and Metis module state do not match: "
            f"missing_parameters={missing[:5]}, "
            f"unexpected_parameters={unexpected[:5]}, "
            f"missing_buffers={missing_buffers[:5]}, "
            f"unexpected_buffers={unexpected_buffers[:5]}"
        )

    for name, target_parameter in target_parameters.items():
        source_parameter = source_parameters[name]
        target_is_zero3 = _is_zero3_parameter(target_parameter)
        source_is_zero3 = _is_zero3_parameter(source_parameter)
        if target_is_zero3 != source_is_zero3:
            raise RuntimeError(
                f"ZeRO-3 state differs for parameter {name}: "
                f"target={target_is_zero3}, source={source_is_zero3}"
            )

        if target_is_zero3:
            if int(target_parameter.ds_numel) != int(source_parameter.ds_numel):
                raise RuntimeError(
                    f"Shape mismatch for partitioned parameter {name}: "
                    f"target_numel={target_parameter.ds_numel}, "
                    f"source_numel={source_parameter.ds_numel}"
                )
            target_shard = target_parameter.ds_tensor
            source_shard = source_parameter.ds_tensor
            if target_shard.shape != source_shard.shape:
                raise RuntimeError(
                    f"Partition mismatch for parameter {name}: "
                    f"target={tuple(target_shard.shape)}, source={tuple(source_shard.shape)}"
                )
            target_shard.copy_(
                source_shard.to(device=target_shard.device, dtype=target_shard.dtype)
            )
        else:
            if target_parameter.shape != source_parameter.shape:
                raise RuntimeError(
                    f"Shape mismatch for parameter {name}: "
                    f"target={tuple(target_parameter.shape)}, "
                    f"source={tuple(source_parameter.shape)}"
                )
            target_parameter.copy_(
                source_parameter.to(
                    device=target_parameter.device,
                    dtype=target_parameter.dtype,
                )
            )

    for name, target_buffer in target_buffers.items():
        source_buffer = source_buffers[name]
        if target_buffer.shape != source_buffer.shape:
            raise RuntimeError(
                f"Shape mismatch for buffer {name}: "
                f"target={tuple(target_buffer.shape)}, source={tuple(source_buffer.shape)}"
            )
        target_buffer.copy_(
            source_buffer.to(device=target_buffer.device, dtype=target_buffer.dtype)
        )


def _fit_rows(src: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Pad or tile source rows to match the target dimension.

    When src_rows < target_rows and target_rows is a multiple of src_rows,
    each source row is repeated to fill the GQA→MHA gap (repeat_interleave).
    Otherwise the first ``min(src_rows, target_rows)`` rows are copied.
    """
    src_rows, cols = src.shape
    if src_rows >= target_rows:
        return src[:target_rows]
    if target_rows % src_rows == 0:
        return src.repeat_interleave(target_rows // src_rows, dim=0)
    return src[:target_rows]


def _extract_backbone_query_rows(
    q_tensor: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor | None:
    """Extract the query rows from a backbone q_proj tensor.

    Qwen3.5 stores q_proj output as per-head ``[query | gate]`` chunks with
    size ``head_dim * 2``.  Older backbones may store query rows directly.
    """
    if q_tensor.shape[0] == num_heads * head_dim:
        return q_tensor
    if q_tensor.shape[0] == num_heads * head_dim * 2:
        return q_tensor.view(num_heads, head_dim * 2, *q_tensor.shape[1:])[:, :head_dim].reshape(
            num_heads * head_dim,
            *q_tensor.shape[1:],
        )
    return None


@torch.no_grad()
def _init_learned_query_from_backbone(model: MetisForCausalLM) -> None:
    """Initialize trainable memory-read query layers from backbone q_proj/q_norm."""
    inited = 0
    skipped = 0
    for block in model.model.metis_blocks:
        query_proj = getattr(block, "query_proj", None)
        query_norm = getattr(block, "query_norm", None)
        if query_proj is None or query_norm is None:
            skipped += 1
            continue

        raw_decoder = block.backbone_decoder.raw_decoder
        self_attn = getattr(raw_decoder, "self_attn", None)
        src_q_proj = getattr(self_attn, "q_proj", None) if self_attn is not None else None
        src_q_norm = getattr(self_attn, "q_norm", None) if self_attn is not None else None
        if src_q_proj is None:
            skipped += 1
            continue

        num_heads = getattr(block, "query_num_heads", None)
        head_dim = getattr(block, "query_head_dim", None)
        if num_heads is None or head_dim is None:
            skipped += 1
            continue

        gathered_parameters = list(src_q_proj.parameters()) + list(query_proj.parameters())
        if query_norm is not None:
            gathered_parameters.extend(query_norm.parameters())
        if src_q_norm is not None:
            gathered_parameters.extend(src_q_norm.parameters())

        with _gather_parameters_for_update(gathered_parameters) as should_update:
            src_weight = _extract_backbone_query_rows(
                src_q_proj.weight, num_heads, head_dim
            )
            if src_weight is None or src_weight.shape != query_proj.weight.shape:
                skipped += 1
                continue

            if should_update:
                query_proj.weight.copy_(
                    src_weight.to(
                        device=query_proj.weight.device,
                        dtype=query_proj.weight.dtype,
                    )
                )

            if query_proj.bias is not None:
                src_bias = getattr(src_q_proj, "bias", None)
                if src_bias is not None:
                    src_bias = _extract_backbone_query_rows(
                        src_bias, num_heads, head_dim
                    )
                    if src_bias is not None and src_bias.shape == query_proj.bias.shape:
                        if should_update:
                            query_proj.bias.copy_(
                                src_bias.to(
                                    device=query_proj.bias.device,
                                    dtype=query_proj.bias.dtype,
                                )
                            )
                    elif should_update:
                        query_proj.bias.zero_()
                elif should_update:
                    query_proj.bias.zero_()

            if query_norm is not None and src_q_norm is not None:
                if not _copy_materialized_module_state(
                    query_norm,
                    src_q_norm,
                    should_update=should_update,
                ):
                    skipped += 1
                    continue

            inited += 1

    print(f"[weight_utils] Learned memory queries initialized from backbone q_proj/q_norm: {inited} layers (skipped {skipped}).")


@torch.no_grad()
def _init_hyper_memory_from_backbone(model: MetisForCausalLM) -> None:
    """Initialize hyper-memory W_k / W_v from decoder projections.

    Supports both layer types in Qwen3.5-style hybrid models:

    full_attention layers  → W_k ← self_attn.k_proj
                             W_v ← self_attn.v_proj
    linear_attention layers→ W_k ← in_proj_qkv key portion
                             W_v ← in_proj_qkv value portion

    Uses k_proj (not q_proj) for W_k init — W_k projects to key space
    so k_proj weights are the semantically correct starting point.
    """
    inited = 0
    skipped = 0
    for block in model.model.metis_blocks:
        hm = getattr(block, "hyper_memory", None)
        if hm is None:
            skipped += 1
            continue

        raw_decoder = block.backbone_decoder.raw_decoder
        src_k = src_v = None

        self_attn = getattr(raw_decoder, "self_attn", None)
        if self_attn is not None:
            # full_attention: k_proj → W_k,  v_proj → W_v
            src_k = getattr(self_attn, "k_proj", None)
            src_v = getattr(self_attn, "v_proj", None)
            if (
                src_k is not None
                and src_v is None
                and getattr(self_attn, "use_alternative_attention", False)
            ):
                src_v = src_k

        if src_k is None or src_v is None:
            linear_attn = getattr(raw_decoder, "linear_attn", None)
            in_proj_qkv = getattr(linear_attn, "in_proj_qkv", None) if linear_attn is not None else None
            in_proj_z   = getattr(linear_attn, "in_proj_z",   None) if linear_attn is not None else None
            if in_proj_qkv is not None and in_proj_z is not None:
                gathered_parameters = (
                    list(hm.W_k.parameters())
                    + list(hm.W_v.parameters())
                    + list(in_proj_qkv.parameters())
                    + list(in_proj_z.parameters())
                )
                with _gather_parameters_for_update(gathered_parameters) as should_update:
                    w_k = hm.W_k.weight
                    w_v = hm.W_v.weight
                    k_rows, k_cols = w_k.shape
                    v_rows, v_cols = w_v.shape

                    # in_proj_qkv layout:
                    # [Q(key_dim) | K(key_dim) | V(value_dim)]
                    qkv_out = in_proj_qkv.weight.shape[0]
                    value_dim = in_proj_z.weight.shape[0]
                    key_total = qkv_out - value_dim
                    qkv_w = in_proj_qkv.weight
                    src_k_w = qkv_w[:key_total]
                    src_v_w = qkv_w[key_total:]

                    if src_k_w.shape[1] != k_cols or src_v_w.shape[1] != v_cols:
                        skipped += 1
                        continue

                    if should_update:
                        w_k.zero_()
                        w_v.zero_()
                        w_k[: min(src_k_w.shape[0], k_rows)].copy_(
                            src_k_w[: min(src_k_w.shape[0], k_rows)]
                        )
                        w_v[: min(src_v_w.shape[0], v_rows)].copy_(
                            src_v_w[: min(src_v_w.shape[0], v_rows)]
                        )
                    inited += 1
                continue
            skipped += 1
            continue

        gathered_parameters = (
            list(hm.W_k.parameters())
            + list(hm.W_v.parameters())
            + list(src_k.parameters())
            + list(src_v.parameters())
        )
        with _gather_parameters_for_update(gathered_parameters) as should_update:
            w_k = hm.W_k.weight
            w_v = hm.W_v.weight
            k_rows, k_cols = w_k.shape
            v_rows, v_cols = w_v.shape
            k_w = src_k.weight
            v_w = src_v.weight

            if k_w.shape[1] != k_cols or v_w.shape[1] != v_cols:
                skipped += 1
                continue

            if should_update:
                w_k.zero_()
                w_v.zero_()
                # Use _fit_rows to handle GQA→MHA dimension mismatch.
                w_k[: min(k_w.shape[0], k_rows)].copy_(
                    _fit_rows(k_w, k_rows)[: min(k_w.shape[0], k_rows)]
                )
                w_v[: min(v_w.shape[0], v_rows)].copy_(
                    _fit_rows(v_w, v_rows)[: min(v_w.shape[0], v_rows)]
                )
            inited += 1

    print(f"[weight_utils] Hyper-memory initialized from backbone projections: {inited} layers (skipped {skipped}).")


def _gemma4_text_model(backbone):
    """Locate the text decoder inside Gemma 4 or Gemma 4 Unified shells."""
    candidates = [
        getattr(getattr(backbone, "model", None), "language_model", None),
        getattr(backbone, "language_model", None),
        getattr(backbone, "model", None),
    ]
    for candidate in candidates:
        if (
            candidate is not None
            and hasattr(candidate, "layers")
            and hasattr(candidate, "embed_tokens")
        ):
            return candidate
    raise RuntimeError(
        f"Could not locate the Gemma 4 text model inside {type(backbone).__name__}."
    )


def _load_gemma4_pretrained(backbone_path: str, config, dtype: torch.dtype):
    """Load the matching Hugging Face Gemma shell."""
    model_type = str(getattr(config.backbone_configs, "model_type", ""))
    if model_type.startswith("gemma4_unified"):
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise ImportError(
                "Gemma 4 Unified requires transformers>=5.10.1."
            ) from exc
        return AutoModelForMultimodalLM.from_pretrained(
            backbone_path,
            dtype=dtype,
        )
    return AutoModelForCausalLM.from_pretrained(backbone_path, dtype=dtype)


def load_metis_from_backbone(
    backbone_path: str,
    backbone_type: str = "qwen3_5",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    metis_block_type: str = "NormedReweightLearnedQueryMetisBlock",
    metis_hyper_memory_type: str = "StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory",
    metis_local_memory_type: str = "NormalizedDeltaNetMetisLocalMemory",
    update_ratio: float = 0.9,
    forget_ratio: float = 1.0,
    commit_hidden_offset: int = 0,
    mem_norm_init: float = 1.0,
    uniform_num_selected: int = 16,
    stride_interval: int = 8,
    pool_temperature: float = 1.0,
    gumbel_topk_noise: bool = True,
    alpha_top_p: float = 0.9,
    alpha_min_tokens: int = 1,
    alpha_max_tokens: int = 0,
    alpha_max_fraction: float = 0.0,
    gated_delta_alpha_init: float = 1.0,
    gated_delta_beta_init: float = 1.0,
    qk_kernel_type: str = "elu_plus_one",
    metis_reweight_gamma: float = 0.9,
    memory_on_sliding_layers: bool = True,
) -> tuple[MetisForCausalLM, "AutoTokenizer"]:
    """Build a MetisForCausalLM and load pretrained backbone weights into it.

    Args:
        backbone_path: Local path (or HF hub id) to the backbone model.
        backbone_type: One of ``qwen3_5`` / ``qwen3`` / ``llama`` /
            ``gemma4`` (lowercase).
        device: Target device.
        dtype: Weight dtype (e.g. torch.float16 / torch.bfloat16).
        metis_block_type: Which MetisBlock variant to use.
        metis_hyper_memory_type: Which HyperMemory variant to use.
        metis_local_memory_type: Which LocalMemory variant to use.
        update_ratio: Blend factor for memory write (1.0 = full update).
        forget_ratio: Stored for checkpoint/config compatibility.
        commit_hidden_offset: 0 → layer input, 1 → layer output.
        mem_norm_init: Initial value for mem_norm RMSNorm weight.
        uniform_num_selected: N for uniformly-spaced token selection.
        stride_interval: K for stride-based token selection.
        pool_temperature: Softmax temperature for attention-pooling.
        gumbel_topk_noise: Whether GumbelTopK injects noise during training.
        alpha_top_p: Probability mass for AlphaTopP token selection.
        alpha_min_tokens: Minimum tokens for AlphaTopP.
        alpha_max_tokens: Optional fixed cap for AlphaTopP.
        alpha_max_fraction: Optional length-relative cap for AlphaTopP.
        gated_delta_alpha_init: Initial sigmoid value for gated-delta alpha.
        gated_delta_beta_init: Initial sigmoid value for gated-delta beta.
        qk_kernel_type: Feature map for kernelized q/k memory variants.
        metis_reweight_gamma: Gate weight for reweight blocks.
        memory_on_sliding_layers: Whether sliding-attention layers carry
            Metis memory. Gemma full-attention-only checkpoints set this false.

    Returns:
        (model, tokenizer)
    """
    backbone_type_camel = _BACKBONE_TYPE_MAP.get(backbone_type, backbone_type)

    config = MetisConfig(
        backbone_meta={
            "backbone_type": backbone_type_camel,
            "backbone_path": backbone_path,
        },
        memory_configs={
            "metis_block_type": metis_block_type,
            "metis_hyper_memory_type": metis_hyper_memory_type,
            "metis_local_memory_type": metis_local_memory_type,
            "update_ratio": update_ratio,
            "forget_ratio": forget_ratio,
            "commit_hidden_offset": commit_hidden_offset,
            "mem_norm_init": mem_norm_init,
            "uniform_num_selected": uniform_num_selected,
            "stride_interval": stride_interval,
            "pool_temperature": pool_temperature,
            "gumbel_topk_noise": gumbel_topk_noise,
            "alpha_top_p": alpha_top_p,
            "alpha_min_tokens": alpha_min_tokens,
            "alpha_max_tokens": alpha_max_tokens,
            "alpha_max_fraction": alpha_max_fraction,
            "gated_delta_alpha_init": gated_delta_alpha_init,
            "gated_delta_beta_init": gated_delta_beta_init,
            "qk_kernel_type": qk_kernel_type,
            "metis_reweight_gamma": metis_reweight_gamma,
            "memory_on_sliding_layers": memory_on_sliding_layers,
        },
    )

    zero3_enabled = is_deepspeed_zero3_enabled()
    if zero3_enabled:
        print("[weight_utils] ZeRO-3 detected: constructing Metis directly as parameter shards.")
        model = MetisForCausalLM._from_config(config, dtype=dtype)
    else:
        model = MetisForCausalLM(config)

    print(f"[weight_utils] Loading backbone weights from {backbone_path} …")
    if backbone_type_camel == "Gemma4":
        backbone = _load_gemma4_pretrained(backbone_path, config, dtype)
    else:
        backbone = AutoModelForCausalLM.from_pretrained(backbone_path, dtype=dtype)
    mb = model.model.metis_backbone
    if backbone_type_camel == "Gemma4":
        text_model = _gemma4_text_model(backbone)
        output_embeddings = backbone.get_output_embeddings()
        if output_embeddings is None:
            output_embeddings = text_model.embed_tokens
        _copy_module_state(mb.model, text_model)
        _copy_module_state(mb.lm_head, output_embeddings)
    else:
        _copy_module_state(mb.model, backbone.model)
        _copy_module_state(mb.lm_head, backbone.lm_head)

    _init_learned_query_from_backbone(model)
    _init_hyper_memory_from_backbone(model)

    del backbone
    torch.cuda.empty_cache()
    print("[weight_utils] Backbone weights loaded.")

    if not zero3_enabled:
        model = model.to(device=device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(backbone_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
