"""Metis wrapper for Gemma 4 and Gemma 4 Unified text backbones."""

import importlib
from collections import UserDict
from collections.abc import Callable
from functools import lru_cache

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt
from transformers import Cache, DynamicCache, __version__ as transformers_version
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import TransformersKwargs
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from typing_extensions import Unpack

from ..utils import CausalLMWrapperForMetis, DecoderLayerWrapperForMetis


def _is_unified_text_config(text_config) -> bool:
    return str(getattr(text_config, "model_type", "")).startswith("gemma4_unified")


@lru_cache(maxsize=2)
def _load_gemma4_components(unified: bool):
    """Load the matching HF implementation without making both variants mandatory."""
    if unified:
        module_name = "transformers.models.gemma4_unified.modeling_gemma4_unified"
        class_prefix = "Gemma4Unified"
        minimum_version = "5.10.1"
    else:
        module_name = "transformers.models.gemma4.modeling_gemma4"
        class_prefix = "Gemma4"
        minimum_version = "5.5.0"

    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        variant = "Gemma 4 Unified" if unified else "Gemma 4"
        raise ImportError(
            f"{variant} is unavailable in transformers {transformers_version}. "
            f"Install transformers>={minimum_version} (5.10.1+ is recommended "
            "when both Gemma 4 checkpoints are used)."
        ) from exc

    return (
        getattr(module, f"{class_prefix}TextModel"),
        getattr(module, f"{class_prefix}TextModelOutputWithPast"),
        module.apply_rotary_pos_emb,
        module.eager_attention_forward,
    )


class Gemma4DecoderLayerForMetis(DecoderLayerWrapperForMetis):
    """Adapt one Hugging Face Gemma 4 text decoder layer to a Metis block."""

    def __init__(self, config, raw_decoder):
        super().__init__(config, raw_decoder)
        text_config = getattr(
            config.backbone_configs, "text_config", config.backbone_configs
        )
        _, _, self._apply_rotary_pos_emb, self._eager_attention_forward = (
            _load_gemma4_components(_is_unified_text_config(text_config))
        )

    def before_mixin(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        del use_cache, per_layer_input
        self_attn = self.raw_decoder.self_attn
        # Set by NonMemoryMetisBlock: no memory read happens, skip the query copy.
        skip_memory_query = getattr(self, "_skip_memory_query", False)

        def _attn(states):
            residual = states
            normed = self.raw_decoder.input_layernorm(states)
            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn.head_dim)

            query_states = self_attn.q_proj(normed).view(hidden_shape)
            query_states = self_attn.q_norm(query_states)

            # Gemma normalizes Q before RoPE.  Metis reads in that same
            # position-independent query space.
            query_for_memory = (
                None if skip_memory_query
                else query_states.transpose(1, 2).contiguous().clone()
            )

            cos, sin = position_embeddings
            query_states = self._apply_rotary_pos_emb(
                query_states, cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)

            if self_attn.is_kv_shared_layer:
                if (
                    shared_kv_states is None
                    or self_attn.layer_type not in shared_kv_states
                ):
                    raise RuntimeError(
                        f"Missing shared KV states for Gemma 4 layer {self_attn.layer_idx}."
                    )
                key_states, value_states = shared_kv_states[self_attn.layer_type]
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
            else:
                key_states = self_attn.k_proj(normed).view(hidden_shape)
                value_states = (
                    self_attn.v_proj(normed).view(hidden_shape)
                    if self_attn.v_proj is not None
                    else key_states
                )

                key_states = self_attn.k_norm(key_states)
                key_states = self._apply_rotary_pos_emb(
                    key_states, cos, sin, unsqueeze_dim=2
                ).transpose(1, 2)
                value_states = self_attn.v_norm(value_states).transpose(1, 2)

            if past_key_values is not None and not self_attn.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self_attn.layer_idx
                )
            if self_attn.store_full_length_kv:
                if shared_kv_states is None:
                    raise RuntimeError("Gemma 4 shared_kv_states was not initialized.")
                shared_kv_states[self_attn.layer_type] = key_states, value_states

            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self_attn.config._attn_implementation,
                self._eager_attention_forward,
            )
            attn_output, _ = attention_interface(
                self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self_attn.attention_dropout if self.training else 0.0,
                scaling=self_attn.scaling,
                sliding_window=self_attn.sliding_window,
                position_ids=position_ids,
                cache_position=cache_position,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self_attn.o_proj(attn_output)
            return query_for_memory, attn_output, residual

        if getattr(self, "_use_gradient_checkpointing", False) and self.training:
            # DeepSpeed ZeRO-3 releases a module's gathered parameters back to
            # zero-sized partitions after forward.  With PyTorch 2.11, the
            # non-reentrant checkpoint implementation then compares those
            # released tensors against the full forward-time metadata and
            # raises CheckpointError during recomputation.  Reentrant
            # checkpointing lets ZeRO-3's module hooks gather the parameters
            # again as part of the backward recompute.
            query_for_memory, attn_output, residual = _ckpt(
                _attn, hidden_states, use_reentrant=True
            )
        else:
            query_for_memory, attn_output, residual = _attn(hidden_states)

        return (
            query_for_memory,
            attn_output,
            {"residual": residual},
            self_attn.o_proj,
        )

    def after_mixin(
        self,
        memory_carrier,
        cache_dict,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        del hidden_states, kwargs
        decoder = self.raw_decoder
        hidden_states = cache_dict["residual"] + decoder.post_attention_layernorm(
            memory_carrier
        )

        def _mlp(states, layer_input=None):
            residual = states
            states = decoder.pre_feedforward_layernorm(states)
            states = decoder.mlp(states)

            if getattr(decoder, "enable_moe_block", False):
                states_1 = decoder.post_feedforward_layernorm_1(states)
                residual_flat = residual.reshape(-1, residual.shape[-1])
                _, top_k_weights, top_k_index = decoder.router(residual_flat)
                states_2 = decoder.pre_feedforward_layernorm_2(residual_flat)
                states_2 = decoder.experts(states_2, top_k_index, top_k_weights)
                states_2 = states_2.reshape(residual.shape)
                states_2 = decoder.post_feedforward_layernorm_2(states_2)
                states = states_1 + states_2

            states = decoder.post_feedforward_layernorm(states)
            states = residual + states

            if getattr(decoder, "hidden_size_per_layer_input", 0):
                if layer_input is None:
                    raise ValueError("Gemma 4 per-layer input is required by this config.")
                residual = states
                states = decoder.per_layer_input_gate(states)
                states = decoder.act_fn(states) * layer_input
                states = decoder.per_layer_projection(states)
                states = decoder.post_per_layer_input_norm(states)
                states = residual + states

            return states * decoder.layer_scalar

        if getattr(self, "_use_gradient_checkpointing", False) and self.training:
            if per_layer_input is None:
                hidden_states = _ckpt(_mlp, hidden_states, use_reentrant=True)
            else:
                hidden_states = _ckpt(
                    _mlp, hidden_states, per_layer_input, use_reentrant=True
                )
        else:
            hidden_states = _mlp(hidden_states, per_layer_input)

        return hidden_states


class Gemma4CausalLMForMetis(CausalLMWrapperForMetis):
    """Text-only Gemma 4 shell whose decoder layers are executed by Metis."""

    def __init__(self, config):
        super().__init__(config)
        text_config = getattr(
            config.backbone_configs, "text_config", config.backbone_configs
        )
        text_model_cls, output_cls, _, _ = _load_gemma4_components(
            _is_unified_text_config(text_config)
        )

        self.model = text_model_cls(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
        )
        self._output_cls = output_cls
        self.final_logit_softcapping = getattr(
            text_config, "final_logit_softcapping", None
        )

    def get_decoder_layer_by_id(self, layer_id: int):
        return self.model.layers[layer_id]

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits

    @merge_with_config_defaults
    @capture_outputs
    def forward_with_memory(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None and per_layer_inputs is not None:
            raise ValueError("You cannot specify per_layer_inputs if input_ids is provided")

        original_input_ids = input_ids
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if getattr(self.model, "hidden_size_per_layer_input", 0):
            if per_layer_inputs is None:
                per_layer_inputs = self.model.get_per_layer_inputs(
                    original_input_ids, inputs_embeds
                )
            per_layer_inputs = self.model.project_per_layer_inputs(
                inputs_embeds, per_layer_inputs
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.model.config)

        if position_ids is None:
            if cache_position is not None:
                position_ids = cache_position.unsqueeze(0)
            else:
                past_seen_tokens = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                position_ids = torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                ) + past_seen_tokens
                position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.model.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        output_hidden_states = kwargs.get(
            "output_hidden_states", self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.model.rotary_emb(
                hidden_states, position_ids, layer_type
            )
            for layer_type in self.model.unique_layer_types
        }
        if shared_kv_states is None:
            shared_kv_states = UserDict()

        for layer_idx in range(self.model.config.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_type = self.model.config.layer_types[layer_idx]
            per_layer_input = (
                per_layer_inputs[:, :, layer_idx, :]
                if per_layer_inputs is not None
                else None
            )
            hidden_states = self._metis_blocks_ref[layer_idx](
                hidden_states,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=causal_mask_mapping[layer_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return self._output_cls(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            shared_kv_states=(
                shared_kv_states
                if kwargs.get("return_shared_kv_states", False)
                else None
            ),
        )
