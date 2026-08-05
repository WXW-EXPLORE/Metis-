from ..utils import DecoderLayerWrapperForMetis, CausalLMWrapperForMetis
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, eager_attention_forward, Qwen3_5TextModel
import torch
from torch.utils.checkpoint import checkpoint as _ckpt
from transformers import Cache
from transformers.utils import TransformersKwargs
from typing_extensions import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import create_causal_mask
from typing import Callable
import torch.nn as nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast

# Qwen3_5DynamicCache was introduced in a later transformers version.
# Fall back to DynamicCache when unavailable (training uses use_cache=False,
# so the custom cache class is only needed for autoregressive generation).
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
except ImportError:
    from transformers import DynamicCache as Qwen3_5DynamicCache 
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.modeling_outputs import BaseModelOutputWithPast


def _zero3_safe_checkpoint(function, hidden_states):
    """Checkpoint frozen-backbone computation safely under ZeRO-3.

    Reentrant checkpoint avoids the PyTorch 2.5 non-reentrant saved-tensor
    metadata conflict with ZeRO-3 parameter re-partitioning.

    If the input does not require gradients, the enclosed backbone operation
    is fully frozen, so checkpointing it provides no backward-memory benefit.
    """
    if not torch.is_grad_enabled() or not hidden_states.requires_grad:
        return function(hidden_states)

    return _ckpt(
        function,
        hidden_states,
        use_reentrant=True,
        preserve_rng_state=True,
    )


def _create_qwen3_5_causal_mask(
    *,
    config,
    inputs_embeds,
    attention_mask,
    past_key_values,
    position_ids,
):
    """Call the Transformers 5.x mask API shared by 5.4 and 5.10.

    Transformers 5.4 accepted a deprecated ``cache_position`` keyword but did
    not use it. Transformers 5.10 removed that keyword. Omitting it preserves
    5.4 behavior and keeps the wrapper compatible with both releases.
    """

    return create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


def _prepare_qwen3_5_generation_cache(
    past_key_values,
    *,
    config,
    use_cache,
):
    """Create the configured hybrid cache on the first generation call.

    Transformers generation may pre-create an empty generic cache. In 5.4 the
    Qwen-specific cache class must replace it; in 5.10 ``DynamicCache`` uses
    the model config to build the full/linear hybrid layers. Replacing only an
    empty cache is safe and preserves the populated cache on later decode
    steps.
    """

    if not use_cache:
        return past_key_values
    if past_key_values is None or past_key_values.get_seq_length() == 0:
        return Qwen3_5DynamicCache(config=config)
    return past_key_values


class Qwen3_5DecoderLayerForMetis(DecoderLayerWrapperForMetis):
    def __init__(self, config, raw_decoder):
        super().__init__(config, raw_decoder)

    # Modified from Qwen3.5
    def before_mixin(self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if self.raw_decoder.layer_type == "linear_attention":
            return None, None, None, None

        self_attn = self.raw_decoder.self_attn
        use_ckpt = getattr(self, "_use_gradient_checkpointing", False) and self.training

        if not use_ckpt:
            residual = hidden_states
            hidden_states = self.raw_decoder.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn.head_dim)

            query_states, gate = torch.chunk(
                self_attn.q_proj(hidden_states).view(*input_shape, -1, self_attn.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)

            query_states = self_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self_attn.k_norm(self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            memory_for_query = query_states.clone()

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self_attn.layer_idx, cache_kwargs
                )

            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self_attn.config._attn_implementation, eager_attention_forward
            )

            attn_output, attn_weights = attention_interface(
                self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self_attn.attention_dropout,
                scaling=self_attn.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self_attn.o_proj(attn_output)
            return memory_for_query, attn_output, {'residual': residual}, self_attn.o_proj

        def _attn(hidden_states):
            residual = hidden_states
            normed = self.raw_decoder.input_layernorm(hidden_states)

            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn.head_dim)

            query_states, gate = torch.chunk(
                self_attn.q_proj(normed).view(*input_shape, -1, self_attn.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)

            query_states = self_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self_attn.k_norm(self_attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
            value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

            memory_for_query = query_states.clone()

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self_attn.layer_idx, cache_kwargs
                )

            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self_attn.config._attn_implementation, eager_attention_forward
            )

            attn_output, attn_weights = attention_interface(
                self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self_attn.attention_dropout,
                scaling=self_attn.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self_attn.o_proj(attn_output)
            return memory_for_query, attn_output, residual

        memory_for_query, attn_output, residual = _zero3_safe_checkpoint(
            _attn,
            hidden_states,
        )

        return memory_for_query, attn_output, {'residual': residual}, self_attn.o_proj
    
    def after_mixin(self, memory_carrier,
        cache_dict,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        use_ckpt = getattr(self, "_use_gradient_checkpointing", False) and self.training

        if not use_ckpt:
            if self.raw_decoder.layer_type == "linear_attention":
                residual = hidden_states
                hidden_states = self.raw_decoder.input_layernorm(hidden_states)
                hidden_states = self.raw_decoder.linear_attn(
                    hidden_states=hidden_states,
                    cache_params=past_key_values,
                    attention_mask=attention_mask,
                )
            elif self.raw_decoder.layer_type == "full_attention":
                residual = cache_dict['residual']
                hidden_states = memory_carrier

            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.raw_decoder.post_attention_layernorm(hidden_states)
            hidden_states = self.raw_decoder.mlp(hidden_states)
            return residual + hidden_states

        if self.raw_decoder.layer_type == "linear_attention":
            def _mix(hidden_states):
                residual = hidden_states
                normed = self.raw_decoder.input_layernorm(hidden_states)
                mixed = self.raw_decoder.linear_attn(
                    hidden_states=normed,
                    cache_params=past_key_values,
                    attention_mask=attention_mask,
                )
                return residual + mixed

            hidden_states = _zero3_safe_checkpoint(_mix, hidden_states)

        elif self.raw_decoder.layer_type == "full_attention":
            residual = cache_dict['residual']
            hidden_states = residual + memory_carrier

        def _mlp(hidden_states):
            residual = hidden_states
            normed = self.raw_decoder.post_attention_layernorm(hidden_states)
            return residual + self.raw_decoder.mlp(normed)

        hidden_states = _zero3_safe_checkpoint(_mlp, hidden_states)

        return hidden_states
    

class Qwen3_5CausalLMForMetis(CausalLMWrapperForMetis):
    def __init__(self, config):
        super().__init__(config)

        self.model = Qwen3_5TextModel(config.backbone_configs.text_config)

        self.vocab_size = config.backbone_configs.text_config.vocab_size
        self.lm_head = nn.Linear(config.backbone_configs.text_config.hidden_size, config.backbone_configs.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.model.post_init()

    def get_decoder_layer_by_id(self, layer_id: int):
        return self.model.layers[layer_id]
    
    @merge_with_config_defaults
    @capture_outputs
    def forward_with_memory(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        past_key_values = _prepare_qwen3_5_generation_cache(
            past_key_values,
            config=self.model.config,
            use_cache=use_cache,
        )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # mrope: the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        causal_mask = _create_qwen3_5_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self.model._update_linear_attn_mask(attention_mask, past_key_values)

        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        all_hidden_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.model.layers[: self.model.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Use Metis Block.
            hidden_states = self._metis_blocks_ref[layer_idx](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.model.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states
        )
