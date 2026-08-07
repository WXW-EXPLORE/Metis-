"""Small generation-config helpers shared by Metis and base-model runners."""

from __future__ import annotations

from typing import Any


def resolve_generation_special_token_ids(
    model: Any,
    tokenizer: Any,
) -> tuple[int | list[int], int]:
    """Resolve model stops without dropping the tokenizer's chat boundary.

    Some Qwen checkpoints retain ``<|endoftext|>`` in the model generation
    config while the tokenizer uses ``<|im_end|>`` to end an assistant turn.
    Generation must stop on either token. Gemma's model-native multi-EOS list
    and valid PAD token zero must remain unchanged.
    """

    generation_config = getattr(model, "generation_config", None)
    model_eos_token_id = getattr(generation_config, "eos_token_id", None)
    tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(generation_config, "pad_token_id", None)

    def as_list(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    eos_ids = as_list(model_eos_token_id)
    for token_id in as_list(tokenizer_eos_token_id):
        if token_id not in eos_ids:
            eos_ids.append(token_id)
    if not eos_ids:
        raise ValueError("Cannot resolve an EOS token id")
    eos_token_id: int | list[int] = (
        eos_ids
        if isinstance(model_eos_token_id, (list, tuple)) or len(eos_ids) > 1
        else eos_ids[0]
    )

    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if isinstance(eos_token_id, (list, tuple)):
            if not eos_token_id:
                raise ValueError("Cannot resolve pad token from an empty EOS list")
            pad_token_id = eos_token_id[0]
        else:
            pad_token_id = eos_token_id

    return eos_token_id, pad_token_id


def decode_generated_response(
    tokenizer: Any,
    generated_ids: Any,
) -> str:
    """Decode chat output and remove model-native reasoning channel markup."""

    raw_response = tokenizer.decode(
        generated_ids, skip_special_tokens=False
    )
    response = None
    parse_response = getattr(tokenizer, "parse_response", None)
    if callable(parse_response):
        try:
            parsed = parse_response(raw_response)
        except (AttributeError, TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, str):
                response = content
        elif isinstance(parsed, str):
            response = parsed

    if response is None:
        response = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

    # Gemma 4 may greedily repeat the already-prefilled disabled-thinking
    # channel tail as `thought\n<channel|>` before its final answer. The
    # tokenizer parser intentionally leaves that incomplete leading channel
    # untouched, so remove only this exact structural prefix.
    duplicate_disabled_thinking_prefix = "thought\n<channel|>"
    if response.startswith(duplicate_disabled_thinking_prefix):
        response = response[len(duplicate_disabled_thinking_prefix) :]
    return response.strip()
