"""LFM2 MoE backend for Glacial.

Owns all LFM2-specific execution: hybrid conv/attention layers, MoE routing,
and the prefill/decode loop with mixed state (conv state + KV cache).

The math lives in ``glacial/lfm2.py`` (proven by parity probes).  This
adapter wires it into the ``CausalLMBackend`` protocol.

State representation: a list of 24 entries where each entry is:
  - For conv layers (18): a conv state tensor [hidden, L_cache]
  - For attention layers (6): a (key, value) tuple of rank-4 tensors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glacial.generate import add_budget_telemetry, embed_input_ids, make_decode_inputs, make_inputs
from glacial.lfm2 import (
    FINAL_NORM_TENSOR,
    lfm2_rmsnorm,
    run_layer_with_optional_state,
    scalar_config,
)
from glacial.logits import chunked_last_argmax, chunked_last_logits
from glacial.sampler import Sampler
from glacial.weights import BF16_BYTES, SafetensorsWeights, WeightBudget


# ---------------------------------------------------------------------------
# Final norm + chunked LM head (LFM2-specific)
# ---------------------------------------------------------------------------

def _final_hidden_to_greedy(
    *,
    hidden,
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    scalars: dict[str, float],
    chunk_rows: int,
    budget: WeightBudget | None = None,
):
    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start, budget=budget)
    with provider.tensor(FINAL_NORM_TENSOR) as final_norm_weight:
        final_hidden = lfm2_rmsnorm(hidden, final_norm_weight, eps=scalars["norm_eps"])
        final_norm_weight_bytes = final_norm_weight.numel() * BF16_BYTES
    token_id, telemetry = chunked_last_argmax(
        final_hidden=final_hidden,
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        chunk_rows=chunk_rows,
        logits_scaling=scalars["logits_scaling"],
        budget=budget,
    )
    telemetry["final_norm_weight_bytes"] = final_norm_weight_bytes
    return final_hidden, token_id, telemetry


def _final_hidden_to_logits(
    *,
    hidden,
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    scalars: dict[str, float],
    chunk_rows: int,
    budget: WeightBudget | None = None,
):
    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start, budget=budget)
    with provider.tensor(FINAL_NORM_TENSOR) as final_norm_weight:
        final_hidden = lfm2_rmsnorm(hidden, final_norm_weight, eps=scalars["norm_eps"])
        final_norm_weight_bytes = final_norm_weight.numel() * BF16_BYTES
    logits, telemetry = chunked_last_logits(
        final_hidden=final_hidden,
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        chunk_rows=chunk_rows,
        logits_scaling=scalars["logits_scaling"],
        budget=budget,
    )
    telemetry["final_norm_weight_bytes"] = final_norm_weight_bytes
    return final_hidden, logits, telemetry


# ---------------------------------------------------------------------------
# Decode loop
# ---------------------------------------------------------------------------

def _next_token(
    *,
    token_ids: list[int],
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    config: dict[str, Any],
    lm_head_chunk_rows: int,
    budget: WeightBudget | None = None,
    sampler: Sampler | None = None,
) -> tuple[int, dict[str, Any]]:
    """No-KV path: recompute full prompt."""
    scalars = scalar_config(config)
    inputs = make_inputs(token_ids)
    hidden, embed_bytes = embed_input_ids(
        input_ids=inputs["input_ids"],
        model_file=model_file, header=header, payload_start=payload_start,
        scalars=scalars, budget=budget,
    )

    cumulative_weight_bytes = embed_bytes
    layer_stats = []
    for layer_idx in range(int(config.get("num_hidden_layers", 24))):
        hidden, _, stats = run_layer_with_optional_state(
            layer_idx=layer_idx, hidden=hidden,
            kv_pair=None, conv_state=None,
            model_file=model_file, header=header, payload_start=payload_start,
            inputs=inputs, config=config, scalars=scalars,
            return_state=False, budget=budget,
        )
        layer_stats.append(stats)
        cumulative_weight_bytes += stats["loaded_nonexpert_bytes"] + stats["cumulative_expert_weight_bytes"]

    if sampler is not None and not sampler.is_greedy():
        _fh, logits, lm_telemetry = _final_hidden_to_logits(
            hidden=hidden, model_file=model_file, header=header, payload_start=payload_start,
            scalars=scalars, chunk_rows=lm_head_chunk_rows, budget=budget,
        )
        next_id = sampler.sample(logits)
    else:
        _fh, next_id, lm_telemetry = _final_hidden_to_greedy(
            hidden=hidden, model_file=model_file, header=header, payload_start=payload_start,
            scalars=scalars, chunk_rows=lm_head_chunk_rows, budget=budget,
        )
    cumulative_weight_bytes += lm_telemetry["final_norm_weight_bytes"] + lm_telemetry["visited_lm_head_bytes"]
    telemetry = {
        "phase": "full",
        "cumulative_weight_bytes": cumulative_weight_bytes,
        "peak_selected_expert_count": max(s["selected_expert_count"] for s in layer_stats),
        "peak_expert_pair_bytes": max(s["peak_expert_pair_bytes"] for s in layer_stats),
        "selected_expert_counts": [s["selected_expert_count"] for s in layer_stats],
        "visited_lm_head_bytes": lm_telemetry["visited_lm_head_bytes"],
        "peak_lm_head_chunk_bytes": lm_telemetry["peak_lm_head_chunk_bytes"],
        "greedy_value_fp32": lm_telemetry["streaming_greedy_value_fp32"],
    }
    return int(next_id), add_budget_telemetry(telemetry, budget)


def _prefill_kv(
    *,
    token_ids: list[int],
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    config: dict[str, Any],
    lm_head_chunk_rows: int,
    budget: WeightBudget | None = None,
    sampler: Sampler | None = None,
) -> tuple[int, list[Any], dict[str, Any]]:
    """Prefill: process full prompt, return first token + mixed state list."""
    scalars = scalar_config(config)
    inputs = make_inputs(token_ids)
    hidden, embed_bytes = embed_input_ids(
        input_ids=inputs["input_ids"],
        model_file=model_file, header=header, payload_start=payload_start,
        scalars=scalars, budget=budget,
    )

    state_list: list[Any] = [None] * int(config.get("num_hidden_layers", 24))
    layer_stats = []
    visited_before_lm = embed_bytes
    for layer_idx in range(int(config.get("num_hidden_layers", 24))):
        hidden, layer_state, stats = run_layer_with_optional_state(
            layer_idx=layer_idx, hidden=hidden,
            kv_pair=None, conv_state=None,
            model_file=model_file, header=header, payload_start=payload_start,
            inputs=inputs, config=config, scalars=scalars,
            return_state=True, budget=budget,
        )
        kv_pair, conv_state = layer_state
        # Store state: KV pair for attention layers, conv state for conv layers
        state_list[layer_idx] = kv_pair if kv_pair is not None else conv_state
        layer_stats.append(stats)
        visited_before_lm += stats["loaded_nonexpert_bytes"] + stats["cumulative_expert_weight_bytes"]

    if sampler is not None and not sampler.is_greedy():
        _fh, logits, lm_telemetry = _final_hidden_to_logits(
            hidden=hidden, model_file=model_file, header=header, payload_start=payload_start,
            scalars=scalars, chunk_rows=lm_head_chunk_rows, budget=budget,
        )
        next_id = sampler.sample(logits)
    else:
        _fh, next_id, lm_telemetry = _final_hidden_to_greedy(
            hidden=hidden, model_file=model_file, header=header, payload_start=payload_start,
            scalars=scalars, chunk_rows=lm_head_chunk_rows, budget=budget,
        )
    telemetry = {
        "phase": "prefill",
        "visited_before_lm_head_bytes": visited_before_lm,
        "cumulative_weight_bytes": visited_before_lm
        + lm_telemetry["final_norm_weight_bytes"] + lm_telemetry["visited_lm_head_bytes"],
        "peak_selected_expert_count": max(s["selected_expert_count"] for s in layer_stats),
        "peak_expert_pair_bytes": max(s["peak_expert_pair_bytes"] for s in layer_stats),
        "selected_expert_counts": [s["selected_expert_count"] for s in layer_stats],
        "visited_lm_head_bytes": lm_telemetry["visited_lm_head_bytes"],
        "peak_lm_head_chunk_bytes": lm_telemetry["peak_lm_head_chunk_bytes"],
    }
    return int(next_id), state_list, add_budget_telemetry(telemetry, budget)


def _decode_kv(
    *,
    input_token_id: int,
    position: int,
    state_list: list[Any],
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    config: dict[str, Any],
    lm_head_chunk_rows: int,
    budget: WeightBudget | None = None,
    sampler: Sampler | None = None,
) -> tuple[int, list[Any], dict[str, Any]]:
    """Decode one token against existing state (conv state + KV cache)."""
    scalars = scalar_config(config)
    inputs = make_decode_inputs(input_token_id, position=position)
    hidden, embed_bytes = embed_input_ids(
        input_ids=inputs["input_ids"],
        model_file=model_file, header=header, payload_start=payload_start,
        scalars=scalars, budget=budget,
    )

    layer_types = config.get("layer_types", [])
    new_state_list: list[Any] = [None] * int(config.get("num_hidden_layers", 24))
    layer_stats = []
    visited_before_lm = embed_bytes
    for layer_idx in range(int(config.get("num_hidden_layers", 24))):
        # Extract state for this layer
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "conv"
        is_attention = layer_type == "full_attention"
        kv_pair = state_list[layer_idx] if is_attention else None
        conv_state = state_list[layer_idx] if not is_attention else None

        hidden, layer_state, stats = run_layer_with_optional_state(
            layer_idx=layer_idx, hidden=hidden,
            kv_pair=kv_pair, conv_state=conv_state,
            model_file=model_file, header=header, payload_start=payload_start,
            inputs=inputs, config=config, scalars=scalars,
            return_state=True, budget=budget,
        )
        new_kv_pair, new_conv_state = layer_state
        new_state_list[layer_idx] = new_kv_pair if new_kv_pair is not None else new_conv_state
        layer_stats.append(stats)
        visited_before_lm += stats["loaded_nonexpert_bytes"] + stats["cumulative_expert_weight_bytes"]

    if sampler is not None and not sampler.is_greedy():
        _fh, logits, lm_telemetry = _final_hidden_to_logits(
            hidden=hidden, model_file=model_file, header=header, payload_start=payload_start,
            scalars=scalars, chunk_rows=lm_head_chunk_rows, budget=budget,
        )
        next_id = sampler.sample(logits)
    else:
        _fh, next_id, lm_telemetry = _final_hidden_to_greedy(
            hidden=hidden, model_file=model_file, header=header, payload_start=payload_start,
            scalars=scalars, chunk_rows=lm_head_chunk_rows, budget=budget,
        )
    telemetry = {
        "phase": "decode",
        "position": position,
        "visited_before_lm_head_bytes": visited_before_lm,
        "cumulative_weight_bytes": visited_before_lm
        + lm_telemetry["final_norm_weight_bytes"] + lm_telemetry["visited_lm_head_bytes"],
        "peak_selected_expert_count": max(s["selected_expert_count"] for s in layer_stats),
        "peak_expert_pair_bytes": max(s["peak_expert_pair_bytes"] for s in layer_stats),
        "selected_expert_counts": [s["selected_expert_count"] for s in layer_stats],
        "visited_lm_head_bytes": lm_telemetry["visited_lm_head_bytes"],
        "peak_lm_head_chunk_bytes": lm_telemetry["peak_lm_head_chunk_bytes"],
        "kv_length": position + 1,
    }
    return int(next_id), new_state_list, add_budget_telemetry(telemetry, budget)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class Lfm2MoeBackend:
    """Backend for Liquid LFM2.5 MoE models.

    Owns all LFM2 execution: hybrid conv/attention layers, MoE routing,
    and the prefill/decode loop with mixed state (conv state + KV cache).
    """

    name = "lfm2"

    def supports_config(self, config: dict[str, Any]) -> bool:
        model_type = str(config.get("model_type", "")).lower()
        architectures = [str(x).lower() for x in config.get("architectures", [])]
        return (
            model_type == "lfm2_moe"
            or model_type == "lfm2"
            or any("lfm2" in arch for arch in architectures)
        )

    def next_token_greedy(
        self,
        *,
        token_ids: list[int],
        model_file: Path,
        header: dict[str, Any],
        payload_start: int,
        config: dict[str, Any],
        lm_head_chunk_rows: int,
        budget: WeightBudget | None = None,
        sampler: Sampler | None = None,
    ) -> tuple[int, dict[str, Any]]:
        return _next_token(
            token_ids=token_ids, model_file=model_file, header=header,
            payload_start=payload_start, config=config,
            lm_head_chunk_rows=lm_head_chunk_rows, budget=budget, sampler=sampler,
        )

    def prefill_kv_greedy(
        self,
        *,
        token_ids: list[int],
        model_file: Path,
        header: dict[str, Any],
        payload_start: int,
        config: dict[str, Any],
        lm_head_chunk_rows: int,
        budget: WeightBudget | None = None,
        sampler: Sampler | None = None,
    ) -> tuple[int, list[Any], dict[str, Any]]:
        return _prefill_kv(
            token_ids=token_ids, model_file=model_file, header=header,
            payload_start=payload_start, config=config,
            lm_head_chunk_rows=lm_head_chunk_rows, budget=budget, sampler=sampler,
        )

    def decode_kv_greedy(
        self,
        *,
        input_token_id: int,
        position: int,
        kv_cache: list[Any],
        model_file: Path,
        header: dict[str, Any],
        payload_start: int,
        config: dict[str, Any],
        lm_head_chunk_rows: int,
        budget: WeightBudget | None = None,
        sampler: Sampler | None = None,
    ) -> tuple[int, list[Any], dict[str, Any]]:
        return _decode_kv(
            input_token_id=input_token_id, position=position, state_list=kv_cache,
            model_file=model_file, header=header, payload_start=payload_start,
            config=config, lm_head_chunk_rows=lm_head_chunk_rows,
            budget=budget, sampler=sampler,
        )