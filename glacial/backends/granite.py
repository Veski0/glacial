"""Granite MoE backend for Glacial.

This backend owns ALL Granite-specific execution logic:

* layer math (RMSNorm, RoPE, attention, MoE routing/experts) — in ``glacial.granite``
* final norm + chunked tied-LM-head — defined below
* the prefill / decode / no-KV decode loop — defined below

The outer runtime (CLI, server, tests) talks only to the
``CausalLMBackend`` protocol and never imports Granite internals directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glacial.generate import add_budget_telemetry, embed_input_ids, make_decode_inputs, make_inputs
from glacial.granite import (
    FINAL_NORM_TENSOR,
    granite_rmsnorm,
    run_layer,
    run_layer_with_optional_kv,
    scalar_config,
)
from glacial.logits import chunked_last_argmax, chunked_last_logits
from glacial.sampler import Sampler
from glacial.weights import BF16_BYTES, SafetensorsWeights, WeightBudget


# ---------------------------------------------------------------------------
# Final norm + chunked tied-LM-head (Granite-specific)
# ---------------------------------------------------------------------------

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
        final_hidden = granite_rmsnorm(hidden, final_norm_weight, eps=scalars["rms_norm_eps"])
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
        final_hidden = granite_rmsnorm(hidden, final_norm_weight, eps=scalars["rms_norm_eps"])
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


# ---------------------------------------------------------------------------
# Decode loop (Granite-specific)
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
    """Slow no-KV path: recompute full prompt for each token.

    If ``sampler`` is provided and is non-greedy, the full logit vector is
    materialized (via ``_final_hidden_to_logits``) and the sampler selects the
    token.  Otherwise the streaming argmax path is used (no full logits).
    """
    scalars = scalar_config(config)
    inputs = make_inputs(token_ids)
    hidden, embed_bytes = embed_input_ids(
        input_ids=inputs["input_ids"],
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        scalars=scalars,
        budget=budget,
    )

    cumulative_weight_bytes = embed_bytes
    layer_stats = []
    for layer_idx in range(int(config.get("num_hidden_layers", 24))):
        hidden, stats = run_layer(
            layer_idx=layer_idx,
            hidden=hidden,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            inputs=inputs,
            config=config,
            scalars=scalars,
            budget=budget,
        )
        layer_stats.append(stats)
        cumulative_weight_bytes += stats["loaded_nonexpert_bytes"] + stats["cumulative_expert_weight_bytes"]

    if sampler is not None and not sampler.is_greedy():
        _final_hidden, logits, lm_telemetry = _final_hidden_to_logits(
            hidden=hidden,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            scalars=scalars,
            chunk_rows=lm_head_chunk_rows,
            budget=budget,
        )
        next_id = sampler.sample(logits)
    else:
        _final_hidden, next_id, lm_telemetry = _final_hidden_to_greedy(
            hidden=hidden,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            scalars=scalars,
            chunk_rows=lm_head_chunk_rows,
            budget=budget,
        )
    cumulative_weight_bytes += lm_telemetry["final_norm_weight_bytes"] + lm_telemetry["visited_lm_head_bytes"]
    telemetry = {
        "phase": "full",
        "cumulative_weight_bytes": cumulative_weight_bytes,
        "peak_selected_expert_count": max(stat["selected_expert_count"] for stat in layer_stats),
        "peak_expert_pair_bytes": max(stat["peak_expert_pair_bytes"] for stat in layer_stats),
        "selected_expert_counts": [stat["selected_expert_count"] for stat in layer_stats],
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
) -> tuple[int, list[tuple[Any, Any]], dict[str, Any]]:
    """Prefill prompt KV and return the first generated token."""
    scalars = scalar_config(config)
    inputs = make_inputs(token_ids)
    hidden, embed_bytes = embed_input_ids(
        input_ids=inputs["input_ids"],
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        scalars=scalars,
        budget=budget,
    )

    kv_cache = []
    layer_stats = []
    visited_before_lm = embed_bytes
    for layer_idx in range(int(config.get("num_hidden_layers", 24))):
        hidden, kv_pair, stats = run_layer_with_optional_kv(
            layer_idx=layer_idx,
            hidden=hidden,
            kv_pair=None,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            inputs=inputs,
            config=config,
            scalars=scalars,
            budget=budget,
        )
        kv_cache.append(kv_pair)
        layer_stats.append(stats)
        visited_before_lm += stats["loaded_nonexpert_bytes"] + stats["cumulative_expert_weight_bytes"]

    if sampler is not None and not sampler.is_greedy():
        _final_hidden, logits, lm_telemetry = _final_hidden_to_logits(
            hidden=hidden,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            scalars=scalars,
            chunk_rows=lm_head_chunk_rows,
            budget=budget,
        )
        next_id = sampler.sample(logits)
    else:
        _final_hidden, next_id, lm_telemetry = _final_hidden_to_greedy(
            hidden=hidden,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            scalars=scalars,
            chunk_rows=lm_head_chunk_rows,
            budget=budget,
        )
    telemetry = {
        "phase": "prefill",
        "visited_before_lm_head_bytes": visited_before_lm,
        "cumulative_weight_bytes": visited_before_lm
        + lm_telemetry["final_norm_weight_bytes"]
        + lm_telemetry["visited_lm_head_bytes"],
        "peak_selected_expert_count": max(stat["selected_expert_count"] for stat in layer_stats),
        "peak_expert_pair_bytes": max(stat["peak_expert_pair_bytes"] for stat in layer_stats),
        "selected_expert_counts": [stat["selected_expert_count"] for stat in layer_stats],
        "visited_lm_head_bytes": lm_telemetry["visited_lm_head_bytes"],
        "peak_lm_head_chunk_bytes": lm_telemetry["peak_lm_head_chunk_bytes"],
    }
    return int(next_id), kv_cache, add_budget_telemetry(telemetry, budget)


def _decode_kv(
    *,
    input_token_id: int,
    position: int,
    kv_cache: list[tuple[Any, Any]],
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    config: dict[str, Any],
    lm_head_chunk_rows: int,
    budget: WeightBudget | None = None,
    sampler: Sampler | None = None,
) -> tuple[int, list[tuple[Any, Any]], dict[str, Any]]:
    """Decode one token against existing KV and return the next token."""
    scalars = scalar_config(config)
    inputs = make_decode_inputs(input_token_id, position=position)
    hidden, embed_bytes = embed_input_ids(
        input_ids=inputs["input_ids"],
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        scalars=scalars,
        budget=budget,
    )

    new_kv_cache = []
    layer_stats = []
    visited_before_lm = embed_bytes
    for layer_idx in range(int(config.get("num_hidden_layers", 24))):
        hidden, kv_pair, stats = run_layer_with_optional_kv(
            layer_idx=layer_idx,
            hidden=hidden,
            kv_pair=kv_cache[layer_idx],
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            inputs=inputs,
            config=config,
            scalars=scalars,
            budget=budget,
        )
        new_kv_cache.append(kv_pair)
        layer_stats.append(stats)
        visited_before_lm += stats["loaded_nonexpert_bytes"] + stats["cumulative_expert_weight_bytes"]

    if sampler is not None and not sampler.is_greedy():
        _final_hidden, logits, lm_telemetry = _final_hidden_to_logits(
            hidden=hidden,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            scalars=scalars,
            chunk_rows=lm_head_chunk_rows,
            budget=budget,
        )
        next_id = sampler.sample(logits)
    else:
        _final_hidden, next_id, lm_telemetry = _final_hidden_to_greedy(
            hidden=hidden,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            scalars=scalars,
            chunk_rows=lm_head_chunk_rows,
            budget=budget,
        )
    telemetry = {
        "phase": "decode",
        "position": position,
        "visited_before_lm_head_bytes": visited_before_lm,
        "cumulative_weight_bytes": visited_before_lm
        + lm_telemetry["final_norm_weight_bytes"]
        + lm_telemetry["visited_lm_head_bytes"],
        "peak_selected_expert_count": max(stat["selected_expert_count"] for stat in layer_stats),
        "peak_expert_pair_bytes": max(stat["peak_expert_pair_bytes"] for stat in layer_stats),
        "selected_expert_counts": [stat["selected_expert_count"] for stat in layer_stats],
        "visited_lm_head_bytes": lm_telemetry["visited_lm_head_bytes"],
        "peak_lm_head_chunk_bytes": lm_telemetry["peak_lm_head_chunk_bytes"],
        "kv_length": position + 1,
    }
    return int(next_id), new_kv_cache, add_budget_telemetry(telemetry, budget)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class GraniteMoeBackend:
    """Backend adapter for IBM Granite MoE causal language models.

    Owns all Granite-specific execution: layer math, MoE routing, final norm,
    chunked tied-LM-head, and the prefill/decode loop.  The outer runtime
    talks only to the ``CausalLMBackend`` protocol methods below.
    """

    name = "granite"

    def supports_config(self, config: dict[str, Any]) -> bool:
        architectures = [str(x).lower() for x in config.get("architectures", [])]
        model_type = str(config.get("model_type", "")).lower()
        return model_type == "granitemoe" or any("granitemoe" in arch for arch in architectures)

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
            token_ids=token_ids,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            config=config,
            lm_head_chunk_rows=lm_head_chunk_rows,
            budget=budget,
            sampler=sampler,
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
    ) -> tuple[int, list[tuple[Any, Any]], dict[str, Any]]:
        return _prefill_kv(
            token_ids=token_ids,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            config=config,
            lm_head_chunk_rows=lm_head_chunk_rows,
            budget=budget,
            sampler=sampler,
        )

    def decode_kv_greedy(
        self,
        *,
        input_token_id: int,
        position: int,
        kv_cache: list[tuple[Any, Any]],
        model_file: Path,
        header: dict[str, Any],
        payload_start: int,
        config: dict[str, Any],
        lm_head_chunk_rows: int,
        budget: WeightBudget | None = None,
        sampler: Sampler | None = None,
    ) -> tuple[int, list[tuple[Any, Any]], dict[str, Any]]:
        return _decode_kv(
            input_token_id=input_token_id,
            position=position,
            kv_cache=kv_cache,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            config=config,
            lm_head_chunk_rows=lm_head_chunk_rows,
            budget=budget,
            sampler=sampler,
        )