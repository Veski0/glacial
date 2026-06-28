"""Generation helpers for the Glacial runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glacial.granite import run_layer, run_layer_with_optional_kv, scalar_config
from glacial.logits import final_hidden_to_greedy
from glacial.weights import BF16_BYTES, EMBED_TENSOR, SafetensorsWeights, WeightBudget


def flatten_ids(nested: Any) -> tuple[list[int], list[int]]:
    """Flatten rank-2 token IDs and return original [batch, seq] shape."""

    if not isinstance(nested, list) or not nested or not isinstance(nested[0], list):
        raise SystemExit("Expected input_ids to be a non-empty rank-2 list")
    batch = len(nested)
    seq = len(nested[0])
    out: list[int] = []
    for row in nested:
        if len(row) != seq:
            raise SystemExit("Ragged input_ids are not supported")
        out.extend(int(x) for x in row)
    return out, [batch, seq]


def make_inputs(token_ids: list[int]) -> dict[str, Any]:
    seq_len = len(token_ids)
    return {
        "input_ids": [token_ids],
        "attention_mask": [[1] * seq_len],
        "position_ids": [list(range(seq_len))],
        "cache_position": list(range(seq_len)),
    }


def make_decode_inputs(token_id: int, *, position: int) -> dict[str, Any]:
    return {
        "input_ids": [[token_id]],
        "attention_mask": [[1] * (position + 1)],
        "position_ids": [[position]],
        "cache_position": [position],
    }


def add_budget_telemetry(telemetry: dict[str, Any], budget: WeightBudget | None) -> dict[str, Any]:
    if budget is not None:
        telemetry.update(
            {
                "weight_budget_current_bytes": budget.current_resident_bytes,
                "weight_budget_peak_bytes": budget.peak_resident_bytes,
                "weight_budget_total_visited_bytes": budget.total_visited_bytes,
                "weight_budget_violation_count": len(budget.violations),
            }
        )
    return telemetry


def embed_input_ids(
    *,
    input_ids: list[list[int]],
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    scalars: dict[str, float],
    budget: WeightBudget | None = None,
):
    flat_ids, input_shape = flatten_ids(input_ids)
    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start, budget=budget)
    with provider.embedding_rows(flat_ids) as (rows, hidden_size):
        hidden = rows.view(input_shape[0], input_shape[1], hidden_size) * scalars["embedding_multiplier"]
        hidden = hidden.to(rows.dtype)
    return hidden, len(flat_ids) * hidden_size * BF16_BYTES


def next_token_greedy(
    *,
    token_ids: list[int],
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    config: dict[str, Any],
    lm_head_chunk_rows: int,
    budget: WeightBudget | None = None,
) -> tuple[int, dict[str, Any]]:
    """Slow no-KV greedy path: recompute full prompt for each token."""

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

    _final_hidden, next_id, lm_telemetry = final_hidden_to_greedy(
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


def prefill_kv_greedy(
    *,
    token_ids: list[int],
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    config: dict[str, Any],
    lm_head_chunk_rows: int,
    budget: WeightBudget | None = None,
) -> tuple[int, list[tuple[Any, Any]], dict[str, Any]]:
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

    _final_hidden, next_id, lm_telemetry = final_hidden_to_greedy(
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


def decode_kv_greedy(
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
) -> tuple[int, list[tuple[Any, Any]], dict[str, Any]]:
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

    _final_hidden, next_id, lm_telemetry = final_hidden_to_greedy(
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
