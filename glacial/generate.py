"""Shared generation utilities for Glacial backends.

These helpers are architecture-agnostic and may be reused by any backend
that follows the standard embed → layers → LM-head decode pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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