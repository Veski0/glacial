"""Shared logits and greedy selection helpers for Glacial backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glacial.weights import BF16_BYTES, EMBED_TENSOR, SafetensorsWeights, WeightBudget


def greedy_from_logits(logits) -> int:
    """HF-compatible greedy token selection.

    Important: greedy is ``argmax``, not ``topk(...)[0]``. In ties, PyTorch
    argmax returns the first/max-lowest index, which is what the HF traces use.
    """

    import torch

    return int(torch.argmax(logits.float()).item())


def chunked_last_logits(
    *,
    final_hidden,
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    chunk_rows: int,
    logits_scaling: float,
    budget: WeightBudget | None = None,
):
    """Compute full last-token logits while visiting tied LM-head rows in chunks."""

    import torch
    import torch.nn.functional as F

    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start, budget=budget)
    meta = header[EMBED_TENSOR]
    vocab_size = int(meta["shape"][0])
    last_hidden = final_hidden[:, -1:, :]
    chunks = []
    best_value = None
    best_id = None
    peak_chunk_bytes = 0
    visited_lm_head_bytes = 0

    for row_start in range(0, vocab_size, chunk_rows):
        row_count = min(chunk_rows, vocab_size - row_start)
        with provider.lm_head_chunk(row_start=row_start, row_count=row_count) as weight:
            peak_chunk_bytes = max(peak_chunk_bytes, weight.numel() * BF16_BYTES)
            visited_lm_head_bytes += weight.numel() * BF16_BYTES
            logits_chunk = F.linear(last_hidden, weight)[0, 0] / logits_scaling
            chunks.append(logits_chunk)

            chunk_values = logits_chunk.float()
            chunk_best_value, chunk_best_offset = torch.max(chunk_values, dim=0)
            chunk_best_id = row_start + int(chunk_best_offset.item())
            chunk_best_float = float(chunk_best_value.item())
            if best_value is None or chunk_best_float > best_value or (chunk_best_float == best_value and chunk_best_id < best_id):
                best_value = chunk_best_float
                best_id = chunk_best_id

    logits = torch.cat(chunks, dim=0)
    return logits, {
        "streaming_greedy_token_id": best_id,
        "streaming_greedy_value_fp32": best_value,
        "peak_lm_head_chunk_bytes": peak_chunk_bytes,
        "visited_lm_head_bytes": visited_lm_head_bytes,
        "chunk_rows": chunk_rows,
    }


def chunked_last_argmax(
    *,
    final_hidden,
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    chunk_rows: int,
    logits_scaling: float,
    budget: WeightBudget | None = None,
):
    """Compute greedy argmax for last-token logits without storing all logits."""

    import torch
    import torch.nn.functional as F

    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start, budget=budget)
    meta = header[EMBED_TENSOR]
    vocab_size = int(meta["shape"][0])
    last_hidden = final_hidden[:, -1:, :]
    best_value = None
    best_id = None
    peak_chunk_bytes = 0
    visited_lm_head_bytes = 0

    for row_start in range(0, vocab_size, chunk_rows):
        row_count = min(chunk_rows, vocab_size - row_start)
        with provider.lm_head_chunk(row_start=row_start, row_count=row_count) as weight:
            chunk_bytes = weight.numel() * BF16_BYTES
            peak_chunk_bytes = max(peak_chunk_bytes, chunk_bytes)
            visited_lm_head_bytes += chunk_bytes
            logits_chunk = F.linear(last_hidden, weight)[0, 0] / logits_scaling
            chunk_best_value, chunk_best_offset = torch.max(logits_chunk.float(), dim=0)
            chunk_best_id = row_start + int(chunk_best_offset.item())
            chunk_best_float = float(chunk_best_value.item())
            if best_value is None or chunk_best_float > best_value or (chunk_best_float == best_value and chunk_best_id < best_id):
                best_value = chunk_best_float
                best_id = chunk_best_id

    return int(best_id), {
        "streaming_greedy_token_id": best_id,
        "streaming_greedy_value_fp32": best_value,
        "peak_lm_head_chunk_bytes": peak_chunk_bytes,
        "visited_lm_head_bytes": visited_lm_head_bytes,
        "chunk_rows": chunk_rows,
    }
