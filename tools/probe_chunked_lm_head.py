#!/usr/bin/env python3
"""Probe chunked tied-LM-head logits and streaming greedy argmax.

This reuses the proven full layer loop, then computes last-token logits by
visiting `model.embed_tokens.weight` in vocab-row chunks instead of loading the
full tied LM head.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glacial.granite import FINAL_NORM_TENSOR, run_layer
from glacial.logits import chunked_last_logits
from probe_embedding import (  # same-directory import when run as tools/probe_*.py
    BF16_BYTES,
    EMBED_TENSOR,
    compare_summaries,
    first,
    flatten_ids,
    latest_trace,
    load_trace,
    read_embedding_rows,
    read_safetensors_header,
    require_torch,
    resolve_model_file,
    tensor_summary,
)
from probe_layer0_input_norm import granite_rmsnorm, read_full_bf16_tensor


def compute_final_hidden(*, model_file: Path, header: dict[str, Any], payload_start: int, inputs: dict[str, Any], config: dict[str, Any]):
    scalars = {
        "embedding_multiplier": float(config.get("embedding_multiplier", 12.0)),
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(config.get("rope_theta", 1500000.0)),
        "attention_multiplier": float(config.get("attention_multiplier", 0.015625)),
        "residual_multiplier": float(config.get("residual_multiplier", 0.22)),
        "logits_scaling": float(config.get("logits_scaling", 6.0)),
    }

    input_ids, input_shape = flatten_ids(inputs["input_ids"])
    rows, hidden_size = read_embedding_rows(
        model_file,
        input_ids,
        tensor_meta=header[EMBED_TENSOR],
        payload_start=payload_start,
    )
    hidden = rows.view(input_shape[0], input_shape[1], hidden_size) * scalars["embedding_multiplier"]
    hidden = hidden.to(rows.dtype)

    cumulative_weight_bytes = len(input_ids) * hidden_size * BF16_BYTES
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
        )
        layer_stats.append(stats)
        cumulative_weight_bytes += stats["loaded_nonexpert_bytes"] + stats["cumulative_expert_weight_bytes"]

    final_norm_weight = read_full_bf16_tensor(
        model_file,
        tensor_name=FINAL_NORM_TENSOR,
        tensor_meta=header[FINAL_NORM_TENSOR],
        payload_start=payload_start,
    )
    cumulative_weight_bytes += final_norm_weight.numel() * BF16_BYTES
    final_hidden = granite_rmsnorm(hidden, final_norm_weight, eps=scalars["rms_norm_eps"])
    return final_hidden, scalars, cumulative_weight_bytes, layer_stats



def compare_top_tokens(actual_logits, final_event: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    import torch

    logits_fp32 = actual_logits.float()
    values, ids = logits_fp32.topk(top_k)
    actual_ids = [int(x) for x in ids.detach().cpu().tolist()]
    actual_values = [float(x) for x in values.detach().cpu().tolist()]
    actual_greedy = int(torch.argmax(logits_fp32).item())
    expected_ids = final_event.get("top_token_ids")
    expected_values = final_event.get("top_token_values_fp32")
    return {
        "actual_token_ids": actual_ids,
        "expected_token_ids": expected_ids,
        "token_ids_pass": actual_ids == expected_ids,
        "actual_values_fp32": actual_values,
        "expected_values_fp32": expected_values,
        "greedy_token_id": actual_greedy,
        "expected_greedy_token_id": final_event.get("greedy_token_id"),
        "greedy_pass": actual_greedy == final_event.get("greedy_token_id"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=None, help="HF golden trace JSONL. Defaults to latest traces/*.jsonl")
    parser.add_argument("--model-file", type=Path, default=None, help="Local model.safetensors path")
    parser.add_argument("--model-id", default=None, help="HF model id fallback. Defaults to trace metadata.")
    parser.add_argument("--revision", default=None, help="HF revision fallback. Defaults to trace metadata.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_torch()

    trace_path = args.trace or latest_trace()
    records = load_trace(trace_path)
    metadata = first(records, "trace_metadata")
    inputs = first(records, "inputs")
    final_event = first(records, "final_logits")

    model_file = resolve_model_file(args, metadata)
    config = metadata.get("config") or {}
    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len

    final_hidden, scalars, bytes_before_lm_head, layer_stats = compute_final_hidden(
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        inputs=inputs,
        config=config,
    )
    logits, telemetry = chunked_last_logits(
        final_hidden=final_hidden,
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        chunk_rows=args.chunk_rows,
        logits_scaling=scalars["logits_scaling"],
    )

    logits_summary = tensor_summary(logits)
    logits_comparison = compare_summaries(logits_summary, final_event["next_token_logits"], atol=args.atol)
    top_tokens = compare_top_tokens(logits, final_event, top_k=int(final_event.get("top_k", 20)))
    streaming_greedy_pass = telemetry["streaming_greedy_token_id"] == final_event.get("greedy_token_id")
    all_pass = bool(
        logits_comparison["pass"]
        and logits_comparison["sha256_pass"]
        and top_tokens["token_ids_pass"]
        and top_tokens["greedy_pass"]
        and streaming_greedy_pass
    )

    result = {
        "trace": str(trace_path),
        "model_file": str(model_file),
        "input_ids": inputs["input_ids"],
        "position_ids": inputs["position_ids"],
        "bytes_before_lm_head": bytes_before_lm_head,
        "chunked_lm_head": telemetry,
        "logits_comparison": logits_comparison,
        "top_tokens": top_tokens,
        "streaming_greedy_pass": streaming_greedy_pass,
        "all_pass": all_pass,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("# Chunked LM-head probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"input ids: `{inputs['input_ids']}`")
        print(f"position ids: `{inputs['position_ids']}`")
        print(f"chunk rows: `{args.chunk_rows}`")
        print(f"bytes before LM head: `{bytes_before_lm_head}`")
        print(f"visited LM-head bytes: `{telemetry['visited_lm_head_bytes']}`")
        print(f"peak LM-head chunk bytes: `{telemetry['peak_lm_head_chunk_bytes']}`")
        print(f"streaming greedy token id: `{telemetry['streaming_greedy_token_id']}` expected `{final_event.get('greedy_token_id')}`")
        print(f"streaming greedy pass: `{streaming_greedy_pass}`")
        print(f"next-token logits pass: `{logits_comparison['pass'] and logits_comparison['sha256_pass']}`")
        print(f"top token ids pass: `{top_tokens['token_ids_pass']}`")
        print(f"greedy pass: `{top_tokens['greedy_pass']}`")
        print(f"all pass: `{all_pass}`")
        print("\n## Next-token logits\n")
        print(f"shape: `{logits_summary['shape']}`")
        print(f"dtype: `{logits_summary['dtype']}`")
        print(f"actual sha256: `{logits_comparison['actual_sha256']}`")
        print(f"expected sha256: `{logits_comparison['expected_sha256']}`")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
