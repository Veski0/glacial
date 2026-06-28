#!/usr/bin/env python3
"""Probe Glacial prefill KV cache + one cached decode step.

This compares Glacial cached-decode logits against an HF cached-decode trace.
It does not instantiate the HF model.
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

from glacial.generate import embed_input_ids
from glacial.granite import FINAL_NORM_TENSOR, required_layer_tensors, run_layer_with_optional_kv, scalar_config
from glacial.logits import final_hidden_to_logits
from probe_embedding import (
    EMBED_TENSOR,
    compare_summaries,
    load_trace,
    read_safetensors_header,
    require_torch,
    resolve_model_file,
    tensor_summary,
)


def first(records: list[dict[str, Any]], event: str) -> dict[str, Any]:
    for record in records:
        if record.get("event") == event:
            return record
    raise SystemExit(f"Trace is missing event {event!r}")


def latest_kv_trace() -> Path:
    traces = sorted(Path("traces").glob("*__kv_decode__*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not traces:
        raise SystemExit("No traces/*__kv_decode__*.jsonl files found. Pass --trace explicitly.")
    return traces[-1]



def compare_top_tokens(actual_logits, expected_event: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    import torch

    logits_fp32 = actual_logits.float()
    values, ids = logits_fp32.topk(top_k)
    actual_ids = [int(x) for x in ids.detach().cpu().tolist()]
    actual_values = [float(x) for x in values.detach().cpu().tolist()]
    actual_greedy = int(torch.argmax(logits_fp32).item())
    expected_ids = expected_event.get("top_token_ids")
    return {
        "actual_token_ids": actual_ids,
        "expected_token_ids": expected_ids,
        "token_ids_pass": actual_ids == expected_ids,
        "actual_values_fp32": actual_values,
        "expected_values_fp32": expected_event.get("top_token_values_fp32"),
        "greedy_token_id": actual_greedy,
        "expected_greedy_token_id": expected_event.get("greedy_token_id"),
        "greedy_pass": actual_greedy == expected_event.get("greedy_token_id"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=None, help="HF KV decode trace JSONL. Defaults to latest traces/*__kv_decode__*.jsonl")
    parser.add_argument("--model-file", type=Path, default=None, help="Local model.safetensors path")
    parser.add_argument("--model-id", default=None, help="HF model id fallback. Defaults to trace metadata.")
    parser.add_argument("--revision", default=None, help="HF revision fallback. Defaults to trace metadata.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_torch()

    trace_path = args.trace or latest_kv_trace()
    records = load_trace(trace_path)
    metadata = first(records, "kv_trace_metadata")
    prefill_inputs = first(records, "prefill_inputs")
    prefill_logits_expected = first(records, "prefill_logits")
    prefill_cache_expected = first(records, "prefill_cache")
    decode_inputs = first(records, "decode_inputs")
    decode_logits_expected = first(records, "decode_logits")
    decode_cache_expected = first(records, "decode_cache")

    # Reuse resolve_model_file's expected metadata names.
    model_meta = {"model_id": metadata.get("model_id"), "revision": metadata.get("revision")}
    model_file = resolve_model_file(args, model_meta)
    config = metadata.get("config") or {}
    scalars = scalar_config(config)
    num_layers = int(config.get("num_hidden_layers", 24))

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    required = [EMBED_TENSOR, FINAL_NORM_TENSOR]
    for layer_idx in range(num_layers):
        required.extend(required_layer_tensors(layer_idx))
    missing = [tensor for tensor in required if tensor not in header]
    if missing:
        raise SystemExit(f"{model_file}: missing tensors: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    hidden, embed_bytes = embed_input_ids(
        input_ids=prefill_inputs["input_ids"],
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        scalars=scalars,
    )
    kv_cache = []
    prefill_stats = []
    prefill_visited_bytes = embed_bytes
    for layer_idx in range(num_layers):
        hidden, kv_pair, stats = run_layer_with_optional_kv(
            layer_idx=layer_idx,
            hidden=hidden,
            kv_pair=None,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            inputs=prefill_inputs,
            config=config,
            scalars=scalars,
        )
        kv_cache.append(kv_pair)
        prefill_stats.append(stats)
        prefill_visited_bytes += stats["loaded_nonexpert_bytes"] + stats["cumulative_expert_weight_bytes"]

    _prefill_final_hidden, prefill_logits, prefill_lm_telemetry = final_hidden_to_logits(
        hidden=hidden,
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        scalars=scalars,
        chunk_rows=args.chunk_rows,
    )
    prefill_logits_summary = tensor_summary(prefill_logits)
    prefill_logits_comparison = compare_summaries(prefill_logits_summary, prefill_logits_expected["next_token_logits"], atol=args.atol)
    prefill_top_tokens = compare_top_tokens(prefill_logits, prefill_logits_expected, top_k=int(prefill_logits_expected.get("top_k", 20)))

    # Decode one token using the prefilled KV cache.
    decode_hidden, decode_embed_bytes = embed_input_ids(
        input_ids=decode_inputs["input_ids"],
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        scalars=scalars,
    )
    decode_stats = []
    decode_visited_bytes = decode_embed_bytes
    new_kv_cache = []
    for layer_idx in range(num_layers):
        decode_hidden, kv_pair, stats = run_layer_with_optional_kv(
            layer_idx=layer_idx,
            hidden=decode_hidden,
            kv_pair=kv_cache[layer_idx],
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            inputs=decode_inputs,
            config=config,
            scalars=scalars,
        )
        new_kv_cache.append(kv_pair)
        decode_stats.append(stats)
        decode_visited_bytes += stats["loaded_nonexpert_bytes"] + stats["cumulative_expert_weight_bytes"]

    _decode_final_hidden, decode_logits, decode_lm_telemetry = final_hidden_to_logits(
        hidden=decode_hidden,
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        scalars=scalars,
        chunk_rows=args.chunk_rows,
    )
    decode_logits_summary = tensor_summary(decode_logits)
    decode_logits_comparison = compare_summaries(decode_logits_summary, decode_logits_expected["next_token_logits"], atol=args.atol)
    decode_top_tokens = compare_top_tokens(decode_logits, decode_logits_expected, top_k=int(decode_logits_expected.get("top_k", 20)))

    prefill_cache_shapes_pass = [stats["kv_key_shape"] for stats in prefill_stats] == prefill_cache_expected.get("key_shapes")
    decode_cache_shapes_pass = [stats["kv_key_shape"] for stats in decode_stats] == decode_cache_expected.get("key_shapes")

    all_pass = bool(
        prefill_logits_comparison["pass"]
        and prefill_logits_comparison["sha256_pass"]
        and prefill_top_tokens["greedy_pass"]
        and decode_logits_comparison["pass"]
        and decode_logits_comparison["sha256_pass"]
        and decode_top_tokens["greedy_pass"]
        and prefill_cache_shapes_pass
        and decode_cache_shapes_pass
    )

    result = {
        "trace": str(trace_path),
        "model_file": str(model_file),
        "prefill_input_ids": prefill_inputs["input_ids"],
        "decode_input_ids": decode_inputs["input_ids"],
        "prefill_logits_comparison": prefill_logits_comparison,
        "decode_logits_comparison": decode_logits_comparison,
        "prefill_top_tokens": prefill_top_tokens,
        "decode_top_tokens": decode_top_tokens,
        "prefill_cache_shapes_pass": prefill_cache_shapes_pass,
        "decode_cache_shapes_pass": decode_cache_shapes_pass,
        "prefill_visited_bytes_before_lm_head": prefill_visited_bytes,
        "decode_visited_bytes_before_lm_head": decode_visited_bytes,
        "prefill_lm_telemetry": prefill_lm_telemetry,
        "decode_lm_telemetry": decode_lm_telemetry,
        "prefill_peak_selected_experts": max(s["selected_expert_count"] for s in prefill_stats),
        "decode_peak_selected_experts": max(s["selected_expert_count"] for s in decode_stats),
        "all_pass": all_pass,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("# KV decode one-step probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"prefill input ids: `{prefill_inputs['input_ids']}`")
        print(f"decode input ids: `{decode_inputs['input_ids']}`")
        print(f"prefill greedy: `{prefill_top_tokens['greedy_token_id']}` expected `{prefill_top_tokens['expected_greedy_token_id']}`")
        print(f"decode greedy: `{decode_top_tokens['greedy_token_id']}` expected `{decode_top_tokens['expected_greedy_token_id']}`")
        print(f"prefill logits pass: `{prefill_logits_comparison['pass'] and prefill_logits_comparison['sha256_pass']}`")
        print(f"decode logits pass: `{decode_logits_comparison['pass'] and decode_logits_comparison['sha256_pass']}`")
        print(f"prefill cache shapes pass: `{prefill_cache_shapes_pass}`")
        print(f"decode cache shapes pass: `{decode_cache_shapes_pass}`")
        print(f"prefill visited bytes before LM head: `{prefill_visited_bytes}`")
        print(f"decode visited bytes before LM head: `{decode_visited_bytes}`")
        print(f"prefill peak selected experts: `{result['prefill_peak_selected_experts']}`")
        print(f"decode peak selected experts: `{result['decode_peak_selected_experts']}`")
        print(f"decode peak LM-head chunk bytes: `{decode_lm_telemetry['peak_lm_head_chunk_bytes']}`")
        print(f"all pass: `{all_pass}`")
        print("\n## Decode next-token logits\n")
        print(f"shape: `{decode_logits_summary['shape']}`")
        print(f"dtype: `{decode_logits_summary['dtype']}`")
        print(f"actual sha256: `{decode_logits_comparison['actual_sha256']}`")
        print(f"expected sha256: `{decode_logits_comparison['expected_sha256']}`")

    return 0 if all_pass else 1



if __name__ == "__main__":
    raise SystemExit(main())
