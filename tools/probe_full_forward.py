#!/usr/bin/env python3
"""Probe full Granite MoE forward pass without instantiating HF model.

This parameterizes the proven layer-0 execution across all decoder layers:

  embedding -> for layer in layers: attention + router + selected expert slices -> final norm -> tied LM head

It compares every layer output and final logits against an HF golden trace.
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

from glacial.granite import FINAL_NORM_TENSOR, granite_rmsnorm, required_layer_tensors, run_layer
from glacial.weights import read_full_bf16_tensor
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
def events_by_layer(records: list[dict[str, Any]], event: str) -> dict[int, dict[str, Any]]:
    out = {}
    for record in records:
        if record.get("event") == event:
            out[int(record["layer"])] = record
    return out



def compare_top_tokens(actual_logits, final_event: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    import torch

    logits_fp32 = actual_logits[0, -1].float()
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
    parser.add_argument("--max-layers", type=int, default=None, help="Debug: run only the first N layers")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_torch()
    import torch.nn.functional as F

    trace_path = args.trace or latest_trace()
    records = load_trace(trace_path)
    metadata = first(records, "trace_metadata")
    inputs = first(records, "inputs")
    embedding_expected = first(records, "embedding_output")
    final_norm_expected = first(records, "final_norm_output")
    final_logits_expected = first(records, "final_logits")
    layer_expected = events_by_layer(records, "layer_output")

    model_file = resolve_model_file(args, metadata)
    config = metadata.get("config") or {}
    num_layers = int(config.get("num_hidden_layers", 24))
    if args.max_layers is not None:
        num_layers = min(num_layers, int(args.max_layers))

    scalars = {
        "embedding_multiplier": float(config.get("embedding_multiplier", 12.0)),
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(config.get("rope_theta", 1500000.0)),
        "attention_multiplier": float(config.get("attention_multiplier", 0.015625)),
        "residual_multiplier": float(config.get("residual_multiplier", 0.22)),
        "logits_scaling": float(config.get("logits_scaling", 6.0)),
    }

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    required = [EMBED_TENSOR, FINAL_NORM_TENSOR]
    for layer_idx in range(num_layers):
        required.extend(required_layer_tensors(layer_idx))
    missing = [tensor for tensor in required if tensor not in header]
    if missing:
        raise SystemExit(f"{model_file}: missing tensors: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    input_ids, input_shape = flatten_ids(inputs["input_ids"])
    rows, hidden_size = read_embedding_rows(
        model_file,
        input_ids,
        tensor_meta=header[EMBED_TENSOR],
        payload_start=payload_start,
    )
    hidden = rows.view(input_shape[0], input_shape[1], hidden_size) * scalars["embedding_multiplier"]
    hidden = hidden.to(rows.dtype)

    embedding_summary = tensor_summary(hidden)
    embedding_comparison = compare_summaries(embedding_summary, embedding_expected["hidden"], atol=args.atol)

    layer_comparisons = []
    layer_stats = []
    cumulative_weight_bytes = len(input_ids) * hidden_size * BF16_BYTES
    peak_expert_pair_bytes = 0
    peak_selected_expert_count = 0

    for layer_idx in range(num_layers):
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
        peak_expert_pair_bytes = max(peak_expert_pair_bytes, stats["peak_expert_pair_bytes"])
        peak_selected_expert_count = max(peak_selected_expert_count, stats["selected_expert_count"])

        actual_summary = tensor_summary(hidden)
        expected_record = layer_expected.get(layer_idx)
        if expected_record is None:
            comparison = {"layer": layer_idx, "available": False, "pass": False, "reason": "missing trace layer_output"}
        else:
            comp = compare_summaries(actual_summary, expected_record["hidden"], atol=args.atol)
            comparison = {"layer": layer_idx, "available": True, **comp}
        layer_comparisons.append(comparison)

    final_norm_weight = read_full_bf16_tensor(model_file, tensor_name=FINAL_NORM_TENSOR, tensor_meta=header[FINAL_NORM_TENSOR], payload_start=payload_start)
    cumulative_weight_bytes += final_norm_weight.numel() * BF16_BYTES
    final_hidden = granite_rmsnorm(hidden, final_norm_weight, eps=scalars["rms_norm_eps"])
    final_norm_summary = tensor_summary(final_hidden)
    final_norm_comparison = compare_summaries(final_norm_summary, final_norm_expected["hidden"], atol=args.atol) if args.max_layers is None else None

    logits_comparison = None
    top_tokens = None
    logits_summary = None
    lm_head_weight_bytes = 0
    if args.max_layers is None:
        # Tied LM head. Under the 128M prototype budget this fits by itself;
        # later we can chunk this by vocab rows.
        lm_head_weight = read_full_bf16_tensor(model_file, tensor_name=EMBED_TENSOR, tensor_meta=header[EMBED_TENSOR], payload_start=payload_start)
        lm_head_weight_bytes = lm_head_weight.numel() * BF16_BYTES
        cumulative_weight_bytes += lm_head_weight_bytes
        logits = F.linear(final_hidden, lm_head_weight)
        logits = logits / scalars["logits_scaling"]
        logits_summary = tensor_summary(logits)
        logits_comparison = compare_summaries(logits_summary, final_logits_expected["logits"], atol=args.atol)
        top_tokens = compare_top_tokens(logits, final_logits_expected, top_k=int(final_logits_expected.get("top_k", 20)))

    all_layer_pass = all(comp.get("pass", False) and comp.get("sha256_pass", False) for comp in layer_comparisons)
    all_pass = bool(
        embedding_comparison["pass"]
        and embedding_comparison["sha256_pass"]
        and all_layer_pass
        and (args.max_layers is not None or (final_norm_comparison and final_norm_comparison["pass"] and final_norm_comparison["sha256_pass"]))
        and (args.max_layers is not None or (logits_comparison and logits_comparison["pass"] and logits_comparison["sha256_pass"]))
        and (args.max_layers is not None or (top_tokens and top_tokens["token_ids_pass"] and top_tokens["greedy_pass"]))
    )

    result = {
        "trace": str(trace_path),
        "model_file": str(model_file),
        "input_ids": inputs["input_ids"],
        "position_ids": inputs["position_ids"],
        "layers_run": num_layers,
        "embedding_comparison": embedding_comparison,
        "layer_comparisons": layer_comparisons,
        "final_norm_comparison": final_norm_comparison,
        "logits_comparison": logits_comparison,
        "top_tokens": top_tokens,
        "layer_stats": layer_stats,
        "cumulative_weight_bytes": cumulative_weight_bytes,
        "lm_head_weight_bytes": lm_head_weight_bytes,
        "peak_expert_pair_bytes": peak_expert_pair_bytes,
        "peak_selected_expert_count": peak_selected_expert_count,
        "all_layer_pass": all_layer_pass,
        "all_pass": all_pass,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("# Full forward probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"input ids: `{inputs['input_ids']}`")
        print(f"position ids: `{inputs['position_ids']}`")
        print(f"layers run: `{num_layers}`")
        print(f"cumulative weight bytes visited: `{cumulative_weight_bytes}`")
        print(f"lm head weight bytes: `{lm_head_weight_bytes}`")
        print(f"peak expert pair bytes: `{peak_expert_pair_bytes}`")
        print(f"peak selected expert count in a layer: `{peak_selected_expert_count}`")
        print(f"embedding pass: `{embedding_comparison['pass'] and embedding_comparison['sha256_pass']}`")
        print(f"all layer pass: `{all_layer_pass}`")
        if final_norm_comparison is not None:
            print(f"final norm pass: `{final_norm_comparison['pass'] and final_norm_comparison['sha256_pass']}`")
        if logits_comparison is not None:
            print(f"logits pass: `{logits_comparison['pass'] and logits_comparison['sha256_pass']}`")
        if top_tokens is not None:
            print(f"top token ids pass: `{top_tokens['token_ids_pass']}`")
            print(f"greedy pass: `{top_tokens['greedy_pass']}`")
            print(f"greedy token id: `{top_tokens['greedy_token_id']}` expected `{top_tokens['expected_greedy_token_id']}`")
        print(f"all pass: `{all_pass}`")

        print("\n## Layer comparisons\n")
        print("| layer | pass | sha256 | selected experts | peak expert pair bytes |")
        print("|---:|---:|---:|---:|---:|")
        for comp, stats in zip(layer_comparisons, layer_stats):
            print(
                f"| {comp['layer']} | `{comp.get('pass')}` | `{comp.get('sha256_pass')}` | "
                f"{stats['selected_expert_count']} | {stats['peak_expert_pair_bytes']} |"
            )

        if logits_comparison is not None:
            print("\n## Logits\n")
            print(f"shape: `{logits_summary['shape']}`")
            print(f"dtype: `{logits_summary['dtype']}`")
            print(f"sha256 pass: `{logits_comparison['sha256_pass']}`")
            print(f"actual sha256: `{logits_comparison['actual_sha256']}`")
            print(f"expected sha256: `{logits_comparison['expected_sha256']}`")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
