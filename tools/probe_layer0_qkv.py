#!/usr/bin/env python3
"""Probe layer 0 q/k/v attention projections from byte-loaded weights.

This extends the proven path:

  input_ids -> embedding -> layer0 input RMSNorm -> q_proj/k_proj/v_proj

It compares projection outputs against HF trace hook events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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

LAYER = 0
NORM_TENSOR = "model.layers.0.input_layernorm.weight"
Q_TENSOR = "model.layers.0.self_attn.q_proj.weight"
K_TENSOR = "model.layers.0.self_attn.k_proj.weight"
V_TENSOR = "model.layers.0.self_attn.v_proj.weight"

EXPECTED_EVENTS = {
    "q": "layer_attention_q_proj",
    "k": "layer_attention_k_proj",
    "v": "layer_attention_v_proj",
}


def find_layer_event(records: list[dict[str, Any]], event: str, layer: int) -> dict[str, Any]:
    for record in records:
        if record.get("event") == event and record.get("layer") == layer:
            return record
    raise SystemExit(
        f"Trace is missing event {event!r} for layer {layer}.\n"
        "Rerun tools/hf_trace_granite.py with the current script so attention projection hook events are present."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=None, help="HF golden trace JSONL. Defaults to latest traces/*.jsonl")
    parser.add_argument("--model-file", type=Path, default=None, help="Local model.safetensors path")
    parser.add_argument("--model-id", default=None, help="HF model id fallback. Defaults to trace metadata.")
    parser.add_argument("--revision", default=None, help="HF revision fallback. Defaults to trace metadata.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--embedding-multiplier", type=float, default=None, help="Override config embedding_multiplier")
    parser.add_argument("--rms-norm-eps", type=float, default=None, help="Override config rms_norm_eps")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_torch()
    import torch
    import torch.nn.functional as F

    trace_path = args.trace or latest_trace()
    records = load_trace(trace_path)
    metadata = first(records, "trace_metadata")
    inputs = first(records, "inputs")
    expected_events = {name: find_layer_event(records, event, LAYER) for name, event in EXPECTED_EVENTS.items()}

    model_file = resolve_model_file(args, metadata)
    config = metadata.get("config") or {}
    embedding_multiplier = (
        float(args.embedding_multiplier)
        if args.embedding_multiplier is not None
        else float(config.get("embedding_multiplier", 12.0))
    )
    rms_norm_eps = (
        float(args.rms_norm_eps)
        if args.rms_norm_eps is not None
        else float(config.get("rms_norm_eps", 1e-6))
    )

    input_ids, input_shape = flatten_ids(inputs["input_ids"])
    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len

    required = {
        "embedding": EMBED_TENSOR,
        "norm": NORM_TENSOR,
        "q": Q_TENSOR,
        "k": K_TENSOR,
        "v": V_TENSOR,
    }
    missing = [tensor for tensor in required.values() if tensor not in header]
    if missing:
        raise SystemExit(f"{model_file}: missing tensors: {missing}")

    rows, hidden_size = read_embedding_rows(
        model_file,
        input_ids,
        tensor_meta=header[EMBED_TENSOR],
        payload_start=payload_start,
    )
    embedding_hidden = rows.view(input_shape[0], input_shape[1], hidden_size) * embedding_multiplier
    embedding_hidden = embedding_hidden.to(rows.dtype)

    norm_weight = read_full_bf16_tensor(
        model_file,
        tensor_name=NORM_TENSOR,
        tensor_meta=header[NORM_TENSOR],
        payload_start=payload_start,
    )
    normed_hidden = granite_rmsnorm(embedding_hidden, norm_weight, eps=rms_norm_eps)

    weights = {
        "q": read_full_bf16_tensor(model_file, tensor_name=Q_TENSOR, tensor_meta=header[Q_TENSOR], payload_start=payload_start),
        "k": read_full_bf16_tensor(model_file, tensor_name=K_TENSOR, tensor_meta=header[K_TENSOR], payload_start=payload_start),
        "v": read_full_bf16_tensor(model_file, tensor_name=V_TENSOR, tensor_meta=header[V_TENSOR], payload_start=payload_start),
    }

    actual_tensors = {
        # Use F.linear to match nn.Linear.forward exactly: y = x A^T + b.
        "q": F.linear(normed_hidden, weights["q"]),
        "k": F.linear(normed_hidden, weights["k"]),
        "v": F.linear(normed_hidden, weights["v"]),
    }
    actual = {name: tensor_summary(tensor) for name, tensor in actual_tensors.items()}
    expected = {name: expected_events[name]["output"] for name in ["q", "k", "v"]}
    comparisons = {
        name: compare_summaries(actual[name], expected[name], atol=args.atol)
        for name in ["q", "k", "v"]
    }
    all_numeric_pass = all(comp["pass"] for comp in comparisons.values())
    all_sha256_pass = all(comp["sha256_pass"] for comp in comparisons.values())
    all_pass = all_numeric_pass and all_sha256_pass

    loaded_weight_bytes = (
        len(input_ids) * hidden_size * BF16_BYTES
        + norm_weight.numel() * BF16_BYTES
        + sum(weight.numel() * BF16_BYTES for weight in weights.values())
    )

    result = {
        "trace": str(trace_path),
        "model_file": str(model_file),
        "layer": LAYER,
        "input_ids": inputs["input_ids"],
        "embedding_tensor": EMBED_TENSOR,
        "norm_tensor": NORM_TENSOR,
        "projection_tensors": {"q": Q_TENSOR, "k": K_TENSOR, "v": V_TENSOR},
        "embedding_multiplier": embedding_multiplier,
        "rms_norm_eps": rms_norm_eps,
        "loaded_weight_bytes": loaded_weight_bytes,
        "actual": actual,
        "expected": expected,
        "comparisons": comparisons,
        "all_numeric_pass": all_numeric_pass,
        "all_sha256_pass": all_sha256_pass,
        "all_pass": all_pass,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("# Layer 0 q/k/v projection probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"input ids: `{inputs['input_ids']}`")
        print(f"embedding multiplier: `{embedding_multiplier}`")
        print(f"rms norm eps: `{rms_norm_eps}`")
        print(f"loaded weight bytes: `{loaded_weight_bytes}`")
        print(f"all numeric pass: `{all_numeric_pass}`")
        print(f"all sha256 pass: `{all_sha256_pass}`")
        print(f"all pass: `{all_pass}`")

        for name in ["q", "k", "v"]:
            comp = comparisons[name]
            print(f"\n## {name}_proj\n")
            print(f"shape: `{actual[name]['shape']}`")
            print(f"dtype: `{actual[name]['dtype']}`")
            print(f"pass: `{comp['pass']}`")
            print(f"sha256 pass: `{comp['sha256_pass']}`")
            print(f"actual sha256: `{comp['actual_sha256']}`")
            print(f"expected sha256: `{comp['expected_sha256']}`")
            print("\n| metric | actual | expected | abs diff | pass |")
            print("|---|---:|---:|---:|---:|")
            for key, row in comp["numeric"].items():
                print(f"| `{key}` | {row['actual']} | {row['expected']} | {row['abs_diff']} | `{row['pass']}` |")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
