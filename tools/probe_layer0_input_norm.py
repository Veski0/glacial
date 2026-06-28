#!/usr/bin/env python3
"""Probe layer 0 input RMSNorm from byte-loaded weights.

This builds on the embedding probe:

  input_ids -> embedding rows -> embedding_multiplier -> layer0 input RMSNorm

It compares against the `layer_input_norm` event in a fresh HF golden trace.
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

from glacial.granite import granite_rmsnorm
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

LAYER = 0
NORM_TENSOR = "model.layers.0.input_layernorm.weight"


def find_layer_event(records: list[dict[str, Any]], event: str, layer: int) -> dict[str, Any]:
    for record in records:
        if record.get("event") == event and record.get("layer") == layer:
            return record
    raise SystemExit(
        f"Trace is missing event {event!r} for layer {layer}.\n"
        "Rerun tools/hf_trace_granite.py with the current script so RMSNorm hook events are present."
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
    import torch  # noqa: F401

    trace_path = args.trace or latest_trace()
    records = load_trace(trace_path)
    metadata = first(records, "trace_metadata")
    inputs = first(records, "inputs")
    expected_event = find_layer_event(records, "layer_input_norm", LAYER)

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

    embed_meta = header.get(EMBED_TENSOR)
    if embed_meta is None:
        raise SystemExit(f"{model_file}: missing tensor {EMBED_TENSOR!r}")
    norm_meta = header.get(NORM_TENSOR)
    if norm_meta is None:
        raise SystemExit(f"{model_file}: missing tensor {NORM_TENSOR!r}")

    rows, hidden_size = read_embedding_rows(model_file, input_ids, tensor_meta=embed_meta, payload_start=payload_start)
    embedding_hidden = rows.view(input_shape[0], input_shape[1], hidden_size) * embedding_multiplier
    embedding_hidden = embedding_hidden.to(rows.dtype)

    norm_weight = read_full_bf16_tensor(
        model_file,
        tensor_name=NORM_TENSOR,
        tensor_meta=norm_meta,
        payload_start=payload_start,
    )
    actual_hidden = granite_rmsnorm(embedding_hidden, norm_weight, eps=rms_norm_eps)

    actual = tensor_summary(actual_hidden)
    expected = expected_event["output"]
    comparison = compare_summaries(actual, expected, atol=args.atol)

    result = {
        "trace": str(trace_path),
        "model_file": str(model_file),
        "layer": LAYER,
        "input_ids": inputs["input_ids"],
        "embedding_tensor": EMBED_TENSOR,
        "norm_tensor": NORM_TENSOR,
        "embedding_multiplier": embedding_multiplier,
        "rms_norm_eps": rms_norm_eps,
        "loaded_weight_bytes": len(input_ids) * hidden_size * BF16_BYTES + norm_weight.numel() * BF16_BYTES,
        "actual": actual,
        "expected": expected,
        "comparison": comparison,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("# Layer 0 input RMSNorm probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"input ids: `{inputs['input_ids']}`")
        print(f"embedding multiplier: `{embedding_multiplier}`")
        print(f"rms norm eps: `{rms_norm_eps}`")
        print(f"loaded weight bytes: `{result['loaded_weight_bytes']}`")
        print("\n## Comparison\n")
        print(f"pass: `{comparison['pass']}`")
        print(f"shape pass: `{comparison['shape_pass']}`")
        print(f"dtype pass: `{comparison['dtype_pass']}`")
        print(f"sha256 pass: `{comparison['sha256_pass']}`")
        print(f"actual sha256: `{comparison['actual_sha256']}`")
        print(f"expected sha256: `{comparison['expected_sha256']}`")
        print("\n| metric | actual | expected | abs diff | pass |")
        print("|---|---:|---:|---:|---:|")
        for key, row in comparison["numeric"].items():
            print(f"| `{key}` | {row['actual']} | {row['expected']} | {row['abs_diff']} | `{row['pass']}` |")

    return 0 if comparison["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
