#!/usr/bin/env python3
"""Probe layer 0 q/k reshape and RoPE from byte-loaded weights.

This extends the proven path:

  input_ids -> embedding -> RMSNorm -> q/k projections -> head reshape -> RoPE

It compares q/k before and after RoPE against HF trace events.
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

from glacial.granite import apply_rotary_pos_emb, granite_rope_cos_sin, rotate_half
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


def find_layer_event(records: list[dict[str, Any]], event: str, layer: int) -> dict[str, Any]:
    for record in records:
        if record.get("event") == event and record.get("layer") == layer:
            return record
    raise SystemExit(
        f"Trace is missing event {event!r} for layer {layer}.\n"
        "Rerun tools/hf_trace_granite.py with the current script so RoPE hook events are present."
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
    parser.add_argument("--rope-theta", type=float, default=None, help="Override config rope_theta")
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
    rotary_expected_event = find_layer_event(records, "layer_attention_rotary", LAYER)

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
    rope_theta = float(args.rope_theta) if args.rope_theta is not None else float(config.get("rope_theta", 1500000.0))
    hidden_size = int(config.get("hidden_size", 1024))
    num_attention_heads = int(config.get("num_attention_heads", 16))
    num_key_value_heads = int(config.get("num_key_value_heads", 8))
    head_dim = hidden_size // num_attention_heads

    input_ids, input_shape = flatten_ids(inputs["input_ids"])
    position_ids = torch.tensor(inputs["position_ids"], dtype=torch.long)

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    required = [EMBED_TENSOR, NORM_TENSOR, Q_TENSOR, K_TENSOR]
    missing = [tensor for tensor in required if tensor not in header]
    if missing:
        raise SystemExit(f"{model_file}: missing tensors: {missing}")

    rows, actual_hidden_size = read_embedding_rows(
        model_file,
        input_ids,
        tensor_meta=header[EMBED_TENSOR],
        payload_start=payload_start,
    )
    if actual_hidden_size != hidden_size:
        raise SystemExit(f"Trace/config hidden_size {hidden_size} != embedding hidden size {actual_hidden_size}")
    embedding_hidden = rows.view(input_shape[0], input_shape[1], hidden_size) * embedding_multiplier
    embedding_hidden = embedding_hidden.to(rows.dtype)

    norm_weight = read_full_bf16_tensor(
        model_file,
        tensor_name=NORM_TENSOR,
        tensor_meta=header[NORM_TENSOR],
        payload_start=payload_start,
    )
    normed_hidden = granite_rmsnorm(embedding_hidden, norm_weight, eps=rms_norm_eps)

    q_weight = read_full_bf16_tensor(model_file, tensor_name=Q_TENSOR, tensor_meta=header[Q_TENSOR], payload_start=payload_start)
    k_weight = read_full_bf16_tensor(model_file, tensor_name=K_TENSOR, tensor_meta=header[K_TENSOR], payload_start=payload_start)
    q_proj = F.linear(normed_hidden, q_weight)
    k_proj = F.linear(normed_hidden, k_weight)

    batch_size, seq_len, _ = q_proj.shape
    q_states = q_proj.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    k_states = k_proj.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    cos, sin = granite_rope_cos_sin(
        position_ids=position_ids,
        head_dim=head_dim,
        rope_theta=rope_theta,
        dtype=q_states.dtype,
    )
    q_rot, k_rot = apply_rotary_pos_emb(q_states, k_states, cos, sin, unsqueeze_dim=int(rotary_expected_event.get("unsqueeze_dim", 1)))

    actual = {
        "q_input": tensor_summary(q_states),
        "k_input": tensor_summary(k_states),
        "cos": tensor_summary(cos),
        "sin": tensor_summary(sin),
        "q_output": tensor_summary(q_rot),
        "k_output": tensor_summary(k_rot),
    }
    expected = {name: rotary_expected_event[name] for name in actual}
    comparisons = {name: compare_summaries(actual[name], expected[name], atol=args.atol) for name in actual}
    all_numeric_pass = all(comp["pass"] for comp in comparisons.values())
    all_sha256_pass = all(comp["sha256_pass"] for comp in comparisons.values())
    all_pass = all_numeric_pass and all_sha256_pass

    loaded_weight_bytes = (
        len(input_ids) * hidden_size * BF16_BYTES
        + norm_weight.numel() * BF16_BYTES
        + q_weight.numel() * BF16_BYTES
        + k_weight.numel() * BF16_BYTES
    )

    result = {
        "trace": str(trace_path),
        "model_file": str(model_file),
        "layer": LAYER,
        "input_ids": inputs["input_ids"],
        "position_ids": inputs["position_ids"],
        "embedding_tensor": EMBED_TENSOR,
        "norm_tensor": NORM_TENSOR,
        "projection_tensors": {"q": Q_TENSOR, "k": K_TENSOR},
        "embedding_multiplier": embedding_multiplier,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "head_dim": head_dim,
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
        print("# Layer 0 RoPE probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"input ids: `{inputs['input_ids']}`")
        print(f"position ids: `{inputs['position_ids']}`")
        print(f"embedding multiplier: `{embedding_multiplier}`")
        print(f"rms norm eps: `{rms_norm_eps}`")
        print(f"rope theta: `{rope_theta}`")
        print(f"loaded weight bytes: `{loaded_weight_bytes}`")
        print(f"all numeric pass: `{all_numeric_pass}`")
        print(f"all sha256 pass: `{all_sha256_pass}`")
        print(f"all pass: `{all_pass}`")

        for name in ["q_input", "k_input", "cos", "sin", "q_output", "k_output"]:
            comp = comparisons[name]
            print(f"\n## {name}\n")
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
