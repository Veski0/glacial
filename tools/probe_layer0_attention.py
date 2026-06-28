#!/usr/bin/env python3
"""Probe layer 0 eager attention through o_proj and residual add.

This extends the proven path:

  input_ids -> embedding -> RMSNorm -> q/k/v -> RoPE -> causal attention -> o_proj -> residual

It compares against HF trace `layer_attention_output` and
`layer_post_attention_norm.input` events.
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

from glacial.granite import build_granite_causal_mask, repeat_kv
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
from probe_layer0_rope import apply_rotary_pos_emb, granite_rope_cos_sin

LAYER = 0
NORM_TENSOR = "model.layers.0.input_layernorm.weight"
Q_TENSOR = "model.layers.0.self_attn.q_proj.weight"
K_TENSOR = "model.layers.0.self_attn.k_proj.weight"
V_TENSOR = "model.layers.0.self_attn.v_proj.weight"
O_TENSOR = "model.layers.0.self_attn.o_proj.weight"


def find_layer_event(records: list[dict[str, Any]], event: str, layer: int) -> dict[str, Any]:
    for record in records:
        if record.get("event") == event and record.get("layer") == layer:
            return record
    raise SystemExit(
        f"Trace is missing event {event!r} for layer {layer}.\n"
        "Rerun tools/hf_trace_granite.py with the current script."
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
    parser.add_argument("--attention-multiplier", type=float, default=None, help="Override config attention_multiplier")
    parser.add_argument("--residual-multiplier", type=float, default=None, help="Override config residual_multiplier")
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
    attention_expected_event = find_layer_event(records, "layer_attention_output", LAYER)
    residual_expected_event = find_layer_event(records, "layer_post_attention_norm", LAYER)

    model_file = resolve_model_file(args, metadata)
    config = metadata.get("config") or {}
    embedding_multiplier = float(args.embedding_multiplier) if args.embedding_multiplier is not None else float(config.get("embedding_multiplier", 12.0))
    rms_norm_eps = float(args.rms_norm_eps) if args.rms_norm_eps is not None else float(config.get("rms_norm_eps", 1e-6))
    rope_theta = float(args.rope_theta) if args.rope_theta is not None else float(config.get("rope_theta", 1500000.0))
    attention_multiplier = float(args.attention_multiplier) if args.attention_multiplier is not None else float(config.get("attention_multiplier", 0.015625))
    residual_multiplier = float(args.residual_multiplier) if args.residual_multiplier is not None else float(config.get("residual_multiplier", 0.22))
    hidden_size = int(config.get("hidden_size", 1024))
    num_attention_heads = int(config.get("num_attention_heads", 16))
    num_key_value_heads = int(config.get("num_key_value_heads", 8))
    head_dim = hidden_size // num_attention_heads
    num_key_value_groups = num_attention_heads // num_key_value_heads

    input_ids, input_shape = flatten_ids(inputs["input_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
    position_ids = torch.tensor(inputs["position_ids"], dtype=torch.long)
    cache_position = torch.tensor(inputs["cache_position"], dtype=torch.long)

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    required = [EMBED_TENSOR, NORM_TENSOR, Q_TENSOR, K_TENSOR, V_TENSOR, O_TENSOR]
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
    residual = rows.view(input_shape[0], input_shape[1], hidden_size) * embedding_multiplier
    residual = residual.to(rows.dtype)

    norm_weight = read_full_bf16_tensor(model_file, tensor_name=NORM_TENSOR, tensor_meta=header[NORM_TENSOR], payload_start=payload_start)
    hidden = granite_rmsnorm(residual, norm_weight, eps=rms_norm_eps)

    q_weight = read_full_bf16_tensor(model_file, tensor_name=Q_TENSOR, tensor_meta=header[Q_TENSOR], payload_start=payload_start)
    k_weight = read_full_bf16_tensor(model_file, tensor_name=K_TENSOR, tensor_meta=header[K_TENSOR], payload_start=payload_start)
    v_weight = read_full_bf16_tensor(model_file, tensor_name=V_TENSOR, tensor_meta=header[V_TENSOR], payload_start=payload_start)
    o_weight = read_full_bf16_tensor(model_file, tensor_name=O_TENSOR, tensor_meta=header[O_TENSOR], payload_start=payload_start)

    q_proj = F.linear(hidden, q_weight)
    k_proj = F.linear(hidden, k_weight)
    v_proj = F.linear(hidden, v_weight)

    batch_size, q_len, _ = q_proj.shape
    query_states = q_proj.view(batch_size, q_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = k_proj.view(batch_size, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = v_proj.view(batch_size, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    cos, sin = granite_rope_cos_sin(position_ids=position_ids, head_dim=head_dim, rope_theta=rope_theta, dtype=query_states.dtype)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attention_multiplier
    causal_mask = build_granite_causal_mask(
        attention_mask=attention_mask,
        input_tensor=hidden,
        cache_position=cache_position,
    )
    attn_weights = attn_weights + causal_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, q_len, -1)
    attn_output = F.linear(attn_output, o_weight)
    attention_residual = residual + attn_output * residual_multiplier

    actual = {
        "attention_output": tensor_summary(attn_output),
        "attention_residual": tensor_summary(attention_residual),
    }
    expected = {
        "attention_output": attention_expected_event["output"],
        "attention_residual": residual_expected_event["input"],
    }
    comparisons = {name: compare_summaries(actual[name], expected[name], atol=args.atol) for name in actual}
    all_numeric_pass = all(comp["pass"] for comp in comparisons.values())
    all_sha256_pass = all(comp["sha256_pass"] for comp in comparisons.values())
    all_pass = all_numeric_pass and all_sha256_pass

    loaded_weight_bytes = (
        len(input_ids) * hidden_size * BF16_BYTES
        + norm_weight.numel() * BF16_BYTES
        + q_weight.numel() * BF16_BYTES
        + k_weight.numel() * BF16_BYTES
        + v_weight.numel() * BF16_BYTES
        + o_weight.numel() * BF16_BYTES
    )

    result = {
        "trace": str(trace_path),
        "model_file": str(model_file),
        "layer": LAYER,
        "input_ids": inputs["input_ids"],
        "position_ids": inputs["position_ids"],
        "embedding_tensor": EMBED_TENSOR,
        "norm_tensor": NORM_TENSOR,
        "projection_tensors": {"q": Q_TENSOR, "k": K_TENSOR, "v": V_TENSOR, "o": O_TENSOR},
        "embedding_multiplier": embedding_multiplier,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "attention_multiplier": attention_multiplier,
        "residual_multiplier": residual_multiplier,
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
        print("# Layer 0 attention probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"input ids: `{inputs['input_ids']}`")
        print(f"position ids: `{inputs['position_ids']}`")
        print(f"loaded weight bytes: `{loaded_weight_bytes}`")
        print(f"attention multiplier: `{attention_multiplier}`")
        print(f"residual multiplier: `{residual_multiplier}`")
        print(f"all numeric pass: `{all_numeric_pass}`")
        print(f"all sha256 pass: `{all_sha256_pass}`")
        print(f"all pass: `{all_pass}`")

        for name in ["attention_output", "attention_residual"]:
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
