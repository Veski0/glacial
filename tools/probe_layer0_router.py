#!/usr/bin/env python3
"""Probe layer 0 post-attention RMSNorm and MoE router/gate.

This extends the proven path:

  input_ids -> attention residual -> post-attn RMSNorm -> router logits/top-k/gates

It compares against HF trace `layer_post_attention_norm` and `layer_router`
events. This is the gate before expert loading.
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
from probe_layer0_attention import build_granite_causal_mask, repeat_kv
from probe_layer0_input_norm import granite_rmsnorm, read_full_bf16_tensor
from probe_layer0_rope import apply_rotary_pos_emb, granite_rope_cos_sin

LAYER = 0
INPUT_NORM_TENSOR = "model.layers.0.input_layernorm.weight"
POST_NORM_TENSOR = "model.layers.0.post_attention_layernorm.weight"
Q_TENSOR = "model.layers.0.self_attn.q_proj.weight"
K_TENSOR = "model.layers.0.self_attn.k_proj.weight"
V_TENSOR = "model.layers.0.self_attn.v_proj.weight"
O_TENSOR = "model.layers.0.self_attn.o_proj.weight"
ROUTER_TENSOR = "model.layers.0.block_sparse_moe.router.layer.weight"


def find_layer_event(records: list[dict[str, Any]], event: str, layer: int) -> dict[str, Any]:
    for record in records:
        if record.get("event") == event and record.get("layer") == layer:
            return record
    raise SystemExit(
        f"Trace is missing event {event!r} for layer {layer}.\n"
        "Rerun tools/hf_trace_granite.py with the current script."
    )


def flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numeric(item))
        return out
    return [float(value)]


def nested_numeric_comparison(actual: Any, expected: Any, *, atol: float) -> dict[str, Any]:
    if not isinstance(expected, list):
        return {
            "available": False,
            "reason": "expected value is not inline in trace",
        }
    actual_flat = flatten_numeric(actual)
    expected_flat = flatten_numeric(expected)
    if len(actual_flat) != len(expected_flat):
        return {
            "available": True,
            "pass": False,
            "actual_len": len(actual_flat),
            "expected_len": len(expected_flat),
            "max_abs_diff": None,
        }
    diffs = [abs(a - e) for a, e in zip(actual_flat, expected_flat)]
    max_abs_diff = max(diffs) if diffs else 0.0
    return {
        "available": True,
        "pass": max_abs_diff <= atol,
        "actual_len": len(actual_flat),
        "expected_len": len(expected_flat),
        "max_abs_diff": max_abs_diff,
    }


def exact_json_comparison(actual: Any, expected: Any) -> dict[str, Any]:
    if not isinstance(expected, list):
        return {
            "available": False,
            "reason": "expected value is not inline in trace",
        }
    return {
        "available": True,
        "pass": actual == expected,
    }


def compute_attention_residual(*, model_file: Path, header: dict[str, Any], payload_start: int, inputs: dict[str, Any], config: dict[str, Any], args: argparse.Namespace):
    import torch
    import torch.nn.functional as F

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

    input_norm_weight = read_full_bf16_tensor(model_file, tensor_name=INPUT_NORM_TENSOR, tensor_meta=header[INPUT_NORM_TENSOR], payload_start=payload_start)
    hidden = granite_rmsnorm(residual, input_norm_weight, eps=rms_norm_eps)

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

    loaded_weight_bytes = (
        len(input_ids) * hidden_size * BF16_BYTES
        + input_norm_weight.numel() * BF16_BYTES
        + q_weight.numel() * BF16_BYTES
        + k_weight.numel() * BF16_BYTES
        + v_weight.numel() * BF16_BYTES
        + o_weight.numel() * BF16_BYTES
    )

    return attention_residual, loaded_weight_bytes, {
        "embedding_multiplier": embedding_multiplier,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "attention_multiplier": attention_multiplier,
        "residual_multiplier": residual_multiplier,
        "hidden_size": hidden_size,
    }


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
    post_norm_expected_event = find_layer_event(records, "layer_post_attention_norm", LAYER)
    router_expected_event = find_layer_event(records, "layer_router", LAYER)

    model_file = resolve_model_file(args, metadata)
    config = metadata.get("config") or {}
    top_k = int(config.get("num_experts_per_tok", 8))

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    required = [
        EMBED_TENSOR,
        INPUT_NORM_TENSOR,
        POST_NORM_TENSOR,
        Q_TENSOR,
        K_TENSOR,
        V_TENSOR,
        O_TENSOR,
        ROUTER_TENSOR,
    ]
    missing = [tensor for tensor in required if tensor not in header]
    if missing:
        raise SystemExit(f"{model_file}: missing tensors: {missing}")

    attention_residual, loaded_weight_bytes, scalars = compute_attention_residual(
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        inputs=inputs,
        config=config,
        args=args,
    )

    post_norm_weight = read_full_bf16_tensor(model_file, tensor_name=POST_NORM_TENSOR, tensor_meta=header[POST_NORM_TENSOR], payload_start=payload_start)
    post_norm_hidden = granite_rmsnorm(attention_residual, post_norm_weight, eps=scalars["rms_norm_eps"])
    router_input = post_norm_hidden.reshape(-1, int(scalars["hidden_size"]))

    router_weight = read_full_bf16_tensor(model_file, tensor_name=ROUTER_TENSOR, tensor_meta=header[ROUTER_TENSOR], payload_start=payload_start)
    router_logits = F.linear(router_input, router_weight).float()
    top_k_logits, top_k_indices = router_logits.topk(top_k, dim=1)
    gate_values_fp32 = torch.softmax(top_k_logits, dim=1)
    gate_values_after_hf_cast = gate_values_fp32.to(router_input.dtype)

    batch_size = len(inputs["input_ids"])
    seq_len = len(inputs["input_ids"][0])
    selected_experts = top_k_indices.view(batch_size, seq_len, top_k).detach().cpu().tolist()
    selected_logits_fp32 = top_k_logits.view(batch_size, seq_len, top_k).detach().cpu().tolist()
    gates_fp32 = gate_values_fp32.view(batch_size, seq_len, top_k).detach().cpu().tolist()
    gates_after_cast = gate_values_after_hf_cast.view(batch_size, seq_len, top_k).float().detach().cpu().tolist()

    actual = {
        "post_attention_norm_output": tensor_summary(post_norm_hidden),
        "router_input": tensor_summary(router_input),
        "router_logits": tensor_summary(router_logits),
    }
    expected = {
        "post_attention_norm_output": post_norm_expected_event["output"],
        "router_input": router_expected_event["router_input"],
        "router_logits": router_expected_event["router_logits"],
    }
    summary_comparisons = {name: compare_summaries(actual[name], expected[name], atol=args.atol) for name in actual}
    list_comparisons = {
        "selected_experts": exact_json_comparison(selected_experts, router_expected_event.get("selected_experts")),
        "selected_logits_fp32": nested_numeric_comparison(selected_logits_fp32, router_expected_event.get("selected_logits_fp32"), atol=args.atol),
        "gate_values_fp32_before_hf_cast": nested_numeric_comparison(gates_fp32, router_expected_event.get("gate_values_fp32_before_hf_cast"), atol=args.atol),
        "gate_values_after_hf_cast": nested_numeric_comparison(gates_after_cast, router_expected_event.get("gate_values_after_hf_cast"), atol=args.atol),
    }

    loaded_weight_bytes += post_norm_weight.numel() * BF16_BYTES + router_weight.numel() * BF16_BYTES
    all_summary_pass = all(comp["pass"] for comp in summary_comparisons.values())
    all_list_pass = all((not comp.get("available", True)) or comp.get("pass", False) for comp in list_comparisons.values())
    all_sha256_pass = all(comp["sha256_pass"] for comp in summary_comparisons.values())
    all_pass = all_summary_pass and all_list_pass and all_sha256_pass

    result = {
        "trace": str(trace_path),
        "model_file": str(model_file),
        "layer": LAYER,
        "input_ids": inputs["input_ids"],
        "position_ids": inputs["position_ids"],
        "post_norm_tensor": POST_NORM_TENSOR,
        "router_tensor": ROUTER_TENSOR,
        "top_k": top_k,
        "loaded_weight_bytes": loaded_weight_bytes,
        "scalars": scalars,
        "actual": actual,
        "expected": expected,
        "summary_comparisons": summary_comparisons,
        "list_comparisons": list_comparisons,
        "selected_experts": selected_experts,
        "all_summary_pass": all_summary_pass,
        "all_list_pass": all_list_pass,
        "all_sha256_pass": all_sha256_pass,
        "all_pass": all_pass,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("# Layer 0 router probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"input ids: `{inputs['input_ids']}`")
        print(f"position ids: `{inputs['position_ids']}`")
        print(f"loaded weight bytes: `{loaded_weight_bytes}`")
        print(f"top_k: `{top_k}`")
        print(f"selected experts first token: `{selected_experts[0][0]}`")
        print(f"all summary pass: `{all_summary_pass}`")
        print(f"all list pass: `{all_list_pass}`")
        print(f"all sha256 pass: `{all_sha256_pass}`")
        print(f"all pass: `{all_pass}`")

        print("\n## Summary comparisons\n")
        for name in ["post_attention_norm_output", "router_input", "router_logits"]:
            comp = summary_comparisons[name]
            print(f"### {name}\n")
            print(f"shape: `{actual[name]['shape']}`")
            print(f"dtype: `{actual[name]['dtype']}`")
            print(f"pass: `{comp['pass']}`")
            print(f"sha256 pass: `{comp['sha256_pass']}`")
            print(f"actual sha256: `{comp['actual_sha256']}`")
            print(f"expected sha256: `{comp['expected_sha256']}`")
            print("")

        print("## Gate list comparisons\n")
        print("| value | available | pass | max abs diff |")
        print("|---|---:|---:|---:|")
        for name, comp in list_comparisons.items():
            print(f"| `{name}` | `{comp.get('available')}` | `{comp.get('pass')}` | `{comp.get('max_abs_diff')}` |")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
