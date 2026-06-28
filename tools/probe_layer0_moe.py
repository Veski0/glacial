#!/usr/bin/env python3
"""Probe layer 0 MoE expert execution with expert-slice loading.

This extends the proven path through the MoE block:

  input_ids -> attention residual -> post-attn RMSNorm -> router -> selected experts
  -> load one expert's input/output slices at a time -> expert MLP -> index_add
  -> MoE residual

It compares against HF trace `layer_moe_output` and `layer_output` events.
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

from glacial.granite import route_tokens
from glacial.weights import read_expert_slice
from probe_embedding import (  # same-directory import when run as tools/probe_*.py
    BF16_BYTES,
    EMBED_TENSOR,
    compare_summaries,
    first,
    latest_trace,
    load_trace,
    read_safetensors_header,
    require_torch,
    resolve_model_file,
    tensor_summary,
)
from probe_layer0_input_norm import granite_rmsnorm, read_full_bf16_tensor
from probe_layer0_router import compute_attention_residual, exact_json_comparison, nested_numeric_comparison

LAYER = 0
POST_NORM_TENSOR = "model.layers.0.post_attention_layernorm.weight"
ROUTER_TENSOR = "model.layers.0.block_sparse_moe.router.layer.weight"
EXPERT_INPUT_TENSOR = "model.layers.0.block_sparse_moe.input_linear.weight"
EXPERT_OUTPUT_TENSOR = "model.layers.0.block_sparse_moe.output_linear.weight"


def find_layer_event(records: list[dict[str, Any]], event: str, layer: int) -> dict[str, Any]:
    for record in records:
        if record.get("event") == event and record.get("layer") == layer:
            return record
    raise SystemExit(
        f"Trace is missing event {event!r} for layer {layer}.\n"
        "Rerun tools/hf_trace_granite.py with the current script."
    )


def run_experts_one_at_a_time(*, model_file: Path, header: dict[str, Any], payload_start: int, router_input, route: dict[str, Any]):
    import torch
    import torch.nn.functional as F

    expert_size = route["expert_size"]
    batch_index = route["batch_index"]
    batch_gates = route["batch_gates"]
    expert_inputs = router_input[batch_index]

    input_meta = header[EXPERT_INPUT_TENSOR]
    output_meta = header[EXPERT_OUTPUT_TENSOR]
    input_shape = [int(x) for x in input_meta["shape"]]
    output_shape = [int(x) for x in output_meta["shape"]]
    num_experts = input_shape[0]
    if output_shape[0] != num_experts:
        raise SystemExit("Input/output expert tensor expert counts differ")

    chunks = []
    offset = 0
    selected_expert_ids = []
    cumulative_expert_weight_bytes = 0
    peak_expert_pair_bytes = 0

    for expert_id, size in enumerate(expert_size):
        size = int(size)
        if size == 0:
            continue

        expert_chunk = expert_inputs[offset : offset + size]
        w_in = read_expert_slice(
            model_file,
            tensor_name=EXPERT_INPUT_TENSOR,
            tensor_meta=input_meta,
            expert_id=expert_id,
            payload_start=payload_start,
        )
        w_out = read_expert_slice(
            model_file,
            tensor_name=EXPERT_OUTPUT_TENSOR,
            tensor_meta=output_meta,
            expert_id=expert_id,
            payload_start=payload_start,
        )
        expert_pair_bytes = w_in.numel() * BF16_BYTES + w_out.numel() * BF16_BYTES
        cumulative_expert_weight_bytes += expert_pair_bytes
        peak_expert_pair_bytes = max(peak_expert_pair_bytes, expert_pair_bytes)
        selected_expert_ids.append(expert_id)

        hidden = F.linear(expert_chunk, w_in)
        first_half, second_half = hidden.chunk(2, dim=-1)
        hidden = F.silu(first_half) * second_half
        expert_output = F.linear(hidden, w_out)
        expert_output = expert_output * batch_gates[offset : offset + size, None]
        chunks.append(expert_output)
        offset += size

    if offset != expert_inputs.shape[0]:
        raise SystemExit(f"Expert chunk offset {offset} != expert input rows {expert_inputs.shape[0]}")

    if not chunks:
        expert_outputs = torch.empty((0, router_input.shape[-1]), dtype=router_input.dtype, device=router_input.device)
    else:
        expert_outputs = torch.cat(chunks, dim=0)

    zeros = torch.zeros((router_input.shape[0], router_input.shape[-1]), dtype=expert_outputs.dtype, device=expert_outputs.device)
    moe_output_flat = zeros.index_add(0, batch_index, expert_outputs)

    return {
        "moe_output_flat": moe_output_flat,
        "selected_expert_ids": selected_expert_ids,
        "cumulative_expert_weight_bytes": cumulative_expert_weight_bytes,
        "peak_expert_pair_bytes": peak_expert_pair_bytes,
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
    router_expected_event = find_layer_event(records, "layer_router", LAYER)
    moe_expected_event = find_layer_event(records, "layer_moe_output", LAYER)
    layer_expected_event = find_layer_event(records, "layer_output", LAYER)

    model_file = resolve_model_file(args, metadata)
    config = metadata.get("config") or {}
    top_k = int(config.get("num_experts_per_tok", 8))
    num_experts = int(config.get("num_local_experts", 32))
    residual_multiplier = float(args.residual_multiplier) if args.residual_multiplier is not None else float(config.get("residual_multiplier", 0.22))

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    required = [
        EMBED_TENSOR,
        POST_NORM_TENSOR,
        ROUTER_TENSOR,
        EXPERT_INPUT_TENSOR,
        EXPERT_OUTPUT_TENSOR,
        # compute_attention_residual checks its own attention dependencies too.
    ]
    missing = [tensor for tensor in required if tensor not in header]
    if missing:
        raise SystemExit(f"{model_file}: missing tensors: {missing}")

    attention_residual, loaded_weight_bytes_before_moe, scalars = compute_attention_residual(
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        inputs=inputs,
        config=config,
        args=args,
    )

    post_norm_weight = read_full_bf16_tensor(model_file, tensor_name=POST_NORM_TENSOR, tensor_meta=header[POST_NORM_TENSOR], payload_start=payload_start)
    router_weight = read_full_bf16_tensor(model_file, tensor_name=ROUTER_TENSOR, tensor_meta=header[ROUTER_TENSOR], payload_start=payload_start)
    post_norm_hidden = granite_rmsnorm(attention_residual, post_norm_weight, eps=scalars["rms_norm_eps"])
    router_input = post_norm_hidden.reshape(-1, int(scalars["hidden_size"]))

    route = route_tokens(router_input, router_weight, top_k=top_k, num_experts=num_experts)
    expert_result = run_experts_one_at_a_time(
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        router_input=router_input,
        route=route,
    )

    batch_size = len(inputs["input_ids"])
    seq_len = len(inputs["input_ids"][0])
    hidden_size = int(scalars["hidden_size"])
    moe_output = expert_result["moe_output_flat"].view(batch_size, seq_len, hidden_size)
    layer_output = attention_residual + moe_output * residual_multiplier

    selected_experts = route["top_k_indices"].view(batch_size, seq_len, top_k).detach().cpu().tolist()
    batch_index = route["batch_index"].detach().cpu().tolist()
    batch_gates = route["batch_gates"].float().detach().cpu().tolist()

    actual = {
        "moe_input": tensor_summary(router_input.view(batch_size, seq_len, hidden_size)),
        "moe_output": tensor_summary(moe_output),
        "layer_output": tensor_summary(layer_output),
    }
    expected = {
        "moe_input": moe_expected_event["input"],
        "moe_output": moe_expected_event["output"],
        "layer_output": layer_expected_event["hidden"],
    }
    summary_comparisons = {name: compare_summaries(actual[name], expected[name], atol=args.atol) for name in actual}
    list_comparisons = {
        "selected_experts": exact_json_comparison(selected_experts, router_expected_event.get("selected_experts")),
        "expert_size": exact_json_comparison(route["expert_size"], router_expected_event.get("expert_size")),
        "batch_index": exact_json_comparison(batch_index, router_expected_event.get("batch_index")),
        "batch_gates": nested_numeric_comparison(batch_gates, router_expected_event.get("batch_gates"), atol=args.atol),
    }

    cumulative_weight_bytes = (
        loaded_weight_bytes_before_moe
        + post_norm_weight.numel() * BF16_BYTES
        + router_weight.numel() * BF16_BYTES
        + expert_result["cumulative_expert_weight_bytes"]
    )
    all_summary_pass = all(comp["pass"] for comp in summary_comparisons.values())
    all_sha256_pass = all(comp["sha256_pass"] for comp in summary_comparisons.values())
    all_list_pass = all((not comp.get("available", True)) or comp.get("pass", False) for comp in list_comparisons.values())
    all_pass = all_summary_pass and all_sha256_pass and all_list_pass

    result = {
        "trace": str(trace_path),
        "model_file": str(model_file),
        "layer": LAYER,
        "input_ids": inputs["input_ids"],
        "position_ids": inputs["position_ids"],
        "top_k": top_k,
        "num_experts": num_experts,
        "selected_expert_ids": expert_result["selected_expert_ids"],
        "selected_expert_count": len(expert_result["selected_expert_ids"]),
        "expert_size": route["expert_size"],
        "loaded_weight_bytes_before_moe": loaded_weight_bytes_before_moe,
        "cumulative_expert_weight_bytes": expert_result["cumulative_expert_weight_bytes"],
        "peak_expert_pair_bytes": expert_result["peak_expert_pair_bytes"],
        "cumulative_weight_bytes": cumulative_weight_bytes,
        "summary_comparisons": summary_comparisons,
        "list_comparisons": list_comparisons,
        "actual": actual,
        "expected": expected,
        "all_summary_pass": all_summary_pass,
        "all_sha256_pass": all_sha256_pass,
        "all_list_pass": all_list_pass,
        "all_pass": all_pass,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("# Layer 0 MoE probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"input ids: `{inputs['input_ids']}`")
        print(f"position ids: `{inputs['position_ids']}`")
        print(f"top_k: `{top_k}`")
        print(f"selected expert ids: `{expert_result['selected_expert_ids']}`")
        print(f"selected expert count: `{len(expert_result['selected_expert_ids'])}` / `{num_experts}`")
        print(f"expert_size: `{route['expert_size']}`")
        print(f"loaded weight bytes before MoE: `{loaded_weight_bytes_before_moe}`")
        print(f"cumulative expert weight bytes: `{expert_result['cumulative_expert_weight_bytes']}`")
        print(f"peak expert pair bytes: `{expert_result['peak_expert_pair_bytes']}`")
        print(f"cumulative weight bytes: `{cumulative_weight_bytes}`")
        print(f"all summary pass: `{all_summary_pass}`")
        print(f"all list pass: `{all_list_pass}`")
        print(f"all sha256 pass: `{all_sha256_pass}`")
        print(f"all pass: `{all_pass}`")

        print("\n## Summary comparisons\n")
        for name in ["moe_input", "moe_output", "layer_output"]:
            comp = summary_comparisons[name]
            print(f"### {name}\n")
            print(f"shape: `{actual[name]['shape']}`")
            print(f"dtype: `{actual[name]['dtype']}`")
            print(f"pass: `{comp['pass']}`")
            print(f"sha256 pass: `{comp['sha256_pass']}`")
            print(f"actual sha256: `{comp['actual_sha256']}`")
            print(f"expected sha256: `{comp['expected_sha256']}`")
            print("")

        print("## Routing list comparisons\n")
        print("| value | available | pass | max abs diff |")
        print("|---|---:|---:|---:|")
        for name, comp in list_comparisons.items():
            print(f"| `{name}` | `{comp.get('available')}` | `{comp.get('pass')}` | `{comp.get('max_abs_diff')}` |")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
