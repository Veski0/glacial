#!/usr/bin/env python3
"""Probe LFM2 layer 2: attention (Q/K layernorm + RoPE + GQA) + MoE router + experts.

Layer 2 is the first attention + MoE layer. It tests the operations that
differ most from Granite:

  - Q/K layernorm (RMSNorm on Q and K after projection)  [NEW]
  - Standard RoPE (theta=5M) + GQA (32 heads, 8 KV heads)
  - MoE router: sigmoid + expert_bias + top-4             [NEW]
  - MoE experts: separate per-expert w1/w2/w3             [NEW]

We run layers 0-1 first (proven correct by probe_lfm2_layer0.py) to get
the input hidden state for layer 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_trace(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def first(records: list[dict[str, Any]], event: str, layer: int | None = None) -> dict[str, Any]:
    for r in records:
        if r.get("event") == event and (layer is None or r.get("layer") == layer):
            return r
    raise SystemExit(f"Trace missing event={event!r} layer={layer}")


def latest_trace() -> Path:
    traces = sorted(Path("traces").glob("*lfm2*"), key=lambda p: p.stat().st_mtime)
    if not traces:
        raise SystemExit("No LFM2 traces found.")
    return traces[-1]


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header_len, header


def tensor_summary(tensor) -> dict[str, Any]:
    import torch
    t = tensor.detach()
    out: dict[str, Any] = {
        "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."),
        "numel": int(t.numel()),
    }
    if t.numel() == 0:
        return out
    tf = t.float()
    out.update({
        "sum_fp32": float(tf.sum().item()), "mean_fp32": float(tf.mean().item()),
        "min_fp32": float(tf.min().item()), "max_fp32": float(tf.max().item()),
    })
    try:
        if t.dtype == torch.bfloat16:
            data = t.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()
        else:
            data = t.detach().cpu().contiguous().numpy().tobytes()
    except Exception:
        data = t.float().numpy().tobytes()
    out["sha256"] = hashlib.sha256(data).hexdigest()
    return out


def compare(actual: dict, expected: dict, *, atol: float, label: str) -> bool:
    shape_ok = actual["shape"] == expected["shape"]
    dtype_ok = actual["dtype"] == expected["dtype"]
    sha_ok = actual.get("sha256") == expected.get("sha256")
    numeric_ok = True
    diffs = {}
    for key in ("sum_fp32", "mean_fp32", "min_fp32", "max_fp32"):
        a = float(actual.get(key, 0)); e = float(expected.get(key, 0))
        d = abs(a - e); diffs[key] = d
        if d > atol: numeric_ok = False
    passed = shape_ok and dtype_ok and numeric_ok
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status} — {label}")
    print(f"    shape: {actual['shape']} vs {expected['shape']} ({'OK' if shape_ok else 'MISMATCH'})")
    print(f"    sha256: {'exact' if sha_ok else 'diff'}")
    for k, v in diffs.items():
        print(f"    {k}: abs_diff={v:.2e}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=None)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    import torch
    from glacial.generate import make_inputs
    from glacial.lfm2 import (
        lfm2_rmsnorm, run_short_conv, run_dense_mlp, run_attention,
        route_tokens, run_experts, run_layer, scalar_config, layer_tensor,
    )
    from glacial.weights import SafetensorsWeights

    args = parse_args()
    trace_path = Path(args.trace) if args.trace else latest_trace()
    records = load_trace(trace_path)
    metadata = first(records, "trace_metadata")
    inputs_ev = first(records, "inputs")
    config = metadata["config"]

    model_file = args.model_file
    if model_file is None:
        from huggingface_hub import hf_hub_download
        model_file = hf_hub_download(repo_id=metadata["model_id"], filename="model.safetensors", local_files_only=True)
    model_file = Path(model_file)

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    scalars = scalar_config(config)

    input_ids = inputs_ev["input_ids"]
    if isinstance(input_ids[0], list):
        flat_ids = [int(x) for row in input_ids for x in row]
        input_shape = [len(input_ids), len(input_ids[0])]
    else:
        flat_ids = [int(x) for x in input_ids]
        input_shape = [1, len(input_ids)]

    batch_size, seq_len = input_shape
    hidden_size = int(config.get("hidden_size", 2048))
    eps = float(config.get("norm_eps", 1e-5))
    num_experts = int(config.get("num_experts", 32))
    top_k = int(config.get("num_experts_per_tok", 4))
    use_expert_bias = bool(config.get("use_expert_bias", True))
    norm_topk_prob = bool(config.get("norm_topk_prob", True))
    routed_scaling_factor = float(config.get("routed_scaling_factor", 1.0))

    inputs = make_inputs(flat_ids)

    print(f"# LFM2 Layer 2 Probe (Attention + MoE)")
    print(f"  trace: {trace_path}")
    print(f"  seq_len={seq_len}, hidden={hidden_size}, eps={eps}")
    print(f"  MoE: {num_experts} experts, top-{top_k}, expert_bias={use_expert_bias}")

    all_pass = True
    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start)

    # --- Run layers 0-1 to get input for layer 2 ---
    with provider.embedding_rows(flat_ids) as (rows, hs):
        hidden = rows.view(input_shape[0], input_shape[1], hs)

    for layer_idx in range(2):
        hidden, _ = run_layer(
            layer_idx=layer_idx, hidden=hidden,
            model_file=model_file, header=header, payload_start=payload_start,
            inputs=inputs, config=config, scalars=scalars,
        )

    # --- Layer 2: operator_norm ---
    op_norm_ev = first(records, "layer_operator_norm", layer=2)
    with provider.tensor(layer_tensor(2, "operator_norm.weight")) as norm_w:
        hidden_norm = lfm2_rmsnorm(hidden, norm_w, eps=eps)
    all_pass &= compare(tensor_summary(hidden_norm), op_norm_ev["output"],
                        atol=args.atol, label="operator_norm (layer 2)")

    # --- Layer 2: attention ---
    attn_ev = first(records, "layer_attention_output", layer=2)
    attn_output, kv_pair = run_attention(
        layer_idx=2, hidden=hidden_norm,
        model_file=model_file, header=header, payload_start=payload_start,
        inputs=inputs, config=config, scalars=scalars,
    )
    all_pass &= compare(tensor_summary(attn_output), attn_ev["output"],
                        atol=args.atol, label="Attention (Q/K layernorm + RoPE + GQA)")

    # --- Conv + residual ---
    hidden_after_attn = hidden + attn_output

    # --- ffn_norm ---
    ffn_ev = first(records, "layer_ffn_norm", layer=2)
    with provider.tensor(layer_tensor(2, "ffn_norm.weight")) as ffn_w:
        hidden_for_moe = lfm2_rmsnorm(hidden_after_attn, ffn_w, eps=eps)
    all_pass &= compare(tensor_summary(hidden_for_moe), ffn_ev["output"],
                        atol=args.atol, label="ffn_norm (layer 2)")

    # --- MoE router ---
    router_ev = first(records, "layer_router_logits", layer=2)
    router_input = hidden_for_moe.reshape(-1, hidden_size)
    gate_name = layer_tensor(2, "feed_forward.gate.weight")
    expert_bias = None
    if use_expert_bias:
        with provider.tensor_any(layer_tensor(2, "feed_forward.expert_bias")) as bias_w:
            expert_bias = bias_w.float()
    with provider.tensor(gate_name) as gate_w:
        route = route_tokens(
            router_input, gate_w, top_k=top_k, num_experts=num_experts,
            expert_bias=expert_bias, norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor, use_expert_bias=use_expert_bias,
        )
    all_pass &= compare(tensor_summary(route["router_logits"]), router_ev["router_logits"],
                        atol=args.atol, label="MoE router logits (sigmoid gate)")

    # --- MoE experts ---
    moe_ev = first(records, "layer_moe_output", layer=2)
    expert_result = run_experts(layer_idx=2, provider=provider, router_input=router_input, route=route)
    moe_out = expert_result["moe_output"].view(batch_size, seq_len, hidden_size)
    all_pass &= compare(tensor_summary(moe_out), moe_ev["output"],
                        atol=args.atol, label=f"MoE experts (selected: {expert_result['selected_expert_ids']})")

    # --- Layer 2 output ---
    layer_ev = first(records, "layer_output", layer=2)
    layer_out = hidden_after_attn + moe_out
    all_pass &= compare(tensor_summary(layer_out), layer_ev["hidden"],
                        atol=args.atol, label="Layer 2 output (attn+residual+MoE+residual)")

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS ✅' if all_pass else 'SOME FAILED ❌'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())