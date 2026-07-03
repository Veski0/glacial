#!/usr/bin/env python3
"""Probe LFM2 layer 0: embedding → RMSNorm → short conv → dense MLP → layer output.

This is the first multi-operation probe for LFM2. It verifies that Glacial's
math module (glacial/lfm2.py) produces the same intermediate tensors as the HF
golden trace, operation by operation:

  1. Embedding (no multiplier)
  2. operator_norm (RMSNorm, eps=1e-5)
  3. Short conv (gated depthwise conv1d: in_proj → B*x → conv → C*y → out_proj)
  4. Conv + residual
  5. ffn_norm (RMSNorm)
  6. Dense MLP (SwiGLU: silu(w1*x)*w3*x → w2)
  7. MLP + residual = layer output

Layer 0 is a conv + dense MLP layer (the simplest layer type).
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
        raise SystemExit("No LFM2 traces found. Run hf_trace_lfm2.py first.")
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
        "shape": list(t.shape),
        "dtype": str(t.dtype).removeprefix("torch."),
        "numel": int(t.numel()),
    }
    if t.numel() == 0:
        return out
    tf = t.float()
    out.update({
        "sum_fp32": float(tf.sum().item()),
        "mean_fp32": float(tf.mean().item()),
        "min_fp32": float(tf.min().item()),
        "max_fp32": float(tf.max().item()),
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
        a = float(actual.get(key, 0))
        e = float(expected.get(key, 0))
        d = abs(a - e)
        diffs[key] = d
        if d > atol:
            numeric_ok = False

    passed = shape_ok and dtype_ok and numeric_ok
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status} — {label}")
    print(f"    shape: {actual['shape']} vs {expected['shape']} ({'OK' if shape_ok else 'MISMATCH'})")
    print(f"    dtype: {actual['dtype']} vs {expected['dtype']} ({'OK' if dtype_ok else 'MISMATCH'})")
    print(f"    sha256: {'exact' if sha_ok else 'diff'}")
    for k, v in diffs.items():
        print(f"    {k}: abs_diff={v:.2e}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=None)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--atol", type=float, default=1e-3, help="Tolerance for BF16 comparisons")
    return parser.parse_args()


def main() -> int:
    import torch
    import torch.nn.functional as F
    from glacial.lfm2 import lfm2_rmsnorm, run_short_conv, run_dense_mlp, scalar_config, layer_tensor
    from glacial.weights import SafetensorsWeights

    args = parse_args()
    trace_path = Path(args.trace) if args.trace else latest_trace()
    records = load_trace(trace_path)
    metadata = first(records, "trace_metadata")
    inputs_ev = first(records, "inputs")
    config = metadata["config"]

    # Resolve model file
    model_file = args.model_file
    if model_file is None:
        from huggingface_hub import hf_hub_download
        model_file = hf_hub_download(repo_id=metadata["model_id"], filename="model.safetensors", local_files_only=True)
    model_file = Path(model_file)

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    scalars = scalar_config(config)

    # Get input IDs
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

    print(f"# LFM2 Layer 0 Probe")
    print(f"  trace: {trace_path}")
    print(f"  model: {model_file}")
    print(f"  prompt tokens: {flat_ids} (seq_len={seq_len})")
    print(f"  hidden_size: {hidden_size}, eps: {eps}")

    all_pass = True

    # --- 1. Embedding ---
    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start)
    with provider.embedding_rows(flat_ids) as (rows, hs):
        hidden = rows.view(input_shape[0], input_shape[1], hs)  # no multiplier

    emb_ev = first(records, "embedding_output")
    emb_expected = emb_ev.get("output", emb_ev.get("hidden"))
    all_pass &= compare(tensor_summary(hidden), emb_expected, atol=args.atol, label="1. Embedding")

    # --- 2. operator_norm (RMSNorm) ---
    op_norm_ev = first(records, "layer_operator_norm", layer=0)
    with provider.tensor(layer_tensor(0, "operator_norm.weight")) as norm_weight:
        hidden_norm = lfm2_rmsnorm(hidden, norm_weight, eps=eps)

    all_pass &= compare(
        tensor_summary(hidden_norm), op_norm_ev["output"],
        atol=args.atol, label="2. operator_norm (RMSNorm)"
    )

    # --- 3. Short conv ---
    conv_ev = first(records, "layer_conv_output", layer=0)
    conv_out, conv_state = run_short_conv(
        layer_idx=0, hidden=hidden_norm,
        model_file=model_file, header=header, payload_start=payload_start,
    )
    all_pass &= compare(
        tensor_summary(conv_out), conv_ev["output"],
        atol=args.atol, label="3. Short conv (gated depthwise conv1d)"
    )

    # --- 4. Conv + residual ---
    hidden_after_conv = hidden + conv_out  # no multiplier

    # --- 5. ffn_norm (RMSNorm) ---
    ffn_norm_ev = first(records, "layer_ffn_norm", layer=0)
    with provider.tensor(layer_tensor(0, "ffn_norm.weight")) as ffn_weight:
        hidden_for_mlp = lfm2_rmsnorm(hidden_after_conv, ffn_weight, eps=eps)

    all_pass &= compare(
        tensor_summary(hidden_for_mlp), ffn_norm_ev["output"],
        atol=args.atol, label="5. ffn_norm (RMSNorm)"
    )

    # --- 6. Dense MLP (SwiGLU) ---
    mlp_ev = first(records, "layer_mlp_output", layer=0)
    mlp_out = run_dense_mlp(layer_idx=0, provider=provider, hidden=hidden_for_mlp)
    all_pass &= compare(
        tensor_summary(mlp_out), mlp_ev["output"],
        atol=args.atol, label="6. Dense MLP (SwiGLU w1/w3/w2)"
    )

    # --- 7. MLP + residual = layer output ---
    layer_ev = first(records, "layer_output", layer=0)
    layer_out = hidden_after_conv + mlp_out  # no multiplier
    all_pass &= compare(
        tensor_summary(layer_out), layer_ev["hidden"],
        atol=args.atol, label="7. Layer 0 output (conv+residual+MLP+residual)"
    )

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS ✅' if all_pass else 'SOME FAILED ❌'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())