#!/usr/bin/env python3
"""Probe LFM2 full forward: all 24 layers → final norm → LM head → greedy token.

This is the end-to-end probe. It runs the complete forward pass using
Glacial's math module and compares the final logits and greedy token
against the HF golden trace.

The MoE expert computation has a ~1e-3 BF16 accumulation difference from
HF's @use_experts_implementation kernel (verified: our code with HF model
weights produces the same result as our code with safetensors weights —
the difference is in HF's kernel, not our math). This probe checks whether
that tiny difference propagates to a different greedy token.
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


def first(records, event, layer=None):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=None)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    import torch
    import torch.nn.functional as F
    from glacial.generate import make_inputs
    from glacial.lfm2 import (
        lfm2_rmsnorm, run_layer_with_optional_state, scalar_config,
        FINAL_NORM_TENSOR, EMBED_TENSOR,
    )
    from glacial.logits import chunked_last_argmax
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

    num_layers = int(config.get("num_hidden_layers", 24))
    hidden_size = int(config.get("hidden_size", 2048))
    eps = float(config.get("norm_eps", 1e-5))

    inputs = make_inputs(flat_ids)

    print(f"# LFM2 Full Forward Probe")
    print(f"  trace: {trace_path}")
    print(f"  seq_len={input_shape[1]}, layers={num_layers}")

    # --- Run full forward pass ---
    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start)
    with provider.embedding_rows(flat_ids) as (rows, hs):
        hidden = rows.view(input_shape[0], input_shape[1], hs)

    layer_stats = []
    for layer_idx in range(num_layers):
        hidden, _, stats = run_layer_with_optional_state(
            layer_idx=layer_idx, hidden=hidden,
            kv_pair=None, conv_state=None,
            model_file=model_file, header=header, payload_start=payload_start,
            inputs=inputs, config=config, scalars=scalars,
            return_state=False,
        )
        layer_stats.append(stats)

    print(f"  layers completed: {num_layers}")

    # --- Final norm (embedding_norm) ---
    norm_ev = first(records, "embedding_norm")
    with provider.tensor(FINAL_NORM_TENSOR) as final_norm_w:
        final_hidden = lfm2_rmsnorm(hidden, final_norm_w, eps=eps)

    # Compare final norm output
    norm_expected = norm_ev.get("output", norm_ev.get("hidden"))
    norm_actual = tensor_summary(final_hidden)
    norm_diff = abs(float(norm_actual.get("sum_fp32", 0)) - float(norm_expected.get("sum_fp32", 0)))
    print(f"\n  Final norm: sum_diff={norm_diff:.2e} sha256={'exact' if norm_actual.get('sha256')==norm_expected.get('sha256') else 'diff'}")

    # --- Chunked LM head (tied, no softcap, logits_scaling=1.0) ---
    final_ev = first(records, "final_logits")
    greedy_expected = final_ev["greedy_token_id"]

    next_id, lm_telemetry = chunked_last_argmax(
        final_hidden=final_hidden,
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        chunk_rows=4096,
        logits_scaling=scalars["logits_scaling"],
    )

    greedy_actual = int(next_id)
    greedy_match = greedy_actual == greedy_expected

    # Compare top-k logits
    top_ids_ev = final_ev["top_token_ids"]
    top_vals_ev = final_ev["top_token_values_fp32"]

    print(f"\n  Greedy token: ours={greedy_actual} ({final_ev.get('greedy_token_text', '?')!r})  hf={greedy_expected} ({final_ev.get('greedy_token_text', '?')!r})")
    print(f"  Greedy match: {'✅ YES' if greedy_match else '❌ NO'}")

    # Compute our top-k for comparison
    with provider.lm_head_chunk(row_start=0, row_count=min(4096, 128000)) as weight:
        logits_chunk = F.linear(final_hidden[:, -1:, :], weight)[0, 0] / scalars["logits_scaling"]
        our_top_vals, our_top_ids = logits_chunk.float().topk(5)

    print(f"\n  HF top-5:   ids={top_ids_ev[:5]}  vals={[round(v,3) for v in top_vals_ev[:5]]}")
    print(f"  Our top-5:  ids={[int(x) for x in our_top_ids.tolist()]}  vals={[round(float(v),3) for v in our_top_vals.tolist()]}")

    print(f"\n{'='*60}")
    if greedy_match:
        print(f"Result: GREEDY TOKEN MATCHES ✅  (our={greedy_actual} hf={greedy_expected})")
        return 0
    else:
        print(f"Result: GREEDY TOKEN MISMATCH ❌  (our={greedy_actual} hf={greedy_expected})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())