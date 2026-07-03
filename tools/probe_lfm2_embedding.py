#!/usr/bin/env python3
"""Probe LFM2 embedding: verify safetensors row loading matches HF.

This does not instantiate a Hugging Face model. It reads only selected rows
from `model.embed_tokens.weight` in a local safetensors file and compares
the result against an HF golden trace produced by `hf_trace_lfm2.py`.

Unlike Granite, LFM2 has no embedding multiplier — the embedding is a raw
nn.Embedding lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

EMBED_TENSOR = "model.embed_tokens.weight"
BF16_BYTES = 2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        print(f"Missing Python dependency: {exc.name}", file=sys.stderr)
        raise SystemExit(2) from exc


def load_trace(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def first(records: list[dict[str, Any]], event: str) -> dict[str, Any]:
    for record in records:
        if record.get("event") == event:
            return record
    raise SystemExit(f"Trace is missing event {event!r}")


def latest_trace() -> Path:
    traces = sorted(Path("traces").glob("*LFM2*"), key=lambda p: p.stat().st_mtime)
    if not traces:
        raise SystemExit("No LFM2 traces found. Run hf_trace_lfm2.py first.")
    return traces[-1]


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        header_len_raw = f.read(8)
        if len(header_len_raw) != 8:
            raise SystemExit(f"{path}: file too small for safetensors header")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise SystemExit(f"{path}: truncated safetensors header")
    return header_len, json.loads(header_raw)


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


def compare_summaries(actual: dict[str, Any], expected: dict[str, Any], *, atol: float) -> dict[str, Any]:
    shape_pass = actual["shape"] == expected["shape"]
    dtype_pass = actual["dtype"] == expected["dtype"]
    sha256_pass = actual.get("sha256") == expected.get("sha256")

    numeric = {}
    for key in ("sum_fp32", "mean_fp32", "min_fp32", "max_fp32"):
        a = float(actual.get(key, 0))
        e = float(expected.get(key, 0))
        diff = abs(a - e)
        numeric[key] = {"actual": a, "expected": e, "abs_diff": diff, "pass": diff <= atol}

    all_pass = shape_pass and dtype_pass and all(row["pass"] for row in numeric.values())
    return {
        "pass": all_pass,
        "shape_pass": shape_pass,
        "dtype_pass": dtype_pass,
        "sha256_pass": sha256_pass,
        "actual_sha256": actual.get("sha256"),
        "expected_sha256": expected.get("sha256"),
        "numeric": numeric,
    }


def read_embedding_rows(path: Path, rows: list[int], *, tensor_meta: dict[str, Any], payload_start: int):
    import torch
    dtype = tensor_meta["dtype"]
    shape = [int(x) for x in tensor_meta["shape"]]
    data_offsets = [int(x) for x in tensor_meta["data_offsets"]]
    if dtype != "BF16":
        raise SystemExit(f"Expected {EMBED_TENSOR} dtype BF16, got {dtype}")
    if len(shape) != 2:
        raise SystemExit(f"Expected {EMBED_TENSOR} rank 2, got shape {shape}")

    vocab_size, hidden_size = shape
    row_bytes = hidden_size * BF16_BYTES

    with path.open("rb") as f:
        result_rows = []
        for token_id in rows:
            if token_id < 0 or token_id >= vocab_size:
                raise SystemExit(f"Token id {token_id} out of vocab range [0, {vocab_size})")
            offset = payload_start + data_offsets[0] + token_id * row_bytes
            f.seek(offset)
            raw = f.read(row_bytes)
            if len(raw) != row_bytes:
                raise SystemExit(f"Truncated read for token {token_id}: got {len(raw)} bytes, expected {row_bytes}")
            row = torch.frombuffer(raw, dtype=torch.uint8).view(torch.bfloat16)
            result_rows.append(row)

    return torch.stack(result_rows), hidden_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=None, help="HF golden trace JSONL. Defaults to latest traces/*LFM2*")
    parser.add_argument("--model-file", default=None, help="Local model.safetensors path.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for FP32 comparisons.")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_torch()
    import torch

    trace_path = Path(args.trace) if args.trace else latest_trace()
    records = load_trace(trace_path)
    metadata = first(records, "trace_metadata")
    inputs = first(records, "inputs")
    embedding_output = first(records, "embedding_output")

    # Resolve model file
    model_file = args.model_file
    if model_file is None:
        from huggingface_hub import hf_hub_download
        model_file = hf_hub_download(
            repo_id=metadata["model_id"],
            filename="model.safetensors",
            local_files_only=True,
        )
    model_file = Path(model_file)
    if not model_file.exists():
        raise SystemExit(f"model.safetensors not found: {model_file}")

    input_ids = inputs["input_ids"]
    if isinstance(input_ids[0], list):
        # Flatten rank-2
        flat_ids = []
        for row in input_ids:
            flat_ids.extend(int(x) for x in row)
        input_shape = [len(input_ids), len(input_ids[0])]
    else:
        flat_ids = [int(x) for x in input_ids]
        input_shape = [1, len(input_ids)]

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    tensor_meta = header.get(EMBED_TENSOR)
    if tensor_meta is None:
        raise SystemExit(f"{model_file}: missing tensor {EMBED_TENSOR!r}")

    rows, hidden_size = read_embedding_rows(
        model_file, flat_ids, tensor_meta=tensor_meta, payload_start=payload_start
    )

    # LFM2: no embedding multiplier — raw embedding lookup
    hidden = rows.view(input_shape[0], input_shape[1], hidden_size)

    actual = tensor_summary(hidden)
    expected = embedding_output["output"] if "output" in embedding_output else embedding_output["hidden"]
    comparison = compare_summaries(actual, expected, atol=args.atol)

    if args.json:
        result = {
            "trace": str(trace_path),
            "model_file": str(model_file),
            "input_ids": input_ids,
            "loaded_weight_bytes": len(flat_ids) * hidden_size * BF16_BYTES,
            "actual": actual,
            "expected": expected,
            "comparison": comparison,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("# LFM2 Embedding Probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"input ids: {input_ids}")
        print(f"loaded weight bytes: {len(flat_ids) * hidden_size * BF16_BYTES}")
        print(f"\n## Comparison\n")
        print(f"pass: `{comparison['pass']}`")
        print(f"shape pass: `{comparison['shape_pass']}`")
        print(f"dtype pass: `{comparison['dtype_pass']}`")
        print(f"sha256 pass: `{comparison['sha256_pass']}`")
        print(f"actual sha256: `{comparison['actual_sha256']}`")
        print(f"expected sha256: `{comparison['expected_sha256']}`")
        print(f"\n| metric | actual | expected | abs diff | pass |")
        print("|---|---:|---:|---:|---:|")
        for key, row in comparison["numeric"].items():
            print(f"| `{key}` | {row['actual']} | {row['expected']} | {row['abs_diff']} | `{row['pass']}` |")

    return 0 if comparison["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())