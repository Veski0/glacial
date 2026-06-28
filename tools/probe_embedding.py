#!/usr/bin/env python3
"""Probe Glacial's first executable slice: embedding row loading.

This does not instantiate a Hugging Face model. It reads only selected rows from
`model.embed_tokens.weight` in a local safetensors file, applies Granite's
embedding multiplier, and compares the result against an HF golden trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

EMBED_TENSOR = "model.embed_tokens.weight"
BF16_BYTES = 2


def require_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        print(
            "Missing Python dependency: torch\n\n"
            "Install trace/probe dependencies, for example:\n\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -r requirements.txt\n",
            file=sys.stderr,
        )
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
    traces = sorted(Path("traces").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not traces:
        raise SystemExit("No traces/*.jsonl files found. Pass --trace explicitly.")
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
    try:
        header = json.loads(header_raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid safetensors JSON header: {exc}") from exc
    return header_len, header


def flatten_ids(nested: Any) -> tuple[list[int], list[int]]:
    """Flatten batch/sequence token IDs and return original [batch, seq] shape."""
    if not isinstance(nested, list) or not nested or not isinstance(nested[0], list):
        raise SystemExit("Expected trace input_ids to be a non-empty rank-2 list")
    batch = len(nested)
    seq = len(nested[0])
    out: list[int] = []
    for row in nested:
        if len(row) != seq:
            raise SystemExit("Ragged trace input_ids are not supported")
        out.extend(int(x) for x in row)
    return out, [batch, seq]


def raw_sha256(tensor) -> str:
    import torch

    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        data = t.view(torch.int16).numpy().tobytes()
    else:
        data = t.numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def tensor_summary(tensor) -> dict[str, Any]:
    import torch

    t = tensor.detach()
    tf = t.float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).removeprefix("torch."),
        "numel": int(t.numel()),
        "sha256": raw_sha256(t),
        "sum_fp32": float(tf.sum().item()),
        "mean_fp32": float(tf.mean().item()),
        "min_fp32": float(tf.min().item()),
        "max_fp32": float(tf.max().item()),
        "l2_fp32": float(torch.linalg.vector_norm(tf.reshape(-1), ord=2).item()),
    }


def read_embedding_rows(path: Path, rows: list[int], *, tensor_meta: dict[str, Any], payload_start: int):
    import torch

    dtype = tensor_meta["dtype"]
    shape = tensor_meta["shape"]
    data_offsets = tensor_meta["data_offsets"]
    if dtype != "BF16":
        raise SystemExit(f"Expected {EMBED_TENSOR} dtype BF16, got {dtype}")
    if len(shape) != 2:
        raise SystemExit(f"Expected {EMBED_TENSOR} rank 2, got shape {shape}")

    vocab_size, hidden_size = int(shape[0]), int(shape[1])
    row_bytes = hidden_size * BF16_BYTES
    tensor_payload_start = payload_start + int(data_offsets[0])
    tensor_payload_end = payload_start + int(data_offsets[1])

    loaded = []
    with path.open("rb") as f:
        for token_id in rows:
            if token_id < 0 or token_id >= vocab_size:
                raise SystemExit(f"Token id {token_id} outside embedding vocab size {vocab_size}")
            offset = tensor_payload_start + token_id * row_bytes
            if offset + row_bytes > tensor_payload_end:
                raise SystemExit(f"Computed row range exceeds tensor range for token id {token_id}")
            f.seek(offset)
            raw = f.read(row_bytes)
            if len(raw) != row_bytes:
                raise SystemExit(f"Short read for token id {token_id}")
            # `torch.frombuffer(bytes(...))` warns because bytes are immutable.
            # `bytearray` gives PyTorch a writable view; clone detaches the row
            # from the temporary bytearray before the next loop iteration.
            row = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()
            loaded.append(row)

    return torch.stack(loaded, dim=0), hidden_size


def compare_summaries(actual: dict[str, Any], expected: dict[str, Any], *, atol: float) -> dict[str, Any]:
    numeric_keys = ["sum_fp32", "mean_fp32", "min_fp32", "max_fp32", "l2_fp32"]
    numeric = {}
    ok = True
    for key in numeric_keys:
        av = actual.get(key)
        ev = expected.get(key)
        diff = None if av is None or ev is None else abs(float(av) - float(ev))
        passed = diff is not None and diff <= atol
        numeric[key] = {"actual": av, "expected": ev, "abs_diff": diff, "pass": passed}
        ok = ok and passed

    shape_pass = actual.get("shape") == expected.get("shape")
    dtype_pass = actual.get("dtype") == expected.get("dtype")
    sha_pass = actual.get("sha256") == expected.get("sha256")

    return {
        "pass": bool(shape_pass and dtype_pass and ok),
        "shape_pass": shape_pass,
        "dtype_pass": dtype_pass,
        "sha256_pass": sha_pass,
        "numeric_atol": atol,
        "numeric": numeric,
        "actual_sha256": actual.get("sha256"),
        "expected_sha256": expected.get("sha256"),
    }


def find_default_model_file() -> Path | None:
    candidates = [
        Path("models/ibm-granite/granite-3.1-1b-a400m-instruct/model.safetensors"),
        Path("models/granite-3.1-1b-a400m-instruct/model.safetensors"),
        Path("model.safetensors"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_model_file(args: argparse.Namespace, metadata: dict[str, Any]) -> Path:
    if args.model_file is not None:
        if not args.model_file.exists():
            raise SystemExit(f"Model file does not exist: {args.model_file}")
        return args.model_file

    local_candidate = find_default_model_file()
    if local_candidate is not None:
        return local_candidate

    model_id = args.model_id or metadata.get("model_id")
    revision = args.revision or metadata.get("revision")
    if not model_id:
        raise SystemExit("Could not infer model_id from trace. Pass --model-file or --model-id.")

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not locate a local model.safetensors and huggingface_hub is not installed.\n"
            "Pass --model-file PATH or install requirements.txt."
        ) from exc

    try:
        return Path(
            hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors",
                revision=revision,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
            )
        )
    except Exception as exc:
        raise SystemExit(
            "Could not resolve model.safetensors from Hugging Face cache/hub.\n"
            "Pass --model-file PATH if you have it elsewhere.\n"
            f"model_id={model_id!r} revision={revision!r} local_files_only={args.local_files_only}\n"
            f"error: {exc}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=None, help="HF golden trace JSONL. Defaults to latest traces/*.jsonl")
    parser.add_argument("--model-file", type=Path, default=None, help="Local model.safetensors path")
    parser.add_argument("--model-id", default=None, help="HF model id fallback. Defaults to trace metadata.")
    parser.add_argument("--revision", default=None, help="HF revision fallback. Defaults to trace metadata.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--embedding-multiplier", type=float, default=None, help="Override config embedding_multiplier")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_torch()
    import torch

    trace_path = args.trace or latest_trace()
    records = load_trace(trace_path)
    metadata = first(records, "trace_metadata")
    model_file = resolve_model_file(args, metadata)
    inputs = first(records, "inputs")
    embedding_output = first(records, "embedding_output")

    input_ids, input_shape = flatten_ids(inputs["input_ids"])
    config = metadata.get("config") or {}
    multiplier = args.embedding_multiplier
    if multiplier is None:
        multiplier = float(config.get("embedding_multiplier", 12.0))

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    tensor_meta = header.get(EMBED_TENSOR)
    if tensor_meta is None:
        raise SystemExit(f"{model_file}: missing tensor {EMBED_TENSOR!r}")

    rows, hidden_size = read_embedding_rows(model_file, input_ids, tensor_meta=tensor_meta, payload_start=payload_start)
    hidden = rows.view(input_shape[0], input_shape[1], hidden_size) * multiplier
    # Match HF's embedding output dtype behavior: BF16 embeddings multiplied by
    # a Python float remain BF16 in PyTorch module execution.
    hidden = hidden.to(torch.bfloat16)

    actual = tensor_summary(hidden)
    expected = embedding_output["hidden"]
    comparison = compare_summaries(actual, expected, atol=args.atol)

    result = {
        "trace": str(trace_path),
        "model_file": str(model_file),
        "tensor": EMBED_TENSOR,
        "input_ids": inputs["input_ids"],
        "embedding_multiplier": multiplier,
        "loaded_weight_bytes": len(input_ids) * hidden_size * BF16_BYTES,
        "actual": actual,
        "expected": expected,
        "comparison": comparison,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("# Embedding probe\n")
        print(f"trace: `{trace_path}`")
        print(f"model file: `{model_file}`")
        print(f"input ids: `{inputs['input_ids']}`")
        print(f"embedding multiplier: `{multiplier}`")
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
