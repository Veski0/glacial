"""Persistent KV-cache checkpoints for resumable Glacial decode."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

BF16_BYTES = 2
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "glacial.greedy_kv.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync for directory entries after atomic renames."""

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as f:
        os.fsync(f.fileno())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _read_manifest_if_present(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "architectures",
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_local_experts",
        "num_experts_per_tok",
        "intermediate_size",
        "rms_norm_eps",
        "rope_theta",
        "embedding_multiplier",
        "attention_multiplier",
        "residual_multiplier",
        "logits_scaling",
        "vocab_size",
    ]
    return {key: config[key] for key in keys if key in config}


def _token_jsonl(token_ids: list[int], *, prompt_token_count: int) -> str:
    lines = []
    for index, token_id in enumerate(token_ids):
        source = "prompt" if index < prompt_token_count else "generated"
        lines.append(json.dumps({"index": index, "id": int(token_id), "source": source}, sort_keys=True))
    return "\n".join(lines) + "\n"


def _parse_token_jsonl(path: Path) -> list[int]:
    token_ids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index = int(record["index"])
            if index != len(token_ids):
                raise SystemExit(f"{path}: token line {line_no} has index {index}, expected {len(token_ids)}")
            token_ids.append(int(record["id"]))
    return token_ids


def _validate_kv_invariant(*, token_ids: list[int], kv_cache: list[tuple[Any, Any]], prompt_token_count: int) -> int:
    if not token_ids:
        raise SystemExit("Cannot checkpoint an empty token sequence")
    if prompt_token_count < 0 or prompt_token_count > len(token_ids):
        raise SystemExit(f"Invalid prompt_token_count {prompt_token_count} for {len(token_ids)} tokens")
    if not kv_cache:
        raise SystemExit("Cannot checkpoint without a KV cache")

    expected_kv_length = len(token_ids) - 1
    for layer_idx, (key, value) in enumerate(kv_cache):
        if tuple(key.shape) != tuple(value.shape):
            raise SystemExit(f"KV layer {layer_idx}: key shape {list(key.shape)} != value shape {list(value.shape)}")
        if len(key.shape) != 4:
            raise SystemExit(f"KV layer {layer_idx}: expected rank-4 tensors, got {list(key.shape)}")
        if int(key.shape[2]) != expected_kv_length:
            raise SystemExit(
                f"KV layer {layer_idx}: cache length {int(key.shape[2])} != expected {expected_kv_length}. "
                "Glacial checkpoints expect KV to contain all tokens except token_ids[-1]."
            )
    return expected_kv_length


def save_decode_checkpoint(
    *,
    run_dir: Path,
    token_ids: list[int],
    prompt_token_count: int,
    kv_cache: list[tuple[Any, Any]],
    model_id: str,
    revision: str,
    model_file: Path,
    backend_name: str | None,
    rendered_text: str | None,
    prompt_mode: str | None,
    messages: Any | None,
    config: dict[str, Any],
    lm_head_chunk_rows: int,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a resumable decode checkpoint.

    Atomicity model: each checkpoint is written to a new immutable snapshot
    directory. ``manifest.json`` is replaced only after the snapshot is fully
    written, so resume sees either the previous complete checkpoint or the new
    complete checkpoint.

    The ``sampler`` dict (if provided) is stored in the manifest so that
    resumed generation reproduces the exact same sampled sequence.  For greedy
    checkpoints, pass ``None`` or a dict with ``type="greedy"``.
    """

    from safetensors.torch import save_file

    run_dir = Path(run_dir)
    old_manifest = _read_manifest_if_present(run_dir)
    now = _utc_now()
    run_id = old_manifest.get("run_id") if old_manifest is not None else uuid4().hex
    created_at = old_manifest.get("created_at") if old_manifest is not None else now
    kv_length = _validate_kv_invariant(token_ids=token_ids, kv_cache=kv_cache, prompt_token_count=prompt_token_count)

    snapshot_name = f"tokens_{len(token_ids):08d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
    snapshots_dir = run_dir / "snapshots"
    tmp_snapshot = snapshots_dir / f".tmp-{snapshot_name}"
    final_snapshot = snapshots_dir / snapshot_name
    if tmp_snapshot.exists():
        shutil.rmtree(tmp_snapshot)
    if final_snapshot.exists():
        raise SystemExit(f"Checkpoint snapshot already exists: {final_snapshot}")

    kv_dir = tmp_snapshot / "kv"
    kv_dir.mkdir(parents=True, exist_ok=False)
    (tmp_snapshot / "tokens.jsonl").write_text(_token_jsonl(token_ids, prompt_token_count=prompt_token_count), encoding="utf-8")
    _fsync_file(tmp_snapshot / "tokens.jsonl")

    kv_layers = []
    for layer_idx, (key, value) in enumerate(kv_cache):
        layer_rel = Path("kv") / f"layer_{layer_idx:02d}.safetensors"
        layer_path = tmp_snapshot / layer_rel
        save_file({"key": key.contiguous(), "value": value.contiguous()}, str(layer_path))
        _fsync_file(layer_path)
        kv_layers.append(
            {
                "layer": layer_idx,
                "path": str(Path("snapshots") / snapshot_name / layer_rel),
                "key_shape": [int(x) for x in key.shape],
                "value_shape": [int(x) for x in value.shape],
                "dtype": "BF16",
            }
        )

    snapshot_info = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": CHECKPOINT_KIND,
        "created_at": now,
        "token_count": len(token_ids),
        "kv_length": kv_length,
    }
    (tmp_snapshot / "snapshot.json").write_text(json.dumps(snapshot_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _fsync_file(tmp_snapshot / "snapshot.json")
    _fsync_dir(kv_dir)
    _fsync_dir(tmp_snapshot)

    snapshots_dir.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_snapshot, final_snapshot)
    _fsync_dir(snapshots_dir)

    prompt: dict[str, Any] = {
        "mode": prompt_mode,
        "rendered_text": rendered_text,
        "prompt_token_count": prompt_token_count,
    }
    if messages is not None:
        prompt["messages"] = messages

    sampler_info: dict[str, Any] = sampler or {}
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": CHECKPOINT_KIND,
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": now,
        "backend": backend_name,
        "model": {
            "model_id": model_id,
            "revision": revision,
            "model_file": str(model_file),
            "config": _config_summary(config),
        },
        "prompt": prompt,
        "sampler": {
            "type": sampler_info.get("type", "greedy"),
            "temperature": sampler_info.get("temperature"),
            "top_k": sampler_info.get("top_k"),
            "top_p": sampler_info.get("top_p"),
            "seed": sampler_info.get("seed"),
            "rng_state": sampler_info.get("rng_state"),
            "lm_head_chunk_rows": int(lm_head_chunk_rows),
        },
        "state": {
            "token_count": len(token_ids),
            "prompt_token_count": prompt_token_count,
            "generated_token_count": len(token_ids) - prompt_token_count,
            "kv_length": kv_length,
            "kv_contains_all_tokens_except_last": True,
            "last_token_id": int(token_ids[-1]),
            "next_decode_position": len(token_ids) - 1,
        },
        "snapshot": str(Path("snapshots") / snapshot_name),
        "tokens_path": str(Path("snapshots") / snapshot_name / "tokens.jsonl"),
        "kv_layers": kv_layers,
    }
    _atomic_write_json(run_dir / "manifest.json", manifest)
    return manifest


def inspect_decode_checkpoint(run_dir: Path, *, validate_kv: bool = True) -> dict[str, Any]:
    """Inspect checkpoint metadata without loading full KV tensors."""

    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"No Glacial checkpoint manifest found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit(f"Unsupported checkpoint schema version in {manifest_path}: {manifest.get('schema_version')}")
    if manifest.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise SystemExit(f"Unsupported checkpoint kind in {manifest_path}: {manifest.get('checkpoint_kind')}")

    tokens_path = run_dir / manifest["tokens_path"]
    token_records = []
    if not tokens_path.exists():
        raise SystemExit(f"Checkpoint token file missing: {tokens_path}")
    with tokens_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index = int(record["index"])
            if index != len(token_records):
                raise SystemExit(f"{tokens_path}: token line {line_no} has index {index}, expected {len(token_records)}")
            token_records.append(
                {
                    "index": index,
                    "id": int(record["id"]),
                    "source": str(record.get("source", "unknown")),
                }
            )

    state = manifest["state"]
    validation_errors: list[str] = []
    if len(token_records) != int(state["token_count"]):
        validation_errors.append(f"token file count {len(token_records)} != manifest token_count {state['token_count']}")

    kv_layers = []
    total_kv_bytes = 0
    if validate_kv:
        from safetensors import safe_open

    for layer in sorted(manifest["kv_layers"], key=lambda item: int(item["layer"])):
        layer_path = run_dir / layer["path"]
        layer_summary = {
            "layer": int(layer["layer"]),
            "path": str(layer["path"]),
            "exists": layer_path.exists(),
            "key_shape": [int(x) for x in layer["key_shape"]],
            "value_shape": [int(x) for x in layer["value_shape"]],
            "dtype": layer.get("dtype", "unknown"),
            "validated": False,
        }
        if not layer_path.exists():
            validation_errors.append(f"missing KV layer file: {layer_path}")
            kv_layers.append(layer_summary)
            continue

        key_numel = 1
        value_numel = 1
        for dim in layer_summary["key_shape"]:
            key_numel *= int(dim)
        for dim in layer_summary["value_shape"]:
            value_numel *= int(dim)
        if layer_summary["dtype"] == "BF16":
            total_kv_bytes += (key_numel + value_numel) * BF16_BYTES

        if validate_kv:
            with safe_open(str(layer_path), framework="pt", device="cpu") as f:
                keys = set(f.keys())
                if keys != {"key", "value"}:
                    validation_errors.append(f"{layer_path}: expected tensors {{'key', 'value'}}, got {sorted(keys)}")
                for tensor_name, shape_field in [("key", "key_shape"), ("value", "value_shape")]:
                    if tensor_name not in keys:
                        continue
                    tensor_slice = f.get_slice(tensor_name)
                    actual_shape = [int(x) for x in tensor_slice.get_shape()]
                    actual_dtype = tensor_slice.get_dtype()
                    if actual_shape != layer_summary[shape_field]:
                        validation_errors.append(
                            f"{layer_path}: {tensor_name} shape {actual_shape} != manifest {layer_summary[shape_field]}"
                        )
                    if actual_dtype != "BF16":
                        validation_errors.append(f"{layer_path}: {tensor_name} dtype {actual_dtype} != BF16")
            layer_summary["validated"] = True
        kv_layers.append(layer_summary)

    snapshot_path = run_dir / manifest["snapshot"]
    snapshots_dir = run_dir / "snapshots"
    snapshot_count = len([p for p in snapshots_dir.iterdir() if p.is_dir() and not p.name.startswith(".tmp-")]) if snapshots_dir.exists() else 0
    tmp_snapshot_count = len([p for p in snapshots_dir.iterdir() if p.is_dir() and p.name.startswith(".tmp-")]) if snapshots_dir.exists() else 0

    return {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "tokens_path": str(tokens_path),
        "tokens": token_records,
        "kv_layers": kv_layers,
        "total_kv_bytes": total_kv_bytes,
        "snapshot_path": str(snapshot_path),
        "snapshot_exists": snapshot_path.exists(),
        "snapshot_count": snapshot_count,
        "tmp_snapshot_count": tmp_snapshot_count,
        "valid": not validation_errors and snapshot_path.exists(),
        "validation_errors": validation_errors + ([] if snapshot_path.exists() else [f"snapshot directory missing: {snapshot_path}"]),
    }


def load_decode_checkpoint(run_dir: Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load the latest complete checkpoint pointed to by ``manifest.json``."""

    import torch
    from safetensors.torch import load_file

    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"No Glacial checkpoint manifest found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit(f"Unsupported checkpoint schema version in {manifest_path}: {manifest.get('schema_version')}")
    if manifest.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise SystemExit(f"Unsupported checkpoint kind in {manifest_path}: {manifest.get('checkpoint_kind')}")

    tokens_path = run_dir / manifest["tokens_path"]
    token_ids = _parse_token_jsonl(tokens_path)
    state = manifest["state"]
    if len(token_ids) != int(state["token_count"]):
        raise SystemExit(f"{tokens_path}: token count {len(token_ids)} != manifest token_count {state['token_count']}")

    kv_cache = []
    for layer in sorted(manifest["kv_layers"], key=lambda item: int(item["layer"])):
        layer_path = run_dir / layer["path"]
        tensors = load_file(str(layer_path), device=device)
        if "key" not in tensors or "value" not in tensors:
            raise SystemExit(f"{layer_path}: expected tensors named 'key' and 'value'")
        key = tensors["key"].contiguous()
        value = tensors["value"].contiguous()
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise SystemExit(f"{layer_path}: expected BF16 key/value tensors, got {key.dtype}/{value.dtype}")
        if [int(x) for x in key.shape] != [int(x) for x in layer["key_shape"]]:
            raise SystemExit(f"{layer_path}: key shape {list(key.shape)} != manifest {layer['key_shape']}")
        if [int(x) for x in value.shape] != [int(x) for x in layer["value_shape"]]:
            raise SystemExit(f"{layer_path}: value shape {list(value.shape)} != manifest {layer['value_shape']}")
        kv_cache.append((key, value))

    prompt_token_count = int(state["prompt_token_count"])
    kv_length = _validate_kv_invariant(token_ids=token_ids, kv_cache=kv_cache, prompt_token_count=prompt_token_count)
    if kv_length != int(state["kv_length"]):
        raise SystemExit(f"Loaded KV length {kv_length} != manifest kv_length {state['kv_length']}")

    return {"manifest": manifest, "token_ids": token_ids, "kv_cache": kv_cache}
