#!/usr/bin/env python3
"""Inspect a Glacial resumable decode checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glacial.kv import inspect_decode_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path, help="Directory containing manifest.json")
    parser.add_argument("--jsonl", action="store_true", help="Emit one JSON object instead of text.")
    parser.add_argument("--show-tokens", action="store_true", help="Show token records from tokens.jsonl.")
    parser.add_argument("--max-token-lines", type=int, default=40, help="Maximum token records to print with --show-tokens.")
    parser.add_argument("--decode-tokens", action="store_true", help="Decode token pieces with the checkpoint tokenizer.")
    parser.add_argument("--cache-dir", default=None, help="Tokenizer cache dir for --decode-tokens.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not hit the network for --decode-tokens.")
    parser.add_argument("--no-validate-kv", action="store_true", help="Skip safetensors KV metadata validation.")
    return parser.parse_args()


def _load_tokenizer(summary: dict[str, Any], args: argparse.Namespace):
    if not args.decode_tokens:
        return None
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("--decode-tokens requires transformers") from exc

    model = summary["manifest"]["model"]
    return AutoTokenizer.from_pretrained(
        model["model_id"],
        revision=model["revision"],
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )


def _piece(tokenizer, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _jsonl_summary(summary: dict[str, Any], tokenizer) -> dict[str, Any]:
    manifest = summary["manifest"]
    tokens = summary["tokens"]
    last = tokens[-1] if tokens else None
    return {
        "kind": "glacial_checkpoint_inspection",
        "valid": summary["valid"],
        "validation_errors": summary["validation_errors"],
        "run_dir": summary["run_dir"],
        "manifest_path": summary["manifest_path"],
        "checkpoint_kind": manifest["checkpoint_kind"],
        "schema_version": manifest["schema_version"],
        "run_id": manifest["run_id"],
        "backend": manifest.get("backend"),
        "model": manifest["model"],
        "created_at": manifest["created_at"],
        "updated_at": manifest["updated_at"],
        "state": manifest["state"],
        "snapshot": manifest["snapshot"],
        "snapshot_count": summary["snapshot_count"],
        "tmp_snapshot_count": summary["tmp_snapshot_count"],
        "tokens_path": manifest["tokens_path"],
        "kv_layer_count": len(summary["kv_layers"]),
        "total_kv_bytes": summary["total_kv_bytes"],
        "last_token": None
        if last is None
        else {
            "index": last["index"],
            "id": last["id"],
            "source": last["source"],
            "piece": _piece(tokenizer, last["id"]),
        },
    }


def _print_text(summary: dict[str, Any], tokenizer, *, show_tokens: bool, max_token_lines: int) -> None:
    manifest = summary["manifest"]
    state = manifest["state"]
    model = manifest["model"]
    prompt = manifest.get("prompt", {})
    sampler = manifest.get("sampler", {})
    tokens = summary["tokens"]
    last = tokens[-1] if tokens else None
    first_layer = summary["kv_layers"][0] if summary["kv_layers"] else None
    last_layer = summary["kv_layers"][-1] if summary["kv_layers"] else None

    print(f"# Glacial checkpoint: {summary['run_dir']}")
    print()
    print(f"- valid: {summary['valid']}")
    print(f"- kind: {manifest['checkpoint_kind']} schema={manifest['schema_version']}")
    print(f"- run_id: {manifest['run_id']}")
    print(f"- backend: {manifest.get('backend') or 'unknown'}")
    print(f"- model: {model['model_id']} @ {model['revision']}")
    print(f"- created: {manifest['created_at']}")
    print(f"- updated: {manifest['updated_at']}")
    print(f"- prompt mode: {prompt.get('mode')}")
    print(f"- sampler: {sampler.get('type')} lm_head_chunk_rows={sampler.get('lm_head_chunk_rows')}")
    print()
    print("## State")
    print()
    print(f"- token_count: {state['token_count']}")
    print(f"- prompt_token_count: {state['prompt_token_count']}")
    print(f"- generated_token_count: {state['generated_token_count']}")
    print(f"- kv_length: {state['kv_length']}")
    print(f"- next_decode_position: {state['next_decode_position']}")
    if last is not None:
        piece = _piece(tokenizer, last["id"])
        suffix = f" {piece!r}" if piece is not None else ""
        print(f"- last_token: index={last['index']} id={last['id']} source={last['source']}{suffix}")
    print()
    print("## Files")
    print()
    print(f"- manifest: {summary['manifest_path']}")
    print(f"- snapshot: {manifest['snapshot']} exists={summary['snapshot_exists']}")
    print(f"- tokens: {manifest['tokens_path']}")
    print(f"- snapshots retained: {summary['snapshot_count']}")
    if summary["tmp_snapshot_count"]:
        print(f"- temporary snapshots present: {summary['tmp_snapshot_count']}")
    print()
    print("## KV")
    print()
    print(f"- layers: {len(summary['kv_layers'])}")
    print(f"- total_kv_bytes: {summary['total_kv_bytes']}")
    if first_layer is not None:
        print(f"- first layer: {first_layer['path']} key={first_layer['key_shape']} value={first_layer['value_shape']}")
    if last_layer is not None and last_layer is not first_layer:
        print(f"- last layer: {last_layer['path']} key={last_layer['key_shape']} value={last_layer['value_shape']}")

    if summary["validation_errors"]:
        print()
        print("## Validation errors")
        print()
        for error in summary["validation_errors"]:
            print(f"- {error}")

    if show_tokens:
        print()
        print("## Tokens")
        print()
        shown = tokens[:max_token_lines]
        for record in shown:
            piece = _piece(tokenizer, record["id"])
            suffix = f" piece={piece!r}" if piece is not None else ""
            print(f"- {record['index']}: id={record['id']} source={record['source']}{suffix}")
        if len(tokens) > len(shown):
            print(f"- ... {len(tokens) - len(shown)} more token(s) omitted")


def main() -> int:
    args = parse_args()
    summary = inspect_decode_checkpoint(args.checkpoint_dir, validate_kv=not args.no_validate_kv)
    tokenizer = _load_tokenizer(summary, args)

    if args.jsonl:
        print(json.dumps(_jsonl_summary(summary, tokenizer), sort_keys=True))
    else:
        _print_text(summary, tokenizer, show_tokens=args.show_tokens, max_token_lines=args.max_token_lines)
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
