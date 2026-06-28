#!/usr/bin/env python3
"""Inspect a Glacial HF golden trace JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def latest_trace() -> Path:
    traces = sorted(Path("traces").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not traces:
        raise SystemExit("No traces/*.jsonl files found. Pass a trace path explicitly.")
    return traces[-1]


def first(records: list[dict[str, Any]], event: str) -> dict[str, Any] | None:
    return next((record for record in records if record.get("event") == event), None)


def summarize(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records:
        event = record.get("event", "<missing>")
        counts[event] = counts.get(event, 0) + 1

    meta = first(records, "trace_metadata") or {}
    inputs = first(records, "inputs") or {}
    final = first(records, "final_logits") or {}
    input_norms = [record for record in records if record.get("event") == "layer_input_norm"]
    q_projs = [record for record in records if record.get("event") == "layer_attention_q_proj"]
    k_projs = [record for record in records if record.get("event") == "layer_attention_k_proj"]
    v_projs = [record for record in records if record.get("event") == "layer_attention_v_proj"]
    rotaries = [record for record in records if record.get("event") == "layer_attention_rotary"]
    attention_outputs = [record for record in records if record.get("event") == "layer_attention_output"]
    post_attention_norms = [record for record in records if record.get("event") == "layer_post_attention_norm"]
    routers = [record for record in records if record.get("event") == "layer_router"]
    moe_outputs = [record for record in records if record.get("event") == "layer_moe_output"]
    layer_outputs = [record for record in records if record.get("event") == "layer_output"]
    config = meta.get("config") or {}
    expected_layers = config.get("num_hidden_layers")

    input_norm_layers = [record.get("layer") for record in input_norms]
    q_proj_layers = [record.get("layer") for record in q_projs]
    k_proj_layers = [record.get("layer") for record in k_projs]
    v_proj_layers = [record.get("layer") for record in v_projs]
    rotary_layers = [record.get("layer") for record in rotaries]
    attention_output_layers = [record.get("layer") for record in attention_outputs]
    post_attention_norm_layers = [record.get("layer") for record in post_attention_norms]
    router_layers = [record.get("layer") for record in routers]
    moe_output_layers = [record.get("layer") for record in moe_outputs]
    layer_output_layers = [record.get("layer") for record in layer_outputs]
    missing_input_norm_layers = []
    missing_q_proj_layers = []
    missing_k_proj_layers = []
    missing_v_proj_layers = []
    missing_rotary_layers = []
    missing_attention_output_layers = []
    missing_post_attention_norm_layers = []
    missing_router_layers = []
    missing_moe_output_layers = []
    missing_output_layers = []
    if isinstance(expected_layers, int):
        expected = set(range(expected_layers))
        missing_input_norm_layers = sorted(expected - set(input_norm_layers))
        missing_q_proj_layers = sorted(expected - set(q_proj_layers))
        missing_k_proj_layers = sorted(expected - set(k_proj_layers))
        missing_v_proj_layers = sorted(expected - set(v_proj_layers))
        missing_rotary_layers = sorted(expected - set(rotary_layers))
        missing_attention_output_layers = sorted(expected - set(attention_output_layers))
        missing_post_attention_norm_layers = sorted(expected - set(post_attention_norm_layers))
        missing_router_layers = sorted(expected - set(router_layers))
        missing_moe_output_layers = sorted(expected - set(moe_output_layers))
        missing_output_layers = sorted(expected - set(layer_output_layers))

    router_table = []
    for record in routers:
        selected = record.get("selected_experts")
        # For the usual one-token trace, selected shape is [[[...]]]. Keep this
        # compact while preserving a generic fallback.
        if (
            isinstance(selected, list)
            and len(selected) == 1
            and isinstance(selected[0], list)
            and len(selected[0]) == 1
        ):
            selected_compact = selected[0][0]
        else:
            selected_compact = selected
        expert_size = record.get("expert_size") or []
        router_table.append(
            {
                "layer": record.get("layer"),
                "selected_experts": selected_compact,
                "nonzero_expert_sizes": [[idx, size] for idx, size in enumerate(expert_size) if size],
            }
        )

    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "line_count": len(records),
        "event_counts": dict(sorted(counts.items())),
        "model_id": meta.get("model_id"),
        "revision": meta.get("revision"),
        "torch_version": meta.get("torch_version"),
        "transformers_version": meta.get("transformers_version"),
        "device": meta.get("device"),
        "requested_dtype": meta.get("requested_dtype"),
        "model_param_dtype": meta.get("model_param_dtype"),
        "attn_implementation": meta.get("attn_implementation"),
        "elapsed_s": meta.get("elapsed_s"),
        "expected_layers": expected_layers,
        "input_norm_layer_count": len(input_norms),
        "q_proj_layer_count": len(q_projs),
        "k_proj_layer_count": len(k_projs),
        "v_proj_layer_count": len(v_projs),
        "rotary_layer_count": len(rotaries),
        "attention_output_layer_count": len(attention_outputs),
        "post_attention_norm_layer_count": len(post_attention_norms),
        "router_layer_count": len(routers),
        "moe_output_layer_count": len(moe_outputs),
        "layer_output_count": len(layer_outputs),
        "missing_input_norm_layers": missing_input_norm_layers,
        "missing_q_proj_layers": missing_q_proj_layers,
        "missing_k_proj_layers": missing_k_proj_layers,
        "missing_v_proj_layers": missing_v_proj_layers,
        "missing_rotary_layers": missing_rotary_layers,
        "missing_attention_output_layers": missing_attention_output_layers,
        "missing_post_attention_norm_layers": missing_post_attention_norm_layers,
        "missing_router_layers": missing_router_layers,
        "missing_moe_output_layers": missing_moe_output_layers,
        "missing_output_layers": missing_output_layers,
        "input_ids": inputs.get("input_ids"),
        "position_ids": inputs.get("position_ids"),
        "greedy_token_id": final.get("greedy_token_id"),
        "greedy_token_text": final.get("greedy_token_text"),
        "top_token_ids": final.get("top_token_ids"),
        "top_token_values_fp32": final.get("top_token_values_fp32"),
        "router_table": router_table,
    }


def print_markdown(summary: dict[str, Any], *, max_router_rows: int | None) -> None:
    print(f"# Trace summary\n")
    print(f"`{summary['path']}`\n")
    print("## Metadata\n")
    for key in [
        "model_id",
        "revision",
        "torch_version",
        "transformers_version",
        "device",
        "requested_dtype",
        "model_param_dtype",
        "attn_implementation",
        "elapsed_s",
    ]:
        print(f"- `{key}`: `{summary.get(key)}`")

    print("\n## Shape\n")
    print(f"- lines: `{summary['line_count']}`")
    print(f"- bytes: `{summary['byte_size']}`")
    print(f"- expected layers: `{summary.get('expected_layers')}`")
    print(f"- input norm layers: `{summary['input_norm_layer_count']}`")
    print(f"- q projection layers: `{summary['q_proj_layer_count']}`")
    print(f"- k projection layers: `{summary['k_proj_layer_count']}`")
    print(f"- v projection layers: `{summary['v_proj_layer_count']}`")
    print(f"- rotary layers: `{summary['rotary_layer_count']}`")
    print(f"- attention output layers: `{summary['attention_output_layer_count']}`")
    print(f"- post-attention norm layers: `{summary['post_attention_norm_layer_count']}`")
    print(f"- router layers: `{summary['router_layer_count']}`")
    print(f"- MoE output layers: `{summary['moe_output_layer_count']}`")
    print(f"- layer outputs: `{summary['layer_output_count']}`")
    print(f"- missing input norm layers: `{summary['missing_input_norm_layers']}`")
    print(f"- missing q projection layers: `{summary['missing_q_proj_layers']}`")
    print(f"- missing k projection layers: `{summary['missing_k_proj_layers']}`")
    print(f"- missing v projection layers: `{summary['missing_v_proj_layers']}`")
    print(f"- missing rotary layers: `{summary['missing_rotary_layers']}`")
    print(f"- missing attention output layers: `{summary['missing_attention_output_layers']}`")
    print(f"- missing post-attention norm layers: `{summary['missing_post_attention_norm_layers']}`")
    print(f"- missing router layers: `{summary['missing_router_layers']}`")
    print(f"- missing MoE output layers: `{summary['missing_moe_output_layers']}`")
    print(f"- missing output layers: `{summary['missing_output_layers']}`")

    print("\n## Inputs / output\n")
    print(f"- input ids: `{summary.get('input_ids')}`")
    print(f"- position ids: `{summary.get('position_ids')}`")
    print(f"- greedy token: `{summary.get('greedy_token_id')}` {summary.get('greedy_token_text')!r}")
    print(f"- top token ids: `{summary.get('top_token_ids')}`")
    print(f"- top token values fp32: `{summary.get('top_token_values_fp32')}`")

    print("\n## Event counts\n")
    for event, count in summary["event_counts"].items():
        print(f"- `{event}`: `{count}`")

    print("\n## Router selections\n")
    rows = summary["router_table"]
    if max_router_rows is not None:
        rows = rows[:max_router_rows]
    print("\n| layer | selected experts |")
    print("|---:|---|")
    for row in rows:
        print(f"| {row['layer']} | `{row['selected_experts']}` |")
    if max_router_rows is not None and len(summary["router_table"]) > max_router_rows:
        print(f"\n... {len(summary['router_table']) - max_router_rows} more router rows omitted.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, help="Trace JSONL path. Defaults to latest traces/*.jsonl")
    parser.add_argument("--json", action="store_true", help="Emit summary JSON instead of Markdown.")
    parser.add_argument("--max-router-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.trace or latest_trace()
    records = load_trace(path)
    summary = summarize(path, records)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print_markdown(summary, max_router_rows=args.max_router_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
