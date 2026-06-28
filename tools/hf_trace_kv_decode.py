#!/usr/bin/env python3
"""Emit an HF golden trace for one cached decode step.

This intentionally instantiates the full HF model. It records:

  prompt prefill -> greedy first token -> cached decode of that token

Output format: JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from hf_trace_granite import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    build_text,
    clean_dtype_name,
    require_deps,
    resolve_device,
    resolve_dtype,
    selected_config,
    tensor_summary,
    write_event,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", default=None, help="Output JSONL path. Defaults to traces/<timestamp>.jsonl")
    parser.add_argument("--prompt", default="Hello", help="Raw prompt text when not using chat options.")
    parser.add_argument("--chat-user", default=None, help="Render a single user message with the model chat template.")
    parser.add_argument("--messages-json", default=None, help="JSON array of chat messages to render with chat template.")
    parser.add_argument("--add-generation-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16, float16, or float32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--logit-top-k", type=int, default=20)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def cache_summary(past_key_values) -> dict[str, Any]:
    # DynamicCache in transformers exposes key_cache/value_cache lists. Legacy
    # cache exposes tuple layers. Record shapes only; do not serialize values.
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
        return {
            "format": type(past_key_values).__name__,
            "layers": len(key_cache),
            "key_shapes": [list(t.shape) for t in key_cache],
            "value_shapes": [list(t.shape) for t in value_cache],
            "key_dtypes": [clean_dtype_name(t.dtype) for t in key_cache],
            "value_dtypes": [clean_dtype_name(t.dtype) for t in value_cache],
        }
    if isinstance(past_key_values, tuple):
        return {
            "format": "legacy_tuple",
            "layers": len(past_key_values),
            "key_shapes": [list(layer[0].shape) for layer in past_key_values],
            "value_shapes": [list(layer[1].shape) for layer in past_key_values],
            "key_dtypes": [clean_dtype_name(layer[0].dtype) for layer in past_key_values],
            "value_dtypes": [clean_dtype_name(layer[1].dtype) for layer in past_key_values],
        }
    return {"format": type(past_key_values).__name__}


def topk_payload(logits, *, top_k: int, tokenizer) -> dict[str, Any]:
    import torch

    logits_fp32 = logits.float()
    values, ids = logits_fp32.topk(top_k)
    # Greedy decode means argmax. Do not use topk()[0] for greedy: topk tie
    # ordering can differ from argmax tie ordering.
    greedy_id = int(torch.argmax(logits_fp32).item())
    return {
        "top_k": top_k,
        "top_token_ids": [int(x) for x in ids.detach().cpu().tolist()],
        "top_token_values_fp32": [float(x) for x in values.detach().cpu().tolist()],
        "greedy_token_id": greedy_id,
        "greedy_token_text": tokenizer.decode([greedy_id]),
    }


def main() -> int:
    args = parse_args()
    require_deps()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if args.output is None:
        safe_model = args.model_id.replace("/", "__")
        output_path = Path("traces") / f"{safe_model}__kv_decode__{timestamp}.jsonl"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, local_files_only=args.local_files_only)
    rendered_text, messages, prompt_mode = build_text(args, tokenizer)
    tokenized = tokenizer(rendered_text, return_tensors="pt")
    batch_size, prompt_len = tokenized["input_ids"].shape

    load_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
    }
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    model.eval()
    model.to(device)
    inputs = {name: value.to(device) for name, value in tokenized.items()}

    prompt_cache_position = torch.arange(prompt_len, device=device)
    prompt_position_ids = prompt_cache_position.unsqueeze(0).expand(batch_size, -1)

    from transformers.cache_utils import DynamicCache

    started = time.time()
    with torch.no_grad():
        prefill = model(
            **inputs,
            position_ids=prompt_position_ids,
            cache_position=prompt_cache_position,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )
    prefill_elapsed_s = time.time() - started

    prefill_next_logits = prefill.logits[0, -1]
    first_token = int(torch.argmax(prefill_next_logits.float()).item())

    decode_input_ids = torch.tensor([[first_token]], dtype=inputs["input_ids"].dtype, device=device)
    decode_attention_mask = torch.ones((batch_size, prompt_len + 1), dtype=inputs["attention_mask"].dtype, device=device)
    decode_cache_position = torch.arange(prompt_len, prompt_len + 1, device=device)
    decode_position_ids = decode_cache_position.unsqueeze(0).expand(batch_size, -1)

    past_key_values = prefill.past_key_values
    if isinstance(past_key_values, tuple):
        # Granite/HF returns legacy tuple caches when the initial call was made
        # without a Cache instance. Convert before the decode call to avoid a
        # deprecation warning and to mirror modern generate() behavior.
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    started = time.time()
    with torch.no_grad():
        decode = model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            position_ids=decode_position_ids,
            cache_position=decode_cache_position,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
    decode_elapsed_s = time.time() - started

    decode_next_logits = decode.logits[0, -1]

    with output_path.open("w", encoding="utf-8") as f:
        write_event(
            f,
            {
                "event": "kv_trace_metadata",
                "created_unix": time.time(),
                "model_id": args.model_id,
                "revision": args.revision,
                "script": Path(__file__).as_posix(),
                "argv": sys.argv,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "device": str(device),
                "requested_dtype": args.dtype,
                "model_param_dtype": clean_dtype_name(next(model.parameters()).dtype),
                "attn_implementation": args.attn_implementation,
                "config": selected_config(model.config),
            },
        )
        write_event(f, {"event": "prompt", "mode": prompt_mode, "messages": messages, "rendered_text": rendered_text})
        write_event(
            f,
            {
                "event": "prefill_inputs",
                "input_ids": inputs["input_ids"].detach().cpu().tolist(),
                "attention_mask": inputs["attention_mask"].detach().cpu().tolist(),
                "position_ids": prompt_position_ids.detach().cpu().tolist(),
                "cache_position": prompt_cache_position.detach().cpu().tolist(),
            },
        )
        write_event(
            f,
            {
                "event": "prefill_logits",
                "elapsed_s": prefill_elapsed_s,
                "logits": tensor_summary(prefill.logits),
                "next_token_logits": tensor_summary(prefill_next_logits),
                **topk_payload(prefill_next_logits, top_k=args.logit_top_k, tokenizer=tokenizer),
            },
        )
        write_event(f, {"event": "prefill_cache", **cache_summary(prefill.past_key_values)})
        write_event(
            f,
            {
                "event": "decode_inputs",
                "input_ids": decode_input_ids.detach().cpu().tolist(),
                "attention_mask": decode_attention_mask.detach().cpu().tolist(),
                "position_ids": decode_position_ids.detach().cpu().tolist(),
                "cache_position": decode_cache_position.detach().cpu().tolist(),
            },
        )
        write_event(
            f,
            {
                "event": "decode_logits",
                "elapsed_s": decode_elapsed_s,
                "logits": tensor_summary(decode.logits),
                "next_token_logits": tensor_summary(decode_next_logits),
                **topk_payload(decode_next_logits, top_k=args.logit_top_k, tokenizer=tokenizer),
            },
        )
        write_event(f, {"event": "decode_cache", **cache_summary(decode.past_key_values)})

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
