#!/usr/bin/env python3
"""Emit a Hugging Face golden trace for LFM2.5 MoE.

This intentionally instantiates the full HF model. It is not the Glacial
executor; it is the reference microscope used to build and debug that executor.

The trace captures intermediate tensors at each operation boundary via forward
hooks, so that Glacial probes can verify op-by-op parity:

  embedding → operator_norm → [conv|attention] → ffn_norm → [dense_mlp|moe] →
  layer_output → ... → embedding_norm → lm_head → logits

Output format: JSONL, one event per line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-8B-A1B"
DEFAULT_REVISION = "main"


# ---------------------------------------------------------------------------
# Shared utilities (same as hf_trace_granite.py)
# ---------------------------------------------------------------------------

def require_deps():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        print(
            "Missing Python dependency: " + exc.name + "\n\n"
            "Install trace dependencies, for example:\n\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ in {"dtype", "device"}:
        return str(value).removeprefix("torch.")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_event(f, event: dict[str, Any]) -> None:
    f.write(json.dumps(event, default=json_default, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    f.write("\n")
    f.flush()


def clean_dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def tensor_raw_sha256(tensor) -> str:
    import torch
    t = tensor.detach().cpu().contiguous()
    try:
        if t.dtype == torch.bfloat16:
            data = t.view(torch.int16).numpy().tobytes()
        else:
            data = t.numpy().tobytes()
    except Exception:
        data = t.float().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def tensor_summary(tensor) -> dict[str, Any]:
    import torch
    t = tensor.detach()
    out: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": clean_dtype_name(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "sha256": tensor_raw_sha256(t),
    }
    if t.numel() == 0:
        return out
    tf = t.float()
    out.update({
        "sum_fp32": float(tf.sum().item()),
        "mean_fp32": float(tf.mean().item()),
        "min_fp32": float(tf.min().item()),
        "max_fp32": float(tf.max().item()),
        "l2_fp32": float(torch.linalg.vector_norm(tf.reshape(-1), ord=2).item()),
    })
    return out


def maybe_inline_tensor(tensor, *, max_elements: int, as_float: bool = False):
    t = tensor.detach().cpu()
    if t.numel() > max_elements:
        return {"inline": False, "summary": tensor_summary(t)}
    if as_float:
        t = t.float()
    return t.tolist()


def resolve_dtype(name: str):
    import torch
    table = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
             "float16": torch.float16, "fp16": torch.float16,
             "float32": torch.float32, "fp32": torch.float32}
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise SystemExit(f"Unsupported dtype {name!r}; choose one of {sorted(table)}") from exc


def resolve_device(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ in {"dtype", "device"}:
        return str(value).removeprefix("torch.")
    return str(value)


# ---------------------------------------------------------------------------
# LFM2-specific config and hooks
# ---------------------------------------------------------------------------

def selected_config(config) -> dict[str, Any]:
    fields = [
        "architectures", "model_type", "vocab_size", "hidden_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "moe_intermediate_size", "num_experts",
        "num_experts_per_tok", "num_dense_layers", "norm_eps",
        "norm_topk_prob", "routed_scaling_factor", "use_expert_bias",
        "conv_L_cache", "conv_bias", "max_position_embeddings",
        "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id",
        "layer_types", "rope_parameters",
    ]
    out = {}
    for field in fields:
        val = getattr(config, field, None)
        if val is None and field in ("rope_parameters", "layer_types"):
            # These might be on the text_config for multimodal models
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                val = getattr(text_config, field, None)
        out[field] = make_jsonable(val)
    return out


def build_text(args, tokenizer) -> tuple[str, Any, str]:
    if args.messages_json:
        messages = json.loads(args.messages_json)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=args.add_generation_prompt)
        return text, messages, "messages_json"
    if args.chat_user is not None:
        messages = [{"role": "user", "content": args.chat_user}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text, messages, "chat_user"
    return args.prompt, None, "raw_prompt"


def make_io_summary_hook(*, event: str, layer_idx: int, records: dict[int, dict[str, Any]]):
    def hook(module, inputs, output):
        records[layer_idx] = {
            "event": event,
            "layer": layer_idx,
            "input": tensor_summary(inputs[0]),
            "output": tensor_summary(output),
        }
    return hook


def make_module_io_hook(*, event: str, layer_idx: int, records: dict[int, dict[str, Any]]):
    """Hook for modules that receive (hidden_states, ...) and return a tensor or tuple."""
    def hook(module, inputs, output):
        hidden_in = inputs[0] if inputs else None
        hidden_out = output[0] if isinstance(output, tuple) else output
        records[layer_idx] = {
            "event": event,
            "layer": layer_idx,
            "input": tensor_summary(hidden_in) if hidden_in is not None else None,
            "output": tensor_summary(hidden_out),
        }
    return hook


def make_router_logit_hook(*, layer_idx: int, records: dict[int, dict[str, Any]]):
    """Capture router logits from the gate Linear module."""
    def hook(module, inputs, output):
        records[layer_idx] = {
            "event": "layer_router_logits",
            "layer": layer_idx,
            "input": tensor_summary(inputs[0]),
            "router_logits": tensor_summary(output),
        }
    return hook


def make_moe_output_hook(*, layer_idx: int, records: dict[int, dict[str, Any]]):
    def hook(module, inputs, output):
        moe_out = output[0] if isinstance(output, tuple) else output
        records[layer_idx] = {
            "event": "layer_moe_output",
            "layer": layer_idx,
            "input": tensor_summary(inputs[0]),
            "output": tensor_summary(moe_out),
        }
    return hook


def make_layer_hook(*, layer_idx: int, layer_records: dict[int, dict[str, Any]]):
    def hook(module, inputs, output):
        hidden_out = output[0] if isinstance(output, tuple) else output
        layer_records[layer_idx] = {
            "event": "layer_output",
            "layer": layer_idx,
            "hidden": tensor_summary(hidden_out),
        }
    return hook


def patch_lfm2_rotary(*, records: dict[int, dict[str, Any]], attention_layer_indices: list[int]):
    """Patch LFM2's apply_rotary_pos_emb so the trace can see q/k after rotation.

    Unlike Granite (where all layers are attention layers), LFM2 only has 6
    attention layers. The call_count maps to the attention layer indices, not
    the sequential layer index.
    """
    from transformers.models.lfm2_moe import modeling_lfm2_moe
    original = modeling_lfm2_moe.apply_rotary_pos_emb
    call_count = {"value": 0}

    def wrapped(q, k, cos, sin, unsqueeze_dim=1):
        idx = call_count["value"]
        call_count["value"] += 1
        layer_idx = attention_layer_indices[idx] if idx < len(attention_layer_indices) else idx
        q_out, k_out = original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        records[layer_idx] = {
            "event": "layer_attention_rotary",
            "layer": layer_idx,
            "unsqueeze_dim": int(unsqueeze_dim),
            "q_input": tensor_summary(q),
            "k_input": tensor_summary(k),
            "cos": tensor_summary(cos),
            "sin": tensor_summary(sin),
            "q_output": tensor_summary(q_out),
            "k_output": tensor_summary(k_out),
        }
        return q_out, k_out

    modeling_lfm2_moe.apply_rotary_pos_emb = wrapped

    def restore():
        modeling_lfm2_moe.apply_rotary_pos_emb = original
    return restore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", default=None, help="Output JSONL path. Defaults to traces/<timestamp>.jsonl")
    parser.add_argument("--prompt", default="Hello", help="Raw prompt text when not using chat options.")
    parser.add_argument("--chat-user", default=None)
    parser.add_argument("--messages-json", default=None)
    parser.add_argument("--add-generation-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--logit-top-k", type=int, default=20)
    parser.add_argument("--max-inline-elements", type=int, default=4096)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


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
        output_path = Path("traces") / f"{safe_model}__{timestamp}.jsonl"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
    }
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, local_files_only=args.local_files_only)
    rendered_text, messages, prompt_mode = build_text(args, tokenizer)
    tokenized = tokenizer(rendered_text, return_tensors="pt")
    batch_size, seq_len = tokenized["input_ids"].shape

    print(f"Loading {args.model_id} (dtype={args.dtype}, attn={args.attn_implementation})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    model.eval()
    model.to(device)

    inputs = {name: value.to(device) for name, value in tokenized.items()}
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    layer_types = getattr(model.config, "layer_types", [])
    num_layers = int(model.config.num_hidden_layers)
    num_dense = int(getattr(model.config, "num_dense_layers", 0))
    attention_layer_indices = [i for i, t in enumerate(layer_types) if t == "full_attention"]

    # Records dicts
    operator_norm_records: dict[int, dict[str, Any]] = {}
    conv_records: dict[int, dict[str, Any]] = {}
    conv_in_proj_records: dict[int, dict[str, Any]] = {}
    conv_conv_records: dict[int, dict[str, Any]] = {}
    conv_out_proj_records: dict[int, dict[str, Any]] = {}
    q_proj_records: dict[int, dict[str, Any]] = {}
    k_proj_records: dict[int, dict[str, Any]] = {}
    v_proj_records: dict[int, dict[str, Any]] = {}
    q_layernorm_records: dict[int, dict[str, Any]] = {}
    k_layernorm_records: dict[int, dict[str, Any]] = {}
    rotary_records: dict[int, dict[str, Any]] = {}
    attention_output_records: dict[int, dict[str, Any]] = {}
    ffn_norm_records: dict[int, dict[str, Any]] = {}
    router_records: dict[int, dict[str, Any]] = {}
    moe_output_records: dict[int, dict[str, Any]] = {}
    mlp_records: dict[int, dict[str, Any]] = {}
    layer_records: dict[int, dict[str, Any]] = {}
    handles = []

    try:
        layers = model.model.layers
        for layer_idx, layer in enumerate(layers):
            is_attn = layer_idx in attention_layer_indices

            # operator_norm (before conv or attention)
            handles.append(
                layer.operator_norm.register_forward_hook(
                    make_io_summary_hook(event="layer_operator_norm", layer_idx=layer_idx, records=operator_norm_records)
                )
            )

            if is_attn:
                # Attention sub-layers
                handles.append(
                    layer.self_attn.q_proj.register_forward_hook(
                        make_io_summary_hook(event="layer_attention_q_proj", layer_idx=layer_idx, records=q_proj_records)
                    )
                )
                handles.append(
                    layer.self_attn.k_proj.register_forward_hook(
                        make_io_summary_hook(event="layer_attention_k_proj", layer_idx=layer_idx, records=k_proj_records)
                    )
                )
                handles.append(
                    layer.self_attn.v_proj.register_forward_hook(
                        make_io_summary_hook(event="layer_attention_v_proj", layer_idx=layer_idx, records=v_proj_records)
                    )
                )
                handles.append(
                    layer.self_attn.q_layernorm.register_forward_hook(
                        make_io_summary_hook(event="layer_attention_q_layernorm", layer_idx=layer_idx, records=q_layernorm_records)
                    )
                )
                handles.append(
                    layer.self_attn.k_layernorm.register_forward_hook(
                        make_io_summary_hook(event="layer_attention_k_layernorm", layer_idx=layer_idx, records=k_layernorm_records)
                    )
                )
                handles.append(
                    layer.self_attn.register_forward_hook(
                        make_module_io_hook(event="layer_attention_output", layer_idx=layer_idx, records=attention_output_records)
                    )
                )
            else:
                # Conv sub-layers
                handles.append(
                    layer.conv.register_forward_hook(
                        make_module_io_hook(event="layer_conv_output", layer_idx=layer_idx, records=conv_records)
                    )
                )
                handles.append(
                    layer.conv.in_proj.register_forward_hook(
                        make_io_summary_hook(event="layer_conv_in_proj", layer_idx=layer_idx, records=conv_in_proj_records)
                    )
                )
                handles.append(
                    layer.conv.conv.register_forward_hook(
                        make_io_summary_hook(event="layer_conv_conv", layer_idx=layer_idx, records=conv_conv_records)
                    )
                )
                handles.append(
                    layer.conv.out_proj.register_forward_hook(
                        make_io_summary_hook(event="layer_conv_out_proj", layer_idx=layer_idx, records=conv_out_proj_records)
                    )
                )

            # ffn_norm (before feed_forward)
            handles.append(
                layer.ffn_norm.register_forward_hook(
                    make_io_summary_hook(event="layer_ffn_norm", layer_idx=layer_idx, records=ffn_norm_records)
                )
            )

            # Feed-forward (MoE or dense MLP)
            if layer_idx >= num_dense:
                # MoE layer
                handles.append(
                    layer.feed_forward.gate.register_forward_hook(
                        make_router_logit_hook(layer_idx=layer_idx, records=router_records)
                    )
                )
                handles.append(
                    layer.feed_forward.register_forward_hook(
                        make_moe_output_hook(layer_idx=layer_idx, records=moe_output_records)
                    )
                )
            else:
                # Dense MLP layer
                handles.append(
                    layer.feed_forward.register_forward_hook(
                        make_module_io_hook(event="layer_mlp_output", layer_idx=layer_idx, records=mlp_records)
                    )
                )

            # Layer output
            handles.append(
                layer.register_forward_hook(
                    make_layer_hook(layer_idx=layer_idx, layer_records=layer_records)
                )
            )

    except AttributeError as exc:
        raise SystemExit(
            f"Could not attach LFM2 MoE hooks: {exc}. Is this an Lfm2MoeForCausalLM architecture?"
        ) from exc

    restore_rotary = patch_lfm2_rotary(records=rotary_records, attention_layer_indices=attention_layer_indices)

    # Also hook the embedding output (before any layers)
    embedding_records: dict[str, dict[str, Any]] = {}
    handles.append(
        model.model.embed_tokens.register_forward_hook(
            make_io_summary_hook(event="embedding_output", layer_idx=0, records=embedding_records)
        )
    )

    # Final norm (embedding_norm) — applied after all layers, before lm_head
    embedding_norm_records: dict[str, dict[str, Any]] = {}
    handles.append(
        model.model.embedding_norm.register_forward_hook(
            make_io_summary_hook(event="embedding_norm", layer_idx=0, records=embedding_norm_records)
        )
    )

    print(f"Running forward pass (seq_len={seq_len})...", flush=True)
    started = time.time()
    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
    finally:
        restore_rotary()
        for handle in handles:
            handle.remove()
    elapsed_s = time.time() - started

    logits = outputs.logits
    next_logits = logits[0, -1]
    next_logits_fp32 = next_logits.float()
    top_values, top_ids = next_logits_fp32.topk(args.logit_top_k)
    greedy_id = int(torch.argmax(next_logits_fp32).item())

    with output_path.open("w", encoding="utf-8") as f:
        write_event(f, {
            "event": "trace_metadata",
            "created_unix": time.time(),
            "model_id": args.model_id,
            "revision": args.revision,
            "script": Path(__file__).as_posix(),
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(device),
            "requested_dtype": args.dtype,
            "model_param_dtype": clean_dtype_name(next(model.parameters()).dtype),
            "attn_implementation": args.attn_implementation,
            "elapsed_s": elapsed_s,
            "config": selected_config(model.config),
            "layer_types": layer_types,
            "attention_layer_indices": attention_layer_indices,
            "num_dense_layers": num_dense,
        })

        write_event(f, {
            "event": "prompt",
            "mode": prompt_mode,
            "prompt": args.prompt if prompt_mode == "raw_prompt" else None,
            "messages": messages,
            "rendered_text": rendered_text,
        })

        write_event(f, {
            "event": "inputs",
            "input_ids": inputs["input_ids"].detach().cpu().tolist(),
            "attention_mask": inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).detach().cpu().tolist(),
            "position_ids": position_ids.detach().cpu().tolist(),
        })

        # Embedding: captured via embed_tokens hook
        if 0 in embedding_records:
            write_event(f, embedding_records[0])
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            write_event(f, {
                "event": "embedding_output",
                "note": "Fallback: HF hidden_states[0].",
                "hidden": tensor_summary(outputs.hidden_states[0]),
            })

        # Per-layer events
        for layer_idx in range(num_layers):
            if layer_idx in operator_norm_records:
                write_event(f, operator_norm_records[layer_idx])
            if layer_idx in conv_in_proj_records:
                write_event(f, conv_in_proj_records[layer_idx])
            if layer_idx in conv_conv_records:
                write_event(f, conv_conv_records[layer_idx])
            if layer_idx in conv_out_proj_records:
                write_event(f, conv_out_proj_records[layer_idx])
            if layer_idx in conv_records:
                write_event(f, conv_records[layer_idx])
            if layer_idx in q_proj_records:
                write_event(f, q_proj_records[layer_idx])
            if layer_idx in k_proj_records:
                write_event(f, k_proj_records[layer_idx])
            if layer_idx in v_proj_records:
                write_event(f, v_proj_records[layer_idx])
            if layer_idx in q_layernorm_records:
                write_event(f, q_layernorm_records[layer_idx])
            if layer_idx in k_layernorm_records:
                write_event(f, k_layernorm_records[layer_idx])
            if layer_idx in rotary_records:
                write_event(f, rotary_records[layer_idx])
            if layer_idx in attention_output_records:
                write_event(f, attention_output_records[layer_idx])
            if layer_idx in ffn_norm_records:
                write_event(f, ffn_norm_records[layer_idx])
            if layer_idx in router_records:
                write_event(f, router_records[layer_idx])
            if layer_idx in moe_output_records:
                write_event(f, moe_output_records[layer_idx])
            if layer_idx in mlp_records:
                write_event(f, mlp_records[layer_idx])
            if layer_idx in layer_records:
                write_event(f, layer_records[layer_idx])

        # Final norm (embedding_norm)
        if 0 in embedding_norm_records:
            write_event(f, embedding_norm_records[0])

        write_event(f, {
            "event": "final_logits",
            "logits": tensor_summary(logits),
            "next_token_logits": tensor_summary(next_logits),
            "top_k": int(args.logit_top_k),
            "top_token_ids": [int(x) for x in top_ids.detach().cpu().tolist()],
            "top_token_values_fp32": [float(x) for x in top_values.detach().cpu().tolist()],
            "greedy_token_id": greedy_id,
            "greedy_token_text": tokenizer.decode([greedy_id]),
        })

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())