#!/usr/bin/env python3
"""Emit a Hugging Face golden trace for Granite MoE.

This intentionally instantiates the full HF model. It is not the Glacial executor;
it is the reference microscope used to build and debug that executor.

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

# The Rust-backed Hugging Face tokenizer can initialize a thread pool. If the
# process later forks while loading/model code is running, tokenizers disables
# that pool and prints a scary-but-harmless warning. We only tokenize tiny trace
# prompts here, so disabling tokenizer parallelism is the right default.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
DEFAULT_REVISION = "b0e4fd07be563ba8bb7689c47dc9bebdff5471ab"


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

    # HF configs may contain torch.dtype / torch.device values after loading.
    # Keep the trace JSONL plain and stable without importing torch at module
    # import time.
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ in {"dtype", "device"}:
        return str(value).removeprefix("torch.")

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_event(f, event: dict[str, Any]) -> None:
    f.write(json.dumps(event, default=json_default, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    f.write("\n")
    f.flush()


def clean_dtype_name(dtype: Any) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


def tensor_raw_sha256(tensor) -> str:
    """Hash exact tensor storage bytes after moving to CPU.

    NumPy does not expose bfloat16 consistently, so BF16 is reinterpreted as
    int16 before hashing. For unsupported dtypes, fall back to FP32 bytes and
    let the summary dtype make that visible.
    """
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
    out.update(
        {
            "sum_fp32": float(tf.sum().item()),
            "mean_fp32": float(tf.mean().item()),
            "min_fp32": float(tf.min().item()),
            "max_fp32": float(tf.max().item()),
            "l2_fp32": float(torch.linalg.vector_norm(tf.reshape(-1), ord=2).item()),
        }
    )
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

    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
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
    # CPU is preferred over MPS for the default BF16 trace path; MPS BF16 support
    # varies by PyTorch/macOS version and can change numerics or fail outright.
    return torch.device("cpu")


def make_jsonable(value: Any) -> Any:
    """Convert common library scalar/config values to plain JSON values."""
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

    # Last resort for config-ish enum/scalar objects. Avoid this for tensors;
    # tensors should be summarized explicitly with tensor_summary().
    return str(value)


def selected_config(config) -> dict[str, Any]:
    fields = [
        "architectures",
        "model_type",
        "torch_dtype",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "num_local_experts",
        "num_experts_per_tok",
        "intermediate_size",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "rms_norm_eps",
        "attention_bias",
        "attention_dropout",
        "attention_multiplier",
        "embedding_multiplier",
        "residual_multiplier",
        "logits_scaling",
        "tie_word_embeddings",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    ]
    return {field: make_jsonable(getattr(config, field, None)) for field in fields}


def build_text(args, tokenizer) -> tuple[str, Any, str]:
    if args.messages_json:
        messages = json.loads(args.messages_json)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=args.add_generation_prompt,
        )
        return text, messages, "messages_json"

    if args.chat_user is not None:
        messages = [{"role": "user", "content": args.chat_user}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text, messages, "chat_user"

    return args.prompt, None, "raw_prompt"


def make_router_hook(
    *,
    layer_idx: int,
    config,
    batch_size: int,
    seq_len: int,
    max_inline_elements: int,
    router_records: dict[int, dict[str, Any]],
):
    def hook(module, inputs, output):
        import torch

        hidden_in = inputs[0]
        index_sorted_experts, batch_index, batch_gates, expert_size, logits = output

        top_k = int(getattr(config, "num_experts_per_tok"))
        top_k_logits, top_k_indices = logits.float().topk(top_k, dim=1)

        # HF semantics: softmax is FP32 because logits are FP32, then gates are
        # cast back to hidden dtype for multiplication with expert outputs.
        top_k_gates_fp32 = torch.softmax(top_k_logits, dim=1)
        top_k_gates_hf_dtype = top_k_gates_fp32.to(hidden_in.dtype)

        router_records[layer_idx] = {
            "event": "layer_router",
            "layer": layer_idx,
            "router_input": tensor_summary(hidden_in),
            "router_logits": tensor_summary(logits),
            "selected_experts_shape": [batch_size, seq_len, top_k],
            "selected_experts": maybe_inline_tensor(
                top_k_indices.view(batch_size, seq_len, top_k),
                max_elements=max_inline_elements,
            ),
            "selected_logits_fp32": maybe_inline_tensor(
                top_k_logits.view(batch_size, seq_len, top_k),
                max_elements=max_inline_elements,
                as_float=True,
            ),
            "gate_values_fp32_before_hf_cast": maybe_inline_tensor(
                top_k_gates_fp32.view(batch_size, seq_len, top_k),
                max_elements=max_inline_elements,
                as_float=True,
            ),
            "gate_values_after_hf_cast": maybe_inline_tensor(
                top_k_gates_hf_dtype.view(batch_size, seq_len, top_k),
                max_elements=max_inline_elements,
                as_float=True,
            ),
            "expert_size": [int(x) for x in expert_size],
            "index_sorted_experts": maybe_inline_tensor(index_sorted_experts, max_elements=max_inline_elements),
            "batch_index": maybe_inline_tensor(batch_index, max_elements=max_inline_elements),
            "batch_gates": maybe_inline_tensor(batch_gates, max_elements=max_inline_elements, as_float=True),
        }

    return hook


def make_io_summary_hook(*, event: str, layer_idx: int, records: dict[int, dict[str, Any]]):
    def hook(module, inputs, output):
        records[layer_idx] = {
            "event": event,
            "layer": layer_idx,
            "input": tensor_summary(inputs[0]),
            "output": tensor_summary(output),
        }

    return hook


def patch_granitemoe_rotary(*, records: dict[int, dict[str, Any]]):
    """Patch Granite's RoPE helper so the trace can see q/k after rotation.

    `apply_rotary_pos_emb` is a function, not an nn.Module, so ordinary forward
    hooks cannot observe it. During a single trace forward pass, calls occur in
    layer order, once per decoder layer.
    """
    from transformers.models.granitemoe import modeling_granitemoe

    original = modeling_granitemoe.apply_rotary_pos_emb
    call_count = {"value": 0}

    def wrapped(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        layer_idx = call_count["value"]
        call_count["value"] += 1
        q_out, k_out = original(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
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

    modeling_granitemoe.apply_rotary_pos_emb = wrapped

    def restore():
        modeling_granitemoe.apply_rotary_pos_emb = original

    return restore


def make_attention_output_hook(*, layer_idx: int, records: dict[int, dict[str, Any]]):
    def hook(module, inputs, kwargs, output):
        hidden_in = kwargs.get("hidden_states") if kwargs else None
        if hidden_in is None and inputs:
            hidden_in = inputs[0]
        hidden_out = output[0] if isinstance(output, tuple) else output
        records[layer_idx] = {
            "event": "layer_attention_output",
            "layer": layer_idx,
            "input": tensor_summary(hidden_in),
            "output": tensor_summary(hidden_out),
        }

    return hook


def make_moe_output_hook(*, layer_idx: int, records: dict[int, dict[str, Any]]):
    def hook(module, inputs, output):
        moe_out = output[0] if isinstance(output, tuple) else output
        router_logits = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        record = {
            "event": "layer_moe_output",
            "layer": layer_idx,
            "input": tensor_summary(inputs[0]),
            "output": tensor_summary(moe_out),
        }
        if router_logits is not None:
            record["router_logits"] = tensor_summary(router_logits)
        records[layer_idx] = record

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
    parser.add_argument("--attn-implementation", default="eager", help="Use eager for the first golden trace.")
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

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    model.eval()
    model.to(device)

    inputs = {name: value.to(device) for name, value in tokenized.items()}
    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

    input_norm_records: dict[int, dict[str, Any]] = {}
    q_proj_records: dict[int, dict[str, Any]] = {}
    k_proj_records: dict[int, dict[str, Any]] = {}
    v_proj_records: dict[int, dict[str, Any]] = {}
    rotary_records: dict[int, dict[str, Any]] = {}
    attention_output_records: dict[int, dict[str, Any]] = {}
    post_attention_norm_records: dict[int, dict[str, Any]] = {}
    router_records: dict[int, dict[str, Any]] = {}
    moe_output_records: dict[int, dict[str, Any]] = {}
    layer_records: dict[int, dict[str, Any]] = {}
    handles = []

    try:
        layers = model.model.layers
        for layer_idx, layer in enumerate(layers):
            handles.append(
                layer.input_layernorm.register_forward_hook(
                    make_io_summary_hook(
                        event="layer_input_norm",
                        layer_idx=layer_idx,
                        records=input_norm_records,
                    )
                )
            )
            handles.append(
                layer.self_attn.q_proj.register_forward_hook(
                    make_io_summary_hook(
                        event="layer_attention_q_proj",
                        layer_idx=layer_idx,
                        records=q_proj_records,
                    )
                )
            )
            handles.append(
                layer.self_attn.k_proj.register_forward_hook(
                    make_io_summary_hook(
                        event="layer_attention_k_proj",
                        layer_idx=layer_idx,
                        records=k_proj_records,
                    )
                )
            )
            handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    make_io_summary_hook(
                        event="layer_attention_v_proj",
                        layer_idx=layer_idx,
                        records=v_proj_records,
                    )
                )
            )
            handles.append(
                layer.self_attn.register_forward_hook(
                    make_attention_output_hook(
                        layer_idx=layer_idx,
                        records=attention_output_records,
                    ),
                    with_kwargs=True,
                )
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    make_io_summary_hook(
                        event="layer_post_attention_norm",
                        layer_idx=layer_idx,
                        records=post_attention_norm_records,
                    )
                )
            )
            handles.append(
                layer.block_sparse_moe.router.register_forward_hook(
                    make_router_hook(
                        layer_idx=layer_idx,
                        config=model.config,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        max_inline_elements=args.max_inline_elements,
                        router_records=router_records,
                    )
                )
            )
            handles.append(
                layer.block_sparse_moe.register_forward_hook(
                    make_moe_output_hook(
                        layer_idx=layer_idx,
                        records=moe_output_records,
                    )
                )
            )
            handles.append(layer.register_forward_hook(make_layer_hook(layer_idx=layer_idx, layer_records=layer_records)))
    except AttributeError as exc:
        raise SystemExit(
            "Could not attach Granite MoE hooks. Is this still a GraniteMoeForCausalLM architecture?"
        ) from exc

    restore_rotary = patch_granitemoe_rotary(records=rotary_records)
    started = time.time()
    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=False,
                output_hidden_states=True,
                output_router_logits=True,
                return_dict=True,
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
    # Greedy decode means argmax. Keep top-k separately for diagnostics; topk
    # tie ordering is not the same semantic as argmax tie ordering.
    greedy_id = int(torch.argmax(next_logits_fp32).item())

    with output_path.open("w", encoding="utf-8") as f:
        write_event(
            f,
            {
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
            },
        )
        write_event(
            f,
            {
                "event": "prompt",
                "mode": prompt_mode,
                "prompt": args.prompt if prompt_mode == "raw_prompt" else None,
                "messages": messages,
                "rendered_text": rendered_text,
            },
        )
        write_event(
            f,
            {
                "event": "inputs",
                "input_ids": inputs["input_ids"].detach().cpu().tolist(),
                "attention_mask": inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).detach().cpu().tolist(),
                "position_ids": position_ids.detach().cpu().tolist(),
                "cache_position": cache_position.detach().cpu().tolist(),
                "input_ids_summary": tensor_summary(inputs["input_ids"]),
            },
        )

        if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
            write_event(
                f,
                {
                    "event": "embedding_output",
                    "note": "HF hidden_states[0]: token embeddings after Granite embedding_multiplier, before layer 0.",
                    "hidden": tensor_summary(outputs.hidden_states[0]),
                },
            )

        for layer_idx in range(int(model.config.num_hidden_layers)):
            if layer_idx in input_norm_records:
                write_event(f, input_norm_records[layer_idx])
            if layer_idx in q_proj_records:
                write_event(f, q_proj_records[layer_idx])
            if layer_idx in k_proj_records:
                write_event(f, k_proj_records[layer_idx])
            if layer_idx in v_proj_records:
                write_event(f, v_proj_records[layer_idx])
            if layer_idx in rotary_records:
                write_event(f, rotary_records[layer_idx])
            if layer_idx in attention_output_records:
                write_event(f, attention_output_records[layer_idx])
            if layer_idx in post_attention_norm_records:
                write_event(f, post_attention_norm_records[layer_idx])
            if layer_idx in router_records:
                write_event(f, router_records[layer_idx])
            if layer_idx in moe_output_records:
                write_event(f, moe_output_records[layer_idx])
            if layer_idx in layer_records:
                write_event(f, layer_records[layer_idx])

        if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
            write_event(
                f,
                {
                    "event": "final_norm_output",
                    "note": "HF hidden_states[-1]: after final RMSNorm, before lm_head.",
                    "hidden": tensor_summary(outputs.hidden_states[-1]),
                },
            )

        write_event(
            f,
            {
                "event": "final_logits",
                "logits": tensor_summary(logits),
                "next_token_logits": tensor_summary(next_logits),
                "top_k": int(args.logit_top_k),
                "top_token_ids": [int(x) for x in top_ids.detach().cpu().tolist()],
                "top_token_values_fp32": [float(x) for x in top_values.detach().cpu().tolist()],
                "greedy_token_id": greedy_id,
                "greedy_token_text": tokenizer.decode([greedy_id]),
            },
        )

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
