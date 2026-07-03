#!/usr/bin/env python3
"""Greedy Glacial generation prototype.

This is intentionally simple and slow, but now uses the proven KV-cache path:
prefill the prompt once, then decode one token at a time against cached K/V.
It does not instantiate an HF model and visits weights from safetensors.

Purpose: make the exact out-of-core prototype interactable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
DEFAULT_REVISION = "b0e4fd07be563ba8bb7689c47dc9bebdff5471ab"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glacial.backends import backend_names, resolve_backend
from glacial.kv import load_decode_checkpoint, save_decode_checkpoint
from glacial.sampler import Sampler
from glacial.weights import WeightBudget
from probe_embedding import read_safetensors_header, require_torch  # same-directory import when run as tools/*.py


def require_deps():
    require_torch()
    try:
        import huggingface_hub  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        print(
            "Missing Python dependency: " + exc.name + "\n\n"
            "Install dependencies, for example:\n\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def resolve_repo_file(*, model_id: str, revision: str, filename: str, cache_dir: str | None, local_files_only: bool) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )


def load_config(*, model_id: str, revision: str, cache_dir: str | None, local_files_only: bool) -> dict[str, Any]:
    path = resolve_repo_file(
        model_id=model_id,
        revision=revision,
        filename="config.json",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return json.loads(path.read_text(encoding="utf-8"))


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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-file", type=Path, default=None, help="Local model.safetensors path. Defaults to HF cache/hub.")
    parser.add_argument("--backend", default="auto", help="Architecture backend to use: auto or one of " + ", ".join(backend_names()))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", default="Hello", help="Raw prompt when not using chat options.")
    parser.add_argument("--chat-user", default=None, help="Render a single user message with the model chat template.")
    parser.add_argument("--messages-json", default=None, help="JSON array of chat messages to render with chat template.")
    parser.add_argument("--add-generation-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096, help="Rows per tied-LM-head chunk for streaming greedy argmax.")
    parser.add_argument("--no-kv-cache", action="store_true", help="Fallback to old full-prefill-every-token generation path.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Persist a resumable KV checkpoint after each emitted token.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume from a Glacial KV checkpoint directory.")
    parser.add_argument("--no-stop-on-eos", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for sampling.")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--show-token-telemetry", action="store_true")
    parser.add_argument("--weight-budget-bytes", type=int, default=None, help="Optional resident weight-byte budget.")
    parser.add_argument("--enforce-weight-budget", action="store_true", help="Raise if resident weights exceed --weight-budget-bytes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_deps()

    import torch
    from transformers import AutoTokenizer

    torch.set_grad_enabled(False)

    if args.model_file is None:
        model_file = resolve_repo_file(
            model_id=args.model_id,
            revision=args.revision,
            filename="model.safetensors",
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    else:
        model_file = args.model_file
    if not model_file.exists():
        raise SystemExit(f"model.safetensors not found: {model_file}")

    config = load_config(
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    backend = resolve_backend(args.backend, config=config)
    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if args.resume_from is not None and args.no_kv_cache:
        raise SystemExit("--resume-from requires the KV-cache path; remove --no-kv-cache")
    if args.checkpoint_dir is not None and args.no_kv_cache:
        raise SystemExit("--checkpoint-dir persists KV checkpoints; remove --no-kv-cache")
    if args.resume_from is None and args.checkpoint_dir is not None and (args.checkpoint_dir / "manifest.json").exists():
        raise SystemExit(f"Checkpoint already exists: {args.checkpoint_dir}. Use --resume-from to continue it.")
    if (
        args.resume_from is not None
        and args.checkpoint_dir is not None
        and args.checkpoint_dir != args.resume_from
        and (args.checkpoint_dir / "manifest.json").exists()
    ):
        raise SystemExit(f"Refusing to overwrite existing checkpoint: {args.checkpoint_dir}")

    checkpoint_state = None
    kv_cache = None
    checkpoint_dir = args.checkpoint_dir
    if args.resume_from is not None:
        checkpoint_state = load_decode_checkpoint(args.resume_from)
        checkpoint_dir = checkpoint_dir or args.resume_from
        manifest = checkpoint_state["manifest"]
        manifest_backend = manifest.get("backend")
        if manifest_backend is not None and manifest_backend != backend.name:
            raise SystemExit(f"Checkpoint backend {manifest_backend!r} does not match selected backend {backend.name!r}")
        manifest_model = manifest["model"]
        if manifest_model.get("model_id") != args.model_id or manifest_model.get("revision") != args.revision:
            raise SystemExit(
                "Checkpoint model does not match CLI model/revision:\n"
                f"  checkpoint: {manifest_model.get('model_id')} @ {manifest_model.get('revision')}\n"
                f"  CLI:        {args.model_id} @ {args.revision}"
            )
        token_ids = list(checkpoint_state["token_ids"])
        kv_cache = checkpoint_state["kv_cache"]
        prompt_info = manifest.get("prompt", {})
        rendered_text = prompt_info.get("rendered_text")
        _messages = prompt_info.get("messages")
        mode = prompt_info.get("mode") or "checkpoint"
        prompt_token_count = int(manifest["state"]["prompt_token_count"])
    else:
        rendered_text, _messages, mode = build_text(args, tokenizer)
        token_ids = tokenizer(rendered_text, return_tensors=None)["input_ids"]
        prompt_token_count = len(token_ids)

    # Construct the sampler.  On resume, a sampling checkpoint always restores
    # its RNG state from the manifest (the CLI sampling args are ignored).  A
    # greedy checkpoint falls back to the CLI sampling args so the user can
    # start sampling from a greedy prefill if desired.
    if checkpoint_state is not None:
        checkpoint_sampler = Sampler.from_manifest(checkpoint_state["manifest"].get("sampler", {}))
        if checkpoint_sampler.is_greedy():
            sampler = Sampler.from_params(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )
        else:
            sampler = checkpoint_sampler
    else:
        sampler = Sampler.from_params(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )

    generated: list[int] = []
    budget = None
    if args.weight_budget_bytes is not None:
        budget = WeightBudget(limit_bytes=args.weight_budget_bytes, enforce=args.enforce_weight_budget)

    if args.show_prompt and rendered_text is not None:
        print(rendered_text)
        print("---")

    if (
        checkpoint_state is not None
        and not args.no_stop_on_eos
        and tokenizer.eos_token_id is not None
        and token_ids[-1] == tokenizer.eos_token_id
    ):
        if args.show_token_telemetry:
            print("[resume: checkpoint already ended on EOS; use --no-stop-on-eos to force continuation]")
        return 0

    print("", end="", flush=True)
    started = time.time()

    def persist_checkpoint() -> dict[str, Any] | None:
        if checkpoint_dir is None:
            return None
        if kv_cache is None:
            raise SystemExit("Internal error: cannot checkpoint without kv_cache")
        return save_decode_checkpoint(
            run_dir=checkpoint_dir,
            token_ids=token_ids,
            prompt_token_count=prompt_token_count,
            kv_cache=kv_cache,
            model_id=args.model_id,
            revision=args.revision,
            model_file=model_file,
            backend_name=backend.name,
            rendered_text=rendered_text,
            prompt_mode=mode,
            messages=_messages,
            config=config,
            lm_head_chunk_rows=args.lm_head_chunk_rows,
            sampler=sampler.to_manifest() if sampler is not None else None,
        )

    def emit_token(step: int, next_id: int, telemetry: dict[str, Any], step_started: float) -> bool:
        token_ids.append(next_id)
        generated.append(next_id)

        # Product guarantee for checkpointed runs: if a token is visible to the
        # user, it has already been durably checkpointed and can be resumed past.
        checkpoint_manifest = persist_checkpoint()

        piece = tokenizer.decode([next_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        print(piece, end="", flush=True)

        if args.show_token_telemetry:
            elapsed = time.time() - step_started
            phase = telemetry.get("phase", "full")
            kv_extra = f" kv_len={telemetry['kv_length']}" if "kv_length" in telemetry else ""
            budget_extra = ""
            if "weight_budget_peak_bytes" in telemetry:
                budget_extra = (
                    f" weight_peak={telemetry['weight_budget_peak_bytes']}B"
                    f" budget_violations={telemetry['weight_budget_violation_count']}"
                )
            print(
                f"\n[token {step + 1}: phase={phase} id={next_id} elapsed={elapsed:.2f}s "
                f"visited={telemetry['cumulative_weight_bytes']}B "
                f"before_lm={telemetry.get('visited_before_lm_head_bytes', 'n/a')}B "
                f"lm_chunk={telemetry['peak_lm_head_chunk_bytes']}B "
                f"peak_experts={telemetry['peak_selected_expert_count']}"
                f"{kv_extra}"
                f"{budget_extra}]",
                flush=True,
            )
            if checkpoint_manifest is not None:
                print(
                    f"[checkpoint: {checkpoint_dir / 'manifest.json'} "
                    f"tokens={checkpoint_manifest['state']['token_count']} "
                    f"kv_len={checkpoint_manifest['state']['kv_length']} durable_before_visible=True]",
                    flush=True,
                )

        return bool(
            not args.no_stop_on_eos
            and tokenizer.eos_token_id is not None
            and next_id == tokenizer.eos_token_id
        )

    if args.no_kv_cache:
        for step in range(args.max_new_tokens):
            step_started = time.time()
            next_id, telemetry = backend.next_token_greedy(
                token_ids=token_ids,
                model_file=model_file,
                header=header,
                payload_start=payload_start,
                config=config,
                lm_head_chunk_rows=args.lm_head_chunk_rows,
                budget=budget,
                sampler=sampler,
            )
            telemetry.setdefault("phase", "full")
            if emit_token(step, next_id, telemetry, step_started):
                break
    elif args.max_new_tokens > 0:
        if checkpoint_state is not None:
            # Resume invariant: KV contains all tokens except token_ids[-1].
            # Decode token_ids[-1] at its absolute position, then checkpoint the
            # returned cache after appending the newly emitted token.
            for step in range(args.max_new_tokens):
                position = len(token_ids) - 1
                step_started = time.time()
                next_id, kv_cache, telemetry = backend.decode_kv_greedy(
                    input_token_id=token_ids[-1],
                    position=position,
                    kv_cache=kv_cache,
                    model_file=model_file,
                    header=header,
                    payload_start=payload_start,
                    config=config,
                    lm_head_chunk_rows=args.lm_head_chunk_rows,
                    budget=budget,
                    sampler=sampler,
                )
                should_stop = emit_token(step, next_id, telemetry, step_started)
                if should_stop:
                    break
        else:
            # Prefill prompt once. The resulting logits produce the first generated
            # token, while the returned KV cache contains the prompt tokens.
            step_started = time.time()
            next_id, kv_cache, telemetry = backend.prefill_kv_greedy(
                token_ids=token_ids,
                model_file=model_file,
                header=header,
                payload_start=payload_start,
                config=config,
                lm_head_chunk_rows=args.lm_head_chunk_rows,
                budget=budget,
                sampler=sampler,
            )
            should_stop = emit_token(0, next_id, telemetry, step_started)
            if not should_stop:
                for step in range(1, args.max_new_tokens):
                    # KV cache currently contains all tokens except the final token
                    # in token_ids. Decode that final token at its absolute position.
                    position = len(token_ids) - 1
                    step_started = time.time()
                    next_id, kv_cache, telemetry = backend.decode_kv_greedy(
                        input_token_id=token_ids[-1],
                        position=position,
                        kv_cache=kv_cache,
                        model_file=model_file,
                        header=header,
                        payload_start=payload_start,
                        config=config,
                        lm_head_chunk_rows=args.lm_head_chunk_rows,
                        budget=budget,
                        sampler=sampler,
                    )
                    should_stop = emit_token(step, next_id, telemetry, step_started)
                    if should_stop:
                        break

    total_elapsed = time.time() - started
    print("", flush=True)
    if args.show_token_telemetry:
        print(f"[done: generated={len(generated)} total_elapsed={total_elapsed:.2f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
