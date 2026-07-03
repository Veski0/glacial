"""Greedy parity tests: Glacial decode must match HF greedy decode token-for-token.

These are the core correctness tests.  They prove that Glacial's out-of-core
weight-visiting decode produces exactly the same tokens as a standard
Hugging Face model instance using greedy argmax.

Run with:

    .venv/bin/python -m pytest tests/test_greedy_parity.py --runslow -v
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

# Diverse prompts: short, factual, and creative.
PROMPTS = [
    "Hello",
    "The capital of France is",
    "Once upon a time, there was a",
]

MAX_NEW_TOKENS = [1, 4, 8]


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("max_new_tokens", MAX_NEW_TOKENS)
def test_greedy_parity_kv(prompt: str, max_new_tokens: int, hf_greedy, glacial_decode, tokenizer):
    """KV-cache path: Glacial greedy == HF greedy, token-for-token."""
    prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    hf_tokens = hf_greedy(prompt, max_new_tokens, stop_on_eos=False)

    _, glacial_tokens = glacial_decode(
        prompt_ids,
        max_new_tokens,
        stop_on_eos=False,
    )

    assert glacial_tokens == hf_tokens, (
        f"Greedy parity (KV path) failed for prompt={prompt!r}, n={max_new_tokens}\n"
        f"HF tokens:      {hf_tokens}\n"
        f"Glacial tokens: {glacial_tokens}\n"
        f"HF text:        {tokenizer.decode(hf_tokens)!r}\n"
        f"Glacial text:   {tokenizer.decode(glacial_tokens)!r}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_parity_no_kv(prompt: str, hf_greedy, glacial_decode, tokenizer, backend, model_file, safetensors_info, config):
    """No-KV path (full recompute each token): Glacial greedy == HF greedy."""
    header, payload_start = safetensors_info
    prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    hf_tokens = hf_greedy(prompt, 4, stop_on_eos=False)

    # Use the no-KV fallback path (recompute full prompt each token).
    all_ids = list(prompt_ids)
    for _ in range(4):
        next_id, _ = backend.next_token_greedy(
            token_ids=all_ids,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            config=config,
            lm_head_chunk_rows=4096,
        )
        all_ids.append(next_id)

    glacial_tokens = all_ids[len(prompt_ids):]

    assert glacial_tokens == hf_tokens, (
        f"Greedy parity (no-KV path) failed for prompt={prompt!r}\n"
        f"HF tokens:      {hf_tokens}\n"
        f"Glacial tokens: {glacial_tokens}\n"
        f"HF text:        {tokenizer.decode(hf_tokens)!r}\n"
        f"Glacial text:   {tokenizer.decode(glacial_tokens)!r}"
    )