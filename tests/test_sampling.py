"""Sampling tests: determinism, resume parity, and greedy equivalence.

These tests verify that Glacial's sampling implementation:

1. Is internally deterministic: same seed → same token sequence.
2. Survives interruption: checkpoint + resume reproduces the same sequence.
3. Produces different sequences for different seeds.
4. Degrades to greedy when temperature=0 (matching HF greedy).

Note: we do *not* test against HF sampled tokens — HF's RNG seeding and
multinomial call pattern differ from ours, so exact sampled-token parity
across implementations is not achievable.  Internal determinism and resume
parity are the real guarantees.

Run with:

    .venv/bin/python -m pytest tests/test_sampling.py --runslow -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glacial.sampler import Sampler

pytestmark = pytest.mark.slow

PROMPTS = [
    "Hello",
    "The capital of France is",
]

MAX_TOKENS = 8


@pytest.mark.parametrize("prompt", PROMPTS)
def test_sampling_determinism(prompt: str, glacial_decode, tokenizer):
    """Same seed must produce the same token sequence across independent runs."""
    prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    sampler_a = Sampler.from_params(temperature=0.8, seed=42)
    _, tokens_a = glacial_decode(prompt_ids, MAX_TOKENS, sampler=sampler_a)

    sampler_b = Sampler.from_params(temperature=0.8, seed=42)
    _, tokens_b = glacial_decode(prompt_ids, MAX_TOKENS, sampler=sampler_b)

    assert tokens_a == tokens_b, (
        f"Sampling determinism failed for prompt={prompt!r}\n"
        f"Run A: {tokens_a}\n"
        f"Run B: {tokens_b}\n"
        f"Text A: {tokenizer.decode(tokens_a)!r}\n"
        f"Text B: {tokenizer.decode(tokens_b)!r}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_sampling_resume(prompt: str, glacial_decode, tokenizer, tmp_path: Path):
    """Checkpoint + resume must reproduce the same sampled sequence."""
    prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    sampler = Sampler.from_params(temperature=0.8, seed=42)
    _, full_tokens = glacial_decode(prompt_ids, MAX_TOKENS, sampler=sampler)

    # Checkpoint after 3 tokens, then resume for the remainder.
    ckpt_dir = tmp_path / "checkpoint"
    sampler_1 = Sampler.from_params(temperature=0.8, seed=42)
    _, first_part = glacial_decode(
        prompt_ids, 3,
        checkpoint_dir=ckpt_dir,
        sampler=sampler_1,
    )

    # On resume, the sampler is restored from the checkpoint manifest
    # (RNG state included), so no explicit sampler is needed.
    _, resumed_part = glacial_decode(
        max_new_tokens=MAX_TOKENS - 3,
        resume_from=ckpt_dir,
    )

    combined = first_part + resumed_part
    assert combined == full_tokens, (
        f"Sampling resume parity failed for prompt={prompt!r}\n"
        f"Uninterrupted: {full_tokens}\n"
        f"Combined:      {combined}\n"
        f"Text (full):    {tokenizer.decode(full_tokens)!r}\n"
        f"Text (combined): {tokenizer.decode(combined)!r}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_different_seeds_differ(prompt: str, glacial_decode, tokenizer):
    """Different seeds should (almost certainly) produce different sequences."""
    prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    sampler_a = Sampler.from_params(temperature=0.8, seed=42)
    _, tokens_a = glacial_decode(prompt_ids, MAX_TOKENS, sampler=sampler_a)

    sampler_b = Sampler.from_params(temperature=0.8, seed=999)
    _, tokens_b = glacial_decode(prompt_ids, MAX_TOKENS, sampler=sampler_b)

    assert tokens_a != tokens_b, (
        f"Different seeds produced identical sequences for prompt={prompt!r}\n"
        f"Seed 42:  {tokens_a}\n"
        f"Seed 999: {tokens_b}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_top_p_sampling_determinism(prompt: str, glacial_decode, tokenizer):
    """Top-p sampling with a fixed seed must be deterministic."""
    prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    sampler_a = Sampler.from_params(temperature=0.8, top_p=0.9, seed=7)
    _, tokens_a = glacial_decode(prompt_ids, MAX_TOKENS, sampler=sampler_a)

    sampler_b = Sampler.from_params(temperature=0.8, top_p=0.9, seed=7)
    _, tokens_b = glacial_decode(prompt_ids, MAX_TOKENS, sampler=sampler_b)

    assert tokens_a == tokens_b, (
        f"Top-p determinism failed for prompt={prompt!r}\n"
        f"Run A: {tokens_a}\n"
        f"Run B: {tokens_b}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_top_k_sampling_determinism(prompt: str, glacial_decode, tokenizer):
    """Top-k sampling with a fixed seed must be deterministic."""
    prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    sampler_a = Sampler.from_params(temperature=0.8, top_k=40, seed=7)
    _, tokens_a = glacial_decode(prompt_ids, MAX_TOKENS, sampler=sampler_a)

    sampler_b = Sampler.from_params(temperature=0.8, top_k=40, seed=7)
    _, tokens_b = glacial_decode(prompt_ids, MAX_TOKENS, sampler=sampler_b)

    assert tokens_a == tokens_b, (
        f"Top-k determinism failed for prompt={prompt!r}\n"
        f"Run A: {tokens_a}\n"
        f"Run B: {tokens_b}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_temperature_zero_matches_greedy(prompt: str, hf_greedy, glacial_decode, tokenizer):
    """temperature=0 must produce the same tokens as HF greedy argmax."""
    prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    hf_tokens = hf_greedy(prompt, MAX_TOKENS, stop_on_eos=False)

    sampler = Sampler.from_params(temperature=0.0)
    _, glacial_tokens = glacial_decode(prompt_ids, MAX_TOKENS, sampler=sampler)

    assert glacial_tokens == hf_tokens, (
        f"temperature=0 != greedy for prompt={prompt!r}\n"
        f"HF:       {hf_tokens}\n"
        f"Glacial:  {glacial_tokens}"
    )


def test_sampling_checkpoint_has_rng_state(glacial_decode, tokenizer, tmp_path: Path):
    """A sampling checkpoint must persist non-null RNG state."""
    from glacial.kv import inspect_decode_checkpoint

    prompt_ids = tokenizer("Hello", return_tensors=None)["input_ids"]
    ckpt_dir = tmp_path / "checkpoint"

    sampler = Sampler.from_params(temperature=0.8, seed=42)
    glacial_decode(prompt_ids, 4, checkpoint_dir=ckpt_dir, sampler=sampler)

    result = inspect_decode_checkpoint(ckpt_dir, validate_kv=True)
    assert result["valid"], f"Checkpoint invalid: {result['validation_errors']}"
    assert result["manifest"]["sampler"]["type"] == "sample"
    assert result["manifest"]["sampler"]["rng_state"] is not None
    assert result["manifest"]["sampler"]["seed"] == 42


def test_greedy_checkpoint_has_no_rng_state(glacial_decode, tokenizer, tmp_path: Path):
    """A greedy checkpoint must have type=greedy and rng_state=None."""
    from glacial.kv import inspect_decode_checkpoint

    prompt_ids = tokenizer("Hello", return_tensors=None)["input_ids"]
    ckpt_dir = tmp_path / "checkpoint"

    glacial_decode(prompt_ids, 4, checkpoint_dir=ckpt_dir)

    result = inspect_decode_checkpoint(ckpt_dir, validate_kv=True)
    assert result["valid"], f"Checkpoint invalid: {result['validation_errors']}"
    assert result["manifest"]["sampler"]["type"] == "greedy"
    assert result["manifest"]["sampler"]["rng_state"] is None