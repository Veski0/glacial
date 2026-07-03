"""Resume parity tests: checkpoint + resume must match uninterrupted decode.

These tests verify Glacial's central product guarantee: *if Glacial showed you
a token, Glacial can resume past that token.*  A checkpointed run that is
interrupted and resumed must produce exactly the same token sequence as a
single uninterrupted run.

Run with:

    .venv/bin/python -m pytest tests/test_resume_parity.py --runslow -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

PROMPTS = [
    "Hello",
    "The capital of France is",
]

TOTAL_TOKENS = 8
SPLIT_AT = 3


@pytest.mark.parametrize("prompt", PROMPTS)
def test_resume_matches_uninterrupted(
    prompt: str,
    glacial_decode,
    tokenizer,
    tmp_path: Path,
):
    """A single checkpoint/resume cycle must match uninterrupted decode."""
    prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    # Uninterrupted reference run.
    _, full_generated = glacial_decode(prompt_ids, TOTAL_TOKENS, stop_on_eos=False)

    # Checkpoint after SPLIT_AT tokens, then resume for the remainder.
    ckpt_dir = tmp_path / "checkpoint"
    _, first_part = glacial_decode(
        prompt_ids,
        SPLIT_AT,
        checkpoint_dir=ckpt_dir,
        stop_on_eos=False,
    )
    _, resumed_part = glacial_decode(
        max_new_tokens=TOTAL_TOKENS - SPLIT_AT,
        resume_from=ckpt_dir,
        stop_on_eos=False,
    )

    combined = first_part + resumed_part
    assert combined == full_generated, (
        f"Resume parity failed for prompt={prompt!r}\n"
        f"Uninterrupted: {full_generated}\n"
        f"Checkpointed: {first_part}\n"
        f"Resumed:      {resumed_part}\n"
        f"Combined:     {combined}\n"
        f"Full text:     {tokenizer.decode(full_generated)!r}\n"
        f"Combined text: {tokenizer.decode(combined)!r}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_multi_step_resume(
    prompt: str,
    glacial_decode,
    tokenizer,
    tmp_path: Path,
):
    """Multiple checkpoint/resume cycles must match uninterrupted decode."""
    prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    _, full_generated = glacial_decode(prompt_ids, TOTAL_TOKENS, stop_on_eos=False)

    # Cycle 1: fresh -> checkpoint after 3 tokens.
    ckpt1 = tmp_path / "ckpt1"
    _, part1 = glacial_decode(prompt_ids, 3, checkpoint_dir=ckpt1, stop_on_eos=False)

    # Cycle 2: resume from ckpt1 -> checkpoint after 3 more tokens.
    ckpt2 = tmp_path / "ckpt2"
    _, part2 = glacial_decode(
        max_new_tokens=3,
        resume_from=ckpt1,
        checkpoint_dir=ckpt2,
        stop_on_eos=False,
    )

    # Cycle 3: resume from ckpt2 -> generate remaining 2 tokens (no checkpoint).
    _, part3 = glacial_decode(
        max_new_tokens=TOTAL_TOKENS - 3 - 3,
        resume_from=ckpt2,
        stop_on_eos=False,
    )

    combined = part1 + part2 + part3
    assert combined == full_generated, (
        f"Multi-step resume parity failed for prompt={prompt!r}\n"
        f"Uninterrupted: {full_generated}\n"
        f"Combined:     {combined}\n"
        f"  part1: {part1}\n"
        f"  part2: {part2}\n"
        f"  part3: {part3}"
    )


def test_resume_checkpoint_inspectable(
    glacial_decode,
    tokenizer,
    tmp_path: Path,
):
    """A saved checkpoint must be inspectable and valid."""
    from glacial.kv import inspect_decode_checkpoint

    prompt_ids = tokenizer("Hello", return_tensors=None)["input_ids"]
    ckpt_dir = tmp_path / "checkpoint"

    glacial_decode(prompt_ids, 4, checkpoint_dir=ckpt_dir, stop_on_eos=False)

    result = inspect_decode_checkpoint(ckpt_dir, validate_kv=True)
    assert result["valid"], f"Checkpoint invalid: {result['validation_errors']}"
    assert result["manifest"]["sampler"]["type"] == "greedy"
    assert result["manifest"]["state"]["generated_token_count"] == 4
    assert result["manifest"]["state"]["kv_contains_all_tokens_except_last"] is True
    assert result["snapshot_count"] == 4  # one snapshot per generated token