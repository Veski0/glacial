"""Shared fixtures and configuration for the Glacial test harness.

Slow tests require a cached Hugging Face model.  They are skipped by default
and only run when ``--runslow`` is passed to pytest:

    .venv/bin/python -m pytest tests/ --runslow

All fixtures are session-scoped where possible so that model loading, tokenizer
construction, and safetensors header parsing happen once per test session.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure the project root is on sys.path so ``import glacial`` works when
# running pytest from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
DEFAULT_REVISION = "b0e4fd07be563ba8bb7689c47dc9bebdff5471ab"

LM_HEAD_CHUNK_ROWS = 4096


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow integration tests (requires a cached model)",
    )
    parser.addoption(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID for parity tests",
    )
    parser.addoption(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face model revision for parity tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Session-scoped model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model_id(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--model-id")


@pytest.fixture(scope="session")
def revision(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--revision")


@pytest.fixture(scope="session")
def model_file(model_id: str, revision: str) -> Path:
    """Resolve the safetensors model file from the HF cache."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            revision=revision,
            local_files_only=True,
        )
    )


@pytest.fixture(scope="session")
def config(model_id: str, revision: str) -> dict[str, Any]:
    """Load the model config.json from the HF cache."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=revision,
            local_files_only=True,
        )
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def safetensors_info(model_file: Path) -> tuple[dict[str, Any], int]:
    """Return (safetensors_header, payload_start) for the model file."""
    from glacial.weights import read_safetensors_header

    header_len, header = read_safetensors_header(model_file)
    payload_start = 8 + header_len
    return header, payload_start


@pytest.fixture(scope="session")
def tokenizer(model_id: str, revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
    )


@pytest.fixture(scope="session")
def backend(config: dict[str, Any]):
    """Resolve the Glacial backend for the model config."""
    from glacial.backends import resolve_backend

    return resolve_backend("auto", config=config)


@pytest.fixture(scope="session")
def hf_model(model_id: str, revision: str):
    """Load the HF oracle model (BF16, eager attention, CPU).

    This is the measuring instrument: a standard Hugging Face model instance
    used to produce golden greedy tokens for parity comparison.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Callable fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hf_greedy(hf_model, tokenizer):
    """Return a callable that runs HF greedy decode on a prompt string.

    The oracle recomputes the full forward pass each step (no KV cache) so
    that the result is the purest possible greedy reference — exactly what the
    Glacial probes validated op-by-op.
    """

    import torch

    def _decode(
        prompt_text: str,
        max_new_tokens: int,
        *,
        stop_on_eos: bool = False,
    ) -> list[int]:
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        eos = tokenizer.eos_token_id if stop_on_eos else None
        generated: list[int] = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = hf_model(input_ids)
            logits = out.logits[:, -1, :].float()
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], dtype=input_ids.dtype)],
                dim=1,
            )
            if eos is not None and next_id == eos:
                break
        return generated

    return _decode


@pytest.fixture
def glacial_decode(
    model_file: Path,
    safetensors_info: tuple[dict[str, Any], int],
    config: dict[str, Any],
    backend,
    model_id: str,
    revision: str,
):
    """Return a callable that runs Glacial greedy decode.

    Supports three modes:

    * **Fresh** — pass ``prompt_ids`` and ``max_new_tokens``.
    * **Checkpointed** — additionally pass ``checkpoint_dir`` to persist a
      resumable KV checkpoint after each generated token.
    * **Resume** — pass ``resume_from`` (a checkpoint directory) and
      ``max_new_tokens`` (number of *new* tokens to generate).  ``prompt_ids``
      is ignored in this mode.

    Returns ``(all_token_ids, generated_token_ids)`` where *generated* excludes
    the prompt tokens.
    """

    from glacial.kv import load_decode_checkpoint, save_decode_checkpoint
    from glacial.sampler import Sampler

    header, payload_start = safetensors_info

    def _save_checkpoint(
        checkpoint_dir: Path,
        all_ids: list[int],
        kv_cache,
        prompt_token_count: int,
        sampler: Sampler | None = None,
    ) -> None:
        save_decode_checkpoint(
            run_dir=checkpoint_dir,
            token_ids=all_ids,
            prompt_token_count=prompt_token_count,
            kv_cache=kv_cache,
            model_id=model_id,
            revision=revision,
            model_file=model_file,
            backend_name=backend.name,
            rendered_text=None,
            prompt_mode="test",
            messages=None,
            config=config,
            lm_head_chunk_rows=LM_HEAD_CHUNK_ROWS,
            sampler=sampler.to_manifest() if sampler is not None else None,
        )

    def _decode(
        prompt_ids: list[int] | None = None,
        max_new_tokens: int = 0,
        *,
        checkpoint_dir: Path | None = None,
        resume_from: Path | None = None,
        stop_on_eos: bool = False,
        eos_token_id: int | None = None,
        sampler: Sampler | None = None,
    ) -> tuple[list[int], list[int]]:

        if resume_from is not None:
            # ---- Resume mode ------------------------------------------------
            ckpt = load_decode_checkpoint(resume_from)
            all_ids: list[int] = list(ckpt["token_ids"])
            kv_cache = ckpt["kv_cache"]
            prompt_token_count = int(ckpt["manifest"]["state"]["prompt_token_count"])
            generated_start = len(all_ids)
            if sampler is None:
                sampler = Sampler.from_manifest(ckpt["manifest"].get("sampler", {}))

            for _ in range(max_new_tokens):
                position = len(all_ids) - 1
                next_id, kv_cache, _ = backend.decode_kv_greedy(
                    input_token_id=all_ids[-1],
                    position=position,
                    kv_cache=kv_cache,
                    model_file=model_file,
                    header=header,
                    payload_start=payload_start,
                    config=config,
                    lm_head_chunk_rows=LM_HEAD_CHUNK_ROWS,
                    sampler=sampler,
                )
                all_ids.append(next_id)
                if checkpoint_dir is not None:
                    _save_checkpoint(checkpoint_dir, all_ids, kv_cache, prompt_token_count, sampler)
                if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                    break
        else:
            # ---- Fresh mode -------------------------------------------------
            if prompt_ids is None:
                raise ValueError("prompt_ids is required when not resuming")
            all_ids = list(prompt_ids)
            prompt_token_count = len(prompt_ids)
            kv_cache = None
            generated_start = len(all_ids)

            if max_new_tokens <= 0:
                return all_ids, []

            # Prefill produces the first generated token + prompt KV.
            next_id, kv_cache, _ = backend.prefill_kv_greedy(
                token_ids=all_ids,
                model_file=model_file,
                header=header,
                payload_start=payload_start,
                config=config,
                lm_head_chunk_rows=LM_HEAD_CHUNK_ROWS,
                sampler=sampler,
            )
            all_ids.append(next_id)
            if checkpoint_dir is not None:
                _save_checkpoint(checkpoint_dir, all_ids, kv_cache, prompt_token_count, sampler)
            if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                return all_ids, all_ids[generated_start:]

            # Decode remaining tokens.
            for _ in range(max_new_tokens - 1):
                position = len(all_ids) - 1
                next_id, kv_cache, _ = backend.decode_kv_greedy(
                    input_token_id=all_ids[-1],
                    position=position,
                    kv_cache=kv_cache,
                    model_file=model_file,
                    header=header,
                    payload_start=payload_start,
                    config=config,
                    lm_head_chunk_rows=LM_HEAD_CHUNK_ROWS,
                    sampler=sampler,
                )
                all_ids.append(next_id)
                if checkpoint_dir is not None:
                    _save_checkpoint(checkpoint_dir, all_ids, kv_cache, prompt_token_count, sampler)
                if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                    break

        return all_ids, all_ids[generated_start:]

    return _decode