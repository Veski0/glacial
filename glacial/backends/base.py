"""Backend interface for Glacial causal-LM executors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from glacial.weights import WeightBudget

try:
    from glacial.sampler import Sampler
except ImportError:  # allow standalone use without the sampler module
    Sampler = None  # type: ignore[assignment, misc]


class CausalLMBackend(Protocol):
    """Architecture-specific execution hooks used by generic runtimes.

    A backend owns model math and tensor naming. The outer runtime owns the
    operator loop: prompt handling, checkpointing, resume, telemetry display,
    and budget wiring.
    """

    name: str

    def supports_config(self, config: dict[str, Any]) -> bool:
        """Return true if this backend can execute the given HF config."""

    def next_token_greedy(
        self,
        *,
        token_ids: list[int],
        model_file: Path,
        header: dict[str, Any],
        payload_start: int,
        config: dict[str, Any],
        lm_head_chunk_rows: int,
        budget: WeightBudget | None = None,
        sampler: "Sampler | None" = None,
    ) -> tuple[int, dict[str, Any]]:
        """No-KV fallback: recompute the full prompt and return greedy token."""

    def prefill_kv_greedy(
        self,
        *,
        token_ids: list[int],
        model_file: Path,
        header: dict[str, Any],
        payload_start: int,
        config: dict[str, Any],
        lm_head_chunk_rows: int,
        budget: WeightBudget | None = None,
        sampler: "Sampler | None" = None,
    ) -> tuple[int, list[tuple[Any, Any]], dict[str, Any]]:
        """Prefill prompt KV and return the first generated token."""

    def decode_kv_greedy(
        self,
        *,
        input_token_id: int,
        position: int,
        kv_cache: list[tuple[Any, Any]],
        model_file: Path,
        header: dict[str, Any],
        payload_start: int,
        config: dict[str, Any],
        lm_head_chunk_rows: int,
        budget: WeightBudget | None = None,
        sampler: "Sampler | None" = None,
    ) -> tuple[int, list[tuple[Any, Any]], dict[str, Any]]:
        """Decode one token against existing KV and return next token."""
