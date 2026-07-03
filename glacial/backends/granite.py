"""Granite MoE backend for Glacial."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glacial.generate import decode_kv_greedy, next_token_greedy, prefill_kv_greedy
from glacial.sampler import Sampler
from glacial.weights import WeightBudget


class GraniteMoeBackend:
    """Backend adapter for IBM Granite MoE causal language models.

    The math still lives in ``glacial.generate`` / ``glacial.granite`` for now;
    this adapter is the architecture boundary used by runtime CLIs.
    """

    name = "granite"

    def supports_config(self, config: dict[str, Any]) -> bool:
        architectures = [str(x).lower() for x in config.get("architectures", [])]
        model_type = str(config.get("model_type", "")).lower()
        return model_type == "granitemoe" or any("granitemoe" in arch for arch in architectures)

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
        sampler: Sampler | None = None,
    ) -> tuple[int, dict[str, Any]]:
        return next_token_greedy(
            token_ids=token_ids,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            config=config,
            lm_head_chunk_rows=lm_head_chunk_rows,
            budget=budget,
            sampler=sampler,
        )

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
        sampler: Sampler | None = None,
    ) -> tuple[int, list[tuple[Any, Any]], dict[str, Any]]:
        return prefill_kv_greedy(
            token_ids=token_ids,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            config=config,
            lm_head_chunk_rows=lm_head_chunk_rows,
            budget=budget,
            sampler=sampler,
        )

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
        sampler: Sampler | None = None,
    ) -> tuple[int, list[tuple[Any, Any]], dict[str, Any]]:
        return decode_kv_greedy(
            input_token_id=input_token_id,
            position=position,
            kv_cache=kv_cache,
            model_file=model_file,
            header=header,
            payload_start=payload_start,
            config=config,
            lm_head_chunk_rows=lm_head_chunk_rows,
            budget=budget,
            sampler=sampler,
        )
