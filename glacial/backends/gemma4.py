"""Gemma 4 backend stub for Glacial.

This backend is under development.  The architecture reference is in
``docs/gemma4-reference.md``.

Implementation plan (one operation at a time, same as Granite):

1. Safetensors header inspection — verify tensor names and shapes
2. Embedding + PLE lookup
3. RMSNorm (Gemma variant)
4. Q/K/V projections (sliding vs global head dims)
5. RoPE (standard for sliding, p-RoPE for global)
6. Sliding window attention + global attention
7. KV sharing for global layers
8. GELU SwiGLU MLP
9. Final norm + logit softcap + chunked tied LM head
10. Prefill / decode loop
11. Parity probes against HF traces
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glacial.sampler import Sampler
from glacial.weights import WeightBudget


class Gemma4Backend:
    """Backend for Google Gemma 4 dense models (text decode path).

    Status: stub.  Math not yet implemented.
    """

    name = "gemma4"

    def supports_config(self, config: dict[str, Any]) -> bool:
        # The full model config nests text_config; also handle bare text configs.
        text_config = config.get("text_config", config)
        model_type = str(text_config.get("model_type", "")).lower()
        architectures = [str(x).lower() for x in config.get("architectures", [])]
        return (
            model_type == "gemma4_text"
            or model_type == "gemma4"
            or any("gemma4" in arch for arch in architectures)
        )

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
        raise NotImplementedError("Gemma 4 backend is not yet implemented")

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
        raise NotImplementedError("Gemma 4 backend is not yet implemented")

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
        raise NotImplementedError("Gemma 4 backend is not yet implemented")