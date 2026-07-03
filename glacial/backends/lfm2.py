"""LFM2 MoE backend stub for Glacial.

This backend targets LiquidAI/LFM2.5-8B-A1B — a hybrid conv/attention MoE
model with 32 experts (4 active).  The architecture reference is in
``docs/lfm2-reference.md``.

Implementation plan (one operation at a time, same as Granite):

1. Safetensors header inspection — verify tensor names and shapes
2. Embedding (no multiplier) + final norm
3. RMSNorm (same variant as Granite)
4. Dense MLP (SwiGLU, layers 0-1)
5. Short conv (gated depthwise conv1d + conv state)
6. Attention with Q/K layernorm + RoPE
7. MoE router (sigmoid + expert_bias + top-k)
8. MoE experts (combined gate_up_proj, SiLU)
9. Final norm + chunked tied LM head (no softcap, logits_scaling=1.0)
10. Prefill / decode loop (conv state + KV cache for 6 attention layers)
11. Checkpoint format (conv state + KV for attention layers only)
12. Parity probes against HF traces
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glacial.sampler import Sampler
from glacial.weights import WeightBudget


class Lfm2MoeBackend:
    """Backend for Liquid LFM2.5 MoE models.

    Status: stub.  Math not yet implemented.  See docs/lfm2-reference.md.
    """

    name = "lfm2"

    def supports_config(self, config: dict[str, Any]) -> bool:
        model_type = str(config.get("model_type", "")).lower()
        architectures = [str(x).lower() for x in config.get("architectures", [])]
        return (
            model_type == "lfm2_moe"
            or model_type == "lfm2"
            or any("lfm2" in arch for arch in architectures)
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
        raise NotImplementedError("LFM2 MoE backend is not yet implemented")

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
        raise NotImplementedError("LFM2 MoE backend is not yet implemented")

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
        raise NotImplementedError("LFM2 MoE backend is not yet implemented")