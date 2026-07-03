"""Token sampler with checkpointable RNG state.

Greedy mode uses argmax (no RNG).  All other modes use a ``torch.Generator``
seeded for deterministic, resumable sampling.  The generator state can be
serialized to/from the checkpoint manifest, extending the durable-before-
visible invariant to sampling:

    If Glacial showed you a sampled token, Glacial can resume past that token
    and reproduce the same sequence.

Internal determinism (same seed -> same sequence, resume reproduces exactly)
is fully achievable.  Matching Hugging Face's exact sampled tokens given the
same seed is *not* guaranteed — HF's RNG seeding and multinomial call pattern
differ from ours.  Greedy parity (which Glacial *does* match exactly) remains
the gold standard.
"""

from __future__ import annotations

import base64
from typing import Any


class Sampler:
    """Token sampler supporting greedy, temperature, top-k, and top-p.

    Parameters
    ----------
    type : str
        ``"greedy"`` (argmax, no RNG) or ``"sample"`` (multinomial with
        optional temperature / top-k / top-p filtering).
    temperature : float
        Logit temperature.  Only meaningful for ``type="sample"``.
    top_k : int | None
        If set, keep only the top-k logits before sampling.
    top_p : float | None
        If set, keep only the nucleus of probability mass ``top_p``.
    seed : int | None
        RNG seed.  Ignored for greedy.  If ``generator`` is also provided,
        ``seed`` is informational only (stored in the manifest for reference).
    generator : torch.Generator | None
        Pre-constructed generator (e.g. restored from a checkpoint).  Takes
        precedence over ``seed``.
    """

    def __init__(
        self,
        *,
        type: str = "greedy",
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        generator: Any | None = None,
    ) -> None:
        self.type = type
        self.temperature = float(temperature) if temperature is not None else None
        self.top_k = int(top_k) if top_k is not None else None
        self.top_p = float(top_p) if top_p is not None else None
        self.seed = seed

        if type == "greedy":
            self.generator = None
        elif generator is not None:
            self.generator = generator
        else:
            import torch

            self.generator = torch.Generator()
            self.seed = seed if seed is not None else 0
            self.generator.manual_seed(self.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_greedy(self) -> bool:
        """Return True if this sampler uses argmax (no RNG)."""
        return self.type == "greedy"

    def sample(self, logits) -> int:
        """Select a token id from a 1-D logit tensor.

        For greedy: ``argmax(logits.float())`` — no RNG, no full-probs
        materialization needed.

        For sampling: apply temperature → top-k → top-p → softmax →
        ``torch.multinomial`` with this sampler's generator.
        """
        import torch

        if self.type == "greedy":
            return int(torch.argmax(logits.float()).item())

        logits = logits.float()

        # Temperature scaling
        if self.temperature is not None and self.temperature > 0 and self.temperature != 1.0:
            logits = logits / self.temperature

        # Top-k filtering: keep only the k highest logits
        if self.top_k is not None and self.top_k > 0:
            k = min(self.top_k, logits.size(-1))
            top_values, _ = torch.topk(logits, k)
            threshold = top_values[-1]
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Top-p (nucleus) filtering: keep the smallest set of tokens whose
        # cumulative probability exceeds top_p
        if self.top_p is not None and self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_mask = cumulative_probs > self.top_p
            # Shift right by one so the first token *above* the threshold
            # is kept (standard HF semantics).
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False
            mask = sorted_mask.scatter(0, sorted_indices, sorted_mask)
            logits = logits.masked_fill(mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        return int(
            torch.multinomial(probs, num_samples=1, generator=self.generator).item()
        )

    # ------------------------------------------------------------------
    # Checkpoint serialization
    # ------------------------------------------------------------------

    def get_rng_state(self) -> str | None:
        """Return base64-encoded generator state, or None for greedy."""
        if self.generator is None:
            return None
        import numpy as np

        state = self.generator.get_state()
        return base64.b64encode(np.asarray(state).tobytes()).decode("ascii")

    def to_manifest(self) -> dict[str, Any]:
        """Return a dict suitable for the checkpoint manifest's ``sampler`` field."""
        return {
            "type": self.type,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
            "rng_state": self.get_rng_state(),
        }

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "Sampler":
        """Reconstruct a sampler from a checkpoint manifest's ``sampler`` dict.

        For greedy checkpoints (``type == "greedy"`` or missing), returns a
        greedy sampler with no generator.

        For sampling checkpoints, restores the ``torch.Generator`` state from
        the base64 ``rng_state`` field so that resumed generation reproduces
        the exact same sequence.
        """
        import torch
        import numpy as np

        sampler_type = manifest.get("type", "greedy")

        if sampler_type == "greedy":
            return cls(type="greedy")

        generator = None
        rng_state = manifest.get("rng_state")
        if rng_state is not None:
            generator = torch.Generator()
            state_bytes = base64.b64decode(rng_state)
            state_tensor = torch.from_numpy(
                np.frombuffer(state_bytes, dtype=np.uint8).copy()
            )
            generator.set_state(state_tensor)

        return cls(
            type=sampler_type,
            temperature=manifest.get("temperature"),
            top_k=manifest.get("top_k"),
            top_p=manifest.get("top_p"),
            seed=manifest.get("seed"),
            generator=generator,
        )

    @classmethod
    def from_params(
        cls,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> "Sampler":
        """Construct a sampler from user-facing parameters.

        ``temperature=0`` with no ``top_k``/``top_p`` produces a greedy sampler
        (matching OpenAI API semantics).  Any non-zero temperature or explicit
        ``top_k``/``top_p`` enables sampling mode.
        """
        if temperature <= 0.0 and top_k is None and top_p is None:
            return cls(type="greedy")

        # temperature=0 with top_k/top_p is equivalent to temperature=1.0
        # (no logit scaling) but with filtering applied.
        effective_temp = temperature if temperature > 0 else 1.0
        return cls(
            type="sample",
            temperature=effective_temp,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )