"""Backend registry for Glacial architecture executors."""

from __future__ import annotations

from typing import Any

from glacial.backends.base import CausalLMBackend
from glacial.backends.lfm2 import Lfm2MoeBackend
from glacial.backends.granite import GraniteMoeBackend

_BACKENDS: tuple[CausalLMBackend, ...] = (GraniteMoeBackend(), Lfm2MoeBackend())


def available_backends() -> tuple[CausalLMBackend, ...]:
    return _BACKENDS


def backend_names() -> list[str]:
    return [backend.name for backend in _BACKENDS]


def resolve_backend(name: str, *, config: dict[str, Any]) -> CausalLMBackend:
    """Resolve a backend by name or auto-detect it from config."""

    if name == "auto":
        matches = [backend for backend in _BACKENDS if backend.supports_config(config)]
        if not matches:
            model_type = config.get("model_type")
            architectures = config.get("architectures")
            raise SystemExit(
                "No Glacial backend supports this model config "
                f"(model_type={model_type!r}, architectures={architectures!r}). "
                f"Available backends: {', '.join(backend_names())}"
            )
        if len(matches) > 1:
            raise SystemExit(f"Multiple Glacial backends matched config: {[backend.name for backend in matches]}")
        return matches[0]

    for backend in _BACKENDS:
        if backend.name == name:
            if not backend.supports_config(config):
                raise SystemExit(f"Backend {name!r} does not support this model config")
            return backend

    raise SystemExit(f"Unknown Glacial backend {name!r}. Available backends: auto, {', '.join(backend_names())}")
