# Backend Abstraction

Glacial now separates the operator/runtime loop from architecture-specific model math.

Generic runtime concerns:

```text
prompt/token handling
weight budget wiring
prefill/decode loop
checkpoint save/load/resume
inspectable run state
durable-before-visible token emission
```

Backend concerns:

```text
model config support detection
tensor names and layouts
embedding/final-head semantics
layer math
attention/KV behavior
MoE/router/expert behavior
```

## Interface

Backends implement `glacial.backends.base.CausalLMBackend`:

```python
supports_config(config) -> bool
next_token_greedy(...)
prefill_kv_greedy(...)
decode_kv_greedy(...)
```

The CLI resolves a backend with:

```bash
python tools/glacial_generate.py --backend auto ...
```

Currently available:

```text
granite    IBM Granite MoE path proven against HF traces
lfm2       Liquid LFM2.5 MoE (stub — math not yet implemented)
```

## Module layout

The architecture boundary is now clean:

```text
glacial/
  generate.py     shared utilities (embed, input formatting, telemetry)
  logits.py       shared chunked LM head + greedy argmax helpers
  weights.py      safetensors loading + WeightBudget
  sampler.py      token sampler with checkpointable RNG state
  kv.py           checkpoint save/load/resume
  granite.py      Granite MoE layer math (private to backend)
  backends/
    base.py       CausalLMBackend protocol
    granite.py    Granite backend (owns ALL Granite decode logic)
    __init__.py   backend registry / auto resolver
```

The shared modules (``generate.py``, ``logits.py``) have **zero** architecture-specific
imports. The Granite math module (``granite.py``) is only imported by the backend
and by development probe tools. The runtime (CLI, server, tests) talks only to
the ``CausalLMBackend`` protocol.

## Adding a backend

A new backend should add a module under:

```text
glacial/backends/<name>.py
```

Then register it in:

```text
glacial/backends/__init__.py
```

The backend owns all architecture-specific execution: layer math, final norm,
LM head, and the prefill/decode loop. It can reuse the shared utilities in
``generate.py`` and ``logits.py`` but must not leak architecture-specific code
into them.
