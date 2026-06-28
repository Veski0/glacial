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
```

## Adding a backend

A future backend should add a module under:

```text
glacial/backends/<name>.py
```

Then register it in:

```text
glacial/backends/__init__.py
```

The existing Granite backend is intentionally thin for now: it adapts the proven `glacial.generate` / `glacial.granite` implementation to the backend interface. Future cleanup can move Granite-specific generation helpers under `glacial/backends/granite.py` without changing the runtime call sites.
