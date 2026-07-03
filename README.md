# Glacial

> The model is not loaded. The model is visited.

Glacial is an exact, out-of-core decode prototype for causal language models. It proves that a model can be executed by visiting only the needed BF16 tensors from `safetensors`, instead of instantiating the full Hugging Face model.

The current working backend targets:

```text
ibm-granite/granite-3.1-1b-a400m-instruct
```

Current status: slow, exact, greedy, checkpointable, inspectable, and OpenAI-compatible enough for local tools.

## What works

- exact Granite MoE forward/decode path proven against HF traces
- KV-cache greedy generation
- chunked tied-LM-head greedy argmax
- optional resident-weight budget telemetry/enforcement
- durable resumable KV checkpoints
- checkpoint inspection CLI
- backend abstraction with a Granite adapter
- local OpenAI-compatible `/v1/chat/completions` shim
- integration test harness: greedy parity + resume parity against HF oracle
- sampling (temperature, top-k, top-p) with checkpointable RNG state

## What this is not

Glacial is **not** a high-throughput inference server. It is a reference/runtime prototype for latency-insensitive exact decode. One generated token may be slow; operational correctness and resumability matter more than speed.

Sampling is supported with temperature, top-k, and top-p. The RNG state is persisted in checkpoints, so sampled sequences are reproducible and resumable. ``temperature=0`` (the default) produces greedy argmax.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the test harness:

```bash
python -m pip install -r requirements-dev.txt
```

The first run may download the Granite model from Hugging Face. Add `--local-files-only` once the model is cached.

## CLI generation

```bash
python tools/glacial_generate.py \
  --chat-user "Say hello in three words." \
  --max-new-tokens 8 \
  --show-token-telemetry
```

Checkpointed run:

```bash
python tools/glacial_generate.py \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --checkpoint-dir runs/hello \
  --show-token-telemetry
```

Resume:

```bash
python tools/glacial_generate.py \
  --resume-from runs/hello \
  --max-new-tokens 1 \
  --show-token-telemetry
```

Inspect:

```bash
python tools/inspect_checkpoint.py runs/hello \
  --decode-tokens \
  --show-tokens \
  --local-files-only
```

## OpenAI-compatible local API

Start the server:

```bash
python tools/glacial_openai_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name glacial-granite \
  --local-files-only
```

Call it with curl:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glacial-granite",
    "messages": [{"role": "user", "content": "Say hello in three words."}],
    "max_tokens": 8,
    "temperature": 0
  }'
```

See [`docs/openai-compatible-api.md`](docs/openai-compatible-api.md) for streaming and client examples.

## Test harness

The test suite proves correctness against the HF oracle. Slow tests require a cached model and are skipped by default.

```bash
# fast: just collect and verify skip behavior
python -m pytest tests/ -v

# slow: full integration tests against HF (requires model cache)
python -m pytest tests/ --runslow -v
```

Tests:

- **greedy parity** — Glacial greedy decode == HF greedy decode, token-for-token (KV-cache and no-KV paths)
- **resume parity** — checkpoint + resume == uninterrupted decode (single-cycle and multi-step)
- **sampling** — determinism (same seed → same sequence), resume parity with RNG state, temperature-zero-equals-greedy, top-k/top-p determinism

## Project map

```text
glacial/
  weights.py            safetensors byte-range loading + budget accounting
  granite.py            Granite MoE layer math (private to backend)
  logits.py             shared chunked LM head + greedy argmax helpers
  sampler.py            token sampler with checkpointable RNG state
  generate.py           shared generation utilities (embed, inputs, telemetry)
  kv.py                 durable decode checkpoints
  backends/             backend protocol and Granite adapter

tools/
  glacial_generate.py          CLI generator
  glacial_openai_server.py     OpenAI-compatible local API
  inspect_checkpoint.py        checkpoint inspector
  hf_trace_*.py, probe_*.py    development parity tools

tests/
  conftest.py                 shared fixtures (model loading, decode helpers)
  test_greedy_parity.py       Glacial vs HF greedy parity
  test_resume_parity.py       checkpoint/resume vs uninterrupted
```

## Docs

- [`docs/cli.md`](docs/cli.md) — CLI usage
- [`docs/openai-compatible-api.md`](docs/openai-compatible-api.md) — local API shim
- [`docs/resumable-decode.md`](docs/resumable-decode.md) — checkpoint semantics
- [`docs/backend-abstraction.md`](docs/backend-abstraction.md) — backend boundary
- [`docs/mental-model.md`](docs/mental-model.md) — how to think about Glacial
- [`docs/granite-reference.md`](docs/granite-reference.md) — Granite-specific reference notes
- [`docs/gemma4-reference.md`](docs/gemma4-reference.md) — Gemma 4 architecture reference (WIP)
- [`docs/vision.md`](docs/vision.md) — project direction

## Important invariants

- Greedy means `torch.argmax(logits.float())`, not `topk()[0]`.
- Checkpointed mode is durable-before-visible: if Glacial showed a token, it can resume past that token.
- Decode checkpoints store KV for all tokens except the final token in `token_ids`.

## Caveats

- Current backend: Granite MoE + Gemma 4 (stub).
- Current sampler: greedy + temperature/top-k/top-p with seed-in-persistence.
- No batching or request scheduler.
- The API shim serializes generation through one engine lock.
- The code is a prototype; expect sharp edges.
