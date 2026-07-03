# Vision

Glacial is an exact out-of-core decode runtime for large causal language models.

The target use case is latency-insensitive decode where model residency is impossible or undesirable, but exact BF16 execution is still required.

## Goal

Build a brutally simple executor that refuses to keep unnecessary weights resident.

A normal runtime does this:

```text
load entire model -> run forward
```

Glacial wants this:

```text
visit needed tensors -> compute -> evict -> persist state -> repeat
```

## Current status

The current prototype is operational on:

```text
ibm-granite/granite-3.1-1b-a400m-instruct
```

Implemented:

```text
✅ exact Granite MoE math proven against HF traces
✅ selected-expert MoE execution
✅ KV prefill + cached greedy decode
✅ chunked tied-LM-head greedy argmax
✅ interactive CLI generator
✅ optional resident-weight budget enforcement
✅ durable resumable KV checkpoints
✅ checkpoint inspection CLI
✅ backend abstraction with Granite adapter
✅ local OpenAI-compatible Chat Completions shim
✅ integration test harness (greedy parity + resume parity against HF oracle)
✅ sampling (temperature / top-k / top-p) with checkpointable RNG state
```

Current package shape:

```text
glacial/
  weights.py            safetensors provider + WeightBudget
  granite.py            Granite layer execution
  logits.py             final norm + chunked LM head + greedy
  generate.py           prefill/decode generation helpers
  kv.py                 persistent checkpoint save/load/inspect
  backends/
    base.py             CausalLMBackend protocol
    granite.py          Granite MoE backend adapter
    __init__.py         backend registry / auto resolver
```

## Design principles

1. **Exact first.** No quantization or lossy approximation in the reference path.
2. **Resident memory is explicit.** Weight visits are scoped and budgetable.
3. **Interruption is normal.** Runtime state should be inspectable and resumable.
4. **Architecture math is backend-owned.** The runtime loop should not bake in Granite forever.
5. **Slow is acceptable.** This is not a high-throughput serving system.

## Near-term work

Most useful next steps:

```text
1. Graceful Ctrl-C / pause messaging.
2. Checkpoint cleanup policy, e.g. keep-last N snapshots.
3. ~~Resume parity smoke script / integration test.~~ ✅ Done.
4. ~~Slow integration test harness for probes.~~ ✅ Done (greedy + resume parity).
5. ~~Sampling with exact RNG/state semantics.~~ ✅ Done.
6. Move Granite-specific generation helpers fully under backend module.
7. Add a second architecture backend.
8. Packaging / CLI polish.
```

## Longer-term target

The aspirational target remains a much larger MoE model:

```text
GLM-family BF16 decode
full weights
no quantization
exact sampling semantics
checkpointed runtime state
```

Glacial succeeds if it can run such a model by visiting only the active tensors/experts needed for each token, while preserving enough state to survive interruption.

## Summary

This is not a high-throughput inference server. The OpenAI-compatible endpoint is a local tooling shim over the same serialized exact greedy runtime.

Glacial is a slow, exact, persistent model vessel.

> Born to Matmul.  
> Forced to Page Fault.
