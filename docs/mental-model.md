# Glacial Mental Model

Glacial separates two worlds:

```text
HF oracle                     Glacial runtime
---------                     ---------------
loads full model              never loads full model
runs reference forward         visits needed weights
emits golden traces            recomputes model math manually
expensive but trusted          exact, bounded, resumable
```

The Hugging Face path is the measuring instrument. The Glacial path is the runtime.

## Core claim

```text
bytes on disk + config + tokens + backend math
= the same next token Hugging Face greedy decode would choose
```

For the Granite prototype, this was built one operation at a time: embeddings, norms, Q/K/V, RoPE, attention, routing, selected experts, final logits, chunked LM head, KV decode, and resume checkpoints.

The historical proof notes are under [`archive/`](archive/).

## Runtime loop

At a high level, Glacial does this:

```text
load tokenizer/config
resolve architecture backend
read safetensors metadata
prefill prompt once
for each generated token:
  visit only needed weight tensors/slices
  compute exact backend forward/decode math
  update KV cache
  choose next token by argmax
  optionally persist checkpoint
  print/return token
```

Important: the model is not instantiated as a resident `torch.nn.Module`. Weights are read from `model.safetensors` by byte range or row/expert slice.

## Granite flow

For Granite 3.1 1B A400M Instruct, a forward pass is:

```text
token ids
  -> embedding rows
  -> * embedding_multiplier

for each layer:
  residual = hidden
  hidden_norm = RMSNorm(hidden)
  attention_out = self_attention(hidden_norm)
  hidden = residual + attention_out * residual_multiplier

  residual = hidden
  hidden_norm = RMSNorm(hidden)
  router_logits = router(hidden_norm)
  selected_experts = top_k(router_logits, k=8)
  moe_out = selected expert MLPs(hidden_norm)
  hidden = residual + moe_out * residual_multiplier

hidden = final_RMSNorm(hidden)
logits = tied_embedding_matrix(hidden) / logits_scaling
next_token = argmax(logits.float())
```

Key Granite scalars:

```text
embedding_multiplier = 12.0
attention_multiplier = 0.015625
residual_multiplier  = 0.22
logits_scaling       = 6.0
```

See [`granite-reference.md`](granite-reference.md) for more Granite-specific details.

## Why checkpointing is central

Glacial assumes interruption is normal. Decode may be slow enough that recomputing from the beginning is unacceptable.

Checkpointed generation preserves this product guarantee:

```text
If Glacial showed you a token, Glacial can resume past that token.
```

It does that by saving the checkpoint before making the token visible. See [`resumable-decode.md`](resumable-decode.md).

## Backend boundary

The outer runtime owns generic operator concerns:

```text
prompt/token handling
weight budget wiring
prefill/decode loop
checkpoint save/load/resume
API/CLI presentation
```

A backend owns model-specific concerns:

```text
config support detection
tensor names and shapes
attention/KV semantics
MoE/router/expert math
embedding/final-head behavior
```

Current backend: `granite`. See [`backend-abstraction.md`](backend-abstraction.md).

## Mantra

> The model is not loaded. The model is visited.
