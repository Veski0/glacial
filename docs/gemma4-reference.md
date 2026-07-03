# Gemma 4 E4B Architecture Reference

Model: `google/gemma-4-E4B` (~8B params with embeddings, ~16GB BF16 safetensors)

This document captures the architecture-specific math needed to implement a
Glacial backend for Gemma 4 E4B.  It is the equivalent of
[`granite-reference.md`](granite-reference.md) for the Granite backend.

## Config summary (text decoder only)

```json
{
  "model_type": "gemma4_text",
  "architectures": ["Gemma4ForConditionalGeneration"],
  "hidden_size": 2560,
  "num_hidden_layers": 42,
  "num_attention_heads": 8,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "global_head_dim": 512,
  "intermediate_size": 10240,
  "vocab_size": 262144,
  "tie_word_embeddings": true,
  "rms_norm_eps": 1e-6,
  "hidden_activation": "gelu_pytorch_tanh",
  "final_logit_softcapping": 30.0,
  "sliding_window": 512,
  "num_kv_shared_layers": 18,
  "hidden_size_per_layer_input": 256,
  "vocab_size_per_layer_input": 262144,
  "enable_moe_block": false,
  "attention_bias": false
}
```

The full model is multimodal (`Gemma4ForConditionalGeneration`).  Glacial
targets only the **text decode** path (`Gemma4ForCausalLM` / text_config).

## Key architectural differences from Granite

| Feature | Granite 3.1 1B | Gemma 4 E4B |
|---|---|---|
| Architecture | MoE (32 experts, top-8) | Dense |
| Layers | 24 | 42 |
| Hidden size | 1024 | 2560 |
| Head dim | 64 | 256 (sliding) / 512 (global) |
| KV heads | 8 | 2 (GQA, 4 groups) |
| Attention | Full causal | Hybrid: sliding (512) + global |
| RoPE | Standard, full head_dim | Standard (sliding) + p-RoPE 25% (global) |
| Norm | RMSNorm ( Granite variant) | RMSNorm (Gemma variant) |
| Activation | SiLU (SwiGLU) | GELU tanh (SwiGLU variant) |
| LM head | Tied, /logits_scaling | Tied, tanh softcap |
| Per-layer embeddings | No | Yes (PLE) |
| KV sharing | No | Yes (18 global layers share KV) |
| Embedding multiplier | 12.0 | sqrt(hidden_size) |
| Residual multipliers | 0.22 | None (standard) |

## Hybrid attention

42 layers with alternating pattern: 5 sliding + 1 global, repeating.  The
last layer (index 41) is always global.

```python
layer_types = [
  "sliding_attention",  # 0-4
  "full_attention",     # 5
  "sliding_attention",  # 6-10
  "full_attention",     # 11
  ...                   # repeats 7 times
  "full_attention",     # 41 (last layer)
]
```

- **Sliding layers**: local attention with window=512.  Only attend to the
  previous 512 tokens.  Standard RoPE (theta=10000), full head_dim=256.
- **Global layers**: full causal attention.  p-RoPE (theta=1000000,
  partial_rotary_factor=0.25 → only 64 of 512 dims get RoPE).  head_dim=512.

## Per-Layer Embeddings (PLE)

Each decoder layer receives an auxiliary input from PLE, in addition to the
main hidden state.  PLE has two components summed and scaled by 1/√2:

1. **Token-identity**: lookup `input_ids` in `embed_tokens_per_layer`
   (shape `[vocab_size, num_layers * ple_dim]`), multiply by
   `√(hidden_size_per_layer_input)`, reshape to
   `[batch, seq, num_layers, ple_dim]`.
2. **Context-aware**: project `input_embeds` through `per_layer_model_projection`
   (Linear), scale by `1/√(hidden_size)`, reshape, RMSNorm.

For text-only decode, both components are available.  The per-layer input is
added to the layer's input (after input norm, before attention/MLP).

## Unified KV sharing

`num_kv_shared_layers: 18` — the 7 global attention layers (indices 5, 11, 17,
23, 29, 35, 41) share KV caches.  This means:
- Only one set of K/V tensors for all global layers (not 7 separate caches).
- The KV projection weights may be shared or the cache is reused.
- This affects checkpoint structure: the KV cache list has fewer entries than
  the number of layers.

**TODO**: Study the exact sharing mechanism in the modeling code.  Is it the
same KV projection weights, or the same cache tensor?  This determines how
the checkpoint invariant (KV for all tokens except last) changes.

## RoPE

### Sliding layers (standard RoPE)
- `rope_theta = 10000.0`
- `rope_type = "default"`
- Applied to full `head_dim = 256`

### Global layers (proportional RoPE / p-RoPE)
- `rope_theta = 1000000.0`
- `rope_type = "proportional"`
- `partial_rotary_factor = 0.25` → only 64 of 256 dims get RoPE
  (but `global_head_dim = 512`, so 128 of 512 dims?)

**TODO**: Clarify how p-RoPE interacts with `global_head_dim`.  The
`partial_rotary_factor` applies to the head_dim, but global layers use
`global_head_dim`.  Need to study the exact RoPE application in the modeling
code.

## RMSNorm (Gemma variant)

```python
# Gemma4RMSNorm
def _norm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def forward(x):
    return self._norm(x.float()) * self.weight.float()  # weight applied in fp32
```

Differs from Granite's RMSNorm:
- Granite: `weight * (x.to(input_dtype))` — weight is BF16, multiply in BF16
- Gemma: `norm(x.float()) * weight.float()` — everything in FP32, cast back

## Final logit softcapping

```python
logits = softcap * torch.tanh(logits / softcap)  # softcap = 30.0
```

This replaces Granite's `logits / logits_scaling`.  The chunked LM head
helper (`chunked_last_logits` / `chunked_last_argmax`) currently takes a
`logits_scaling` divisor.  For Gemma, we need a `logits_softcap` parameter
instead (or a general post-processing hook).

## MLP (SwiGLU with GELU)

```python
gate = F.linear(x, gate_weight)
up = F.linear(x, up_weight)
hidden = F.gelu(gate, approximate="tanh") * up
output = F.linear(hidden, down_weight)
```

Granite uses SiLU (`F.silu`).  Gemma uses GELU tanh approximation.

## Embedding

```python
# Gemma4TextScaledWordEmbedding
hidden = F.embedding(input_ids, weight) * sqrt(hidden_size)
```

Granite uses `embedding_multiplier = 12.0`.  Gemma uses `sqrt(hidden_size)`.
For E4B: `sqrt(2560) ≈ 50.6`.

## Tensor naming (estimated)

Based on standard HF naming conventions and Gemma 3 patterns:

```text
model.embed_tokens.weight                      [vocab, hidden]
model.embed_tokens_per_layer.weight            [vocab, num_layers * ple_dim]
model.per_layer_model_projection.weight        [hidden, num_layers * ple_dim]
model.per_layer_projection_norm.weight         [num_layers * ple_dim]
model.norm.weight                              [hidden]  (final norm)

model.layers.{i}.input_layernorm.weight        [hidden]
model.layers.{i}.post_attention_layernorm.weight [hidden]
model.layers.{i}.self_attn.q_proj.weight       [hidden, num_heads * head_dim]
model.layers.{i}.self_attn.k_proj.weight       [hidden, num_kv_heads * head_dim]
model.layers.{i}.self_attn.v_proj.weight       [hidden, num_kv_heads * head_dim]
model.layers.{i}.self_attn.o_proj.weight       [num_heads * head_dim, hidden]
model.layers.{i}.mlp.gate_proj.weight          [hidden, intermediate]
model.layers.{i}.mlp.up_proj.weight            [hidden, intermediate]
model.layers.{i}.mlp.down_proj.weight          [intermediate, hidden]
```

**TODO**: Verify actual tensor names from the safetensors header once the
model is cached.  Global layers may have different Q/K/V projection sizes
(`global_head_dim` vs `head_dim`).

## Blockers / TODOs

1. **transformers version**: Config requires `5.5.0.dev0`.  Installed is
   `4.47.0`.  Must upgrade to use HF oracle for parity testing.
2. **Model download**: ~16GB.  Must cache before first run.
3. **Config nesting**: The text config is nested under `text_config` in the
   full model config.  The backend needs to extract it.
4. **KV sharing**: Understand the exact mechanism for unified KV in global
   layers.  This affects the checkpoint format.
5. **p-RoPE**: Study the exact proportional RoPE implementation for global
   layers, especially the interaction with `global_head_dim`.
6. **Sliding window attention**: Implement the sliding window causal mask
   for local attention layers.
7. **PLE**: Implement per-layer embedding lookup and projection.  This adds
   a large embedding table (`embed_tokens_per_layer`) that must be visited
   per token — a key out-of-core weight visit.
8. **Logit softcap**: Modify the chunked LM head to support tanh softcap
   instead of division scaling.