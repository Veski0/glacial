# LFM2.5-8B-A1B Architecture Reference

Model: `LiquidAI/LFM2.5-8B-A1B` (8.3B total / 1.5B active, ~16.9GB BF16 safetensors)

This is the second Glacial backend target.  It stays on the MoE path
(Granite is MoE, LFM2.5 is MoE, the long-horizon GLM target is MoE).

## Config summary

```json
{
  "model_type": "lfm2_moe",
  "architectures": ["Lfm2MoeForCausalLM"],
  "hidden_size": 2048,
  "num_hidden_layers": 24,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 64,
  "intermediate_size": 7168,
  "moe_intermediate_size": 1792,
  "num_experts": 32,
  "num_experts_per_tok": 4,
  "num_dense_layers": 2,
  "vocab_size": 128000,
  "tie_word_embeddings": true,
  "norm_eps": 1e-5,
  "norm_topk_prob": true,
  "routed_scaling_factor": 1.0,
  "use_expert_bias": true,
  "conv_L_cache": 3,
  "conv_bias": false,
  "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"},
  "max_position_embeddings": 128000
}
```

## Architecture overview

LFM2.5 is a **hybrid conv/attention MoE** architecture:

```text
24 layers = 18 conv layers + 6 attention layers
            ─────────────────  ──────────────────
            short 1D conv       GQA with RoPE
            no KV cache         KV cache
            cheap, local         global context

First 2 layers: dense MLP (intermediate_size=7168)
Layers 2-23:    MoE (32 experts, 4 active, moe_intermediate_size=1792)

layer_types = [
  "conv", "conv", "full_attention",      # 0-2
  "conv", "conv", "conv", "full_attention",  # 3-6
  "conv", "conv", "conv", "full_attention",  # 7-10
  "conv", "conv", "conv", "full_attention",  # 11-14
  "conv", "conv", "conv", "full_attention",  # 15-18
  "conv", "conv", "full_attention",          # 19-21
  "conv", "conv"                             # 22-23
]
```

## Comparison with Granite

| Feature | Granite 3.1 1B | LFM2.5-8B-A1B |
|---|---|---|
| **Type** | MoE | MoE (hybrid conv/attention) |
| Experts | 32, top-8 | 32, top-4 |
| Expert activation | SiLU | SiLU (same) |
| Expert weight layout | Separate input_linear, output_linear | Combined gate_up_proj, down_proj |
| Router | Softmax top-k | Sigmoid + expert_bias + top-k |
| Dense layers | None | First 2 layers (indices 0-1) |
| Attention layers | All 24 | 6 of 24 (indices 2,6,10,14,18,21) |
| Conv layers | None | 18 of 24 |
| KV cache | All 24 layers | Only 6 attention layers |
| Q/K layernorm | No | Yes (RMSNorm on Q and K) |
| Embedding multiplier | 12.0 | None (standard) |
| Residual multiplier | 0.22 | None (standard) |
| Attention multiplier | 0.015625 | None (standard 1/sqrt(d)) |
| Logits scaling | /6.0 | None (standard) |
| RoPE theta | 1,500,000 | 5,000,000 |
| head_dim | 64 | 64 (same!) |
| hidden_size | 1024 | 2048 |
| vocab_size | 256K | 128K |
| Final norm | model.norm.weight | model.embedding_norm.weight |
| O proj name | self_attn.o_proj | self_attn.out_proj |

## Decoder layer structure

```python
# Lfm2MoeDecoderLayer.forward
residual = hidden_states
if is_attention_layer:
    hidden_states, _ = self_attn(operator_norm(hidden_states), ...)
else:
    hidden_states = self.conv(operator_norm(hidden_states), ...)
hidden_states = hidden_states + residual           # no multiplier
hidden_states = hidden_states + feed_forward(ffn_norm(hidden_states))  # no multiplier
```

No `residual_multiplier` or `attention_multiplier` — standard pre-norm residuals.

## Attention (6 layers)

Standard GQA with RoPE, but with **Q/K layernorm**:

```python
query_states = q_layernorm(q_proj(hidden).view(..., head_dim)).transpose(1, 2)
key_states = k_layernorm(k_proj(hidden).view(..., head_dim)).transpose(1, 2)
value_states = v_proj(hidden).view(..., head_dim).transpose(1, 2)

# Standard RoPE (theta=5,000,000)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# Standard attention: Q @ K^T * (1/sqrt(head_dim))
attn_weights = softmax(Q @ K^T * scaling + mask)
attn_output = attn_weights @ V
output = out_proj(attn_output)
```

New vs Granite:
- **Q/K RMSNorm** after projection, before RoPE
- **out_proj** (not `o_proj`)
- No `attention_multiplier` — standard `1/sqrt(head_dim)`
- RoPE theta is 5M (vs Granite's 1.5M)

## Short conv (18 layers)

Liquid's signature — a gated depthwise 1D convolution replacing attention:

```python
# Lfm2MoeShortConv.forward (slow_forward path)
BCx = in_proj(hidden).transpose(-1, -2)   # [hidden, 3, seq]
B, C, x = BCx.chunk(3, dim=-2)             # three gates

Bx = B * x                                  # input gate
conv_out = conv1d(Bx, kernel=3, groups=hidden)  # depthwise short conv
y = C * conv_out                             # output gate
y = out_proj(y.transpose(-1, -2))
```

For **decode** (single token): maintains a conv state cache of the last
`L_cache - 1 = 2` tokens' Bx values.  The conv1d is a depthwise (groups=hidden)
1D conv with kernel_size=3.

No KV cache needed — the conv state is just `hidden_size * 2` floats per layer.

## MoE router (layers 2-23)

```python
# Lfm2MoeTopKRouter
router_logits = F.linear(hidden, router_weight)
routing_weights = router_logits.sigmoid()           # sigmoid, not softmax!
if use_expert_bias:
    scores = routing_weights + expert_bias           # additive bias
    _, selected = torch.topk(scores, k=4)
    routing_weights = gather(routing_weights, selected)
else:
    routing_weights, selected = torch.topk(routing_weights, k=4)

if norm_topk_prob:
    routing_weights = routing_weights / (routing_weights.sum(-1, keepdim=True) + 1e-6)
routing_weights = routing_weights * routed_scaling_factor  # 1.0
```

Key differences from Granite:
- **Sigmoid** activation (not softmax on full logits)
- **Expert bias**: additive bias per expert before top-k
- **norm_topk_prob**: normalize selected probs (Granite uses softmax on selected)
- Router weight shape: `[num_experts, hidden]` (same as Granite)

## MoE experts

```python
# Lfm2MoeExperts
gate_up_proj: [num_experts, 2 * moe_intermediate, hidden]  # combined gate+up
down_proj:    [num_experts, hidden, moe_intermediate]

# Forward (per selected expert):
gate, up = F.linear(x, gate_up_proj[expert_idx]).chunk(2, dim=-1)
hidden = F.silu(gate) * up
output = F.linear(hidden, down_proj[expert_idx])
output = output * routing_weights
```

Key difference from Granite:
- **Combined** `gate_up_proj` (gate and up in one tensor) vs Granite's separate `input_linear`
- The expert weight is `[num_experts, 2*intermediate, hidden]` — a 3D tensor
- SiLU activation (same as Granite)
- No second output multiplier (Granite splits into first_half/second_half; LFM2 chunks into gate/up)

## Dense MLP (layers 0-1)

```python
# Lfm2MoeMLP
w1: [intermediate, hidden]  # gate
w3: [intermediate, hidden]  # up
w2: [hidden, intermediate]  # down

output = w2(silu(w1(x)) * w3(x))
```

Standard SwiGLU.  Same as Granite's expert math, just not routed.

## Final norm + LM head

```python
hidden = embedding_norm(hidden)      # model.embedding_norm.weight
logits = lm_head(hidden)             # tied to embed_tokens.weight
```

No `logits_scaling` — raw logits.  The chunked LM head helpers
(`chunked_last_argmax`, `chunked_last_logits`) can be used with
`logits_scaling=1.0`.

## Verified tensor names (from safetensors header inspection)

Total tensors: 2302.  All BF16 except `expert_bias` which is F32.

```text
model.embed_tokens.weight                        [128000, 2048]   BF16
model.embedding_norm.weight                       [2048]          BF16

# Dense conv layer (layers 0-1):
model.layers.{i}.operator_norm.weight             [2048]          BF16
model.layers.{i}.ffn_norm.weight                  [2048]          BF16
model.layers.{i}.conv.in_proj.weight              [6144, 2048]    BF16  (3 * hidden)
model.layers.{i}.conv.out_proj.weight             [2048, 2048]    BF16
model.layers.{i}.conv.conv.weight                 [2048, 1, 3]    BF16  (depthwise, kernel=3)
model.layers.{i}.feed_forward.w1.weight           [7168, 2048]    BF16  (dense MLP gate)
model.layers.{i}.feed_forward.w3.weight           [7168, 2048]    BF16  (dense MLP up)
model.layers.{i}.feed_forward.w2.weight           [2048, 7168]    BF16  (dense MLP down)

# Attention + MoE layer (layers 2,6,10,14,18,21):
model.layers.{i}.operator_norm.weight             [2048]          BF16
model.layers.{i}.ffn_norm.weight                  [2048]          BF16
model.layers.{i}.self_attn.q_proj.weight          [2048, 2048]    BF16  (32 heads * 64)
model.layers.{i}.self_attn.k_proj.weight          [512, 2048]     BF16  (8 kv heads * 64)
model.layers.{i}.self_attn.v_proj.weight          [512, 2048]     BF16
model.layers.{i}.self_attn.out_proj.weight        [2048, 2048]    BF16
model.layers.{i}.self_attn.q_layernorm.weight     [64]            BF16
model.layers.{i}.self_attn.k_layernorm.weight     [64]            BF16
model.layers.{i}.feed_forward.gate.weight         [32, 2048]      BF16  (router)
model.layers.{i}.feed_forward.expert_bias         [32]            F32   (per-expert bias)
model.layers.{i}.feed_forward.experts.{j}.w1.weight  [1792, 2048]  BF16  (per expert)
model.layers.{i}.feed_forward.experts.{j}.w3.weight  [1792, 2048]  BF16
model.layers.{i}.feed_forward.experts.{j}.w2.weight  [2048, 1792]  BF16

# Conv + MoE layer (layers 3,4,5,7,8,9,...):
model.layers.{i}.operator_norm.weight             [2048]          BF16
model.layers.{i}.ffn_norm.weight                  [2048]          BF16
model.layers.{i}.conv.in_proj.weight              [6144, 2048]    BF16
model.layers.{i}.conv.out_proj.weight             [2048, 2048]    BF16
model.layers.{i}.conv.conv.weight                 [2048, 1, 3]    BF16
model.layers.{i}.feed_forward.gate.weight         [32, 2048]      BF16
model.layers.{i}.feed_forward.expert_bias         [32]            F32
model.layers.{i}.feed_forward.experts.{j}.w1.weight  [1792, 2048]  BF16
model.layers.{i}.feed_forward.experts.{j}.w3.weight  [1792, 2048]  BF16
model.layers.{i}.feed_forward.experts.{j}.w2.weight  [2048, 1792]  BF16
```

**Key finding**: Experts are stored as **separate per-expert w1/w2/w3 tensors**
(like Granite's layout), NOT as the combined 3D `gate_up_proj`/`down_proj`
that the HF `Lfm2MoeExperts` class uses internally.  This is actually closer
to Granite's layout than expected.

**Expert bias is F32**, not BF16 — important for exact reproduction.

## What we can reuse from shared code

- `chunked_last_argmax` / `chunked_last_logits` — tied LM head, `logits_scaling=1.0`
- `embed_input_ids` — with `embedding_multiplier=1.0` (no scaling)
- `make_inputs`, `make_decode_inputs`, `flatten_ids`, `add_budget_telemetry` — all generic
- `SafetensorsWeights`, `WeightBudget` — generic
- `Sampler` — generic
- `rotate_half`, `apply_rotary_pos_emb`, `repeat_kv` — can be copied from granite.py or shared

## What's new

1. **Short conv layers** — gated depthwise conv1d with conv state cache
2. **Conv state persistence** — conv layers need state in checkpoints (not KV)
3. **Q/K layernorm** — RMSNorm on Q and K after projection
4. **Sigmoid router** with expert bias — different routing math
5. **Separate per-expert w1/w2/w3** — the safetensors stores experts as
   individual tensors (like Granite), NOT the combined 3D gate_up_proj/
   down_proj that the HF modeling code uses internally. This is actually
   closer to Granite's layout than expected.
6. **Dense MLP layers** — first 2 layers use standard MLP, not MoE
7. **No multipliers** — simpler residual structure
8. **Checkpoint format** — conv layers have state, not KV; the KV invariant
   changes: only 6 layers have KV cache, 18 have conv state

## Implementation plan (one operation at a time, same as Granite)

1. Safetensors header inspection — verify tensor names and shapes
2. Embedding (no multiplier) + final norm
3. RMSNorm (same as Granite)
4. Dense MLP (SwiGLU, layers 0-1)
5. Short conv (gated conv1d + conv state)
6. Attention with Q/K layernorm + RoPE
7. MoE router (sigmoid + expert_bias + top-k)
8. MoE experts (combined gate_up_proj, SiLU)
9. Final norm + chunked tied LM head (no softcap)
10. Prefill / decode loop (conv state + KV cache)
11. Checkpoint format (conv state + KV for attention layers only)
12. Parity probes against HF traces

## Blockers

1. **transformers version**: Config requires 5.9.0 (installed: 4.47.0)
2. **~16.9GB model download**
3. **Conv state in checkpoints**: The checkpoint format needs to store
   conv state for 18 layers AND KV cache for 6 layers
4. **Expert weight layout**: separate per-expert w1/w2/w3 2D tensors (like
   Granite, NOT the combined 3D tensors the HF modeling code uses). The
   `SafetensorsWeights.expert_slice` method should work with minor naming
   changes. Expert bias is F32, not BF16.