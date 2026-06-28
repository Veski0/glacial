# Granite 3.1 1B A400M Instruct Map

> First atlas for Glacial's exact out-of-core prototype.
>
> The model is not loaded. The model is visited.

## Target checkpoint

```text
model id:        ibm-granite/granite-3.1-1b-a400m-instruct
revision:        b0e4fd07be563ba8bb7689c47dc9bebdff5471ab
architecture:    GraniteMoeForCausalLM
model type:      granitemoe
weight dtype:    BF16
license:         Apache-2.0
```

Archived machine-readable tensor inventory:

```text
docs/archive/reference/granite-3.1-1b-a400m-instruct-tensors.jsonl
```

The inventory was generated from the safetensors header only. It is kept as a development reference, not as part of the quick-start path.

## Config facts

From `config.json`:

| field | value |
|---|---:|
| `num_hidden_layers` | 24 |
| `hidden_size` | 1024 |
| `vocab_size` | 49155 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 64 |
| `num_local_experts` | 32 |
| `num_experts_per_tok` | 8 |
| `intermediate_size` | 512 |
| `max_position_embeddings` | 131072 |
| `rope_theta` | 1500000.0 |
| `rms_norm_eps` | 1e-6 |
| `tie_word_embeddings` | true |

Granite-specific scalars:

| scalar | value | where used |
|---|---:|---|
| `embedding_multiplier` | 12.0 | input embeddings are multiplied before layer 0 |
| `attention_multiplier` | 0.015625 | attention score scale |
| `residual_multiplier` | 0.22 | attention/MoE residual branches |
| `logits_scaling` | 6.0 | logits are divided before output |

## Safetensors file

```text
file:                model.safetensors
file size:           2,669,283,096 bytes
safetensors header:  26,384 bytes
payload start:       26,392 bytes
tensor records:      218
tensor data bytes:   2,669,256,704 bytes
parameters:          1,334,628,352 BF16 values
```

`data_offsets` in the JSONL are safetensors payload-relative offsets. Actual file byte ranges are:

```text
file_offset = 8 + header_len + data_offset
```

The JSONL already includes both `data_offsets` and `file_offsets`.

## Weight inventory summary

| component | tensors | bytes | MiB |
|---|---:|---:|---:|
| embedding / tied LM head | 1 | 100,669,440 | 96.006 |
| attention weights | 96 | 150,994,944 | 144.000 |
| all MoE input weights | 24 | 1,610,612,736 | 1536.000 |
| all MoE output weights | 24 | 805,306,368 | 768.000 |
| routers | 24 | 1,572,864 | 1.500 |
| layer norms | 48 | 98,304 | 0.094 |
| final norm | 1 | 2,048 | 0.002 |

Largest individual tensors:

| tensor | shape | MiB |
|---|---:|---:|
| `model.embed_tokens.weight` | `[49155, 1024]` | 96.006 |
| `model.layers.N.block_sparse_moe.input_linear.weight` | `[32, 1024, 1024]` | 64.000 |
| `model.layers.N.block_sparse_moe.output_linear.weight` | `[32, 1024, 512]` | 32.000 |

## Per-layer shape and budget

For every decoder layer `N`:

| tensor group | shapes | MiB |
|---|---|---:|
| attention | q `[1024,1024]`, k `[512,1024]`, v `[512,1024]`, o `[1024,1024]` | 6.000 |
| router | `[32,1024]` | 0.0625 |
| norms | two `[1024]` tensors | 0.0039 |
| all input experts | `[32,1024,1024]` | 64.000 |
| all output experts | `[32,1024,512]` | 32.000 |
| full layer total | all above | 102.066 |

Under a 128M loaded-weight budget, a full Granite layer fits:

```text
full layer = 107,024,384 bytes = 102.066 MiB
128M       = 128,000,000 bytes, or 128 MiB if interpreted binary
```

This gives us two useful schedules:

1. **Simple parity schedule:** load one whole layer, compute, evict. This is easier and still proves the model is visited rather than resident.
2. **MoE-faithful schedule:** load router first, then only selected expert slices. This is the path that generalizes to the monster.

## Expert slicing

Each layer has two expert tensors:

```text
input_linear.weight:  [32, 1024, 1024]
output_linear.weight: [32, 1024, 512]
```

Expert axis is axis `0`.

Per expert:

| slice | bytes | MiB |
|---|---:|---:|
| input projection | 2,097,152 | 2.000 |
| output projection | 1,048,576 | 1.000 |
| pair total | 3,145,728 | 3.000 |

For a single decode token, top-k routing selects 8 experts:

```text
8 selected expert pairs = 8 * 3 MiB = 24 MiB
```

Approximate single-token active layer residency:

```text
attention weights     6 MiB
router/norms          <1 MiB
8 selected experts   24 MiB
---------------------------
active layer         ~30 MiB
```

This is the good news.

## Prefill caveat

For multi-token prefill, each token selects 8 experts. The union of selected experts across the prompt may become all 32 experts.

Naively loading the full expert union means the layer approaches full-layer residency:

```text
all experts 96 MiB + attention 6 MiB = ~102 MiB
```

For Granite under 128M, that still fits. For the eventual target, and for testing the real MoE discipline, the safe exact schedule is expert-grouped:

1. Compute router logits for all active tokens.
2. Record token/expert assignments and gate weights.
3. For expert `e = 0..31`:
   - load only expert `e` input/output slices,
   - process tokens routed to `e`,
   - accumulate with `index_add` semantics,
   - evict expert `e`.

This keeps MoE expert residency at about 3 MiB, at the cost of more passes.

## Exact forward mechanics

Reference implementation is Hugging Face Transformers `GraniteMoeForCausalLM`.

Critical math details:

1. Token embeddings are multiplied by `embedding_multiplier = 12.0`.
2. RMSNorm computes variance in FP32, then casts normalized states back to input dtype before multiplying by the BF16 norm weight.
3. RoPE uses `rope_theta = 1500000.0`; cos/sin are computed in FP32 and cast to hidden dtype.
4. Attention score scale is `attention_multiplier = 0.015625`, not ordinary `1/sqrt(head_dim)` by accident.
5. Attention residual is:

   ```text
   hidden = residual + attention_output * 0.22
   ```

6. Router logits are computed as linear output then cast to FP32.
7. Router uses top-k over logits, then softmax only over the selected top-k logits.
8. MoE expert input projection emits `2 * intermediate_size`; it is chunked into two halves:

   ```text
   hidden = silu(first_half) * second_half
   ```

9. Expert outputs are multiplied by their router gate and accumulated with `index_add` into token positions.
10. MoE residual is:

    ```text
    hidden = residual + moe_output * 0.22
    ```

11. Final hidden states pass through final RMSNorm.
12. LM head is tied to embeddings.
13. Output logits are divided by `logits_scaling = 6.0`.

## Output logits budget

The embedding / tied LM-head matrix is about 96.006 MiB. Under 128M it fits by itself.

Chunked vocab streaming is still useful because it generalizes, avoids co-residency surprises, and gives us a strict-budget mode for larger checkpoints.

```text
for vocab rows chunk:
  load embedding rows
  logits_chunk = hidden @ rows.T
  write logits_chunk into output vector
  evict rows
logits /= 6.0
```

The output logits vector itself is small:

```text
49155 BF16/FP32 logits ~= 96 KiB / 192 KiB
```

So full logits can still be materialized without keeping the full LM-head matrix resident.

## Development proof notes

The step-by-step parity probes and HF golden-trace notes are archived under:

```text
docs/archive/probes/
```

Those notes explain how the current Granite implementation was built and checked, but they are not needed for normal CLI/API use.
