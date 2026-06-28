# CLI Usage

Glacial's main command-line entry point is:

```text
tools/glacial_generate.py
```

It runs exact greedy decode without instantiating the Hugging Face model. The tokenizer/config still come from Hugging Face, while model weights are visited directly from `model.safetensors`.

## Basic generation

Raw prompt:

```bash
python tools/glacial_generate.py \
  --prompt "Hello" \
  --max-new-tokens 8
```

Chat-template prompt:

```bash
python tools/glacial_generate.py \
  --chat-user "Please explain Glacial in one sentence." \
  --max-new-tokens 32
```

Show rendered prompt and per-token telemetry:

```bash
python tools/glacial_generate.py \
  --chat-user "Please say hello." \
  --max-new-tokens 8 \
  --show-prompt \
  --show-token-telemetry
```

If the model is already cached locally, add:

```bash
--local-files-only
```

## Resumable checkpoints

Start a checkpointed run:

```bash
python tools/glacial_generate.py \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --checkpoint-dir runs/hello \
  --show-token-telemetry
```

Resume it:

```bash
python tools/glacial_generate.py \
  --resume-from runs/hello \
  --max-new-tokens 1 \
  --show-token-telemetry
```

Inspect it:

```bash
python tools/inspect_checkpoint.py runs/hello \
  --decode-tokens \
  --show-tokens \
  --local-files-only
```

See [`resumable-decode.md`](resumable-decode.md) for checkpoint invariants and file layout.

## Weight budget telemetry

Observe resident-weight budget use:

```bash
python tools/glacial_generate.py \
  --prompt "Hello" \
  --max-new-tokens 2 \
  --show-token-telemetry \
  --weight-budget-bytes 1000000000
```

Enforce a hard resident-weight budget:

```bash
python tools/glacial_generate.py \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --weight-budget-bytes 9000000 \
  --enforce-weight-budget \
  --show-token-telemetry
```

## Backend selection

Auto-detect backend from config:

```bash
python tools/glacial_generate.py --backend auto --prompt "Hello" --max-new-tokens 1
```

Current explicit backend:

```bash
python tools/glacial_generate.py --backend granite --prompt "Hello" --max-new-tokens 1
```

## Fallback path

The default path uses KV-cache prefill/decode. For debugging, the old full-prefill-every-token path still exists:

```bash
python tools/glacial_generate.py \
  --prompt "Hello" \
  --max-new-tokens 2 \
  --no-kv-cache
```
