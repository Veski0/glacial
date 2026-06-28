# Resumable Decode Checkpoints

Glacial can persist greedy KV-cache decode state and resume from the last complete checkpoint.

In checkpointed mode, token emission is **durable before visible**: Glacial writes the checkpoint for a generated token before printing that token. If a token appeared in the terminal, the run can resume past it.

## CLI

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
python tools/inspect_checkpoint.py runs/hello --decode-tokens --show-tokens
```

Portable one-record JSONL inspection:

```bash
python tools/inspect_checkpoint.py runs/hello --jsonl
```

On resume, `--max-new-tokens` means additional tokens for this invocation.

## Checkpoint invariant

After every emitted token:

```text
token_ids = prompt tokens + generated tokens
kv_cache  = all tokens except token_ids[-1]
```

Resume decodes:

```text
input_token_id = token_ids[-1]
position       = len(token_ids) - 1
```

Then it appends the returned KV and emitted token, restoring the same invariant for the next checkpoint.

## Atomicity model

The latest checkpoint is selected only by:

```text
run/manifest.json
```

Each save writes a new immutable snapshot directory:

```text
run/
  manifest.json
  snapshots/
    tokens_00000002_.../
      tokens.jsonl
      snapshot.json
      kv/
        layer_00.safetensors
        layer_01.safetensors
        ...
```

`manifest.json` is atomically replaced after the snapshot is fully written. If the process crashes during checkpoint save, resume sees either the previous complete checkpoint or the new complete checkpoint, never a partially-overwritten KV set.

Checkpointed generation saves before printing each token:

```text
compute next token
append token in memory
write complete snapshot
atomically replace manifest.json
print token
```

So visible generated text is never ahead of the latest durable checkpoint.

## Files

- `manifest.json`: atomic pointer to the latest complete snapshot.
- `tokens.jsonl`: portable token stream, one token per line.
- `kv/layer_XX.safetensors`: BF16 `key` and `value` tensors for one layer.

Old snapshots are intentionally retained for now. This favors recovery confidence over disk cleanup.
