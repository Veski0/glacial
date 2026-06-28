# OpenAI-Compatible Chat API

Glacial has a small OpenAI-shaped HTTP shim for local tools that expect Chat Completions.

This is a compatibility wrapper around the exact greedy prototype, **not** a high-throughput inference server. Requests are serialized through one decode lock.

## Start the server

```bash
python tools/glacial_openai_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --local-files-only \
  --served-model-name glacial-granite
```

If the model is not already cached, omit `--local-files-only` to allow Hugging Face download.

Optional durable per-request checkpoints:

```bash
python tools/glacial_openai_server.py \
  --port 8000 \
  --served-model-name glacial-granite \
  --checkpoint-root runs/openai-api
```

When enabled, the server saves each generated token before streaming/returning it and includes:

```text
X-Glacial-Checkpoint-Dir: runs/openai-api/chatcmpl-...
```

## Endpoints

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

## curl smoke

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

Streaming:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glacial-granite",
    "messages": [{"role": "user", "content": "Say hello in three words."}],
    "max_tokens": 8,
    "stream": true
  }'
```

## Python OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="glacial-granite",
    messages=[{"role": "user", "content": "Say hello in three words."}],
    max_tokens=8,
    temperature=0,
)
print(response.choices[0].message.content)
```

## Supported request shape

Supported:

- `model`
- `messages` with `system`, `user`, `assistant`, or `developer` roles
- string content, `null` content, or text-only content parts
- `max_tokens` / `max_completion_tokens`
- `stream`
- `stream_options.include_usage`
- `stop`
- `n=1`

Accepted but currently ignored:

- sampling knobs such as `temperature`, `top_p`, penalties, etc.

Unsupported:

- `n > 1`
- tool calling
- logprobs
- image/audio/non-text content parts
- structured `response_format`

## Important caveats

- Generation is exact greedy only.
- Only one request decodes at a time.
- The API is stateless unless `--checkpoint-root` is enabled.
- `--checkpoint-root` creates inspectable Glacial KV checkpoints, but OpenAI clients do not have a resume protocol for them yet.
