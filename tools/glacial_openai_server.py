#!/usr/bin/env python3
"""OpenAI-compatible chat API shim for Glacial.

This is deliberately a small compatibility wrapper around the exact
prototype, not a high-throughput inference server. It exposes enough of the
OpenAI Chat Completions shape for local tools to talk to Glacial via:

    POST /v1/chat/completions
    GET  /v1/models

Generation is serialized through one engine lock. Sampling (temperature,
top-p) is supported with checkpointable RNG state; ``temperature=0`` (the
default) produces exact greedy decode.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse
from uuid import uuid4

DEFAULT_MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
DEFAULT_REVISION = "b0e4fd07be563ba8bb7689c47dc9bebdff5471ab"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glacial.backends import backend_names, resolve_backend
from glacial.kv import save_decode_checkpoint
from glacial.sampler import Sampler
from glacial.weights import WeightBudget, read_safetensors_header


class OpenAIHTTPError(Exception):
    def __init__(self, status: int, message: str, *, param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code


def require_deps() -> None:
    try:
        import torch  # noqa: F401
        import huggingface_hub  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        print(
            "Missing Python dependency: " + exc.name + "\n\n"
            "Install dependencies, for example:\n\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def resolve_repo_file(*, model_id: str, revision: str, filename: str, cache_dir: str | None, local_files_only: bool) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )


def load_config(*, model_id: str, revision: str, cache_dir: str | None, local_files_only: bool) -> dict[str, Any]:
    path = resolve_repo_file(
        model_id=model_id,
        revision=revision,
        filename="config.json",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def json_dumps(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def unix_now() -> int:
    return int(time.time())


def normalize_stop(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "stop must be a string or list of strings", param="stop")
            if item:
                out.append(item)
        return out
    raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "stop must be a string or list of strings", param="stop")


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "messages must be a non-empty array", param="messages")

    normalized: list[dict[str, str]] = []
    for idx, message in enumerate(value):
        if not isinstance(message, dict):
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, f"messages[{idx}] must be an object", param="messages")
        role = message.get("role")
        if role == "developer":
            # OpenAI's newer developer role is closest to system for HF chat templates.
            role = "system"
        if role not in {"system", "user", "assistant"}:
            raise OpenAIHTTPError(
                HTTPStatus.BAD_REQUEST,
                f"unsupported message role {role!r}; supported roles are system, user, assistant, developer",
                param=f"messages[{idx}].role",
            )

        content = message.get("content", "")
        if content is None:
            content_text = ""
        elif isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part_idx, part in enumerate(content):
                if not isinstance(part, dict):
                    raise OpenAIHTTPError(
                        HTTPStatus.BAD_REQUEST,
                        f"messages[{idx}].content[{part_idx}] must be an object",
                        param=f"messages[{idx}].content",
                    )
                if part.get("type") != "text":
                    raise OpenAIHTTPError(
                        HTTPStatus.BAD_REQUEST,
                        "only text message content parts are supported",
                        param=f"messages[{idx}].content[{part_idx}]",
                    )
                text = part.get("text", "")
                if not isinstance(text, str):
                    raise OpenAIHTTPError(
                        HTTPStatus.BAD_REQUEST,
                        "text content parts must contain string text",
                        param=f"messages[{idx}].content[{part_idx}].text",
                    )
                parts.append(text)
            content_text = "\n".join(parts)
        else:
            raise OpenAIHTTPError(
                HTTPStatus.BAD_REQUEST,
                f"messages[{idx}].content must be a string, null, or text-part array",
                param=f"messages[{idx}].content",
            )

        normalized.append({"role": role, "content": content_text})
    return normalized


def request_max_tokens(body: dict[str, Any], *, default: int, limit: int) -> int:
    param = "max_tokens"
    if body.get("max_completion_tokens") is not None:
        value = body["max_completion_tokens"]
        param = "max_completion_tokens"
    elif body.get("max_tokens") is not None:
        value = body["max_tokens"]
    else:
        value = default

    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, f"{param} must be an integer", param=param)
    if value < 0:
        raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, f"{param} must be non-negative", param=param)
    if value > limit:
        raise OpenAIHTTPError(
            HTTPStatus.BAD_REQUEST,
            f"{param} {value} exceeds this server's limit of {limit}",
            param=param,
            code="context_length_exceeded",
        )
    return value


class StopFilter:
    """Incremental stop-sequence filter that avoids leaking stop text."""

    def __init__(self, stop: list[str]):
        self.stop = stop
        self.max_stop_len = max((len(item) for item in stop), default=0)
        self.pending = ""
        self.finished = False

    def push(self, text: str) -> tuple[str, bool]:
        if not self.stop:
            return text, False

        self.pending += text
        earliest: int | None = None
        for item in self.stop:
            idx = self.pending.find(item)
            if idx >= 0 and (earliest is None or idx < earliest):
                earliest = idx

        if earliest is not None:
            out = self.pending[:earliest]
            self.pending = ""
            self.finished = True
            return out, True

        keep = max(self.max_stop_len - 1, 0)
        if keep == 0 or len(self.pending) <= keep:
            return "", False

        out = self.pending[:-keep]
        self.pending = self.pending[-keep:]
        return out, False

    def flush(self) -> str:
        out = self.pending
        self.pending = ""
        return out


@dataclass
class PreparedChat:
    messages: list[dict[str, str]]
    rendered_text: str
    token_ids: list[int]
    prompt_token_count: int


@dataclass
class TokenEvent:
    index: int
    token_id: int
    text: str
    is_eos: bool
    telemetry: dict[str, Any]


class GlacialChatEngine:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        model_file: Path | None,
        backend_name: str,
        cache_dir: str | None,
        local_files_only: bool,
        lm_head_chunk_rows: int,
        weight_budget_bytes: int | None,
        enforce_weight_budget: bool,
        served_model_name: str | None,
        accept_any_model_name: bool,
        checkpoint_root: Path | None,
    ):
        import torch
        from transformers import AutoTokenizer

        torch.set_grad_enabled(False)

        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.lm_head_chunk_rows = lm_head_chunk_rows
        self.weight_budget_bytes = weight_budget_bytes
        self.enforce_weight_budget = enforce_weight_budget
        self.served_model_name = served_model_name or model_id
        self.accept_any_model_name = accept_any_model_name
        self.checkpoint_root = checkpoint_root
        self.lock = threading.Lock()

        if model_file is None:
            self.model_file = resolve_repo_file(
                model_id=model_id,
                revision=revision,
                filename="model.safetensors",
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        else:
            self.model_file = model_file
        if not self.model_file.exists():
            raise SystemExit(f"model.safetensors not found: {self.model_file}")

        self.config = load_config(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.backend = resolve_backend(backend_name, config=self.config)
        self.header_len, self.header = read_safetensors_header(self.model_file)
        self.payload_start = 8 + self.header_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    @property
    def accepted_model_names(self) -> set[str]:
        return {self.served_model_name, self.model_id}

    def check_model_name(self, requested: Any) -> str:
        if not isinstance(requested, str) or not requested:
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "model must be a non-empty string", param="model")
        if self.accept_any_model_name or requested in self.accepted_model_names:
            return requested
        names = ", ".join(sorted(self.accepted_model_names))
        raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, f"unknown model {requested!r}; available: {names}", param="model")

    def prepare_chat(self, messages: list[dict[str, str]]) -> PreparedChat:
        rendered_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = self.tokenizer(rendered_text, return_tensors=None)["input_ids"]
        return PreparedChat(
            messages=messages,
            rendered_text=rendered_text,
            token_ids=list(token_ids),
            prompt_token_count=len(token_ids),
        )

    def new_budget(self) -> WeightBudget | None:
        if self.weight_budget_bytes is None:
            return None
        return WeightBudget(limit_bytes=self.weight_budget_bytes, enforce=self.enforce_weight_budget)

    def checkpoint_dir_for(self, request_id: str) -> Path | None:
        if self.checkpoint_root is None:
            return None
        return self.checkpoint_root / request_id

    def generate_tokens(
        self,
        *,
        prepared: PreparedChat,
        max_tokens: int,
        request_id: str,
        response_model: str,
        sampler: Sampler | None = None,
    ) -> Iterator[TokenEvent]:
        if max_tokens <= 0:
            return

        # The prototype has no scheduler. Serialize decode so two HTTP clients do
        # not interleave large transient tensor visits and KV allocations.
        with self.lock:
            token_ids = list(prepared.token_ids)
            prompt_token_count = prepared.prompt_token_count
            kv_cache = None
            budget = self.new_budget()
            checkpoint_dir = self.checkpoint_dir_for(request_id)

            def persist_checkpoint() -> None:
                if checkpoint_dir is None:
                    return
                if kv_cache is None:
                    raise RuntimeError("internal error: cannot checkpoint without kv_cache")
                save_decode_checkpoint(
                    run_dir=checkpoint_dir,
                    token_ids=token_ids,
                    prompt_token_count=prompt_token_count,
                    kv_cache=kv_cache,
                    model_id=self.model_id,
                    revision=self.revision,
                    model_file=self.model_file,
                    backend_name=self.backend.name,
                    rendered_text=prepared.rendered_text,
                    prompt_mode="openai_chat",
                    messages=prepared.messages,
                    config=self.config,
                    lm_head_chunk_rows=self.lm_head_chunk_rows,
                    sampler=sampler.to_manifest() if sampler is not None else None,
                )

            def make_event(step: int, next_id: int, telemetry: dict[str, Any]) -> TokenEvent:
                token_ids.append(next_id)
                persist_checkpoint()
                text = self.tokenizer.decode([next_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                return TokenEvent(
                    index=step,
                    token_id=next_id,
                    text=text,
                    is_eos=self.tokenizer.eos_token_id is not None and next_id == self.tokenizer.eos_token_id,
                    telemetry=telemetry,
                )

            next_id, kv_cache, telemetry = self.backend.prefill_kv_greedy(
                token_ids=token_ids,
                model_file=self.model_file,
                header=self.header,
                payload_start=self.payload_start,
                config=self.config,
                lm_head_chunk_rows=self.lm_head_chunk_rows,
                budget=budget,
                sampler=sampler,
            )
            event = make_event(0, next_id, telemetry)
            yield event
            if event.is_eos:
                return

            for step in range(1, max_tokens):
                position = len(token_ids) - 1
                next_id, kv_cache, telemetry = self.backend.decode_kv_greedy(
                    input_token_id=token_ids[-1],
                    position=position,
                    kv_cache=kv_cache,
                    model_file=self.model_file,
                    header=self.header,
                    payload_start=self.payload_start,
                    config=self.config,
                    lm_head_chunk_rows=self.lm_head_chunk_rows,
                    budget=budget,
                    sampler=sampler,
                )
                event = make_event(step, next_id, telemetry)
                yield event
                if event.is_eos:
                    return


class GlacialOpenAIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], **kwargs: Any):
        self.engine: GlacialChatEngine = kwargs.pop("engine")
        self.api_key: str | None = kwargs.pop("api_key")
        self.default_max_tokens: int = kwargs.pop("default_max_tokens")
        self.max_tokens_limit: int = kwargs.pop("max_tokens_limit")
        self.max_request_bytes: int = kwargs.pop("max_request_bytes")
        self.default_seed: int | None = kwargs.pop("default_seed")
        super().__init__(server_address, handler_class)


class OpenAICompatHandler(BaseHTTPRequestHandler):
    server: GlacialOpenAIServer
    server_version = "GlacialOpenAI/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send_headers(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.send_response(int(status))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "authorization,content-type")
        self.send_header("Connection", "close")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.close_connection = True

    def _send_json(self, status: int, obj: Any, headers: dict[str, str] | None = None) -> None:
        data = json_dumps(obj)
        all_headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(data)),
        }
        if headers:
            all_headers.update(headers)
        self._send_headers(status, all_headers)
        self.wfile.write(data)

    def _send_openai_error(self, exc: OpenAIHTTPError) -> None:
        self._send_json(
            exc.status,
            {
                "error": {
                    "message": exc.message,
                    "type": "invalid_request_error" if exc.status < 500 else "server_error",
                    "param": exc.param,
                    "code": exc.code,
                }
            },
        )

    def _check_auth(self) -> None:
        expected = self.server.api_key
        if expected is None:
            return
        actual = self.headers.get("Authorization", "")
        if actual != f"Bearer {expected}":
            raise OpenAIHTTPError(HTTPStatus.UNAUTHORIZED, "invalid or missing bearer token", code="unauthorized")

    def _read_json_body(self) -> dict[str, Any]:
        length_raw = self.headers.get("Content-Length")
        if length_raw is None:
            raise OpenAIHTTPError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        try:
            length = int(length_raw)
        except ValueError as exc:
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from exc
        if length > self.server.max_request_bytes:
            raise OpenAIHTTPError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"request body exceeds {self.server.max_request_bytes} bytes",
            )
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
        return body

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib callback name
        self._send_headers(HTTPStatus.NO_CONTENT, {"Content-Length": "0"})

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        try:
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            self._check_auth()
            if path == "/v1/models":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": self.server.engine.served_model_name,
                                "object": "model",
                                "created": 0,
                                "owned_by": "glacial",
                            }
                        ],
                    },
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "not_found_error"}})
        except OpenAIHTTPError as exc:
            self._send_openai_error(exc)

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        try:
            self._check_auth()
            path = urlparse(self.path).path
            if path != "/v1/chat/completions":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "not_found_error"}})
                return
            body = self._read_json_body()
            self._handle_chat_completions(body)
        except OpenAIHTTPError as exc:
            self._send_openai_error(exc)
        except BrokenPipeError:
            return
        except BaseException as exc:  # Keep the server process alive for request-local failures.
            if isinstance(exc, KeyboardInterrupt):
                raise
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"message": str(exc), "type": "server_error", "param": None, "code": None}},
            )

    def _validate_chat_request(self, body: dict[str, Any]) -> tuple[str, list[dict[str, str]], int, list[str], bool, bool, float, float | None, int | None]:
        n = body.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or n != 1:
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "only n=1 is supported", param="n")
        if body.get("logprobs") or body.get("top_logprobs"):
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "logprobs are not supported", param="logprobs")
        if body.get("tools") or body.get("tool_choice"):
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "tool calling is not supported", param="tools")
        if body.get("response_format") not in (None, {"type": "text"}):
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "response_format is not supported", param="response_format")

        response_model = self.server.engine.check_model_name(body.get("model"))
        messages = normalize_messages(body.get("messages"))
        max_tokens = request_max_tokens(body, default=self.server.default_max_tokens, limit=self.server.max_tokens_limit)
        stop = normalize_stop(body.get("stop"))
        stream_raw = body.get("stream", False)
        if not isinstance(stream_raw, bool):
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "stream must be a boolean", param="stream")
        stream = stream_raw
        stream_options = body.get("stream_options") or {}
        if not isinstance(stream_options, dict):
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "stream_options must be an object", param="stream_options")
        include_usage_raw = stream_options.get("include_usage", False)
        if not isinstance(include_usage_raw, bool):
            raise OpenAIHTTPError(
                HTTPStatus.BAD_REQUEST,
                "stream_options.include_usage must be a boolean",
                param="stream_options.include_usage",
            )
        include_usage = include_usage_raw

        temperature = body.get("temperature", 0.0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "temperature must be a number", param="temperature")
        temperature = float(temperature)
        if temperature < 0:
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "temperature must be non-negative", param="temperature")
        if temperature > 2:
            raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "temperature must be at most 2", param="temperature")

        top_p = body.get("top_p", None)
        if top_p is not None:
            if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
                raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "top_p must be a number", param="top_p")
            top_p = float(top_p)
            if top_p < 0 or top_p > 1:
                raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "top_p must be between 0 and 1", param="top_p")

        seed = body.get("seed", None)
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise OpenAIHTTPError(HTTPStatus.BAD_REQUEST, "seed must be an integer", param="seed")

        return response_model, messages, max_tokens, stop, stream, include_usage, temperature, top_p, seed

    def _handle_chat_completions(self, body: dict[str, Any]) -> None:
        response_model, messages, max_tokens, stop, stream, include_usage, temperature, top_p, seed = self._validate_chat_request(body)
        prepared = self.server.engine.prepare_chat(messages)
        request_id = "chatcmpl-" + uuid4().hex
        checkpoint_dir = self.server.engine.checkpoint_dir_for(request_id)
        headers = {}
        if checkpoint_dir is not None:
            headers["X-Glacial-Checkpoint-Dir"] = str(checkpoint_dir)

        sampler = Sampler.from_params(
            temperature=temperature,
            top_p=top_p,
            seed=seed if seed is not None else self.server.default_seed,
        )

        if stream:
            self._stream_chat_completion(
                request_id=request_id,
                response_model=response_model,
                prepared=prepared,
                max_tokens=max_tokens,
                stop=stop,
                include_usage=include_usage,
                headers=headers,
                sampler=sampler,
            )
        else:
            self._complete_chat_completion(
                request_id=request_id,
                response_model=response_model,
                prepared=prepared,
                max_tokens=max_tokens,
                stop=stop,
                headers=headers,
                sampler=sampler,
            )

    def _complete_chat_completion(
        self,
        *,
        request_id: str,
        response_model: str,
        prepared: PreparedChat,
        max_tokens: int,
        stop: list[str],
        headers: dict[str, str],
        sampler: Sampler | None = None,
    ) -> None:
        created = unix_now()
        stop_filter = StopFilter(stop)
        chunks: list[str] = []
        completion_tokens = 0
        finish_reason = "length" if max_tokens == 0 else "length"

        token_iter = self.server.engine.generate_tokens(
            prepared=prepared,
            max_tokens=max_tokens,
            request_id=request_id,
            response_model=response_model,
            sampler=sampler,
        )
        try:
            for event in token_iter:
                completion_tokens += 1
                if event.is_eos:
                    tail = stop_filter.flush()
                    if tail:
                        chunks.append(tail)
                    finish_reason = "stop"
                    break
                text, stopped = stop_filter.push(event.text)
                if text:
                    chunks.append(text)
                if stopped:
                    finish_reason = "stop"
                    break
            else:
                tail = stop_filter.flush()
                if tail:
                    chunks.append(tail)
        finally:
            close = getattr(token_iter, "close", None)
            if close is not None:
                close()

        content = "".join(chunks)
        usage = {
            "prompt_tokens": prepared.prompt_token_count,
            "completion_tokens": completion_tokens,
            "total_tokens": prepared.prompt_token_count + completion_tokens,
        }
        self._send_json(
            HTTPStatus.OK,
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": response_model,
                "system_fingerprint": "glacial-greedy-v1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            },
            headers=headers,
        )

    def _write_sse(self, obj: Any) -> None:
        self.wfile.write(b"data: ")
        self.wfile.write(json_dumps(obj).rstrip(b"\n"))
        self.wfile.write(b"\n\n")
        self.wfile.flush()

    def _write_sse_done(self) -> None:
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _stream_chat_completion(
        self,
        *,
        request_id: str,
        response_model: str,
        prepared: PreparedChat,
        max_tokens: int,
        stop: list[str],
        include_usage: bool,
        headers: dict[str, str],
        sampler: Sampler | None = None,
    ) -> None:
        created = unix_now()
        self._send_headers(
            HTTPStatus.OK,
            {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                **headers,
            },
        )

        def chunk(delta: dict[str, Any], finish_reason: str | None = None, usage: dict[str, int] | None = None) -> dict[str, Any]:
            obj: dict[str, Any] = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response_model,
                "system_fingerprint": "glacial-greedy-v1",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            if usage is not None:
                obj["usage"] = usage
            return obj

        completion_tokens = 0
        finish_reason = "length"
        stop_filter = StopFilter(stop)
        token_iter = self.server.engine.generate_tokens(
            prepared=prepared,
            max_tokens=max_tokens,
            request_id=request_id,
            response_model=response_model,
            sampler=sampler,
        )
        try:
            self._write_sse(chunk({"role": "assistant"}))
            for event in token_iter:
                completion_tokens += 1
                if event.is_eos:
                    tail = stop_filter.flush()
                    if tail:
                        self._write_sse(chunk({"content": tail}))
                    finish_reason = "stop"
                    break

                text, stopped = stop_filter.push(event.text)
                if text:
                    self._write_sse(chunk({"content": text}))
                if stopped:
                    finish_reason = "stop"
                    break
            else:
                tail = stop_filter.flush()
                if tail:
                    self._write_sse(chunk({"content": tail}))

            usage = {
                "prompt_tokens": prepared.prompt_token_count,
                "completion_tokens": completion_tokens,
                "total_tokens": prepared.prompt_token_count + completion_tokens,
            }
            self._write_sse(chunk({}, finish_reason=finish_reason))
            if include_usage:
                self._write_sse(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": response_model,
                        "system_fingerprint": "glacial-greedy-v1",
                        "choices": [],
                        "usage": usage,
                    }
                )
            self._write_sse_done()
        except BrokenPipeError:
            return
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            traceback.print_exc()
            try:
                self._write_sse({"error": {"message": str(exc), "type": "server_error", "param": None, "code": None}})
                self._write_sse_done()
            except BrokenPipeError:
                pass
        finally:
            close = getattr(token_iter, "close", None)
            if close is not None:
                close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-file", type=Path, default=None, help="Local model.safetensors path. Defaults to HF cache/hub.")
    parser.add_argument("--served-model-name", default=None, help="Model id exposed by /v1/models. Defaults to --model-id.")
    parser.add_argument("--accept-any-model-name", action="store_true", help="Accept any request model name and serve this engine anyway.")
    parser.add_argument("--backend", default="auto", help="Architecture backend to use: auto or one of " + ", ".join(backend_names()))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096, help="Rows per tied-LM-head chunk for streaming greedy argmax.")
    parser.add_argument("--default-max-tokens", type=int, default=16, help="Default completion length when a request omits max_tokens.")
    parser.add_argument("--max-tokens-limit", type=int, default=256, help="Hard per-request completion token limit.")
    parser.add_argument("--max-request-bytes", type=int, default=1_000_000)
    parser.add_argument("--weight-budget-bytes", type=int, default=None, help="Optional resident weight-byte budget.")
    parser.add_argument("--enforce-weight-budget", action="store_true", help="Raise if resident weights exceed --weight-budget-bytes.")
    parser.add_argument("--checkpoint-root", type=Path, default=None, help="Optional root for per-request durable KV checkpoints.")
    parser.add_argument("--api-key", default=None, help="Optional bearer token required for /v1 endpoints.")
    parser.add_argument("--seed", type=int, default=None, help="Default RNG seed for sampling (when request omits seed).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.default_max_tokens < 0:
        raise SystemExit("--default-max-tokens must be non-negative")
    if args.max_tokens_limit < args.default_max_tokens:
        raise SystemExit("--max-tokens-limit must be >= --default-max-tokens")

    require_deps()
    engine = GlacialChatEngine(
        model_id=args.model_id,
        revision=args.revision,
        model_file=args.model_file,
        backend_name=args.backend,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        lm_head_chunk_rows=args.lm_head_chunk_rows,
        weight_budget_bytes=args.weight_budget_bytes,
        enforce_weight_budget=args.enforce_weight_budget,
        served_model_name=args.served_model_name,
        accept_any_model_name=args.accept_any_model_name,
        checkpoint_root=args.checkpoint_root,
    )

    server = GlacialOpenAIServer(
        (args.host, args.port),
        OpenAICompatHandler,
        engine=engine,
        api_key=args.api_key,
        default_max_tokens=args.default_max_tokens,
        max_tokens_limit=args.max_tokens_limit,
        max_request_bytes=args.max_request_bytes,
        default_seed=args.seed,
    )
    print(
        f"Glacial OpenAI-compatible API listening on http://{args.host}:{args.port}/v1 "
        f"model={engine.served_model_name!r} backend={engine.backend.name!r}",
        flush=True,
    )
    if args.checkpoint_root is not None:
        print(f"Per-request checkpoints: {args.checkpoint_root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
