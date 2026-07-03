"""Safetensors weight loading helpers for Glacial.

These functions intentionally load only the requested tensor region and return a
fresh BF16 torch tensor detached from the file bytes.
"""

from __future__ import annotations

import json
import struct
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

BF16_BYTES = 2
EMBED_TENSOR = "model.embed_tokens.weight"


@dataclass
class WeightBudget:
    """Track currently resident and visited weight bytes.

    By default this is observational. Set ``enforce=True`` with ``limit_bytes``
    to raise ``MemoryError`` when a visit would exceed the resident limit.
    """

    limit_bytes: int | None = None
    enforce: bool = False
    current_resident_bytes: int = 0
    peak_resident_bytes: int = 0
    total_visited_bytes: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)

    @contextmanager
    def resident(self, byte_count: int, *, name: str) -> Iterator[None]:
        self.current_resident_bytes += int(byte_count)
        self.total_visited_bytes += int(byte_count)
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.current_resident_bytes)
        if self.limit_bytes is not None and self.current_resident_bytes > self.limit_bytes:
            violation = {
                "name": name,
                "resident_bytes": self.current_resident_bytes,
                "limit_bytes": self.limit_bytes,
            }
            self.violations.append(violation)
            if self.enforce:
                self.current_resident_bytes -= int(byte_count)
                raise MemoryError(f"weight budget exceeded for {name}: {violation}")
        try:
            yield
        finally:
            self.current_resident_bytes -= int(byte_count)


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        header_len_raw = f.read(8)
        if len(header_len_raw) != 8:
            raise SystemExit(f"{path}: file too small for safetensors header")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise SystemExit(f"{path}: truncated safetensors header")
    try:
        header = json.loads(header_raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid safetensors JSON header: {exc}") from exc
    return header_len, header


def _require_bf16_tensor(*, tensor_name: str, tensor_meta: dict[str, Any]) -> tuple[list[int], list[int]]:
    dtype = tensor_meta["dtype"]
    shape = [int(x) for x in tensor_meta["shape"]]
    data_offsets = [int(x) for x in tensor_meta["data_offsets"]]
    if dtype != "BF16":
        raise SystemExit(f"Expected {tensor_name} dtype BF16, got {dtype}")
    return shape, data_offsets


def read_full_bf16_tensor(
    path: Path,
    *,
    tensor_name: str,
    tensor_meta: dict[str, Any],
    payload_start: int,
    budget: WeightBudget | None = None,
):
    import torch

    shape, data_offsets = _require_bf16_tensor(tensor_name=tensor_name, tensor_meta=tensor_meta)

    byte_size = int(data_offsets[1]) - int(data_offsets[0])
    expected_byte_size = BF16_BYTES
    for dim in shape:
        expected_byte_size *= dim
    if byte_size != expected_byte_size:
        raise SystemExit(f"{tensor_name}: header byte size {byte_size} != shape-derived {expected_byte_size}")

    with path.open("rb") as f:
        f.seek(payload_start + int(data_offsets[0]))
        raw = f.read(byte_size)
    if len(raw) != byte_size:
        raise SystemExit(f"Short read for {tensor_name}")

    if budget is not None:
        raise ValueError("budgeted weight access requires SafetensorsWeights context-manager methods")
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone().view(*shape)


def read_embedding_rows(
    path: Path,
    rows: list[int],
    *,
    tensor_meta: dict[str, Any],
    payload_start: int,
    budget: WeightBudget | None = None,
):
    import torch

    shape, data_offsets = _require_bf16_tensor(tensor_name=EMBED_TENSOR, tensor_meta=tensor_meta)
    if len(shape) != 2:
        raise SystemExit(f"Expected {EMBED_TENSOR} rank 2, got shape {shape}")

    vocab_size, hidden_size = int(shape[0]), int(shape[1])
    row_bytes = hidden_size * BF16_BYTES
    tensor_payload_start = payload_start + int(data_offsets[0])
    tensor_payload_end = payload_start + int(data_offsets[1])

    loaded = []
    with path.open("rb") as f:
        for token_id in rows:
            if token_id < 0 or token_id >= vocab_size:
                raise SystemExit(f"Token id {token_id} outside embedding vocab size {vocab_size}")
            offset = tensor_payload_start + token_id * row_bytes
            if offset + row_bytes > tensor_payload_end:
                raise SystemExit(f"Computed row range exceeds tensor range for token id {token_id}")
            f.seek(offset)
            raw = f.read(row_bytes)
            if len(raw) != row_bytes:
                raise SystemExit(f"Short read for token id {token_id}")
            loaded.append(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone())

    if budget is not None:
        raise ValueError("budgeted weight access requires SafetensorsWeights context-manager methods")
    tensor = torch.stack(loaded, dim=0)
    return tensor, hidden_size


def read_row_chunk(
    path: Path,
    *,
    tensor_name: str,
    tensor_meta: dict[str, Any],
    row_start: int,
    row_count: int,
    payload_start: int,
    budget: WeightBudget | None = None,
):
    import torch

    shape, data_offsets = _require_bf16_tensor(tensor_name=tensor_name, tensor_meta=tensor_meta)
    if len(shape) != 2:
        raise SystemExit(f"Expected {tensor_name} rank 2, got shape {shape}")
    vocab_size, hidden_size = shape
    if row_start < 0 or row_start >= vocab_size:
        raise SystemExit(f"row_start {row_start} outside row count {vocab_size}")
    row_count = min(row_count, vocab_size - row_start)

    row_bytes = hidden_size * BF16_BYTES
    byte_count = row_count * row_bytes
    offset = payload_start + int(data_offsets[0]) + row_start * row_bytes
    with path.open("rb") as f:
        f.seek(offset)
        raw = f.read(byte_count)
    if len(raw) != byte_count:
        raise SystemExit(f"Short read for {tensor_name} rows {row_start}:{row_start + row_count}")

    if budget is not None:
        raise ValueError("budgeted weight access requires SafetensorsWeights context-manager methods")
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone().view(row_count, hidden_size)


def read_lm_head_chunk(
    path: Path,
    *,
    tensor_meta: dict[str, Any],
    row_start: int,
    row_count: int,
    payload_start: int,
    budget: WeightBudget | None = None,
):
    return read_row_chunk(
        path,
        tensor_name="tied LM head",
        tensor_meta=tensor_meta,
        row_start=row_start,
        row_count=row_count,
        payload_start=payload_start,
        budget=budget,
    )


def read_expert_slice(
    path: Path,
    *,
    tensor_name: str,
    tensor_meta: dict[str, Any],
    expert_id: int,
    payload_start: int,
    budget: WeightBudget | None = None,
):
    import torch

    shape, data_offsets = _require_bf16_tensor(tensor_name=tensor_name, tensor_meta=tensor_meta)
    if len(shape) != 3:
        raise SystemExit(f"Expected {tensor_name} rank 3 expert tensor, got shape {shape}")
    num_experts, out_size, in_size = shape
    if expert_id < 0 or expert_id >= num_experts:
        raise SystemExit(f"Expert id {expert_id} outside {tensor_name} expert count {num_experts}")

    slice_bytes = out_size * in_size * BF16_BYTES
    tensor_start = payload_start + int(data_offsets[0])
    tensor_end = payload_start + int(data_offsets[1])
    offset = tensor_start + expert_id * slice_bytes
    if offset + slice_bytes > tensor_end:
        raise SystemExit(f"Computed expert slice range exceeds tensor range for {tensor_name} expert {expert_id}")

    with path.open("rb") as f:
        f.seek(offset)
        raw = f.read(slice_bytes)
    if len(raw) != slice_bytes:
        raise SystemExit(f"Short read for {tensor_name} expert {expert_id}")

    if budget is not None:
        raise ValueError("budgeted weight access requires SafetensorsWeights context-manager methods")
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone().view(out_size, in_size)


class SafetensorsWeights:
    """Context-manager weight provider for a single safetensors file.

    This is the preferred runtime API for budgeted execution:

        with provider.tensor(name) as weight:
            ...

    The budget's resident byte count stays elevated for the whole ``with``
    block, i.e. for the actual lifetime of the loaded weight reference.
    """

    def __init__(
        self,
        path: Path,
        *,
        header: dict[str, Any] | None = None,
        payload_start: int | None = None,
        budget: WeightBudget | None = None,
    ):
        self.path = path
        if header is None or payload_start is None:
            self.header_len, self.header = read_safetensors_header(path)
            self.payload_start = 8 + self.header_len
        else:
            self.header_len = payload_start - 8
            self.header = header
            self.payload_start = payload_start
        self.budget = budget

    def meta(self, tensor_name: str) -> dict[str, Any]:
        try:
            return self.header[tensor_name]
        except KeyError as exc:
            raise SystemExit(f"{self.path}: missing tensor {tensor_name!r}") from exc

    @contextmanager
    def _resident_tensor(self, tensor, *, name: str):
        byte_count = tensor.numel() * BF16_BYTES
        if self.budget is None:
            yield tensor
            return
        with self.budget.resident(byte_count, name=name):
            yield tensor

    @contextmanager
    def tensor(self, tensor_name: str):
        tensor = read_full_bf16_tensor(
            self.path,
            tensor_name=tensor_name,
            tensor_meta=self.meta(tensor_name),
            payload_start=self.payload_start,
        )
        with self._resident_tensor(tensor, name=tensor_name):
            yield tensor

    @contextmanager
    def tensor_any(self, tensor_name: str):
        """Read a tensor of any dtype (not just BF16).

        Needed for tensors like LFM2's expert_bias which is stored as F32.
        """
        import torch

        meta = self.meta(tensor_name)
        dtype_str = meta["dtype"]
        shape = [int(x) for x in meta["shape"]]
        data_offsets = [int(x) for x in meta["data_offsets"]]

        dtype_map = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
        torch_dtype = dtype_map.get(dtype_str)
        if torch_dtype is None:
            raise SystemExit(f"Unsupported dtype {dtype_str} for tensor {tensor_name}")

        byte_size = int(data_offsets[1]) - int(data_offsets[0])
        with self.path.open("rb") as f:
            f.seek(self.payload_start + int(data_offsets[0]))
            raw = f.read(byte_size)
        if len(raw) != byte_size:
            raise SystemExit(f"Short read for {tensor_name}")
        tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype).clone().view(*shape)
        with self._resident_tensor(tensor, name=tensor_name):
            yield tensor

    @contextmanager
    def embedding_rows(self, rows: list[int]):
        tensor, hidden_size = read_embedding_rows(
            self.path,
            rows,
            tensor_meta=self.meta(EMBED_TENSOR),
            payload_start=self.payload_start,
        )
        with self._resident_tensor(tensor, name=f"{EMBED_TENSOR}[rows]"):
            yield tensor, hidden_size

    @contextmanager
    def row_chunk(self, tensor_name: str, *, row_start: int, row_count: int):
        tensor = read_row_chunk(
            self.path,
            tensor_name=tensor_name,
            tensor_meta=self.meta(tensor_name),
            row_start=row_start,
            row_count=row_count,
            payload_start=self.payload_start,
        )
        with self._resident_tensor(tensor, name=f"{tensor_name}[{row_start}:{row_start + row_count}]"):
            yield tensor

    @contextmanager
    def lm_head_chunk(self, *, row_start: int, row_count: int):
        tensor = read_lm_head_chunk(
            self.path,
            tensor_meta=self.meta(EMBED_TENSOR),
            row_start=row_start,
            row_count=row_count,
            payload_start=self.payload_start,
        )
        with self._resident_tensor(tensor, name=f"tied LM head[{row_start}:{row_start + row_count}]"):
            yield tensor

    @contextmanager
    def expert_slice(self, tensor_name: str, *, expert_id: int):
        tensor = read_expert_slice(
            self.path,
            tensor_name=tensor_name,
            tensor_meta=self.meta(tensor_name),
            expert_id=expert_id,
            payload_start=self.payload_start,
        )
        with self._resident_tensor(tensor, name=f"{tensor_name}[expert {expert_id}]"):
            yield tensor
