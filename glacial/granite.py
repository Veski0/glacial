"""Granite MoE math helpers proven by the prototype probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glacial.weights import BF16_BYTES, SafetensorsWeights, WeightBudget

FINAL_NORM_TENSOR = "model.norm.weight"


def scalar_config(config: dict[str, Any]) -> dict[str, float]:
    return {
        "embedding_multiplier": float(config.get("embedding_multiplier", 12.0)),
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(config.get("rope_theta", 1500000.0)),
        "attention_multiplier": float(config.get("attention_multiplier", 0.015625)),
        "residual_multiplier": float(config.get("residual_multiplier", 0.22)),
        "logits_scaling": float(config.get("logits_scaling", 6.0)),
    }


def layer_tensor(layer_idx: int, suffix: str) -> str:
    return f"model.layers.{layer_idx}.{suffix}"


def required_layer_tensors(layer_idx: int) -> list[str]:
    return [
        layer_tensor(layer_idx, "input_layernorm.weight"),
        layer_tensor(layer_idx, "post_attention_layernorm.weight"),
        layer_tensor(layer_idx, "self_attn.q_proj.weight"),
        layer_tensor(layer_idx, "self_attn.k_proj.weight"),
        layer_tensor(layer_idx, "self_attn.v_proj.weight"),
        layer_tensor(layer_idx, "self_attn.o_proj.weight"),
        layer_tensor(layer_idx, "block_sparse_moe.router.layer.weight"),
        layer_tensor(layer_idx, "block_sparse_moe.input_linear.weight"),
        layer_tensor(layer_idx, "block_sparse_moe.output_linear.weight"),
    ]


def granite_rmsnorm(hidden, weight, *, eps: float):
    import torch

    input_dtype = hidden.dtype
    hidden_f32 = hidden.to(torch.float32)
    variance = hidden_f32.pow(2).mean(-1, keepdim=True)
    normalized = hidden_f32 * torch.rsqrt(variance + eps)
    # Match HF GraniteMoeRMSNorm.forward:
    #   return self.weight * hidden_states.to(input_dtype)
    return weight * normalized.to(input_dtype)


def rotate_half(x):
    import torch

    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)


def granite_rope_cos_sin(*, position_ids, head_dim: int, rope_theta: float, dtype):
    import torch

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, *, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep: int):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def build_granite_causal_mask(*, attention_mask, input_tensor, cache_position):
    import torch

    dtype = input_tensor.dtype
    device = input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = attention_mask.shape[-1] if attention_mask is not None else sequence_length + 1

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
    return causal_mask


def route_tokens(router_input, router_weight, *, top_k: int, num_experts: int):
    import torch
    import torch.nn.functional as F

    router_logits = F.linear(router_input, router_weight).float()
    top_k_logits, top_k_indices = router_logits.topk(top_k, dim=1)
    top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(router_input)

    zeros = torch.zeros(
        [top_k_gates.size(0), num_experts],
        dtype=top_k_gates.dtype,
        device=top_k_gates.device,
    )
    gates = zeros.scatter(1, top_k_indices, 1)
    expert_size = gates.long().sum(0).tolist()

    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(top_k, rounding_mode="trunc")

    batch_gates = top_k_gates.flatten()[index_sorted_experts]
    return {
        "router_logits": router_logits,
        "top_k_logits": top_k_logits,
        "top_k_indices": top_k_indices,
        "top_k_gates": top_k_gates,
        "expert_size": [int(x) for x in expert_size],
        "index_sorted_experts": index_sorted_experts,
        "batch_index": batch_index,
        "batch_gates": batch_gates,
    }


def run_experts_one_at_a_time(
    *,
    layer_idx: int,
    provider: SafetensorsWeights,
    router_input,
    route: dict[str, Any],
):
    import torch
    import torch.nn.functional as F

    input_tensor = layer_tensor(layer_idx, "block_sparse_moe.input_linear.weight")
    output_tensor = layer_tensor(layer_idx, "block_sparse_moe.output_linear.weight")

    expert_size = route["expert_size"]
    batch_index = route["batch_index"]
    batch_gates = route["batch_gates"]
    expert_inputs = router_input[batch_index]

    chunks = []
    offset = 0
    selected_expert_ids = []
    cumulative_expert_weight_bytes = 0
    peak_expert_pair_bytes = 0

    for expert_id, size in enumerate(expert_size):
        size = int(size)
        if size == 0:
            continue

        expert_chunk = expert_inputs[offset : offset + size]
        with provider.expert_slice(input_tensor, expert_id=expert_id) as w_in, provider.expert_slice(
            output_tensor,
            expert_id=expert_id,
        ) as w_out:
            expert_pair_bytes = w_in.numel() * BF16_BYTES + w_out.numel() * BF16_BYTES
            cumulative_expert_weight_bytes += expert_pair_bytes
            peak_expert_pair_bytes = max(peak_expert_pair_bytes, expert_pair_bytes)
            selected_expert_ids.append(expert_id)

            hidden = F.linear(expert_chunk, w_in)
            first_half, second_half = hidden.chunk(2, dim=-1)
            hidden = F.silu(first_half) * second_half
            expert_output = F.linear(hidden, w_out)
            expert_output = expert_output * batch_gates[offset : offset + size, None]
            chunks.append(expert_output)
        offset += size

    if offset != expert_inputs.shape[0]:
        raise SystemExit(f"Layer {layer_idx}: expert chunk offset {offset} != expert input rows {expert_inputs.shape[0]}")

    if not chunks:
        expert_outputs = torch.empty((0, router_input.shape[-1]), dtype=router_input.dtype, device=router_input.device)
    else:
        expert_outputs = torch.cat(chunks, dim=0)

    zeros = torch.zeros((router_input.shape[0], router_input.shape[-1]), dtype=expert_outputs.dtype, device=expert_outputs.device)
    moe_output_flat = zeros.index_add(0, batch_index, expert_outputs)

    return {
        "moe_output_flat": moe_output_flat,
        "selected_expert_ids": selected_expert_ids,
        "cumulative_expert_weight_bytes": cumulative_expert_weight_bytes,
        "peak_expert_pair_bytes": peak_expert_pair_bytes,
    }


def run_layer(
    *,
    layer_idx: int,
    hidden,
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    inputs: dict[str, Any],
    config: dict[str, Any],
    scalars: dict[str, float],
    budget: WeightBudget | None = None,
):
    layer_output, _, stats = run_layer_with_optional_kv(
        layer_idx=layer_idx,
        hidden=hidden,
        kv_pair=None,
        model_file=model_file,
        header=header,
        payload_start=payload_start,
        inputs=inputs,
        config=config,
        scalars=scalars,
        return_kv=False,
        budget=budget,
    )
    return layer_output, stats


def run_layer_with_optional_kv(
    *,
    layer_idx: int,
    hidden,
    kv_pair: tuple[Any, Any] | None,
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    inputs: dict[str, Any],
    config: dict[str, Any],
    scalars: dict[str, float],
    return_kv: bool = True,
    budget: WeightBudget | None = None,
):
    import torch
    import torch.nn.functional as F

    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start, budget=budget)

    hidden_size = int(config.get("hidden_size", 1024))
    num_attention_heads = int(config.get("num_attention_heads", 16))
    num_key_value_heads = int(config.get("num_key_value_heads", 8))
    num_key_value_groups = num_attention_heads // num_key_value_heads
    head_dim = hidden_size // num_attention_heads
    top_k = int(config.get("num_experts_per_tok", 8))
    num_experts = int(config.get("num_local_experts", 32))

    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long, device=hidden.device)
    position_ids = torch.tensor(inputs["position_ids"], dtype=torch.long, device=hidden.device)
    cache_position = torch.tensor(inputs["cache_position"], dtype=torch.long, device=hidden.device)

    input_norm_name = layer_tensor(layer_idx, "input_layernorm.weight")
    post_norm_name = layer_tensor(layer_idx, "post_attention_layernorm.weight")
    q_name = layer_tensor(layer_idx, "self_attn.q_proj.weight")
    k_name = layer_tensor(layer_idx, "self_attn.k_proj.weight")
    v_name = layer_tensor(layer_idx, "self_attn.v_proj.weight")
    o_name = layer_tensor(layer_idx, "self_attn.o_proj.weight")
    router_name = layer_tensor(layer_idx, "block_sparse_moe.router.layer.weight")

    loaded_nonexpert_bytes = 0

    residual = hidden
    with provider.tensor(input_norm_name) as input_norm_weight:
        loaded_nonexpert_bytes += input_norm_weight.numel() * BF16_BYTES
        hidden_norm = granite_rmsnorm(hidden, input_norm_weight, eps=scalars["rms_norm_eps"])

    with provider.tensor(o_name) as o_weight:
        loaded_nonexpert_bytes += o_weight.numel() * BF16_BYTES
        with provider.tensor(q_name) as q_weight, provider.tensor(k_name) as k_weight, provider.tensor(v_name) as v_weight:
            loaded_nonexpert_bytes += (q_weight.numel() + k_weight.numel() + v_weight.numel()) * BF16_BYTES
            q_proj = F.linear(hidden_norm, q_weight)
            k_proj = F.linear(hidden_norm, k_weight)
            v_proj = F.linear(hidden_norm, v_weight)

        batch_size, q_len, _ = q_proj.shape
        query_states = q_proj.view(batch_size, q_len, num_attention_heads, head_dim).transpose(1, 2)
        key_states = k_proj.view(batch_size, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = v_proj.view(batch_size, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        cos, sin = granite_rope_cos_sin(position_ids=position_ids, head_dim=head_dim, rope_theta=scalars["rope_theta"], dtype=query_states.dtype)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if kv_pair is None:
            key_cache = key_states.contiguous()
            value_cache = value_states.contiguous()
        else:
            past_key, past_value = kv_pair
            key_cache = torch.cat([past_key, key_states], dim=2).contiguous()
            value_cache = torch.cat([past_value, value_states], dim=2).contiguous()

        key_for_attn = repeat_kv(key_cache, num_key_value_groups)
        value_for_attn = repeat_kv(value_cache, num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_for_attn.transpose(2, 3)) * scalars["attention_multiplier"]
        causal_mask = build_granite_causal_mask(attention_mask=attention_mask, input_tensor=hidden_norm, cache_position=cache_position)
        attn_weights = attn_weights + causal_mask[:, :, :, : key_for_attn.shape[-2]]
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_for_attn)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, -1)
        attn_output = F.linear(attn_output, o_weight)

    hidden_after_attn = residual + attn_output * scalars["residual_multiplier"]

    residual = hidden_after_attn
    with provider.tensor(post_norm_name) as post_norm_weight, provider.tensor(router_name) as router_weight:
        loaded_nonexpert_bytes += (post_norm_weight.numel() + router_weight.numel()) * BF16_BYTES
        moe_input = granite_rmsnorm(hidden_after_attn, post_norm_weight, eps=scalars["rms_norm_eps"])
        router_input = moe_input.reshape(-1, hidden_size)
        route = route_tokens(router_input, router_weight, top_k=top_k, num_experts=num_experts)

    expert_result = run_experts_one_at_a_time(
        layer_idx=layer_idx,
        provider=provider,
        router_input=router_input,
        route=route,
    )
    moe_output = expert_result["moe_output_flat"].view(batch_size, q_len, hidden_size)
    layer_output = residual + moe_output * scalars["residual_multiplier"]

    stats = {
        "layer": layer_idx,
        "selected_expert_ids": expert_result["selected_expert_ids"],
        "selected_expert_count": len(expert_result["selected_expert_ids"]),
        "expert_size": route["expert_size"],
        "loaded_nonexpert_bytes": loaded_nonexpert_bytes,
        "cumulative_expert_weight_bytes": expert_result["cumulative_expert_weight_bytes"],
        "peak_expert_pair_bytes": expert_result["peak_expert_pair_bytes"],
    }
    if budget is not None:
        stats.update(
            {
                "weight_budget_current_bytes": budget.current_resident_bytes,
                "weight_budget_peak_bytes": budget.peak_resident_bytes,
                "weight_budget_total_visited_bytes": budget.total_visited_bytes,
                "weight_budget_violation_count": len(budget.violations),
            }
        )
    if return_kv:
        stats.update(
            {
                "kv_key_shape": list(key_cache.shape),
                "kv_value_shape": list(value_cache.shape),
            }
        )
        return layer_output, (key_cache, value_cache), stats
    return layer_output, None, stats
